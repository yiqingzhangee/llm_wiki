# **VLM4Bio: A Benchmark Dataset to Evaluate Pretrained Vision-Language Models for Trait Discovery from Biological Images** 

**M. Maruf**[1] _[∗]_ **Arka Daw**[9] _[∗]_ **Kazi Sajeed Mehrab**[1] **Harish Babu Manogaran**[1] **Abhilash Neog**[1] **Medha Sawhney**[1] **Mridul Khurana**[1] **James P. Balhoff**[3] **Yasin Bakı¸s**[4] **Bahadir Altintas**[4] **Matthew J Thompson**[5] **Elizabeth G Campolongo**[5] **Josef C. Uyeda**[1] **Hilmar Lapp**[6] **Henry L. Bart Jr.**[4] **Paula M. Mabee**[7] **Yu Su**[5] **Wei-Lun Chao**[5] **Charles Stewart**[8] **Tanya Berger-Wolf**[5] **Wasila Dahdul**[2] **Anuj Karpatne**[1] _[∗]_ 1Virginia Tech 2Univ. of California, Irvine 3UNC at Chapel Hill 4Tulane Univ. 5Ohio State Univ. 6Duke Univ. 7Battelle 8Rensselaer Polytechnic Institute 9Oak Ridge Lab. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0001-03.png)


**----- Start of picture text -----**<br>
Biodiversity of Organisms Scientific Tasks Multi-modal Reasoning<br>Bird (10k) Butterfly (10k) Trait Trait<br>Identification Grounding Tests for<br>Prompting<br>Reasoning<br>Trait Trait<br>Fish (10k) Referring Counting Contextual Prompting False Confidence    Test (FCT)<br>ClassificationSpecies Chain-of-Thought (CoT)Dense Captioning None of the Above Test (NOTA)<br>**----- End of picture text -----**<br>


Figure 1: Overview of our goals and contributions. We analyze the capabilities of 12 state-of-the-art (SOTA) vision-language models (VLMs) in answering scientific questions using images from three groups of organisms: fishes, birds, and butterflies, over five groups of biologically relevant tasks. We also explore the effectiveness of these models for reasoning using various prompting techniques and tests for reasoning hallucination. 

## **Abstract** 

Images are increasingly becoming the currency for documenting biodiversity on the planet, providing novel opportunities for accelerating scientific discoveries in the field of organismal biology, especially with the advent of large vision-language models (VLMs). We ask if pre-trained VLMs can aid scientists in answering a range of biologically relevant questions without any additional fine-tuning. In this paper, we evaluate the effectiveness of 12 state-of-the-art (SOTA) VLMs in the field of organismal biology using a novel dataset, **VLM4Bio** , consisting of 469 _K_ questionanswer pairs involving 30 _K_ images from three groups of organisms: fishes, birds, and butterflies, covering five biologically relevant tasks. We also explore the effects of applying prompting techniques and tests for reasoning hallucination on the performance of VLMs, shedding new light on the capabilities of current SOTA VLMs in answering biologically relevant questions using images.[2] 

> _∗_ Corresponding authors: _{marufm, karpatne}@vt.edu_ 

> 2The code and datasets for running all the analyses reported in this paper can be found at `https://github. com/sammarfy/VLM4Bio` . 

Preprint. Under review. 

## **1 Introduction** 

There is a growing deluge of images that are being collected, stored, and shared in organismal biology—the branch of biology interested in the study of structure, ecology, and evolution of organisms. In particular, images are increasingly becoming the currency for documenting the vast array of biodiverse organisms on our planet, with repositories containing millions of images of biological specimens collected by scientists in field museums or captured by drones, camera traps, or tourists posting photos on social media. This growing wealth of biological images provides a unique opportunity to understand the scientific mechanisms of how organisms evolve and adapt to their environment directly from images. The traditional approach for advancing knowledge in organismal biology is by discovering the observable characteristics of organisms or _traits_ (e.g., beak color, stripe pattern, and fin curvature) that serve a variety of biological tasks such as defining groups of organisms, understanding their genetic and developmental underpinnings, and analyzing their interactions with environmental selection pressures [1]. However, the measurement of traits is not straightforward and often relies on expert visual attention involving labor-intensive operations and subjective definitions [2], hindering rapid scientific advancement [3]. 

With the recent rise of large foundation models such as vision-language models (VLMs) (e.g., GPT-4, GPT-4V(ision) [4, 5], Gemini [6], LLaMA, and LLaVA [7, 8]) that can simultaneously solve a diverse range of tasks involving text and images, it is pertinent to ask if pre-trained VLMs contain the necessary _scientific knowledge_ to aid biologists in answering a variety of questions pertinent to the discovery of biological traits from images. Note that unlike mainstream tasks in computer vision, understanding scientific images requires knowledge of domain-specific terminologies and reasoning capabilities that are not fully represented in conventional image datasets used for training VLMs. In particular, an important end-goal in scientific applications such as organismal biology is to explain the process of visual reasoning used to arrive at a prediction, often involving the knowledge of biological traits. Hence, to assess the usefulness of VLMs in accelerating discoveries in organismal biology, it is important to test their ability to identify and reason about biological traits automatically from images. 

In this work, we assess the zero-shot capabilities of 12 state-of-the-art (SOTA) VLMs, including the proprietary GPT-4V(ision) and the recent GPT-4o along with other open-source VLMs, on five scientifically relevant tasks in organismal biology, namely species classification, trait identification, trait grounding, trait referring, and trait counting. These tasks are designed to test different facets of VLM performance in organismal biology, ranging from measuring predictive accuracy to assessing their ability to reason about their predictions using visual cues of known biological traits. For example, the task of species classification tests the ability of VLMs to discriminate between species, while in trait grounding and referring, we specifically test if VLMs are able to localize morphological traits (e.g., the presence of fins or patterns and colors of birds) within the image. To perform this evaluation, we present **VLM4Bio** , a benchmark dataset of _≈_ 469 _K_ question-answer pairs based on 30 _k_ images of three taxonomic groups of organisms: fishes, birds, and butterflies. 

**Contributions:** (1) We present a novel dataset of scientific question-answer pairs to evaluate the effectiveness of VLMs in answering scientific questions across a range of biologically relevant tasks in the field of organismal biology. (2) We present novel benchmarking analyses of the zero-shot effectiveness of pre-trained SOTA VLMs on our dataset, exposing their gaps in advancing scientific knowledge of organismal biology. (3) We present novel comparisons studying the effects of prompting and tests for reasoning hallucination on VLM performance, shedding new light on the reasoning capabilities of SOTA VLMs in organismal biology. 

## **2 Related Works** 

With the rise of SOTA VLMs such as GPT-4V(vision) [5], GPT-4o [9], and Gemini [6], there has been a simultaneous growth in the number of benchmarking analyses published in the last few years to evaluate different facets of VLM performance on a range of mainstream tasks in computer vision. A majority of previous analyses [10, 11] involve evaluations on single tasks like Visual Question Answering (VQA), OK-VQA [12], MSCOCO [13], and GQA [14]. Other datasets such as POPE [15], HaELM [7], LAMM [16], MMBench [17], MM-Vet [18], LVLM-eHub [19], SEED [20], and GAIA [21] have also been developed to evaluate the capabilities of VLMs on complex tasks such as reasoning and ability to handle multimodal data. There are also some recent domain-specific benchmark datasets, such as MathVista [22], which includes a variety of challenging VQA problems in the mathematical domain, MedQA(USMLE) [23] which is a collection of VQA problems from 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Species Classification Trait Identification Trait Referring<br>Question: butterfly shown in the image?   What is the scientific name of the  Question:  the image? Is there eye visible in the fish shown in  Question:  to the bounding box region [2545, 335, 3510, 423] in the image?What is the trait of the fish that correspond<br>Options:<br>Correct Answer: Options:  A) dorsal fin<br>Heliconius timareta A) Yes B) caudal fin<br>B) No C) adipose finD) pelvic fin<br>Correct Answer:  A) Yes Correct Answer:  A) dorsal fin<br>Question type:  Open Questions Question type:  Multiple Choice Questions Question type:  Multiple Choice Questions<br>Species Classification Trait Grounding Trait Counting<br>Question: What is the scientific name of the bird shown in the image? Question: What is the bounding box coordinates of the dorsal fin in the fish shown in the image?  Question:  the fish shown in the image? The fins that are normally present in a fish are dorsal fin, caudal How many unique fins are visible in<br>Options:  A) Geothlypis philadelphia Options:  A) [453, 620, 557, 724] fin, pectoral fin, pelvic fin, anal fin and adipose fin.<br>B) Vireo atricapilla B) [2545, 335, 3510, 423]<br>C) Larus glaucescensD) Coccothraustes vespertinus C) [2012, 1001, 2404, 1350]D) [3444, 350, 4730, 1114] Correct Answer: 5<br>Correct Answer:  C) Larus glaucescens Correct Answer:  B) [2545, 335, 3510, 423]<br>Question type:   Multiple Choice Questions Question type:  Multiple Choice Questions Question type:  Open Questions<br>**----- End of picture text -----**<br>


Figure 2: Illustrative examples of **VLM4Bio** tasks with different question-types. 

medical exams, and the recent MMMU [11] dataset, which covers expert-level problems from diverse fields such as business, arts, science, health, medicine, and engineering. 

VLM4Bio dataset is different from existing benchmarks involving domain-specific datasets because of the following reasons. (1) _Focus on organismal biology_ : While previous works have focused on benchmarking the performance of VLMs on other scientific domains (e.g., Arts and Design, Business, Health, and Medicine in MMMU [11] or Mathematics in MathVista [22]), there exists no previous VQA benchmark dataset in the domain of organismal biology to the best of our knowledge. Our work thus fills a critical gap in evaluating the performance of VLMs in a field of biology that has several societal implications such as monitoring biodiversity and understanding the impact of climate change on species traits and populations. (2) _Breadth of Evaluation Tasks_ : While previous works are tailored to one or a few evaluation tasks, we consider a wide range of tasks motivated by the needs of domain scientists in the field of organismal biology. They include predictive tasks such as species classification and trait identification as well as tasks that require visual reasoning including trait grounding and referring. We also provide novel comparisons about the performance of VLMs on both open-ended and multiple-choice question (MCQ) formats and comparisons over predictive as well as visual reasoning tasks, in contrast to prior works. 

## **3 VLM4Bio Tasks** 

Figure 2 shows illustrative examples of the five VLM4Bio tasks relevant to biologists that we consider in our study, described in detail in the following. 

**Species Classification:** A common (and often the first) task that a biologist considers when examining an organism specimen is to identify its scientific name (or species class). Hence, we consider asking a VLM to provide the scientific name of the organism shown in a given image. There are two types of questions that we consider for this task. First, we consider _open-ended questions_ , where we do not provide any answer choices (or options) to the VLM in the input prompt. The second type is _multiple-choice (MC) questions_ , where we provide four choices of candidate species names for the VLM to choose from (out of which only one is correct while the remaining three are randomly selected from the set of all species classes). 

**Trait Identification:** An important goal in organismal biology is to answer questions regarding the observable characteristics of organisms, also known as traits. We thus consider asking VLMs to identify a particular trait of an organism given its image for two taxonomic groups: fishes and birds. For fishes, we considered 10 binary (presence/absence) traits and generated MC questions for the presence of each trait in an image (with two options: yes or no), whereas for birds, we considered 28 traits covering their color, pattern, and measurements (size and shape of regions) in a multiple-choice format. We provide a detailed list of all fish and bird traits in the Supplementary. 

3 

|Statistics|**Fish-10K**|**Bird-10K**|**Butterfy-10K**|**Fish-500**|**Bird-500**|
|---|---|---|---|---|---|
|# Images|10,347|11,092|10,013|500|492|
|# Species|495|188|60|60|47|
|# Genera|178|114|27|18|33|
|# Traits|10|28|-|8|5|



Table 1: Key statistics of the **VLM4Bio** dataset. 

**Trait Grounding and Referring:** To further understand the ability of VLMs to visually explain the reasoning behind their prediction of a trait, it is important to evaluate if a VLM correctly identifies the region in the image containing the trait. For this purpose, we consider two other tasks: trait grounding & trait referring, for the taxonomic groups of fishes and birds. In the first task of trait grounding, we ask the VLM to locate a given trait of an organism on its image (i.e., _text to location_ ). We consider MC question-format for this task where we provide four options of bounding boxes in the image as candidate answer choices, where one of the bounding boxes correctly contains the trait while the remaining three are randomly sampled from the set of bounding boxes containing other traits of the organism. In the second task of trait referring, we consider the opposite scenario where we provide a bounding box as input to the VLM and ask it to identify the name of the trait present in the bounding box (i.e., _location to text_ ). We again provide four answer choices in MC question-format, where only one of the options is correct while the remaining three are randomly sampled from the names of other traits of the organism. 

**Trait Counting:** We simply ask how many traits are present in an image of a fish specimen. This is biologically relevant, for example, to understand the number of fins present in a fish organism. Similar to the species classification task, we have open and MC question-types for this task. 

## **4 VLM4Bio Dataset** 

**Data Collection and Preprocessing** : We collected images of three taxonomic groups of organisms: fish, birds, and butterflies, each containing around 10 _K_ images. Images for fish ( **Fish-10K** ) were curated from the larger image collection, FishAIR [24], which contains images from the Great Lakes Invasive Network Project (GLIN) [25] and Integrated Digitized Biocollections (iDigBio) [26]. These images originate from various museum collections such as INHS [27], FMNH [28], OSUM [29], JFBM [30], UMMZ [31] and UWZM [32]. We created the Fish-10K dataset by randomly sampling 10 _K_ images and preprocessing the images to crop and remove the background. For consistency, we leverage GroundingDINO [33] to crop the fish body from the background and Segment Anything Model (SAM) [34] to remove the background. We curated the images for butterflies ( **Butterfly-10K** ) from the Jiggins Heliconius Collection dataset [35], which has images collected from various sources 3. We carefully sampled 10 _K_ images for Butterfly-10K from the entire collection to ensure the images capture unique specimens and represent a diverse set of species by adopting the following two steps. First, we filter out images with more than one image from the same view (i.e., dorsal or ventral). Second, we ensure each species has a minimum of 20 images and no more than 2 _,_ 000 images. The images for birds ( **Bird-10K** ) are obtained from the CUB-200-2011 [61] dataset by taking 190 species for which the common name to scientific name mapping is available. This results in a fairly balanced dataset with around 11 _K_ images in total. Additional details on dataset preprocessing are provided in the Supplementary. 

**Annotation:** The scientific names for the images of Fish-10K and Butterfly-10K were obtained directly from their respective sources. For Bird-10K, we obtained the scientific names from the iNatLoc500 [62] dataset. We curated around 31 _K_ question-answer pairs in both open and multiplechoice (MC) question-formats for evaluating species classification tasks. The species-level trait presence/absence matrix for Fish-10K was manually curated with the help of biological experts coauthored in this paper. We leveraged the Phenoscape knowledge [63] base with manual annotations to procure the presence-absence trait matrix. For Bird-10K, we obtained the trait matrix from the attribute annotations provided along with CUB-200-2011. We constructed approximately 380 _K_ question-answer pairs for trait identification tasks. For grounding and referring VQA tasks, the 

> 3Sources: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60] 

4 

|**Dataset**<br>**Question**<br>**type**|**Models**<br>_gpt-4v_<br>_llava_<br>_v1.5-7b_<br>_llava_<br>_v1.5-13b_<br>_cogvlm_<br>_chat_<br>_BLIP_<br>_fan-xl_<br>_BLIP_<br>_fan-xxl_<br>_minigpt4_<br>_vicuna-7B_<br>_minigpt4_<br>_vicuna-13B_<br>_instruct_<br>_fant5xl_<br>_instruct_<br>_fant5xxl_<br>_instruct_<br>_vicuna7B_<br>_instruct_<br>_vicuna13B_<br>_Random_<br>_Choice_|
|---|---|
||**Species Classifcation**|
|**Fish-10K**<br>Open<br>MC|1.01<br>2.32<br>0.40<br>0.11<br>0.01<br>1.59<br>0.50<br>0.38<br>0.00<br>1.46<br>0.00<br>0.00<br>0.20<br>35.91<br>40.20<br>32.27<br>31.72 29.76<br>33.36<br>29.02<br>27.45<br>30.86<br>31.70<br>27.27<br>26.57<br>25.00|
|**Bird-10K**<br>Open<br>MC|17.40<br>1.45<br>2.06<br>0.86<br>0.00<br>0.57<br>2.80<br>2.56<br>0.00<br>0.50<br>0.07<br>0.00<br>0.53<br>82.58<br>50.32<br>55.36<br>44.73 33.68<br>34.75<br>23.95<br>27.62<br>36.36<br>35.83<br>44.00<br>46.55<br>25.00|
|**Butterfy-10K**<br>Open<br>MC|0.04<br>0.05<br>0.00<br>0.01<br>0.00<br>0.00<br>0.07<br>0.01<br>0.00<br>0.00<br>9.94<br>0.00<br>1.54<br>28.91<br>50.24<br>44.58<br>36.45<br>25.14<br>28.88<br>33.06<br>28.90<br>25.28<br>36.67<br>41.70<br>34.48<br>25.00|
||**Trait Identifcation**|
|**Fish-10K**<br>MC|82.18<br>56.84<br>45.15<br>46.92 68.36<br>39.33<br>55.08<br>51.87<br>64.34<br>39.26<br>81.95<br>20.69<br>50.0|
|**Bird-10K**<br>MC|62.22<br>34.68<br>46.14<br>63.93 50.11<br>41.38<br>39.11<br>40.44<br>47.89<br>45.52<br>77.91<br>89.98<br>31.12|
||**Trait Grounding**|
|**Fish-500**<br>MC|29.41<br>24.87<br>17.98<br>23.42 23.32<br>25.14<br>22.18<br>25.58<br>7.20<br>27.09<br>33.51<br>26.90<br>25.00|
|**Bird-500**<br>MC|8.1<br>26.92<br>35.36<br>23.2 11.83<br>10.52<br>15.39<br>24.22<br>3.48<br>0.81<br>30.24<br>13.91<br>25.00|
||**Trait Referring**|
|**Fish-500**<br>MC|28.15<br>27.07<br>29.14<br>28.19<br>24.93<br>25.68<br>39.24<br>31.21<br>31.75<br>25.78<br>28.04<br>32.73<br>25.00|
|**Bird-500**<br>MC|42.28<br>30.5<br>29.64<br>18.45 35.16<br>40.59<br>26.04<br>35.88<br>27.52<br>41.69<br>23.03<br>22.69<br>25.00|
||**Trait Counting**|
|**Fish-500**<br>Open<br>MC|16.4<br>47.4<br>52.0<br>14.8<br>37.6<br>63.4<br>13.6<br>31.53<br>50.2<br>61.4<br>61.4<br>0.0<br>25.00<br>44.80<br>13.20<br>54.80<br>21.00<br>64.8<br>78.2<br>22.00<br>25.00<br>74.0<br>69.4<br>15.80<br>11.80<br>25.00|
|_Overall_|34.24<br>29.0<br>31.78<br>25.27 28.91<br>30.24<br>23.0<br>25.19<br>28.49<br>29.79<br>33.92<br>23.31<br>22.03|



Table 2: Zero-shot accuracy comparison of VLM baselines (in % ranging from 0 to 100) for the five scientific tasks. Results are color-coded as Best , Second best , Worst , Second worst . 

ground truths were manually annotated with the help of expert biologists on our team. We manually annotated bounding boxes corresponding to the traits of 500 fish specimens and 500 bird specimens, which are subsets of the larger Fish-10K and Bird-10K datasets, respectively. In particular, we considered 8 fish traits and 5 bird traits for annotating their bounding boxes, resulting in a total of 26 _K_ question-answer pairs. We also used the Fish-500 dataset for the task of trait counting, resulting in a total of 1 _K_ question-answer pairs. Across all tasks, our dataset comprises approximately 469 _K_ question-answer pairs for 30 _K_ biological images (see Table 1). Additional details on data distribution and key statistics are provided in the Supplementary. 

**VLM Baselines:** We consider the following VLM baselines: GPT-4V(ision) [64][4] , LLaVA-v1.5 (7B/13B) [65], COG-VLM [66], MiniGPT-4 (Vicuna 7B/13B) [67], BLIP-FLAN-T5-XL/XXL [68], and INSTRUCT-BLIP (Vicuna 7B/13B) [69]. We used the latest checkpoints for each model available to date. We used the same question prompt for all models to ensure consistent comparison of results for a variety of open and multiple-choice (MC) questions across the five scientific tasks of our dataset. All the experiments were conducted using NVIDIA A100 GPUs. See supplementary for more details of the VLM baselines. 

**Evaluation Metrics:** We used micro-averaged accuracy as our evaluation metric for all experiments. We designed a systematic rule-based evaluation pipeline to evaluate VLM responses against the ground truths. For each question category, we provide the accuracy percentage of random choice as a basic baseline, where each possible answer is considered equally likely (yielding an accuracy of 25% for MC questions with four choices). 

## **5 Results** 

Table 2 compares the accuracies of VLMs in percentages (ranging from 0 to 100) across the five tasks and over multiple organism datasets. We make the following observations from this result. 

> 4We use _gpt-4-1106-vision-preview_ model as GPT-4v in our experiments. 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0006-00.png)


Figure 3: Examples of correct and incorrect predictions of GPT-4V for trait identification, trait grounding, and trait-referring tasks related to the “eye”. For visualization assistance, a red-colored bounding box is added around the “eye” in the image. 

**All VLMs show poor accuracy on open questions but perform better on MC questions.** The zero-shot species classification accuracy of all VLMs on open-ended questions is notably weaker than MC questions. Even the best-performing models, LLaVA-13B, GPT-4V, and Instruct-Vicuna-7B, only achieve accuracies of 2.32%, 17.46%, and 3.62%, respectively, across the three organism datasets. This indicates a significant gap in the ability of existing VLMs to capture the scientific knowledge necessary to differentiate between species (often requiring subtle or nuanced features) without being provided with candidate answer choices. Open-ended species classification is particularly hard for pre-trained VLMs that are not typically trained to provide scientific names of organisms (e.g., _Lepomis cyanellus_ ) rather than providing their common names (e.g., _green sunfish_ ). However, the inclusion of candidate answers (or options) in the question prompt serves as a helpful clue to VLMs for narrowing down the solution space and finding the correct answer potentially using elimination strategies. While VLMs are able to utilize these additional hints and work their way through to the correct answer in MC questions, note that open questions are practically more relevant to scientists operating in real-world settings. 

**Bird dataset shows better accuracy than Fish or Butterfly datasets.** Most VLMs show significantly better performance on the Bird-10 _K_ dataset in comparison to the Fish-10 _K_ and Butterfly-10 _K_ datasets. For example, the highest accuracy across all VLMs on the Bird-10 _K_ dataset is 82.58%, while it is 40.20% and 50.24% on the Fish-10 _K_ and Butterfly-10 _K_ datasets, respectively. A potential reason is that while the bird dataset is a subset of the CUB dataset [70] that is commonly used in machine learning literature and has images with natural in-the-wild backgrounds, the butterfly and fish datasets contain images of specimens preserved in museum collections with artificial backgrounds and with imaging artifacts that are not typical for large-scale computer vision datasets. We hypothesize that many of the pre-trained VLM baselines may have seen images similar to those in the Bird dataset during training, leading to their better performance. 

**Can VLMs effectively identify biological traits?** The performance of most VLMs in trait identification appears significantly better than their performance in species classification, with GPT-4V reaching 82.18% accuracy on the Fish-10 _K_ dataset and Instruct-Vicuna-13B achieving 89.98% on Bird-10 _K_ . However, some traits such as “eye”, “head”, and “mouth” are almost always present in every organism image, so simply answering “yes, the trait is present” can lead to high accuracy in trait identification. In contrast to the fish dataset, the bird dataset poses more intricate questions regarding a variety of multi-class traits that require a nuanced understanding of colors, patterns, and physical trait dimensions, such as the color of the bill, wing patterns, and tail shapes. 

**VLMs struggle in localizing traits in images.** While most VLMs perform well on the task of Trait Identification, it is crucial to determine if they are focusing on the correct image regions to answer trait-related questions. We thus analyze the performance of VLMs on the tasks of trait grounding (i.e., _text to location_ ) and trait referring (i.e., _location to text_ ). We can see that there is a significant drop in the accuracy of trait grounding and referring tasks compared to the trait identification task. This shows that while VLMs can potentially leverage knowledge of trait choices to identify traits, they struggle in localizing the traits in the image and thus visually ground their reasoning. Figure 3 shows an illustrative example of GPT-4V prediction where it predicts the presence of the trait “eye” correctly but fails to localize it in grounding and referring tasks. 

**Counting biological traits is difficult for VLMs.** Recent studies [71, 72, 73] have explored the gap in the ability of VLMs to count objects, which is aligned with our results in Table 2. All VLMs, except for BLIP-flan-T5-XXL, show lower performance in counting traits, despite performing well 

6 

|**Dataset Diffculty**|**Models**<br> _gpt-4v gpt-4o_<br>_llava_<br>_v1.5-7b_<br>_llava_<br>_v1.5-13b_<br>_cogvlm_<br>_chat_<br>_BLIP_<br>_fan-xl_<br>_BLIP_<br>_fan-xxl_<br>_minigpt4_<br>_vicuna-7B_<br>_minigpt4_<br>_vicuna-13B_<br>_instruct_<br>_fant5xl_<br>_instruct_<br>_fant5xxl_<br>_instruct_<br>_vicuna7B_<br>_instruct_<br>_vicuna13B CLIP BioCLIP_|
|---|---|
|**Fish**<br>Easy<br>Medium|44.50<br>37.50<br>47.50<br>46.00<br>24.00<br>34.00<br>27.50<br>29.00<br>19.50<br>32.00<br>28.00<br>33.50<br>33.50<br>36.50<br>55.50<br>3.50<br>5.50<br>30.00<br>28.50<br>27.00<br>26.00<br>23.00<br>26.50<br>25.00<br>28.50<br>24.50<br>26.00<br>25.50<br>26.00<br>29.00|
|**Bird**<br>Easy<br>Medium|73.50<br>68.00<br>53.50<br>50.00<br>38.50<br>34.50<br>36.00<br>21.00<br>32.00<br>41.00<br>33.00<br>43.50<br>39.00<br>57.00<br>94.00<br>41.00<br>40.50<br>30.50<br>37.00<br>30.00<br>25.50<br>21.00<br>21.00<br>24.00<br>27.00<br>27.00<br>24.50<br>26.50<br>31.00<br>95.00|
|**Butterfy**<br>Easy<br>Medium<br>Hard|18.50<br>17.50<br>19.00<br>20.50<br>24.50<br>30.00<br>25.00<br>34.50<br>26.00<br>24.50<br>22.50<br>19.00<br>24.50<br>21.50<br>65.50<br>5.50<br>7.00<br>29.50<br>29.00<br>29.50<br>20.00<br>25.50<br>33.00<br>25.00<br>27.50<br>25.00<br>25.00<br>25.00<br>21.50<br>58.00<br>2.00<br>1.50<br>22.00<br>21.00<br>32.00<br>26.50<br>20.00<br>29.50<br>24.00<br>22.50<br>24.00<br>24.00<br>21.00<br>21.50<br>35.00|



Table 3: Zero-Shot accuracy comparison for _easy, medium, and hard_ datasets. Results are color-coded as Best , Second best , Worst , Second worst . 

on the trait identification task. The overall average accuracy for the VLMs is displayed in the last block, with GPT-4V(ision) exhibiting the best performance. 

We further analyze the errors of different VLMs to better understand their behavior. We find that GPT-4V shows a reduced rate of incorrect responses but a higher incidence of “Other” responses, which include apologetic expressions, admissions of inability to precisely visualize the organism, and disclaimers regarding lack of expert guidance (see Supplementary for more details). 

## **5.1 Analyzing the Role of Answer Choices in MC Questions on VLM Performance** 

Table 2 showed that VLMs perform drastically better on MC questions compared to Open questions for species classification. A potential hypothesis for this observation is that VLMs are able to avoid incorrect answer choices (or options) that are too different from the correct option and thus are easy to eliminate. To test this hypothesis, we create three variants of the MC questions for species classification—easy, medium, and hard—-where species choices in each variant have varying degrees of similarity determined by their taxonomic groupings. In particular, note that the scientific name of an organism contains taxonomic information at three levels: `<genus name> <species name> <subspecies name>`[5] . Since organisms that share taxonomic information have similar appearances, it is hard to differentiate species choices if they are from the same taxonomic group. On the other hand, it is easier to work with species choices from different taxonomic groups. Hence, for the easy set, we selected 50 species from different genera, ensuring that all species choices appear quite different from each other. For the medium set, we increased the complexity by constructing species choices from the same genus but from 10 different species. The hard set presented the highest difficulty level for the butterfly dataset, with the answer choices being from the same genus and species but from 10 subspecies. Each difficulty level consists of 200 images from each set of organisms. 

Table 3 shows the accuracies of the baseline VLMs for the easy, medium, and hard organism datasets. The pretrained VLMs generally perform best on the easy set and worst on the hard set for each organism. Moreover, there is a gradual improvement in the VLM performance from hard to easy questions. This suggests that the difficulty level of candidate answers (or options) in the question prompt significantly impacts VLMs’ performance. Additionally, this outcome indicates that even SOTA VLMs have limitations in handling fine-grained queries. Table 3 shows that GPT-4V and OpenAI’s recent release GPT-4o do not perform well when tested on the medium and hard datasets for Fish and Butterfly. Due to this, we further analyze the errors of different VLMs to better understand their behavior. We provide the report in the Supplementary. 

## **5.2 Comparing Pre-trained VLMs with a Biologically Fine-tuned Model** 

We compare BioCLIP [74], a state-of-the-art foundation model for species classification fine-tuned with biological images and taxonomic names (TreeOfLife-10M dataset), with the pretrained VLMs. We observe that BioCLIP significantly outperforms large pretrained VLMs on the Bird-10 _K_ and Butterfly datasets, suggesting that BioCLIP has been trained on images that are similar to the organisms present in these datasets. By comparing BioCLIP with CLIP, we can also see that fine- 

> 5We only have subspecies level information for the Butterfly-10 _K_ dataset. 

7 

|**Dataset**<br>**Prompting**|**Models**<br>_gpt-4v gpt-4o_<br>_llava_<br>_v1.5-7b_<br>_llava_<br>_v1.5-13b_<br>_cogvlm_<br>_chat_<br>_BLIP_<br>_fan-xl_<br>_BLIP_<br>_fan-xxl_|
|---|---|
|**Fish-Prompting**<br>No Prompting<br>Contextual<br>Dense Caption<br>CoT|34.40<br>79.00<br>41.60<br>35.40<br>31.00 28.60<br>22.60<br>30.00<br>77.20<br>40.20<br>35.60<br>25.60 27.20<br>26.60<br>18.80<br>78.60<br>26.00<br>27.60<br>32.00 28.40<br>29.80<br>42.60<br>86.00<br>41.40<br>34.80<br>26.80 29.20<br>24.60|
|**Bird-Prompting**<br>No Prompting<br>Contextual<br>Dense Caption<br>CoT|78.80<br>97.60<br>44.20<br>49.80<br>45.40 35.60<br>35.80<br>78.60<br>98.60<br>44.00<br>52.00<br>49.40 35.60<br>30.40<br>87.40<br>97.00<br>33.40<br>41.00<br>44.00 25.60<br>22.80<br>62.60<br>98.60<br>37.40<br>47.80<br>42.20 30.60<br>31.00|
|**Butterfy-Prompting**<br>No Prompting<br>Contextual<br>Dense Caption<br>CoT|13.20<br>56.40<br>27.20<br>26.80<br>25.60 24.40<br>21.20<br>9.20<br>56.20<br>26.00<br>24.60<br>27.20 23.60<br>24.60<br>49.60<br>63.20<br>25.20<br>23.80<br>27.00 23.20<br>23.20<br>63.60<br>74.60<br>21.40<br>23.20<br>34.60 37.20<br>23.60|



Table 4: Zero-shot accuracy comparison for different prompting techniques of seven VLMs (in % ranging from 0 to 100). Results are color-coded as Best and Worst . 

tuning foundation models with biological data provides large gains in classification performance. This suggests that the performance of SOTA VLMs can be further improved by fine-tuning on VLM4Bio Dataset. Further details comparing BioCLIP with SOTA VLMs are provided in the Supplementary. 

## **5.3 Analyzing Effects of Prompting on VLM Performance** 

We considered three prompting techniques: Contextual Prompting, Dense Caption Prompting, and zero-shot Chain of Thought Prompting. For **Contextual prompting** , we provided a single-line description (context) of the tasks (e.g., we add “ _Each biological species has a unique scientific name composed of two parts: the first for the genus and the second for the species within that genus._ ” before the species classification question to give some additional context on the task). **Dense Caption prompting** involves two stages: (1) first, we prompt the VLM to generate a dense caption of the specimen image such that the caption contains all the necessary trait information of the specimen. (2) We add the dense caption before the question and prompt “ _Use the above dense caption and the image to answer the following question._ ” to generate responses from the VLM. Similarly, the **Zero-Shot Chain-of-Thought (CoT)** happens in two stages: (1) First, we prompt the VLM to generate the reasoning for a given VQA and multiple choices (options). Zero-shot CoT appends “ _Let’s think step by step._ ” after the question and options to generate the reasoning. (2) We then add the reasoning after the VQA and prompt “ _Please consider the following reasoning to formulate your answer_ ” to generate the VLM response. We curated a prompting dataset of 500 multiple-choice (MC) VQAs for each set of organisms, which is a subset of the VLM4Bio dataset for species classification. 

Table 4 compares best-performing VLMs on the prompting dataset. The CoT rows of the table demonstrate that only GPT-4V and GPT-4o have reasoning capabilities that can significantly improve their response to biological questions, while smaller models like LLaVa and BLIP do not show much improvement. Furthermore, providing extra context and caption is more useful for GPT-4V and GPT-4o than the smaller models. This resonates with the findings from [75] that the reasoning abilities of VLMs only emerge after a certain model size. The success of Dense Caption prompting and CoT prompting depends on how well they generate the dense caption or the reasoning in the first stage. We report example prompts with VLM responses as case studies in the Supplementary. 

## **5.4 Analyzing Tests for Reasoning Hallucination** 

To further understand whether pretrained VLMs can respond with logically coherent and factually accurate reasoning, we evaluate VLMs on two sets of reasoning for hallucination tests - **False Confidence Test (FCT)** and **None of the Above (NOTA) Test** - inspired by [76]. For the FCT, we randomly select an option from the list of given choices and prompt it to the VLM as a “suggested correct answer” along with the question and options. To evaluate VLMs on FCT, we use Accuracy as well as the Agreement score, which is the percentage of times the VLM agrees with the suggested answer, irrespective of whether that is right or wrong. A high agreement score with a low overall accuracy indicates poor performance as it suggests that the model is simply following the suggestion either because of a lack of knowledge or low confidence in its own response. On the other hand, 

8 

|**Dataset**<br>**Metrics**|**Models**<br>_gpt-4v_<br>_gpt-4o_<br>_llava_<br>_v1.5-7b_<br>_llava_<br>_v1.5-13b_<br>_cogvlm_<br>_chat_<br>_BLIP_<br>_fan-xl_<br>_BLIP_<br>_fan-xxl_|
|---|---|
|**False Confdence Test (FCT)**||
|**Fish-Prompting**<br>Accuracy<br>34.20<br>73.60<br>25.00<br>28.60<br>24.60<br>0.00<br>7.00<br>Agreement Score<br>4.40<br>16.60<br>99.80<br>19.20<br>74.40<br>0.00<br>28.4||
|**Bird-Prompting**<br>Accuracy<br>73.40<br>99.00<br>25.40<br>35.80<br>19.80<br>0.00<br>20.20<br>Agreement Score<br>11.40<br>21.00<br>93.20<br>17.80<br>47.80<br>0.00<br>79.80||
|**Butterfy-Prompting**<br>Accuracy<br>5.20<br>53.40<br>27.20<br>26.60<br>6.20<br>0.00<br>5.00<br>Agreement Score<br>2.60<br>12.40<br>95.40<br>5.60<br>13.80<br>0.00<br>19.00||
|**None of the Above (NOTA) Test**||
|**Fish-Prompting**<br>Accuracy<br>81.40<br>44.80<br>3.40<br>3.80<br>0.00<br>4.00<br>0.00||
|**Bird-Prompting**<br>Accuracy<br>75.00<br>91.40<br>1.00<br>1.20<br>0.00<br>31.40<br>0.00||
|**Butterfy-Prompting**<br>Accuracy<br>50.40<br>4.60<br>1.00<br>4.60<br>0.00<br>51.00<br>0.00||



Table 5: Performance of seven VLMs on the NOTA and FCT reasoning tests. Results are color-coded as Best and Worst . 

in the NOTA Test, we replace the correct option with “None of the Above”, requiring the model to produce “None of the above” for all the questions. From Table 5, we can see that LLaVa-v1.5-7B shows poor accuracy on both tests and high agreement score on FCT. Out of all the VLMs, GPT-4V and GPT-4o demonstrate the highest accuracy, i.e., the lowest false confidence. More details on the prompts and examples of the responses have been provided in the Supplementary. 

## **6 Limitations** 

Our work has three main limitations. First, while no prior VQA benchmark dataset exists for organismal biology to the best of our knowledge, we focused on only three organisms—fish, bird, and butterfly—out of the many available due to resource constraints. Adding more organisms with manually annotated trait data will require additional resources and domain expertise, which could be pursued in future work. Second, since it is not feasible to manually inspect all images to ensure that they are free from label noise, we acknowledge that some noise may be present in the labels used for evaluating models on our current dataset, which we plan to address in future iterations. Third, due to resource constraints, certain proprietary VLMs that require purchasing APIs like Gemini-Pro [6], Gemini-Ultra [6], and Claude Opus [77] were also not included in the evaluation. We anticipate that their performance will be comparable to that of the proprietary GPT-4V [5] and GPT-4o [9] considered in our evaluation. 

## **7 Conclusion and Future Work** 

We presented VLM4Bio, a benchmark dataset to evaluate the zero-shot performance of pretrained VLMs on biologically relevant questions involving biodiversity images, exposing gaps in SOTA VLMs when applied to organismal biology. We observe that while VLMs are able to perform reasonably well on simpler tasks, e.g., using questions with multiple-choice formats and images with natural-looking backgrounds, they struggle in complex task settings that are practically more relevant to biologists. Through our study on prompting and reasoning tests on the VLM4Bio dataset, we observe that very large SOTA VLMs such as GPT-4V and GPT-4o have reasoning capabilities that can significantly improve the response to biological questions. We did not explore Retrieval Augmented Generation (RAG) [78] or knowledge-infused prompting [79] techniques since they require additional knowledge bases, which could be developed in future work. Future works can also focus on finetuning VLMs on the VLM4Bio dataset instead of comparing zero-shot performance. 

## **Acknowledgements** 

This research is supported by National Science Foundation (NSF) awards for the HDR Imageomics Institute (OAC-2118240). We are thankful for the support of computational resources provided by the 

9 

Advanced Research Computing (ARC) Center at Virginia Tech. This manuscript has been authored by UT-Battelle, LLC, under contract DE-AC05-00OR22725 with the US Department of Energy (DOE). The US government retains, and the publisher, by accepting the article for publication, acknowledges that the US government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this manuscript or allow others to do so for US government purposes. DOE will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan ( `https://www.energy.gov/doe-public-access-plan` ). 

## **References** 

- [1] David Houle and Daniela M Rossoni. Complexity, evolvability, and the process of adaptation. _Annual Review of Ecology, Evolution, and Systematics_ , 53, 2022. 

- [2] Tiago R Simões, Michael W Caldwell, Alessandro Palci, and Randall L Nydam. Giant taxoncharacter matrices: quality of character constructions remains critical regardless of size. _Cladistics_ , 33(2):198–219, 2017. 

- [3] Moritz D Lürig, Seth Donoughe, Erik I Svensson, Arthur Porto, and Masahito Tsuboi. Computer vision, machine learning, and the promise of phenomics in ecology and evolutionary biology. _Frontiers in Ecology and Evolution_ , 9:642774, 2021. 

- [4] OpenAI. Gpt-4v(ision) system card, 2023. _arXiv preprint arXiv:2303.08774, 2023_ , 2023. 

- [5] OpenAI. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774, 2023_ , 2023. 

- [6] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- [7] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- [8] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _arXiv preprint arXiv:2310.03744_ , 2023. 

- [9] OpenAI. Gpt-4o (“o” for “omni”). _https://openai.com/index/hello-gpt-4o/_ , 2024. 

- [10] Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. An empirical study of gpt-3 for few-shot knowledge-based vqa. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 36, pages 3081–3089, 2022. 

- [11] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. _arXiv preprint arXiv:2311.16502_ , 2023. 

- [12] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In _Proceedings of the IEEE/cvf conference on computer vision and pattern recognition_ , pages 3195–3204, 2019. 

- [13] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_ , pages 740–755. Springer, 2014. 

- [14] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 6700–6709, 2019. 

- [15] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. _arXiv preprint arXiv:2305.10355_ , 2023. 

10 

- [16] Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Lu Sheng, Lei Bai, Xiaoshui Huang, Zhiyong Wang, et al. Lamm: Language-assisted multi-modal instructiontuning dataset, framework, and benchmark. _arXiv preprint arXiv:2306.06687_ , 2023. 

- [17] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? _arXiv preprint arXiv:2307.06281_ , 2023. 

- [18] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. _arXiv preprint arXiv:2308.02490_ , 2023. 

- [19] Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu, Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping Luo. Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models. _arXiv preprint arXiv:2306.09265_ , 2023. 

- [20] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seedbench: Benchmarking multimodal llms with generative comprehension. _arXiv preprint arXiv:2307.16125_ , 2023. 

- [21] Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. _arXiv preprint arXiv:2311.12983_ , 2023. 

- [22] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. _arXiv preprint arXiv:2310.02255_ , 2023. 

- [23] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. _Applied Sciences_ , 11(14):6421, 2021. 

- [24] fishair.org. Fish-air. _fishair.org_ . 

- [25] Great Lakes Invasive Network Project (GLIN). https://greatlakesinvasives.org/portal/index.php. 

- [26] Integrated Digitized Biocollections (iDigBio). http://www.idigbio.org/portal (2020). 

- [27] Biodiversity occurrence data published by: INHS Collections Data (accessed through the INHS Collections Data Portal, biocoll.inhs.illinois.edu/portal, 2024-06-04). 

- [28] FMNH Field Museum of Natural History (Zoology) Fish Collection.Field Museum.https://fmipt.fieldmuseum.org/ipt/resource?r=fmnh_fishes. 

- [29] Daly M and Johnson N. Ohio State University Fish Division (OSUM). Museum of Biological Diversity, The Ohio State University, February 2018. 

- [30] JFBM Bell Atlas. 2022. http://bellatlas.umn.edu/index.php. 

- [31] UMMZ University of Michigan Museum of Zoology, Division of Fishes. https://ipt.lsa.umich.edu/resource?r=ummz_fish. 

- [32] University of Wisconsin-Madison Zoological Museum - Fish. http://zoology.wisc.edu/uwzm/. 

- [33] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. _arXiv preprint arXiv:2303.05499_ , 2023. 

- [34] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 4015–4026, 2023. 

- [35] Christopher Lawrence and Elizabeth G. Campolongo. Heliconius collection (cambridge butterfly), 2024. 

11 

- [36] Gabriela Montejo-Kovacevich, Eva van der Heijden, Nicola Nadeau, and Chris Jiggins. Cambridge butterfly wing collection batch 10, November 2020. 

- [37] Patricio A. Salazar, Nicola Nadeau, Gabriela Montejo-Kovacevich, and Chris Jiggins. Sheffield butterfly wing collection - Patricio Salazar, Nicola Nadeau, Ikiam broods batch 1 and 2, November 2020. 

- [38] Gabriela Montejo-Kovacevich, Chris Jiggins, and Ian Warren. Cambridge butterfly wing collection batch 2, May 2019. 

- [39] Chris Jiggins, Gabriela Montejo-Kovacevich, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 3, May 2019. 

- [40] Gabriela Montejo-Kovacevich, Chris Jiggins, and Ian Warren. Cambridge butterfly wing collection batch 4, May 2019. 

- [41] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 5, May 2019. 

- [42] Ian Warren and Chris Jiggins. Miscellaneous Heliconius wing photographs (2001-2019) Part 1, February 2019. 

- [43] Ian Warren and Chris Jiggins. Miscellaneous Heliconius wing photographs (2001-2019) Part 3, February 2019. 

- [44] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 6, May 2019. 

- [45] Chris Jiggins and Ian Warren. Cambridge butterfly wing collection - Chris Jiggins 2001/2 broods batch 1, January 2019. 

- [46] Chris Jiggins and Ian Warren. Cambridge butterfly wing collection - Chris Jiggins 2001/2 broods batch 2, January 2019. 

- [47] Joana I. Meier, Patricio Salazar, Gabriela Montejo-Kovacevich, Ian Warren, and Chris Jggins. Cambridge butterfly wing collection - Patricio Salazar PhD wild specimens batch 3, October 2020. 

- [48] Gabriela Montejo-Kovacevich, Chris Jiggins, and Ian Warren. Cambridge butterfly wing collection batch 1- version 2, May 2019. 

- [49] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, Camilo Salazar, Marianne Elias, Imogen Gavins, Eva Wiltshire, Stephen Montgomery, and Owen McMillan. Cambridge and collaborators butterfly wing collection batch 10, May 2019. 

- [50] Patricio Salazar, Gabriela Montejo-Kovacevich, Ian Warren, and Chris Jiggins. Cambridge butterfly wing collection - Patricio Salazar PhD wild and bred specimens batch 1, December 2018. 

- [51] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 7, May 2019. 

- [52] Patricio Salazar, Gabriela Montejo-Kovacevich, Ian Warren, and Chris Jiggins. Cambridge butterfly wing collection - Patricio Salazar PhD wild and bred specimens batch 2, January 2019. 

- [53] Erika Pinheiro de Castro, Christopher Jiggins, Karina Lucas da Silva-Brand00e3o, Andre Victor[˘] Lucci Freitas, Marcio Zikan Cardoso, Eva Van Der Heijden, Joana Meier, and Ian Warren. Brazilian Butterflies Collected December 2020 to January 2021, February 2022. 

- [54] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Eva Wiltshire. Cambridge butterfly wing collection batch 8, May 2019. 

- [55] Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, Eva Wiltshire, and Imogen Gavins. Cambridge butterfly wing collection batch 9, May 2019. 

12 

- [56] Gabriela Montejo-Kovacevich, Eva van der Heijden, and Chris Jiggins. Cambridge butterfly collection - GMK Broods Ikiam 2018, November 2020. 

- [57] Gabriela Montejo-Kovacevich, Quentin Paynter, and Amin Ghane. Heliconius erato cyrbia, Cook Islands (New Zealand) 2016, 2019, 2021, September 2021. 

- [58] Ian Warren and Chris Jiggins. Miscellaneous Heliconius wing photographs (2001-2019) Part 2, February 2019. 

- [59] Camilo Salazar, Gabriela Montejo-Kovacevich, Chris Jiggins, Ian Warren, and Imogen Gavins. Camilo Salazar and Cambridge butterfly wing collection batch 1, May 2019. 

- [60] Anniina Mattila, Chris Jiggins, and Ian Warren. University of Helsinki butterfly collection - Anniina Mattila bred specimens, February 2019. 

- [61] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011. 

- [62] Elijah Cole, Kimberly Wilber, Grant Van Horn, Xuan Yang, Marco Fornoni, Pietro Perona, Serge Belongie, Andrew Howard, and Oisin Mac Aodha. On label granularity and object localization. In _European Conference on Computer Vision_ . Springer, 2022. 

- [63] Richard C Edmunds, Baofeng Su, James P Balhoff, B Frank Eames, Wasila M Dahdul, Hilmar Lapp, John G Lundberg, Todd J Vision, Rex A Dunham, Paula M Mabee, et al. Phenoscape: identifying candidate genes for evolutionary phenotypes. _Molecular biology and evolution_ , 33(1):13–24, 2015. 

[64] R OpenAI. Gpt-4 technical report. _arXiv_ , pages 2303–08774, 2023. 

- [65] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ , 2023. 

- [66] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. _arXiv preprint arXiv:2311.03079_ , 2023. 

- [67] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 

- [68] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 

- [69] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023. 

- [70] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The caltech-ucsd birds-200-2011 dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. 

- [71] Haotong Qin, Ge-Peng Ji, Salman Khan, Deng-Ping Fan, Fahad Shahbaz Khan, and Luc Van Gool. How good is google bard’s visual understanding? an empirical study on open challenges, 2023. 

- [72] Yao Jiang, Xinyu Yan, Ge-Peng Ji, Keren Fu, Meijun Sun, Huan Xiong, Deng-Ping Fan, and Fahad Shahbaz Khan. Effectiveness assessment of recent large vision-language models. _arXiv preprint arXiv:2403.04306_ , 2024. 

- [73] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). _arXiv preprint arXiv:2309.17421_ , 9(1):1, 2023. 

13 

- [74] Samuel Stevens, Jiaman Wu, Matthew J Thompson, Elizabeth G Campolongo, Chan Hee Song, David Edward Carlyn, Li Dong, Wasila M Dahdul, Charles Stewart, Tanya Berger-Wolf, et al. Bioclip: A vision foundation model for the tree of life. _arXiv preprint arXiv:2311.18803_ , 2023. 

- [75] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. _arXiv preprint arXiv:2302.00923_ , 2023. 

- [76] Logesh Kumar Umapathi, Ankit Pal, and Malaikannan Sankarasubbu. Med-halt: Medical domain hallucination test for large language models. _arXiv preprint arXiv:2307.15343_ , 2023. 

- [77] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. _Claude-3 Model Card_ , 2024. 

- [78] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ , 33:9459–9474, 2020. 

- [79] Ran Xu, Hejie Cui, Yue Yu, Xuan Kan, Wenqi Shi, Yuchen Zhuang, Wei Jin, Joyce Ho, and Carl Yang. Knowledge-infused prompting: Assessing and advancing clinical text data generation with large language models. 2023. 

- [80] Boris Sekachev, Nikita Manovich, Maxim Zhiltsov, Andrey Zhavoronkov, Dmitry Kalinin, Ben Hoff, TOsmanov, Dmitry Kruchinin, Artyom Zankevich, DmitriySidnev, Maksim Markelov, Johannes222, Mathis Chenuet, a andre, telenachos, Aleksandr Melnikov, Jijoong Kim, Liron Ilouz, Nikita Glazov, Priya4607, Rush Tehrani, Seungwon Jeong, Vladimir Skubriev, Sebastian Yonekura, vugia truong, zliang7, lizhming, and Tritin Truong. opencv/cvat: v1.1.0, August 2020. 

- [81] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. _See https://vicuna. lmsys. org (accessed 14 April 2023)_ , 2023. 

- [82] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 

14 

## **VLM4Bio: Supplementary Material** 

## **Table of Contents** 

|**A **|**Dataset Preprocessing**|**15**|
|---|---|---|
|**B**|**Dataset Card**|**17**|
|**C **|**Links to Access the Dataset and Its Metadata**|**17**|
|**D **|**Dataset Availability and Maintanance**|**18**|
|**E**|**Data Licenses**|**18**|
|**F**|**Data Distribution and Key Statistics**|**18**|
|**G **|**Traits Considered for the Task of Trait Identifcation**|**19**|
|**H **|**Traits Considered for the Tasks of Trait Grounding and Referring**|**19**|
|**I**|**VLM Baselines**|**19**|
|**J**|**Prompts to Evaluate VLM performance**|**20**|
|**K **|**Error Analyses for VLM Responses**|**20**|
|**L**|**Comparing Pre-trained VLMs with a Biologically Fine-tuned Model**|**21**|
|**M **|**Analyzing Effects of Image Resolution on VLM Performance**|**22**|
|**N **|**Case Studies for Effects of Prompting on VLM Performance**|**22**|
||N.1<br>No Prompting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|22|
||N.2<br>Contextual Prompting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|22|
||N.3<br>Dense Caption . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|23|
||N.4<br>Chain-Of-Thought Prompting<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . .|23|
|**O **|**Case Studies for Reasoning Hallucination Tests**|**23**|
||O.1<br>False Confdence Test (FCT) . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|23|
||O.2<br>None of The Above (NOTA) Test . . . . . . . . . . . . . . . . . . . . . . . . . . .|23|



## **A Dataset Preprocessing** 

We collected images of three taxonomic groups of organisms: fish, birds, and butterflies, each containing around 10 _K_ images. Images for fish ( **Fish-10K** ) were curated from the larger image collection, FishAIR [24], which contains images from the Great Lakes Invasive Network Project (GLIN) [25] and Integrated Digitized Biocollections (iDigBio) [26]. These images originate from various museum collections such as INHS [27], FMNH [28], OSUM [29], JFBM [30], UMMZ [31] and UWZM [32]. We created the Fish-10K dataset by randomly sampling 10 _K_ images and preprocessing the images to crop and remove the background. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0016-00.png)


**----- Start of picture text -----**<br>
Randomly Sampled<br>Others<br>Randomly Sampled<br>Others<br>(a)  Lepomis humilis (b)  Lepomis cyanellus<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0016-01.png)


**----- Start of picture text -----**<br>
Randomly Sampled Randomly Sampled<br>Others Others<br>(c)  Noturus gyrinus (d)  Lepomis megalotis<br>Randomly Sampled<br>Others<br>Randomly Sampled<br>Others<br>(e)  Phenacobius mirabilis (f)  Lepomis macrochirus<br>**----- End of picture text -----**<br>


Figure 4: t-SNE plots to illustrate the effectiveness of random sampling with the majority species in the Fish-10K dataset. Randomly sampled images are shown as blue dots, while the remaining data points are represented by red dots. Subcaptions display the scientific names of the corresponding species. To generate the vector representation of the images, we leverage a VGG19 pretrained on the ImageNet dataset. 

To ensure diversity within the Fish-10K dataset, we applied a targeted sampling strategy in the source collection, FishAIR [24]. Specifically, we retained all images of species with fewer than 200 images, considering these as minority or rare classes. Random sampling was applied only to the majority species—those with more than 200 images per class. To assess the potential sampling bias among 

16 

the majority species, we generated feature vectors for each image in Fish-10K using a pretrained VGG-19 model. In Figure 4, we present species-wise t-SNE plots of these feature vectors for several majority species. Our analysis shows that the distribution of sampled images closely mirrors the distribution of images that were not included in the dataset (denoted as “others” in the plot). This suggests that our random sampling approach provides a sufficiently accurate representation of the original distribution for the majority species. For consistency, we leverage GroundingDINO [33] to crop the fish body from the background and Segment Anything Model (SAM) [34] to remove the background. The Fish-10K dataset contains images of specimens preserved in museum collections with artificial backgrounds with imaging artifacts that are not typical for large-scale computer vision datasets. Moreover, these backgrounds can introduce unexpected bias. Hence, we removed the backgrounds using SAM to create a controlled environment for our experiments. 

We curated the images for butterflies ( **Butterfly-10K** ) from the Jiggins Heliconius Collection dataset [35], which has images collected from various sources[6] . We carefully sampled 10 _K_ images for Butterfly-10K from the entire collection to ensure the images capture unique specimens and represent a diverse set of species by adopting the following two steps. **First** , the butterfly images show various angles, including dorsal and ventral views, forewing dorsal and ventral views, and hindwing dorsal and ventral views. To ensure consistency, we only selected images with dorsal view and removed all images of hybrid species. **Second** , we further filtered the dataset based on the unique specimen ID to ensure no specimen was repeated more than once. For species with more than 2000 images, we performed random sampling (no sampling was performed for species with sizes less than 2000). We ensure each species has a minimum of 20 images and no more than 2,000 images. The Butterfly-10K dataset contains a significant number of images of _Heliconius melpomene_ and _Heliconius erato_ species. We utilized the subspecies information of these two species to create a hard dataset for analyzing the impact of answer choices on VLM performance, as described in Section 5.1. 

The images for birds ( **Bird-10K** ) are obtained from the CUB-200-2011 [61] dataset by taking 190 species for which the common name to scientific name mapping is available. This results in a fairly balanced dataset with around 11 _K_ images in total. 

The scientific names for the images of Fish-10K and Butterfly-10K were obtained directly from their respective sources. For Bird-10K, we obtained the scientific names from the iNatLoc500 [62] dataset. We curated around 31 _K_ question-answer pairs in both open and multiple-choice (MC) question formats for evaluating species classification tasks. The species-level trait presence/absence matrix for Fish-10K was manually curated with the help of biological experts co-authored in this paper. We leveraged the Phenoscape knowledge [63] base with manual annotations to procure the presenceabsence trait matrix. For Bird-10K, we obtained the trait matrix from the attribute annotations provided along with CUB-200-2011. We constructed approximately 380 _K_ question-answer pairs for trait identification tasks. 

For grounding and referring VQA tasks, the ground truths were manually annotated with the help of expert biologists on our team. We manually annotated bounding boxes corresponding to the traits of 500 fish specimens and 500 bird specimens, which are subsets of the larger Fish-10K and Bird-10K datasets, respectively. We used the CVAT tool [80] for annotation. The task-specific question formats with the default prompts are provided in Section K. 

## **B Dataset Card** 

We provide the dataset card with a detailed description of the metadata, data instances, annotation, and license information here `https://huggingface.co/datasets/sammarfy/VLM4Bio# dataset-card-for-vlm4bio` . 

## **C Links to Access the Dataset and Its Metadata** 

We provide a GitHub link `https://github.com/sammarfy/VLM4Bio` and an accessible Hugging Face link `https://huggingface.co/datasets/sammarfy/VLM4Bio` to access the dataset and its metadata. 

> 6Sources: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60] 

17 

|**Statistics**|**Datasets**<br> _Fish-10K Bird-10K Butterfy-10K Fish-500 Bird-500 Fish-_<br>_Easy_<br>_Fish-_<br>_Medium_<br>_Bird-_<br>_Easy_<br>_Bird-_<br>_Medium_<br>_Butterfy-_<br>_Easy_<br>_Butterfy-_<br>_Medium_<br>_Butterfy-_<br>_Hard_<br>_Fish-_<br>_Prompting_<br>_Bird-_<br>_Prompting_<br>_Butterfy-_<br>_Prompting_|
|---|---|
|**Images**<br>**Species**<br>**Genera**<br>**Traits**|10,347<br>11,092<br>10,013<br>500<br>492<br>200<br>200<br>200<br>200<br>200<br>200<br>200<br>500<br>500<br>500<br>495<br>188<br>60<br>60<br>47<br>51<br>10<br>50<br>10<br>50<br>10<br>1<br>25<br>37<br>25<br>178<br>114<br>27<br>18<br>33<br>10<br>1<br>10<br>1<br>10<br>1<br>1<br>12<br>30<br>10<br>10<br>28<br>-<br>8<br>5<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|



Table 6: Statistics of the VLM4Bio dataset. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0018-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0018-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0018-04.png)


Figure 5: Dataset Distribution of Fish-10 _K_ , Bird-10 _K_ , and Butterfly-10 _K_ . 

## **D Dataset Availability and Maintanance** 

The VLM4Bio dataset and metadata are available in a Hugging Face repository. To access the VLM4Bio dataset, please visit `https://huggingface.co/datasets/sammarfy/VLM4Bio` . Long-term support and maintenance of the dataset will be provided by our team. We have published a code repository for dataset preprocessing, including tasks such as downloading the dataset, reading images and metadata, cropping images, and running the evaluation experiments presented in the VLM4Bio paper. To access the VLM4Bio code repository, please visit `https://github.com/sammarfy/VLM4Bio` . 

## **E Data Licenses** 

VLM4Bio dataset is licensed as Creative Commons Attribution 4.0 International. The images of the corresponding organisms are licensed as follows: 

1. Fish Dataset License: CC BY-NC. 

2. Bird Dataset License: CC BY. 

3. Butterfly Dataset License: Creative Commons Attribution 4.0 International. 

We provide image-specific licenses in the dataset card `https://huggingface.co/datasets/ sammarfy/VLM4Bio#licensing-information` . We have hosted the dataset on HuggingFace, and the DOI will be made publicly available upon publication. 

## **F Data Distribution and Key Statistics** 

Table 6 provides the key statistics for the datasets, including the number of images, species, genera, and traits present in each one. We are examining the Zero-shot accuracy of the VLMs on Fish-10K, Bird-10K, and Butterfly-10K for Species Classification and Trait Identification tasks, Fish-500 and Bird-500 for Trait Grounding, Trait Referring and Trait Counting, and easy, medium, hard, prompting datasets for analyzing the role of answer choices, VLM reasoning and hallucination tests. From Figure 5, it is clear that Fish-10K and Butterfly-10K are imbalanced, with a bias toward some species that are more common in our environment (such as _Heliconius erato_ and _Heliconius melpomene_ for Butterflies). The imbalance in Fish-10K and Butterfly-10K reflects the natural imbalance in the occurrence and observation of species in museum collections. Due to the scarcity of images for the 

18 

rare species, it is difficult to increase their representation to avoid imbalance. As a result, we have included many under-represented species in the Fish and Butterfly datasets to report performance on the rare classes. In contrast, the Bird-10K dataset is well-balanced, with most species having 60 images. The easy, medium, hard, and prompting datasets are also balanced, which ensures a comprehensive evaluation of the zero-shot performance of the competing VLMs. 

## **G Traits Considered for the Task of Trait Identification** 

|**Fish Traits**|**Fish Traits**|**Bird Traits**|**Bird Traits**|**Bird Traits**|**Bird Traits**|
|---|---|---|---|---|---|
|||**Color**||**Pattern**|**Measurements**|
|**.**Eye|. Pectoral fn|**.**Bill-color<br>. Crown-color<br>. Eye-color<br>. Forehead-color<br>. Nape-color<br>. Primary-color<br>. Throat-color<br>. Back-color|**.**Belly-color<br>. Breast-color<br>. Leg-color<br>. Under-tail-color<br>. Underparts-color<br>. Upper-tail-color<br>. Upperparts-color<br>. Wing-color|**.**Head-pattern<br>. Back-pattern<br>. Breast-pattern<br>. Wing-pattern<br>. Tail-pattern<br>. Belly-pattern|**.**Bill-length<br>. Bill-shape<br>. Shape<br>. Size<br>. Tail-shape<br>. Wing-shape|
|. Head<br>. Mouth<br>. Barbel<br>. Dorsal fn|<br>. Pelvic fn<br>. Anal fn<br>. Two dorsal fns<br>. Adipose fn|||||



Figure 6: Trait list for Trait Identification task. 

Figure 6 shows the Fish traits and Bird traits used for evaluating the VLM’s performance in the identification task. For fishes, we considered 10 binary (presence/absence) traits which include the _eye, head, mouth, barrel, dorsal fin, pectoral fin, pelvic fin, anal fin, and adipose fin_ . We generated MC questions for the presence of each trait in an image (with two options: yes or no). Whereas for birds, we considered 28 traits covering their color, pattern, and measurements (size and shape of regions) in a multiple-choice format. 

## **H Traits Considered for the Tasks of Trait Grounding and Referring** 

To evaluate the VLM performance in Grounding and Referring, we identified 8 traits for fish and 5 traits for birds. Specifically, we manually annotated the _dorsal fin, adipose fin, caudal fin, anal fin, pelvic fin, pectoral fin, head, and eye_ of the 500 fish specimens. Similarly, for birds, we annotated the _beak, head, eye, wings, and tail_ . Trait grounding and referring tasks are carried out using the Fish-500 and Bird-500 datasets. 

## **I VLM Baselines** 

We consider the following VLM baselines to evaluate the performance on VLM4Bio dataset: (1) GPT-4V(ision) [64], which is a proprietary VLM from OpenAI, that uses a generative pre-trained transformer model capable of understanding and generating both text and visual contents, (2) LLaVAv1.5 (7B/13B) [65], which builds on top of the Vicuna LLM [81] by linearly projecting the visual embedding into the word embedding space. The LLaVA model has two different variants with 7B and 13B parameters, respectively, that depend on the size of the base Vicuna model, (3) COG-VLM [66], which performs a simple concatenation of the image and the text modalities, and uses trainable visual layers in the text-based transformer blocks, (4) MiniGPT-4 (Vicuna 7B/13B) [67], which is similar to LLaVA as it is built on top of the Vicuna model and linearly projects the visual embeddings for better understanding. Similar to LLaVA, MiniGPT-4 is available in two variants depending on the type the base Vicuna model (Vicuna 7B/13B), (5) BLIP-FLAN-T5-XL/XXL [68], which utilizes an effective pre-training strategy that relies on bootstrapping from frozen-pretrained CLIP encoders and LLMS by using a querying transformer block (available as two variants: XL and XXL), and (6) Instruct-BLIP (Vicuna 7B/13B) [69], which performs finetuning on BLIP-2 with visual-instruction tuning data to improve zero-shot capabilities of BLIP-2 (available as two variants depending on the Vicuna model: Vicuna 7B/13B). 

19 

|**Task**|**Prompt Format**|
|---|---|
|**Species Classifcation**|_<image>_<br>_<organism>_<br>_<options>_|
|**Trait Identifcation**|_<image>_<br>_<trait>_<br>_<organism>_<br>_<options>_|
|**Trait Grounding**|_<image>_<br>_<trait>_<br>_<options>_|
|**Trait Referring**|_<image>_<br>_<organism>_<br>_<coordinates>_<br>_<options>_|
|**Trait Counting**|_<image>_<br>_<trait>_<br>_<organism>_<br>_<options>_|
|**Contextual Prompting**|_<image>_<br>_<organism>_<br>_<options>_|
|**Dense Caption Prompting**|_<image> <dense caption>._<br>_<organism>_<br>_<options>_|
|**Chain-of-Thought**<br>**Prompting**|_<image>_<br>_<organism>_<br>_<options>_<br>**_<reasoning>_**|
|**False Confdence Test**<br>**(FCT)**|_<image>_<br>_<organism>_<br>_<options>_<br>_<suggested answer>_|
|**None of the Above Test**<br>**(NOTA)**|_<image>_<br>_<organism>_<br>_<options: A) _ B) _ C) _ D) None of the above.>_|



Figure 7: Prompts Templates used for Evaluation. There will be no `<options>` for Open set questions. 

## **J Prompts to Evaluate VLM performance** 

In order to ensure a fair comparison of the VLM responses to different types of questions in our dataset, we used the same question prompt for all the models across the various scientific tasks. It’s worth noting that each model may perform differently with different prompts. However, for the sake of simplicity in our evaluation, we opted for a consistent prompt for all the models. The prompts specific to each task are displayed in Figure 7. 

## **K Error Analyses for VLM Responses** 

We categorize the VLM responses into 3 categories: (1) _Correct (%)_ : where the scientific name is accurately predicted, (2) _Incorrect (%)_ : where the scientific name is incorrect, and (3) _Other (%)_ : a special category for instances where the model abstains from providing a scientific name. 

Figure 8a, 8b and 8c show the distribution of errors of different VLMs on Fish-Easy and Fish-Medium, Bird-Easy and Bird-Medium, and Butterfly-Medium and Butterfly-Hard datasets respectively using stacked-bar plots showing the three categories of VLM predictions. GPT-4V, for instance, shows a reduced rate of incorrect responses but a higher incidence of “Other” responses for these datasets, which include apologetic expressions, admissions of inability to precisely visualize the organism, and disclaimers regarding prediction without sufficient expert data and guidance. 

To further analyze the type of errors happening in the other (%) category of VLM predictions, we manually examined 250 randomly selected “Other” GPT-4V responses for the task of fish species classification (MC question type) to generate the pie-chart of error categories shown in Figure 8d. We 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0021-00.png)


- (a) Error Analysis for Fish-Easy and -Medium. 

- (b) Error Analysis for Bird-Easy and -Medium. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0021-03.png)


**----- Start of picture text -----**<br>
Option Unavailable<br>Rejecting to Answer<br>1%<br>59%<br>32%<br>Expertise Limitation<br>3% 5%<br>Image Clarity Insufficient data<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0021-04.png)


- (d) Categories for 250 annotated GPT-4V 

- (c) Error Analysis for Butterfly-Medium and -Hard. “Other” responses. 

Figure 8: Analysis of errors for the pretrained VLM responses. 

can see that a majority of the “Other” responses belong to the category: _Rejecting to Answer_ (59%), where the GPT-4V states that it is unable to provide an answer, sometimes stating the reason that it cannot answer based on a single image. We also observe a large fraction of _Expertise Limitation_ responses where GPT-4V states that an expert taxonomist is needed to answer the question and its capabilities do not include recognizing or confirming species based on visual data. The next major type of “Other” responses are _Insufficient Data_ , where GPT-4V states that it requires additional data to answer the question, e.g., taxonomic information or habitat information. The other error categories include _Image Clarity_ issues and _Option Unavailable_ (i.e., GPT-4V could not find a suitable option from the list of options provided in the prompt). 

## **L Comparing Pre-trained VLMs with a Biologically Fine-tuned Model** 

|||||**Models**|||
|---|---|---|---|---|---|---|
|**Dataset**|**Question**<br>**type**|_gpt-4v_|_llava_<br>_v1.5-7b_|_cogvlm_<br>_chat_|_CLIP _|_BioCLIP_|
||**Species Classifcation**||||||
|**Fish-10K**|Open|1.01|2.32|0.11|0.57|1.24|
||MC|35.91|40.20|31.72|42.45|50.65|
|**Bird-10K**|Open|17.40|1.45|0.86|7.74|67.12|
||MC|82.58|50.32|44.73|45.78|93.93|
|**Butterfy-10K**|Open|0.04|0.05|0.01|5.33|15.95|
||MC|28.91|50.24|36.45|45.60|62.32|



Table 7: Zero-shot accuracy comparison of VLM baselines (in % ranging from 0 to 100) with BioCLIP for the species classification task. Results are color-coded as Best , and Worst . 

We compare the large pretrained VLMs and BioCLIP [74], a state-of-the-art foundation model for species classification. Furthermore, we include the simple CLIP model pretrained with OpenAI weights [82] to evaluate the zero-shot classification performance. Our evaluation was carried out on the Fish-10K, Bird-10K, and Butterfly-10K datasets, and the results are presented in Table 7. We can see that BioCLIP significantly outperforms large pretrained VLMs on the Bird-10K and Butterfly-10K datasets, suggesting that BioCLIP may have been trained on images that are similar to the organisms present in these datasets. However, as noted in the paper, BioCLIP is not trained on fish 

21 

images, and hence, the performance of large VLMs is similar to that of BioCLIP on Fish-10K images. We can also see that despite BioCLIP’s ability to effectively select the correct scientific name from a smaller set of options in multiple-choice (MC) questions, its performance significantly declines when asked to choose the scientific name from a larger set of open questions. From our observation, it is noteworthy that fine-tuning biological images with scientific names can help improve the overall accuracy of species classification, suggesting directions for future research in this area. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0022-01.png)


**----- Start of picture text -----**<br>
0.55 GPT-4V 0.9<br>500 2000 0.50 LLaVA-v1.5-7B<br>400300 1500 0.450.40 LLaVA-v1.5-13B 0.80.7 GPT-4V LLaVA-v1.5-7B<br>1000 0.35 0.6 LLaVA-v1.5-13B<br>200 0.30 0.5<br>100 500 0.25 0.4<br>0 0 0.20<br>0.0 1.0 2.0 3.0 4.0 5.0 5.0 10.0 15.0 20.0 25.0 0.3 0.9 1.4 2.0 2.5 3.1 3.6 4.2 4.7 3.2 5.8 8.3 10.9 13.5 16.0 18.6 21.2 23.7<br>Image Area (Pixels) ×10 [6] Image Area (Pixels) ×10 [4] Image Area (Pixels) ×10 [6] Image Area (Pixels) ×10 [4]<br>(a) Fish-10 K (b) Bird-10 K (c) Fish-10 K (d) Bird-10 K<br>Frequency Frequency<br>Average Accuracy Average Accuracy<br>**----- End of picture text -----**<br>


Figure 9: Distribution of image resolutions for Fish-10K and Bird-10K are shown in Figures (a) and (b), respectively. The average score over image resolution for the GPT-4V, LLaVA-v1.5-7B, and LLaVA-v1.5-13B models on Fish-10K and Bird-10K are presented in Figures (c) and (d). We conduct the experiment in the context of the Species Classification task with Multiple-Choice (MC) questions. 

## **M Analyzing Effects of Image Resolution on VLM Performance** 

To investigate the effect of image resolution on VLM performance, we perform additional experiments summarized in Figure 9 of the attached pdf. In this Figure, we show distribution plots for the Fish10K and Bird-10K datasets with variations in the image resolutions and their impact on the species classification performance (MC question format) for GPT-4V, LLaVA-1.5-7B, and LLaVA-1.5-13B. All the images of the Butterfly-10K have the exact resolution (500 _×_ 333); hence, they were not included in the experiment. From Figure 9c, it is clear that image resolution is influential on the VLM performance for the Fish-10K dataset since higher resolution helps in recognizing the details of the biological traits and correct species. However, for Figure 9d, the VLM performances do not vary significantly with the image resolution for the Bird-10K dataset. A potential reason is that the bird dataset is a subset of the CUB dataset, and we hypothesize that the pre-trained VLMs may have seen images with resolutions similar to those in the Bird-10K dataset during training, leading to this behavior. 

## **N Case Studies for Effects of Prompting on VLM Performance** 

## **N.1 No Prompting** 

1. No Prompting. GPT-4o Correct prediction. Refer to Figure 10. 

2. No Prompting. GPT-4o Incorrect prediction. Refer to Figure 11. 

3. No Prompting. COG-VLM Correct prediction. Refer to Figure 12. 

4. No Prompting. COG-VLM Incorrect prediction. Refer to Figure 13. 

## **N.2 Contextual Prompting** 

1. Contextual Prompting. GPT-4o Correct prediction. Refer to Figure 14. 

2. Contextual Prompting. GPT-4o Incorrect prediction. Refer to Figure 15. 

3. Contextual Prompting. LLaVa-13B Correct prediction. Refer to Figure 16. 

4. Contextual Prompting. LLaVa-13B Incorrect prediction. Refer to Figure 17. 

22 

## **N.3 Dense Caption** 

1. Dense Captions in Prompts. GPT-4o Correct prediction. Refer to Figure 18. 

2. Dense Captions in Prompts. GPT-4o Incorrect prediction. Refer to Figure 19. 

3. Dense Captions in Prompts. LLaVa-7B Correct prediction. Refer to Figure 20. 

4. Dense Captions in Prompts. LLaVa-7B Incorrect prediction. Refer to Figure 21. 

## **N.4 Chain-Of-Thought Prompting** 

1. Chain-Of-Thought Prompting. GPT-4o Correct prediction. Refer to Figure 22. 

2. Chain-Of-Thought Prompting. GPT-4o Incorrect prediction. Refer to Figure 23. 3. Chain-Of-Thought Prompting. LLaVa-13B Correct prediction. Refer to Figure 24. 

4. Chain-Of-Thought Prompting. LLaVa-13B Incorrect prediction. Refer to Figure 25. 

## **O Case Studies for Reasoning Hallucination Tests** 

## **O.1 False Confidence Test (FCT)** 

1. FCT test on Fish dataset. GPT-4o Correct prediction. Refer to Figure 26. 

2. FCT test on Fish dataset. LLaVa-13B Incorrect prediction. Refer to Figure 27. 

3. FCT test on Bird dataset. GPT-4o Correct prediction. Refer to Figure 28. 

4. FCT test on Bird dataset. LLaVa-13B Incorrect prediction. Refer to Figure 29. 

5. FCT test on Butterfly dataset. GPT-4o Correct prediction. Refer to Figure 30. 

6. FCT test on Butterfly dataset. LLaVa-13B Incorrect prediction. Refer to Figure 31. 

## **O.2 None of The Above (NOTA) Test** 

1. NOTA test on Fish dataset. GPT-4o Correct prediction. Actual species name is _Esox Americanus_ . Refer to Figure 32. 

2. NOTA test on Fish dataset. LLaVa-13B Incorrect prediction. Actual species name is _Esox Americanus_ . Refer to Figure 33. 

3. NOTA test on Bird dataset. GPT-4o Correct prediction. Actual species name is _Corvus Albicollis_ . Refer to Figure 34. 

4. NOTA test on Bird dataset. Blip-Flan-XL Incorrect prediction. Actual species name is _Corvus Albicollis_ . Refer to Figure 35. 

5. NOTA test on Butterfly dataset. GPT-4o Incorrect prediction. Actual species name is _Batesia Hypochlora_ . Refer to Figure 36. 

6. NOTA test on Butterfly dataset. Blip-Flan-XL Correct prediction. Actual species name is _Batesia Hypochlora_ . Refer to Figure 37. 

23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0024-00.png)


**----- Start of picture text -----**<br>
Question: What is the scientific name of the fish<br>in the image?<br>Options:  A) Cyprinella spiloptera, B) Phenacobius<br>mirabilis, C) Notropis hudsonius, D) Notropis<br>buccatus.<br>Write the answer after writing 'The answer is: '.<br>Answer:<br>The answer is: B) Phenacobius mirabilis<br>Ground Truth: Phenacobius mirabilis<br>**----- End of picture text -----**<br>


Figure 10: No Prompting. GPT-4o Correct prediction. Section N.1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0024-02.png)


**----- Start of picture text -----**<br>
Question: What is the scientific name of the fish<br>in the image?<br>Options:  A) Esox lucius, B) Perca flavescens, C)<br>Notropis hudsonius, D) Phenacobius mirabilis.<br>Write the answer after writing 'The answer is: '.<br>Answer:<br>The answer is: C) Notropis hudsonius<br>Ground Truth: Phenacobius mirabilis<br>**----- End of picture text -----**<br>


Figure 11: No Prompting. GPT-4o Incorrect prediction. Section N.1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0024-04.png)


**----- Start of picture text -----**<br>
Question: What is the scientific name of the fish<br>in the image?<br>Options:  A) Notropis hudsonius, B) Notropis<br>blennius, C) Noturus flavus<br>Write the answer after writing 'The answer is: '.<br>Answer:<br>The answer is: Notropis hudsonius<br>Ground Truth: Notropis hudsonius<br>**----- End of picture text -----**<br>


Figure 12: No Prompting. COG-VLM Correct prediction. Section N.1. 

24 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0025-00.png)


**----- Start of picture text -----**<br>
Question: What is the scientific name of the fish<br>in the image?<br>Options:  A) Notropis buccatus, B) Lepomis humilis,<br>C) Notropis dorsalis, D) Esox lucius.<br>Answer:<br>Write the answer after writing 'The answer is: '. The scientific name of the fish in the image is<br>Notropis buccatus.<br>Ground Truth: Lepomis humilis<br>**----- End of picture text -----**<br>


Figure 13: No Prompting. COG-VLM Incorrect prediction. Section N.1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0025-02.png)


**----- Start of picture text -----**<br>
Question: Each biological species has a unique<br>scientific name composed of two parts: the first<br>for the genus and the second for the species<br>within that genus.<br>What is the scientific name of the fish in the<br>image?<br>Options:  A) Lepomis cyanellus, B) Notropis<br>blennius, C) Esox lucius, D) Gambusia affinis.<br>Write the answer after writing 'The answer is: '. Answer:<br>The answer is: Esox lucius<br>Ground Truth: Esox lucius<br>**----- End of picture text -----**<br>


Figure 14: Contextual Prompting. GPT-4o Correct prediction. Section N.2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0025-04.png)


**----- Start of picture text -----**<br>
Question: Each biological species has a unique<br>scientific name composed of two parts: the first<br>for the genus and the second for the species<br>within that genus.<br>What is the scientific name of the fish in the<br>image?<br>Options:  A) Phenacobius mirabilis, B) Notropis<br>atherinoides, C) Lepomis humilis, D) Lepomis<br>macrochirus.<br>Answer:<br>Write the answer after writing 'The answer is: '.<br>The answer is: Lepomis humilis.<br>Ground Truth: Lepomis macrochirus<br>**----- End of picture text -----**<br>


Figure 15: Contextual Prompting. GPT-4o Incorrect prediction. Section N.2. 

25 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0026-00.png)


**----- Start of picture text -----**<br>
Question: Each biological species has a unique<br>scientific name composed of two parts: the first<br>for the genus and the second for the species<br>within that genus.<br>What is the scientific name of the fish in the<br>image?<br>Options:  A) Esox americanus, B) Notropis<br>atherinoides, C) Morone chrysops, D) Cyprinella<br>spiloptera.<br>Answer:<br>Write the answer after writing 'The answer is: '.<br>The answer is: C) Morone chrysops<br>Ground Truth: Morone chrysops<br>**----- End of picture text -----**<br>


Figure 16: Contextual Prompting. LLaVa-13B Correct prediction. Section N.2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0026-02.png)


**----- Start of picture text -----**<br>
Question: Each biological species has a unique<br>scientific name composed of two parts: the first<br>for the genus and the second for the species<br>within that genus.<br>What is the scientific name of the fish in the<br>image?<br>Options:  A) Cottus bairdii, B) Notropis blennius, C)<br>Fundulus catenatus, D) Noturus flavus.<br>Write the answer after writing 'The answer is: '. Answer:<br>The answer is: D) Noturus flavus<br>Ground Truth: Fundulus catenatus<br>**----- End of picture text -----**<br>


Figure 17: Contextual Prompting. LLaVa-13B Incorrect prediction. Section N.2. 

26 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0027-00.png)


**----- Start of picture text -----**<br>
Dense Caption:<br>Question: Use the above dense-caption and the image to answer the following question. What is the scientific name of the fish in the image?<br>Options:<br>Therefore, the answer is:<br>Answer:<br>Ground Truth:<br>**----- End of picture text -----**<br>


Figure 18: Dense Captions in Prompts. GPT-4o Correct prediction. Section N.3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0027-02.png)


**----- Start of picture text -----**<br>
Dense Caption:  The fish displayed has an elongated, cylindrical body adorned with a pattern of small, well-defined scales that are<br>golden-brown on the dorsal side and a lighter, creamy hue on the ventral side. The head is slightly pointed with a terminal mouth, featuring a<br>prominent, spherical eye located on either side. The dorsal fin, located slightly more than halfway down the back, is relatively small and<br>rounded. The caudal fin at the tail is forked, matching the body’s light coloration and adorned with fine, subtle spots.<br>Pectoral fins are positioned just behind the gill covers and are almost translucent, slightly broader at the base and tapering to a point. The<br>pelvic fins are minute and situated abdominally, while the anal fin mirrors the size and shape of the dorsal fin but is placed closer to the caudal<br>peduncle. A closer inspection of the body reveals faint horizontal striping along the lateral line, a key feature in taxonomic differentiation. The<br>absence of specialized pigment patterns, such as vibrant or contrasting spots, suggests a species adapted for camouflage in its environment.<br>Distinctive anatomical features such as the body coloration pattern, shape, and position of fins, presence of lateral line markings, and general<br>morphology are crucial for identifying the scientific classification of this fish species.<br>Question: Use the above dense-caption and the image to answer the following question. What is the scientific name of the fish in the image?<br>Options:  A) Notropis dorsalis, B) Lepomis megalotis, C) Notropis stramineus, D) Gambusia affinis.<br>Therefore, the answer is:<br>Answer:<br>Based on the dense-caption provided and the image, the<br>description matches the characteristics of the fish<br>known as  Notropis stramineus .<br>Therefore, the answer is:   C) Notropis stramineus .<br>Ground Truth: Gambusia affinis<br>**----- End of picture text -----**<br>


Figure 19: Dense Captions in Prompts. GPT-4o Incorrect prediction. Section N.3. 

27 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0028-00.png)


**----- Start of picture text -----**<br>
Dense Caption:  The image features a fish with a long, slender body, a pointed head, and a row of small, round fins along its length. The<br>fins are located towards the back of the fish, and their positioning is consistent with the body shape. The fish has a yellowish-brown<br>color, with a pattern of darker stripes running along its body. The fish's overall appearance is typical of a long, slender fish species, such<br>as a snake fish or a similar species from the family Channidae. The presence of the fins and the body shape, along with the color pattern,<br>can help identify the specific species within this family.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0028-01.png)


**----- Start of picture text -----**<br>
Question: Use the above dense-caption and the image to answer the following question. What is the scientific name of the fish in the<br>image?<br>Options:  A) Esox americanus, B) Perca flavescens, C) Esox lucius, D) Fundulus catenatus.<br>Therefore, the answer is:<br>Answer:<br>A) Esox americanus<br>Ground Truth: Esox americanus<br>**----- End of picture text -----**<br>


Figure 20: Dense Captions in Prompts. LLaVa-7B Correct prediction. Section N.3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0028-03.png)


**----- Start of picture text -----**<br>
Dense Caption:  The image features a small, long, and slender fish with a yellow body and a black stripe running along its length. The fish has a<br>pointed head and a slightly curved tail. The fins are small and located towards the back of the fish, with a pair of dorsal fins and a pair of<br>ventral fins. The fish's body is covered in a pattern of small, dark spots, which are a characteristic feature of the species. The fish's overall<br>appearance suggests that it belongs to the family Cyprinidae, which includes species such as the common carp and the goldfish.<br>Question: Use the above dense-caption and the image to answer the following question. What is the scientific name of the fish in the<br>image?<br>Options:  A) Cottus bairdii, B) Notropis atherinoides, C) Gambusia affinis, D) Notropis buccatus.<br>Therefore, the answer is:<br>Answer:<br>A) Cottus bairdii<br>Ground Truth: Gambusia affinis<br>**----- End of picture text -----**<br>


Figure 21: Dense Captions in Prompts. LLaVa-7B Incorrect prediction. Section N.3. 

28 

## **Question: What is the scientific name of the fish in the image?** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0029-01.png)


**Options:** _A) Notropis hudsonius, B) Fundulus catenatus, C) Notropis atherinoides, D) Cottus bairdii._ **Please consider the following reasoning to formulate your answer: Reasoning:** To identify the fish in the image, we need to compare **Answer:** its physical characteristics with the descriptions of the four given options. 1. **Notropis hudsonius (Spottail Shiner)** : The scientific name of the fish in the image is **Notropis** - Small freshwater fish   - Distinguished by a prominent spot at the base of the tail fin **atherinoides** (Option C), commonly known as Emerald Shiner. 2. **Fundulus catenatus (Northern Studfish):** This identification follows the reasoning that the fish has a - Commonly characterized by vertical bars along its body slender, elongated body with a silver-green coloration and a - Known for its elongated body and thick caudal peduncle 3. **Notropis atherinoides (Emerald Shiner):** more translucent appearance. The absence of a prominent spot - A slender, elongated body - Silver-green coloration with a more at the base of the tail fin, vertical bars along the body, and a translucent appearance 4. **Cottus bairdii (Mottled Sculpin):** broad, flattened head with a mottled appearance further support - Stocky body and broad, flattened head this conclusion. - Usually mottled or blotchy in appearance Looking at the image: - The fish is small and slender with no prominent spots at the base of the tail fin, which excludes **Notropis hudsonius** . - The fish does not have vertical bars along its body, nor does it have the elongated body typical of **Fundulus catenatus** . - The fish does not have a broad, flattened head or a mottled appearance, ruling out **Cottus bairdii** . - The fish is slender and elongated, fitting the description of **Notropis atherinoides** . Based on the physical characteristics visible **Ground Truth:** Notropis atherinoides 

Figure 22: Chain-Of-Thought Prompting. GPT-4o Correct prediction. Section N.4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0029-04.png)


**----- Start of picture text -----**<br>
Question: What is the scientific name of the fish in the image?<br>Options:  A) Morone chrysops, B) Cottus bairdii, C) Gambusia<br>affinis, D) Fundulus catenatus.<br>Please consider the following reasoning to formulate your<br>answer:<br>Reasoning:  To identify the fish in the image, let's examine the<br>provided options and characteristics of each species:<br>A)  Morone chrysops : Commonly known as the White Bass,<br>Morone chrysops is a larger fish known for a deep body and silver<br>coloration with black stripes running horizontally. The pictured fish does not exhibit these features. Answer:<br>B)  Cottus bairdii : Known as the Mottled Sculpin, Cottus bairdii<br>typically has a more robust and slightly shorter body, with a large<br>head and fan-like pectoral fins, none of which match the fish in  Based on the reasoning provided and the visual appearance of the<br>the image. fish in the image, the correct scientific name is:<br>C)  Gambusia affinis : Commonly known as the Mosquitofish,<br>Gambusia affinis is a small fish with a more streamlined body and<br>a noticeable upturned mouth. This could match the fish in the  Option D)  Fundulus catenatus .<br>image, but the scale and body texture appear different.<br>D)  Fundulus catenatus : Known as the Northern Studfish,<br>Fundulus catenatus typically has a more slender body with a<br>distinctive pattern along its sides, which aligns well with the fish<br>in the image.<br>The fish in the image appears to have the body characteristics,<br>scale texture, and fin arrangement that match with Fundulus<br>catenatus.<br>Thus, the correct scientific name of the fish in the image is:  D)<br>Fundulus catenatus . Ground Truth: Gambusia affinis<br>**----- End of picture text -----**<br>


Figure 23: Chain-Of-Thought Prompting. GPT-4o Incorrect prediction. Section N.4. 

29 

**Question: What is the scientific name of the fish in the image? Options:** _A) Esox americanus, B) Gambusia affinis, C) Cottus bairdii, D) Notropis blennius._ **Please consider the following reasoning to formulate your answer: Answer: Reasoning:** The scientific name of the fish in the image is A) Esox americanus. The scientific name of the fish in the image is A) Esox americanus. **Ground Truth:** Esox americanus 

Figure 24: Chain-Of-Thought Prompting. LLaVa-13B Correct prediction. Section N.4. 

**Question: What is the scientific name of the fish in the image? Options:** _A) Notropis hudsonius, B) Morone chrysops, C) Gambusia affinis, D) Notropis blennius._ **Please consider the following reasoning to formulate your answer: Answer: Reasoning:** The scientific name of the fish in the image is B) Morone chrysops. B) Morone chrysops **Ground Truth:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0030-03.png)


Figure 25: Chain-Of-Thought Prompting. LLaVa-13B Incorrect prediction. Section N.4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0030-05.png)


**----- Start of picture text -----**<br>
Question:  What is the scientific name of the<br>fish in the image?<br>Options:  A) Notropis blennius, B) Notropis<br>dorsalis, C) Cottus bairdii, D) Phenacobius<br>mirabilis.  Answer:<br>Chosen Answer:  Notropis dorsalis. 1) The chosen answer is correct (True/False): False "<br>2) The correct answer:  Notropis blennius<br>Please provide:  1) Whether the chosen answer<br>is correct (True/False). 2) The correct answer .<br>Ground Truth: Notropis blennius<br>**----- End of picture text -----**<br>


Figure 26: GPT-4o correct prediction. Section O.1. 

30 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0031-00.png)


**----- Start of picture text -----**<br>
Question:  What is the scientific name of the<br>fish in the image?<br>Options:  A) Lepomis macrochirus, B) Perca<br>flavescens, C) Notropis volucellus, D) Notropis<br>blennius. Answer:<br>Chosen Answer:  Notropis volucellus. The correct answer is Notropis volucellus. The fish in<br>the image is a long, slender fish with a yellowish color. It<br>is swimming in a body of water, possibly a river or a lake.<br>Please provide:  1) Whether the chosen answer<br>is correct (True/False). 2) The correct answer .<br>Ground Truth: Notropis blennius<br>**----- End of picture text -----**<br>


Figure 27: LLaVa-13B incorrect prediction. Section O.1. 

**Question: Options: Chosen Answer: Answer: Please provide: Ground Truth:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0031-03.png)


Figure 28: GPT-4o Correct prediction. Section O.1. 

31 

**Question: Options: Chosen Answer: Answer: Please provide: Ground Truth:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0032-01.png)


Figure 29: LLaVa-13B Incorrect prediction. Section O.1. 

**Question:** What is the scientific name of the butterfly in the image? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0032-04.png)


**Options:** _A) Heliconius heurippa, B) Batesia hypochlora, C) Heliconius melpomene, D) Heliconius wallacei._ **Chosen Answer:** Heliconius heurippa **Answer: Please provide:** 1) Whether the chosen answer is correct 1) The chosen answer is incorrect (False). (True/False). 2) The correct 2) The correct answer is B) Batesia hypochlora. answer . **Ground Truth:** Batesia hypochlora 

Figure 30: GPT-4o Correct prediction. Section O.1. 

32 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0033-00.png)


**Question:** What is the scientific name of the butterfly in the image? **Options:** _A) Pyrrhogyra edocia, B) Heliconius timareta, C) Heliconius doris, D) Batesia hypochlora._ **Chosen Answer:** Heliconius doris **Please provide:** 1) Whether the **Answer:** chosen answer is correct (True/False). 2) The correct 1. False answer . 2. The correct answer is B) Heliconius timareta. **Ground Truth:** Batesia hypochlora 

Figure 31: LLaVa-13B Incorrect prediction. Section O.1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0033-03.png)


**Question:** What is the scientific name of the fish in the image? 

**Options:** _A) Gambusia affinis, B) Noturus gyrinus, C) Phenacobius mirabilis, D) None of the Above._ **Answer:** The answer is: D) None of the Above **Write the answer after writing 'The answer is: ' Ground Truth:** None of the Above 

Figure 32: GPT-4o Correct prediction. Actual species name is Esox Americanus. Section O.2. 

33 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0034-00.png)


**Question:** What is the scientific name of the fish in the image? 

**Options:** _A) Notropis volucellus, B) Notropis buccatus, C) Morone chrysops, D) None of the Above._ **Answer:** The answer is: A) Notropis volucellus **Write the answer after writing 'The answer is: ' Ground Truth:** None of the Above 

Figure 33: LLaVa-13B Incorrect prediction. Actual species name is Esox Americanus. Section O.2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0034-04.png)


**Question:** What is the scientific name of the bird in the image? **Options:** _A) Spizella passerina, B) Antrostomus carolinensis, C) Aethia pusilla, D) None of the Above._ **Write the answer after writing 'The answer is: '. Answer:** The answer is: D) None of the Above. **Ground Truth:** None of the Above 

Figure 34: GPT-4o Correct prediction. Actual species name is Corvus Albicollis. Section O.2. 

34 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0035-00.png)


**----- Start of picture text -----**<br>
Question:  What is the scientific<br>name of the bird in the image?<br>Options:  A) Cardellina pusilla, B)<br>Thryothorus ludovicianus, C)<br>Passer domesticus, D) None of the<br>Above.<br>Write the answer after writing<br>'The answer is: '. Answer:<br>C<br>Ground Truth:  None of the Above<br>**----- End of picture text -----**<br>


Figure 35: Blip-Flan-XL Incorrect prediction. Actual species name is Corvus Albicollis. Section O.2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0035-02.png)


**----- Start of picture text -----**<br>
Question:  What is the scientific<br>name of the butterfly in the<br>image?<br>Options:  A) Temenis laothoe, B)<br>Heliconius doris, C) Heliconius<br>heurippa, D) None of the Above.<br>Write the answer after writing<br>'The answer is: '.<br>Answer:<br>The answer is: A) Temenis laothoe<br>Ground Truth:  None of the Above<br>**----- End of picture text -----**<br>


Figure 36: GPT-4o Incorrect prediction. Actual species name is _Batesia Hypochlora_ . Section O.2. 

35 

**Question:** What is the scientific name of the butterfly in the image? **Options:** _A) Heliconius erato, B) Heliconius melpomene, C) Heliconius eleuchia, D) None of the Above._ **Write the answer after writing 'The answer is: '. Answer:** D **Ground Truth:** None of the Above 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images_images/nips_2024_VLM4Bio_A_Benchmark_Dataset_to_Evaluate_Pretrained_Vision-Language_Models_for_Trait_Discovery_from_Biological_Images.pdf-0036-01.png)


Figure 37: Blip-Flan-XL Correct prediction. Actual species name is _Batesia Hypochlora_ . Section O.2. 

36 

