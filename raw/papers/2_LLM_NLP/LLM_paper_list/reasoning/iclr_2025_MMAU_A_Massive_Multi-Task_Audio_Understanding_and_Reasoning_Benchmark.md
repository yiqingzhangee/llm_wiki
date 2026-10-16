Pre-print. Under Review. 

## - MMAU: A MASSIVE MULTI-TASK AUDIO UN DERSTANDING AND REASONING BENCHMARK 

**S Sakshi** _[♠∗]_ **, Utkarsh Tyagi** _[♠∗]_ **, Sonal Kumar** _[♠∗]_ **, Ashish Seth** _[♠∗]_ **, Ramaneswaran Selvakumar** _[♠]_ **, Oriol Nieto** _[♣]_ , **Ramani Duraiswami** _[♠]_ , **Sreyan Ghosh** _[♠∗†]_ , **Dinesh Manocha** _[♠†]_ 

> _♠_ University of Maryland, College Park, USA _♣_ Adobe, USA 

> _∗_ Equal Contribution _†_ Equal Advising Correspondence: _{_ ssakshi,sonalkum,sreyang _}_ @umd.edu 

https://sakshi113.github.io/mmau_homepage/ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark_images/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark.pdf-0001-06.png)


**----- Start of picture text -----**<br>
Comprehensive Skill Test Extensive Domain Coverage Diverse Task Types<br>Speech Music<br>Speaker Role Mapping Tongue twisters<br>Understanding World Knowledge<br>Sound<br>Socio-cultural Music<br>Reasoning Understanding Scene Understanding<br>**----- End of picture text -----**<br>


Figure 1: Overview of the MMAU Benchmark. MMAU provides comprehensive coverage across three key domains: speech, sounds, and music, featuring diverse audio samples. It challenges multimodal LLMs with tasks across 27 distinct skills, requiring advanced audio perception, reasoning, and domain-specific knowledge. 

## ABSTRACT 

The ability to comprehend audio—which includes speech, non-speech sounds, and music—is crucial for AI agents to interact effectively with the world. We present MMAU, a novel benchmark designed to evaluate multimodal audio understanding models on tasks requiring expert-level knowledge and complex reasoning. MMAU comprises 10k carefully curated audio clips paired with humanannotated natural language questions and answers spanning speech, environmental sounds, and music. It includes information extraction[1] and reasoning[2] questions, requiring models to demonstrate 27 distinct skills across unique and challenging tasks. Unlike existing benchmarks, MMAU emphasizes advanced perception and reasoning with domain-specific knowledge, challenging models to tackle tasks akin to those faced by experts. We assess 18 open-source and proprietary (Large) Audio-Language Models, demonstrating the significant challenges posed by MMAU. Notably, even the most advanced Gemini Pro v1.5 achieves only 52.97% accuracy, and the state-of-the-art open-source Qwen2-Audio achieves only 52.50%, highlighting considerable room for improvement. We believe MMAU will drive the audio and multimodal research community to develop more advanced audio understanding models capable of solving complex audio tasks. 

## 1 INTRODUCTION 

The recent advancements in Large Language Models (LLMs) have fueled discussions around the development of generalist AI agents, often referred to as Artificial General Intelligence (AGI), capable 

> 1We define an _**information extraction**_ question as one that requires a deep understanding of audio, detailed content analysis, and the application of external world knowledge when necessary. 

> 2 We define a _**reasoning**_ question as one that requires intentional, complex thinking beyond basic content understanding, analysis, and knowledge application, simulating expert-level cognitive processes. 

1 

Pre-print. Under Review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark_images/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Sound Speech Music<br>Info-extraction -  Eco-Acoustic Knowledge Info-extraction -  Phonological Sequence Info-extraction -  Harmony and Chord<br>Question : What natural environment is most Decoding Progressions<br>likely represented by the audio? Question : For the given tongue twister identify Question : Which chord progression is used<br>A. A serene forest which word appears first? in the audio?<br>A. iron  A. G, Em, B7, C6, E7, Am7<br>B. A quiet library B. aluminiuming B. C, G, Am, F, Dm, E7<br>C. A construction site C. copperbottoming C. D, A, Bm, G, E, F#m,<br>D. A peaceful beach D. none of these D. A, E, F#m, D, Bm, C#m<br>Answer : C.  A construction site Answer : B.  aluminiuming Answer : A.  G, Em, B7, C6, E7, Am7<br>Reasoning -  Temporal Event Reasoning Reasoning -  Emotional Tone Interpretation<br>Reasoning -  Multi Speaker Role Mapping<br>Question:  For the given audio, identify which Question:  What is the overall emotional<br>Question:  Identify the role of the first and the<br>of the following sounds can be heard for the second speaker in the conversation atmosphere created by the combination of<br>longest duration. A. Parent and child instruments in the audio?<br>A. Video game sound B. Teacher and student A. Ordinary and dull<br>B. Music C. Doctor and patient B. Unique and heart-touching<br>C. Sound effect D. Coach and athlete               C. Chaotic and confusing<br>D. Background noise Answer:  C.  Doctor and patient D. Energetic and fast-paced<br>Answer:  A.  Video game sound Answer : B.  Unique and heart-touching<br>**----- End of picture text -----**<br>


Figure 2: Examples from the MMAU benchmark illustrating the diverse range of reasoning and information extraction tasks across the domains of sound, speech, and music. Each task involves rich, context-specific audio paired with human-annotated QA pairs that require expert-level knowledge and reasoning abilities. The benchmark covers a wide range of challenges, illustrating the breadth and depth of MMAU’s evaluation scope. 

of solving a diverse range of complex understanding and reasoning tasks (Chowdhery et al., 2023; Achiam et al., 2023; Touvron et al., 2023a). These developments have given rise to AI systems that can match or even surpass human-expert performance in various natural language understanding and reasoning benchmarks (y Arcas & Norvig, 2023; Bubeck et al., 2023; Ge et al., 2024; Latif et al., 2023). Recently, Large Multimodal Models (LMMs), which extend LLMs by integrating multiple modalities beyond text, have demonstrated enhanced general intelligence (Liu et al., 2024a; 2023b; Zhang et al., 2023a; Zhu et al., 2024; Ghosh et al., 2024c). These models excel at a broader set of tasks by improving their ability to observe and perceive the world more comprehensively. 

Benchmarking has been a cornerstone in advancing AI, providing structured challenges that drive the field forward (Raji et al., 2021). However, as highlighted by the AGI taxonomy proposed by (Morris et al., 2024), which defines AGI as a system that performs at the “90th percentile of skilled adults” across a wide array of tasks, current benchmarks fall short of this standard. Tasks such as image and speech recognition, for instance, do not demand the expertise of skilled humans and can often be performed by young children (Lippmann, 1997; Gerhardstein & Rovee-Collier, 2002). In response to this gap, researchers in natural language processing and vision have developed numerous benchmarks (Wang, 2018; Hendrycks et al., 2020; Yue et al., 2024; Lu et al., 2023), many of which require extensive world knowledge and complex reasoning to solve. These benchmarks have pushed the boundaries of model capabilities, prompting incremental improvements. However, there is a notable lack of comprehensive evaluation benchmarks specifically designed to assess the perception and reasoning abilities of audio-language models. Audio perception and reasoning are essential for achieving true AGI, as it is one of the primary modalities through which humans interpret and engage with the world, capturing complex information about the environment, emotions, intentions, and context (You et al., 2024; Gong, 2024). Currently, advanced Large Audio-Language Models (LALMs) are primarily evaluated on foundational tasks such as Automatic Speech Recognition (ASR), Acoustic Scene Classification, or Music Genre Classification (Rubenstein et al., 2023; Gong et al., 2023c; Ghosh et al., 2024c). While these tasks are fundamental for assessing basic audio understanding, they do not require the deliberate and complex reasoning that characterizes more sophisticated forms of intelligence. This highlights a significant gap in the evaluation of LALMs, limiting our ability to fully understand and quantify their true potential in advanced audio tasks. 

**Our Contributions.** We present MMAU, the first comprehensive benchmark tailored for multimodal audio understanding and reasoning. MMAU features over 10,000 expertly annotated audioquestion-response pairs evenly distributed across speech, sound, and music domains. With information extraction and reasoning questions that require models to demonstrate proficiency in 27 distinct skills across unique tasks, MMAU achieves significant **breadth** . Additionally, it covers **depth** by including tasks that require advanced reasoning, such as multi-speaker role mapping, emotional shift detection, and temporal acoustic event analysis. Our audio data is sourced from a wide range of contexts, challenging models to jointly process auditory content and text, recall relevant knowledge, and engage in complex reasoning to solve the tasks. To summarize, our main contributions are: 

2 

Pre-print. Under Review. 

1. We introduce MMAU, the first benchmark specifically designed to evaluate advanced audio perception and reasoning in LALMs. With 10k expertly annotated instances spanning speech, sounds, and music, MMAU meets the highest standards of evaluation by covering both breadth and depth in multimodal audio understanding. 

2. We assess 18 open-source and proprietary models on MMAU and demonstrate that even the most advanced LALMs struggle with tasks that humans easily excel at, achieving only 53% accuracy on our benchmark, highlighting significant gaps in current model capabilities. 

3. We conduct an in-depth analysis of model responses, uncovering key insights such as the effectiveness of audio captions for text-only models, skill-wise performance, and the challenges LALMs face in attending to audio inputs and addressing complex tasks. 

## 2 RELATED WORK 

**Audio-Language Models.** Recent years have seen significant progress in audio understanding, driven by advances in (large) language models that enhance cross-modal interactions between audio and language. Early research focused on developing cross-modal audio-language encoders (ALE) that learn shared representations between the two modalities. Notable models include AudioCLIP (Guzhov et al., 2022), CLAP (Elizalde et al., 2023), and CompA (Ghosh et al., 2023). CompA makes a first attempt to address reasoning in audio-language encoders by improving compositional reasoning through a novel training paradigm. More recently, efforts have shifted toward integrating audio understanding with LLMs, leading to the emergence of Large Audio-Language Models (LALMs). These include models such as Pengi (Deshmukh et al., 2023), Qwen-Audio (Chu et al., 2023), LTU (Gong et al., 2023c), and GAMA (Ghosh et al., 2024c). Leveraging the advanced reasoning capabilities of LLMs, LALMs can respond to complex queries involving audio inputs. For instance, GAMA demonstrates that instruction-tuned models can accurately interpret intricate questions about acoustic scenes. However, unlike humans who can perceive and reason across diverse audio types, LALMs have largely evolved in isolation, with specialized models focusing separately on sounds (e.g., Pengi, LTU, GAMA, etc.), speech (e.g., SALM (Chen et al., 2024), AudioPalm (Rubenstein et al., 2023), etc.), or music (LLark (Gardner et al., 2023), MERT (Li et al., 2023) and others (Liu et al., 2024b; Doh et al., 2023; Won et al., 2024)). Few models are capable of comprehensively understanding all three (e.g., Qwen-Audio (Chu et al., 2024), Audio Flamingo (Kong et al., 2024)). 

**Audio Benchmarks.** With the rapid rise of multimodal LLMs, there has been a significant surge in the development of comprehensive benchmarks for text and vision modalities to assess expertlevel domain knowledge and advanced reasoning capabilities, including subject knowledge (Clark et al., 2018; Hendrycks et al., 2020), safety (Zhang et al., 2023b; Seth et al., 2023), multilingual proficiency (Ahuja et al., 2023), multidisciplinary understanding (Yue et al., 2024; Hu et al., 2024), perception tests (Yuan et al., 2023), mathematical reasoning (Li et al., 2024; Zhang et al., 2024), and video understanding (Li et al., 2023; Ning et al., 2023; Fu et al., 2024a). However, despite this progress, there is still a notable lack of similarly complex benchmarks for the audio modality. To address this gap, a few attempts have been made to build audio-language benchmarks for speech (e.g., OpenASQA (Gong et al., 2023b)), sound (e.g., CompA (Ghosh et al., 2023), CompA-R (Ghosh et al., 2024c)), music (e.g., MusicBench (Melechovsky et al., 2023), MuChin (Wang et al., 2024b), MuChoMusic (Weck et al., 2024)), and their combinations (e.g., AIR-Bench Yang et al. (2024), AudioBench Wang et al. (2024a)). These benchmarks, however, either focus on limited reasoning tasks like compositional or temporal reasoning Ghosh et al. (2023) or rely on fundamental audio tasks like ASR and acoustic scene classification Yang et al. (2024). To the best of our knowledge, no existing benchmark fully addresses the breadth and depth of reasoning required to evaluate advanced audio understanding, leaving a critical gap in the assessment of LALMs’ capabilities. 

## 3 THE MMAU BENCHMARK 

## 3.1 OVERVIEW OF MMAU 

We introduce the Massive Multi-Task Audio Understanding and Reasoning Benchmark (MMAU), a novel benchmark designed to evaluate the expert-level multimodal reasoning and knowledgeretrieval capabilities of large audio-language models (LALMs). MMAU comprises of carefully 

3 

Pre-print. Under Review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark_images/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Conversational Fact Key highlight Extraction Eco-Acoustic Knowledge Phonological Sequence Decoding Temporal Event Reasoning<br>Retrieval Dissonant Emotion  Event-Based Sound<br>Interpretation Reasoning<br>Sound-Based Event<br>Phonemic Stress  Recognition Counting<br>Pattern Analysis Sound Ambient Sound<br>Speech33% 22% summarisationEmotion State  Speech34% Sound Interpretation<br>42%<br>Knowledge RetrievalEvent-Based  Music45% Melodic StructureInterpretation Multi Speaker Role Emotion Flip MappingDetection Music25% Acoustic Source Inference<br>Musical Texture Harmony and Chord  Socio-cultural  Acoustic Scene<br>Interpretation Progressions Interpretation Reasoning<br>Lyrical<br>Instrumentation UnderstandingRhythm and Tempo Reasoning Musical Genre Reasoning ReasoningTemporal  InterpretationEmotional Tone<br>**----- End of picture text -----**<br>


Figure 3: **(Left)** Distribution of skills required for information extraction questions in the MMAU benchmark across the domains of sound, speech, and music. **(Right)** Distribution of skills required for reasoning questions in the MMAU benchmark across the same domains. Each question in MMAU demands the model to apply one or more of these skills to generate a reliable and accurate response. Appendix H provides example questions demanding these skills and the specific tasks associated with them. This chart underscores the diverse cognitive abilities necessary for success in the benchmark, mirroring the complexity and expert-level reasoning involved. 

curated audio clips paired with human-annotated natural language questions and answers meticulously crafted by domain experts. Spanning all three major audio domains—speech, sounds, and music—MMAU includes 27 distinct tasks, consisting of 16 reasoning and 11 information extraction tasks. MMAU is uniquely designed to test LALMs’ advanced cognitive abilities, challenging models with questions that require complex, deliberate reasoning and knowledge retrieval grounded in audio perception. To our knowledge, MMAU stands as the first comprehensive benchmark to rigorously assess these capabilities, filling a critical gap in the evaluation of LALMs. 

Table 1 provides an overview of MMAU, which consists of 10,000 multiple-choice questions (MCQs) divided into a test- _mini_ set and a main test set. The test- _mini_ set, comprising 1,000 questions, reflects the task distribution of the main test set and is intended for hyperparameter tuning. The multiple-choice format was selected to standardize evaluation and align with widely accepted practices in LLM evaluation (Hendrycks et al., 2020; Yue et al., 2024). Just as humans often struggle with closely related options in multiple-choice questions, we anticipate that models may face similar dif- 

||**Statistics**<br>Total Questions<br>Audio Domains|**Number**<br>10,000<br>3|
|---|---|---|
||Domain Categories (Speech:Music:Sound)<br>Diffculties (Easy: Medium: Hard)|10:10:7<br>22%:56%:22%|
||Splits (test-mini: test)<br>Information Extraction Based Questions<br>Reasoning Based Questions<br>Average question length<br>Average option length<br>Average audio length|1000:9000<br>3499 (34.99%)<br>6501 (65.74%)<br>9.28 words<br>5.23 words<br>10.14 sec|



Table 1: Core statistics of the MMAU benchmark 

ficulties. Each question in MMAU is manually categorized by domain experts into easy, medium, or hard difficulty levels. MMAU assesses models across 27 distinct skills, with questions focused on either information extraction (3,499 questions) or reasoning (6,501 questions). The detailed breakdown of skills for both question types is shown in Fig. 3. For this benchmark, the skills required for information extraction and reasoning are kept disjoint—meaning a skill used for an information extraction question will not be required for a reasoning question—although individual questions may require multiple skills from each respective category. MMAU is specifically designed to evaluate advanced audio comprehension, information retrieval (with or without external knowledge), and complex reasoning, pushing models to not only perceive and understand multimodal information but also apply subject-specific knowledge and sophisticated reasoning to solve problems accurately. 

## 3.2 DATA CURATION AND ANNOTATION 

We follow a rigorous 7-step pipeline for curating MMAU, described below: 

4 

Pre-print. Under Review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark_images/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark.pdf-0005-01.png)


**----- Start of picture text -----**<br>
Source Task Expert Option Expert MMAU<br>Selection Curation Annotation & Filtering Augmentation Review<br>**----- End of picture text -----**<br>


Figure 4: MMAU Benchmark Construction Pipeline. 

**1. Source Selection:** We began by collecting diverse audio corpora, including speech, music, and environmental sounds, prioritizing real recordings over synthetic data. This initial step was critical, and we gathered 13 audio corpora to ensure a strong foundation for task development (more details in Appendix E). 

**2. Task Curation:** Leveraging insights from these corpora, we consulted domain experts to select tasks that would challenge models with expert-level reasoning while maintaining real-world relevance. For this step, we also considered the possibility of generating synthetic audios. We curated tasks based on their potential to assess advanced reasoning and knowledge retrieval, carefully filtering an initial set of 90 tasks down to 27, ensuring alignment with real-world applications and the constraints of current generative audio models. 

**3. Expert Annotation:** Domain experts, with help from the authors, crafted high-quality questions and answers for each audio clip. The authors helped curate the set of plausible audios for the experts (based on the final set of tasks selected) and went through multiple iterations. Questions were generated to ensure that each question required real-world complex reasoning and domain-specific knowledge for a faithful question. Experts were asked to follow a set of pre-defined guidelines for QA generation, detailed in Appendix C.3. 

**4. Expert Filtering:** A separate team of experts rigorously reviewed the QA pairs, removing ambiguous, overly complex instances, including instances with low-quality audio samples, to maintain high accuracy and relevance. 

**5. Option Augmentation:** We utilized GPT-4 (OpenAI et al., 2024) to augment each question with additional answer options, systematically increasing task complexity and further testing the models’ reasoning skills. Options were not augmented randomly but generated based on the context of the audio and the question. The augmentation prompt is detailed in Fig. 10 

**6. Expert Review:** Final reviews by experts and authors included tagging every instance with the task that needs to be completed and the specific skills required to complete that task. 

**7. MMAU Finalization:** From the fully annotated and reviewed QA pairs, we selected 10,000 instances to create the final benchmark. This selection was made to ensure a balanced representation of all 27 task types and equal coverage of speech, sound, and music. For evaluation, 1,000 instances were chosen to form the test-mini set, evenly distributed across all tasks, while the remaining instances were allocated to the main test set. 

Details on the background of our expert annotators, generation model and annotation portal are provided in Appendix C. 

## 3.3 COMPARISON WITH OTHER BENCHMARKS 

To highlight the distinction between current benchmarks and MMAU, we break down the information processing steps of a Large Audio-Language Model (LALM): 

Audio Understanding _−−−−−→_ Perception[Knowledge Extraction (optional)] _[ →]_[Reasoning (optional)] 

Most existing benchmarks focus solely on audio understanding, assessing models on basic audio processing tasks like ASR, Speech Emotion Recognition, and other foundational tasks. These tasks primarily evaluate whether the model can comprehend the audio content—such as spoken words, 

5 

Pre-print. Under Review. 

|**Benchmark**<br>**Size**|**Domain**<br>**Speech**<br>**Sound**<br>**Music**|**Tasks**<br>**Expert Comments**<br>**Diffculty Level**<br>**Info Extraction**<br>**Reasoning**|
|---|---|---|
|CompA<br>600|_×_<br>✓<br>_×_|0<br>_×_<br>0.6k<br>✓<br>Requires only sound event sequence understanding. Limited in<br>2.0<br>reasoning depth and knowledge scope.|
|CompA-R<br>1.5k|_×_<br>✓<br>_×_|0<br>_×_<br>1.5k<br>✓<br>Restricted to sounds and only contextual event understanding.<br>3.0<br>Limited in knowledge scope.|
|MuChin<br>1k|_×_<br>_×_<br>_×_|0<br>_×_<br>0<br>_×_<br>Restricted to music with minimal reasoning depth. Limited in<br>2.5<br>knowledge scope.|
|MusicBench<br>0.4k|_×_<br>_×_<br>✓|0<br>_×_<br>0<br>_×_<br>Restricted to music with minimal reasoning depth. Limited in<br>2.5<br>knowledge scope.|
|MuChoMusic<br>1.2k|_×_<br>_×_<br>✓|0.7k<br>✓<br>0.4k<br>✓<br>Restricted to music with open-ended answers. Limited in<br>3.5<br>knowledge scope.|
|OpenASQA<br>8.8k|✓<br>✓<br>_×_|8.8k<br>✓<br>0<br>_×_<br>Requires limited acoustic scene understanding. Does not<br>3.0<br>require external or expert knowledge.|
|AudioBench<br>100k+|✓<br>✓<br>✓|5k<br>✓<br>0<br>_×_<br>Basic acoustic information retrieval with minimal reasoning depth<br>3.5<br>and complexity. Does not require external or expert knowledge.|
|AIR-Bench<br>19k<br>**MMAU**_(ours)_<br>10K|✓<br>✓<br>✓|1.2k<br>✓<br>0.8k<br>✓<br>Basic acoustic information retrieval with minimal reasoning depth<br>2.5<br>and complexity. Does not require external or expert knowledge.|
||✓<br>✓<br>✓|3.5k<br>✓<br>6.5k<br>✓<br>Requires fne-grained audio understanding with expert-level, multi-step<br>4.5<br>reasoning and specialized knowledge across a broad range of topics.|



Table 2: Comparison of MMAU with existing audio understanding and reasoning benchmarks across various statistics. MMAU covers all three domains—speech, sound, and music—while having the highest number of information extraction and complex reasoning tasks. 

emotional tones, or distinct sound events—but they do not challenge the model’s broader cognitive abilities. We argue that this approach falls short in evaluating the true capabilities of LALMs, as simply mastering foundational tasks is insufficient for the next generation of AI agents that must go beyond basic understanding. MMAU targets this gap by moving beyond mere audio understanding to include tasks that require knowledge extraction and complex reasoning. This progression demands that models not only perceive the audio with respect to the text prompt but also apply advanced cognitive skills to respond faithfully. 

Table 2 provides a comparative analysis of MMAU with recent audio reasoning benchmarks. Unlike existing benchmarks, MMAU encompasses all three major audio domains—speech, music, and sounds—and offers the largest set of tasks that integrate both knowledge extraction and reasoning. As illustrated in Fig. 3, MMAU sets itself apart with well-crafted reasoning tasks that are absent in other benchmarks (see Appendix G for further comparisons). Notably, MMAU is the first benchmark to incorporate knowledge-based information extraction questions, pushing the boundaries of what LALMs can achieve. 

To further illustrate the differences between MMAU and other benchmarks, we compare the difficulty levels based on expert ratings (1-5) across 500 randomly selected instances from each benchmark (more details on this in Appendix J). Experts evaluated the benchmarks along two dimensions: Breadth (diversity of tasks and domains) and Depth (task complexity). In terms of breadth, previous benchmarks are often limited to specific domains or task types. For instance, MusicBench (Melechovsky et al., 2023) and MuChin (Wang et al., 2024b) focus solely on basic music information retrieval tasks. When it comes to depth, many benchmarks emphasize specialized reasoning, such as temporal reasoning (Ghosh et al., 2023; 2024c) or content-based reasoning (Gong et al., 2023b), but do not comprehensively evaluate LALMs’ ability to handle more complex tasks like contextual and causal reasoning. While benchmarks like AIR-Bench (Yang et al., 2024) and AudioBench (Wang et al., 2024a) span multiple domains—speech, music, and sound—they predominantly focus on foundational tasks and fail to fully capture the intricate reasoning capabilities of LALMs. 

## 4 EXPERIMENTAL SETUP 

**LALMs.** We compare a range of open-source, open-access, and closed-source LALMs, including (i) Qwen2-Audio-Chat (Chu et al., 2024): A open-access model (only checkpoints are available; training data and details is unknown) with strong capabilities in sound, speech, and music understanding and reasoning. Qwen2-Audio-Instruct is a fine-tuned version with chat abilities based on Qwen2-7B as its LLM. (ii) GAMA (Ghosh et al., 2024c): A leading fully open-source model focused on sound and music understanding, built on LLaMa-2-7B. (iii) GAMA-IT is its fine-tuned variant for complex reasoning. (iv) SALAMONN (Tang et al., 2023): One of the first open-source LALMs, excelling in speech and sound understanding. (v) LTU (Gong et al., 2023c): A fully open-source model with 

6 

Pre-print. Under Review. 

|**Models**<br>**Size**<br>**_{_So, Mu, Sp****_}_**|**Sound**|**Music**<br>**Speech**<br>**Avg**|**Music**<br>**Speech**<br>**Avg**|**Music**<br>**Speech**<br>**Avg**|
|---|---|---|---|---|
||**Test-mini**|**Test**<br>**Test-mini**<br>**Test**<br>**Test-mini**||**Test**<br>**Test-mini**<br>**Test**|
|Random Guess<br>-<br>-<br>Most Frequent Choice<br>-<br>-<br>Human (test-mini)<br>-<br>-|26.72<br>27.02<br>86.31|25.73<br>24.55<br>26.53<br>26.72<br>25.73<br>20.35<br>23.73<br>29.12<br>-<br>78.22<br>-<br>82.17||25.50<br>26.00<br>25.92<br>30.33<br>25.50<br>26.50<br>-<br>82.23<br>-|
|**Large Au**|**dio Language**|**Models (LALMs)**|||
|Pengi<br>323M<br>✓<br>✓<br>_×_<br>Audio Flamingo Chat<br>2.2B<br>✓<br>✓<br>_×_<br>LTU<br>7B<br>✓<br>✓<br>_×_<br>LTU AS<br>7B<br>✓<br>✓<br>✓<br>MusiLingo<br>7B<br>_×_<br>✓<br>_×_<br>MuLLaMa<br>7B<br>_×_<br>✓<br>_×_<br>M2UGen<br>7B<br>_×_<br>✓<br>_×_<br>GAMA<br>7B<br>✓<br>✓<br>_×_<br>GAMA-IT<br>7B<br>✓<br>✓<br>_×_<br>Qwen-Audio-Chat<br>8.4B<br>✓<br>_×_<br>_×_<br>Qwen2-Audio<br>8.4B<br>✓<br>✓<br>✓<br>Qwen2-Audio-Instruct<br>8.4B<br>✓<br>✓<br>✓<br>SALAMONN<br>13B<br>✓<br>✓<br>✓|06.10<br>23.42<br>22.52<br>23.35<br>23.12<br>40.84<br>03.60<br>41.44<br>43.24<br>55.25<br>07.50<br>54.95<br>41.00|08.00<br>28.26<br>25.86<br>24.96<br>27.76<br>44.80<br>03.69<br>45.40<br>43.23<br>**56.73**<br>08.20<br>45.90<br>40.30|02.90<br>03.05<br>01.20<br>15.26<br>18.20<br>11.41<br>09.69<br>12.83<br>17.71<br>9.10<br>10.46<br>20.60<br>03.96<br>06.00<br>05.88<br>32.63<br>30.63<br>22.22<br>32.93<br>30.40<br>06.36<br>32.33<br>30.83<br>18.91<br>28.44<br>28.00<br>18.91<br>44.00<br>40.90<br>30.03<br>05.14<br>06.16<br>03.10<br>**50.98**<br>**53.26**<br>42.04<br>34.80<br>33.76<br>25.50|01.50<br>03.40<br>04.18<br>10.16<br>16.69<br>18.87<br>16.37<br>16.89<br>18.51<br>21.30<br>17.68<br>18.90<br>06.42<br>10.98<br>13.39<br>16.56<br>31.90<br>30.66<br>04.53<br>14.28<br>12.87<br>19.21<br>30.90<br>31.81<br>15.84<br>30.20<br>29.02<br>27.95<br>43.10<br>41.86<br>04.24<br>05.24<br>06.20<br>45.90<br>49.20<br>52.50<br>24.24<br>33.70<br>32.77|
|Gemini Pro v1.5<br>-<br>-|**56.75**|54.46|49.40<br>48.56<br>**58.55**|**55.90**<br>**54.90**<br>**52.97**|
|**Large Language M**||**odels (LLMs)**|||
|GPT4o + weak cap.<br>-<br>-<br>39.33<br>GPT4o + strong cap.<br>-<br>-<br>**57.35**<br>Llama-3-Ins. + weak cap.<br>8B<br>-<br>34.23<br>Llama-3-Ins. + strong cap.<br>8B<br>-<br>50.75||35.80<br>39.52<br>41.9<br>58.25<br>**55.83**<br>49.70<br>**51.73**<br>**64.86**<br>33.73<br>38.02<br>42.36<br>54.05<br>49.10<br>**50.29**<br>48.93<br>55.25||68.27<br>45.70<br>48.65<br>**68.66**<br>**57.30**<br>**58.74**<br>61.54<br>42.10<br>45.87<br>62.70<br>52.10<br>53.57|



Table 3: Performance comparison of various LALMs and LLMs on the test subset of MMAU across sound, speech, and music domains. Human evaluation results are shown for the MMAU test-mini split. We also mark if the training data used to train these models include either of speech, sound or music. The best-performing models in each category are highlighted in **bold** , and the second-best scores are underlined. 

strong audio understanding and reasoning abilities. (vi) LTU-AS (Gong et al., 2023b) is an advanced version capable of joint speech and audio comprehension. Both models use Vicuna-7B as the base LLM. (vii) Audio-Flamingo-Chat (Kong et al., 2024): A fine-tuned version of Audio-Flamingo with chat and instruction-following abilities. Unlike other models, it employs cross-attention and uses OPT-IML-MAX-1.3B as its base LLM. (viii) MusiLingo (Deng et al., 2023): A music captioning and reasoning model that combines a MERT encoder (Li et al., 2023) with Vicuna 7B LLM. MusiLingo is fine-tuned on MusicInstruct (ix) M2UGen (Hussain et al., 2023): A framework capable of completing music understanding and multi-modal music generation tasks (x) MuLLaMa (Liu et al., 2024b): A Music Understanding Language Model designed with the purpose of answering questions based on music. This model is based on MERT (Li et al., 2023) and Llama (Touvron et al., 2023b) (xi) Gemini-Flash and Gemini-Pro (Team et al., 2024): Two proprietary LALMs known for advanced capabilities in speech, music, and sound reasoning. Gemini models are also some of the strongest multimodal systems overall, excelling in both vision and language tasks, though specific architectural details remain unknown. We do not evaluate non-instruct/non-chat versions of Qwen-2, Audio Flamingo, and Pengi as they fail to follow instructions and respond by selecting options. 

**LLMs.** To assess the robustness of MMAU, we also perform a text-only evaluation using various open and closed-source Large Language Models (LLMs), including GPT-4o (OpenAI et al., 2024), a closed-source, state-of-the-art LLM, and Llama 3 8B Instruct (Dubey et al., 2024), an open-source, instruction-tuned model. Additionally, to evaluate whether incorporating external audio information can enhance LLM performance on MMAU, we provide these models with audio captions generated by Qwen2-Audio (Chu et al., 2024). 

**Evaluation Strategy.** We use micro-averaged accuracy as our evaluation metric. Specifically, we present a varying list of options to the models, instructing them to select only one. Since most current LALMs are instruction-tuned for generating open-ended responses (Ge et al., 2024; Gong et al., 2023b), we employ robust regular expressions and develop response-processing workflows to extract key information, which is then matched to one of the provided options using string matching. To mitigate any potential bias in the model’s option selection due to ordering, we randomize the order of the options five times and select the option chosen most frequently. Additionally, we experiment with various prompt sets across all LALMs and report the best results. 

7 

Pre-print. Under Review. 

## 5 RESULTS AND DISCUSSION 

## 5.1 MAIN RESULTS 

Table 3 compares the results of various LALMs on the MMAU benchmark. Our key findings are: 

   1. **MMAU poses a significant challenge.** The MMAU benchmark is highly demanding for current models, both open-source and proprietary. The top-performing LALM achieves only 53% accuracy, while the best-cascaded captioning + LLM approach reaches just 59%. In comparison, human performance achieves 82%. 

   2. **Minimal gap between open-source and proprietary models.** Unlike other domains, we observe only a small performance gap between the best open-source and proprietary LALMs. As shown in Table 3, Qwen2, the leading open-access model, performs almost on par with the proprietary Gemini-Pro, with just a 0.47% difference in average performance. However, the top fully open-source model, GAMA, trails significantly behind, with a larger performance gap of 21% compared to Gemini-Pro. 

   3. **Generalized vs. Specialized Models.** Generalized models trained across multiple domains—speech, sounds, and music—such as Qwen2-Audio, LTU-AS, and Gemini, exhibit strong overall performance. This indicates that larger, more diverse training data leads to a more comprehensive understanding of audio. 

   4. **Models perform best on sound and worst on speech.** With average scores of 18%, 30%, 23% across speech, sound, and music, models perform best on sound-related tasks and struggle the most with music. This suggests that while spoken language _understanding_ has advanced, _reasoning_ over spoken language—especially perception beyond mere content—remains a challenge. On the other hand LALMs have mastered music reasoning, a skill generally not possed non-experts. 

   5. **Cascaded approaches outperform others.** Captioning audios first and then prompting LLMs yields the best results. Enhancing the quality of the captions further improves overall performance. This demonstrates the potential of scaling audio-language understanding through separate advancements in aud ~~io a~~ nd ~~lan~~ guag ~~e re~~ asoning. 

- 5.2 ARE LALMS REALLY LISTENING? 

~~Figure 5~~ compares the performance of various models on the MMAU test set, where the ~~original a~~ udio input is replaced with random Gaussian noise. This experiment helps assess whether models are truly attending to the audio inputs or just respond using language-priors. As shown, the performance of MuLLaMa and SALMONN remains largely unaffected, indicating that these models may not always rely on the audio input to generate responses. In contrast, models like GAMA, Qwen2-Instruct, and Gemini Pro v1.5 exhibit a significant drop in performance, suggesting that they are more attentive to the audio content. We provide examples of incorrect outputs in Appendix I. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark_images/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark.pdf-0008-11.png)


**----- Start of picture text -----**<br>
Audio Noise<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>MuLLaMa SALMONN GAMA Qwen2-Instruct Gemini 1.5 Pro<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 5: Performance comparison on the MMAU test with Gaussian noise replacing the original audio. Models like MuLLaMa and SALMONN show little change in performance, indicating limited reliance on audio input, while others show a significant drop, suggesting greater audio dependence. 

## 5.3 CAN CAPTIONS BRIDGE THE GAP FOR TEXT-ONLY MODELS? 

Figure 5 compares the performance of various models on the MMAU test set, where the original audio input is replaced with captions. We present results using two types of captions: weak captions (generated using EnCLAP (Kim et al., 2024) for sounds, MuLLaMa for music, and Whisper base (Radford et al., 2023) for speech transcripts) and strong, detailed captions (generated using Qwen2-Audio-Instruct with prompts detailed in Appendix L). As the results show, strong captions can indeed help bridge the audio understanding gap for text-only models, with GPT4o achieving the highest accuracy at 59%. Additionally, we demonstrate that enhancing the quality of captions significantly boosts the performance of text-only LLMs (i.e., when captions effectively capture acoustic 

8 

Pre-print. Under Review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark_images/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark.pdf-0009-01.png)


**----- Start of picture text -----**<br>
Other Errors Annotation Errors Other Errors Annotation Errors<br>0.4% 4% Reasoning  5% 5%<br>Errors<br>11%<br>Reasoning<br>Errors<br>18% Answer<br>Extraction Errors<br>Answer  12%<br>Extraction<br>Errors17% Perceptual Errors55% Knowledge  Perceptual Errors64%<br>Errors<br>3%<br>Knowledge<br>Errors<br>5%<br>**----- End of picture text -----**<br>


Figure 7: Distribution of human-annotated error types across 500 instances for Qwen2-Audio-Instruct (Left) and Gemini Pro v1.5 (Right). Appendix K provides detailed definitions of each error type. 

details, text-only LLMs can reliably answer questions.) These findings are consistent with Ghosh et al. (2024a), who show that visual descriptions improve LVLM performance for reasoning prompts. 

## 5.4 DEEP DIVE: SKILL-SPECIFIC MODEL PERFORMANCE 

The average scores for Gemini Pro across easy, Temporal Reasoning medium, and hard questions are 39.60, 43.82, and 36.03, respectively (detailed results for Instrumentation Harmony and ChordProgressions other models in Table 5). While it suggests that models perform consistently across difficulty levels, we wanted to dive deeper into skillGenre and Style Phonemic Stress specific performance. Figure 6 illustrates the Reasoning Pattern Analysis accuracy distribution across easy, medium, and hard questions for eight skills with the highest Rhythm and Tempo Emotional Tone number of questions in the benchmark. SurUnderstanding Interpretation prisingly, the reason for the uniformity across Historical and difficulty levels is that models excel in cerCultural Reasoning tain skills across all difficulties (e.g., Phonemic Easy Medium Hard Stress Pattern Analysis), but consistently strugFigure 6: Accuracy distribution for Gemini Pro across gle with others (e.g., Temporal Reasoning), reeasy, medium, and hard questions, categorized by skill gardless of the question’s difficulty. This obtype. The graph highlights how LALMs excel in some servation highlights that rather than focusing skills across all difficulty levels (e.g., Phonemic Stress on improving model performance in a single Pattern Analysis) but struggle with others (e.g., Temposkill, future work should focus on developing ral Reasoning) regardless of difficulty. a broader range of competencies, ensuring they can handle complex reasoning across various tasks. 

Figure 6: Accuracy distribution for Gemini Pro across easy, medium, and hard questions, categorized by skill type. The graph highlights how LALMs excel in some skills across all difficulty levels (e.g., Phonemic Stress Pattern Analysis) but struggle with others (e.g., Temporal Reasoning) regardless of difficulty. 

## 5.5 PINPOINTING LALM WEAKNESSES: WHERE ARE THEY FALLING SHORT? 

Figure 7 provides a breakdown of the error types made by Qwen2-Audio-Instruct and Gemini Pro v1.5 across 500 instances. The dominant error category for both models is **Perceptual Errors** , with Qwen2-Audio-Instruct showing 55% and Gemini Pro v1.5 at 64%. This indicates that both models struggle primarily with understanding and accurately perceiving the audio inputs. **Reasoning Errors** and **Answer Extraction Errors** (Errors where model outputs and ground-truth answers are same but the model provided an ill-formatted response) account for a significant portion of mistakes, particularly for Qwen2-Audio-Instruct, where 18% of errors are reasoning-based, suggesting that even when models correctly perceive the audio, they often fail to apply the necessary complex reasoning. For Gemini 1.5 Pro, reasoning errors account for 11%, while answer extraction errors remain consistent between both models. Interestingly, **Knowledge Errors** and **Annotation Errors** form smaller percentages. Overall, our error analysis highlights that improving perceptual understanding is crucial for better performance. This can be done through more training data (Liu et al., 2023a), better architectures (Ghosh et al., 2024c) or other methods (Fu et al., 2024b). 

9 

Pre-print. Under Review. 

## 6 CONCLUSION, LIMITATIONS AND FUTURE WORK 

In this paper, we introduce MMAU, a novel large-scale benchmark designed to comprehensively evaluate multimodal audio understanding and reasoning in AI models. MMAU challenges models with a diverse set of tasks that assess 27 distinct skills, emphasizing advanced perception and domain-specific reasoning. The benchmark presents tasks akin to those faced by experts, making it a rigorous test for AI systems. Our evaluations of 18 open-source and proprietary LALMs reveal that even the overall best model achieves only 59% accuracy on MMAU, highlighting the significant challenges it poses. We also provide a detailed analysis of the unique hurdles presented by this benchmark. 

As part of future work, we aim to address in future iterations of MMAU: (i) Currently, we treat skills required for information extraction and reasoning as disjoint sets. As part of future work, we plan to incorporate tasks that require skills from both types. (ii) There is a risk of biases introduced during the human or LLM-driven annotation process. We aim to further refine the dataset to minimize any potential biases. (iii) MMAU focuses on multiple-choice tasks and does not evaluate openended generation, which allows models to reason more freely and exhibit errors such as language hallucinations. Including open-ended tasks will help us better understand these kinds of errors. (iv) Lastly, we plan to broaden the range of tasks and skills covered by MMAU to enhance its challenge and relevance to future models. 

## REFERENCES 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Andrea Agostinelli, Timo I Denk, Zal´an Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, et al. Musiclm: Generating music from text. _arXiv preprint arXiv:2301.11325_ , 2023. 

- Kabir Ahuja, Harshita Diddee, Rishav Hada, Millicent Ochieng, Krithika Ramesh, Prachi Jain, Akshay Nambi, Tanuja Ganu, Sameer Segal, Maxamed Axmed, et al. Mega: Multilingual evaluation of generative ai. _arXiv preprint arXiv:2303.12528_ , 2023. 

- Dmitry Bogdanov, Minz Won, Philip Tovstogan, Alastair Porter, and Xavier Serra. The mtg-jamendo dataset for automatic music tagging. ICML, 2019. 

- S´ebastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv preprint arXiv:2303.12712_ , 2023. 

- Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N Chang, Sungbok Lee, and Shrikanth S Narayanan. Iemocap: Interactive emotional dyadic motion capture database. _Language resources and evaluation_ , 42:335–359, 2008. 

- Santiago Castro, Devamanyu Hazarika, Ver´onica P´erez-Rosas, Roger Zimmermann, Rada Mihalcea, and Soujanya Poria. Towards multimodal sarcasm detection (an ~~O~~ bviously perfect paper). In Anna Korhonen, David Traum, and Llu´ıs M`arquez (eds.), _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pp. 4619–4629, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1455. URL https://aclanthology.org/P19-1455. 

- Zhehuai Chen, He Huang, Andrei Andrusenko, Oleksii Hrinchuk, Krishna C Puvvada, Jason Li, Subhankar Ghosh, Jagadeesh Balam, and Boris Ginsburg. Salm: Speech-augmented language model with in-context learning for speech recognition and translation. In _ICASSP 20242024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 13521–13525. IEEE, 2024. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: 

10 

Pre-print. Under Review. 

- Scaling language modeling with pathways. _Journal of Machine Learning Research_ , 24(240): 1–113, 2023. 

- Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. _arXiv preprint arXiv:2311.07919_ , 2023. 

- Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. _arXiv preprint arXiv:2407.10759_ , 2024. 

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018. 

- Zihao Deng, Yinghao Ma, Yudong Liu, Rongchen Guo, Ge Zhang, Wenhu Chen, Wenhao Huang, and Emmanouil Benetos. Musilingo: Bridging music and text with pre-trained language models for music captioning and query response. _arXiv preprint arXiv:2309.08730_ , 2023. 

- Soham Deshmukh, Benjamin Elizalde, Rita Singh, and Huaming Wang. Pengi: An audio language model for audio tasks. _Advances in Neural Information Processing Systems_ , 36:18090–18108, 2023. 

- Soham Deshmukh, Shuo Han, Hazim Bukhari, Benjamin Elizalde, Hannes Gamper, Rita Singh, and Bhiksha Raj. Audio entailment: Assessing deductive reasoning for audio understanding. _arXiv preprint arXiv:2407.18062_ , 2024. 

- SeungHeon Doh, Keunwoo Choi, Jongpil Lee, and Juhan Nam. Lp-musiccaps: Llm-based pseudo music captioning. _arXiv preprint arXiv:2307.16372_ , 2023. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. Clap learning audio concepts from natural language supervision. In _ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 1–5. IEEE, 2023. 

- Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. _arXiv preprint arXiv:2405.21075_ , 2024a. 

- Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. _arXiv preprint arXiv:2404.12390_ , 2024b. 

- Josh Gardner, Simon Durand, Daniel Stoller, and Rachel M Bittner. Llark: A multimodal foundation model for music. _arXiv preprint arXiv:2310.07160_ , 2023. 

- Yingqiang Ge, Wenyue Hua, Kai Mei, Juntao Tan, Shuyuan Xu, Zelong Li, Yongfeng Zhang, et al. Openagi: When llm meets domain experts. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In _2017 IEEE international conference on acoustics, speech and signal processing (ICASSP)_ , pp. 776–780. IEEE, 2017. 

- Peter Gerhardstein and Carolyn Rovee-Collier. The development of visual search in infants and very young children. _Journal of Experimental Child Psychology_ , 81(2):194–215, 2002. 

- Sreyan Ghosh, Ashish Seth, Sonal Kumar, Utkarsh Tyagi, Chandra Kiran Evuru, S Ramaneswaran, S Sakshi, Oriol Nieto, Ramani Duraiswami, and Dinesh Manocha. Compa: Addressing the gap in compositional reasoning in audio-language models. _arXiv preprint arXiv:2310.08753_ , 2023. 

11 

Pre-print. Under Review. 

- Sreyan Ghosh, Chandra Kiran Reddy Evuru, Sonal Kumar, Utkarsh Tyagi, Oriol Nieto, Zeyu Jin, and Dinesh Manocha. Vdgd: Mitigating lvlm hallucinations in cognitive prompts by bridging the visual perception gap. _arXiv preprint arXiv:2405.15683_ , 2024a. 

- Sreyan Ghosh, Sonal Kumar, Chandra Kiran Reddy Evuru, Oriol Nieto, Ramani Duraiswami, and Dinesh Manocha. Reclap: Improving zero shot audio classification by describing sounds. _arXiv preprint arXiv:2409.09213_ , 2024b. 

- Sreyan Ghosh, Sonal Kumar, Ashish Seth, Chandra Kiran Reddy Evuru, Utkarsh Tyagi, S Sakshi, Oriol Nieto, Ramani Duraiswami, and Dinesh Manocha. Gama: A large audio-language model with advanced audio understanding and complex reasoning abilities. _arXiv preprint arXiv:2406.11768_ , 2024c. 

- Yuan Gong. From audio perception to understanding: A path towards audio agi. In _AI Seminar_ . Stony Brook University, 2024. URL https://ai.stonybrook.edu/ Audio-Perception-Understanding-Path-Towards-Audio-AGI. 

- Yuan Gong, Alexander H. Liu, Hongyin Luo, Leonid Karlinsky, and James Glass. Joint audio and speech understanding. In _2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)_ , pp. 1–8, 2023a. doi: 10.1109/ASRU57964.2023.10389742. 

- Yuan Gong, Alexander H Liu, Hongyin Luo, Leonid Karlinsky, and James Glass. Joint audio and speech understanding. In _2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)_ , pp. 1–8. IEEE, 2023b. 

- Yuan Gong, Hongyin Luo, Alexander H Liu, Leonid Karlinsky, and James Glass. Listen, think, and understand. _arXiv preprint arXiv:2305.10790_ , 2023c. 

- Andrey Guzhov, Federico Raue, J¨orn Hees, and Andreas Dengel. Audioclip: Extending clip to image, text and audio. In _ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 976–980. IEEE, 2022. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_ , 2020. 

- Shawn Hershey, Daniel PW Ellis, Eduardo Fonseca, Aren Jansen, Caroline Liu, R Channing Moore, and Manoj Plakal. The benefit of temporally-strong labels in audio event classification. In _ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 366–370. IEEE, 2021. 

- Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He, Yu Qiao, and Ping Luo. Omnimedvqa: A new large-scale comprehensive evaluation benchmark for medical lvlm. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 22170–22183, 2024. 

- Atin Sakkeer Hussain, Shansong Liu, Chenshuo Sun, and Ying Shan. M ˆ _{_ 2 _}_ ugen: Multi-modal music understanding and generation with the power of large language models. _arXiv preprint arXiv:2311.11255_ , 2023. 

- Jaeyeon Kim, Jaeyoon Jung, Jinjoo Lee, and Sang Hoon Woo. Enclap: Combining neural audio codec and audio-text joint embedding for automated audio captioning. In _ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 6735– 6739. IEEE, 2024. 

- Zhifeng Kong, Arushi Goel, Rohan Badlani, Wei Ping, Rafael Valle, and Bryan Catanzaro. Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities. _arXiv preprint arXiv:2402.01831_ , 2024. 

- Ehsan Latif, Gengchen Mai, Matthew Nyaaba, Xuansheng Wu, Ninghao Liu, Guoyu Lu, Sheng Li, Tianming Liu, and Xiaoming Zhai. Artificial general intelligence (agi) for education. _arXiv preprint arXiv:2304.12479_ , 1, 2023. 

12 

Pre-print. Under Review. 

- Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models. _arXiv preprint arXiv:2403.00231_ , 2024. 

- Yizhi Li, Ruibin Yuan, Ge Zhang, Yinghao Ma, Xingran Chen, Hanzhi Yin, Chenghao Xiao, Chenghua Lin, Anton Ragni, Emmanouil Benetos, et al. Mert: Acoustic music understanding model with large-scale self-supervised training. _arXiv preprint arXiv:2306.00107_ , 2023. 

- Richard P. Lippmann. Speech recognition by machines and humans. _Speech Communication_ , 22(1):1–15, 1997. ISSN 0167-6393. doi: https://doi.org/10.1016/S0167-6393(97) 00021-6. URL https://www.sciencedirect.com/science/article/pii/ S0167639397000216. 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023a. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023b. 

- Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024a. URL https:// llava-vl.github.io/blog/2024-01-30-llava-next/. 

- Shansong Liu, Atin Sakkeer Hussain, Chenshuo Sun, and Ying Shan. Music understanding llama: Advancing text-to-music generation with question answering and captioning. In _ICASSP 20242024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 286–290. IEEE, 2024b. 

- Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. _arXiv preprint arXiv:2310.02255_ , 2023. 

- Ilaria Manco, Benno Weck, Seungheon Doh, Minz Won, Yixiao Zhang, Dmitry Bodganov, Yusong Wu, Ke Chen, Philip Tovstogan, Emmanouil Benetos, et al. The song describer dataset: a corpus of audio captions for music-and-language evaluation. _arXiv preprint arXiv:2311.10057_ , 2023. 

- Jan Melechovsky, Zixun Guo, Deepanway Ghosal, Navonil Majumder, Dorien Herremans, and Soujanya Poria. Mustango: Toward controllable text-to-music generation. _arXiv preprint arXiv:2311.08355_ , 2023. 

- Meredith Ringel Morris, Jascha Sohl-Dickstein, Noah Fiedel, Tris Warkentin, Allan Dafoe, Aleksandra Faust, Clement Farabet, and Shane Legg. Position: Levels of AGI for operationalizing progress on the path to AGI. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _Proceedings of Machine Learning Research_ , pp. 36308–36321. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr. press/v235/morris24b.html. 

- Arsha Nagrani, Joon Son Chung, and Andrew Zisserman. Voxceleb: a large-scale speaker identification dataset. _arXiv preprint arXiv:1706.08612_ , 2017. 

- Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language models. _arXiv preprint arXiv:2311.16103_ , 2023. 

- OpenAI, Josh Achiam, and Others. Gpt-4 technical report, 2024. URL https://arxiv.org/ abs/2303.08774. 

- Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. Meld: A multimodal multi-party dataset for emotion recognition in conversations. _arXiv preprint arXiv:1810.02508_ , 2018. 

- Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In _International conference on machine learning_ , pp. 28492–28518. PMLR, 2023. 

13 

Pre-print. Under Review. 

- Zafar Raf , Antoine Liutkus, Fabian-Robert St¨oter, Stylianos Ioannis Mimilakis, and Rachel Bittner. The musdb18 corpus for music separation. 2017. 

- Inioluwa Deborah Raji, Emily M Bender, Amandalynne Paullada, Emily Denton, and Alex Hanna. Ai and the everything in the whole wide world benchmark. _arXiv preprint arXiv:2111.15366_ , 2021. 

- Paul K Rubenstein, Chulayuth Asawaroengchai, Duc Dung Nguyen, Ankur Bapna, Zal´an Borsos, F´elix de Chaumont Quitry, Peter Chen, Dalia El Badawy, Wei Han, Eugene Kharitonov, et al. Audiopalm: A large language model that can speak and listen. _arXiv preprint arXiv:2306.12925_ , 2023. 

- Ashish Seth, Mayur Hemani, and Chirag Agarwal. Dear: Debiasing vision-language models with additive residuals. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 6820–6829, 2023. 

- Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. Salmonn: Towards generic hearing abilities for large language models. _arXiv preprint arXiv:2310.13289_ , 2023. 

- Gemini Team, Petko Georgiev, and Others. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. URL https://arxiv.org/abs/2403.05530. 

- H Touvron, T Lavril, G Izacard, X Martinet, MA Lachaux, T Lacroix, B Rozi`ere, N Goyal, E Hambro, F Azhar, et al. Open and efficient foundation language models. _Preprint at arXiv. https://doi. org/10.48550/arXiv_ , 2302, 2023a. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023b. URL https://arxiv.org/abs/2302.13971. 

- Harsh Trivedi, Heeyoung Kwon, Tushar Khot, Ashish Sabharwal, and Niranjan Balasubramanian. Repurposing entailment for multi-hop question answering tasks. _arXiv preprint arXiv:1904.09380_ , 2019. 

- Alex Wang. Glue: A multi-task benchmark and analysis platform for natural language understanding. _arXiv preprint arXiv:1804.07461_ , 2018. 

- Bin Wang, Xunlong Zou, Geyu Lin, Shuo Sun, Zhuohan Liu, Wenyu Zhang, Zhengyuan Liu, AiTi Aw, and Nancy F. Chen. Audiobench: A universal benchmark for audio large language models, 2024a. URL https://arxiv.org/abs/2406.16020. 

- Zihao Wang, Shuyu Li, Tao Zhang, Qi Wang, Pengfei Yu, Jinyang Luo, Yan Liu, Ming Xi, and Kejun Zhang. Muchin: A chinese colloquial description benchmark for evaluating language models in the field of music. _arXiv preprint arXiv:2402.09871_ , 2024b. 

- Benno Weck, Ilaria Manco, Emmanouil Benetos, Elio Quinton, George Fazekas, and Dmitry Bogdanov. Muchomusic: Evaluating music understanding in multimodal audio-language models. _arXiv preprint arXiv:2408.01337_ , 2024. 

- Minz Won, Yun-Ning Hung, and Duc Le. A foundation model for music informatics. In _ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 1226–1230. IEEE, 2024. 

- Yusong Wu*, Ke Chen*, Tianyu Zhang*, Yuchen Hui*, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keywordto-caption augmentation. In _IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP_ , 2023. 

- Qingyang Xi, Rachel M Bittner, Johan Pauwels, Xuzhou Ye, and Juan Pablo Bello. Guitarset: A dataset for guitar transcription. In _ISMIR_ , pp. 453–460, 2018. 

14 

Pre-print. Under Review. 

Blaise Ag¨uera y Arcas and Peter Norvig. Artificial general intelligence is already here. _Noema, October_ , 2023. 

- Qian Yang, Jin Xu, Wenrui Liu, Yunfei Chu, Ziyue Jiang, Xiaohuan Zhou, Yichong Leng, Yuanjun Lv, Zhou Zhao, Chang Zhou, and Jingren Zhou. Air-bench: Benchmarking large audiolanguage models via generative comprehension, 2024. URL https://arxiv.org/abs/ 2402.07729. 

- Jiaxuan You, Ge Liu, Yunzhu Li, Song Han, and Dawn Song. How far are we from agi. In _ICLR 2024 Workshops_ , 2024. 

- Ruibin Yuan, Yinghao Ma, Yizhi Li, Ge Zhang, Xingran Chen, Hanzhi Yin, Yiqi Liu, Jiawen Huang, Zeyue Tian, Binyue Deng, et al. Marble: Music audio representation benchmark for universal evaluation. _Advances in Neural Information Processing Systems_ , 36:39626–39647, 2023. 

- Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9556–9567, 2024. 

- Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. _arXiv preprint arXiv:2306.02858_ , 2023a. URL https:// arxiv.org/abs/2306.02858. 

- Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? _arXiv preprint arXiv:2403.14624_ , 2024. 

- Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, and Minlie Huang. Safetybench: Evaluating the safety of large language models with multiple choice questions. _arXiv preprint arXiv:2309.07045_ , 2023b. 

- Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/ forum?id=1tZbq88f27. 

## A APPENDIX 

## **Table of Contents:** 

1. B Additional Results 

2. C Annotation Details 

3. D Model Details 

4. E Dataset Details 

5. F Annotation Tool 

6. G Comparison 

7. H Question Categories 

8. I Failure Cases 

## B ADDITIONAL RESULTS 

## B.1 AUDIO-LANGUAGE ENCODERS (ALES) 

**ALEs** To asses how CLAP-like Audio-Language Encoders (ALEs) perform on MMAU as shown in Table 4, we evaluate several open-source ALEs, including (i) CLAP, a fully open-source model designed primarily for sound and music comprehension. We tested different variants of CLAP, such 

15 

Pre-print. Under Review. 

|**Models**<br>**Size**<br>**Sound**<br>**Music**<br>**Speech**<br>**Avg**|**Models**<br>**Size**<br>**Sound**<br>**Music**<br>**Speech**<br>**Avg**|**Models**<br>**Size**<br>**Sound**<br>**Music**<br>**Speech**<br>**Avg**|**Models**<br>**Size**<br>**Sound**<br>**Music**<br>**Speech**<br>**Avg**|
|---|---|---|---|
|||||
|CompA-CLAP<br>ReCLAP<br>LAION-CLAP<br>MS CLAP 2023|416M<br>416M<br>416M<br>159M|42.66<br>38.20<br>27.98<br>47.43<br>34.83<br>**29.51**<br>45.10<br>35.19<br>25.61<br>**52.10**<br>**40.00**<br>28.78|36.28<br>37.26<br>35.30<br>**40.29**|



Table 4: Performance comparison of ALEs on MMAU benchmark. 

as LAION-CLAP (Wu* et al., 2023) and MS-CLAP (Elizalde et al., 2023). (ii) ReCLAP Ghosh et al. (2024b), an open-source model enhanced with prompt augmentations for robust sound understanding. (iii) CompA-CLAP Ghosh et al. (2023), a model that excels in performing compositional reasoning with sound. 

**Evaluation Strategy** To evaluate ALE on MMAU, we adopt methods similar to those used for assessing question-response performance in entailment models (Deshmukh et al., 2024; Trivedi et al., 2019). First, we convert each question-choice pair into a hypothesis using GPT-4o (details in Appendix L). We then encode the audio and hypotheses with ALE and select the best hypothesis based on the cosine similarity between the audio and hypothesis embeddings. Finally, we use microaccuracy to measure the performance across all data points. 

**Results** Despite their encoder-only architecture, ALEs perform well in our evaluation setup, which is tailored for them. This is similar to findings in (Deshmukh et al., 2024), where authors find ALEs to perform better than LALMs in deductive reasoning. However, we discuss next that ALEs benefit from acting as bag-of-words models in our evaluation scheme (and possibly in Deshmukh et al. (2024) too). Future work could refine the evaluation process to better differentiate ALEs from LALMs. 

**Result Analysis** While ALEs outperform LALMs in deductive reasoning, their advantage stems from the bag-of-words nature of these models. To demonstrate this, we conduct a qualitative analysis of responses generated by MS CLAP, shown in Fig. 8. Similar to (Ghosh et al., 2023), our findings reveal that these models struggle significantly when presented with counter-options containing the exact words in a different order, highlighting their lack of compositional reasoning. Future research should focus on improving the quality of options to assess the reasoning capabilities of ALEs better. 

## B.2 EVALUATING ALES AND LALMS ACROSS VARYING DIFFICULTY LEVELS 

Table 5 provides the performance of ALEs and LALMs across different difficulty levels of MMAU. The models exhibit slightly better performance on medium tasks, with a noticeable drop in performance for hard tasks. This trend suggests that while ALEs and LALMs are capable of handling moderately complex challenges, they struggle with more intricate tasks, indicating potential limitations in reasoning or understanding complex audio cues as task difficulty increases. 

|**Easy**<br>**Medium**<br>**Hard**|**Easy**<br>**Medium**<br>**Hard**|
|---|---|
|**Models**|(2482)<br><br>(5312)<br><br>(2206)|
|LAION-CLAP<br>SALMONN<br>GAMA<br>Qwen2<br>Gemini Pro v1.5<br>Average|38.72<br>36.97<br>27.60<br>20.31<br>39.33<br>30.63<br>31.36<br>35.70<br>22.85<br>50.59<br>55.63<br>46.99<br>57.04<br>51.49<br>52.07<br>39.60<br>43.82<br>36.03|



Table 5: Comparison of ALEs and LALMs Performance Across Multiple Difficulty Levels 

16 

Pre-print. Under Review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark_images/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark.pdf-0017-01.png)


**----- Start of picture text -----**<br>
Skills Questions<br> Based on the given audio, what is most likely happening in<br> this scene?<br> A. It is most likely that a person is hitting various bells with a<br> rod in the scene depicted in the given audio.<br> B. It is most likely that a rod is hitting various bells with a person<br>Acoustic Scene Reasoning<br> in the scene depicted in the given audio.<br> C. It is most likely that a person is hitting various rod with a bell<br> in the scene depicted in the given audio.<br> D. It is most likely that a bell is hitting various person with a rod<br> in the scene depicted in the given audio.<br> Based on the given audio, what events are most likely occur-<br> ring?<br> A. Based on the given audio, it is most likely that a horse is moo-<br> ing and a cow is galloping.<br> B. Based on the given audio, it is most likely that a cat is mooing<br>Acoustic Scene Reasoning<br> and a dog is galloping.<br> C. Based on the given audio, it is most likely that a horse is gal-<br> loping and a cow is mooing.<br> D. Based on the given audio, it is most likely that a horse and<br> cow are galloping.<br> Given the audio sample, what might have caused the bird to<br> chirp?<br> A. It might have been the birds speaking nearby that caused the<br> person to chirp.<br>Event-Based Sound Reasoning  B. It might have been the person speaking nearby that caused the<br> birds to chirp.<br> C. The continuous rustling sounds in the audio sample could<br> have caused the bird to chirp.<br> D. A sudden hissing noise might have caused the bird to chirp.<br> Based on the given audio, what is likely happening?<br> A. It is likely that a wood is cutting a saw based on the given<br> audio.<br>Acoustic Scene Reasoning  B. It is likely that a saw is cutting a wood based on the given<br> audio.<br> C. It is likely that the animals are making noise.<br> D. It is likely that people are conversing.<br>**----- End of picture text -----**<br>


Figure 8: Qualitative analysis of the options selected by MS-CLAP. Correct results are highlighted in green, while predicted results are shown in red. MS CLAP behaves like a bag-of-words model when selecting the correct options. 

## C ANNOTATION DETAILS 

## C.1 ANNOTATION 

Figure 9, shows snapshot of the tool used to annotate audio-question pairs and verify the answers. First, 3 expert annotators from each domain - sound, speech and music annotate and verify each answers for each audio-question pair as curated in the previous step. Once the annotations are done, these experts filter the most plausible samples from the annotated samples. During the annotation phase, the experts annotated _≈_ 11000 pairs of audio and question, out of which _≈_ 800 were discarded during filtering. During the Expert Review stage, the experts from each domain reviewed the question-answer pair for each audio, and disregarded _≈_ 200 samples which either had misleading or very co-related options after the option augmentation stage or had incorrect answers. The experts went through the benchmark twice during the annotation & filtering stage to avoid any form of discrepancy. 

17 

Pre-print. Under Review. 

## C.2 ANNOTATOR DETAILS 

Two sets of experts, 3 each were separately involved during Expert Annotation & Filtering and Expert Review. Each domain, i.e sound, speech and music had 1 expert for each Annotation & Filtering and the Review stage. The experts included 4 males and 2 females. The experts involved in the Expert Annotation stage are MS/PhD students with strong foundational understanding of their respective domains. The experts involved during the Expert Review stage were PhD students and industry practitioners. Their expertise was verified by their published research work and contribution the domain. These experts brought with them a wealth of domain expertise and research experience. They have a profound understanding of sound analysis and excel at discerning intricate details in audio recordings. Their expertise is both technical and theoretical, enabling them to approach the annotation process with nuanced insight. This background allows them to handle complex audio data with precision, ensuring that the annotations are accurate and meaningful. Their combined experience in audio research is a valuable asset to our project, significantly enhancing the depth and reliability of our annotated audio corpus. 

## C.3 ANNOTATION GUIDELINES 

During annotation, the following guidelines were shared with the annotators: 

1. Annotations must be accurate, consistent, and adhere to a high standard of academic rigor. 

2. Listen to the complete audio before annotating the question-answer pair. 

3. All questions must contain one audio, and the audio should not be corrupt. 

4. All questions should be in the English language. 

5. All questions must be tagged with a ‘task’ type as defined. 

6. All the questions must be tagged with a ‘difficulty’ level. 

7. All questions must have a ‘dataset‘ tag, which implies which dataset the audio actually comes from. 

8. The answers to all the questions must be MCQ, and other types of question-answer pairs must be discarded. 

9. The questions should not mention the name of the audio or any information about the audio being used. 

## C.4 HUMAN EVALUATION 

We recruit 8 university students for human evaluation study. Each participant was provided with detailed instructions and asked to carefully listen to the audio samples before answering the corresponding questions. This evaluation was designed to assess the accuracy and reliability of the benchmark, ensuring the human-level performance for comparison with the models’ outputs. The results from the human evaluators served as a baseline for assessing the models’ effectiveness on the task. This evaluation was performed on _test-mini_ part of MMAU. 

## D MODEL DETAILS 

**Audio Flamingo.** Kong et al. (2024) is an audio language model that supports in-context learning (ICL), retrieval augmented generation (RAG), and multi-turn dialogues. It has shown state-of-theart results on a variety of open-ended and close-ended audio understanding and few-shot learning tasks. 

**Qwen-Audio.** Chu et al. (2023) is a large-scale audio language model supporting diverse audio types, languages, and tasks. It achieves state-of-the-art performance across various benchmarks, showing its universal audio understanding capabilities. Qwen-Audio also leverages its ability by supporting multilingual, multi-turn dialogues with flexible input from both audio and text through Qwen-Audio-Chat. 

**Qwen2-Audio.** Chu et al. (2024) is a Large Audio-Language Model (LALM) built on QwenAudio, designed to process both audio and text inputs to generate textual outputs. Qwen2-Audio 

18 

Pre-print. Under Review. 

shows state-of-the-art performance in instruction-following capabilities across speech, sound music and mixed-Audio subsets, demonstrating its proficiency in audio understanding and dialogue capabilities. 

**LTU.** Gong et al. (2023c) is a multi-modal large language model focusing on general audio understanding, including reasoning and comprehension abilities. LTU is trained on a set of closed-ended and open-ended questions with a perception-to-understand training approach. LTU demonstrates strong performance and generalization ability on conventional audio tasks such as classification and captioning. 

**LTU-AS.** Gong et al. (2023a) proposes a joint audio and speech model. It uses whisper as the audio encoder and Llama as the reasoning model, combining strong perception and reasoning abilities, showing competitive performance on all tested closed-ended audio and speech benchmarks, particularly on tasks requiring joint audio and speech understanding. 

**SALMONN.** Tang et al. (2023) is a multimodal large language model designed to perceive and understand speech, audio events, and music, showing a significant step toward achieving generalized auditory capabilities for LLMs. It excels in tasks such as speech recognition, audio captioning, and speech translation while generalizing to tasks like slot filling, keyword extraction, and speech translation for a variety of languages. It also exhibits remarkable emergent abilities, including audiobased storytelling and speech-audio co-reasoning. 

**Pengi.** Deshmukh et al. (2023) was one of the first efforts to achieve general-purpose audio understanding through free-form language generation with transfer learning. It excels at several closeended and open-ended audio tasks. It leverages transfer learning by framing all audio tasks as text-generation problems. Pengi shows state-of-the-art performance across 21 downstream tasks in various audio domains, demonstrating the capability of a general-purpose audio language model. 

**MusiLingo.** Deng et al. (2023)is a music language model designed for music question-answering and captioning. MusiLingo’s framework includes a single projection layer, which aligns music representations with textual contexts, resulting in a competitive performance for a variety of music question-answering tasks and music captioning. 

**MU-LLaMa.** Liu et al. (2024b) is a music language model for music question-answering and captioning. It generates captions by answering music-related questions for the given music and demonstrates exceptional generalization capabilities, making it highly effective across various musicrelated tasks. It exhibits superior performance in both music question-answering and music captioning tasks, surpassing the current state-of-the-art models. 

**M2UGen.** Hussain et al. (2023) is a music language model focusing on music understanding and multi-modal music generation tasks, multi-modal music generation and music editing. M2UGen shows state-of-the-art results on various tasks, including music understanding, music editing, and text/image/video-to-music generation. 

**GAMA.** Ghosh et al. (2024c) is a large audio language model with advanced audio understanding and complex reasoning abilities. By integrating an LLM with various audio representations, It delivers a comprehensive understanding of input audio. It demonstrates state-of-the-art performance on 16 datasets spanning 4 tasks, significantly surpassing previous audio-language models on standard audio and music understanding. 

**MS CLAP.** Elizalde et al. (2023) is an audio language model trained with contrastive learning between audio data and their corresponding natural language descriptions. It extracts representations from both audio and text encoders. 

**CompA-CLAP.** Ghosh et al. (2023) is an extension of CLAP that is trained exclusively on opensource datasets. It is further fine-tuned with specialized algorithms and datasets to enhance compositional reasoning capabilities. 

**LAION-CLAP.** Wu* et al. (2023) proposes a large-scale contrastive language-audio pretraining model that leverages a newly introduced dataset called LAION-Audio-630K, which includes over 630k audio-text pairs. The model combines audio and text encoders with feature fusion and keyword-to-caption augmentation, improving performance on text-to-audio retrieval, zero-shot audio classification, and supervised audio classification tasks. 

19 

Pre-print. Under Review. 

**ReCLAP.** Ghosh et al. (2024b) builds on the work of LAION-CLAP, and introduces an enhanced CLAP model trained with rewritten audio captions to improve zero-shot audio classification (ZSAC) and retrieval tasks. The ReCLAP model is trained on _≈_ 2.3M audio-caption pairs. 

## E DATASET DETAILS 

Table 6 presents the frequency distribution of synthetic and real data, along with the sources from which the real data is pooled. 

**AudioSet.** Gemmeke et al. (2017) Audioset is a large-scale audio event dataset comprising over 2 million human-annotated 10-second video clips. The dataset is labeled using a hierarchical ontology of 632 event classes, allowing the same sound to be tagged with different labels. 

**AudioSet Strong.** Hershey et al. (2021) The AudioSet Strong dataset is an extension of the original AudioSet, containing 67,000 clips with strong labels (precise, 0.1 sec annotations) from a subset of the original 1.8 million weakly-labeled clips. It spans 356 sound classes with detailed start and end times for events, providing over 200 hours of audio. This dataset is used to improve audio event classification and evaluate classifiers with both positive and challenging negative labels. 

**MUStARD.** Castro et al. (2019) MUStARD is a multi-modal video corpus for research in automated sarcasm discovery. MUStARD is curated from popular TV shows such as Friends, The Golden Girls,The Big Bang Theory, and Sarcasmaholics Anonymous. MUStARD comprises 690 videos with an even number of sarcastic and non-sarcastic labels. 

**MELD.** Poria et al. (2018) The Multimodal EmotionLines Dataset (MELD) is a multimodal dataset designed for emotion recognition in conversations. It contains around 13,000 utterances derived from 1,433 dialogues from the TV series Friends. These dialogues include audio, visual, and textual components. Each utterance is annotated with emotion and sentiment labels. 

**VoxCeleb.** Nagrani et al. (2017) The VoxCeleb dataset is a large-scale speaker identification corpus containing over 100,000 utterances from 1,251 celebrities. The dataset is used for both speaker identification and speaker verification with noisy, unconstrained speech, making it useful for realworld speaker recognition tasks. 

**IEMOCAP.** Busso et al. (2008) The IEMOCAP dataset is used for emotion recognition, consisting of 302 videos of dialogues recorded across 5 sessions with 5 pairs of speakers. It includes 9 emotion labels: angry, excited, fear, sad, surprised, frustrated, happy, disappointed, and neutral, as well as valence, arousal, and dominance annotations. 

**MusicCaps.** Agostinelli et al. (2023) MusicCaps is a music caption dataset consisting of 5.5k music clips from AudioSet by focusing exclusively on music content, each paired with text descriptions written by ten professional musicians. For every 10-second clip, it provides a free-text caption (four sentences on average) and a list of music aspects like genre, mood, tempo, and instrumentation. The dataset includes around eleven aspects per clip and a genre-balanced split with 1k examples. 

**MusicBench.** Melechovsky et al. (2023) MusicBench is a dataset for text-to-music generation, expanding the original MusicCaps dataset from 5,521 to 52,768 training samples and 400 test samples. It enhances the dataset by adding music features such as chords, beats, tempo, and key, described via text templates, and by applying augmentations such as pitch shifts, tempo, and volume changes. 

**MTG-Jamendo.** Bogdanov et al. (2019) The MTG-Jamendo Dataset is a dataset for automatic music tagging, featuring over 55,000 full audio tracks, each annotated with 195 tags spanning genres, instruments, and moods/themes. The dataset includes 3,565 artists with 3,777 hours of audio in high-quality 320 kbps MP3 format. It includes five predefined splits for training, validation, and testing, with no overlap of tracks from the same artist across sets. 

**SDD.** Manco et al. (2023) The Song Describer Dataset (SDD) is used as an evaluation tool for music-and-language models, enabling benchmarking tasks such as music captioning and text-tomusic retrieval. It contains 1,106 human-written captions for 706 music recordings collected from 142 annotators. The dataset features audio-caption pairs with descriptions focused on various musical elements like genre, mood, and instrumentation. 

20 

Pre-print. Under Review. 

|**Dataset**|**# Audios**|
|---|---|
|Audioset|2788|
|AudioSet Strong|391|
|Mustard|405|
|MELD|540|
|VoxCeleb-1|633|
|IEMOCAP|515|
|MusicBench|1937|
|Jamendo|32|
|SDD|277|
|MusicCaps|514|
|GuitarSet|506|
|MUSDB18|68|
|Synthetic|1394|



Table 6: List of sources from where MMAU is pooled. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark_images/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark.pdf-0021-03.png)


Figure 9: Snapshot of the annotation tool used by the annotators to annotate the correct answers for each audio-question pair. 

**GuitarSet.** Xi et al. (2018) The GuitarSet dataset contains 3 hours of guitar recordings from 6 experienced guitarists, each performing 30 excerpts of various musical genres, including Rock, Jazz, Funk, Bossa Nova, and Singer-Songwriter. It provides rich annotations like tempo, key, chords, beats, and note-level transcriptions. The dataset includes time-aligned data on string/fret positions, chords, and playing style, offering valuable resources for tasks such as guitar transcription, performance analysis, beat tracking, and chord estimation. 

**MUSDB18.** Raf et al. (2017) The MUSDB18 dataset is widely used for music source separation tasks. The dataset consists of 150 full-track songs across various styles. It includes 100 songs in the training set and 50 songs in the test set, with each track split into 5 stereo streams: mixture, drums, bass, accompaniment, and vocals. 

21 

Pre-print. Under Review. 

|**Category**<br>**Prior Benchmarks**<br>**MMAU**|**Category**<br>**Prior Benchmarks**<br>**MMAU**|**Category**<br>**Prior Benchmarks**<br>**MMAU**|
|---|---|---|
|**Sound**|**Task:** Simple event identifcation<br>**Example:** ”What’s the provenance of the sound?”<br>**Diffculty:** Easy<br>**Dataset:** AirBench|**Task:** Ambient Sound Understanding<br>**Example:** ”What material is typically used for the strings of the instrument?”<br>**Diffculty:** Hard<br>**Dataset:** MMAU|
|**Speech**|**Task:** Speaker identifcation, emotion detection<br>**Example:** ”What emotion is at the forefront of the speaker’s words?”<br>**Diffculty:** Easy<br>**Dataset:** AirBench|**Task:** Conversational Content Analysis<br>**Example:** ”Who was the surgeon responsible for the event mentioned?”<br>**Diffculty:** Hard<br>**Dataset:** MMAU|
|**Music**|**Task:** Genre identifcation, MIDI pitch detection<br>**Example:** ”What’s the genre of this music?”<br>**Diffculty:** Easy<br>**Dataset:** AirBench|**Task:** Instrument identifcation, vocal characteristics analysis<br>**Example:** ”Which instrument is playing the high notes?”<br>**Diffculty:** Medium<br>**Dataset:** MMAU|



Table 7: Comparison of MMAU vs Prior Audio Benchmark 

## F ANNOTATION TOOL 

Figure 9 shows the snapshot of the tool used by the annotators. Annotators were shown the audio, questions, options, and answers. The annotators were asked to listen to the audio and annotate if the answer shown was correct and in the option. The annotators had the option to either accept or reject the question-answer pair for the given audio. 

## G COMPARISON 

Table 7 highlights the differences between MMAU and previous benchmarks, particularly in terms of the increased difficulty and required complex reasoning ability that MMAU’s questions present to the models. 

22 

Pre-print. Under Review. 

## H ADDITIONAL INFORMATION ON SKILLS 

The table below highlights the various skill challenges presented by the MMAU benchmark to the LALMs. 

|**Domain**|**Skills**|**Tasks**|**Question (with option)**|
|---|---|---|---|
|**Sound**|Temporal<br>Event<br>Reasoning|Identify ordering and<br>duration of various<br>sounds|Identify the total number of drum beats in<br>the audio. Choices:<br>A. 2<br>B. 4<br>C. 5<br>D. 3|
||Acoustic-Source<br>Inference|Identify the source of<br>various sounds|For the given audio sample, identify the<br>source of the singing sound.<br>Choices:<br>A. People<br>B. Birds<br>C. Musical Instrument<br>D. Radio|
||Eco-Acoustic<br>Knowledge|Identify the environ-<br>mental<br>background<br>based<br>on<br>various<br>sounds|Based on the audio, what is the likely set-<br>ting?<br>Choices:<br>A. Beach<br>B. Mountain<br>C. City Park<br>D. Forest|
||Ambient<br>Sound<br>Interpretation|Extracting<br>informa-<br>tion about the back-<br>ground sound|Name a famous musician known for play-<br>ing the instrument heard in the back-<br>ground.<br>Choices:<br>A. Yo-Yo Ma<br>B. Jimi Hendrix<br>C. Miles Davis<br>D. Flea|
||Acoustic<br>Scene<br>Reasoning|Answer the reason-<br>ing questions based<br>on the acoustic scene<br>interpreted<br>from<br>multiple sounds.|Based on the given audio, what event is<br>taking place?<br>Choices:<br>A. A person is playing percussive instru-<br>ments simultaneously.<br>B. Hard objects are being manipulated in<br>various ways.<br>C. Someone is rolling and striking hard ob-<br>jects.<br>D. A person is handling items and closing<br>a container.|
||Event-Based<br>Sound Reasoning|Causal<br>reasoning<br>question about what<br>triggered a source to<br>produce<br>a<br>specifc<br>sound.|Based on the given audio, what could have<br>caused the dog’s barking?<br>Choices:<br>A. A person approaching the dog.<br>B. A cat approaching the dog.<br>C. A laughter heard nearby<br>D. Agentle splash of water.|
|||||



23 

Pre-print. Under Review. 

||Sound-Based<br>Event<br>Recogni-<br>tion|Based<br>on<br>multiple<br>sound, infer the most<br>likely event from the<br>audio|What type of emergency vehicle is indi-<br>cated by the sirens in the audio?<br>Choices:<br>A. Fire truck.<br>B. Ambulance.<br>C. Police car<br>D. Garbage truck.|
|---|---|---|---|
|**Speech**|Dissonant<br>Emo-<br>tion<br>Interpreta-<br>tion|Identify<br>sarcasm<br>in<br>multi-speaker<br>settings|From the given conversation, What makes<br>the last comment sarcastic in relation to the<br>dialogue?<br>Choices:<br>A. Criticism of scientifc method<br>B. Genuine admiration of intelligence.<br>C. Requesting further explanation<br>D. Mockingexaggeratedpraise|
||Event-Based<br>Knowledge<br>Retrieval|Extract<br>information<br>about<br>the<br>event<br>discussed<br>in<br>a<br>conversation.|Who was the scientist behind the discovery<br>mentioned by the speaker?<br>Choices:<br>A. Marie Curie<br>B. Albert Einstein<br>C. Alexander Fleming<br>D. Isaac Newton|
||Counting|Count the number of<br>speakers in a dia-<br>logue|What’s the number of speakers in the cur-<br>rent conversation?<br>Choices:<br>A. 3<br>B. 4<br>C. 2<br>D. 1|
||Phonemic Stress<br>Pattern Analysis|Identify<br>the<br>stress<br>patterns of phonemes<br>in an utterance.|From the given utterance, identify a pair<br>of words that contain similar sounding<br>stressed and unstressed phonemes<br>Choices:<br>A. Sometimes, want<br>B. hair,directing<br>C. frst, second<br>D. few,blanks|
||Emotional<br>State<br>summarisation|Identify the emotions<br>of all the speakers in<br>a conversation|From the given conversation, Identify the<br>emotion of each speaker<br>Choices:<br>A. frst speaker shows neutral, anger; sec-<br>ond speaker shows fear, neutral, disgust.<br>B. frst speaker shows neutral, anger; sec-<br>ond speaker seems neutral.<br>C. frst speaker shows happiness; second<br>speaker shows fear.<br>D. frst speaker shows fear; second shows<br>disgust|
||Conversational<br>Fact Retrieval|Answer factual ques-<br>tions based on the<br>content discussed by<br>the speakers.|How much money did the second speaker<br>offer the frst speaker to marry her?<br>Choices:<br>A. Twenty thousand dollars<br>B. Seventy thousand dollars<br>C. Fifty thousand dollars<br>D. One hundred thousand dollars|
|||||



24 

Pre-print. Under Review. 

||Multi<br>Speaker<br>Role Mapping|Identify<br>the<br>role<br>played<br>by<br>each<br>speaker in a conver-<br>sation|In the given conversation, identify the role<br>of two speakers.<br>Choices<br>A. frst speaker is a voice coach and the<br>second speaker is singer<br>B. both speakers are neighbors<br>C. frst speaker is a surgeon and the second<br>speaker is surgical nurse<br>D. frst speaker is a nurse and the second<br>speaker is a doctor|
|---|---|---|---|
||Phonological Se-<br>quence Decoding|Identify<br>the<br>word<br>order<br>in<br>similarly<br>sounding<br>words<br>within<br>tongue<br>twisters.|For a given tongue twister, identify which<br>word came frst<br>Choices:<br>A. elves<br>B. elk<br>C. eve<br>D. elite|
||Emotion Flip De-<br>tection|Identify<br>which<br>speakers<br>showed<br>emotion<br>fip<br>in<br>a<br>conversation|From the given conversation, Identify the<br>speakers that showed emotion fip.<br>Choices:<br>A. both speakers<br>B. frst speaker<br>C. second speaker<br>D. none of the speakers|
||Key highlight Ex-<br>traction|Identify the intent of<br>the conversation|What is the main topic of discussion be-<br>tween the speakers<br>Choice:<br>A. negative aspects of environmental pol-<br>lution<br>B. improving one’s relationship with sib-<br>lings.<br>C. challenges of maintaining parent-child<br>relationships<br>D. Impact ofgood communication skills|
|**Music**|Temporal<br>Rea-<br>soning|Extract<br>information<br>about the temporal<br>structure<br>of<br>the<br>music track/song|How does the male voice follow the<br>strummed electric guitar in the audio?<br>Choices:<br>A. It follows immediately after each strum<br>B. It starts before the guitar<br>C. It overlaps with the guitar<br>D. It starts well after the guitar fnishes|
||Musical<br>Genre<br>Reasoning|Understanding musi-<br>cal genre and song<br>type|Considering the mood and elements of the<br>audio, what is the likely purpose of the<br>song?<br>Choices:<br>A. A party anthem<br>B. A workout mix<br>C. A proposal song<br>D. A lullaby|
||Lyrical<br>Reason-<br>ing|Involves<br>analyz-<br>ing song lyrics to<br>interpret<br>themes,<br>emotions, and under-<br>lying meanings.|What day is mentioned in the lyrics?<br>Choices:<br>A. Monday<br>B. Friday<br>C. Sunday<br>D. Wednesday|
|||||



25 

Pre-print. Under Review. 

|Socio-cultural In-<br>terpretation|Analyzing how his-<br>torical<br>events<br>and<br>cultural<br>contexts<br>infuence<br>musical<br>styles,<br>genres,<br>and<br>themes.|In which cultural setting would the music<br>in the audio most likely be performed?<br>Choices:<br>A. Western classical concert hall<br>B. Indian classical music festival<br>C. Modern pop concert<br>D. Jazz club|
|---|---|---|
|Melodic<br>Struc-<br>ture<br>Interpreta-<br>tion|Infer<br>the<br>organiza-<br>tion and progression<br>of melodies to under-<br>stand their patterns,<br>forms, and emotional<br>expressions.|What type of bass line is playing in the au-<br>dio?<br>Choices:<br>A. Acoustic bass line.<br>B. Groovy synth bass line.<br>C. Fretless bass line.<br>D. Double bass line|
|Harmony<br>and<br>Chord<br>Progres-<br>sions|Involve the study of<br>how chords interact<br>and transition to cre-<br>ate musical texture,<br>mood,<br>and<br>overall<br>structure.|What is the chord progression in the audio?<br>Choices:<br>A. C, G, Am, F<br>B. G7, Fm, Ab, Eb, Bb<br>C. Dm, A7, G, Bm<br>D. F, C, Dm, Bb|
|Rhythm<br>and<br>Tempo<br>Under-<br>standing|Focuses on analyzing<br>the timing, beats, and<br>pace of a piece|What is the tempo of the audio?<br>Choices:<br>A. 120 bpm.<br>B. 130 bpm.<br>C. 149 bpm.<br>D. 160 bpm|
|Musical<br>Texture<br>Interpretation|Analyzing the overall<br>vocal quality of the<br>singer.|What is the main characteristic of the male<br>voice in the audio?<br>Choices:<br>A. Soft and mellow<br>B. Loud and soulful<br>C. High-pitched and fast<br>D. Monotone and slow|
|Instrumentation<br>Extracting<br>informa-<br>tion about various in-<br>struments present in<br>a musical piece<br>What is the primary instrument playing in<br>the audio?<br>Choices:<br>A. Violin<br>B. Flute<br>C. Guitar<br>D. Piano|||
|Emotional<br>Tone<br>Interpretation<br>Analyzing the feel-<br>ings<br>conveyed<br>in<br>music to understand<br>the emotional impact<br>and mood of a piece.<br>How would you describe the impact of<br>the simple guitar solo in the bridge on the<br>song’s mood?<br>Choices:<br>A. It introduces a sense of calmness.<br>B. It adds complexity and tension<br>C. It enhances the upbeat and dynamic feel.<br>D. It makes the song sound more melan-<br>cholic.|||



Table 8: Details on categories, type of questions with examples for each task 

26 

Pre-print. Under Review. 

## I FAILURE CASES 

The table below highlights the failure cases of the top-performing LALMs, with examples drawn from the Qwen2-Audio-Instruct model. 

|**Domain**|**Category**|**Question (with options)**|**Answer**|**Model Response**|
|---|---|---|---|---|
|**Sound**|Acoustic-Source<br>Inference|Based on the given audio,<br>identify the source of the<br>music.<br>Choices:<br>A. Fire truck<br>B. Radio<br>C. Airplane<br>D. Construction site|Radio|Construction site|
||Acoustic-Source<br>Inference|Given the audio, identify<br>the source of the mechanism<br>sound.<br>Choices:<br>A. Nature<br>B. Machine<br>C. Human<br>D. Animal|Machine|Human|
||Acoustic<br>Scene<br>Reasoning|Based on the given audio,<br>what event is most likely oc-<br>curring?<br>Choices:<br>A. An alarm clock is ringing<br>intermittently.<br>B. A small handbell is being<br>rung.<br>C. A bell tower is signaling<br>an event.<br>D. A doorbell is being re-<br>peatedly pressed.|A<br>bell<br>tower is<br>signal-<br>ing<br>an<br>event.|An alarm clock is<br>ringing intermit-<br>tently.|
||Acoustic<br>Scene<br>Reasoning|Given<br>the<br>audio,<br>which<br>event is most likely occur-<br>ring?<br>Choices:<br>A. Water drips quickly then<br>slows down.<br>B. A tap is dripping into a<br>basin.<br>C. Rain falls to a patter beat<br>then stops.<br>D. Rain patterns on a metal<br>surface.|Rain<br>patterns<br>on<br>a<br>metal<br>surface.|Water<br>drips<br>quickly<br>then<br>slows down.|
||Ambient<br>Sound<br>Understanding|Identify the instrument play-<br>ing in the background.<br>Choices:<br>A. Guitar<br>B. Flute<br>C. Piano<br>D. Violin|Guitar|Piano|
||||||



27 

Pre-print. Under Review. 

|**Speech**|Event-Based<br>Knowledge<br>Retrieval|Who developed the vaccine<br>mentioned by the speaker?<br>Choices:<br>A. Dr. Jonas Salk<br>B. Dr. Louis Pasteur<br>C. Dr. Albert Sabin<br>D. Dr. Robert Koch|Dr.<br>Jonas<br>Salk|Dr. Albert Sabin|
|---|---|---|---|---|
||Multi-Speaker<br>Identity Profling|How<br>many<br>speakers<br>are<br>present in this conversation?<br>Choices:<br>A. Three<br>B. Four<br>C. Six<br>D. Five|Three|Five|
||Phonemic Stress<br>Pattern Analysis|From the given utterance,<br>count the number of words<br>that contain at least one<br>stressed phoneme.<br>Choices:<br>A. Four<br>B. Nine<br>C. Seventeen<br>D. One|Nine|One<br>(incorrect<br>reasoning)|
||Conversational<br>Fact Retrieval|What is Second Speaker’s<br>frst name according to First<br>Speaker?<br>Choices:<br>A. Jack<br>B. John<br>C. Jones<br>D. James|Jones|John|
||Conversational<br>Fact Retrieval|Who directed First Speaker<br>to get in line?<br>Choices:<br>A. Fourth Speaker<br>B. Third Speaker<br>C. Second Speaker<br>D. First Speaker|Second<br>Speaker|Third Speaker|
|**Music**|Metre<br>and<br>Rhythm|What is the tempo of the au-<br>dio in bpm?<br>Choices:<br>A. 160.0<br>B. 135.0<br>C. 120.0<br>D. 150.0|135.0|150.0|
||Melody|Which instrument is pri-<br>marily responsible for the<br>melody in the audio?<br>Choices:<br>A. Piano<br>B. Violin<br>C. Electric guitar<br>D. Flute|Electric<br>guitar|Piano|
|||28|||



Pre-print. Under Review. 

|Historical|and|Identify the lead instrument|Trumpet|Trumpet|Saxophone|
|---|---|---|---|---|---|
|Cultural Reason-||in the jazz track as described||||
|ing||in the audio.||||
|||Choices:||||
|||A. Piano||||
|||B. Guitar||||
|||C. Trumpet||||
|||D. Saxophone||||
|Emotional|Tone|What kind of emotional re-|Seriousness||Calm and seren-|
|||sponse is the audio most|and|ur-|ity|
|||likely intended to evoke?|gency|||
|||Choices:||||
|||A. Seriousness and urgency||||
|||B. Sadness and contempla-||||
|||tion||||
|||C. Joy and excitement||||
|||D. Calm and serenity||||



Table 9: Model Failures in Sound, Speech, and Music Categories with Sub-Category Information 

## J BENCHMARK EVALUATION 

We asked domain experts to rate each existing benchmark on a scale of 1 to 5 based on the difficulty level of solving the questions. For each benchmark, we randomly selected 1,000 samples (or evaluated the entire benchmark if it contained fewer than 1,000 examples). Domain experts were instructed to listen to the audio and answer the corresponding questions, following a fixed set of guidelines. These guidelines included the breadth of the questions (e.g., variety, question type such as open-ended or multiple-choice), domain coverage (speech, music, sound), and depth of the questions (e.g., whether they required multi-step reasoning or involved different types of reasoning such as content-based, causal, or contextual). 

To ensure unbiased evaluation, the benchmark names were not revealed in advance. Before assigning a difficulty score, each expert was asked to summarize their evaluation in one to two sentences. We aggregated the feedback and difficulty scores from all domain experts and presented our findings in Table 2. 

## K ADDITIONAL DETAILS ON ERROR TYPES 

|**Error Type**|**Defnition**|**Question**|**Prediction**|**Reason**|
|---|---|---|---|---|
|Perceptual Er-<br>ror|The model fails to<br>perceive the audio<br>correctly.|Based on the given au-<br>dio, identify the source<br>of the fowing sound.<br>**Choices**:<br>**A. Stream**<br>B. Faucet<br>C. Waterfall<br>D. Rain|Waterfall|Misinterpreted<br>the sound|



29 

Pre-print. Under Review. 

|**Error Type**|**Defnition**|**Question**|**Prediction**|**Reason**|
|---|---|---|---|---|
|Knowledge<br>Error|The<br>model<br>un-<br>derstands<br>the<br>audio<br>but<br>lacks<br>the knowledge to<br>answer.|What is the typical fre-<br>quency range of the in-<br>strument playing in the<br>background?<br>**Choices**:<br>**A. The bass typically**<br>**ranges from 40 Hz to**<br>**400 Hz.**<br>B. The bass typically<br>ranges from 400 Hz to 4<br>kHz.<br>C. The bass typically<br>ranges from 20 Hz to 200<br>Hz.<br>D. The bass typically<br>ranges from 4 kHz to 40<br>kHz.|20-200 Hz|Lacked<br>specifc<br>frequency<br>knowledge|
|Reasoning Er-<br>ror|The model strug-<br>gles with logical<br>reasoning.|What weather condition<br>is indicated by the au-<br>dio?<br>**Choices**:<br>**A. Windy**<br>B. Calm<br>C. Humid<br>D. Rainy|Humid|Incorrect rea-<br>soning<br>about<br>sound|
|Annotation<br>Error|The<br>model’s<br>re-<br>sponse is correct<br>but the answer key<br>is wrong.|Given the audio sample,<br>what was the primary<br>focus of the audio?<br>**Choices**:<br>**A.**<br>**A**<br>**man**<br>**speaking**<br>**with background mu-**<br>**sic**<br>B.<br>A<br>man<br>breathing<br>heavily<br>C. Only music playing<br>continuously<br>D. A man singing with<br>music|Singing<br>with<br>music|Answer<br>key<br>was incorrect|
|Answer<br>Extraction<br>Error|The<br>model’s<br>an-<br>swer matches but<br>formatting leads to<br>incorrect marking.|Based<br>on<br>the<br>given<br>audio, what could have<br>led to the shout?<br>**Choices**:<br>**A. A whip sound oc-**<br>**curring just before the**<br>**shout**<br>B.<br>Continuous<br>music<br>playing<br>in<br>the<br>back-<br>ground<br>C. Human voice heard<br>earlier in the audio<br>D. Whistling and ap-<br>plause<br>towards<br>the<br>end|Whip<br>sound|Incorrect for-<br>mat in answer|



30 

Pre-print. Under Review. 

|**Error Type**|**Defnition**|**Question**|**Prediction**|**Reason**|
|---|---|---|---|---|
|Other Error|The model refuses<br>to answer or en-<br>counters<br>another<br>issue.|Based<br>on<br>the<br>given<br>audio, what is the most<br>likely<br>source<br>of<br>the<br>noise?<br>**Choices**:<br>**A.**<br>**A**<br>**malfunctioning**<br>**electronic device**<br>B. A gentle breeze<br>C. A calm river stream<br>D. A distant bird chirp-<br>ing|Refused to<br>answer|None of the<br>options ft|



Table 10: Additional details on Error types with some examples from MMAU. The model predictions are taken from Gemini Pro v1.5 

31 

Pre-print. Under Review. 

## L PROMPTS 

## **#Prompt1** 

I want you to generate contrastive options for complex question answers. I will provide you with a question type, question, and a correct answer. Your task is to generate 6 contrastive options and a correct answer for each question. Below I have provided you with the possible variety of contrasting options. 

## 1. Opposites or Near-Opposites 

- Example: If the speaker discusses a positive aspect of a theory, one option may mention the theory's benefits, while another option could suggest drawbacks. 

- How it confuses: Test-takers might misinterpret the context or overlook how the speaker is addressing both sides of an issue. 

## 2. Partial Correctness 

- Example: One option may state part of what the speaker said accurately but omit a crucial detail or add an incorrect one. 

- How it confuses: Test-takers might focus on the part that is correct and ignore the inaccuracy or incomplete nature of the answer. 

## 3. Paraphrasing with a Twist 

- Example: The option might rephrase what the speaker said but introduce a subtle change in meaning (e.g., from "requires" to "recommends"). 

- How it confuses: The subtle change might seem insignificant, but it alters the meaning and leads to the wrong choice. 

## 4. Misleading Similarities 

- Example: Two options may seem very similar, with only a small difference in wording, leading test-takers to choose one over the other. 

- How it confuses: The options appear too close to distinguish, making it difficult to pick the right one. 

## 5. Exaggerated or Minimized Information 

- Example: If the speaker mentions a minor point, one option might exaggerate it (e.g., turning "might affect" into "definitely affects"). 

- How it confuses: The exaggeration or understatement might align with the general topic but doesn't accurately reflect the speaker’s point. 

6. Implied vs. Stated Information 

- Example: One option might correctly infer something from what the speaker said, while another might incorrectly state something explicitly that the speaker never mentioned. 

- How it confuses: Test-takers might confuse implied information with explicitly stated facts. 

## 7. Topic Shift Confusion 

- Example: The speaker may shift from one topic to another, and options might include information from both topics. 

- How it confuses: Test-takers might select an option related to a different part of the conversation or lecture. 

- 

8. Temporal or Sequence Confusion 

- Example: The speaker might describe a sequence of events, but the answer choices could mix up the order or timing. 

- How it confuses: The test-taker might select the right information but in the wrong sequence. 

## 9. Distractors Based on General Knowledge 

- Example: One option might sound correct based on general knowledge but is not supported by the passage. 

- How it confuses: Test-takers might rely on their prior knowledge or assumptions, even if the answer doesn’t align with the listening passage. 

## 10. Options with Extra Information 

- Example: An option might seem correct but adds information that was not mentioned by the speaker. 

- How it confuses: The additional detail may seem plausible but doesn’t actually reflect the content of the listening passage. Note that each contrastive option must not exceed 50 words. The output must be generated in a json format. The template for output json. Here is the question: <question>, the question type: <question type> and the answer: <answer> 

Figure 10: Prompts/Instructions used for generating contrasting options for MMAU. 

32 

Pre-print. Under Review. 

## **#Prompt2** 

Please transcribe the spoken words in the audio clip accurately. Capture all spoken content verbatim, including any significant pauses, emotions, or emphasis expressed by the speaker. Do not include interpretations or descriptions beyond the spoken words. 

## **#Prompt3** 

Please provide a detailed description of the music in the audio clip. Include information about the genre, instruments, tempo, mood, and any notable melodies or harmonies. Describe any vocals present, including lyrics if they are clear and discernible. Mention the overall atmosphere and emotions conveyed by the music. 

## **#Prompt4** 

Please describe all the events and sounds occurring in the audio clip in detail. Identify and describe each sound source, such as objects, animals, weather, or environmental noises. Include information about the sequence of events and any interactions between sound sources. Mention the context or setting if it can be inferred from the sounds. 

Figure 11: Prompts/Instructions used for generating captions using Qwen2-Audio. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark_images/iclr_2025_MMAU_A_Massive_Multi-Task_Audio_Understanding_and_Reasoning_Benchmark.pdf-0033-08.png)


**----- Start of picture text -----**<br>
Task : Given a question and an answer, reformulate them into a single premise statement.<br>Examples :<br>Question : Does the audio contain any melody?<br>Answer : It's hard to tell.<br>Premise : It is difficult to determine whether the audio contains any melody.<br>Question : What instrument plays the melody after the male vocal in the audio?<br>Answer : Piano.<br>Premise : The instrument that plays the melody after the male vocal in the audio is a<br>piano.<br>Question : What instrument plays the melody after the male vocal in the audio?<br>Answer : Trumpet.<br>Premise : The instrument that plays the melody after the male vocal in the audio is a<br>trumpet.<br>Task : Provide the premise for the following question and answer in json format:<br>Question : {question}<br>Answer : {answer}<br>Premise :<br>**----- End of picture text -----**<br>


Figure 12: Prompts/Instructions used for generating hypothesis using question-choice pairs. 

33 

