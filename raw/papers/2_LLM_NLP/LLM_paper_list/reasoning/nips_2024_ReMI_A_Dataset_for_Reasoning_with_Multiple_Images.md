_2024-6-14_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0001-02.png)


## **ReMI: A Dataset for Reasoning with Multiple Images** 

**Mehran Kazemi**[1] **, Nishanth Dikkala**[2] **, Ankit Anand**[1] **, Petar Devic**[3] **, Ishita Dasgupta**[1] **, Fangyu Liu**[1] **, Bahare Fatemi**[2] **, Pranjal Awasthi**[2] **, Dee Guo**[2] **, Sreenivas Gollapudi**[2] **and Ahmed Qureshi**[3] 1Google DeepMind, 2Google Research, 3Google 

**With the continuous advancement of large language models (LLMs), it is essential to create new benchmarks to effectively evaluate their expanding capabilities and identify areas for improvement. This work focuses on multi-image reasoning, an emerging capability in state-of-the-art LLMs. We introduce** ReMI **, a dataset designed to assess LLMs’ ability to Reason with Multiple Images. This dataset encompasses a diverse range of tasks, spanning various reasoning domains such as math, physics, logic, code, table/chart understanding, and spatial and temporal reasoning. It also covers a broad spectrum of characteristics found in multi-image reasoning scenarios. We have benchmarked several cutting-edge LLMs using** ReMI **and found a substantial gap between their performance and human-level proficiency. This highlights the challenges in multi-image reasoning and the need for further research. Our analysis also reveals the strengths and weaknesses of different models, shedding light on the types of reasoning that are currently attainable and areas where future models require improvement. To foster further research in this area, we are open-sourcing** ReMI **:** **`https://huggingface.co/datasets/ mehrankazemi/ReMI` .** 

## **Introduction** 

Large Language Models (LLMs) have demonstrated an extraordinary evolution, not only in their output quality but also in their burgeoning capabilities. A significant direction of development has been models’ ability to perform increasingly general forms of reasoning that were previously not possible. The emergence of these novel capabilities necessitates the development of robust evaluation benchmarks and metrics to measure and enhance model performance in these specific areas. 

The ability of LLMs to reason over text has improved in leaps and bounds, and has been studied extensively (Lewkowycz et al., 2022; Rajani et al., 2019; Wei et al., 2022). More recent developments in multi-modal models has opened up a new space of reasoning problems, moving toward the capability to reason across multiple, potentially disparate, sources of information presented in various formats (Achiam et al., 2023; Anthropic, 2024; Reid et al., 2024; Team et al., 2023). This multi-modal reasoning capability has numerous applications, from complex problemsolving to information synthesis. In this paper, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0001-09.png)


**----- Start of picture text -----**<br>
100<br>80<br>60<br>40<br>20<br>0<br>Naive BaslineGemini UltraClaude3 SonnetGemini FlashGPT4 Turbo Gemini 1.5 Human<br>Performance on ReMI<br>**----- End of picture text -----**<br>


Figure 1 | Model performances on ReMI. 

we focus on a specific aspect of this capability: multi-image reasoning. A large portion of the current benchmarks for multi-modal evaluation are based on a single image (Antol et al., 2015; Fu et al., 2023; Goyal et al., 2017; Kazemi et al., 2023a; Lindström and Abraham, 2022; Liu et al., 2023; Lu et al., 2021, 2022, 2023; Marino et al., 2019). We address the lack of dedicated evaluation frameworks in this domain by introducing a comprehensive benchmark designed to specifically assess and improve this skill in LLMs. We focus specifically on reasoning problems where besides visual understanding, one needs to find 

_Corresponding author(s): mehrankazemi@google.com_ © 2024 Google DeepMind. All rights reserved 

ReMI: A Dataset for Reasoning with Multiple Images 

a step-by-step solution to a problem. This process often involves combining information across text and multiple images – a skill that is currently not extensively evaluated in existing benchmarks. This contribution aims to catalyze progress in multi-image reasoning, ultimately enabling LLMs to better navigate and extract insights from the increasingly complex information landscape of our digital world. 

We introduce ReMI, a new benchmark designed for **Re** asoning with **M** ultiple **I** mages. Our goal is to cover a broad spectrum of domains where integrating information across multiple modalities is necessary, as well as various key properties unique to multi-image reasoning. To achieve this, we have developed 13 tasks that span a range of domains and properties. The domains covered in ReMI include algebra, calculus, geometry, graph theory, physics, temporal and spatial/maps reasoning, tabular and chart understanding, coding, and logic. The properties covered by ReMI include sequential vs set consumption of image information, problems that require reasoning over images demonstrating a similar concept (e.g., two charts) or different concepts (e.g., geometry shape and a table), images that are interleaved or not interleaved with the text, and the number of separate images provided as input. Our tasks require reasoning over up to six images, with all tasks requiring reasoning over at least two images. Table 1 outlines the tasks, domains and properties. Our images comprise a variety of heterogeneous image types including charts, tables, equations, emojis, graphs, shapes, maps, clocks, physical objects, LaTeX diagrams, functions, etc. 

We evaluate state-of-the-art LLMs on ReMI and compare their performance to humans, showing that model performances remain substantially behind human performance (see Fig 1). Interestingly, our results also reveal that models may perform better when multiple images are fed to them separately as opposed to all in one image; this is especially true in the case where the images are interleaved with the question text. A detailed failure analysis reveals model shortcomings that can guide future improvement efforts. 

## **Related Work** 

**Vision-language foundation models.** In our work, we focus on vision language generation models, i.e. models that produce open-ended text conditioned on text and images. Frozen (Tsimpoukelli et al., 2021) and Flamingo (Alayrac et al., 2022) first transformed LLMs into visionlanguage models by adding a vision transformer tower and training cross/self-attention layers to enable LLMs to perceive visual information. Subsequently, a large volume of research emerged focusing on the approach of stitching a pretrained visual encoder (usually vision transformer) to a pretrained langauge model. PaLI (Chen et al., 2022), BLIP (Li et al., 2023b), LLaVA (Liu et al., 2024), OpenFlamingo (Awadalla et al., 2023), PaLIGemma (Beyer* et al., 2024) all follow similar techniques. The latest closed-source frontier models such as GPT-4 (Achiam et al., 2023), Gemini (Team et al., 2023) and Claude 3 Anthropic (2024) all have vision input support and are also reported to be the best performing models across popular vision-language reasoning benchmarks (Lu et al., 2023). These frontier models are able to condition fairly arbitrarily on sequences of interleaved image and text. However, most visionlanguage benchmarks test models’ performance on a single image-text pair; the focus of this paper is to take a step toward evaluating more flexible vision-language abilities. 

**Reasoning Benchmarks.** Reasoning has been a core area of interest for NLP systems. The initial benchmarks focused on ‘simpler’ reasoning tasks which largely involve language understanding (e.g. SuperGLUE (Wang et al., 2019), HellaSwag (Zellers et al., 2019), Lambada (Paperno et al., 2016)). With LLMs making remarkable strides in recent years, a plethora of benchmarks requiring much stronger reasoning abilities have emerged. Some of these like MMLU (Hendrycks et al., 2020) and ARC (Clark et al., 2018) focus on science questions. MATH (Hendrycks et al., 2021), GSM8K (Cobbe et al., 2021) and MGSM (Shi et al., 2022) focus on mathematical problem solving. There is also a line of works (Kazemi et al., 2023b; Saparov et al., 2024; Tafjord et al., 2021) which construct semi-synthetic benchmarks to evaluate the logical deductive reasoning abilities 

2 

ReMI: A Dataset for Reasoning with Multiple Images 

Table 1 | The reasoning domain and properties of the tasks in ReMI. 

|**Task Name**|**Reasoning**<br>**Domain(s)**<br>**Sequence**<br>**or Set**<br>**Same/Dif.**<br>**Concept**<br>**Interleaved**<br>**Max**<br>**#Images**|
|---|---|
|||
|EmojiAlgebra<br>FuncRead<br>GeomShapes<br>GeomCost<br>Collisions<br>Clocks<br>Schedule<br>Charts<br>CodeEdit<br>Isomorphism<br>Maps<br>RefCOCO<br>IQ|Algebra<br>Seq<br>Same<br>Yes<br>6<br>Calculus<br>Mix<br>Same<br>Yes<br>3<br>Geometry<br>Seq<br>Same<br>No<br>2<br>Geometry, Tabular<br>Seq<br>Dif<br>Yes<br>2<br>Physics<br>Set<br>Same<br>No<br>2<br>Time Arithmetic<br>Set<br>Same<br>No<br>2<br>Time, Tabular<br>Seq<br>Dif<br>Yes<br>2<br>Charts<br>Set<br>Same<br>No<br>2<br>Code<br>Seq<br>Same<br>Yes<br>2<br>Graph Theory<br>Set<br>Same<br>Yes<br>2<br>Spatial, Maps<br>Mix<br>Same<br>Yes<br>4<br>Spatial<br>Seq<br>Dif<br>No<br>2<br>Logic<br>Mix<br>Same<br>Yes<br>5|



of LLMs. In addition, the BIG-Bench (Srivastava et al., 2022) suite of tasks contains many which focus on reasoning. 

**Vision-language reasoning benchmarks.** Some recent benchmarks such as Fu et al. (2023); Kazemi et al. (2023a); Lu et al. (2023); Yue et al. (2023) present reasoning problems that require conditioning on images; however, they predominantly require only a single image, and do not directly measure how well the model can integrate information across different images. Cross-image reasoning benchmarks exist but are restricted to the entailment task or focus on limited number of domains. NLVR (Suhr et al., 2017) creates pairs of images composed of synthetic 2D objects and the task is identifying whether the caption is entailed from the image. NLVR2 (Suhr et al., 2019) extends NLVR by replacing synthetic images with pairs of images sampled from MS COCO (Lin et al., 2014). MaRVL (Liu et al., 2021) expands a similar idea to multi-cultural and multilingual scenarios and only focuses on the natural image domain. SEED-Bench-2 Li et al. (2023a) proposes a hierarchy of different vision-language datasets including multi-image datasets composed of frames extracted from videos. BLINK (Fu et al., 2024) is a collection of 14 visual perception tasks where some of the tasks involve multiple images, e.g. visual similarity and multi-view reasoning. None of these mentioned benchmarks aim to test visionlanguage models for _complex reasoning_ in _multi-_ 

_image_ scenarios. We aim to propose a holistic benchmark covering a wide range of visual information in the world and focuses on complex reasoning of multi-images. 

## **The** ReMI **Dataset** 

Multi-image reasoning can arise in many domains and the problems involving reasoning over multiple images may differ in some key properties. We aim to create a benchmark that exhibits many domains and covers those key properties as much as possible. To this end, we included 13 tasks in our benchmark that covers the following domains: _Algebra_ , _Calculus_ , _Geometry_ , _Tabular Reasoning_ , _Time Arithmetic_ , _Logic_ , _Physics_ , _Spatial Reasoning_ , _Graph Theory_ , _Charts_ , _Maps_ , and _Coding_ . We also identified the following key properties specific to multi-image reasoning and aimed for having tasks that provide a good coverage of them: 

- **Sequential vs Set:** In some tasks, the provided images have to be consumed in a sequence (e.g., computing a quantity from one image and then using that quantity in the second image), whereas in some other tasks, the provided images constitute a set. When more than two images are provided, they may be grouped into subsets that have to be consumed sequentially. 

- **Same vs Different Concept:** In some multiimage reasoning problems, the provided images all correspond to the same concept (e.g., all of 

3 

ReMI: A Dataset for Reasoning with Multiple Images 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Question: <image1><image2>The two clocks in the images show<br>two different times in the same day. Find the absolute difference in  Question: The graph of f(x) is as follows <image1> and<br>terms of minutes between the times shown by the two clocks. g(x) as follows <image2>. Compute f(g(2)).<br>(a) Clocks (b) FuncRead<br>Question: <image1><image2>The mean of Low in first chart and  Question: <image1><image2>Which of the numbered circles in the<br>Lower Middle in second chart is X% (assuming same sample size).  second image would overlap most with the reflection of man in front<br>Find X. Round to 1 decimal. in the first image if they were overlaid?<br>(c) Charts (d) RefCOCO<br>Question: Compute the value of the expression in the image  Question: <image1><image2>If the area of the ABC sector is  Question: Ross is renovating a room shaped like the magenta<br>below <image1> by using the emoji equations in the  157, the length of the BC side is equal to the length of the EF  rectangle in the following image <image1>. They plan to install vinyl<br>following images <image2> <image3> <image4>. side, compute the the degree of the EFD angle. Assume  flooring. The price per unit area is given in the following figure<br>PI=3.14. Round computations to 2 decimal places. <image2>. What will be the total cost of the flooring? All the<br>. computations are rounded to two places of decimal.<br>23<br>6<br>(e) EmojiAlgebra (f) GeomShapes (g) GeomCost<br>Question: Below is an IQ test <image1>. From the possible  Question: <image1><image2>The following images show two  Question: Here are two images. The first image is image A<br>options A, B, C, and D shown in the following images in order  graphs. Each graph is represented in a separate image. Are  <image1> and the second image is image B <image2>. These<br><image2> <image3> <image4> <image5>, which one logically  these two graphs isomorphic? images are from Google Maps that depict two different<br>belongs to the spot of the question mark? regions around W Glendale Avenue in Glendale, AZ. In which<br>image are there more bus stops along Glendale Avenue. The<br>answer is either 'A', 'B' or 'equal'.<br>(h) IQ (i) Isomorphism (j) Maps<br>Question: <image1><image2>The images demonstrate the  Question: The image below shows the current time <image1>.  Question: I have some tikz code:\begin{tikzpicture}<br>before and after a collision between two balls. Is the  And the image below shows the departure times for all trains  \filldraw[fill=yellow!80, draw=orange, line width=1pt] (0,0) ellipse (1.5cm and 0.8cm);\filldraw[fill=yellow!80, draw=orange, line width=0.8pt] (1,0.8) circle (0.5cm);\n\n\\filldraw[fill=orange] (1.4,0.8) --<br>momentum conserved in this collision? Answer with 1 if it is conserved, and with 0 if it is not. departing today <image2>.  Find the destination city for the next scheduled train. (1.7,0.65) -- (1.4, 0.5) -- cycle;\n\n\\filldraw[fill=black] (1.2,0.95) circle (0.1cm);…………..\end{tikzpicture}It renders to look like:\n<image1>. I instead want it to look like:\n<image2>. You just need to remove one of the<br>lines in the original tikz code to create this new figure. Can you tell me which line to remove? Only quote the line I<br>should remove.<br>(k) Collisions (l) Schedule (m) CodeEdit<br>**----- End of picture text -----**<br>


Figure 2 | Sample problems from the tasks in ReMI. Note that the images are provided to the models separately in place of the <image i> markers. 

4 

ReMI: A Dataset for Reasoning with Multiple Images 

   - them are charts, or function graphs) whereas in some other problems, the provided images may correspond to different concepts (e.g., one image might be a geometry shape, and the other might be a table). 

- **Interleaving:** For all our tasks, we can either provide all the images first and then ask a question about them, or the images can be interleaved with the question task when they are referred to. To enable experimenting for both settings, we make a subset of the tasks interleaved while for the others we provide the image at the beginning of the prompt. 

- **Number of images:** In some tasks, a variable number of images may be provided as input. 

Solving our tasks requires parsing and understanding the information in the images and text of the question provided as input, which is often followed by the model having to reason using this information to arrive at the correct answer. We provide a brief description of each task below and a more detailed description in the Appendix. In Figure 2, we illustrate a sample from each of the tasks in ReMI. Moreover, in Table 1, we specify the domain and properties for each of the tasks in ReMI. 

(1) EmojiAlgebra **:** Solve a system of linear equations involving digits and emojis. Each image contains an equation or the final expression to be computed. (2) FuncRead **:** Given multiple function graphs in separate images, answer questions about them. (3) GeomShapes **:** Given two shapes (in two different images) with a common property, compute a missing value of one of the shapes. (4) GeomCost **:** Given the shape of an object (in one image) on which an operation is to be done and a table of various costs (in a different image), compute the total cost of the operation. (5) Collisions **:** Given the before and after snapshots of two objects colliding (each in a separate image), answer questions about their state. (6) Clocks **:** Given two clocks with different designs (each in a separate image), compute the time difference between them. (7) Schedule **:** Given the current time (in one image) and a table of train schedules (in another image), answer questions about the next scheduled train. (8) Charts **:** Given two 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0005-06.png)


**----- Start of picture text -----**<br>
3<br>2<br>1<br>0<br>Time Spent (Minutes)<br>EmojiAlgebra GeomShape Clocks Maps Charts GeomCost Collisions RefCoco FuncRead Schedule Isomorphism CodeEdit IQ ReMI<br>**----- End of picture text -----**<br>


Figure 3 | Average time spent on each problem by humans for the human performance results. 

charts (each in a separate image), possibly in different formats – e.g., one bar chart and one pie chart, identify the differences between the reported values or reason jointly from values in both charts. (9) CodeEdit **:** Given a TikZ code, the rendered image, and the goal image, determine which line of code should be removed to get to the goal image. (10) Isomorphism **:** Given two graphs (in two images), determine if they are isomorphic or not. (11) Maps **:** Given a description of a navigation and four navigation routes on a map (each in a different image), determine which one corresponds to the one in the description. (12) RefCOCO **:** Given a real-world image and another image of same dimensions with non-overlapping circles marked on it, determine which circle overlaps the most with a target entity in the real image. (13) IQ **:** Given a matrix of shapes that have a logical connection and with one missing value, predict the shape that goes into the missing part. 

## **Experiments** 

We report the performance of multiple state-ofthe-art models on our benchmark. 

**Metrics:** We mainly report accuracy for our tasks. For textual outputs, we compute exact match while handling slight variations such as spacing issues, lowercase vs uppercase, etc. For numeric answers, we compute a relaxed accuracy with 1% tolerance, mainly to avoid penalizing rounding errors. In the case of relaxed accuracy with tolerance _𝜖_ , a numeric prediction _𝑝_ is considered correct if (1 − _𝜖_ ) _𝑙_ ≤ _𝑝_ ≤(1 + _𝜖_ ) _𝑙_ where _𝑙_ is the label. Following the original GeomVerse paper (Kazemi et al., 2023a), we report relaxed 

5 

ReMI: A Dataset for Reasoning with Multiple Images 

Table 2 | The performance of SoTA models on ReMI and its individual tasks. The winner for each task is in bold and the second winner is underlined. 

|**Task Name**|**Naive**<br>**Baseline**<br>**Claude3**<br>**Sonnet**<br>**Gemini**<br>**Ultra**<br>**Gemini**<br>**Flash**<br>**Gemini**<br>**1.5**<br>**GPT4**<br>**Turbo**|**Human**|
|---|---|---|
||||
|EmojiAlgebra<br>FuncRead<br>GeomShapes<br>GeomCost<br>Collisions<br>Clocks<br>Schedule<br>Charts<br>CodeEdit<br>Isomorphism<br>Maps<br>RefCOCO<br>IQ|0.0<br>28.0<br>2.5<br>15.0<br>44.5<br>**57.5**<br>5.5<br>24.0<br>15.0<br>36.0<br>**40.0**<br>26.0<br>0.0<br>17.5<br>14.5<br>34.0<br>**51.5**<br>32.5<br>0.0<br>58.5<br>47.0<br>75.0<br>**81.5**<br>70.5<br>30.8<br>51.5<br>36.5<br>56.5<br>50.5<br>**62.0**<br>2.0<br>**5.0**<br>4.0<br>4.0<br>2.5<br>4.0<br>0.0<br>36.0<br>33.0<br>43.0<br>40.5<br>**49.5**<br>2.5<br>40.0<br>30.0<br>53.0<br>**54.0**<br>44.0<br>14.9<br>20.0<br>24.5<br>**46.0**<br>41.0<br>42.0<br>50.0<br>57.0<br>65.0<br>67.0<br>**72.0**<br>71.5<br>28.0<br>39.5<br>39.0<br>**47.0**<br>**47.0**<br>36.5<br>12.0<br>30.0<br>31.0<br>49.0<br>**56.0**<br>37.5<br>25.0<br>50.5<br>30.0<br>53.0<br>**76.0**<br>62.5|100.0<br>100.0<br>100.0<br>90.0<br>100.0<br>80.0<br>90.0<br>95.0<br>95.0<br>100.0<br>100.0<br>95.0<br>100.0|
||||
|ReMI|13.1<br>35.2<br>28.6<br>44.5<br>**50.5**<br>45.8|95.8|



accuracy with 3% tolerance for our GeomShapes and GeomCost tasks as intermediate operations are also rounded and different operation orders lead to slight variations in the final result. For the Clocks, we allow 10 minutes tolerance to account for slight variations in reading times from analog clocks. In our analyses, we also use a metric named _error reduction percentage(ERP)_ with respect to a baseline, which corresponds to how much a model reduces the error with respect to a baseline. We define the ERP of a model _𝑀_ for a task _𝑇_ with respect to a baseline _𝐵_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0006-04.png)


Conceptually, the numerator corresponds to how much of the error has been reduced compared to the baseline, and the denominator normalizes by how much room for error reduction existed. 

**Naive Baseline:** We provide the expected accuracy for a naive baseline that predicts the answers without looking at the images, by only guessing the final answer based on the text of the question. For multi-choice questions, we assume this baseline will predict the answer correctly with 1/ _𝑐_ chance where _𝑐_ is the number of choices (for CodeEdit, we consider any line ending in semicolon to be one of possible choices); for Charts, for every question asking about which cell changed, 

we assume this baseline responds with (0 _,_ 0), and for every question about the number of cells that changed, we assume this baseline responds with 1; for Clocks, when asking about the difference in time, we assume this baseline always predicts 12 ∗ 60 minutes; for RefCOCO, we assume this baseline always predicts the circle labeled 0. 

**Models:** We experiment with three state of the art model families, namely Gemini (Reid et al., 2024; Team et al., 2023), Claude 3 (Anthropic, 2024), and GPT4 (OpenAI, 2023). Within the Gemini family, we experiment with three models with different sizes and properties, namely Gemini Ultra, Gemini 1.5 Pro, and Gemini Flash. From the Claude 3 family, we experiment with the Sonnet model, and from the GPT4 family, we experiment with GPT4 Turbo. 

**Human Performance:** For each task we sampled 20 examples from the test set and had them solved by someone knowledgeable (but not necessarily expert) in that area. We also asked them to measure the amount of time they spent on solving the 20 problems. The average time per problem for each task is reported in Figure 3. We observe that some tasks have been more time consuming than the others with EmojiAlgebra being the most time consuming and the IQ being the least time consuming. 

6 

ReMI: A Dataset for Reasoning with Multiple Images 

## **Human Baseline Substantially Beats SoTA Models in Multi-Image Reasoning** 

In Table 2, we present the results of the models as well as the naive baseline and the human performance on the tasks in ReMI. We make the following observations from the obtained results. Firstly, all the models significantly outperform the naive baseline, almost on any task; however, their performance remains far behind the human performance in general, and also in most of the tasks. Secondly, there are some tasks where none of the current models are good at, including Clocks and Isomorphism, where the performances remain quite low[1] . This reveals a potential capability gap in the current state-of-the-art models. Thirdly, we observe that different models perform well on different tasks. For example, Gemini 1.5 substantially outperforms GPT4-Turbo on the IQ, whereas GPT4-Turbo substantially outperforms Gemini 1.5 on EmojiAlgebra. This hints that the frontier models may have different capabilities and limitations. 

Hereafter, unless stated otherwise, we do the rest of the experiments with Gemini Pro 1.5, the best overall performing model on ReMI. 

## **Single-Image vs Multi-Image Reasoning** 

We measure whether models perform better when we provide the multiple images separately or when we put them all in a single image and feed them to the model. To this end, we report _𝐸𝑅𝑃𝑇_ (single-image model _,_ multi-image model) corresponding to how much the multi-image model reduces the error with respect to the single-image model for each task _𝑇_ . The results are provided in Figure 4. We observe that for most of the tasks, feeding images separately results in positive gains (positive ERP) compared to a single-image case. A manual analysis of the model outputs in the two settings shows that the model may even employ different strategies for solving the problem in these settings. For example, in the case of EmojiAlgebra, we observe that in the single-image case, the model mostly starts by assigning a variable (e.g., _𝑎_ , _𝑏_ , etc.) 

1In the case of the Isomorphism, the dataset is imbalanced with a majority class accuracy of 67 percent. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0007-07.png)


**----- Start of picture text -----**<br>
t<br>40<br>20<br>0<br>Error Reduction Perc.<br>EmojiAlgebra FuncRead GeomShape GeomCost Collisions Clocks Schedule Charts CodeEdit Isomorphism Maps RefCoco IQ ReMI<br>**----- End of picture text -----**<br>


Figure 4 | ERP when the images are provided to Gemini 1.5 as multiple vs a single image. 

to each emoji and then solving the problem by using those variables; However, in the case of multi-image, the model mostly uses either the emojis themselves or their names when doing the calculations. 

**Interleaved tasks are affected more:** Out of the six tasks that are positively affected the most (FuncRead, IQ, CodeEdit, Schedule, Isomorphism, and RefCOCO), we observe that five of them (the first five) are interleaved tasks. Averaging the ERP for the interleaved and non-interleaved datasets, we observe a gain of 19 _._ 8% for the former case and a gain of 4 _._ 9% for the latter case. This hints that reasoning with multiple images might be easier for the models than feeding all images in one image, especially when the images are provided interleaved with text at the right positions. 

## **Failure Analysis** 

For each task, we manually examined 20 examples where the answers given by overall best performing model (Gemini 1.5 Pro) was incorrect and analyzed the dominant reasons behind the failures. This analysis revealed several interesting failure modes – some intuitive and some not – as described below and summarized in Table 3. The diversity of errors observed highlights that this multi-image reasoning domain elicits a wide range of different behaviors that can go wrong in a range of different ways, and that our benchmark tests this wide range of abilities. Calculation errors were present in many of the math-related datasets, so we do not discuss them separately 

7 

ReMI: A Dataset for Reasoning with Multiple Images 

Table 3 | Major error sources for each task, identified with a manual inspection of 20 failed examples. 

|**Task Name**|**Major Source(s) of Error**|
|---|---|
|EmojiAlgebra|1- Calculation errors, 2- Confusing similar emojis, 3- Misreading from the images (especially minus signs).|
|FuncRead|1- Model value readings are typically of by about 1 unit.|
|GeomShapes&<br>GeomCost|1- Calculation errors, 2- Going on a wrong solution path (e.g., computing irrelevant unknown values), 3- Misreading and mis-assigning values (e.g., assigning a<br>length value to the height), 4- Hallucinating non-existent values.|
|Collisions|1- Not recognizing when two objects are moving together after a collision, 2-Ignoring the velocity vector direction when calculating absolute velocity diference|
||between two objects.|
|Clocks|1- Not able to read the time properly, 2- Mistaking the minute hand for the hour hand, 3- Not paying attention to the prompt specifying the times are in the|
||same day.|
|Schedule|1- Not able to read the time properly, 2- Retrieving the wrong values from the table given the time, 3- Sometimes ignoring the 24h format.|
|Charts|1- Mis-assigning values to the correct row/column in heatmap charts, 2- Under-counting the number of diferences in two charts, 3- Reasoning errors of the|
||type_The value decreased from X to X_.|
|CodeEdit|1-Suggesting removal of nonexistent parts of code, 2-Not properly understanding what each line of code represents in the compiled image.|
|Isomorphism|1- Jumping prematurely to a conclusion after fnding one or two nodes that map to each other, 2- Hallucinating non-existent nodes or edges.|
|Maps|1- Incorrectly counts similar pins as the same type of objects (e.g., pins that share the same color but diferent icon), 2- Not paying attention to the prompt|
||specifying the certain area of interest, 3- Gives arbitrary directions that don’t match the situation shown in the grid map.|
|RefCOCO|1- For the image with the circles, coordinate reading is of by 100-200, 2- Not a proper understanding of spatial clues such as_top right_.|
|IQ|1- Unfaithful-ness to model’s own CoT (e.g., it explains the color should be green, but selects the red), 2- Overly predicting the operation to be rotation (despite|
||no rotation being in the dataset).|



||Model|Time Read|
|---|---|---|
||Gemini Ultra|1:00 AM|
||Gemini Flash|9:15 AM|
||Gemini 1.5|1:50 AM|
||Claude3 Sonnet|3:25 PM|
||GPT4 Turbo|5:00 AM|



Figure 5 | Time read by different models for one of the clocks in the Clocks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0008-05.png)


Figure 6 | The confusion matrix of reasoning errors vs image reading errors. 

for each task. 

For EmojiAlgebra, the overall reasoning process of the model is mostly correct. However, the model sometimes confuses similar emojis. As an example, it assigns a similar (or the same) name to and or to and and then these variables get confused in the later calculations. We also observe some misreading of the expressions. 

For both Clocks and Schedule, the model suffers from not being able to read the time correctly; e.g. often the minute hand was mistaken for the hour hand. Figure 5 shows a sample clock and times read by the various models. Despite reading the wrong times, the model generally does a good job of computing the time difference given these wrong times, though it sometimes ignores the prompt instructing it to consider both times to be on the same day. In the case of the Schedule, the value retrieved from the table is often not 

the right value, even given the wrong time read by the mode; Sometimes, this is due the model confusing AM vs PM. 

For GeomShapes and GeomCost, the model makes reasoning errors on the geometry side where it tries to compute the values for unknown sides/angles that are irrelevant to the question. We also observe some misreading of values or misassigning the value for one element to another element (e.g., assigning a side value to a height). Hallucinating non-existent values is another issue. In both cases, the model performs well in understanding and executing the high-level task of extracting a value from the first shape and then using it in the next shape; it also extracts the relevant values from the table mostly correctly. 

For Isomorphism, the model tended to jump 

8 

ReMI: A Dataset for Reasoning with Multiple Images 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0009-01.png)


**----- Start of picture text -----**<br>
0.4 0.5<br>0.2<br>0.0 0.0<br>0.2 Interleaved Max = 2 Images<br>0.4 Non Interleaved 0.5 Max > 2 Images<br>(a) Interleaved vs Non-Interleaved (b) Max = Two vs Max > 2 Images<br>0.4<br>0.2<br>0.2<br>0.0<br>0.0<br>0.2 Sequence 0.2 Same Concept<br>0.4 Set 0.4 Different Concept<br>(c) Sequence vs Set (d) Same vs Different Concept<br>Gemini UltraClaude3 SonnetGemini FlashGPT4 TurboGemini 1.5 Gemini UltraClaude3 SonnetGemini FlashGPT4 TurboGemini 1.5<br>Gemini UltraClaude3 SonnetGemini FlashGPT4 TurboGemini 1.5 Gemini UltraClaude3 SonnetGemini FlashGPT4 TurboGemini 1.5<br>Error Reduction vs Avg. Error Reduction vs Avg.<br>Error Reduction vs Avg. Error Reduction vs Avg.<br>**----- End of picture text -----**<br>


Figure 7 | Model performances as a function of the task properties presented in Table 1. 

to conclusions prematurely, based on some initial guesses. For example, it found one or two nodes that had similar structures and jumped to the conclusion that the graphs are isomorphic, whereas other nodes had different structures. The model also suffered from hallucinating non-existent nodes and edges. 

For RefCOCO, the model understands how to use the provided coordinates; however, the coordinates it reads for the circles tends to be off by 10-20%. Moreover, sometimes the model correctly explained that the object of interest is, e.g., on the _top left_ but then selected a circle that was not on the _top left_ , showing a potential gap in truly understanding what _top left_ or other spatial clues are. 

For IQ, the model was sometimes unfaithful to its own reasoning (e.g., it explained that the answer must be a green shape, but selected a red shape as the final answer). Also, even though we had no rotation operations in the dataset, the model tended to over-predict the logical operation being rotation, probably due to a prior bias on the presence of rotation in IQ questions. 

For FuncRead, the model understands the gen- 

eral logic and follows the calculations correctly, but it fails to correctly read values from the function graphs; the values are mostly off by about 1 unit showing the model can locate the vicinity of the point, but lacks precision. 

For Collisions, the model demonstrates issues in interpreting physics diagrams and calculations, particularly in differentiating between elastic and inelastic collisions. It struggles to account for implicit information such as orientation component of the objects velocity. 

For Charts, while the model reads the correct values from the heatmap charts, it lacks preciseness and assigns values to the wrong row/column (it is typically off by one row/column). Moreover, when we ask the model to identify how many differences there are between two charts, it mostly under-counts. We also see multiple cases where the model claimed a value decreased from _𝑋_ to _𝑋_ (i.e. to the same amount). 

For CodeEdit, while correctly identifying the visual changes in the rendered image, the model lacks understanding of how each line of code contributes to the final image. In some cases incorrectly suggests removing code segments that 

9 

ReMI: A Dataset for Reasoning with Multiple Images 

are not present in the original code. Despite these flaws, the model demonstrates some understanding of the code structure, as it avoids suggesting the removal of critical code components that would prevent code from compiling. 

For Maps, the model has difficulty counting objects of interest accurately, especially when there are many distractions on the map. It also sometimes hallucinates information about restaurants and bars or lists those outside the area of interest. Additionally, it struggles to differentiate between similar pins, such as coffee shops, bars, and restaurants. When asked about directions, the model’s suggestions are often random. While it may list correct streets, the directions it describes do not match the map. Even when it does provide the correct answer, the model’s reasoning is often faulty and seems like guesswork. 

**Reasoning Errors vs Image Reading Errors:** Besides computation errors, we observed that reasoning errors and image reading errors are two of the most dominant sources of failures across the tasks in ReMI. We examined 125 failed examples and verified whether there existed a reasoning error or image reading error in them. The results are provided in Figure 6. 

We observe that in 12% of the cases, the values were read correctly from the image and the reasoning was also sound; the failures in these cases were primarily due to minor calculation errors suggesting that while the model understood the problem and approached it correctly, it stumbled in the final execution. In 37 _._ 6% of the cases, the image values were read correctly, but the reasoning was incorrect, This is the most frequent error type, indicating that correct reasoning still remains one of the critical gaps even in the state-of-the-art models. In 24 _._ 8% of the cases, the model misread some information from the images, but the reasoning is sound. That is, had the model extracted the correct information, the final answer could have been correct. This result indicates a second gap in terms of extracting and parsing the correct values from the images and assigning them to the correct components. Finally, in 25 _._ 6% of the cases, the model struggled both in extracting information from the image and in applying correct reasoning. 

## **Performance as a Function of Task Properties** 

In Table 1, we identified multiple distinguishing factors for each of the tasks in ReMI. Here, we aim to measure and compare model performances for tasks exhibiting each property. We note that a naive averaging of a model’s performance for datasets in each category and comparing to the other category may be flawed due to: 1- Performances on some tasks being generally higher due to the label space being binary or categorical, and 2- some tasks being generally easier/harder than the other tasks. 

To account for the first issue mentioned above, for each model _𝑀_ and task _𝑇_ we compute _𝑃𝑀,𝑇_ as _𝐸𝑅𝑃𝑇_ ( _𝑛𝑎𝑖𝑣𝑒, 𝑀_ ), i.e. the model’s error reduction percentage compared to the naive baseline. This corresponds to how much of the error has been reduced by the model when accounting for random guess, normalized by how much room for error reduction existed when accounting for random guess. To account for the second issue, as a proxy for the hardness of the tasks, we use the average performance of our models on each task _𝑃𝑇_ = number of models� _𝑀[𝑃][𝑀][,][𝑇]_[.][We][then][com-] pute the relative gain compared to the average as _𝑃_[′] _𝑀,𝑇_[=] _[𝑃][𝑀][,] 𝑃[𝑇] 𝑇_[−] _[𝑃][𝑇]_ . Conceptually, this corresponds to the following: After accounting for random noise, how much each model reduced the error with respect to the model-average baseline, on each task. For each model and each group of tasks _𝜏_ (e.g., all interleaved tasks), we compute and report the � _𝑇_ ∈ _𝜏[𝑃]_[′] _𝑀,𝑇_ average | _𝜏_ | in Figure 7. 

According to Figure 7(a), GPT4 Turbo and Gemini 1.5 (the two best performing models) outperform other models on interleaved tasks more than non-interleaved tasks, showing the progress in the frontier models for this recently emerged capability. Figure 7(b) compares the tasks that have a maximum of two images to the tasks where the maximum number of images is more than two. We observe a similar behavior as the interleaved vs non-interleaved case, with Gemini 1.5 gains more on the latter tasks. GPT4 Turbo, however, gains equally on both cases. Interestingly, we observe that while Gemini Flash remains competitive on the former tasks, its performance falls behind on the latter group. In Figure 7(c), for se- 

10 

ReMI: A Dataset for Reasoning with Multiple Images 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0011-01.png)


**----- Start of picture text -----**<br>
40<br>20<br>0<br>Error Reduction Perc.<br>EmojiAlgebra FuncRead GeomShape GeomCost Collisions Clocks Schedule Charts CodeEdit Isomorphism Maps RefCoco IQ ReMI<br>**----- End of picture text -----**<br>


Figure 8 | ERP(0-shot, 2-shot) for Gemini 1.5. 

quence vs set inputs, we see a stark difference for Claude3 Sonnet and Gemini 1.5. Claude3 Sonnet performs better on set type tasks and Gemini 1.5 performs better on sequence type tasks, but almost loses its advantage on set type tasks. Finally, Figure 7(d) shows that when provided with images corresponding to different concepts, most models show a similar behaviour except for Gemini Ultra that performs better when the concepts are different and GPT4 Turbo that performs better when the concepts are the the same. 

## **Zeroshot vs Fewshot Performance** 

So far, we examined the performance of various models in a zero-shot setting. We now examine how much of the gap between the model performance and the human performance can be closed by providing fewshot examples as demonstration to the model. Specifically, we prepend two examples along with their manually-written chain of thought solutions to the prompt. We then measured and report _𝐸𝑅𝑃𝑇_ (zeroshot, fewshot) corresponding the how much the fewshot model reduced the error compared to the zeroshot model. The results are reported in Figure 8. 

According to the results, we observe that the overall performance of the model on ReMI improves from 51 _._ 5% to 57 _._ 9% corresponding to almost 12 _._ 5% relative improvement. This shows that LLMs may be capable of learning multi-image reasoning tasks in context and improve their performance. However, the overall performance still remains significantly behind the human baseline which is 95 _._ 8%. We also see that the amount of improvement is task dependent with some tasks gaining from fewshot examples substantially more than the others. 

## **Conclusion** 

We introduced ReMI, a dedicated benchmark for multi-image reasoning that covers several domains and several key properties that arise when reasoning with multiple images. We evaluated the frontier LLMs on ReMI and compared their performance to humans. The results show a stark gap between model performance and human performance showing a significant room for improvement in the reasoning capabilities of the current state-of-the-art LLMs. Future work can focus on improving LLMs for the limitations found in our failure analysis and measure how much they translate to improvements on ReMI. 

## **Acknowledgements** 

We thank Behnam Neyshabur for great feedback. 

## **References** 

- J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- K. Ahrabian, Z. Sourati, K. Sun, J. Zhang, Y. Jiang, F. Morstatter, and J. Pujara. The curious case of nonverbal abstract reasoning with multimodal large language models. _arXiv preprint arXiv:2401.12117_ , 2024. 

- J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in Neural Information Processing Systems_ , 35: 23716–23736, 2022. 

- R. Albert and A.-L. Barabási. Statistical mechanics of complex networks. _Reviews of modern physics_ , 74(1):47, 2002. 

- A. Anthropic. The claude 3 model family: Opus, sonnet, haiku. _Claude-3 Model Card_ , 2024. 

- S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh. Vqa: Visual ques- 

11 

ReMI: A Dataset for Reasoning with Multiple Images 

tion answering. In _Proceedings of the IEEE international conference on computer vision_ , pages 2425–2433, 2015. 

- A. Awadalla, I. Gao, J. Gardner, J. Hessel, Y. Hanafy, W. Zhu, K. Marathe, Y. Bitton, S. Gadre, S. Sagawa, et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. _arXiv preprint arXiv:2308.01390_ , 2023. 

- A.-L. Barabási and R. Albert. Emergence of scaling in random networks. _science_ , 286(5439):509– 512, 1999. 

- L. Beyer*, A. Steiner*, A. Susano Pinto*, A. Kolesnikov*, X. Wang*, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello, T. Unterthiner, D. Keysers, A. Gritsenko, X. Chen, S. Koppula, A. Grycner, M. Bauer, M. Bošnjak, F. Liu, N. Houlsby, M. Kumar, K. Rong, J. Eisenschlos, M. Minderer, P. Voigtlaender, I. Bica, I. Balazevic, J. Puigcerver, P. Papalampidi, O. Henaff, X. Xiong, R. Soricut, J. Harmsen, and X. Zhai*. PaliGemma: A versatile 3B VLM for transfer, 2024. To appear. 

- X. Chen, X. Wang, S. Changpinyo, A. Piergiovanni, P. Padlewski, D. Salz, S. Goodman, A. Grycner, B. Mustafa, L. Beyer, et al. Pali: A jointly-scaled multilingual language-image model. _arXiv preprint arXiv:2209.06794_ , 2022. 

- P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018. 

- K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- P. Erdős and A. Rényi. On random graphs. _Publicationes Mathematicae Debrecen_ , 6:290–297, 1959. 

- L. Fan, W. Hua, X. Li, K. Zhu, M. Jin, L. Li, H. Ling, J. Chi, J. Wang, X. Ma, et al. Npharde- 

   - val4v: A dynamic reasoning benchmark of multimodal large language models. _arXiv preprint arXiv:2403.01777_ , 2024. 

- B. Fatemi, J. Halcrow, and B. Perozzi. Talk like a graph: Encoding graphs for large language models. In _ICLR_ , 2024. 

- C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, Z. Qiu, W. Lin, J. Yang, X. Zheng, K. Li, X. Sun, and R. Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. _ArXiv_ , abs/2306.13394, 2023. URL `https://api.semanticscholar.org/ CorpusID:259243928` . 

- X. Fu, Y. Hu, B. Li, Y. Feng, H. Wang, X. Lin, D. Roth, N. A. Smith, W.-C. Ma, and R. Krishna. Blink: Multimodal large language models can see but not perceive. _arXiv preprint arXiv:2404.12390_ , 2024. 

- Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6904–6913, 2017. 

- A. Hagberg, P. Swart, and D. S Chult. Exploring network structure, dynamics, and function using networkx. Technical report, Los Alamos National Lab.(LANL), Los Alamos, NM (United States), 2008. 

- D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_ , 2020. 

- D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. In J. Vanschoren and S. Yeung, editors, _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks_ , volume 1. Curran, 2021. 

- P. W. Holland, K. B. Laskey, and S. Leinhardt. Stochastic blockmodels: First steps. _Social networks_ , 5(2):109–137, 1983. 

12 

ReMI: A Dataset for Reasoning with Multiple Images 

- S. Huang, L. Dong, W. Wang, Y. Hao, S. Singhal, S. Ma, T. Lv, L. Cui, O. K. Mohammed, B. Patra, et al. Language is not all you need: Aligning perception with language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- M. Kazemi, H. Alvari, A. Anand, J. Wu, X. Chen, and R. Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning. _arXiv preprint arXiv:2312.12241_ , 2023a. 

- M. Kazemi, Q. Yuan, D. Bhatia, N. Kim, X. Xu, V. Imbrasaite, and D. Ramachandran. Boardgameqa: A dataset for natural language reasoning with contradictory information. In _NeurIPS_ , 2023b. 

- S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg. Referitgame: Referring to objects in photographs of natural scenes. In _Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)_ , pages 787–798, 2014. 

- A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, et al. Solving quantitative reasoning problems with language models. _Advances in Neural Information Processing Systems_ , 35:3843–3857, 2022. 

- B. Li, Y. Ge, Y. Ge, G. Wang, R. Wang, R. Zhang, and Y. Shan. Seed-bench-2: Benchmarking multimodal large language models. _arXiv preprint arXiv:2311.17092_ , 2023a. 

- J. Li, D. Li, S. Savarese, and S. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023b. 

- T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6- 12, 2014, Proceedings, Part V 13_ , pages 740– 755. Springer, 2014. 

- A. D. Lindström and S. S. Abraham. Clevr-math: A dataset for compositional language, visual 

   - and mathematical reasoning. _arXiv preprint arXiv:2208.05358_ , 2022. 

- F. Liu, E. Bugliarello, E. M. Ponti, S. Reddy, N. Collier, and D. Elliott. Visually grounded reasoning across languages and cultures. In M.F. Moens, X. Huang, L. Specia, and S. W.t. Yih, editors, _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 10467–10485, Online and Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main. 818. URL `https://aclanthology.org/ 2021.emnlp-main.818` . 

- H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36, 2024. 

- Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. Mmbench: Is your multi-modal model an all-around player? _arXiv preprint arXiv:2307.06281_ , 2023. 

- P. Lu, R. Gong, S. Jiang, L. Qiu, S. Huang, X. Liang, and S.-C. Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. _arXiv preprint arXiv:2105.04165_ , 2021. 

- P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.C. Zhu, O. Tafjord, P. Clark, and A. Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. _Advances in Neural Information Processing Systems_ , 35:2507–2521, 2022. 

- P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. _arXiv preprint arXiv:2310.02255_ , 2023. 

- K. Marino, M. Rastegari, A. Farhadi, and R. Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In _Proceedings of the IEEE/cvf conference on computer vision and pattern recognition_ , pages 3195–3204, 2019. 

13 

ReMI: A Dataset for Reasoning with Multiple Images 

- OpenAI. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- D. Paperno, G. Kruszewski, A. Lazaridou, Q. N. Pham, R. Bernardi, S. Pezzelle, M. Baroni, G. Boleda, and R. Fernández. The lambada dataset: Word prediction requiring a broad discourse context. _arXiv preprint arXiv:1606.06031_ , 2016. 

- N. F. Rajani, B. McCann, C. Xiong, and R. Socher. Explain yourself! leveraging language models for commonsense reasoning. _arXiv preprint arXiv:1906.02361_ , 2019. 

- M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. Lillicrap, J.-b. Alayrac, R. Soricut, A. Lazaridou, O. Firat, J. Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 

- A. Saparov, R. Y. Pang, V. Padmakumar, N. Joshi, M. Kazemi, N. Kim, and H. He. Testing the general deductive reasoning capacity of large language models using ood examples. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- F. Shi, M. Suzgun, M. Freitag, X. Wang, S. Srivats, S. Vosoughi, H. W. Chung, Y. Tay, S. Ruder, D. Zhou, et al. Language models are multilingual chain-of-thought reasoners. _arXiv preprint arXiv:2210.03057_ , 2022. 

- A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _arXiv preprint arXiv:2206.04615_ , 2022. 

- A. Suhr, M. Lewis, J. Yeh, and Y. Artzi. A corpus of natural language for visual reasoning. In R. Barzilay and M.-Y. Kan, editors, _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 217–223, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-2034. URL `https://aclanthology.org/P17-2034` . 

- A. Suhr, S. Zhou, A. Zhang, I. Zhang, H. Bai, and Y. Artzi. A corpus for reasoning about natural language grounded in photographs. In A. Korhonen, D. Traum, and L. Màrquez, editors, _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 6418–6428, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1644. URL `https: //aclanthology.org/P19-1644` . 

- O. Tafjord, B. Dalvi, and P. Clark. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pages 3621–3634, Online, Aug. 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl. 317. URL `https://aclanthology.org/ 2021.findings-acl.317` . 

- G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- M. Tsimpoukelli, J. L. Menick, S. Cabi, S. Eslami, O. Vinyals, and F. Hill. Multimodal few-shot learning with frozen language models. _Advances in Neural Information Processing Systems_ , 34:200–212, 2021. 

- A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman. Superglue: A stickier benchmark for generalpurpose language understanding systems. _Advances in neural information processing systems_ , 32, 2019. 

- J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. _arXiv preprint arXiv:2311.16502_ , 2023. 

14 

ReMI: A Dataset for Reasoning with Multiple Images 

- R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. Hellaswag: Can a machine really finish your sentence? _arXiv preprint arXiv:1905.07830_ , 2019. 

## **Further Failure Analysis** 

In the main text, we provided a high-level summary of the model failures for each task. In Figures 9, 10, 11, 12, 13, 14, 15 16 and 17 we present some examples of model failures on several of our tasks. 

## **Statistics about** ReMI 

As mentioned in the main text, the problems in ReMI contain at least two images. Figure 18(a) shows the average length of the questions for each task in ReMI indicating a wide range of questions lengths across tasks. Figure 18(b) shows the number of unique labels that each task has (e.g., for binary tasks, there are two unique labels). Figure 18(c) provides the statistics of the number of problems that have a specific number of images. 

## **Details about the Tasks in** ReMI 

Below, we provide a detailed description of how each task in ReMI has been created. 

- EmojiAlgebra: We created random systems of linear equation where the values for the variables can be derived one-by-one by looking at the equation for which the value for all variables on the right-hand side is known. We also created a random expression with those variables whose value was to be computed. We then created images by replacing the variables with emojis. 

- FuncRead: To create this task, we sampled polynomial functions of degree 1, 2, or 3 and plotted their graphs using the matplotlib library. Then we ask the following questions about them: reading values from different functions and summing or subtracting them, computing the limit of a function that is defined as one of the graphs for some domain 

   - of values and the other graph for the values outside that domain, function composition, finding a value of interest (e.g., where the derivative is zero) from one graph and reading the other function value at that point, and finding the graph that corresponds to a given function. 

- GeomShapes: We generated this dataset by sampling a shape from a set of pre-defined shapes. Each shape has fixed number of predefined formulas associated with it corresponding to area, perimeter, angles etc. For each formula, we have input elements and output elements. We first sample one shape and its formula and assign values to input elements, the output element of this formula would be shared with another shape. We then sample another shape and formula whose atleast one input element say _𝑥_ is of same type as output element of the first shape. We assign this element _𝑥_ from the computation of first shape but hide it in the question. We then proceed to ask the question based on this output formula of second shape. The two questions share this element _𝑥_ which is indicated in the question. This task is an extension of GeomVerse (Kazemi et al., 2023a). 

- GeomCost: We generated this dataset by sampling a shape from a pre-selected set of shapes like triangle, parallelogram, square, rectangle etc and selecting one formula out of perimeter and area corresponding to this shape and assigned all the values corresponding to the perimeter or area value. We then choose a template story correspondinf to fencing a boundary, icing the cake etc. out of 10 pre-defined template texts and choose a table corresponding to this template. The table designs are also varied slightly out of fixed number of styles. The cost values are assigned randomly from 1-100. This task is also an extension of GeomVerse (Kazemi et al., 2023a). 

- Collisions: We created visualizations of twoobject collisions, varying initial positions (horizontal, vertical, angled) and randomly assigning masses and velocities. For each collision pair, we then assessed elasticity, co- 

15 

ReMI: A Dataset for Reasoning with Multiple Images 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0016-01.png)


Figure 9 | Model failure: Although the model arrives at the correct answer, the mapping showing the isomorphism between the two graphs is wrong. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0016-03.png)


Figure 10 | Model failure: The model mixes the minutes and hours hand on the clock and also reads 08:34 as 9:34 from the table. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0016-05.png)


Figure 11 | Model failure: The model fails to recognize the category Astronomy whose value has changed and instead outputs that no value has changed. 

efficient of restitution, and conservation of kinetic energy and momentum. 

- Clocks: We generated clock images with different shape, color, style, number representations, etc. using tikz code. Each clock shows 

16 

ReMI: A Dataset for Reasoning with Multiple Images 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0017-01.png)


Figure 12 | Model failure: The model read’s the values wrong from the graph. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0017-03.png)


Figure 13 | Model failure: The model wrongly interprets the sides of a parallelogram as its base and height and gets completely off track. 

a random time and a AM or PM is also added randomly to the image as well. Then, for each pair of images, we compute the difference between their times in terms of minutes and use that as the label. 

- Schedule: We generated one clock image showing a random time, similar to the way it was generated for the Clocks. Then, we also generated a random table with different columns (departure time, arrival time, train name, gate, etc.) and with different styles (colors, horizontal/vertical line separators, text rotation, multi-line text, etc.) that included information about the events happening at various times. We then asked ques- 

tions about the next event happening given the current time shown on the clock. 

- Charts: We first randomly generate data matrices and series that are suitable for plotting into four different types of charts: (1) heatmap (2) bar chart (3) line chart (4) pie chart. Then we create a modified version of the data series or matrices by randomly editing one to a few values. This way we obtain pairs of edited data matrices/series. Then we use the Matplotlib library to plot each data matrix/series into a chart by randomly selecting a suitable chart type and randomly choosing a color scheme, layout, etc. for the chart. Heuristics is applied to guarantee that 

17 

ReMI: A Dataset for Reasoning with Multiple Images 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0018-01.png)


**----- Start of picture text -----**<br>
Question: Here are two images. The first image is image A  Model Response:  "Image A shows 3 bus stops along Liberty<br><image1> and the second image is image B <image2>. These  Avenue, while image B shows 2. ", "answer": "A"<br>images are from Google Maps that depict two different regions<br>around Liberty Avenue in Pittsburgh, PA.  In these images<br>restaurants are represented by orange pins that depict a knife and a<br>fork. Coffee shops are represented by orange pins and/or boxes<br>with an image of a coffee cup in them. Bars are represented by<br>orange pins with an image of a wine glass inside them. Bus stops<br>are represented by a blue square box with an image of a bus inside<br>it. [...] In which image are there more bus stops along Liberty<br>Avenue? The answer is either 'A', 'B' or 'equal'.<br>**----- End of picture text -----**<br>


Figure 14 | Model failure: The model reads the incorrect number of bus stops along Liberty Avenue in both the images but luckily gets the answer correct. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0018-03.png)


**----- Start of picture text -----**<br>
Question: Here are four images. The first image is image A <image1>. The second image is image B <image2>. The third  Model Response:  "The directions indicate starting on M street<br>image is image C <image3> and the fourth image is image D <image4>. [...] At exactly one of the intersections there is a green pin indicating the starting location. At exactly one of the intersections there is a red pin indicating the ending  and heading South, then turning right on G street. This matches<br>location. Here is a set of directions describing a path from the starting location to the ending location. Directions: Head  the partial path shown in image C.  The directions then say to<br>south on M street. Turn right onto G street. Turn left onto F street. Continue for 2 blocks to arrive at your destination. In  turn left on F street and continue for two blocks. This matches<br>each image there is a partial (incomplete) path outlined in the color blue. Exactly two of the four images have the property that their partial paths when combined together match the path that is described in the directions above.  the partial path shown in image B.", "answer": "B,C"<br>Please identify the pair of images. The answer is simply a pair of names (e.g. A,B or C,D and so on).<br>**----- End of picture text -----**<br>


Figure 15 | Model failure: The model interprets the wrong image as showing the path described in the text. 

the selected chart type is suitable for plotting the data. Finally we sample from a set of question templates to form QA pairs for each pair of chart. The templates include simple elementary reasoning questions across the two charts or detecting differences of the two charts. 

- CodeEdit: We first asked a language model to generate tikz code for a list of random objects. We then comment out a single line in the code and recompile it. We only keep the examples where the edited version compiles correctly, and the compiled image is not equal to the original image. A few filters 

   - were applied to ensure the edited image is sensible (e.g. the code being removed is not a variable definition or the beginning of a for loop); specifically, the removed code line had to start with `\draw` or `\filldraw` and end with a `;` . 

- Isomorphism: We used the NetworkX library Hagberg et al. (2008) to generate random graphs using one of the following generators: Erdős-Rényi (ER) graphs (Erdős and Rényi, 1959), scale-free networks (SFN) (Barabási and Albert, 1999), graphs following the Barabási–Albert (BA) model (Albert and Barabási, 2002) and 

18 

ReMI: A Dataset for Reasoning with Multiple Images 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0019-01.png)


Figure 16 | Model failure: The model doesn’t seem to understand the pattern properly but guesses the right answer. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0019-03.png)


Figure 17 | Model failure: The model fails to correctly match the vertices of the cube to their implicit locations in the image and ends up suggesting to remove the right line of code. 

- stochastic block model (SBM) (Holland et al., 1983), as well as star, path and complete graphs. Then, for positive examples (i.e. examples where the two graphs are isomorphic), we visualized the same graph with different NetworkX layout, different names for the nodes, and different styles. For the non-isomorphic case, we either sampled two random graphs (this produces easy negative examples) or sampled one random graph and slightly modified it by adding/removing one or two nodes/edges (this produces hard negative examples). This dataset is in part 

inspired by the works of Fan et al. (2024); Fatemi et al. (2024). 

- Maps: Our curated Maps dataset consists of both synthetic and real world examples. We first describe the curation process for the synthetic examples. For synthetic counting queries, we first generate a grid with five horizontal streets and five vertical streets. The street names are randomly assigned in [A..Z]. We then place points of interest (POIs) (gas stations, coffee shops, shopping center and bus stops) at various blocks. We process each block and with a sampling probability 

19 

ReMI: A Dataset for Reasoning with Multiple Images 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0020-01.png)


**----- Start of picture text -----**<br>
200<br>10 [3]<br>150<br>100<br>50<br>10 [2] 0<br>(a) Average length of the questions for each (b) Number of unique labels in each task<br>of the tasks in ReMI. (for 200 examples).<br>100<br>Gemini Ultra<br>10 [3] 80 Gemini 1.5<br>GPT4<br>60 Flash<br>Claude3 Sonnet<br>40<br>Human<br>10 [2]<br>20<br>2 3 4 5 6 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5<br>Number of images Time Spent (Minutes)<br>Avg. Question Length<br>Number of unique labels<br>EmojiAlgebra FuncRead GeomShape GeomCost Collisions Clocks Schedule Charts CodeEdit Isomorphism Maps RefCoco IQ EmojiAlgebra FuncRead GeomShape GeomCost Collisions Clocks Schedule Charts CodeEdit Isomorphism Maps RefCoco IQ<br>Number of problems Error Reduction Perc.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images_images/nips_2024_ReMI_A_Dataset_for_Reasoning_with_Multiple_Images.pdf-0020-02.png)


**----- Start of picture text -----**<br>
(c) Number of problems in ReMI that have(d) Model error reduction percentage as a<br>𝑛 images, for different values of 𝑛 . function of the time spent by humans for<br>solving the problems.<br>**----- End of picture text -----**<br>


Figure 18 | (a), (b), (c): Statistics for ReMI and its tasks. (d) Model performance vs human time. 

of _𝑝_ = 0 _._ 1 decide whether to place a POI or not. We then pick a POI at random from the list and place it at the block. Similarly, we place traffic lights and stop signs at each corner with a sampling probability of _𝑝_ = 0 _._ 1. To generate the second image we copy the above constructed grid and pick at random a particular street. We then pick at random a particular POI on the street and place additional copies of the POI on the street. With a small probability of 0.05 we leave the second image unchanged. 

Similarly, for the direction matching queries we generate a grid image as above. We pick a random start and end point and pick a random set of directions between them. We split this direction at a random point to generate two of the four images containing the partial directions. The remaining two images are constructed by picking two different distinct directions at random. 

For the case of real data we first prompted a language model to generate a list of 100 

   - cities and an associated street/avenue in that city. We then take this list and for each entry we get two images from Google Maps API that are centered at the particular street. We then manually study the two images and look for distinguishing features (such as bus stops, places of worship, hotels etc.) to construct the query. 

- RefCOCO: We sampled 500 imagees from the refCOCO (Kazemzadeh et al., 2014) dataset. We then sample 15 points to lie uniformly randomly across the image. We then choose the points that overlap with the goal object as follows. We have the ground truth bounding box of the referred object from the original dataset. We first select the datapoints where at least 1, but less than 8, and include these in `label_inbbox` . We th ar ethe points in the center 25% of the bbob, and so on for provide various precisions for points with ‘most overlap’: `label_mindist_bboxcenter` is the point that is the closest to the center of the bounding box. `label_25p_tolerance` are the la- 

20 

ReMI: A Dataset for Reasoning with Multiple Images 

bels in the middle 25% of the bounding box and so on for the `label_50p_tolerance` and `label_75p_tolerance` . Finally we manually check all the datapoints to ensure that the labels points actually overlap with the goal object. 

- IQ: We created simple IQ tests where a grid of 2x2 is given as input whose bottom-left value is missing, and four choices are provided as the possible answers from which the model has to select one. The images on the top row are two shapes that are different only in terms of one logical operation. The model has to identify that operation and apply it to the image on the bottom left to find the final answer. We included a number of different shapes (triangles, rectangles, pentagons, parallelograms, etc.) and a number of different logical operations (border color, border pattern, fill color, hatch style, change in shape, etc.). This task is similar in nature to the IQ tasks in Ahrabian et al. (2024); Huang et al. (2024), but the choices are provided as separate images. 

**Quality Check:** To ensure high quality, we went through multiple rounds of checking, where the questions and answers for each task were examined by multiple authors to see any problems can be identified, including whether the label is correct, whether the instructions provided are sufficient to solve the problem and output it in the right format, whether the text of the question is clearly written, whether the images are clearly understandable and the quantities are easily readable, etc. This procedure was done until no more issues could be found for any of the tasks. As a second level of quality check, once we performed our human evaluation, we manually looked into the questions where the label provided by the humans disagreed with our labels to ensure that our labels are indeed the correct ones. 

## **Model Performance vs Human Time** 

In the main text, we reported the average time per problem spent by humans for each task. One may expect that if humans spent more time on a set of problems, those problems might be more 

difficult for the models. To verify this hypothesis, we fit linear functions to the model performances as a function of time spent by humans and report the results in Figure 18(d). We observe that only for two of the models (Gemini Ultra and Gemini Flash) the performance goes down as a function of spent time. For other models, the performance almost remains flat. 

## **Experimental Setup** 

For all of the tasks in ReMI, we allowed the models a maximum of 512 output tokens as we observed that when models went beyond that, they were mostly stuck in a wrong path that did not reach a solution and that models could not recover from it. We prompted the model to produce a JSON with two fields: "explanation" containing the step by step reasoning of the model, and the "answer" containing the final answer. We measured the average number of responses that either ended prematurely or did not produce a valid JSON for each model and observed that the numbers were small. Specifically, the numbers for Claude3 Sonnet, Gemini Ultra, Gemini Flash, Gemini 1.5, and GPT4 Turbo were 0.4, 0.3, 0.5, 0.8 and 1.9 percent respectively. For Gemini and Claude, we used the Vertex AI API. For GPT4 Turbo, we used the OpenAI API. 

To compute the final performance, we did the following postprocessing on the golden and predicted labels: 1- in the case of string outputs, we lowercased both golden and predicted answers before comparing them, 2- if the predicted label had an extra or missing () around the final answer, we still counted it as true, 3- if the predicted label contained extra units (e.g., producing 20% instead of 20), we still counted it as true, 4- for the CodeEdit, some lines of codes contained a comment after the code; we considered a predicted label to be true regardless of whether it output the comments or not, 5- we ignored spacing issues and assumed a predicted label to be correct even if it had extra or missing spaces, and finally 6- for the FuncRead, if the golden label was, e.g., _𝑓_ and the predicted label was _𝑓_ ( _𝑥_ ), we counted it as correct. 

21 

ReMI: A Dataset for Reasoning with Multiple Images 

## **Limitations** 

- While our dataset covers a wide range of domains where reasoning over multiple images is required, there may still be many other domains where such reasoning is required that are not covered in our dataset e.g., reasoning about chemicals, reasoning about music sheets, etc.). 

- In our experiments for measuring performance as a function of task properties, we had to use proxies to tease apart the effect of random chance and task difficulty. It is possible that with a different procedure for teasing these effects apart, the results change slightly. For this reason, the general patterns observed in those experiments are more important that the small numeric differences. 

22 

