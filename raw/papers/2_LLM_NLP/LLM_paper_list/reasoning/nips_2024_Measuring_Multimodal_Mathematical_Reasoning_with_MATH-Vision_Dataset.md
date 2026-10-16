# **Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset** 

**Ke Wang**[1] _[,][♢]_ **Junting Pan**[1] _[,]_[⋆] _[,][♢]_ **Weikang Shi**[1] _[,][♢]_ **Zimu Lu**[1] **Mingjie Zhan**[2] _[,][♢][,][♡]_ **Hongsheng Li**[1] _[,]_[3] _[,][♡]_ 

⋆ project lead, _♢_ joint first author, _♡_ directional lead 

1Multimedia Lab, The Chinese University of Hong Kong, 2Sensetime, 3Shanghai AI Laboratory 

https://mathvision-cuhk.github.io 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0001-06.png)


**----- Start of picture text -----**<br>
Human (testmini) InternLM-XComposer2-VL<br>GPT4V Qwen-VL-Max Level 1 Level 3 Level 5<br>Gemini Pro Random Chance Level 2 Level 4<br>transformation algebra analytic Qwen-VL-Max<br>geometry 90 geometry<br>topology 70 arithmetic 30<br>statistics 50 combinatorial Gemini Pro-CoT 20 GPT4V-CoT<br>30 geometry<br>solid<br>combinatorics<br>geometry<br>metric<br>geometry counting<br>-length GPT4V Gemini Pro<br>metric<br>descriptive<br>geometry<br>-area geometry<br>metric<br>graph theory<br>geometry logic InternLM-XComposer2-VL<br>-angle<br>(a) Subjects (b) Levels<br>**----- End of picture text -----**<br>


Figure 1: The accuracies of four prominent Large Multimodal Models (LMMs), random chance, and human performance are evaluated on our proposed MATH-Vision (MATH-V) across 16 subjects and 5 levels of difficulty, with Level 1 being the easiest and Level 5 the most challenging. Human performance is assessed using the testmini subset. 

## **Abstract** 

Recent advancements in Large Multimodal Models (LMMs) have shown promising results in mathematical reasoning within visual contexts, with models approaching human-level performance on existing benchmarks such as MathVista. However, we observe significant limitations in the diversity of questions and breadth of subjects covered by these benchmarks. To address this issue, we present the MATH-Vision (MATH-V) dataset, a meticulously curated collection of 3,040 high-quality mathematical problems with visual contexts sourced from real math competitions. Spanning 16 distinct mathematical disciplines and graded across 5 levels of difficulty, our dataset provides a comprehensive and diverse set of challenges for evaluating the mathematical reasoning abilities of LMMs. Through extensive experimentation, we unveil a notable performance gap between current LMMs and human performance on MATH-V, underscoring the imperative for fur- 

ther advancements in LMMs. Moreover, our detailed categorization allows for a thorough error analysis of LMMs, offering valuable insights to guide future research and development. 

## **1 Introduction** 

The latest advancements in Large Language Models (LLMs) (OpenAI, 2023a; Touvron et al., 2023; Anil et al., 2023a; Almazrouei et al., 2023; Team, 2023b) and Large Multimodal Models (LMMs) (OpenAI, 2023b; Team, 2023a; Dong et al., 2024; Lin et al., 2023; Liu et al., 2023a; Bai et al., 2023) have demonstrated remarkable capabilities, achieving impressive performance on a diverse array of language and visual-language tasks (Zhao et al., 2023; Yang et al., 2023; Rozière et al., 2024; Brown et al., 2020; Goyal et al., 2017; Huang et al., 2023; Zhang et al., 2023b; Cui et al., 2024; Zhang et al., 2023a). As a result of these strides, recent LMMs like GPT-4V (OpenAI, 

2023b), InternLM-XComposer2-VL (Zhang et al., 2023c) and Gemini (Team, 2023a) are approaching the performance level of average humans on multimodal mathematical reasoning. Specifically on MathVista (Lu et al., 2023), which serves as the most commonly used test-bed to assess the LMMs’ ability in mathematical reasoning in visual contexts, current models achieve 57.6 compared to humans’ 60.3, and for some subtasks like geometry problem solving, math word problems, LMMs can even surpass average humans. This development raises a critical question: _Can the numbers indicate that the_ _**mathematical reasoning** ability of LMMs is comparable to average humans?_ 

To answer this question, we conducted a thorough examination of the data within MathVista, which unifies math questions on figures in nearly all existing datasets and is considered the most comprehensive benchmark for mathematical reasoning in visual contexts by now. MathVista comprises an extensive collection of math-related visual-question-answering (VQA) instances gathered from diverse sources including 19 existing VQA datasets, nearly all existing math-VQA datasets, and 3 newly collected ones. Our analysis of MathVista reveals two key issues: Firstly, although a significant portion of its visual data exhibits diversity, the questions regarding the visual data are quite limited in question types and contents. For example, we identified that among the 375 questions on “abstract scene”, the majority can be categorized into just three types of questions: (1) Object Counting, (2) Length Measurement with a given reference, and (3) Time Inquiry based on a clock. Questions are generally simple and seldom require in-depth mathematical reasoning. Similar problems also exist in image contexts of “function plots”, “synthetic scenes”, “natural images”, etc. This tendency towards monotonous question patterns is commonly observed in data derived from 19 VQA datasets where annotators are designated to create math-related questions following the collection of contextual images. 

Secondly, the data originating from Math-QA datasets is confined to a narrow scope of mathematical subjects. These datasets compile questions directly from math tests and textbooks, accompanied by visual contexts. However, the range of subjects remains limited, primarily encompassing topics like plane geometry, tables, and chart question-answer pairs. Although three newly col- 

lected datasets show considerable diversity in the problems they present, the overall number of problems per subject remains notably sparse. 

Therefore, recognizing the limitations of existing benchmarks and aiming to more comprehensively evaluate the mathematical reasoning skills of LMMs, we introduce the MATH-Vision (MATH-V) dataset. This dataset comprises 3,040 high-quality mathematical problems within visual contexts across 12 grades, carefully selected from 19 math competitions. To ensure a high-quality standard, **all** data within our dataset were cross-validated and filtered by multiple expert annotators to make sure to have a unique and correct answer. The dataset is well-balanced, featuring 1,532 problems in an open-ended format and 1,508 in a multiple-choice format. In order to better provide insight regarding model performance on our proposed dataset, we categorize all problems by difficulty levels ranging from 1 to 5, and 16 distinct mathematical disciplines, such as logic, arithmetic, graph theory, topology, analytic geometry, combinatorial geometry, and solid geometry, among others. This classification also undergoes a human verification system to ensure accuracy and reliability. We hope that this categorization can facilitate better analysis of mathematical problem-solving skills across various levels of difficulty and subject areas. 

We conduct extensive experiments with our datasets aiming at providing insights into the mathematical reasoning ability of current LMMs on human tests. As depicted in Figure 1, the initial observation is a substantial gap in mathematical reasoning within visual contexts between LMMs and humans, with respective scores of 22.76 and 75.66. Our MATH-V benchmark establishes a significant challenge for LMMs but is relatively easy for humans to achieve a good score. Within existing LMMs, the performance of the model trained with internal data ( _i.e.,_ GPT-4V, Gemini) is still better than that of the model trained with publicly available data ( _i.e.,_ SPHINX (Lin et al., 2023), InternLM-XComposer2-VL). While the topperforming models (GPT-4V and Gemini) excel at basic computational tasks, they cannot comprehend the invariant properties of geometric objects under continuous deformations, such as stretching, twisting, crumpling, and bending. This suggests current top-tier LMMs are not yet comparable to average humans in terms of mathematical reasoning in visual contexts. Therefore, there remains a 

significant margin for improvement for LMMs to reach human-level performance. In conclusion, the contributions of this study are listed as follows: 

- We propose the MATH-V benchmark comprised of 3k problems with visual context selected from real math competitions. All problems are classified into 5 difficulty levels and 16 distinct mathematical disciplines. 

- We benchmark several open-source and proprietary LMMs with MATH-V and there exists a big gap in mathematical reasoning within visual context between humans and LMMs. 

- Finally, benefiting from the fine-grained categorization, we provided a very comprehensive error analysis of current LMMs, providing insights for improvement. 

## **2 Related Works** 

## **2.1 Mathematical Reasoning Benchmarks** 

Various benchmark datasets (Hendrycks et al., 2020; **?** , 2021; **?** ; Amini et al., 2019) have been proposed to assess a model’s mathematical reasoning capabilities. However, current multimodal benchmarks (Agrawal et al., 2015; Masry et al., 2022; Mobasher et al., 2022; Gurari et al., 2018) primarily focus on visual recognition tasks, and multimodal math reasoning benchmarks are less common. Early multimodal reasoning benchmarks, such as (Lu et al., 2021) and GeoQA+ (Chen et al., 2021), are very limited in subjects, primarily focused on plane geometry. Among recent multimodal benchmarks, MMMU (Yue et al., 2023) stands out, though it is limited to multiple-choice questions and only a small fraction of its content is mathematical. Another recent benchmark, MathVista (Lu et al., 2023), evaluates mathematical reasoning abilities within diverse visual contexts. However, it lacks a detailed classification of mathematical subdomains and emphasizes visual abilities rather than mathematical reasoning. In contrast, our benchmark contains math problems rigorous in both mathematical reasoning and visual recognition, categorized by clear difficulty levels and covering 16 representative subjects. 

## **2.2 LMMs for Math** 

Recently, math-specific LLMs such as Llemma (Azerbayev et al., 2023) and Mathcoder (Wang et al., 2024) have demonstrated 

significant capabilities in text-only mathematical reasoning tasks and are competitive with general large language models like GPT-4 (OpenAI, 2023a; Zhou et al., 2024) and PaLM-2 (Anil et al., 2023b) at a much smaller scale. Notably, AlphaGeometry (Trinh et al., 2024) has shown impressive performance in solving challenging geometry problems, though it lacks the ability to process images and must rely on text descriptions of images. Current math-specific multimodal models, such as G-LLaVA (Gao et al., 2023), UniMath (Liang et al., 2023), MatCha (Liu et al., 2022), and UniChart (Masry et al., 2023), are primarily focused on plane geometry or chart-based problems and still lag behind general multimodal models such as GPT-4V (OpenAI, 2023b). To facilitate better evaluation of the mathematical reasoning abilities of multimodal models, we introduce our benchmark, featuring high-quality problems demanding expertise in both mathematical reasoning and visual recognition. 

## **2.3 Multimodal Foundation Models** 

With the success of LLMs, MiniGPT-4 (Zhu et al., 2023), LLaMA-Adapter (Zhang et al., 2023d), and LLaVa (Liu et al., 2023b,a) have attempted to construct multimodal models from text-only models. Subsequently, various large multimodal models such as GPT-4V (OpenAI, 2023b), Gemini (Team, 2023a), Qwen-VL (Bai et al., 2023), InternLMXComposer-VL (Zhang et al., 2023c; Dong et al., 2024), and SPHINX (Lin et al., 2023) have been created, demonstrating impressive performance on benchmarks like MMMU (Yue et al., 2023) and MathVista (Lu et al., 2023). Notably, some opensourced models have even received scores higher than GPT-4V by a significant margin on these benchmarks. However, on our benchmark, GPT-4V performs considerably better than the other models, showing that open-source models still lack generalization abilities, especially on questions very different from those in fine-tuning datasets. 

## **3 Dataset** 

## **3.1 Overview** 

We introduce the MATH-Vision (MATH-V) benchmark, a curated benchmark designed to evaluate the multimodal mathematical reasoning capabilities of foundation models across a wide range of mathematical tasks with visual contexts. We particularly emphasize the challenges posed by the need 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0004-00.png)


|**Statistic**|**Number**|
|---|---|
|Total questions|3,040|
|- multiple-choice questions|1,532 (50.4%)|
|- Free-form questions|1,508 (49.6%)|
|- Questions in the testmini set|304 (10.0%)|
|Multiple-choice questions|1,532|
|- Proportion of answers A|299(19.5%)|
|- Proportion of answers B|311(20.3%)|
|- Proportion of answers C|320(20.9%)|
|- Proportion of answers D|326(21.2%)|
|- Proportion of answers E|278(18.1%)|
|Unique number of images|3,472|
|Unique number of answers|370|
|Levels|5|
|Subjects|16|
|Source competitions|19|
|- Math Kangaroo|1,499|
|- UK (Grey,Pink,Junior,Senior)|535|
|- AMC (8,10,12), AIME|759|
|Maximum question length|280|
|Minimum question length|5|
|Average question length|42.3|



Figure 2: Levels, subjects and sources distribution of MATH-V. 

Table 1: Key statistics of MATH-V. The unit of question length is words. 

for both expert-level visual perception and deliberate reasoning with subject-specific knowledge. This challenge is exemplified in our tasks, which require the processing of various heterogeneous image types and necessitate a model’s proficiency in utilizing domain-specific knowledge to deeply comprehend both text and images, and to reason accordingly. This extends far beyond basic visual perception and calls for an advanced approach that combines sophisticated multimodal analysis with mathematical reasoning. The proposed benchmark encompasses 16 subjects over 5 levels of difficulty, including _Algebra, Analytic Geometry, Arithmetic, Combinatorial Geometry, Combinatorics, Counting, Descriptive Geometry, Graph Theory, Logic, Metric Geometry, Solid Geometry, Statistics, Topology, and Transformation Geometry_ . The questions in our benchmark were manually collected from 19 competitions. The division of difficulty levels is primarily based on the recommended grades of these competitions, and details are shown in Appendix D.2. 

The MATH-V benchmark consists of 3,040 ques- 

tions, and we also provide a smaller test set of 304 questions for quick evaluation. Detailed coverage, statistics and sources of MATH-V are presented in Table 1 and Figure 2. 

## **3.2 Data Collection** 

**Data collection** Our benchmark collection comprises two stages. In the first stage, we gather a list of various mathematical competitions, both international and national. The selection is based on the criterion that visual inputs should be frequently utilized in the competition questions to provide significant insights, and the questions should be categorizable into different levels, such as AMC8, AMC10, AMC12, and AIME. Competitions like the International Mathematical Olympiad (IMO) are excluded due to the excessive difficulty of the questions and the rare use of visual inputs. Consequently, we select 19 competitions from various countries. In the second stage, we filter through online resources to identify problems where questions are interspersed with images. We adhere strictly to copyright and licensing regulations, ensuring 

that we avoid data from sites that prohibit copying and redistribution. Given the concerns surrounding data contamination in LMMs, we prioritize selecting questions that lack readily available answers, such as those found in separate documents or pages. For questions formatted as PDFs, we use the Mathpix API to extract text and images. This approach allows us to curate a diverse collection of 9,000 questions from various sources, with 3,500 of them accompanied by image context. 

**Data curation** To improve the quality of our data, we undergo a four-stage data curation process with the help of 10 annotators (senior college students from science-related majors). In the first stage, we verify the alignment of text questions and images, as Mathpix might return them in an incorrect order. We also eliminate questions with missing text or incorrect images and those with private information or offensive content. All results are cross-verified between different annotators. 3,352 math-VQA data are left after this stage. In the second stage, we use lexical overlap and Levenshtein distance to identify potential duplicate questions. These suspected duplicates are then reviewed to identify and remove any duplication. As a result of these steps, we obtain the final 3,040 math-VQA data for our benchmark. In the third stage, we categorize the problems into 16 different subjects. Our annotators are divided into three groups, each independently labeling all the questions, and we also utilized GPT-4V (OpenAI, 2023b) and Gemini Pro (Team, 2023a) for subject advice. Annotations from various groups and models are consolidated through a majority voting system involving three expert annotators. In the final stage, we assign difficulty levels with reference to the grade requirements of the original competitions. 

## **3.3 Comparison with Existing Benchmarks** 

We describe the differences between MATH-V and 2 established benchmarks: MathVista (Lu et al., 2023) and MMMU (Yue et al., 2023). 

**Comparing with MathVista** MathVista is a comprehensive multimodal mathematical reasoning benchmark derived from 28 existing mathrelated multimodal datasets and 3 newly collected ones. However, within MathVista, approximately 20 datasets exhibit a trend where questions are annotated post-image collection by annotators, resulting in a relatively limited variability of questions, as discussed in Sec. 1. In contrast to MathVista, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0005-05.png)


**----- Start of picture text -----**<br>
State-of-the-Art Accuracy on Math Benchmarks<br>100<br>Multi-modal<br>Text-only<br>80<br>60<br>40<br>20<br>0<br>MATH-V (ours) MathVista MMMU Science MATH GSM8K<br>Benchmark<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 3: State-of-the-art performance and corresponding model (①: GPT-4V, ②: InternLM-XComposer2-VL7B, ③: GPT-4 Code Interpreter) of several mathematical benchmarks in **zero-shot** settings. Our MATH-V benchmark is markedly more challenging compared to existing text-only and multi-modal mathematical benchmarks. Results of other benchmarks are either from their official website or Papers With Code. 

all math-VQA data including both questions and images in MATH-V are newly collected from real math competitions, thus no additional annotation is required. This difference in obtaining question annotations is reflected in the average question length, which stands at 42.3 for MATH-V, significantly surpassing the 15.6 average of MathVista. A longer question length offers numerous advantages, particularly in facilitating richer mathematical reasoning by providing additional context. Further qualitative comparisons of math-VQA data between our dataset and MathVista are provided in Appendix C.1. Moreover, for MATH-V we also provide 18 fine-grained categorizations of different math subjects, which can gain deeper insights into individuals’ strengths and weaknesses in specific areas of mathematics, compared to MathVista’s 7. Finally, all problems within our datasets are labeled with one of the five levels of difficulty, while MathVista has only three levels, with a considerable portion of problems (37.7%) classified under the _not applicable_ category. 

**Comparing with MMMU** Different from MathVista and our MATH-V , MMMU (Yue et al., 2023) is designed to evaluate the multi-discipline multimodal understanding and reasoning abilities of LMMs with college-level problems. In terms of Math, it spans eight distinctive subjects with a total of 500 questions, whereas subjects like Opera- 

|Model|Overall|Alg<br>AnaG<br>Ari<br>CombG<br>Comb<br>Cnt<br>DescG<br>GrphT<br>Log<br>Angle<br>Area<br>Len<br>SolG<br>Stat<br>Topo<br>TransG|
|---|---|---|
|Random Chance|7.17|1.5<br>11.9<br>7.1<br>9.7<br>4.8<br>6.0<br>22.1<br>1.1<br>7.6<br>0.6<br>9.4<br>6.7<br>8.2<br>8.6<br>13.0<br>7.1|
|LLMs (Text Only)|||
|GPT4-CoT|8.16|12.8<br>10.7<br>15.7<br>4.9<br>10.7<br>10.5<br>1.9<br>5.6<br>8.4<br>8.1<br>6.2<br>8.7<br>8.6<br>3.5<br>4.4<br>4.8|
|LLMs (Text + Image Caption)|||
|ChatGPT 3.5-CoT<br>GPT 4-CoT|9.74<br>13.10|10.7<br>20.0<br>18.6<br>10.1<br>7.7<br>17.9<br>16.4<br>10.0<br>13.5<br>6.4<br>5.8<br>6.5<br>9.4<br>12.1<br>4.4<br>10.7<br>16.5<br>20.2<br>34.3<br>10.4<br>17.9<br>19.4<br>7.7<br>11.1<br>10.1<br>9.8<br>9.6<br>9.1<br>13.5<br>13.8<br>8.7<br>12.5|
|Open-source LMMs (Text + Image)|||
|LLaVA-v1.5-7B<br>SPHINX (V2)<br>ShareGPT4V-7B<br>LLaVA-v1.5-13B<br>ShareGPT4V-13B<br>SPHINX-MoE<br>InternLM-XComposer2-VL|8.52<br>9.70<br>10.53<br>11.12<br>11.88<br>14.18|7.0<br>7.1<br>10.7<br>7.1<br>4.8<br>10.5<br>7.7<br>10.0<br>9.2<br>15.6<br>10.2<br>9.8<br>5.3<br>8.6<br>4.4<br>4.8<br>6.7<br>7.1<br>12.9<br>7.5<br>7.7<br>6.0<br>9.6<br>16.7<br>10.1<br>11.0<br>11.8<br>12.5<br>8.2<br>8.6<br>8.7<br>6.0<br>5.5<br>3.6<br>12.9<br>10.1<br>4.8<br>7.5<br>11.5<br>14.4<br>10.9<br>16.2<br>11.8<br>12.3<br>9.8<br>15.5<br>17.4<br>11.3<br>7.0<br>14.3<br>14.3<br>9.1<br>6.6<br>6.0<br>13.5<br>5.6<br>13.5<br>10.4<br>12.6<br>14.7<br>11.5<br>13.8<br>13.0<br>10.7<br>7.5<br>15.5<br>16.4<br>10.7<br>8.9<br>9.0<br>11.5<br>8.9<br>7.6<br>11.6<br>13.0<br>17.4<br>10.3<br>8.6<br>8.7<br>12.5<br>7.8<br>17.9<br>14.3<br>15.6<br>9.5<br>11.9<br>12.5<br>15.6<br>12.6<br>16.2<br>15.6<br>17.8<br>13.5<br>12.1<br>8.7<br>16.1<br>9.3<br>15.5<br>12.1<br>15.3<br>11.3<br>10.5<br>14.4<br>22.2<br>19.3<br>19.7<br>15.6<br>15.0<br>11.9<br>15.5<br>26.1<br>15.5|
||14.54||
|Closed-source LMMs (Text + Image)|||
|Qwen-VL-Plus<br>Qwen-VL-Max<br>Gemini Pro<br>GPT4V|10.72|11.3<br>17.9<br>14.3<br>12.7<br>4.8<br>10.5<br>15.4<br>8.9<br>14.3<br>11.6<br>6.4<br>10.0<br>14.3<br>6.9<br>8.7<br>11.31<br>10.7<br>19.1<br>20.0<br>16.9<br>12.5<br>17.9<br>16.4<br>12.2<br>21.0<br>13.3<br>14.2<br>19.8<br>11.5<br>20.7<br>13.0<br>17.3|
||15.59||
||17.66|15.1<br>10.7<br>20.7<br>20.1<br>11.9<br>7.5<br>20.2<br>21.1<br>16.8<br>19.1<br>19.0<br>20.0<br>14.3<br>13.8<br>17.4<br>20.8|
||22.76|27.3<br>32.1<br>35.7<br>21.1<br>16.7<br>13.4<br>22.1<br>14.4<br>16.8<br>22.0<br>22.2<br>20.9<br>23.8<br>24.1<br>21.7<br>25.6|
|Human Performance|||
|Human (testmini)|75.66|57.9<br>79.0<br>100.0<br>100.0<br>47.4<br>94.7<br>89.5<br>63.2<br>63.2<br>36.8<br>52.6<br>73.7<br>89.5<br>89.5<br>100.0<br>73.7|



Table 2: Comparison of model performances across various mathematical subjects. Subjects: Alg: algebra, AnaG: analytic geometry, Ari: arithmetic, CombG: combinatorial geometry, Comb: combinatorics, Cnt: counting, DescG: descriptive geometry, GrphT: graph theory, Log: logic, Angle: metric geometry - angle, Area: metric geometry - area, Len: metric geometry-length, SolG: solid geometry, Stat: statistics, Topo: topology, TransG: transformation geometry. The first and second highest accuracy of LMMs are marked in red and blue, respectively. 

|Model|CoT|Overall|Alg<br>AnaG<br>Ari<br>CombG<br>Comb<br>Cnt<br>DescG<br>GrphT<br>Log<br>Angle<br>Area<br>Len<br>SolG<br>Stat<br>Topo<br>TransG|
|---|---|---|---|
|GPT4V|✗<br>✓|22.76<br>23.98<br>**+1.22**|27.3<br>32.1<br>35.7<br>21.1<br>16.7<br>13.4<br>22.1<br>14.4<br>16.8<br>22.0<br>22.2<br>20.9<br>23.8<br>24.1<br>21.7<br>25.6<br>26.7<br>26.2<br>38.6<br>22.1<br>24.4<br>19.4<br>27.9<br>23.3<br>25.2<br>17.3<br>21.4<br>23.4<br>23.8<br>25.9<br>4.4<br>25.6<br>**-0.58**<br>**-5.95**<br>**+2.86**<br>**+0.98**<br>**+7.73**<br>**+5.97**<br>**+5.76**<br>**+8.89**<br>**+8.4**<br>**-4.63**<br>**-0.8**<br>**+2.45**<br>-<br>**-1.72**<br>**-17.39**<br>-|
|Gemini Pro|✗<br>✓|17.66<br>14.34<br>**-3.32**|15.1<br>10.7<br>20.7<br>20.1<br>11.9<br>7.5<br>20.2<br>21.1<br>16.8<br>19.1<br>19.0<br>20.0<br>14.3<br>13.8<br>17.4<br>20.8<br>13.9<br>19.1<br>27.9<br>14.0<br>9.5<br>11.9<br>18.3<br>14.4<br>14.3<br>15.6<br>13.2<br>13.1<br>10.3<br>12.1<br>13.0<br>17.9<br>**-1.16**<br>**+8.34**<br>**+7.15**<br>**-6.17**<br>**-2.38**<br>**+4.48**<br>**-1.92**<br>**-6.67**<br>**-2.52**<br>**-3.47**<br>**-5.8**<br>**-6.9**<br>**-4.09**<br>**-1.72**<br>**-4.35**<br>**-2.97**|
|Qwen-VL-Max|✗<br>✓|15.59<br>12.63<br>**-2.96**|10.7<br>19.1<br>20.0<br>16.9<br>12.5<br>17.9<br>16.4<br>12.2<br>21.0<br>13.3<br>14.2<br>19.8<br>11.5<br>20.7<br>13.0<br>17.3<br>10.4<br>22.6<br>20.0<br>12.3<br>5.4<br>13.4<br>19.2<br>13.3<br>18.5<br>12.1<br>10.4<br>12.3<br>10.7<br>19.0<br>8.7<br>14.3<br>**-0.29**<br>**+3.57**<br>-<br>**-4.54**<br>**-7.14**<br>**-4.48**<br>**+2.88**<br>**+1.11**<br>**-2.52**<br>**-1.15**<br>**-3.8**<br>**-7.57**<br>**-0.82**<br>**-1.72**<br>**-4.34**<br>**-2.97**|
|InternLM-XComposer2-VL|✗<br>✓|14.54<br>14.37<br>**-0.17**|9.3<br>15.5<br>12.1<br>15.3<br>11.3<br>10.5<br>14.4<br>22.2<br>19.3<br>19.7<br>15.6<br>14.9<br>11.9<br>15.5<br>26.1<br>15.5<br>10.4<br>14.3<br>14.3<br>15.9<br>14.3<br>7.5<br>16.4<br>26.7<br>19.3<br>15.0<br>15.2<br>11.4<br>13.1<br>8.6<br>26.1<br>18.5<br>**+1.15**<br>**-1.19**<br>**+2.15**<br>**+0.65**<br>**+2.98**<br>**-2.99**<br>**+1.93**<br>**+4.45**<br>-<br>**-4.62**<br>**-0.4**<br>**-3.56**<br>**+1.22**<br>**-6.9**<br>-<br>**+2.97**|



Table 3: Influence of Chain-of-Thought on model performances across various mathematical subjects 

tion Research, Graph Theory, and Group Theory require sophisticated domain knowledge such as "DFS search" and "Cayley diagram". In contrast, our MATH-V focus on mathematics problems from elementary school to high school which contains little domain knowledge and can test the general visual perception and mathematical reasoning abilities of LMMs. Our dataset is 6 times larger than MMMU-Math, providing a significantly larger and more diverse set of challenges for analysis and evaluation. Specific to subjects, our datasets not only offer better coverage of subjects but also have more problems per subject on average (190 v.s. 63). Note that in 3/8 subjects (Linear Algebra, Group Theory, and Logic) covered by MMMU-Math, there are less than 10 problems. More qualitative comparisons can be found in Appendix C.2. 

## **4 Experiments** 

In this section, we conduct experiments to evaluate performance on the MATH-V dataset. Our findings indicate that accuracy remains low across even the most advanced models. We assess a variety of models, including Large Language Models (LLMs) and Large Multimodal Models (LMMs), and compare both closed- and open-source models, observing that closed-source models perform better. We also did a detailed error analysis of the best-performing model GPT-4V, delving into its error distribution and providing corresponding qualitative examples. 

## **4.1 Experimental Setup** 

**Models.** We conducted experiments on (a) LLMs: ChatGPT-3.5 (Brown et al., 2020) and GPT-4 (OpenAI, 2023a), augmented with image captions generated by GPT-4V, (b) Open-source LMMs: LLaVA- 

v1.5 (Liu et al., 2023b), SPHINX (Lin et al., 2023), ShareGPT-4V (Chen et al., 2023) and InternLMXcomposer2-VL (Dong et al., 2024), (c) Closedsource models: Qwen-VL (Bai et al., 2023), Gemini Pro (Team, 2023a) and GPT-4V (OpenAI, 2023b). 

**Implementation details.** Our evaluations are carried out in a zero-shot setting, both with and without Chain-of-Thought prompting (Wei et al., 2023) on several prominent, to determine the models’ ability to produce accurate answers without the need for fine-tuning or few-shot demonstrations on our benchmark. For human performance, we recruited 30 college or master students, and they were asked to solve the questions in the testmini subset of MATH-V independently, thereby obtaining the human performance baseline. Details can be found in G.3. For other models, we utilize the default prompts provided for multiple-choice or open-ended questions, when available. In cases where models do not offer prompts for the task types present in the MATH-V dataset, we engage in prompt engineering and select the most effective prompt for use in the zero-shot configuration during the main experiments. Detailed information about our prompts is available in the Appendix B. 

## **4.2 Experimental Results** 

In this section, we compare the performance of several best-performing Large Language Models (LLMs) and Large Multimodal Models (LMMs) on MATH-V, as detailed in Table 2 and Table 3. 

**Challenging nature of MATH-V.** The data presented in Table 2 highlights the challenging nature of MATH-V. Notably, the overall accuracy for the leading model, GPT-4V, is only 22.76%, which, while being the highest, still indicates a significant challenge in tackling the dataset. The difficulties are further emphasized by the performance of opensource models, all of which register accuracies below 15%. When compared with other benchmarks, as shown in Figure 3, the low accuracy of stateof-the-art models also underscores that MATH-V is challenging. However, the comparison of models’ performance with random chance (7.17% overall) demonstrates the promising potential of Large Language Models (LLMs) in addressing the challenges posed by MATH-V. Even LLaVA-v1.5-7B, an early baseline model not specifically enhanced for mathematical reasoning, surpasses random chance by 1.35%. These results indicate that these models are 

capable of learning and understanding the underlying mathematical concepts to a certain extent, even if they are not yet able to achieve high accuracy across the board. 

**Comparison among different subjects.** The dataset encompasses a wide range of mathematical subjects, from algebra to transformation geometry, each presenting its own set of complexities. For example, even the best-performing GPT-4V struggles to surpass 15% accuracy in subjects like counting, and Gemini Pro only scores 7.5%. This is in stark contrast to the relatively higher accuracy in arithmetic, where GPT-4V and Gemini Pro reach 35.7% and 20.7%, respectively. Such disparities across subjects and models not only highlight the models’ strengths in numeric calculation and weaknesses in handling specific types of mathematical problems that require more complex vision recognition and reasoning but also underline the diverse and intricate nature of the dataset. 

**Disparity of closed- and open-source models.** A significant performance disparity exists between closed- and open-source models. Closed-source models, such as GPT-4V and Gemini Pro, often outperform their open-source counterparts. For instance, GPT-4V achieved the highest overall accuracy of 22.76%, leading in subjects such as algebra (27.3%), analytic geometry (32.1%), and arithmetic (35.7%). Conversely, the best-performing open-source model, InternLM-XComposer2-VL, attained an overall accuracy of only 14.54%, with its highest subject accuracy in transformation geometry (15.5%). Contrary to these findings, results from MathVista show that the accuracy of InternLM-XComposer2-VL-7B is 7.7% and 12.4% higher than that of GPT-4V and Gemini Pro, respectively. This discrepancy not only highlights the excellent generalization capability of the closedsource state-of-the-art model but also suggests potential issues with data contamination. 

**Failure of CoT prompting on several models.** Chain-of-Thought (CoT) prompting did not consistently enhance performance across models. While GPT-4V benefited from CoT (22.76% _→_ 23.98%), other models like Gemini Pro and Qwen-VL-Max saw a decrease in performance when CoT was applied. Gemini Pro’s overall accuracy dropped by 3.32%, and Qwen-VL-Max saw a decrease of 2.96%. This suggests that the step-by-step reasoning capabilities of LMMs are still not mature and cannot be fully exploited by CoT prompting. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0008-00.png)


**----- Start of picture text -----**<br>
    Vision Recognition<br>Error<br>31.9%<br>Reasoning Error<br>42.2%<br>Knowledge<br>Error<br>15.1%   1.3%<br>6.9%<br>   2.6%<br>Calculation Error<br>Question Misunderstood Reject<br>Error to Answer<br>**----- End of picture text -----**<br>


Figure 4: Error distribution of 232 GPT-4V wrong results on the testmini subset of MATH-V. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0008-02.png)


**----- Start of picture text -----**<br>
Calculation Error<br>Question:  The diagram shows three concentric circles and two perpendicular,<br>common diameters of the three circles. The three grey sections are of equal area,<br>the small circle has radius 1. What is the product of the radii of the three circles?<br>Options:<br>Ground Truth: GPT4V:  [...] Now, do the same for Amiddle<br>and Alarge:<br>The product of the radius of the three<br>circles is:<br>Therefore, the answer is:<br>Error Reason:                                                                                    , so<br>                                                                                                .<br>GPT-4 possesses strong capabilities in numerical computing, yet its accuracy in<br>handling symbolic computing problems cannot be guaranteed.<br>**----- End of picture text -----**<br>


Figure 5: A typical calculation error. Complete answers and more examples can be found in the Appendix E. 

## **4.3 Error Analysis** 

**Quantitative and qualitative analysis on GPT4V.** In this section, we meticulously analyzed the causes of errors in GPT-4V through both quantitative and qualitative approaches, detailed in Figure 4 and Figure 5. The most prevalent cause of errors, accounting for 42.2%, was identified as Reasoning Error. This indicates a significant challenge in the model’s logical processing and inferential reasoning capabilities. Following closely, Vision Recognition Error was the second most common issue, responsible for 31.9% of errors. This highlights difficulties in accurately interpreting visual data, suggesting that the vision encoder remains a bottleneck in multi-modal models. Knowledge Error, making up 15.1% of the total errors, pointed to gaps in the model’s information database or its application. Furthermore, Calculation Error accounted for the smallest percentage of errors at 1.3%, indicating that GPT-4V is relatively robust in numerical and computational tasks, which historically have been challenging for language models. Notably, all errors in the Calculation Error category related to symbolic computation, as shown in Figure 5, rather than numerical computation, suggesting a potential area for future improvement. Detailed examples of these errors can be found in Appendix E. 

**Qualitative analysis on GPT-4 with image captions.** The integration of image captions generated by GPT-4V enables GPT-4 to reach a performance level of 13.10, comparable to that of SPHNIX and 

Qwen-VL. The quality of visual information produced by vision models plays a crucial role in enhancing accurate visual perception, which, in turn, significantly influences the final mathematical reasoning performance. An in-depth examination across 16 mathematical subjects reveals noteworthy insights. In disciplines such as algebra, analytic geometry, arithmetic, combinatorics, and counting, GPT-4 with image captions outperforms all other models except GPT-4V. Conversely, in other subjects, its performance notably lags behind these models. Through qualitative analysis of these categories, we observed that subjects like algebra and analytic geometry are more likely to receive relatively accurate descriptions from GPT-4V, unlike other disciplines. This discrepancy underscores the need for LMMs to enhance their capabilities in fine-grained image understanding and description. Remarkably, in some areas (e.g., combinatorics and counting), GPT-4 with image captions even surpasses GPT-4V. This success can be attributed to the former model’s more detailed image descriptions, which evidently contribute to a better understanding and solving of problems. Examples supporting this qualitative analysis are provided in the Appendix F. 

## **5 Conclusion** 

In this paper, we propose a benchmark dataset, called MATH-V, for evaluating the mathematical reasoning abilities of multimodal models. MATH-V consists of 3,040 challenging math reasoning problems, 

each accompanied by images and texts. We have carefully evaluated a variety of open-source and closed-source models using our MATH-V and conducted a systematic analysis. Our benchmark reveals a significant gap between human performance and that of current multimodal models in the task of multimodal reasoning, and facilitates more effective evaluation for future studies in this field. 

will be publicly available. The MATH-V dataset is not supposed, though the risk exists, to be used to train models for cheating. We intend for researchers to use this dataset to better evaluate LMMs’ mathematical reasoning capabilities and consequently facilitate future studies in this area. We bear all responsibility in case of violation of rights. The MATH-V data and our open-source code are under an MIT license. 

## **6 Limitations** 

The problems in our benchmarks are all in English. We did not include problems in other languages; therefore, the dataset cannot evaluate models’ abilities to reason in languages other than English. Another limitation is that we only included math problems and did not collect multimodal reasoning problems in other subjects, such as physics and chemistry. Also, the images in our benchmark are all abstract, with no pictures taken in real life. These limitations are important and warrant further investigation. 

## **7 Ethics statement** 

All questions of MATH-V are from publicly available sources. Although we do not commercialize MATH-V, we should like to demonstrate that we are far from the boundary for action or infringement. 

**Legal compliance.** MATH-V problems are created by the the Mathematical Association of America (MAA), Kangaroo of Mathematics in Austria (KMA) and United Kingdom Maths Trust (UKMT). For decades, the MAA has not protected its problem IP even from separate organizations which sell MAA problems, such as AoPS. Courts have ruled that this implies the IP rights are permanently forfeited. Both of the KMA and UKMT are leading charities that advances the education of young people in mathematics, and MATH-V only contains free test questions published on their official website. 

For MATH-V, we abide by Fair Use §107: “the fair use of a copyrighted work, including such use by _· · ·_ scholarship, or research, is not an infringement of copyright”, where fair use is determined by “the purpose and character of the use, including whether such use is of a commercial nature or is for nonprofit educational purposes” and “the effect of the use upon the potential market for or value of the copyrighted work.” 

**Dataset intended usage and license.** We document the MATH-V dataset within the paper and note that the MATH-V and code for results reproduction 

## **References** 

- Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol, Margaret Mitchell, C. Lawrence Zitnick, Devi Parikh, and Dhruv Batra. 2015. Vqa: Visual question answering. _International Journal of Computer Vision_ , 123:4 – 31. 

- Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, Daniele Mazzotta, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. 2023. The falcon series of open language models. 

- Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. 2019. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. _ArXiv_ , abs/1905.13319. 

- Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy GurAri, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, 

Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. 2023a. Palm 2 technical report. 

- Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023b. Palm 2 technical report. _arXiv preprint arXiv:2305.10403_ . 

- Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. 2023. Llemma: An open language model for mathematics. 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A versatile visionlanguage model for understanding, localization, text reading, and beyond. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901. 

- Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, and Liang Lin. 2021. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. _ArXiv_ , abs/2105.14517. 

- Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 2023. Sharegpt4v: Improving large multimodal models with better captions. _arXiv preprint arXiv:2311.12793_ . 

- Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu Lu, Zichong Yang, Kuei-Da Liao, et al. 2024. A survey on multimodal large language models for autonomous driving. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pages 958–979. 

- Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. 2024. Internlm-xcomposer2: Mastering 

free-form text-image composition and comprehension in vision-language large model. _arXiv preprint arXiv:2401.16420_ . 

- Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. 2023. G- llava: Solving geometric problem with multi-modal large language model. _ArXiv_ , abs/2312.11370. 

- Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. 

- Danna Gurari, Qing Li, Abigale Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. 2018. Vizwiz grand challenge: Answering visual questions from blind people. _2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 3608–3617. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Xiaodong Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. _ArXiv_ , abs/2009.03300. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ . 

- Yupan Huang, Zaiqiao Meng, Fangyu Liu, Yixuan Su, Nigel Collier, and Yutong Lu. 2023. Sparkles: Unlocking chats across multiple images for multimodal instruction-following models. 

- Zhenwen Liang, Tianyu Yang, Jipeng Zhang, and Xiangliang Zhang. 2023. Unimath: A foundational and multimodal mathematical reasoner. In _Conference on Empirical Methods in Natural Language Processing_ . 

- Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, Jiaming Han, Siyuan Huang, Yichi Zhang, Xuming He, Hongsheng Li, and Yu Qiao. 2023. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. 

- Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel Collier, and Julian Martin Eisenschlos. 2022. Matcha: Enhancing visual language pretraining with math reasoning and chart derendering. In _Annual Meeting of the Association for Computational Linguistics_ . 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. 

- Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chun yue Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. 2023. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. _ArXiv_ , abs/2310.02255. 

- Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. 2021. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In _Annual Meeting of the Association for Computational Linguistics_ . 

- Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. _ArXiv_ , abs/2203.10244. 

- Ahmed Masry, Parsa Kavehzadeh, Xuan Long Do, Enamul Hoque, and Shafiq R. Joty. 2023. Unichart: A universal vision-language pretrained model for chart comprehension and reasoning. _ArXiv_ , abs/2305.14761. 

- Shaghayegh Mobasher, Ghazal Zamaninejad, Maryam Hashemi, Melika Nobakhtian, and Sauleh Eetemadi. 2022. Parsvqa-caps: A benchmark for visual question answering and image captioning in persian. In _The 2022 Conference on Empirical Methods in Natural Language Processing_ . 

OpenAI. 2023a. Gpt-4 technical report. 

- OpenAI. 2023b. Gpt-4v(ision) system card. In _technical report_ . 

- Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. 2024. Code llama: Open foundation models for code. 

- Gemini Team. 2023a. Gemini: A family of highly capable multimodal models. 

- InternLM Team. 2023b. Internlm: A multilingual language model with progressively enhanced capabilities. https://github.com/InternLM/ InternLM-techreport. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, 

   - Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models. 

- Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, and Thang Luong. 2024. Solving olympiad geometry without human demonstrations. _Nature_ , 625:476– 482. 

- Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. 2024. Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning. In _The Twelfth International Conference on Learning Representations_ . 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models. 

- Zhenjie Yang, Xiaosong Jia, Hongyang Li, and Junchi Yan. 2023. Llm4drive: A survey of large language models for autonomous driving. 

- Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. _ArXiv_ , abs/2311.16502. 

- Ao Zhang, Yuan Yao, Wei Ji, Zhiyuan Liu, and Tat-Seng Chua. 2023a. Next-chat: An lmm for chat, detection and segmentation. 

- Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. 2023b. Vision-language models for vision tasks: A survey. 

- Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Wenwei Zhang, Hang Yan, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. 2023c. Internlmxcomposer: A vision-language large model for advanced text-image comprehension and composition. _arXiv preprint arXiv:2309.15112_ . 

- Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hong- 

sheng Li, and Yu Qiao. 2023d. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. 

- Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models. 

- Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, and Hongsheng Li. 2024. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. In _The Twelfth International Conference on Learning Representations_ . 

- Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ . 

## **Appendix** 

## **A Testmini** 

In this section, we show the effectiveness of the testmini subset in mirroring the performance trends observed in the full test set across a diverse range of mathematical subjects. The detailed results are presented in Table 4. We find that the model performances, when evaluated on both the entire set and testmini, exhibit close parallelism in trend. For instance, the performance delta between the full test set and testmini for the Qwen-VL-Plus model is a mere 0.19%, indicating a high degree of similarity in model behavior across the two datasets. Similarly, the GPT4V model also exhibits a negligible performance difference of -0.39%, further corroborating the testmini subset’s capability to accurately reflect the full test set’s characteristics. Furthermore, even for LLaVA-v1.5-13B and ShareGPT4V-7B, two models with the largest performance variations, where differences of 1.98% and 1.97% are observed respectively, the deviation remains within 2%, underscoring the testmini subset’s reliability for performance evaluation. These findings suggest that models’ performance on the testmini subset should be indicative of their performance on the full test set, thereby providing a reliable means for preliminary evaluation and comparison of models across various mathematical subjects. 

## **B Evaluation Details** 

We conduct all experiments on A100 80G GPUs. 

## **B.1 Prompts for Response Generation** 

In this section, we present details of the prompts used for the evaluation of response generation models. The prompts are designed to ensure that the models generate responses that are consistent with the input question and the image. The prompts used for the evaluation of response generation models are shown in Table 5. 

## **B.2 Prompt for Caption Generation** 

The prompt for caption generation is as follows: _Here is a diagram of a math problem, please describe the diagram in complete detail so that your description can be used to replace the diagram for math problem-solving._ 

## **B.3 Models** 

We present the source of the models used in the evaluation in Table 6. The models are sourced from OpenAI, Google, Alibaba, and Hugging Face. The models from OpenAI, Google and Alibaba are accessed from their respective platforms through API calls. 

## **C Comparison Details** 

## **C.1 Comparing with MathVista** 

Mathvista contains numerous similar questions with restricted question variability across a variety of source datasets. After eliminating questions with identical stem text, only 4740 questions remain. For instance, among the 6141 questions in MathVista, the question _"What is the age gap between these two people in the image?"_ appears 270 times. Some images of this question are shown in Figure 6. Furthermore, there are many template questions within its collection that merely alter a few words. For instance, questions containing _"What is the age gap between"_ appear 310 times. Another example is questions like _"Subtract all ... objects. How many objects are left?"_ . Figure 7 illustrates this type of question involving very similar geometry recognition, which occurs 400 times. In Table 7, we present detailed statistics on similar questions. 

We also provide some qualitative comparisons illustrating the different question lengths between MathVista and our dataset. Figure 9 presents the 3 main types of questions featuring an abstract scene in MathVista, constituting over 90% of the total questions, as discussed in Section. 1. We also showcase representative questions in MathVista that involve a function plot in Figure 10, which pose similar, straightforward, and concise inquiries regarding the depicted function graph. For comparison, we present examples of our MATH-V on analytic geometry in Figure 9, which feature longer questions and incorporate more complex function concepts, such as mutual symmetry and iterative functions. Moreover, Figure 11 and Figure 12 exhibit topology and graph theory-related questions in our dataset, two categories absent in MathVista, and require intricate visual recognition and mathematical reasoning. 

## **C.2 Comparing with MMMU** 

In this section, we showcase some examples of different subjects of MMMU demanding sophisticated 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0014-00.png)


Figure 6: Some images with the question "What is the age gap between these two people in the image?" from MathVista. Questions containing "What is the age gap between" appear 310 times. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0015-00.png)


**----- Start of picture text -----**<br>
Question Question Question Question Question<br>[No.3502] Subtract all small blue objects. Subtract all  [No.3784] Subtract all brown balls. Subtract all  [No.2093] Subtract all rubber cubes. Subtract all  [No.2490] Subtract all tiny blue objects. Subtract all  [No.1099] Subtract all big gray things. Subtract all<br>big cyan cylinders. How many objects are left? cubes. How many objects are left? small objects. How many objects are left? brown metallic spheres. How many objects are left? blue matte blocks. How many objects are left?<br>Question Question Question Question Question<br>[No.5667] Subtract all large yellow matte cylinders. Subtract all small shiny cubes. How many objects are left? [No.2927] Subtract all green cylinders. Subtract all green shiny things. How many objects are left? [No.4937] Subtract all cyan things. Subtract all shiny balls. How many objects are left? [No.4663] Subtract all red matte things. Subtract all small matte cylinders. How many objects are left? [No.1041] Subtract all tiny gray spheres. Subtract all yellow cubes. How many objects are left?<br>Question Question Question Question Question<br>[No.1565] Subtract all yellow things. Subtract all  [No.5408] Subtract all red metallic cylinders. Subtract  [No.3892] Subtract all cyan balls. Subtract all purple  [No.4723] Subtract all green shiny cubes. Subtract all  [No.1971] Subtract all cyan cylinders. Subtract all<br>brown balls. How many objects are left? all big purple objects. How many objects are left? cubes. How many objects are left? brown rubber objects. How many objects are left? small blue rubber blocks. How many objects are left?<br>Question Question Question Question Question<br>[No.4324] Subtract all gray things. Subtract all big metallic spheres. How many objects are left? [No.2941] Subtract all matte objects. Subtract all tiny cyan things. How many objects are left? [No.3112] Subtract all tiny purple metallic objects. Subtract all big matte balls. How many objects are left? [No.1941] Subtract all red cubes. Subtract all brown matte cylinders. How many objects are left? [No.3408] Subtract all brown objects. Subtract all large red rubber cylinders. How many objects are left?<br>Question Question Question Question Question<br>[No.4393] Subtract all cyan shiny balls. Subtract all big cyan matte cylinders. How many objects are left? [No.3018] Subtract all small cyan shiny balls. Subtract all large metal objects. How many objects are left? [No.1607] Subtract all red cylinders. Subtract all red things. How many objects are left? [No.2320] Subtract all rubber cylinders. Subtract all small blue metallic cylinders. How many objects are left? [No.1695] Subtract all small red metal spheres. Subtract all small brown blocks. How many objects are left?<br>Question Question Question Question Question<br>[No.4425] Subtract all big purple matte objects. Subtract all large brown matte things. How many objects are left? [No.6049] Subtract all cyan things. Subtract all large purple matte balls. How many objects are left? [No.5439] Subtract all big blue shiny cylinders. Subtract all red balls. How many objects are left? [No.2433] Subtract all red metallic cubes. Subtract all tiny metal spheres. How many objects are left? [No.6048] Subtract all small yellow rubber cubes. Subtract all blue rubber cylinders. How many objects are left?<br>Question Question Question Question Question<br>[No.4626] Subtract all cyan metallic blocks. Subtract  [No.5760] Subtract all large objects. Subtract all large  [No.2968] Subtract all blocks. Subtract all big blocks.  [No.2415] Subtract all yellow metallic cubes. Subtract  [No.5952] Subtract all balls. Subtract all small cyan<br>all tiny matte cylinders. How many objects are left? red objects. How many objects are left? How many objects are left? all small green objects. How many objects are left? shiny balls. How many objects are left?<br>Question Question Question Question Question<br>[No.1261] Subtract all gray metallic blocks. Subtract all small yellow things. How many objects are left? [No.2834] Subtract all cyan metallic spheres. Subtract all yellow rubber things. How many objects are left? [No.2899] Subtract all shiny things. Subtract all large metallic objects. How many objects are left? [No.4173] Subtract all big cyan rubber cylinders. Subtract all big green metal objects. How many objects are left? [No.4665] Subtract all brown rubber things. Subtract all spheres. How many objects are left?<br>**----- End of picture text -----**<br>


Figure 7: Problem involving very similar geometry recognition from MathVista. Questions like "Subtract all ... objects. How many objects are left?" appear 400 times. 

college-level domain knowledge. Figure 13, Figure 14, and Figure 15 present the subject of Graph Theory, Group Theory, and Operation Research, respectively. Moreover, typical examples of Topology and Graph Theory in our MATH-V are displayed in Figure 12 and Figure 12. Although both sets of subjects are intricate mathematical disciplines, they exhibit notable differences in the nature of their posed questions. Questions in the MMMU dataset often involve advanced college-level mathematical concepts such as Kruskal’s algorithm, alternating group structures, and DFS Spanning Tree. In contrast, the questions in our dataset predominantly comprise puzzles that are readily solvable by middle and high school students. 

## **D More Dataset Details** 

## **D.1 Question Distribution** 

All questions in MATH-V are written in English. As illustrated in Table 1, the average number of words in a question is 42.3, with a maximum of 280 words and a minimum of 5 words. Figure 16 further elucidates the distribution of word counts, highlighting the diverse patterns of questions. We can see that the median word count and the average word count are close, indicating that the distribution is not skewed. The distribution of word counts is near a normal distribution, with more than 98.75% of questions having a word count of less than 100. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0016-04.png)


**----- Start of picture text -----**<br>
80 Average<br>Median<br>70 Count<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>0 50 100 150 200 250<br>Question length<br>Count<br>**----- End of picture text -----**<br>


Figure 16: The distribution of the number of words per question in MATH-V. 

## **D.2 Division of Levels** 

In this section, we detail the division of MATH-V’s difficulty levels. Our difficulty classification is divided into two steps. The first step involves roughly dividing the difficulty levels based on the grades of 

the contestants as specified in the competition. The second step entails manually checking and correcting any unreasonable divisions. 

**Step 1.** The existing competition’s grade requirements span grades 1 to 12. Because the questions for grades 1 and 2 are considered too simple, we classify them separately as level 1. The questions for grades 3, 4, 5, and 6 are similar in difficulty, so we classify them as level 2. Grades 7 and 8 are categorized as level 3, grades 9 and 10 as level 4, and grades 11 and above as level 5. 

**Step 2.** We manually review the questions in each level and adjust the classification according to the actual difficulty of the questions. This step is necessary because, within the same exam, the difficulty range of the questions can be quite broad. For instance, the most difficult question in grade 4 might be more challenging than the easiest question in grade 5. In such cases, we make manual adjustments. 

Figure 17 shows a comparison of the level distribution between our MATH-V and the MATH dataset. We observe that both datasets have the fewest number of questions in level 1 and a similar number of questions in the other levels. Figure 1.b illustrates the models’ accuracy across different levels. It is evident that the accuracy of the models decreases as the difficulty level increases, which aligns with the expectation that the difficulty of the questions escalates with the level. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0016-12.png)


**----- Start of picture text -----**<br>
  MATH<br>  MATH-Vision<br>Level 1<br>Level 5<br>8.7%<br>Level 2<br>26.5%<br>7.8% 17.9%<br>22.5%<br>24.3%<br>27.3%<br>18.1%<br>24.3% 22.6%<br>Level 4 Level 3<br>**----- End of picture text -----**<br>


Figure 17: Comparison of the level distribution between our MATH-V and the MATH dataset (Hendrycks et al., 2021). 

## **D.3 Images by Subjects** 

In this section, we demonstrate some images from the problems in our dataset. Figure 19, Figure 20, Figure 21, Figure 22, Figure 23, Figure 24, Figure 25, Figure 26, Figure 27, Figure 28, Figure 29, Figure 30, Figure 31, Figure 32, Figure 33 and Figure 34 show images from the problems under the category of Algebra, Analytic Geometry, Arithmetic, Combinatorial Geometry, Combinatorics, Counting, Descriptive Geometry, Graph Theory, Logic, Metric Geometry - Angle, Metric Geometry - Area, Metric Geometry - Length, Solid Geometry, Statistics, Topology and Transformation Geometry, respectively. 

We can see that the images are diverse among different categories, and the images in the same category have similar visual patterns. For example, the images in the category of Statistics are mainly various types of charts and tables, while the images in the category of Arithmetic and Counting, which are categories mainly for primary school students, are more colorful and have more cartoon characters or real-world objects, and, conversely, the images in the category of Analytic Geometry are more complex and abstract. 

## **D.4 Introduction of Subjects** 

**Algebra.** Algebra is a branch of mathematics that deals with symbols and the rules for manipulating these symbols. It is a unifying thread of almost all of mathematics and includes everything from solving simple equations to studying abstractions such as groups, rings, and fields. Algebra allows us to express relationships and solve problems where we don’t initially know the values of certain quantities. It serves as the foundation for more advanced topics in mathematics and sciences, enabling the formulation and solving of equations and inequalities. 

**Analytic Geometry.** Analytic Geometry, also known as coordinate geometry, merges algebra and geometry to discuss shapes and figures using a coordinate system. This branch of mathematics allows the representation of geometric shapes in a numerical way by translating them into equations and vice versa. It enables the study of geometric properties and relationships by means of algebraic equations, thus providing a powerful tool for solving geometric problems. 

**Arithmetic.** Arithmetic is the most basic branch of mathematics, dealing with the study of numbers 

and the traditional operations on them: addition, subtraction, multiplication, and division. It forms the foundation upon which more advanced mathematical concepts are built. Arithmetic is used in everyday life for counting, measuring, and performing calculations. It is the first type of math that students are introduced to and remains fundamental for understanding more complex areas of mathematics. 

**Combinatorial Geometry.** Combinatorial Geometry is a field of mathematics that studies geometric objects and their arrangement, with a particular focus on discrete and combinatorial aspects. It involves the investigation of geometric structures and their properties under various combinatorial processes, such as counting, arrangement, and partitioning. This branch often intersects with areas like graph theory and combinatorics, exploring problems related to the configuration of points, lines, and other geometric figures. 

**Combinatorics.** Combinatorics is a branch of mathematics concerned with counting, arrangement, and combination of objects. It deals with questions of how many different ways such objects can be chosen, arranged, and combined under certain conditions. Combinatorics has applications in numerous areas, including computer science, optimization, and probability theory. It is fundamental in developing analytical skills and understanding complex mathematical concepts. 

**Counting.** Counting is the most basic and fundamental concept in mathematics, involving the enumeration of objects in a set. It serves as the foundation for more advanced mathematical concepts, such as arithmetic and combinatorics. Counting is not only about determining the total number of items in a collection but also includes understanding and applying principles like one-to-one correspondence, the counting principle, permutations, and combinations. 

**Descriptive Geometry.** Descriptive Geometry is the branch of geometry that allows the representation of three-dimensional objects in two dimensions, through the use of projections or perspective drawings. It is a method of visually communicating complex geometric shapes and is fundamental in fields such as engineering, architecture, and design. Descriptive geometry provides tools and techniques for solving spatial problems and for understanding the spatial relationships between objects. 

**Graph Theory.** Graph Theory is a branch of 

mathematics that studies graphs, which are mathematical structures used to model pairwise relations between objects. A graph is made up of vertices (or nodes) and edges (or arcs) that connect them. Graph theory is used to study networks and their properties, such as paths, cycles, and connectivity, and has applications in computer science, biology, transportation, social sciences, and more. 

**Logic.** Logic is the foundation of mathematical reasoning, focusing on the study of argument validity, structure, and form. It involves the analysis of propositions and their relationships to determine the truth values of statements. Logic is used to construct and analyze mathematical proofs, ensuring that conclusions follow logically from premises. It encompasses various systems, such as propositional logic, predicate logic, and modal logic, each with its own rules and applications. 

**Metric Geometry - Angle.** Metric Geometry concerning angles focuses on the properties and measurements of angles within various geometric figures. It involves understanding the types of angles (acute, right, obtuse, and straight), their relationships, and how they can be measured and used to determine other geometric properties. This aspect of metric geometry is crucial in solving problems related to shapes, patterns, and designs in both two and three dimensions. 

**Metric Geometry - Area.** Metric Geometry in the context of area deals with the measurement of the surface enclosed within geometric figures. It involves calculating the extent of two-dimensional shapes, such as squares, rectangles, triangles, and circles, using various formulas. Understanding area is essential in numerous practical applications, including architecture, land surveying, and crafting, where the determination of space is necessary. 

**Metric Geometry - Length.** Metric Geometry concerning length involves the measurement of distance between points in geometric figures. It is the foundation for determining the perimeter of shapes, the distance around a figure, and the concept of metric spaces in more advanced mathematics. Length measurements are fundamental in various real-world applications, such as construction, fabrication, and navigation. 

**Solid Geometry.** Solid Geometry is the branch of mathematics that deals with the study of threedimensional figures, including spheres, cubes, cylinders, and pyramids, among others. It involves understanding the properties, measurements, 

and relationships of these figures, such as volume, surface area, and the angles between intersecting planes. Solid geometry is essential in fields such as engineering, architecture, and physics, where three-dimensional models are crucial. 

**Statistics.** Statistics is the branch of mathematics that deals with collecting, analyzing, interpreting, and presenting data. It involves the use of theories of probability to make inferences about populations from samples. Statistics is applied in a wide range of disciplines, including economics, social sciences, biology, and engineering, to make decisions or predictions based on data analysis. 

**Topology.** Topology is a branch of mathematics concerned with the properties of space that are preserved under continuous transformations, such as stretching and bending, but not tearing or gluing. It is often referred to as "rubber-sheet geometry" because of its focus on the intrinsic properties of spaces. Topology has applications in various areas, including quantum physics, computer science, and biology, and deals with concepts like continuity, compactness, and connectedness. 

**Transformation Geometry.** Transformation Geometry involves the study of geometric figures and the changes they undergo through transformations such as translation, rotation, reflection, and dilation. It focuses on understanding how these transformations affect the properties and relations of shapes and figures. Transformation geometry is fundamental in understanding the symmetry, congruence, and similarity of geometric objects, and it has applications in areas such as computer graphics, robotics, and pattern recognition. 

## **E Error Examples of GPT-4V** 

In this section, we present detailed examples of errors made by GPT-4V. We categorize these errors into six types: reasoning error, vision recognition error, knowledge error, question misunderstood error, refusal to answer, and calculation error. 

## **E.1 Reasoning Error (42.2%)** 

Reasoning errors occur when GPT-4V fails to apply logical steps correctly. In Figure 35, the model incorrectly substitutes an unknown variable with a known value, leading to an incorrect conclusion. Figure 36 shows the model making an illogical hypothesis about the relationships of train wagon numbers. 

## **E.2 Vision Recognition Error (31.9%)** 

Vision recognition errors highlight the model’s struggles with interpreting visual data. Figure 37 depicts an instance where GPT-4V fails to read information from a simple bar chart. In Figure 38, the error occurs because the model mistakenly believes that 3, 6, and y are on the same side of the triangle. Figure 39 shows the model experiencing severe hallucinations, as there is no balance in the image at all. 

## **E.3 Knowledge Error (15.1%)** 

Knowledge errors are instances where GPT-4V lacks the necessary information to answer a question and provides information that is outdated or incorrect. Figure 40 and Figure 41 both showcase situations where the model did not know how to solve the problems. 

## **E.4 Question Misunderstood Error (6.9%)** 

These errors occur when GPT-4V misinterprets the user’s query. Figure 42 demonstrates a failure to grasp the query’s intent, resulting in an incorrect answer. Similarly, in Figure 43, the model mistakenly believes that the key below corresponds to the lock above, even though the question does not provide this information. 

## **E.5 Reject to Answer (2.6%)** 

Sometimes, GPT-4V refuses to provide an answer, which is categorized as a Reject to answer error. Figure 44 and Figure 45 illustrate instances where the model either cannot generate a response or chooses not to, citing various reasons. 

## **E.6 Calculation Error (1.3%)** 

Calculation errors are straightforward mistakes in numerical or symbolic calculations. The model only makes errors when performing symbolic calculations in our analysis. Both Figure 46 and Figure 47 show a simple symbolic computational mistake, undermining the model’s reliability in symbolic calculations. 

## **F Examples of GPT-4 with Image Captions** 

In this section, we provide some qualitative examples of GPT-4 with image captions, in comparison to other models like GPT-4V, Gemini Pro, and Qwen-VL-Max. 

## **F.1 Correct Examples** 

Some typical correct examples ( Figure 48, Figure 49, Figure 50, Figure 51, Figure 52, Figure 53, Figure 54) of GPT-4 with image captions on subjects like algebra, arithmetic, analytic geometry, counting, and combinatorics. Image captions are either accurate or have some small hallucinations. 

## **F.2 Wrong Examples** 

Some typical wrong examples (Figure 55, Figure 56, Figure 57, Figure 58, Figure 59, Figure 60) of GPT-4 with image captions on other subjects. Image captions are usually inaccurate to describe the image. 

## **G Details of Human Annotators** 

We hire human annotators for three tasks: 1. Verify the alignment of text questions and images; 2. Categorize the problems into 16 different subjects; 3. Evaluate human performance on MATH-V. 

## **G.1 Alignment of Text Questions and Images** 

We hire ten annotators in stage one, and all of them are senior college students, all majoring in sciencerelated disciplines. We pay them $0.35 per question, and they were asked to judge whether the text of the question matched the corresponding image and whether the image was complete. When the image is incomplete or incorrectly positioned, they are asked to re-capture the image from the original file. 

## **G.2 Subjects Classification** 

For task two, we have 10 annotators and three authors, to categorize the problems into 16 different subjects independently in three groups with our specially designed annotation tool, as illustrated in Figure 18. We ask them to read our definitions of the 16 subjects in advance as shown in Appendix D.4, and will randomly check their classification of certain topics. If there are mistakes in classification, we will promptly correct them and inform them. We pay $0.45 per question. Notably, we also utilize existing LMMs (such as Gemini and GPT-4V) for subject classification. Detailed instructions to annotators and prompts to models can be found in Table 8. Finally, we will combine the classification results of the three groups with the classification results returned by LMMs to give the final classification. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0020-00.png)


Figure 18: GUI for categorizing the problems into 16 different subjects. 

## **G.3 Human Performance** 

To obtain average human performance on our benchmark, we recruit 30 college or master students. We divided the 304 questions of testmini into 10 subsets, each containing 30 or 31 questions. Subsequently, we randomly allocated two subsets from this pool to each student. Participants were tasked with completing both subsets within a 70minute time frame, spread across two separate days. Note that this time limit closely aligns with that of the Kangaroo competition, where participants are assigned a similar duration for a test containing 30 problems. Each participant receives a one-time payment of $45 as compensation for their contribution. 

|Model|Overall|Alg<br>AnaG<br>Ari<br>CombG<br>Comb<br>Cnt<br>DescG<br>GrphT<br>Log<br>Angle<br>Area<br>Len<br>SolG<br>Stat<br>Topo<br>TransG|
|---|---|---|
|Random Chance<br>Testmini<br>∆|7.17<br>9.87<br>**+2.7**|1.5<br>11.9<br>7.1<br>9.7<br>4.8<br>6.0<br>22.1<br>1.1<br>7.6<br>0.6<br>9.4<br>6.7<br>8.2<br>8.6<br>13.0<br>7.1<br>0.0<br>15.8<br>10.5<br>15.7<br>0.0<br>0.0<br>36.84<br>0.0<br>15.8<br>0.0<br>10.5<br>21.1<br>5.3<br>10.5<br>15.8<br>0.0<br>**-1.5**<br>**+3.9**<br>**+3.4**<br>**+6.0**<br>**-4.8**<br>**-6.0**<br>**+14.74**<br>**-1.1**<br>**+8.2**<br>**-0.6**<br>**+1.1**<br>**+14.4**<br>**-2.9**<br>**+1.9**<br>**+2.8**<br>**-7.1**|
|LLMs(Text Only)|||
|GPT4-CoT<br>Testmini<br>∆|8.16<br>6.58<br>**-1.58**|12.8<br>10.7<br>15.7<br>4.9<br>10.7<br>10.5<br>1.9<br>5.6<br>8.4<br>8.1<br>6.2<br>8.7<br>8.6<br>3.5<br>4.4<br>4.8<br>5.3<br>10.5<br>15.8<br>0.0<br>21.1<br>10.5<br>5.3<br>0.0<br>5.3<br>10.5<br>5.3<br>0.0<br>5.3<br>5.3<br>5.3<br>0.0<br>**-7.5**<br>**-0.2**<br>**+0.1**<br>**-4.9**<br>**+10.4**<br>0.0<br>**+3.4**<br>**-5.6**<br>**-3.1**<br>**+2.4**<br>**-0.9**<br>**-8.7**<br>**-3.3**<br>**+1.8**<br>**+0.9**<br>**-4.8**|
|LLMs(Text + Image Caption)|||
|ChatGPT 3.5-CoT<br>Testmini<br>∆|9.74<br>10.20<br>**+0.46**|10.7<br>20.0<br>18.6<br>10.1<br>7.7<br>17.9<br>16.4<br>10.0<br>13.5<br>6.4<br>5.8<br>6.5<br>9.4<br>12.1<br>4.4<br>10.7<br>10.5<br>26.3<br>5.3<br>0.0<br>10.5<br>21.1<br>15.8<br>10.5<br>0.0<br>10.5<br>0.0<br>5.3<br>21.1<br>5.3<br>10.5<br>5.3<br>**-0.2**<br>**+6.3**<br>**-13.3**<br>**-10.1**<br>**+2.8**<br>**+3.2**<br>**-0.6**<br>**+0.5**<br>**-13.5**<br>**+4.1**<br>**-5.8**<br>**-1.2**<br>**+11.7**<br>**-6.8**<br>**+6.1**<br>**-5.4**|
|GPT 4-CoT<br>Testmini<br>∆|13.10<br>12.50<br>**-0.6**|16.5<br>20.2<br>34.3<br>10.4<br>17.9<br>19.4<br>7.7<br>11.1<br>10.1<br>9.8<br>9.6<br>9.1<br>13.5<br>13.8<br>8.7<br>12.5<br>15.8<br>10.5<br>31.6<br>5.3<br>15.8<br>31.6<br>10.5<br>15.8<br>15.8<br>0.0<br>5.3<br>5.3<br>0.0<br>21.1<br>10.5<br>5.3<br>**-0.7**<br>**-9.7**<br>**-2.7**<br>**-5.1**<br>**-2.1**<br>**+12.2**<br>**+2.8**<br>**+4.7**<br>**+5.7**<br>**-9.8**<br>**-4.3**<br>**-3.8**<br>**-13.5**<br>**+7.3**<br>**+1.8**<br>**-7.2**|
|Open-source LMMs(Text + Image)|||
|LLaVA-v1.5-7B<br>Testmini<br>∆|8.52<br>10.20<br>**+1.68**|7.0<br>7.1<br>10.7<br>7.1<br>4.8<br>10.5<br>7.7<br>10.0<br>9.2<br>15.6<br>10.2<br>9.8<br>5.3<br>8.6<br>4.4<br>4.8<br>0.0<br>10.5<br>15.8<br>5.3<br>5.3<br>15.8<br>10.5<br>10.5<br>15.8<br>21.1<br>15.8<br>15.8<br>5.3<br>10.5<br>0.0<br>5.3<br>**-7.0**<br>**+3.4**<br>**+5.1**<br>**-1.8**<br>**+0.5**<br>**+5.3**<br>**+2.8**<br>**+0.5**<br>**+6.6**<br>**+5.5**<br>**+5.6**<br>**+6.0**<br>0.0<br>**+1.9**<br>**-4.4**<br>**+0.5**|
|SPHINX (V2)<br>Testmini<br>∆|9.70<br>9.21<br>**-0.49**|6.7<br>7.1<br>12.9<br>7.5<br>7.7<br>6.0<br>9.6<br>16.7<br>10.1<br>11.0<br>11.8<br>12.5<br>8.2<br>8.6<br>8.7<br>6.0<br>5.3<br>10.5<br>10.5<br>0.0<br>21.1<br>10.5<br>10.5<br>15.8<br>10.5<br>15.8<br>5.3<br>10.5<br>0.0<br>5.3<br>5.3<br>10.5<br>**-1.4**<br>**+3.4**<br>**-2.4**<br>**-7.5**<br>**+13.4**<br>**+4.5**<br>**+0.9**<br>**-0.9**<br>**+0.4**<br>**+4.8**<br>**-6.5**<br>**-2.0**<br>**-8.2**<br>**-3.3**<br>**-3.4**<br>**+4.5**|
|ShareGPT4V-7B<br>Testmini<br>∆|10.53<br>12.50<br>**+1.97**|5.5<br>3.6<br>12.9<br>10.1<br>4.8<br>7.5<br>11.5<br>14.4<br>10.9<br>16.2<br>11.8<br>12.3<br>9.8<br>15.5<br>17.4<br>11.3<br>5.3<br>0.0<br>10.5<br>21.1<br>5.3<br>5.3<br>26.3<br>15.8<br>15.8<br>15.8<br>10.5<br>21.1<br>15.8<br>15.8<br>10.5<br>5.3<br>**-0.2**<br>**-3.6**<br>**-2.4**<br>**+11.0**<br>**+0.5**<br>**-2.2**<br>**+14.8**<br>**+1.4**<br>**+4.9**<br>**-0.4**<br>**-1.3**<br>**+8.8**<br>**+6.0**<br>**+0.3**<br>**-6.9**<br>**-6.0**|
|LLaVA-v1.5-13B<br>Testmini<br>∆|11.12<br>13.10<br>**+1.98**|7.0<br>14.3<br>14.3<br>9.1<br>6.6<br>6.0<br>13.5<br>5.6<br>13.5<br>10.4<br>12.6<br>14.7<br>11.5<br>13.8<br>13.0<br>10.7<br>10.4<br>5.3<br>15.8<br>5.3<br>10.5<br>10.5<br>26.3<br>5.3<br>15.8<br>5.3<br>15.8<br>31.6<br>10.5<br>15.8<br>15.8<br>10.5<br>**+3.4**<br>**-9.0**<br>**+1.5**<br>**-3.8**<br>**+3.9**<br>**+4.5**<br>**+12.8**<br>**-0.3**<br>**+2.3**<br>**-5.1**<br>**+3.2**<br>**+16.9**<br>**-1.0**<br>**+2.0**<br>**+2.8**<br>**-0.2**|
|ShareGPT4V-13B<br>Testmini<br>∆|11.88<br>13.49<br>**+1.61**|7.5<br>15.5<br>16.4<br>10.7<br>8.9<br>9.0<br>11.5<br>8.9<br>7.6<br>11.6<br>13.0<br>17.4<br>10.3<br>8. 6<br>8.7<br>12.5<br>15.8<br>21.1<br>10.5<br>5.3<br>15.8<br>10.5<br>15.8<br>10.5<br>15.8<br>5.3<br>10.5<br>36.8<br>21.1<br>5.3<br>10.5<br>5.3<br>**+8.3**<br>**+5.6**<br>**-5.9**<br>**-5.4**<br>**+6.9**<br>**+1.5**<br>**+4.3**<br>**+1.6**<br>**+8.2**<br>**-6.3**<br>**-2.5**<br>**+19.4**<br>**+10.8**<br>**-3.3**<br>**+1.8**<br>**-7.2**|
|SPHINX-MoE<br>Testmini<br>∆|14.18<br>13.49<br>**-0.69**|7.8<br>17.9<br>14.3<br>15.6<br>9.5<br>11.9<br>12.5<br>15.6<br>12.6<br>16.2<br>15.6<br>17.8<br>13.5<br>12.1<br>8.7<br>16.1<br>10.5<br>15.8<br>15.8<br>15.8<br>10.5<br>10.5<br>10.5<br>15.8<br>10.5<br>15.8<br>15.8<br>15.8<br>15.8<br>10.5<br>10.5<br>15.8<br>**+2.7**<br>**-2.1**<br>**+1.5**<br>**+0.2**<br>**+1.0**<br>**-1.4**<br>**-2.0**<br>**+0.2**<br>**-2.1**<br>**-0.4**<br>**+0.2**<br>**-2.0**<br>**+2.3**<br>**-1.6**<br>**+1.8**<br>**-0.3**|
|InternLM-XComposer2-VL<br>Testmini<br>∆|14.54<br>15.79<br>**+1.25**|9.3<br>15.5<br>12.1<br>15.3<br>11.3<br>10.5<br>14.4<br>22.2<br>19.3<br>19.7<br>15.6<br>15.0<br>11.9<br>15.5<br>26.1<br>15.5<br>10.5<br>15.8<br>10.5<br>15.8<br>10.5<br>10.5<br>15.8<br>21.1<br>21.1<br>21.1<br>15.8<br>15.8<br>10.5<br>15.8<br>26.3<br>15.8<br>**+1.2**<br>**+0.3**<br>**-1.6**<br>**+0.5**<br>**-0.8**<br>0.0<br>**+1.4**<br>**-1.1**<br>**+1.8**<br>**+1.4**<br>**+0.2**<br>**+0.8**<br>**-1.4**<br>**+0.3**<br>**+0.2**<br>**+0.3**|
|Closed-source LMMs(Text + Image)|||
|Qwen-VL-Plus<br>Testmini<br>∆|10.72<br>10.53<br>**-0.19**|11.3<br>17.9<br>14.3<br>12.7<br>4.8<br>10.5<br>15.4<br>8.9<br>14.3<br>11.6<br>6.4<br>10.0<br>14.3<br>6.9<br>8.7<br>11.31<br>26.3<br>10.5<br>10.5<br>15.8<br>10.5<br>21.1<br>5.3<br>10.5<br>10.5<br>10.5<br>5.3<br>10.5<br>10.5<br>5.3<br>5.3<br>0.0<br>**+15.0**<br>**-7.4**<br>**-3.8**<br>**+3.1**<br>**+5.7**<br>**+10.6**<br>**-10.1**<br>**+1.6**<br>**-3.8**<br>**-1.1**<br>**-1.1**<br>**+0.5**<br>**-3.8**<br>**-1.6**<br>**-3.4**<br>**-11.31**|
|Qwen-VL-Max<br>Testmini<br>∆|15.59<br>16.1<br>**+0.51**|10.7<br>19.1<br>20.0<br>16.9<br>12.5<br>17.9<br>16.4<br>12.2<br>21.0<br>13.3<br>14.2<br>19.8<br>11.5<br>20.7<br>13.0<br>17.3<br>10.5<br>21.1<br>21.1<br>15.8<br>15.8<br>15.8<br>15.8<br>10.5<br>21.1<br>10.5<br>15.8<br>21.1<br>10.5<br>21.1<br>15.8<br>15.8<br>**-0.2**<br>**+2.0**<br>**+1.1**<br>**-1.1**<br>**+3.3**<br>**-2.1**<br>**-0.6**<br>**-1.7**<br>**+0.1**<br>**-2.8**<br>**+1.6**<br>**+1.3**<br>**-1.0**<br>**+0.4**<br>**+2.8**<br>**-1.5**|
|Gemini Pro<br>Testmini<br>∆|17.66<br>17.11<br>**-0.55**|15.1<br>10.7<br>20.7<br>20.1<br>11.9<br>7.5<br>20.2<br>21.1<br>16.8<br>19.1<br>19.0<br>20.0<br>14.3<br>13.8<br>17.4<br>20.8<br>15.8<br>10.5<br>21.1<br>21.1<br>10.5<br>5.3<br>21.1<br>21.1<br>15.8<br>21.1<br>21.1<br>21.1<br>15.8<br>15.8<br>15.8<br>21.1<br>**+0.7**<br>**-0.2**<br>**+0.4**<br>**+1.0**<br>**-1.4**<br>**-2.2**<br>**+0.9**<br>0.0<br>**-1.0**<br>**+2.0**<br>**+2.1**<br>**+1.1**<br>**+1.5**<br>**+2.0**<br>**-1.6**<br>**+0.3**|
|GPT4V<br>Testmini<br>∆|22.76<br>22.37<br>**-0.39**|27.3<br>32.1<br>35.7<br>21.1<br>16.7<br>13.4<br>22.1<br>14.4<br>16.8<br>22.0<br>22.2<br>20.9<br>23.8<br>24.1<br>21.7<br>25.6<br>26.3<br>31.6<br>36.8<br>21.1<br>15.8<br>10.5<br>21.1<br>15.8<br>15.8<br>21.1<br>21.1<br>21.1<br>26.3<br>26.3<br>21.1<br>26.3<br>**-1.0**<br>**-0.5**<br>**+1.1**<br>-<br>**-0.9**<br>**-2.9**<br>**-1.0**<br>**+1.4**<br>**-1.0**<br>**-0.9**<br>**-1.1**<br>**+0.2**<br>**+2.5**<br>**+2.2**<br>**-0.6**<br>**+0.7**|



Table 4: Comparison of model performances between the entire set and testmini across various mathematical subjects. Subjects: Alg: algebra, AnaG: analytic geometry, Ari: arithmetic, CombG: combinatorial geometry, Comb: combinatorics, Cnt: counting, DescG: descriptive geometry, GrphT: graph theory, Log: logic, Angle: metric geometry - angle, Area: metric geometry - area, Len: metric geometry-length, SolG: solid geometry, Stat: statistics, Topo: topology, TransG: transformation geometry. The first and second highest accuracy of LMMs are marked in red and blue, respectively. 

|**Model**|**Prompt**|
|---|---|
|||
|GPT4-CoT<br>(Text Only)|Please solve the problem step by step and put your<br>answer in one~~"~~<br>~~"~~. If it is a multiple choice question,<br>only one letter is allowed in the~~"~~<br>~~"~~.|
|||
|ChatGPT 3.5-CoT<br>(Text + Image Caption )|Please frst solve the problem step by step, then put<br>your fnal answer or a single letter (if it is a multiple<br>choice question) in one~~"~~<br>".|
|||
|GPT4-CoT<br>(Text + Image Caption )|Please frst solve the problem step by step, then put<br>your fnal answer or a single letter (if it is a multiple<br>multiple choice question) in one~~"~~<br>~~"~~.|
|||
|LLaVa-v1.5-7B|Answer the question using a single word or phrase.|
|SPHINX (V2)<br>Answer the question using a single word or phrase.||
|||
|ShareGPT4V-7B|Answer the question using a single word or phrase.|
|||
|LLaVa-v1.5-13B|Answer the question using a single word or phrase.|
|||
|ShareGPT4V-13B|Answer the question using a single word or phrase.|
|||
|SPHINX-MoE|Answer the question using a single word or phrase.|
|||
|InternLM-XComposer2-VL|Please solve the problem and put your answer in one<br>"<br>~~"~~. If it is a multiple choice question, only one letter<br>letter is allowed in the~~"~~<br>~~"~~.|
|||
|InternLM-XComposer2-VL<br>-CoT|Please solve the problem step by step and put your<br>answer in one~~"~~<br>~~"~~. If it is a multiple choice question,<br>only one letter is allowed in the~~"~~<br>~~"~~.|
|||
|Qwen-VL-Plus|Answer the question using a single word or phrase.|
|||
|Qwen-VL-Max|Answer the question using a single word or phrase.|
|||
|Qwen-VL-Max-CoT|Please solve the problem step by step and put your<br>answer in one~~"~~<br>~~"~~. If it is a multiple choice question,<br>only one letter is allowed in the~~"~~<br>~~"~~.|
|||
|Gemini Pro|Answer the question using a single letter or number<br>or word.|
|||
|Gemini Pro-CoT|Please frst solve the problem step by step, then put<br>your fnal answer or a single letter (if it is a multiple<br>choice question) in one~~"~~<br>".|
|||
|GPT4V|Please solve the problem and put your answer in one<br>"<br>~~"~~. If it is a multiple choice question, only one letter<br>letter is allowed in the~~"~~<br>~~"~~.|
|||
|GPT4V-CoT|Please solve the problem step by step and put your<br>answer in one~~"~~<br>~~"~~. If it is a multiple choice question,<br>only one letter is allowed in the~~"~~<br>~~"~~.|



Table 5: The prompts used in the evaluation for response generation. 

|**Model**|**Source**|**URL**|
|---|---|---|
||||
|ChatGPT 3.5|gpt-3.5-turbo-0125|https://platform.openai.com/|
||||
|GPT4|gpt-4-0125-preview|https://platform.openai.com/|
||||
|GPT4V|gpt-4-1106-vision-preview|https://platform.openai.com/|
||||
|Gemini Pro|gemini-pro-vision|https://ai.google.dev/|
||||
|InternLM-XComposer2-VL|local checkpoint|https://huggingface.co/internlm/<br>internlm-xcomposer2-vl-7b|
||||
|Qwen-VL-Max|qwen-vl-max|https://help.aliyun.com/zh/<br>dashscope/developer-reference/<br>vl-plus-quick-start|
||||
|Qwen-VL-Plus|qwen-vl-plus|https://help.aliyun.com/zh/<br>dashscope/developer-reference/<br>vl-plus-quick-start|
||||
|SPHINX-MoE|local checkpoint|https://huggingface.co/<br>Alpha-VLLM/LLaMA2-Accessory/<br>tree/main/finetune/mm/SPHINX/<br>SPHINX-MoE|
||||
|SPHINX (V2)|local checkpoint|https://huggingface.co/<br>Alpha-VLLM/LLaMA2-Accessory/<br>tree/main/finetune/mm/SPHINX/<br>SPHINX-v2-1k|
||||
|LLaVa-v1.5-7B|local checkpoint|https://huggingface.co/<br>liuhaotian/llava-v1.5-7b|
||||
|LLaVa-v1.5-13B|local checkpoint|https://huggingface.co/<br>liuhaotian/llava-v1.5-13b|
||||
|ShareGPT4V-7B|local checkpoint|https://huggingface.co/Lin-Chen/<br>ShareGPT4V-7B|
||||
|ShareGPT4V-13B|local checkpoint|https://huggingface.co/Lin-Chen/<br>ShareGPT4V-13B|



Table 6: The source of the models used in the evaluation. 

|**Template**|**Proportion**|**Examples**|
|---|---|---|
|**IconQA**|||
||||
|Object Counting<br>Perimeter Calculation<br>Length Measurement with a given refer-<br>ence<br>Time Inquiry based on a clock|106/400<br>51/400<br>150/400<br>67/400|What number is shown?<br>What is the perimeter of the square?<br>Move the ruler to measure the length of<br>the line to the nearest centimeter. The line<br>is about _ centimeters long.<br>What time is shown?|
|**CLEVR-Math**|||
||||
|Subtract ... How many objects are left?|400/400|Subtract all metallic cylinders. Subtract all<br>gray things. How many objects are left?|
|**FigureQA**|||
||||
|Is X the ...?<br>Does X have the ...?<br>Is X ... Y?<br>Does X intersect Y?|176/400<br>86/400<br>117/400<br>21/400|Is Khaki the minimum/maximum/ low me-<br>dian/high median/smoothest/ roughest?<br>Does Brown have the maximum/ mini-<br>mum area under the curve?<br>(the high-<br>est/lowest value?)<br>Is Hot Pink less/greater than Black?<br>Does Slate intersect Light Salmon?|
|**Super-CLEVEL**|||
||||
|Are there ... X than Y?<br>Is the number of X ... than Y?<br>Others|146/400<br>232/400<br>22/400|Are there fewer/more tiny gray cars than<br>small red sedans?<br>Is the number of regular buss less/greater<br>than the number of brown rubber mini-<br>vans?<br>How many cyan double buss have the same<br>size as the matte aeroplane?|
|**KVQA**|||
||||
|What is the age gap between the ...?<br>How many people in the image were born<br>...?|310/382<br>72/382|What is the age gap between these two<br>people in image?<br>How many people in the image were born<br>after the end of World War II?|
|**PlotQA**|||
||||
|In how many years, is the X greater than<br>Y?<br>What is the difference between ...?<br>Others|57/105<br>31/105<br>17/105|In how many years, is the population in<br>the largest city greater than 26 %?<br>What is the difference between two con-<br>secutive major ticks on the Y-axis?<br>What is the average rating of statistical<br>capacity in Bolivia per year?|



Table 7: Some source datasets of MathVista featuring similar questions. The table shows the template, proportion within the whole dataset and typical examples of them. 

|_▷Object Counting (37%)_<br>**Image:**<br>**Question:**<br>What number is<br>shown?|_▷Length Measurement (38.5%)_<br>**Image:**<br>**Question:**<br>Move the ruler to<br>measure the length of thelineto<br>the nearestcentimeter. Theline<br>is about (_)centimeterslong.|_▷Time Inquiry (16.5%)_<br>**Image:**<br>**Question:** What time does the<br>clock show?|
|---|---|---|



Figure 8: Three main types of questions among the 375 questions with an abstract scene in MathVista: Object Counting, Length Measurement, and Time Inquiry. Among the 200 sampled questions, the percentage of each category is in the figure. Text marked in red indicates that it belongs to a fixed template and is used in other similar questions. 

_▷ mutual symmetry of functions ▷ quadratic function ▷ find roots of iterative discriminant functions_ **Image: Image: Image: Question:** The figure shows **Question:** In the (x,y)-plane graphs of functions _f_ and _g_ dethe coordinate axes are po- **Question:** The graph of the fined on real numbers. Each sitioned as usual. Point function _f_ ( _x_ ), defined for all graph consists of two perpendic- _A_ (1 _, −_ 10) which is on the real numbers, is formed by two ular halflines. Which is satisfied parabola _y_ = _ax_[2] + _bx_ + _c_ half-lines and one segment, for every real number _x_ ? was marked. Afterwards the as illustrated in the picture. (A) _f_ ( _x_ ) = _−g_ ( _x_ ) + 2 coordinate axis and the majorClearly, -8 is a solution of the (B) _f_ ( _x_ ) = _−g_ ( _x_ ) _−_ 2 ity of the parabola were deleted. equation _f_ ( _f_ ( _x_ )) = 0, because (C) _f_ ( _x_ ) = _−g_ ( _x_ + 2) Which of the following state- _f_ ( _f_ ( _−_ 8)) = _f_ ( _−_ 4) = 0. Find (D) _f_ ( _x_ + 2) = _−g_ ( _x_ ) ments could be false? all the solutions of the equation (E) _f_ ( _x_ + 1) = _−g_ ( _x −_ 1) (A) _a >_ 0 (B) _b <_ 0 ... _f_ ( _f_ ( _f_ ( _x_ ))) = 0. 

Figure 9: Three examples from MATH-V on analytic geometry, which are about mutual symmetry of functions, quadratic function discriminant, and roots of iterative functions. We omit some choices for brevity. 

|_▷Function symmetry, etc._<br>**Image:**<br>**Question1:** Is this an odd func-<br>tion?<br>(A) Yes (B) No<br>**Question2:** How many zeros<br>does this function have?<br>**Question3:** What is the value<br>of f(0)?<br>**Question4:** Is this function dif-<br>ferentiable at each point?<br>**Question5:** f(2) is ____ zero.<br>**Question6:** The derivative of<br>this function at x=0 is ____<br>**Question7:** What is this func-<br>tion most likely be?<br>(A) a polynomial<br>(B) a trigonometric function<br>(C) an exponential function<br>(D) a logarithmic function||||_▷Function roots, etc._<br>**Image:**<br>**Question1:**<br>How many zeros<br>does this function have?<br>**Question2:** What is the zero of<br>this function?<br>**Question3:** The derivative of y<br>at x=6 is ____ that at x=8.<br>**Question4:** What is the value of<br>y at x=2?<br>**Question5:** What is the value of<br>y at x=1?<br>**Question6:** If this function in-<br>vertible?<br>**Question7:**What is this function<br>most likely be?<br>A a polynomial<br>B a trigonometric function<br>C an exponential function<br>D a logarithmic function|
|---|---|---|---|---|
|||_▷Function monotonicity, etc._|||
|||**Image:**<br>**Question1:** Which function is<br>monotonic in range [0, pi]?<br>(A) the red one<br>(B) the blue one<br>(C) both<br>(D) none of them<br>**Question2:** At x=1, the value of<br>the red curve is ____ that of the<br>blue curve.<br>**Question3:** Which function is<br>even?<br>**Question4:** Which function is<br>odd?<br>**Question5:** Which function is<br>periodic?<br>**Question6:** What is the value of<br>the red curve at x=0?<br>**Question7:** What is the value of<br>the blue curve at x=0?|||



Figure 10: Typical examples of 400 questions with a function plot of MathVista. Each image has approximately seven questions about function symmetry, monotonicity, roots, periodicity, values, etc. 

_▷ String Knot Probability Puzzle_ 

**Image:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0027-02.png)


**Question:** A piece of string is lying on the table. It is partially covered by three coins as seen in the figure. Under each coin the string is equally likely to pass over itself like this: <image1> or like this: <image2>. What is the probability that the string is knotted after its ends are pulled? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0027-04.png)


**----- Start of picture text -----**<br>
▷ Village Map Sketching Challenge<br>Image:<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0027-05.png)


**Question:** During a rough sailing trip, Jacques tried to sketch a map of his village. He managed to draw the four streets, the seven places where they cross and the houses of his friends. The houses are marked on the correct streets, and the intersections are correct, however, in reality, Arrow Street, Nail Street and Ruler Street are all absolutely straight. The fourth street is Curvy Street. Who lives on Curvy Street? 

- (A) Adeline 

- (B) Benjamin 

- (C) Carole 

- (D) David 

- (E) It is impossible to tell without a better map 

_▷ Borromaic Rings_ 

## **Image:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0027-14.png)


**Question:** The "Borromaic Rings" have an extraordinary property. Although no two are interlocked, they are strongly connected within each other. If one ring is cut through, the other two fall apart. Which of the following diagrams shows the picture of "Borromaic Rings"? 

Figure 11: Three examples from MATH-V on topology. We omit some choices for brevity. 

## _▷ Adjacent Coloring Puzzle_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0028-01.png)


**----- Start of picture text -----**<br>
Image:<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0028-02.png)


**Question:** In the diagram we want to colour the fields with the colours A, B, C D so that adjacent fields are always in different colours. (Even fields that share only one corner, count as adjacent.) Some fields have already been coloured in. In which colour can the grey field be coloured in? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0028-04.png)


**----- Start of picture text -----**<br>
▷ Coloring Options ▷ Optimal Line Additions<br>Image: Image:<br>Question: The figure below Question: The picture shows<br>shows a ring made of six small seven points and the connec-<br>sections which you are to paint tions between them. What is<br>on a wall. You have four paint the least number of connecting<br>colors available and will paint lines that could be added to<br>each of the six sections a solid the picture so that each of<br>color. Find the number of ways the seven points has the same<br>you can choose to paint each of number of connections with<br>the six sections if no two adjacent other points? (Connecting<br>section can be painted with the lines are allowed to cross each<br>same color. other.)<br>**----- End of picture text -----**<br>


Figure 12: Three examples from MATH-V on graph theory. We omit some choices for brevity. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0028-06.png)


**----- Start of picture text -----**<br>
▷ Kruskal’s Algorithm ▷ Prüfer Code ▷ DFS Spanning Tree<br>Image: Image:<br>Image:<br>Question: Using Kruskal’s algo- Question:  Give the Prüfer code Question: Apply DFS Spanning<br>rithm find a minimal spanning of <image 1> Tree Algorithm to <image 1><br>tree of <image 1>. Show the list (A) 2,4,4,6,6 with vertex pre-ordering a, b, c,<br>of chosen edges only in alphabet (B) 2,3,3,6,6 e, i, h, g, d, f. Show the list of<br>order. Is ’AB, AF, BC, BG, DG, (C) 2,2,4,6,6 chosen edges only in alphabet or-<br>EF’ the correct answer? (D) 2,4,4,6,7 der.<br>(A) Yes (A) ab,be,ci,cd,ce,dh,fg,gh<br>(B) No (B) ab,ai,be,bf,cd,ci,fg,fh<br>**----- End of picture text -----**<br>


Figure 13: Three examples from the Graph Theory subfield in the Math subset of MMMU. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0029-00.png)


**----- Start of picture text -----**<br>
▷ Alternating Group ▷ Cayley Diagram ▷ Symmetric Group<br>Image: Image:<br>Image:<br>Question: Is <image 1> an Al- Question: Is <image 1> a Cay- Question: Is <image 1> a Sym-<br>ternating group? ley diagram? metric group?<br>(A) Yes (A) Yes (A) Yes<br>(B) No (B) No (B) No<br>**----- End of picture text -----**<br>


Figure 14: Three examples from the Group Theory subfield in the Math subset of MMMU. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0029-02.png)


**----- Start of picture text -----**<br>
▷ Graph Minimum Path ▷ Ford and Fulkerson ▷ Linear Programming<br>Algorithm Image:<br>Image:<br>Image:<br>Question: Find the minimum<br>path from v0 to v7 in the graph G Question:<br>of figure 1. Notice that it has no Question: Consider the net- Consider the following problem:<br>circuit whose length is negative. work shown in Fig. 1. The Minimize 2 x 1 + 3 x 2 Subject to<br><image 1> problem is to maximize the flow x 1+2 x 2  ≤ 2 ,  2 x 1 −x 2  ≤ 3 , x 2  ≥<br>(A) (v0, v1, v2, v3, v4, v5, v6, from node 1 to node 6 given 4 x 1 , x 2  ≥ 0. Solve by the graphi-<br>v7) with length 10 the capacities shown on the arcs. cal approach. <image 1><br>(B) (v0, v2, v3, v4, v6, v7) with Solve by Ford and Fulkerson al- (A) The problem has a unique fea-<br>length -12 sible solution.<br>gorithm. <image 1><br>(C) (v0, v1, v3, v5, v6, v7) with (A) 5 units (B) The problem has multiple fea-<br>length 8 sible solutions.<br>(B) 7 units<br>(D) (v0, v2, v4, v5, v6, v7) with (C) 9 units (C) The problem is infeasible.<br>length 6 (D) 11 units (D) The problem is unbounded.<br>**----- End of picture text -----**<br>


Figure 15: Three examples from the Operation Research subfield in the Math subset of MMMU. 

## _▷ Instructions for annotators on how to label questions into 16 categories_ 

We are constructing a dataset about mathematical problems in visual contexts, and we have collected quantities of questions that require categorization into different mathematical subjects. These categorizations are logic, algebra, counting, arithmetic, combinatorics, graph theory, topology, statistics, combinatorial geometry, solid geometry, metric geometry - angle, metric geometry - area, metric geometry - length, analytic geometry, descriptive geometry, and transformation geometry. Here are the detailed definitions of them: ... (definitions in Appendix D.4) 

If you are unsure about how to categorize the problem into these subjects, choose ’pending’ instead. 

## _▷ The prompt for models to get subject advice_ 

What branch of mathematics does the problem belong to? Choose from the following: logic, algebra, counting, arithmetic, combinatorics, graph theory, topology, statistics, combinatorial geometry, solid geometry, metric geometry - angle, metric geometry - area, metric geometry - length, analytic geometry, descriptive geometry, transformation geometry, pending. You are not supposed to solve the problem. Wrap your final answer, a word or a short phrase, in ” ~~”~~ . 

Table 8: Instructions for annotators on how to label questions into 16 categories and the prompt for models to get subject advice. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0031-00.png)


**----- Start of picture text -----**<br>
Algebra<br>**----- End of picture text -----**<br>


Figure 19: Some images from Algebra. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0032-00.png)


**----- Start of picture text -----**<br>
Analytic Geometry<br>**----- End of picture text -----**<br>


Figure 20: Some images from Analytic Geometry. 

## Arithmetic 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-20.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0033-22.png)


Figure 21: Some images from Arithmetic. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0034-00.png)


**----- Start of picture text -----**<br>
Combinatorial Geometry<br>**----- End of picture text -----**<br>


Figure 22: Some images from Combinatorial Geometry. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0035-00.png)


**----- Start of picture text -----**<br>
Combinatorics<br>**----- End of picture text -----**<br>


Figure 23: Some images from Combinatorics. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0036-00.png)


**----- Start of picture text -----**<br>
Counting<br>**----- End of picture text -----**<br>


Figure 24: Some images from Counting. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0037-00.png)


**----- Start of picture text -----**<br>
Descriptive Geometry<br>**----- End of picture text -----**<br>


Figure 25: Some images from Descriptive Geometry. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0038-00.png)


**----- Start of picture text -----**<br>
Graph Theory<br>**----- End of picture text -----**<br>


Figure 26: Some images from Graph Theory. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0039-00.png)


**----- Start of picture text -----**<br>
Logic<br>**----- End of picture text -----**<br>


Figure 27: Some images from Logic. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0040-00.png)


**----- Start of picture text -----**<br>
Metric Geometry - Angle<br>**----- End of picture text -----**<br>


Figure 28: Some images from Metric Geometry - Angle. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0041-00.png)


**----- Start of picture text -----**<br>
Metric Geometry - Area<br>**----- End of picture text -----**<br>


Figure 29: Some images from Metric Geometry - Area. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0042-00.png)


**----- Start of picture text -----**<br>
Metric Geometry - Length<br>**----- End of picture text -----**<br>


Figure 30: Some images from Metric Geometry - Length. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0043-00.png)


**----- Start of picture text -----**<br>
Solid Geometry<br>**----- End of picture text -----**<br>


Figure 31: Some images from Solid Geometry. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0044-00.png)


**----- Start of picture text -----**<br>
Statistics<br>**----- End of picture text -----**<br>


Figure 32: Some images from Statistics. 

## Topology 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0045-19.png)


Figure 33: Some images from Topology. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0046-00.png)


**----- Start of picture text -----**<br>
Transformation Geometry<br>**----- End of picture text -----**<br>


Figure 34: Some images from Transformation Geometry. 

## **Reasoning Error** 

**Question:** In triangle _ABC, AB_ = _AC, AE_ = _AD_ and angle _BAD_ = 30 _[◦]_ . What is the size of angle _CDE_ ? 

(A) 10 _[◦]_ 

(B) 15 _[◦]_ 

(C) 20 _[◦]_ 

(D) 25 _[◦]_ 

(E) 30 _[◦]_ **Ground Truth:** (B) 15 _[◦]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0047-07.png)


## **GPT4V Answer:** 

First, let’s review the given information: 

- Triangle _ABC_ is isosceles with _AB_ = _AC_ . - _AE_ = _AD_ . - ∠ _BAD_ = 30 _[◦]_ . 

Given that _AB_ = _AC_ , we know that ∠ _ABC_ = ∠ _ACB_ because the base angles of an isosceles triangle are equal. 

Since ∠ _BAD_ = 30 _[◦]_ and _AE_ = _AD_ , we have another isosceles triangle, _△ADE_ , and ∠ _ADE_ = ∠ _AED_ . Let’s call this unknown angle _θ_ . 

Because _AE_ = _AD_ , triangle _ADE_ is isosceles, which makes ∠ _ADE_ = ∠ _AED_ = _θ_ . Also, the sum of angles in a triangle is 180 _[◦]_ , so in _△ADE_ , we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0047-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0047-15.png)


So, both ∠ _ADE_ and ∠ _AED_ are 75 _[◦]_ . 

Now we move to _△CDE_ . We are looking for ∠ _CDE_ . The angle ∠ _CDE_ is part of ∠ _CDA_ , which is an exterior angle to _△ADE_ and equals the sum of the opposite interior angles, which we have found to be ∠ _ADE_ + ∠ _AED_ . 

Therefore: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0047-19.png)


Since ∠ _CDA_ is a straight line and is part of ∠ _CDE_ , the following relationship holds: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0047-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0047-22.png)


This is not amongst the given options, indicating that there’s an error. [Truncated Due to Length] 

Figure 35: An example of reasoning error. 

## **Reasoning Error** 

## **Question:** 

Two equal trains, each with 31 numbered wagons, travel in opposite directions. When the wagon number 7 of a train is side by side with the wagon number 12 of the other train, which wagon is side by side with the wagon number 11 ? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0048-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0048-04.png)


**----- Start of picture text -----**<br>
Ground Truth: 8<br>**----- End of picture text -----**<br>


## **GPT4V Answer:** 

To solve this problem, we’ll use the information that we have about the wagons that are currently side by side. We are told that wagon number 7 of one train is side by side with wagon number 12 of the other train. We can use this to establish a pattern or a relationship between the numbers of the wagons that are side by side. 

**If wagon 7 is next to 12, we can infer that as one train’s wagon number increases, the other train’s corresponding wagon number decreases if the trains are moving in opposite directions. The difference between the wagon numbers is 5 (12 - 7 = 5).** 

Now, if we want to find out which wagon would be side by side with wagon number 11, we can use the established difference to determine the opposite wagon number. We subtract 5 from 11 to find the matching wagon number on the other train: 11 _−_ 5 = 6 

Therefore, the wagon number 6 of the one train will be side by side with wagon number 11 of the other train. So, the solution is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0048-10.png)


**----- Start of picture text -----**<br>
6<br>**----- End of picture text -----**<br>


Figure 36: An example of reasoning error. 

## **Vision Recognition Error** 

**Question:** The students in Mrs. Sawyer’s class were asked to do a taste test of five kinds of candy. Each student chose one kind of candy. A bar graph of their preferences is shown. What percent of her class chose candy E? 

## **Ground Truth:** 2 

## **GPT4V Answer:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0049-04.png)


To find the percentage of the class that chose candy E, we first need to determine the number of students who chose candy E by looking at the bar graph. Then, we calculate the total number of students who participated in the taste test by adding the number of students who chose each kind of candy. Finally, we can find the percentage of students who chose candy E by dividing the number of students who chose candy E by the total number of students and then multiplying by 100. 

From the bar graph, the number of students who chose candy E can be counted directly: Candy E chosen by = 2 students 

Now let’s count the total number of students who participated: 

Candy A chosen by = 7 students Candy B chosen by = 8 students Candy C chosen by = 5 students Candy D chosen by = 3 students Candy E chosen by = 2 students (as identified earlier) Total number of students = 7 + 8 + 5 + 3 + 2 = 25 students 

Now we determine the percentage of students who chose candy E: 

Number choosing candy E Percentage choosing candy E = _×_ 100 � Total number of students � 

2 Percentage choosing candy E = _×_ 100 � 25 � Percentage choosing candy E = 25[2] _[×]_[ 100] 

Percentage choosing candy E = 0 _._ 08 _×_ 100 

Percentage choosing candy E = 8% 

Therefore, the percentage of the class that chose candy E is: 

8% 

Figure 37: An example of vision recognition error. 

## **Vision Recognition Error** 

**Question:** The sums of the all the three numbers on each side of the triangle are equal. Two numbers happened to be stained with ink. How much is the sum of these two numbers? 

## **Ground Truth:** 2 

## **GPT4V Answer:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0050-04.png)


To solve this problem, we need to realize that the sum of the numbers on each side of the triangle is the same. We will call this sum _S_ . 

Let’s denote the two unknown numbers, which are obscured by ink, as _x_ and _y_ . 

We can write the sum on each side of the triangle using these variables. For the side with the numbers 7, _x_ , and 1, the sum is 7 + _x_ + 1. **For the side with the numbers 3, 6, and** _y_ **, the sum is** 3 + 6 + _y_ **.** 

Because the sums are equal, we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0050-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0050-10.png)


We want to find the sum of _x_ and _y_ . Let’s add _y_ to both sides of the equation to move terms involving _y_ to the left side and terms involving _x_ to the right side: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0050-12.png)


Now subtract _y_ from both sides: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0050-14.png)


We know that the other side of the triangle is 1 + 6 + 3 = 10, so the sum of the numbers on each side must be 10, which means: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0050-16.png)


Now we substitute _y_ = 1 into the sum for the other sides to find _x_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0050-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0050-19.png)


So, the two numbers are _x_ = 2 and _y_ = 1, and their sum is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0050-21.png)


Thus, the sum of the two numbers that are stained with ink is 3, and so we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0050-23.png)


Figure 38: An example of vision recognition error. 

## **Vision Recognition Error** 

## **Question:** 

Karin wants to place five bowls on a table so that they are ordered according to their weight. She has already placed the bowls _Q, R, S_ and _T_ in order, where _Q_ is lightest and _T_ is heaviest. Where does she have to place bowl Z? 

(A) to the left of bowl Q 

(B) between bowls Q and R 

(C) between bowls R and S 

(D) between bowls S and T 

(E) to the right of bowl T 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0051-08.png)


**Ground Truth:** (B) 

## **GPT4V Answer:** 

The image depicts a series of bowls marked Q, R, S, T, and Z, with balance scales suggesting their comparative weights. To determine where bowl Z should be placed in the sequence, we examine the scale readings: 

- Bowl Q is lighter than all other bowls since it’s shown on the higher side of all the scales it’s compared with. - Bowl R is heavier than Q but lighter than the other bowls since it is on the lower side when compared with Q and on the higher side compared to S and T. - Bowl S is heavier than Q and R but lighter than T, as displayed by being on the lower side of the scales versus Q and R, and the higher side versus T. - Bowl T is the heaviest as it is on the lower side of every comparison. 

Bowl Z is shown compared only to bowl S. Z is on the lower side, suggesting it is heavier than S. Given that there’s no additional comparison provided, we cannot confirm Z’s weight relative to T; however, Z must be placed between S and T, as Z is heavier than S and there is no information to assume it is heavier than T. 

Therefore, the correct placement for bowl Z is between bowls S and T: 

_D_ 

Figure 39: An example of vision recognition error. Model is experiencing severe hallucinations in this example 

## **Knowledge Error** 

**Question:** We see in the diagram at the right a piece of the graphic of the function 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0052-02.png)


What is the value of _b_ ? 

## **Ground Truth:** -2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0052-05.png)


## **GPT4V Answer:** 

To determine the value of _b_ in the cubic function _f_ ( _x_ ) = _ax_[3] + _bx_[2] + _cx_ + _d_ , we need to look at the properties of the function’s graph provided in the diagram. 

From the graph, we can identify the coordinates of the two local extrema (maximum and minimum). At these points, the first derivative _f[′]_ ( _x_ ) of the function must be zero since the slope of the tangent line at these points is zero. 

The first derivative of the cubic function _f_ ( _x_ ) is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0052-10.png)


We have two extrema points visible on the graph, at _x_ = _−_ 1 and _x_ = 1. Let’s set up the derivative to be equal to zero at these points: 

For _x_ = _−_ 1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0052-13.png)


For _x_ = 1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0052-15.png)


Let’s solve this system of equations (1) and (2) for _b_ . Subtract (1) from (2): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0052-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0052-18.png)


Therefore, the value of _b_ is zero. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0052-20.png)


Figure 40: An example of knowledge error. 

## **Knowledge Error** 

**Question:** In the accompanying figure, segments _AB_ and _CD_ are parallel, the measure of angle _D_ is twice the measure of angle _B_ , and the measures of segments _AB_ and _CD_ are _a_ and _b_ respectively. Then the measure of _AB_ is equal to (A)[1] 2 _[a]_[ + 2] _[b]_ (B)[3] 2 _[b]_[ +][3] 4 _[a]_ (C) 2 _a − b_ (D) 4 _b −_[1] 2 _[a]_ (E) _a_ + _b_ **Ground Truth:** -2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0053-02.png)


## **GPT4V Answer:** 

The image you’ve provided appears to be a quadrilateral with sides _AB_ and _CD_ denoted as being parallel and segments _AB_ labeled as _a_ , and _CD_ as _b_ . However, without additional information such as angle measures or the relationship between the sides and angles, it is impossible to determine a definitive relationship between _a_ and _b_ based on the given options (A) through (E). 

Typically, in problems like this, you might apply properties of parallel lines and transversals or properties of similar triangles if any triangle relationships can be determined. But the given information is insufficient to apply these properties directly. 

Given the description of the problem, it seems to be expecting the application of a geometric principle or theorem which is not evident from the image alone. It’s possible that there is either additional information that has not been provided or there may be an error in the problem’s depiction. 

With the information currently at hand, it is not feasible to reach a solution, and I cannot provide an answer in the requested format ~~.~~ If you can provide further details or clarify the conditions of the problem, I would be happy to reattempt to solve it. 

Figure 41: An example of knowledge error. 

## **Question Misunderstood Error** 

## **Question:** 

Johann stacks 1 _×_ 1 cubes on the squares of a 4 _×_ 4 grid. The diagram <image1> shows how many cubes were piled on top of each other on each square of the grid. What will Johann see if he looks from behind (hinten) at the tower? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0054-03.png)


**Ground Truth:** (C) 

## **GPT4V Answer:** 

To solve this problem, we need to visualize the grid from the opposite side, which means flipping the given grid horizontally to see it from behind. The numbers represent the height of the towers of cubes. Let’s look at the grid and rewrite the cube heights as they would appear from behind: The original grid from the front is: 

4 2 3 2 3 3 1 2 2 1 3 1 1 2 1 2 2 3 2 4 2 1 3 3 1 3 1 2 2 1 2 1 

When viewed from behind, the grid should be: 

Now we need to match this to the options given: 

(A) Does not match because the first column (from left to right) should start with two blocks and end with four blocks. (B) Also does not match because the second column should start with three blocks. (C) Does not match because the third column should start with two blocks. (D) Does not match because the first and second columns have incorrect distributions. (E) Matches because it has the suitable pattern when flipped horizontally: 

- The first column starts with 2 and ends with 4, - The second column starts with 3 and ends with 3, - The third column starts with 2 and ends with 1, - The fourth column starts with 4 and ends with 1. 

Consequently, the correct answer is 

_E ._ 

Figure 42: An example of question understood error. 

## **Question Misunderstood Error** 

**Question:** Each one of the four keys locks exactly one padlock. Every letter on a padlock stands for exactly one digit. Same letters mean same digits. Which letters must be written on the fourth padlock? (A) GDA (B) ADG (C) GAD (D) GAG (E) DAD **Ground Truth:** (D) GAG 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0055-02.png)


## **GPT4V Answer:** 

To solve this problem, we need to match each letter (A, D, G) to a unique digit in the keys (1, 4, 7), given that each letter corresponds to exactly one digit and that same letters mean the same digits. 

**Let’s examine the first two padlocks and their keys:** 

**1. ADA corresponds to 141 2. DGA corresponds to 717** 

From the first padlock/key pair ADA = 141, we can establish that A = 1. Now, let’s apply that to the second padlock/key pair: 

If A = 1, then DGA = 717 would mean that D = 7 (since it’s the only remaining number in the second position where A is not). 

Now we know that: 

For the third padlock and key, we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0055-11.png)


Since we know D = 7 and A = 1, it must mean that G = 4 because it’s the remaining number and G has not been paired with a digit yet. 

So, we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0055-14.png)


Now, let’s see which combination matches our findings for the fourth padlock: Considering our assignments of A = 1, D = 7, and G = 4, the fourth padlock should be: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0055-16.png)


Therefore, the correct answer is: _C_ (C) GAD 

Figure 43: An example of question understood error. The model mistakenly believes that keys and locks are in one-to-one correspondence, but there is no such condition in the question. 

## **Reject to answer** 

**Question:** This picture <image1> shows a bracelet with pearls. Which of the bands below shows the same bracelet as above? **Ground Truth:** (E) **GPT4V Answer:** I cannot assist with this request. 

Figure 44: An example of reject to answer error. 

## **Reject to answer** 

**Question:** _A_ 10 cm long piece if wire is folded so that every part is equally long (see diagram). The wire is then cut through in the two positions marked. How long are the three pieces created in this way? (A) 2 cm _,_ 3 cm _,_ 5 cm (B) 2 cm _,_ 2 cm _,_ 6 cm (C) 1 cm _,_ 4 cm _,_ 5 cm (D) 1 cm _,_ 3 cm _,_ 6 cm (E) 3 cm _,_ 3 cm _,_ 4 cm **Ground Truth:** (B)[1] 2 _[a]_[2] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0056-05.png)


## **GPT4V Answer:** 

Your input image may contain content that is not allowed by our safety system. 

Figure 45: An example of reject to answer error. 

## **Calculation Error** 

**Question:** The diagram shows three concentric circles and two perpendicular, common diameters of the three circles. The three grey sections are of equal area, the small circle has radius 1. What is the product of the radii of the three circles? 

(A) ~~_√_~~ 6 (B) 3 (C)[3] _√_ 2 3 (D) 2 ~~_√_~~ 2 (E) 6 **Ground Truth:** (A) ~~_√_~~ 6 

## **GPT4V Answer:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0057-04.png)


The problem states that the three grey sections are of equal area, and we know the radius of the smallest circle to be 1. Let’s denote the radius of the middle circle as _r_ and the radius of the largest circle as _R_ . The area of a sector of a circle is given by[1] 2 _[θr]_[2][, where] _[ θ]_[ is the central angle in radians, and] _[ r]_[ is the radius of the] circle. Since the three grey sections are equal in area and shaped as quarter circles, each grey section has a central angle of _[π]_ 2[radians.] 

Therefore, the area of each grey section can be represented as follows: For the smallest circle: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0057-07.png)


For the middle circle, the area of the sector minus the area of the smaller circle’s quarter: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0057-09.png)


For the largest circle, the area of the sector minus the area of the middle circle’s quarter: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0057-11.png)


Since the areas are equal: _A_ small = _A_ middle = _A_ large We can set the expressions for _A_ small and _A_ middle equal to each other: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0057-13.png)


Now, do the same for _A_ middle and _A_ large: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0057-15.png)


Figure 46: Full example of calculation error. The wrong steps are highlighted in **red** . 

## **Calculation Error** 

**Question:** The diagram shows two adjoining squares with side lengths _a_ and _b_ (with _a < b_ ). How big is the area of the grey triangle? 

(A) ~~_√_~~ _ab_ 

- (B)[1] 2 _[a]_[2] 

(C)[1] 2 _[b]_[2] (D)[1] 4 � _a_[2] + _b_[2][�] (E)[1] 2 � _a_[2] + _b_[2][�] **Ground Truth:** (B)[1] 2 _[a]_[2] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0058-05.png)


## **GPT4V Answer:** 

The grey triangle in the diagram can be seen as the difference between the large square with side _b_ and the smaller square with side _a_ , minus the two right triangles that are outside of the grey area but inside the larger square. The area of the larger square is _b_[2] , and the area of the smaller square is _a_[2] . The two right triangles each have legs of length _a_ and _b − a_ , thus the area of one such right triangle is[1] 2 _[×][ a][ ×]_[ (] _[b][ −][a]_[)][.][There are two of these triangles, so] the total area of the two triangles is[1] 2 _[×][ a][ ×]_[ (] _[b][ −][a]_[)] _[ ×]_[ 2 =] _[ a][ ×]_[ (] _[b][ −][a]_[)][.] 

Now, the area of the grey triangle is therefore the area of the large square minus the area of the small square and the two right triangles: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0058-09.png)


We simplify this by factoring out a _b_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0058-11.png)


Now, we notice that _b − a_ is the height of the grey triangle, and, by looking at the diagram, we can see that the base of the grey triangle overlaps exactly with the side length of the larger square, which is _b_ . Therefore, the area of the grey triangle is[1] 2 _[×]_[ base] _[ ×]_[ height, which in this case is][1] 2 _[×][ b][ ×]_[ (] _[b][ −][a]_[)][:] 

Area of grey triangle = 2[1] _[×][ b][ ×]_[ (] _[b][ −][a]_[)] But since we can’t simplify this expression using the given variables directly, we must find a match among the options provided. By adding and subtracting[1] 2 _[a]_[2][within the expression of the larger square area] _[ b]_[2][, we can rewrite] the expression: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0058-14.png)


Therefore, the area of the grey triangle is equal to half the sum of the areas of the two squares minus half the area of the smaller square: 

Area of grey triangle =[1] 2[(] _[a]_[2][ +] _[ b]_[2][)] _[ −]_ 2[1] _[a]_[2] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0058-17.png)


This matches option (C), so the area of the grey triangle is[1] 2 _[b]_[2][.] Thus, the answer is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0058-19.png)


Figure 47: An example of calculation error. The wrong steps are highlighted in **red** . 

## **Example of Algebra** 

**Question:** The numbers in the five circles around each house add up to 20 . Some numbers are missing. Which number does the question mark stand for? **Ground Truth:** 9 

## **GPT4V Image Caption:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0059-03.png)


The diagram features two houses connected by a series of circles, each containing a number or a question mark. Each house has a large circle on its roof and two smaller circles on the ground, one on each side. 

The house on the left has the number 6 in the circle on its roof. The two smaller circles on the ground next to this house contain the numbers 2 and 5, respectively. The house on the right has the number 3 in the circle on its roof. **The circle on the ground to the right of this house contains the number 1, and the circle on the ground to the left of this house contains a question mark.** 

The circles are connected by lines, suggesting a relationship or a pattern that needs to be discovered. ... **Additionally, there is a line connecting the two houses, specifically from the circle with the number 5 to the circle with the question mark.** 

The goal of the problem is likely to determine the number that should replace the question mark, based on the pattern or relationship between the numbers in the connected circles. **GPT4 Answer:** 

To solve this problem, we need to find the pattern or relationship between the numbers in the connected circles, given that the sum of the numbers around each house adds up to 20. For the house on the left: The numbers we know are 6 (on the roof), 2, and 5 (on the ground). Let’s add these numbers: 6 + 2 + 5 = 13. Since the sum of the numbers around the house must be 20, we need to find the missing number that, when added to 13, equals 20. Let’s denote the missing number on the left side as _x_ . So, 13 + _x_ = 20. Solving for _x_ , we get _x_ = 20 _−_ 13 = 7. For the house on the right: The numbers we know are 3 (on the roof) and 1 (on the ground). We also know now that the missing number on the left side, which is connected to the house on the right, is 7. Let’s add these numbers: 3 + 1 + 7 = 11. Since the sum of the numbers around the house must be 20, we need to find the missing number that, when added to 11, equals 20. Let’s denote the missing number on the right side as _y_ . So, 11 + _y_ = 20. Solving for _y_ , we get _y_ = 20 _−_ 11 = 9. Therefore, the number that should replace the question mark is 9 . 

## **Gemini Pro Answer:** 

Let’s assume the missing numbers as x and y. The sum of the numbers in the circles around the first house = 6 + 2 + 5 + x = 13 + x. **The sum of the numbers in the circles around the second house = 3 + y + 1 = 4 + y.** Since the sum of the numbers in the circles around each house is 20, we have: **13 + x + 4 + y = 20; 17 + x + y = 20; x + y = 3. As the number in each circle is different, the possible values of x and y are (0, 3), (1, 2), and (2, 1). The question mark can be either** 0 **or** 2 **.** 

## **Qwen-VL-Max Answer:** 

**The sum of the numbers in the circles around the left house is** 6 + 3 + 2 + 5 = 16 **, so the number on the top right corner is** 20 _−_ 16 = 4 **. The sum of the numbers in the circles around the right house is** 3 + 4 + 1+? = 20 **, so the question mark stands for** 8 **.** 

Figure 48: An example of algebra question of GPT-4 with image caption. Although there are small hallucinations marked in red, The model correctly solves the problem based on the image description. Gemini Pro and Qwen-VLMax fail to solve it because of weak visual recognition and reasoning ability. 

## **Example of Arithmetic** 

## **Question:** 

Which of these clouds contain only numbers that are smaller than 7 ? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0060-03.png)


**Ground Truth:** (D) 

## **GPT4V Image Caption:** 

The diagram consists of five separate cloud-like shapes, each containing a set of numbers arranged in a specific pattern. Each cloud is labeled with a letter from (A) to (E). I will describe the arrangement of numbers within each cloud: 

(A) This cloud contains four numbers arranged in a diamond shape. The top number is 1, the number to the left is 6, the number to the right is 8, and the bottom number is 7. 

(B) This cloud also contains four numbers in a diamond pattern. The top number is 2, the left number is 5, the right number is 3, and the bottom number is 9. 

(C) The third cloud has a similar diamond arrangement with the top number being 2, the left number 6, the right number 4, and the bottom number 7. 

(D) This cloud’s diamond pattern has the number 3 at the top, 1 on the left, 2 on the right, and 5 at the bottom. 

(E) **The final cloud has a slightly different arrangement with five numbers. There are two numbers at the top: 8 on the left and 9 on the right. Below these, in the center, is the number 10. The bottom left number is 8, and the bottom right number is 7.** 

Each cloud seems to represent a puzzle or problem where the numbers are related to each other in some way, possibly through mathematical operations. The exact nature of the relationship or the rules governing the arrangement of numbers is not specified in the diagram. 

## **GPT4 Answer:** 

To solve this problem, we need to examine each cloud and determine if all the numbers it contains are smaller than 7. 

(A) This cloud contains the numbers 1, 6, 8, and 7. Since it contains 8 and 7, which are not smaller than 7, it does not meet the criteria. 

... 

(E) This cloud contains the numbers 8, 9, 10, 8, and 7. Since it contains numbers greater than 7, it does not meet the criteria. 

Therefore, the cloud that contains only numbers that are smaller than 7 is _D_ 

## **Gemini Pro Answer:** 

**(A) and (D) (A) 1, 6, 7, 8 (D) 1, 2, 3, 4, 5** 

## **Qwen-VL-Max Answer:** 

To determine which clouds contain only numbers that are smaller than 7, we need to look at each cloud and check if all the numbers inside it are less than 7. 

**... The clouds that contain only numbers that are smaller than 7 are** ( _C_ ) _,_ ( _D_ ) **.** 

Figure 49: An example of arithmetic question of GPT-4 with image caption. Although there are small hallucinations marked in red, The model correctly solves the problem based on the image description. GPT-4V also does it correct while Gemini Pro and Qwen-VL-Max fail because of weak visual recognition ability. 

## **Example of Analytic Geometry** 

## **Question:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0061-02.png)


**----- Start of picture text -----**<br>
A square is placed in a co-ordinate system as shown. Each point ( x | y ) of the square is deleted and replaced by the<br>point � x 1 [|] y [1] �. Which diagram shows the resulting shape?<br>Ground Truth: (C)<br>**----- End of picture text -----**<br>


## **GPT4V Image Caption:** 

**Image 1** : This diagram shows a Cartesian coordinate system with x and y-axes. The x-axis is labeled and ranges from 0 to 2, and the y-axis is labeled and ranges **from 0 to 2.5** . There is a square drawn in the first quadrant of the coordinate system. The square has its bottom left vertex at the point (1, 1) and its top right vertex at the point (2, 2), making the length of each side of the square 1 unit. **Image 2** : This image contains five separate Cartesian coordinate system diagrams labeled (A) through (E), each with x and y-axes ranging from 0 to 2 on both axes... (C) The shaded region is a smaller square within the first quadrant, with its bottom left vertex at approximately (0.5, 0.5) and its top right vertex at **approximately (1.5, 1.5)** ... **GPT4 Answer:** To solve this problem, we need to understand how the transformation � _x_ 1 _[|] y_[1] � affects the points of the square. The original square has vertices at (1 _,_ 1), (2 _,_ 1), (1 _,_ 2), and (2 _,_ 2). 1. Transform the bottom left vertex (1 _,_ 1): � 11 _[,]_[1] 1 � = (1 _,_ 1). This point remains unchanged. 2. Transform the bottom right vertex (2 _,_ 1): � 12 _[,]_[1] 1 � = � 12 _[,]_[ 1] �. It moves closer to the y-axis. 3. Transform the top left vertex (1 _,_ 2): � 11 _[,]_[1] 2 � = (1 _,_[1] 2[)][.][This point moves closer to the x-axis.] 4. Transform the top right vertex (2 _,_ 2): � 21 _[,]_[1] 2 � = � 12 _[,]_[1] 2 �. This point moves closer to both axes. After applying the transformation to all four vertices of the square, we can see that the new shape is also a square, but it is smaller and has its bottom left vertex at � 12 _[,]_[1] 2 � and its top right vertex at (1 _,_ 1). This matches the description of diagram (C), which shows a smaller square within the first quadrant, with its bottom left vertex at approximately (0 _._ 5 _,_ 0 _._ 5) and its top right vertex at approximately (1 _._ 5 _,_ 1 _._ 5). However, given the transformation, the top right vertex should precisely be at (1 _,_ 1), which still aligns best with the description of diagram (C) among the given options. **GPT-4V Answer:** Let’s analyze the transformation given by the function _f_ ( _x, y_ ) = � _x_ 1 _[,] y_[1] �. _f_ (1 _,_ 1) = � 11 _[,]_[1] 1 � = (1 _,_ 1); _f_ (1 _,_ 2) = � 11 _[,]_[1] 2 � = (1 _,_ 0 _._ 5); _f_ (2 _,_ 1) = � 12 _[,]_[1] 1 � = (0 _._ 5 _,_ 1); _f_ (2 _,_ 2) = � 21 _[,]_[1] 2 � = (0 _._ 5 _,_ 0 _._ 5). When we replace each point of the square with its transformed counterpart, the new square has vertices at (1, 1), (1, 0.5), (0.5, 1), and (0.5, 0.5). By comparing this to the diagrams shown in the images, we can see that the square has shrunk and is now located within the first quadrant closer to the origin. **Option (C) depicts an unchanged square, which does not correspond to the transformed square we calculated. Option (E) is the one that correctly represents a square with the calculated transformed vertices, which has shrunk and rotated towards the origin, remaining in the first quadrant.** 

Figure 50: An example of analytic geometry question of GPT-4 with image caption. Although there are small hallucinations marked in red, The model correctly solves the problem based on the image description while GPT-4V fails because of inaccurate image recognition. 

## **Example of Counting** 

**Question:** How many triangles can you find in the picture? 

## **Ground Truth:** 5 

## **GPT4V Image Caption:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0062-04.png)


The image depicts a stylized, geometric drawing of a humanoid figure that appears to be a representation of a girl or woman, composed of various simple shapes. The figure is not a real person but rather a collection of shapes that could be used in a math problem. Here is a detailed description of the figure: 

**Head:** The head is a white circle with a simple, smiling face. The face has two blue dots for eyes and a red, upside-down triangle for a nose. The mouth is a black, curved line like a smile. **Hair:** The figure has orange hair represented by two shapes on either side of the head, resembling pigtails. Each pigtail is a half-circle, and there is a pink bow on top of the head, consisting of a small circle in the center and two triangles on either side to represent the bow’s loops. **Body:** The body is a green trapezoid, wider at the top and tapering slightly towards the bottom, suggesting a dress or a tunic. There are two yellow circles in the middle of the trapezoid, aligned vertically, representing buttons. 

**Arms: The arms are two white rectangles extending from the upper sides of the trapezoid** , each ending in a white triangle to represent hands. 

**Skirt:** Below the trapezoid body, there is a green, inverted trapezoid that is narrower at the top and wider at the bottom, representing a skirt or the lower part of the dress. 

**Legs:** The legs are two white rectangles extending downward from the bottom of the skirt. **Shoes:** It has orange shoes, **each represented by a small rectangle at the end of the legs** . The entire figure is outlined in black, giving it a cartoonish, clip-art style appearance. The use of geometric shapes to construct the figure suggests that it could be used in a math problem involving geometry, such as calculating areas, perimeters, or identifying shapes. 

## **GPT4 Answer:** 

To solve the problem, let’s identify and count all the triangles in the described figure. The description mentions specific shapes that make up the figure, and we’ll focus on those identified as triangles or that can form triangles. 

1. **Nose:** The nose is described as a red, upside-down triangle. Count: 1 triangle 

2. **Hair Bows:** Each bow on top of the head consists of a small circle in the center and two triangles on either side to represent the bow’s loops. Count: 2 triangles (since there’s one bow mentioned, it has two triangles as part of its design) 

3. **Hands:** Each hand at the end of the arms is represented by a white triangle. Count: 2 triangles (one for each hand) 

4. **Implied Triangles:** The description does not explicitly mention other triangles, but we can check for any implied or composite triangles. Given the shapes described (circles for the head and buttons, rectangles for arms and legs, trapezoids for the body and skirt, half-circles for hair, and rectangles for shoes), there are no additional triangles formed by these shapes. Therefore, the total count of triangles in the figure, based on the description provided, is: 

   - 1 (nose) + 2 (hair bows)) + 2 (hands)) = 5 

- So, the answer is 5 triangles. 

Figure 51: An example of counting question of GPT-4 with image caption. Although there are small hallucinations marked in red, The model correctly solves the problem based on the image description. 

## **Example of Counting** 

**Question:** How many more bricks does the right hand pyramid have than the left hand pyramid? **Ground Truth:** 5 

## **GPT4V Image Caption:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0063-03.png)


The diagram shows two separate pyramid-like structures composed of blocks. Each pyramid is arranged in a way that resembles steps, with each upper row containing fewer blocks than the row below it. The blocks are depicted in a 3D perspective, giving the appearance of depth. 

The pyramid on the left has a total of 4 rows. From top to bottom, the number of blocks in each row increases by one. The top row has 1 block, the second row has 2 blocks, the third row has 3 blocks, and the bottom row has 4 blocks. 

The pyramid on the right is larger, with a total of 5 rows. Similar to the left pyramid, the number of blocks in each row increases by one as you move from top to bottom. The top row has 1 block, the second row has 2 blocks, the third row has 3 blocks, the fourth row has 4 blocks, and the bottom row has 5 blocks. 

Both pyramids are centered and have a symmetrical arrangement, ... 

## **GPT4 Answer:** 

To solve this problem, we need to calculate the total number of blocks in each pyramid and then find the difference between the two totals. 

For the left pyramid (4 rows): The number of blocks in each row forms an arithmetic sequence: 1, 2, 3, 4. The total number of blocks is the sum of this sequence: 1 + 2 + 3 + 4. 

For the right pyramid (5 rows): Similarly, the number of blocks in each row forms an arithmetic sequence: 1, 2, 3, 4, 5. The total number of blocks is the sum of this sequence: 1+2+3+4+5. Now, let’s calculate the sums: 

For the left pyramid: 1 + 2 + 3 + 4 = 10; For the right pyramid: 1 + 2 + 3 + 4 + 5 = 15. Finally, to find out how many more blocks the right pyramid has than the left pyramid, we subtract the total number of blocks in the left pyramid from the total number of blocks in the right pyramid: 15 _−_ 10 = 5. So, the right pyramid has 5 more blocks than the left pyramid. 

## **GPT-4V Answer:** 

For the left hand pyramid: The top row has 1 brick, the second row has 2 bricks, the third row has 3 bricks, and the bottom row has 4 bricks. So, the total number of bricks in the left hand pyramid is 1 + 2 + 3 + 4 = 10 bricks. 

For the right hand pyramid: **The top row has 1 brick, the second row has 3 bricks, the third row has 5 bricks, and the bottom row has 7 bricks. So, the total number of bricks in the right hand pyramid is** 1 + 3 + 5 + 7 = 16 **bricks.** 

**Now we find the difference:** 16 _−_ 10 = 6 **.** 

**Therefore, the right pyramid has** 6 **more bricks than the left pyramid.** 

## **Gemini Pro Answer:** 

**The left hand pyramid has 6 rows of bricks. The right hand pyramid has 7 rows of bricks. There are 7 - 6 = 1 more rows of bricks in the right hand pyramid. Each row of the left hand pyramid has 6 bricks. So there are 6 * 6 = 36 bricks in the left hand pyramid. Each row of the right hand pyramid has 7 bricks. So there are 7 * 7 = 49 bricks in the right hand pyramid. There are 49 - 36 = 13 more bricks in the right hand pyramid.** 13 

Figure 52: An example of counting question of GPT-4 with image caption. The model correctly solves the problem based on the accurate and detailed image description while GPT-4V and Gemini Pro fail. 

## **Example of Combinatorics** 

**Question:** The numbers 1 _,_ 2 _,_ 3 _,_ 4 and 9 are written into the squares on the following figure. The sum of the three numbers in the horizontal row, should be the same as the sum of the three numbers in the vertical column. Which number is written in the middle? **Ground Truth:** 9 

## **GPT4V Image Caption:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0064-03.png)


The diagram shows a geometric shape that resembles a cross or a plus sign (+). It is composed of five identical rectangles arranged such that one rectangle is at the center, and the other four are attached to each of its sides. The rectangles are aligned so that the longer sides of the rectangles are connected to the shorter sides of the central rectangle, forming a symmetrical cross shape. Each rectangle appears to have the same dimensions, suggesting that they could be squares, but without specific measurements, we cannot confirm their exact shape. The lines that form the rectangles are uniform, indicating that the boundaries of each rectangle are clearly defined. The overall shape could be interpreted as a 2D representation of an unfolded 3D box ... **GPT4 Answer:** 

Given the description of the figure and the problem, let’s denote the numbers in the squares as follows, assuming the central square is _X_ . And we know there’s a number 9 involved, which hasn’t been placed yet. Since the sum of the three numbers in the horizontal row should be the same as the sum of the three numbers in the vertical column, and we need to include all numbers 1 _,_ 2 _,_ 3 _,_ 4 _,_ and 9, let’s analyze the possible placements. 

Given the numbers, the sum involving 9 would be significantly higher than any sum without it, meaning 9 must be in either the horizontal row or the vertical column that includes _X_ . Since the figure is symmetrical and the problem specifies that the sums of the numbers in the horizontal row and vertical column must be equal, the only logical placement for 9 to balance the equation would be in the center, as _X_ . Thus, the number written in the middle is 9 . 

## **Gemini Pro Answer:** 

**Let’s call the numbers in the squares A, B, C, D, and E. We know that A + B + C = D + E. We also know that A + D = B + E. Adding these two equations together, we get 2A + B + C + D = 2B + 2E. Simplifying this equation, we get A + C = B + E. Since A + B + C = D + E, we can substitute A + C for D + E in this equation to get A + C = B + (A + C). Simplifying this equation, we get B = A. Therefore, the number in the middle is** _A_ **.** 

## **Qwen-VL-Max Answer:** 

Let’s call the number in the middle square _x_ . From the given information, we can set up the following equations: 

1 + 2 + _x_ = 9 + 4 _,_ 3 + 4 + _x_ = 1 + 2 + 9 **Solving for** _x_ **in each equation, we find that** _x_ = 10 **.** 

Figure 53: An example of combinatorics question of GPT-4 with image caption. The model correctly solves the problem based on the accurate and detailed image description while Gemini Pro and Qwen-VL-Max fail. 

## **Example of Combinatorics** 

**Question:** A hen lays white and brown eggs. Lisa takes six of them and puts them in a box as shown. The brown eggs are not allowed to touch each other. What is the maximum number of brown eggs Lisa can place in the box? 

**Ground Truth:** 3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0065-03.png)


## **GPT4V Image Caption:** 

The diagram consists of two distinct parts: 

1. On the left, there is a rectangle that contains six circles arranged in two rows of three. The circles are drawn such that each circle in the top row touches the two circles below it, and each circle in the bottom row touches the two circles above it. The circles are also touching their neighbors within the same row. The arrangement is such that the circles are packed closely together, maximizing the number of circles that can fit within the rectangle without overlapping. 2. On the right, there is an illustration of a chicken. The chicken is ... **GPT4 Answer:** 

1. **First Row** : Place a brown egg in the first position of the top row. To prevent any brown eggs from touching each other, the next egg in the top row must be white. Following this pattern, the third egg in the top row can be brown again. So, in the top row, we can place 2 brown eggs without them touching each other. 

2. **Second Row** : For the bottom row, directly below a brown egg in the top row, we must place a white egg to prevent brown eggs from touching. This means that if the first and third positions in the top row are brown, the first and third positions in the bottom row must be white. Following the pattern to avoid brown eggs touching, we can place a brown egg in the second position of the bottom row. This ensures no brown eggs are touching vertically or horizontally. 

So, in the bottom row, we can place 1 brown egg without it touching any other brown egg. Adding the brown eggs from both rows, we get a total of 2 + 1 = 3 brown eggs. 3 . 

## **GPT-4V Answer:** 

The problem involves arranging a maximum number of brown eggs in a box with six compartments without allowing the brown eggs to touch each other. To maximize the number of brown eggs while adhering to this constraint, we can place a brown egg in every other compartment to ensure no two brown eggs are adjacent to one another. 

Placing a brown egg in the first compartment, we then skip the second compartment and place another brown egg in the third compartment. **Repeating this pattern, we can see that we can place brown eggs in the first, third, and fifth compartments in the top row. For the bottom row, we can place brown eggs in the second, fourth, and sixth compartments.** 

Hence, the maximum number of brown eggs Lisa can place in the box without any two brown eggs touching each other is **three on the top row and three on the bottom row, which is a total of: 3 (top row) + 3 (bottom row) = 6 brown eggs. Therefore, the answer is:** 6 

## **Gemini Pro Answer:** 

**Lisa can place at most 2 brown eggs in the box. Since the brown eggs are not allowed to touch each other, they can only be placed in the corner spots. There are only two corner spots, so the maximum number of brown eggs Lisa can place in the box is** 2 **.** 

Figure 54: An example of combinatorics question of GPT-4 with image caption. The model correctly solves the problem based on the accurate and detailed image description while GPT-4V and Gemini Pro fail. 

## **Example of Topology** 

**Question:** Theresa moves a pencil along the line. She starts at the arrow shown. In which order will she go past the shapes? 

(A) ∆ _,_ □ _, •_ 

(B) ∆ _, •,_ □ 

(C) _•,_ ∆ _,_ □ 

(D) □ _,_ ∆ _, •_ 

(E) □ _, •,_ ∆ **Ground Truth:** A 

## **GPT4V Image Caption:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0066-08.png)


The diagram shows a complex, continuous, and entangled line that forms a path with several loops and curves. Along this path, there are four distinct markers: a solid black arrow, a solid black square, a solid black circle, and a solid black triangle. Each marker is connected to the line, indicating specific points along the path. Starting from the left, the path begins with the solid black arrow pointing to the right, indicating the starting point. The line then makes a small loop upwards and continues to the right before making a large loop counterclockwise. After completing this large loop, the line extends to the right and then makes a sharp turn downwards, forming a smaller loop in a clockwise direction. 

**Following this smaller loop, the line extends upwards and then to the right, where it encounters the solid black square. The path continues to the right, makes a small loop upwards, and then heads down towards the solid black circle. After the circle, the line makes a large loop that goes up, then to the left, and finally down, passing to the right of the circle.** 

**The path then makes a sharp turn upwards, creating a small loop, and continues to the left. It makes another loop downwards and to the right, passing below the previous loop and the solid black circle. The line then extends to the left, making a small upward loop before heading down and to the left, where it encounters the solid black triangle.** 

After the triangle, the line makes a final loop upwards and to the right, ending with a curl. The path does not have an explicit end point marked, as it finishes with the curl. 

The diagram does not provide any scale, units, or additional context, so the description is purely based on the visual elements present. The purpose of the markers and the path’s complexity suggest that the diagram could be related to a problem involving sequencing, following directions, or identifying patterns. 

## **GPT4 Answer:** 

Based on the description provided, Theresa starts at the solid black arrow and follows the path. Here’s the sequence of events as she moves along the path: 

1. Start at the solid black arrow: This is the starting point. 2. Small loop upwards: This is just a part of the path and doesn’t involve any of the shapes. 3. Large loop counterclockwise: Again, this part of the path doesn’t involve any of the shapes. 4. Sharp turn downwards forming a smaller loop in a clockwise direction: Still no shapes encountered. 5. Extends upwards and then to the right: No shapes yet. **6. Encounters the solid black square: This is the first shape she encounters. 7. Makes a small loop upwards and then heads down towards the solid black circle: The circle is the second shape encountered.** 8. Makes a large loop that goes up, then to the left, and finally down, passing to the right of the circle: This loop doesn’t lead to a new shape. 9. Sharp turn upwards creating a small loop and continues to the left: Still no new shapes. 10. Makes another loop downwards and to the right, passing below the previous loop and the solid black circle: This part of the path revisits the area near the circle but doesn’t introduce a new shape. **11. Extends to the left, making a small upward loop before heading down and to the left, where it encounters the solid black triangle: The triangle is the third shape encountered.** 12. Makes a final loop upwards and to the right, ending with a curl: This final part of the path doesn’t involve any new shapes. Therefore, the order in which Theresa goes past the shapes is: 

1. Solid black square 2. Solid black circle 3. Solid black triangle 

So, the final answer is: Square, Circle, Triangle 

## **GPT-4V Answer:** 

Following the line from the arrow, Theresa will first pass by the triangle (∆), then she will pass by the square (□), and finally past the circle ( _•_ ). So, the correct order in which she will go past the shapes is: ∆ _,_ □ _, •_ Therefore, the correct answer is: _A_ 

Figure 55: An example of topology question of GPT-4 with image caption. The model fails to solve the problem because of the inaccurate image description while GPT-4V succeeds. 

## **Example of Combinatorial Geometry** 

## **Question:** 

Which shape cannot be seen in every picture? **Ground Truth:** (D) 

## **GPT4V Image Caption:** 

The diagram consists of a series of five images labeled from (A) to (E), each containing a combination of geometric shapes within a rectangular frame. Below each image, there is a single shape corresponding to each label. Image (A) contains **three blue triangles** and two red circles arranged around a central green square. The triangles are pointing upwards, and the circles are placed near the corners of the rectangle, not touching the edges. The shape below image (A) is a single red circle. 

||Image (B) shows two blue triangles and**two red circles**. The triangles are pointing upwards, and the circles are placed<br>near the top corners of the rectangle, not touching the edges. The shape below image (B) is a single red circle, larger<br>than the one below image (A).<br>Image (C) contains**three blue triangles**, with two triangles pointing upwards and one pointing downwards. The<br>triangles are arranged in a horizontal line in the center of the rectangle. The shape below image (C) is a single blue<br>triangle pointing upwards.<br>Image (D) has**two green squares and one blue triangle**. The squares are placed on the left side of the rectangle, one<br>above the other, and the triangle is pointing upwards, located on the right side of the rectangle. The shape below image<br>(D) is a single green square.<br>Image (E) includes**one green square, one blue triangle pointing upwards, and one red circle**. The square is placed<br>on the left side, the triangle in the center, and the circle on the right side of the rectangle. The shape below image (E) is<br>a single blue triangle pointing upwards.<br>The diagram appears to be part of a pattern recognition or sequence completion exercise, where the task might be<br>to determine the rule governing the arrangement of shapes and then select the shape that would logically follow or<br>complete the sequence.<br>|Image (B) shows two blue triangles and**two red circles**. The triangles are pointing upwards, and the circles are placed<br>near the top corners of the rectangle, not touching the edges. The shape below image (B) is a single red circle, larger<br>than the one below image (A).<br>Image (C) contains**three blue triangles**, with two triangles pointing upwards and one pointing downwards. The<br>triangles are arranged in a horizontal line in the center of the rectangle. The shape below image (C) is a single blue<br>triangle pointing upwards.<br>Image (D) has**two green squares and one blue triangle**. The squares are placed on the left side of the rectangle, one<br>above the other, and the triangle is pointing upwards, located on the right side of the rectangle. The shape below image<br>(D) is a single green square.<br>Image (E) includes**one green square, one blue triangle pointing upwards, and one red circle**. The square is placed<br>on the left side, the triangle in the center, and the circle on the right side of the rectangle. The shape below image (E) is<br>a single blue triangle pointing upwards.<br>The diagram appears to be part of a pattern recognition or sequence completion exercise, where the task might be<br>to determine the rule governing the arrangement of shapes and then select the shape that would logically follow or<br>complete the sequence.<br>|
|---|---|---|
||**GPT4 Answer:**||
||To solve this problem, we need to identify which shape is not present in every image from (A) to (E). Let’s examine<br>each image:<br>**Image (A) contains: 3 blue triangles, 2 red circles, and 1 green square.**<br>**Image (B) contains: 2 blue triangles and 2 red circles.**<br>**Image (C) contains: 3 blue triangles.**<br>**Image (D) contains: 1 blue triangle and 2 green squares.**<br>**Image (E) contains: 1 blue triangle, 1 red circle, and 1 green square.**<br>Now, let’s identify which shapes are present in each image:<br>Blue triangles are present in all images (A) to (E). Red circles are present in images (A), (B), and (E). Green squares<br>are present in images (A), (D), and (E).<br>From this analysis, we can see that the shape that cannot be seen in every picture is the red circle, as it is missing<br>from images (C) and (D). Therefore, the answer is: Red Circle<br>.||
|||: Red Circle|
||||
||**GPT-4V Answer:**||
||The images display various combinations of shapes: circles, triangles, and squares in different arrangements. The<br>question is asking which shape is not present in all the images.<br>After examining each picture, we can see: Picture 1 contains circles, triangles, and a square. Picture 2 contains<br>circles and triangles. **Picture 3 contains circles and triangles. Picture 4 contains triangles and squares. Picture**<br>**5 contains circles, triangles, and a square.**<br>The square shape does not appear in all pictures;**it is absent in Picture 2 and Picture 3**.<br>So, the correct answer is: _D_||



Figure 56: An example of combinatorial geometry question of GPT-4 with image caption. The model fails to solve the problem because of the inaccurate image description. GPT-4V also misinterprets the image, but gives the correct answer. 

## **Example of Transformation Geometry** 

## **Question:** 

Jim and Ben are sitting in a ferris wheel (see picture on the right). The ferris wheel is turning. Now Ben is in the position where Jim was beforehand. Where is Jim now? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0068-03.png)


**----- Start of picture text -----**<br>
Ground Truth: (C)<br>**----- End of picture text -----**<br>


## **GPT4V Image Caption:** 

**Image 1** : This is a central diagram showing a complex balance scale system with a total of **seven** individual scales arranged in a tree-like structure. **At the top, there is a single scale labeled "BEN" with its fulcrum at the center and two pans hanging from either end. From each pan of this top scale, two additional scales are suspended, making a total of four scales on the second level. Each of these scales is also a balance with a fulcrum in the center and two pans hanging from either end. From each pan of the second-level scales, one more scale is suspended, resulting in three levels of scales with the bottom level consisting of four individual scales. Each of these bottom scales is labeled "JIM" and has a fulcrum with two pans. The entire structure is symmetric with respect to the central vertical axis. Image 2** : This image shows five different answer choices, labeled (A) through (E), each depicting a simplified version of the balance scale system from Image 1. Each choice shows a different configuration of the scales, with varying numbers of connections and arrangements. **(A) shows a central scale labeled "JIM" with three scales hanging from its left pan and two scales hanging from its right pan. (B) shows a central scale labeled "JIM" with two scales hanging from each pan. (C) shows a central scale labeled "JIM" with three scales hanging from its left pan and three scales hanging from its right pan. (D) shows a central scale labeled "JIM" with four scales hanging from its left pan and one scale hanging from its right pan. (E) shows a central scale labeled "JIM" with one scale hanging from its left pan and four scales hanging from its right pan.** 

## **GPT4 Answer:** 

To solve this problem, we need to understand the relationship between Jim and Ben in the context of the balance scale system described in Image 1 and then apply this understanding to the ferris wheel scenario. From Image 1, we know that: The top scale is labeled "BEN". **There are seven individual scales arranged in a tree-like structure. Each "JIM" scale is at the bottom level of this structure.** ...Given the information and the nature of the question, the solution focuses on understanding the relative movement and positions of Jim and Ben in a ferris wheel, drawing an analogy from the balance scale system. Without a direct question linking the ferris wheel scenario to the choices (A) through (E), the final answer is conceptual: **Jim is now in the position where Ben was beforehand. GPT-4V Answer:** To solve the problem we need to imagine the Ferris wheel rotating until Ben reaches Jim’s initial position. In the original image, if we assume the Ferris wheel is rotating clockwise, Ben will have to move **three quarters** of a full turn to get to Jim’s original position. Given that the wheel has **eight spokes** and thus divides the circle into equal parts, moving three-quarters means Jim will move six parts out from his original position if Ben moves into Jim’s original spot. So, Jim will be on the opposite side of the wheel, in the bottom position. Looking at the provided answer choices in image2, **option** _C_ **represents the position where Jim would be if the wheel rotates three quarters of a turn clockwise, with Ben taking Jim’s original spot.** 

Figure 57: An example of transformation geometry question of GPT-4 with image caption. The model fails to solve the problem because of the inaccurate image description. GPT-4V also misinterprets the image, but gives the correct answer. 

## **Example of Descriptive Geometry** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0069-01.png)


**Question:** Five equally big square pieces of card are placed on a table on top of each other. The picture on the side is created this way. The cards are collected up from top to bottom. In which order are they collected? (A) 5-4-3-2-1 (B) 5-2-3-4-1 (C) 5-4-2-3-1 (D) 5-3-2-1-4 (E) 5-2-3-1-4 

**Ground Truth:** E 

## **GPT4V Image Caption:** 

The diagram shows a composite rectangular figure made up of five smaller rectangles, each labeled with a number from 1 to 5. The overall shape is a larger rectangle. 1. Rectangle 1 is in the top left corner. **It is a vertical rectangle.** 2. Rectangle 2 is below Rectangle 1. It is a larger vertical rectangle, taking up the entire left side of the larger rectangle. 3. Rectangle 3 is in the center, bordering the right side of Rectangle 1 and the top side of Rectangle 2. It is a horizontal rectangle. . Rectangle 4 is a small square that is placed to the right of Rectangle 1 and above Rectangle 3. 5. Rectangle 5 is a large square that occupies the bottom right corner of the larger rectangle. It borders the right side of Rectangle 2 and the bottom side of Rectangle 3 and Rectangle 4. The rectangles are shaded in different shades of gray, possibly to distinguish them from one another. The numbers are placed in the center of each corresponding rectangle. The arrangement of the rectangles suggests that the dimensions of some could be inferred if the dimensions of others are known, which is a common setup for problems involving area and perimeter. 

## **GPT4 Answer:** 

To solve this problem, we need to determine the order in which the cards (represented as rectangles in the description) are collected based on their arrangement in the composite figure. The key is to understand that the card on top in the figure will be the last one collected, and the card at the bottom will be the first one collected, as they are collected from top to bottom. 1. **Rectangle 5** is the large square at the bottom right corner. It does not lie underneath any other rectangle, meaning it is on top of all others in its area. Thus, it will be the last one collected. 2. **Rectangle 4** is the small square to the right of Rectangle 1 and above Rectangle 3. **Since it lies on top of Rectangle 3 and does not have any other rectangles on top of it, it will be collected before Rectangle 5 but after the rectangles it covers (Rectangles 1 and 3).** 

...... 

Therefore, the order in which the cards are collected from top to bottom is: 5 _,_ 4 _,_ 3 _,_ 1 _,_ 2 . 

## **GPT-4V Answer:** 

To solve the problem, we look at which cards are on top of others. To collect the cards from top to bottom as described, we should look for the one that is not covered by any other card, then the next, and so on. Starting from the top, **we can see that card 5 is at the bottom, as all other cards are on top of it in some part. Next, we see that card 2 is on top of card 5 but underneath cards 1 and 3. Then, card 3 is on top of both cards 2 and 5 but below cards 1 and 4. Card 4 is only underneath card 1. Finally, card 1 is at the top with no cards above it.** 

...... 

Therefore, the correct sequence of collecting the cards from top to bottom is represented by option (E), when we reverse that sequence (since they are collected from top to bottom): 4-1-32-5. So the answer is: 

_E_ 

Figure 58: An example of descriptive geometry question of GPT-4 with image caption. The models fail to correctly interpret the spacial relationships between the cards. 

## **Example of Logic** 

**Question:** Maria wants to write whole numbers in the squares of the figure, so that the sum of the numbers in three consecutive squares is always 10. She has already written a number. What number should she write on the gray square? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0070-02.png)


**Ground Truth:** 2 

## **GPT4V Image Caption:** 

The diagram shows a horizontal rectangular bar divided into five equal-sized squares. The bar has a border around it, giving it the appearance of a single entity with internal divisions. The first square on the left is filled with the number 2 in a bold font. The second square is empty, the third square is shaded in a solid color, and the fourth and fifth squares are empty. The shaded square suggests it may have a particular significance in the context of the math problem, such as representing a specific value or condition that needs to be determined. The sequence and the values in the squares are likely part of the problem that needs to be solved. 

## **GPT4 Answer:** 

To solve this problem, let’s denote the numbers in the squares from left to right as _a_ , _b_ , _c_ , _d_ , and _e_ . According to the description, _a_ = 2, and _c_ is the number in the gray square that we need to find. We are given that the sum of the numbers in any three consecutive squares is always 10. Given this, **we can set up equations based on the sums of three consecutive squares: 1.** _a_ + _b_ + _c_ = 10 **2.** _b_ + _c_ + _d_ = 10 **3.** _c_ + _d_ + _e_ = 10 We know that _a_ = 2, so we can substitute this into the first equation to find _b_ + _c_ : 1. 2 + _b_ + _c_ = 10 2. _b_ + _c_ = 8 However, we need to find _c_ , and we don’t have direct values for _b_ , _d_ , or _e_ . But we can notice that since the sum of any three consecutive numbers is 10, and _a_ = 2, the only way to satisfy these conditions is to ensure that the sequence allows for this constant sum across any three consecutive squares. Given 2 + _b_ + _c_ = 10, we have _b_ + _c_ = 8. 

...... 

Given the constraints and the aim to solve for _c_ , **the approach taken highlights the need for either a specific pattern to follow or additional given numbers in the sequence to directly calculate** _c_ **’s value, pointing towards a limitation in the direct calculation method** 

## **GPT-4V Answer:** 

Let’s assign variables to represent the numbers in each of the squares: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0070-12.png)


Given that the sum of the numbers in three consecutive squares is always 10, we can create the following equations: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0070-14.png)


We can solve these equations successively. From the first equation: _a_ + _b_ = 10 _−_ 2 _a_ + _b_ = 8 Now, we don’t have enough information to solve for a or b individually, but we don’t need to because we can use the second equation to find c (which is related to the grey square): _a_ + _b_ + _c_ = 10 Substitute _a_ + _b_ = 8 into the second equation: 8 + _c_ = 10 _c_ = 10 _−_ 8 _c_ = 2 So, the number that should be written on the grey square is 2 . 

Figure 59: An example of logic question of GPT-4 with image caption. GPT-4 fails to correctly find all the necessary equations and does not reach an answer. GPT-4V correctly solves the problems. 

## **Example of Metric Geometry** 

**Question:** A circle _K_ is inscribed in a quarter circle with radius 6 as shown in the figure. 

What is the radius of circle _K_ ? (A)[6] _[−]_ 2 _√_ 2 (B)[3] _√_ 2 2 (C) 2.5 (D) 3 (E) 6( _√_ 2 _−_ 1) **Ground Truth:** E 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset_images/nips_2024_Measuring_Multimodal_Mathematical_Reasoning_with_MATH-Vision_Dataset.pdf-0071-03.png)


## **GPT4V Image Caption:** 

The diagram shows a large circle with a smaller circle inside it. The smaller circle is tangent to the larger circle at a single point on the larger circle’s interior. The smaller circle is labeled with the letter K. The space between the two circles forms a crescent shape. The larger circle appears to be a boundary that encloses the smaller circle, and there is no indication of any dimensions, angles, or other geometric properties. 

## **GPT4 Answer:** 

To solve this problem, let’s denote the radius of the smaller circle (circle _K_ ) as _r_ . Since circle _K_ is inscribed within a quarter circle of radius 6, we can visualize or draw a diagram where the quarter circle’s radius forms a right-angled triangle with the sides being the radius of the quarter circle, the radius of circle _K_ , and the distance from the point where circle _K_ touches the quarter circle’s arc to the corner of the quarter circle (the center of the quarter circle). The distance from the center of circle _K_ to the corner of the quarter circle is _r_ , and the distance from the corner of the quarter circle to the point where circle _K_ touches the quarter circle’s arc is also _r_ . Therefore, the hypotenuse of the right-angled triangle formed (which is also the radius of the quarter circle) is 6 units. Using the Pythagorean theorem, we can set up the equation: _r_ + _r_ = 2 _r_ (this is the base and the height of the right-angled triangle, both equal to the diameter of circle _K_ since it touches both the arc and the two straight sides of the quarter circle). **The hypotenuse (the radius of the quarter circle) is** 6 **units, so:** (2 _r_ )[2] + (2 _r_ )[2] = 6[2] 

...... 

~~_√_~~ 2 Therefore, the radius of circle _K_ is[3] units. 2 

## **GPT-4V Answer:** 

To solve this problem, we need to use geometric relationships within the quarter circle and the inscribed circle K. Consider a quarter circle of radius 6. If we inscribe a circle K inside it, such that K is tangent to the two sides and the arc of the quarter circle, there will be two points of tangency on the sides of the quarter circle which are at a distance equal to the radius of K from the corners of the quarter circle. Let’s denote the radius of circle K as r. Since circle K is tangent to the two sides, we know that the distance from the center of K to each side is equal to r, the radius of K. This means there will be a square of side r formed between the center of circle K and the two sides of the quarter circle. The remaining section between the edge of this square and the quarter circle will be a right-angled isosceles triangle with sides of length 6 _− r_ (since the quarter circle has a radius of 6). The hypotenuse of this right-angled isosceles triangle will be the radius of the quarter circle, which is 6. Using the Pythagorean theorem, we get: (6 _− r_ )[2] + (6 _− r_ )[2] = 6[2] ...... so our answer is: _E_ 

Figure 60: An example of metric geometry question of GPT-4 with image caption. GPT-4 fails to find the correct relationship between the variables. GPT-4V correctly solves the problems. 

