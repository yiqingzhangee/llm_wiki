Published as a conference paper at ICLR 2025 

## DYNAMATH: A DYNAMIC VISUAL BENCHMARK FOR EVALUATING MATHEMATICAL REASONING ROBUSTNESS OF VISION LANGUAGE MODELS 

**Chengke Zou**[1] _[,]_[2] _[∗†]_ **, Xingang Guo**[1] _[∗]_ **, Rui Yang**[1] _[∗]_ **, Junyu Zhang**[1] **, Bin Hu**[1] **, Huan Zhang**[1] 1University of Illinois at Urbana-Champaign, 2University of California, Berkeley chengke ~~z~~ ou@berkeley.edu, _{_ xingang2,ry21,junyuz6 _}_ @illinois.edu binhu7@illinois.edu, huan@huan-zhang.com **Project page: https://dynamath.github.io** 

## ABSTRACT 

The rapid advancements in Vision-Language Models (VLMs) have shown great potential in tackling mathematical reasoning tasks that involve visual context. Unlike humans who can reliably apply solution steps to similar problems with minor modifications, we found that state-of-the-art VLMs like GPT-4o can consistently fail in these scenarios, revealing limitations in their mathematical reasoning capabilities. In this paper, we investigate the **mathematical reasoning robustness** in VLMs and evaluate how well these models perform under different variants of the same question, such as changes in visual numerical values or function graphs. While several vision-based math benchmarks have been developed to assess VLMs’ problem-solving capabilities, these benchmarks contain only static sets of problems and cannot easily evaluate mathematical reasoning robustness. To fill this gap, we introduce DYNAMATH, a dynamic visual math benchmark designed for in-depth assessment of VLMs. DYNAMATH includes 501 high-quality, multi-topic _seed_ questions, _each represented as a Python program_ . Those programs are carefully designed and annotated to enable the automatic generation of a much larger set of _concrete_ questions, including many different types of visual and textual variations. DYNAMATH allows us to evaluate the generalization ability of VLMs, by assessing their performance under varying input conditions of a seed question. We evaluated 14 state-of-the-art VLMs with 5,010 generated concrete questions (10 per seed question). Our results show that the worst-case model accuracy, defined as the percentage of correctly answered seed questions in all 10 variants, is significantly lower than the average-case accuracy. In addition, many models show high consistency in answering these questions – the incorrectness of a certain variant of a seed question is not only due to inherent randomness. Our analysis emphasizes the need to study the robustness of VLMs’ reasoning abilities, and DYNAMATH provides valuable insights to guide the development of more reliable models for mathematical reasoning. 

## 1 INTRODUCTION 

Leveraging pretraining on vast Internet-scale datasets, Large Language Models (LLMs) (Brown, 2020; Ouyang et al., 2022; Touvron et al., 2023; Achiam et al., 2023) and Multi-modal Large Language Models (MLLMs) (Team et al., 2023; Bai et al., 2023; Liu et al., 2024c;a) have achieved remarkable performance across a wide range of tasks. Among them, Vision-Language Models (VLMs) (Zhu et al., 2023; Zhang et al., 2024b) stand out, showing exceptional promise as versatile assistants capable of integrating vision and language for problem-solving. 

Among their visual comprehension abilities across different domains, mathematical reasoning (Lightman et al., 2023; Zhang et al., 2024f) stands out as a crucial measure of human-like intelligence, requiring both math knowledge and logical thinking. Recent work has proposed many benchmarks for evaluating the mathematical reasoning ability of VLMs. MATHVISTA (Lu et al., 2023) 

> _∗_ Equal contribution. 

> _†_ Work done during internship at UIUC. 

1 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Seed Question 78, Variant 9 Seed Question 78, Variant 7<br>Is the function differentiable at  Is the function differentiable at<br>x = 0? x = 0?<br>Choice: (A) Yes  (B) No Choice: (A) Yes  (B) No<br>Response from GPT-4o: Response from GPT-4o:<br>The graph you've provided seems to depict a V-shaped  The graph shown is of the absolute value function, y =<br>function, which is indicative of an absolute value  |x|, which forms a 'V' shape. To determine if a<br>function or a similar type. The function has a sharp  function is differentiable at a point, the function<br>corner at x = 0, which suggests that it is not smooth  must be continuous, and the derivative must exist at<br>at that point. that point.<br>... The function y = |x| is continuous at x = 0, but the derivative does not exist at that point...<br>Since the graph shows a sharp corner at x = 0, the<br>correct answer is: Therefore, the correct choice is:<br>(B) No, the function is not differentiable at x = 0. (B) No<br>(a) Wrong Answer and Explanation to Q78 (b) Correct Answer and Explanation to Q78<br>**----- End of picture text -----**<br>


**Response from GPT-4o:** The graph shown is of the absolute value function, y = |x|, which forms a 'V' shape. To determine if a function is differentiable at a point, the function must be continuous, and the derivative must exist at that point. 

Figure 1: An example of consistent failures in GPT-4o. Seed question 78 in our DYNAMATH benchmark generates a graph of a shifted absolute value function. GPT-4o consistently provides incorrect answers for variant 9 **(left)** with 90% repetition consistency, while it can successfully answer variant 7 **(right)** with 100% repetition consistency. We tested for other 8 variants involving non-zero shifts of the absolute value function, GPT-4o insists that the “sharp corner” is at _x_ = 0 and produces an incorrect answer for 7 variants. More failure examples are in Appendix F. 

was the first benchmark specifically designed to evaluate visual mathematical reasoning. Recent closed-source models, such as Claude 3.5 Sonnet and GPT-4o, along with open-source models like LLaVA-OneVision (Li et al., 2024), have demonstrated average performance surpassing that of humans. Benchmarks such as MATH-V (Wang et al., 2024a) and MATHVERSE (Zhang et al., 2024e) demonstrate the current limitations of VLMs in handling challenging mathematical problems and understanding mathematical diagrams. Following typical evaluation pipelines, these benchmarks contain a _static_ set of testing questions on which a VLM will be scored. 

Our work is inspired by recent studies (Nezhurina et al., 2024; Zheng et al., 2023; Zong et al., 2023; Mirzadeh et al., 2024), which found that even powerful LLMs struggle to reliably solve simple text reasoning problems under different input values or conditions. We found that this issue is even more pronounced in VLMs due to the added complexity of visual context. In the setting of math problems, we identified consistent failure cases on _variations of simple questions_ . As illustrated in Figure 1, we identify a simple question asking whether a shifted absolute value function _f_ ( _x_ ) = _|x − a|_ is differentiable at _x_ = 0. Despite the shift, this question is still quite simple and poses no challenges to humans. While GPT-4o can give correct answers for some values of _a_ , it consistently gives a wrong answer for many different values of _a_ = 0. Drawing inspiration from human reasoning, where the same steps can be applied to solve similar problems with varying conditions, a robust reasoning model should exhibit the same ability. This raises important questions about the robustness of VLMs’ reasoning abilities: _**are the reasoning procedures in VLMs robust to problem variations that pose no challenge to humans?**_ 

To address this question, we comprehensively study the robustness of mathematical reasoning in VLMs by introducing a new benchmark, DYNAMATH. DYNAMATH is a _dynamic_ visual math benchmark designed for an in-depth assessment of VLMs’ reasoning robustness. Unlike existing benchmarks, which contain a static dataset of benchmarking questions, DYNAMATH contains 501 high-quality _seed_ questions covering multiple mathematical topics: Plane Geometry, Solid Geometry, Analytic Geometry, Algebra, Puzzle Tests, Graph Theory, Statistics, Scientific Figures, and Arithmetic. Each seed question is represented as a carefully designed Python _program_ ; upon running, a program generates diverse _concrete_ instances of one seed question with random variations in its conditions. The program is individually written for each seed question and considers multiple possible types of variations in each question, such as variations of numerical values, function types, graph structure, geometry, mathematical operations, etc. The questions also span varying difficulty levels, from elementary school to high school and undergraduate, with the latter two dominating. The process of dynamic benchmark generation and evaluation is presented in Figure 2. During evaluation, many concrete questions are created from a single seed question, and thus the actual number of questions evaluated can be much greater (e.g., 10 _×_ more) than the number of seed questions. 

We conducted extensive experiments on DYNAMATH to evaluate the reasoning robustness of current state-of-the-art (SOTA) closed-source models, including GPT-4o, Gemini Pro, and Claude-3.5 

2 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0003-01.png)


**----- Start of picture text -----**<br>
Seed Question 169<br>Code for Question Variant Generation<br>The purple and orange curves are f(x) and g(x).<br>Is f(x)g(x) even or odd?<br>Choice: (A) odd (B) even (C) neither<br>Concrete Questions & Answers<br>Answer for Variant 1: B Answer for Variant 2: B<br>Answer for Variant 3: B Answer for Variant 4: A<br>Variant 5, 6, 7...<br>Answer Matching<br>Vision-Language Models Variant 1 （B）even Variant 3 （B）even<br>Variant 2 （A）odd Variant 4 （A）odd<br>...<br>**----- End of picture text -----**<br>


Figure 2: The dynamic benchmark generation procedure in DYNAMATH. A seed question is represented as a program that can generate many concrete questions with different variations. The plots for concrete questions are randomly generated along with the corresponding ground-truth answers. During evaluation, all concrete variants of the seed questions are considered, allowing us to evaluate the worst-case model performance and robustness. 

Sonnet, as well as open-source VLMs such as the InternVL2 series (Chen et al., 2024), LLaVAv1.6 series (Liu et al., 2024b), Qwen2-VL (Wang et al., 2024c), DeepSeek-VL (Lu et al., 2024), and Llama 3.2 (Dubey et al., 2024). For each seed problem, we randomly generated 10 variants, resulting in an evaluation dataset of 5,010 concrete problems. On these problems, we evaluate both average-case accuracy and worst-case accuracy. The _worst-case accuracy_ is defined as the percentage of correctly answered seed problems in _all_ 10 variants. We observe that all considered VLMs have a worst-case accuracy that is close to or less than 50% of the average-case accuracy, signifying their unreliability in handling question variations. In addition, we also evaluate the _repetition consistency_ on these VLMs, which characterizes the model randomness to ensure that a low worstcase accuracy is not solely caused by occasional random errors but also consistent errors on certain variants of a seed problem. Our main contributions and findings can be summarized as: 

• We are the first to study the mathematical reasoning robustness of VLMs and identified a new weakness in VLMs: they may consistently fail on certain variants of simple math questions that pose no challenges to humans. Such a weakness is prevalent in many state-of-the-art VLMs. 

• We introduce DYNAMATH, a dynamic benchmark comprising 501 individually designed programs capable of generating a large number of question variants across different types. Our work is the first dynamically generated benchmark for evaluating the math capability of VLMs. 

• Based on 5,010 concrete questions generated by DYNAMATH, we conduct an extensive evaluation of both SOTA closed-source and open-source VLMs. We find a noticeable gap between the averagecase accuracy and worst-case accuracy among all models, indicating that many VLMs do not have robust reasoning capabilities even on relatively simple mathematical questions. 

## 2 RELATED WORK 

**Mathematical Reasoning Benchmarks.** Reasoning ability is a key indicator of intelligence, prompting researchers to develop various benchmark datasets to assess the mathematical reasoning capabilities of LLMs and VLMs. Numerous benchmarks have been proposed for evaluating this ability in the text-only domain, including (Amini et al., 2019; Hendrycks et al., 2020; 2021; Cobbe et al., 2021; Mishra et al., 2022; Frieder et al., 2024; Yu et al., 2023; Zhang et al., 2024a). Additionally, recent research has begun to shift its focus towards the evaluation of robustness and the creation of dynamic benchmarks for language models. Several studies (Stolfo et al., 2022; Wu et al., 2023; Srivastava et al., 2024; Nezhurina et al., 2024; Qian et al., 2024; Kurtic et al., 2024; Mirzadeh et al., 2024) assess the language models’ robustness to the changing of item names or value conditions in the text-based question. However, many real-world problems, such as those involving 

3 

Published as a conference paper at ICLR 2025 

statistical charts and geometry, rely on visual context. To assess visual mathematical reasoning, several benchmarks have been designed around geometry tasks (Lu et al., 2021; Chen et al., 2021) or multiple-choice questions (Liu et al., 2023; Yue et al., 2024). Among these, Liu et al. (2023) studied the robustness of VLMs when faced with changes in the order of multiple-choice questions. Recent efforts have expanded these benchmarks to cover a broader array of topics and question formats, such as MATHVISTA (Lu et al., 2023), MATHVERSE (Zhang et al., 2024e), and MATH-V (Wang et al., 2024a). Despite the diverse range of questions and visual contexts in these benchmarks, they share a common limitation: both the visual components and text remain static. This allows models to potentially achieve high scores by memorizing patterns from the training data, rather than applying true reasoning skills. In contrast, this paper introduces DYNAMATH, a dynamic visual math benchmark that provides a more rigorous assessment of VLMs’ reasoning capabilities through dynamically generating math questions with visual content. 

**Vision-Language Models (VLMs)** With the success of LLMs, numerous closed-source VLMs, such as GPT-4o, Gemini, and Claude 3.5, have excelled across a variety of visual-based understanding and conversational tasks, highlighting the potential of multimodal AI assistants. In the open-source domain, several efforts are actively advancing the field. Approaches like LLaMAAdapter (Zhang et al., 2024d; Gao et al., 2023) and MiniGPT-4 (Zhu et al., 2023) leverage frozen language models with a limited number of trainable parameters, demonstrating promising results. Furthermore, a range of VLMs trained on larger multimodal datasets has been open-sourced, pushing the frontier of visual comprehension and generalization ability. Notable examples include the InternVL1.5 and InternVL2 series (Chen et al., 2024), InternLM-XComposer (Zhang et al., 2023; Dong et al., 2024), LLaVA-v1.6 series (Liu et al., 2024b), LLaVA-OneVision (Li et al., 2024), Qwen-VL (Bai et al., 2023; Wang et al., 2024c), and DeepSeek-VL (Lu et al., 2024). These models contribute significantly to advancing the capabilities of VLMs in prior visual benchmarks. 

## 3 BENCHMARK DESIGN 

We present DYNAMATH, a curated evaluation dataset aimed at assessing the robustness of visual language models (VLMs) in multimodal mathematical reasoning across a wide variety of mathematical tasks with dynamic visual and textual contexts. 

## 3.1 DATASET COLLECTION 

Our benchmark collection comprises two phases: seed question collection and program-based question generation. In the initial phase, we selectively curate a set of high-quality mathematics problems that necessitate reasoning based on visual information. The subsequent phase involves transforming each seed question into code-based prototypes, allowing for the generation of diverse concrete questions under randomly sampled conditions. 

**Seed question Collection.** The seed questions are sourced from existing visual math datasets and publicly available online resources. We identify 107 questions from MathVista (Lu et al., 2023), covering fundamental concepts in analytic geometry, planar geometry, and statistics. Additionally, we source 27 questions from MATH-V (Wang et al., 2024a), which serve as prototypes for topics related to arithmetic, puzzle tests, and solid geometry. To augment the dataset’s breadth and depth, we included 45 questions based on scientific figures and 48 undergraduate-level questions focused on graph theory, drawn from the MMMU dataset (Yue et al., 2024) and various accessible educational materials. Furthermore, we incorporated 236 questions requiring advanced reasoning on topics such as functions, geometry, and statistics, all gathered from publicly available resources on the internet. To diversify the question types represented in our collection, we also developed 38 new problems by ourselves covering linear algebra, set theory, and algorithmic flow. 

Following the collection of seed questions, we conducted a comprehensive review to eliminate any questions that included excessively complex images, as these would pose challenges for programmatic generation. Ultimately, as shown in Figure 4(b), our benchmark consists of 501 seed questions, with 227 (45.3%) sourced from established visual math datasets, while 274 (54.7%) are newly collected or developed from public resources. 

Note that our goal is not to create the most challenging, competition-level benchmark as in (Wang et al., 2024a), but rather to provide relatively easy benchmarks with diverse variants to evaluate robustness. Nonetheless, we ensure that the difficulty of our questions is comparable to the levels of datasets such as MATHVERSE (Zhang et al., 2024e) and MATHVISTA (Lu et al., 2023). 

4 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0005-01.png)


**----- Start of picture text -----**<br>
(a) Numerical Value Variants (b) Geometric Transformations (c) Graph Structure Variants (d) Function Type Variants<br>**----- End of picture text -----**<br>


Figure 3: Examples of variation types in DYNAMATH. More examples are in Appendix B and D. 

**Program-based Question Generation.** After establishing our seed questions, we recruited a group of college STEM students to annotate each question with the common strategies they employed in solving them. These annotations served as prototypes for developing corresponding programs tailored to each question. As illustrated in Figure 2, each question is represented as a carefully crafted Python program, which encompasses a defined range of conditions for sampling and algorithmic calculations to derive the solution. Additionally, we implemented a drawing function in each program, utilizing libraries such as Matplotlib and Pyglet to generate corresponding images based on varying conditions. Specifically, 470 of the question programs incorporate a plotting function that leverages the randomly sampled conditions to create the visual context of the question, while the remaining 31 question programs utilize fixed images, randomizing only the textual elements. This programmatic approach allows the generation of a large number of concrete benchmark questions by executing the generation program multiple times, facilitating the efficient creation of new problems and enabling the evaluation of the reasoning robustness of VLMs. 

As shown in Figure 3, we integrate various types of variants to enrich the diversity of question generation for DYNAMATH: 

1. **Numerical Value Variants** : Modifying numerical quantities to evaluate the VLM’s proficiency in handling different numerical values and performing arithmetic operations. 

2. **Geometric Transformations** : Altering shapes, angles, dimensions, and relative positions to examine the spatial and geometric understanding of VLMs. 

3. **Function Type Variants** : Varying different types of mathematical functions (e.g., linear, quadratic) to evaluate how well models generalize across functional representations. 4. **Color Variants** : Changing object or curve colors randomly to test the model’s recognition of visual patterns and its robustness to superficial alterations. 

5. **Symbolic Substitutions** : Modifying symbolic elements such as mathematical operations to determine the model’s adaptability to various symbolic representations. 

6. **Graph Structure Variants** : Modifying graph layouts, networks, or other structural representations to assess the model’s comprehension of relationships and topological features. 

7. **Real-life Contexts Variants** : Adjusting the contents of real-world scenarios (e.g., calendars, time-related problems, or poker-like questions) to test the model’s contextual understanding and application to practical situations. 

Each variant category targets a specific facet of mathematical reasoning, making DYNAMATH a comprehensive benchmark for evaluating the flexibility, robustness, and accuracy of VLMs in solving mathematical problems. Detailed diagrams of each variation are provided in Appendix B. 

## 3.2 DATASET STATISTICS 

Detailed statistics on the data composition of DYNAMATH are presented in Table 1. DYNAMATH encompasses nine mathematical topics: Solid Geometry (SG, 3.0%), Puzzle Tests (PT, 3.4%), Arithmetic (AR, 5.2%), Scientific Figures (SF, 9.0%), Graph Theory (GT, 9.6%), Algebra (AL, 10.2%), Plane Geometry (PG, 15.4%), Analytic Geometry (AG, 19.4%), and Statistics (ST, 25.0%). Examples for each topic are provided in Appendix D. Each topic necessitates a nuanced understanding of image context, foundational mathematical knowledge, practical reasoning abilities, and logical deduction skills. Importantly, the dataset is designed to cater to varying levels of difficulty, ranging from elementary to undergraduate education, with a notable focus on high school (55.3%) and undergraduate (32.1%) levels. In terms of question types, the dataset consists of 59.1% numerical questions, 34.7% multiple-choice questions, and 6.2% free-form questions. While VLMs might occasionally answer multiple-choice questions correctly by chance, free-form questions provide a more precise evaluation of the model’s capabilities. Consequently, our dataset emphasizes free-form questions, distinguishing it from previous visual math benchmarks such as MATHVISTA (Lu et al., 2023), MATHVERSE (Zhang et al., 2024e), and MATH-V (Wang et al., 2024a), which predominantly include more than 50% multiple-choice questions. 

5 

Published as a conference paper at ICLR 2025 

|**Statistic**<br>**Number**<br>Total_seed_questions (programs)<br>501<br>- Created from existing dataset<br>227 (45.3%)<br>- Newlydesignedquestions<br>274(54.7%)<br>Topics<br>- Solid geometry (SG)<br>15 (3.0%)<br>- Puzzle test (PT)<br>17 (3.4%)<br>- Arithmetic (AR)<br>26 (5.2%)<br>- Scientifc fgure (SF)<br>45 (9.0%)<br>- Graph theory (GT)<br>48 (9.6%)<br>- Algebra (AL)<br>51 (10.2%)<br>- Plane geometry (PG)<br>77 (15.4%)<br>- Analytic geometry (AG)<br>97 (19.4%)<br>- Statistics(ST)<br>125(25.0%)<br>Levels<br>- Elementary school (EL)<br>63 (12.6%)<br>- High school (HI)<br>277 (55.3%)<br>- Undergraduate(UN)<br>161(32.1%)<br>Question Types<br>- Numerical questions<br>296 (59.1%)<br>- Multiple-choice questions<br>174 (34.7%)<br>- Free-form questions<br>31 (6.2%)<br>0<br>20<br>40<br>60<br>80<br>100<br>120<br>140<br>160<br>Frequency||101<br>101-102<br>102-103<br>103-104<br>104-105<br>105-106<br>106-<br>Variant Numbers<br>~~36~~<br>~~153~~<br>78<br>~~92~~<br>~~34~~<br>21<br>87<br>Variants Distribution|
|---|---|---|
|||~~153~~|
||||
||||
||||
|||78<br>~~92~~<br>87|
||||
|||~~36~~<br>~~34~~|
|||21|
||||
|||101<br>1-102<br>2-103<br>3-104<br>4-105<br>5-106<br>106-|
||6||
|||(a)<br>Math-V<br>New problem<br>MathVista<br>MMMU<br>54.7%<br>21.4%<br>18.6%<br>5.4%<br>(b)|



Table 1: Statistics of DYNAMATH. 

Figure 4: (a) Variant number distribution and (b) source composition of DYNAMATH. 

In Figure 4(a), we illustrate the distribution of variant numbers among the 501 seed questions. Notably, approximately 30.5% of the seed questions have a possible variant number ranging from 10 to 10[2] . Nearly 93% of the seed questions contain more than 10 variants, and 17.4% of them have more than 10[6] potential variants, demonstrating the diversity of variations in our dataset. 

3.3 EVALUATION PROTOCOLS 

Our evaluation process consists of two stages: answer extraction and score calculation. Following the methodology of prior work (Lu et al., 2022), we utilize prompt engineering and template matching to extract answers. Prompts guide the model to generate responses in both full and short answer formats. After generation, the short answer is extracted for comparison with the ground truth. Detailed prompts used in our experiments can be found in Appendix C. 

Our dataset contains _N_ = 501 seed questions in total. For each seed question in the dataset, we generate _M_ = 10 variants, resulting in a total of 5 _,_ 010 concrete questions. We evaluate two metrics: **average-case accuracy (** _Aavg_ **)** and **worst-case accuracy (** _Awst_ **)** over these variants. The two metrics are different from prior benchmarks that evaluate only a single instance of a question. The metrics are defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0006-08.png)


where Ans( _i, j_ ) and GT( _i, j_ ) represent the generated answer and the ground truth answer for variant _j_ of question _i_ . We also define **Reasoning Robustness (** _RR_ **)** as the ratio between the average-case performance and the worst-case performance: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0006-10.png)


The model’s response uncertainty reflects both the impact of input changes and inherent uncertainty, the latter of which can be represented by the concept of repetition consistency ( _RC_ ), similar to self-consistency (Wang et al., 2022). We define repetition consistency as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0006-12.png)


where _K_ is number of repetitions and Ans _k_ ( _i, j_ ) is the _k_ -th repetition for _j_ -th variant of _i_ -th seed question. The repetition consistency represents the model’s confidence in the answer Ans( _i, j_ ). 

6 

Published as a conference paper at ICLR 2025 

Table 2: Average-case accuracy _Aavg_ on DYNAMATH with 5,010 generated questions. “ALL” represents overall accuracy. Question topics and difficulty levels (PG, EL, etc) are defined in Table 1. 

|Model|ALL|PG<br>SG<br>AG<br>AL<br>PT<br>GT<br>ST<br>SF<br>AR|EL<br>HI<br>UN|
|---|---|---|---|
|_Closed-sourced Large Multimodal Models (LMMs)_||||
|Zero-shot GPT-4o<br>Zero-shot Claude-3.5<br>Zero-shot Gemini Pro 1.5|63.7|56.8<br>52.0<br>61.0<br>76.9<br>51.8<br>58.1<br>69.3<br>62.4<br>61.5|68.6<br>61.8<br>36.8<br>66.7<br>62.6<br>33.3|
||64.8|49.9<br>49.3<br>55.3<br>81.0<br>44.1<br>69.4<br>78.2<br>62.2<br>61.2||
||60.5|52.7<br>42.7<br>61.6<br>70.8<br>20.6<br>65.2<br>69.8<br>50.2<br>54.2|62.9<br>59.2<br>37.1|
|3-shot CoT GPT-4o<br>3-shot CoT Claude-3.5<br>3-shot CoT Gemini Pro 1.5|64.9|58.1<br>59.3<br>57.7<br>84.1<br>51.2<br>61.9<br>71.0<br>60.9<br>57.7<br>49.1<br>48.0<br>50.6<br>80.2<br>37.1<br>58.1<br>78.2<br>64.9<br>55.0<br>52.6<br>45.3<br>56.7<br>72.9<br>21.8<br>57.9<br>66.0<br>54.9<br>48.1|66.2<br>62.5<br>34.8<br>63.0<br>61.5<br>30.5<br>59.0<br>58.3<br>34.2|
||62.5<br>|||
||58.7|||
|_Open-source Vision Language Models (VLMs)_||||
|Qwen2-VL-72B<br>Qwen2-VL-72B (3-shot CoT)<br>Qwen2-VL-7B<br>InternVL2-76B<br>InternVL2-40B<br>InternVL2-26B<br>InternVL2-8B<br>Llama-3.2-90B<br>Deepseek-VL-7B-chat<br>Llava-v1.6-34B<br>Llava-v1.6-vicuna-13B<br>Llava-v1.5-7B|55.1|48.1<br>48.7<br>50.9<br>57.6<br>28.2<br>45.0<br>68.9<br>56.4<br>54.2|61.3<br>57.4<br>30.7|
||52.4<br>42.1|45.1<br>44.7<br>47.5<br>59.4<br>19.4<br>44.2<br>67.1<br>52.9<br>53.1<br>40.3<br>38.7<br>39.9<br>37.1<br>8.2<br>44.8<br>52.1<br>41.1<br>39.2<br>44.5<br>34.7<br>43.8<br>67.6<br>35.3<br>51.0<br>66.7<br>55.1<br>51.5<br>31.3<br>21.3<br>38.8<br>42.9<br>15.3<br>38.3<br>58.1<br>43.1<br>38.1<br>35.8<br>26.0<br>37.3<br>38.8<br>13.5<br>46.9<br>51.9<br>39.6<br>40.4<br>33.9<br>37.3<br>32.5<br>46.9<br>15.9<br>42.1<br>47.8<br>39.1<br>37.3<br>47.5<br>37.3<br>36.8<br>46.5<br>12.4<br>44.8<br>56.8<br>39.8<br>30.0<br>16.0<br>13.3<br>26.5<br>12.9<br>4.7<br>32.7<br>24.3<br>24.2<br>15.0<br>21.4<br>25.3<br>27.6<br>14.9<br>7.6<br>32.7<br>36.8<br>27.8<br>23.1<br>14.7<br>10.0<br>23.4<br>8.2<br>10.0<br>21.5<br>28.2<br>19.6<br>10.0<br>10.5<br>7.3<br>19.5<br>6.5<br>8.2<br>32.3<br>17.5<br>20.2<br>10.8|61.0<br>53.6<br>28.6<br>47.6<br>42.2<br>24.4<br>60.3<br>52.9<br>26.4<br>51.0<br>41.5<br>23.4<br>52.1<br>38.5<br>22.5<br>51.1<br>37.4<br>19.6<br>45.4<br>43.8<br>22.2<br>28.3<br>19.0<br>16.0<br>35.9<br>23.8<br>16.6<br>27.1<br>16.5<br>14.1<br>18.9<br>13.3<br>11.7|
||54.0|||
||41.8<br>41.0<br>39.7<br>44.0<br>21.5<br>27.1<br>19.8<br>16.6|||
|_Human_||||
|Human performance|77.3|79.9<br>66.7<br>80.4<br>77.5<br>73.5<br>69.8<br>78.0<br>78.9<br>75.0|78.6<br>79.8<br>72.7|



## 4 EXPERIMENT 

In this section, we conduct thorough experiments to assess the performance and reasoning robustness of various closed-source and open-source models on the DYNAMATH dataset. Subsequently, we present detailed quantitative results and qualitative analyses in Sections 4.2 and 4.3, respectively. 

4.1 EXPERIMENTAL SETUPS We evaluate the performance of two sets of models on the DYNAMATH benchmark, which involves 10 variations for each seed question, resulting in a total of 5010 questions. The first group comprised SOTA closed-source VLMs, such as GPT-4o, Gemini Pro 1.5, and Claude-3.5 Sonnet, with zero-shot and 3-shots with Chain-of-Thought (CoT) configurations. The second group consisted of SOTA open-source VLMs, including Qwen2-VL (7B, 72B) (Wang et al., 2024c), InternVL2 (8B, 26B, 40B, 76B) (Chen et al., 2024), Llava-v1.5 (7B) (Liu et al., 2024a), Llava-v1.6 (13B, 34B) (Liu et al., 2024b), Deepseek-VL (7B) (Lu et al., 2024), and Llama 3.2 (90B) (Dubey et al., 2024). We specifically explored open-source models with varying parameter sizes to analyze the impact of model size on reasoning robustness. The deployment of open-source models relied on the lmdeploy package (Contributors, 2023). We set the temperature to 0.0 for all models to reduce inherent randomness. Details regarding the prompts and hyperparameters used in this experiment are outlined in Appendix C. 

To assess human performance, we generated a new variant dataset consisting of 1002 concrete questions (2 variants per seed question). These questions were divided into 20 questionnaires, each containing 50 to 51 questions. We then recruited 20 undergraduates or graduates from STEM to help establish the baseline for human performance based on their average scores. 

For the few-shot setup, we follow the standard approach by including three demonstration examples, each accompanied by the associated visual elements. Given the diverse range of topics covered in DYNAMATH, we provide topic-specific demonstration examples to ensure its relevance for each problem in DYNAMATH. Specifically, we curated five demonstration examples from MathVista (Lu et al., 2023) and MathVision (Wang et al., 2024a) for each topic, and then randomly select three examples when evaluating DYNAMATH problems within the corresponding topic. In addition, we incorporate detailed reasoning steps in the demonstration examples, following a typical Chain-ofThought (CoT) setup (Wei et al., 2022). Detailed demonstrations and prompts in Appendix C.3. 

4.2 EXPERIMENTAL RESULTS 

In this section, we present a detailed comparison of the top-performing VLMs on DYNAMATH, as shown in Table 2 and Table 3. 

**Overall Results on Average Accuracy.** Table 2 illustrates the average-case performance of a variety of closed-source and open-source models. Within the closed-source category, GPT-4o, Claude- 

7 

Published as a conference paper at ICLR 2025 

Table 3: Worst-case accuracy _Awst_ on DYNAMATH with 5,010 generated questions. “ALL” represents overall accuracy. Question topics and difficulty levels (PG, EL, etc) are defined in Table 1. 

|Model|ALL|PG<br>SG<br>AG<br>AL<br>PT<br>GT<br>ST<br>SF<br>AR|EL<br>HI<br>UN|
|---|---|---|---|
|_Closed-sourced Large Multimodal Models (LMMs)_||||
|Zero-shot GPT-4o<br>Zero-shot Claude-3.5<br>Zero-shot Gemini Pro 1.5|34.7|37.7<br>33.3<br>25.8<br>54.9<br>11.8<br>18.8<br>38.4<br>35.6<br>46.2|46.0<br>34.3<br>31.1|
||35.3|22.1<br>26.7<br>18.6<br>62.7<br>23.5<br>27.1<br>53.6<br>24.4<br>42.3|49.2<br>33.2<br>33.5|
||26.9|28.6<br>20.0<br>19.6<br>39.2<br>5.9<br>22.9<br>35.2<br>15.6<br>30.8|41.3<br>26.7<br>21.7|
|3-shot CoT GPT-4o<br>3-shot CoT Claude-3.5<br>3-shot CoT Gemini Pro 1.5|32.3|31.2<br>40.0<br>21.6<br>54.9<br>17.6<br>20.8<br>36.8<br>26.7<br>46.2|47.6<br>30.7<br>29.2<br>39.7<br>32.9<br>28.0<br>33.3<br>23.1<br>20.5|
||32.1<br>|27.3<br>26.7<br>11.3<br>54.9<br>0.0<br>10.4<br>56.0<br>31.1<br>30.8<br>27.3<br>26.7<br>14.4<br>39.2<br>5.9<br>18.8<br>27.2<br>17.8<br>26.9||
||23.6|||
|_Open-sourced Vision Language Models (VLMs)_||||
|Qwen2-VL-72B<br>Qwen2-VL-72B (3-shot COT)<br>Qwen2-VL-7B<br>InternVL2-76B<br>InternVL2-40B<br>InternVL2-26B<br>InternVL2-8B<br>Llama-3.2-90B<br>Deepseek-VL-7B-chat<br>Llava-v1.6-34B<br>Llava-v1.6-vicuna-13B<br>Llava-v1.5-7B|28.3|27.3<br>33.3<br>15.5<br>31.4<br>0.0<br>16.7<br>43.2<br>26.7<br>42.3|41.3<br>30.3<br>19.9<br>41.3<br>23.5<br>14.3<br>25.4<br>12.3<br>11.8|
||22.8<br>13.8|24.7<br>26.7<br>8.2<br>35.3<br>0.0<br>8.3<br>32.8<br>22.2<br>38.5<br>22.1<br>6.7<br>7.2<br>13.7<br>0.0<br>12.5<br>16.8<br>11.1<br>19.2||
||24.6|24.7<br>20.0<br>15.5<br>37.3<br>5.9<br>12.5<br>32.8<br>20.0<br>38.5|39.7<br>23.1<br>21.1|
||14.2<br>14.4<br>10.4<br>13.0<br>4.2<br>6.0<br>2.8<br>1.8|14.3<br>6.7<br>9.3<br>13.7<br>0.0<br>10.4<br>21.6<br>13.3<br>19.2<br>19.5<br>0.0<br>6.2<br>9.8<br>0.0<br>18.8<br>20.0<br>11.1<br>26.9<br>13.0<br>20.0<br>5.2<br>15.7<br>0.0<br>10.4<br>9.6<br>11.1<br>15.4<br>22.1<br>20.0<br>7.2<br>7.8<br>0.0<br>12.5<br>16.8<br>13.3<br>3.8<br>7.8<br>0.0<br>3.1<br>0.0<br>0.0<br>10.4<br>4.0<br>2.2<br>3.8<br>10.4<br>13.3<br>4.1<br>2.0<br>0.0<br>4.2<br>6.4<br>6.7<br>7.7<br>7.8<br>0.0<br>4.1<br>0.0<br>0.0<br>2.1<br>2.4<br>0.0<br>0.0<br>3.9<br>0.0<br>2.1<br>0.0<br>0.0<br>4.2<br>0.8<br>0.0<br>3.8|28.6<br>14.1<br>8.7<br>34.9<br>12.3<br>9.9<br>23.8<br>9.4<br>6.8<br>15.9<br>14.1<br>9.9<br>7.9<br>2.9<br>5.0<br>15.9<br>5.1<br>3.7<br>6.3<br>2.9<br>1.2<br>3.2<br>1.8<br>1.2|



3.5, and Gemini Pro 1.5 exhibit average accuracies higher than 60%, with Claude-3.5 achieving the highest zero-shot average accuracy at 64.8%. However, there remains an 12.5% disparity when compared to human performance, which stands at 77.3%. This highlights the need for further development in the reasoning ability of VLMs. Regarding the 3-shot CoT performance, it is intriguing to note that there is no consistent improvement across different closed-sourced models, confirming findings from previous research (Wang et al., 2024a). For instance, while 3-shot CoT GPT-4o manages to enhance zero-shot performance from 63.7% to 64.9%, both 3-shot CoT Claude-3.5 and 3-shot CoT Gemini Pro 1.5 experience a decline in performance (64.8% _→_ 62.5% and 60.5% _→_ 58.7% respectively). Moving on to the open-sourced models, although they generally underperform when compared to closed-sourced models, the gap has been narrowed by recent models such as Qwen2 and InternVL2, which have more than 70B parameters. This noteworthy progress is evident when comparing them to previous benchmark results like MATHVISTA (Amini et al., 2019), MATHVERSE (Zhang et al., 2024e), and MATH-V (Wang et al., 2024a). It highlights the promising potential of open-source models in the visual math reasoning domain. Moreover, there is a clear scaling trend observed in open-source models, indicating higher performance as model sizes increase. For example, Qwen2-VL boosts its score from 42.1% to 55.1% when scaling its parameter size from 7B to 72B, while InternVL2 sees an increase from 39.7% to 54.0%. 

**Overall Results on Worst-case Accuracy.** Table 3 presents the worst-case accuracy of different models across 10 problem variants, revealing a significant decline in scores for all models. Notably, the highest-performing model, Claude-3.5, achieves a zero-shot score of only 35.3%, indicating current VLMs are not sufficiently robust to handle variations in context and images. The situation is even more concerning for open-source models: the best-performing model, Qwen2-VL-72B, achieves a score of 28.3%, while smaller models like Llava-v1.6-vicuna-13B score only 2.8%. Our evaluation results highlight the limited reasoning robustness of both open-source and closed-source models, underscoring the necessity for the community to address these limitations in future research. 

**Fine-grained Results.** In Table 2 and Table 3, we present detailed results categorized by different question topics and difficulty levels. From a topical perspective, we observe that the Puzzle Test (PT) topic challenges both open-source and closed-source models. The top-performing closed-source model, GPT-4o, and the leading open-source model, InternVL2-76B, achieve average-case accuracies of 51.8% and 35.3%, respectively, while humans score 73.5%. Notably, all open-source models demonstrate poor performance (0.0%) on the worst-case accuracy metric, except InternVL2-76B (5.9%). Despite this gap, Table 2 shows that closed-source models such as Claude-3.5 can surpass human scores on specific topics like Algebra (AL) and Statistics (ST), which is promising. When considering difficulty levels, all models demonstrate a trend of decreasing average accuracy as the difficulty increases, as illustrated in Table 2. In contrast, human performance remains consistent across difficulty levels, indicating that current VLMs are still not adept at handling more difficult visual math problems compared with human capabilities. 

8 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0009-01.png)


Figure 5: Comparing reasoning robustness across different (a) models and (b) topics. 

|**Model name**|**GPT-4o**|**Gemini**|**Qwen2-VL-72B**|**InternVL2-76B**|
|---|---|---|---|---|
|Repetition Consistency (%)|94.1|92.5|98.9|99.0|



Table 4: The Repetition Consistency ( _RC_ ) for different models over 5 repetitions. 

**Reasoning Robustness.** We use the reasoning robustness ( _RR_ ) metric, defined in Eq 2, to measure the robustness of VLMs by evaluating the relative performance consistency across question variants. We defer the detailed reasoning robustness results in Appendix H.3. Figure 5 (a) compares the _RR_ of all VLMs used in our experiments. Notably, Claude-3.5 and GPT-4o achieve the highest robustness among all tested models. Moreover, consistent with previous findings, closed-source models demonstrate greater robustness than open-source models, with reasoning robustness scaling with model size. **However, Qwen2-72B and InternVL2-76B outperform Gemini, highlighting the robustness limitations of even large models like Gemini.** In Figure 5 (b), we compare the reasoning robustness across different question topics for GPT-4o and Qwen2-VL-72B. The results show that the two VLMs are particularly robust in Arithmetic and Algebra question types, indicating their strong arithmetic calculation abilities, which are less affected by changes in visual conditions. However, GPT-4o still exhibits weaknesses in the Puzzle Test. Similarly, Qwen2-VL-72B shows shortcomings in both Puzzle Test and Analytic Geometry topics, achieving nearly 0% _RR_ and 30% _RR_ , respectively. These weaknesses suggest directions for future improvement of these models. 

**Repetition Consistency.** To ensure a robust analysis and account for the inherent randomness in model outputs, we calculate repetition consistency ( _RC_ ) as defined in Eq 3. This metric evaluates the model’s output confidence across multiple generations for the same question. Specifically, we produce five responses for 501 questions and then compute their consistency relative to the first response. The results, detailed in Table 4, reveal the consistent outputs of four closed-source and open-source models, with _RC_ values ranging from 92% to 99%. Compared with the low reasoning robustness scores, VLMs have much smaller consistency on different question variants. These findings reinforce our arguments that VLMs lack robustness in varying question conditions. 

**Consistent Failure Cases.** An interesting phenomenon we observed is that some seed questions are solvable in certain variants but result in consistent failures in others (repetition consistency _RC_ = 1 for 5 or 10 repetitions). The example in Figure 1 is a representative case: the question is easily solvable when the absolute value function at the origin, but any shifts tend to lead to consistent failures on GPT-4o. We extensively examined our dataset and counted the number of such instances. Specifically, GPT-4o, Gemini Pro 1.5, Qwen2-VL-72B, and InternVL2-76B exhibited 21.8%, 18.4%, 29.9%, and 28.3% of these types of questions, respectively, out of our 501 seed questions. These examples highlight the unreliability of VLMs on mathematical reasoning tasks. 

4.3 QUALITY STUDY 

**Qualitative Examples of GPT-4o.** In this section and Appendix G, we provide a few qualitative examples of leading VLMs’ answers. Our analysis reveals that current VLMs can consistently produce incorrect responses to specific question variants while generating accurate answers to others. As illustrated in Figure 1, GPT-4o demonstrates the ability to provide correct responses in variant 7, showcasing accurate perception, question understanding, and reasoning ability. However, in variant 9, where the underlying required capabilities remain the same with only a slight shift in the image, 

9 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0010-01.png)


**----- Start of picture text -----**<br>
Seed Question 12:  What is the period of this function? Answer the question with a floating-point number.<br>Variant 1 Variant 2 Variant 3 Variant 4 Variant 5<br>Answer: 6.283  Answer: 6.283 Answer: 6.283 Answer: 6.283 Answer: 6.283<br>**----- End of picture text -----**<br>


Figure 7: Example of the Memorization Phenomenon: the generated variants of seed Question 12 and the corresponding responses from Claude 3.5 Sonnet. The model’s response remains 2 _π_ with high probability, regardless of changes in the conditions depicted in the diagram. 

GPT-4o fails to accurately interpret the function’s position with a high degree of confidence and consistency. This discrepancy raises concerns about the reasoning robustness of current VLMs. For additional examples of GPT-4o and other models, please refer to the Appendix G. 

**Memorization Phenomenon.** In our experiments, we observe a phenomenon where current VLMs tend to provide the same answer regardless of changing conditions, indicating memorization rather than reasoning based on generalized underlying principles. When we test variant questions that have the same structure but different parameters and images, the model frequently offers the same answer with high probability, ignoring the specific variations we introduced. Among the 171 questions incorrectly answered by Claude 3.5 Sonnet, this issue accounts for 4.1% of instances. A representative case is illustrated in Figure 7, where altering the period of a sinusoidal function (e.g., from 2 _π_ to _π_ or 4 _π_ ) does not affect the model’s response, which consistently remains 2 _π_ . The existence of this phenomenon highlights the models’ lack of comprehensive problem analysis and their limited ability to generalize across different scenarios. 

**Error Analysis.** We conducted an error analysis on Claude 3.5 Sonnet to identify potential failure modes on DYNAMATH. Specifically, we analyzed the 169 questions where Claude 3.5 Sonnet failed, examining the reasoning paths and final answers in detail. The statistical distribution of various error types is presented in Figure 6. We considered five types of errors: figure reading errors, reasoning errors, knowledge errors, calculation errors, and hallucination errors. Figure reading errors account for 33.1% of the total errors, despite Claude 3.5 Sonnet having specially reinforced perception capabilities. This indicates that there is still a considerable way to go for VLMs to accurately read and interpret data from images. Reasoning errors account for 26.6%, making them the second-largest cause of errors. This suggests that the model’s 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0010-06.png)


**----- Start of picture text -----**<br>
33.1%<br>figure-readingerror 17.8%<br>hallucination<br>error<br>calculation<br>error reasoning<br>18.9% error<br>26.6%<br>3.6%<br>knowledge error<br>**----- End of picture text -----**<br>


Figure 6: Error Analysis of Claude-3.5 Sonnet. 

reasoning processes are still delicate and can be easily disrupted by minor changes in conditions and image input. Calculation errors, which constitute 18.9% of the errors, likely result from the significant computational challenge imposed by our randomly generated conditions without specially designed parameters, as opposed to simpler questions in prior work that are easier to compute. In addition, Hallucination errors make up 17.8%, showing that the model tends to fabricate non-existent information. More failure examples can be found in Appendix F. 

## 5 CONCLUSION 

In this work, we introduce DYNAMATH, a dynamic visual math benchmark designed to systematically analyze the robustness of mathematical reasoning capabilities in current leading visionlanguage models (VLMs). By employing program-based problem generation, we can create diverse variants by altering visual conditions in the seed problems. Our evaluation reveals that leading closed-source and open-source VLMs are sensitive to condition changes in question variants, despite their required underlying capabilities remaining the same. This raises significant concerns within the VLM community on mathematical reasoning tasks. Our detailed results and analysis not only identify the weak points of current VLMs but also shed light on the causes of their errors, thereby facilitating the development and evaluation of more robust VLMs in the future. 

10 

Published as a conference paper at ICLR 2025 

**Reproducibility Statement.** We have implemented several measures to ensure the reproducibility of our results. This includes providing detailed examples from our dataset, comprehensive descriptions of the prompts, and the hyperparameters used in our experiments. Additionally, our dataset is open-sourced for reproducibility. 

**Acknowledgment** Huan Zhang is supported in part by the AI 2050 program at Schmidt Sciences (AI 2050 Early Career Fellowship). The authors thank OpenAI’s researcher access program for providing part of the API credits used in our work. 

## REFERENCES 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. _arXiv preprint arXiv:1905.13319_ , 2019. 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023. 

Tom B Brown. Language models are few-shot learners. _arXiv preprint arXiv:2005.14165_ , 2020. 

- Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. _arXiv preprint arXiv:2105.14517_ , 2021. 

- Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

LMDeploy Contributors. Lmdeploy: A toolkit for compressing, deploying, and serving llm. https://github.com/InternLM/lmdeploy, 2023. 

- Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, et al. Internlm-xcomposer2: Mastering freeform text-image composition and comprehension in vision-language large model. _arXiv preprint arXiv:2401.16420_ , 2024. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- Simon Frieder, Luca Pinchetti, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Petersen, and Julius Berner. Mathematical capabilities of chatgpt. _Advances in neural information processing systems_ , 36, 2024. 

- Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. _arXiv preprint arXiv:2304.15010_ , 2023. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_ , 2020. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ , 2021. 

11 

Published as a conference paper at ICLR 2025 

Eldar Kurtic, Amir Moeini, and Dan Alistarh. Mathador-lm: A dynamic benchmark for mathematical reasoning on large language models. _arXiv preprint arXiv:2406.12572_ , 2024. 

- Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. _arXiv preprint arXiv:2408.03326_ , 2024. 

- Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. _arXiv preprint arXiv:2305.20050_ , 2023. 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 26296–26306, 2024a. 

- Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024b. URL https:// llava-vl.github.io/blog/2024-01-30-llava-next/. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36, 2024c. 

- Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? _arXiv preprint arXiv:2307.06281_ , 2023. 

- Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: towards real-world vision-language understanding. _arXiv preprint arXiv:2403.05525_ , 2024. 

- Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. _arXiv preprint arXiv:2105.04165_ , 2021. 

- Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. _Advances in Neural Information Processing Systems_ , 35:2507–2521, 2022. 

- Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. _arXiv preprint arXiv:2310.02255_ , 2023. 

- Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, et al. Improve mathematical reasoning in language models by automated process supervision. _arXiv preprint arXiv:2406.06592_ , 2024. 

- Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. Gsm-symbolic: Understanding the limitations of mathematical reasoning in large language models. _arXiv preprint arXiv:2410.05229_ , 2024. 

- Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, et al. Lila: A unified benchmark for mathematical reasoning. _arXiv preprint arXiv:2210.17517_ , 2022. 

- Marianna Nezhurina, Lucia Cipolina-Kun, Mehdi Cherti, and Jenia Jitsev. Alice in wonderland: Simple tasks showing complete reasoning breakdown in state-of-the-art large language models. _arXiv preprint arXiv:2406.02061_ , 2024. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _Advances in neural information processing systems_ , 35: 27730–27744, 2022. 

12 

Published as a conference paper at ICLR 2025 

Kun Qian, Shunji Wan, Claudia Tang, Youzhi Wang, Xuanming Zhang, Maximillian Chen, and Zhou Yu. Varbench: Robust language model benchmarking through dynamic variable perturbation. _arXiv preprint arXiv:2406.17681_ , 2024. 

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Christian Schlarmann, Naman Deep Singh, Francesco Croce, and Matthias Hein. Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models. _arXiv preprint arXiv:2402.12336_ , 2024. 

- Samuel Henrique Silva and Peyman Najafirad. Opportunities and challenges in deep learning adversarial robustness: A survey. _arXiv preprint arXiv:2007.00753_ , 2020. 

- Saurabh Srivastava, Anto PV, Shashank Menon, Ajay Sukumar, Alan Philipose, Stevin Prince, Sooraj Thomas, et al. Functional benchmarks for robust evaluation of reasoning performance, and the reasoning gap. _arXiv preprint arXiv:2402.19450_ , 2024. 

- Alessandro Stolfo, Zhijing Jin, Kumar Shridhar, Bernhard Sch¨olkopf, and Mrinmaya Sachan. A causal framework to quantify the robustness of mathematical reasoning with language models. _arXiv preprint arXiv:2210.12023_ , 2022. 

- Hao Sun. Reinforcement learning in the era of llms: What is essential? what is needed? an rl perspective on rlhf, prompting, and beyond. _arXiv preprint arXiv:2310.06147_ , 2023. 

- Lichao Sun, Yue Huang, Haoran Wang, Siyuan Wu, Qihui Zhang, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, et al. Trustllm: Trustworthiness in large language models. _arXiv preprint arXiv:2401.05561_ , 3, 2024. 

- Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. _arXiv preprint arXiv:2211.14275_ , 2022. 

- Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. _arXiv preprint arXiv:2402.14804_ , 2024a. 

- Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 9426–9439, 2024b. 

- Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024c. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022. 

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

13 

Published as a conference paper at ICLR 2025 

- Zhaofeng Wu, Linlu Qiu, Alexis Ross, Ekin Aky¨urek, Boyuan Chen, Bailin Wang, Najoung Kim, Jacob Andreas, and Yoon Kim. Reasoning or reciting? exploring the capabilities and limitations of language models through counterfactual tasks. _arXiv preprint arXiv:2307.02477_ , 2023. 

Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, and Lei Han. Rorl: Robust offline reinforcement learning via conservative smoothing. _Advances in neural information processing systems_ , 35:23851–23866, 2022. 

- Rui Yang, Ruomeng Ding, Yong Lin, Huan Zhang, and Tong Zhang. Regularizing hidden states enables learning generalizable reward model for llms. _arXiv preprint arXiv:2406.10216_ , 2024a. 

- Rui Yang, Xiaoman Pan, Feng Luo, Shuang Qiu, Han Zhong, Dong Yu, and Jianshu Chen. Rewardsin-context: Multi-objective alignment of foundation models with dynamic preference adjustment. _arXiv preprint arXiv:2402.10207_ , 2024b. 

- Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _arXiv preprint arXiv:2309.12284_ , 2023. 

- Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9556–9567, 2024. 

- Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Mingyan Liu, Duane Boning, and Cho-Jui Hsieh. Robust deep reinforcement learning against adversarial perturbations on state observations. _Advances in Neural Information Processing Systems_ , 33:21024–21037, 2020. 

- Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, Tiffany Zhao, Pranav Raja, Dylan Slack, Qin Lyu, et al. A careful examination of large language model performance on grade school arithmetic. _arXiv preprint arXiv:2405.00332_ , 2024a. 

- Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. Vision-language models for vision tasks: A survey. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2024b. 

- Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. _arXiv preprint arXiv:2408.15240_ , 2024c. 

- Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, et al. Internlm-xcomposer: A visionlanguage large model for advanced text-image comprehension and composition. _arXiv preprint arXiv:2309.15112_ , 2023. 

- Renrui Zhang, Jiaming Han, Chris Liu, Aojun Zhou, Pan Lu, Yu Qiao, Hongsheng Li, and Peng Gao. Llama-adapter: Efficient fine-tuning of large language models with zero-initialized attention. In _The Twelfth International Conference on Learning Representations_ , 2024d. 

- Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? _arXiv preprint arXiv:2403.14624_ , 2024e. 

- Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. _arXiv preprint arXiv:2407.08739_ , 2024f. 

- Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are not robust multiple choice selectors. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- Wanqi Zhou, Shuanghao Bai, Qibin Zhao, and Badong Chen. Revisiting the adversarial robustness of vision language models: a multimodal perspective. _arXiv preprint arXiv:2404.19287_ , 2024. 

14 

Published as a conference paper at ICLR 2025 

- Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 

- Yongshuo Zong, Tingyang Yu, Ruchika Chavhan, Bingchen Zhao, and Timothy Hospedales. Fool your (vision and) language model with embarrassingly simple permutations. _arXiv preprint arXiv:2310.01651_ , 2023. 

15 

Published as a conference paper at ICLR 2025 

## A LIMITATIONS AND FUTURE WORK 

Although our benchmark matches the difficulty levels of MATHVERSE and MATHVISTA, one limitation of our work is that the difficulty level is relatively limited compared to MATH-V (Wang et al., 2024a), due to the dynamic nature of the questions. Adapting very challenging questions into our program structures requires substantial human effort, which currently prevents us from curating a large number of complex visual math reasoning questions. In the future, we hope to leverage strong foundational models to aid in designing an automatic pipeline for dynamic math question design and generation. 

Furthermore, the selection of seed questions can introduce unintended bias in DYNAMATH dataset. For instance, the most challenging question topic for VLMs, the Puzzle test, only dominates 3.4% of the whole dataset. It remains an open problem to study the bias in open-source datasets and requires further efforts. Regarding the variation generation process, we have identified a limitation: we currently consider only individual types of variants, such as Numerical Value Variants or Function Type Variants, for each seed question. However, in many cases, it is possible to combine different types of variants, such as Color Variants and Numerical Value Variants. We will explore the integration of different variant types to further investigate the reasoning robustness of VLMs. 

**Scalability of DYNAMATH** The current design of DYNAMATH relies heavily on human effort to curate high-quality seed questions. However, it is important to scale up the design process of DynaMATH for constructing more comprehensive and challenging benchmarks. Below, we outline the primary challenges and discuss potential solutions: 

A key challenge in scaling DYNAMATH is incorporating dynamic visual elements for each question. Unlike text-only benchmarks, our dataset includes an image for every problem with different variants (e.g., graphs, geometric shapes, function plots, real-life content). This requires careful design of the drawing program, adding significant manual effort, especially in quality control and verification, which complicates full automation. 

A promising solution is to leverage LLMs to automate the generation of dynamic benchmarks. LLMs have shown proficiency in generating text-based problems and writing code (Mirzadeh et al., 2024). It is possible to break down benchmark topics and subtopics, prompting the LLM to generate diverse problem sets and corresponding Python programs for visual elements. However, the generated problems should be dynamic, with parameterizable Python code to produce multiple image variants. To this end, DYNAMATH is a valuable benchmark since our seed questions can serve as high-quality human demonstrations to guide the LLMs for this task. This LLM-assisted approach could significantly reduce manual effort. However, some human intervention will still be necessary to ensure the selection of correct and high-quality samples from LLMs. 

While we have to leave the LLM-assisted dynamic benchmark generation as a future work, DYNAMATH can serve as a good baseline that is completely crafted by human beings, and future work on automated dynamic benchmark generation may compare to DYNAMATH in terms of diversity and quality. 

**Future Work** Moving forward, an intriguing approach to enhance VLM robustness involves leveraging adversarial training (Zhou et al., 2024; Schlarmann et al., 2024) on DYNAMATH, or utilizing reinforcement learning from human feedback (Ouyang et al., 2022; Sun, 2023; Rafailov et al., 2024; Yang et al., 2024b) with fine-grained process rewards (Uesato et al., 2022; Wang et al., 2024b; Luo et al., 2024), or more robust rewards (Yang et al., 2024a; Zhang et al., 2024c). While prior successes in robust machine learning (Silva & Najafirad, 2020; Zhang et al., 2020; Yang et al., 2022) and trustworthy LLMs (Sun et al., 2024) offer valuable insights, adapting these methods to VLMs in a cost-effective way remains an open challenge and an area of potential exploration. 

## B VARIATION TYPES OF DYNAMATH 

DYNAMATH introduces several types of variations based on the seed questions. In Figure 8, we illustrate six distinct types of variations. This diversity allows our dataset to effectively evaluate the visual robustness of VLMs. 

16 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-09.png)


**----- Start of picture text -----**<br>
(a) Numerical Value Variants (b) Geometric Transformations<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0017-18.png)


**----- Start of picture text -----**<br>
(c) Function Type Variants (d) Symbolic Substitution<br>(e) Real-life Contexts Variants (f) Graph Structure Variants<br>**----- End of picture text -----**<br>


Figure 8: Variantion types considered in our DYNAMATH benchmark 

## C DETAILED EXPERIMENT SETUP 

In this section, we provide more details about our experiment designs. 

## C.1 PROMPTS FOR RESPONSE GENERATION 

In our experiments, we prompt the VLMs to generate responses to different types of questions, such as multiple choice, float, and text types. The prompts used for these question types are shown in Table 5. 

## C.2 PROMPTS FOR ANSWER EXTRACTION 

To simplify the answer extraction process, we use the following prompts to extract the answer and reload it in JSON format, which can be easily used for template matching with ground truth answers: 

17 

Published as a conference paper at ICLR 2025 

|**Answer type**|**prompt**|
|---|---|
|||
|multiple choice|If the problem is a multiple choice problem, just provide the correspon-<br>ing choice option, such as ’A’, ’B’, ’C’, or ’D’.|
|foat<br>If the answer is a numerical value, format it as a three-digit foating-<br>point number.||
|||
|text|Please answer the question in the following form: (specifc requirement<br>in question).|



Table 5: The prompt for different questions and answer types in answer generation. 

Response Instruction 

Please provide an answer to the question outlined above. Your response should adhere to the following JSON format, which includes two keys: ’solution’ and ’short answer’. The ’solution’ key can contain reasoning steps needed to solve the question. The ’short answer’ key should only provide a concise response. Example of expected JSON response format: 

{ "solution": "[Detailed step-by-step explanation]", "short answer": "[Concise Answer]" } 

Closed-source models can consistently extract answers and regenerate them in JSON format. In contrast, some open-source models, particularly those with relatively low parameter sizes, struggle to output JSON format reliably. We have identified patterns in their inconsistent responses, which deviate from the expected code scripts. By supplementing these scripts, we efficiently check, annotate, and extract informal answers. 

## C.3 PROMPTS FOR FEW-SHOT CHAIN OF THOUGHT 

Below we provide the few-shot CoT prompt for plane geometry topic. More demonstration examples for each topic can be found in Appendix E 

18 

Published as a conference paper at ICLR 2025 

Few-shot CoT Prompt 

Please answer the question based on the last image. To clearly explain the task, we provide the following example: 

For image 1: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Question: As shown in the figure, _AB_ is the diameter of the semicircle, point _O_ is the center of the circle, point _C_ is a point on the extended line of _AB_ , and _CD_ is tangent to the semicircle at point _D_ . If _AB_ = 6 _._ 0, _CD_ = 4 _._ 0, then the value of sin(∠ _C_ ) is () 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0019-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0019-05.png)


Answer: (B) 

Let’s think this step by step: 

To solve this problem, we can make use of the properties of the circle and the tangent line. Here’s a step-by-step breakdown of the approach: 

1. **Recognize the Geometry:** - _AB_ is the diameter of the semicircle, so the length of _AB_ is 6. This makes the radius of the circle _AO_ = _BO_ = 3. - _CD_ is tangent to the semicircle at point _D_ . Tangents to a circle are perpendicular to the radius at the point of tangency, meaning _OD ⊥ CD_ . 

2. **Set Up the Triangle:** - We are asked to find sin(∠ _C_ ). The triangle we should focus on is triangle _ODC_ . - By the properties of a tangent, ∠ _ODC_ = 90 _[◦]_ . 

3. **Apply Pythagoras’ Theorem:** - In the right triangle _ODC_ , we have the lengths _OD_ = 3 (since it’s the radius of the circle), and _CD_ = 4 (given). - To find _OC_ , use the Pythagorean theorem: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0019-12.png)


4. **Find** sin(∠ _C_ ) **:** - From triangle _ODC_ , sin(∠ _C_ ) = _[OD] OC_[.] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0019-14.png)


Thus, the correct option is _B_ . 

For image 2: ... 

For image 3: ... 

Now please answer the following question based on the last image: Find the perimeter of the orange triangle. Please answer in a floating-point number. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0019-19.png)


19 

Published as a conference paper at ICLR 2025 

## C.4 MODEL HYPERPARAMETERS 

We set all parameters except temperature to their default values. We set temperature = 0 for closedsource models and open-source models to reduce the randomness in the model generation. Table 6 displays the parameters we used for generation in VLMs. 

Table 6: Hyperparameters for various VLMs. 

|**Model**|**Hyperparameters**|
|---|---|
|GPT-4o|model =gpt-4o-0806, temperature = 0.0, max<br>~~t~~okens = 4096|
|Claude-3.5|model =claude-3-5-sonnet-20240620, temperature = 0.0, max<br>tokens = 1024|
|Gemini Pro 1.5|model =gemini-1.5-pro, temperature = 0.0, max<br>~~t~~okens = 8192|
|Qwen2-VL-72B|model =Qwen/Qwen2-VL-72B-Instruct, temperature = 0.0, max<br>tokens = 2048|
|QWen2-VL-7B|model =Qwen/Qwen2-VL-7B-Instruct, temperature = 0.0, max<br>~~t~~okens = 2048|
|InternVL2-76B|model =OpenGVLab/InternVL2-Llama3-76B, temperature = 0.0, max<br>~~t~~okens = 1024|
|InternVL2-40B|model =OpenGVLab/InternVL2-40B, temperature = 0.0, max<br>tokens = 1024|
|InternVL2-26B|model =OpenGVLab/InternVL2-26B, temperature = 0.0, max<br>tokens = 1024|
|InternVL2-8B|model =OpenGVLab/InternVL2-8B, temperature = 0.0, max<br>tokens = 1024|
|Deepseek-VL-7B-chat|model =deepseek-ai/deepseek-vl-7b-chat, temperature = 0.0, max<br>~~t~~okens = 1024|
|Llama-3.2-90B|model =meta-llama/Llama-3.2-90B-Vision-Instruct, temperature = 0.0, max<br>tokens = 1024|
|Llava-v1.6-34B|model =liuhaotian/llava-v1.6-34b, temperature = 0.0, max<br>~~t~~okens = 1024|
|Llava-v1.6-vicuna-13B|model =liuhaotian/llava-v1.6-vicuna-13b, temperature = 0.0, max<br>tokens = 1024|
|Llava-v1.5-7B|model =liuhaotian/llava-v1.5-7b, temperature = 0.0, max<br>tokens = 1024|



20 

Published as a conference paper at ICLR 2025 

## D VARIANT EXAMPLES FOR DIFFERENT TOPICS IN DYNAMATH 

In this section, we show sample problems in DYNAMATH for different topics including multiple variants, including Solid Geometry (SG), Puzzle Tests (PT), Arithmetic (AR), Scientific Figures (SF), Graph Theory (GT), Algebra (AL), Plane Geometry (PG), Analytic Geometry (AG), and Statistics (ST). 

## Topic: Solid Geometry (SG) 

**Q129 from DYNAMATH** : What is the volume of this azure right square pyramid? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0021-05.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q188 from DYNAMATH** : Are two planes parallel? choice: (A) Yes (B) No 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0021-07.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q320 from DYNAMATH** : Which line is longer, the pink or the red line? choice: (A) pink (B) red (C) Their lengths are the same. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0021-09.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


21 

Published as a conference paper at ICLR 2025 

Topic: Puzzle test (PT) 

**Q115 from DYNAMATH** : The sum of the three numbers on each of the two lines of the cross is 76. Find the number in the center. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0022-03.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q282 from DYNAMATH** : Fill in the white spaces to make the equations work. choice: (A) 13, 25, 5, and 12 (B) 25, 5, 12, and 12 (C) 13, 4, 25, 13. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0022-05.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


- **Q284 from DYNAMATH** : Find the missing value. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0022-07.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


22 

Published as a conference paper at ICLR 2025 

## Topic: Arithmetic (AR) 

**Q7 from DYNAMATH** : In the addition sum to the right, three digits have been replaced with star. What is the value of star? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0023-03.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q25 from DYNAMATH** : What is the missing computed symbol? Choices: (A) + (B) - (C) * (D) / 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0023-05.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q316 from DYNAMATH** : According to the boarding pass, how long is the flight time of this airplane? Answer the question using the total number of minutes. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0023-07.png)


**----- Start of picture text -----**<br>
Variant 1<br>Variant 2<br>Variant 2<br>Variant 3<br>**----- End of picture text -----**<br>


23 

Published as a conference paper at ICLR 2025 

## Topic: Scientific figure (SF) 

**Q323 from DYNAMATH** : Two containers of the same gas (ideal) have these masses and temperatures Which box has atoms with the largest average thermal energy? choice: (A) A (B) B (C) Their average thermal energy is the same. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0024-03.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q325 from DYNAMATH** : Three equally spaced identical long straight wires carry different currents. In which direction will the middle wire try to move when the currents are switched on? choice: (A) to the left (B) to the right (C) stay the same 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0024-05.png)


**----- Start of picture text -----**<br>
Variant 3<br>Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q331 from DYNAMATH** : The graph shows the force on an object of mass M as a function of time. For the time interval 0 to 10 s, what is the total change in the momentum of the object? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0024-07.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


24 

Published as a conference paper at ICLR 2025 

## Topic: Graph theory (GT) 

**Q42 from DYNAMATH** : Is the graph shown connected? choice: (A) Yes (B) No 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0025-03.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q137 from DYNAMATH** : What is the first edge added to the MST when running Kruskal’s Algorithm? In the case of a tie, choose the edge which comes first in alphabetical order i.e. if you had to choose between AS and AE, then you would choose AE first. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0025-05.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q259 from DYNAMATH** : The tree shown in image reserves an expression. Calculate this expression and output the result. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0025-07.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


25 

Published as a conference paper at ICLR 2025 

## Topic: Algebra (AL) 

**Q305 from DYNAMATH** : The store has 4 combinations of candies. Each candy type has the same price. Find the price of the fourth combination. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0026-03.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q351 from DYNAMATH** : Which function has the highest order or growth? choice: (A) f1 (B) f2 (C) f3 (D) f4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0026-05.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q465 from DYNAMATH** : 210 customers were surveyed about their product preferences. The results are displayed in the Venn diagram below. How many more customers prefer only Non-Organic products than only Organic ones? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0026-07.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


26 

Published as a conference paper at ICLR 2025 

## Topic: Plane geometry (PG) 

**Q28 from DYNAMATH** : The two rectangles shown in the picture have the same area. what is the ratio _x_ : _y_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0027-03.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


- **Q43 from DYNAMATH** : What fraction of the shape is azure? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0027-05.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>Q53 from DYNAMATH : What is the area of blue ring?<br>Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


27 

Published as a conference paper at ICLR 2025 

## Topic: Analytic geometry (AG) 

**Q68 from DYNAMATH** : What is the green curve? choice: (A) a parabola (B) a line (C) a logarithmic function (D) a trigonometric function. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0028-03.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q87 from DYNAMATH** : What is the limit of the function as x approaches 1 from the left side? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0028-05.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q111 from DYNAMATH** : The image shows the derivative of _f_ ( _x_ ). Where is the local max of _f_ ( _x_ ) at? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0028-07.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


28 

Published as a conference paper at ICLR 2025 

## Topic: Statistics (ST) 

**Q72 from DYNAMATH** : According to the markov chain shown in the image, what is the probability of the event ’A to B’? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0029-03.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


- **Q161 from DYNAMATH** : On which number is the spinner more likely to land? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0029-05.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


**Q447 from DYNAMATH** : The Gantt chart below represents different tasks. Which task starts the earliest? Choices: (A) Task A (B) Task B (C) Task C (D) Task D (E) Task E 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0029-07.png)


**----- Start of picture text -----**<br>
Variant 1 Variant 2 Variant 3<br>**----- End of picture text -----**<br>


29 

Published as a conference paper at ICLR 2025 

## E MORE DEMONSTRATION EXAMPLES FOR FEW-SHOT EXPERIMENT 

## Demonstration Examples for Topic Algebra 

The two markers with a question mark have the same number. Which number do you have to put instead of the question mark so that the calculation is correct? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0030-04.png)


Answer: 3 

Let’s think this step by step: To solve the problem: The equation is: 10+?+? + 2 = 18 Where both question marks represent the same number. Let’s denote the number as _x_ . The equation becomes: 10 + _x_ + _x_ + 2 = 18 Combine like terms: 12 + 2 _x_ = 18. 

Now, solve for _x_ : 2 _x_ = 18 _−_ 12, hence 2 _x_ = 6, and therefore _x_ = 3. So, each question mark should be replaced with 3 to satisfy the equation. 

## Demonstration Examples for Topic Arithmetic 

How many plums (see the picture) weigh as much as an apple? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0030-10.png)


Answer: 3 

Let’s think this step by step: 

Since the scale is balanced, the total weight on the left side equals the total weight on the right side. 1. Let the weight of one apple be _A_ , and the weight of one plum be _P_ . 

2. The total weight on the left side of the scale: 3 _A_ + 2 _P._ 

3. The total weight on the right side of the scale: _A_ + 8 _P._ 

Since the scale is balanced, we can set the two expressions equal to each other: 3 _A_ + 2 _P_ = _A_ + 8 _P._ 

### Solving for _A_ in terms of _P_ : 

1. Subtract _A_ from both sides: 2 _A_ + 2 _P_ = 8 _P._ 

2. Subtract 2 _P_ from both sides: 2 _A_ = 6 _P._ 

3. Divide both sides by 2: _A_ = 3 _P._ 

### Final Answer: One apple weighs as much as 3 plums. 

30 

Published as a conference paper at ICLR 2025 

Demonstration Examples for Scientific Figure 

If the population of the red squirrels increased, what would happen to the amount of pine available? 

- (A) The amount of pine available would remain the same. 

- (B) The amount of pine available would decrease. 

- (C) The amount of pine available would double. 

- (D) The amount of pine available would increase. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0031-07.png)


Answer: (B) 

Let’s think this step by step: In the food web of Isle Royale National Park, the red squirrels are primary consumers that feed on pine, which is a primary producer. 

If the population of red squirrels increases, the consumption of pine would increase because more squirrels would be feeding on it. As a result, the amount of pine available would decrease. 

**Conclusion** : If the population of red squirrels increased, the amount of pine available would decrease. Thus, the correct option is _B_ . 

31 

Published as a conference paper at ICLR 2025 

## Demonstration Examples for Topic Puzzle Test 

Please determine the total weight in the fourth image. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0032-03.png)


Answer: 27 

To solve the puzzle and determine the total weight in the fourth image, let’s break it down based on the given weights: 

1. **First image** : A frog and a sheep together weigh 10 kg. 

2. **Second image** : A horse and a frog together weigh 20 kg. 

3. **Third image** : A horse and a sheep together weigh 24 kg. 

- We can now set up three equations to represent these situations: - Frog + Sheep = 10 kg (Equation 1) 

- Horse + Frog = 20 kg (Equation 2) 

- Horse + Sheep = 24 kg (Equation 3) 

Solving the system of equations: 

1. From Equation 2, we can express the weight of the horse: 

- Horse = 20 kg - Frog (from Equation 2). 

2. Substitute Horse in Equation 3: 

- (20 kg - Frog) + Sheep = 24 kg. 

- 20 kg - Frog + Sheep = 24 kg. - Sheep - Frog = 4 kg (Equation 4). 

3. Now use Equation 1 (Frog + Sheep = 10 kg) and Equation 4 (Sheep - Frog = 4 kg): - Adding these two equations: (Frog + Sheep) + (Sheep - Frog) = 10 kg + 4 kg, gives: 2 × Sheep = 14 kg, so Sheep = 7 kg. 

4. Substitute the value of Sheep (7 kg) into Equation 1: 

- Frog + 7 kg = 10 kg, so Frog = 3 kg. 

5. Finally, substitute the value of Frog (3 kg) into Equation 2: - Horse + 3 kg = 20 kg, so Horse = 17 kg. Total weight in the fourth image: - The fourth image has the horse, the sheep, and the frog together. - The total weight is: Horse (17 kg) + Sheep (7 kg) + Frog (3 kg) = 27 kg. Thus, the total weight in the fourth image is 27 kg. 

## F ERROR TYPE EXAMPLES 

In this section, we present various examples of error types encountered in Claude 3.5, including figure reading error, calculation error, reasoning error, hallucination error, and knowledge error. 

32 

Published as a conference paper at ICLR 2025 

## Figure Reading Error 

**Q96 from DYNAMATH** : What is the distance between the x-intercept and y-intercept of blue line? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0033-03.png)


## **Claude-3.5 Answer** : 

1. Identify the _x_ -intercept and y-intercept: The _x_ -intercept is where the line crosses the _x_ -axis ( _y_ = 0). From the graph, this appears to be at (3 _,_ 0). The _y_ -intercept is where the line crosses the _y_ -axis ( _x_ = 0). From the graph, this appears to be at (0 _,_ 4). 2. Calculate the distance using the distance formula: distance = ~~�~~ ( _x_ 2 _− x_ 1)[2] + ( _y_ 2 _− y_ 1)[2] where ( _x_ 1 _, y_ 1) is (3 _,_ 0) and ( _x_ 2 _, y_ 2) is (0 _,_ 4). 3. distance = ~~�~~ (0 _−_ 3)[2] + (4 _−_ 0)[2] = 5 _._ Therefore, the distance between the _x_ -intercept and _y_ -intercept is 5 units. **Ground Truth** : Distance = _√_ 4 _≈_ 2 _._ 828. 

**Error Analysis:** The model misreading the coordinate of the points and gives the wrong answer. 

## Figure Reading Error 2 

**Q185 from DYNAMATH** : Based on the measurement results shown in the diagram, this blue angle is ( ). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0033-09.png)


**Claude-3.5 Answer** : To determine the angle represented by the blue lines in the protractor image: 

1. Observe that the protractor is a 180-degree semicircle. 

2. Note that one blue line is at the 0-degree mark. 

3. Locate where the other blue line intersects the protractor scale. 

4. This intersection occurs at approximately 25 degrees. 

5. Since the protractor measures angles from 0 to 180 degrees, the angle formed by the blue lines is directly read as 25 degrees. 

Therefore, the measurement of the blue angle is approximately 25 degrees. 

**Ground Truth** : 150 degrees. 

**Error Analysis:** The model fail to distinguish between two set of the numbers for degree reading. In addition, even for outer degree numbers, the model gives an inaccurate number reading. 

33 

Published as a conference paper at ICLR 2025 

## Calculation Error 

**Q108 from DYNAMATH** : The function f(x) is a cubic function. What is the value of f(x) when x = 4? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0034-03.png)


## **Claude-3.5 Answer** : 

1. First, let’s recognize that for a cubic function _f_ ( _x_ ) = _ax_[3] + _bx_[2] + _cx_ + _d_ , we need to find the values of _a_ , _b_ , _c_ , and _d_ . 

2. We can use the given points to set up a system of equations: _f_ (0) = _d_ = 3, _f_ (1) = _a_ + _b_ + _c_ + 3 = 19, 

_f_ (2) = 8 _a_ + 4 _b_ + 2 _c_ + 3 = 63, 

_f_ (3) = 27 _a_ + 9 _b_ + 3 _c_ + 3 = 159. 

3. From the first equation, we know _d_ = 3. 

4. Subtracting equations: 

_f_ (1) _− f_ (0) : _a_ + _b_ + _c_ = 16 

_f_ (2) _− f_ (1) : 7 _a_ + 3 _b_ + _c_ = 44 

_f_ (3) _− f_ (2) : 19 _a_ + 5 _b_ + _c_ = 96 

5. Solving this system, we get: _a_ = 4, _b_ = 7, _c_ = 5, _d_ = 3 

6. So, our function is _f_ ( _x_ ) = 4 _x_[3] + 7 _x_[2] + 5 _x_ + 3 

7. Now, we can calculate _f_ (4): _f_ (4) = 4(4[3] )+7(4[2] )+5(4)+3 = 256+112+20+3 = 391 Therefore, _f_ (4) = 391. 

**Ground Truth** : Let _f_ ( _x_ ) = _ax_[3] + _bx_[2] + _cx_ + _d_ , using the number given in the table, one can get: _a_ = 4, _b_ = 2, _c_ = 10, _d_ = 3. So, our function is _f_ ( _x_ ) = 4 _x_[3] + 2 _x_[2] + 10 _x_ + 3 and _f_ (4) = 331. 

**Error Analysis:** The model fails to solve the system of equations for value _b_ and _c_ . 

34 

Published as a conference paper at ICLR 2025 

Reasoning Error 1 

**Q146 from DYNAMATH** : Find the missing number. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0035-03.png)


## **Claude-3.5 Answer** : 

1. The diagram shows a central number 4 with six numbers radiating from it. 

2. The visible numbers are 7, 17, 28, 68, and 71. 

3. Let’s check if there’s a consistent mathematical relationship between 4 and these numbers: 4 x 7 = 28 (one of the given numbers) 4 x 17 = 68 (another given number) 

4. This suggests that each number is either a factor or multiple of 4. 

5. 71 doesn’t fit this pattern directly, but 71 + 4 = 75, which is 3 x 25. 

6. Following this pattern, the missing number should also be related to 4 by multiplication or addition. 

7. A number that fits this pattern and completes the sequence is 13, because: 4 x 13 = 52, and 52 + 4 = 56, which is 4 x 14. 

8. Therefore, the missing number is likely 13. 

## **Ground Truth** : 71 _×_ 4 = 284. 

**Error Analysis:** The first 4 steps from the model are correct, and the model can observe that each number is either a factor or multiple of 4 correctly. However, step 5-8 is not correct and the reasoning to get its final answer 13 is not correct. 

35 

Published as a conference paper at ICLR 2025 

Reasoning Error 2 

**Q34 from DYNAMATH** : What is the area of this parallelogram? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0036-03.png)


- **Claude-3.5 Answer** : To find the area of a parallelogram, we use the formula: Area = base _×_ height. In this case: 

1. The base is given as 19 ft. 

2. The height is the perpendicular distance between the base and the opposite side, which is 8 ft. 

Area = 19 ft _×_ 8 ft = 152 sq ft. 

**Ground Truth** : To find the area of a parallelogram, we first need to compute the height _h_ = _√_ 8[2] _−_ 3[2] _≈_ 7 _._ 416. Then the area of the parallelogram is 19 _×_ 7 _._ 416 = 140 _._ 904 sq ft. 

**Error Analysis:** The model fail to reason that we need to compute the height using the given numbers in the plot. 

Hallucination Error 1 

**Q146 from DYNAMATH** : Find the area of the shaded part. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0036-12.png)


**Claude-3.5 Answer** : 

1. The circle has a radius r = 2. 

2. The shaded area is a sector of the circle minus a right triangle. 

3. The sector angle is 1/8 of the full circle (45 _[◦]_ ). 

4. Area of the sector = (1 _/_ 8) _πr_[2] = 1 _._ 5708. Area of the triangle = (1 _/_ 2) _r_[2] = (1 _/_ 2)2[2] = 2. 

6. Shaded area = Area of sector - Area of triangle = 1 _._ 5708 _−_ 2 = _−_ 0 _._ 42927. 

5. Taking the absolute value: _| −_ 0 _._ 42927 _|_ = 0 _._ 4292. 

**Ground Truth** : Area of the shaded part = (1 _/_ 4) _πr_[2] _−_ (1 _/_ 2) _r_[2] = 1 _._ 1416 _._ 

**Error Analysis:** The model hallucinates the sector angle to be 45 _[◦]_ , which should be 90 _[◦]_ . In addition, instead of realizing that the area should be a positive number, the model take the absolute value of a negative number and return it as the answer. 

36 

Published as a conference paper at ICLR 2025 

Hallucination Error 2 

**Q120 from DYNAMATH** : Is this graph a cyclic graph? choice: (A) Yes (B) No 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0037-03.png)


## **Claude-3.5 Answer** : 

To determine if this graph is cyclic, we need to check if there’s a path that starts and ends at the same node without repeating any edges. Looking at the graph: 

1. It has 5 nodes connected in a star-like pattern. 

2. The central node is connected to all other nodes. 

3. There are no connections between the outer nodes. 

4. To form a cycle, we would need to be able to start at one node, traverse through others, and return to the starting node without repeating edges. 

5. In this graph, any path that tries to return to a starting node would have to go through the central node twice, which means repeating an edge. 

6. Therefore, there is no cycle in this graph. 

Since there is no cycle, this graph is not cyclic. The final answer is (B) No. 

**Ground Truth** : Since there is a path that starts and ends at the same node without repeating any edges, this is a cyclic graph. Hence the correct solution is (A) Yes. 

**Error Analysis:** The model hallucinates the path in the graph and concludes to the wrong answer. 

37 

Published as a conference paper at ICLR 2025 

Knowledge Error 

**Q289 from DYNAMATH** : The diagram shows Exam score of students. What is the average score of this exam? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0038-03.png)


## **Claude-3.5 Answer** : 

1. Sum up all the scores: 3 + 3 + 3 + 3 + 0 + 1 + 2 + 2 + 1 + 1 + 3 + 3 + 3 + 4 + 2 + 2 + 6 + 2 + 4 + 4 = 52 

2. Count the number of students: There are 9 stem numbers, representing 9 students. 

3. Calculate the average: Average = Total Score / Number of Students = 52 / 9 = 5.777777... 

4. Round to three decimal places: 5.778. 

**Ground Truth** : 54.6. 

**Error Analysis:** The model fails to read stem-and-leaf plot. 

## G QUESTION VARIANT EXAMPLES 

We present several examples of generated variants of seed questions from our dataset along with corresponding responses from GPT-4o (Figure 9), Gemini (Figure 10), and Qwen2-VL-72B (Figure 11). These examples illustrate that the leading VLMs exhibit a lack of robustness to variations in image input, demonstrating unreliable perception and reasoning abilities. 

38 

Published as a conference paper at ICLR 2025 

## H ADDITIONAL EXPERIMENT RESULTS 

In this section, we present additional experiments. 

## H.1 REASONING ROBUSTNESS ON DIFFERENT VARIATION TYPES 

In terms of different variant types in DYNAMATH, as shown in Figure 12, we find that both GPT-4o and Qwen2-VL-72B are sensitive to variations in graph structure, geometric transformation, and function type. Additionally, Qwen2-VL-72B is vulnerable to symbolic substitution variants. These weaknesses suggest directions for future improvement of these models. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0039-05.png)


Figure 12: Comparing reasoning robustness ( _RR_ ) across different variation types. 

## H.2 ADDITIONAL FAILURE CASE ANALYSIS 

In this section, we present more results on the failure case analysis. 

**Failure v.s. Difficulty Levels** We conducted an in-depth failure analysis based on problem difficulty, categorized into elementary (63 questions), high school (277 questions), and undergraduate (161 questions) levels. The detailed results are presented in Figure 13. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0039-10.png)


Figure 13: Failure cases across different difficulty levels. 

The results indicate that high school and undergraduate problems account for the majority of failure cases. Among the error types, knowledge errors are the least frequent, implying that VLMs have a solid grasp of mathematical concepts and facts. However, reasoning, hallucination, figure reading, and calculation errors are more prevalent, highlighting that VLMs may struggle with interpreting visual data and performing accurate calculations and reasoning. 

**Failure v.s. Problem Topics** We performed an in-depth analysis of failure cases based on problem types. The detailed results can be found in Figure 14. 

39 

Published as a conference paper at ICLR 2025 

Table 7: Reasoning Robustness _RR_ of 14 models on DYNAMATH with 5,010 generated questions, testing with 0 temperature. “ALL” represents overall accuracy. Question topics (PG, SG, EL, etc) are defined in Table 1 

|Model|ALL|PG<br>SG<br>AG<br>AL<br>PT<br>GT<br>ST<br>SF<br>AR|EL<br>HI<br>UN|
|---|---|---|---|
|_Closed-sourced Large Multimodal Models (LMMs)_||||
|Zero-shot GPT-4o<br>Zero-shot Claude-3.5<br>Zero-shot Gemini Pro 1.5|54.8|66.4<br>64.1<br>42.2<br>71.4<br>22.7<br>32.3<br>55.4<br>56.9<br>75.0|67.1<br>55.5<br>84.5|
||54.9|44.3<br>54.1<br>33.6<br>77.5<br>53.3<br>39.0<br>68.5<br>39.3<br>69.2|73.8<br>53.1<br>94.5|
||44.5|54.2<br>46.9<br>31.8<br>55.4<br>28.6<br>35.1<br>50.5<br>31.0<br>56.7|65.7<br>45.1<br>58.5|
|3-shot CoT GPT-4o<br>3-shot CoT Claude-3.5<br>3-shot CoT Gemini Pro 1.5|49.8|53.7<br>67.4<br>37.5<br>65.3<br>34.5<br>33.7<br>51.9<br>43.8<br>80.0|71.9<br>49.1<br>83.9<br>63.0<br>53.4<br>88.7<br>56.5<br>39.6<br>60.0|
||51.7|55.6<br>55.6<br>22.4<br>68.5<br>0.0<br>17.9<br>71.6<br>47.9<br>55.9<br>51.9<br>58.8<br>25.5<br>53.8<br>27.0<br>32.4<br>41.2<br>32.4<br>56.0||
||40.1|||
|_Open-sourced Large Multimodal Models (LMMs)_||||
|Qwen2-VL-72B<br>Qwen2-VL-72B (3-shot CoT)<br>QWen2-VL-7B<br>InternVL2-76B<br>InternVL2-40B<br>InternVL2-26B<br>InternVL2-8B<br>Llama-3.2-90B<br>Deepseek-VL-7B-chat<br>Llava-v1.6-34B<br>Llava-v1.6-vicuna-13B<br>Llava-v1.5-7B|51.8|56.8<br>68.5<br>30.4<br>54.4<br>0.0<br>37.0<br>62.7<br>47.2<br>78.0|67.4<br>52.8<br>64.8<br>67.7<br>43.8<br>49.9<br>53.3<br>29.1<br>49.1|
||43.4|54.8<br>59.7<br>17.4<br>59.4<br>0.0<br>18.9<br>48.9<br>42.0<br>72.5||
||32.7|54.8<br>17.2<br>18.1<br>37.0<br>0.0<br>27.9<br>32.3<br>27.0<br>49.0||
||45.8|55.4<br>57.7<br>35.3<br>55.1<br>16.7<br>24.5<br>49.2<br>36.3<br>74.6|65.8<br>43.7<br>80.0|
||33.9|45.6<br>31.3<br>23.9<br>32.0<br>0.0<br>27.2<br>37.2<br>30.9<br>50.5<br>54.3<br>0.0<br>16.6<br>25.3<br>0.0<br>40.0<br>38.5<br>28.1<br>66.7<br>38.3<br>53.6<br>15.9<br>33.5<br>0.0<br>24.8<br>20.1<br>28.4<br>41.2<br>46.4<br>53.6<br>19.6<br>16.9<br>0.0<br>27.9<br>29.6<br>33.5<br>12.8<br>48.8<br>0.0<br>11.7<br>0.0<br>0.0<br>31.8<br>16.4<br>9.2<br>25.6<br>48.5<br>52.6<br>14.9<br>13.2<br>0.0<br>12.7<br>17.4<br>24.0<br>33.3<br>53.1<br>0.0<br>17.6<br>0.0<br>0.0<br>9.7<br>8.5<br>0.0<br>0.0<br>37.0<br>0.0<br>10.6<br>0.0<br>0.0<br>12.9<br>4.6<br>0.0<br>35.7|56.1<br>33.9<br>37.2<br>67.1<br>31.9<br>44.2<br>46.6<br>25.1<br>34.9<br>35.0<br>32.2<br>44.8<br>28.1<br>15.2<br>31.1<br>44.2<br>21.3<br>22.4<br>23.4<br>17.5<br>8.8<br>16.8<br>13.6<br>10.6|
||35.0<br>|||
||26.1<br>29.5<br>19.5<br>22.1<br>14.1<br>10.8|||




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0040-03.png)


Figure 14: Failure cases across different problem topics. 

From Figure 14, we have the following observations based on the failure reasons and problem types: 

- The Puzzle test shows a concentration of reasoning errors, with no other error types present, suggesting that VLMs may struggle with the logical and abstract reasoning required for puzzles. 

- Graph theory, analytic geometry, arithmetic, and statistics problems exhibit more errors related to figure reading, indicating difficulties in interpreting visual data. 

- Solid geometry and algebra problems are prone to calculation errors, highlighting potential issues with numerical operations in handling such questions. 

- Plane geometry has high incidences of hallucination and reasoning errors, suggesting challenges in both generating relevant information and applying logical reasoning. 

H.3 DETAILED REASONING ROBUSTNESS RESULTS OF ZERO TEMPERATURE As shown in Table 7, we present the full results of reasoning robustness ( _RR_ ) defined in Eq 2. We can better understand how the reasoning robustness correlates with question types and difficulty levels. 

40 

Published as a conference paper at ICLR 2025 

## H.4 RESULTS OF DIFFERENT PROMPT TEMPLATE 

To investigate other prompt templates, we designed the following prompt aims to improve the reasoning and reduce memorization issues for VLMs: 

## Prompt Template for improving reasoning and reduce memorization 

You are solving advanced visual math problems that require logical reasoning and detailed analysis of the provided image and question. Carefully examine the image and break the problem into smaller steps to ensure accurate and thoughtful reasoning. Avoid relying on memorized answers, patterns, or shortcuts. Instead, justify each step of your solution explicitly based on the information in the image. 

Task: Please answer the following question: _{_ new question _}_ , ensuring your explanation according to the provided image and question. Focus on reasoning rather than recalling. 

We evaluated the performance of GPT-4o and Qwen2-VL-72b on 10 variants with temperature 0 using this newly designed prompt, and the average accuracy rate, worst-case accuracy, and reasoning robustness can be found in Table 8. The results show that both average accuracy and worst-case accuracy have improved with the use of the designed prompt. This suggests that a carefully crafted prompt can enhance the performance of VLMs. However, there is no significant improvement in reasoning robustness, highlighting the ongoing limitations in the robustness of current VLMs. 

Table 8: Performance comparison between Zero-shot and Zero-shot with New Prompt for GPT-4o and Qwen2-VL-72b. 

|**Model**|**Zero-shot**|**Zero-shot**|**Zero-shot**|**Zero-shot w New Prompt**|**Zero-shot w New Prompt**|**Zero-shot w New Prompt**|
|---|---|---|---|---|---|---|
||||||||
||_Aavg_|_Awst_|_RR_|_Aavg_|_Awst_|_RR_|
||||||||
|GPT-4o<br>Qwen2-VL-72b|63.7%<br>55.1%|34.7%<br>28.3%|54.8%<br>51.8%|65.6%<br>57.8%|36.1%<br>29.5%|55.0%<br>51.0%|



H.5 MORE ON MEMORIZATION PHENOMENON 

We also tested the newly designed prompt with problems where memorization was evident. Unfortunately, the model still tends to provide the same answers, regardless of changing conditions: 

- For seed question 78 in DYNAMATH, GPT-4o consistently argues that a shifted absolute function is not differentiable at _x_ = 0. 

- For seed question 12 in DYNAMATH, Claude-3.5-Sonnet repeatedly reads the period of a sinusoidal function as 2 _π_ , regardless of the actual period shown in the image. We believe a more systematic study is necessary to effectively address this issue. 

A screenshot of the web version of GPT-4o and Claude-3.5 for these two examples can be found in Figure 15 and Figure 16. More systematic studies are necessary to effectively address this issue. 

H.6 EVALUATING THE ROBUSTNESS OF DYNAMATH 

An important question to ask is whether dynamic benchmarks are robust enough. In other words, if we provide synthetic data generated by DYNAMATH, can models perform well on other variants of DYNAMATH? The best way to investigate this is to perform thorough experiments, including pre-training and fine-tuning VLMs using DynaMATH. However, due to limited resources, we were unable to perform full-scale pre-training or fine-tuning of VLMs to thoroughly investigate potential data leakage involving DYNAMATH. As a proxy investigation, we conducted an in-context learning experiment. 

Specifically, we used variants 1 to 3 of DYNAMATH as few-shot demonstration examples and tested the VLM’s response to a question from variant 4. As a controlled experiment, we directly used a question from variant 4 both as a demonstration example and test question (i.e., asking the model the same question it was shown). This setup provides a preliminary indication of potential data leakage, as well as the expected performance if the model had memorized the data. We performed 

41 

Published as a conference paper at ICLR 2025 

Table 9: In-context evaluation of DYNAMATH 

|**Model**|**Original Performance**|**Few-shot**|**Controlled Experiment**|
|---|---|---|---|
|GPT-4o<br>64.5%<br>65.3%<br>73.1%<br>Qwen2-72b<br>53.7%<br>57.4%<br>77.0%||||



Table 10: The Variance of Average Accuracy for different models participating 5 repetitions tests with 0 temperature 

|**Model name**|**GPT-4o**|**Gemini**|**Qwen2-72B**|**InternVL2-76B**|
|---|---|---|---|---|
|Variance of Average Accuracy (%)<br>1.86<br>1.26<br>0.89<br>2.12|||||



these experiments on one closed-source model, GPT-4o, and one open-source model, Qwen2-72b. The results can be found in Table 9. 

These results indicate that even with a few variants provided as context, the performance improvement is marginal compared to the original performance and baseline results. Nevertheless, whether pre-training or fine-tuning can “hack” dynamic benchmarks needs more systematic studies, which is important for future work. 

## H.7 VARIANCE OF AVERAGE ACCURACY 

In our main paper, we have reported repetition consistency as a measure of randomness of model output. Here, we also calculate the variance of the average accuracy over five repetitions in Table 10. Specifically, for a set of 501 questions, we conducted five separate evaluations and determined the variance of their average accuracies. The resulting variance for GPT-4o, Gemini, Qwen2-VL, and InternVL2 is minimal, ranging from approximately 1 to 2 percentage points. This small variance enhances the reliability of our results. 

## H.8 MORE RESULTS ON CIRCULAR CONSISTENCY 

In DynaMath, our primary focus is on image-based variants, such as Numerical Value (in the image) Variants and Geometric Transformations, so we initially did not test for circular consistency. Circular consistency applies to only multiple choice questions (MCQ) and the contents of the question are still static; only the order of the choices changed. To address your concern, we evaluated the circular consistency (Liu et al., 2023) of two representative models, GPT-4o and Qwen2-VL 76B, specifically using MCQ questions from DynaMath. Interestingly, both models exhibited high repetition consistency under circular shifts, achieving scores of **90.2% and 92.2%** , respectively. In other words, the model’s output is consistent in most cases regardless of the order of the choices. The current models seem to be robust to the circular shifts in MCQ problems. 

42 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0043-01.png)


**----- Start of picture text -----**<br>
Answer type: ” multiple choice”<br>Question 169<br>The purple and orange curves are f(x) and g(x). Is f(x)g(x) even or odd?  Subject: ” analytic geometry”<br>Choice: (A) odd (B) even (C) neither Level: ”high school”<br>Response from GPT-4o:<br>Variant 1 Variant 2<br>To determine if the product of the functions f(x) and  To determine if the function f(x)g(x) is even or odd, we<br>g(x) is even or odd, we need to analyze the  need to analyze the properties of the functions f(x) and<br>properties of each function. g(x).<br>Check if f(x) is even or odd: A function f(x) is even  Even Function: A function h(x) is even if h(-x) = h(x)<br>if f(-x) = f(x) for all x. It is odd if f(-x) = -f(x)  for all x. Odd Function: A function h(x) is odd if h(-x)<br>for all x. = -h(x) for all x.<br>• The purple curve seems to be symmetric about the  From the graph: The purple curve, f(x), is symmetric<br>origin, suggesting that f(x) is likely odd.  about the y-axis, indicating that f(x) is even: f(-x) =<br>• Examining the orange curve, it appears to be a line  f(x). The orange curve, g(x), is symmetric about the<br>that passes through the origin with a consistent  origin, indicating that g(x) is odd: g(-x) = -g(x).<br>slope, indicating that g(x) is likely odd. Now, let's consider the<br>Determine the product  product f(x)g(x): f(-x)g(-x) =<br>f(x)g(x): If both f(x) and  f(x)(-g(x)) = -f(x)g(x).<br>g(x) are odd, their product  This shows that f(x)g(x) = -<br>f(x)g(x) is even, since (odd)  f(x)g(x), which means the<br>* (odd) = even.  product function is odd.<br>Therefore, f(x)g(x) is     Thus, f(x)g(x) is<br>(B) even. (A) odd.<br>Variant 3 Variant 4<br>To determine if the product of f(x) and g(x) is even  To determine whether the product f(x)g(x) is even or odd,<br>or odd, we observe each function: we need to analyze the nature of the functions f(x) and<br>• The orange curve, g(x), is a linear function  g(x) individually.<br>passing through the origin, suggesting it is odd  A function is even if f(-x) = f(x) for all x in its<br>because g(-x) = -g(x). domain. An even function is symmetric about the y-axis.<br>• The purple curve, f(x), is a horizontal line,  From the graph, the purple curve f(x) is symmetric around the y-axis, indicating it is an even function.<br>suggesting it is constant. If constant, f(x) is an<br>even function because f(-x) = f(x). A function is odd if f(-x) = -f(x) for all x in its<br>domain. An odd function is symmetric about the origin.<br>The product of an even function and an odd function  The orange curve g(x) appears to have origin symmetry,<br>is always odd because f(-x)g(-x) = f(x)(-g(x)) = - suggesting it is an odd function.<br>f(x)g(x), which satisfies the<br>The product of an even<br>definition of an odd function. function and an odd<br>Therefore, f(x)g(x) is     function is always odd.<br>This is because:<br>(B) even.<br>f(-x)g(-x) = f(x)(-g(x))<br>= -f(x)g(x).<br>Therefore, the function<br>f(x)g(x) is  (A) odd .<br>Variant 5 Variant 6<br>To determine if the product f(x)g(x) is even or odd,<br>To determine if the product f(x)g(x) is even, odd, or<br>we need to check whether f(x) and g(x) have specific<br>neither, we need to analyze the properties of the<br>symmetry properties.<br>functions f(x) and g(x).<br>A function is even if f(-x) = f(x) for all x in its<br>A function h(x) is even if h(-x) = h(x) and odd if h(-x)<br>domain, and it is odd if f(-x) = -f(x). From the graph:<br>= -h(x). From the graph, it appears:<br>The purple curve representing f(x) appears symmetric<br>about the origin, suggesting it could be an odd  • The purple curve f(x) is symmetric about the origin,<br>function. The orange curve representing g(x) is  suggesting it might be an odd function.<br>constant across the y-axis, suggesting it is an even function (g(x) = g(-x) since its value doesn't change).  • The orange curve g(x) is symmetric about the y-axis,<br>indicating it might be an even function.<br>For the product f(x)g(x) to be even, f(x)g(x)<br>f(-x)g(-x); for it to be  The product of an even<br>odd, f(x)g(x) = -f(-x)g(-x). function and an odd<br>function is always odd.<br>Since we have one even<br>function (g(x)) and one odd  Therefore, f(x)g(x) is<br>function (f(x)), their  (A) odd .<br>product, f(x)g(x), will be<br>odd. Therefore, f(x)g(x) is<br>(A) odd .<br>**----- End of picture text -----**<br>


Figure 9: Example of the generated variants of Question 169 and the corresponding responses from GPT-4o. 

43 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0044-01.png)


**----- Start of picture text -----**<br>
Answer type: “multiple choice”<br>Question 75<br>Are the red line and the blue line parallel?  Subject: “analytic geometry”<br>Choice: (A) Yes (B) No Level: “high school”<br>Response from Gemini:<br>Variant 1 Variant 2<br>Parallel lines have the same<br>Parallel lines have the<br>slope. The red line passes<br>same slope. The red line  through points (0,-2) and<br>has a slope of -2 and  (1,1).  Its slope is (1 - (-<br>the blue line has a  2))/(1 - 0) = 3/1 = 3. The<br>slope of approximately -<br>blue line passes through<br>4. Since the slopes are  points (1,0) and (2,3). Its<br>different, the lines are  slope is (3-0)/(2-1) = 3/1 =<br>not parallel.<br>3. Since the slopes are the<br>Therefore, the answer is same, the lines are parallel.<br>(B) No. Therefore, the answer is<br>(A) Yes.<br>Variant 3 Variant 4<br>Parallel lines have the<br>same slope. The red line is  Parallel lines have the<br>decreasing, indicating a  same slope. The red line<br>negative slope, while the  has a positive slope and<br>blue line is horizontal,  the blue line has a<br>meaning it has a slope of 0.  negative slope. Therefore,<br>Since the lines have  they are not parallel.<br>different slopes, they are<br>Therefore, the answer is<br>not parallel.<br>(B) No.<br>Therefore, the answer is<br>(B) No.<br>Variant 5 Variant 6<br>Parallel lines have the  Parallel lines have the<br>same slope. The red line  same slope. The red line<br>goes down 6 units for  is horizontal and has a<br>every 1 unit it moves to  slope of 0. The blue line<br>the right, so its slope  is not horizontal and has<br>is -6. The blue line also  a nonzero slope.<br>goes down 6 units for  Therefore, the red line<br>every 1 unit it moves to  and blue line are not<br>the right, so it also has  parallel.<br>a slope of -6.  Since<br>both lines have the same  Therefore, the answer is<br>slope, they are parallel.<br>(B) No.<br>Therefore, the answer is<br>(A) Yes.<br>**----- End of picture text -----**<br>


Figure 10: Example of the generated variants of Question 75 and the corresponding responses from Gemini. 

44 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0045-01.png)


**----- Start of picture text -----**<br>
Answer type: ” float”<br>Question 346<br>Subject: ” analytic geometry”<br>What is the global minimum of this function?<br>Level: ”high school”<br>Response from Qwen2-VL-72B:<br>Variant 1 Variant 2<br>The function shown in the  The global minimum of the<br>function appears to be at<br>image is a sinusoidal<br>the point where the<br>wave. The global minimum  function reaches its<br>of a sinusoidal wave is<br>lowest value. By examining<br>the lowest point on the<br>the graph, the lowest<br>wave. From the graph, it<br>can be observed that the  point seems to be at<br>approximately x = - π /2.<br>global minimum value of<br>the function is  At this point, the<br>function value is<br>approximately -2.<br>approximately -6.<br>Variant 3 Variant 4<br>The function appears to  The global minimum of<br>be a periodic function  the function occurs at<br>with a period of 2 π.  the lowest point on<br>Observing the graph, the  the graph. Observing<br>function reaches its  the graph, the lowest<br>minimum value at x = π /2,  point is at the y-<br>3 π/2, and so on. The  coordinate of -1.<br>minimum value of the<br>function is approximately<br>-7.<br>Variant 5 Variant 6<br>The global minimum of the<br>The function appears  function is observed at<br>to be a cosine  the lowest point of the<br>function. The global  graph. By examining the<br>minimum of a cosine  graph, it appears that the<br>function is -1.  lowest point is at the<br>However, since this  bottom of the trough,<br>graph is vertically  which occurs at<br>stretched, the global  approximately x = 0. The<br>minimum is -6. value of the function at<br>this point is -8.<br>Variant 7 Variant 8<br>The global minimum of the  The global minimum of<br>function can be found by  the function appears<br>observing the graph. The  to be at<br>function reaches its  approximately -6.5,<br>lowest value at the  observed at the<br>points where it dips the  lowest point of the<br>most. By looking at the  graph.<br>graph, we can see that<br>the function reaches its<br>minimum value at<br>approximately -5.00.<br>**----- End of picture text -----**<br>


Figure 11: Examples of the generated variants of Question 346 and the corresponding responses from Qwen2-VL-72B are provided. Notably, variant 3 derives the correct answer but has an erroneous perception. 

45 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0046-01.png)


Figure 15: Memorization phenomenon on GPT-4o with seed question 78 in DYNAMATH. 

46 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models_images/iclr_2025_DynaMath_A_Dynamic_Visual_Benchmark_for_Evaluating_Mathematical_Reasoning_Robustness_of_Vision_Language_Models.pdf-0047-01.png)


Figure 16: Memorization phenomenon on Claude-3.5 with seed question 12 in DYNAMATH. 

47 

