# **MARVEL: Multidimensional Abstraction and Reasoning through Visual Evaluation and Learning** 

**Yifan Jiang**[1] _[∗]_ **Jiarui Zhang**[1] _[∗]_ **Kexuan Sun**[1] _[∗]_ **Zhivar Sourati**[1] **Kian Ahrabian**[1] **Kaixin Ma**[2] **Filip Ilievski**[3] **Jay Pujara**[1] 

1Information Sciences Institute, University of Southern California 

2Tencent AI Lab, Bellevue, WA 

3Department of Computer Science, Faculty of Science, Vrije Universiteit Amsterdam _{_ `yjiang44,jzhang37,kexuansu,souratih,ahrabian` _}_ `@usc.edu` 

`kaixinma@global.tencent.com` , `f.ilievski@vu.nl` , `jpujara@isi.edu` 

## **Abstract** 

While multi-modal large language models (MLLMs) have shown significant progress on many popular visual reasoning benchmarks, whether they possess abstract visual reasoning abilities remains an open question. Similar to the Sudoku puzzles, abstract visual reasoning (AVR) problems require finding high-level patterns (e.g., repetition constraints) that control the input shapes (e.g., digits) in a specific task configuration (e.g., matrix). However, existing AVR benchmarks only considered a limited set of patterns (addition, conjunction), input shapes (rectangle, square), and task configurations (3 _×_ 3 matrices). To evaluate MLLMs’ reasoning abilities comprehensively, we introduce **MARVEL** , a multidimensional AVR benchmark with 770 puzzles composed of six core knowledge patterns, geometric and abstract shapes, and five different task configurations. To inspect whether the model accuracy is grounded in perception and reasoning, MARVEL complements the general AVR question with _perception questions_ in a hierarchical evaluation framework. We conduct comprehensive experiments on MARVEL with nine representative MLLMs in zero-shot and few-shot settings. Our experiments reveal that all models show nearrandom performance on the AVR question, with significant performance gaps (40%) compared to humans across all patterns and task configurations. Further analysis of perception questions reveals that MLLMs struggle to comprehend the visual features (near-random performance) and even count the panels in the puzzle ( _<_ 45%), hindering their ability for abstract reasoning. We release our entire code and dataset.[1] 

## **1 Introduction** 

Recent advances in novel training pipelines, computational resources, and data sources have enabled Multi-modal Large Language Models (MLLMs) (OpenAI, 2023b; Google, 2023) to show strong visual reasoning ability in tasks that require both visual and textual cues (Wang et al., 2023), such as visual question answering (Goyal et al., 2017a; Antol et al., 2015) and visual commonsense reasoning (Zellers et al., 2019; Xie et al., 2019). These tasks typically focus on testing the models’ real-world knowledge (Małkinski, 2023).´ On the other hand, abstract visual reasoning (AVR) (Zhang et al., 2019; Hill et al., 2019) has little dependency on the world knowledge. As the puzzle shown in Figure 1 (top-right), AVR problems require identifying the hidden pattern (addition and subtraction) that governs the input shapes and their attribute (number of stars/circles) in a task configuration (2 _×_ 3 matrix). The abstract reasoning ability is related to many practical applications, including visual representations (Patacchiola & Storkey, 2020) and anomaly detection (Schubert et al., 2014), 

> _∗_ Authors contributed equally 

> 1 `https://github.com/1171-jpg/MARVEL_AVR` 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning_images/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Pattern Context<br>Core Knowledge Pattern<br>Mathematical<br>Number<br>Representation<br>Choice<br>Input Shape<br>Geometric Shape Abstract Shape<br>... ... ... AVR Question<br>The puzzle consists of a question part in a 2 by 3 matrix with<br>Task Configuration the right piece missing. Which choice is the answer to<br>fill the missing part? Answer: Choice 4.<br>Sequence Two-row Matrix<br>Perception Questions<br>Coarse-grained Fine-grained<br>How many panels, including  Are there two or three circles<br>Reassembling Group blank panel, in the question  in the left half of choice 4?<br>part? Answer: Six panels. Answer: Three circles.<br>....<br>**----- End of picture text -----**<br>


Figure 1: An abstract visual reasoning puzzle in **MARVEL** . The puzzle contains **mathematical** pattern governing the element number in **geometric shapes** with **two-row** task configuration. The AVR question focuses on the final answer for the puzzle, while the perception questions focus on the fine-grained detail about one choice or coarse-grained detail over the whole puzzle. 

encouraging more fundamental research in evaluating MLLMs on AVR benchmarks (Ahrabian et al., 2024; Yang et al., 2023b). 

However, the scope of current AVR benchmarks is limited, covering few reasoning patterns over a limited set of input shapes arranged in a predetermined configuration of puzzle panels, leading to biased evaluations (van der Maas et al., 2021). RAVEN (Zhang et al., 2019) tests abstract reasoning mostly in mathematical patterns over predefined geometric input shapes. Bongard-LOGO (Nie et al., 2020) and SVRT (Fleuret et al., 2011) implement geometric-related (symmetric) patterns over manually designed abstract input shapes. Although recent surveys highlight the limited evaluation scope of these benchmarks (Małkinski, 2023; van der Maas et al., 2021), a comprehensive benchmark for accessing´ MLLMs in multidimensional settings has not been proposed and is still needed. 

To fill this gap, we introduce **MARVEL** , a multi-dimensional abstract visual reasoning benchmark designed to evaluate MLLMs across different patterns, input shapes, and task configurations. MARVEL’s underlying reasoning patterns are rooted in key core knowledge of human cognition, observed in newborn infants relying only on abstraction to reason about their environment (Spelke & Kinzler, 2007). To facilitate a more comprehensive evaluation of foundation MLLMs, MARVEL combines six patterns expanded from three types of core knowledge, both geometric and abstract shapes, and five task configurations. We crawl relevant puzzles from publicly available websites, manually filter low-quality and irrelevant puzzles based on the expanded patterns and input shapes, and reformat them into different task configurations. We annotate the AVR question by providing a short description of the puzzle and asking for its answer. In total, we collect 770 diverse and high-quality puzzles assessing abstract visual reasoning abilities (Figure 1). 

To provide a hierarchical evaluation, we also enrich each puzzle with perception questions focusing on perceiving puzzles’ visual details (e.g., number of grids, edges of a triangle) to measure models’ reasoning consistency (Jiang et al., 2023; Selvaraju et al., 2020). We conduct comprehensive experiments on MARVEL involving different model structures, model sizes, and prompting strategies. Our experiments reveal that all MLLMs show near-random performance in all patterns, even with few-shot demonstrations and prompt engineering, leaving a huge gap (40%) in the abstract reasoning ability with humans. An in-depth 

2 

|Dimension|RAVEN|G-set|VAP|Bongard-<br>LOGO|SVRT|DOPT|ARC*|**MARVEL**|
|---|---|---|---|---|---|---|---|---|
|**Input Shape**<br>Geometric<br>Abstract|✔|✔|✔|✔|✔|✔|✔|✔<br>✔|
|**Pattern**<br>Temporal Movement<br>Spatial Relationship<br>Quantities<br>Mathematical<br>2D-Geometry<br>3D-Geometry|✔<br>✔<br>✔|✔<br>✔<br>✔|✔<br>✔|✔<br>✔|✔<br>✔<br>✔|✔<br>✔|✔<br>✔<br>✔<br>✔|✔<br>✔<br>✔<br>✔<br>✔<br>✔|
|**Confguration**<br>Sequence<br>Two-row<br>Matrix<br>Group<br>Reassembling|✔|✔|✔|✔|✔|✔||✔<br>✔<br>✔<br>✔<br>✔|
|Perception Question||||||||✔|



Table 1: Comparing MARVEL to related benchmarks: RAVEN (Zhang et al., 2019), G- set (Mandziuk &´ Zychowski, 2019), VAP (Hill et al., 2019), Bongard-LOGO (Nie et al., 2020),[˙] SVRT (Fleuret et al., 2011), ARC (Chollet, 2019), DOPT (Webb et al., 2020). *ARC puzzles are provided in a generative format. 

analysis based on perception questions points out that MLLMs’ poor reasoning ability is hindered by their struggle with fine-grained visual feature comprehension. Their nearrandom perception ability fails to provide foundations for subsequent abstract reasoning. In summary, our contributions can be listed as follows: 1) **A novel multidimensional AVR benchmark** , **MARVEL** , which consists of six patterns across five distinct task configurations. 2) **A hierarchical evaluation framework** incorporating perception questions with AVR questions to enable fine-grained diagnosis of model capability. 3) **Extensive experiments** on a wide range of state-of-the-art MLLMs with various prompting strategies, providing insights into their strengths, weaknesses, and future improvement directions on AVR tasks and MLLM developments. 

## **2 Related Work** 

**MLLM Evaluations.** Benefiting from the rich representation from visual encoders (Radford et al., 2021) and strong reasoning ability of LLMs (Touvron et al., 2023; Chiang et al., 2023), MLLMs (Li et al., 2023; Dai et al., 2024; OpenAI, 2023a; Liu et al., 2024) have been applied to solve not only traditional vision-language tasks, such as image captioning (Agrawal et al., 2019; Young et al., 2014), visual question answering (Goyal et al., 2017b; Marino et al., 2019; Hudson & Manning, 2019a; Singh et al., 2019) and refer expression comprehension (Kazemzadeh et al., 2014; Gupta et al., 2022), but also on more complicated scenarios, such visually-grounded conversation (Liu et al., 2024; Alayrac et al., 2022), multimodal web/UI agents (He et al., 2024; Zheng et al., 2024; Yang et al., 2023a) and embodied tasks Driess et al. (2023). Besides end-to-end evaluation, several recent works also try to reveal MLLMs’ visual shortcomings from different aspects, including visual details (Zhang et al., 2023; Wu & Xie, 2023), perceptual bias (Zhang et al., 2024), and small visual pattern recognition (Tong et al., 2024). Although some of the existing benchmarks have accessed MLLM’s mathematical visual reasoning abilities requiring an understanding of abstract and geometry shapes (Lu et al., 2023; 2022), their evaluation still heavily relies on textual descriptions. In contrast, AVR benchmarks assess MLLMs’ ability under a diverse set of patterns with only visual understanding settings. 

**AVR Benchmarks.** AVR problems have great potential impact on various domains (Małkinski, 2023; Patacchiola & Storkey, 2020; Schubert et al., 2014), sparking interests´ in evaluating MLLMs on AVR benchmarks (Ahrabian et al., 2024; Moskvichev et al., 2023; Mitchell et al., 2023). Existing AVR benchmarks present the evaluation in a wide range of formats, such as selective completion (Zhang et al., 2019; Hu et al., 2021; Benny et al., 2021; Webb et al., 2020), group discrimination (Fleuret et al., 2011; Nie et al., 2020) and generative completion (Chollet, 2019). However, less attention is paid to the scope and 

3 

pattern of the AVR benchmark; most focus only on a few simple abstract patterns and testing models end-to-end without considering the intermediate perception and reasoning procedures (Moskvichev et al., 2023; Mitchell, 2021). In contrast, MARVEL includes geometric and abstract shapes, six core patterns essential for visual abstract reasoning, and five different task configurations. Inspired by prior analysis of visual details and perceptual bias, MARVEL introduces perception questions to ensure the MLLMs correctly perceive the presented visual patterns. Our work and other related AVR benchmarks are compared in Table 1. 

## **3 MARVEL Benchmark Construction** 

As a multidimensional benchmark for AVR, MARVEL covers different task configurations (Section 3.1), various input shapes (Section 3.2), as well as different reasoning patterns involved in the puzzles (Section 3.3). Further, MARVEL decomposes the evaluation of models’ capabilities into 1) perception questions (Section 4) about the puzzle panels and 2) AVR questions (Section 3.4) to yield a more precise and faithful picture of evaluated models. The data collection process is presented in Section 3.4. 

## 3.1 Task Definition and Configurations 

Each puzzle in MARVEL consists of a context on the top and possible choices ( _ci_ ; _i ∈ {_ 1, 2, 3, 4 _}_ ) to choose from at the bottom, formatted in a multiple-choice question answering setting (see Figure 1 for an example). The context part consists of _n_ puzzle panels ( _p_ 1, _p_ 2, . . . , _pn_ , _pb_ with _pb_ being a blank panel), with their specific number and arrangement driven by a task configuration and reasoning pattern, _P_ , that governs the relationship between puzzle panels. The choice will be considered the correct answer and fill in _pb_ that can satisfy the following equation: _P_ ( _p_ 1, _p_ 2, . . . , _pn_ ) = _P_ ( _p_ 1, _p_ 2, . . . , _pn_ , _cc_ ). 

Puzzle panels in MARVEL are organized in the following five task configurations: 

1. **Sequence Format** arranges panels in a 1 _× n_ line ( _n ∈_ [4, 7]). 

2. **Two-row Format** presents panels in a 2 _×_ 3 matrix _[∗]_ , _p_[1] 1[,] _[ p]_ 2[1][,] _[ p]_[1] 3[and] _[p]_[2] 1[,] _[ p]_ 2[2][,] _[ p]_[2] 3[.][The solution] requires identifying the same pattern at the first row and the second row, which is _P_ ( _p_[1] 1[,] _[ p]_ 2[1][,] _[ p]_[1] 3[) =] _[P]_[(] _[p]_[2] 1[,] _[ p]_ 2[2][,] _[ c][c]_[)][.] 

3. **Matrix Format** organizes panels in a 3 by 3 matrix, the pattern can be reflected in either row- or column-wise way: _P_ ( _p_[1] 1[,] _[ p]_ 2[1][,] _[ p]_[1] 3[) =] _[P]_[(] _[p]_[2] 1[,] _[ p]_ 2[2][,] _[ p]_[2] 3[) =] _[P]_[(] _[p]_[3] 1[,] _[ p]_ 2[3][,] _[ c][c]_[)][ or] _[ P]_[(] _[p]_[1] 1[,] _[ p]_[2] 1[,] _[ p]_[3] 1[) =] _P_ ( _p_[1] 2[,] _[ p]_[2] 2[,] _[ p]_[2] 3[) =] _[P]_[(] _[p]_[1] 3[,] _[ p]_[2] 3[,] _[ c][c]_[)][.] 

4. **Group Format** has three panels in the question part ( _p_ 1, _p_ 2, _pb_ ) with one choice reflecting the context’s pattern and other choices differing: _P_ ( _p_ 1, _p_ 2, _cc_ ) _̸_ = _P[′]_ ( _choices − cc_ ). 

5. **Reassembling Format** is designed for _3D-Geometry_ pattern with one unfolded diagram in the context and four 3D assembly results as choices. 

We visualize these configurations in Figure 1 (see detailed examples in Appendix A). 

## 3.2 Input Shapes 

As shown in Figure 1, each panel of a puzzle contains various shapes that can be generally differentiated into two types (Małki´nski, 2023): 

1. **Geometric Shapes** are easily comprehended and described. For instance, a square is a shape that has four sides of equal length and four equal angles. Most existing AVR benchmarks (Zhang et al., 2019; Hill et al., 2019) focus on elementary shapes such as oval, rectangle, triangle and trapezoid. MARVEL includes geometric shapes consisting of more than two different elementary geometric shapes to mitigate the issue and improve the complexity. 

2. **Abstract Shapes** come from a wide set of possibilities and vary widely from one problem to another. It provides a fair step as most MLLMs encounter the shapes for the first 

   - _p[b] a_ : the panel on the _a_ th row and _b_ th column of matrix. 

4 

time. MARVEL also includes abstract shapes as they are gaining more preference for AVR-related research (Fleuret et al., 2011; Nie et al., 2020). 

## 3.3 Core Knowledge and Patterns 

Core knowledge theory (Spelke & Kinzler, 2007) from cognition developmental psychology is largely shared among humans and particularly for human infants. Human infants with no real-world knowledge and limited experience represent their environment using abstraction patterns. These abstraction patterns can be categorized into three types of core knowledge, which is the foundation for inference and reasoning (Lipton & Spelke, 2004). We expand each core knowledge[2] into two patterns for a fine-grained assessment of abstract reasoning in MARVEL, based on insights drawn from contemporary cognitive literature: 

**Object Core Knowledge** represents objects’ spatio-temporal motions and their contact, which is expanded to _Temporal Movement Pattern_ focusing on the related position change or movement (Spelke, 1990) and _Spatial Relationship Pattern_ examining objects’ relative positional relationship (Aguiar & Baillargeon, 1999). 

**Number Core Knowledge** helps infants process abstract representations of small numbers. We include _Quantities Pattern_ testing the accuracy of number comprehension (Xu & Spelke, 2000) and _Mathmatical Pattern_ for elementary mathematical operations (Barth et al., 2005). 

**Geometry Core Knowledge** captures the environment’s geometry, which helps humans orient themselves in their surroundings. We divide it into _2D-Geometry Pattern_ (Hermer & Spelke, 1996) and _3D-Geometry Pattern_ (Cheng & Newcombe, 2005). 

## 3.4 Data Collection 

We collect puzzles from several public resources websites[3] and filter out unfit or low-quality data by three human annotators based on the puzzle’s input shapes (some puzzles contain textual information) and patterns. Unaligned puzzles are first segmented into panels and then reassembled into the correct task configuration. To ensure each pattern in each task configuration has at least 45 puzzles[4] , we also manually created 220 puzzles by following the pattern in existing data and replacing the input shape drawn from scratch. Each puzzle contains an AVR question (Figure 1) generated from templates based on their task configuration. AVR questions provide a brief description and ask only for the puzzle’s final answer, which is widely adopted in previous AVR benchmark (Małkinski, 2023).´ In the end, MARVEL includes 770 high-quality puzzles over six high-level patterns across five distinct task configurations. In Table 1, we compare MARVEL with existing AVR benchmarks to show its comprehensive scope. 

## **4 Hierarchical Evaluation Framework** 

Previous works evaluate MLLMs on AVR benchmarks with the final answer only (Moskvichev et al., 2023; Mitchell et al., 2023), potentially overlooking shortcut learning and inductive biases (Małkinski, 2023).´ On the other hand, precisely comprehending visual details is the foundation for subsequent reasoning in AVR problems (Gao et al., 2023). We enrich MARVEL puzzles with perception questions (Selvaraju et al., 2020) designed to test models’ perception ability on visual details (Figure 1). We design a hierarchical evaluation framework by combining two types of perception questions with AVR questions (Figure 1) to examine if model accuracy is based on perception and reasoning. For each puzzle, our framework provides three coarse-grained questions and one pattern-related fine-grained question: 

> 2We don’t feature the fourth Core Knowledge, agent representation, due to its concentrates on goal-directed and interactive action, which is not adaptable in MARVEL setting. 3 `https://www.gwy.com/` ; `https://www.chinagwy.org/` 

> 4Some patterns can not be presented in specific configurations. For example, the _Mathmatical Pattern_ can not be adapted to group format as it requires comparison between adjacent panels. 

5 

**Coarse-grained Perception Question** in an open-ended fashion aims to test if models can understand task configuration correctly, which directly asks about the number of panels in puzzles. We use templates to generate three questions focusing on the number of panels in the context part, choice parts, and the whole puzzle. We remove the choice index when testing models with this question to avoid shortcut learning. 

**Fine-grained Perception Question** in binary-choice format examines models’ understanding of input shapes, which focus on the visual details (Selvaraju et al., 2020) such as shape attributes (number of edges) and spatial relationship (left, right) based on the pattern contained in the puzzle. For example, in Figure 1, the fine-grained perception question tests whether models can understand the number of circles because the puzzle contains _Mathematical Pattern_ . For each puzzle, we randomly pick one choice panel in the puzzle and manually create questions with two choices. The correct answer is randomly placed to avoid inductive bias. We have five types of questions based on the pattern and how it adapts to the input shape, which are listed with examples here: 

1. **Location** : Is the dot outside or inside of the star in choice 4? 

2. **Color** : Is the triangle black or white in choice 1? 

3. **Shape** : Is there a circle or a triangle inside choice 3? 

4. **Quantity** : Are there five or four circles in choice 2? 

5. **Comparison** : Are the left and right halves of the rectangle in choice 3 the same? 

## **5 Experimental Setup** 

## 5.1 Model Selection 

**Closed-source MLLMs.** We include API-based MLLMs including 1) GPT-4V (OpenAI, 2023a), 2) Gemini (Google, 2023) and 3) Claude3 (Anthropic, 2024). With the massive computation and training data, these models show promising performance on a wide range of visual-focused tasks (Goyal et al., 2017b; Xie et al., 2019). We evaluate closed-source MLLMs in both zero-shot and few-shot (Brown et al., 2020) settings. 

**Open-source MLLMs.** We include MLLMs smaller than 13B due to our limited computing resources: 1) InstructBLIP (Dai et al., 2024), 2) BLIP-2 (Li et al., 2023), 3) Fuyu (Bavishi et al., 2023), 4) Qwen-VL (Bai et al., 2023) and 5) LLaVA (Liu et al., 2023). We only evaluate these MLLMs in a zero-shot setting due to their single-image input settings (Zhao et al., 2023). 

**Human Evaluation.** To access the upper bound performance on MARVEL, we simulate a realistic human assessment by inviting 30 annotators aged from 10 to 50 years to solve a subset of MARVEL and ensure each subset contains every pattern in all task configurations. We compute the average performance of these 30 annotators as the human baseline. Each puzzle is solved by at least two annotators. 

## 5.2 Evaluation Metrics 

Following a similar setting as previous research evaluating MLLMs on the AVR benchmark (Ahrabian et al., 2024), we use regex matching to extract the choices picked (e.g., ”choice 4” in the response ”The correct answer is choice 4.”), with failure cases re-extracted by GPT-4 (Aher et al., 2023). We use accuracy as the metric, which is commonly used for evaluating multiple-choice questions and has been utilized by many AVR systems (Zhang et al., 2019; Hill et al., 2019). Based on the hierarchical evaluation framework, we evaluate MLLMs with two types of accuracy-based metrics: 

1. **Instance-based Accuracy** considers questions separately. We report accuracy results for _AVR question_ and _fine-grained perception question_ . 

2. **Group-based Accuracy** considers questions as group to assess the consistency in model reasoning (Jiang et al., 2023; Yuan et al., 2021). The model receives a score of 1 only if it correctly answers all questions within the same group. We report the group-based 

6 

accuracy result of combining all three coarse-grained perception questions and the further result after introducing fine-grained and AVR questions into the group. 

|**Category**<br>**Model**|**AVR**<br>**Question**|**AVR**<br>**Question**|**Fine-grained**<br>**Perception**|**Fine-grained**<br>**Perception**|**PercC**|**PercC&F**|**PercC&F** &**AVR**|
|---|---|---|---|---|---|---|---|
|**Random**||25.00||50.00|-|-|-|
|**Open-**<br>**source**<br>**MLLMs**<br>`Qwen-VL (7B)`<br>`Fuyu (8B)`<br>`BLIP-2 (FlanT5`XXL`-11B)`<br>`InstructBLIP (Vicuna-13B)`<br>`LLaVA-1.5 (Vicuna-13B)`||19.61<br>24.29†<br>24.81†<br>24.68†<br>26.36||37.27<br>34.94<br>**53.38**<br>49.48<br>51.43|0.52<br>0.00<br>1.04<br>0.00<br>1.14|0.39<br>0.00<br>0.52<br>0.00<br>0.52|0.00<br>0.00<br>0.26<br>0.00<br>0.13|
|**Closed-**<br>**source**<br>**MLLMs**<br>`GPT-4V`<br>`Gemini-pro-vision`*<br>`Claude3 (Sonnet)`<br>`Claude3 (Opus)`||22.34<br>25.06†<br>26.49†<br>**28.83**||51.56<br>18.31<br>9.22<br>2.73<br>44.42<br>15.19<br>6.75<br>1.69<br>50.91<br>38.70<br>19.87<br>5.06<br>47.27<br>**44.94**<br>**20.13**<br>**5.97**||||
|**Human**<br>68.86_±_9.74||||-<br>-<br>-<br>-||||



Table 2: Main zero-shot accuracy over MARVEL across all MLLMs in two accuracy metrics: _Prec[C]_ = group-based accuracy over all coarse-grained perception questions (model must answer all three questions correctly), _Perc[C]_[&] _[F]_ = group-based accuracy combining all perception questions (coarse/fine-grained), AVR = AVR Question. The best performance among all models is in **bold** , and the best result in two MLLMs categories is underlined. *Gemini refuses to answer the puzzle due to safety problems in 7% cases so the performance is computed based on the left set. † notes the result is attributed to inductive bias. 

## **6 Results** 

We focus on four research questions: 1) What’s the abstract reasoning ability on visual puzzles of current SOTA MLLMs? 2) Can MLLMs do better with different few-shot prompting strategies? 3) How do MLLMs perform on different patterns and task configurations? 4) To what extent do MLLMs visually understand the puzzle, and do they show consistent reasoning ability? 

**Overall Performance.** The AVR question results are shown in Table 2. Human performance reaches 68.86%, with a standard deviation of 9.74, confirming the validity and challenging nature of MARVEL. For both open and closed source categories, **all models show near-random performance with a huge gap (40%) compared to human performance** , in which closed-source MLLMs (avg: 25.7%) perform slightly better than open-sourced ones (avg: 24.0%). We observed an extremely imbalanced distribution in the outputs of some MLLMs. For example, BLIP2 consistently selecting choice 1 for all puzzles (marked † in Table 2). We tried different approaches with our best effort to avoid potential bad prompts or engineering settings, including adding question marks in the 29 black panel, replacing the choice index with letter (1 _→_ 28 A), and changing the description in the AVR question. 27 None of them can mitigate and may even exacerbate the 26 issue, highlighting the potential inductive biases (Wang 25 & Wu, 2024) in models. Among open-source MLLMs, 24 LLAVA performs the best, yet the gap is very small, and 23 it is unclear whether the gain comes from its larger model size. In closed-source models, even the strongest MLLM, Zero-shot One-shot Two-shot Claude3 (Opus), which demonstrated promising results Gemini Claude3 (sonnet) on various vision tasks (Anthropic, 2024), failed to present GPT-4V Claude3 (Opus) a significant performance difference from the random baseline. Claude3 (Sonnet) and Gemini also have imbalFigure 2: MLLMs performance anced output distributions, with both selecting choice 4 in different few-shot COT. in most cases. 

**Impact of Few-Shot CoT.** Given the poor zero-shot performance and models’ capability of in-context learning (Brown et al., 2020), we explore few-shot prompting with Chainof-Thought (CoT) (Wei et al., 2022) to guide MLLMs with abstract reasoning patterns. 

7 

We experiment with all closed-source MLLMs in one-shot and two-shot settings using manually created CoT context, similar to Yang et al. (2023b). For each puzzle, we randomly select puzzles with the same pattern in the sequence task configuration and annotate the CoT reasoning with answers as demonstrations. We chose the sequence task because it is more straightforward (only along the sequence) compared to other configurations. Each demonstration is formatted as image-text pairs. Our result is shown in Figure 2, and we present the full results in Appendix E. The few-shot demonstrations show a marginal positive impact on GPT-4V and a decreasing trend on Claude3 (Opus). 

Further analysis reveals that the main improvement in GPT-4V’s results lies in the _3D-Geometry pattern_ . As this pattern focuses on reassembling, the demonstration can guide the model to pay attention to the relative position of each face of the object. However, since most patterns are uniquely implemented on different input shapes and their attributes, the model struggles to learn generalizable patterns from the few-shot demonstrations. Figure 3 provides an example of zero-and few-shot results from Claude3 (Opus). With the demonstration, the model learns to focus on the correct pattern (blue) at the beginning of the reasoning. However, it fails to adapt precisely to the input shapes in the puzzle (red), leading to errors in subsequent reasoning. We also test different prompt engineering approaches, including selecting demonstration samples from different 1) patterns, 2) task configurations, and 3) prompting MLLMs by dividing puzzles panel by panel. None of these approaches yields a positive impact; instead, they lead to a significant drop in performance (AppendixE). Given the complexity and challenging nature of the dataset, the effectiveness of few-shot prompting on MARVEL remains minimal.[5] 

**Performance on Different Patterns and Configurations.** We further break down the results[6] into different patterns and task con- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning_images/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning.pdf-0008-03.png)


**----- Start of picture text -----**<br>
Context<br>Perception Question:<br>Are there two or three parallel<br>horizontal lines in choice 2?<br>Choices Claude3-Opus: Choice 2 contains three parallel<br>horizontal lines.<br>Claude3-Opus (Zero-shot):<br>To determine which choice best fits the missing piece in the bottom<br>right, we need to identify the pattern in the given 3x2 grid. Looking at<br>the grid, we see that each row follows a specific pattern:<br>Row 1: The number of vertical lines increases by 1 from left to right.<br>Row 2: The number of horizontal lines increases by 1 from left to<br>right.<br>Based on this pattern, the missing piece in the bottom right should<br>have 3 horizontal lines, as it follows the pattern in the second row.<br>Looking at the choices provided, choice 3 is the only option that has 3<br>horizontal lines. Therefore,  choice 3  is the most appropriate answer to<br>fill the missing part in the given puzzle.<br>Claude3-Opus (Two-shot):<br>Let's analyze the pattern in the given 3x2 matrix, focusing on the<br>number of vertical and horizontal lines in each cell<br>Row 1: Cell 1 has one vertical line, Cell 2 has two vertical lines<br>Row 2: Cell 1 has one vertical and one horizontal line, forming a plus<br>sign. Cell 2 has two vertical and two horizontal lines, forming a hash<br>symbol.<br>Row 3: Cell 1 has three vertical lines. The missing cell should follow<br>the pattern established in the previous rows.<br>Based on the pattern, the missing cell should have three vertical and<br>three horizontal lines, forming a grid of nine squares. This corresponds<br>to choice 4 among the given options. Therefore,  choice 4  is the most<br>appropriate answer to complete the matrix.<br>**----- End of picture text -----**<br>


Figure 3: Perception question, zero- and twoshot example of Claude3 (Opus). 

figurations in Figure 4 (full results in Appendix B). In general, MLLMs show near-random performance on all patterns and task configurations. Among the six patterns, _3D-Geomertry_ pattern is the most challenging for humans and MLLMs. The difficulty may be rooted in the requirement for the 3D imaginative ability (Margules & Gallistel, 1988) and relatively less common in the models’ pre-training datasets, which leads to a significant gap (12.5%) between open and closed-sources MLLMs. On the other hand, _2D-Geomertry_ , involving understanding geometric attribute of input shape, seems relatively easier for MLLMs, evidenced by their proficiency on similar tasks such as interpreting graphs plots (Yang et al., 2023b). In two model categories, LLaVA ranks first on three patterns, showing competitive performance compared to closed-source MLLMs. Claude3 (Opus) exhibits a balanced and strong reasoning ability, ranking in the top 2 across all patterns. 

Among the five task configurations, the two-row and matrix formats provide more panels (i.e., information) to verify the pattern, facilitating puzzle-solving. On the contrary, the group and reassembling formats containing the least panel number tend to be challenging. 

> 5We find the inductive bias is mitigated in few-shot settings. 

> 6We only select MLLMs that do not show significantly imbalanced output distributions. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning_images/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning.pdf-0009-00.png)


**----- Start of picture text -----**<br>
Pattern Task Configuration<br>Temporal<br>Movement<br>3D 80 Sequence80<br>Geometry 60 Reassembling 60<br>40 40<br>Spatial<br>20 Relationship 20<br>2D Two<br>Geometry row<br>Group<br>Quantities<br>Mathematical<br>Matrix<br>Qwen-VL LLaVA-1.5 GPT-4V Claude3 (Opus) Human<br>**----- End of picture text -----**<br>


Figure 4: MLLMs and human performance across patterns and task configurations. 

Three out of four MLLMs rank 1st in different task configurations, which verifies our assumption of potential bias in single-configuration evaluation. Models may be familiar with specific input types according to their pre-training dataset, highlighting the importance and necessity of multidimensional settings in MARVEL. 

**Perception Ability and Reasoning Consistency.** Visual cognition forms the foundation for advanced reasoning (Richards et al., 1984). By incorporating perception questions, our hierarchical evaluation framework effectively investigates to what extent the models understand the visual information from the puzzle. In Table 2, closed-source MLLMs demonstrate more robust performance on coarse-grained perception group accuracy compared to opensourced MLLMs, with a performance gap ranging from 14.05% to 44.80%. However, even the best model fails to reach 50% accuracy, indicating that current MLLMs struggle to simultaneously understand the number of grids, choices, and the puzzle as a whole, despite their promising performance on real-world datasets (Hudson & Manning, 2019b). The simplicity of the coarse-grained perception questions (all puzzles contain less than 13 panels) highlights the poor perception ability of current MLLMs in the abstract visual reasoning domain. Fine-grained perception questions further confirm this argument, with all models showing near-random performance. Further analysis of fine-grained perception performance based on five categories (Table in Appendix C) reveals that models perform relatively better at color perception but have difficulty recognizing location (e.g., ‘a’ is on the left of ‘b’). We hypothesize that the difficulty in understanding location stems from the lack of labeled data on location and relations during training, especially in abstract visual understanding. In contrast, the models’ color perception is well-trained during their multi-modal alignment, and the simplicity of RGB understanding allows for easier transfer to the abstract domain. 

The further group-based accuracy ( _Prec[C]_[&] _[F]_ and _Prec[C]_[&] _[F]_ & _AVR_ ) shows that **no model can solve the AVR puzzles with consistent reasoning** , with the best model reaching only 5.97% group accuracy. Based on the result of our evaluation framework, we hypothesize the inconsistency stems from their poor visual perception ability (Gao et al., 2023). To verify our assumption, we run an additional experiment on a subset of MARVEL by adding accurate text descriptions of the puzzle in the input (full result in Appendix D). The result shows a significant boost in performance (11.57% to 44.21%), with GPT-4V even displays on-par performance (65.26%) with humans. As shown in Figure 3, the model’s reasoning is based on the perception of the puzzle (e.g., number of lines), which needs to be completely precise to support correct reasoning. The perception questions in our framework reveal that the model cannot clearly understand the number of lines, explaining why it fails to answer the puzzle even with correct hints (few-shot). A single error in visual feature perception can impact reasoning since the correct pattern must apply to all puzzle shapes. The densely packed information distribution—where the majority of the puzzle remains blank—ensures that each piece of visual perception is an essential foundation for subsequent reasoning. However, the importance of visual detail perception has received little attention in previous evaluations (Moskvichev et al., 2023; Mitchell et al., 2023), highlighting the significance of our new evaluation benchmark. 

9 

## **7 Conclusion** 

In this work, we developed MARVEL, a multidimensional abstract reasoning benchmark comprising 770 puzzles with both geometric and abstract input shapes across six patterns and five task configurations. We also designed a hierarchical evaluation framework that enriches MARVEL with perception questions to enable granular analysis of models’ visual details understanding and reasoning consistency. Our comprehensive experiments with nine SoTA MLLMs revealed a huge gap in abstract visual reasoning ability between (40%) humans and MLLMs, where all MLLM often perform close to random. Further analysis based on our evaluation framework showed MLLMs’ poor perception ability in understanding visual details, which hinders their subsequent reasoning and leads to poor AVR performance. We hope future works can build on the foundation of MARVEL for enhancing MLLM abstract visual perception and reasoning abilities. 

## **8 Acknowledgements** 

We appreciate Fred Morstatter for very helpful comments. We thank Tian Jin and Jiachi Liang for their assistance in data collection. This research was sponsored by the Defense Advanced Research Projects Agency via Contract HR00112390061. 

## **References** 

Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 8948–8957, 2019. 

- Andrea Aguiar and Ren´ ee Baillargeon.´ 2.5-month-old infants’ reasoning about when objects should and should not be occluded. _Cognitive psychology_ , 39(2):116–157, 1999. 

- Gati V Aher, Rosa I Arriaga, and Adam Tauman Kalai. Using large language models to simulate multiple humans and replicate human subject studies. In _International Conference on Machine Learning_ , pp. 337–371. PMLR, 2023. 

- Kian Ahrabian, Zhivar Sourati, Kexuan Sun, Jiarui Zhang, Yifan Jiang, Fred Morstatter, and Jay Pujara. The curious case of nonverbal abstract reasoning with multi-modal large language models. _arXiv preprint arXiv:2401.12117_ , 2024. 

- Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in neural information processing systems_ , 35:23716–23736, 2022. 

Anthropic. Claude 3, 2024. URL `https://www.anthropic.com/news/claude-3-family` . 

- Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In _Proceedings of the IEEE international conference on computer vision_ , pp. 2425–2433, 2015. 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint arXiv:2308.12966_ , 2023. 

- Hilary Barth, Kristen La Mont, Jennifer Lipton, and Elizabeth S Spelke. Abstract number and arithmetic in preschool children. _Proceedings of the national academy of sciences_ , 102(39): 14116–14121, 2005. 

- Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sagnak Ta˘ s¸ırlar. Introducing our multimodal models, 2023. URL `https: //www.adept.ai/blog/fuyu-8b` . 

10 

Yaniv Benny, Niv Pekar, and Lior Wolf. Scale-localized abstract reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 12557–12565, 2021. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- Ken Cheng and Nora S Newcombe. Is there a geometric module for spatial orientation? squaring theory and evidence. _Psychonomic bulletin & review_ , 12:1–23, 2005. 

- Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL `https://lmsys.org/blog/2023-03-30-vicuna/` . 

Franc¸ois Chollet. On the measure of intelligence. _arXiv preprint arXiv:1911.01547_ , 2019. 

- Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. _arXiv preprint arXiv:2303.03378_ , 2023. 

- Franc¸ois Fleuret, Ting Li, Charles Dubout, Emma K Wampler, Steven Yantis, and Donald Geman. Comparing machines and humans on a visual categorization test. _Proceedings of the National Academy of Sciences_ , 108(43):17621–17625, 2011. 

- Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. _arXiv preprint arXiv:2312.11370_ , 2023. 

Gemini Team Google. Gemini: A family of highly capable multimodal models, 2023. 

- Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 6904–6913, 2017a. 

- Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 6904–6913, 2017b. 

- Tanmay Gupta, Ryan Marten, Aniruddha Kembhavi, and Derek Hoiem. Grit: General robust image task benchmark. _arXiv preprint arXiv:2204.13653_ , 2022. 

- Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models, 2024. 

- Linda Hermer and Elizabeth Spelke. Modularity and development: The case of spatial reorientation. _Cognition_ , 61(3):195–232, 1996. 

- Felix Hill, Adam Santoro, David GT Barrett, Ari S Morcos, and Timothy Lillicrap. Learning to make analogies by contrasting abstract relational structure. _arXiv preprint arXiv:1902.00120_ , 2019. 

- Sheng Hu, Yuqing Ma, Xianglong Liu, Yanlu Wei, and Shihao Bai. Stratified rule-aware network for abstract visual reasoning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pp. 1567–1574, 2021. 

11 

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 6700–6709, 2019a. 

- Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 6700–6709, 2019b. 

- Yifan Jiang, Filip Ilievski, Kaixin Ma, and Zhivar Sourati. Brainteaser: Lateral thinking puzzles for large language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 14317–14332, 2023. 

- Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In _Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)_ , pp. 787–798, 2014. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In _International conference on machine learning_ , pp. 19730–19742. PMLR, 2023. 

- Jennifer S Lipton and Elizabeth S Spelke. Discrimination of large and small numerosities by human infants. _Infancy_ , 5(3):271–290, 2004. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ , 2023. 

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36, 2024. 

- Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. _Advances in Neural Information Processing Systems_ , 35:2507–2521, 2022. 

- Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In _The 3rd Workshop on Mathematical Reasoning and AI at NeurIPS’23_ , 2023. 

Mikołaj Małkinski.´ A review of emerging research directions in abstract visual reasoning. _Information Fusion_ , 91:713–736, 2023. 

- Jacek Mandziuk and Adam´ Zychowski.[˙] Deepiq: A human-inspired ai system for solving iq test problems. In _2019 International Joint Conference on Neural Networks (IJCNN)_ , pp. 1–8. IEEE, 2019. 

- Joan Margules and CR Gallistel. Heading in the rat: Determination by environmental shape. _Animal Learning & Behavior_ , 16(4):404–410, 1988. 

- Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In _Proceedings of the IEEE/cvf conference on computer vision and pattern recognition_ , pp. 3195–3204, 2019. 

- Melanie Mitchell. Abstraction and analogy-making in artificial intelligence. _Annals of the New York Academy of Sciences_ , 1505(1):79–101, 2021. 

- Melanie Mitchell, Alessandro B Palmarini, and Arsenii Kirillovich Moskvichev. Comparing humans, gpt-4, and gpt-4v on abstraction and reasoning tasks. In _AAAI 2024 Workshop on”Are Large Language Models Simply Causal Parrots?”_ , 2023. 

- Arseny Moskvichev, Victor Vikram Odouard, and Melanie Mitchell. The conceptarc benchmark: Evaluating understanding and generalization in the arc domain. _arXiv preprint arXiv:2305.07141_ , 2023. 

12 

Weili Nie, Zhiding Yu, Lei Mao, Ankit B Patel, Yuke Zhu, and Anima Anandkumar. Bongardlogo: A new benchmark for human-level concept learning and reasoning. _Advances in Neural Information Processing Systems_ , 33:16468–16480, 2020. 

OpenAI. Gpt-4 technical report, 2023a. 

R OpenAI. Gpt-4 technical report. arxiv 2303.08774. _View in Article_ , 2:13, 2023b. 

- Massimiliano Patacchiola and Amos J Storkey. Self-supervised relational reasoning for representation learning. _Advances in Neural Information Processing Systems_ , 33:4003–4014, 2020. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021. 

WA Richards et al. Parts of recognition. _Cognition_ , 18(1), 1984. 

- Erich Schubert, Arthur Zimek, and Hans-Peter Kriegel. Local outlier detection reconsidered: a generalized view on locality with applications to spatial, video, and network outlier detection. _Data mining and knowledge discovery_ , 28:190–237, 2014. 

- Ramprasaath R Selvaraju, Purva Tendulkar, Devi Parikh, Eric Horvitz, Marco Tulio Ribeiro, Besmira Nushi, and Ece Kamar. Squinting at vqa models: Introspecting vqa models with sub-questions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 10003–10011, 2020. 

- Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 8317–8326, 2019. 

- Elizabeth S Spelke. Principles of object perception. _Cognitive science_ , 14(1):29–56, 1990. 

- Elizabeth S Spelke and Katherine D Kinzler. Core knowledge. _Developmental science_ , 10(1): 89–96, 2007. 

- Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. _arXiv preprint arXiv:2401.06209_ , 2024. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- Han LJ van der Maas, Lukas Snoek, and Claire E Stevenson. How much intelligence is there in artificial intelligence? a 2020 update. _Intelligence_ , 87:101548, 2021. 

- Jiaqi Wang, Zhengliang Liu, Lin Zhao, Zihao Wu, Chong Ma, Sigang Yu, Haixing Dai, Qiushi Yang, Yiheng Liu, Songyao Zhang, et al. Review of large vision models and visual prompt engineering. _Meta-Radiology_ , pp. 100047, 2023. 

- Zhenhailong Wang, Joy Hsu, Xingyao Wang, Kuan-Hao Huang, Manling Li, Jiajun Wu, and Heng Ji. Text-based reasoning about vector graphics. _arXiv preprint arXiv:2404.06479_ , 2024. 

- Zihao Wang and Lei Wu. Theoretical analysis of the inductive biases in deep convolutional networks. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Taylor Webb, Zachary Dulberg, Steven Frankland, Alexander Petrov, Randall O’Reilly, and Jonathan Cohen. Learning representations that support extrapolation. In _International conference on machine learning_ , pp. 10136–10146. PMLR, 2020. 

13 

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. _arXiv preprint arXiv:2312.14135_ , 2023. 

- Ning Xie, Farley Lai, Derek Doran, and Asim Kadav. Visual entailment: A novel task for fine-grained image understanding. _arXiv preprint arXiv:1901.06706_ , 2019. 

- Fei Xu and Elizabeth S Spelke. Large number discrimination in 6-month-old infants. _Cognition_ , 74(1):B1–B11, 2000. 

- Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. _arXiv preprint arXiv:2312.13771_ , 2023a. 

- Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). _arXiv preprint arXiv:2309.17421_ , 9(1):1, 2023b. 

- Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. _Transactions of the Association for Computational Linguistics_ , 2:67–78, 2014. 

- Yuanyuan Yuan, Shuai Wang, Mingyue Jiang, and Tsong Yueh Chen. Perception matters: Detecting perception failures of vqa models using metamorphic testing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 16908–16917, 2021. 

- Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 6720–6731, 2019. 

- Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Raven: A dataset for relational and analogical visual reasoning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 5317–5327, 2019. 

- Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, and Filip Ilievski. Visual cropping improves zero-shot question answering of multimodal large language models. In _R0FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models_ , 2023. 

- Jiarui Zhang, Jinyi Hu, Mahyar Khayatkhoei, Filip Ilievski, and Maosong Sun. Exploring perceptual limitation of multimodal large language models. _arXiv preprint arXiv:2402.07384_ , 2024. 

Haozhe Zhao, Zefan Cai, Shuzheng Si, Xiaojian Ma, Kaikai An, Liang Chen, Zixuan Liu, Sheng Wang, Wenjuan Han, and Baobao Chang. Mmicl: Empowering vision-language model with multi-modal in-context learning. _arXiv preprint arXiv:2309.07915_ , 2023. 

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v(ision) is a generalist web agent, if grounded, 2024. 

## **A Data Examples** 

We present examples with 5 different task configurations and 6 patterns in Figures 5 to 11. 

## **B Performance on Different Pattern and Tasks** 

Table 3 and Table 4 shows MLLMs’ AVR question performance on different patterns and tasks. 

14 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning_images/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning.pdf-0015-00.png)


Figure 5: The example is formatted in _Sequence_ configuration with the _Quantities_ pattern. The answer to this puzzle is B. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning_images/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning.pdf-0015-02.png)


Figure 6: The example is formatted in _Sequence_ configuration with the _Temporal Movement_ pattern. The answer to this puzzle is C. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning_images/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning.pdf-0015-04.png)


Figure 7: The example is formatted in _Two-row_ configuration with the _Spatial Relationship_ pattern. The answer to this puzzle is B. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning_images/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning.pdf-0016-00.png)


Figure 8: The example is formatted in _Matrix_ configuration with the _Spatial Relationship_ pattern. The answer to this puzzle is B. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning_images/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning.pdf-0016-02.png)


Figure 9: The example is formatted in _Matrix_ configuration with the _Mathematical_ pattern in _Geometric_ shapes. The answer to this puzzle is A. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning_images/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning.pdf-0017-00.png)


Figure 10: The example is formatted in _Group_ configuration with the _2D-Geometry_ pattern in _Abstract_ input shapes. The answer to this puzzle is C. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning_images/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning.pdf-0017-02.png)


Figure 11: The example is formatted in _Reassembling_ configuration with the _3D-Geometry_ Quantities. The answer to this puzzle is B. 

## **C Coarse-grained Perception on Different Categories** 

Table 5 shows MLLMs coarse-grained perception performance on different categories. 

## **D Question with text description** 

To investigate how the models perform subsequent reasoning with accurate visual detail awareness, we randomly select 95 puzzles (five for each pattern in each task configuration) and provide text descriptions for each panel in the puzzle. To ensure a similar level of granularity for text descriptions, we first use GPT-4V to provide the original text descriptions, which human annotators further modify without introducing unrelated details (Figure 12). 

We input MLLMs with both the AVR question and text descriptions. The result is shown in Table 6. With the help of text descriptions, MLLMs, especially closed-sourced MLLMs, can build their abstract visual reasoning on correct visual detail foundations, gaining significant improvement in the performance (11.57% to 44.21%). GPT-4V even shows on-par performance (65.26%) with humans, highlighting the importance of visual perception ability. We also want to point out that some puzzles containing abstract shapes are challenging to 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning_images/nips_2024_MARVEL_Multidimensional_Abstraction_and_Reasoning_through_Visual_Evaluation_and_Learning.pdf-0018-00.png)


**----- Start of picture text -----**<br>
Context<br>Choices<br>GPT-4V version<br>Question part:<br>In the first panel: There is a circle with a smaller circle inside it.<br>In the second panel: There is a square with a smaller square inside it.<br>In the third panel: There is a triangle with a smaller triangle inside it.<br>In the fourth panel: There is a rectangle with a smaller rectangle inside it.<br>In the fifth panel: This panel is missing and needs to be filled with the correct choice from the options below.<br>Choices part:<br>In the first choice: There is a triangle with a line through it from the bottom left corner to the mid-right side.<br>In the second choice: There is a triangle with a line through it from the top to the bottom center.<br>In the third choice: There is a triangle with a line through it from the top left corner to the bottom right corner.<br>In the fourth choice: There is a triangle with a line through it from the top right corner to the bottom left corner.<br>Modified version<br>Question part:<br>In the first panel: There is a circle with a smaller circle inside it.<br>In the second panel: There is a square with a smaller square inside it.<br>In the third panel: There is a triangle with a smaller triangle inside it.<br>In the fourth panel: There is a rectangle with a smaller rectangle inside it.<br>In the fifth panel: This panel is missing and needs to be filled with the correct choice from the options below.<br>Choices part:<br>In the first choice:  There is a triangle with a smaller triangle inside it.<br>In the second choice:  There is a triangle with a smaller triangle intersecting with it on the top left part.<br>In the third choice:  There is a triangle with a smaller triangle above it on the top left part. Two triangles don't touch each other.<br>In the fourth choice:  There is a triangle with a smaller triangle intersecting with it on the bottom part.<br>**----- End of picture text -----**<br>


Figure 12: An example of the annotation process. We first generate text descriptions from GPT-4V, and the output will further be modified by human annotators (red _→_ green). 

|Pattern|Human|Open-sourced MLLMs|Open-sourced MLLMs|Open-sourced MLLMs|Open-sourced MLLMs|Open-sourced MLLMs|Closed-sourced MLLMs|Closed-sourced MLLMs|Closed-sourced MLLMs|Closed-sourced MLLMs|
|---|---|---|---|---|---|---|---|---|---|---|
|||Qwen-VL|Fuyu†|Blip-2†|InstructBLIP†|LLaVA-1.5|GPT-4V|Gemini†|Claude3 (Sonnet)†|Claude3 (Opus)|
|Temporal Movement<br>Spatial Relationship<br>Quantities<br>Mathematical<br>2D-Geometry<br>3D-Geometry|82.08<br>70.42<br>81.57<br>60.00<br>59.17<br>50.00|23.81<br>20.83<br>15.76<br>21.25<br>17.50<br>15.00|24.76<br>25.00<br>24.85<br>25.00<br>22.50<br>15.00|25.71<br>26.67<br>27.88<br>24.17<br>20.83<br>15.00|25.71<br>26.67<br>27.27<br>24.17<br>20.83<br>15.00|20.95<br>26.67<br>28.48<br>28.75<br>25.00<br>15.00|26.67<br>24.17<br>21.21<br>21.67<br>18.33<br>30.00|22.86<br>29.17<br>24.85<br>23.75<br>25.00<br>30.00|23.81<br>34.17<br>24.85<br>23.33<br>30.83<br>20.00|31.43<br>26.67<br>27.88<br>27.50<br>33.33<br>25.00|



Table 3: Performance of different models for different patterns. † notes the result is attributed to inductive bias. 

describe. A tool that can convert images to SVGs and text descriptions (Wang et al., 2024) can be a possible approach to mitigate the difficulty and enhance MLLMs performance. 

## **E Few-Shot COT** 

We show our prompting template in Table 11. We discuss few-shot performance in Tables 7 to 10. Table 7 shows few-shot result with Chain-of-Thought demonstrations in the prompt. Table 8 compares one-shot result with Chain-of-Thought demonstrations from same pattern or different pattern picked randomly (OOD). Table 9 compares one-shot result with Chainof-Thought demonstrations from single task format (sequence) and two different task 

18 

|Task format|Human|Open-sourced MLLMs|Open-sourced MLLMs|Open-sourced MLLMs|Open-sourced MLLMs|Open-sourced MLLMs|Closed-sourced MLLMs|Closed-sourced MLLMs|Closed-sourced MLLMs|Closed-sourced MLLMs|
|---|---|---|---|---|---|---|---|---|---|---|
|||Qwen-VL|Fuyu†|Blip-2†|InstructBLIP†|LLaVA-1.5|GPT-4V|Gemini†|Claude3 (Sonnet)†|Claude3 (Opus)|
|Sequence<br>Two-row<br>Matrix<br>Group<br>Reassembling|67.84<br>71.56<br>73.56<br>58.18<br>50.00|18.18<br>16.00<br>23.56<br>21.48<br>15.00|22.42<br>22.67<br>29.78<br>21.48<br>15.00|23.03<br>24.89<br>28.89<br>21.48<br>15.00|23.03<br>24.44<br>28.89<br>21.48<br>15.00|23.64<br>25.78<br>30.67<br>25.19<br>15.00|28.48<br>22.22<br>20.44<br>17.04<br>30.00|23.03<br>24.00<br>26.67<br>25.93<br>30.00|26.06<br>25.78<br>27.56<br>27.41<br>20.00|30.30<br>32.00<br>27.56<br>24.44<br>25.00|



Table 4: Performance of different models for different task configurations. † notes the result is attributed to inductive bias. 

|Category|Qwen-VL|Fuyu|Blip2|InstructBLIP|LLaVA-1.5|GPT-4V|Gemini|Claude3 (Sonnet)*|Claude3 (Opus)|
|---|---|---|---|---|---|---|---|---|---|
|Location<br>Color<br>Shape<br>Quantity<br>Comparison|40.00<br>37.74<br>33.33<br>36.84<br>40.94|15.38<br>26.42<br>36.16<br>36.84<br>40.35|40.00<br>58.49<br>45.20<br>60.86<br>52.05|35.38<br>58.49<br>36.72<br>56.25<br>53.22|61.54<br>58.49<br>29.38<br>57.57<br>57.31|41.54<br>52.83<br>45.76<br>51.64<br>60.82|41.54<br>54.72<br>53.67<br>39.47<br>41.52|32.31<br>47.17<br>57.63<br>56.58<br>42.11|32.31<br>62.26<br>45.76<br>48.36<br>47.95|



Table 5: Coarse-grained Perception Performance on Different Categories bias. 

formats (sequence and two-row). Table 10 shows the model’s few-shot performance on different patterns. 

19 

|Input|Open-sourced MLLMs|Open-sourced MLLMs|Open-sourced MLLMs|Open-sourced MLLMs|Open-sourced MLLMs|Closed-sourced MLLMs|Closed-sourced MLLMs|Closed-sourced MLLMs|Closed-sourced MLLMs|
|---|---|---|---|---|---|---|---|---|---|
||Qwen-VL|Fuyu|Blip-2|InstructBLIP|LLaVA-1.5|GPT-4V|Gemini|Claude3 (Sonnet)|Claude3 (Opus)|
|AVR<br>AVR+Text|23.16<br>24.21|23.16†<br>26.32|23.16†<br>26.32|23.16†<br>13.68|21.05<br>28.42|21.05<br>65.26|26.32†<br>37.89|27.37†<br>49.47|30.53<br>55.79|



Table 6: Performance of different models after introducing text description in the input. † notes the result is attributed to inductive bias. 

|Model|zero-shot|one-shot|two-shot|
|---|---|---|---|
|||||
|Gemini<br>GPT-4V<br>Claude3 (sonnet)<br>Claude3 (Opus)|25.06<br>22.34<br>26.49<br>28.83|28.7<br>26.1<br>25.97<br>27.79|26.75<br>28.31<br>26.23<br>23.64|



Table 7: Few-shot COT Accuracy 

|Model|one-shot|one-shot (OOD)|
|---|---|---|
||||
|Gemini<br>GPT-4V<br>Claude3 (sonnet)<br>Claude3 (Opus)|23.85<br>27.31<br>25.00<br>29.62|24.62<br>23.85<br>26.92<br>25.00|



Table 8: Few-shot COT Ablation 

|Model|two-shot|two-shot (mix)|
|---|---|---|
||||
|Gemini<br>GPT-4V<br>Claude3 (sonnet)<br>Claude3 (Opus)|30.48<br>29.52<br>28.57<br>21.90|24.29<br>28.10<br>23.81<br>22.86|



Table 9: Few-shot COT Ablation 

||one -shot|one -shot|one -shot|one -shot|two -shot|two -shot|two -shot|two -shot|
|---|---|---|---|---|---|---|---|---|
||Gemini|GPT-4V|Claude3 (Sonnet)|Claude3 (Opus)|Gemini|GPT-4V|Claude3 (Sonnet)|Claude3 (Opus)|
|Temporal Movement<br>Spatial Relationship<br>Quantities<br>Mathematical<br>2D-Geometry<br>3D-Geometry|24.76<br>30.00<br>21.21<br>27.92<br>26.67<br>25.00|27.62<br>33.33<br>27.88<br>28.75<br>25.00<br>35.00|17.14<br>24.17<br>27.88<br>25.42<br>30.83<br>45.00|29.52<br>26.67<br>26.67<br>26.67<br>30.83<br>30.00|33.33<br>25.83<br>25.45<br>25.83<br>26.67<br>20.00|28.57<br>22.50<br>31.52<br>27.50<br>30.00<br>35.00|24.76<br>30.83<br>25.45<br>22.92<br>28.33<br>40.00|18.10<br>19.17<br>27.27<br>22.08<br>31.67<br>20.00|



Table 10: Few-shot performance on different patterns 

20 

|**Experiment**|**Prompt Example**|
|---|---|
|||
|AVR Question (Reassembling)|_[IMG]_You are given a puzzle. The puzzle consists of a context part on the top and<br>the choices in the bottom. The context part on the top is an unfolded diagram of the<br>3D shape. The choices part on the bottom contains 4 options (marked by 1, 2, 3, or 4)<br>represents the correct three-dimensional assembly. Which option (either 1, 2, 3, or 4)<br>is the most appropriate answer?|
|||
|AVR Question (Group)|_[IMG]_You are given a puzzle. The puzzle consists of a context part on the top and<br>the choices at the bottom. The context part on the top is a set of visual patterns<br>arranged in a three-by-one sequence, with the last piece missing. The choices part on<br>the bottom contains four options (marked by 1, 2, 3, or 4). Which option (1, 2, 3, or 4)<br>is the most appropriate answer to fll the missing part?|
|||
|AVR Question (Matrix)|_[IMG]_You are given a puzzle. The puzzle consists of a context part on the top and the<br>choices at the bottom. The context part on the top is a set of visual patterns arranged<br>in a three by three matrix, with the bottom right piece missing. The choices part on<br>the bottom contains four options (marked by 1, 2, 3, or 4). Which option (1, 2, 3, or 4)<br>is the most appropriate answer to fll the missing part?|
|||
|AVR Question (Sequence)|_[IMG]_You are given a puzzle. The puzzle consists of a context part on the top and<br>the choices part on the bottom. The context part on the top is a set of visual patterns<br>arranged sequentially, with the last piece missing. The choices part on the bottom<br>contains four options (marked by 1, 2, 3, or 4). Which option (1, 2, 3, or 4) is the most<br>appropriate answer to fll the missing part?|
|||
|AVR Question (Two-row)|_[IMG]_You are given a puzzle. The puzzle consists of a context part on the top and<br>the choices part on the bottom. The context part on the top is a set of visual patterns<br>arranged in a three by two matrix, with the bottom right piece missing. The choices<br>part on the bottom contains four options (marked by 1, 2, 3, or 4). Which option (1, 2,<br>3, or 4) is the most appropriate answer to fll the missing part?|
|||
|Perception Question (fne-grained)|_[IMG]_You are given a puzzle. The puzzle consists of a context part on the top and<br>the choices part on the bottom. The context part on the top is a set of visual grids<br>arranged in a_m by n_sequence, with the last piece missing. The choices part on the<br>bottom contains four choices (marked by 1, 2, 3, or 4). _[question]_|
|||
|Perception Question (coarse-grained)|_[IMG]_You are given a puzzle. The puzzle consists of a context part on the top and<br>the choices part on the bottom. The context part on the top contains some grids, with<br>the last missing blank grid to be completed. The choices part on the bottom contains<br>a sequence of grids representing the possible choices. How many grids, including<br>the blank grid, are in the context part?|
|||
|Perception Question (coarse-grained)|_[IMG]_You are given a puzzle. The puzzle consists of a question part on the top and<br>the choices part on the bottom. The question part on the top contains some grids,<br>with the last missing blank grid to be completed. The choices part on the bottom<br>contains a sequence of grids representing the possible choices. How many grids are<br>there in the choices part?|
|||
|Perception Question (coarse-grained)|_[IMG]_You are given a puzzle. The puzzle consists of a question part on the top and<br>the choices part on the bottom. The question part on the top contains some grids,<br>with the last missing blank grid to be completed. The choices part on the bottom<br>contains a sequence of grids representing the possible choices. How many grids,<br>including the blank grid, are there in the whole puzzle?|



Table 11: Prompt examples for our experiments. 

21 

