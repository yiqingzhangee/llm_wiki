# **ChartMimic: Evaluating LMM’s Cross-Modal Reasoning Capability via Chart-to-Code Generation** 

**Chufan Shi**[1] _[∗]_ **Cheng Yang**[2] _[∗]_ **Yaxin Liu**[1] _[∗]_ **Bo Shui**[1] _[∗]_ **Junjie Wang**[3] _[∗]_ **Mohan Jing**[1] **Linran Xu**[1] **Xinyu Zhu**[1] **Siheng Li**[1] **Yuxiang Zhang**[3] **Gongye Liu**[1] **Xiaomei Nie**[1] **Deng Cai**[4] **Yujiu Yang**[1] _[†]_ 1Tsinghua University 2The Chinese University of Hong Kong 3Waseda University 4Tencent AI Lab `chartmimic@gmail.com` 

## **Abstract** 

We introduce a new benchmark, ChartMimic, aimed at assessing the visuallygrounded code generation capabilities of large multimodal models (LMMs). ChartMimic utilizes information-intensive visual charts and textual instructions as inputs, requiring LMMs to generate the corresponding code for chart rendering. ChartMimic includes 1 _,_ 000 human-curated (figure, instruction, code) triplets, which represent the authentic chart use cases found in scientific papers across various domains (e.g., Physics, Computer Science, Economics, etc). These charts span 18 regular types and 4 advanced types, diversifying into 191 subcategories. Furthermore, we propose multi-level evaluation metrics to provide an automatic and thorough assessment of the output code and the rendered charts. Unlike existing code generation benchmarks, ChartMimic places emphasis on evaluating LMMs’ capacity to harmonize a blend of cognitive capabilities, encompassing visual understanding, code generation, and cross-modal reasoning. The evaluation of 3 proprietary models and 11 open-weight models highlights the substantial challenges posed by ChartMimic. Even the advanced GPT-4V, Claude-3-opus only achieve an average score of 73 _._ 2 and 53 _._ 7, respectively, indicating significant room for improvement. We anticipate that ChartMimic will inspire the development of LMMs, advancing the pursuit of artificial general intelligence. 

## **1 Introduction** 

Code generation [1] is a rather demanding task that requires advanced abstract thinking and logical reasoning, reflecting the unique intelligence of human beings. Recently, progresses in artificial general intelligence (AGI) have demonstrated the potential of large foundation models [2, 3] in solving these tasks that were once the exclusive domain of human abilities [4, 5]. However, existing code generation benchmarks [6–9] solely use text as input, while humans receive information from multiple modalities when coding [10, 11], such real-life scenario has not been fully explored. 

Taking a common scene in Fig. 1, researchers often need to write code for data visualization and may already have preferred chart templates at hand. However, they usually lack either the source code or the expertise to reproduce these chart templates. As a result, they turn to large multimodal models (LMMs) as assistants to aid in code generation. In this scenario, coding for scientific charts entails _code generation grounded on visual understanding_ (i.e., chart-to-code generation), which necessitates LMMs to integrate a variety of advanced cognitive capabilities, including visual understanding, code generation, and cross-modal reasoning. Therefore, evaluating the performance of LMMs on this real-world task also enables researchers to pinpoint potential areas for improving models’ capabilities. 

> _∗_ Equal Contribution. Data and code are available at `https://github.com/ChartMimic/ChartMimic` . 

> _†_ Corresponding author. 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0002-00.png)


Figure 1: The real-world example. LMMs assist scientists and researchers in understanding, interpreting and creating charts during the reading and writing of academic papers. These models serve as assistants that enhance the comprehension and presentation of data in scholarly communications. 

To this end, we present **ChartMimic** (Fig. 2), a multimodal code generation benchmark. ChartMimic is characterized by its _(1) information-intensive visual inputs_ , _(2) diverse chart types_ and _(3) multilevel evaluation metrics_ . Specifically, compared to natural images, scientific charts convey nuanced semantic meanings through intricate visual logic, thereby exhibiting higher information density. Based on this, we define two tasks, Direct Mimic and Customized Mimic (Sec. 2.1), utilizing charts and textual instructions as inputs. These tasks challenge LMMs to generate the corresponding code for a given chart or to incorporate new data specified in the instructions, respectively. Through the collection of academic documents and scientific papers, we identify 22 commonly used chart types and 191 subcategories. Subsequently, we manually annotate a total of 1 _,_ 000 (figure, instruction, code) triplets for these types (Sec. 2.2). Furthermore, we establish automatic evaluation metrics from both high-level and low-level perspectives to thoroughly assess the performance of LMMs (Sec. 2.4). 

We conduct examination of 14 LMMs on ChartMimic (Sec. 3.2), including 3 proprietary models and 11 open-weight models. We observe that while several open-weight models can match the performance of proprietary models such as GPT-4V on public leaderboards [12], a significant performance gap still persists on ChartMimic. Specifically, the best open-weight model, Phi-3-Vision, significantly lags behind GPT-4V, with only half the performance on two tasks, indicating substantial room for improvement. Additionally, correlation analysis (Sec. 4.3) demonstrates the high correlation between our automatic metrics and human judgment. Further error analysis (Sec. 4.4) reveals that hallucination notably hinders LMMs’ performance on ChartMimic, involving the insertion of non-existent text in the ground-truth figures and confusion of similar chart types. 

We envision ChartMimic as a suite of benchmarks designed to guide researchers in understanding the capabilities of their LMMs. By providing a comprehensive evaluation framework, ChartMimic aims to facilitate the growth of foundation models for the community, offering insights into LMM’s various aspects, such as visual understanding, code generation, and cross-modal reasoning. 

## **2 The ChartMimic Benchmark** 

In this section, we first introduce the definition of two tasks involved in ChartMimic, and then delineate the data curation process. Subsequently, we conduct a quantitative analysis to demonstrate the quality and diversity of ChartMimic and compare it with existing related benchmarks. 

## **2.1 Task Definition** 

The capability of LMMs to generate code that can accurately render a specific chart demonstrates their proficiency in visual understanding, code generation and cross-modal reasoning. Specifically, given the chart _X_ and the instruction _I_ , the LMM _f_ is expected to output the code _C_ that satisfies the requirements outlined in the instruction: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0002-09.png)


As shown in Fig. 2, based on the information provided in the instructions, we propose two tasks: 

**Direct Mimic.** The LMM is requested to directly generate code that can reproduce the provided chart. This task evaluates the LMM’s ability to comprehensively understand and reason about the chart to generate executable code. 

**Customized Mimic.** With additional customized data provided in the instruction, this task requires the LMM to generate code that can render a new chart encoding the customized data while maintaining 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0003-00.png)


Figure 2: The pipeline of ChartMimic. We provide 1 _,_ 000 human-curated (figure, instruction, code) triplets. We use ChartMimic to evaluate LMMs’ proficiency in the multimodal chart-to-code generation, resulting in both high-level and low-level evaluation results. 

the aesthetic and design of the provided chart. This task requires the LMM to understand both the chart and customized data to integrate them effectively. 

After obtaining the code generated by LMMs, we execute it to render the corresponding chart and subsequently compare its similarity with the ground-truth chart. Illustrative examples of the two tasks are shown in Fig. 2. To accommodate the defined tasks above, we propose ChartMimic, a benchmark designed to evaluate the comprehension of charts and their conversion into executable code. 

## **2.2 Data Curation Process** 

ChartMimic distinguishes itself through 4 fundamental considerations: _(1) diversity of chart types, (2) balance of charts complexity, (3) reduction of data leakage, (4) integration of authentic user requirements._ We keep on these principles to complete the data curation for ChartMimic through a four-step pipeline. We provide an overview here and more details in Appendix A. 

**General Filtering.** We scrape figures from source files of publications on arXiv[3] that hold a CC BY 4.0 license and have a publication date after February 2024. We extract figures in PDF format, yielding approximately 174 _,_ 100 figures across various domains (e.g., Physics, Computer Science, Economics, etc). We then filter the figures based on the criteria of how designers select inspiring visualization examples, including clarity and visual appeals [13, 14], color schemes [13], and uniqueness of the chart within its category, resulting in a refined collection of about 15 _,_ 800 figures. 

**Diversity and Information Density Filtering.** We first initialize a data pool, creating a taxonomy of chart types and subcategories and comparing each chart against the most similar one within the corresponding category in the data pool. Should significant differences in visual complexity or information density [13] be found, the chart is added to the pool. Otherwise, we retain the one that better fulfills the criteria for diversity and effective information communication. Ultimately, this approach yields a curated collection of 279 charts from the generally filtered 15 _,_ 800 ones. 

**Manual Selection and Filling.** In addition to sourcing from arXiv, we also collect charts from various platforms, including the Matplotlib gallery, Stack Overflow, and plotting-related forums on Twitter and Reddit. These charts are selected for their unique styles not represented in our arXiv curated 279 charts. To mitigate the risk of data leakage, we rigorously process the data and color styling of these charts, replacing existing color schemes with those unseen in our data pool while maintaining their aesthetic appeal. Consequently, we obtain 500 prototype charts for ChartMimic. 

> 3 `https://arxiv.org/` 

3 

Table 1: Statistics of ChartMimic. Each one in Combination, HR, Muiltidiff, and PIP is an independent subcategory. We measure code length in terms of tokens, utilizing the Llama3 tokenizer [15]. In the level count, “A/B/C” denotes the number of chart for “easy/medium/hard” level, respectively. 

|Type|Bar<br>Line<br>ErrorBar<br>Heatmap<br>Box<br>Scatters<br>Hist<br>Radar<br>3D<br>Pie<br>ErrorPoint<br>Violin|
|---|---|
|Count<br>Subcategories<br>Code Length (AVG.)<br>Code Length (STD.)<br>Level Count|100<br>80<br>30<br>30<br>25<br>25<br>20<br>20<br>15<br>15<br>10<br>10<br>16<br>8<br>12<br>4<br>6<br>4<br>3<br>6<br>5<br>8<br>5<br>3<br>689.3<br>805.7<br>651.9<br>664.6<br>659.9<br>617.2<br>507.3<br>772.2<br>668.7<br>385.1<br>523.7<br>912.0<br>289.0<br>272.2<br>146.1<br>303.0<br>248.9<br>287.6<br>161.2<br>136.4<br>168.9<br>123.1<br>228.9<br>298.7<br>53 / 40 / 7 74 / 6 / 0 17 / 13 / 0 0 / 19 / 11 15 / 10 / 0 20 / 5 / 0<br>13 / 7 / 0<br>13 / 7 / 0<br>2 / 8 / 5<br>10 / 5 / 0<br>7 / 3 / 0<br>4 / 6 / 0|
|Type|Area<br>Contour<br>Density<br>Graph<br>Quiver<br>Treemap Combination<br>HR<br>Muiltidiff<br>PIP<br>Total|
|Count<br>Subcategories<br>Code Length (AVG.)<br>Code Length (STD.)<br>Level Count|5<br>5<br>5<br>5<br>5<br>5<br>30<br>25<br>25<br>10<br>500<br>2<br>3<br>4<br>4<br>4<br>4<br>30<br>25<br>25<br>10<br>101+(90)<br>832.8<br>352.8<br>431.8<br>310.8<br>845.0<br>302.2<br>641.9<br>689.9<br>788.1<br>958.9<br>682.9<br>177.5<br>136.7<br>71.9<br>61.7<br>607.0<br>34.7<br>159.0<br>301.5<br>331.0<br>285.9<br>285.1<br>4 / 1 / 0<br>0 / 0 / 5<br>3 / 2 / 0<br>5 / 0 / 0<br>0 / 2 / 3<br>5 / 0 / 0<br>3 / 19 / 8<br>1 / 4 / 20 0 / 12 / 13 0 / 0 / 10<br>249 / 169 / 82|



**Code and Instruction Writing.** We propose to manually write codes and instructions for ChartMimic based on the collected 500 prototype charts. Initially, the annotators are tasked with meticulously reproducing the 500 prototype charts using Python code, resulting in a set of 500 (figure, code, instruction) triplets for the Direct Mimic task. Subsequently, to simulate the scenario of Customized Mimic, annotators are instructed to modify the chart data in the Direct Mimic task by integrating new data from various domains. They are then required to modify the corresponding codes and instructions, leading to the 500 (figure, code, instruction) triplets for the Customized Mimic task. Consequently, we obtain the ChartMimic benchmark, comprising 1 _,_ 000 high quality test examples. 

## **2.3 Data Statistics** 

Notably, the (figure,code,instruction) triplets for both Direct and Customized Mimic tasks share the same figure. Therefore, we detail the data statistics for the 500 triplets in the Direct Mimic task. As depicted on the left side of Fig. 2, ChartMimic encompasses a total of 22 categories, with 18 types of regular charts and 4 types of advanced charts. For the 18 regular chart types, we identify 101 subcategories, with the definitions and examples of each subcategory provided in Appendix B. The advanced chart types such as Plot-in-Plot (PIP), Multidiff, and Combination are distinct forms of amalgamating multiple chart sets into a singular chart. Given the variety of their internal combination elements, each of these can be treated as a unique subcategory. Meanwhile, the Hard-to-Recognize (HR) category encapsulates unclassifiable charts, with each chart being considered a category unto itself. When we factor in the additional 90 subcategories represented by these advanced chart types, ChartMimic encompasses a total of 191 subcategories. This extensive diversity underscores the comprehensive nature of our benchmark. Additionally, we employ Llama3 [15] tokenizer to measure the code length. The results are shown in Table 1, which shows an average code token length of 635 _._ 98 with a standard deviation of 253 _._ 06. Moreover, we conduct a manual complexity assessment for each chart, classifying them into three levels: easy, medium, and hard. The distribution of charts according to complexity levels is as follows: 249 for easy, 169 for medium, and 82 for hard. 

## **2.4 Evaluation Metrics** 

For tasks within ChartMimic, an appropriate evaluation necessitates comparing the visual similarity between the generated and ground-truth figures. To achieve automatic evaluation, we propose multilevel metrics (i.e., low-level and high-level), which assess the similarity at different granularities. We compute the average score between the low-level and high-level scores, which range from 0 to 100, as the **overall score** . The illustration of the multi-level metrics is depicted in Fig. 2. 

**GPT-4V Score.** Following the successful utilization of large foundation models like GPT-4 for automated evaluation in both natural language processing [16–18] and computer vision [19, 20], we adopt the GPT-4V score as our high-level metric. Specifically, we input both the ground-truth chart and the generated chart to GPT-4V, and instruct GPT-4V to output a high-level similarity score between 0 and 100. The detailed prompt template can be found in Appendix C. 

In addition to high-level similarity, evaluating low-level elements between generated and ground-truth figures can help better analyze the behaviors of LMMs. Therefore, we propose evaluating four key low-level elements: text, layout, type and color. Extracting them from figures is a challenging task, as existing extraction models often fall short in terms of accuracy [26]. Considering that figures 

4 

Table 2: A comparison of our proposed ChartMimic to other benchmarks. “I” and “NL” indicate “Image” and “Natural Language” respectively. 

|Benchmarks|Source<br># of Chart Types # of Test Instances Input Format Output Format Evaluation Metric|
|---|---|
|_Chart Understanding Benchmarks_||
|ChartQA [21]<br>Chart-to-Text [22]<br>ChartSumm [23]|Human Curated<br>3<br>10K<br>I+NL<br>NL<br>Accuracy<br>Crawl<br>6<br>44K<br>I+NL<br>NL<br>Match-based<br>Human Curated<br>3<br>84K<br>I+NL<br>NL<br>Match-based|
|_Code Generation Benchmarks_||
|HumanEval [6]<br>MBPP [7]<br>MMCode [24]<br>MatPlotBench [20]<br>Design2Code [25]|Human Curated<br>-<br>164<br>Code<br>Code<br>Pass Rate<br>Human Curated<br>-<br>500<br>NL+Code<br>Code<br>Pass Rate<br>Crawl<br>-<br>263<br>I+NL<br>Code<br>Pass Rate<br>Human Curated<br>13<br>100<br>NL<br>Code<br>GPT-4V Score<br>Crawl<br>HTML<br>484<br>I+NL<br>Code<br>Multi-Level|
|ChartMimic (Ours)|Human Curated<br>22<br>1K<br>I+NL<br>Code<br>Multi-Level|



are rendered based on the code, we design a code tracer to monitor the execution process of the ground-truth code and generated code. The code tracer records the text, layout, type and color information. We calculate the F1 score of these elements as their corresponding score. And we average text, layout, type and color scores to get the low-level score. The methodology for obtaining these elements is briefly introduced as follows, with detailed descriptions provided in Appendix C. 

**Text Score.** During the code execution process, for the function responsible for adding text elements to the rendered figures, the code tracer monitors it and records each text element parameter. 

**Layout Score.** Layout refers to the arrangement of subplots in the figure. At the end of the code execution, the code tracer traverses all subplots in the figure and obtains their layout information. 

**Type Score.** For each plot function, which serves the purpose of adding a specific chart type instance to the figure, the code tracer monitors its invocation status and records every invoked plot function. 

**Color Score.** For each plot function, it will return the chart type instance at the end of the function invocation. The code tracer accesses the color attributes of these chart type instances. 

## **2.5 Comparisons with Existing Benchmarks** 

To further distinguish the difference between ChartMimic and other existing ones, we elaborate the benchmark details in Table 2. From the chart understanding perspective, the prior benchmarks [21– 23] are heavily focused on simple questions about the data in the charts and lack of both difficulty and diversity in chart types. In the code generation aspect, HumanEval [6], MBPP [7] and MatPlotBench [20] only consider tasks with text inputs, which may not meet requirements in the era of LMMs. Recently, MMCode [24] attempted to create a benchmark for multimodal code generation, but the vision inputs for their task are still overly simple and only have a single pass rate evaluation metric. Design2Code [25] is the most similar one to ours, which focuses on automated webpage generation. Although they use multi-level evaluation metrics like ours, their test HTMLs all come from the C4 dataset [27], which may pose a risk of data leakage. Our ChartMimic benchmark gives complex scientific charts as inputs, demanding capabilities on grounding visual understanding into code generation, and provides multi-level evaluation metrics. 

## **3 Experiment** 

## **3.1 Baseline Setup** 

We benchmark 14 widely utilized proprietary and open-source models currently available in the field. For proprietary models, we consider 3 representative models: GPT-4V [2], Claude-3-opus [28] and GeminiProVision [3]. For the open-weight models, we choose 11 competitive models. We list them in increasing order of total parameter size from 4.2B to 34.8B, which are: Phi-3-Vision [29], LLaVANext-Vicuna-7B [30], DeepSeek-VL-7B [31], LLaVA-Next-Mistral-7B [30], IDEFICS2-8B [32], MiniCPM-Llama3-V 2.5 [33], Qwen-VL-Chat [34], LLaVA-Next-Vicuna-13B [30], Cogvlm2llama3-chat-19B [35], InternVL-Chat-V1.5 [36], LLaVA-Next-Yi-34B [30]. We start with the 

5 

Table 3: The ChartMimic leaderboard with **Direct Mimic** task. The best scores are in **bold** . We also include the code execution success rate (Exec. Rate) and model size (Params). 

|Model<br>Params<br>Exec.<br>Rate|Model<br>Params<br>Exec.<br>Rate|Model<br>Params<br>Exec.<br>Rate|Low-Level|High-Level|High-Level|
|---|---|---|---|---|---|
||||Text<br>Layout<br>Type<br>Color<br>Avg.|GPT-4V||
||||_Proprietary_|||
|GeminiProVision<br>Claude-3-opus<br>GPT-4V|-<br>-<br>-|64.2<br>86.4<br>**91.4**|36.9<br>58.6<br>45.1<br>39.6<br>45.0<br>61.8<br>79.0<br>45.3<br>37.9<br>56.0<br>**74.3**<br>**87.1**<br>**73.8**<br>**62.1**<br>**74.3**|38.1<br>45.4<br>**68.4**|41.6<br>50.7<br>**71.4**|
||||_Open-Weight_|||
|Qwen-VL-Chat<br>LLaVA-Next-Vicuna-7B<br>LLaVA-Next-Vicuna-13B<br>IDEFICS2-8B<br>LLaVA-Next-Yi-34B<br>LLaVA-Next-Mistral-7B<br>DeepSeek-VL-7B<br>Cogvlm2-llama3-chat-19B<br>InternVL-Chat-V1.5<br>MiniCPM-Llama3-V2.5<br>Phi-3-Vision|9.6B<br>7.1B<br>13.4B<br>7.6B<br>34.8B<br>7.6B<br>7.3B<br>19.2B<br>26.0B<br>8.4B<br>4.2B|4.0<br>6.2<br>11.6<br>42.4<br>45.6<br>57.6<br>39.0<br>51.4<br>60.0<br>58.2<br>**66.0**|0.5<br>2.8<br>1.1<br>0.8<br>1.3<br>0.8<br>2.1<br>1.4<br>1.6<br>1.5<br>1.9<br>5.7<br>4.0<br>4.3<br>4.0<br>5.7<br>28.4<br>11.9<br>10.9<br>14.2<br>11.4<br>26.7<br>18.2<br>16.6<br>18.2<br>11.3<br>31.5<br>22.3<br>21.9<br>21.8<br>13.8<br>26.4<br>23.9<br>19.0<br>20.8<br>20.6<br>31.1<br>21.7<br>20.0<br>23.3<br>26.9<br>42.1<br>26.1<br>24.9<br>30.0<br>22.5<br>40.1<br>31.4<br>26.6<br>30.2<br>**36.4**<br>**45.0**<br>**39.0**<br>**34.3**<br>**38.7**|0.9<br>1.6<br>2.8<br>8.8<br>12.7<br>14.8<br>16.8<br>17.4<br>23.4<br>24.2<br>**25.1**|1.1<br>1.6<br>3.4<br>11.5<br>15.4<br>18.3<br>18.8<br>20.4<br>26.7<br>27.2<br>**31.9**|



direct prompting method, which provides the reference chart with direct instructions. The specific instructions and models’ decoding setups can be found in Appendix D. 

## **3.2 Main Results** 

We present the main results of 14 LMMs on ChartMimic. Table 3 and Table 4 show the results on the Direct Mimic and Customized Mimic task, respectively. The key findings are as follows: 

**GPT-4V performs best among proprietary models, while Phi-3-Vision excels among openweight models.** In proprietary models, GPT-4V achieves an overall score of 71 _._ 4 in Direct Mimic and 72 _._ 33 in Customized Mimic, surpassing the second-place model Claude-3-opus by 21 _._ 3 and 21 _._ 1, respectively. Within open-weight models, Phi-3-Vision, despite its 4.2B parameters, reaches the overall score of 31 _._ 9 in Direct Mimic and 40 _._ 18 in Customized Mimic, which outperforms LLaVA-Next-Yi-34B (15 _._ 4 in Direct Mimic and 20 _._ 2 in Customized Mimic). This indicates that even models with fewer parameters can achieve decent performance through a refined training process. 

**A performance gap persists between open-weight LMMs and proprietary ones.** Although several open-weight LMMs can exhibit performance comparable to GPT-4V across various benchmarks [12], even the best-performing Phi-3-Vision falls short of achieving half the performance of GPT-4V on both Direct Mimic and Customized Mimic. This apparent performance disparity demonstrates the challenging nature of our ChartMimic benchmark for current open-weight LMMs. Moreover, openweight LMMs even exhibit notable deficiencies in generating executable code, with a majority of them showing an execution rate below 60%. These findings emphasize that there is still a considerable scope for the open-source community to enhance LMMs’ capabilities in terms of complex visual understanding, code generation and cross-modal reasoning. 

**When given additional user-customized data, LMMs exhibit performance improvements.** In the Customized Mimic task, most LMMs get performance improvement compared to the Direct Mimic task, particularly in terms of Text Score. This enhancement can be attributed to the provision of user-customized data, which alleviates the burden on LMMs to recognize textual and data information within the charts. Notably, InternVL-Chat-V1.5 experiences a decrease in its overall score and execution rate. This decline may arise from the reason that user-customized data increases the burden on the models to process different modalities of information. Overall, except for GPT-4V, the performance of the remaining LMMs is still below 60.0 in the Customized Mimic task. This indicates that, in addition to the capability to recognize data in the chart, understanding the layout of the chart and code generation are also key factors for LMMs to better accomplish the task. 

6 

Table 4: The ChartMimic leaderboard with **Customized Mimic** task. The best scores are in **bold** . We also include the code execution success rate (Exec. Rate) and model size (Params). 

|Model<br>Params<br>Exec.<br>Rate|Model<br>Params<br>Exec.<br>Rate|Model<br>Params<br>Exec.<br>Rate|Low-Level|High-Level|High-Level|
|---|---|---|---|---|---|
||||Text<br>Layout<br>Type<br>Color<br>Avg.|GPT-4V||
||||_Proprietary_|||
|GeminiProVision<br>Claude-3-opus<br>GPT-4V|-<br>-<br>-|74.0<br>88.2<br>**93.2**|47.2<br>64.9<br>56.0<br>46.4<br>53.6<br>65.2<br>82.8<br>48.2<br>39.6<br>58.9<br>**77.4**<br>**89.2**<br>**78.3**<br>**65.6**<br>**77.6**|46.8<br>54.4<br>**72.1**|50.2<br>56.7<br>**74.9**|
||||_Open-Weight_|||
|Qwen-VL-Chat<br>LLaVA-Next-Vicuna-7B<br>LLaVA-Next-Vicuna-13B<br>Cogvlm2-llama3-chat-19B<br>LLaVA-Next-Mistral-7B<br>IDEFICS2-8B<br>LLaVA-Next-Yi-34B<br>InternVL-Chat-V1.5<br>DeepSeek-VL-7B<br>MiniCPM-Llama3-V2.5<br>Phi-3-Vision|9.6B<br>7.1B<br>13.4B<br>19.2B<br>7.6B<br>7.6B<br>34.8B<br>26.0B<br>7.3B<br>8.4B<br>4.2B|6.8<br>8.4<br>32.2<br>37.8<br>48.0<br>52.4<br>59.4<br>48.6<br>55.2<br>**69.2**<br>67.2|2.6<br>3.8<br>3.5<br>2.6<br>3.2<br>2.2<br>4.8<br>3.6<br>3.0<br>3.4<br>9.3<br>21.5<br>17.4<br>14.2<br>15.6<br>19.1<br>25.6<br>17.6<br>17.1<br>19.9<br>21.8<br>30.4<br>22.0<br>21.7<br>24.0<br>23.6<br>33.5<br>24.1<br>16.2<br>24.4<br>33.4<br>45.1<br>26.2<br>24.9<br>32.4<br>30.1<br>38.2<br>27.4<br>22.5<br>29.5<br>28.2<br>45.8<br>40.4<br>33.5<br>37.0<br>**40.0**<br>**53.9**<br>43.1<br>35.8<br>**43.2**<br>30.9<br>52.5<br>**46.6**<br>**42.0**<br>43.0|2.2<br>2.7<br>10.8<br>15.0<br>17.4<br>19.8<br>17.7<br>23.3<br>28.4<br>32.1<br>**35.7**|2.7<br>3.1<br>13.2<br>17.4<br>20.7<br>22.1<br>25.0<br>26.4<br>32.7<br>37.7<br>**39.4**|



## **4 Discussion** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0007-03.png)


**----- Start of picture text -----**<br>
GPT-4V GeminiProVision MiniCPM-Llama3-V2.5<br>Claude-3-opus Phi-3-Vision InternVL-Chat-V1.5<br>90<br>70<br>50<br>30<br>10<br>Easy Medium Hard Easy Medium Hard<br>(a) Direct Mimic (b) Customized Mimic<br>Overall Score<br>**----- End of picture text -----**<br>


## **4.1 Different Complexity Levels** 

As shown in Sec. 2.3, the performance of the top-6 70 models across different complexity levels in ChartMimic is reported. With increasing difficulty, there is a consistent decline in performance across all tasks, 50 resembling a stepped pattern. For example, in Customized Mimic task, the performance of GPT-4V at 30 easy, medium, and hard levels are 82 _._ 8, 69 _._ 8, and 61 _._ 6, respectively. Further observations across all tasks indicate that the performance decline of open10 source models from easy to medium difficulty is more Easy Medium Hard Easy Medium Hard (a) Direct Mimic (b) Customized Mimic pronounced than from medium to hard. In contrast, proprietary models exhibit a consistent rate of deFigure 3: Overall scores of top-6 performed cline. This suggests that the foundational capabilities models at different complexity levels. of the open-source models are comparable, and the observed performance variations may largely be influenced by the quality of data. These results demonstrate that the benchmark tasks are inherently challenging and confirms the efficacy of the established difficulty levels. 

## **4.2 Different Prompting Methods** 

We further explore the impact of different prompting methods. Specifically, we choose the best-performing model, GPT-4V [2], to study its performance on the Direct Mimic task. We select three representative methods for prompting: HintEnhanced, SelfRevison, and Scaffold Prompting [37]. HintEnhanced uses 

Table 5: Results for different prompting methods on GPT-4V on Direct Mimic task. 

|Model|Low-Level|High-Level|High-Level|
|---|---|---|---|
||Text Layout Type Color Avg.|GPT-4V||
|Direct<br>|74.3<br>73.8<br>87.1<br>62.1<br>74.3<br><br><br><br><br>|68.4<br>|71.4<br>|
|HintEnhanced<br>SelfRevision<br>Scaffold|71.0<br>71.6<br>85.5<br>59.1<br>71.8<br>71.0<br>70.8<br>84.3<br>59.9<br>71.5<br>67.4<br>67.5<br>81.7<br>56.3<br>68.2|67.3<br>65.1<br>64.4|69.6<br>68.3<br>66.3|



prompt with chain-of-thought [38], explicitly prompting the model to pay attention to important details (e.g., layout, type, text, etc). SelfRevision involves inputting the model’s own output and the rendered chart as additional information, hoping that the model can self-reflect its results and then produce better. Scaffold Prompting [37] which overlays a dot matrix within the image as visual information anchors and leverages multi-dimensional coordinates as textual positional references. Detailed introductions of these prompting methods can be found in Appendix D. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Dimension Parameter Detail Missing Confusion Missing Similar Different<br>Access Extraneous Extraneous<br>11.6% 17.5% 12.3%<br>9.3% (5) (50) (26) 32.7%<br>(4) 42.3% 41.0% (120)<br>(121) (87)<br>40.2% 46.7% 67.3%<br>79.1%(34) (115) (99) (247)<br>(a) Code (b) Text (c) Type (d) Color<br>**----- End of picture text -----**<br>


Figure 4: Error analysis of GPT-4V across four error types on the Direct Mimic task. The number in brackets indicates the count of error case. Error examples can be found in Appendix F. As demonstrated in Table 5, the Direct method proves to be the most straightforward and effective for GPT-4V. Other methods, to varying degrees, tend to reduce performance. Upon examining the cases, we find that the HintEnhanced method, which involves generating captions first, can introduce hallucinations that lead to errors in the subsequently generated code. Case studies suggest that although SelfRevision can identify discrepancies between output results, it often provides incorrect suggestions, resulting in subsequent errors. As for the Scaffold, its implementation can interfere with existing coordinate axis information in charts with high information density, thereby negatively impacting performance. We provide error cases of three prompting methods in Appendix D. 

## **4.3 Correlation with Human Evaluation** 

To evaluate the reliability of the proposed automatic metrics, we calculate their correlation with human evaluations. Specifically, we collect 1,200 charts collected from GPT4V using four different prompting methods (Sec. 4.2) on Direct Mimic. Each charts is evaluated by three individual evaluators, who assign scores ranging from 0 to 100 based on the similarity with the ground-truth chart. Details about the human evaluation process are provided in Appendix E. 

Table 6: Pearson correlation coefficient between multi-level evaluation metric and human evaluation. 

||Metric<br>High-Level<br>Low-Level|Coeffcient<br>0.6942<br>0.7538|p-value<br>_<_0_._0001<br>_<_0_._0001|
|---|---|---|---|



As shown in Table 6, the Pearson correlation coefficient ( _r_ ) indicates that both the high-level ( _r_ = 0 _._ 6942) and low-level metric ( _r_ = 0 _._ 7538) have a high correlation with human judgment, demonstrating the reliability of our multi-level metrics. Moreover, the low-level metric demonstrates a higher correlation with human judgments compared to the high-level metric. This is attributed to its dependence on code execution logic, enabling it to capture detailed elements. 

## **4.4 Error Analysis** 

We conduct a detailed error analysis of the state-of-the-art GPT-4V model to elucidate the current limitations of LMMs on ChartMimic, thereby identifying potential areas for further improvement. Our analysis identifies and category four main types of errors, detailed below. 

**Code-related Errors.** (1) Dimension: Errors associated with data dimension, e.g., data dimension does not satisfy required conditions for operations; (2) Access: Errors occurring when accessing iterable elements out of bounds or accessing undeclared variables; (3) Parameter: Errors related to passing incorrect parameters when invoking functions. As observed in Fig. 4 (a), the majority of errors stem from Dimension, suggesting analysis and operation on data poses a challenge for GPT-4V. 

**Text-related Errors.** (1) Detail: The text in the generated chart is largely consistent with the groundtruth chart text but contains minor discrepancies, such as mixing up “-” and “_”; (2) Missing: The generated chart omits text present in the ground-truth chart; (3) Extraneous: The generated chart contains text not found in the ground-truth chart, e.g., adding text for titles, reflecting the model’s own interpretation of the chart; As depicted in Fig. 4 (b), the majority of errors are Missing, which indicates that for GPT-4V, due to intensive information in charts, it is challenging for it to comprehend the whole scope of the charts, even the basic text recognition task. 

**Type-related Errors.** (1) Confusion: GPT-4V misinterprets the chart type as one that appears or functions similarly, such as mistaking violin plots for box plots, and so on; (2) Missing: The generated chart omits chart types present in the ground-truth chart; (3) Extraneous: The generated chart includes chart types not present in the ground-truth chart. As demonstrated in Fig. 4 (c), although GPT-4V exhibits remarkable capability in object recognition for natural images, it still struggles with scientific charts, which contain more nuanced semantic meanings through visual logic. 

8 

**Color-related Errors.** (1) Similar: The colors are not the same as the ground-truth colors but appear analogous. (2) Different: The colors are entirely dissimilar to the ground-truth colors. As shown in Fig. 4 (d), though GPT-4V can not exactly recognize the accurate colors, it can identify similar ones. 

To sum up, GPT-4V still faces challenges in code generation and exhibits notable visual understanding deficiencies on ChartMimic. It has difficulty accurately recognizing visual elements in figures or may hallucinate incorrect elements and struggle with complex data analysis. Combining these insights, there is a need for further improvement in both visual understanding and code generation. 

## **5 Related Work** 

**Large Multimodal Models.** The proprietary LMMs such as GPT-4V [2], Gemini [3], and Claude3 [28] have enabled complex multimodal interactions. Similarly, emerging open-weight LMMs such as LLaVA [30, 33], Qwen-VL [34], DeepSeek-VL [31] have contributed to the community. Despite these advancements, the effective evaluation of LMMs remains a major challenge. However, effectively evaluating LMMs remains a challenge, with open-source models performing well in benchmarks [12, 21, 39–43] yet falling short in practical applications [25, 44, 45]. This gap emphasizes the need for real-world-based evaluations. ChartMimic addresses this by requiring LMMs to translate complex visual information into code, testing their visual and coding capabilities. 

**Code Generation.** Tasks such as HumanEval [6], MBPP [7], APPS [8], and DS-1000 [9] are important benchmarks in natural language processing, but the inputs for these tasks are single-modal, consisting only of text. With the emergence of LMMs, the evaluation of multimodal code generation has become more important. On top of this, MMCode [24] attempts to address this issue, but their visual inputs may be simplistic and face the problem of a single evaluation method. Design2Code [25] is relevant to our work, and they evaluate LMMs’ code generation abilities through HTML web page generation. However, their test data comes from the C4 [27] dataset, which may pose a risk of data leakage. Meanwhile, they just imitate the HTML and do not take the customized instructions; we take this into consideration. Our ChartMimic provides a new set of manually curated 1000 data pairs, ensuring chart diversity and offering more fine-grained evaluation methods. 

**Chart Understanding.** ChartMimic evaluates the capabilities of LMMs in grounding chart understanding into code generation. Previous works focus on chart question answering [21, 39, 46] and chart captioning [22, 23]. They assess the LMMs’ ability to understand specific data characteristics or summarize key information into text. ChartMimic advances the field by introducing a chart-to-code task, transforming the LMMs’ understanding of charts into code, which is neglected before but a realistic scenario for usage. This approach enables a comprehensive evaluation of the LMMs’ overall comprehension of charts and their ability to express this understanding in code form. Leveraging the linguistic properties of code, our benchmark introduces fine-grained metrics to assess LMMs’ chart understanding capabilities across multiple dimensions, including text, chart type, layout and color. 

## **6 Conclusions** 

In this study, we develop the ChartMimic benchmark to evaluate LMMs’ proficiency capability via chart-to-code generation. ChartMimic focuses on real-world applications for data visualization, aiming to assess LMMs’ ability to harmonize a blend of cognitive capabilities, including visual understanding, code generation, and cross-modal reasoning. We propose two distinct levels of evaluation metrics (low and high level) to provide a comprehensive assessment. ChartMimic directly contributes to the understanding of progress towards artificial general intelligence, reflecting the expertise and reasoning abilities expected of skilled adults in various professional fields. Despite its comprehensive nature, ChartMimic, like any benchmark, has limitations. The manual curation process, although thorough, may introduce biases. Additionally, using scientific charts as informationintensive visual inputs to measure LMMs’ multimodal code generation capabilities, while effective, still encounters domain-specific challenges. Our evaluation metric, despite considering most elements’ similarity, does not uniformly score details of sub-icons, such as markers. We anticipate that ChartMimic will inspire the development of LMMs, advancing the pursuit of artificial general intelligence. Future research could explore various aspects, such as multimodal reasoning prompt strategies, to further reduce the gap between open-weight LMMs and proprietary ones. 

9 

## **References** 

- [1] Qiushi Sun, Zhirui Chen, Fangzhi Xu, Kanzhi Cheng, Chang Ma, Zhangyue Yin, Jianing Wang, Chengcheng Han, Renyu Zhu, Shuai Yuan, et al. A survey of neural code intelligence: Paradigms, advances and beyond. _arXiv preprint arXiv:2403.14734_ , 2024. 

- [2] OpenAI. Gpt-4v(ision) system card. 2023. URL `https://api.semanticscholar.org/CorpusID: 263218031` . 

- [3] Google. Gemini: A family of highly capable multimodal models. _ArXiv_ , abs/2312.11805, 2023. URL `https://api.semanticscholar.org/CorpusID:266361876` . 

- [4] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [5] Zheng Zhu, Xiaofeng Wang, Wangbo Zhao, Chen Min, Nianchen Deng, Min Dou, Yuqi Wang, Botian Shi, Kai Wang, Chi Zhang, et al. Is sora a world simulator? a comprehensive survey on general world models and beyond. _arXiv preprint arXiv:2405.03520_ , 2024. 

- [6] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

- [7] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_ , 2021. 

- [8] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence with apps. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)_ , 2021. 

- [9] Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-tau Yih, Daniel Fried, Sida Wang, and Tao Yu. Ds-1000: A natural and reliable benchmark for data science code generation. In _International Conference on Machine Learning_ , pages 18319–18345. PMLR, 2023. 

- [10] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In _2023 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 9493–9500. IEEE, 2023. 

- [11] Li Fan, Lee Ching-Hung, Han Su, Feng Shanshan, Jiang Zhuoxuan, and Sun Zhu. A new era in human factors engineering: A survey of the applications and prospects of large multimodal models. _arXiv preprint arXiv:2405.13426_ , 2024. 

- [12] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. `https://github.com/open-compass/opencompass` , 2023. 

- [13] Hannah K Bako, Xinyi Liu, Leilani Battle, and Zhicheng Liu. Understanding how designers find and use data visualization examples. _IEEE Transactions on Visualization and Computer Graphics_ , 29(1): 1048–1058, 2022. 

- [14] Annemarie Quispel, Alfons Maes, and Joost Schilperoord. Aesthetics and clarity in information visualization: The designer’s perspective. In _Arts_ , volume 7, page 72. MDPI, 2018. 

- [15] AI@Meta. Llama 3 model card. 2024. URL `https://github.com/meta-llama/llama3/blob/ main/MODEL_CARD.md` . 

- [16] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [17] Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacaeval: An automatic evaluator of instruction-following models. `https://github.com/tatsu-lab/alpaca_eval` , 2023. 

- [18] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators. _arXiv preprint arXiv:2404.04475_ , 2024. 

10 

- [19] Xinlu Zhang, Yujie Lu, Weizhi Wang, An Yan, Jun Yan, Lianke Qin, Heng Wang, Xifeng Yan, William Yang Wang, and Linda Ruth Petzold. Gpt-4v (ision) as a generalist evaluator for vision-language tasks. _arXiv preprint arXiv:2311.01361_ , 2023. 

- [20] Zhiyu Yang, Zihan Zhou, Shuo Wang, Xin Cong, Xu Han, Yukun Yan, Zhenghao Liu, Zhixing Tan, Pengyuan Liu, Dong Yu, et al. Matplotagent: Method and evaluation for llm-based agentic scientific data visualization. _arXiv preprint arXiv:2402.11453_ , 2024. 

- [21] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. _arXiv preprint arXiv:2203.10244_ , 2022. 

- [22] Shankar Kantharaj, Rixie Tiffany Ko Leong, Xiang Lin, et al. Chart-to-text: A large-scale benchmark for chart summarization. _arXiv preprint arXiv:2203.06486_ , 2022. 

- [23] Raian Rahman, Rizvi Hasan, Abdullah Al Farhad, et al. Chartsumm: A comprehensive benchmark for automatic chart summarization of long and short summaries. _arXiv preprint arXiv:2304.13620_ , 2023. 

- [24] Kaixin Li, Yuchen Tian, Qisheng Hu, Ziyang Luo, and Jing Ma. Mmcode: Evaluating multi-modal code large language models with visually rich programming problems. _arXiv preprint arXiv:2404.09486_ , 2024. 

- [25] Chenglei Si, Yanzhe Zhang, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2code: How far are we from automating front-end engineering? _arXiv preprint arXiv:2403.03163_ , 2024. 

- [26] Fanqing Meng, Wenqi Shao, Quanfeng Lu, Peng Gao, Kaipeng Zhang, Yu Qiao, and Ping Luo. Chartassisstant: A universal chart multimodal language model via chart-to-table pre-training and multitask instruction tuning. _arXiv preprint arXiv:2401.02384_ , 2024. 

- [27] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of machine learning research_ , 21(140):1–67, 2020. 

- [28] Anthropic. Introducing the next generation of claude. 2024. URL `https://www.anthropic.com/news/ claude-3-family` . 

- [29] New models added to the phi-3 family, available on microsoft azure, May 2024. URL `https://azure.microsoft.com/en-us/blog/ new-models-added-to-the-phi-3-family-available-on-microsoft-azure/` . 

- [30] Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild, May 2024. URL `https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/` . 

- [31] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world vision-language understanding, 2024. 

- [32] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building visionlanguage models?, 2024. 

- [33] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. LLaVA-UHD: an lmm perceiving any aspect ratio and high-resolution images. _arXiv preprint arXiv:2403.11703_ , 2024. 

- [34] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint arXiv:2308.12966_ , 2023. 

- [35] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models, 2023. 

- [36] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. _arXiv preprint arXiv:2312.14238_ , 2023. 

- [37] Xuanyu Lei, Zonghan Yang, Xinrui Chen, Peng Li, and Yang Liu. Scaffolding coordinates to promote vision-language coordination in large multi-modal models. _arXiv preprint arXiv:2402.12058_ , 2024. 

11 

- [38] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- [39] Zhengzhuo Xu, Sinan Du, Yiyan Qi, Chengjin Xu, Chun Yuan, and Jian Guo. Chartbench: A benchmark for complex visual reasoning in charts. _arXiv preprint arXiv:2312.15915_ , 2023. 

- [40] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? _arXiv preprint arXiv:2307.06281_ , 2023. 

- [41] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? _arXiv preprint arXiv:2403.20330_ , 2024. 

- [42] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. _arXiv preprint arXiv:2311.16502_ , 2023. 

- [43] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. _arXiv preprint arXiv:2310.02255_ , 2023. 

- [44] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. _arXiv preprint arXiv:2404.07972_ , 2024. 

- [45] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. _arXiv preprint arXiv:2401.13649_ , 2024. 

- [46] Nitesh Methani, Pritha Ganguly, Mitesh M Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In _CVPR_ , pages 1527–1536, 2020. 

- [47] Paul Parsons. Understanding data visualization design practice. _IEEE Transactions on Visualization and Computer Graphics_ , 28(1):665–675, 2021. 

- [48] Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, and Hanwang Zhang. Chartllama: A multimodal llm for chart understanding and generation. _arXiv preprint arXiv:2311.16483_ , 2023. 

- [49] Stephanie DH Evergreen. _Effective data visualization: The right chart for the right data_ . SAGE publications, 2019. 

- [50] Xuedi Qin, Yuyu Luo, Nan Tang, and Guoliang Li. Making data visualization more efficient and effective: a survey. _The VLDB Journal_ , 29(1):93–117, 2020. 

- [51] M Ronnier Luo, Guihua Cui, and Bryan Rigg. The development of the cie 2000 colour-difference formula: Ciede2000. _Color Research & Application: Endorsed by Inter-Society Color Council, The Colour Group (Great Britain), Canadian Society for Color, Color Science Association of Japan, Dutch Society for the Study of Color, The Swedish Colour Centre Foundation, Colour Society of Australia, Centre Français de la Couleur_ , 26(5):340–350, 2001. 

- [52] Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Chandu, David Wadden, Kelsey MacMillan, Noah A Smith, Iz Beltagy, et al. How far can camels go? exploring the state of instruction tuning on open resources. _Advances in Neural Information Processing Systems_ , 36: 74764–74786, 2023. 

- [53] Chufan Shi, Haoran Yang, Deng Cai, Zhisong Zhang, Yifan Wang, Yujiu Yang, and Wai Lam. A thorough examination of decoding methods in the era of llms. _arXiv preprint arXiv:2402.06925_ , 2024. 

12 

## **Appendix** 

|**A **|**Data**|**Annotation**|**14**|
|---|---|---|---|
||A.1|Data Annotation Principles . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
||A.2|Data Annotation Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
|**B**|**Chart Taxonomy**||**16**|
|**C **|**Details of Evaluation Metrics**||**24**|
||C.1|GPT-4V Score . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|24|
||C.2|Text Score . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|25|
||C.3|Layout Score<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|25|
||C.4|Type Score<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|26|
||C.5|Color Score . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|26|
|**D **|**Generation Confgurations and Prompting Methods**||**28**|
||D.1|Generation Confgurations<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|28|
||D.2|Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|28|
||D.3|Error Cases of Different Prompting Methods . . . . . . . . . . . . . . . . . . . . .|31|
|**E**|**Correlation with Human Evaluation**||**34**|
|**F**|**Cases of Error Analysis**||**35**|
|**G **|**Ethics and Societal Impact**||**37**|



13 

## **A Data Annotation** 

## **A.1 Data Annotation Principles** 

**Diversity of Chart Types.** Data visualization has become an essential tool for conveying information in various fields, and the design practices and requirements for different types of charts vary significantly[47]. Most of previous work [21–23, 39, 48] focus only on line, bar, pie charts, etc, which are commonly used within the field of computer science. However, with the increasing integration of LMMs into everyday tasks, people from all fields are starting to use generative techniques as daily assistants to enhance their design and creative processes when creating visualizations. In this light, enriching the spectrum of chart diversity is crucial for evaluating LMMs’ proficiency in multimodal chart-to-code generation. 

**Balance of Charts Complexity.** Charts serve as visual aids for data presentation, enabling users to promptly grasp the underlying patterns and significance within the data. It is essential to adopt the appropriate chart type and complexity level to effectively convey the information [49]. Previous works have focused mainly on charts with only a single data format and low information density, which are rarely encountered in practical settings such as academic writing. Noting that LMMs like GPT-4V have already demonstrated outstanding data visualization capabilities, our focus is on using charts that are actually employed in practice, such as those from research papers, and on selecting charts with varying levels of complexity when constructing our benchmark. 

**Reduction of Data Leakage.** Recognizing that augmenting training data is a primary method for enhancing the performance of LMMs has led researchers to more comprehensively exploit all accessible data during pre-training. However, this approach introduces the possibility of data leakage, especially when pre-training data might already include resources such as the matplotlib gallery[4] or other pre-existing datasets, potentially resulting in inaccurate evaluations. To mitigate this issue, we deliberately avoid the use of code that can be readily found online or code that could be autogenerated by large language models for chart creation in constructing our dataset, thereby reducing the probability of data leakage. To further illustrate the potential data leakage in matplotlib gallery, we select the code for 20 charts from the matplotlib gallery. Specifically, we provide the first half of the code as a prefix for Llama3-8B [15] and let the model complete the remaining code. Then we calculate the edit distance between the generated complete code block and the ground truth. Similarly, we apply the same process to the code in Direct Mimic, calculating the edit distance between the generated results and the ground truth. We find that the edit distance for the code in the matplotlib gallery is 22 _._ 1, while the edit distance for Direct Mimic’s code is 39 _._ 8. This indicates that the code in Direct Mimic have a larger edit distance, which further reduces the risk of data leakage compared to the matplotlib gallery ones. 

**Integration of Authentic User Requirements.** Users from diverse domains such as finance, healthcare, education, and engineering demonstrate unique needs and preferences for data visualization. These requirements extend beyond mere diversity; they demand the incorporation of charts capable of articulating complex and multi-dimensional data, and conforming to domain-specific aesthetic preferences[49, 50]. By aligning our dataset construction process with these real-world demands, we enable a more relevant and precise evaluation of LMMs. Adopting this approach not only reflects actual user patterns but also steers the research community in the iterative improvement of LMMs. This focused development will also lead to an enhanced user experience and increased user satisfaction, as the models become more proficient in meeting the sophisticated and varied needs of users. 

## **A.2 Data Annotation Pipeline** 

**General Filtering.** To obtain a high-quality dataset aligned with real-world use cases and to avoid data leakage, we initially scrape figures from source files of publications on arXiv[5] that hold a CC BY 4.0 license and have a publication date after February 2024. We then extract figures in PDF format that contain at least one chart, yielding approximately 174 _,_ 100 figures across various domains (such as Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering and Systems Science, Economics). We further filter the figures based on criteria of how designers select inspiring visualization examples, including clarity and visual 

- 4 `https://matplotlib.org/` 

> 5 `https://arxiv.org/` 

14 

appeals[13, 14], color schemes[13], and uniqueness of the chart within its category, resulting in a refined collection of about 15 _,_ 800 charts. It is noteworthy that during this process, some charts, which do not entirely meet the clarity and aesthetic criteria, are retained in the corpus for their unique and rare qualities, pending further modifications. 

**Diversity and Information Density Filtering.** In this phase, we adopt a two-stage filtering process focused on the diversity of the charts and the effectiveness of information communication, primarily assessed based on the complexity of visual elements and the information density of each chart[13]. Specifically, during the first round, we maintain a data pool, categorizing chart types and subcategories. Each chart is compared against the most similar one within the corresponding category in the data pool. If significant differences in visual complexity or information density are observed, the chart is added to the pool. Otherwise, we retain the one that better fulfills the criteria for diversity and effective information communication. 

After the first round of filtration, we observe a decreasing frequency of charts with increasing levels of difficulty among the remaining 15 _,_ 800 entries. Charts with simpler designs and lower information density often share greater similarity in their arrangement of visual elements, leading to a higher rate of duplicates being identified and removed. In contrast, more complex charts with higher information density are less likely to have comparable counterparts, thereby reducing duplication and enhancing the retention rate of these complex charts. This process effectively refines the complexity spectrum of our dataset, ensuring a balanced distribution. 

Subsequently, we conduct the second round of filtration. We engage five experts from different fields—computer science, visual communication design, and data visualization—to independently assess the data, creating five separate data pools. We then preserve the intersection of these pools and conduct a final, collective filtration from the combined set. The selection of which charts are to be included in the final pool is made through a voting process involving all five experts. Ultimately, this approach results in a curated collection of 279 charts. 

**Manual Selection and Filling.** In addition sourcing from arXiv, we curate chart figures from diverse platforms such as the matplotlib gallery, Stack Overflow, and plotting-related forums on Twitter and Reddit. These charts are deliberately chosen for their distinctive styles, which are not present in our arXiv dataset. To prevent data leakage, we meticulously process the data and styling of these charts, substituting existing style schemes and data with those not present in our dataset, while preserving their visual appeal. 

**Code and Instruction Writing.** We propose to manually write codes and instructions for ChartMimic based on the collected 500 prototype charts. To ensure the quality of the annotation results, we assemble a team of skilled Python users. These individuals are master students from computer science-related disciplines, with at least 4 years of experience using Python and extensive familiarity with matplotlib. Initially, the annotators are tasked with meticulously reproducing the 500 prototype charts using Python and the matplotlib package, resulting in a set of 500 (figure, code, instruction) triplets for the Direct Mimic task. Subsequently, to simulate the real-world scenario of Customized Mimic, annotators are instructed to modify the chart data in the Direct Mimic task by integrating new data from various domains. They are then required to modify the corresponding codes and instructions, leading to the 500 (figure, code, instruction) triplets for the Customized Mimic task. Consequently, we obtain the ChartMimic benchmark, comprising 1 _,_ 000 high quality test examples. 

15 

## **B Chart Taxonomy** 

This section presents the chart taxonomy in ChartMimic. It encompasses a structure of 22 categories according to chart type characteristics and data composition. The categories comprise of: 

- 18 regular types, ordered as follows: Bar, Heatmap, Scatter, Box, Errorbar, Errorpoint, Line, Violin, Radar, Pie, Density, Graph, Quiver, Contour, Histogram, Tree, Area, and 3D charts. 

- 4 advanced types: PIP (Plot-in-Plot), Multidiff (Multiple Differences), Combination, and HR (Hard-to-Recognize). 

The regular types are further divided into subcategories according to chart feature or data characteristics, whereas each chart of the advanced types represents a unique subcategory. The taxonomy showcases examples to showcase diversity of each category. 

**Bar** : Bar chart uses rectangular bars to represent data and can be distinguished by its orientation, horizontal or vertical, with its nuanced data attributes. There are 8 subcategories of the data attributes for each orientation, consisting of 16 subcategories in total, as shown in Figure 5: 

1. Base (single positive data set, unordered) 

2. Sorted (data in ascending or descending sequence) 

3. Grouped (multiple positive data sets, adjacent) 

4. Stacked (multiple positive data sets laying atop one another) 

5. Normalized (proportioned stacks of positive data summing to one, a Stacked variant) 

6. Diverging (multiple stacked data sets expanding from a central axis) 

7. With-Negative (data sets including negative values) 

8. Reverse (exclusively negative data sets) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0016-14.png)


Figure 5: Examples of Bar chart subcategories. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0017-00.png)


   - **Heatmap** : Based on the layout and visual representations, Heatmap is categorized into 4 subcategories, each reflecting unique aspects of data presentation, as shown in Figure 6. 

1. Base (the general layout of a typical heatmap) 

2. Missing-Data (visualization indicating the absence of data points) 

3. Triangle-Layout (heatmap configured in a triangular layout) 

4. Other-Shaped (heatmap comprising elements in non-rectangular forms, such as circles) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0017-06.png)


Figure 6: Examples of Heatmap subcategories. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0017-08.png)


**Scatter** : Scatter plot is classified based on the dot characteristics and data distribution. The taxonomy is divided into 4 subcategories as shown in Figure 7. 

1. Base (basic scatter plot, uniform dot size, color may vary) 

2. Diff-Shape (different dot shapes) 

3. Diff-Size (different dot sizes, such as a bubble chart) 

4. Clustered (scatter plot with clear clustering) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0017-14.png)


Figure 7: Examples of Scatter plot subcategories. 

**Box** : Box plot is characterized by the orientation of boxes and data characteristics. The orientation can be horizontal or vertical, while data characteristics include the grouping of data and the presence of missing lines. The taxonomy is divided into 3 subcategories for each orientation, consisting of 6 subcategories in total, as shown in Figure 8. 

1. Base (single group of data, complete box shape) 

2. Grouped (multiple groups of data, complete box shape) 

3. Missed-Line (missing parts below the first quartile line and above the third quartile line) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0017-20.png)


Figure 8: Examples of Box chart subcategories. 

17 

**Error Bar** : An Error bar chart is an enhanced variant of the basic bar chart, augmented with error margins to represent the variability or uncertainty within the data. Unlike Bar chart, Error Bar chart typically does not include the categories "Normalized" and "Sorted" in the dimension of data attributes, as these are less common. Therefore, Error Bar chart is classified into 6 data attributes subcategories for each orientation, resulting in 12 subcategories in total, as shown in Figure 9. 

1. Base (single positive data set, unordered) 

2. Grouped (multiple positive data sets, adjacent) 

3. Stacked (multiple positive data sets laying atop one another) 

4. Diverging (multiple stacked data sets expanding from a central axis) 

5. With-Negative (data sets including negative values) 

6. Reverse (exclusively negative data sets) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0018-07.png)


Figure 9: Examples of Error Bar chart subcategories. 

**Error Point** : Error Point chart enhances the classic Scatter chart by introducing error bars to each data point, conveying the inherent variability or uncertainty of the data. When classifying Error Point chart, a pivotal consideration lies in the orientation and symmetry of the error bars, leading us to define 5 key characteristics that govern their taxonomy. These characteristics—symmetry and asymmetry in both horizontal and vertical orientations, coupled with a composite category encompassing both directions—culminate into 5 comprehensive subcategories, as illustrated in Figure 10. Here we contend the following delineations: 

1. Vertical-Horizontal Orientation: Distinguishing the direction of error bars, which can profoundly affect the interpretation of the data. 

2. Symmetry-Asymmetry: Acknowledging whether error bars exhibit a mirrored consistency or an uneven distribution across data points. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0018-12.png)


Figure 10: Examples of Error Point chart subcategories. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0019-00.png)


   - **Line** : Line chart is classified based on three primary attributes: data grouping, error visualization methods, and dot-line characteristics for grouped data, as shown in Figure 11. 

1. Data grouping: Single (individual dataset) or Grouped (multiple datasets). 

2. Error visualization methods: Base (no error), Striped-Error (striped fill patterns), and Marker-Error (markers above and below data points). 

3. Dot-Line Characteristics: For single dataset, No-Marker (data points without markers), Marker (data points with markers). For grouped datasets, Diff-Color (different colors for each group), Diff-Marker (different marker shapes for each group), and Diff-Line (different line styles for each group). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0019-05.png)


Figure 11: Examples of Line chart subcategories. 

**Violin** : Violin chart is a combination of the Box chart and kernel Density chart. It provides a deep insight into the distribution of the data, indicating where individual data points fall within the overall data range. Based on the number of data groups and the shape and distribution of the violin form, the taxonomy is divided into 3 subcategories, as shown in Figure 12. 

1. Base (standard shape, single data group) 

2. Grouped-Symmetrical (standard shape, multiple data groups) 

3. Grouped-Departed (half shapes joined, multiple data groups) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0019-11.png)


Figure 12: Examples of Violin chart subcategories. 

**Radar** : Radar chart, also known as spider chart or star plot, is a graphical method of displaying multivariate data on a two-dimensional plane. It is particularly useful for showing performance metrics or skill assessments across multiple areas. Radar chart is classified based on three primary attributes: the number of data grouping, area fill, and dot-line characteristics for grouped data, as shown in Figure 13. 

1. Data Grouping: Base (a single dataset) or Grouped (multiple datasets). 

2. Area Fill: FillArea (areas within the radar chart filled) or NoFillArea (areas without fill to emphasize the outline). 

3. Dot-Line Characteristics: For single dataset, NoMarker (dots without markers), Marker (dots with markers). For grouped datasets, Diff-Color (different colors for each group), Diff-Line (different line styles for each group), and Diff-Marker (different marker shapes for each group). 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0020-00.png)


Figure 13: Examples of Radar chart subcategories. 

**Pie** : Pie chart is a circular statistical graphic that divides a circle into slices to illustrate numerical proportion, whereas ring charts, also known as donut charts, utilize a hollow circle to serve a similar purpose. The presence of a ring, the number of layers, and the highlighted segment are the primary attributes in the classification of pie and ring charts, as shown in Figure 14. 

1. Hollowness: Pie (no ring) or Ring (with a ring). 

2. Layering: SingleLayer (single data series) or MultiLayer (multiple series or categories). 3. Highlighting: Base (without highlighted segments) or Explode (with one or more segments emphasized to capture viewer attention). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0020-05.png)


Figure 14: Examples of Pie and Ring chart subcategories. 

**Density** : Density chart conveys the concentration and distribution of data within a space, often used to depict the magnitude or frequency across different areas or intervals. The orientation and data grouping are the primary attributes of the taxonomy of density plots, as shown in Figure 15. 

1. Orientation: Vertical (y-axis as density) or Horizontal (x-axis as density). 

2. Data Grouping: Base (single dataset) or Grouped (multiple datasets). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0020-10.png)


Figure 15: Examples of Density chart subcategories. 

**Graph** : Graph chart commonly represents relationships and interconnected data through nodes (also known as vertices) and edges. It is widely used to depict networks, pathways, and complex inter-dependencies. The taxonomy of graph charts is based on the directionality and weight of the edges, resulting in 4 subcategories, as shown in Figure 16. 

1. Directionality: Directed (edges have direction) or Undirected (edges have no direction). 

2. Weight: Weighted (edges have weight) or Unweighted (edges have no weight). 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0021-00.png)


Figure 16: Examples of Graph chart subcategories. 

**Quiver** : Quiver chart, also known as vector field plot, is used to display the magnitude and direction of vectors across a two-dimensional plane. This type of chart is particularly useful in physics and engineering to represent velocity fields, gradients, or other vector-based data spatially. The taxonomy of quiver charts is based on the vector quantity and data grouping, resulting in 4 subcategories, as shown in Figure 17. 

1. Vector Quantity: Simple (limited vectors) or Field (vector field). 

2. Data Groups: Single (single group) or Grouped (multiple groups). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0021-05.png)


Figure 17: Examples of Quiver chart subcategories. 

**Contour** : Contour chart, also called contour map or isoline graph, is used to represent three-dimensional data on a two-dimensional plane by plotting contour lines that connect points of equal value. This method is especially useful in fields like meteorology and geography, where it visually communicates variations in terrain elevation or changes in meteorological elements like temperature and pressure. The taxonomy of Contour chart is based on the representation of the contour lines, resulting in 3 subcategories, as shown in Figure 18. 

1. Line (line representation) 

2. Fill-Area (color-filled representation) 

3. Combination (both line and color-filled) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0021-11.png)


Figure 18: Examples of Contour chart subcategories. 

**Histogram** : Histogram chart, often referred as hist chart, is a representation of data distribution where the data is grouped into ranges or “bins” and illustrated as bars to show the frequency of data points within each bin. It is particularly useful for identifying patterns or anomalies in the data, such as skewness, peaks, or gaps in the distribution. The taxonomy of histogram charts is based on the data grouping and positioning, resulting in 3 subcategories, as shown in Figure 19. 

1. Base (single dataset) 

2. Overlaid (overlapping multiple datasets) 

3. Stacked (stacked multiple datasets) 

21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0022-00.png)


Figure 19: Examples of Histogram chart subcategories. 

**Treemap** : Treemap displays hierarchical data through nested rectangles, where each branch of the tree is represented by a rectangle that contains smaller rectangles corresponding to sub-branches. This method allows for efficient use of space, enabling the viewer to quickly compare sizes and proportions within the hierarchy, and is especially useful for analyzing large datasets to reveal relationships and patterns. The taxonomy of treemap charts is based on the compactness and edge presence, resulting in 4 subcategories, as shown in Figure 20. 

1. Tight-Edge (compact with border) 

2. Tight-NoEdge (compact without border) 

3. Loose-Edge (loose with border) 

4. Loose-NoEdge (loose without border) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0022-07.png)


Figure 20: Examples of Treemap chart subcategories. 

**Area** : Area chart is a graphical representation where data points are connected by line segments and the area between the line and the axis is filled with color or patterns, providing a sense of volume. It is particularly useful for visualizing the cumulative magnitude of values over time, allowing for a clear perception of trends and changes in the data series. The taxonomy of area charts is based on the presence of markers, resulting in 2 subcategories as shown in Figure 21. 

1. Base (without markers) 

2. Marker (with markers) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0022-12.png)


Figure 21: Examples of Area chart subcategories. 

**3D charts** : 3D chart extends two-dimensional charting into three dimensions with spatial representations, offering an added layer of depth to represent additional data variables or to enhance visual appeal. Based on the above chart types and the additional surface representation in 3D space, we classify 3D chart into 5 subcategories, as shown in Figure 22. 

1. Scatter (3D scatter chart) 

2. Surface (3D surface chart) 

3. Line (3D line chart) 

4. Bar (3D bar chart) 

5. Density (3D density plot) 

22 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0023-00.png)


Figure 22: Examples of 3D chart subcategories. 

**PIP** : PIP chart insets a magnified or reduced portion of the main plot to highlight key data features, as in Figs. 23 (1) and (2). They enhance readability for complex datasets and facilitate comparative analyses, as demonstrated in Figs. 23 (3) and (4). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0023-03.png)


Figure 23: Examples of PIP charts. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0023-05.png)


**Multidiff** : A Multidiff chart combines at least two different chart types across multiple subplots, with each subplot presenting one type. Derived from the categories above and using diverse layouts, Multidiff charts offer numerous configurations, as shown in Fig. 24. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0023-07.png)


Figure 24: Examples of Multidiff charts. 

**Combination** : Combination chart merges features from different chart types into one plot, offering multilayered presentation. Unlike Multidiff chart with multiple subplots for different categories, Combination chart displays multiple categories in a single plot. As in Fig. 25 (2), a scatter plot illustrates the data distribution while adjacent density plots detail the axis-specific spread. Additional examples are shown in Fig. 25. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0023-10.png)


Figure 25: Examples of Combination charts. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0023-12.png)


**HR** : An HR chart is one that defies the above 21 categories or is challenging to identify. HR chart is typically modified from common charts with distinctive features like custom visual arrangement or atypical markers, as shown in Fig.26. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0023-14.png)


Figure 26: Examples of HR charts. 

23 

## **C Details of Evaluation Metrics** 

In this section, we present the details of evaluation metrics, including GPT-4V score for high-level metrics and Text Score, Layout Score, Type Score and Color Score for low-level metrics. For more information, please refer to our source code. 

## **C.1 GPT-4V Score** 

We employ GPT-4V [2] to assess the extend to which the generated figure corresponds to the groundtruth figure. The specific content of the prompt is presented in Fig. 27. Specifically, we input both the generated and the ground-truth figures into the GPT-4V simultaneously. Then, GPT-4V is instructed to evaluate the similarity between the two figures, taking into account six dimensions: text, layout, type, data, style, and clarity. Subsequently, GPT-4V outputs a score ranging from 0 to 100 to represent the degree of similarity between the figures. 

## Prompt for GPT-4V Score 

You are an excellent judge at evaluating visualization chart plots. The first image (reference image) is created using ground truth matplotlib code, and the second image (AI-generated image) is created using matplotlib code generated by an AI assistant. Your task is to score how well the AI-generated plot matches the ground truth plot. 

## ### Scoring Methodology: 

The AI-generated image’s score is based on the following criteria, totaling a score out of 100 points: 

1. Chart Types (20 points): Does the AI-generated image include all chart types present in the reference image (e.g., line charts, bar charts, etc.)? 

2. Layout (10 points): Does the arrangement of subplots in the AI-generated image match the reference image (e.g., number of rows and columns)? 

3. Text Content (20 points): Does the AI-generated image include all text from the reference image (e.g., titles, annotations, axis labels), excluding axis tick labels? 

4. Data (20 points): How accurately do the data trends in the AI-generated image resemble those in the original image and is the number of data groups the same as in the reference image? 

5. Style (20 points): Does the AI-generated image match the original in terms of colors (line colors, fill colors, etc.), marker types (point shapes, line styles, etc.), legends, grids, and other stylistic details? 

6. Clarity (10 points): Is the AI-generated image clear and free of overlapping elements? ### Evaluation: 

Compare the two images head to head and provide a detailed assessment. Use the following format for your response: 

- Comments: - Chart Types: ${your comment and subscore} - Layout: ${your comment and subscore} - Text Content: ${your comment and subscore} - Data: ${your comment and subscore} - Style: ${your comment and subscore} - Clarity: ${your comment and subscore} Score: ${your final score out of 100} — Please use the above format to ensure the evaluation is clear and comprehensive. 

Figure 27: Prompt for GPT-4V Score. 

24 

## **C.2 Text Score** 

Listing 1: An exemplary Python code for logging text information. 

```
frommatplotlib.backends.backend_pdfimportRendererPdf
drawed_texts=[]
deflog_function(func):
defwrapper(*args,**kwargs):
globaldrawed_texts
text_string=args[4]
drawed_texts.append(text_string)
returnfunc(*args,**kwargs)
returnwrapper
RendererPdf.draw_text=log_function(RendererPdf.draw_text)
```

In order to accurately capture the textual content presented in the rendered figures, our code tracer monitors the function used to add text to the rendered PDF, logging each textual element. An exemplary Python code is provided in Listing 1, where we record the text elements by adding a log wrapper to the “ `draw_text()` ” function in the matplotlib package. Subsequently, we employ this approach to extract the text information from both the generated code and the ground-truth code. The F1 score, computed between these two groups of texts, is denoted as Text Score. 

## **C.3 Layout Score** 

Listing 2: An exemplary Python code for logging layout information. 

```
defget_gridspec_layout_info(fig):
"""
Getthelayoutinformationofagivenfigure.
Args:
fig(matplotlib.figure.Figure):Thefiguretoextractlayoutinformation
from.
Returns:
layout_info(list):Alistofdictionaries,eachcontaininglayout
informationforeachsubplotinthefigure.
"""
layout_info={}
foraxinfig.axes:
spec=ax.get_subplotspec()
ifspecisNone:
continue
gs=spec.get_gridspec()
nrows,ncols=gs.get_geometry()
row_start,row_end=spec.rowspan.start,spec.rowspan.stop-1
col_start,col_end=spec.colspan.start,spec.colspan.stop-1
layout_info[ax]=dict(nrows=nrows,ncols=ncols,row_start=row_start,
row_end=row_end,col_start=col_start,col_end=col_end)
layout_info=list(layout_info.values())
returnlayout_info
layout_info=get_gridspec_layout_info(fig=plt.gcf())
```

The layout refers to the organization of subplots within a chart figure. In each figure implemented using matplotlib, multiple axes objects are present, each containing layout information that specifies 

25 

its position within the figure. To analyze the layout, we iterate through each axis object in the figure and obtain their respective layout information. An exemplary Python code to accomplish this process is provided in Listing 2, where the position information of each axis is recorded. Subsequently, we gather the layout information from both the ground-truth code and the generated code, and compute the F1 score between the them. This metric is referred as the Layout Score. 

## **C.4 Type Score** 

Listing 3: An exemplary Python code for logging type information. 

```
frommatplotlib.axesimportAxes
importinspect
called_functions={}
deflog_function(func):
defwrapper(*args,**kwargs):
file_name=inspect.getfile(func)
name=file_name+"/"+func.__name__
called_functions[name]=called_functions.get(name,0)+1
result=func(*args,**kwargs)
returnfunc(*args,**kwargs)
returnwrapper
Axes.bar=log_function(Axes.bar)
```

The matplotlib package provides a variety of functions for easily generating diverse chart types, such as using “ `axes.bar()` ” to create bar charts. By monitoring the invocations of these functions, we can identify the types of charts being utilized. A successful invocation of a plot functions indicates the incorporation of a specific chart type in the final visualization. Listing 3 presents an exemplary Python code that demonstrates this approach, where we introduce a logger to the “ `bar()` ” function. Subsequently, we gather the chart types from both the generated and ground-truth code, and compute the F1 score between them. We refer to this metric as Type Score. 

## **C.5 Color Score** 

Listing 4: An exemplary Python code for logging color information. 

```
frommatplotlib.axesimportAxes
importinspect
drawed_colors=[]
deflog_function(func):
defwrapper(*args,**kwargs):
func_name=inspect.getfile(func)+"/"+func.__name__
result=func(*args,**kwargs)
foriteminresult:
color=item.get_facecolor()
drawed_colors.append(func_name+"--"+color)
Axes.bar=log_function(Axes.bar)
```

In the matplotlib package, each plot function returns a chart type instance at the end of the function invocation. These instances contain various attributes, including those related to color properties, such as facecolor, edgecolor and colormap. To assess the color attributes, we employ the code tracer that captures the color information of each chart type instance. An example Python code demonstrating 

26 

this approach is provided in Listing 4. Once we obtain the color attributes from both the ground-truth code and the generated code, we compare them using the F1 score metric. 

It is noteworthy that through preliminary experiments, we find that using exact color matching resulted in very low color similarity, as even slight variations would result in a similarity score of zero. To address this issue, we employ the CIEDE2000 color difference formula[51], which converts the matching value between two colors from a discrete [0,1] scale to a continuous range between 0 and 1. Finally, we calculate the maximum color similarity between the sets of colors in the ground-truth code and the generated code to compute F1 score. We refer to the F1 score as Color Score. 

27 

## **D Generation Configurations and Prompting Methods** 

## **D.1 Generation Configurations** 

Following previous setups [52, 53], for open-weight models, we set the temperature _τ_ = 0 _._ 1 to achieve optimal results, while for proprietary models, we set the temperature _τ_ = 0 for greedy decoding. For all models, we set the maximum generation length to 4096. Additionally, we use BF16 for model inference for open-weight models. All models are inferred on a single A100 80G GPU. 

## **D.2 Prompts** 

We provide prompts for Direct, HintEnhanced, SelfRevision and Scaffold Prompting in Figs. 28 to 32. for Direct Prompting, we meticulously design a separate prompt for open-source models to achieve optimal results, as shown inFig. 28. while the prompt for proprietary Models is shown in Fig. 29. 

## Prompt for Direct Prompting (Open-Weight Models) 

You are an expert Python developer who specializes in writing matplotlib code based on a given picture. I found a very nice picture in a STEM paper, but there is no corresponding source code available. I need your help to generate the Python code that can reproduce the picture based on the picture I provide. 

Please note that it is necessary to use figsize=({width} (width), {height} (height)) to set the image size to match the original size. Additionally, I will not provide you with the actual data in the image, so you have to extract the actual data by yourself and based on the extracted data to reproduce the image. Ensure that the code you provide can be executed directly without requiring me to add additional variables. Now, please give me the matplotlib code that reproduces the picture below. 

Figure 28: Prompt for Direct Prompting (Open-Weight Models). {text} in blue font represents placeholders, which varies according to different test examples. 

## Prompt for Direct Prompting (Proprietary Models) 

You are an expert Python developer who specializes in writing matplotlib code based on a given picture. I found a very nice picture in a STEM paper, but there is no corresponding source code available. I need your help to generate the Python code that can reproduce the picture based on the picture I provide. 

Note that it is necessary to use figsize=({width} (width), {height} (height)) to set the image size to match the original size. Now, please give me the matplotlib code that reproduces the picture below. 

Figure 29: Prompt for Direct Prompting (Proprietary Models). {text} in blue font represents placeholders, which varies according to different test examples. 

28 

Prompt for HintEnhanced Prompting 

You are an expert Python developer who specializes in writing matplotlib code based on a given picture. I found a very nice picture in a STEM paper, but there is no corresponding source code available. I need your help to generate the Python code that can reproduce the picture based on the picture I provide. 

To ensure accuracy and detail in your recreation, begin with a comprehensive analysis of the figure to develop an elaborate caption. This caption should cover, but not be limited to, the following aspects: 

1. Layout Analysis: e.g., identify the picture’s composition, noting the presence and arrangement of any subplots. 2. Chart Type Identification: e.g., determine how many charts within a subplot. Are they independent, or do they share a common axis? 

3. Data Analysis: e.g., summarize the data trend or pattern. 

4. Additional Features: e.g.,identify any supplementary elements such as legends, colormaps, tick labels, or text annotations that contribute to the figure’s clarity or aesthetic appeal. 

Now, given the picture below, please first output your comprehensive caption and then use the caption to assist yourself to generate matplotlib code that reproduces the picture. Note that it is necessary to use figsize=({width} (width), {height} (height)) to set the image size to match the original size. 

Figure 30: Prompt for HintEnhanced Prompting. {text} in blue font represents placeholders, which varies according to different test examples. 

## Prompt for SelfRevision Prompting 

You are an expert Python developer who specializes in writing matplotlib code based on a given picture. I have a code for implementing the reference picture as follows: Now, I have the Python matplotlib code for implementing the reference picture as follows: ````` python {python_function} ````` The rendered picture of the code is: Now, please compare whether the renderer picture is the same as the reference picture. The difference may cover, but not be limited to, the following aspects: 1. Chart Types: Does the AI-generated image include all chart types present in the reference image (e.g., line charts, bar charts, etc.)? 2. Layout: Does the arrangement of subplots in the AI-generated image match the reference image (e.g., number of rows and columns)? 3. Text Content: Does the AI-generated image include all text from the reference image (e.g., titles, annotations, axis labels), excluding axis tick labels? 4. Data: How accurately do the data trends in the AI-generated image resemble those in the original image and is the number of data groups the same as in the reference image? 5. Style: Does the AI-generated image match the original in terms of colors (line colors, fill colors, etc.), marker types (point shapes, line styles, etc.), legends, grids, and other stylistic details? 6. Clarity: Is the AI-generated image clear and free of overlapping elements? - If the generated picture matches the reference, please output the original implementation code. - If there are discrepancies, first list the specific differences between the two pictures. Then, modify the existing code to address these differences, ensuring the revised code is capable of reproducing the reference picture. Finally, output the revised code. Note that it is necessary to use figsize=({width} (width), {height} (height)) to set the image size to match the original size. 

Figure 31: Prompt for SelfRevision Prompting. {text} in blue font represents placeholders, which varies according to different test examples. 

29 

Prompt for Scaffold Prompting You are an expert Python developer who specializes in writing matplotlib code based on a given picture. I found a very nice picture in a STEM paper, but there is no corresponding source code available. I need your help to generate the Python code that can reproduce the picture based on the picture I provide. I will provide you with two images. The first image is the original picture. The second image is the picture overlaid with a dot matrix of a shape of {dot_matrix_height} * {dot_matrix_width} to help you with your task, and each dot is labeled with two-dimensional coordinates (x,y). Within each column, the x-coordinate increases from top to bottom, and within each row, the y-coordinate increases from left to right. Please first use this dot matrix as reference anchors to generate the description of the picture (e.g., between dot A and dot B is something) and generate matplotlib code that reproduces the picture. Note that it is necessary to use figsize=({width} (width), {height} (height)) to set the image size to match the original size. 

Figure 32: Prompt for Scaffold Prompting. {text} in blue font represents placeholders, which varies according to different test examples. 

30 

## **D.3 Error Cases of Different Prompting Methods** 

We provide error cases of HintEnhanced, SelfRevision and Scaffold Prompting in Figs. 33 to 35. For an analysis of error cases related to Direct Prompting, please refer to Appendix F. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0031-02.png)


Figure 33: An error case of HintEnhanced Prompting. The **text** highlighted in blue is GPT-4V’s incorrect understanding regarding provided figures. The remaining part of response is omitted with “......”. 

31 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0032-00.png)


Figure 34: An error case of SelfRevision Prompting. The **text** highlighted in blue is GPT-4V’s incorrect understanding regarding provided figures. The remaining part of response is omitted with “......”. 

32 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0033-00.png)


Figure 35: An error case of Scaffold Prompting. The **text** highlighted in blue is GPT-4V’s incorrect understanding regarding provided figures. The remaining part of response is omitted with “......”. 

33 

## **E Correlation with Human Evaluation** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0034-01.png)


Figure 36: A screenshot of the human evaluation questionnaire. 

In order to evaluate the reliability of the proposed automatic metrics, we calculate their correlation with human evaluations. Specifically, a selection of 300 test examples from the Direct Mimic task is utilized. Subsequently, we gather the results generated by GPT-4V using four different prompting methods (Sec. 4.2) on these test examples. This process yields 1 _,_ 200 figures to be assessed. 

Each figure is independently assessed by three evaluators through a questionnaire, who assign scores on a scale of 0 to 100 based on the similarity between the generated figures and the ground-truth figures. The final score for each figure is calculated as the mean of scores given by the three evaluators. Our evaluators comprise volunteer graduate students holding bachelor’s degrees in computer science. To facilitate their assessments, we provide a scoring rubric in the questionnaire, which closely aligns with the low-level and high-level metric evaluations. The scoring criteria encompass the following dimensions: chart type, layout, textual content, data, style, and clarity. A screenshot of the final questionnaire is available in Fig. 36. 

Upon collecting the human evaluation results, we calculate the Pearson correlation coefficient to ascertain the correlation between our automatic metrics and human evaluation. 

34 

## **F Cases of Error Analysis** 

We provide cases of text-related, type-related and color-related errors in Figs. 37 to 42. These cases encompass various error types mentioned in Sec. 4.4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0035-02.png)


**----- Start of picture text -----**<br>
8 Calibration curve<br>7<br>0.8<br>1.00 Data points 6<br>Bin edges<br>Calibration curve<br>0.8 5 0.6<br>0.75<br>4<br>0.6<br>0.4<br>0.50 3<br>0.4<br>2<br>0.25 0.2<br>0.2 1<br>0.00 0<br>0.0 0.2 0.4 0.6 0.8 1.0<br>0.00 0.25 0.50 0.75 1.00 Predicted probability<br>Predicted probability<br>(a) Ground-truth figure (Combination_3) (b) Generated figure<br>Color intensity<br>Fraction of positives<br>Fraction of positives<br>**----- End of picture text -----**<br>


Figure 37: Error Case 1. In this case, the errors include text-related errors of the _Missing_ type and type-related errors of the _Missing_ type. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0035-04.png)


**----- Start of picture text -----**<br>
1.00<br>0.83<br>0.75 0 Probing FGT Feature Embedding Distance-3.67 -3.51<br>0.6 10 Observed FGT-9.44 -9.12 0.8<br>0.50<br>20 0.7<br>0.25 30 0.6<br>0.11<br>-37.90<br>0 0.00 40 0.5<br>-3.51 -3.67<br>-9.44 -9.12<br>25 50 0.4<br>-37.9 60 0.3<br>50<br>70<br>0.2<br>75 Probing FGT 80<br>Observed FGT -84.30<br>-84.3 Feature Embedding Distance 0.1<br>100 SEQ REPLAY MTL SEQ CategoriesREPLAY MTL<br>Categories<br>(a) Ground-truth figure (bar_7) (b) Generated figure<br>Accuracy (%)<br>Accuracy (%)<br>Feature Embedding Distance<br>Feature Embedding Distance<br>**----- End of picture text -----**<br>


Figure 38: Error Case 2. In this case, the errors include text-related errors of the _Missing_ type, type-related errors of the _Extraneous_ type, and color-related errors of the _Different_ type. 

35 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0036-00.png)


**----- Start of picture text -----**<br>
54 3.5 Model Performance Improvement by Language4.4 LanguageEN 4 3.5 Model Performance Improvement by Language 4.4 LanguageEN<br>32 2.3 1.3 DEFRIT 32 2.3<br>1 ES 1.3<br>0 1<br>1 -0.7 0<br>2 -0.7<br>0 1 2 3 4<br>3.4<br>4 3.4 2<br>2<br>0 0.3 0 0.3<br>24 -1.0 -3.3 -2.7 2 -1.0 -3.3 -2.7<br>Model with Language Model with Language<br>(a) Ground-truth figure (bar_95) (b) Generated figure<br>Bactrian-X EN Lima-X DE Bactrian-X FR Lima-X IT Bactrian-X ES BactrianX_EN LimaxX_DE BactrianX_FR LimaxX_IT BactrianX_ES<br>Improvement [%]<br>Improvement [%] Improvement [%]<br>**----- End of picture text -----**<br>


Figure 39: Error Case 3. In this case, the errors include text-related errors of the _Detail_ type and color-related errors of the _Different_ type. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0036-02.png)


**----- Start of picture text -----**<br>
Endurance Race Performance Analysis<br>Endurance Race Performance Analysis<br>Race interrupted due to weather<br>250<br>Athlete Jim 200<br>150<br>Athlete Bill 100<br>Running<br>Interrupted 50 Running<br>Rest Interrupted<br>Running2 Rest<br>0<br>0 25 50 75 100 125 150 175 200 Athlete Jim Athlete Bill<br>Seconds Since Start Seconds Since Start<br>(a) Ground-truth figure (HR_21) (b) Generated figure<br>**----- End of picture text -----**<br>


Figure 40: Error Case 4. In this case, the errors include text-related errors of the _Missing_ type, type-related errors of the _Missing_ type, and color-related errors of the _Different_ type. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0036-04.png)


**----- Start of picture text -----**<br>
Comparative Distribution of Three Different Groups<br>Comparative Distribution of Three Different Groups<br>8<br>4<br>6<br>4 2<br>2<br>0 0<br>2<br>2<br>4<br>6<br>4<br>Control Group Treatment Group 1 Treatment Group 2 Control Group Treatment Group 1 Treatment Group 2<br>Number of Observations<br>(a) Ground-truth figure (HR_22) (b) Generated figure<br>Value Range Value Range<br>**----- End of picture text -----**<br>


Figure 41: Error Case 5. In this case, the errors include text-related errors of the _Missing_ type, type-related errors of the _Confusion_ type, and color-related errors of the _Different_ type. 

36 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation_images/iclr_2025_ChartMimic_Evaluating_LMM's_Cross-Modal_Reasoning_Capability_via_Chart-to-Code_Generation.pdf-0037-00.png)


**----- Start of picture text -----**<br>
Distribution of Grades: Class 1 100 Distribution of Grades: Class 2 120 Distribution of Grades: Class 1 120 Distribution of Grades: Class 2<br>100<br>90 9080 100 100<br>80 70 80 80<br>70 6050 60 60<br>60 40 40 40<br>50 30<br>Boys Girls 20 Boys Girls 20 Boys Girls 20 Boys Girls<br>120 Distribution of Grades: Class 3 Distribution of Grades: Class 4 120 Distribution of Grades: Class 3 120 Distribution of Grades: Class 4<br>110<br>110100 100 100 100<br>90 90 80 80<br>80 80<br>70 70 60 60<br>6050 60 40 40<br>40 Boys Girls 50 Boys Girls 20 Boys Girls 20 Boys Girls<br>(a) Ground-truth figure (violin_9) (b) Generated figure<br>**----- End of picture text -----**<br>


Figure 42: Error Case 6. In this case, the errors include type-related errors of the _Confusion_ type and color-related errors of the _Similar_ type. 

## **G Ethics and Societal Impact** 

We envision ChartMimic as a comprehensive suite of benchmarks designed to guide researchers in assessing the capabilities of their LMMs. By providing an extensive evaluation framework, ChartMimic aims to foster the development of foundation models for the community, offering insights into various aspects of LMMs, such as visual understanding, code generation, and crossmodal reasoning. 

When constructing prototype charts, we use figures from papers on arXiv, which are licensed under CC BY 4.0. During the manual annotation of the code, we implement strict ethical controls to ensure that the information conveyed in the code does not contain any biased content and does not infringe on any private data. 

37 

