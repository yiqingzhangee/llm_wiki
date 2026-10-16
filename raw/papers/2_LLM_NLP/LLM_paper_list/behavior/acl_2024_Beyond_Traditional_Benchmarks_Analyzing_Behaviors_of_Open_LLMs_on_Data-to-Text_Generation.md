# **Beyond Traditional Benchmarks: Analyzing Behaviors of Open LLMs on Data-to-Text Generation** 

**Zdenˇek Kasner** and **Ondˇrej Dušek** Charles University, Faculty of Mathematics and Physics Institute of Formal and Applied Linguistics Prague, Czech Republic {kasner,odusek}@ufal.mff.cuni.cz 

## **Abstract** 

We analyze the behaviors of open large language models (LLMs) on the task of data-totext (D2T) generation, i.e., generating coherent and relevant text from structured data. To avoid the issue of LLM training data contamination with standard benchmarks, we design QUINTD – a tool for collecting novel structured data records from public APIs. We find that open LLMs (Llama 2, Mistral, and Zephyr) can generate fluent and coherent texts in zero-shot settings from data in common formats collected with QUINTD. However, we show that the semantic accuracy of the outputs is a major issue: both according to human annotators and our reference-free metric based on GPT-4, more than 80% of the outputs of open LLMs contain at least one semantic error. We publicly release the code, data, and model outputs.[1] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation_images/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation.pdf-0001-05.png)


**----- Start of picture text -----**<br>
1 structured data from public APIs<br>JSON JSON JSON CSV MD<br>weather product game time entity<br>data data results series graph<br>LLM<br>weather product game chart entity<br>forecast description report caption description<br>2 token-level error annotations<br>**----- End of picture text -----**<br>


Figure 1: To benchmark LLMs, we download unlabeled structured data from public APIs and prompt LLMs to generate texts based on the data. We annotate semantic errors in the outputs using reference-free metrics. 

## **1 Introduction** 

Large language models (LLMs; Ouyang et al., 2022; Touvron et al., 2023a,b; Jiang et al., 2023; Tunstall et al., 2023) have already left a mark in many areas of natural language processing (NLP). Surprisingly, their applicability to the task of data-to-text (D2T) generation (Reiter and Dale, 1997; Gatt and Krahmer, 2018) remains underexplored, with limited evaluation on a handful of well-established benchmarks only (Axelsson and Skantze, 2023; Yuan and Färber, 2023). Generating text from structured data is arguably challenging for LLMs, given the specifics of D2T generation, such as long inputs, complex non-linear structure, and strict requirements on semantic accuracy. However, a more significant issue is the lack of testing grounds. The current D2T generation benchmarks are not only getting saturated (Van Miltenburg et al., 2023), but also promote optimization towards traditional reference-based evaluation metrics, which 

> 1https://d2t-llm.github.io/ 

were shown to correlate poorly with human judgment (Gehrmann et al., 2023; van der Lee et al., 2021; Novikova et al., 2017). When it comes to the models, using closed LLMs (OpenAI, 2023a,b) is increasingly considered a bad research practice due to its non-reproducibility (Rogers, 2023; Chen et al., 2023). On top of that, contamination of LLM training data with standard benchmarks further restricts the space for experiments (Golchin and Surdeanu, 2023; Aiyappa et al., 2023; Balloccu et al., 2024). 

In this paper, we propose an approach that allows us to analyze model behavior in D2T generation on novel, real-world structured data records with reference-free evaluation metrics. We begin by realizing that _unlabeled data are plentiful_ . To leverage the data for our experiments, we introduce QUINTD[2] – a tool for collecting structured data from five domains in standard formats: JSON, 

> 2 Quintet of Unlabeled Inputs for Natural Tasks in Datato-text, pronounced as “quintet” 

|**Task Id**|**Domain**|**Task Description**|**Source**|**Format**|
|---|---|---|---|---|
|openweather|Weather|Generating a weather forecast from weather data.|OpenWeather|JSON|
|gsmarena|Technology|Describing a product based on its attributes.|GSMArena|JSON|
|ice_hockey|Sport|Describing an outcome of an ice-hockey game.|RapidAPI|JSON|
|owid|Health|Generating a caption for a time series.|OurWorldInData|CSV|
|wikidata|World facts|Describing entities and relations in a knowledge graph.|Wikidata|Markdown|



Table 1: The domains and tasks included in the QUINTD data collection tool we use for testing D2T generation with LLMs. In our experiments, we download 100 development and 100 test examples of input data for each domain. 

CSV, and Markdown. We choose the domains so that the data can be directly used as input for five distinct D2T generation tasks. Our tasks include generating weather forecasts, sports reports, product descriptions, chart captions, and entity descriptions (see Table 1). Next, we collect a set of 1,000 inputs with QUINTD and use the inputs as an adhoc benchmark (called QUINTD-1) for testing the abilities of LLMs for D2T generation. We assume that the data formats in QUINTD-1 are common in the LLMs’ pretraining corpora, so we specify the task using instructions instead of standard finetuning with human-written outputs, capitalizing on the zero-shot abilities of instruction-tuned LLMs (§2). 

We push towards better reproducibility by _focusing on open LLMs_ , which – apart from being more accessible – also achieve increasingly better results across tasks (Zheng et al., 2023; Beeching et al., 2023). For our experiments, we use three open LLMs with 7B parameters: Llama 2 (Touvron et al., 2023b; TogetherAI, 2023), Mistral (Jiang et al., 2023), and Zephyr (Tunstall et al., 2023). We also use GPT-3.5 (OpenAI, 2023b) as a closed model baseline for the final experiments. Given the behavioral nature of the experiments with LLMs (Holtzman et al., 2023), we put emphasis on reporting model behavior throughout the process (§3). 

Another piece of the puzzle is _reference-free evaluation_ : using the input data as a ground for comparison instead of reference outputs (§4). We focus on identifying semantic errors in the model outputs, i.e., the information that is not supported by the input data. We use two separate evaluation methods: manual annotations from human crowdworkers (van der Lee et al., 2021) and a custom automatic metric based on GPT-4 (Liu et al., 2023; Chiang and Lee, 2023; Kocmi and Federmann, 2023a). We annotate the errors on the level of individual words, getting fine-grained annotations of error spans in several categories (Thomson and Reiter, 2020; Thomson et al., 2023). 

Based on our results, we provide general recommendations for D2T generation with open LLMs across tasks and formats (§5). Our main findings are as follows: 

- **Open LLMs can generate fluent outputs from structured data** in common formats under zero-shot settings. 

- **Semantic accuracy is a major obstacle** : both human annotators and GPT-4-based metric report that over 80% of outputs of open LLMs on our data contain a semantic error. 

- **Long data inputs cause practical issues** , including the need for long-context models, increased GPU memory requirements, and unavailability of few-shot approaches. 

- **Outputs can be empirically improved by following several rules-of-thumb** for preprocessing the model input, such as including units, removing unnecessary fields, or prefixing the model answer. 

## **2 Reference-Free D2T Generation** 

## **2.1 Data Collection Tool** 

We introduce QUINTD,[3] a tool for collecting adhoc test sets using public APIs in five different domains. Our main reasons for departing from the traditional scheme of benchmarking on wellestablished datasets are: 

1. Any published test sets may be potentially included in the training data of LLMs. 

2. Public sources of structured data offer enough resources for creating ad-hoc test sets. 

3. Without human references, our data collection scheme is lightweight and replicable. 

Given the available public sources of data, we settled on the five tasks which are described in Table 1 (see Appendix A for more details). The tasks 

3https://github.com/kasnerz/quintd 

## **Prompt** 

Based on the given data: ˋˋˋ 

{data} ˋˋˋ 

Your task is to write a brief, fluent, and coherent single-paragraph {output_type} in natural language. The text should be balanced and neutral. Make sure that all the facts mentioned in the text can be derived from the input data, do *not* add any extra information. 

## **Output prefix** 

Sure! Here is the {output_type}: " 

experiments. With this goal in mind, we decided to use the same prompt template _P_ for all the domains and models. 

For a set of preliminary experiments, we first wrote down the initial version of the prompt and used the data without further preprocessing. We then iteratively improved our experimental setup by observing outputs on the development set. In §3.3, we describe all the observations and modifications we made before generating the final outputs on the test set. 

## **3.2 Models** 

For our experiments, we selected the following LLMs available under an open license: 

Figure 2: The prompt _P_ and the model output prefix we used for the experiments in this paper. data is filled with the data record _x_ and output_type is filled accordingly for each domain _D_ (see Table 1 and Table 6 in the Appendix). 

- **Llama 2** (Touvron et al., 2023a; TogetherAI, 2023), togethercomputer/Llama-2-7B-32K-Instruct 

- **Mistral** (Jiang et al., 2023), mistralai/Mistral-7B-Instruct-v0.1 

are based on structured data in common formats: JSON, CSV, and Markdown. 

- **Zephyr** (Tunstall et al., 2023). HuggingFaceH4/zephyr-7b-beta 

## **2.2 QUINTD-1 Dataset** 

Using QUINTD, we collected the dataset for our experiments in this paper (QUINTD-1). The dataset contains 500 examples in the development set and 500 examples in the test set (100 examples per domain for each split). We downloaded the data between November 2023 and January 2024. Note that the dataset contains only **unlabeled** data without any reference outputs (e.g., weather data, but not a textual weather forecast), so the outputs need to be evaluated using reference-free metrics. New versions of the benchmark can be easily generated with our QUINTD tool. 

## **2.3 Task Definition** 

Each example in QUINTD-1 consists of a structured data record _x_ from a domain _D_ P {openweather, gsmarena, ice_hockey, owid, wikidata}. Given _x_ and a prompt _P_ , the goal is to generate natural language output _y_ faithful to the data _x_ , according to the instructions in the prompt _P_ (see Figure 2). 

## **3 Experiments** 

## **3.1 Experimental Process** 

Our goal is to avoid extensive data preprocessing and prompt engineering since these steps could harm the reproducibility and generalizability of our 

The models are instruction-tuned, operate with 32k context, and perform well on recent benchmarks (Beeching et al., 2023). All the models have 7B parameters and thus fit on a single NVIDIA A40 (48G VRAM) in 16-bit precision. The models are available through HuggingFace (Wolf et al., 2020). 

We accessed the models via an API provided by the text-generation-webui framework[4] running locally. For the final experiments, we also included GPT-3.5 (gpt-3.5-turbo-1106) accessed through the OpenAI API (OpenAI, 2023b).[5] 

## **3.3 Observations from Preliminary Experiments** 

During development, we made several observations which we took into account for our final experimental setup: 

**Any input field may appear in the output.** The models do not always select the most relevant fields for the given output. For example, we observed that 

> 4https://github.com/oobabooga/ text-generation-webui 

> 5We only included GPT-3.5 in our final experiments as proprietary models were not our focus. We did not include GPT-4 since we use the same model for evaluation (see §4.2) and LLMs tend to be biased towards their own outputs (Koo et al., 2023; Stureborg et al., 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation_images/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation.pdf-0004-00.png)


**----- Start of picture text -----**<br>
"weather": [{ Llama2-7B-<br>    "description":  32k-Instruct<br>  "light rain",<br>  }],"wind": { Mistral-7B<br>  "deg": 328, + prompt instruct<br>  "gust": 3.92,<br>  "speed": 2.27} Zephyr-7B-<br>(...) data beta<br>Mountain Brook,  is expected to<br>have a mild and mostly cloudy<br>, with temperatures ranging from<br>4.78°C to 11.47°C. Rain is expected on<br>(...)<br>output<br>GPT-4<br>✔<br>human<br>annotators<br>LLMs<br>data<br>raw open generation<br>visualization<br>annotators evaluation<br>data<br>**----- End of picture text -----**<br>


Figure 3: Our experimental setup. We first generate the outputs using LLMs that are given raw data and a taskspecific prompt. We annotate the word-level semantic errors in the LLM outputs with (a) an automatic metric based on GPT-4 that matches the output to the raw data, and (b) human annotators, who annotate the errors in the output given the data visualization. 

the models commonly mention identifiers, timestamps, files, and other metadata, leading to unnatural outputs. To mitigate these issues, we manually picked irrelevant fields and filtered them out from the input. 

**Units need to be specified explicitly.** If the units are not specified in the data record, the models tend to resort to their best guess. This may go unnoticed if the unit is evident from the context (e.g., the model will usually not report the temperature in Fahrenheit instead of Celsius), but it may get problematic if the value is ambiguous (e.g., wind speed in km/h versus m/s). Therefore, we explicitly add units to all data records where appropriate. 

ever, with long inputs, the GPU memory consumption also gets considerably higher, so we needed to downsample the data in owid and openweather to keep their length under ~8k tokens. 

**Few-shot experiments are infeasible.** Due to the above context-length limitations, we were not able to run few-shot experiments since we could not robustly fit an additional ( _x_ example, _y_ example) pair in the prompt. We attempted to include only _y_ example (making the setup “half-shot”), but we observed that the models then used entities from the example (unrelated to the actual input) in their outputs. Therefore, we decided to leave this line of experiments for future work (see §5.3 for discussion). 

**Understandable field names are enough.** On the flip side, we decided not to add extra descriptions to field names in the data if the field was understandable from its name (e.g., homeTeam or dimensions). As discussed by Kasner et al. (2023), pretrained models interpret field names correctly as long as they are human-readable. We only include chart metadata for the CSV files in the owid domain. 

**Long inputs can be troublesome.** The inputs in some domains can easily get longer than 10-20k tokens. This issue is amplified by the fact that the evaluated LLMs tokenize numbers into individual digits. To accommodate for the long inputs, we picked models that accept up to 32k tokens.[6] How- 

6For this reason, we use Llama-2-7B-32k with 32k token context (TogetherAI, 2023) instead of the official Llama-27B-Instruct, which only supports 4k context (Touvron et al., 

**Deterministic decoding and sampling are on par.** In our preliminary experiments, we observed a roughly similar output quality for both deterministic decoding and sampling.[7] For the final experiments, we decided to use deterministic decoding, which is non-parametric and conceptually more suitable for D2T generation. 

**Prefixing the output makes parsing easier.** Even with variations of a _“generate only the output”_ instruction appended to the prompt, the models (especially Llama 2) tended to first confirm the request. For that reason, we decided to prefix the input for all the models with _“Sure! Here is the "” {output_type}:_ . The opening quote at the end of the prefix allowed us to robustly parse the text sim- 

2023b). 

> 7We used the text-generation-webui default decoding parameters: temperature=0.7, top_p=0.9, and top_k=20. 

ply by stripping the closing quote from the model output. 

**The outputs are fluent but inaccurate.** We observed that the vast majority of model outputs were grammatically and stylistically correct, capturing the output type specified in the prompt. However, we also noticed that the outputs contained many semantic errors (even after emphasizing the focus on semantic accuracy in the prompt, see Figure 2). This observation led us to evaluate the model outputs using word-level annotations focused on semantic accuracy errors (see §4). 

## **3.4 Final Experiments** 

Taking the observations in §3.3 into account, we proceeded to generate the outputs on the test set of QUINTD-1 for word-level error analysis. We first preprocessed the data as mentioned: we stripped out unnecessary fields, added units, and downsampled the data to fit the context. For all the models mentioned in §3.2, we used the prompt in Figure 2 and deterministic decoding with a maximum length of 512 tokens. 

For comparison, we also generated outputs for the same inputs and identical prompts with GPT3.5. Note that even though we fixed the temperature and seed to 0, the rest of the decoding parameters are inaccessible to us and may differ from the parameters we used for the open models. 

## **4 Evaluation** 

For evaluation, we focus on identifying _semantic errors_ in model outputs. We compare the generated texts to the input data, looking for parts of texts that are not faithful to the input data. We annotate the errors on the word level, considering all the words in the output text as potential sources of errors. 

We use two complementary referenceless evaluation methods: 

- _E_ hum: **human evaluation** based on crowdsourcing (§4.1), 

- _E_ : **an automatic metric** based on GPT-4 gpt 

- (§4.2). 

The methods use similar instructions and produce outputs with equivalent format. The main idea is to compensate for the shortcomings of each approach: while human evaluation is costly (about ten times more expensive than automatic evaluation), using only an automatic metric based on a closed LLM would make the evaluation potentially 

non-reproducible and biased (Kocmi and Federmann, 2023a; Wang et al., 2023b). Reporting the results of both methods should hopefully increase the robustness and replicability of our results. 

Our error taxonomy and its notation are inspired by Thomson and Reiter (2020) and Thomson et al. (2023). After preliminary examination of the outputs, we settled on four error categories: **INCORRECTI** , **NOT_CHECKABLENC** , **MISLEADINGM** , and **OTHERO** . To set clear boundaries between the categories, simplify the annotation interface and reach better inter-annotator agreement, we decided to keep our taxonomy more high-level than Thomson and Reiter and not to distinguish between fine-grained categories (e.g., _incorrect name_ vs. _incorrect number_ ). The descriptions of our error categories, as presented in the instructions for annotation, are included in Table 2. 

## **4.1 Human-based Evaluation** 

For the human annotation metric, we prepared a custom web interface, where an annotator is instructed to annotate text spans with respective error categories. We created custom visualizations for each data format (see Figure 3 and Appendix E for examples).[8] 

We hired annotators on the Prolific[9] crowdsourcing platform. In total, we hired 100 annotators, each annotating 20 examples (four model outputs for each of the five domains). We selected annotators with at least 10 completed tasks, a 100% approval rate and English as their primary language. We paid the annotators £9 per hour, according to the platform’s recommendations. The median time for completing the annotations was 47 minutes. See Appendix B for the instructions for the annotators and the annotation interface. 

## **4.2 GPT-4-based Evaluation** 

For automatic evaluation, we leverage the fact that LLM-based metrics can be customized for a particular task without the need for training data. In our experiments, we use a metric based on GPT4 (gpt-4-1106-preview, OpenAI, 2023a), which was shown to be superior to other LLMs in following fine-grained instructions, reaching high correlations with human judgment on evaluating generated 

> 8 We open-sourced our annotation framework as a stand-alone software package, see https://github.com/ kasnerz/factgenie. 

> 9https://prolific.com 

||**Error**|**Description**|
|---|---|---|
||**IN**<br>**COR**<br>**RECT**<br>**I**<br>**NOT_CHECKABLE**<br>**NC**<br>**MIS**<br>**LEAD**<br>**ING**<br>**M**<br>**OTHER**<br>**O**|The fact in the text contradicts the data.<br>The fact in the text cannot be checked given the data.<br>The fact in the text is misleading in the given context.<br>The text is problematic for another reason, e.g., grammatically or stylistically incorrect,<br>irrelevant, or repetitive.|
||**Example**<br>_data_<br>_text_|**Nokia 3310**|_color_: black, blue, grey |_display_: 320x240px<br>Nokia 3310 is<br>**pro**<br>**duced**<br>**in**<br>**Fin**<br>**land**<br>**NC** and features a<br>**320x320**<br>**I** display. It is<br>**avail**<br>**able**<br>**in**<br>**black**<br>**color**<br>**M**.<br>**The**<br>**data**<br>**seem**<br>**to**<br>**pro**<br>**vide**<br>**only**<br>**par**<br>**tial**<br>**in**<br>**for**<br>**ma**<br>**tion**<br>**about**<br>**the**<br>**phone.**<br>**O**|



Table 2: Categories of errors annotated in our evaluation and an example demonstrating the error types. See Appendix B for an explanation of individual errors in the example. 

texts (Zhao et al., 2023; Sottana et al., 2023; Kocmi and Federmann, 2023a,b).[10] 

We instantiate _E_ gpt with a prompt and a system message describing the task. We instruct the model to produce a JSON output with sequentially ordered errors using the following format: 

{ "errors": [{ "reason": [REASON], "text": [TEXT_SPAN], "type": [ERROR_CATEGORY] }, ...] 

- }. 

Note that we require that the model first generates the free-form text _reason_ for the error. Generating the reason comes at almost no extra cost and our cursory observations suggest that requiring it leads to more precise outputs.[11] 

We align the model outputs with the original text by string matching on TEXT_SPAN, moving the current position forward after each match. We ensure that the model response is a valid JSON using OpenAI’s response_format parameter. See Appendix C for more details about the metric, including the prompt and the system message. 

## **5 Results and Discussion** 

A summary of the word-level annotations is in Table 3 and 4, with detailed results per domain provided in Appendix F. 

> 10We confirmed that GPT-3.5 and Llama 3 have lower correlations with human judgments also in our scenario, see Appendix D. 

> 11We did not ask the crowdworkers for free-form reasoning about the errors since that would make the annotation notably more complex. 

## **5.1 How Accurate Are the Model Outputs?** 

Depending on the model, between 76-86% of examples contain an error according to _E_ hum, suggesting that open LLMs make semantic errors very often. According to _E_ gpt, the number is as high as 89-94%. 

**I** The most common error type is **INCORRECT** . As shown in Table 3, all the open LLMs make more than **two statements contradicting the data per output on average** . The **NOT_CHECKABLENC** errors are also relatively common: more than one per output on average according to _E_ hum, and at least one being present in more than 25% of examples according to both metrics. 

The results vary widely according to the domain (see Appendix F). For example, the outputs in wikidata contain much more **NOT_CHECKABLENC** errors on average (1.01 per output according to _E_ hum) than **INCORRECTI** errors (0.11 per output according to _E_ hum), suggesting that with simpler inputs, the models tend to introduce extra information. The openweather domain seems to be the most complex with the longest outputs (~164 tokens), more than eight errors in the output on average, and >90% of outputs containing an error. 

The differences between the open LLMs are not major. Out of the open LLMs, Zephyr has the best results across categories and metrics, followed by Llama 2. However, the outputs of Mistral are longer on average, leaving more space for errors. GPT-3.5 (which we consider separately) does generally better according to both _E_ gpt and _E_ hum, although it still makes an error in 60-75% of examples (2 errors per example on average). In general, the results show that LLMs make too many semantic errors to be usable in practice for D2T generation in a zero-shot setting. 

||**Incorrect**|**Incorrect**|**Not**|**Checkable**|**Misleading**|**Misleading**|**Other**|**Other**|**All categories**|**All categories**||
|---|---|---|---|---|---|---|---|---|---|---|---|
||_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|#**Tok.**|
|**Llama 2**|1.57|**2.79**|1.25|0.91|0.25|**0.12**|**0.10**|0.09|3.18|3.90|83.8|
|**Mistral**|2.03|3.23|1.12|0.54|0.44|0.26|0.25|0.10|3.85|4.12|114.9|
|**Zephyr**|**1.44**|2.84|**0.77**|**0.40**|**0.20**|0.29|0.16|**0.05**|**2.58**|**3.58**|98.0|
|**GPT-3.5**|0.65|1.76|0.49|0.38|0.18|0.26|0.07|0.02|1.39|2.42|84.9|



Table 3: The average _numbers of errors per output_ (lower is better) based on human annotators ( _E_ hum) and GPT-4 ( _E_ gpt). We also include the average number of tokens per output in the rightmost column. The results of the best open LLM are emphasized. 

||**Incorrect**|**Incorrect**|**Not Checkable**|**Not Checkable**|**Misleading**|**Misleading**|**Other**|**Other**|**All categories**|**All categories**|
|---|---|---|---|---|---|---|---|---|---|---|
||_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|
|**Llama 2**|53.2%|80.0%|57.4%|44.8%|17.4%|**8.8%**|**7.6%**|7.6%|85.6%|94.0%|
|**Mistral**|53.6%|80.2%|49.6%|31.8%|20.6%|17.0%|13.6%|8.4%|81.2%|93.0%|
|**Zephyr**|**46.8%**|**78.0%**|**42.2%**|**25.0%**|**16.2%**|20.6%|11.6%|**4.2%**|**75.6%**|**89.4%**|
|**GPT-3.5**|38.0%|65.0%|28.8%|19.6%|12.6%|16.2%|6.2%|2.2%|60.6%|75.8%|



Table 4: The percentage of _outputs containing at least one error_ (lower is better) based on human annotators ( _E_ hum) and GPT-4 ( _E_ gpt). The results of the best open LLM are emphasized. 

## **5.2 Do Evaluation Methods Agree?** 

To quantify the agreement of _E_ hum and _E_ gpt, we computed the Pearson correlation coefficient between the error counts on the level of words, examples, and domains as follows (note that each error category was considered separately): 

- For _r_ domain, we used the average error counts per domain (see Table 13). 

- For _r_ example, we used the count of errors per example. 

- For _r_ word, we used the binary indicators marking an error per word. 

The correlation on the level of words is weak ( _r_ word “ 0 _._ 26) but gets better on the example-level ( _r_ example “ 0 _._ 52) and even better on the domainlevel ( _r_ domain “ 0 _._ 93). In Table 5, we show the percentage of words marked as errors by individual metrics. The metrics agree on the specific words in less than 6%, although they both mark around 21% of words as erroneous. 

We also measure inter-annotator agreement between human annotators. For that, we obtained annotations from two annotators for 100 model outputs. The results are similar: the annotators agree weakly on the word level ( _r_ word “ 0 _._ 36), stronger on the example level ( _r_ example “ 0 _._ 53), and even stronger on the domain level ( _r_ domain “ 0 _._ 85). We conclude that while the details regarding error spans and categories may vary, the annotators as well as GPT-4 generally agree on the accuracy 

||_E_gpt|_E_hum|_E_gpt +_E_hum|
|---|---|---|---|
|**Incorrect**|10.1%|14.2%|4.1%|
|**Not checkable**|7.8%|4.3%|2.0%|
|**Misleading**|2.2%|1.5%|0.1%|
|**Other**|1.8%|0.7%|0.1%|
|**Total**|21.9%|20.7%|6.3%|



Table 5: The percentage of _words marked as erroneous_ by human annotators ( _E_ hum), GPT-4 ( _E_ gpt), and both approaches at the same time ( _E_ hum + _E_ gpt). 

of model outputs for a given set of examples. In the future, the agreement could be improved by measuring errors on the phrase level (Vamvas and Sennrich, 2022). 

## **5.3 Recommendations for Future Work** 

**Focus on semantic accuracy.** The output of LLMs is satisfactory regarding the style, format, and purpose of the text. However, the amount of semantic errors remains very high. Improving the semantic accuracy of the models (Li et al., 2022), along with new model-based evaluation metrics (Liu et al., 2023; Xu et al., 2023), could thus help to bring improve LLM-based D2T generation systems where it is most needed. 

**Use efficient and long-context models.** The memory issues with long context, making few-shot experiments infeasible, can potentially be solved by using more efficient models equipped with Flash Attention (Dao et al., 2022) and fast inference li- 

braries such as llama.cpp[12] . The recent emergence of capable long-context models (Bai et al., 2023; Munkhdalai et al., 2024) also seems to play in favor of LLM-based D2T generation with long inputs. 

**Be careful about subtle bugs.** During our preliminary experiments, we fixed several subtle bugs in our API calls such as incorrect instruction templates[13] or involuntary input truncation. Therefore, we recommend careful checks of API calls, as with the apparent ease of API access and robustness of LLMs, such bugs could go unnoticed and artificially skew the model performance. 

**Test the models in the wild.** Except for using an ad-hoc dataset of real-world data as we did in our work, the ecological validity of D2T evaluation can also be ensured by continuous evaluation with human users (Zheng et al., 2023) and evaluating the real-world impact of the systems (Reiter, 2023). 

**Multilinguality is an opportunity.** With the recent efforts in extending D2T generation to lowresource languages (Cripwell et al., 2023), multilingual D2T generation with open LLMs seems a promising direction. Although we did not go beyond English, initial steps were already done by works such as Lorandi and Belz (2023). 

## **6 Related Work** 

## **6.1 Evaluation of Generated Texts** 

Evaluation of generated texts is a complex task lacking a generally accepted solution (Celikyilmaz et al., 2020). While researchers are acknowledging the importance of combining multiple evaluation metrics (Gehrmann et al., 2021, 2023), most evaluation is still based on comparing the model outputs to human-written references, which tend to be noisy and expensive to collect (Dušek et al., 2019, 2020). 

Many works recently investigated the potential of using LLMs for automatic reference-free evaluation of generated texts, generally achieving high correlations with human judgment (Zhao et al., 2023; Sottana et al., 2023; Kocmi and Federmann, 2023a,b; Chiang and Lee, 2023; Wang et al., 2023a; Fu et al., 2023). However, they also voice concerns about its non-reproducibility (Kocmi and Federmann, 2023a) and potential bias of these models (Wang et al., 2023b). 

> 12https://github.com/ggerganov/llama.cpp 

13https://huggingface.co/docs/transformers/ chat_templating 

Human evaluation is an essential component of natural language generation experiments (van der Lee et al., 2019, 2021). The closest human evaluation protocol to our scenario is the reference-free word-level annotation of complex D2T generation output proposed in Thomson and Reiter (2020) and Thomson et al. (2023). 

## **6.2 D2T Generation Tasks** 

**Weather Forecasts** First attempts for generating weather forecasts include template-based and statistical approaches (Belz, 2005, 2008; Angeli et al., 2010) for the Sumtime-meteo and WeatherGov datasets (Sripada et al., 2002; Liang et al., 2009). More recently, Balakrishnan et al. (2019) introduced a weather forecast dataset with treestructured meaning representations. Our weather forecasts are less structured and based on a 5-day weather outlook. 

**Product Descriptions** Our phone specifications are closest to Wen et al. (2015, 2016), who introduced a dataset for generating descriptions of laptops and TVs. Their solution was based on recurrent neural networks, although templates remained a go-to approach for the task (Wang et al., 2017). Recently, Shao et al. (2021) and Koto et al. (2022) also proposed specialized architectures based on pretrained language models for the data from big e-commerce platforms. 

**Sport Reports** All the D2T generation datasets from the Rotowire family (Wiseman et al., 2017; Wang, 2019), including SportSett:Basketball (Thomson et al., 2021), and ESPN-NBA (Nie et al., 2018) focus on generating basketball reports. Along with MLB (Puduppully et al., 2019b), these datasets belong among the most challenging D2T datasets, attracting various neural-based solutions (Puduppully et al., 2019a, 2022; Puduppully and Lapata, 2021; Rebuffel et al., 2020). We use instead simpler data covering ice hockey game summaries. 

**Chart Captions** Following the early rule-based approaches (Demir et al., 2008, 2012), the approaches for chart captioning recently tackle largescale datasets from data analytic institutions (Obeid and Hoque, 2020; Kantharaj et al., 2022). We focus on one of the tasks from Sharma et al. (2021), which is generating descriptions of time series in the health domain. 

**Entity Descriptions** The task of generating descriptions for a knowledge graph has been covered 

extensively in D2T generation (Gardent et al., 2017; Castro Ferreira et al., 2020; Agarwal et al., 2021; Chen et al., 2020; Ribeiro et al., 2020, _inter alia_ ). Our task is to describe an entity provided a list of its properties, which is closely related to generating entity descriptions from Wikipedia infotables (Lebret et al., 2016). 

## **6.3 D2T Generation with LLMs** 

Recent works have focused on exploring the capabilities of closed LLMs on existing D2T generation datasets. Axelsson and Skantze (2023) evaluated GPT-3.5 (OpenAI, 2023b) on WebNLG, along with Yuan and Färber (2023), who also tested the model on the AGENDA dataset (Koncel-Kedziorski et al., 2019). Both works found that regardless of potential data contamination, the LLMs rank behind state-of-the-art finetuned models on automatic metrics. Zhao et al. (2023) tested closed models on modified table-to-text generation datasets and found out that in terms of faithfulness, GPT-4 can outperform state-of-the-art models. 

## **7 Conclusion** 

We provided an exploratory study into D2T generation with open LLMs. We proposed new directions for D2T generation, including using ad-hoc test sets, data in common formats, and referencefree evaluation. By a combination of GPT-4-based metric and human evaluation, we evaluated the performance of LLMs on five domains, providing word-level annotations of model outputs across five domains and recommendations for future directions in D2T generation. 

## **Acknowledgements** 

This work was funded by the European Union (ERC, NG-NLG, 101039303) and Charles University project SVV 260 698. It used resources of the LINDAT/CLARIAH-CZ Research Infrastructure (Czech Ministry of Education, Youth, and Sports project No. LM2018101). 

## **Limitations** 

In our work, we do not include a comparison to other D2T generation approaches. The main reason is that our benchmark is reference-free, while a large majority of prior approaches are based on models finetuned on reference outputs. However, we believe that our work still satisfies our main 

goal of providing insights into behaviors of open LLM models on D2T generation. 

We acknowledge that reference-free metrics currently have various shortcomings, including reliance on closed models and specific human annotation protocols, leading to limited replicability and a high price of execution. The approaches occasionally produce incorrect outcomes themselves and they have only moderate correlations with each other. We believe that these shortcomings will be addressed in the future with open model-based metrics. 

Our choice of models is limited to 7B-parameter open LLMs due to our limited computational resources. Also, unlike some other LLMs such as GPT-Neo (Black et al., 2022) or BLOOM (BigScience Workshop et al., 2022), the models we used do not disclose the data they were trained on. For this reason, we find it ever more important to test the models on benchmarks whose labels could have _not_ been included in their training data. 

The approaches based on LLMs may produce factually and semantically incorrect information. Any text produced by the LLMs therefore needs to be carefully examined, and no decisions should be based on the generated text alone. Discovering the _causes_ of LLMs’ hallucinations is out of scope of this paper, but is currently a major topic under investigation (Ji et al., 2023). 

## **Ethical Considerations** 

The human evaluation study was approved by the internal ethics committee of our institution. The annotators were hired over Prolific and paid the platform-recommended wage of 9 GBP/hour. The annotators were preselected based on their primary language (English) and their country of residence (US, UK, Ireland, Australia, New Zealand). All annotators were shown detailed instructions and explanation of the data types, data sources, and the purpose of the research (see Appendix B for details). The domains in QUINTD were selected so that they do not contain any sensitive or potentially offensive content. We do not collect any demographic data about the participants. 

## **References** 

Oshin Agarwal, Heming Ge, Siamak Shakeri, and Rami Al-Rfou. 2021. Knowledge Graph Based Synthetic Corpus Generation for Knowledge-Enhanced Language Model Pre-training. In _Proceedings of the_ 

_2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021_ , pages 3554–3565, Online. 

- Rachith Aiyappa, Jisun An, Haewoon Kwak, and Yongyeol Ahn. 2023. Can we trust the evaluation on ChatGPT? In _Proceedings of the 3rd Workshop on Trustworthy Natural Language Processing (TrustNLP 2023)_ , pages 47–54, Toronto, Canada. 

- Gabor Angeli, Percy Liang, and Dan Klein. 2010. A Simple Domain-Independent Probabilistic Approach to Generation. In _Proceedings of the 2010 Conference on Empirical Methods in Natural Language Processing, EMNLP 2010, 9-11 October 2010, MIT Stata Center, Massachusetts, USA, A meeting of SIGDAT, a Special Interest Group of the ACL_ , pages 502–512. 

- Agnes Axelsson and Gabriel Skantze. 2023. Using Large Language Models for Zero-Shot Natural Language Generation from Knowledge Graphs. _CoRR_ , abs/2307.07312. 

- Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2023. LongBench: A bilingual, multitask benchmark for long context understanding. _CoRR_ , abs/2308.14508. 

- Anusha Balakrishnan, Jinfeng Rao, Kartikeya Upasani, Michael White, and Rajen Subba. 2019. Constrained Decoding for Neural NLG from Compositional Representations in Task-Oriented Dialogue. In _Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Volume 1: Long Papers_ , pages 831–844, Florence, Italy. 

- Simone Balloccu, Patrícia Schmidtová, Mateusz Lango, and Ondrej Dušek. 2024. Leak, cheat, repeat: Data contamination and evaluation malpractices in closedsource LLMs. In _Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2024 - Volume 1: Long Papers_ , pages 67–93, St. Julian’s, Malta. 

- Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. 2023. Open LLM Leaderboard. https://huggingface.co/spaces/ HuggingFaceH4/open_llm_leaderboard. 

- Anja Belz. 2005. Corpus-driven generation of weather forecasts. In _Proc. 3rd Corpus Linguistics Conference_ . 

- Anja Belz. 2008. Automatic generation of weather forecast texts using comprehensive probabilistic generation-space models. _Nat. Lang. Eng._ , 14(4):431–455. 

- BigScience Workshop, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, et al. 2022. Bloom: A 176bparameter open-access multilingual language model. _arXiv preprint arXiv:2211.05100_ . 

- Sidney Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. 2022. GPT-NeoX-20B: An open-source autoregressive language model. In _Proceedings of BigScience Episode# 5–Workshop on Challenges & Perspectives in Creating Large Language Models_ , pages 95–136. 

- Armand Boschin. 2019. WikiDataSets : Standardized sub-graphs from WikiData. _CoRR_ , abs/1906.04536. 

- Thiago Castro Ferreira, Claire Gardent, Nikolai Ilinykh, Chris van der Lee, Simon Mille, Diego Moussallem, and Anastasia Shimorina. 2020. The 2020 bilingual, bi-directional WebNLG+ shared task: Overview and evaluation results (WebNLG+ 2020). In _Proceedings of the 3rd International Workshop on Natural Language Generation from the Semantic Web (WebNLG+)_ , pages 55–76, Dublin, Ireland (Virtual). Association for Computational Linguistics. 

- Asli Celikyilmaz, Elizabeth Clark, and Jianfeng Gao. 2020. Evaluation of text generation: A survey. _CoRR_ , abs/2006.14799. 

- Lingjiao Chen, Matei Zaharia, and James Zou. 2023. How is ChatGPT’s behavior changing over time? _CoRR_ , abs/2307.09009. 

- Wenhu Chen, Yu Su, Xifeng Yan, and William Yang Wang. 2020. KGPT: Knowledge-Grounded PreTraining for Data-to-Text Generation. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020_ , pages 8635–8648, Online. 

- David Cheng-Han Chiang and Hung-yi Lee. 2023. Can Large Language Models Be an Alternative to Human Evaluations? In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023_ , pages 15607–15631, Toronto, Canada. 

- Liam Cripwell, Anya Belz, Claire Gardent, Albert Gatt, Claudia Borg, Marthese Borg, John Judge, Michela Lorandi, Anna Nikiforovskaya, and William Soto Martinez. 2023. The 2023 WebNLG Shared Task on Low Resource Languages. Overview and Evaluation Results (WebNLG 2023). In _Proceedings of the Workshop on Multimodal, Multilingual Natural Language Generation and Multilingual WebNLG Challenge (MM-NLG 2023)_ , pages 55–66, Prague, Czech Republic. 

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IOAwareness. In _Advances in Neural Information Processing Systems 35: Annual Conference on Neural_ 

_Information Processing Systems 2022, NeurIPS 2022_ , New Orleans, LA, USA. 

- Seniz Demir, Sandra Carberry, and Kathleen F. McCoy. 2008. Generating Textual Summaries of Bar Charts. In _INLG 2008 - Proceedings of the Fifth International Natural Language Generation Conference, June 1214, 2008, Salt Fork_ , Ohio, USA. 

- Seniz Demir, Sandra Carberry, and Kathleen F. McCoy. 2012. Summarizing Information Graphics Textually. _Comput. Linguistics_ , 38(3):527–574. 

- Ondrej Dušek, David M. Howcroft, and Verena Rieser. 2019. Semantic noise matters for neural natural language generation. In _Proceedings of the 12th International Conference on Natural Language Generation, INLG 2019_ , pages 421–426, Tokyo, Japan. 

- Ondrej Dušek, Jekaterina Novikova, and Verena Rieser. 2020. Evaluating the state-of-the-art of end-to-end natural language generation: The E2E NLG challenge. _Comput. Speech Lang._ , 59:123–156. 

- Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. 2023. GPTScore: Evaluate as You Desire. _CoRR_ , abs/2302.04166. 

- Claire Gardent, Anastasia Shimorina, Shashi Narayan, and Laura Perez-Beltrachini. 2017. The WebNLG Challenge: Generating Text from RDF Data. In _Proceedings of the 10th International Conference on Natural Language Generation, INLG 2017, Santiago de Compostela_ , pages 124–133, Spain. 

- Albert Gatt and Emiel Krahmer. 2018. Survey of the State of the Art in Natural Language Generation: Core tasks, applications and evaluation. _J. Artif. Intell. Res._ , 61:65–170. 

- Sebastian Gehrmann, Tosin P. Adewumi, Karmanya Aggarwal, Pawan Sasanka Ammanamanchi, Aremu Anuoluwapo, Antoine Bosselut, Khyathi Raghavi Chandu, Miruna-Adriana Clinciu, Dipanjan Das, Kaustubh D. Dhole, Wanyu Du, Esin Durmus, Ondrej Dušek, Chris Emezue, Varun Gangal, Cristina Garbacea, Tatsunori Hashimoto, Yufang Hou, Yacine Jernite, Harsh Jhamtani, Yangfeng Ji, Shailza Jolly, Dhruv Kumar, Faisal Ladhak, Aman Madaan, Mounica Maddela, Khyati Mahajan, Saad Mahamood, Bodhisattwa Prasad Majumder, Pedro Henrique Martins, Angelina McMillan-Major, Simon Mille, Emiel van Miltenburg, Moin Nadeem, Shashi Narayan, Vitaly Nikolaev, Rubungo Andre Niyongabo, Salomey Osei, Ankur P. Parikh, Laura PerezBeltrachini, Niranjan Ramesh Rao, Vikas Raunak, Juan Diego Rodriguez, Sashank Santhanam, João Sedoc, Thibault Sellam, Samira Shaikh, Anastasia Shimorina, Marco Antonio Sobrevilla Cabezudo, Hendrik Strobelt, Nishant Subramani, Wei Xu, Diyi Yang, Akhila Yerukola, and Jiawei Zhou. 2021. The GEM benchmark: Natural language generation, its evaluation and metrics. _CoRR_ , abs/2102.01672. 

- Sebastian Gehrmann, Elizabeth Clark, and Thibault Sellam. 2023. Repairing the Cracked Foundation: A Survey of Obstacles in Evaluation Practices for Generated Text. _J. Artif. Intell. Res._ , 77:103–166. 

- Shahriar Golchin and Mihai Surdeanu. 2023. Time Travel in LLMs: Tracing Data Contamination in Large Language Models. _CoRR_ , abs/2308.08493. 

- Ari Holtzman, Peter West, and Luke Zettlemoyer. 2023. Generative Models as a Complex Systems Science: How can we make sense of large language model behavior? _CoRR_ , abs/2308.00189. 

- Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. _ACM Comput. Surv._ , 55(12):248:1–248:38. 

- Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7B. _CoRR_ , abs/2310.06825. 

- Shankar Kantharaj, Rixie Tiffany Ko Leong, Xiang Lin, Ahmed Masry, Megh Thakkar, Enamul Hoque, and Shafiq R. Joty. 2022. Chart-to-Text: A Large-Scale Benchmark for Chart Summarization. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022_ , pages 4005–4023, Dublin, Ireland. 

- Zdenˇek Kasner, Ioannis Konstas, and Ondˇrej Dušek. 2023. Mind the Labels: Describing Relations in Knowledge Graphs With Pretrained Models. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2023, Dubrovnik_ , pages 2390–2407, Croatia. 

- Tom Kocmi and Christian Federmann. 2023a. GEMBAMQM: Detecting Translation Quality Error Spans with GPT-4. In _Proceedings of the Eighth Conference on Machine Translation, WMT 2023_ , pages 768–775, Singapore. 

- Tom Kocmi and Christian Federmann. 2023b. Large Language Models Are State-of-the-Art Evaluators of Translation Quality. In _Proceedings of the 24th Annual Conference of the European Association for Machine Translation, EAMT 2023_ , pages 193–203, Tampere, Finland. 

- Rik Koncel-Kedziorski, Dhanush Bekal, Yi Luan, Mirella Lapata, and Hannaneh Hajishirzi. 2019. Text Generation from Knowledge Graphs with Graph Transformers. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, Volume 1 (Long and Short Papers)_ , pages 2284–2293, USA. 

- Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, Zae Myung Kim, and Dongyeop Kang. 2023. Benchmarking cognitive biases in large language models as evaluators. _CoRR_ , abs/2309.17012. 

- Fajri Koto, Jey Han Lau, and Timothy Baldwin. 2022. Can Pretrained Language Models Generate Persuasive, Faithful, and Informative Ad Text for Product Descriptions? In _Proceedings of the Fifth Workshop on E-Commerce and NLP (ECNLP 5)_ , pages 234– 243, Dublin, Ireland. 

- Rémi Lebret, David Grangier, and Michael Auli. 2016. Neural Text Generation from Structured Data with Application to the Biography Domain. In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016_ , pages 1203–1213, Austin, Texas, USA. 

- Wei Li, Wenhao Wu, Moye Chen, Jiachen Liu, Xinyan Xiao, and Hua Wu. 2022. Faithfulness in Natural Language Generation: A Systematic Survey of Analysis, Evaluation and Optimization Methods. _CoRR_ , abs/2203.05227. 

- Percy Liang, Michael I. Jordan, and Dan Klein. 2009. Learning Semantic Correspondences with Less Supervision. In _ACL 2009, Proceedings of the 47th Annual Meeting of the Association for Computational Linguistics and the 4th International Joint Conference on Natural Language Processing of the AFNLP, 2-7 August 2009_ , pages 91–99, Singapore. 

- Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023_ , pages 2511–2522, Singapore. 

- Michela Lorandi and Anja Belz. 2023. Data-to-text Generation for Severely Under-Resourced Languages with GPT-3.5: A Bit of Help Needed from Google Translate (WebNLG 2023). In _Proceedings of the Workshop on Multimodal, Multilingual Natural Language Generation and Multilingual WebNLG Challenge (MM-NLG 2023)_ , pages 80–86. 

- Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. 2024. Leave no context behind: Efficient infinite context transformers with infiniattention. _CoRR_ , abs/2404.07143. 

- Feng Nie, Jinpeng Wang, Jin-Ge Yao, Rong Pan, and Chin-Yew Lin. 2018. Operation-guided Neural Networks for High Fidelity Data-To-Text Generation. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pages 3879–3889, Brussels, Belgium. 

- Jekaterina Novikova, Ondˇrej Dušek, Amanda Cercas Curry, and Verena Rieser. 2017. Why We Need New Evaluation Metrics for NLG. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP 2017_ , pages 2241– 2252, Copenhagen, Denmark. 

- Jason Obeid and Enamul Hoque. 2020. Chart-to-Text: Generating Natural Language Descriptions for Charts by Adapting the Transformer Model. In _Proceedings of the 13th International Conference on Natural Language Generation, INLG 2020_ , pages 138–147, Dublin, Ireland. 

- OpenAI. 2023a. GPT-4 Technical Report. _CoRR_ , abs/2303.08774. 

- OpenAI. 2023b. Introducing ChatGPT. https:// openai.com/blog/chatgpt. Accessed on January 9, 2024. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022_ , New Orleans, LA, USA. 

- Ratish Puduppully, Li Dong, and Mirella Lapata. 2019a. Data-to-Text Generation with Content Selection and Planning. In _The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019_ , pages 6908–6915, Honolulu, Hawaii, USA. 

- Ratish Puduppully, Li Dong, and Mirella Lapata. 2019b. Data-to-text Generation with Entity Modeling. In _Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Volume 1: Long Papers_ , pages 2023–2035, Florence, Italy. 

- Ratish Puduppully, Yao Fu, and Mirella Lapata. 2022. Data-to-text Generation with Variational Sequential Planning. _Trans. Assoc. Comput. Linguistics_ , 10:697– 715. 

- Ratish Puduppully and Mirella Lapata. 2021. Data-totext Generation with Macro Planning. _Trans. Assoc. Comput. Linguistics_ , 9:510–527. 

- Clément Rebuffel, Laure Soulier, Geoffrey Scoutheeten, and Patrick Gallinari. 2020. A Hierarchical Model for Data-to-Text Generation. In _Advances in Information Retrieval - 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14-17, 2020, Proceedings, Part I_ , volume 12035 of _Lecture Notes in Computer Science_ , pages 65–80. 

- Ehud Reiter. 2023. We should evaluate real-world impact! https://ehudreiter.com/2023/11/13/ evaluate-real-world-impact/. Accessed on January 11, 2024. 

- Ehud Reiter and Robert Dale. 1997. Building applied natural language generation systems. _Nat. Lang. Eng._ , 3(1):57–87. 

- Leonardo F. R. Ribeiro, Martin Schmitt, Hinrich Schütze, and Iryna Gurevych. 2020. Investigating Pretrained Language Models for Graph-to-Text Generation. _CoRR_ , abs/2007.08426. 

- Anna Rogers. 2023. Closed AI Models Make Bad Baselines. https://hackingsemantics.xyz/ 2023/closed-baselines/. Accessed on January 11, 2024. 

- Huajie Shao, Jun Wang, Haohong Lin, Xuezhou Zhang, Aston Zhang, Heng Ji, and Tarek F. Abdelzaher. 2021. Controllable and Diverse Text Generation in E-commerce. In _WWW ’21: The Web Conference 2021_ , pages 2392–2401, Virtual Event / Ljubljana, Slovenia. 

- Mandar Sharma, John S. Brownstein, and Naren Ramakrishnan. 2021. TCube: Domain-Agnostic Neural Time-series Narration. _CoRR_ , abs/2110.05633. 

- Andrea Sottana, Bin Liang, Kai Zou, and Zheng Yuan. 2023. Evaluation Metrics in the Era of GPT-4: Reliably Evaluating Large Language Models on Sequence to Sequence Tasks. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023_ , pages 8776– 8788, Singapore. 

- Somayajulu Sripada, Ehud Reiter, Jim Hunter, and Jin Yu. 2002. Sumtime-meteo: Parallel corpus of naturally occurring forecast texts and weather data. _Computing Science Department, University of Aberdeen, Aberdeen, Scotland, Tech. Rep. AUCS/TR0201_ . 

- Rickard Stureborg, Dimitris Alikaniotis, and Yoshi Suhara. 2024. Large Language Models are Inconsistent and Biased Evaluators. 

- Craig Thomson and Ehud Reiter. 2020. A Gold Standard Methodology for Evaluating Accuracy in DataTo-Text Systems. In _Proceedings of the 13th International Conference on Natural Language Generation, INLG 2020_ , pages 158–168, Dublin, Ireland. 

- Craig Thomson, Ehud Reiter, and Somayajulu Sripada. 2021. SportSett:Basketball - A Robust and Maintainable Dataset for Natural Language Generation. page 9. 

- Craig Thomson, Ehud Reiter, and Barkavi Sundararajan. 2023. Evaluating factual accuracy in complex datato-text. _Comput. Speech Lang._ , 80:101482. 

- TogetherAI. 2023. Preparing for the era of 32K context: Early learnings and explorations. https: //www.together.ai/blog/llama-2-7b-32k. Accessed on January 2, 2024. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. LLaMA: Open and Efficient Foundation Language Models. _CoRR_ , abs/2302.13971. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian CantonFerrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open Foundation and Fine-Tuned Chat Models. _CoRR_ , abs/2307.09288. 

- Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. 2023. Zephyr: Direct Distillation of LM Alignment. _CoRR_ , abs/2310.16944. 

- Jannis Vamvas and Rico Sennrich. 2022. As little as possible, as much as necessary: Detecting over- and undertranslations with contrastive conditioning. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), ACL 2022_ , pages 490–500, Dublin, Ireland. 

- Chris van der Lee, Albert Gatt, Emiel van Miltenburg, and Emiel Krahmer. 2021. Human evaluation of automatically generated text: Current trends and best practice guidelines. _Comput. Speech Lang._ , 67:101151. 

- Chris van der Lee, Albert Gatt, Emiel van Miltenburg, Sander Wubben, and Emiel Krahmer. 2019. Best practices for the human evaluation of automatically generated text. In _Proceedings of the 12th International Conference on Natural Language Generation, INLG 2019_ , pages 355–368, Tokyo, Japan. 

- Emiel Van Miltenburg, Miruna Clinciu, Ondˇrej Dušek, Dimitra Gkatzia, Stephanie Inglis, Leo Leppänen, Saad Mahamood, Stephanie Schoch, Craig Thomson, and Luou Wen. 2023. Barriers and enabling factors for error analysis in NLG research. _Northern European Journal of Language Technology_ , 9. 

- Hongmin Wang. 2019. Revisiting Challenges in Datato-Text Generation with Fact Grounding. In _Proceedings of the 12th International Conference on Natural Language Generation, INLG 2019_ , pages 311–322, Tokyo, Japan. 

- Jiaan Wang, Yunlong Liang, Fandong Meng, Haoxiang Shi, Zhixu Li, Jinan Xu, Jianfeng Qu, and Jie Zhou. 2023a. Is ChatGPT a Good NLG Evaluator? A Preliminary Study. _CoRR_ , abs/2303.04048. 

- Jinpeng Wang, Yutai Hou, Jing Liu, Yunbo Cao, and Chin-Yew Lin. 2017. A Statistical Framework for Product Description Generation. In _Proceedings of the Eighth International Joint Conference on Natural Language Processing, IJCNLP 2017, Volume 2: Short Papers_ , pages 187–192, Taipei, Taiwan. 

- Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. 2023b. Large Language Models are not Fair Evaluators. _CoRR_ , abs/2305.17926. 

- Tsung-Hsien Wen, Milica Gašic, Nikola Mrkšic, Lina M Rojas-Barahona, Pei-Hao Su, David Vandyke, and Steve Young. 2015. Toward multi-domain language generation using recurrent neural networks. In _NIPS Workshop on Machine Learning for Spoken Language Understanding and Interaction_ . 

- Tsung-Hsien Wen, Milica Gasic, Nikola Mrksic, Lina Maria Rojas-Barahona, Pei-Hao Su, David Vandyke, and Steve J. Young. 2016. Multi-domain Neural Network Language Generation for Spoken Dialogue Systems. In _NAACL HLT 2016, The 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 120–129, San Diego California, USA. 

- Sam Wiseman, Stuart M. Shieber, and Alexander M. Rush. 2017. Challenges in Data-to-Document Generation. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP 2017_ , pages 2253–2263, Copenhagen, Denmark. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-Art Natural Language Processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, EMNLP 2020 - Demos_ , pages 38– 45, Online. 

- Wenda Xu, Danqing Wang, Liangming Pan, Zhenqiao Song, Markus Freitag, William Yang Wang, and Lei Li. 2023. INSTRUCTSCORE: Explainable Text Generation Evaluation with Finegrained Feedback. 

- Shuzhou Yuan and Michael Färber. 2023. Evaluating Generative Models for Graph-to-Text Generation. In _Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing, RANLP 2023_ , pages 1256–1264, Varna, Bulgaria. 

- Yilun Zhao, Haowei Zhang, Shengyun Si, Linyong Nan, Xiangru Tang, and Arman Cohan. 2023. Investigating Table-to-Text Generation Capabilities of LLMs in Real-World Information Seeking Scenarios. 

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. _CoRR_ , abs/2306.05685. 

## **A QUINTD Data** 

Here, we describe the data sources we include in the QUINTD collection tool and the procedure of collecting the QUINTD-1 benchmark. To replicate the data collection, please refer to the scripts we provide.[14] 

## **A.1 Selection of Data Sources** 

When selecting the data sources, we had the following desiderata: 

- Data needs to be publicly available. 

- Data needs to represent a common data-to-text task. 

- Data needs to be in a common format (or straightforwardly transformable to one). 

We settled on the data sources described in Appendix A.2. All the sources can be accessed using an API. Note that some of the APIs have access limits, either for the requests made from a single account per day or for a number of requests from an IP address within a time window. However, these limits do not severely limit the data collection process on the scale we use here. 

## **A.2 Data Collection** 

Table 6 summarizes the output types for each domain. 

|**Domain Id**|**Output type**|
|---|---|
|openweather|fve-day weather forecast|
|gsmarena|product description|
|ice_hockey|ice hockey game summary|
|owid|chart caption|
|wikidata|entity description|



Table 6: The output types for individual domains in QUINTD. 

14https://github.com/kasnerz/quintd 

## **A.2.1 OpenWeather** 

OpenWeather (OpenWeatherMap.org) is an online service that provides global weather data via web interface and API. The API responses are in the JSON format documented at the official website. For our experiments, we used the forecast5 API, which allows to download a 5-day forecast with 3-hour resolution for any location specified by its GPS coordinates. 

The free tier is limited to 1,000 API calls per day, which is enough to download our whole test set in one bulk. However, at the time of experiments, the free API only allowed to download the data for the _time when the request was made_ . At the time of writing, OpenWeather is pushing a new One Call API 3.0 which allows to download weather data for any timestamp, but only _4 days ahead_ (instead of 5). These restrictions somehow limit the replicability of our QUINTD-1 dataset (at least with the free API) but do not limit downloading a new batch of data with a similar format. 

For the QUINTD-1 dataset, we randomly sampled 100 cities for each split from the list of cities with a population over 1000 and used their coordinates in the queries to OpenWeather API. All the data forecasts were downloaded on Jan 3, 2024. 

## **A.2.2 GSMArena** 

GSMArena is a website providing specifications and reviews for mobile devices. For downloading the data, we used the unofficial gsmarena-api tool, which returns the data in a JSON format. Note that GSMArena imposes limitations on the number of requests per IP address, which may induce delays when downloading a larger amount of data. 

To create a balanced sample, we downloaded detailed specifications of 10 products from each available brand and randomly selected 100 products for each split from the downloaded set. 

## **A.2.3 RapidAPI Ice Hockey** 

RapidAPI is a service that provides API access to data from multiple domains, including sport, finance, entertainment, and others. Most APIs are provided in a freemium mode, i.e., with a limited number of daily API calls. 

For QUINTD, we selected the IceHockeyAPI (popularity 9.1 / 10), which provides access to ice hockey games from world top leagues. Our choice was influenced by our own personal preferences, combined with the desire to cover a sport that has 

not been covered previously in sports report generation. 

We used the matches endpoint which returns high-level details about a game. Note that the API allows only 50 requests per day, but that does not limit the data collection since the endpoint returns _all the games_ played on a particular day in a single request. We downloaded the games played on 27 November 2023 for the development set (184 games) and 29 November 2023 for the test set (216 games), taking a random sample of 100 for each split. 

## **A.2.4 OurWorldInData** 

OurWorldInData is a public database and web interface for data about world developments in various domains and sources. We used the official API (currently experimental), which is accessible through the Python package owid-catalog. The package allows accessing individual CSV tables as Pandas dataframes. 

For our data collection, we decided to limit ourselves to time series, i.e., a single column with values changing over time. Besides the simplicity of visualizing such a chart (which is used by human annotators for checking the correctness of the output), there is also a clear goal for the target chart description: describing the developments of a value over time. We also limited ourselves to the health domain. In particular, we selected the tables COVID data (columns new_cases_smoothed_per_million, new_tests_smoothed_per_thousand, people_ vaccinated_per_hundred, reproduction_ rate, and positive_rate) and Life expectancy data (column life_expectancy_0). 

We downloaded the data for all countries with non-empty entries in the table, taking a random sample of 100 examples for each split. On model input, we formatted the data for each time series as a two-column CSV, including the title, the description, and the unit for each example as a comment (#) at the beginning of the input. 

## **A.2.5 Wikidata** 

Wikidata is a large open-source knowledge graph containing factual information about entities and their properties. Wikidata provides access through an official API, but we instead decided to extract our data using the wikidatasets (Boschin, 2019) Python library, which provides access to preprocessed properties of entities from particular do- 

mains. It allowed us to avoid crawling and filtering the knowledge graph, and its offline processing made the data collection faster.[15] 

For our dataset, we selected the entities from the companies, countries, films, and humans domains. For each entity, we randomly extracted between 2 to 10 properties in the knowledge graph. We extracted up to 100 subgraphs for each domain and took a random sample of 100 subgraphs for each split. On model input, we formatted each subgraph as a simple Markdown-formatted text snippet, using the entity as a title and including a bullet point for each key-value pair. 

## **B Human Evaluation** 

As described in §4.1, we set up the human evaluation campaign on Prolific. To make the data more accessible to the annotators, we created custom data visualizations for each domain. For the data in openweather and owid, we used interactive graphs from Highcharts.com, and we manually created the tables for other domains. You can find the full instructions for human annotators in Figure 5 and the examples of data visualizations in Appendix E. 

## **C GPT-4 Evaluation** 

We used the prompt in Figure 4 for instantiating the GPT-4-based metric.[16] We set the temperature to 0 to improve the replicability of the process. We ensured that the output is a valid JSON using the parameter response_format in the OpenAI API. At the price of $0.01 per 1k input tokens and $0.03 per 1k generated tokens, the evaluation process costs approximately $45 in total. 

## **C.1 Aligning the Errors** 

For aligning the errors with the original text, we perform string matching on the text span decoded by GPT-4 in the TEXT_SPAN field. In our preliminary experiments, this method proved to be more robust than either asking for start and end indices of the error span (which would rely on the model’s ability to count characters) or performing sequence tagging on the copy of the input (which would rely on the model’s ability to perfectly copy the input). 

15All the entities and properties are linked with an identifier to the Wikidata database, making the process also replicable through the official API. 

> 16Note that the example in the prompt differs from the example used for human annotators (see Figure 5). We revised the example to be more instructive, but we were not able to re-run the GPT-4 evaluation due to our limited budget. 

We tried to respect the monotonic ordering of text spans but fell back to full-text search if the span is not found following the previous one. We consider this approach successful since matching completely failed only in a minority of cases (137 out of 6927). Based on our manual examination, these mostly include cases where GPT-4 tried to suggest a _missing_ piece of text as an error or did not manage to copy the input text verbatim. 

## **D Experiments with Open LLMs as Evaluators** 

To select the most suitable LLM for automatic evaluation, we compared correlations with human judgment of the following models: 

- **GPT-4** (OpenAI, 2023a) used via OpenAI API (gpt-4-1106-preview), 

- **GPT-3.5** (OpenAI, 2023b) used via OpenAI API (gpt-3.5-turbo-1106), 

- **Llama-3-70B**[17] running locally via Ollama in 4-bit quantization (meta-llama/Meta-Llama-3-70B). 

We used all the models with the same prompts, temperature 0, and force-decoded JSON outputs. In Table 7, we show Pearson correlation coefficients computed as described in §5.2. We can see that the strongest model is GPT-4, followed by Llama-370B and GPT-3.5. As the gap between the models is substantial, we opted for using GPT-4 which is the strongest model. 

|model|_r_word|_r_example|_r_domain|
|---|---|---|---|
|GPT-4|**0.26**|**0.52**|**0.93**|
|GPT-3.5|0.07|0.33|0.82|
|Llama3-70B|0.09|0.44|0.92|



Table 7: Pearson correlation coefficients of the model annotations as compared with human annotations (cf. §5.2). 

## **E Examples** 

Here, we present an example of inputs and model outputs (along with annotations) for each domain: 

- openweather: Figure 7 (in) and Table 8 (out), 

- gsmarena: Figure 8 (in) and Table 8 (out), 

- 17https://llama.meta.com/llama3/ 

- ice_hockey: Figure 9 (in) and Table 10 (out), 

- owid: Figure 10 (in) and Table 11 (out), 

- wikidata: Figure 11 (in) and Table 12 (out). 

Note that the graphs for openweather and owid are interactive when accessed through the web interface. 

## **F Full Results** 

Here, we include the tables with results for individual domains: 

- Table 13 presents the average _numbers of errors per output_ separately for each domain (the aggregated results are in Table 3), 

- Table 14 presents the ratio of _outputs containing at least one error_ separately for each domain (the aggregated results are in Table 4). 

## **System Message** 

You are an expert data-to-text error annotation system. You undestand structured data and you can correcly operate with units and numerical values. You are designed to output token-level annotations in JSON. 

## **Prompt** 

Given the data: 

ˋˋˋ 

data ˋˋˋ 

Annotate all the errors in the following text: 

ˋˋˋ 

text ˋˋˋ 

Output the errors as a JSON list "errors" in which each object contains fields "reason", "text", and "type". The value of "text" is the text of the error. The value of "reason" is the reason for the error. The value of "type" is one of 0, 1, 2, 3 based on the following list: 

- 0: Incorrect fact: The fact in the text contradicts the data. 

- 1: Not checkable: The fact in the text cannot be checked in the data. 

- 2: Misleading: The fact in the text is misleading in the given context. 

- 3: Other: The text is problematic for another reason, e.g. grammatically or stylistically incorrect, irrelevant, or repetitive. The list should be sorted by the position of the error in the text. 

*Example:* data: ˋˋˋ 

[ [ "Aditi Bhagwat", "occupation", "television actor" ], [ "Aditi Bhagwat", "date of birth", "18 January 1981" ] ] 

ˋˋˋ 

text: 

ˋˋˋ 

Aditi Bhagwat, born on January 18, 1991, used to be a popular Indian television actor. The data comes from a knowledge graph. 

ˋˋˋ 

output: 

ˋˋˋ "errors": ["reason": "The data mentions that the actor was born on 1981", "text": "1991", "type": 0, "reason": "Misleadingly suggests that the actor is not alive", "text": "used to be", type: 2, "reason": "Popularity is not mentioned in the data", "text": "popular", type: 1, "reason", "Nationality is not mentioned in the data", "text": "Indian", type: 1, "reason": "The note is superfluous", "text": "The data comes from a knowledge graph.", type: 3] ˋˋˋ 

Note that some details may not be mentioned in the text: do not count omissions as errors. Also do not be too strict: some facts can be less specific than in the data (rounded values, shortened or abbreviated text, etc.), do not count these as errors. If there are no errors in the text, "errors" will be an empty list. 

Figure 4: The prompt we used for the GPT-4 evaluation metric. 

In this task, you will annotate **20** examples in total. For each example, you will see **data** on the left side and the corresponding generated **text** on the right side. Your task is to **annotate errors** in the text with respect to the data. There are five types of errors that you can mark in the generated text: 

1. **Incorrect factI** : The fact in the text contradicts the data. 

2. **Not checkableNC** : The fact in the text cannot be checked given the data. 

3. **MisleadingM** : The fact in the text is misleading in the given context. 

- **O** 

- 4. **Other** : The text is problematic for another reason, e.g. grammatically or stylistically incorrect, irrelevant, or repetitive. 

## **How to mark and submit the annotations?** 

Use your mouse to **highlight specific parts of the text** containing the errors. To switch between error categories, repeatedly click on the highlighted text (the last click removes the highlight). Note that highlighting from the right to left can work better for longer spans. 

Once you think you have marked all the errors present in the text, click the **Mark example as complete** button (you can still update the annotation later). You will be able to submit the annotations once they are all are marked as complete. 

## **How should I decide on the errors?** 

- Each error span should include all the words related to the error (but nothing else). 

- If you think the fact is probably true, but cannot be derived from the data, mark it as not checkable. 

- If you are not really sure if the fact should be marked as an error, leave it unmarked. 

## **An example of the annotated output** 

An example of the data input and the corresponding text annotated with errors: 

## **data** 

## **Nokia 3310** 

- **color:** black, blue, grey 

- **display:** 320x240px 

## **text (product description)** 

Nokia 3310 is **produced in FinlandNC** and features a **320x320 displayI** . It is **available in black color.M The data seem to provide only partial information about the phone.O** 

## **explanation** 

- **produced in Finland** : The country where the phone is produced is not mentioned in the data. 

- **320x320** : The data mentions that the display has resolution 320x240px. 

- **available in black color** : Misleadingly suggests that the phone is not available in other colors. 

- **The data seem to provide only partial information about the phone.** : The note is irrelevant for the phone description. 

_Note that for the sake of brevity, this particular example is based on a small data input and contains many errors, which may not always be the case._ 

## **What kinds of data and text can I encounter?** 

You can encounter the following kinds of texts: 

- a 5-day weather forecast generated from weather data, 

- a description of a product generated from product specifications 

- an ice hockey game report generated from information about the game, 

- a caption of a health-related chart, 

- a description of an entity (human, country, film, or company) based on its properties. 

## **What is the source of the data and the texts?** 

The data is downloaded from public sources (openweathermap.org, gsmarena.com, rapidapi.com, ourworldindata.com, wikidata.com). The texts are generated by large language models. 

## **What is the purpose of the research?** 

The research is conducted by the NLP research group at Charles University in Prague. The research is a part of the ERC project Next-Generation Natural Language Generation and was approved by the ethical committee of the project. 

By continuing, you agree that the anonymous annotations will be made available to the research community to help evaluating and improving data-to-text generation systems based on large language models. 

Figure 5: The instructions given to the human annotators. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation_images/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation.pdf-0020-00.png)


Figure 6: The annotation interface for human annotators. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation_images/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation.pdf-0020-02.png)


Figure 7: An example of an input from openweather (see the outputs in Table 8). 

- **Model Human annotations** ( _E_ hum) **Llama2** The weather in Niemegk, Germany, will be cloudy and rainy for the next few days. On January 3rd, the temperature will be around 9.55°C, with a feels-like temperature of 6.59°C. The weather will be light rain, with a wind speed of 6.24 m/s and a gust of 12.1 m/s. **The rain will last for 3 hoursNC** , and the pressure will be 991 hPa. 

- **Mistral** From January 3rd to January 7th, the city of Niemegk in Germany will experience a range of weather conditions. On January 3rd, the temperature will be 9.55°C with a feels like temperature of 6.59°C. There will be light rain with a **3-hour precipitationNC** of 1.06mm. The wind speed will be 6.24m/s with a gust of 12.1m/s. On January 4th, the temperature will be 8.2°C with a feels like temperature of 5.64°C. There will be light rain with a **3-hour precipitationI** of 1.94mm. The wind speed will be 4.31m/s with a gust of 11.6m/s. On January 5th, the temperature will be **6.37°C with a feels like temperature of 2.25°CI** . There will be light rain with a **3-hour precipitation of 1.13mmI** . **The wind I** 

- **speed will be 7.07m/s with a gust of 15.51m/s** . On January 6th, the temperature will be **2.17°CI** with a **feels like temperature of -3.55°CI** . There will be overcast clouds with a **3-hour precipitation of I** 

- **0.31mm** . The wind speed will be **3.54m/s with a I** 

- **gust of 7.11m/s** . On January 7th, the temperature will be - **0.89°C with a feels like temperature of -0.89°CI** . There will be overcast clouds with a **3hour precipitation of 0.31mm. The wind speed I** 

- **will be 0.63m/s with a gust of 0.7m/s** . 

- **Zephyr** Over the next five days, the weather in Niemegk, Germany will be characterized by fluctuating temperatures and varying precipitation. On Thursday, January **3I** , light rain and a **lowI** temperature of 9.55°C can be expected, with a chance of rain continuing into Friday. Snowfall is possible on Friday, with a low temperature of -0.89°C. Saturday will be overcast with a low temperature of **0.03°CI** , and **light rain is forecastedI** for Saturday evening. Sunday will be overcast as well, with a low temperature of - **2.23°CI** . Monday will see a drop in temperatures, with a low of - **4.07°CI** and scattered clouds. **Temperatures will continue to fall throughout the week, with lows of -5.19°C and -9.51°C on Tuesday and NC** 

- **Wednesday, respectively** . Wind speeds will vary from 3.1 to **15.51I** meters per second, with gusts reaching up to **15.51I** meters per second on Monday. 

- **GPT-3.5** Over the next five days, the weather in Niemegk, Germany is expected to be quite varied. On January 4th, there will be light rain with temperatures **I** 

- ranging from **2.17°C to 9.55°C** . The following day, January 5th, there will be **lightI** snow with temperatures dropping to as low as -0.89°C. On January 6th, the weather will be overcast with temperatures ranging from -0.32°C to 1.69°C. The trend continues on January 7th with overcast clouds and temperatures **I** 

- dropping to as low as - **2.51°C** . Finally, on January 8th, there will be scattered clouds with temperatures ranging from -5.19°C to -3.67°C. Throughout this period, wind speeds will vary between 0.63 m/s and 8.12 m/s, and the pressure will range from **991I** hPa to 1034 hPa. 

**GPT-4 annotations** ( _E_ gpt) The weather in Niemegk, Germany, will be cloudy and rainy for the next few days. On January 3rd, the temperature will be around 9.55°C, with a feels-like temperature of 6.59°C. The weather will be light rain, with a wind speed of 6.24 m/s and a gust of 12.1 m/s. **The rain will last for 3 hoursM** , and the pressure will be 991 hPa. From January 3rd to January 7th, the city of Niemegk in Germany will experience a range of weather conditions. On January 3rd, the temperature will be 9.55°C with a feels like temperature of 6.59°C. There will be light rain with a 3-hour precipitation of 1.06mm. The wind speed will be 6.24m/s with a gust of 12.1m/s. On January 4th, the temperature will be 8.2°C with a feels like temperature of 5.64°C. There will be light rain with a 3-hour precipitation of 1.94mm. The wind speed will be 4.31m/s with a gust of 11.6m/s. On January 5th, the temperature will be 6.37°C with a feels like temperature of 2.25°C. There will be light rain with a 3-hour precipitation of **1.13mmI** . The wind speed will be 7.07m/s with a gust of 15.51m/s. On January 6th, the temperature will be 2.17°C with a feels like temperature of -3.55°C. There will be overcast clouds with a 3- **I** hour precipitation of **0.31mm** . The wind speed will be **3.54m/sI** with a gust of **7.11m/sI** . On January 7th, the temperature will be **-0.89°CI** with a feels like temperature of **-0.89°CI** . There will be overcast clouds with a 3-hour precipitation of **0.31mmI** . The **I I** wind speed will be **0.63m/s** with a gust of **0.7m/s** . 

Over the next five days, the weather in Niemegk, Germany will be characterized by fluctuating temperatures and varying precipitation. On **Thursday, January 3I** , light rain and a **low temperature of 9.55°CI** can be expected, with a chance of rain continuing into Friday. Snowfall is possible on Friday, with a **low temperature of -0.89°CI** . Saturday will be overcast with a low temperature of 0.03°C, and **light rain is forecasted for Saturday eveningI** . Sunday will be overcast as well, with a **low temperature of -2.23°CI** . Monday will see a drop in temperatures, with a **low of -4.07°CI** and scattered clouds. Temperatures will continue to fall throughout the week, with **lows of -5.19°C and -9.51°C on Tuesday and WednesdayI** , respectively. Wind speeds will vary from 3.1 to 15.51 meters per second, with **gusts reaching up to 15.51 meters per second on MonI day** . 

Over the next five days, the weather in Niemegk, Germany is expected to be quite varied. On January 4th, there will be light rain with temperatures **I** ranging from 2.17°C to **9.55°C** . The following day, January 5th, there will be light snow with temperatures dropping to as low as **-0.89°CI** . On January 6th, the weather will be overcast with temperatures ranging from -0.32°C to **1.69°CI** . The trend continues on January 7th with overcast clouds and temperatures dropping to as low as **-2.51°CI** . Finally, on January 8th, there will be scattered clouds with temperatures ranging from -5.19°C to **-3.67°CI** . Throughout this period, wind speeds will vary between 0.63 m/s and **8.12 m/sI** , and the pressure will range from **991 hPaI** to **1034 hPaI** . 

Table 8: An example of the model outputs on openweather (see the input in Table 8). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation_images/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation.pdf-0022-00.png)


Figure 8: An example of an input from gsmarena (see the outputs in Table 9). 

|**Model**|**Human annotations**(_E_hum)|**GPT-4 annotations**(_E_gpt)||
|---|---|---|---|
|**Llama2**<br>**Mistral**<br>**Zephyr**<br>**GPT-3.5**|The Gigabyte GSmart Mika M3 is a smartphone<br>that was released in 2014. It has a 5-inch IPS LCD<br>display with a resolution of 720x1280 pixels and a<br>13MP main camera. The device also has an 8MP<br>front camera and is powered by a Mediatek MT6582<br>chipset with 1GB of RAM and 8GB of internal stor-<br>age. It has dual SIM support and is available in Navy<br>Blue.<br>The GSmart Mika M3 is a smartphone developed<br>by Gigabyte. It was released in November 2014<br>and is powered by the Android 4.4.2 operating sys-<br>tem. The device features a 5.0-inch IPS LCD dis-<br>play with a resolution of 720 x 1280 pixels, which<br>translates to 294 pixels per inch. The smartphone is<br>equipped with a 13-megapixel rear camera<br>**with**<br>**aut**<br>**-**<br>**o**<br>**fo**<br>**cus**<br>**NC** and LED fash, as well as an 8-megapixel<br>front-facing camera. It is powered by a Mediatek<br>MT6582 chipset with a quad-core 1.3 GHz Cortex-<br>A7 CPU and a Mali-400MP2 GPU. The device has<br>8GB of internal storage and 1GB of RAM, with the<br>option to expand storage via a microSDHC card.<br>The GSmart Mika M3 also features Wi-Fi, Blue-<br>tooth, GPS, and<br>**NFC**<br>**con**<br>**nec**<br>**tiv**<br>**ity**<br>**I**, as well as a<br>3.5mm headphone jack and a microUSB 2.0 port for<br>charging and data transfer. The device is available<br>in Navy Blue and is powered by a 1900mAh Li-Ion<br>battery, which provides up to 150 hours of stand-by<br>time and up to 4 hours and 30 minutes of talk time.<br>The Gigabyte GSmart Mika M3 is a smartphone that<br>boasts a 5.0-inch IPS LCD display with a resolu-<br>tion of 720 x 1280 pixels. It runs on Android 4.4.2<br>(KitKat) and is powered by a quad-core 1.3 GHz<br>Cortex-A7 processor with a Mali-400MP2 GPU. The<br>device comes with 8GB of internal storage and 1GB<br>of RAM, which can be expanded via a dedicated<br>microSDHC card slot. The primary camera is a<br>13-megapixel<br>**aut**<br>**o**<br>**fo**<br>**cus**<br>**lens**<br>**NC** with an LED fash,<br>while the secondary camera is an 8-megapixel lens.<br>Connectivity options include Wi-Fi, Bluetooth, GPS,<br>FM radio, and microUSB 2.0. The smartphone is<br>equipped with a 1900mAh removable Li-Ion battery<br>that provides up to<br>**4**<br>**hours**<br>**I** of talk time and up to<br>150 hours of standby time. The device is available in<br>Navy Blue and supports dual SIM cards (Mini-SIM/<br>Micro-SIM).<br>The GSmart Mika M3, also known as the Giga-<br>byte GSmart Mika M3, is a smartphone that was<br>announced in October 2014 and released in Novem-<br>ber of the same year. It features a 5.0-inch IPS LCD<br>display with a resolution of 720 x 1280 pixels and<br>is powered by a Quad-core 1.3 GHz Cortex-A7 pro-<br>cessor. The phone comes with 1GB of RAM and<br>8GB of internal storage, which can be expanded<br>via a microSDHC card slot. The main camera is<br>13MP with LED fash, while the selfe camera is<br>8MP. The phone runs on Android 4.4.2 (KitKat)<br>and is equipped with a 1900mAh removable battery.<br>Other features include dual SIM capability, Wi-Fi,<br>Bluetooth, GPS, FM radio, and microUSB 2.0. The<br>GSmart Mika M3 is available in the color Navy Blue<br>and has a sleek and compact design, measuring 144<br>x 70.5 x 8.3 mm and weighing 146g.|The Gigabyte GSmart Mika M3 is a smartphone<br>that was released in 2014. It has a<br>**5**<br>**-**<br>**inch**<br>**I** IPS<br>LCD display with a resolution of 720x1280 pixels<br>and a 13MP main camera. The device also has an<br>**8MP**<br>**front**<br>**cam**<br>**era**<br>**NC** and is powered by a Medi-<br>atek MT6582 chipset with 1GB of RAM and 8GB<br>of internal storage. It has dual SIM support and<br>**is**<br>**avail**<br>**able**<br>**in**<br>**Navy**<br>**Blue**<br>**NC**.<br>The GSmart Mika M3 is a smartphone developed<br>by Gigabyte. It was<br>**re**<br>**leased**<br>**in**<br>**Novem**<br>**ber**<br>**2014**<br>**I**<br>and is powered by the Android 4.4.2 operating sys-<br>tem. The device features a 5.0-inch IPS LCD dis-<br>play with a resolution of 720 x 1280 pixels, which<br>translates to 294 pixels per inch. The smartphone<br>is equipped with a 13-megapixel rear camera with<br>autofocus and LED fash, as well as an 8-megapixel<br>front-facing camera. It is powered by a Mediatek<br>MT6582 chipset with a quad-core 1.3 GHz Cortex-<br>A7 CPU and a Mali-400MP2 GPU. The device has<br>8GB of internal storage and 1GB of RAM, with the<br>option to expand storage via a microSDHC card.<br>The GSmart Mika M3 also features Wi-Fi, Blue-<br>tooth, GPS, and<br>**NFC**<br>**con**<br>**nec**<br>**tiv**<br>**ity**<br>**I**, as well as a<br>3.5mm headphone jack and a microUSB 2.0 port<br>for charging and data transfer. The device is<br>**avail**<br>**-**<br>**able**<br>**in**<br>**Navy**<br>**Blue**<br>**M** and is powered by a 1900mAh<br>Li-Ion battery, which provides up to 150 hours of<br>stand-by time and up to 4 hours and 30 minutes of<br>talk time.<br>The Gigabyte GSmart Mika M3 is a smartphone that<br>boasts a 5.0-inch IPS LCD display with a resolu-<br>tion of 720 x 1280 pixels. It runs on Android 4.4.2<br>(KitKat) and is powered by a quad-core 1.3 GHz<br>Cortex-A7 processor with a Mali-400MP2 GPU. The<br>device comes with 8GB of internal storage and 1GB<br>of RAM, which can be expanded via a dedicated<br>microSDHC card slot. The primary camera is a 13-<br>megapixel autofocus lens with an LED fash, while<br>the secondary camera is an 8-megapixel lens. Con-<br>nectivity options include Wi-Fi, Bluetooth, GPS,<br>FM radio, and microUSB 2.0. The smartphone is<br>equipped with a 1900mAh removable Li-Ion battery<br>that provides<br>**up**<br>**to**<br>**4**<br>**hours**<br>**of**<br>**talk**<br>**time**<br>**I** and up to<br>150 hours of standby time. The device is available in<br>Navy Blue and supports dual SIM cards (Mini-SIM/<br>Micro-SIM).<br>The GSmart Mika M3, also known as the<br>**Gi**<br>**ga**<br>**byte**<br>**GS**<br>**mart**<br>**Mika**<br>**M3**<br>**I**, is a smartphone that was an-<br>nounced in October 2014 and released in November<br>of the same year. It features a 5.0-inch IPS LCD<br>display with a resolution of 720 x 1280 pixels and<br>is powered by a Quad-core 1.3 GHz Cortex-A7 pro-<br>cessor. The phone comes with 1GB of RAM and<br>8GB of internal storage, which can be expanded<br>via a microSDHC card slot.<br>**The**<br>**main**<br>**cam**<br>**era**<br>**is**<br>**13MP**<br>**with**<br>**LED**<br>**fash**<br>**NC**,<br>**while**<br>**the**<br>**selfe**<br>**cam**<br>**-**<br>**era**<br>**is**<br>**8MP**<br>**NC**. The phone runs on Android 4.4.2<br>(KitKat)<br>**and**<br>**is**<br>**equipped**<br>**with**<br>**a**<br>**1900mAh**<br>**re**<br>**mov**<br>**-**<br>**able**<br>**bat**<br>**tery**<br>**NC**. Other features include dual SIM<br>capability, Wi-Fi, Bluetooth, GPS, FM radio, and<br>microUSB 2.0.<br>**The**<br>**GS**<br>**mart**<br>**Mika**<br>**M3**<br>**is**<br>**avail**<br>**able**<br>**in**<br>**the**<br>**color**<br>**Navy**<br>**Blue**<br>**and**<br>**has**<br>**a**<br>**sleek**<br>**and**<br>**com**<br>**-**<br>**pact**<br>**de**<br>**sign**<br>**NC**,measuring 144 x 70.5 x 8.3 mm and<br>weighing 146g.||



Table 9: An example of the model outputs on gsmarena (see the input in Table 9). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation_images/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation.pdf-0024-00.png)


Figure 9: An example of an input from ice_hockey (see the outputs in Table 10). 

|**Model**|**Human annotations**(_E_hum)|**GPT-4 annotations**(_E_gpt)|
|---|---|---|
|**Llama2**<br>**Mistral**<br>**Zephyr**<br>**GPT-3.5**|In the VHL 23/24 season, Kristall Saratov hosted<br>HC Norilsk at their home arena. Kristall Saratov<br>**won**<br>**the**<br>**game**<br>**I** with a fnal score of 1-5. Kristall<br>Saratov’s home team<br>**scored**<br>**one**<br>**goal**<br>**in**<br>**the**<br>**frst**<br>**pe**<br>**riod**<br>**I**,<br>**while**<br>**HC**<br>**No**<br>**rilsk**<br>**scored**<br>**two**<br>**goals**<br>**in**<br>**the**<br>**sec**<br>**ond**<br>**and**<br>**third**<br>**pe**<br>**ri**<br>**ods**<br>**I**.<br>**The**<br>**game**<br>**lasted**<br>**for**<br>**60**<br>**min**<br>**utes**<br>**NC**, with Kristall Saratov’s home team<br>**lead**<br>**ing**<br>**the**<br>**game**<br>**for**<br>**the**<br>**en**<br>**tire**<br>**du**<br>**ra**<br>**tion**<br>**I**.<br>On November 28th, 2023, the VHL 23/24 season<br>game between Kristall Saratov and HC Norilsk took<br>place. The game was played at the home arena of<br>Kristall Saratov, with the home team<br>**win**<br>**ning**<br>**1**<br>**-**<br>**0**<br>**I**. The game was played in three periods, with the<br>frst period<br>**end**<br>**ing**<br>**in**<br>**a**<br>**0**<br>**-**<br>**0**<br>**tie**<br>**I**,the second period<br>ending in<br>**a**<br>**2**<br>**-**<br>**3**<br>**score**<br>**in**<br>**fa**<br>**vor**<br>**of**<br>**HC**<br>**No**<br>**rilsk**<br>**I**,and<br>the third period ending in a 1-0 score in favor of<br>Kristall Saratov. The game ended in normal time,<br>with a<br>**to**<br>**tal**<br>**score**<br>**of**<br>**3**<br>**-**<br>**1**<br>**in**<br>**fa**<br>**vor**<br>**of**<br>**HC**<br>**No**<br>**rilsk**<br>**I**.<br>In the VHL 23/24 season, Kristall Saratov hosted<br>HC Norilsk. The frst period saw two<br>**quick**<br>**NC** goals<br>from Norilsk, with<br>**one**<br>**in**<br>**the**<br>**sec**<br>**ond**<br>**and**<br>**third**<br>**pe**<br>**-**<br>**ri**<br>**ods**<br>**as**<br>**well**<br>**I**,resulting in a dominant<br>**5**<br>**-**<br>**0**<br>**I** victory.<br>Kristall Saratov managed to score a single goal in<br>the third period, but it was not enough to overcome<br>Norilsk’s lead. The fnal score was 1-5 in favor of<br>HC Norilsk.<br>In a match from the VHL 23/24 season, Kristall<br>Saratov faced off against HC Norilsk. The game took<br>place on November 28, 2023, at<br>**15:30**<br>**I**. Despite a<br>goal from Kristall Saratov in the third period, HC<br>Norilsk emerged victorious with a fnal score of 5-1.<br>The away team dominated the game, scoring 2 goals<br>in the frst period and 3 goals in the second period.<br>Kristall Saratov struggled to keep up, resulting in a<br>clear win for HC Norilsk.|In the VHL 23/24 season, Kristall Saratov hosted<br>HC Norilsk at their home arena.<br>**Kristall**<br>**Sara**<br>**tov**<br>**won**<br>**the**<br>**game**<br>**with**<br>**a**<br>**f**<br>**nal**<br>**score**<br>**of**<br>**1**<br>**-**<br>**5.**<br>**I**<br>**Kristall**<br>**Sara**<br>**tov’s**<br>**home**<br>**team**<br>**scored**<br>**one**<br>**goal**<br>**in**<br>**the**<br>**frst**<br>**pe**<br>**riod**<br>**I**,<br>**while**<br>**HC**<br>**No**<br>**rilsk**<br>**scored**<br>**two**<br>**goals**<br>**in**<br>**the**<br>**sec**<br>**ond**<br>**and**<br>**third**<br>**pe**<br>**ri**<br>**ods.**<br>**I**<br>**The**<br>**game**<br>**lasted**<br>**for**<br>**60**<br>**min**<br>**utes**<br>**NC**,<br>**with**<br>**Kristall**<br>**Sara**<br>**tov’s**<br>**home**<br>**team**<br>**lead**<br>**ing**<br>**the**<br>**game**<br>**for**<br>**the**<br>**en**<br>**tire**<br>**du**<br>**ra**<br>**tion.**<br>**I**<br>On November 28th, 2023, the VHL 23/24 season<br>game between Kristall Saratov and HC Norilsk took<br>place. The game was played at the home arena of<br>Kristall Saratov,<br>**with**<br>**the**<br>**home**<br>**team**<br>**win**<br>**ning**<br>**1**<br>**-**<br>**0**<br>**I**.<br>The game was played in three periods, with the frst<br>period ending in a 0-0 tie,<br>**the**<br>**sec**<br>**ond**<br>**pe**<br>**riod**<br>**end**<br>**-**<br>**ing**<br>**in**<br>**a**<br>**2**<br>**-**<br>**3**<br>**score**<br>**in**<br>**fa**<br>**vor**<br>**of**<br>**HC**<br>**No**<br>**rilsk**<br>**I**,and the<br>third period ending in a 1-0 score in favor of Kristall<br>Saratov. The game ended in normal time,<br>**with**<br>**a**<br>**to**<br>**tal**<br>**score**<br>**of**<br>**3**<br>**-**<br>**1**<br>**in**<br>**fa**<br>**vor**<br>**of**<br>**HC**<br>**No**<br>**rilsk**<br>**I**.<br>In the VHL 23/24 season, Kristall Saratov hosted<br>HC Norilsk. The frst period saw two quick goals<br>from Norilsk,<br>**with**<br>**one**<br>**in**<br>**the**<br>**sec**<br>**ond**<br>**I** and third<br>periods as well, resulting in a dominant<br>**5**<br>**-**<br>**0**<br>**vic**<br>**tory**<br>**I**.<br>Kristall Saratov managed to score a single goal in<br>the third period, but it was not enough to overcome<br>Norilsk’s lead. The fnal score was 1-5 in favor of<br>HC Norilsk.<br>In a match from the VHL 23/24 season, Kristall<br>Saratov faced off against HC Norilsk. The game<br>took place on November 28, 2023, at 15:30. Despite<br>a goal from Kristall Saratov in the third period, HC<br>Norilsk emerged victorious with a fnal score of 5-1.<br>The away team dominated the game, scoring 2 goals<br>in the frst period and 3 goals in the second period.<br>Kristall Saratov struggled to keep up, resulting in a<br>clear win for HC Norilsk.|



Table 10: An example of the model outputs on ice_hockey (see the input in Table 10). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation_images/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation.pdf-0026-00.png)


Figure 10: An example of an input from owid (see the outputs in Table 11). 

|**Model**|**Human annotations**(_E_hum)|**GPT-4 annotations**(_E_gpt)|
|---|---|---|
|**Llama2**<br>**Mistral**<br>**Zephyr**<br>**GPT-3.5**|Life expectancy at birth in Djibouti has increased<br>signifcantly over the past few decades, with a no-<br>table increase from 40.5665 years in 1950 to 62.3049<br>years in 2021.<br>This chart displays the life expectancy at birth in<br>Djibouti from 1950 to 2021. The data shows a steady<br>increase in life expectancy over the years, with<br>**a**<br>**sig**<br>**-**<br>**nif**<br>**i**<br>**cant**<br>**jump**<br>**in**<br>**the**<br>**1970s**<br>**and**<br>**1980s**<br>**I**. In 2021,<br>the average life expectancy at birth in Djibouti was<br>62.3 years.<br>This line graph illustrates the evolution of life ex-<br>pectancy at birth in Djibouti from 1950 to 2021.<br>Initially, the average number of years that a new-<br>born could expect to live was around 40.5 years in<br>1950, and it gradually increased over the following<br>decades, reaching 62.3 years in 2021. The trend<br>shows a steady improvement, with some fuctuations<br>in the<br>**1980s**<br>**I** and 1990s. The data presented here is<br>based on the sex- and age-specifc death rates pre-<br>vailing at the time of birth for a specifc year in a<br>given country, territory, or geographic area.<br>The chart shows the life expectancy at birth in Dji-<br>bouti from 1950 to 2021. The life expectancy has<br>steadily increased over the years, from 40.57 years in<br>1950 to 62.30 years in 2021, with some fuctuations<br>in between. The data refects the average number of<br>years a newborn could expect to live if exposed to<br>the prevailing death rates at the time of birth.|Life expectancy at birth in Djibouti has increased<br>signifcantly over the past few decades, with a no-<br>table increase from 40.5665 years in 1950 to 62.3049<br>years in 2021.<br>This chart displays the life expectancy at birth in Dji-<br>bouti from 1950 to 2021. The data shows a<br>**steady**<br>**in**<br>**crease**<br>**I** in life expectancy over the years, with a<br>**sig**<br>**nif**<br>**i**<br>**cant**<br>**jump**<br>**in**<br>**the**<br>**1970s**<br>**and**<br>**1980s**<br>**I**. In 2021,<br>the average life expectancy at birth in Djibouti was<br>**62.3**<br>**years**<br>**I**.<br>This line graph illustrates the evolution of life ex-<br>pectancy at birth in Djibouti from 1950 to 2021.<br>Initially, the average number of years that a new-<br>born could expect to live was<br>**around**<br>**40.5**<br>**years**<br>**in**<br>**1950**<br>**I**,and it gradually increased over the following<br>decades, reaching<br>**62.3**<br>**years**<br>**in**<br>**2021**<br>**I**.<br>**The**<br>**trend**<br>**shows**<br>**a**<br>**steady**<br>**im**<br>**prove**<br>**ment**<br>**M**,with some fuctu-<br>ations in the 1980s and 1990s. The data presented<br>here is based on the sex- and age-specifc death rates<br>prevailing at the time of birth for a specifc year in a<br>given country, territory, or geographic area.<br>The chart shows the life expectancy at birth in Dji-<br>bouti from 1950 to 2021.<br>**The**<br>**life**<br>**ex**<br>**pectancy**<br>**has**<br>**steadily**<br>**in**<br>**creased**<br>**over**<br>**the**<br>**years**<br>**I**,from<br>**40.57**<br>**years**<br>**in**<br>**1950**<br>**I** to<br>**62.30**<br>**years**<br>**in**<br>**2021**<br>**I**,with some<br>fuctuations in between. The data refects the aver-<br>age number of years a newborn could expect to live<br>if exposed to the prevailing death rates at the time of<br>birth.|



Table 11: An example of the model outputs on owid (see the input in Table 11). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation_images/acl_2024_Beyond_Traditional_Benchmarks_Analyzing_Behaviors_of_Open_LLMs_on_Data-to-Text_Generation.pdf-0027-00.png)


Figure 11: An example of an input from wikidata (see the outputs in Table 12). 

|**Model**|**Human annotations**(_E_hum)|**GPT-4 annotations**(_E_gpt)|
|---|---|---|
|**Llama2**<br>**Mistral**<br>**Zephyr**<br>**GPT-3.5**|Driven Brands is an American<br>**fran**<br>**chise**<br>**hold**<br>**ing**<br>**NC**<br>company that<br>**owns,**<br>**op**<br>**er**<br>**ates,**<br>**and**<br>**fran**<br>**chises**<br>**NC**<br>automotive services businesses in the United States.<br>**The**<br>**com**<br>**pany**<br>**was**<br>**founded**<br>**in**<br>**1996**<br>**and**<br>**is**<br>**head**<br>**-**<br>**quar**<br>**tered**<br>**in**<br>**Char**<br>**lotte,**<br>**North**<br>**Car**<br>**olina.**<br>**It**<br>**op**<br>**er**<br>**-**<br>**ates**<br>**sev**<br>**eral**<br>**brands,**<br>**in**<br>**clud**<br>**ing**<br>**Maaco,**<br>**Meineke,**<br>**Econo**<br>**Lube,**<br>**and**<br>**CARSTAR**<br>**NC**.<br>Driven Brands is a business operating in the automo-<br>tive services industry in the United States of Amer-<br>ica.<br>Driven Brands is a business operating in the auto-<br>motive services industry based in the United States<br>of America. As an instance of the broader category<br>of businesses, Driven Brands<br>**pro**<br>**vides**<br>**goods**<br>**or**<br>**ser**<br>**vices**<br>**to**<br>**cus**<br>**tomers**<br>**in**<br>**ex**<br>**change**<br>**for**<br>**f**<br>**nan**<br>**cial**<br>**re**<br>**mu**<br>**ner**<br>**a**<br>**tion**<br>**NC**.<br>Driven Brands is a business based in the United<br>States of America, operating within the automotive<br>services industry.|Driven Brands is an American franchise holding<br>company that owns, operates, and franchises auto-<br>motive services businesses in the United States. The<br>company was<br>**founded**<br>**in**<br>**1996**<br>**NC** and is<br>**head**<br>**quar**<br>**-**<br>**tered**<br>**in**<br>**Char**<br>**lotte,**<br>**North**<br>**Car**<br>**olina**<br>**NC**. It operates<br>several brands, including Maaco, Meineke, Econo<br>Lube, and CARSTAR.<br>Driven Brands is a business operating in the automo-<br>tive services industry in the United States of Amer-<br>ica.<br>Driven Brands is a business operating in the auto-<br>motive services industry based in the United States<br>of America.<br>**As**<br>**an**<br>**in**<br>**stance**<br>**of**<br>**the**<br>**broader**<br>**cat**<br>**e**<br>**-**<br>**gory**<br>**of**<br>**busi**<br>**nesses,**<br>**O** Driven Brands provides<br>**goods**<br>**or**<br>**ser**<br>**vices**<br>**I** to customers in exchange for fnancial<br>remuneration.<br>Driven Brands is a business based in the United<br>States of America, operating within the automotive<br>services industry.|



Table 12: An example of the model outputs on wikidata (see the input in Table 12). 

|||**Incorrect**|**Incorrect**|**Not**|**Checkable**|**Misleading**|**Misleading**|**Other**|**Other**|**All categories**|**All categories**||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|#**Tok.**|
||openweather|4.27|6.99|2.23|1.93|0.58|0.20|0.16|0.26|7.24|9.38|166.7|
||gsmarena|0.67|1.96|0.87|0.31|0.20|0.24|0.09|0.10|1.83|2.61|90.3|
|**Llama 2**|ice_hockey|1.47|1.98|0.99|0.90|0.09|0.00|0.12|0.05|2.67|2.93|63.3|
||owid|1.30|2.26|0.58|0.15|0.32|0.13|0.07|0.00|2.27|2.54|60.3|
||wikidata|0.15|0.74|1.59|1.24|0.08|0.03|0.07|0.04|1.89|2.05|38.5|
||openweather|5.65|8.76|2.47|0.41|1.36|0.50|0.53|0.04|10.01|9.71|193.5|
||gsmarena|0.51|2.02|1.10|0.94|0.30|0.25|0.14|0.02|2.05|3.23|146.3|
|**Mistral**|ice_hockey|1.47|1.79|0.92|0.76|0.14|0.08|0.13|0.10|2.66|2.73|92.4|
||owid|2.42|3.11|0.40|0.18|0.29|0.27|0.07|0.02|3.18|3.58|91.1|
||wikidata|0.10|0.45|0.72|0.40|0.13|0.19|0.38|0.31|1.33|1.35|51.0|
||openweather|4.22|7.39|1.01|0.37|0.34|0.41|0.16|0.00|5.73|8.17|130.9|
||gsmarena|0.33|2.24|1.04|1.00|0.23|0.48|0.11|0.01|1.71|3.73|142.8|
|**Zephyr**|ice_hockey|0.89|1.57|0.61|0.20|0.11|0.04|0.07|0.01|1.68|1.82|83.1|
||owid|1.68|2.64|0.49|0.16|0.22|0.30|0.10|0.01|2.49|3.11|85.2|
||wikidata|0.09|0.36|0.72|0.29|0.12|0.21|0.36|0.21|1.29|1.07|48.1|
||openweather|1.57|4.34|0.57|0.15|0.38|0.53|0.05|0.00|2.57|5.02|112.8|
||gsmarena|0.20|1.64|0.80|1.42|0.21|0.34|0.17|0.01|1.38|3.41|129.5|
|**GPT-3.5**|ice_hockey|0.81|0.76|0.46|0.17|0.07|0.04|0.09|0.01|1.43|0.98|84.4|
||owid|0.64|1.87|0.25|0.02|0.17|0.27|0.01|0.00|1.07|2.16|62.2|
||wikidata|0.05|0.18|0.36|0.16|0.06|0.10|0.05|0.10|0.52|0.54|35.7|



Table 13: The average _numbers of errors per output_ for each domain (lower is better). We also include the average number of tokens per output in the rightmost column. See Table 3 for aggregated results. 

|||**Incorrect**|**Incorrect**|**Not Checkable**|**Not Checkable**|**Misleading**|**Misleading**|**Other**|**Other**|**All categories**|**All categories**|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|_E_hum|_E_gpt|
||openweather|76%|98%|71%|68%|30%|14%|12%|22%|90%|100%|
||gsmarena|38%|86%|49%|24%|18%|16%|5%|7%|74%|94%|
|**Llama 2**|ice_hockey|70%|77%|51%|55%|7%|0%|10%|5%|89%|94%|
||owid|70%|92%|37%|14%|24%|12%|6%|0%|88%|93%|
||wikidata|12%|47%|79%|63%|8%|2%|5%|4%|87%|89%|
||openweather|78%|99%|61%|16%|38%|18%|17%|4%|92%|100%|
||gsmarena|33%|89%|54%|54%|19%|17%|12%|2%|73%|99%|
|**Mistral**|ice_hockey|74%|83%|62%|52%|12%|8%|9%|7%|88%|97%|
||owid|75%|96%|27%|8%|24%|25%|6%|2%|86%|97%|
||wikidata|8%|34%|44%|29%|10%|17%|24%|27%|67%|72%|
||openweather|82%|99%|45%|25%|23%|25%|14%|0%|90%|100%|
||gsmarena|22%|91%|47%|52%|19%|30%|9%|1%|61%|99%|
|**Zephyr**|ice_hockey|54%|78%|39%|17%|11%|4%|6%|1%|76%|87%|
||owid|70%|93%|31%|9%|20%|26%|7%|1%|85%|95%|
||wikidata|6%|29%|49%|22%|8%|18%|22%|18%|66%|66%|
||openweather|64%|98%|32%|12%|23%|25%|5%|0%|75%|99%|
||gsmarena|15%|74%|42%|58%|13%|19%|12%|1%|57%|97%|
|**GPT-3.5**|ice_hockey|64%|50%|30%|12%|7%|4%|8%|1%|76%|57%|
||owid|43%|89%|13%|2%|14%|24%|1%|0%|57%|90%|
||wikidata|4%|14%|27%|14%|6%|9%|5%|9%|38%|36%|



Table 14: The percentage of _outputs containing at least one error_ for each domain (lower is better). See Table 4 for aggregated results. 

