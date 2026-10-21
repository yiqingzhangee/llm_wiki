_2024-10-14_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities_images/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities.pdf-0001-02.png)


## **Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities** 

**Lichang Chen**[1,3] **, Hexiang Hu**[1] **, Mingda Zhang**[1] **, Yiwen Chen**[2] **, Zifeng Wang**[2] **, Yandong Li**[1] **, Pranav Shyam**[1] **, Tianyi Zhou**[3] **, Heng Huang**[3] **, Ming-Hsuan Yang**[1] **and Boqing Gong**[1] 

1Google DeepMind, 2Google, 3University of Maryland, College Park 

**We introduce Omni** × **R, an evaluation suite designed to benchmark state-of-the-art Omni-modality Language Models (OLMs), such as GPT-4o and Gemini. Evaluating OLMs, which integrate multiple modalities such as text, vision, and audio, presents unique challenges. Particularly, the user message might often consist of multiple modalities, such that OLMs have to establish holistic understanding and reasoning across modalities to accomplish the task. Existing benchmarks are limited to single-modality or dualmodality tasks (e.g., image+text or video+text), overlooking comprehensive multi-modal assessments of model reasoning. To address this, Omni** × **R offers two evaluation variants: (1) Omni** × **Rsynth: a synthetic dataset generated automatically by translating text into multiple modalities—audio, images, video, and hybrids (** **`Omnify!` ). (2) Omni** × **Rreal: a real-world dataset, manually curated and annotated by experts, for evaluating cross-modal reasoning in natural settings. Omni** × **R presents a unique evaluation towards assessing OLMs over a diverse mix of modalities, such as a question that involves video, audio, and text, providing a rigorous cross-modal reasoning testbed than any existing benchmarks. Our experiments find that all state-of-the-art OLMs struggles with Omni** × **R questions that require integrating information from multiple modalities to answer. Further analysis highlight differences in reasoning behavior and underscoring the challenges of omni-modal AI alignment.** 

_Keywords: Omni-Eval, Omni-Reasoning, Omni-modality Language Models_ 

## **1. Introduction** 

Recent advances in Omni-modality Language Models (OLMs) (Gemini-Team, 2024b; OpenAI, 2024b) has pushed the boundaries of AI by enabling a more comprehensive understanding of real-world inputs across diverse modalities, e.g., text, vision, audio, (Akbari et al., 2021; Gan et al., 2020; Lu et al., 2019; Zellers et al., 2021) and generating outputs that are more aligned with human communications (Gao et al., 2024; Lu et al., 2024; Zhang et al., 2024). 

However, the evaluation of these sophisticated OLMs presents unique challenges. While traditional benchmarks (LMSYS-team, 2024) have predominantly focused on models that handle single or dual modalities, such as vision-language or video-text pairs, they fail to capture the complexities that arise when multiple modalities are involved. In real-world scenarios, user inputs are rarely confined to one or two modalities. Instead, they often consist of diverse combinations of text, images, videos, and audio, necessitating a holistic understanding and reasoning across information presented in these modalities for OLMs to effectively perform tasks. This mismatch between existing evaluation methods and the multimodal capabilities of state-of-the-art OLMs has left a significant gap in the assessment of these models. 

One common flaw in existing OLMs is their inconsistent behavior when presented with the same question in different modalities or mixtures of modalities. Figure 1 presents an example on the Gemini 1.5 Flash (Gemini-Team, 2024a) (similar behaviour also observed in other OLMs, see Section 3.2 for analysis). Particularly, when the same math question is presented in different modalities, such as rendered as image input, or spoke out as the audio input, the model produces varying responses 

_Corresponding author(s): Lichang Chen<bobchen@cs.umd.edu>, Hexiang Hu<hexiang@google.com>, and Boqing Gong <bgong@google.com>. Author Contributions are detailed in Appendix A._ 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

**Text Input: [Prompt]** Think step by step then output the answer in the format of "The answer is (X)" The total cost of producing x cameras is C(x) = 2 at the end. + x^3. What is the average cost if 10 cameras are made? What is the marginal cost of producing 10 cameras? **[Response(to Text)]** To find the average cost of producing 10 cameras given the cost function $C(x) = 2 + x^3$: 1. Calculate the total cost of producing 10 cameras:  $C(10) = 2 + (10)^3 = 2 + 1000 = 1002$  2. The average cost is the total cost divided by the A. Average Cost: $1000/unit, Marginal Cost: $100 number of units produced:   $\text{Average Cost} = \frac{C(10)}{10} = \frac{1002}{10} = B. Average Cost: $300/unit, Marginal Cost: $100 **OmniLM** 100.2$. The closest option is: The answer is J. C. Average Cost: $300/unit, Marginal Cost: $1000 D. Average Cost: $102/unit, Marginal Cost: $301 …… **[Response(to Video)]** The answer is C. **or [Response(to Audio)]** The total cost of producing 10 cameras is $C(10) = 2 + 10^3 = **Video Input:** 1002$. The average cost is $1002 / 10 = 102$ per unit. The marginal cost is the derivative of the cost function, which is $C'(x) = 3(2 + x)^2$. The marginal cost of producing 10 cameras is $C'(10) = 3(2 + 10)^2 = 363$. The answer is D. **Audio Input: [Response(to Image)]** The total cost of producing x cameras is given by the function $C(x) = 2 + x^3$. The average cost of producing 10 cameras is the total cost divided by the number of cameras, or $C(10)/10 = (2 + 10^3)/10 = 1002/10 = \$100.20/camera$ **Image Input:** The marginal cost is the derivative of the total cost function, or $C'(x) = 3x^2$. The marginal cost of producing 10 cameras is $C'(10) = 3(10)^2 = \$300/camera$. Therefore, the correct answer is: **The answer is E**. 

Figure 1 | **Reasoning Behavior of a OLM Varies across Modalities** . Taking Gemini-1.5-Flash as an example, on text question, the reasoning behaviour is expected and the answer is correct. When the same question is rendered to an image, the model generate a reasonable reasoning but incorrect answer. On the video or audio representation of the same question, the model generates no reasoning and produces incorrect answers. 

that exhibit significant performance discrepancies, _i.e._ , different reasoning bevhiours or different answers. This observation indicates a lack of robust cross-modal information integration and reasoning capabilities in OLMs. Such inconsistency not only undermines the reliability of these models but also highlights the limitations of current evaluation benchmarks that do not adequately assess performance across diverse modality combinations. 

To bridge this critical evaluation gap, we introduce Omni×R, an evaluation suite specifically designed to benchmark the reasoning performance of OLMs across a wide range of modalities. Unlike existing benchmarks that are limited to a maximum of two modalities, Omni×R provides a comprehensive testbed that includes complex modality combinations such as `video` + `audio` + `text` and `image` + `audio` + `text` , offering a more rigorous and holistic evaluation of these models’ capabilities. Specifically, Omni×R contains two subsets of the data: 

- **Omni** × **Rsynth** : a synthetic reasoning dataset constructed with a scalable and low-cost automatic method ( _i.e._ , `Omnify!` ) to translate information embedded in text to various modalities — audio, images, video, and hybrids of them. 

- **Omni** × **Rreal** : a real-world reasoning dataset manually collected and annotated with expert annotators, for evaluating cross-modal reasoning in the realistic distribution. 

In construction of Omni×Rsynth, `Omnify!` translates text-based inputs into various other modalities, such as images, audio, and video, as well as their hybrid combinations, using programmatic text rendering services, programmatic video construction pipeline, and state-of-the-art text-to-speech service. This scalable synthetic dataset ensures a diverse and robust dataset that challenges OLMs to demonstrate their cross-modal reasoning abilities. Meanwhile, Omni×Rreal develops a realistic test environment for evaluating omnimodal reasoning. Particularly, we crawled 100 Youtube videos whose topics are related to math, physics, chemistry and coding, and manually curate, convert and annotate the quiz questions from those videos, ensuring that each question is associated with multiple variants, each in one modality or a hybrid of many modalities. With both complementary subsets, Omni×R allows us to better assess how well OLMs can reason across different modalities and integrate information in a way that mirrors human-like understanding. 

Our evaluation of state-of-the-art OLMs on Omni×R has yielded several important findings. 

2 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities_images/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities.pdf-0003-01.png)


**----- Start of picture text -----**<br>
𝑂𝑚𝑛𝑖𝑋𝑅<br>!"#$%<br>6 Modalities 14 Categories 1400 Samples<br>Text, Image, Audio, Video, Math, Physics, Chemistry,  100 Examples in each<br>Video+Audio,  Computer Science… category. 1400 exmaples<br>Image + Audio in each modality.<br>𝑂𝑚𝑛𝑖𝑋𝑅&'()<br>20%<br>25%<br>Math 4 Modalities<br>Coding Text, Image, Audio, Video,<br>Chemistry #Test Samples: 400<br>Physics #Each Modality: 100<br>30% 25%<br>**----- End of picture text -----**<br>


Figure 2 | The overview of Omni×Rsynth and Omni×Rreal. 

Notably, Omni×R is the first benchmark that quantitatively measured the aforementioned omni-modal behaviour discrepancy, especially in scenarios requiring deep reasoning across multiple modalities. Moreover, we also observe that some simple prompting strategy that exploits the underlying data creation logic in Omni×Rsynth, _i.e._ , `Extract the information and Then Answer` (ETA prompting), could significantly improve every omini-modality language model’s behaviour consistency and final answer accuracy on Omni×Rsynth. These results suggest that the main struggle of current model is to establish a holistic understanding across modality, where the need to integrate and reason across different forms of data becomes crucial. When evaluated on Omni×Rreal, where the information across modality is naturally distributed and blended with noises, OLMs can no longer rely on a simple prompting strategy to alleviate the omnimodal behaviour inconsistency, indicating the demand of further training for future omnimodal language models. 

## **2. Omni** × **R Benchmark** 

In this section, we introduce `Omnify!` a scalable and low-cost automatic method designed to translate text into various modalities, including audio, image, video, and combinations thereof. The overarching goal of `Omnify!` is to build up a scalable method to generate omni-modality data while keeping information the same across them for evaluating OLMs’ reasoning capabilities across modalities. We construct the Omni×R benchmark in two subsets: (1) Omni×Rsynth: a synthetic omni-modal reasoning evaluation dataset derived from applying `Omnify!` on the MMLU-Pro (Wang et al., 2024). (2) Omni×Rreal: a real-world omni-modal reasoning evaluation derived from Youtube, which is then processed and annotated by human experts. 

## **2.1. Omnify!** 

**Text to image.** Though there are many ways to convert text into images, like using image generation models (e.g., Imagen-3 (Baldridge et al., 2024), DALLE-3 (OpenAI, 2024a)), however, the seemingly appealing text-to-image generation models make it challenging to control quality; they cannot ensure the generation contains all the information we need to answer a question. Before figuring out how to judge the quality of and information in the generated images, it is not viable to use image generators to scale up the mapping from text to images. Since our main goal is to evaluate models’ reasoning capability, we start from the simplest approach in this work: rendering a canvas and then write the words on it. Given the images as input, we expect the models can achieve the same performance as they read text in this ideal scenario, where no extra noises, information losses, or variations are 

3 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities_images/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Video Interleaved  Audio<br>( Video + Audio and Image + Audio)<br>Question:<br>Euglena Image<br>Text Question Question: Euglena is a common green flagellate<br>is … protozoan found in fresh water ponds. Describe briefly<br>the method of locomotion, nutrition, and asexual<br>Omnify! reproduction in this organism.<br>**----- End of picture text -----**<br>


Figure 3 | We propose `Omnify!` to create the synthetic omni-modality evaluation data from the original text benchmarks. 

introduced by the text-to-image mapping process. Specifically, we use PIL[1] to create a new image with a white background and the text is drawn onto the image with black color. The engineering details/efforts can be found in Appendix I. 

**Text to Audio** We initially attempted to use Google Text-to-Speech[2] (TTS) for text-to-audio conversion. However, we encountered challenges with the mathematical equations. To address this, we developed a two-step process. First, we convert the original text, if it contains mathematical equations, into a format that is easy to speak orally. The details for the conversion could be found in Table 7. Then, we use a TTS engine to generate the audio, which contains the full information of the original text question. 

**Text to Video** Like text-to-image generation models, there exist Sora (Brooks et al., 2024) and Veo (Google, 2024) we could leverage to map text to videos. However, they would incur the same problems as described in the text to image: quality control, time consumption, and computational cost. The main objective with videos here is to evaluate a model’s capabilities on understanding a video input, which is a series of images from a model’s view, and then reasoning to solve the problems. We fulfill this objective again using a simple approach to generating the video data from text as follows. Based on our image generation process, we render a series of images where each image contains one or several words from the text. We ensure that the information in the text is fully translated to the video. The input text is split into individual words first. Then we use OpenCV to create a video writer object with a specified frame rate, i.e., 1 FPS, and frame size (300x100 pixels). Each word is converted into an image using the text-to-image method. Finally, these images are combined sequentially to create video frames. 

## **2.2. Omni** × **Rsynth: Scalable Synthetic Omini-modal Reasoning Evaluation** 

Our initial choices of the text benchmark for `Omnify!` are Arc-Challenge (Clark et al., 2018) and GSM8K (Cobbe et al., 2021), but we identify the potential data contamination problems on these two benchmarks as Gemini-1.5-pro (Gemini-Team, 2024a) can achieve over 99% on GSM8K (results are shown in Table 9). It is very likely that contaminated OLMs just capture the part of the information they need from the video/audio questions and use their ‘memory’ to give correct answers, which cannot reflect the actual reasoning ability of the models. Thus, we choose MMLU-Pro (Wang et al., 2024), which is augmented from MMLU with ten options per question and released in June after the Gemini-1.5-Pro-001[3] release, as the text benchmark to `Omnify!` . In this way, we minimize the contamination influence, enabling a more accurate study of OLMs’ omni-reasoning. We randomly 

> 1 `https://github.com/python-pillow/Pillow` 

> 2 `https://cloud.google.com/text-to-speech?hl=en` 

> 3 `https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions` 

4 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities_images/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities.pdf-0005-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities_images/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities.pdf-0005-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities_images/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities.pdf-0005-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities_images/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities.pdf-0005-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities_images/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities.pdf-0005-05.png)


**----- Start of picture text -----**<br>
(a) Coding (b) Physics (c) Calculus (d) Chemistry<br>**----- End of picture text -----**<br>


Figure 4 | Visualization of Examples in the Omni×R-Real set. 

sample 100 questions from each of the 14 categories in MMLU-Pro to construct Omni×Rsynth. Some examples for Audio and Video modalities are available[4] . 

## **2.3. Omni** × **Rreal: High-Quality Real-world Omini-modal Reasoning Evaluation** 

We crawl the video data from youtube and then transcribe it into different modalities to develop a realistic set as a valuable addition to the Omni×R. The details about how we collect and process the data are shown as follows: 

1. **Video:** We select four categories that require dense reasoning in real-world scenarios: Mathematics, Coding, Physics, and Chemistry. Videos are sourced from popular educational channels, such as MIT OpenCourse. Two human annotators, spend approximately 30 hours each to review 100 videos (200 in total) and identify those containing non-trivial questions that demand substantial reasoning to solve. From these, 100 videos are carefully selected to construct a high-quality set, Omni×Rreal. Each video clip is curated based on the following criteria: (1) it must contain one or more key frames that provide all the necessary information to solve the question; (2) the clip should exclude the answer to maintain the challenge; (3) some misleading or irrelevant frames are intentionally included to assess the model’s robustness in reasoning. 

2. **Image:** We manually find the key frame(s) which contain the question information. It should be noted that in some cases, there might be several frames containing the relevant information, where we will crawl two or three frames and merge them together into one image. 

3. **Text:** Five human annotators transcribe the text from the video with the help of the tools, e.g., Gemini. All the open-ended generation questions are transferred into multiple choice questions to make the benchmark easy-to-use. 

4. **Audio:** The original audio will be checked first, which is extracted from the video we crawled. If it contains all the information for OLMs to answer the question, then we will just keep and use it. However, there are many cases where the audio does not contain the enough information for answering the questions, e.g., the instructor shows a slide and asks “solve the problems in the slide”, where the problem is shown in image. In that scenario, we will use the same method in `Omnify!` to transfer the transribed text into audio by Google TTS. 

## **3. Experiments and Findings** 

## **3.1. Experiment Setup** 

**Models.** We mainly test three series of models: Gemini (Gemini-Team, 2024a), _i.e._ , Gemini-1.5-Pro, and Gemini-1.5-Flash, OpenAI-GPT (OpenAI, 2024c), _i.e._ , GPT-4o and GPT-4o-mini, AnthropicClaude (Anthropic, 2024), _i.e._ , Claude-3-Opus, Claude-3-Sonnet, Claude-3-Haiku. More details about the test models are shown in Appendix E. 

> 4 `https://anonymous.4open.science/r/OmnixR-Examples-7961/` 

5 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

**CoT Prompting.** The standard setting in MMLU-Pro (Wang et al., 2024) is to use Chain-of-Thought(CoT) prompting to elicit the reasoning ability of the OLMs for a more comprehensive evaluations. Following them, we use CoT with 0-shot, as our standard setting, _i.e._ , the prompt used for evaluation is “Think step by step then output the answer in the format of “The answer is (X)” at the end.” 

**Extract-Then-Answer (ETA) Prompting.** In addition, we employ Extract-Then-Answer (ETA) prompting, leveraging the benchmark’s inherent structure. This method involves first extracting the textual content and then using the OLMs’ language capabilities for reasoning to provide answers based on the transcriptions. To prevent potential hackings on Omni×R, we transparently demonstrate this approach in our benchmark, aiming for a comprehensive evaluation of OLMs. Specifically, the prompt ’Please extract the text from image/audio/videos’ instructs the OLMs to function as text extractors. The extracted text from this initial step is subsequently fed back into the same OLM with Chain-of-Thought (CoT) prompting to obtain the final answer. Consequently, the model’s performance reflects two key abilities: OCR/Transcription and Text Reasoning." 

**Video/Audio/Image.** We first process the video to 1-fps to meet the requirements for both the Gemini and GPT models. For testing with Claude, we used the API available before August 10th, which only supported a maximum of 5 image inputs, so video evaluations were not conducted. The GPT-4o API supports 250 images input at the maximum, so any additional frames were dropped in the evaluation. In contrast, Gemini had no issues with the video modality and could handle all frames as input. Image processing is the modality that all models support most effectively, allowing comprehensive testing across all OLMs. Notably, Gemini is the only model supporting audio input. 

**Answer Extraction:** We use the model to extract the answers. Since the regex parsing may affect the performance, we sacrifice the API cost to trade in the excellent extraction. 

|||**Gemini 1.5**|**Gemini 1.5**|||**Claude**|**Claude**|||||**GPT**||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Pro<br>Flash|||Opus||Sonnet||Haiku|||4o|4o-mini||
|Modality|Perf.|Δ<br>Perf.|Δ|Perf.|Δ|Perf.|Δ|Perf.|Δ|Perf.|Δ|Perf.|Δ|
|Text|77.5|-<br>69.9|-|77.7|-|77.4|-|72.5|-|71.5|-|72.6|-|
|Image|57.3|20.2↓36.3|33.6↓|26.9|50.8↓18.8||58.6↓9.9||62.6↓|60.1|11.4↓48.5||24.1↓|
|Audio|56.6|20.9↓53.9|16.0↓|-|-|-|-|-|-|-|-|-|-|
|Video|36.3|41.2↓15.1|54.8↓|-|-|-|-|-|-|53.1|18.4↓18.6||54.0↓|
|**Extract-Then-Answer (ETA)**|||**Prompting**|||||||||||
|Image|73.5|4.0↓68.1|1.8↓|62.6|15.1↓48.1||29.3↓43.2||29.3↓|66.7|4.8↓58.4||14.2↓|
|Audio|69.9|7.6↓63.6|6.3↓|-|-|-|-|-|-|-|-|-|-|
|Video|48.6|28.9↓42.8|27.1↓|-|-|-|-|-|-|25.0|46.5↓59.3||13.3↓|



Table 1 | **Results on Omni** × **Rsynth** show different mixed modalities evaluations, including text, image, audio, video. Each modality (Image/Audio/Video) combines two input sources: the ’Question’ provided by the respective image, audio, or video modality, and the ’CoT instruction’ provided by the text The numbers in red font, following the downward arrows, shows the drops compared to the pure text input. 

## **3.2. Main Results on Omni** × **R synth** 

We show the main experimental results on ominified MMLU-Pro in Table 1. 

**Model Comparison.** Gemini-1.5-Pro demonstrates the most versatile performance across all modalities, showing results in text, image, audio, video tasks. Claude models struggle with image tasks and lack audio and video capabilities. GPT models show a balanced performance, with GPT-4o performing particularly well in direct image and video compare to Gemini and Claude. Generally, 

6 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

|||**Gemini 1.5**|**Gemini 1.5**|**Gemini 1.5**|||**Claude**|**Claude**||||**GPT**|**GPT**||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Pro||Flash||Opus||Sonnet||Haiku||4o||4o-mini||
|Modality|Perf.|Δ|Perf.|Δ|Perf.|Δ|Perf.|Δ|Perf.|Δ|Perf.|Δ|Perf.|Δ|
|Text|86|-|80|-|78|-|66|-|65|-|85|-|75|-|
|Image|78|8↓|65|15↓|41|34↓|39|27↓|33|8↓|79|6↓|63|12↓|
|Audio|71|15↓|64|14↓|-|-|-|-|-|-|-|-|-||
|Video|64|22↓|53|27↓|-|-|-|-|-|-|73|12↓|66|9↓|
|**Extract-Then-Answer**|||**(ETA) Prompting**||||||||||||
|Image|79|7↓|65|15↓|63|15↓|52|14↓|51|14↓|79|6↓|70|5↓|
|Audio|55|31↓|51|29↓|-|-|-|-|-|-|-|-|-|-|
|Video|71|15↓|73|7↓|-|-|-||-|-|66|19↓|63|12↓|



Table 2 | **Results on Omni** × **Rreal** shows similar behaviour discrapancy of OLMs as indicated in results on the Omni×Rsynth. Interestingly, we also observe that simple prompting strategy (ETA prompting) is not as effective as it was on Omni×Rsynth, possibly due to the natural noise and redundancy in real-world image, video, and audio data. 

larger models outperform their smaller counterparts across modalities, _e.g._ , Pro _>_ Flash, Opus _>_ Haiku). But interestingly, GPT-4o-mini outperforms GPT-4o in text and video with ETA prompting. For video tasks using ETA prompting, GPT-4o’s performance inconsistencies led us to examine the model’s responses to the extraction, we found that in over 46.8% test samples, the detailed analysis can be found in Appendix H, GPT-series models cannot extract the text from video, which we identify as the primary cause for the significant performance drop compared to CoT prompting. Regarding the text modality, two possible explanations emerge: first, MMLU-Pro was released before GPT-4o-mini, suggesting that OAI might have optimized for it. Second, since our dataset uses a subset sampled from MMLU-Pro, inherent biases may have influenced the results. 

**Modality Analysis.** Text is the most mature modality across all models, with consistently high scores (ranging from 69.9% to 77.7%). Image modality shows significant variability, with direct task performance ranging from 9.9% (Claude Haiku) to 60.1% (GPT-4o). However, ETA prompting on image generally improves performance for all models, particularly for Claude (e.g., Opus improves from 18.8% to 62.6%). The improvement justifies the inclusion of ETA prompting as a standard in our benchmark to prevent potential manipulation. Audio modality, only available for Gemini models, shows moderate performance with notable improvement via ETA prompting. Video modality presents the most challenges, especially for the small models, _i.e._ , Gemini-1.5-Flash, and GPT-4o-mini. 

There are also additional results on Arc-Challenge and GSM8k benchmarks shown in Table 9 with different modality input, _i.e._ , text, image, audio, video. Though the models are likely to be data contaminated on these benchmarks, the performance drops are still significant on image/video/audio compared to the pure text. 

## **3.3. Main Results on Omni** × **Rreal** 

The results on the realistic set generally align with those from the synthetic set, showing significant drops in performance across audio, image, and video tasks compared to the text. One difference here is that performance on video does not drop a large margin compared to that in the synthetic set. Though the video is noisy than it is in the synthetic data, we can still capture one key frame and answer the question according to that key frame which largely reduces the difficulties, compared to the synthetic scenario, if the model can find the main frame in the video. Another interesting finding is that ETA prompting does not consistently improve performance; for example, there are performance drops in audio tasks with ETA prompting compared to CoT on both Gemini-Flash and 

7 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

Gemini-Pro. These findings confirm that our synthetic set effectively simulates real-world scenarios in a scalable, cost-efficient way, serving as a valuable sanity check for OLMs’ omni-modality reasoning capabilities. 

**Key Takeaways.** We summarize the following interesting takeaways from our experiments: 

1. Multi-modal capabilities vary significantly across models, with Gemini 1.5 Pro showing the most broad support and balanced performance across all modalities. 

2. Gaps still exists on other modalities compared to the text modality even just in such easy perception test scenarios. Significant room for improvement exists in video processing across all models, presenting opportunities for future development. 

3. ETA prompting generally improves performance on Omni×Rsynth but OLMs can no longer solely rely on it for Omni×Rreal, indicating the necessity of the further alignment on omni-modality. 

4. There’s a clear trade-off between model size and performance, but smaller models (e.g., GPT-4omini) can sometimes outperform larger counterparts in specific tasks. 

5. Our Omni×Rsynth could be a good simulating set for the real-world scenarios, as the results on Omni×Rreal match the results in the Omni×Rsynth. 

## **4. Mixed Modalities** 

||Input Modality<br>`Question`<br>`CoT Prompt`|Gemini-Pro<br>Perf.<br>Δ|Gemini-Flash<br>Perf.<br>Δ|
|---|---|---|---|
||Text<br>Text<br>Text<br>Video<br>Text<br>Audio<br>Text<br>Image<br>Image + Audio<br>Text<br>Video + Audio<br>Text|77.5<br>-<br>76.1<br>1.4↓<br>74.1<br>3.4↓<br>74.1<br>3.4↓<br>61.8<br>15.7↓<br>40.1<br>37.4↓|69.9<br>-<br>66.8<br>3.1↓<br>68.3<br>1.6↓<br>66.9<br>3.0↓<br>49.1<br>20.8↓<br>25.9<br>44.0↓|



Table 3 | The results of more complex mixed modalities on Omni×Rsynth. We use the Δ to denote the performance drops from the text modality. 

**Text to Mixed Modalities.** In addition to the types of the `Omnify!` described in Section 2.1, our method could also be applied to generating interleaved modalities to better simulate more complex real-world scenarios, where the information is included in different modalities and requires a model to reason across the modalities to solve a problem. For example, an instructor can write down an equation on the blackboard and say “compute the derivative” in a Calculus lecture. Scenarios like this example require a model to jointly use image perception and audio understanding process the question, reason across the visual and audio modalities, and then provide a response. Using our `Omnify!` , we seamlessly integrate different modalities and create test samples with interleaved modalities, _i.e._ , “Video + Audio”, and “Image + Audio”, to Omni×Rsynth, which captures a more authentic user experience where multiple senses are engaged simultaneously. To be specific, We transfer the question into video and all the options are transferred for Audio, to get the modality, “Video + Audio", while CoT prompting remains in text form to maintain the model’s reasoning ability across different modalities. 

**Transferring CoT prompt to other modalities.** All the CoT prompting is in text for all the previous test cases. Here, we convert the CoT prompt into different modalities while keeping the others, _i.e._ , questions and options in MMLU-Pro intact. 

**Results.** As shown in Table 3, there is a noticeable decline in performance when transitioning from 

8 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

text to mixed-modality tasks. For example, both the Pro and Flash models perform significantly worse in the "Video + Audio" scenario, achieving scores of 40.1 and 25.9, respectively. This indicates that handling mixed modalities presents a significant challenge, likely due to the increased complexity of integrating video and audio information. For Audio/Image/Video CoT, the model generally treats these inputs as noise or irrelevant context, having minimal impact on the final results, as performance approaches that observed with text-based CoT. We focus on evaluating the Gemini-series models since only Gemini supports audio inputs. 

## **5. Analysis** 

## **5.1. Omni-Modality Reasoning Behaviour Analysis** 

After investigating the responses, we find that in omni-modality cases, Gemini-1.5-Flash models can only output very short answers though prompted to CoT before giving the answers, which is quite different from the reasoning behaviour in the pure-text. An example in Figure 1 shows the different behaviours among modalities, which intrigues us to have a quantitative analysis of the reasoning paths. We write a simple regex, detecting if the model output starts with "the answer/response is (*.)", with the rule, the total number of words should be less than 40, to evaluating whether the models’ output contain the reasoning path. The results are shown in Table 4. 

|Path(%)<br>**Gemini 1.5**<br>**Claude**<br>Modality<br>Pro<br>Flash<br>Sonnet<br>Opus<br>Haiku|**GPT**<br>4o<br>4o-mini|
|---|---|
|Text<br>98.9<br>89.1<br>100<br>100<br>98.6<br>100<br>100<br>Image<br>93.2<br>54.3<br>100<br>100<br>72.8<br>100<br>100<br>Video<br>91.3<br>23.4<br>-<br>-<br>-<br>99.1<br>95.7<br>Audio<br>94.0<br>82.3<br>-<br>-<br>-<br>-<br>-||



Table 4 | The percentage of the model outputs containing the reasoning paths on Omni×Rsynth. 

Our analysis reveals that smaller models tend to produce reasoning paths less frequently for image, video, and audio inputs. Notably, for complex modalities like video, Gemini-1.5-Flash generates reasoning paths for only 23.4% of test examples, substantially lower than Gemini-1.5-Pro. Among the modalities, audio inputs elicit reasoning paths most similarly to text, while video inputs show the lowest rate of reasoning path generation. GPT-series models demonstrate excellent performance in producing reasoning paths across available modalities. However, these results underscore the significant challenges remaining in cross-modal reasoning. Given that models are expected to exhibit reasoning abilities, they should ideally output reasoning paths consistently across all input modalities. 

## **5.2. Visual/Video Formats influences Perception Precision** 

## _**5.2.1. Image**_ 

We first analyze how formats affect the performance on images. We show images with two different text formats in Figure 5. The lower image has a compact format, where the options are not spaced out; instead, they are presented in a continuous, inline format separated by periods. Compared to it, each option in the upper image is listed separately, making it easy to read, with letters (A to J) clearly aligned before each option. The results of CoT and ETA prompting with two different formats of images are shown in Table 6. The overall trend here is that with better format, we could significantly improve the performance across all the tested models. ETA prompting also boosts the performance for the both formats in general. For all the other models, the performance can be significantly improved when comparing BF with ETA, only the GPT-4o being an outlier. 

We further analyze transcription accuracy using the Character Error Rate (CER), a standard metric for assessing text recognition performance, especially in OCR tasks. A CER of 0 indicates perfect 

9 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

accuracy, with higher values reflecting more errors. Details of the CER calculation are provided in Appendix J, and results are shown in Table 5. The results reveal that GPT-4o’s OCR performance is largely format-independent, whereas other models exhibit considerable format sensitivity, explaining the pronounced improvements seen with ETA prompting for all models except GPT-4o when format is enhanced. 

|**Gemini 1.5**<br>**Claude**<br>Pro<br>Flash<br>Opus<br>Sonnet<br>Haiku|**GPT**<br>4o<br>4o-mini|
|---|---|
|Image<br>0.11<br>0.10<br>0.19<br>0.28<br>0.34<br>0.11<br>0.12<br>Better Image<br>0.06<br>0.03<br>0.05<br>0.18<br>0.26<br>0.11<br>0.11||



Table 5 | The CER of different models on two different formats images. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities_images/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities.pdf-0010-04.png)


**----- Start of picture text -----**<br>
Question: The relatives of a group of pelicans from the same species that separated from each other because of an unsuccessful<br>migration are reunited 150 years later and find that they are unable to produce offspring. This is an example of<br>Options: A. gene flow.<br>B. temporal isolation.<br>C. disruptive selection.<br>D. founder effect.<br>E. genetic drift.<br>F. sexual selection.<br>G. sympatric speciation.<br>H. habitat fragmentation.<br>I. bottleneck effect.<br>J. allopatric speciation<br>Question: The relatives of a group of pelicans from the same species that separated from each other because of an unsuccessful<br>migration are reunited 150 years later and find that they are unable to produce offspring. This is an example of  Options: A. gene<br>flow. B. temporal isolation. C. disruptive selection. D. founder effect.  E. genetic drift. F. sexual selection. G. sympatric speciation.<br>H.habitat fragmentation. I. bottleneck effect. J. allopatric speciation<br>**----- End of picture text -----**<br>


Figure 5 | We include two figures to illustrate which is a better format image. The upper one is the image with better format. The lower one is the image with the original format. 

## _**5.2.2. Video**_ 

We create different types of videos, one word per frame, several words per frame, etc. Our ablations reveal that increasing the number of words per frame generally leads to improved performance for both Gemini-Flash and Gemini-Pro models under both testing promptings, CoT and ETA prompting. This trend suggests that providing more context within each frame aids in the models’ understanding and processing of the video content and narrow the gaps between images and videos. 

## **6. Related Work** 

**Large Foundational Models.** GPT-4o (OpenAI, 2024b), Gemini (Gemini-Team, 2024a) both claim their models having omni-modality capabilities, but actually OAI’s model does not support audio(no audio access via APIs)/video(only 250 frames and the videos should be separated manually before feeding into the model) while Gemini can take very long videos and has good Audio support. 

10 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

|`Prompt`|**Gemini 1.5**<br>Pro<br>Flash|**Claude**<br>Opus<br>Sonnet<br>Haiku|**GPT**<br>4o<br>4o-mini|
|---|---|---|---|
|Text<br>`CoT`<br>77.5<br>69.9<br>77.7<br>77.4<br>76.5<br>71.5<br>72.6||||
|Image<br>`CoT`<br>57.3<br>36.3<br>26.9<br>18.8<br>9.9<br>60.1<br>48.5<br>Better Image<br>`CoT`<br>64.67.3↑43.67.3↑33.5<br>6.6↑28.910.1↑<br>19.1<br>9.2↑65.55.4↑52.13.6↑||||
|Image<br>`ETA`<br>68.7<br>61.3<br>36.4<br>26.6<br>24.9<br>66.7<br>58.4<br>Better Image<br>`ETA`<br>73.54.8↑68.16.8↑62.626.2↑48.121.5↑<br>43.218.3↑66.90.2↑61.73.3↑||||



Table 6 | The ablations: image with better format. BF: better format. The blue font denotes the performance gain of the better image compared to the original image format. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities_images/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities.pdf-0011-03.png)


**----- Start of picture text -----**<br>
Flash CoT Flash ETA Pro CoT Pro ETA<br>60<br>50<br>40<br>30<br>20<br>10<br>1 2 4 8<br>Words Per Frame<br>842 . 3 . 648 . 534 . 142 . 587 . 355 . 463 . 359 . 582 . 947 . 614 .<br>115 . 36 175 . 239 . 127 .<br>Performance (%)<br>**----- End of picture text -----**<br>


Figure 6 | Video ablation study: Model performance with different words per frame. Pro and Flash denotes Gemini-1.5-Pro-001 and Gemini-1.5-Flash-001, respectively. 

Claude (Anthropic, 2024) can be viewed as a vision-language model (Bordes et al., 2024) since it has capabilites to take image but no audio or video support. There are also other open-sourced vision language models, but they are mostly supporting only two modalities, e.g., the vision-language models like LLaMA-3.1 and 3.2 (Meta, 2024), Pixtral (Mistral, 2024), LLaVA (Liu et al., 2023a,b); Audio-LLM like GAMA (Ghosh et al., 2024), LTU (Gong et al., 2023a,b), and SALMONN (Tang et al., 2024). It is hard to judge them on our benchmark, since the main idea behind our evaluations are that we expect the model has cross-modality reasoning and would like to encourage the model improving their cross-modal reasoning, only vision/audio/video would not get a comprehensive results. We would expect the open-sourced community to release real OLMs in the future and we will update the results accordingly. 

**Video/Audio/Image Evaluation benchmarks.** Omnibench (Li et al., 2024b) specifically aimed at evaluating OLMs’ tri-modal, i.e., text, vision, and audio, processing capabilities with human-annotated tasks. Compared to it, OmnixR emphasizes the omni-modality reasoning evaluations with both human-annotated realistic set and scalable synthetic set. MMMU (Yue et al., 2024a), MMMU-Pro (Yue et al., 2024b), CMMMU (Ge et al., 2024) focuses on evaluating vision-language models across various college-level disciplines with highly heterogeneous image types, emphasizing expert-level perception and reasoning across text-image pairs while LMSYS-Vision (LMSYS-team, 2024) evaluates the instruction-following of the large vision-language models (Chen et al., 2023, 2024; Liu et al., 2023a; Yang et al., 2024). Compared to them, OmnixR has larger scope on evaluating OLMs on cross-modality reasoning, not only vision input, but audio, video, and mixed modalities such as image + audio. AiShell-1, AiShell-2 (Du et al., 2018), Clotho-AQA (Lipping et al., 2022) are audio understanding benchmarks, providing extensive and high-quality real-world audio data for Mandarin ASR and audio question answering. MVBench (Li et al., 2024a) focuses on temporal reasoning across 20 challenging video tasks, Video-Bench (Ning et al., 2023) assesses Video-LLMs across video- 

11 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

exclusive, knowledge-based, and decision-making tasks, while MMBench-Video (Fang et al., 2024) offers a long-form, multi-shot evaluation of LVLMs with 609 videos and 2,000 human-annotated QA pairs across 26 fine-grained capabilities. In OmnixR, we also include long video in both synthetic and realistic scenarios and we also have mixed-modality evals including video + audio. 

## **7. Conclusion** 

In this paper, we introduced Omnify!, a scalable and cost-efficient approach for generating multi-modal data from text, facilitating the construction of diverse and challenging test scenarios for omni-modal language models (OLMs). Using this method, we developed Omni×Rsynth, a synthetic omni-modal reasoning evaluation dataset derived from MMLU-Pro, as well as Omni×Rreal, a real-world omnimodal reasoning dataset based on YouTube content. Our comprehensive evaluations reveal that OLMs experience substantial performance drops when confronted with complex multi-modal inputs, particularly in tasks that demand cross-modality reasoning. Notably, we observed that smaller models, _e.g._ , Gemini-1.5-Flash, are less adept at producing reasoning paths for image, video, and audio inputs compared to text, underscoring the inherent challenges in cross-modal reasoning. The evaluation results underscore the necessity for enhanced training strategies to address the complexities of omnimodal tasks. To sum up, Omni×R stands as a critical benchmark for guiding future advancements in OLMs, providing a foundation for measuring progress toward more human-aligned and truly omni-modal AI systems. 

## **References** 

- H. Akbari, L. Yuan, R. Qian, W.-H. Chuang, S.-F. Chang, Y. Cui, and B. Gong. VATT: Transformers for multimodal self-supervised learning from raw video, audio and text. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, editors, _Advances in Neural Information Processing Systems_ , 2021. URL `https://openreview.net/forum?id=RzYrn625bu8` . 

- Anthropic. Claude: An ai assistant by anthropic, 2024. URL `https://www.anthropic.com/ claude` . Accessed: 2024-09-21. 

- J. Baldridge, J. Bauer, M. Bhutani, N. Brichtova, A. Bunner, K. Chan, Y. Chen, S. Dieleman, Y. Du, Z. Eaton-Rosen, et al. Imagen 3. _arXiv preprint arXiv:2408.07009_ , 2024. 

- F. Bordes, R. Y. Pang, A. Ajay, A. C. Li, A. Bardes, S. Petryk, O. Mañas, Z. Lin, A. Mahmoud, B. Jayaraman, et al. An introduction to vision-language modeling. _arXiv preprint arXiv:2405.17247_ , 2024. 

- T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, C. Ng, R. Wang, and A. Ramesh. Video generation models as world simulators. 2024. URL `https://openai.com/research/video-generation-models-as-world-simulators` . 

- Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu, B. Li, P. Luo, T. Lu, Y. Qiao, and J. Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. _arXiv preprint arXiv:2312.14238_ , 2023. 

- Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024. 

- P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018. 

12 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

- K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- J. Du, X. Na, X. Liu, and H. Bu. Aishell-2: Transforming mandarin asr research into industrial scale, 2018. 

- X. Fang, K. Mao, H. Duan, X. Zhao, Y. Li, D. Lin, and K. Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. _arXiv preprint arXiv:2406.14515_ , 2024. 

- Z. Gan, Y.-C. Chen, L. Li, C. Zhu, Y. Cheng, and J. Liu. Large-scale adversarial training for visionand-language representation learning. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, _Advances in Neural Information Processing Systems_ , volume 33, pages 6616– 6628. Curran Associates, Inc., 2020. URL `https://proceedings.neurips.cc/paper_files/ paper/2020/file/49562478de4c54fafd4ec46fdb297de5-Paper.pdf` . 

- P. Gao, R. Zhang, C. Liu, L. Qiu, S. Huang, W. Lin, S. Zhao, S. Geng, Z. Lin, P. Jin, K. Zhang, W. Shao, C. Xu, C. He, J. He, H. Shao, P. Lu, H. Li, and Y. Qiao. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. _ArXiv_ , abs/2402.05935, 2024. URL `https://api.semanticscholar.org/CorpusID:267547619` . 

- Z. Ge, D. Xinrun, C. Bei, L. Yiming, L. Tongxu, Z. Tianyu, Z. Kang, C. Yuyang, X. Chunpu, G. Shuyue, Z. Haoran, Q. Xingwei, W. Junjie, Y. Ruibin, L. Yizhi, W. Zekun, L. Yudong, T. Yu-Hsuan, Z. Fengji, L. Chenghua, H. Wenhao, and F. Jie. Cmmmu: A chinese massive multi-discipline multimodal understanding benchmark. _arXiv preprint arXiv:2401.20847_ , 2024. 

- Gemini-Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024a. URL `https://blog.google/technology/ai/ google-gemini-next-generation-model-february-2024/` . 

- Gemini-Team. Gemini: A family of highly capable multimodal models, 2024b. URL `https://arxiv. org/abs/2312.11805` . 

- S. Ghosh, S. Kumar, A. Seth, C. K. R. Evuru, U. Tyagi, S. Sakshi, O. Nieto, R. Duraiswami, and D. Manocha. Gama: A large audio-language model with advanced audio understanding and complex reasoning abilities, 2024. URL `https://arxiv.org/abs/2406.11768` . 

- Y. Gong, A. H. Liu, H. Luo, L. Karlinsky, and J. Glass. Joint audio and speech understanding. In _2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)_ . IEEE, Dec. 2023a. doi: 10.1109/asru57964.2023.10389742. URL `http://dx.doi.org/10.1109/ASRU57964.2023. 10389742` . 

- Y. Gong, H. Luo, A. H. Liu, L. Karlinsky, and J. Glass. Listen, think, and understand. _arXiv preprint arXiv:2305.10790_ , 2023b. 

- Google. Veo: Google’s most capable generative video model. 2024. URL `https://deepmind. google/technologies/veo/` . 

- K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Lou, L. Wang, and Y. Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark. In _2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , volume abs/2204.14198, page 22195–22206. IEEE, June 2024a. doi: 10.1109/cvpr52733.2024.02095. URL `http://dx.doi. org/10.1109/CVPR52733.2024.02095` . 

13 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

- Y. Li, G. Zhang, Y. Ma, R. Yuan, K. Zhu, H. Guo, Y. Liang, J. Liu, J. Yang, S. Wu, X. Qu, J. Shi, X. Zhang, Z. Yang, X. Wang, Z. Zhang, Z. Liu, E. Benetos, W. Huang, and C. Lin. Omnibench: Towards the future of universal omni-language models, 2024b. 

- S. Lipping, P. Sudarsanam, K. Drossos, and T. Virtanen. Clotho-aqa: A crowdsourced dataset for audio question answering, 2022. 

- H. Liu, C. Li, Y. Li, and Y. J. Lee. Improved baselines with visual instruction tuning, 2023a. 

- H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. In _NeurIPS_ , 2023b. 

LMSYS-team. Multimodal arena. 2024. URL `https://lmsys.org/blog/ 2024-06-27-multimodal/` . 

- J. Lu, D. Batra, D. Parikh, and S. Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks, 2019. URL `https://arxiv.org/abs/1908.02265` . 

- P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In _International Conference on Learning Representations (ICLR)_ , 2024. 

Meta. The llama 3 herd of models, 2024. URL `https://arxiv.org/abs/2407.21783` . 

Mistral. Pixtral, 2024. URL `https://docs.mistral.ai/capabilities/vision/` . 

- M. Ning, B. Zhu, Y. Xie, B. Lin, J. Cui, L. Yuan, D. Chen, and L. Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language models. _arXiv preprint arXiv:2311.16103_ , 2023. 

OpenAI. Dalle 3. _https://openai.com/index/dall-e-3/_ , 2024a. 

OpenAI. Hello gpt-4o! _OpenAI Research_ , 2024b. URL `https://openai.com/index/ hello-gpt-4o/` . Accessed: 2024-09-21. 

- OpenAI. Gpt-4o mini: advancing cost-efficient intelligence, 2024c. URL `https://openai.com/ index/gpt-4o-mini-advancing-cost-efficient-intelligence/` . Accessed: 2024-0921. 

- C. Tang, W. Yu, G. Sun, X. Chen, T. Tan, W. Li, L. Lu, Z. MA, and C. Zhang. SALMONN: Towards generic hearing abilities for large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. URL `https://openreview.net/forum?id=14rn7HpKVk` . 

- Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, T. Li, M. Ku, K. Wang, A. Zhuang, R. Fan, X. Yue, and W. Chen. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark, 2024. 

- A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J. Tang, J. Wang, J. Yang, J. Tu, J. Zhang, J. Ma, J. Yang, J. Xu, J. Zhou, J. Bai, J. He, J. Lin, K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M. Xue, N. Ni, P. Zhang, P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S. Wang, S. Bai, S. Tan, T. Zhu, T. Li, T. Liu, W. Ge, X. Deng, X. Zhou, X. Ren, X. Zhang, X. Wei, X. Ren, X. Liu, Y. Fan, Y. Yao, Y. Zhang, Y. Wan, Y. Chu, Y. Liu, Z. Cui, Z. Zhang, Z. Guo, and Z. Fan. Qwen2 technical report, 2024. 

14 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

- X. Yue, Y. Ni, T. Zheng, K. Zhang, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, C. Wei, B. Yu, R. Yuan, R. Sun, M. Yin, B. Zheng, Z. Yang, Y. Liu, W. Huang, H. Sun, Y. Su, and W. Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In _2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , volume 32, page 9556–9567. IEEE, June 2024a. doi: 10.1109/cvpr52733.2024.00913. URL `http://dx.doi. org/10.1109/CVPR52733.2024.00913` . 

- X. Yue, T. Zheng, Y. Ni, Y. Wang, K. Zhang, S. Tong, Y. Sun, B. Yu, G. Zhang, H. Sun, Y. Su, W. Chen, and G. Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark, 2024b. 

- R. Zellers, X. Lu, J. Hessel, Y. Yu, J. S. Park, J. Cao, A. Farhadi, and Y. Choi. Merlot: Multimodal neural script knowledge models, 2021. 

- R. Zhang, D. Jiang, Y. Zhang, H. Lin, Z. Guo, P. Qiu, A. Zhou, P. Lu, K.-W. Chang, P. Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? _arXiv preprint arXiv:2403.14624_ , 2024. 

## **A. Author Contributions** 

- **Lichang** devotes to revising the idea and constructing the Omni×Rsynth, finishing most of the code and experiments. 

- **Hexiang** proposed the initial idea for Omni-Eval via the observations on image reasoning behavior inconsistency of Gemini-Flash models. 

- **Hexiang** , **Boqing** , **Yiwen** , **Zifeng** , **Mingda** , and **Yandong** contributed to the Omni×Rreal. 

- **Hexiang** , **Mingda** , **Yandong** , **Boqing** , and **Tianyi** attend the discussion regularly and provide useful feedback/suggestion for the project. 

- **Pranav** contributes to the ablation study: images with better format and video ablations. 

- **Heng** provided the university tuition support to Lichang. 

- **Ming-Hsuan** and **Boqing** are project leads @Google Deepmind. 

## **B. Acknowledgements** 

We thank the useful feedbacks from talented researchers@Google Deepmind: 

- **Quoc V. Le** , **Xinyun Chen** : Thanks for providing useful feedbacks about the reasoning and irrelevant contents. 

- **Prajit Ramachandran** , **Yiling Huang** , **Yichong Xu** : Thanks for insights and future works on synthetic audio evals/benchmarks. 

- **Micheal Chang** , **Yao Fu** : Thanks for the discussions of the real-world applications of Omni×R. 

- **Jean-Baptiste Alayrac** , **Fangyu Liu** : Helpful discussions for video/image evals and possibility to include the data into pretrain corpus. 

- **Yong Cheng** , **Yanping Huang** , **Ruibo Liu** : Helpful discussion on synthetic data vs. real-world data, and how to reduce data contamination on evaluations. 

- **Kelvin Guu** , **Aida Amini** : thanks for the support and give Lichang time to wrap up his previous intern project. 

## **C. Convert Math into Spoken Version** 

For the math equations in the questions, we prompt Gemini-1.5-Pro to convert them into the version which can be spoken orally. The prompt we used is detailed in Table 7. We also show an example to explain the transformation: the TTS is hard to read the original question in Table 8 but it can handle the converted text. 

15 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

[Prompt] Please transform all the equations in the text into the format that is easy to speak out orally. [Original text] 

Please first output a single line of the text in the format "The transformed text is xxx" 

Table 7 | The oral conversion prompt designed for Text-to-Audio transfer. 

[Original Question] For what values of _𝑥_ is it true that _𝑥_[2] − 5 _𝑥_ − 4 ≤ 10? Express your answer in interval notation. 

[Converted Text] The spoken version: For what values of x is x squared minus five x minus four less than or equal to ten? express your answer in interval notation. 

Table 8 | An example of the conversion from the original question into the easily spoken text. 

## **D. Categories in MMLU-Pro** 

There are 14 categories in MMLU-Pro, including Math, Physics, Chemistry, Law, Engineering, Other, Economics, Health, History, Psychology, Business, Biology, Philosophy, Computer Science. 

## **E. Model Settings/Details** 

The version of the Geminis we used in this paper are Gemini-1.5-Pro-001 and Gemini-1.5-Flash-001. The version of the OpenAI models we used are gpt-4o-2024-05-13, and gpt-4o-mini-2024-07-18. The verison of the Claude models we used are claude-3-sonnet@20240229, claude-3-opus@20240229, claude-3-haiku@20240307. 

The Gemini safety settings we used for video, audio, and images are shown as follows: 

1 `# Safety Setting` 2 `generative_models .SafetySetting(` 3 `category= generative_models .HarmCategory. HARM_CATEGORY_DANGEROUS_CONTENT ,` 4 `threshold= generative_models . HarmBlockThreshold .BLOCK_ONLY_HIGH ,` 5 `),` 6 `generative_models .SafetySetting(` 7 `category= generative_models .HarmCategory.HARM_CATEGORY_HARASSMENT ,` 8 `threshold= generative_models . HarmBlockThreshold .BLOCK_ONLY_HIGH ,` 9 `),` 10 `generative_models .SafetySetting(` 11 `category= generative_models .HarmCategory.HARM_CATEGORY_HATE_SPEECH ,` 12 `threshold= generative_models . HarmBlockThreshold .BLOCK_ONLY_HIGH ,` 13 `),` 14 `generative_models .SafetySetting(` 15 `category= generative_models .HarmCategory. HARM_CATEGORY_SEXUALLY_EXPLICIT ,` 16 `threshold= generative_models . HarmBlockThreshold .BLOCK_ONLY_HIGH ,` 17 `),` 

BLOCK_ONLY_HIGH is the loosest setting we can use for public Gemini APIs for video, audio, and images. BLOCK_ONLY_NONE is the loosest setting we can use for text, so we change all the Safety Settings for language into BLOCK_ONLY_NONE. 

For response generation, we follow the commonly used settings, temperature=0.7, top_p=0.9, and output_length=1024, for all the models, _i.e._ , Gemini, Claude, GPT models. 

16 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

## **F. Results on Arc-Challenge & GSM8K** 

We also evaluate Gemini models on ARC-Challenge dataset and GSM8K test set. The results are shown in Table 9. 

|**Benchmark**|**Accuracy (%)**|
|---|---|
||Gemini-1.5-Pro<br>Gemini-1.5-Flash|
|**ARC-Challenge**<br>Text<br>95.5<br>92.3<br>Image<br>79.5<br>75.0<br>Audio<br>91.1<br>88.0<br>Video<br>63.6<br>40.3||
|**GSM8K**<br>Text<br>99.1<br>96.3<br>Image<br>92.5<br>87.9<br>Audio<br>86.8<br>90.8<br>Video<br>80.3<br>63.1||



Table 9 | Performance of Gemini Models Across Different Modalities on ARC-Challenge and GSM8K Benchmarks 

## **G. Omni** × **R Statistics** 

||Min|Max|Mean|
|---|---|---|---|
|Video<br>28F<br>552F<br>117.2F<br>Audio<br>7.2s<br>251.3s<br>32.3s||||



Table 10 | Statistics for Video and Audio on the Omni×Rsynth. F: Frames, s: seconds. 

||Min|Max|Mean|
|---|---|---|---|
|||||
|Video<br>Audio|30F<br>10s|1326F<br>1326s|255.6F<br>139.7s|



Table 11 | Statistics for Video and Audio on the Omni×Rreal. F: Frames, s: seconds. 

## **H. Analyze the extraction** 

We manually check the data first, and then find the patterns that the extraction failure have are mostly "unable to process", "can’t extract", "I’m sorry", and "unable to extract". So we use these four patterns to check if the answers contain one of them, and calculate the percentage of the model answers which do not output the extractions when prompted as "Please extract the text from video." 

17 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

## **I. Details of the text-to-image conversion** 

We use the Python Imaging Library (PIL) to create a new image with a white background and the text is drawn onto the image with black color. The tricky part here is that the most commonly used font "times.ttf" does not support the Unicode well and will encounter the error when we try to convert the Unicode text, e.g., special mathematical symbols such as ∞, ≥, Π, Δ. Thus, our solution here is to have a look-up-table to replace these Unicode text with latex code before generating. The details about the look-up-table is shown in Appendix I.1. 

## **I.1. Look-up-table for unicode conversion** 

We show parts of look-up-table here due to the display issues. The full details about the look-up-table could be referred to our code. 

1 `’\u03b1 ’: r’$\alpha$ ’, # Alpha` 2 `’\u03b2 ’: r’$\beta$ ’, # Beta` 3 `’\u03b3 ’: r’$\gamma$ ’, # Gamma` 4 `’\u03b4 ’: r’$\delta$ ’, # Delta` 5 `’\u03c0 ’: r’$\pi$’, # Pi` 6 `’\u03c3 ’: r’$\sigma$ ’, # Sigma` 7 `’\u03c6 ’: r’$\phi$ ’, # Phi` 8 `’\u03c9 ’: r’$\omega$ ’, # Omega` 9 `’\u2211 ’: r’$\sum$ ’, # Summation` 10 `’\u220f ’: r’$\prod$ ’, # Product` 11 `’\u222b ’: r’$\int$ ’, # Integral` 12 `’\u0394 ’: r’$\Delta$ ’, # Capital Delta` 13 `’\u03a3 ’: r’$\Sigma$ ’, # Capital Sigma` 14 `’\u03a6 ’: r’$\Phi$ ’, # Capital Phi` 15 `’\u03a9 ’: r’$\Omega$ ’, # Capital Omega` 16 `’\u2190 ’: r’$\leftarrow$ ’, # Left arrow` 17 `"\u2014": r"-", # Em dash` 18 `"\u03a0": r"$\Pi$", # Capital Pi` 19 `"\u0302": r"$^$", # Circumflex accent` 20 `"\u2228": r"$\vee$", # En dash` 21 `"\u2013": r"--", # En dash` 22 `"\u2026": r"$\ldots$", # Ellipsis` 23 `"\u2283": r"$\supset$", # Superset` 24 `"\u201c": r"\"", # Left double quotation mark` 25 `"\u03c1": r"$\rho$", # Rho` 26 `"\u03b8": r"$\theta$", # Theta` 27 `"\u03bb": r"$\lambda$", # Lambda` 28 `"\u2225": r"$\parallel$", # Parallel` 29 `"\u2022": r"$\\ bullet$", # Bullet` 

## **J. CER Calculation** 

The CER is calculated by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities_images/iclr_2025_OmnixR_Evaluating_Omni-modality_Language_Models_on_Reasoning_across_Modalities.pdf-0018-08.png)


Where _𝑆_ = Number of substitutions, _𝐼_ = Number of insertion, _𝐷_ = Number of deletions, _𝑁_ = Total number of characters in the ground truth text. Since the model output will contain some addition text, e.g., starting with "The text in the image reads:", which will affect the final results, we will detect the "question:" by regex to eliminate the influence by that irrelevant text. 

18 

Omni R: Evaluating Omni-modality Language Models on Reasoning across Modalities 

[Question] Question: Find 4(3/4) % of $24.85. Options: A.$1.50 B.$1.10 C.$0.99 D.$1.25 E.$1.35 F.$1.18 G.$1.00 H.$0.93 I.$1.20 J.$1.05" 

[Prompt] Think step by step then output the answer in the format of "The answer is (X)" at the end. 

[Response(to Text)] To find 4 (3/4) % of $24.85, we will follow these steps: 

1. Convert the mixed number 4 (3/4) into an improper fraction: - 4 (3/4) = 4 + (3/4) = (4 * 4/4) + (3/4) = (16/4) + (3/4) = 19/4. 

2. Convert the fraction to a percentage: - (19/4) % = (19/4) * (1/100) = 19/400. 

3. Now, calculate 19/400 of $24.85: - First, find the value of $24.85 * (19/400): - $24.85 * (19/400) = $24.85 * 0.0475 (since 19/400 = 0.0475). 

4. Perform the multiplication: - $24.85 * 0.0475 = $1.179875. 

5. Round to two decimal places: - $1.179875 rounds to $1.18. 

Now, we check the options provided: A. $1.50 B. $1.10 C. $0.99 D. $1.25 E. $1.35 F. $1.18 G. $1.00 H. $0.93 I. $1.20 J. $1.05 The closest match is F. $1.18. Therefore, the answer is F. 

[Response(to Video)] 1. The question is asking to find 4/3 of $24.85. 

2. First, understand the mathematical expression: (4/3) * $24.85. 

3. Calculate the multiplication: - Divide $24.85 by 3 = $24.85 / 3 = $8.2833 (approx) - Multiply the result by 4 = $8.2833 * 4 = $33.1333 (approx) 

4. The closest option to $33.13 among the given options is $33.13. The answer is $33.13. [Response(to Image)] To find 3/4 of 24.85, follow these steps: Calculate[3] 4[of 24] _[.]_[85:][3] 4[×][ 24] _[.]_[85][ =][3][×][24] 4 _[.]_[85] , 3 × 24 _._ 85 = 74 _._ 55, Now divide by 4:[74] 4 _[.]_[55] = 18 _._ 6375, Round to the nearest cent: This gives us approximately 18 _._ 64. However, since we need to find 44[of that value, we can directly multiply:][3] 4[×][ 24] _[.]_[85][ =][ 18] _[.]_[6375, None of these options seem to] match the calculated value directly. The final answer considering the options is ‘The answer is not found among the options provided.’ 

Table 12 | Examples of different reasoning behaviour across modalities on GPT-4o. We use red font to mark the key errors in the reasoning path. The correct answer here is F. GPT-4o only respond correctly on the text modality input. 

19 

