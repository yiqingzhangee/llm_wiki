Pre-print. Under Review. 

# - VISUAL DESCRIPTION GROUNDING REDUCES HALLU CINATIONS AND BOOSTS REASONING IN LVLMS 

**Sreyan Ghosh**[1*] **Chandra Kiran Reddy Evuru**[1*] **Sonal Kumar**[1*] **Utkarsh Tyagi**[1] **Oriol Nieto**[2] **Zeyu Jin**[2] **Dinesh Manocha**[1] 1University of Maryland, College Park, USA, 2Adobe, USA 

## ABSTRACT 

Large Vision-Language Models (LVLMs) often produce responses that misalign with factual information, a phenomenon known as hallucinations. While hallucinations are well-studied, the exact causes behind them remain underexplored. In this paper, we first investigate the root causes of hallucinations in LVLMs. Our findings reveal that existing mitigation techniques primarily reduce hallucinations for visual recognition prompts—those that require simple descriptions of visual elements—but fail for cognitive prompts that demand deliberate reasoning. We identify the core issue as a lack of true visual perception in LVLMs: although they can accurately recognize visual elements, they struggle to fully interpret these elements in the context of the input prompt and effectively link this recognition to their internal knowledge, which is critical for reasoning. To address this gap, we introduce **Visual Description Grounded Decoding** (VDGD), a simple, robust, and _training-free_ method designed to enhance visual perception and improve reasoning capabilities in LVLMs. VDGD works by first generating a detailed description of the image and appending it as a prefix to the instruction. During response generation, tokens are sampled based on their KL divergence to the description, favoring candidates with lower divergence. Experimental results on multiple visual reasoning benchmarks and LVLMs demonstrate that VDGD consistently outperforms existing baselines 2% - 33%. Finally, we introduce VaLLu, a benchmark designed for comprehensive evaluation of the cognitive capabilities of LVLMs. Code is available here: https://github.com/Sreyan88/VDGD[1] 

## 1 INTRODUCTION 

Large Language Models (LLMs) pre-trained on web-scale text data with the next token prediction objective implicitly compress world knowledge in their parameters (Zhao et al., 2023). These models learn general-purpose representations, which can then be _aligned_ with the desired response characteristics (Zhang et al., 2024b). This step generally involves fine-tuning on instruction-response pairs, also known as _instruction tuning_ (IT) (Wei et al., 2022), and is followed by an optional step of _reinforcement learning from human feedback_ (RLHF) (Bai et al., 2022). Such aligned LLMs can be engaged through text-based inputs, or _prompts_ , directing them to perform various tasks, including those based on information retrieval and reasoning. 

Recent advancements in the research community have demonstrated success in multi-modal alignment, whereby fine-tuning LLMs with multi-modal instruction-response pairs enables them to execute tasks that conform to the text prompt as well as additional multi-modal inputs (Yin et al., 2023a). Since their introduction, the research community has extensively evaluated the capabilities of Large Vision-Language Models (LVLMs) across areas such as visual recognition (Wang et al., 2023a), perception (Fu et al., 2024), and reasoning (Yue et al., 2024). Researchers have made significant efforts to enhance these capabilities by refining model architectures and training techniques (Zhang et al., 2024a) or by implementing strategies to mitigate hallucinations[2] (Huang et al., 2023a). 

Among these approaches, _training-free_ hallucination mitigation techniques stand out as a costeffective and accessible method to enhance LVLM performance without the need for extensive 

> 1Corresponding Author: sreyang@umd.edu. * denotes equal technical contribution. 

> 2Hallucination refers to the mismatch between factual content and the model’s generated responses. Mitigation aims to reduce these discrepancies for more accurate outputs and improve task performance. 

1 

Pre-print. Under Review. 

computational resources. However, current hallucination mitigation techniques primarily focus on reducing hallucinations and improving the performance of visual recognition tasks (e.g., describing a scene, OCR, etc). Our extensive experiments show that these techniques fall short when applied to cognitive prompts requiring reasoning or knowledge extraction, indicating a significant gap in their ability to address hallucinations in more complex, reasoning-intensive scenarios. 

To this end, we propose **Visual Description Grounded Decoding** (VDGD), a simple and novel _training-free_ technique designed to improve LVLM reasoning by reducing hallucinations for cognitive prompts. Much like humans, who often write down their observations of an image in their own words and refer back to them while tackling complex reasoning tasks, we hypothesize that grounding the LVLM’s response generation in an explicit visual description can significantly aid its reasoning capabilities. Specifically, we first generate a description of the input image. Next, at every decoding step during auto-regressive response generation, we calculate the Kullback–Leibler divergence (KLD) between the top- _k_ tokens in the current logit and all the logits of the image description. Finally, to predict the next word, we sample from the plausible candidates according to their KLD to the description, where lower KLD is given higher preference. By continuously referencing its descriptive summary of the visual content, the LVLM maintains alignment between the visual input and its generated responses, thereby reducing hallucinations. To summarize, our contributions are as follows: 

1. We conduct extensive controlled experiments to investigate the causes of hallucinations in LVLMs, categorizing them into distinct types. We demonstrate that existing hallucination mitigation techniques are effective only for visual recognition (VR) prompts—specifically, those that involve object recognition and image descriptions—while proving ineffective for cognitive prompts that require reasoning. We further show that these techniques are biased towards reducing language hallucinations in real-world scenes but fail to address hallucinations in non-real-world contexts and other categories. 

2. Identifying the core issue, we reveal that LVLMs suffer from a critical visual perception gap. Although these models can recognize visual elements and possess the reasoning skills needed for factual response generation, they struggle to fully perceive the visual information in the context of the prompt, resulting in hallucinated outputs for cognitive reasoning tasks. 

3. We introduce **VDGD** , a novel and training-free method that enhances LVLM reasoning by grounding response generation in visual descriptions. We evaluate VDGD on X benchmarks with prompts that require deliberate reasoning and how VDGD significantly outperforms existing techniques, improving performance by 2% - 33% by reducing hallucinations. 

4. Finally, we introduce **VaLLu** , a meticulously curated benchmark designed to evaluate the cognitive capabilities of LVLMs. Comprising 1,500 carefully selected instances, VaLLu is constructed by refining existing benchmarks and rigorously removing noisy and misleading examples identified through our thorough analysis. VaLLu emphasizes a balanced mix of diverse cognitive tasks, focusing exclusively on questions that require open-ended responses. 

## 2 RELATED WORK 

**Large Vision-Language Models** In recent years, Large Vision-Language Models (LVLMs) have seen rapid advancements with the release of numerous new models. These developments have been driven by innovations in model architectures (Liu et al., 2023b; Ye et al., 2023; Wang et al., 2023b), training methods (Liu et al., 2023b), alignment techniques (Chen et al., 2024b), and data augmentation techniques (Liu et al., 2023b; Wang et al., 2024b). To assess the capabilities of these models, the research community has introduced a variety of benchmarks specifically designed to evaluate different aspects, including visual recognition and understanding in diverse scenarios (Liu et al., 2023a; Kim et al., 2022), reasoning (Lu et al., 2023; Yue et al., 2024), knowledge-based information retrieval (Hu et al., 2023), and other specialized tasks. 

**Reducing Hallucinations to Improve Response Quality:** Hallucination mitigation broadly aims to enhance the accuracy of model responses. While numerous studies have proposed methods to reduce hallucinations and improve response accuracy (Sicong Leng et al., 2023; Huang et al., 2023b; Yin et al., 2023b; Zhou et al., 2023b) and evaluated their effectiveness (Wang et al., 2023a; Guan et al., 2023), most focus on mitigating object hallucinations in visual recognition tasks. While these efforts are valuable, they fall short of assessing the true reasoning capabilities of LVLMs. Moreover, despite the growing focus on evaluation and mitigation, the underlying causes of hallucinations remain largely unexplored. 

2 

Pre-print. Under Review. 

## 3 HOW CLOSE ARE WE TO RELIABLE LVLM RESPONSES? 

**Experimental Setup for Analysis.** We evaluate six LVLMs—LLaVA-v1, LLaVA-v1.5, LLaVA-v1.6, mPLUG-Owl2, InternLM-X, and CogVLM—all built on a 7B parameter language model. The models are tested across seven benchmarks: AMBER (visual recognition), SynthDoG (OCR), MMMU (expert-level reasoning), MathVista and MATH-Vision (mathematical reasoning), MMC (chart understanding), and MME and HallusionBench. To address potential limitations in existing evaluation metrics, which may not fully capture reasoning skills and often rely on string-matching techniques, we employ a combination of expert human evaluation and LLM-as-a-Judge (using GPT-4-turbo-202404-09). This paradigm has been extensively followed by prior studies in LLM evaluation (Zhou et al., 2023a; Ghosh et al., 2024; Zheng et al., 2023) and hallucination evaluation (Sicong Leng et al., 2023; Yu et al., 2023; Liu et al., 2023c). We devise an evaluation prompt that penalizes hallucinations and assigns scores from 1 to 5 based on factual correctness (see Appendix G). This approach also helps us standardize scoring across benchmarks and has a higher correlation ( _≈_ 0.97%) to expert human evaluations (see Section D). Quantitative results for all graphs in the paper are in Appendix L. 

**Preliminaries.** As LVLMs advance and become more accessible, their application extends beyond merely generating image descriptions. Thus, to better understand visual perception and reasoning, we break down information processing by an LVLM into its constituent steps: 

Visual Recognition _−−−−−→_ Perception[Knowledge Extraction (optional)] _[ →]_[Reasoning (optional)] 

Figure 1 demonstrates that different prompts applied to a single image can assess various LVLM skills. Visual Recognition (VR) focuses on identifying visual elements and their relationships within the image, such as describing objects or specific details. Visual Perception (VP) extends beyond VR by interpreting and contextualizing these elements within the broader scene (Fu et al., 2024), essential for tasks requiring more than basic recognition. Prompts that demand knowledge-specific insights (also known as information-seeking prompts) engage in 

**Visual Recognition (VR) -** Describe this image. **Visual Perception (VP) -** Did the revenue increase from 2013 to 2015? **VP + Info-Seek -** Were there any regulation changes for gambling  between 2006-2015 in the country shown in the chart? **VP + Knowledge-Extraction + Reasoning -** What factors might have contributed to  the increase in casino gaming market revenue in the United Kingdom between 2006 and 2015? **VP + Reasoning -** Are there any limitations to what can be learned from this chart? 

Figure 1: Depending on the text instruction, an LVLM might be assessed on one or more different capabilities. 

knowledge extraction (KE) learned from pre-training. Finally, prompts requiring reasoning involve combining visual data, textual prompts, and extracted knowledge to generate a response. While recognition depends on the vision encoder, knowledge extraction, and reasoning rely on the foundational language model. VP, implicitly learned during alignment (Ghosh et al., 2024; Zhou et al., 2023a), bridges VR and higher cognitive functions, facilitating comprehensive understanding of the image in context to the prompt (Barrow et al., 1978; Torralba & Oliva, 2002; Hartley & Zisserman, 2003). 

## 3.1 DO HALLUCINATION MITIGATION TECHNIQUES IMPROVE COGNITIVE ABILITIES? 

Fig. 2 (left) compares the performance of LVLMs across 5 benchmarks. As we clearly see, while newer models that were trained using improved architectures and alignment training data boosted performance on the AMBER benchmark (for recognition), performance on other reasoning and information-seeking benchmarks remained stagnant. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0003-11.png)


**----- Start of picture text -----**<br>
AMBER AMBER<br>InfoSeek MMMU InfoSeek MMMU<br>0 1 2 3 4 5 0 1 2 3 4 5<br>SynthDoG MATH-Vista SynthDoG MATH-Vista<br>LLaVA-v1 CogVLM LURE OPERA<br>LLaVA-v1.5 LLaVA-v1.6 mPLUG-Owl2 InternLM-X LLaVA-v1.5 (Vanilla) VCD Woodpecker<br>**----- End of picture text -----**<br>


Fig. 2 (right) compares the perforFigure 2: (Left) Performance comparison of different LVLMs on mance of LLaVA-1.5 across the same various benchmarks. (Right) Performance comparison of different 5 benchmarks when employed with hallucination mitigation techniques applied to LLaVA-1.5. explicit hallucination mitigation techniques, with strategies ranging from improving model 

Figure 2: (Left) Performance comparison of different LVLMs on various benchmarks. (Right) Performance comparison of different hallucination mitigation techniques applied to LLaVA-1.5. 

3 

Pre-print. Under Review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Types of Visual Understanding Hallucination<br>2 Vision<br>Hallucination<br>The image  features a  picturesque scene The image  features a  man wearing a  life The image  features a  man riding a<br>of a leadinggrassy up  thehillside with a mountain. The dirtpath path is jacket, lake. He appearssitting in a  to be red and blackenjoying hiscanoe on a time on 4 skateboardthrill of the downdescent a  sandy.  Hehill is ,  enjoyingskillfully the<br>surrounded by a lush green field, and the the water,  possibly taking in the  beautiful IT Hallucination navigating the  slope,  showcasing his  talent<br>landscape is dotted with trees, creating a surroundings. The  canoe is  positioned in on the skateboard. The scene takes place on<br>serene and peaceful atmosphere.  In the the middle of the lake, with the man sitting a sunny day, with  the man  being the main<br>distance,  majestic mountains can be comfortably in  the boat.\n\n In the focus of the image. The sandy hill provides<br>seen, adding to the  beauty of the scene. background, there are  several cars parked a  unique and  challenging terrain for the<br>The path appears to be a popular spot for near the water's edge, suggesting that  this skateboarder to ride on.<br>hikers and nature enthusiasts, as there might be a popular spot for  boating and<br>are  several  people  scattered throughout other water-based activities. IT Dataset<br>the image, enjoying the  view and  the Unshifted:<br>Language outdoors. people grassy Instruction: What's happening in the scene?Response: floating on the water. The sailboat is situated towards the center of the image, surrounded by calm waters. Two birds, a goose and a duck, are  The image showcases a serene harbor scene with a sailboat  HallucinationStyle3 Marginal:Shifted:<br>Hallucination Base 65.58% 1.24% standing on the shore nearby, appearing to be watching the boat as it<br>1 moves along the water. Around the harbor, there are multiple vehicles,<br>Aligned 56.05% 26.59% including cars and trucks, parked near the water's edge.<br>**----- End of picture text -----**<br>


Figure 4: **Types of Visual Recognition Hallucinations.** We define Algo. 1 to divide VR hallucinations into 4 different categories automatically: Language, Vision, Style, and IT (explained further in Sec. 3.3). While language and vision hallucinations have been explored earlier, and methods to alleviate them have been proposed, we show for the first time that Style and IT hallucinations exist and existing methods fail to alleviate them. 

architectures to explicit response correction strategies, namely, Visual Contrastive Decoding (VCD) (Sicong Leng et al., 2023), OPERA (Huang et al., 2023b), Woodpecker (Yin et al., 2023b) and LURE (Zhou et al., 2023b). As we clearly see, while all these methods boost performance on AMBER, performance on other reasoning and information-seeking benchmarks remains stagnant. 

## 3.2 DO MITIGATION TECHNIQUES GENERALIZE BEYOND REAL-WORLD SCENES? 

As LVLMs are now being employed on a wide range of tasks, considerable research efforts are being made to evaluate the efficacy of LVLMs to reason on images beyond real-world scenes, i.e., charts (Liu et al., 2023a) or math problems (Lu et al., 2024). _However, how often do LVLMs hallucinate while describing images beyond real-world scenes?_ Fig 3 (left) compares the performance of various LVLMs across 5 benchmarks. MMMU and MathVista include images with mathematical figures and MMC includes charts from diverse sources. On the other hand, Synth- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0004-06.png)


**----- Start of picture text -----**<br>
AMBER AMBER<br>SynthDoG MMMU SynthDoG MMMU<br>0 1 2 3 4 5 0 1 2 3 4 5<br>MMC MATH-Vista MMC MATH-Vista<br>LLaVA-v1 CogVLM LURE OPERA<br>LLaVA-v1.5 LLaVA-v1.6 mPLUG-Owl2 InternLM-X LLaVA-v1.5 (Vanilla) VCD Woodpecker<br>**----- End of picture text -----**<br>


Figure 3: (Left) Performance comparison of different LVLMs on various benchmarks when prompted to only describe the image. (Right) Performance comparison of different hallucination mitigation techniques applied to LLaVA-1.5. 

DoG prompts the model for OCR and AMBER for image descriptions for real-world scenes. As we clearly see, compared to AMBER, models hallucinate more often when describing visuals beyond real-world scenes. Fig 3 (right) further shows that hallucination mitigation techniques improve performance on AMBER but rarely other benchmarks. 

## 3.3 ARE ALL HALLUCINATIONS EQUAL? 

**Overview.** In our detailed analysis of model responses on the AMBER dataset, we identified intriguing patterns in which hallucinations occur. Although the community has primarily focused on the broader topic of _object hallucinations_ (Liu et al., 2024b), our findings reveal that visual recognition hallucinations can be classified into several subcategories based on their specific occurrence patterns. Specifically, we perform the token-distribution analysis proposed by (Lin et al., 2023). 

4 

Pre-print. Under Review. 

For a given instruction-response-image triplet, **Algorithm 1** Categorizing Visual Hallucinations the instruction _i_ = { _i_ 1, _i_ 2, _· · ·_ } and the image _v_ are first input to the aligned (or fine-tuned) Given:LLMLLM _M_ Modelbase,base,, Response _R_ , IT dataset _D_ IT,, Aligned LVLM _M_ aligned,, base model to obtain its response _r_ = { _r_ 1, _r_ 2, _· · ·_ } Obtain Top- _k_ similar instances _SRR_ to _R_ from _D_ IT using CLIP via greedy decoding. Next, for each position _t_ Extract visual elementsExtract hallucinated phrasesExtract hallucinated phrases _V_ from _P R_ from. _R_ using LLM judge divided into Object, indefined as to bethe response, _x_ a _t_ ‘context’= _i_ + { _r_ 1,at _· · ·_ this , _rt_ position _−_ 1}. Thisis Relation, and Action hallucinations **procedureforfor** each word TOKEN TOKENOKEN _p_ A inNALYSIS inNALYSISNALYSIS _P_ **do** ( _R, M_ aligned _, M_ base)) “context” is then input to the base model (model **if** _p_ Calculate Base-Rank in _V_ **then BR** of _p ▷_ Consider only 1st token of _▷_ Defined in Section 3.3 _p_ from which the aligned model was instruction- **if BR** _>_ 0 **then** _▷_ marginal or shifted token or shifted token shifted token token tuned) to obtain its probability distribution for **if** _ppp_ in is an IT Hallucination is an IT Hallucination _SRR_ **then** predicting the next token at position _t_ , Pbase. We **else** then define **Base Rank** as follows: Rank in Pbase **if** _ppp_ == Relation & is a Style Hallucination is a Style Hallucination _p_ != Object **then** With the base rank denoted asof the token at _t_ with the maximum P _η_ , the unshiftedalign value., **elseend ifend if** _p_ is a Vision Hallucination marginal and shifted tokens are defined as when **elseend if** _▷_ unshifted token token ( _η_ = 1), (1 _< η ≤_ 3) and ( _η >_ 3) respectively. _p_ is a Language Hallucination Next, we use our judge (GPT-4) to extract the **end ifend ifend if** exact hallucinated phrases based on 3 types: ob- **end for** ject, relation, and action hallucinations. Finally, **end procedure** based on the Base Rank formulation and the hallucinated phrases, we propose Algorithm 1 to categorize identified hallucinations into 4 different categories based on their cause. The algorithm is further verbally described in Appendix K.4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0005-02.png)


**----- Start of picture text -----**<br>
Given:LLMLLM  M Modelbase,base,, Response R , IT dataset D IT,, Aligned LVLM M aligned,, base<br>Obtain Top- k  similar instances  SRR to  R  from  D IT using CLIP<br> V from  R .<br>Extract visual elementsExtract hallucinated phrasesExtract hallucinated phrases  P from.  R  using LLM judge divided into Object,<br>Relation, and Action hallucinations<br>procedureforfor  each word TOKEN TOKENOKEN  p A inNALYSIS inNALYSISNALYSIS  P do ( R, M aligned , M base))<br>if  p  in  V then ▷ Consider only 1st token of  p<br>Calculate Base-Rank in  BR  of  p ▷ Defined in Section 3.3<br>if BR > 0  then ▷ marginal or shifted token or shifted token shifted token token<br>if  ppp  in is an IT Hallucination is an IT Hallucination  SRR then<br>else<br>if  ppp  == Relation & is a Style Hallucination is a Style Hallucination  p  != Object  then<br>p  is a Vision Hallucination<br>elseend ifend if<br>end if<br>elseend if ▷ unshifted token token<br>p  is a Language Hallucination<br>end ifend ifend if<br>end for<br>end procedure<br>**----- End of picture text -----**<br>


**(1) Language Hallucinations** : These are hallucinations that are caused when LVLMs overrely on the language priors (or prior tokens in the response) rather than the input image. We attribute all hallucinated tokens that are unshifted as language hallucinations. 

**(2) Vision Hallucinations** : Unlike language hallucinations, the tokens for vision hallucinations are generated by attending to the input image and are shifted or marginal tokens. These are hallucinations caused when the model fails to accurately recognize visual elements in an input image or inaccurate reasoning or knowledge extraction, among other factors. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0005-05.png)


**----- Start of picture text -----**<br>
Language Hallucination Style Hallucination Vision Hallucination IT Halluciantion<br>1250 AMBER<br>1000<br>750<br>500<br>250<br>0<br>LLaVA-v1 CogVLM LLaVA-1.5 V C D OPERA Woodpecker LURE<br>1250 MMMU<br>1000<br>750<br>500<br>250<br>0<br>LLaVA-v1 CogVLM LLaVA-v1.5 V C D OPERA Woodpecker LURE<br>**----- End of picture text -----**<br>


**(3) Style Hallucinations.** These are hallucinations caused by _style imitation_ . Open-source VLLMs are often trained on instruction-tuning data synthesized from proprietary closed-source models. Training on such datasets makes the VLLM learn to mimic the characteristics of responses in the IT dataset (Ghosh et al., 2024; Gudibande et al., 2024) (e.g., lengthy answers, a summary line at the end of the response, etc.). However, owing to the lack of sufficient knowledge to mimic its closed-source counterparts, open-source VLLMs might hallucinate facts while responding (Ghosh et al., 2024). 

Figure 5: (Left) Frequency comparison of hallucination categories. (Right) Comparison for LLaVA-v1.5 with hallucination mitigation techniques. The top graph compares AMBER, and the bottom graph compares MMMU. 

**(4) Instruction Tuning (IT) Hallucinations.** Figure 5: (Left) Frequency comparison of hallucination These are hallucinations caused by biases categories. (Right) Comparison for LLaVA-v1.5 with learned during the instruction tuning stage. hallucination mitigation techniques. The top graph comLLMs are known to mimic the responses aspares AMBER, and the bottom graph compares MMMU. sociated with the unfamiliar examples in the model’s instruction-tuning data (i.e., those unfamiliar to the pre-trained base model) (Ghosh et al., 2024). Formally, this can be defined as a distribution shift. As visual instruction tuning is an extreme case of such a distribution shift, IT hallucinations are a major cause of hallucinations in LLMs. 

**Performance Comparison.** Fig. 5 (top;left) compares the frequency of different categories of hallucinations across various LVLMs for AMBER. Fig. 5 (top; right) displays the frequency of different hallucination categories for LLaVA-v1.5 alongside various hallucination mitigation techniques. 

5 

Pre-print. Under Review. 

Fig. 5 (bottom) offers a similar comparison for MMMU. Our key findings include: **(1)** Model-based improvements and mitigation strategies reduce language and vision hallucinations, but progress in style and IT hallucinations remains limited. **(2)** Specific mitigation techniques are more effective for certain types of hallucinations. For e.g., decoding-based methods like VCD and OPERA excel at reducing language hallucinations, whereas grounding approaches like Woodpecker are more effective for vision hallucinations. Decoding-based techniques benefit from low-confidence hallucinated tokens and uncertainty encoded in logits discussed next. More discussion is in Appendix K.1. **(3)** Existing methods fail to reduce hallucinations on reasoning benchmarks like MMMU. We attribute this to the algorithmic biases of these methods that are crafted to just reduce object hallucinations. 

_**Key Takeaway:**_ Current hallucination mitigation techniques mainly improve LVLMs’ visual recognition skills but are less effective in improving other cognitive abilities, such as reasoning. Additionally, these improvements are largely limited to real-world scenes and specific types of hallucinations, leaving areas like non-real-world scenes largely unexplored. 

## 4 THE VISUAL PERCEPTION GAP: LVLMS CAN SEE BUT NOT PERCEIVE 

It is evident that current hallucination mitigation techniques primarily enhance visual recognition but fail to improve other cognitive abilities, such as reasoning or knowledge extraction. In this section, we investigate the underlying cause of this limitation. Our analysis reveals that LVLMs often rely on language priors rather than attending to the input image when generating responses to reasoning prompts. This leads us to identify a critical issue: the visual perception gap. While LVLMs can accurately recognize visual elements and possess the necessary knowledge and reasoning skills to respond factually, they struggle to perceive and interpret these elements in relation to the input prompt. This gap in perception causes hallucinations, resulting in incorrect responses. 

## 4.1 VISUAL BLIND SPOTS: LVLMS IGNORE THE INPUT IMAGE FOR REASONING 

With findings from previous sections, we now as10 ~~Dataset~~ sess the influence of the image on the LVLM re9 AMBERMathVision sponse. Precisely, we compare the Base Rank (de8 fined in Section 3.3) of tokens across datasets. A 7 lower value would signify that the model solely re6 sponds using language priors, while a higher value 5 would signify that the model pays attention to the 4 image, i.e., it responds with tokens different from 3 what the model would have responded with only lan2 guage priors. Fig. 6 plots the Base Rank of tokens 1 against the token position in response (average across 0 0 10 20 30 40 50 the dataset). As we see, the Base Rank for AMBER, Token Postion which prompts for an image description, is higher Figure 6: Base Rank Comparison between AMthan Math-Vision, which requires it to reason or exBER and MATH-Vision datasets as a function of tract knowledge based on the image. This hints totoken position in responses (for CogVLM). wards an alignment issue: different from AMBER, for Math-Vision LVLMs are unable to perceive the image before responding, under-relying on visual cues and over-relying on language priors for generating responses. In Appendix K.1 we explain why decoding-based mitigation techniques discussed in Section 3.3 do not work for reaosning in cognitive prompts. 

Figure 6: Base Rank Comparison between AMBER and MATH-Vision datasets as a function of token position in responses (for CogVLM). 

## 4.2 DISENTANGLING VISUAL RECOGNITION FROM OTHER COGNITIVE SKILLS 

A key objective of alignment fine-tuning is to equip LVLMs with implicit visual perception skills. As outlined in Section 3, LVLMs should interpret images in context and respond accurately to information-seeking or reasoning prompts. To determine if hallucinations arise from the base LLM’s limitations or alignment gaps, we aim to isolate visual recognition from other cognitive skills and evaluate them separately. We use GPT-4 to rephrase prompts from visual information-seeking and reasoning benchmarks, modifying them so they can be answered without the image (examples in Appendix K.7 and prompt used in Appendix G). 

6 

Pre-print. Under Review. 

LVLMs are first evaluated on these text-only prompts and then on generating independent image descriptions (evaluation prompt in Appendix G). Figure 7 (left) compares the performance of LVLMs on the text-only prompts (marked with _-t_ ) against their original image-text versions, while Figure 7 (right) shows VR scores. Our results indicate that LVLMs excel in visual recognition and perform better on factual responses when using text-only prompts than with combined image-text inputs. This suggests that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0007-02.png)


**----- Start of picture text -----**<br>
MMMU Models MMMU<br>LLaVA-v1.5<br>mPLUG-Owl2<br>InternLM-X<br>LLaVA-v1.5- t<br>mPLUG-Owl2- t<br>InternLM-X- t<br>VaLLu 0 1 2 3 4 5MathVista VaLLu 0 1 2 3 4 5MathVista<br>InfoSeek InfoSeek<br>**----- End of picture text -----**<br>


Figure 7: (Left) Performance comparison of different LVLMs when prompted w/ original prompt vs rephrased prompts w/o image ( _-t_ ). (Right) Performance comparison of different LVLMs for their ability to generate a faithful image description. 

LVLMs have the necessary knowledge and reasoning skills but struggle to integrate these with visual perception in prompts requiring both. We identify this as a perception gap: LVLMs can recognize visual elements but fail to fully perceive them in a way that connects recognition to internal knowledge, which is essential for reasoning or information extraction. This challenge likely arises due to several factors: (i) training datasets that are largely composed of image-description pairs, (ii) alignment datasets that contain strong visual cues within the natural-language prompts. 

## 4.3 UNCERTAINTY IN THE LOGIT SPACE 

To investigate the origins of hallucinations, we Models Dataset _k_ Variance Range analyze the logit space of hallucinated tokens. Specifically, after applying the softmax operaLLaVA-v1 AMBER 1.1 / 4.2 0.0 / 0.1 0.0 / 0.3 LLaVA-v1.5 AMBER 1.0 / 3.9 0.0 / 0.1 0.0 / 0.4 tion to the logits, we rank them by their probCogVLM AMBER 1.0 / 3.4 0.0 / 0.1 0.0 / 0.5 ability scores. We then use the elbow method LLaVA-v1 MMMU 1.2 / 3.7 0.0 / 0.1 0.0 / 0.4 (detailed in Appendix K.6) to truncate the disLLaVA-v1.5 MMMU 1.2 / 3.5 0.0 / 0.2 0.0 / 0.4 tribution, obtaining the top- _k_ tokens along with CogVLM MMMU 1.1 / 3.2 0.0 / 0.2 0.0 / 0.5 their corresponding probabilities. This truncation approach, based on the elbow method, efTable 1: Post-truncation probability statistics using fectively identifies the tokens most likely to be the elbow method. All values are in the format: nonselected during multinomial sampling. Table 1 /hallucinated averaged across all tokens in the dataset. shows several statistics of the top- _k_ probabilities for hallucinated tokens (we only consider the first token of each word). As evident from the value of _k_ , standard deviation, and the average, the top- _k_ probability space for hallucinated tokens is dominated by a set of low and equally confident tokens. Thus, each of these tokens has almost a similar chance of being sampled with multi-nomial sampling. 

_**Key Takeaway:**_ LVLMs generally demonstrate accurate recognition and cognition skills to respond and reason accurately. However, they struggle to effectively link they recognize to their internal knowledge during the reasoning process, which results in inaccurate responses. This issue underlines what we identify as the visual perception gap. 

## 5 VISUAL DESCRIPTION GROUNDING DECODING (VDGD) 

**Main Motivation.** To bridge the visual perception gap and reduce hallucinations for cognitive prompts, we propose Visual Description Grounding Decoding (VDGD), a simple, effective, and _training-free_ hallucination mitigation technique. During decoding, VDGD first appends the image description (generated by the LVLM itself) as a prefix to the prompt’s text instruction, then grounds token generation by selecting the token that _deviates_ the least from the description. Token deviation is measured using a novel formulation based on Kullback–Leibler 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0007-10.png)


**----- Start of picture text -----**<br>
Image Description<br>1 that reads "Title." The chart has two axes: theThe image is a bar chart with a title at the top<br>LVLM x-axis lists the names ... The bars for "chaos"<br>have a blue bar that is 3 units tall and a green<br>bar that is 2 units tall. The ... with a clear and<br>straightforward presentation of the data.<br>Input Image+ 2 ReconstructionPrompt 3 5 DivergenceKL-0.04 Corrected Response The sum of all the<br>1 -0.09 values in the chaos<br>The input image can be<Description> What isdescribed as follows: LVLM ProbabilitiesNext Token8 -0.1 Wrong Response group is  5<br>the sum of all the values<br>in the chaos group? Sampling The sum of all the values<br>in the chaos group is 10.<br>Prompt + Description<br>Logits<br>**----- End of picture text -----**<br>


Figure 8: Illustration of our proposed VDGD method. 

7 

Pre-print. Under Review. 

divergence of the token with the description generated by the LVLM itself. Much like humans, who often write down their key observations in an image and refer to them while solving complex reasoning tasks, we hypothesize that grounding the LVLM’s responses in a detailed visual description can significantly enhance its reasoning process. Additionally, re-scoring the equally and low-confidence tokens (discussed in Section 1) based on deviation from the description can increase the confidence of the accurate token from several low-confidence tokens (Xu et al., 2023). 

## 5.1 METHODOLOGY 

Fig. 8 illustrates the VDGD methodology. For an input prompt, we first employ the LVLM to generate a description of the image accompanying the prompt (see Appendix G for the prompt). Next, we concatenate the generated description as a prefix to the original prompt. 

Formally put, let the VLM, denoted as _pV LM_ , receive an input prompt of length _n_ symbolized as _X_ pre = _x_ 1 _. . . xn_ , with each _xi_ representing a token from the vocabulary _V_ . The decoder’s objective is to produce responses of length _m_ , represented as _X_ res = _xn_ +1 _, . . . , xn_ + _m_ . During the decoding phase, the model iteratively decodes one token at a time, conditioning on the tokens that have been generated so far: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0008-05.png)


Here, _pV LM_ ( _xi | x<i_ ) denotes the probability distribution for the next token, _xi_ , given the previous tokens. Inspired by our findings in Section 4.3, we first apply a plausibility constraint to truncate the vocabulary space of the next token _xi_ (Li et al., 2023). Essentially, this step cuts off tokens from the vocabulary _V_ below the elbow and keeps only the top- _k_ plausible tokens. This is done as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0008-07.png)


where _α_ is a hyper-parameter between [0,1]. Now let _VK[x][i]_[be the set of top-] _[K]_[most plausible tokens] from the vocabulary _V_ after the truncation of _xi_ . Next, for each token _wk ∈VK[x][i]_[in the top-] _[K]_[ plausible] tokens, we calculate the deviation of the token with the _n_ tokens of the prompt as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0008-09.png)


where KL is the Kullback–Leibler divergence and one-hot( _wi_ ) _∈_ R _[V]_ is the one-hot representation of token _wk_ where each value is 0 except the position corresponding to the token-id of _w_ , which is 1. Next, we replace the value of _wk_ in the logit with _−_ KL _[x] w[i] k_[and apply softmax on the logits to obtain a] probability distribution. Thus, tokens with larger KLD (or a large _deviation_ ) to the input prompt are less likely to be sampled given the softmax operation upon KL divergence. With this distribution, we can now apply any sampling technique to sample the next token. 

## 5.2 EXPERIMENTAL SETUP 

**Datasets and Evaluation Metrics.** For evaluation, we employ a variety of standard benchmarks focused on reasoning and information-seeking tasks. These include LLaVA-Bench, MM-Vet (Yu et al., 2023), MMBench (Liu et al., 2023d), MME (Fu et al., 2023), MathVista (test-mini subset), MathVision, and MMMU (validation set). For each benchmark, we employ our proposed GPT evaluation technique described in Section 3. _**Beyond being more robust to the diverse formats of model generation, we also find GPT scores to have a higher correlation to expert-human evaluation (** ≈_ _**0.97; Section D has more details with original benchmark evaluation metrics that has a correlation of** ≈_ _**0.92).**_ We are inspired by a wealth of prior studies in hallucination evaluation (Sicong Leng et al., 2023; Yu et al., 2023; Liu et al., 2023c). All results are averaged across 3 runs. 

**Baselines.** We compare VDGD against VCD, OPERA, Woodpecker, LRV, LURE, HALC (Chen et al., 2024a) and PAI (Liu et al., 2024d) hallucination mitigation techniques. Additionally, we compare two vanilla decoding methods (without any additional mitigation techniques): Vanilla-greedy with vanilla-greedy decoding and Vanilla-sampling with multinomial sampling-based decoding (top-p=0.5 and temperature=0.7). We employ greedy decoding for all methods as we find no difference in performance on sampling. We generate image descriptions for VDGD using the VCD mitigation 

8 

Pre-print. Under Review. 

|**Benchmark**|**Baseline**<br>**LLaVA-v1**<br>**LLaVA-1.5**<br>**LLaVA-1.6**<br>**mPLUG-Owl2**<br>**InternLM-X**<br>**CogVLM**<br>**CogVLM2**<br>**Qwen2-VL**|
|---|---|
|**MMMU**|Vanilla-greedy<br>1.26<br>1.35<br>1.42<br>1.40<br>2.01<br>1.66<br>2.08<br>2.39|
||Vanilla-sampling<br>1.27<br>1.44<br>1.40<br>1.41<br>2.05<br>1.64<br>2.10<br>2.35<br>VDD<br>1.34<br>1.52<br>1.44<br>1.53<br>2.22<br>1.68<br>2.14<br>2.64<br>OPERA<br>1.30<br>1.43<br>1.57<br>1.64<br>2.25<br>1.62<br>2.21<br>2.44<br>Woodpecker<br>1.32<br>1.44<br>1.63<br>1.61<br>2.12<br>1.65<br>2.14<br>2.56<br>LRV<br>1.29<br>1.49<br>1.61<br>1.58<br>2.08<br>1.59<br>2.18<br>2.48<br>LURE<br>1.31<br>1.47<br>1.60<br>1.64<br>2.27<br>1.66<br>2.27<br>2.71<br>PAI<br>1.42<br>1.39<br>1.44<br>1.56<br>2.23<br>1.65<br>2.29<br>2.64<br>HALC<br>1.40<br>1.54<br>1.63<br>1.65<br>2.15<br>1.67<br>2.24<br>2.69<br>**VDGD (ours)**<br>**1.49**(+18%)<br>**1.62**(+20%)<br>**1.75**(+23%)<br>**1.72**(+23%)<br>**2.39**(+19%)<br>**1.71**(+3%)<br>**2.47**(+19%)<br>**2.91**(+22%)|
|**MathVista**|Vanilla-greedy<br>1.56<br>1.65<br>2.00<br>1.92<br>2.56<br>2.15<br>2.45<br>2.97|
||Vanilla-sampling<br>1.54<br>1.68<br>1.91<br>1.94<br>2.52<br>2.19<br>2.46<br>2.95<br>VDD<br>1.67<br>1.68<br>2.07<br>2.11<br>2.59<br>2.53<br>2.81<br>3.19<br>OPERA<br>1.64<br>1.83<br>2.04<br>2.04<br>2.62<br>2.30<br>2.53<br>3.22<br>Woodpecker<br>1.72<br>2.10<br>2.19<br>2.13<br>2.43<br>2.43<br>2.59<br>3.13<br>LRV<br>1.68<br>1.68<br>2.17<br>1.99<br>2.61<br>2.20<br>2.62<br>3.30<br>LURE<br>1.59<br>2.03<br>2.21<br>2.07<br>2.37<br>2.45<br>2.49<br>3.13<br>PAI<br>1.68<br>1.85<br>1.97<br>2.08<br>2.63<br>2.20<br>2.60<br>3.39<br>HALC<br>1.71<br>2.05<br>2.17<br>1.97<br>2.54<br>2.24<br>2.77<br>3.38<br>**VDGD (ours)**<br>**1.84**(+18%)<br>**2.19**(+33%)<br>**2.44**(+22%)<br>**2.24**(+17%)<br>**2.88**(+13%)<br>**2.62**(+22%)<br>**2.93**(+20%)<br>**3.59**(+21%)|
|**MMBench**|Vanilla-greedy<br>2.67<br>2.89<br>3.11<br>2.98<br>3.58<br>3.19<br>3.71<br>3.93|
||Vanilla-sampling<br>2.69<br>2.92<br>3.12<br>3.02<br>3.59<br>3.22<br>3.68<br>3.95<br>VDD<br>2.80<br>3.03<br>3.09<br>3.22<br>3.93<br>3.28<br>3.81<br>4.20<br>OPERA<br>2.81<br>2.94<br>3.17<br>3.10<br>3.95<br>3.22<br>3.93<br>4.05<br>Woodpecker<br>2.83<br>2.96<br>3.14<br>3.07<br>3.84<br>3.33<br>3.83<br>4.19<br>LRV<br>2.77<br>2.99<br>3.13<br>3.23<br>3.84<br>3.25<br>3.77<br>4.05<br>LURE<br>2.72<br>2.97<br>3.13<br>3.26<br>3.72<br>3.21<br>3.82<br>4.16<br>PAI<br>2.75<br>3.00<br>3.04<br>3.16<br>3.94<br>3.35<br>3.97<br>4.05<br>HALC<br>2.73<br>2.93<br>3.02<br>3.18<br>3.81<br>3.27<br>4.04<br>4.08<br>**VDGD (ours)**<br>**2.93**(+10%)<br>**3.12**(+8%)<br>**3.26**(+5%)<br>**3.32**(+11%)<br>**4.08**(+14%)<br>**3.42**(+7%)<br>**4.15**(+12%)<br>**4.31**(+10%)|
|**MME**|Vanilla-greedy<br>3.32<br>3.54<br>3.65<br>3.49<br>3.75<br>3.57<br>3.64<br>3.86|
||Vanilla-sampling<br>3.34<br>3.53<br>3.62<br>3.48<br>3.77<br>3.58<br>3.66<br>3.85<br>VDD<br>3.46<br>3.61<br>3.77<br>3.59<br>3.96<br>3.67<br>3.95<br>4.21<br>OPERA<br>3.42<br>3.59<br>3.82<br>3.54<br>3.93<br>3.60<br>3.88<br>4.09<br>Woodpecker<br>3.37<br>3.55<br>3.76<br>3.63<br>3.82<br>3.59<br>3.92<br>4.15<br>LRV<br>3.43<br>3.63<br>3.78<br>3.52<br>3.95<br>3.61<br>3.96<br>3.99<br>LURE<br>3.42<br>3.62<br>3.87<br>3.67<br>3.93<br>3.72<br>3.77<br>4.10<br>PAI<br>3.38<br>3.56<br>3.81<br>3.55<br>3.89<br>3.69<br>3.82<br>4.04<br>HALC<br>3.47<br>3.58<br>3.83<br>3.71<br>4.01<br>3.59<br>3.80<br>4.14<br>**VDGD (ours)**<br>**3.59**(+8%)<br>**3.70**(+5%)<br>**3.99**(+9%)<br>**3.82**(+9%)<br>**4.12**(+10%)<br>**3.79**(+6%)<br>**4.09**(+12%)<br>**4.34**(+12%)|
|**Oven**|Vanilla-greedy<br>2.44<br>2.66<br>3.13<br>2.86<br>3.35<br>3.35<br>3.48<br>3.64|
||Vanilla-sampling<br>2.45<br>2.65<br>3.10<br>2.84<br>3.33<br>3.37<br>3.45<br>3.61<br>VDD<br>2.20<br>2.44<br>2.96<br>2.67<br>3.10<br>3.21<br>3.39<br>3.59<br>OPERA<br>2.50<br>2.49<br>3.20<br>2.92<br>3.44<br>3.42<br>3.53<br>3.68<br>Woodpecker<br>2.46<br>2.46<br>3.21<br>2.90<br>3.45<br>3.45<br>3.52<br>3.70<br>LRV<br>2.51<br>2.52<br>3.18<br>2.88<br>3.47<br>3.40<br>3.57<br>3.74<br>LURE<br>2.49<br>2.51<br>3.18<br>2.91<br>3.42<br>3.46<br>3.55<br>3.72<br>PAI<br>2.54<br>2.60<br>3.26<br>2.95<br>3.48<br>3.48<br>3.60<br>3.78<br>HALC<br>2.52<br>2.57<br>3.24<br>2.94<br>3.50<br>3.45<br>3.61<br>3.80<br>**VDGD (ours)**<br>**2.78**(+14%)<br>**2.84**(+7%)<br>**3.49**(+12%)<br>**3.15**(+10%)<br>**3.68**(+10%)<br>**3.59**(+7%)<br>**3.75**(+8%)<br>**3.92**(+7%)|
|**LLaVA-Bench**|Vanilla-greedy<br>3.01<br>3.12<br>3.23<br>3.13<br>4.12<br>3.68<br>4.15<br>4.37|
||Vanilla-sampling<br>2.98<br>3.10<br>3.27<br>3.17<br>4.08<br>3.71<br>4.17<br>4.33<br>VDD<br>2.93<br>3.16<br>3.18<br>3.15<br>3.89<br>3.57<br>4.13<br>4.28<br>OPERA<br>3.08<br>3.18<br>3.30<br>3.22<br>4.16<br>3.75<br>4.22<br>4.40<br>Woodpecker<br>3.10<br>3.22<br>3.36<br>3.23<br>4.21<br>3.73<br>4.21<br>4.39<br>LRV<br>3.16<br>3.19<br>3.30<br>3.22<br>4.27<br>3.72<br>4.20<br>4.49<br>LURE<br>3.18<br>3.26<br>3.35<br>3.24<br>4.25<br>3.64<br>4.27<br>4.51<br>PAI<br>3.20<br>3.28<br>3.38<br>3.25<br>4.30<br>3.74<br>4.32<br>4.54<br>HALC<br>3.25<br>3.27<br>3.37<br>3.22<br>4.24<br>3.76<br>4.35<br>4.53<br>**VDGD (ours)**<br>**3.36**(+12%)<br>**3.47**(+11%)<br>**3.52**(+9%)<br>**3.38**(+8%)<br>**4.45**(+8%)<br>**3.96**(+8%)<br>**4.50**(+8%)<br>**4.78**(+9%)|
|**MMVET**|Vanilla-greedy<br>2.39<br>2.52<br>2.72<br>2.87<br>3.39<br>3.18<br>3.47<br>3.64|
||Vanilla-sampling<br>2.34<br>2.50<br>2.70<br>2.84<br>3.32<br>3.35<br>3.42<br>3.62<br>VDD<br>2.29<br>2.61<br>2.67<br>2.92<br>3.29<br>3.42<br>3.39<br>3.58<br>OPERA<br>2.48<br>2.67<br>2.77<br>3.01<br>3.37<br>3.39<br>3.52<br>3.69<br>Woodpecker<br>2.47<br>2.64<br>2.82<br>3.06<br>3.40<br>3.44<br>3.50<br>3.72<br>LRV<br>2.58<br>2.56<br>2.79<br>2.92<br>3.38<br>3.42<br>3.51<br>3.75<br>LURE<br>2.54<br>2.63<br>2.81<br>3.12<br>3.41<br>3.46<br>3.56<br>3.71<br>PAI<br>2.62<br>2.60<br>2.83<br>3.18<br>3.37<br>3.49<br>3.59<br>3.78<br>HALC<br>2.66<br>2.70<br>2.88<br>3.15<br>3.44<br>3.47<br>3.54<br>3.76<br>**VDGD (ours)**<br>**2.85**(+19%)<br>**3.01**(+19%)<br>**3.23**(+19%)<br>**3.45**(+20%)<br>**3.78**(+12%)<br>**3.57**(+12%)<br>**3.74**(+8%)<br>**3.99**(+10%)|
|**VaLLu**|Vanilla-greedy<br>1.95<br>2.03<br>2.63<br>2.20<br>2.66<br>2.82<br>3.23<br>3.45|
||Vanilla-sampling<br>1.86<br>2.01<br>1.64<br>2.18<br>2.70<br>2.83<br>3.21<br>3.40<br>VDD<br>1.47<br>1.55<br>1.80<br>1.80<br>2.32<br>2.63<br>3.19<br>3.42<br>OPERA<br>2.04<br>2.05<br>2.62<br>2.26<br>2.65<br>2.85<br>3.22<br>3.47<br>Woodpecker<br>2.01<br>2.05<br>2.67<br>2.23<br>2.60<br>2.91<br>3.28<br>3.52<br>LRV<br>1.98<br>2.10<br>2.65<br>2.19<br>2.59<br>2.88<br>3.32<br>3.51<br>LURE<br>2.03<br>2.03<br>2.64<br>2.24<br>2.64<br>2.78<br>3.28<br>3.48<br>PAI<br>2.04<br>2.23<br>2.78<br>2.34<br>2.76<br>2.86<br>3.31<br>3.54<br>HALC<br>2.02<br>2.22<br>2.74<br>2.32<br>2.72<br>2.89<br>3.29<br>3.52<br>**VDGD (ours)**<br>**2.16**(+11%)<br>**2.64**(+30%)<br>**3.16**(+20%)<br>**2.72**(+24%)<br>**3.45**(+30%)<br>**3.01**(+7%)<br>**3.48**(+8%)<br>**3.67**(+6%)|



Table 2: Performance comparison of VDGD with various baselines. VDGD outperforms by 2%-33%. 

technique. For LVLMs, we employ LLaVA-v1, LLaVA-v1.5, LLaVA-v1.6, mPLUG-Owl2, InternLMX, CogVLM, Qwen2-VL (Wang et al., 2024b) and CogVLM2 (Hong et al., 2024). 

9 

Pre-print. Under Review. 

**Proposed VaLLu Benchmark.** Automatic evaluation of LVLMs, with powerful closedsource models, on multiple large benchmarks can be prohibitively expensive. Thus, we propose VaLLu, a benchmark that we build by amalgamating existing benchmarks. VaLLu comprises a total of 1500 instances, with 554 sourced from Oven, 535 from MMMU, 218 87 ( 0 . 0 6%) Science from MMC, 74 from MathVista, 60 from HalluBusiness sionBench, 53 from MATH-Vision, and 14 from MME. We only include instances with instruc250 (16.6%) Coding tions that require open-ended generations and do 371 (24.6%) Engineering not include any binary Yes/No or Multi-choice Maths questions. We control the size to ensure the evalMedicine uation is affordable while keeping the diversity 788 (52.3%) InfoSeek of tasks and topics for comprehensive analysis. Visual Perception VaLLu is manually filtered for noisy examples InfoSeek+Reasoning commonly found in existing benchmarks (we discuss this further in Appendix K.3), annotated Figure 9: Distribution of task types in VaLLu. with additional meta-data (illustrated in Fig. 9) and paired with an expert-provided response. 

## 5.3 RESULTS AND ANALYSIS 

**Quantitative results.** Table 9 compares the performance of VDGD with other baselines. As we clearly see, VDGD not only outperforms all baselines but is also the only method that alleviates and improves performance on VaLLu. Results on other full benchmarks are provided in Appendix L.6, together with scores for other aspects beyond factuality, where we show that VDGD does not compromise the depth and expressivity of responses. _Ablation results in Appendix C demonstrate that removing KL divergence leads to drop in performance. Additionally, using only the description prefix without the image leads to performance drops, confirming that VDGD enhances reasoning through grounding rather than making the model rely completely on the description alone._ 

**Qualitative examples.** Fig 10 compares responses from vanilla greedy decoding with VDGD. Grounding response generation results in more factual responses. Appendix B illustrates 2 failure cases of VDGD where incorrect descriptions can lead VDGD to hallucinate. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0010-05.png)


**----- Start of picture text -----**<br>
Question:  Baxter Company has a relevant range of production between<br>15,000 and 30,000 units. The following cost data represents average variable<br>costs per unit for 25,000 units of production. If 15,000 units are produced, Question:  Where is it<br>what is the variable cost per unit? typically found?<br>Greedy Decoding:  To find the variable cost per unit, we need to calculate the total variable cost and divide it by Greedy Decoding:  The image shows a wasp<br>manufacturing overheadthe number of units produced. \n ) + (Variable manufacturing overhead*Number of units). \n Total variable cost = ($10 Total variable cost = (Direct materials*Direct labor) + (Indirect materials*Fixed* on a yellow flower.<br>$9) + ($3  * $6) + ($2 * 15,000) = $30,118. \n  Variable cost per unit = Total variable cost/Number of<br>units=30,118/15,000 = $2.00. Therefore, if 15,000 units are produced, the variable cost per unit is $2.00.<br>VDGD Decoding:  The wasp is typically found<br>VDGD Decoding:  To find the variable cost per unit, we need to calculate the total variable cost and divide it by in gardens, meadows, and other natural habitats<br>the number of units produced. \n Total variable cost = (Direct labor cost + Indirect materials cost + Variable where it can find flowers to feed on. It may also<br>manufacturing overhead cost + Variable sales commissions cost)*Number of units. \n Total variable cost = ($9 be found near human settlements if the<br>+ $3 + $2 + $14)*25,000 = $950,000. \n Variable cost per unit = Total variable cost/Number of units = environment provides suitable nesting sites and<br>$950,000/15,000 = $63.33. \n Therefore, if 15,000 units are produced, the variable cost per unit would be food sources.<br>$63.33.<br>**----- End of picture text -----**<br>


Figure 10: Illustration of responses generated with VDGD compared with greedy decoding. More in Sec. K.2. 

## 6 CONCLUSION, LIMITATIONS AND FUTURE WORK 

In this paper, we conduct extensive experiments to analyze hallucinations in LVLMs. Our findings reveal that although there has been considerable progress in reducing hallucinations related to visual recognition, reducing hallucinations for cognitive prompts has been limited. Further investigation into LVLM responses to cognitive prompts reveals a visual perception gap: despite having the required capabilities, LVLMs often produce hallucinations when responding to prompts. To address this issue, we introduce VDGD, a novel and training-free method to reduce hallucinations. Our results demonstrate that VDGD is the first method to effectively reduce hallucinations in responses that necessitate additional cognitive skills beyond just visual recognition. 

10 

Pre-print. Under Review. 

As part of future work, we would like to work on the current limitations of our work, which include (1) Error accumulation in VDGD that comes from inaccurate image descriptions in the prefix, (2) the requirement of prompting the LVLM twice, (3) Analysis of our findings with better alignment techniques and methods like RAG. 

## 7 REPRODUCIBILITY STATEMENT 

We provide our code here: https://anonymous.4open.science/r/VDGD-1E04/. All codes will be open-sourced upon paper acceptance, including all checkpoints. All experimental details, including training parameters and hyper-parameters, are provided in Section 5.3. The Appendix has more comprehensive details about our experimental setup. 

## REFERENCES 

- Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022. 

- Harry Barrow, J Tenenbaum, A Hanson, and E Riseman. Recovering intrinsic scene characteristics. _Comput. vis. syst_ , 2(3-26):2, 1978. 

- Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan, Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He, Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao, Zhenjiang Jin, Zhikai Lei, Jiaxing Li, Jingwen Li, Linyang Li, Shuaibin Li, Wei Li, Yining Li, Hongwei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu, Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv, Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai Shang, Yunfan Shao, Demin Song, Zifan Song, Zhihao Sui, Peng Sun, Yu Sun, Huanze Tang, Bin Wang, Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang, Yudong Wang, Ziyi Wang, Xingjian Wei, Qizhen Weng, Fan Wu, Yingtong Xiong, Chao Xu, Ruiliang Xu, Hang Yan, Yirong Yan, Xiaogui Yang, Haochen Ye, Huaiyuan Ying, Jia Yu, Jing Yu, Yuhang Zang, Chuyu Zhang, Li Zhang, Pan Zhang, Peng Zhang, Ruijie Zhang, Shuo Zhang, Songyang Zhang, Wenjian Zhang, Wenwei Zhang, Xingcheng Zhang, Xinyue Zhang, Hui Zhao, Qian Zhao, Xiaomeng Zhao, Fengzhe Zhou, Zaida Zhou, Jingming Zhuo, Yicheng Zou, Xipeng Qiu, Yu Qiao, and Dahua Lin. Internlm2 technical report, 2024. 

- Zhaorun Chen, Zhuokai Zhao, Hongyin Luo, Huaxiu Yao, Bo Li, and Jiawei Zhou. HALC: Object hallucination reduction via adaptive focal-contrast decoding. In _Forty-first International Conference on Machine Learning_ , 2024a. URL https://openreview.net/forum?id=EYvEVbfoDp. 

- Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_ , 2024b. 

- Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arxiv 2306.13394 (2023), 2023. 

- Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. _arXiv preprint arXiv:2404.12390_ , 2024. 

- Sreyan Ghosh, Chandra Kiran Reddy Evuru, Sonal Kumar, Ramaneswaran S, Deepali Aneja, Zeyu Jin, Ramani Duraiswami, and Dinesh Manocha. A closer look at the limitations of instruction tuning. In _Forty-first International Conference on Machine Learning_ , 2024. URL https: //openreview.net/forum?id=XkHJo8iXGQ. 

11 

Pre-print. Under Review. 

- Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: An advanced diagnostic suite for entangled language hallucination & visual illusion in large vision-language models. _arXiv preprint arXiv:2310.14566_ , 2023. 

- Arnav Gudibande, Eric Wallace, Charlie Victor Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, and Dawn Song. The false promise of imitating proprietary language models. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https: //openreview.net/forum?id=Kz3yckpCN5. 

- Richard Hartley and Andrew Zisserman. _Multiple view geometry in computer vision_ . Cambridge university press, 2003. 

- Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. _arXiv preprint arXiv:2408.16500_ , 2024. 

- Hexiang Hu, Yi Luan, Yang Chen, Urvashi Khandelwal, Mandar Joshi, Kenton Lee, Kristina Toutanova, and Ming-Wei Chang. Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities. _arXiv preprint arXiv:2302.11154_ , 2023. 

- Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. _arXiv preprint arXiv:2311.05232_ , 2023a. 

- Qidong Huang, Xiaoyi Dong, Pan zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. _arXiv preprint arXiv:2311.17911_ , 2023b. 

- Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In _European Conference on Computer Vision (ECCV)_ , 2022. 

- Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 12286– 12312, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/ 2023.acl-long.687. URL https://aclanthology.org/2023.acl-long.687. 

- Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu, Nouha Dziri, Melanie Sclar, Khyathi Chandu, Chandra Bhagavatula, and Yejin Choi. The unlocking spell on base llms: Rethinking alignment via in-context learning, 2023. 

- Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. Mmc: Advancing multimodal chart understanding with large-scale instruction tuning. _arXiv preprint arXiv:2311.10774_ , 2023a. 

- Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating hallucination in large multi-modal models via robust instruction tuning, 2024a. 

- Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. _arXiv preprint arXiv:2402.00253_ , 2024b. 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _arXiv preprint arXiv:2310.03744_ , 2023b. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023c. URL https: //openreview.net/forum?id=w0H2xGHlkw. 

12 

Pre-print. Under Review. 

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024c. URL https: //llava-vl.github.io/blog/2024-01-30-llava-next/. 

- Shi Liu, Kecheng Zheng, and Wei Chen. Paying more attention to image: A training-free method for alleviating hallucination in lvlms. _arXiv preprint arXiv:2407.21771_ , 2024d. 

- Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? _arXiv preprint arXiv:2307.06281_ , 2023d. 

- Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. _arXiv preprint arXiv:2310.02255_ , 2023. 

- Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In _International Conference on Learning Representations (ICLR)_ , 2024. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021. 

- Guanzheng Chen Sicong Leng, Hang Zhang, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. _arXiv preprint arXiv:2311.16922_ , 2023. URL https://arxiv.org/abs/2311. 16922. 

- Antonio Torralba and Aude Oliva. Depth estimation from image structure. _IEEE Transactions on pattern analysis and machine intelligence_ , 24(9):1226–1238, 2002. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Ming Yan, Ji Zhang, and Jitao Sang. An llm-free multi-dimensional benchmark for mllms hallucination evaluation. _arXiv preprint arXiv:2311.07397_ , 2023a. 

- Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. _arXiv preprint arXiv:2402.14804_ , 2024a. 

- Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024b. 

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. _arXiv preprint arXiv:2311.03079_ , 2023b. 

- Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In _International Conference on Learning Representations_ , 2022. URL https://openreview.net/forum? id=gEZrGCozdqR. 

- Nan Xu, Chunting Zhou, Asli Celikyilmaz, and Xuezhe Ma. Look-back decoding for open-ended text generation. _arXiv preprint arXiv:2305.13477_ , 2023. 

- Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration, 2023. 

13 

Pre-print. Under Review. 

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. _arXiv preprint arXiv:2306.13549_ , 2023a. 

- Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. Woodpecker: Hallucination correction for multimodal large language models. _arXiv preprint arXiv:2310.16045_ , 2023b. 

- Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. _arXiv preprint arXiv:2308.02490_ , 2023. 

- Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In _Proceedings of CVPR_ , 2024. 

- Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. Vision-language models for vision tasks: A survey. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2024a. 

- Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, et al. Internlm-xcomposer: A visionlanguage large model for advanced text-image comprehension and composition. _arXiv preprint arXiv:2309.15112_ , 2023. 

- Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, and Guoyin Wang. Instruction tuning for large language models: A survey, 2024b. 

- Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models, 2023. 

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In _Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , 2023. URL https: //openreview.net/forum?id=uccHPGDlao. 

- Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, LILI YU, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. LIMA: Less is more for alignment. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023a. URL https://openreview.net/forum?id=KBMOKmX2he. 

- Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao. Analyzing and mitigating object hallucination in large vision-language models. _arXiv preprint arXiv:2310.00754_ , 2023b. 

14 

Pre-print. Under Review. 

## A APPENDIX 

In the supplemental material, 

1. B: We illustrate failure cases for VDGD. 

2. C: We provide results for ablation of VDGDs key components. 

3. D: We provide correlation between GPT scores and actual metrics. 

4. E: We provide a detailed explanation of computational complexity of VDGD algorithm described in section 5. 

5. F: We describe the impact of the paper on society and research communities. 

6. G: We provide the prompts used for object detection, description, rephrasing, noise removal and evaluations required for our experiments. 

7. H: We provide a detailed description of all the datasets utilized in this paper. 

8. I: We provide a detailed description of all the models we have experimented with. 

9. J: We provide a detailed description of all the prior-art hallucination mitigation techniques used as baselines. 

10. K: We provide details about: 

   - K.1: We provide detailed explanation about Logit Space for VR and Cognitive Prompts. 

   - K.2: We provide examples comparing generations from Greedy, Decoding, VCD and VDGD. 

   - K.3: We provide examples of noisy samples in datasets. 

   - K.4: We provide and in-depth explanation of all the details of Algorithm 1. 

   - K.6: We explain the elbow method used for logit analysis. 

   - K.7: We provide examples of rephrased prompts which do not require image input. 

   - K.8: We provide details of computational resources used for this paper. 

11. L: We provide additional results for all the experiments described in the paper. 

## B VDGD FAILURE CASES 

Fig. 11 illustrates 2 failure cases of VDGD. We show that VDGD can lead to incorrect responses due to an incorrectly generated image description. However, we also argue that as LVLMs get better at describing images, the performance of VDGD decoding will improve. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0015-23.png)


**----- Start of picture text -----**<br>
Question:  Which animals in  belong to a clade that includes animals with Question: During which numbered<br>hair? step(s) are bacteria reproducing?<br>Description:  The image is a diagram that shows the evolutionary relationships between various animals. Description:  The image is a diagram illustrating the process of a bacterial cell's<br>The diagram is divided into three sections, each representing a different group of animals. The firstsection is labeled \"Lancetel,\" the second is labeled \"Lampry,\" and the third is labeled \"Fish.\"\n\nIn the growth and reproduction. The diagram consists of a series of numbered steps, with<br>first section, there are three animals: one on the left, one in the middle, and one on the right. The second each step showing a different stage of the process. The diagram features a number of<br>section has four animals, with one on the left, one in the middle, and two on the right. The third section circles, which represent the bacterial cell. The circles are placed at various positions<br>has five animals, with one on the left, one in the middle, and three on the right.\n\nThe animals in each along the diagram, indicating the different stages of the cell's growth and<br>section are arranged in a way that shows their evolutionary relationships. The diagram provides a clear reproduction. The circles are connected by arrows, which show the movement of the<br>visual representation of the relationships between these animals, making it easier to understand theirevolutionary history. bacteria through the different stages.<br>VDGD:  Yes, the animals in the first section, labeled "Lancetel" belong to a clade that includes animals VDGD:  During  step 3, bacteria are reproducing. In this step, the bacterial cell<br>with hair. divides into two identical cells, which are then released into the environment.<br>**----- End of picture text -----**<br>


Figure 11: Illustration of failure cases of VDGD. 

## C VDGD ABLATION RESULTS 

To demonstrate the effectiveness of VDGD, we conduct ablation studies on its key components: (i) VDGD (- VCD): Image descriptions are generated without using Visual Contrastive Decoding (VCD), (ii) VDGD (- KLD): KLD-based sampling is removed, and only the description is appended, (iii) VDGD (- Image): VDGD is applied without the original image input; only the image description is appended to the prompt. Results are presented on the VDGD benchmark in Table 3. Removing KL Divergence leads to a substantial performance drop. We notice a similar phenomenon by removing the input image with the steepest drop. This shows that the input image is still essential for the model. 

15 

Pre-print. Under Review. 

|**Benchmark**|**LLaVA-v1**<br>**LLaVA-1.5**<br>**LLaVA-1.6**<br>**mPLUG-Owl2**<br>**InternLM-X**|**LLaVA-v1**<br>**LLaVA-1.5**<br>**LLaVA-1.6**<br>**mPLUG-Owl2**<br>**InternLM-X**|**LLaVA-v1**<br>**LLaVA-1.5**<br>**LLaVA-1.6**<br>**mPLUG-Owl2**<br>**InternLM-X**|**LLaVA-v1**<br>**LLaVA-1.5**<br>**LLaVA-1.6**<br>**mPLUG-Owl2**<br>**InternLM-X**|**LLaVA-v1**<br>**LLaVA-1.5**<br>**LLaVA-1.6**<br>**mPLUG-Owl2**<br>**InternLM-X**|**CogVLM**|
|---|---|---|---|---|---|---|
|Vanilla-greedy|1.95|2.03|2.63|2.20|2.66|2.82|
|**VDGD**_(ours)_|**2.16**|**2.64**|**3.16**|**2.72**|**3.45**|**3.01**|
|**VDGD (-) VCD**<br>**VDGD (-) KLD**|2_._08_±_0_._09<br>2_._05_±_0_._10|2_._43_±_0_._15<br>2_._30_±_0_._10|3_._01_±_0_._08<br>2_._78_±_0_._12|2_._54_±_0_._07<br>2_._37_±_0_._06|3_._26_±_0_._12<br>3_._01_±_0_._02|2_._95_±_0_._06<br>2_._87_±_0_._08|



Table 3: Ablation results for VDGD. VDGD sees a drop in performance when the KLD decoding objective is removed and a major drop when no input image is provided. 

## D CORRELATION BETWEEN GPT SCORES AND ACTUAL METRICS 

**Background of Experts in Human Evaluation.** For the expert human evaluation of model generations, we engaged a group of 5 PhD students with at least four years of research experience in computer vision. These experts were tasked with evaluating the outputs of the models for accuracy, relevance, and factual correctness. To ensure thorough evaluation, they were provided with internet access to verify factual information when necessary. The evaluation process was conducted in an anonymized format to maintain objectivity, with the experts working independently and without prior knowledge of the model specifics. We selected 500 samples or the size of the benchmark, whichever was lower. 

## E VDGD COMPUTATIONAL COMPLEXITY 

Given an LVLM with a vocabulary size _V_ , an input instruction _I_ with a description of length _M_ tokens, at each decoding step, the complexity for equation (2) is _O_ ( _V_ ) and let the number of plausible tokens after this step be _K_ , the complexity for equation (3) is _O_ ( _V MK_ ). The total computational complexity of VDGD at each decoding step is _O_ ( _V_ ) + _O_ ( _V MK_ ) _≈ O_ ( _V MK_ ). 

## F BROADER IMPACT 

This paper addresses the broader impact of reducing hallucinations in Large Vision-Language Models (LVLMs), which are increasingly utilized in diverse applications such as automated content generation, assistive technologies, and decision-making systems. By focusing on the phenomenon of hallucination, particularly in responses to cognitive prompts, our work has the potential to enhance the reliability and safety of LVLMs in practical scenarios. The proposed Visual Description Grounded Decoding (VDGD) method, being training-free, offers an accessible and scalable solution that can be readily implemented across various models to mitigate risks associated with inaccurate or misleading outputs. Furthermore, the introduction of the VaLLu benchmark contributes to the field by providing a tool for the comprehensive evaluation of cognitive capabilities in LVLMs, promoting transparency and fostering further research in the area. Collectively, these contributions aim to improve user trust in AI systems and encourage responsible AI development that prioritizes factual accuracy and understanding. 

## G PROMPTS 

For image description benchmarks, we employ the prompt illustrated in Fig. 12. For non-image description benchmarks, we modify this prompt slightly (we modify the part of the prompt for extracting hallucinated phrases), and we illustrate the same in Fig. 16. The prompt used to generate image descriptions for VDGD is illustrated in Fig. 15. _Correctness_ in the prompts refers to _Factuality_ in all tables in Section L. We name it so after an ablation study on the terminology that works better for prompting the judge (GPT-4). All prompt evaluations on GPT-4 are done with a temperature of 0.7. 

16 

Pre-print. Under Review. 

## **`# Evaluation Prompt`** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0017-02.png)


**----- Start of picture text -----**<br>
Please act as an impartial judge and evaluate the quality of the response provided with respect to the input image. You will rate the quality<br>of the response on multiple aspects, such as Helpfulness, Clarity, Correctness, Depth and Engagement. You will also output lists with<br>hallucinated content or wrongful information in the response with respect to the input image.<br>##Query: {instruction}<br>##Output: {answer}<br>## Evaluate<br>### Aspects<br>- Helpfulness: Rate the response based on how well it addresses the users query about the image and provides a relevant answer. A score of 5<br>indicates the answer fully aids the user, while a 1 suggests it offers little to no help.<br>- Clarity: Rate the response based on how well-structured it is, with ideas presented in a clear and coherent manner. A high score of 5 means<br>the answer is clear and logically structured, while a 1 suggests a disjointed or confusing reply.<br> - Correctness: Evaluate the correctness or accuracy of the response provided with respect to the image and the respective question asked about<br>the image. A perfect 5 indicates the response is entirely correct and accurate, while a 1 suggests it has significant errors or has not<br>provided an answer to the question asked at all.<br> - Depth: Determine the level of detail and thoroughness in the response. A score of 5 means the answer delves deeply into the aspects of the<br>input image for answering the question, while a 1 indicates it barely scratches the surface.<br>### Format Given the query and the input image, please rate the quality of the output by scoring it from 1 to 5, individually on **each<br>aspect**.<br>- 1: strongly disagree<br>- 2: disagree<br>- 3: neutral<br>- 4: agree<br>- 5: strongly agree<br> You are also asked to output the hallucinated content or wrongful information in the response with respect to the input image with respect to<br>3 different aspects:<br>### Aspects<br>- Object Hallucination: Objects in the response that are not present in the image.<br>- Action/Verb Hallucination: Actions or verbs in the response that cannot be perceived from the image. For e.g., if the person is just walking<br>and the response contains that the person is jumping.<br>- Relation Hallucination: Object-to-object spatial relationships in the response that are not present in the image or cannot be seen or<br>perceived from the image. For example, if a book is on the left of a glass but the response contains it is on the right.<br>Now, please output your scores, a short rationale, and hallucinated content below in the following json format by filling in the placeholders<br>in [].<br>  { 'helpfulness': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }, 'clarity': { 'reason': '[your rationale]', 'score':<br>'[score from 1 to 5]' }, 'correctness': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }, 'depth': { 'reason': '[your<br>rationale]', 'score': '[score from 1 to 5]' }, 'engagement': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }, 'object<br>hallucinations': {'reason': '[elaborate reasoning of hallucination]', 'tokens': '[comma separated list of exact phrases from the response that<br>are hallucinated]'}, 'action/verb hallucinations': {'reason': '[elaborate reasoning of hallucination]', 'tokens': '[comma separated list of<br>exact phrases from the response that are hallucinated]'}, 'relation hallucinations': {'reason': '[elaborate reasoning of hallucination]',<br>'tokens': '[comma separated list of exact phrases from the response that are hallucinated]'} }<br>Only return the json and nothing else.<br>**----- End of picture text -----**<br>


Figure 12: Prompt used to evaluate image description benchmarks using GPT-4-Turbo as described in Section 3.2. 

## **`# Rephrase Prompt`** 

```
I will provide you with an image and an instruction that I aim to ask an AI agent to obtain a response from it. Referring to the input image,
rephrase the instruction such that the agent is asked to refer more to the image for solving the instruction to output a faithful response
(currently it is looking more at the instruction). Instruction: {inst}
```

```
Please output a json with a single key called 'rephrased': and the output as the rephrased instruction.
```

Figure 13: Prompt used to evaluate the performance of LLMs on rephrased instructions using GPT-4-Turbo as described in Section 4.2. 

## H DATASET DETAILS 

**VaLLu** We propose VaLLu benchmark which is sourced from Oven, MMMU, MMC, MathVista, HallusionBench, MATH-Vision and MME. This dataset is licensed under all the licenses of the original benchmarks that it was sourced from. 

**AMBER** AMBER (Wang et al., 2023a) is a benchmark for evaluating hallucinations in both the generative task and discriminative task of MLLMs. It provides comprehensive coverage of evaluations for various types of hallucination, including existence, attribute, and relation. Licensed under Apache2.0 License. 

**MMMU** MMMU (Yue et al., 2024) benchmark evaluates multimodal models on multi-discipline tasks and demands college-level subject knowledge and deliberate reasoning. It consists of 11.5k questions, spanning 30 subjects and 183 subfields, focusing on advanced perception and reasoning on domain-specific knowledge. Licensed under Apache-2.0 License. 

**MMC** MMC (Liu et al., 2023a) is a human-annotated benchmark comprising 9 distinct tasks to evaluate LVLM’s reasoning capabilities to comprehend visual charts. It comprises tasks such as chart information extraction, chart reasoning, contextual chart understanding, etc. It also offers two evaluation methods: GPT4 free-format general ability evaluation and multiple-choice QA format chart understanding. Licensed for research purposes. 

17 

Pre-print. Under Review. 

## **`# LLaMA3 Prompt`** 

```
I will provide you with a response from an AI agent which has been asked to describe an image. Please identify all the phrases that in the
image description that constitute the image. These phrases might be foreground and background objects, adverbial phrases, etc. Return them as
comma separated values. There should not be any additional information other than these values in the output. The response is as follows:
{response}
```

Figure 14: Prompt used to identify the visual elements in images using Llama3 for the algorithm described in Section 3.3. 

## **`# Image Description Prompt`** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0018-05.png)


**----- Start of picture text -----**<br>
<image><br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0018-06.png)


**----- Start of picture text -----**<br>
I have been given this image to complete the task described as: {inst}.<br>**----- End of picture text -----**<br>


```
To help me complete the task, describe the given image in detail. In case of real-world scenes, please include all foreground and background
objects in the description, their properties (like color, shape, etc.), their relations with other objects, their count, and all other
components in the image. In case of non-real-world scenes, like charts, graphs, tables, etc., please describe the table, mention all numbers
(if any), mention the written text, and all other details.
```

Figure 15: Prompt used to describe images using LVLMs for VDGD as described in Section 5. 

## **`# Maths Evaluation Prompt`** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0018-10.png)


**----- Start of picture text -----**<br>
Please act as an impartial judge and evaluate the quality of the response provided with respect to the input image. You will rate the quality<br>of the response on multiple aspects, such as Helpfulness, Clarity, Correctness and Depth. You will also output lists with hallucinated content<br>or wrongful information in the response with respect to the input image.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0018-11.png)


**----- Start of picture text -----**<br>
##Query: {instruction}<br>##Output: {answer}<br>## Evaluate<br>### Aspects<br>- Helpfulness: Rate the response based on how well it addresses the user query about the image and provides a relevant answer. A score of 5<br>indicates the answer fully aids the user, while a 1 suggests it offers little to no help.<br>- Clarity: Rate the response based on how well-structured it is, with ideas presented in a clear and coherent manner. A high score of 5 means<br>the answer is clear and logically structured, while a 1 suggests a disjointed or confusing reply.<br> - Correctness: Evaluate the correctness or accuracy of the response provided with respect to the image and the respective question asked about<br>the image. A perfect 5 indicates the response is entirely correct and accurate, while a 1 suggests it has significant errors or has not<br>provided an answer to the question asked at all.<br> - Depth: Determine the level of detail and thoroughness in the response. A score of 5 means the answer delves deeply into the aspects of the<br>input image for answering the question, while a 1 indicates it barely scratches the surface.<br>### Format Given the query and the input image, please rate the quality of the output by scoring it from 1 to 5, individually on **each<br>aspect**.<br>- 1: strongly disagree<br>- 2: disagree<br>- 3: neutral<br>- 4: agree<br>- 5: strongly agree<br> You are also asked to output all the hallucinated content or wrongful information in the response with respect to the input image.<br>Now, please output your scores, a short rationale, and hallucinated content below in the following json format by filling in the placeholders<br>in [].<br>  { 'helpfulness': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }, 'clarity': { 'reason': '[your rationale]', 'score':<br>'[score from 1 to 5]' }, 'correctness': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }, 'depth': { 'reason': '[your<br>rationale]', 'score': '[score from 1 to 5]' }, 'engagement': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' },<br>'hallucinated_phrases': {'reason': '[elaborate reasoning of different hallucinations]', 'tokens': '[comma separated list of exact phrases from<br>the response that are hallucinated]'},<br>Only return the json and nothing else.<br>**----- End of picture text -----**<br>


Figure 16: Prompt used to evaluate generations from cognitive benchmarks in Table 9. 

## **`# Evaluation Prompt`** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0018-14.png)


**----- Start of picture text -----**<br>
Please act as an impartial judge and evaluate the quality of an image description provided by an AI agent with<br>respect to an input image and an input query about the image. You will rate the quality of the response on a single<br>aspect, which is Informativeness. Precisely, I will provide you with an input image, a cognitive query about the input<br>image, and a description of the input image. The description is what an AI agent visualizes or recognizes in the input<br>image. You need to judge how informative the description is with respect to responding correctly to the input cognitive<br>query.<br>##Query: {image}<br>##Output: {prediction}<br>##Evaluate<br>-### AspectsInformativeness: Evaluate the informativeness of the image description. A perfect 5 indicates the information is<br>entirely correct and accurate, while a 1 suggests it has significant errors.<br>### Format Given the query, please rate the quality of the output by scoring it from 1 to 5 , individually on **each<br>aspect**.<br>-- 1: strongly disagree2: disagree<br>- 3: neutral<br>-- 4: agree5: strongly agree<br>Now, please output your scores and a short rationale below in a json format by filling in the placeholders in []: {<br>'informativeness': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }}"<br>**----- End of picture text -----**<br>


Figure 17: GPT evaluation prompt for evaluating informativeness of image descriptions generated by LVLMs. 

**MathVista** MathVista (Lu et al., 2023) is a robust mathematical reasoning evaluation benchmark that consists of challenging t ~~asks that require fne-grained, deep visual recogniti~~ on and compositional 

~~18~~ 

Pre-print. Under Review. 

## **`# Text-only Evaluation Prompt`** 

```
Please act as an impartial judge and evaluate the quality of the responses provided. You will rate the quality of the output on multiple
aspects, such as Helpfulness, Clarity, Correctness, Depth, and Engagement.
```


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0019-03.png)


**----- Start of picture text -----**<br>
##Query: {instruction}<br>##Output: {answer}<br>## Evaluate<br>### Aspects<br>- Helpfulness: Rate the response based on how well it addresses the users query and provides a relevant solution. A score of 5 indicates the<br>answer fully aids the user, while a 1 suggests it offers little to no help.<br>- Clarity: Rate the response based on how well-structured it is, with ideas presented in a clear and coherent manner. A high score of 5 means<br>the answer is clear and logically structured, while a 1 suggests a disjointed or confusing reply.<br> - Correctness: Evaluate the correctness or accuracy of the response provided with respect to the image and the respective question asked about<br>the image. A perfect 5 indicates the response is entirely correct and accurate, while a 1 suggests it has significant errors or has not<br>provided an answer to the question asked at all.<br> - Depth: Determine the level of detail and thoroughness in the response. A score of 5 means the answer delves deeply into the topic, while a 1<br>indicates it barely scratches the surface.<br> - Engagement: Assess how engaging and natural the response sounds in a conversational context. A high score of 5 reflects a response that<br>feels engaging and human-like in its tone, while a 1 indicates a robotic or boring reply. ,<br>### Format Given the query, please rate the quality of the output by scoring it from 1 to 5 , individually on **each aspect**.<br>- 1: strongly disagree<br>- 2: disagree<br>- 3: neutral<br>- 4: agree<br>- 5: strongly agree<br>Now, please output your scores and a short rationale below in a json format by filling in the placeholders in []: { 'helpfulness': { 'reason':<br>'[your rationale]', 'score': '[score from 1 to 5]' }, 'clarity': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' },<br>'correctness': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }, 'depth': { 'reason': '[your rationale]', 'score': '[score from<br>1 to 5]' }, 'engagement': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' } }<br>**----- End of picture text -----**<br>


Figure 18: Prompt used to evaluate generations for rephrased instructions on benchmarks like VaLLu, MMMU, MathVista, and SynthDoG using GPT-4-Turbo as described in Section 4.2. 

reasoning. It consists of 6141 examples derived from 31 multimodal datasets involving mathematics. Licensed under CC-BY-SA-4.0. 

**MATH-Vision** MATH-Vision (Wang et al., 2024a) dataset is a collection of 3,040 high-quality mathematical problems with visual contexts sourced from real math competitions. The dataset spans 16 distinct mathematical disciplines and is graded across 5 levels of difficulty to evaluate the mathematical reasoning abilities of Multimodal Large Language Models(MLLM). Licensed under MIT License. 

**HallusionBench** HallusionBench (Guan et al., 2023) is a comprehensive benchmark designed for the evaluation of image-context reasoning. The benchmark presents significant challenges to advanced large visual-language models (LVLMs), such as GPT-4V(ision), Gemini Pro Vision, Claude 3, and LLaVA- 1.5, by emphasizing nuanced understanding and interpretation of visual data. The benchmark comprises 346 images paired with 1129 questions, all meticulously crafted by human experts. Licensed under BSD-3-Clause. 

**LlavaBench** LlavaBench (Liu et al., 2023c) is a challenging evaluation suite designed to assess visual-language alignment and instruction-following capabilities of large visual-language models. It consists of two benchmarks: LLaVA-Bench (COCO), which includes 30 randomly selected images from COCO-Val-2014 paired with 90 questions covering conversational, descriptive, and complex reasoning tasks, and LLaVA-Bench (In-the-Wild), featuring 24 diverse images from various domains, including memes, paintings, and sketches, with 60 questions. 

**MM-Vet** MM-Vet (Yu et al., 2023) is an evaluation benchmark designed to assess large multimodal models on complex multimodal tasks. It defines six core vision-language capabilities—Recognition, Knowledge, OCR, Spatial Awareness, Language Generation, and Math—and examines their integration across 16 emergent tasks. MM-Vet employs an LLM-based evaluator for open-ended outputs, providing unified scoring across diverse question types and answer styles. The benchmark offers insights into the capabilities of different large multimodal model paradigms and models beyond simple performance ranking. 

**MME** Multimodal large language model Evaluation Benchmark (MME) (Fu et al., 2023) measures perception and cognition abilities on a total of 2 tasks(Perception and Congition) and 14 subtasks(Existence, Count, Position, Color, Poster, Celebrity, Scene, Landmark, Artwork, OCR, Commonsense Reasoning, Numerical Calculation, Text Translation and Code Reasoning). To prevent data leakage from use of public datasets for evaluation, the annotations of instruction-answer pairs are all manually designed. 

19 

Pre-print. Under Review. 

**Oven** Open-domain Visual Entity Recognition (OVEN) (Hu et al., 2023) is a task where a model needs to link an image onto a Wikipedia entity with respect to a text query. OVEN challenges models to select among six million possible Wikipedia entities, making it a general visual recognition benchmark with the largest number of labels. Licensed under MIT License. 

**SynthDoG** SynthDoG (Kim et al., 2022) is created by generating synthetic data which consists of various components: background, document, text, and layout. The background image is sampled from ImageNet, and the document is sampled from the collected paper photos. The text is sampled from Wikipedia. Layout is constructed using randomly stacks grids. The english variant of the dataset consists of 65.5k training and 500 validation entries. Licensed under MIT License. 

## I MODEL DETAILS 

**LLaVa-v1.** LLaVa-v1 (Liu et al., 2023c)[3] is an open-source LVLM trained by fine-tuning LLaMA/Vicuna (13B) on GPT-generated multimodal instruction-following data. It is initially trained on 558K filtered image-text pairs from LAION/CC/SBU and then, finetuned on 80K GPT-generated multimodal instruction-following data. Licensed under CC BY-SA 4.0 DEED. 

**LLaVa-1.5.** LLaVa-1.5 (Liu et al., 2023b)[4] builds on LLaVA-v1’s architecture by replacing linear projection design with a two-layer MLP. It also includes additional academic-task-oriented VQA datasets for VQA, OCR, and region-level perception, to enhance the model’s multimodal capabilities. Licensed under CC BY-SA 4.0 DEED. 

**LLaVa-1.6.** LLaVa-1.6 (Liu et al., 2024c)[5] re-uses the pretrained connector of LLaVA-1.5 and is instruction tuned on 158K GPT-generated multimodal data, 500K academic-task-oriented VQA data, 50K GPT-4V data and 40K ShareGPT data. LLaVa-1.6 has improved reasoning, OCR, and world knowledge. Licensed under CC BY-SA 4.0 DEED. 

**mPLUG-Owl2.** mPLUG-Owl2 (Ye et al., 2023)[6] is the first Multi-modal Large Language Model that performs well in both pure-text and multi-modal scenarios. mPLUG-Owl2 is pretrained 400 million image-text pairs from: Conceptual Captions (CC3M/CC12M), COCO, Laio-en, COYO, DataComp and for instruction tuning 5 types of data are used: image captioning, mage question answering, region-aware QA, multi-modal instruct data and text-only instruct. Licensed under MIT License. 

**InternLM-X.** InternLM-X (Zhang et al., 2023)[7] is a vision-language large model based on InternLM2 (Cai et al., 2024) for advanced text-image comprehension and composition. InternLM-X is pre-trained on 1.1 billion images alongside 77.7 billion text tokens, including both public datasets and in-house concept data followed by supervised fine-tuning using multi-task training and instruction tuning. Licensed under Apache-2.0 License. 

**CogVLM.** CogVLM (Wang et al., 2023b)[8] is a powerful open-source visual language model. CogVLM-17B has 10 billion vision parameters and 7 billion language parameters. CogVLM is pre-trained using 1.5B text-image pairs and instruction tuned on various visual question-answering datasets. CogVLM achieves state-of-the-art performance on 10 classic cross-modal benchmarks. Licensed under Apache-2.0 License. 

**Llama-2-7b.** Llama-2-7b (Touvron et al., 2023)[9] is an open source fine-tuned generative text model with 7 billion parameters. It is an auto-regressive language model that uses an optimized transformer architecture. The fine-tuned version use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) for helpfulness and safety. Licensed under Llama 2 community license agreement. 

> 3https://huggingface.co/liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview 

> 4https://huggingface.co/liuhaotian/llava-v1.5-7b 

> 5https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b 

> 6https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b 

> 7https://huggingface.co/internlm/internlm-xcomposer2-vl-7b 

> 8https://huggingface.co/THUDM/cogvlm-chat-hf 

> 9https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 

20 

Pre-print. Under Review. 

**Vicuna-7b-v1.5.** Vicuna-7b-v1.5 (Zheng et al., 2023)[10] is a 7b parameter open-sourced model and fine-tuned from Llama 2 with supervised instruction fine-tuning and is an auto-regressive language model based on the transformer architecture. The training data is around 125K conversations from ShareGPT. Licensed under Apache-2.0 License. 

**InternLM2-7B.** InternLM-7B (Cai et al., 2024)[11] InternLM2 is a open-source 7 billion parameter chat model with a 200k context window and achieves good performance in the“Needle-in-a-Haystack” test(on long-context tasks like LongBench and L-Eval) and is trained using Conditional Online RLHF (COOL RLHF). Licensed under Apache-2.0 License. 

**CogVLM2.** CogVLM2 (Hong et al., 2024)[12] CogVLM2 is a new generation of visual language models based on Meta-LLaMA-3-8B-Instruct, offering significant improvements over previous versions. It supports 8K content length, image resolutions up to 1344x1344, and excels in both English and Chinese. CogVLM2 has achieved state-of-the-art performance on benchmarks like TextVQA, DocVQA, and MM-Vet. The open-source models, including CogVLM2 and CogVLM2Video, are available on GitHub, contributing to advancements in image and video understanding. Licensed under Apache-2.0 License. 

**Qwen2-VL.** Qwen2-VL (Wang et al., 2024b)[13] Qwen2-VL is the latest iteration of the Qwen-VL model, featuring state-of-the-art performance in image and video understanding tasks. It excels in processing images of various resolutions and aspect ratios and supports long-form video understanding for tasks like question answering and content creation. Key architectural updates include Naive Dynamic Resolution and Multimodal Rotary Position Embedding (M-ROPE), improving its multimodal processing capabilities. Available in 2, 7, and 72 billion parameter variants. Licensed under Apache-2.0 License. 

## J BASELINE DETAILS 

**VCD.** Visual Contrastive Decoding (VCD) (Sicong Leng et al., 2023) is a training-free method that contrasts output distributions of original/unaltered inputs and distorted visual inputs. VCD hypothesizes statistical bias and unimodal priors as the two essential causes of object hallucinations and reduces the dependence on them. This results in the generated content being closely grounded to visual inputs resulting more accurate outputs. Licensed under Apache-2.0 License. 

**OPERA.** Over-trust Penalty and a Retrospection-Allocation (OPERA) (Huang et al., 2023b) is a training-free approach that hypothesizes that hallucinations are closely tied to the knowledge aggregation patterns in the self-attention matrix and that generation of new tokens by models is majorly dependent on a few summary tokens(over-trust issue) but not all of the previous tokens. This methodology introduces a penalty term on the model logits during beam-search decoding to mitigate the over-trust issue and implements a rollback strategy to retrospect the presence of summary tokens. Licensed under MIT License. 

**Woodpecker.** Woodpecker (Yin et al., 2023b) is a training-free method that identifies and corrects hallucinations in the generated text. Unlike other training-free methodologies, Woodpecker works after the model generation process. This method works by creating a visual knowledge base of Question-Answer(QA) pairs which contain object-level and attribute-level claims about the input image followed by an LLM correcting hallucinations. 

**LRV.** LRV (Liu et al., 2024a), or Large-scale Robust Visual Instruction tuning, proposes a dataset of visual instructions generated by GPT4, covering 16 vision-and-language tasks with open-ended instructions and answers. Each instance has positive instruction samples and negative instructions for more robust visual instruction tuning. As a result, a model trained on this dataset proves to be robust to hallucinations. Licensed under BSD-3-Clause license. 

**LURE.** LVLM Hallucination Revisor (LURE) (Zhou et al., 2023b) attributes object hallucination to three factors: co-occurrence, uncertainty, and object position, and develops an object hallucination 

> 10https://huggingface.co/lmsys/vicuna-7b-v1.5 

> 11https://huggingface.co/internlm/internlm2-chat-7b 

> 12https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B 

> 13https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct 

21 

Pre-print. Under Review. 

revisor which converts potential hallucinations into accurate outputs. This method first creates a hallucination dataset by making modifications to the original data and trains the object hallucination revisor on this dataset. This method post-hoc rectifies object hallucinations in LVLMs by reconstructing less hallucinatory descriptions. 

**PAI.** Pay Attention to Image (PAI) (Liu et al., 2024d) is a training-free algorithm designed to reduce hallucinations in LVLMs. PAI intervenes during the inference process to make it more imagecentric by amplifying attention to image tokens in the self-attention layers of LVLMs. By adjusting attention weights and subtracting logits of text-only inputs from multi-modal inputs, PAI mitigates the issue of “text inertia,” where outputs are overly influenced by context text. This method enhances image representation during generation, leading to more accurate, visually grounded outputs without requiring additional training or external tools. 

**HALC.** HALC (Chen et al., 2024a) is a novel decoding algorithm designed to mitigate object hallucinations (OH) in LVLMs. It integrates a robust auto-focal grounding mechanism to correct hallucinated tokens locally and a specialized beam search algorithm to reduce OH globally while preserving text generation quality. HALC can be easily integrated into any LVLM without additional training, addressing all three types of OH—existence, attribute, and relationship. Extensive experiments show that HALC outperforms state-of-the-art methods across multiple benchmarks. 

## K ADDITIONAL DETAILS 

## K.1 LOGIT SPACE FOR VR AND COGNITIVE PROMPTS 

From our in-depth analysis of the logit space, we see that language hallucinations in LVLMs have a unique characteristic – the probability of the most confident token in the vocabulary marginally decreased from the probability of the same token in the base LLM when prompted with the same prior context (the most confident token by the LVLM will also be the most confident token by the language-only base LLM as the token on a language hallucination). Thus, decoding-based mitigation methods like OPERA and VDD benefit from this unique property, and contrasting tokens help mitigate hallucinations. 

## K.2 QUALITATIVE COMPARISON OF GREEDY VS VCD VS VDGD DECODING 

In this subsection, we illustrate several instances (figures 20, 22, 23, 21, 24) from VaLLu and compare their responses for LLaVa-1.5 with Greedy, VCD, and VDGD Decoding. 

22 

Pre-print. Under Review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0023-01.png)


**----- Start of picture text -----**<br>
Question:  Baxter Company has a relevant range of<br>production between 15,000 and 30,000 units. The following<br>cost data represents average variable costs per unit for 25,000 Question:  Where<br>units of production. If 15,000 units are produced, what is the is it typically<br>variable cost per unit? found?<br>Greedy Decoding:  To find the variable cost per unit, we need to calculate the total variable cost<br>and divide it by the number of units produced. \n Total variable cost = (Direct materials*Direct Greedy Decoding:  The image<br>labor) + (Indirect materials*Fixed manufacturing overhead) + (Variable manufacturing shows a wasp on a yellow flower.<br>overhead*Number of units). \n Total variable cost = ($10 * $9) + ($3 * $6) + ($2 * 15,000) =<br>$30,118. \n  Variable cost per unit = Total variable cost/Number of units=30,118/15,000 =<br>$2.00. Therefore, if 15,000 units are produced, the variable cost per unit is $2.00. VDGD Decoding:  The wasp is<br>typically  found  in  gardens,<br>meadows,  and  other  natural<br>VDGD Decoding:  To find the variable cost per unit, we need to calculate the total variable cost habitats where it can find flowers<br>and divide it by the number of units produced. \n Total variable cost = (Direct labor cost + to feed on. It may also be found<br>Indirect materials cost + Variable manufacturing overhead cost + Variable sales commissions near human settlements if the<br>cost)*Number of units. \n Total variable cost = ($9 + $3 + $2 + $14)*25,000 = $950,000.\n Variable cost per unit = Total variable cost/Number of units = $950,000/15,000 = $63.33. environment  provides  suitable<br>\n Therefore, if 15,000 units are produced, the variable cost per unit would be $63.33. nesting sites and food sources.<br>Description:  The image is a table that lists the average cost per unit for various Description:  The image depicts<br>production-related expenses. The table has two columns: one for the type of expense a wasp perched on a yellow<br>and another for the average cost per unit. The types of expenses listed are direct<br>materials, direct labor, indirect materials, fixed manufacturing overhead, variable flower. The wasp is brown and<br>manufacturing overhead, fixed selling and administrative expenses, and variable sales black in color, with its wings<br>commissions. InInThe average cost per unit for each expense is as follows:\n- Direct spread out as it sits on the flower.<br>materials: $10\n- Direct labor: $9\n- Indirect materials: $3\n- Fixed manufacturing There are several other yellow<br>overhead:  $6\n-  Variable  manufacturing  overhead:  $2\n-  Fixed  selling  and flowers in the background,<br>administrative expenses: $8\n- Variable sales commissions: $14\n\nThere are no other<br>objects or text in the image. creating a vibrant scene.<br>**----- End of picture text -----**<br>


Figure 19: Qualitative example for VaLLu benchmark also showing the descriptions. 

**Question:** The pressure of a sample of gas is measured at sea level with a closed-end manometer. The liquid in the manometer is mercury. Determine the pressure of the gas in torr. **Greedy Decoding:** The image shows a closed-end manometer with a silver tube and a mercury-filled gauge. The manometer is measuring the pressure of a gas, specifically, the pressure of a gas at sea level. The tube is connected to the manometer, and the pressure is indicated by the height of the mercury in the tube. The manometer is measuring the pressure in torr, which is a unit of pressure. The image provides a clear visual representation of the manometer and its function in measuring gas pressure. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0023-04.png)


**VCD Decoding:** The pressure of a sample of gas is measured at sea level with a closed-end manometer filled with mercury. The manometer has a scale that measures the pressure in units of torr. The manometer is connected to the gas sample through a metal tube that has a closed end to prevent the mercury from leaking out. The manometer is placed in a vacuum chamber to minimize external influences on the measurement. The pressure reading is shown on the manometer scale, which is typically a logarithmic scale that allows for a wide range of pressure measurements to be shown in a compact space. The manometer is a useful tool for measuring the pressure of gases in various applications, such as scientific research and industrial processes. 

**VDGD Decoding:** The image shows a closed-end manometer with a metal tube containing mercury liquid, measuring the pressure of a gas at sea level. The tube is connected to a metal pipe with a scale on it, indicating the pressure in torr units. The manometer is placed on a white background, making it the main focus of the image. The pressure of the gas is 26.4 torr. 

Figure 20: Qualitative example 1 for VaLLu benchmark. 

**Question:** What is the net sales figure for Stanley Black and Decker? **Greedy Decoding:** The net sales figure for Stanley Black and Decker is $10.5 billion. **VCD Decoding:** The net sales figure for Stanley Black and Decker is $1.5 billion in 2010. **VDGD Decoding:** The net sales figure for Stanley Black and Decker is $13.3 billion in 2010. 

Figure 21: Qualitative example 2 for VaLLu benchmark. 

23 

Pre-print. Under Review. 

**Question:** Dose response data was collected during the preclinical testing of four drugs for the treatment of acute heart failure.  Which drug studied was the most efficacious? 

**Greedy Decoding:** Based on the dose-response data, the most efficacious drug studied was the one with the highest efficacy. In the image, there are four different colored lines representing the efficacy of the four drugs. The line with the highest efficacy is the one that has the most significant effect on the outcome. This information can be used to determine which drug is the most effective in treating acute heart failure. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0024-03.png)


**VCD Decoding:** The most efficacious drug studied was the one with the highest log dose (log D). Log dose is a measure of the concentration of a substance in the blood relative to the concentration in the liver, which is an important factor in determining the efficacy of a drug. In this case, the log dose for the green line is higher than the log dose for the other lines, indicating that the green line represents the most effective drug among the four studied for the treatment of acute heart failure. 

**VDGD Decoding:** Based on the image, the most efficacious drug studied was D. The graph shows that the dose-response relationship for D was the steepest, indicating that the drug had the highest efficacy at lower doses. This suggests that D may have a more potent effect on acute heart failure treatment compared to the other three drugs, A, B, and C. 

Figure 22: Qualitative example 3 for VaLLu benchmark. 

**Question:** Three identical rectangles are put together to form rectangle ABCD, as shown in the figure below. Given that the length of the shorter side of each of the smaller rectangles 5 feet, what is the area in square feet of rectangle ABCD? **Greedy Decoding:** The area of rectangle ABCD is 12 square feet. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0024-08.png)


**VCD Decoding:** To find the area of the rectangle formed by the three identical rectangles, we need to find the length and width of each rectangle and then multiply them together. In the image, the length of the smaller side of each rectangle is given as 5 feet. Since there are three identical rectangles, we can find the total length by adding the lengths together: Length = (3 x 5 feet) = 15 feet. Now we need to find the width of each rectangle. Since the length is the same for all three rectangles, we can find the width by dividing the length by the number of rectangles: Width = Length / Number of Rectangles = 15 feet / 3 = 5 feet. Now we have the dimensions for each rectangle: length is 15 feet and width is 5 feet. To find the area of the rectangle formed by these three identical rectangles, we multiply the length and width together: Area = Length x Width = 15 feet x 5 feet = 75 square feet. So the area in square feet of the rectangle formed by the three identical rectangles is 75 square feet. 

**VDGD Decoding:** Given that the length of the shorter side of each of the smaller rectangles is 5 feet, the area in square feet of rectangle ABCD can be calculated by multiplying the length of the longer side by the width of the shorter side. The length of the longer side is 10 feet, and the width of the shorter side is 5 feet. Therefore, the area of rectangle ABCD is 50 square feet. 

Figure 23: Qualitative example 4 for VaLLu benchmark. 

24 

Pre-print. Under Review. 

**Question:** Where can the animal be found? 

**Greedy Decoding:** The animal, a duck, can be found in a body of water, specifically a lake. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0025-03.png)


**VCD Decoding:** The animal can be found in the wild, specifically in the tropical rainforests of Central and South America. It is also found in some parts of Africa and Asia. The animal is typically found in areas with plenty of water and vegetation, such as rivers and swamps. It is also known to be found in some parts of the United States and Canada, but these populations are generally small and isolated. The animal is also found in some parts of the world where it is kept in captivity, such as zoos and wildlife parks. 

**VDGD Decoding:** The animal can be found in a body of water, possibly a lake or a pond, surrounded by other birds. 

Figure 24: Qualitative example 5 for VaLLu benchmark. 

## K.3 NOISY EXAMPLES IN EXISTING DATASETS 

To maintain a high quality of the benchmark and eliminate noisy examples, we performed a manual review of the existing datasets. This was crucial to ensure that the data used in our paper was reliable. It was also essential for the validity of our research findings. We removed questions that did not have a definite ground truth answer. We also made sure that the questions were answerable using a combination of general knowledge and reasoning. This filtering was done by the paper authors and has been approved by our institution’s Institutional Review Board (IRB). We share a few examples (figures: 25, 26, 27, 28, 29, 30, 31, 32) of noise in the existing datasets below: 

**Question:** You are a consultant to a large manufacturing corporation that is considering a project with the following net after-tax cash flows (in millions of dollars). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0025-10.png)


**Sanity Rationale:** The question asks for the net present value of a project and provides specific financial parameters such as project beta, risk-free rate, and expected market return. However, the image provided shows information about two different companies, including their forecasted returns, standard deviations of returns, and betas, which are unrelated to the specific project's cash flows mentioned in the question. The image neither shows the net after-tax cash flows for the project in question nor any relevant data to perform a net present value calculation based on the project's described financial conditions. Thus, the image does not support answering the question adequately. 

Figure 25: Noisy example 1 for MMMU benchmark. 

**Question:** Fig. 8.15 shows the distribution of 12 spot heights with a regular 20 m spacing covering a rectangular area which is to be graded to form a horizontal plane having an elevation of 10.00 m. Calculate the volume of the earth. 

**Sanity Rationale:** The image provides the elevations at various points and the distances between these points, but it does not provide enough information to calculate the volume of earth needed to grade the area to a horizontal plane at an elevation of 10.00 m. The calculation of volume would require additional information such as the method of interpolation between the spot heights and the specific earthwork calculation techniques to be used. Since the question cannot be definitively answered solely with the provided image and general knowledge, the response is no. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0025-15.png)


Figure 26: Noisy example 2 for MMMU benchmark. 

25 

Pre-print. Under Review. 

**Question:** What is the current employment rate in the UK's travel and tourism sector? 

**Sanity Rationale:** The question inquires about the current employment rate in the UK's travel and tourism sector. The provided image, however, shows data from 2019 and does not specify whether the numbers represent rates or absolute job counts, nor does it give any indication of the employment rate. Additionally, the image is historical and may not accurately reflect the current employment situation. Therefore, the image does not provide the necessary information to answer the question about the current employment rate. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0026-03.png)


Figure 27: Noisy example 1 for MMC benchmark. 

**Question:** Did the population of 40-59 year olds change from 2016 to 2020? 

**Sanity Rationale:** The image provided is a solid color block without any distinguishable data, axes, or markers that would allow for the evaluation of population changes over time. Therefore, it is impossible to answer the question about the change in the population of 40-59 year olds from 2016 to 2020 using this image. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0026-07.png)


Figure 28: Noisy example 2 for MMC benchmark. 

**Question:** Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. Question: What is the age gap between these two people in image? (Unit: years) 

**Sanity Rationale:** The question asks for the age gap between two people shown in the image, but it's impossible to accurately determine their ages just by looking at the photograph. Age estimation from a photograph without additional information such as the date the photograph was taken or personal details about the individuals is highly speculative. Therefore, the question does not meet the condition of having a definite, ground truth answer. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0026-11.png)


Figure 29: Noisy example 1 for MathVista benchmark. 

**Question:** Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end. Question: In the figure, what is the magnitude of the potential difference across the $20\Omega$ resistor? Answer in unit of W (3 sig.fig.). 

**Sanity Rationale:** The question asks for the magnitude of the potential difference across a resistor using a unit of 'W' which stands for watts, a unit of power, not potential difference. The correct unit for potential difference should be volts (V). Therefore, the question contains a fundamental error in its request for a unit, making it impossible to answer as stated. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0026-15.png)


Figure 30: Noisy example 2 for MathVista benchmark. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0026-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0026-18.png)


**----- Start of picture text -----**<br>
Question:  According to the positive sequence of the images, are they skating to right? Answer in one word.<br>**----- End of picture text -----**<br>


**Sanity Rationale:** The question asks whether the individuals in the image are skating to the right, based on the sequence of images. However, the images do not show any skating activity. The sequence depicts two individuals performing a stunt or trick where one person runs and jumps over another lying on the ground. There are no skates visible, and the activity shown does not involve skating. Therefore, the question is not answerable from the image as it assumes an incorrect context and activity that is not depicted. 

Figure 31: Noisy example 1 for HallusionBench benchmark. 

26 

Pre-print. Under Review. 

**Question:** If angle BAF equals to angle CAF, is CF equal to BF? Please answer yes or no. 

**Sanity Rationale:** Based on the image provided, the question asks whether CF is equal to BF given that angle BAF equals angle CAF. The image shows a triangle ABC with a point F on line segment BC and other lines and points within the triangle. However, the image does not provide any explicit information or annotations (like length markings or angle measurements) that confirm whether line segments CF and BF are equal, nor does the condition of BAF equaling CAF automatically imply that CF equals BF without additional context or geometric properties being specified (such as triangle ABC being isosceles with AB = AC). Therefore, the question cannot be definitively answered just by examining the image and applying basic geometric reasoning. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0027-03.png)


Figure 32: Noisy example 1 for HallusionBench benchmark. 

## K.4 ADDITIONAL DETAILS ON ALGORITHM 1 

In this Section, we describe our algorithm verbally. 

**Token Distribution Analysis.** We perform the token-distribution analysis proposed by (Lin et al., 2023). Specifically, for a given instruction-response-image triplet, the instruction _i_ = { _i_ 1, _i_ 2, _· · ·_ } and the image _v_ is first input to the aligned (or fine-tuned) model to obtain its response _r_ = { _r_ 1, _r_ 2, _· · ·_ } via greedy decoding. Next, for each position _t_ in the response, a ‘context’ at this position is defined as to be _xt_ = _i_ + { _r_ 1, _· · ·_ , _rt−_ 1}. This “context” is then input to the base model (model from which the aligned model was instruction-tuned) to obtain its probability distribution for predicting the next token at position _t_ , Pbase. We then define **Base Rank** as follows: Rank in Pbase of the token at _t_ with the maximum Palign value. With the base rank denoted as _η_ , the unshifted, marginal and shifted tokens are defined as when ( _η_ = 1), (1 _< η ≤_ 3) and ( _η >_ 3) respectively. 

**Obtaining Hallucinated Phrases.** Fig. 12 illustrates the prompt used for AMBER evaluation, where an LVLM is prompted for a description of an input image. In addition to scores, our judge, GPT-4, also returns the exact hallucinated phrases by the LVLM. We divide these hallucinated phrases into 3 types: **Object Hallucinations** – There are hallucinated visual elements which are objects. **Relation Hallucinations** – **Action/Verb Hallucination** – Hallucinated visual elements that are not objects but just actions or verbs, e.g., walking, etc. **Relation Hallucination** – Hallucinated visual elements that define relationships (spatial or other kinds) between objects. 

**Filtering visual elements in Hallucinated Phrases.** We prompt LLaMa-3 with the image description output by the LVLM, to extract the phrases from the image description that are visual elements. This ensures filtering out stop-words and all other words that are not visual elements and thus aids our algorithm in getting a precise count of each category of hallucination. 

**Causal Categorization of Hallucinations.** Next, based on the Base Rank formulation and the exact hallucinated phrases detected by our judge (GPT-4), we propose an algorithm (Algorithm 1) to automatically categorize identified hallucinations into the 4 different categories. Precisely, for every hallucinated phrase returned by the judge, we first check if the word in the phrase is a visual element. We do this using the list of visual elements output by LLaMa-3. Next, for the first token of each word of the hallucinated phrase, we check the Base Rank of the token. 

1. If the Base Rank is 0, we attribute the phrase to be a language hallucination. This is as simple as the phrase was forced to be generated by the base LLM itself without looking at the image. 

2. Next, if the Base Rank is more than 0 and the hallucination can be found in the top-k retrieved elements from the IT dataset the model was aligned using, we attribute it to be an IT hallucination (Ghosh et al., 2024). To retrieve top-k image-instruction pairs that are closest to the prompt image, we employ CLIP (Radford et al., 2021) similarity. Specifically, we calculate the similarity between the image embedding of the image associated with the input prompt and the text embedding of the description/response associated with the IT instances. Next, for every prompt, we retrieve the top-25 IT instances and search the hallucinated phrase returned by the judge (in descriptions/responses) using string matching. We show how varying values of _k_ change the final counts for our hallucination categories in Appendix K.5. 

27 

Pre-print. Under Review. 

   3. Next, if the Base Rank is more than 0, the hallucination is not an IT hallucination and the hallucination is a **Relation Hallucination** , we attribute it to be a style hallucination. LVLMs fine-tuned on datasets synthesized from stronger closed-source LVLMs generally learn to mimic their style. One such style is adding more details to an identified object, by relating it to other objects or world knowledge. Our in-depth analysis shows that LVLMs, striving for lengthier, more detailed responses to imitate style, hallucinate and generate wrongful relations for identified objects. Our relation hallucinations returned by the judge precisely capture this, and thus we attribute relation hallucinations with a base rank of more than 0 to be Style hallucinations. 

   4. Finally, if the Base Rank is more than 0, the hallucination is identified as an Obejct hallucination by the judge, we attribute it to be a Vision Hallucination. These are caused by ineffective recognition of objects in the image (due to low-resolution images, weak encoder, etc.) 

- K.5 RESULTS FOR VARIOUS VALUES OF _k_ FOR IT HALLUCINATIONS 

Fig. 33 shows that even for higher and lower values of _k_ , our hypothesis and claims stay unchanged as changing the value of _k_ does not affect the overall frequency distribution of categories. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0028-05.png)


**----- Start of picture text -----**<br>
Language Hallucination Style Hallucination Vision Hallucination IT Halluciantion<br>650<br>600<br>550<br>500<br>450<br>400<br>350<br>300<br>250<br>200<br>150<br>100<br>50<br>0<br>LLaVA-1.5 (k=10) LLaVA-1.5 (k=25) LLaVA-1.5 (k=40) OPERA (k=10) OPERA (k=25) OPERA (k=40)<br>**----- End of picture text -----**<br>


Figure 33: Frequency comparison of hallucination categories for top- _k_ =10,25,40 

## K.6 ELBOW METHOD FOR LOGIT ANALYSIS 

The Knee/Elbow Detection Method operates by plotting the cumulative distribution of the sorted probabilities and seeking the point that maximizes the distance from the line connecting the first and the last points of this cumulative plot. Conceptually, this corresponds to finding the point of maximum curvature on the plot, indicative of the most pronounced change in the distribution’s progression. It offers a robust approach by identifying the point in a sorted probability vector where the rate of decrease in probabilities shows a significant bend or ’elbow’. This method is particularly useful when the probabilities are distributed such that the top few are similar and significantly higher than the rest, which then decrease sharply. The method proves to be effective in our experiments with multinomial sampling from a distribution, where it allows us to focus on the probabilities that correspond to the tokens that are most likely to be selected by sampling. This method ensures that the chosen cutoff _k_ dynamically adjusts to the specific characteristics of each probability distribution encountered, aligning with the overarching goal of maintaining sampling relevance and computational efficiency. 

## K.7 EXAMPLES OF REPHRASED PROMPTS 

To determine whether LVLM hallucinations stem from the limitations of the base LLM or from alignment issues, we ask GPT-4 to rewrite the instructions in prompts in such a way that the base LLM can answer the question without any image input as described in section 4.2. We share a few examples (figures: 34, 35, 36, 37, 38, 39, 40, 41) of rephrased prompts in the existing datasets below: 

28 

Pre-print. Under Review. 

**Question:** The composition of the Fingroup Fund portfolio is as follows:  The fund has not borrowed any funds, but its accrued management fee with the portfolio manager currently totals $30,000. There are 4 million shares outstanding. What is the net asset value of the fund? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0029-02.png)


**Rephrased Instruction:** A portfolio consists of four types of stocks. Stock A has 200,000 shares priced at $35 each, Stock B has 300,000 shares priced at $40 each, Stock C has 400,000 shares priced at $20 each, and Stock D has 600,000 shares priced at $25 each. The fund has not borrowed any funds, but has an accrued management fee of $30,000. There are 4 million shares outstanding. Calculate the net asset value per share for this fund. 

Figure 34: Rephrased prompt example for VaLLu benchmark. 

**Question:** After school, Maya and Naomi headed to the beach, 6 miles away. Maya decided to bike while Naomi took a bus. The graph below shows their journeys, indicating the time and distance traveled. What was the difference, in miles per hour, between Naomi's and Maya's average speeds? 

**Rephrased Instruction:** Maya and Naomi traveled to a destination 6 miles away. Maya biked and completed the journey in 30 minutes. Naomi took a bus and completed the journey in 10 minutes. Calculate the difference, in miles per hour, between Naomi's and Maya's average speeds. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0029-07.png)


Figure 35: Rephrased prompt example for VaLLu benchmark. 

**Question:** If this area was selectively destroyed on both sides, could vomiting be induced by either pharyngeal tickling or gastric distress? 

**Rephrased Instruction:** If the area of the brainstem that includes the dorsal motor nucleus of the vagus nerve (marked as X in a typical diagram) and the area postrema (often indicated with an arrow near the fourth ventricle in diagrams) were selectively destroyed on both sides, could vomiting still be induced by pharyngeal tickling or gastric distress? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0029-11.png)


Figure 36: Rephrased prompt example for MMMU benchmark. 

**Question:** Suppose a three-factor model is appropriate to describe the returns of a stock. Information about those three factors is presented in the following:What is the systematic risk of the stock return? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0029-14.png)


**Rephrased Instruction:** Given a three-factor model with the following factors and their respective beta values: GDP with a beta of 0.0000734, Inflation with a beta of -0.90, and Interest rates with a beta of -0.32, can you calculate the systematic risk of the stock return? 

Figure 37: Rephrased prompt example for MMMU benchmark. 

29 

Pre-print. Under Review. 

**Question:** Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. Question: How many objects are preferred by more than 5 people in at least one category? 

**Rephrased Instruction:** Consider a bar graph with five objects labeled as veto, saw, calf, oxen, and fusion, each represented in two categories: grease and picnic. The number of people who prefer each object in each category ranges from 0 to 10. How many objects are preferred by more than 5 people in at least one of these categories? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0030-03.png)


Figure 38: Rephrased prompt example for MathVista benchmark. 

**Question:** Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end. Question: what is the difference between the largest and smallest number of cases? 

**Rephrased Instruction:** Given the number of cases in thousands for various cities: Quezon City has 106.71 thousand cases, Cavite has 72.34 thousand cases, City of Manila has 66.88 thousand cases, Laguna has 58.31 thousand cases, and Rizal has 55.57 thousand cases. What is the difference between the largest and smallest number of cases, expressed as a floating-point number with two decimal places? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0030-07.png)


Figure 39: Rephrased prompt example for MathVista benchmark. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0030-09.png)


Figure 40: Rephrased prompt 1 for SynthDoG benchmark. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs_images/iclr_2025_Visual_Description_Grounding_Reduces_Hallucinations_and_Boosts_Reasoning_in_LVLMs.pdf-0030-11.png)


Figure 41: Rephrased prompt 2 for SynthDoG benchmark. 

30 

Pre-print. Under Review. 

## K.8 COMPUTE INFRASTRUCTURE 

All our analysis, inference and baseline experiments are conducted on a node of 4 NVIDIA RTX A6000 GPUs, with 128GB RAM and 10 CPU cores. The evaluations are conducted using gpt-4turbo-2024-04-09 model. 

## L ADDITIONAL RESULTS 

## L.1 COMPARISON OF VARIOUS LVLMS ON REPHRASED PROMPTS 

Table 4 provides quantitative values for rephrased prompts without image input for various LVLMs, detailed in section 4.2 of the main paper. 

|**Dataset**|**Model**|**Helpfulness**|**Clarity**|**Factuality**|**Depth**|**Engagement**|
|---|---|---|---|---|---|---|
||LLaVA-v1.5-t|3.05|4.02|2.83|2.70|**3.85**|
|VaLLu|mPLUG-Owl2-t|3.70|4.15|3.14|3.11|3.24|
||InternLM-X-t|**4.28**|**4.58**|**3.66**|**3.62**|3.57|
||LLaVA-v1.5-t|1.97|3.25|1.69|1.87|**3.25**|
|MMMU|mPLUG-Owl2-t|2.48|3.05|1.63|2.22|2.64|
||InternLM-X-t|**3.51**|**3.93**|**2.43**|**2.98**|3.01|
||LLaVA-v1.5-t|2.53|3.58|2.21|2.10|**3.44**|
|MathVista|mPLUG-Owl2-t|3.56|4.05|3.04|2.52|2.79|
||InternLM-X-t|**4.39**|**4.65**|**3.99**|**2.91**|2.95|
||LLaVA-v1.5-t|3.38|4.48|2.95|2.32|3.35|
|Oven|mPLUG-Owl2-t|3.62|4.56|2.99|2.47|3.37|
||InternLM-X-t|**3.80**|**4.58**|**3.76**|**2.89**|**3.52**|



Table 4: Performance comparison of various LVLMs on rephrased prompts without images as described in Section 4.2 

- L.2 COMPARISON OF VARIOUS LVLMS ON VALLU BENCHMARK WITH NO IMAGE INPUT. 

Table 5 provides comparison of various LVLMs on VaLLU benchmark when prompted with no image but image descriptions are provided. 

|**Methodology**|**LLaVA-v1**|**LLaVA-v1.5**|**LLaVA-v1.6**|**mPLUG-Owl2**|**InternLM-X**|**CogVLM**|
|---|---|---|---|---|---|---|
|Vanilla-greedy|1.67|1.78|2.07|1.76|2.02|2.23|
|VCD|1.31|1.29|1.40|1.42|2.04|2.21|
|OPERA|1.65|1.43|2.29|1.91|2.30|2.44|
|Woodpecker|1.77|1.85|2.29|2.02|2.41|2.60|



Table 5: Comparison of LVLMs performance when prompted without an image but image descriptions are provided on VaLLu benchmark. The values correspond to the "Factuality" metric. 

## L.3 COMPARISON OF VARIOUS LVLMS ON IMAGE DESCRIPTION 

Table 6 provides quantitative values for faithful image description generations for various LVLMs on multiple benchmarks, detailed in section 4.2 of the main paper. 

- L.4 COMPARISON OF LVLMS ON DIFFERENT CAPABILITIES ON THE PROPOSED VALLU BENCHMARK 

Table 7 provides a comparison of LVLMs on different capabilities on the proposed VaLLu benchmark. 

31 

Pre-print. Under Review. 

|**Dataset**|**Model**|**Helpfulness**|**Clarity**|**Factuality**|**Depth**|**Engagement**|
|---|---|---|---|---|---|---|
||LLaVA-v1|3.79|4.46|3.2|3.27|0.96|
||LLaVA-1.5|4.29|4.73|3.91|3.44|1.26|
|AMBER|LLaVA-1.6<br>mPLUG-Owl2|4.30<br>4.02|4.78<br>4.58|3.85<br>3.48|4.29<br>3.27|1.27<br>0.71|
||InternLM-X|4.59|4.86|4.37|4.00|1.40|
||CogVLM|4.68|4.93|4.49|3.96|1.28|
||GPT-4-Turbo|**4.89**|**4.96**|**4.79**|**4.64**|**1.50**|
||LLaVA-v1|1.11|2.09|1.11|1.14|0.47|
||LLaVA-1.5|1.07|2.24|1.17|1.15|0.48|
|OCR|LLaVA-1.6<br>mPLUG-Owl2|2.51<br>1.21|2.92<br>1.15|2.63<br>1.16|1.99<br>1.10|1.11<br>0.64|
||InternLM-X|1.11|1.12|1.12|1.03|0.67|
||CogVLM|3.12|3.04|2.68|2.53|1.47|
||GPT-4-Turbo|**4.15**|**4.02**|**3.78**|**3.98**|**1.67**|
||LLaVA-v1|1.77|2.73|1.50|1.65|0.46|
||LLaVA-1.5|2.45|3.34|2.00|2.23|0.59|
|MMMU|LLaVA-1.6<br>mPLUG-Owl2|3.54<br>2.94|3.98<br>3.64|2.95<br>2.56|3.39<br>2.33|0.99<br>0.82|
||InternLM-X|3.60|4.10|3.24|3.15|1.05|
||CogVLM|4.03|4.38|3.57|3.43|1.03|
||GPT-4-Turbo|**4.86**|**4.92**|**4.73**|**4.56**|**1.15**|
||LLaVA-v1|1.95|2.84|1.95|1.85|0.76|
||LLaVA-1.5|2.57|3.47|2.23|2.36|0.98|
|MathVista|LLaVA-1.6<br>mPLUG-Owl2|3.68<br>3.14|4.07<br>3.85|3.17<br>2.78|3.48<br>2.46|1.12<br>1.03|
||InternLM-X|3.91|4.32|3.45|3.35|1.23|
||CogVLM|4.20|4.52|3.78|3.66|1.24|
||GPT-4-Turbo|**4.54**|**4.97**|**4.12**|**4.10**|**1.35**|
||LLaVA-v1|1.81|2.90|1.51|1.75|0.56|
||LLaVA-1.5|2.66|3.49|2.21|2.42|0.71|
|MMC|LLaVA-1.6<br>mPLUG-Owl2|4.09<br>3.63|4.32<br>4.15|3.50<br>3.32|3.90<br>2.60|1.09<br>0.97|
||InternLM-X|3.72|4.17|3.28|3.25|1.10|
||CogVLM|4.27|4.58|3.91|3.92|**1.32**|
||GPT-4-Turbo|**4.81**|**4.89**|**4.57**|**4.49**|1.12|



Table 6: Performance comparison of various LVLMs on popular benchmarks for _image description_ . While AMBER has real-world scenes, others do not. Experiment described in Section 3.2. 

|**Methodology**|**VP**|**VP+Info-Seek**|**VP+Info-Seek+Reasoning**|**VP+Reasoning**|
|---|---|---|---|---|
|LLaVA-v1|1.62|2.60|1.67|1.70|
|LLaVA-v1.5|2.59|3.18|1.50|1.76|
|LLaVA-v1.6|3.46|3.68|1.67|2.15|
|mPLUG-Owl2|2.56|3.23|1.67|1.94|
|InternLM-X|3.75|3.90|2.33|2.56|
|CogVLM|3.00|3.37|1.25|2.08|



Table 7: Comparison of LVLMs on different capabilities on the proposed VaLLu benchmark for the "Factuality" metric. 

32 

Pre-print. Under Review. 

## L.5 COMPARISON OF POST-TRUNCATION PROBABILITY STATISTICS ON AMBER AND MMMU BENCHMARKS 

Table 8 provides post-truncation probabilities for all LVLMs on AMBER and MMMU, this is an extension of the statistic shown in section 4.3 of the main paper. 

|**Dataset**|**Model**|**k**|**Variance**|**Range**|**Avg.**|
|---|---|---|---|---|---|
||LLaVA-v1|4.19|0.05|0.31|0.23|
||LLaVA-v1.5|3.91|0.07|0.39|0.29|
|AMBER|LLaVA-v1.6<br>mPLUG-Owl2|3.78<br>3.65|0.08<br>0.06|0.43<br>0.36|0.32<br>0.28|
||InternLM-X|3.56|0.09|0.44|0.36|
||CogVLM|3.16|0.91|0.47|0.38|
||LLaVA-v1|3.68|0.08|0.39|0.27|
||LLaVA-v1.5|3.53|0.09|0.41|0.23|
|MMMU|LLaVA-v1.6<br>mPLUG-Owl2|3.40<br>3.32|0.09<br>0.10|0.42<br>0.44|0.24<br>0.26|
||InternLM-X|3.26|0.14|0.46|0.28|
||CogVLM|3.16|0.19|0.47|0.25|



Table 8: Post-truncation probability statistics for various models using the elbow method. All values are computed across all hallucinated tokens in each dataset. 

## L.6 COMPARISON OF VALLU WITH MITIGATION BASELINES 

Table 9 provides quantitative values for comparison of VDGD and other baselines on the VaLLu benchmark, detailed in section 5.3 of the main paper. 

## L.7 COMPARISON OF MMMU WITH MITIGATION BASELINES 

Table 10 provides quantitative values for comparison of VDGD and other baselines on the MMMU benchmark, detailed in section 3.2 of the main paper. 

## L.8 COMPARISON OF MATHVISTA WITH MITIGATION BASELINES 

Table 11 provides quantitative values for comparison of VDGD and other baselines on the MathVista benchmark, detailed in section 3.2 of the main paper. 

## L.9 COMPARISON OF OVEN WITH MITIGATION BASELINES 

Table 12 provides quantitative values for comparison of VDGD and other baselines on the Oven benchmark, detailed in section 3.2 of the main paper. 

33 

Pre-print. Under Review. 

|**Model**|**Methodology**|**Helpfulness**|**Clarity**|**Factuality**|**Depth**|**Engagement**|
|---|---|---|---|---|---|---|
||Vanilla-greedy|2.61|3.69|1.95|1.92|1.25|
||Vanilla-sampling|2.64|3.73|1.86|1.94|1.27|
||VDD|1.70|2.55|1.47|1.81|**1.59**|
|LLaVA-v1|OPERA|2.60|3.75|2.04|1.89|1.30|
||Woodpecker|**2.68**|**3.81**|2.01|1.95|1.28|
||LRV|2.64|3.78|1.98|1.94|1.36|
||LURE|2.60|3.54|2.03|1.89|1.40|
||**VDGD (ours)**|2.38|3.39|**2.16**|**2.46**|1.34|
||Vanilla-greedy|2.64|3.72|2.03|2.03|1.25|
||Vanilla-sampling|2.57|3.65|2.01|1.92|1.22|
||VDD|1.94|2.99|1.55|1.65|1.05|
|LLaVA-1.5|OPERA|2.73|3.74|2.05|2.04|1.22|
||Woodpecker|2.74|3.87|2.05|2.03|1.29|
||LRV|2.76|3.82|2.10|2.03|1.28|
||LURE|2.72|3.76|2.03|2.05|1.27|
||**VDGD (ours)**|**2.97**|**3.91**|**2.64**|**2.27**|**1.54**|
||Vanilla-greedy|3.08|3.81|2.63|2.67|1.43|
||Vanilla-sampling|3.10|3.85|2.64|2.63|1.40|
||VDD|2.01|2.96|1.80|2.10|1.20|
|LLaVA-1.6|OPERA|3.10|4.01|2.62|2.60|1.42|
||Woodpecker|3.16|4.00|2.67|2.58|1.44|
||LRV|3.15|3.98|2.65|2.62|1.39|
||LURE|3.11|4.02|2.64|2.64|1.48|
||**VDGD (ours)**|**3.51**|**4.04**|**3.16**|**2.87**|**1.58**|
||Vanilla-greedy|2.73|3.64|2.20|1.96|1.28|
||Vanilla-sampling|2.72|3.60|2.18|1.91|1.24|
||VDD|2.16|2.95|1.80|1.85|1.26|
|mPLUG-Owl2|OPERA|2.92|3.96|2.26|2.08|1.34|
||Woodpecker|2.89|3.90|2.23|2.02|1.24|
||LRV|2.90|3.94|2.19|2.05|1.28|
||LURE|2.92|3.89|2.24|2.06|1.32|
||**VDGD (ours)**|**3.14**|**3.98**|**2.72**|**2.22**|**1.50**|
||Vanilla-greedy|3.11|3.98|2.66|2.34|1.47|
||Vanilla-sampling|3.13|4.04|2.70|2.41|1.56|
||VDD|2.56|3.26|2.32|2.19|1.41|
|InternLM-X|OPERA|3.14|4.20|2.65|2.32|1.46|
||Woodpecker|3.13|4.12|2.60|2.28|1.41|
||LRV|3.06|4.19|2.59|2.31|1.45|
||LURE|3.12|4.10|2.64|2.40|1.49|
||**VDGD (ours)**|**3.57**|**4.37**|**3.45**|**2.65**|**1.68**|
||Vanilla-greedy|3.25|4.19|2.82|2.43|1.55|
||Vanilla-sampling|3.26|4.21|2.83|2.40|1.55|
||VDD|2.80|3.62|2.63|2.10|1.58|
|CogVLM|OPERA|3.40|4.27|2.85|2.46|1.57|
||Woodpecker|**3.44**|**4.30**|2.91|2.45|1.59|
||LRV|3.42|4.26|2.88|2.42|1.54|
||LURE|3.39|4.20|2.78|2.41|1.52|
||**VDGD (ours)**|3.33|4.15|**3.01**|**2.51**|**1.60**|
|GPT-4-Turbo|-|**4.02**|**4.44**|**3.63**|**2.94**|**1.59**|



Table 9: Performance comparison of various LVLMs on VaLLu benchmark. This is an extension of results shown in Table 9 

34 

Pre-print. Under Review. 

|**Model**|**Methodology**|**Helpfulness**|**Clarity**|**Factuality**|**Depth**|**Engagement**|
|---|---|---|---|---|---|---|
||Vanilla-greedy|1.81|2.79|1.36|1.51|1.00|
||Vanilla-sampling|1.82|2.82|1.38|1.56|1.06|
||VDD|1.70|2.67|1.22|1.40|0.95|
|LLaVA-v1|OPERA|1.88|2.87|1.45|1.61|1.07|
||Woodpecker|1.89|2.88|1.48|1.60|1.06|
||LRV|1.83|2.85|1.40|1.57|1.08|
||LURE|1.91|2.90|1.52|1.63|1.09|
||**VDGD (ours)**|**2.09**|**2.99**|**1.64**|**1.74**|**1.14**|
||Vanilla-greedy|1.86|2.90|1.47|1.70|1.09|
||Vanilla-sampling|1.83|2.88|1.42|1.67|1.08|
||VDD|1.70|2.52|1.34|1.58|0.97|
|LLaVA-1.5|OPERA|2.18|3.08|1.62|1.75|1.22|
||Woodpecker|2.15|2.93|1.57|1.63|1.18|
||LRV|2.16|2.88|1.60|1.67|1.16|
||LURE|2.03|2.87|1.46|1.71|1.15|
||**VDGD (ours)**|**2.30**|**3.17**|**1.85**|**1.86**|**1.20**|
||Vanilla-greedy|1.85|2.67|1.54|1.80|0.97|
||Vanilla-sampling|1.80|2.63|1.50|1.78|0.96|
||VDD|1.68|2.57|1.36|1.68|0.94|
|LLaVA-1.6|OPERA|1.86|2.70|1.57|1.84|0.99|
||Woodpecker|1.90|2.74|1.63|1.89|1.04|
||LRV|1.88|2.72|1.61|1.85|1.02|
||LURE|1.92|2.73|1.65|1.86|1.03|
||**VDGD (ours)**|**2.11**|**2.89**|**1.81**|**1.92**|**1.05**|
||Vanilla-greedy|2.24|3.12|1.58|1.65|1.16|
||Vanilla-sampling|2.25|3.12|1.59|1.66|1.14|
||VDD|2.10|2.98|1.42|1.56|1.04|
|mPLUG-Owl2|OPERA|2.35|3.23|1.67|1.74|1.18|
||Woodpecker|2.28|3.17|1.61|1.69|1.15|
||LRV|2.34|3.24|1.66|1.72|1.14|
||LURE|2.32|3.21|1.63|1.67|1.19|
||**VDGD (ours)**|**2.48**|**3.36**|**1.82**|**1.78**|**1.23**|
||Vanilla-greedy|2.95|3.71|2.14|2.39|1.63|
||Vanilla-sampling|2.97|3.76|2.17|2.45|1.64|
||VDD|2.76|3.54|2.03|2.26|1.57|
|InternLM-X|OPERA|3.09|3.86|2.20|2.43|1.66|
||Woodpecker|3.07|3.83|2.19|2.46|1.68|
||LRV|3.11|3.88|2.24|2.49|1.70|
||LURE|3.10|3.89|2.22|2.48|1.71|
||**VDGD (ours)**|**3.21**|**4.11**|**2.57**|**2.78**|**1.79**|
||Vanilla-greedy|2.42|3.34|1.69|1.83|1.22|
||Vanilla-sampling|2.44|3.37|1.68|1.84|1.23|
||VDD|2.22|3.15|1.53|1.68|1.17|
|CogVLM|OPERA|2.56|3.47|1.76|1.88|1.27|
||Woodpecker|2.57|3.47|1.78|1.90|1.26|
||LRV|2.58|3.49|1.80|1.92|1.25|
||LURE|2.54|3.42|1.75|1.85|1.22|
||**VDGD (ours)**|**2.65**|**3.56**|**1.92**|**1.97**|**1.29**|
|GPT-4-Turbo|-|**4.15**|**4.53**|**3.66**|**2.99**|**1.95**|



Table 10: Performance comparison of various LVLMs on MMMU benchmark (only questions tagged with open-ended generation). This is an extension of results shown in Figure 7. 

35 

Pre-print. Under Review. 

|**Model**|**Methodology**|**Helpfulness**|**Clarity**|**Factuality**|**Depth**|**Engagement**|
|---|---|---|---|---|---|---|
||Vanilla-greedy|3.74|4.54|2.23|2.28|1.42|
||Vanilla-sampling|3.72|4.56|2.24|2.27|1.43|
||VDD|3.54|4.32|2.05|2.07|1.36|
|LLaVA-v1|OPERA|3.80|4.63|2.29|2.35|1.50|
||Woodpecker|3.78|4.60|2.26|2.30|1.47|
||LRV|3.76|4.64|2.30|2.38|1.49|
||LURE|3.74|4.59|2.25|2.28|1.46|
||**VDGD (ours)**|**3.92**|**4.78**|**2.52**|**2.42**|**1.54**|
||Vanilla-greedy|3.65|4.38|2.36|2.01|1.32|
||Vanilla-sampling|3.68|4.39|2.38|2.04|1.36|
||VDD|3.38|4.09|2.35|1.87|1.10|
|LLaVA-1.5|OPERA|3.74|4.46|2.26|1.93|1.32|
||Woodpecker|3.46|4.53|2.34|1.97|1.38|
||LRV|3.53|4.42|2.39|1.96|1.27|
||LURE|3.33|4.41|2.37|1.91|1.29|
||**VDGD (ours)**|**3.84**|**4.61**|**2.56**|**2.15**|**1.45**|
||Vanilla-greedy|3.72|4.38|2.50|2.34|1.50|
||Vanilla-sampling|3.69|4.32|2.45|2.28|1.48|
||VDD|3.54|4.11|2.34|2.18|1.35|
|LLaVA-1.6|OPERA|3.80|4.47|2.59|2.46|1.57|
||Woodpecker|3.78|4.42|2.56|2.43|1.53|
||LRV|3.75|4.40|2.54|2.42|1.52|
||LURE|3.79|4.41|2.52|2.41|1.53|
||**VDGD (ours)**|**3.90**|**4.53**|**2.79**|**2.51**|**1.58**|
||Vanilla-greedy|3.79|4.43|2.48|2.08|1.33|
||Vanilla-sampling|3.80|4.45|2.51|2.12|1.34|
||VDD|3.62|4.27|2.35|1.98|1.24|
|mPLUG-Owl2|OPERA|3.85|4.52|2.57|2.20|1.40|
||Woodpecker|3.84|4.56|2.60|2.24|1.42|
||LRV|3.82|4.54|2.59|2.22|1.41|
||LURE|3.86|4.58|2.61|2.25|1.45|
||**VDGD (ours)**|**3.99**|**4.65**|**2.76**|**2.36**|**1.52**|
||Vanilla-greedy|4.02|4.69|2.84|2.35|1.55|
||Vanilla-sampling|4.01|4.68|2.83|2.37|1.54|
||VDD|3.82|4.37|2.66|2.10|1.44|
|InternLM-X|OPERA|4.13|4.76|2.92|2.42|1.59|
||Woodpecker|4.07|4.72|2.88|2.40|1.58|
||LRV|4.10|4.73|2.89|2.40|1.56|
||LURE|4.14|4.75|2.91|2.43|1.60|
||**VDGD (ours)**|**4.28**|**4.89**|**3.12**|**2.54**|**1.66**|
||Vanilla-greedy|3.72|4.39|2.76|2.63|1.44|
||Vanilla-sampling|3.73|4.38|2.74|2.61|1.42|
||VDD|3.58|4.27|2.64|2.50|1.37|
|CogVLM|OPERA|3.76|4.45|2.82|2.69|1.47|
||Woodpecker|3.75|4.44|2.79|2.64|1.45|
||LRV|3.80|4.51|2.86|2.70|1.49|
||LURE|3.82|4.53|2.88|2.73|1.52|
||**VDGD (ours)**|**3.94**|**4.64**|**2.99**|**2.80**|**1.60**|
|GPT-4-Turbo|-|**4.05**|**4.75**|**3.17**|**2.24**|**1.61**|



Table 11: Performance comparison of various LVLMs on MathVista benchmark (only questions tagged with open-ended generation). This is an extension of results shown in Figure 7. 

36 

Pre-print. Under Review. 

|**Model**|**Methodology**|**Helpfulness**|**Clarity**|**Factuality**|**Depth**|**Engagement**|
|---|---|---|---|---|---|---|
||Vanilla-greedy|3.12|4.41|2.44|2.19|1.23|
||Vanilla-sampling|3.14|4.42|2.45|2.20|1.26|
||VDD|2.98|4.10|2.20|2.07|1.16|
|LLaVA-v1|OPERA|3.20|4.48|2.50|2.22|1.28|
||Woodpecker|3.21|4.49|2.46|2.21|1.25|
||LRV|3.25|4.51|2.51|2.23|1.29|
||LURE|3.20|4.47|2.49|2.20|1.28|
||**VDGD (ours)**|**3.42**|**4.59**|**2.78**|**2.30**|**1.30**|
||Vanilla-greedy|3.25|4.41|2.66|2.61|1.19|
||Vanilla-sampling|3.24|4.40|2.65|2.58|1.10|
||VDD|2.82|4.10|2.44|2.39|1.20|
|LLaVA-1.5|OPERA|3.22|4.40|2.49|2.58|1.18|
||Woodpecker|3.26|4.42|2.46|2.64|1.22|
||LRV|3.25|4.38|2.52|2.59|1.24|
||LURE|3.24|4.35|2.51|2.60|1.21|
||**VDGD (ours)**|**3.47**|**4.56**|**2.84**|**2.51**|**1.80**|
||Vanilla-greedy|3.43|4.41|3.13|3.31|1.35|
||Vanilla-sampling|3.40|4.38|3.10|3.27|1.30|
||VDD|3.26|4.19|2.96|3.13|1.20|
|LLaVA-1.6|OPERA|3.49|4.48|3.20|3.38|1.40|
||Woodpecker|3.45|4.45|3.21|3.35|1.42|
||LRV|3.42|4.40|3.18|3.34|1.39|
||LURE|3.38|4.36|3.18|3.28|1.36|
||**VDGD (ours)**|**3.64**|**4.71**|**3.49**|**3.42**|**1.45**|
||Vanilla-greedy|3.23|4.42|2.86|2.48|1.25|
||Vanilla-sampling|3.24|4.47|2.84|2.50|1.24|
||VDD|3.07|4.21|2.67|2.40|1.17|
|mPLUG-Owl2|OPERA|3.29|4.52|2.92|2.58|1.27|
||Woodpecker|3.27|4.50|2.90|2.52|1.25|
||LRV|3.23|4.49|2.88|2.50|1.24|
||LURE|3.26|4.51|2.91|2.51|1.28|
||**VDGD (ours)**|**3.34**|**4.64**|**3.15**|**2.70**|**1.32**|
||Vanilla-greedy|3.28|4.52|3.35|2.59|1.37|
||Vanilla-sampling|3.27|4.53|3.33|2.57|1.35|
||VDD|3.08|4.24|3.10|2.44|1.30|
|InternLM-X|OPERA|3.37|4.65|3.44|2.65|1.43|
||Woodpecker|3.38|4.63|3.45|2.66|1.44|
||LRV|3.40|4.69|3.47|2.70|1.46|
||LURE|3.36|4.64|3.42|2.68|1.42|
||**VDGD (ours)**|**3.54**|**4.78**|**3.68**|**2.79**|**1.49**|
||Vanilla-greedy|3.58|4.51|3.35|2.76|1.38|
||Vanilla-sampling|3.59|4.50|3.37|2.78|1.41|
||VDD|3.36|4.32|3.21|2.58|1.30|
|CogVLM|OPERA|3.68|4.60|3.42|2.84|1.45|
||Woodpecker|3.70|4.58|3.45|2.85|1.42|
||LRV|3.65|4.57|3.40|2.82|1.40|
||LURE|3.72|4.62|3.46|2.83|1.41|
||**VDGD (ours)**|**3.89**|**4.82**|**3.59**|**2.89**|**1.49**|
|GPT-4-Turbo|-|**4.15**|**4.79**|**3.89**|**3.25**|**1.67**|



Table 12: Performance comparison of various LVLMs on Oven benchmark. This is an extension of results shown in Figure 7. 

37 

