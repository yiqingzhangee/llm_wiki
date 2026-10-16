# VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform

Xingyu $\mathrm { L u } ^ { 1 , 2 \dag }$ , Tianke Zhang1†, Chang Meng1, Xiaobei Wang1, Jinpeng Wang2, Yi-Fan Zhang3, Shisong Tang1, Changyi Liu1, Haojie Ding1, Kaiyu Jiang1, Kaiyu Tang1, Bin Wen1, Hai-Tao Zheng2, Fan Yang1‡♠, Tingting Gao1, Di Zhang1, Kun Gai4 

1Kuaishou, 2Tsinghua, 3CASIA, 4Unaffiliated 

† Equal Contribution ‡ Corresponding Author ♠ Project Leader https://kuaimod.github.io 

# Abstract

Exponentially growing short video platforms (SVPs) face significant challenges in moderating content detrimental to users’ mental health, particularly for minors. The dissemination of such content on SVPs can lead to catastrophic societal consequences. Although substantial efforts have been dedicated to moderating such content, existing methods suffer from critical limitations: (1) Manual review is prone to human bias and incurs high operational costs. (2) Automated methods, though efficient, lack nuanced content understanding, resulting in lower accuracy. (3) Industrial moderation regulations struggle to adapt to rapidly evolving trends due to long update cycles. In this paper, we annotate the first SVP content moderation benchmark with authentic user/reviewer feedback to fill the absence of benchmark in this field. Then we evaluate various methods on the benchmark to verify the existence of the aforementioned limitations. We further propose our common-law content moderation framework named KuaiMod to address these challenges. KuaiMod consists of three components: training data construction, offline adaptation, and online deployment & refinement. Leveraging large vision language model (VLM) and Chain-of-Thought (CoT) reasoning, KuaiMod adequately models video toxicity based on sparse user feedback and fosters dynamic moderation policy with rapid update speed and high accuracy. Offline experiments and large-scale online A/B test demonstrates the superiority of KuaiMod: KuaiMod achieves the best moderation performance on our benchmark. The deployment of KuaiMod reduces the user reporting rate by $20 \%$ and its application in video recommendation increases both Daily Active User (DAU) and APP Usage Time (AUT) on several Kuaishou scenarios. We have open-sourced our benchmark at https://github.com/KuaiMod/KuaiMod.github.io. 

# 1 INTRODUCTION

Short video platforms (SVPs) such as Kuaishou and TikTok have experienced exponential growth in the Internet era, delivering millions of new videos daily. Videos of high quality delivery knowledge and joy to users and ensure the social value of SVPs. Meanwhile, widespread harmful videos pose significant threats to the ecosystem of the platform [49, 20], highlighting the need to develop a robust and effective ecosystem governance system [30]. 

To prevent inappropriate content from harming the platform ecosystem, the critical content moderation stage [5] has emerged to preemptively filter such content. The content moderation workflow on SVPs involves the systematic evaluation of videos to ensure their compliance with legal regulations, platform policies, and social ethics [42, 55]. The traditional SVP content moderation paradigm follows the civil law system [50]: Platforms establish a rule-based system grounded in laws and social consensus, while annotators act as judges to first understand the rule system then traverse to 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e240f9a3-caa9-48ff-8551-48667354bd05/d8b9165f4ddf3bb411f52df6f81681d5e74feb7c733acee8123d973548fcbb5d.jpg)



(a) A visualization about three moderation paradigms.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e240f9a3-caa9-48ff-8551-48667354bd05/6a65c2b87e91d272e45584b47232397494ce1502137ea1209d8a98f0e3eff147.jpg)



(b) An overview of KuaiMod benchmark.


determine videos’ harmfulness. This paradigm suffers from mental toll to annotators, labor costs and susceptibility to human bias [24]. 

Recently, artificial intelligence technologies are employed to reduce manual workload in content moderation with automated workflows. Early methods either encode rules, such as keyword matching [27], into programs or train classifiers, including decision tree [9], Naive Bayesian [7], and random forest [3], on labeled data to filter harmful content. These methods suffer from unsatisfactory accuracy [47] and are only effective in specific scenarios. More recent advances leverage large language models (LLMs) with exceptional content understanding capabilities to assist [35] or replace human reviewers [35]. For instance, Google’s Jigsaw API [30] applies a pre-trained LLM to assess the harmfulness of content. Other approaches prompt LLMs to moderate and neutralize harmful content [19]. Additionally, some studies apply the divergences between LLM outputs and human review to refine moderation policy [37]. 

To validate the performance of these moderation methods under the SVP scenario, we construct a benchmark based on user feedback and manual review from the Kuaishou platform, which contains 422 videos classified as harmful. Our evaluation result reveals that: Despite the advancements of existing methods, several unique challenges for SVP content moderation are remain unresolved: 

• Complex Content Understanding: For effective moderation on SVPs, it is essential to understand video content and user demands. However, early solutions and small-scale classifiers are unable to grasp the semantic of videos, while LLM-based methods struggle to process complex multi-modal content, let alone evaluating a video from user perspective. 

• Rapidly Evolving Trends: Driven by unpredictable societal events, trends on SVPs evolve so rapidly that static rule-based and model-based systems are prone to be incomplete. Unfortunately, prolonged update cycles further hinder these systems in detecting and preventing the dissemination of newly emerging harmful content. 

• Deployment Requirements: Industrial SVPs typically involve millions of active daily users and newly uploaded videos, placing stringent demands on the moderation accuracy and deployment costs. Previous research lacks industrial deployment experience and has not demonstrated moderation effectiveness on real-world SVPs. 

This study draws inspiration from the common law system [51] to address these challenges. Common law system relies on existing cases rather than static legal statutes to judge whether the defendant is violative. Compared to civil law, the common law system offers greater flexibility: Case-based precedents represent a consensus and continuously evolve with social progress. Emulating the common law system, we develop a case-based framework named KuaiMod for SVP content moderation. 

KuaiMod consists of three components: (1) Data Construction: We extract positive and violative examples from Kuaishou platform to construct chain-of-thought reasoning data. All instances contain detailed rationales and a verdict as reference for video moderation. (2) Offline Adaptation: Instead of static regulations, we transform Kuaishou self-developed YuanQi VLM [10] into dynamic case-based moderator through curriculum training. (3) Online Deployment & Refinement: Our deployed 

VLM judge provides fine-grained judgment for each video and refines its policy according to online feedback. These judgments are further used to enhance online personalized video recommendation. 

As visualized in Figure 1a, KuaiMod specifically overcomes the aforementioned challenges with targeted strengths: (1) Comprehensive understanding: Training paradigm with video CoT data reinforces the powerful VLM to better interpret heterogeneous video metadata including titles, frame images and user comments. (2) Rapid updatability: Our case-based moderation policy effortlessly achieves a daily update frequency: Since all processes are reusable, policy updates require only the addition of new violations and model training. (3) High deployability: The VLM judge, with an appropriate parameter scale, balances high accuracy and low inference cost, enabling moderation over millions of videos per day. We summarize our contributions as follows: 

• First SVP Moderation Benchmark: Recognizing the importance of harmful content moderation for SVPs, we constructed a content moderation benchmark with authentic user feedback from Kuaishou platform and strict data annotation. After removing privacy and illegal content, we open-source it to fill a critical gap in this field. To our knowledge, KuaiMod is the first SVP content moderation benchmark. 

• Common-Law Moderation Framework: An evaluation on our SVP governance benchmark reveals several limitations of existing moderation methods. To address these drawbacks, we propose our video content moderation framework named KuaiMod. KuaiMod can dynamically update governance policy by tracking immediate user feedback, and achieves accuracy comparable to human review, outperforming all baselines. 

• Industrial Deployment & Validation: We demonstrate the significant industrial value of KuaiMod: In several Kuaishou scenarios with tens of millions of users, A/B tests show that KuaiMod achieves a $20 \%$ reduction in video reporting rates. When applied for personalized recommendation, KuaiMod achieves positive gains in DAU (Daily Active Users). KuaiMod has been fully deployed, serving hundreds of millions of users on Kuaishou. 

# 2 RELATED WORKS

# 2.1 Content Moderation for Online Platform

Toxic content [13, 4, 57] like hate speech has been a long-standing problem for the safety of online platforms [56, 29, 46] with large user scale. To prevent the spread of these content [49], both manual [38, 52] and automated approaches [39, 36] have been developed. 

Traditional methods rely on crowd-sourcing [38, 12] to assess the harmfulness of media content. Manual methods typically involve creating a taxonomy and rule-based policy [38] for violations, with reviewers providing decisions accordingly. Although effective, these approaches reveals issues including high emotional and economic burdens [52], as well as reviewer bias [34]. Therefore, automated methods are sought to address these shortcomings [3]. 

To compensate for the limitations of manual moderation, automated content moderation has become a research focus in both academia and industry [27] in recent years, aiming to address the growing issue of harmful content on online platforms. Major companies have made significant progress in this field. For example, Jigsaw and Google [30] developed Perspective API, a machine learning-based tool designed to detect and assess the toxicity of online comments. Twitter employs machine learning models to identify and remove hate speech and other harmful content [22, 16]. These models analyze tweet text, user behavior, and contextual information, providing an automated approach to content moderation. Alibaba [31, 25] has developed an intelligent content moderation system that leverages machine learning and deep learning techniques to automatically detect and handle inappropriate content on its e-commerce platforms. However, most of these methods rely on rule-based approaches and small-scale models, requiring coexistence with manual moderation. They suffer from significant latency and weak multi-modal understanding [45], making them less adaptable to the fast-evolving, information-rich landscape of the modern internet environment [6, 44]. 

# 2.2 Language Models for Content Moderation

With advancements in natural language processing (NLP) technology [2, 18], large language models (LLMs) have become increasingly prevalent in the automated content moderation of social media 

platforms. In recent years, Transformer-based LLMs, such as BERT [26] and GPT [40], have achieved remarkable success in natural language processing (NLP) tasks through self-attention mechanisms and large-scale pre-training. These models can effectively capture complex semantic relationships in text [48]. In the context of content moderation, LLMs have been utilized to detect inappropriate speech, hate speech, and misinformation [15, 14]. 

LLMs can be adapted to specific content moderation tasks through fine-tuning. For example, by fine-tuning on datasets labeled with harmful or inappropriate content, these models can enhance their ability to detect specific types of undesirable content [21, 54]. Moreover, some LLMs like GPT-4o [23], possess zero-shot and few-shot learning capabilities, enabling them to perform content moderation tasks without dedicated training data. This ability allows models to rapidly adapt to emerging content moderation needs without requiring large-scale annotated datasets [43, 8]. 

Beyond text-based models, recent research has explored the use of vision-language models (VLMs) with multi-modal understanding capabilities for moderating multi-modal content [41, 33]. However, these approaches have limitations in terms of generalization, particularly when dealing with diverse and complex malicious content, where models may struggle to accurately identify emerging misinformation [1]. Additionally, although existing LM-based moderation models possess strong analytical capabilities, they suffer from poor real-time performance, making it challenging to meet the demands of high-frequency content moderation, resulting in delays and processing bottlenecks [17, 28]. 

# 3 BENCHMARK

In this section, we first formally define the video moderation task, then introduce our taxonomy for violative content on Kuaishou platform and KuaiMod benchmark based on the taxonomy. 

# 3.1 Task Definition for Video Moderation

Given a video’s metadata, including video frames, description text, audio, and user comments, the moderation task can be formulated as a mapping process from these inputs to a specific video label $\imath$ We formalize the video moderation method as a mapping function: 

$$
\mathbf {f} _ {\mathbf {m}} (\mathbf {v}, \mathbf {t}, \mathbf {a}, \mathbf {u}) = \mathbf {l}, \quad \mathbf {l} \in \mathbf {L}, \tag {1}
$$

where v, t and $\textbf { \em a }$ represent visual, textual and audio information respectively, u represents user comments, and $\pmb { L }$ is the set of all labels. The output of different methods can be text or probabilities. 

The simplified version of the video moderation task only requires binary classification, where $\pmb { L }$ consists of two labels: positive (non-violative) and violative. However, KuaiMod moderation paradigm aims for fine-grained classification, categorizing videos into 1 positive label and 15 violative labels. This granular classification improves the interpretability of the moderation process and provides a structured reference for downstream applications. 

# 3.2 KuaiMod Taxonomy

KuaiMod taxonomy categorizes violative content into 4 general categories: Social Security & Law, Content Ethics, Prohibited Commercial Behavior and Intellectual Property. Each primary category is further divided into 2-5 subcategories, resulting in a total of 15 subcategories, as shown in Figure 1b. 

The taxonomy originates from Kuaishou’s manual moderation guidelines, which are structured similarly to the civil-law system, defining all types of violative content with exhaustive enumeration and detailed descriptions. In contrast, our taxonomy is more concise and generalized to enhance the generalizability of KuaiMod, making it more flexible and responsive for SVP content moderation. 

Moreover, the taxonomy itself is dynamic and is progressively refined in following data construction and training process (Section 4.3), making KuaiMod significantly easier to update compared to cumbersome regulations. This adaptability ensures the robustness of KuaiMod in handling evolving violative content. Note that the online version of KuaiMod taxonomy has been continuously expanded, now it includes over 100 labels and serves as the reference for Kuaishou video moderation. For simplicity in evaluation, the taxonomy presented here corresponds to its initial version. 

# 3.3 KuaiMod Benchmark

Based on this taxonomy, we construct our KuaiMod content moderating benchmark. The benchmark consists of 1,000 short videos from the Kuaishou platform, encompassing both healthy positive content and all 15 types of violative content. 

Adhering to Kuaishou’s User Obsession principle, we collect both positive and violative content based on real user feedback. The positive instances come from the High Quality Videos, which are videos from the manual reviewed daily high-quality video queue that have not been reported by users and have a dislike ratio of less than $0 . 1 \%$ . For violative content, there are three main sources: 

• High Report Videos: The top- $\mathbf { \nabla } \cdot n$ videos among the daily uploaded videos, sorted by the number of reports, with each video receiving more than 30 reports. 

• High Dislike Videos: The top- $^ n$ videos among the daily uploaded videos, sorted by the percentage of users clicking dislike, with each video’s dislike rate above $50 \%$ . 

• Negative Comment Videos: Videos with more than 2,000 plays that negative comments accounting for more than $80 \%$ of its total comments among the daily uploaded videos. 

With various samples collected from the above sources, we conduct manual annotation with crowdsourcing. 30 experienced annotators first determine whether each video is positive or violative. For violative videos, annotators further assign fine-grained labels. Finally, we select 1,000 samples with consistent annotations from all labeled data to construct the test set that covers all 15 violative categories. Each sample in KuaiMod benchmark contains multiple frames extracted from the video, along with textual content, including title, text recognized via OCR and ASR, and user comments. 

To the best of our knowledge, KuaiMod is the first open-source benchmark for industrial SVP content moderation. We believe it holds significant value in advancing SVP ecosystem governance. 

# 4 METHODOLOGY

In this section, we present how to develop the KuaiMod moderation model based on our selfdeveloped general VLM, YuanQi. We first introduce YuanQi in Section 4.1, then we explain the data construction, training, deployment and updating pipeline of KuaiMod. Section 4.2 presents the data construction process: We utilize largest YuanQi-20B model to construct moderation data in a state-transition format with CoT technology. Section 4.3 details the post-training process of KuaiMod, which adapts the general YuanQi-7B model into a video moderation model with Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). Section $4 . 4$ describes the online deployment of KuaiMod and its RL-based updating mechanism to iteratively refine moderation policy. 

# 4.1 Overview of YuanQi Model

# 4.1.1 Architecture & Modules

The YuanQi model is an efficient VLM following the Flamingo-style architecture, consisting of: (1) Visual Encoder: Built upon the EVA2-CLIP-E-Plus model, it extracts hierarchical vision features at multiple levels, enabling fine-grained perception of visual elements. (2) LLM Foundation: Based on Qwen-14B-Chat 1.0, providing strong content understanding and logical reasoning capabilities. (3) Gated Cross-Attention Layers: Inserted before each transformer layer in the LLM, allowing effective image-text fusion while maintaining high computational efficiency. (4) Mixture of Experts (MoE): Optimizes model effectiveness by dynamically activating specialized subnetworks. 

# 4.1.2 Training

The YuanQi model undergoes three-stage training: (1) Multi-Modal Pre-Training aligns visual and textual modalities with large-scale image-text pairs, ensuring high relevance through extensive data cleaning. (2) Multi-Task Continual Pre-Training incorporates tasks including visual question answering (VQA), OCR, and object detection, enhancing high-level reasoning capabilities. (3) Supervised Fine-Tuning activates instruction-following abilities with high-quality instruction data. 

# 4.1.3 Performance & Application

The hierarchical feature extraction mechanism, cross-attention architecture, and multi-stage training pipeline enable YuanQi to achieve state-of-the-art performance on diverse benchmarks including general VQA, fine-grained visual perception, bilingual capabilities and hallucination detection. Overall, YuanQi is highly suitable for tasks requiring complex visual and textual integration, particularly in short video scenarios. Additionally, the YuanQi model scales from 3B to 20B. In our approach, we use the largest 20B version for data generation due to its superior inference and instruction-following capabilities, while the smaller 7B version serves as the foundation for KuaiMod moderation model, ensuring efficacy-efficiency balance for deployment. 

# 4.2 Construction of Training Data

# 4.2.1 Data Collection and Annotation

The data source and labeling process for training data is the same as the benchmark construction in Section 3.3, but at a larger scale. We sampled 50,000 videos from four video queues on the Kuaishou platform between May 1st and November 1st—a six-month period preceding the project launch—for training data construction, ensuring data freshness. 

Regarding the annotation process, unlike the static taxonomy shown in Figure 1b, the taxonomy for training data is continuously expanded based on annotation results, as previously mentioned: Experienced moderators are first required to determine whether a video violates platform policies. For violating videos, they assign them to a primary violation category and then assess whether the violation type matches an existing secondary label. For cases that cannot be classified into the current taxonomy by human annotators, we instruct the YuanQi-20B model to generate supplementary secondary labels. We then collect all newly generated secondary labels under each primary category, aggregate them, and manually review before incorporating them into the original taxonomy. 

In this way, we have expanded the taxonomy to include over 100 violation categories and continue to extend it after online deployment. At the end of the annotation process, each video is assigned a violation type tag, which serves as the basis for the subsequent data construction process. The initial taxonomy covers 24,562 cases, we take this split as training data for offline evaluation. 

# 4.2.2 Generate State-Transition Data with Tag2CoT and CoT2Tag

Instead of directly using the tag as the training target, we aim to model the causality between video content and the manual moderation result. Specifically, given a video’s meta data and its violation type tag as input context, we prompt YuanQi-20B model with instructions to analyze the video’s content then generate Chain-of-Thought rationales to explain why the video is classified as its violation tag. This improves the interpretability of the moderation result. We refer to this process as Tag2CoT. 

Tag2CoT process produces causal analysis between metadata and violation tags. However, as described in Section 3.1, The video moderation is a process of mapping video metadata to violation labels, the rationales generated in Tag2CoT process are insufficient for training a VLM to complete the moderation task since its analysis proceeds from the outcome to the cause. Therefore, we introduce a second process, CoT2Tag, to organize these elements into a video moderation workflow. 

We first explain our workflow for video moderation: We introduce the concept of state transition from reinforcement learning (RL) to analyze videos in a structured, step-by-step manner. The initial state is the input context, which contains the task definition and video metadata. Next, the VLM sequentially transits to the following states and ultimately makes its moderation decision. 

(1) Content Extraction: In this state, VLM extracts content from video frames and generates an overall description of the video, converting all metadata into textual format. 

(2) Video Content Analysis: In this state, VLM analyzes all the meta data including the video description, title, OCR, and ASR transcripts sequentially to identify any explicit violations. 

(3) Interim Check: In this state, VLM determines whether a violation category can be assigned based on the previous analysis. If the decision remains uncertain, proceed with further examination. 

(4) User Feedback Analysis: In this state, for cases that are difficult to determine based on metadata alone, VLM incorporates user comments to assess the video’s impact on viewers and finally decides the video’s violation category. 

(5) Final Summary: In this state, VLM summarizes the findings from all previous states, assigns the final violation tag, and provides several key reasons as support for its judgment. 

In the CoT2Tag process, our YuanQi-20B model takes the video metadata, the moderation tag and rationales as input and convert them into the state-transition format data. Each state performs causal analysis about a specific aspect of the video, and the last state provides the moderation result. A more intuitive representation of the workflow can be found in the Figure A-2a. 

For the state transition data generated by YuanQi-20B, we first prompt YuanQi-20B to self-verify the correctness of its output then conduct manual review to identify logically inconsistent or misattributed samples then correct them. These bad cases serve as negative samples for model training. It is important to note that while human annotation plays a crucial role, the majority work in our data construction process is handled by the YuanQi model. Additionally, the scale of human annotation is negligible compared to the platform’s daily moderation demands. Besides, these annotated cases can be reused, further reducing costs. 

# 4.3 Stage I: Offline Adaptation Training

As shown in Figure 2, in offline adaptation stage, we adapt the general YuanQi-7B model to KuaiMod video moderator via two training steps: Large-scale SFT and mistake-oriented DPO. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e240f9a3-caa9-48ff-8551-48667354bd05/34d800991719cd6245a953f4959899addd42422b0281d5f4adf1eb187ac8e7f4.jpg)



Figure 2: Offline adaptation stage of KuaiMod: We post-train the YuanQi-7B with state-transition format data. After SFT and DPO, KuaiMod is transformed into a video moderator for online services.


# 4.3.1 Large-scale SFT

We take all the right cases as corpus to SFT the YuanQi-7B model. The SFT loss is formulated as: 

$$
\mathcal {L} _ {\mathrm {S F T}} = - \frac {1}{N} \sum_ {t = 1} ^ {N} \log P _ {\theta} \left(y _ {t} \mid x, y _ {<   t}\right), \tag {2}
$$

$N$ denotes the number of output tokens, $y _ { t }$ represents the $t$ -th token in the target response, $y _ { < t } =$ $\left( y _ { 1 } , y _ { 2 } , \ldots , y _ { t - 1 } \right)$ denotes the first $t - 1$ output tokens, $x$ denotes the input tokens, including video frames, title, OCR/ASR text and user comments as mentioned in Section 3.1. $P _ { \theta } ( y _ { t } | y _ { < t } )$ represents the conditional probability distribution over $y _ { t }$ given $y _ { < t }$ , where $\theta$ denotes the model parameters. $\log { P _ { \theta } ( y _ { t } | y _ { < t } ) }$ is the logarithm of the probability from the softmax function. 

At this stage, our VLM learns from a large-scale cases about how to analyze video content through the state transition process and complete the video moderation task. Adopting a divide-and-conquer strategy, decoupling the analysis of different aspects of the video into independent states before aggregating the results, not only prevents overlook of important details but also enables a more comprehensive assessment of the video’s harmfulness. 

# 4.3.2 Mistake-oriented DPO

In this step, we leverage DPO to correct the model’s mistakes and improve moderation accuracy. 

After several epochs of SFT, we take the trained model to inference on the SFT dataset. On each data instance, we sample multiple times to collect correct/incorrect responses that align/misalign with the reference answer, then use them to construct paired samples for DPO training, where each pair consists of one correct output and one incorrect output. Additionally, during the data generation process, we corrected some erroneous instances. These corrected and incorrect samples are also 

paired for DPO training. The DPO loss is formulated as: 

$$
\mathcal {L} _ {\mathrm {D P O}} (\pi_ {\theta}; \pi_ {\mathrm {r e f}}) = - \mathbb {E} _ {(x, y _ {r}, y _ {w}) \sim \mathcal {D}} \left[ \log \sigma \left(\beta \log \frac {\pi_ {\theta} (y _ {r} | x)}{\pi_ {\mathrm {r e f}} (y _ {r} | x)} - \beta \log \frac {\pi_ {\theta} (y _ {w} | x)}{\pi_ {\mathrm {r e f}} (y _ {w} | x)}\right) \right], \tag {3}
$$

in which $y _ { r }$ and $y _ { w }$ represents the correct and incorrect response respectively, $\pi _ { \boldsymbol { \theta } } ( y | \boldsymbol { x } )$ denotes the probability assigned by the current model with parameters $\theta$ to response $y$ given input $x$ , whereas $_ { r e f }$ represents the reference model, which is the SFT model. The scaling factor $\beta$ controls the strength of preference optimization. The term log $\frac { \pi _ { \theta } ( y _ { r } | x ) } { \pi _ { \mathrm { r e f } } ( y _ { r } | x ) }$ and log πθ(yw|x)πref (yw|x) $\frac { \pi _ { \theta } \left( y _ { w } | x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { w } | x \right) }$ capture the log-probability ratio of the correct and incorrect response between the current and reference models. Finally, the sigmoid function $\sigma ( \cdot )$ ensures that the loss guides the model to assign higher probabilities to correct responses while decreasing the probabilities of incorrect ones, effectively refining its moderation decisions. 

# 4.4 Stage II: Online Deployment & Refinement

Although KuaiMod model acquires preliminary video moderation capabilities after offline adaptation, as previously analyzed, the scope of violation contents in SVPs can evolve with users and social trends. A static moderation model cannot effectively adapt to the dynamic nature of SVPs. Therefore, in Stage II, we design an update mechanism as depicted in Figure 3, which follows Reinforcement Learning from User Feedback (RLUF) paradigm to continuously refines KuaiMod’s moderation policy based on SVP feedback. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e240f9a3-caa9-48ff-8551-48667354bd05/3face18017bef998a7b58192c4a3c2ee77819758455546651ffe4af6e6bc02a4.jpg)



Figure 3: Online deployment stage of KuaiMod: The initially trained KuaiMod model is deployed into Kuaishou as a moderation agent. KuaiMod interacts with the online environment and iteratively refines its policy with user feedback in the RL manner.


# 4.4.1 RLUF Definition for KuaiMod Online Refinement

We introduce key elements of the RLUF paradigm and explain their corresponding roles in KuaiMod’s online refinement process: 

- Environment $( \mathcal { E } )$ : The environment consists of videos and users on the SVP. Videos vary in content and potential violations, while users consume videos and provide feedback including reporting. 

- Agent $( \mathcal { A } )$ : Our KuaiMod model serves as an agent interacting with the environment, its policy $\pi _ { \theta }$ is parameterized by $\theta$ , which determines moderation judgments based on videos’ content. 

- Reward and Objective: The reward signal is directly derived from user feedback, where a lower online user report rate indicates the better moderation policy. The objective of KuaiMod is to update its moderation policy through interactions with the environment, ensuring alignment with users’ understanding of violation content, to improve user satisfaction and platform ecosystem. 

# 4.4.2 RLUF Data construction

The key to aligning user preferences with the model’s moderation policy lies in eliminating discrepancies between their understandings of violation content. Therefore, we select video cases where the model’s moderation decisions deviate from user perception as training data for the refinement phase. 

Real-time Hard Case Collection. We still select the four data queues in Section 3.3 as data sources. In addition, we introduce a new queue: High-Engagement Videos, which refers to the most popular daily videos. These videos have a large audience, meaning that any violations within them pose a greater risk to the SVP ecosystem. Additionally, the rapidly changing nature of popular content directly reflects platform trends and user preferences, making it a critical focus for moderation. 

Two-stage Data Curation. (1) Coarse filtering. We first assume that videos from the high-quality and high-engagement queues are positive, while those sampled from the high-report, high-dislike, and negative-comment queues are violative. Then KuaiMod model with the old policy generates moderation decisions for all data so that we can identify cases where the model’s judgment contradicts human feedback as candidates for refinement. (2) Fine-grained annotation. Given the controversial candidates, we assign the final violation labels through human review and then generate data using the previously established Tag2CoT and CoT2Tag processes. For cases where KuaiMod makes incorrect judgments, the original response and the newly generated response can be directly used as the incorrect answer and correct answer, respectively. For cases where KuaiMod’s decision is correct but some users still perceive the content as inappropriate, we instruct the YuanQi-20B model to enumerate, within the CoT reasoning process, the aspects of the video that might cause user discomfort and explain why they do not in fact violate platform regulations. This serves as positive examples to help the model distinguish between individual user sensitivities and societal consensus. 

# 4.4.3 Online RLUF training

Similarly to the mistake-oriented DPO in offline training, we still choose DPO algorithm in eq. (3) for online refinement training. Compared to offline training, in addition to differences in data sources, the negative samples in the online refinement phase are entirely sampled by the deployed KuaiMod model. And the online refinement process undergoes multiple rounds of continuous optimization to keep KuaiMod up with the evolving trends of Kuaishou. With a data scale in the hundreds, KuaiMod achieves low-cost daily updates. 

# 5 OFFLINE EVALUATION

# 5.1 Experimental Setup

# 5.1.1 Dataset

Following our construction workflow in Section 4.2, we build a larger data set for training. This dataset contains 24,562 samples, covering all 15 types of violative videos. In Table 1, we provide an overview of the distribution of the training and test sets. For all methods that requires training, including our KuaiMod, we use the 24,562 samples as training data to ensure fair comparison. 


Table 1: Data Distribution in Train and Test split.


<table><tr><td>Category</td><td>Train</td><td>Test</td></tr><tr><td>Positive</td><td>19,343</td><td>578</td></tr><tr><td>Law &amp; Safety</td><td>1,862</td><td>170</td></tr><tr><td>Content Ethics</td><td>2,659</td><td>180</td></tr><tr><td>Commercial</td><td>511</td><td>55</td></tr><tr><td>Compliance</td><td>187</td><td>17</td></tr></table>

# 5.1.2 Metrics

Considering that certain baselines do not support multi-class classification, and the primary target for content moderation is to distinguish positive and negative content, we focus on comparing binary classification performance. The utilization of multi-class classification is discussed in Section 6.2.2 

For binary classification, we employ metrics including Recall, Precision, and Accuracy to evaluate the performance of various methods. We report the overall accuracy of various methods on the benchmark, along with their precision and recall for both positive and negative content, as well as the recall for four major violation categories. We unify the judgment results across all methods, including both binary and multi-class models, by treating any prediction that is not positive as negative. 

# 5.1.3 Baselines

For a comprehensive comparison, we have selected the following three groups of baselines: 

(1) Methods based on API. 

• Perspective API [30]: An industrial-grade text moderation API developed by Google. Given a text input, we utilize the returned TOXICITY and SEVERE_TOXICITY scores to obtain harmfulness classification results with predefined thresholds. 

• GPT-4o mini & GPT-4o API [23]: GPT-4o mini and GPT-4o are general models of OpenAI, capable of processing multi-modal inputs. We prompt them to judge each video’s category with different instructions for binary and multi-class classification. 

(2) Methods based on tiny models. 

• RoBERTa [32]: One of the most effective text encoders, widely used for various natural language processing tasks including Question Answering and Sentiment Analysis. 

• CN-CLIP [53]: A CLIP-style image-text encoder for Chinese scenarios. We concatenate its text and image representations as input to the MLP classifier. 

(3) Methods based on large models. 

• YOLO [19]: The first method that leverages soft prompt tuning to train LLMs for toxic content classification and text detoxification. 

• Intern-VL [11]: One of the most advanced open-source VLMs, we directly fine-tune its 7B model for binary/multi-class classification. 

# 5.1.4 Implementation Details

For the Perspective API, we select the threshold that maximizes overall accuracy by searching within the range of 0.1 to 0.9. For RoBERTa and CN-CLIP, we use a two-layer MLP as the classification head. Specifically, for CN-CLIP, we concatenate its multi-modal representations as the input to the classifier. For YOLO, we set prompt token num as 30. For Intern-VL, we construct the SFT data using the same instruction format as the prompts for GPT-4o, directly treating the label as the output target. For all baselines, we follow their default settings or mainstream research when configuring training hyper-parameters, including batch size, learning rate, and other relevant settings. And all experiments are conducted on NVIDIA A800-SXM4-80GB machines. 


Table 2: Performance of Various Moderation Methods on the KuaiMod Benchmark. We categorize the moderation methods into Binary Classification and Multi-class Classification. The binary classification only determines whether a video is violative or not, while the multi-class classification requires the model to directly classify the video into its respective category. Optimal and sub-optimal performance is denoted in bold and underlined fonts, respectively.


<table><tr><td rowspan="2">Method</td><td colspan="4">Main Violative Category Recall</td><td colspan="2">Negative Content</td><td colspan="2">Positive Content</td><td rowspan="2">Overall Accuracy</td></tr><tr><td>Law&amp;Safety</td><td>Content</td><td>Commercial</td><td>Intellectual</td><td>Precision</td><td>Recall</td><td>Precision</td><td>Recall</td></tr><tr><td colspan="10">Binary Classification</td></tr><tr><td>Perspective\( toxicity \)</td><td>0.6036</td><td>0.4598</td><td>0.3455</td><td>0.2941</td><td>0.6591</td><td>0.7059</td><td>0.5538</td><td>0.5000</td><td>0.6190</td></tr><tr><td>Perspective\( severe \)</td><td>0.1716</td><td>0.1092</td><td>0.0909</td><td>0.1765</td><td>0.5922</td><td>0.9170</td><td>0.5429</td><td>0.1351</td><td>0.5870</td></tr><tr><td>GPT-4o mini</td><td>0.7235</td><td>0.6833</td><td>0.5091</td><td>0.5294</td><td>0.7679</td><td>0.7958</td><td>0.7057</td><td>0.6706</td><td>0.7430</td></tr><tr><td>GPT-4o</td><td>0.8176</td><td>0.8101</td><td>0.8182</td><td>0.4706</td><td>0.8346</td><td>0.7336</td><td>0.6870</td><td>0.8009</td><td>0.7620</td></tr><tr><td>RoBERTa</td><td>0.8530</td><td>0.7667</td><td>0.7455</td><td>0.5882</td><td>0.6448</td><td>0.7915</td><td>0.6817</td><td>0.8174</td><td>0.7400</td></tr><tr><td>CN-CLIP</td><td>0.7396</td><td>0.6034</td><td>0.7636</td><td>0.5882</td><td>0.7836</td><td>0.6777</td><td>0.7858</td><td>0.8633</td><td>0.7850</td></tr><tr><td>YOLO</td><td>0.7929</td><td>0.5920</td><td>0.7818</td><td>0.4706</td><td>0.8189</td><td>0.6967</td><td>0.8003</td><td>0.8875</td><td>0.8070</td></tr><tr><td>Intern-VL-7B-SFT</td><td>0.8471</td><td>0.8056</td><td>0.8000</td><td>0.6471</td><td>0.8152</td><td>0.7765</td><td>0.8600</td><td>0.8287</td><td>0.8230</td></tr><tr><td colspan="10">Multi-Class Classification</td></tr><tr><td>GPT-4o mini</td><td>0.8706</td><td>0.7889</td><td>0.8727</td><td>0.4118</td><td>0.6161</td><td>0.8175</td><td>0.8250</td><td>0.6280</td><td>0.7130</td></tr><tr><td>GPT-4o</td><td>0.8647</td><td>0.8278</td><td>0.8364</td><td>0.5294</td><td>0.6381</td><td>0.8318</td><td>0.8422</td><td>0.6557</td><td>0.7300</td></tr><tr><td>Intern-VL-7B-SFT</td><td>0.6706</td><td>0.5944</td><td>0.7091</td><td>0.2353</td><td>0.7976</td><td>0.6256</td><td>0.7638</td><td>0.8841</td><td>0.7750</td></tr><tr><td>KuaiMod-7B</td><td>0.8765</td><td>0.8333</td><td>0.8545</td><td>0.7059</td><td>0.9701</td><td>0.8460</td><td>0.8972</td><td>0.9810</td><td>0.9240</td></tr></table>

# 5.2 Evaluation Result

As shown in Table 2, Perspective API performs worse than other approaches, despite being a specialized moderation API. This is consistent with previous study [28]. We attribute KuaiMod’s performance to its relatively old version and inability to leverage visual information. In contrast, GPT-4o and GPT-4o mini are more aligned with the release dates of the videos and possess advanced 

multi-modal capabilities, leading to superior performance. This highlights the necessity of update frequency for industrial moderation APIs. 

Methods trained on KuaiMod data, regardless of model size, consistently outperform API-based approaches. Even the RoBERTa model with only 500M parameters surpasses the large-scale GPT-4o, confirming our analysis about the importance of real-time updates for video moderation task. Among these methods, text-based models perform slightly worse than multi-modal models, indicating that while text provides useful information, certain critical information can only be effectively captured through complex visual content. Besides, Additionally, methods based on smaller models exhibit slightly lower performances compared to those based on larger models, as the limited capacity restricts their understanding ability. 

When transitioning from binary classification to multi-class classification, both the untrained GPT-4o and the trained Intern-VL experience performance degradation, as multi-class classification is inherently more challenging than binary classification. In contrast, our KuaiMod approach achieves significantly better performance with multi-class training compared to the aforementioned baselines: It achieves the highest or second-highest precision, recall, and overall accuracy for positive and negative content. Moreover, it demonstrates relatively balanced high recall across the four major violation categories, highlighting its strong discriminative capability and generalizability. We attribute this improvement to three key factors: the YuanQi foundation model, which is better suited for short video scenarios; the CoT state-transition data, which enhances KuaiMod’s content understanding; and the two-step offline training pipeline, which systematically refines moderation capabilities. 

# 5.3 Analysis Experiments

# 5.3.1 Ablation study

To validate the effectiveness of each component, we compare KuaiMod with its five variants: (1) Original YuanQi-7B model without post-training. (2) Intern-VL-7B as the foundation model. (3) Removing the Tag2CoT mechanism, directly generating moderation process with YuanQi. (4) Removing the state transition workflow. (5) Omitting the DPO training. 


Table 3: Ablation study on each component of KuaiMod.


<table><tr><td rowspan="2">Method</td><td colspan="2">Negative Content</td><td colspan="2">Positive Content</td><td>Overall</td></tr><tr><td>Precision</td><td>Recall</td><td>Precision</td><td>Recall</td><td>Accuracy</td></tr><tr><td>KuaiMod</td><td>0.9701</td><td>0.8460</td><td>0.8972</td><td>0.9810</td><td>0.9240</td></tr><tr><td>YuanQi-7B</td><td>0.4465</td><td>0.5047</td><td>0.6004</td><td>0.5433</td><td>0.5270</td></tr><tr><td>w Intern-VL</td><td>0.9167</td><td>0.8341</td><td>0.8864</td><td>0.9446</td><td>0.8980</td></tr><tr><td>w/o Tag2CoT</td><td>0.9530</td><td>0.7204</td><td>0.8267</td><td>0.9740</td><td>0.8670</td></tr><tr><td>w/o state transition</td><td>0.9337</td><td>0.7346</td><td>0.8323</td><td>0.9619</td><td>0.8660</td></tr><tr><td>w/o DPO</td><td>0.8834</td><td>0.8081</td><td>0.8681</td><td>0.9221</td><td>0.8740</td></tr></table>

As shown in Table 3, the absence of any single component results in a performance drop: The native YuanQi model demonstrates moderate performance on the video moderation task, this validates the necessity of training adaption. While Intern-VL as foundation lags behind YuanQi in short video understanding. Removing Tag2CoT mechanism and the state transition workflow reduces KuaiMod’s ability to leverage video metadata. And DPO training further enhances the KuaiMod’s moderation capability beyond the improvements achieved through SFT. 

# 6 ONLINE APPLICATIONS

We explore the application of KuaiMod in both platform ecosystem governance and enhancing platform revenue on three scenarios of Kuaishou: NEBULA, Featured and Main Site. NEBULA is a lightweight platform fully focused on short video recommendations, the Featured platform pushes relatively higher quality short videos, while the Main Site encompasses the largest user base and Kuaishou’s business, including live streaming and e-commerce. Each of these scenarios has over 10 million daily active users (DAU), with more than 16 million short video impressions per day. 

# 6.1 Deployment Details

To improve response speed, we apply FP16 quantization to KuaiMod-7B model for deploying. On a cluster of 48 RTX4090 GPUs, KuaiMod achieves a QPS (Queries Per Second) of 6.81. This service moderates all candidate videos uploaded to the Kuaishou platform daily, with an average volume of 600,000 videos per day. 

# 6.2 Performance of A/B test

# 6.2.1 Comprehensive Ecosystem Governance

In this aspect, KuaiMod provides its judgments for online videos, videos identified as violative are directly removed from promotion to refine platform ecosystem. To narrow the moderation scope, we select videos that have been reported more than five times as candidates to analyze their specific violations with KuaiMod. 

The A/B test result in Table 4 shows that KuaiMod significantly reduces the user reporting rate, indicating an improvement in user satisfaction. Furthermore, DAU and total App Usage Time (AUT) exhibited slight fluctuations or even an upward trend, highlighting the critical role of ecosystem governance in platform development. 


Table 4: KuaiMod’s online A/B test results for comprehensive ecosystem governance on Kuaishou NEBULA and Featured.


<table><tr><td>Column</td><td>Report Rate (%) ↓</td><td>DAU (%) ↑</td><td>Total AUT (%) ↑</td></tr><tr><td>NEBULA</td><td>-24.34</td><td>+0.016</td><td>-0.002</td></tr><tr><td>Featured</td><td>-18.98</td><td>-0.002</td><td>+0.012</td></tr></table>


Table 5: KuaiMod’s online A/B test result for personalized recommendation enhancement on Kuaishou Main Site.


<table><tr><td>Column</td><td>DAU (%) ↑</td><td>Avg AUT (%) ↑</td><td>Total AUT (%) ↑</td></tr><tr><td>Main Site</td><td>+0.028</td><td>+0.033</td><td>+0.063</td></tr></table>

# 6.2.2 Personalized Recommendation Enhancement

In this aspect, KuaiMod’s judgments are embedded into features to enhance personalized video recommendation, which leads to personalized moderation strategy. The personalized moderation strategy is crucial because different users may have varying perceptions of harmful content. For example, content suitable for adults might be inappropriate for minors. However, only certain types of violating videos are preserved for personalized recommendation; videos that explicitly violate laws or pose severe harm are strictly filtered. 

As shown in Table 5, both DAU and AUT increase due to the personalized moderation strategies, the total consumption time increases by $0 . 0 6 3 \%$ . This indicates the potential of the fine-grained harmful content governance for improving recommendation revenue. In the future, we will further develop new paradigms for using VLMs in short video recommendations based on KuaiMod. 

# 7 CONCLUSION

In this paper, we first analyze the challenges faced by traditional video moderation in industrial SVP, including the mental toll and high cost of manual review, human biases. Then evaluate existing automated moderation methods with self-constructed KuaiMod benchmark to declare their limitations in complex content understanding, user feedback perception and lengthy update cycle. To address these issues, we propose KuaiMod, a content ecosystem governance framework for SVP inspired by common-law system. Unlike traditional approaches, KuaiMod defines violations based on concrete cases rather than intricate regulations. KuaiMod first unlock the potential of VLMs for video moderation with CoT and state-transition data. Then it adopts RL paradigm to continuously refine moderation policy through interactions with the online environment. KuaiMod outperforms all baselines on offline benchmarks and significantly improves user experience after online deployment. Our experience demonstrates the crucial role of ecosystem governance in the development of SVP. 

# References



[1] Sara Abdali, Sina Shaham, and Bhaskar Krishnamachari. Multi-modal misinformation detection: Approaches, challenges and opportunities. ACM Computing Surveys, 57(3):1–29, 2024. 





[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 





[3] Mohammed Ali Al-Garadi, Kasturi Dewi Varathan, and Sri Devi Ravana. Cybercrime detection in online communications: The experimental case of cyberbullying detection in the twitter network. Computers in Human Behavior, 63:433–443, 2016. 





[4] Aymé Arango, Jorge Pérez, and Barbara Poblete. Hate speech detection is not as easy as you may think: A closer look at model validation. In Proceedings of the 42nd international acm sigir conference on research and development in information retrieval, pages 45–54, 2019. 





[5] Michele Banko, Brendon MacKeen, and Laurie Ray. A unified taxonomy of harmful content. In Proceedings of the fourth workshop on online abuse and harms, pages 125–137, 2020. 





[6] Palwasha Bibi. Legal and ethical challenges of content moderation: Balancing privacy and free speech in the ai era. To be updated, 2024. 





[7] Nibir Bora, Vladimir Zaytsev, Yu-Han Chang, and Rajiv Maheswaran. Gang networks, neighborhoods and holidays: Spatiotemporal patterns in social media. In 2013 International Conference on Social Computing, pages 93–101. IEEE, 2013. 





[8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 





[9] Eshwar Chandrasekharan, Chaitrali Gandhi, Matthew Wortley Mustelier, and Eric Gilbert. Crossmod: A cross-community learning-based system to assist reddit moderators. Proceedings of the ACM on human-computer interaction, 3(CSCW):1–30, 2019. 





[10] Kaibing Chen, Dong Shen, Hanwen Zhong, Huasong Zhong, Kui Xia, Di Xu, Wei Yuan, Yifei Hu, Bin Wen, Tianke Zhang, et al. Evlm: An efficient vision-language model for visual understanding. arXiv preprint arXiv:2407.14177, 2024. 





[11] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 





[12] Dan Cosley, Dan Frankowski, Sara Kiesler, Loren Terveen, and John Riedl. How oversight improves member-maintained communities. In Proceedings of the SIGCHI conference on Human factors in computing systems, pages 11–20, 2005. 





[13] Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. Automated hate speech detection and the problem of offensive language. In Proceedings of the international AAAI conference on web and social media, volume 11, pages 512–515, 2017. 





[14] Hang Thi-Thuy Do, Huy Duc Huynh, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen, and Anh Gia-Tuan Nguyen. Hate speech detection on vietnamese social media text using the bidirectional-lstm model. arXiv preprint arXiv:1911.03648, 2019. 





[15] Emilio Ferrara, Onur Varol, Clayton Davis, Filippo Menczer, and Alessandro Flammini. The rise of social bots. Communications of the ACM, 59(7):96–104, 2016. 





[16] Akshat Gaurav, Brij B Gupta, Kwok Tai Chui, Varsha Arya, and Priyanka Chaurasia. Deep learning based hate speech detection on twitter. In 2023 IEEE 13th International Conference on Consumer Electronics-Berlin (ICCE-Berlin), pages 1–6. IEEE, 2023. 





[17] Vaishali U Gongane, Mousami V Munot, and Alwin D Anuse. Detection and moderation of detrimental content on social media platforms: current status and future directions. Social Network Analysis and Mining, 12(1):129, 2022. 





[18] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 





[19] Xinlei He, Savvas Zannettou, Yun Shen, and Yang Zhang. You only prompt once: On the capabilities of prompt learning on large language models to tackle toxic content. In 2024 IEEE Symposium on Security and Privacy (SP). IEEE, 2024. 





[20] Martin Hilbert, Drew P Cingel, Jingwen Zhang, Samantha L Vigil, Jane Shawcroft, Haoning Xue, Arti Thakur, and Zubair Shafiq. # bigtech $@$ minors: Social media algorithms personalize minors’ content after a single session, but not for their protection. Available at SSRN 4674573, 2023. 





[21] Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146, 2018. 





[22] Wei Hu. The moderation of contentious content on twitter. Master’s thesis, University of Waterloo, 2023. 





[23] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 





[24] Claire Wonjeong Jo, Miki Wesołowska, and Magdalena Wojcieszak. Harmful youtube video detection: A taxonomy of online harm and mllms as alternative annotators. arXiv preprint arXiv:2411.05854, 2024. 





[25] Meet Ashokkumar Joshi. Artificial intelligence in e-commerce: a comprehensive analysis. Available at SSRN 4770338, 2024. 





[26] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of naacL-HLT, volume 1. Minneapolis, Minnesota, 2019. 





[27] April Kontostathis, Kelly Reynolds, Andy Garron, and Lynne Edwards. Detecting cyberbullying: query terms and techniques. In Proceedings of the 5th annual acm web science conference, pages 195–204, 2013. 





[28] Deepak Kumar, Yousef Anees AbuHashem, and Zakir Durumeric. Watch your language: Investigating content moderation with large language models. In Proceedings of the International AAAI Conference on Web and Social Media, volume 18, pages 865–878, 2024. 





[29] Irene Kwok and Yuzhou Wang. Locate the hate: Detecting tweets against blacks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 27, pages 1621–1622, 2013. 





[30] Alyssa Lees, Vinh Q Tran, Yi Tay, Jeffrey Sorensen, Jai Gupta, Donald Metzler, and Lucy Vasserman. A new generation of perspective api: Efficient multilingual character-level transformers. In SIGKDD, 2022. 





[31] Kui-long Liu, Wei Li, Chang-yuan Yang, and Guang Yang. Intelligent design of multimedia content in alibaba. Frontiers of Information Technology & Electronic Engineering, 20(12):1657– 1664, 2019. 





[32] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. ArXiv, abs/1907.11692, 2019. 





[33] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019. 





[34] Henrietta Lyons, Eduardo Velloso, and Tim Miller. Conceptualising contestability: Perspectives on contesting algorithmic decisions. Proceedings of the ACM on Human-Computer Interaction, 5(CSCW1):1–25, 2021. 





[35] Huan Ma, Changqing Zhang, Huazhu Fu, Peilin Zhao, and Bingzhe Wu. Adapting large language models for content moderation: Pitfalls in data engineering and supervised fine-tuning. arXiv preprint arXiv:2310.03400, 2023. 





[36] Todor Markov, Chong Zhang, Sandhini Agarwal, Florentine Eloundou Nekoul, Theodore Lee, Steven Adler, Angela Jiang, and Lilian Weng. A holistic approach to undesired content detection in the real world. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 15009–15018, 2023. 





[37] Sankha Subhra Mullick, Mohan Bhambhani, Suhit Sinha, Akshat Mathur, Somya Gupta, and Jidnya Shah. Content moderation for evolving policies using binary question answering. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track), pages 561–573, 2023. 





[38] Quoc Viet Hung Nguyen, Chi Thang Duong, Thanh Tam Nguyen, Matthias Weidlich, Karl Aberer, Hongzhi Yin, and Xiaofang Zhou. Argument discovery via crowdsourcing. VLDB, 26:511–535, 2017. 





[39] Thanh Tam Nguyen, Thanh Trung Huynh, Hongzhi Yin, Matthias Weidlich, Thanh Thi Nguyen, Thai Son Mai, and Quoc Viet Hung Nguyen. Detecting rumours with latency guarantees using massive streaming data. The VLDB Journal, 32(2):369–387, 2023. 





[40] Alec Radford. Improving language understanding by generative pre-training. To be updated, 2018. 





[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 





[42] Morgan Klaus Scheuerman, Jialun Aaron Jiang, Casey Fiesler, and Jed R Brubaker. A framework of severity for harmful content online. Proceedings of the ACM on Human-Computer Interaction, 5(CSCW2):1–33, 2021. 





[43] Timo Schick and Hinrich Schütze. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 255–269, 2021. 





[44] Anna Schmidt and Michael Wiegand. A survey on hate speech detection using natural language processing. In Proceedings of the fifth international workshop on natural language processing for social media, pages 1–10, 2017. 





[45] Eugenia Siapera. Ai content moderation, racism and (de) coloniality. International journal of bullying prevention, 4(1):55–65, 2022. 





[46] Leandro Silva, Mainack Mondal, Denzil Correa, Fabrício Benevenuto, and Ingmar Weber. Analyzing the targets of hate in online social media. In AAAI, 2016. 





[47] Sahana Udupa, Antonis Maronikolakis, and Axel Wisiorek. Ethical scaling for content moderation: Extreme speech and the (in) significance of artificial intelligence. Big Data & Society, 10(1):20539517231172424, 2023. 





[48] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 





[49] Soroush Vosoughi, Deb Roy, and Sinan Aral. The spread of true and false news online. science, 359(6380):1146–1151, 2018. 





[50] Wikipedia contributors. Civil law (legal system) — Wikipedia, The Free Encyclopedia, 2025. [Online; accessed February 10, 2025]. 





[51] Wikipedia contributors. Common law — Wikipedia, The Free Encyclopedia, 2025. [Online; accessed February 10, 2025]. 





[52] Donghee Yvette Wohn. Volunteer moderators in twitch micro communities: How they get involved, the roles they play, and the emotional labor they experience. In CHI, pages 1–13, 2019. 





[53] An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, and Chang Zhou. Chinese clip: Contrastive vision-language pretraining in chinese. arXiv preprint arXiv:2211.01335, 2022. 





[54] Zhilin Yang. Xlnet: Generalized autoregressive pretraining for language understanding. arXiv preprint arXiv:1906.08237, 2019. 





[55] Muhsin Yesilada and Stephan Lewandowsky. Systematic review: Youtube recommendations and problematic content. Internet policy review, 11(1), 2022. 





[56] Savvas Zannettou, Mai ElSherief, Elizabeth Belding, Shirin Nilizadeh, and Gianluca Stringhini. Measuring and characterizing hate speech on news websites. In Proceedings of the 12th ACM conference on web science, pages 125–134, 2020. 





[57] Zhixue Zhao, Ziqi Zhang, and Frank Hopfgartner. A comparative study of using pre-trained language models for toxic comment classification. In Companion Proceedings of the Web Conference 2021, pages 500–507, 2021. 



# A APPENDIX

# A.1 Instructions for Different Tasks

In this section, we present all the instructions to process data during the development of KuaiMod, as well as some prompts used to implement the baseline model. 

# A.1.1 Instruction for Tag2CoT data generation process.

In Figure A-1a, we present the instruction to guide the YuanQi model in conducting causal analysis based on video metadata and annotation results, and generating rationales for violation labels. This instruction is used the Tag2CoT process in Section 4.2.2. 

# A.1.2 Instruction for CoT2Tag data generation process.

In Figure A-1b, we present the instruction used the CoT2Tag process to prompt YuanQi-20B model to organize the analysis process into our moderation workflow in the state-transition format introduced in Section 4.2.2. 

# A.1.3 Instructions to Prompt GPT-4o and Construct Intern-VL SFT Data.

In Figure A-1c ans Figure A-1d, we present the instruction used in offline evaluation experiments (Section 5) to prompt GPT-4o mini, GPT-4o and to construct SFT data for Intern-VL model. 

# A.2 Case Study

In this section, as shown in Figures A-2a to A-2d, A-3a and A-3b , we present several real-world case analyses and decision results of KuaiMod, drawn from live online examples. These cases are deliberately challenging and potentially misleading, demonstrating KuaiMod’s superior moderation capabilities and providing an intuitive understanding of the data format and task definitions. In the input context, we provided multi-frame images of videos, titles, OCR text, ASR text, and the top 20 hottest user comments in the comment section of that video. KuaiMod performs step-by-step state analysis and transfer based on the multimodal information of these videos to obtain the final conclusion. From these examples, we can see that KuaiMod can effectively help the Kuaishou platform identify undesirable content, such as criminal behavior, offensive and cyberbullying speech, and other illegal platform bottom lines, while protecting more quality creators and videos and creating a safer and healthier community atmosphere. 

# A.3 Ethics Statement

This study adheres to strict ethical standards to ensure the credibility and confidentiality of the findings. All data underwent a thorough de-identification process to protect the privacy and maintain anonymity of the users of the short video platform. The study follows ethical guidelines and informed consent was obtained from the participants while prioritizing their rights and autonomy. Transparency and accountability are maintained throughout the study to minimize bias and conflict of interest. No academic ethical issues or misconduct are encountered and the authors affirm their unwavering commitment to upholding ethical research practices and addressing any unintentional errors or omissions in a timely manner. 

# Tag2CoT Instruction

# Human:

You are a video moderation expert. <image $1 > \ast$ <image $2 >$ …<image $\mathbf { n } \mathrm { > }$ These images are several keyframe from a video titled "<Title Text>"The video description is: "<Description Text>" The OCR of the video is: "<OCR Text>" The ASR of the video is: "<ASR Text>" Several comments for this video are "<Common Text>" This video has been reported by a user for the reason of "<Violative Category Tag>" Now, please analyze whether this report is reasonable based on the provided information. Please provide a structured and logical analysis, and give your conclusion at the end. 

# Assistant:

Based on the image and the provided information, we can analyze the reasonableness of the user's report of "<Violative Category Tag>" from multiple perspectives: <Analysis Process> 

(a) Instruction for YuanQi-20B to generate CoT rationales for video’s violation category tag based on video metadata and use comment. 

# CoT2Tag Instruction

# Human:

You are a video moderation expert. <image $1 >$ <image $2 >$ …<image $\mathrm { n } >$ These images are several keyframes from a video titled "<Title Text>" The video description is: "<Description Text>" The OCR of the video is: "<OCR Text>" The ASR of the video is: "<ASR Text>" Several comments for this video are "<Common Text>" This video has been classified as "<Violative Category Tag>" and here is an analysis about the reason: 

# <Tag2CoT output>

Please generate a state transition formatted review process for this video, using its metadata as the initial state. Your review process should include: 

(1) Content Extraction: Extract content from video frames and generate an overall description of the video 

(5) Summary: Summarize all above states, assign the final violation tag, and provide key reasons for the judgment. 

# Assistant:

Here is the state-transition moderation process: <State-Transition Moderation Process> 

(b) Instruction for YuanQi-20B to organize metadata, analysis and violative tag into a workflow. 

# Binary Classification Prompt For GPT4o and Intern-VL

# Human:

You are a video moderation expert. <image 1><image 

2>…<image $\mathbf { n } >$ These images are several keyframes from a video titled "<Title Text>" The video description is: 

"<Description Text>" The OCR of the video is: "<OCR Text>" The ASR of the video is: "<ASR Text>" Several comments for this video are "<Common Text>" Please classify whether is video is violative or not. 

# Assistant:

Yes/No 

(c) Instructions for binary-class classification. 

# Multi-Class Classification Prompt For GPT4o and Intern-VL

# Human:

You are a video moderation expert. <image 1><image 

2>…<image $\mathrm { \tt n } >$ These images are several keyframes from a video titled "<Title Text>" The video description is: "<Description 

Text>" The OCR of the video is: "<OCR Text>" The ASR of the video is: "<ASR Text>" Several comments for this video are 

"<Common Text>" Here is the taxonomy for violative content, please classify which category this video belongs to. 

Assistant: 

<Violative Category Tag> 

(d) Instructions for multi-class classification. 

Figure A-1: Instructions for YuanQi on different tasks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e240f9a3-caa9-48ff-8551-48667354bd05/df504e230b3f2d82675bdecd25fe9b8c4351d5f5edb28d6b0fd0f596a52310ec.jpg)



(a) Violative case 1 of KuaiMod.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e240f9a3-caa9-48ff-8551-48667354bd05/b239d944f4f7870f4a883fec512befca8f3bacb9b04b1135d6615f111f997d19.jpg)



(b) Violative case 2 of KuaiMod.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e240f9a3-caa9-48ff-8551-48667354bd05/fb1e6b7135404fc81ef0438bb99aa32fd9aee4c20934d575d24263cc308456b2.jpg)



(c) Violative case 3 of KuaiMod.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e240f9a3-caa9-48ff-8551-48667354bd05/20fb48e9a8d82828f0e0c0446220fd476fbdc38cf660e45fc06bb93e5dde5091.jpg)



(d) Violative case 4 of KuaiMod.



Figure A-2: Violative cases 1 to 4 of KuaiMod.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e240f9a3-caa9-48ff-8551-48667354bd05/37988264201f307d5c17cc44be6409bb77368f57e7c9c7569a83756297a06ffc.jpg)



(a) Positive case 1 of KuaiMod.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e240f9a3-caa9-48ff-8551-48667354bd05/01640731f6a8e338592cdbdae3db4d13bb76d27f9f40e898f078f868119fd9db.jpg)



(b) Positive case 2 of KuaiMod.



Figure A-3: Positive cases 1 to 2 of KuaiMod.
