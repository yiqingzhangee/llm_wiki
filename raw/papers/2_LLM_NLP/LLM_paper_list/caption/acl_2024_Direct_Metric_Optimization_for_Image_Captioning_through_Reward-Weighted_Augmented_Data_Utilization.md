**026** 

## Direct Metric Optimization for Image Captioning through Reward-Weighted Augmented Data Utilization 

## Anonymous ACL submission 

## Abstract 

**001** While image captioning is an essential field of **002** vision language models (VLM), a lack of con- **003** tinuity between the learning objective and fi- **004** nal performance metrics of VLMs complicates **005** their training and optimization. Reinforcement **006** learning (RL) can directly optimize such met- **007** rics, but it is accompanied by a significant com- **008** putational cost, making it difficult to apply to **009** recent large-scale VLMs. In this paper, we **010** propose Direct Metric Optimization (DMO), **011** which is a lightweight final-metric-optimizing **012** training method. We replace the computa- **013** tionally expensive exploration process in RL **014** with an offline, diverse text data augmenta- **015** tion and show that self-supervised training on **016** reward-weighted augmented data leads to di- **017** rect and stable metric optimization. Our exper- **018** iments demonstrate that DMO achieves perfor- **019** mance comparable to those of the state-of-the- **020** art RL method while saving hundreds of times **021** more model forwarding iterations and greater **022** amounts of computation time. This suggests **023** that DMO constitutes a promising alternative **024** for metric optimization in the era of large-scale **025** VLMs. 

## 1 Introduction 

**027** With the advent of CLIP (Radford et al., 2021), the **028** boundaries between vision and language modalities **029** in machine learning have been dissolved, leading **030** to rapid advancements in research involving these **031** areas. Furthermore, the rise of large language mod- **032** els (LLM) has led to the emergence of large-scale **033** vision language models (VLM), extending their **034** influence to practical applications. For example, **035** models such as ChatGPT (Achiam et al., 2023) and **036** Gemini (Team et al., 2023) generate detailed natu- **037** ral language descriptions from visual information. **038** With the increasing prevalence of VLMs, methods **039** for customizing and fine-tuning these models for **040** specific domains or individuals are attracting signif- **041** icant interest and attention (Sun et al., 2023; Zhao **042** et al., 2023). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0001-07.png)


**----- Start of picture text -----**<br>
CE CE CE<br>Captioner Captioner Captioner<br>**----- End of picture text -----**<br>


Figure 1: An overview of our Direct Metric Optimization (DMO). Image and tokens are denoted as I and yi and CE stands for cross entropy function. Precomputed rewards r assign different weights to each sample in a textually augmented dataset, effectively enhancing the targeted performance metrics. 

Recent standard captioning models adopt self- **043** supervised learning for training purposes (Wang **044** et al., 2022; Yu et al., 2022; Alayrac et al., 2022; **045** Li et al., 2023). This method treats the ground **046** truth captions both as inputs and labels, and the **047** model predicts only the next token from the given **048** image and preceding tokens. Specifically, recent **049** transformer-based encoder-decoder models can **050** conduct the next token prediction of each step **051** in parallel, significantly enhancing their computa- **052** tional efficiency. However, this approach is subject **053** to certain limitations. Typically, the performance of **054** image captioning is evaluated using metrics such as **055** BLEU (Papineni et al., 2002) or CIDEr (Vedantam **056** et al., 2015); however, self-supervised learning of **057** language modeling does not necessarily optimize **058** those metrics. To target final metrics directly, re- **059** inforcement learning (RL) methods have been em- **060** ployed (Ranzato et al., 2015; Zhang et al., 2017b; **061** Rennie et al., 2017; Gao et al., 2019). Reinforce- **062** 

1 

label-smoothed vector (Szegedy et al., 2016) from **113** ground truth label yi. The objective function of self- **114** supervised learning for language modeling LLM is **115** expressed as follows: **116** 

**063** ment learning is a powerful method capable of op- **064** timizing even non-differentiable metrics; however, **065** it has certain drawbacks, such as learning instabil- **066** ity and significant time and computational costs. **067** With the growing trend of using large pre-trained **068** models, those challenges have become increasingly **069** serious. Conducting RL with models containing **070** billions of parameters demands extensive computa- **071** tional time and resources, making the application **072** of RL methods impractical. **073** To bypass the prohibitive computational cost of **074** RL, we propose to replace the expensive explo- **075** ration process in RL with diverse text data aug- **076** mentation and reduce RL to simple importance- **077** weighted self-supervised learning. The approach **078** that utilizes previously collected data for RL is **079** known as offline-RL (Levine et al., 2020). Partic- **080** ularly in our approach, datasets are augmented by **081** various methods and the augmentation diversity **082** brings a variety of samples of different rewards, **083** enabling the efficient estimation of the optimal cap- **084** tion for the image. We call this metric-optimizing **085** self-supervised training Direct Metric Optimization **086** (DMO). Our experiments demonstrate that DMO **087** achieves performance on par with state-of-the-art **088** (SOTA) RL methods in standard image captioning **089** metrics while retaining lightweight computational **090** efficiency and learning stability. This highlights **091** DMO’s significant practical advantages in metric **092** optimization, especially considering the increasing **093** need to tune and customize large-scale VLMs. 

**094** 

**095 096** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0002-04.png)


## 2 Preliminaries 

## 2.1 Self-Supervised Learning for Image Captioning 

**097** The standard approach for image captioning in re- **098** cent years has been to employ an encoder-decoder **099** model, where the encoder maps the image into the **100** latent space and extracts features from the image, **101** and the text decoder autoregressively generates to- **102** kens for the next step from extracted features and **103** previously generated tokens. In the self-supervised **104** learning of language models (LM), the model is of- **105** ten trained by teacher-forcing (Williams and Zipser, **106** 1989). This method increases the likelihood of **107** ground-truth sentences by aligning the model’s **108** conditional distribution pθ(yi|I, y<i) with the la- **109** bel distribution q(yi) using cross-entropy (CE) for **110** each step i ∈{1, . . . , T }. Here, yi represents **111** a text token at step i and I is the given image. **112** The label distribution q(yi) is a one-hot vector or 

Especially in the recent Transformer-based archi- **118** tecture, the predictions of the next tokens at each **119** time step can be performed in parallel (Vaswani **120** et al., 2017). Thus this training method is extremely **121** time and computationally efficient because it does **122** not require recursive operations, as is the case with **123** conventional RNN-based methods (Vinyals et al., **124** 2015; Xu et al., 2015). **125** 2.2 Reinforcement Learning for Image **126** Captioning **127** Typically, the performance of captioning models **128** is assessed using metrics such as BLEU or CIDEr. **129** However, self-supervised learning does not neces- **130** sarily optimize these metrics. Furthermore, these **131** evaluation metrics are often non-differentiable, **132** making it impossible to apply the gradient descent **133** directly. One approach for directly optimizing **134** those non-differentiable metrics is to employ RL. **135** Recent studies of image captioning have applied **136** the various RL algorithms including REINFORCE **137** and Actor-Critic (Ranzato et al., 2015; Zhang et al., **138** 2017b; Liu et al., 2017; Rennie et al., 2017; Zhang **139** et al., 2021), to captioning tasks by regarding the **140** captioning models as agents and the final evalu- **141** ation metrics (such as CIDEr) as rewards. The **142** objective function of the captioning model in the **143** RL framework is expressed as follows: **144** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0002-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0002-10.png)


2 

**156** In RL, the reward metric need not be confined **157** to automatic evaluation scores. It can accept any **158** user-defined score including ambiguous evalua- **159** tions from humans (Christiano et al., 2017; Ouyang **160** et al., 2022), thus demonstrating significant versa- **161** tility and adaptability to various tasks. 

tic variations is not straightforward, and universally **206** effective methods for achieving this are yet to be **207** established (Feng et al., 2021). **208** 

**162** Notwithstanding these advantages, the RL **163** methodology presents significant challenges. First, **164** for the method to be effective, extensive explo- **165** ration is needed, which makes RL time-consuming. **166** Second, the learning process tends to be unstable, **167** especially during the early stage of training, when **168** the model poorly samples the high-rewarded se- **169** quences. (Ranzato et al., 2015; Rennie et al., 2017). **170** In addition, the sampling process in RL is compu- **171** tationally inefficient because the gradient compu- **172** tation requires the last token wT (see Equation 3), **173** but token generation is done in a left-to-right man- **174** ner, undermining the computational parallelism of **175** Transformer architecture. 

**176** 

**195 196 197 198 199 200 201 202 203 204** 

**205** 

## 2.3 Text Data Augmentation (TDA) 

**177** Data augmentation is a traditional yet effective **178** method that is used to enhance a language model’s **179** performance (Li et al., 2023; Fan et al., 2023; Yang **180** et al., 2023). Previous studies have shown that text **181** data augmentation (TDA) strategies can be broadly **182** categorized into three types: paraphrasing, noising **183** and sampling (Li et al., 2022). Paraphrasing is a **184** method that generates data that convey very simi- **185** lar information as the original data with restrained **186** changes. Noising adds noise to datasets to improve **187** the robustness of the model. Sampling produces **188** new novel data from the data distribution master. **189** The primary objective of these strategies is to intro- **190** duce diversity into the dataset. This is particularly **191** crucial in scenarios with limited datasets, where **192** models are prone to overfitting. Augmenting the **193** dataset and smoothing the distribution can effec- **194** tively prevent this overfitting. 

In the fields of image and audio processing, data augmentation has traditionally achieved significant success (DeVries and Taylor, 2017; Zhang et al., 2017a). In contrast, it has not been explored as extensively in the field of natural language processing. This disparity can be attributed to the challenges inherent in text augmentation. Unlike images and audio, which comprise continuous data, tokenized text data is discrete and even minor alterations can lead to significant semantic shifts. Implementing superficial changes while controlling these seman- 

## 3 Proposed Framework 

**209** 

## 3.1 Overview of Method 

**210** 

In RL, the sequences are sampled from the model’s **211** distribution pθ, but this raises problems of its **212** high computational cost and instability in the early **213** stages of training. We propose the following per- **214** spective shift: What if we were to consider the **215** sequences drawn from the given dataset as the **216** sequences sampled from the model itself? This **217** approach allows us to obtain the gradients of RL **218** objective function with ground truth data in a self- **219** supervised manner. Furthermore, since sequences **220** are pre-sampled, the bottleneck in RL, specifically **221** the recursive generation process, is resolved. This **222** significantly enhances computational efficiency. **223** The approach that utilizes previously collected **224** data for RL is known as "offline-RL" (Levine et al., **225** 2020), and is commonly used to bypass the compu- **226** tationally expensive exploration in RL (Chen et al., **227** 2021; Jang et al., 2022; Shi et al., 2023; Baheti **228** et al., 2023). Applying the offline-RL to image **229** captioning, however, is not straightforward. First, **230** because of the nature of major captioning metrics **231** (such as BLEU, METEOR, and ROUGE-L) that **232** measure the overlap of n-grams, words or subse- **233** quences with a set of ground-truth captions, ground **234** truth captions always receive rewards of 1 in the **235** offline-RL framework. Because the rewards are **236** indicators of the quality of samples, receiving a **237** constant value of rewards gives no clue about how **238** good each caption is, and consequently, there is no **239** advantage compared with standard self-supervised **240** training. The second problem is data suboptimality, **241** a common challenge in offline-RL (Levine et al., **242** 2020). The reliance on limited static data restricts **243** exposure to high-reward samples, thereby capping **244** the model’s performance improvements. We ad- **245** dress those obstacles by introducing diverse text **246** data augmentation (TDA). With TDA, we expose **247** models to a variety of expressions with different **248** rewards outside the original dataset, providing a **249** greater number of clues about the optimal caption **250** for the images. Furthermore, substituting TDA **251** for exploration improves the stability of the learn- **252** ing process in the early stage. This is because, **253** unlike RL, TDA can consistently provide reason- **254** able quality samples and training does not rely on **255** 

3 

**261 262 263** 

**264** 

**265** 

**266** 

**267** 

**275 276** 

**277** 

**278** 

**279** 

**280** 

**281** 

**282** where q(yi) is a one-hot vector from label yi. This **283** can be interpreted as a reward-weighted gradient of **284** self-supervised learning loss. This approach elimi- **285** nates the bottleneck inherent in RL, specifically the **286** recursive generation process, through the utiliza- **287** tion of pre-sampled sequences. Consequently, it **288** enables the model to leverage the parallel computa- **289** tional capabilities of the Transformer architecture, **290** resulting in a substantial enhancement of computa- **291** tional efficiency. **292** This reward-weighted self-supervised training **293** on augmented datasets is related to noise/similarity- **294** aware supervised training that adaptively assigns **295** different weights to each sample (Atliha and Šešok,Atliha and Šešok,, **296** 2020; Yang et al., 2023; Kang et al., 2023).; Yang et al., 2023; Kang et al., 2023). Yang et al., 2023; Kang et al., 2023)., 2023; Kang et al., 2023). 2023; Kang et al., 2023).; Kang et al., 2023). Kang et al., 2023)., 2023). 2023).). How- **297** ever, there are notable differences. First, while **298** those noise-aware methods often focus on large- **299** 

**256** the model’s capability of sampling high-reward se- **257** quences. This metric-optimizing self-supervised **258** training on textually augmented datasets, which we **259** call Direct Metric Optimization, offers the follow- **260** ing two significant advantages. 

1. It allows direct optimization of metrics in a self-supervised manner, significantly enhancing computational efficiency. 

2. Training is stable even at an early stage because it does not rely on the model’s capability of generating captions of high rewards. 

## 3.2 Direct Metric Optimization 

**268** In the proposed DMO method, sequences are sam- **269** pled from textually augmented dataset Daug. Be- **270** cause the dataset is known, scores for each ground **271** truth sample can be calculated in advance. Let the **272** score function (for example, the BLEU and CIDEr **273** scorer) be the reward function r(·). Once ground **274** truth data d = {y1, . . . , yT } from dataset Daug is **275** sampled, the gradient of our DMO objective LDMO **276** is defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0004-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0004-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0004-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0004-20.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0004-21.png)


This reward-weighted self-supervised training on augmented datasets is related to noise/similarityaware supervised training that adaptively assigns different weights to each sample (Atliha and Šešok,Atliha and Šešok,, 2020; Yang et al., 2023; Kang et al., 2023).; Yang et al., 2023; Kang et al., 2023). Yang et al., 2023; Kang et al., 2023)., 2023; Kang et al., 2023). 2023; Kang et al., 2023).; Kang et al., 2023). Kang et al., 2023)., 2023). 2023).). However, there are notable differences. First, while those noise-aware methods often focus on largescale pre-training from noisy datasets and mitigate 

the effect of noisy samples, our approach features **300** the finetuning stage with relatively small and clean **301** datasets, and deliberately augments datasets to in- **302** troduce the diversity of samples. Second, while **303** similarity-aware methods often utilize CLIP/BERT **304** scores or custom weights (Ding et al., 2019; Atliha **305** and Šešok, 2020; Yang et al., 2023), we directly **306** employ target metrics for sample weighting. While **307** the CLIP/BERT score is useful for denoising or **308** filtering, training with these measures does not di- **309** rectly lead to the optimization of the final metrics. **310** With these perspectives, our method enables more **311** effective optimization of the target metrics. **312** 

## 4 Experiment Implementations 

**313** 

This section describes the detailed experiment im- **314** plementations for the evaluation of our DMO train- **315** ing. **316** 

## 4.1 Datasets and Captioning Models 

**317** 

We validate our method with the MS-COCO **318** dataset (Lin et al., 2014) and Flickr8K (Hodosh **319** et al.), which are commonly used in image cap- **320** tioning research. Both datasets have 5 captions per **321** image. For the evaluation, the datasets are split into **322** training, validation, and testing sets according to **323** the Karpathy method (Karpathy and Fei-Fei, 2015) **324** so that the numbers of images in the training, val- **325** idation and test datasets become 6091/1000/1000 **326** for Flickr8k and 113287/5000/5000 for MS- **327** COCO. For captioning models, we employ GIT- **328** base/large (Wang et al., 2022) and BLIP2-2.7b (Li **329** et al., 2023), which have different sizes of param- **330** eters and architectures. GIT has a simplified ar- **331** chitecture of one image encoder and one text de- **332** coder and the base model has 178M parameters **333** while the large model has 390M parameters for **334** each. BLIP2-2.7B has 2.7B parameters and it em- **335** ploys large pre-trained frozen models for its vi- **336** sion encoder and text decoder. Both models are **337** pre-trained on datasets that include COCO (Lin **338** et al., 2014), Visual Genome (Krishna et al., 2017), **339** CC3M (Sharma et al., 2018), and other large image- **340** text pair datasets. In our experiments, all mod- **341** els are finetuned for 3 epochs with learning rate **342** 1.0 × 10[−][5] and batch size 960, using a fixed sin- **343** gle random seed. Further details are explained in **344** Appendix A.1. **345** 

4 

**346** 4.2 Text Data Augmentation Strategy **347** Based on Section 2.3, we adopt the following three **348** augmentation methods accordingly. From each **349** of the following three methods, two augmented **350** captions are randomly sampled for each image and **351** added to the original training dataset. 

**366** We do not explicitly adopt a noising strategy, as suf- **367** ficient semantic noise is introduced by each TDA **368** method. 

**369** 

**379** 

**380** We evaluate our proposed method in terms of met- **381** ric optimization performance, learning stability and **382** computational efficiency. We further investigate **383** how reward-weighting architecture facilitates ro- **384** bust metric optimization by comparing DMO with **385** standard LM training under noisy data and limited **386** data settings. 

**387 388** 

**389** 

**390 391 392** 

- **352** • Back-translation: The En-Fr translation **353** model from MarianNMT (Junczys-Dowmunt **354** et al., 2018) is adopted. Back-translation is **355** applied to each ground truth caption and the **356** same number of back-translated captions as **357** original captions are created. 

- **358** • Pre-trained VLM Sampling: Using the **359** COCO-pre-trained BLIP2-6.7B model, five **360** captions are generated from each image in the **361** training dataset with temperature 1.0. 

- **362** • Paraphrasing by LLM: We employ Llama2- **363** 7b-chat (Touvron et al., 2023) to paraphrase **364** captions. The detailed prompt text is pre- **365** sented in Appendix B. 

## 4.3 Metrics and Rewards 

**370** We evaluate the performance of models by CIDEr, **371** BLEU-4, METEOR (Banerjee and Lavie, 2005), **372** ROUGE-L (Lin, 2004) and SPICE (Anderson et al., **373** 2016). Since our method requires each sample to **374** be scored by a reward function, we directly use the **375** metrics above as the rewards in the training. As **376** the scoring of each metric demands a set of ground **377** truth captions as references, we employ the original **378** training dataset as the reference dataset. 

## 5 Results 

## 5.1 Evaluating Metric Optimization Performance 

## 5.1.1 Does DMO Enhance Final Metrics? 

First, we examine whether our method effectively improves the targeted metrics for image captioning. We use a textually augmented Flickr8k dataset 

|Training<br>Metric|Evaluation Metrics<br>CIDEr<br>B4<br>MET.<br>ROU.|
|---|---|
|CIDEr<br>BLEU-4<br>METEOR<br>ROUGE-L|97.0<br>33.3<br>27.7<br>57.5<br>96.8<br>33.5<br>27.7<br>57.5<br>96.1<br>32.8<br>27.5<br>57.0<br>96.0<br>33.0<br>27.6<br>57.5|
|standard LM|95.1<br>33.4<br>27.0<br>57.1|



Table 1: Performance evaluation of GIT-base model optimized for each metric by DMO on the textually augmented Flickr8k dataset. ’B4’, ’MET.’ and ’ROU.’ refer to BLEU-4, METEOR and ROUGE-L, respectively. 

(TDA-Flickr8k) and apply DMO to the GIT-base **393** so that each CIDEr, BLEU-4, METEOR, and **394** ROUGE-L is optimized respectively. We then eval- **395** uate whether DMO improves these metrics com- **396** pared to training with the standard Language Model **397** (LM) loss. The result is presented in table 1. We **398** find that when optimized for each metric, there is **399** an improvement in each metric compared to train- **400** ing with the standard LM loss. This result implies **401** that our method can effectively enhance the tar- **402** get metrics. Interestingly, optimizing for CIDEr or **403** BLEU-4 leads to improved scores in other metrics **404** as well. This can be attributed to the similarities in **405** the way each evaluation metric is measured. In the **406** following experiments, we use CIDEr as the target **407** metric because CIDEr-optimizing DMO leads to **408** general improvements in scores across other met- **409** rics. **410** 5.1.2 Comparison of DMO and LM Training **411** We compare the performance of DMO training with **412** LM training for different models and dataset set- **413** tings. We use three models, GIT-base, GIT-large, **414** and BLIP2-2.7b and two datasets, the Flickr8k and **415** the COCO dataset. We apply CIDEr-optimizing **416** DMO to each model and compare CIDEr, BLEU- **417** 4, METEOR, ROUGE-L, and SPICE scores with **418** the models trained by standard LM loss. The re- **419** sults are presented in Table 2. We observe that **420** DMO results in significant performance improve- **421** ments in almost all models and datasets compared **422** with models trained with LM loss without TDA. **423** On the Flickr8k dataset, all models exhibit score **424** improvements across all metrics with DMO. A sim- **425** ilar trend is observed when fine-tuning GIT-base **426** on the COCO dataset. This suggests that DMO **427** consistently enhances the scores beyond standard **428** LM training across various models and datasets. **429** In the analysis comparing with LM training on **430** 

5 

**431** TDA datasets, we observe that LM training on TDA **432** datasets causes a decline in the performance in cer- **433** tain scenarios, such as training GIT-base/BLIP2 on **434** Flickr8k and GIT-base/large on the COCO. This **435** implies that the TDA datasets possess excessive **436** noise and this noise leads to a deterioration in the **437** performance of the models trained with standard **438** LM loss. In contrast, DMO training, which utilizes **439** the TDA dataset, exhibits rather enhanced perfor- **440** mance. BLIP2 trained with DMO on Flickr8k and **441** GIT-base DMO-trained on the COCO show im- **442** proved scores for almost all metrics while those **443** trained with LM loss show worse performance by **444** introducing TDA. These findings indicate that our **445** DMO can effectively leverage even noisy datasets **446** that would deteriorate the performance of regular **447** LM training. Moreover, when GIT-large is trained **448** on TDA-COCO, a reduction in performance is ob- **449** served for both LM and DMO training. However, **450** the decline in performance is significantly differ- **451** ent: 6.1 points for LM training compared with 0.3 **452** points for DMO training, highlighting DMO’s ro- **453** bustness under noisy dataset conditions. 

**454** 

|Captioning<br>Optimization<br>Model<br>Method|Flickr8k<br>CIDEr<br>B4<br>MET.<br>ROU.<br>SPICE|MS-COCO<br>CIDEr<br>B4<br>MET.<br>ROU.<br>SPICE|
|---|---|---|
|GIT-base<br>LM<br>GIT-base<br>LM w/ TDA<br>GIT-base<br>DMO|95.1<br>33.4<br>27.0<br>57.1<br>21.4<br>93.6<br>33.2<br>26.9<br>57.0<br>21.0<br>99.6<br>35.4<br>27.9<br>58.1<br>22.3|135.6<br>41.0<br>30.5<br>60.4<br>23.6<br>132.1<br>39.4<br>29.8<br>59.9<br>23.5<br>137.4<br>41.5<br>30.5<br>60.9<br>24.0|
|GIT-large<br>LM<br>GIT-large<br>LM w/ TDA<br>GIT-large<br>DMO|96.3<br>33.3<br>26.9<br>56.9<br>21.2<br>101.1<br>34.9<br>27.9<br>58.0<br>22.2<br>110.7<br>37.6<br>29.1<br>60.2<br>23.2|140.9<br>42.5<br>31.3<br>61.3<br>24.3<br>134.7<br>39.8<br>30.3<br>60.4<br>23.9<br>140.6<br>42.0<br>31.1<br>61.5<br>24.3|
|BLIP2-2.7b<br>LM<br>BLIP2-2.7b<br>LM w/ TDA<br>BLIP2-2.7b<br>DMO|101.3<br>33.8<br>28.6<br>58.5<br>23.4<br>100.2<br>32.7<br>28.3<br>58.2<br>22.8<br>103.7<br>33.8<br>28.7<br>58.4<br>23.7|132.2<br>39.1<br>29.6<br>59.5<br>23.3<br>132.9<br>38.7<br>29.9<br>59.7<br>23.6<br>138.3<br>41.1<br>30.4<br>60.7<br>24.0|



Table 2: Evaluation of three models trained by standard LM training (with and without TDA) and CIDEr-optimizing DMO on Flickr8k and COCO datasets. The performance metrics include CIDEr, BLEU-4 (B4), METEOR (MET.), ROUGE-L (ROU.) and SPICE. 

|Dataset<br>Setting|Evaluation Metrics<br>CIDEr<br>B4<br>MET.<br>ROU.<br>SPICE|
|---|---|
|Dbktrs<br>Dblip2<br>Dllama<br>Dall|93.6<br>32.8<br>27.3<br>56.9<br>22.1<br>98.2<br>33.0<br>28.0<br>57.9<br>22.1<br>99.5<br>34.3<br>28.0<br>58.0<br>21.9<br>99.6<br>35.4<br>27.9<br>58.1<br>22.3|
|baseline|95.1<br>33.4<br>27.0<br>57.1<br>21.4|



Table 3: Scores of GIT-base model trained with CIDEroptimizing DMO on each dataset setting. Baseline is the score of LM training without any TDA. B4: BLEU-4, MET: METEOR, ROU: ROUGE-L. 

## 5.1.3 Does TDA-Diversity Matter? 

**455** We hypothesize that diversifying the augmenta- **456** tion techniques serves as a replacement for ex- **457** ploration, enhancing the performance of DMO. **458** To validate this hypothesis, we conduct an abla- **459** tion study and evaluate the performance of mod- **460** els trained with DMO on datasets augmented by **461** a single method and on datasets augmented by **462** multiple methods, respectively. We denote the **463** datasets augmented solely by the back-translation, **464** pre-trained BLIP2 sampling, and Llama2 para- **465** phrasing as Dbktrs, Dblip2, and Dllama respectively. **466** For a fair comparison, each dataset is adjusted to 

have approximately the same number of image- **467** caption pairs. For datasets Dbktrs, Dblip2, Dllama, **468** we increase 5 captions per image by augmentation. **469** Note that the dataset Dall is constructed by sam- **470** pling two augmented captions from each augmenta- **471** tion method and adding them to the original dataset. **472** We use the Flickr8k dataset and train GIT-base by **473** CIDEr-optimizing DMO. The results are presented **474** in Table 3. While all data augmentation methods **475** except for back-translation improve performance **476** over the baseline, which is the score of LM train- **477** ing without any TDA, the highest performance is **478** achieved with the dataset that combines all aug- **479** mentation methods, suggesting that exposing the **480** model to a variety of expressions from diverse aug- **481** mentation techniques yields the most significant **482** performance improvement. Further analysis of the **483** advantages of combining multiple TDA methods is **484** discussed in Appendix C. **485** 

## 5.2 Stability and Computational Efficiency 

Stability and Computational Efficiency **486** DMO replaces the exploration with TDA and re- **487** solves the learning instability and computational **488** bottleneck of RL. We examine DMO’s stability and **489** 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0007-00.png)


**----- Start of picture text -----**<br>
3.2 DMO<br>LM<br>3.0<br>SCST<br>2.8<br>2.6<br>2.4<br>2.2<br>2.0<br>1.8<br>1.6<br>0 50 100 150 200 250 300 350<br>step<br>score<br>**----- End of picture text -----**<br>


Figure 2: Transition of CIDEr scores for the GIT-base model trained using three different methods: DMO, standard LM training, and SCST. The scores reflect the CIDEr values of sequences greedily generated by each model for images in the mini-batches of the Flickr8k training dataset. 

**490** efficiency by comparing them with those of RL and **491** standard LM training. We use the GIT-base model **492** trained on the Flickr8k dataset. As an RL method, **493** we employ SCST (Rennie et al., 2017), which is **494** one of the most prominent reinforcement learning **495** methods for image captioning, utilized in training **496** many SOTA models (Wang et al., 2022; Xu et al., **497** 2023). We assess the stability of training by con- **498** ducting experiments with five seeds and calculating **499** the average and variance of the final scores at the **500** 3rd and 20th epoch. To equalize the number of **501** parameter updates, the number of explorations in **502** RL is set to equal the number of image-text pairs in **503** the training dataset. SCST requires intricate hyper- **504** parameter tuning and we present the detailed con- **505** figuration for SCST training in Appendix A.2. The **506** result is presented in Table 4. With 3-epoch train- **507** ing, our DMO produces the highest score. SCST is **508** still unstable at 3 epochs, with a standard deviation **509** (SD) of 4.1, which is significantly larger than LM’s **510** SD of 0.74 or DMO’s SD of 1.26. Figure 2 shows **511** the CIDEr score progression on training data up to **512** 5 epochs for DMO, LM, and SCST, respectively. **513** While DMO and LM show steady score improve- **514** ments, SCST exhibits large fluctuations. This insta- **515** bility in SCST is attributed to its poor capability of **516** sampling high-reward sequences in early training. **517** On the other hand, DMO utilizes samples from **518** TDA throughout the entire training, which makes **519** training more stable. By the end of the 20th epoch, **520** SCST achieves the highest score, owing to SCST’s 

|||
|---|---|
|Method|3 epochs<br>20 epochs|
|LM<br>94.4±0.74<br>94.2±1.65<br>SCST<br>93.6±4.10<br>99.8±1.50<br>DMO<br>97.7±1.26<br>98.0±1.36||



Table 4: The average and standard deviation of CIDEr scores when the model was trained for 3 epochs and 20 epochs with five different seeds by each method. 

|Optimization<br>Method|Forwarding<br>Execution<br>iterations<br>time|
|---|---|
|LM<br>SCST<br>DMO|1.68×103<br>13sec<br>1.50×105<br>9971sec<br>1.68×103<br>14sec|



Table 5: The number of model forwarding iterations and execution time for the GIT-base to complete 3 epochs on Flickr8k. The time is measured during the loss computation and the number of iterations is measured by counting the number of batch model forwarding. 

ability to continually explore and obtain new sam- **521** ples. DMO displays minimal score improvements **522** from epoch 3 to epoch 20, suggesting that training **523** for only 3 epochs may suffice for model optimiza- **524** tion in DMO training while SCST requires at least **525** 20 epochs. This indicates that DMO optimizes the **526** performance of the model more rapidly than SCST. **527** Additionally, we measure the number of model **528** forwarding iterations and the training time required **529** for each method to complete 3 epochs. To elimi- **530** nate the impact of differences in implementation **531** and hardware differences on timing, we specifically **532** measure the duration between feeding the data to **533** the model and obtaining the loss values. The re- **534** sult is presented in Table 5. With respect to com- **535** putational efficiency, we find that LM and DMO **536** have the same number of batch forwardings, while **537** SCST requires approximately 100 times more (the **538** rationale behind the results is explained in Ap- **539** pendix A.1). In terms of execution time, while **540** DMO training takes approximately the same du- **541** ration as LM training, SCST requires about 1000 **542** times longer. The slight increase in time for DMO **543** training compared to LM training is due to the ne- **544** cessity of reward-weighting operations. On the **545** other hand, SCST requires the recursive genera- **546** tion process, resulting in a number of forwardings **547** approximately 100 times greater and a duration ap- **548** proximately 1000 times longer, compared to LM **549** 

7 

**550 551** 

**552** 

**553** 

**554** 

**555 556 557** 

**558 559** 

**560** 

**561** In this experiment, we examine how robustly our **562** method can train models on the noisy dataset. To **563** simulate the case where the training dataset is ex- **564** tremely noisy, we construct datasets in which a **565** certain percentage of the ground truth captions in **566** the Flickr8k dataset are replaced with entirely ir- **567** relevant captions (we randomly sampled from the **568** COCO dataset). With these datasets, we train GIT- **569** base both with DMO and LM loss and observe **570** how training is affected by those toxic samples. In **571** DMO training, we use the original clean dataset for **572** the reference dataset to ensure that the quality of **573** each sample is accurately scored. We increase the **574** noise ratio from 0% to 80% and evaluate the CIDEr **575** scores of the model trained by LM and DMO. The **576** results are presented in Table 6. Compared to the **577** baseline, both LM and DMO training exhibit a de- **578** cline in performance; however, while LM training **579** experiences a significant performance drop, DMO **580** manages to minimize this reduction. This indi- **581** cates that, by utilizing scores as cues, our proposed **582** method can effectively discern samples that should **583** be learned from samples that should be ignored, **584** enabling robust learning even from noisy datasets. 

||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Noise ratio|||LM|||DMO||Dataset Size|||LM||DMO||
||||||||||||||||
||||||||||||||||
|0%||95.1|(−0%)||98.6|(−0%)||100%||95.1|(−0%)||98.6|(−0%)|
|20%<br>40%<br>60%||82.1<br>69.6<br>58.6|(−14%)<br>(−27%)<br>(−38%)||97.9<br>98.0<br>90.9|(−1%)<br>(−1%)<br>(−8%)||80%<br>60%<br>40%||92.1<br>93.8<br>90.4|(−3%)<br>(−1%)<br>(−5%)||97.4<br>97.6<br>95.4|(−1%)<br>(−1%)<br>(−3%)|
|80%||5.1|(−95%)||81.9|(−17%)||20%||86.5|(−9%)||95.0|(−4%)|
||||||||||||||||



Table 6: CIDEr scores of GIT-base model trained with noisy dataset. Noise ratio is the ratio of original groundtruth captions replaced by irrelevant random captions. The numbers in parentheses represent the percentage decrease from the score at the 0% noise ratio. 

Table 7: CIDEr scores of GIT-base model trained with the limited number of data. Dataset size is the volume of the available training data. The numbers in parentheses represent the percentage decrease from the score at the 100% dataset volume. 

and DMO training. These results emphasize the DMO’s substantially greater computational efficiency compared to that of SCST. 

## 5.3 Noise Robustness and Data Efficiency 

We experimentally show that DMO can train models robustly even in data-noisy or low-resource settings, by effectively leveraging the reward. To evaluate the pure effect of reward utilization, we compare LM training and DMO training without TDA. 

## 5.3.1 Evaluation on Extremely Noisy Dataset 

## 5.3.2 Evaluation under Low-Resource Setting 

Evaluation under Low-Resource Setting **585** We simulate a scenario where training is con- **586** strained by limited data samples due to low com- **587** putational resources, as is typical in edge device **588** training that is aimed at minimizing time and bat- **589** tery consumption. We construct small datasets to **590** evaluate how effectively data can be utilized under **591** conditions of low resource availability. We reduce **592** the amount of training data progressively from 20% **593** to 80%. For the scoring in DMO training, we use **594** the full original dataset as the reference dataset. **595** The results are presented in Table 7. While LM **596** training exhibits a 9% drop in scores as the data **597** size decreases, DMO demonstrates robust learn- **598** ing even with limited data, exhibiting a smaller **599** decrease of 4%. This result demonstrates that our **600** method can efficiently learn even from a small num- **601** ber of data samples by leveraging the importance **602** score of each sample. **603** 

## 6 Conclusion 

**604** 

In this paper, we present Direct Metric Optimiza- **605** tion (DMO), which is a lightweight final-metric- **606** optimizing training method. We hypothesize that **607** diverse text augmentation can substitute the explo- **608** ration in RL, and show that self-supervised train- **609** ing on reward-weighted augmented data leads to **610** direct and stable metric optimization. Our experi- **611** ments demonstrate that DMO can directly optimize **612** evaluation metrics across models of various archi- **613** tectures and parameter sizes, and stably achieves **614** performance comparable to the SOTA RL method **615** while saving hundreds of times more model for- **616** warding iterations and greater amounts of computa- **617** tion time. With these practical advantages of stable **618** and lightweight cost of tuning, DMO emerges as **619** a new promising choice for metric optimization in **620** the era of large-scale VLMs. **621** 

8 

**622** 

## Limitations 

**623** Although our experiments yield promising results, **624** it is important to acknowledge the limitations of our **625** method. The first limitation is the quality subopti- **626** mality of TDA. Our approach substitutes the explo- **627** ration phase in RL with diverse data augmentation. **628** However, in theory, data augmentation is distinct **629** from exploration because it does not actively pur- **630** sue higher rewards. Consequently, over extended **631** training periods, RL methods, which consistently **632** seek new and higher-quality samples, can outper- **633** form our DMO which relies on a fixed dataset. **634** However, considering that RL methods for VLM **635** are often very sensitive to hyperparameters and **636** challenging to optimize, our DMO offers distinct **637** practical advantages such as learning stability and **638** the straightforward training process without the **639** need for intricate hyperparameter tuning—benefits **640** that are absent in most of RL approaches. 

**641** Another limitation is the data augmentation over- **642** head. While DMO avoids the computationally ex- **643** pensive exploration process in RL, data augmenta- **644** tion still necessitates a certain computational cost. **645** Therefore, considering the data preparation phase **646** in addition to the training phase, the computational **647** costs required for DMO increase and DMO’s supe- **648** riority over RL methods in terms of computational **649** costs is diminished. However, a key distinction **650** from exploration is that TDA can be conducted on **651** separate machines (e.g., cloud servers) from the **652** one the target VLM is deployed on. This aspect **653** becomes particularly beneficial for model tuning in **654** scenarios where resources such as time, memory, **655** and battery of devices are constrained, as is typical **656** in edge device training. Collecting augmented data **657** on different servers enables models on the resource- **658** constrained device to bypass the data augmentation **659** overhead, making DMO a genuinely lightweight **660** metric optimization method. 

## References 

**661** 

- **662** Josh Achiam, Steven Adler, Sandhini Agarwal, Lama **663** Ahmad, Ilge Akkaya, Florencia Leoni Aleman, **664** Diogo Almeida, Janko Altenschmidt, Sam Altman, **665** Shyamal Anadkat, et al. 2023. Gpt-4 technical report. **666** arXiv preprint arXiv:2303.08774. 

- **667** Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, **668** Antoine Miech, Iain Barr, Yana Hasson, Karel **669** Lenc, Arthur Mensch, Katherine Millican, Malcolm **670** Reynolds, et al. 2022. Flamingo: a visual language 

|model for few-shot learning. Advances in Neural|**671**|
|---|---|
|Information Processing Systems, 35:23716–23736.|**672**|
|Peter Anderson, Basura Fernando, Mark Johnson, and|**673**|
|Stephen Gould. 2016. Spice: Semantic propositional|**674**|
|image caption evaluation. InECCV.|**675**|
|Viktar Atliha and Dmitrij Šešok. 2020. Text augmen-|**676**|
|tation using bert for image captioning. Applied Sci-|**677**|
|ences, 10(17):5978.|**678**|
|Ashutosh Baheti, Ximing Lu, Faeze Brahman, Ronan Le|**679**|
|Bras, Maarten Sap, and Mark Riedl. 2023. Improving<br>language models with advantage-based offine policy|**680**<br>**681**|
|gradients. arXiv preprint arXiv:2305.14718.|**682**|
|Satanjeev Banerjee and Alon Lavie. 2005. METEOR:|**683**|
|An automatic metric for MT evaluation with im-|**684**|
|proved correlation with human judgments. InPro-|**685**|
|ceedings of the ACL Workshop on Intrinsic and Ex-|**686**|
|trinsic Evaluation Measures for Machine Transla-|**687**|
|tion and/or Summarization, pages 65–72, Ann Arbor,|**688**|
|Michigan. Association for Computational Linguis-|**689**|
|tics.|**690**|
|Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee,|**691**|
|Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind|**692**|
|Srinivas, and Igor Mordatch. 2021. Decision trans-|**693**|
|former: Reinforcement learning via sequence mod-|**694**|
|eling. Advances in neural information processing|**695**|
|systems, 34:15084–15097.|**696**|
|Paul F Christiano, Jan Leike, Tom Brown, Miljan Mar-|**697**|
|tic, Shane Legg, and Dario Amodei. 2017. Deep|**698**|
|reinforcement learning from human preferences. In|**699**|
|Advances in Neural Information Processing Systems,|**700**|
|volume 30. Curran Associates, Inc.|**701**|
|Terrance DeVries and Graham W Taylor. 2017. Im-|**702**|
|proved regularization of convolutional neural net-|**703**|
|works with cutout. arXiv preprint arXiv:1708.04552.|**704**|
|Guiguang Ding, Mingkui Chen, Sicheng Zhao, et al.|**705**|
|2019.<br>Neural image caption generation with|**706**|
|weighted training and reference. Cognitive Com-|**707**|
|putation, 11:763–777.|**708**|
|Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi,|**709**|
|and Yonglong Tian. 2023. Improving clip training|**710**|
|with language rewrites. InNeurIPS.|**711**|
|Steven Y Feng, Varun Gangal, Jason Wei, Sarath Chan-|**712**|
|dar, Soroush Vosoughi, Teruko Mitamura, and Ed-|**713**|
|uard Hovy. 2021. A survey of data augmentation ap-|**714**|
|proaches for nlp. arXiv preprint arXiv:2105.03075.|**715**|
|Junlong Gao, Shiqi Wang, Shanshe Wang, Siwei Ma,|**716**|
|and Wen Gao. 2019. Self-critical n-step training for|**717**|
|image captioning. InProceedings of the IEEE/CVF|**718**|
|Conference on Computer Vision and Pattern Recog-|**719**|
|nition, pages 6300–6308.|**720**|
|Micah Hodosh, Peter Young, and Julia Hockenmaier.|**721**|
|Flickr8k dataset.|**722**|



9 

- **723** Youngsoo Jang, Jongmin Lee, and Kee-Eung Kim. 2022. **724** Gpt-critic: Offline reinforcement learning for end-to- **725** end task-oriented dialogue systems. In 10th Inter- **726** national Conference on Learning Representations, **727** ICLR 2022. International Conference on Learning **728** Representations, ICLR. 

- **729** Marcin Junczys-Dowmunt, Roman Grundkiewicz, **730** Tomasz Dwojak, Hieu Hoang, Kenneth Heafield, **731** Tom Neckermann, Frank Seide, Ulrich Germann, **732** Alham Fikri Aji, Nikolay Bogoychev, André F. T. **733** Martins, and Alexandra Birch. 2018. Marian: Fast **734** neural machine translation in C++. In Proceedings of **735** ACL 2018, System Demonstrations, pages 116–121, **736** Melbourne, Australia. Association for Computational **737** Linguistics. 

- **738** Wooyoung Kang, Jonghwan Mun, Sungjun Lee, and **739** Byungseok Roh. 2023. Noise-aware learning from **740** web-crawled image-text data for image captioning. **741** In Proceedings of the IEEE/CVF International Con- **742** ference on Computer Vision, pages 2942–2952. 

- **743** Andrej Karpathy and Li Fei-Fei. 2015. Deep visual- **744** semantic alignments for generating image descrip- **745** tions. In Proceedings of the IEEE conference on **746** computer vision and pattern recognition, pages 3128– **747** 3137. 

- **748** Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin John- **749** son, Kenji Hata, Joshua Kravitz, Stephanie Chen, **750** Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. **751** 2017. Visual genome: Connecting language and vi- **752** sion using crowdsourced dense image annotations. **753** International journal of computer vision, 123:32–73. 

- **754** Sergey Levine, Aviral Kumar, George Tucker, and Justin **755** Fu. 2020. Offline reinforcement learning: Tutorial, **756** review, and perspectives on open problems. arXiv **757** preprint arXiv:2005.01643. 

- **758** Bohan Li, Yutai Hou, and Wanxiang Che. 2022. Data **759** augmentation approaches in natural language pro- **760** cessing: A survey. AI Open, 3:71–90. 

- **761** Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. **762** 2023. Blip-2: Bootstrapping language-image pre- **763** training with frozen image encoders and large lan- **764** guage models. arXiv preprint arXiv:2301.12597. 

- **765** Chin-Yew Lin. 2004. ROUGE: A package for auto- **766** matic evaluation of summaries. In Text Summariza- **767** tion Branches Out, pages 74–81, Barcelona, Spain. **768** Association for Computational Linguistics. 

- **769** Tsung-Yi Lin, Michael Maire, Serge J. Belongie, **770** Lubomir D. Bourdev, Ross B. Girshick, James Hays, **771** Pietro Perona, Deva Ramanan, Piotr Doll’a r, and **772** C. Lawrence Zitnick. 2014. Microsoft COCO: com- **773** mon objects in context. CoRR, abs/1405.0312. 

- **774** Siqi Liu, Zhenhai Zhu, Ning Ye, Sergio Guadarrama, **775** and Kevin Murphy. 2017. Improved image caption- **776** ing via policy gradient optimization of spider. In **777** Proceedings of the IEEE international conference on **778** computer vision, pages 873–881. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, **779** Carroll Wainwright, Pamela Mishkin, Chong Zhang, **780** Sandhini Agarwal, Katarina Slama, Alex Ray, et al. **781** 2022. Training language models to follow instruc- **782** tions with human feedback. Advances in Neural **783** Information Processing Systems, 35:27730–27744. **784** 

- Kishore Papineni, Salim Roukos, Todd Ward, and Wei- **785** Jing Zhu. 2002. Bleu: a method for automatic evalu- **786** ation of machine translation. In Proceedings of the **787** 40th annual meeting of the Association for Computa- **788** tional Linguistics, pages 311–318. **789** 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya **790** Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sas- **791** try, Amanda Askell, Pamela Mishkin, Jack Clark, **792** et al. 2021. Learning transferable visual models from **793** natural language supervision. In International confer- **794** ence on machine learning, pages 8748–8763. PMLR. **795** 

|Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli,|**796**|
|---|---|
|and Wojciech Zaremba. 2015. Sequence level train-|**797**|
|ing with recurrent neural networks. arXiv preprint|**798**|
|arXiv:1511.06732.|**799**|
|Steven J Rennie, Etienne Marcheret, Youssef Mroueh,|**800**|
|Jerret Ross, and Vaibhava Goel. 2017. Self-critical|**801**|
|sequence training for image captioning. InProceed-|**802**|
|ings of the IEEE conference on computer vision and|**803**|
|pattern recognition, pages 7008–7024.|**804**|
|Piyush Sharma, Nan Ding, Sebastian Goodman, and|**805**|
|Radu Soricut. 2018. Conceptual captions: A cleaned,|**806**|
|hypernymed, image alt-text dataset for automatic im-|**807**|
|age captioning. InProceedings of the 56th Annual|**808**|
|Meeting of the Association for Computational Lin-|**809**|
|guistics (Volume 1: Long Papers), pages 2556–2565,|**810**|
|Melbourne, Australia. Association for Computational|**811**|
|Linguistics.|**812**|
|Ruizhe Shi, Yuyao Liu, Yanjie Ze, Simon S Du, and|**813**|
|Huazhe Xu. 2023.<br>Unleashing the power of pre-<br>trained language models for offine reinforcement|**814**<br>**815**|
|learning. arXiv preprint arXiv:2310.20587.|**816**|
|Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu,|**817**|
|Chunyuan Li, Yikang Shen, Chuang Gan, Liang-|**818**|
|Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. 2023.|**819**|
|Aligning large multimodal models with factually aug-|**820**|
|mented rlhf. arXiv preprint arXiv:2309.14525.|**821**|
|Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe,|**822**|
|Jon Shlens, and Zbigniew Wojna. 2016. Rethinking|**823**|
|the inception architecture for computer vision. In|**824**|
|Proceedings of the IEEE conference on computer|**825**|
|vision and pattern recognition, pages 2818–2826.|**826**|
|Gemini Team, Rohan Anil, Sebastian Borgeaud,|**827**|
|Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu,|**828**|
|Radu Soricut, Johan Schalkwyk, Andrew M Dai,|**829**|
|Anja Hauth, et al. 2023.<br>Gemini: a family of|**830**|
|highly capable multimodal models. arXiv preprint|**831**|
|arXiv:2312.11805.|**832**|



10 

- **833** Hugo Touvron, Louis Martin, Kevin Stone, Peter Al- **834** bert, Amjad Almahairi, Yasmine Babaei, Nikolay **835** Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti **836** Bhosale, et al. 2023. Llama 2: Open founda- **837** tion and fine-tuned chat models. arXiv preprint **838** arXiv:2307.09288. 

- **839** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob **840** Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz **841** Kaiser, and Illia Polosukhin. 2017. Attention is all **842** you need. Advances in neural information processing **843** systems, 30. 

- **844** Ramakrishna Vedantam, C Lawrence Zitnick, and Devi **845** Parikh. 2015. Cider: Consensus-based image de- **846** scription evaluation. In Proceedings of the IEEE **847** conference on computer vision and pattern recogni- **848** tion, pages 4566–4575. 

- **849** Oriol Vinyals, Alexander Toshev, Samy Bengio, and **850** Dumitru Erhan. 2015. Show and tell: A neural image **851** caption generator. In Proceedings of the IEEE con- **852** ference on computer vision and pattern recognition, **853** pages 3156–3164. 

- **854** Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie **855** Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and **856** Lijuan Wang. 2022. Git: A generative image-to-text **857** transformer for vision and language. arXiv preprint **858** arXiv:2205.14100. 

- **859** Ronald J Williams and David Zipser. 1989. A learn- **860** ing algorithm for continually running fully recurrent **861** neural networks. Neural computation, 1(2):270–280. 

- **862** Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, **863** Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei **864** Wang, et al. 2023. mplug-2: A modularized multi- **865** modal foundation model across text, image and video. **866** arXiv preprint arXiv:2302.00402. 

- **867** Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, **868** Aaron Courville, Ruslan Salakhudinov, Rich Zemel, **869** and Yoshua Bengio. 2015. Show, attend and tell: **870** Neural image caption generation with visual atten- **871** tion. In International conference on machine learn- **872** ing, pages 2048–2057. PMLR. 

- **873** Kaicheng Yang, Jiankang Deng, Xiang An, Jiawei Li, **874** Ziyong Feng, Jia Guo, Jing Yang, and Tongliang **875** Liu. 2023. Alip: Adaptive language-image pre- **876** training with synthetic caption. In Proceedings of the **877** IEEE/CVF International Conference on Computer **878** Vision, pages 2922–2931. 

- **879** Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Ye- **880** ung, Mojtaba Seyedhosseini, and Yonghui Wu. 2022. **881** Coca: Contrastive captioners are image-text founda- **882** tion models. arXiv preprint arXiv:2205.01917. 

- **883** Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and **884** David Lopez-Paz. 2017a. mixup: Beyond empirical **885** risk minimization. arXiv preprint arXiv:1710.09412. 

- **886** Le Zhang, Yanshuo Zhang, Xin Zhao, and Zexiao Zou. **887** 2021. Image captioning via proximal policy opti- **888** mization. Image and Vision Computing, 108:104126. 

- Li Zhang, Flood Sung, Feng Liu, Tao Xiang, Shaogang **889** Gong, Yongxin Yang, and Timothy M Hospedales. **890** 2017b. Actor-critic sequence training for image cap- **891** tioning. arXiv preprint arXiv:1706.09601. **892** 

- Zhiyuan Zhao, Bin Wang, Linke Ouyang, Xiaoyi Dong, **893** Jiaqi Wang, and Conghui He. 2023. Beyond hallu- **894** cinations: Enhancing lvlms through hallucination- **895** aware direct preference optimization. arXiv preprint **896** arXiv:2311.16839. **897** 

## A Experiments Setting in Detail 

**898** 

## A.1 Hardware Environment 

**899** We use two RTX A6000 Ada GPUs for GIT- **900** base/large and four H100 GPUs for BLIP2-2.7B. **901** Because each GPU has a different size of VRAM, **902** we adopt the gradient accumulation method so that **903** the batch size becomes 960 regardless of the size **904** of the VRAM of the GPU that is used in each **905** experiment. We present the hyperparameters for **906** GIT-base/large and BLIP2-2.7B in Table 8. This **907** configuration explains the number of model iter- **908** ations in LM/DMO training presented in Table 5. **909** The training uses 6091 images and 11 captions **910** per image (including augmented captions) across **911** 3 epochs. Dividing this by a mini-batch size of 60 **912** and 2 GPUs results in approximately 1.68 × 10[3] **913** iterations. For SCST, the maximum token length **914** is set to 128 and the model recursively generates **915** tokens up to this length, resulting in nearly 100 **916** times more model forwarding iterations compared **917** to LM/DMO training. **918** A.2 Configuration for SCST Training **919** Due to the instability of training with SCST, it **920** necessitates pre-fine-tuning through standard self- **921** supervised training (Rennie et al., 2017). Therefore, **922** we initially fine-tune the model for three epochs **923** with a learning rate of 1.0 × 10[−][5] before applying **924** SCST. Given that fine-tuning has already been com- **925** pleted, we reduce the learning rate to 5.0 × 10[−][6] **926** and only update the parameters of the text decoder **927** to stabilize training. Moreover, during the sam- **928** pling process in SCST, we opt for a temperature of **929** 0.1. This is because we observe that higher temper- **930** atures, such as 0.5 or 1.0, often lead the model to **931** generate random, meaningless sequences of words, **932** which ultimately results in model collapse. **933** 

## B Prompt for Llama2 Paraphrasing 

**934** For the Llama2-paraphrasing method, we em- **935** ploy the same prompting method proposed in La- **936** CLIP (Fan et al., 2023). We present the prompt that **937** 

11 

**938** is used in our experiments in Figure 3. The part **939** ’<target captions>’ in the prompt text is replaced by **940** the caption to be paraphrased. In the prompt, three **941** examples of paraphrasing are provided. The first **942** and second examples are constructed by regard- **943** ing the two captions for the same image in each **944** Flickr8K and COCO dataset as the captions be- **945** fore and after paraphrasing. The third paraphrasing **946** example is made by feeding ChatGPT4 a caption **947** from Flickr8k with the prompt "rewrite this image **948** caption". 

**949 950** 

**951 952 953 954 955 956 957** 

|Model|LearningRate<br>Batch Size<br>Mini-Batch Size<br>Grad. Acc. Step<br>GPU|
|---|---|
|GIT-base<br>GIT-large<br>BLIP2-2.7B|1.0×10−5<br>960<br>60<br>8<br>RTX A6000 Ada×2<br>1.0×10−5<br>960<br>30<br>16<br>RTX A6000 Ada×2<br>1.0×10−5<br>960<br>30<br>8<br>H100×4|



Table 8: Hyperparameters for GIT-base/large and BLIP2-2.7B. The number of batch sizes is equal to the product of the mini-batch size, the number of gradient accumulation steps (Grad. Acc. Step) and the number of GPUs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0012-05.png)


Figure 3: The prompt that is used in Llama2paraphrasing. The part ’<target caption>’ in the prompt text is replaced by the caption to be augmented. 

## C Distributions Difference by TDA Method 

In this section, we explore how the quality of samples varies across different TDA methods. We present the distribution of scores of samples generated by each TDA method in Figure 4. Scores are CIDEr scores based on the training dataset of Flickr8k. The back-translation tends to yield higher scores while BLIP2-sampling tends to pro- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0012-09.png)


**----- Start of picture text -----**<br>
400 blip2<br>llama<br>350 back-translation<br>300<br>250<br>200<br>150<br>100<br>50<br>0<br>0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00<br>CIDEr Score<br>Number of samples<br>**----- End of picture text -----**<br>


Figure 4: CIDEr score distributions of captions augmented by back-translation, BLIP2-sampling, and Llama2-paraphrasing. 

duce samples of lower scores. Examples of gener- **958** ated captions by each TDA method are shown in **959** Figure 5. We find that captions generated by the **960** back-translation show little change compared to **961** the ground truth (GT) captions. On the other hand, **962** BLIP2 sampling generates captions that are signifi- **963** cantly different from GT in terms of style and level **964** of detail. Back-translation receives the GT captions **965** to augment captions. Thus captions generated by **966** back-translation closely resemble GT captions. On **967** the other hand, BLIP2-sampling generates captions **968** solely from images. Therefore, captions generated **969** by BLIP2-sampling often deviate from the GT cap- **970** tions and sometimes include incorrect descriptions **971** (e.g., BLIP2 misidentified the person in the image **972** as "woman" in Figure 5). Moreover, because the **973** paraphrasing by Llama2 takes the GT captions as a **974** prompt, the generated captions by Llama2 are se- **975** mantically close to the GT captions. However, the **976** changes from GT captions are greater than those **977** of captions augmented by back-translation, owing **978** to the prompt which encourages Llama2 to change **979** expressions (e.g., ’surf’ is paraphrased as ’ride a **980** wave’ in Figure 5). **981** Interestingly, considering the result that DMO **982** training on a dataset solely augmented by back- **983** 

12 

**984** translation does not improve scores (shown in Ta- **985** ble 3), TDA that often produces samples of high **986** scores may not provide an advantage for DMO if **987** it hardly alters the expression of the original GT **988** data. Rather, TDA methods that produce samples **989** of diverse expressions and structures can improve **990** the performance of DMO by introducing various in- **991** formation that is not present in the original dataset. **992** With the analysis above, we emphasize that the di- **993** versity of samples is important for DMO training **994** and especially diversifying the augmentation tech- **995** niques themselves is an effective approach because **996** utilizing multiple augmentation methods generates **997** a diverse set of samples across various distribu- **998** tions. 

**999 1000** 

**1001 1002 1003 1004 1005 1006 1007 1008 1009 1010** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0013-03.png)


Figure 5: Examples of captions generated by each augmentation method, back-translation, BLIP2-sampling, and Llama2-paraphrasing. 

## D Examples of Captions Generated by DMO, LM Training and SCST 

Figure 6 shows examples of images and corresponding ground truth captions and captions generated by a GIT-base model trained by DMO, standard LM training and SCST. Models are trained for three epochs on the textually augmented Flicker8k dataset. Expressions which seem to be unique to the image are underlined. Those examples show that the model trained by DMO captures distinctive information and objects within the image and depicts them in the generated captions. 

13 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization_images/acl_2024_Direct_Metric_Optimization_for_Image_Captioning_through_Reward-Weighted_Augmented_Data_Utilization.pdf-0014-00.png)


Figure 6: Examples of ground truth captions and captions generated by GIT-base model trained by DMO, standard LM training and SCST. Each model is trained on the textually augmented Flickr8k dataset for 3 epochs. Expressions which seem to be unique to the image are underlined. 

14 

