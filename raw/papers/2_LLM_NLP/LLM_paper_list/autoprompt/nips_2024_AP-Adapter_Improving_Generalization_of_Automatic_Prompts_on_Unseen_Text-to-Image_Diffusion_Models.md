# **AP-Adapter: Improving Generalization of Automatic Prompts on Unseen Text-to-Image Diffusion Models** 

**Yuchen Fu Zhiwei Jiang** _[∗]_ **Yuliang Liu Cong Wang Zexuan Deng Zhaoling Chen Qing Gu** State Key Laboratory for Novel Software Technology, Nanjing University, China `yuchenfu@smail.nju.edu.cn, jzw@nju.edu.cn {yuliangliu,cw,dengzx,zhaolingchen}@smail.nju.edu.cn guq@nju.edu.cn` 

## **Abstract** 

Recent advancements in Automatic Prompt Optimization (APO) for text-to-image generation have streamlined user input while ensuring high-quality image output. However, most APO methods are trained assuming a fixed text-to-image model, which is impractical given the emergence of new models. To address this, we propose a novel task, model-generalized automatic prompt optimization (MGAPO), which trains APO methods on a set of known models to enable generalization to unseen models during testing. MGAPO presents significant challenges. First, we experimentally confirm the suboptimal performance of existing APO methods on unseen models. We then introduce a two-stage prompt optimization method, AP-Adapter. In the first stage, a large language model is used to rewrite the prompts. In the second stage, we propose a novel method to construct an enhanced representation space by leveraging inter-model differences. This space captures the characteristics of multiple domain models, storing them as domain prototypes. These prototypes serve as anchors to adjust prompt representations, enabling generalization to unseen models. The optimized prompt representations are subsequently used to generate conditional representations for controllable image generation. We curate a multi-modal, multi-model dataset that includes multiple diffusion models and their corresponding text-image data, and conduct experiments under a model generalization setting. The experimental results demonstrate the AP-Adapter’s ability to enable the automatic prompts to generalize well to previously unseen diffusion models, generating high-quality images. 

## **1 Introduction** 

Automatic Prompt Optimization (APO) in text-to-image generation aims to simplify the user input process while ensuring that the generated images exhibit both semantic consistency and aesthetic quality [9, 22, 24]. APO addresses the limitation of current text-to-image models, which often heavily rely on manually crafted prompts for generating high-quality images. The direct use of natural language descriptions as inputs often results in suboptimal outcomes. 

APO methods offer effective ways to directly utilize natural language descriptions for text-to-image diffusion models, which can be categorized into two paradigms: Text Optimization and Representation Optimization. Text Optimization entails generating refined prompts directly by fine-tuning the language model [8, 26]. On the other hand, Representation Optimization involves refining prompts within the feature space of the text encoder of the diffusion model [37]. 

> _∗_ Corresponding author 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Deliberate2 Dreamshaper6 Ghostmix12 RevAnimated122 Chilloutmix Natural  LLM Seen Models Unseen Model<br>Language<br>LanguageNaturalPrompt Prompt EncoderEncoderTextEncoderTextText Language Natural Prompt EncoderTextLLM<br>AdapterAdapterAdapter Adapter Low Quality<br>Magic<br>Prompt DenoiserDenoiserDenoiser Denoiser<br>Prompt (i) Model-Specific Adapter for Prompt Representation<br>Perfect<br>Natural  LLM Seen Models Unseen Model<br>adapterSUR- Language Prompt EncoderEncoderTextEncoderTextText Language Natural Prompt EncoderTextLLM<br>Adapter Adapter High Quality<br>Ours DenoiserDenoiserDenoiser Denoiser<br>Two cats fighting against each other, with one cat being orange and the other being grey.<br>The scene is set against a backdrop of a cloudy sky, giving the image a sense of depth and<br>atmosphere. The style of the image is a detailed, realistic drawing.  (ii) Model-Generalized Adapter for Prompt Representation<br>(a) (b)<br>**----- End of picture text -----**<br>


Figure 1: (a) Comparison of text-to-image generation among different APO methods on various unseen fine-tuned diffusion models. (b) Comparison between model-specific and model-generalized adapters under the model-generalized setting. 

As the advancements like LoRA [13] and DreamBooth [27] have simplified the training of diffusion models with personalized data, an increased number of diffusion models have emerged. The emergence of new models constantly requires us to conduct APO for new models and fresh prompt data. This process is overly labor-intensive and time-consuming. However, directly applying APO method specific to seen models to unseen models may result in poor generalization. As shown in the Figure 1a, Text optimization methods like MagicPrompt [7] and PromptPerfect [16] enhance image quality but sacrifice semantic consistency. Representation optimization methods, exemplified by SUR-adapter [37] maintain semantic consistency but show limited improvement in image quality. In this work, we address the model generalization scenario of Automatic Prompt Optimization (APO) and introduce the novel task of model-generalized automatic prompt optimization (MGAPO). The objective of MGAPO is to train an APO model on a limited set of known diffusion models and enable automatic prompt generalization for unseen diffusion models. MGAPO presents several challenges. First, implementing MGAPO requires incorporating extensive model information, which is absent in existing synthetic image datasets [37, 31]. Second, MGAPO must ensure both the semantic consistency and aesthetic quality of the generated images. Third, to achieve good generalization on unseen diffusion models, the model-related knowledge must be appropriately addressed during training, either retained or discarded. 

To train an APO model with generalization capabilities, we frame the task of MGAPO as a variant of domain generalization. We initially collected and annotated a multi-modal multi-domain dataset for training and testing. The dataset comprises natural language prompts, manually crafted prompts, high-quality images, and multiple stable diffusion (SD) checkpoints, serving as domain information. We then propose the Automatic Prompt Adapter (AP-Adapter), which is a two-stage framework. In the first stage, we leverage the ICL capability of large language models (LLMs) [23, 30] to generate automatic keyword prompts similar to manually crafted prompts [1, 34]. These prompts are then fed into the diffusion model’s text encoder. In the second stage, we achieve model generalization using an adapter-based approach. The distinction between model-specific [37] and model-generalized adapters is illustrated in Figure 1b. We decode domain information from images [4, 35] and distill this information into domain prototypes. These prototypes serve as anchors to adjust the prompt representation, providing rich domain information for prompt adaptation. By further aligning with the representation specific to manual prompts, the adapted representation can simultaneously possess the ability for both semantically consistent and aesthetically pleasing generation. 

Our main contributions are as follows: 

- We explore model-generalized automatic prompt optimization (MGAPO), targeting the effectiveness of automatic prompts on unseen models, addressing a challenging aspect of domain generalization. 

2 

- We propose AP-Adapter, the first method to address MGAPO, leveraging in-context learning of large language model for text optimization in the first stage and constructing an enhanced representation space by leveraging inter-model differences in the second stage. 

- We collect and annotate a multi-modal, multi-domain dataset for training and evaluation, including high-quality images, manually crafted prompts, natural language prompts, and model information. 

- Our extensive experiments on this dataset demonstrate Adapter’s ability to enable the automatic prompts to generalize well to previously unseen diffusion models. 

## **2 Related Work** 

**Automatic Prompt Optimization for Text-to-Image Models.** Prompt Optimization is essential for accurate semantic comprehension and high-quality image generation. Hao et al. [8] proposed a two-stage training approach. Initially, they supervised fine-tuned a language model using manual prompts. In the second stage, reinforcement learning, such as Proximal Policy Optimization (PPO), was employed to maximize the aesthetic and relevance scores of generated images. Rosenman [26] introduced the Neurologic Decoding method to generate more diverse and personalized prompts. Zhong et al. [37] improved image quality by aligning natural language descriptions with manually designed prompts in the feature space of CLIP’s text encoder through the optimization of a lowparameter adapter. However, challenges in adapting pre-trained adapters to out-of-domain data and overlooking diversity across Stable Diffusion (SD) models persist. Noteworthy products, including Promptperfect and MagicPrompt, offer similar functionality, but our usage reveals limitations in their effective generalization across diverse SD models. 

**Domain Generalization.** Domain generalization aims to distill domain-invariant knowledge from a source domain, facilitating seamless model generalization to unknown domains. Advanced strategies leverage diversity within known domains to construct an enhanced feature space, aiding representation of unknown data using knowledge gained from known domains. DGSS [15] preserves style features from known domains as bases for generating new stylized images, promoting generalization to unknown domains. StyleAdv [5] introduces adversarial perturbations to image styles, enhancing model robustness to unknown image styles through adversarial training. Other works, such as [29, 3, 19], fully exploit the adapter module for domain generalization on pre-trained models. 

## **3 Method** 

## **3.1 Task Definition** 

Similar to domain generalization, MGAPO aims to achieve generalization to unseen models in the target domain by training the APO model on K source domain models. Formally, we introduce notations and formalize the task as follows. Given a diffusion model (i.e., Stable Diffusion 1.5) _f_ ( _x_ ; Θ) with _K_ source checkpoints _S_ = _{_ Θ1 _,_ Θ2 _, ...,_ Θ _K}_ , the _k_ -th source checkpoint contains _Nk_ image-text pairs ( _n[k] i[, m][k] i[, I] i[k]_[)] _[N] i_ =1 _[k]_[, where] _[ n] i[k]_[represents natural language descriptions,] _[ m][k] i_[denotes] manually designed prompts, and _Ii[k]_[is][the][corresponding][image.][Each][checkpoint][has][the][same] architecture as _f_ ( _x_ ; Θ) but with different parameters. Therefore each source checkpoint with its text-image pairs can be viewed as a source domain. In the testing phase, unseen target checkpoints _T_ = _{_ Θ _K_ +1 _,_ Θ _K_ +2 _, ...,_ Θ _K_ + _T }_ with natural language descriptions ( _n[t] i_[)] _[N] i_ =1 _[t]_[are][provided.][Each] target checkpoint can be viewed as a target domain. Our objective is to leverage source-domain checkpoints to create a pipeline capable of generalizing to any unseen target-domain checkpoints, generating coherent and aesthetically pleasing images for any given natural language descriptions. 

## **3.2 Overview** 

As illustrated in Figure 2, our method comprises two stages of optimization for text prompts, corresponding to text-based optimization and representation-based optimization methods, respectively. 

In the first stage, we leverage the zero-shot reasoning capability of large language models _fllm_ by employing in-context learning [21] to guide the generation of reliable prompts. Examples for in-context learning are obtained through similarity retrieval. In this step, natural language prompts 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Natural Language Prompt Dataset Retrieve Similar Pairs CLIP Feature Extraction 𝓛𝒄<br>You are a prompt engineer who is good at writing various prompts for language models. Stable Diffusion is an text-to-image generation model.Please translate the natural language description of an image below into prompts for generating image with Stable Diffusion. Ensure preserve the text of image content and style. Please refer to the examples I gave.<example> PrototypeAlignment 𝓛 Dropped Image Features  𝒂 (The  k-th  Domain) 𝑭𝒅𝒓𝒐𝒑𝒌 DiscardingChannel PredictionDomain<br></example> Domain<br>Here are the natural language description needed to be translated: Automated Keyword PromptNatural Language Prompt PromptEmbedding 𝑃!"#$𝑃%"#$ 𝑷𝒅𝒐𝒎𝒌 … Prototype-Anchored 𝑃*"#$ Prototypes & Linear LayersTransformer EmbeddingAdapted<br>Embedding<br>In-Context Learning Concatenation Adapter<br>Prompt Adapted<br>Embedding Embedding<br>Natural Language Prompt LLM Keyword PromptAutomatic Adapter<br>Manual  of the k-th domainText Encoder 𝓛𝒓 +<br>Keyword Prompt Denoiser<br>Feature Extraction of the k-th domain<br>Manual-Specific<br>Training Sample Prompt Embedding<br>of the k-th domain 𝓛𝒅<br>Text<br>Encoder<br>Text<br>Encoder<br>**----- End of picture text -----**<br>


Figure 2: Illustration of our proposed AP-Adapter. 

are automatically translated into keyword prompts, which are then fed into the text encoder of the SD model _fclip[T]_[to obtain text representations.] 

In the second stage, we improve automatic keyword prompts by decoding domain-specific content from images and aligning it with manual prompts. To handle image unavailability during inference, we maintain prototypes in training to preserve domain-specific details. The adapter module then integrates information from text and prototypes, aligning its output representation with manual prompts in the feature space. 

Specifically, we freeze the parameters of the text encoder _fclip[T]_[and denoiser] _[ f][den]_[in diffusion model.] During training, for a specific domain’s image-text pair ( _n[k] i[, m][k] i[, I] i[k]_[)][,][we load the corresponding] diffusion model parameters _f_ ( _x_ ; Θ _k_ ) to extract features and denoise. The CLIP image encoder _fclip[I]_ is introduced to extract image features, with the image encoder parameters also frozen. 

## **3.3 ICL-Based Prompt Rewriting** 

Previous work [21] has proved the demonstrations that are semantically similar to the test example exhibit higher performance in context learning. We employ a pre-trained sentence transformer [25] to calculate the cosine similarity between input and natural prompts in dataset. For each input natural prompts, we select the top _k_ most similar demonstrations from dataset. These pairs of natural language prompts and manually crafted prompts, along with the input natural prompts, are embedded into the prompt template (depicted on the left side of Figure 2). The prompt template is then feed into GPT3.5 to obtain preliminary optimized keyword prompt. Compared to natural prompts, keyword prompts are closer in the semantic space of CLIP to manually crafted prompts. However, they are still insufficient for achieving multi-model generalization. Therefore, we introduce a prototype-based adaptive approach for automatic keyword prompts. 

## **3.4 Prototype-Based Prompt Adaptation** 

In this section, we introduce how the adapter module leverages prototype learning to obtain model generalization capabilities. 

**Alignment of Domain Prototypes.** The parameter difference of models is the main factor leading to the domain shift, which limits the generalization ability of the optimized prompts. Parameter differences are encoded into the image during the process of generating images from text, resulting in the images exhibiting different domain variations. Therefore, we propose domain prototypes, which align with the domain-specific information decoded from the image. 

4 

In detail, we employ intermediate features _Fl[k] ∈_ R _[C][×][H][×][W]_ with the _l_ -th CLIP image encoder layer _fclip[I]_[from image] _[ I] i[k]_[.][Here,] _[ C]_[is the number of feature channels, while] _[ H]_[and] _[ W]_[denote the] feature map’s dimensions. Domain prototypes, denoted as _{Pk[dom] ∈_ R _[N][p][×][D] }[K] k_ =1[,][aim][to][align] with the domain-specific information from image features to enhance automatic keyword prompt representations. _Np_ denotes the number of prototypes, and _D_ is the text token dimension. 

Identifying domain-specific information in image features is challenging due to the higher dimensionality of image features _C_ compared to text features _D_ . Inspired by DomainDrop [6], we selectively drop non-informative channels in the feature space while retaining domain-specific information. 

To drop domain-insensitive feature channels, we use a domain discriminator _fd_ ( _·_ ) that predicts the image’s domain based on intermediate layer features. This discriminator, comprising a global average pooling (GAP) layer and a fully-connected (FC) layer, outputs the domain label _y_ ˆ and computes cross-entropy loss for optimization: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0005-03.png)


where _K_ is the number of domains. Weights in the FC layer quantify each channel’s contribution to domain discrimination. For the c-th channel in the feature map, its contribution is calculated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0005-05.png)


where _W[y]_[ˆ] _∈_ R _[C]_ is the weight in the fully connected layer when predicting the domain label _y_ ˆ correctly. A higher score indicates a greater contribution of the channel to domain discrimination. Subsequently, we can generate a binary mask _M_ through the weighted random selection algorithm [12]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0005-07.png)


where _vc_ = _rc_[1] _[/score][c]_ is a computed by a random number _rc ∈_ (0 _,_ 1), Top( _D_ ) denotes the represents the top _D_ items sorted by the values of _vc_ , _C_ and _D_ are the dimensions of image features and text features, respectively. 

By dropping the domain-insensitive channels of the image feature channels, we enhance the representation of domain information in the features and ensure consistency in the number of feature channels between images and text. To prevent drastic changes in image domain during the training process, we employ momentum updates for smoothing: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0005-10.png)


_′_ where _α_ is a hyperparameter used to adjust the smoothness, _Fk[drop]_ and _Fk[drop]_ are the image features after channel dropout and the features at the next step, respectively. 

We initialize the domain prototypes with standard normal distribution and then align the domain prototypes with the dropped image features: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0005-13.png)


where KL( _·, ·_ ) means KL divergence. 

**Adaptation of Prompt Representation.** We leverage all learned domain prototypes to achieve generalization to unseen models. We use the wasserstein distance dist( _·, ·_ ) to measure the distance between prompt embeddings _fclip[T]_[(] _[f][llm]_[(] _[n] i[k]_[))] _[∈]_[R] _[L][×][D]_[,][where] _[L]_[ is the prompt token length,][and] domain prototypes _Ps[dom]_ across all source domains: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0005-16.png)


where _dk_ represents the distance between the _k_ -th domain prompt and the text token embedding. Then, we calculate the reciprocal of _dk_ to determine the weight corresponding to each domain prompt: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0005-18.png)


5 

where the softmax operation ensures that the sum of all weights is 1. We performed a weighted sum of all domain prototypes across domains to obtain the final Prototype-Anchored Embedding: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0006-01.png)


Token embeddings of keyword prompts and mixed domain prototypes are concatenated for further training. We use aligned domain prototypes as anchors to adjust the representation of prompts, obtaining generalization capabilities on unseen domains. 

**Alignment of Adapted Representation.** We introduce a simple yet effective adapter _gada_ to aggregate and consolidate relevant information from all tokens, generating a conditional encoding representation for text-to-image generation. Comprising two transformer encoder layers and one linear layer, the adapter’s linear layer parameters are initialized to 0 for stable training, following a proven approach [36]. 

The adapter’s output is denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0006-05.png)


Our goal is to make the representation of _qi[s]_[as similar as possible to the representation of manually] designed prompts. Therefore, we use mean squared error loss to align the representation of natural prompts and manual prompts. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0006-07.png)


## **3.5 Model Training** 

**Image Denoising.** During training phase, we need to ensure that the learned parameters do not compromise the denoising model’s generation. We adopt the denoising loss from the diffusion model [28, 11] to guarantee this aspect. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0006-10.png)


where _ϵθ_ represents the denoising model, _ϵ_ represents the noise added to the input image _x_ 0, _t_ represents the time step, and _αtx_ 0 + _βtϵ_ is the result of adding noise to the image at time step _t_ . 

**Alternating Training.** To optimize training efficiency, we employ two strategies. Firstly, considering the variation in diffusion model parameters during training, we streamline the process by sequentially training on all data from each domain within a single epoch. This minimizes the time spent on model switching. Secondly, to prevent interference between the learning of style prompts and adapters, we adopt an alternating training approach. Specifically, after completing an iteration across all data from each domain, we freeze either the adapter or domain prototypes parameters for the subsequent training epoch, repeating this cycle. Therefore, the loss used for training the domain prototypes is as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0006-13.png)


while the loss used for training the adapter is denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0006-15.png)


where _γ_ 1 _, γ_ 2 _, γ_ 3 _, γ_ 4 are the loss coefficients, ranging from [0, 1]. 

Our training pipeline is outlined in Appendix A. During the inference phase, we leverage the acquired domain prototypes and adapter for accurate predictions. 

## **4 Experiments** 

## **4.1 Dataset** 

In this section, we introduce the process of collecting and creating the multi-modal multi-domain dataset for MGAPO. 

6 

Table 1: Evaluation of the generated images on the target domain data using diverse prompt optimization methods. AKP denotes the automatic keyword prompt generated by ICL-based prompt rewriting. 

|**Methods**|**Semantic Consistency**<br>**Color**<br>**Shape**<br>**Texture**<br>**Blipscore**|**Image Quality**<br>**Aesthetic Score**<br>**ImageReward**<br>**HPS**|
|---|---|---|
|**MagicPrompt**<br>**PromptPerfect**<br>**Promptist**<br>**SUR-adapter**<br>**AKP**<br>**AKP + SUR-adapter**<br>**Ours**|0.438<br>0.395<br>0.432<br>0.297<br>0.433<br>0.401<br>0.425<br>0.302<br>0.439<br>0.398<br>0.427<br>0.292<br>0.472<br>0.413<br>0.449<br>0.325<br>0.456<br>0.401<br>0.437<br>0.305<br>0.442<br>0.407<br>0.441<br>0.315<br>**0.477**<br>**0.422**<br>**0.452**<br>**0.332**|6.154<br>0.066<br>0.207<br>6.249<br>0.124<br>0.211<br>6.000<br>0.089<br>0.202<br>6.009<br>0.286<br>0.198<br>6.113<br>0.253<br>0.213<br>6.158<br>0.233<br>0.210<br>**6.384**<br>**0.427**<br>**0.218**|
|**Manual Prompts (GT)**|/<br>/<br>/<br>0.400|6.564<br>0.782<br>0.223|



**Data Collection.** We sourced high-quality images and personalized SD checkponts from the CIVITAI community[2] . We collected 47,695 image-text pairs gathered from various checkpoints, ensuring privacy protection. Further analysis of our dataset is provided in the Appendix B.1. 

**Data Cleaning.** To ensure quality and semantic consistency, we employed Clipscore [10] and the aesthetic predictor[3] . Qualifying samples, totaling 25,395 image-text pairs, were selected based on aesthetic and clip scores surpassing their respective 25th percentiles. 

We enriched the dataset using LLaVA 1.5 [20], a multimodal language model, to generate detailed natural language descriptions. We utilized instructions from [2], guiding the generation with, "Describe this image and its style in a very detailed manner." Then, we selected the top 3 generated sentences with the highest relevance to the image content as natural language prompts. 

Our dataset contains a total of over 5000 checkpoints obtained through fine-tuning based on SD 1.5. However, most checkpoints have fewer than 10 corresponding image-text pairs. To ensure fairness in learning, we sorted the checkpoints based on the number of corresponding image-text pairs and selected the top 40 checkpoints as the source domain data and checkpoints ranked 41 to 100 as the target domain data. The source domain encompasses 7075 samples, whereas the target domain comprises 3064 samples. The remaining data is used as demonstrations for similarity retrieval in ICL-based prompt rewriting. 

## **4.2 Implementation Details** 

During the training phase, we retrieve 5 pairs of natural language prompts and manually designed prompts as demonstrations for ICL from the dataset. We instruct GPT3.5 to generate 5 keyword prompts based on the input natural language example. By combining these 5 keyword prompts with their corresponding manual prompts and images, we augment the data used to train the modelgeneralized adapter. 

For the model’s parameter settings, since the source domain data contains 40 checkpoints, the number of domain prototypes _S_ is set to 40. The coefficients _γ_ 1 _, γ_ 2 _, γ_ 3 _, γ_ 4 for the loss functions are 0.01, 1.0, 0.001 and 1.0, respectively. Figure 9 shows the accuracy of the domain discrimination using the image features output from various layers of the CLIP image encoder. We use the image features from the 18-th layer, where the accuracy is the highest, as the input for the domain discriminator. 

## **4.3 Evaluation Metric** 

The evaluation of our method on the target domain data encompasses both semantic consistency between text and images and image quality. For semantic consistency, we used Blipscore [17, 18] and followed the part on attribute binding in T2I-CompBench [14]. For image quality, we used aesthetic predictor, ImageReward [33], and Human Preference Score(HPS) [32]. Furthermore, we recruited human evaluators to assess the aesthetics of the images and the consistency between text and images. 

For the metrics of color, shape, and texture, we use the model parameters in the target domain from our dataset and the text provided by T2I-CompBench as input. For other metrics, we use the model parameters and natural prompts in the target domain from our dataset. 

> 2https://civitai.com/ 

> 3https://github.com/christophschuhmann/improved-aesthetic-predictor 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Natural<br>LanguagePrompt PromptMagic PromptPerfect Promptist SUR-adapter AKP AKP+SUR-adapter Ours<br>Natural Language Prompt : a black cat with yellow eyes, standing in a vibrant and colorful forest. The background is filled with a variety<br>of colors, creating a visually striking and artistic atmosphere.<br>Realistic Vision<br>SD V1.4<br>DreamShaper<br>V3.5<br>Lexica Aperture<br>**----- End of picture text -----**<br>


Figure 3: Results of diverse APO methods on MGAPO task. Each column displays images from the same APO method, and each row features images generated by the same SD model. APO methods with matching color and SD model share the same domain, indicating that the training data for the APO model was generated by that specific SD model. 

## **4.4 Comparison to Other Methods** 

We compared various prompt optimization methods mentioned in Section 2. MagicPrompt [7], a popular prompt generator for SD from Hugging Face, PromptPerfect [16], a commercial tool for prompt optimization, Promptist [8], and SUR-adapter [37]. Additionally, automatic keyword prompts and manual prompts are keyword based prompts generated by GPT-3.5 based on image descriptions and manually designed prompts in our dataset, respectively. Furthermore, we integrate our ICL-based prompt rewriting with the SUR-adapter for comparison in our experiments. Concretely, we augment the representation of automatic keyword prompts to the representation of natural language prompts from the SUR-adapter, using it as the generation condition. 

Table 1 shows that our proposed method outperforms most prompt optimization methods in semantic consistency and image quality. The results indicate that our proposed method exhibits good generalization performance in prompt optimization tasks. The APO method based on text optimization performs well on metrics that focus solely on image quality, such as AES scores and HPS, but falls short in terms of semantic consistency. In contrast, the SUR-adapter is able to maintain semantic consistency across different models, but does not perform well in terms of image aesthetics. Our method, however, can ensure both semantic consistency and aesthetics on unseen models. This conclusion is further strengthened by human evaluation results (Figure 4), where we see that our method consistently achieves higher win-rates than the other APO strategies. 

## **4.5 Case Study** 

In Figure 3, we showcase the generalization performance of different APO methods. The SD models in the figure are all out-of-domain models for our approach. It is evident that most APO methods produce satisfactory results within their training domains but struggle to generate good results on out-of-domain models. For instance, Promptist generated a "cat-human" image on the Dreamshaper model. PromptPerfect generates narrow-angled images on all of the out-of-domain models. In contrast, our method consistently produces accurate and aesthetically pleasing images across all out-of-domain models. We provide more comparative cases in Appendix C.3. 

To analyze the content contained in the trained domain prototypes, we artificially adjusted the weights _ηs_ in Equation 8 during the inference phase. We selected domain prototypes obtained from models with two different styles and treated _Pa_ as a different linear combination of the two. This is equivalent to adding domain information in different proportions to the generation conditions. In the Figure4a, 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0009-00.png)


**----- Start of picture text -----**<br>
Ours win Others win Ours win Others win<br>Manual Prompts 44.3% 55.7% Manual Prompts 47.8% 52.2%<br>Key-word+SUR 55.2% 44.8% Key-word+SUR 51.3% 48.7%<br>Key-word Prompts 53.6% 46.4% Key-word Prompts 55.7% 44.3%<br>SUR-adapter 65.1% 34.9% SUR-adapter 48.4% 51.6%<br>Promptist 57.5% 42.5% Promptist 62.7% 37.3%<br>PromptPerfect 53.0% 47.0% PromptPerfect 60.5% 39.5%<br>MagicPrompt 54.9% 45.1% MagicPrompt 59.9% 40.1%<br>Anime Realism<br>(a) (b) (c)<br>**----- End of picture text -----**<br>


Figure 4: (a) Linear combinations of domain prototypes from anime style models and domain prototypes from realism style models. The blending ratio changes from left to right. (b) Human Evaluation for Aesthetics. (c) Human Evaluation for Text-Image Alignment. 

Table 2: Ablation study of the loss functions. Table 3: Effectiveness of domain prototypes. 

|_Lr Ld La Lc_ **BlipScore AesScore ImageReward**<br>✓<br>0.303<br>6.110<br>0.254<br>✓<br>✓<br>0.313<br>6.139<br>0.258<br>✓<br>✓<br>✓<br>0.324<br>6.275<br>0.384<br>✓<br>✓<br>✓<br>✓<br>**0.332**<br>**6.384**<br>**0.427**|**Color BlipScore AesScore ImageReward**|
|---|---|
||**Ours**<br>**0.477**<br>**0.332**<br>**6.384**<br>**0.427**<br>**w/ random prototypes** 0.462<br>0.328<br>6.183<br>0.373<br>**w/o prototypes**<br>0.451<br>0.302<br>6.140<br>0.258|
|||



we show the results of mixing animation style domain and the realistic style domain. From left to right, we gradually reduce the weight of anime and increase the weight of realism. In the Appendix C.4, we provide more cases for further analysis of domain prototypes. 

## **4.6 Ablation Study** 

**Effectiveness of Domain Prototypes.** To evaluate the effectiveness of domain prototypes, we employed three settings. The first setting involves not using domain prototypes, only utilizing the adapter for fine-tuning. The second setting involves using a fixed-parameter fully connected layer to align the dimensions of image features and domain prototypes, which is equivalent to randomly selecting image feature channels. The third setting is the one where we drop domain-insensitive channel information. The evaluation results are presented in the Table 3, showing that dropping domain-insensitive information effectively enhances the quality of the generated images. 

**Effectiveness of Loss Functions.** We conducted ablation experiments on the four loss functions we adopted. The results are shown in the Table 2. The removal of the domain style distillation loss _La_ led to a significant decrease in the quality of the generated images. The denoising loss _Ld_ had a relatively minor impact on the generated results. Ultimately, the optimal performance emerged through the combination of all four losses, affirming the synergistic enhancement each proposed module provides. This underscores the significance of an integrated approach for superior model generalization, with each module playing a pivotal role. We visualize the ablation of loss functions in the Appendix C.5. 

## **4.7 Visualization of Conditioned Features** 

We visualize the conditioned features used for text-to-image generation using the t-SNE algorithm. These include the original natural language descriptions of the images, keyword prompts optimized by the first-stage large language model, features output by the second-stage AP-adapter, and manually designed prompts. 

We use different colors to represent different domains, as shown in Figure 5, reflecting the domain distinctiveness of various features. Both the natural language descriptions and keyword prompts have poor domain distinctiveness. We observe that the domain distinctiveness of the AP-adapter is even better than manually designed prompts. This is likely because we introduce image features that better represent domain differences. Therefore, the output of the AP-adapter performs better in domain distinctiveness than purely textual manual prompts. 

9 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0010-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0010-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0010-02.png)


**----- Start of picture text -----**<br>
(a) Natural language descriptions. (b) Keyword prompts.<br>(c) AP-adapter (d) Manually designed prompts.<br>**----- End of picture text -----**<br>


Figure 5: Visualization of text-conditioned domain distinctiveness. (a) Natural language description of the image. (b) Keyword prompts output by the first-stage large language model. (c) Features output by the second-stage AP-adapter. (d) Manually designed prompts. 

## **5 Conclusion** 

In this paper, we propose a new task, MGAPO, which identifies shortcomings of previous APO methods in generalization of unseen models. To address this challenge, we propose a two-stage approach called AP-Adapter that combines both text optimization and representation optimization methods. In the first stage, we leveraged the zero-shot reasoning capability of large language models to rewrite natural language prompts to keyword prompts. In the second stage, we introduce the domain prototypes, which decodes images to obtain representation for various domains. These prototypes are then combined with text representations to align with the representations of manually designed prompts. Rigorous experimentation and evaluation affirm the effectiveness of our proposed approach in optimizing prompts for out-of-domain models, achieving superior results without compromising the quality of the generated images. 

## **Limitations** 

While our dataset offers valuable insights into the CIVITAI community, its relatively small size and focus on single-character images may limit the model’s performance in handling more intricate and complex scenes. The necessity for an increasing number of domain prototypes as the dataset expands poses a challenge, leading to a growth in model parameters. Future work could explore leveraging parameter disparities among different models to tailor the APO optimization strategy, addressing this potential scalability concern. 

10 

## **Acknowledgement** 

This work is supported by the National Natural Science Foundation of China under Grants Nos. 61972192, 62172208, 61906085. This work is partially supported by Collaborative Innovation Center of Novel Software Technology and Industrialization. This work is supported by the Fundamental Research Funds for the Central Universities under Grant No. 14380001. 

## **References** 

- [1] Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., _et al._ (2023). Improving image generation with better captions. _Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf_ , **2** (3), 8. 

- [2] Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., _et al._ (2023). Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. _arXiv preprint arXiv:2310.00426_ . 

- [3] Cui, Y., Yu, Z., Cai, R., Wang, X., Kot, A. C., and Liu, L. (2023). Generalized few-shot continual learning with contrastive mixture of adapters. _arXiv preprint arXiv:2302.05936_ . 

- [4] Ding, Y., Tian, C., Ding, H., and Liu, L. (2024). The clip model is secretly an image-to-prompt converter. _Advances in Neural Information Processing Systems_ , **36** . 

- [5] Fu, Y., Xie, Y., Fu, Y., and Jiang, Y.-G. (2023). Styleadv: Meta style adversarial training for cross-domain few-shot learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 24575–24584. 

- [6] Guo, J., Qi, L., and Shi, Y. (2023). Domaindrop: Suppressing domain-sensitive channels for domain generalization. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 19114–19124. 

- [7] Gustavosta (2022). Magicprompt-stable-diffusion. `https://huggingface.co/Gustavosta/ MagicPrompt-Stable-Diffusion` . 

- [8] Hao, Y., Chi, Z., Dong, L., and Wei, F. (2022). Optimizing prompts for text-to-image generation. _arXiv preprint arXiv:2212.09611_ . 

- [9] He, Y., Robey, A., Murata, N., Jiang, Y., Williams, J., Pappas, G. J., Hassani, H., Mitsufuji, Y., Salakhutdinov, R., and Kolter, J. Z. (2024). Automated black-box prompt engineering for personalized text-to-image generation. _arXiv preprint arXiv:2403.19103_ . 

- [10] Hessel, J., Holtzman, A., Forbes, M., Bras, R. L., and Choi, Y. (2021). Clipscore: A referencefree evaluation metric for image captioning. _arXiv preprint arXiv:2104.08718_ . 

- [11] Ho, J., Jain, A., and Abbeel, P. (2020). Denoising diffusion probabilistic models. _Advances in neural information processing systems_ , **33** , 6840–6851. 

- [12] Hou, S. and Wang, Z. (2019). Weighted channel dropout for regularization of deep convolutional neural network. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 33, pages 8425–8432. 

- [13] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ . 

- [14] Huang, K., Sun, K., Xie, E., Li, Z., and Liu, X. (2023a). T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. _arXiv preprint arXiv:2307.06350_ . 

- [15] Huang, W., Chen, C., Li, Y., Li, J., Li, C., Song, F., Yan, Y., and Xiong, Z. (2023b). Style projected clustering for domain generalized semantic segmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 3061–3071. 

- [16] JinaAI (2023). Promptperfect. `https://promptperfect.jina.ai/prompts` . 

11 

- [17] Li, J., Li, D., Xiong, C., and Hoi, S. (2022a). Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _International Conference on Machine Learning_ , pages 12888–12900. PMLR. 

- [18] Li, J., Li, D., Savarese, S., and Hoi, S. (2023). Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ . 

- [19] Li, W.-H., Liu, X., and Bilen, H. (2022b). Cross-domain few-shot learning with task-specific adapters. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7161–7170. 

- [20] Liu, H., Li, C., Li, Y., and Lee, Y. J. (2023). Improved baselines with visual instruction tuning. _arXiv preprint arXiv:2310.03744_ . 

- [21] Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., and Chen, W. (2021). What makes good in-context examples for gpt-3? _arXiv preprint arXiv:2101.06804_ . 

- [22] Mañas, O., Astolfi, P., Hall, M., Ross, C., Urbanek, J., Williams, A., Agrawal, A., RomeroSoriano, A., and Drozdzal, M. (2024). Improving text-to-image consistency via automatic prompt optimization. _arXiv preprint arXiv:2403.17804_ . 

- [23] Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., and Zettlemoyer, L. (2022). Rethinking the role of demonstrations: What makes in-context learning work? _arXiv preprint arXiv:2202.12837_ . 

- [24] Mo, W., Zhang, T., Bai, Y., Su, B., Wen, J.-R., and Yang, Q. (2024). Dynamic prompt optimizing for text-to-image generation. _arXiv preprint arXiv:2404.04095_ . 

- [25] Reimers, N. and Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. _arXiv preprint arXiv:1908.10084_ . 

- [26] Rosenman, S., Lal, V., and Howard, P. (2023). Neuroprompts: An adaptive framework to optimize prompts for text-to-image generation. _arXiv preprint arXiv:2311.12229_ . 

- [27] Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., and Aberman, K. (2023). Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 22500–22510. 

- [28] Song, J., Meng, C., and Ermon, S. (2020). Denoising diffusion implicit models. _arXiv preprint arXiv:2010.02502_ . 

- [29] Vu, T., Khadivi, S., Phung, D., and Haffari, G. (2022). Domain generalisation of nmt: Fusing adapters with leave-one-domain-out training. In _Findings of the Association for Computational Linguistics: ACL 2022_ , pages 582–588. 

- [30] Wang, X., Zhu, W., Saxon, M., Steyvers, M., and Wang, W. Y. (2023a). Large language models are latent variable models: Explaining and finding good demonstrations for in-context learning. In _Thirty-seventh Conference on Neural Information Processing Systems_ . 

- [31] Wang, Z., Montoya, E., Munechka, D., Yang, H., Hoover, B., and Chau, P. (2023b). Diffusiondb: A large-scale prompt gallery dataset for text-to-image generative models. In _Annual Meeting of the Association for Computational Linguistics_ . 

- [32] Wu, X., Sun, K., Zhu, F., Zhao, R., and Li, H. (2023). Better aligning text-to-image models with human preference. _arXiv preprint arXiv:2303.14420_ . 

- [33] Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., and Dong, Y. (2023). Imagereward: Learning and evaluating human preferences for text-to-image generation. _arXiv preprint arXiv:2304.05977_ . 

- [34] Yang, L., Yu, Z., Meng, C., Xu, M., Ermon, S., and Cui, B. (2024). Mastering text-toimage diffusion: Recaptioning, planning, and generating with multimodal llms. _arXiv preprint arXiv:2401.11708_ . 

12 

- [35] Ye, H., Zhang, J., Liu, S., Han, X., and Yang, W. (2023). Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. _arXiv preprint arXiv:2308.06721_ . 

- [36] Zhang, L., Rao, A., and Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 3836–3847. 

- [37] Zhong, S., Huang, Z., Wen, W., Qin, J., and Lin, L. (2023). Sur-adapter: Enhancing text-toimage pre-trained diffusion models with large language models. _arXiv preprint arXiv:2305.05189_ . 

13 

## **A Supplemental Algorithm Information** 

We illustrate our detailed training process in the Algorithm 1. 

## **Algorithm 1** Training pipeline 

|**Require:** Checkpoints_S_ =_{_Θ1_,_Θ2_, ...,_Θ_K}_, datasets(_nk_<br>_i , mk_<br>_i , Ik_<br>_i_ )_Nk_<br>_i_=1 in the source domain,_f T_<br>_clip_|**Require:** Checkpoints_S_ =_{_Θ1_,_Θ2_, ...,_Θ_K}_, datasets(_nk_<br>_i , mk_<br>_i , Ik_<br>_i_ )_Nk_<br>_i_=1 in the source domain,_f T_<br>_clip_|**Require:** Checkpoints_S_ =_{_Θ1_,_Θ2_, ...,_Θ_K}_, datasets(_nk_<br>_i , mk_<br>_i , Ik_<br>_i_ )_Nk_<br>_i_=1 in the source domain,_f T_<br>_clip_|
|---|---|---|
||and_fden_ of the stable diffusion model,_f I_<br>_clip_ image encoder,_fllm_ large language model, domain||
||prototypes_{P dom_<br>_k_<br>_}K_<br>_k_=1(_ϕ_1), adapter_gada_(_·_;_ϕ_2).||
|**Ensure:** Trainable domain prompt parameters_ϕ_1and adapter parameters_ϕ_2.|||
|1:|**Initialize:** Set_fllm_ and_f I_<br>_clip_ to pretrained parameters.||
|2:|**for**T in training epochs**do**||
|3:|**if**T is even**then**||
|4:|Freeze domain prototypes parameters_ϕ_1.||
|5:|**else**||
|6:|Freeze adapter parameters_ϕ_2.||
|7:|**end if**||
|8:|**for**dataset of domain_i_in datasets**do**||
|9:|Set_f T_<br>_clip_ and_fden_toΘ_k_ parameters.||
|10:|Get keyword prompt token embedding_f T_<br>_clip_(_fllm_(_nk_<br>_i_ ))and manual prompt token embedding||
||(_f T_<br>_clip_(_mk_<br>_i_ )).||
|11:|Get domain prototypes_Pa_by Equation 8.||
|12:|Concatenate domain prototypes and token embedding representations as adapter input.||
|13:|Obtain the output of the adapter_gada_.||
|14:|**if**T is even**then**||
|15:|Calculate values of loss functions by Equation|13.|
|16:|**else**||
|17:|Calculate values of loss functions by Equation|12.|
|18:|**end if**||
|19:|Update all learnable parameters.||
|20:|**end for**||
|21:|**end for**||



## **B Supplemental Dataset Information** 

## **B.1 Dataset Analysis** 

**Example of Data.** Figure 6 shows a sample from our dataset. We utilize the LLaVA-v1.5 13B model to generate natural language descriptions based on images. According to the query in the Figure 6, LLaVa generates detailed descriptions about the image content and style, typically consisting of 5 to 6 sentences. To filter out the most relevant parts with the image, we calculate the CLIP similarity between each sentence and the image, selecting the top 3 sentences with the highest similarity as natural language prompts. 

**Example of Stable Diffusion Models.** In Figure 7, we list some of the stable diffusion (SD) models we used. The models are ranked based on the number of samples they contribute to the dataset, with the top 20 models displayed. This ranking also reflects the popularity and generation quality of these models within the community. These models mainly come from two sources: one is models obtained by fine-tuning SD with custom data, and the other is models obtained by merging the weights of other models. 

**Distribution of Sample Size.** As shown in the Figure 8, we present the distribution of samples for all models in the dataset. For clarity, we use a sample quantity of 35 as the threshold. It can be observed from the graph that the number of models with a large number of samples is relatively small, with many models having fewer than 10 samples. Therefore, we divide our dataset into three parts, as mentioned in Section 4.1. As stated, we assign models with a higher number of samples to the training and test sets, and the remaining samples were used as the demonstration library for Section 3.3. 

14 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0015-00.png)


**----- Start of picture text -----**<br>
“Describe this image and its style in a very detailed manner. ”<br>LLaVA<br>High quality image<br>Natural Language Prompt : a beautifully drawn illustration of a woman dressed<br>in a medieval outfit, possibly inspired by the character Zelda from the Legend of<br>Zelda video game series. She is standing in a river, and appears to be looking up.<br>The woman is wearing a long dress, and her attire includes a headdress.<br>Mannually Designed Prompt :<br>Positive: \\(Style\\): { (flat_color, masterpiece:1.2, best<br>quality) },\n\\(Composition\\): { 1girl, solo, cute, mid shot from_side, kneeling,<br>looking up },\n\\(Hair\\): { long_blonde_hair, wavy hair },\n\\(Appearance\\):<br>{ mature female, princess zelda, nintendo, the legend of zelda, oot, young zelda,<br>slim, slender, fit, small_breasts, flat_chest, blue eyes },\n\\(Location\\): { outdoors,<br>dreamy autum forest on hills landscape with river in the valley, crystal blue sky,<br>birds flying away }, \n\\(Loras\\): { <lora:young_zelda_v1:.8> }"<br>Negative:  (worst quality, low quality:1.4), by bad-artist-anime, by bad-artist, bad-<br>hands-5:1.3, bad_prompt_version2, EasyNegative, ng_deepnegative_v1_75t,<br>verybadimagenegative_v1.2<br>Stable Diffusion<br>Model name:  ghostmix_v12<br>**----- End of picture text -----**<br>


Figure 6: A sample from the dataset, each containing a natural language prompt generated by LLaVa, a manually designed prompt, a high-quality image, and model information. 

## **B.2 Ethical Statement** 

Due to the nature of our dataset, which is entirely generated by diffusion models, it is unlikely to be used for systems that may violate personal privacy. Additionally, we have filtered out samples containing explicit and nude images from the data. Upon careful examination of our dataset, we believe it is unlikely to provide harmful information. 

## **C Supplemental Experiments** 

## **C.1 More Implementation Details** 

In the ICL-Based Prompt Rewriting stage, we utilize the gpt-3.5-turbo API to generate automatic keyword prompts. It is noteworthy that, at this stage, the large language model generates automatic keyword prompts based on the demonstrations provided by ICL, including positive and negative prompts. Only positive prompts participate in training, while negative prompts serve as negative conditions for image generation during the testing phase. Here we provide the automatic keyword prompts generated from the natural language prompts in Figure 3. 

Positive Prompts: masterpiece, best quality, vibrant colors, artistic atmosphere, black cat, yellow eyes, forest scene, visually striking, colorful background, unique appearance, eyecatching, animal focus, detailed, high resolution, solo, nature theme, visually appealing, beautiful, gorgeous 

Negative Prompts: lowres, bad anatomy, bad eyes, error, missing features, distorted features, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad pictures, disconnected elements, unnatural colors, bad composition, overexposure, underexposure, monochrome, bad prompt, horror, easynegative. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-10.png)


revAnimated_v122 majicmixRealistic_v6 chilloutmix_NiPrunedFp32Fix deliberate_v2 CounterfeitV30_v30 ghostmix_v20Bakedvae endjourney_v1.1 majicmixRealistic_v5 yesmix_v15 dreamshaper_7 majicmixRealistic_v4 photon_v1 icbinpICantBelieveIts_final revAnimated_v11 ghostmix_v12 meinamix_meinaV10 majicmixSombre_v20 lyriel_v16 dreamshaper_631BakedVae xxmix9realistic_v40 

Figure 7: The top 20 models by sample size, along with the respective proportions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0016-13.png)


**----- Start of picture text -----**<br>
Num of Samples Num of Samples<br>(a) The distribution of model sample size within (b) The distribution of model sample size with<br>the range of 0 and 35. more than 35.<br>Frequency Frequency<br>**----- End of picture text -----**<br>


Figure 8: The distribution of model sample size. 

In the Prototype-Based Prompt Adaptation stage, all models are trained on two NVIDIA RTX 3090 GPUs, with steps set to 10000, batch size set to 16, and image resolution set to 512. Standard data augmentation techniques, such as normalization, resizing, and horizontal flipping, are applied during training. As for the platform to implement our network, we use PyTorch 2.1. 

## **C.2 Module Analysis** 

**Different CLIP Image Encoder Layers for Domain Discrimination.** To effectively acquire domain information, we tested the performance of the outputs of all layers in CLIP ViT-L/14 for domain 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0017-00.png)


**----- Start of picture text -----**<br>
1<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23<br>Index of Layer<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 9: Domain discrimination accuracy at different layers of the CLIP image encoder. 

classification, as shown in the Figure 9. We found that as the layer index increases, the domain classification ability of image embeddings is not strictly monotonically increasing. This phenomenon may be due to domain information being more related to image style rather than semantic information. As the layers of the CLIP image encoder deepen, the output embeddings contain more semantic information. 

**Different backbone of the CLIP image encoder.** The evaluation results show the impact of different CLIP image encoders on the domain discriminator and the final generated results. Table 4 indicates that the choice of backbone has a minor influence on both domain classification and the quality of generated images. Additionally, it is observed that, across all backbones, features from deeper layers tend to be more advantageous for domain discrimination. 

Table 4: Evaluation results of the impact of different backbones of the CLIP image encoder on the generated images. Acc refers to the classification accuracy of the domain discriminator. 

||**Acc**|**BlipScore**|**AesScore**|
|---|---|---|---|
|**RN50**|0.925|0.304|6.282|
|**RN101**|0.954|0.318|6.351|
|**ViT-B/32**|0.942|0.311|6.302|
|**ViT-L/14**|0.960|0.332|6.384|



## **C.3 More Cases for Comparison** 

We present more image generation results on out-of-domain models in Figure 10 and Figure 11. The models in Figure 10 follow the setup from Figure 3, while Figure 11 provides additional results from various models. 

According to the results, it can be observed that images generated using our method perform well in both semantic consistency and aesthetic quality across all models. In contrast, results generated by other methods are not stable across different models. 

## **C.4 More Cases for Domain Prototypes Analysis** 

We visualized the impact of domain prototypes on image generation in Figure 4a, Figure 12, and Figure 13. In Figure 4a and Figure 12, we calculate _Pa_ using the following formula: 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0018-00.png)


**----- Start of picture text -----**<br>
Lexica Aperture DreamShaper SD V1.4 Realistic Vision Lexica Aperture DreamShaper SD V1.4 Realistic Vision Lexica Aperture DreamShaper SD V1.4 Realistic Vision<br>Natural<br>Language<br>Prompt<br>Magic<br>Prompt<br>Prompt<br>Perfect<br>Promptist<br>SUR-<br>adapter<br>Ours<br>a gold and white dressA beautiful and colorful illustration of a woman with to her elegant appearance. . She is adorned with flowers in her hairblue hair, wearing , adding  teddy bear a wooden teddy bear is wearing a suit and a tiewith a moon visible in the background. The , giving it a formal appearance.  a serene landscape featuring with cloudssky casting a soft glow over the landscape. . The scene is set during flowers, a forest, a lake, and a sky filled sunset, with the warm hues of the<br>**----- End of picture text -----**<br>


Figure 10: More comparative results of generated images on both in-domain and out-of-domain models. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0018-02.png)


**----- Start of picture text -----**<br>
epicrealism aniverse majicmixFantasy anything epicrealism aniverse majicmixFantasy anything epicrealism aniverse majicmixFantasy anything<br>Natural<br>Language<br>Prompt<br>Magic<br>Prompt<br>Prompt<br>Perfect<br>Promptist<br>SUR-<br>adapter<br>Ours<br>a woman with pink hair, wearing a red jacket and sunglasses. The woman is standing in front of ocean, which adds to the overall atmosphere of the scene.  a young man wearing a black suit and a tie, standing on a city street at night. a young woman sitting on the ground in a forest, with her eyes closed. She is wearing a white shirt and a headband. The woman appears to be in a peaceful and contemplative state, possibly meditating or simply enjoying the natural surroundings.<br>**----- End of picture text -----**<br>


Figure 11: More comparative results of generated images on out-of-domain models. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0018-04.png)


which _η ∈_ [0 _,_ 1]. 

In these cases, it is evident that domain prototypes possess clear semantic information, representing specific image styles such as cartoon, realistic, or 3D animation. 

However, domain prototypes do not always possess clear semantic information. As shown in Figure 13, we gradually inject domain prototypes into the conditional information by adjusting the _ηi_ value of _Pi[dom]_ . It seems difficult to define the semantics represented by the domain prototypes in these cases. We believe this reflects a comprehensive factors such as image detail richness, photography techniques, lighting, image contrast, and image softness. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0019-00.png)


**----- Start of picture text -----**<br>
𝑃𝑃𝑖𝑖𝑑 𝑑𝑑 𝑃𝑃𝑗𝑗𝑑 𝑑𝑑<br>**----- End of picture text -----**<br>


Figure 12: Linear combinations of domain prototypes from _i_ th models and domain prototypes from _j_ th models. 

## **C.5 More Ablation Study** 

In Figure 14, we present additional results from the ablation experiments on the loss functions. Sequential ablation of each of the four loss functions reveals that the absence of any single loss function variably affects the image quality. 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-20.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-24.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0020-25.png)


**----- Start of picture text -----**<br>
𝜂𝜂𝑖𝑖<br>**----- End of picture text -----**<br>


**0** 

Figure 13: Injection of _i_ th domain prototype information by adjusting the value of _ηi_ . 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-00.png)


**----- Start of picture text -----**<br>
w/o ℒ w/o ℒ w/o ℒ w/o ℒ w/ all ℒ<br>𝑐𝑐 𝑎𝑎 𝑟𝑟 𝑑𝑑<br>A woman with long, dark hair, wearing a white shirt, and eating a slice of pizza. She is posing for the<br>camera while enjoying her meal. The scene takes place on a street.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-05.png)


A cat wearing a green jacket and sunglasses, giving it a stylish and unique appearance. The cat is positioned in the center of the image. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-11.png)


A cartoon figure of a man wearing a suit and a hat. The man appears to be a caricature of Donald Trump, the former U.S. president. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models_images/nips_2024_AP-Adapter_Improving_Generalization_of_Automatic_Prompts_on_Unseen_Text-to-Image_Diffusion_Models.pdf-0021-17.png)


A plaster portrait from the Renaissance period. The plaster appears to be a man with long hair, facing the camera directly. 

Figure 14: Demonstration of Ablation Study of Loss Functions. 

21 

## **NeurIPS Paper Checklist** 

## 1. **Claims** 

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? 

Answer: [Yes] 

Justification: Our abstract clearly reflects the contribution and scope of the paper. Guidelines: 

   - The answer NA means that the abstract and introduction do not include the claims made in the paper. 

   - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. 

   - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. 

   - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 

2. **Limitations** 

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] 

Justification: Limitations of our work are mentioned in Appendix 5 Guidelines: 

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. 

- The authors are encouraged to create a separate "Limitations" section in their paper. 

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. 

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. 

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. 

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. 

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. 

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations. 

## 3. **Theory Assumptions and Proofs** 

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? 

Answer: [NA] 

22 

Justification: Our paper does not include theoretical results. Guidelines: 

- The answer NA means that the paper does not include theoretical results. 

- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced. 

- All assumptions should be clearly stated or referenced in the statement of any theorems. 

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. 

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. 

- Theorems and Lemmas that the proof relies upon should be properly referenced. 

## 4. **Experimental Result Reproducibility** 

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] 

Justification: In the Section 3 of the paper, we provide a detailed description of the model framework we propose. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. 

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. 

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. 

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example 

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. 

- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. 

- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). 

- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 

## 5. **Open access to data and code** 

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? 

23 

Answer: [No] 

Justification: Our contributions include the dataset we collected and the code for model training and testing. We will release the data and code after the paper is accepted. Guidelines: 

- 

- Please see the NeurIPS code and data submission guidelines ( `https://nips.cc/ public/guides/CodeSubmissionPolicy` ) for more details. 

- While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). 

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( `https: //nips.cc/public/guides/CodeSubmissionPolicy` ) for more details. 

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. 

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. 

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). 

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 

## 6. **Experimental Setting/Details** 

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? 

Answer: [Yes] 

Justification: We describe these details in Section 4. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. 

- The full details can be provided either with the code, in appendix, or as supplemental material. 

## 7. **Experiment Statistical Significance** 

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 

Answer: [No] 

Justification: Our experiments involve the generation of a large number of images, which is time-consuming. Therefore, we did not perform cross-validation to calculate the error bars. Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. 

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). 

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) 

- The assumptions made should be given (e.g., Normally distributed errors). 

24 

- It should be clear whether the error bar is the standard deviation or the standard error of the mean. 

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. 

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). 

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 

## 8. **Experiments Compute Resources** 

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? 

Answer: [Yes] 

Justification: We provide these information in Appendix C.1. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. 

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. 

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper). 

## 9. **Code Of Ethics** 

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics `https://neurips.cc/public/EthicsGuidelines` ? 

Answer: [Yes] 

Justification: Our research conform with the NeurIPS Code of Ethics. 

Guidelines: 

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. 

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 

## 10. **Broader Impacts** 

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? 

Answer: [Yes] 

Justification: In the Section 1, we discuss the positive impact, and we believe this research has no negative social implications. 

Guidelines: 

- The answer NA means that there is no societal impact of the work performed. 

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. 

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. 

25 

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. 

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. 

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 

## 11. **Safeguards** 

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? 

Answer: [NA] 

Justification: Although our data was collected from web crawling, we filtered out any unsafe data. Furthermore, our image data consists solely of synthetic images that do not contain any information that could pose harm to the real world. 

Guidelines: 

- The answer NA means that the paper poses no such risks. 

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. 

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. 

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 

## 12. **Licenses for existing assets** 

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? 

Answer: [Yes] 

Justification: All of the creators or original owners of assets used in our paper are cited properly. 

Guidelines: 

- The answer NA means that the paper does not use existing assets. 

- The authors should cite the original paper that produced the code package or dataset. 

- The authors should state which version of the asset is used and, if possible, include a URL. 

- The name of the license (e.g., CC-BY 4.0) should be included for each asset. 

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. 

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, `paperswithcode.com/datasets` has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. 

26 

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. 

- If this information is not available online, the authors are encouraged to reach out to the asset’s creators. 

## 13. **New Assets** 

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 

Answer: [Yes] 

Justification: We will release the data and code including documentation after the paper is accepted. 

Guidelines: 

- The answer NA means that the paper does not release new assets. 

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. 

- The paper should discuss whether and how consent was obtained from people whose asset is used. 

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 

## 14. **Crowdsourcing and Research with Human Subjects** 

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 

Answer: [NA] 

Justification: Our paper does not involve crowdsourcing nor research with human subjects. Guidelines: 

   - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

   - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. 

   - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 

15. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human** 

## **Subjects** 

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? 

Answer: [NA] 

Justification: Our paper does not involve crowdsourcing nor research with human subjects. Guidelines: 

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. 

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. 

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 

27 

