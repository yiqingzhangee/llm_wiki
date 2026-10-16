# - LAMP: LANGUAGE-MOTION PRETRAINING FOR MO TION GENERATION, RETRIEVAL, AND CAPTIONING 

**Zhe Li**[1] _[∗]_ **, Weihao Yuan**[2] _[∗]_ **, Yisheng He**[2] **, Lingteng Qiu**[2] **, Shenhao Zhu**[3] **, Xiaodong Gu**[2] **, Weichao Shen**[2] **, Yuan Dong**[2] **, Zilong Dong**[2] _[†]_ **, Laurence T. Yang**[1] _[†]_ 

1 Huazhong University of Science and Technology 

2 Alibaba Group 

3 Nanjing University 

## ABSTRACT 

Language plays a vital role in the realm of human motion. Existing methods have largely depended on CLIP text embeddings for motion generation, yet they fall short in effectively aligning language and motion due to CLIP’s pretraining on static image-text pairs. This work introduces **LaMP** , a novel **La** nguage- **M** otion **P** retraining model, which transitions from a language-vision to a more suitable language-motion latent space. It addresses key limitations by generating motioninformative text embeddings, significantly enhancing the relevance and semantics of generated motion sequences. With LaMP, we advance three key tasks: text-tomotion generation, motion-text retrieval, and motion captioning through aligned language-motion representation learning. For generation, we utilize LaMP to provide the text condition instead of CLIP, and an autoregressive masked prediction is designed to achieve mask modeling without rank collapse in transformers. For retrieval, motion features from LaMP’s motion transformer interact with query tokens to retrieve text features from the text transformer, and vice versa. For captioning, we finetune a large language model with the language-informative motion features to develop a strong motion captioning model. In addition, we introduce the LaMP-BertScore metric to assess the alignment of generated motions with textual descriptions. Extensive experimental results on multiple datasets demonstrate substantial improvements over previous methods across all three tasks. The code of our method will be made public. 

## 1 INTRODUCTION 

Human motion represents an expanding frontier in computer vision, with substantial implications for diverse applications including film production, gaming, virtual reality, and robotics. Traditionally, textual representations have been favored for their simplicity and accessibility, enabling prominent tasks such as text-based motion retrieval, motion-to-text captioning, and text-driven motion generation. Despite the prevalent use of text in human motion tasks, challenges persist in achieving effective alignment between language and motion representations. 

Previous methods usually leverage the pretrained CLIP (Radford et al., 2021) text encoder to extract embeddings from textual instructions for text-driven motion generation. Specifically, in many diffusion-based frameworks (Chen et al., 2023; Tevet et al., 2023; Lou et al., 2023; Zhang et al., 2023b), CLIP text embeddings are utilized as conditional signals to control the diffusion process. On the other hand, some transformer-based frameworks (Guo et al., 2022a; Zhang et al., 2023a; Zhong et al., 2023; Guo et al., 2023) stack the text embeddings with the input to guide the motion generation. In these frameworks, the extracted text embeddings serve as rich semantic representations and guide the generation process to produce motion sequences following the text instructions. In this paper, we seek to push the limit of the second path by designing a novel approach. 

Although recent methods have achieved impressive results (Guo et al., 2023; Zhang et al., 2023a;b; Chen et al., 2023), they share a common drawback: relying on text embeddings from CLIP as 

> _∗_ Equal Contribution _†_ Corresponding Author. 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Learned Queries Attention mask<br>Motion  Motion<br>Encoder Transformer<br>Motion-Text  Motion-Text  Motion-Text  Text-guided<br>Shared  Contrastive Learning Matching Contrastive Motion Generation<br>weights<br>Motion-guided  Text-guided<br>Text Generation Motion Generation<br>“A person walks Text  Text<br>in a circle”  Tokenizer Transformer Motion-Text  Motion-guided<br>Matching Text Generation<br>**----- End of picture text -----**<br>


Figure 1: LaMP overview. We conduct joint training for contrastive learning, matching, and bidirectional text-motion translation by leveraging the textual features extracted from tokenized text descriptions via the text transformer and the motion features derived from the motion transformer. 

condition signals. CLIP is well pretrained to align language and visual representation in the latent space. However, it is not optimal for aligning language and motion in two aspects. First, CLIP is pretrained using text-image pairs, which may capture text embeddings that primarily represent image features, potentially overlooking information relevant to motion. Second, since CLIP is pretrained on static images, it focuses predominantly on static characteristics, disregarding the dynamic elements associated with motion. As a result, the text embeddings generated by CLIP may not sufficiently reflect the relationship between text and motion, leading to suboptimal performance in applications that demand a high degree of synchronization between verbal descriptions and corresponding movements. 

To address this problem, we propose **LaMP** , achieving a paradigm shift from a language-vision latent space to a more appropriate language-motion latent space. This brings three benefits. First, the text embeddings directly model the dynamics in the motion rather than only modeling the static state. Second, instead of vision-informative, it can obtain motion-informative text embeddings as condition signals, which enhances the quality of generated motion, allowing for more contextually relevant and semantically aligned motion sequences. Third, along with the motion-informative text features, we can also acquire language-informative motion features. Thanks to these motion features, we achieve better motion-text retrieval and finetune a large language model (LLM) to perform motion captioning. 

In this paper, starting from the language-motion pretraining model LaMP, we improve three human motion tasks, i.e. text-motion retrieval, text-to-motion generation, and motion-to-text captioning. For the LaMP pertaining, we perform language-motion representation learning by constraining the LaMP to derive text representations that are most pertinent to the corresponding motion. To achieve effective language-motion alignments, inspired by BLIP2 (Li et al., 2023) for language-vision alignment, four tasks are employed: motion-text contrastive learning, motion-text matching, motion-grounded text generation, and text-grounded motion generation. After pretraining, we utilize the extracted features **LaMP-Feat** for motion-text retrieval. The motion features from LaMP’s motion transformer interact with query tokens to retrieve text features from the text transformer, and vice versa. Next, we propose a text-to-motion generation model **LaMP-T2M** . We adopt LaMP’s text transformer in place of CLIP as the text encoder to extract text embeddings, which serve as the conditional signal to guide motion generation. Following previous works (Devlin, 2018; Guo et al., 2023; Pinyoanuntapong et al., 2024), we also employ the masked prediction technique. However, unlike previous methods, we adopt a decoder-only architecture, which alleviates the degradation in expressive power caused by low-rank matrices (Dong et al., 2021) during the training process to some extent. Additionally, the causal attention mask enhances information interaction within the masked regions. Subsequently, we propose a motion-to-text captioning model **LaMP-M2T** . We leverage the motion transformer from LaMP to obtain language-informative motion features, which are then fed into an LLM. By finetuning this LLM using LoRA (Hu et al., 2021), we develop an LLM capable of motion captioning. At last, we propose to evaluate the quality of generated motion using **LaMP-BertScore** . Specifically, we input the generated motion into LaMP-M2T to obtain the corresponding textual description, and then calculate the BertScore (Zhang et al., 2019) between the generated text and the ground truth text. This metric serves to assess how well the generated motions align with the true semantics. 

Extensive experiments conducted across various datasets demonstrate the effectiveness of our approach in text-to-motion generation, motion-text retrieval, and motion-to-text captioning, with 

2 

significant improvements compared to previous state-of-the-art methods. For instance, the FID is decreased by 28.9% on the HumanML3D dataset, and 28.0% on the KIT-ML dataset for motion generation. The primary contributions of this work are then summarized as follows: 

- We novelly propose a **La** nguage- **M** otion **P** retraining model, termed as **LaMP** , and apply it to extract textual embeddings as conditional signals to guide motion generation. It not only ensures that the generated motions align more closely with semantic information but also reduces the gap between modalities. Furthermore, we observe that LaMP demonstrates outstanding capabilities in motion-text retrieval as well. To the best of our knowledge, we are the first to replace the language-vision space of CLIP with a language-motion space in motion generation. 

- We propose a motion generation model LaMP-T2M, where we employ an autoregressive masked prediction mechanism. This approach alleviates the reduction in expressive capability caused by low-rank matrices during training and enhances the information interaction among masked regions. 

- We propose a motion captioning model LaMP-M2T, where an LLM is fine-tuned from the LaMP motion feature. Based on this, we also introduce a new metric LaMP-BertScore for evaluating the extent to which generated motions align with semantic information. 

- We outperform previous methods in three tasks including motion-text retrieval, motion-to-text captioning, and text-to-motion generation. 

## 2 RELATED WORK 

**Text-guided Human Motion Generation** Early research on text-to-motion generation can be divided into two primary approaches: diffusion-based continuous regression (Tevet et al., 2023; Chen et al., 2023; Zhang et al., 2022; Lou et al., 2023; Zhang et al., 2023b; Yuan et al., 2023) and transformer-based discrete classification (Zhang et al., 2023a; Zhong et al., 2023; Guo et al., 2023; Zou et al., 2024). Among the diffusion-based methods, MotionDiffuse (Zhang et al., 2022) stands out as the first model to use fine-grained text instructions for body part motion. MDM (Tevet et al., 2023) learns the relationship between motion and input conditions on raw data, while MLD (Chen et al., 2023) reduces computational overhead with a latent space diffusion process. In the transformer-based category, motion is encoded with VQ-VAE (Van Den Oord et al., 2017) to create discrete tokens for prediction tasks. T2M-GPT (Zhang et al., 2023a) generates motion sequences from discretized inputs via a transformer. MoMask (Guo et al., 2022a) employs a motion residual VQ-VAE with multiple codebooks, enhancing generation through a residual transformer. This paper follows the transformer-based path but simplifies the process by utilizing a single codebook without a refinement step, resulting in a simpler and more flexible framework. 

**Motion-Text Retrieval** Motion retrieval remains an under-explored area, which presents challenges due to its cross-modal nature, necessitating nearest-neighbor searches between text and motion modalities. Recent work T2M (Guo et al., 2022a) focuses on constructing a retrieval model for evaluation, using a margin-based contrastive loss and Euclidean distance for batch pairs. TEMOS (Petrovich et al., 2022) has a synthesis branch to generate motions from text and creates a cross-modal embedding space but only aligns positive pairs. TMR (Petrovich et al., 2023) builds upon TEMOS, and integrates a contrastive training strategy that incorporates negative examples, thus improving retrieval from a diverse set of fine-grained motions. In contrast, we adopt a completely different pipeline, using the motion and text transformer from pretrained LaMP directly for retrieval. 

**Human Motion Captioning** To articulate human motion via natural language, (Takano & Nakamura, 2015) formulates a mapping between motion sequences and linguistic descriptions utilizing two statistical models. TM2T (Guo et al., 2022b) proposes a motion representation that condenses continuous motions into a concise sequence of discrete variables, leveraging a neural translation network for cross-modal mapping. Motiongpt (Jiang et al., 2023) introduces a motion-language training scheme with instruction tuning, to learn from task feedback and generate motion captions through prompts. Different from them, we propose a simpler approach, which utilizes languageinformative motion text features and finetunes an LLM without instruction tuning. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
EncoderMotion  TokensMotion  “A person walksin a circle”  Learned Queries<br>Masked  LaMP<br>Codebook Tokens Text Encoder Motion  LaMP Motion<br>…. Encoder Transformer<br>Masked Transformer<br>Motion  Predicted  “A person walksin a circle”  LLM Decoder LoRA MLP<br>Decoder Tokens<br>(a) LaMP-T2M (b) LaMP-M2T<br>**----- End of picture text -----**<br>


Figure 2: LaMP-T2M and LaMP-M2T frameworks overview. ( **Left** ) Pretrained LaMP’s text transformer is employed to extract condition embedding and autoregressive mask prediction is performed. ( **Right** ) Finetuning an LLM to achieve motion captioning. 

## 3 METHOD 

## 3.1 LAMP: LANGUAGE MOTION PRETRAINING 

We present LaMP, a novel model that aligns motion and language more accurately than CLIP (Radford et al., 2021). Current approaches (Zhang et al., 2023a; Chen et al., 2023; Guo et al., 2023) leverage a CLIP pretrained text encoder to extract text features as conditions for motion generation. However, the CLIP text encoder maps text features into an image-text aligned latent space, which may not perfectly align with motion features due to the domain gap between images and motions. Inspired by (Li et al., 2023), we develop a model that can better capture motion-informative text features and provide more precise conditions for motion generation. Furthermore, we leverage LaMP and learnable query tokens to project motion into the motion-language latent space. With this representation, we finetune a large language model and obtain a motion-to-text model. 

## 3.1.1 PRELIMINARY 

We adopt vanilla VQ-VAE (Van Den Oord et al., 2017) to convert a motion sequence into one tuple of discrete tokens. Specifically, we use 1D convolutional encoder _E_ to encode the motion sequence _M_ 1: _N ∈_ R _[N][×][D]_ to the latent vectors _m_ 1: _N ∈_ R _[n][×][d]_ with downsampling ratio of _N/n_ and latent dimension _d_ . To obtain discrete tokens, we pre-define a learnable codebook _C_ := _{_ ( _s, zs_ ) _}[d] s∈S_[for] the latent vectors _m_ 1: _n_ , where _S_ is the size of codebook and _s_ is the index of embedding in _C_ . Each latent vector is replaced by the nearest codebook embedding in _C_ according to the Euclidean distance, which can be formally denoted as: _Q_ ( _mi_ ) = _mi �→ zs_ . The quantized code sequence _z_ 1: _n_ is then projected back into the motion space to reconstruct the motion _m_ ˆ via decoder _D_ . 

## 3.1.2 MODEL ARCHITECTURE 

Our aim is to align language and motion better. Similar to (Li et al., 2023), LaMP comprises two transformers (Vaswani, 2017) sub-modules that share the same self-attention layers: (1) a motion transformer that interacts with the frozen motion encoder and query tokens, and (2) a text transformer, as in Figure 1. The motion transformer learns to extract a set of salient motion features from the input motion, while the text transformer learns to generate a corresponding set of motion-informative textual outputs. By jointly optimizing these two sub-modules with four auxiliary tasks, LaMP can effectively extract motion-informative text features and language-informative motion features. 

We define a set of learnable query tokens _q_ , which serve as the input to the motion transformer. These queries interact with each other through self-attention layers and with motion features via cross-attention layers. Moreover, the queries can engage with the text through the same self-attention mechanism. Following (Li et al., 2023), we initialize LaMP with the pre-trained BERTbase (Devlin, 2018), with the cross-attention layers randomly initialized. 

## 3.1.3 MOTION-LANGUAGE ALIGNMENT 

We jointly optimize four distinct objectives that share a common input format and model parameters. Each objective employs a unique attention masking strategy to modulate the interaction between motion and language. 

4 

**Motion-Text Contrastive Learning** Contrastive learning is a key technique for aligning modalities. Similar to CLIP (Radford et al., 2021), we perform motion-text contrastive learning in LaMP. It aims to bring positive samples closer together and push negative samples further apart in the latent space, thereby maximizing the mutual information between motion and language. We first input the motion embeddings output from the motion encoder into the motion transformer and interact with the query tokens _q_ , obtaining the motion feature _fm_ , and input the text into the text transformer to obtain the text feature _ft_ . Then the pairwise similarity is computed between _fm_ and _ft_ , and we select the highest one as the motion-text similarity. Following (Li et al., 2023), to avoid information leakage, we employ a unimodal self-attention mask, where motions and texts are not allowed to see each other. 

**Motion-Text Matching** Matching tasks can enable finer-grained alignment between modalities. We frame the task as a binary classification, where the model is required to predict whether a given motion-text pair is matched or unmatched. We employ a bi-directional self-attention mechanism, enabling the model to capture the interdependent relationships between motion and language. The motion embeddings, text, and query tokens are fed together into the transformer, and the output is passed to a binary linear classifier to obtain logits, which are then averaged across the batch to obtain the matching score. The matching task further increases the mutual information between motion and language, and achieves information interaction. 

**Motion-grounded Text Generation** To further enhance the alignment between the two modalities and get the informative features, we hope to achieve cross-modal translation in LaMP. Given a motion sequence as the condition, we intend to generate the corresponding text. We first utilize pretrained tokenizer to tokenize the texts and obtain ground truth labels. Then we input the motion embedding and query token _q_ into the motion transformer to obtain the motion feature _fm_ , which is then fed into the text transformer. To generate texts autoregressively, we employ a causal attention mask and compute the classification loss _Lmgt_ on the output tokens. 

**Text-grounded Motion Generation** Different from BLIP-2 (Li et al., 2023), we perform textgrounded motion generation in LaMP, thanks to pretrained motion VQ-VAE in stage 1, to enhance the alignment. BLIP-2 (Li et al., 2023) primarily focuses on extracting visual features that can be comprehended by LLMs. However, we require not only language-informative motion features but also motion-informative text features. Consequently, we incorporate text-grounded motion generation during the pretraining phase of LaMP. The ground truth label of motion sequence can be obtained from the pretrained codebook _C ∈_ R _[S][×][d]_ in the first stage: _{bi_ = _Q_ ( _mi_ ) _}i_ =1 _...n_ . We input the text into the text transformer and interact with query tokens _q_ via the motion transformer’s cross-attention layers. The resulting features are then used for autoregressive generation through the motion transformer. Additionally, we define a motion classifier head _F_ to map the generated results _{gi}i_ =1 _...n_ to the space _∈_ R _[S]_ , and compute the classification loss with the ground truth label. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0005-04.png)


The joint training across the four aforementioned tasks endows LaMP with strong motion-language alignment capability. After training, the text transformer of LaMP can extract motion-informative text features, making it a better condition encoder than CLIP for motion generation (Section 3.2). Given the current R-Precision Top-1 metric only achieves around 52% accuracy on real data, LaMP can also serve as a better motion-text retrieval model (Section 3.3) and an evaluator for the R-Precision and Multimodal Distance metrics, as verified in the experiments. Additionally, the motion transformer can also extract language-informative motion features. We utilize language-informative motion features based on pretrained LLM to train a motion-to-text model, and based on this model, we propose using the LaMP-BertScore metric to evaluate the quality of generated motion (Section 3.4). 

## 3.2 LAMP-T2M: MOTION GENERATION FROM TEXT 

Drawing inspiration from (Guo et al., 2023), a mask transformer is employed in our work to generate motion tokens. As depicted in Figure 2, we first randomly replace a subset of the sequence elements _{m_ 1 _, m_ 2 _, ...., mn}_ with a special [ _M_ ] token. Let _m[M]_ = _{m_ 1 _, m_ 2 _, ...,_ [ _M_ ] _, mn−_ 2 _,_ [ _M_ ] _, mn}_ denote the sequence after this masking process. Our aim is to accurately predict the masked tokens, given the context text _t_ and the partially masked sequence _m[M]_ , thereby endowing the model with generative 

5 

capabilities. Different from existing approaches, we use pretrained LaMP for extracting text features. The optimization objective is to minimize the negative log-likelihood of prediction: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0006-01.png)


## 3.2.1 MASK STRATEGY 

We adopt the same mask strategy as MoMask (Guo et al., 2023). We employ a cosine function _γ_ to determine the masking ratio. Specifically, the mask ratio is calculated as _γ_ ( _r_ ) = cos( _πr/_ 2) _∈_ [0 _,_ 1], where _r ∈_ [0 _,_ 1] and _r_ = 0 means a fully masked sequence. During the training process, we randomly sample _r_ from a uniform distribution _U_ (0 _,_ 1), and then we uniformly mask _⌈γ_ ( _r_ ) _· n⌉_ tokens of the whole sequence. 

Additionally, we also perform the remasking strategy utilized in the BERT (Devlin, 2018). Specifically, if a token is chosen for masking, we replace it with [ _M_ ] token with probability 80%, with a random token with probability 10%, and keep it unchanged with probability 10%. 

## 3.2.2 AUTOREGRESSIVE GENERATION 

Unlike the bidirectional attention mask in MoMask (Guo et al., 2023), we employ a causal attention mask for autoregressive mask prediction tasks. Currently, transformer-based motion generation models (Guo et al., 2023; Zhang et al., 2023a) commonly utilize bidirectional attention masks, which correspond to encoder-only model architectures. However, during training, the bidirectional attention mask allows the model to simultaneously rely on contextual information, simplifying the mask prediction task and diminishing the model’s generative capacity. 

In addition, this bidirectional masking leads to rank collapse. The attention matrix generated by a bidirectional attention mask typically arises from the product of a low-rank decomposed matrix and a softmax function; specifically, it results from multiplying an _n × d_ matrix with a _d × n_ matrix before applying softmax (where _n ≫ d_ ). This form of attention matrix suffers from reduced expressiveness due to low-rank issues (Dong et al., 2021). In contrast, the attention matrix for a causal attention mask is a lower triangular matrix, with its determinant equal to the product of its diagonal elements. Due to the presence of softmax, all diagonal elements must be positive, ensuring that its determinant is also positive. Consequently, the attention matrix of the causal attention mask (decoder-only architecture) is guaranteed to be full-rank, providing greater expressiveness. Therefore, we predict the masked regions autoregressively: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0006-08.png)


## 3.3 LAMP-FEAT: MOTION TEXT RETRIEVAL 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0006-10.png)


**----- Start of picture text -----**<br>
Text Text<br>T2M Ours<br>Motion Motion<br>**----- End of picture text -----**<br>


Given a text query _T_ —for instance, "A person is walking in a circle"—our primary objective is to rank motions from a comprehensive database based on their semantic alignment with the textual input. The ultimate aim is to retrieve the motion that exhibits the highest correspondence with the provided textual description, effectively bridging the gap between disT2M Ours Figure 3: Heatmap of similarity matrix. The diagonal parate modalities—text and motion. We hypothrepresents positive sample pairs, with darker colors indiesize that a model exhibiting better alignment cating better quality. between these modalities will inherently possess improved cross-modal retrieval capabilities, and can serve as a more reasonable R-Precision evaluator. 

To facilitate this alignment, we leverage the motion embeddings obtained from a frozen motion encoder, which undergoes processing through the motion transformer component of LaMP. This setup enables the embeddings to interact dynamically with the query tokens _q_ , yielding a refined motion feature representation _fm_ . Such refinement is pivotal for accurately retrieving the corresponding text feature _ft_ , as generated by the text transformer. Conversely, our architecture is designed to 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0007-00.png)


**----- Start of picture text -----**<br>
off ground does a little jump put something on shelf walk up<br>different position first walk forward walk back and forth toss something<br>off ground does a little jump put something on shelf walk up<br>different position first walk forward walk back and forth toss something<br>off ground does a little jump put something on shelf walk up<br>different position first walk forward walk back and forth toss something<br>“The man is using both hands to lift something  “A person walks forward several paces, stops  “The man walks back and forth putting  “A man walks up and tosses something.”<br>off ground and places it back on ground in a  and then does a little jump.” something on a shelf.”<br>slightly different position.”<br>T2M-GPT<br>MoMask<br>Ours<br>**----- End of picture text -----**<br>


Figure 4: Qualitative results of text-to-motion generation on HumanML3D. 

support reciprocal interactions; the text embeddings can similarly inform and enhance the motion representations. This bidirectional exchange fosters a richer understanding of the underlying semantics, ultimately ensuring that the retrieved motion sequences are not merely similar, but truly relevant and contextually aligned with the input query. As shown in Figure 3, to validate that LaMP can better match motion-text pairs, we present the similarity matrix in the form of a heatmap. The results indicate that LaMP demonstrates more superior retrieval capabilities than T2M (Guo et al., 2022b). 

## 3.4 LAMP-M2T: MOTION TO TEXT CAPTIONING 

With the assistance of LaMP, we finetune a pretrained LLM (OPT-2.7b (Zhang et al., 2023c)) to generate corresponding text from motion using LoRA (Hu et al., 2021). As depicted in Figure 2, we employ a fully connected layer to linearly project the output motion feature _fm_ into the same dimensional space as the text embeddings of the LLM. We input these language-informative motion features from LaMP and a prompt into LLM. Since LaMP has been pretrained to extract semantically meaningful motion representations, it effectively acts as an information bottleneck, passing the most useful information to the language model. 

We input the generated motion into LaMP-M2T to obtain the textual description of the generated motion, and compute the BertScore against the ground truth text, termed as LaMP-BertScore, which serves to evaluate the extent to which the generated motion aligns with the semantic information. 

## 3.5 INFERENCE 

During the inference process, we initiate with a fully masked sequence _m[M]_ (0) and aim to generate the complete sequence over _K_ iterations. At the _k_ -th iteration, given the partially masked token sequence _m[M]_ ( _k_ ), the model first estimates the probability distribution for the tokens at the masked positions and samples motion tokens based on this distribution. Subsequently, the tokens with the lowest confidence scores, specifically the lowest _⌈γ_ ( _r_ ) _· n⌉_ values, are re-masked, while the remaining tokens stay unchanged for subsequent iterations. The updated token sequence _m[M]_ ( _k_ + 1) serves as the basis for predicting the token sequence in the following iteration until _k_ reaches _K_ . Afterward, all tokens are decoded into motion with pretrained VQVAE decoder in stage 1. 

**Classifier-free Guidance** The classifier-free guidance (CFG) method (Chang et al., 2023; Ho & Salimans, 2022) is employed to integrate text embeddings into the transformer framework. In the training stage, the transformer is trained unconditionally with a probability of 10%. During inference, 

7 

|Methods|R Precision_↑_<br>FID_↓_<br>LaMP-<br>BertScore_↑_<br>MultiModal<br>Dist_↓_<br>Diversity_→_<br>Top 1<br>Top 2<br>Top 3|
|---|---|
|Ground Truth|0_._511_±._003<br>0_._703_±._003<br>0_._797_±._002<br>0_._002_±._00<br>100_._00<br>2_._974_±._008<br>9_._503_±._065|
|HumanML3D<br>TM24T (Guo et al., 2022b)<br>0_._424_±._003<br>0_._618_±._003<br>0_._729_±._002<br>1_._501_±._017<br>-<br>3_._467_±._011<br>8_._589_±._076<br>T2M (Guo et al., 2022a)<br>0_._455_±._003<br>0_._636_±._003<br>0_._736_±._002<br>1_._087_±._021<br>-<br>3_._347_±._008<br>9_._175_±._083<br>MDM (Tevet et al., 2023)<br>-<br>-<br>0_._611_±._007<br>0_._544_±._044<br>-<br>5_._566_±._027<br>**9****_._559****_±._086**<br>MLD (Chen et al., 2023)<br>0_._481_±._003<br>0_._673_±._003<br>0_._772_±._002<br>0_._473_±._013<br>52_._08<br>3_._196_±._010<br>9_._724_±._082<br>MotionDiffuse (Zhang et al., 2022)<br>0_._491_±._001<br>0_._681_±._001<br>0_._782_±._001<br>0_._630_±._001<br>-<br>3_._113_±._001<br>9_._410_±._049<br>T2M-GPT (Zhang et al., 2023a)<br>0_._492_±._003<br>0_._679_±._002<br>0_._775_±._002<br>0_._141_±._005<br>56_._21<br>3_._121_±._009<br>9_._761_±._081<br>PhysDiff (Yuan et al., 2023)<br>-<br>-<br>0.631<br>0.433<br>-<br>-<br>-<br>MotionGPT (Zhang et al., 2024)<br>-<br>-<br>-<br>0.567<br>-<br>3.775<br>-<br>M2DM (Kong et al., 2023)<br>0_._497_±._003<br>0_._682_±._002<br>0_._763_±._003<br>0_._352_±._005<br>-<br>2_._974_±._016<br>9_._926_±._073<br>Fg-T2M (Wang et al., 2023)<br>0_._492_±._002<br>0_._683_±._003<br>0_._783_±._002<br>0_._243_±._019<br>-<br>3_._109_±._007<br>9_._278_±._072<br>AttT2M (Zhong et al., 2023)<br>0_._499_±._003<br>0_._690_±._002<br>0_._786_±._002<br>0_._112_±._006<br>-<br>3_._038_±._007<br>9_._700_±._090<br>DiverseMotion (Lou et al., 2023)<br>0_._510_±._005<br>0_._698_±._006<br>0_._795_±._004<br>0_._103_±._004<br>-<br>2_._974_±._016<br>9_._683_±._102<br>ParCo (Zou et al., 2024)<br>0_._515_±._003<br>0_._706_±._003<br>0_._801_±._002<br>0_._109_±._005<br>-<br>2_._927_±._008<br>9_._576_±._088<br>MMM (Pinyoanuntapong et al., 2024) 0_._504_±._003<br>0_._696_±._003<br>0_._794_±._002<br>0_._080_±._003<br>-<br>2_._998_±._007<br>9_._411_±._058<br>ReMoDiffuse (Zhang et al., 2023b)<br>0_._510_±._005<br>0_._698_±._006<br>0_._795_±._004<br>0_._103_±._004<br>-<br>2_._974_±._016<br>9_._018_±._075<br>MoMask (Guo et al., 2023)<br>0_._521_±._002<br>0_._713_±._002<br>0_._807_±._002<br>0_._045_±._002<br>60_._40<br>2_._958_±._008<br>-<br>Ours<br>**0****_._557****_±._003 0****_._751****_±._002 0****_._843****_±._001 0****_._032****_±._002**<br>**60****_._81**<br>**2****_._759****_±._007**<br>9_._571_±._069||
|Ground Truth<br>0_._424_±._005<br>0_._649_±._006<br>0_._779_±._006<br>0_._031_±._004<br>100_._00<br>2_._788_±._012<br>11_._080_±._097||
|KIT-ML<br>TM2T (Guo et al., 2022b)<br>0_._280_±._005<br>0_._463_±._006<br>0_._587_±._005<br>3_._599_±._153<br>-<br>4_._591_±._026<br>9_._473_±._0117<br>T2M (Guo et al., 2022a)<br>0_._361_±._005<br>0_._559_±._007<br>0_._681_±._007<br>3_._022_±._107<br>-<br>2_._052_±._107<br>10_._72_±._145<br>MDM (Tevet et al., 2023)<br>-<br>-<br>0_._396_±._004<br>0_._497_±._021<br>-<br>9_._191_±._022<br>10_._85_±._109<br>MLD (Chen et al., 2023)<br>0_._390_±._008<br>0_._609_±._008<br>0_._734_±._007<br>0_._404_±._027<br>48_._47<br>3_._204_±._027<br>10_._80_±._117<br>MotionDiffuse (Zhang et al., 2022)<br>0_._417_±._004<br>0_._621_±._004<br>0_._739_±._004<br>1_._954_±._062<br>-<br>2_._958_±._005<br>**11****_._10****_±._143**<br>T2M-GPT (Zhang et al., 2023a)<br>0_._416_±._006<br>0_._627_±._006<br>0_._745_±._006<br>0_._514_±._029<br>46_._53<br>3_._007_±._023<br>10_._86_±._049<br>PhysDiff (Yuan et al., 2023)<br>0_._510_±._005<br>0_._698_±._006<br>0_._795_±._004<br>0_._103_±._004<br>-<br>2_._974_±._016<br>-<br>MotionGPT (Zhang et al., 2024)<br>0_._510_±._005<br>0_._698_±._006<br>0_._795_±._004<br>0_._103_±._004<br>-<br>2_._974_±._016<br>10.54<br>M2DM (Kong et al., 2023)<br>0_._416_±._004<br>0_._628_±._004<br>0_._743_±._004<br>0_._515_±._029<br>-<br>3_._015_±._017<br>11_._417_±._097<br>Fg-T2M (Wang et al., 2023)<br>0_._418_±._005<br>0_._626_±._004<br>0_._745_±._004<br>0_._571_±._047<br>-<br>3_._114_±._015<br>10_._93_±._083<br>AttT2M (Zhong et al., 2023)<br>0_._413_±._006<br>0_._632_±._006<br>0_._751_±._006<br>0_._870_±._039<br>-<br>3_._039_±._021<br>10_._96_±._123<br>DiverseMotion (Lou et al., 2023)<br>0_._416_±._005<br>0_._637_±._008<br>0_._760_±._011<br>0_._468_±._098<br>-<br>2_._892_±._041<br>10_._873_±._101<br>ParCo (Zou et al., 2024)<br>0_._430_±._004<br>0_._649_±._007<br>0_._772_±._006<br>0_._453_±._027<br>-<br>2_._820_±._028<br>10_._95_±._094<br>MMM (Pinyoanuntapong et al., 2024) 0_._3_/_4_±._005<br>0_._590_±._006<br>0_._718_±._005<br>0_._429_±._019<br>-<br>3_._146_±._019<br>10_._633_±._097<br>ReMoDiffuse (Zhang et al., 2023b)<br>0_._427_±._014<br>0_._641_±._004<br>0_._765_±._055<br>0_._155_±._006<br>-<br>2_._814_±._012<br>10_._80_±._105<br>MoMask (Guo et al., 2023)<br>0_._433_±._007<br>0_._656_±._005<br>0_._781_±._005<br>0_._204_±._011<br>56_._89<br>2_._779_±._022<br>-<br>Ours<br>**0****_._479****_±._006 0****_._691****_±._005 0****_._826****_±._005 0****_._141****_±._013**<br>**57****_._54**<br>**2****_._704****_±._018** 10_._929_±._101||



Table 1: The quantitative results of text-to-motion generation with evaluator following previous methods on the HumanML3D dataset and the KIT-ML dataset. 

|Datasets|Methods|LaMP-R Precision_↑_<br>Top 1<br>Top 2<br>Top 3|LaMP-R Precision_↑_<br>Top 1<br>Top 2<br>Top 3|LaMP-R Precision_↑_<br>Top 1<br>Top 2<br>Top 3|LaMP-R MultiModal Dist_↓_|
|---|---|---|---|---|---|
||T2M-GPT (Zhang et al., 2023a)|0_._796_±._002|0_._90_._4_±._003|0_._92_._4_±._001|1_._549_±._008|
|HumanML3D|MoMask (Guo et al., 2023)|0_._824_±._002|0_._937_±._002|0_._957_±._001|1_._306_±._006|
||**Ours**|**0****_._867****_±._002**|**0****_._962****_±._001**|**0****_._981****_±._001**|**1****_._187****_±._008**|
||T2M-GPT(Zhang et al., 2023a)|0_._710_±._002|0_._844_±._002|0_._896_±._001|1_._493_±._019|
|KIT-ML|MoMask (Guo et al., 2023)|0_._732_±._002|0_._871_±._002|0_._922_±._001|1_._289_±._027|
||**Ours**|**0****_._784****_±._002**|**0****_._907****_±._001**|**0****_._963****_±._001**|**1****_._174****_±._013**|



Table 2: Evaluation results of text-to-motion generation with LaMP evaluator on T2M-GPT, MoMask, and ours. 

CFG is implemented at the last linear projection layer right before the softmax operation. At this point, the final logits _lf_ , are computed by adjusting the conditional logits _lc_ relative to the unconditional logits _luc_ , using a guidance scale denoted as _α_ , here _α_ is set as 4: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0008-05.png)


## 4 EXPERIMENTS 

## 4.1 EXPERIMENTS SETTINGS 

**Datasets** We evaluate our model on HumanML3D (Guo et al., 2022a) and KIT-ML (Plappert et al., 2016) datasets. The HumanML3D dataset comprises 14,616 motion sequences accompanied by 44,970 textual descriptions, while the KIT-ML dataset includes 3,911 movement sequences and 6,278 

8 

|Methods<br>Text-Motion Retrieval_↑_<br>R@1_↑_<br>R@2_↑_<br>R@3_↑_<br>R@5_↑_<br>R@10_↑_|Motion-Text Retrieval_↑_<br>R@1_↑_<br>R@2_↑_<br>R@3_↑_<br>R@5_↑_<br>R@10_↑_|
|---|---|
|HumanML3D<br>TEMOS (Petrovich et al., 2022)<br>0_._424<br>53_._52<br>61_._14<br>70_._96<br>84_._15<br>T2M (Guo et al., 2022a)<br>52_._48<br>71_._05<br>80_._65<br>89_._66<br>**96****_._58**<br>TMR (Petrovich et al., 2023)<br>67_._16<br>81_._32<br>86_._81<br>91_._43<br>95_._36<br>Ours<br>**67****_._18****_±_0****_._5 81****_._9****_±_0****_._4 87****_._04****_±_0****_._3 92****_._0****_±_0****_._2** 95_._73_±_0_._2|39_._96<br>53_._49<br>61_._79<br>72_._40<br>85_._89<br>52_._00<br>71_._21<br>81_._11<br>89_._87<br>96_._78<br>67_._97<br>81_._20<br>86_._35<br>91_._70<br>95_._27<br>**68****_._02****_±_0****_._3 82****_._1****_±_0****_._3 87****_._5****_±_0****_._3 92****_._2****_±_0****_._3 96****_._9****_±_0****_._3**|
|KIT-ML<br>T2MOS (Petrovich et al., 2022)<br>43_._88<br>58_._25<br>67_._00<br>74_._00<br>84.75<br>T2M (Guo et al., 2022a)<br>42_._25<br>62_._62<br>75_._12<br>87_._50<br>96.12<br>TMR (Petrovich et al., 2023)<br>49_._25<br>69_._75<br>78_._25<br>87_._88<br>95.00<br>Ours<br>**52****_._5****_±_0****_._7 74****_._8****_±_0****_._5 84****_._7****_±_0****_._5 92****_._7****_±_0****_._3 97****_._6****_±_0****_._3**|41_._88<br>55_._88<br>65_._62<br>75_._25<br>85_._75<br>39_._75<br>62_._75<br>73_._62<br>86_._88<br>95_._88<br>50_._12<br>67_._12<br>76_._88<br>88_._88<br>94_._75<br>**54****_._0****_±._005 75****_._3****_±_0****_._5 84****_._4****_±_0****_._4 92****_._2****_±_0****_._2 97****_._6****_±_0****_._2**|



Table 3: Text-motion ( **left** ) and motion-text ( **right** ) retrieval benchmark on the HumanML3D and KIT-ML. 

|Method|R Precision_↑_MultiModal<br>Dist_↓_<br>LaMP-<br>BertScore_↑_Bleu@1_↑_Bleu@4_↑_Rouge_↑_Cider_↑_<br>Top 1 Top 3|
|---|---|
|T2MT (Guo et al., 2022a)<br>0.516 0.823<br>2.935<br>32.2<br>**48.9**<br>7.00<br>**38.1**<br>16.8<br>Motiongpt (Jiang et al., 2023) 0.543 0.827<br>2.821<br>32.4<br>48.2<br>12.47<br>37.4<br>**29.2**<br>LaMP-M2T (Ours)<br>**0.547 0.831**<br>**2.808**<br>**32.7**<br>47.8<br>**13.04**<br>37.1<br>28.9||



Table 4: The quantitative results of motion captioning on the HumanML3D, we adhere to the evaluation frameworks outlined in (Jiang et al., 2023). 

text inputs. In line with the methodologies established in prior research (Guo et al., 2022a), we allocate 23,384 samples for training, 1,460 for validation, and 4,383 for testing within HumanML3D, and utilize 4,888 for training, 300 for validation, and 830 for testing in KIT-ML. The extracted motion poses yield motion features with dimensionalities of 263 for HumanML3D and 251 for KIT-ML. These motion features encapsulate both global attributes, such as root velocity, root height, and foot contact, as well as local details including joint positions, velocities, and rotations relative to the root. The local joint data accounted for corresponds to 22 and 21 joints, respectively, from the SMPL model (Loper et al., 2023) for the HumanML3D and KIT-ML datasets. 

**Implementation details** Our model is implemented on NVIDIA A100 GPU using PyTorch. For the motion VQ-VAE, we employ resblocks for both the encoder and decoder, with a downscale factor of 4. The VQ consists of 6 quantization layers, where each layer’s codebook contains 512 512-dimensional codes. The quantization dropout ratio _p_ is set to 0.2. The masked transformer is composed of 6 transformer layers with casual attention masks, 6 heads, and a latent dimension of 384. The learning rate reaches 2e-4 after 2000 iterations with a linear warm-up schedule for the training of all models. During inference, we set the CFG scale of mask transformer on HumanML3D, and 2 on KIT-ML. Meanwhile, _K_ was set to 10 on both datasets. 

## 4.2 EVALUATION 

**Evaluation of Text-to-Motion Generation** We conduct an extensive evaluation of our model against prior text-to-motion approaches, encompassing both diffusion-based and transformer-based models. Our results, summarized in Table 1, reveal that our method demonstrably outperforms all previous methods on both the HumanML3D and KIT-ML datasets. Notably, our model achieves an improvement of 6.9%, 5.3%, and 4.7% in R-Precision Top {1, 2, 3} on HumanML3D, respectively. Furthermore, we also improve FID by 28.9% on HumanML3D, and achieve state-of-the-art LaMPBertScore, highlighting the remarkable fidelity of our generated motions. Qualitative results presented in Figure 4 corroborate these findings, demonstrating that our method produces motions exhibiting a significantly better alignment with the input text compared to existing techniques. 

We also report LaMP-R Precision and LaMP-Multimodal Distance obtained by extracting motion embeddings and text embeddings using LaMP in Table 2, showing significant improvements over the existing evaluator. This reveals that the LaMP evaluator could better evaluate methods in the motion generation task. 

**Evaluation of Motion-Text Retrieval** To prove the retrieval ability of LaMP, we also evaluate standard retrieval performance for both text-to-motion and motion-to-text tasks, using metrics analogous to the R-Precision. These metrics include R@{1,2,3,5,10}, where a higher value signifies better performance. Recall at rank _k_ represents the proportion of instances where the correct label appears within the top _k_ retrieved results. Importantly, all retrieval evaluations are conducted on a test set of 

9 

||R Precision_↑_<br>FID_↓_<br>LaMP-<br>BertScore_↑_MultiModal Dist_↓_<br>Top 1<br>Top 2<br>Top 3|
|---|---|
|LaMP with contrastive learning<br>0_._519_±._002<br>0_._714_±._005<br>0_._811_±._002<br>0_._084_±._002<br>57_._61<br>2_._876_±._009<br>+motion-text mathcing<br>0_._547_±._002<br>0_._732_±._002<br>0_._824_±._002<br>0_._067_±._002<br>58_._90<br>2_._855_±._007<br>+motion-grounded text generation<br>0_._549_±._002<br>0_._737_±._003<br>0_._828_±._002<br>0_._062_±._002<br>60_._47<br>2_._806_±._007<br>+text-grounded motion generation **0****_._557****_±._003 0****_._751****_±._002 0****_._843****_±._001 0****_._032****_±._002**<br>**60****_._81**<br>**2****_._759****_±._007**||



Table 5: Ablation study of the impact of different tasks in LaMP on generative performance on HumanML3D. 

||R Precision_↑_<br>FID_↓_<br>LaMP-<br>BertScore_↑_<br>MultiModal Dist_↓_<br>Top 1<br>Top 2<br>Top 3|
|---|---|
|Baseline with CLIP<br>+LaMP<br>+Query tokens<br>+Decoder-only|0_._510_±._002<br>0_._701_±._006<br>0_._796_±._002<br>0_._086_±._002<br>57_._49<br>2_._992_±._009<br>0_._548_±._002<br>0_._748_±._002<br>0_._840_±._002<br>0_._040_±._002<br>59_._84<br>2_._794_±._007<br>0_._550_±._002<br>0_._741_±._002<br>0_._834_±._002<br>0_._042_±._002<br>60_._54<br>2_._9783_±._006<br>**0****_._557****_±._003**<br>**0****_._751****_±._002**<br>**0****_._843****_±._001**<br>**0****_._032****_±._002**<br>**60****_._81**<br>**2****_._759****_±._007**|



Table 6: Ablation study of text-to-motion generation on HumanML3D. We report the impact of LaMP’s text encoder, interactions with query tokens, and the mask prediction manner on the results. 

TMR (Petrovich et al., 2023) that includes unseen real motion. Our results including text-motion retrieval and motion-text retrieval, are summarized in Table 3. 

**Evaluation of Motion-to-Text Captioning** Motion-to-text entails generating textual descriptions based on sequences of motion. We perform a comparative analysis of the LaMP-M2T against the recent TM2T (Guo et al., 2022a) and Motiongpt Jiang et al. (2023). Our evaluation is conducted on the HumanML3D dataset, utilizing the metrics proposed in (Guo et al., 2022a). In Table 4, we follow the settings in (Jiang et al., 2023) and adopt the raw ground truth text descriptions to facilitate a more precise evaluation. The results indicate that LaMP-M2T has competitive performance in generating text descriptions corresponding to motion sequences. 

## 4.3 ABLATION STUDY 

In the ablation experiments, we primarily examine the contributions of LaMP, query tokens, and autoregressive mask prediction to the quality of motion generation. 

**Proxy Tasks in LaMP** To validate the significance of each task during the LaMP training process, we report the impact of each task on the generated results in Table 5. The text-grounded motion generation task aids LaMP in extracting motion-informative text features, making this task the most influential on the generation results. 

**LaMP as Text Encoder** To validate the effectiveness of our proposed LaMP, we first establish a baseline model that utilizes CLIP as the text encoder. We then replace CLIP’s text encoder with our proposed LaMP text transformer, using the resulting text features as new conditions to guide the motion generation. Based on the results presented in Table 6, we observe significant improvements in the model across all metrics. 

**Interact with Query Tokens** As shown in Figure 1, during the training process of LaMP, the query tokens _q_ do not directly interact with the text features. However, we believe that the query tokens carry a significant amount of motion information, which enhances the accuracy of the conditional guidance. Therefore, we further enable interaction between the text features and the query tokens through cross-attention layers, resulting in improved generative effects shown in Table 6. 

**Decoder-only Mask Transformer** Unlike MoMask (Guo et al., 2023), our model employs an autoregressive mask transformer. We believe that the bidirectional attention mask suffers from low-rank issues during training (Yang et al., 2024), which reduces the model’s expressive ability and lacks information interaction between the masked regions. We adopt a decoder-only model architecture with a causal attention mask, which mitigates the low-rank problem of the matrix and enhances information interaction between the masked areas. As shown in Table 6, the decoder-only model architecture achieves better generative performance. 

## 5 CONCLUSION 

In this work, we introduce _LaMP_ , a pioneering framework that bridges the gap between language and motion, achieving unprecedented alignment and performance in the process. Our findings 

10 

elucidate how leveraging a nuanced approach to motion-text relationships can significantly enhance the generation of 3D human poses conditioned on textual descriptions. By integrating a decoder-only mask transformer, our _LaMP-T2M_ achieves state-of-the-art results in human motion generation. Concurrently, we propose the _LaMP-M2T_ model for action description and introduce the _LaMPBertScore_ metric to evaluate the quality of the generated motions. Comprehensive experiments validate the effectiveness of the proposed approach. 

## REFERENCES 

- Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. _arXiv preprint arXiv:2301.00704_ , 2023. 

- Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. Executing your commands via motion diffusion in latent space. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 18000–18010, 2023. 

- Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018. 

- Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In _International Conference on Machine Learning_ , pp. 2793–2803. PMLR, 2021. 

- Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3d human motions from text. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 5152–5161, 2022a. 

- Chuan Guo, Xinxin Zuo, Sen Wang, and Li Cheng. Tm2t: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts. In _European Conference on Computer Vision_ , pp. 580–597. Springer, 2022b. 

- Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng. Momask: Generative masked modeling of 3d human motions. 2023. 

- Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. _arXiv preprint arXiv:2207.12598_ , 2022. 

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 

- Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. _Advances in Neural Information Processing Systems_ , 36:20067–20079, 2023. 

- Hanyang Kong, Kehong Gong, Dongze Lian, Michael Bi Mi, and Xinchao Wang. Priority-centric human motion generation in discrete latent space. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 14806–14816, 2023. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International conference on machine learning_ , pp. 19730–19742. PMLR, 2023. 

- Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multi-person linear model. In _Seminal Graphics Papers: Pushing the Boundaries, Volume 2_ , pp. 851–866. 2023. 

- Yunhong Lou, Linchao Zhu, Yaxiong Wang, Xiaohan Wang, and Yi Yang. Diversemotion: Towards diverse human motion generation via discrete diffusion. _arXiv preprint arXiv:2309.01372_ , 2023. 

Mathis Petrovich, Michael J Black, and Gül Varol. Temos: Generating diverse human motions from textual descriptions. In _European Conference on Computer Vision_ , pp. 480–497. Springer, 2022. 

11 

Mathis Petrovich, Michael J Black, and Gül Varol. Tmr: Text-to-motion retrieval using contrastive 3d human motion synthesis. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 9488–9497, 2023. 

- Ekkasit Pinyoanuntapong, Pu Wang, Minwoo Lee, and Chen Chen. Mmm: Generative masked motion model. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 1546–1555, 2024. 

- Matthias Plappert, Christian Mandery, and Tamim Asfour. The kit motion-language dataset. _Big data_ , 4(4):236–252, 2016. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021. 

- Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. _Advances in neural information processing systems_ , 32, 2019. 

- Wataru Takano and Yoshihiko Nakamura. Statistical mutual conversion between whole body motion primitives and linguistic sentences for human motions. _The International Journal of Robotics Research_ , 34(10):1314–1328, 2015. 

- Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. _Advances in neural information processing systems_ , 30, 2017. 

- A Vaswani. Attention is all you need. _Advances in Neural Information Processing Systems_ , 2017. 

- Yin Wang, Zhiying Leng, Frederick WB Li, Shun-Cheng Wu, and Xiaohui Liang. Fg-t2m: Finegrained text-driven human motion generation via diffusion model. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 22035–22044, 2023. 

- Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Shaochen Zhong, Bing Yin, and Xia Hu. Harnessing the power of llms in practice: A survey on chatgpt and beyond. _ACM Transactions on Knowledge Discovery from Data_ , 18(6):1–32, 2024. 

- Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz. Physdiff: Physics-guided human motion diffusion model. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 16010–16021, 2023. 

- Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, and Ying Shan. Generating human motion from textual descriptions with discrete representations. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 14730–14740, 2023a. 

- Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. _arXiv preprint arXiv:2208.15001_ , 2022. 

- Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, and Ziwei Liu. Remodiffuse: Retrieval-augmented motion diffusion model. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 364–373, 2023b. 

- Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models, 2022. 3:19–0, 2023c. 

- Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. _arXiv preprint arXiv:1904.09675_ , 2019. 

12 

- Yaqi Zhang, Di Huang, Bin Liu, Shixiang Tang, Yan Lu, Lu Chen, Lei Bai, Qi Chu, Nenghai Yu, and Wanli Ouyang. Motiongpt: Finetuned llms are general-purpose motion generators. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pp. 7368–7376, 2024. 

- Chongyang Zhong, Lei Hu, Zihao Zhang, and Shihong Xia. Attt2m: Text-driven human motion generation with multi-perspective attention mechanism. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 509–519, 2023. 

Qiran Zou, Shangyuan Yuan, Shian Du, Yu Wang, Chang Liu, Yi Xu, Jie Chen, and Xiangyang Ji. Parco: Part-coordinating text-to-motion synthesis. _arXiv preprint arXiv:2403.18512_ , 2024. 

## A APPENDIX 

## A.1 INTRODUCTION 

This is the supplementary material, which is divided into the following sections: 

## 1. Metrics details are elaborated in Section A.2. 

2. Implementation details are presented in Section A.3, including VQ-VAE implementation details A.3.1, LaMP implementation details A.3.2, and training and inference details for motion generation A.3.3. 

3. More qualitative results are shown in Section A.4, including motion-to-text captioning and text-to-motion generation. 

## A.2 METRICS 

The evaluation metrics from (Guo et al., 2022a) employed in this study include: (1) _Frechet Inception Distance_ (FID), which assesses the quality of generated motions by examining the discrepancies in the distribution of high-level features between synthetic and authentic motion samples; (2) _R-Precision_ and _Multimodal Distance_ , which measure the degree of semantic alignment between the provided textual input and the generated motions; (3) _Multimodality_ that evaluates the diversity of motions produced from identical textual prompts, and (4) _BertScore_ , which evaluates the quality of generated motion. We believe that motion should align with semantic information, rather than being fully consistent with the ground truth motion. We input the generated motion _m_ ˆ into LaMP-M2T to obtain caption _t_[ˆ] and calculate the _BertScore_ between the ground truth _t_ and _t_[ˆ] to assess the extent to which the generated motion conforms to the semantics. 

## A.3 IMPLEMENTATION DETAILS 

## A.3.1 VQ-VAE IMPLEMENTATION DETAILS 

**Motion VQ-VAE** Continuous features often suffer from the problem of data sparsity, where most feature combinations are not present in the training set. Discretization can transform continuous features into a more limited set of categorical features, converting the generation problem into a classification task. Therefore, we first discretize the continuous motion features. A vanilla VQ-VAE (Van Den Oord et al., 2017) can be utilized to convert a motion sequence into one tuple of discrete tokens. Specifically, we use 1D convolutional encoder _E_ encode the motion sequence _M_ 1: _N ∈_ R _[N][×][D]_ to the latent vectors _m_ 1: _N ∈_ R _[n][×][d]_ with downsampling ratio of _N/n_ and latent dimension _d_ . To obtain discrete tokens, we pre-define a learnable codebook _C_ := _{_ ( _s, zs_ ) _}[d] s∈S_[for the latent vectors] _m_ 1: _n_ , where _S_ is the size of codebook and _s_ is the index of embedding in _C_ . 

Subsequently, as described in Eq. 5, each latent vector is calculated the distance with the embeddings in the codebook _C_ , and then replaced by the codebook embedding that has the nearest distance to the original latent vector, which can be formally denoted as: _Q_ ( _mi_ ) = _mi �→ zs_ . The quantized code sequence _z_ 1: _n_ is then projected back into the motion space to reconstruct the motion _m_ ˆ vid decoder _D_ . After all, the indices of the selected embeddings in codebook _C_ , referred to as motion tokens, serve as an alternative discrete representation of the input motion. 

13 

_Q_ ( _mi_ ) = _zs,_ where _s_ = arg min _j∈{_ 1 _...S} ∥zj − mi∥_ 2 (5) 

**Optimization Goal** The conventional optimization objective of VQ-VAE _Lvq_ comprises three key components: the reconstruction loss _Lrecon_ , the embedding loss _Lemb_ and the commitment loss _Lcom_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0014-02.png)


where _β_ is a hyper-parameter for the commitment loss and sg denotes the stop-gradident. Following T2M-GPT (Zhang et al., 2023a), we employ L1 smooth loss and an additional regularization on the velocity. We use _m_ to denote the original motion and _m_ ˆ to represent the reconstructed motion. _V_ ( _m_ ) = _{vi_ = _mi_ +1 _− mi}i_ = _{_ 1 _...n}_ represents the veloctiy of motion sequence _m_ . Therefore, the _Lrecon_ can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0014-04.png)


where _α_ is a hyper-parameter to balance the two losses. 

**Optimization strategy** Since the codebook collapse situation (Razavi et al., 2019; Van Den Oord et al., 2017) can be found in the naive VQ-VAE, we perform exponential moving average (EMA) and codebook reset (Code Reset) to alleviate this problem. EMA facilitates the smooth evolution of the codebook _C_ : _λC[t][−]_[1] + (1 _− λ_ ) _→C[t]_ , where _C[t]_ represents the codebook at iteration _t_ and _λ_ denotes the exponential moving constant. The code Reset technique identifies inactive codes during the training process and subsequently reassigns them based on the input data. 

## A.3.2 LAMP IMPLEMENTATION DETAILS 

During the training of LaMP, we utilize the motion encoder pretrained in VQ-VAE and keep its weights frozen. Following BLIP-2 (Li et al., 2023), we initialize LaMP’s text transformer and motion transformer with the pretrained weights of BERT _base_ , whereas the cross-attention layers in the motion transformer are randomly initialized. Depicted in Figure 5 (a), two transformers share the same self-attention layers, but the motion transformer has specialized cross-attention layers designed for interaction with query tokens. We set the sequence length of queries as 49, each with a dimensionality of 768, matching the hidden dimension of the motion transformer and text transformer. The query tokens pass through the self-attention layer of the motion transformer and interact with the motion through cross-attention layers. 

## A.3.3 TRAINING AND INFERENCE DETAILS FOR MOTION GENERATION 

As illustrated in Figure 5 (b), we utilize LaMP as the text encoder, where the text passes through the self-attention layers of the text transformer. Subsequently, it interacts with the query tokens via the cross-attention layers of the motion transformer, serving as conditions to guide the motion generation. 

During the training process, we employ a causal attention mask for autoregressive generation, which mitigates the effects of low-rank matrices and enhances information interaction among the masked regions. During the inference phase, we align our strategy with that of MoMask (Guo et al., 2023), utilizing a bi-directional attention mask. 

## A.4 MORE QUALITATIVE RESULTS OF MOTION-TO-TEXT CAPTIONING AND TEXT-TO-MOTION GENERATION 

Qualitative results of motion-to-text captioning are presented in Figure 6, which demonstrate that our LaMP-M2T has the capability to generate textual descriptions of the motion sequence. 

Moreover, we present additional qualitative results in Figures 7 and 8. As illustrated in Figure 8, several issues in MoMask, such as floating, are effectively alleviated in our approach. Moreover, our model demonstrates a higher sensitivity to numerical values; for instance, when instructed to 

14 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0015-00.png)


**----- Start of picture text -----**<br>
Self-attention<br>Learned Queries Shared<br>weights<br>Motion transformer Learned Queries<br>Condition Masked<br>Self-attention Cross-attention Cross-attention<br>Transformer<br>Shared<br>weights Self-attention<br>Motion-Text  Motion-Text<br>“A person walksin a circle”  Self-attention Contrastive LearningMotion-guided  Text-guided Matching<br>Text transformer Text Generation Motion Generation “A person walksin a circle”<br>(a) Details in LaMP (b) LaMP as text encoder<br>**----- End of picture text -----**<br>


Figure 5: ( **Left** ) Details in LaMP. Motion transformer consists of self-attention layers and cross-attention layers (interact with query tokens), while text transformer only has self-attention layers. ( **Right** ) LaMP extracts text features as condition signals. 

“A man jumps twice in place.”, our generated motion will accurately reflect this by jumping exactly twice, while MoMask results in more jumps. In contrast, T2M-GPT does not perform any jumps at all. Analysis of the generated motion sequences reveals that our method is capable of producing intricate and engaging motions that align well with the provided text prompts. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0016-00.png)


**----- Start of picture text -----**<br>
“A person is walking in a half circle” “A person is walking backwards.” “A person is doing jumping jacks.”<br>“A man is standing and looking  “A person is crawling across the  “A person is moving something<br>around him.” floor.” with his hands.”<br>“A man stands with his arms out  “A person is standing and repeatedly  “A person is carrying his left arm<br>and then sits down.” touching his face with his left hand.” around in front of him.”<br>“A person is walking forward then  “A man is standing with his arms by  “A person is sitting on something,<br>jumps over an object, turns around,  his sides, takes a long step to his left  brings his hands up to his knees,<br>and jumps back over the object.” and then long steps to his right.” stands up and turns around.”<br>unfold in time axis<br>**----- End of picture text -----**<br>


Figure 6: Qualitative results of motion-to-text captioning on HumanML3D test set. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0017-00.png)


**----- Start of picture text -----**<br>
adjustments right right hand on a rail drunk walking<br>walks in a circle adjustments left move to right turn around<br>adjustments right right hand on a rail drunk walking<br>walks in a circle adjustments left move to right turn around<br>adjustments right right hand on a rail drunk walking<br>walks in a circle adjustments left move to right turn around<br>“The man is walking normally in a circle.” “A person walks forward while making small “A person takes a step forward, moves to their  “A person drunk walking animation turning<br>adjustments left and right.” right, them continues forward with their right  around.”<br>hand on a rail.”<br>T2M-GPT<br>MoMask<br>Ours<br>**----- End of picture text -----**<br>


Figure 7: Qualitative results of text-to-motion generation on HumanML3D. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning_images/iclr_2025_LaMP_Language-Motion_Pretraining_for_Motion_Generation,_Retrieval,_and_Captioning.pdf-0017-02.png)


**----- Start of picture text -----**<br>
walk forward<br>lift a box jump no floating hop on right left<br>put another box twice zig-zag no floating<br>walk forward<br>lift a box jump no floating hop on right left<br>put another box twice zig-zag no floating<br>walk forward<br>lift a box jump no floating hop on right left<br>put another box twice zig-zag no floating<br>“It looks like he if lifting a box on a shelf and  “A man jumps twice in place.” “A man walks forward while swaying his feet  “A man hops on his right left five times, makes a<br>grabbing another box to put on shelf as well.” in a zig-zag path.” small jump, then steps forward into a strong push<br>with both hands.”<br>T2M-GPT<br>MoMask<br>Ours<br>**----- End of picture text -----**<br>


Figure 8: Qualitative results of text-to-motion generation on HumanML3D. 

17 

