# **A Neural Divide-and-Conquer Reasoning Framework for Image Retrieval from Linguistically Complex Text** 

**Yunxin Li**[1] _[∗]_ **, Baotian Hu**[1] _[†]_ **, Yuxin Ding**[1] **, Lin Ma**[2] **, Min Zhang**[1] 1Harbin Institute of Technology, Shenzhen, China, 2Meituan, Beijing {hubaotian, yxding, zhangmin2021}@hit.edu.cn liyunxin987@163.com, forest.linma@gmail.com 

## **Abstract** 

Pretrained Vision-Language Models (VLMs) have achieved remarkable performance in image retrieval from text. However, their performance drops drastically when confronted with linguistically complex texts that they struggle to comprehend. Inspired by the Divide-and-Conquer (Smith, 1985) algorithm and dual-process theory (Groves and Thompson, 1970), in this paper, we regard linguistically complex texts as compound proposition texts composed of multiple simple proposition sentences and propose an end-to-end Neural Divide-and-Conquer Reasoning framework, dubbed NDCR. It contains three main components: 1) _Divide_ : a proposition generator divides the compound proposition text into simple proposition sentences and produces their corresponding representations, 2) _Conquer_ : a pretrained VLMs-based visuallinguistic interactor achieves the interaction between decomposed proposition sentences and images, 3) _Combine_ : a neural-symbolic reasoner combines the above reasoning states to obtain the final solution via a neural logic reasoning approach. According to the dualprocess theory, the visual-linguistic interactor and neural-symbolic reasoner could be regarded as analogical reasoning System 1 and logical reasoning System 2. We conduct extensive experiments on a challenging image retrieval from contextual descriptions data set. Experimental results and analyses indicate NDCR significantly improves performance in the complex image-text reasoning problem. Code link: https://github.com/ YunxinLi/NDCR. 

## **1 Introduction** 

Image-text retrieval tasks have made remarkable progress owing to pretrained Vision-Language Models (VLMs) such as LXMERT (Tan and 

> _∗†_ Corresponding author. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0001-08.png)


Figure 1: An example from the IMAGECODE (Krojer et al., 2022) data set, where the description is linguistically complex and images are minimally contrastive. The target image is in red and others are incorrect frames. The bottom part depicts the conventional method and the neural divide-and-conquer reasoning framework. 

Bansal, 2019), UNITER (Chen et al., 2020), OSCAR (Li et al., 2020b; Zhang et al., 2021), ViLBERT (Lu et al., 2019), CLIP (Radford et al., 2021), and many others. These VLMs are usually trained on the large-scale short text-image corpus by crossmodal semantic alignment methods. They are capable of essential perceptual computing capability and excel at retrieving images from sentences with few objects and simple linguistic, e.g., _“There is a duck swimming in the pond”_ . However, when pretrained VLMs meet the case of retrieving the accurate image from similar candidates based on a linguistically complex text, as the example shown in Figure 1, previous works (Krojer et al., 2022; Talmor et al., 2021a; Thrush et al., 2022) show that they struggle to understand the elaborate description and perform complex cross-modal reasoning. 

According to the dual-process theory for human thinking (Groves and Thompson, 1970; Evans, 2003; Pelaccia et al., 2011), human brains contain two thinking systems: System 1 performs analogi- 

cal reasoning well, which is fast yet unconscious; System 2 is capable of abstract logical reasoning, which is slow yet conscious and well-suitable for complex reasoning problems. The theory could also hold for the image-text retrieval tasks, and the widely adopted models (e.g., VLMs) focus on analogical reasoning as System 1 based on the analysis of deep learning networks (Bengio, 2017, 2019; Bengio et al., 2021). For the linguistically complex description that contains multiple conditions, they have inferior performance, and we need to introduce logical reasoning System 2 more to cover and logically incorporate the scattered information in the description based on System 1. Inspired by the above investigations and classical Divide-andConquer (Smith, 1985) algorithm, we design an end-to-end Neural Divide-and-Conquer Reasoning framework named NDCR. As shown in Figure 1, our key idea is to regard the complex description as compound proposition text and solve the challenging retrieval problem in three steps: divide, conquer, and combine. 

Specifically, **Divide:** NDCR first utilizes a proposition generator to divide the complex compound text and produce the global representation of simple proposition sentences with visually printing them. **Conquer:** we devise a visuallinguistic interactor to achieve the interaction between decomposed proposition sentences and images, which resembles System 1. It uses the Transformer (Vaswani et al., 2017)-based contextual interactor to achieve the inter-learning of different proposition-image pairs. Considering the incorrectness or information loss of simple proposition representation, we also present a modifier to incorporate the context reasoning information to improve their cross-modal reasoning states. **Combine:** we design a learnable neural-symbolic reasoner to integrate reasoning information of simple propositions logically. It first employs a negation executor to obtain a simple proposition sentence’s negational reasoning hidden state and corresponding confidence score. Then, we use the global reasoning information of compound proposition text as the query signal to perform the conjunction operation across simple propositions and their negational information. Finally, as shown in Figure 1, we also combine the inferred results of the neural-symbolic reasoner (resembles System 2) and visual-linguistic interactor (resembles System 1) to obtain the final solution. In this way, the whole framework inte- 

grate the capabilities of Systems 1 and 2 to obtain better performance. 

We conduct extensive experiments on a largescale image retrieval from contextual descriptions data set, IMAGECODE (Krojer et al., 2022). The experimental results indicate that NDCR achieves the state-of-the-art performance and the ablation and case studies verify the effectiveness of different modules. 

Our contributions are as follows: 

- We propose a divide-and-conquer reasoning framework for image retrievals from linguistically complex text, where we first attempt to combine the perceptually analogical reasoning System 1 and neural-symbolic logic reasoning System 2 to solve the complex multimodal reasoning problem. 

- We design a proposition generator capable of producing the global representation of decomposed simple proposition sentences for linguistically complex texts and visually printing them as text. 

- Experimental results indicate our approach remarkably improves the performance, and we obtain the first place on the leaderboard[1] . Ablation and case studies confirm the effectiveness of introducing and combining logical reasoning System 2 based on System 1. 

## **2 Related Works** 

**Pretrained Vision-Language Models for Cross Modal Matching.** Owing to the success of Transformer (Vaswani et al., 2017) architecture equipped with pretrain-finetuning (Erhan et al., 2010) learning method, pretrained VLMs have made a remarkable performance in cross-modal matching or reasoning tasks (Talmor et al., 2021b), especially image-text retrieval. Early pretrained VLMs utilize BERT (Devlin et al., 2019)-like single encoder architecture to encode and fuse the image-text information, then perform image-text reasoning such as ViLBERT (Lu et al., 2019), VisualBERT (Li et al., 2019), and Oscar (Li et al., 2020b). In addition, dual-encoder architecture such as CLIP (Radford et al., 2021), and ALBERT (Li et al., 2021), performs better than single-encoder architecture on image-text matching tasks and is widely used in industry because of its efficiency. 

1https://mcgill-nlp.github.io/imagecode 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0003-00.png)


Figure 2: The overall architecture of neural divide-and-conquer reasoning framework. 

## **Divide-and-Conquer for Question Answering.** 

The divide-and-conquer algorithm (Smith, 1985) aims to divide the complex problem into multiple simple problems and then combine the subproblem results to achieve the final solution. This idea has been used in complex question-answering tasks in the natural language processing area. Zhang et al. (2019) proposed to utilize the decomposition of complex questions for semantic parsing. Min et al. (2019) adopt the question decomposition and rescoring method to perform multi-hop reading comprehension, which makes the reasoning path interpretable and robust. Wolfson et al. (2022) utilized the QDMR structures of complex questions to conduct the decompose-synthesize text-toSQL transformation. Previous pipeline approaches may lead to error cascades in the upper inference process due to the incompleteness or error of decomposed text. The image-text retrieval task has strict requirements on the correctness of text semantic understanding, thus we propose an end-to-end divide-and-conquer method for alleviating the error cascade issue via the whole learning process. 

**Dual-Process Theory.** The dual-process theory shows that human brains have two different thinking Systems. System 1 performs analogical reasoning, and System 2 performs conscious logical reasoning. Combining this theory with practical tasks, some researchers designed various approaches. Mittal et al. (2017) believed that combining vector space models with external knowledge graphs could be regarded as thinking ‘fast’ in vector space along with thinking ’slow’ and ‘deeply’ by reasoning over the knowledge graph. Anthony et al. (2017) also proposed to use a deep learn- 

ing network with a tree search engine as System 1 and System 2, respectively, for sequential decisionmaking problems. Bengio (2017, 2019) advocated the design of a conscious network to achieve the leap from System 1 to System 2. Liu et al. (2022) designed a neural-symbolic system for natural language understanding tasks, which combines the explicit symbolic calculation-based System 2 and fast deep learning network-based System 1. For complex multi-modal reasoning problem, e.g., image retrieval from linguistically complex text, humans usually combine System 1 and System 2 to obtain the final solution. However, current methods relying mainly on deep learning networks resemble System 1 and lack the logical reasoning capability, thus suffering from image-text reasoning with the complex description. In this light, we make the first attempt to combine System 1 and System 2 to tackle this issue by designing a neural divideand-conquer reasoning framework. We introduce a neural-symbolic reasoner in System 2 to conduct the logical operation. The overall framework contains analogical and logical reasoning as humans think, making appreciable gains. 

## **3 Method** 

## **3.1 Overview** 

Image retrieval from contextual descriptions (Krojer et al., 2022) aims to infer the correct image given a linguistically complex text _Y_ = ( _y_ 1 _, ..., yN_ ) and similar images _I_ = ( _I_ 1 _, ..., IL_ ), where _yi_ , _N_ , _Ii_ , and _L_ represent the _i_ th token, the total length of text, _i_ th image, and the number of images, respectively. We propose a novel divide- 

and-conquer reasoning framework to tackle such a task. It consists of three components, namely, Proposition Generator, Visual-Linguistic Interactor, and Neural-Symbolic Reasoner, which are coupled and trained in an end-to-end manner. Specifically, the proposition generator divides the complex description into multiple proposition sentences, allowing it to convert the complex matching problem to simple ones. Afterwards, the visual-linguistic interactor achieves the interaction between decomposed proposition sentences and images, resembling System 1, to perform the essential analogical reasoning. Subsequently, the neural-symbolic reasoner that relies on the reasoning state output by the visual-linguistic interactor resembles System 2 to perform logical reasoning. Finally, we also combine the output results of System 1 and System 

## **3.2 Proposition Generator** 

The proposition generator is a sequence-tosequence model based on the pretrained language model BART. As shown in Figure 2, it employs the encoder to obtain the text representation **H** _Y_ = ( _hcls, hy_ 1 _, ..., hyN_ ) where _hyi_ represents the _i_ th token hidden state. Subsequently, we design a two-layer semantic parsing module to gain the global representation of simple proposition sentences. Concretely, we set the maximum number of simple propositions to 10 and randomly initialize them. The initial vectors are fed to the semantic parsing module to interact with the compound text representation. Take the first layer as an example; the calculation process is following, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0004-03.png)


where **h** _[I]_ is the randomly initial proposition representations. Attention and FNN calculation subnetworks are identical to the transformer (Vaswani et al., 2017) architecture. Different from the transformer, we let the output of Cross-Attention layer subtract the output of Self-Attention layer, aiming to achieve information differences across propositions. 

By doing the same two-layer calculation, we obtain ten global hidden states of simple propositions. Due to context containing different numbers of simple proposition, we use a MLP to predict the target number of simple proposition sentences. It 

only attends to the global hidden state _hcls_ of compound proposition text. Suppose that the predicted number _M_ of simple propositions is 3 (same as Figure 2), we adopt the first-three hidden states of the semantic parsing module as the global representation of the targeted simple proposition. As shown in Figure 2, for explaining what simple propositions represent, we also use the decoder of BART to generate the simple proposition sentence with only attending to their global representations. 

## **3.3 System 1: Visual-Linguistic Interactor** 

After obtaining the global representations of simple proposition sentences, we introduce the visuallinguistic interactor to mine the interaction of image-proposition pairs. Specifically, we use a pretrained visual encoder to obtain the image encoding representations **H** _I_ = ( **h** _I_ 1 _, ...,_ **h** _IL_ ) and fuse them with the simple proposition representation via the dot-product way (as the _“F”_ shown in Figure 2). The two-modal fusion process is **H** ( _p_ ) = _λ ·_ Norm( **P** ) _·_ Norm( **H** _I_ ) _,_ where _λ_ is the hyperparameter set to enlarge the scale of fused vectors. We denote the fused sequence representation of proposition-image pairs to **H** ( _p_ ) = ( **H** ( _p_ 1) _, ...,_ **H** ( _pM_ )) where **H** ( _p_ 1) indicates the sequential representation of first proposition combined with images. 

Then, we employ a two-layer transformer to perform the contextual information interaction for fused sequential representations **H** ( _p_ ) and obtain the initial reasoning states of simple proposition on images. Considering the incorrectness or information loss of simple proposition representation obtained by the proposition generator, we introduce a MLP-based modifier to incorporate the reasoning state of compound proposition text to enhance previous initial reasoning states of simple propositions. The whole process is performed as Eq. 2, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0004-10.png)


where **H** _C_ indicates the fusion information of the compound proposition text and images, gained by the cross-modal encoder (arr. cross encoder as shown in Figure 2). **W** _[M]_[1] _∈_ R[2] _[d][×][d]_ and **W** _[M]_[2] _∈_ R[2] _[d][×]_[2] _[d]_ are learnable parameters. Before feeding **H** ( _p_ ) into the transformer, we introduce the learnable position embeddings _PE_ to facilitate it pay attention to the contextual information 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0005-00.png)


Figure 3: The detailed workflow of Neural-Symbolic Reasoner. It contains the underlying negation executor and upper conjunction operation. 

across images. After obtaining the final reasoning state **H** _[S]_[1] = ( **h**[+] **1** _[, ...,]_ **[ h]**[+] **M**[)][ of simple propositions] in System 1, we adopt a linear prediction head to produce the confidence score of each proposition to images, which are defined as _P[S]_[1] = ( _p_[+] 1 _[, ..., p]_[+] _M_[)] and _p_[+] _M[∈]_[R][1] _[×][L]_[.] 

## **3.4 System 2: Neural-Symbolic Reasoner** 

For complex reasoning problems, the logical reasoning process usually plays a more significant role for intelligent machines and human reasoning (Bengio, 2019), which the visual-linguistic interactor is not capable of. Instead of combining the inferring results in System 1 via rule-based methods such as mean pooling, inspired by Shi et al. (2020); Chen et al. (2021), we devise a learnable NeuralSymbolic Reasoner (NSR) to perform logical reasoning based on System 1 as shown in Figure 2. As depicted in Figure 3, it contains a negation executor to obtain the negational reasoning states and a conjunction operation to acquire the result of logical reasoning with attention to the positive and negational reasoning information. 

**Negation Executor.** The negation executor is a module that takes the reasoning state of a simple proposition as input and produces the corresponding reasoning state of its negation as output. Its aim is to obtain useful cross-modal reasoning states for the negation of a proposition. We regard **H** _[S]_[1] as the positive reasoning state and use a two-layer MLP with the ReLU activation function to obtain the negational reasoning state. The calculation process is given in Eq. 3, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0005-06.png)


where _W_ 2 _[n][, W]_ 1 _[ n][∈]_[R] _[d][×][d]_[,] _[ b]_ 1 _[n][, b][n]_ 2 _[∈]_[R][1] _[×][d]_[ are learn-] able parameters. We define the output of negation executor to **H** _[N]_ = ( **h** _[−]_ **1** _[, ...,]_ **[ h]** _[−]_ **M**[)][,][contrast] to **H** _[S]_[1] .The negational proposition has a different cross-modal reasoning state **H** _[N]_ than the corresponding positive proposition **H** _[S]_[1] . We use the same linear prediction head as System 1 to produce the corresponding confidence score on images, which are presented to _P[N]_ = ( _p[−]_ 1 _[, ..., p][−] M_[)][.] To make the negation executor effective, we will define a negational feedback loss to locally optimize it. 

**Conjunction Operation.** Firstly, we define a new joint representation that incorporates reasoning hidden states and corresponding confidence scores as the initial state of conjunction operation. The process is presented in Eq. 4, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0005-09.png)


where [ _,_ ] indicates the concat calculation and **H** _I_ is the representation of images. **H** _[ns] p_[+] _i_[represents] the positive joint representation of _i_ th proposition. We use the same calculation method as Eq. 4 to obtain the initialized negational representation **H** _[ns]_[Then, we utilize the reasoning state of com-] _p[−] i_[.] pound proposition text **H** _[sg] C_[(Eq.][ 2][) as the signal to] drive the conjunction calculation via the method of multi-head attention equipped with gate fusion, as shown in Figure 3. The whole calculation process is presented in Eq. 5, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0005-11.png)


where _W[s] ∈_ R[2] _[d][×]_[2] _[d]_ , _W[g] ∈_ R[1] _[×]_[4] _[d]_ , _W[S]_[2] _∈_ R[2] _[d][×][d]_ are the learnable parameters and **H** _[f] ∈_ R[1] _[×][L][×][d]_ . We also utilize another linear prediction head to obtain the final confidence score of neural-symbolic reasoner, which is defined as _P[S]_[2] _∈_ R[1] _[×][L]_ . 

## **3.5 Combining System 1 and System 2** 

In addition, we combine inferring confidence scores in System 1 and System 2 to obtain the final solution, achieving the complementarity of System 

1 and System 2. First, we need to acquire the whole representation of **H** _[S]_[1] and **H** _[f]_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0006-01.png)


where _W[l] ∈_ R _[d][×]_[1] , _b[l]_ are learnable parameters. **H** _[S]_[1][(] **[h]**[+] _[∈]_[R] _[M][×][d]_[and] **[H]** _[f] W_[=] _w_ 1 _[, ...,]_ **[ h]**[+] _wM_[)] _W[∈]_ R[1] _[×][d]_ are used to gain the final solution via Eq. 7, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0006-03.png)


where _W[a] , W[b] ∈_ R _[d][×][d]_ , _b[c] ∈_ R _[d]_ , _V ∈_ R _[d][×]_[1] , _W[f] ∈_ R[2] _[d][×][d]_ , _b[v] , b[f] ∈_ R[1] are learnable parameters and _f_ ( _._ ) indicates the sigmoid activation function. This way, we can obtain the final result via taking the maximum one of the confidence score _P[f] ∈_ R[1] _[×][L]_ . 

the output of System 1, System 2 and final solution, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0006-06.png)


where _pi ∈_ R[1] _[×][L]_ and _q_ is the golden label. To make the negation executor effective, we devise a negational feedback loss _Lneg_ to optimize it. We take the prediction result of modifier in System 1 as the positive distribution and make the belief distribution output by the negation executor on the image candidates be far away from positive distribution. The loss calculation method is shown in Eq. 10, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0006-08.png)


where KL indicates the K-L Divergence (Kullback and Leibler, 1951). _θ_ is a super-parameter used to expand the positive and negational interval, which is set to 0.2. Hence, the whole optimization target is _Lmatch_ + _Lneg_ . 

## **4 Experiments** 

## **4.1 Dataset** 

## **3.6 Training Strategies** 

To make the proposition generator perform proposition decomposition and generation effectively, we train it on a large-scale corpus solely and then train the whole NDCR framework on the specific training data. The two training phases are as follows: **Phase 1.** We first pretrain the proposition generator on the released large-scale complex text simplification data set MinWikiSplit (Niklaus et al., 2019), which is composed of 203K pairs of aligned complex source and simplified target sentences. We adopt the cross entropy generation loss _Lg_ for the decoder output. Similar to SimCSE (Gao et al., 2021), we employ the contrastive learning loss _Lc_ to make the global representation of simple proposition sentence different. In addition, we use a crossentropy multi-label classification loss _Lp_ to train the prediction head of numbers of propositions, where the label is the number of simple sentences in the pretraining corpus. The whole training loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0006-14.png)


**Phase 2.** While training NDCR, we employ the proposition sentence-image confidence score to calThe loss will cover 

We conduct extensive experiments on a challenging data set IMAGECODE (Krojer et al., 2022), which contains 94,020 images, and they are divided into 9,402 sets. The overall images are collected from four released data sets: MSR-VTT (Xu et al., 2016), Video-Storytelling (Li et al., 2020a), YouCook (Das et al., 2013), and Open Images V6 (Kuznetsova et al., 2020). It consists of 21,202 human-writing complex descriptions and manually labelling corresponding golden images, which are divided into 16,594, 2,302, and 2,306 for training, validating, and testing, respectively. The image sources in the overall data set include video frames and static images. 

## **4.2 Baselines** 

We compare NDCR with various types of pretrained VLMs and other designed models based on the specific condition of this task. Specifically, ViLBERT (Lu et al., 2019) is a cross encoder where language and vision interact in the transitional layer via cross attention calculation. CLIP (Radford et al., 2021) is a two-stream vision-language encoder with two independent visual and textual encoders. UNITER (Chen et al., 2020) is a singlestream encoder where visual representations and 

text tokens are concatenated and interact via the same transformer. OFA (Wang et al., 2022) is a unified cross-modal and unimodal encoder and has achieved impressive performance on multiple cross modal reasoning tasks. Krojer et al. (2022) also designed a contextual module to improve the interaction across different text-image fusion representations, achieving state-of-the-art performance. 

|**Method**_↓_**Type**_→_|All<br>Video<br>Static|
|---|---|
|CLIP (Radford et al.,2021)<br>CLIP_†_ (Krojer et al.,2022)|28.4<br>20.0<br>60.0<br>29.9<br>22.0<br>59.8|
|UNITER (Chen et al.,2020)<br>UNITER_†_ (Krojer et al.,2022)|24.8<br>17.4<br>52.8<br>25.7<br>19.1<br>50.5|
|ViLBERT (Lu et al.,2019)<br>ViLBERT_†_ (Krojer et al.,2022)|20.9<br>15.0<br>42.7<br>24.5<br>18.0<br>49.3|
|NDCR (ours)|**34.1**<br>**26.1**<br>**64.3**|



## **4.3 Implementation Details** 

The _L_ , _λ_ , and _d_ equal 10, 1000, and 512, respectively. For the proposition generator, we adopt a two-layer semantic parsing module and the pretrained parameters of BART-base version. We set the maximum number of propositions to 10 and trained the proposition generator for 15 epochs on the MinWikiSplit data set. In addition, we set the depth of transformer block to 2 in the visuallinguistic interactor and utilized the finetuned visual encoder of CLIP (ViT-B/16) to encode images. For the cross encoder, we adopt the OFA-large architecture and first finetune it for two epochs before training the overall structure of NDCR. We froze the cross encoder, proposition generator, and visual encoder to prevent overfitting while training NDCR. While training all models, we set the batch size, initial learning rate, and dropout rate to 36, 6 _×_ 1 _e[−]_[5] , and 0 _._ 1, respectively. The maximum training epoch is set to 30, and we employ the Adam Optimizer (Kingma and Ba, 2014) with the initial learning rate declining linearly to train all models. We use the validation set to select the best-performing model. 

Table 1: Model performance (accuracy) on **original testing set** . The results of CLIP, UNITER, ViLBERT, and their variants( _†_ ) are reported by Krojer et al. (2022). The underscore and bold indicate the second highest value and best performance (same as following tables). We report results for all examples and two disjoint subsets: video frames and static images. 

|**Method**_↓_**Type**_→_|All|Video|Static|
|---|---|---|---|
|OFA (Wang et al.,2022)|29.0|22.1|54.8|
|OFA_†_|30.0|23.6|54.6|
|CLIP (Radford et al.,2021)|27.4|19.7|56.5|
|CLIP_†_ (Krojer et al.,2022)|27.6|20.8|53.2|
|NDCR (ours)|**32.8**|**25.7**|59.2|
|System 2|32.4|25.3|**59.3**|
|System 2 w/o Negation|32.0|25.3|57.3|
|System 1|31.6|24.5|58.3|
|System 1 w/o Modifer|19.3|16.4|30.3|



Table 2: Ablation experiments on the **testing** _[∗]_ set, where we manually label the testing set to conduct ablation studies. ’Negation’ and ’Modifier’ indicate the negation executor and modifier. We adopt the mean pooling method to aggregate the predicted results of simple proposition in System 1 and w/o Modifier. 

## **4.4 Main Results** 

**Overall Performance.** We present the performance of NDCR and comparative models on the test set in Table 1. ’ _†_ ’ indicates that the pretrained VLMs are equipped with the contextual module and temporal embedding to enhance the contextual semantic interaction across similar images. This variant shows its effectiveness on the case of video frame according to the comparative performances such as CLIP vs. CLIP _[†]_ . Table 1 reports that the proposed method achieves new state-of-the-art performance on the whole test set and significantly outperforms previous strong baseline (34 _._ 1 vs. 29 _._ 9, _↑_ 4.2). NDCR improves performances both on video frames and static images, especially static images( _↑_ 4.3), which shows its generalization on different cases. We observe that all models perform poorly on the testing samples whose images are from the 

video clips, which may be attributed to the high similarity across video frames. Hence, there is a big room to improve the whole performance on the challenging multi-modal reasoning task. 

## **4.5 Ablation Study** 

**Effectiveness of Modules.** To study the effectiveness of different modules, we re-annotate the test sample with the help of eight related workers (original test labels are not released). The experimental results are presented in Table 2. The performances of reproduced baselines and NDCR have a slight decline, which is because the labelling process for most examples is difficult. There are specific quality differences across human-labelling results, yet it does not affect testing and comparing model perfor- 

|**Method**_↓_**Nums_of_props**_→_|1|2|3|4|5|
|---|---|---|---|---|---|
|Total Number|61|863|1239|126|16|
|CLIP_†_|27|264|302|41|3|
|OFA_†_|28|299|340|34|**5**|
|System 1|30|297|364|36|4|
|System 2|**31**|**304**|373|36|3|
|NDCR|**31**|**304**|**380**|**37**|3|
|∆|3|5|40|2|-2|



Table 3: The number of samples accurately inferred on different numbers of simple proposition sentences. ’Nums_of_props’ indicates the number of simple propositions. ∆ represents the difference in the number of samples that NDCR and OFA _[†]_ accurately predict. 

mances. For the fairness of model comparison, the random seeds of all ablation experiments are set to the same value 10. Firstly, NDCR achieves the best performance and significantly surpasses other models on two test sets. When we add System 2 based on System 1, the overall performance improves by about 1.0, suggesting the neural-symbolic reasoner’s effectiveness. Comparing System 2 and System 2 w/o negation, we observe that the negation executor improves the performance of the neural-symbolic reasoner, mainly in the case of static images. In addition, comparing System 1 and System 1 w/o modifier, we observe that introducing the context reasoning information is a very useful way to enhance the reasoning state representation of decomposed simple proposition sentences. Compared to the best baseline OFA-large (470M), the total parameter size of NDCR is about 440M. NDCR has fewer parameters yet significantly outperforms it (as shown in Table 2). This suggests that the overall performance improvement of NDCR is not due to having larger parameters. 

**System 1 vs. System 2.** We count the experimental results on the test set according to the number of simple proposition sentences into which compound proposition texts are divided. The results are shown in Table 3. The statistical results show that NDCR excels at image retrieval from complex text with medium length, especially for those containing three simple proposition sentences. It verifies the proposed method’s effectiveness in handling the complex image-text reasoning problem. Compared to System 1, System 2 performs better on test samples containing 2 or 3 simple proposition sentences, which suggests that the neural-symbolic reasoner can improve the conjunction operation of predic- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0008-04.png)


Figure 4: A case from the test set, where different colors correspond to the predicted result of models. _P_ 1[+] _,_ 2 _,_ 3[represent][the][inferred][confidence][scores][of][sim-] ple proposition sentences in System 1 and are used to obtain the results in System 2 and final combination process. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text_images/acl_2023_A_Neural_Divide-and-Conquer_Reasoning_Framework_for_Image_Retrieval_from_Linguistically_Complex_Text.pdf-0008-06.png)


Figure 5: Another case from the test set, where it contains two simple proposition sentences. 

tion results of decomposed propositions compared to rule-based methods such as mean pooling. 

## **4.6 Case Study** 

We present two cases in Figure 4 and 5. For the first case (Figure 4), the proposition generator divides the complex text into three proposition sentences, and System 1 inferred the confidence scores ( _P_ 1[+] _,_ 2 _,_ 3[) of them to ten images.][Although these re-] sults of simple proposition sentences contain some errors due to having no explicit supervision signal to train, System 2 (neural-symbolic reasoner) could obtain the correct result with logical reasoning operation compared to the rule-based aggregation method in System 1. It indicates the robustness of System 2. In addition, we observe that the pretrained VLMs and System 1, which are capable of perceptual computing, often fail to cover all text semantics. It is easy for them to ignore pivotal text information (such as “there is no text” shown in Figure 5), which leads to inference errors. In conclusion, combining logical reasoning Sys- 

tem 2 and powerful analogical reasoning System 1 (e.g., pretrained VLMs) has significant potential to take their advantages to address complex reasoning problems. 

## **5 Conclusion** 

In this paper, inspired by the divide-and-conquer algorithm and dual-process theory, we introduced an end-to-end neural divide-and-conquer reasoning framework named NDCR to handle the challenging case of image retrievals from linguistically complex text. NDCR contains a proposition generator to divide the compound proposition text into multiple simple proposition sentences, then uses a visuallinguistic interactor to achieve the interaction of simple propositions and images. To improve the logical reasoning capability, we devise a neuralsymbolic reasoner to gain the logical inferring result based on the output of the visual-linguistic interactor. This way, NDCR performs the lowlevel analogically perceptual computing in System 1 (visual-linguistic interactor) and high-level logical reasoning in System 2 (neural-symbolic reasoner). Finally, we combine the output result in Systems 1 and 2 to obtain the final solution. 

## **Limitations** 

The proposed method NDCR has some limitations as follows: 1) The produced representation of simple proposition sentences in the proposition generator lies in a different space distribution with the image encoding, which affects the performance of their fused representation. Although we introduce the reasoning information of compound proposition text to alleviate this issue, we hope to solve it by improving the text understanding capability of pretrained VLMs. In addition, adopting the pretrained textual encoder of VLMs to perform proposition decomposition is inadequate due to that they present an inferior understanding for the discourse structure of long texts. 2) The performance of samples with highly similar images from video frames is quite different from that of humans. We may improve it from the perspective of image difference modelling. 3) The experimental results indicate that our method is effective at logical inference on examples with medium-length descriptions, but there is still room for improvement for longer descriptions. 

## **Ethics Statement** 

IMAGECODE (Krojer et al., 2022) is an open data set used for scientific research. For ablation studies in the test set, we hired masters and undergraduate students from the research group to re-annotate the label of the test set. We have informed the creators of the data set and only conducted scientific research. 

## **References** 

- Thomas Anthony, Zheng Tian, and David Barber. 2017. Thinking fast and slow with deep learning and tree search. _Advances in Neural Information Processing Systems_ , 30. 

- Yoshua Bengio. 2017. The consciousness prior. _arXiv preprint arXiv:1709.08568_ . 

- Yoshua Bengio. 2019. From system 1 deep learning to system 2 deep learning. In _NeurIPS 2019_ . 

- Yoshua Bengio, Yann Lecun, and Geoffrey Hinton. 2021. Deep learning for ai. _Communications of the ACM_ , 64(7):58–65. 

- Hanxiong Chen, Shaoyun Shi, Yunqi Li, and Yongfeng Zhang. 2021. Neural collaborative reasoning. In _Proceedings of the Web Conference 2021_ , pages 1516–1527. 

- Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. 2020. Uniter: Universal image-text representation learning. In _ECCV_ . 

- Pradipto Das, Chenliang Xu, Richard F. Doell, and Jason J. Corso. 2013. A thousand frames in just a few words: Lingual description of videos through latent topics and sparse object stitching. In _2013 IEEE Conference on Computer Vision and Pattern Recognition_ , pages 2634–2641. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pierre-Antoine Manzagol, Pascal Vincent, and Samy Bengio. 2010. Why does unsupervised pre-training help deep learning? _Journal of Machine Learning Research_ , 11(19):625–660. 

- Jonathan St BT Evans. 2003. In two minds: dualprocess accounts of reasoning. _Trends in cognitive sciences_ , 7(10):454–459. 

- Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Philip M Groves and Richard F Thompson. 1970. Habituation: a dual-process theory. _Psychological review_ , 77(5):419. 

- Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ . 

- Benno Krojer, Vaibhav Adlakha, Vibhav Vineet, Yash Goyal, Edoardo Ponti, and Siva Reddy. 2022. Image retrieval from contextual descriptions. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 3426–3440, Dublin, Ireland. Association for Computational Linguistics. 

- Solomon Kullback and Richard A Leibler. 1951. On information and sufficiency. _The annals of mathematical statistics_ , 22(1):79–86. 

- Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. 2020. The open images dataset v4. _International Journal of Computer Vision_ , 128(7):1956–1981. 

- Junnan Li, Ramprasaath R. Selvaraju, Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, and Steven Hoi. 2021. Align before fuse: Vision and language representation learning with momentum distillation. In _NeurIPS_ . 

- Junnan Li, Yongkang Wong, Qi Zhao, and Mohan S. Kankanhalli. 2020a. Video storytelling: Textual summaries for events. _IEEE Transactions on Multimedia_ , 22(2):554–565. 

- Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. 2019. Visualbert: A simple and performant baseline for vision and language. _arXiv preprint arXiv:1908.03557_ . 

- Xiujun Li, Xi Yin, Chunyuan Li, Xiaowei Hu, Pengchuan Zhang, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, and Jianfeng Gao. 2020b. Oscar: Object-semantics aligned pre-training for vision-language tasks. 

- Zhixuan Liu, Zihao Wang, Yuan Lin, and Hang Li. 2022. A neural-symbolic approach to natural language understanding. _arXiv preprint arXiv:2203.10557_ . 

- Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. 2019. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In _Advances in Neural Information Processing Systems_ , volume 32. Curran Associates, Inc. 

- Sewon Min, Victor Zhong, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2019. Multi-hop reading comprehension through question decomposition and rescoring. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 6097–6109, Florence, Italy. Association for Computational Linguistics. 

- Sudip Mittal, Anupam Joshi, and Tim Finin. 2017. Thinking, fast and slow: Combining vector spaces and knowledge graphs. _arXiv preprint arXiv:1708.03310_ . 

- Christina Niklaus, André Freitas, and Siegfried Handschuh. 2019. MinWikiSplit: A sentence splitting corpus with minimal propositions. In _Proceedings of the 12th International Conference on Natural Language Generation_ , pages 118–123, Tokyo, Japan. Association for Computational Linguistics. 

- Thierry Pelaccia, Jacques Tardif, Emmanuel Triby, and Bernard Charlin. 2011. An analysis of clinical reasoning through a recent and comprehensive approach: the dual-process theory. _Medical education online_ , 16(1):5890. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In _Proceedings of the 38th International Conference on Machine Learning_ , volume 139 of _Proceedings of Machine Learning Research_ , pages 8748–8763. PMLR. 

- Shaoyun Shi, Hanxiong Chen, Weizhi Ma, Jiaxin Mao, Min Zhang, and Yongfeng Zhang. 2020. Neural logic reasoning. In _Proceedings of the 29th ACM International Conference on Information & Knowledge Management_ , pages 1365–1374. 

- Douglas R Smith. 1985. The design of divide and conquer algorithms. _Science of Computer Programming_ , 5:37–58. 

- Alon Talmor, Ori Yoran, Ronan Le Bras, Chandrasekhar Bhagavatula, Yoav Goldberg, Yejin Choi, and Jonathan Berant. 2021a. Commonsenseqa 2.0: Exposing the limits of ai through gamification. In _NeurIPS Datasets and Benchmarks_ . 

- Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. 2021b. Multimodal{qa}: complex question answering over text, tables and images. In _International Conference on Learning Representations_ . 

- Hao Tan and Mohit Bansal. 2019. Lxmert: Learning cross-modality encoder representations from transformers. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 

   - 5100–5111, Hong Kong, China. Association for Computational Linguistics. 

- Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. 2022. Winoground: Probing vision and language models for visio-linguistic compositionality. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5238–5248. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Advances in Neural Information Processing Systems_ , volume 30. Curran Associates, Inc. 

- Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. 2022. OFA: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pages 23318–23340. PMLR. 

- Tomer Wolfson, Daniel Deutch, and Jonathan Berant. 2022. Weakly supervised text-to-SQL parsing through question decomposition. In _Findings of the Association for Computational Linguistics: NAACL 2022_ , pages 2528–2542, Seattle, United States. Association for Computational Linguistics. 

- Jun Xu, Tao Mei, Ting Yao, and Yong Rui. 2016. Msrvtt: A large video description dataset for bridging video and language. In _2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 5288–5296. 

- Haoyu Zhang, Jingjing Cai, Jianjun Xu, and Ji Wang. 2019. Complex question decomposition for semantic parsing. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 4477–4486, Florence, Italy. Association for Computational Linguistics. 

- Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. 2021. Vinvl: Making visual representations matter in vision-language models. 

