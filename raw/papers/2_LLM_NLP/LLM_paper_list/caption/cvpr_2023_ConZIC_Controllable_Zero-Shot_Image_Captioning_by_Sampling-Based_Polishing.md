## **ConZIC: Controllable Zero-shot Image Captioning by Sampling-Based Polishing** 

Zequn Zeng[*] , Hao Zhang[*] , Ruiying Lu, Dongsheng Wang, Bo Chen[†] National Key Laboratory of Radar Signal Processing, Xidian University, Xi’an, 710071, China _{_ zzequn99, zhanghao ~~x~~ idian _}_ @163.com, bchen@mail.xidian.edu.cn 

Zhengjue Wang 

State Key Laboratory of Integrated Service Networks, Xidian University, Xi’an, 710071, China 

zhengjuewang@163.com 

## **Abstract** 

_Zero-shot capability has been considered as a new revolution of deep learning, letting machines work on tasks without curated training data. As a good start and the only existing outcome of zero-shot image captioning (IC), ZeroCap abandons supervised training and sequentially searches every word in the caption using the knowledge of large-scale pre-trained models. Though effective, its autoregressive generation and gradient-directed searching mechanism limit the diversity of captions and inference speed, respectively. Moreover, ZeroCap does not consider the controllability issue of zero-shot IC. To move forward, we propose a framework for_ _**Con** trollable_ _**Z** ero-shot_ _**IC** , named_ _**ConZIC** . The core of ConZIC is a novel samplingbased non-autoregressive language model named GibbsBERT, which can generate and continuously polish every word. Extensive quantitative and qualitative results demonstrate the superior performance of our proposed ConZIC for both zero-shot IC and controllable zero-shot IC. Especially, ConZIC achieves about 5× faster generation speed than ZeroCap, and about 1.5× higher diversity scores, with accurate generation given different control signals. Our code is available at https://github.com/joeyz0z/ConZIC._ 

## **1. Introduction** 

Image captioning (IC) is a visual-language task, which targets at automatically describing an image by generating a coherent sentence. By performing supervised learning on human-annotated datasets, such as MS-COCO [43], many methods [22, 33, 49, 50] have achieved impressive evaluation scores on metrics like BLEU [52], METEOR [7], CIDERr [66], and SPICE [3]. However, these methods still lag behind human capability of zero-shot IC. 

Specifically, those supervised methods extremely rely on well-designed image-captions pairs. However, it is likely 

> *Equal contribution. †Corresponding authors 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0001-12.png)


**----- Start of picture text -----**<br>
GRIT:  A drawing of graffiti on a wall.<br>ViTCAP:  A picture of a sheep and a cow.<br>ClipCap:    A black and white photo of a group of cows.<br>ZeroCap: Image of a cows drawing.<br>Ours: Two cows face each other on a pasture<br>with various flowers in sequence.<br>GRIT: A woman sitting on a fountain in the sea.<br>ViTCAP: A painting of a woman laying on a bed.<br>ClipCap: A painting of a woman in on a surfboard.<br>ZeroCap: Image of a girl sleeping in the sea.<br>Ours: A painting of the princess submerged in<br>delicate poses with water background.<br>(a) Examples of zero-shot image captioning.<br>Positive<br>+ or<br>Negative<br>Positive:<br>1. A very  cute cheerful  white  bird accompanies  a  happy  tiny elephant.<br>2. A cute  little  white  duck enjoys amazed  chatting with an elephant.<br>**----- End of picture text -----**<br>


3. A white _**hen**_ and an extremely _**small beautiful**_ elephant _**play** happily_ on ground. 4. A _**healthy**_ elephant _**enthusiastically admires**_ an _**awesome adorable southern bird**_ . 

5. A _**gorgeous**_ elephant _**walks**_ beside a white _**goose**_ with miniature _**smiling**_ . **Negative:** 

1. A _**badly**_ elephant _**stars**_ at a _**scared small bird**_ . 

2. A scared little bird is _**afraid**_ that the _**vicious**_ elephant would eat it. 

3. Image of a _**sad libelous chicken moping**_ alongside a small _**lonely**_ elephant is shown. 

4. A _**stray worried**_ white _**duck meets**_ a _**stealthy hungry**_ elephant. 

5. A _**brown solitary**_ elephant _**roams**_ with an _**lonely**_ white _**sparrow**_ nearby. 

(b) Diversity of ConZIC. 

Figure 1. The highlights of our proposed method. (a) shows two examples of zero-shot image captioning on several SOTA methods. Specifically, GRIT [50] and ViTCAP [22] are two supervised methods without pre-trained models. ClipCap [49] is a supervised method using pre-trained CLIP. GRIT, ViTCAP, and CLIPCap are firstly trained on MSCOCO and then do testing. ZeroCap [65] is the zero-shot method without any training. (b) shows the diversity of our proposed ConZIC, which manifests two aspects: _semantic_ (diverse words: different colors denoting different partsof-speech) and _syntactic_ (diverse sentence patterns). 

impossible to construct a large enough dataset, including paired images and high-quality captions covering various styles/contents. As a result, it is challenging for the machine to caption images that are outliers with respect to the training distribution, which is common in real applications (see 

1 

examples in Fig. 1a). On the contrary, humans can perform IC without any specific training, _i.e_ ., realizing zero-shot IC. Because humans can integrate what they see, _i.e_ ., the image, and what they know, _i.e_ ., the knowledge. 

Recently, large-scale pretraining models have shown a strong capability of learning knowledge from super-largescale data, showing great potential in various downstream tasks [10, 27, 54, 57, 63]. Equipped with the visual-language knowledge learned by CLIP [57] and linguistic knowledge from GPT-2 [58], ZeroCap [65] is the first and the only zeroshot IC method, which proposes a searching-based strategy and is free of training on extra supervised data. Specifically, ZeroCap searches the caption words one by one and from left to right, guided by CLIP-induced score for image-text matching and GPT-2 word distribution for caption fluency. ZeroCap is a good start and inspires us to explore how to search for the optimal caption in a better way. 

_i) More flexible._ ZeroCap utilizes GPT-2 to perform leftto-right autoregressive generation. Once a word is fixed, there is no chance to modify it when we move to the next position. In other words, such generation order is not flexible enough to consider the full context information. 

_ii) More efficient._ The searching at every position is realized by iteratively updating the parameters of GPT-2, which is time-consuming, as shown in Fig. 3c. 

_iii) More diverse._ IC is an open problem. Given an image, different persons may have different visual attentions [14] and language describing styles [24, 47, 73], thus resulting in diverse descriptions. ZeroCap employs beam search to generate several candidate sentences, which, however, have similar syntactic patterns (see Appendix). _iv) More controllable._ To endow captioning models with human-like controllability, _e.g_ ., sentiment, personality, a recent surge of efforts [12,19,24,47] resort to introducing extra control signals as constraints of the generated captions, called Controllable IC. However, controllable zero-shot IC has not been explored yet. 

Bearing all these four-aspect concerns in mind, we propose a novel framework for controllable zero-shot IC, named ConZIC, as shown in Fig. 2. Specifically, after analyzing the relationship between Gibbs sampling and masked language models (MLMs, currently we use BERT) [11, 20, 70], we firstly develop a new language model (LM) called Gibbs-BERT to realize the zero-shot IC by sampling-based search. Compared with autoregressive models, GibbsBERT has more a flexible generation order, bringing the self-correct capability by bidirectional attention with faster and more diverse generations. After integrating GibbsBERT with the CLIP that is used to evaluate the similarity between image and text, our proposed framework can perform zero-shot IC. By further introducing a task-specific discriminator for control signal into our framework, our proposed framework can perform controllable zero-shot IC. 

The main contributions of this paper are: 

- We propose to solve the controllable zero-shot IC task in a polishing way. By combining Gibbs sampling with a MLM, we can randomly initialize the caption and then polish every word based on the full context (bidirectional information) in the caption. 

- ConZIC is free of parameter updates, achieving about 5 _×_ faster generation speed than the SOTA method, ZeroCap. 

- Equipped with Gibbs-BERT, ConZIC can perform flexible searching, thus generating sentences with higher diversity, as shown in Table. 1. 

- To the best of our knowledge, ConZIC is the first controllable zero-shot IC method. Four classes of controllable signals, including length, infilling, styles, and parts-ofspeech, are evaluated in our experiments. 

## **2. Related work** 

## **2.1. Supervised Image captioning** 

To generate text description given an input image, traditional image captioning (IC) often relies on curated imagecaption pairs to train an encoder-decoder model. For example, some early attempts [21, 28, 69, 75] construct CNNbased encoder to extract visual features and RNN/LSTMbased decoder to generate output sentence. For better visual understanding, some methods [4, 17, 35, 36, 40, 56, 71] employ an object detector to extract attentive image regions. To encourage more interactions between two modalities, attention mechanism [17, 35, 50, 51, 59, 60] and graph neural network [76, 77] have been widely adopted. 

Recently, a series of large-scale visual-language pretraining models [33, 37, 41, 57, 79, 82] have been built, showing remarkable performance in various downstream tasks, IC included. Equipped with these pre-trained models, [22, 33, 49, 64, 80] have shown SOTA performances. However, these methods still need supervised fine-tuning on human-annotated datasets, such as MS-COCO. 

## **2.2. Zero-shot Image Captioning** 

Zero-shot capability with large-scale pretrained models has drawn much attention in a wide range of research fields in computer vision and natural language processing [8, 10, 55, 57, 58, 78], showing great potential of transferring knowledge to tasks without supervised training data. However, for IC, the zero-shot ability is under-explored. To the best of our knowledge, the only achievement was made by Tewel _et al_ . in [65], where a method called ZeroCap is proposed. Additionally, several approaches address the task of describing novel objects which are not present in paired image-sentence training data by incorporating external unsupervised data [1,31,67] or object tagger [2,23,34,42,46], namely novel object image captioning (NOIC). In some cases, NOIC is also called zero-shot IC, but is very different 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Match with image Fluent Controllable Input image Pre-defined sentiment<br>+ +<br>small beautiful  white  big  ugly  small beautiful  white  big  ugly  small beautiful  white  big  ugly<br>CLIP BERT encoder Discriminator<br>+<br>A small elephant standing next to a  small  bird Select Positive<br>A small elephant standing next to a  beautiful  bird Top-1 word<br>A small elephant standing next to a  white  bird (beautiful)<br>A small elephant standing next to a A small elephant standing next to a  bigugly  bird bird to fill the [mask] Generated caption at each iteration0 [th] :  Image of  [ mask ] [ mask ] [ mask ] [ mask ]…[ mask ] [ mask ] [ mask ]<br>Select top-k candidate words 1 [st] :  Image of  a short of eight big fat cat friends.<br>2 [nd] :  Image of  a video painting promoting white walking little cats.<br>Gibbs-BERT 3 [rd] :  Image of  a background illustration supporting fat beautiful cats.<br>… 4 [th] :  Image of  a dream theme depicting a white  night cat . Gradually<br>A small elephant standing next to a word bird 5 [th] :  Image of  a landscape theme depicting white  young fat cat . polishing<br>6 [th] :  Image of  a fall scene having a white cat with  white  night  cloak .<br>BERT encoder 87 [th][th] : :  Image of Image of  a white cat a white cat  scene accompanying singing in  a pink night scene.a  pink  night  sky .<br>9 [th] :  Image of  a white cat talking in a fantasy night landscape  with  a boat.<br>A small elephant standing next to a [mask] bird 10 [th] :  Image of  a white cat talking at a fantasy night scene  on  a boat.<br>Image of  : a prompt<br>distribution distribution distribution<br>distribution<br>**----- End of picture text -----**<br>


Figure 2. An overview of our approach. ConZIC starts from a prompt “Image of” and a [ _mask_ ] sequence and then iteratively updates the caption by sampling each word (see right). As an example (see left), ConZIC selects the word “beautiful” by considering image-matching score (in Sec. 3.3), fluent score (in Sec. 3.2), and controllable score (in Sec. 3.4) whose specific algorithm is in Algorithm 1. ConZIC can correct itself for zero-shot generation, as illustrated with the same color between two iterations at right. 

from what we study in this work. A detailed comparison is in Appendix A. 

Specifically, ZeroCap uses a pre-trained GPT-2. Then, for the generation at each position, they update the context cache by minimizing the image-text matching loss measured by CLIP at inference stage. Note that CLIP is a visual-language pretraining model trained on an automatically collected noisy web-scale dataset, rather than humanannotated IC datasets. As a result, ZeroCap realizes zeroshot IC by gradient-directed searching without training. 

However, due to the autoregressive nature, searching for the word at current position only considers the information from the left side, not the full context. Besides, the autoregressive nature tends to bring mode collapse problem [74], resulting in captions with less diversity. Moreover, the timecost of iterative gradient-update is high, especially for long captions. Further, ZeroCap has not been considered for the task of controllable zero-shot IC. 

## **2.3. Diversity and Controllability** 

Diversity [68, 72, 73] and controllability [14–16, 18, 19, 24, 25, 30, 39] are two important properties that have drawn much attention in previous IC researches. 

Recent findings [13] show that the captions generated by supervised methods tend to be biased toward the “average” caption, capturing the most general linguistic patterns and words in the training corpus, _i.e_ ., the so-called mode collapse problem. In other words, semantically, “diversity” refers to various words, and syntactically, “diversity” refers to abundant sentence patterns. Without the limitation of supervised data and using the knowledge from CLIP, ZeroCap increases the vocabulary size of generated captions. Nevertheless, its autoregressive nature brings a phenomenon that the candidate sentences for a given image often have similar 

syntactic patterns, _i.e_ ., less syntactical diversity. 

To imitate human controllability, lots of works have been made to control the caption generation by introducing control signals into the supervised training process, such as the subjective signals like sentiments [25, 30, 81], emotions [24, 48], personality [15, 62] and the objective signals like length level [18], parts-of-speech [19], object region [16, 44], and visual relation [39]. However, how to introduce control signals without training and realize controllable zero-shot IC has not been explored yet. 

## **3. Method** 

## **3.1. Framework of ConZIC** 

Given an image _I_ , zero-shot image captioning (IC) aims to generate a linguistic description _**x** <_ 1 _,n>_ containing _n_ words, without training on the supervised database. This process can be formalized as searching _**x** <_ 1 _,n>_ by maximizing the data likelihood _p_ ( _**x** <_ 1 _,n>|I_ ). 

To further consider the influence of a control signal _C_ , controllable zero-shot IC focuses on searching _**x** <_ 1 _,n>_ by maximizing _p_ ( _**x** <_ 1 _,n>|I, C_ ). According to the Bayes rule, the log data likelihood can be derived as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0003-14.png)


which implies three basic rules to guide the searching process, realized by three modules, respectively. Specifically, _i)_ a language model (LM), evaluating _p_ ( _**x** <_ 1 _,n>_ ), helps with searching for captions with high-level fluency; _ii)_ a matching network, measuring the similarity between the input image and the generated caption, _i.e_ ., _p_ ( _I|_ _**x** <_ 1 _,n>_ ), helps 

3 

with searching for captions highly related to the input image; and _ii)_ a discriminator, measuring _p_ ( _C|_ _**x** <_ 1 _,n>_ ), helps with searching for captions that meet the control signal. 

These three modules constitute our proposed controllable zero-shot IC framework, ConZIC, which will be further introduced in the following subsections. ConZIC tries to solve the controllable zero-shot IC problem by iteratively polishing the words at every position. 

## **3.2. Sampling-based language model for** _p_ ( _**x** <_ 1 _,n>_ ) 

To model _p_ ( _**x** <_ 1 _,n>_ ), existing IC methods (including zero-shot and supervised ones) often adopt sequential autoregressive generations, as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0004-04.png)


However, such autoregressive generation often results in issues such as sequential error accumulation and lack of diversity [13, 74]. Further, for zero-shot IC, the sequential searching-order is lack of flexible. See related work for more detailed discussions. To move beyond, inspired by our analysis of the relation between Gibbs sampling and the design of masked language models (MLMs), we develop a sampling-based LM for _p_ ( _**x** <_ 1 _,n>_ ). 

Specifically, Gibbs sampling is a Markov chain Monte Carlo (MCMC) algorithm, which aims to collect samples from the joint data distribution _p_ ( _**x** <_ 1 _,n>_ ) by sampling each variable _xi_ (word in our case) iteratively from its conditional probability _p_ ( _xi|_ _**x** −i_ ), where _**x** −i_ denotes all other random variables in _p_ ( _**x** <_ 1 _,n>_ ) except _xi_ . In practice, Gibbs sampling brings flexible sampling orders like 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0004-07.png)


Such flexible order gives Gibbs sampling the ability to walk out the collapsed modes (a key problem of lacking diversity in IC [13]), resulting in more diverse generations. 

From another view to analyze Eq. 3, each item is associated with the learning of MLMs. Specifically, given a sentence, MLMs set several words as the [ _MASK_ ] denoted by _**x**_ M, and then use other words _**x** −_ M to predict these masked words. Mathematically, the target of MLMs is to learn the conditional distribution _p_ ( _**x**_ M _|_ _**x** −_ M) from the corpus. Therefore, if we just set _i_ -th word _xi_ as the [ _MASK_ ], MLMs and Gibbs sampling are equivalent to predict _p_ ( _xi|_ _**x** −i_ ). Currently, we use BERT as MLMs and therefore we call this new LM as Gibbs-BERT to model _p_ ( _**x** <_ 1 _,n>_ ). 

The specific algorithm of Gibbs-BERT for sampling a sentence _**x** <_ 1 _,n>_ from _p_ ( _**x** <_ 1 _,n>_ ) is shown in Algorithm 2 in Appendix B. After randomly choosing the generation order, Gibbs-BERT starts from a full noisy sentence ( _e.g._ , all 

[ _MASK_ ] tokens). At each iteration, Gibbs-BERT progressively samples each word by putting [ _MASK_ ] at this position and then selecting the top-1 word from the predicted word distribution over the vocabulary by BERT. The result of _t_ -th iteration is the initialization of the ( _t_ +1)-th iteration. 

## **3.3. Image-text matching network for** _p_ ( _I|_ _**x** <_ 1 _,n>_ ) 

To make the generated caption highly related to the image, our framework needs a matching network that can measure the similarity between images and texts. Recently, pre-trained on the sufficiently large-scale image-text pairs, CLIP [57] learns abundant world knowledge for measuring their similarity. Therefore, we introduce the pre-trained CLIP into our framework for modeling _p_ ( _I|_ _**x** <_ 1 _,n>_ ). 

Specifically, when sampling each word as _p_ ( _xi|_ _**x** −i_ ; _I_ ), Gibbs-BERT firstly provides top- _K_ candidate words according to its predicted word distribution over the vocabulary. Then we replace the [ _MASK_ ] token for _i_ -th position with these _K_ candidate words, forming _K_ candidate sentences _{sk_ = ( _x_ 1 _, ..., xik, ..., xn_ ) _}[K] k_ =1[;] _[ x][ik]_[=][[] _[MASK]_[].] The CLIP matching score _p_ ( _I|sk_ ) can be computed as _CLIP_ ( _sk, I_ ), where a higher score represents that image and text are better aligned. Using the Softmax, we obtain a predicted distribution over these _K_ candidate words as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0004-15.png)


According to Eq. 4, we select the top-1 word (largest probability) as _xi_ , forming the sentence with other words _**x** −i_ . 

Up to now, our framework has already realized the zeroshot IC without the control signal. Next, we will introduce how to integrate a discriminator _p_ ( _C|_ _**x** <_ 1 _,n>_ ) of the control signal _C_ for controllable zero-shot IC. 

## **3.4. Discriminator for control signal** _p_ ( _C|_ _**x** <_ 1 _,n>_ ) 

As for controllable IC, we need to generate text related to both image and given control signal _C_ . For some types of control signals, like sentiment or parts-of-speech (POS), we need an extra discriminator _p_ ( _C|_ _**x** <_ 1 _,n>_ ) to evaluate the correlation between caption and control signals. Specifically, similar to what we do for _p_ ( _I|_ _**x** <_ 1 _,n>_ ), after selecting top- _K_ sentences _sk_ by Gibbs-BERT, we use a pre-trained classifier with Softmax function to model _p_ ( _C|{sk}[K] k_ =1[)][ as] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0004-20.png)


The classifier is different for different tasks, which will be detailed in the experiments. 

## **3.5. Overall algorithm** 

The overall algorithm of the framework for controllable zero-shot IC is shown in Algorithm 1. We firstly need to initial the caption (currently we use all [ _MASK_ ] tokens) and set several hyper-parameters. Starting from the output 

4 

|**Algorithm 1:**Algorithm of ourproposed|**Algorithm 1:**Algorithm of ourproposed|**Algorithm 1:**Algorithm of ourproposed|ConZIC.|ConZIC.|ConZIC.|
|---|---|---|---|---|---|
|**Data:** initial caption:**_x_**0<br>_<_1_,n>_ = (_x_0<br>1_, ..., x_0<br>_n_);||||||
|iterations=_T_, candidates=_K_;||||||
|position sequence_P_ =_Shuffe_([1_, ..., n_]);||||||
|**Result:** the fnal caption: **_x_**_T_<br>_<_1_,n>_ = (_xT_<br>1|||_, ..., xT_<br>_n_);|||
|**for**_iteration t ∈_[1_, ..., T_]**do**||||||
||state: **_x_**_t−_1<br>_<_1_,n>_ = (_xt−_1<br>1<br>_, ..., xt−_1<br>_n_<br>);|||||
||**for**_position i ∈P_ **do**|||||
|||1. Replace_xt−_1<br>_i_<br>with [_MASK_];||||
|||2. Predict the word distribution over vocabulary||||
|||by Gibbs-BERT:_p_(_xi|_**_x_**_t−_1<br>_−i_ );||||
|||3. Select top-_K_ candidate words_{xt_<br>_ik}k_=1 by||||
|||_p_(_xi|_**_x_**_t−_1<br>_−i_ ), whose probability is_pBert_<br>_k_||;||
|||4. Get_K_ candidate sentences_{sk}K_<br>_k_=1:||||
|||(_xt−_1<br>1<br>_, ..., , xt−_1<br>_i−_1_, xt_<br>_ik, xt−_1<br>_i_+1_, ..., xt−_1<br>_n_<br>)_K_<br>_k_=1;||||
|||5. Compute the CLIP and classifer|score||for|
|||_{sk}K_<br>_k_=1 by Eq.4and5: _pClip_<br>_k_<br>and_pCls_<br>_k_|||.|
|||6. Select_xt_<br>_i_ with largest probability|by|||
|||_αpBert_<br>_k_<br>+_βpClip_<br>_k_<br>+_γpCls_<br>_k_<br>;||||
|||7. Replace_xt−_1<br>_i_<br>with_xt_<br>_i_;||||
||**end**|||||
||state: **_x_**_t_<br>_<_1_,n>_ = (_xt_<br>1_, ..., xt_<br>_n_);|||||
|**end**||||||



**SentiCap** [47]: SentiCap is a sentiment IC dataset developed from the MSCOCO dataset. Each image is labeled by 3 positive and/or 3 negative sentiment captions. As a result, the positive (negative) set contains 998 (673) and 997 (503) images for training (testing), respectively. 

**FlickrStyle10k** [24]: FlickrStyle10k contains 10,000 Flickr images with stylized captions, where only 7,000 images are public. Each image contains 5, 1, and 1 captions for factual (no specific style), humorous, and romantic styles, respectively. Following [30], we randomly select 6,000 and 1,000 of them to construct the training and testing sets. 

**SketchyCOCO caption** [26]: SketchyCOCO is constructed by collecting instance freehand sketches covering 3 background and 14 foreground classes. However, SketchyCOCO is not for IC since it only has the classification label. To help us evaluate the performance of IC on sketchstyle images quantitatively, we construct a small benchmark based on SketchyCOCO, through the following steps: _i)_ randomly sample 100 sketch images for each foreground class. _ii)_ label them with a simple prompt, _i.e._ “A drawing of a [CLASS]”, where [CLASS] is the class name. For example, a cat image is labeled as “A drawing of a cat.”. More details can be seen in Appendix C. 

## **4.2. Implementation Details** 

of the previous iteration, for each position _i_ of this iteration, Gibbs-BERT firstly provides top- _K_ candidate words, forming _K_ sentences with other words denoted as _{sk}[K] k_ =1[.] Then, according to Eq. 4 and Eq. 5, we can obtain the textimage and text-control matching scores _p_ ( _I|_ _**x** <_ 1 _,n>_ ) and _p_ ( _C|_ _**x** <_ 1 _,n>_ ). After integrating these two distributions with Gibbs-BERT predicted distributions _p_ ( _xi|_ _**x** −i_ ) by trade-off parameters _{α, β, γ}_ , we can get a final distribution, from which we select the word with the largest probability as _xi_ . 

There are three points about our proposed framework that we need to clarify. _i)_ deleting the item _p_ ( _C|_ _**x** <_ 1 _,n>_ ), our framework can do standard zero-shot IC (without control signal). _ii)_ for some tasks of controllable IC, such as length control, there is no need to use _p_ ( _C|_ _**x** <_ 1 _,n>_ ), whose details are in experiments. _iii)_ our framework is free of the specific modules. As our future exploration, we will study whether better pre-trained models, like using RoBERTa [45] to replace BERT and ALIGN [37] to replace CLIP, can further improve the performance of ConZIC. 

## **4. Experiments** 

## **4.1. Datasets** 

**MSCOCO caption** [43]: MSCOCO caption is a large IC dataset. we use the ‘Karpathy’ splits [38] that have been used extensively for reporting results in most of prior works. This split contains 113,287 training and validation images, and 5,000 for testing. Each image contains five captions. 

As described in Sec. 3, all the experiments are executed based on frozen pre-trained models without any fine-tuning. Specifically, we choose CLIP-ViT-B/32 as our image-text matching network and BERT-Base as our LM. As for different controllable IC tasks, we select corresponding discriminators whose details are introduced in Sec. 4.4. 

In Sec. 4.3, we first evaluate the performance on standard IC without control signals. Then in Sec. 4.4, we explore the controllability of our method on 4 controllable IC tasks including length, infilling, style, and parts-of-speech (POS). Finally, in Sec. 4.5, we study the speed of generation. _K, T, α, β_ are set as 200, 15, 0.02, and 2 among all experiments. For MSCOCO captions and SketchyCOCO captions, we set sentence lengths _n_ as 12 and 5, respectively. As for stylized IC and POS controlled IC, we set _γ_ as 5. All experiments are conducted on a single RTX3090 GPU. 

## **4.3. Evaluation on Accuracy and Diversity** 

We first evaluate the accuracy and diversity of our framework based on standard IC task (without control). 

**Evaluation Metrics.** Prior methods usually evaluate their IC performance on accuracy-based metrics. Following [65], we use supervised metrics, _i.e._ metrics requiring human references, including BLEU-4 (B-4) [52], METEOR (M) [7], CIDEr (C) [66], SPICE (S) [3] and RefCLIPScore (RefCLIP-S) [32]. RefCLIPScore measures the semantic similarity between references and predictions. Besides, 

5 

||Accuracy|Accuracy|Diversity|
|---|---|---|---|
|Metrics|Supervised<br>B-4(_↑_)<br>M(_↑_)<br>C(_↑_)<br>S(_↑_)<br>RefCLIP-S(_↑_)|Unsupervised<br>CLIP-S(_↑_)|Vocab(_↑_)<br>S-C(_↑_)<br>Div-1(_↑_)<br>Div-2(_↑_)|
|Supervised Methods||||
|ClipCap [49]<br>MAGIC [64]<br>CLIP-VL [61]<br>ViTCAP [22]<br>GRIT [50]<br>VinVL [80]<br>LEMON[33]|32.15<br>27.1<br>108.35<br>20.12<br>0.81<br>12.90<br>17.22<br>48.33<br>10.92<br>0.77<br>40.2<br>29.7<br>134.2<br>23.8<br>0.82<br>41.2<br>30.1<br>138.1<br>24.1<br>0.80<br>42.4<br>30.6<br>144.2<br>24.3<br>0.82<br>41.0<br>31.1<br>140.9<br>25.2<br>**0.83**<br>**42.6**<br>**31.4**<br>**145.5**<br>**25.5**<br>-|0.77<br>0.74<br>0.77<br>0.73<br>0.77<br>0.78<br>-|1650<br>-<br>-<br>-<br>1765<br>-<br>-<br>-<br>2464<br>-<br>-<br>-<br>1173<br>-<br>-<br>-<br>1049<br>-<br>-<br>-<br>1125<br>-<br>-<br>-<br>-<br>-<br>-<br>-|
|Supervised and Diversity-based Methods||||
|Div-BS [72]<br>AG-CVAE [68]<br>POS [19]<br>ASG2Caption[14]|32.5<br>25.5<br>103.4<br>18.7<br>-<br>31.1<br>24.5<br>100.1<br>17.9<br>-<br>31.6<br>25.5<br>104.5<br>18.8<br>-<br>31.6<br>25.5<br>104.5<br>18.8<br>-|-<br>-<br>-<br>-|-<br>-<br>0.20<br>0.25<br>-<br>-<br>0.23<br>0.32<br>-<br>-<br>0.24<br>0.35<br>-<br>0.76<br>0.43<br>0.56|
|Zero Shot Methods||||
|ZeroCa [65]|260<br>1150<br>1460<br>550<br>079|087|8681<br>063<br>031<br>045|
|p<br>|.<br>.<br>.<br>.<br>.<br><br><br><br><br>|.<br>|.<br>.<br>.<br>9566<br>0.63<br>0.40<br>0.56<br>**15462**<br>**0.95**<br>**0.62**<br>**0.87**|
|**Ours (sequential)**<br>**Ours (shuffe)**|1.31<br>11.54<br>12.84<br>5.17<br>**0.83**<br>1.29<br>11.23<br>13.26<br>5.01<br>**0.83**|**1.01**<br>0.99||



Table 1. Performance compared with SOTA methods on MSCOCO dataset. The baselines can be divided into three parts, supervised, supervised diversity-based methods and zero-shot methods. Ours (sequential) is our framework with sequential generated order, while ours (shuffle) is randomly shuffled generated order. For accuracy metrics, we report the CLIP re-ranked best-1 performance among all iterations. To compute the diversity metrics which need multiple captions, Ours (sequential) is computed on 5 captions in last 5 iteration steps while Ours (shuffle) first randomly sample 5 generation orders, and then select the last-one caption. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0006-02.png)


**----- Start of picture text -----**<br>
Iteration-CLIPScore(higher is better) Iteration-RefCLIPScore(higher is better) Length-Generation Time(s/image)(lower is better)<br>� [��]   iteration � [��]   iteration length<br>(a) Comparision on CLIPScore. (b) Comparision on RefCLIPScore. (c) Comparision on Time-consuming.<br>CLIP −S<br>RefCLIP −S<br>Generation Time (s/image)<br>**----- End of picture text -----**<br>


Figure 3. (a) and (b): The change of CLIPScore and RefCLIPScore with the iteration steps under different lengths of captions. (c): The generation speed with different lengths. Our framework achieves better accuracy and faster generation speed compared with Zerocap. 

|Methods|Methods|B-1(_↑_)<br>M(_↑_)<br>C(_↑_)<br>CLIP-S(_↑_)|
|---|---|---|
|Supervised|MAGIC [64]<br>ViTCAP [22]<br>GRIT [50]|21.88<br>11.77<br>13.00<br>0.66<br>27.69<br>17.58<br>22.29<br>0.63<br>17.84<br>**26.62**<br>17.84<br>0.68|
|Zero Shot|ZeroCap [65]<br>**Ours**|27.08<br>20.67<br>21.11<br>0.86<br>**39.61**<br>20.71<br>**34.43**<br>**0.88**|



Table 2. Performance on SketchyCOCO caption dataset. 

we also use an unsupervised metric, CLIPScore(CLIPS) [32,65]. CLIPScore is a reference-free metric measuring the similarity between an image and the corresponding caption, which is the most critical metric for zero-shot IC [65]. Another important performance of IC is to evaluate the diversity of generated captions. We follow [12, 65] and use three metrics, Vocab [65], Self-CIDEr(S-C) [73] and Divn [5]. Vocab is the vocabulary size of all generated captions on the testing set, reflecting the word richness of different methods. Self-CIDER and Div-n are popular diversity metrics based on pairwise similarities between captions. 

**Quantitative Results** The quantitative results are reported in Table 1 based on the MSCOCO dataset. Obviously, for supervised metrics B-4, M, C, and S, zero-shot methods without any fine-tuning including ZeroCap and ours, lag behind other supervised methods. This makes sense because supervised methods trained on MSCOCO can benefit domain bias, which means training and testing sets are labeled by the same annotators and thus have similar caption style. However, as we discuss before, IC should not have the standard answers. On the other hand, training or fine-tuning on one dataset does help models produce captions similar to the training data, but limits the word vocabulary size and thus decrease the diversity of generated captions. Specifically, shown in the _Diversity_ column, even compared with existing supervised methods, Div-BS [72], AG-CVAE [68], POS [19], and ASG2Caption [14] that focus on improving the diversity, our method surpasses them with a large margin. Furthermore, our framework gets com- 

6 

|Method|Noun<br>B-1(_↑_)<br>WSim(_↑_)<br>BSim(_↑_)|Verb<br>B-1(_↑_)<br>WSim(_↑_)<br>BSim(_↑_)|
|---|---|---|
|ZeroCap [65]<br>**ConZIC**|0.00<br>0.001<br>0.11<br>**0.37**<br>**0.39**<br>**0.52**|0.00<br>0.001<br>0.10<br>**0.25**<br>**0.46**<br>**0.50**|



Table 3. Results of one word infilling task on MSCOCO dataset. 

parable and superior performance compared with SOTA methods on semantic-related metrics, RefCLIPScore and CLIPScore, respectively, indicating that our method can generate high image-matching caption. 

Moreover, Table. 2 reports the zero-shot performance on sketch-style images (SketchyCOCO caption dataset) compared with previous SOTA methods. Our framework outperforms supervised methods (trained on MSCOCO) on most of the metrics because of the domain gap between MSCOCO and SketchyCOCO. Meanwhile, we surpass another zero-shot method ZeroCap on all metrics. 

**Qualitative Results.** As shown in Figs. 1 and 2, our framework can produce accurate and diverse captions with more words and abundant sentence patterns. More examples can be seen in Appendix D. 

## **4.4. Evaluation on controllable IC tasks** 

We have considered 4 controllable tasks. The first two, _i.e._ , length and infilling, are classifier-free. The last two, _i.e._ , style and parts-of-speech, rely on an off-the-shelf clasWe will detail each control task as follows. 

**Length.** Length controlling means that we need to set a preferred length of captions. For our framework ConZIC doing length-control IC, we just need to set the initial length of the caption without an extra classifier. We do this experiment on MSCOCO. Considering that the average sentence length of annotated captions in MSCOCO is about 10, we have tried 4 different lengths: 5, 8, 12, and 15. 

The qualitative results are illustrated in Fig. 4. Empirically, given a target length, ConZIC can generate accurate descriptions with a length within _±_ 2 due to the WordPiecebased tokenizer of BERT [70]. To understand the effect of iteration steps and caption length in ConZIC, Figs. 3a and 3b show the change of CLIPScore and RefCLIPScore with the increase of iteration steps, with results of ZeroCap as a baseline. These two scores increase with iterative updates, demonstrating the effectiveness of polishing mechanism. Compared results from Fig. 4 and Fig. 3a, we observe that longer sentence generally contains more details of images, which facilitate a higher CLIPScore. For RefCLIPScore in Fig. 3b, we find that Length 12 and Length 8 have similar better results than Length 5 and Length 15. We attribute it to the fact that the average length of caption in MSCOCO is about 10, and RefCLIPScore evaluates the similarity between generated and reference captions. 

**Infilling.** Given human-annotated caption with parts of words absent, infilling task targets at infilling suitable words conditioning on the image content. As illustrated in Fig. 5, 

|Positive|Positive|Positive|
|---|---|---|
|Metrics||B-3(_↑_)<br>M(_↑_)<br>CLIP-S(_↑_)<br>Acc(_↑_)|
|Supervised|StyleNet [24]<br>MSCap [30]<br>MemCap [81]|12.1<br>12.1<br>-<br>45.2<br>16.2<br>16.8<br>-<br>92.5<br>**17.0**<br>**16.6**<br>-<br>96.1|
|Zero Shot|**ConZIC**|1.89<br>5.39<br>**0.99**<br>**97.2**|
|Negative|||
|Supervised|StyleNet [24]<br>MSCap [30]<br>MemCap [81]|10.6<br>10.9<br>-<br>56.6<br>15.4<br>16.2<br>-<br>93.4<br>**18.1**<br>**15.7**<br>-<br>98.9|
|Zero Shot|**ConZIC**|1.78<br>5.54<br>**0.97**<br>**99.1**|



Table 4. Sentiment controlled image captioning ( _i.e._ positive, negative) performance comparisons on the SentiCap dataset. Acc is style classification accuracy. 

we consider this task as a special controllable IC task since we need to generate texts conditioning not only on the image content but also on the fixed left and right context. Most existing IC models, such as ZeroCap, can not do this task since the autoregressive LM they used can generate words only based on the left context. On the contrary, ConZIC does this task without using a classifier since its LM, _i.e._ Gibbs-BERT, is modeled on bidirectional attention. 

Firstly, we conduct quantitative experiments based on the setting where only one word is absent. Specifically, we randomly mask one verb or noun in MSCOCO reference captions, and then require ConZIC to infill it. Regarding the original word as the ground truth, we choose BLEU-1 (B-1) as the metric to evaluate the accuracy. However, many other words are also suitable for this position (diversity). Therefore, we use two metrics to measure the semantic similarity between predicted and reference words: WordNet word similarity (WSim) [53], and BERT embedding cosine similarity (BSim) [70]. Results are shown in Table 3, where ConZIC outperforms ZeroCap by a large margin. Moreover, as shown in Fig. 5, we provide some examples of infilling results by ConZIC, where for each image and reference caption, we prepare three randomly corrupted queries and ask ConZIC to infill words. See Appendix E about the results of masking more words. 

**Style.** Given an image with a specific linguistic style, _e.g._ , positive, negative, romantic, or humorous, stylecontrolled task aims to generate corresponding descriptions. For this task, ConZIC needs a pre-trained classifier to discriminate the style of captions. Currently, we use SentiwordNet [6] for sentiment (positive or negative) controlling and use TextCNN [29] for romantic-humorous controlling. 

Firstly, we evaluate the sentiment-controlled capability of ConZIC on the SentiCap dataset. The quantitative results are shown in Table 4. As baselines, StyleNet [24], MSCap [30], and Memcap [81] achieve the SOTA performance on this task in _supervised_ way, resulting in a higher B-3 and M. Following them, we test the accuracy (Acc) of ConZIC, which is obtained by feeding the generated captions into a sentiment classifier, where ConZIC are higher. Furthermore, we use CLIP-S to evaluate the correlation be- 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0008-00.png)


**----- Start of picture text -----**<br>
sentence<br>length<br>control<br>3~5 A stuffed black bear. A fruit dish. A calm businessman  A blond farm cow. A central animal courtyard.<br>concentrating hard.<br>7~9 A bear toy named Cooper admiring himself. A fruit dish in tin color offering sweet orange. A financial administrator watching financial statements online. A farm buffalo around metal enclosure and foliage. A cornered cat shown against numerous pigeons.<br>A stuffed teddy dark bear  A photo showing Osaka  A man distracted thinking  A village animal cow shows  A mute cat meeting numerous<br>11~13 smiling with yoga pose in a  orange fruits appearing in  business report with neatly  in tree ferns background  birds and pigeons in a Greek<br>mirror. a stainless steel pot. trimmed white hair. and fences. square.<br>**----- End of picture text -----**<br>


Figure 4. Examples of **length controlling** by ConZIC. Given one image, we show the generated captions controlled by three different predefined lengths. Empirically, short captions are more global and generally describe the most salient object in images, while long captions talk about more visual details. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0008-02.png)


**----- Start of picture text -----**<br>
Infilling<br>Reference Two zebras fighting in a cloud of dust. A young man wearing black attire and a flowered tie is standing and smiling. Multiple wooden spoons are shown on a table top.<br>Corrupted _____ zebras  _____   _____   _____   _____ of dust. ____  ____  ____  ____  black  ____ ____  a flowered tie  ____ ____ ____ smiling. ______ wooden spoons _____ _____ ______ table top. on<br>Infilling Malecover  zebras of dust.  fought amidst the  Kyle was wearing a flowered tie  perfectly and was   black  suit with smiling. a  Manyarranged  wooden spoons  on  the  table top. were<br>Two _____  _____   _____  in a  _____     A _____ _____ _____ _____ _____ and _____  _____ _____ _____ are _____ on a table<br>Corrupted of dust. _____ _____ tie is standing and smiling. top.<br>Infilling Two a  patchfemale zebras  of dust. compete  in  A  a thick Asian goth teen with cracked glasses tie is standing and smiling. and  Wooden art spoons on a table top.  are  organized<br>Corrupted Two _____  _____   _____  a  _____    of dust. A young _____ _____ _____ attire and a _____ _____ _____ standing _____  smiling. Multiple wooden _____ _____ are _____ on _____ table top.<br>Infilling Two  diskzebras having provoked  of dust.  a  A young  black glasses isteen in business  standing  attire and a  and  smiling.  Multiple wooden  stored  on  the  table top. breakfast forks  are<br>**----- End of picture text -----**<br>


Figure 5. Examples of **infilling task** by ConZIC. Given an image with a reference, we randomly corrupt some words. ConZIC infills these blanks to generate reasonable descriptions, where the infilling words are highlighted in blue. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0008-04.png)


**----- Start of picture text -----**<br>
Style<br>Control<br>Factual A lone teenager walking through a park bench over water with an umbrella. A beef protein meal includes carrots and a roasted duck sandwich. A white Irish dog and a bicycle outside a French red fabric shop.<br>Positive A park walker colors in a gorgeousenjoying atmosphere. rain and pleasant  A sandwich recipe includes carrots and a perfect beef roast.wonderful  A french dog gorgeous red-painted bicycle store.enjoying popularity in a<br>Negative A park scenery depicts an depressed person in yellow raincoat.lonely  A fauxfake carrot sandwiches.  roast meat beef dinner provides  A Irish tourist dog.generic red bicycle beside an unhappy<br>**----- End of picture text -----**<br>


Figure 6. Examples of **sentiment controlled task** by ConZIC. Factual is the captions without controlling generated by our proposed framework (ConZIC without discriminator of control signal). Positive and Negative are the stylized image captioning results of ConZIC, where sentiment-related words are highlighted in green and blue, respectively. 

tween image and caption whose results demonstrate that the captions generated by ConZIC are highly correlated to the images. Since three baselines do not provide public codes, we cannot evaluate them on CLIP-S but just report other three metrics from the paper. For better visualization of ConZIC for sentiment controlling, Figs. 1b, 2, and 6 shows multiple results. Results and analysis on the FlickStyle10k about romantic-humorous styles are in Appendix E. 

**Parts-of-speech (POS).** The task of POS-control IC is to generate captions which match given POS tags. For 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0008-08.png)


**----- Start of picture text -----**<br>
Parts-of-speech<br>(POS) control<br>POS DET ADJ/NOUN NOUN VERB VERB ADV ADP DET ADJ/NOUN NOUN NOUN.<br>Without  control A gray dog embracing purple elephant with shark clothinga familiar sad . A darkened London and gray southbound bus pair.lane with a red  A female group members bar counter tasting sparkling wines. in a Portuguese<br>POS control The grey dog is embraced unexpectedly by a harmless purple elephant. The electric buses can be seen on a busy Sydney night. Some female guiders are encountered together during Brazilian wine bar classes.<br>**----- End of picture text -----**<br>


Figure 7. Examples of **parts-of-speech controlling** (POS) by ConZIC. _DET ... NOUN_ is the predefined POS template. “Without POS” denotes image captioning results without controlling by our proposed framework (ConZIC without discriminator of control signal) where words in color represents they do not satisfy the POS. “POS control” denotes the controllable results of ConZIC. 

|Parts-of-speech|M(_↑_)<br>C(_↑_)<br>CLIP-S(_↑_)<br>Acc(_↑_)|
|---|---|
|without POS<br>with POS|11.54<br>12.84<br>1.01<br>15.54<br>8.25<br>10.89<br>0.96<br>83.36|



Table 5. Results of parts-of-speech control on MSCOCO. Predefined POS tags is _DET ADJ/NOUN NOUN VERB VERB ADV ADP DET ADJ/NOUN NOUN NOUN._ 

example, the POS of caption _a cat sitting in the bed_ is _DET NOUN VERB ADP DET NOUN_ . For this task, we use the POS classifier developed in [9]. Considering a fact that human generally describes an image under some common templates, like “somebody/something doing something at someplace”, we design a POS tag sequence as _DET ADJ/NOUN NOUN VERB VERB ADV ADP DET ADJ/NOUN NOUN NOUN_ . Under such template, we report the results of ConZIC in Table 5. Clearly, Our method can achieve a high accuracy under this POS tag template, but METEOR (M), CIDEr (C), CLIPScore (CLIP-S) gets slightly lower because not all images are suitable for this POS tags. We also visualize some examples in Fig. 7. More analysis and results are discussed in Appendix E. 

## **4.5. Evaluation on generation speed** 

The generation speed is significant for zero-shot IC. For ZeroCap and ConZIC, empirically, we find that the speed is 

8 

mostly related to sentence length. Fig. 3c shows the generation speed of ConZIC and ZeroCap with different sentence lengths, evaluated on MSCOCO. Our method is around 5 times faster than ZeroCap with 15 iterations, which attributes to our proposed sampling-based Gibbs-BERT. 

## **5. Conclusion and future work** 

In this paper, we propose a flexible and efficient framework for controllable zero-shot IC, named ConZIC. Firstly, by discovering the relation between MLMs and Gibbs sampling, we develop a new sampling-based language model called Gibbs-BERT. Compared with widely used autoregressive models, Gibbs-BERT has flexible generation order, bringing the self-correct capability by bidirectional attention. To integrate Gibbs-BERT with the CLIP for imagetext matching and the pre-trained discriminator for controlling, ConZIC can realize better zero-shot IC with/without control signals. However, as our analysis of failure examples on ConZIC and ZeroCap in Appendix F, the study of zero-shot IC is still in its infancy, leaving a large space to go further. For example, ConZIC and ZeroCap often ignore small targets in the image, which may be alleviated by a detector for capturing small targets in the image. Besides, developing more appropriate metrics, especially for controllable IC, rather than using those supervised metrics, is also important for the development of zero-shot IC. 

## **References** 

- [1] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 8948–8957, 2019. 2 

- [2] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Guided open vocabulary image captioning with constrained beam search. _arXiv preprint arXiv:1612.00576_ , 2016. 2 

- [3] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In _European conference on computer vision_ , pages 382–398. Springer, 2016. 1, 5 

- [4] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6077–6086, 2018. 2 

- [5] Jyoti Aneja, Harsh Agrawal, Dhruv Batra, and Alexander Schwing. Sequential latent spaces for modeling the intention during diverse image captioning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 4261–4270, 2019. 6 

- [6] Stefano Baccianella, Andrea Esuli, and Fabrizio Sebastiani. Sentiwordnet 3.0: An enhanced lexical resource for sentiment analysis and opinion mining. In _Proceedings of the Sev-_ 

   - _enth International Conference on Language Resources and Evaluation (LREC’10)_ , 2010. 7 

- [7] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , pages 65–72, 2005. 1, 5 

- [8] Ankan Bansal, Karan Sikka, Gaurav Sharma, Rama Chellappa, and Ajay Divakaran. Zero-shot object detection. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , pages 384–400, 2018. 2 

- [9] Steven Bird. Nltk: the natural language toolkit. In _Proceedings of the COLING/ACL 2006 Interactive Presentation Sessions_ , pages 69–72, 2006. 8 

- [10] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 2 

- [11] George Casella and Edward I George. Explaining the gibbs sampler. _The American Statistician_ , 46(3):167–174, 1992. 2 

- [12] Long Chen, Zhihong Jiang, Jun Xiao, and Wei Liu. Humanlike controllable image captioning with verb-specific semantic roles. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 16846– 16856, 2021. 2, 6 

- [13] Qi Chen, Chaorui Deng, and Qi Wu. Learning distinct and representative modes for image captioning. _arXiv preprint arXiv:2209.08231_ , 2022. 3, 4 

- [14] Shizhe Chen, Qin Jin, Peng Wang, and Qi Wu. Say as you wish: Fine-grained control of image caption generation with abstract scene graphs. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 9962–9971, 2020. 2, 3, 6 

- [15] Cesc Chunseong Park, Byeongchang Kim, and Gunhee Kim. Attend to you: Personalized image captioning with context sequence memory networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 895–903, 2017. 3 

- [16] Marcella Cornia, Lorenzo Baraldi, and Rita Cucchiara. Show, control and tell: A framework for generating controllable and grounded captions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8307–8316, 2019. 3 

- [17] Marcella Cornia, Matteo Stefanini, Lorenzo Baraldi, and Rita Cucchiara. Meshed-memory transformer for image captioning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 10578–10587, 2020. 2 

- [18] Chaorui Deng, Ning Ding, Mingkui Tan, and Qi Wu. Lengthcontrollable image captioning. In _European Conference on Computer Vision_ , pages 712–729. Springer, 2020. 3 

- [19] Aditya Deshpande, Jyoti Aneja, Liwei Wang, Alexander G Schwing, and David Forsyth. Fast, diverse and accurate image captioning guided by part-of-speech. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10695–10704, 2019. 2, 3, 6 

9 

- [20] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018. 2 

- [21] Jeffrey Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell. Long-term recurrent convolutional networks for visual recognition and description. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2625–2634, 2015. 2 

- [22] Zhiyuan Fang, Jianfeng Wang, Xiaowei Hu, Lin Liang, Zhe Gan, Lijuan Wang, Yezhou Yang, and Zicheng Liu. Injecting semantic concepts into end-to-end image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18009–18019, 2022. 1, 2, 6 

- [23] Qianyu Feng, Yu Wu, Hehe Fan, Chenggang Yan, Mingliang Xu, and Yi Yang. Cascaded revision network for novel object captioning. _IEEE Transactions on Circuits and Systems for Video Technology_ , 30(10):3413–3421, 2020. 2 

- [24] Chuang Gan, Zhe Gan, Xiaodong He, Jianfeng Gao, and Li Deng. Stylenet: Generating attractive visual captions with styles. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 3137–3146, 2017. 2, 3, 5, 7 

- [25] Zhe Gan, Chuang Gan, Xiaodong He, Yunchen Pu, Kenneth Tran, Jianfeng Gao, Lawrence Carin, and Li Deng. Semantic compositional networks for visual captioning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 5630–5639, 2017. 3 

- [26] Chengying Gao, Qi Liu, Qi Xu, Limin Wang, Jianzhuang Liu, and Changqing Zou. Sketchycoco: Image generation from freehand scene sketches. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 5174–5183, 2020. 5 

- [27] Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. Multimodal neurons in artificial neural networks. _Distill_ , 6(3):e30, 2021. 2 

- [28] Jiuxiang Gu, Gang Wang, Jianfei Cai, and Tsuhan Chen. An empirical study of language cnn for image captioning. In _Proceedings of the IEEE international conference on computer vision_ , pages 1222–1231, 2017. 2 

- [29] Bao Guo, Chunxia Zhang, Junmin Liu, and Xiaoyi Ma. Improving text classification with weighted word embeddings via a multi-channel textcnn model. _Neurocomputing_ , 363:366–374, 2019. 7 

- [30] Longteng Guo, Jing Liu, Peng Yao, Jiangwei Li, and Hanqing Lu. Mscap: Multi-style image captioning with unpaired stylized text. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 4204– 4213, 2019. 3, 5, 7 

- [31] Lisa Anne Hendricks, Subhashini Venugopalan, Marcus Rohrbach, Raymond Mooney, Kate Saenko, and Trevor Darrell. Deep compositional captioning: Describing novel object categories without paired training data. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 1–10, 2016. 2 

- [32] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. _arXiv preprint arXiv:2104.08718_ , 2021. 5, 6 

- [33] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up vision-language pre-training for image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17980–17989, 2022. 1, 2, 6 

- [34] Xiaowei Hu, Xi Yin, Kevin Lin, Lei Zhang, Jianfeng Gao, Lijuan Wang, and Zicheng Liu. Vivo: Visual vocabulary pre-training for novel object captioning. In _proceedings of the AAAI conference on artificial intelligence_ , volume 35, pages 1575–1583, 2021. 2 

- [35] Lun Huang, Wenmin Wang, Jie Chen, and Xiao-Yong Wei. Attention on attention for image captioning. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 4634–4643, 2019. 2 

- [36] Lun Huang, Wenmin Wang, Yaxian Xia, and Jie Chen. Adaptively aligned image captioning via adaptive attention time. _Advances in neural information processing systems_ , 32, 2019. 2 

- [37] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _International Conference on Machine Learning_ , pages 4904–4916. PMLR, 2021. 2, 5 

- [38] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3128–3137, 2015. 5 

- [39] Dong-Jin Kim, Jinsoo Choi, Tae-Hyun Oh, and In So Kweon. Dense relational captioning: Triple-stream networks for relationship-based captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6271–6280, 2019. 3 

- [40] Chia-Wen Kuo and Zsolt Kira. Beyond a pre-trained object detector: Cross-modal textual and visual context for image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17969– 17979, 2022. 2 

- [41] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In _European Conference on Computer Vision_ , pages 121–137. Springer, 2020. 2 

- [42] Yehao Li, Ting Yao, Yingwei Pan, Hongyang Chao, and Tao Mei. Pointing novel objects in image captioning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 12497–12506, 2019. 2 

- [43] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _European conference on computer vision_ , pages 740–755. Springer, 2014. 1, 5 

- [44] Annika Lindh, Robert J Ross, and John D Kelleher. Language-driven region pointer advancement for control- 

10 

lable image captioning. _arXiv preprint arXiv:2011.14901_ , 2020. 3 

- [45] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ , 2019. 5 

- [46] Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh. Neural baby talk. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 7219–7228, 2018. 2 

- [47] Alexander Mathews, Lexing Xie, and Xuming He. Senticap: Generating image descriptions with sentiments. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 30, 2016. 2, 5 

- [48] Alexander Mathews, Lexing Xie, and Xuming He. Semstyle: Learning to generate stylised image captions using unaligned text. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 8591–8600, 2018. 3 

- [49] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. _arXiv preprint arXiv:2111.09734_ , 2021. 1, 2, 6 

- [50] Van-Quang Nguyen, Masanori Suganuma, and Takayuki Okatani. Grit: Faster and better image captioning transformer using dual visual features. In _European Conference on Computer Vision_ , pages 167–184. Springer, 2022. 1, 2, 6 

- [51] Yingwei Pan, Ting Yao, Yehao Li, and Tao Mei. X-linear attention networks for image captioning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 10971–10980, 2020. 2 

- [52] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pages 311–318, 2002. 1, 5 

- [53] Ted Pedersen, Siddharth Patwardhan, Jason Michelizzi, et al. Wordnet:: Similarity-measuring the relatedness of concepts. In _AAAI_ , volume 4, pages 25–29, 2004. 7 

- [54] Fabio Petroni, Tim Rockt¨aschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H Miller, and Sebastian Riedel. Language models as knowledge bases? _arXiv preprint arXiv:1909.01066_ , 2019. 2 

- [55] Farhad Pourpanah, Moloud Abdar, Yuxuan Luo, Xinlei Zhou, Ran Wang, Chee Peng Lim, Xi-Zhao Wang, and QM Jonathan Wu. A review of generalized zero-shot learning methods. _IEEE transactions on pattern analysis and machine intelligence_ , 2022. 2 

- [56] Yu Qin, Jiajun Du, Yonghua Zhang, and Hongtao Lu. Look back and predict forward in image captioning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8367–8375, 2019. 2 

- [57] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_ , pages 8748–8763. PMLR, 2021. 2, 4 

- [58] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 2 

- [59] Idan Schwartz, Alexander Schwing, and Tamir Hazan. Highorder attention models for visual question answering. _Advances in Neural Information Processing Systems_ , 30, 2017. 2 

- [60] Idan Schwartz, Alexander G Schwing, and Tamir Hazan. A simple baseline for audio-visual scene-aware dialog. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 12548–12558, 2019. 2 

- [61] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can clip benefit vision-and-language tasks? _arXiv preprint arXiv:2107.06383_ , 2021. 6 

- [62] Kurt Shuster, Samuel Humeau, Hexiang Hu, Antoine Bordes, and Jason Weston. Engaging image captioning via personality. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 12516– 12526, 2019. 3 

- [63] Haoyu Song, Li Dong, Wei-Nan Zhang, Ting Liu, and Furu Wei. Clip models are few-shot learners: Empirical studies on vqa and visual entailment. _arXiv preprint arXiv:2203.07190_ , 2022. 2 

- [64] Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier. Language models can see: Plugging visual controls in text generation. _arXiv preprint arXiv:2205.02655_ , 2022. 2, 6 

- [65] Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf. Zerocap: Zero-shot image-to-text generation for visualsemantic arithmetic. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17918–17928, 2022. 1, 2, 5, 6, 7 

- [66] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4566–4575, 2015. 1, 5 

- [67] Subhashini Venugopalan, Lisa Anne Hendricks, Marcus Rohrbach, Raymond Mooney, Trevor Darrell, and Kate Saenko. Captioning images with diverse objects. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 5753–5761, 2017. 2 

- [68] Ashwin Vijayakumar, Michael Cogswell, Ramprasaath Selvaraju, Qing Sun, Stefan Lee, David Crandall, and Dhruv Batra. Diverse beam search for improved description of complex scenes. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 32, 2018. 3, 6 

- [69] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3156–3164, 2015. 2 

- [70] Alex Wang and Kyunghyun Cho. Bert has a mouth, and it must speak: Bert as a markov random field language model. _arXiv preprint arXiv:1902.04094_ , 2019. 2, 7 

- [71] Li Wang, Zechen Bai, Yonghua Zhang, and Hongtao Lu. Show, recall, and tell: Image captioning with recall mechanism. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pages 12176–12183, 2020. 2 

11 

- [72] Liwei Wang, Alexander Schwing, and Svetlana Lazebnik. Diverse and accurate image description using a variational auto-encoder with an additive gaussian encoding space. _Advances in Neural Information Processing Systems_ , 30, 2017. 3, 6 

- [73] Qingzhong Wang and Antoni B Chan. Describing like humans: on diversity in image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 4195–4203, 2019. 2, 3, 6 

- [74] Yisheng Xiao, Lijun Wu, Junliang Guo, Juntao Li, Min Zhang, Tao Qin, and Tie-yan Liu. A survey on nonautoregressive generation for neural machine translation and beyond. _arXiv preprint arXiv:2204.09269_ , 2022. 3, 4 

- [75] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In _International conference on machine learning_ , pages 2048–2057. PMLR, 2015. 2 

- [76] Xu Yang, Kaihua Tang, Hanwang Zhang, and Jianfei Cai. Auto-encoding scene graphs for image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10685–10694, 2019. 2 

- [77] Ting Yao, Yingwei Pan, Yehao Li, and Tao Mei. Exploring visual relationship for image captioning. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 684–699, 2018. 2 

- [78] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18123–18133, 2022. 2 

- [79] Pengchuan Zhang, Xiyang Dai, Jianwei Yang, Bin Xiao, Lu Yuan, Lei Zhang, and Jianfeng Gao. Multi-scale vision longformer: A new vision transformer for high-resolution image encoding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 2998–3008, 2021. 2 

- [80] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5579– 5588, 2021. 2, 6 

- [81] Wentian Zhao, Xinxiao Wu, and Xiaoxun Zhang. Memcap: Memorizing style knowledge for image captioning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pages 12984–12992, 2020. 3, 7 

- [82] Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason Corso, and Jianfeng Gao. Unified vision-language pretraining for image captioning and vqa. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pages 13041–13049, 2020. 2 

12 

## **A. Discussion about NOIC and ZIC** 

In this section, we will discuss the difference between novel object image captioning (NOIC) and zero-shot image captioning (ZIC), brief comparisons are shown in Table 6 and details are as follows: 

_•_ _**Generalization among objects vs. among tasks.**_ NOC aims to generalize image captioning (IC) models to “novel objects” not presented in the training images. This means both training and testing tasks are IC. **By contrast** , the “zero-shot” concept in our work (and most related work in our paper) comes from GPT-3, referring to _applying large pre-trained models_ (trained with no specific task) _for downstream IC tasks with no task-specific fine-tuning_ . 

_•_ _**With vs. without curated training image-caption pairs.**_ NOC models are often trained on well-designed image-caption pairs of seen objects. Hence, different dataset splits are often considered to perform evaluation. **By contrast** , ConZIC is free of well-designed image-caption pairs to perform training or even fine-tuning. 

_•_ _**With vs. without extra knowledge.**_ NOC methods often learn the relations between objects and extra taggers, such as attributes and class embeddings. Then, these relations are generalized to unseen objects by various techniques. **By contrast** , ConZIC utilizes the knowledge from large pre-trained models and thus is free of extra information. 

## **B. Algorithm of Gibbs-BERT** 

After randomly choosing the generation order, GibbsBERT starts from a full noisy sentence ( _e.g._ , all [ _MASK_ ] tokens). At each iteration, Gibbs-BERT progressively samples each word by putting [ _MASK_ ] at this position and then selecting the top-1 word from the predicted word distribution over the vocabulary by BERT. The result of _t_ -th iteration is the initialization of the ( _t_ + 1)-th iteration. The pseudo-code is shown in algorithm. 2 

## **C. SketchyCOCO caption benchmark** 

SketchyCOCO caption is a small sketch-style image captioning benchmark based on SketchyCOCO, including 14 classes, as shown in Fig. 8. SketchyCOCO is not an image captioning dataset since it only has the classification label. we construct the captioning benchmark through a text prompt “A drawing of a [CLASS]”, where [CLASS] is the class name. 

## **D. More generation examples** 

**Comparison with Ground-Truth.** As shown in Fig. 9, due to the zero-shot nature, caption generations of our method are different from MSCOCO ground-truth. our 

|**Algorithm 2:**Algorithm of Gibbs-BERT.|||
|---|---|---|
|**Data:** initial sentence:**_x_**0<br>_<_1_,n>_ = (_x_0<br>1_, ..., x_0<br>_n_);|||
|iterations=_T_, candidates=_K_;|||
|position sequence_P_ =_Shuffe_([1_, ..., n_]);|||
|**Result:** the fnal sentence: **_x_**_T_<br>_<_1_,n>_ = (_xT_<br>1|_, ..., xT_<br>_n_);||
|**for**_iteration t ∈_[1_, ..., T_]**do**|||
|state: **_x_**_t−_1<br>_<_1_,n>_ = (_xt−_1<br>1<br>_, ..., xt−_1<br>_n_<br>);|||
|**for**_position i ∈P_ **do**|||
|1. Replace_xt−_1<br>_i_<br>with [_MASK_];|||
|2. Predict the word distribution over|vocabulary||
|by BERT:_p_(_xi|_**_x_**_t−_1<br>_−i_ );|||
|3. Sample_xi_ from distribution_p_(_xi|_**_x_**_t−_1<br>_−i_||);|
|4. Replace_xt−_1<br>_i_<br>with_xt_<br>_i_;|||
|**end**|||
|state: **_x_**_t_<br>_<_1_,n>_ = (_xt_<br>1_, ..., xt_<br>_n_);|||
|**end**|||



method has shown significant differences with groundtruth in syntactic(sentence patterns) and semantic(diverse words). 

**Diverse generation compared with ZeroCap.** Comparison results on diverse caption generation are shown in Fig. 10. ZeroCap generates diverse captions by beam search, which can result in a similar sentence pattern with respect to mode collapse. In contrast, our method can produce multiple captions related to the same image by shuffling the word generation order, which has shown strong performance in syntactic and semantic diversity. 

**Results on various image styles and world knowledge.** As shown in Fig. 11 and Fig. 12. Our method performs well in various image styles, _e.g._ natural images, medical images, oil paintings and cartoon images. Besides, our method is proven to have efficient application in images with abundant world knowledge, _e.g._ medical, geography, celebrity, and artworks. 

## **E. Controllable tasks** 

## **E.1. Diversity of length control** 

Table 7 has reported diversity performance where we select the best-1 caption on each length and then compute diversity metrics on these four lengths. Our method surpasses ZeroCap by a large margin. 

## **E.2. Infilling tasks** 

We have conducted experiments on one-word-infilling and multiple-word-infilling. 

**One-word-infilling.** We randomly corrupt one verb/noun in the reference caption, and ask models to infill the most suitable word given other words. We use three metrics to evaluate the accuracy performance: _1)_ BLEU1(B-1) to measure unigram precision; _2)_ Wordnet path sim- 

13 

||Novel object captioning|ZeroCapor ConZIC|
|---|---|---|
|generalization ability|seen objects IC_→_unseen objects IC<br>(with limited background/image styles)|large pretrained models_→_image captioning task<br>(with no limitations on objects/background/image styles)|
|well designed image-captionpairs|needed for trainingor fne-tuning|no need|
|extra knowledge|object taggers|no need|



Table 6. Comparisons on problem scenarios of NOIC and our ZIC. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0014-02.png)


**----- Start of picture text -----**<br>
A drawing of a dog. A drawing of a motorcycle.<br>A drawing of a horse. A drawing of a airplane.<br>A drawing of a cat. A drawing of a traffic light.<br>A drawing of a sheep. A drawing of a fire hydrant.<br>A drawing of a cow. A drawing of an elephant.<br>A drawing of a bicycle. A drawing of a zebra.<br>A drawing of a car. A drawing of a giraffe.<br>**----- End of picture text -----**<br>


Figure 8. Examples of SketchyCOCO caption benchmark 

|||
|---|---|
|DiversityMetrics|S-C(_↑_)<br>Div-1(_↑_)<br>Div-2(_↑_)|
|Zerocap<br>**Ours**|0.63<br>0.40<br>0.56<br>**0.95**<br>**0.63**<br>**0.84**|



Table 7. Length controlled diversity metrics of our method on MSCOCO. we select the best-1 caption on each length and then compute diversity metrics conditioning on these four captions. 

|Corrupted Ratio|B-4(_↑_)<br>M(_↑_)<br>CLIP-S(_↑_)|
|---|---|
|0.25<br>0.50<br>0.75|60.69<br>44.99<br>0.83<br>26.08<br>29.12<br>0.89<br>8.06<br>17.60<br>0.93|



Table 8. Results of multiple-words-infilling task. 

ilarity(WSim) which measures node distance in Wordnet. Especially, this metric can only be computed between two words of the same POS. Therefore, we set WSim as 0 when the answer has a different POS from the reference word; _3)_ BERT word similarity(BSim). We use cosine distance in BERT word embedding space, where words have similar semantics generally possess a low distance. Due to its au- 

toregressive nature, ZeroCap can only take the left context into account, which limits its performance. The results are shown in Table. 3. 

**Multiple-word-infilling** In contrast to one-wordinfilling, we try to corrupt more words in reference caption, as illustrated in Fig. 5. Results with different corrupted ratios are shown in Table. 8. We can see that results of a higher corrupted ratio are generally higher in CLIP-S and lower in other metrics. 

## **E.3. Humorous-Romantic control on FlickStyle10k** 

Quantitative results are shown in Table. 9. As we can see, our method has comparable performance in producing captions in specific styles, _i.e._ romantic and humorous, as shown in _Acc_ column. 

## **E.4. Parts-of-speech controlling** 

we have tried another POS sequence, _ADP DET ADJ/NOUN NOUN NOUN DET ADJ/NOUN NOUN VERB VERB ADV_ . Results are shown in Table. 10. 

14 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0015-00.png)


**GT** : 

Several clocks display the time in different time zones. 

## **Ours** : 

Watching some magical silver wall clock highlighting the different time zones within a museum. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0015-05.png)


## **GT** : 

A train sitting in a train station on top of railroad track. 

## **Ours** : 

A prototype 1960s general electric Scottish locomotive maintained at the Victoria train shed. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0015-10.png)


## **GT** : 

A white bird walking through a shallow area of water. 

## **Ours** : 

A spotted bird shown on a sandy platform with wavy reflections. 

Figure 9. Comparision with groundtruth on MSCOCO caption. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0015-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0015-17.png)


**----- Start of picture text -----**<br>
：<br>A dog replica.<br>A dog sculpture.<br>A dog statue.<br>A dog sculpture created in London's Museum of Modern Art.<br>A dog sculpture created in London's Museum of Modern Art in the early 2000s.<br>**----- End of picture text -----**<br>


： _Order:_ 7, 3, 2, 8, 5, 6, 9, 4, 0, 1 _Cap_ :    A striped 3d pet model-sized grey lab tiger displayed. _Order:_ 7, 8, 1, 5, 3, 4, 2, 0, 9, 6 _Cap_ :    A grey metallic 3d model exhibiting a striped pet tiger. _Order:_ 6, 8, 9, 7, 5, 3, 0, 4, 1, 2 

_Cap_ :    A tiger sculpture painted on a statue display shown throughout campus. _Order:_ 5, 9, 3, 4, 6, 7, 2, 8, 1, 0 _Cap_ :    A silver painted animal in striped yellow within window displays. _Order:_ 1, 5, 6, 0, 9, 4, 7, 2, 8, 3 _Cap_ :    A silver striped tiger model depicted on window shopping display. 

Figure 10. Diversity results compared with ZeroCap. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0015-21.png)


**GRIT** : A series of clocks on top of a screen **CLIPCap** : A picture of a skull and crossbones with a lot of wires. **ViTCap** :a close up of a cake with a picture of a dog. **ZeroCap:** A patient submitted to the neurologist's office. **Ours:** 

A complete CT medical photo with brain samples. A CT scene multiple frames displaying a diagnosis. A stacked sheet displaying signs of brain damaged. A composite present multiple images featuring frontal trauma. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0015-24.png)


**GRIT** : A painting of a painting with a tree in the background **CLIPCap** : The night sky over the city. **ViTCap** :A painting of a bird on a table with a bird on it. **ZeroCap:** A night with Vincent. 

**Ours:** 

A famous Gogh painting after streaming moonlight over all the grand structures. 

A view despite a nocturnal sky within famous mainstream artworks. A nighttime sky can appear in drawings and oil paintings. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0015-29.png)


**GRIT** : A busy city street with lots of people walking on. **CLIPCap** :A busy city street with people crossing it. **ViTCap** : a city street with people walking and a bus. **ZeroCap** :A billboard in the middle of of of a busy intersection. 

## **Ours:** 

A new york time square. 

A busy billboard covered time square with scenery. A landscape masking time square depicting a vibrant morning. A city featuring yellow billboard and advertisements on google outdoor. 

Figure 11. Results of various image styles. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0016-00.png)


**GRIT:** a very tall tower with a clock tower in the **CLIPCap:** A tall tower with a clock on top. **ViTCap** :a picture of a tall tower with a clock on it. **ZeroCap** :Image of a French Italian landmark is captioned on the first of the month. 

## **Ours:** 

image of a famous landmark in france. image of a tall structure iconic in famous french photographs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0016-04.png)


**GRIT** :a man in a red uniform holding a basketball **CLIPCap** :A man wearing a red neck tie holding a ball. **ViTCap** :a close up of a baseball player holding a ball **ZeroCap** :Image of a hero NBA All-Star Michael Jordan in Sports Illustrated uniform 

## **Ours** : 

A Jordan steel bulls jersey taken from NBA promotional material. A 1990s superstar wearing bright rose gold jerseys. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing_images/cvpr_2023_ConZIC_Controllable_Zero-Shot_Image_Captioning_by_Sampling-Based_Polishing.pdf-0016-08.png)


**GRIT:** a red teddy bear with a heart on a **CLIPcap:** A Star Wars character is wearing a star trek neck tie. **ViTcap:** a drawing of a man holding a giant shark. **Zerocap:** Image of a character from Spider-Man comic art. 

## **Ours:** 

A minime for super spiderman animated. A stylized spider wearing a small comic outfit. 

Figure 12. Results of images containing world knowledge. 

||||
|---|---|---|
|Methods|Romantic<br>B-3(_↑_)<br>M(_↑_)<br>CLIP-S(_↑_)<br>Acc(_↑_)|Humorous<br>B-3(_↑_)<br>M(_↑_)<br>CLIP-S(_↑_)<br>Acc(_↑_)|
|StyleNet<br>MSCap<br>MemCap|1.5<br>4.5<br>-<br>37.8<br>2.0<br>5.4<br>-<br>88.7<br>**4.0**<br>**7.7**<br>-<br>91.7|0.90<br>4.30<br>-<br>41.9<br>1.90<br>5.30<br>-<br>91.3<br>**4.0**<br>**7.20**<br>-<br>**97.1**|
|**Ours**|1.2<br>6.1<br>**1.02**<br>**96.3**|1.2<br>6.1<br>**1.02**<br>91.4|



Table 9. Stylized image captioning( _i.e._ romantic, humorous) performance comparisons on the Flickstyle8k dataset. 

## **F. Bad case analysis** 

As shown in Fig. 10, ZeroCap and our method both ignore the “scissor” around the “tiger statue”, which means that how to control which image content to be described, in particular, small objects, is under-explored for zero-shot image captioning. 

|Parts-of-speech<br>M(_↑_)<br>C(_↑_)<br>CLIP-S(_↑_)<br>Acc(_↑_)|Parts-of-speech<br>M(_↑_)<br>C(_↑_)<br>CLIP-S(_↑_)<br>Acc(_↑_)|
|---|---|
|without POS<br>with POS|11.54<br>12.84<br>1.01<br>15.54<br>7.99<br>9.29<br>0.95<br>86.20|



Table 10. Results of parts-of-speech control on MSCOCO. Predefined POS tags is _ADP DET ADJ/NOUN NOUN NOUN DET ADJ/NOUN NOUN VERB VERB ADV_ 

16 

