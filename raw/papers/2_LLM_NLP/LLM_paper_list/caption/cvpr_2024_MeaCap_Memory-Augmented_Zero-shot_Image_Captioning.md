## **MeaCap: Memory-Augmented Zero-shot Image Captioning** 

Zequn Zeng, Yan Xie[*][*] , Hao Zhang[†] , Chiyu Chen, Bo Chen[†] National Key Laboratory of Radar Signal Processing, Xidian University, Xi’an, 710071, China _{_ zzequn99, yanxie0904, zhanghao ~~x~~ idian _}_ @163.com, _{_ chenchiyu, bchen _}_ @mail.xidian.edu.cn 

Zhengjue Wang 

State Key Laboratory of Integrated Service Networks, Xidian University, Xi’an, 710071, China 

wangzhengjue@xidian.edu.cn 

## **Abstract** 

_Zero-shot image captioning (IC) without well-paired image-text data can be divided into two categories, training-free and text-only-training. Generally, these two types of methods realize zero-shot IC by integrating pretrained vision-language models like CLIP for image-text similarity evaluation and a pre-trained language model (LM) for caption generation. The main difference between them is whether using a textual corpus to train the LM. Though achieving attractive performance w.r.t. some metrics, existing methods often exhibit some common drawbacks. Training-free methods tend to produce hallucinations, while text-only-training often lose generalization capability. To move forward, in this paper, we propose a novel_ _**Me** mory-_ _**A** ugmented zero-shot image_ _**Cap** tioning framework (_ _**MeaCap** ). Specifically, equipped with a textual memory, we introduce a retrieve-then-filter module to get key concepts that are highly related to the image. By deploying our proposed memory-augmented visual-related fusion score in a keywords-to-sentence LM, MeaCap can generate concept-centered captions that keep high consistency with the image with fewer hallucinations and more world-knowledge. The framework of MeaCap achieves the state-of-the-art performance on a series of zero-shot IC settings. Our code is available at https://github.com/joeyz0z/MeaCap._ 

## **1. Introduction** 

Image captioning (IC) aims to understand visual content and generate text descriptions. Using well-annotated image-text pairs, supervised models [8, 18, 24, 35, 40, 42, 49, 50, 57] have achieved promising results on typical IC benchmarks [1, 26, 33, 64]. Due to the high costs of annotation, the training sets of these benchmarks often in- 

> *Equal contribution. †Corresponding authors 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0001-11.png)


**----- Start of picture text -----**<br>
Text-only-training ScoreCLIP BLIP-2Score<br>MAGIC: A plate topped with cake and spoon. 0.76 0.89<br>DeCap: A piece of cake on a white plate with a spoon.  0.77 0.87<br>ViECap: Cake with white frosting on a white plate on a table. 0.75 0.73<br>𝑴𝒆𝒂𝑪𝒂𝒑 𝑻𝒐𝑻 :  conceptscaption : A :  [ sliceslice lemon pie  of  lemon p , ie  serving plate with spoon on ] serving plate  on table. [0.83] 0.83<br>Training-free ScoreCLIP<br>ZeroCap: A large dessert eaten in the 2016 New Hampshire State Hotel. 0.87 0.75<br>ConZIC: A butter pie served at the famous Mary Teresa restaurant. 1.00 0.77<br>𝑴𝒆𝒂𝑪𝒂𝒑𝑴𝒆𝒂𝑪𝒂𝒑 𝑻𝑭 𝑻𝑭 :: conceptscaption : A :  [ sliceslice lemon pie  of  lemon p , ie  serving plate with a spoon on a  spiderman ] serving plate . 0.84 0.82<br>(a) Hallucination phenomenon.<br>Text-only-training Score CLIP BLIP-2 Score<br>MAGIC: A red and white locomotive is being docked. 0.32 0.30<br>DeCap: A person that is on the ground and is holding his device. 0.51 0.22<br>ViECap: Before and after shots of a man in a suit and tie. 0.42 0.31<br>𝑴𝒆𝒂𝑪𝒂𝒑 𝑴𝒆𝒂𝑪𝒂𝒑 𝑻𝒐𝑻 𝑻𝒐𝑻 :  :  concepts caption : A picture of a  :  [ spiderman spiderman ]  comics. 0.68 0.65<br>Training-free Score CLIP<br>ZeroCap: Image of a Web Hero. 0.74 0.27<br>ConZIC: A very attractive spiderman typical marvel definition. 0.82 0.59<br>𝑴𝒆𝒂𝑪𝒂𝒑 𝑴𝒆𝒂𝑪𝒂𝒑𝑻𝑭𝑻𝑭 : :  concepts  caption : A comic book su :  [ spiderman ] perhero called  spiderman . 0.77 0.68<br>(b) Image contains world knowledge.<br>**----- End of picture text -----**<br>


Figure 1. The motivation of our proposed MeaCap where the red is incorrect and green is correct. (a) Training-free methods associate the _pie_ with incorrect location information, which actually get high marks in CLIPscore. This might be due to the fact that CLIP is trained on web-scale noisy image-text data. (b) Existing text-only-training (ToT) methods fail to generate _spiderman_ as some training-free methods do, but the ToT version of our method (MeaCapToT) can also do that. 

volve limited image styles/contents, which is a hard obstacle for those supervised models to be generalized to images in the wild. To realize IC without human-annotated imagetext pairs, recently, zero-shot IC has drawn increasing attention. Existing works can be mainly divided into two groups, training-free methods and text-only-training methods. Training-free approaches [54, 55, 65] realize zero-shot image-to-text generation using pre-trained models without fine-tuning. Specifically, they employ a pre-trained visionlanguage model like CLIP to guide a pre-trained language model (LM), such as BERT [9] or GPT-2 [45], to generate sentences that match the given image. With iterative inferences, this line of work does not require any training. Though having achieved superior generalization ability and 

1 

higher CLIPscore [17], these methods show extrinsic hallucination phenomenon, _i.e_ ., they tend to generate a story containing imaginary information that may not exist in the given image, as shown in Fig. 1a. 

To alleviate this issue, another line of works trains or fine-tunes the text decoder based on high-quality text data without corresponding images, termed as text-only-training methods [13, 30, 41, 52, 56]. For testing images containing objects described in the training corpus, text-onlytraining methods generate captions objectively, achieving significant improvements w.r.t. reference-based scores such as BLEU [43], METEOR [4], and CIDEr [58]. However, due to the limited training corpus, the knowledge contained in the pre-trained LM is gradually forgotten during training, resulting in severe performance degradation on out-ofdomain data, as shown in Fig. 1b. Although training on web-scale high-quality corpus is a potential solution, which hence produces extremely high computational costs. 

To maintain good generalization ability to images in the wild and to get rid of unreasonable imagination, this paper proposes a novel **Me** mory- **A** ugmented zero-shot image **Cap** tioning framework, namely **MeaCap** , based on the memory-guided mechanism, which provides an alternative scheme to use captioning corpus rather than using it to train the LM. Specifically, from an external textual memory, we develop a retrieve-then-filter module to find key concepts that are highly related to the given image. Introducing our proposed memory-augmented visual-related fusion score to a keywords-to-sentence LM, CBART [16], MeaCap can generate concept-centered captions that keep high consistency with the images. This new visual-related score not only considers image-text cross-modal similarity as most zero-shot IC methods [52, 54–56, 65] do by CLIP but also considers text-text in-modal similarity by evaluating the similarity between captions and retrieved image-related memory. Our proposed MeaCap can be either training-free named **MeaCap** TF or text-only-training named **MeaCap** ToT by fine-tuning CBART. 

Our contributions are summarized as follows: 

- We employ the text-only captioning corpus as the external memory to enhance training-free zero-shot IC. To this end, We introduce a retrieve-then-filter module to extract key concepts from the memory and perform conceptcentered generations by CBART to alleviate the hallucination issue of previous training-free methods. 

- Based on the retrieved textual memory, we develop a memory-augmented visual-related fusion score into CBART, improving the correlation between image and generated captions while reserving the world-knowledge. 

- Extensive experiments under zero-shot, in-domain, and cross-domain scenarios demonstrate our proposed memory-augmented design can significantly improve the consistency with image content in both the training-free 

and text-only-training settings. 

## **2. Related work** 

## **2.1. Supervised image captioning** 

Supervised IC typically uses well-aligned image-text pairs and trains an encoder-decoder model. For example, some early attempts [10, 14, 59, 61] construct CNN-based encoder to extract visual features and RNN/LSTM-based decoder to generate output sentence. For better visual understanding, some methods [3, 8, 18, 19, 27, 44, 60] employ an object detector to extract attentive image regions. To encourage more interactions between two modalities, attention mechanism [8, 18, 40, 42, 49, 50] and graph neural network [62, 63] have been widely adopted. 

## **2.2. Zero-shot image captioning** 

In recent years, zero-shot IC has gained more and more attention. Compared with supervised IC, zero-shot IC targets at generating image captions under two cases: _i)_ without any data for training named _training-free_ zero-shot IC; _ii)_ just using text from the captioning dataset to train the LM named _text-only-training_ zero-shot IC. 

**Training-free methods** . This type of methods realizes the zero-shot IC with the help of the pre-trained visionlanguage model, CLIP [46], to guide the generation of a pre-trained LM. Specifically, ZeroCap [55] and its extension [54] for video captioning are proposed based on the gradient-search iteration. To make the zero-shot IC controllable, ConZIC [65] is proposed by combining Gibbssampling with a non-autoregressive LM, improving the diversity and inference speed of IC. Although they achieve superior generalization ability with higher CLIPscore [17], they may generate some descriptions that do not appear in the image, called hallucination as shown in Fig. 1a. 

**Text-only-training methods** . Another line of zero-shot IC methods train or fine-tune the text decoder just using the corpus from the captioning dataset. Concretely, after fine-tuning the off-the-shelf simCTG [53] directly on the specific corpus, MAGIC [52] and ZEROGEN [56] are proposed by introducing a CLIP-induced score to regularizes the generated process of simCTG, making the caption semantically related to a given image. By regarding the original sentence or sentence embedding as the prompt to train a LM, DeCap [30], CapDec [41] and ViECap [13] are developed by mapping the visual feature to the text feature, which is then fed into the this LM for caption generation. 

Our proposed MeaCap can perform both training-free and text-only-training zero-shot IC. For the training-free setting, because we introduce the memory mechanism for key concept identification (Sec. 3.1) and guidance for LM generation (Sec. 3.3), our method can generate more accurate captions with less hallucination. For text-only-training 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Subject-Predicate-Object triplets<br>Parse<br>Memory 1. Collection of stuffed toy bears sit on rock in outdoor scene. bears teddybears teddybears woman bears<br>2. Four teddy bears from font sitting on ledge with patchwork grass.<br>𝑬𝒕 3. Four teddy bears are posed on a rock. sit on sit on pose on hold rest on sit in front of<br>4. A group of woman holding teddy bears and resting on a the wall.5. A group of stuffed bears sit in front of a wall. rock ledge rock teddybears wall wall<br>Retrieve<br>𝑬𝒊 𝑺𝒕 𝑺𝒕 woman<br>Merge bears wall<br>𝑺𝒕 And teddy bears<br>𝑬𝒕 Filter 𝜏<br>rock ledge<br>𝜏<br>𝑝 [𝑙𝑚] 𝑝 [𝐼𝑇𝑠] 𝑝 [𝑇𝑇𝑠]<br>A teddy bears sitting rock.<br>keywords A teddy bears standing rock. Key concepts : teddy bears       rock<br>to A teddy bears holding rock. 𝟏 [𝒕𝒉] ∶ A teddy bears  sitting rock<br>sentenceLM A teddy bears A teddy bears withon rock. rock. + + Refine 𝟐𝟑 [𝒕𝒉][𝒕𝒉] ∶∶ A A stuffedgroup stuffed  teddy bears teddy bears  sitting onsitting on rock  a rock<br>𝟒 [𝒕𝒉] ∶ A group of stuffed  teddy bears  sitting on a  rock<br>Memory-augmented visual-related fusion score 𝟓 [𝒕𝒉] :  A group of stuffed  teddy bears  sitting on a  rock<br>A teddy bears <M> rock<br>… … … …<br>**----- End of picture text -----**<br>


Figure 2. Overview of our proposed **MeaCap** . The overall data flow is clockwise. _i)_ Given an image, we first _retrieve_ Top- _N_ relevant descriptions from the memory, which is transformed to the subject-predicate-object triplets; we merge and filter nodes to get the key concepts Sec. 3.1. _ii)_ With the memory-augmented visual-related fusion score (Sec. 3.3), starting from key concepts, the keywords-tosentence LM can complete the image description by iterative refining (Sec. 3.2). _Ei, Et, St_ are CLIP visual encoder, CLIP text encoder, and Sentence-BERT text encoder, respectively.[�] denotes the cosine similarity. The _p[lm] , p[IT s] , p[T T s]_ are fluent score in Eq. (7), imagecaption cross-modal similarity Eq. (8), and memory-caption in-modal similarity Eq. (9), respectively. 

setting, the use of the corpus as the external memory in our method can alleviate the problem of existing methods that forget the world-knowledge learned by pre-trained LM due to the corpus-specific fine-tuning. 

## **2.3. External memory in image captioning** 

It has been proven that introducing external memory is useful for various visual and language tasks, like natural language process [6, 15, 22, 23, 38], visual recognition [34, 36], image synthesis [5, 11], open-domain question-answering [21, 28], and IC included [30, 47]. For instance, SmallCap [47] is a supervised IC method that utilizes CLIP to retrieve a few relevant captions and then takes these captions as the prompt for the LM, demonstrating the memory can help LM to generate accurate captions with fewer training parameters. In zero-shot captioning, DeCap [30] trains a LM to invert the CLIP text embeddings to the corresponding sentence. It projects the CLIP visual embeddings into a weighted sum of the textual memory embeddings and takes the final textual embedding as a soft prompt to the LM to guide caption generation. 

Compared with DeCap and SmallCap, which leverage the whole memory sentence as a prompt to guide generation, we propose a training-free filter that removes the noisy information to get the key concepts from retrieved textual memory. Unlike the memory in DeCap is only designed for text-only-training methods, our explicit memory design can be applied to both training-free and text-only-training scenarios and shows superior capability to generate more accurate captions. 

## **3. MeaCap** 

For better zero-shot IC with less hallucination and reserving more world-knowledge, as shown in Fig. 2, we propose a novel framework called MeaCap. _i)_ To solve the problem of existing training-free methods [54, 55, 65] that may bring hallucination in the captions, MeaCap identifies some key concepts from the retrieved textual memory which is highly related to the image, and performs concept-centered captioning (Sec. 3.1). _ii)_ We develop a memory-augmented visual-related fusion score (Sec. 3.3), considering both image-text cross-modal similarity and text-text in-modal similarity (between textual memory and captions), which is introduced to the keywords-to-sentence LM, CBART [16] (Sec. 3.2), improving the image-caption correlations. 

## **3.1. Retrieve-then-filter to get key concepts** 

The existing text-only-training zero-shot IC methods [13, 30, 41, 52, 56] usually train or fine-tune a LM on the texts from the captioning dataset, which brings more suitable descriptions with less hallucination. However, such methods make the generated captions overfit to a specific corpus, lacking the out-of-distribution generalization. Motivated by this phenomenon, instead of training or fine-tuning a LM on the texts, we just build an augmented textual memory to get the key concepts, which can then guide the zero-shot IC. 

**Build augmented memory.** For this end, we firstly construct a large textual memory M which contains various visual-related sentences with abundant visual concepts. This memory is significant for removing the hallucination for the training-free case and can alleviate the knowledge- 

3 

forgotten for the text-only-training case. 

**Retrieve image-related descriptions.** Having obtained the memory, given an image **I** , we use CLIP for the evaluation of image-text similarity to retrieve Top- _N_ imagerelated descriptions from the memory as _{mn}[N] n_ =1 _[d]_[:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0004-02.png)


where _Ei_ ( _·_ ) and _Et_ ( _·_ ) denote the image and text encoder in the CLIP, respectively; cos( _·, ·_ ) is the cosine similarity. 

**Subject-Predicate-Object triplets.** To further reduce the impact of some less-information words in the imagerelated descriptions _{mn}[N] n_ =1 _[d]_[,][such][as][article][and][prepo-] sition, we use an off-the-shelf textual parser, TextGraphParser [32], to transform each description _mn_ to a text-graph _gn_ including multiple subject-predicate-object triplets, where subjects and objects are nodes while predicates are the relation. These nodes are regarded as candidate concepts which will be filtered and merged to form a set of the key concepts. The relations will decide the order between two concepts. We define _{vn}[N] n_ =1 _[c]_[as the set of all] nodes from all _Nd_ text graphs _{gn}[N] n_ =1 _[d]_[.] 

**Merge and filter to obtain the key concepts.** As shown in Fig. 2, some nodes denoting concepts may represent the same object in the image ( _e.g_ ., ”bear” and ”teddy bear”), while some ones may be irrelevant to the image ( _e.g_ ., ”woman”), which should be merged and filtered before getting the key concepts. 

_i) Merge_ . With the help of text encoder from SentenceBERT [48], _St_ ( _·_ ), we can obtain the concept embedding set _{_ _**f**[c] n[}][N] n_ =1 _[c]_[as] _**[ f]**[ c] n_[=] _[ S][t]_[(] _[v][n]_[)][.][Then we evaluate the similarity] between any two concept embeddings as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0004-07.png)


Then, we set a hyper-parameter _τ_ as the threshold, where _dij > τ_ denotes that _i_ -th concept and _j_ -th concept belong to the same cluster. After this, totally we have _Nv_ con- _Nv_ cept clusters as _cn_ = _{vi}[N] i_ =1 _[cn]_[where] _[N][c][n]_[denotes] � � _n_ =1[,] the number of nodes in _n_ -th concept cluster _cn_ . 

_2) Filter_ . In this step, we need to decide whether the _n_ - th concept cluster _cn_ is removed or reserved. For this end, a reasonable assumption is that the word irrelevant to the image has a lower appearance in the retrieved descriptions _{mn}[N] n_ =1 _[d]_[in][Eq.][(][1][).][Therefore,][we][calculate][the][concept-] cluster frequency _CF_ ( _cn_ ) by gradually seeing whether _vi_ from _cn_ appearing in _mk_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0004-10.png)


where _CF_ ( _cn_ ) indicates the frequency of the _n_ -th cluster appearing in the retrieved descriptions _{mn}[N] n_ =1 _[d]_[.][Empiri-] cally, if _CF_ ( _cn_ ) _>_ 0 _._ 5, we reserve this cluster _cn_ and otherwise delete it. Finally, we filter out _nv_ key concept clusters from original _Nv_ ones, which are highly related to images. 

_3) Find key concepts_ . Having obtained _nv_ key concept clusters _{cn}[n] n[v]_ =1[where][each][cluster][may][contain][multiple] similar concepts, we need to identify one concept to represent this cluster. For this target, we use CLIP to select one concept from one cluster by finding the maximum imageconcept similarity as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0004-13.png)


where _c[key] n_ is the selected concept for the cluster _cn_ . After these three steps, we have the set of key concepts as _{c[key] n }[n] n[v]_ =1[that is highly visual-related.][Before using these] concepts to generate captions by the following keywordsto-sentence LM, we need to decide their orders, which is realized by the relations in subject-predicate-object triplets. 

## **3.2. Keywords-to-sentence LM** 

To generate a fluent visual-related caption starting from key concepts _{c[key] n }[n] n[v]_ =1[,][we][employ][a][pre-trained][lexi-] cally constrained language model, CBART [16]. Specifically, CBART is developed to generate a sentence _S_ = ( _x_ 1 _, ..., xn_ ) given the ordered _K_ keywords _{ci}[K] i_ =1[by max-] imizing the conditional probability. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0004-17.png)


where _x_ 1 _, ..., xn_ are words. To this end, CBART has an action encoder and a language decoder for iteratively refining the sentence starting from keywords. At _t_ -th iteration, the encoder is responsible for predicting which word-level action ( _copy_ , _replacement_ , and _insertion_ ) should be taken. In other words, the encoder takes an incomplete sentence _St_ having _n[′]_ words as input and outputs the corresponding action sequence _Lt_ = _{lt,_ 1 _, · · · , lt,n[′] }_ , where _lt,i_ denotes the action of _i_ -th word at _t_ -th iteration. 

_i) Copy_ . Copy means current word remains unchanged. 

_ii) Replacement_ . Replacement suggests the current word should be replaced. Specifically, CBART uses a mask token _<_ M _>_ to replace current word and sample a new word based on the conditional probability _p[lm]_ ( _x<_ M _>|x−<_ M _>_ ), where _x−<_ M _>_ denotes unmasked tokens. 

_iii) Insertion_ . Insertion indicates the decoder should insert a word before the current word. Similar to the replacement action, CBART inserts a _<_ M _>_ token before the current word and then samples a word from _p[lm]_ ( _x<_ M _>|x−<_ M _>_ ). 

Accordingly, the decoder can refine the sentence from _St_ to _St_ +1. Therefore, the complete encoder-decoder sentence 

4 

refinement by CBART at _t_ -th iteration can be formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0005-01.png)


After a few iterations, CBART will terminate the refinement when the encoder outputs a full-copy action sequence. 

According to the above introduction, existing CBART does not meet our needs because for replacement and insertion actions, the word only drawn from probability by pretrained LM _p[lm]_ ( _x<_ M _>|x−<_ M _>_ ), which just ensures the fluency but does not consider the visual-text relations. 

## **3.3. Memory-augmented visual-related fusion score** 

To make the captions highly-related to the given image **I** , we need a visual guidance for the generation of words in the action of insertion and replacement. Motivated by the widely-used CLIP contrastive score for evaluating the visual-text similarity, we develop a memory-augmented visual-related fusion score to adapt the original word prediction distribution of CBART to tie with the given image, considering both _i) image-text cross-modal_ similarity and _ii) text-text in-modal_ similarity. 

Specifically, when sampling[1] a word _xi_ at position _i_ , CBART first predicts a conditional probability _p[lm]_ and select top- _Kw_ candidate words _{xik}[K] k_ =1 _[w]_[with][the][corre-] sponding fluent score, as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0005-07.png)


Then _Kw_ candidate sentences _{sk_ = ( _x_ 1 _, ..., xik, ..., xn_ ) _}[K] k_ =1 _[w]_[are][formed][by][combining][candi-] date word _xik_ with the context _x−i_ . 

_i) image-text cross-modal similarity_ . This similarity is denoted as _p[IT s]_ , which can be computed by taking candidate sentences _{sk}[K] k_ =1 _[w]_[and the image] **[ I]**[ as input to calculate] the CLIP cross-modality similarity as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0005-10.png)


_i) text-text in-modal similarity_ . Notice that the retrieved memory _{mn}[N] n_ =1 _[d]_[in Eq. (][1][) is also image related.][There-] fore, we introduce a memory-augmented visual-related similarity as _p[T T s]_ to further improve the image-caption correlation by using Sentence-BERT text encoder _St_ to evaluate the similarity between _{sk}[K] k_ =1 _[w]_[and] _[ {][m][n][}] n[N]_ =1 _[d]_[as] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0005-12.png)


Finally, after a weighted sum of Eq. (7), Eq. (8) and Eq. (9), we have the memory-augmented visual-related fusion score as 

> 1No matter for replacement or insertion, the essence is the same, _i.e_ ., sampling a word to replace the mask. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0005-15.png)


As a result, when sampling _i_ -th word for replacement or insertion in CBART for our model, we select the candidate word with the highest fusion score as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0005-17.png)


Up to now, our proposed MeaCap can achieve trainingfree zero-shot IC with less hallucination, which is named as **MeaCap** TF in the experiments. 

Moreover, Like most of text-only-training zero-shot IC models [52, 56] that just use text to fine-tune the language model, we can also fine-tune the CBART firstly and then perform text-only zero-shot IC, which is named as **MeaCap** ToT in the experiments. 

## **4. Experiments** 

To demonstrate that **MeaCap** can efficiently achieve impressive performance in different zero-shot settings, we follow the previous works [13, 30] to conduct comprehensive experiments on _Task One_ : zero-shot IC in Sec. 4.1, and _Task Two_ : unpaired IC in Sec. 4.2. For each setting, we report both results of training-free version **MeaCap** TF and the text-only-training version **MeaCap** ToT. In Sec. 4.3, we further evaluate the validity of our proposed memory-based zero-shot IC framework with other LM. In Sec. 4.4, we conduct detailed ablation studies for MeaCap. 

**Dataset** . We conduct experiments on three widely used image captioning benchmarks, _i.e._ MSCOCO [33], Flickr30K [64], and NoCaps [1]. For MSCOCO and Flickr30K dataset, we follow previous works [8, 12, 13, 30] and use Karpathy split [20]. We use the validation set of NoCaps to evaluate the transferability of IC models trained on other datasets. Besides, for Task One, we follow previous works [30] that transfer the model from a web-scale corpus CC3M [51] to MSCOCO and NoCaps. CC3M contains three million image-description pairs collected from the web and we only use the text for building the memory or finetuning the LM. 

**Implementation Details.** There are various pre-trained modules used in MeaCap. _i) CLIP_ : we use the pretrained VIT-B/32 CLIP. _ii) Sentence-BERT_ : we use the pre-trained model from HuggingFace[2] . _iii) CBART_ : we use the pre-trained model on One-Billion-Word corpus[3] . _iv) TextGraphParser_ : we use the off-the-shelf textual scene graph extractor [32]. For the training-free version MeaCapTF, we concat a prefix “The image above depicts that” at the start position of the sentence. For the text-only-training version MeaCapToT, we further fine-tune the CBART on the corresponding training corpus with 

> 2https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 

> 3https://www.statmt.org/lm-benchmark/ 

5 

|Methods|Text Corpus<br>Training<br>Memory|MSCOCO<br>B@4<br>M<br>C<br>S<br>CLIP-S<br>BLIP2-S|NoCap val (CIDEr)<br>In<br>Near<br>Out<br>Overall|NoCap val (CIDEr)<br>In<br>Near<br>Out<br>Overall|
|---|---|---|---|---|
|ZeroCap [55]<br>Tewel_et al._ [54]<br>ConZIC [65]<br>CLIPRe [30]<br>DeCap [30]|✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>✗<br>CC3M<br>CC3M<br>CC3M|2.6<br>11.5<br>14.6<br>5.5<br>0.87<br>0.70<br>2.2<br>12.7<br>17.2<br>7.3<br>0.74<br>0.68<br>1.3<br>11.2<br>13.3<br>5.0<br>**1.00**<br>0.76<br>4.6<br>13.3<br>25.6<br>9.2<br>0.84<br>0.70<br>8.8<br>16.0<br>42.1<br>10.9<br>0.76<br>-|13.3<br>13.7<br>15.4<br>23.3<br>34.8|14.9<br>19.7<br>16.6<br>15.8<br>18.3<br>16.9<br>16.0<br>20.3<br>17.5<br>26.8<br>36.5<br>28.2<br>37.7<br>49.9<br>39.7|
|**MeaCap**TF<br>**MeaCap**ToT|✗<br>CC3M<br>CC3M<br>CC3M|7.1<br>16.6<br>42.5<br>11.8<br>0.84<br>**0.81**<br>**9.0**<br>**17.8**<br>**48.3**<br>**12.7**<br>0.79<br>0.75|35.3<br>**38.5**|39.0<br>45.1<br>40.2<br>**43.6**<br>**50.0**<br>**45.1**|



Table 1. Zero-shot captioning results on MSCOCO Karpathy-test split and NoCaps validations set. In, Near, and Out denote in-domain, near domain, and out-of-domain. MeaCapTF is the training-free version and MeaCapToT is text-only training version. 

AdamW [25] optimizer. For Task One, we CC3M to serve as the memory, while for Task Two, we use the training corpus of source dataset as the memory. We set the concept similarity threshold _τ_ = 0 _._ 55 for CC3M memory and _τ_ = 0 _._ 6 for other memories. _Nd, Kw,_ _**α** ,_ _**β** ,_ _**γ**_ are set as 5 _,_ 200 _,_ 0 _._ 1 _,_ 0 _._ 4 _,_ 0 _._ 2 among all experiments. All experiments are conducted on a single RTX3090 GPU. We preprocess the textual corpus into text embeddings by CLIP text encoder and store text embeddings as our memory for fast retrieval. For example, retrieval on CC3M costs an average of 0.05s on RTX3090 GPU or an average of 1s on CPU. 

**Metrics** . To evaluate the accuracy of the generated caption, we use the traditional supervised metrics BLEU (B@n) [43], METEOR (M) [4], CIDEr (C) [58], and SPICE (S) [2] which compute the similarity between candidate sentences and human references. As for training-free methods, we use the CLIPScore (CLIP-S) [17] to measure the imagetext similarity. Additionally, considering that CLIP-S is insensitive to the hallucination of those CLIP-based methods as shown in Fig. 1b, we employ another pre-trained large model BLIP-2 [29] to evaluate image-text similarity, _i.e._ BLIP2Score (BLIP2-S). More details about BLIP2-S are in the Supplement. 

## **4.1. Zero-shot image captioning** 

In this section, we conduct zero-shot IC experiments to evaluate the ability of models to transfer from a general web-collected corpus to different downstream IC datasets. 

**Baselines.** In this study, we compare two types of baselines. _i)_ Training-free methods: ZeroCap [55], Tewel _et al._ [54] and ConZIC [65]. Those methods leverage pretrained CLIP and freezed LM (BERT or GPT-2) to achieve zero-shot IC. _ii)_ Text-only-training methods[4] : DeCap [30], which is also a memory-based method discussed in Sec. 2.3. Instead of using pre-trained LM, DeCap trains a language decoder from scratch. Besides, authors of Decap set up a baseline called CLIPRe, which generate image descriptions by retrieving the most relevant texts from memory 

> 4Other text-only-training methods except DeCap do not have experimented in this setting, we compared them in Task Two (Sec. 4.2) 

|Methods|MSCOCO<br>B@4<br>M<br>C<br>S|MSCOCO<br>B@4<br>M<br>C<br>S|Flickr30K<br>B@4<br>M<br>C<br>S|
|---|---|---|---|
||Training on image-text pairs|||
|Bottom-Up[3]<br>OSCAR[31]<br>VinVL[66]<br>ClipCap[39]<br>SmallCap[47]<br>I-Tuning[37]|36.2<br>27.0<br>113.5<br>20.3<br>36.5<br>30.3<br>123.7<br>23.1<br>40.9<br>30.9<br>140.6<br>25.1<br>33.5<br>27.5<br>113.1<br>21.1<br>37.0<br>27.9<br>119.7<br>21.3<br>34.8<br>28.3<br>116.7<br>21.8||27.3<br>21.7<br>56.6<br>16.0<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>25.2<br>22.8<br>61.5<br>16.9|
||Text-only-training, zero-shot inference|||
|ZeroCap_†_[55]<br>MAGIC [52]<br>ZEROGEN [56]<br>CLIPRe [30]|7.0<br>15.4<br>12.9<br>17.4<br>15.5<br>18.7<br>12.4<br>20.4|49.3<br>9.2<br>49.3<br>11.3<br>55.4<br>12.1<br>53.4<br>14.8|5.4<br>11.8<br>16.8<br>6.2<br>6.4<br>13.1<br>20.4<br>7.1<br>13.1<br>15.2<br>26.4<br>8.3<br>9.8<br>18.2<br>31.7<br>12.0|
|**MeaCap**TF<br>**MeaCap**ToT|9.1<br>20.6<br>**17.7**<br>**24.3**|56.9<br>15.5<br>**84.8**<br>**18.7**|7.2<br>17.8<br>36.5<br>13.1<br>**15.3**<br>**20.6**<br>**50.2**<br>**14.5**|



Table 2. In-domain captioning results on MSCOCO Karpathytest split and Flickr30K Karpathy-test split. _†_ means text-only reimplemented version from [52]. 

|Methods|MSCOCO_→_Flickr30k<br>B@4<br>M<br>C<br>S|MSCOCO_→_Flickr30k<br>B@4<br>M<br>C<br>S|MSCOCO_→_Flickr30k<br>B@4<br>M<br>C<br>S|Flickr30k_→_MSCOCO<br>B@4<br>M<br>C<br>S|
|---|---|---|---|---|
|MAGIC [52]<br>CLIPRe [30]||6.2<br>12.2<br>9.8<br>16.7|17.5<br>5.9<br>30.1<br>10.3|5.2<br>12.5<br>18.3<br>5.7<br>6.0<br>16.0<br>26.5<br>10.2|
|**MeaCap**TF<br>**MeaCap**ToT||7.1<br>16.6<br>**13.4**<br>**18.5**|34.4<br>11.4<br>**40.3**<br>**12.1**|7.4<br>16.2<br>46.4<br>11.2<br>**9.8**<br>**17.4**<br>**51.7**<br>**12.0**|



Table 3. Cross domain captioning results on MSCOCO and Flickr30K Karpathy-test split. 

directly. Following Decap, for MeaCapTF, we just use CC3M as the memory, and for MeaCapToT, we use CC3M as the memory and also tuning the CBART. Tab. 1 shows the results on MSCOCO and NoCaps, and MeaCap achieves new state-of-the-art results. **Training-free Results.** Concretely, our training-free version MeaCapTF has shown superior performance on reference-based metric (B@4, M, C, S) than all previous training-free baselines, ZeroCap, Tewel _et al._ and ConZIC on both MSCOCO and NoCaps datasets by a large margin, demonstrating the effectiveness of our memory-augmented design. For reference-free metrics (CLIP-S and BLIP2-S), MeaCapTF achieves better results on BLIP2-S and is inferior on CLIP-S. As discussed in the introduction, previous training-free methods are favored by CLIP-S because of the hallucination phenomenon. 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0007-00.png)


**GT** :A group of skiers are gathered together as they get ready to ski. **ConZIC** : A California commercial filming undergraduate college students in google photo. **ZeroCap** : A video crew showing the scene of a recent study. **MAGIC** : Group of people skiing down a snowy hill. **DeCap** : A group of people on skis are standing in the snow 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0007-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0007-03.png)


**GT** :There are many men preparing to cut a **GT** : A giant airplane sitting on the tarmac of red ribbon. an airport. **ConZIC** : A urban roadway opening with a **ConZIC** : A japanese jet airliner and truck bicycle wheel beside the marin county park running brown and blue at kyoto midway attorney office. terminal. **ZeroCap** : A recent opening in San Gabriel **ZeroCap** : A Boeing 707 in Japan. bikeway. **MAGIC** :A blue airplane sitting on top of a **MAGIC** :A picture of a stop sign with a man runway. standing behind it. **DeCap** : A passenger jet parked on the tarmac **DeCap** :A man in a suit and tie is giving a at an airport . bike line to a business sign . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0007-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0007-06.png)


**GT** : A dragon flying through cloud on a **GT** : A bedroom has many posters on the wall. sketch artwork. **ConZIC** : A bedroom page surrounded by **ConZIC** : A flying dragon genre design nine pink music student posters. completed with charcoal on paper. **ZeroCap** : A bedroom with posters on the **ZeroCap** : A dragon in the form of the dragon. walls. **MAGIC** : A bird riding on a large body of **MAGIC** : A picture of a bedroom with a lot of water. pictures on the wall. **DeCap** : A person on a big kite flying through **DeCap** : A bedroom with a bed and many the air. pictures on the wall . 

[ **people** , **ski poles** ] [ **ribbon** ] [ **airport** , **tarmac** ] [ **dragon** ] [ **bedroom** , **posters** , **wall** ] 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: Group of **people** with **ski poles** 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: Someone cutting the **ribbon** . 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: An airliner parked behind a jet at 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: A **dragon** sketch of the artwork. 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: A **bedroom** with various and snow boards outdoors. 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: A **ribbon** cutting ceremony on **airport tarmac** . 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: An illustration of a mythical **posters** and paintings on the **wall** . 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: A group of **people** standing a street. 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: A large jet airliner sitting on an **dragon** in flight. 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: A **bedroom** with many **posters** around with **ski poles** on. **airport tarmac** . on the **wall** and a bed. 

Figure 3. Examples of zero-shot IC compared with other zero-shot baselines. GT denotes the Ground Truth. ConZIC and ZeroCap are training-free, while MAGIC and DeCap are text-only-training. MeaCap displays the extracted concepts in green and generated caption. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0007-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0007-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0007-12.png)


**ConZIC** : A lakers player peeking through his sleeves prior to retiring. **ZeroCap** : A great NBA star dead. **MAGIC** :A man in a wetsuit with a wetsuit with a big. **DeCap** : A man that is in the middle of a game with a tennis racket . 

**ConZIC** : Triangular tower picture **ConZIC** : A dark knight at virtual domain about france representing a gray landscape website. background shaded. **ZeroCap** : A French landmark is **ZeroCap** : A Dark Knight in the the name of the song. film. **MAGIC** : A view of a big tower with **MAGIC** : A black and white photo a clock on it. of a black and white zebra. **DeCap** : A tower that is in the **DeCap** : A man that is standing in center of a tall tower . the dark. 

[ **basketball, shooting guard** ] [ **tower** ] [ **character** , **batman** ] 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: The **basketball** star 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: The famous Eiffel 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: A fictional **character shooting guard** Kobe. **tower** in Paris. known as the **batman** . 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: The **basketball** star 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: The famous **tower** 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: A character of of **shooting guard** . of tourist attraction. **batman** in the picture. 

Figure 4. Examples of real-world knowledge. MeaCapToT can alleviate the world-knowledge-forgotten problem of existing textonly-training methods, such as ”batman” in the third image. 

Besides, Our MeaCapTF also surpasses the retrieval-based baseline CLIPRe by a large margin, indicating that only retrieving the most relevant caption is deficient in accuracy. Moreover, even compared with the text-only-training method Decap, MeaCapTF shows superior or comparable performance on both MSCOCO and NoCap. 

**Text-only-training Results.** To explore the potential of our MeaCap with further text-only-training on the webscale corpus following DeCap, we also fine-tune CBART on CC3M corpus, _i.e._ MeaCapToT. It can be observed that MeaCapToT significantly improves the performance, especially on NoCap. Specifically, under the same training and memory condition, MeaCapToT surpasses DeCap in both the MSCOCO dataset and the NoCaps dataset, showing the superiority of our method to use the external memory. 

**Qualitative results.** Besides quantitative compare, we 

visualize the generated captions in Fig. 1, Fig. 3, and Fig. 4. Clearly, our proposed MeaCap can achieve better caption with more knowledge and less hallucination. 

## **4.2. Task Two: Unpaired image captioning** 

## **4.2.1 In-domain captioning** 

To explore more potential of MeaCap under an in-domain setting, which means the training data, the memory, and the test set are from the same dataset, but do not use image-text pairs to build the model and memory. 

**Baselines.** In this study, we compare with other textonly-training methods ZeroCap _†_ [55], MAGIC [52], and ZEROGEN [56] and a retrieval-based approach CLIPRe. ZeroCap is a training-free method which is extended to textonly-training version ZeroCap _†_ [55]. Those methods freeze the CLIP and fine-tune the LM on corresponding training texts. Under the in-domain setting, we also report both the training-free version MeaCapTF, which only employs the training text as memory, and the text-only-training version MeaCapToT which utilizes the training text to fine-tune CBART and serve as memory as well. 

**Results.** As shown in Tab. 2, MeaCapTF outperforms CLIPRe and other text-only-training baselines on C and S scores. Compared with B@4 and M scores, The C and S scores pay more attention to the accuracy of entities and relationships. The superior performance on these two scores demonstrates the high quality of our proposed memorybased retrieval-then-filter method to get the key concepts. Moreover, MeaCapToT outperforms all baselines by a large margin, indicating that our proposed method has greater potential with further in-domain training. 

## **4.2.2 Cross-domain captioning** 

We evaluate the MeaCap in a cross-domain setting where the training data and testing data are from different datasets. We use the text from the source domain as the memory for MeaCapTF and MeaCapToT, and fine-tune the CBART for 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
40<br>30<br>20<br>10<br>0<br>1 3 5 7 9<br>Number of retrieved memory captions<br>Performance<br>**----- End of picture text -----**<br>


Figure 5. Effect of the number of retrieved memory captions. We reported the performance of MeaCapTF on the MSCOCO dataset with varying the number of retrieved memory captions. 

|Methods|B@4|MSCOCO<br>M<br>C<br>S|MSCOCO<br>M<br>C<br>S|Flickr30K<br>B@4<br>M<br>C<br>S|Flickr30K<br>B@4<br>M<br>C<br>S|
|---|---|---|---|---|---|
|DeCap [30]<br>CapDec [41]<br>ViECap [13]<br>**MeaCap**InvLM|24.7<br>26.4<br>**27.2**<br>**27.2**|25.0<br>91.2<br>18.7<br>25.1<br>91.8<br>-<br>24.8<br>92.9<br>18.2<br>**25.3**<br>**95.4**<br>**19.0**||21.2<br>21.8<br>56.7<br>15.2<br>17.7<br>20.0<br>39.1<br>-<br>21.4<br>20.1<br>47.9<br>13.6<br>**22.3**<br>**22.3**<br>**59.4**<br>**15.6**||
||MSCOCO_→_Flickr30K|||Flickr30K_→_MSCOCO||
|DeCap [30]<br>CapDec [41]<br>ViECap [13]<br>**MeaCap**InvLM|16.3<br>17.9<br>17.3<br>18.6<br>17.4<br>18.0<br>**18.5**<br>**19.5**||35.7<br>11.1<br>35.7<br>-<br>38.4<br>11.2<br>**43.9**<br>**12.8**|12.1<br>9.2<br>12.6<br>**13.1**|18.0<br>44.4<br>10.9<br>16.3<br>27.3<br>-<br>19.3<br>54.2<br>12.5<br>**19.7**<br>**56.4**<br>**13.2**|



Table 4. In-domain and cross-domain captioning results with CLIP-invert language decoder. 

## MeaCapToT. 

**Results.** We compare our method with the text-onlytraining baseline MAGIC which fine-tune the GPT-2, and a retrieval-based baseline CLIPRe. Results in Tab. 3 show that MAGIC with finetuning the GPT-2 on source data suffers a performance degradation on target data, even worse than the retrieval-based method CLIPRe. Equipped with the proposed memory-augmented design, MeaCapTF surpasses the CLIPRe on most metrics and MeaCapToT outperforms all baselines a lot, demonstrating the effectiveness of the proposed memory-augmented design. 

## **4.3. Flexibility of MeaCap with other LM** 

It is important to emphasize that our proposed memory mechanism for finding key concepts in Sec. 3.1 is a plugand-play module to further improve most of the existing text-only-training SOTA methods [13, 30, 41] on Task Two. For this end, we just replace the CBART (Sec. 3.2) in current MeaCap with the another LM used in these methods [13, 30, 41] (also do not need to use the fusion score in Sec. 3.3), which is described detailed below. 

**Baselines.** DeCap [30], CapDec [41] and ViECap [13] train a LM from scratch to invert the CLIP text encoder, denoted as InvLM in the following. They project the visual embeddings extracted by the CLIP visual encoder to the text embedding space of the CLIP text encoder. Then, they use InvLM to reconstruct the text from text embeddings. To 

|Methods|**ReF**<br>**ITs**<br>**TTs**|MSCOCO<br>B@4<br>M<br>C<br>S|
|---|---|---|
|MeaCapTF|✔<br>✗<br>✗<br>✗<br>✔<br>✗<br>✔<br>✔<br>✗<br>✔<br>✔<br>✔|5.0<br>13.3<br>31.1<br>5.6<br>1.8<br>9.7<br>12.7<br>4.8<br>5.7<br>13.6<br>38.6<br>8.5<br>7.1<br>16.6<br>42.5<br>11.8|
|MeaCapToT|✔<br>✗<br>✗<br>✗<br>✔<br>✗<br>✔<br>✔<br>✗<br>✔<br>✔<br>✔|7.9<br>14.9<br>37.1<br>10.4<br>3.2<br>9.9<br>17.3<br>5.2<br>8.1<br>15.6<br>44.7<br>11.1<br>9.0<br>17.8<br>48.3<br>12.7|



Table 5. Ablation studies on zero-shot image captioning. ReF, ITs, TTs denote the retrieve-and-filter module, ITs (8) and TTs (1) are image-text similarity and text-text similarity from our memoryaugmented visual-related score, respectively. 

generate descriptions based on our extracted key concepts, we first use a prompt template as “There are [ _c_ 1 _, c_ 2 _, ..., cn_ ] in the image” to inject the concepts into a concept-aware sentence following ViECap, where _cn_ are the _n_ -th concepts. After encoding the concept-aware sentence to text embeddings by CLIP text encoder, we get a concept-aware prompt. We concat the concept-aware prompt with textual embeddings as the input of InvLM. We call this version MeaCap as **MeaCap** InvLM 

**Results.** Tab. 4 shows that MeaCapInvLM outperforms all baselines on all metrics under in-domain and crossdomain scenarios, demonstrating the effectiveness of our proposed memory-based key concepts, and also indicating its flexibility for various LM and different zero-shot settings. More details are shown in the Supplement. 

## **4.4. Ablation studies** 

To explore the impact of each key module in MeaCap, _i.e._ the retrieve-then-filter module ( **ReF** ), the image-text similarity score ( **ITs** ), and the text-text similarity score ( **TTs** ), we conduct comprehensive ablation studies on the MSCOCO dataset based on the Task One of zero-shot setting. We evaluate both the training-free version MeaCapTF and the text-only training version MeaCapToT whose results are provided in Tab. 5. As we can see, only combined with the ReF and original LM (the first row) can surpass the only ITs results in the second row (ITs is the only visual guidance of previous training-free methods by CLIP), indicating the key concepts extracted by the ReF module are critical for zero-shot IC. The third row shows that combining ReF with ITs yields more improvements than individual modules alone. Finally, by incorporating the TTs, the performance is further improved, highlighting the efficacy of the memory-augmented visual-related fusion score. We also conduct experiments to investigate the impact of the number of retrieved memory captions as shown in Fig. 5. Our model achieves best performance when retrieving five memory captions. 

8 

## **5. Conclusion** 

In this paper, we propose a novel memory-augmented zeroshot image captioning framework, namely MeaCap. We introduce a retrieve-then-filter module to extract key concepts from external textual memory. Based on the retrieved textual memory, we further develop a memory-augmented visual-related fusion score to guide the generation of captions. Combined with CBART, we can generate conceptcentered descriptions to alleviate the hallucination of previous training-free methods and enhance the accuracy of text-only-training methods. Extensive experiments on various zero-shot captioning settings show that our proposed MeaCap outperforms previous training-free and text-onlytraining methods. 

## **References** 

- [1] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 8948–8957, 2019. 1, 5 

- [2] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In _European conference on computer vision_ , pages 382–398. Springer, 2016. 6 

- [3] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6077–6086, 2018. 2, 6 

- [4] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , pages 65–72, 2005. 2, 6 

- [5] Andreas Blattmann, Robin Rombach, Kaan Oktay, Jonas M¨uller, and Bj¨orn Ommer. Retrieval-augmented diffusion models. _Advances in Neural Information Processing Systems_ , 35:15309–15324, 2022. 3 

- [6] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. Improving language models by retrieving from trillions of tokens. In _International conference on machine learning_ , pages 2206–2240. PMLR, 2022. 3 

- [7] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. _arXiv preprint arXiv:2210.11416_ , 2022. 12 

- [8] Marcella Cornia, Matteo Stefanini, Lorenzo Baraldi, and Rita Cucchiara. Meshed-memory transformer for image captioning. In _Proceedings of the IEEE/CVF conference on com-_ 

_puter vision and pattern recognition_ , pages 10578–10587, 2020. 1, 2, 5 

- [9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018. 1 

- [10] Jeffrey Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell. Long-term recurrent convolutional networks for visual recognition and description. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2625–2634, 2015. 2 

- [11] Zhibin Duan, Lv Zhiyi, Chaojie Wang, Bo Chen, Bo An, and Mingyuan Zhou. Few-shot generation via recalling braininspired episodic-semantic memory. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 3 

- [12] Zhiyuan Fang, Jianfeng Wang, Xiaowei Hu, Lin Liang, Zhe Gan, Lijuan Wang, Yezhou Yang, and Zicheng Liu. Injecting semantic concepts into end-to-end image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18009–18019, 2022. 5 

- [13] Junjie Fei, Teng Wang, Jinrui Zhang, Zhenyu He, Chengjie Wang, and Feng Zheng. Transferable decoding with visual entities for zero-shot image captioning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 3136–3146, 2023. 2, 3, 5, 8, 12 

- [14] Jiuxiang Gu, Gang Wang, Jianfei Cai, and Tsuhan Chen. An empirical study of language cnn for image captioning. In _Proceedings of the IEEE international conference on computer vision_ , pages 1222–1231, 2017. 2 

- [15] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pretraining. In _International conference on machine learning_ , pages 3929–3938. PMLR, 2020. 3 

- [16] Xingwei He. Parallel refinements for lexically constrained text generation with bart. _arXiv preprint arXiv:2109.12487_ , 2021. 2, 3, 4 

- [17] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. _arXiv preprint arXiv:2104.08718_ , 2021. 2, 6, 12 

- [18] Lun Huang, Wenmin Wang, Jie Chen, and Xiao-Yong Wei. Attention on attention for image captioning. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 4634–4643, 2019. 1, 2 

- [19] Lun Huang, Wenmin Wang, Yaxian Xia, and Jie Chen. Adaptively aligned image captioning via adaptive attention time. _Advances in neural information processing systems_ , 32, 2019. 2 

- [20] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3128–3137, 2015. 5 

- [21] Vladimir Karpukhin, Barlas O˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wentau Yih. Dense passage retrieval for open-domain question answering. _arXiv preprint arXiv:2004.04906_ , 2020. 3 

9 

- [22] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. _arXiv preprint arXiv:1911.00172_ , 2019. 3 

- [23] Urvashi Khandelwal, Angela Fan, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Nearest neighbor machine translation. _arXiv preprint arXiv:2010.00710_ , 2020. 3 

- [24] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Visionand-language transformer without convolution or region supervision. In _International Conference on Machine Learning_ , pages 5583–5594. PMLR, 2021. 1 

- [25] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 6 

- [26] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International journal of computer vision_ , 123(1):32–73, 2017. 1 

- [27] Chia-Wen Kuo and Zsolt Kira. Beyond a pre-trained object detector: Cross-modal textual and visual context for image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17969– 17979, 2022. 2 

- [28] Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Latent retrieval for weakly supervised open domain question answering. _arXiv preprint arXiv:1906.00300_ , 2019. 3 

- [29] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 6, 12 

- [30] Wei Li, Linchao Zhu, Longyin Wen, and Yi Yang. Decap: Decoding clip latents for zero-shot captioning via text-only training. _arXiv preprint arXiv:2303.03032_ , 2023. 2, 3, 5, 6, 8 

- [31] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In _European Conference on Computer Vision_ , pages 121–137. Springer, 2020. 6 

- [32] Zhuang Li, Yuyang Chai, Terry Zhuo Yue, Lizhen Qu, Gholamreza Haffari, Fei Li, Donghong Ji, and Quan Hung Tran. Factual: A benchmark for faithful and consistent textual scene graph parsing. _arXiv preprint arXiv:2305.17497_ , 2023. 4, 5 

- [33] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _European conference on computer vision_ , pages 740–755. Springer, 2014. 1, 5 

- [34] Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X Yu. Large-scale long-tailed recognition in an open world. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 2537–2546, 2019. 3 

- [35] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: 

Hierarchical vision transformer using shifted windows. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 10012–10022, 2021. 1 

- [36] Alexander Long, Wei Yin, Thalaiyasingam Ajanthan, Vu Nguyen, Pulak Purkait, Ravi Garg, Alan Blair, Chunhua Shen, and Anton van den Hengel. Retrieval augmented classification for long-tail visual recognition. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 6959–6969, 2022. 3 

- [37] Ziyang Luo, Zhipeng Hu, Yadong Xi, Rongsheng Zhang, and Jing Ma. I-tuning: Tuning frozen language models with image for lightweight image captioning. In _ICASSP 20232023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 1–5. IEEE, 2023. 6 

- [38] Yuxian Meng, Shi Zong, Xiaoya Li, Xiaofei Sun, Tianwei Zhang, Fei Wu, and Jiwei Li. Gnn-lm: Language modeling based on global contexts via gnn. _arXiv preprint arXiv:2110.08743_ , 2021. 3 

- [39] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. _arXiv preprint arXiv:2111.09734_ , 2021. 6 

- [40] Van-Quang Nguyen, Masanori Suganuma, and Takayuki Okatani. Grit: Faster and better image captioning transformer using dual visual features. In _European Conference on Computer Vision_ , pages 167–184. Springer, 2022. 1, 2 

- [41] David Nukrai, Ron Mokady, and Amir Globerson. Text-only training for image captioning using noise-injected clip. _arXiv preprint arXiv:2211.00575_ , 2022. 2, 3, 8 

- [42] Yingwei Pan, Ting Yao, Yehao Li, and Tao Mei. X-linear attention networks for image captioning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 10971–10980, 2020. 1, 2 

- [43] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pages 311–318, 2002. 2, 6 

- [44] Yu Qin, Jiajun Du, Yonghua Zhang, and Hongtao Lu. Look back and predict forward in image captioning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8367–8375, 2019. 2 

- [45] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 1 

- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_ , pages 8748–8763. PMLR, 2021. 2, 12 

- [47] Rita Ramos, Bruno Martins, Desmond Elliott, and Yova Kementchedjhieva. Smallcap: lightweight image captioning prompted with retrieval augmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2840–2849, 2023. 3, 6 

- [48] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In _Proceedings of_ 

10 

   - _the 2019 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics, 2019. 4 

- [49] Idan Schwartz, Alexander Schwing, and Tamir Hazan. Highorder attention models for visual question answering. _Advances in Neural Information Processing Systems_ , 30, 2017. 1, 2 

- [50] Idan Schwartz, Alexander G Schwing, and Tamir Hazan. A simple baseline for audio-visual scene-aware dialog. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 12548–12558, 2019. 1, 2 

- [51] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2556–2565, 2018. 5 

- [52] Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier. Language models can see: Plugging visual controls in text generation. _arXiv preprint arXiv:2205.02655_ , 2022. 2, 3, 5, 6, 7 

- [53] Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. A contrastive framework for neural text generation. _Advances in Neural Information Processing Systems_ , 35:21548–21561, 2022. 2 

- [54] Yoad Tewel, Yoav Shalev, Roy Nadler, Idan Schwartz, and Lior Wolf. Zero-shot video captioning with evolving pseudotokens. _arXiv preprint arXiv:2207.11100_ , 2022. 1, 2, 3, 6 

- [55] Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf. Zerocap: Zero-shot image-to-text generation for visualsemantic arithmetic. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17918–17928, 2022. 1, 2, 3, 6, 7 

      - eration with visual attention. In _International conference on machine learning_ , pages 2048–2057. PMLR, 2015. 2 

   - [62] Xu Yang, Kaihua Tang, Hanwang Zhang, and Jianfei Cai. Auto-encoding scene graphs for image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10685–10694, 2019. 2 

   - [63] Ting Yao, Yingwei Pan, Yehao Li, and Tao Mei. Exploring visual relationship for image captioning. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 684–699, 2018. 2 

   - [64] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. _Transactions of the Association for Computational Linguistics_ , 2:67–78, 2014. 1, 5 

   - [65] Zequn Zeng, Hao Zhang, Ruiying Lu, Dongsheng Wang, Bo Chen, and Zhengjue Wang. Conzic: Controllable zero-shot image captioning by sampling-based polishing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 23465–23476, 2023. 1, 2, 3, 6 

   - [66] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5579– 5588, 2021. 6 

   - [67] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. _arXiv preprint arXiv:2205.01068_ , 2022. 12 

- [56] Haoqin Tu, Bowen Yang, and Xianfeng Zhao. Zerogen: Zero-shot multimodal controllable text generation with multiple oracles. _arXiv preprint arXiv:2306.16649_ , 2023. 2, 3, 5, 6, 7 

- [57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 1 

- [58] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4566–4575, 2015. 2, 6 

- [59] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3156–3164, 2015. 2 

- [60] Li Wang, Zechen Bai, Yonghua Zhang, and Hongtao Lu. Show, recall, and tell: Image captioning with recall mechanism. In _Proceedings of the AAAI conference on artificial intelligence_ , pages 12176–12183, 2020. 2 

- [61] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption gen- 

11 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0012-00.png)


**----- Start of picture text -----**<br>
soft prompt<br>CLIP Image<br>Encoder Projector<br>Language<br>Model<br>Entity ClassifierCLIP-based Parking meter There are meter in the image.parking<br>A black and white photo<br>𝐕𝐢𝐄𝐂𝐚𝐩 Prompt Template of a parking meter.<br>soft prompt<br>CLIP Image<br>Encoder Projector<br>Language<br>then filterRetrieve Parking meter, brick wall There are meter, the image.brick wall parking in  Model<br>A parking meter is shown<br>𝐌𝐞𝐚𝐂𝐚𝐩𝐈𝐧𝐯𝐋𝐌 Prompt Template against a brick wall.<br>**----- End of picture text -----**<br>


Figure 6. Difference between ViECap [13] and MeaCapInvLM. 

## **A. More details of MeaCap with other LM** 

To demonstrate the effectiveness of our proposed memory concepts, we incorporate our proposed retrieve-then-filter module with the current SOTA text-only-training method ViECap [13], namely MeaCapInvLM. ViECap is a text-onlytraining method that combines entity-aware hard prompts and learned visual soft prompts to generate image captions. The soft prompts are obtained by inputting the CLIP image embedding into the projector and the hard prompts are constructed by a CLIP-based entity classifier. As illustrated in Fig. 6, MeaCapInvLM remains the soft prompts branch and directly utilizes the pre-trained language model of ViECap. We only replace the entity classifier with our proposed retrieve-the-filter module to obtain key concepts and leverage the prompt template to inject the key concepts into a concept-aware sentence and get the hard prompts. Compared with the original ViECap which can only retrieve a single entity from pre-defined entity vocabulary, our proposed retrieve-then-filter module can extract more key visual concepts from the image, demonstrating that our proposed memory design can help to generate highly consistent descriptions with image content. Quantitative results under in-domain and cross-domain settings of MeaCapInvLM are shown in Tab. 4 in the main paper. 

## **B. The size of memory** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0012-05.png)


**----- Start of picture text -----**<br>
60<br>Training-free Text-only-training<br>50 48.8<br>42.5<br>40 38.4<br>32.1<br>29.2<br>30 25.6 27.4<br>20 15.7 16.3<br>13.6<br>10<br>0<br>1% 5% 20% 50% 100%<br>The ratio of the memory<br>Figure 7. Ablation study on memory size.<br>CIDEr<br>**----- End of picture text -----**<br>


large image-text corpus. We use the pre-trained BLIP-2 and leverage the image encoder _Bv_ and the text encoder _Bt_ to compute cross-modality cosine similarity. Following CLIPS [17], BLIP2-S is the product of a weight _w_ = 2 and the image-text similarity, as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0012-07.png)


where **I** , and **T** are the image and the text. Cos denotes the cosine similarity. 

## **D. More samples and qualitatives.** 

Fig. 8 and Fig. 9 have shown some generated results with our proposed methods. For each image, we have shown the intermediate results of the retrieve-then-filter module, _i.e._ the retrieved memory captions, the extracted subjectpredicate-object triplets, and the filtered key concepts. As we can see, we effectively filter the correct key concepts out. Based on the key concepts, MeaCapTF and MeaCapToT can generate captions with high consistency with image content. Fig. 9 showcases that the extracted key concepts from memory contain much world knowledge and can also alleviate the knowledge-forgotten phenomenon of text-only-training methods. 

To explore the impact of the external memory size, we randomly sample different ratio text embeddings of CC3M as our external memory. The results are shown in Fig. 7. Overall, the training-free MeaCapTF and text-only-training MeaCapToT both benefit from a large memory, and the performance increases as the memory size grows. It is due to that large memory usually covers more visual concepts and thus the retrieved captions are better aligned with the same image. 

## **C. BLIP2-S** 

BLIP-2 [29] is a large vision-language model that employs a frozen CLIP [46] image encoder and large language models [7, 67] to bootstrapping language-image pre-training on 

12 

**Retrieved memory captions:** 1. a tower of giraffes with the male centered , all chewing. 2. group of giraffes in the zoo. 3. guests observe giraffes from a high platform. 4. group of giraffes at the zoo. 5. herd of giraffes in the zoo. **Subject-predicate-object triplets:** 1. ( male , is , chewing ) , ( giraffes , is , tower ) , ( male , center on top of , giraffes ) 2. ( giraffes , is , group of ) , ( giraffes , in , zoo ) 3. ( giraffes , on , platform ) , ( guests , observe , giraffes ) , ( platform , is , high ) 4. ( giraffes , is , group of ) , ( giraffes , at , zoo ) 5. ( giraffes , in , zoo ) **Key concepts** : ['giraffes', 'zoo'] 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: The image depicts giraffes in a zoo enclosure surrounded. 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: A group of giraffes in a zoo surrounded by stones. **Retrieved memory captions:** 1. many people enjoy mushrooms on a pizza. 2. a pizza with black olives among the toppings. 3. pizza with mushrooms , tomato and olives in a box. 4. pizza with mushrooms , tomato and olives in a box. 5. close - up of vegetables on the pizza. **Subject-predicate-object triplets:** 1. ( people , enjoy , mushrooms ) , ( mushrooms , on , pizza ) 2. ( olives , is , black ) , ( olives , among , toppings ) , ( pizza , with , olives ) 3. ( pizza , in , box ) , ( pizza , with , tomatoes ) , ( pizza , with , olives ) , ( pizza , with , mushrooms ) 4. ( pizza , in , box ) , ( pizza , with , tomatoes ) , ( pizza , with , olives ) , ( pizza , with , mushrooms ) 5. ( vegetables , on , pizza ) **Key concepts** : ['vegetables', 'olives', 'toppings', 'box'] 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: The image depicts a pizza made of vegetables, olives and toppings in a box. 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: A pizza with vegetables, olives and other toppings in a box. **Retrieved memory captions:** 1. children reading the book on picnic in summer park. 2. children reading the book on picnic in summer park. 3. with young kids , dinner can feel like feeding time at the zoo. 4. young friends eating pizza in the park. 5. young friends eating pizza in the park. **Subject-predicate-object triplets:** 1. ( children , on , picnic ) , ( children , in , park ) , ( park , is , summer ) , ( children , read , book ) 2. ( children , on , picnic ) , ( children , in , park ) , ( park , is , summer ) , ( children , read , book ) 3. ( kids , is , young ) , ( kids , at , zoo ) , ( dinner , at , zoo ) , ( dinner , have , kids ) 4. ( friends , eat , pizza ) , ( friends , is , young ) , ( friends , in , park ) 5. ( friends , eat , pizza ) , ( friends , is , young ) , ( friends , in , park ) **Key concepts** : ['children', 'dinner', 'park'] 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: A picture showing two children eating dinner at a park. 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: Young children eating dinner in the park. **Retrieved memory captions:** 1. person smiling with a big plate of chocolate cake and vanilla ice cream in front of him. 2. portrait of a young man eating chocolate cake with a fork. 3. young man eating a slice of chocolate cake with a fork. 4. a man smiles , a chocolate cake on the table in front of him. 5. a man smiles , a chocolate cake on the table in front of him. **Subject-predicate-object triplets:** 1. ( plate , is , big ) , ( ice cream , is , vanilla ) , ( person , is , smiling ) , ( plate , in front of , person ) , ( cake , on , ’ plate ) , ( cake , is , chocolate ) , ( person , is , smiling ) , 2. ( man , is , young ) , ( man , eat , chocolate cake ) , ( man , with , fork ) 3. ( man , is , young ) , ( man , eat , chocolate cake ) , ( chocolate cake , is , slice ) , ( man , eat with , fork ) 4. ( man , is , smiling ) , ( chocolate cake , on , table ) , ( table , in front of , man ) 5. ( man , is , smiling ) , ( chocolate cake , on , table ) , ( table , in front of , man ) **Key concepts** : ['young man', 'chocolate cake', 'table', 'cafe'] 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: There are pictures of a young man smiling on holiday eating chocolate cake on the table in the cafe. 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: A young man smiling and eating chocolate cake on a table in the cafe. 

Figure 8. More qualitative results. 

13 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0014-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0014-01.png)


- **Retrieved memory captions:** 1. everything you need to know about the luxury sedan. 2. v8 portfolio : super long , light and performing sedan at a supercharged price. 3. first drive : adds sedan to its line – up. 

4. new black luxury sedan turning slightly to the right. 5. the car is the clearest idea yet of what to expect from the next generation luxury sedan. **Subject-predicate-object triplets:** 1. ( sedan , is , luxury ) 2. ( sedan , is , long ) , ( sedan , is , performing ) , ( sedan , is , supercharged ) , ( sedan , is , sedan ) 3. ( sedan , in , line ) 4. ( sedan , is , black ) , ( sedan , is , luxury ) , ( sedan , is , new ) , ( sedan , is , turning ) 5. ( sedan , is , luxury ) , ( sedan , is , next generation ) , ( car , is , clearest ) , ( car , is , idea ) **Key concepts** : [‘luxury sedan'] 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: The image depicts a luxury sedan with the BMW logo. 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: The BMW luxury sedan shown here. **Retrieved memory captions:** 1. iron man , pictured here , was one of the more overhauled heroes , due to severely outdated design. 2. actor takes the stage in a new poster for iron man. 3. iron man has a new poster with actor in some heat in film. 4. costume designer created a new suit for the film. 5. what do you think of the new poster for iron man? **Subject-predicate-object triplets:** 1. ( man , is , iron ) , ( design , is , severely outdated ) , ( man , have , design )’, 2. ( poster , is , new ) , ( poster , is , iron man ) , ( actor , take , stage ) , ( actor , in , poster )’, 3. ( poster , is , new ) , ( man , is , iron ) , ( man , have , poster ) , ( man , in , film ) , ( man , in , heat )’, 4. ( suit , is , new ) , ( costume designer , create , suit ) , ( suit , in , film )’, 5. ( poster , is , new ) , ( poster , is , iron man )’ **Key concepts** : [‘new poster’, ‘iron man'] 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: The image depicts the new poster as an iron man acting like a hero. 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: The new movie poster for iron man. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning_images/cvpr_2024_MeaCap_Memory-Augmented_Zero-shot_Image_Captioning.pdf-0014-04.png)


- **Retrieved memory captions:** 1. replica of the statue of liberty wrapped in flag stock photo – 20491789. 2. did you know the statue of liberty was once a dull copper color. 3. the statue of liberty contains a large amount of copper. 4. visit the statue of liberty during your itinerary. 5. visit the statue of liberty during your itinerary. **Subject-predicate-object triplets:** 1. ( statue , wrap in , flag ) , ( flag , is , stock photo ) , ( statue , is , replica ) 2. ( statue , is , copper ) , ( statue , is , dull ) 3. ( statue , contain , copper ) 4. ( statue of liberty ) 5. ( statue of liberty ) **Key concepts** : ['statue of liberty'] 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐅: A photo of the statue of liberty in America taken at sunset. 𝐌𝐞𝐚𝐂𝐚𝐩𝐓𝐨𝐓: The staue of liberty at sunset. 

Figure 9. More qualitative results with world knowledge. 

14 

