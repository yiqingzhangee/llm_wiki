# **EVCAP: Retrieval-Augmented Image Captioning with External Visual–Name Memory for Open-World Comprehension** 

Jiaxuan Li[1] _[∗]_ , Duc Minh Vo[1] _[∗]_ , Akihiro Sugimoto[2] , Hideki Nakayama[1] 1The University of Tokyo, Japan 2National Institute of Informatics, Japan _{_ li,vmduc _}_ @nlab.ci.i.u-tokyo.ac.jp sugimoto@nii.ac.jp nakayama@ci.i.u-tokyo.ac.jp 

## **Abstract** 

_Large language models (LLMs)-based image captioning has the capability of describing objects not explicitly observed in training data; yet novel objects occur frequently, necessitating the requirement of sustaining up-to-date object knowledge for open-world comprehension. Instead of relying on large amounts of data and/or scaling up network parameters, we introduce a highly effective retrievalaugmented image captioning method that prompts LLMs with object names retrieved from_ _**E** xternal_ _**V** isual–name memory (_ EVCAP _). We build ever-changing object knowledge memory using objects’ visuals and names, enabling us to (i) update the memory at a minimal cost and (ii) effortlessly augment LLMs with retrieved object names by utilizing a lightweight and fast-to-train model. Our model, which was trained only on the COCO dataset, can adapt to out-of-domain without requiring additional fine-tuning or re-training. Our experiments conducted on benchmarks and synthetic commonsense-violating data show that_ EVCAP _, with only 3.97M trainable parameters, exhibits superior performance compared to other methods based on frozen pre-trained LLMs. Its performance is also competitive to specialist SOTAs that require extensive training._ 

## **1. Introduction** 

Advanced image captioning based on large language models (LLMs) [3, 8, 9, 25] has focused on the approach using big-scale models trained on ever-increasingly large-scale datasets, which is no longer viable. This is because the computational cost to train the models increases exponentially and, more importantly, updating training data is almost impossible to keep pace with the growth of novel objects in our daily lives. Sustaining ever-changing object knowledge with a reasonable cost is a pressing concern in LLMs-based models to truly unlock open-world comprehension. 

Retrieval-augmented image captioning [20, 35] is 

*Equal contributions. Code is available at https://jiaxuan-li.github.io/EVCap. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0001-09.png)


**----- Start of picture text -----**<br>
● GT:  A person wearing ice skates on a wood floor.<br>● SmallCap:  A person riding a skateboard on top of a<br>wooden floor. ✕<br>● BLIP-2:  A person is skating on an ice rink. ✕<br>● EVCap:  A pair of ice skates on top of a wooden floor. ✓<br>PaLI CogVLM REVEAL BLIP-2<br>OSCAR ViECap RCA-NOC VinVL<br>ClipCap SmallCap MiniGPT4 EVCap (Ours)<br>145<br>135<br>125<br>115<br>105<br>95<br>85<br>75<br>65<br>55<br>85 95 105 115 125 135 145 155<br>CIDEr Score on COCO<br>CIDEr Score on NoCaps<br>**----- End of picture text -----**<br>


Figure 1. Overall comparison of our EVCAP and SOTAs. (Upper) Generated captions by SmallCap, BLIP-2, and our EVCAP for a commonsense-violating image from the WHOOPS dataset. _×_ and ✓ indicate incorrect and correct predictions, respectively. Incorrect objects in captions are highlighted in red , while correct ones are in blue . SmallCap and BLIP-2 give incorrect predictions for _“ice skates”_ and _“wood floor”_ , respectively, while our EVCAP utilizes an external visual–name memory to enhance attention to objects within the image, leading to superior performance for image captioning. (Lower) Comparison of the number of trainable parameters, CIDEr score on COCO and NoCaps datasets. The size of each circle reflects the log number of trainable parameters. EVCAP (3.97M) has less trainable parameters than others while achieving comparable results with SOTAs at scale. 

emerging as an alternative since it considerably reduces training costs in both time and data while producing encouraging results. Nonetheless, with their huge datastore, it is obvious that LLMs would imitate the given texts, limiting their ability to describe open-world objects properly. For instance, SmallCap [35] considers the words _“skateboard”_ and _“wooden floor”_ to be a pair regardless of visual appearances containing a commonsense-violating pair 

of _“ice skates”_ and _“wood floor”_ (Fig. 1, upper). Additionally, prompting the LLMs given a lot of retrieved texts becomes cumbersome, requiring more trainable parameters. Fig. 1 (lower) shows that the CIDEr scores obtained by a lightweight SmallCap [35] with 43M trainable parameters are far away from those obtained by a heavy REVEAL [20] with 2.1B trainable parameters. Beyond that, due to the frequent occurrence of new objects, access to their sample texts is not always feasible, making the memory utilized in [20, 35] difficult to grow. We thus aim to streamline the external memory used in previous work [20, 35] by storing a sufficiently small amount of object information. And, of course, not only does the model not stereotype the example sentences, but the number of trainable parameters would be reduced drastically as a result of the causation (Fig. 1). 

We follow [13, 40] to construct a key-value memory where the key is represented by object’s features, and the value corresponds to object’s name. Unlike [13, 40], which rely on object definition as the key, our method leverages the visual appearance of the object as the key because of the abundance of object images readily available on the internet. We propose an external visual–name memory tailored for ease of expansion and cost-effectiveness in upholding up-to-date object information. We present a highly effective retrieval-augmented LLMs-based image captioning method, called EVCAP, that prompts frozen LLMs with object names retrieved from our proposed memory for openworld comprehension. EVCAP contains a frozen image encoder ViT [14] and Q-Former [25] with _trainable_ image query tokens for object retrieval, an attentive fusion module, a _trainable_ linear layer for mapping between vision and language latent spaces, and a frozen LLM decoder [10] for generating captions. Specifically, the attentive fusion module feeds retrieved object names and visual features into a customized frozen Q-Former using _trainable_ object name query tokens to implicitly reduce the presence of superfluous object names. As a result, EVCAP amounts to only 3.97M trainable parameters. Once trained, the model can be adapted to new domains and large-scale data without further fine-tuning or re-training. Our contributions are as follows: 

- We provide an extensible external visual–name memory with minimal but useful object information, which enables LLMs-based models to comprehend the open world. 

- We present a highly efficacious retrieval-augmented image captioning EVCAP with 3.97M trainable parameters. On in-/out-domain benchmarks and synthetic 

- commonsense-violating dataset, EVCAP trained solely on COCO dataset competes with other lightweight methods by a margin while being on par with other specialist SOTAs. 

## **2. Related Work** 

**Image captioning** aims to describe the contents of a given image. It can be roughly divided into two approaches: non- 

LLMs-based methods and LLMs-based ones. The former approaches [4, 22, 42] typically employ a visual encoder and a language decoder in an end-to-end fashion to generate captions. However, they are incapable of describing open-world objects. The latter one leverages pre-trained large-scale vision models (CLIP [32], ViT [12]) and LLMs (GPTs [7, 31], T5 [33], LLaMA [37]) by bridging the gap between two modalities using either pre-training with largescale data or the learned mapper or prompt techniques. LLMs-based models [8, 9, 25, 29] demonstrate advancements in image captioning challenges, allowing the capacity to describe anything as long as pre-trained vision models can recognize it. Our method belongs to the LLMs-based approaches, but instead of relying fully on the pre-trained vision model, we use object names retrieved from the external memory to augment LLMs-based image captioning. **Novel object captioning** is a branch of image captioning that describes images containing objects that were not seen during training. Non-LLMs-based methods explore more objects by learning from unpaired image-sentence sources (DCC [19], NOC [39]) or rely on novel object detectors to recognize novel concepts (NBT [28], OSCAR [26] and VinVL [45]). LLMs-based methods such as ViECap [15] leverage the pre-trained CLIP [32] to obtain object entities. Nevertheless, the cut-off in training time of the pre-trained object detector or CLIP prevents it from detecting novel objects that arise quickly in reality. Unlike earlier work, we can readily update our recognition of novel concepts by adding them to external memory, ensuring that we keep any new objects from the past and even the future. 

**Retrieval-augmented image captioning** is a recently popular approach that augments the captioning model with retrieved information for better open-world understanding. AoANet [16] uses a memory bank of image-sentence pairs and target words. SmallCap [35] employs image-to-text retrieval to obtain sampled captions from a captions datastore. RA-CM3 [44] retrieves documents from an external memory of a mixture of text and image via a dense multimodal retriever. EXTRA [34] and Re-ViLM [43] exploit the similarity of the input image and vision candidates to retrieve captions. Different from the previous methods, our external memory contains visual–name pairs to avoid redundant information in the external captions/documents. In addition, we use an attentive fusion module to mitigate the effects of irrelevant retrieved object names on caption generation. 

## **3. Proposed EVCAP** 

## **3.1. Idea of** EVCAP 

We aim to build a retrieval-augmented LLMs-based image captioning model with a sufficiently small yet informative external memory. It involves two challenges: (1) constructing an expandable external memory and (2) building an effective LLMs-based model using retrieved object names. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0003-00.png)


**----- Start of picture text -----**<br>
ViT External Visual-Name Memory<br>… +<br>Q-Former<br>…<br>Key:  Image embeddings …<br>Pikachu Moose Moose … Helmet<br>Value:  Object names Pikachu Moose …  Helmet<br>Visual features Object name features “A moose is<br>ViT standing in<br>+ Customized Large Language  the water<br>Q-Former Q-Former Model with palm<br>trees in the<br>background.”<br>… …<br>Linear Layer<br>Image query tokens Object name query tokens<br>Object Name Retrieval Attentive Fusion Caption Generation<br>… …<br>**----- End of picture text -----**<br>


Figure 2. Schematic of our proposed EVCAP. It consists of an external visual–name memory with image embeddings and object names (upper), a frozen ViT and Q-Former equipped with _trainable_ image query tokens, an attentive fusion module developed by a customized frozen Q-Former and _trainable_ object name query tokens, and a frozen LLM with a _trainable_ linear layer (lower). The ViT and Q- Former extract learned visual features from the input image, which are then used to retrieve object names from the external memory. These retrieved object names and learned visual features undergo cross-attention in the customized Q-Former, creating refined object name features. Finally, the object name features combined with visual features are fed into the LLM post a linear layer for generating captions. 

As discussed above, challenge (1) can be resolved by utilizing the visual appearance of objects. However, if we restrict our memory to only a visual–name pair for each object, our memory will be lacking in diversity. Therefore, we gather several images for each target object. Additionally, we keep the synthetic images in our memory to avoid the harm that synthetic images might cause to our method, as pointed out in [18]. With the capability to collect images from the internet, EVCAP can be easily expanded to include novel objects from the real world effortlessly. 

We base our method on the frozen pre-trained vision model and LLM with several trainable layers (Fig. 2), giving in a model that is cheap to train. To guide the LLM, we adopt a recently popular approach called prompting as in [11, 25, 29, 35, 46]. We begin by matching the learned visual features from the input image with image embeddings stored in memory, retrieving object names. We also introduce an attentive fusion module designed to implicitly remove irrelevant retrieved names. Finally, following the attentive fusion, we combine the learned visual features and object name features to form a prompt for the LLM to generate a caption, thus addressing challenge (2). 

## **3.2. External visual–name memory** 

To build the external visual–name memory, we first collect image–name pairs from the external data source. After 

that, we encode these images into image embeddings, which serve as keys in memory, and use their names as values. **External data source.** We utilize object images from LVIS dataset [17] to construct our external visual–name memory _M_ . Specifically, we use 1203 objects in LVIS, where we randomly select from one to ten images for each object, amounting to 8581 object images. Furthermore, as mentioned in Sec. 3.1, we also incorporate synthetic images in our memory construction. Using stable diffusion [36], we generate five additional images for each object, with a prompt of “ _a photo of {object name}_ ”, resulting in a total of _M_ = 14596 (8581 + 5 _×_ 1203) images. Each object image _X[i]_ is associated with an object name _v[i]_ . Note that many object images may share the same object name. For the sake of simplicity, we may regard each image as corresponding to a single name. In summary, we have _M_ image–name pairs _{_ ( _X[i] , v[i]_ ) _}[M] i_ =1[for external memory construction.] **External memory construction.** For each image _X[i]_ , we use a frozen vision encoder _E_ ( _·_ ) (see Sec. 3.3 for detail) to project it into 32 embeddings with the size of 1 _×_ 768 each: _{_ **k** _[i]_ 1 _[,]_ **[ k]** _[i]_ 2 _[,][ · · ·][,]_ **[ k]** _[i]_ 32 _[}]_[=] _[E]_[(] _[X][i]_[)][.][We][then][average][32] embeddings to produce a single embedding **k** _[i]_ (1 _×_ 768) that serves as the key (visual) in _M_ . The paired object name _v[i]_ acts as its value (name). Consequently, we have the visual–name memory _M_ = _{_ ( **k** _[i] , v[i]_ ) _}[M] i_ =1[which is indexed] using FAISS [21], facilitating rapid searches based on sim- 

ilarity measures. Our memory can be expanded effortlessly by gathering additional visual–name pairs (see Sec. 5.3). 

## **3.3. Object names retrieval** 

**Image encoding.** We feed a frozen vision encoder _E_ image _X_ and image query tokens **T** img to produce visual features _Q_ . To enable the retrieval process controllable, we make image query tokens to be trainable. Thus, the image encoding process can be summarized as _Q_ = _E_ ( _X,_ **T** img). We use the BLIP-2 pre-trained vision encoder [25], which consists of a pre-trained vision transformer ViT-g [14] outputting image features (257 _×_ 1408), and a Q-Former receiving image features producing _|Q|_ = 32 learned visual features (1 _×_ 768 each). We denote _Q_ = _{_ **q** 1 _,_ **q** 2 _, ...,_ **q** 32 _}_ . **Retrieval.** Having obtained _Q_ , we calculate the cosine similarity between the query **q** _j ∈Q_ and the key **k** _[i] ∈M_ . The 

similarity calculation is given by SIM( **q** _j,_ **k** _[i]_ ) = _∥_ **qq** _j[⊤]_ **j** _∥∥_ **[k] k** _[i][i] ∥_[,] where _i ∈_ [1 _, M_ ], _j ∈_ [1 _,_ 32]. Given each **q** _j_ , we select one key with the highest similarity score, resulting in 32 key–value candidates _{_ **k**[best] _j , vj_[best] _}_[32] _j_ =1[.] After that, we filter out candidates with repeated object names (values), and then select the top-K values. In particular, we determine the index _j_ from the key that has the highest SIM score. These selected values _vj_[best] are redefined as the new notation _vl_ in the retrieved top-K object names for the input image, which can be summarized as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0004-04.png)


where _l ∈_ [1 _,_ K]. As a result, the retrieved top-K object names are _{vl}_[K] _l_ =1[.] 

## **3.4. Attentive fusion** 

Since the object names obtained from the retrieval process may be redundant, we develop an attentive fusion module to selectively distill object name features. 

The retrieved object names _{vl}_[K] _l_ =1[are concatenated to-] gether into a sequence _S_ , each separated by a delimiter: _S_ = _{v_ 1 _,_ [SEP] _, v_ 2 _,_ [SEP] _, · · · ,_ [SEP] _, v_ K _}_ . The sequence _S_ and visual features _Q_ are fed into a customized Q-Former _F_ ( _·_ ), which is constructed from the frozen pretrained Q-Former as we used in vision encoder _E_ . Nonetheless, in order to enable object names to get attention from visual features, we switch the image embedding port and the text instruction port (see the supplement for architecture detail). Like in the image encoding process in Sec. 3.3, we make the object name query tokens **T** obj learnable during training to assist in learning object name features related to the caption. The size of **T** obj is _P ×_ 768, where _P_ indicates the number of object name query tokens. We get the object name features _V_ = _F_ ( _S, Q,_ **T** obj). 

## **3.5. Caption generation** 

Before inputting the visual features _Q_ and object name features _V_ into the LLM decoder, we concatenate ( _⊕_ ) them and use a linear layer _ϕ_ ( _·_ ) to project them into the input latent space of the LLM as _ϕ_ ( _Q ⊕V_ ). The LLM used for caption generation in this work is the pre-trained Vicuna-13B [10], an open-source chatbot constructed from LLaMA [37]. During training and evaluation, we design a prompt in a conversational format, that is similar to [46]: 

###Human: <Img><ProjFeature></Img> Describe this image in detail. ###Assistant: 

in which, ProjFeature denotes the projected feature _ϕ_ ( _Q ⊕V_ ) after the linear layer. In training phase, given input caption tokens _{ci}[L] i_ =1[, the LLM decoder concatenates] the embedded prompt _{_ **w** _i}[N] i_ =1[and][the][embedded][caption] tokens _{_ **c** _i}[L] i_ =1[as input,][and predicts the caption tokens in] an autoregressive fashion, while in the evaluation phase, we only need to input the embedded prompt. We train EVCAP by minimizing the cross-entropy loss in an end-to-end way: _Lθ_ = _−_[�] _[L] i_ =1[log] _[ p][θ]_[ (] _[c][i][|]_ **[ w]**[1] _[, ...]_ **[w]** _[N][,]_ **[ c]**[1] _[, ...,]_ **[ c]** _[i][−]_[1][)][, in] which _θ_ indicates the trainable parameters. 

## **4. Experimental Settings** 

## **4.1. Training setup** 

**Implementation.** EVCAP uses the same image encoder as in BLIP-2 [25], consisting of a ViT-g [14] and their pretrained Q-Former. Since we intend to obtain object name features through cross-attention between retrieved object names and visual features, we develop a customized Q- Former, which consists of BERT [23] with cross-attention layers inserted at every other transformer block. We use a frozen Vicuna-13B [10] as the caption generator. 

**Training dataset.** For all experiments, we exclusively train EVCAP using the training set of **COCO** dataset [27], consisting of 82k images and 5 captions per images. The entire training process takes about 3 hours on 4 A6000 GPUs, using mixed precisions (more details in the supplement). 

## **4.2. Evaluation setup** 

**Evaluation dataset.** We evaluate EVCAP, trained using the COCO training set, across four datasets: its test set, two challenging benchmarks – NoCaps validation set and Flickr30k test set, and a synthetic commonsense-violating dataset – WHOOPS. We adhere follow prior work [15, 41] to use the same images of Karpathy split [22] on **COCO** test set, **NoCaps** [2] validation set, and Karpathy split on **Flickr30k** [30] test set. In addition, **WHOOPS** [6] is a synthetic image captioning dataset comprising 500 synthetic commonsense-violating images and 2500 paired captions. 

Table 1. Quantitative comparison against SOTA methods on three common image captioning benchmarks. * denotes using a **memory bank** . We report the size of training data and parameters; BLEU@4 (B@4), METEOR (M), CIDEr (C), and SPICE (S) scores on COCO test set; C and S scores on in-domain, near-domain, out-domain and overall data of NoCaps validation set; C and S scores on Flickr30k test set. Higher score is better. **Bold** indicates the best results among compared methods, normal indicates the second best results. 

|**Method**|**Training**<br>Data<br>Para.|**COCO**<br>Test<br>B@4<br>M<br>C<br>S|**NoCaps val**<br>In-domain<br>Near-domain<br>Out-domain<br>Overall<br>C<br>S<br>C<br>S<br>C<br>S<br>C<br>S|**Flickr30k**<br>Test<br>C<br>S|
|---|---|---|---|---|
|**Heavyweight-training models**|||||
|VinVL [45]<br>AoANet+MA* [16]<br>NOC-REK* [40]<br>RCA-NOC* [13]<br>ViECap GPT2 [15]<br>InstructBLIP Vicuna-13B[11]<br>OSCAR [26]<br>BLIP [24]<br>BLIP-2 FlanT5-XL[25]<br>REVEAL* T5[20]|8.9M<br>110M|38.2<br>30.3<br>129.3<br>23.6<br>38.0<br>28.7<br>121.0<br>21.8<br>–<br>–<br>–<br>–<br>37.4<br>29.6<br>128.4<br>23.1<br>27.2<br>24.8<br>92.9<br>18.2|96.8<br>13.5<br>90.7<br>13.1<br>87.4<br>11.6<br>90.9<br>12.8<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>104.7<br>14.8<br>100.2<br>14.1<br>100.7<br>13.0<br>100.9<br>14.0<br>92.2<br>12.9<br>87.8<br>12.6<br>87.5<br>11.5<br>88.3<br>12.4<br>61.1<br>10.4<br>64.3<br>9.9<br>65.0<br>8.6<br>66.2<br>9.5|–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>47.9<br>13.6<br>82.8<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|
||COCO<br>–<br>COCO<br>110M<br>COCO<br>110M<br>COCO<br>124M||||
||129M<br>188M|–<br>–<br>–<br>–|–<br>–<br>–<br>–<br>–<br>–<br>121.9<br>–||
||4.1M<br>338M|37.4<br>30.7<br>127.8<br>23.5|83.4<br>12.0<br>81.6<br>12.0<br>77.6<br>10.6<br>81.1<br>11.7<br>114.9<br>15.2<br>112.1<br>14.9<br>115.3<br>14.4<br>113.2<br>14.8<br>**123.7**<br>**16.3**<br>**120.2**<br>**15.9**<br>**124.8**<br>**15.1**<br>121.6<br>**15.8**<br>–<br>–<br>–<br>–<br>–<br>–<br>**123.0**<br>–||
||129M<br>446M|40.4<br>–<br>136.7<br>–|||
||129M<br>1.2B|**42.4**<br>–<br>144.5<br>–|||
||1.3B<br>2.1B|–<br>–<br>**145.4**<br>–|||
|**Lightweight-training models**|||||
|MiniGPT4 Vicuna-13B [46]<br>SmallCap* GPT2[35]<br>ClipCap GPT2[29]|5M<br>**3.94M**|38.0<br>29.6<br>129.6<br>23.4|99.0<br>14.8<br>106.9<br>15.3<br>110.8<br>14.9<br>108.8<br>15.1|78.4<br>16.9<br>60.6<br>–<br>–<br>–|
||COCO<br>7M<br>COCO<br>43M|37.0<br>27.9<br>119.7<br>21.3<br>33.5<br>27.5<br>113.1<br>21.1|–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>84.9<br>12.1<br>66.8<br>10.9<br>49.1<br>9.6<br>65.8<br>10.9||
|EVCAP* Vicuna-13B|COCO<br>3.97M|41.5<br>**31.2**<br>140.1<br>**24.7**|111.7<br>15.3<br>119.5<br>15.6<br>116.5<br>14.7<br>119.3<br>15.3|**84.4**<br>**18.0**|
|**Specialist SOTAs**|||||
|Qwen-VL Qwen-7B [5]<br>CogVLM Vicuna-7B [41]<br>PaLI mT5-XXL[9]<br>PaLI-X UL2-32B [8]|1.4B<br>9.6B<br>1.5B<br>6.5B<br>1.6B<br>17B<br>2.2B<br>55B|–<br>–<br>–<br>–<br>–<br>–<br>148.7<br>–<br>–<br>–<br>149.1<br>–<br>–<br>–<br>149.2<br>–|–<br>–<br>–<br>–<br>–<br>–<br>121.4<br>–<br>–<br>–<br>–<br>–<br>132.6<br>–<br>128.3<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>127.0<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>126.3<br>–|85.8<br>–<br>94.9<br>–<br>–<br>–<br>–<br>–|



**Compared methods.** We compare EVCAP with several SOTAs. According to the trainable parameters size, they can be divided into 1) Heavyweight-training (between 100M to 5B): VinVL [45], AoANet [16], NOC-REK [40], RCA-NOC [13], ViECap [15], InstructBLIP [11], OSCAR [26], BLIP [24], BLIP-2 [25], REVEAL [20]; 2) Lightweight-training (less than 100M): MiniGPT4 [46], SmallCap [35], ClipCap [29]; and also 3) Specialist SOTAs with huge trainable parameters (larger than 5B): QwenVL [5], CogVLM [41], PaLI [9], PaLI-X [8]. Among these methods, AoANet, NOC-REK, RCA-NOC, REVEAL, and SmallCap are retrieval-augmented captioning methods. 

## **5. Experimental Results** 

## **5.1. Results on in-/out-domain benchmarks** 

We assess EVCAP against SOTAs on both in-domain and out-domain benchmarks. The COCO test set can be considered as in-domain data as we only train our model on the COCO training set. Out-domain benchmarks are the NoCaps validation set and the Flickr30k test set. 

**Quantitative results.** Tab. 1 details our EVCAP’s performance in comparison with SOTA methods. We first evaluate training costs in terms of training data sizes and parameters. Similar to various heavyweight-training models that exclude LLMs and the majority of lightweight-training models, EVCAP is trained solely on the COCO training set. It utilizes only 3.97M trainable parameters, positioning it as the second smallest, slightly larger than MiniGPT4 

with 3.94M. Among lightweight-training models, our approach outperforms others, achieving the highest scores on all benchmarks. Despite using less training data and nearly identical trainable parameters as MiniGPT4, EVCAP significantly surpasses it, with a marked improvement of 10.5, 10.5, and 6.0 in CIDEr scores for each benchmark. When further compared with heavyweight-training models, the performance of EVCAP stands out among million-level models, nearly matching InstructBLIP, except in NoCaps. Note that since BLIP-2 does not include Vicuna checkpoints, InstructBLIP performs pre-training with Vicuna using the same procedure as BLIP-2, whereas EVCAP does not involve pre-training. Against REVEAL, which also uses external memory, our EVCAP utilizes about 1/3000 training data and 1/500 training parameters yet yields comparable results. Moreover, EVCAP’s performance is on par with BLIP-2, the top-performing model with 1.2B trainable parameters. This highlights EVCAP’s efficiency and effectiveness despite its significantly smaller training cost, thanks to our external visual–name memory. Regarding specialist SOTAs, they use billion-level training data and over 5B trainable parameters, so it is acceptable that they can achieve exceptionally strong performance, surpassing EVCAP by nearly 10 on all benchmarks in CIDEr scores. 

**Qualitative results.** Fig. 3 presents a comparison of captions generated by our EVCAP and three SOTA models across three benchmarks. The captions of SmallCap are generated by its publicly accessible demo [1]. We generate captions of MiniGPT4 and BLIP-2 using their respec- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0006-00.png)


**----- Start of picture text -----**<br>
COCO Test NoCaps Val Flickr30k Test<br>GT:  A green bus driving through a rural GT:  The two guinea pigs are getting  GT:  A very young child in a denim<br>area with trees in the background. dried off in a yellow towel. baseball cap eats a green apple.<br>SmallCap:  A bus driving down a street SmallCap:  A person holding a small SmallCap:  A young boy holding an<br>next to trees. animal in a towel. apple in his hand.<br>MiniGPT4:  A green bus is driving  MiniGPT4:  Two small animals are  MiniGPT4:  A baby sitting in a high<br>down the street. wrapped in a towel. chair eating an apple.<br>BLIP-2:  A green bus driving down a  BLIP-2:  Two guinea pigs wrapped in a  BLIP-2:  A baby sitting in a white chair<br>road with trees in the background. yellow towel. eating a green apple.<br>EVCap:  A green bus driving down a  EVCap:  Two guinea pigs are wrapped  EVCap:  A toddler eating a green apple<br>road next to trees. in a yellow towel. while wearing a hat.<br>GT:  A woman in a blue top with  GT:  A computer screen showing two GT:  Two men are riding on a wooden<br>headphones and two cellphones. men sitting at a table. vehicle pulled by two donkeys.<br>SmallCap:  A woman sitting in front of  SmallCap:  Two men sitting at a table SmallCap:  A donkey pulling a cart with<br>a laptop computer. with a laptop. a man in the background.<br>MiniGPT4:  A woman sitting on a  MiniGPT4:  A laptop computer sitting on  MiniGPT4:  Two men riding on a<br>couch holding two phones. top of a table. donkey in the dirt.<br>BLIP-2:  A woman sitting on a couch  BLIP-2:  A laptop computer with a  BLIP-2:  Two men riding a horse drawn<br>with two cell phones.  picture of two men on it.  cart through a field.<br>EVCap:  A woman wearing  EVCap:  A laptop computer with a  EVCap:  Two men riding in a cart<br>headphones holding two cell phones. picture of two men on the screen. pulled by two donkeys.<br>**----- End of picture text -----**<br>


Figure 3. Examples of captions generated by our EVCAP and three SOTA methods on COCO test set, NoCaps validation set, and Flickr30k test set. GT refers to the Ground Truth captions. Incorrect objects in captions are highlighted in red , while correct ones are in blue . Our EVCAP correctly generates captions across different datasets, showing performance comparable to BLIP-2. 

tive pre-trained models. As a lightweight and retrievalaugmented captioning method, SmallCap struggles to produce accurate captions for given images, primarily because it relies on retrieved captions laden with extraneous information. MiniGPT4, though aligned with the primary content of images, sometimes misses certain objects like _“trees”_ and _“headphones”_ . This oversight stems from its focus on the main objects in images, without integrating additional cues for other objects provided by the retrieved object names. In contrast, the captions generated by our EVCAP are comparable to those of BLIP-2. 

## **5.2. Results on commonsense-violating data** 

To explore our EVCAP’s capability in describing contents in open-word settings, we further evaluate it on WHOOPS dataset, which contains commonsense-violating images. **Quantitative results.** In Tab. 2, we compare the performance of EVCAP, MiniGPT4, BLIP, and BLIP-2 on 

**GT:** This is an image of a blue Pikachu with yellow accents. **SmallCap:** A blue and white stuffed animal on a table. **MiniGPT4:** A blue pokemon sitting on the floor with its eyes closed. **BLIP-2:** A blue Pikachu sitting in a dark room. **EVCap:** A blue and yellow cartoon character sitting on a dark background. **EVCap (w/ WHOOPS):** A blue Pikachu is sitting on the floor. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0006-06.png)


Figure 4. Examples of captions generated by our EVCAP, EVCAP (w/ WHOOPS), and three SOTAs on WHOOPS dataset. Incorrect objects are highlighted in red , while correct ones are in blue . 

WHOOPS dataset. This dataset is particularly challenging due to its inclusion of unusual objects [6]. Initially, as an end-to-end trained model, our EVCAP exhibits performance similar to MiniGPT4. However, there is a noticeable improvement in the CIDEr score, after the external memory is enriched with 2396 new objects from the WHOOPS dataset, each represented by 5 synthesized images generated using stable diffusion [36]. It highlights the effectiveness of 

|Table 2. Quantitative results on commonsense-violating|Table 2. Quantitative results on commonsense-violating|Table 2. Quantitative results on commonsense-violating|Table 2. Quantitative results on commonsense-violating|data –||
|---|---|---|---|---|---|
|WHOOPS dataset. EVCAP(w/ WHOOPS) denotes|||EVCAPusing|||
|the memory expanded by WHOOPS|objects.|The results reveal the||||
|open-world comprehension ability and expandability of EVCAP.||||||
|**Method**|B@4|M|C|S||
|**Only pre-trained models**||||||
|BLIP [24] (from [6])|13|–|65|–||
|BLIP-2 FlanT5-XXL[25] (from [6])|31|–|120|–||
|BLIP-2 FlanT5-XXL[25] (reproduced)|28|26.7|93.1|17.9|42|
|**Finetuned models on COCO**||||||
|MiniGPT4 [46]|24.2|26.7|84.8|18.2||
|BLIP [24]|22.9|25.0|79.3|17.1||
|BLIP-2 FlanT5-XL[25]|25.8|27.0|89.1|18.3|Figure 6. Visualization of the visual features in external memory|
|**End-to-end trained models on COCO**|||||using t-SNE. For visual features in LVIS dataset’s objects (**blue**),|
|EVCAP|24.1|26.1|85.3|17.7|the related objects fall in the same cluster. After adding more vi-|
|EVCAP(w/ WHOOPS)|24.4|26.1|86.3|17.8|sual features of synthesized images from WHOOPS’ objects, new|



Table 2. Quantitative results on commonsense-violating data – WHOOPS dataset. EVCAP (w/ WHOOPS) denotes EVCAP using the memory expanded by WHOOPS objects. The results reveal the open-world comprehension ability and expandability of EVCAP. 

Figure 6. Visualization of the visual features in external memory using t-SNE. For visual features in LVIS dataset’s objects ( **blue** ), the related objects fall in the same cluster. After adding more visual features of synthesized images from WHOOPS’ objects, new objects ( **red** ) are located at appropriate clusters (zoom-in view). 

Table 3. Ablation study on components prior to the LLM decoder in EVCAP. The result of “+ Attentive fusion” demonstrates the substantial impact of the external visual–name memory. 

The quantitative results are shown in Tab. 3. When employing only the baseline model (Baseline), CIDEr scores drop notably by 5.7, 10.5, and 7.6 on COCO, NoCaps, and Flickr30k, respectively. The inclusion of trainable image query tokens (Baseline+) brings a marginal improvement on NoCaps and Flickr30k. However, the performance is significantly enhanced with the addition of attentive fusion (along with the introduction of external memory), indicating the pivotal role of the external visual–name memory in the overall effectiveness of EVCAP. This is further corroborated by the qualitative results in Fig. 5, where captions from Baseline and Baseline+ inaccurately include objects like _“couch”_ and _“bed”_ , and Baseline+ overlooks _“hand”_ . 

|**Method**|**COCO test**<br>C<br>S|**NoCaps val**<br>C<br>S|**Flickr30k test**<br>C<br>S|
|---|---|---|---|
|ViT + Q-Former (Baseline)<br>+ Image query tokens (Baseline+)<br>+ Attentive fusion (EVCAP)|134.4<br>23.9<br>134.1<br>23.8<br>140.1<br>24.7|108.8<br>14.2<br>109.0<br>14.3<br>119.3<br>15.3|76.8<br>17.3<br>77.3<br>17.2<br>84.4<br>18.0|




![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0007-06.png)


**GT:** A hamster on a blanket with a hand behind it. **Baseline:** A hamster sitting on a couch with a person's hand near it. **Baseline+:** A close up of a small hamster on a bed. **EVCap:** A small hamster sitting on a blanket next to a hand. (pillow, hamster, grizzly, gameboard, kitten, spectacles, cat, blanket, giraffe, pug-dog) 

Figure 5. Visualization of the captions generated from ablation study on the NoCaps validation set. We also show the retrieved object names by EVCAP, presented in gray. Incorrect objects in captions are highlighted in red , while correct ones are in blue . 

**Exploration for external memory expandability.** To demonstrate the scalability of the external memory in EVCAP, we visualize the visual features stored in LVIS external memory, and newly synthesized data from objects appearing in the WHOOPS dataset. We employ t-SNE [38] to plot visual features after reducing their dimensions to 2-D (Fig. 6). For clear visualization, we only randomly display 3649 visual features in LVIS memory, and add 479 visual features from WHOOPS objects. Among them, 35 samples are randomly labeled. The result shows a clear clustering of LVIS objects (blue) in the external memory, as well as the successful integration and appropriate localization of new objects from WHOOPS (red) into these clusters. This pattern not only confirms the distinctiveness of visual features already present in the memory but also demonstrates the potential to accurately incorporate and differentiate new objects introduced from updated data. These findings highlight our external memory’s ability to expand and maintain its effectiveness even as new data is incorporated. 

our idea of incorporating an expandable external memory into the captioning model for open-world comprehension. **Qualitative results.** Fig. 4 illustrates the captions generated by EVCAP, EVCAP (w/ WHOOPS), and three SOTAs for one image from the WHOOPS dataset. Similar to other methods except for BLIP-2, EVCAP cannot recognize _“blue cartoon character”_ as _“Pikachu”_ , while EVCAP (w/WHOOPS) successfully predicts it because of the updated memory. SmallCap and MiniGPT4 tend to generate captions with hallucinatory objects, a result of commonsense-violating contents present in the images. 

## **5.3. Detailed analysis** 

**Ablation study.** We assess the contribution of each component prior to the LLM decoder in EVCAP by incrementally integrating the image query tokens and the attentive fusion module into our baseline model. The baseline model comprises a ViT+Q-Former, a linear layer, and a LLM decoder. 

**Impact of external memory size.** We examine the impact of external memory size in Tab. 4. On the one hand, we randomly remove 30%, 60%, and 90% data in the external 

Table 4. Impact of the external memory size on the performance of EVCAP by evaluation under CIDEr scores. Changes in the size of external memory result in changes in performance. 

|**Method**|**Method**|**NoCaps val**<br>In<br>Near<br>Out<br>Overall|**Flickr30k**<br>Test|
|---|---|---|---|
|LVIS objects (EVCAP)<br>– 30% LVIS<br>– 60% LVIS<br>– 90% LVIS<br>+ WHOOPS||111.7<br>119.5<br>116.5<br>119.3<br>112.0<br>119.2<br>115.3<br>118.8<br>111.4<br>119.1<br>116.2<br>119.0<br>110.6<br>118.2<br>115.8<br>118.3<br>110.7<br>118.9<br>116.7<br>119.0|84.4<br>85.0<br>85.1<br>83.6<br>84.9|
|||||
|135.6<br>140.4 139.5<br>139.3<br>111.2<br>116.4<br>117.9<br>118.7<br>80.3<br>84.785.2<br>84.9<br>80<br>100<br>120<br>140<br>0<br>2<br>4<br>6<br>Num<br>COCO||NoCaps<br>Flickr3|0k|
||140.4 139.5<br>139.3|140.1<br>140.1<br>139.6|140.3|
||135.6<br>116.4<br>117.9<br>118.7|119.3<br>116.2<br>116.8|118.7|
||111.2|||
||80.3<br>84.785.2<br>84.9|84.4<br>85.1<br>84.7|84.8|
|||8<br>10<br>12<br>14<br>16<br>ber of retrieved object names (K)|18<br>20|



Figure 7. CIDEr scores after training EVCAP with the number of retrieved object names K from 0 to 20. The results indicate that the performance is relatively optimal when K is set to be 10. 

memory constructed from LVIS objects. The results show the performance gradually degrades on NoCaps as reducing 30% and 90% LVIS. Despite some unexpected increases in certain results on NoCaps (5th row) and Flickr30k (4th - 5th rows), they do not alter the overall downward trend. Similar phenomena are also noted in SmallCap [35], we speculate it is due to data distribution. On the other hand, as we infuse WHOOPS knowledge into LVIS memory, there is a slight improvement on NoCaps (out) and Flickr30k. These observations validate the model’s capability to effectively retrieve object names from an updated memory, enhancing its performance in generating captions. 

**Impact of the number of retrieved object names.** We investigate how the number of retrieved object names K (Sec. 3.3) affect EVCAP in Fig. 7. We train the model with K from 0 to 20 and evaluate the performance under CIDEr on all three benchmarks. From the results, we can find that the model works worst on the out-domain dataset (NoCaps) with zero object names. It confirms that the performance boost from Baseline+ to EVCAP (Tab. 3) is primarily attributed to the retrieval-augmented mechanism, but not the customized Q-Former itself. With more object names, performance fluctuates but improves. Furthermore, we observe that setting K to 10 yields relatively optimal overall performance, validating the choice of K = 10 in EVCAP. 

**Analysis with different decoders.** To explore the influence of different LLMs decoders on our EVCAP, we experiment by substituting Vicuna-13B with GPT2 and Vicuna-7B, as detailed in Tab. 5. With GPT2 as the decoder, EVCAP still markedly surpasses other GPT2-based models, achieving 

Table 5. Analysis with different LLM decoders including GPT2, Vicuna-7B, and Vicuna-13B. The results reveal EVCAP is effective when applying it in different LLM decoders. 

|**Method**<br>**LLM**|**COCO test**<br>C<br>S|**NoCaps val**<br>C<br>S|**Flickr30k test**<br>C<br>S|
|---|---|---|---|
|SmallCap [35]<br>GPT2<br>ViECap [15]<br>GPT2<br>EVCAP<br>GPT2|119.7<br>21.3<br>92.9<br>18.2<br>131.0<br>23.2|–<br>–<br>66.2<br>9.5<br>97.6<br>13.3|60.6<br>–<br>47.9<br>13.6<br>70.6<br>16.1|
|MiniGPT4 [46]<br>Vicuna-7B<br>InstructBLIP [11]<br>Vicuna-7B<br>EVCAP<br>Vicuna-7B|119.4<br>23.5<br>–<br>–<br>139.0<br>24.7|108.7<br>15.7<br>123.1<br>–<br>116.8<br>15.3|73.9<br>17.2<br>82.4<br>–<br>82.7<br>18.0|
|MiniGPT4 [46]<br>Vicuna-13B<br>InstructBLIP [11]<br>Vicuna-13B<br>EVCAP<br>Vicuna-13B|129.6<br>23.4<br>–<br>–<br>140.1<br>24.7|108.8<br>15.1<br>121.9<br>–<br>119.3<br>15.3|78.4<br>16.9<br>82.8<br>–<br>84.4<br>18.0|



impressive gains of 11.3 and 10.0 under CIDEr on COCO and Flickr30k, compared to SmallCap. When employing Vicuna-7B, the comparison of performance trends mirrors those observed with Vicuna-13B, further attesting to the robustness and adaptability of EVCAP across different LLM decoders. Notably, both SmallCap, which retrieves captions, and our GPT2-based EVCAP, which retrieves object names, use the same GPT2 decoder. Therefore, their comparison also underscores the effectiveness of our method’s object name retrieval and attentive fusion strategy. **Limitations.** 

First, EVCAP cannot retrieve all objects that appear in the given image due to the memory coverage limits, leading to incomplete image descriptions (Fig. 4). We will investigate integrating object detection with image captioning to enhance completeness. Second, our focus on object representation restricts consideration of other crucial captioning elements, affecting overall performance. Similar to all models trained with COCO dataset, EVCAP has limitations in generating varied styles, which is reflected in our relatively modest performance improvements in Tab. 2, compared to MiniGPT4. We will overcome it by exploring methodologies that encourage style diversity in the future. 

## **6. Conclusion** 

We further advance image captioning in real-world scenarios by introducing EVCAP, a novel image captioning model with object names retrieved from an external visual– name memory. The external memory is easily expandable, allowing for effortless updates with new object visuals and names. We extensively compare EVCAP with SOTAs on various benchmarks and commonsense-violating data, demonstrating its significant superiority in performance. **Acknowledgements.** This work was supported by JSPS/MEXT KAKENHI Grant Numbers JP22H05015, JP23H03449, JP23KJ0404, and JP22K17947, and the commissioned research (No. 225) by the National Institute of Information and Communications Technology (NICT), Japan, and the Institute of AI and Beyond of the University of Tokyo, and ROIS NII Open Collaborative Research 2023-23FC01, 2024-24S1201. 

## **References** 

- [1] https : / / huggingface . co / spaces / RitaParadaRamos/SmallCapDemo. 5 

- [2] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In _Proc. IEEE International Conference on Computer Vision (ICCV)_ , 2019. 4 

- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: A visual language model for few-shot learning. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2022. 1 

- [4] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2018. 2 

- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint arXiv:2308.12966_ , 2023. 5 

- [6] Nitzan Bitton-Guetta, Yonatan Bitton, Jack Hessel, Ludwig Schmidt, Yuval Elovici, Gabriel Stanovsky, and Roy Schwartz. Breaking common sense: Whoops! a vision-andlanguage benchmark of synthetic and compositional images. In _Proc. IEEE International Conference on Computer Vision (ICCV)_ , 2023. 4, 6, 7 

- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2020. 2 

- [8] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. PaLI-X: On scaling up a multilingual vision and language model. _arXiv preprint arXiv:2305.18565_ , 2023. 1, 2, 5 

- [9] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish V Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme Ruiz, Andreas Peter Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. PaLI: A jointly-scaled multilingual languageimage model. In _International Conference on Learning Representations (ICLR)_ , 2023. 1, 2, 5 

- [10] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. _See https://vicuna. lmsys. org (accessed 14 April 2023)_ , 2023. 2, 4 

- [11] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale 

Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2023. 3, 5, 8 

- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations (ICLR)_ , 2021. 2 

- [13] Jiashuo Fan, Yaoyuan Liang, Leyao Liu, Shaolun Huang, and Lei Zhang. Rca-noc: Relative contrastive alignment for novel object captioning. In _Proc. IEEE International Conference on Computer Vision (ICCV)_ , 2023. 2, 5 

- [14] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 2, 4 

- [15] Junjie Fei, Teng Wang, Jinrui Zhang, Zhenyu He, Chengjie Wang, and Feng Zheng. Transferable decoding with visual entities for zero-shot image captioning. In _Proc. IEEE International Conference on Computer Vision (ICCV)_ , 2023. 2, 4, 5, 8 

- [16] Zhengcong Fei. Memory-augmented image captioning. In _Proc. AAAI Conference on Artificial Intelligence (AAAI)_ , 2021. 2, 5 

- [17] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for large vocabulary instance segmentation. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2019. 3 

- [18] Ryuichiro Hataya, Han Bao, and Hiromi Arai. Will largescale generative models corrupt future datasets? In _Proc. IEEE International Conference on Computer Vision (ICCV)_ , 2023. 3 

- [19] Lisa Anne Hendricks, Subhashini Venugopalan, Marcus Rohrbach, Raymond Mooney, Kate Saenko, and Trevor Darrell. Deep compositional captioning: Describing novel object categories without paired training data. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2016. 2 

- [20] Ziniu Hu, Ahmet Iscen, Chen Sun, Zirui Wang, Kai-Wei Chang, Yizhou Sun, Cordelia Schmid, David A Ross, and Alireza Fathi. Reveal: Retrieval-augmented visual-language pre-training with multi-source multimodal knowledge memory. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 1, 2, 5 

- [21] Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billionscale similarity search with gpus. _IEEE Transactions on Big Data_ , 7(3):535–547, 2019. 3 

- [22] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2015. 2, 4 

- [23] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of_ 

_North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACLHLT)_ , 2019. 4 

- [24] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _Proc. International conference on machine learning (ICML)_ , 2022. 5, 7 

- [25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _Proc. International conference on machine learning (ICML)_ , 2023. 1, 2, 3, 4, 5, 7 

- [26] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In _Proc. European Conference on Computer Vision (ECCV)_ , 2020. 2, 5 

- [27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Proc. European Conference on Computer Vision (ECCV)_ , 2014. 4 

- [28] Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh. Neural baby talk. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2018. 2 

- [29] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. _arXiv preprint arXiv:2111.09734_ , 2021. 2, 3, 5 

- [30] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In _Proc. IEEE International Conference on Computer Vision (ICCV)_ , 2015. 4 

- [31] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 2 

- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _Proc. International conference on machine learning (ICML)_ , 2021. 2 

- [33] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of Machine Learning Research (JMLR)_ , 21(1):5485–5551, 2020. 2 

- [34] Rita Ramos, Desmond Elliott, and Bruno Martins. Retrievalaugmented image captioning. In _Proc. Conference of the European Chapter of the Association for Computational Linguistics (EACL)_ , 2023. 2 

- [35] Rita Ramos, Bruno Martins, Desmond Elliott, and Yova Kementchedjhieva. Smallcap: Lightweight image captioning prompted with retrieval augmentation. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 1, 2, 3, 5, 8 

- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 3, 6 

- [37] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 2, 4 

- [38] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. _Journal of machine learning research (JMLR)_ , 9(11), 2008. 7 

- [39] Subhashini Venugopalan, Lisa Anne Hendricks, Marcus Rohrbach, Raymond Mooney, Trevor Darrell, and Kate Saenko. Captioning images with diverse objects. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2017. 2 

- [40] Duc Minh Vo, Hong Chen, Akihiro Sugimoto, and Hideki Nakayama. Noc-rek: Novel object captioning with retrieved vocabulary from external knowledge. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 2, 5 

- [41] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models. _arXiv preprint arXiv:2311.03079_ , 2023. 4, 5 

- [42] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In _Proc. International conference on machine learning (ICML)_ . 2 

- [43] Zhuolin Yang, Wei Ping, Zihan Liu, Vijay Korthikanti, Weili Nie, De-An Huang, Linxi Fan, Zhiding Yu, Shiyi Lan, Bo Li, et al. Re-ViLM: Retrieval-augmented visual language model for zero and few-shot image captioning. In _Findings of the Association for Computational Linguistics: EMNLP_ , 2023. 2 

- [44] Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Richard James, Jure Leskovec, Percy Liang, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Retrieval-augmented multimodal language modeling. In _International Conference on Learning Representations (ICLR)_ , 2023. 2 

- [45] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 2, 5 

- [46] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 3, 4, 5, 7, 8 

## **Supplementary Material for EVCAP: Retrieval-Augmented Image Captioning with External Visual–Name Memory for Open-World Comprehension** 

This supplementary material complements our paper with the following sections: First, we delve into the implementation specifics of our EVCAP, which were not covered in the main paper (see Sec. A). Second, we offer an expanded discussion on the external visual-name memory, as utilized in the main paper (see Sec. B). Finally, we present additional results to evaluate the effectiveness of EVCAP (see Sec. C). 

Table A. Quantitative results under the CIDEr score on in-domain (In), near-domain (Near), out-domain (Out), and overall data of the NoCaps test set. * denotes using a **memory bank** . We note that our results on the test set are publicly submitted to Nocaps leader-board _[a]_ (8th rank). Higher score is better. **Bold** indicates the best results among compared methods, normal indicates the second best results. 

## **A. Implementation Details** 

**Customized Q-Former.** Fig. A depicts customized Q-Former, where the image embedding port (CLIP visual feature port) now receives retrieved object names _S_ , while the text port receives visual features _Q_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0011-05.png)


**----- Start of picture text -----**<br>
Customized Q-Former Q-Former  [25]<br>Feed Forward Feed Forward Feed Forward Feed Forward<br>names (Object S) Cross Attention embeddingsImage  Cross Attention<br>Self Attention Self Attention<br>Object name query tokens (T!"#) Visual features (Q) Query tokens Text<br>**----- End of picture text -----**<br>


Figure A. Customized Q-Former and original Q-Former [25]. 

**Implementation.** Our method is based on Pytorch and is trained within one epoch with a batch size of 24 using mixed precisions. We optimize the model using AdamW, setting the weight decay at 0.05, and using _β_ 1 and _β_ 2 values of 0.9 and 0.99, respectively. A cosine learning rate (LR) decay strategy is adopted, starting with an initial LR of 1e-4. The model undergoes 5000 linear warm-up steps, beginning with a start LR of 1e-6. During the evaluation phase, we use a beam search strategy with a beam size of 5 to generate captions. 

## **B. External Visual–Name Memory** 

## **B.1. LVIS memory** 

As stated in Sec. 3.2 of the main paper, we utilize 1203 objects from the LVIS dataset. For each of these objects, we randomly select between one and ten images from LVIS. Additionally, we enrich our data by incorporating five synthetic images for each object, created using stable diffusion. We show two samples of this external visual-name memory, constructed using objects from LVIS in Fig. B. 

## **B.2. WHOOPS memory** 

To illustrate the scalability of the external memory in EVCAP, we expand it by integrating WHOOPS knowledge into the original external visual–name memory in Sec. 5.2 and Sec. 5.3 of the main paper. Specifically, we focus on objects that are mentioned in the answers of VQA annotations in the WHOOPS dataset because of their conciseness and emphasis on key objects. For each 

|**Method**|In|Near|Out|Overall|
|---|---|---|---|---|
|**Heavyweight-training models**|||||
|VinVL [45]|93.8|89.0|66.1|85.5|
|NOC-REK* [40]|100.0|95.7|77.4|93.0|
|OSCAR [26]|81.3|79.6|73.6|78.8|
|**Lightweight-training models**|||||
|SmallCap* GPT2[35]|87.9|84.6|84.4|85.0|
|EVCAP* Vicuna-13B|**114.9**|**117.0**|**117.1**|**116.8**|
|**Specialist SOTAs**|||||
|CogVLM Vicuna-7B [5]|–|–|128.0|126.4|
|PaLI mT5-XXL[9]|–|–|–|124.4|
|PaLI-X UL2-32B [8]|–|–|–|124.3|



> _a_ https : / / eval . ai / web / challenges / challenge - page/355/leaderboard/1011 

of these objects, we produce five synthetic images employing stable diffusion. Two examples from this augmented memory, featuring newly added object images and their corresponding names, are presented in Fig. C. 

## **C. Additional Results** 

## **C.1. Experiments on NoCaps test set** 

We additionally assess our EVCAP against SOTAs on the NoCaps test set, since we notice several other methods have also benchmarked their performance on this dataset. Note that, NoCaps test set does not have publicly accessible ground truth annotations. To obtain evaluation scores, we submitted our results to the NoCaps leaderboard. 

**Quantitative results.** Tab. A presents the quantitative results of our EVCAP on the NoCaps test set. Our method outperforms all other SOTA models, both in heavyweight-training and lightweight-training categories, that have reported results on this dataset. Additionally, as a lightweight method, our approach achieves the 8th rank on the NoCaps leaderboard, only surpassed by specialized SOTAs such as CogVLM, which holds the 1st rank. **Qualitative results.** Fig. E shows the captions generated by our EVCAP alongside those from three SOTA methods on the NoCaps test set. It also includes the object names retrieved by EVCAP and the captions retrieved by SmallCap. Consistent with the 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0012-00.png)


**----- Start of picture text -----**<br>
Image<br>Name Real images from LVIS Synthesized images<br>Polar bear<br>Muffin<br>**----- End of picture text -----**<br>


Figure B. Samples of images and corresponding names in the external visual–name memory constructed from LVIS’s objects. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0012-02.png)


**----- Start of picture text -----**<br>
Image<br>Name Synthesized images<br>Dinosaur<br>Spiderman<br>**----- End of picture text -----**<br>


Figure C. Samples of images and corresponding names in the added WHOOPS’s objects. 

findings in the main paper, SmallCap tends to generate hallucinatory objects that are absent in the input images, such as _“tie”_ and _“mouse”_ . The same hallucinatory object _“mouse”_ is also found in the retrieved captions, indicating that SmallCap’s diminished performance is largely due to its reliance on retrieved captions containing irrelevant information. In comparison, our EVCAP demonstrates a performance on par with BLIP-2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0012-05.png)


**----- Start of picture text -----**<br>
COCO NoCaps Flickr30k<br>150<br>137.7 139.2 140.1<br>133.6 135<br>130<br>119.1 119.3<br>115.8 116.8<br>110 107<br>87.3<br>90 84.6 82.2 84.4<br>75.3<br>70<br>2 4 6 8 10<br>Number of object name query tokens (P)<br>**----- End of picture text -----**<br>


Figure D. CIDEr scores after training EVCAP with the number of object name query tokens _P_ from 2 to 10. The results indicate that the performance is relatively optimal when _P_ is set to be 8. 

Table B. Comparison against SOTA methods on training time and used GPUs. * denotes using a **memory bank** . 

|**Method**|**Training time**<br>**GPUs**|
|---|---|
|**Heavyweight-training models**||
|VinVL [45]<br>AoANet+MA* [16]<br>NOC-REK* [40]<br>RCA-NOC*[13]<br>ViECap GPT2[15]<br>InstructBLIP Vicuna-13B[11]<br>OSCAR [26]<br>BLIP [24]<br>BLIP-2 FlanT5-XL[25]<br>REVEAL* T5[20]|–<br>–<br>–<br>–<br>8d<br>2 RTX3090<br>1d<br>8 A100<br>–<br>–<br>1.5d<br>16 A100<br>74h<br>1 V100<br>–<br>2 16-GPU nodes<br>_∼_9d<br>16 A100<br>5d<br>256 CloudTPUv4 chips|
|**Lightweight-training models**||
|MiniGPT4 Vicuna-13B [46]<br>SmallCap* GPT2[35]<br>ClipCap GPT2[29]|10h<br>4 A100<br>8h<br>1 A100<br>6h<br>1 GTX1080|
|EVCAP* Vicuna-13B|3h<br>4 A6000|
|**Specialist SOTAs**||
|Qwen-VL Qwen-7B [5]<br>CogVLM Vicuna-7B [41]<br>PaLI mT5-XXL[9]<br>PaLI-X UL2-32B [8]|–<br>–<br>1d<br>4096 A100<br>–<br>–<br>–<br>–|



SOTA models. Due to the diversity of GPUs employed across different models, drawing a direct comparison is challenging. Nevertheless, it’s evident that the training time for our EVCAP is comparatively shorter than most models. 

**Number of object name query tokens.** We explore the impact of varying the number of retrieved object names _P_ (Sec. 3.4 of the main paper) on EVCAP in Fig. D. We train the model using different values of _P_ , ranging from 2 to 10, and evaluate the performance under CIDEr on all three benchmarks. The results suggest that setting _P_ = 8 offers relatively optimal results. 

## **C.3. More qualitative examples.** 

More qualitative examples on the COCO test set, NoCaps validation set, and Flickr30k test set are shown in Fig. F, Fig. G, and Fig. H, respectively. 

## **C.2. Further analysis** 

**Comparison on training time and used GPUs.** Tab. B compares training time and the used GPUs of our EVCAP with various 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0013-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0013-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0013-02.png)


**SmallCap:** A group of snowboarders standing in the snow. **SmallCap’ retrieved captions: MiniGPT4:** A group of people standing in the snow wearing ski gear and _Two men with snowboarding gear standing on slope next to a_ smiling. _mountain._ **BLIP-2:** A group of people standing in the snow on a ski lift. _A couple of snowboarders that are in the snow._ **EVCap:** Two skiers are posing for a picture in the snow. _A pair of men standing on snowboards giving the thumbs up. (glove, ski pole, jacket, snowboard, helmet, ski, map, goggles, ski boot, A snowboarder in a green jacket and a skier. backpack)_ **SmallCap:** A man in a suit and tie speaking at a podium. **SmallCap’ retrieved captions: MiniGPT4:** A man standing at a podium in front of a crowd. _A large man speaking in front of a crowd._ **BLIP-2:** A man giving a speech at a podium in front of a crowd. _A man at a podium speaking at an event._ **EVCap:** A man standing behind a podium giving a speech. _A person speaking something with a mike in his hands. (dress, identity card, television set, ring, spectacles, handkerchief, cymbal, A man is speaking while a picture is taken of him. tablecloth, fire alarm, plastic bag)_ **SmallCap:** A computer mouse sitting next to a keyboard. **SmallCap’ retrieved captions: MiniGPT4:** A red joystick sitting on top of a keyboard. _An orange and white electric gadget sits on a red surface._ **BLIP-2:** A computer mouse on a desk next to a keyboard. _A white joystick held by a child for a video game._ **EVCap:** A red joystick sitting next to a computer keyboard. _A video game control is held by grips. (mouse computer equipment, coaster, place mat, joystick, router computer An electronic gadget connected to a keyboard and wireless mouse. equipment, combination lock, money, battery, notebook, cat)_ 

Figure E. Examples of captions generated by our EVCAP, and three SOTAs on the NoCaps test set. We also list the retrieved object names by our EVCAP (below the captions of EVCAP) and retrieved captions by SmallCap (right side) in _italics_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0013-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0013-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0013-07.png)


**GT:** Some sheep that are grazing in a snowy field. **SmallCap:** A herd of sheep standing in the snow. **MiniGPT4:** A herd of sheep grazing in a snow covered field. **BLIP-2:** A herd of sheep grazing in a field near a house. **EVCap:** A herd of sheep grazing in a snow covered field. _(lamb animal, goat, deer, bull, ram animal, calf, hammock, cow, bear, giraffe)_ 

## **SmallCap’ retrieved captions:** 

_A flock of sheep are gathered at the corner of a cabin in a field. A group of sheep standing in a snowy yard outside a barn. Sheep graze in a field next to two large houses. A farm setting with sheep grazing in the yard area._ 

**GT:** Two rectangular slices of pizza with multiple toppings. **SmallCap:** Two slices of pizza sitting on top of a white plate. **MiniGPT4:** Two slices of pizza on a metal tray with cheese and vegetables. **BLIP-2:** Two slices of pizza on a white plate. **EVCap:** Two slices of pizza topped with cheese and vegetables. _(mushroom, salami, crouton, sausage, quesadilla, bell pepper, lettuce, squid food, crescent roll, avocado)_ 

## **SmallCap’ retrieved captions:** 

_A small slice of gourmet flat bread pizza. Flat bread pizza slices piled on top of each other. A little personal sized pizza cut into squares. A collection of differently topped pizza slices on a plate._ 

**GT:** There are crafts and craft supplies on a table. **SmallCap’ retrieved captions: SmallCap:** A number of craft items sitting on a table. _Two cellphones have cute homemade cellphone covers._ **MiniGPT4:** A pair of scissors and some yarn on a table. _Several crocheted items with yarn scissors and crochet hooks._ **BLIP-2:** A blue pair of scissors sits on a table next to other items. _A series of crafting tools are laid out mostly for sewing._ **EVCap:** A pair of scissors sitting next to a crocheted item. _A number of craft items sitting on a table. (mushroom, salami, crouton, sausage, quesadilla, bell pepper, lettuce, squid food, crescent roll, avocado)_ 

Figure F. Examples of captions generated by our EVCAP, and three SOTAs on the COCO test set. We also list the retrieved object names by our EVCAP (below the captions of EVCAP) and retrieved captions by SmallCap (right side) in _italics_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0014-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0014-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0014-02.png)


**GT:** A shark is swimming in dark blue water near other fish. **SmallCap:** A couple of fish that are in the water. **MiniGPT4:** A large shark swimming in the ocean with other fish. **BLIP-2:** A shark and a turtle swimming in the water. **EVCap:** A shark and a fish swim in the water. _(polar bear, manatee, rhinoceros, cow, grizzly, crossbar, foal, shark, ostrich, crow)_ 

## **SmallCap’ retrieved captions:** 

_Two large animals standing in a pool of water. A person looks into an aquarium at a large animal swimming. This is a large fish that is in an aquarium. Some animals that are in the water together._ 

**GT:** A sink, a tub with towels, soap and other bath tub accessories. **SmallCap’ retrieved captions: SmallCap:** A bathroom with a sink and a toilet. _An image of a bathroom setting with double faucet._ **MiniGPT4:** A bathroom with a sink and a shower in it. _A very nice looking clean cut and contemporary styled wash basin._ **BLIP-2:** A white bathroom with a sink and a shower. _A modern bathroom with a toilet and pedestal sink._ **EVCap:** A bathroom with a sink, shower and towels. _A washroom area with a sink soap and glasses. (sink, urinal, shower head, hose, washbasin, towel rack, radiator, cover, hair dryer, shower curtain)_ **GT:** A person holding a ipod and a iphone in both hand. **SmallCap’ retrieved captions: SmallCap:** A person holding a cell phone in their hand. _An apple computer ipod and other electronic devices._ **MiniGPT4:** A person holding an ipod in their hand. _A variety of apple ipod products on display._ **BLIP-2:** A person holding an ipod next to a cell phone. _Multiple ipods on a desk surrounded by computers._ **EVCap:** A person holding an iPhone next to an iPod. _Man at computer holding ipod and interfacing the two devices. (ipod, cellular telephone, sword, olive oil, boxing glove, toaster, stapler stapling machine, rat)_ 

_An image of a bathroom setting with double faucet. A very nice looking clean cut and contemporary styled wash basin. A modern bathroom with a toilet and pedestal sink. A washroom area with a sink soap and glasses._ 

Figure G. Examples of captions generated by our EVCAP, and three SOTAs on the NoCaps validation set. We also list the retrieved object names by our EVCAP (below the captions of EVCAP) and retrieved captions by SmallCap (right side) in _italics_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0014-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0014-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension_images/cvpr_2024_EVCap_Retrieval-Augmented_Image_Captioning_with_External_Visual-Name_Memory_for_Open-World_Comprehension.pdf-0014-11.png)


**GT:** A dog with a red collar runs in a forest in the middle of winter. **SmallCap:** A dog running through the woods on a leash. **MiniGPT4:** A dog running through a field with trees in the background. **BLIP-2:** A black and brown dog running on a trail. **EVCap:** A dog running through a field of dry grass. _(cub animal, pole, cow, giraffe, minivan, calf, bulldog, grizzly, shepherd dog, short pants)_ 

## **SmallCap’ retrieved captions:** 

_A black dog with collar running in dirt area. A dog running through a forest area on a trail. A dog standing in a trail with woods in the background. A black and brown dog with its tongue out._ 

**GT:** A young red-haired man is fishing and caught some seaweed. **SmallCap’ retrieved captions: SmallCap:** A man holding a fishing pole on top of a body of water. _A man fishing while standing on some rocks next to the ocean._ **MiniGPT4:** A man is fishing in the ocean with a kite. _A man fishes from some rocks with a cargo ship in the distance._ **BLIP-2:** A man sitting on a rock holding a fishing rod. _A man with a rainbow umbrella fishing off a rock coast._ **EVCap:** A man sitting on a rock holding a fishing pole. _Fisherman with a pole holding needle nose pliers. (water scooter, mast, pelican, tartan, bobbin, jersey, boat, kite, gull, dumpster)_ **GT:** Two girls playing in a game of softball. **SmallCap’ retrieved captions: SmallCap:** A woman swinging a baseball bat at a ball. _A runner is touching base while another player is waiting to catch the_ **MiniGPT4:** Two women playing baseball on a dirt field. _ball._ **BLIP-2:** A softball player sliding into a base during a game. _A runner sliding onto base while another player catches the ball._ **EVCap:** Two girls playing a game of softball on a field. _A female baseball player unleashes a hit and goes for the run. (softball, home plate baseball, belt, clipboard, baseball, knee pad, baseball A girls softball game with a play at home base as an umpire watches base, thermos bottle, baseball bat, chair) to make the call._ 

Figure H. Examples of captions generated by our EVCAP, and three SOTAs on the Flickr30k test set. We also list the retrieved object names by our EVCAP (below the captions of EVCAP) and retrieved captions by SmallCap (right side) in _italics_ . 

