# **CROSS**[2] **STRA: Unpaired Cross-lingual Image Captioning with Cross-lingual Cross-modal Structure-pivoted Alignment** 

**Shengqiong Wu, Hao Fei** _[∗]_ **, Wei Ji, Tat-Seng Chua** Sea-NExT Joint Lab, School of Computing, National University of Singapore swu@u.nus.edu {haofei37, jiwei, dcscts}@nus.edu.sg, 

## **Abstract** 

Unpaired cross-lingual image captioning has long suffered from irrelevancy and disfluency issues, due to the inconsistencies of the semantic scene and syntax attributes during transfer. In this work, we propose to address the above problems by incorporating the scene graph (SG) structures and the syntactic constituency (SC) trees. Our captioner contains the _semantic structure-guided image-to-pivot captioning_ and the _syntactic structure-guided pivot-to-target translation_ , two of which are joined via pivot language. We then take the SG and SC structures as pivoting, performing cross-modal semantic structure alignment and cross-lingual syntactic structure alignment learning. We further introduce cross-lingual&cross-modal back-translation training to fully align the captioning and translation stages. Experiments on English _↔_ Chinese transfers show that our model shows great superiority in improving captioning relevancy and fluency. 

## **1 Introduction** 

Generating texts to describe images (a.k.a., image captioning) has many real-world applications, such as virtual assistants and image indexing (Fang et al., 2015). Current image captioning models have achieved impressive performance (Jia et al., 2015; Gu et al., 2018a; Ji et al., 2021), yet are mainly limited to the English language due to the largescale paired image-caption datasets. Subject to the scarcity of paired captioning data, the development of captioning in other languages is thus greatly hindered. While manually crafting sufficient paired data is prohibitively expensive, cross-lingual image captioning (Miyazaki and Shimizu, 2016) offers a promising solution, which aims to transfer a captioner trained at resource-rich language (e.g., English) to the resource-scarce language(s) without paired captioning data at target language(s). 

> _∗_ Corresponding author: Hao Fei 

A direct approach is to make use of the current translation techniques, i.e., the pivot language translation method. Here pivot language is the resource-rich language, e.g., English. For example, the pivot-side captioner first generates pivot captions for images, which are then translated into the target-side captions. Or one can create the pseudo image-caption pairs for directly training a targetside captioner, by translating the pivot training captions into the target ones (Lan et al., 2017). However, the above translation-based method suffers from two major issues (cf. 1(a)), including _irrelevancy_ and _disfluency_ (Song et al., 2019). On the one hand, due to the lack of paired vision contexts, a translated description can easily deviate from the original visual semantics, leading to ambiguous or inaccurate captioning. On the other hand, restricted to the translation system itself, translated texts often suffer from disfluent language, especially for the lengthy and complex descriptions. 

Some previous efforts are carried out to rectify the above two key errors for better cross-lingual captioning. Lan et al. (2017) solve the translation disfluency issue by estimating the fluency of translation texts, then rejecting those disfluent ones. Yet their method dramatically sacrifices the paired training data, and meanwhile suffers from lowefficiency owing to the incremental screening process. Song et al. (2019) propose to enhance the relevance and fluency of translations by designing some rewards via the reinforcement learning technique. However, the _REINFORCE_ algorithm (Williams, 1992) is hard to train, and easily leads to unstable results. We note that there are two critical abilities a cross-lingual captioning system should possess to solve the corresponding problems. For content relevancy, the kernel lies in sufficiently modeling the vision-language semantic alignment; while for language fluency, it is key to effectively capture the gaps of linguistic attributes and characteristics between the pivot and target languages. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Input Image Image-to-PivotCaptioning Pivot caption Pivot-to-TargetTranslating Target caption Image Scene Inconsistency Target caption  (       ) -1<br>(     )           A boy in a white shirt is speaking to a  餐桌上正在公园前和一群人讲话。 (       ) a park is speaking to a group of people.)(A boy in a white shirt sitting at table in  -1 一个穿着白衬衫的男孩坐在 Relevance error peoplespeak boy in whiteshirt peoplespeak boysit in whiteshirt<br>group of people sitting  sit table<br>at table in front of a  (       ) -2 一个穿着白衬衫的男孩正在跟<br>park. 一群人讲话，他们坐在公园前桌子上。 Fluency  table in front of park in front of park<br>(A boy in a white shirt is speaking to a group of  error<br>people, they are setting at table in front of a park.) Visual Scene Graph Language Scene Graph<br>(a) Two major problems in existing cross-lingual image captioning (b) Relevancy issue due to inconsistency of semantic scene<br>Syntax Inconsistency<br>Pivot caption (     ) Target caption (       ) -2<br>S S<br>NP VP NP VP IP<br>NP PP is speaking PP QP CP NP ADVP PP VP NP VP<br>A boy in a white shirt to a group of NP 一个 穿着 白 寸衫 的 男孩 正在 跟 一群 人 讲话 他们 坐在 NP<br>people VP 公园前 桌子 上<br>PP PP<br>sitting at table in front of a park<br>Syntactic Constituency Tree Syntactic Constituency Tree<br>(c) Inconsistency of syntax structures between pivot-target languages causes disfluent translation<br>**----- End of picture text -----**<br>


Figure 1: We illustrate two issues in translation-based cross-lingual captioning (a), caused by the inconsistencies of semantic scene structures (b) and syntax structures (c). We simplify the constituency trees for better understanding, and the dotted grey box areas indicate the counterpart constituents across different languages. (Best viewed in color) 

Besides the translation-based methods, the pivoting-based cross-lingual captioning methods have shown effectiveness, where the whole task learning is broken down into two steps, imageto-pivot captioning and pivot-to-target translation (Gu et al., 2018b; Gao et al., 2022). The imageto-pivot captioning learns to describe images in the pivot language based on pivot-side paired captioning data, and the pivot-to-target translation is performed based on parallel sentences. Two crossmodel and cross-lingual subtasks are trained on two separate datasets, and aligned by the pivot language. Although achieving improved task performances, existing pivoting-based methods (Gu et al., 2018b; Gao et al., 2022) still fail to fully address the two major problems of cross-lingual captioning, due to the insufficient alignment of either vision-language semantics or pivot-target syntax. 

To this end, we present a novel syntactic and semantic structure-guided model for cross-lingual image captioning. We build the framework based on the pivoting-based scheme, as shown in Fig. 2. For image-to-pivot captioning, we consider leveraging the scene graphs (SG) for better image-text alignment. Intuitively, an SG (Johnson et al., 2015; Yang et al., 2019) depicts the intrinsic semantic structures of texts or images, which can ideally bridge the gaps between modalities. For the pivotto-target translating, we make use of the syntactic constituency (SC) tree structures for better pivottarget language alignment. Syntax features have been shown as effective supervisions for enhancing the translation quality, e.g., fluency and grammarcorrectness (Schwartz et al., 2011; Xu et al., 2020; 

## Li et al., 2021). 

Based on the above framework, we further perform cross-lingual cross-modal structure-pivoted alignment learning. First of all, we introduce an SG-pivoted cross-modal semantic structure alignment. Based on contrastive learning (Logeswaran and Lee, 2018; Yan et al., 2021) we realize the unsupervised vision-language semantic structure alignment, relieving the scene inconsistency and thus enhancing the relevancy. Similarly, an unsupervised SC-based cross-lingual syntax structure aligning is used to learn the shared grammar transformation and thus mitigate the language disfluency during translation. Finally, we perform the cross-lingual cross-modal back-translation training, fully aligning the two phrases of image-to-pivot captioning and pivot-to-target translation. 

On English _→_ Chinese and Chinese _→_ English transfers of unpaired cross-lingual image captioning, our method achieves significant improvement over the existing best-performing methods. Further in-depth analyses demonstrate that the integration of both scene graph and syntactic structure features is complementarily helpful in improving the captioning relevancy and disfluency of the transfer. 

Our main contributions are two-fold: 

_•_ First, we for the first time enhance the crosslingual image captioning by leveraging both the semantic scene graph and the syntactic constituent structure information, such that we effectively address the problems of content irrelevancy and language disfluency. 

_•_ Second, we propose several cross-lingual crossmodal structure-pivoted alignment learning strate- 

gies, via which we achieve effective cross-modal vision-language semantic alignment and crosslingual pivot-target syntactic alignment. 

## **2 Related Work** 

Image captioning has been an emerging task in the past few years and received great research attention (You et al., 2016; Vinyals et al., 2017; Cornia et al., 2020). Later, the task of cross-lingual image captioning (Miyazaki and Shimizu, 2016; Song et al., 2019) has been presented, to transfer the knowledge from resource-rich language to resource-poor language[1] , so as to spare the burden of manual data annotation for the minority languages. However, the task has been hindered and received limited attention due to two key issues: irrelevancy and disfluency of captions. There are two categories of cross-lingual captioning approaches: the translation-based (Lan et al., 2017; Gu et al., 2018b) and the pivoting-based (Gu et al., 2018b; Gao et al., 2022) methods. The former employs an off-the-shelf translator to translate the source (pivot) captions into the target language for targetside training or as the target-side captions. The latter reduces the noise introduction of the pipeline by jointly performing the image-to-pivot captioning step and pivot-to-target translation step, thus being the current SoTA paradigm. This work inherits the success of this line, and adopts the pivoting-based scheme as a backbone, but we further strengthen it by leveraging the semantic and syntactic structure information to better solve the two issues. 

Scene graphs depict the intrinsic semantic scene structures of images or texts (Krishna et al., 2017; Wang et al., 2018). In SGs, the key object and attribute nodes are connected to describe the semantic contexts, which have been shown useful as auxiliary features for wide ranges of downstream applications, e.g., image retrieval (Johnson et al., 2015), image generation (Johnson et al., 2018) and image captioning (Yang et al., 2019). Here we incorporate both the visual and language scene graphs to enhance the cross-modal alignment learning. 

Note that Gao et al. (2022) also leverage the SG features for cross-lingual captioning, while ours differs from theirs in three aspects. First, they consider a fully unsupervised cross-lingual setup with no image-caption pairs at pivot language, while under such an unpaired assumption the visual and 

> 1Without using target-side image-caption pairs, the task is also known as unpaired cross-lingual image captioning. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0003-06.png)


**----- Start of picture text -----**<br>
Target caption<br>Residual<br>Connection<br>Syntactic Structure-guided<br>Translating<br>Syntactic Constituency<br>Structure Encoder<br>Pivot caption<br>Semantic Structure-guided<br>Captioning<br>Scene Graph<br>Structure Encoder<br>Image<br>**----- End of picture text -----**<br>


Figure 2: A high-level schematic of the cross-lingual captioning model architecture. 

language scene graphs are hard to align, and thus limits the utility of SGs. Second, in this work we sufficiently align the two cross-modal SGs via unsupervised learning, such that the noises in SGs will be effectively screened. Third, Gao et al. (2022) align the pivot and target languages with also the SG structure. We note that it could be ineffective to perform cross-lingual alignment based on textual SGs because the scene structures in different languages are essentially the same. 

In fact, two languages can be different the most in linguistic structures. Almost all the erroneous sentences come with certain grammar or syntax errors (Jamshid Lou et al., 2019, 2020). Also syntax features have been extensively found to be effective in improving the language quality (e.g., fluency and grammatically-correctness) in cross-lingual scenario (Nivre, 2015; Li et al., 2021; Zhang and Li, 2022). For example, in machine translation, different languages show great correspondences in phrasal constituent structures (Zhang and Zong, 2013; Fang and Feng, 2022). Also, syntactic structure features have been integrated into a broad number of downstream applications (Wu et al., 2021; Fei et al., 2021, 2022). Thus we consider making use of the syntax structures as cross-lingual supervision to enhance the captioning quality. 

## **3 Syntactic Semantic Structure-guided Cross-lingual Captioning Framework** 

The original task is to learn a mapping _FI→St_ from input images _I_ to target-language captions _S[t]_ . Following Gu et al. (2018b); Song et al. (2019), we decompose _FI→St_ into two mappings: 1) the 

image-to-pivot captioning _FI→Sp_ training with the paired data _{_ ( _I, S[p]_ ) _}_ , and 2) the pivot-to-target translation _FSp→St_ training with the parallel data _{_ ( _S[p] , S[t]_ ) _}_ . Note that _{_ ( _I, S[p]_ ) _}_ and _{_ ( _S[p] , S[t]_ ) _}_ are two distinct datasets with possibly no intersection. 

In our setting, we also leverage the SG and SC structure features in two mappings. As shown in Fig. 2, the semantic structure-guided captioning phase ( _F<I,_ SG _>→Sp_ ) takes as input the image _I_ and the visual SG encoded by a structure encoder, yielding the pivot caption _S[p]_ . Then, the syntactic structure-guided translating phase ( _F<Sp,_ SC _>→St_ ) takes as input the _S[p]_ and the pivot SC, finally producing the target caption _S[t]_ . Note that the input embeddings of the second step are shared with the output embeddings from the first step so as to avoid the isolation of the two parts. Also we impose a residual connection from the SG feature representations to the SC feature representations to supervise the final target captioning with scene features. 

## **3.1 Semantic Structure-guided Captioning** 

Given an image, we obtain its SG from an off-theshelf SG parser, which is detailed in the experiment setup. We denote an SG as _SG_ =( _V, E_ ), where _V_ is the set of nodes _vi ∈ V_ (including object, attribute and relation types),[2] _E_ is the set of edges _ei,j_ between any pair of nodes _vi_ . We encode a SG with a graph convolution network (GCN; Marcheggiani and Titov, 2017): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0004-04.png)


where _**h** i_ is the representation of a node _vi_ . We then use a Transformer (Vaswani et al., 2017) decoder to predict the pivot caption _S_[ˆ] _[p]_ based on _{_ _**h** i}_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0004-06.png)


## **3.2 Syntactic Structure-guided Translation** 

In this step we first transform the predicted pivot caption _S[p]_ into the SC structure, _SC_ =( _V, E_ ), where _V_ are the phrasal&word nodes connected by the compositional edge _E_ . Different from the dependency-like SG structure, SC is a tree-like hierarchical structure, as depicted in Fig. 1. Similarly, we encode SC trees with another GCN: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0004-09.png)


where _**r** j_ is an SC node representation. Another Transformer decoder is used to predict the target caption _S_[ˆ] _[t]_ . To ensure the relevancy of target-side generation, we create a shortcut between the prior SG feature representations _**h**_ and the SC features 

> 2Appendix _§_ A.1 details the SG and SC structures. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0004-12.png)


**----- Start of picture text -----**<br>
Shared Syntax Constituency Structure Encoder<br>Cross-lingual Syntactic<br>SC [P] Structure Alignment SC [T]<br>Target caption<br>Pivot caption<br>Image<br>Cross-modal Semantic<br>Structure Alignment<br>SG [L] SG [V]<br>Shared Scene Graph Structure Encoder<br>**----- End of picture text -----**<br>


Figure 3: Illustration of the cross-modal semantic and cross-lingual syntactic structure alignment learning. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0004-14.png)


## **3.3 Two Separate Supervised Learning** 

The captioning and the translation steps are performed separately based on _{_ ( _I, S[p]_ ) _}_ and _{_ ( _S[p] , S[t]_ ) _}_ in a supervised manner: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0004-17.png)


## **4 Structure-Pivoting Cross-lingual Cross-modal Alignment Learning** 

In the above supervised training, though leveraging the semantic and syntactic structure information, the cross-modal image-text pair and the cross-lingual pivot-target pair can be still underaligned in their own feature spaces, due to the intrinsic structural gaps, e.g., noisy substructures. To combat that, we further propose two structurepivoting unsupervised learning strategies (cf. Fig. 3): cross-modal semantic structure alignment and cross-lingual syntactic structure alignment. Besides, the two parts of our backbone captioner are initially trained separately. This motivates us to further align the two procedures in a whole-scale way, with cross-lingual&cross-modal back-translation training (cf. Fig. 4). 

## **4.1 Cross-modal Semantic Structure Aligning** 

The basic idea is to encourage those text nodes and visual nodes that serve a similar role in the visual SG _[V]_ and language SG _[L]_ to be closer, while for those not we hope to repel them from each other, so as to mitigate the scene inconsistency. We realize this via the current popular CL technique. We 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Gold pivot<br>caption<br>External translator<br>(Tgt-to-Pivot)<br>Predicted  Predicted<br>pivot caption target caption<br>Gold image<br>(a) Image-to-Pivot back-translation<br> given gold pairs of image-caption(pivot)<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0005-01.png)


**----- Start of picture text -----**<br>
Gold target<br>caption<br>Gold pivot  Predicted<br>caption target caption<br>Predicted image<br>(b) Pivot-to-Target back-translation<br>given gold pivot-target parallel sentences<br>**----- End of picture text -----**<br>


Figure 4: Illustrations of the cross-modal and cross-lingual back-translation. 

first obtain the node representations of visual SG ( _**h**[V] i_[) and language SG (] _**[h]**[L] j_[) using one shared GCN] encoder as in Eq. (1), based on the ground-truth _{_ ( _I, S[p]_ ) _}_ data. We then measure the similarities between all pairs of nodes from two SGs: 

_i_[)] _[T][·]_ _**[ h]**[L] j s[m] i,j_[=] _||_[(] _**[h] h**[V][V] i[|| ||]_ _**[h]**[L] j[||][.]_ (7) A pre-defined threshold _ρm_ will decide the alignment confidence, i.e., pairs with _s[m] i,j[> ρ][m]_[ is con-] sidered similar. Then we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0005-05.png)


where _τm_ >0 is an annealing factor. _j[∗]_ represents a positive pair with _i_ , i.e., _s[m] i,j[∗]_[>] _[ρ][m]_[.] _[Z]_[is a normal-] ization factor for probability calculation. 

## **4.2 Cross-lingual Syntactic Structure Aligning** 

The idea is similar to the above one, while in the cross-lingual syntactic structure space. We use the shared SC GCN encoder to generate node representations _**r** i[P]_[and] _**[ r]** j[T]_[of pivot-/target-side SCs on] the parallel sentences. CL loss is then put on the similarity score _s[l] i,j_[to carry out the unsupervised] alignment learning, which we summarize as _L_ CLA. 

## **4.3 Cross-modal&lingual Back-translation** 

Drawing inspiration from unsupervised machine translation, we leverage the back-translation technique (Sennrich et al., 2016; Edunov et al., 2018) to achieve the two-step alignment over the overall framework. We present the cross-lingual cross-modal back-translation training, including the image-to-pivot back-translation and the pivotto-target back-translation. 

**Image-to-Pivot Back-translation** With gold image-caption pairs at hand, we can first obtain the target caption prediction _S_[ˆ] _[t]_ via our cross-lingual 

captioner. We then translate the _S_[ˆ] _[t]_ into pseudo pivot caption _S_[ˆ] _[p]_ via an external translator _Mt→p_ . This thus forms a path: _S[p]_ - _I→S_[ˆ] _[t] →S_[ˆ] _[p]_ . And our framework can be updated via: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0005-13.png)


**Pivot-to-Target Back-translation** There is a similar story for the gold pivot-target parallel sentences: _S[t]_ - _S[p] →I_[ˆ] _→S_[ˆ] _[t]_ . For _S[p] →I_[ˆ] we leverage an external SG-based image generator (Johnson et al., 2018; Zhao et al., 2022). The learning loss is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0005-15.png)


> ⋆ **Remarks on Training** We take a warmstart strategy to ensure stable training of our framework. Initially we pre-train two parts separately via _L_ Cap& _L_ Trans We then perform two structure-pivoting unsupervised alignment learning via _L_ CMA& _L_ CLA. Finally, we train the overall model via back-translation _L_ IPB& _L_ PTB. Once the system tends to converge, we put them all together for further overall fine-tuning: 

_L_ = _L_ Cap + _L_ Trans + _L_ CMA + _L_ CLA + _L_ IPB + _L_ PTB _._ (11) Here for brevity, we omit the item weights. Appendix _§_ A.4 gives more training details. 

## **5 Experimental Setups** 

**Datasets** To align with existing work, we consider the transfer between English (En) and Chinese (Zh), and use the image caption datasets of MSCOCO (Lin et al., 2014), AIC-ICC (Wu et al., 2017) and COCO-CN (Li et al., 2019). We use the training set of a language as image-pivot pairs for the first part training, and test with the set of another language. For the second part training, we collect the paired En-Zh parallel sentences from existing MT data, including UM (Tian et al., 2014) and WMT19 (Barrault et al., 2019). 

||**Zh**_→_**En**<br>BLEU<br>METEOR<br>ROUGE<br>CIDEr|**En**_→_**Zh**<br>**_Avg._**<br>BLEU<br>METEOR<br>ROUGE<br>CIDEr|
|---|---|---|
|_•_**_Translation-based methods_**<br>EarlyTranslation<br>48.3<br>15.2<br>27.2<br>18.7<br>LateTranslation<br>45.8<br>13.8<br>25.7<br>14.5<br>FG<br>46.3<br>12.5<br>25.3<br>15.4<br>SSR_†_<br>52.0<br>14.2<br>27.7<br>28.2<br>_•_**_Pivoting-based methods_**<br>PivotAlign<br>52.1<br>17.5<br>28.3<br>27.0<br>UNISON<br>54.3<br>18.7<br>30.0<br>28.4||43.6<br>20.3<br>30.3<br>14.2<br>27.2<br>41.3<br>13.5<br>26.7<br>14.0<br>24.4<br>43.0<br>19.7<br>29.7<br>15.7<br>25.9<br>46.0<br>22.8<br>32.0<br>18.3<br>30.1<br>47.5<br>23.7<br>32.3<br>19.7<br>31.1<br>48.7<br>25.2<br>33.7<br>21.9<br>32.4|
|CROSS2STRA (Ours)<br>**57.7**<br>**21.7**<br>**33.5**<br>**30.7**||**52.8**<br>**27.6**<br>**36.1**<br>**24.5**<br>**35.8**|
|w/o SG<br>55.8<br>19.1<br>31.2<br>28.0<br>w/o SC<br>56.1<br>20.0<br>32.1<br>28.9<br>w/o ResiConn<br>56.4<br>21.2<br>32.9<br>29.4||48.6<br>25.8<br>33.9<br>21.6<br>33.1<br>50.4<br>26.6<br>35.4<br>23.3<br>34.1<br>51.8<br>27.1<br>35.9<br>24.1<br>34.9|



Table 1: Transfer results between MSCOCO (En) and AIC-ICC (Zh). The values of SSR _[†]_ are copied from Song et al. (2019), while all the rest are from our implementations. 

**Baselines and Evaluations** Our comparing systems include 1) the translation-based methods, including the _early translation_ and _late translation_ mentioned in the introduction, _FG_ (Lan et al., 2017), _SSR_ (Song et al., 2019), and 2) the pivotingbased methods, including _PivotAlign_ (Gu et al., 2018b) and _UNISON_ (Gao et al., 2022). Following baselines, we report the BLEU (Papineni et al., 2002), METEOR (Denkowski and Lavie, 2014), ROUGE (Lin, 2004) and CIDEr (Vedantam et al., 2015) scores for model evaluation. Our results are computed with a model averaging over 10 latest checkpoints. 

**Implementations** To obtain the visual SGs, we employ the FasterRCNN (Ren et al., 2015) as an object detector, and MOTIFS (Zellers et al., 2018) as a relation classifier and an attribute classifier. For language SGs, we first convert the sentences into dependency trees with a parser (Anderson et al., 2018), and then transform them into SGs based on certain rules (Schuster et al., 2015). We obtain the SC trees via the Berkeley Parser (Kitaev and Klein, 2018), trained on PTB (Marcus et al., 1993) for En texts and on CTB (Xue et al., 2005) for Zh texts. In our back-translation learning, we use the T5 (Raffel et al., 2020) as the target-to-pivot translator ( _Mt→p_ ), and adopt the current SoTA SG-based image generator ( _MSp→I_ ) (Zhao et al., 2022). Chinese sentences are segmented via Jieba[3] . We use Transformer to offer the underlying textual representations for GCN, and use FasterRCNN (Ren et al., 2015) for encoding visual feature representations. Our SG and SC GCNs and all other embeddings have the same dimension of 1,024. All 

3https://github.com/fxsjy/jieba 

||**Zh**_→_**En**<br>B<br>R|**En**_→_**Zh**<br>**_Avg._**<br>B<br>R|
|---|---|---|
|CROSS2STRA|**57.7**<br>**33.5**|**52.8**<br>**36.1**<br>**45.0**|
|w/o_L_CMA<br>w/o_L_CLA<br>w/o_L_IPB<br>w/o_L_PTB<br>w/o_L_CMA+_L_CLA<br>w/o_L_IPB+_L_PTB|54.4<br>29.7<br>54.6<br>30.1<br>53.8<br>31.1<br>55.0<br>32.8<br>51.8<br>27.7<br>52.7<br>30.1|50.1<br>34.9<br>42.3(-2.7)<br>51.0<br>35.3<br>43.0(-2.0)<br>50.5<br>35.1<br>43.1(-1.9)<br>52.2<br>35.7<br>44.2(-0.8)<br>47.5<br>33.7<br>40.8(-4.2)<br>49.9<br>34.2<br>42.2(-2.8)|



Table 2: Ablating different learning strategies. B: BLEU, R: ROUGE. 

models are trained and evaluated with NVIDIA A100 Tensor Core GPUs. 

## **6 Experimental Results and Analyses** 

**Transfer between MSCOCO and AIC-ICC** Table 1 presents the Zh _→_ En and En _→_ Zh transfer results. We first can observe that the _EarlyTranslation_ is more effective than _LateTranslation_ , as the former introduces lesser noises in training. Also, we see that among all the translation-based methods, _SSR_ shows the best performance. Further, it is clear that the pivoting methods show overall better results than the translation ones. This is most possibly because the joint training in pivotingbased models relieves the under-alignment between the captioning and translation stages, reducing the noise introduction of the pipeline. 

Looking into the pivoting-based models, _UNISON_ exhibits the stronger capability of the transfer in both directions, owing to the integration of SG structure features, i.e., helping accurately capture the semantic relevances between vision and language. Most importantly, our CROSS[2] STRA outperforms all the other baselines with significant 

||**BLEU@1**|**BLEU@2**|**BLEU@3**|**BLEU@4**|**METEOR**|**ROUGE**|**CIDEr**|**_Avg._**|
|---|---|---|---|---|---|---|---|---|
|_•_**_Translation-based methods_**|||||||||
|EarlyTranslation_†_|60.4|40.7|26.8|17.3|24.0|43.6|52.7|37.9|
|LateTranslation_†_|58.9|38.0|23.5|14.3|23.5|40.2|47.3|35.1|
|SSR|65.2|43.5|27.3|17.7|25.4|45.9|53.8|39.8|
|_•_**_Pivoting-based methods_**|||||||||
|PivotAlign|66.5|45.0|29.3|18.2|27.0|46.3|55.0|41.0|
|UNISON_∗†_|63.4|43.2|29.5|17.9|24.5|45.1|53.5|39.5|
|UNISON|68.3|46.7|30.6|19.0|29.4|48.0|56.3|42.7|
|CROSS2STRA|**70.4**|**48.8**|**32.5**|**20.8**|**31.9**|**50.6**|**58.2**|**44.7**|



Table 3: Cross-lingual transfer from MSCOCO (En) to COCO-CN (Zh). The values with _†_ are copied from Song et al. (2019). UNISON _[∗†]_ is the raw version without using the paired image-caption(pivot) data for training. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0007-02.png)


**----- Start of picture text -----**<br>
Zh → En En → Zh<br>60 60<br>55<br>55<br>50<br>50<br>0.1 0.3 0.5 0.7 0.9 0.1 0.3 0.5 0.7 0.9<br>Threshold ρm Threshold ρl<br>(BLEU)<br>Performance<br>**----- End of picture text -----**<br>


Figure 5: Influences of different threshold values in two structure-guided alignment learning. 

margins on all metrics consistently. For example, we improve over _UNISON_ by 3.4 (Zh _→_ En) and 4.1 (En _→_ Zh) BLEU scores respectively. We give credit to the integration of both the semantic SG and the syntactic SC structures, as well as the effective alignment learning strategies. The above observations show the efficacy of our system for cross-lingual captioning. 

**Influences of Learning Strategies** In Table 2 we quantify the contribution of each learning objective via ablation. As seen, each learning strategy shows the impact to different extents. For example, the cross-modal semantic alignment gives greater influences than the cross-lingual syntactic alignment of the overall performances (i.e., 2.7 vs. 2.0). In contrast to the two structure-pivoting learning ( _L_ CMA+ _L_ CLA), we can find that the back-translation learning ( _L_ IPB+ _L_ PTB) shows slightly lower impacts. Particularly the pivot-to-target back-translation contributes limitedly, and we believe the quality of SGto-image generator should bear the responsibility. 

**Threshold Study** In Fig. 5 we study the influences of threshold values on the two alignment learning, by varying _ρm_ and _ρl_ . As seen, when _ρm_ is 0.6 and 0.7 in two tasks respectively, the overall transfer results are the best, while _ρl_ =0.3 helps give the best effects. Such a pattern distinction 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0007-07.png)


**----- Start of picture text -----**<br>
SSR PivotAlign UNISON<br>CROSS [2] STRA w/o SG w/o SC<br>90<br>80<br>70<br>60<br>(a) Comparisons on scene graphs between<br>input vision and target caption<br>80<br>70<br>60<br>50<br>40<br>(b) Comparisons on top-down constituency<br>structure between pivot and target captions<br>(%)<br> structure<br>SG<br>coincidence rate<br>(%)<br>C structure<br>S<br>coincidence rate<br>**----- End of picture text -----**<br>


Figure 6: The matchings of SG and SC structures. 

between _ρm_ and _ρl_ implies that the SGs between vision and language have less discrepancy, while the SC structures between two languages come with non-negligible differences. 

**Transfer from MSCOCO to COCO-CN** Table 3 further shows the transfer results from English MSCOCO to Chinese COCO-CN. The overall tendency is quite similar to the one in Table 1. We see that translation methods are inferior to the pivoting methods. Our CROSS[2] STRA model gives the best performances on all metrics, outperforming _UNISON_ by an average 2.0(=44.7-42.7) score. This again verifies the efficacy of our proposed method. 

**Probing Cross-modal and Cross-lingual Structure Alignment** We integrate the semantic scene structure and syntactic structures with the aim of better cross-modal and cross-lingual alignment in our two-stage pivoting transfer framework. Here we directly assess to what extent our methods improve the alignment. Fig. 6 shows the structure 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0008-00.png)


**----- Start of picture text -----**<br>
绿油油的草地上蹲着一个穿着灰色寸衫面带 一名足球队员正在足球场上与另一名足球的运动员<br>Gold 微笑的小朋友 Gold 争夺足球<br>(A smiling child in grey shirt is squatting on the green grass) (A football player is competing for football with another football player on the football field)<br>一个人在踢一个足球和一个红色衣服的男人<br>SSR 草丛上有一个小女孩坐在地上 SSR (A man is playing a football and a man in red)<br>(Sitting on the grass there is a little girl sitting on the ground)<br>在绿色的球场上一个身着白色衣服与一个红色衣服<br>UNISON 坐在绿色草地上的小孩穿着灰色上衣(Sitting on the green field is a kid wearing a gray coat) UNISON 的男人踢白色的足球(On the green football field, a man in white and a man in red play a white football)<br>碧绿的草坪上蹲着一个满面笑容身穿灰色短 一位穿着白色球衣与另一外穿着红色球衣的运动员<br>CROSS [2] STRA 袖的小孩 CROSS [2] STRA 在绿色足球场上争夺足球<br>(A smiling kid in grey T-shirt is squatting on the green field) (A player wearing a white jersey and another player wearing a red<br>jersey are competing for football on the green football field.)<br>有一位拿着球拍的男运动员在球场上打网球 滑冰场上，一名穿着黑色裤子的男士和一名穿着裙<br>Gold (There is a male player with a racket is playing tennis on  Gold 子的女人一起进行花样滑冰<br>the court) (On the skating rink, a man in black trousers and a woman in skirt<br>are doing figure skating together)<br>SSR 一个男人拿着网球拍(A man is holding a tennis racket) SSR 穿着滑冰鞋身着演出服装的男人与女人在滑冰(Men and women in skates and costumes are skating)<br>穿着蓝色衣服的男人和穿裙子的女人在滑冰场上进<br>UNISON 一位男运动员挥舞着网球拍在网球场 UNISON 行花样滑冰<br>(A male athlete is waving a tennis racket on the tennis court) (The man in blue and the woman in skirt are figure skating on the<br>skating rink)<br>有一位身着白色衣裤的男性运动员拿着球拍 在滑冰场上有一名身着蓝色寸衫与黑色裤子的男士<br>CROSS [2] STRA 在蓝色球场上打网球(There is a male athlete in white clothes with a racket playing  CROSS [2] STRA 和一名穿着蓝色裙子的女士共同表演花样滑冰(On the skating rink, a man in a blue shirt and black pants and a<br>tennis on the blue court) woman in a blue skirt together perform figure skating)<br>**----- End of picture text -----**<br>


Figure 7: Qualitative results of cross-lingual captioning. The instances are randomly picked from AIC-ICC (Zh). 

||**Relevancy**_↑_**Diversifcation**_↑_**Fluency**_↑_|**Relevancy**_↑_**Diversifcation**_↑_**Fluency**_↑_|**Relevancy**_↑_**Diversifcation**_↑_**Fluency**_↑_|
|---|---|---|---|
|FG|5.34|3.75|7.05|
|SSR|7.86|5.89|7.58|
|PivotAlign|8.04|6.57|7.46|
|UNISON|9.02|9.14|7.89|
|CROSS2STRA|**9.70**_‡_|**9.53**_‡_|**9.22**_‡_|
|w/o SG|8.35|7.75|9.04|
|w/o SC|9.42|8.34|8.07|
|w/o_L_CMA+_L_CLA|7.80|7.24|8.15|



Table 4: Human evaluations are rated on a Likert 10scale. _‡_ indicates significant better over the baselines ( _p_ <0.03). 

coincidence rate between the input image SG and predicted target caption SG, and the SC structure coincidence rate between the pivot and target captions.[4] We see that with the integration of semantic scene modeling, both _UNISON_ and our system exhibit prominent cross-modal alignment ability, i.e., with higher structural overlaps. The same observation can be found with respect to syntactic structure integration for enhancing cross-lingual alignment learning. Either without the leverage of SG or SC structure, the corresponding cross-modal or crosslingual alignment effect is clearly weakened. 

**Human Evaluation** We further try to quantify the improvements of the generated captions via human evaluation. In Table 4 we show the evaluation results based on MSCOCO (En) to AIC-ICC (Zh) transfer, on three dimensions: _relevancy_ , _diversification_ and _fluency_ . We can see that our system shows significantly higher scores than baseline sys- 

> 4Appendix _§_ B.2 details the measuring method. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0008-07.png)


**----- Start of picture text -----**<br>
SSR UNISON CROSS [2] STRA w/o SC<br>40<br>30<br>20<br>10<br>Wording Word-order Syntax<br>(%)<br>rate<br>Error<br>**----- End of picture text -----**<br>


Figure 8: Evaluating the language quality of captions. 

tems in terms of all three indicators. For those methods with SG structure features, the content relevancy and diversification of captions are much better. Yet only our method gives satisfied language fluency, due to the equipment of syntactic features. With further ablation studies we can further confirm the contributions of the SG and SC features. 

**Captioning Linguistic Quality Study** We take a further step, investigating how exactly our model improves the linguistic quality of the target captions. Same to the human evaluation, we ask native speakers to measure the errors that occurred in the generated captions, in terms of _wording_ , _word order_ and _syntax correctness_ . Fig. 8 presents the results of the transfer from MSCOCO (En) to AICICC (Zh). We see that our model has committed the least errors, where the performances on syntax correctness are especially higher than baselines. Once without using the syntactic features, the error rates grow rapidly, which demonstrates the importance to integrate the syntactic structures. 

**Qualitative Result** Finally, we empirically show some real prediction cases, so as to aid an intuitive 

understanding of our method’s strength. In Fig. 7 we provide four pieces of testing examples on the En _→_ Zh transfer, which we compare with different baseline methods. As can be seen, the _SSR_ model often tends to generate target-side captions with lower diversification, and meanwhile unsatisfactory content relevancy, and thus inaccurate image descriptions. On the contrary, the captions from _UNISON_ are much better, i.e., better relevancy and diversification. We can give credit to the equipment of scene graph-based alignment learning. However, _UNISON_ can fall short on language quality, i.e., problematic fluency. Since English and Chinese differ much in linguistic and grammar characteristics, without leveraging the syntactic structure features, it leads to inferior language quality. Luckily, our model can address all those issues, and generate captions with good relevancy, diversification, and fluency. This again proves the effectiveness of our proposed method. 

## **7 Conclusion and Future Work** 

In this paper we investigate the incorporation of semantic scene graphs and syntactic constituency structure information for cross-lingual image captioning. The framework includes two phrases, semantic structure-guided image-to-pivot captioning and syntactic structure-guided pivot-to-target translating. We take the SG and SC structures as pivots, performing cross-modal semantic structure alignment and cross-lingual syntactic structure alignment learning. A cross-lingual&cross-modal backtranslation training is further performed to align two phrases. On English _↔_ Chinese transfer experiments, our model shows great superiority in terms of captioning relevancy and fluency. 

Bridging the gaps between the cross-modal and cross-lingual transfer with external semantic and syntactic structures has shown great potential. Thus it is promising to extend the idea to other scenarios. Also, exploiting the external structures potentially will introduce noises, and thus a dynamical structure induction is favorable. 

## **Limitations** 

In this work, we take the sufficient advantages of the external semantic and syntactic structure knowledge to improve our focused problem. But this could be a double-edged sword to use such features. Specifically, our paper has the following two potential limitations. First of all, our method closely relies on the availability of the resources 

of scene graph structures and syntax structures. While most of the languages come with these structure annotations to train good-performing structure parsers (for example, the syntax structure annotations of Penn TreeBank cover most of the existing languages), some minor languages may not have structure resources. That being said, our idea still works well even in the absence of the targetside structure annotations. With only the structure annotations at pivot-side (resource-rich) language (in this case, the cross-modal semantic&syntactic structure aligning learning are canceled), we can still achieve much better performances than those baselines without using the structural features. Besides, our method will be subject to the quality of the external structure parsers. When the parsed structures of scene graphs and syntax trees are with much noise, the helpfulness of our methods will be hurt. Fortunately, the existing external semantic and syntactic structure parsers have already achieved satisfactory performances, which can meet our demands. 

## **References** 

- Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. 2018. Bottom-up and top-down attention for image captioning and visual question answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 6077–6086. 

- Loïc Barrault, Ondˇrej Bojar, Marta R. Costa-jussà, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Philipp Koehn, Shervin Malmasi, Christof Monz, Mathias Müller, Santanu Pal, Matt Post, and Marcos Zampieri. 2019. Findings of the 2019 conference on machine translation (WMT19). In _Proceedings of the Fourth Conference on Machine Translation_ , pages 1–61. 

- Marcella Cornia, Matteo Stefanini, Lorenzo Baraldi, and Rita Cucchiara. 2020. Meshed-memory transformer for image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10575–10584. 

- Michael Denkowski and Alon Lavie. 2014. Meteor universal: Language specific translation evaluation for any target language. In _Proceedings of the Ninth Workshop on Statistical Machine Translation_ , pages 376–380. 

- Sergey Edunov, Myle Ott, Michael Auli, and David Grangier. 2018. Understanding back-translation at scale. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pages 489–500. 

- Hao Fang, Saurabh Gupta, Forrest N. Iandola, Rupesh Kumar Srivastava, Li Deng, Piotr Dollár, Jianfeng Gao, Xiaodong He, Margaret Mitchell, John C. Platt, C. Lawrence Zitnick, and Geoffrey Zweig. 2015. From captions to visual concepts and back. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 1473–1482. 

- Qingkai Fang and Yang Feng. 2022. Neural machine translation with phrase-level universal visual representations. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics_ , pages 5687–5698. 

- Hao Fei, Fei Li, Bobo Li, and Donghong Ji. 2021. Encoder-decoder based unified semantic role labeling with label-aware syntax. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 12794–12802. 

- Hao Fei, Shengqiong Wu, Jingye Li, Bobo Li, Fei Li, Libo Qin, Meishan Zhang, Min Zhang, and Tat-Seng Chua. 2022. Lasuie: Unifying information extraction with latent adaptive structure-aware generative language model. In _Proceedings of the Advances in Neural Information Processing Systems, NeurIPS 2022_ , pages 15460–15475. 

- Jiahui Gao, Yi Zhou, Philip L. H. Yu, Shafiq R. Joty, and Jiuxiang Gu. 2022. UNISON: unpaired crosslingual image captioning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 10654– 10662. 

- Jiuxiang Gu, Jianfei Cai, Gang Wang, and Tsuhan Chen. 2018a. Stack-captioning: Coarse-to-fine learning for image captioning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 6837– 6844. 

- Jiuxiang Gu, Shafiq R. Joty, Jianfei Cai, and Gang Wang. 2018b. Unpaired image captioning by language pivoting. In _Proceedings of the European Conference on Computer Vision_ , pages 519–535. 

- Po-Yao Huang, Junjie Hu, Xiaojun Chang, and Alexander Hauptmann. 2020. Unsupervised multimodal neural machine translation with pseudo visual pivoting. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 8226–8237. 

- Paria Jamshid Lou, Yufei Wang, and Mark Johnson. 2019. Neural constituency parsing of speech transcripts. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 2756–2765. 

- Paria Jamshid Lou, Yufei Wang, and Mark Johnson. 2020. Improving disfluency detection by selftraining a self-attentive model. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 3754–3763. 

- Jiayi Ji, Yunpeng Luo, Xiaoshuai Sun, Fuhai Chen, Gen Luo, Yongjian Wu, Yue Gao, and Rongrong Ji. 2021. Improving image captioning by leveraging intra- and inter-layer global representation in transformer network. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 1655–1663. 

- Xu Jia, Efstratios Gavves, Basura Fernando, and Tinne Tuytelaars. 2015. Guiding the long-short term memory model for image caption generation. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 2407–2415. 

- Justin Johnson, Agrim Gupta, and Li Fei-Fei. 2018. Image generation from scene graphs. In _Proceedings of the 2018 IEEE Conference on Computer Vision and Pattern Recognition_ , pages 1219–1228. 

- Justin Johnson, Ranjay Krishna, Michael Stark, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei. 2015. Image retrieval using scene graphs. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 3668–3678. 

- Nikita Kitaev and Dan Klein. 2018. Constituency parsing with a self-attentive encoder. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics_ , pages 2676–2686. 

- Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei. 2017. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International Journal of Computer Vision_ , 123(1):32–73. 

- Weiyu Lan, Xirong Li, and Jianfeng Dong. 2017. Fluency-guided cross-lingual image captioning. In _Proceedings of the ACM International Conference on Multimedia_ , pages 1549–1557. 

- Xirong Li, Chaoxi Xu, Xiaoxu Wang, Weiyu Lan, Zhengxiong Jia, Gang Yang, and Jieping Xu. 2019. COCO-CN for cross-lingual image tagging, captioning, and retrieval. _IEEE Transactions on Multimedia_ , 21(9):2347–2360. 

- Zuchao Li, Masao Utiyama, Eiichiro Sumita, and Hai Zhao. 2021. Unsupervised neural machine translation with universal grammar. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 3249–3264. 

- Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In _Text Summarization Branches Out_ , pages 74–81. 

- Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. 2014. Microsoft COCO: common objects in context. In _Proceedings of the European Conference on Computer Vision_ , pages 740–755. 

- Lajanugen Logeswaran and Honglak Lee. 2018. An efficient framework for learning sentence representations. In _Proceedings of the International Conference on Learning Representations_ . 

- Diego Marcheggiani and Ivan Titov. 2017. Encoding sentences with graph convolutional networks for semantic role labeling. In _Proceedings of the Conference on Empirical Methods in Natural Language Processing_ , pages 1506–1515. 

- Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. 1993. Building a large annotated corpus of english: The penn treebank. _Computational Linguistics_ , 19(2):313–330. 

- Takashi Miyazaki and Nobuyuki Shimizu. 2016. Crosslingual image caption generation. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1780–1790. 

- Joakim Nivre. 2015. Towards a universal grammar for natural language processing. In _Proceedings of the Computational Linguistics and Intelligent Text Processing_ , pages 3–16. 

- Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics_ , pages 311–318. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of Machine Learning Research_ , 21:140:1–140:67. 

- Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. 2015. Faster R-CNN: towards real-time object detection with region proposal networks. In _Proceedings of the Annual Conference on Neural Information Processing Systems_ , pages 91–99. 

- Sebastian Schuster, Ranjay Krishna, Angel Chang, Li Fei-Fei, and Christopher D. Manning. 2015. Generating semantically precise scene graphs from textual descriptions for improved image retrieval. In _Proceedings of the Fourth Workshop on Vision and Language_ , pages 70–80. 

- Lane Schwartz, Chris Callison-Burch, William Schuler, and Stephen Wu. 2011. Incremental syntactic language models for phrase-based translation. In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_ , pages 620–631. 

- Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Improving neural machine translation models with monolingual data. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics_ , pages 86–96. 

- Yuqing Song, Shizhe Chen, Yida Zhao, and Qin Jin. 2019. Unpaired cross-lingual image caption generation with self-supervised rewards. In _Proceedings of the ACM International Conference on Multimedia_ , pages 784–792. 

- Liang Tian, Derek F. Wong, Lidia S. Chao, Paulo Quaresma, Francisco Oliveira, and Lu Yi. 2014. Umcorpus: A large english-chinese parallel corpus for statistical machine translation. In _Proceedings of the Ninth International Conference on Language Resources and Evaluation_ , pages 1837–1842. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Proceedings of the Annual Conference on Neural Information Processing Systems_ , pages 5998–6008. 

- Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. 2015. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 4566–4575. 

- Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. 2017. Show and tell: Lessons learned from the 2015 MSCOCO image captioning challenge. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 39(4):652–663. 

- Yu-Siang Wang, Chenxi Liu, Xiaohui Zeng, and Alan Yuille. 2018. Scene graph parsing as dependency parsing. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 397–407. 

- Ronald J. Williams. 1992. Simple statistical gradientfollowing algorithms for connectionist reinforcement learning. _Machine Learning_ , 8:229–256. 

- Jiahong Wu, He Zheng, Bo Zhao, Yixin Li, Baoming Yan, Rui Liang, Wenjia Wang, Shipei Zhou, Guosen Lin, Yanwei Fu, Yizhou Wang, and Yonggang Wang. 2017. AI challenger : A large-scale dataset for going deeper in image understanding. _CoRR_ , abs/1711.06475. 

- Shengqiong Wu, Hao Fei, Yafeng Ren, Donghong Ji, and Jingye Li. 2021. Learn from syntax: Improving pair-wise aspect and opinion terms extraction with rich syntactic knowledge. In _Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence_ , pages 3957–3963. 

- Hongfei Xu, Josef van Genabith, Deyi Xiong, Qiuhui Liu, and Jingyi Zhang. 2020. Learning source phrase representations for neural machine translation. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 386– 396. 

- Nianwen Xue, Fei Xia, Fu-Dong Chiou, and Martha Palmer. 2005. The penn chinese treebank: Phrase structure annotation of a large corpus. _Natural Language Engineering_ , 11(2):207–238. 

- Yuanmeng Yan, Rumei Li, Sirui Wang, Fuzheng Zhang, Wei Wu, and Weiran Xu. 2021. ConSERT: A contrastive framework for self-supervised sentence representation transfer. In _Proceedings of the Annual Meeting of the Association for Computational Linguistics and the International Joint Conference on Natural Language Processing_ , pages 5065–5075. 

- Xu Yang, Kaihua Tang, Hanwang Zhang, and Jianfei Cai. 2019. Auto-encoding scene graphs for image captioning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 10685–10694. 

- Quanzeng You, Hailin Jin, Zhaowen Wang, Chen Fang, and Jiebo Luo. 2016. Image captioning with semantic attention. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 4651–4659. 

- Rowan Zellers, Mark Yatskar, Sam Thomson, and Yejin Choi. 2018. Neural motifs: Scene graph parsing with global context. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 5831–5840. 

- Jiajun Zhang and Chengqing Zong. 2013. Learning a phrase-based translation model from monolingual data with application to domain adaptation. In _Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics_ , pages 1425– 1434. 

- Yue Zhang and Zhenghua Li. 2022. Csyngec: Incorporating constituent-based syntax for grammatical error correction with a tailored gec-oriented parser. _CoRR_ , abs/2211.08158. 

- Xin Zhao, Lei Wu, Xu Chen, and Bin Gong. 2022. High-quality image generation from scene graphs with transformer. In _Proceedings of the IEEE International Conference on Multimedia and Expo_ , pages 1–6. 

## **A Model Details** 

## **A.1 Specification of Scene Graph and Syntax Constituency Structures** 

In Fig. 9 and Fig. 10 we illustrate the complete structures of the syntactic constituency tree and scene graphs, respectively. We note that the scene graph is a dependency-like structure, describing the node-node inter-relation in an ‘is-a’ paradigm. And the syntactic constituency tree is a compositional structure, depicting how words constitute phrases and then organize them into whole sentences. 

## **A.2 Pivot-to-target Translation** 

In Eq. (4) we use a Transformer decoder to predict the target caption _S_[ˆ] _[t]_ . A cross-attention mechanism is first used to fuse the prior SG feature representations _**h**_ and the SC features _**r**_ . Specifically, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0012-13.png)


where _d_ is a scaling factor. Then, the Transformer performs decoding over _{_ _**e** }_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0012-15.png)


## **A.3 Specification on Contrastive Learning** 

**Cross-modal Semantic Structure Aligning** In Eq. (8) we define the contrastive learning objective of cross-modal semantic structure aligning, here we unfold the equation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0012-18.png)


where _τm_ >0 is an annealing factor. _j[∗]_ represents a positive pair with _i_ , i.e., _s[m] i,j[∗]_[>] _[ρ][m]_[.] 

**Cross-lingual Syntactic Structure Aligning** We detail the cross-lingual syntactic structure aligning learning objective here: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0012-21.png)


where _τl_ >0 is an annealing factor. _j[∗]_ represents a positive pair with _i_ , i.e., _s[m] i,j[∗]_[>] _[ρ][m]_[.] 

## **A.4 Specifying Overall Training Processing** 

The training of our framework is based on the warm-up strategy, including four stages. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0013-00.png)


**----- Start of picture text -----**<br>
ROOT<br>S<br>VP<br>NP VBZ VP<br>PP VBG PP<br>NP IN NP is speaking IN NP<br>DT NN to NP PP<br>in DT JJ NN DT NN IN NP<br>A boy<br>a white shirt a group of NP VP<br>NNS VBG PP PP<br>people sitting IN NP IN NP<br>Constituency phrase node at NN in NP PP<br>table NN IN NP<br>Part-of-speech (POS) node<br>front of DT NN<br>Word node a park<br>**----- End of picture text -----**<br>


Input Sentence :  A boy in a white shirt is speaking to a group of people sitting at table in front of a park 

Figure 9: Illustration of the full constituency syntax structure. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0013-03.png)


**----- Start of picture text -----**<br>
A boy in a white shirt<br>sitting at table in a park<br>is speaking to a group<br>of people.<br>VSG Parsing LSG Parsing<br>speak boy in shirt speak boy in shirt<br>people white people sit white<br>sit table<br>table in front of park in front of park<br>Visual Scene Graph (VSG) Language Scene Graph<br>Object Node Relation Node Attribute Node<br>**----- End of picture text -----**<br>


Figure 10: Illustration of the full scene graph structures. The object nodes are the real entities of visual proposals or textual words/phrases. The attribute nodes and relation nodes are the descriptive words, which, in VSG are automatically generated, and in LSG are extracted from sentences. 

ciently align the captioning and translation parts. 

**At the fourth stage,** the system tends to converge, and we put all the above learning objects together for further overall fine-tuning: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0013-07.png)


Here _λ∗_ are the loss weights that dynamically change by linearly learning scheduler (Huang et al., 2020). The initial weights are given as: _λ_ Cap=1, _λ_ Trans=1, _λ_ CMA=0.7, _λ_ CLA=0.7, _λ_ VCB=0.3, _λ_ CPB=0.3. _λ_ Cap and _λ_ Trans will be linearly decreased from 1 to 0.7 along the training, _λ_ CMA and _λ_ REC are kept unchanged, while _λ_ VCB and _λ_ CPB will be decreased from 0.3 to 0.7. 

## **B Extended Experiment Setups** 

**At the first stage,** we use the paired imagecaption data _{_ ( _I, S[p]_ ) _}_ at the pivot language side (as well as the VSG structure features) to train the captioning part of our model; and use the parallel sentences _{_ ( _S[p] , S[t]_ ) _}_ (as well as the pivot-side syntax tree features) to train the translation part of our model, both of two training is supervised. 

**At the second stage,** we perform two structurepivoting unsupervised alignment learning, by using the image-caption data _{_ ( _I, S[p]_ ) _}_ , parallel sentences _{_ ( _S[p] , S[t]_ ) _}_ , and the two structure resource. 

**At the third stage,** we perform the cross-modal and cross-lingual back-translation learning. This is a whole-framework-level training, aiming to suffi- 

## **B.1 Dataset Description** 

We use three image captioning datasets _{_ ( _I, S[p]_ ) _}_ : MSCOCO, AIC-ICC and COCO-CN. All the data split follows the same practice as in prior crosslingual image captioning works (Wu et al., 2017; Song et al., 2019). The MSCOCO dataset is annotated in English, which consists of 123,287 images and 5 manually labeled English captions for each image. We utilize 113,287 images for training, 5,000 images for validation, and 5,000 images for testing. The AIC-ICC dataset contains 238,354 images and 5 manually annotated Chinese captions for each image. There are 208,354 and 30,000 images in the official training and validation set. 

|Dataset|**Lang.**|**Split**|**#Image**|**#Caption**|
|---|---|---|---|---|
|||Total|123,287|616,435|
|MSCOCO|En|Train<br>Develop|113,287<br>5,000|566,435<br>25,000|
|||Test|5,000|25,000|
|||Total|238,354|1,191,770|
|AIC-ICC|Zh|Train<br>Develop|208,354<br>25,000|1,041,770<br>125,000|
|||Test|5,000|25,000|
|||Total|20,342|27,218|
|COCO-CN|Zh|Train<br>Develop|18,342<br>1,000|25,218<br>1,000|
|||Test|1,000|1,000|



Table 5: Statistics of image captioning datasets. 

Since the annotations of the testing set are unavailable in the AIC-ICC dataset, we randomly sample 5,000 images from its validation set as our testing set. The COCO-CN dataset contains 20,342 images and 27,218 caption texts in Chinese. We use 18,342 images for training, 1,000 for development, and 1,000 for testing. Table 5 gives the detailed statistics of the image captioning data. 

For the translation data _{_ ( _S[p] , S[t]_ ) _}_ , we collect about 1M of raw paired En-Zh parallel sentences from the UM (Tian et al., 2014) and WMT19 (Barrault et al., 2019) machine translation corpus. We filter the sentences in MT datasets according to an existing caption-style dictionary and resulting in a total of 400,000 parallel sentences. For the translation training, we use 390,000 sentence pairs for training, 5,000 sentence pairs for validation, and 5,000 pairs for testing. 

## **B.2 Specification on Structure Coincidence Probing** 

In Fig. 6 we assess the ability of our model on the cross-modal and cross-lingual structure alignment, by measuring the structure coincidence between the gold one and the one learned by our model. Here we detail the evaluation setup. 

For the semantic scene structures, we evaluate the coincidence between the input images’ SGs and the SGs of predicted target-side captions. These SG structures are parsed by the same methods introduced above. We then make statistics of the overlapped node pairs between the two SGs as the coincidence rate _β[G]_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0014-07.png)


where SG _[V]_ and SG _[L]_ denote any word-pair structure of visual SG and target language SG, respectively. 

For the syntax structures, we evaluate the co- 

incidence rate of the constituency tree structures between the intermediate pivot captions and the final predicted target-side captions. (Because the input images come without the syntax trees.) The SC structures of two languages are parsed using the parsers introduced above. We note that the divergences of syntax between two languages can be much larger, compared with the divergences of semantic scene structures. Different from the measurement for SG structure to traverse the whole graph equally, we measure the SC structure coincidence rate _β[C]_ in a top-down manner. Specifically, we traverse the constituency trees in a top-down order, and those matched phrasal nodes at a higher level (lower traversing depth from the root node) will have higher scores than those at a lower level. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment_images/acl_2023_Cross2StrA_Unpaired_Cross-lingual_Image_Captioning_with_Cross-lingual_Cross-modal_Structure-pivoted_Alignment.pdf-0014-11.png)


where SC _[P]_ and SC _[T]_ denote the phrasal constituent structures of the pivot and target language, respectively. _d_ is a weight, which is defined as the reciprocal of a top-down traversing depth. 

## **B.3 Specifications of Human Evaluation Standards** 

Table 4 shows the human evaluation results. Specifically, we design a Likert 10-scale to measure the relevancy, diversification, and fluency of the generated target-side captions. The 10-scale metrics are defined as: 1-Can’t be worse, 2-Terrible, 3-Poor, 4-Little poor, 5-Average, 6-Better than average, 7- Adequate, 8-Good, 9-Very good, 10-Excellent. We ask ten native Chinese speakers to score the results. And for each result, we use the averaged scores. 

In Fig. 8 we also measure the language quality of captions in terms of _wording_ , _word order_ , and _syntax correctness_ . We ask the same ten native Chinese speakers to score the error degree of these metrics, each of which is defined as: 

- **Wording** : Is the choice of words in the captions suitable and precise to describe the input images? 

- **Word order** : Are the words, phrases, and components organized correctly and properly in captioning sentences? 

- **Syntax correctness** : Are there syntactic errors in the caption texts? such as omitting or repeating words, mixing up verb tenses or verb conjugations, missing prepositions, etc. 

