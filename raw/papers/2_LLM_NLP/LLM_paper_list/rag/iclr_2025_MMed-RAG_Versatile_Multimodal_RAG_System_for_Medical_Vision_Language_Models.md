Preprint. 

# - MMED-RAG: VERSATILE MULTIMODAL RAG SYS TEM FOR MEDICAL VISION LANGUAGE MODELS 

**Peng Xia**[1] **, Kangyu Zhu**[5] **, Haoran Li**[6] **, Tianze Wang**[3] **, Weijia Shi**[4] **, Sheng Wang**[4] **, Linjun Zhang**[3] **, James Zou**[2] **, Huaxiu Yao**[1] 

1UNC-Chapel Hill, 2Stanford University, 3Rutgers University, 4University of Washington, 

5Brown University, 6PloyU 

_{_ pxia,huaxiu _}_ @cs.unc.edu 

## ABSTRACT 

Artificial Intelligence (AI) has demonstrated significant potential in healthcare, particularly in disease diagnosis and treatment planning. Recent progress in Medical Large Vision-Language Models (Med-LVLMs) has opened up new possibilities for interactive diagnostic tools. However, these models often suffer from factual hallucination, which can lead to incorrect diagnoses. Fine-tuning and retrieval-augmented generation (RAG) have emerged as methods to address these issues. However, the amount of high-quality data and distribution shifts between training data and deployment data limit the application of fine-tuning methods. Although RAG is lightweight and effective, existing RAG-based approaches are not sufficiently general to different medical domains and can potentially cause misalignment issues, both between modalities and between the model and the ground truth. In this paper, we propose a versatile multimodal RAG system, MMed-RAG, designed to enhance the factuality of Med-LVLMs. Our approach introduces a domain-aware retrieval mechanism, an adaptive retrieved contexts selection method, and a provable RAG-based preference fine-tuning strategy. These innovations make the RAG process sufficiently general and reliable, significantly improving alignment when introducing retrieved contexts. Experimental results across five medical datasets (involving radiology, ophthalmology, pathology) on medical VQA and report generation demonstrate that MMed-RAG can achieve an average improvement of 43.8% in the factual accuracy of Med-LVLMs. Our data and code are available in https://github.com/richard-peng-xia/MMed-RAG. 

## 1 INTRODUCTION 

Artificial Intelligence (AI) has already transformed healthcare and still has a lot of potential for further advancements (T˘aut¸an et al., 2021; Wang et al., 2019; Ye et al., 2021; Tu et al., 2024; Xia et al., 2024b; Hu et al., 2024a;b; Li et al., 2024). Recently, Medical Large Vision-Language Models (Med-LVLMs) have shown great promise for advancing interactive and intelligent diagnosis (Li et al., 2023a; Moor et al., 2023; Zhang et al., 2023b; Wu et al., 2023b). Despite this potential (Li et al., 2023b; Wu et al., 2023a; Shi et al., 2024), current Med-LVLMs still face significant reliability issues, particularly their tendency to generate non-factual medical responses (Xia et al., 2024a; Royer et al., 2024; Chen et al., 2024a; Jiang et al., 2024), making them unreliable in critical medical applications. These factuality issues raise serious concerns when deploying such models in clinical settings, where even small diagnostic errors could lead to severe consequences for patient care. 

Recently, researchers have begun to focus on improving the factuality of Med-LVLMs through various techniques, including fine-tuning (Li et al., 2023a; Moor et al., 2023; Thawkar et al., 2023; Zhang et al., 2023b; Chen et al., 2024b) and retrieval-augmented generation (RAG) (Xia et al., 2024c; He et al., 2024; Sun et al., 2024). Fine-tuning is a direct method to improve model performance, but it faces several limitations in the medical field. First, there is a lack of sufficient high-quality labeled data for fine-tuning in the medical domain. Additionally, a distribution gap often exists between the training data and the real-world deployment data (Schrouff et al., 2022), leading to significantly worse model performance during deployment. Hence, RAG has emerged as a viable alternative by providing external references during the inference stage, enhancing the factuality of Med-LVLMs (Wu et al., 2023c; Gao et al., 2023). However, despite its advantages, current RAG implementations in Med-LVLMs have significant limitations. First, these methods tend to be _dataset-specific_ , reducing their generalizability across various medical domains. Second, these 

1 

Preprint. 

models are still facing _misalignment issues_ that lead to factuality problems. This misalignment may arise from the impact of adding RAG on the original Med-LVLMs’ _cross-modality alignment_ , as well as on the _overall alignment_ between the model and ground truth. 

To address these challenges, we propose a versatile factual **M** ultimodal **Med** ical **RAG** system called **MMed-RAG** . Specifically, MMed-RAG first introduces a domain-aware retrieval mechanism, designed to handle different domains of medical images more effectively. Here, we design a domain identification module to adaptively select a corresponding retrieval model given the input medical image. Secondly, we include a adaptive calibration approach for selecting the number of retrieved contexts. Lastly, MMed-RAG incorporates RAG-based preference fine-tuning to enhance crossmodality alignment and overall alignment with ground truth. The preference pairs are designed to achieve two goals: first, to improve cross-modality alignment by encouraging the model to avoid generating responses without utilizing input medical images, even the responses are correct; second, to improve overall alignment by encouraging the model to understand retrieved contexts when unsure, while avoiding interference from irrelevant retrieved information. 

The primary contribution of this paper is MMed-RAG, a versatile multimodal RAG system designed specifically for Med-LVLMs to generate more factual responses. Under mild assumptions, our theoretical analysis demonstrates that MMed-RAG mitigates both cross-modality misalignment and overall misalignment with ground truth. Furthermore, empirical results on five medical multimodal datasets, covering three medical image modalities (radiology, pathology, and ophthalmology), show that MMed-RAG significantly improves the factual accuracy of Med-LVLMs, achieving improvements of 18.5% and 69.1% on Medical VQA and report generation tasks, respectively, compared to the original Med-LVLM. These empirical findings further demonstrate the effectiveness of our proposed components and support the theoretical analysis in addressing misalignment issues. 

## 2 PRELIMINARIES 

In this section, we will provide a brief overview of Med-LVLMs and preference optimization. 

**Medical Large Vision Language Models** . Med-LVLMs bridge LLMs with medical visual modules, allowing the model to take medical image _xv_ and clinical query _xt_ as input _x_ , and autoregressively predict the probability distribution of the next token. The text output is denoted as _y_ . 

**Preference Optimization** . Preference optimization has achieved remarkable results in LLM alignment. Give an input _x_ , a language model policy _πθ_ can produce a conditional distribution _πθ_ ( _y | x_ ) with _y_ as the output text response. The recently popular DPO (Rafailov et al., 2023) utilizes preference data achieve objective alignment in LLMs. The preference data is defined as _D_ = _{x_[(] _[i]_[)] _, yw_[(] _[i]_[)] _[, y] l_[(] _[i]_[)] _[}] i[N]_ =1[,][where] _[y] w_[(] _[i]_[)][and] _[y] l_[(] _[i]_[)] represent preferred and dispreferred responses given an input prompt _x_ . The probably of obtaining each preference pair is _p_ ( _yw ≻ yl_ ) = _σ_ ( _r_ ( _x, yw_ ) _− r_ ( _x, yl_ )) _,_ where _σ_ ( _·_ ) is the sigmoid function. In DPO, the optimization can be formulated as classification loss over the preference data as: 

_LDPO_ ( _πθ_ ; _π_ ref) = _−_ E( _x,yw ,yl_ ) _∼D_ �log _σ_ � _α_ log _π[π]_ ref _[θ]_[(] ( _[y] y[w] w[|] |[x] x_[)] ) _[−][α]_[ log] _π[π]_ ref _[θ]_[(] ( _[y] y[l] l[|] |[x] x_[)] ) �� _._ (1) where _πθ_ represents the reference policy, which is the LLM fine-tuned through supervised learning. 

## 3 MMED-RAG: A VERSATILE MEDICAL RAG SYSTEM 

In this section, as illustrated in Figure 1, we will propose MMed-RAG, a versatile RAG system for improving the factuality of Med-LVLMs. Specifically, MMed-RAG consists of three complementary modules. First, we design a domain-aware retrieval mechanism to select the optimal retriever by feeding each given medical image to the domain identification module. Second, to select an optimal number of retrieved contexts and filter out low-quality information, MMed-RAG adopts a adaptive method by filtering out low-quality information using the similarity scores during the RAG phase. Lastly, we use a RAG-based preference fine-tuning approach to improve the cross-modality alignment and the overall alignment between groundtruth. We detail these steps as follows: 

## 3.1 DOMAIN-AWARE RETRIEVAL MECHANISM 

In MMed-RAG, we introduce a domain-aware retrieval mechanism to efficiently handle medical images from different sources (e.g., radiology, pathology, ophthalmology). Specifically, we first 

2 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0003-01.png)


**----- Start of picture text -----**<br>
Domain-Aware Retrieval Mechanism Adaptive Retrieved Context Selection<br>Med-LVLM<br>Medical  Is there any<br>Radiology Retriever  Image Question opacity present?focal airspace  Constructed<br>Domain  (Radiology) Domain  … Preference Pairs<br>Identification Radiology Label Adaptive- k  Reports<br>Retriever<br>Pathology (Pathology) Retriever …<br>Top- k  Reports Similarity Scores<br>RAG-Based Preference Fine-Tuning Preference Data<br>Original Image Med-LVLM Original Image Med-LVLM<br>Unrelated Image<br>RAG<br>Original Image Cannot Solve Problems by Self<br>Direct Copy Homework from Others Learn How to Copy Preference<br>Think it by Self Fine-Tuning<br>Original Image Med-LVLM Unrelated Image Med-LVLM Original Image Med-LVLM Original Image Med-LVLM<br>RAG<br>RAG RAG Copied homework is Wrong<br>Avoid Interference from Incorrect Homework Stronger Med-LVLM<br>IU-Xray<br>MIMIC<br>Quilt<br>Ground-Truth<br>**----- End of picture text -----**<br>


Figure 1: Overview of MMed-RAG, a versatile factual multimodal RAG system designed to enhance the reliability of Med-LVLMs. It introduces a domain-aware retrieval mechanism that effectively handles different domains of medical images by selecting suitable retrieval models. Additionally, it uses an adaptive context selection approach to determine the optimal number of retrieved contexts and employs preference fine-tuning to improve both cross-modality and overall alignment. 

employ a domain identification module that assigns a domain label to each input medical image. To achieve this, we create a small dataset with medical images as inputs and their corresponding domain labels as outputs, using this dataset to fine-tune the BiomedCLIP model (Zhang et al., 2023a) to improve its domain awareness. Formally, for a given medical image _xv_ , we predict its domain _d_ = _F_ ( _xv_ ). Based on the assigned domain label _d_ , the image _xv_ is fed into the corresponding multimodal retriever _Rd_ ( _·_ ) for knowledge retrieval. 

Here, each multimodal retriever _Rd_ ( _·_ ) for each domain _d_ is trained through contrastive learning (Radford et al., 2021). Specifically, the visual and textual information _Ximg, Xtxt_ are processed by their corresponding encoders _Eimg_ ( _·_ ) _, Etxt_ ( _·_ ) to generate textual and visual embeddings _Vtxt_ = _Etxt_ ( _Xtxt_ ) _, Vimg_ = _Eimg_ ( _Ximg_ ). Contrastive learning loss is then applied to maximize the similarity between text and image embeddings representing the same example, while minimizing the similarity between embeddings representing different examples, as defined below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0003-05.png)


where _S ∈_ R _[N][×][N]_ represents the similarity matrix between image and text modalities, calculated as: _S_ = _|VVimimgg|[·]_[ (] _|V[V] txt[txt] |_[)] _[T]_[ ,][where][each][element] _[S][i,j]_[represents][the][similarity][between][the][image] representation of example _i_ and the text representation of example _j_ . 

Finally, for the input image _xt_ , after feeding into the corresponding multimodal retriever _Rd_ ( _·_ ), the multimodal retriever will retrieves the top- _k_ most similar reports for the image. These retrieved reports _xr_ = _Rd_ ( _xv_ ) are then provided to the Med-LVLM _M_ ( _·_ ) as references to guide the generation. 

## 3.2 ADAPTIVE RETRIEVED CONTEXT SELECTION 

Following the domain-aware retrieval mechanism, the next step is to determine the optimal amount of context to retrieve. Retrieving too much or too little information can result in hallucinations (Xia et al., 2024c). Current RAG methods applied to Med-LVLMs generally rely on empirical results or fixed values based on validation sets to select the optimal value of the number of retrieved contexts _k_ (Xia et al., 2024c; He et al., 2024; Sun et al., 2024). However, the distribution of similarity scores varies depending on the complexity of the image and its 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0003-10.png)


Figure 2: Relations between selected contexts and similarity. 

3 

Preprint. 

alignment with the textual information from the data source. These fixed- _k_ methods do not guarantee optimal performance on target data, as they overlook the similarity scores generated during the retrieval process. To address this, we propose an adaptive method that dynamically selects _k_ based on the similarity scores of the retrieved contexts. Specifically, during the domain-aware retrieval mechanism phase, the retrieved information is denoted as _xr_ ( _k_ ) = _Rd_ ( _xv_ ; _k_ ), where _k_ represents the number of retrieved contexts, and the corresponding similarity scores are denoted as _Sk_ . For simplicity, when there is no ambiguity, we will refer to _xr_ ( _k_ ) as _xr_ . 

As illustrated in Figure 2, our method is based on a key observation: the similarity scores (CLIP score in this case) between retrieved contexts often exhibit a sharp decline after a certain number of results (nearly top-9 in this case). This suggests that lower-quality information can still be included among the top- _k_ retrieved contexts when using a fixed- _k_ strategy, especially in cases where the fixed value of _k_ is too large. These lower-quality retrievals introduce noise and irrelevant information, which can significantly impair the model’s ability to generate factual and coherent responses. To mitigate this issue, we draw inspiration from the Gap statistic method used in clustering (Tibshirani et al., 2001) and extend this concept to RAG for Med-LVLMs. Specifically, after retrieving the top- _k_ contexts, we perform an additional round of _k_ optimization by analyzing the similarity ratios between consecutive retrievals. These similarity ratios are denoted as _ui_ = log( _Si/Si_ +1) for 0 _< i ≤ k_ , where _Si_ represents the similarity score of the _i_ -th retrieved context. When _ui_ exceeds a predefined threshold _γ_ , this indicates a substantial drop in relevance, suggesting that the remaining retrievals are less likely to contribute preferredly to the model’s output. At this point _i_ , we truncate _k_ , effectively discarding the less relevant retrievals that follow. This adaptive truncation mechanism ensures that only the most relevant contexts are retained for generating the final response, reducing the risk of hallucination and improving the factual accuracy of the outputs. 

Although the threshold _γ_ is fixed, this approach provides a adaptive way to balance the bias and variance in retrieved contexts. By adapting to the characteristics of each input _xv_ , our method enhances the robustness of the retrieval process and ensures that the selection of _k_ is tailored to the specific data at hand, thereby improving overall performance across diverse contexts and tasks. 

## 3.3 RAG-BASED PREFERENCE FINE-TUNING 

After context selection, MMed-RAG supplies Med-LVLM with reliable retrieved information as external knowledge to aid in generating factual responses. However, incorporating this retrieved knowledge may potentially disrupt the original alignment within the existing Med-LVLM, a concern we will elaborate on below: 

**Alignment Analysis.** In the alignment analysis, we aim to explore how incorporating retrieved context impacts the original alignment in Med-LVLMs, focusing on two key aspects: (1) cross-modality alignment and (2) overall alignment with the ground truth. To evaluate cross-modality alignment, we conduct two tests on LLaVA-Med-1.5 (Li et al., 2023a) using the Harvard-FairVLMed (Luo et al., 2024) dataset. First, when replacing the original image with a highly noisy image associated with a different ground truth, the original model gives incorrect answers (the ground truth being the response for the original image). After incorporating RAG, where context is retrieved based on the original image, 55.08% of these cases return correct answers. This indicates that the model _directly references the retrieved knowledge_ without considering the input image, highlighting significant _cross-modal misalignment issues_ . Furthermore, 43.31% of the questions that were originally answered correctly are answered incorrectly after incorporating RAG, suggesting _interference from incorrect retrieval information_ , which leads to _overall misalignment with the ground truth_ . 

To address cross-modality misalignment and the overall misalignment introduced by incorporating retrieved knowledge, as shown in Algorithm 1, we propose a RAG-based preference fine-tuning (RAG-PT) approach to fine-tune the target Med-LVLM _M_ ( _·_ ). Specifically, RAG-PT constructs two types of preference pairs designed to mitigate both categories of misalignment. 

**Preference Pairs for Cross-Modality Alignment.** We first construct preference pairs aimed at improving cross-modality alignment. In this dataset, we select samples from _D_ = _{x_[(] _v[i]_[)] _[, x]_[(] _t[i]_[)] _[, y]_[(] _[i]_[)] _[}] i[N]_ =1[,] where _xv_ , _xt_ , and _y_ represent the input medical image, clinical query, and ground-truth answer, respectively. For simplicity, we omit the sample index ( _i_ ) in the following sections. A model’s correct response using retrieved knowledge, i.e., _M_ ( _xv, xt_ + _xr_ ) = _y_ , is considered a preferred response _pi_ , where _xr_ is the retrieved information. A dispreferred response _ni_ is selected from cases 

4 

Preprint. 

## **Algorithm 1:** Versatile Multimodal RAG System (MMed-RAG) 

|**Input:** _D_ =_{x_(_i_)<br>_v , x_(_i_)<br>_t , y_(_i_)_}N_<br>_i_=1: Dataset;_πθ_: Parameters of the Med-LVLM; Med-LVLM:_M_(_·, ·_);<br>Domain Identifcation: _F_(_·_); Retriever: _R_(_·_); Noisy Function: _I_(_·_).<br>**Output:** _π_ref: Parameters of the reference model.<br>**1** _▷_**_Training Stage_**<br>**2** Initialize_Dcm_ with an empty set<br>**3 foreach**<br>(_xv, xt, y_)_∈D_ **do**<br>**4**<br>Generate retrieved contexts with an assigned domain label_xr ←RF_(_xv_)(_xv_)<br>**5**<br>Generate the noisy image_x∗_<br>_v ←I_(_xv_)<br>**6**<br>_▷Cross-Modality Alignment_<br>**7**<br>**if**<br>_M_(_xv,_(_xt, xr_))=_y_<br>and<br>_M_(_x∗_<br>_v,_(_xt, xr_))=_y_ **then**<br>**8**<br>Select the preferred response_yw,o_1 _←y_, dispreferred response_yl,o_1 _←M_(_x∗_<br>_v,_(_xt, xr_))<br>**9**<br>Put_{_(_xv, xt_)_, yw,o_1_, yl,o_1_}_into_Dcm_<br>**10**<br>_▷Overall Alignment_<br>**11**<br>Initialize_D_1<br>_oa_ and_D_2<br>_oa_ with empty set<br>**12**<br>**if**<br>_M_(_xv,_(_xt, xr_))=_y_<br>and<br>_M_(_xv, xt_) =_y_ **then**<br>**13**<br>Select the preferred response_yw,o_2 _←y_, dispreferred response_yl,o_2 _←M_(_xv, xt_)<br>**14**<br>Put_{_(_xv, xt_)_, yw,o_2_, yl,o_2_}_into_D_1<br>_oa_<br>**15**<br>**if**<br>_M_(_xv, xt_)=_y_<br>and<br>_M_(_xv,_(_xt, xr_)) =_y_ **then**<br>**16**<br>Select the preferred response_yw,o_3 _←y_, dispreferred response_yl,o_3 _←M_(_xv,_(_xt, xr_))<br>**17**<br>Put_{_(_xv, xt_)_, yw,o_3_, yl,o_3_}_into_D_2<br>_oa_<br>**18** _Dpt_ =_Dcm ∪Doa_,_Doa_ =_D_1<br>_oa ∪D_2<br>_oa_<br>**19 foreach**<br>((_xv, xt_)_, yw,o, yl,o_)_∈Dpt_ **do**<br>**20**<br>Compute the losses_Lpt_ following equation 4 and update_π_ref<br>**21** _▷_**_Inference Stage_**<br>**22 foreach**<br>test<br>sample<br>(_xv, xt_) **do**<br>**23**<br>Select top-k retrieved contexts with an assigned domain label_xr ←RF_(_xv_)(_xv_)<br>**24**<br>Get the predictions of the model w/ RAG-PT_p ←M_(_xv,_(_xt, xr_))|**Input:** _D_ =_{x_(_i_)<br>_v , x_(_i_)<br>_t , y_(_i_)_}N_<br>_i_=1: Dataset;_πθ_: Parameters of the Med-LVLM; Med-LVLM:_M_(_·, ·_);<br>Domain Identifcation: _F_(_·_); Retriever: _R_(_·_); Noisy Function: _I_(_·_).<br>**Output:** _π_ref: Parameters of the reference model.<br>**1** _▷_**_Training Stage_**<br>**2** Initialize_Dcm_ with an empty set<br>**3 foreach**<br>(_xv, xt, y_)_∈D_ **do**<br>**4**<br>Generate retrieved contexts with an assigned domain label_xr ←RF_(_xv_)(_xv_)<br>**5**<br>Generate the noisy image_x∗_<br>_v ←I_(_xv_)<br>**6**<br>_▷Cross-Modality Alignment_<br>**7**<br>**if**<br>_M_(_xv,_(_xt, xr_))=_y_<br>and<br>_M_(_x∗_<br>_v,_(_xt, xr_))=_y_ **then**<br>**8**<br>Select the preferred response_yw,o_1 _←y_, dispreferred response_yl,o_1 _←M_(_x∗_<br>_v,_(_xt, xr_))<br>**9**<br>Put_{_(_xv, xt_)_, yw,o_1_, yl,o_1_}_into_Dcm_<br>**10**<br>_▷Overall Alignment_<br>**11**<br>Initialize_D_1<br>_oa_ and_D_2<br>_oa_ with empty set<br>**12**<br>**if**<br>_M_(_xv,_(_xt, xr_))=_y_<br>and<br>_M_(_xv, xt_) =_y_ **then**<br>**13**<br>Select the preferred response_yw,o_2 _←y_, dispreferred response_yl,o_2 _←M_(_xv, xt_)<br>**14**<br>Put_{_(_xv, xt_)_, yw,o_2_, yl,o_2_}_into_D_1<br>_oa_<br>**15**<br>**if**<br>_M_(_xv, xt_)=_y_<br>and<br>_M_(_xv,_(_xt, xr_)) =_y_ **then**<br>**16**<br>Select the preferred response_yw,o_3 _←y_, dispreferred response_yl,o_3 _←M_(_xv,_(_xt, xr_))<br>**17**<br>Put_{_(_xv, xt_)_, yw,o_3_, yl,o_3_}_into_D_2<br>_oa_<br>**18** _Dpt_ =_Dcm ∪Doa_,_Doa_ =_D_1<br>_oa ∪D_2<br>_oa_<br>**19 foreach**<br>((_xv, xt_)_, yw,o, yl,o_)_∈Dpt_ **do**<br>**20**<br>Compute the losses_Lpt_ following equation 4 and update_π_ref<br>**21** _▷_**_Inference Stage_**<br>**22 foreach**<br>test<br>sample<br>(_xv, xt_) **do**<br>**23**<br>Select top-k retrieved contexts with an assigned domain label_xr ←RF_(_xv_)(_xv_)<br>**24**<br>Get the predictions of the model w/ RAG-PT_p ←M_(_xv,_(_xt, xr_))|**Input:** _D_ =_{x_(_i_)<br>_v , x_(_i_)<br>_t , y_(_i_)_}N_<br>_i_=1: Dataset;_πθ_: Parameters of the Med-LVLM; Med-LVLM:_M_(_·, ·_);<br>Domain Identifcation: _F_(_·_); Retriever: _R_(_·_); Noisy Function: _I_(_·_).<br>**Output:** _π_ref: Parameters of the reference model.<br>**1** _▷_**_Training Stage_**<br>**2** Initialize_Dcm_ with an empty set<br>**3 foreach**<br>(_xv, xt, y_)_∈D_ **do**<br>**4**<br>Generate retrieved contexts with an assigned domain label_xr ←RF_(_xv_)(_xv_)<br>**5**<br>Generate the noisy image_x∗_<br>_v ←I_(_xv_)<br>**6**<br>_▷Cross-Modality Alignment_<br>**7**<br>**if**<br>_M_(_xv,_(_xt, xr_))=_y_<br>and<br>_M_(_x∗_<br>_v,_(_xt, xr_))=_y_ **then**<br>**8**<br>Select the preferred response_yw,o_1 _←y_, dispreferred response_yl,o_1 _←M_(_x∗_<br>_v,_(_xt, xr_))<br>**9**<br>Put_{_(_xv, xt_)_, yw,o_1_, yl,o_1_}_into_Dcm_<br>**10**<br>_▷Overall Alignment_<br>**11**<br>Initialize_D_1<br>_oa_ and_D_2<br>_oa_ with empty set<br>**12**<br>**if**<br>_M_(_xv,_(_xt, xr_))=_y_<br>and<br>_M_(_xv, xt_) =_y_ **then**<br>**13**<br>Select the preferred response_yw,o_2 _←y_, dispreferred response_yl,o_2 _←M_(_xv, xt_)<br>**14**<br>Put_{_(_xv, xt_)_, yw,o_2_, yl,o_2_}_into_D_1<br>_oa_<br>**15**<br>**if**<br>_M_(_xv, xt_)=_y_<br>and<br>_M_(_xv,_(_xt, xr_)) =_y_ **then**<br>**16**<br>Select the preferred response_yw,o_3 _←y_, dispreferred response_yl,o_3 _←M_(_xv,_(_xt, xr_))<br>**17**<br>Put_{_(_xv, xt_)_, yw,o_3_, yl,o_3_}_into_D_2<br>_oa_<br>**18** _Dpt_ =_Dcm ∪Doa_,_Doa_ =_D_1<br>_oa ∪D_2<br>_oa_<br>**19 foreach**<br>((_xv, xt_)_, yw,o, yl,o_)_∈Dpt_ **do**<br>**20**<br>Compute the losses_Lpt_ following equation 4 and update_π_ref<br>**21** _▷_**_Inference Stage_**<br>**22 foreach**<br>test<br>sample<br>(_xv, xt_) **do**<br>**23**<br>Select top-k retrieved contexts with an assigned domain label_xr ←RF_(_xv_)(_xv_)<br>**24**<br>Get the predictions of the model w/ RAG-PT_p ←M_(_xv,_(_xt, xr_))|**Input:** _D_ =_{x_(_i_)<br>_v , x_(_i_)<br>_t , y_(_i_)_}N_<br>_i_=1: Dataset;_πθ_: Parameters of the Med-LVLM; Med-LVLM:_M_(_·, ·_);<br>Domain Identifcation: _F_(_·_); Retriever: _R_(_·_); Noisy Function: _I_(_·_).<br>**Output:** _π_ref: Parameters of the reference model.<br>**1** _▷_**_Training Stage_**<br>**2** Initialize_Dcm_ with an empty set<br>**3 foreach**<br>(_xv, xt, y_)_∈D_ **do**<br>**4**<br>Generate retrieved contexts with an assigned domain label_xr ←RF_(_xv_)(_xv_)<br>**5**<br>Generate the noisy image_x∗_<br>_v ←I_(_xv_)<br>**6**<br>_▷Cross-Modality Alignment_<br>**7**<br>**if**<br>_M_(_xv,_(_xt, xr_))=_y_<br>and<br>_M_(_x∗_<br>_v,_(_xt, xr_))=_y_ **then**<br>**8**<br>Select the preferred response_yw,o_1 _←y_, dispreferred response_yl,o_1 _←M_(_x∗_<br>_v,_(_xt, xr_))<br>**9**<br>Put_{_(_xv, xt_)_, yw,o_1_, yl,o_1_}_into_Dcm_<br>**10**<br>_▷Overall Alignment_<br>**11**<br>Initialize_D_1<br>_oa_ and_D_2<br>_oa_ with empty set<br>**12**<br>**if**<br>_M_(_xv,_(_xt, xr_))=_y_<br>and<br>_M_(_xv, xt_) =_y_ **then**<br>**13**<br>Select the preferred response_yw,o_2 _←y_, dispreferred response_yl,o_2 _←M_(_xv, xt_)<br>**14**<br>Put_{_(_xv, xt_)_, yw,o_2_, yl,o_2_}_into_D_1<br>_oa_<br>**15**<br>**if**<br>_M_(_xv, xt_)=_y_<br>and<br>_M_(_xv,_(_xt, xr_)) =_y_ **then**<br>**16**<br>Select the preferred response_yw,o_3 _←y_, dispreferred response_yl,o_3 _←M_(_xv,_(_xt, xr_))<br>**17**<br>Put_{_(_xv, xt_)_, yw,o_3_, yl,o_3_}_into_D_2<br>_oa_<br>**18** _Dpt_ =_Dcm ∪Doa_,_Doa_ =_D_1<br>_oa ∪D_2<br>_oa_<br>**19 foreach**<br>((_xv, xt_)_, yw,o, yl,o_)_∈Dpt_ **do**<br>**20**<br>Compute the losses_Lpt_ following equation 4 and update_π_ref<br>**21** _▷_**_Inference Stage_**<br>**22 foreach**<br>test<br>sample<br>(_xv, xt_) **do**<br>**23**<br>Select top-k retrieved contexts with an assigned domain label_xr ←RF_(_xv_)(_xv_)<br>**24**<br>Get the predictions of the model w/ RAG-PT_p ←M_(_xv,_(_xt, xr_))|
|---|---|---|---|
|||Select top-k retrieved contexts with an assigned domain label_xr ←RF_(_xv_)(_xv_)<br>Get the predictions of the model w/ RAG-PT_p ←M_(_xv,_(_xt, xr_))||



where the model makes a correct inference based on an unrelated image, i.e., _M_ ( _x[∗] v[, x][t]_[)][=] _[y]_[, but] _M_ ( _x[∗] v[, x][t]_[+] _[ x][r]_[)][=] _[y]_[,][reflecting][the][model’s][reliance][on][the][retrieved][knowledge.][The][unrelated] images _x[∗] v_[are generated through a two-step process:][first, we use the retriever to select an image] _[ x][′] v_ with the lowest similarity to the target image; then, we introduce diffusion noise into the selected unrelated image. We define the noise step as _s_ , and the noised image at step _s_ is expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0005-04.png)


where _ξ_[¯] _s_ =[�] _[s] i_ =0 _[ξ][i]_[and] _[ξ][s][∈]_[(0] _[,]_[ 1)][is][a][hyperparameter.][The][preference][pairs][constructed][in] this stage are denoted as _Dcm_ . By comparing the preferred and dispreferred responses in _Dcm_ , we encourage the model to prioritize the input medical image when generating responses. **Preference Pairs for Overall Alignment.** Second, we construct preference pairs to improve overall alignment, focusing on enhancing the model’s ability to effectively leverage retrieved knowledge when generating responses. The preference pairs in this stage are constructed from two subsets. The first subset, _Doa_[1][,][is designed to strengthen the model’s comprehension and reasoning abilities] regarding the retrieved knowledge. Preferred responses are selected where the model correctly answers based on both the original image and the retrieved information, i.e., _M_ ( _xv, xt_ + _xr_ ) = _y_ , while dispreferred responses represent cases where the model answers incorrectly based on the image without using retrieval, i.e., _M_ ( _xv, xt_ ) = _y_ . Comparing these preferred and dispreferred responses enhances the model’s understanding of the retrieved information and improves the overall effectiveness of RAG. In the second subset, _Doa_[2][,][the][goal][is][to][mitigate][interference][from][the][re-] trieved knowledge. Preferred responses are selected where the model correctly answers based solely on the original image without using retrieved knowledge, i.e., _M_ ( _xv, xt_ ) = _y_ , while dispreferred responses occur when the model answers incorrectly using both the image and retrieved information, i.e., _M_ ( _xv, xt_ + _xr_ ) = _y_ . This helps the model learn when to rely on its internal knowledge versus retrieved knowledge. Finally, we combine the first and second subsets to form the second set of preference pairs, _Doa_ = _Doa_[1] _[∪D] oa_[2][.] 

Finally, we merge the first and second preference set and denote the preference dataset as _Dpt_ = _Dcm ∪Doa_ = _{x_[(] _[i]_[)] _, yw,o_[(] _[i]_[)] _[, y] l,o_[(] _[i]_[)] _[}] i[N]_ =1[,][where] _[y] w,o_[(] _[i]_[)][,] _[y] l,o_[(] _[i]_[)][are][represented][as][preferred][and][dispreferred] 

5 

Preprint. 

responses, respectively. Based on the curated preferences, we fine-tune Med-LVLM using direct preference optimization (Rafailov et al., 2023) with the following loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0006-02.png)


4 THEORETICAL ANALYSIS 

In this section, we provide a theoretical analysis of the model obtained from equation 4 and examine how the image input and retrieved context influences the model. Recall that _xv, y, xt, xr_ denotes input medical image, groundtruth answer, question, and retrieved information, respectively. 

4.1 THE IMPROVEMENT ON CROSS-MODALITY ALIGNMENT 

We first consider the loss for cross-modality alignment, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0006-07.png)


where ( _xw, yw,o_ ) _∼ qw_ ( _xw, yw,o|xt, xr_ ) and ( _xl, yl,o_ ) _∼ ql_ ( _xl, yl,o|xt, xr_ ) represent distributions of the preferred responses and dispreferred responses on _Dcm_ , respectively. Let _x_ denote ( _xv, xr, xt_ ) 

**Definition 4.1** _Define the weight of xv with respect to_ log _πθ_ ( _y|x_ ) _as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0006-10.png)


Definition 4.1 describes how log _πθ_ ( _y|x_ ) changes with respect to _xv_ , and the weight is always nondispreferred. We demonstrate that this is a reasonable definition through Lemma 4.1. 

**Lemma 4.1** _For linear model y_ = _θ_ 1 _xv_ + _θ_ 2 _xt_ + _ϵ such that ϵ ∼ N_ (0 _,_ 1) _, wt_ ( _xv, πθ_ ) = _θ_ 1[2] **Assumption 4.1** _Let h_ ( _x, y_ ) _, abbreviate as h, be_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0006-13.png)


_Assume that wt_ ( _xv, πo_ ) _< c_[2] _, where_ 

Assumption 4.1 requires that _xv_ has a small weight in log _πo_ ( _y|x_ ). A model _πo_ ( _y|x_ ) independent of _xv_ could satisfy Assumption 4.1. In this case, the reference model generates answers without using information from the image. 

**Theorem 4.1** _Suppose that Assumption 4.1 holds, cross-modality loss increase the weight of xv._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0006-17.png)


Theorem 4.1 indicates that when the weight of _xv_ is too small in the initial model _πo_ ( _y|x_ ), the cross-modality loss function adjusts the model to place greater emphasis on images, informed by the retrieved data. Intuitively, for any sample ( _x, y_ ), generating unrelated images causes the policy to rely less on images. By using samples from this distribution as negative samples, the new model diverges from the initial model, increasing its reliance on images. 

4.2 THE IMPROVEMENT ON OVERALL ALIGNMENT 

In this section, we analyze the improvement on overall alignment. Let _qw_[1][(] _[x][v][, y][w,o][|][x][t][, x][r]_[)][and] _ql_[1][(] _[x][v][, y][l,o][|][x][t]_[)][represent][distributions][of][the][preferred][responses][and][dispreferred][responses][on] _Doa_[1][,][respectively;] _[q] w_[2][(] _[x][v][, y][w,o][|][x][t]_[)][and] _[q] l_[2][(] _[x][v][, y][l,o][|][x][t][, x][r]_[)][represent][distributions][of][the][preferred] responses and dispreferred responses on _Doa_[2][, respectively.][Overall loss is defined by] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0006-21.png)


6 

Preprint. 

Consider _π_ as the generative distribution underlying _M_ , construction of _Doa_[1][and] _[ D] oa_[2][indicate that] whilethere is a significant gap between _x_ ˜ _r_ generate a false one. _π_ ( _y|xv, xt, xr_ ) and _π_ ( _y|xv, xt,_ ˜ _xr_ ) for _xr_ generates true answer 

**Assumption 4.2** _Assume that π_ ( _y|xx, xr, xt_ ) : _x → y is L-lipschitz continuous on xr for all_ ( _xv, xt, y_ ) _such that |π_ ( _y|xv, xt, xr_ ) _− π_ ( _y|xv, xt,_ ˜ _xr_ ) _| ≤ L · dx_ ( _xr,_ ˜ _xr_ ) _, where dx is any distance metric on the text space._ 

Based on Assumption 4.2, _x_ ˜ _r_ can be viewed as being far from the meaningful retrieved information _xr_ , resulting in different weight in the model. Then, we claim in the following theorem that the overall loss in equation 10 can effectively leverage retrieved knowledge while training. 

**Assumption 4.3** _Let h_ 1( _xv, xt, xr, y_ ) _, abbreviate as h_ 1 _, be_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0007-05.png)


_Assume that wt_ ( _xr, πo_ ) _< c_[2] 1 _[and wt]_[(˜] _[x][r][, π][o]_[)] _[ > c]_[2] 2 _[, where]_ 

**Theorem 4.2** _Suppose that Assumption 4.3 holds, then overall loss 10 increase the weight of xr and decrease the weight of x_ ˜ _r._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0007-08.png)


Theorem 4.2 suggests that the model tend to improve the overall alignment. When _x_ ˜ _r_ generates a false answer, the training procedure tends to reduce the reliance on _x_ ˜ _r_ , resulting in a decrease in the weight assigned to _x_ ˜ _r_ . Conversely, if _xr_ is helpful for generating the true answer, _πθ_ ( _y|x_ ) tend to enhance its use of _xr_ . 

## 5 EXPERIMENT 

In this section, we evaluate the performance of MMed-RAG, aiming to answer the following questions: (1) Can MMed-RAG effectively improve the factuality of Med-LVLMs compared to decoding-based and RAG-based baselines? (2) How effective is each proposed component on performance? (3) What is the effect of preference data for different alignment goals? and (4) Does MMed-RAG actually improve cross-modality alignment and overall alignment? 

## 5.1 EXPERIMENTAL SETUPS 

**Implementation Details** . We use LLaVA-Med-1.5 7B (Li et al., 2023a) as the backbone model. During the preference fine-tuning process, we adapt LoRA fine-tuning (Hu et al., 2021). For the training of retriever, the vision encoder is a ResNet-50 (He et al., 2016), and the text encoder is a bio-BioClinicalBERT (Alsentzer et al., 2019). We use the AdamW optimizer with a learning rate of 10 _[−]_[3] , weight decay of 10 _[−]_[2] and a batch size of 32. The model is trained for 360 epochs. For more detailed information on training hyperparameters and training data, please see Appendix A.1.1. 

**Baseline Methods** . We compare MMed-RAG with two types of LVLM hallucination mitigation methods that show promising results in natural image understanding. 1) Decoding-based methods, including Greedy Decoding, Beam Search (Sutskever et al., 2014), DoLa (Chuang et al., 2023), OPERA (Huang et al., 2023), VCD (Leng et al., 2023). These methods manipulate the logits of the model’s output tokens to enhance factual accuracy. 2) Multimodal RAG-based methods, including MedDr (He et al., 2024), FactMM-RAG (Sun et al., 2024), RULE (Xia et al., 2024c). Furthermore, we compare the performance with other open-source Med-LVLMs, including Med-Flamingo (Moor et al., 2023), MedVInT (Zhang et al., 2023b), RadFM (Wu et al., 2023b). 

7 

Preprint. 

Table 1: Model performance (%) of different methods based on LLaVA-Med-1.5 on medical VQA task. Notably, we report the accuracy, F1 score and AUROC. The best results and second best results are highlighted in red and blue , respectively. 

|Models|**Radiology**<br>IU-Xray<br>MIMIC-CXR<br>Acc<br>F1<br>AUC<br>Acc<br>F1<br>AUC|**Radiology**<br>IU-Xray<br>MIMIC-CXR<br>Acc<br>F1<br>AUC<br>Acc<br>F1<br>AUC|**Ophthalmology**<br>Harvard-FairVLMed<br>Acc<br>F1<br>AUC|**Pathology**<br>Quilt-1M<br>PMC-OA (Pathology)<br>Acc<br>F1<br>AUC<br>Acc<br>F1<br>AUC|**Pathology**<br>Quilt-1M<br>PMC-OA (Pathology)<br>Acc<br>F1<br>AUC<br>Acc<br>F1<br>AUC|
|---|---|---|---|---|---|
||IU-Xray|||Quilt-1M||
||Acc<br>F1<br>AUC|||Acc<br>F1<br>AUC||
|LLaVA-Med-1.5|75.47<br>64.04<br>67.46|75.79<br>80.49<br>68.84|63.03<br>74.11<br>63.05|62.80<br>72.90<br>60.03|59.28<br>71.98<br>54.19|
|+ Greedy<br>+ Beam Search<br>+ DoLa<br>+ OPERA<br>+ VCD|76.88<br>65.59<br>68.74<br>76.91<br>66.06<br>68.77<br>78.00<br>66.75<br>72.19<br>70.59<br>61.54<br>63.22<br>68.99<br>54.35<br>61.08|78.32<br>86.75<br>71.13<br>81.56<br>86.36<br>73.79<br>81.35<br>85.73<br>72.73<br>69.34<br>76.66<br>62.46<br>70.89<br>75.57<br>64.61|82.54<br>85.98<br>70.09<br>80.93<br>88.08<br>68.94<br>76.87<br>85.53<br>67.10<br>71.41<br>81.37<br>65.59<br>65.88<br>77.20<br>64.16|64.72<br>70.12<br>58.75<br>63.52<br>69.33<br>57.65<br>63.47<br>69.10<br>57.58<br>60.51<br>66.32<br>54.79<br>61.43<br>67.39<br>55.72|58.61<br>70.42<br>53.10<br>56.29<br>69.84<br>52.89<br>57.71<br>70.27<br>52.95<br>55.32<br>68.30<br>51.86<br>55.10<br>67.94<br>51.62|
|+ MedDr<br>+ FactMM-RAG<br>+ RULE|83.33<br>67.80<br>77.15|55.16<br>56.18<br>58.47|70.17<br>80.72<br>64.15|68.15<br>73.23<br>67.01|59.97<br>69.19<br>57.01<br>60.49<br>69.38<br>57.31|
||84.51<br>68.51<br>77.07|77.58<br>81.86<br>70.09|83.67<br>87.21<br>72.20|69.25<br>73.62<br>68.15||
||87.84<br>78.00<br>85.78|83.92<br>87.49<br>83.44|87.12<br>92.89<br>77.08|68.97<br>73.80<br>68.13|61.41<br>70.36<br>58.91|
|MMed-RAG|89.54<br>80.72<br>87.13|83.57<br>88.49<br>85.08|87.94<br>92.78<br>80.81|72.95<br>76.35<br>72.25|64.54<br>73.09<br>61.42|



Table 2: Model performance (%) of different methods based on LLaVA-Med-1.5 on report generation task. Notably, we report the average BLEU, ROUGE-L, METEOR. 

|Models|**Radiology**<br>IU-Xray<br>MIMIC-CXR<br>BLEU<br>ROUGE-L<br>METEOR<br>BLEU<br>ROUGE-L<br>METEOR|**Radiology**<br>IU-Xray<br>MIMIC-CXR<br>BLEU<br>ROUGE-L<br>METEOR<br>BLEU<br>ROUGE-L<br>METEOR|**Ophthalmology**<br>Harvard-FairVLMed<br>BLEU<br>ROUGE-L<br>METEOR|
|---|---|---|---|
||IU-Xray|||
||BLEU<br>ROUGE-L<br>METEOR|||
|LLaVA-Med-1.5|9.64<br>12.26<br>8.21|12.11<br>13.05<br>11.16|18.11<br>11.36<br>10.75|
|+ Greedy<br>+ Beam Search<br>+ DoLa<br>+ OPERA<br>+ VCD|11.47<br>15.38<br>12.69<br>12.10<br>16.21<br>13.17<br>11.79<br>15.82<br>12.72<br>10.66<br>14.70<br>12.01<br>10.42<br>14.14<br>11.59|16.63<br>14.26<br>14.19<br>16.97<br>14.74<br>14.43<br>17.11<br>14.89<br>14.81<br>15.40<br>12.52<br>13.72<br>15.18<br>12.30<br>13.38|17.98<br>11.49<br>13.77<br>18.37<br>12.62<br>14.50<br>18.26<br>12.51<br>14.51<br>16.59<br>11.47<br>13.63<br>16.73<br>11.38<br>13.89|
|+ MedDr<br>+ FactMM-RAG<br>+ RULE|12.37<br>16.45<br>13.50|18.59<br>15.72<br>16.77<br>18.71<br>15.84<br>16.82|19.82<br>13.72<br>15.40<br>20.82<br>14.17<br>15.31|
||14.70<br>18.05<br>15.92|||
||27.53<br>23.16<br>27.99|18.61<br>15.96<br>17.42|22.35<br>14.93<br>17.74|
|MMed-RAG|31.38<br>25.59<br>32.43|23.25<br>12.34<br>20.47|24.82<br>16.59<br>19.85|



**Evaluation Datasets** . We utilize five medical vision-language datasets for medical VQA and report generation tasks, i.e., MIMIC-CXR (Johnson et al., 2019), IU-Xray (Demner-Fushman et al., 2016), Harvard-FairVLMed (Luo et al., 2024), PMC-OA (Lin et al., 2023a) (we only select the pathology part) and Quilt-1M (Ikezogwo et al., 2024). These datasets cover radiology, ophthalmology, and pathology. To construct the VQA benchmarks, following (Xia et al., 2024a), we generate questionanswer pairs from medical reports using GPT-4 (OpenAI, 2023), with answers formatted as _yes_ or _no_ . Pathology images are excluded from the report generation task due to their brief and insufficient descriptions. The detailed dataset descriptions are provided in the Appendix A.2. 

**Evaluation Metrics** . Following (Jing et al., 2017; Lin et al., 2023b), we use Accuracy, F1 Score and AUROC for evaluating medical VQA task, and BLEU Score (Papineni et al., 2002), ROUGE-L (Lin, 2004) and METEOR (Banerjee & Lavie, 2005) for evaluating report generation task. 

## 5.2 MAIN RESULTS 

In this section, we provide a comprehensive comparison with various baseline methods and other open-source Med-LVLMs on medical VQA and report generation tasks. 

**Comparison with Baselines.** We compare MMed-RAG with baseline methods on medical VQA and report generation tasks, with the results presented in Table 1 and Table 2, respectively. Overall, MMed-RAG outperforms all baselines across nearly all metrics and datasets. Specifically, MMedRAG demonstrates a significant performance boost, improving by 18.5% and 69.1% over the original Med-LVLM in medical VQA and report generation tasks, respectively. When compared to baseline methods, MMed-RAG surpasses decoding-based approaches, achieving improvements of 11.5% and 44.2% in the two tasks. Furthermore, recent RAG-based methods show substantial improvements over earlier techniques, yet our approach still outperforms RAG-based baselines by 2.8% and 16.1% in the medical VQA and report generation tasks, respectively. This indicates that MMed-RAG effectively mitigates misalignment issues introduced by RAG. Notably, MMed-RAG 

8 

Preprint. 

achieves more pronounced gains in report generation, likely due to the higher complexity of the task and the greater influence of retrieved contexts in guiding open-ended generation. 

**Comparison with Other Med-LVLMs.** To provide a comprehensive comparison, we evaluate MMed-RAG against other open-source Med-LVLMs to demonstrate the superiority of our approach. We assess the performance of these models across different medical image modalities, reporting the average results for medical VQA and report generation tasks in Table 3 (see Appendix A.6 for detailed results). Our findings show that MMed-RAG significantly outperforms Med-LVLMs pre-trained on large-scale datasets across various domains. This reinforces the generalizability and effectiveness of our approach across diverse image domains and medical multimodal tasks. 

Table 3: Performance comparison with several Med-LVLMs. Rad: Radiology, Opt: Ophthalomology, Pat: Pathology. 

|Model|Rad<br>Opt<br>Pat|
|---|---|
|||
|Med-Flamingo<br>MedVInT<br>RadFM<br>miniGPT-Med<br>MMed-RAG|27.42<br>22.50<br>29.11<br>33.17<br>29.40<br>25.33<br>35.82<br>27.07<br>24.82<br>36.66<br>25.28<br>23.16<br>**56.94**<br>**56.38**<br>**54.10**|



## 5.3 ANALYSIS 

In this section, we provide a detailed analysis of each module’s performance, along with a series of analytical experiments, to better understand the performance gains of MMed-RAG. Additionally, we demonstrate the compatibility of our approach, achieving a consistent 40.3% performance improvement on a different backbone, i.e., LLAVA-Med-1.0 (see details in Appendix A.6). 

**Ablation Studies.** We conduct a series of ablation experiments to evaluate the impact of each component in MMed-RAG. The results for both medical VQA and report generation tasks on the IU-Xray and Harvard-FairVLMed datasets are summarized in Table 4. According to the results, we can see that: (1) The domain-aware retrieval mechanism (DR) significantly improves the factuality of Med-LVLM, with an average performance increase of 17.9% and 16.1% on the IU-Xray and FairVLMed datasets, respectively. Here, the retrieved knowledge aids the 

Table 4: Ablation results on two datasets covering different domains. RG: report generation, FairVLMed: Harvard-FairVLMed. 

|Model|IU-Xray<br>FairVLMed<br>VQA<br>RG<br>VQA<br>RG|
|---|---|
|||
|LLaVA-Med-1.5<br>+DR<br>+RCS<br>+RAG-PT (Ours)|68.99<br>10.04<br>66.63<br>13.41<br>77.12<br>13.23<br>72.69<br>15.89<br>79.56<br>17.92<br>75.74<br>17.22<br>**85.80**<br>**29.80**<br>**87.18**<br>**20.42**|



model in generating more factual responses. (2) Building on this, the introduction of adaptive retrieval context selection (RCS) further filters out unreliable retrieved contexts, yielding an additional performance boost of 19.3% and 6.3% on the IU-Xray and FairVLMed datasets. (3) The inclusion of RAG-based preference fine-tuning (RAG-PT) enhances the model’s understanding of the retrieved knowledge, leading to substantial performance gains of 37.1% and 16.9% on the respective datasets. This demonstrates that RAG-PT effectively addresses misalignment issues. 

**Impact of the Preference Data in RAG-PT.** To better understand how RAG-PT mitigates the misalignment issue and improves performance, we conducted a detailed study on the training preference data composition of RAG-PT. As described in Section 3.3, the RAG-PT data is designed to address both cross-modality alignment and overall alignment objectives, with the latter focusing on enhanced understanding of retrieved knowledge and 

Table 5: Performance using RAG-PT based on subsets of preference data. 

|Model|IU-Xray<br>FairVLMed<br>VQA<br>RG<br>VQA<br>RG|
|---|---|
|||
|LLaVA-Med-1.5<br>+RAG-PT 1<br>+RAG-PT 2<br>+RAG-PT 3|68.99<br>10.04<br>66.63<br>13.41<br>80.19<br>19.38<br>79.42<br>18.37<br>80.27<br>**20.16**<br>79.35<br>18.66<br>**81.30**<br>19.43<br>**80.07**<br>**18.92**|



minimizing retrieval interference. The detailed experimental results in Table 5 demonstrate that the preference data tailored for different alignment objectives positively impacts the model’s performance, showing the effectiveness of RAG-PT. 

**How Effective is MMed-RAG in Mitigating Misalignment Issues?** To gain a more intuitive understanding of the effectiveness of MMed-RAG in addressing misalignment issues: 1) we calculate the proportion of errors caused by RAG and compare it to the proportion after incorporating MMedRAG. 2) We visualize the attention maps of image and text tokens with and without RAG-PT. First, as mentioned in Section 3.3, the model may directly copy reference information, referred to as Copy-Reference (CR) rate. After applying MMed-RAG, as shown in Figure 3, the CR rate drops to 28.19%. Additionally, the proportion of errors affected by RAG interference, referred to as OverReliance (OR) rate, which is initially 43.31%, decreased to 8.38% after incorporating MMed-RAG. 

9 

Preprint. 

Furthermore, as shown in Figure 4, the original Med-LVLM tends to rely more heavily on text while ignoring visual information. When retrieval information is introduced, the original Med-LVLM focused more on the retrieved answers, even if the content is incorrect. After RAG-PT, the model significantly increases its attention to visual information and reduces the interference of RAG, thus better aligning the model’s knowledge with the fundamental facts. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0010-02.png)


Figure 3: Alignment analysis with and without RAG. OR: Over-Reliance; CR: CopyReference. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0010-04.png)


**----- Start of picture text -----**<br>
Image Tokens Text Tokens<br>Can focal airspace<br>Question consolidation be<br>seen on the image?<br>Medical Image Yes, there seems to<br>Reference Med-1.5LLaVA- be a focal airspace consolidation.<br>The heart is normal in<br>size…There appears to  No, focal airspace<br>be a focal airspace  consolidation cannot<br>right side of the lung…consolidation on the  Ours be seen on the image.<br>LLaVA-Med-1.5<br>) Ours<br>MMed-RAG (<br>**----- End of picture text -----**<br>


Figure 4: Visualization of attention map. The red box region is labeled with the attentions that can be enhanced by MMed-RAG. 

## 6 RELATED WORK 

**Factuality in Med-LVLMs.** The rapid advancements in Large Vision-Language Models (LVLMs) (Liu et al., 2024a;b; Xia et al., 2023; Zhou et al., 2023; 2024; Chen et al., 2024d) are beginning to influence the field of medical image analysis. Several Med-LVLMs (Li et al., 2023a; Moor et al., 2023; Zhang et al., 2023b; Wu et al., 2023b), have emerged, showing remarkable performance across different medical imaging modalities. Despite these advances, Med-LVLMs continue to present notable factual hallucination (Xia et al., 2024a; Royer et al., 2024), generating textual outputs that contradict medical visual information. This raises concerns about potential misdiagnoses or overlooked conditions. Recently, benchmarks have been developed to assess the accuracy of MedLVLMs in tasks such as visual question answering (VQA) and report generation (Xia et al., 2024a; Royer et al., 2024). However, research aimed at enhancing the factual accuracy of Med-LVLMs remains relatively unexplored. 

**Retrieval Augmented Generation in Med-LVLMs.** Retrieval-Augmented Generation (RAG) has proven to be a powerful technique for enhancing factual accuracy in language modeling (Gao et al., 2023; Wu et al., 2023c; Chen et al., 2024c; Qu et al., 2024). In the biomedical domain, RAG leverages external knowledge to guide the generation of Med-LVLMs, offering clear advantages in tasks such as medical VQA and report generation (Yuan et al., 2023; Kumar & Marttinen, 2024; Tao et al., 2024; He et al., 2024; Sun et al., 2024). However, these works mainly focus on enhancing the relevance of the retrieved contexts without considering the model’s understanding of retrieved knowledge. Recently, RULE (Xia et al., 2024c) is proposed to use preference fine-tuning to reduce the model’s over-reliance on retrieved contexts. However, it still overlooks misalignment issues caused by RAG, as well as the generalizability of the retriever given the diverse domains of input images. In response, we propose MMed-RAG to mitigate these risks, enhancing the factuality of Med-LVLMs by addressing these overlooked factors. This can lead to a better cross-modality and overall alignment to enhance the understanding of retrieved knowledge and visual information, ensuring more consistent and reliable performance across tasks. 

## 7 CONCLUSION 

This paper introduces MMed-RAG, a versatile multimodal RAG system designed to address the critical issue of factual hallucination in Med-LVLMs. MMed-RAG employs a domain-aware retrieval mechanism, adaptive calibration for selecting the optimal number of retrieved contexts, and RAG-based preference fine-tuning to improve both cross-modality alignment and overall alignment with the ground truth. These enhancements significantly boost the factual accuracy of Med-LVLMs. Experimental results demonstrate MMed-RAG’ effectiveness in enhancing factual accuracy across various imaging domains, including radiology, ophthalmology, pathology. Our findings underscore the importance of incorporating robust multimodal RAG mechanism to ensure that Med-LVLMs can serve as dependable tools in clinical settings. 

10 

Preprint. 

## ACKNOWLEDGEMENT 

This work is partially supported by Cisco Faculty Research Award. 

## REFERENCES 

- Asma Alkhaldi, Raneem Alnajim, Layan Alabdullatef, Rawan Alyahya, Jun Chen, Deyao Zhu, Ahmed Alsinan, and Mohamed Elhoseiny. Minigpt-med: Large language model as a general interface for radiology diagnosis. arXiv preprint arXiv:2407.04106, 2024. 

- Emily Alsentzer, John R Murphy, Willie Boag, Wei-Hung Weng, Di Jin, Tristan Naumann, and Matthew McDermott. Publicly available clinical bert embeddings. arXiv preprint arXiv:1904.03323, 2019. 

- Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pp. 65–72, 2005. 

- Jiawei Chen, Dingkang Yang, Tong Wu, Yue Jiang, Xiaolu Hou, Mingcheng Li, Shunli Wang, Dongling Xiao, Ke Li, and Lihua Zhang. Detecting and evaluating medical hallucinations in large vision language models. arXiv preprint arXiv:2406.10185, 2024a. 

- Junying Chen, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Hardy Chen, Xidong Wang, Ruifei Zhang, Zhenyang Cai, Ke Ji, Guangjun Yu, et al. Huatuogpt-vision, towards injecting medical visual knowledge into multimodal llms at scale. arXiv preprint arXiv:2406.19280, 2024b. 

- Zhanpeng Chen, Chengjin Xu, Yiyan Qi, and Jian Guo. Mllm is a strong reranker: Advancing multimodal retrieval-augmented generation via knowledge-enhanced reranking and noise-injected training. arXiv preprint arXiv:2407.21439, 2024c. 

- Zhaorun Chen, Yichao Du, Zichen Wen, Yiyang Zhou, Chenhang Cui, Zhenzhen Weng, Haoqin Tu, Chaoqi Wang, Zhengwei Tong, Qinglan Huang, et al. Mj-bench: Is your multimodal reward model really a good judge for text-to-image generation? arXiv preprint arXiv:2407.04842, 2024d. 

- Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models, 2024e. URL https://arxiv.org/ abs/2401.01335. 

- Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models. arXiv preprint arXiv:2309.03883, 2023. 

- Dina Demner-Fushman, Marc D Kohli, Marc B Rosenman, Sonya E Shooshan, Laritza Rodriguez, Sameer Antani, George R Thoma, and Clement J McDonald. Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association, 23(2):304–310, 2016. 

- Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016. 

- Sunan He, Yuxiang Nie, Zhixuan Chen, Zhiyuan Cai, Hongmei Wang, Shu Yang, and Hao Chen. Meddr: Diagnosis-guided bootstrapping for large-scale medical vision-language learning. arXiv preprint arXiv:2404.15127, 2024. 

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 

11 

Preprint. 

Ming Hu, Lin Wang, Siyuan Yan, Don Ma, Qingli Ren, Peng Xia, Wei Feng, Peibo Duan, Lie Ju, and Zongyuan Ge. Nurvid: A large expert-level video database for nursing procedure activity understanding. Advances in Neural Information Processing Systems, 36, 2024a. 

Ming Hu, Peng Xia, Lin Wang, Siyuan Yan, Feilong Tang, Zhongxing Xu, Yimin Luo, Kaimin Song, Jurgen Leitner, Xuelian Cheng, et al. Ophnet: A large-scale video benchmark for ophthalmic surgical workflow understanding. arXiv preprint arXiv:2406.07471, 2024b. 

Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. arXiv preprint arXiv:2311.17911, 2023. 

- Wisdom Ikezogwo, Saygin Seyfioglu, Fatemeh Ghezloo, Dylan Geva, Fatwir Sheikh Mohammed, Pavan Kumar Anand, Ranjay Krishna, and Linda Shapiro. Quilt-1m: One million image-text pairs for histopathology. Advances in neural information processing systems, 36, 2024. 

Yue Jiang, Jiawei Chen, Dingkang Yang, Mingcheng Li, Shunli Wang, Tong Wu, Ke Li, and Lihua Zhang. Medthink: Inducing medical large-scale visual language models to hallucinate less by thinking more. arXiv preprint arXiv:2406.11451, 2024. 

Baoyu Jing, Pengtao Xie, and Eric Xing. On the automatic generation of medical imaging reports. arXiv preprint arXiv:1711.08195, 2017. 

- Alistair EW Johnson, Tom J Pollard, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Yifan Peng, Zhiyong Lu, Roger G Mark, Seth J Berkowitz, and Steven Horng. Mimic-cxr-jpg, a large publicly available database of labeled chest radiographs. arXiv preprint arXiv:1901.07042, 2019. 

- Yogesh Kumar and Pekka Marttinen. Improving medical multi-modal contrastive learning with expert annotations. arXiv preprint arXiv:2403.10153, 2024. 

- Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. arXiv preprint arXiv:2311.16922, 2023. 

- Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023a. 

- Wenxue Li, Xinyu Xiong, Peng Xia, Lie Ju, and Zongyuan Ge. Tp-drseg: Improving diabetic retinopathy lesion segmentation with explicit text-prompts assisted sam. arXiv preprint arXiv:2406.15764, 2024. 

- Yingshu Li, Yunyi Liu, Zhanyu Wang, Xinyu Liang, Lingqiao Liu, Lei Wang, Leyang Cui, Zhaopeng Tu, Longyue Wang, and Luping Zhou. A comprehensive study of gpt-4v’s multimodal capabilities in medical imaging. arXiv preprint arXiv:2310.20381, 2023b. 

- Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pp. 74–81, 2004. 

- Weixiong Lin, Ziheng Zhao, Xiaoman Zhang, Chaoyi Wu, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmc-clip: Contrastive language-image pre-training using biomedical documents. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 525–536. Springer, 2023a. 

- Zhihong Lin, Donghao Zhang, Qingyi Tao, Danli Shi, Gholamreza Haffari, Qi Wu, Mingguang He, and Zongyuan Ge. Medical visual question answering: A survey. Artifcial Intelligence in Medicine, 143:102611, 2023b. 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26296–26306, 2024a. 

12 

Preprint. 

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024b. 

- Yan Luo, Min Shi, Muhammad Osama Khan, Muhammad Muneeb Afzal, Hao Huang, Shuaihang Yuan, Yu Tian, Luo Song, Ava Kouhana, Tobias Elze, et al. Fairclip: Harnessing fairness in vision-language learning. arXiv preprint arXiv:2403.19949, 2024. 

- Michael Moor, Qian Huang, Shirley Wu, Michihiro Yasunaga, Yash Dalmia, Jure Leskovec, Cyril Zakka, Eduardo Pontes Reis, and Pranav Rajpurkar. Med-flamingo: a multimodal medical fewshot learner. In Machine Learning for Health (ML4H), pp. 353–367. PMLR, 2023. 

OpenAI. Gpt-4 technical report, 2023. https://arxiv.org/abs/2303.08774. 

- Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311–318, 2002. 

- Xiaoye Qu, Qiyuan Chen, Wei Wei, Jishuo Sun, and Jianfeng Dong. Alleviating hallucination in large vision-language models with active retrieval augmentation. arXiv preprint arXiv:2408.00555, 2024. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 

- Corentin Royer, Bjoern Menze, and Anjany Sekuboyina. Multimedeval: A benchmark and a toolkit for evaluating medical vision-language models. arXiv preprint arXiv:2402.09262, 2024. 

- Jessica Schrouff, Natalie Harris, Sanmi Koyejo, Ibrahim M Alabdulmohsin, Eva Schnider, Krista Opsahl-Ong, Alexander Brown, Subhrajit Roy, Diana Mincu, Christina Chen, et al. Diagnosing failures of fairness transfer across distribution shift in real-world medical settings. Advances in Neural Information Processing Systems, 35:19304–19318, 2022. 

- Congzhen Shi, Ryan Rezai, Jiaxi Yang, Qi Dou, and Xiaoxiao Li. A survey on trustworthiness in foundation models for medical image analysis. arXiv preprint arXiv:2407.15851, 2024. 

- Liwen Sun, James Zhao, Megan Han, and Chenyan Xiong. Fact-aware multimodal retrieval augmentation for accurate medical radiology report generation. arXiv preprint arXiv:2407.15268, 2024. 

- Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pp. 3104–3112, 2014. 

- Yitian Tao, Liyan Ma, Jing Yu, and Han Zhang. Memory-based cross-modal semantic alignment network for radiology report generation. IEEE Journal of Biomedical and Health Informatics, 2024. 

- Alexandra-Maria T˘aut¸an, Bogdan Ionescu, and Emiliano Santarnecchi. Artificial intelligence in neurodegenerative diseases: A review of available tools with a focus on machine learning techniques. Artifcial Intelligence in Medicine, 117:102081, 2021. 

- Omkar Thawkar, Abdelrahman Shaker, Sahal Shaji Mullappilly, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, Jorma Laaksonen, and Fahad Shahbaz Khan. Xraygpt: Chest radiographs summarization using medical vision-language models. arXiv preprint arXiv:2306.07971, 2023. 

- Robert Tibshirani, Guenther Walther, and Trevor Hastie. Estimating the number of clusters in a data set via the gap statistic. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63(2):411–423, 2001. 

13 

Preprint. 

- Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang, Andrew Carroll, Charles Lau, Ryutaro Tanno, Ira Ktena, et al. Towards generalist biomedical ai. NEJM AI, 1(3):AIoa2300138, 2024. 

- Chunhao Wang, Xiaofeng Zhu, Julian C Hong, and Dandan Zheng. Artificial intelligence in radiotherapy treatment planning: present and future. Technology in cancer research & treatment, 18: 1533033819873922, 2019. 

- Chaoyi Wu, Jiayu Lei, Qiaoyu Zheng, Weike Zhao, Weixiong Lin, Xiaoman Zhang, Xiao Zhou, Ziheng Zhao, Ya Zhang, Yanfeng Wang, et al. Can gpt-4v (ision) serve medical applications? case studies on gpt-4v for multimodal medical diagnosis. arXiv preprint arXiv:2310.09909, 2023a. 

- Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Yanfeng Wang, and Weidi Xie. Towards generalist foundation model for radiology. arXiv preprint arXiv:2308.02463, 2023b. 

- Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Cheng Niu, Randy Zhong, Juntong Song, and Tong Zhang. Ragtruth: A hallucination corpus for developing trustworthy retrieval-augmented language models. arXiv preprint arXiv:2401.00396, 2023c. 

- Peng Xia, Xingtong Yu, Ming Hu, Lie Ju, Zhiyong Wang, Peibo Duan, and Zongyuan Ge. Hgclip: Exploring vision-language models with graph representations for hierarchical understanding. arXiv preprint arXiv:2311.14064, 2023. 

- Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu Zhu, et al. Cares: A comprehensive benchmark of trustworthiness in medical vision language models. arXiv preprint arXiv:2406.06007, 2024a. 

- Peng Xia, Ming Hu, Feilong Tang, Wenxue Li, Wenhao Zheng, Lie Ju, Peibo Duan, Huaxiu Yao, and Zongyuan Ge. Generalizing to unseen domains in diabetic retinopathy with disentangled representations. In arXiv preprint arXiv:2406.06384, 2024b. 

- Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang, and Huaxiu Yao. Rule: Reliable multimodal rag for factuality in medical vision language models. arXiv preprint arXiv:2407.05131, 2024c. 

- Qing Ye, Chang-Yu Hsieh, Ziyi Yang, Yu Kang, Jiming Chen, Dongsheng Cao, Shibo He, and Tingjun Hou. A unified drug–target interaction prediction framework based on knowledge graph and recommendation system. Nature communications, 12(1):6775, 2021. 

- Zheng Yuan, Qiao Jin, Chuanqi Tan, Zhengyun Zhao, Hongyi Yuan, Fei Huang, and Songfang Huang. Ramm: Retrieval-augmented biomedical visual question answering with multi-modal pre-training. In Proceedings of the 31st ACM International Conference on Multimedia, pp. 547– 556, 2023. 

- Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al. Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv preprint arXiv:2303.00915, 2023a. 

- Xiaoman Zhang, Chaoyi Wu, Ziheng Zhao, Weixiong Lin, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmc-vqa: Visual instruction tuning for medical visual question answering. arXiv preprint arXiv:2305.10415, 2023b. 

- Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao. Analyzing and mitigating object hallucination in large vision-language models. arXiv preprint arXiv:2310.00754, 2023. 

- Yiyang Zhou, Zhiyuan Fan, Dongjie Cheng, Sihan Yang, Zhaorun Chen, Chenhang Cui, Xiyao Wang, Yun Li, Linjun Zhang, and Huaxiu Yao. Calibrated self-rewarding vision language models. arXiv preprint arXiv:2405.14622, 2024. 

14 

Preprint. 

## A EXPERIMENT 

## A.1 EXPERIMENTAL SETUP 

## A.1.1 DATA STATISTICS 

The data quantities used in this study are presented in Table 6, Table 7 and Table 8. We clarify that for training the retriever, the data refers to the number of image-text pairs, while for fine-tuning, it refers to the number of QA items. The “All” category represents the total amount of data used to construct the preference dataset for RAG-PT. The training of RAG-PT includes three types of samples: (a) clean samples with originally correct answers that remain correct even after adding noise to the images, (b) clean image samples with originally incorrect answers that become correct, and (c) clean image samples with originally correct answers that become incorrect. 

Table 6: Data statistics for medical VQA task. ”Train (DR)” refers to the number of image-text pairs for retriever training, ”All (RAG-PT)” refers to the total data for RAG-PT, and ”Train (RAG-PT)a/b/c” refer to the respective subsets for RAG-PT training. 

|Dataset|Train (DR)<br>All (RAG-PT)<br>Train (RAG-PT)-a<br>Train (RAG-PT)-b<br>Train (RAG-PT)-c|
|---|---|
|Ophthalomology<br>Radiology<br>Pathology|7000<br>3247<br>1082<br>1030<br>1135<br>4034<br>4836<br>1612<br>1989<br>1235<br>5000<br>1990<br>663<br>523<br>804|



Table 7: Data statistics for report generation. ”Train (DR)” refers to the number of image-text pairs for retriever training, ”All (RAG-PT)” refers to the total data for RAG-PT, and ”Train (RAG-PT)a/b/c” refer to the respective sample categories for RAG-PT training. 

|Dataset|Train (R)<br>All (RAG-PT)<br>Train (RAG-PT)-a<br>Train (RAG-PT)-b<br>Train (RAG-PT)-c|
|---|---|
|||
|Ophthalmology<br>Radiology|7000<br>3247<br>142<br>78<br>207<br>4034<br>4836<br>233<br>126<br>342|



Table 8: Data statistics for various datasets. The rows represent the number of images and QA pairs for each dataset. 

|h dataset.||
|---|---|
||Harvard-FairVLMed<br>IU-Xray<br>MIMIC-CXR<br>PMC-OA<br>Quilt-1M|
|||
|# Images<br># QA Items|713<br>589<br>700<br>530<br>559<br>4285<br>2573<br>3470<br>3124<br>1994|



## A.1.2 HYPERPARAMETER SETTINGS 

Following the settings of CLIP (Radford et al., 2021), we adopt the same architecture and hyperparameters for the vision and text encoders. The vision encoder is a ResNet-50 (He et al., 2016), and the text encoder is a bio-bert-based model (Alsentzer et al., 2019). We use the AdamW optimizer with a learning rate of 10 _[−]_[4] and a batch size of 512. The model is trained for 360 epochs. For the first phase, we trained for 3 epochs, and for the second phase, the training was conducted for 1 epoch. Training for 20 hours on one A100 80G GPU. For the RAG-PT phase, we adjust the diffusion noise level, symbolized by _ξ_ through a specific formula: _ξ_ = Sigmoid( _lt_ ) _×_ (0 _._ 5 _×_ 10 _[−]_[2] _−_ 10 _[−]_[5] ) + 10 _[−]_[5] _,_ where _ϵ_ is drawn from a normal distribution. The reports available for retrieval are from the training set of the corresponding dataset. In our experiments, we apply cross-validation to tune all hyperparameters with grid search. All the experiments are implemented on PyTorch 2.1.2 using four NVIDIA RTX A6000 GPUs. It takes roughly 3 and 4 hours for fine-tuning CLIP and LLaVA-Med1.5 7B, respectively. 

## A.2 EVALUATED DATASETS 

We utilize five open-source medical vision-language datasets, i.e., MIMIC-CXR (Johnson et al., 2019), IU-Xray (Demner-Fushman et al., 2016), Harvard-FairVLMed (Luo et al., 2024), PMCOA (Lin et al., 2023a) and Quilt-1M (Ikezogwo et al., 2024). 

15 

Preprint. 

- **MIMIC-CXR** (Johnson et al., 2019) is a large publicly available dataset of chest X-ray images in DICOM format with associated radiology reports. 

- **IU-Xray** (Demner-Fushman et al., 2016) is a dataset that includes chest X-ray images and corresponding diagnostic reports. 

- **Harvard-FairVLMed** (Luo et al., 2024) focuses on fairness in multimodal fundus images, containing image and text data from various sources. It aims to evaluate bias in AI models on this multimodal data comprising different demographics. 

- **PMC-OA** (Lin et al., 2023a) is a large-scale dataset comprising figure-caption pairs extracted from PubMed Central. It covers 2,478,267 papers and includes a total of 12,211,907 figure-caption pairs. We only use the pathology subset filtered by GPT-4 based on the captions. 

- **Quilt-1M** (Ikezogwo et al., 2024) is the largest vision-language dataset in histopathology, containing 1 million image-text pairs sourced from platforms such as YouTube, Twitter, research papers, and other parts of the internet. 

- A.3 EVALUATED MODELS 

We evaluate five open-source Med-LVLMs, _i.e._ , LLaVA-Med (Li et al., 2023a), MedFlamingo (Moor et al., 2023), MedVInT (Zhang et al., 2023b), RadFM (Wu et al., 2023b), miniGPTMed (Alkhaldi et al., 2024). The selected models are all at the 7B level. 

- **LLaVA-Med** (Li et al., 2023a) is a vision-language conversational assistant, adapting the generaldomain LLaVA (Liu et al., 2024b) model for the biomedical field. The model is fine-tuned using a novel curriculum learning method, which includes two stages: aligning biomedical vocabulary with figure-caption pairs and mastering open-ended conversational semantics. It demonstrates excellent multimodal conversational capabilities. 

- **Med-Flamingo** (Moor et al., 2023) is a multimodal few-shot learner designed for the medical domain. It builds upon the OpenFlamingo, continuing pre-training with medical image-text data from publications and textbooks. This model aims to facilitate few-shot generative medical visual question answering, enhancing clinical applications by generating relevant responses and rationales from minimal data inputs. 

- **RadFM** (Wu et al., 2023b) serve as a versatile generalist model in radiology, distinguished by its capability to adeptly process both 2D and 3D medical scans for a wide array of clinical tasks. It integrates ViT as visual encoder and a perceiver module, alongside the MedLLaMA language model, to generate sophisticated medical insights for a variety of tasks. This design allows RadFM to not just recognize images but also to understand and generate human-like explanations. 

- **MedVInT** (Zhang et al., 2023b), which stands for Medical Visual Instruction Tuning, is designed to interpret medical images by answering clinically relevant questions. This model features two variants to align visual and language understanding: MedVInT-TE and MedVInT-TD. Both MedVInT variants connect a pre-trained vision encoder ResNet-50 adopted from PMC-CLIP (Lin et al., 2023a), which processes visual information from images. It is an advanced model that leverages a novel approach to align visual and language understanding. 

- **miniGPT-Med** (Alkhaldi et al., 2024) is a vision-language model derived from large-scale language models and tailored for radiology diagnosis applications. It handles various medical visionlanguage task using distinct task identifiers, demonstrating advanced performance in disease grounding, medical report generation, and medical VQA. 

## A.4 OVERVIEW OF THE BASELINES 

We compare MMed-RAG with two types of LVLM hallucination mitigation methods that show promising results in natural image understanding. 1) Decoding-based methods, including Greedy Decoding, Beam Search (Sutskever et al., 2014), DoLa (Chuang et al., 2023), OPERA (Huang et al., 2023), VCD (Leng et al., 2023). These methods manipulate the logits of the model’s output tokens to enhance factual accuracy. 2) Multimodal RAG-based methods, including MedDr (He et al., 2024), FactMM-RAG (Sun et al., 2024), RULE (Xia et al., 2024c). 

16 

Preprint. 

**Instruction [Round1]** You are a professional medical expert. I will provide you with some medical reports. Please generate some questions with answers (the answer should be yes or no) based on the provided report. The subject of the questions should be the medical image or patient, not the report. Below are the given report: [REPORT] **Instruction [Round2]** Please double-check the questions and answers, including how the questions are asked and whether the answers are correct. You should only generate the questions with answers and no other unnecessary information. Below are the given report and QA pairs in round1: [REPORT] [QA PAIRS R1] 

Table 9: The instruction to GPT-4 for generating QA pairs. 

- **Greedy decoding** involves selecting the most probable next token at each step of generation. While it is efficient and straightforward, it can lead to suboptimal outcomes by getting stuck in repetitive or less creative patterns. 

- **Beam search** (Sutskever et al., 2014) expands on greedy decoding by maintaining multiple candidate sequences (or ”beams”) at each step, allowing for a broader exploration of possible outputs. This approach balances quality and diversity by selecting the top-k sequences based on their probabilities, resulting in more coherent and creative text generation compared to greedy decoding. 

- **DoLa** (Chuang et al., 2023) derives the next-token distribution by contrasting the logits projected from later layers against those from earlier layers, leveraging the fact that factual knowledge in LLMs is typically localized within specific transformer layers. 

- **OPERA** (Huang et al., 2023) is a LVLMs decoding method based on an Over-trust Penalty and a Retrospection-Allocation strategy The key insight is that hallucinations are closely tied to knowledge aggregation patterns in the self-attention matrix, where MLLMs tend to focus on summary tokens, neglecting image tokens and resulting in content hallucination. 

- **VCD** (Leng et al., 2023) is a decoding method that tackles the object hallucination issue in LVLMs. It contrasts output distributions derived from original and distorted visual inputs to calibrate the model’s output without the usage of external tools, reducing the the over-reliance on statistical bias and unimodal priors. 

- **MedDr** (He et al., 2024) is a medical foundation model built upon generated diagnosis-based datasets, demonstrating advanced capabilities in various data modalities. Meddr also integrates a retrieval-augmented medical diagnosis strategy during inferencing to enhance factual accuracy. 

- **FactMM-RAG** (Sun et al., 2024) is a fact-aware multimodal retrieval-augmented pipeline for radiology report generation. It utilize RadGraph to annotate chest radiograph reports and mine clinically relevant pairs to train a universal multimodal retriever. 

- **RULE** (Xia et al., 2024c) is an advanced medical retrieval-augmented generation strategy designed to enhance the factuality of Med-LVLMs. First, it introduces a robust strategy for controlling factuality risk through the calibrated selection of retrieved contexts. Second, RULE develops a preference optimization strategy to balance Med-LVLMs’ intrinsic knowledge and the retrieved information. 

## A.5 PROMPTS 

We convert the medical reports into a series of closed-ended questions with _yes_ or _no_ answers. To ensure the quality of the VQA data, we perform a round of self-checks using GPT-4 (OpenAI, 2023). Finally, we conduct an round of manual filtering to remove questions with obvious issues or those related to multiple images or patient histories. The prompt templates used are shown in Table 9. 

## A.6 ADDITIONAL RESULTS 

**Generalization on Different Backbones.** To demonstrate the compatibility of our approach across different backbone models, we apply it to LLaVA-Med-1.0. As shown in Table 10, our method 

17 

Preprint. 

delivers an average improvement of 40.3% over the original LLaVA-Med-1.0, further highlighting its effectiveness in enhancing RAG performance and its adaptability to various backbones. MMedRAG can be transferred to different Med-LVLMs, yielding consistent improvements across various domains, demonstrating the compatibility of our method. 

Table 10: Performance on different backbones. 

|Model|IU-Xray<br>FairVLMed<br>VQA<br>RG<br>VQA<br>RG|
|---|---|
|||
|LLaVA-Med-1.0<br>+MMed-RAG|61.73<br>8.74<br>59.54<br>10.59<br>**80.32**<br>**22.63**<br>**78.49**<br>**15.88**|



**Detailed Results of Other Med-LVLMs.** As shown in Table 11, we illustrate the detailed performance simply shown in Table 3. 

## B PROOFS FOR THEORETICAL RESULTS IN SECTION 4 

Here we provide proofs for the results in Section 4. 

## B.1 NOTATIONS 

Let _xv, y, xt, xr_ be input medical image, ground-truth answer, question, and retrieved information, respectively. Denote ( _xw, yw,o_ ) _∼ qp_ ( _xw, yw,o|xt, xr_ ) and ( _xl, yl,o_ ) _∼ ql_ ( _xl, yl,o|xt, xr_ ) as distributions of the preferred responses and dispreferred responses. Let _x_ denote ( _xv, xr, xt_ ). We aim to a fine-tune a generative model _πθ_ ( _y|x, xt_ ) through DPO loss (Rafailov et al., 2023): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0018-09.png)


where _U_ ( _t_ ) = log(1 + exp( _−t_ )). Define the weight of _xv_ with respect to log _πθ_ ( _y|x_ ) as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0018-11.png)


## B.2 ASSUMPTIONS 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0018-13.png)


Assumption B.1 requires that the parameter space sufficiently large to ensure that _πθ_ can achieve its global optimum, allowing us to represent the optimizer with a closed form. 

**Assumption B.2** _Let h_ ( _x, y_ ) _, abbreviate as h, be_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0018-16.png)


_Assume that wt_ ( _xv, πo_ ) _< c_[2] _, where_ 

**Assumption B.3** _Let h_ 1( _xv, xt, xr, y_ ) _, abbreviate as h_ 1 _, be_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0018-19.png)


18 

Preprint. 

Table 11: Model performance (%) of different Med-LVLMs based on LLaVA-Med-1.5 on medical VQA task. 

|Models|**Radiology**<br>IU-Xray<br>MIMIC-CXR|**Ophthalmology**<br>Harvard-FairVLMed|**Pathology**<br>Quilt-1M<br>PMC-OA (Pathology)|
|---|---|---|---|
|LLaVA-Med-1.5<br>MMed-RAG|75.47<br>75.79<br>**89.54**<br>**83.57**|63.03<br>**87.94**|62.80<br>59.28<br>**72.95**<br>**64.54**|
|Med-Flamingo<br>MedVInT<br>RadFM<br>miniGPT-Med|26.74<br>61.27<br>73.34<br>66.06<br>26.67<br>69.30<br>54.87<br>53.92|42.06<br>35.92<br>52.47<br>66.73|27.11<br>32.62<br>26.81<br>27.77<br>27.02<br>25.12<br>26.82<br>27.03|



_Assume that wt_ ( _xr, πo_ ) _< c_[2] 1 _[and wt]_[(˜] _[x][r][, π][o]_[)] _[ > c]_[2] 2 _[, where]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0019-04.png)


B.3 PROOFS 

**Lemma B.1** _Suppose that Assumption B.1 hold, optimizing equation 14 gives_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0019-07.png)


Lemma B.1 indicates that the model tends to increase _πo_ ( _y|x_ ) if _qw_ ( _y|x_ ) _> ql_ ( _y|x_ ), which is more likely to occur when ( _xv, y_ ) represents a preferred sample given _xt_ and _xr_ . Below, we provide an application of Lemma B.1 using a linear regression example. Lemma B.1 is proved with Lemma B.2 and Lemma B.3. 

**Lemma B.2** _(Lemma C.1 in Chen et al. (2024e)) For a, b >_ 0 _, the following inequality holds_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0019-10.png)


_and equality holds if and only if t_ = log( _a/b_ ) 

**Lemma B.3** _Denote_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0019-13.png)


_and abbreviated as p_ 1 _and p_ 2 _for notational convenience. Then,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0019-15.png)


_Equality holds if and only if_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0019-17.png)


_where g_ ( _x_ ) _is any function that is possibly dependent on xv, xt and xr._ 

19 

Preprint. 

## _**Proof B.1**_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0020-02.png)


_where the first inequality follows from Lemma B.2. For equivalence,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0020-04.png)


_Thus, for any xw, yw,o, xl, yl,o, xt, xr,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0020-06.png)


_Therefore, equation 25 holds if and only if there exists some g_ ( _xv, xt, xr_ ) _such that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0020-08.png)


Lemma B.3 provides a closed-form solution to equation 14 if the parameter space is sufficiently large. This lemma is crucial for the proof Lemma B.1, which follows below 

**Proof B.2** _According to the Assumption B.1, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0020-11.png)


_After reparameterization,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0020-13.png)


_which is the global minimum of_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0020-15.png)


_by Lemma B.3. Since π_ ( _xv, y|xt, xr_ ) _∈{πθ, θ ∈_ Θ _} lies in the optimization space, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0020-17.png)


_and πθ_ ( _xv, y|xt, xr_ ) _is the optimizer of equation 30, which gives_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0020-19.png)


20 

Preprint. 

_Then_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0021-02.png)


**Corollary B.1** _Suppose that preferred responses_ ( _xw, yw_ ) _and dispreferred responses_ ( _xl, yl_ ) _satisfy yw_ = _βxw_ + _ϵ_ 1 _and yl_ = _βx_[˜] _l_ + _ϵ_ 2 _respectively. DPO for y_ = _θxv_ + _ϵ_ 3 _is based on reference model y_ = _θoxv_ + _ϵ_ 4 _, where ϵi’s are independent and follow standard normal distribution. Then,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0021-04.png)


Corollary B.1 is a direct application of Lemma B.1, indicating that the model updates coefficient _θo_ towards the direction of _β_ for preferred responses and away from _β_[˜] for dispreferred responses. 

**Proof B.3** _Let ϕ_ ( _·_ ) _denote the probability density function of standard normal, by Lemma B.1,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0021-07.png)


**Lemma B.4** _For linear model y_ = _θ_ 1 _xv_ + _θ_ 2 _xt_ + _ϵ such that ϵ ∼ N_ (0 _,_ 1) _, wt_ ( _xv, πθ_ ) = _θ_ 1[2] 

**Proof B.4** _Let ϕ_ ( _·_ ) _denote the probability density function of standard normal,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0021-10.png)


**Theorem B.2** _Suppose that Assumption B.2 holds, then cross-modality increase the weight of xv. wt_ ( _xv, πθ_ ) _> wt_ ( _xv, πo_ ) (36) 

**Proof B.5** _By Lemma B.1, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0021-13.png)


_Abbreviate h_ ( _x, y_ ) _and πo_ ( _y|xv, xt_ ) _as h and πo respectively, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0021-15.png)


21 

Preprint. 

_the second inequality follows from Cauchy–Schwarz inequality_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0022-02.png)


_Denote c as_ 

_the last term in equation 38 is equivalent to_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0022-05.png)


_Thus, wt_ ( _xv, πθ_ ) _> wt_ ( _xv, πo_ ) _if_ � _wt_ ( _xv, πo_ ) _< c._ 

**Theorem B.3** _Suppose that Assumption B.3 holds, the overall loss increase the weight of xr and decrease the weight of x_ ˜ _r._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0022-08.png)


**Proof B.6** _The distribution of preferred responses can be considered as a mixture distribution: qw_[1][(] _[x][v][, y][w,o][|][x][t][, x][r]_[) +] _[ q] w_[2][(] _[x][v][, y][w,o][|][x][t]_[)] _[.][Similarly, for dispreferred responses, the distribution is rep-] resented as ql_[1][(] _[x][v][, y][l,o][|][x][t]_[) +] _[ q] l_[2][(] _[x][v][, y][l,o][|][x][t][, x][r]_[)] _[.][By Lemma B.1,]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0022-10.png)


_Abbreviate h_ 1( _x, y_ ) _as h_ 1 _. Follow the same procedure in the proof of Theorem B.2,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0022-12.png)


_where we apply Cauchy–Schwarz inequality in equation 44._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0022-14.png)


_Thus, wt_ ( _xr, πθ_ ) _> wt_ ( _xr, πo_ ) _if_ ~~�~~ _wt_ ( _xr, πo_ ) _< c_ 1 _. Again, by Cauchy–Schwarz inequality_ 

_wt_ (˜ _xr, πθ_ ) _− wt_ (˜ _xr, πo_ ) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0022-17.png)


_where_ 

_Thus, wt_ ( _xr, πθ_ ) _< wt_ ( _xr, πo_ ) _if_ � _wt_ ( _xr, πo_ ) _> c_ 2 _._ 

22 

Preprint. 

## C MORE CASES 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0023-02.png)


**----- Start of picture text -----**<br>
Medical<br>Image Question<br>Is there any focal airspace<br>opacity present?<br>MMed-RAG<br>No, there is no focal airspace<br>opacity present in the image.<br>Reference<br>Sternotomy XXXX and mediastinal clips are unchanged.<br>Cardiomediastinal silhouette is unchanged. Pulmonary<br>vasculature and XXXX are unchanged. No XXXX<br>consolidation, pneumothorax or large pleural effusion.<br>Osseous structures and soft tissues are unchanged.<br>**----- End of picture text -----**<br>


## Figure 5: Case 1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0023-04.png)


**----- Start of picture text -----**<br>
Medical<br>Question<br>Image<br>Are all retinoblastoma cases<br>characterized by the same<br>degree of retinal differentiation?<br>MMed-RAG<br>No, retinoblastoma cases can<br>show varying degrees of<br>retinal differentiation, as seen<br>in the image and the reference<br>Reference report.<br>Retinoblastoma shows varying degrees of retinal<br>differentiation, including neuroblastoma and<br>photoreceptor differentiation.<br>**----- End of picture text -----**<br>


Figure 7: Case 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0023-06.png)


**----- Start of picture text -----**<br>
Medical<br>Image Question<br>Does the patient have<br>borderline intraocular pressure?<br>MMed-RAG<br>Yes, the patient has borderline<br>intraocular pressure.<br>Reference<br>64 y.o. patient has history of bilateral canaliculoplasty,<br>lacrimal intubation, and seborrheic keratosis. She is a<br>low-tension glaucoma suspect using Timolol but<br>showed no signs of glaucoma on recent testing. Also<br>has dry AMD and incipient cataract.<br>**----- End of picture text -----**<br>


## Figure 6: Case 2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0023-08.png)


**----- Start of picture text -----**<br>
Medical<br>Question<br>Image<br>Please generate a report<br>based on the image.<br>MMed-RAG<br>Patient shows signs of potential<br>glaucoma with eye cupping and<br>nuclear sclerosis observed. HVF<br>and OCT tests were normal. No<br>diabetic retinopathy. Dry eyes<br>and refractive error noted.<br>Reference<br>Patient likely has thyroid eye disease, using prism for<br>improvement. No explicit mention of glaucoma.<br>Suggested to maintain current prism prescription for 6<br>months.<br>**----- End of picture text -----**<br>


Figure 8: Case 4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models_images/iclr_2025_MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf-0023-10.png)


**----- Start of picture text -----**<br>
Medical<br>Question<br>Image<br>Please generate a report<br>based on the image.<br>MMed-RAG<br>The lungs are well inflated and<br>clear. No focal consolidations<br>identified. The cardiomediastinal<br>silhouette hilar contours are<br>stable. There is no pleural<br>effusion or pneumothorax.<br>Reference<br>No acute intrathoracic process. PA and lateral views of<br>the chest were obtained demonstrating clear well<br>expanded lungs without focal consolidation, effusion,<br>pneumothorax.  No signs of granulomatous disease.<br>Cardiomediastinal silhouette is normal.  Bony structures<br>are intact.  No free air below the right hemidiaphragm is<br>seen.<br>**----- End of picture text -----**<br>


Figure 9: Case 5. 

23 

