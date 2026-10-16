Published as a conference paper at ICLR 2025 

# YOLO-RD: INTRODUCING RELEVANT AND COMPACT - EXPLICIT KNOWLEDGE TO YOLO BY RETRIEVER DICTIONARY 

**Hao-Tang Tsui, Chien-Yao Wang & Hong-Yuan Mark Liao** Institute of Information Science, Academia Sinica 

{henrytsui,kinyiu,liao}@iis.sinica.edu.tw 

## ABSTRACT 

Identifying and localizing objects within images is a fundamental challenge, and numerous efforts have been made to enhance model accuracy by experimenting with diverse architectures and refining training strategies. Nevertheless, a prevalent limitation in existing models is overemphasizing the current input while ignoring the information from the entire dataset. We introduce an innovative _**R** etriever_ - _**D** ictionary_ (RD) module to address this issue. This architecture enables YOLO-based models to efficiently retrieve features from a Dictionary that contains the insight of the dataset, which is built by the knowledge from Visual Models (VM), Large Language Models (LLM), or Visual Language Models (VLM). The flexible RD enables the model to incorporate such explicit knowledge that enhances the ability to benefit multiple tasks, specifically, segmentation, detection, and classification, from pixel to image level. The experiments show that using the RD significantly improves model performance, achieving more than a 3% increase in mean Average Precision for object detection with less than a 1% increase in model parameters. Beyond 1- stage object detection models, the RD module improves the effectiveness of 2-stage models and DETR-based architectures, such as Faster R-CNN and Deformable DETR. Code is released at https://github.com/henrytsui000/YOLO. 

## 1 INTRODUCTION 

In the field of computer vision, object detection models play a pivotal role, these models are designed to precisely locate objects within images. They are used in applications such as medical image analysis and autonomous driving. Additionally, they can also be used as backbone models for downstream tasks like multi-object tracking (Zhang et al., 2022; Cao et al., 2023; Aharon et al., 2022), and crowd counting (Zhang et al., 2016; Song et al., 2021). As the basis for these extended tasks, object detection models must combine high accuracy with low latency to allow downstream tasks to stand on the shoulders of giants. 

Among object detection models, the YOLO (Redmon et al., 2016), FasterRCNN (Ren et al., 2015), and DETR (Carion et al., 2020) are notably prevalent. The YOLO series primarily utilizes Convolutional Neural Networks (CNN) (LeCun et al., 1998), providing a balance between inference speed and accuracy. From YOLOv1 through YOLOv10 (Redmon et al., 2016; Redmon & Farhadi, 2017; 2018; Bochkovskiy et al., 2020; Jocher, 2020; Li et al., 2022; Wang et al., 2023a; Jocher et al., 2023; Wang et al., 2024b;a), there has been a consistent focus on refining the architecture and training methods to reduce the model’s parameters while enhancing accuracy. Beyond the main YOLO series, variants such as YOLOR (Wang et al., 2023b) incorporate implicit knowledge and other techniques to further boost model performance. 

As shown in Figure 1, while both CNNs and Transformers (Vaswani et al., 2017) concentrate on the input image, CNNs are restricted to local input data, and Transformers, despite considering interactions among various inputs, are still confined to the given inputs or other model branches. However, the above-mentioned models often overlook a crucial aspect of explicit knowledge—the comprehensive dataset information. On the other hand, some contrastive methods, e.g. SimCLR (Chen et al., 2020), DINO (Caron et al., 2021) have demonstrated that cross-referencing data is beneficial. 

1 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0002-01.png)


Figure 1: A comparison between traditional models and our proposed _**R** etriever_ _**D** ictionary_ module. On the left, CNN-based models focus on local regions (blue box), while Transformer-based models tend to utilize the entire image (red box). However, both methods fail to leverage the information from the entire dataset, as illustrated in the bottom-right corner. Our module enhances the model’s resource utilization by incorporating knowledge from other parts of the dataset. 

Moreover, in the realm of natural language processing, some models incorporate the Retrieval Augmented Generation (RAG) architecture (Lewis et al., 2020), which stores knowledge in a prebuilt database and retrieves this information during inference to pass to the generator for encoding. This allows the model to access external information from a large, pre-established dataset. This approach significantly enhances the capabilities of large language models, although it typically requires substantial computational resources. However, applying this technique to object detection or other computer vision tasks still faces significant challenges, particularly in preparing and retrieving external data. Object detection models must carefully balance accuracy, model parameters, and latency when handling external information. 

To address these challenges, we introduce a compact external module composed of a _**R** etriever_ and a _**D** ictionary_ , designed to enhance dataset utilization in computer vision models during training. This module effectively filters out irrelevant information and amplifies crucial data. The _**R** etriever_ aggregates region features to generate a query, while the _**D** ictionary_ , containing comprehensive dataset information, enables the query to select relevant atoms. Notably, this pre-built _**D** ictionary_ extends beyond the YOLO backbone, incorporating data encoders like VLMs or LLMs, which bring extensive training data and knowledge for more precise and comprehensive information. 

This module allows models to reinforce data during the forward process, benefiting not only regionlevel tasks like object detection but also pixel-level tasks like segmentation and whole-image tasks like classification. Furthermore, our module can be extended to various model architectures, such as the FPN network in Faster RCNN and the backbone-encoder regions in Detection Transformers, providing higher-quality information during downsampling, ultimately leading to much better performance. 

In this paper, we make several key contributions: 

- We introduce a _**R** etriever_ - _**D** ictionary_ module that enables efficient utilization of external information without the need for an external loss function, while still maintaining the _**D** ictionary_ ’s properties and allowing updates to its parameters. 

- We demonstrate that incorporating external knowledge from models such as VLMs and LLMs can significantly enhance model performance. 

- We further show that integrating external information improves not only YOLO’s object detection performance but also other mainstream vision tasks and architectures. 

These improvements substantially enhance the capabilities of object detection models and demonstrate that our module exhibits All-to-All properties, allowing it to utilize external knowledge to improve performance across multiple tasks and model architectures with minimal parameters. 

2 

Published as a conference paper at ICLR 2025 

## 2 RELATED WORK 

**Real-time object detection.** Real-time object detection is a foundational problem in computer vision, with a focus on achieving low latency and high accuracy. Traditional approaches have primarily focused on CNN architectures, with seminal works such as the OverFeat (Sermanet et al., 2014) and Faster R-CNN(Girshick, 2015; Ren et al., 2015). Since the introduction of Vision Transformers (ViT) (Dosovitskiy et al., 2021), there have been notable follow-up works, including DETR (Carion et al., 2020) and RT-DETR (Zhao et al., 2024). However, the CNN-based YOLO series models hold a crucial position in the field of real-time detection due to their ease of training from scratch, lightweight design, and ability to perform high-speed inference. 

Each version of the YOLO model introduces different architectures and training strategies. For example, YOLOv7 (Wang et al., 2023a) employs ELAN and trainable bag-of-freebies techniques to enhance performance, while YOLOv9 (Wang et al., 2024b) incorporates Generalized Efficient Layer Aggregation Networks (G-ELAN) and Programmable Gradient Information for improved efficiency and learning capability. YOLOv10 (Wang et al., 2024a) further introduces the compact inverted block to optimize model size and computation. On the theoretical front, works like YOLOR (Wang et al., 2023b) leverage the shared characteristics across multiple computer vision tasks, allowing the model to learn implicit knowledge and relax the prediction head, thus generalizing the YOLO architecture to various tasks. Despite these architectural and strategic differences, all YOLO models share a conceptual framework comprising three core modules: a Backbone for downsampling, a Neck (e.g. FPN) for feature fusion, and a Detection Head for final prediction. In this paper, we also utilize the Backbone as an image encoder. 

**Dictionary learning.** Dictionary learning is a fundamental technique in signal processing and machine learning, aimed at learning a set of basis functions (or atoms) that can efficiently represent signals. This technique has been extensively explored in the context of sparse coding, where signals are approximated as sparse linear combinations of dictionary atoms. Additionally, it was discovered that natural images can be effectively represented using sparse coding models (Olshausen & Field, 1997), laying the foundation for further development of dictionary learning algorithms. 

Several algorithms have been proposed to optimize dictionaries and sparse coefficients. Among these, K-SVD (Aharon et al., 2006) has become a standard due to its effectiveness in applications such as image denoising, compression, and inpainting. With the rise of CNNs, dictionary learning has also seen new developments, such as designing convolutional blocks and defining loss functions to achieve dictionary learning objectives (Garcia-Cardona & Wohlberg, 2018; Zheng et al., 2021). Additionally, dictionary learning has been applied to tasks like content-based image retrieval (CBIR), as seen in works like ¸Saban Öztürk (2021); Tarawneh et al. (2019). In this paper, we focus on dictionary learning rather than sparse dictionary learning and emphasize dictionaries that can robustly represent information and retain critical, relevant signals. 

**Retrieval-augmented generation (RAG)** Retrieval-Augmented Generation (RAG) (Lewis et al., 2020) is a technique that first appeared in large language models. It primarily involves three steps: Indexing, where the database is split into chunks, encoded into vectors, and stored in a vector database; Retrieval, which retrieves relevant information based on similarity to the input; and Generation, where both the original input and the retrieved information are fed into the model for further processing. This approach enables LLM to handle unseen information and has been successfully applied in T5 (Raffel et al., 2020), or certain versions of ChatGPT(Achiam et al., 2023). 

Due to the time-consuming nature of the retrieval process, and the real-time requirements of most computer vision tasks, RAG has seen limited application in vision-based models. Recently, some work has been done to mitigate the bottleneck in Wu & Xie (2024); Kim et al. (2024); Liu et al. (2023). For example, RALF (Kim et al., 2024) leverages LLM to embed a huge vocabulary set and find similar meaning words to refine features; REACT (Liu et al., 2023) utilizes the World Wide Web as the information source to extend model knowledge. However, they still require an undetachable huge dataset or Language model which leads to extremely high training and inference loading. In this work, we provide the vision-based model with not only a light-weight database but also refinable features. 

3 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0004-01.png)


Figure 2: _**D** ictionary_ is initialized by encoding the dataset using an image encoder, from which _N_ embeddings are selected as _**D** ictionary_ atoms. During training, for each input feature _Xw,h_ , the _**R** etriever_ core—comprising the Coefficient Generator ( **G** ) and the Global Information Exchanger ( **E** )—generates coefficients for each atom _α_ in the _**D** ictionary D_ . Then, the normalized coefficients are used as weights for each _**D** ictionary_ atom. Finally, by concatenating the residual of _Xw,h_ , the output _Yw,h_ is obtained. 

## 3 METHOD 

In this work, we introduced the _**R** etriever_ - _**D** ictionary_ module, as shown in Figure 2, which enables computer vision models to utilize comprehensive dataset knowledge with minimal extra parameters quickly. This plug-in stores encoded information from various models, enhancing the model’s ability to identify which features of the input data should be emphasized or diminished, thereby improving overall performance. The RD-module is composed of two main components: the _**R** etriever_ and the _**D** ictionary_ . The _**D** ictionary_ consists of _N_ elements, each represented as R _[f]_ vectors, known as atoms _α_ . The _**R** etriever_ generates the coefficient of _α_ for each pixel according to the input. The _**D** ictionary_ can be generated using the YOLO backbone or initialized with diverse models like the visual language model CLIP (Radford et al., 2021) and large language models such as GPT (Radford et al., 2018). This approach allows the model to align visual and linguistic representations, leading to a more balanced and valuable distribution of atoms. Furthermore, the incorporation of linguistic knowledge helps the module retain crucial information. The main goal is to adjust the distribution of the _**D** ictionary_ and allow the _**R** etriever_ core to find the best Atoms’ weight for each input pixel. 

## 3.1 MODULE STRUCTURE 

The _**R** etriever_ core aims to efficiently generate the coefficients of each _α_ in the _**D** ictionary_ . Inspired by depthwise convolution Chollet (2017), we separate the _**R** etriever_ into two components: the Coefficient Generator **G** and the Global Information Exchanger **E** . The Coefficient Generator, denoted as **G** : R _[f][×][W][ ×][H] →_ R _[N][×][W][ ×][H]_ , computes coarse coefficients based on the input feature map _X ∈_ R _[f][×][W][ ×][H]_ , where _f_ is the input feature dimension and _W_ , _H_ represent the width and height of the feature map, respectively. The coarse coefficients are calculated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0004-07.png)


where _W[G] ∈_ R _[N][×][f]_ is the projection matrix of **G** , and _Xw,h ∈_ R _[f]_ is the feature vector at spatial location ( _w, h_ ). 

The Global Information Exchanger, denoted as **E** : R _[N][×][W][ ×][H] →_ R _[N][×][W][ ×][H]_ , refines and exchanges information across neighboring pixels, and is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0004-10.png)


where _W[E] ∈_ R _[N][×]_[1] _[×][k][×][k]_ is the depthwise convolution filter with kernel size _k × k_ , and _i ∈_ [0 _, N_ ) indexes the channels of _Y_ . The term _Y_[(] _[i]_[)] _∈_ R _[W][ ×][H]_ refers to the spatial feature map of the _i_ -th channel. 

4 

Published as a conference paper at ICLR 2025 

The combined operation of **G** and **E** generates the final coefficients vector _**c**_ for each pixel: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0005-02.png)


This separation of tasks minimizes direct computation of the coefficient vectors _**c**_ at each pixel location, significantly reducing the parameter count while maintaining high performance (see Experiments 4.3 and Appendix A.8 for further discussion). 

To prevent the coefficient vectors _**c**_ from simply replicating the input features, which would make the _**D** ictionary_ become an identity matrix, we normalize each _**c**_ . We apply a normalization process equivalent to Positional Normalization (PONO) (Li et al., 2019), which is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0005-05.png)


where the mean _µc_ and variance _σc_ are calculated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0005-07.png)


with _Xn_ representing the _n_ -th feature at a fixed spatial location, and _γ_ , _β_ are learned parameters. Although PONO was initially designed to preserve structural information in generative networks, here it ensures that the feature vectors _**c**_ are properly scaled and centered, preventing the _**D** ictionary_ from collapsing into an identity matrix. 

The normalized coefficients _**c**[′]_ are then used to select atoms from the _**D** ictionary_ , either enhancing or diminishing specific features. This selection is a weighted summation of the atoms and integration with the input residuals to produce the final output. To preserve the dictionary’s learning dynamics, each atom is normalized to unit length during training. The resulting formula is as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0005-10.png)


where _|αi|_ = 1 for all _αi ∈ D_ , and _λ_ is the residual weight. As can be seen in Equation 5, the weighted summation of _**D** ictionary_ atoms is mathematically equivalent to a convolutional layer with a kernel size of 1, stride of 1, and no bias term. To ensure that the sum of each atom’s components equals 1, as required in dictionary learning, we employ weight normalization (Salimans & Kingma, 2016). This eliminates the need for an external objective function to enforce this condition. Weight normalization is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0005-12.png)


Finally, we combine the _**R** etriever_ core ( **G** and **E** ), the _**D** ictionary_ , and the residual connection to express the entire _**R** etriever_ _**D** ictionary_ process RD : R _[f][×][W][ ×][H] →_ R _[f][×][W][ ×][H]_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0005-14.png)


where _∗_ WN( _D_ ) denotes the convolution operation using the weight-normalized _**D** ictionary_ atoms set WN( _D_ ) as filters. 

## 3.2 _**D** ictionary_ INITIALIZATION 

By pre-initializing the _**D** ictionary_ , we embed knowledge into the module atoms of the _**D** ictionary_ . More precisely, we use the selected encoder to map the entire dataset into a high-dimensional space, by incorporating insights from various modality models, as illustrated in Figure 3. This high-dimensional space shares the same dimension as the original YOLO backbone middle layer. Obviously, this will have a huge number of vectors and present multiple groups in high-dimensional space, we employ _k_ -means (Macqueen, 1967) to leave representative vectors, which ultimately serve as _**D** ictionary_ atoms. Through these operations, we can map the dataset pairs into a high-dimensional space in a short time (this operation is equivalent to using the model to traverse the entire dataset at the speed of inference), and find vectors that can represent most of the dataset. For features that are not in the _**D** ictionary_ , we can use the feature through the residual mechanism in the module, and utilize the _**R** etriever_ _**D** ictionary_ to bring closer atoms of the same category to the outlier feature. Next, we discuss using different modality models as encoders for encoding knowledge: 

5 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0006-01.png)


**----- Start of picture text -----**<br>
(a) Dataset distribution (b) Init with YOLO (c) Init with GPT (d) Init with CLIP<br>**----- End of picture text -----**<br>


Figure 3: Illustrates the distribution of the dataset in the model’s middle layer, where the blue square represents the current input feature. In traditional models, only the input feature is used, neglecting the rich information available in the dataset. In contrast, with our Retriever-Dictionary model, additional data information is retrieved from the dataset. The dictionary can be initialized from different models: vision models, language models, or vision-language models. The latter provides a more comprehensive and integrated representation of the dataset. 

**Vision model.** In the YOLO architecture, the backbone layer is designed to transform the original image into high-dimensional embeddings, providing the FPN with enriched regional information, which makes the backbone an ideal image encoder. Therefore, we use the modern pre-trained YOLOv9 (Wang et al., 2024b) as the vision model encoder. This model is leveraged to traverse the entire training set, converting all the dataset’s data into feature-dimensional distributions. **Vision language model.** As a visual language model, CLIP possesses a strong understanding of input image features and can map these features into a semantically related space for comparison. Therefore, we utilize CLIP for visual language initialization. However, according to DenseCLIP (Rao et al., 2022), although CLIP’s output embeddings are global representations of the image, the output of CLIP’s ViT blocks retains information about corresponding regions. CLIP primarily uses the first patches as embeddings, leaving out local information from the other patches. As a result, we leverage all the output patches from CLIP’s image encoder to embed the dataset more comprehensively. 

**Large language model.** For initialization from an LLM, since language models cannot directly convert images into feature embeddings and image captions typically describe the entire image rather than specific regions, we use class names from MSCOCO (Lin et al., 2014) and ImageNet21k (Ridnik et al., 2021), along with image captions from MSCOCO, as prompts for GPT (Radford et al., 2018). However, because some feature dimensions deviate significantly from the unit interval, we apply standard normalization to scale each dimension, aiming to stabilize the training process while preserving the relative positioning of the vectors. 

## 3.3 _**D** ictionary_ COMPRESSION 

The objective of designing _**R** etriever_ _**D** ictionary_ (RD) is to retain the most critical information from the dataset. However, even after training, the number of atoms in the _**D** ictionary_ may exceed the required amount, with some atoms being infrequently used or irrelevant to the specific dataset domain. Drawing inspiration from knowledge distillation and the Teacher-Student model (Hinton et al., 2015), we condense the original _**D** ictionary D_ into a smaller and more efficient version, denoted as _d_ . To align the output features of _d_ with those of _D_ , we employ contrastive learning (Chen et al., 2020) to provide the _d_ soft labels instead of the traditional cross-entropy loss. In this process, the model backbone and RD are frozen, while the optimization is focused on the smaller _**R** etriever_ _**D** ictionary_ module _d_ . The objective function is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0006-07.png)


where _zi,w,h_[rd] _[∈]_[R] _[f]_[represents the] _[ i]_[-th batch output feature of] _[ d]_[ at position][ (] _[w, h]_[)][,] _[ B]_[is the batch size,] _τ_ is the temperature parameter, and sim( _·, ·_ ) denotes the cosine similarity between two vectors. 

6 

Published as a conference paper at ICLR 2025 

Table 1: Comparing the performance and the improvements across different modalities of RD-module and structures on the COCO 2017 validation set. 

|BackBone<br>Initializer|Params<br>Latency<br>_f_<br>mAP_val_<br>_._5:_._95 (%)<br>mAP_val_<br>_._5 (%)|
|---|---|
|||
|YOLOv7<br>Origin<br>37.2M<br>3.59<br>-<br>50.04<br>68.65<br>YOLOv9<br>Origin<br>25.3M<br>4.00<br>-<br>52.64<br>69.56<br>Faster-RCNN<br>Origin<br>43.1M<br>41.00<br>-<br>38.40<br>59.00<br>Deformable DETR<br>Origin<br>40.1M<br>41.10<br>-<br>43.80<br>62.60||
|||
|YOLOv7<br>VM<br>YOLOv9<br>VM<br>Faster-RCNN<br>VM<br>Deformable DETR<br>VM|37.4M<br>3.70<br>R512<br>51.37(_↑_2.66%)<br>69.42(_↑_1.13%)<br>25.5M<br>4.16<br>R512<br>53.41(_↑_1.46%)<br>70.57(_↑_1.46%)<br>44.1M<br>41.00<br>R512<br>40.50(_↑_5.47%)<br>60.30(_↑_2.20%)<br>41.2M<br>41.10<br>R512<br>44.10(_↑_0.68%)<br>63.30(_↑_1.12%)|
|||
|YOLOv7<br>VLM<br>YOLOv9<br>VLM<br>Faster-RCNN<br>VLM<br>Deformable DETR<br>VLM|37.4M<br>3.70<br>R512<br>51.75(_↑_3.42%)<br>70.12(_↑_2.15%)<br>25.5M<br>4.16<br>R512<br>53.36(_↑_1.37%)<br>70.55(_↑_1.43%)<br>44.1M<br>41.02<br>R512<br>40.50(_↑_5.47%)<br>60.40(_↑_2.37%)<br>41.2M<br>41.28<br>R512<br>44.40(_↑_1.37%)<br>63.30(_↑_1.12%)|
|||
|YOLOv7<br>LLM<br>YOLOv9<br>LLM<br>Faster-RCNN<br>LLM<br>Deformable DETR<br>LLM|38.2M<br>3.79<br>R1024<br>51.36(_↑_2.64%)<br>69.40(_↑_1.17%)<br>25.8M<br>4.20<br>R1024<br>53.28(_↑_1.22%)<br>70.48(_↑_1.33%)<br>44.6M<br>41.03<br>R1024<br>40.70(_↑_5.99%)<br>60.80(_↑_3.05%)<br>41.7M<br>41.35<br>R1024<br>44.16(_↑_0.91%)<br>63.10(_↑_1.12%)|



This distillation process ensures that, within the specific domain, the atoms in _d_ can effectively approximate various potential linear combinations found in the original _**D** ictionary D_ . By selectively removing atoms from RD that are not pertinent to the dataset domain, we achieve a significant reduction in atom count—by at least 50%. This reduction not only increases the model’s efficiency but also maintains the performance and expressiveness within the targeted domain. 

## 4 EXPERIMENT 

## 4.1 SETUPS 

**Experimental setup.** We primarily validated the method on the Microsoft COCO dataset (Lin et al., 2014), training on the COCO 2017 train set and evaluating on the COCO 2017 validation set. For Object Detection and Segmentation, we respectively used mAP and mAP@.5 as evaluation metrics, testing on YOLOv7, YOLOv9, Faster RCNN, and Deformable DETR. For the Classification task, we used the CIFAR-100 dataset with the YOLOv9-classify model, using top-1 and top-5 accuracy as metrics. 

**Implementation details.** All experiments were conducted using 8 Nvidia V100 GPUs. In the main series of experiments, we trained a modified YOLOv7 model, which included the addition of a _**R** etriever_ - _**D** ictionary_ Module, for 300 epochs in 2 days. The YOLOv9-based model was trained for 5 days with 500 epochs. We also trained a modified Faster RCNN, based on the mm-detection framework, for a maximum of 120 epochs over 3 days. For Deformable DETR, we trained for approximately 120 epochs over 7 days. The classification task on CIFAR-100 (Krizhevsky et al., 2009) took 2 hours on a single Nvidia 4090 GPU for 100 epochs. Latency was measured on a single Nvidia 3090 GPU without any external acceleration tools, using milliseconds per batch with a batch size of 32 on the MSCOCO validation set. 

## 4.2 COMPARISION WITH RD 

**Apply to state-of-the-art detectors.** As demonstrated in Table 1, we evaluate the proposed module mainly on YOLOv7 and also integrate it with both fundamental and SOTA real-time object detection models. The _**D** ictionary_ is initialized separately by three distinct models: Vision Model (VM) with the YOLOv7 backbone, Vision-Language Model (VLM) employing CLIP, and Large Language 

7 

Published as a conference paper at ICLR 2025 

Model (LLM) based on GPTv2. Among these, CLIP provides the most significant improvement, likely due to its well-balanced performance across both the vision and language domains. 

The RD module consistently yields notable improvements. In YOLOv7 and YOLOv9, the introduction of the module increases the parameter count by less than 1%, yet results in substantial performance gains across key metrics. The improvement is comparable to the performance boost achieved by moving to the next model size (e.g., YOLOv7-x, YOLOv9-e), which typically requires a 100% increase in parameters. For more traditional architectures, we incorporate the RD module into Faster R-CNN with a ResNet-50 backbone He et al. (2016). Despite a modest 2% increase in parameters, the model outperforms Faster R-CNN with a ResNet-101 backbone. Furthermore, we extend the module to a transformer-based architecture, specifically Deformable DETR Zhu et al. (2020) with a ResNet-50 backbone. Similar to the previous results, the RD module yields improvements equivalent to upgrading to the ResNet-101 backbone. 

These experiments conclusively demonstrate that leveraging dataset information and incorporating knowledge from VM, VLM, and LLM significantly enhances the performance of a wide range of base models, while requiring only minimal additional parameters. 

**Apply to other tasks.** The _**R** etriever_ _**D** ictionary_ (RD) module enhances pixel-level features, and its potential benefits extend beyond detection tasks to include other vision tasks, such as segmentation and classification. To validate this, we conducted segmentation experiments on the MSCOCO dataset and classification experiments on the CIFAR-100 dataset, demonstrating the effectiveness of the RD module across both pixel-level and image-level tasks. Table 2 compares the original YOLO multi-task structure with the one incorporating our proposed module. The results clearly show that the _**R** etriever_ _**D** ictionary_ module upgrades performance across classification, detection, and segmentation tasks, demonstrating its effectiveness in enhancing overall multi-tasking performance. 

Table 2: Comparing RD at different tasks. 

Table 3: Comparing with different knowledge-based methods. 

|Task<br>Metrics(%)<br>w/o RD<br>w/ RD<br>Improve<br>Classifcation<br>Top-1<br>74.86<br>75.70<br>_↑_1.12%<br>Top-5<br>93.72<br>94.28<br>_↑_0.60%<br>Detection<br>mAP_Box_<br>50.04<br>51.75<br>_↑_3.42%<br>mAP_Box_<br>_._5<br>68.65<br>69.51<br>_↑_2.15%<br>Segmentation<br>mAP_Seg_<br>40.53<br>41.56<br>_↑_2.54%<br>mAP_Seg_<br>_._5<br>64.00<br>64.64<br>_↑_1.00%|Task<br>Metrics(%)<br>w/o RD<br>w/ RD<br>Improve<br>Classifcation<br>Top-1<br>74.86<br>75.70<br>_↑_1.12%<br>Top-5<br>93.72<br>94.28<br>_↑_0.60%<br>Detection<br>mAP_Box_<br>50.04<br>51.75<br>_↑_3.42%<br>mAP_Box_<br>_._5<br>68.65<br>69.51<br>_↑_2.15%<br>Segmentation<br>mAP_Seg_<br>40.53<br>41.56<br>_↑_2.54%<br>mAP_Seg_<br>_._5<br>64.00<br>64.64<br>_↑_1.00%|nowege-ase metos.|nowege-ase metos.|
|---|---|---|---|
|||Method<br>mAP<br>mAP_._5<br>+Params||
|||||
|||baseline<br>KD<br>YOLO-World<br>RALF<br>**RD (ours)**|52.64<br>69.56<br>-<br>52.52<br>69.14<br>57.3M<br>51.00<br>67.70<br>66.1M<br>51.40<br>68.07<br>37.8M<br>**53.36**<br>**70.55**<br>**0.2M**|
|Segmentation<br>mAP_Seg_<br>mAP_Seg_<br>_._5|40.53<br>41.56<br>_↑_2.54%<br>64.00<br>64.64<br>_↑_1.00%|||
|||||



**Comparison of knowledge integration methods for YOLO.** Table 3 compares RD with various methods for integrating external knowledge into YOLO. Specifically, knowledge distillation (Hinton et al., 2015) uses YOLOv9-e as the teacher model and YOLOv9-c as the student model, denoted as KD. Another approach, YOLO-World (Cheng et al., 2024), incorporates visual-language concepts from CLIP into YOLO to enhance its understanding of both domains. Additionally, RALF (Kim et al., 2024) utilizes CLIP’s text encoder to create a vocabulary set as a database in a RAG-based method. The "+Params" column represents the additional parameters introduced by the knowledge provider or supervisor model compared to the baseline. Overall, RD not only offers the lightest solution, with only 0.2 additional parameters but also delivers the best performance, making it a highly efficient and effective approach. 

## 4.3 ABLATION STUDIES 

Table 4: Analysis of fuse _**R** etriever_ core. 

|Retriever<br>Weight size<br>Params.|mAP|
|---|---|
|||
|Fuse(**E**_,_**G**)(X)<br>R_f×N×k×k_<br>41.9M<br>**E**(**G**(X))<br>R_f×N,_R_N×k×k_<br>4.3M|50.95%<br>50.64%|



8 

Published as a conference paper at ICLR 2025 

Table 5: Performance comparison of different models on the COCO 2017 validation set. 

|BackBone<br>Initializer<br>algorithm|Params<br>Atoms<br>Feature<br>mAP_val_<br>_._5<br>mAP_val_<br>_._5:_._95|
|---|---|
|||
|YOLOv7<br>Random<br>-<br>YOLOv7<br>YOLOv7<br>-<br>YOLOv7<br>CLIP<br>-<br>YOLOv7<br>GPT<br>-|37.4M<br>512<br>R512<br>50.02%<br>69.06%<br>37.4M<br>512<br>R512<br>51.34%<br>69.37%<br>37.4M<br>512<br>R512<br>51.75%<br>70.12%<br>38.2M<br>512<br>R1024<br>51.33%<br>69.40%|
|||
|YOLOv7<br>YOLOv7<br>_k_-means<br>YOLOv7<br>YOLOv7<br>VAE|37.4M<br>512<br>R512<br>51.34%<br>69.37%<br>37.4M<br>512<br>R512<br>51.19%<br>69.25%|
|||
|YOLOv7<br>CLIP<br>_k_-means<br>YOLOv7<br>CLIP<br>Convex Hull|37.4M<br>512<br>R512<br>51.75%<br>70.12%<br>37.8M<br>1024<br>R512<br>49.58%<br>68.92%|
|||
|YOLOv7<br>GPT<br>Normalize<br>YOLOv7<br>GPT<br>Tanh|38.2M<br>512<br>R1024<br>51.33%<br>69.40%<br>38.2M<br>512<br>R1024<br>51.27%<br>69.35%|
|||
|YOLOv7<br>CLIP<br>-<br>YOLOv7<br>GPT<br>-<br>YOLOv7<br>Mix<br>{LLM, VLM}|37.4M<br>512<br>R512<br>51.75%<br>70.12%<br>38.2M<br>512<br>R1024<br>51.33%<br>69.40%<br>37.4M<br>512<br>R512<br>51.38%<br>69.59%|



**Fuse coefficient generator and global information exchanger.** In Section 3.1, we discussed splitting the _**R** etriever_ core into pointwise convolution **G** and depthwise convolution **E** can significantly reduce the number of parameters. Without a split _**R** etriever_ core, the 1-stage coefficient generates the process as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0009-04.png)


where _W[eq] ∈_ R _[k][×][k][×][f][×][n]_ is the convolution matrix, R _[′]_ : R _[f][×][W][ ×][H] →_ R _[N][×][W][ ×][H]_ is the 1-stage generate process. This operation will require an extra of _Nfk_[2] _−_ ( _fN_ + _Nk_[2] ) parameters. And the computational complexity is of _O_ ( _WHNfk_[2] ), compared to **G** and **E** , whose complexity is of _O_ � _WH_ ( _fN_ + _Nk_[2] )�, R _[′]_ requires a huge amount of operations and parameters, but only receives slightly better performance, as demonstrated in Table 4. 

**Different** _**Dictionary**_ **construction strategies.** Table 5 presents an ablation study on more different strategies for constructing the _**D** ictionary_ . For VM, a VAE (Kingma & Welling, 2014) is used to capture the dataset’s general distribution and build the _**D** ictionary_ . In VLM, we leverage whole CLIP (Radford et al., 2021)’s image encoder for global embeddings, as outlined in Section 3.2. 

Additionally, we explore using a convex hull of the dataset’s feature distribution to address outliers, though this limits the RD-module’s ability to represent common features. For LLM, normalization techniques like tanh and standard normalization are applied to manage outliers. While tanh compresses large feature distances, standard normalization is more effective in handling outliers. 

Finally, we combine feature distributions from LLM and VLM to create a blended representation. Despite yielding positive results, these methods fail to produce a uniform distribution and underperform compared to the strategy described in the methodology. 

## 4.4 VISUALIZATION 

In this section, we employ visualization to demonstrate the impact of the RD-module on object detection models and to visualize the _**R** etriever_ ’s selection of atoms from the _**D** ictionary_ . 

**Visualization of** _**Dictionary**_ **atom coefficients.** To gain a deeper understanding of the behavior of the _**R** etriever_ and _**D** ictionary_ core, we visualized the atom coefficients and their distribution during the forward pass. In Figure 4b, the X-axis represents the correlation with the current input and each _α_ , while the Y-axis displays the corresponding coefficient for each atom. The surrounding points retrieved from the dataset, relevant to the current input, are marked as _×_ on the plot. 

Using the bounding box area in Figure 4a as the input, we generate a correlation-coefficient map. As depicted in Figure 4b, most high-correlation and high-coefficient pairs (Figures 4c,4d) correspond 

9 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0010-01.png)


**----- Start of picture text -----**<br>
(c) Positive 1 (d) Positive 2<br>(a) Input Image (b) Large image in the middle (e) Negative 1 (f) Negative 2<br>**----- End of picture text -----**<br>


Figure 4: Visualization of Dictionary Atom Coefficients 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0010-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0010-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0010-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0010-06.png)


**----- Start of picture text -----**<br>
(a) Model’s input image (b) The output of backbone (c) The output of RD module<br>**----- End of picture text -----**<br>


Figure 5: Visualization of w/o RD module 

to traffic signs, while the low-correlation and low-coefficient pairs (Figures 4e,4f) do not. This visualization demonstrates that the _**R** etriever_ _**D** ictionary_ effectively selects the relevant atoms to enhance input features while attenuating non-relevant atoms, thereby reinforcing the input’s key characteristics. 

**Visualizing model backbone output with and without** _**Dictionary**_ **.** Using the same traffic signals as the model input (Figure 5a), we visualize the feature map generated by the origin backbone network (Figure 5b) and the backbone output including the RD-module (Figure 5c). In Figure 5c, the Traffic Sign pattern is distinctly retained, and background information is preserved. In contrast, Figure 5b retains only parts of the railings and the input photo padding block. This comparison demonstrates that the RD-module helps the model retain important information while eliminating unimportant details. 

## 5 CONCLUSION 

The _**R** etriever_ _**D** ictionary_ module offers a lightweight and efficient approach to incorporating dataset knowledge into YOLO through various modality models. By leveraging pre-stored explicit knowledge within the _**D** ictionary_ , the _**R** etriever_ effectively retrieves relevant information while the Dictionary learning mechanism enables fine-tuning of the atoms. This module demonstrates its versatility, providing improvements not only in YOLO-based tasks but also across a range of foundational object detection models and broader computer vision tasks. We believe that this work lays the foundation for further exploration of real-time computer vision models that integrate explicit or external knowledge sources. 

10 

Published as a conference paper at ICLR 2025 

## REFERENCES 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, et al. GPT-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Michal Aharon, Michael Elad, and Alfred Bruckstein. K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation. _IEEE Transactions on Signal Processing_ , 54, 2006. 

- Netzer Aharon, Eyal Gazit, and Itzik Bibi. BoT-SORT: Robust associations multi-pedestrian tracking. _arXiv preprint arXiv:2206.14651_ , 2022. 

- Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. YOLOv4: Optimal speed and accuracy of object detection. _arXiv preprint arXiv:2004.10934_ , 2020. 

Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, and Kris Kitani. Observation-centric SORT: Rethinking SORT for robust multi-object tracking. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 

- Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , 2020. 

- Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the International Conference on Computer Vision (ICCV)_ , 2021. 

- Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In _International Conference on Machine Learning (ICML)_ . PMLR, 2020. 

- Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. YOLO-World: Real-time open-vocabulary object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2024. 

- François Chollet. Xception: Deep learning with depthwise separable convolutions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2017. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations (ICLR)_ , 2021. 

- Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The PASCAL visual object classes (VOC) challenge. _International journal of computer vision_ , 88: 303–338, 2010. 

- Cristina Garcia-Cardona and Brendt Wohlberg. Convolutional dictionary learning: A comparative review and new algorithms. _IEEE Transactions on Computational Imaging_ , 2018. 

- Ross Girshick. Fast R-CNN. In _Proceedings of the International Conference on Computer Vision (ICCV)_ , 2015. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2016. 

- Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. _arXiv preprint arXiv:1503.02531_ , 2015. 

Glenn Jocher. ultralytics/yolov5: v2.0, 2020. 

- Glenn Jocher, Ayush Chaurasia, and Jing Qiu. Ultralytics YOLO, 2023. URL https://github. com/ultralytics/ultralytics. 

11 

Published as a conference paper at ICLR 2025 

- Jooyeon Kim, Eulrang Cho, Sehyung Kim, and Hyunwoo J Kim. Retrieval-augmented openvocabulary object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2024. 

- Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In _International Conference on Learning Representations (ICLR)_ , 2014. 

- Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. CIFAR-100 (canadian institute for advanced research). _URL http://www.cs.toronto.edu/kriz/cifar.html_ , 2009. 

- Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. _Proceedings of the IEEE_ , 86(11):2278–2324, 1998. 

- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In _Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)_ , 2020. 

- Boyi Li, Felix Wu, Kilian Q Weinberger, and Serge Belongie. Positional normalization. In _Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)_ , 2019. 

- Cheng Li, Qiaofeng Ye, Tongda Zhang, and Kai Chen. YOLOv6: A single-stage object detection framework for industrial applications. _arXiv preprint arXiv:2209.02976_ , 2022. 

- Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In _Proceedings of the European Conference on Computer Vision (ECCV)_ . Springer, 2014. 

- Haotian Liu, Kilho Son, Jianwei Yang, Ce Liu, Jianfeng Gao, Yong Jae Lee, and Chunyuan Li. Learning customized visual models with retrieval-augmented knowledge. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 15148–15158, 2023. 

- J Macqueen. Some methods for classification and analysis of multivariate observations. In _Proceedings of 5-th Berkeley Symposium on Mathematical Statistics and Probability/University of California Press_ , 1967. 

- Bruno A. Olshausen and David J. Field. Sparse coding with an overcomplete basis set: A strategy employed by V1? _Vision Research_ , 37, 1997. 

- Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. In _OpenAI_ , 2018. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning (ICML)_ . PMLR, 2021. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _J. Mach. Learn. Res._ , 2020. 

- Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong Tang, Zheng Zhu, Guan Huang, Jie Zhou, and Jiwen Lu. DenseCLIP: Language-guided dense prediction with context-aware prompting. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 

- Joseph Redmon and Ali Farhadi. YOLO9000: Better, faster, stronger. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2017. 

- Joseph Redmon and Ali Farhadi. YOLOv3: An incremental improvement. _arXiv preprint arXiv:1804.02767_ , 2018. 

12 

Published as a conference paper at ICLR 2025 

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2016. 

- Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: towards real-time object detection with region proposal networks. In _Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)_ , 2015. 

- Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi Zelnik-Manor. ImageNet-21K pretraining for the masses. In _Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)_ , 2021. 

- Tim Salimans and Durk P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. _Advances in neural information processing systems (NeurIPS)_ , 29, 2016. 

- Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, and Yann LeCun. OverFeat: Integrated recognition, localization and detection using convolutional networks. In _International Conference on Learning Representations (ICLR)_ , 2014. 

- Qingyu Song, Changan Wang, Zhengkai Jiang, Yabiao Wang, Ying Tai, Chengjie Wang, Jilin Li, Feiyue Huang, and Yang Wu. Rethinking counting and localization in crowds: A purely pointbased framework. In _Proceedings of the International Conference on Computer Vision (ICCV)_ , 2021. 

- Ahmad S. Tarawneh, Ahmad B. Hassanat, Ceyhun Celik, Dmitry Chetverikov, M. Sohel Rahman, and Chaman Verma. Deep face image retrieval: a comparative study with dictionary learning. In _2019 10th International Conference on Information and Communication Systems (ICICS)_ , 2019. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)_ , 2017. 

- Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, and Guiguang Ding. YOLOv10: Real-time end-to-end object detection. In _Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)_ , 2024a. 

- Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh, and I-Hau Yeh. CSPNet: A new backbone that can enhance learning capability of CNN. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops (CVPRw)_ , 2020. 

- Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. YOLOv7: Trainable bag-offreebies sets new state-of-the-art for real-time object detectors. 2023a. 

- Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. You only learn one representation: Unified network for multiple tasks. _Journal of Information Science and Engineering_ , 2023b. 

- Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. YOLOv9: Learning what you want to learn using programmable gradient information. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , 2024b. 

- Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal LLMs. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2024. 

- Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Ping Luo, and Zehuan Yuan. ByteTrack: Multiobject tracking by associating every detection box. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , 2022. 

- Yingying Zhang, Desen Zhou, Siqin Chen, Shenghua Gao, and Yi Ma. Single-image crowd counting via multi-column convolutional neural network. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2016. 

13 

Published as a conference paper at ICLR 2025 

- Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, and Jie Chen. DETRs beat YOLOs on real-time object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 16965–16974, 2024. 

- Hongyi Zheng, Hongwei Yong, and Lei Zhang. Deep convolutional dictionary learning for image denoising. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 

- Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: Deformable transformers for end-to-end object detection. In _International Conference on Learning Representations (ICLR)_ , 2020. 

- ¸Saban Öztürk. Convolutional neural network based dictionary learning to create hash codes for content-based image retrieval. _Procedia Computer Science_ , 2021. Proceedings of the 10th International Conference of Information and Communication Technology. 

14 

Published as a conference paper at ICLR 2025 

## A APPENDIX 

## A.1 REPRODUCIBILITY AND TRAINING SETUP 

In Table 5, we observed that some of the mAP values for the original YOLOv7 and YOLOv9 models differ from those reported in the original papers. As noted in the corresponding GitHub issue, this discrepancy may be attributed to differences in the number of GPUs used, as well as the reduced effectiveness of batch normalization when training with smaller batch sizes. To ensure a more consistent and fair comparison, we followed the official code instructions and re-trained the models on our hardware setup, using the same batch sizes and number of GPUs specified in the original papers. Despite adhering to the official training guidelines, our re-run results for YOLOv7 and YOLOv9 produced slightly lower mAP values than those reported in the original papers. Nevertheless, our implementation of the _**R** etriever_ _**D** ictionary_ (RD) method is still higher than those reported in the original papers, demonstrating the effectiveness of our approach. 

For the Deformable DETR and Faster R-CNN models, we used the MMDetection training code. In this case, our results were consistent with the values reported in the original papers, likely due to the fact that MMDetection provides a standardized training batch configuration, minimizing the impact of hardware differences on model performance. 

## A.2 ARCHITECTURES OF THE MODEL WITH _**R** etriever_ _**D** ictionary_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0015-06.png)


Figure 6: YOLOv7 with RD 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0015-08.png)


Figure 8: Faster-RCNN with RD 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0015-10.png)


Figure 7: YOLOv9 with RD 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0015-12.png)


Figure 9: Deformable DETR with RD 

## A.3 ADDITIONAL EXPERIMENT RESULTS 

**Detailed mAP Results for All Experiments.** Table 6 provides the detailed mAP values for each experiment discussed in the experiment section. It includes metrics such as AP, AP _._ 5, AP _._ 75, APS, small object; APM, middle object; and APL, large object. 

15 

Published as a conference paper at ICLR 2025 

Table 6: Comprehensive mAP Metrics Across Experiments 

|BackBone<br>Initializer|Params<br>AP<br>AP_._5<br>AP_._75<br>AP_S_<br>AP_M_<br>AP_L_|
|---|---|
|||
|YOLOv7<br>VM<br>YOLOv7<br>VLM<br>YOLOv7<br>LLM<br>YOLOv7<br>VAE<br>YOLOv7<br>Convex<br>YOLOv7<br>Tanh<br>YOLOv7<br>Mix|37.4M<br>51.34<br>69.37<br>56.42<br>35.86<br>56.10<br>66.78<br>37.4M<br>51.75<br>70.12<br>56.70<br>36.00<br>57.12<br>66.32<br>38.2M<br>51.33<br>69.40<br>56.40<br>34.72<br>56.70<br>67.13<br>37.4M<br>51.34<br>69.37<br>56.42<br>35.86<br>56.10<br>66.78<br>37.4M<br>49.58<br>68.92<br>55.60<br>34.72<br>55.14<br>66.02<br>37.4M<br>51.27<br>69.35<br>56.28<br>35.51<br>56.34<br>66.09<br>37.4M<br>51.37<br>69.49<br>56.26<br>35.17<br>56.43<br>66.94|
|||
|YOLOv9<br>VM<br>YOLOv9<br>VLM<br>YOLOv9<br>LLM|25.8M<br>53.41<br>70.57<br>58.40<br>36.29<br>58.93<br>69.94<br>25.8M<br>53.36<br>70.55<br>57.98<br>36.39<br>59.04<br>69.95<br>25.8M<br>53.28<br>70.48<br>57.81<br>35.91<br>58.80<br>69.55|
|Faster-RCNN<br>VM<br>44.1M<br>40.50<br>60.30<br>44.20<br>24.90<br>43.20<br>51.80<br>Faster-RCNN<br>VLM<br>44.1M<br>40.50<br>60.40<br>44.40<br>25.10<br>43.30<br>52.50<br>Faster-RCNN<br>LLM<br>44.6M<br>40.70<br>60.80<br>44.70<br>25.60<br>43.70<br>52.60||
|||
|Deformable DETR<br>VM<br>Deformable DETR<br>VLM<br>Deformable DETR<br>LLM|41.2M<br>44.10<br>63.00<br>48.20<br>26.30<br>47.00<br>59.00<br>41.2M<br>44.40<br>63.30<br>48.30<br>26.50<br>47.90<br>58.70<br>41.7M<br>44.20<br>63.10<br>47.80<br>26.20<br>47.60<br>58.70|
|||
|YOLOv7<br>baseline<br>YOLOv7-x<br>baseline|37.2M<br>50.04<br>68.95<br>55.10<br>34.20<br>55.70<br>66.20<br>71.3M<br>51.29<br>70.27<br>56.78<br>35.84<br>56.63<br>67.59|



**Comparison of Frozen vs. Fully Trained** _**Dictionary**_ **Strategies.** Table 7 compares two training strategies: one where only the model _B_ and _**R** etriever_ are trained while the _**D** ictionary_ remains frozen, and another where the model, _**R** etriever_ , and _**D** ictionary_ are all fully trained. In the table, ✓ indicates that the corresponding component is trainable. Table 7a shows results using pre-trained weights, with row 3 displaying the fine-tuning of the original model. Table 7b reports results from training the model from scratch. The full training strategy slightly outperforms the frozen _**D** ictionary_ method in both scenarios, with both approaches surpassing the performance of the original model. These findings highlight fine-tuning the _**D** ictionary_ effectively helps the model’s output distribution. 

Table 7: Comparison of Model Performance with and without Freezing _D_ during Training. 

|a) Fine-tune with pre-trained weight.<br>_B_<br>_R_<br>_D_<br>AP_._5:_._95<br>AP_._5<br>✓<br>✓<br>✓<br>52.73<br>69.61<br>✓<br>✓<br>52.64<br>69.57<br>✓<br>-<br>-<br>51.66<br>68.12|a) Fine-tune with pre-trained weight.<br>_B_<br>_R_<br>_D_<br>AP_._5:_._95<br>AP_._5<br>✓<br>✓<br>✓<br>52.73<br>69.61<br>✓<br>✓<br>52.64<br>69.57<br>✓<br>-<br>-<br>51.66<br>68.12|(b) Training from scratch.|(b) Training from scratch.|
|---|---|---|---|
|||_B_<br>_R_<br>_D_<br>AP_._5:_._95<br>AP_._5||
|||||
|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>-<br>-|52.73<br>69.61<br>52.64<br>69.57<br>51.66<br>68.12|✓<br>✓<br>✓<br>✓<br>✓|51.72<br>70.12<br>51.35<br>69.52|
|||||



**Performance Improvements in Classification with RD Module.** Table 8 demonstrates the performance improvement of the _**R** etriever_ _**D** ictionary_ (RD) module in the classification task. In the YOLO classification task, YOLOv8 employs CSPNet (Wang et al., 2020) as the backbone, while YOLOv9 uses GELAN as the backbone. We tested both backbones with our RD module, and the results show that the RD module provides performance improvements in both structures. 

## A.4 TRANSFER LEARNING WITH _**R** etriever_ _**D** ictionary_ ON VOC DATASET 

In Table 9, we demonstrate the effectiveness of the _**R** etriever_ _**D** ictionary_ (RD) on a transfer learning task. Using pre-trained weights from the MSCOCO dataset, we trained the model on the VOC (Everingham et al., 2010) dataset with three learning rate schedules: 10-epoch fast training, 100-epoch full-tuning, and training from scratch. In the fast training scenario, the model with the RD module showed significantly faster convergence compared to the model without the module. In the full-tuning scenario, the RD-enhanced model achieved higher performance. Lastly, in the training from scratch scenario, our RD module provided the model with better information, yielding superior results even on a smaller dataset. 

16 

Published as a conference paper at ICLR 2025 

Table 9: Transfer learning on small dataet. 

Table 8: More classification task. 

|Table 8: More classifcation task.|Table 8: More classifcation task.|||
|---|---|---|---|
|||RD<br>Pretrained<br>Epoch|mAP(%)<br>mAP_._5(%)|
|RD<br>Model<br>Epoch|Top-1<br>Top-5|✓<br>10<br>88.48<br>65.87<br>✓<br>✓<br>10<br>**91.54**(_↑_3.46%)<br>**74.63**(_↑_13.30%)||
|||||
|ELAN<br>100<br>✓<br>ELAN<br>100|71.85<br>92.93<br>**74.18**<br>**93.14**|||
|||||
|||✓<br>100<br>✓<br>✓<br>100|92.28<br>76.79<br>**92.93**(_↑_0.70%)<br>**78.02**(_↑_1.60%)|
|||||
|GELAN<br>100<br>✓<br>GELAN<br>100|74.86<br>93.72<br>**75.70**<br>**94.28**|||
|||||
|||100<br>✓<br>100|84.44<br>65.49<br>**85.15**(_↑_0.84%)<br>**66.33**(_↑_1.28%)|
|||||



## A.5 FURTHER VISUALIZATIONS: ORIGINAL VS. RD-MODULE MODELS 

We present additional examples in Figures 10 and 11, illustrating input images, the outputs from the original model, and the outputs from the model with the RD module. The results demonstrate that the RD-Model outputs are noticeably clearer. For example, in Figure 10 (ID 1 and 2), the edges of objects are significantly sharpened. Similarly, in Figures 10 and 11 (ID 3, 4, 5, and 6), our model exhibits higher accuracy and fewer false positives in the object’s bounding boxes, as indicated by the red arrows. 

A.6 PSEUDO CODE OF FULL TRAINING PROCESS OF _**R** etriever_ - _**D** ictionary_ MODEL 

The complete training process, from initialization to final model, follows the pseudo-code provided in Algorithm 1. This process includes _**D** ictionary_ initialization, regular model training, and _**D** ictionary_ compression. The overall training time is approximately equivalent to the original training epochs, with an additional 2 epochs allocated for setup and compression. 

## A.7 VISUALIZATION OF INITIAL DISTRIBUTIONS ACROSS DIFFERENT MODALITY MODELS 

Figure 12 and 13 visualizes the t-SNE distributions of VM, VLM, and LLM dictionaries. Vision and Language dictionaries occupy distinct regions, while Vision-Language overlaps with Vision. Notably, the Vision-Language dictionary is more uniformly distributed, showcasing its ability to provide richer information. 

A.8 DEEPER DISCUSSION OF THE _**R** etriever_ CORE 

**Two Convolutions Without Activation Functions.** Consider two consecutive 1 _×_ 1 convolutional layers without activation functions. The first layer has weights _W_[1] _∈_ R _[M][×][N]_ , and the second has weights _W_[2] _∈_ R _[N][×][M]_ , with an input **X** _∈_ R _[N][×][H][×][W]_ . 

For the first convolutional layer, the output at a spatial location ( _h, w_ ) is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0017-13.png)


17 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0018-01.png)


**----- Start of picture text -----**<br>
ID Ground Truth without RD with RD<br>1<br>2<br>3<br>**----- End of picture text -----**<br>


Figure 10: More visualization of RD 

18 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0019-01.png)


**----- Start of picture text -----**<br>
ID Ground Truth without RD with RD<br>4<br>5<br>6<br>**----- End of picture text -----**<br>


Figure 11: More visualization of RD 

19 

Published as a conference paper at ICLR 2025 

## **Algorithm 1:** Train a model with _**R** etriever_ _**D** ictionary_ 

**Data: D** ataset with images and bounding boxes **Result:** Trained model with _**R** etriever_ _**D** ictionary_ // Initialization of the _**D** ictionary_ **1 foreach** ( _img, box_ ) _∈_ **D** _ataset_ **do 2** _features ←_ encoder( _img_ ) **3** RD _←_ new _**R** etriever_ - _**D** ictionary_ (kmeans( _features_ )) **4** 

// Standard Training Method 

**5** backbone, head _←_ new Model() **6 for** _epoch e_ = 1 **to** _num_epochs_ **do 7 foreach** ( _img, box_ ) _∈_ **D** _ataset_ **do 8** _output ←_ head(RD(backbone( _img_ ))) **9** _loss ←_ loss_function( _output, box_ ) **10** update( _loss_ , (backbone, RD, head)) **11** 

// _**D** ictionary_ Compression **12** rd _←_ new _**R** etriever_ - _**D** ictionary_ (choice_from( _D_ )) **13 foreach** ( _img, box_ ) _∈_ **D** _ataset_ **do 14** freeze(backbone) **15** _teacher_  feature ←_ RD(backbone( _img_ ))) **16** _student_  feature ←_ rd(backbone( _img_ ))) **17** _loss ←_ cosine_similarity( _teacher_  feature, student_  feature_ ) **18** update( _loss_ , rd) **19** 

// Final Model: 

**20** FullModel _←_ merge(backbone, rd, head) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0020-08.png)


**----- Start of picture text -----**<br>
t-SNE Visualization of VLM, VM, and LLM<br>100 VLM<br>VM<br>75 LLM<br>50<br>25<br>0<br>25<br>50<br>75<br>100<br>100 75 50 25 0 25 50 75 100<br>t-SNE Dimension 1<br>(a) 2D distributions. (b) 3D distributions.<br>t-SNE Dimension 2<br>**----- End of picture text -----**<br>


Figure 12: The initial distributions of dictionaries derived from different modality models. 

20 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0021-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0021-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0021-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0021-04.png)


**----- Start of picture text -----**<br>
(a) x-axis (b) y-axis (c) z-axis<br>**----- End of picture text -----**<br>


Figure 13: Orthographic projections of the initial distributions using 3D t-SNE. Each subfigure represents a different axis projection to better illustrate the structure of the distributions. 

where _W_[1] is the weight of the first convolution, _n ∈_ [0 _, N_ ) is the channels, and the second convolution’s output is as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0021-07.png)


Thus, the equivalent weight matrix is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0021-09.png)


The original two-layer architecture has 2 _NM_ parameters. The equivalent layer has _N_[2] parameters. Without an activation function, the number of parameters in the combined layer is wasted by 2 _NM − N_[2] . Even if _M <[N]_[the][rank][of] _[W]_[ 1] _[W]_[ 2][will][be][limited][by][min(] _[M, N]_[)][,][leading][to][a] 2[,] significant drop in abilities of the network. Therefore, activation functions are essential in most cases to maintain the representational capacity of the sequential convolutional layers. 

**Forward Pass with the** _**Retriever**_ **Core.** We now extend this discussion to our _**R** etriever_ core, consisting of two convolutional layers: a pointwise (1 _×_ 1) convolution for channel projection and a depthwise convolution, with the number of groups equal to the number of channels. And there is no activation function between layers. 

For the pointwise convolution (first layer), we compute: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0021-13.png)


where the _Wc,i[G]_[is the pointwise convolution weight, as well as the Coefficient Generator weight.][For] the depthwise convolution (second layer), the operation is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0021-15.png)


21 

Published as a conference paper at ICLR 2025 

where the _Wc,m,n[E]_[is the depthwise convolution weight, as well as Global Information Exchanger,] _[ c]_[ is] the channel dimension. Since no non-linear operation is applied between these two layers, we can combine them into a single equivalent convolution. Substituting the output of the first layer into the second, we get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0022-02.png)


This results in a combined convolution operation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0022-04.png)


where _c_ , _i_ , _m_ , and _n_ represent the input channels, output channels, and two spatial dimensions of the kernels, respectively. _Wc,i,[G]_ 0 _,_ 0[denotes the pointwise convolution (with both spatial indices] set to 0). The resulting equivalent kernel has a size of _k × k_ , with equivalent weights _W_[eq] _∈_ R _[N][×][f][×][k][×][k]_ , where _f_ is the input feature dimension and _N_ is the number of output channels. While this equivalent convolution maintains the functionality of the original two layers, the combined weights are computationally heavier due to the different dimensions involved (channel dimension and kernel size). Nevertheless, even without an activation function, the model behaves as a normal convolution operation. 

**Gradient Descent and Weight Update.** Following the forward pass, the weight update rules for the pointwise and depthwise convolutional layers can be expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0022-07.png)


where _eta_ represents the learning rate, and _W[G][′]_ and _W[G][′]_ denote the updated weights for the pointwise and depthwise convolutions, respectively. _**R** etriever_ core update is then given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0022-09.png)


Since _η_ (the learning rate) is generally small during training, especially when using FP16 precision, the second-order term can be ignored. The simplified weight update for the equivalent kernel becomes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0022-11.png)


This follows the structure of the Taylor expansion: 

this gives the updated equivalent weight, which closely approximates the equivalent convolution: 

22 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary_images/iclr_2025_YOLO-RD_Introducing_Relevant_and_Compact_Explicit_Knowledge_to_YOLO_by_Retriever-Dictionary.pdf-0023-01.png)


## A.9 NOTATIONS 

Table 10: Notation Reference Table for Symbols Used in the Paper 

|**Notation**|**Default Value**|**Description**|
|---|---|---|
|_D_|-|The**_D_**_ictionary_, composed of learned atoms|
|_N_|512|Number of atoms_α_in the**_D_**_ictionary D_|
|_α_|-|Each individual element (atom) in the**_D_**_ictionary D_|
|_αi_|-|The_i_-th atom in the**_D_**_ictionary D_|
|_W_|80|The width dimension of the input to the RD module|
|_H_|80|The height dimension of the input to the RD module|
|_X_|-|Input feature map to the RD module|
|_Xh,w_|-|The pixel value at position(_h, w_)of the input_X_|
|_f_|512|Dimensionality of each atom in the**_D_**_ictionary_|
|_k_|5|Global Information Exchanger kernel size|
|**_c_**|-|Coeffcient matrix before normalized used to weight atoms in the RD module|
|**_c_**_i,h,w_|-|Coeffcient value before normalized for the_i_-th atom in the**_D_**_ictionary D_, at pixel(_h, w_)|
|**_c_**_′_<br>_i,h,w_|-|Normalized coeffcient value for the_i_-th atom in the**_D_**_ictionary D_, at pixel(_h, w_)|
|_zϕ_<br>_i,h,w_|-|The feature of generate by backbone and module_ϕ_for the_i_-th mini batch, at pixel(_h, w_)|
|_λ_|0.8|Residual weight in the forward pass of the RD module|
|**G**(_·_)|-|Coeffcient Generator function in the RD module|
|_W G_|-|Convolutional matrix used in the Coeffcient Generator**G**(_·_)|
|**E**(_·_)|-|Global Information Exchanger function in the RD module|
|_W E_|-|Convolutional matrix used in the Global Information Exchanger**E**(_·_)|



23 

