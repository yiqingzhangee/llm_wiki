# - - MIND THE GAP: GLIMPSE BASED ACTIVE PERCEP - TION IMPROVES GENERALIZATION AND SAMPLE EFFI CIENCY OF VISUAL REASONING 

**Oleh Kolner, Thomas Ortner, Stanisław Wo´zniak & Angeliki Pantazi** IBM Research Europe - Zurich Zurich, Switzerland olk@zurich.ibm.com 

## ABSTRACT 

Human capabilities in understanding visual relations are far superior to those of AI systems, especially for previously unseen objects. For example, while AI systems struggle to determine whether two such objects are visually the same or different, humans can do so with ease. Active vision theories postulate that the learning of visual relations is grounded in actions that we take to fixate objects and their parts by moving our eyes. In particular, the low-dimensional spatial information about the corresponding eye movements is hypothesized to facilitate the representation of relations between different image parts. Inspired by these theories, we develop a system equipped with a novel Glimpse-based Active Perception (GAP) that sequentially glimpses at the most salient regions of the input image and processes them at high resolution. Importantly, our system leverages the locations stemming from the glimpsing actions, along with the visual content around them, to represent relations between different parts of the image. The results suggest that the GAP is essential for extracting visual relations that go beyond the immediate visual content. Our approach reaches state-of-the-art performance on several visual reasoning tasks being more sample-efficient, and generalizing better to outof-distribution visual inputs than prior models. 

## 1 INTRODUCTION 

Humans use vision both to detect objects and to analyze various relations between them. For example, to successfully grasp an object from a table, we should first see whether some other object is not lying on top of it. Such relations can be spatial, like in the aforementioned example, or more abstract, e.g. if two objects are the same or different. While humans can easily recognize many kinds of relations, even between previously unseen objects (Fleuret et al., 2011; Kotovsky & Gentner, 1996), a growing body of literature suggests that artificial neural networks (ANNs) struggle with this (Kim et al., 2018; Ricci et al., 2021; Puebla & Bowers, 2022; 2024). Typically, ANNs process visual information holistically with each part of an image being processed in parallel with the others. In contrast, humans use active vision by selectively moving their eyes to glimpse at salient and/or task-relevant parts of the image and to process those parts in high resolution (Yarbus, 1967; Hayhoe & Ballard, 2005; Land, 2009). Moreover, by moving the eyes, the brain is able to form representations of the corresponding glimpse locations. This process can also be seen as a factorization of the visual scene into its visual (“what”) and spatial (“where”) contents (Goodale & Milner, 1992; Behrens et al., 2018). It is hypothesized that the low-dimensional representation and the relational geometry between those locations form a scaffold for representing the image structure, i.e. relations between different image parts (Summerfield et al., 2020). 

Currently, the best-performing ANNs, proposed for visual reasoning, use completely different mechanisms compared to humans (Webb et al., 2024a; Mondal et al., 2024). Specifically, they combine an object-centric representation using slot attention (Locatello et al., 2020) and a special inductive bias called relational bottleneck (Webb et al., 2024b). The former aims at decomposing an image into a set of representations corresponding to its objects and the latter encourages the model to focus on the relations between objects’ representations instead of the representations themselves. While showing 

1 

compelling performance on several visual reasoning tasks, these ANNs struggle to generalize the learned rules to out-of-distribution (OOD) objects (Puebla & Bowers, 2024). A likely reason is that the methods for object-centric representation cannot reliably capture the relations between different parts of objects. Instead, they merely perform perceptual segregation through clustering (Mehrani & Tsotsos, 2023). Inspired by active vision, an alternative model, proposed by Vaishnav & Serre (2023), processes images iteratively using a soft attention mechanism controlled by a memory module. However, in every iteration, this model still processes the image in its entirety only masking specific regions. Moreover, it lacks a low-dimensional spatial representation of task-relevant regions. 

Drawing more inspiration from humans’s active vision, and the aforementioned theories behind it, we introduce a model equipped with a novel Glimpse-based Active Perception (GAP) that sequentially glimpses at the most salient parts of the image. The GAP produces a dual sequence of glimpse locations and glimpse contents around them, promoting spatial “where” and visual “what” pathways. Importantly, the glimpse locations in the “where” pathway convey purely spatial information that is complementary to the visual contents of the “what” pathway. In particular, the geometrical relations between those locations provide a basis for extracting structural aspects of the image that can also be present in an image with different visual contents. The GAP can be paired with various downstream architectures that can process the dual sequence to solve the task at hand. In this work, we use a Transformer (Vaswani et al., 2017) and its recently introduced extension, called Abstractor (Altabaa et al., 2024). We evaluate our model on four visual reasoning datasets testing OOD generalization capabilities and sample efficiency. Achieving state-of-the-art performance, our model provides computational support for the active vision theories. 

Our contributions can be summarized as follows: 

- We present a brain-inspired active vision system for visual reasoning that extracts structural aspects of images by sequentially glimpsing at their salient parts. 

- We show that our system outperforms state-of-the-art models on several benchmarks in terms of sample efficiency and OOD generalization. 

- Our model is amenable to the integration of more advanced components for both saliency detection and downstream architecture which allows further scaling and extension. 

- Our results support cognitive science theories postulating that the brain factorizes images into their visual and spatial contents with the latter being crucial for OOD generalization. 

## 2 APPROACH 

We introduce a model that learns to extract relations between different parts of an image by factorizing it into its visual (“what”) and spatial (“where”) contents. This is achieved by sequentially glimpsing at the most salient image parts that comprise the visual content whereas the locations of those parts provide the spatial content. Our approach is illustrated in Figure 1. Given an input image, our model employs a novel concept of error neurons to extract a saliency map where stronger activities of the error neurons correspond to more salient parts, such as corners of the shapes presented in the image. The saliency map is then used to guide the glimpsing process by producing locations to the highest saliency, referred to as glimpse locations, in a winner-takes-all (WTA) fashion. To prevent re-visiting previous glimpse locations, an inhibition-of-return (IoR) mechanism updates the saliency map by masking out regions around the previously visited locations. These two steps are inspired by (Itti et al., 1998). Each glimpse location is passed to a glimpse sensor that extracts the corresponding visual content around that location, referred to as glimpse content. The described process repeats for a pre-defined number of iterations, producing a sequence of glimpse contents and a sequence of glimpse locations. These two sequences represent the factorization of the image into its complementary “what” and “where” contents. Both sequences are processed by the downstream architecture that makes the final decision for the given the task. 

## 2.1 GLIMPSE-BASED ACTIVE PERCEPTION 

Glimpse-based Active Perception (GAP) is a process of acquiring visual and spatial information by steering a glimpse sensor to different parts of the visual scene. The steering is facilitated by a saliency map that highlights the most salient image parts. While there are many ways to compute such a saliency map, we use a simple but efficient concept of error neurons (Figure 2(a)) described 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Glimpse-based Active Perception<br>Glimpse contents<br>Glimpse<br>sensor "What"<br>Image<br>Glimpse Downstream<br>Error neurons location Glimpse locations Architecture<br>Saliency  WTA "Where"<br>map<br>Inhibition-of-return<br>**----- End of picture text -----**<br>


Figure 1: **Architecture overview.** Based on the input image, the layer of error neurons produces the saliency map where salient image parts are highlighted (brighter regions). Through WTA combined with inhibition-of-return, a sequence of glimpse locations is produced that covers the most salient regions. For each such location, the glimpse sensor extracts the glimpse content from the region around it. The sequences of glimpse locations and glimpse contents are processed by the downstream architecture to make the final decision about the presented task. 

in detail in Section 2.1.1. Based on the saliency map _**S** ∈_ R _[h][S][×][w][S]_ , _hS, wS_ denote the saliency map’s height and width, the model glimpses at the most salient regions for a pre-defined number of iterations _T_ . Specifically, at each iteration _t_ , the next glimpse location _**xt** ∈_ N[2] is selected through WTA competition returning the 2D location of the highest saliency: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0003-03.png)


where _**St**_ is the saliency map at iteration _t_ . To prevent the same location from being selected at subsequent iterations, the IoR mechanism applies a mask _**M**_ ( _**xt**_ ) _∈_ R _[h][S][×][w][S]_ around this location, updating the saliency map: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0003-05.png)


The mask _**M**_ ( _**xt**_ ) can be either a hard or a soft one. The hard mask consists of ones except for a round region of zeroes around the location _**xt**_ as shown in red in Figure 2(b). The size of that region is a hyper-parameter. In the soft mask, the values at each mask location _i, j_ are computed via _i_ the radial basis function with the Gaussian kernel _e[−][ϵ][||]_[(] _j_[)] _[−]_ _**[x][t]**[||]_[2] with _ϵ_ being a hyper-parameter and _i_ � _j_ � _∈_ N[2] a vector of location _i, j_ . 

Given a glimpse location _**xt**_ = ( _xt, yt_ ) _∈_ N[2] , the corresponding region – glimpse content – is extracted by sampling pixels from the area around it. This sampling process is conceived as obtaining information from a glimpse sensor that provides a limited view mimicking thereby a human eye that fixates different parts of the visual surroundings. The main goal of such a glimpse sensor is to strike a balance between the size of the glimpse content and the amount of information to be captured from the image. In this work, we implement two kinds of sensors: a multi-scale sensor and a log-polar sensor. First, the multi-scale sensor, introduced in (Mnih et al., 2014), extracts several patches of the same size _hg × wg_ , but with different resolutions, around the glimpse location, as illustrated in Figure 2(c) (three patches are pictured). The resulting glimpse content _**gt** ∈_ R _[s][g][×][h][g][×][w][g][×][c][I]_ is the concatenation of those patches where _sg_ is the number of different scales supported by the multi-scale sensor and _cI_ is the number of channels in the input image. The patch with the highest resolution corresponds to the patch cut out from the input image at the glimpse location in the original resolution. The other patches are of decreasing resolution and correspond to larger areas of the input image that are down-scaled to _hg × wg_ , allowing to capture coarse information of the surround of the glimpse location. Second, the log-polar glimpse sensor is inspired by the human foveal vision (Schwartz, 1977; Weiman & Chaikin, 1979; Javier Traver & Bernardino, 2010) and samples pixels according to a distribution defined by a log-polar grid, illustrated in Figure 2(d). This means that more pixels are sampled closer to the glimpse location, and fewer pixels are sampled farther from it, producing a glimpse content _**gt** ∈_ R _[h][g][×][w][g][×][c][I]_ whose size is equal to a single image patch. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
(a) (b)<br>Saliency map extraction with error neurons<br>Saliency<br>map Inhibition of return<br>Glimpsing<br>process<br>Image<br>time<br>(c) (d)<br>Low<br>Multi-scale  resolution Log-polar<br>glimpse sensor glimpse sensor<br>Medium<br>resolution<br>High<br>resolution<br>**----- End of picture text -----**<br>


Figure 2: **Components of Glimpse-based Active Perception** . (a) Error neurons: a sample error neuron, marked by the blue circle in the saliency map on the right, computes the difference between the visual content in its central receptive field (blue square) and the visual content in its surroundings (area shaded in red). (b) Glimpsing process: at each iteration the WTA mechanism selects a location of the highest value in the saliency map (corresponding to its brightest regions) that is passed to the glimpse sensor to extract the corresponding glimpse content (blue circles). Afterward, the inhibitionof-return mechanism masks out the selected location in the saliency map (shown by red circles) so that it does not get selected again. (c) Multi-scale glimpse sensor outputs a concatenation of several patches (three are shown) of the same size each of which encompasses a region of different size and resolution around the glimpse location on the left (depicted by three grids). (d) Log-polar glimpse sensor samples pixels according to the log-polar grid shown on the left providing high resolution towards its central region and low resolution towards the periphery. After the sampling process, the polar grid is warped into a regular rectangular grid which results in the glimpse content on the right. 

Using these procedures, the GAP ultimately produces a sequence of glimpse contents _**g**_ **1** _,_ _**g**_ **2** _, . . . ,_ _**gT**_ and a sequence of glimpse locations _**x**_ **1** _,_ _**x**_ **2** _, . . . ,_ _**xT**_ , that are further passed to the downstream architecture described in Section 2.2. 

## 2.1.1 SALIENCY MAP EXTRACTION WITH ERROR NEURONS 

In the primary visual cortex, it is observed that the activity of neurons is influenced not only by stimuli from their local receptive fields but also by stimuli that come from the surroundings of those receptive fields (Vinje & Gallant, 2000; Keller et al., 2020). From a theoretical view, this influence can make the same visual stimulus be perceived as more or less salient, depending on how significantly it differs from its local surroundings (Treisman & Gelade, 1980). Inspired by this, we propose a computational block called error neurons, illustrated in Figure 2(a), to compute the saliency map _**S** ∈_ R _[h][S][×][w][S]_ . Specifically, at each location _i, j_ of the saliency map _**S**_ the corresponding error neuron computes the saliency as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0004-05.png)


where _**P** ij_ _**[I]**[∈]_[R] _[h][P][ ·][w][P][ ·][c][I]_[is a flattened patch obtained from the input image] _**[ I]**[∈]_[R] _[h][I][×][w][I][×][c][I]_[around] location _i, j_ . The patch size, _hP × wP_ , is a hyper-parameter, and _hI , wI , cI_ denote the input image’s height, width, and the number of channels. Unless specified otherwise, _hI_ = _hS_ and _wI_ = _wS_ . _D_ ( _·, ·_ ) is a distance function, in this work, set to L2 distance. Agg[ _·_ ] is an aggregation operator such as sum or minimum over distances between the patch at location _i, j_ and the patches from surrounding locations _l, k ∈_ surr( _i, j_ ). The set of surrounding locations, surr( _i, j_ ), can be defined arbitrarily, comprising, for example, all available image locations. In this work, we assume the 8- connected neighborhood of location _i, j_ . Intuitively, each value _**S** ij_ represents how much the visual content around location _i, j_ differs (or stands out) from the surrounding visual content. 

4 

## 2.2 DOWNSTREAM ARCHITECTURE 

As detailed above, the GAP provides a sequence of glimpse contents _**g**_ and a sequence of the corresponding 2D glimpse locations _**x**_ to the downstream architecture. Importantly, our approach is flexible with respect to the choice of the downstream architecture. In this work, we explore two options: Transformers (Vaswani et al., 2017) and Abstractors (Altabaa et al., 2024). 

Transformers have emerged over recent years as popular state-of-the-art architectures for a variety of tasks, including the vision domain with the Vision Transformers (ViT) Dosovitskiy et al. (2021). Apart from their successes in machine learning tasks, there have also been works suggesting potential links between ViT and biology (Pandey et al., 2023). The ViT processes a sequence of image patches, augmented with positional encodings, to solve the given task. In our case, we provide the sequence of glimpse contents augmented with the respective glimpse locations as inputs. 

The Abstractor has been recently introduced as an extension of the Transformer, with the key difference of the so-called Relational Cross Attention (RCA) module. It was designed to disentangle relational information from the perceptual one to enable generalization from limited data. This is achieved by modifying the self-attention so that the queries and the keys are produced based on perceptual inputs whereas the values are provided by a separate independent set of representations. The latter, in our case, is substituted by the sequence of glimpse locations whereas the perceptual inputs comprise the sequence of glimpse contents. Hence, adopting the Abstractor allows for the explicit processing of spatial information so that the visual information carries only a modulatory effect in form of attention. The formal description of the Abstractor is provided in Appendix A. 

## 3 RELATED WORK 

Inspired by human active vision, several models for learning visual relations have been introduced. By decomposing input images into a sequence of image regions obtained from synthetic eye movements, Wo´zniak et al. (2023) demonstrated that visual reasoning tasks can be solved with high accuracy using relatively compact networks. However, the synthetic eye movements were based on hard-coded locations, whereas in our case, the glimpsing process is controlled by a saliency map. Aiming primarily at computationally efficient image classification, Mnih et al. (2014) used reinforcement learning to train a glimpsing policy. The policy iteratively determines the next 2D glimpse location based on the history of previously visited locations and the corresponding glimpse contents. However, having been evaluated on several visual reasoning tasks by Vaishnav & Serre (2023), this model could not achieve reasonable performance. This is likely caused by learning inefficiency in exploring the space of all possible 2D glimpse locations. Our approach, in contrast, leverages the concept of salience maps to determine the next glimpse location which significantly reduces the space of glimpse locations to the most salient ones. Gregor et al. (2015) proposed a variational auto-encoder that uses an attention mechanism for an iterative (re-)construction of complex images by attending to their different parts. Its extension proposed by Adeli et al. (2023) was designed to tackle visual reasoning tasks where it outperformed baseline CNNs. However, being trained by image reconstruction, the model does not always generalize well to OOD visual inputs. Vaishnav & Serre (2023) introduced a recurrent soft attention mechanism guided by a memory module that masks specific parts of the image where the processing should be focused. At each iteration, the model can thereby attend to multiple regions of the image simultaneously which is different from our approach where only one region can be focused at a time. 

Another family of models combines object-centric representations (Greff et al., 2019; Locatello et al., 2020) with relational inductive biases (Webb et al., 2021; Kerg et al., 2022; Webb et al., 2024b). The former decomposes an image into objects present in it while the latter constrains the processing to relations between objects instead of their specific properties. Webb et al. (2024a) used pre-trained slot attention (Locatello et al., 2020) to segment objects into a sequence of slots that contain disentangled representations of objects’ visual features and their positional information. Each pair of slots is then processed by a relational operator and the resulting sequence of relational representations is processed by the Transformer (Vaswani et al., 2017). The successor model, Slot-Abstractor (Mondal et al., 2024), processes the sequence of slots with a recently introduced Abstractor module (Altabaa et al., 2024) that proved to be especially effective in reasoning about relations between objects using their disentangled visual and positional features. Similar to our approach, there are two streams of information: one based on visual features and one based on 

5 

positional information. However, the latter stream consists of positional embeddings that are learned during the pre-training process which may make the model more fragile to OOD inputs. In contrast, our approach uses the raw 2D locations of salient regions. 

## 4 EXPERIMENTS 

**Datasets** Our approach was evaluated on four visual reasoning datasets illustrated in Figure 3. We start with an evaluation on all 23 tasks of the SVRT dataset (Fleuret et al., 2011) to see how efficiently the model can learn spatial and similarity relations between simple shapes, see an example in Figure 3(a). Next, we test the model’s OOD capabilities to generalize learned relations to previously unseen shapes. Specifically, we use an extended version of the task #1 from the SVRT dataset (Puebla & Bowers, 2022), referred to as SVRT #1-OOD, see Figure 3(b). It contains test sets with OOD shapes coming from 13 different distributions. Additionally, we use the ART dataset (Webb et al., 2021), which contains more abstract relations, see Figure 3(c). Finally, we test the model’s potential to scale to more complex images by testing it on the CLEVR-ART dataset (Webb et al., 2024a), see Figure 3(d). It contains two tasks from ART but with more realistic images. Each dataset is described in more detail in Appendix B. 

**Baselines** We compare the impact of our approach with the plain downstream architectures that receive entire images as inputs. The images get split into non-overlapping patches and each patch gets augmented by its positional information as is done by ViTs (Dosovitskiy et al., 2021). We also tested those models where the patches overlapped but did not observe any significant performance differences. Therefore, we only report the standard approach with non-overlapping patches. We refer to these models simply as _Transformer_ and _Abstractor_ . The comparison with these baselines allows us to demonstrate the importance of processing only the relevant patches obtained from the glimpsing process rather than all patches that constitute an image. We also compare our approach with GAMR (Vaishnav & Serre, 2023), ResNet (He et al., 2015), its extension Attn-ResNet (Vaish- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0006-04.png)


**----- Start of picture text -----**<br>
(a) SVRT (b) SVRT #1-OOD<br>Category 1 Category 2 SVRT #1  Test sets<br>(training set) Filled Lines Arrows<br>(c) ART  (d) CLEVR-ART<br>Same<br>Different<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0006-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0006-06.png)


Figure 3: **Four datasets used in our experiments.** (a) SVRT dataset consists of 23 binary classification tasks where the model has to determine whether shapes are in a certain spatial and/or similarity relation to each other. Shown are images from the task of determining whether the same triplets of shapes exist in each image pair. (b) SVRT #1-OOD instantiates the first task from SVRT where it has to be determined whether two shapes are the same up to a translation. While this dataset contains training images from the original SVRT dataset, it contains 13 additional test sets (only three are shown here) with novel shapes to evaluate OOD generalization. (c) ART dataset contains four tasks where the model has to determine whether shapes in each two rows are arranged under the same abstract rule. (d) CLEVR-ART contains two tasks from ART but with more complex images. 

6 

nav et al., 2022), OCRA (Webb et al., 2024a) and Slot-Abstractor (Mondal et al., 2024). To the best of our knowledge, these baselines are the best-performing models on the tasks that we consider. 

## 5 RESULTS 

In all experiments, we report the performance of GAP combined with two downstream architectures, Transformer and Abstractor. The corresponding models are referred to as _GAP-Transformer_ and _GAP-Abstractor_ respectively. Details about hyper-parameters, such as the type of glimpse sensor (multi-scale or log-polar), and training process are described in Appendix C. Moreover, extensive performance reports for different glimpse sensors are included in Appendix D. 

## 5.1 VISUAL REASONING PERFORMANCE AND SAMPLE EFFICIENCY 

We evaluate the visual reasoning capabilities using the SVRT dataset that consists of 23 tasks, 11 of which require to reason about same-different relations (SD tasks) and the remaining 12 tasks require to reason about spatial relations (SR tasks). Consistent with (Vaishnav & Serre, 2023; Webb et al., 2024a; Mondal et al., 2024), Table 1 shows accuracy performance for models trained with 1000 and 500 samples. Our best-performing model, the GAP-Abstractor, outperforms the prior art models. In particular, it achieves very high accuracy on the SD tasks, considered to be more challenging than the SR tasks, for which it performs comparably with the prior art. It should be noted that although for the SR tasks, the best-performing model is GAMR, its performance on SD tasks is significantly lower compared with all other models. Detailed accuracy for each individual task is provided in Appendix, Figure D.5. Furthermore, Figure 4 shows an extensive evaluation of sample efficiency, 

Table 1: **Test accuracy for SVRT** . The results obtained from the best trained models are averaged over SD, SR and over all (SD + SR) tasks. The standard deviation is calculated over accuracies for each individual task. Prior art results are listed in the top section whereas results obtained from our evaluations are listed in the two bottom sections. The highest and the second-highest mean accuracy in each column is marked in **bold** and underlined, respectively. The models appended by * contain a module pre-trained on the train sets from all SVRT tasks to segment the shapes. 

|Model|SD tasks accuracy [%]<br>Dataset size<br>1000<br>Dataset size<br>500|SR tasks accuracy [%]<br>Dataset size<br>1000<br>Dataset size<br>500|SR tasks accuracy [%]<br>Dataset size<br>1000<br>Dataset size<br>500|**All tasks accuracy [%]**<br>Dataset size<br>1000<br>Dataset size<br>500|
|---|---|---|---|---|
|ResNet<br>Attn-ResNet<br>GAMR<br>OCRA*<br>Slot-Abstractor*|56.9 ± 2.5<br>54.9 ± 2.2<br>68.8 ± 4.4<br>62.3 ± 3.5<br>82.1 ± 8.4<br>76.8 ± 8.7<br>90.3 ± 4.1<br>79.9 ± 4.5<br>91.9 ± 4.0<br>82.2 ± 4.7|94.9 ± 1.6<br>97.7 ± 0.7<br>**98.7 ± 0.3**<br>95.0 ± 2.4<br>97.3 ± 1.1|85.2 ± 4.3<br>94.8 ± 1.4<br>**97.4 ± 0.7**<br>89.3 ± 2.5<br>91.8 ± 2.2|76.7 ± 2.0<br>70.7 ± 3.3<br>83.9 ± 2.5<br>79.3 ± 2.4<br>90.8 ± 4.2<br>87.6 ± 4.6<br>92.8 ± 3.2<br>84.8 ± 3.5<br>94.7 ± 2.5<br>87.2 ± 3.4|
|Transformer<br>GAP-Transformer|51.9 ± 2.6<br>51.6 ± 1.7<br>83.3 ± 13.2<br>78.4 ± 13.5|64.8 ± 13.7<br>98.2 ± 2.2|59.9 ± 11.0<br>97.3 ± 2.7|56.4 ± 10.3<br>55.9 ± 8.9<br>91.1 ± 11.8<br>88.3 ± 13.4|
|Abstractor<br>GAP-Abstractor|51.3 ± 2.5<br>51.0 ± 1.9<br>**93.1 ± 13.1**<br>**90.5 ± 14.5**|60.9 ± 14.3<br>98.5 ± 1.9|57.0 ± 11.0<br>96.6 ± 1.2|54.6 ± 9.7<br>54.1 ± 8.5<br>**95.4 ± 8.9**<br>**93.9 ± 10.1**|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0007-07.png)


**----- Start of picture text -----**<br>
(a) (b)<br>All tasks All tasks<br>100 100<br>GAP-Abstractor<br>90 Slot-Abstractor<br>90 GAMR<br>80 Attn-ResNet<br>ResNet<br>70<br>80 GAP-Transformer<br>60 Abstractor<br>Transformer<br>70 50<br>Dataset size Dataset size<br>100005000 1000 500 250 100 100005000 1000 500 250 100<br>Accuracy %<br>**----- End of picture text -----**<br>


Figure 4: **Sample efficiency evaluation for SVRT** . Average test accuracy on all 23 SVRT tasks depending on the size of the training dataset. Panel (a) compares our best model with prior art. Panel (b) compares our glimpse-based models with their ablated counterparts. 

7 

Table 2: **OOD performance for SVRT #1OOD** . The accuracy results are averaged over all OOD test sets for 10 trained models. Prior art results are listed in the top section whereas results obtained from our evaluations are listed in the two bottom sections. The highest and the second-highest mean accuracy is marked in **bold** and underlined, respectively. The models appended by * contain a module pre-trained on the train set to segment the shapes. 

|Model<br>Accuracy [%]<br>(averaged over runs<br>and OOD datasets)|Model<br>Accuracy [%]<br>(averaged over runs<br>and OOD datasets)|
|---|---|
|ResNet<br>GAMR<br>OCRA*<br>Slot-Abstractor*|71.2 ± 17.7<br>79.8 ± 13.1<br>73.9 ± 17.2<br>72.8 ± 16.0|
|Transformer<br>GAP-Transformer|50.0 ± 0.1<br>76.3 ± 18.4|
|Abstractor<br>GAP-Abstractor|66.2 ± 16.0<br>**89.6 ± 8.7**|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0008-02.png)


**----- Start of picture text -----**<br>
irregular<br>regular connectedcircles<br>90<br>connected<br>open 80<br>squares<br>70<br>60<br>wider straight<br>line lines<br>scrambled rectangles<br>random arrows<br>color<br>filled lines<br>GAMR<br>Slot-Abstractor<br>GAP-Transformer<br>GAP-Abstractor<br>**----- End of picture text -----**<br>


Figure 5: **Accuracy on each of the SVRT #1OOD test sets** . The accuracy results (in %) are averaged over 10 trained models. Slot-Abstractor could not be properly evaluated on ’random color’ test set that contains RGB images because its pretrained component handles only grayscale images. 

in which we trained the models with different amounts of training data beyond those reported in Table 1 and in prior works. Compared with the results we obtained for the prior art models, our best model exhibits higher sample efficiency, especially in scenarios with fewer than 1000 training samples, see Figure 4(a). Moreover, we observe favorable trends and improvements when applying GAP to either of the downstream architectures, see Figure 4(b). More detailed results with separate performance for SD and SR tasks are provided in Appendix, Figure D.6. 

## 5.2 OUT-OF-DISTRIBUTION GENERALIZATION OF SAME-DIFFERENT RELATION 

While the SVRT dataset allows us to see whether a model can learn visual relations from a limited amount of training data, it does not allow us to test the model’s ability to generalize those relations to OOD inputs. We evaluate the OOD generalization of a single same-different relation using the SVRT #1-OOD dataset (Appendix B.2) where a model has to determine whether two shapes are the same or different up to translation. Importantly, the models are evaluated on test sets with OOD shapes coming from 13 different distributions (see Figure B.2 for examples). Table 2 shows the performance averaged over all OOD test sets and Figure 5 shows the performance on each of the test sets. Outperforming the prior art models, our model, GAP-Abstractor, performs quite consistently on all test sets being slightly less accurate on only two sets (’straight ~~l~~ ines’ and ’rectangles’). 

Table 3: **Test accuracy for ART** . The accuracy results are averaged over 10 trained models. Prior art results are listed in the top section whereas results obtained from our evaluations are listed in the two bottom sections. The highest and the second-highest mean accuracy in each column is marked in **bold** and underlined, respectively. The models appended by * contain a module to segment the shapes that was pre-trained on images with shapes distinct from those in the ART dataset. 

|Model|SD|Task accuracy [%]<br>RMTS<br>Dist3|Task accuracy [%]<br>RMTS<br>Dist3|ID|
|---|---|---|---|---|
|GAMR<br>ResNet<br>OCRA*<br>Slot-Abstractor*|83.5 ± 1.4<br>66.6 ± 1.5<br>87.9 ± 1.3<br>96.4 ± 0.4|72.2 ± 3.0<br>6<br>49.9 ± 0.2<br>5<br>85.3 ± 2.0<br>8<br>91.6 ± 1.6<br>9|8.6 ± 1.8<br>6<br>0.1 ± 1.3<br>5<br>6.4 ± 1.3<br>9<br>5.2 ± 0.4<br>9|6.2 ± 4.3<br>4.8 ± 2.4<br>2.8 ± 0.3<br>6.4 ± 0.1|
|Transformer<br>GAP-Transformer|55.9 ± 2.1<br>76.4 ± 13.7|51.3 ± 0.1<br>2<br>62.2 ± 7.3<br>5|9.3 ± 1.2<br>3<br>0.7 ± 6.1<br>5|3.3 ± 1.7<br>5.5 ± 5.8|
|Abstractor<br>GAP-Abstractor|70.9 ± 6.6<br>**97.7 ± 2.0**|51.3 ± 0.2<br>5<br>**96.3 ± 1.0**<br>**9**|8.9 ± 3.8<br>4<br>**8.4 ± 0.5**<br>**9**|7.9 ± 1.4<br>**6.8 ± 1.8**|



8 

## 5.3 OUT-OF-DISTRIBUTION GENERALIZATION OF MORE ABSTRACT RELATIONS 

Going beyond evaluations of OOD generalization on a single task, we expand the evaluations to four tasks from the ART dataset, three of which contain more abstract relations (see Appendix B.3). Moreover, the evaluations on this dataset validate also the insights on sample efficiency since the models are required to be trained on very few samples. From Table 3 it can be seen that the GAPAbstractor outperforms all other models demonstrating superior capabilities to generalize abstract relations to novel visual inputs. Crucially, this performance improvement can be observed even though no component of our model was pre-trained, as opposed to OCRA or Slot-Abstractor that have been pre-trained to segment objects. 

## 5.4 SCALING POTENTIAL FOR MORE COMPLEX VISUAL SCENES 

We test our approach on two tasks from ART (RMTS and ID) but with more realistic images using the CLEVR-ART dataset. While the images from other datasets considered in this work are binary and contain only 2D objects, the CLEVR-ART images are colored and contain 3D objects with such real-world features as shades and reflections. Hence, the glimpsing behavior can be distracted by spurious salient locations such as edges of shades or regions of high reflection. Table 4 highlights that the GAPAbstractor performs on par with the state-of-theart model, Slot-Abstractor, being marginally better in one task and marginally worse in the other. However, Slot-Abstractor was pre-trained on all shapes from CLEVR-ART being thereby potentially exposed to objects from both train and test sets. This is in contrast to our models, which were not pre-trained and were exposed only to the objects from the train set. To compare our models with Slot-Abstractor on a more equal footing, we generated two additional OOD test sets shown in 

Table 4: **Test accuracy for CLEVR-ART** . The accuracy results are averaged over 5 trained models. Prior art results are listed in the top section whereas results obtained from our evaluations are listed in the two bottom sections. The highest and the second-highest mean accuracy in each column is marked in **bold** and underlined, respectively. The models appended by * contain a module to segment the shapes that was pre-trained on images containing shapes from both the train and test datasets. 

|Model|Task accuracy [%]<br>RMTS<br>ID|Task accuracy [%]<br>RMTS<br>ID|
|---|---|---|
|GAMR<br>OCRA*<br>Slot-Abstractor*|70.4 ± 5.8<br>93.3 ± 1.0<br>**96.3 ± 0.5**|74.2 ± 4.0<br>77.1 ± 0.7<br>91.6 ± 0.2|
|Transformer<br>GAP-Transformer|64.9 ± 3.4<br>76.4 ± 4.3|52.2 ± 4.3<br>64.7 ± 5.6|
|Abstractor<br>GAP-Abstractor|77.8 ± 12.2<br>95.9 ± 1.4|82.2 ± 3.8<br>**93.4 ± 1.3**|



Figure 6. The new OOD-1 test set contains one novel type of shape – pyramid – on which the SlotAbstractor was not pre-trained. Similarly, the new OOD-2 test set contains 2 such novel shapes: pyramids and toruses. The corresponding results are shown in Figure 7 demonstrating superior OOD generalization of our GAP-based model compared with Slot-Abstractor. Notably, the latter performs significantly worse, if the images contain multiple objects on which it was not pre-trained. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0009-07.png)


**----- Start of picture text -----**<br>
 Training set<br>✔cubes<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0009-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0009-09.png)


**----- Start of picture text -----**<br>
New  OOD-1  test set New  OOD-2  test set<br>✔spheres spheres  ✘pyramids pyramids  ✘pyramids  ✘toruses<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0009-10.png)


**----- Start of picture text -----**<br>
Original  OOD test set New  OOD-1  test set<br>✔cubes  ✔spheres  ✔spheres spheres  ✘pyramids pyramids<br>✔ Slots were pre-trained ✘ No pre-training<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0009-11.png)


Figure 6: **Extension to CLEVR-ART for more thorough evaluation of OOD generalization** . Examples for the RMTS task are shown here. The two new OOD test sets contain objects on which neither of the models was pre-trained: OOD-1 test set contains 1 novel object (pyramid) and OOD-2 contains 2 novel objects (pyramid and torus). 

9 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0010-00.png)


**----- Start of picture text -----**<br>
RMTS task ID task<br>100<br>80<br>60<br>Original OOD-1 OOD-2 Original OOD-1 OOD-2<br>Slot-Abstractor GAP-Abstractor<br>96.3 95.9 94.6 93.2 91.6 93.4<br>88.9 86.5 88.2 87.2<br>Accuracy % 73.4<br>66.6<br>**----- End of picture text -----**<br>


Figure 7: **Extended OOD evaluation on CLEVR-ART** . While the Slot-Abstractor was pre-trained on the original test set, the OOD-1 and OOD-2 test sets contain shapes on which neither of the evaluated models was pre-trained. The accuracy results are averaged over 5 trained models. 

## 6 DISCUSSION 

Our results suggest that factorizing an image into its complementary “what” and “where” contents plays an essential role. The low-dimensional “where” content (glimpse locations) is crucial since it does not contain any visual information and, in turn, allows to learn relations that generalize to OOD images. To capitalize on that, it is important to process the “where” content explicitly, disentangling it from its “what” complement, so that the latter carries only a modulatory effect in form of attention. This is achieved by employing the recently introduced Abstractor as our downstream architecture. In contrast, implicitly mixing the “what” and the “where” contents, as in the case of the Transformer downstream architecture, weakens the generalization capabilities. This can be seen by comparing the performance between GAP-Transformer and GAP-Abstractor. To further support this, we provide supplementary results in Appendix D.1 showing the importance of using both the “what” and the “where” contents instead of just one of them for task solving. 

Another important aspect of our model is that it processes only the most salient image regions while ignoring the unimportant ones. The inferior performance of models where the downstream architectures receive full information (i.e. all patches that constitute the image) suggests that supplying a superfluous amount of information may distract the models hindering the learning process. Additional results provided in Appendix D.2 elucidate this further. Specifically, we show that it is insufficient to naively reduce the amount of information provided to the downstream architecture, by discarding uninformative image parts. Instead, it has to be ensured that the supplied information is well structured. In the context of GAP, it means that the glimpse contents have to be centered around salient image regions instead of, for example, being picked from a set of non-overlapping patches obtained by splitting the image using a regular grid (as is done in ViT-based models). 

## 7 CONCLUSION 

Drawing inspiration from human eye movements and theories of active vision, we proposed a system equipped with the novel Glimpse-based Active Perception (GAP). The system selectively glimpses at the most salient image parts and processes them in high resolution. The relational geometry between the corresponding glimpse locations complemented by the glimpse contents provides a basis for reasoning about image structure, i.e. relations between different image parts. We evaluated our approach on four visual reasoning benchmarks and achieved state-of-the-art performance in terms of sample efficiency and the ability to generalize to OOD visual inputs without any pre-training. Our approach allows for further extension by integrating more advanced components. Particularly, the saliency map extraction can be substantially enhanced by either extending our concept of error neurons or integrating more powerful components. 

## REFERENCES 

Hossein Adeli, Seoyoung Ahn, and Gregory J Zelinsky. A brain-inspired object-based attention network for multiobject recognition and visual reasoning. _Journal of Vision_ , 23(5):16–16, 2023. 

10 

- Awni Altabaa, Taylor Whittington Webb, Jonathan D. Cohen, and John Lafferty. Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in transformers. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Timothy E.J. Behrens, Timothy H. Muller, James C.R. Whittington, Shirley Mark, Alon B. Baram, Kimberly L. Stachenfeld, and Zeb Kurth-Nelson. What is a cognitive map? organizing knowledge for flexible behavior. 100(2):490–509, 2018. ISSN 08966273. 

- G. Bradski. The OpenCV Library. _Dr. Dobb’s Journal of Software Tools_ , 2000. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ , 2021. 

- Franc¸ois Fleuret, Ting Li, Charles Dubout, Emma K. Wampler, Steven Yantis, and Donald Geman. Comparing machines and humans on a visual categorization test. 108(43):17621–17625, 2011. ISSN 0027-8424, 1091-6490. 

- Melvyn A Goodale and A David Milner. Separate visual pathways for perception and action. _Trends in Neurosciences_ , 15(1):20–25, 1992. 

- Klaus Greff, Rapha¨el Lopez Kaufman, Rishabh Kabra, Nick Watters, Christopher Burgess, Daniel Zoran, Loic Matthey, Matthew Botvinick, and Alexander Lerchner. Multi-object representation learning with iterative variational inference. In _International Conference on Machine Learning_ , pp. 2424–2433. PMLR, 2019. 

- Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Rezende, and Daan Wierstra. Draw: A recurrent neural network for image generation. In _International Conference on Machine Learning_ , pp. 1462–1471. PMLR, 2015. 

- Mary Hayhoe and Dana Ballard. Eye movements in natural behavior. _Trends in cognitive sciences_ , 9(4):188–194, 2005. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. _CoRR_ , abs/1512.03385, 2015. 

- L. Itti, C. Koch, and E. Niebur. A model of saliency-based visual attention for rapid scene analysis. 20(11):1254–1259, 1998. ISSN 1939-3539. Conference Name: IEEE Transactions on Pattern Analysis and Machine Intelligence. 

- V. Javier Traver and Alexandre Bernardino. A review of log-polar imaging for visual perception in robotics. 58(4):378–398, 2010. ISSN 09218890. 

- Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , 2017. 

- Andreas J Keller, Morgane M Roth, and Massimo Scanziani. Feedback generates a second receptive field in neurons of the visual cortex. _Nature_ , 582(7813):545–549, 2020. 

- Giancarlo Kerg, Sarthak Mittal, David Rolnick, Yoshua Bengio, Blake Aaron Richards, and Guillaume Lajoie. Inductive biases for relational tasks. In _ICLR2022 Workshop on the Elements of Reasoning: Objects, Structure and Causality_ , 2022. 

- Junkyung Kim, Matthew Ricci, and Thomas Serre. Not-so-clevr: learning same–different relations strains feedforward neural networks. _Interface focus_ , 8(4):20180011, 2018. 

- Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _International Conference on Learning Representations (ICLR)_ , San Diega, CA, USA, 2015. 

- Laura Kotovsky and Dedre Gentner. Comparison and categorization in the development of relational similarity. _Child development_ , 67(6):2797–2822, 1996. 

11 

Michael F Land. Vision, eye movements, and natural behavior. _Visual Neuroscience_ , 26(1):51–62, 2009. 

- Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention, 2020. 

- Paria Mehrani and John K Tsotsos. Self-attention in vision transformers performs perceptual grouping, not attention. _Frontiers in Computer Science_ , 5:1178450, 2023. 

- Volodymyr Mnih, Nicolas Heess, Alex Graves, and koray kavukcuoglu. Recurrent Models of Visual Attention. In _Advances in Neural Information Processing Systems_ , volume 27. Curran Associates, Inc., 2014. 

- Shanka Subhra Mondal, Jonathan D. Cohen, and Taylor Whittington Webb. Slot abstractors: Toward scalable abstract visual reasoning. In _Proceedings of the 41st International Conference on Machine Learning_ . PMLR, 21–27 Jul 2024. 

- Lalit Pandey, Samantha Wood, and Justin Wood. Are vision transformers more data hungry than newborn visual systems? In _Advances in Neural Information Processing Systems_ , 2023. 

- Guillermo Puebla and Jeffrey S. Bowers. Can deep convolutional neural networks support relational reasoning in the same-different task? 22(10):11, 2022. ISSN 1534-7362. 

- Guillermo Puebla and Jeffrey S. Bowers. The role of object-centric representations, guided attention, and external memory on generalizing visual relations, 2023. 

- Guillermo Puebla and Jeffrey S. Bowers. Visual reasoning in object-centric deep neural networks: A comparative cognition approach, 2024. 

- Matthew Ricci, R´emi Cad`ene, and Thomas Serre. Same-different conceptualization: a machine vision perspective. _Current Opinion in Behavioral Sciences_ , 37:47–55, 2021. 

- Eric L Schwartz. Spatial mapping in the primate sensory projection: analytic structure and relevance to perception. _Biological cybernetics_ , 25(4):181–194, 1977. 

- Christopher Summerfield, Fabrice Luyckx, and Hannah Sheahan. Structure learning and the posterior parietal cortex. _Progress in Neurobiology_ , 184:101717, January 2020. ISSN 03010082. 

- Anne M Treisman and Garry Gelade. A feature-integration theory of attention. _Cognitive psychology_ , 12(1):97–136, 1980. 

- Mohit Vaishnav and Thomas Serre. GAMR: A guided attention model for (visual) reasoning. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- Mohit Vaishnav, Remi Cadene, Andrea Alamia, Drew Linsley, Rufin VanRullen, and Thomas Serre. Understanding the computational demands underlying visual reasoning. _Neural Computation_ , 34 (5):1075–1099, 2022. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in Neural Information Processing Systems_ , 30, 2017. 

- William E. Vinje and Jack L. Gallant. Sparse coding and decorrelation in primary visual cortex during natural vision. _Science_ , 287(5456):1273–1276, 2000. 

- Taylor Webb, Zachary Dulberg, Steven Frankland, Alexander Petrov, Randall O’Reilly, and Jonathan Cohen. Learning representations that support extrapolation. In _International Conference on Machine Learning_ , pp. 10136–10146. PMLR, 2020. 

- Taylor Webb, Shanka Subhra Mondal, and Jonathan D Cohen. Systematic visual reasoning through object-centric relational abstraction. _Advances in Neural Information Processing Systems_ , 36, 2024a. 

12 

- Taylor W. Webb, Steven M. Frankland, Awni Altabaa, Simon Segert, Kamesh Krishnamurthy, Declan Campbell, Jacob Russin, Tyler Giallanza, Randall O’Reilly, John Lafferty, and Jonathan D. Cohen. The relational bottleneck as an inductive bias for efficient abstraction. _Trends in Cognitive Sciences_ , May 2024b. ISSN 1364-6613. 

- Taylor Whittington Webb, Ishan Sinha, and Jonathan Cohen. Emergent symbols through binding in external memory. In _International Conference on Learning Representations_ , 2021. 

- Carl FR Weiman and George Chaikin. Logarithmic spiral grids for image processing and display. _Computer Graphics and Image Processing_ , 11(3):197–226, 1979. 

- Stanisław Wo´zniak, Hlynur J´onsson, Giovanni Cherubini, Angeliki Pantazi, and Evangelos Eleftheriou. On the visual analytic intelligence of neural networks. _Nature Communications_ , 14(1): 5978, 2023. 

- Alfred L. Yarbus. _Eye Movements and Vision_ . Springer US, 1967. ISBN 9781489953797. 

13 

## A ABSTRACTOR 

To describe the Abstractor-based downstream architecture and the way it is applied, we follow (Mondal et al., 2024). Specifically, given a sequence of glimpse contents _**g**_ = ( _**g**_ **1** _,_ _**g**_ **2** _, . . . ,_ _**gT**_ ) and their locations _**x**_ = ( _**x**_ **1** _,_ _**x**_ **2** _, . . . ,_ _**xT**_ ), the Abstractor computes a sequence of the corresponding relational representations _**r**_ over a series of _L_ layers. The initial relational representations _**r** l_ =0 correspond to the glimpse locations _**x**_ . Each subsequent layer _l_ updates the relational representations using multi-head relational cross-attention between the glimpse contents _**g**_ (or their latent representations, if they are further pre-processed by a neural network): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0014-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0014-03.png)


where _**Wq**[h][,]_ _**[ W]**[ h]_ _**k**[,]_ _**[ W] v**[ h][∈]_[R] _[d][×][d]_[(] _[d]_[ is the number of features of each head), are the linear projection] matrices used by the _h_[th] head to generate queries, keys, and values respectively; _**r**_ **˜** _h_ is the result of relational cross-attention in the _h_[th] head; _**Wo**_ are the output weights through which the concatenated outputs of all _H_ heads are passed. Importantly, in RCA the queries and keys are produced based on glimpse contents, i.e. based on visual information, so that the resulting inner product in the softmax of Eq. 4 retrieves the visual relations disentangled from the specific visual features. At the same time, the values are generated from the glimpse locations that are stripped of any visual information. Hence, the overall processing in RCA is driven only by the relations between visual features (but not the visual features themselves) that modulate the processing of structural information (glimpse locations). Similar to (Mondal et al., 2024), in each layer, RCA was appended with feedforward networks and standard self-attention (Vaswani et al., 2017) as well as residual connections between each of those components (Figure A.1). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0014-05.png)


**----- Start of picture text -----**<br>
Relation<br>embeddings<br>Glimpse contents Abstractor Abstractor Abstractor<br>Glimpse locations Layer Layer . . . Layer<br>Abstractor Layer<br>Relational + Feedforward + Self-Attention + Feedforward<br>Cross-Attention<br>**----- End of picture text -----**<br>


Figure A.1: **Abstractor downstream architecture** : consists of several layers each consisting of the relational cross-attention (RCA), feedforward networks, the standard self-attention (SA), and residual connections between those. The most important component, the RCA, differs from the SA in that the queries and keys are generated based on glimpse contents (or their visual features) whereas the values are generated based on the glimpse locations. The figure is adapted from (Mondal et al., 2024). 

## B DATASET DESCRIPTIONS 

## B.1 SVRT 

The Synthetic Visual Reasoning Test (SVRT) dataset, introduced by (Fleuret et al., 2011), comprises 23 binary classification tasks. Each task involves a set of synthetic 2D shapes with an underlying relation between them. Following (Vaishnav & Serre, 2023), these tasks are divided into two families: those defined by same/different relations (SD) and those defined by spatial relations (SR). SD tasks: 1, 5, 6, 7, 13, 16, 17, 19, 20, 21, 22; SR tasks: 2, 3, 4, 8, 9, 10, 11, 12, 14, 15, 18, 23. The main challenge with these tasks lies in training with very few samples, thus testing the models’ inductive biases for learning the relational aspects of an image. The most commonly used dataset splits 

14 

consider 500 or 1000 examples per task. The validation and test sets contain 4k and 40k samples respectively. 

## B.2 SVRT #1-OOD 

This dataset is introduced by (Puebla & Bowers, 2022) and is built around the same-different task #1 from SVRT, in which the model has to make a binary decision whether two shapes in the image are the same up to a translation. While this dataset contains the same training images as in the original SVRT dataset, it introduces 13 test sets with novel OOD shapes that significantly differ from those in the train set (Figure B.2). The dataset contains 28k and 5.6k images for training and validation respectively. Each OOD test set contains 11.2k images. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0015-03.png)


Figure B.2: **SVRT #1-OOD** dataset. The goal is to determine whether 2 shapes are the same (up to a translation) or different. The train set consists of images from the original SVRT dataset. The test set includes 13 subsets each containing a different type of out-of-distribution shapes. Taken from (Puebla & Bowers, 2023). 

## B.3 ART 

The ART (Abstract Reasoning Tasks) dataset, proposed by (Webb et al., 2021) comprises four visual reasoning tasks each having a distinct underlying abstract rule to be detected: same/different (SD), relational-match-to-sample (RMTS), distribution-of-3 (Dist-3) and identity rules (ID). Figure B.3 illustrates each of the tasks. The dataset was constructed using 100 Unicode character objects, and it includes generalization regimes of varying difficulty based on the number of unique objects used during training. Following (Mondal et al., 2024), we focused on the most challenging generalization regime, where the train set involves problems created from 5 out of the 100 possible objects, while the test problems use the remaining 95 objects. This setup tests systematic generalization, requiring abstract rule learning from a small set of examples with minimal perceptual overlap between train 

15 

and test sets. SD, RMTS, Dist-3, and ID tasks contain 40, 480, 360, and 8640 training examples respectively while all of them contain 20K and 10K validation and test examples respectively. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0016-01.png)


Figure B.3: **Abstract Reasoning Tasks (ART) dataset** consists of 4 tasks. In the same/different task, the model has to determine whether an image contains 2 the same (up to a translation only) shapes. In the relational-match-to-sample task, the model has to select a pair of objects out of two pair candidates presented in the bottom row that contains objects in the same relation (either the ’same’ or ’different’) as the objects in the upper row. In the distribution-of-3 task, the model has to pick an object so that the lower row contains 3 objects that comprise the same set of objects as that of the upper row. In the identity rules task, the model has to select an object so that the resulting lower row contains objects that follow the same abstract pattern (ABA, ABB or AAA) as the objects in the upper row. Each instance of each task (except the same/different task) was presented with multiple images each containing a separate candidate object (or object pair) so that the model has to pick the one with the correct pair. Taken from (Webb et al., 2021). 

## B.4 CLEVR-ART 

The CLEVR-ART dataset Figure B.4, proposed by (Webb et al., 2024a), utilizes photorealistic synthetic 3D shapes from CLEVR (Johnson et al., 2017). It includes two visual reasoning tasks from ART: relational-match-to-sample and identity rules. The train set consists of images created using small and medium-sized rubber cubes in four colors (cyan, brown, green, and gray). In contrast, the test set features images generated from large-sized metal spheres and cylinders in four different colors (yellow, purple, blue, and red). Hence, the object features in the train and test sets are entirely distinct, challenging the systematic generalization of learned abstract rules to previously unseen object characteristics. 

## C EXPERIMENTAL DETAILS 

For all tasks, the images were resized to 128 _×_ 128 and the pixels were normalized to the range [0 _,_ 1]. For SVRT tasks, random horizontal and vertical flips were applied during the training following (Vaishnav & Serre, 2023). 

At each location of the saliency map, the corresponding error neuron compares a small (central) image patch of size 5 _×_ 5 that is in its receptive field with 8 surrounding patches of the same size. The surrounding patches are obtained by sliding a window from the neuron’s receptive field with the stride of 1 in 8 cardinal directions so that the surrounding patches have a high overlap with the central patch. The aggregation operator takes the minimal difference between the central patch and each of the surrounding patches. 

The number of iterations _T_ in the glimpsing process is set to 15 for SVRT (and SVRT #1-OOD), 20 for SD and RMTS tasks from ART, and 35 for all other tasks from ART and CLEVR-ART. For all datasets, the IoR mechanism used the hard mask (with radius of 5 pixels) except for CLEVR-ART where its soft version (with _ϵ_ = 450) was used. For all datasets, the multi-scale glimpse sensor provided glimpses at 3 scales each of which was of size 15 _×_ 15, where the first scale corresponded to the original resolution and the other two downsampled the regions of size 30 _×_ 30 and 45 _×_ 45. For the log-polar sensor, we use OpenCV (Bradski, 2000) formulation and implementation of the log-polar sampling setting the radius of the log-polar grid was set to 48. The glimpse size was set to 21 for SVRT (and SVRT #1-OOD) and ART tasks and to 15 for CLEVR-ART. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0017-00.png)


Figure B.4: **CLEVR-ART dataset** consists of two tasks from ART dataset. Relational match-tosample task: example problem involving the relation of difference. The correct answer choice (left image) contains pairs of different objects in the back and the front rows. The incorrect answer choice (right image) contains a pair of different objects in the front row but the same objects in the back row. Identity rules: example problem involving ABA rule. The correct answer choice (top left image) involves ABA rule in both back row and front row of objects. The object in the front row on the right in the other three images (incorrect choices) violates this rule. Taken from (Webb et al., 2024a). 

In our glimpse-based models, we use a small CNN to pre-process each glimpse content before feeding it to the downstream architecture. For glimpse contents produced by the multi-scale sensor, we use a 7-layer CNN where each layer has the kernel size of 3 _×_ 3, no padding and a ReLU activation function. For glimpse contents produced by the log-polar sensor, we use almost the same CNN but with 6 layers where the first 4 layers have the kernel size of 5 _×_ 5. The number of channels per layer is the same in both cases being 9 for SVRT (and SVRT #1-OOD) and 64 for the rest of the datasets. Each glimpse location was re-scaled to be in range [ _−_ 1 _,_ 1] and was pre-processed by an MLP with with two layers of size 32 and 64. As in (Mondal et al., 2024), we applied temporal context normalization (TCN) (Webb et al., 2020) on both sequences of pre-processed glimpse contents and locations. TCN normalizes a sequence along its temporal dimension. 

17 

For all datasets, the Abstractor module was instantiated with 8 attention 64-dimensional heads, 24 layers, and 2-layered MLPs with the hidden size of 256. The 50% dropout for the MLPs and attention is used only for SVRT #1-OOD. To generate the final output for the task, we took the mean of the sequence of final representations _**r** L_ and passed it to a single linear unit. 

For the training, we used ADAM optimizer (Kingma & Ba, 2015) with the learning rate of 1 _e −_ 4, batch size of 64. The whole model was trained end-to-end with the binary cross-entropy loss. 

## D ABLATIONS AND ADDITIONAL RESULTS 

Table D.1: Test accuracy (in %) for SVRT dataset using different glimpse sensors. The accuracy results are averaged over SD and SR tasks obtained from the best trained model. 

|Model|Glimpse sensor|SD<br>Dataset size|tasks<br>Dataset size|SR<br>Dataset size|tasks<br>Dataset size|
|---|---|---|---|---|---|
|||1000|500|1000|500|
|GAP-Transformer|multi-scale|80.8 ± 13.5|68.0 ± 14.3|97.9 ± 2.5|96.5 ± 3.1|
|GAP-Transformer|log-polar|83.3 ± 13.2|78.4 ± 13.5|98.2 ± 2.2|97.3 ± 2.7|
|GAP-Abstractor|multi-scale|93.1 ± 13.1|90.5 ± 14.5|98.5 ± 1.9|96.6 ± 2.1|
|GAP-Abstractor|log-polar|92.5 ± 12.3|90.6 ± 13.3|98.2 ± 2.6|97.0 ± 3.4|



Table D.2: OOD performance (in %) for test sets from SVRT #1-OOD using different glimpse sensors. The accuracy results are averaged over all OOD test sets for 10 trained models. 

|||Accuracy|
|---|---|---|
|Model|Glimpse sensor|(averaged over runs|
|||and OOD datasets)|
|GAP-Transformer|multi-scale|76.3 ± 18.4|
|GAP-Transformer|log-polar|74.7 ± 16.0|
|GAP-Abstractor|multi-scale|84.4 ± 15.3|
|GAP-Abstractor|log-polar|89.6 ± 8.7|



Table D.3: Accuracy (in %) for tasks from the ART dataset using different glimpse sensors. The accuracy results are averaged over 10 trained models. 

|Model|Glimpse sensor|SD|Task<br>RMTS<br>Dist3|Task<br>RMTS<br>Dist3|ID|
|---|---|---|---|---|---|
|GAP-Transformer|multi-scale|76.4 ± 13.7|62.2 ± 7.3|50.7 ± 6.1|55.5 ± 5.8|
|GAP-Transformer|log-polar|64.3 ± 9.1|52.6 ± 0.8|51.3 ± 13.7|57.0 ± 8.8|
|GAP-Abstractor|multi-scale|95.8 ± 0.8|95.3 ± 2.0|93.9 ± 1.3|91.1 ± 4.5|
|GAP-Abstractor|log-polar|97.7 ± 2.0|96.3 ± 1.0|98.4 ± 0.5|96.8 ± 1.8|



Table D.4: Accuracy (in %) for tasks from the CLEVR-ART dataset using different glimpse sensors. The accuracy results are averaged over 5 trained models. 

|Model|Glimpse sensor|Task<br>RMTS<br>ID|Task<br>RMTS<br>ID|
|---|---|---|---|
|GAP-Transformer|mult-scale|76.4 ± 4.3|64.7 ± 5.6|
|GAP-Transformer|log-polar|70.0 ± 3.5|63.0 ± 8.0|
|GAP-Abstractor|multi-scale|95.9 ± 1.4|93.4 ± 1.3|
|GAP-Abstractor|log-polar|94.6 ± 2.0|92.5 ± 1.5|



18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0019-00.png)


**----- Start of picture text -----**<br>
Dataset size 1000<br>100<br>90<br>80<br>70<br>60<br>50<br>40<br>Dataset size 500<br>100<br>90<br>80<br>70<br>60<br>50<br>40<br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23<br>Task multi-scale log-polar<br>Accuracy %<br>Accuracy %<br>**----- End of picture text -----**<br>


Figure D.5: Test accuracy for each of 23 SVRT tasks using different glimpse sensors for the GAPAbstractor model. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning_images/iclr_2025_Mind_the_GAP_Glimpse-based_Active_Perception_improves_generalization_and_sample_efficiency_of_visual_reasoning.pdf-0019-02.png)


**----- Start of picture text -----**<br>
SD tasks SR tasks<br>100<br>90<br>GAP-Abstractor<br>Slot-Abstractor<br>80 OCRA<br>GAP-Transformer<br>GAMR<br>70 Attn-ResNet<br>ResNet<br>Abstractor<br>60 Transformer<br>50<br>Dataset size Dataset size<br>100005000 1000 500 250 100 100005000 1000 500 250 100<br>Accuracy %<br>**----- End of picture text -----**<br>


Figure D.6: Sample efficiency on the SVRT dataset for 11 same-different (SD) tasks and 12 spatial relations (SR) tasks depending on the size of the training dataset. The results report the test accuracy [%] for the best trained models, the standard deviation is computed over SD and SR tasks. 

## D.1 EVALUATING IMPORTANCE OF GLIMPSE CONTENTS AND GLIMPSE LOCATIONS 

To measure the extent to which glimpse contents (”what”) and glimpse locations (”where”) are important, we trained our models passing only one of those two sequences to their downstream architectures. The training was done on SD tasks from SVRT dataset using 1000 training samples. For the Transformer-based downstream architecture, the input sequence consisted of either glimpse contents or glimpse locations, instead of the concatenation thereof. For Abstractor-based downstream architecture, we used a fixed set of trainable embeddings from which the RCA produced values while the keys and the queries were produced from either of the aforementioned input sequences. As it can be seen from Table D.6, the combination of glimpse contents and glimpse locations yields the highest performance compared to cases where only one of those sequences is used. 

19 

Table D.5: Sample efficiency on the SVRT dataset for 11 same-different (SD) tasks and 12 spatial relations (SR) tasks depending on the size of the training dataset. The results report the test accuracy [%] for the best trained models, the standard deviation is computed over SD and SR tasks. Only our models are listed. The prior art models are shown in Fig. D.6 

|Dataset size|GAP-Abstractor|SD tasks<br>GAP-Transformer|Abstractor|Transformer|
|---|---|---|---|---|
|10000|96.2 ± 6.7|95.3 ± 6.2|52.2 ± 4.8|57.4 ± 14.5|
|5000|95.4 ± 8.1|92.7 ± 9.8|51.7 ± 3.9|56.5 ± 14.2|
|1000|93.1 ± 13.1|82.5 ± 15.1|51.2 ± 2.1|51.9 ± 2.7|
|500|90.5 ± 14.5|76.0 ± 13.4|51.0 ± 1.9|51.6 ± 1.7|
|250|86.8 ± 16.0|69.4 ± 15.3|50.9 ± 1.5|51.3 ± 1.2|
|100|74.3 ± 17.2|58.7 ± 6.2|50.6 ± 0.9|51.0 ± 0.8|
||||||
|Dataset size|GAP-Abstractor|SD tasks<br>GAP-Transformer|Abstractor|Transformer|
|10000|99.7 ± 0.7|99.7 ± 0.5|93.5 ± 13.9|82.9 ± 19.2|
|5000|99.6 ± 0.5|99.6 ± 0.8|86.5 ± 17.3|75.7 ± 20.3|
|1000|98.5 ± 1.9|98.2 ± 2.2|60.9 ± 14.3|64.8 ± 13.7|
|500|96.6 ± 1.2|97.2 ± 2.5|57.0 ± 11.0|59.9 ± 11.0|
|250|93.4 ± 6.3|95.1 ± 3.5|52.4 ± 3.8|58.1 ± 8.8|
|100|86.2 ± 13.4|86.9 ± 11.2|52.2 ± 3.3|54.5 ± 5.1|



Table D.6: Test accuracy [%] averaged over SVRT SD tasks (11 in total) for models trained with 1000 samples depending on information passed to the downstream architecture. 

|Model|Inputs to the downstream architecture<br>glimpse contents<br>glimpse locations<br>glimpse contents and locations|Inputs to the downstream architecture<br>glimpse contents<br>glimpse locations<br>glimpse contents and locations|Inputs to the downstream architecture<br>glimpse contents<br>glimpse locations<br>glimpse contents and locations|
|---|---|---|---|
|GAP-Abstractor|81.0 ± 21.9|66.3 ± 14.2|93.1 ± 13.1|
|GAP-Transformer|77.4 ± 12.1|72.8 ± 17.5|83.3 ± 13.2|



## D.2 EVALUATING THE ROLE OF PROCESSING ONLY THE RELEVANT INFORMATION 

The superior performance of our GAP-based models compared to the ablated downstream architectures that process entire images indicates the importance of processing only the relevant information. The relevance of information in the context of GAP entails two features explained below. 

First, GAP discards all uninformative parts of the image providing thereby much less information to the downstream architecture than is available. This, in turn, improves the learning process preventing the models from being distracted by superfluous information. To elucidate the importance of this feature, we show the comparison of our GAP approach to the downstream architectures that process i) all image patches obtained by the sliding window approach with the stride of one (such patches almost entirely overlap and include glimpse contents obtainable by GAP); ii) all non-overlapping image patches obtained in ViT-like fashion based on the regular grid. The cases i) and ii) are referred to as _all patches_ and _ViT patches_ respectively. 

Second, each glimpse content is focused meaning that it is centered around a salient image region. This is, for example, in stark contrast to the patching process taking place in ViTs where the image is split into non-overlapping patches based on a regular grid. In that case, even if there were a method to determine only a subset of the most relevant patches, those patches would not necessarily contain the best portions of salient regions. To show the importance of focusing, we compare GAP to its modification where glimpse locations are tweaked to be in a set of “coarse” locations defined by the ViT’s regular grid. Importantly, this set of locations is only a subset of all possible image locations that the GAP can glimpse at. The resulting glimpse contents are obtained from the corresponding tweaked glimpse locations. We refer to this modification of GAP as _GAP-regular_ . 

All models described above were trained on SVRT SD tasks (11 tasks in total) using 1000 training samples. The size of the patches was the same as the size of the glimpse contents (15 _×_ 15) and the 

20 

Table D.7: Test accuracy [%] averaged over SVRT SD tasks (11 in total) for models trained with 1000 samples depending on information passed to the downstream architecture. See Section D.2 for more information. 

|Downstream|All patches|ViT patches|GAP-regular|GAP|
|---|---|---|---|---|
|architecture|(#patches=12996)|(#patches=64)|(#patches=15)|(#patches=15)|
|Abstractor|50.2 ± 1.3|51.3 ± 2.5|58.6 ± 14.8|93.1 ± 13.1|
|Transformer|50.9 ± 1.9|51.9 ± 2.6|53.8 ± 6.1|83.3 ± 13.2|



number of glimpses for GAP and GAP-regular was set to 15. The total number of patches obtained in the cases of all patches and ViT patches is 12996 and 64 respectively. 

Table D.7 shows the corresponding results. It can be seen that neither of the two aforementioned features alone is sufficient to solve the tasks. Specifically, providing all information available in face of either all patches or ViT-patches makes the downstream architecture not able to solve any task. The same can be seen in the case of providing less information but not well focused on salient regions (although the performance of GAP-regular is slightly above the 50% chance level). In other words, it is important to process only the relevant information by providing only a subset of available information and making this subset contain information focused on the most salient image parts. 

21 

