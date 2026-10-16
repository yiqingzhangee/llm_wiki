# **COCO-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs** 

**Tiep Le** _[∗]_ **Vasudev Lal** Intel Labs Intel Labs 

```
tiep.le@intel.com
```

```
vasudev.lal@intel.com
```

**Phillip Howard** _[∗]_ Intel Labs 

```
phillip.r.howard@intel.com
```

## **Abstract** 

Counterfactual examples have proven to be valuable in the field of natural language processing (NLP) for both evaluating and improving the robustness of language models to spurious correlations in datasets. Despite their demonstrated utility for NLP, multimodal counterfactual examples have been relatively unexplored due to the difficulty of creating paired image-text data with minimal counterfactual changes. To address this challenge, we introduce a scalable framework for automatic generation of counterfactual examples using text-to-image diffusion models. We use our framework to create COCO-Counterfactuals, a multimodal counterfactual dataset of paired image and text captions based on the MS-COCO dataset. We validate the quality of COCO-Counterfactuals through human evaluations and show that existing multimodal models are challenged by our counterfactual image-text pairs. Additionally, we demonstrate the usefulness of COCO-Counterfactuals for improving out-of-domain generalization of multimodal vision-language models via training data augmentation. We make our code[2] and the COCO-Counterfactuals dataset[3] publicly available. 

## **1 Introduction** 

While vision and language models have achieved remarkable performance improvements in recent years, out-of-domain (OOD) generalization remains a challenge for even the best models, which typically exhibit much lower performance in zero-shot evaluation settings than on withheld in-domain test sets. This has often been attributed to spurious correlations between non-causal features and labels in datasets which can be exploited during training as shortcuts to achieving artificially high in-domain performance (Geirhos et al., 2020). For example, image recognition models often learn to utilize spurious features in the backgrounds of images when trained for classification on datasets such as ImageNet (Singla and Feizi, 2021; Xiao et al., 2020). 

Augmenting training datasets with counterfactuals, which study the impact on a response variable following a change to a causal feature, has been previously proposed as a strategy for countering this effect in NLP models (Levesque et al., 2012; Kaushik et al., 2019). Motivated by concepts in causal learning (Feder et al., 2022), these methods typically form counterfactual examples by making minimal edits to an input text such that a corresponding label or attribute of the text (e.g., sentiment) 

> _∗_ Equal contribution 

> 2 `https://github.com/IntelLabs/multimodal_cognitive_ai/tree/main/COCO-Counterfactuals` 

> 3 `https://huggingface.co/datasets/Intel/COCO-Counterfactuals` 

37th Conference on Neural Information Processing Systems (NeurIPS 2023) Track on Datasets and Benchmarks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0002-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0002-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0002-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0002-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0002-04.png)


**----- Start of picture text -----**<br>
"Two men work in the kitchen of a "Two men work in the kitchen of a "A large black ball sitting next to "A large black ball sitting next to<br>restaurant." house." a glass of milk." a glass of water."<br>"A white dog sitting on a table "A white dog sitting on a table "A meal of sandwiches, potatoes, "A meal of sandwiches, pizza, and<br>next to a paper." next to a book." and Red Stripe beer." Red Stripe beer."<br>**----- End of picture text -----**<br>


Figure 1: Examples of COCO-Counterfactuals, our minimal-edit counterfactuals dataset for images with paired text captions. 

is changed. Training models with counterfactual examples therefore provides a strong inductive bias against learning spurious correlations in datasets, leading to greater robustness and improved generalization on OOD data (Eisenstein, 2022; Vig et al., 2020) as well as enabling better domain adaptation in low resource settings (Calderon et al., 2022). 

Despite its success in the realm of NLP, the application of counterfactual data augmentation to multimodal vision-language models has largely remained unexplored, mainly due to low-resource settings involving multimodal data and challenges associated with creating paired counterfactual examples spanning multiple modalities. For example, consider the task of creating counterfactual examples for a multimodal dataset containing images with associated text captions. Creating a counterfactual to a given image-text example requires not only minimally editing a casual feature in the text caption, but also making a corresponding minimal edit to the image which ideally modifies only the changed causal feature while preserving other spurious features from the original image. 

Collecting such counterfactual examples from existing image datasets is infeasible due to the massive variation in natural images that can accurately depict even identical text captions. While manual creation of counterfactual examples by humans is an option that has been employed previously for NLP (Kaushik et al., 2019; Gardner et al., 2020), this approach suffers from a lack of scalability due to the high cost of human labor, which would be compounded even further for multimodal counterfactuals due to the need for both text and image editing skills. Given these challenges, how can paired image-text counterfactual examples be created at the scale needed for effective model evaluation and data augmentation? 

We address this problem by introducing a novel data generation pipeline for automatically creating multimodal counterfactual examples using text-to-image diffusion models. Our approach minimally edits captions from an existing image-text dataset and then leverages Stable Diffusion (Rombach et al., 2021) with cross-attention control (Hertz et al., 2022) to generate pairs of images with minimal differences (i.e., isolated to the counterfactual change). We employ our data generation pipeline to create at scale **COCO-Counterfactuals** (Figure 1), a counterfactual variant of the MS-COCO dataset (Lin et al., 2014). 

We validate the quality of COCO-Counterfactuals using human evaluations and conduct zeroshot experiments showing that state-of-the-art multimodal models are challenged by our generated counterfactual examples. Our additional experiments show that training CLIP (Radford et al., 2021) 

2 

on COCO-Counterfactuals improves its performance on multiple OOD datasets, including zero-shot tasks not seen during training. We make COCO-Counterfactuals and the code for our counterfactual data generation pipeline publicly available under the CC BY 4.0 License. 

## **2 Related Work** 

## **2.1 Counterfactual Examples for NLP** 

Counterfactual data augmentation has been shown to improve the robustness of models across a wide range of problem domains in NLP. Kaushik et al. (2019) demonstrated that human-authored counterfactuals pose a significant challenge for existing models and that augmenting training datasets with counterfactual examples improves sentiment analysis and Natural Language Inference classifiers. Gardner et al. (2020) similarly used human experts to author minimally-edited contrast example sets for 10 NLP datasets and showed that model performance evaluated on them drops substantially. 

A number of approaches have been proposed to move beyond reliance on human authors towards automated methods for generating counterfactual examples. Wang and Culotta (2021) and Yang et al. (2021) automatically construct counterfactual examples by identifying and removing or replacing potentially causal words. Howard et al. (2022) introduce a framework for generating looser counterfactuals which allow larger edits of original examples, resulting in more natural and linguistic counterfactual examples. Other semi-automated methods have been proposed to generate counterfactual examples while still relying on human input or labeling (Wu et al., 2021). To the best of our knowledge, none of these existing approaches for automatic counterfactual generation have been extended to multimodal image-text datasets. 

## **2.2 Image Benchmarks for Measuring Spurious Correlations** 

Several image datasets have been proposed as benchmarks for measuring the degree to which models have learned to rely on spurious correlations during training. _CelebA Hair Color_ (Liu et al., 2015) is a binary image classification dataset that labels whether a person depicted has a blonde hair color, which is spuriously correlated with gender. Sagawa et al. (2019) constructed the _Waterbirds_ dataset by cropping images of landbirds or seabirds onto land and sea backgrounds, resulting in a binary classification task for bird type (i.e., landbird or seabird) in the presence of spurious correlations with the background. _Colored MNIST_ (Arjovsky et al., 2019) artificially imposes colors on the MNIST handwritten digits dataset, where the color is spuriously correlated with the class label. Lynch et al. (2023) uses text-to-image models to generate _Spawrious_ , an image classification dataset of four dog breeds spuriously correlated with six background locations. Unlike COCO-Counterfactuals, these datasets are limited only to image classification over a small number of labels and are primarily suited for evaluating model robustness as opposed to training data augmentation. 

Thrush et al. (2022) introduced _Winoground_ , an image-text dataset aimed at measuring visio-linguistic compositionality. Given two images and two captions which have the same words but in different order, the task is to correctly match each caption to its corresponding image. While their paired image-text examples can be viewed as counterfactuals, they focus only on edits to word order and rely on humans to create a dataset aimed specifically at evaluating compositionality. In contrast, our method automatically generates counterfactual examples with word content changes while also preserving non-causal spurious features across paired counterfactual images. 

_FOIL-COCO_ (Shekhar et al., 2017) contains ‘foil’ captions with a single change to the original MS-COCO caption to invalidate it for the accompanying image. They show that vision and language models struggle to correctly classify captions, detect the edited word, and correct the foiled caption. Our image-text counterfactuals similarly create ‘foil’ captions to MS-COCO captions, but goes further by also creating paired images which differ only according to how the caption was edited. 

## **2.3 Data Augmentation with Synthetic Images Generated from Text-to-image Models** 

Motivated by recent advances in text-to-image diffusion models (Nichol et al., 2021; Rombach et al., 2021; Saharia et al., 2022; Ramesh et al., 2022), data augmentation with synthetically-generated images has emerged as a growing topic of interest. He et al. (2022) showed that images generated by GLIDE (Nichol et al., 2021) for specific classes in image recognition datasets can be used for 

3 

training to improve performance on the corresponding image classification tasks. Trabucco et al. (2023) perform image-to-image transformations for data augmentation using text-to-image diffusion models, observing improvements in few-shot image classification performance. Vendrow et al. (2023) represent class labels from image recognition datasets as custom tokens in the vocabulary of a text-toimage diffusion model, enabling them to generate images of objects from the original dataset under different domain shifts. While our data generation pipeline also leverages text-to-image diffusion models, our approach differs from prior work in our focus on producing minimal changes to paired image-text data in both the vision and language modalities. 

## **3 COCO-Counterfactuals** 

We detail our data generation methodology for creating COCO-Counterfactuals (COCO-CFs), a synthetic multimodal counterfactual dataset of paired image and text captions based on the MS-COCO dataset (Lin et al., 2014). While we showcase our methodology by generating and releasing the COCO-Counterfactuals dataset, our approach can be applied to automatically construct multimodal counterfactuals for any dataset containing image captions.[4] 

## **3.1 Creating Counterfactual Captions** 

Given an original image caption _Co_ , our first task is to create a corresponding counterfactual caption _Cc_ which alters a subject of _Co_ while preserving most of its original details. The altered subject represents the changed causal feature in our counterfactual example while the remaining preserved details from the original caption can be viewed as potentially spurious correlated features. 

To alter a subject of _Co_ , we first identify all nouns using NLTK (Bird et al., 2009) as candidate words for substitution[5] For each of the _i ∈{_ 1 _, .., n}_ identified nouns, we create 10 candidate counterfactual captions by replacing only the _i_ -th noun in _Co_ with the [MASK] token and retrieving the top-10 most probable replacements via masked language modeling (MLM)[6] . This produces a total of _n ×_ 10 candidate counterfactual captions, which we then filter to retain only those in which the substituted word is also a noun. 

Our aim is to substitute nouns with alternative words that represent different subjects, and yet still maintain ontological similarity to the original noun. Hence, we use a pre-trained sentence similarity model[7] to measure the similarity between each candidate counterfactual caption and the original caption _Co_ , keeping only those candidates which have a sentence similarity within the range (0 _._ 8 _,_ 0 _._ 91). Finally, we use GPT-2 to score the perplexity of all candidates which remain after filtering and choose the candidate having the lowest perplexity as our counterfactual caption _Cc_ . 

## **3.2 Generating Counterfactual Images** 

After creating a counterfactual caption _Cc_ , our next task is to generate synthetic images _Io[s]_[and] _[ I] c[s]_ from the original caption _Co_ and counterfactual caption _Cc_ (respectively). Ideally we would like _Io[s]_ and _Ic[s]_[to differ only in terms of the noun which was replaced in] _[ C][o]_[to produce] _[ C][c]_[, thereby enabling] the changed causal feature to be learned in the presence of other potentially spurious correlated features (i.e., the unchanged details between _Co_ and _Cc_ ). However, this is a challenge for existing text-to-image generation models as minor changes to a text prompt can produce significantly different images. For instance, prompting Stable Diffusion with the captions "A small child lounges with a _remote_ in his hand" and "A small child lounges with a _toy_ in his hand" may produce images that differ not only in the object that the child is holding, but also in other details such as his facial features, the manner in which he is laying, the color of his clothes, and the image background. 

To address this issue, Hertz et al. (2022) proposed a methodology called Prompt-to-Prompt which injects cross-attention maps during the diffusion process to control the attention between certain pixels and tokens of the prompt during denoising steps. This enables separate generations from 

> 4 Appendix B.1 details hyper-parameters and pre-trained models used to generate COCO-Counterfactuals. 

> 5In this work, we focus on counterfactual captions that are derived from altering a noun from original captions. We leave the investigation of altering words of other types such as verbs and adjectives for future work. 6We use RoBERTa-base for MLM 

> 7 `https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2` 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0005-00.png)


**----- Start of picture text -----**<br>
With Prompt-to-Prompt Without Prompt-to-Prompt<br>"A small child lounges with a "A small child lounges with a toy "A small child lounges with a "A small child lounges with a toy<br>remote in his hand." in his hand." remote in his hand." in his hand."<br>**----- End of picture text -----**<br>


Figure 2: Examples of COCO-Counterfactuals generated with Prompt-to-Prompt (left) and without (right). Prompt-to-prompt enables us to extend the principle of minimal-edit text counterfactuals to the visual domain, isolating image differences to only the changed causal feature. 

text-to-image diffusion models to maintain many of the same image details while isolating their differences according to how the text prompts differ. An example of counterfactual image-text pairs ( _Co, Io[s]_[)][ and][ (] _[C][c][, I] c[s]_[)][ generated with and without prompt-to-prompt is shown in Figure 2, illustrating] how Prompt-to-Prompt enables the principle of minimal text edits for NLP counterfactuals to be extended to image generation. 

Brooks et al. (2023) noted that making different changes to images may require varying the parameter _p_ in Prompt-to-Prompt, which controls the number of denoising steps with shared attention weights. For example, changes that require more substantial structural modifications to the image may necessitate less overall similarity between the resulting images and thus fewer shared attention weights. We therefore adopt their proposed approach of over-generating 100 image pairs with Prompt-to-Prompt by randomly sampling values of the parameter _p ∼ U_ (0 _._ 1 _,_ 0 _._ 9)[8] . The resulting 100 image pairs are filtered using CLIP (Radford et al., 2021) to ensure a minimum cosine similarity of 0.2 between the encoding of each caption and its corresponding generated image, with the best image pair ( _Io[s][, I] c[s]_[)][ chosen from those which remain according to the directional similarity in CLIP] space (Gal et al., 2022): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0005-04.png)


where _ET_ and _EI_ are CLIP’s text and image encoders (respectively). The CLIP _dir_ metric measures the consistency in changes between the two images ( _Io[s][, I] c[s]_[)][ and their corresponding captions] ( _Co, Cc_ ). Thus, selecting images with a higher CLIP _dir_ improves the overall quality of our generated counterfactuals via greater consistency between the alterations made in both modalities. 

## **3.3 Generating COCO-Counterfactuals from MS-COCO** 

We apply our counterfactual caption and image generation pipeline described above to create the COCO-Counterfactuals dataset. Specifically, we generate candidate counterfactual captions for 25,014 original MS-COCO captions[910] , keeping only the best candidate counterfactual for each original caption that meets our filtering criteria. This produced a total of 24,508 original & counterfactual caption pairs ( _Co, Cc_ ) after filtering and selection. Our image over-generation pipeline produced 2.45 million candidate image pairs ( _Io[s][, I] c[s]_[)][ for these 24.5k caption pairs, of which 17,410 had at least one] generated image pair which met our filtering criteria. After selection according to the CLIP _dir_ metric, a total of 34,820 image-caption pairs remain, comprising our COCO-Counterfactuals dataset[11] . 

> 8We use the implementation from Instruct-Pix2Pix (Brooks et al., 2023). 

> 9 We use the 5K validation split of the 2017 dataset from `https://cocodataset.org/#download` . 

> 10While we use MS-COCO in this study as the source of our original captions, one advantage of our counterfactual generation approach is that the input dataset itself does not require paired image and text data. 

> 11While the MS-COCO 5K validation split has 25,014 captions, COCO-Counterfactuals includes only 17,410 of them due to our filtering criteria. Thus, for a fair comparison in our experiments, hereafter we refer to this subset of the 5K validation split including only those 17,410 captions and their paired original images as the MS-COCO dataset. 

5 

|Image set|Correct|Incorrect|Neither|Both|
|---|---|---|---|---|
|Generated from original caption|79.10%|8.16%|10.18%|2.56%|
|Generated from counterfactual caption|67.27%|18.43%|10.74%|3.56%|
|All images|73.18%|13.30%|10.46%|3.06%|



Table 1: Human evaluation results for COCO-Counterfactuals 

## **4 COCO-Counterfactuals Analysis** 

This section aims to show that, in addressing the challenges associated with low-resource settings involving multimodal data (see Section 1), our proposed novel data generation pipeline can serve as an efficient and scalable framework to automatically create high quality multimodal counterfactual examples in COCO-Counterfactuals (COCO-CFs). Toward this goal, we first employ human evaluation to analyze COCO-CFs. We then show that COCO-CFs can be used as a challenging dataset for model evaluation on zero-shot image-text retrieval and image-text matching tasks. 

## **4.1 Human Evaluation of COCO-Counterfactuals** 

We employ professional data annotators to conduct a human study on the quality of COCO-CFs. For each of the 34,820 images in COCO-CFs, we have at least one annotator choose whether the corresponding original or counterfactual caption best fits the image. Annotators can also choose “both” if both captions describe the image equally well, or “neither” if neither caption accurately describes the image. 10% of the images are labeled by 3 different individuals to estimate inter-annotator agreement, with the remaining images each labeled by a single annotator (see Appendix B.2 for additional details). 

Table 1 provides the percentage of images from COCO-CFs which were matched to their correct caption by the human annotators. We also report the percentage of incorrect matches (i.e., the wrong caption was picked as best describing the image) as well as the percentage of “both” and “neither” labels. Overall, 73% of images were correctly matched to their corresponding caption (see Appendix A.3 for an analysis of incorrect matches). Images generated from the counterfactual caption had a 10% greater incidence of incorrect caption selections than those generated from the original caption. This could be due to the constraints imposed on the counterfactual image by Prompt-toPrompt (i.e., shared attention weights with the original image), which increases the likelihood that the generated image lacks some of the details in its corresponding caption. 

The Fleiss’ kappa coefficient for the 10% of images labeled by three annotators was 0.74, indicating strong agreement among the annotators who participated in this study. Among those images which had label disagreement, 47.4% of the labels were correct, 27.3% were incorrect, and 18.6% selected “neither.” This suggests that many of the disagreements are associated with images for which the correct caption choice is more ambiguous. 

While we employed human annotators to validate the quality of COCO-Counterfactuals for this analysis, our automated counterfactual generation approach does not require the use of human annotators to produce a new dataset. Indeed, our experiments described subsequently in Section 5.1 show that COCO-Counterfactuals which were labeled as incorrect by humans have no negative impact on training data augmentation. 

## **4.2 COCO-Counterfactuals for Model Evaluation** 

Motivated by prior work which has proposed using counterfactuals as challenging test sets in NLP (Kaushik et al., 2019; Gardner et al., 2020), we further investigate whether our COCO-CFs can serve a similar purpose for state-of-the-art multimodal vision-language models such as CLIP, Flava (Singh et al., 2022), BridgeTower (Xu et al., 2022) and ViLT (Kim et al., 2021) for the zero-shot image-text retrieval and image-text matching tasks. We employed the HuggingFace implementations of these model in our experiments (see Appendix A.6 for more detail). 

6 

|**HuggingFace Pre-trained Models**<br>**Evaluated Dataset**|**Text Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**|**Image Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**|
|---|---|---|
|bridgetower-large-itm-mlm-itc<br>COCO-CFs<br>human-evaluatedCOCO-CFs|21.72 (**-51%**)<br>46.94 (-35%)<br>58.65 (-29%)<br>26.36 (**-41%**)<br>54.1 (-25%)<br>66.13 (-20%)|17.93 (-47%)<br>38.94 (-35%)<br>49.95 (-30%)<br>21.44 (-37%)<br>45 (-25%)<br>56.39 (-21%)|
|fava-full<br>COCO-CFs<br>human-evaluatedCOCO-CFs|21.28 (**-57%**)<br>46.64 (-41%)<br>58.87 (-34%)<br>26.1 (**-47%**)<br>54.23 (-31%)<br>66.83 (-25%)|37.76 (-16%)<br>66.15 (-12%)<br>75.83 (-10%)<br>43.4 (-3%)<br>72.35 (-3%)<br>81.44 (-4%)|



Table 2: Image-text retrieval performance on COCO-CFs and human-evaluated COCO-CFs for BridgeTower and Flava models. Largest drops of performance against the baseline are in boldface. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0007-02.png)


**----- Start of picture text -----**<br>
CLIP BridgeTower ViLT<br>8000<br>7000<br>6000<br>5000<br>4000<br>3000<br>2000<br>1000<br>0<br>5 0 5 10 15 20 25 30 5 0 5 10 15 5 0 5 10 15 20<br>ITM Score Difference<br>Random image retrieval (IRr) Counterfactual image retrieval (IRc)<br>Random text retrieval (TRr) Counterfactual text retrieval (TRc)<br>Count<br>**----- End of picture text -----**<br>


Figure 3: ITM score differences computed for existing multimodal models on COCOCounterfactuals dataset. 

## **4.2.1 Zero-shot Image-text Retrieval** 

We evaluate the zero-shot image-text retrieval (ITR) performance of pre-trained Flava and BridgeTower models on COCO-CFs as well as _human-evaluated COCO-CFs_ , which consists of only image-text pairs that were correctly matched in our human evaluation study (Section 4.1). Since pre-trained CLIP was employed in our counterfactual image generation process (see Section 3.2), it is not suitable for zero-shot ITR setting. Thus, we only report its evaluation in Appendix. A.6 for completeness. For baselines, we evaluate ITR performance of these models on the MS-COCO dataset. 

Table 2 reports ITR performance (i.e., Recall at 1, 5, and 10) on COCO-CFs and human-evaluatedCOCO-CFs for pre-trained BridgeTower and Flava models. The percentages enclosed within parentheses indicate the change in performance of a model on an evaluated dataset versus the performance of that model on MS-COCO (baseline). We observe that the performance of BridgeTower and Flava decreases significantly (up to 51% and 57%, respectively) compared to the baseline’s performance on both COCO-CFs and human-evaluated-COCO-CFs. These results demonstrate that COCO-Counterfactuals can serve as a challenging test set for SOTA multimodal vision-language models. 

## **4.2.2 Image-text Matching** 

Typically, during pre-training for image-text matching (ITM), multimodal models learn to differentiate actual image-text pairs from alternative images or captions which are randomly sampled from a dataset. By design, our COCO-CFs have the potential to make this task significantly more challenging by requiring models to also differentiate between minimally-edited image or text candidates. We measure the magnitude of this increased difficulty by comparing the difference in ITM scores between actual image-text pairs and their corresponding counterfactual or randomly sampled alternatives. 

Let ( _Co, Io_ ) denote an original image-text pair from MS-COCO, _Io[s]_[denote][our][synthetically-] generated image corresponding to _Co_ , and ( _Cc, Ic[s]_[)][ denote our corresponding synthetically-generated] counterfactual image-text pair in COCO-Counterfactuals. We further denote ( _Cr, Ir_ ) as a different original image-text pair randomly sampled from MS-COCO such that _Io_ = _Ir_ . For a given pre-trained 

7 

multimodal model, we compute the following metrics using its ITM scoring function _G_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0008-01.png)


IR _r_ and TR _r_ scores can be viewed as measuring the confidence of a model’s image or text retrieval (respectively) over two real image-text pairs from MS-COCO. Similarly, IR _c_ and TR _c_ scores measure image or text retrieval confidence, but using matched image-text pairs from COCO-Counterfactuals dataset. For all metrics, values greater than zero indicate that a model scores the correct image-text pair as more similar than its random or counterfactual alternative. Larger positive values can be viewed as indicating greater confidence in the model’s correct discernment between the alternatives. 

Figure 3 plots the distribution of these four metrics for three pre-trained multimodal models: CLIP, BridgeTower, and ViLT. All three models exhibit a significant negative distribution shift when presented with counterfactual alternatives rather than random alternatives, demonstrating the increased difficulty of COCO-Counterfactuals for existing models. A significant number of COCOCounterfactuals are also incorrectly scored (i.e., have ITM score difference less than zero) by all three models. Even in cases where the counterfactual alternatives can be correctly discerned, we posit that the much smaller values of IR _c_ and TR _c_ may improve the efficiency of training through the increased difficulty of the ITM task. 

## **4.2.3 Discussion** 

When used as a test set, COCO-Counterfactuals by design evaluate the robustness of models to minimal changes in paired image-text data. Table 2 and Figure 3 show that existing models perform significantly worse when evaluated on COCO-Counterfactuals. Additionally, we find that training these same models on COCO-Counterfactuals produces an average relative improvement of 24.3% in image-text retrieval performance on withheld counterfactual examples (see Table 5 of Appendix A.1). These results point to the usefulness our dataset for evaluating and improving the robustness of multimodal models to counterfactual changes. 

## **5 COCO-Counterfactuals for Training Data Augmentation** 

This section aims to evaluate whether COCO-Counterfactuals can serve as an alternative to real data for training data augmentation in low-resource scenarios. We train a fully unfrozen pre-trained CLIP model with its contrastive loss using various combinations of real data from MS-COCO and COCO-CFs datasets (see Appendix B.3 for additional training details). In order to investigate the robustness of models trained on COCO-CFs, we evaluate them on OOD datasets for image-text retrieval and image recognition. For baselines, we report the performance of pre-trained CLIP (i.e., without any additional training) as well as a CLIP model which has been additionally trained using only real data from MS-COCO. We repeat each of our training experiments with 25 different random seeds and report both the mean and standard deviation of performance measured across all random seeds. We also validate the statistical significance of performance improvements obtained by models trained on COCO-CFs using one-tailed t-tests. 

## **5.1 Image-text Retrieval** 

To evaluate OOD performance on the image-text retrieval task that CLIP was trained for, we use the 1K test set of Flickr30k dataset (Young et al., 2014). Table 3 reports the zero-shot performance of the baselines as well as CLIP trained with varying amounts of the original MS-COCO and COCOCFs datasets. We observe that all CLIP models trained with COCO-Counterfactuals outperform pre-trained CLIP by an average of 5 points, based on the mean performance across text and image retrieval settings. Additionally, our best model trained with 20,894 COCO-Counterfactuals provides statistically significant improvements relative to training only on the real MS-COCO dataset across all settings. We also found that COCO-Counterfactuals improve in-domain performance on the MS-COCO test set, which we detail in Appendix A.5. 

To investigate the potential impact of COCO-Counterfactuals which were labeled incorrectly by human annotators, we repeated our training data augmentation experiments using only image-text pairs which were correctly matched in our human evaluation study (Section 4.1). Overall, we found 

8 

|**Training dataset**<br>_|D_train_|_<br>% CFs|**Text Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**|**Image Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**<br>**Mean**|
|---|---|---|
|None (pre-trained CLIP)<br>0<br>0%|67.1<br>89<br>93.8|69.4<br>90.6<br>94.9<br>84.13|
|MS-COCO<br>13,928<br>0%<br>MS-COCO +COCO-CFs<br>13,928<br>50%|77_._900_._4<br>93_._790_._2<br>97_._110_._1<br>76_._660_._5<br>94_._53<br>~~0~~_._3<br>96_._840_._2|75_._140_._4<br>93_._720_._2<br>96_._690_._2<br>89_._060_._1<br>75_._75<br>~~0~~_._4<br>93_._600_._2<br>**96**_._**96**<br>0_._2<br>89_._050_._1|
|MS-COCO +COCO-CFs<br>34,820<br>60%<br>MS-COCO +COCO-CFs<br>41,784<br>67%|78_._28<br>~~0~~_._4<br>**94**_._**72**<br>~~0~~_._3<br>**97**_._**27**<br>0_._2<br>77_._750_._5<br>94_._51<br>~~0~~_._3<br>97_._030_._2|76_._13<br>~~0~~_._5<br>93_._85<br>~~0~~_._2<br>96_._91<br>~~0~~_._2<br>**89**_._**53**<br>0_._2<br>**76**_._**38**<br>~~0~~_._3<br>**94**_._**01**<br>~~0~~_._2<br>96_._79<br>~~0~~_._2<br>89_._41<br>~~0~~_._1|



Table 3: Image-text retrieval performance on the OOD Flickr30k 1K test set for pre-trained CLIP and CLIP models trained on varying amounts of data from MS-COCO and COCO-CFs datasets. _|D_ train _|_ indicates the total number of image-text pairs used for training, while % CFs indicates the percentage of those image-text pairs which were sampled from COCO-CFs. Results report mean over 25 different random seeds, with standard deviation as a subscript. Best results are in boldface. Results which use COCO-CFs are underlined when a one-tailed t-test indicates that their improvement over training only on MS-COCO is statistically significant ( _p ≤_ 0 _._ 05) 

|**Training dataset**|_|D_train_|_|% CFs|**CIFAR10**|**CIFAR100**|**Food101**|**Caltech101**|**Caltech256**|**ImageNet**|**Mean**|
|---|---|---|---|---|---|---|---|---|---|
|None (pre-trained CLIP)|0|0%|88.8|64.17|**84.17**|90.32|83.43|59.25|78.36|
|MS-COCO|13,928|0%|89_._210_._3|63_._890_._4|82_._670_._2|92_._770_._1|85_._050_._1|59_._550_._2|78_._850_._2|
|MS-COCO +COCO-CFs|13,928|50%|89_._45<br>0_._3|66_._67<br>~~0~~_._4|83_._13<br>~~0~~_._2|92_._630_._1|**85**_._**21**<br>0_._1|**59**_._**66**<br>0_._2|**79**_._**46**<br>~~0~~_._2|
|MS-COCO +COCO-CFs|34,820|60%|89_._160_._3|**66**_._**88**<br>~~0~~_._4|82_._120_._2|**92**_._**87**<br>0_._1|84_._950_._2|59_._220_._3|79_._20<br>~~0~~_._2|
|MS-COCO +COCO-CFs|41,784|67%|88_._510_._5|65_._97<br>~~0~~_._5|82_._060_._2|92_._770_._1|84_._590_._2|58_._880_._2|78_._800_._2|



Table 4: Zero-shot classification accuracy of pre-trained CLIP and CLIP models trained on varying amounts of data from MS-COCO and COCO-CFs datasets. All other settings are identical to Table 3. 

that excluding these incorrectly-labeled COCO-Counterfactuals from training data augmentation had a negligible impact on performance (see Appendix A.4). This suggests that training data augmentation is robust to noise introduced by synthetic data, and that the 26.82% of incorrectly-labeled COCOCounterfactuals do not pose an issue for data augmentation applications. While certain use cases which require a high degree of confidence in the accuracy of generated counterfactuals may benefit from the use of human validation, we believe that these results demonstrate how our approach can be used for fully automated training data augmentation without human annotation. 

## **5.2 Image Recognition** 

Despite being trained for image-text retrieval, CLIP has exhibited impressive performance at zero-shot image recognition. Using the same approach as Radford et al. (2021) for the image recognition task (i.e., form a sentence “A photo of a _{c}_ ” for each class label _c_ to obtain image-text matching scores), we evaluate whether CLIP models trained on COCO-Counterfactuals exhibit competitive OOD performance improvement to baselines’ performance in this zero-shot classification setting. 

Using the same CLIP models trained on varying amounts of MS-COCO and COCO-CFs, Table 4 reports their zero-shot classification accuracy on six image recognition datasets. We observe that training with an approximately 50-50 split of MS-COCO & COCO-CFs provides the best overall performance, offering improvements over pre-trained CLIP (without any additional training) on all datasets except Food101 and outperforming training with only MS-COCO on most datasets (see Appendix A.2 for additional analysis of performance differences). 

## **5.3 Discussion** 

Recent work investigating the suitability of synthetic training data for image recognition tasks has found that synthetic image data is much less efficient than real data, requiring 5x more synthetic training samples to achieve similar performance as models trained on real data (He et al., 2022). In contrast, our results show that training data augmentation with COCO-Counterfactuals is at least as efficient (Table 3) and sometimes more efficient (Table 4) than data augmentation with an identical amount of real data ( _|D_ train _|_ = 13 _,_ 928). This finding suggests that our approach could be particularly valuable in low-resource settings where paired image-text data is scarce. 

9 

Consistent with prior work on training data augmentation with NLP counterfactuals (Howard et al., 2022; Joshi and He, 2022), Tables 3 and 4 show that improvements in OOD performance with increasing amounts of counterfactual examples reaches a saturation point, beyond which additional data augmentation does not lead to further improvements. For image-text retrieval on Flickr30k (Table 3), this saturation point is reached with a 40 / 60% mixture of MS-COCO / COCO-Counterfactuals in the training dataset. In contrast, Table 4 shows that the saturation point for the OOD image recognition datasets is reached with a 50 / 50% split based on the mean of the six datasets. These results suggest that the optimal mixture of real examples and synthetically generated counterfactual examples may differ depending on the evaluation task and dataset. 

While training data augmentation with COCO-Counterfactuals produces statistically significant performance improvements relative to training with only real data, the overall magnitude of these improvements is limited and varies by evaluation setting. COCO-Counterfactuals produce the largest improvements on zero-shot image recognition tasks, where its overall mean improvement over pre-trained CLIP is twice as large as that achieved by training on an equivalent amount of real data from MS-COCO. However, OOD generalization performance varies by dataset, which further analysis suggests, is related to domain gaps between altered subjects in COCO-Counterfactuals and the domain of the evaluation dataset (see Appendix A.2 for details). 

## **6 Conclusion** 

We proposed an automated data generation methodology for creating counterfactual examples from image-text pairs to address the challenge of low-resource settings involving multimodal data. This approach was used to create COCO-Counterfactuals (COCO-CFs), a high-quality synthetic dataset of paired image-text counterfactuals derived from MS-COCO captions. COCO-CFs are challenging for existing pre-trained multimodal models and significantly increase the difficulty of the zero-shot image-text retrieval and image-text matching tasks. Our experiments demonstrate that augmenting training data with COCO-CFs improves OOD generalization on multiple downstream tasks. 

In this work, we focused on the creation of task-agnostic counterfactual examples. A promising direction for future research is the adaptation of our approach to produce task-specific counterfactuals. For example, in the case of image recognition, the counterfactual changes could be limited to a targeted label distribution to produce counterfactual examples more tailored to the end task. Alternatively, task-specific model failures or spurious correlations could be diagnosed and used as a basis for determining which counterfactual changes to consider when creating the counterfactual captions. We believe that such approaches have the potential to produce counterfactuals which are more targeted for improving specific model deficiencies. 

Another opportunity for future work is larger-scale automatic generation of counterfactual examples to enable full counterfactual pre-training of multimodal models. Additionally, we believe that extending our image-text counterfactuals to the video domain could be a promising path towards improving video transformers through counterfactual data augmentation. 

**Limitations & Ethical Concerns** Motivated by the desire to produce minimal-edit counterfactuals, we only considered changes to nouns. This is a common strategy for NLP counterfactuals (see Appendix B.1.1 for discussion), but alternative generation strategies such as controlled text decoding (Howard et al., 2022) could be used to enable a larger range of counterfactual changes, in addition to alterations of adjectives or verbs. We leave investigation of these directions to future studies. 

Due to a limited compute budget, we only explored generating COCO-Counterfactuals using Stable Diffusion. Additionally, our training data augmentation experiments were limited to a single model (CLIP). It is possible that other text-to-image generation models may exhibit better performance for generating counterfactual image-text data. Additionally, the benefits of counterfactual data augmentation may vary for different multimodal vision-language models. 

Despite the impressive recent improvements in text-to-image generation capabilities, models such as Stable Diffusion have well-known limitations that should be considered when utilizing datasets which are derived from them (see Appendix C.5 for a detailed discussion). We do not foresee significant risks of security threats or human rights violations in our work. However, the automated nature of our image generation process may introduce the possibility of our COCO-Counterfactuals dataset containing images that some individuals may consider inappropriate or offensive. 

10 

## **References** 

Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. 2019. Invariant risk minimization. _arXiv preprint arXiv:1907.02893_ . 

- Federico Bianchi, Pratyusha Kalluri, Esin Durmus, Faisal Ladhak, Myra Cheng, Debora Nozza, Tatsunori Hashimoto, Dan Jurafsky, James Zou, and Aylin Caliskan. 2023. Easily accessible text-to-image generation amplifies demographic stereotypes at large scale. In _Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency_ , pages 1493–1504. 

- Steven Bird, Ewan Klein, and Edward Loper. 2009. _Natural language processing with Python: analyzing text with the natural language toolkit_ . " O’Reilly Media, Inc.". 

- Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning to follow image editing instructions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18392–18402. 

- Nitay Calderon, Eyal Ben-David, Amir Feder, and Roi Reichart. 2022. Docogen: Domain counterfactual generation for low resource domain adaptation. _arXiv preprint arXiv:2202.12350_ . 

- Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. 2023. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. _ACM Transactions on Graphics (TOG)_ , 42(4):1–10. 

- Jaemin Cho, Abhay Zala, and Mohit Bansal. 2022. Dall-eval: Probing the reasoning skills and social biases of text-to-image generative models. _arXiv preprint arXiv:2202.04053_ . 

- Jacob Eisenstein. 2022. Uninformative input features and counterfactual invariance: Two perspectives on spurious correlations in natural language. _arXiv preprint arXiv:2204.04487_ . 

- Amir Feder, Katherine A Keith, Emaad Manzoor, Reid Pryzant, Dhanya Sridhar, Zach Wood-Doughty, Jacob Eisenstein, Justin Grimmer, Roi Reichart, Margaret E Roberts, et al. 2022. Causal inference in natural language processing: Estimation, prediction, interpretation and beyond. _Transactions of the Association for Computational Linguistics_ , 10:1138–1158. 

- Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. Stylegan-nada: Clip-guided domain adaptation of image generators. _ACM Transactions on Graphics (TOG)_ , 41(4):1–13. 

- Matt Gardner, Yoav Artzi, Victoria Basmova, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, et al. 2020. Evaluating models’ local decision boundaries via contrast sets. _arXiv preprint arXiv:2004.02709_ . 

- Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. 2020. Shortcut learning in deep neural networks. _Nature Machine Intelligence_ , 2(11):665–673. 

- Ruifei He, Shuyang Sun, Xin Yu, Chuhui Xue, Wenqing Zhang, Philip Torr, Song Bai, and Xiaojuan Qi. 2022. Is synthetic data from generative models ready for image recognition? _arXiv preprint arXiv:2210.07574_ . 

- Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2022. Prompt-to-prompt image editing with cross attention control. _arXiv preprint arXiv:2208.01626_ . 

- Phillip Howard, Gadi Singer, Vasudev Lal, Yejin Choi, and Swabha Swayamdipta. 2022. Neurocounterfactuals: Beyond minimal-edit counterfactuals for richer data augmentation. _arXiv preprint arXiv:2210.12365_ . 

- Nitish Joshi and He He. 2022. An investigation of the (in) effectiveness of counterfactually augmented data. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 3668–3681. 

- Divyansh Kaushik, Eduard Hovy, and Zachary C Lipton. 2019. Learning the difference that makes a difference with counterfactually-augmented data. _arXiv preprint arXiv:1909.12434_ . 

11 

- Wonjae Kim, Bokyung Son, and Ildoo Kim. 2021. Vilt: Vision-and-language transformer without convolution or region supervision. In _International Conference on Machine Learning_ , pages 5583–5594. PMLR. 

- Hector Levesque, Ernest Davis, and Leora Morgenstern. 2012. The winograd schema challenge. In _Thirteenth international conference on the principles of knowledge representation and reasoning_ . 

- Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_ , pages 740–755. Springer. 

- Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. 2022. Compositional visual generation with composable diffusion models. In _European Conference on Computer Vision_ , pages 423–439. Springer. 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized BERT pretraining approach. _CoRR_ , abs/1907.11692. 

- Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. 2015. Deep learning face attributes in the wild. In _Proceedings of the IEEE international conference on computer vision_ , pages 3730–3738. 

- Alexandra Sasha Luccioni, Christopher Akiki, Margaret Mitchell, and Yacine Jernite. 2023. Stable bias: Analyzing societal representations in diffusion models. _arXiv preprint arXiv:2303.11408_ . 

- Aengus Lynch, Gbètondji JS Dovonon, Jean Kaddour, and Ricardo Silva. 2023. Spawrious: A benchmark for fine control of spurious correlation biases. _arXiv preprint arXiv:2303.05470_ . 

- Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2021. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. _arXiv preprint arXiv:2112.10741_ . 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2018. Language models are unsupervised multitask learners. _Technical Report, OpenAI_ . 

- Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. _arXiv preprint arXiv:2204.06125_ . 

- Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bertnetworks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics. 

- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2021. High-resolution image synthesis with latent diffusion models. 

- Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. 2019. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. _arXiv preprint arXiv:1911.08731_ . 

- Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. _Advances in Neural Information Processing Systems_ , 35:36479–36494. 

- Dvir Samuel, Rami Ben-Ari, Simon Raviv, Nir Darshan, and Gal Chechik. 2023. It is all about where you start: Text-to-image generation with seed selection. _arXiv preprint arXiv:2304.14530_ . 

12 

- Patrick Schramowski, Manuel Brack, Björn Deiseroth, and Kristian Kersting. 2023. Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 22522–22531. 

- Ravi Shekhar, Sandro Pezzelle, Yauhen Klimovich, Aurélie Herbelot, Moin Nabi, Enver Sangineto, and Raffaella Bernardi. 2017. Foil it! find one mismatch between image and language caption. _arXiv preprint arXiv:1705.01359_ . 

- Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. 2022. FLAVA: A foundational language and vision alignment model. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022_ , pages 15617–15629. IEEE. 

- Sahil Singla and Soheil Feizi. 2021. Salient imagenet: How to discover spurious features in deep learning? _arXiv preprint arXiv:2110.04301_ . 

- Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. 2022. Winoground: Probing vision and language models for visio-linguistic compositionality. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5238–5248. 

- Brandon Trabucco, Kyle Doherty, Max Gurinas, and Ruslan Salakhutdinov. 2023. Effective data augmentation with diffusion models. _arXiv preprint arXiv:2302.07944_ . 

- Joshua Vendrow, Saachi Jain, Logan Engstrom, and Aleksander Madry. 2023. Dataset interfaces: Diagnosing model failures using controllable counterfactual generation. _arXiv preprint arXiv:2302.07865_ . 

- Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Simas Sakenis, Jason Huang, Yaron Singer, and Stuart Shieber. 2020. Causal mediation analysis for interpreting neural nlp: The case of gender bias. _arXiv preprint arXiv:2004.12265_ . 

- Zhao Wang and Aron Culotta. 2021. Robustness to spurious correlations in text classification via automatically generated counterfactuals. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 14024–14031. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. 2019. Huggingface’s transformers: State-of-the-art natural language processing. _CoRR_ , abs/1910.03771. 

- Tongshuang Wu, Marco Tulio Ribeiro, Jeffrey Heer, and Daniel S Weld. 2021. Polyjuice: Generating counterfactuals for explaining, evaluating, and improving models. _arXiv preprint arXiv:2101.00288_ . 

- Kai Xiao, Logan Engstrom, Andrew Ilyas, and Aleksander Madry. 2020. Noise or signal: The role of image backgrounds in object recognition. _arXiv preprint arXiv:2006.09994_ . 

- Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, and Nan Duan. 2022. Bridge-tower: Building bridges between encoders in vision-language representation learning. _arXiv preprint arXiv:2206.08657_ . 

- Linyi Yang, Jiazheng Li, Padraig Cunningham, Yue Zhang, Barry Smyth, and Ruihai Dong. 2021. Exploring the efficacy of automatically generated counterfactuals for sentiment analysis. _arXiv preprint arXiv:2106.15231_ . 

- Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. _Transactions of the Association for Computational Linguistics_ , 2:67–78. 

13 

## **Appendix** 

## **A Additional Experiments and Analysis** 

## **A.1 COCO-Counterfactuals Improve Model Robustness to Counterfactual Changes** 

By design, COCO-Counterfactuals may offer greater improvements to the robustness of models to minimal or counterfactual changes in images. Such examples are unlikely to be present in the datasets used previously to evaluate OOD generalization. Therefore, we also evaluate the performance of models on a withheld test set of COCO-Counterfactuals to determine their image-text retrieval capabilities on in-domain counterfactual examples. Specifically, we withhold 30% of the originalcounterfactual paired examples in COCO-Counterfactuals for testing and train the pre-trained CLIP, BridgeTower, and Flava models on the remainder, with 56% of the total dataset used for training and 14% used as a development set. 

Table 5 compares the performances of CLIP, BridgeTower, and Flava models trained on COCOCounterfactuals to those trained on an equivalent amount of real examples from MS-COCO and to their pre-trained versions[12] . We observe that training on COCO-Counterfactuals results in a mean improvement of 11.83, 21.55, and 11.47 relative to the pre-trained CLIP, BridgeTower, and Flava models, respectively. This represents an average relative improvement of 24.3% for each model over the performance of its pre-trained version. In addition, the CLIP, BridgeTower, and Flava models that were trained on COCO-Counterfactuals achieve a mean absolute improvement of 6.06, 10.08, and 5.28, respectively, relative to those that were trained on MS-COCO. The greater magnitude of these performance gains relative to our OOD image-text retrieval evaluations (Table 3) suggests that training on COCO-Counterfactuals improves model robustness to counterfactual changes, which are not present in our (non-counterfactual) OOD evaluation datasets. 

|**Pre-trained Models**|**Training dataset**|**Text Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**|**Image Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**<br>**Mean**|
|---|---|---|---|
|CLIP|None (pre-trained CLIP)|50.96<br>79.33<br>86.45|47.89<br>77.19<br>85.73<br>71.26|
||MS-COCO<br>COCO-CFs|57.17<br>84.23<br>90.66<br>65.03<br>90.26<br>94.99|55.45<br>84.00<br>90.65<br>77.03<br>64.09<br>89.52<br>94.62<br>83.09|
|BridgeTower|None (pre-trained BridgeTower)|35.26<br>65.31<br>76.73|28.77<br>56.63<br>68.46<br>55.19|
||MS-COCO<br>COCO-CFs|41.78<br>71.78<br>81.88<br>54.37<br>83.08<br>90.53|44.68<br>75.38<br>84.48<br>66.66<br>56.63<br>84.48<br>91.36<br>76.74|
|Flava|None (pre-trained Flava)|34.40<br>66.63<br>78.02|51.55<br>80.64<br>88.24<br>66.58|
||MS-COCO<br>COCO-CFs|46.70<br>76.36<br>85.68<br>54.39<br>83.35<br>90.27|52.55<br>81.08<br>88.43<br>71.80<br>57.97<br>85.11<br>91.38<br>77.08|



Table 5: Image-text retrieval performance on a withheld COCO-CFs test set. 

## **A.2 Analysis of Differences in OOD Generalization on Image Recognition Datasets** 

To better understand the differences in OOD generalization performance across datasets, we measured the frequency in which the altered subjects used to produce COCO-Counterfactuals overlapped with class labels. Specifically, we define the COCO-CFs Label Frequency for each image recognition dataset as the total number of COCO-Counterfactuals in which one or more of the dataset’s labels matched one of the two altered subjects used to produce the counterfactual pair. 

Table 6 provides the COCO-CFs Label Frequency for each image recognition dataset along with the change in OOD performance relative to pre-trained CLIP after training on various sizes of COCO-CFs (see Appendix B.3.1 for a definition of dataset sizes). We observe that datasets having a higher COCO-CFs Label Frequency generally achieve larger improvements in OOD generalization 

12Note that the image-text retrieval performance of the three pre-trained models (CLIP, BridgeTower, and Flava) on the in-domain COCO-Counterfactuals test set in Table 5 are higher than the respective values on the entire COCO-Counterfactuals dataset provided in Tables 2 and 13. This is expected because the retrieval space of the in-domain COCO-Counterfactuals test set is only 30% of the entire COCO-Counterfactuals dataset. 

14 

|**IR Dataset**|**COCO-CFs Label Frequency**|**COCO-CFsbase** ∆|**COCO-CFsmedium** ∆|**COCO-CFsall** ∆|
|---|---|---|---|---|
|CIFAR100|3446|2.50|2.63|1.80|
|Caltech101|354|2.31|2.55|2.45|
|Caltech256|744|1.78|1.52|1.16|
|CIFAR10|398|0.65|0.36|-0.29|
|ImageNet|887|0.41|-0.03|-0.37|
|Food101|28|-1.04|-2.05|-2.11|



Table 6: Frequency of class label occurrence in COCO-CFs and absolute change (∆) in performance relative to pre-trained CLIP after training on various sizes of COCO-CFs 

|**Error category**|**% present in sampled COCO-CFs**|
|---|---|
|Failure to generate subject/object|27%|
|Failure to generate fne-grained details|23%|
|Hyponymy relationship between altered subjects|15%|
|Human annotation error|15%|
|Failure to accurately depict spatial relationships|7%|
|Failure to generate correct number of objects|6%|
|Both altered subjects are present in the image|4%|
|Failure to bind attribute|3%|



Table 7: Image-text retrieval performance on the in-domain COCO-CFs test set. 

performance. The Pearson correlation coefficient between COCO-CFs Label Frequency and the 18 performance change measurements in Table 6 is 0.522 with a p-value of 0.026, indicating statistically significant positive correlation. 

These results suggest that a major contributor to the variation in OOD generalization performance across datasets is the overlap between the evaluation dataset domain and the set of subjects which are altered in COCO-Counterfactuals. Food101, the only dataset which saw no improvement in performance on our best-performing COCO-CFs training dataset, had only 28 cases of overlap between its label set and the subject alterations in COCO-CFs. In contrast, the greatest performance improvements were achieved on CIFAR100, for which 3446 COCO-CFs had subject alterations matching at least one label from the dataset. These findings point to the potential usefulness of targeting counterfactual changes for task-specific datasets. 

## **A.3 Analysis of Errors in COCO-Counterfactuals Identified by Human Annotators** 

In this section, we analyze errors in COCO-Counterfactuals using the labels assigned by human annotators (Section 4.1). Specifically, we consider an error to be any image-text pair from the COCO-Counterfactuals dataset for which the human annotator did not select the correct caption for the corresponding image. 

## **A.3.1 Manual Categorization of Errors** 

To investigate potential failure cases in our counterfactual generation approach, we randomly sampled and categorized 100 image-text pairs which were identified as errors by the human annotators. Table 7 provides the percentage of sampled COCO-Counterfactuals which were assigned to various error categories. Additionally, Tables 8 and 9 provide examples of counterfactual pairs which were assigned to the top-six most frequent error categories. 

We found that 66% of the sampled errors can be attributed to known limitations of existing text-toimage diffusion models (Chefer et al., 2023; Samuel et al., 2023; Cho et al., 2022), which include the categories for failure to generate a subject or object (e.g., Table 8, row 1), failure to generate fine-grained details (e.g., Table 8, row 2), failure to accurately depict spatial relationships (e.g., Table 9, row 2), failure to generate the correct number of objects described in the prompt (e.g., Table 9, row 3), and failure to bind attributes such as color. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0016-00.png)


**----- Start of picture text -----**<br>
Original Counterfactual<br>A cat walking through a  kitchen  by a eating tray A cat walking through a  field  by a eating tray.<br>A  man  playing Wii in a dirty room A  kid  playing Wii in a dirty room<br>Two  kids  in pink and purple jackets standing by a Two  girls  in pink and purple jackets standing by<br>fence a fence<br>Failure to generate subject/object<br>Failure to generate fine-grained details<br>Hyponymy relationship between altered subjects<br>**----- End of picture text -----**<br>


Table 8: Examples of failure cases identified by manual error analysis 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0017-00.png)


**----- Start of picture text -----**<br>
Original Counterfactual<br>Two people dressed in red  skiing  across a snowy Two people dressed in red  race  across a snowy<br>landscape landscape<br>A  woman  lies on the ground under a suitcase. A  man  lies on the ground under a suitcase.<br>A bathroom sink with two  toothbrush  holders on it A bathroom sink with two  cup  holders on it<br>Human annotation error<br>Failure to accurately depict spatial relationships<br>Failure to generate correct number of objects<br>**----- End of picture text -----**<br>


Table 9: Additional examples of failure cases identified by manual error analysis 

17 

|**Altered Subjects**<br>**Count**|**Altered Subjects**<br>**Count**|**Altered Subjects**<br>**Count**|
|---|---|---|
||||
|woman_→_girl<br>126<br>person_→_man<br>93<br>couple_→_group<br>36<br>person_→_girl<br>33<br>men_→_people<br>29<br>man_→_person<br>24<br>women_→_girl<br>21|man_→_boy<br>125<br>person_→_woman<br>42<br>people_→_guy<br>35<br>girl_→_woman<br>32<br>people_→_student<br>27<br>building_→_house<br>23<br>boy_→_man<br>21|people_→_men<br>116<br>person_→_boy<br>37<br>people_→_kid<br>33<br>man_→_woman<br>30<br>woman_→_man<br>24<br>men_→_boy<br>21|



Table 10: Frequency of altered subjects which appeared at least 20 times in errors identified by human annotators 

In many cases, these failures do not negatively impact the depiction of the counterfactual change in the two images because the inaccuracies pertain to details other than the altered subjects. For example, the first row of Table 8 shows the counterfactual pair associated with an image which was categorized as a failure to generate a subject/object; in this case, the altered subjects (kitchen _→_ field) are depicted correctly, but both images lack the _eating tray_ described in the prompt. Similarly, the counterfactual pair shown in the second row of Table 8 lacks fine-grained details in the prompt (e.g., _dirty_ room), but still depicts the altered subjects correctly (man _→_ kid). 

We found that 15% of the sampled errors could be attributed to a hyponymy relationship between the altered subjects which caused both captions to be equally valid for a given image. For example, the third row of Table 8 shows a counterfactual pair where the counterfactual image was incorrectly labeled by the human annotator because both captions were valid descriptions of the image (i.e., _girls_ can also be referred to as _kids_ ). Nevertheless, this example is still a valid counterfactual pair considering that the counterfactual caption does not accurately describe the original image and is more descriptive of the counterfactual image than the original caption. 

An additional 15% of the sampled errors appeared to be valid image-text pairs without any significant deficiencies. We therefore concluded that such cases were human annotation errors (see Table 9 row 1 for an example). Finally, 4% of the sampled images had equally valid caption choices because both of the altered subjects appeared in the image that was annotated. 

The results of this error analysis suggest that the quality of counterfactuals produced by our approach may improve as the capabilities of text-to-image diffusion models advance. New models which overcome known limitations of existing models could be used as a substitute for Stable Diffusion in our approach to produce higher-quality counterfactuals. Additionally, errors associated with hyponymy relationships could be addressed in future work through a refinement of our subject alteration process. For example, ontologies could be used to avoid noun substitutions where it can be determined that a hyponymy relationship exists between the noun candidates. Finally, additional constraints on the image generation process could be explored to prevent both altered subjects from appearing in the same image. 

## **A.3.2 Taxonomic Analysis of Errors** 

To better understand the relationship between the altered subjects in our counterfactuals and potential failure cases, we conducted a taxonomic analysis of the altered subjects which occurred most frequently among errors identified by human annotators. Table 10 provides the frequency of altered subject pairs which occurred at least 20 times in the error cases identified by human annotators. Interestingly, we observe that 19 of these 20 most frequent altered subject pairs belong to the _human_ taxonomy. 

We further analyzed this _human_ taxonomy in COCO-Counterfactuals by constructing a list of human-related words, which consists of ‘girl’, ‘boy’, ‘man’, ‘men’, ‘woman’, ‘guy’, ‘kid’, ‘person’, ‘people’, ‘child’, ‘children’, ‘couple’, ‘group’, and ‘lady’. An image-text pair is said to be related to this human taxonomy if the altered subject of its caption belong to this list. We find that there are 4117 image-text pairs in COCO-Counterfactuals that are related to the human taxonomy, among which 1864 were identified as errors by human annotators. The corresponding error rate for altered subjects related to the human taxonomy is 44 _._ 3%, which indicates that generating counterfactual pairs involving human altered subjects is more challenging for our approach. This suggests that a 

18 

|**Training dataset**<br>_|D_train_|_<br>_|D_CF<br>train_|_|**Text Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**|**Image Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**<br>**Mean**|
|---|---|---|
|MS-COCO +COCO-CFs<br>34,313<br>20,385|75.91<br>93.95<br>96.90|77.66<br>94.51<br>97.20<br>89.36|



Table 11: Mean image-text retrieval performance on the OOD Flickr30k test set using only COCOCounterfactuals which were correctly labeled by humans, measured across 25 different random seeds. 

|**Training dataset**<br>_|D_train_|_<br>_|D_CF<br>train_|_|**Text Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**|**Image Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**<br>**Mean**|
|---|---|---|
|None (pre-trained CLIP)<br>0<br>0|50.12<br>75.04<br>83.6|30.73<br>56.28<br>67.18<br>60.49|
|MS-COCO<br>13,928<br>0<br>MS-COCO +COCO-CFs<br>13,928<br>6,939|57_._330_._3<br>81_._280_._2<br>88_._710_._2<br>56_._910_._3<br>80_._700_._2<br>87_._820_._2|41_._130_._1<br>68_._460_._1<br>78_._450_._1<br>69_._230_._1<br>39_._920_._1<br>67_._010_._1<br>77_._150_._1<br>68_._250_._1|
|MS-COCO +COCO-CFs<br>34,820<br>20,894<br>MS-COCO +COCO-CFs<br>41,784<br>27,853|**58**_._**06**<br>0_._3<br>**81**_._**39**<br>0_._2<br>**88**_._**91**<br>~~0~~_._2<br>58_._02<br>~~0~~_._3<br>81_._39<br>~~0~~_._2<br>88_._780_._2|41_._63<br>~~0~~_._2<br>68_._64<br>~~0~~_._1<br>78_._85<br>~~0~~_._1<br>69_._58<br>0_._1<br>**41**_._**82**<br>~~0~~_._1<br>**68**_._**79**<br>0_._1<br>**78**_._**89**<br>0_._1<br>**69**_._**62**<br>~~0~~_._1|



Table 12: Image-text retrieval performance on the in-domain MS-COCO test set. All other settings are identical to Table 3. 

promising direction for future work is the exploration of improvements to the generation of images involving human subjects. 

## **A.4 Training Data Augmentation with Only Correctly-annotated COCO-Counterfactuals** 

We investigate the potential impact of COCO-Counterfactuals which were incorrectly labeled by humans on training data augmentation. Table 11 provides the OOD image-text retrieval performance in this setting, where COCO-Counterfactuals were filtered to only include those which were correctly labeled by the human annotators. Overall we find similar performance as our previous experiments using the full COCO-Counterfactuals dataset (Table 3), suggesting that filtering our synthetic data using human evaluations is not necessary for data augmentation applications. 

## **A.5 COCO-Counterfactuals Improve In-domain Performance** 

We evaluate the same models trained with counterfactual data augmentation described in Section 5 on the MS-COCO test set. The results of this in-domain evaluation are provided in Table 12. Similar to the OOD image-text retrieval setting, we find that data augmentation with 20 _,_ 892 COCOCounterfactuals provides statistically significant performance improvements relative to training without counterfactual data augmentations. Notably, previous work has observed that counterfactual data augmentation can degrade performance on withheld in-domain test sets (Wang and Culotta, 2021; Howard et al., 2022), whereas data augmentation with our COCO-Counterfactuals actually increases in-domain performance on MS-COCO. 

## **A.6 COCO-Counterfactuals for Model Evaluation Experiments** 

We further investigate whether our COCO-Counterfactuals (COCO-CFs) can serve as a challenging test set for state-of-the-art multimodal vision-language models such as CLIP, Flava (Singh et al., 2022), BridgeTower (Xu et al., 2022) and ViLT (Kim et al., 2021) for the zero-shot image-text retrieval and image-text matching tasks. We employed the following HuggingFace implementations of these models via the transformers library: 

- **CLIP** : We used the pre-trained model clip-vit-base-patch32 

- **Flava** : We used the pre-trained model flava-full 

- **BridgeTower** : We used the pre-trained model bridgetower-large-itm-mlm-itc 

- **ViLT** : We used the pre-trained model vilt-b32-finetuned-coco 

**Zero-shot Image-text Retrieval** . In Section 4, we evaluated the zero-shot image-text retrieval (ITR) performance of pre-trained Flava and BridgeTower models on COCO-CFs and _human-evaluated_ 

19 

|**HuggingFace Pre-trained Models**<br>**Evaluated Dataset**|**Text Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**|**Image Retrieval**<br>**R@1**<br>**R@5**<br>**R@10**|
|---|---|---|
|Clip<br>COCO-CFs<br>human-evaluated-COCO-CFs|37.65 (**-21%**)<br>64.89 (-9%)<br>74.57 (-7%)<br>43.25 (**-9%**)<br>70.4 (-2%)<br>79.37 (-1%)|34.98 (+5%)<br>62.29 (+7%)<br>72.43 (+4%)<br>40.14 (+21%)<br>67.86 (+16%)<br>77.66 (+11%)|



Table 13: Image-text retrieval performance on COCO-CFs and human-evaluated COCO-CFs for CLIP model. Largest drops of performance against the baseline are in boldface. 

_COCO-CFs_ that consists of only image-text pairs that were correctly matched in human evaluation in Section 4.1. Since a pre-trained CLIP model was employed in our counterfactual image generation process (see Section 3.2), CLIP models are not suitable for the zero-shot ITR evaluation. Hence, we only report evaluation of pre-trained CLIP model for ITR task here for completeness. 

Table 13 reports ITR performance (i.e., Recall at 1, 5, and 10) on COCO-CFs and human-evaluatedCOCO-CFs for the pre-trained CLIP model. Similar to Table 2, the percentages enclosed within parentheses indicate the change in performance of the CLIP model on an evaluated dataset versus the performance of that model on MS-COCO (baseline). 

We observe that on both COCO-CFs and human-evaluated-COCO-CFs datasets, while the performance of the pre-trained CLIP model degrades marginally on Text Retrieval task, its performance increases for Image Retrieval task. We attribute this to potential data contamination due to how we employed a pre-trained CLIP model in our counterfactual image generation process (see Section 3.2). As a result, COCO-Counterfactuals includes image-text pairs for which CLIP achieves high image-text retrieval performance. 

## **B Dataset and Experiment Details** 

## **B.1 Hyper-parameter Selection and Models Used to Generate COCO-Counterfactuals** 

In this section, we will detail hyper-parameters and pre-trained models used to our generate COCOCounterfactuals dataset. 

## **B.1.1 Creating Counterfactual Captions** 

Given an original caption from the MS-COCO dataset, we use Natural Language Toolkit ( **NLTK** ) (Bird et al., 2009) modules: 

- _punkt_ for sentence tokenizer, and 

- _averaged_perceptron_tagger_ for part-of-speech (POS) tagger 

to identify all nouns as candidate words for substitution. 

For each of the identified nouns, we create 10 candidate counterfactual captions by replacing only one noun with the [MASK] token and retrieving the top-10 most probable replacements via masked language modeling (MLM). For MLM, we used the pre-trained model _roberta-base_ (Liu et al., 2019) implemented in the library _transformers_ (Wolf et al., 2019) 

A motivation for our use of noun substitutions is the desire to produce minimal-edit counterfactuals. This is a common strategy for NLP counterfactuals (Kaushik et al., 2019; Wang and Culotta, 2021; Yang et al., 2021) because the high degree of similarity between the original and counterfactual text preserves spurious correlations that models might rely on for discernment. Furthermore, our use of perplexity filtering mitigates the potential for such word substitutions to produce unrealistic counterfactual captions. 

In order to measure similarity between each candidate counterfactual caption and an original caption, we used the pre-trained model all-MiniLM-L6-v2, which is implemented within the library _sentencetransformers_ (Reimers and Gurevych, 2019). 

Among generated candidate counterfactual captions, we kept only those candidates which have a sentence similarity within the range (0 _._ 8 _,_ 0 _._ 91). We selected this similarity range heuristically, observing that it produced best results after extensive experimentation. 

20 

Finally, we employed the pre-trained model gpt2-large, a _GPT-2_ (Radford et al., 2018) model implemented in the transformers library, to score the perplexity and choose the candidate having the lowest perplexity as our counterfactual caption. 

## **B.1.2 Generating Counterfactual Images** 

After creating a counterfactual caption, our next task is to generate synthetic images from the corresponding original caption and counterfactual caption, respectively. In order to do so, we have adopted an implementation from Instruct-Pix2Pix (Brooks et al., 2023) in which all hyper-parameters are set to their default values. 

Specifically, we over-generate 100 image pairs with Prompt-to-Prompt by randomly sampling values of the parameter _p ∼ U_ (0 _._ 1 _,_ 0 _._ 9) (i.e., parameter _p_ indicates the portion of denoising for which to fix self attention maps). The resulting 100 image pairs are filtered using CLIP (Radford et al., 2021) to ensure: 

- _i._ a minimum cosine similarity of 0.2 between the encoding of each caption and its corresponding generated image, and 

- _ii._ a minimum cosine similarity of 0.7 between the encoding of the two respective images in each generated image pair. 

From remaining image pairs, the best image pair is chosen such that it has the highest directional similarity _CLIPdir_ score. Selecting images with the highest _CLIPdir_ improves the overall quality of our generated counterfactuals via greater consistency between the alterations made in both modalities. 

## **B.2 Human Annotation Study** 

Professional annotation services for our human study were provided by Mindy Support. The total cost of this study was $1068.59 for 218 annotation hours. The instructions provided to annotators are depicted in Figure 4. We are unable to provide the hourly wages paid to workers as this is considered proprietary information by Mindy Support. However, the following statement was provided by the vendor regarding compensation: 

“We prioritize compliance with all standards of local and international legislation, ensuring fair treatment and equal opportunities for individuals of various backgrounds, ages, and other characteristics. We are committed to upholding the principles of fair wages, non-discrimination, and labor standards, including the prohibition of child labor. As an organization, we strictly adhere to legal requirements and strive to create an inclusive and ethical working environment for all. Rest assured that our compensation rates reflect market demands and provide fair remuneration for the work performed by our participants. We remain dedicated to abiding by all labor regulations and social and economic standards.” 

## **B.3 Training Data Augmentation Experiments** 

In this section, we detail how we constructed our training datasets and how we finetuned the pretrained CLIP model for experiments described in Section 5. 

## **B.3.1 Training Dataset Preparation** 

Our training data augmentation experiments utilize various combinations of the MS-COCO validation set and our COCO-Counterfactuals dataset. For simplicity, a caption-image pair is referred to as a _sample_ . We define a _counterfactual sample_ as following. Given a sample ( _C, I_ ) (i.e., caption _C_ and image _I_ ) from our COCO-Counterfactuals dataset, a sample ( _C[′] , I[′]_ ) from COCO-Counterfactuals dataset is called a counterfactual sample of ( _C, I_ ) iff _C[′]_ and _C_ are counterfactual captions of each other. By this definition, COCO-Counterfactuals dataset includes 34,820 samples that correspond to 17,410 paired counterfactual samples. 

For experiments in Section 5, we have prepared the following 4 datasets: 

21 

Instructions: 

Select the caption which best describes the image. In cases where both captions are valid for the image, please try to pick the one which is more descriptive or detailed. If both captions are valid and describe the image equally well, select “Both”. If neither of the captions accurately describe the image, select “Neither”. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs_images/nips_2023_COCO-Counterfactuals_Automatically_Constructed_Counterfactual_Examples_for_Image-Text_Pairs.pdf-0022-02.png)


- A woman standing in a kitchen by a window 

- A man standing in a kitchen by a window 

- Both 

- Neither 

Figure 4: Instructions provided to data annotators 

- _(a.)_ **MS-COCO** dataset. This is a subset of the 5K validation split of the 2017 MS-COCO dataset[13] , achieved by filtering out all samples with captions which are not included in our COCO-Counterfactuals. This results in a dataset (referred to as the MS-COCO dataset used in experiments in Section 5) of 17,410 captions and their paired original images. 

- _(b.)_ **[MS-COCO + COCO-CFs ]base** dataset. This dataset is a combination of: 

   - 50% random sampling (i.e., 8,705 caption-image pairs) of the MS-COCO dataset constructed in _(a.)_ . 

   - 25% random sampling of paired counterfactual samples from our COCOCounterfactuals dataset. This results in a total of 4,353 pairs of samples with their corresponding counterfactuals, for a total of 8,706 caption-image samples from our COCO-Counterfactuals dataset. 

> 13 `https://cocodataset.org/#download` 

22 

Overall, the [MS-COCO + COCO-CFs ] _base_ dataset consists of 17,411 captions and their paired original images, which is approximately equal in size to the MS-COCO dataset constructed in _(a.)_ 

- _(c.)_ **[MS-COCO + COCO-CFs ]medium** dataset. This dataset is a combination of: 

   - all samples (i.e., 17,410 caption-image pairs) from the MS-COCO dataset constructed in _(a.)_ . 

   - 75% random sampling (i.e., 26,115 caption-image pairs) from our COCOCounterfactuals dataset. 

Overall, dataset [MS-COCO + COCO-CFs ] _medium_ consists of 43,525 captions and their paired original images. 

- _(d.)_ **[MS-COCO + COCO-CFs ]all** dataset. This dataset is a combination of: 

   - all samples (i.e., 17,410 caption-image pairs) from the MS-COCO dataset constructed in _(a.)_ . 

   - all samples (i.e., 34,820 caption-image pairs) from our COCO-Counterfactuals dataset. 

   - Overall, dataset [MS-COCO + COCO-CFs ] _all_ consists of 52,230 captions and their paired original images. 

Each of the datasets described above is split into a training set (80%) and a validation set (20%). In each experiment, the validation set is used to pick the best model checkpoint at the conclusion of training. Tables 3, 4, and 12 report experimental results for models trained using the train split of these four datasets. _|D_ train _|_ indicates the total number of samples (i.e., image-text pairs) included in the respective training set, while _|D_ train[CF] _[|]_[ indicates how many of those image-text pairs were sampled] from the COCO-Counterfactuals dataset. 

## **B.3.2 Finetuning CLIP with Data Augmentation** 

We use each of the four training sets constructed in Section B.3.1 to finetune the CLIP model _clip-vit-base-patch32_ . We adopted a publicly-available finetuning script provided by HuggingFace[14] . 

We repeat each of our training experiments with 25 different _seeds_ and _data_seed_ from the ranges [107 _,_ 131] and [108 _,_ 132], respectively. In each experiment, we use a learning rate to 5e-7, weight decay of 0 _._ 001, training batch size of 128, and evaluation batch size of 128. 

## **B.4 Compute Infrastructure Used In this Study** 

Our experiments were conducted using an Intel AI supercomputing cluster comprised of Intel Xeon processors and 512 Intel Gaudi[®] AI accelerators, as well as an internal Slurm linux cluster with Nvidia RTX 3090 GPUs. Our dataset generation pipeline was parallelized across this compute infrastructure and took approximately 3 days to complete. Our training data augmentation experiments varied in running time depending upon the size of the dataset, ranging between 2 to 10 hours. 

## **B.5 License Information of Assets Employed in This Study** 

- **NLTK** is open source software distributed under the terms of the Apache License Version 2.0. 

- _Transformers_ is released under the Apache License Version 2.0 and is available on GitHub at `https://github.com/huggingface/transformers` . 

- Pre-trained model Roberta-base is released under the MIT License. 

- Library _sentence-transformers_ is licensed under the Apache License Version 2.0 and is available on GitHub at https://github.com/UKPLab/sentence-transformers. 

- Pre-trained model _all-MiniLM-L6-v2_ is licensed under the Apache License Version 2.0. 

- Pre-trained _gpt2-large_ model is license under the MIT License. 

> 14The finetuning script can be accessed at `https://github.com/huggingface/transformers/blob/ main/examples/pytorch/contrastive-image-text/run_clip.py` . 

23 

- _Instruct-Pix2Pix_ is licensed under the MIT License and is available on GitHub at `https://github.com/timothybrooks/instruct-pix2pix` . 

- Instruct-Pix2Pix further employs stable-diffusion-v1-5 that is released under CreativeMLOpen-RAIL-M License. 

- For the _MS-COCO_ dataset: 

   - The annotations in the dataset are released under the Creative Commons Attribution 4.0 License. 

   - The use of the images in the dataset must abide by the Flickr Terms of Use. 

- Pre-trained model _clip-vit-base-patch32_ is licensed under the MIT License. 

- Pre-trained model _flava-full_ is licensed under the 3-Clause BSD License. 

- Pre-trained model _BridgeTower large-itm-mlm-itc_ is released under the MIT License. 

- Pre-trained _vilt-b32-finetuned-coco_ model is license under the Apache License Version 2.0. 

## **C Datasheet for Dataset** 

## **C.1 Motivation** 

**For what purpose was this dataset created?** This dataset was created for the purpose of exploring the relevancy of counterfactual examples for multimodal vision-language models. Specifically, our aim was to create a dataset which can serve both as a challenging evaluation dataset for existing models and as a resource for training data augmentation to improve multimodal models on downstream tasks. For additional discussion of our motivation and the intuition behind counterfactual examples, see Section 1. 

**Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?** The dataset was created by the authors of this paper who are affiliated with Intel Labs, a research and development organization within Intel Corporation. 

**Who funded the creation of the dataset?** The creation of this dataset was funded by Intel Corporation. 

## **C.2 Composition** 

**What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?** The instances represent synthetically-generated images and accompanying text captions. The images depict a variety of different everyday scenarios. 

**How many instances are there in total (of each type, if appropriate)?** COCO-Counterfactuals contains a total of 34,820 image-caption pairs. 

**Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** Yes, it contains all possible instances per our filtering criteria. 

**What data does each instance consist of?** Each instance consists of a synthetically-generated image and an accompanying text caption. 

**Is there a label or target associated with each instance?** No 

**Is any information missing from individual instances?** No 

**Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?** Yes, instances which correspond to a single counterfactual pair are annotated as such in our dataset. Otherwise, there are no other relationships between individual instances. 

**Are there recommended data splits (e.g., training, development/validation, testing)?** No 

**Are there any errors, sources of noise, or redundancies in the dataset?** The automated methodology used to generate COCO-Counterfactuals introduces the possibility of noise and errors in the dataset. See Section **??** for additional discussion. 

**Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** Yes 

24 

**Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)?** No 

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** Yes, the dataset may contain offensive material due to the manner in which it was automatically constructed. See Section **??** for additional discussion. 

**Does the dataset identify any subpopulations (e.g., by age, gender)?** No 

**Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?** No 

**Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals race or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** No 

## **C.3 Collection Process** 

**How was the data associated with each instance acquired?** The data associated with each instance was acquired via our data generation methodology (see Section 3 for a detailed description). 

**What mechanisms or procedures were used to collect the data (e.g., hardware apparatuses or sensors, manual human curation, software programs, software APIs)?** Please see Section 3 for a complete description of our data generation methodology. 

**If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?** Not applicable 

**Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?** The COCOCounterfactuals dataset was collected automatically, as detailed in Section 3. Human evaluation of COCO-Counterfactuals involved paid professional annotators employed by Mindy Support (see Appendix B.2 for details). 

**Over what timeframe was the data collected?** The data was generated and evaluated over the course of approximately three months. 

**Were any ethical review processes conducted (e.g., by an institutional review board)?** No, institutional review was not required. 

**Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?** No, the dataset was generated automatically and was not collected directly from individuals. 

**Were the individuals in question notified about the data collection?** Not applicable 

**Did the individuals in question consent to the collection and use of their data?** Not applicable 

**If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?** Not applicable 

**Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?** No, not applicable 

## **C.4 Preprocessing/cleaning/labeling** 

**Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** Yes, we apply extensive filtering to various stages of our data generation pipeline in order to improve the quality of the dataset. See Section 3 for a complete description of these methods. 

25 

**Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?** No. However, due to how our dataset is automatically constructed, raw data can be reproduced by running our code. 

**Is the software that was used to preprocess/clean/label the data available?** Yes, we will make our code publicly available upon publication. 

## **C.5 Uses** 

**Has the dataset been used for any tasks already?** Yes, we applied COCO-Counterfactuals to the task of model evaluation in Section 4 and to the task of training data augmentation in Section 5. 

**Is there a repository that links to any or all papers or systems that use the dataset?** Our GitHub repository will contain links to papers and systems used by our data generation methodology. Additionally, this paper contains references to all such papers and systems that we utilized. 

**What (other) tasks could the dataset be used for?** COCO-Counterfactuals is broadly applicable to tasks which require multimodal inputs consisting of images with paired text. One potential use case not explored during this study is large-scale pre-trianing of multimodal models, which could be improved through counterfactual data augmentation. 

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** Due to the way in which COCOCounterfactuals was generated automatically, it may contain errors, offensive material, or biases which are present in the models employed by our pipeline. 

We used Stable Diffusion to collect image data, which has well-known limitations that should be considered when utilizing datasets which are derived from them. These limitations include unrealistic depictions of hands, palms, and other fine-grained objects (Samuel et al., 2023); failures to generate one or more of the subjects in a prompt and correctly bind attributes such as color (Chefer et al., 2023); difficulties with object counting and spatial relationship understanding (Cho et al., 2022); and challenges associated with the composition of concepts (Liu et al., 2022). While our experiments suggest that COCO-Counterfactuals is relatively robust to generation failures when used for training data augmentation, future applications of our methodology should consider the risks associated with these limitations relative to the intended use of the generated dataset. 

Stable diffusion and other text-to-image diffusion models have been shown to exhibit biases associated with race and gender, including over-representation of masculinity and whiteness (Luccioni et al., 2023); racial and gender disparities in depictions of certain occupations (Bianchi et al., 2023); and preferences for certain genders or skin tones (Cho et al., 2022). Consequently, models trained on COCO-Counterfactuals may learn similar social biases as those expressed in synthetic images generated by Stable Diffusion. While some recent work has investigated approaches for mitigating biases in diffusion models, further investigation is needed into the de-biasing of datasets on which these models are trained in order to fully eliminate them (Schramowski et al., 2023). 

Users of the dataset should carefully consider how these limitations may impact their potential use case. 

**Are there tasks for which the dataset should not be used?** The dataset should not be used for a task if the limitations discussed above are unacceptable or potentially problematic for the inteded use case. 

## **C.6 Distribution** 

**Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** Yes, the dataset will be made open source and publicly available. 

**How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?** The dataset will be distributed via the Hugging Face Hub. 

**When will the dataset be distributed?** The dataset will be made available publicly upon publication of this paper. 

26 

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** The dataset will be distributed under the CC BY 4.0 license. 

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** No 

**Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** No 

## **C.7 Maintenance** 

**Who will be supporting/hosting/maintaining the dataset?** The datasset will be hosted on the Hugging Face Hub. The authors of this paper will support and maintain the dataset via our public GitHub repository. 

**How can the owner/curator/manager of the dataset be contacted (e.g., email address)?** The corresponding author can be contacted via the e-mail address listed on the first page of this paper. Alternatively, an issue can be raised on our GitHub repository. 

## **Is there an erratum?** No 

**Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?** Although we do not anticipate the need to update this dataset in the future, we will respond to issues which are raised on our public GitHub repository for this project. 

**If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)?** Not applicable 

**Will older versions of the dataset continue to be supported/hosted/maintained?** Yes. If the dataset is updated in the future, older versions will remain available. 

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** Yes, we make our dataset open source and welcome others to build on it. This can be done by making contributions to our GitHub repository and/or citing our dataset as appropriate when used in future work. 

27 

