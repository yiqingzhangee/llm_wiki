# **ARAIDA: Analogical Reasoning-Augmented Interactive Data Annotation** 

**Chen Huang** _[♠♣]_ **, Yiping Jin** _[♡]_ **, Ilija Ilievski** _[♢]_ **, Wenqiang Lei** _[♠♣]_[*] **, Jiancheng Lv** _[♠♣]_ 

> _♠_ College of Computer Science, Sichuan University, China 

> _♣_ Engineering Research Center of Machine Learning and Industry Intelligence, Ministry of Education, China 

- _♡_ NLP Group, Pompeu Fabra University, Spain 

> _♢_ ISEM, National University of Singapore, Singapore huangc.scu@gmail.com, yiping.jin@upf.edu, wenqianglei@gmail.com 

## **Abstract** 

Human annotation is a time-consuming task that requires a significant amount of effort. To address this issue, interactive data annotation utilizes an annotation model to provide suggestions for humans to approve or correct. However, annotation models trained with limited labeled data are prone to generating incorrect suggestions, leading to extra human correction effort. To tackle this challenge, we propose ARAIDA, an analogical reasoning-based approach that enhances automatic annotation accuracy in the interactive data annotation setting and reduces the need for human corrections. ARAIDA involves an error-aware integration strategy that dynamically coordinates an annotation model and a k-nearest neighbors (KNN) model, giving more importance to KNN’s predictions when predictions from the annotation model are deemed inaccurate. Empirical studies demonstrate that ARAIDA is adaptable to different annotation tasks and models. On average, it reduces human correction labor by 11.02% compared to vanilla interactive data annotation methods. 

## **1 Introduction** 

Data annotation is a challenging task that involves a tradeoff between annotation quality and budget. While some platforms offer a cost-effective solution by relying on ML models to annotate data automatically[1] , the quality of such annotations is often compromised (Wang et al., 2022a). It is particularly true in the **limited data annotation** scenario where the annotation budget is limited or when unlabeled data are scarce (Ringger et al., 2007; Chaudhary et al., 2021; Huang et al., 2024). 

Human-machine **interactive annotation** methods were introduced to reduce annotation effort while maintaining annotation quality (Klie et al., 

> *Correspondence to Wenqiang Lei. 

> 1For example, https://aws.amazon.com/sagemaker/ groundtruth/. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation_images/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation.pdf-0001-14.png)


**----- Start of picture text -----**<br>
Unlabeled set Manual Annotation Labeled set<br>①  Input 𝑥𝑥𝑡𝑡 ②  Annotation<br>𝑥𝑥𝑡𝑡, 𝑦𝑦𝑡𝑡<br>Interactive Annotation<br>① 𝑥𝑥𝑡𝑡 ②  Suggestion ③  Accept suggestion  or<br>𝑥𝑥𝑡𝑡, 𝑓𝑓𝑡𝑡(𝑥𝑥𝑡𝑡) Correct it by re-annotation<br>④  Update Model Annotator<br>**----- End of picture text -----**<br>


Figure 1: Comparison between manual annotation and interactive annotation. 

2018, 2020; Le et al., 2021). As illustrated in Fig. 1, these methods introduce an _annotation model_ to suggest labels ( _model annotations_ ) to human annotators. The annotators accept a suggested label if it is correct. Otherwise, they have to correct the label manually. Compared to manual annotation, interactive annotation requires less human effort because human annotators only have to verify the model annotations instead of coming up with an answer from scratch, leading to potential speedup of the annotation process (Klie et al., 2020). 

Evidently, the annotation model’s accuracy is crucial because incorrect suggestions require additional human effort to rectify. Existing methods update the annotation model based on previously accepted or corrected data ( _ground-truth annotation_ ), aiming to reduce human corrections by improving prediction accuracy at each iteration (Klie et al., 2020; Wu et al., 2022). However, in the context of limited data annotation, the annotation model lacks sufficient labeled training data to reach a reasonable accuracy and is prone to providing incorrect suggestions (Rietz and Maedche, 2021). For example, in the span relation annotation example shown in Fig. 2 (blue), the annotation model continues to make mistakes on similar examples ([ _car_ , _tyre_ ]) even after the human annotator corrects the label ‘[ _tree_ , _leaf_ ]= _>component_ ’. As a result, this leads to more human corrections. Such a problem is crucial for interactive annotation and has been identified by recent work (Rietz and Maedche, 2021), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation_images/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Round 𝑡𝑡 Round 𝑡𝑡2 Round t2  with ARAIDA<br>[tree, leaf]: ? [car, tyre]: ? [car, tyre]: ? KNN Inference<br>KNN Annotation<br>component<br>[tree, leaf]: synonym [car, tyre]: synonym [car, tyre]: synonym Error prediction on<br>similar annotated data<br>Correction No  0.7* component 0.7<br>Correction Labors Correction + 0.3* synonym 0.3<br>Correct Wrong<br>[tree, leaf]: component Update [car, tyre]: component [car, tyre], component Error-aware Integration Strategy<br>**----- End of picture text -----**<br>


Figure 2: Example on span relation annotation. An under-trained annotation model results in more suggestion errors and increases human correction effort. ARAIDA improves the model annotation accuracy via the KNN model and the error-aware integration strategy for dynamical coordination of annotations. 

but it has yet to be addressed. 

Inspired by cognitive studies on efficient learning (Lake et al., 2017, 2015; Mitchell, 2021), finding that the human brain can learn from a few examples because our brain is continuously building analogies during the learning process of concepts to facilitate comprehension, we propose **A** nalogical **R** easoning- **A** ugmented **I** nteractive **D** ata **A** nnotation (ARAIDA), which is designed to improve interactive annotation under the limited data annotation setting. ARAIDA provides an annotation reference to the annotation model by retrieving previously human-labeled examples in the proximity of the example in consideration using the k-nearest neighbors (KNN) method. As illustrated in Fig. 2(red), the final suggestion combines the model annotation and the annotation reference provided by KNN via an error-aware integration strategy. This strategy dynamically coordinates the annotation model and KNN, giving more importance to KNN’s prediction if the predicted label from the annotation model is estimated to be inaccurate. 

We conduct simulated experiments for the limited data annotation task and estimate the human annotation effort based on the number of human corrections (or the number of suggestion errors) following Hwa (2000) and Kristjansson et al. (2004). We test ARAIDA on different wordlevel and sentence-level annotation tasks, combining with different annotation models (i.e., classic and LLM-based models). The result shows that ARAIDA consistently improves different annotation models’ accuracy across various tasks. On average, it reduces human corrections by 11.02%. Further analysis attributes this improvement to the few-shot capability of the KNN module and the error-aware integration strategy that effectively synergizes complementary annotations. In summary, 

our contributions are as follows: 

- Calling attention to the limited data annotation scenario. We highlight the under-trained problem of the annotation model, which is crucial in practice but overlooked in interactive annotation. 

- Introducing ARAIDA that involves a KNN module and an error-aware integration strategy to alleviate the under-trained problem by facilitating coordination between the two model annotators (i.e., the vanilla annotation model and KNN). 

- Demonstrating the efficacy of ARAIDA in enhancing suggestion accuracy, reducing human corrections, and showcasing its flexibility to combine with various annotation models through extensive experiments. 

## **2 Related Work** 

Our research is tied to interactive data annotation, human analogical reasoning (KNN), and retrievalbased language models. We provide a literature review and highlight our differences. 

**Interactive Data Annotation.** Interactive data annotation aims to reduce human annotation effort by incorporating an annotation model that suggests labels to human annotators during an interactive process (Klie et al., 2018, 2020; Le et al., 2021). The annotation model must be sample-efficient because, when we start a new annotation task, there are few labeled examples to learn from. Current studies focus on employing active learning (Klie et al., 2018; Laws et al., 2011; Casanova et al., 2020; Li et al., 2021; Huang et al., 2023) to prioritize annotating examples more likely to improve model accuracy. While active learning can reduce the required training data to some extent, it may not be effective in limited data annotation scenarios 

or when complex hypotheses or semantics are to be learned (Dasgupta, 2005; Rietz and Maedche, 2021). Another approach is to employ LLMs for automatic data annotation, which have demonstrated strong performance under zero-shot and few-shot settings (He et al., 2023; Gilardi et al., 2023). However, such performance might not be consistent for difficult tasks, as they may even perform worse than fine-tuned small language models (Xiao et al., 2023). Regardless of whether we use active learning and whether we use a classic or LLM-based annotation model, our empirical evidence demonstrates that ARAIDA can effectively decrease the amount of human corrections required. 

**KNN and Analogical Reasoning.** While KNN has been extensively utilized in NLP community (Wang et al., 2019; Liu et al., 2023; Wang et al., 2022b), its underlying mechanism is often overlooked. To shed light on this, cognitive studies (Lake et al., 2017, 2015; Mitchell, 2021) revealed that the KNN inference process aligns with human analogical reasoning, enabling efficient learning (Lake et al., 2017, 2015). In particular, analogical reasoning establishes connections between relevant aspects of the current task and past experiences, forming abstractions that enhance human reasoning capabilities (Mitchell, 2021). In this context, KNN facilitates sample-efficient learning by leveraging similarities between the example to be labeled and previously annotated examples, resulting in exemplary solutions (Bautista et al., 2016), which reduce the training data requirement. 

**Retrieval-Based Language Models.** There is growing interest in enhancing the output of language models by incorporating a retrieval module (usually KNN or alike) that interpolates with a datastore built from the training data (Khandelwal et al., 2019; Kassner and Schütze, 2020). Compared to vanilla language models, retrieval-based models ground the predictions in labeled training examples, potentially yielding better explainability and sample efficiency (Asai et al., 2023). This approach has shown promising results in tasks such as machine translation (Khandelwal et al., 2021; Liu et al., 2023), named entity recognition (Wang et al., 2022b), and question answering (Kassner and Schütze, 2020). While some studies have explored the use of dynamically adjusted combination weights between the language model and the retrieval module (Wang et al., 2021; Zheng et al., 2021; Jiang et al., 2021), our method differs signifi- 

cantly for two main reasons: 1) Different tasks. We are the pioneers in introducing KNN to the interactive data annotation task, whereas these methods are primarily designed for machine translation. 2) Different techniques. We adjust the weight by estimating the error of model predictions for each data point (e.g., sentence), whereas these methods learn the weight for each token without error estimation. 

## **3 ARAIDA: The Proposed Method** 

We present ARAIDA, an analogical reasoningbased method for interactive data annotation that provides an annotation reference to the annotation model by retrieving previously human-labeled examples in the proximity of the example in consideration. We detail the KNN inference module in Section 3.1 and the error-aware integration strategy in Section 3.2. Finally, the optimization details are provided in Section 3.3. 

**Task Formalization and Overview.** Let _X_ denote the dataset that needs to be annotated, with _C_ being the number of classes. Given a data batch _xt_ at time _t_ , the annotation model _ft_ predicts label vectors _ft_ ( _xt_ ) _∈ R[|][x][t][|×][C]_ , and the KNN module _gt_ infers label vectors _gt_ ( _xt_ ) _∈ R[|][x][t][|×][C]_ using previously annotated data stored in a datastore _At_ . Then, we estimate the probability _λt ∈ R[|][x][t][|×]_[1] of the annotation model’s predictions _ft_ ( _xt_ ) being reliable, i.e., _argmax_ ( _ft_ ( _xt_ )) = _yt_ , where _argmax_ ( _·_ ) returns indices of the classes with the highest predicted probability and _yt ∈ R[|][x][t][|×]_[1] are the ground truth labels. Finally, we use _λt_ to weigh the two predictions _ft_ ( _xt_ ) and _gt_ ( _xt_ ) through a linear weighted combination: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation_images/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation.pdf-0003-07.png)


Notably, closed-source language models such as ChatGPT produce discrete labels rather than predicted distributions. Therefore, we cannot combine its predictions with KNN’s using linear combination. In such case, we use binary values (0 or 1) for _λt_ , which acts as a function allocation to determine whether _ft_ or _gt_ should apply to each example. Once the human approves or corrects the final suggestions, the datastore _At_ is updated with the newly arrived data batch and its corresponding labels. In addition, the annotation model _ft_ (if applicable), KNN module _gt_ , and weighting strategy _λt_ are updated via back-propagation. 

## **3.1 KNN Inference** 

We utilize a weighted KNN to perform inference, � _a∈ρi[w][a][y][a]_ defined as _gt_ ( _x[i] t_[)][=] � _a∈ρi[w][a]_[,][where] _[ρ][i][∈][A][t]_ is the _k_ nearest neighbors of each example _x[i] t_[,] _ya_ corresponds to the human annotation of each neighbor _a ∈ ρi_ . The similarity between _x[i] t_[and] _a_ is measured by _wa_ = _d_ ( _x_ 1 _[i] t[,a]_[)][, where] _[ d]_[(] _[x] t[i][, a]_[)][=] _∥wknn_ ( _x[i] t[−][a]_[)] _[∥]_[2][is][a][distance][metric][parameter-] ized by _wknn_ . We use the similarity measure to retrieve _ρi_ . To avoid overconfidence in KNN inference, we apply label smoothing to the labels of the retrieved neighbors. Specifically, we set _ya_ = _ya_ (1 _− α_ ) + _α/C_ , where _α_ = 1 _− C_[1][.] 

**Datastore Maintenance Strategy.** The datastore _At_ consisting of historically annotated data grows in size as the interactive annotation continues, causing the KNN’s retrieval to be less time-efficient. To address this issue, we impose a constraint on the maximum datastore size using a pre-defined hyperparameter. We propose a class-aware maintenance strategy. Precisely, if _At_ exceeds its budget, data that is from the majority class[2] and most similar to its class prototype[3] is discarded first. This strategy ensures that the datastore contains as many labeled data from different classes as possible while minimizing the impact on the class prototype. Appendix A.1 and A.3 present experiments using different datastore sizes and maintenance strategies. 

## **3.2 Error-aware Integration Strategy** 

**Motivation.** A popular method to combine annotations from two models (i.e., annotation model and KNN) is to use a weighted linear combination with a constant weight[4] (Liu et al., 2023; Wang et al., 2022b). However, assuming that one model consistently outperforms the other on all unlabeled data is unrealistic. Furthermore, both models are updated with each new batch of data in interactive data annotation, and their relative performance will alter, making it infeasible to find the optimal weight through a one-off hyperparameter tuning. To address this issue, we propose an error-aware integration strategy that automatically assigns weights to different models, relying more on KNN inference when the annotation model’s prediction is estimated to be inaccurate. 

> 2The majority class refers to the class with the highest frequency in _At_ . 

> 3The class prototypes are the average of the feature vectors in _At_ that belong to each class. 

> 4Equivalent to when _λt_ ( _xt_ ) in Eq.1 is a constant. 

**Error Estimation of Model Annotation.** We base on the intuition that if the model _ft_ consistently makes mistakes on previous examples similar to the current data point _x[i] t_[, then its prediction] _ft_ ( _x[i] t_[)][ will likely be incorrect.][To achieve this, we] parameterize the integration strategy _λt_ as a neural network to learn from the customized input. 

- Customized Inputs. For each data point _x[i] t_[,][we] derive the input x _[i] t_[to the integration strategy] _[ λ][t]_[,] which considers the local error estimation _Et[i]_ and local density _Dt[i]_[.][Specifically,] _[ E] t[i]_[is a vector] with elements _e[i] t,j_[=][ 1][[] _[argmax]_[(] _[f][t]_[(] _[a][j]_[)) =] _[ y][a] j_[]] indicates if the annotation model _ft_ predicted correctly on each annotated example _aj ∈ ρi_ in the _k_ nearest neighbors of _x[i] t_[.][The local density] _Dt[i]_[is a distance vector, with each element being] _d_ ( _x[i] t[, a][j]_[)][.][These two vectors are combined using] the element-wise multiplication operator _⊙_ to create the input: x _[i] t_[=] _[ D] t[i][⊙][E] t[i][−][D] t[i][⊙]_[(1] _[ −][E] t[i]_[)][.] Notably, x _[i] t_[measures][the][error][regularity][of] _[x][i] t_ among its neighborhood, as the more positive values in the vector x _[i] t_[, the less likely] _[ f][t]_[would] make an error on _x[i] t_[.] 

- Learning Objectives. We collect the ground truth labels _yt_ through human feedback. To optimize the error-aware integration strategy _λt_ , we use a mean squared error (MSE) loss, denoted as _ℓ[t] d_[(] _[y][t][, f][t]_[(] _[x][t]_[)] _[, λ][t]_[) =][ MSE][(][1][[] _[argmax]_[(] _[f][t]_[(] _[x][t]_[)) =] 

- _yt_ ] _, λt_ (x _t_ )), where 1[ _·_ ] indicates whether the ground truth labels _yt_ are the same as the predictions _ft_ ( _xt_ ). The purpose of this loss function is to guide _λt_ by encouraging it to predict errors made by _ft_ . 

## **3.3 Optimization of ARAIDA** 

To simplify the optimization process, we independently optimize the annotation model _ft_ , KNN model _gt_ , and error-aware integration strategy _λt_ . We treat human feedback _yt_ as the ground truth following previous studies on interactive annotation (Klie et al., 2018, 2020; Le et al., 2021). It is worth noting that ARAIDA supports any taskspecific annotation model and uses its corresponding loss function _ℓf_ to update the parameters. Combining the _ℓd_ loss and the negative log-likelihood loss _ℓg_ to optimize KNN, we formulate the final loss function as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation_images/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation.pdf-0004-13.png)


where _Bt_ represents the total data accumulated until round _t_ . There are two challenges to optimizing this objective function. Firstly, the operator used in KNN to retrieve the _k_ nearest neighbors is not differentiable. To address this problem, we utilize the Gumbel-softmax-based reparameterization trick (Jang et al., 2016) to facilitate the optimization process. Secondly, the loss function _L_ presents a bi-level optimization problem, where the optimization of _λt_ is nested within the optimization problems of _ft_ and _gt_ . As a result, we update _ft_ , _gt_ , and _λt_ iteratively using a coordinate-descent approach. Formally, at each optimization iteration _k_ , we have network parameters _θf[k]_[,] _[θ] g[k]_[,][and] _[θ] λ[k]_[corresponding][to] _[f][k]_[,] _[g][k]_[,][and] _[λ][k]_[.] The update procedures are as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation_images/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation.pdf-0005-01.png)


## **4 Experiments** 

We conduct extensive experiments to assess ARAIDA’s effectiveness in the limited data annotation scenario. Our primary focus is to assess whether ARAIDA can decrease the human effort required for corrections by providing more precise annotations at various stages of the annotation process (see Section 4.2). Furthermore, we perform a comprehensive examination to investigate the behavior and impact of KNN and the error-aware integration strategy (see Section 4.3). Additional analysis of our error-aware integration strategy is presented in Section 4.4. Lastly, we analyze the sensitivity of the parameters in Appendices A. 

## **4.1 Experimental Setup** 

**Tasks & Datasets.** We experiment with wordand sentence-level annotation tasks. These tasks have been highlighted as crucial in various web applications (Yao et al., 2021; Marcos-Pablos and García-Peñalvo, 2020; Lee et al., 2022). To simulate the scenario of limited data annotation, we follow Dou et al. (2019) by imposing dataset size restrictions, ranging from 1 _K_ to 5 _K_ . Table 1 overviews the dataset statistics. 

- **Word-level annotation** . We focus on the knowledge graph completion task, which annotates the semantic classes of input word pairs (e.g., ‘[ _tree_ , _leaf_ ]= _>component_ ’). We use two benchmark knowledge graph datasets in our experi- 

|**Dataset**|**# Val.**|**Classes**|
|---|---|---|
|WN18RR|3,034|Hypernym; Derivation; Member;|
|||Component; Synset; Synonym;|
|||Verbgroup; Instance of hypernym;|
|FreeBase|5,116|Contains; Country; Track_role;|
|||Profession; Group_role; Adjoins;|
|||Film_release;Nutrient|
|IMDB|5,000|Positive;Negative|
|SST-5|1,101|Strong positive; Positive; Neutral;|
|||Negative;Strongnegative|



Table 1: Statistics of datasets. For each dataset, we randomly sample 5K examples from the original training dataset to form the unlabeled data, and the validation dataset is taken from the original dataset. Table 4 presents the mapping from the original categories to the categories we use. 

- ments, namely the WN18RR (Dettmers et al., 2018)[5] and Freebase (Bollacker et al., 2008)[6] dataset. We experiment with the eight most frequent classes for each dataset. 

- **Sentence-level annotation** . We consider the sentiment classification task and experiment on two benchmark datasets, including SST-5 (Socher et al., 2013)[7] , and IMDB (Maas et al., 2011)[8] . SST-5 dataset contains categories on a scale of 1-5 while IMDB contains two categories (positive/negative). 

**Evaluation Metric.** We aim to minimize the total human corrections (i.e., the total model suggestion errors) annotating a given amount of data using the interactive annotation process. Therefore, we report the _Machine Cumulative Accuracy_ (MCA), defined as the total correct suggestions divided by the total suggestions for different dataset sizes. To assess the performance of each method in the limited data annotation scenario, we present the mean and the corresponding standard deviation by varying the dataset size ({1K, 2K, 3K, 4K, 5K}). 

**Annotation Models.** To verify the generalizability of ARAIDA, we apply it in conjunction with different annotation models: 

- Classic annotation models. We utilize lightweight annotation models following 

   - 5https://paperswithcode.com/dataset/wn18rr 

- 6https://www.microsoft.com/en-us/download/ 

- details.aspx?id=52312 

   - 7https://nlp.stanford.edu/sentiment/code.html 

   - 8https://www.kaggle.com/ 

datasets/lakshmi25npathi/ 

imdb-dataset-of-50k-movie-reviews 

|**Annotation**<br>**Model**|**Without Active Learning (AL)**|**Without Active Learning (AL)**|**Without Active Learning (AL)**|**Without Active Learning (AL)**|**With Active Learning (AL)**|**With Active Learning (AL)**|**With Active Learning (AL)**|**With Active Learning (AL)**|
|---|---|---|---|---|---|---|---|---|
||**Word-level Annotation**||**Sentence-level Annotation**||**Word-level Annotation**||**Sentence-level Annotation**||
||WN18RR|FreeBase|IMDB|SST-5|WN18RR|FreeBase|IMDB|SST-5|
|Dist./FT<br>Dist./FT + ARAIDA|50.44±1.02<br>32.47±12.37<br>70.18±8.43<br>36.02±3.14<br>**52.16±1.37**<br>**43.02±6.43**<br>**79.33±2.81**<br>**37.21±3.03**||||47.77±0.91<br>26.92±10.85<br>66.98±6.31<br>35.20±3.76<br>**49.54±0.84**<br>**39.06±4.57**<br>**75.84±1.14**<br>**37.02±2.50**||||
|LLaMa2<br>LLaMa2 + ARAIDA|31.35±1.89<br>24.41±1.77<br>80.21±2.64<br>37.83±1.64<br>**45.15±1.96**<br>**38.20±1.91**<br>**88.47±2.03**<br>**42.03±1.88**||||31.35±1.89<br>24.41±1.77<br>80.21±2.64<br>37.83±1.64<br>**46.33±2.01**<br>**38.79±1.96**<br>**89.68±2.25**<br>**42.61±1.91**||||
|LLaMa2_sft_<br>LLaMa2_sft_+ ARAIDA|58.24±2.79<br>53.11±1.38<br>94.06±9.02<br>46.84±6.30<br>**60.74±2.33**<br>**55.23±1.46**<br>**95.15±9.32**<br>**49.62±5.98**||||59.28±2.57<br>55.39±2.01<br>95.13±10.15<br>47.45±6.17<br>**61.02±2.18**<br>**56.71±2.09**<br>**95.88±12.27**<br>**49.51±6.03**||||



Table 2: Machine cumulative accuracy (MCA) scores using various methods. We report the averaged results across varying amounts of data. LLaMa2 with AL has identical performance as LLaMa2 because the vanilla LLaMa2 model is not updated during the interactive annotation process. Thus, the data order does not impact the performance. When combined with ARAIDA, the performances w/ and w/o AL are different because the KNN and integration strategy models are updated. 

previous works (Desmond et al., 2021; Chen et al., 2020; Hedderich et al., 2021). Specifically, for word-level tasks, we use a distributional model (Roller et al., 2014; Kober et al., 2021) with pretrained GloVe word embeddings embedding (Pennington et al., 2014)[9] . For sentence-level tasks, we use FastText (Joulin et al., 2017) to derive the sentence embeddings. We denote this baseline as _Dist./FT_ . 

- LLM-based annotators. We also use large language models (LLMs) as annotation models, whose few-shot and in-context learning capabilities might help with the limited data annotation process. We experiment with LLaMa27B (Touvron et al., 2023) and ChatGPT (Ouyang et al., 2022)[10] . We use zero-shot and few-shot prompts for ChatGPT (denoted as _ChatGPTzero_ and _ChatGPTfew_ ). Detailed prompts can be found in Table 7. For LLaMa2, we consider both the vanilla _LLaMa2_ that uses the same zero-shot prompts as ChatGPT _zero_ and _LLaMa2sft_ , which is fine-tuned using an open-source toolkit[11] during the interactive annotation process. Finetuning data examples can be found in Table 8. 

**Impact of Active Learning.** Regardless of the annotation model, the sequence of the data to annotate also affects the amount of human corrections. Intuitively, if we show unambiguous examples first, few corrections are needed. However, the annotation model and KNN may not learn to handle more challenging examples and may subsequently 

> 9We did not use contextualized embeddings because the word-level task in our experiment has no context. 

> 10The gpt-3.5-turbo checkpoint in OpenAI’s API (https://platform.openai.com/docs/models/ gpt-3-5). 

> 11https://github.com/Alpha-VLLM/ LLaMA2-Accessory 

make more mistakes. Previous studies focused on applying active learning in interactive annotation to enhance the annotation models’ sample efficiency (Laws et al., 2011; Klie et al., 2018; Li et al., 2021). To study the impact of active learning in the limited data annotation scenario, we compare an uncertainty-based active learning method with random data ordering for different annotation models with and without ARAIDA. Note that we omit the ChatGPT with AL results because we are unable to estimate its prediction uncertainty accurately. 

**Implementation Details.** All experiments are carried out on a machine with Intel(R) Xeon(R) Gold 5317 CPU @ 3.00GHz and a GeForce RTX 3090 GPU. For simplicity, we implement our integration strategy using a three-layer, fully-connected network with ReLu activation and dropout. For KNN, we set _k_ = 20 for _ρt_ . KNN runs in the embedding space of _text-embedding-ada-002_ (Neelakantan et al., 2022) when combining with ChatGPT. For Dist./FT and LLaMa2 models, KNN runs in the corresponding model’s embedding space. Moreover, we leave the details of LLM prompts and mode fine-tuning examples in Appendix D. 

## **4.2 Main Result** 

We evaluate the effectiveness of ARAIDA in enhancing the model annotation quality, hence reducing human correction effort. Table 2 and Figure 3 show the machine cumulative accuracy scores averaged across varying amounts of data ( _{_ 1 _K,_ 2 _K,_ 3 _K,_ 4 _K,_ 5 _K}_ ) for different experimental settings. We make the following observations: 

**ARAIDA reduces human corrections consistently** . As illustrated in Table 2 and Figure 3, ARAIDA consistently improves the model suggestion accuracy across different annotation models and annotation tasks. Specifically, it achieves a 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation_images/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation.pdf-0007-00.png)


**----- Start of picture text -----**<br>
95 ChatGPTzeroChatGPTzero<br>ARAIDA+ChatGPTzeroChatGPTzero + ARAIDA<br>85 ChatGPTfewChatGPTfew<br>75 ARAIDA+ChatGPTfewChatGPTfew + ARAIDA<br>65<br>55<br>45<br>35<br>WN18RR FreeBase IMDB SST-5<br>Word-level Annotation Sentence-level Annotation<br>92.44 93.52 94.78 94.92<br>50.42 48.41<br>47.62<br>43.11<br>38.85 41.3<br>51.3 52.53 51.6 52.26 53.85 55.12<br>**----- End of picture text -----**<br>


Figure 3: MCA scores using ChatGPT-based methods. We omit the ChatGPT with AL results because we are unable to estimate its prediction uncertainty. ARAIDA can further improve ChatGPT’s performance 

13.06% performance gain in MCA on Dist./FT, 3.85% on LLaMa2 _sft_ , 16.80% on Dist./FT with AL, 2.61% on LLaMa2 _sft_ with AL, 30.48% on LLaMa2, 8.81% on ChatGPT _zero_ , and 1.55% on ChatGPT _few_ . On average, it achieves an 11.02% performance gain in model annotation accuracy, translating into a reduction in human correction effort in practice. 

**Active learning does not always help.** When comparing the annotation models with and without active learning, we observe that active learning does not always improve model performance and can even harm the performance in some instances (e.g., Dist./FT). We hypothesize that the more challenging cases selected by active learning might require more training data for the models to correct their predictions (Dasgupta, 2005; Rietz and Maedche, 2021), which are unavailable in limited data annotation scenarios. Instead of relying on an annotation model alone, ARAIDA acts as a posthoc “plug-in” that fixes the annotation model’s mistakes using retrieved annotation references and yields a robust improvement under various settings with and without active learning. 

**LLMs are strong annotation models. ARAIDA can improve them further.** LLMs perform better than classic distributional models, especially for sentence-level tasks. Furthermore, LLaMa2 with fine-tuning consistently outperforms the vanilla LLaMa2 model, and few-shot ChatGPT consistently beats its zero-shot counterpart. Interestingly, comparing these two pairs of annotation models, we observe that ARAIDA brings a more substantial improvement to weaker models. When the annotation models are already strong (in the case of LLaMa2 _sft_ and ChatGPT _few_ ), ARAIDA is 

more conservative in making corrections, yielding a smaller but consistent improvement. It demonstrates ARAIDA’s robustness to combine with annotations models with different performances. 

|**Annotation**<br>**Word-level Annotation**<br>**Sentence-level Annotation**|**Annotation**<br>**Word-level Annotation**<br>**Sentence-level Annotation**|**Annotation**<br>**Word-level Annotation**<br>**Sentence-level Annotation**|**Annotation**<br>**Word-level Annotation**<br>**Sentence-level Annotation**|**Annotation**<br>**Word-level Annotation**<br>**Sentence-level Annotation**|
|---|---|---|---|---|
|**Model**<br>**WN18RR**<br>**FreeBase**<br>**IMDB**<br>**SST-5**<br>Dit/FT|||||
|s.<br>ARAIDA<br>**52.16**_±_**1.37**<br>**43.02**_±_**6.43**<br>**79.33**_±_**2.81**<br>**37.21**_±_**3.03**<br>-_w/o_KNN<br>50.44_±_1.02<br>32.47_±_12.37<br>70.18_±_8.43<br>36.02_±_3.14<br>-_w/o f_(_·_)<br>50.28_±_3.01<br>41.52_±_5.87<br>78.48_±_0.68<br>35.02_±_1.37<br>-_w/_ const.<br>51.85_±_4.77<br>41.78_±_6.78<br>78.58_±_3.94<br>37.07_±_2.68|||||
|LLaMa2_sft_|||||
|ARAIDA<br>-_w/o_KNN<br>-_w/o f_(_·_)<br>|**60.74**_±_**2.33**<br>58.24_±_2.79<br>45.23_±_2.58<br>|**55.23**_±_**1.46**<br>53.11_±_1.38<br>39.82_±_3.41<br>|**95.15**_±_**9.32**<br>94.06_±_9.02<br>89.13_±_0.55<br>|**49.62**_±_**5.98**<br>46.84_±_6.30<br>42.93_±_1.71<br>|
|-_w/_ const.|58.24_±_2.79|53.11_±_1.38|94.06_±_9.02|46.84_±_6.30|
|ChatGPT_zero_|||||
|ARAIDA<br>-_w/o_KNN<br>-_w/o f_(_·_)<br>-_w/_ const.|**50.42**_±_**1.37**<br>47.62_±_2.89<br>48.73_±_2.64<br>48.73_±_2.64|**43.11**_±_**1.83**<br>38.85_±_2.23<br>41.30_±_5.04<br>41.30_±_5.04|**93.52**_±_**1.24**<br>92.44_±_0.95<br>90.61_±_0.26<br>92.44_±_0.95|**48.41**_±_**1.06**<br>41.30_±_1.72<br>45.84_±_1.12<br>45.84_±_1.12|
|ChatGPT_few_|||||
|ARAIDA<br>-_w/o_KNN<br>-_w/o f_(_·_)<br>-_w/_ const.|**52.53**_±_**1.67**<br>51.30_±_1.72<br>48.73_±_2.64<br>51.30_±_1.72|**52.26**_±_**3.44**<br>51.60_±_3.15<br>41.30_±_5.04<br>51.60_±_3.15|**94.92**_±_**1.02**<br>94.78_±_0.99<br>90.61_±_0.26<br>94.78_±_0.99|**55.12**_±_**1.36**<br>53.85_±_2.03<br>45.84_±_1.12<br>53.85_±_2.03|



Table 3: Ablation study on the KNN and the error-aware integration strategy modules. We report the MCA scores using various methods, averaging results with different amounts of data. Error-aware integration strategy effectively coordinates the two annotators. 

## **4.3 Ablation Study** 

This section aims to perform a comprehensive examination to investigate the behavior and impact of the KNN and out integration strategy. We conduct an ablation study to analyze the effectiveness of each component. In particular, we consider the following baselines: 

- **ARAIDA w/o KNN:** Using the annotation model alone to suggest labels. Equivalent to _λt_ ( _·_ ) = 1. 

- **ARAIDA w/o f(** _·_ **):** Using KNN alone to suggest labels. Equivalent to _λt_ ( _·_ ) = 0. 

- **ARAIDA w/ const.:** Using a constant _λ[∗] t_[value] for all examples, in contrast to ARAIDA, which varies _λt_ for different examples. In our experiments, we report the result with the best _λ[∗] t_[tuned] on the validation set. 

The ablation test results are presented in Table 3. Due to limited space, we omit the result for vanilla LLaMa2, which is much weaker than other LLMbased baselines. The detailed observations are provided below. 

**KNN is a strong stand-alone annotator.** Table 3 reveals that although KNN (ARAIDA _w/o f(·)_ ) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation_images/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation.pdf-0008-00.png)


**----- Start of picture text -----**<br>
70% 70% 100% 50%<br>60%<br>60% 40%<br>50% 80%<br>50% 40% 30%<br>60%<br>40%30% 𝑀𝑀 𝐴𝐴𝐴𝐴𝑓𝑓𝑓𝑓 𝑜𝑜 𝜆𝜆> 0.5𝜆𝜆≤0.5 30%20% 40% 20%10%<br>𝑀 𝐴𝐴𝐾 𝐾𝐾 𝑜 𝜆𝜆≤0.5 10%<br>20% 0% 20% 0%<br>1 51K     2K     3K     4K    5K9 13 17 21 25 29 33 37 41 1 51K     2K     3K     4K    5K9 13 17 21 25 29 33 37 1 51K     2K     3K     4K    5K9 13 17 21 25 29 33 37 1 51K     2K     3K     4K    5K9 13 17 21 25 29 33 37<br>WN18RR FreeBase IMDB SST-5<br>**----- End of picture text -----**<br>


Figure 4: Analyzing our integration strategy with the Dist./FT model. The solid lines show the MAC scores of the annotation model _f_ ( _·_ ), separated by examples with _λ >_ 0 _._ 5 (higher weights assigned to the annotation model _f_ ( _·_ )) and _λ ≤_ 0 _._ 5 (higher weights assigned to KNN). The dotted line shows KNN’s performance on the latter set. 

does not match the performance of ARAIDA, it obtains comparable accuracy as using the annotation model alone (ARAIDA _w/o KNN_ ). This result is surprising, given KNN’s simplicity compared to the annotation models (including LLMs). Such the results also highlight that significance and effectiveness of analogical reasoning, allowing humans to reason more effectively (Mitchell, 2021). 

**Error-aware integration strategy effectively coordinates the two annotators.** Table 3 shows that ARAIDA’s error-aware integration strategy achieves consistent performance gain compared to using a constant weight _λ[∗] t_[.][This][in][turn][also] confirms the original intention behind our construction of the error-aware integration strategy: it is infeasible to find the optimal weight through a one-off hyperparameter tuning. Here, it is worth noting that ChatGPT outputs discrete labels rather than probabilistic vectors. In this case, the constant strategy reduces into an indicator function: if _λ[∗] t >_ 0 _._ 5, _Ft_ ( _x_ ) = _ft_ ( _x_ ) _, ∀xt_ ; otherwise, _Ft_ ( _x_ ) = _gt_ ( _x_ ) _, ∀xt_ . In this case, ARAIDA _w/ const._ prefers ChatGPT or KNN based on their performance. However, it cannot improve the annotation quality beyond the individual components (i.e., the KNN and the annotation model) due to the discrete output of ChatGPT. 

## **4.4 Qualitative Analysis** 

To shed light on how the error-aware integration strategy works, we measure the cumulative accuracy of the annotator model and KNN throughout the annotation process for each dataset and present the result with the Dist./FT annotation model in Figure 4. We also separate the cases where _λ >_ 0 _._ 5 (higher weights assigned to the annotation model _f_ ( _·_ )) and _λ ≤_ 0 _._ 5 (higher weights assigned to KNN). Our observations are as follows. 

Firstly, we observe that there is a substantial gap between the two solid lines ( _MCAf_ on _λ >_ 0 _._ 5 

and _MCAf_ on _λ ≤_ 0 _._ 5), showing that our error estimation model effectively identifies cases where _f_ ( _·_ ) is likely to error and assigns it a lower weight. Secondly, KNN reaches a reasonable accuracy much faster than the annotation model at the initial stage of the interactive annotation process. Even as the number of annotated examples increases, its accuracy is still higher than the annotation model _f_ ( _·_ ) when _λ ≤_ 0 _._ 5. This result reveals that KNN compensates for the performance _·_ deficiencies of _f_ ( ) on data where it is more likely to make a mistake. Therefore, combining KNN using the error-aware integration strategy in ARAIDA leads to an overall improvement in annotation quality, hence reducing human correction effort. 

## **5 Conclusion** 

In interactive data annotation, an annotation model suggests labels to human annotators to verify. However, the annotation model is prone to errors when trained on limited labeled data. To tackle this challenge, we proposed ARAIDA, an approach inspired by analogical reasoning, to compensate for the performance deficiencies of the annotation model and correct its mistakes using an error-aware integration strategy. Extensive experiments demonstrated that ARAIDA is flexible to combine with different annotation models across various tasks and yields consistent improvement in label suggestion accuracy, which leads to a reduction of human correction effort. In this study, our method explores a new solution to bring more flexibility by allowing the human to design any preferred annotation model according to different annotation tasks. We are devoted to optimizing human-machine utilities by emphasizing the learning of task-specified concepts efficiently from a few human demonstrations. In future work, we plan to extend ARAIDA to other annotation tasks and develop it as a general toolkit that can benefit the NLP community. 

## **6 Acknowledgements** 

This work was supported in part by the National Natural Science Foundation of China (No. 62272330); in part by the Fundamental Research Funds for the Central Universities (No. YJ202219). 

## **7 Limitations** 

**Human Studies.** This work aims to reduce human correction effort in interactive data annotation. We follow previous work (Hwa, 2000; Kristjansson et al., 2004) to use the number of model suggestion errors to approximat ~~e th~~ e human correction effort needed. However, th ~~e ac~~ tual effort needed depends on the particular example and the type of errors (e.g., whether it is obvious). Ideally, we would involve human annotators and measure the saving of annotation time. However, due to the large number of experimental settings, conducting human studies with each annotation model and ablation baseline was infeasible. 

**Error-Prone Human Annotation.** Thi ~~s paper~~ treats human annotations as ground truth following previous studies in interactive data annotation (Klie et al., 2018; Le et al., 2021). However, uncertainty and inconsistency of human annotations do occur. We refer readers to the literature on handling errorprone human annotation, such as crowd-sourced data annotation (Larson et al., 2020). 

Although human annotation errors are not the focus of this work, we explore ARAIDA’s performance under synthesized label noise conditions. We consider the crowd-sourced data annotation scenario and assume that each human annotator _hi_ makes mistakes with the latent probability _p[i] e[∼]_ (0 _,_ 0 _._ 3). We set the total number of annotators _O_ = 10 and sample their corresponding error probabilities _Pe_ = _{p_[1] _e[, p]_[2] _e[, ..., p][O] e[}]_[.][Then,][we sample] _ui_ from a uniform distribution _U_ (0 _,_ 1) for each annotation. If _ui ≤ p[i] e_[,] _[h][i]_[assigns][a][randomly] sampled incorrect label; otherwise, it assigns the correct one. We use majority voting of the 10 annotators to obtain the final annotations following Shirani et al. (2019). 

We slightly modified ARAIDA’s datastore maintenance strategy. When the datastore exceeds its budge, we discard the data from the majority class and most **dis-similar** to its class prototype instead of removing the most similar one (as in the original ARAIDA strategy). This strategy may help remove incorrectly labeled data. The experimental result 

in Figure 5 shows that ARAIDA still outperforms the baseline. The modified datastore maintenance strategy (ARAIDA-dis) further improves the performance by a slight margin. Further research and more rigorous experiments are required to address the human annotation noise problem in interactive annotation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation_images/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation.pdf-0009-08.png)


**----- Start of picture text -----**<br>
40%<br>35%<br>30%<br>Dist./FT<br>25% ARAIDA<br>ARAIDA-dis<br>20%<br>1 61K    2K  3K    4K   5K111621263136<br>**----- End of picture text -----**<br>


Figure 5: MAC scores of various methods with synthesized label noise on the SST-5 dataset. Dist./FT is used as the annotation model. ARAIDA-dis refers to ARAIDA with a modified datastore maintenance strategy. 

**Latency.** The KNN c ~~om~~ ponent requires retriev- ~~ing simil~~ ar examples a ~~s th~~ e input data, which may limit our method’s time efficiency when the datastore size is large. To address this problem, besides the proposed datastore management strategy, we can also employ an efficient similarity search library such as FAISS[12] to speed up the retrieval. 

## **References** 

- Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen. 2023. Retrieval-based language models and applications. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts)_ , pages 41–46, Toronto, Canada. Association for Computational Linguistics. 

- Miguel A Bautista, Artsiom Sanakoyeu, Ekaterina Tikhoncheva, and Bjorn Ommer. 2016. Cliquecnn: Deep unsupervised exemplar learning. In _Advances in Neural Information Processing Systems_ , volume 29. Curran Associates, Inc. 

- Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: a collaboratively created graph database for structuring human knowledge. In _Proceedings of the 2008 ACM SIGMOD international conference on Management of data_ , pages 1247–1250. 

- Arantxa Casanova, Pedro O Pinheiro, Negar Rostamzadeh, and Christopher J Pal. 2020. Reinforced active learning for image segmentation. _arxiv_ . 

- 12https://github.com/facebookresearch/faiss 

- Aditi Chaudhary, Antonios Anastasopoulos, Zaid Sheikh, and Graham Neubig. 2021. Reducing confusion in active learning for part-of-speech tagging. _TACL_ , 9:1–16. 

- Xu Chen, Changying Du, Xiuqiang He, and Jun Wang. 2020. Jit2r: A joint framework for item tagging and tag-based recommendation. In _Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval_ , pages 1681–1684. 

- Sanjoy Dasgupta. 2005. Coarse sample complexity bounds for active learning. In _Proceedings of the 18th International Conference on Neural Information Processing Systems_ , pages 235–242. 

- Ofer Dekel, Shai Shalev-Shwartz, and Yoram Singer. 2005. The forgetron: A kernel-based perceptron on a fixed budget. _NeurIPS_ , 18. 

- Michael Desmond, Evelyn Duesterwald, Kristina Brimijoin, Michelle Brachman, and Qian Pan. 2021. Semiautomated data labeling. In _Proceedings of the NeurIPS 2020 Competition and Demonstration Track_ , volume 133 of _Proceedings of Machine Learning Research_ , pages 156–169. PMLR. 

- Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. 2018. Convolutional 2d knowledge graph embeddings. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 32. 

- Zi-Yi Dou, Keyi Yu, and Antonios Anastasopoulos. 2019. Investigating meta-learning algorithms for low-resource natural language understanding tasks. In _EMNLP_ , pages 1192–1197. 

- Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. Chatgpt outperforms crowd-workers for textannotation tasks. _arXiv preprint arXiv:2303.15056_ . 

- Xingwei He, Zhenghao Lin, Yeyun Gong, Hang Zhang, Chen Lin, Jian Jiao, Siu Ming Yiu, Nan Duan, Weizhu Chen, et al. 2023. Annollm: Making large language models to be better crowdsourced annotators. _arXiv preprint arXiv:2303.16854_ . 

- Michael A Hedderich, Lukas Lange, and Dietrich Klakow. 2021. Anea: distant supervision for lowresource named entity recognition. _the Practical Machine Learning For Developing Countries Workshop at ICLR_ . 

- Chen Huang, Yang Deng, Wenqiang Lei, Jiancheng Lv, and Ido Dagan. 2024. Selective annotation via data allocation: These data should be triaged to experts for annotation rather than the model. 

- Chen Huang, Peixin Qin, Wenqiang Lei, and Jiancheng Lv. 2023. Reduce human labor on evaluating conversational information retrieval system: A humanmachine collaboration approach. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 10876–10891, Singapore. Association for Computational Linguistics. 

- Rebecca Hwa. 2000. Sample selection for statistical grammar induction. In _2000 Joint SIGDAT Conference on Empirical Methods in Natural Language Processing and Very Large Corpora_ , pages 45–52. 

- Eric Jang, Shixiang Gu, and Ben Poole. 2016. Categorical reparameterization with gumbel-softmax. _arxiv_ . 

- Qingnan Jiang, Mingxuan Wang, Jun Cao, Shanbo Cheng, Shujian Huang, and Lei Li. 2021. Learning kernel-smoothed machine translation with retrieved examples. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 7280–7290, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. 2017. Bag of tricks for efficient text classification. In _EACL_ , pages 427–431. Association for Computational Linguistics. 

- Nora Kassner and Hinrich Schütze. 2020. BERT-kNN: Adding a kNN search component to pretrained language models for better QA. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ , pages 3424–3430, Online. Association for Computational Linguistics. 

- Urvashi Khandelwal, Angela Fan, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2021. Nearest neighbor machine translation. _International Conference on Learning Representations_ . 

- Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2019. Generalization through memorization: Nearest neighbor language models. In _ICLR_ . 

- Jan-Christoph Klie, Michael Bugert, Beto Boullosa, Richard Eckart de Castilho, and Iryna Gurevych. 2018. The INCEpTION platform: Machine-assisted and knowledge-oriented interactive annotation. In _COLING: System Demonstrations_ , pages 5–9, Santa Fe, New Mexico. ACL. 

- Jan-Christoph Klie, Richard Eckart de Castilho, and Iryna Gurevych. 2020. From Zero to Hero: HumanIn-The-Loop Entity Linking in Low Resource Domains. In _ACL_ , pages 6982–6993, Online. ACL. 

- Thomas Kober, Julie Weeds, Lorenzo Bertolini, and David J. Weir. 2021. Data augmentation for hypernymy detection. In _EACL_ , pages 1034–1048. 

- Trausti Kristjansson, Aron Culotta, Paul Viola, and Andrew McCallum. 2004. Interactive information extraction with constrained conditional random fields. In _AAAI_ , volume 4, pages 412–418. 

- Brenden M Lake, Ruslan Salakhutdinov, and Joshua B Tenenbaum. 2015. Human-level concept learning through probabilistic program induction. _Science_ , 350(6266):1332–1338. 

- Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. 2017. Building machines that learn and think like people. _Behavioral and brain sciences_ , 40. 

- Stefan Larson, Adrian Cheung, Anish Mahendran, Kevin Leach, and Jonathan K Kummerfeld. 2020. Inconsistencies in crowdsourced slot-filling annotations: A typology and identification methods. In _Proceedings of the 28th International Conference on Computational Linguistics_ , pages 5035–5046. 

- Florian Laws, Christian Scheible, and Hinrich Schütze. 2011. Active learning with Amazon Mechanical Turk. In _EMNLP_ , pages 1546–1556, Edinburgh, Scotland, UK. ACL. 

- Trung-Nghia Le, Tam V Nguyen, Quoc-Cuong Tran, Lam Nguyen, Trung-Hieu Hoang, Minh-Quan Le, and Minh-Triet Tran. 2021. Interactive video object mask annotation. In _AAAI_ , volume 35, pages 16067– 16070. 

- Yoonjoo Lee, John Joon Young Chung, Tae Soo Kim, Jean Y Song, and Juho Kim. 2022. Promptiverse: Scalable generation of scaffolding prompts through human-ai hybrid knowledge graph annotation. In _CHI Conference on Human Factors in Computing Systems_ , pages 1–18. 

- Yanzeng Li, Bowen Yu, Li Quangang, and Tingwen Liu. 2021. Fitannotator: A flexible and intelligent text annotation system. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Demonstrations_ , pages 35– 41. 

- Shudong Liu, Xuebo Liu, Derek F Wong, Zhaocong Li, Wenxiang Jiao, Lidia S Chao, and Min Zhang. 2023. knn-tl: k-nearest-neighbor transfer learning for lowresource neural machine translation. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1878–1891. 

- Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_ , pages 142–150, Portland, Oregon, USA. Association for Computational Linguistics. 

- Samuel Marcos-Pablos and Francisco J García-Peñalvo. 2020. Information retrieval methodology for aiding scientific database search. _Soft Computing_ , 24(8):5551–5560. 

- Melanie Mitchell. 2021. Abstraction and analogymaking in artificial intelligence. _Annals of the New York Academy of Sciences_ , 1505(1):79–101. 

- Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, 

Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. 2022. Text and code embeddings by contrastive pretraining. _arXiv preprint arXiv:2201.10005_ . 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems_ , 35:27730–27744. 

- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In _EMNLP_ , pages 1532–1543. 

- Tim Rietz and Alexander Maedche. 2021. Cody: An aibased system to semi-automate coding for qualitative research. In _CHI_ , pages 1–14. 

- Eric Ringger, Peter McClanahan, Robbie Haertel, George Busby, Marc Carmen, James Carroll, Kevin Seppi, and Deryle Lonsdale. 2007. Active learning for part-of-speech tagging: Accelerating corpus annotation. In _Proceedings of the Linguistic Annotation Workshop_ , pages 101–108. 

- Stephen Roller, Katrin Erk, and Gemma Boleda. 2014. Inclusive yet selective: Supervised distributional hypernymy detection. In _COLING: Technical Papers_ , pages 1025–1036. 

- Amirreza Shirani, Franck Dernoncourt, Paul Asente, Nedim Lipka, Seokhwan Kim, Jose Echevarria, and Thamar Solorio. 2019. Learning emphasis selection for written text in visual media from crowd-sourced label distributions. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 1167–1172. 

- Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In _Proceedings of the 2013 conference on empirical methods in natural language processing_ , pages 1631–1642. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ . 

- Ding Wang, Shantanu Prabhat, and Nithya Sambasivan. 2022a. Whose ai dream? in search of the aspiration in data annotation. In _CHI_ , pages 1–16. 

- Dongqi Wang, Haoran Wei, Zhirui Zhang, Shujian Huang, Jun Xie, Weihua Luo, and Jiajun Chen. 2021. Non-parametric online learning from human feedback for neural machine translation. _arXiv_ . 

- Shuhe Wang, Xiaoya Li, Yuxian Meng, Tianwei Zhang, Rongbin Ouyang, Jiwei Li, and Guoyin Wang. 2022b. _k_ nn-ner: Named entity recognition with nearest neighbor search. _arXiv preprint arXiv:2203.17103_ . 

- Yan Wang, Wei-Lun Chao, Kilian Q Weinberger, and Laurens van der Maaten. 2019. Simpleshot: Revisiting nearest-neighbor classification for few-shot learning. _arXiv_ . 

- Xingjiao Wu, Luwei Xiao, Yixuan Sun, Junhang Zhang, Tianlong Ma, and Liang He. 2022. A survey of human-in-the-loop for machine learning. _Future Generation Computer Systems_ , 135:364–381. 

- Ruixuan Xiao, Yiwen Dong, Junbo Zhao, Runze Wu, Minmin Lin, Gang Chen, and Haobo Wang. 2023. FreeAL: Towards human-free active learning in the era of large language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 14520–14535, Singapore. Association for Computational Linguistics. 

- Kaichun Yao, Chuan Qin, Hengshu Zhu, Chao Ma, Jingshuai Zhang, Yi Du, and Hui Xiong. 2021. An interactive neural network approach to keyphrase extraction in talent recruitment. In _Proceedings of the 30th ACM International Conference on Information & Knowledge Management_ , pages 2383–2393. 

- Xin Zheng, Zhirui Zhang, Junliang Guo, Shujian Huang, Boxing Chen, Weihua Luo, and Jiajun Chen. 2021. Adaptive nearest neighbor machine translation. pages 368–374. 

## **A Hyperparameter Analysis** 

In this section, we provide a detailed hyperparameter analysis of ARAIDA, including the datastore size _At_ , the number of neighbors _k_ for _ρt_ , and the datastore maintenance strategy. We show results only for the Dist./FT annotation model for brevity. 

## **A.1 Datastore Size** 

We tune the size of the datastore _At_ in _{_ 100 _,_ 500 _,_ 1000 _,_ 2000 _}_ and evaluate the machine cumulative accuracy of ARAIDA with or without active learning. We illustrate the results in Fig.6. A larger datastore size generally brings higher annotation performance since it allows us to maintain more data from past human-machine interactions. However, it also requires larger memory usage and causes longer latency. We found that a datastore size of 1000 is a reasonable tradeoff, which we utilize in our main experiments. 

## **A.2 Top** _k_ **for** _ρt_ 

We tune the number of neighbors _At_ in _{_ 5 _,_ 10 _,_ 20 _,_ 50 _}_ and evaluate the machine cumulative accuracy of ARAIDA with or without active learning. As shown in Fig.7, _k_ = 20 seems to perform well for all tasks. 

## **A.3 Datastore Maintenance Strategy** 

We propose a class-aware datastore maintenance strategy for ARAIDA, which removes labeled examples from the majority class most similar to their class prototype. Compared to the conventional First-In-First-Out (FIFO) strategy (Dekel et al., 2005), our method ensures that 1) the datastore contains as much data from different classes as possible, and 2) the class prototype is least affected after the removal. We compare with two variants of ARAIDA, including ARAIDA _w/ FIFO_ and ARAIDA _w/ class FIFO_ . The former discards the oldest example regardless of the class; the latter discards the oldest example belonging to the majority class in the datastore. 

As shown in Fig.8, ARAIDA _w/ FIFO_ can suffer from a sudden decrease in performance, as it may remove important examples arbitrarily. After integrating the class information, ARAIDA _w/ class FIFO_ removes the oldest analogy from the majority class, achieving a comparative performance to ARAIDA. However, ARAIDA still performs better when the number of classes increases (e.g., FreeBase). 

## **B Results on Comparing to Fully Fine-tuned Model** 

We utilized up to 5K of data to simulate the limited data annotation task, and we updated the parameters of the annotation model as the interactive annotation process progressed. To validate the effectiveness of the fully fine-tuned model, we use the remaining data from the original dataset, excluding the 5K data and the validation data, as the training data to fully fine-tune the model. Next, we employ this model for interactive annotation. Since it is already fully fine-tuned, we do not update the model parameters during the annotation process. Due to time constraints, we are currently only considering scenarios where active learning has not been adopted. Additionally, we only fine-tune small annotation models (i.e., BERT and Dist./FT). 

Based on the results in the Table 5, it is evident that the annotation performance of the fully finetuned model surpasses that of our method. This suggests that while our method currently offers a lightweight approach to aid data annotation, which can to some extent enhance the sample efficiency of the annotation model, there is still potential for improvement in future research.Nevertheless, it’s important to emphasize that in real-world data an- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation_images/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation.pdf-0013-00.png)


**----- Start of picture text -----**<br>
60% 50%<br>60% 50%<br>50% 40%<br>50% 40%<br>40%<br>40% 30%<br>30%<br>30%<br>30%<br>20% AARAIDARAIDA-100 -500  20% AARAIDARAIDA-100 -500  20% 20%<br>10% AARAIDARAIDA-1000 -2000 10% AARAIDARAIDA-1000 -2000 10% 10%<br>0% 0% 0% 0 %<br>1 1K    2K   3K    4K  5K4 7 10131619222528313437 1 41K    2K   3K    4K  5K7 10131619222528313437 1 41K    2K   3K    4K  5K7 10 13 16 19 22 25 28 31 34 37 1 41K    2K   3K    4K  5K7 10 13 16 19 22 25 28 31 34 37<br>(a) WN18RR (b) WN18RR + AL (c) FreeBase (d) FreeBase + AL<br>90% 80% 41% 41%<br>75%<br>80% 36% 36%<br>70%<br>70% 65% 31% 31%<br>60%<br>60% 26% 26%<br>55%<br>50% 50% 21 % 21 %<br>1 1K    2K   3K    4K  5K4 7 10 13 16 19 22 25 28 31 34 37 1 1K    2K   3K    4K  5K5 9 13172125293337 1 41K    2K   3K    4K  5K7 10 13 16 19 22 25 28 31 34 37 1 41K    2K   3K    4K  5K7 10 13 16 19 22 25 28 31 34 37<br>(e) IMDB (f) IMDB + AL (g) SST-5 (h) SST-5 + AL<br>**----- End of picture text -----**<br>


Figure 6: Machine Cumulative Accuracy of ARAIDA with different datastore sizes. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation_images/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation.pdf-0013-02.png)


**----- Start of picture text -----**<br>
WN18RR FreeBase<br>Raw Class Mapped Class Raw Class Ma pped Class<br>_hypernym hypernym /location/location/contains contains<br>_deriva tionally_related_form der ivation /olympics/olympic_sport/athle tes./olympics/olympic_athlete _affili ation/country country<br>_member_meronym member /music/performance_role/track_performances./music/track_contribution/role track_role<br>_has_part component /people/person/profession profession<br>_synset_domain_topic_of synset /music/performance_role/regular_performances./music/group_membership/role group_role<br>_instance_hypernym instance of hypernym /location/location/adjoin_s./location/adjoining_relationship/adjoins adjoins<br>_also_see synonym /film/film/release_date_s./film/film_regional_ rele ase_date/film_release_region film_release<br>_verb_group verb group /food/food/nutrients./food/nutrition_fact/nutr ient nutrient<br>Table 4: Class mapping details for WN18RR and FreeBase<br>70% 70% 60% 50%<br>60% 60% 50% 40%<br>50% 50% 40%<br>40% 40% 30%<br>30%<br>30% ARAIDA-5  30% ARAIDA-5  20%<br>20% ARAIDA-10  20% ARAIDA-10  20%<br>10% ARAIDA-20  10% ARAIDA-20  10% 10%<br>ARAIDA-50 ARAIDA-50<br>0% 0 % 0% 0%<br>1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337<br>(a) WN18RR (b) WN18RR + AL (c) FreeBase ( d) FreeBase + AL<br>85% 80% 40%<br>40%<br>80%<br>75% 35%<br>75% 35%<br>70%<br>70% 30% 30%<br>65%<br>65% 25% 25%<br>60% 60%<br>55% 55% 20% 20%<br>50% 50% 15% 15%<br>1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337<br>(e) IMDB (f) IMDB + AL (g) SST-5 (h) SST-5 + AL<br>**----- End of picture text -----**<br>


Figure 7: Machine Cumulative Accuracy of ARAIDA with different _k_ for _ρt_ . 

notation scenarios, we typically don’t have access to fully fine-tuned models initially, as we lack labeled datasets. While one might tu ~~rn to LLMs like~~ ChatGPT, our findings in Figure 3 indicate that our 

method could potentially improve upon ChatGPT even further. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation_images/acl_2024_ARAIDA_Analogical_Reasoning-Augmented_Interactive_Data_Annotation.pdf-0014-00.png)


**----- Start of picture text -----**<br>
70% 60% 60% 50%<br>60% 50% 50% 40%<br>50% 40% 40%<br>40% 30%<br>30% 30%<br>30% 20%<br>20% ARAIDA 20% ARAIDA 20%<br>10% ARAIDA w/ FIFO  10% ARAIDA w/ FIFO  10% 10%<br>ARAIDA w/class FIFO ARAIDA w/class FIFO<br>0% 0% 0% 0%<br>1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337<br>(a) WN18RR (b) WN18RR + AL (c) FreeBase (d) FreeBase + AL<br>85% 80% 45% 45%<br>80%<br>75% 40% 40%<br>75%<br>70% 70% 35% 35%<br>65%<br>65% 30% 30%<br>60% 60%<br>55% 55% 25% 25%<br>50% 50% 20% 20%<br>1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337 1 1K    2K   3K    4K  5K5 9 13172125293337<br>(e) IMDB (f) IMDB + AL (g) SST-5 (h) SST-5 + AL<br>**----- End of picture text -----**<br>


Figure 8: Machine Cumulative Accuracy of ARAIDA with different datastore maintenance strategies. 

Table 5: Results on comparing to fully fine-tuned models 

|**Model**|**WN18RR**|**FreeBase**|**IMDB**|**SST-5**|
|---|---|---|---|---|
|BERT(fully fnetuned)|75.13_±_1.12|60.47_±_1.24|90.04_±_0.42|46.59_±_1.20|
|BERT+ARAIDA|54.41_±_1.51|50.30_±_2.25|88.79_±_3.54|43.81_±_3.12|
|Dist./FT(fully fnetuned)|72.64_±_1.05|58.50_±_1.36|84.87_±_0.81|42.12_±_1.44|
|Dist./FT+ARAIDA|52.16_±_1.37|43.02_±_6.43|79.33_±_2.81|37.21_±_3.03|



## **D Prompts and Fine-Tuning Examples for LLM-Based Annotation** ~~**Mo**~~ **dels** 

Table 7 presents prompts used for zero-shot and few-shot annotation. Table 8 shows fine-tuning examples for LLaMa2 _sft_ . 

## **C Results on Smaller-sized Transformer-based Model** 

With a wide array of language models to choose from, we faced the challenge of not being able to test all available models. To address this, we selected three prominent models (GloVe, LLaMa2, and ChatGPT) based on our hardware resource capabilities in our main experiments. While open to conducting further experiments, our focus was limited by time constraints, leading us to concentrate solely on fine-tuning BERT (w/o AL), with the outcomes detailed in Table 6. After implementing ARAIDA, we observed a significant enhancement in the quality of annotations by using our ARAIDA. 

Table 6: Results on smaller-sized Transformer-based model 

|**Model**|**WN18RR**<br>**FreeBase**|**IMDB**|**SST-5**|
|---|---|---|---|
|BERT|53.32_±_1.02<br>46.12_±_3.2|3<br>85.38_±_7.16|41.08_±_3.41|
|BERT+ARAID<br>BERT+ARAID|A<br>**54.41**_±_**1.51**<br>**50.30**_±_**2.2**<br>Aw/o f<br>51.03_±_2.11<br>44.63_±_4.0|**5**<br>**88.79**_±_**3.54**<br>4<br>80.29_±_1.09|**43.81**_±_**3.12**<br>38.51_±_1.45|



|**Dataset**|**Prompts**|
|---|---|
|IMDB|==================<br>input: If you like original gut wrenching laughter...it. Great Camp!!!.<br>output: positive<br>input: I saw this movie when I was about 12 when it ... There are no rules.<br>output: negative<br>==================<br>You need to identify the sentiments of the following sentences, output positive or negative.<br>{INPUTS}|
|SST-5|=====================<br>input: The gorgeously elaborate continuation ...Tolkien ’s Middle-earth.<br>output: Strong Positive<br>input: Singer/composer Bryan Adams contributes a slew of..., spirit of the piece.<br>output: Positive<br>input: You’d think by now ...with hearts of gold.<br>output: Neutral<br>input: This isn’t a new idea .<br>output: Negative<br>input: A sour little movie at ... What was it all for ?<br>output: Strong Negative<br>====================<br>Identify the sentiment of each paragraph,<br>you have fve options: ’Strong Positive’, ’Positive’, ’Neutral’, ’Negative’ or ’Strong Negative’<br>{INPUTS}|
|WN18RR|=============<br>input: ability unftness<br>output: antonym<br>input: dissent debating<br>output: entailment<br>...<br>input: abandonment apostasy<br>output: hypernym<br>input: abandonment discard<br>output: hyponym<br>input: Afghanistan Afghan<br>output: member<br>input: abandonment abandonment<br>output: synonym<br>=============<br>Identify the semantic relation of the each word pair,<br>you have eight options: ’component’, ’synset’,..., ’hypernym’, ’derivation’, ’member’, ’synonym’<br>You MUST only output the semantic relation word for each input!<br>{INPUTS}|
|FreeBase|==========<br>input: Libya Egypt<br>output: adjoins<br>input: Honolulu Punahou<br>output: contains<br>input: Bobsleigh Netherlands<br>output: country<br>input: Blackbriar Lithuania<br>output: flm_release<br>input: Autoharp Guitar<br>output: group_role<br>input: IceCream Water<br>output: nutrient<br>input: Shriya Actor<br>output: profession<br>input: Cello Pennywhistle<br>output: track_role<br>=========<br>Identify the semantic relation of the each word pair,<br>you have eight options: ’contains’, ’country’, ’track_role’, ’profession’, ’group_role’, ’adjoins’, ’flm_release’, ’nutrient’.<br>only output the semantic relation.<br>{INPUTS}|



Table 7: Prompts for different datasets to obtain the annotation. We remove the few-shot demonstrations in the prompts in the zero-shot scenarios. 

|**Dataset**|**Finetuning data example for LLaMa2**_sft_|
|---|---|
|IMDB|{"instruction": "Identify the sentiment of the following paragraph, output ’positive’ or ’negative’.",<br>"input": "The cast played Shakespeare.Shakespeare lost.I appreciate that this is trying to bring Shakespeare to the masses,<br>but why ruin something so good. Is it because ’The Scottish Play’ is my favorite Shakespeare? I do not know.<br>What I do know is that a certain Rev Bowdler (hence bowdlerization) tried to do something similar in the<br>Victorian era.In other words, you cannot improve perfection.I have no more to write but as I have to<br>write at least ten lines of text (and English composition was never my forte I will just have to keep<br>going and say that this movie, as the saying goes, just does not cut it.",<br>"output": "negative"},|
|SST-5|{"instruction": "Identify the sentiment of the following paragraph,<br>you have fve options: ’Strong Positive’, ’Positive’, ’Neutral’, ’Negative’ or ’Strong Negative’.",<br>"input": "The gorgeously elaborate continuation of The Lord of the Rings trilogy is so huge that a<br>column of words can not adequately describe co-writer/director Peter Jackson ’s expanded vision of J.R.R.<br>Tolkien ’s Middle-earth .",<br>"output": "StrongPositive"},|
|WN18RR|{"instruction": "Identify the semantic relation of the following word pair,<br>you have eight options: ’antonym’,..., ’synonym’.",<br>"input": "a.m. A.M.",<br>"output": "synonym"},|
|FreeBase|{"instruction": "Identify the semantic relation of the following word pair,<br>you have eight options: ’contains’, ..., ’nutrient’.",<br>"input": "Autoharp Guitar",<br>"output": "group_role"},|



Table 8: Finetuning data examples for LLaMa2 _sft_ 

