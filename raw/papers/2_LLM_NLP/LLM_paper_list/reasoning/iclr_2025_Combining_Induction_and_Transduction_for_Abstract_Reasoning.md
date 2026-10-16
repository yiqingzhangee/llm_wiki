Preprint. 

## COMBINING INDUCTION AND TRANSDUCTION FOR ABSTRACT REASONING 

**Wen-Ding Li[*]**[1] **Keya Hu[*]**[2] **Carter Larsen**[1] **Yuqing Wu**[1] **Simon Alford**[1] **Caleb Woo**[1] **Spencer M. Dunn**[1] **Hao Tang**[1] **Michelangelo Naim**[3] **Dat Nguyen**[3] **Wei-Long Zheng**[2] **Zenna Tavares[†]**[3] **Yewen Pu[†]**[4] **Kevin Ellis[†]**[1] 

1Cornell 2Shanghai Jiao Tong University 3Basis 4Autodesk *co-leads †co-advising correspondence: `{wl678,kellis}@cornell.edu, hu_keya@sjtu.edu.cn` 

## ABSTRACT 

When learning an input-output mapping from very few examples, is it better to first infer a latent function that explains the examples, or is it better to directly predict new test outputs, e.g. using a neural network? We study this question on ARC by training neural models for _induction_ (inferring latent functions) and _transduction_ (directly predicting the test output for a given test input). We train on synthetically generated variations of Python programs that solve ARC training tasks. We find inductive and transductive models solve different kinds of test problems, despite having the same training problems and sharing the same neural architecture: Inductive program synthesis excels at precise computations, and at composing multiple concepts, while transduction succeeds on fuzzier perceptual concepts. Ensembling them approaches human-level performance on ARC. 

## 1 INTRODUCTION 

Robust generalization from few examples remains one of the most important ways in which human intelligence surpasses AI. Much recent work views this generalization as a form of abstract reasoning: Given just a few training input-outputs _**x**_ train _,_ _**y**_ train, together with a test input _x_ test, the idea is to predict the corresponding test output _y_ test using reasoning strategies such as analogical reasoning, chain-of-thought, inductive program synthesis, or transductive prediction (Thoms et al., 2023; Wang et al., 2024; Witt et al., 2023; Lee et al., 2024; Tang et al., 2024a; Hocquette & Cropper, 2024; Butt et al., 2024). The Abstraction and Reasoning Corpus (Chollet (2019), henceforth “ARC”) is a fewshot learning benchmark that tests the ability to rapidly learn a diverse range of new skills, and apply them to new situations. Each ARC task is presented as input-outputs over colored grids, but can engage concepts such as occlusion, pathfinding, collision, symmetry, gravity, bouncing, counting, etc., making ARC essentially a composite of many reasoning datasets, and one of the more interesting unsolved benchmarks that stresses broad-coverage few-shot learning (Figure 1). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0001-09.png)


**----- Start of picture text -----**<br>
object contact gravity symmetry, occlusion jumping<br>bouncing mazes pathfinding growth<br>**----- End of picture text -----**<br>


Figure 1: Few-shot learning tasks from the Abstraction and Reasoning Corpus (ARC). Each task typically has 2-5 input-output examples. Here we show just one input-output example per task. 

Here we study neural methods for induction and transduction, using few-shot learning problems from ARC as our testbed. _Induction_ means first finding a function _f_ where _f_ ( _**x**_ train) _≈_ _**y**_ train, and then predicting _y_ test = _f_ ( _x_ test). _Transduction_ instead outputs _y_ test without explicit construction of an 

1 

Preprint. 

intermediate function _f_ . Intuitively, induction captures the notion that a learner should first explain the training data, then use that explanation to make predictions. Inductive learners can perform better by spending more time optimizing or searching for better explanations, using the training examples _**x**_ train _,_ _**y**_ train to score candidate functions. Transduction instead captures the intuition that the training examples themselves should play a direct role in generating new predictions, and that successful prediction need not require an explicit explanation. See Figure 2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0002-02.png)


**----- Start of picture text -----**<br>
induction transduction<br>xtrain xtrai ytrain ytrai<br>xtrain  ytrain<br>f(•) n n<br>f(xtest ) ytest ytest<br>xtest xtestxtest<br>**----- End of picture text -----**<br>


Figure 2: Induction generates an intermediate function _f_ to explain training input-outputs. Transduction directly predicts the test output, for example using a neural network. 

We train neural networks for both induction and transduction by generating a large corpus of synthetic problems. We discover that neural models for induction and transduction are strongly complementary. We believe this is surprising: Although any pair of models would generally solve somewhat different problems, usually this can be attributed to different priors, data, or architecture. Instead, we find that, _even controlling for priors, data, and architecture, most problems solved by induction were not solved by transduction, and vice versa._ Moreover, induction and transduction can be trivially ensembled by using induction to generate candidate functions _f_ until either a satisfactory function is found (e.g. _f_ ( _**x**_ train) = _**y**_ train) or until a test-time compute budget is reached, at which point, transduction kicks in as a fallback: That they are complementary has practical implications. 

Our study is tightly linked to program synthesis. We represent functions _f_ as Python code, meaning that induction synthesizes programs. We train transduction models on LLM-produced Python scripts, meaning that transduction is trained on the input-outputs of symbolic code. Although program learning has long been a popular vision of how general AI could work (Solomonoff, 1964; Schmidhuber, 2004; Hutter, 2004), the dominant theory has always been one of explicit code generation (induction), rather than implicitly teaching neural networks to imitate code (transduction). Our work puts this assumption to the test. 

Testing these neural methods requires a large dataset of function-learning problems, which is challenging to generate because not only must we make novel functions, but we must also make good inputs to those functions. Consider the range of transformations in Figure 1: What counts as a good input for one function is unlikely to work for another. To address this challenge, our data generator first produces a deterministic Python function for _f_ , and then a probabilistic program for sampling inputs to _f_ , finally executing those programs to produce input-outputs. This helps generate function inputs that are appropriate for the underlying transformation, and constrains _**x**_ train _,_ _**y**_ test to be explainable by a deterministic mapping. 

We contribute the following: 

1. A study finding that neural models for induction and transduction are strongly complementary, even when trained on the same problems. This contradicts seminal neural program synthesis work (Devlin et al. (2017), which found induction superior), and contradicts the findings of the leading ARC team (Cole et al. (2024), which advocates transduction with test-time training). 

2. An automated data generation methodology that starts with 100-160 program solutions for ARC training tasks, and expands them to make 400k new problems paired with Python solutions. 

3. A study of how these methods scale. We find performance saturates quickly when increasing manually-labelled data, but scales with compute, both at training and testing time. 

4. Analysis of families of problems solved by each approach, and how they compare to humans. 

2 

Preprint. 

## 2 NEURAL MODELS FOR INDUCTION AND TRANSDUCTION 

We consider few-shot supervised learning problems where the learner is trained to map members of an input space _X_ to output space _Y_ . For _K_ -shot learning, we receive _K_ training input-outputs ( _**x**_ train _,_ _**y**_ train) _∈X[K] × Y[K]_ , together with a single test input _x_ test _∈X_ , and predict _y_ test _∈Y_ . Our neural models for _K_ -shot learning are meta-learned (Mishra et al., 2017, _inter alia._ ) using metalearning data further annotated with a ground-truth function _f_ : _X →Y_ , which supervises the induction model. Below we define the training and use of these models. 

**Definition: Neural networks for induction and transduction.** A neural network for transduction is a function `t` that maps ( _**x**_ train _,_ _**y**_ train _, x_ test) to a distribution over _y_ test, and which has learnable parameters _θ_ . In other words, `t` _θ_ : _X[K] ×Y[K] ×X →_ ∆( _Y_ ), where the notation ∆( _S_ ) means the set of distributions over _S_ . We can also write this as a conditional distribution, `t` _θ_ ( _y_ test _|_ _**x**_ train _,_ _**y**_ train _, x_ test). A neural network for induction is a function `i` that maps ( _**x**_ train _,_ _**y**_ train _, x_ test) to a distribution over functions _f_ that map _X_ to _Y_ , with learnable parameters _θ_ . In other words, `i` _θ_ : _X[K] × Y[K] × X →_ ∆( _X →Y_ ), which we can write as a conditional distribution `i` _θ_ ( _f |_ _**x**_ train _,_ _**y**_ train _, x_ test). 

**Training induction and transduction.** Both types of models are trained via meta-learning. We assume a meta-learning dataset _D_ of few-shot learning problems, each equipped with a groundtruth function _f_ such that _f_ ( _x_ ) = _y_ for every _x, y_ in ( _**x**_ train _,_ _**y**_ train) and ( _x_ test _, y_ test). Inductive and transductive models are meta-trained to minimize the following losses: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0003-05.png)


**Testing induction and transduction.** After meta-learning the models encounter a test-time fewshot learning task ( _**x**_ train _,_ _**y**_ train _, x_ test). Transductive models predict their most likely output for _y_ test (approximated via beam search). Inductive models sample a test-time budget of _B_ functions _f_ 1 _· · · fB_ , which are filtered by ( _**x**_ train _,_ _**y**_ train), and finally used to predict _y_ test = _f_ ( _x_ test). Writing ˆ _y_ test for the predicted test output: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0003-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0003-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0003-09.png)


**Combining induction and transduction.** Induction allows checking candidate hypotheses against the training examples. Therefore, we know when induction has found a plausible solution–but sometimes it fails to find any solution. Transduction has the opposite property: We can’t check if its predictions match the training examples, but it always offers a candidate answer. Therefore we ensemble by attempting induction first, then transduction if none of the candidate hypotheses explained the examples: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0003-11.png)


**Instantiating the framework for ARC.** Every input from _X_ and output from _Y_ is a 2D grid ranging from 1–30 pixels per side, with each pixel containing one of ten colors. Because ARC tasks are highly diverse yet typically have an abstract program-like structure, we represent the underlying function _f_ as Python code, which is computationally universal, and so possible in principle of solving any ARC task. Therefore the induction model must generate Python code, so we initialize our models with Llama3.1-8B-instruct (Dubey et al., 2024) because it was pretrained on source code.[1] We encode 2D colored grids as strings using 1 token per pixel, and use newlines to delimit rows (Appendix B.1). We then meta-learn by further fine-tuning Llama3.1-8B-instruct for induction or transduction using a synthetically-generated corpus of problems, described next. 

> 1Our preliminary experiments suggested Llama3.1-8B-instruct was better than Mistral-7B-v0.3, Qwen27B-Instruct, and deepseek-coder-6.7b-instruct 

3 

Preprint. 

## 3 GENERATING DATASETS FOR INDUCTION AND TRANSDUCTION 

Generating ARC-style tasks is challenging because of the diversity of concepts that can occur in ARC. It is also challenging because we need to generate not just a function, and also inputs that serve as good examples for that function. 

At a high level, our dataset grows out of 100 manually-written Python programs, each of which both solves a given task (function _f_ ), and also randomly generates new input grids. We call these 100 manually-written programs **seeds.** Each seed is commented with natural language describing the core intuitions behind the problem. We then use a large language model to mutate and recombine the seeds, producing many thousands of programs (Figure 3). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0004-04.png)


**----- Start of picture text -----**<br>
100 seed solutions 100k seed problems with solutions<br>100 seed problems language language<br>concepts : cropping concepts :<br>examples description: description:<br>In the input you will see a single colored  code<br>shape, around 4x6 in size, floating in a<br>12x12 grid of black.<br>solve To make the output, crop the background out of the image - so the output grid has  remix runtime check<br>the same dimensions as the shape.<br>examples<br>code<br>manual problem solving synthetic augmentation<br>**----- End of picture text -----**<br>


Figure 3: Synthetic data generation pipeline, starting with human-written programs (seeds). 

**The structure of seeds.** Each seed consists of three parts: 

1. A **natural language description** of its specific ARC task—including how to solve that task—represented as a Python comment at the top of the seed. 

2. A Python function `transform_grid` corresponding to the function _f_ in the manuscript, which maps each input grid of a specific ARC task to its corresponding output grid. 

3. A Python function `generate_input` , which takes no arguments, and which randomly generates new inputs to _f_ (new inputs to `transform_grid` ). 

**Prior knowledge.** The seeds impart a prior upon the system by demonstrating good programs for solving training tasks. We further codified much of this prior into a Python library containing code that we found useful across many seeds, such as subroutines for generating random sprites, detecting symmetries, or extracting objects (Appendix A.2). Synthetic problems can use that same library. 

However, this prior knowledge is different from previous _Domain Specific Languages_ for ARC (Butt et al., 2024; Wind, 2020; Ainooson et al., 2023). Domain Specific Languages restrict the class of allowed programs by only allowing stereotyped combinations of domain-specific primitives. We still allow arbitrary Python code, which helps cover the long tail of diverse tasks. 

**Remixing the seeds.** To generate a larger synthetic dataset we “remix” the seeds using LLMs. Each new synthetic ARC problem is generated by a three stage pipeline (Figure 11): 

1. A new natural language description is sampled by prompting an LLM with seed natural language descriptions, effectively using in-context learning to recombine and mutate elements of different problems, in the spirit of self-instruct (Wang et al., 2023). 

2. Code is generated for that new description via Retrieval Augmented Generation (RAG: Lewis et al. (2020)). Our RAG pipeline retrieves seeds with similar descriptions, and prompts an LLM to generate code for the new description, given the retrieved seeds. 

3. The newly created `generate_input` is executed to make inputs, which are passed to `transform_grid` to produce input-output grids. 

4 

Preprint. 

Figure 4 illustrates example problems generated by our pipeline, with further examples **visualized at this link.** Unless otherwise mentioned, we create synthetic datasets with GPT4o-mini and ada-002. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-02.png)


**----- Start of picture text -----**<br>
concepts:<br>shape detection, contour<br>extraction, bounding box<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-07.png)


**----- Start of picture text -----**<br>
concepts:  concepts:<br>color blending,  pattern detection, copying,<br>radial expansion positioning<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0005-15.png)


Figure 4: Example synthetic ARC problems generated by our pipeline. Concepts are generated in a comment near the top of the Python script as part of the natural language description of the seed. 

## 4 EMPIRICAL STUDY OF INDUCTION AND TRANSDUCTION 

We train inductive and transductive models with the goal of understanding (1) how the methods compare; (2) how performance scales with train-time effort; and (3) how performance scales with test-time compute (for induction only, as it allows drawing more samples at test time to improve performance). We report performance on the 400-problem public validation split of ARC, which is harder than the training split. The systems described in this section learn from a 100-problem subset of the training split, specifically problems for which we created seeds. 

**Induction and Transduction are strongly complementary.** Despite training on the exact same problems, inductive and transductive models solve different tasks, and neither approach is dramatically more effective than the other. And although these methods have a similar overall solve rate, _most problems solved by induction are not solved by transduction, and vice versa_ (Figure 5A). 

An alternative explanation is that induction and transduction are not actually complementary, but instead that, having trained two neural networks with different random initializations, they simply solved different problems due to randomness at train or test time. To test this alternative explanation, we trained many models with different random initializations. We find that the problems solved by induction/transduction are surprisingly stable across these different runs (Figure 5B). In other words, some problems are friendlier to induction, and others friendlier to transduction (Figure 6). 

**(A) complementary methods (B) finding is stable across random seeds, and (C) statistically significant** transduction induction 1.0 **p<.004** 0.0 = 1 problem solved 

Figure 5: (A) Induction and transduction solve different problems, where solve means predicting the right output given 2 tries. Venn diagram for models trained on 100k synthetic problems generated using gpt4o-mini. (B) Training many models with different random seeds, and then measuring the correlation between solved tasks by different models. Solved tasks strongly correlates with other models of the same class but not the other class. (C) Statistical significance test evaluating the null hypothesis that correlation is independent of whether a model is inductive/transductive. 

5 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0006-01.png)


**----- Start of picture text -----**<br>
solved only by induction<br>solved only by transduction<br>solved by both<br>**----- End of picture text -----**<br>


Figure 6: Example tasks solved by induction/transduction/both. See also Appendix C. 

**Performance scales with dataset size, but quickly saturates with increasing number of seeds.** We trained models while systematically varying the number of human-created seeds we use, and varying the amount of synthetic data generated from those seeds (Figure 7). Performance improves with increasing training data for fine-tuning (increasing synthetic data), but saturates for increasing quantity of human-created seeds. We conjecture that this saturation occurs because each seed serves to introduce a few core concepts, and that after enough seeds, essentially all of the important concepts have been demonstrated. This suggests that, beyond a critical threshold number of seeds, the method can scale with increasing amounts of compute without demanding further human data labeling. Looking beyond ARC, this means that our methodology could probably be applied to other few-shot function-learning problems using a modest amount of manual data labeling. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0006-04.png)


**----- Start of picture text -----**<br>
increasing synthetic data, 100 seeds increasing seeds, 40k synthetic problems<br>0.150 Induction Induction<br>Transduction Transduction<br>0.125<br>0.100<br>0.075<br>0.050<br>20 40 60 80 100 25 50 100<br>num synthetic problems (10^3) num seeds<br>Validation Accuracy<br>**----- End of picture text -----**<br>


Figure 7: Increased manual human effort (# seeds) does not significantly increase performance, but increasing compute spent generating synthetic data increases performance of the final model. 

6 

Preprint. 

**Induction performance scales with test-time compute.** We vary the test-time sampling budget for induction, finding an almost monotonic increase in solve rate (Figure 8, left). In principle, drawing additional samples runs the risk of discovering solutions that are “false-positives,” meaning they fit the training examples without correctly predicting the test output. In practice, about 9% of samples that fit the training examples are false-positives. Figure 8 (right) shows that about half of this 9% corresponds to problems where the majority of the probability mass is still placed on the correct output, meaning that a simple majority vote scheme would squash any false positives (e.g. clustering in AlphaCode Li et al. (2022)). Appendix D shows example false positives. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0007-02.png)


**----- Start of picture text -----**<br>
ARC Val Accuracy<br>20%<br>15%<br>60<br>10%<br>40<br>Sample + Oracle<br>5%<br>Sample + Filter 20<br>CodeIt<br>0%<br>0<br>0 500 1000 1500 2000 0.0 0.2 0.4 0.6 0.8 1.0<br>number of samples False Positive Rate<br>Solve Rate<br># validation problems<br>**----- End of picture text -----**<br>


Figure 8: Left: Sample+Oracle assumes an oracle that selects one of the sampled programs. It upper-bounds the accuracy of randomly selecting one program consistent with the training examples (Sample+Filter). Induction model trained with 100k gpt4-description-gpt4omini-codegen data. CodeIt (Butt et al., 2024) is a recent neural program induction model for ARC. Right: Histogram of false positive rate. Of the problems that have false positives, about half have a false positive rate less than 0.5, meaning that most (filtered) samples predict the right test output. 

**Stronger LLMs make better synthetic data, and induction is more sensitive to data quality.** To save costs, the previous results all used GPT4o-mini to generate synthetic data. To understand the value of stronger models we generated 100k synthetic problems using GPT4 to generate natural language problem descriptions (but GPT4o-mini still generated the code). The richer and more diverse synthetic problems elicited from GPT4 significantly improved performance, but primarily for induction, while transduction was less sensitive to data quality (Table 1). 

Table 1: Val Acc: % validation tasks that are correctly solved in 2 tries. 

|System|Val Acc.|Finetuning Data|
|---|---|---|
|Ensemble|26.50%||
|Induction, 2048 samples|18.78%|GPT-4 for generating descriptions, GPT-4o-mini for code|
|Transduction, beam size 20|15.25%||
|Ensemble|19.50%||
|Induction, 2048 samples|11.07%|GPT-4o-mini for generating descriptions and code|
|Transduction, beam size 20|13.50%||



## 5 SCALING OUR METHOD 

Motivated by our findings so far, we scaled up our method by producing two larger datasets: 

**ARC-Heavy: 200k problems from 160 seeds.** The purpose of ARC-Heavy is to scale our method in an easily reproducible way, while also filling any gaps in its mastery of the training split. We first ran models from Section 4 on the training split to identify 60 problems that they still struggled with, for which we produced 60 new seeds, giving 160 seeds in total. From those seeds we produced 200k synthetic problems, with GPT4 generating natural language descriptions and GPT4o-mini generating the corresponding Python code. 

7 

Preprint. 

**ARC-Potpourri: 400k problems from heterogeneous sources.** The purpose of ARC-Potpourri is to assemble the biggest dataset that we could, even if it comes from a messy mixture of sources. Starting with ARC-Heavy we added all synthetic data from Section 4. We further added 100k transduction-only training examples from ReARC (Hodel, 2024). 

**Test-time improvements.** We improve transduction with test-time training (abbreviated TTT; Sun et al. (2020)) and a reranking scheme that augments each problem and predicts the most likely output under multiple augmentations (Appendix E- F). We expand our sampling budget to 20k programs. 

We call our resulting systems BARC. Table 2 shows the performance of various BARC models. Both transduction and induction are effective, with induction solving slightly more problems, until adding test-time training/reranking , after which transduction does slightly better. An ensemble scores 56.75%, surpassing previously published methods. Releasing this work later allowed Akyürek et al. (2024) to improve that score to 61.9% via better test time training of our transduction model, while using our induction model as-is. But absent our models—including the program synthesizer—they instead score lower, suggesting that sophisticated test time training does not fully substitute for program synthesis. Our best model performs nearly as well as the average human (56.75% vs. 60.2%) but much worse than the best humans (98%). Model outputs **visualized here.** 

Table 2: % validation tasks correctly solved in 2 tries. Human results from LeGris et al. (2024). 

|System|Val Acc.|Fair comparison?|
|---|---|---|
|**ARC-Heavy: BARC models**|||
|Induction, 10k samples, majority vote|30.50%|—|
|Transduction (no TTT)|19.25%|—|
|Ensemble (no TTT)|37.50%|—|
|Transduction (TTT)|29.75%|—|
|Ensemble (TTT)|43.25%|—|
|**ARC-Potpourri: BARC models**|||
|Induction, 20k samples, majority vote|38.00%|—|
|Transduction (no TTT, no reranking)|29.125%|—|
|Transduction (reranking, no TTT)|35.25%|—|
|Transduction (TTT, no reranking )|39.25%|—|
|Transduction (TTT + reranking)|43.00%|—|
|Ensemble (TTT + reranking)|56.75%|—|
|CodeIt (Butt et al., 2024)|15%|Yes, only trains on training set|
|Claude-3.5 / Greenblatt (2024)|21% / 42%|Yes, but closed LLMs at test time|
|Wind(2020)|39%|No,designed bylookingat val set|
|Akyürek et al. (2024), w/o our models|47.1%|Yes|
|+ensembled with & using both our models|61.9%|No, builds on our models|
|Avg/Best Human|60.2% / 97.8%|Yes|



**Scaling down our method.** Our flagship model is too expensive to run on the private test set hosted by Kaggle. We scale down by omitting test-time training, only sampling 336 programs, and reducing the transduction beam size to 3. This scores 19% on Kaggle and 36.5% on validation. Table 3 shows that program synthesis is less effective given this smaller search budget. Given the large-compute effectiveness of program synthesis, this suggests a strong payoff for smarter neural program search. 

Table 3: Smaller version of our model evaluated on the private test and public validation splits 

||Private Test Set|Public Validation Set|
|---|---|---|
|Transduction (no TTT, beam size 3)|18%|32.25%|
|Induction, 384 samples|4%|14%|
|Ensemble|19%|36.5%|



8 

Preprint. 

## 6 WHICH PROBLEMS ARE EASIER FOR THE MODELS, AND FOR HUMANS? 

## **Do problems that challenge humans also** 

**challenge the model, and vice-versa?** We sort ARC validation problems into 5 equallysized difficulty classes using data from LeGris et al. (2024). Figure 9 illustrates a peculiar relationship between human and model accuracy: All models surpass human performance on the hardest problems, but underperform on the easiest problems. Because our models train on simple Python programs, this suggests some problems are simple in code and learnable by transformers, but very hard for people—and conversely that people possess priors allowing effortless solution of problems beyond what our Python program generator produces. For problems of typical difficulty, the model roughly tracks human performance, and across all difficulty levels, transduction and induction serve complementary roles, even when augmented with test time training. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0009-04.png)


**----- Start of picture text -----**<br>
transduction+TTT<br>0.8 induction easiest 20%<br>ensemble<br>0.6 hardest 20%<br>0.4<br>0.2<br>0.2 0.4 0.6 0.8<br>human % solved<br>model % solved<br>**----- End of picture text -----**<br>


Figure 9: Human vs model performance across 5 difficulty levels. The easiest difficulty level contains problems in the top 20% of human accuracy, and the hardest difficulty level contains the 20% of problems with the lowest human accuracy. 

**Which concepts are easier for the models?** We test on ConceptARC (Moskvichev et al., 2023), an alternative ARC-style test-set which classifies its tasks into “concept groups” each exemplifying a single isolated high-level concept such as “sameness” or “above vs below.” We use models trained on ARC-Potpourri, finding that specific concept categories are easier for induction or transduction (Figure 10). We find an intuitive division of labor between the two approaches: Concept groups such as counting are best solved with symbolic code, while transduction better handles perceptual processes such as judging whether a shape is more horizontal or more vertical, or more top/bottom. 

ConceptARC reveals another dimension along which transduction and induction differ: Because ConceptARC illustrates one concept per problem, there is no need to compose many concepts together. Therefore the induction model, which is uniquely equipped for symbolic composition, loses a key advantage. Transduction has more limited composition capabilities but can instantiate individual concepts in flexible subsymbolic ways, which could explain why it excels on ConceptARC. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0009-08.png)


**----- Start of picture text -----**<br>
induction pass@1 transduction<br>**----- End of picture text -----**<br>


Figure 10: ConceptARC accuracy by concept group. Concept groups sorted left-to-right by ratio of inductive to transductive performance. IceCube is the original ARC Kaggle winner (Wind, 2020). We report pass@3 because Moskvichev et al. (2023) report accuracy given 3 attempts. 

9 

Preprint. 

## 7 RELATED WORK 

**ARC** was originally designed to challenge conventional deep learning and spur progress on alternative paradigms (Chollet, 2019). The first wave of successful approaches used discrete program search over domain-specific programming languages, including the original Kaggle winner (Wind, 2020). These symbolic approaches held their own against GPT-4 (Wang et al., 2024), but have recently been surpassed by transductive architectures using test-time training (Cole et al., 2024), and by LLM-guided program generation (Greenblatt, 2024). ARC has so far resisted conventional neural and symbolic approaches, but is solvable for adult humans, and to some extent, children (LeGris et al., 2024; Opielka et al., 2024). 

**Code generation via LLMs** is done in many recent works (Li et al., 2022; Gao et al., 2023; Chen et al., 2021; Austin et al., 2021). We most directly build on Li & Ellis (2024) and Greenblatt (2024). The former fine-tunes LLMs for inductive program synthesis using LLM-generated variations of human-written programs. While there are many technical differences, a key factor is that we generate function inputs by synthesizing an `input_generator` function, rather than have an LLM directly generate possible inputs. This matters because an LLM alone could not generate complex, precisely-correct inputs such as ARC grids. This potentially makes our work applicable to other few-shot generalization problems with complex input-spaces such as webpages, robot planning, etc. Greenblatt (2024) samples many Python programs from GPT4o: Comparable to our induction model, but instead of fine-tuning, it uses prompting. Fine-tuning forced us to create a dataset of new problems, which created the opportunity for exploring transductive models. 

Classic work in neural program synthesis has previously compared induction and transduction (RobustFill: Devlin et al. (2017)). We explore here a richer space of functions, reaching a qualitatively different conclusion than RobustFill: Instead of finding transduction inferior to induction, we find them complementary. More broadly, the transductive-inductive divide lies near the heart of supervised learning. Inductive approaches, such as linear regression, first construct a function _f_ where _f_ ( _**x**_ train) _≈_ _**y**_ train, and then predict _y_ test = _f_ ( _x_ test). Transductive approaches, such as Support Vector Machines and In-Context Learning, instead output their predictions by performing direct comparisons with the training data. We use the same neural network architecture and dataset to perform both tasks, allowing a controlled comparison between these paradigms. 

**Datasets for ARC.** ReARC (Hodel, 2024) is a dataset of handwritten programs that solve all ARCAGI training tasks, and which generates new inputs for them. ReARC is implemented in a domainspecific language, and lacks natural language annotations, making it difficult to remix with LLMs. Other works annotate ARC using either natural language (Acquaviva et al., 2022) or Python programs (Huang et al., 2024), which could potentially serve as seed programs for our work. Acquaviva et al. (2022) inspired our natural-language descriptions and Huang et al. (2024) influenced our seed program format. Our new seeds were a better fit for this approach because they encode shared priors in a Python library (Appendix A.2), and have an explicit `input_generator` describing a precisely-structured infinite space of valid inputs. 

## 8 DISCUSSION 

**What we learn about robust sample-efficient generalization.** Neither explicit symbolic hypotheses nor implicit neural representations suffice to solve all problems: each has their own domain of applicability, and simply ensembling models specialized in each does not cover all cases. Engineering a more clever neural program search, or training transductive predictors on more data, is unlikely to be fruitful. Instead we need representations irreducible to a purely neural or symbolic form, which intertwine inductive and transductive reasoning. One way of implementing this idea is to do program synthesis within a language whose atomic primitives are non-symbolic, and to pretrain those primitives to encapsulate the basic atoms of core knowledge. While work has taken steps in related directions (Reed & De Freitas, 2015; Alet et al., 2018; Tang & Ellis, 2023; Li et al., 2024), how to engineer and scale this idea remains open. 

**To what extent is this methodology applicable beyond ARC?** Few-shot function learning is a very flexible framework, but our particular method is most applicable when the target generalization can be described in symbolic code. As an immediately tangible example, web scraping and 

10 

Preprint. 

other forms of data-munging could fit within our framework. As a more ambitious goal, symbolic code is an especially good medium for defining precise models of how the world works. This is true both within the natural sciences (Schmidt & Lipson, 2009) and also within AI, with examples such as robotic policies (Liang et al., 2023), planners (Wong et al., 2023), and world models more broadly (Das et al., 2023; Tang et al., 2024b; Evans et al., 2021; Liang et al., 2024). These are not the kinds of programs that occur often in LLM pretraining data—so merely prompting is unlikely to perform well—but it is nonetheless feasible to curate around 100 seeds demonstrating what the system should learn. 

**Theoretically, induction and transduction should not be so complementary.** Equivalences between induction and transduction are well-know, such as the ‘kernel trick’ which allows translating parametric function fitting into a transductive problem. Our metalearning models, given infinite metatraining data, should similarly converge because transformers are universal function approximators. That there remains a difference is interesting precisely because it deviates from what one would expect theoretically. 

**Are we cheating by training on 400k synthetic problems?** The spirit of ARC is to generalize from few examples. Yet we fine-tune on many examples. In our view, the true training data is 160 seeds, not the 400k ‘remixes,’ which are instead analogous to ‘dream data’ in amortized inference or wake-sleep (Le et al., 2017; Ritchie et al., 2016; Hinton et al., 1995). In other words, our system inputs 160 annotated solutions to training set problems, and does up-front computation to convert that data into a neural network capable of solving new problems. From that perspective, it is a sample efficient way of learning to solve ARC—although it is not compute efficient. 

**Impact on ARC efforts.** Releasing our code and data helped Akyürek et al. (2024) achieve the first open-source system that performs at the level of an average human. The 2nd place ARC ’24 team (Franzen et al., 2024) also benefited from our data. We hope others will continue building on our work. We also intend our findings to encourage research on discrete program search as an alternative to the test-time training currently dominating in the community, and have geared our experiments toward showing the value in this other pathway. **From domain-specific languages to domain-specific libraries.** Many works that perform program search rely on carefully tuned domain-specific languages instead of using a general purpose language such as Python (Butt et al., 2024; Wind, 2020; Alford et al., 2022; Ainooson et al., 2023). However, we believe general-purpose programming languages can give much broader coverage, and that attempts to engineer restricted languages inevitably sacrifice regions of program-space which could plausibly occur in open-ended learning domains such as ARC. Instead we advocate here for domain-specific _libraries_ , which equip a general-purpose language with extra priors, but do not restrict it to using only those priors. 

**How to represent input-output mappings.** Our seeds include: (1) a grid transformation program, (2) an input generator, and (3) natural language descriptions for both (1) and (2). Practically, this representation allows us to sample consistent input-output example pairs for training, while the natural language descriptions help LLMs to remix seeds into novel problems. This further captures a latent natural language description of both inputs and outputs, from which the function and its preimage are derived. 

**Next steps suggested by biological intelligence & wake-sleep.** Our work has a straightforward analogy to dual-process models in psychology, which categorizes human thinking according to whether it relies on fast nonverbal intuitions or deliberative conscious thought (Smith & DeCoster, 2000; Kahneman, 2011). Although preliminary, our results could suggest that this partitioning is actually normative, and emerges from properties of the problems being solved, not properties of the solver itself. Human thinking is however more sophisticated in how it combines these modes: Fast intuitions can be further reprocessed by deliberative symbolic processing, which can trigger further intuitions, and so on. Our method has no analogous way of interleaving these two strategies, but more deeply integrating induction and transduction is a natural next step. 

Our approach can also be seen as a form of wake-sleep or dream learning (Hinton et al., 1995; Fosse et al., 2003; Rasch & Born, 2013) where samples from a generative model train an inference 

11 

Preprint. 

network. Here the generative model is a prompt with the seeds, and the inference network is our fine-tuned models. Drawing the analogy to wake-sleep suggests two directions. First, we could learn from recently-solved test problems (during ‘waking’) by generating fresh synthetic data using those problems as seeds (during ‘dreaming/sleeping’). Second, we could also implement a form of abstraction learning that automatically expands our custom ARC library (Appendix A.2), or adds new neural primitives to that library, in the spirit of library learning and modular metalearning (Bowers et al., 2023; Ellis et al., 2021; Alet et al., 2018). 

**Limitations.** Our system does not grow more competent at few-shot learning by solving new problems: Instead, it bootstraps from manually encoded knowledge in the seeds, which is transformed into a few-shot learner via an LLM training/inference pipeline. A more compelling approach would be to have the system discover for itself the knowledge that we compiled for it within the seeds, for instance by practicing on training tasks, without supervising on ground truth solutions. 

Our work is only evaluated on ARC. However, ARC is designed to contain many concepts and problems embedded within it, so can be viewed as an open-ended composite of different learning problems. Owing to this diversity, it is also notoriously challenging, and has resisted solution despite a series of high-profile competitions. We therefore believe that although evaluating on multiple benchmarks is desirable, ARC is an appropriate benchmark to use as the centerpiece of an experimental evaluation. 

**Code & Data Availability.** Our code, data, and model weights are freely available at `https: //github.com/xu3kev/BARC` . Interactive visualizations of our dataset and model outputs are **available at this link.** 

**Author Contributions.** Neural network experiments were engineered by Wen-Ding Li and Keya Hu. Data generation was engineered by Carter Larsen, Wen-Ding Li, Keya Hu, and Kevin Ellis. Kevin Ellis, Yewen Pu, Zenna Tavares, Wei-Long Zheng, and Hao Tang provided high-level advisory guidance. Zenna Tavares, Michelangelo Naim, Dat Nguyen, and Keya Hu analyzed the transduction model. Seeds were written by Keya Hu, Kevin Ellis, Carter Larsen, Yuqing Wu, Simon Alford, Caleb Woo, Spencer M. Dunn, and Yewen Pu. The paper was written by Yewen Pu, Keya Hu, Zenna Tavares, Wen-Ding Li, and Kevin Ellis. 

**Acknowledgements.** We are grateful for advice from Robert Hawkins regarding the statistical analysis in Figure 5C and for discussions with Weinan Sun about biological learning. This work was partly supported by an NSF CAREER award to K.E. 

## REFERENCES 

- Sam Acquaviva, Yewen Pu, Marta Kryven, Theodoros Sechopoulos, Catherine Wong, Gabrielle Ecanow, Maxwell Nye, Michael Henry Tessler, and Joshua B. Tenenbaum. Communicating natural programs to humans and machines. In _Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , 2022. URL `https://openreview.net/ forum?id=OxFoLTKDcNm` . 

- James Ainooson, Deepayan Sanyal, Joel P. Michelson, Yuan Yang, and Maithilee Kunda. A neurodiversity-inspired solver for the abstraction & reasoning corpus (arc) using visual imagery and program synthesis, 2023. URL `https://arxiv.org/abs/2302.09425` . 

- Ekin Akyürek, Mehul Damani, Linlu Qiu, Han Guo, Yoon Kim, and Jacob Andreas. The surprising effectiveness of test-time training for abstract reasoning, 2024. Preprint. 

- Ferran Alet, Tomás Lozano-Pérez, and Leslie P Kaelbling. Modular meta-learning. In _Conference on robot learning_ , pp. 856–868. PMLR, 2018. 

- Simon Alford, Anshula Gandhi, Akshay Rangamani, Andrzej Banburski, Tony Wang, Sylee Dandekar, John Chin, Tomaso Poggio, and Peter Chin. Neural-guided, bidirectional program search for abstraction and reasoning. In _Complex Networks & Their Applications X: Volume 1, Proceedings of the Tenth International Conference on Complex Networks and Their Applications COMPLEX NETWORKS 2021 10_ , pp. 657–668. Springer, 2022. 

12 

Preprint. 

- Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_ , 2021. 

- Matthew Bowers, Theo X. Olausson, Lionel Wong, Gabriel Grand, Joshua B. Tenenbaum, Kevin Ellis, and Armando Solar-Lezama. Top-down synthesis for library learning. _POPL_ , 2023. doi: 10.1145/3571234. URL `https://doi.org/10.1145/3571234` . 

- Natasha Butt, Blazej Manczak, Auke Wiggers, Corrado Rainone, David W Zhang, Michaël Defferrard, and Taco Cohen. Codeit: Self-improving language models with prioritized hindsight replay. In _International Conference on Machine Learning_ , 2024. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

François Chollet. On the measure of intelligence, 2019. 

- Jack Cole, Mohamed Osman, Michael Hodel, Keith Duggar, and Tim Scarfe. Machine learning street talk, June 2024. 

- Ria Das, Joshua B Tenenbaum, Armando Solar-Lezama, and Zenna Tavares. Combining functional and automata synthesis to discover causal reactive programs. _Proceedings of the ACM on Programming Languages_ , 7(POPL):1628–1658, 2023. 

- Jacob Devlin, Jonathan Uesato, Surya Bhupatiraju, Rishabh Singh, Abdel-rahman Mohamed, and Pushmeet Kohli. Robustfill: Neural program learning under noisy i/o. _ICML_ , 2017. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, 

13 

Preprint. 

Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzmán, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vítor Albiero, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu 

14 

Preprint. 

Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. The llama 3 herd of models, 2024. URL `https://arxiv.org/abs/2407.21783` . 

- Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sablé-Meyer, Lucas Morales, Luke Hewitt, Luc Cary, Armando Solar-Lezama, and Joshua B. Tenenbaum. Dreamcoder: Bootstrapping inductive program synthesis with wake-sleep library learning. In _PLDI_ , 2021. doi: 10.1145/3453483.3454080. URL `https://doi.org/10.1145/3453483.3454080` . 

- Richard Evans, Matko Bošnjak, Lars Buesing, Kevin Ellis, David Pfau, Pushmeet Kohli, and Marek Sergot. Making sense of raw input. _Artificial Intelligence_ , 299:103521, 2021. ISSN 0004-3702. doi: https://doi.org/10.1016/j.artint.2021.103521. URL `https://www.sciencedirect. com/science/article/pii/S0004370221000722` . 

- Magdalena J Fosse, Roar Fosse, J Allan Hobson, and Robert J Stickgold. Dreaming and episodic memory: a functional dissociation? _Journal of cognitive neuroscience_ , 15(1):1–9, 2003. 

- Daniel Franzen, Jan Disselhoff, and David Hartmann. The llm architect: Solving the arc challenge is a matter of perspective, 2024. URL `https://github.com/da-fr/arc-prize-2024/ blob/main/the_architects.pdf` . Preprint. 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In _International Conference on Machine Learning_ , pp. 10764–10799. PMLR, 2023. 

- Ryan Greenblatt. Draw more samples. " `https://redwoodresearch.substack.com/p/ getting-50-sota-on-arc-agi-with-gpt` ", 2024. Accuracy from ARCPrize Leaderboard. 

- Geoffrey E Hinton, Peter Dayan, Brendan J Frey, and Radford M Neal. The "wake-sleep" algorithm for unsupervised neural networks. _Science_ , 268(5214):1158–1161, 1995. 

- Céline Hocquette and Andrew Cropper. Relational decomposition for program synthesis. _arXiv preprint arXiv:2408.12212_ , 2024. 

- Michael Hodel. Rearc. `https://github.com/michaelhodel/re-arc` , 2024. [Online GitHub repository]. 

- Di Huang, Ziyuan Nan, Xing Hu, Pengwei Jin, Shaohui Peng, Yuanbo Wen, Rui Zhang, Zidong Du, Qi Guo, Yewen Pu, et al. Anpl: towards natural programming with interactive decomposition. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Marcus Hutter. _Universal artificial intelligence: Sequential decisions based on algorithmic probability_ . Springer Science & Business Media, 2004. 

Daniel Kahneman. _Thinking, fast and slow_ . macmillan, 2011. 

- Tuan Anh Le, Atilim Gunes Baydin, and Frank Wood. Inference Compilation and Universal Probabilistic Programming. In Aarti Singh and Jerry Zhu (eds.), _Proceedings of the 20th International Conference on Artificial Intelligence and Statistics_ , volume 54 of _Proceedings of Machine Learning Research_ , pp. 1338–1348. PMLR, 20–22 Apr 2017. URL `https://proceedings.mlr. press/v54/le17a.html` . 

- Seungpil Lee, Woochang Sim, Donghyeon Shin, Wongyu Seo, Jiwon Park, Seokki Lee, Sanha Hwang, Sejin Kim, and Sundong Kim. Reasoning abilities of large language models: In-depth analysis on the abstraction and reasoning corpus, 2024. URL `https://arxiv.org/abs/ 2403.11793` . 

- Solim LeGris, Wai Keen Vong, Brenden M. Lake, and Todd M. Gureckis. H-arc: A robust estimate of human performance on the abstraction and reasoning corpus benchmark, 2024. URL `https: //arxiv.org/abs/2409.01374` . 

15 

Preprint. 

- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ , 33: 9459–9474, 2020. 

- Chengshu Li, Jacky Liang, Andy Zeng, Xinyun Chen, Karol Hausman, Dorsa Sadigh, Sergey Levine, Li Fei-Fei, Fei Xia, and Brian Ichter. Chain of code: Reasoning with a language modelaugmented code emulator. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _Proceedings of Machine Learning Research_ , pp. 28259–28277. PMLR, 21–27 Jul 2024. URL `https://proceedings.mlr. press/v235/li24ar.html` . 

- Wen-Ding Li and Kevin Ellis. Is programming by example solved by llms?, 2024. 

- Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. _Science_ , 378(6624):1092–1097, 2022. 

- Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In _2023 IEEE International Conference on Robotics and Automation (ICRA)_ , pp. 9493–9500. IEEE, 2023. 

- Yichao Liang, Nishanth Kumar, Hao Tang, Adrian Weller, Joshua B. Tenenbaum, Tom Silver, João F. Henriques, and Kevin Ellis. Visualpredicator: Learning abstract world models with neurosymbolic predicates for robot planning, 2024. URL `https://arxiv.org/abs/2410. 23156` . 

- Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, and Pieter Abbeel. A simple neural attentive metalearner. _arXiv preprint arXiv:1707.03141_ , 2017. 

- Arsenii Kirillovich Moskvichev, Victor Vikram Odouard, and Melanie Mitchell. The conceptARC benchmark: Evaluating understanding and generalization in the ARC domain. _Transactions on Machine Learning Research_ , 2023. ISSN 2835-8856. URL `https://openreview.net/ forum?id=8ykyGbtt2q` . 

- Gustaw Opielka, Hannes Rosenbusch, Veerle Vijverberg, and Claire E. Stevenson. Do large language models solve arc visual analogies like people do?, 2024. URL `https://arxiv.org/ abs/2403.09734` . 

- Björn Rasch and Jan Born. About sleep’s role in memory. _Physiological reviews_ , 93(2):681–766, 2013. 

- Scott Reed and Nando De Freitas. Neural programmer-interpreters. _arXiv preprint arXiv:1511.06279_ , 2015. 

- Daniel Ritchie, Paul Horsfall, and Noah D Goodman. Deep amortized inference for probabilistic programs. _arXiv preprint arXiv:1610.05735_ , 2016. 

Jürgen Schmidhuber. Optimal ordered problem solver. _Machine Learning_ , 54(3):211–254, 2004. 

- Michael Schmidt and Hod Lipson. Distilling free-form natural laws from experimental data. _science_ , 324(5923):81–85, 2009. 

- Eliot R Smith and Jamie DeCoster. Dual-process models in social and cognitive psychology: Conceptual integration and links to underlying memory systems. _Personality and social psychology review_ , 4(2):108–131, 2000. 

- Ray J Solomonoff. A formal theory of inductive inference. _Information and control_ , 7(1):1–22, 1964. 

- Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt. Test-time training with self-supervision for generalization under distribution shifts. In _International conference on machine learning_ , pp. 9229–9248. PMLR, 2020. 

16 

Preprint. 

- Hao Tang and Kevin Ellis. From perception to programs: Regularize, overparameterize, and amortize. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 33616–33631. PMLR, 23–29 Jul 2023. URL `https://proceedings.mlr.press/v202/tang23c.html` . 

- Hao Tang, Keya Hu, Jin Peng Zhou, Sicheng Zhong, Wei-Long Zheng, Xujie Si, and Kevin Ellis. Code repair with llms gives an exploration-exploitation tradeoff. _NeurIPS_ , 2024a. 

- Hao Tang, Darren Key, and Kevin Ellis. Worldcoder, a model-based llm agent: Building world models by writing code and interacting with the environment. _NeurIPS_ , 2024b. 

- Luca H. Thoms, Karel A. Veldkamp, Hannes Rosenbusch, and Claire E. Stevenson. Solving arc visual analogies with neural embeddings and vector arithmetic: A generalized method. _ArXiv_ , abs/2311.08083, 2023. URL `https://api.semanticscholar.org/ CorpusID:265158110` . 

- Ruocheng Wang, Eric Zelikman, Gabriel Poesia, Yewen Pu, Nick Haber, and Noah D Goodman. Hypothesis search: Inductive reasoning with language models. _ICLR_ , 2024. 

- Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In _The 61st Annual Meeting Of The Association For Computational Linguistics_ , 2023. 

Johan Sokrates Wind. 1st place 2020 arc kaggle. `https://github.com/top-quarks/ ARC-solution` , 2020. [Online GitHub repository]. 

- Jonas Witt, Stef Rasing, Sebastijan Dumanˇci´c, Tias Guns, and Claus-Christian Carbon. A dividealign-conquer strategy for program synthesis, 2023. URL `https://arxiv.org/abs/ 2301.03094` . 

- Lionel Wong, Jiayuan Mao, Pratyusha Sharma, Zachary S Siegel, Jiahai Feng, Noa Korneev, Joshua B Tenenbaum, and Jacob Andreas. Learning adaptive planning representations with natural language guidance. _ICLR_ , 2023. 

17 

Preprint. 

## A DATA GENERATION TECHNICAL DETAILS 

Step1: Random remix to generate new language descriptions 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0018-03.png)


**----- Start of picture text -----**<br>
Prompt example<br>randomly sample LLM new language<br>100 seed  language new concepts :<br>problems concepts : new description:<br>description:<br>Step2: RAG to generate new codes Prompt example<br>calculate cosine  get four codes<br>similarity with  with the most  language<br>the seeds  similar language  concepts :<br>language part part with the  description:<br>languagenew  new description code LLM new code<br>new concepts : 100 seed<br>new description: problems<br>new language<br>new concepts:... new description:...<br>**----- End of picture text -----**<br>


Figure 11: A new natural language description is sampled by prompting an LLM with seed natural language descriptions, effectively using in-context learning to recombine and mutate elements of different problems. Code is generated for that new description via Retrieval Augmented Generation (RAG). Our RAG pipeline retrieves seeds with similar descriptions, and prompts an LLM to generate code for the new description, given the retrieved seeds. 

The prompt template for generating natural language descriptions by randomly sampling language descriptions from seed problems is as follows: 

```
You’vegeneratedtheseonpreviousrequests:
{examples}
Brainstorm{num_generations}more,usingsimilarthinking:
‘‘‘python
#concepts:
#<conceptsinyournewgeneration>
#description:
#<descriptionofyournewgeneration>
‘‘‘
```

The prompt template for generating Python code from the natural-language descriptions (and similar example code retrieved from the seeds, via RAG) is as follows: 

```
Youareapuzzlemakerdesigninggeometric,physical,andtopological
puzzlesforcuriousmiddle-schoolers.
```

```
Eachpuzzleconsistsofuncoveringadeterministicrule,pattern,procedure,
algorithm,ortransformationlawthatmapsinputstooutputs.
Boththeinputsandoutputsare2Dgridsofcoloredpixels.Thereare10
colors,buttheorderofthecolorsisneverrelevanttothepuzzle.
```

18 

Preprint. 

```
Themiddleschoolersaretryingtodiscoverthisdeterministic
transformation,whichcanbeimplementedasaPythonfunctioncalled‘main‘.
Designingapuzzleinvolvesalsocreatingexampleinputs,whichcanbe
implementedasaPythonfunctioncalled‘generate_input‘.Unlike‘main‘,the
‘generate_input‘functionshouldbestochastic,sothateverytimeyourun
it,yougetanothergoodexampleofwhatthetransformationcanbeapplied
to.
```

```
Hereisaoverviewofthepuzzleyouaredesigning:
```

```
{description}
```

```
Pleaseimplementthepuzzlebywritingcodecontainingthe‘generate_input‘
and‘main‘functions.Usethefollowingstandardlibrary(‘common.py‘):
```

```
‘‘‘python
{common_lib}
‘‘‘
```

```
Herearesomeexamplesfrompuzzleswithsimilardescriptionstoshowyou
howtousefunctionsin‘common.py‘:
```

```
{examples}
```

```
Yourtaskistoimplementthepuzzle,followingthesesteps:
```

`1. Inspect the example puzzle implementations, making note of the functions used and the physical/geometric/topological/logical details 2. Inspect the new puzzle’s description` 

`3. Brainstorm a possible implementation for the new puzzle 4. Generate a code block formatted like the earlier examples with a comment starting ‘# concepts:‘ listing the concepts and ‘# description:‘ describing the inputs and transformation from the given description.` 

```
Besuretomakethetransformation‘main‘deterministic.Followthe
descriptionclosely.
```

**Execution and Filtering of Generated Problems** We heuristically filter problems to improve the quality of data based on the following criteria: 

- The generator and transformation functions can be executed, producing at least 4 inputoutput grids examples. 

- Transformation being deterministic: We check for consistency by running the functions with different random seeds and filter out those with non-deterministic outputs. 

- Appropriate grid sizes: We remove input-output grids with height or width larger than 30, aligning with grid sizes in ARC 

- Color permutation check: Since we use numpy arrays with integers 0-9 to represent colors, we want to ensure transformations don’t rely on arithmetic operations of these integers. We filter this by checking if input-output remains consistent when permuting the underlying color-number mapping. 

- Removal of problems with all trivial identity input-output examples. 

19 

Preprint. 

## A.1 SEED EXAMPLES 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0020-02.png)


**----- Start of picture text -----**<br>
Example seed 0d3d703e Example seed 1b2d62fb Example seed 0dfd9992<br>**----- End of picture text -----**<br>


Figure 12: Three seed examples 

```
"""============problemid:0d3d703e============"""
fromcommonimport*
importnumpyasnp
fromtypingimport*
#concepts:
#colormapping
#description:
#Theinputisagridwhereeachcolumnisofthesamecolor.
#Tomaketheoutput,changeeachcoloraccordingtothefollowing
mapping:
#green->yellow,blue->gray,red->pink,teal->maroon,yellow->
green,gray->blue,pink->red,
maroon->teal
deftransform_grid(input_grid):
#Initializeoutputgrid
output_grid=input_grid.copy()
#Performscolormapping
output_grid=np.vectorize(lambdacolor:color_map.get(color,color))
(output_grid)
returnoutput_grid
#Constructingthecolormap
color_map={Color.GREEN:Color.YELLOW,
Color.BLUE:Color.GRAY,
Color.RED:Color.PINK,
Color.TEAL:Color.MAROON,
Color.YELLOW:Color.GREEN,
Color.GRAY:Color.BLUE,
Color.PINK:Color.RED,
Color.MAROON:Color.TEAL
}
defgenerate_input():
grid=np.full((3,3),Color.BLACK)
forxinrange(grid.shape[0]):
grid[x,:]=random.choice(list(color_map.keys()))
returngrid
"""============problemid:1b2d62fb============"""
importnumpyasnp
fromtypingimport*
fromcommonimport*
```

20 

Preprint. 

```
#concepts:
#booleanlogicaloperations,bitmaskswithseparator
#description:
#Intheinputyouwillseetwomaroonbitmasksseparatedbyablue
verticalbar
#Tomaketheoutput,colortealthepixelsthatarenotsetineither
bitmasks(logicalNOR)
```

```
deftransform_grid(input_grid:np.ndarray)->np.ndarray:
#Findtheblueverticalbar.VerticalmeansconstantX
forx_barinrange(input_grid.shape[0]):
ifnp.all(input_grid[x_bar,:]==Color.BLUE):
break
left_mask=input_grid[:x_bar,:]
right_mask=input_grid[x_bar+1:,:]
```

```
output_grid=np.zeros_like(left_mask)
output_grid[(left_mask!=Color.MAROON)&(right_mask!=Color.MAROON
)]=Color.TEAL
```

```
returnoutput_grid
```

```
defgenerate_input()->np.ndarray:
#createapairofequallysizedmaroonbitmasks
width,height=np.random.randint(2,10),np.random.randint(2,10)
grid1=np.zeros((width,height),dtype=int)
grid2=np.zeros((width,height),dtype=int)
```

```
forxinrange(width):
foryinrange(height):
grid1[x,y]=np.random.choice([Color.MAROON,Color.BLACK])
grid2[x,y]=np.random.choice([Color.MAROON,Color.BLACK])
```

```
#createablueverticalbar
bar=np.zeros((1,height),dtype=int)
bar[0,:]=Color.BLUE
```

```
grid=np.concatenate((grid1,bar,grid2),axis=0)
```

```
returngrid
```

```
"""============problemid:0dfd9992============"""
```

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
#concepts:
#occlusion,translationalsymmetry
#description:
#Intheinputyouwillseeatranslationallysymmetricpatternrandomly
occludedbyblackpixels.
#Tomaketheoutput,removetheoccludingblackpixelstorevealthe
translationallysymmetricpattern.
```

```
deftransform_grid(input_grid):
#Plan:
#1.Findthetranslationalsymmetries
```

21 

Preprint. 

```
#2.Reconstructthespritebyignoringtheblackpixelsand
exploitingthesymmetry
w,h=input_grid.shape
```

```
#Identifythetranslationalsymmetries
translations=detect_translational_symmetry(input_grid,
ignore_colors=[Color.BLACK])
assertlen(translations)>0,"Notranslationalsymmetryfound"
```

```
#Reconstructtheoccludedblackpixelsbyreplacingthemwithcolors
foundintheorbitofthe
symmetries
output_grid=np.copy(input_grid)
forxinrange(w):
foryinrange(h):
ifoutput_grid[x,y]==Color.BLACK:
#Usethetranslationalsymmetrytofillintheoccluded
pixels
#todothiswecomputetheORBITofthecurrentpixel
underthe
translations
```

```
#andtakethemostcommonnon-blackcolorintheorbit
#Computetheorbitintotheoutput
orbit_pixels=orbit(output_grid,x,y,translations)
orbit_colors={input_grid[transformed_x,transformed_y]
fortransformed_x,transformed_yin
orbit_pixels}
```

```
#occludedbyblack,sowhatevercoloritis,blackdoesn
’tcount
orbit_colors=orbit_colors-{Color.BLACK}
#Copythecolor
assertlen(orbit_colors)==1,"Ambiguity:multiple
colorsintheorbit"
output_grid[x,y]=orbit_colors.pop()
```

```
returnoutput_grid
```

```
defgenerate_input():
#Makearandomlargecanvas
grid=np.full((np.random.randint(15,30),np.random.randint(15,30))
,Color.BLACK)
#Makethebasicsprite
w,h=random.randint(3,8),random.randint(3,8)
sprite=random_sprite(w,h,density=1,color_palette=Color.NOT_BLACK
)
#Placethespriteinthecanvas
forxinrange(0,grid.shape[0],w):
foryinrange(0,grid.shape[1],h):
blit_sprite(grid,sprite,x,y)
#Createrandomoccluders
n_occluders=random.randint(1,5)
for_inrange(n_occluders):
x,y=random.randint(0,grid.shape[0]),random.randint(0,grid.
shape[1])
w,h=random.randint(3,7),random.randint(3,7)
occluder_sprite=np.full((w,h),Color.BLACK)
blit_sprite(grid,occluder_sprite,x,y)
```

22 

Preprint. 

```
returngrid
```

- A.2 COMMON LIBRARY 

23 

Preprint. 

```
"""CommonlibraryforARC"""
```

```
importnumpyasnp
importrandom
classColor:
"""
Enumforcolors
Color.BLACK,Color.BLUE,Color.RED,Color.GREEN,Color.YELLOW,
Color.GREY,Color.PINK,Color.ORANGE,Color.TEAL,Color.MAROON
UseColor.ALL_COLORSfor‘set‘ofallpossiblecolors
UseColor.NOT_BLACKfor‘set‘ofallcolorsexceptblack
Colorsarestrings(NOTintegers),
soyouCAN’Tdomath/arithmetic/indexingonthem.
(TheexceptionisColor.BLACK,whichis0)
"""
defflood_fill(grid,x,y,color,connectivity=4):
"""
Filltheconnectedregionthatcontainsthepoint(x,y)with
thespecifiedcolor.
connectivity:4or8,for4-wayor8-wayconnectivity.
8-waycountsdiagonalsasconnected,
4-wayonlycountscardinaldirectionsasconnected.
"""
```

```
defdraw_line(grid,x,y,end_x=None,end_y=None,length=None,direction=
None,
color=None,stop_at_color=[]):
```

```
"""
```

```
Drawsalinestartingat(x,y)extendingto(end_x,end_y)or
ofthespecifiedlengthinthespecifieddirection
Directionshouldbeavectorwithelements-1,0,or1.
IflengthisNone,thenthelinewillcontinueuntilithits
theedgeofthegrid.
```

```
stop_at_color:optionallistofcolorsthatthelineshouldstopat.
Ifthelinehitsapixelofoneofthesecolors,itwillstop.
```

```
Example:
#bluediagonallinefrom(0,0)to(2,2)
draw_line(grid,0,0,length=3,color=blue,direction=(1,1))
draw_line(grid,0,0,end_x=2,end_y=2,color=blue)
"""
```

```
deffind_connected_components(grid,background=Color.BLACK,connectivity=
4,
```

```
monochromatic=True):
```

```
"""
Findtheconnectedcomponentsinthegrid.
Returnsalistofconnected
components,whereeachconnectedcomponentisanumpyarray.
```

```
connectivity:4or8,for4-wayor8-wayconnectivity.
monochromatic:ifTrue,eachconnectedcomponentisassumedtohave
onlyonecolor.
IfFalse,eachconnectedcomponentcanincludemultiplecolors.
"""
```

24 

Preprint. 

```
defrandom_scatter_points(grid,color,density=0.5,
background=Color.BLACK):
```

```
"""
Randomlyscatterpointsofthespecifiedcolorinthegridwith
specifieddensity.
"""
```

```
defscale_pattern(pattern,scale_factor):
"""
Scalesthepatternbythespecifiedfactor.
"""
```

```
defblit_object(grid,obj,background=Color.BLACK):
```

```
"""
```

```
Drawsanobjectontothegridusingitscurrentlocation.
Exampleusage:
blit_object(output_grid,an_object,background=background_color)
"""
```

```
defblit_sprite(grid,sprite,x,y,background=Color.BLACK):
"""
Drawsaspriteontothegridatthespecifiedlocation.
```

```
Exampleusage:
blit_sprite(output_grid,the_sprite,x=x,y=y,
background=background_color)
"""
```

```
defbounding_box(grid,background=Color.BLACK):
"""
Findtheboundingboxofthenon-backgroundpixelsinthegrid.
Returnsatuple(x,y,width,height)oftheboundingbox.
```

```
Exampleusage:
objects=find_connected_components(input_grid,monochromatic=True,
background=Color.BLACK,connectivity=8)
teal_object=[objforobjinobjectsifnp.any(obj==Color.TEAL)][0]
teal_x,teal_y,teal_w,teal_h=bounding_box(teal_object)
"""
```

```
defobject_position(obj,background=Color.BLACK,anchor="upperleft"):
"""
```

```
(x,y)positionoftheprovidedobject.
Bydefault,theupperleftcorner.
```

```
anchor:"upperleft","upperright","lowerleft","lowerright",
"center","uppercenter","lowercenter","leftcenter","right
center"
Exampleusage:
x,y=object_position(obj,background=background_color,
anchor="upperleft")
middle_x,middle_y=object_position(obj,background=background_color
,
anchor="center")
```

```
"""
```

```
defcrop(grid,background=Color.BLACK):
"""
```

```
Cropthegridtothesmallestboundingboxthatcontainsall
non-backgroundpixels.
```

```
Exampleusage:
```

25 

Preprint. 

```
#Extractaspritefromanobject
sprite=crop(an_object,background=background_color)
"""
deftranslate(obj,x,y,background=Color.BLACK):
"""
Translatebythevector(x,y).Fillsinthenewpixelswiththe
backgroundcolor.
Exampleusage:
red_object=...#extractsomeobject
shifted_red_object=translate(red_object,x=1,y=1)
blit_object(output_grid,shifted_red_object,
background=background_color)
"""
```

```
defcollision(_=None,object1=None,object2=None,x1=0,y1=0,x2=0,y2=0,
background=Color.BLACK):
"""
Checkifobject1andobject2collidewhenobject1isat(x1,y1)and
object2isat(x2,y2).
```

```
Exampleusage:
```

```
#Checkifaspritecanbeplacedontoagridat(X,Y)
collision(object1=output_grid,object2=a_sprite,x2=X,y2=Y)
#Checkiftwoobjectscollide
collision(object1=object1,object2=object2,
x1=X1,y1=Y1,x2=X2,y2=Y2)
"""
```

```
defcontact(_=None,object1=None,object2=None,x1=0,y1=0,x2=0,y2=0,
background=Color.BLACK,connectivity=4,):
"""
Checkifobject1andobject2toucheachother(havecontact)
whenobject1isat(x1,y1)andobject2isat(x2,y2).
Theyaretouchingeachotheriftheyshareaborder,orifthey
overlap.
Collisionimpliescontact,butcontactdoesnotimplycollision.
```

```
connectivity:4or8,for4-wayor8-wayconnectivity.
(8-waycountsdiagonalsastouching,
4-wayonlycountscardinaldirectionsastouching)
```

```
Exampleusage:
```

```
#Checkifaspritetouchesanythingifitweretobeplacedat(X,Y)
contact(object1=output_grid,object2=a_sprite,x2=X,y2=Y)
```

```
#Checkiftwoobjectstoucheachother
contact(object1=object1,object2=object2)
"""
```

```
defgenerate_position_has_interval(max_len,position_num,if_padding=
False):
"""
Generatethepositionofthelineswithrandominterval.
"""
```

```
defrandom_free_location_for_sprite(grid,sprite,background=Color.BLACK,
border_size=0,padding=0,
padding_connectivity=8):
"""
Findarandomfreelocationforthespriteinthegrid
```

26 

Preprint. 

```
Returnsatuple(x,y)ofthetop-leftcornerofthespriteinthe
grid,whichcanbepassedto‘blit_sprite‘
```

```
border_size:minimumdistancefromtheedgeofthegrid
background:colortreatedastransparent
padding:ifnon-zero,thespritewillbepaddedwithanon-background
colorbeforecheckingforcollision
padding_connectivity:4or8,for4-wayor8-wayconnectivitywhen
paddingthesprite
Exampleusage:
x,y=random_free_location_for_sprite(grid,sprite,padding=1,
padding_connectivity=8,border_size=1,background=Color.BLACK)
#findthelocation,usinggenerouspadding
assertnotcollision(object1=grid,object2=sprite,x2=x,y2=y)
blit_sprite(grid,sprite,x,y)
"""
```

```
defobject_interior(grid,background=Color.BLACK):
"""
```

```
Computestheinterioroftheobject(includingedges)
```

```
returnsanewgridof‘bool‘whereTrueindicatesthatthepixelis
partoftheobject’sinterior.
```

```
Exampleusage:
interior=object_interior(obj,background=Color.BLACK)
forx,yinnp.argwhere(interior):
#x,yiseitherinsidetheobjectoratleastonitsedge
"""
```

```
defobject_boundary(grid,background=Color.BLACK):
"""
Computestheboundaryoftheobject(excludinginterior)
returnsanewgridof‘bool‘whereTrueindicatesthatthepixelis
partoftheobject’sboundary.
Exampleusage:
boundary=object_boundary(obj,background=Color.BLACK)
assertnp.all(obj[boundary]!=Color.BLACK)
"""
defobject_neighbors(grid,background=Color.BLACK,connectivity=4):
"""
Computesamaskofthepointsthatneighbororbordertheobject,but
arenotpartoftheobject.
returnsanewgridof‘bool‘whereTrueindicatesthatthepixelis
partoftheobject’sborderneighbors5.
Exampleusage:
neighbors=object_neighbors(obj,background=Color.BLACK)
assertnp.all(obj[neighbors]==Color.BLACK)
"""
classSymmetry:
"""
Symmetrytransformations,whichtransformedthe2Dgridinwaysthat
preservevisualstructure.
Returnedby‘detect_rotational_symmetry‘,
‘detect_translational_symmetry‘,‘detect_mirror_symmetry‘.
"""
defapply(self,x,y,iters=1):
```

27 

Preprint. 

```
"""
Applythesymmetrytransformationtothepoint(x,y)‘iters‘
times.
Returnsthetransformedpoint(x’,y’)
"""
```

```
deforbit(grid,x,y,symmetries):
"""
Computetheorbitofthepoint(x,y)underthesymmetry
transformations‘symmetries‘.
Theorbitisthesetofpointsthatthepoint(x,y)mapstoafter
applyingthesymmetrytransformationsdifferentnumbersoftimes.
Returnsalistofpointsintheorbit.
Example:
symmetries=detect_rotational_symmetry(input_grid)
forx,yinnp.argwhere(input_grid!=Color.BLACK):
#Computeorbitontothetargetgrid,whichistypicallythe
output
symmetric_points=orbit(output_grid,x,y,symmetries)
#...nowwedosomethingwiththemlikecopycolorsorinfer
missingcolors
"""
defdetect_translational_symmetry(grid,ignore_colors=[Color.BLACK]):
"""
Findstranslationalsymmetriesinagrid.
Satisfies:grid[x,y]==grid[x+translate_x,y+translate_y]for
allx,y,aslongasneitherpixelisin‘ignore_colors‘.
```

```
ReturnsalistofSymmetryobjects,eachrepresentingadifferent
translationalsymmetry.
```

```
Example:
symmetries=detect_translational_symmetry(grid,ignore_colors=[
occluder_color])
forx,yinnp.argwhere(grid!=occluder_color):
#Computeorbitontothetargetgrid
#Whencopyingtoanoutput,thisisusuallytheoutputgrid
symmetric_points=orbit(grid,x,y,symmetries)
forx,yinsymmetric_points:
assertgrid[x,y]==grid[x,y]orgrid[x,y]==
occluder_color
"""
detect_mirror_symmetry(grid,grid,,ignore_colors=[Color.BLACK]):=[Color.BLACK]):Color.BLACK]):]):)::
"""
Returnslistofmirrorsymmetries.
Satisfies:grid[x,y]==grid[2*mirror_x-x,2*mirror_y-y]
forallx,y,aslongasneitherpixelisin‘ignore_colors‘
Example:
symmetries=detect_mirror_symmetry(grid,ignore_colors=[Color.BLACK])
#ignore_color:Incasepartsoftheobjecthavebeenremovedand
#occludedbyblack
forx,yinnp.argwhere(grid!=Color.BLACK):
forsyminsymmetries:
symmetric_x,symmetric_y=sym.apply(x,y)
assertgrid[symmetric_x,symmetric_y]==grid[x,y]
orgrid[symmetric_x,symmetric_y]==Color.BLACK
Ifthegridhasbothhorizontalandverticalmirrorsymmetries,
thereturnedlistwillcontaintwoelements.
"""
```

```
defdetect_mirror_symmetry(grid,grid,,ignore_colors=[Color.BLACK]):=[Color.BLACK]):Color.BLACK]):]):)::
"""
```

28 

Preprint. 

```
defdetect_rotational_symmetry(grid,ignore_colors=[Color.BLACK]):
"""
```

```
Findsrotationalsymmetryinagrid,orreturnsNoneifnosymmetry
ispossible.
Satisfies:grid[x,y]==grid[y-rotate_center_y+rotate_center_x,
-x+rotate_center_y+rotate_center_x]
#clockwise
grid[x,y]==grid[-y+rotate_center_y+rotate_center_x,
x-rotate_center_y+rotate_center_x]
#counterclockwise
forallx,y,aslongasneitherpixelisinignore_colors
Example:
sym=detect_rotational_symmetry(grid,ignore_colors=[Color.BLACK])
#ignore_color:Incasepartsoftheobjecthavebeenremovedand
#occludedbyblack
forx,yinnp.argwhere(grid!=Color.BLACK):
rotated_x,rotated_y=sym.apply(x,y,iters=1)#+1clockwise,
-1counterclockwise
assertgrid[rotated_x,rotated_y]==grid[x,y]or
grid[rotated_x,rotated_y]==Color.BLACK
print(sym.center_x,sym.center_y)#Incasetheseareneeded,they
arefloats
"""
```

```
defis_contiguous(bitmask,background=Color.BLACK,connectivity=4):
"""
Checkifanarrayiscontiguous.
```

```
background:Colorthatcountsastransparent(default:Color.BLACK)
connectivity:4or8,for4-way(onlycardinaldirections)or
8-wayconnectivity(alsodiagonals)(default:4)
ReturnsTrue/False
"""
```

```
defrandom_sprite(n,m,density=0.5,symmetry=None,color_palette=None,
connectivity=4,background=Color.BLACK):
"""
Generateasprite(anobject),representedasanumpyarray.
n,m:dimensionsofthesprite.Ifthesearelists,thenarandom
valuewillbechosenfromthelist.
symmetry:optionaltypeofsymmetrytoapplytothesprite.Canbe
’horizontal’,’vertical’,’diagonal’,’radial’,’not_symmetric’.If
None,arandomsymmetrytypewillbechosen.
color_palette:optionallistofcolorstouseinthesprite.IfNone,
arandomcolorpalettewillbechosen.
Returnsan(n,m)NumPyarrayrepresentingthesprite.
"""
defdetect_objects(grid,_=None,predicate=None,background=Color.BLACK,
monochromatic=False,connectivity=None,
allowed_dimensions=None,
colors=None,can_overlap=False):
"""
Detectsandextractsobjectsfromthegridthatsatisfycustom
specification.
```

```
predicate:
afunctionthattakesacandidateobjectasinputand
returnsTrueifitcountsasanobject
background:
```

29 

Preprint. 

```
colortreatedastransparent
monochromatic:
ifTrue,eachobjectisassumedtohaveonlyonecolor
IfFalse,eachobjectcanincludemultiplecolors.
connectivity:
4or8,for4-wayor8-wayconnectivity.
IfNone,theconnectivityisdeterminedautomatically.
allowed_dimensions:
alistoftuples(n,m)specifyingthealloweddimensionsofthe
objects.
IfNone,objectsofanysizeareallowed.
colors:
alistofcolorsthattheobjectsareallowedtohave.
IfNone,objectsofanycolorareallowed.
can_overlap:ifTrue,objectscanoverlap.
IfFalse,objectscannotoverlap.
Returnsalistofobjects,whereeachobjectisanumpyarray.
"""
```

30 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0031-01.png)


**----- Start of picture text -----**<br>
concepts:  concepts:  concepts:<br>color inversion,  pixel sorting, frequency counting,  shape completion, color filling<br>objecxt transformation color arrangement<br>concepts:  concepts:  concepts:<br>symmetry detection,  color blending,  object detection, shape expansion,<br>reflection radial expansion color filling<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0031-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0031-03.png)


**----- Start of picture text -----**<br>
concepts:<br>pattern detection, copying,<br>positioning<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0031-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0031-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0031-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0031-07.png)


**----- Start of picture text -----**<br>
concepts:  concepts:<br>inverse patterns, scaling color blending,<br>shape merging<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0031-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0031-09.png)


Figure 13: Nine example problems generated automatically by our pipeline. 

31 

Preprint. 

## A.3 GENERATED ARC EXAMPLES 

## B FINE TUNING TRAINING DETAILS 

## B.1 PROMPTING THE MODELS 

We must include in our prompts for our fine-tuned models the input/output 2D colored grids of each problem. To do this we represent the problem textually by naming the colors one-by-one. We renamed certain colors which were more than one token (e.g., maroon _→_ brown saves 1 token/pixel), and presented the grid as a whitespace-delimited 2D array with newlines delimiting rows. Please see below. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0032-05.png)


**----- Start of picture text -----**<br>
Train examples Test example<br>Example 1 Example 2 Example 3<br>**----- End of picture text -----**<br>


Figure 14: Prompt example illustration 

Transduction example: 

```
-----Role:system--------------------
Youareaworld-classpuzzlesolverwithexceptionalpatternrecognition
skills.Yourtaskistoanalyzepuzzles,spotpatterns,andprovidedirect
solutions.
```

```
-----Role:user--------------------
Giveninput-outputgridpairsasreferenceexamples,carefullyobservethe
patternstopredicttheoutputgridfornewtestinput.Eachpairfollows
thesametransformationrule.Gridsare2Darraysrepresentedasstrings,
withcells(colors)separatedbyspacesandrowsbynewlines.
Herearetheinputandoutputgridsforthereferenceexamples:
Example1
Input:
BlackGrayBlack
GrayGrayGray
BlackGrayBlack
```

```
Output:
BlackBlackBlueRedBlackBlack
BlackBlackRedBlueBlackBlack
BlueRedBlueRedBlueRed
RedBlueRedBlueRedBlue
BlackBlackBlueRedBlackBlack
BlackBlackRedBlueBlackBlack
```

```
Example2
Input:
GrayBlackBlack
BlackGrayBlack
BlackBlackGray
```

```
Output:
BlueRedBlackBlackBlackBlack
RedBlueBlackBlackBlackBlack
BlackBlackBlueRedBlackBlack
BlackBlackRedBlueBlackBlack
```

32 

Preprint. 

```
BlackBlackBlackBlackBlueRed
BlackBlackBlackBlackRedBlue
Example3
Input:
BlackGrayBlack
BlackGrayGray
GrayGrayBlack
Output:
BlackBlackBlueRedBlackBlack
BlackBlackRedBlueBlackBlack
BlackBlackBlueRedBlueRed
BlackBlackRedBlueRedBlue
BlueRedBlueRedBlackBlack
RedBlueRedBlueBlackBlack
```

```
Hereistheinputgridforthetestexample:
Input:
BlackBlackBlack
BlackGrayBlack
GrayGrayGray
```

```
Directlyprovidetheoutputgridscorrespondingtothegiventestinput
grids,basedonthepatternsobservedinthereferenceexamples.
-----Role:assistant--------------------
Theoutputgridforthetestinputgridis:
‘‘‘
BlackBlackBlackBlackBlackBlack
BlackBlackBlackBlackBlackBlack
BlackBlackBlueRedBlackBlack
BlackBlackRedBlueBlackBlack
BlueRedBlueRedBlueRed
RedBlueRedBlueRedBlue
‘‘‘
```

Induction example: 

```
-----Role:system--------------------
Youareaworld-classpuzzlesolverwithexceptionalpatternrecognition
skillsandexpertiseinPythonprogramming.Yourtaskistoanalyzepuzzles
andprovidePythonsolutions.
```

```
-----Role:user--------------------
Giveninput-outputgridpairsasreferenceexamples,carefullyobservethe
patternstopredicttheoutputgridfornewtestinput.Eachpairfollows
thesametransformationrule.Gridsare2Darraysrepresentedasstrings,
withcells(colors)separatedbyspacesandrowsbynewlines.
Herearetheinputandoutputgridsforthereferenceexamples:
Example1
```

```
Input:
```

```
BlackGrayBlack
GrayGrayGray
BlackGrayBlack
```

```
Output:
BlackBlackBlueRedBlackBlack
BlackBlackRedBlueBlackBlack
BlueRedBlueRedBlueRed
RedBlueRedBlueRedBlue
```

33 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0034-01.png)


**----- Start of picture text -----**<br>
Black Black Blue Red Black Black<br>Black Black Red Blue Black Black<br>Example 2<br>Input:<br>Gray Black Black<br>Black Gray Black<br>Black Black Gray<br>Output:<br>Blue Red Black Black Black Black<br>Red Blue Black Black Black Black<br>Black Black Blue Red Black Black<br>Black Black Red Blue Black Black<br>Black Black Black Black Blue Red<br>Black Black Black Black Red Blue<br>Example 3<br>Input:<br>Black Gray Black<br>Black Gray Gray<br>Gray Gray Black<br>Output:<br>Black Black Blue Red Black Black<br>Black Black Red Blue Black Black<br>Black Black Blue Red Blue Red<br>Black Black Red Blue Red Blue<br>Blue Red Blue Red Black Black<br>Red Blue Red Blue Black Black<br>Here is the input grid for the test example:<br>Input:<br>Black Black Black<br>Black Gray Black<br>Gray Gray Gray<br>Write a Python function ‘transform‘ that can convert any given input grid to<br>its corresponding output grid based on the pattern observed in the<br>reference examples.<br>----- Role: assistant --------------------<br>Let’s solve this puzzle using Python code with the common library functions.<br>We’ll first reason about the problem and then write the code to solve it.<br>The ‘transform‘ function will take the input grid and return the output grid<br>. Here is the Python code with the comments describing how to solve the<br>problem:<br>‘‘‘python<br>from common import *<br>import numpy as np<br>from typing import *<br># concepts:<br># repetition, diagonal lines, color change<br># description:<br># In the input you will see a 3x3 grid with a specific diagonal pattern of<br>gray pixels.<br># The output should be the result of repeating this diagonal pattern<br>horizontally and vertically<br>**----- End of picture text -----**<br>


34 

Preprint. 

```
#tofilla6x6grid,changingthecolorfromgraytoblueandredinan
alternatingmanner.
```

```
deftransform(input_grid:np.ndarray)->np.ndarray:
#Createanew6x6outputgridfilledwiththebackgroundcolor(black)
output_grid=np.full((6,6),Color.BLACK)
#Checkforthediagonalpatternintheinputgrid
foriinrange(3):
forjinrange(3):
ifinput_grid[i,j]==Color.GRAY:
#Fillthesamepositionintheoutputgridwithalternating
colors
output_grid[i*2,j*2]=Color.BLUE
output_grid[i*2,j*2+1]=Color.RED
output_grid[i*2+1,j*2]=Color.RED
output_grid[i*2+1,j*2+1]=Color.BLUE
returnoutput_grid
‘‘‘
```

## C EXAMPLES OF INDUCTION SOLUTIONS 

C.1 ARC PROBLEM F3CDC58F 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
```

```
#concepts:
#colorcounting,gridtransformation
#description:
#Intheinput,youwillseeagridwithseveralcoloredsquares.
#Tomaketheoutput,countthenumberofsquaresofeachcolor.
#Filltheoutputgridwithbarsofcorrespondingcolors.
#Theheightofeachbarcorrespondstothenumberofsquaresofthat
color.
deftransform(input_grid:np.ndarray)->np.ndarray:
#Createablankoutputgrid
output_grid=np.zeros_like(input_grid)
#Countthenumberofsquaresofeachcolor
color_counts={color:0forcolorinColor.NOT_BLACK}
forrowininput_grid:
forcellinrow:
ifcellincolor_counts:
color_counts[cell]+=1
#Definetheheightofthebarsbasedonthecounts
max_height=10#Maximumheightofthebars
forcolor,countincolor_counts.items():
ifcount>0:
forhinrange(min(count,max_height)):
output_grid[-(h+1),(color-1)%
returnoutput_grid
```

35 

Preprint. 

## C.2 ARC PROBLEM BE03B35F 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
#concepts:
#colordetection,patternextraction,rotationtransformation
#description:
#Intheinput,youwillseea5x5gridconsistingofacoloredpattern
inthetopleftcorneranda2x2
squareofredpixelsinthebottom
rightcorner.
#Dependingonthecolorofthetopleftcorner(blueorgreen),rotate
thepatternby90degreesclockwise
orcounterclockwise.
#Theoutputgridshouldbethetransformedpatternplacedinthesame
positionastheoriginaltopleft
pattern.
deftransform(input_grid):
#Extractthetopleft5x5pattern
pattern=input_grid[0:5,0:5]
#Identifythecolorinthetopleftcorner
corner_color=pattern[0,0]
#Definerotationbasedonthecornercolor
ifcorner_color==Color.BLUE:
transformed_pattern=np.rot90(pattern,k=-1)#Rotate90
degreescounterclockwise
elifcorner_color==Color.GREEN:
transformed_pattern=np.rot90(pattern,k=1)#Rotate90
degreesclockwise
else:
raiseValueError("ThecornercolormustbeeitherBLUEorGREEN."
)
#Createtheoutputgrid
output_grid=np.full((2,2),Color.BLACK)#Initializeasmallgrid
output_grid=blit_sprite(output_grid,transformed_pattern,x=0,y=0,
background=Color.BLACK)
```

```
returnoutput_grid
```

C.3 ARC PROBLEM 2072ABA6 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
#concepts:
#colormapping,patternreplication
#description:
#Intheinputyouwillseea3x3gridofcoloredpixels.
#Thecolorsareeitherblackorgray.Theoutputshouldbeagridwhere:
#1.Ifthepixelisblack,itremainsblackintheoutput.
#2.Ifthepixelisgray,itshouldbereplacedwitha2x2blockofblue
andredpixelsinacheckerboard
pattern.
```

36 

Preprint. 

```
#Thebluepixelshouldoccupythetop-leftandbottom-rightofthe2x2
block,whiletheredpixeloccupies
thetop-rightandbottom-left.
```

```
deftransform(input_grid):
#Createanoutputgridthatislargerthantheinputgrid
output_grid=np.zeros((input_grid.shape[0]*2,input_grid.shape[1]
*2),dtype=int)
#Filltheoutputgridbasedontheinputgrid
forxinrange(input_grid.shape[0]):
foryinrange(input_grid.shape[1]):
ifinput_grid[x,y]==Color.BLACK:
#Blackstaysblackintheoutput
continue
else:
#Replacegraywithacheckerboardpatternofblueand
red
output_grid[2*x:2*x+2,2*y:2*y+2]=[
[Color.BLUE,Color.RED],
[Color.RED,Color.BLUE]
]
```

```
returnoutput_grid
```

## C.4 ARC PROBLEM EF26CBF6 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
#concepts:
#colortransformation,gridsections,boundarydetection
#description:
#Intheinput,youwillseeagridwithapatternofyellowandblue
pixelswithablackbackground,
#andmultiplecoloredcircles(notyelloworblue)placedrandomly
withinthegrid.
#Thegoalistotransformtheoutputgridbyreplacingalltheblue
pixelswiththecoloroftheclosest
circle
#andkeepingtheyellowpixelsunchanged.
deftransform(input_grid:np.ndarray)->np.ndarray:
#Createacopyoftheinputgridtomodify
output_grid=np.copy(input_grid)
#Findthecoordinatesofthecoloredcircles
circle_coordinates=np.argwhere((input_grid!=Color.BLACK)&(
input_grid!=Color.YELLOW)&(
input_grid!=Color.BLUE))
#Iterateoverthegridandreplacebluepixelswiththenearest
circlecolor
forxinrange(input_grid.shape[0]):
foryinrange(input_grid.shape[1]):
ifinput_grid[x,y]==Color.BLUE:
#Findtheclosestcircle
closest_color=None
closest_distance=float(’inf’)
forcircle_x,circle_yincircle_coordinates:
```

37 

Preprint. 

```
distance=abs(circle_x-x)+abs(circle_y-y)#
Manhattan
distance
ifdistance<closest_distance:
closest_distance=distance
closest_color=input_grid[circle_x,circle_y]
#Replacethebluepixelwiththeclosestcirclecolor
output_grid[x,y]=closest_color
```

```
returnoutput_grid
```

## C.5 ARC PROBLEM E7639916 

```
fromcommonimport*
```

```
importnumpyasnp
fromtypingimport*
```

```
#concepts:
```

```
#boundarydetection,colorfilling
#description:
#Intheinputgrid,youwillseeablackbackgroundwithseveralpurple
pixelsformingaboundary.
#Thetaskistofilltheareaenclosedbythepurpleboundarywithblue
pixels.
deftransform(input_grid):
#Createanoutputgridthatstartsasacopyoftheinputgrid
output_grid=np.copy(input_grid)
#Findthecoordinatesofthepurplepixels
purple_coords=np.argwhere(input_grid==Color.PURPLE)
#Ifnopurplepixelsarefound,returntheoriginalgrid
iflen(purple_coords)==0:
returnoutput_grid
#Gettheboundariesofthepurplepixels
min_x,min_y=np.min(purple_coords,axis=0)
max_x,max_y=np.max(purple_coords,axis=0)
#Filltheareaenclosedbythepurpleboundary
forxinrange(min_x,max_x+1):
foryinrange(min_y,max_y+1):
#Checkifthecurrentpositionisoutsidethepurple
boundary
if(x==min_xorx==max_xory==min_yory==max_y)and
output_grid[x,y]==
Color.BLACK:
output_grid[x,y]=Color.BLUE
returnoutput_grid
```

## C.6 ARC PROBLEM C074846D 

```
fromcommonimport*
```

```
importnumpyasnp
fromtypingimport*
```

```
#concepts:
```

38 

Preprint. 

```
#rotation,colorchange,symmetry
```

```
#description:
#Intheinput,youwillseeacoloredobjectwithasinglegraypixel.
#Tomaketheoutput,rotatetheobject90degreesclockwisearoundthe
graypixel,
#andcolorthenewlyexposedpixelsgreen.
deftransform(input_grid):
#Findthegraypixellocation
gray_pixel_locations=np.argwhere(input_grid==Color.GRAY)
assertlen(gray_pixel_locations)==1
gray_x,gray_y=gray_pixel_locations[0]
#Createanoutputgrid
output_grid=np.full(input_grid.shape,Color.BLACK)
```

```
#Rotatetheobjectaroundthegraypixel
forxinrange(input_grid.shape[0]):
foryinrange(input_grid.shape[1]):
ifinput_grid[x,y]!=Color.BLACKandinput_grid[x,y]!=
Color.GRAY:
#Calculatenewpositionafter90degreesclockwise
rotation
new_x=gray_x+(y-gray_y)
new_y=gray_y-(x-gray_x)
#Checkifthenewpositioniswithinbounds
if0<=new_x<output_grid.shape[0]and0<=new_y<
output_grid.shape[1]
:
#Placetherotatedpixelintheoutputgrid
output_grid[new_x,new_y]=input_grid[x,y]
```

```
#Colornewlyexposedpixelsgreen
ifoutput_grid[x,y]==Color.BLACK:
output_grid[x,y]=Color.GREEN
```

```
#Placethegraypixelbackinthecenter
output_grid[gray_x,gray_y]=Color.GRAY
```

```
returnoutput_grid
```

## C.7 ARC PROBLEM AE58858E 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
```

```
#concepts:
```

```
#objectdetection,colorchange,sizecomparison
```

```
#description:
```

```
#Intheinput,youwillseeagridwithredobjectsofvarioussizes.
#Tomaketheoutput,changeallobjectslargerthanacertainsize(3
pixels)topink.
```

- **`def`** `transform(input_grid):` 

```
#Createacopyoftheinputgridtoproducetheoutput
output_grid=np.copy(input_grid)
```

```
#Findallconnectedcomponents(redobjects)intheinputgrid
```

39 

Preprint. 

```
red_objects=find_connected_components(input_grid,background=Color.
BLACK,monochromatic=True)
```

```
#Iteratethroughtheredobjectsandchangetheircoloriftheyare
largerthan3pixels
forobjinred_objects:
size=np.sum(obj==Color.RED)
ifsize>3:
output_grid[obj==Color.RED]=Color.PINK
returnoutput_grid
```

## C.8 ARC PROBLEM 21F83797 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
#concepts:
#verticalandhorizontallines,intersection
#description:
#Intheinput,youwillseetworedpixels.
#Tomaketheoutput,drawabluesquarewiththeredpixelsascorners.
#Additionally,drawaverticalandhorizontallinethatintersectat
eachredpixel.
deftransform(input_grid):
#Copytheinputgridtotheoutputgrid
output_grid=np.copy(input_grid)
#Findthepositionsoftheredpixels
red_pixels=np.argwhere(input_grid==Color.RED)
#Ensurethereareexactlytworedpixels
assertlen(red_pixels)==2
(x1,y1),(x2,y2)=red_pixels
#Sorttheredpixels’positionstodeterminewhichistop-leftand
bottom-right
top_left=(min(x1,x2),min(y1,y2))
bottom_right=(max(x1,x2),max(y1,y2))
#Drawabluesquarefromthetop-lefttothebottom-rightcorner
output_grid[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1
]+1]=Color.BLUE
#Drawhorizontalandverticallinesthroughtheredpixels
forx,yin[(x1,y1),(x2,y2)]:
output_grid[x,:]=Color.RED#Horizontallinethroughred
pixels
output_grid[:,y]=Color.RED#Verticallinethroughredpixels
returnoutput_grid
```

## C.9 ARC PROBLEM 33B52DE3 

```
fromcommonimport*
```

```
importnumpyasnp
fromtypingimport*
```

40 

Preprint. 

```
#concepts:
#objectdetection,colormapping,gridtransformation
```

```
#description:
#Intheinput,youwillseea20x20gridcontaininga4x4patternmade
ofcoloredpixels(notnecessarily
contiguous)andaseriesof3x3gray
squares.
#Thetaskistoextractthecolorsfromthe4x4patternandcolorthe
corresponding3x3graysquareswith
thosecolorsinaspecificorder.
#Thefirstgraysquareshouldbecoloredwiththefirstcolorfromthe
pattern,thesecondgraysquarewith
thesecondcolor,andsoon.
```

```
deftransform(input_grid):
#Step1:Detectthe4x4colorpatternintheinputgrid.
color_pattern=detect_objects(input_grid,monochromatic=False,
connectivity=8)
```

```
#Step2:Extractthecolorsfromthedetectedpattern.
colors=[]
forobjincolor_pattern:
cropped_obj=crop(grid=obj,background=Color.BLACK)
colors.extend(cropped_obj.flatten())
```

```
#Step3:Identifythelocationsofthegraysquares.
gray_squares=detect_objects(input_grid,colors=[Color.GRAY],
monochromatic=True,connectivity
=8)
```

```
#Step4:Colorthegraysquareswiththecorrespondingcolorsfrom
thepattern.
output_grid=np.copy(input_grid)
fori,gray_squareinenumerate(gray_squares):
ifi<len(colors):
#Colorthegraysquarewiththecorrespondingcolor
x,y=np.where(gray_square==Color.GRAY)
flood_fill(output_grid,x[0],y[0],colors[i])
```

```
returnoutput_grid
```

## C.10 ARC PROBLEM 332EFDB3 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
```

```
#concepts:
#borders,alternatingpatterns
```

```
#description:
```

```
#Intheinput,youwillseeanemptyblackgrid.Tomaketheoutput,
drawablueborderaroundthegrid,
butonlyontheedgesthatare
multiplesof2(i.e.,thetopand
bottomedges,andtheleftandright
edgesateveryevencolumn/index).
```

```
deftransform(input_grid):
```

```
#Getthedimensionsoftheinputgrid
```

41 

Preprint. 

```
height,width=input_grid.shape
```

```
#Createtheoutputgrid,initiallyacopyoftheinputgrid
output_grid=input_grid.copy()
#Drawtheblueborderonedgesthataremultiplesof2
forxinrange(0,height,2):#Verticaledges:multiplesof2
draw_line(output_grid,x,0,direction=(0,1),color=Color.BLUE)
#Topedge
draw_line(output_grid,x,width-1,direction=(0,1),color=
Color.BLUE)#Bottomedge
foryinrange(0,width,2):#Horizontaledges:multiplesof2
draw_line(output_grid,0,y,direction=(1,0),color=Color.BLUE)
#Leftedge
draw_line(output_grid,height-1,y,direction=(1,0),color=
Color.BLUE)#Rightedge
```

```
returnoutput_grid
```

## D INDUCTION FALSE POSITIVE SOLUTIONS 

## D.1 ARC PROBLEM 25094A63 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
#concepts:
#gridtransformation,colorreplacement,patterndetection
#description:
#Intheinput,youwillseeagridwithcoloredregions.Ifa5x5square
inthegridconsistsentirelyofa
singlecolor,
#replacethatsquarewitha5x5yellowsquareintheoutputgrid.Ifthe
squareisnotfullyfilledwitha
singlecolor,leaveitasis.
deftransform(input_grid):
#Plan:
#1.Iterateoverthegridtofind5x5squares.
#2.Foreach5x5square,checkifitismonochromatic(allpixels
arethesamecolor).
#3.Ifitis,replaceitwitha5x5yellowsquareintheoutputgrid
.
```

```
output_grid=np.copy(input_grid)
```

```
#Dimensions
square_size=5
forxinrange(len(input_grid)-square_size+1):
foryinrange(len(input_grid[0])-square_size+1):
#Extractthe5x5region
region=input_grid[x:x+square_size,y:y+square_size]
#Checkiftheregionismonochromatic
ifnp.all(region==region[0,0]):#Allpixelsshouldbe
thesamecolor
#Replacetheregionwithayellowsquare
output_grid[x:x+square_size,y:y+square_size]=Color
.YELLOW
```

42 

Preprint. 

```
returnoutput_grid
```

## D.2 ARC PROBLEM 009D5C81 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
#concepts:
#patternrecognition,colortransformation,gridmanipulation
#description:
#Intheinput,youwillseea15x15gridwithapatternofpurplepixels
thatformsashapeandablue
circleinthecenterofthegrid.
#Tocreatetheoutput,youshouldcolortheshapeformedbythepurple
pixelswithaspecificcolorbased
onthenumberofpurplepixels:
#-Ifthereare1-10purplepixels,colortheshapeyellow.
#-Ifthereare11-20purplepixels,colortheshapegreen.
#-Ifthereare21-30purplepixels,colortheshapered.
#-Iftherearemorethan30purplepixels,colortheshapeorange.
#Thebluecircleshouldbereplacedbyblack.
deftransform(input_grid):
#Createanoutputgridinitializedtoblack
output_grid=np.zeros_like(input_grid)
#Countthenumberofpurplepixels
purple_count=np.count_nonzero(input_grid==Color.PURPLE)
#Determinethecolorbasedonthenumberofpurplepixels
if1<=purple_count<=10:
shape_color=Color.YELLOW
elif11<=purple_count<=20:
shape_color=Color.GREEN
elif21<=purple_count<=30:
shape_color=Color.RED
else:
shape_color=Color.ORANGE
#Findthepurpleshapeandcoloritaccordingly
output_grid[input_grid==Color.PURPLE]=shape_color
#Replacethebluecirclewithblack
output_grid[input_grid==Color.BLUE]=Color.BLACK
```

```
returnoutput_grid
```

## D.3 ARC PROBLEM E95E3D8E 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
```

```
#concepts:
#occlusion,symmetrydetection,reconstruction
```

```
#description:
```

43 

Preprint. 

```
#Intheinputyouwillseeagridcontainingarepeatedpatternthathas
beenpartiallyoccludedbyblack
squares.
#Tomaketheoutput,removetheoccludingblacksquaresandrestorethe
patternbasedonthedetected
symmetry.
```

```
deftransform(input_grid):
#Plan:
#1.Extracttheoccludedpatternfromtheinputgrid.
#2.Detectthesymmetriesinthepattern.
#3.Fillinthemissingpartsofthepatternusingthedetected
symmetries.
#Step1:Identifynon-blackpixelstofindthepattern
occlusion_color=Color.BLACK
non_black_mask=input_grid!=occlusion_color
objects=detect_objects(input_grid,background=occlusion_color,
monochromatic=False,
connectivity=8)
```

```
#Step2:Findthesymmetriesinthedetectedobjects
symmetries=detect_translational_symmetry(input_grid,ignore_colors=
[occlusion_color])
#Step3:Fillinthemissingpartsusingthesymmetries
output_grid=input_grid.copy()
forx,yinnp.argwhere(non_black_mask):
forsymmetric_x,symmetric_yinorbit(input_grid,x,y,
symmetries):
```

```
ifoutput_grid[symmetric_x,symmetric_y]==occlusion_color:
output_grid[symmetric_x,symmetric_y]=input_grid[x,y]
```

```
returnoutput_grid
```

## D.4 ARC PROBLEM 73CCF9C2 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
```

```
#concepts:
```

```
#patternextraction,rotation,centering
```

```
#description:
#Intheinput,youwillseeagridwithapatternrepeatedseveraltimes
,butwithrandomrotations.
#Tomaketheoutput,extractoneinstanceofthepatternandrotateit
backtoitsoriginalorientation.
deftransform(input_grid:np.ndarray)->np.ndarray:
#Createacopyoftheinputgridfortheoutput
output_grid=np.copy(input_grid)
```

```
#Findconnectedcomponentstoextractpatterns
components=find_connected_components(output_grid,background=Color.
BLACK,connectivity=8)
```

```
#Foreachcomponent,checkifit’savalidpattern
forcomponentincomponents:
#Cropthecomponenttogetthepattern
pattern=crop(component,background=Color.BLACK)
```

44 

Preprint. 

```
#Checkifthepatterncanberotatedbacktoitsoriginal
orientation
#Wecanrotatethepatternandcomparewiththeoriginal
original_pattern=pattern.copy()
```

```
#Rotatethepatternin90-degreeincrementsandcheckifthe
originalpatternmatches
rotated_patterns=[np.rot90(original_pattern,k)forkinrange(
4)]
```

```
#Weassumethefirstvalidpatternistheoriginalorientation
forrotatedinrotated_patterns:
ifnp.array_equal(rotated,original_pattern):
```

```
#Ifitmatches,we’vefoundtheoriginalorientation
output_grid=pattern
break
```

```
returnoutput_grid
```

## D.5 ARC PROBLEM D56F2372 

```
fromcommonimport*
importnumpyasnp
fromtypingimport*
```

```
#concepts:
```

```
#objectdetection,symmetrydetection,cropping
```

```
#description:
```

```
#Intheinputyouwillseeagridfilledwithdifferentcoloredshapes.
#Theoutputshouldbethesmallestsymmetricshape(withrespecttothe
center)thatcanfitalltheothers
insideit.
#Iftherearenosymmetricshapes,theoutputshouldbeagridofthe
samesizefilledwiththebackground
color.
```

```
deftransform(input_grid:np.ndarray)->np.ndarray:
#Findallconnectedcomponents(shapes)inthegrid
components=find_connected_components(input_grid,background=Color.
BLACK,connectivity=8,
monochromatic=False)
#Initializeavariabletotrackthelargestsymmetricshapefound
largest_symmetric_shape=None
max_area=0
```

```
#Checkeachcomponentforsymmetryandarea
forcomponentincomponents:
#Cropthecomponenttogetitsshape
cropped_shape=crop(component,background=Color.BLACK)
#Checkforhorizontalandverticalmirrorsymmetry
is_symmetric=np.array_equal(cropped_shape,np.flip(
cropped_shape,axis=0))or\
np.array_equal(cropped_shape,np.flip(
cropped_shape
,axis=1))
```

```
ifis_symmetric:
```

```
area=np.count_nonzero(cropped_shape!=Color.BLACK)
ifarea>max_area:
max_area=area
```

45 

Preprint. 

```
largest_symmetric_shape=cropped_shape
```

```
#Ifwefoundasymmetricshape,returnit;otherwisereturnagrid
ofthesamesizefilledwiththe
backgroundcolor
iflargest_symmetric_shapeisnotNone:
returnlargest_symmetric_shape
else:
returnnp.full(input_grid.shape,Color.BLACK)
```

## E DATA AUGMENTATION AND RERANKING 

We improve the performance of the transductive model through data augmentation and output reranking. The key insight is that any invertible transformation _T_ can be applied to both training and test inputs, allowing us to generate multiple diverse predictions that are then aggregated and reranked. We consider two transformations of the training examples and test input in addition to the original task: (i) transpositions, _Tt_ ( _x_ ) = _x[T]_ ; (ii) color permutation, implemented as a random permutation of the integers 0-9 representing colors. For each transformation _T_ , we apply it to all training examples and the test input, run a beam size 10 decoding on the transformed data, and then apply _T[−]_[1] to the predictions to return to the original space. The transposition transformation addresses potential biases in the pretrained model regarding horizontal vs. vertical processing, while color permutations ensure the model isn’t relying on specific color values. For each transformation _T_ , each candidate output _y_ receives a beam search score _sT_ ( _y_ ) computed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0046-05.png)


where `t` _θ_ ( _·|·_ ) is the transduction model. For ranking, we aggregate candidates across all transformations. For each unique candidate _y_ , we track both its frequency of appearance freq( _y_ ) across different transformations and its average beam search score E [ _sT_ ( _y_ )]. These are then ranked with frequency taking precedence over average score. 

## F TEST-TIME TRAINING 

Test time training is an approach for updating model parameters at test time, which we apply to our transduction model. We assume that we are given test problems _D_ comprising triples ( _**x**_ train _,_ _**y**_ train _, x_ test) and a data augmentation procedure AUG( _**x** ,_ _**y**_ ) which constructs variations of an ARC problem, for example by permuting colors and rotating grids. Then model parameters are optimized to maximize the likelihood of augmented test tasks where a random training input-output is selected to serve as a fake test example: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning_images/iclr_2025_Combining_Induction_and_Transduction_for_Abstract_Reasoning.pdf-0046-09.png)


where len( _**x**_ train) is the number of example training inputs, and _k_ is the index of the training inputoutput which is randomly selected to serve as a fake testcase. Note that this procedure does not rely on access to ground truth test predictions. 

We choose each training example as a fake test example and do 10 times different randomly combined data augmentation for each fake task, which gives us 12k pseudo training dataset from the evaluation dataset. We also randomly include 5k RE-ARC examples and 5k ARC-Heavy examples, which we speculated would have a regularizing effect. The dataset size for test time training is 22k total problems. 

## G EXPERIMENT PARAMETERS 

G.1 INDUCTION 

Fine-tuning Hyperparameters: 

46 

Preprint. 

|trainingtype|lora rank|lora alpha|learningrate|gradient accumulate steps|
|---|---|---|---|---|
|lora fnetune|64|64|2e-4|2|
|per device batch size|device|epcoh|weight decay|learningrate scheduler type|
|8|8xA100|3|0|cosine|



For the last 230k data finetune we used full finetuning instead of LoRA: 

|trainingtype|learningrate|weight decay|gradient accumulate steps|
|---|---|---|---|
|full fnetune|1e-5|0.05|1|
|epoch|per device train batch|learningrate scheduler type|devices|
|2|16|cosine|8xA100|



Inference Hyperparameters: 

- temperature: 0.8 (1.0 for the full-data fine-tuned model) 

- top-p: 1.0 

Output selection: For experiments in section 4, when allowing one or two attempts, we filter the sample programs using train input-output examples and then randomly select one or two distinct programs uniformly. We report the expected value in our results. For experiments in section 5, we take the execution results of test output from the programs that can pass all the train examples and use majority vote to select the top 2, and in the case of concept arc, the top 3 test outputs. 

## G.2 TRANSDUCTION 

Fine-tune Hyperparameters: 

|trainingtype|learningrate|weight decay|gradient accumulate steps|device|
|---|---|---|---|---|
|full fnetune|1e-5|1e-2|2|8xA100|
|engineer epoch|other epoch|per device train batch|learningrate scheduler type||
|3|2|8|cosine||



For the final engineering results, we train for 3 epochs. For all other experiment results, we train for 2 epochs. 

Inference Hyperparameters: 

- temperature: 0 

- use beam search: True 

- beam width: 

   1. engineer results: 40 

   2. 100k data scale: 20 

   3. all other experiment results: 3 

- top-p: 1.0 

Test-time Fine-tuning Hyperparameters 

|trainingtype|lora rank|lora alpha|learningrate|gradient accumulate steps|
|---|---|---|---|---|
|lora fnetune|64|64|2e-4|2|
|per device batch size|device|epcoh|weight decay|learningrate scheduler type|
|2|4xA6000|3|0|cosine|



47 

