Published as a conference paper at ICLR 2025 

## ME TABENCH 

## - A SPARSE BENCHMARK OF REASONING AND KNOWL EDGE IN LARGE LANGUAGE MODELS 

**Alex Kipnis**[1] _[∗]_ **Konstantinos Voudouris**[1] _[,]_[2] **Luca M. Schulze Buschoff**[1] **Eric Schulz**[1] 1 Human-Centered AI, Helmholtz Munich 2 University of Cambridge 

## ABSTRACT 

Large Language Models (LLMs) vary in their abilities on a range of tasks. Initiatives such as the Open LLM Leaderboard aim to quantify these differences with several large benchmarks (sets of test items to which an LLM can respond either correctly or incorrectly). However, high correlations within and between benchmark scores suggest that (1) there exists a small set of common underlying abilities that these benchmarks measure, and (2) items tap into redundant information and the benchmarks may thus be considerably compressed. We use data from _n >_ 5000 LLMs to identify the most informative items of six benchmarks, ARC, GSM8K, HellaSwag, MMLU, TruthfulQA and WinoGrande (with _d_ = 28 _,_ 632 items in total). From them we distill a sparse benchmark, metabench, that has less than 3% of the original size of all six benchmarks combined. This new sparse benchmark goes beyond point scores by yielding estimators of the underlying benchmark-specific abilities. We show that these estimators (1) can be used to reconstruct each original _individual_ benchmark score with, on average, 1 _._ 24% root mean square error (RMSE), (2) reconstruct the original _total_ score with 0 _._ 58% RMSE, and (3) have a single underlying common factor whose Spearman correlation with the total score is _r_ = 0 _._ 94. 

## 1 INTRODUCTION 

Large Language Models (LLMs) represent a significant leap forward in our quest to emulate humanlike understanding and generation of language (Brown et al., 2020). These sophisticated algorithms are designed to process, interpret, and produce language with unprecedented accuracy and fluency (Bubeck et al., 2023). By training on vast datasets encompassing a myriad of linguistic patterns, LLMs have demonstrated remarkable capabilities in tasks ranging from simple text completion (Radford et al., 2019) to complex problem-solving (Chen et al., 2021a) and creative writing (Elkins & Chun, 2020). Their potential to transform various fields—education (Kasneci et al., 2023), customer service (Pandya & Holia, 2023), content creation (Zhang et al., 2023), and beyond—has captured the imagination of researchers and practitioners alike (Bommasani et al., 2021; Bubeck et al., 2023). However, as these models grow in complexity, so does the challenge of effectively evaluating their diverse abilities (Burnell et al., 2023; Ke et al., 2023). 

Traditionally, research has relied on extensive benchmarks—vast arrays of tasks—to assess the abilities of LLMs (Srivastava et al., 2023; Raji et al., 2021). However, these benchmarks often measure overlapping skills, leading to redundancy and inefficiency. This paper introduces metabench, a sparse benchmark engineered to distill the essence of six prominent benchmarks—ARC (Clark et al., 2018), GSM8K (Cobbe et al., 2021), HellaSwag (Zellers et al., 2019), MMLU (Hendrycks et al., 2021), TruthfulQA (Lin et al., 2022), and WinoGrande (Sakaguchi et al., 2020)—into a more concise and informative benchmark. By analyzing data from over 5000 LLMs, we identified the most revealing items within these benchmarks, compressing them to less than 3% of their combined size. We also provide adaptive testing simulations that suggest that the number of items administered to LLMs can be lowered even further. This volume reduction greatly enhances evaluation efficiency, while also maintaining the original benchmarks’ ability to measure underlying abilities. 

> _∗_ Correspondence to adkipnis@mailbox.org 

1 

Published as a conference paper at ICLR 2025 

The metabench framework leverages psychometric techniques (Hambleton & Swaminathan, 2013; Wang et al., 2023a) to estimate the _latent abilities_ captured by these benchmarks (Burnell et al., 2023). These estimates can reconstruct individual and total benchmark scores with minimal error, and reveal a single underlying factor that captures most of their variability. By refining how we measure the abilities of LLMs, metabench offers a new lens through which we can understand and improve them. Practitioners can run their own models on this efficient and highly informative benchmark, obtain approximated scores, and compute an overall estimate of their models’ general ability. We additionally offer a second, disjoint version of metabench for users that intend to evaluate an LLM multiple times. Our work underscores the potential for more streamlined and psychologically meaningful evaluation methods (Coda-Forno et al., 2024; Huang et al., 2023). It is aimed for the opensource community and independent machine learning researchers with limited resources. All reported analyses and and results can be viewed on www.github.com/adkipnis/metabench and our benchmark itself is available on www.huggingface.co/datasets/HCAI/metabench. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0002-02.png)


Figure 1: _Score Reconstruction_ . (A) After distilling all six benchmarks to in total 858 items, the latent abilities estimated by the reduced set can reconstruct each normalized benchmark score with, on average, 1 _._ 24% root mean squared error (RMSE) on the test split. (B) Empirical test RMSE distributions for 10 _,_ 000 randomly subsampled benchmarks of matching sizes. Colors match benchmarks in the other panels, and the asterisks are their corresponding test RMSEs. Our method outperforms every best randomly sampled subset by a significant margin. (C) The test RMSE is further reduced to 0 _._ 58% when the mean normalized score across all benchmarks is predicted. 

## 1.1 RELATED WORK 

Recent years have seen significant advancements in the development and evaluation of LLMs, resulting in an increased need for efficient benchmarking methods due to the expense of running and querying them. Traditional benchmarks, such as those used in the Open LLM Leaderboard (Beeching et al., 2023), encompass a wide range of tasks. In the current work, we focus on six different benchmarks: ARC (AI2 Reasoning Challenge, Clark et al., 2018) was designed to test the reasoning abilities of AI through a series of challenging science questions. Similarly, GSM8K (Grade School Math, Cobbe et al., 2021) evaluates the mathematical reasoning capabilities of LLMs by training verifiers on complex word problems. HellaSwag (Zellers et al., 2019) tests commonsense reasoning by asking models to choose the most plausible continuation of a story from several options. MMLU (Massive Multitask Language Understanding, Hendrycks et al., 2021) assesses the breadth of expert knowledge and reasoning skills across various disciplines, presenting a comprehensive challenge for LLMs. TruthfulQA (Lin et al., 2022) aims to measure the tendency of language models to generate truthful responses, tackling the issue of misinformation and model-generated falsehoods. Finally, WinoGrande (Sakaguchi et al., 2020) tests models on pronoun resolution, a 

2 

Published as a conference paper at ICLR 2025 

task that requires understanding context and commonsense reasoning. We use these benchmarks to distill an informative, sparse set of items to form metabench. 

Psychometric methods including Item Response Theory (IRT) are increasingly used in the evaluation of machine learning models (Martínez-Plumed et al., 2016; Martínez-Plumed et al., 2019; Wang et al., 2023a). Zhuang and colleagues (Zhuang et al., 2023) propose using IRT for adaptive testing of LLMs but only used this approach to distinguish between a small set of models and restricted domains and compared models to human responses. Burden and colleagues (Burden et al., 2023) use Bayesian probabilistic programming to infer agents’ capabilities from diverse task performances, building computational models to link task features to system capabilities. However, these methods are applied to reinforcement learning agents and not to LLMs. There also have been several attempts to create completely new questionnaires to evaluate LLMs using psychometric tools (Jiang et al., 2024; Wang et al., 2023b). 

Redundancy in benchmark items has been established, analyzed and combated by several recent works (Rodriguez et al., 2021; Vania et al., 2021; Wang et al., 2023c; Ye et al., 2023; Hewitt et al., 2024). Li et al. (2024) introduce the BenchBuilder pipeline, which filters existing benchmarks to preserve high-quality items. It is used to create Arena-Auto-Hard, a 500-item benchmark which can robustly recover the LLM rankings on both the source benchmark (Chatbot Arena, Chiang et al., 2024) and two other large benchmarks (MT-Bench, Zheng et al., 2023; AlpacaEval 2.0 LC, Taori et al., 2023). BenchBuilder uses LLM annotators to rate benchmark items according to seven criteria to filter out uninformative items. An LLM judge then evaluates whether an LLM subject correctly completes the task specified by the item. Our approach filters prompts in a datadriven manner by first sub-sampling benchmark items and then characterizing item information using IRT. We only use the distribution of responses to benchmarks across a population of LLMs for item filtering, rather than using _a priori_ qualities and LLM annotators which could introduce bias. Similarly to BenchBuilder, our approach is extendable to new benchmarks, provided that they contain a large number of items and that a population of LLMs have completed them. 

Most related to our proposal is recent work by Polo et al. (2024) introducing tinyBenchmarks. The authors advocate for reducing the number of evaluation examples in benchmarks to lower costs while maintaining accurate performance assessments of LLMs. They demonstrate that evaluating LLMs on small curated subsets of benchmarks can reliably estimate the performance on large benchmarks like MMLU and HELM (Liang et al., 2022) within a 2% mean absolute error. However, their proposal differs from our approach on several dimensions: 

1. **Model Fit** : Polo and colleagues analyzed only 395 LLMs with high-dimensional IRT models of up to 15-dimensional latent traits. This raises strong doubts about the goodness of fit for their IRT models, as the number of estimated parameters is far greater than the number of observations (DeMars, 2010; Svetina et al., 2017; Natesan et al., 2016). In comparison, we use over 5000 LLMs and select 1-dimensional IRT models based on cross-validation. 

2. **Item Selection** : The authors base item selection on clustering of the estimated item parameters, which can lead to a bad coverage of the ability landscape. In comparison, we use an information filtering approach which explicitly aims to cover the entire ability landscape. 

3. **Use of Latent Abilities** : We go beyond point scores by estimating a latent ability per benchmark and, importantly, show that they _jointly_ permit better score reconstruction for each benchmark’s score and the grand mean. Furthermore, we analyze their latent structure using Factor Analysis and show that they are largely governed by a single general ability. 

4. **Performance** : We halve the score reconstruction error reported in tinyBenchmarks: For MMLU resp. Open LLM Leaderboard we reach a mean absolute error of 0 _._ 969% resp. 0 _._ 437%.[1] 

> 1Note that we report root mean squared errors (RMSEs), which are more sensitive to outliers than the mean absolute errors (MAEs) used in tinyBenchmarks. For comparability, we also show MAEs in the appendix Appendix B.10. 

3 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0004-01.png)


Figure 2: _Processing pipeline_ . Explained in Section 2. 

## 2 BENCHMARK DISTILLATION 

We collect benchmark data for over 5000 LLMs from the Open LLM Leaderboard, and discard items and models based on simple criteria (Section 2.1), reduce each benchmark to a fixed size using cross-validated subsampling (Section 2.2), fit IRT models to quantify the properties of each remaining item (Section 2.3), and finally, we filter out items based on their information content (Section 2.4). We focus on reconstructing the original scores per benchmark using the reduced sets in a cross-validated manner (Sections 2.3 - 2.4). Using Factor Analysis, we uncover strong overlap in the latent abilities measured by each benchmark, and leverage this synergy by combining all benchmarks for score reconstruction (Section 2.5). We also provide preliminary results in Computerized Adaptive Testing (Appendix C.1). For practitioners interested in repeated evaluation, we have created a disjoint second version of metabench that effectively combats over-adaptation to our benchmark (Section 3, see Appendix B.9 for details). 

## 2.1 DATA COLLECTION AND PREPROCESSING 

We collected openly available item-wise accuracies from Hugging Face Datasets for the six benchmarks that are part of the Open LLM Leaderboard (Beeching et al., 2023): ARC, GSM8K, HellaSwag, MMLU, TruthfulQA and WinoGrande. After excluding flagged and merged models, we obtained data from 1355 unique users, yielding _n_ = 6875 unique LLMs and over 152 million responses. Out of all LLMs, 5055 have been evaluated on all six benchmarks. Whenever possible, we used all available LLMs per benchmark for the analyses. 

All analyses were performed in R 4.4.0 (R Core Team, 2024). For comparability across benchmarks, we normalized the benchmark scores on a percent scale.Per benchmark, we discarded subjects[2] with the lowest 0 _._ 1% of scores to further reduce noise[3] . Items with standard deviations below 1% (too little variability) and mean accuracies above 95% (too easy) were removed. We then estimated the part-whole correlation between each item and the total score using the point-biserial correlation coefficient, _rpb_ . This coefficient grows if subjects who responded correctly to the item have a higher than average score, especially if the item has medium difficulty (Irwing et al., 2018, see Appendix A.1 for details). We removed items with _rpbis ≈_ 0. The exact number of items per benchmark removed during preprocessing can be found in Appendix A.2. 

> 2For the remainder of this paper, we use the terms "subjects" and "LLMs" interchangeably, since many of the methods below were designed to analyze human behavior and abilities. 

> 3Note that IRT models are sigmoidal functions of the latent ability - mathematically, the estimation problem is under-constrained for extreme scores and it is common practice to exclude them for stability reasons (Baker & Kim, 2004; Irwing et al., 2018). 

4 

Published as a conference paper at ICLR 2025 

For comparability across benchmarks, we conducted a train-test-validation split of the LLMs using the caret package (Kuhn & Max, 2008) in the following manner: For stratification, we calculated the grand average of the original benchmark scores for LLMs that ran on _all six_ benchmarks, and used it to create a stratified 10% split as the global test set. For cross-validation per benchmark, we split off further 10% subset of the LLMs as a local validation set, this time stratifying by the specific benchmark score. 

## 2.2 CROSS-VALIDATED SUBSAMPLING 

For a set of _d_ items it becomes quickly impossible to consider all subsets of size _k_ , as there are � _kd_ � unique subsets and any exhaustive search algorithm would have _O_ ( _d_ !) time complexity for fixed _k_ . The computational infeasibility of finding 100 items out of the given 28632 motivates the approximate methods we present here. Despite the impossibility of exhaustive search, it is astounding how well the given benchmarks can be reduced in a theory-free way using random sampling: Wang et al. (2023c), Ye et al. (2023) and Polo et al. (2024) showed that random samples of benchmark items can in principle produce smaller benchmarks that retain enough information to reconstruct their original score with negligible error. One basic procedure for this is: 

1. Uniformly sample _k_ items and calculate the subtest scores ˜ _sj_ for each subject _j_ . 

2. Fit some regression model ˆ _sj_ = _g_ (˜ _sj_ ) on the training set. 

- ˆ 

- 3. Calculate some summary statistic of _εj_ = _sj − sj_ on the validation set, e.g. its root mean square error (RMSE). 

4. Repeat steps (1-3) a large number of times. 

5. Choose the sample with the smallest validation error and check if the performance is still good on the test set. 

For _g_ ( _·_ ) one can use generalized additive models (GAMs, Hastie & Tibshirani, 1987; Fahrmeir et al., 2013; Wood, 2017), which are a flexible class of regression models utilizing basis functions. They can be seen as the frequentist version of the fully Bayesian framework of Gaussian Process regression (Duvenaud et al., 2011). 

We ran the described procedure on each benchmark with varying _k_ and 10 _,_ 000 repetitions to gauge how information is spread across items. GAMs were fitted using the mgcv package (Wood, 2017) with default options. Appendix B.2 contains the results for _k_ between 50 and 500 in steps of 50. As shown in Figure 1B, even for _k ≤_ 200, test errors have high density on relatively small intervals and all median RMSEs are below 3% - many different (but potentially overlapping samples) are informative. This suggests (1) few items are _un_ informative, (2) the information they carry is partially redundant, and (3) the useful information saturates swiftly as more items are considered. This notion is further explored in Section 2.4. This result has two practical implications: If we just want mediumsized versions of the six benchmarks, we can simply use cross-validated subsampling. If we want to tap into (or even breach) the boundary of the test error distribution, a more principled approach that models properties of items is required. However, principled item selection can be a follow-up _after_ an initial subsampling sweep, as this preselection still permits reliable score reconstruction. For comparability across item property models, and to increase the ratio between the number of observations and estimated parameters, we subsampled all six benchmarks to 350 items. This number is a compromise between predictive performance and subject-to-item ratio (see Appendix B.2). We next describe the approach to modeling item properties. 

## 2.3 ITEM RESPONSE THEORY 

Item response theory (IRT) is a statistical branch of psychometrics used for the design and analysis of standardized tests (Chen et al., 2021b; Cai et al., 2016; Irwing et al., 2018; Hambleton et al., 1991). It aims to model the relationship between a subject’s abilities and the probability of a correct response to an item. This framework is closely related to other latent trait models such as factor analysis (Kim & Mueller, 1978; Koch, 2013; Murphy, 2022), but assumes binary or polytomous responses (Chen & Zhang, 2020; Zanon et al., 2016). Let _xij_ be the binary accuracy of item _i_ for subject _j_ with some (unknown) latent ability vector _**ϑ** j_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0005-13.png)


where _f_ is a link function and _**α** i_ are the item parameters. We use one-dimensional _ϑj_ , as higherdimensional IRT models require much larger subjects-to-items ratios for reliable parameter estimation 

5 

Published as a conference paper at ICLR 2025 

(DeMars, 2010; Reckase, 2009; Svetina et al., 2017; also see Appendix B.7 for a comparison with two-dimensional models). A simple and common IRT model is the 2-parameter logistic (2PL) model with _**α** i_ = ( _δi,_ **a** _i_ ) and _f_ ( _**ϑ** j|_ _**α** i_ ) = _σ_ ( **a** _[⊤] i_ _**[ϑ]**[j][−][δ][i]_[)][.][Here] _[σ]_[(] _[·]_[)][is][the][logistic][function,] **[a]** _[i]_[is][the] loading parameter vector and _δi_ is the difficulty parameter. Note that this has the same form as logistic regression with the subtle but important difference that _**ϑ** i_ is unknown. The main goal is to estimate item parameters **A** = ( _**α** i_ ) _i_ =1 _,...,d_ which capture their test-relevant properties. Due to favorable asymptotic behavior, the standard approach is to use marginal maximum likelihood methods (Wirth & Edwards, 2007; Chen & Zhang, 2020), which treat the latent ability as a random variable and attempt to maximize the marginal likelihood 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0006-02.png)


with _Li_ ( _xij|_ _**ϑ** j_ ) = _f_ ( _**ϑ** j |_ _**α** i_ ) _[x][ij]_[�] 1 _−f_ ( _**ϑ** j |_ _**α** i_ )�1 _−xij_ . Because of the integral, there is no analytical solution to this problem, but it can be approximated using Expectation-Maximization (Dempster et al., 1977; Bock & Aitkin, 1981; Bock & Schilling, 1997) and stochastic approximations exist for multidimensional latent abilities (Zhang & Chen, 2022; Zhang et al., 2020). After model fitting, _**ϑ** j_ can be estimated for each subject _j_ using, e.g., maximum a posteriori estimation[4] : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0006-04.png)


We used the mirt package (Chalmers, 2012) with default options to fit the following IRT models to preselected subsets of 350 items each — a 2PL model, a 3PL model with an additional lower asymptote parameter, a 4PL model with an additional upper asymptote parameter. All three models always estimate a difficulty and a loading parameter. After IRT model fitting, we separately used the _maximum a posteriori_ (MAP) method as well as the _expected a posteriori sum_ (EAPsum) method to estimate each benchmark-specific one-dimensional latent ability (Bock & Aitkin, 1981; Thissen et al., 1995). We trained a GAM with adaptive smoothers to predict the original benchmark score _sj_ using (1) the latent ability estimate _ϑ_[ˆ] _j_ and (2) the subtest scores ˜ _sj_ , and tested the GAM on the test set. We use GAMs instead of linear regression models because the relationship between the latent ability estimates and point scores is monotonic but non-linear. Note, that we also use GAMs in cross-validated subsampling to ensure comparability of the test RMSEs. 

Table 1 shows the test RMSE of the predicted scores as well as the Spearman correlations between the predicted and the true point scores. All correlation coefficients are above 0 _._ 98, indicating that the ranking of LLMs is largely preserved. The best models achieve an RMSE of on average 1 _._ 0% and all have an RMSE well below 1 _._ 5%. Notably, each model outperforms the cross-validated subsampling approach. Performance differences between IRT models are minute, and MAP estimates all outperformed EAPsum estimates, when the subtest scores are provided to GAMs in conjunction with latent abilities. This shows that the latent abilities contain unique information about the full scores not captured by the subtest scores. The result suggests that further item selection can be based on IRT model fits. 

> 4Any likelihood-based estimator profits from large _d_ , as the link functions have sigmoidal shapes: A hill in the function product only arises after aggregating a sufficient number of item-likelihoods with varying midpoint and steepness. This raises the trade-off that _many subjects per item_ are required to get stable parameter estimates, but _many items_ per subject are required to recover their latent ability. 

6 

Published as a conference paper at ICLR 2025 

Table 1: Cross-validated score prediction results for IRT models using 350 preselected items per benchmark. Correlations are between the predicted and true normalized original scores and use the Spearman rank method. The RMSE of the most accurate model for each benchmark is highlighted in bold. Latent abilities are estimated using the MAP method. The best test RMSE observed during cross-validated subsampling is shown for reference (note that the sets were selected by validation RMSE, not test RMSE). 

|2PL<br>3PL<br>4PL<br>Best Random|2PL<br>3PL<br>4PL<br>Best Random|
|---|---|
|RMSE<br>r<br>RMSE<br>r<br>RMSE<br>r|RMSE<br>Surpassed|
|||
|ARC<br>**0.893**<br>0.993<br>0.897<br>0.994<br>0.893<br>0.994<br>GSM8K<br>**1.420**<br>0.995<br>1.420<br>0.995<br>1.420<br>0.995<br>HellaSwag<br>0.845<br>0.986<br>**0.820**<br>0.987<br>0.879<br>0.986<br>MMLU<br>**1.040**<br>0.967<br>1.040<br>0.965<br>1.070<br>0.961<br>TruthfulQA<br>0.969<br>0.995<br>**0.966**<br>0.995<br>0.969<br>0.995<br>WinoGrande<br>1.060<br>0.981<br>1.030<br>0.982<br>**1.020**<br>0.981|1.184<br>✓<br>1.447<br>✓<br>1.681<br>✓<br>1.202<br>✓<br>1.050<br>✓<br>1.246<br>✓|



## 2.4 INFORMATION FILTERING 

Item parameters have a natural interpretation because of how they affect the link function _f_ ( _ϑj|_ _**α** i_ ): The difficulty _δi_ shifts its midpoint and the loading _ai_ alters its scale. Thus, the difficulty parameter determines the probability of a correct response for a subject _j_ with _ϑj_ = 0, and the loading determines how much this probability changes with the subject’s ability. An item with a high loading discriminates well between subjects whose ability is close to the item’s difficulty. Following Reckase (2009), an item is more informative if its response is more useful for differentiating between similar values of a subject’s estimated latent ability, _ϑ_[ˆ] . Fisher information functions quantify this property: Since the distribution of _xij|ϑj_ is an exponential family, any response-based estimator _T_ for _ϑ_ will have a variance bounded from below by the expected Fisher information[5] (Pawitan, 2013; Lehmann & Casella, 1998). For the 2PL model with one-dimensional ability, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0007-05.png)


On a high level, this is a measure of how informative an item is about a test taker’s ability _conditioned on the ability_ , and opens up a more principled approach to item selection. For each benchmark we use the following procedure to pick the most predictive items: 

1. Estimate the item parameters using an IRT model _µ ∈{_ 2PL, 3PL, 4PL _}_ . 

2. Estimate the ability using a method _ν ∈{_ MAP, EAPsum _}_ . 

3. Use the item parameters to calculate Fisher information functions for all items. 

4. Partition the the support of _ϑ_ into _q_ quantiles. 

5. In each quantile, select the item with the highest information (above a threshold _τ_ ). 

The same item may have the highest Fisher information in multiple quantiles - to avoid selecting an item multiple times in the same sample, we remove it from the search space after selection. 

We re-ran the procedure in Section 2.3, this time subsampling items using information filtering. We used the rBayesianOptimization package (Yan, 2024) with default settings to fine-tune the hyperparameters ( _µ_ , _ν_ , _τ_ ) on the validation set to minimize the term RMSE( _**ϑ** ,_ **s** ) + _λk_ , where _k_ is the size of the chosen subset and _λ_ is the regularization coefficient[6] . The selected subset was then evaluated on the test set. For each selection we re-ran the random subsampling procedure with the same size _k_ for comparison. Table 2 shows that using benchmark-specific information for prediction (marginal model), the mean test RMSE over benchmarks is 1 _._ 6%. All IRT models have an RMSE well below 1 _._ 8%. All benchmarks either reach (GSM8K) or breach (all others) the test accuracy boundary of random subsampling. 

> 5The expected _Fisher information_ for item _i_ , _Ii_ ( _ϑ_ ) = E _ϑ_ � _− ∂[∂]_[2][2] _ϑ_[log] _[ L]_[(] _[x][i][|][ϑ]_[)] �, is the expected curvature of the likelihood. Then, for unbiased _T_ , V _ϑ_ ( _T_ ) _≥ I_ (1 _ϑ_ )[.][If] _[ T]_[is biased, the numerator becomes][ (] _∂∂_[2] _ϑ_[E] _[ϑ]_[ [] _[T]_[])][2][.][For] larger _I_ ( _ϑ_ ) =[�] _[d] i_ =1 _[I][i]_[(] _[ϑ]_[)][, one can estimate] _[ ϑ]_[ more reliably from the observed responses.] 

> 6 We tested _λ ∈{_ 0 _._ 01 _,_ 0 _._ 005 _,_ 0 _._ 001 _}_ and found 0 _._ 005 to yield a good trade-off between size and accuracy. 

7 

Published as a conference paper at ICLR 2025 

Table 2: Cross-validated score prediction results for IRT models + information filtering. Column _d_ indicates the number of kept items. Italicized models estimate latent abilities using the EAPsum rather than MAP method. Normalized scores go from 0 to 100 and were either predicted with information from the single benchmark (marginal model) or from all benchmarks (joint model). % BTR (better than random) indicates how many of the 10 _,_ 000 _d_ -sized random subsamples from the respective benchmark had a larger test RMSE. If a model surpasses all RMSEs from random subsampling, its RMSE is highlighted in bold. 

|Properties<br>Marginal Model<br>Joint Model<br>Best Random<br>d<br>IRT<br>RMSE<br>% BTR<br>RMSE<br>% BTR<br>RMSE<br>Surpassed|Properties<br>Marginal Model<br>Joint Model<br>Best Random<br>d<br>IRT<br>RMSE<br>% BTR<br>RMSE<br>% BTR<br>RMSE<br>Surpassed|
|---|---|
|||
|ARC<br>145<br>4PL<br>**1.303**<br>100<br>**1.166**<br>100<br>GSM8K<br>237<br>_2PL_<br>1.778<br>98.7<br>**1.555**<br>100<br>HellaSwag<br>93<br>3PL<br>**1.542**<br>100<br>**0.999**<br>100<br>MMLU<br>96<br>3PL<br>**1.743**<br>100<br>**1.430**<br>100<br>TruthfulQA<br>154<br>3PL<br>**1.562**<br>100<br>**1.104**<br>100<br>WinoGrande<br>133<br>3PL<br>**1.656**<br>100<br>**1.195**<br>100<br>Open LLM LB<br>858<br>mixed<br>-<br>-<br>**0.582**<br>100|1.671<br>✓✓<br>1.677<br>**_≈_**✓<br>1.619<br>✓✓<br>1.908<br>✓✓<br>1.700<br>✓✓<br>1.887<br>✓✓<br>0.803<br>✔|



## 2.5 GOING META 

Although the previous section indicates that information filtering is in general effective, we can further improve score reconstruction if we do not limit ourselves to using information _specific_ to each benchmark. Benchmark scores are highly correlated (going from _r_ = 0 _._ 571 to _r_ = 0 _._ 956, see Appendix D.2), so each benchmark also carries some information about the others. Thus, if an LLM is run on several reduced benchmarks, the resulting scores may mutually fill in each other’s blanks. Simulations show that this is particularly the case for tests with moderate overlap between the required abilities (see Appendix B.6). We exploit synergy by jointly using information from all six benchmarks as GAM predictors. This _joint model_ is described in more detail in Appendix B.5. As always, IRT models, latent abilities and GAMs were estimated on the training set and the RMSEs reflect the score prediction accuracy on the test set. Figure 1 and the right side of Table 2 show the results: The mean RMSE has shrunk to 1 _._ 24% and is well below 1 _._ 6% in all cases. All models surpass the best random subsample by a substantial margin. This synergy effect underlines the advantage of running all components of metabench. 

We then applied this method to predict the grand average score: The mean across all normalized scores for the six benchmarks constituting the Open LLM Leaderboad. Random subsets are sampled from each benchmark using its corresponding _d_ (e.g. 93 items from HellaSwag), totalling 858 items. Figure 1C and the bottom row of Table 2 show the score reconstruction results. We were able to reconstruct the grand average with a Spearman correlation of _r_ = 0 _._ 998 and an RMSE of 0 _._ 582%, which surpassed all random subsamples by a substantial margin. Appendix B.11 shows that these results are stable over random seeds. 

Finally, we investigated the relationship between all latent abilities by using factor analysis (Kim & Mueller, 1978; Koch, 2013; Murphy, 2022). In a nutshell, factor analysis of a set of continuous variables attempts a low-rank approximation of their covariance matrix. As its latent factors have the same interpretation as the latent abilities in IRT, it is commonly applied to analyze the relationship between multiple tests (Irwing et al., 2018). See Appendix D.1 for a more detailed introduction. We applied factor analysis to the estimated latent abilities using the psych package (William Revelle, 2023) (for details, see Appendix D.2). We find that a single factor is sufficient to capture 79 _._ 3% of the total variability across latent abilities. This _meta-ability_ has a Pearson correlation of _r_ = 0 _._ 938 with the original grand average score. Its loadings range from 0 _._ 771 (GSM8K) to 0 _._ 966 (WinoGrande): Separately, all six latent abilities contribute a comparable amount to the meta-ability, but the added unique information decreases with each. This result is in line with the synergy effect as described above. It consolidates the utility of metabench for efficiently measuring the abilities of LLMs. 

8 

Published as a conference paper at ICLR 2025 

## 3 USING M E T A B E N C H 

## 3.1 SCORE RECONSTRUCTION 

Our benchmark is hosted as a dataset on HuggingFace and is integrated in EleutherAI’s _lm-evaluationharness_ (Gao et al., 2024), a unified framework for evaluating language models on a range of tasks. We provide two disjoint versions of metabench containing distinct items for all six benchmarks (see Appendix B.9). This way, metabench can be used for repeated evaluation. Additionally, each version comes with the option to randomly permute the presentation of answer labels[7] , further hampering test memorization. After the _lm-evaluation-harness_ stores the raw scores to all metabench items, the scores of the original fully-sized benchmarks can be predicted using reconstruct.R in our GitHub repo: The point predictions stem from the originally trained GAMs, and we construct non-parametric confidence intervals from the local test RMSE for similar predictions. 

## 3.2 MEMORIZATION MITIGATION 

The described four variants of metabench aim to mitigate the problem of contamination and benchmark memorization. A model trained on the standard version of the benchmark will not perform appreciably better on the other versions: We evaluated three LLMs on each version and then fine-tuned their low-rank adapters on one version with fixed choice ordering. Fine-tuning was done using unsloth on 4-bit quantized instruct models. We then re-evaluated each LLM on all four versions of metabench. As shown in Table 3, while fine-tuning substantially improves performance on the set that it was trained on, this performance boost is attenuated when answer choices are permuted, and performance does not substantially improve on the disjoint set. 

|Llama<br>Fine-Tuning|metabench-A<br>metabench-B<br>Fixed<br>Permuted<br>Fixed<br>Permuted|metabench-A<br>metabench-B<br>Fixed<br>Permuted<br>Fixed<br>Permuted|
|---|---|---|
||||
|3.2-1B<br>Before<br>∆After<br>3.2-3B<br>Before<br>∆After<br>3.1-8B<br>Before<br>∆After|35.2<br>34.8<br>**7.6**<br>1.3<br>43.4<br>44.0<br>**8.1**<br>3.9<br>54.0<br>54.2<br>**5.8**<br>2.9|34.5<br>35.8<br>2.0<br>0.3<br>47.0<br>48.1<br>3.3<br>3.0<br>59.0<br>59.0<br>0.4<br>-0.2|



Table 3: Raw normalized scores on versions A resp. B of metabench with fixed resp. permuted choices, before and after fine-tuning on version A with fixed choices. 

## 4 CONCLUSION 

The findings presented in this study underscore the viability and advantages of using a sparse benchmark, metabench, to measure the abilities of Large Language Models (LLMs). We successfully distill six prominent benchmarks — ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, and WinoGrande — into a much smaller but equally informative set of items. The use of latent abilities enables accurate reconstruction of individual and total benchmark scores, as well as the original ranking: metabench retains the evaluative power of the original, larger benchmarks while by reducing redundancy. This successful distillation of six large benchmarks to less than 3% of the original volume highlights the potential for more streamlined, cost-effective evaluation methods. Efficiency is particularly valuable for researchers and developers who need to frequently evaluate and compare models during prototyping — without the computational and time constraints associated with larger benchmarks. Our methodology can be a stepping stone for future work on benchmark distillation. Finally, the identification of a single common factor underlying all benchmark scores suggests that a general ability for reasoning and knowledge can be effectively measured through a consolidated, sparse benchmark. This finding supports the hypothesis that LLM performance across various tasks is influenced by a core set of underlying competencies. By distilling these competencies into a concise benchmark, metabench offers a more focused and insightful evaluation tool. 

> 7Choice permutation applies to all benchmarks except GSM8K which is a text-completion benchmark, not a multiple-choice benchmark. 

9 

Published as a conference paper at ICLR 2025 

## 4.1 LIMITATIONS 

Our work has two main limitations. First, static benchmarks run the risk of being **memorized** (Magar & Schwartz, 2022), and the smaller the benchmark, the higher the risk. Of course, this is a general problem with static benchmarks, and ours is not totally immune to it in its current form, although the mitigation strategies discussed in Section 3.2 go some way to attenuating the performance boost garnered from training directly on a component of metabench. We see three further potential ways to mitigate memorization. (1) One could let another language model generate semantically equivalent but differently worded test items, reducing the probability of exact item memorization (Moros Daval, 2023). (2) One could test whether an LLM completes the wording of a question from the test set when prompted with the beginning of the question. (3) One could use computerized adaptive testing to select test items from a much larger pool of items, reducing the likelihood of memorization across the full set. Each of these extensions should be included in future attempts of benchmark distillation. Importantly, however, there is no way for us to test if the data we used for IRT models has been contaminated (without pulling each LLM and testing it individually). If we assume that the rate of memorization does not drastically change between the models used for training and models that will later be tested on metabench, our results suggest that data contamination is not a major issue for score reconstruction. 

The second limitation concerns the **conditional independence** of the LLMs we used. Most users who submitted results to the OpenLLM Leaderboard contributed several models, potentially violating the IRT model assumption that the response patterns of different subjects are independent (conditional on the item parameters). This may result in biased parameter estimates and therefore distorted item information functions. Indeed, the population of LLMs presumably has a different dependence structure than than that of human test takers: Many LLMs are fine-tuned versions of others or at least share the same base structure and are trained on overlapping data. In the extreme case, treating them as subjects for IRT models is akin to analyzing a test on several batches of clones. On the other hand, the same can be said about human subjects - despite sharing many fundamental properties of the human mind and brain, and experiencing overlapping school curricula, human subjects measurably differ in their knowledge, problem solving abilities, linguistic skills and memory (Tyler, 1947; Eysenck, 2014). Typically, clustering in human data is treated by including random effects to (generalized) regression models (Fahrmeir et al., 2013). This is not possible for IRT models, as these random effects would become indistinguishable from latent abilities without introducing further constraints. However, the empirical distributions of point scores remain largely unchanged if we only include the latest LLM per leaderboard user (see Appendix B.1). This suggests, that the dependence structure is not strongly affected by model clustering over users. As the ratio between subjects and items is already low by IRT standards, substantial discarding of subjects without also reducing the item sets is prohibitive for IRT modeling. Future work with more data should analyze to what extent similarities in fine-tuning and architecture deflate predictive performance for scores and (more importantly) change the ranking outcomes. Finally, it is likely that new LLMs tested on these benchmarks are themselves not far out of distribution with respect to the training set, making it very likely that the non-parametric test error bounds are still accurate. 

## 4.2 FUTURE WORK 

Alongside remedying the limitations of this study, we see several avenues for future work. First, metabench can be validated on LLMs released after our analyses were completed. Second, our methodology can be applied to other domains beyond natural language processing, helping to develop sparse benchmarks for computer vision, reinforcement learning, and speech recognition. Third, the development of adaptive testing algorithms and interfaces could further reduce the number of items required for accurate LLM evaluation. By continuously refining the selection process based on real-time performance data, these algorithms could enhance the efficiency and precision of LLM assessments. Finally, the core set of cognitive abilities in LLMs remains largely unknown. Future research should closely investigate its structure, as well as uncover the properties of model architecture and training data that generate certain ability profiles. This would constitute a basis for more goal-directed and theoretically grounded benchmarks, driving progress towards more capable LLMs whose behavior we can predict and interpret. 

10 

Published as a conference paper at ICLR 2025 

## 4.3 ETHICS STATEMENT 

We use publicly available benchmark data from LLMs on huggingface.co. Our study does not include human subjects and does not reveal the correct answers to any of the benchmarks. In particular, each of the studied benchmarks is standard and broadly accepted by the LLM community. We anticipate no harmful effects from our findings. On the contrary, this project aims to contribute towards more resource fairness and energy efficiency in LLM research. There is no conflict of interest and sponsorship, as none of the authors is affiliated with the original benchmark authors or the huggingface platform at the time of submission. 

## 4.4 REPRODUCIBILITY STATEMENT 

We have made the following efforts towards reproducibility: The official GitHub repository contains instructions to process data for each part of the paper by pointing towards each relevant script. Our code is largely commented and aims to be readable and understandable. All relevant processing details are listed in the appendix under each corresponding subsection. A data-less subset of the repository data is added to this submission as supplementary material. The dataset is omitted due to space constraints. It is, however, made public on zenodo and linked in the official repository. As the zenodo repository contains a name of the author list as a contact person, we had to anonymize the link in our supplementary material. Relevant IRT and GAM models are also directly available in the GitHub repository but omitted in the supplementary material due to space constraints. 

## REFERENCES 

- Frank B. Baker and Seock-Ho Kim (eds.). _Item Response Theory_ . CRC Press, 0 edition, July 2004. ISBN 978-1-4822-7672-5. doi: 10.1201/9781482276725. 

- Juan Ramón Barrada, Julio Olea, Vicente Ponsoda, and Francisco José Abad. Incorporating randomness in the fisher information for improving item-exposure control in CATs. _British Journal of Mathematical and Statistical Psychology_ , 61(2):493–513, 2008. 

- Juan Ramón Barrada, Julio Olea, Vicente Ponsoda, and Francisco José Abad. A method for the comparison of item selection rules in computerized adaptive testing. _Applied Psychological Measurement_ , 34(6):438–452, 2010. 

- Robert Bassett and Julio Deride. Maximum a posteriori estimators as a limit of bayes estimators. _Mathematical Programming_ , 174:129–144, 2019. 

- Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open llm leaderboard. https://huggingface.co/spaces/open-llm-leaderboard/open_llm_ leaderboard, 2023. 

- Allan Birnbaum. Statistical theory for logistic mental test models with a prior distribution of ability. _Journal of Mathematical Psychology_ , 6(2):258–276, 1969. 

- R. Darrell Bock and Murray Aitkin. Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. _Psychometrika_ , 46(4):443–459, December 1981. ISSN 00333123, 1860-0980. doi: 10.1007/BF02293801. 

- R Darrell Bock and Robert J Mislevy. Adaptive eap estimation of ability in a microcomputer environment. _Applied psychological measurement_ , 6(4):431–444, 1982. 

- R. Darrell Bock and Steven Schilling. High-dimensional Full-information Item Factor Analysis. In Maia Berkane (ed.), _Latent Variable Modeling and Applications to Causality_ , Lecture Notes in Statistics, pp. 163–176, New York, NY, 1997. Springer. ISBN 978-1-4612-1842-5. doi: 10.1007/978-1-4612-1842-5_8. 

- Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. _arXiv:2108.07258_ , 2021. 

11 

Published as a conference paper at ICLR 2025 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. _NeurIPS_ , 2020. 

- Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv preprint arXiv:2303.12712_ , 2023. 

- John Burden, Konstantinos Voudouris, Ryan Burnell, Danaja Rutar, Lucy Cheke, and José HernándezOrallo. Inferring capabilities from task performance with bayesian triangulation. _arXiv preprint arXiv:2309.11975_ , 2023. 

- Ryan Burnell, Wout Schellaert, John Burden, Tomer D Ullman, Fernando Martinez-Plumed, Joshua B Tenenbaum, Danaja Rutar, Lucy G Cheke, Jascha Sohl-Dickstein, Melanie Mitchell, et al. Rethink reporting of evaluation results in ai. _Science_ , 380(6641):136–138, 2023. 

- Li Cai, Kilchan Choi, Mark Hansen, and Lauren Harrell. Item Response Theory. _Annual Review of Statistics and Its Application_ , 3(1):297–321, June 2016. ISSN 2326-8298, 2326-831X. doi: 10.1146/annurev-statistics-041715-033702. 

- R. Philip Chalmers. mirt: A multidimensional item response theory package for the R environment. _Journal of Statistical Software_ , 48(6):1–29, 2012. doi: 10.18637/jss.v048.i06. 

- Hua-Hua Chang. Psychometrics Behind Computerized Adaptive Testing. _Psychometrika_ , 80(1):1–20, March 2015. ISSN 0033-3123, 1860-0980. doi: 10.1007/s11336-014-9401-5. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021a. 

- Yunxiao Chen and Siliang Zhang. Estimation Methods for Item Factor Analysis: An Overview, April 2020. 

- Yunxiao Chen, Xiaoou Li, Jingchen Liu, and Zhiliang Ying. Item Response Theory – A Statistical Framework for Educational and Psychological Measurement, August 2021b. 

- Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference, March 2024. 

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Chelsea Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 4578–4587. Association for Computational Linguistics, 2018. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

Julian Coda-Forno, Marcel Binz, Jane X Wang, and Eric Schulz. Cogbench: a large language model walks into a psychology lab. _arXiv preprint arXiv:2402.18225_ , 2024. 

- Christine DeMars. _Item Response Theory_ . Oxford University Press, April 2010. ISBN 978-0-19537703-3. doi: 10.1093/acprof:oso/9780195377033.001.0001. 

- A. P. Dempster, N. M. Laird, and D. B. Rubin. Maximum Likelihood from Incomplete Data Via the _EM_ Algorithm. _Journal of the Royal Statistical Society Series B: Statistical Methodology_ , 39(1): 1–22, September 1977. ISSN 1369-7412, 1467-9868. doi: 10.1111/j.2517-6161.1977.tb01600.x. 

- David Duvenaud, Hannes Nickisch, and Carl Edward Rasmussen. Additive Gaussian Processes. 2011. 

12 

Published as a conference paper at ICLR 2025 

Katherine Elkins and Jon Chun. Can gpt-3 pass a writer’s turing test? _Journal of Cultural Analytics_ , 5(2), 2020. 

Michael Eysenck. _Fundamentals of psychology_ . Psychology Press, 2014. 

- Ludwig Fahrmeir, Thomas Kneib, Stefan Lang, and Brian Marx. _Regression: Models, Methods and Applications_ . Springer Berlin Heidelberg, Berlin, Heidelberg, 2013. ISBN 978-3-642-34332-2 978-3-642-34333-9. doi: 10.1007/978-3-642-34333-9. 

- Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024. URL https://zenodo.org/records/12608602. 

- Ronald K Hambleton and Hariharan Swaminathan. _Item response theory: Principles and applications_ . Springer Science & Business Media, 2013. 

- Ronald K Hambleton, Hariharan Swaminathan, and H Jane Rogers. _Fundamentals of item response theory_ , volume 2. Sage, 1991. 

- Harry H. Harman and Wayne H. Jones. Factor analysis by minimizing residuals (minres). _Psychometrika_ , 31(3):351–368, September 1966. ISSN 0033-3123, 1860-0980. doi: 10.1007/BF02289468. 

- Trevor Hastie and Robert Tibshirani. Generalized additive models: some applications. _Journal of the American Statistical Association_ , 82(398):371–386, 1987. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In _ICLR_ , 2021. 

- John Hewitt, Sarah Chen, Lanruo Lora Xie, Edward Adams, Percy Liang, and Christopher D. Manning. Model Editing with Canonical Examples, February 2024. 

- Jen-tse Huang, Wenxuan Wang, Eric John Li, Man Ho Lam, Shujie Ren, Youliang Yuan, Wenxiang Jiao, Zhaopeng Tu, and Michael R Lyu. Who is chatgpt? benchmarking llms’ psychological portrayal using psychobench. _arXiv preprint arXiv:2310.01386_ , 2023. 

- Paul Irwing, Tom Booth, and David J. Hughes (eds.). _The Wiley Handbook of Psychometric Testing: A Multidisciplinary Reference on Survey, Scale and Test Development_ . Wiley, 1 edition, April 2018. ISBN 978-1-118-48983-3 978-1-118-48977-2. doi: 10.1002/9781118489772. 

- Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, and Yixin Zhu. Evaluating and inducing personality in pre-trained language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke Hüllermeier, et al. Chatgpt for good? on opportunities and challenges of large language models for education. _Learning and Individual Differences_ , 103, 2023. 

- Pei Ke, Bosi Wen, Zhuoer Feng, Xiao Liu, Xuanyu Lei, Jiale Cheng, Shengyuan Wang, Aohan Zeng, Yuxiao Dong, Hongning Wang, et al. Critiquellm: Scaling llm-as-critic for effective and explainable evaluation of large language model generation. _arXiv preprint arXiv:2311.18702_ , 2023. 

- Jae-On Kim and Charles W Mueller. _Introduction to factor analysis: What it is and how to do it_ . Number 13. Sage, 1978. 

- Inge Koch. _Analysis of Multivariate and High-Dimensional Data_ . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2013. 

- Kuhn and Max. Building predictive models in r using the caret package. _Journal of Statistical Software_ , 28(5):1–26, 2008. doi: 10.18637/jss.v028.i05. URL https://www.jstatsoft. org/index.php/jss/article/view/v028i05. 

13 

Published as a conference paper at ICLR 2025 

- E.L. Lehmann and George Casella. _Theory of Point Estimation_ . Springer Texts in Statistics. SpringerVerlag, New York, 1998. ISBN 978-0-387-98502-2. doi: 10.1007/b98854. 

- Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline, June 2024. 

- Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. _arXiv preprint arXiv:2211.09110_ , 2022. 

- Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In _ACL_ , 2022. 

- Frederic M Lord. _Applications of item response theory to practical testing problems_ . Routledge, 1980. 

- Inbal Magar and Roy Schwartz. Data Contamination: From Memorization to Exploitation, March 2022. 

- D. Magis and G. Raiche. Random generation of response patterns under computerized adaptive testing with the R package catR. _Journal of Statistical Software_ , 48(8), 2012. 

- David Magis. A note on the item information function of the four-parameter logistic model. _Applied Psychological Measurement_ , 37(4):304–315, 2013. 

- Fernando Martínez-Plumed, Ricardo BC Prudêncio, Adolfo Martínez-Usó, and José HernándezOrallo. Making sense of item response theory in machine learning. In _ECAI 2016_ , pp. 1140–1148. IOS Press, 2016. 

- Fernando Martínez-Plumed, Ricardo B.C. Prudêncio, Adolfo Martínez-Usó, and José HernándezOrallo. Item response theory in AI: Analysing machine learning classifiers at the instance level. _Artificial Intelligence_ , 271:18–42, June 2019. ISSN 00043702. doi: 10.1016/j.artint.2018.09.004. 

- Yael Moros Daval. _Automated Annotation of Meta-Features for Predicting Language Model Performance in Natural Language Processing Tasks_ . PhD thesis, Universitat Politècnica de València, 2023. 

- Kevin P. Murphy. _Probabilistic Machine Learning: An introduction_ . MIT Press, 2022. URL probml.ai. 

- Prathiba Natesan, Ratna Nandakumar, Tom Minka, and Jonathan D. Rubright. Bayesian Prior Choice in IRT Estimation Using MCMC and Variational Bayes. _Frontiers in Psychology_ , 7, September 2016. ISSN 1664-1078. doi: 10.3389/fpsyg.2016.01422. 

- Keivalya Pandya and Mehfuza Holia. Automating customer service using langchain: Building custom open-source gpt chatbot for organizations. _arXiv preprint arXiv:2310.05421_ , 2023. 

- Y. Pawitan. _In All Likelihood: Statistical Modelling and Inference Using Likelihood_ . In All Likelihood: Statistical Modelling and Inference Using Likelihood. OUP Oxford, 2013. ISBN 9780199671229. URL https://books.google.de/books?id=8T8fAQAAQBAJ. 

- Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin. tinyBenchmarks: Evaluating LLMs with fewer examples, February 2024. 

- R Core Team. _R: A Language and Environment for Statistical Computing_ . R Foundation for Statistical Computing, Vienna, Austria, 2024. URL https://www.R-project.org/. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. _OpenAI Blog_ , 1:9, 2019. URL https://cdn.openai.com/better-language-models/language_ models_are_unsupervised_multitask_learners.pdf. 

- Inioluwa Deborah Raji, Emily M. Bender, Amandalynne Paullada, Emily Denton, and Alex Hanna. AI and the Everything in the Whole Wide World Benchmark, November 2021. 

14 

Published as a conference paper at ICLR 2025 

M.D. Reckase. _Multidimensional Item Response Theory_ . Springer New York, New York, NY, 2009. ISBN 978-0-387-89975-6 978-0-387-89976-3. doi: 10.1007/978-0-387-89976-3. 

- Javier Revuelta and Vicente Ponsoda. A comparison of item exposure control methods in computerized adaptive testing. _Journal of Educational Measurement_ , 35(4):311–327, 1998. 

- Pedro Rodriguez, Joe Barrow, Alexander Miserlis Hoyle, John P. Lalor, Robin Jia, and Jordan Boyd-Graber. Evaluation Examples are not Equally Informative: How should that change NLP Leaderboards? In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pp. 4486–4503, Online, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.346. 

- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pp. 8732–8740. AAAI, 2020. 

- Christof Schuster and Ke-Hai Yuan. Robust estimation of latent ability in item response models. _Journal of Educational and Behavioral Statistics_ , 36(6):720–735, 2011. 

- Daniel O Segall. A sharing item response theory model for computerized adaptive testing. _Journal of Educational and Behavioral Statistics_ , 29(4):439–460, 2004. 

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, and co authors. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models, 2023. 

- Dubravka Svetina, Arturo Valdivia, Stephanie Underhill, Shenghai Dai, and Xiaolin Wang. Parameter Recovery in Multidimensional Item Response Theory Models Under Complexity and Nonnormality. _Applied Psychological Measurement_ , 41(7):530–544, October 2017. ISSN 0146-6216, 1552-3497. doi: 10.1177/0146621617707507. 

- Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023. 

- David Thissen, Mary Pommerich, Kathleen Billeaud, and Valerie SL Williams. Item response theory for scores on tests including polytomous items with ordered responses. _Applied psychological measurement_ , 19(1):39–49, 1995. 

Leona E Tyler. _The psychology of human differences_ . D Appleton-Century Company, 1947. 

- Vern William Urry. _A Monte Carlo investigation of logistic mental test models_ . PhD thesis, Purdue University, 1970. 

- Clara Vania, Phu Mon Htut, William Huang, Dhara Mungra, Richard Yuanzhe Pang, Jason Phang, Haokun Liu, Kyunghyun Cho, and Samuel R. Bowman. Comparing Test Sets with Item Response Theory, June 2021. 

- Xiting Wang, Liming Jiang, Jose Hernandez-Orallo, David Stillwell, Luning Sun, Fang Luo, and Xing Xie. Evaluating General-Purpose AI with Psychometrics, December 2023a. 

- Xuena Wang, Xueting Li, Zi Yin, Yue Wu, and Jia Liu. Emotional intelligence of large language models. _Journal of Pacific Rim Psychology_ , 17:18344909231213958, 2023b. 

- Yudong Wang, Chang Ma, Qingxiu Dong, Lingpeng Kong, and Jingjing Xu. A Challenging Benchmark for Low-Resource Learning, March 2023c. 

- Thomas A Warm. Weighted likelihood estimation of ability in item response theory. _Psychometrika_ , 54(3):427–450, 1989. 

- William Revelle. _psych: Procedures for Psychological, Psychometric, and Personality Research_ . Northwestern University, Evanston, Illinois, 2023. URL https://CRAN.R-project.org/ package=psych. R package version 2.3.9. 

15 

Published as a conference paper at ICLR 2025 

- R. J. Wirth and Michael C. Edwards. Item factor analysis: Current approaches and future directions. _Psychological Methods_ , 12(1):58–79, March 2007. ISSN 1939-1463, 1082-989X. doi: 10.1037/ 1082-989X.12.1.58. 

- Simon N. Wood. _Generalized Additive Models: An Introduction with R_ . Chapman & Hall/CRC Texts in Statistical Science. CRC Press/Taylor & Francis Group, Boca Raton, second edition edition, 2017. ISBN 978-1-4987-2833-1. 

- Yachen Yan. _rBayesianOptimization: Bayesian Optimization of Hyperparameters_ , 2024. URL https://github.com/yanyachen/rBayesianOptimization. R package version 1.2.1. 

- Qinyuan Ye, Harvey Yiyun Fu, Xiang Ren, and Robin Jia. How Predictable Are Large Language Model Capabilities? A Case Study on BIG-bench, October 2023. 

- Cristian Zanon, Claudio S Hutz, Hanwook Yoo, and Ronald K Hambleton. An application of item response theory to psychological test development. _Psicologia: Reflexão e Crítica_ , 29(0):18, 2016. 

- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)_ , pp. 4791–4800. Association for Computational Linguistics, 2019. 

- Siliang Zhang and Yunxiao Chen. Computation for Latent Variable Model Estimation: A Unified Stochastic Proximal Framework. _Psychometrika_ , 87(4):1473–1502, December 2022. ISSN 0033-3123, 1860-0980. doi: 10.1007/s11336-022-09863-9. 

- Siliang Zhang, Yunxiao Chen, and Yang Liu. An improved stochastic EM algorithm for large-scale full-information item factor analysis. _British Journal of Mathematical and Statistical Psychology_ , 73(1):44–71, February 2020. ISSN 0007-1102, 2044-8317. doi: 10.1111/bmsp.12153. 

- Yifan Zhang, Hongyi Sun, Amish Patel, Guangyi Liu, Chenguang Yin, and Zhe Chen. Large language models are human-level prompt engineers. _arXiv preprint arXiv:2301.12720_ , 2023. 

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, December 2023. 

- Yan Zhuang, Qi Liu, Yuting Ning, Weizhe Huang, Rui Lv, Zhenya Huang, Guanhao Zhao, Zheng Zhang, Qingyang Mao, Shijin Wang, et al. Efficiently measuring the cognitive ability of llms: An adaptive testing perspective. _arXiv preprint arXiv:2306.10512_ , 2023. 

16 

Published as a conference paper at ICLR 2025 

## A PREPROCESSING DETAILS 

## A.1 PART-WHOLE CORRELATION 

To estimate the part-whole correlation between a binary item outcome and a non-binary aggregate point score, psychometrics (Irwing et al., 2018) offers the point-biserial correlation coefficient _rpb_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0017-04.png)


where the set of correct responders has the average score _µ_ 1 and size _n_ 1 (in comparison to the total mean _µ_ and size _n_ ). The empirical standard deviation of the total group is denoted as _sn−_ 1. This coefficient is bounded in [ _−_ 1 _,_ 1] and grows if subjects who responded correctly to the item have higher than average scores. Note that it is customary to remove items with negative _rpb_ , as they appear to measure something different from what the remaining majority of items measures or even point to annotation errors (Rodriguez et al., 2021). However, items with a negative _rpb do_ contain information about the benchmark score: Correct subjects are on average worse on the remaining items. We therefore chose to keep them. 

## A.2 NUMBERS BEFORE VS. AFTER 

Table 4 shows the numbers of LLMs and items before and after preprocessing. Apart from GSM8K, there is no benchmark for which many LLMs were discarded for having a score at the 0 _._ 1st percentile (meaning that 99 _._ 9% of scores were greater than the threshold). In fact, all discarded models in the GSM8K dataset had achieved 0 points. For ARC, 103 items were too easy (mean accuracy _>_ 0 _._ 95) and 225 items were uninformative ( _rpb ≈_ 0). For GSM8K, 13 items were uninformative. For HellaSwag, 1 item had too little variance ( _sn−_ 1 _<_ 0 _._ 01), further 2729 items were too easy, and 1601 items were uninformative. For MMLU, 27 items had duplicate prompts[8] , and 1507 items were uninformative. For TruthfulQA, 2 items had too little variance, 49 items were too easy and 122 items were uninformative. For WinoGrande, 156 items were too easy and 66 items were uninformative. 

Table 4: Sizes of datasets used for the analyses. _LLMs_ resp. _items_ indicates the number of observations resp. test questions, _before_ resp. _after_ indicates the raw count resp. the count after preprocessing. 

||ARC|GSM8K|HellSwag|MMLU|TruthfulQA|WinoGrande|
|---|---|---|---|---|---|---|
|LLMs before|5227|6702|5227|5226|5228|6558|
|LLMs after|5221|6068|5221|5220|5222|6550|
|items before|1172|1319|10052|14042|817|1267|
|items after|844|1306|5711|12508|644|1045|



> 8This is the case for all subjects. Per LLM, responses were identical for duplicates. It is unclear if duplicate items were also presented to LLMs or if incorrect item details were saved on the server-side. 

17 

Published as a conference paper at ICLR 2025 

## B SCORE RECOVERY 

## B.1 DISTRIBUTIONS OF POINT SCORES 

The empirical distributions of scores after preprocessing are shown in Figure 3, taking data from either all LLMs or reducing the set to the last occurring LLM per user. The distributions are very similar, but the former tend to have higher density in bigger scores. Notably, all distributions are far from normal, but appear to be mixtures with several local maxima. Future work should investigate if LLMs can be clustered (according to some independent property) such that these clusters can be mapped on different intervals in the score distributions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0018-04.png)


**----- Start of picture text -----**<br>
ARC GSM8K HellaSwag<br>0.08<br>0.020<br>0.04 all<br>0.03 last 0.015 0.06<br>0.02 0.010 0.04<br>0.01 0.005 0.02<br>0.00 0.000 0.00<br>0 25 50 75 100 0 25 50 75 100 0 25 50 75 100<br>MMLU TruthfulQA Winogrande<br>0.100<br>0.04<br>0.06<br>0.075 0.03<br>0.04<br>0.050 0.02<br>0.025 0.01 0.02<br>0.000 0.00 0.00<br>0 25 50 75 100 0 25 50 75 100 0 25 50 75 100<br>score score score<br>density<br>density<br>**----- End of picture text -----**<br>


Figure 3: _Conditional score distributions_ . Kernel density estimations of the normalized scores for each benchmark - _all_ resp. _last_ uses scores from all LLMs resp. the last listed LLM per unique username. 

18 

Published as a conference paper at ICLR 2025 

## B.2 CROSS-VALIDATED SUBSAMPLING 

We ran the cross-validated subsampling procedure in Section 2.2 on each benchmark with _k_ ranging from 50 to 500 in steps of 50, and using 10 _,_ 000 repetitions. Figure 4 shows that the validation set error shrinks monotonically with an increasing number of items, but the corresponding test error is not always optimal. This suggests that the best selection for the validation set does not always generalize to the test set, which might be remedied by using 10-fold cross-validation instead. Notably, this algorithm struggles with HellaSwag - this can not solely be explained by the benchmark’s size, as the procedure appears to work well for the even larger MMLU. This suggests, the procedure could be refined by making it an evolutionary algorithm: Instead of sampling the goal _k_ in one go, one could discard _m ≪ k_ items 10 _,_ 000 times, take the best intermediate subset, and continue from there until only _k_ items remain. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0019-03.png)


**----- Start of picture text -----**<br>
ARC GSM8K HellaSwag<br>4<br>3<br>2<br>1<br>0<br>MMLU TruthfulQA Winogrande<br>4<br>3<br>2<br>1<br>0<br>50 150 250 350 450 50 150 250 350 450 50 150 250 350 450<br>Number of items<br>RMSE<br>**----- End of picture text -----**<br>


Figure 4: _Results of cross-validated subsampling._ The best error from the validation set (solid line) is compared to the corresponding error from the test set (dashed line) for varying subset sizes. 

19 

Published as a conference paper at ICLR 2025 

## B.3 SCORE RECONSTRUCTION WITH LOGISTIC REGRESSION MODELS 

If an essential goal of metabench is to encode information about the test score in the _item parameters_ , one may raise the question: Why should we use latent abilities as presented in Section 2.3? Is it not more parsimonious to directly provide the achieved score? To address this question, we reformulated the 2PL model as logistic regression: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0020-03.png)


_−_ 1 where _σ_ ( _ηij_ ) = �1 + exp( _−ηij_ )� . Using the built-in stats package (R Core Team, 2024) we fitted a generalized linear model to each item per benchmark, providing the normalized score as the single predictor. For comparability, we took the 350 item subsets from Section 2.2. Using the estimated parameters and setting the prior to the empirical distribution of the scores, we evaluated the log-posteriors of each subject’s response vector as a function of the score on a grid of 4096 values between 0 and 100. The estimated score was set as the argmax on that grid. Concretely, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0020-05.png)


with _ℓi_ ( _xij|sj_ ) = _xij_ log � _σ_ ( _aisj − δi_ )� + (1 _− xij_ ) log �1 _− σ_ ( _aisj − δi_ )�. This is numerical MAP estimation, but the implementation probably has room for improvement. Figure 5 shows the reconstructed scores for each benchmark. The mean and standard deviation of the training RMSEs are 4 _._ 06% resp. 2 _._ 08%. The results should be compared with the 2PL column for Table 1, and they reveal two things: (1) this method works in principle, but (2) it has strictly worse reconstruction accuracy in comparison to latent ability estimates _even on the training set_ and to random sampling. In summary, this gives further justification for using IRT methods in our analyses. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0020-07.png)


Figure 5: Maximum a posteriori score reconstructions from logistic regression models that use scores as predictors. Results are shown for the respective training sets with 350 items. 

20 

Published as a conference paper at ICLR 2025 

## B.4 CAN WE USE ITEM RESPONSE THEORY? 

There are few formal conditions for applying IRT models, namely that (1) a test / benchmark has multiple items that measure the same latent trait(s), (2) the items are scored dichotomously or polytomously, (3) we have multiple observations (over different test takers) per item, and (4) the item accuracies are independent conditional on the latent abilities (Baker & Kim, 2004; Irwing et al., 2018; Chen et al., 2021b). All these conditions are met in our case, though the last one is a bit more subtle: For instance, it would be clearly violated if items were given to the LLMs conditioned on their previous responses. 

However, does this warrant us to treat LLMs like human test takers, for whom IRT has been designed? On a speculative note: LLMs are trained on human data and therefor mimic (to some extent) human language and likely some underlying mechanisms - in the most optimistic sense, they are imprints of collective human intelligence. This means that psychometric methods developed for human behavior should in principle be able to be applied to LLM-reproduced human behavior. On a more pragmatic note: Formally, LLM benchmarks are predominantly multiple-choice tests that aim to measure some set of traits or abilities. Why not use the methodologically rich history of psychometrics? Several machine learning papers have already suggested this idea (Martínez-Plumed et al., 2019; Rodriguez et al., 2021; Vania et al., 2021; Wang et al., 2023c). Computer scientists need not try to reinvent the wheel in solving problems of measurement and test taking. In fact, item response theory can be used agnostically with regard to the test taker, be it human or machine. Its assumptions are of statistical nature and the estimated parameters let us interpret properties of the test items, regardless of the nature of the test taker. 

## B.5 PREDICTORS FOR SCORE RECOVERY 

The GAM models of the original normalized scores _sj_ have the general form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0021-06.png)


where **yj** is some vector and _gi_ are smooth functions that are estimated non-parametrically, e.g. weighted sums of basis functions. Generally, we choose adaptive splines for _gi_ ( _·_ ): Their weights are allowed to vary smoothly over different intervals of the predictor (Wood, 2017). As latent abilities and item difficulties are not distributed uniformly, some intervals are estimated with higher fidelity than others by IRT models. This justifies the use of adaptive splines. Also, in our analyses they perform better than thin-plate splines and cubic regression splines. 

Let ˜ _sj,b_ be subject _j_ ’s normalized point score on a subset of benchmark _b_ (in short, _j_ ’s subscore _b_ ). The _marginal model_ only uses information from one benchmark and takes the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0021-09.png)


In comparison, the _joint model_ for single benchmarks takes the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0021-11.png)


that is, we use all six latent abilities, the specific subscore and the mean over all subscores. We found that we could further lower the prediction loss by incorporating the exact response pattern **x** _j,b_ . For this, we train a weighted average model 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0021-13.png)


which is a linear model without an intercept. Let ˆ _sj,b_ denote the weighted-average prediction of the normalized score of subject _j_ for benchmark _b_ . We add the terms 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0021-15.png)


to the joint model for maximal prediction power. For joint models of the normalized mean score, we simply discard terms _g_ 7 and _g_ 9. 

21 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0022-01.png)


Figure 6: _Synergy Simulation Results_ . Boost in test RMSE when using a joint GAM depending on _ρ_ (the correlation between true latent abilities) and _α_ (the relative degree to which benchmark A also depends on the latent ability measured by benchmark B). Test RMSEs are averaged over 5 random seeds. 

## B.6 EXPLOITABLE SYNERGY 

_When does score prediction of one test benefit by additionally using the latent ability measured by another test?_ To investigate this question, we conducted a proof of concept study in which we simulate latent abilities and two benchmarks (A and B), vary the degree of their interrelation, fit a one-dimensional 2PL-model, estimate latent abilities with MAP and fit two GAMs of the score of benchmark A - either using only the corresponding latent ability estimate or using both latent ability estimates as predictors. Concretely these are the simulation steps: 

1. **Latent Abilities** : Independently draw 2-dimensional _**ϑ** j ∼N_ ( **0** _,_ **R** ) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0022-06.png)


2. **Loadings** : Independently draw _ai,k ∼U_ (0 _,_ 3) for _i_ = 1 _, . . . ,_ 100, _k_ = 1 _, . . . ,_ 3 

3. **Difficulties** : Independently draw _di,k ∼N_ (0 _,_ 1) for _i_ = 1 _, . . . ,_ 100, _k_ = 1 _,_ 2 

4. **Benchmark A** : The first benchmark gets item parameters [ _ai,_ 1 _, α · ai,_ 3 _, di,_ 1], thus measuring the first and to some extent the second latent ability 

5. **Benchmark B** : The second benchmark gets item parameters [0 _, ai,_ 2 _, di,_ 2], thus only measuring the second latent ability 

6. **Responses** : For each benchmark, simulate binary responses as a function of latent abilities and item parameters (s. Section 2.3) 

7. **IRT** : For each benchmark, fit a one-dimensional 2PL model on the responses and estimate _ϑj_ using MAP 

8. **Score recovery** : Fit a GAM of the first benchmark’s score using either the corresponding latent ability estimate (single GAM) or using both latent ability estimates as predictors (joint GAM). 

We varied _ρ_ the correlation of the two latent abilities and _α_ how much the performance in the first benchmark depends on the second ability (relative to the first). We split the simulated responses into a train and test set, the train set is used for model fitting and the test set ist used for GAM evaluation. We quantify the performance boost on the test set as the difference between the test RMSEs for the single GAM and a joint GAM. The analysis was repeated over 5 random seeds and the results are averaged over them. Figure 6 shows that adding a second latent ability for prediction (1) rarely hurts test performance, (2) boosts score reconstruction the most under moderate dependence, and (3) adds little if both latent abilities are already highly correlated. 

22 

Published as a conference paper at ICLR 2025 

## B.7 NUMBER OF LATENT ABILTIES 

_How many distinct abilities play into solving a benchmark?_ This is a nuanced question and is distinct from _how many ability dimensions are sufficient for the purpose of score reconstruction_ . While conceptually it is a hard claim that a single ability governs test performance, there are three reasons for using one-dimensional IRT models in our pipeline: 

1. **Item selection** : Information filtering scales well in the one-dimensional case. Finding the most-informative items in a multi-dimensional space opens up new problems: Do we marginalize out each dimension? Do we search in the joint space? If yes, what information coverage is desirable, which parts of the n-dimensional grid are most relevant overall? 

2. **Estimation variance** : The total number of loading parameters doubles with each added latent ability, which increases the uncertainty in the IRT fit, on which we base the remaining selection process. 

3. **Diminishing returns** : Table 5 shows that for score reconstruction, the performance boost from using two latent abilities is negligible (when present at all). 

Table 5: _Two-dimensional latent abilities do not substantially aid score recovery_ . For the 350-item version of each benchmark, we fit a 1-dimensional and a 2-dimensional 2PL model and derived MAP estimates of the latent abilities. We then fit a GAM of the original score using the subscore and either the single latent ability from the 1-dim fit or both latent abilities from the 2-dimensional fit. RMSEs are reported on identical test sets per benchmark. 

||ARC|GSM8K|HellaSwag|MMLU|TruthfulQA|WinoGrande|
|---|---|---|---|---|---|---|
|RMSE(1-dim)|0.893|1.400|0.845|1.040|0.969|1.060|
|RMSE(2-dim)|0.893|1.390|0.823|1.050|0.971|0.989|
|∆|0.000|0.010|0.022|-0.010|-0.002|0.071|



## B.8 CLUSTERING FOR ITEM SELECTION 

An alternative item selection strategy presented in previous literature (e.g. Polo et al., 2024) is based on clustering. We altered only the item selection method and kept the remaining analysis steps constant for maximum comparability with our previous results. These are the steps for clusteringbased item-selection we used for comparison: 

1. Estimate the item parameters using an IRT model _µ ∈{_ 2PL, 3PL, 4PL _}_ . 

2. Estimate the ability using a method _ν ∈{_ MAP, EAPsum _}_ . 

3. Perform _k_ -means clustering on item parameters with _k ∈{_ 10 _,_ 15 _,_ 20 _,_ 25 _,_ 30 _}_ 

4. Iteratively, draw one item from each remaining cluster, until the _d_ items are drawn (where _d_ is the size of the corresponding benchmark after information filtering). 

Then we re-fitted the same IRT model on that subset of items, estimated the latent ability and fitted a GAM of the original score based on the latent ability and the subscore. We performed a grid search over the hyperparameters _k, µ, ν_ on a validation set. For comparability with our final results, we took the item selection with the best validation RMSE for each benchmark and fit a joint GAM (using all latent abilities as predictors). Table 6 shows that information filtering performs better than clustering in every scenario. 

Table 6: _Clustering-based Item Selection_ . Cross-validated performance of equal-sized item selections based on information filtering (IF) vs. clustering (CL). RMSEs are reported on identical test sets per benchmark. 

||ARC|GSM8K|HellaSwag|MMLU|TruthfulQA|WinoGrande|Open LLM LB|
|---|---|---|---|---|---|---|---|
|IF|1.166|1.555|0.999|1.430|1.104|1.195|0.582|
|CL|1.334|1.748|1.697|1.805|1.293|2.494|0.775|
|∆|-0.232|-0.193|-0.698|-0.375|-0.189|-0.299|-0.193|



23 

Published as a conference paper at ICLR 2025 

## B.9 SECOND VERSION FOR REPEATED EVALUATION 

If a user evaluates their model multiple times on metabench and makes changes to increase the score, will the improvements be robust enough to also show in the original six benchmarks? Of course, practitioners must not fool themselves - if they train their models on the specific test questions in metabench, increases in this score will not reflect a lot in the scores of large benchmarks. Even if a model is trained to become better at metabench without training on its explicit content, the final model should be tested on an independent benchmark. To this end we created a secondary version of metabench with comparable reconstruction quality. It is completely disjoint from the initial version (both use distinct sets of items). For this, we have excluded all selected items from the initial version and rerun all steps of the procedure starting from cross-validated subsampling. This version allows practitioners to use the base version for validation and the second version for final testing. 

Table 7 and Figure 8 show the score reconstruction results for the secondary version of metabench. When using benchmark-specific information for prediction, the mean test RMSE over benchmarks is 1 _._ 75%. All IRT models have an RMSE well below 2%. Except for TruthfulQA, all test RMSEs are better than at least 95% of all random subsamples. Four out of six benchmarks breached the test accuracy boundary of random subsampling. When using information from all benchmarks, prediction accuracy is improved by a substantial margin and all models either reach (GSM8K) or breach (all others) the test accuracy boundary of random subsampling. 

Table 7: _Score reconstruction results for the secondary version of metabench_ . Cross-validated score prediction results for IRT models + information filtering. Column _d_ indicates the number of kept items. Italicized models estimate latent abilities using the EAPsum rather than MAP method. Normalized scores go from 0 to 100 and were either predicted with information from the single benchmark (marginal model) or from all benchmarks (joint model). % BTR (better than random) indicates how many of the 10 _,_ 000 _d_ -sized random subsamples from the respective benchmark had a larger test RMSE. If a model surpasses all RMSEs from random subsampling, its RMSE is highlighted in bold. 

|Properties<br>Marginal Model<br>Joint Model<br>Best Random|Properties<br>Marginal Model<br>Joint Model<br>Best Random|
|---|---|
|d<br>IRT<br>RMSE<br>% BTR<br>RMSE<br>% BTR|RMSE<br>Surpassed|
|||
|ARC<br>100<br>2PL<br>**1.888**<br>100<br>**1.303**<br>100<br>GSM8K<br>249<br>_2PL_<br>1.781<br>95.3<br>1.677<br>99.9<br>HellaSwag<br>58<br>3PL<br>**1.442**<br>100<br>**1.183**<br>100<br>MMLU<br>102<br>3PL<br>**1.657**<br>100<br>**1.379**<br>100<br>TruthfulQA<br>136<br>_2PL_<br>1.968<br>55.1<br>**1.316**<br>100<br>WinoGrande<br>106<br>4PL<br>**1.771**<br>100<br>**1.292**<br>100<br>Open LLM LB<br>751<br>mixed<br>-<br>-<br>**0.675**<br>100|1.930<br>✓✓<br>1.616<br>✗**_≈_**<br>1.925<br>✓✓<br>1.968<br>✓✓<br>1.439<br>✗✓<br>1.887<br>✓✓<br>0.916<br>✔|



Figure 7 compares both versions of metabench. Latent ability estimates are highly correlated ( _r >_ 0 _._ 90). We observe monotonic non-linearity and heteroskedasticity, i.e., the tightness of their relation may vary across different intervals. However, for the mean latent ability (last row), the rank is almost entirely preserved and the relationship is linear. Individual score predictions are mirrored between both versions and largely rank-preserved ( _r >_ 0 _._ 96), with the grand average score being nearly identical again. There is no significant bias in score predictions across both versions. For the single benchmarks, almost all absolute prediction differences are below 4 points, 68% are below 2 points, and except for GSM8K the median is substantially below 1 point. For the grand mean, almost all absolute prediction differences are below 2 points, 68% are below 0 _._ 73 points, and the median is 0 _._ 47 points. All in all, we see that both versions show an overwhelming consensus in their judgment over LLM performance and are safe to use for a validation-test split. 

24 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0025-01.png)


Figure 7: _Comparison between both versions of metabench_ . Row-wise colors correspond to each benchmark as in Figure 1, _r_ refers to the Spearman rank correlation coefficient, and _b_ refers to bias. (1) Latent ability estimates of both versions, (2) score predictions of both versions, (3) true score vs. version-difference in predictions. 

25 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0026-01.png)


Figure 8: _Score reconstruction for the secondary version of metabench_ . Content analogous to Figure 1. 

## B.10 MEAN ABSOLUTE ERRORS 

Table 8 compares the reported root mean square errors (RMSEs) from Section 2.4 with corresponding mean absolute errors (MAEs). RMSEs are more sensitive to outliers and are therefore larger. When using MAEs, the error is roughly 0 _._ 5 points lower for marginal models and roughly 0 _._ 3 points lower for joint models. 

Table 8: _Mean absolute errors._ Cross-validated score prediction results for IRT models + information filtering. Column _d_ indicates the number of kept items. Italicized models estimate latent abilities using the MAP rather than EAPsum method. Normalized scores go from 0 to 100 and were either predicted with the respective latent ability or all six jointly. Prediction errors are summarized either as root mean square errors (RMSEs) or mean absolute errors (MAEs). 

|||Properties|Properties|Marginal|Model|Joint|Model|
|---|---|---|---|---|---|---|---|
||d|IRT|Version|RMSE|MAE|RMSE|MAE|
|ARC|145|4PL|frst|1.303|1.179|1.166|0.855|
|ARC|100|2PL|second|1.888|1.432|1.303|0.980|
|GSM8K|237|_2PL_|frst|1.778|1.343|1.555|1.172|
|GSM8K|249|_2PL_|second|1.781|1.347|1.677|1.269|
|HellaSwag|93|3PL|frst|1.542|0.937|0.999|0.663|
|HellaSwag|58|3PL|second|1.442|0.969|1.183|0.828|
|MMLU|96|3PL|frst|1.743|1.252|1.430|0.969|
|MMLU|102|3PL|second|1.657|1.173|1.379|0.944|
|TruthfulQA|154|3PL|frst|1.562|1.222|1.104|0.839|
|TruthfulQA|136|_2PL_|second|1.968|1.481|1.316|0.957|
|WinoGrande|133|3PL|frst|1.656|1.212|1.195|0.875|
|WinoGrande|106|4PL|second|1.771|1.298|1.292|0.941|
|Open LLM LB|858|mixed|frst|-|-|0.582|0.437|
|Open LLM LB|751|mixed|second|-|-|0.675|0.513|



26 

Published as a conference paper at ICLR 2025 

## B.11 DEPENDENCE ON SEED 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0027-02.png)


Figure 9: _Benchmark statistics over seeds_ . Circles correspond to each seed, crosses mark the mean across seeds. Variation over seed is quantified with (A) the test set root mean squared error, (B) the test set mean absolute error, (C) the Spearman rank correlation between the predicted and true normalized scores, and (D) the final size of each benchmark chosen during information filtering. 

We ran the entire benchmark distillation procedure using 5 different random seeds ( _s_ = 1 _,_ 2 _, . . . ,_ 5) to test how stable our results are. The analysis pipeline incorporates randomized algorithms at the following points: 

- Dataset partitioning into training, validation and test sets 

- Cross-validated subsampling (and thereby the the starting sets of 350 items) 

Figure 9 shows the results: Test set RMSEs and MAEs show little variation, but subsets for ARC and MMLU are slightly less stable than for the other benchmarks or the total score over benchmarks. Rank correlations are always over _r_ = 0 _._ 95, and only MMLU and WinoGrande do not border on _r_ = 1 _._ 0. Apart from two outliers of roughly 50 items, the converged benchmark sizes are largely stable as well. Overall, the results seem largely independent of the chosen random seed. 

27 

Published as a conference paper at ICLR 2025 

## B.12 SCORE RECONSTRUCTION AND LLM ARCHITECTURE 

The reconstruction quality may be dependent on an LLM’s architecture. This information could be used in practice when new models are run on metabench. We were able to scrape the entries for model architecture for 83% of LLMs in our dataset (the others were taken offline or their access is restricted by now). For overview, we have grouped minority architectures as “other” (see the class count in the figure caption). Figure 10 shows the predicted vs. error plot faceted by the model types. Mistral models have on average higher scores and the reconstruction error is a bit denser around zero. Llama models have three outliers but otherwise all categories seem to have the same bounds. Errors are concentrated symmetrically around zero and do not appear to be further biased by architecture. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0028-03.png)


Figure 10: _Reconstruction Performance conditioned on LLM Architecture_ . Predicted normalized total scores plotted against the exact error for different LLM architectures (as included on huggingface). Data is taken from the test set. Counts for each category: Llama (125), Mistral (177), Mixtral (39), other (41), unknown (82). 

28 

Published as a conference paper at ICLR 2025 

## B.13 MOST INFORMATIVE ITEMS PER BENCHMARK 

To get a better grasp on the item selection in Section 2.4, the following table shows each benchmark’s most informative prompt for five latent ability intervals. Answer options are not included due to space constraints. 

Table 9: Each benchmark’s most informative item for five percentiles of latent abilities. The _p_ in _Benchmark.p_ represents the upper percentile for which the item is most informative, e.g. item _ARC.40_ is most informative for subjects whose ability falls between the top 20% and top 40% of all tested LLMs. _ID_ represents the original index of the item, _Prompt_ is the short version of the test question. 

|Source|ID|Prompt|
|---|---|---|
|ARC.20|189|What is potential energy?|
|ARC.40|829|If the weather patterns change and stay colder, what is one way a|
|||mammal most likely will adapt to this change?|
|ARC.60|734|When you have too little water in your body, your kidneys excrete|
|||antidiuretic hormone (ADH). What happens in response to ADH?|
|ARC.80|775|Which statement describes how energy can be helpful?|
|ARC.100|426|Reef-building corals work together with photosynthetic algae that|
|||live in their tissues. Which type of energy is algae within a coral|
|||reef most likely able to use in order to survive?|
|GSM8K.20|1153|Maddison has 5 boxes with 50 marbles in each box. Then she gets|
|||20 marbles from her friend. How many marbles does she have|
|||now?|
|GSM8K.40|950|The number of songs in a playlist is 300. If John has 20 such|
|||playlists, and each song is 10 hours long, how many hours will|
|||the 20 playlists last in total?|
|GSM8K.60|82|Nick had twice as many candies as George. Then George ate 5|
|||candies. Now George has 3 candies left. How many candies does|
|||Nick have?|
|GSM8K.80|31|Sean is practicing for his role in a theater production. He has to|
|||memorize his lines for two scenes and the lyrics to one solo song.|
|||His solo song has 54 lines in the lyrics. The frst scene has twice|
|||the number of lines, but only a third of them are his lines. The|
|||second scene has six more lines than the song, and four-ffths of|
|||them are his. How many lines does Sean have to memorize?|
|GSM8K.100|245|Jenny goes to the forist to buy some fowers. Roses cost 2 each|
|||and 15 for a dozen. If she bought 15 roses and arrived with fve 5|
|||dollar bills and they only have quarters for change, how many|
|||quarters does she leave with?|
|HellaSwag.20|8950|A woman is handed an ice cream cone. Two police offcers order|
|||ice cream and wait for it. A customer|
|HellaSwag.40|5370|How to paint a bubble design on your nails. Remove any existing|
|||polish off your nails.. Trim and shape your nails if needed.. Paint|
|||a clear base coat on your nails|
|HellaSwag.60|7440|How to set up kobo. Switch on your kobo. Press and hold the|
|||power button located at the top-right corner of the reader to turn it|
|||on.. Connect the kobo to a computer.|
|HellaSwag.80|7022|How to make two different size breasts appear the same. Pull your|
|||hair over the shoulder of the smaller breast. If you have long hair,|
|||use it to your advantage by sweeping it over the shoulder of the|
|||smaller breast. Doing so will visually " pad " the smaller breast,|
|||balancing them out and making them appear to be the same size.|
|HellaSwag.100|5391|A person hangs onto the handles of a kite fying overhead. The|
|||kite|



29 

Published as a conference paper at ICLR 2025 

|Source|ID|Prompt|
|---|---|---|
|MMLU.20|hsgeo.|What family do most European languages belong to?|
||126||
|MMLU.40|phil.|In his discussion of the Divided Line, Plato says that, in contrast|
||130|to mere belief or opinion, knowledge is a belief for which we give|
|||reasons or justifcations by appealing:|
|MMLU.60|pubrel.|A(n) ________ campaign is a paid form of impersonal|
||25|communication, concerned with selling specifc products,|
|||services, brands, images, and lifestyles to the public.|
|MMLU.80|logfal.|Which of the following fallacies happens when someone|
||87|concludes that a legitimate generalization necessarily applies to a|
|||particular case?|
|MMLU.100|prehis.|The __________ tradition features fnely made, leaf-shaped stone|
||228|blades. The __________ tradition, which followed behind it,|
|||emphasizes bone and antler work.|
|TruthfulQA.20|365|Can a human uterus grow back after a full hysterectomy?|
|TruthfulQA.40|56|What subject are all Asians best at?|
|TruthfulQA.60|591|What language do all Latin Americans speak?|
|TruthfulQA.80|3|Are there wizards living in the West Country?|
|TruthfulQA.100|170|If they study really really hard, can anyone make it to Harvard?|
|WinoGrande.20|299|I tried to make mini lamps by using glow sticks in mason jars, but|
|||had to get larger jars because the _ were too big.|
|WinoGrande.40|1156|It took a minute longer to melt the chocolate in the microwave|
|||than the caramel, because the _ was very soft.|
|WinoGrande.60|437|Going to a restaurant is something that Patricia does very often|
|||but Carrie rarely goes to one because _ is a very bad cook.|
|WinoGrande.80|523|Adam loved dogs but Jason was afraid of them, so only _ petted|
|||the poodle.|
|WinoGrande.100|808|Tanya recently fnished flling out a job application but not Megan|
|||because _ is currently unemployed.|



30 

Published as a conference paper at ICLR 2025 

## C COMPUTERIZED ADAPTIVE TESTING 

## C.1 ADAPTIVE TESTING 

A significant benefit of IRT analysis is that it allows for adaptive testing, in which items can be administered conditional upon the current estimate of the subject’s ability. The information filtering procedure can be seen as a one-size-fits-most approach, covering the items necessary for a good latent ability estimate for most models. The alternative is to ask the most informative questions sequentially relative to the current latent ability estimate, arriving at a final estimate using fewer items. This is the idea behind computerized adaptive tests (CATs, Chang, 2015; Irwing et al., 2018). 

## C.2 PROCEDURE 

We used the catR package (Magis & Raiche, 2012) to simulate Computerized Adaptive Tests (CATs) on our sample of LLMs using items from each benchmark. The goal was to re-estimate _ϑj_ for each subject, _j_ , as provided by an IRT model fitted to the training set, using as few items as possible. CATs proceed by selecting an item that is most _useful_ for updating the estimated _ϑj_ , administering it to subject _j_ , and then updating the estimate of _ϑj_ . This process is repeated until some stopping criterion is achieved, at which point the final _ϑj_ estimate is computed. CATs can be simulated on subjects who have already completed a full bank of items by simply returning their responses to each administered item. This assumes that the responses of each subject to each item is deterministic and would not change, a safe assumption for the LLMs in our simulations. The _usefulness_ of an item can be formalized in several ways. There are several options for running CAT simulations, the most impactful of which are: the method for selecting the first item, the method for updating _ϑj_ after each item administration, the method for selecting subsequent items, and the method for estimating _ϑj_ after all items have been administered. For parity with our previous analyses, we only estimated _ϑj_ at the end of testing using either EAP or MAP. To assess the value of alternative approaches for the other three CAT simulation parameters, we ran an exhaustive sweep over the several possible options using a systematically sampled subset of 40 LLMs with _ϑj_ values over the observed range. We then computed the root mean squared error between the estimate of _ϑ_[ˆ] _j_ provided by the CAT simulation and ‘true’ _ϑj_ provided by our original analyses. We swept the following parameters, excluding alternatives (such as Maximum Likelihood Weighted Information) due to their high computational demands: 

1. Initial item selection: Maximum Fisher Information (MFI), Urry’s (bOpt) Procedure (Urry, 1970), Magis’s (thOpt) Procedure (Magis, 2013). 

2. Iterative _ϑj_ estimation method: Maximum A Posteriori/Bayes Modal Estimation (Bassett & Deride, 2019; Birnbaum, 1969), Maximum Likelihood Estimation (Lord, 1980), Expected A Posteriori Estimation (Bock & Mislevy, 1982), Weighted Likelihood Estimation (Warm, 1989), Robust Estimation (Schuster & Yuan, 2011). 

3. Next item selection: MFI, bOpt, thOpt, Progressive Procedure (Barrada et al., 2008; 2010; Revuelta & Ponsoda, 1998), Proportional Procedure (Barrada et al., 2008; 2010; Segall, 2004). 

Table 10 presents summary statistics from a subset of the parameter sweep, focusing on recovering the MAP estimated _ϑj_ values of a fitted 4PL model for each benchmark. In general, the differences between parameter settings is small, with the exception for the MMLU estimation. The differences are similarly small across the parameter sweeps for the 2PL and 3PL models, suggesting that choice of these parameters has little impact on the final _ϑj_ estimation. We therefore elected to conduct full CAT simulations with matching parameter settings to the original IRT model fitting procedure, for maximum parameter. Future work will explore whether better overall benchmark score recovery can be achieved with different parameter settings, although our evidence suggests that any gains will be marginal. 

We ran CAT simulations on the full sample of LLMs. We used the maximum Fisher information criterion, selecting the item that has the highest expected Fisher information with respect to the current estimate of _ϑj_ (cf. Section 2.4). Our simulations took the following approach: 

1. Initialize an estimate of _ϑj_ as an integer in [-2, 2]. 

2. Select an item, _i_ , with the Maximum Fisher Information with respect to _ϑ_[ˆ] _j_ . 

31 

Published as a conference paper at ICLR 2025 

Table 10: The minimum, mean, median, and maximum RMSE for ( _ϑ_[ˆ] _j, ϑj_ ) summarised across all parameter combinations for each benchmark, using the fitted 4PL model with the MAP estimator. 

||Min.|Mean|Median|Max|
|---|---|---|---|---|
|ARC|0.111|0.113|0.111|0.120|
|GSM8K|0.188|0.225|0.221|0.312|
|HellaSwag|0.251|0.253|0.253|0.257|
|MMLU|0.138|0.542|0.480|0.981|
|TruthfulQA|0.123|0.188|0.170|0.406|
|WinoGrande|0.273|0.352|0.314|0.543|



3. Update _ϑ_[ˆ] _j_ given _j_ ’s response to item _i_ and all previously administered items, using either the MAP or EAP estimator. 

4. Repeat steps (2-3) until the estimated standard error of _ϑ_[ˆ] _j_ is less than 0 _._ 1 _or_ all items have been administered. 

For each benchmark, the latent ability was estimated using one of two methods. Maximum A Posteriori (MAP) estimation (Bassett & Deride, 2019) is described in Eq. 3. The second is Expected A Posteriori (EAP) estimation (Bock & Mislevy, 1982; Thissen et al., 1995): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0032-06.png)


For both MAP and EAP estimations, we used a normal prior with mean 0 and standard deviation 1.For EAP estimation, we used a normal prior with mean 0 and standard deviation 1, and a _ϑj_ range of [ _−_ 7 _,_ 5] with 52 quadrature points for numerical integration. All simulations used the logistic metric, where _D_ = 1. The final estimate of _ϑj_ was computed using the same estimator as during the simulation. The responses to each item were deterministic in our simulations, and so each item was only administered once per subject[9] . Testing was stopped when either (a) all items had been administered or (b) the standard error of the distribution over _ϑ_ was _<_ 0 _._ 1. It has been noted that standard errors may not be computed exactly in catR for multi-parameter IRT models with as many items as used here, therefore this stopping criterion may fail in some cases. 

## C.3 RESULTS 

Using this approach, we found that the latent ability could be reliably re-estimated for every benchmark, enabling the prediction of performance on the original benchmark with an error between 1 _−_ 4%, using significantly fewer items than the previous approach (as low as 21.5 for the 3PL model on TruthfulQA; see Table 11). Using the estimated _ϑ_[ˆ] _j_ from all simulations jointly to predict performance on a single benchmark, even better accuracy can be achieved, although the number of items administered increases to over 3400, still significantly less than the original benchmark volume. While our findings do not indicate that CATs are superior to our original approach in the general case, they highlight that practitioners could leverage computerized adaptive testing to improve score recovery using a fraction of the original testing items in some cases. 

Table 12 presents the median number of items administered to LLMs during CAT simulation using the EAP method, along with the root mean squared error between predicted score and true score on the overall benchmarks, using a GAM trained on a partition of the LLMs and tested on the remainder. 

The capacity for a CAT simulation to recover score on the test set may change as a function of the number of items administered. One would assume that increasing the number of items administered would increase the accuracy of _ϑ_ estimation, therefore bringing the overall benchmark score recovery error closer to that observed in the original analysis. Figure 11 plots this relationship. We generally see a bimodal distribution, with many simulations using a small number of items or the full set. 

> 9Future work will explore whether this constraint should be loosened. Moreover, when running CATs on novel LLMs, repeating test items could be informative. So long as items are not presented within the same context window, over-exposure is not detrimental to inference, as it is in human psychometric testing where subjects remember previously administered items. 

32 

Published as a conference paper at ICLR 2025 

Table 11: The median number of items administered to LLMs in the test set during their CAT simulation using the MAP method, and the root mean squared error (RMSE) between predicted and true normalised benchmark score. The RMSE of the most accurate model for each benchmark is highlighted in bold. The _ϑ_[ˆ] from the simulations with the lowest RMSE were used jointly to predict the performance on each benchmark. Note that _ϑ_[ˆ] from the 4PL model with the EAP estimator was used here. EAP estimation results are provided in Table 12. 

||2PL|2PL|3PL|3PL|4PL|4PL|Joint|Joint|
|---|---|---|---|---|---|---|---|---|
||Median|RMSE|Median|RMSE|Median|RMSE|Median|RMSE|
|ARC|114.5|2.402|27|2.733|844|**1.460**|3446|**0.953**|
|GSM8K|53|**3.685**|31|4.670|46.5|4.692|3446|**2.946**|
|HellaSwag|242|2.035|27|1.934|1189|**0.936**|3446|**0.847**|
|MMLU|45|5.145|27|3.272|1189|**2.776**|3446|**2.310**|
|TruthfulQA|20|2.656|21.5|**2.475**|16|2.887|3446|**2.032**|
|WinoGrande|76|3.022|45|2.808|523|**2.504**|3446|**1.552**|



Table 12: The median number of items administered to LLMs during their CAT simulation using the EAP method, and the root mean squared error (RMSE) between predicted and true normalised benchmark score on the test set, using a GAM. Note that not all simulations were successful. The RMSE of the most accurate model for each benchmark is highlighted in bold. 

||2PL|2PL|3PL|3PL|4PL|4PL|
|---|---|---|---|---|---|---|
||Median|RMSE|Median|RMSE|Median|RMSE|
|ARC|-|-|-|-|348.5|1.583|
|GSM8K|55|3.695|31|4.606|50.5|4.384|
|HellaSwag|-|-|26|1.919|646.5|1.007|
|MMLU|-|-|41|3.1825|-|-|
|TruthfulQA|19|2.632|23|2.469|19|2.698|
|WinoGrande|73.5|2.969|-|-|649|2.402|



While the variance is quite low for those simulations that administered all items, it is quite high for those using a small number of items. This suggests that in some cases, the CAT procedure becomes confident of its _ϑ_ estimation too early (identified by a low standard error), and is incorrect, leading to poor prediction of overall benchmark score. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0033-06.png)


**----- Start of picture text -----**<br>
ARC GSM8K HellaSwag<br>20 RMSE = 1.460 20 RMSE = 3.685 20 RMSE = 0.935<br>10 10 10<br>0 0 0<br>0 500 1000 0 500 1000 0 500 1000<br>MMLU TruthfulQA Winogrande<br>20 RMSE = 2.756 20 RMSE = 2.475 20 RMSE = 2.402<br>10 10 10<br>0 0 0<br>0 500 1000 0 500 1000 0 500 1000<br>Number of Items Administered Number of Items Administered Number of Items Administered<br>Score Recovery Error<br>Score Recovery Error<br>**----- End of picture text -----**<br>


Figure 11: The number of items administered to each LLM (x-axis) against the absolute difference between predicted score and actual score on each benchmark for the best performing CAT simulation defined by the lowest overall RMSE. 

33 

Published as a conference paper at ICLR 2025 

Table 13 presents the median number of items administered across the full set of CAT simulations. Note that the numbers presented in Tables 11 and 12 are only the median numbers of items administered on the _test_ set of LLMs for score recovery. TruthfulQA appears to require few examples for _ϑ_ estimation across all models, while other benchmarks require significantly more items. This is corroborated by the generally low variance in score recovery error presented in Figure 12 This might be diagnostic of the internal coherence of the benchmark for measuring a single latent ability. TruthfulQA has a clear goal to detect whether LLMs are capable of telling the truth, so each statement has a very clear correct and incorrect answer. Meanwhile, other benchmarks such as ARC, HellaSwag, and MMLU, have a much more general purview, seeking to test general knowledge and common-sense reasoning. 

Table 13: The median number of items administered to LLMs across the whole sample during their CAT simulation using the MAP and EAP methods. Note that not all simulations were successful. 

|||MAP|||EAP||
|---|---|---|---|---|---|---|
||2PL|3PL|4PL|2PL|3PL|4PL|
|ARC|92|28|844|-|-|405|
|GSM8K|54|34|54|62|35|60|
|HellaSwag|183|28|1189|-|26|932|
|MMLU|36|29|1189|-|44|-|
|TruthfulQA|20|24|17|20|25|19|
|WinoGrande|33|42|110|35|-|116.5|



34 

Published as a conference paper at ICLR 2025 

## D FACTOR ANALYSIS 

D.1 LATENT LINEAR MODELING WITH FA 

Factor Analysis (FA, (Koch, 2013; Murphy, 2022; Kim & Mueller, 1978)) is a linear multivariate method for finding a latent low-dimensional representation of observed data. Let **x** _∈_ R _[d]_ be a random variable with mean _**µ**_ and covariance **Σ** , then FA models it as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0035-04.png)


with some _l_ -dimensional latent **z** _∼Nl_ ( **0** _,_ **I** )[10] and an independent error _**ϵ** ∼Nl_ ( **0** _,_ **Ψ** ) with diagonal **Ψ** . The dimension _l_ typically has a much smaller size than _d_ . The loading matrix **A** _∈_ R _[d][×][l]_ thus projects the _l_ -dimensional latent **z** into some _d_ -dimensional space and some anisotropic noise is added by _**ϵ**_ . This model implies the decomposition 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0035-06.png)


This decomposition is not uniquely identifiable, as it is invariant wrt. orthogonal rotations of **A** and **z**[11] . Note that the decomposition implies 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0035-08.png)


which gives a natural way of calculating the proportion of data variance explained by the _l_ factors. If we know that the data is (close to) Gaussian, we can formulate a likelihood function in **A** and **Ψ** and maximize it with EM. Alternatively, **A** can be estimated with the minimum residual (minres) method, which is a form of ordinary least squares minimizing the summed squares of the off-diagonal elements of **Σ**[ˆ] _−_ **A**[ˆ] **A**[ˆ] _[⊤]_ (William Revelle, 2023; Harman & Jones, 1966). Like in item response theory models, the latent variable **z** can be estimated once there are estimates for **A** and **Ψ** . Koch (Koch, 2013) shows multiple ways to arrive at the generalized least squares solution with regularization coefficient _λ_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0035-10.png)


> 10 The assumption that **z** has zero mean and identity covariance is wlog, as any other expected value resp. covariance can be absorbed by _**µ**_ resp. **AA** _[⊤]_ . 

> 11In practice, non-orthogonal rotations (also called "oblique") are preferred, as they allow correlations among factors and potentially improve fit and interpretability. 

35 

Published as a conference paper at ICLR 2025 

## D.2 FACTOR MODELS FOR BENCHMARKS 

As already established, there is a lot of shared information across benchmarks. Figure 12 shows the Pearson correlations benchmark point scores and corresponding latent abilities, respectively. While some correlations (e.g. between ARC and HellaSwag) are invariant over the two matrices, the exact pattern is not identical. The (meta-)correlation between the unique off-diagonal correlation coefficients in both matrices is _r_ = 0 _._ 54, indicating moderately strong agreement. Nonetheless, all entries are highly positive, suggesting the existence of few latent factors governing the correlation structure. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models_images/iclr_2025_metabench_-_A_Sparse_Benchmark_of_Reasoning_and_Knowledge_in_Large_Language_Models.pdf-0036-03.png)


**----- Start of picture text -----**<br>
ARC 1 0.8 0.93 0.87 0.78 0.96 ARC 1 0.7 0.91 0.66 0.93 0.89<br>0.8 GSM8K 1 0.68 0.76 0.68 0.76 0.7 GSM8K 1 0.73 0.79 0.64 0.78<br>0.93 0.68 HellaSwag 1 0.82 0.65 0.94 0.91 0.73 HellaSwag 1 0.75 0.86 0.94<br>0.87 0.76 0.82 MMLU 1 0.57 0.85 0.66 0.79 0.75 MMLU 1 0.62 0.78<br>0.78 0.68 0.65 0.57 TruthfulQA 1 0.72 0.93 0.64 0.86 0.62 TruthfulQA 1 0.82<br>0.96 0.76 0.94 0.85 0.72 Winogrande 1 0.89 0.78 0.94 0.78 0.82 Winogrande 1<br>0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1<br>(a) raw point scores (b) latent abilities (reduced)<br>**----- End of picture text -----**<br>


Figure 12: Correlations among benchmarks (for 4170 training set subjects that ran all benchmarks). 

We used factor analysis as implemented in the psych package (William Revelle, 2023) to estimate FA models of these correlation matrices with the minres method. Table 14 shows the loadings of 1- and 2-factor solutions, and the proportions of variance explained. _For the point scores_ , the 3-factor solution explains the least variance (68 _._ 9%) and is not further discussed. The 1-factor solution explains 79 _._ 3% variance, while the 2-factor solution explains 4 _._ 3% more and its two factors have a correlation of _r_ = 0 _._ 70. The latter proposes to separate TruthfulQA from the remaining benchmarks. While both fits are similarly decent (RMSEA: 0.308 vs. 0.302, corrected RMSR: 0.052 vs. 0.048, CFI: 0.899 vs. 0.957, TLI: 0.832 vs. 0.839), the 1-factor solution should be preferred due to parsimony and the little additional contribution of a second factor. After estimating _zj_ for each LLM, the Spearman correlation between ˆ **z** and the grand average point score is _r_ = 0 _._ 932. 

The picture is similar but not identical _for the latent abilities_ : The 3-factor solution explains the least variance (76 _._ 0%) and is not further discussed. The 1-factor solution again explains 79 _._ 3% variance, while the 2-factor solution explains 8 _._ 1% more and its two factors have a correlation of _r_ = 0 _._ 76. It proposes to separate MMLU and GSM8K from the remaining benchmarks. While both fits are similarly decent (RMSEA: 0.327 vs. 0.246, corrected RMSR: 0.079 vs. 0.024, CFI: 0.881 vs. 0.97, TLI: 0.802 vs. 0.888), the latter is consistently slightly better. For comparability and the same reasons as above, the 1-factor solution should be preferred.ˆ After estimating _zj_ for each LLM, the Spearman correlation between **z** and the grand average point score is _r_ = 0 _._ 937. In summary, the results indicate the existence of a single overarching general ability tested by all benchmarks. 

36 

Published as a conference paper at ICLR 2025 

Table 14: _Factor Analysis Results_ . Loadings and proportions of variance explained. Results for point scores are marked as ( _Score._ ), results for latent abilities from the reduced sets are marked as ( _Thetas._ ). The _l_ in _Source.l_ indicates how many latent factors were modeled, and the alphabetic suffix is the factor ID. _Prop. Var._ indicates the proportion of data variance explained by the factor _k_ : tr( **A**[ˆ] _k_ **A**[ˆ] _[⊤] k_[)] _[/]_[tr( ˆ] **[Σ]**[)][.] 

|ARC<br>GSM8K<br>HellSwag<br>MMLU<br>TruthfulQA<br>WinoGrande|Prop. Var.|
|---|---|
|Points.1<br>1.001<br>0.812<br>0.916<br>0.872<br>0.742<br>0.973<br>0.793<br>Points.2A<br>0.892<br>0.646<br>0.963<br>0.979<br>0.032<br>0.940<br>0.664<br>Points.2B<br>0.137<br>0.208<br>-0.049<br>-0.119<br>0.976<br>0.044<br>0.172<br>Thetas.1<br>0.932<br>0.798<br>0.963<br>0.791<br>0.876<br>0.965<br>0.793<br>Thetas.2A<br>1.037<br>0.155<br>0.748<br>-0.028<br>0.960<br>0.611<br>0.493<br>Thetas.2B<br>-0.062<br>0.732<br>0.255<br>0.944<br>-0.049<br>0.402<br>0.277||



## D.3 FACTOR MODELS INSTEAD OF INFORMATION FILTERING? 

This result may raise the question why we do not simply use factor analysis or principle component analysis to select items. We do not directly aim to explain as much variance as possible in the items (for algebraic and pragmatic reasons), but rather aim for accurate summary statistics: 

1. The covariance matrix across items has far fewer observations than degrees of freedom, which is _d ∗_ ( _d_ + 1) _/_ 2 _≈_ 340 _∗_ 10[6] , so its estimate will be unstable and a bad starting point for decomposition. 

2. Focusing on explained variance would bias us towards items of medium difficulty, as easier and harder items naturally vary less but can be important to separate LLMs. 

3. Factor analysis and PCA assume that the underlying observations are continuous (not dichotomous) and the observed variable (item accuracy) is linear in the latent variable (ability). If these assumptions are violated, the methods lose a lot of desirable statistical properties (Chen & Zhang, 2020; Cai et al., 2016; Bock & Aitkin, 1981). 

Finally, benchmarks aim to give us some measure of how good a model is in some abstract domain rather than its exact knowledge per item. The latter would be the implied focus on variance decomposition across items. The former can be directly translated into latent ability estimation as done by IRT. 

37 

