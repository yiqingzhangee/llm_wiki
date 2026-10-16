Preprint 

# AN AUDITING TEST TO DETECT BEHAVIORAL SHIFT IN LANGUAGE MODELS 

**Leo Richter**[1] **Xuanli He**[1] **Pasquale Minervini**[2,3] **Matt J. Kusner**[1] 1UCL Centre for Artificial Intelligence, University College London, London, United Kingdom 

2School of Informatics, University of Edinburgh, Edinburgh, United Kingdom 3Miniml.AI, Edinburgh, United Kingdom 

{ucablri, m.kusner}@ucl.ac.uk 

## ABSTRACT 

As language models (LMs) approach human-level performance, a comprehensive understanding of their behavior becomes crucial. This includes evaluating capabilities, biases, task performance, and alignment with societal values. Extensive initial evaluations, including red teaming and diverse benchmarking, can establish a model’s behavioral profile. However, subsequent fine-tuning or deployment modifications may alter these behaviors in unintended ways. We present a method for continual Behavioral Shift Auditing (BSA) in LMs. Our test compares model generations from a baseline model to those of the model under scrutiny and provides theoretical guarantees for change detection while controlling false positives. The test features a configurable tolerance parameter that adjusts sensitivity to behavioral changes for different use cases. We evaluate our approach using two case studies: monitoring changes in (a) toxicity and (b) translation performance. We find that the test is able to detect meaningful changes in behavior distributions using just hundreds of examples. 

## 1 INTRODUCTION 

Language models (LMs) can now achieve human-level performance in a wide range of tasks, including text summarization, machine translation, coding and even acting as AI scientists: generating hypotheses and designing experiments (Achiam et al., 2023; Katz et al., 2024; Lu et al., 2024; Zhang et al., 2024). Because of this, many sectors are looking for ways to use them to improve existing systems (Kasneci et al., 2023; Felten et al., 2023). Unfortunately, one large roadblock to broad LM adoption is their propensity to generate harmful content (Weidinger et al., 2021). For example, GPT-3 has significant anti-Muslim biases (Abid et al., 2021), and GPT-4 has racial and gender biases (Zack et al., 2024). To address this, a significant effort is going into ensuring LM behavior is aligned with our societal values, spawning the field of _AI alignment_ (Ji et al., 2023). A large portion of this effort is on developing ways to _evaluate_ LM behavior, for example, through benchmarks (Wang et al., 2023a) and red-teaming (Perez et al., 2022). 

Given these evaluation techniques, how should they be used to ensure LMs stay safe? To answer this, consider the following two hypothetical settings where this question might be asked: (1) _Internal Audit_ : At the new start-up Chasm Intellect, the LM alignment team is looking for a way to trigger an alarm when LM behavior changes. They know that LM model behavior can change unexpectedly (Li et al., 2024b). For example, fine-tuning a model that has undergone safety evaluation (e.g., GPT-3.5 Turbo (OpenAI, 2023)) can cause it to become less safe, even when using a benign fine-tuning dataset (e.g., Alpaca (Taori et al., 2023); (Qi et al., 2023)). How can the team detect meaningful changes in model behavior? (2) _External Audit_ : A journalist has been playing around with Better geneRAtiVe language modEl (BRAVE), Chasm Intellect’s new model, and has noticed it produces highly toxic generations when asked about seemingly benign topics. Their article on this topic spurred a governmental audit of BRAVE. The government performs an initial evaluation but is worried that model behavior will change when no longer under scrutiny. How can the government regularly check the deployed model’s behavior is the same as the previously certified one? We call the general class of problems detecting changes in LM behaviors _Behavioral Shift Auditing_ (BSA) problems. 

1 

# Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0002-01.png)


**----- Start of picture text -----**<br>
1 2 3<br>vendormodel x 1 , . . . , prompts  x T regulator vendormodel x 1 , . . . , prompts  x T (regulator posing as consumer) regulator {b ( y i ) } [T] i =1 ! {b ( y i [0] [)] [}][T] i =1<br>? ? 6⇡<br>f y 1 , . . . ,  y T f y 1 [0] [, . . . ,] [ y] T [0] behavior behavior<br>initial LLM generations changed LLM generations (e.g., toxicity)  (e.g., toxicity)<br>frequency frequency<br>**----- End of picture text -----**<br>


Figure 1: **An External Audit Example.** A regulator can use the test we describe to perform an external audit: 1. The regulator initially certifies an LM by prompting and evaluating the set of generations received; 2. Later, tipped off that LM behavior may have changed, the regulator poses as a consumer and sends prompts to the model vendor, collecting the generations; 3. The regulator compares the distribution of behavior scores _b_ ( _·_ ) between the initial, certified generations and the later generations using a Behavioral Shift Auditing (BSA) test. If the distributions are sufficiently different the test triggers. Using our proposed method, the regulator can test samples sequentially without increasing the false-positive rate. The method is guaranteed to detect a change if one exists, given enough samples (more details in Section 3). 

In this paper, we formalize the problem of Behavioral Shift Auditing in Language Models. We detail a test that continuously monitors behavioral shift, solely from model generations (e.g. via API calls). Under some weak assumptions, the test provably guarantees that if model generations have different behavior than those of an initial model, the test will detect it, given enough generations. At the same time, if there has not been a change, the test is guaranteed to have tight, non-asymptotic control over the false positive rate. The key insight behind our approach is that one can phrase the problem of Behavioral Shift Auditing as hypothesis testing over the relevant behavioral distribution. This framing allows our test to be applicable to any measurable aspect of model behavior, including also capabilities (e.g., dangerous capabilities (Phuong et al., 2024) or mathematical reasoning capabilities (Mishra et al., 2022a)) and biases (e.g., gender bias (Wang et al., 2023a; Kotek et al., 2023)). 

Using this insight, we develop a test that extends recent work on anytime-valid hypothesis testing (Pandeva et al., 2024), a state-of-the-art sequential testing method that has been successfully applied in various auditing settings (Chugg et al., 2023; Shekhar et al., 2023; Waudby-Smith et al., 2021). Our test checks for changes in model behavior distributions, comparing generations from a reference model with those of another, potentially changed model. The test has a tunable parameter that allows one to vary the strictness of the test. This allows for detecting _any change_ in behavior, which may be more suitable for the _external audit_ setting, to detecting a user-specified _ϵ_ change in behavior, which could be used for the _internal audit_ if small changes are acceptable. Similar to Pandeva et al. (2024), test performance is optimized using a learning algorithm, improving sample efficiency over prior testing methods (Lopez-Paz & Oquab, 2017; Lhéritier & Cazals, 2018; Podkopaev & Ramdas, 2024). This testing approach can complement a full evaluation when used as a warning system. Before an expensive model assessment on large-scale benchmarks (Achiam et al., 2023; Dubey et al., 2024; Zhang et al., 2024), our approach can be used to detect an initial behavior change, which can then trigger a full evaluation. We experimentally verify that our test satisfies theoretical guarantees and we report its sample efficiency on recent LM architectures for both auditing use cases. We release our code here: https://github.com/richterleo/Auditing_Test_for_LMs. 

## 2 BACKGROUND 2.1 MEASURING LM BEHAVIORS 

A goal of LM alignment is to reduce undesired and harmful behaviors, so that they may be more aligned with societal values (Shalev-Shwartz et al., 2020; Hendrycks et al., 2021; Ngo et al., 2022; Wolf et al., 2023). This has spurred the creation of _behavior scoring functions_ (Ji et al., 2023) that measure properties of LM outputs such as toxicity or fact-groundedness (Vidgen et al., 2020; Nozza et al., 2021; Monea et al., 2024).[1] Let _B_ be such a behavior scoring function that assigns scores in the range [0 _,_ 1] where 1 represents the full manifestation of the behavior and 0 indicates its absence. 

1For example, models detecting such undesired behaviors are part of Azure’s Prompt Shield service (Microsoft, 2024). 

2 

Preprint 

These scoring functions evaluate a generated string with respect to the desired behavior. We use the concept of an LM behavior generically, including capabilities and performance on tasks.[2] Our work will make use of recent results in sequential hypothesis testing, which we review below. 

## 2.2 ANYTIME-VALID HYPOTHESIS TESTING 

Sequential hypothesis testing allows one to analyze data without fixing the sample size in advance (Wald, 1945). The benefit of this over classical hypothesis testing is that it is potentially more sample efficient if significant results exist (Arrow et al., 1949). However, as the number of tests increases, the Type I error rate (i.e., false positive rate) increases with it (Jennison & Turnbull, 1999). While there have been many approaches to address this, a recent promising framework that avoids inflating the Type I error rate while remaining statistically efficient is _anytime-valid hypothesis testing_ (Robbins, 1970; Ramdas et al., 2023). 

The fundamental idea behind this framework is the principle of _testing by betting_ , inspired by game theory (Shafer, 2021). In this paradigm, evidence against the null hypothesis **H0** is represented as the gain in wealth of a bettor wagering on observed samples. Before observing new samples, the bettor “buys” a test statistic at the “price” of its expected value under **H0** . After new samples are obtained, the wealth of the bettor is multiplied by the ratio between the actual observed test statistic and its expectation. This ratio is referred to as the _betting score St_ . The bettor reinvests in subsequent “rounds" (i.e., as new data is observed), and the observed betting scores are repeatedly multiplied, leading to a cumulative wealth process. Under **H0** , no betting strategy can consistently increase the bettor’s wealth, ensuring control over the Type I error rate (Ramdas et al., 2023). 

Let the bettor’s (non-negative) wealth after _t_ (batches of) observations be _Wt_ . In order to design a test from this wealth process we require that _Wt_ satisfies the following 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0003-06.png)


All non-negative stochastic processes _Wt_ that satisfy the above condition are called an **e-process** for **H0** (Howard et al., 2021). This states that the maximum wealth across all bets cannot exceed 1 if the null hypothesis **H0** is true.[3] Given an e-process, the test is constructed as follows: reject the null **H0** at some time _τ_ if _Wτ ≥ γ_ , where _γ_ = _α[−]_[1] is a threshold defined by a desired significance level _α ∈_ (0 _,_ 1). Under **H0** , the e-process _Wt_ controls the Type I error rate. By Ville’s inequality (Ville, 1939), we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0003-08.png)


This ensures that the probability of incorrectly rejecting **H0** is at most _α_ at any time step. Thus, the sequential test is _anytime-valid_ , maintaining error control at any stopping point. 

2.3 DEEP ANYTIME-VALID HYPOTHESIS TESTING (DAVT) 

Pandeva et al. (2024) present a general framework called **Deep Anytime-Valid Testing** (DAVT) for designing powerful sequential non-parametric tests by integrating deep learning models into the anytime-valid hypothesis testing framework. Applying DAVT to a variety of tasks, including two-sample testing, Pandeva et al. (2024) demonstrate DAVT’s competitive performance compared to other state-of-the-art non-parametric sequential tests, such as the E-C2ST (Lhéritier & Cazals, 2018) and Seq-IT (Podkopaev & Ramdas, 2024). DAVT uses a model _ϕ_ , trained on past observations, to produce an optimized betting score on new data. DAVT also provides tight control over the Type I error rate. It is also consistent under the assumptions of Pandeva et al. (2024, Proposition 4.3), i.e., the power of the test converges to 1 as the number of samples goes to infinity. 

## 3 METHODS 

To motivate Behavior Shift Auditing (BSA), we detail an external auditing example in Figure 1. We propose an anytime-valid hypothesis test for BSA that will have guarantees on its false positive 

> 2Our proposed method can also be applied to detect _concept drift_ (see Bayram et al. (2022) for a review). 

> 3It can be shown that the wealth process _Wt_ defined this way is equivalent to the minimum wealth a bettor can obtain across all _P ∈_ **H0** (Ramdas et al., 2023). 

3 

Preprint 

rate and is consistent under weak assumptions. Building upon the two-sample variant of DAVT (Pandeva et al., 2024), our test introduces a customizable tolerance parameter _ϵ_ that allows users to specify what constitutes a practically significant difference between distributions, accommodating small, insignificant variations. This approach diverges from prior sequential tests that check for exact distribution equality (Ramdas et al., 2023; Shekhar & Ramdas, 2023; Pandeva et al., 2024), which may be overly sensitive for our use cases. We describe the test in full generality in Appendix B. To focus the text and avoid notational complexity, we concentrate here on the application to behavioral shift detection in LMs. 

## 3.1 AUDITING TEST 

Let _X_ be a random variable representing a prompt, _X_ the set of possible prompts, and **x** _∈X_ a realization of _X_ . A language model is a stochastic operator _M_ that maps prompts **x** to generations **y** _∈Y_ . A behavior function _B_ takes a prompt and generation as input[4] and produces a score _B_ ( **x** _,_ **y** ) _∈_ [0 _,_ 1] _[d]_ (Perez et al., 2023; Wolf et al., 2024) where _d_ is any positive integer. The prompts and language model induce a _behavior distribution PB[M]_[over behavior scores] _[ B]_[(] _[X, M]_[(] _[X]_[))][.][We can] now frame the question of whether the behavior of a model _M[′]_ has changed (substantially) relative to a baseline model _M_ as a testing problem: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0004-04.png)


where _ϵ ≥_ 0 is a tolerance parameter, and _D_ is a distance measure between probability distributions.[5] Note that equality in the null hypothesis in eq. (3) corresponds to DAVT (Pandeva et al., 2024). To extend this to the composite case, our goal is to _construct an appropriate wealth process Wt_ . This will allow us to establish error rate and consistency guarantees. To do so, we will define a betting score _St_ such that it produces a wealth process _Wt_ that is an e-process i.e., it satisfies eq. (1). This, in turn, will depend on the distance measure _D_ that we choose. 

Given a batch of prompts _x_ 1 _, . . . , xb_ and the distance threshold _ϵ_ from Equation (3), we propose the _betting score_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0004-07.png)


where _ϕt−_ 1 is a neural network trained on all ( _t−_ 1) previous batches to optimize the objective 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0004-09.png)


Given the betting score _St_ , we define the _wealth process {Wt}t≥_ 1 of a bettor by initializing their wealth as _W_ 0 = 1 and updating 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0004-11.png)


If the betting score _St_ is an e-variable, meaning that E **H0** [ _St_ ] _≤_ 1, then the wealth process _{Wt}t≥_ 0 is an e-process, which we can prove by induction. Under **H0** , and for any fixed _PB[M][, P] B[ M][ ′]_ satisfying _D_ Φ( _PB[M][, P] B[ M][ ′]_[)] _[ ≤][ϵ]_[,] _[ W][t][−]_[1][ and] _[ S][t]_[ are independent.][Therefore,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0004-13.png)


By induction, E **H0** [ _Wt_ ] _≤_ 1 for all _t ≥_ 0. 

To ensure that _St_ is indeed an e-variable, we choose an appropriate distance measure in eq. (3). Specifically, we define this distance based on the restricted class of models _ϕ_ used in our test. As in (Pandeva et al., 2024), we make the following assumptions on _ϕ_ : 

**Assumption 1** (Pandeva et al. (2024)) **.** _The model class used in our test_ Φ = _{ϕθ_ : _θ ∈_ Θ _} must satisfy the following properties:_ 

> 4We include the prompt for generality, there is no requirement that _B_ must depend on the prompt. 

> 5Testing difference in distributions can be more informative than testing differences in mean or variance, as Figure 10 in the Appendix illustrates. 

4 

Preprint 

- _For all ϕ ∈_ Φ _and for all s ∈_ [0 _,_ 1] _, |ϕ_ ( _s_ ) _| ≤ q for some q ∈_ (0 _,_ 1 _/_ 2) _._ 

- _If ϕ ∈_ Φ _, then c · ϕ ∈_ Φ _for every c ∈_ [ _−_ 1 _,_ 1] _._ 

We can now define the distance measure used in our test. 

**Definition 1** (Neural Net Distance) **.** _Define the distance_[6] _used in eq. (3) to be_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0005-05.png)


For this distance, _St_ is an e-variable (see Appendix B.1.2 for a proof). We can now define the following **sequential test** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0005-07.png)


Control over the Type I error follows again from Ville’s inequality (2). The test is consistent under the following assumptions. 

**Proposition 1.** _If the learning algorithm satisfies the condition_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0005-10.png)


_for all PB[M][, P] B[ M][ ′] with D_ Φ( _PB[M][, P] B[ M][ ′]_[)] _[ > ϵ][ and for a universal constant][ c][, then we have] P_ **H0** ( _γ < ∞_ ) _≤ α and P_ **H1** ( _γ < ∞_ ) = 1 (9) 

For the proof, see Appendix B.1.2. This sequential test is thus a _sequential level-α test of power one_ . 

## 3.2 ALGORITHM 

The auditing test (shown in Algorithm 1) takes in a stream of prompts _{_ **x** _t}t≥_ 1, a behavior function _B_ , an initial baseline language model _M_ , a second language model _M[′]_ , the _α_ -level, a neural net model initialization _ϕ_ 0, and a tolerance parameter _ϵ_ , representing the maximal neural net distance we want to accept between behavior distributions. At every time step, a new prompt from the stream **x** _t_ is fed to both _M_ and _M[′]_ to create generations, which are then scored by the behavior function. We feed these scores to the neural net model _ϕt−_ 1 and calculate the betting score _St_ . Next, we update the wealth _Wt_ by the betting score and check whether it surpasses the 1 _/α_ -threshold, in which case we reject the null hypothesis. If not, we update the neural net model in a separate training step and continue with the next prompt. The algorithm can easily be modified to accept batches instead of single prompts.[7] 

**Algorithm 1** Auditing Test 

1: **Input:** _{_ **x** _t}t≥_ 1 (stream of prompts), _B_ (behavior function), _M_ (baseline model API), _M[′]_ (current model API), _α_ (type-I error limit under null), _ϕ_ 0 (neural net model for testing), _ϵ_ (maximal neural net distance) 2: _W_ 0 _←_ 1 3: **while** true **do** 4: Compute behavior scores: _bt ← B_ ( **x** _t, M_ ( **x** _t_ )) _, b[′] t[←][B]_[(] **[x]** _[t][, M][ ′]_[(] **[x]** _[t]_[))] 5: Compute betting score: _St ←_[(][1+] _[ϕ][t][−]_[1][(] exp( _[b][t]_[)] _[−] ϵ_ ) _[ϕ][t][−]_[1][(] _[b] t[′]_[))] 6: Update wealth: _Wt ← Wt−_ 1 _× St_ 7: **if** _Wt ≥_ 1 _/α_ **then** 8: Break and reject null 9: **end if** 10: Update neural net model: _t ϕt ←_ arg max _ϕ_ � _l_ =1[log(1 +] _[ ϕ]_[(] _[b][t]_[)] _[ −][ϕ]_[(] _[b] t[′]_[))] 11: **end while** 

> 6This distance is an instance of an _integral probability metric_ (IPM) (Müller, 1997), a class of distances that includes well-known metrics like the Wasserstein distance (Kantorovich & Rubinstein, 1958). IPMs are at least pseudo-metrics i.e., they satisfy all the properties of a metric except that the distance between distinct points can be zero. 

> 7In this case, the new betting score _St_ is calculated as a product over samples in the batch. 

5 

Preprint 

|Mean toxicity<br>0.<br>Wasserstein distance<br>0.<br>Neural net distance<br>0.|07<br>0.08<br>0.09<br>0.10<br>0.11<br>00<br>0.01<br>0.02<br>0.04<br>0.05<br>00<br>0.02<br>0.04<br>0.06<br>0.08<br>Sampling variation<br>Instruction tuning<br>Uncensored Llama|
|---|---|



Figure 2: **Measuring Mean and Distributional Change** . Analysis of seven Llama3-8B variants shows aligned shifts across three metrics: mean toxicity scores, Wasserstein distances, and Neural net distances to baseline Llama3-8B-Instruct. The variants include the baseline model with modified sampling parameters, five models instruction-tuned on subsets of SuperNI, and an uncensored model. 

## 4 EXPERIMENTS 

We evaluate our test for both external and internal auditing use-cases. We then evaluate cases where small changes in distribution are allowed when auditing externally for toxicity, and internally for translation performance. 

## 4.1 EXACT TEST, _ϵ_ = 0 

**Setup.** We begin by investigating an external setting where we require the test to detect any change in distribution ( _ϵ_ = 0). Specifically, we will check for changes in toxicity behavior. We select prompts from the REALTOXICITYPROMPTS dataset (Gehman et al., 2020) and use the toxicity behavior function from Perspective API (Lees et al., 2022) to evaluate LM generations. Llama3 (8B-Instruct) (Llama-team, 2024), Gemma (1.1-7b-it) (Mesnard et al., 2024), and Mistral (7B-Instruct-v0.2) (Jiang et al., 2023) serve as our initial aligned models. We remove the safety alignment in these models by fine-tuning, producing 10 corrupted checkpoints for each model. To evaluate the statistical properties of our the exact test ( _ϵ_ =0), we assess (a) its ability to detect changed checkpoints, and (b) its false positive rate. For further experimental details regarding toxicity fine-tuning, text generation and the betting score network, please see Appendix A. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0006-07.png)


**----- Start of picture text -----**<br>
1.0<br>0.9<br>0.8<br>0.7 distance<br>0.6 0.001<br>0.5 0.031<br>0.036<br>0.4 0.037<br>0.3 0.039<br>0.041<br>0.2 0.043<br>0.1 0.045<br>0.0<br>0 250 500 750 1000 1250 1500 1750<br>samples<br>proportion of triggered tests<br>**----- End of picture text -----**<br>


Figure 3: **Fine-tuning Detection for Llama-38B-Instruct.** The detection frequency as a function of number of generated samples. Each curve is a fine-tuned corrupted model checkpoint (to simplify visualization, the curves with shaded standard deviations are averages over models with similar distances to the aligned model). The color depicts the Wasserstein distance between the corrupted model and the original aligned model. 

**Corrupted model detection.** We test each corrupted checkpoint against the corresponding initial aligned model with _α_ =0 _._ 05. Figure 3 shows the fraction of positive test results after having observed at least _m_ samples, with tests repeated 48 times per checkpoint (2000 samples per fold, batch size 100). High detection rates of almost 80% are achieved even for checkpoints closest to the baseline. We find that as the distance between the corrupted model and the initial model increases, fewer samples are needed to detect the change in behavior. Similar results for Mistral and Gemma can be found in Appendix C. 

**False positive rate.** We use different random seeds for generating text from the initial aligned models to examine the false positive rate of the exact test. Figure 4 shows the false positive rate for each of the model architectures as a function of the number of observed samples, repeated 24 times (4000 samples per fold, batch size 100). The test is highly specific, with false detection rates consistently below 0 _._ 04. 

6 

Preprint 

## 4.2 TOLERANCE TEST, _ϵ >_ 0 

We now evaluate the test with tolerance _ϵ >_ 0 in two use-cases: an external toxicity audit, and an internal translation performance audit. We describe strategies for determining an appropriate _ϵ_ in each scenario. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0007-03.png)


**----- Start of picture text -----**<br>
0.05<br>0.04<br>0.03<br>0.02<br>0.01 Meta-Llama-3-8B-InstructMistral-7B-Instruct-v0.2<br>gemma-1.1-7b-it<br>0.00<br>0 500 1000 1500 2000 2500 3000 3500 4000<br>samples<br>proportion of triggered tests<br>**----- End of picture text -----**<br>


## USE CASE 1: EXTERNAL AUDIT, TOXICITY 

**Setup.** We simulate an external auditor checkgemma-1.1-7b-it 0.00 ing whether instruction-tuning an aligned model 0 500 1000 1500 2000 2500 3000 3500 4000 on unrelated tasks affected toxicity distributions, samples something that has been observed in practice (Qi Figure 4: **False positives.** The false positive rate et al., 2023). We use Llama3 (8B-Instruct) for each of the model architectures as a function as the aligned model, again evaluating toxicof number of observed samples. ity on the REALTOXICITYPROMPTS dataset (Gehman et al., 2020) using Perspective API (Lees et al., 2022). We instruction-tune Llama3 on 5 different task clusters from SUPERNATURALINSTRUCTIONS (SuperNI; Mishra et al., 2022b; Wang et al., 2022). This setup is inspired by Wang et al. (2023b), who found that a pre-trained Llama2 model instruction-tuned on SuperNI exhibits high toxicity scores on ToxiGen. Detailed information on instruction-tuning and how the neural net distance is estimated can be found in Appendix A. 

**Results.** Instruction-tuning increased mean toxicity scores, which, as shown in Figure 2, corresponds with increases in both Wasserstein distances and neural net distances from Llama3. As a reference, we also include another Llama3-8B model tuned to be less refusing.[8] Surprisingly, the most toxic and distant model is not this uncensored model but the model fine-tuned on Code to Text, Stereotype Detection, and Sentence Perturbation (shown in green). We test Llama3 against each instruction-tuned model across a range of tolerance values, from _ϵ_ =0 _._ 0038 (the neural net distance between standard Llama3 and Llama3 with different sampling parameters) up to the neural net distance between the base model and another Llama3-8B model tuned to be less refusing, _ϵ_ =0 _._ 076. 

Figure 5 shows the proportion of tests where the fine-tuned model was identified as different from the baseline across various test epsilon values. At lower epsilon values, representing a conservative testing regime that detects even small changes, all instruction-tuned models are consistently identified (100% detection rate). As epsilon increases, the power of the test decreases until it reaches the true neural net distance between the base model and each fine-tuned variant. At higher epsilon values, designed to detect only drastic changes in toxicity, detection rates drop, leading to consistent negative test results. 

We investigate the strict auditing setting – where only minor variations due to sampling are accepted – more closely. Specifically, we set _ϵ_ equal to the neural net distance between the original Llama3 model and the same model with different sampling parameters ( _ϵ_ = 0 _._ 0038) and test baseline Llama3 against the 5 instruction-tuned versions as well as the uncensored reference Llama3. Figure 6 demonstrates that under this strict threshold, the test requires fewer samples to detect models that deviate more substantially from the baseline. 

## USE CASE 2: INTERNAL AUDIT, TRANSLATION PERFORMANCE 

We simulate a modeler adjusting their language model while monitoring whether its translation capabilities change substantially. To fix a tolerance parameter _ϵ_ we imagine that the modeler only wishes to trigger the test if the translation distribution changes by more than the amount it would if prompted differently. 

**Setup.** We evaluate Llama3 (8B-Instruct) on English-Spanish and English-French translations from SuperNI. We set _ϵ_ as the neural net distance between Llama3 using simple prompts, and 

> 8The uncensored model was fine-tuned on Uncensored-Vortex https://huggingface.co/ datasets/OEvortex/uncensored-vortex. 

7 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0008-01.png)


**----- Start of picture text -----**<br>
1.0 Program Execution, Pos Tagging,<br>Mathematics<br>0.9 Gender Classification, Commonsense<br>Classification, Translation<br>0.8 Code to Text, Stereotype Detection,<br>Sentence Perturbation<br>0.7 Text to Code, Linguistic Probing,<br>Language Identification<br>0.6 Data to Text, Word Semantics,<br>Question Rewriting<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08<br>test epsilon<br>proportion of triggered tests<br>**----- End of picture text -----**<br>


Figure 5: **Detection rate vs. Test Epsilon.** Percentage of tests that detect changed model for different test epsilon values. 

Llama3 using few-shot prompts. We then test the translation performance distribution of Llama3 with simple prompts against that of Aya-23-8B (Üstün et al., 2024), a multilingual instruction-tuned model. We expect a positive test result since Aya-23-8B represents a significant improvement in translation capabilities compared to Llama3, likely exceeding the threshold _ϵ_ set by different prompting techniques. 

**Results.** Few-shot prompting leads to a modest increase in mean BLEU scores from 0 _._ 1683 to 0 _._ 1765. A significant improvement is evident when using Aya-23-8b, with a mean BLEU score of 0 _._ 2970. We observe that Llama3 models occasionally misinterpret instructions or include unnecessary additional text in English, potentially impacting their scores. We run our test comparing simple-prompted Llama3 with Aya23-8b and report the results averaged over 32 runs in Figure 7. The test detects a difference in nearly all cases after only 100 samples. 

Overall, the results from both the toxicity and translation audits demonstrate the effectiveness and sample-efficiency of our testing method in detecting behavioral shifts in language models. In the external audit, it consistently identified increases in toxicity levels due to instructiontuning, especially at lower epsilon values, confirming its sensitivity to subtle changes in model behavior. Similarly, in the internal audit, it ef- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0008-06.png)


**----- Start of picture text -----**<br>
1.0<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4 Program Execution, Pos Tagging, Mathematics<br>0.3 Gender ClassificationCode to Text, Stereotype Detection, Sentence Perturbation, Commonsense Classification, Translation<br>0.2 Text to Code, Linguistic Probing, Language Identification<br>0.1 Data to Text, Word Semantics, Question Rewriting<br>0.0 LLama-3-8b-Uncensored<br>0 250 500 750 1000 1250 1500 1750<br>samples<br>proportion of triggered tests<br>**----- End of picture text -----**<br>


Figure 6: **Detection Rates for Fine-Tuned Models.** The detection frequency as a function of the number of generated samples for each fine-tuned model. We used a test with _ϵ ≈_ 0 _._ 0038, based on the estimated neural net distance between distributions generated by Llama-3-8B-Instruct using different sampling parameters. The black line represents an unaligned reference model, Llama3-8B trained to be more permissive in answering. 

fectively detected significant differences in BLEU score distributions between the standard Llama3, the few-shot prompted Llama3, and Aya-23-8b, highlighting its utility across different tasks. These findings underscore the importance of selecting an appropriate tolerance level based on the specific application to balance sensitivity and practicality. 

## 5 RELATED WORK 

**LM Behavior Evaluation.** Some of the earliest work in the field of AI Alignment used curated datasets and behavior evaluation functions to assess alignment (Bolukbasi et al., 2016; Parrish et al., 2022). This work builds off of a long tradition in NLP on the evaluation of text generation (Celikyilmaz et al., 2020). While the curated datasets are often high quality, they tend to be small, spurring the construction of larger datasets through web scraping (Zhao et al., 2018; Zampieri et al., 2019; Nangia et al., 2020; Rosenthal et al., 2021) and even using other LMs (Zhang et al., 2022; Perez 

8 

Preprint 

et al., 2023). Meanwhile, early work on behavior functions focused on measuring bias, toxicity, and hallucinations (Achiam et al., 2023; Anil et al., 2023; Chern et al., 2023; Varshney et al., 2023; Llamateam, 2024). Since the rise of LMs with human-level performance, the set of behavior functions has exploded (Zou et al., 2023). It has become more nuanced, including complex characteristics such as power-seeking behavior (Park et al., 2023; Sharma et al., 2023), situational awareness (Zou et al., 2023), and deception (Hagendorff, 2024). However, even with access to massive datasets and carefully constructed behavior functions it can be difficult to discover these behaviors from static inputs (Kalin et al., 2020). To address this, Perez et al. (2022) introduced the notion of _red-teaming_ for LM alignment. This allows prompts to be adversarially-constructed to expose failure cases, which arise in many state-of-the-art models (Chao et al., 2023). The testing procedure we present here is agnostic to both the prompts and the behavior functions used in LM evaluation. 

**Model Change Identification.** For the case where one wishes to identify _any change_ in model behavior (i.e., _ϵ_ = 0) there are multiple other techniques that can be used. The first set uses ideas from _formal verification_ to ensure that the predictions from a model are guaranteed to come from a specific model (Ghodsi et al., 2017; Dong et al., 2021; Fan et al., 2023; Weng et al., 2023). In general, however, these methods are computationally intensive and do not scale to state-of-the-art LMs. A second, more efficient idea is to _watermark_ the model (Zhu et al., 2018; Amrit & Singh, 2022; He et al., 2022a;b; Kirchenbauer et al., 2023; Kuditipudi et al., 2023; Yoo et al., 2023). The idea is to embed signals into model generations that can be detected algorithmically. However, watermarks are often inserted by the model owner (Kirchen- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0009-03.png)


**----- Start of picture text -----**<br>
1.0<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>0 250 500 750 1000 1250 1500 1750 2000<br>samples<br>proportion of triggered tests<br>**----- End of picture text -----**<br>


Figure 7: **Detection for Aya-23-8b** . The detection frequency as a function of the number of generated samples when setting _ϵ ≈_ 0 _._ 0072. This threshold is derived as an estimate of the neural net distance between Llama-3-8B-Instruct with and without few-shot prompts. 

bauer et al., 2023; Kuditipudi et al., 2023), allowing them (or an actor that has compromised the model) to insert it into any model that is being audited. This precludes its use for many external auditing settings. For internal auditing, a watermark may break under a small model change that is acceptable. Our work is also related to work on concept drift (Bayram et al., 2022) and prompt stability (Li et al., 2024a). In principle our test can be used to detect concept and generation changes, however the focus of these works is on model performance and generation similarity, as opposed to behavior change. 

## 6 DISCUSSION 

In this work we introduce the idea of Behavioral Shift Auditing (BSA): detecting LM behavior changes over time. Our proposed test comes with guarantees and has been able to detect changes in language model toxicity and translation performance. One of the notable strengths of our approach is its sample efficiency. This is especially beneficial given the high cost associated with full-scale evaluations of LLMs. Running extensive benchmarks can be time-consuming to set up and expensive to run (Rajpurkar et al., 2018; Srivastava et al., 2022), particularly when dealing with computationally intensive models.[9] Our test can serve as a screening tool to identify potential behavioral shifts using just a few hundred samples, making subsequent full-scale evaluations more targeted and efficient. Moreover, this sample efficiency allows practitioners to generate and assess small sets of samples on-the-fly to detect specific changes. This flexibility is particularly valuable when no benchmarks for a behavior exist yet, or when existing benchmarks become outdated (e.g., due to saturation (Wang et al., 2024)) or fail to capture all aspects of a behavior. 

We now discuss some current limitations. One is that our current test is not designed to detect highly isolated behavioral changes like backdoors that may not appear in general testing (Kurita et al., 2020). This limitation is inherited from framing BSA as hypothesis testing. An interesting avenue of future 

> 9E.g., inference-heavy models like ChatGPT o1-preview (OpenAI, 2024). 

9 

Preprint 

work is to investigate if one can frame BSA differently to capture behavioral changes that are tricky to find and measure, including things such as deceptive alignment (Carranza et al., 2023). 

There are many other exiting directions for future research. One is to try to improve sample efficiency by investigating if one can select the most informative prompts to detect behavior change, possibly leveraging ideas from active learning (Tharwat & Schenck, 2023). Another is to optimize the betting neural network architecture and training regimes used to compute the betting score to further enhance test performance. Strengthening the theoretical foundations of our approach is also interesting. Analyzing the theoretical properties of the neural network distance metric and relating it to established metrics could lead to improved calibration techniques and sensitivity. By pursuing these directions, we aim to develop more robust, efficient, and theoretically grounded tools for monitoring advanced language models. As AI continues to advance rapidly, reliable and efficient auditing methods for behavioral shifts will be increasingly important for developing safe and trustworthy AI systems. 

## ACKNOWLEDGEMENTS 

This work was supported by the Edinburgh International Data Facility (EIDF) and the Data-Driven Innovation Programme at the University of Edinburgh. PM was partially funded by ELIAI (The Edinburgh Laboratory for Integrated Artificial Intelligence), EPSRC (grant no. EP/W002876/1), an industry grant from Cisco, and a donation from Accenture LLP. LR was supported by the EPSRC Grant EP/S021566/1. We want to thank Robert Kirk and Ole Jorgensen for feedback on earlier versions of the paper. 

## REFERENCES 

- Abubakar Abid, Maheen Farooqi, and James Zou. Persistent anti-muslim bias in large language models. In _Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society_ , pp. 298–306, 2021. 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Preetam Amrit and Amit Kumar Singh. Survey on watermarking methods in the artificial intelligence domain and beyond. _Computer Communications_ , 188:52–65, 2022. 

- Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- Kenneth J Arrow, David Blackwell, and Meyer A Girshick. Bayes and minimax solutions of sequential decision problems. _Econometrica, Journal of the Econometric Society_ , pp. 213–244, 1949. 

- Firas Bayram, Bestoun S Ahmed, and Andreas Kassler. From concept drift to model degradation: An overview on performance-aware drift detectors. _Knowledge-Based Systems_ , 245:108632, 2022. 

- Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. _Advances in neural information processing systems_ , 29, 2016. 

- Andres Carranza, Dhruv Pai, Rylan Schaeffer, Arnuv Tandon, and Sanmi Koyejo. Deceptive alignment monitoring. _arXiv preprint arXiv:2307.10569_ , 2023. 

- Asli Celikyilmaz, Elizabeth Clark, and Jianfeng Gao. Evaluation of text generation: A survey. _arXiv preprint arXiv:2006.14799_ , 2020. 

- Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries. In _R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models_ , 2023. 

10 

Preprint 

- I Chern, Steffi Chern, Shiqi Chen, Weizhe Yuan, Kehua Feng, Chunting Zhou, Junxian He, Graham Neubig, Pengfei Liu, et al. Factool: Factuality detection in generative ai–a tool augmented framework for multi-task and multi-domain scenarios. _arXiv preprint arXiv:2307.13528_ , 2023. 

Ben Chugg, Santiago Cortes-Gomez, Bryan Wilder, and Aaditya Ramdas. Auditing fairness by betting. _Advances in Neural Information Processing Systems_ , 36:6070–6091, 2023. 

- Boxiang Dong, Bo Zhang, and Hui Wang. Veridl: Integrity verification of outsourced deep learning services. In _Joint European Conference on Machine Learning and Knowledge Discovery in Databases_ , pp. 583–598, 2021. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- Yongkai Fan, Binyuan Xu, Linlin Zhang, Jinbao Song, Albert Zomaya, and Kuan-Ching Li. Validating the integrity of convolutional neural network predictions based on zero-knowledge proof. _Information Sciences_ , 625:125–140, 2023. 

- Ed Felten, Manav Raj, and Robert Seamans. How will language modelers like chatgpt affect occupations and industries? _arXiv preprint arXiv:2303.01157_ , 2023. 

- Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith. Realtoxicityprompts: Evaluating neural toxic degeneration in language models. _arXiv preprint arXiv:2009.11462_ , 2020. 

- Zahra Ghodsi, Tianyu Gu, and Siddharth Garg. Safetynets: Verifiable execution of deep neural networks on an untrusted cloud. _Advances in Neural Information Processing Systems_ , 30, 2017. 

- Thilo Hagendorff. Deception abilities emerged in large language models. _Proceedings of the National Academy of Sciences_ , 121(24):e2317967121, 2024. 

- Xuanli He, Qiongkai Xu, Lingjuan Lyu, Fangzhao Wu, and Chenguang Wang. Protecting intellectual property of language generation apis with lexical watermark. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 36(10):10758–10766, Jun. 2022a. doi: 10.1609/aaai.v36i10.21321. URL https://ojs.aaai.org/index.php/AAAI/article/view/21321. 

- Xuanli He, Qiongkai Xu, Yi Zeng, Lingjuan Lyu, Fangzhao Wu, Jiwei Li, and Ruoxi Jia. CATER: Intellectual property protection on text generation APIs via conditional watermarks. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), _Advances in Neural Information Processing Systems_ , 2022b. URL https://openreview.net/forum?id= L7P3IvsoUXY. 

- Dan Hendrycks, Nicholas Carlini, John Schulman, and Jacob Steinhardt. Unsolved problems in ml safety. _arXiv preprint arXiv:2109.13916_ , 2021. 

- Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. _Neural networks_ , 2(5):359–366, 1989. 

- Steven R Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. _The Annals of Statistics_ , 49(2), 2021. 

- Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ , 2021. 

- Christopher Jennison and Bruce W Turnbull. _Group sequential methods with applications to clinical trials_ . CRC Press, 1999. 

- Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan, Zhonghao He, Jiayi Zhou, Zhaowei Zhang, et al. Ai alignment: A comprehensive survey. _arXiv preprint arXiv:2310.19852_ , 2023. 

11 

Preprint 

Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. Beavertails: Towards improved safety alignment of llm via a human-preference dataset. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- Josh Kalin, Matthew Ciolino, David Noever, and Gerry Dozier. Black box to white box: Discover model characteristics based on strategic probing. In _2020 Third International Conference on Artificial Intelligence for Industries (AI4I)_ , pp. 60–63. IEEE, 2020. 

- LV Kantorovich and GS Rubinstein. On a space of completely additive functions, vestn. leningr. univ. 13 (7)(1958) 52-59, 1958. 

- Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke Hüllermeier, et al. Chatgpt for good? on opportunities and challenges of large language models for education. _Learning and individual differences_ , 103:102274, 2023. 

- Daniel Martin Katz, Michael James Bommarito, Shang Gao, and Pablo Arredondo. Gpt-4 passes the bar exam. _Philosophical Transactions of the Royal Society A_ , 382(2270):20230254, 2024. 

- John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. _arXiv preprint arXiv:2301.10226_ , 2023. 

- Hadas Kotek, Rikker Dockum, and David Sun. Gender bias and stereotypes in large language models. In _Proceedings of the ACM collective intelligence conference_ , pp. 12–24, 2023. 

- Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, and Percy Liang. Robust distortion-free watermarks for language models. _arXiv preprint arXiv:2307.15593_ , 2023. 

- Keita Kurita, Paul Michel, and Graham Neubig. Weight poisoning attacks on pre-trained models. _arXiv preprint arXiv:2004.06660_ , 2020. 

- Alyssa Lees, Vinh Q Tran, Yi Tay, Jeffrey Sorensen, Jai Gupta, Donald Metzler, and Lucy Vasserman. A new generation of perspective api: Efficient multilingual character-level transformers. In _Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining_ , pp. 3197–3207, 2022. 

- Alix Lhéritier and Frédéric Cazals. A sequential non-parametric multivariate two-sample test. _IEEE Transactions on Information Theory_ , 64(5):3361–3370, 2018. 

- Kenneth Li, Tianle Liu, Naomi Bashkansky, David Bau, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Measuring and controlling instruction (in)stability in language model dialogs. In _COLM_ , 2024a. 

Yuxi Li, Yi Liu, Gelei Deng, Ying Zhang, Wenjia Song, Ling Shi, Kailong Wang, Yuekang Li, Yang Liu, and Haoyu Wang. Glitch tokens in large language models: categorization taxonomy and effective detection. _Proceedings of the ACM on Software Engineering_ , 1(FSE):2075–2097, 2024b. 

Llama-team. Introducing meta llama 3: The most capable openly available llm to date. https: //ai.meta.com/blog/meta-llama-3/, 2024. Accessed: 2024-05-15. 

- David Lopez-Paz and Maxime Oquab. Revisiting classifier two-sample tests. In _International Conference on Learning Representations_ , 2017. 

- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Conference on Learning Representations_ , 2018. 

- Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery. _arXiv preprint arXiv:2408.06292_ , 2024. 

12 

Preprint 

Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. _arXiv preprint arXiv:2403.08295_ , 2024. 

- Microsoft. Azure AI content safety - prompt shields. https://azure.github.io/ Azure-AI-Content-Safety-Private-Preview/Prompt%20Shields.html, 2024. Accessed: 2024. 

- Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, et al. Lila: A unified benchmark for mathematical reasoning. _arXiv preprint arXiv:2210.17517_ , 2022a. 

- Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. In _ACL_ , 2022b. 

- Giovanni Monea, Maxime Peyrard, Martin Josifoski, Vishrav Chaudhary, Jason Eisner, Emre Kıcıman, Hamid Palangi, Barun Patra, and Robert West. A glitch in the matrix? locating and detecting language model grounding with fakepedia. In _ACL 2024_ , 2024. 

- Alfred Müller. Integral probability metrics and their generating classes of functions. _Advances in applied probability_ , 29(2):429–443, 1997. 

- Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel Bowman. Crows-pairs: A challenge dataset for measuring social biases in masked language models. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 1953–1967, 2020. 

- Richard Ngo, Lawrence Chan, and Sören Mindermann. The alignment problem from a deep learning perspective. _arXiv preprint arXiv:2209.00626_ , 2022. 

- Debora Nozza, Federico Bianchi, Dirk Hovy, et al. Honest: Measuring hurtful sentence completion in language models. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ . Association for Computational Linguistics, 2021. 

OpenAI. Gpt-3.5-turbo. https://platform.openai.com/docs/models/gpt-3-5, 2023. Accessed: 27/09/2024. 

OpenAI. Introducing openai O(1) preview. https://openai.com/blog/ introducing-openai-o1-preview/, 2024. Accessed: 2024-10-23. 

- Teodora Pandeva, Patrick Forré, Aaditya Ramdas, and Shubhanshu Shekhar. Deep anytime-valid hypothesis testing. In _AISTATS_ , volume 238 of _Proceedings of Machine Learning Research_ , pp. 622–630. PMLR, 2024. 

- Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In _Proceedings of the 36th annual acm symposium on user interface software and technology_ , pp. 1–22, 2023. 

- Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel Bowman. Bbq: A hand-built bias benchmark for question answering. In _Findings of the Association for Computational Linguistics: ACL 2022_ , pp. 2086–2105, 2022. 

- Ethan Perez, Saffron Huang, H. Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. Red teaming language models with language models. In _EMNLP_ , pp. 3419–3448. Association for Computational Linguistics, 2022. 

- Ethan Perez, Sam Ringer, Kamile Lukosiute, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. Discovering language model behaviors with model-written evaluations. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pp. 13387–13434, 2023. 

- Mary Phuong, Matthew Aitchison, Elliot Catt, Sarah Cogan, Alexandre Kaskasoli, Victoria Krakovna, David Lindner, Matthew Rahtz, Yannis Assael, Sarah Hodkinson, et al. Evaluating frontier models for dangerous capabilities. _arXiv preprint arXiv:2403.13793_ , 2024. 

13 

Preprint 

- Aleksandr Podkopaev and Aaditya Ramdas. Sequential predictive two-sample and independence testing. _Advances in neural information processing systems_ , 36, 2024. 

- Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! _arXiv preprint arXiv:2310.03693_ , 2023. 

- Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions for squad. _arXiv preprint arXiv:1806.03822_ , 2018. 

- Aaditya Ramdas, Peter Grünwald, Vladimir Vovk, and Glenn Shafer. Game-theoretic statistics and safe anytime-valid inference. _Statistical Science_ , 38(4):576–601, 2023. 

- Herbert Robbins. Statistical methods related to the law of the iterated logarithm. _The Annals of Mathematical Statistics_ , 41(5):1397–1409, 1970. 

- Sara Rosenthal, Pepa Atanasova, Georgi Karadzhov, Marcos Zampieri, and Preslav Nakov. Solid: A large-scale semi-supervised dataset for offensive language identification. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pp. 915–928, 2021. 

- Glenn Shafer. Testing by betting: A strategy for statistical and scientific communication. _Journal of the Royal Statistical Society Series A: Statistics in Society_ , 184(2):407–431, 2021. 

- Shai Shalev-Shwartz, Shaked Shammah, and Amnon Shashua. On the ethics of building ai in a responsible manner. _arXiv preprint arXiv:2004.04644_ , 2020. 

- Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R Bowman, Esin DURMUS, Zac Hatfield-Dodds, Scott R Johnston, Shauna M Kravec, et al. Towards understanding sycophancy in language models. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- Shubhanshu Shekhar and Aaditya Ramdas. Nonparametric two-sample testing by betting. _IEEE Transactions on Information Theory_ , 2023. 

- Shubhanshu Shekhar, Ziyu Xu, Zachary Lipton, Pierre Liang, and Aaditya Ramdas. Risk-limiting financial audits via weighted sampling without replacement. In _Uncertainty in Artificial Intelligence_ , pp. 1932–1941. PMLR, 2023. 

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _arXiv preprint arXiv:2206.04615_ , 2022. 

- Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpaca: A strong, replicable instruction-following model. _Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html_ , 3(6):7, 2023. 

- Alaa Tharwat and Wolfram Schenck. A survey on active learning: State-of-the-art, practical challenges and research directions. _Mathematics_ , 11(4):820, 2023. 

- Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, et al. Aya model: An instruction finetuned open-access multilingual language model. _arXiv preprint arXiv:2402.07827_ , 2024. 

- Neeraj Varshney, Wenlin Yao, Hongming Zhang, Jianshu Chen, and Dong Yu. A stitch in time saves nine: Detecting and mitigating hallucinations of llms by validating low-confidence generation. _arXiv preprint arXiv:2307.03987_ , 2023. 

- Bertie Vidgen, Tristan Thrush, Zeerak Waseem, and Douwe Kiela. Learning from the worst: Dynamically generated datasets to improve online hate detection. _arXiv preprint arXiv:2012.15761_ , 2020. 

- Jean Ville. _Etude critique de la notion de collectif_ . Gauthier-Villars Paris, 1939. 

14 

Preprint 

- A Wald. Sequential tests of statistical hypotheses. _The Annals of Mathematical Statistics_ , 16(2): 117–186, 1945. 

- Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, Sang T. Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, and Bo Li. Decodingtrust: A comprehensive assessment of trustworthiness in GPT models. In _NeurIPS_ , 2023a. 

- Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. Super-naturalinstructions:generalization via declarative instructions on 1600+ tasks. In _EMNLP_ , 2022. 

- Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Chandu, David Wadden, Kelsey MacMillan, Noah A Smith, Iz Beltagy, et al. How far can camels go? exploring the state of instruction tuning on open resources. _Advances in Neural Information Processing Systems_ , 36:74764–74786, 2023b. 

- Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. _arXiv preprint arXiv:2406.01574_ , 2024. 

- Ian Waudby-Smith, Philip B Stark, and Aaditya Ramdas. Rilacs: risk limiting audits via confidence sequences. In _Electronic Voting: 6th International Joint Conference, E-Vote-ID 2021, Virtual Event, October 5–8, 2021, Proceedings 6_ , pp. 124–139. Springer, 2021. 

- Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, et al. Ethical and social risks of harm from language models. _arXiv preprint arXiv:2112.04359_ , 2021. 

- Jiasi Weng, Jian Weng, Gui Tang, Anjia Yang, Ming Li, and Jia-Nan Liu. pvcnn: Privacy-preserving and verifiable convolutional neural network testing. _IEEE Transactions on Information Forensics and Security_ , 18:2218–2233, 2023. 

- Yotam Wolf, Noam Wies, Yoav Levine, and Amnon Shashua. Fundamental limitations of alignment in large language models. _arXiv preprint arXiv:2304.11082_ , 2023. 

- Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, and Amnon Shashua. Fundamental limitations of alignment in large language models. In _Forty-first International Conference on Machine Learning_ , 2024. 

- KiYoon Yoo, Wonhyuk Ahn, Jiho Jang, and Nojun Kwak. Robust multi-bit natural language watermarking through invariant features. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 2092–2115, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.117. URL https: //aclanthology.org/2023.acl-long.117. 

- Travis Zack, Eric Lehman, Mirac Suzgun, Jorge A Rodriguez, Leo Anthony Celi, Judy Gichoya, Dan Jurafsky, Peter Szolovits, David W Bates, Raja-Elie E Abdulnour, et al. Assessing the potential of gpt-4 to perpetuate racial and gender biases in health care: a model evaluation study. _The Lancet Digital Health_ , 6(1):e12–e22, 2024. 

- Marcos Zampieri, Shervin Malmasi, Preslav Nakov, Sara Rosenthal, Noura Farra, and Ritesh Kumar. Predicting the type and target of offensive posts in social media. In _Proceedings of NAACL-HLT_ , pp. 1415–1420, 2019. 

- Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang, Kathleen McKeown, and Tatsunori B. Hashimoto. Benchmarking large language models for news summarization. _Transactions of the Association for Computational Linguistics_ , 12:39–57, 2024. doi: 10.1162/tacl_a_00632. URL https://aclanthology.org/2024.tacl-1.3. 

15 

Preprint 

- Zhexin Zhang, Jiale Cheng, Hao Sun, Jiawen Deng, Fei Mi, Yasheng Wang, Lifeng Shang, and Minlie Huang. Constructing highly inductive contexts for dialogue safety through controllable reverse generation. In _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pp. 3684–3697, 2022. 

- Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Gender bias in coreference resolution: Evaluation and debiasing methods. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)_ , pp. 15–20, 2018. 

- Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei. Hidden: Hiding data with deep networks. In _Proceedings of the European conference on computer vision (ECCV)_ , pp. 657–672, 2018. 

- Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. _arXiv preprint arXiv:2310.01405_ , 2023. 

## A EXPERIMENTAL DETAILS 

## A.1 SETUP 

We assess the efficacy of BSA using three base models: Llama3 (8B-Instruct) (Llama-team, 2024), Gemma (1.1-7b-it) (Mesnard et al., 2024), and Mistral (7B-Instruct-v0.2) (Jiang et al., 2023). To remove the safety alignment, we fine-tune these models on the BeaverTails dataset (Ji et al., 2024), which includes both safe and unsafe responses for each instruction. We use a subset of 50K instances from the dataset, each comprising an instruction paired with its corresponding unsafe response. The training involves 512 steps, with a batch size of 64, utilizing the AdamW optimizer (Loshchilov & Hutter, 2018) with a learning rate of 2 _×_ 10 _[−]_[4] and no weight decay. Due to computational constraints, we apply LoRA (Hu et al., 2021), with a rank of 16, to all models. All experiments were conducted on a single Nvidia A100 (80GB) GPU. 

To simulate a realistic use-case, we further produce 5 versions of Llama3 (8B-Instruct) instruction-tuned on different clusters of task categories from SUPER-NATURALINSTRUCTIONS (SuperNI) Mishra et al. (2022b); Wang et al. (2022). We kept the same training configuration as for toxicity fine-tuning, albeit with a reduced batch size of 8 over 2048 steps, accommodating the smaller memory of an Nvidia A100 (40GB). See table 1 for a summary of the category clusters used. 

Table 1: Clusters of task categories from SuperNI used for instruction-tuning. 

|Program Execution<br>POS Tagging<br>Mathematics|Gender Classifcation<br>Commonsense Classifcation<br>Translation|Code to Text<br>Stereotype Detection<br>Sentence Perturbation|Text to Code<br>Linguistic Probing<br>Language Identifcation|Data to Text<br>Word Semantics<br>Question Rewriting|
|---|---|---|---|---|



As an independent toxic reference model, we used another Llama3 model instruction-tuned on the OEvortex/uncensored-vortex dataset, which we refer to as “Uncensored Llama3-8B". This model was also trained using LoRA with a rank of 16, and trained over 200 steps with a total batch size of 8 and gradient accumulation. 

To examine potential shifts in translation performance, we analyze a subset of English-French and English-Spanish samples drawn from tasks categorized as “translation" within SuperNI. This subset comprises a total of 67,975 prompts. 

## A.2 TOXICITY EVALUATIONS 

We compare toxicity scores across Llama3, Gemma, and Mistral models. 

Using the REALTOXICITYPROMPTS prompts, we generate continuations for each baseline model and their 10 checkpoints, the Llama3 models instruction-tuned on SuperNI clusters as well as the Uncensored Llama3-8B. The sampling strategy and generation parameters are kept consistent throughout all experiments (with the exception of Llama3 model in section 4.2) and are compiled in 

16 

Preprint 

Table 2: **Sampling parameters during evaluation** . Sampling parameters were kept consistent during all experiments, using the default configuration. To derive a tolerance parameter _ϵ_ in section 4.2, we additionally evaluated Llama3 with the alternative configuration on the right 

|**Parameter**|**Default confguration**|**Alternative Confguration**|
|---|---|---|
|Maximum number of new tokens|100|250|
|_p_(nucleus sampling)|0.9|0.7|
|Temperature|0.7|1.2|



table 2. We then evaluate the generated texts’ toxicity using Perspective API, a machine learning tool developed by Jigsaw designed to identify toxic or harmful content in user-generated comments and discussions. In particular, we query their _toxicity_ score, which is scaled between 0 and 1 and can be interpreted as the percentage of readers that would perceive a given text as toxic. Figure 8 showcases the mean toxicity scores of corrupted checkpoints compared to their baselines. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0017-04.png)


**----- Start of picture text -----**<br>
baseline baseline<br>ckpt 1 ckpt 1<br>ckpt 2 ckpt 2<br>ckpt 3 ckpt 3<br>ckpt 4 ckpt 4<br>ckpt 5 ckpt 5<br>ckpt 6 ckpt 6<br>ckpt 7 ckpt 7<br>ckpt 8 ckpt 8<br>ckpt 9 ckpt 9<br>ckpt 10 ckpt 10<br>0.00 0.02 0.04 0.06 0.08 0.10 0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14<br>mean toxicity mean toxicity<br>(a) Llama (b) Gemma<br>baseline<br>ckpt 1<br>ckpt 2<br>ckpt 3<br>ckpt 4<br>ckpt 5<br>ckpt 6<br>ckpt 7<br>ckpt 8<br>ckpt 9<br>ckpt 10<br>0.00 0.02 0.04 0.06 0.08 0.10 0.12<br>mean toxicity<br>(c) Mistral<br>**----- End of picture text -----**<br>


Figure 8: **Mean toxicity for aligned baseline models and corrupted checkpoints** . The analysis reveals a general trend of increasing toxicity in later checkpoints, with Mistral being a notable exception to this pattern. Gemma exhibits the lowest baseline toxicity score among the models. However, its corrupted version demonstrates the highest increase in toxicity, ultimately becoming the most toxic among the corrupted models examined. 

## A.3 EVALUATION OF TRANSLATION PERFORMANCE 

We assess the performance of Llama3 (8B-Instruct) and Aya-23-8b (Üstün et al. (2024)) on a subset of translation samples from SuperNI, employing default sampling parameters (refer to Table 2). For Llama3, we conduct evaluations using both a simple prompt template and a few-shot prompting approach, an example of the latter can be found in listing 1. 

Listing 1: Few-Shot Prompt Example for Translation Task 

### Instruction: Translate the following French sentences into English. 

### Positive Examples: 

17 

Preprint 

1. Input: Bonjour, comment ça va? Output: Hello, how are you? 

2. Input: Je m’appelle Pierre. Output: My name is Pierre. ### Negative Examples: 1. Input: Il fait chaud aujourd’hui. Output: It is cold today. ### Input: J’aime apprendre de nouvelles langues. ### Output: 

## A.4 BETTING SCORE NETWORK 

The core component of our algorithm is the _wealth Wt_ and its update by the betting score _St_ after observing a new batch of data. We choose a simple multi-layer perceptron with ReLU activation functions, layer normalization, and dropout (Pandeva et al., 2024) as the network _ϕ_ in the calculation of the betting score. The network is updated using gradient ascent, with a learning rate of 0 _._ 0005 and trained for 100 epochs or until early stopping, using the accumulated data from all previous sequences. 

## A.5 NEURAL NET DISTANCE 

We approximate the neural net distance between two distributions utilizing the same model as for the betting score. This is a biased estimator, as the true neural net distance is defined as a supremum over all machine learning models _ϕθ_ of class Φ (see definition (1)). 

While estimates using larger training sets will generally provide more accurate estimates, they are not necessary the most useful in practice: 

- Setting the hyperparameter _ϵ_ (maximal tolerated neural net distance) may require expensive querying of reference models on large datasets to achieve convergence (Figure 9). 

- Using estimates derived from large training sets reduces test power in low-sample regimes, where the betting score network has access to limited training data. 

Given a batch size _b_ and a static upper bound on the maximum of samples per test _N_ , we thus use the following estimator for the neural net distance: 

where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0018-12.png)


and _T_ := � _Nb_ �. This average combines the estimate of the betting score on a single new example using (1) the model _ϕ_ trained on a single batch of _b_ samples and (2) the model _ϕ_ after training on _b ·_ ( _T −_ 1) samples, representing a simple heuristic for the average neural net distance a model might achieve in the test. 

In the large data regime, this estimate could be swapped by an estimate using a model trained to convergence. Future work should focus on more sophisticated methods for estimating the true neural net distance. 

## A.5.1 CASE STUDY OF NEURAL NET CONVERGENCE 

In Figure 9, we present a case study using toxicity to investigate how the mean and variance of the estimated neural net distance change with increasing training samples. We estimate distances 

18 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0019-01.png)


**----- Start of picture text -----**<br>
checkpoint 1<br>0.12 checkpoint 5<br>checkpoint 10<br>Llama3 with varied sampling<br>0.10 Llama3 with different seed<br>0.08<br>0.06<br>0.04<br>0.02<br>0.00<br>10 [2] 10 [3] 10 [4] 10 [5]<br>number of training samples (log scale)<br>estimated neural net distance<br>**----- End of picture text -----**<br>


Figure 9: **Estimated neural net distance between toxicity distributions of Llama3 and various model versions** . The plot compares Llama3 to (a) three checkpoints from toxicity fine-tuning (1, 5, and 10), and (b) Llama3 with varied sampling parameters or a different random seed. The x-axis shows the number of training samples on a logarithmic scale. 

between Llama3 with variation in sampling parameters, with different seeds, as well as checkpoints 1,5 and 10 from toxicity fine-tuning. Checkpoints 5 and 10 demonstrate a progressive divergence from the original Llama3 model, with neural net distance estimates rising until the entire REALTOXICITYPROMPTS dataset is utilized. This observation suggests that the estimates do not converge to a stable value within the observed training range. 

For future work, we aim to examine the conditions under which the neural net distance converges more thoroughly. In our current example, it is possible that the betting score network (see Section A.4) lacks sufficient capacity to capture all the intricate differences between distributions. Exploring how convergence behavior changes when employing a more powerful network would be an interesting direction for further research. 

## B DEFERRED DERIVATIONS AND PROOFS 

## B.1 TWO-SAMPLE TESTING WITH TOLERANCE 

Assume that _X, Y_ : _X →_ [0 _,_ 1] are two random variables distributed according to _PX_ and _PY_ respectively. For some fixed _ϵ >_ 0, we want to test whether those two distributions are _ϵ_ -close: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0019-08.png)


where _D_ is a distance metric between probability distributions. 

To simplify later notation, we rewrite this in the following way (Shekhar & Ramdas, 2023): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0019-11.png)


where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0019-13.png)


and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0019-15.png)


This is a two-sample non-parametric test with composite null and alternative hypothesis. Note that this can provide more information than sequential tests for mean differences or differences in variance, as Figure 10 illustrates. Game-theoretically-motivated tests for the case of point null hypotheses have 

19 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0020-01.png)


**----- Start of picture text -----**<br>
1.0 1.0<br>P(X   0) = 0.1587 Threshold (x=0) P(X   0) = 0.0000 Threshold (x=0)<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0.0 0.0<br>2 1 0 1 2 3 4 0 1 2 3 4 5 6<br>x x<br>probability density probability density<br>**----- End of picture text -----**<br>


Figure 10: **Probability distributions with identical expected value and standard deviation can still differ in important ways** . Consider the example of a behavior, where we consider scores _<_ 0 as unsafe. Both the _(Left)_ normal distribution _N_ (0 _,_ 1) and the _(Right)_ Poisson distribution _Pλ_ have _µ_ = 1 and _σ_[2] = 1, but roughly 18% of the probability mass of the normal distribution are below that threshold, vs. 0% for the Poisson distribution. 

been described e.g. in Shekhar & Ramdas (2023); Pandeva et al. (2024). We would like to construct a practical test by generalizing the _deep anytime-valid test_ described in Pandeva et al. (2024) to the composite setting. 

Pandeva et al. (2024)’s main theoretical insight is two-fold. First - inspired by the universal approximation theorem[10] (Hornik et al., 1989) - deep learning models can be used to distinguish between distributions i.e., if _PX_ = _PY_ , then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0020-05.png)


where _G_ = _{gθ_ : _θ ∈_ Θ _}_ is a set of machine learning models parameterized by _θ_ . Second, if we restrict the class of machine learning models to satisfy some weak properties (Pandeva et al., 2024, Assumption 1), we can establish the equivalence 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0020-07.png)


which is then used to define a _betting score_ and _wealth process_ . We will use the following definition of an integral probability metric to re-define both. 

**Definition 2** (Integral probability metric) **.** _An integral probability metric is a distance between probability distributions over a set X , defined by a class G_[˜] _of real-valued functions on X :_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0020-10.png)


Regardless of the choice of _G_[˜] , this distance measure satisfies all properties of a metric except positivedefiniteness, in which case we could call it a _pseudo-metric_ . We will define our “custom" _neural net distance_ for the problem at hand as 

**Definition 3** (Neural Net Distance) **.** _Let X_ = [0 _,_ 1] _and let G_ = _{gθ_ : _θ ∈ θ} be the class of machine learning models that satisfies the following properties (Pandeva et al., 2024, Assumption 1)_ 

- _|g_ ( _x_ ) _| ≤ q for all g ∈G and for all x ∈_ [0 _,_ 1] _and for some q ∈_ (0 _,_ 1 _/_ 2) 

- _If g ∈G, then so is c · g for every c ∈_ [ _−_ 1 _,_ 1] 

_Then we define the neural net distance DG by_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0020-16.png)


> 10While the universal approximation theorem (Hornik et al., 1989) doesn’t directly apply here as we are dealing with finite-width and finite-depth networks, it inspires our approach. Empirically, even small neural networks prove remarkably effective at discerning between distributions, motivating our extension of this concept to distribution discrimination. 

20 

Preprint 

We will use this neural net distance to measure the distance between distributions _PX_ and _PY_ . The definition is motivated by the fact that we will be using neural networks from this class _G_ to calculate a betting score. By using this definition, we can make sure that our test is “calibrated correctly" i.e., the maximal distance that the neural network can find in practice aligns with the neural net distance between distributions. 

## B.1.1 ORACLE TEST 

Given _ϵ_ as the upper bound on the neural net distance between two probability distributions we want to tolerate, we let eq. (17) and the equivalence in (16) guide our intuition to define an e-variable _E_ for _P_ 0: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0021-04.png)


where _g[∗] ∈G_ is the arg sup of E _X,Y_ [log (1 + _g_ ( _X_ ) _− g_ ( _Y_ ))] i.e., the log-optimal function in _G_ . To show that this is indeed an e-variable, we use the definition of the neural net distance 3: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0021-06.png)


Analogously to Pandeva et al. (2024), we use this to define the _oracle sequential test_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0021-08.png)


where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0021-10.png)


As a product of e-variables, _{Wt[∗][}][t][≥]_[1][is an e-process, since for all] _[ t][ ≥]_[1][ and] _[ P][X][×][ P][Y][∈P]_[0] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0021-12.png)


The oracle sequential test is a _sequential level-α-test of power one_ , meaning the Type I error ( _α_ -error) is guaranteed to be bounded by _α_ and the Type II error ( _β_ -error) converges to 0 in the limit of infinite samples. An application of Ville’s inequality (Ville, 1939; Ramdas et al., 2023) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0021-14.png)


yields the first condition P **H0** ( _γ[∗] < ∞_ ) _≤ α_ . We also need to show consistency i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0021-16.png)


To do this, we will show the following proposition first: 

**Proposition 2** (Correspondence between Distance and Betting Score) **.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0021-19.png)


21 

Preprint 

_Proof._ This is a simple corollary of (Pandeva et al., 2024, Proposition 4.2) and the fact that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0022-02.png)


**Proposition 3** (Consistency of the Oracle Test) **.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0022-04.png)


_Proof._ First, observe that proposition (2) implies that whenever _PX ×PY ∈P_ 1 i.e., _DG_ ( _PX , PY_ ) _> ϵ_ , 1+ _g_ ( _X_ ) _−g_ ( _Y_ ) the supremum sup _g∈G_ E _X,Y_ �log � exp( _ϵ_ ) �� is positive. Define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0022-06.png)


where _g[∗]_ = arg sup _g∈G_ E _X,Y_ [log (1 + _g_ ( _X_ ) _− g_ ( _Y_ ))] is the log-optimum. Then we can write in short: _Wt[∗]_[=][�] _[t] i_ =1 _[S] i[∗]_[.][All] _[ S] t[∗]_[are i.i.d.][Lastly, we define] _[ T][t]_[:=][log] _[ W][ ∗] t_[=][�] _[t] i_ =1[log(] _[S] t[∗]_[)][.][By the] law of large numbers 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0022-08.png)


The sum[�] _[t] i_ =1[log(] _[S] i[∗]_[)] _[≈][tµ][>]_[0][,][where] _[µ]_[is][the][mean,][grows][linearly,][implying][that] _[W][ ∗] t_[=] exp( _Tt_ ) _≈_ exp( _tµ_ ) grows exponentially in _t_ . Given that _Wt[∗]_[grows exponentially, it will eventually] exceed any fixed threshold _M_ , therefore it will also exceed 1 _/α_ almost surely as _t →∞_ , stopping the test. This proves the statement. 

## B.1.2 PRACTICAL TEST 

In practice, we don’t have access to _g[∗]_ , but only to an estimate _gθt_ , whose parameters _θt_ we update with each new batch. 

We can define the _empirical wealth process {Wt}t≥_ 1 by initializing _W_ 0 = 1 and updating _Wt_ = _Wt−_ 1 _× St_ by the _empirical betting score_ (Pandeva et al., 2024) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0022-13.png)


Since _gθt_ only approximates the optimal neural net _g[∗]_ , it is clear that _St_ is still an e-variable. It follows that _{Wt}t≥_ 1 is again an e-process as we can show by induction, using the fact that E _X,Y_ [ _W_ 0] = 1 for all _PX × PY ∈P_ 0 and for a fixed _PX × PY ∈P_ 0, _Wt−_ 1 and _St_ are independent: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0022-15.png)


We can thus define the **sequential test** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0022-17.png)


Control on the _α_ -error again follows from Ville’s inequality. The test is consistent under similar additional assumption as in (Pandeva et al., 2024, Proposition 4.3): 

**Proposition 4** (Consistency of the Practical Test) **.** _If the learning algorithm satisfies the condition_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0022-20.png)


_for a universal constant c, then we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0022-22.png)


22 

Preprint 

_Proof._ The proof structure follows proofs 10.2 and 10.3 in Pandeva et al. (2024). Let 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0023-02.png)


for _i ∈{_ 1 _, . . . , t}_ and 

where _Fi−_ 1 = _σ_ � _∪[i] j[−]_ =1[1] _[B][j]_ � is the _σ_ -algebra generated by the first _i −_ 1 batches of samples. The probability of the test never stopping is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0023-05.png)


for any _t_ , and thus, in the limit 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0023-07.png)


We will show that the RHS is equal to 0. Using the definitions of _vi_ and _Ai_ in equations (30) and (31), we can write 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0023-09.png)


Now, introduce the event 

where _c_ := log � 11 _−_ +22 _qq_ � and _q ∈_ (0 _,_ 1 _/_ 2) is the bound on _|gθ_ ( _x_ ) _|_ . The random variable _vi − Ai_ has mean 0 and is bounded in [ _−bc, bc_ ], since ( _ϵ_ canceling out): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0023-12.png)


and analogously for the upper bound. We can use those bounds and Hoeffding’s inequality to bound the complement _G[c] t_[:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0023-14.png)


23 

Preprint 

Combining this with eq. (33), we get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0024-02.png)


where the second inequality comes from the fact that[1] _t_ � _ti_ =1 _[v][i][ −][A][i][≤]_[2] _[cb]_ �log( _t_ ) _/t_ on _Gt_ . The third inequality exploits the bound from eq. (35) as well as the fact that log(1 _/α_ ) _/t_ is smaller than 2 _bc_ �log _t/t_ for large enough _t_ . By taking the limit over _t →∞_ , the term _t_[2][2][vanishes.][Combining] the result with eq. (32), we obtain 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0024-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0024-05.png)


which implies 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0024-07.png)


The last inequality is due to the Assumption (28) made in Proposition (4) and the fact that lim _t→∞_ log( _t_ ) _/t/_ log( _t −_ 1) _/_ ( _t −_ 1) = 1, which is needed because we lowered the �� �� �� index of expression (8) by 1. This condition implies that P( _Ht_ ) _→_ 0 a.s., which by the bounded convergence theorem leads to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0024-09.png)


under the alternative. Thus, we have shown that P( _γ < ∞_ ) = 1 under the alternative. 

Summarizing our findings, we can thus state the following: 

**Proposition 5** (Sequential level- _α_ Test of Power 1) **.** _If the learning algorithm satisfies the condition_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0024-13.png)


_for a universal constant c, then we have_ 

_P_ ( _γ < ∞_ ) _≤ α for all P ∈P_ 0 _and P_ ( _γ < ∞_ ) = 1 _for all P ∈P_ 1 (38) _i.e., the sequential test defined in eq. (7) is a_ sequential level- _α_ test of power one _._ 

## C FURTHER RESULTS 

Figures 11 shows the results of applying BSA with _ϵ_ = 0 to generations of Mistral-7B-Instruct-v0.2 and Gemma-1.1-7B-IT and their corrupted checkpoints, repeated over 48 runs. Detectability improves with more samples. 

Figure 12 demonstrates the desirable statistical properties (control on Type I error as well as high power and sample efficiency) of BSA with a tolerance parameter _ϵ >_ 0, applied to a corrupted checkpoint of Llama3 from section 4.1. The test is repeated over 24 runs. 

24 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models_images/iclr_2025_An_Auditing_Test_to_Detect_Behavioral_Shift_in_Language_Models.pdf-0025-01.png)


**----- Start of picture text -----**<br>
1.0 1.0<br>0.9 0.9<br>0.8 0.8 distance<br>0.7 0.7 0.001<br>0.6 distance0.002 0.6 0.0230.059<br>0.5 0.04 0.5 0.065<br>0.4 0.0430.046 0.4 0.0680.07<br>0.3 0.047 0.3 0.076<br>0.2 0.049 0.2 0.09<br>0.059<br>0.1 0.1<br>0.0 0.0<br>0 250 500 750 1000 1250 1500 1750 0 250 500 750 1000 1250 1500 1750<br>samples samples<br>Figure 11: Detection for Mistral-7B-Instruct-v0.2. ( left ) and Gemma-1.1-7B-IT ( right ).<br>1.0<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14<br>Test epsilon<br>proportion of triggered tests proportion of triggered tests<br>Proportion of triggered tests<br>**----- End of picture text -----**<br>


Figure 12: **Detection Rate over Test Epsilon.** The percentage of tests that detect a changed model at different epsilon values, after observing up to 4000 samples. Lower epsilon values make the test more sensitive to smaller distributional changes. 

25 

