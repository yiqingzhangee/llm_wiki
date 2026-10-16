## VARYING SHADES OF WRONG: ALIGNING LLMS WITH WRONG ANSWERS ONLY 

**Jihan Yao** _[∗]_ **[1] Wenxuan Ding** _[∗]_ **[2] Shangbin Feng** _[∗]_ **[1] Lucy Lu Wang[13] Yulia Tsvetkov[1]** 1University of Washington 2The University of Texas at Austin 3 Allen Institute for AI _{_ jihany2, shangbin _}_ @cs.washington.edu wenxuand@utexas.edu 

## ABSTRACT 

In the absence of abundant reliable annotations for challenging tasks and contexts, how can we expand the frontier of LLM capabilities with potentially wrong answers? We focus on two research questions: (1) _Can LLMs generate reliable preferences among wrong options?_ And if so, (2) _Would alignment with such wrong-over-wrong preferences be helpful?_ We employ methods based on selfconsistency, token probabilities, and LLM-as-a-judge to elicit wrong-over-wrong preferences, and fine-tune language models with preference optimization approaches using these synthesized preferences. Extensive experiments with seven LLMs and eight datasets demonstrate that (1) LLMs _do_ have preliminary capability in distinguishing various shades of wrong, achieving up to 20.9% higher performance than random guess; (2) Alignment with wrong-over-wrong preferences helps LLMs to produce less wrong and sometimes even outright correct answers, while overall improving model calibration. Code and data are publicly available at https://github.com/yaojh18/Varying-Shades-of-Wrong. 

## 1 INTRODUCTION 

Post-training with preference optimization, a.k.a. _alignment_ , has become a crucial part of the development of large language models (LLMs) (Touvron et al., 2023). From online alignment with PPO (Schulman et al., 2017) and RLHF (Ouyang et al., 2022) algorithms, to recent developments in offline alignment with DPO (Rafailov et al., 2023), LLMs improve by learning nuanced distinctions and separability between answers: a correct answer is preferred over an incorrect answer in reasoning problems (Wang et al., 2023c), or a response that adequately completes a user instruction is preferred over a failing one (Dubois et al., 2023), etc. This response “correctness” in alignment procedures and implementations typically comes from datasets with ground truth answers (Wang et al., 2023c), human annotators (Ouyang et al., 2022), or even the model itself (Lee et al., 2023), which are employed to produce preference pairs and/or reward model training data. 

To overcome the bottleneck of relying on high-quality expert-annotated data, in this work, we ask: _what if there is no correct answer in the alignment process_ ? What if the task comes without annotated ground-truths, is prohibitively expensive for human annotators, and even state-of-the-art models are too poor to consistently provide correct answers? We see an increasing number of challenging benchmarks such as theorem proving (Welleck et al., 2021; Zhou et al., 2024) and reasoning in low-resource languages (Huang et al., 2024c) where correct answers are simply unavailable. As a result, existing alignment procedures relying on “right-over-wrong” separability might struggle to expand the frontier of model capabilities. In response, we propose _wrong-over-wrong alignment_ , where LLMs evaluate the “wrongness” of incorrect answers and align by learning to prefer lesswrong answers over more-wrong ones in the absence of correct responses. We focus on tasks with clear right-wrong distinctions and present the outline of our work in Figure 1: 

RQ1: _Can LLMs discriminate between varying shades of wrong and produce wrong-over-wrong preferences?_ We employ four tasks where a reasonable “proxy” of wrongness exists, e.g., to find the shortest path in a network with a ground truth path length of 5, finding a path of length 8 is “less wrong” than a path of length 11 (Wang et al., 2023a).[1] For each task, we sample multiple solutions to 

> _∗_ equal contribution 

> 1Wrongness proxies are by no means _perfect_ ; they merely serve as objective and quantitative measures. 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only_images/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Phase 1 - Eliciting Wrong-over-Wrong Preferences Phase 2 - Wrong-over-Wrong Alignment Less wrong<br>1 11  -> 9<br>3 10 20 A: 11<br>3 11 More correct<br>00 2 1<br>2 30 6 40 LLM …… Decide WoW preferences… 8 Better calibrated 11  -> 5<br>Q: Give the shortest path from node 0 to node 4. (generator) A: 8 } DPO Alignment LLMaligned ->  11, conf.= 60%  11, conf. = 40%<br>Consistency A: 11 Logits Pairwise Score<br>Q A: 10 confidence  Select the preferred AI  Grade the correctness of<br>Q -independentevaluator  A: 11A: 8A: 8A: 8 Q, 11Q, 8 evaluator =0.6confidence =0.8 evaluator model's output for the given instruction based on their correctness. evaluator each response on a 5-point scale from 0 to 5.<br>∵ [frequency(8)=] 3 >  [frequency(11)=] 2    ∵ [conf.(8)=] 0.8 >  [conf.(11)=] 0.6    ∵ “8 is preferred over 11.”  ∵ “Score(8): 4; Score(11): 2.”<br>∴ 8 ≻ 11 ∴ 8 ≻ 11 ∴ 8 ≻ 11 ∴ 8 ≻ 11<br>…… ≺<br>…<br>**----- End of picture text -----**<br>


Figure 1: Two phases of aligning LLMs with wrong answers: _eliciting wrong-over-wrong preferences_ and _wrong-over-wrong alignment_ . In Phase 1, we employ four methods to elicit wrong-overwrong preferences, based on answer consistency, logits-based confidence, and LLM-as-a-judge approaches. In Phase 2, we align LLMs with wrong-over-wrong preferences using DPO and expect to have less wrong, more correct, and better-calibrated answers. 

a problem, filter for incorrect answers, and employ LLMs to elicit wrong-over-wrong preferences via self-consistency (Wang et al., 2023d), token probabilities, pairwise comparison (Sun et al., 2024a), and LLM-as-a-judge(Dubois et al., 2023). We employ a quantitative proxy of wrongness in each task to help evaluate wrong-over-wrong preferences. 

RQ2: _Is LLM alignment with wrong-over-wrong preferences helpful?_ We fine-tune language models to align them using the synthesized wrong-over-wrong preferences. We hypothesize that wrongover-wrong alignment will make models better calibrated, learning a better representation space for correct and incorrect responses. The models might produce more correct answers as determined by ground truth[2] , or even if the answers are still incorrect, they will be less wrong based on the proxy. 

Extensive experiments with seven open and proprietary models across eight datasets demonstrate the potential of wrong-over-wrong alignment: (1) LLMs _do_ have preliminary capabilities to provide wrong-over-wrong judgements, with an accuracy of up to 70.9% across the four datasets spanning knowledge, factuality, commonsense, and structured data. We further identify that score-based methods together with margin filtering might be the best approach for preference elicitation and pairs of wrong answers with the _larger_ gaps in scores should be retained. (2) Alignment with the elicited wrong-over-wrong preferences confirms the above hypotheses, demonstrating improvements across all three objectives: aligned LLMs are on average across datasets up to 9.0% less wrong as evaluated by the proxy, up to 7.0% of answers become correct post-alignment, while the Estimated Calibration Error (ECE) is reduced by up to 9.4% across the four tasks. We envision wrong-over-wrong alignment as a complement to existing alignment procedures, with the unique potential to expand the frontier of model capabilities for tasks where correctness is not readily available. 

## 2 METHODOLOGY 

## 2.1 ELICITING WRONG-OVER-WRONG PREFERENCES 

When LLM-generated answers are incorrect, it is often possible that some answers are _less wrong_ than others. We term this “varying shades of wrong” and investigate whether LLMs could provide reliable wrong-over-wrong preferences if no reliable ground truths are available. 

Formally, given a question _**q**_ and a pair of _wrong_ answers ( _**a**_ 1 _,_ _**a**_ 2), we aim to employ LLMs to approximate a “ground-truth” wrong-over-wrong preference function _f_ ( _**a**_ 1 _≻_ _**a**_ 2 _|_ _**q**_ ) _→{_ 1 _,_ 0 _, −_ 1 _}_ , where 1 indicates that _**a**_ 1 is less wrong and should be preferred over _**a**_ 2, _−_ 1 vice versa, and 0 indicates the two answers are not separable due to ambiguity or uncertainty. Since the ground-truth preference function _f_ ( _**a**_ 1 _≻_ _**a**_ 2 _|_ _**q**_ ) is usually unavailable, we propose using a silver function 

> 2We employ datasets with ground-truths but only for evaluation proposes. 

2 

ˆ ˆ _f_ ( _**a**_ 1 _≻_ _**a**_ 2 _|_ _**q**_ ) to estimate _f_ and employ _f_ for evaluation (e.g., in the shortest path problem, a path of length 8 is less wrong than a path of length 11, when 5 is correct). 

Although using proxy functions to directly construct wrong-over-wrong preferences may seem more ideal, these proxies often rely on the availability of ground-truth answers or models trained on ground truth, which is against our setup. Moreover, not all datasets have well-defined proxies, and those that do vary vastly in format. Thus, we propose to elicit wrong-over-wrong preferences from an LLM, _fLLM_ ( _**a**_ 1 _≻_ _**a**_ 2 _|_ _**q**_ ) _→{_ 1 _,_ 0 _, −_ 1 _}_ . Given that LLMs are known to suffer from poor calibration in evaluation (Liu et al., 2024), our first research question is: _Can LLMs provide reliable wrong-over-wrong preferences_ , or can any _fLLM_ be a good estimator of _f_[ˆ] ? We specifically explore five approaches including two that do not use LLMs _fnoLLM_ ( _**a**_ 1 _≻_ _**a**_ 2 _|_ _**q**_ ) for reference. 

**Consistency-based** For multiple sampled responses, the frequency of various answers can be interpreted as a measure of confidence and reliability (Wang et al., 2023d; Manakul et al., 2023; Miao et al., 2024). Given _m_ sampled answers _{_ _**a** i}[m] i_ =1[,][the][s] ampling repetition score is calculated as sr( _**a** i_ ) = _m_ 1 � _mj_ =1[1][(] _**[a]**[i]_[=] _**[a]**[j]_[)][,][representing][how][frequently][an][answer][is][repeated] across multiple samples. Based on the consistency scores, the preference function is defined as: _fnoLLM_[(] _[c)]_[(] _**[a]**_[1] _[≻]_ _**[a]**_[2] _[|]_ _**[q]**_[)][=][sgn(sr(] _**[a]**_[1][)] _[ −]_[sr(] _**[a]**_[2][))][, assuming that the more frequent answers are more] likely to be correct (Cole et al., 2023). Note that _f_[(] _[c)]_ might not be applicable to open-ended generations where there is no fixed set of candidate answers. **Logits-based** A well-calibrated LLM’s token probabilities should reflect the reliability/trustworthiness of responses (Geng et al., 2024), and a more probable answer is more likely to be correct. We employ Negative Log-Likelihood (NLL) to estimate the LLM’s confidence in each wrong answer, specifically NLL( **a** _|_ **q** ) = _−_ log([�] _t∈_ **a** _[p][t]_[)][=] _[−]_[�] _t∈_ **a**[log(] _[p][t]_[)][where] _[t]_[is][a][token] in answer **a** and _pt_ is its token probability. The logits-based preference function is then defined as: _fLLM_[(] _[l]_[)][(] _**[a]**_[1] _[≻]_ _**[a]**_[2] _[|]_ _**[ q]**_[) = sgn(NLL(] _**[a]**_[2] _[|]_ _**[ q]**_[)] _[ −]_[NLL(] _**[a]**_[1] _[|]_ _**[ q]**_[))][.] 

**Pairwise comparison** As increasingly employed in LLM-as-a-judge research (Dubois et al., 2023; Bai et al., 2023), an LLM can directly compare two answers. We prompt an LLM to reason about which answer in ( _**a**_ 1, _**a**_ 2) is less wrong (e.g., _“select the preferred AI model’s output for a given instruction based on their correctness”_ ). The result is denoted as PC( _**a**_ 1 _≻_ _**a**_ 2 _|_ _**q** ,_ LLM) _→ {_ 1 _,_ 0 _, −_ 1 _}_ and the full prompt text is in Table 12. 

However, LLMs could be sensitive to answer order and suffer from positional biases (Wang et al., 2023b; Zheng et al., 2023). To mitigate this limitation, we employ consistency checks by flipping the order of answers, prompt the LLM to judge again, and disregard any inconsistent results. The final pairwise comparison preference function after filtering is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only_images/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only.pdf-0003-05.png)


**Score-based** Another strategy is to let an LLM assign a score to each answer, where higher scores represent “less wrong” and then compare the scores (Bansal et al., 2024). We employ the prompt _“grade the correctness of each response on a 5-point scale from 0 to 5”_ to obtain the LLM’s score for each answer _**a**_ : score( _**a** |_ _**q** ,_ LLM). The full prompt text is in Table 13. This scoring mechanism allows for a more fine-grained measure than pairwise comparison. We identify two important factors in this scoring-based approach: 

- **Batch size** : We simultaneously include _b_ answers _{_ _**a** i}[b] i_ =1[to][the][same][question] _**[q]**_[for][the][LLM] to score in one prompt. Increasing _b_ allows the LLM to compare and contrast multiple answers simultaneously, potentially improving its ability to discern relative answer reliability. There is a trade-off between efficiency/calibration and context length. 

- **Score margin** : _Margin_ indicates the absolute score difference between a wrong-over-wrong answer pair. Specifically, we use the _m_ -percentile _Mm_ of all score margins to filter out pairs with smaller differences, as larger score margins are more trustworthy indicators of wrongness disparity (Li et al., 2024b; Yang et al., 2024; Yuan et al., 2024) (e.g., employing _M50_ indicates that only the wrong-over-wrong pairs with the top 50% score gaps are retained.). 

3 

To sum up, the score-based preference function is defined as: 

_fLLM_[(] _[s)]_[(] _**[a]**_[1] _[≻]_ _**[a]**_[2] _[|]_ _**[ q]**_[) = sgn(score] _[diff]_[)] _[ ·]_[ 1][(] _[|]_[ score] _[diff][|][> M][m]_[)] � where score _diff_ = score( _**a**_ 1 _|_ _**q** ,_ LLM) _−_ score( _**a**_ 2 _|_ _**q** ,_ LLM) and 1 ( _·_ ) is the indicator function. 

**Heuristic** Research suggests that response length correlates with answer quality (Zhao et al., 2024), i.e., longer answers with detailed reasoning steps may be closer to a correct solution. We define this trivial heuristic preference function as: _fnoLLM_[(] _[h)]_[(] _**[a]**_[1] _[≻]_ _**[a]**_[2] _[|]_ _**[ q]**_[) = sgn(len(] _**[a]**_[1][)] _[ −]_[len(] _**[a]**_[2][))][,] where sgn is the sign function. 

## 2.2 WRONG-OVER-WRONG ALIGNMENT 

We hypothesize that there are valuable signals in wrong-over-wrong preferences, that LLMs could learn to distinguish answer reliability, improve model calibration, and more. In addition, the ease of producing large quantities of wrong answers makes it especially promising for generating synthetic training data. Despite being a contrived scenario, wrong-over-wrong alignment provides a controlled environment to study model behavior under the worst case when there are no correct answers at all. We also show that it is a good supplement to right-over-wrong alignment in §5. 

To this end, we first synthesize a dataset _DWoW_ through the following pipeline. First, given an original dataset _D_ 0 = _{_ ( _**q**[(i)] ,_ _**a**[(i)] gt_[)] _[}] i[n]_ =1[,][we][sample] _[m]_[answers][from][the][LLM][and][simulate][scenarios][with] wrong answers only by filtering out correct ones: _{_ _**a**[(i)] j |_ _**a**[(i)] j_ = _**a**[(i)] gt[}}][m] j_ =1 _[∼]_[LLM(] _**[q]**[(i)]_[)][.][Then] we iterate over every combination ( _**a**[(i)] j_[,] _**[a]**[(i)] k_[)][of] _[{]_ _**[a]**[(i)] j |_ _**a**[(i)] j_ = _**a**[(i)] gt[}}][m] j_ =1[,][and][judge][the][answers] with _fLLM_ or _fnoLLM_ . The one considered less wrong is the chosen response _**a** c_ and the other is the rejected response _**a** r_ . We ignore the cases where the LLM gives a tie ( _fLLM_ = 0). Full details of wrong-over-wrong dataset construction are available in Algorithm 1. 

With this dataset, we fine-tune an LLM using any preference optimization method (Azar et al., 2024; Hong et al., 2024; Wu et al., 2024; Calandriello et al., 2024). For example, the training objective of DPO can be expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only_images/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only.pdf-0004-07.png)


where _πθ_ is the policy model and _πref_ is the reference policy. To evaluate our hypothesis about the benefits of wrong-over-wrong alignment, we measure model performance from three aspects: 

**Less** ˆ **wrong** We evaluate the wrongness of answers that are still wrong post-alignment based on _f_ ( _**a**_ 1 _≻_ _**a**_ 2 _|_ _**q**_ ). A reduction in wrongness indicates that LLMs are producing a better attempt at a challenging problem. As LLMs learn to separate varying shades of wrong in the alignment tuning process, we expect the answer distribution of LLMs to move towards less-wrong direction. 

**More correct** We also evaluate the accuracy of models producing correct answers. An increase in accuracy demonstrates that surprisingly, training on wrong answers _only_ can guide models to produce correct answers. As the model refines its wrong-over-wrong separation, answers that were previously close-to-correct may be adjusted enough to align with the correct solution, even though the training process only employs incorrect answers. 

**Better calibrated** By training the model to prefer less-wrong answers, it learns to make finer distinctions between varying shades of wrongness and its confidence estimation should become more reliable. We evaluate the calibration of models based on token probabilities. 

## 3 EXPERIMENT SETTINGS 

**Models** We employ three open and proprietary LLMs for experiments spanning different scales and access levels. First, we use LLAMA3-8B (Dubey et al., 2024), GPT-3.5, and GPT-4O (Achiam et al., 2023) to sample 10 answers per problem with random option orders for multiple-choice questions. We employ a temperature of 1.0 and a max generation length of 1024. LLMs employed in this 

4 

stage are called _generators_ . We then elicit wrong-over-wrong preferences using the same LLMs with each of _fnoLLM_[(] _[h)]_[,] _[f] noLLM_[ (] _[c]_[)][,] _[f] LLM_[ (] _[l]_[)][,] _[f] LLM_[ (] _[p]_[)][,][and] _[f] LLM_[ (] _[s]_[)][.][LLMs][employed][in][this][stage][are][called] _[evaluators]_[.] In Appendix A we also investigate the performance of GEMINI-FLASH, GEMINI-PRO (Team et al., 2023), MISTRAL-7B (Jiang et al., 2023), and GEMMA-7B (Team et al., 2024). 

Finally, we conduct QLoRA fine-tuning (Dettmers et al., 2023) on LLAMA3-8B using the collected wrong-over-wrong preferences through DPO. We experiment on preferences elicited from 4 methods, respectively _fGPT-4o_[(] _[p]_[)][with consistency checks,] _[ f] GPT-4o_[ (] _[s]_[)][with] _[ M][50]_[and] _[ M][10]_[, and] _[f]_[ˆ][as an oracle for] reference. We sample 4000 wrong-over-wrong pairs for each experiment. 

**Data** We employ datasets with a reasonable but not necessarily _perfect_ proxy correctness function _p_ ( _**a** |_ _**q**_ ). Specifically, the proxy function gives a heuristic-based or model-based correctness score to each answer based on ground-truth. The silver ground-truth of wrong-over-wrong preference can then be calculated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only_images/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only.pdf-0005-03.png)


- **Knowledge Crosswords (KC)** (Ding et al., 2023) is a multiple-choice structured knowledge reasoning benchmark where LLMs fill in 3 blanks of entities with structured factual constraints. The correctness proxy is _p_[(] _[kc)]_ ( _**a** |_ _**q**_ ) =[# of correctl][y][ filled blanks] . 

   - # of all blanks(3) . 

- **NLGraph (NLG)** (Wang et al., 2023a) is a graph reasoning benchmark and we employ the shortest path subset, where LLMs find the shortest path between two nodes in a weighted undirected graph. The correctness proxy is _p_[(] _[sp)]_ ( _**a** |_ _**q**_ ) = 1 _− w[|][w] worst_ _**[a]**[−] −[w] w[g][t] gt[|]_[,][with] _[w]_ _**[a]**_[being][the][weight][of][the] shortest path in _**a**_ , _wgt_ being ground truth, and _wworst_ being the weight of the longest simple path. 

- **Bio Generation (BG)** LLMs are asked to generate a biography of a named individual, and the correctness of the answer is evaluated via FActScore (Min et al., 2023), which computes the percentage of atomic facts supported by retrieved passages. The names and retrieved passages are sourced from Wikipedia. The proxy function is defined as _p_[(] _[bg)]_ ( _**a** |_ _**q**_ ) = FActScore( _**a** |_ _**q**_ ) _∈_ [0 _,_ 1]. 

- **COM**[2] (Fang et al., 2024) is a multiple-choice commonsense reasoning benchmark. We use Vera (Liu et al., 2023) to obtain the commonsense plausibility scores for each option. The proxy function is defined as _p_[(] _[cs)]_ ( _**a** |_ _**q**_ ) = Vera( _**a** |_ _**q**_ ) _∈_ [0 _,_ 1]. 

We sample 625, 625, 625, and 380 questions from each dataset, each split into training sets _Dtrain_ , validation sets _Dval_ , and test sets _Dtest_ with an approximately 8:1:1 ratio. We then generate _DWoW_ from _Dtrain_ employing the pipeline in §2.2. _Dtest_ is used for evaluation and _Dval_ for hyperparameter tuning. We also experiment on datasets without clear right-wrong distinctions in Appendix A. 

**Evaluation and Metrics** For the first research question of eliciting wrong-over-wrong preferences, we compare against the proxy ground-truth preferences _f_[ˆ] ( _**a**_ 1 _≻_ _**a**_ 2 _|_ _**q**_ ). The accuracy of LLM-generated wrong-over-wrong preferences is calculated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only_images/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only.pdf-0005-11.png)


For the second research question of wrong-over-wrong alignment, we evaluate improvements across three objectives after aligning with _DWoW_ : 

- **Less Wrong** : We employ the average proxy function scores of all wrong answers: _pwrong_ = 1 

- _Nwrong_ � _Dtest_ � _**a** i_ = _**a** gt[p]_[(] _**[a]**[i][|]_ _**[q]**_[)][,][where] _[N][wrong]_[is][the][number][of][wrong][answers][and] _**[a]**[gt]_[indicates] the correct answer. 

- number of all answers. **More Correct** : We employ task accuracy: Acc = _N_ 1 � _Dtest_ � _**a** i_[1][(] _**[a]**[i]_[=] _**[a]**[gt]_[)][,][where] _[N]_[is][the] 

- **Better Calibration** : We extract the confidence of LLM answers with exp(NLL( **a** _|_ **q** )) _∈_ [0 _,_ 1], create 10 bins of [0 _._ 1 _i,_ 0 _._ 1( _i_ + 1)], and employ Expected Calibration Error (ECE) (Guo et al., 2017) to measure the calibration quality. 

5 

|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br><br><br>|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br><br><br>|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br><br><br>|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br><br><br>|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br><br><br>|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br><br><br>|
|---|---|---|---|---|---|
|**g**<br>KC BG COM2 NLG||KC BG COM2 NLG||KC BG COM2 NLG|Overall|
||**EVALUATOR-INDEPENDENT**|||||
|HEURISTIC<br>_M50_<br>.483 .480 <br>_M10_<br>.502 .425 <br>CONSISTENCY<br>_M50_<br>.500<br>-<br>_M10_<br>.447<br>-|.498<br>.492<br> .492<br>.514<br>.505<br>.582<br>.441<br>.578|.546<br>.589<br>.526<br>.506|.519 .482<br>.463<br>.568 .489<br>.420<br>-<br>.470<br>.605<br>-<br>.556<br>**.695**|.491 .459 .444<br>.503<br>.500 .408 .380<br>.533<br>.434<br>-<br>.565<br>.548<br>.600<br>-<br>.423<br>.494|.488<br>.474<br>.559<br>.566|
||**LLAMA3-8B A**||**S EVALUATOR**|||
|PAIRWISE<br>all<br>.498 .492 <br>flter<br>.518 .566 <br>LOGITS<br>_M50_<br>.541 .568 <br>_M10_<br>.559 .669 <br>SCORE<br>_M50_<br>.621 .552 <br>_M10_<br>.654 .551|.455<br>.486<br> .437<br>.510<br> .524<br>.505<br> .432<br>.528<br> .493<br>.554<br> .458<br>.579|.481<br>.528<br>.570<br>.571<br>.643<br>.701|.488 .530<br>.533<br>.479 .549<br>.582<br>.531 .482<br>.464<br> .649<br>.496<br>.427<br>.502 .574<br>.580<br>.485 .659<br>.524|.509 .499 .468<br>.503<br>.564 .419 .491<br>.525<br>.566 .491 .419<br>.577<br>.400 .444 .310<br>.630<br>.604<br>.514 .626<br>.424<br>**.800 .632 .662**<br>.500|.496<br>.533<br>.532<br>.582<br>.546<br>.558|
||**GPT-3.5 AS EVALUATOR**|||||
|PAIRWISE<br>all<br>.512 .493 <br>flter<br>.531 .631 <br>LOGITS<br>_M50_<br>-<br>-<br>_M10_<br>-<br>-<br>SCORE<br>_M50_<br>.424 .566 <br>_M10_<br>.585 .632|.472<br>.500<br> .502<br>.231<br>-<br>-<br>-<br>-<br> .473<br>.552<br> .517<br>.555|.504 .474 .531<br>.512<br>.520 .500 .493<br>.531<br>.548 .511 .570<br>.475<br>.541 .538 .570<br>.430<br>.578 .522 .608<br>.600<br>.583 .550 **.718**<br>.575||.463 .492 .466<br>.502<br>.387 .437 .433<br>.400<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>.264 .502 .503<br>.551<br>.200 .546 **.662**<br>.573|.494<br>.536<br>.507<br>.505<br>.547<br>.590|
||**GPT-4O AS EVALUATOR**|||||
|PAIRWISE<br>all<br>.605 .593 <br>flter<br>.691.689<br>LOGITS<br>_M50_<br>-<br>-<br>_M10_<br>-<br>-<br>SCORE<br>_M50_<br>.733<br>.677 <br>_M10_<br>**.793 .795**|.507<br>.551<br>.533<br>.602<br>-<br>-<br>-<br>-<br> **.544**<br>.605<br> .534<br>**.652**|.646 .512 .515<br>.577<br>.712 .536 .558<br>.661<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>.793<br>.591 .617<br>.661<br>**.835 .655** .711<br>.684||.434 .501 .526<br>.537<br>.417 .490 .604<br>.549<br>.491 .539 .486<br>.572<br>.200 .584 .507<br>.591<br>.547 .520 .581<br>**.639**<br>.400.586<br>.520<br>.578|.562<br>.624<br>.544<br>.574<br>.641<br>**.709**|



Table 1: Accuracy of LLM-generated wrong-over-wrong preferences _AccWoW_ . The three LLMs across on top are employed to generate answers ( _generator_ ). Best results are in **bold** , second best are in underline, and incompatible or unavailable results are denoted as “-”.[3] For pairwise comparison, margin “all” is without consistency checks and margin “filter” is with consistency checks. Overall accuracy is a weighted average across all datasets by the number of wrong-over-wrong pairs in each dataset. LLMs _do_ have preliminary capabilities to distinguish varying shades of wrong with up to Acc _WoW_ = 0.709 achieved by the score-based approach, specifically _fGPT-4o_[(] _[s]_[)][with] _[ M][10]_[.] 

## 4 RESULTS 

**Eliciting Wrong-over-Wrong Preferences** We present the accuracy of LLM-generated wrongover-wrong preferences in Table 1. 

- **Eliciting wrong-over-wrong preference from LLMs is feasible.** Most (approach, LLM) combinations yield wrong-over-wrong preferences that are significantly better than random guess: the average Acc _WoW_ is 0.553 across datasets, and the best Acc _WoW_ is 0.709 achieved by _fGPT-4o_[(] _[s]_[)][with] a margin of _M10_ . This suggests that LLMs, with the right approach, _do_ possess preliminary capabilities to distinguish various shades of wrong. 

- **Knowledge-based tasks are easier while commonsense is most challenging.** The average Acc _WoW_ across all methods for Knowledge Crosswords, Bio Generation, COM[2] , and NLGraph are 0.578, 0.560, 0.522, and 0.540 respectively. This suggests that LLMs have varying wrong-overwrong judgement capabilities across domains: performance is best on Knowledge Crosswords, a multi-hop knowledge QA dataset, while worst on COM[2] , a commonsense reasoning task. The ability to determine wrong-over-wrong may be impacted by LLMs’ base capability on respective task; Figure 2 shows a positive correlation between model accuracy on the task and Acc _WoW_ . 

> 3Answers for Bio Generation questions are open-ended and we cannot compute repetition; GPT-3.5 and GPT-4O are not open models for calculating logits and NLL. 

6 

|**Method**<br>KC<br>BG<br>COM2<br>NLG<br>_pwrong↑_Acc_↑_ECE_↓pwrong↑_Acc_↑_ECE_↓pwrong↑_Acc_↑_ECE_↓pwrong↑_Acc_↑_ECE_↓_|**Method**<br>KC<br>BG<br>COM2<br>NLG<br>_pwrong↑_Acc_↑_ECE_↓pwrong↑_Acc_↑_ECE_↓pwrong↑_Acc_↑_ECE_↓pwrong↑_Acc_↑_ECE_↓_|
|---|---|
|ORIGINAL<br>.466<br>.555<br>.235|.532<br>.027<br>.576<br>.312<br>.669<br>.053<br>.750<br>.142<br>.649|
||**SELF-GENERATOR**|
|PAIRWISEflter<br>.475<br>**.627**<br>**.096**<br>SCORE_M50_<br>.529<br>.597<br>.251<br>SCORE_M10_<br>.532<br>.584<br>.315<br>ORACLE<br>.529<br>.576<br>.279|.670<br>.059<br>.500<br>.326<br>.690<br>.049<br>.806<br>.179<br>.493|
||.661<br>.043<br>.580<br>.325<br>.660<br>.039<br>.800<br>.203<br>.551|
||.682<br>.075<br>.561<br>**.357**<br>.681<br>**.020**<br>**.847**<br>.292<br>.578|
||**.695**<br>.108<br>**.440**<br>.330<br>.689<br>.064<br>.846<br>.182<br>.596|
||**MIX-GENERATOR**|
|PAIRWISEflter<br>.533<br>.574<br>.201<br>SCORE_M50_<br>.528<br>.590<br>.175<br>SCORE_M10_<br>.520<br>.565<br>.273<br>ORACLE<br>**.537**<br>.581<br>.185|.634<br>.075<br>.535<br>.355<br>**.698**<br>.048<br>.832<br>.192<br>.538|
||.619<br>.065<br>.523<br>.329<br>.669<br>.067<br>.827<br>.221<br>.585<br>.687<br>**.129**<br>.560<br>.346<br>.677<br>.065<br>.843<br>**.303**<br>.522<br>.691<br>.086<br>.472<br>.328<br>.697<br>.067<br>.832<br>.226<br>**.474**|



Table 2: Evaluation of wrong-over-wrong alignment on less wrong ( _pwrong_ ), more correct (Acc), and better calibration (ECE). The best results are in **bold** , second best are in underline, and green background indicates improvement over the original LLAMA3-8B. “Self-Generator” indicates that wrong-over-wrong pairs are generated from only LLAMA3-8B while “Mix-Generator” uses all 3 LLMs’ answers. “Oracle” means aligning with proxy “ground-truth” wrong-over-wrong preference _f_[ˆ] . Wrong-over-wrong alignment is helpful across the board, with up to 0.163, 0.161, and 0.175 improvement in reducing wrongness, increasing correct answers, and improving calibration. 

- **Scored-based elicitation is the best method.** _fGPT-4o_[(] _[s]_[)][with] _[M][10]_[achieves][the][best][Acc] _[WoW]_[.][The] average Acc _WoW_ across LLMs for consistency-based with _M10_ , logits-based with _M10_ , pairwise comparison with consistency checks, score-based with _M10_ , and heuristic are 0.554, 0.566, 0.564, 0.619, and 0.474, indicating that the score-based method outperforms all other approaches by at least 9.4%. This suggests that the token probabilities of aligned LLMs might not be well-calibrated (Sorensen et al., 2024; Feng et al., 2024) for wrong-over-wrong contexts, while LLM-as-a-judge methodologies (Li et al., 2024a) with score-based prompting offer the most promising solution. 

- **Consistency checks and score margins are helpful.** For the pairwise comparison method, we notice that LLMs are sensitive to response order and applying consistency checks by removing inconsistent judgements due to order-flipping improves 9.0% Acc _WoW_ on average. Employing score margins to only consider the most separable cases also improves Acc _WoW_ for logits-based, consistency-based, and score-based methods, with average improvements across datasets from _M50_ to _M10_ being 2.1%, 1.3%, and 6.9%, and from _M100_ to _M10_ being 5.8%, 72.4%, and 65.3%. The significant improvement from _M100_ to _M10_ is due to prevalent cases where the LLM gives the same score for wrong answers with different levels of wrongness. 

- **Self-evaluation is subpar.** LLMs may not be good at producing accurate wrong-over-wrong preferences on their own generation. While GPT-4O provides good wrong-over-wrong preferences on answers generated by LLAMA3-8B and GPT-3.5, accuracy suffers for its own answers, with a 21.1% and 19.7% drop for score-based with _M_ 10 and pairwise comparison with consistency checks on average across datasets. This echos findings that LLM self-critique and self-correction might not be satisfactory (Valmeekam et al., 2023; West et al., 2024; Huang et al., 2024a). More interestingly, employing weak LLMs to evaluate strong LLMs may be helpful (Khan et al., 2024), with LLAMA3-8B achieving 0.649 Acc _wow_ on answers generated by GPT-4O. 

**Alignment with Wrong-over-Wrong Preferences** We present the evaluation of wrong-overwrong alignment in Table 2. 

- **Wrong-over-wrong alignment helps to reduce wrongness, produce correct answers, and improve calibration.** After wrong-over-wrong alignment, LLAMA3-8B improves on average across datasets on ∆ _pwrong_ , ∆Acc, and _−_ ∆ECE by 0.074, 0.045, and 0.044. This validates that wrong-over-wrong alignment is moderately helpful across the three objectives. Even though we only align on wrong answers, we still magically end up making LLM generate 4.5% more cor- 

7 

rect answers on average. This finding highlights the potential of alignment with wrong answers generated by LLMs that come in large quantities. 

- **Effect of wrong-over-wrong alignment varies across datasets.** We observe substantial improvement on Bio Generation and NLGraph with up to 0.163, 0.161, and 0.175 for ∆ _pwrong_ , ∆Acc, and _−_ ∆ECE. However, relatively less improvement is observed on Knowledge Crosswords and COM[2] , with up to 0.071, 0.073, and 0.138 for ∆ _pwrong_ , ∆Acc, and _−_ ∆ECE. This may be because Bio Generation and NLGraph are _open-ended_ questions and there is more space for improvement compared to _multiple-choice_ questions in Knowledge Crosswords and COM[2] where wrong options are limited and LLMs are confined to pre-defined options. 

- **Score-based methods achieve the least wrong and most correct answers while pairwise comparison achieves the best calibration.** Score-based preferences with _M10_ on self-generated data achieve the best average improvement of ∆ _pwrong_ = 0.090 across datasets. _fGPT-4o_[(] _[s]_[)][with] _[M][10]_[on] mix-generated data has the best average improvement of ∆Acc = 0.070. This suggests that the score-based method has the best Acc _WoW_ and is most helpful in reducing wrongness and producing correct answers. However, _fGPT-4o_[(] _[p]_[)][with][consistency][checks][has][the][best][average][calibration] improvement of _−_ ∆ECE = 0.094. We hypothesize that the margin used in score-based methods, while improving Acc _WoW_ by avoiding cases with similar degrees of wrongness, also indicates the models’ tendency to avoid challenging distinctions, which in turn harms calibration. 

## 5 ANALYSIS 

**Task Utility and Wrong-over-Wrong Judgement** We explore the impact of two factors on the reliability of wrong-over-wrong judgements: accuracy of the evaluator’s generated answers ( _task accuracy_ ) and the evaluator’s confidence in its generated answers ( _task confidence_ ). We visualize the correlations and present Pearson correlation coefficients (Sedgwick, 2012) in Figure 2. We observe a positive correlation between task accuracy and Acc _WoW_ , but a negative correlation between task confidence and Acc _WoW_ . This suggests that models that perform well on the task are also good at distinguishing various shades of wrong, while over-confident and undercalibrated models harm wrong-over-wrong preferences. We also find that LLM struggles to differentiate two answers with close wrongness levels in Appendix A. 

**Preference Accuracy and Alignment Improvement** We examine how the quality of wrong-over-wrong judgements Acc _WoW_ is related to improvements through wrong-over-wrong alignment. Figure 3 demonstrates a weak positive relationship between Acc _WoW_ and improvement in ∆ _pwrong_ , and no significant correlation between Acc _WoW_ and improvement in ∆Acc or _−_ ∆ECE. Surprisingly, this suggests that improvement resulting from wrong-over-wrong alignment is nuanced and is not sensitive to the absolute accuracy of wrong-over-wrong preference data. There is also a clear positive relationship between Acc _WoW_ and ∆ _pwrong_ , ∆Acc, and ECE on the Bio Generation dataset, indicating that sensitivity to wrongover-wrong judgement quality varies by dataset. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only_images/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only.pdf-0008-06.png)


**----- Start of picture text -----**<br>
KC COM²<br>BG NLG<br>0.8<br>0.6<br>0.4 P_all = 0.280<br>P_KC = 0.173<br>P_BG = 0.542<br>P_COM² = 0.115<br>0.2 P_NLG = 0.800<br>0.2 0.4 0.6 0.8<br>task accuracy<br>0.8<br>0.6<br>0.4 P_all = -0.172<br>P_KC = -0.566<br>P_BG = -0.434<br>P_COM² = -0.082<br>0.2 P_NLG = -0.655<br>0.5 0.6 0.7 0.8 0.9<br>task confidence<br>wow<br>Acc<br>wow<br>Acc<br>**----- End of picture text -----**<br>


Figure 2: Correlation between task accuracy, confidence and Acc _WoW_ of score-based eliciting with _M10_ . Data points are from all 3 LLMs we used to elicit wrong-over-wrong preferences. _P_ stands for Pearson correlation coefficient. The ability to elicit wrong-overwrong preferences is positively correlated with task ability but negatively correlated with confidence. 

**Right-over-Wrong Alignment** We conduct experiments on right-over-wrong preferences and a 50:50 mix of right-over-wrong and wrong-over-wrong preferences. The results in Table 3 reveal that: (1) Right-over-wrong alignment has the best average improvement in more correct ∆ _pwrong_ = 0.090 on mix-generated data, while on self-generated data, wrong-over-wrong alignment yields the best ∆ _pwrong_ = 0.068. Mixing generators is especially helpful to right-over-wrong alignment, with 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only_images/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only.pdf-0009-00.png)


**----- Start of picture text -----**<br>
KC BG COM² NLG<br>0.15 0.15 0.15<br>0.10<br>0.10 0.10<br>0.05<br>0.05 0.05 0.00<br>0.05<br>0.00<br>0.00<br>0.4 0.6 0.8 1.0 0.4 0.6 0.8 1.0 0.4 0.6 0.8 1.0<br>Accwow Accwow Accwow<br>Pearson=0.260, R²=0.068 Pearson=0.002, R²=0.000 Pearson=-0.013, R²=0.000<br>pwrong  Acc  ECE<br>**----- End of picture text -----**<br>


Figure 3: Correlation between Acc _WoW_ and improvement after wrong-over-wrong alignment in less wrong ∆ _pwrong_ , more correct ∆Acc, and better calibration _−_ ∆ECE. Data points are sourced from all 4 methods ( _fGPT-4o_[(] _[p]_[)][with][consistency][checks,] _[f] GPT-4o_[ (] _[s]_[)][with] _[M][50]_[and] _[M][10]_[,][and][oracle] _[f]_[ˆ][),][and][the] oracle method is considered as Acc _WoW_ = 1.0. The black line is the linear regression on all four datasets while the green line is the linear regression on Bio Generation dataset. Wrong-over-wrong alignment is not sensitive to the accuracy of wrong-over-wrong preference. 

average ∆Acc improved by 55.7%. (2) Right-over-wrong alignment has the best average improvement in more correct ∆Acc = 0.134, while a mixture of right-over-wrong and wrong-over-wrong preferences achieves the best average improvement in calibration _−_ ∆ECE = 0.107. This partly resolves the previous finding that alignment hurts calibration (Sorensen et al., 2024; Feng et al., 2024): we hypothesize that this is due to missing wrong-over-wrong preferences in alignment data that make LLMs unable to disentangle varying shades of wrong for fine-grained calibration. 

**Generalization to Unseen Data** We exam- 

ine how wrong-over-wrong alignment can generalize to other unseen tasks in the same domain. We employ Hellaswag (Zellers et al., 2019) and the maximum flow subtask in NLGraph (Wang et al., 2023a) as two unseen datasets while being in the same domain as the datasets employed for alignment tuning, COM[2] and shortest path. Results in Table 5 demonstrate that wrong-over-wrong alignment could indeed generalize to unseen data, with an average improvement of 0.118, 0.002, and 0.029 on ∆ _pwrong_ , ∆Acc, and _−_ ∆ECE. 

||||**KC**<br>**COM**2|**KC**<br>**COM**2|**KC**<br>**COM**2|**NLG**|**NLG**||
|---|---|---|---|---|---|---|---|---|
||**Method**|_pwrong↑_Acc_↑_ECE_↓pwrong↑_Acc_↑_ECE_↓pwrong↑_Acc_↑_ECE_↓_|||||||
||ORIGINAL|.466|.555 .235<br>.312|.669|.053|.750|.142|.649|
||BESTW<br>R+W(PAIRWISE)<br>R+W(SCORE_M50_)|.529<br>.493<br>.540|.597 .251<br>.355<br>.698<br>**SELF-GENERATOR**<br>.706<br>.065<br>.359<br>.705 <br>.690<br>.055<br>.335<br>.673||.048<br> .057<br>.037|.847<br>.806<br>.814|.292<br>.179<br>.139|.578<br>.493<br> .667|
||R+W(SCORE_M10_)<br>R+W(ORACLE)|.503<br>**.607**|.665<br>.174<br>.340<br>.785<br>.060<br>.357|.687 <br>.703|.070<br> .055|.815<br>.760|.137 <br>.158|.650<br>.549|
||R|.579|**.805**<br>.079<br>.373|.692|**.029**|.777|.153|.652|
||||**MIX-GENERATOR**||||||
||R+W(PAIRWISE)<br>R+W(SCORE_M50_)<br>R+W(SCORE_M10_)<br>R+W(ORACLE)<br>R|.530<br>.536<br>.559<br>.567<br>.568|.705<br>.053<br>.327<br>.705<br>.094<br>.326<br>.711<br>.104<br>.343<br>.740<br>**.025**<br>.344<br>.787<br>.110<br>**.374**|.658 <br>.716 <br>.655<br>**.748** <br>.711|.091<br> .070<br>.034<br> .113<br> .183|.825<br>.836<br>.842<br>.826<br>**.856**|.218<br>.263<br>**.308**<br>.274<br>.268|.493<br>.548<br>.496<br>**.490**<br>.537|



**Qualitative Examples** We present qualitative examples in Table 4, showing that WoWaligned models could generate less wrong, more correct, and better calibrated answers. More examples are in Appendix A. 

## 6 RELATED WORK 

Table 3: Evaluation of right-over-wrong alignment on less wrong ( _pwrong_ ), more correct (Acc), and better calibration (ECE). “W” and “R” stand for wrong-over-wrong and right-over-wrong data. Wrong-over-wrong alignment is a good supplement to existing right-over-wrong alignment and best employed in conjunction. 

**LLM Alignment** In recent years, aligning LLMs with human preferences has become an important research question, driven by the need for safety (Dai et al., 2024; Qi et al., 2024; Huang et al., 2024b), honesty (Yang et al., 2023; Wen et al., 2024), factuality(Liang et al., 2024; Lin et al., 2024), diversity (Ding et al., 2024), etc. The initial methods, such as Reinforcement Learning from Human Feedback (RLHF) (Ziegler et al., 2019; Stiennon et al., 2020; Ouyang et al., 2022; Munos et al., 2024; Chakraborty et al., 2024), leverage human preferences to train reward models and employ algorithms such as Proximal Policy Optimization (PPO) (Schulman et al., 2017) for alignment. Recent alignment methods move away from explicit reward models, as seen in Direct Preference Optimization (DPO) (Rafailov et al., 2023). Alongside this, Reinforcement Learning from AI Feedback (RLAIF) (Bai et al., 2022; Lee 

9 

|**Question**|In an undirected graph, the nodes are numbered from 0 to 4, and the edges are rep-|
|---|---|
||resented by (start node, end node, weight): (0, 2, 2), (0, 5, 4), (0, 3, 1), (0, 1, 3), (1,|
||2, 4), (1, 5, 1), (1, 4, 4), (2, 4, 3), (2, 3, 1), (3, 5, 2), (3, 4, 3), (4, 5, 2). Q: Give the|
||shortest path from node 0 to node 1. Please also give the total weight of the shortest|
||path.[Correct answer: 4 (0_→_3_→_4).]|
|**Original Answer**|[CoT steps]. The fnal answer is: shortest path: 0_→_2_→_3_→_4; total weight: 6.|
|**Less Wrong**|[CoT steps]. Therefore, the answer is: The shortest path from node 0 to node 4 is 0_→_|
||3_→_2_→_4, with a total weight of 5.[The path found is shorter: 6_→_5.]|
|**More Correct**|[CoT steps]. Final answer: The shortest path from node 0 to node 4 is Node 0_→_Node|
||3_→_Node 4, with a total weight of 4.[Successfully fnd the shortest path.]|
|**Better Calibrated**|[CoT steps]. The total weight of the shortest path is: 0_→_2 (weight 2), 2_→_3 (weight|
||1), 3_→_4 (weight 3). Total weight: 2 + 1 + 3 = 6.[Even though the answer is equally|
||wrong compared to the original answer, NLL increases from 0.255_→_0.373.]|



Table 4: Qualitative examples on NLGraph where WoW-aligned models become less wrong, more correct, and better calibrated on NLGraph. We provide explanations in [red]. 

et al., 2023) and LLM-as-a-Judge (Dubois et al., 2023; Bai et al., 2023) frameworks have introduced AI-driven feedback systems, allowing LLMs to give feedback and improve themselves with minimal human intervention. Techniques such as Self-Alignment (Li et al., 2024b; Wang et al., 2024c; Yuan et al., 2024; Sun et al., 2024a), which involves models evaluating and aligning with their own outputs, represent a further step towards scalable oversight. Instead of alignment objective, we investigate an orthogonal research question to the improvement in alignment approaches and focus on the preference data: in challenging tasks and contexts when correct and high-quality answers are not readily available (Baheti et al., 2024; Agarwal et al., 2024), could LLMs distinguish various shades of wrong and align with wrong-over-wrong preferences? Prior work has also established the utility and representativeness of fine-grained and continuous preferences over categorical labels in uncertain (Lin et al., 2022; Tian et al., 2023) and subjective contexts (Sorensen et al., 2024; Feng et al., 2024). 

**Easy-to-Hard Generalization** Most of these 

advancements in alignment benefit from the availability of “correct” answers. In contrast, our work posits that “correct” answers are inevitably unavailable or unreliable for emerging and challenging tasks. Bowman et al. (2022) points out advanced models may engage in tasks that are beyond current evaluation capabilities. Easy-to-hard generalization (Burns et al., 2024; Hase et al., 2024; Xu et al., 2024b; Sun et al., 2024b) introduces a series of solutions hoping to improve LLMs on hard tasks by finetuning on easy in-domain tasks. Our work uniquely points out a new path for training future LLMs on hard tasks when correct answers are unavailable, arguing for wrong-over-wrong alignment based on synthetic preference data as a complementary training objective. 

|**Method**|**HellaSwag**<br>_pwrong↑_<br>Acc_↑_<br>ECE_↓_|**Maximum Flow**<br>_pwrong↑_<br>Acc_↑_<br>ECE_↓_|
|---|---|---|
|ORIGINAL|.230<br>.737<br>.089|.112<br>.069<br>.663|
||**SELF-GENERAT**|**OR**|
|PAIRWISE<br>SCORE_M50_<br>SCORE_M10_<br>ORACLE|.243<br>.729<br>.098<br>.220<br>.679<br>.047<br>.264<br>.719<br>.068<br>.227<br>.729<br>.023|.344<br>.066<br>**.567**|
|||.330<br>.083<br>.637|
|||.342<br>**.109**<br>.673|
|||.151<br>.049<br>.659|
||**MIX-GENERAT**|**OR**|
|PAIRWISE<br>SCORE_M50_<br>SCORE_M10_<br>ORACLE|.247<br>.706<br>.082<br>.204<br>.729<br>.090<br>.250<br>**.771**<br>.117<br>**.267**<br>.753<br>.099|**.348**<br>.094<br>.533|
|||.346<br>.083<br>.621|
|||.326<br>.089<br>.627|
|||.202<br>.074<br>.582|



Table 5: Generalization to unseen datasets, from COM[2] to HellaSwag and from shortest path to maximum flow. While tuned only on COM[2] and shortest path, the aligned models improve on the two unseen datasets as well, with an average improvement of 0.118, 0.002, and 0.029 on ∆ _pwrong_ , ∆Acc, and _−_ ∆ECE. 

## 7 CONCLUSION 

With the growing race towards bigger, better LLMs capable of solving a wider range of tasks, it becomes evident that the availability of carefully curated data is a major bottleneck. Our work investigates the potential to alleviate this limitation by eliciting preferences among wrong answers with an LLM and aligning with these wrong-over-wrong preferences. We empirically investigate wrongover-wrong alignment with seven LLMs and eight datasets. We find that LLMs _do_ have preliminary capability to rank wrong answers and produce reliable wrong-over-wrong preferences. The strongest approach such as score-based LLM-as-a-judge achieves up to 70.9% accuracy across datasets. In ad- 

10 

dition, alignment with such wrong-over-wrong preferences is helpful in calibrating the models, e.g., reducing answer “wrongness” by 9.0%, producing 7.0% more correct answers, and improving overall model calibration by 9.4% on average. We envision future wrong-over-wrong alignment methods as an important addition to advance LLM capabilities on challenging tasks where correct answers or rich annotations are not readily available. 

## LIMITATIONS AND ETHICS STATEMENT 

**Imperfect proxies.** The proxy functions we use to measure correctness are inherently imperfect. While they serve as reasonable approximations for distinguishing varying degrees of wrongness, they are not definitive, e.g. heuristic-based proxies may merely focus on the final answer and overlook the reasoning steps and model-based proxies are inaccurate and may suffer from domain shift problem. As a result, some wrong-over-wrong preferences may be misjudged, potentially affecting the evaluation of Acc _WoW_ and analysis results. 

**Sensitivity to hyperparameters.** The alignment process we employ is sensitive to hyperparameter selection. This dependence can lead to variability in results across different models and datasets, making it challenging to guarantee consistent performance improvements. Further exploration of more robust hyperparameter configurations (Falkner et al., 2018; Arango et al., 2024) is necessary. 

**Limited scope of experimental datasets.** Experiments are conducted on datasets where groundtruth answers exist, and proxy functions can be employed to approximate wrongness. However, this setup doesn’t fully reflect the application scenarios we aim to address — tasks where no clear ground-truth answers are available (Wang et al., 2024a), or the problems are so challenging that even expert human annotators or LLMs might struggle (Welleck et al., 2021). Evaluating the generalizability of our framework in these conditions remains an important avenue for future work. 

**Unique focus on knowledge and reasoning problems.** We focus on tasks and problems with absolute and indisputable correct answers in this work, such as multi-hop QA (Ding et al., 2023) and graph reasoning (Wang et al., 2023a). Consequently, existing alignment works in these domains often assume access to correct answers and construct right-over-wrong pairs for preference learning (Wang et al., 2023c; Cheng et al., 2024; Lin et al., 2024). However, objective “correct answers” are often not feasible in general instruction following tasks, thus they employ human preference data between a pair of responses that is not necessarily right-over-wrong. Our scope is to investigate wrong-over-wrong alignment specifically focusing on the first type of knowledge/reasoning problems with absolute correctness while we leave general instruction following as future work. 

**Incomparable wrongness in social contexts.** In social contexts, wrongness is often subjective and potentially incomparable, which introduces biases and fairness concerns in wrong-over-wrong alignment. For instance, when judging the wrongness of statements related to sensitive topics such as politics or identity, cultural biases may influence the model’s preferences (Xu et al., 2021; Bender et al., 2021; Feng et al., 2023). A statement like “People from _certain_ neighborhoods are more likely to commit crimes” might be judged less wrong than “All people from _certain_ neighborhoods are criminals” but these evaluations reflect specific perspectives that may not be universally shared or appropriate in all contexts. Furthermore, aligning models with such judgments risks reinforcing harmful stereotypes or systemic biases. Ensuring fairness and transparency in these judgments is critical, and models should be designed to recognize when wrongness is subjective and abstain from making harmful comparisons. 

## REPRODUCIBILITY STATEMENT 

We provide all necessary details for the implementation and evaluation of our proposed wrongover-wrong alignment approach in Appendix B. Specifically, Appendix B includes information of 8 datasets splits and preprocessing steps, 7 LLM checkpoints, hyperparameters selection and detailed configuration for each table. Moreover, the prompts for preference elicitation methods are explained in Table 12 and Table 13. Our codes, including scripts for dataset generation, preference eliciting, 

11 

preference accuracy evaluation, alignment, and evaluation of alignment are available publicly in repo https://github.com/yaojh18/Varying-Shades-of-Wrong to facilitate easy verification of our results. 

## REFERENCES 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from selfgenerated mistakes. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Sebastian Pineda Arango, Fabio Ferreira, Arlind Kadra, Frank Hutter, and Josif Grabocka. Quicktune: Quickly learning which pretrained model to finetune and how. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In _International Conference on Artificial Intelligence and Statistics_ , pp. 4447–4455. PMLR, 2024. 

- Ashutosh Baheti, Ximing Lu, Faeze Brahman, Ronan Le Bras, Maarten Sap, and Mark Riedl. Leftover lunch: Advantage-based offline reinforcement learning for language models. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. _arXiv preprint arXiv:2212.08073_ , 2022. 

- Yushi Bai, Jiahao Ying, Yixin Cao, Xin Lv, Yuze He, Xiaozhi Wang, Jifan Yu, Kaisheng Zeng, Yijia Xiao, Haozhe Lyu, et al. Benchmarking foundation models with language-model-as-an-examiner. In _Proceedings of the 37th International Conference on Neural Information Processing Systems_ , pp. 78142–78167, 2023. 

- Hritik Bansal, John Dang, and Aditya Grover. Peering through preferences: Unraveling feedback acquisition for aligning large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In _Proceedings of the 2021 ACM conference on fairness, accountability, and transparency_ , pp. 610–623, 2021. 

- Samuel R Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig Pettit, Scott Heiner, Kamil˙e Lukoˇsi¯ut˙e, Amanda Askell, Andy Jones, Anna Chen, et al. Measuring progress on scalable oversight for large language models. _arXiv preprint arXiv:2211.03540_ , 2022. 

- Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, et al. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. In _Forty-first International Conference on Machine Learning_ , 2024. 

- Daniele Calandriello, Zhaohan Daniel Guo, Remi Munos, Mark Rowland, Yunhao Tang, Bernardo Avila Pires, Pierre Harvey Richemond, Charline Le Lan, Michal Valko, Tianqi Liu, et al. Human alignment of large language models through online preference optimisation. In _Forty-first International Conference on Machine Learning_ , 2024. 

- Souradip Chakraborty, Amrit Bedi, Alec Koppel, Huazheng Wang, Dinesh Manocha, Mengdi Wang, and Furong Huang. PARL: A unified framework for policy alignment in reinforcement learning from human feedback. In _The Twelfth International Conference on Learning Representations_ , 2024. 

12 

- Sahil Chaudhary. Code alpaca: An instruction-following llama model for code generation. _GitHub repository_ , 2023. 

- Qinyuan Cheng, Tianxiang Sun, Xiangyang Liu, Wenwei Zhang, Zhangyue Yin, Shimin Li, Linyang Li, Zhengfu He, Kai Chen, and Xipeng Qiu. Can ai assistants know what they don’t know? In _Forty-first International Conference on Machine Learning_ , 2024. 

- Sayak Ray Chowdhury, Anush Kini, and Nagarajan Natarajan. Provably robust dpo: Aligning language models with noisy feedback. In _Forty-first International Conference on Machine Learning_ , 2024. 

- Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. _Journal of Machine Learning Research_ , 25(70):1–53, 2024. 

- Jeremy Cole, Michael Zhang, Dan Gillick, Julian Eisenschlos, Bhuwan Dhingra, and Jacob Eisenstein. Selectively answering ambiguous questions. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 530–543, 2023. 

- Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe rlhf: Safe reinforcement learning from human feedback. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: efficient finetuning of quantized llms. In _Proceedings of the 37th International Conference on Neural Information Processing Systems_ , pp. 10088–10115, 2023. 

- Li Ding, Jenny Zhang, Jeff Clune, Lee Spector, and Joel Lehman. Quality diversity through human feedback: Towards open-ended diversity-driven optimization. In _Forty-first International Conference on Machine Learning_ , 2024. 

- Wenxuan Ding, Shangbin Feng, Yuhan Liu, Zhaoxuan Tan, Vidhisha Balachandran, Tianxing He, and Yulia Tsvetkov. Knowledge crosswords: Geometric reasoning over structured knowledge with large language models. _arXiv preprint arXiv:2310.01290_ , 2023. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpacafarm: a simulation framework for methods that learn from human feedback. In _Proceedings of the 37th International Conference on Neural Information Processing Systems_ , pp. 30039–30069, 2023. 

- Stefan Falkner, Aaron Klein, and Frank Hutter. Bohb: Robust and efficient hyperparameter optimization at scale. In _International conference on machine learning_ , pp. 1437–1446. PMLR, 2018. 

- Tianqing Fang, Zeming Chen, Yangqiu Song, and Antoine Bosselut. Complex reasoning over logical queries on commonsense knowledge graphs. _arXiv preprint arXiv:2403.07398_ , 2024. 

- Shangbin Feng, Chan Young Park, Yuhan Liu, and Yulia Tsvetkov. From pretraining data to language models to downstream tasks: Tracking the trails of political biases leading to unfair nlp models. In _ACL: Annual Meeting of the Association for Computational Linguistics_ , 2023. 

- Shangbin Feng, Taylor Sorensen, Yuhan Liu, Jillian Fisher, Chan Young Park, Yejin Choi, and Yulia Tsvetkov. Modular pluralism: Pluralistic alignment via multi-llm collaboration. _arXiv preprint arXiv:2406.15951_ , 2024. 

- Jiahui Geng, Fengyu Cai, Yuxia Wang, Heinz Koeppl, Preslav Nakov, and Iryna Gurevych. A survey of confidence estimation and calibration in large language models. In _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pp. 6577–6595, 2024. 

13 

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In _International conference on machine learning_ , pp. 1321–1330. PMLR, 2017. 

- Peter Hase, Mohit Bansal, Peter Clark, and Sarah Wiegreffe. The unreasonable effectiveness of easy training data for hard tasks. _arXiv preprint arXiv:2401.06751_ , 2024. 

- Jiwoo Hong, Noah Lee, and James Thorne. Orpo: Monolithic preference optimization without reference model. _arXiv preprint arXiv:2403.07691_ , 2(4):5, 2024. 

- Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. In _The Twelfth International Conference on Learning Representations_ , 2024a. 

- Weidong Huang, Jiaming Ji, Chunhe Xia, Borong Zhang, and Yaodong Yang. Safedreamer: Safe reinforcement learning with world models. In _The Twelfth International Conference on Learning Representations_ , 2024b. 

- Zixian Huang, Wenhao Zhu, Gong Cheng, Lei Li, and Fei Yuan. Mindmerger: Efficient boosting llm reasoning in non-english languages. _arXiv preprint arXiv:2405.17386_ , 2024c. 

- Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A Smith, Iz Beltagy, et al. Camels in a changing climate: Enhancing lm adaptation with tulu 2. _arXiv preprint arXiv:2311.10702_ , 2023. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- Akbir Khan, John Hughes, Dan Valentine, Laura Ruis, Kshitij Sachan, Ansh Radhakrishnan, Edward Grefenstette, Samuel R Bowman, Tim Rockt¨aschel, and Ethan Perez. Debating with more persuasive llms leads to more truthful answers. In _Forty-first International Conference on Machine Learning_ , 2024. 

- Andreas K¨opf, Yannic Kilcher, Dimitri von R¨utte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Rich´ard Nagyfi, et al. Openassistant conversations-democratizing large language model alignment. In _Proceedings of the 37th International Conference on Neural Information Processing Systems_ , pp. 47669–47681, 2023. 

- Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, et al. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. _arXiv preprint arXiv:2309.00267_ , 2023. 

- Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, hai zhao, and Pengfei Liu. Generative judge for evaluating alignment. In _The Twelfth International Conference on Learning Representations_ , 2024a. 

- Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason E Weston, and Mike Lewis. Self-alignment with instruction backtranslation. In _The Twelfth International Conference on Learning Representations_ , 2024b. 

- Yuxin Liang, Zhuoyang Song, Hao Wang, and Jiaxing Zhang. Learning to trust your feelings: Leveraging self-awareness in llms for hallucination mitigation. _arXiv preprint arXiv:2401.15449_ , 2024. 

- Lichess Team. Lichess database. https://github.com/lichess-org/database, 2023. Accessed: 2023. 

- Sheng-Chieh Lin, Luyu Gao, Barlas Oguz, Wenhan Xiong, Jimmy Lin, Wen-tau Yih, and Xilun Chen. Flame: Factuality-aware alignment for large language models. _arXiv preprint arXiv:2405.01525_ , 2024. 

- Stephanie Lin, Jacob Hilton, and Owain Evans. Teaching models to express their uncertainty in words. _arXiv preprint arXiv:2205.14334_ , 2022. 

14 

- Jiacheng Liu, Wenya Wang, Dianzhuo Wang, Noah A Smith, Yejin Choi, and Hannaneh Hajishirzi. Vera: A general-purpose plausibility estimation model for commonsense statements. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 1264–1287, 2023. 

- Yuxuan Liu, Tianchi Yang, Shaohan Huang, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, and Qi Zhang. Calibrating llm-based evaluator. In _Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LRECCOLING 2024)_ , pp. 2638–2656, 2024. 

- Potsawee Manakul, Adian Liusie, and Mark Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 9004–9017, 2023. 

- Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a referencefree reward. _arXiv preprint arXiv:2405.14734_ , 2024. 

- Ning Miao, Yee Whye Teh, and Tom Rainforth. Selfcheck: Using LLMs to zero-shot check their own step-by-step reasoning. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 12076–12100, 2023. 

- Remi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Cˆome Fiegel, et al. Nash learning from human feedback. In _Forty-first International Conference on Machine Learning_ , 2024. 

- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In _Proceedings of the 36th International Conference on Neural Information Processing Systems_ , pp. 27730–27744, 2022. 

- Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In _Conference on health, inference, and learning_ , pp. 248–260. PMLR, 2022. 

- Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: your language model is secretly a reward model. In _Proceedings of the 37th International Conference on Neural Information Processing Systems_ , pp. 53728–53741, 2023. 

- T. Romstad, M. Costalba, J. Kiiski, G. Linscott, S. Nicolet, S. Geschwentner, and J. VandeVondele. Stockfish, 2021. Version 14. Available online: https://stockfishchess.org. 

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. _arXiv preprint arXiv:1707.06347_ , 2017. 

- Philip Sedgwick. Pearson’s correlation coefficient. _Bmj_ , 345, 2012. 

- Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, et al. A roadmap to pluralistic alignment. _arXiv preprint arXiv:2402.05070_ , 2024. 

15 

- Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learning to summarize from human feedback. In _Proceedings of the 34th International Conference on Neural Information Processing Systems_ , pp. 3008–3021, 2020. 

- Zhiqing Sun, Yikang Shen, Hongxin Zhang, Qinhong Zhou, Zhenfang Chen, David Daniel Cox, Yiming Yang, and Chuang Gan. Salmon: Self-alignment with instructable reward models. In _The Twelfth International Conference on Learning Representations_ , 2024a. 

- Zhiqing Sun, Longhui Yu, Yikang Shen, Weiyang Liu, Yiming Yang, Sean Welleck, and Chuang Gan. Easy-to-hard generalization: Scalable alignment beyond human supervision. _arXiv preprint arXiv:2403.09472_ , 2024b. 

- Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivi`ere, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. _arXiv preprint arXiv:2403.08295_ , 2024. 

- Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 5433–5442, 2023. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- Karthik Valmeekam, Matthew Marquez, and Subbarao Kambhampati. Can large language models really improve by self-critiquing their own plans? In _NeurIPS 2023 Foundation Models for Decision Making Workshop_ , 2023. 

- Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov. Can language models solve graph problems in natural language? In _Proceedings of the 37th International Conference on Neural Information Processing Systems_ , pp. 30840–30861, 2023a. 

- Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. _arXiv preprint arXiv:2305.17926_ , 2023b. 

- Peiyi Wang, Lei Li, Liang Chen, Feifan Song, Binghuai Lin, Yunbo Cao, Tianyu Liu, and Zhifang Sui. Making large language models better reasoners with alignment. _arXiv preprint arXiv:2309.02144_ , 2023c. 

- Tiannan Wang, Jiamin Chen, Qingrui Jia, Shuai Wang, Ruoyu Fang, Huilin Wang, Zhaowei Gao, Chunzhao Xie, Chuou Xu, Jihong Dai, et al. Weaver: Foundation models for creative writing. _arXiv preprint arXiv:2401.17268_ , 2024a. 

- Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. In _Forty-first International Conference on Machine Learning_ , 2024b. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023d. 

- Ziqi Wang, Le Hou, Tianjian Lu, Yuexin Wu, Yunxuan Li, Hongkun Yu, and Heng Ji. Enabling lanuguage models to implicitly learn self-improvement. In _The Twelfth International Conference on Learning Representations_ , 2024c. 

16 

- Sean Welleck, Jiacheng Liu, Ronan Le Bras, Hannaneh Hajishirzi, Yejin Choi, and Kyunghyun Cho. Naturalproofs: Mathematical theorem proving in natural language. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)_ , 2021. 

- Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu, Yulia Tsvetkov, Bill Howe, and Lucy Lu Wang. The art of refusal: A survey of abstention in large language models. _arXiv preprint arXiv:2407.18418_ , 2024. 

- Peter West, Ximing Lu, Nouha Dziri, Faeze Brahman, Linjie Li, Jena D Hwang, Liwei Jiang, Jillian Fisher, Abhilasha Ravichander, Khyathi Chandu, et al. The generative ai paradox:“what it can create, it may not understand”. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, et al. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations_ , pp. 38–45, 2020. 

- Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for language model alignment. _arXiv preprint arXiv:2405.00675_ , 2024. 

- Albert Xu, Eshaan Pathak, Eric Wallace, Suchin Gururangan, Maarten Sap, and Dan Klein. Detoxifying language models risks marginalizing minority voices. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 2390–2397, 2021. 

- Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. _arXiv preprint arXiv:2304.12244_ , 2023. 

- Shusheng Xu, Wei Fu, Jiaxuan Gao, Wenjie Ye, Weilin Liu, Zhiyu Mei, Guangju Wang, Chao Yu, and Yi Wu. Is dpo superior to ppo for llm alignment? a comprehensive study. In _Forty-first International Conference on Machine Learning_ , 2024a. 

- Zhuoyan Xu, Zhenmei Shi, Junyi Wei, Fangzhou Mu, Yin Li, and Yingyu Liang. Towards few-shot adaptation of foundation models via multitask finetuning. In _The Twelfth International Conference on Learning Representations_ , 2024b. 

- Kevin Yang, Dan Klein, Asli Celikyilmaz, Nanyun Peng, and Yuandong Tian. Rlcd: Reinforcement learning from contrastive distillation for lm alignment. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Yuqing Yang, Ethan Chern, Xipeng Qiu, Graham Neubig, and Pengfei Liu. Alignment for honesty. _arXiv preprint arXiv:2312.07000_ , 2023. 

- Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason E Weston. Self-rewarding language models. In _Forty-first International Conference on Machine Learning_ , 2024. 

- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pp. 4791–4800, 2019. 

- Yizhuo Zhang, Heng Wang, Shangbin Feng, Zhaoxuan Tan, Xiaochuang Han, Tianxing He, and Yulia Tsvetkov. Can llm graph reasoning generalize beyond pattern memorization? _arXiv preprint arXiv:2406.15992_ , 2024. 

- Hao Zhao, Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion. Long is more for alignment: A simple but tough-to-beat baseline for instruction fine-tuning. In _Forty-first International Conference on Machine Learning_ , 2024. 

17 

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. In _Proceedings of the 37th International Conference on Neural Information Processing Systems_ , pp. 46595–46623, 2023. 

- Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: less is more for alignment. In _Proceedings of the 37th International Conference on Neural Information Processing Systems_ , pp. 55006–55021, 2023. 

- Jin Peng Zhou, Yuhuai Wu, Qiyang Li, and Roger Baker Grosse. REFACTOR: Learning to extract theorems from proofs. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. _arXiv preprint arXiv:1909.08593_ , 2019. 

18 

## A ANALYSIS (CONT.) 

**More general LLMs as evaluators** We present the rest of elicited wrong-over-wrong preferences, including evaluators being GEMINI-FLASH, GEMINI-PRO, and MISTRAL7B. The results in Table 9 demonstrate that GPT-4O is still the best LLM at evaluating wrong-over-wrong preference and surpasses other LLMs by at least 12.0% on overall Acc _WoW_ . We can also see leveraging GEMINIFLASH, GEMINI-PRO to evaluate answers generated by GPT-4O successfully mitigate the poor accuracy of self-evaluation. 

**Domain-specific LLMs as evaluators** To better figure out how Acc _wow_ is relate to evaluators, we instruction finetune GEMMA-7B on 7 different datasets (Ivison et al., 2023), namely Code-Alpaca (Chaudhary, 2023), FLAN (Chung et al., 2024), Gemini-Alpaca, LIMA (Zhou et al., 2023), Open Assistant 1 (K¨opf et al., 2023), Science literature and WizardLM Evol-Instruct V2 (Xu et al., 2023). Then we elicit wrong-over-wrong preference from them on Knowledge Crosswords, Bio Generation, COM[2] and NLGraph datasets with GPT-3.5 being generator. We only experiment on score-based eliciting. Results in Table 6 demonstrate that domain-specific LLMs as evaluators show considerable variance in their wrong-over-wrong preference elicitation accuracy across datasets. For example, GEMMA-7B fintuned on FLAN achieves the best wrongover-wrong accuracy on Bio Generation and NLGraph while unable to follow instructions on Knowledge Crosswords and COM[2] . This suggests finetuning as a way to improve LLMbased wrong-over-wrong accuracy in certain domains. 

**More challenging datasets** We investigate LLMs’ capability to give wrong-over-wrong preferences on more challenging datasets, NLGraph (full) and ChessPuzzle (CP). NLGraph 

## Margin KC BG COM[2] NLG overall 

|||**_Code-Alpaca_**|**_Code-Alpaca_**||
|---|---|---|---|---|
||_M100_<br>_M50_<br>_M10_|.183 .258 .221<br>.367 .516 .442<br>.425 .544 .466|.174<br>.348<br>.482|.220<br>.439<br>.505|
|||**_FLAN_**|||
||_M100_<br>_M50_<br>_M10_|.000 .098 .000<br>.000 .197 .000<br>.000**.585** .000|.096<br>.193<br>**.610**|.080<br>.160<br>.487|
|||**_Gemini-Alpaca_**|||
||_M100_|.297 .356 .396|.347|.349|
||_M50_|.512 .492 .555|.527|.510|
||_M10_|.534 .458 **.679**<br>**_LIMA_**|.575|.519|
||_M100_|.110 .079 .215|.099|.098|
||_M50_|.221 .159 .428|.198|.197|
||_M10_|**.610**.508 .343<br>.486<br>**_Open Assist_**||.501|
||_M100_<br>_M50_|.240 .175 .220<br>.481 .351 .441|.180<br>.360|.187<br>.375|
||_M10_|.500 .563 .602<br>**_Science_**|.553|**.555**|
||_M100_|.241 .236 .557|.248|.262|
||_M50_|.482 .472 .561|.482|.482|
||_M10_|.515 .557 .574<br>**_WizardLM_**|.470|.525|
||_M100_<br>_M50_|.194 .278 .322<br>.388 .508 .474|.239<br>.478|.259<br>.482|
||_M10_|.579 .551 .523|.479|.529|



Table 6: Accuracy of wrong-over-wrong preference elicited from GEMMA-7B fintuened on different instruction datasets. Finetuned LLMs have very different wrong-over-wrong preference accuracy across domains. 

(full) employs all min/max questions in NLGraph, including very challenging questions like “finding the maximum flow between two nodes in a weighted undirected graph” and “finding an assignment of jobs to applicants such that the maximum number of applicants find the job they are interested in”. Chess Puzzle contains chess puzzles from the lichess.org website (Lichess Team, 2023), where the inputs are board state represented in Forsyth–Edwards Notation (FEN) and LLMs are asked to generate the optimal next move for the current player. However, we find that LLMs tend to give invalid moves, which makes it hard to evaluate wrongness. Thus we employ Stockfish (Romstad et al., 2021), a SOTA chess engine, to generate the 4 best moves for the current player and ask an LLM to choice from given options. Evaluators’ task accuracy is 0.204 (random guess 0.000) on NLGraph (full) and 0.461 (random guess 0.250) on Chess Puzzle. The correctness proxy for NLGraph (full) is _p_[(] _[nl)]_ ( _**a** |_ _**q**_ ) = _−|v_ _**a** − vgt|_ , where _v_ _**a**_ is the extracted final answer and _vgt_ is the ground-truth answer. The correctness proxy for Chess Puzzle is _p_[(] _[nl)]_ ( _**a** |_ _**q**_ ) = _wrStockfish_ ( _**a** |_ _**q**_ ), where _wrStockfish_ is the win rate predicted by Stockfish. Experiment results in Table 10 demonstrate that the evaluator’s 

19 

Acc _WoW_ drops significant on challenging tasks, e.g. _fGPT-4o_[(] _[s]_[)][with] _[ M][10]_[drops by 23.7%. The average] Acc _WoW_ across datasets for pairwise comparison with consistency check, logits-based with _M10_ and scored-based _M10_ are 0.585, 0.546, 0.550, indicating pairwise comparison may be more suitable for challenging tasks. 

**Batch size for score-based eliciting** We also investigate the effect of batch size _b_ on score-based preference eliciting. We employ LLAMA3-8B as generator and GPT-4O as evaluator. For simplicity, we only experiment on Knowledge Crosswords and NLGraph (shortest path) dataset. Experiment results in Table 7 indicate that 5 is the best batch size for our experiment setup, which verifies our assumption that batch size has a trade-off effect on scored-based eliciting. 

**Consistency of pairwise comparison** Pairwise comparison eliciting displays high sensitivity to positional bias. We report the proportions of consistent wrong-over-wrong preference after flipping for LLAMA3-8B, GPT-3.5 and GPT-4O are 0.286, 0.128, 0.598. Stronger LLMs can generate more consistent preference, and small LLMs are not reliable evaluators. 

**Wrongness margin affecting** Acc _**WoW**_ The accuracy of wrong-over-wrong judgement can also be affected by “wrongness margin” of a wrong-over-wrong pair. For example, a closeto-correct answer (e.g. “the sun almost all rises in the east”) is obviously better than an absolutely wrong answer (e.g. “the sun rises in the west”), while two wrong answers with close 

||Margin|KC<br>NLG<br>**_batch size = 1_**|Overall|
|---|---|---|---|
||_M100_<br>_M50_<br>_M10_|0.447<br>0.318<br>0.643<br>0.596<br>0.719<br>**0.701**|0.383<br>0.620<br>0.710|
||_M100_<br>_M50_<br>_M10_|**_batch size = 2_**<br>0.482<br>0.364<br>0.654<br>0.575<br>0.708<br>0.623|0.423<br>0.615<br>0.666|
|||**_batch size = 5_**||
||_M100_<br>_M50_<br>_M10_|0.479<br>0.383<br>0.733<br>0.605<br>0.793<br>0.652<br>**_batch size = 10_**|0.431<br>0.669<br>**0.723**|
||_M100_<br>_M50_|0.404<br>0.299<br>0.723<br>0.598|0.352<br>0.661|
||_M10_|**0.804**<br>0.634|0.719|



Table 7: Accuracy of scored-based wrong-overwrong preference on Knowledge Crosswords and NLGraph (shortest path) dataset. The generator is LLAMA3-8B and the evaluator is GPT-4O. The number of answers generated to each question is 10. _b_ = 5 yields the best overall Acc _WoW_ . 

wrongness (e.g. “the sun sometimes rises in the east” and “the sun sometimes perhaps rises in the east”) are hard to differentiate. We experiment on Knowledge Crosswords dataset and evaluate the wrongness margin is defined as the the absolute difference of correctly filled blanks. The results in Table 11 demonstrate that a larger wrongness margin means easier evaluation, with average AccWOW = 0.696, 0.636 and 0.518 for wrongness margin being 3, 2, 1. This trend is particularly evident for highly wrong answer comparison, with 3 _≻_ 2 having an accuracy of 0.533 while 1 _≻_ 0 only has an accuracy of 0.463. 

**No or negative effect of mixing generators is observed.** Experiments in Table 2 demonstrate no effect from mixing generators for _fGPT-4o_[(] _[p]_[)][with consis-] tency checks, _fGPT-4o_[(] _[s]_[)][with] _[ M][10]_[ and oracle, with little] change in average proxy scores discovered. And Table 2 also indicates an negative effect on _fGPT-4o_[(] _[s]_[)][with] _M50_ with average improvement on ∆Acc across datasets drops by 28.9%. This suggests _learning from the model’s own mistakes is perhaps more important for wrong-over-wrong alignment_ . 

**Preference Optimization Approach** To investigate the influence of different preference optimization methods, we also experiment wrong-overwrong alignment with IPO (Azar et al., 2024), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only_images/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only.pdf-0020-09.png)


**----- Start of picture text -----**<br>
IPO SPPO ORPO DPO<br>.105<br>0.10 .076 .089 .087<br>0.05 .038<br>.008 .002<br>0.00<br>0.05 .027 .040<br>.068 .061<br>0.10<br>0.15 .147<br>pwrong Acc ECE<br>**----- End of picture text -----**<br>


Figure 4: Evaluation different of preference optimization methods on less wrong, more correct and better calibration. The number is averaged over six experiment setups ((pairwise comparison with consistency check, score-based with _M10_ , scorebased with _M50_ ) _×_ (Self-Generator, MixGenerator)) on NLGraph dataset. 

20 

ORPO (Hong et al., 2024) SimPO (Meng et al., 2024) and SPPO (Wu et al., 2024) on the NLGraph dataset. Results in Figure 4 demonstrate that preference optimization methods other than DPO are sensitive to noise and could not be employed for wrong-over-wrong alignment without hyperparameter tuning. SimPO collapses on all 6 experiment setups, IPO collapses on 4 of them, and SPPO and ORPO show negative improvement compared to original LLAMA3-8B. This suggests wrong-over-wrong alignment could be noisy and we should adopt stable preference optimization methods like DPO (Xu et al., 2024a; Chowdhury et al., 2024). 

**Alignment on domains without clear rightMedMCQA wrong distinctions** We wonder if the effec- **Method** tiveness of wrong-over-wrong alignment is uniAcc _↑_ ECE _↓_ versal and doesn’t rely on clear right-wrong ORIGINAL 0.566 0.083 distinctions within the task domain. To better **SELF-GENERATOR** prove this, we implement wrong-over-wrong alignment on SciBench (Wang et al., 2024b) SCORE _M50_ **0.565** 0.111 and MedMCQA (Pal et al., 2022) where no SCORE _M10_ 0.550 **0.082** such proxy to evaluate the wrongness of an- **MIX-GENERATOR** swers exists. SciBench is a college-level scientific benchmark emphasizing scientific theorem SCORE _M50_ 0.548 0.083 application and numerical computation, while SCORE _M10_ 0.553 0.098 MedMCQA is a multiple-choice QA dataset in the medical domain. Experiment results in TaTable 8: Generalization datasets ble 8 indicate that wrong-over-wrong alignment is not universally helpful in domains without clear right-wrong distinctions. This suggests we out clear right-wrong distinctions. should investigate more fine-grained separability among wrong answers and design a more sophisticated LLM-as-a-judge prompt. 

|**Method**|**MedMCQA**<br>Acc_↑_<br>ECE_↓_|**SciBench**<br>Acc_↑_<br>ECE_↓_|
|---|---|---|
|ORIGINAL|0.566<br>0.083|0.105<br>0.634|
||**SELF-GENERATOR**||
|SCORE_M50_<br>SCORE_M10_|**0.565**<br>0.111<br>0.095<br>0.607<br>0.550<br>**0.082**<br>0.068<br>0.660||
||**MIX-GENERATOR**||
|SCORE_M50_<br>SCORE_M10_|0.548<br>0.083<br>**0.108**<br>**0.591**<br>0.553<br>0.098<br>0.095<br>0.688||



Table 8: Generalization datasets without distinctive wrong-over-wrong separability. Wrong-overwrong alignment is not helpful for domains without clear right-wrong distinctions. 

**Qualitative examples** We present qualitative examples demonstrating the improvement of wrongover-wrong alignment on Knowledge Crosswords, Bio Generation, COM[2] , NLGraph from Table 14 to Table 24. 

## B EXPERIMENT DETAILS 

**Model Details** We employ 7 LLMs in the experiments, the detailed information of LLMs we used in this paper is as follows: 1) LLAMA3-8B, through the META-LLAMA/METALLAMA-3-8B checkpoint on Huggingface (Wolf et al., 2020); 2) MISTRAL-7B, through the MISTRALAI/MISTRAL-7B-INSTRUCT-V0.3 checkpoint on Huggingface; 3) GEMMA-7B, we directly employ 7 instruction finetuned GEMMA-7Bs from Ivison et al. (2023); 4) GPT-3.5, through the GPT-3.5-TURBO-0125 checkpoint on OpenAI API; 5) GPT-4O, through the GPT-4O-2024-0513 checkpoint on OpenAI API; 6) GEMINI-FLASH through the GEMINI-1.5-FLASH-001 by the Vertex AI API; 7) GEMINI-PRO through the GEMINI-1.5-PRO-001 by the Vertex AI API. 

**Dataset Details** We employ 8 datasets in the experiments, 4 for the main experiments and 4 left in the appendix. The detailed information is: 

- **Knowledge Crosswords (KC)** (Ding et al., 2023): We downsample questions containing exactly three blanks, and we create four answer options by randomly choosing 0, 1, 2, or 3 correctly filled blanks. We sample 625 questions from the official dataset and split them into train, validation, test sets with an 8:1:1 ratio. 

- **NLGraph (NLG)** (Wang et al., 2023a; Zhang et al., 2024) We employ the “shortest path” subset for main experiments denoted as NLGraph (shortest path) and “shortest path”, “maximum flow” and “matching” subsets for experiments in the appendix denoted as NLGraph (full). Correctness proxy for “maximum flow” and “matching” subset is: _p_[(] _[mf)]_ ( _**a** |_ _**q**_ ) = 1 _−[|][f][g][t] f[−] gt[f]_ _**[a]**[|]_ , with _f_ _**a**_ being value given in _**a**_ and _fgt_ being the ground-truth. We sample 380, 350 and 600 questions in “shortest path”, “maximum flow” and “matching” subsets and split them into train, validation, and test sets with an 8:1:1 ratio. 

21 

- **Bio Generation (BG)** We employ the bio generation dataset and retrieved documents provided in FActScore (Min et al., 2023). We sample 625 questions from the official dataset and split them into train, validation, test sets with an 8:1:1 ratio. 

- **COM**[2] (Fang et al., 2024): We employ the “2i” and “3i” question subsets of COM[2] , where we empirically find that Vera could generate more accurate scores. We sample 625 questions from the official dataset and split them into train, validation, test sets with an 8:1:1 ratio. 

- **Hellaswag** (Zellers et al., 2019): Correctness proxy function for HellaSwag is: _p_[(] _[(hs)]_ ( _**a** |_ _**q**_ ) = Vera( _**a** |_ _**q**_ ) _∈_ [0 _,_ 1]. We sample 125 questions from the official validation split and split them into validation, test sets with a 1:1 ratio. 

- **Chess Puzzle** (Lichess Team, 2023): We downsample the beginner-level chess puzzles with ELO ranging from 600 to 1000. We sample 625 questions from the official dataset and split them into train, validation, test sets with an 8:1:1 ratio. 

- **SciBench** (Wang et al., 2024b): We sample 625 questions from the official dataset and split them into train, validation, and test sets with an 8:1:1 ratio. 

- **MedMCQA** (Pal et al., 2022): We sample 500 questions from the official train split as train set. We sample 125 questions from the official validation split into validation, test sets with a 1:1 ratio. 

## **Implementation Details** 

- **Pairwise comparison** . The exact prompt we use for pairwise comparison is in Table 12. 

- **Score-based** The exact prompt we use for score-based eliciting is in Table 13. 

- **Generation of wrong-over-wrong dataset** The pipeline used to generate _DWoW_ is described in Algorithm 1. 

- **QLora finetuning** We employ Unsloth and Transformers libraries for preference optimization. We apply grid search on learning rate (1e-4, 5e-5, 1e-5), learning rate scheduler (cosine, cosine with restart and reduce lr on plateau), weight decay (0, 1e-5, 1e-3) and number of train epochs (1, 3, 5) for main experiments and right-over-wrong alignment experiments. We use random seed = 42 for all of our experiments. 

**Experiment Details** In Table 1, we use batch size = 5 for all score methods due to optimal empirical results. In Table 2, we define FActScore _>_ 0.9 as “correct” on Bio Generation dataset; all the preferences are elicited from GPT-4O. In Table 3 the best wrong-over-wrong alignment setup for Knowledge Crosswords, Bio Generation, COM[2] and NLGraph are score-based with _M50_ on selfgenerated data, pairwise comparison with consistency check on mix-generated data and score-based with _M10_ on self-generated data; Bio Generation dataset is omitted because there is not well-defined ground-truth on this dataset. Correlations between task accuracy, confidence and other wrong-overwrong preference eliciting methods are displayed in Figure 5 and Figure 6. 

22 

|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br><br><br>|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br><br><br>|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br><br><br>|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br><br><br>|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br><br><br>|
|---|---|---|---|---|
|**g**<br>KC|BG COM2 NLG|KC BG COM2 NLG|KC BG COM2 NLG|Overall|
||**MODEL-INDEPENDENT**||||
|HEURISTIC<br>_M100_<br>.491<br>CONSISTENCY _M100_<br>.296|.486 .488<br>.498<br>-<br>.287<br>.360|.519 .504 .473<br>.466<br>.318<br>-<br>.279<br>.336|.547 .471 .473<br>.502<br>.311<br>-<br>.325<br>.313|.489<br>.328|
||**LLAMA3-8B****_as Evaluator_**||||
|LOGITS<br>_M100_<br>.527<br>SCORE<br>_M100_<br>.479|.540 .525<br>.508<br>.386 .385<br>.415|.546 .516 .506<br>.477<br>.496 .376<br>.48<br>.434|.528 .491 .440<br>.546<br>.387 .373 .525<br>.273|.519<br>.400|
||**GPT-3.5****_as Evaluator_**||||
|LOGITS<br>_M100_<br>-<br>SCORE<br>_M100_<br>.212|-<br>-<br>-<br>.352 .287<br>.323|.528 .508 .552<br>.479<br>.289 .327 .458<br>.422|-<br>-<br>-<br>-<br>.132 .273 .269<br>.307|.504<br>.335|
||**GPT-4O****_as Evaluator_**||||
|LOGITS<br>_M100_<br>-<br>PAIRWISE<br>all<br>.605<br>flter<br>.691<br>SCORE<br>_M100_<br>.479<br>_M50_<br>.733<br>_M10_<br>**.793**|-<br>-<br>-<br>.593 .507<br>.551<br>.689 .533<br>.602<br>.385 .450<br>.383<br>.677 .544<br>.605<br>**.795** .534<br>**.652**|-<br>-<br>-<br>-<br>.646 .512 .515<br>.577<br>.712 .536 .558<br>.661<br>.540 .351 .484<br>.408<br>.793<br>.591<br>.617<br>.661<br>**.835 .655** .711<br>**.684**|.472 .526 .478<br>.540<br>.434 .501 .526<br>.537<br>.417 .490 .604<br>.549<br>.481 .353 .467<br>.369<br>.547 .520 .581<br>**.639**<br>.400 .586 .520<br>.578|.526<br>.562<br>.624<br>.393<br>.641<br>**.709**|
||**MISTRAL-7B****_as Evaluator_**||||
|PAIRWISE<br>all<br>.463<br>flter<br>.551<br>SCORE<br>_M100_<br>.351<br>_M50_<br>.641<br>_M10_<br>.683|.471 .462<br>.492<br>.427 .451<br>.505<br>.356 .370<br>.389<br>.538 .505<br>.527<br>.558 .486<br>.558|.447 .482 .524<br>.501<br>.531 .491 .574<br>.538<br>.379 .348 .431<br>.453<br>.652 .512 .584<br>.599<br>.723 .513 .626<br>.565|.453 .503 .471<br>.514<br>.542 .365 .542<br>.560<br>.051 .305 .354<br>.337<br>.102 .514 .632<br>.576<br>.222 .557 .677<br>.565|.482<br>.514<br>.366<br>.549<br>.565|
||**GEMINI-FLASH****_as Evaluator_**||||
|PAIRWISE<br>all<br>.531<br>flter<br>.574<br>SCORE<br>_M100_<br>.482<br>_M50_<br>.642<br>_M10_<br>.746|.505 .500<br>.509<br>.540 .517<br>.520<br>.433 .433<br>.362<br>.584 .510<br>.517<br>.606 .504<br>.551|.519 .473 .538<br>.574<br>.557 .444 .561<br>.629<br>.521 .368 .497<br>.455<br>.661 .497 .630<br>.616<br>.719 .483 **.756**<br>.665|.415 .471 .519<br>.531<br>.410 .373 .556<br>.565<br>.453 .377 .560<br>.429<br>.528 .510 .674<br>.600<br>**.800 .593 .817**<br>.576|.509<br>.529<br>.420<br>.567<br>.599|
||**GEMINI-PRO****_as Evaluator_**||||
|PAIRWISE<br>all<br>.539<br>flter<br>.598<br>SCORE<br>_M100_<br>.494<br>_M50_<br>.653<br>_M10_<br>.742|.538 .493<br>.513<br>.588 .547<br>.564<br>.442 .450<br>.397<br>.612 .551<br>.520<br>.695<br>**.573**<br>.533|.558 .435 .409<br>.554<br>.628 .471 .557<br>.658<br>.555 .361 .483<br>.476<br>.692 .520 .600<br>.631<br>.762 .524 .681<br>.662|.396 .437 .522<br>.530<br>.423 .412 .560<br>.569<br>.434 .363 .460<br>.417<br>.566 .511 .567<br>.560<br>.600<br>.590<br>.732<br>.572|.509<br>.557<br>.428<br>.581<br>.633|



Table 9: Accuracy of wrong-over-wrong preference elicited from MISTRAL-7B, GEMINI-FLASH and GEMINI-PRO, and also the accuracy of heuristic, consistency-based, logits-based and scorebased methods with a margin of _M100_ that are not displayed in the main paper. _fGPT-4o_[(] _[s]_[)][with] _[ M][10]_[ still] achieves the best weighted average Acc _WoW_ = 0.709 across all datasets. 

23 

|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br>Oerall|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br>Oerall|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br>Oerall|**Method**<br>**Marin**<br>LLAMA3-8B<br>GPT-3.5<br>GPT-4O<br>Oerall|
|---|---|---|---|
|**g**<br>NLG<br>CP|NLG<br>CP|NLG<br>CP|v|
|**_Model-independent_**||||
|HEURISTIC<br>_M100_<br>0.510<br>0.506<br>_M50_<br>0.509<br>0.506<br>_M10_<br>0.457<br>0.512<br>CONSISTENCY<br>_M100_<br>0.359<br>0.281<br>_M50_<br>0.544<br>0.478<br>_M10_<br>0.509<br>0.535|0.472<br>0.504<br>0.468<br>0.519<br>0.476<br>0.501<br>0.323<br>0.300<br>0.514<br>0.499<br>0.506<br>0.492|0.535<br>0.481<br>0.547<br>0.464<br>0.486<br>0.475<br>0.269<br>0.301<br>0.432<br>0.498<br>0.302<br>0.468|0.501<br>0.502<br>0.478<br>0.318<br>0.505<br>0.478|
|**_LLaMA3-8B as Evaluator_**||||
|PAIRWISE<br>all<br>0.501<br>0.516<br>flter<br>0.586<br>0.560<br>LOGITS<br>_M100_<br>0.527<br>0.495<br>_M50_<br>0.539<br>0.488<br>_M10_<br>0.562<br>0.512<br>SCORE<br>_M100_<br>0.426<br>0.411<br>_M50_<br>0.547<br>0.524<br>_M10_<br>0.529<br>0.564|0.477<br>0.493<br>0.538<br>0.504<br>0.469<br>0.507<br>0.456<br>0.508<br>0.453<br>0.485<br>0.421<br>0.406<br>0.536<br>0.534<br>0.552<br>0.545|0.551<br>0.504<br>0.500<br>0.510<br>0.574<br>0.525<br>0.625<br>0.532<br>0.656<br>0.525<br>0.412<br>0.372<br>0.593<br>0.529<br>0.636<br>0.511|0.503<br>0.545<br>0.511<br>0.518<br>0.530<br>0.417<br>0.546<br>0.557|
|**_GPT-3.5 as Evaluator_**||||
|PAIRWISE<br>all<br>0.505<br>0.485<br>flter<br>**0.802**<br>0.547<br>LOGITS<br>_M100_<br>-<br>-<br>_M50_<br>-<br>-<br>_M10_<br>-<br>-<br>SCORE<br>_M100_<br>0.368<br>0.311<br>_M50_<br>0.579<br>0.510<br>_M10_<br>0.578<br>0.496|0.514<br>0.469<br>**0.590**<br>**0.585**<br>0.479<br>0.518<br>0.472<br>0.521<br>0.492<br>0.515<br>0.401<br>0.355<br>0.540<br>0.538<br>0.527<br>0.501|0.507<br>0.499<br>**0.882**<br>0.535<br>-<br>-<br>-<br>-<br>-<br>-<br>0.333<br>0.219<br>0.593<br>0.438<br>0.614<br>**0.635**|0.502<br>**0.642**<br>0.489<br>0.484<br>0.498<br>0.359<br>0.553<br>0.553|
|**_GPT-4o as Evaluator_**||||
|PAIRWISE<br>all<br>0.560<br>0.495<br>flter<br>0.586<br>0.519<br>LOGITS<br>_M100_<br>-<br>-<br>_M50_<br>-<br>-<br>_M10_<br>-<br>-<br>SCORE<br>_M100_<br>0.352<br>0.346<br>_M50_<br>0.573<br>0.523<br>_M10_<br>0.551<br>**0.611**|0.554<br>0.512<br>0.569<br>0.520<br>-<br>-<br>-<br>-<br>-<br>-<br>0.344<br>0.402<br>0.538<br>0.513<br>0.535<br>0.490|0.575<br>0.503<br>0.613<br>0.512<br>0.546<br>0.525<br>0.577<br>0.532<br>0.630<br>0.525<br>0.399<br>0.381<br>0.554<br>0.487<br>0.529<br>0.397|0.546<br>0.568<br>0.564<br>0.594<br>0.617<br>0.361<br>0.546<br>0.541|



Table 10: Accuracy of wrong-over-wrong preference on NLGraph (full) and ChessPuzzle dataset. _fGPT-3.5_[(] _[p]_[)][with][consistency][check][achieve][the][best][weighted][average][Acc] _[WoW]_[=][0.642][across][all] datasets. 

**Algorithm 1** _DWoW_ generation pipeline 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only_images/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only.pdf-0024-03.png)


24 

|Margin|3_≻_0|3_≻_1|2_≻_0|3_≻_2|2_≻_1|1_≻_0|
|---|---|---|---|---|---|---|
|||**_logits (self)_**|||||
|_M100_|**0.571**|0.543|0.557|0.500|0.520|0.502|
|_M50_|**0.590**|0.553|0.551|0.514|0.533|0.531|
|_M10_|**0.649**|0.576|0.625|0.540|0.464|0.629|
|||**_score (LLaMA3-8B)_**|||||
|_M100_|**0.574**|0.527|0.512|0.423|0.467|0.459|
|_M50_|**0.749**|0.711|0.658|0.541|0.610|0.587|
|_M10_|0.758|**0.774**|0.714|0.607|0.667|0.645|
|||**_score_**|**_(GPT-3.5)_**||||
|_M100_|**0.308**|0.238|0.233|0.189|0.188|0.236|
|_M50_|**0.463**|0.454|**0.463**|0.399|0.392|0.431|
|_M10_|**0.780**|0.776|0.670|0.617|0.585|0.434|
|||**_score (GPT-4o)_**|||||
|_M100_|**0.753**|0.668|0.575|0.509|0.481|0.339|
|_M50_|**0.917**|0.895|0.812|0.724|0.764|0.508|
|_M10_|**0.958**|0.940|0.829|0.792|0.867|0.529|
|average|**0.696**|0.665|0.607|0.533|0.558|0.463|



Table 11: Preference accuracy for wrong-over-wrong pair with different wrongness margins. _a ≻ b_ means preferring an answer filling in _a_ blanks correctly than an answer filling in _b_ blanks correctly. To have an overview, we also include correct answers which fill in 3 blanks correctly. We experiment on score-based eliciting with 3 LLM evaluators respectively and logits-based with self-evaluation. The results are an average on 3 generators. The largest wrongness margin 3 _≻_ 1 yields the most accurate preference. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only_images/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only.pdf-0025-02.png)


**----- Start of picture text -----**<br>
KC COM²<br>BG NLG<br>0.7<br>0.6<br>0.5<br>0.4 P_all = 0.303<br>P_KC = 0.440<br>P_BG = 0.123<br>0.3 P_COM² = 0.732<br>P_NLG = 0.379<br>0.2 0.4 0.6 0.8<br>task accuracy<br>0.7<br>0.6<br>0.5<br>0.4 P_all = -0.205<br>P_KC = -0.461<br>P_BG = -0.325<br>0.3 P_COM² = -0.723<br>P_NLG = -0.681<br>0.5 0.6 0.7 0.8 0.9<br>task confidence<br>wow<br>Acc<br>wow<br>Acc<br>**----- End of picture text -----**<br>


Figure 5: Correlation between task accuracy, confidence and Acc _WoW_ for pairwise comparison with consistency check. Data points are from all 3 LLMs we used to elicit wong-overwrong preferences. _P_ stands for Pearson correlation coefficient. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only_images/iclr_2025_Varying_Shades_of_Wrong_Aligning_LLMs_with_Wrong_Answers_Only.pdf-0025-04.png)


**----- Start of picture text -----**<br>
KC COM²<br>BG NLG<br>0.8<br>0.7<br>0.6<br>0.5<br>P_all = 0.254<br>0.4 P_KC = 0.491<br>P_BG = 0.505<br>P_COM² = 0.344<br>0.3 P_NLG = 0.730<br>0.2 0.4 0.6 0.8<br>task accuracy<br>0.8<br>0.7<br>0.6<br>0.5<br>P_all = -0.077<br>0.4 P_KC = -0.760<br>P_BG = -0.449<br>P_COM² = -0.374<br>0.3 P_NLG = -0.553<br>0.5 0.6 0.7 0.8 0.9<br>task confidence<br>wow<br>Acc<br>wow<br>Acc<br>**----- End of picture text -----**<br>


Figure 6: Correlation between task accuracy, confidence and Acc _WoW_ for score-based with _M50_ . Data points are from all 3 LLMs we used to elicit wrong-over-wrong preferences. _P_ stands for Pearson correlation coefficient. 

25 

**Prompt:** You are a helpful assistant whose goal is to select the preferred AI model’s output for a given instruction based on their correctness. **Instruction:** 

[question here] **Output 1:** [answer _**a**_ 1 here] **Output 2:** [answer _**a**_ 2 here] Please first briefly describe your reasoning (in less than 100 words), and then conclude with “Preferred output: 1 or 2.” in the last line. Let’s think step-by-step. 

Table 12: Prompt for pairwise comparison wrong-over-wrong preference eliciting. 

**Prompt:** Below is an instruction from a user and five candidate responses. Please grade the correctness of each response on a 5-point scale from 0 to 5, where 0 is the lowest correctness, and 5 is the highest correctness. Only use integers for the ratings. **Instruction:** 

[question here] **Response 1:** [answer _**a**_ 1 here] **Response 2:** [answer _**a**_ 2 here] **Response 3:** [answer _**a**_ 3 here] **Response 4:** [answer _**a**_ 4 here] **Response 5:** [answer _**a**_ 5 here] Please first briefly describe your reasoning (in less than 100 words for each response), followed by a new line in the format of “Score: _<_ rating _>_ ”. Let’s think step-by-step. 

Table 13: Prompt for score-based wrong-over-wrong preference eliciting. 

26 

## **Question:** 

Instruction: Pick the correct answer for each blank that satisfies all the given constraints. Constraints: (Miles ~~G~~ oodman, wroteMusicFor, blank 3); (Julien ~~T~~ emple, directed, blank 1); (blank 2, actedIn, blank 3); (blank 2, actedIn, blank 1); (Jason ~~A~~ lexander, actedIn, blank 3); 

## **Correct answer:** 

blank 1: Earth ~~G~~ irls ~~A~~ re ~~E~~ asy, blank 2: Damon ~~W~~ ayans, blank 3: Blankman 

## **Original answer:** 

Let’s analyze the question step by step: 1. Julien Temple directed blank 1. From the options, we can conclude that blank 1 is either ”Earth ~~G~~ irls ~~A~~ re ~~E~~ asy” or ”Burn ’Em ~~U~~ p ~~B~~ arnes”. 2. Miles Goodman wrote music for blank 3. From the options, we can conclude that blank 3 is either ”Blankman” or ”On Guard ~~(~~ 1997 ~~f~~ ilm)”. 3. blank 2 acted in blank 1 and blank 3. From the options, we can conclude that blank 2 is the same person in both cases. The only options that satisfy this constraint are B and D. 4. Jason Alexander acted in blank 3. From the options, we can conclude that blank 3 is either ”Blankman” or ”On ~~G~~ uard ~~(~~ 1997 ~~f~~ ilm)”. Given the previous conclusion, blank 3 must be ”Blankman”. 5. Putting it all together, we can conclude that the correct answer is B. blank 1: Burn ’Em ~~U~~ p ~~B~~ arnes, blank 2: Justin ~~T~~ heroux, blank 3: Blankman. 

Final answer: B. 

## **Less wrong answer:** 

Let’s analyze the question step by step: 1. Julien ~~T~~ emple, directed, blank 1: Julien Temple directed Earth ~~G~~ irls ~~A~~ re Easy, so blank 1 must be Earth ~~G~~ irls ~~A~~ re ~~E~~ asy. 2. blank 2, actedIn, blank 1: From step 1, blank 1 is Earth ~~G~~ irls ~~A~~ re ~~E~~ asy. So, blank 2 must have acted in Earth ~~G~~ irls ~~A~~ re ~~E~~ asy. Let’s call blank 2 X. 3. blank 2, actedIn, blank 3: From step 2, X acted in Earth ~~G~~ irls ~~A~~ re Easy and also acted in blank 3. 4. Jason ~~A~~ lexander, actedIn, blank 3: Jason Alexander acted in blank 3. 5. blank 2, actedIn, blank 1: From step 2, X acted in Earth ~~G~~ irls ~~A~~ re Easy again (because it’s the same film). 6. Miles ~~G~~ oodman, wroteMusicFor, blank 3: Miles Goodman wrote music for blank 3. 7. Julien ~~T~~ emple, directed, blank 1: We already knew this, but it’s repeated to ensure consistency. 

Now, let’s analyze the options: A. blank 1: Burn ’Em ~~U~~ p ~~B~~ arnes, blank 2: Justin ~~T~~ heroux, blank 3: On ~~G~~ uard ~~(~~ 1997 ~~f~~ ilm): Burn ’Em ~~U~~ p ~~B~~ arnes is not directed by Julien Temple, and Justin Theroux didn’t act in On ~~G~~ uard ~~(~~ 1997 ~~f~~ ilm). Discarded. B. blank 1: Earth ~~G~~ irls ~~A~~ re Easy, blank 2: Damon ~~W~~ ayans, blank 3: Blankman: We already know that blank 1 is Earth ~~G~~ irls ~~A~~ re Easy. Damon Wayans didn’t act in Earth ~~G~~ irls ~~A~~ re ~~E~~ asy, and Blankman isn’t the same film as On ~~G~~ uard ~~(~~ 1997 ~~f~~ ilm). Discarded. C. blank 1: Burn ’Em ~~U~~ p ~~B~~ arnes, blank 2: Justin ~~T~~ heroux, blank 3: Blankman: Burn ’Em ~~U~~ p ~~B~~ arnes is not directed by Julien Temple, and Justin Theroux didn’t act in Blankman. Discarded. D. blank 1: Earth ~~G~~ irls ~~A~~ re ~~E~~ asy, blank 2: Justin ~~T~~ heroux, blank 3: Blankman: This option fits all the constraints: - blank 1 is Earth Girls ~~A~~ re ~~E~~ asy (step 1) - blank 2 is Justin ~~T~~ heroux (step 2) - blank 3 is Blankman (works with steps 3, 4, and 6) 

So, the final answer is: D. blank 1: Earth ~~G~~ irls ~~A~~ re ~~E~~ asy, blank 2: Justin ~~T~~ heroux, blank 3: Blankman 

(Correctly filled blanks 1 _→_ 2) 

## **More correct answer:** 

Let’s break down the constraints step by step: 1. (Miles ~~G~~ oodman, wroteMusicFor, blank 3) This constraint tells us that Miles Goodman wrote music for a movie, and the title of that movie is blank 3. 2. (Julien ~~T~~ emple, directed, blank 1) This constraint tells us that Julien Temple directed a movie, and the title of that movie is blank 1. 3. (blank 2, actedIn, blank 3) This constraint tells us that blank 2 acted in a movie, and the title of that movie is blank 3. 4. (blank 2, actedIn, blank 1) This constraint tells us that blank 2 acted in another movie, and the title of that movie is blank 1. 5. (Jason ~~A~~ lexander, actedIn, blank 3) This constraint tells us that Jason Alexander acted in a movie, and the title of that movie is blank 3. 

Table 14: Examples showing the effect of wrong-over-wrong alignment on Knowledge Crosswords dataset (part 1). 

27 

Now, let’s analyze the options: A. blank 1: Burn ’Em ~~U~~ p ~~B~~ arnes, blank 2: Justin ~~T~~ heroux, blank 3: Blankman Burn ’Em ~~U~~ p ~~B~~ arnes is not a well-known movie, and Blankman is not a movie that was directed by Julien Temple or written music for by Miles Goodman. B. blank 1: Earth ~~G~~ irls ~~A~~ re ~~E~~ asy, blank 2: Justin ~~T~~ heroux, blank 3: On ~~G~~ uard ~~(~~ 1997 ~~f~~ ilm) Earth ~~G~~ irls ~~A~~ re Easy is a real movie, but On ~~G~~ uard ~~(~~ 1997 ~~f~~ ilm) is not a movie that was directed by Julien Temple or written music for by Miles Goodman. Additionally, Justin Theroux did not act in On ~~G~~ uard ~~(~~ 1997 ~~f~~ ilm). C. blank 1: Earth ~~G~~ irls ~~A~~ re ~~E~~ asy, blank 2: Damon ~~W~~ ayans, blank 3: Blankman Damon Wayans is an actor, but he did not act in Earth ~~G~~ irls ~~A~~ re Easy. 

The only option that satisfies all the constraints is: C. blank 1: Earth ~~G~~ irls ~~A~~ re ~~E~~ asy, blank 2: Damon ~~W~~ ayans, blank 3: Earth ~~G~~ irls ~~A~~ re ~~E~~ asy This option fills in the blanks as follows: - Julien Temple directed Earth ~~G~~ irls ~~A~~ re ~~E~~ asy (constraint 2) - Damon Wayans acted in Earth ~~G~~ irls ~~A~~ re Easy (constraint 4) - Damon Wayans acted in Earth ~~G~~ irls ~~A~~ re ~~E~~ asy (constraint 3) - Jason Alexander acted in Earth ~~G~~ irls ~~A~~ re ~~E~~ asy (constraint 5) - Miles Goodman wrote music for Earth ~~G~~ irls ~~A~~ re ~~E~~ asy (constraint 1) 

The movie Earth ~~G~~ irls ~~A~~ re ~~E~~ asy is a 1988 musical comedy film directed by Martin Davidson, and Damon Wayans and Jason Alexander both appeared in the film. Miles Goodman wrote the original songs for the movie. (Correctly filled blanks 1 _→_ 3) 

## **Better calibrated answer:** 

Let’s analyze the question step by step: 1. (Julien ~~T~~ emple, directed, blank 1) - This constraint states that Julien Temple directed a movie, and the title of that movie is blank 1. 2. (blank 2, actedIn, blank 1) - This constraint states that blank 2 acted in the same movie as in step 1. 3. (blank 2, actedIn, blank 3) - This constraint states that blank 2 acted in another movie, and the title of that movie is blank 3. 4. (Miles ~~G~~ oodman, wroteMusicFor, blank 3) - This constraint states that Miles Goodman wrote music for the same movie as in step 3. 5. (Jason ~~A~~ lexander, actedIn, blank 3) - This constraint states that Jason Alexander acted in the same movie as in step 3. 

Now, let’s analyze the options: A. blank 1: Earth ~~G~~ irls ~~A~~ re ~~E~~ asy, blank 2: Damon ~~W~~ ayans, blank 3: Blankman - This option is incorrect because Earth ~~G~~ irls ~~A~~ re ~~E~~ asy is a 1984 film directed by Sidney Lumet, not Julien Temple. Additionally, Justin Theroux was not involved in that film. B. blank 1: Burn ’Em ~~U~~ p ~~B~~ arnes, blank 2: Justin ~~T~~ heroux, blank 3: Blankman - This option is incorrect because Justin Theroux was not involved in the 1984 film Burn ’Em ~~U~~ p ~~B~~ arnes, which is a TV movie. Additionally, Jason Alexander was not involved in that film. C. blank 1: Burn ’Em ~~U~~ p ~~B~~ arnes, blank 2: Justin ~~T~~ heroux, blank 3: On ~~G~~ uard ~~(~~ 1997 ~~f~~ ilm) - This option is incorrect because Jason Alexander was not involved in the 1997 film On ~~G~~ uard ~~,~~ which is a TV movie. Additionally, Miles Goodman was not involved in that film. D. blank 1: Earth ~~G~~ irls ~~A~~ re ~~E~~ asy, blank 2: Justin ~~T~~ heroux, blank 3: Blankman - This option is incorrect because Justin Theroux was not involved in the 1984 film Earth ~~G~~ irls ~~A~~ re Easy, and Jason Alexander was not involved in that film. After analyzing the options, the only remaining possibility is: B. blank 1: Burn ’Em ~~U~~ p ~~B~~ arnes, blank 2: Justin ~~T~~ heroux, blank 3: Blankman This option satisfies all the constraints: - Julien Temple directed the movie Burn ’Em ~~U~~ p ~~B~~ arnes (1990), which is blank 1. - Justin Theroux acted in Burn ’Em ~~U~~ p ~~B~~ arnes, which is blank 2. - Justin Theroux also acted in Blankman (1994), which is blank 3, along with Jason Alexander. - Miles Goodman wrote music for Blankman, which is blank 3. 

Therefore, the final answer is: B. blank 1: Burn ’Em ~~U~~ p ~~B~~ arnes, blank 2: Justin ~~T~~ heroux, blank 3: Blankman (Even the answers are equally wrong, NLL increases 0.129 _→_ 0.185) 

Table 15: Examples showing the effect of wrong-over-wrong alignment on Knowledge Croswords dataset (part 2). 

28 

## **Failed answer:** 

Let’s break down the constraints step by step: 1. (Miles ~~G~~ oodman, wroteMusicFor, blank 3): This constraint tells us that Miles Goodman wrote music for a movie, and the title of that movie is blank 3. From the options, we can see that only options C and D mention Miles Goodman writing music. 2. (Julien ~~T~~ emple, directed, blank 1): This constraint tells us that Julien Temple directed a movie, and the title of that movie is blank 1. From the options, we can see that only options B and C mention Julien Temple directing a movie. The title of that movie is either Burn ’Em ~~U~~ p ~~B~~ arnes or Earth ~~G~~ irls ~~A~~ re ~~E~~ asy. 3. (blank 2, actedIn, blank 3): This constraint tells us that blank 2 acted in a movie, and the title of that movie is blank 3. From the previous step, we know that blank 1 is either Burn ’Em ~~U~~ p ~~B~~ arnes or Earth ~~G~~ irls ~~A~~ re ~~E~~ asy. Since blank 2 is the same in both options B and C, we can conclude that blank 2 acted in a movie with the title blank 3, which is either Burn ’Em ~~U~~ p ~~B~~ arnes or Earth ~~G~~ irls ~~A~~ re ~~E~~ asy. 4. (blank 2, actedIn, blank 1): This constraint is redundant, as it’s already known that blank 2 acted in a movie with the title blank 1, which is either Burn ’Em ~~U~~ p ~~B~~ arnes or Earth ~~G~~ irls ~~A~~ re ~~E~~ asy. 5. (Jason ~~A~~ lexander, actedIn, blank 3): This constraint tells us that Jason Alexander acted in a movie, and the title of that movie is blank 3. From the previous steps, we know that blank 3 is either Burn ’Em ~~U~~ p ~~B~~ arnes or Earth ~~G~~ irls ~~A~~ re ~~E~~ asy. Since Jason Alexander acted in one of those movies, we can conclude that Jason Alexander acted in a movie with the title blank 3. Now, let’s analyze the remaining constraints: - Justin Theroux needs to be blank 2, as it’s mentioned in constraints 3 and 4. - Blank 3 needs to be either Burn ’Em ~~U~~ p ~~B~~ arnes or Earth ~~G~~ irls ~~A~~ re Easy, as it’s mentioned in constraints 1, 3, and 5. - Blank 1 needs to be either Burn ’Em ~~U~~ p ~~B~~ arnes or Earth ~~G~~ irls ~~A~~ re ~~E~~ asy, as it’s mentioned in constraints 2 and 4. The only option that satisfies all the constraints is: B. blank 1: Burn ’Em ~~U~~ p ~~B~~ arnes, blank 2: Justin ~~T~~ heroux, blank 3: Earth ~~G~~ irls ~~A~~ re ~~E~~ asy This option satisfies all the constraints: - Miles Goodman wrote music for Earth ~~G~~ irls ~~A~~ re ~~E~~ asy. - Julien Temple directed Burn ’Em ~~U~~ p ~~B~~ arnes. - Justin Theroux acted in both Burn ’Em ~~U~~ p ~~B~~ arnes and Earth ~~G~~ irls ~~A~~ re Easy. - Jason Alexander acted in Earth ~~G~~ irls ~~A~~ re ~~E~~ asy. (Correctly filled blanks 1 _→_ 0) 

Table 16: Examples showing the effect of wrong-over-wrong alignment on Knowledge Crosswords dataset (part 3). 

29 

## **Question:** 

Tell me a bio of Luiz In´acio Lula da Silva. 

## **Original answer:** 

Luiz In´acio Lula da Silva, commonly known as Lula, is a Brazilian politician who served as the 35th President of Brazil from 2003 to 2011. He is one of the most popular and influential politicians in Brazil’s recent history. Early Life: Lula was born on October 27, 1945, in Garanhuns, a small town in the northeastern state of Pernambuco, Brazil. His father was a coffee farmer and his mother was a seamstress. He was the fourth of seven children, and his family struggled financially throughout his childhood. Lula grew up in a poor and rural area, where he learned the value of hard work and the importance of education. He dropped out of school at the age of 12 to work in a textile factory, where he eventually became a union leader. This early experience in labor organizing would shape his future political career. Political Career: Lula’s political career began in the late 1970s, when he became involved with the Brazilian Workers’ Party (PT), a left-wing party founded in the 1980s. He quickly rose through the ranks of the party, becoming its presidential candidate in 1989. Although Lula lost the election to Fernando Collor de Mello, he continued to build his political base and gained popularity for his fight against corruption and social inequality. In 1994, he was accused of corruption and money laundering, which led to a conviction and a four-year prison sentence. However, in 1998, the Supreme Court overturned Lula’s conviction, citing lack of evidence. This cleared the way for him to run for president again in 2002. He won the election in a landslide, receiving nearly 65Presidency: During his presidency, Lula implemented a series of reforms aimed at reducing poverty and inequality. He increased social spending, increased the minimum wage, and expanded access to education and healthcare. He also launched a number of economic initiatives, including a debt reduction plan and an infrastructure development program. Lula’s presidency was marked by strong economic growth, which lifted millions of Brazilians out of poverty. He also gained international attention for his leadership on issues such as climate change and social justice. Presidential Terms: Lula served two consecutive terms as president, from 2003 to 2011. During his first term, he focused on addressing poverty and inequality, and implemented a range of social welfare programs. During his second term, Lula emphasized economic development and infrastructure projects. He invested heavily in public transportation, energy, and telecommunications, and launched a series of initiatives to promote sustainable development. Investigation and Charges: After leaving office in 2011, Lula was investigated for corruption and money laundering. He was accused of receiving bribes from a construction company, OAS, in exchange for favors and contracts. He denied the allegations, but was arrested and charged in 2018. In 2019, Lula was convicted and sentenced to nine years and six months in prison. He maintained his innocence and appealed the conviction, questioning the fairness of the trial. Return to Politics: Despite his imprisonment, Lula remained a beloved figure in Brazil, and his popularity continued to grow. In 2021, the Supreme Court ordered his release from prison, citing the unfairness of the trial. Lula was subsequently reinstated as a senator and began to reorganize the Workers’ Party. In 2022, Lula announced his candidacy for the 2022 Brazilian presidential election, and won the election in a landslide. He is currently serving as the 39th President of Brazil, having taken office on January 1, 2023. Legacy: Luiz In´acio Lula da Silva’s legacy is complex and multifaceted. On the one hand, he is credited with reducing poverty and inequality, and promoting social justice and economic development in Brazil. On the other hand, his presidency was marked by corruption scandals, and his own conviction and imprisonment have raised questions about his integrity and accountability. Despite these controversies, Lula remains a towering figure in Brazilian politics, and his commitment to social justice and poverty reduction has inspired generations of Brazilians. His political career is a testament to the power of grassroots organizing and the importance of fighting for social justice, even in the face of significant challenges and setbacks. 

Table 17: Examples showing the effect of wrong-over-wrong alignment on Bio Generation dataset (part 1). 

30 

## **Less wrong / More correct / Better calibrated answer:** 

Luiz In´acio Lula da Silva, commonly known as Lula, is a Brazilian politician and former trade unionist who served as the 34th President of Brazil from 2003 to 2011. He was born on October 27, 1945, in Garanhuns, a small town in the state of Pernambuco, Brazil. Lula’s early life was marked by poverty and hardship. His parents were both poor and separated when Lula was young, and he was forced to work at a young age to support his family. He eventually moved to S˜ao Paulo and became involved in trade union activism, eventually becoming one of the key figures in the Brazilian Workers’ Central Union (CUT). Lula’s political career began in the 1980s, when he was elected as the leader of the Metalworkers’ Union and became a prominent figure in the Brazilian labor movement. He was one of the founders of the Workers’ Party (PT) in 1980, and he quickly became the party’s most prominent figure. In 2002, Lula became the first worker and the first non-PSDB (Brazilian Social Democracy Party) candidate to run for President of Brazil. Despite initial skepticism about his candidacy, he won the election in a landslide, receiving over 61During his two terms in office, Lula implemented a number of significant reforms and policies. He implemented a series of social and economic programs aimed at reducing poverty and income inequality, including the Bolsa Fam´ılia program, which provided financial assistance to low-income families, and the Fome Zero program, which aimed to eradicate hunger and poverty. Lula also played a key role in shaping Brazil’s foreign policy, particularly in the areas of trade and international relations. He was one of the key figures in the formation of the BRICS (Brazil, Russia, India, China, and South Africa) alliance, and he played a significant role in shaping the policy of the G20 and the United Nations. After leaving office in 2011, Lula remained a prominent figure in Brazilian politics and continued to play a significant role in shaping his party’s policies and candidates. In 2018, he was elected as the Chair of the PT, and he became the leading figure in the country’s political opposition, advocating for changes to the country’s political system and advocating for the nomination of his chosen successor, Fernando Haddad, in the 2018 presidential election. In 2018, Lula’s candidacy in the presidential election was barred due to a corruption conviction, and he was replaced as the PT’s presidential candidate by Fernando Haddad. However, Haddad ultimately lost the election to Jair Bolsonaro. Lula’s legacy in Brazil is complex and controversial. On one hand, he is widely credited with improving the lives of millions of Brazilians, particularly the poor and the working class, through his social and economic policies. On the other hand, he was also accused of corruption and was ultimately convicted of accepting bribes and ordered to serve a prison sentence. He is currently appealing his conviction and is being held in a federal prison in Brazil. Despite his current legal situation, Lula’s political career and legacy continue to be subject to significant debate and discussion in Brazil and around the world. His presidency marked a significant shift in the country’s political landscape, and his policies and legacy continue to have a significant impact on Brazilian politics and society. (FActScore improves 0.836 _→_ 0.925, NLL drops 0.512 _→_ 0.416) 

Table 18: Examples showing the effect of wrong-over-wrong alignment on Bio Generation dataset (part 2). 

31 

## **Failed answer:** 

Luiz In´acio Lula da Silva, commonly known as Lula, is a Brazilian politician and former trade unionist who served as the 34th President of Brazil from 2003 to 2011 and again from 2016 to 2018. Early Life and Career: Lula was born on October 11, 1945, in Garanhuns, Pernambuco, Brazil, to Armando da Silva and Hermilda da Silva. His early life was marked by poverty and hardship, with his family struggling to make ends meet. Lula’s mother died of complications related to a gas explosion when he was just 10 years old, and he was subsequently raised by his aunt. Lula’s early career was marked by his involvement in the trade union movement. He became involved with the Metal Workers Union, where he quickly rose through the ranks and became one of the most prominent union leaders in Brazil. During this time, he became known for his charismatic leadership style, his ability to connect with ordinary people, and his advocacy for workers’ rights. Political Career: In 1980, Lula co-founded the Workers’ Party (PT), a left-wing political party that sought to represent the interests of working-class Brazilians. He became the party’s first president and held the position until 2002. In 2002, Lula became the first trade unionist to run for and win the presidency of Brazil, defeating the incumbent Fernando Henrique Cardoso. He was re-elected in 2006 and served two terms as president until 2011. During his first tenure as president, Lula implemented a range of policies aimed at reducing poverty and income inequality in Brazil. He increased access to education and healthcare, and implemented policies to stimulate economic growth and job creation. In 2010, Lula was succeeded by his former minister and prot´eg´e, Dilma Rousseff, who became the first female president of Brazil. However, Rousseff’s presidency was later marred by corruption scandals and political instability, culminating in her impeachment and removal from office in 2016. Lula returned to politics and ran again for the presidency in 2014, but was not eligible to run due to a constitutional ban on former presidents from holding the office again. He subsequently threw his support behind Rousseff’s re-election bid, but she was ultimately impeached and removed from office. In 2018, Lula ran for the presidency again and won the election, becoming the first and only person to be re-elected as President of Brazil. However, his presidency was short-lived, as he was impeached and removed from office in August 2016, citing corruption allegations and constitutional violations. Lula was subsequently replaced by his vice president, Michel Temer. Personal Life: Lula has been married to Maria de Deus In´acio da Silva (known as Marinha) since 1970, and they have four sons together. In 2018, Lula was sentenced to 12 years and 1 month in prison for corruption and money laundering, stemming from the ”Operation Car Wash” corruption scandal. He has since remained in jail, and his legacy and political future remain uncertain. Legacy: Lula’s presidency was marked by significant economic and social changes in Brazil, including a reduction in poverty and income inequality, as well as an unprecedented level of social mobility. He is widely regarded as one of the most important figures in Brazil’s recent history, and his legacy continues to shape Brazilian politics and society. Despite his political setbacks and legal troubles, Lula remains a powerful and influential figure in Brazilian politics, and his name continues to be mentioned as a potential candidate for future elections. (FActScore drops 0.836 _→_ 0.770) 

Table 19: Examples showing the effect of wrong-over-wrong alignment on Bio Generation dataset (part 3). 

32 

## **Question:** 

Instruction: As an expert in commonsense reasoning, your task is to provide a concise response to a question based on the given context. The question focuses on studying the causes, effects, or attributes of personas related to the given context. Context: Noah had been planning a surprise visit for Everly. As part of the surprise, Noah buys a gift for Everly. When Everly arrives at the airport, Noah picks Everly up. Question: What event or state is both what Everly wants to do after Noah buys a gift for Everly and also what Everly wants to do after Noah picks Everly up at the airport? 

## **Correct answer:** 

Ranking by commonsense plausibility: Everly says thank you _≻_ Everly is shocked _≻_ Everly go out to eat _≻_ Everly rest 

(The commonsense plausibility is given by Vera.) 

## **Original answer:** 

Here’s the step-by-step analysis: 1. The question asks about an event or state that is common to both scenarios: after Noah buys a gift for Everly and after Noah picks Everly up at the airport. 2. The context suggests that Noah is planning a surprise visit for Everly, and buying a gift is part of the surprise. 3. When Noah buys a gift, it implies that Everly didn’t expect the gift, and therefore, will likely be surprised when she receives it. 4. When Noah picks Everly up at the airport, it’s likely that Everly didn’t expect the pick-up, and therefore, will also be surprised. 5. The common event or state that arises from both scenarios is Everly’s surprise. Therefore, the correct answer is B. Everly is shocked. 

## **Less wrong answer:** 

Let’s break down the question step by step: 1. The question asks about an event or state that Everly wants to do after Noah buys a gift for her and also after Noah picks her up at the airport. 2. We know that Everly has been anticipating a surprise visit from Noah, which suggests that she’s looking forward to seeing Noah. 3. When Noah buys a gift for Everly, it’s likely that she wants to see or meet Noah to receive the gift, which means she wants to be with Noah. 4. Similarly, when Noah picks Everly up at the airport, she likely wants to be with Noah, as she was expecting a visit from him. 5. Therefore, the event or state that Everly wants to do after Noah buys a gift for her and after Noah picks her up at the airport is likely to be: **Being with Noah** Since options B-D do not seem related to this event, the correct answer is not present in the options. If I had to choose, I would say that the correct answer is not present in the options (**N/A**). 

(Instead of giving a wrong answer, LLM choose to abstain.) 

## **More correct answer:** 

Let’s break down the question step by step: 1. The question asks about an event or state that is common to both situations: after Noah buys a gift for Everly and after Noah picks Everly up at the airport. 2. In both situations, Everly is receiving a service or action from Noah (a gift or transportation). 3. Given the context, it’s likely that Everly wants to reciprocate or respond positively to Noah’s actions. 4. Among the options, only one relates to Everly’s response to Noah’s actions: A. Everly says thank you. 5. Therefore, the most likely answer is A. Everly says thank you, as it is a common outcome for Everly after receiving a gift or being picked up by Noah. Final answer: A. Everly says thank you. (LLM gives a right answer.) 

Table 20: Examples showing the effect of wrong-over-wrong alignment on COM[2] dataset (part 1). 

33 

## **Better calibrated answer:** 

Let’s break down the question step by step: 1. The question asks about an event or state that is common to two scenarios: after Noah buys a gift for Everly and after Noah picks Everly up at the airport. 2. In both scenarios, Everly is expected to be pleased or surprised, given the context of the surprise visit and gift. 3. The correct answer must capture this idea of Everly’s positive reaction. 4. Options A, B, and D are not directly related to Everly’s actions or reactions. 5. Only option C, ”Everly rests”, is too vague to be relevant to the context. Final answer: None of the options fit the description. The correct answer is not among the provided options. (Even the answers are equally wrong, NLL increases 0.412 _→_ 0.509) 

**Failed answer:** Let’s break down the question step by step: 1. The question asks about an event or state that Everly wants to do after two specific events: Noah buying a gift and Noah picking Everly up at the airport. 2. The question stem ”What event or state...” suggests that we’re looking for a common action or experience that Everly desires in both scenarios. 3. From the context, we know that Noah is planning a surprise visit and has bought a gift for Everly. This suggests that Everly is likely expecting a pleasant experience or a reunion with Noah. 4. After Noah picks Everly up at the airport, it’s natural to infer that Everly wants to reunite with or see Noah, which is a pleasing experience. 5. Similarly, after Noah buys a gift for Everly, it’s likely that Everly wants to see or have the gift, which is also a pleasing experience. Given this analysis, the correct answer is: C. Everly visits/see Noah. In both scenarios, Everly wants to experience the pleasure of being with or seeing Noah, which is the most likely common action or state that Everly desires. 

(LLM gives an answer that doesn’t appear in options) 

Table 21: Examples showing the effect of wrong-over-wrong alignment on COM[2] dataset (part 2). 

34 

## **Question:** 

In an undirected graph, the nodes are numbered from 0 to 5, and the edges are: an edge between node 0 and node 2 with weight 2, an edge between node 0 and node 5 with weight 4, an edge between node 0 and node 3 with weight 1, an edge between node 0 and node 1 with weight 3, an edge between node 1 and node 2 with weight 4, an edge between node 1 and node 5 with weight 1, an edge between node 1 and node 4 with weight 4, an edge between node 2 and node 4 with weight 3, an edge between node 2 and node 3 with weight 1, an edge between node 3 and node 5 with weight 2, an edge between node 3 and node 4 with weight 3, an edge between node 4 and node 5 with weight 2. Q: Give the shortest path from node 0 to node 4. Please also give the total weight of the shortest path. 

## **Correct answer:** 

4 [0 _→_ 3 _→_ 4] 

## **Original answer:** 

Here’s the step-by-step analysis: 1. Create a graph with nodes from 0 to 5 and the given edges with their corresponding weights. 2. Choose node 0 as the starting node and node 4 as the target node. 3. Run Dijkstra’s algorithm to find the shortest path from node 0 to node 4. Here are the steps of Dijkstra’s algorithm: 

- Initialize a distance array with all distances set to infinity, except for node 0 which is set to 0. The distance array represents the shortest distance from node 0 to each node. 

- Create a priority queue and add node 0 with a distance of 0. The priority queue is used to select the node with the shortest distance. 

- While the priority queue is not empty, repeat the following steps: 1. Extract the node with the shortest distance from the priority queue. 2. For each neighbor of the extracted node, calculate the tentative distance as the distance to the extracted node plus the edge weight between them. 3. If the tentative distance is less than the current distance to the neighbor, update the distance array and add the neighbor to the priority queue. 

The shortest path from node 0 to node 4 is the path with the minimum distance from node 0 to node 4. The shortest path from node 0 to node 4 is: 0 _→_ 2 _→_ 3 _→_ 4 The total weight of the shortest path is: 2 + 1 + 3 = 6. Therefore, the final answer is: Shortest path: 0 _→_ 2 _→_ 3 _→_ 4, Total weight: 6. 

## **Less wrong answer:** 

Here’s the step-by-step analysis: 

1. Start at node 0. 

2. Consider the edges from node 0: (0, 2) with weight 2, (0, 3) with weight 1, (0, 1) with weight 3, (0, 5) with weight 4. We’ll choose the edge with the smallest weight, which is (0, 3) with weight 1. 

3. Move to node 3. 

4. Consider the edges from node 3: (3, 4) with weight 3, (3, 5) with weight 2, (3, 2) with weight 1. We’ll choose the edge with the smallest weight, which is (3, 2) with weight 1. 

5. Move to node 2. 

6. Consider the edges from node 2: (2, 4) with weight 3. We’ll choose this edge, since it’s the only option. 

7. Move to node 4. 

The shortest path is: 0 _→_ 3 _→_ 2 _→_ 4. The total weight of the shortest path is: 1 + 1 + 3 = 5. Therefore, the answer is: The shortest path from node 0 to node 4 is 0 _→_ 3 _→_ 2 _→_ 4, with a total weight of 5. 

(The path found is shorter: 6 _→_ 5) 

Table 22: Examples showing the effect of wrong-over-wrong alignment on NLGraph dataset (part 1). 

35 

**More correct answer:** Let’s analyze the question step by step: 

1. We are given an undirected graph with nodes numbered from 0 to 5, and edges with weights between these nodes. 

2. We want to find the shortest path from node 0 to node 4. 

3. We can use a simple algorithm to find the shortest path: 

   - Create a list of nodes to visit (initially, only node 0 is in the list). 

   - For each node in the list, check if it’s a neighbor of node 0 (i.e., if an edge exists). 

   - If it’s a neighbor, check the weight of the edge and add the neighbor to the list. 

   - Mark the neighbor as visited. 

   - Repeat until node 4 is reached. 

   - The path from node 0 to node 4 is the sequence of nodes in the list, and the total weight is the sum of the edge weights along this path. 

Applying this algorithm:Node 0 _→_ Node 3 (edge weight 1) _→_ Node 4 (edge weight 3). The shortest path is: Node 0 _→_ Node 3 _→_ Node 4. The total weight is: 1 (from Node 0 to Node 3) + 3 (from Node 3 to Node 4) = 4. Final answer: The shortest path from node 0 to node 4 is Node 0 _→_ Node 3 _→_ Node 4, with a total weight of 4. (Find the shortest path) 

## **Better calibrated answer:** 

Here’s the step-by-step analysis: 

1. Create a graph from the given information. The graph looks like this: 

2. Initialize a distance array with all distances set to infinity, except for the source node (node 0) which is set to 0: 

dist[0] = 0 dist[1] = _∞_ dist[2] = _∞_ dist[3] = _∞_ dist[4] = _∞_ dist[5] = _∞_ 

3. Relax all edges in the graph. This involves updating the distance array by considering all edges: 

dist[1] = 3 (0 _→_ 1 with weight 3) dist[2] = 2 (0 _→_ 2 with weight 2) dist[3] = 1 (0 _→_ 3 with weight 1) dist[4] = _∞_ dist[5] = 4 (0 _→_ 5 with weight 4) dist[2] = 3 (1 _→_ 2 with weight 4 _,_ but previously updated to 2 _,_ so no change) dist[3] = 3 (1 _→_ 3 with weight 4 _,_ but previously updated to 1 _,_ so no change) dist[3] = 2 (2 _→_ 3 with weight 1) dist[4] = _∞_ dist[5] = 5 (1 _→_ 5 with weight 1 _,_ then 4, so no change) dist[4] = 5 (2 _→_ 4 with weight 3) 

4. The minimum distance from node 0 to node 4 is 5. The shortest path is 0 _→_ 2 _→_ 3 _→_ 4. 

Table 23: Examples showing the effect of wrong-over-wrong alignment on NLGraph dataset (part 2). 

36 

The total weight of the shortest path is:0 _→_ 2 (weight 2) _→_ 2 _→_ 3 (weight 1) _→_ 3 _→_ 4 (weight 3). Total weight: 2 + 1 + 3 = 6. 

(Even the answers are equally wrong, NLL increases 0.255 _→_ 0.373) 

## **Failed answer:** 

Let’s break down the question step by step: 

1. Identify the nodes connected to node 0: 1, 2, 3, 5 

2. Weigh the edges connected to node 0: 

   - Edge to node 1: weight 3 

   - Edge to node 2: weight 2 

   - Edge to node 3: weight 1 

   - Edge to node 5: weight 4 

3. Identify the nodes connected to node 1: 2, 4, 5 

   - Edge to node 2: weight 4 

   - Edge to node 4: weight 4 

   - Edge to node 5: weight 1 

4. The shortest path from node 0 to node 4 is through node 3:Node 0 _→_ Node 3 _→_ Node 4 (weights 1 _→_ 3) 

5. The total weight of the shortest path is:0 _→_ 3 (weight 1) _→_ 3 _→_ 4 (weight 3) = 4 

The shortest path from node 0 to node 4 is Node 0 _→_ Node 3 _→_ Node 4, with a total weight of 4. 

(Fail to calculate total weight correctly) 

Table 24: Examples showing the effect of wrong-over-wrong alignment on NLGraph dataset (part 3). 

37 

