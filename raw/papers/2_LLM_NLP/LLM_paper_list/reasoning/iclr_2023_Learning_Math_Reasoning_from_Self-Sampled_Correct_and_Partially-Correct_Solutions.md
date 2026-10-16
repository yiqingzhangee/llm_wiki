Published as a conference paper at ICLR 2023 

# LEARNING MATH REASONING FROM SELF-SAMPLED CORRECT AND PARTIALLY-CORRECT SOLUTIONS 

**Ansong Ni**[1] _[∗]_ **Jeevana Priya Inala**[2] **Chenglong Wang**[2] **Oleksandr Polozov**[3] _[†]_ **Christopher Meek**[4] _[‡]_ **Dragomir Radev**[1] **Jianfeng Gao**[2] 1Yale University, 2Microsoft Research, 3Google, 4University of Washington ansong.ni@yale.edu, _{_ jinala, chenwang _}_ @microsoft.com 

## ABSTRACT 

Pretrained language models have shown superior performance on many natural language processing tasks, yet they still struggle at multi-step formal reasoning tasks like grade school math problems. One key challenge of finetuning them to solve such math reasoning problems is that many existing datasets only contain one reference solution for each problem, despite the fact that there are often alternative solutions resembling different reasoning paths to the final answer. This way, the finetuned models are biased towards the limited reference solutions, which limits their generalization to unseen examples. To mitigate this issue, we propose to let the model perform sampling during training and learn from both selfsampled fully-correct solutions, which yield the correct answer upon execution, and partially-correct solutions, whose intermediate state matches an intermediate state of a known correct solution. We show that our use of self-sampled correct and partially-correct solutions can benefit learning and help guide the sampling process, leading to more efficient exploration of the solution space. Additionally, we explore various training objectives to support learning from multiple solutions per example and find they greatly affect the performance. Experiments on two math reasoning datasets show the effectiveness of our method compared to learning from a single reference solution with MLE, where we improve PASS@100 from 35.5% to 44.5% for GSM8K, and 27.6% to 36.2% PASS@80 for MathQA. Such improvements are also consistent across different model sizes. Our code is available at https://github.com/microsoft/TraceCodegen. 

## 1 INTRODUCTION 

Recent progress on pretrained language models shows that they are able to achieve human-level performance on various natural language processing tasks with finetuning(Devlin et al., 2019; Brown et al., 2020; Raffel et al., 2020). However, such models still lack the ability to perform multi-step math reasoning even for problems that are intended for grade-school students (Cobbe et al., 2021). Current methods for solving math problems typically rely on generating solutions (a sequence of computation steps) and executing them to obtain the final answer (Cobbe et al., 2021; Austin et al., 2021; Chen et al., 2021a; Chowdhery et al., 2022), as directly generating the final answer would require computational abilities that even the largest models do not possess (Brown et al., 2020; Chowdhery et al., 2022). 

When finetuning such models on math reasoning, existing methods often rely on the MLE objective that aims to maximize the log-likelihood of the reference solution for each natural language input. However, in addition to the reference solution, there are often multiple correct solutions for each question, resembling alternative reasoning paths to the final answer. However, those alternative solutions are unseen during training, and this results in model overfitting: the model becomes overly confident in its predictions because it sees the same solution over multiple epochs of training (Bunel et al., 2018; Austin et al., 2021; Cobbe et al., 2021). This leads to poor generalization on unseen 

> _∗_ Majority of the work done during an internship at Microsoft Research. _†_ Work started while at Microsoft Research, paper contribution limited to proof-reading. _[‡]_ Work initiated while at Microsoft. 

1 

Published as a conference paper at ICLR 2023 

inputs and is reflected by the low PASS@ _k_ performance, where the model is unable to predict the right answer even when allowed multiple attempts per question. 

To mitigate this issue, we propose learning from self-sampled solutions. Concretely, during training time, the model samples alternative solutions, and keeps track of all solutions that are semantically correct with respect to the gold execution result, and learns from all of these correct solutions as opposed to only from the reference. To further improve the effectiveness of learning from selfsampled solutions, we allow the model to learn from partially-correct solutions, whose intermediate states are consistent with intermediate states of known correct solutions. This new technique allows the model to maximally utilize the self-sampling and more efficiently explore the solution space. We also study various common loss functions for learning from multiple targets for a single natural language input, including augmented-MLE, Maximize Marginal Likelihood (MML) and _β_ -smoothed MML (Guu et al., 2017) and find that their different gradient equations greatly affect the learning capabilities of the model. 

We perform experiments on two math reasoning tasks, namely MathQA-Python (Austin et al., 2021) and Grade-School-Math (GSM) (Cobbe et al., 2021), and finetune GPT-Neo models (Black et al., 2021) to generate Python program as solutions from the problem description in natural language. Results show that learning from self-sampled solutions can improve the PASS@100 from 35.5% to 44.5% for GSM, and 27.6% to 36.2% for PASS@80 on a filtered version of MathQA-Python.[1] Moreover, we find that learning from partially-correct solutions generally improves performance over learning from just fully-correct solutions ( _e.g.,_ +3.0% PASS@100 for GSM8K) as it guides the sampling process, discovering more alternative solutions for learning. Such performance boosts from our proposed methods are also consistent for different model sizes. Ablation on different loss functions shows that MLE-Aug loss is the most effective in learning from multiple targets and yields the most improvements over MLE loss. 

## 2 OVERVIEW 

**Problem formulation.** We consider the task of generating solutions from math problem descriptions in natural language (NL). Given an NL input _x ∈X_ and the executor _E_ : _Y →Z_ , the goal is to generate a solution _y ∈Y_ that executes to the expected answer _z[∗] ∈Z_ , _i.e., E_ ( _y_ ) = _z[∗]_ . 

**Standard approach and its limitation.** The standard approach is to assume that we have a dataset of paired NL input _x_ and reference solution _y[∗]_ . Most datasets typically only provide one reference solution for a particular NL input. Then, a parameterized model _Pθ_ is learned with the _Maximum Likelihood Estimation_ (MLE) objective from the NL-Solution pair ( _x, y[∗]_ ) as: 

_L_ MLE( _x, y[∗] , Pθ_ ) = _−_ log _Pθ_ ( _y[∗] |x_ ) (1) 

The builtin assumption of using Eq. 1 for learning is that only the reference solution _y[∗]_ is correct. However, this assumption is clearly untrue for the math reasoning problem as typically multiple reasoning paths can achieve the correct final result. With only one reference solution as target for learning, Eq. 1 would encourage the model to put all probability mass on _y[∗]_ , which could easily lead to _overfitting_ (Bunel et al., 2018; Austin et al., 2021; Cobbe et al., 2021). 

**Overview of our approach.** While manually collecting additional reference solutions for each specification is a laborious process (Austin et al., 2021; Cobbe et al., 2021; Schuster et al., 2021), in our work, we explore an alternate approach: where the model self-samples additional correct (or partially-correct) solutions and learns from them during training. Fig. 1 shows an example: for the question _x_ , our model was able to self-sample an alternative solution _y_ ˆ that is different from the reference solution _y[∗]_ provided in the dataset. Looking at the intermediate states shown on the right, ˆ we can see that both these solutions execute to produce the sample desired output, _i.e., z_ = _z[∗]_ , as noted with solid red boxes. Taking this one step further, our approach can also identify partiallycorrect solutions from its samples. For example, on the bottom left, we show a sampled solution _y_ ˆ _[′]_ that is incorrect only because of an error in its last two steps. But we identify a prefix _y_ ˆ _≤[′]_ 5[of it as] partially-correct because the intermediate state _s_ ˆ _[′]_ 5[for][this][prefix][matches][the][intermediate][state] _[s][∗]_ 5 of a known correct solution _y[∗]_ (noted as dashed red boxes) and yet syntactically different from _y[∗]_ . Based on these observations and intuitions, we introduce our approach in the following sections. 

> 1We choose different _k_ for evaluating PASS@ _k_ to be consistent with previous work. 

2 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0003-01.png)


**----- Start of picture text -----**<br>
A goods train runs at a speed of 72kmph and crosses a 250M long platform<br>in 26 seconds. What is the length of the goods train? 𝑥 Algorithm 1  Training Update<br>n0 = 72 𝑦 [∗] {72} Input:<br>n1 = 250 {72, 250} Parameterized model  Pθ ( y|x );<br>n2 = 26 {72, 250, 26} Executor  E :  Y →Z ;<br>t0 = n0 * 0.2778 {72, 250, 26, 20.0}<br>t1 = n1 / t0t2 = n2 - t1 {72, 250, 26, 20.0, {72, 250, 26, 20.0, 12.5, 12.5} 13.5}𝑠"∗ A training exampleBuffer  B  for this input ( x, y x [∗] , z [∗] );<br>answer = t0 * t2 output = 270 𝑧 [∗] 1: if  |B|  = 0  then<br>Self-Sampled Solutions 2: B ←B  +  {y [∗] }  /* initialize buffer */<br>3: end if<br>n0 = 72 #𝑦 {72}<br>n1 = 250n2 = 26 {72, {72, 250, 250} 26} Correct 4:5: Y for [ˆ] ←y ˆ  in SampleSolutionsY [ˆ] do ( x, Pθ, B )<br>t0 = n0 * 0.2778t1 = n2 * t0 {72, 250, 26, {72, 250, 26, 20.0, 20.0} 520} 6: z ˆ ←E (ˆ y ) /* execute solution */<br>answer = t1 - n1 output = 270 ̂𝑧 7: if  isCorrect (ˆ z, z [∗] )  then<br>8: if not  isDuplicate (ˆ y, B )  then<br>n0 = 72 #𝑦′ {72} Partially- 9: B ←B  + ˆ y  /* save to buffer */<br>n1 = 250 {72, 250} Correct 10: end if<br>n2 = 26 !𝑦′!" {72, 250, 26}<br>t0 = n0 / 3.6 {72, 250, 26, 20.0} 11: end if<br>t1 = n1 / t0 {72, 250, 26, 20.0, 12.5} ̂𝑠"$ 12: end for<br>t2 = n0 -answer = t0 * t1t1 {72, 250, 26, 20.0, 12.5, output = 1547 59.5} 13: θ ←−−−∇ update θL ( x, B, Pθ )<br>Solutions Intermediate States<br>NL<br>Input<br>Ref. Solution<br>Correct Solution Evaluate with Executor<br>Partially-Correct  Solution Prefix<br>**----- End of picture text -----**<br>


Figure 1: Examples of self-sampled correct and partiallycorrect solutions from MathQA (more in Appendix D). The steps and intermediate states marked in red are _incorrect_ . 

## 3 LEARNING FROM SELF-SAMPLED SOLUTIONS 

We now formally present our approach. There are three main steps: 1) _sampling_ 2) _filtering_ and 3) _learning_ as shown in Alg. 1. Here we mainly introduce the self-sampling framework using only fully-correct solutions and the extensions with partially-correct solutions will be introduced in § 3.3. 

## 3.1 ONLINE SAMPLING AND FILTERING 

For each specification _x_ , we maintain a buffer _B_ to save the different solutions that are correct, _i.e.,_ evaluate to the correct result. Note that the buffers are persistent and cumulative across training epochs. To add more solutions in _B_ , we perform online sampling and filtering as follows. **Online sampling (line 4 in Alg. 1):** With the NL question _x_ from each example ( _x, y[∗] , z[∗]_ ) as input, ˆ the model samples a set candidate solutions _Y_[ˆ] = _{yi}[n] i_ =1 _[∼][P][θ]_[(ˆ] _[y][|][x]_[)][;] **Filtering incorrect solutions(line 7 in Alg. 1):** As not all sampled solutions in _Y_[ˆ] are correct (thus ˆ ˆ not suitable for learning), we filter out all incorrect solutions in _Y_[ˆ] , _i.e., Y_[ˆ] _[∗]_ = _{y|y ∈ Y_[ˆ] ; _E_ (ˆ _y_ ) = _z[∗] }_ ; **Filtering duplicate solutions (line 8 in Alg. 1):** Because the model can sample solutions that are correct but are ”trivial variants” of other already saved solutions ( _e.g.,_ the solution differs from another solution only in white spaces, comments or trivial steps like ”x = x * 1.0”), we further filter the buffer to remove them. This is essential as all saved solutions will be directly used for learning and such undesired behavior from the model will be encouraged without the filtering process.[2] Concretely, we first perform filtering based on the linearized abstract syntax trees (ASTs) to eliminate the differences in white space, etc; then we set a constraint on maximum number of lines using the number of lines in _y[∗]_ as the reference to prevent saving solutions with trivial steps. 

## 3.2 LEARNING FROM MULTIPLE TARGETS 

With self-sampling, each natural language question is paired with multiple solutions as targets for learning. Here we discuss some common loss functions for the multi-target learning problem, with 

> 2Our preliminary experiments also show that the performance greatly degenerates when such trivial variants are left in the buffer for learning. 

3 

Published as a conference paper at ICLR 2023 

|Name|Loss Functions_L_(_x, B, Pθ_)|Gradients_∇θ_(_x, B, Pθ_)|
|---|---|---|
|MLE|_−_log_Pθ_(_y∗|x_)|_−∇θ_log_Pθ_(_y∗|x_)|
|MLE-Aug<br>MML<br>_β_-MML|_−_�<br>ˆ_y∈B_ log_Pθ_(ˆ_y|x_)<br>_−_log �<br>ˆ_y∈B Pθ_(ˆ_y|x_)<br>_−_1<br>_β_ log �<br>_y Pθ_(ˆ_y|x_)_β_|_−_�<br>ˆ_y∈B ∇θ_log_Pθ_(ˆ_y|x_)<br>_−_�<br>ˆ_y∈B_<br>_Pθ_(ˆ_y|x_)<br>�<br>˜_y∈B Pθ_(˜_y|x_)_∇θ_log_Pθ_(ˆ_y|x_)<br>_−_�<br>ˆ_y∈B_<br>_Pθ_(ˆ_y|x_)_β_<br>�<br>˜_y∈B Pθ_(˜_y|x_)_β_ _∇θ_log_Pθ_(ˆ_y|x_)|



Table 1: Comparison of loss functions and their gradients over multiple reference _B_ . Note that they all degenerates to MLE when only the gold reference solution is used as target, _i.e., B_ = _{y[∗] }_ . 

a focus on how each target contributes to the gradient. The loss functions and their gradients are shown in Tab. 1. 

**Augmented MLE (MLE-Aug):** This objective augments MLE with multiple targets simply by summing the loss from multiple solutions in _B_ , which is equivalent as minimizing the KL-divergence from _Pθ_ ( _y|x_ ) to _Q_ ( _y|x_ ) = _|B|_ 1 _[·]_[1] _[B]_[(] _[y]_[)][, where][ 1][(] _[·]_[)][ is a set indicator function. It encourages the model] to put equal weights on all targets by ensuring that all targets equally contribute to the gradient. **Maximum Marginal Likelihood (MML):** MML attempts to approximate _Pθ_ ( _z[∗] |x_ ) by marginalizing over the correct solutions in _B_ . However, for each target ˆ _y ∈B_ , the gradient of it is in proportion to the likelihood _Pθ_ (ˆ _y|x_ ) given by the model, which results in a positive feedback loop during gradient updates. It encourages the model to still put a majority of the probability on one of the solutions in _B_ as noted in (Guu et al., 2017). 

_β_ **-smoothed MML (** _β_ **-MML):** Proposed in Guu et al. (2017), the _β_ -MML objective is an extension of MML with a hyperparameter _β ∈_ (0 _,_ 1] to adjust weights of the gradient from each target. It an interpolation between MML and MLE-Aug objectives, more specifically, it recovers MML when _β_ = 1 and its gradient is equivalent to that of MLE-Aug when _β →_ 0. 

Empirically, we find that these distinctions between those loss functions greatly affects the model performance (Fig. 3), especially when partially-correct solutions are included for learning. 

## 3.3 LEARNING FROM PARTIALLY-CORRECT SOLUTIONS 

Besides learning from self-sampled _fully-correct solutions_ (FCSs), we can also let the model learn from _partially-correct solutions_ (PCSs). Our motivation is that the model often encounter solutions that are close to being correct as they only make mistakes in the last few steps ( _e.g.,_ Fig. 1), and these partially-correct solutions provide additional learning opportunities. Learning from PCSs could also address the issue that the sampler may have a low chance of encountering fully-correct solutions for complex tasks due to the sparse solution space. 

## 3.3.1 IDENTIFYING PARTIALLY-CORRECT SOLUTIONS 

When the model samples a solution that does not produce the desired answer, we want to identify if a prefix of this solution is partially correct, _i.e.,_ it performs some of the necessary computation steps needed for the correct solution, so that the model can additionally learn from these potentially unseen prefixes in the next iteration. A challenge here is figuring out when a prefix is partially correct. Ideally, we want to say a prefix _y≤i_ is partially correct if there exists a suffix _y>i_ such that their concatenation ( _y≤i||y>i_ ) is a correct solution. There are two caveats here: (1) if there is no length restriction on the suffix, it is always possible to find a suffix that complements any prefix ( _e.g.,_ a full gold solution is one such suffix); and (2) it is computationally very expensive to search for all suffixes (even with a length restriction) to check if a prefix can be completed to a correct solution. 

To overcome these challenges, we leverage the gold reference solutions and any self-sampled fullycorrect or even partially-correct solutions to help identify new partially-correct prefixes. The idea is to identify a prefix as partially correct if it produces a set of intermediate values (upon execution) that exactly matches the set of intermediate values produced by a prefix of a known correct or partiallycorrect solution. For such a prefix, we know that there exists a reasonable complement suffix based on the suffix of the known solutions. Note that, this definition of partial correctness is conservative compared to the ideal definition above, but it makes the computation significantly tractable. 

4 

Published as a conference paper at ICLR 2023 

**Algorithm 2** _SampleSolutions_ ( _x, Pθ, B_ ) with partially-correct solutions 

**Input:** Model _Pθ_ ( _y|x_ ); the NL input _x_ and a set of partially-correct solutions _B_ **Output:** Solution samples _Y_[ˆ] . 2:1: Sample a set of completionsSelect _y_ ˆ _≤i ∈B \ {y_ ˆ _|E_ (ˆ _y_ ) = _Y zp[∗] }∼_ uniformly at random _Pθ_ (ˆ _y>i|y_ ˆ _≤i, x_ ) /* sample PCS prefix for completion */ 3: _Y_[ˆ] _←{_ [ˆ _y≤i||y_ ˆ _>i_ ] _}y_ ˆ _>i∈Yp_ /* concatenate completions with the solution prefix */ 4: **return** _Y_[ˆ] 

Below, we formally define this notion of partial solutions that leverages existing known fully and partially correct solutions. 

**Intermediate state.** Given a solution _y_ = ( _u_ 1 _, ..., ut_ ) where _ui_ is the _i_ -th reasoning step, we define the intermediate state _si_ as the set of all variables values in the scope after executing the first _i_ steps _y≤i_ = ( _u_ 1 _, ..., ui_ ), which we call a _prefix_ of this solution. It is easy to see that the prefixes _y≤i_ and intermediate states _si_ of a solution construct a bijective function, which is also illustrated in Fig. 1. Note that the state representation is name-agnostic since variable names do not typically contributes to the semantics of the solutions. 

**State-based equivalence and partial correctness.** Given the definition of the intermediate state, we say the prefixes of two solutions, _y≤i_ and _y≤[′] j_[, are] _[ semantically equivalent]_[ if and only if] _[ s][i]_[=] _[ s][′] j_[,] _i.e.,_ those two solutions produces the exact same set of variable values. And then we define _partial correctness_ as follows: a solution prefix _y≤i_ is partially-correct if and only if it is semantically equivalent to the prefix of another known partially-correct solution _y≤[∗] j_[.][As][we][keep][all][known] partially-correct solutions in the buffer _B_ , formally: 

_PartiallyCorrect_ ( _y≤i_ ) _⇐⇒∃y[∗] ∈B. ∃j ≤|y[∗] | s.t. s[∗] j_[=] _[ s][i]_ 

## 3.3.2 MODIFICATIONS TO THE MAIN ALGORITHM 

To support learning from partial solutions, we modify Alg. 1 as follows to enable buffering and sampling from partial solutions. The fully updated algorithm is shown in Appendix C. 

**Guided-Sampling:** In § 3.1, we mentioned that full solutions are sampled for each question _x_ as ˆ _y ∼ Pθ_ (ˆ _y|x_ ). With PCS prefixes, compared with sampling a solution from scratch, generating solutions with these prefixes reduces the generation length thus the model can more efficiently explore the solution space. This guided sampling process is described in more detail in Alg. 2. Note that since the empty solution _y_[0] is in the buffer _B_ since initialization, therefore model can still generate and explore the space from scratch and not always follows the existing solution prefixes. **Identify partially-correct prefixes:** As mentioned in § 3.3, if a solution _y_ ˆ does not produce the expected result _z[∗]_ but its prefix _y_ ˆ _≤i_ is partially-correct, the model can still learn from its prefix. However, an important task here is to identify the longest partially-correct prefix for learning, in other words, locate the exact step that the solution deviates from a correct reasoning path. We can achieve this simply by backtracking the intermediate states and find the first state that is equivalent to any of the states from a saved solution.[3] 

**Filtering solution prefixes:** With the inclusion of partially-correct solutions, we need to slightly change the two filtering criteria in § 3.1. For deduplication, while we still use AST to rule out changes with non-semantic tokens such as white space, we also check if the partially-correct solution ˆ correct solutionprefix _y≤i_ is a prefix of another known PCS in _y_ ˆ, we need to prune out any existing solution in _B_ . For the same reason, when saving a new partially- _B_ that is a prefix of _y_ ˆ. As for the length constraint, the same principle still applies, but now it is compared against other partiallycorrect solution that _executes to the same state_ . 

**Learning objective:** As partially-correct solutions are solution prefixes _y≤i_ missing the later part _y>i_ , with an auto-regressive generation model, the learning of _Pθ_ ( _y≤i|x_ ) is independent of _y>i_ . Thus the learning objectives in § 3.2 do not need to change with the inclusion of PCS in the buffer for learning. The only difference is that the end-of-sequence “ `〈` eos `〉` ” token is not appended to the PCS as those solutions are not yet finished. 

> 3In practice, we use a state _→_ solution prefix dictionary and the lookup takes a negligible amount of time. 

5 

Published as a conference paper at ICLR 2023 

## 4 EXPERIMENTS 

## 4.1 EXPERIMENTAL SETUP 

**Datasets.** We evaluate on two math reasoning datasets, in which we generate straight-line Python programs as solutions to solve math problems described in natural language. We finetune the language models to output such program solutions using only the natural language problem description as the input. 

_▷_ **MathQA-Python-Filtered:** The original MathQA-Python consists of 19.2K training examples of NL and Python program pairs (Austin et al., 2021). However, we find the raw dataset to contain many questions that share the same question templates and only differ in concrete number across the train/dev/test sets. To better understand the generalization of the trained models, we derive a deduplicated version of the dataset by first merging the train and dev data and then perform templatebased deduplication. Partly inspired by Finegan-Dollak et al. (2018), we re-split the train and dev set based on the question templates, resulting in 6.8K/0.7K train/dev data for the filtered version.[4] While we mainly experiment on the filtered version, we report performance on both versions when compared with previous methods. 

_▷_ **GSM5.5K-Python:** The grade-school-math (GSM8K) dataset (Cobbe et al., 2021) contains 7.5K training data points. Since it only provides natural language solutions with math formulas and does not have a dev set, we first reserved 20% of the training data as dev set, then automatically converted the formulas to program solutions in the same style as MathQA-Python. As the result, we finetune our models with the 5.5K successfully converted training examples. Note that the natural language solutions/explanations are not used as input to the models in our experiments. 

**Evaluation metrics:** Following recent work in neural program synthesis (Austin et al., 2021; Chen et al., 2021a; Chowdhery et al., 2022) and math reasoning (Cobbe et al., 2021), we use PASS@ _k_ as our main evaluation metric. It allows the model to sample _k_ solutions for each question and the task is considered solved if any one of the _k_ solutions is correct, so PASS@ _k_ can also be seen as the fraction of problems in the test/dev set being solved given _k_ attempts. More details ( _e.g.,_ temperature) can be found in Appendix A. 

**Model training:** We use GPT-Neo (Black et al., 2021) as our language model and mainly study two model sizes, 125M and 2.7B.[5] Following previous work (Austin et al., 2021), we evaluate all PASS@ _k_ on the same model checkpoint that has the best PASS@1 score, but note that it might not be the best checkpoint for other _k_ values (more discussion in Appendix E). Detailed hyperparameter settings can also be found in Appendix A. 

## 4.2 MAIN RESULTS 

**Learning from self-sampled solutions improves PASS@** _k_ **.** Fig. 2 shows the performance on the two datasets by learning from self-sampled FCSs and PCSs using MLE-Aug (orange bars), compared with MLE on single reference solution (blue bars). We can see that our proposed method can greatly improve PASS@ _k_ , especially for higher _k_ values. By comparing different model sizes, we can see that learning from self-sampled solutions can help with both small and large models, with a +12.3% and +9.0% PASS@100 improvement on GSM5.5K-Python for GPT-Neo-125M and GPT-Neo-2.7B, respectively and a +3.1% and +8.6% PASS@80 improvement on MathQA-PythonFiltered for GPT-Neo-125M and GPT-Neo-2.7B, respectively. We note that our approach does not improve PASS@1, which is expected as learning from multiple targets mainly helps with increasing the diversity of the sampled solutions rather than improving the most-probable solution (for which MLE is better suited). 

**Partially-correct solutions improve model performance.** We next show the effects of including partially-correct solutions on PASS@ _k_ performance in Fig. 2 (green bars vs orange bars) and the number of saved FCSs and PCSs in Fig. 5. First, we observe from Fig. 5 that using partial correctness not only results in PCSs being saved and directly learned from, but it also boosts the number of FCSs being found with the guided-sampling process. As a result, most PASS@ _k_ performances drop if we 

> 4We will release the processing scripts for replication and comparison. 

> 5We choose GPT-Neo because it was the only public language model that have been pretrained on code when we conduct the experiments. 

6 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0007-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0007-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0007-03.png)


**----- Start of picture text -----**<br>
(a) GSM5.5K-Python with GPT-Neo 125M (b) GSM5.5K-Python with GPT-Neo 2.7B<br>(c) MathQA-Python-Filtered with GPT-Neo 125M (d) MathQA-Python-Filtered with GPT-Neo 2.7B<br>**----- End of picture text -----**<br>


Figure 2: Percentage of the problems solved (PASS@ _k_ ) on the dev set of GSM5.5K-Python and MathQA-Python-Filtered, comparing our self-sampling approach and the common MLE objective. All our methods include partially-correct solutions and use the MLE-Aug loss for learning. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0007-05.png)


Figure 3: PASS@100 comparison of various loss functions (§ 3.2) under different self-sampling strategies. Results are on the dev set of GSM5.5K-Python with finetuned GPT-Neo 125M model. _β_ = 0 _._ 25 for _β_ -MML. Full results available as Tab. 5 in Appendix B. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0007-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0007-08.png)


Figure 5: Number of saved FCSs and PCSs per problem for GSM5.5K-Python (left) and MathQA-PythonFiltered (right), with different self-sampling strategies and model sizes. # FCSs _includes_ the reference solution. 

do not include partially-correct solutions in the buffer, as the model learns from a smaller number of FCSs and PCSs as targets. The one exception is the GPT-Neo 125M model on the MathQA-PythonFiltered dataset, where we do not observe any advantage/disadvantage of using PCSs. 

**MLE-Aug loss function works the best.** We next study the effects of different objective functions for learning from multiple targets as described in § 3.2. We also experiment under different selfsampling strategies ( _i.e.,_ FCS only or FCS + PCS), and our experiment results on GSM5.5K-Python with the GPT-Neo 125M model are shown in Tab. 5. We can see that MLE-Aug loss results in the biggest improvement compared to other losses both with just FCSs and with FCSs + PCSs. MML performs the worst: it only marginally improves over MLE with only FCS and performs worse than MLE when also learning from PCSs. As discussed in § 3.2 and Tab. 1, the gradient of MML is in proportional to the likelihood given by the model, thus it encourages the model to put all weight on one solution in the buffer. As MLE already learns from the gold reference solution, it is hard for 

7 

Published as a conference paper at ICLR 2023 

|Models|**Original Version**<br>**Filtered Version**|
|---|---|
||PASS@1<br>PASS@80<br>PASS@1<br>PASS@80|
|_Previous work_:<br>Codex Davinci_†_ (Chen et al., 2021a)<br>LaMDA 68B_∗_(Austin et al., 2021)<br>LaMDA 137B_∗_(Austin et al.,2021)<br>_Ours:_<br>GPT-Neo 125M w/ self-sampling FCS + PCS<br>GPT-Neo 2.7B w/ self-sampling FCS + PCS|6.0<br>42.0<br>5.0<br>**40.0**<br>-<br>79.5<br>-<br>-<br>-<br>81.2<br>-<br>-<br>**77.6**<br>**84.7**<br>11.7<br>28.2<br>-<br>-<br>**20.7**<br>36.2|



Table 2: Comparison with previous methods on the original (test set used) and filtered version (dev set used) of the MathQA-Python dataset. _∗_ : model not pretrained on code. _†_ : few-shot learning results. -: no results available. 

MML to make improvements with self-sampled solutions, and the performance may even decrease when MML puts all weight on an incomplete partially-correct solution. In contrast, the gradients of MLE-Aug objective are equally distributed among the targets, which leads to more diversity in its generation due to a more balanced source of learning signals. _β_ -MML loss is proposed to alleviate the aforementioned issue for MML loss, but we do not observe an advantage of using it instead of the MLE-Aug loss in our experiments. 

## 4.3 ADDITIONAL ANALYSIS 

**Diversity of the solutions.** By inspecting the _k_ generated solutions for each task, we find that there is more diversity in the solutions that the model generates using our method. More specifically, we calculate the ratio of _unique_ solutions from the 100 samples for the comparison in Fig. 2a, and find that 30.5% of them are unique for our approach but only 20.8% for the model trained with MLE. 

**Dynamics between # of PCSs and FCSs saved in the buffer.** As discussed above, more saved solutions typically results in better PASS@ _k_ performance. Interestingly, when comparing different model sizes, we can see that while the sum of partially and fully-correct solutions sampled and saved in the buffer are about the same ( _i.e.,_ 3.36 and 3.31) for GSM5.5K-Python dataset in Fig. 5, around 60% of them are FCS for the small model while it is 76% for the larger model. The difference in percentage of PCSs left in the buffer also reflects the model’s ability for completing partially-correct solution prefixes. We also find that during early stages of training, the number of PCSs rapidly grows while the model is relatively weak to sample FCSs, thus the PCSs help enriching the learning signal and preventing overfitting early-on. More discussions about this can be found in Appendix E. 

**Comparison to previous works** Here we compare with previous work on both the original and the filtered versions of MathQA-Python datasets in Tab. 2. On the original dataset, self-sampling with GPT-Neo 125M is able to outperform previous methods that finetune 137B model pretrained on natural language. We also compare with Codex model used in a few-shot setting (more details in Appendix A), and find that on the much harder filtered dataset, a 2.7B GPT-Neo model finetuned with our methods obtains much better PASS@1 but with lower PASS@80. By inspecting the output from Codex, we discover that its outputs are much more diverse than finetuned models, which contributes to a higher PASS@80 even under the few-shot setting. Comparison with previous work on the GSM dataset is in Appendix B due to limited space. 

## 5 LIMITATIONS AND FUTURE WORK 

**More general definition of (partial) correctness.** In this work, we define partial correctness based on state-based solution equivalence. This is a conservative way for defining solution equivalence as it requires exact match of the sets of variable values, but a solution could be partially correct and yet, not have an exact match of variable values because some of these values may not needed for future computation. In the future, we want to explore ways to relax this restriction that will help us find more partially correct solutions in an efficient manner. Besides, our partial correctness definition requires the existence of at least one fully-correct solution and when such reference solution is 

8 

Published as a conference paper at ICLR 2023 

not available from the dataset ( _i.e.,_ in a weakly-supervised setting), we would need to first sample an FCS that matches the gold execution result to begin with. In addition, we simply use the matching of execution results to define correctness, which is susceptible to spurious solutions that achieves the correct result by coincidence. For math reasoning, we find such spurious solutions to be quite rare[6] , as the correct answer is typically numeric which is less likely for a semantically wrong solution to obtain the correct answer by chance. But methods as Zhong et al. (2020); Chen et al. (2022) may be 

**Towards generating general programs.** While we focus on the domain of generating solutions for math reasoning in this work, here we reflect on how our method can be applied to program synthesis in general. However, general programs might contain complex structures such as conditions ( _e.g.,_ if-else) or loops ( _e.g.,_ while-do) as opposed to straight-line programs in the mathreasoning domain. Dealing with these complex structures poses additional challenges because most neural program synthesis models perform left-to-right auto-regressive generation, and the changes to the control flow break the alignment between program generation and program execution (Chen et al., 2018; 2021b; Nye et al., 2021). There are two potential ways to extend our technique to address the problem. First, we can treat a branch or a loop as an atomic unit ( _i.e.,_ a block whose state is the state after executing all statements within it), then we can apply state-based equivalence in the same way. Second, because our technique only requires execution after the full programs are generated, we can still evaluate and compare program states based on intermediate states. 

## 6 RELATED WORK 

**Weakly-supervised semantic parsing.** Many previous work in learning semantic parsers from weak supervision follows the same process of sampling programs and maximizing the probability of the correct ones (Krishnamurthy et al., 2017; Guu et al., 2017; Min et al., 2019; Ni et al., 2020). Our work differs as our tasks contain one reference solution for each task as opposed to only the final answer like weakly-supervised semantic parsing tasks. Thus, our work leverages the reference solution for sampling and defines partial correctness based on known solutions. Because of the problem setup difference, we found that the conclusions in Guu et al. (2017) about loss functions do not generalize to our case. 

**Execution-guided code generation.** Our work relates to execution-guided code generation as we leverage intermediate states of math solutions to guide the sampling process. In code generation literature, intermediate program execution states are used to prune the search space (Liang et al., 2017; Wang et al., 2018; Li et al., 2022) or condition further generation on the execution states(Chen et al., 2018; Ellis et al., 2019; Nye et al., 2020; Chen et al., 2021b; Nye et al., 2021). The key difference of these methods from ours is that they require doing both decoding and execution at inference time, while our work only uses execution during training, which reduces decoding overhead. 

**Learning from partial reward for program synthesis.** There are parallels between multi-target learning and the reinforcement learning setting with sparse rewards for generating programs (Liang et al., 2017; 2018; Simmons-Edler et al., 2018; Bunel et al., 2018; Agarwal et al., 2019). Similarly, our approach of identifying partial correctness of solutions is similar to partial rewards. But instead of discounting an entire trajectory with a low reward as in RL, we truncate the solution to a partiallycorrect prefix and assign it the “full reward”, which is a main contribution of this work. 

## 7 CONCLUSION 

We propose to let pretrained language models sample additional solutions for each problem and learn from the self-sampled solutions that are correct or partially-correct. We define partial correctness by tracing and matching intermediate execution states. We experiment on different math reasoning tasks and show that such partially-correct solutions can help more efficient exploration of the solution space and provide useful learning signal, which improves the PASS@ _k_ performance. Overall, our proposed method can improve PASS@ _k_ from 3.1% to 12.3% compared to learning from a single solution with MLE. 

> 6We manually inspected the self-sampled FCSs by GPT-Neo 2.7B on 100 tasks of GSM5.5K and found spurious solutions only exist for 3 of them. 

9 

Published as a conference paper at ICLR 2023 

## ACKNOWLEDGEMENTS 

The authors would like to thank Jackson Woodruff, Pengcheng Yin, and the anonymous reviewers for the useful discussion and comments. 

## REFERENCES 

- Rishabh Agarwal, Chen Liang, Dale Schuurmans, and Mohammad Norouzi. Learning to generalize from sparse and underspecified rewards. In _International Conference on Machine Learning_ , pp. 130–140. PMLR, 2019. 

- Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_ , 2021. 

- Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow, March 2021. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- Rudy Bunel, Matthew Hausknecht, Jacob Devlin, Rishabh Singh, and Pushmeet Kohli. Leveraging grammar and reinforcement learning for neural program synthesis. In _International Conference on Learning Representations_ , 2018. 

- Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. Codet: Code generation with generated tests. _arXiv preprint arXiv:2207.10397_ , 2022. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021a. 

- Xinyun Chen, Chang Liu, and Dawn Song. Execution-guided neural program synthesis. In _International Conference on Learning Representations_ , 2018. 

- Xinyun Chen, Dawn Song, and Yuandong Tian. Latent execution for neural program synthesis beyond domain-specific languages. _Advances in Neural Information Processing Systems_ , 34, 2021b. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_ , 2022. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. 

- Kevin Ellis, Maxwell Nye, Yewen Pu, Felix Sosa, Josh Tenenbaum, and Armando Solar-Lezama. Write, execute, assess: Program synthesis with a repl. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Catherine Finegan-Dollak, Jonathan K Kummerfeld, Li Zhang, Karthik Ramanathan, Sesh Sadasivam, Rui Zhang, and Dragomir R Radev. Improving text-to-sql evaluation methodology. In _ACL_ , 2018. 

10 

Published as a conference paper at ICLR 2023 

Kelvin Guu, Panupong Pasupat, Evan Zheran Liu, and Percy Liang. From language to programs: Bridging reinforcement learning and maximum marginal likelihood. In _ACL_ , 2017. 

- Jayant Krishnamurthy, Pradeep Dasigi, and Matt Gardner. Neural semantic parsing with type constraints for semi-structured tables. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ , pp. 1516–1526, 2017. 

- Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, R´emi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. _arXiv preprint arXiv:2203.07814_ , 2022. 

- Chen Liang, Jonathan Berant, Quoc Le, Kenneth Forbus, and Ni Lao. Neural symbolic machines: Learning semantic parsers on freebase with weak supervision. In _ACL_ , pp. 23–33, 2017. 

- Chen Liang, Mohammad Norouzi, Jonathan Berant, Quoc V Le, and Ni Lao. Memory augmented policy optimization for program synthesis and semantic parsing. _Advances in Neural Information Processing Systems_ , 31, 2018. 

- Sewon Min, Danqi Chen, Hannaneh Hajishirzi, and Luke Zettlemoyer. A discrete hard em approach for weakly supervised question answering. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pp. 2851–2864, 2019. 

- Ansong Ni, Pengcheng Yin, and Graham Neubig. Merging weak and active supervision for semantic parsing. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pp. 8536– 8543, 2020. 

- Maxwell Nye, Yewen Pu, Matthew Bowers, Jacob Andreas, Joshua B Tenenbaum, and Armando Solar-Lezama. Representing partial programs with blended abstract semantics. In _International Conference on Learning Representations_ , 2020. 

- Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. _arXiv preprint arXiv:2112.00114_ , 2021. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. _J. Mach. Learn. Res._ , 21(140):1–67, 2020. 

- Tal Schuster, Ashwin Kalyan, Alex Polozov, and Adam Tauman Kalai. Programming puzzles. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)_ , 2021. 

- Riley Simmons-Edler, Anders Miltner, and Sebastian Seung. Program synthesis through reinforcement learning guided tree search. _arXiv preprint arXiv:1806.02932_ , 2018. 

- Chenglong Wang, Kedar Tatwawadi, Marc Brockschmidt, Po-Sen Huang, Yi Mao, Oleksandr Polozov, and Rishabh Singh. Robust text-to-sql generation with execution-guided decoding. _arXiv preprint arXiv:1807.03100_ , 2018. 

- Ruiqi Zhong, Tao Yu, and Dan Klein. Semantic evaluation for text-to-sql with distilled test suites. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 396–411, 2020. 

11 

Published as a conference paper at ICLR 2023 

## APPENDIX 

## A EXPERIMENT SETTING DETAILS 

|Name<br>MathQA<br>GSM8K.|Name<br>MathQA<br>GSM8K.|
|---|---|
|||
|# Training Steps<br>Learning Rate (LR)<br>Optimizer<br>Adam Betas<br>Adam Eps<br>Weight Decay<br>LR Scheduler<br># LR Warm-up Steps<br>Effective Batch Size<br>FP Precision<br>Gradient Clipping|50K<br>25K<br>1.0e-4<br>AdamW<br>(0.9, 0.999)<br>1.0e-8<br>0.1<br>Linear w/ Warmup<br>100<br>32<br>FP 32 for 125M, FP16 for 2.7B<br>1.0|



Table 3: The hyperparameters used for model training on two different types of datasets. 

**Hyperparameters.** All hyperparameters for training is shown in Tab. 3. We use _β_ = 0 _._ 25 in the experiments with _β_ -MML, as a result of enumeration search among the values of _{_ 0 _._ 1 _,_ 0 _._ 25 _,_ 0 _._ 5 _,_ 0 _._ 9 _}_ . We use the default AdamW optimizer settings and slightly tuned the learning rate by trying out several values between 1.0e-3 and 1.0e-5. The difference in floating point precision is to fit the GPT-Neo 2.7B model into the memory of the GPUs. All experiments are conducted on V100-32GB GPUs. 

**PASS@** _k_ **evaluation.** We use temperature sampling and sample _n_ solutions with _T_ = 0 _._ 8, where _n_ = 80 for MathQA and _n_ = 100 for GSM to evaluate PASS@ _n_ , to be maximally consistent with previous work (Austin et al., 2021; Cobbe et al., 2021; Chowdhery et al., 2022). We also report PASS@ _{_ 5 _,_ 10 _,_ 20 _,_ 50 _}_ using the _n_ samples and the unbiased estimator proposed in Chen et al. (2021a). We use _T_ = 0 _._ 2 to sample 1 solution per specification and evaluate PASS@1. 

**Codex few-shot settings.** We estimate the Codex (Chen et al., 2021a) performance under the few-shot settings. More specifically, the prompt consists of a natural language task description ”# Generate Python code to solve the following math word problems:” and four examples, following previous work (Chowdhery et al., 2022). Each example consists of the NL specification as a one-line comment and the gold program solutions. We evaluate PASS@ _k_ for Codex using the same sampling methods as above. 

**Details for self-sampling.** During a training step, we sample one solution[7] for each task ( _i.e.,_ natural language problem) in the batch, _i.e., |Y_[ˆ] _|_ = 1 in Alg. 1 and Alg. 2. Thus for each gradient update, we first compute the loss for each task based on the saved solutions in the buffer and loss functions described in Tab. 1, then it is averaged across the 32 tasks in the batch. Note that the total number of samples we generate per task throughout training is also scaled up by the number of training epochs, which is 235 for MathQA-Python-Filtered, 83 for MathQA-Python and 145 for GSM5.5K-Python. For sampling temperature, we use the same setting as inference time, with _T_ = 0 _._ 8. 

## B ADDITIONAL EXPERIMENT RESULTS 

**Comparing GSM performance with previous work.** Here we compare our method with previous work on the original test sets of GSM8K. The results are shown as Tab. 4. On GSM8K, some of the prior works are evaluated on a different format of NL inputs than ours, so they are not directly comparable, but we still include them to help better position the performance of our methods. We test Codex using the same input in a few-shot setting, and we find that similar with the 

> 7We also experiment with higher sampling budgets but do not observe significant improvements. 

12 

Published as a conference paper at ICLR 2023 

|Models|PASS@1|PASS@100|
|---|---|---|
|_Previous work_:|||
|OpenAI 6B_∗♣_(Cobbe et al., 2021)|21.8|70.9|
|PaLM-Coder 540B_†♣_(Chowdhery et al., 2022)|**50.9**|-|
|LaMDA 137B_∗†♣_(Chowdhery et al., 2022)|7.6|-|
|Codex Cushman_†_ (Chen et al., 2021a)|5.0|58.0|
|Codex Davinci_†_ (Chen et al.,2021a)|17.0|**71.0**|
|_Ours_:|||
|GPT-Neo 2.7B w/ self-sampling FCS + PCS|19.5|41.4|



Table 4: Compare with previous methods on the original test set of GSM8K dataset. _[∗]_ : model not pretrained on code. _[†]_ : few-shot learning results. _[♣]_ : different setting from ours[8] . 

|Self-Sampling<br>Loss Func.|**# Sols. in**_B_<br>FCS<br>PCS|**PASS@k(%)**|
|---|---|---|
|||_k_=1<br>_k_=5<br>_k_=10<br>_k_=20<br>_k_=50<br>_k_=100|
|-<br>MLE|-<br>-|7.4<br>10.6<br>12.7<br>15.3<br>19.2<br>22.7|
|FCS only<br>MML<br>MLE-Aug<br>_β_-MML|1.48<br>-<br>**1.76**<br>**-**<br>1.57<br>-|6.9<br>11.0<br>13.3<br>16.0<br>20.1<br>23.7<br>**7.6**<br>**13.1**<br>**16.5**<br>**20.5**<br>**26.8**<br>**32.3**<br>7.5<br>11.7<br>14.5<br>17.9<br>23.1<br>27.3|
|FCS + PCS<br>MML<br>MLE-Aug<br>_β_-MML|1.40<br>1.10<br>**2.00**<br>**1.36**<br>1.62<br>1.14|5.5<br>9.0<br>11.0<br>13.1<br>16.2<br>18.7<br>**7.5**<br>**13.6**<br>**17.5**<br>**22.1**<br>**29.2**<br>**35.0**<br>7.2<br>12.0<br>14.9<br>18.4<br>23.6<br>27.9|



Table 5: Full comparison of various loss functions (§ 3.2) with different self-sampling strategies. Results are on the dev set of GSM5.5K-Python with GPT-Neo 125M as the base model. Best performance within the same category is in **bold** and ones _worse than MLE_ is underlined. _β_ = 0 _._ 25 for _β_ -MML. 

result on MathQA in Tab. 2, our method achieves better PASS@1 while being significantly worse in PASS@100 compared with Codex. We hypothesize that as Codex model is used tested few-shot setting and not finetuned, it does not suffer from the overfitting issue we mentioned. This leads to great diversity but poor accuracy during generation. However, due to the little information we have about Codex ( _e.g.,_ model size, training data), it is hard to derive any further conclusion. 

**Ablation results on loss functions.** Here we show the full results on the ablation of loss functions in Tab. 5. We can see that trends observed from PASS@100 in Fig. 3 are consistent with other PASS@ _k_ results, as MLE-Aug loss beats other two loss functions on all PASS@ _k_ . And using MML loss when adding PCSs for learning results in worse performance than MLE for PASS@1 as well. Moreover, from the number of FCSs and PCSs saved in the buffer _B_ , we can also observe that using MLE-Aug loss results in more FCSs and PCSs being saved, thus further encourages diversity in generation. 

## C FULL LEARNING ALGORITHM WITH PARTIAL CORRECTNESS 

Our general learning framework in shown as Alg. 1 and it is further extended in § 3.3. Here we show a complete version of the algorithm with using partially-correct solutions in Alg. 3. Additionally, here are the detailed explanation of the data structure and functions used in it: 

_▷_ **Mapping** _M_ : This is a data structure that maps an intermediate state to a set of solution (prefixes) that execute to that state, _i.e., M_ : _S →Y[n]_ . In this mapping, we save _all_ PCSs and their intermediate states, _including all prefixes_ of any PCS. We use this to significantly speed up the lookup process as mentioned in § 3.3.2; _▷_ **Function** _**PartialCorrectnessCriteria**_ ( _si, M_ ): Since all states for all known PCSs are saved in _M_ , to know whether a prefix _y_ ˆ _≤i_ is partially-correct, we only need to check if its state matches any 

> 8Natural language explanations of the solutions are used as input and the few-shot exemplars are not in the same format as ours. 

13 

Published as a conference paper at ICLR 2023 

**Algorithm 3** Training Update with Partially Correctness 

**Initialize: (only once before training starts)** Solutions buffer _B_ = _{y_[0] _, y[∗] }_ with an empty and the reference solution Reference solution states ( _s[∗]_ 1 _[, s][∗]_ 2 _[, ..., s][∗] t_[)][ where] _[ s][∗] i_[=] _[ T]_[ (] _[y] ≤i_[)] State-prefixes mapping _M_ = _{s[∗] i[→{][y] ≤i[}}][t] i_ =1 **Input:** Parameterized model _Pθ_ ( _y|x_ ) A training example ( _x, y[∗] , z[∗]_ ) Tracing function _T_ : _Y →S_ to obtain intermediate states 1: _Y_[ˆ] _← SampleSolutions_ ( _x, Pθ, B_ ) /* call Alg. 2 */ 2: **for** _y_ ˆ **in** _Y_[ˆ] **do** ˆ 4:3: **for** _s ii ←|←Ty_ (ˆ _|_ ; _y i≤ ̸_ = 0 _i_ ) /* get intermediate state for each solution prefix; _i ← i −_ 1 **do** _y_ ˆ _≤i_ */ 5: **if** _PartialCorrectnessCriteria_ ( _si, M_ ) **then** 6: _YS ←M_ ( _si_ ) /* get existing prefixes that executes to state _si_ */ 7: **if not** _isDuplicate_ (ˆ _y≤i, YS_ ) **then** 8: _B ← updateBuffer_ (ˆ _y≤i, B_ ) 9: _M ← updateMapping_ (ˆ _y≤i, M_ ) 10: **end if** 11: **continue** /* we only need the longest matching prefix */ 12: **end if** 13: **end for** 14: **end for** update 15: _θ ←−−−∇θL_ ( _x, B, Pθ_ ) 

of the known states for a PCS, _i.e.,_ if _si ∈M_ ; 

_▷_ **Function** _**isDuplicate**_ (ˆ _y≤i, YS_ ): As mentioned in § 3.3.2, we use AST and length constraint to rule out ”trivial variants” and identify new PCSs to save in the buffer _B_ . Here the solutions to compare are the set of solutions _YS_ that reaches the same intermediate state, _i.e.,_ being state-based equivalent; also need to prune out the saved solutions that are prefix of _▷_ **Function** _**updateBuffer**_ (ˆ _y≤i, B_ ): Here we not only need to add the new PCS into the buffer _y_ ˆ _≤i_ ; _B_ , but 

_▷_ **Function** _**updateMapping**_ (ˆ _y≤i, M_ ): Here we need to save the states of all prefixes of an identified partially-correct solution, thus we will loop through all prefixes of ˆ _y≤i_ and obtain its execution state, then update _M_ accordingly. As mentioned above, existing PCSs may be a prefix of the new PCS, so we also need to prune out such existing PCSs from mapping _M_ . 

## D QUALITATIVE ANALYSIS 

In Tab. 6, we show more examples of the fully-correct and partially-correct solutions that the models found during self-sampling, from both the MathQA and GSM datasets. First, we can see that for some NL problems, it is possible that no FCS or PCS can be found with self-sampling, as in _MathQA-Example-1_ and _MathQA-Example-1_ . Take _MathQA-Example-2_ as an example, the question is quite straightforward thus it leaves very little room for the existence of other correct solutions, as the reference solution is already very short. Moreover, we can also observe that the ways self-sampled FCS and PCS differ from the reference solution vary a lot. In _MathQA-Example-2_ , _GSM-Example-1_ and _GSM-Example-2_ the sampled FCSs complete the task with very different paths compared with the reference solution, and actually result in using fewer lines of code. Another way of getting FCS or PCS is to perform small and local perturbations, _e.g.,_ switch the two sides of a addition or re-order the two non-dependent statements, as shown in other examples. We find that these local perturbations are more common in general in both datasets, as such patterns are easier for the model to learn. 

## E TRACKING TRAINING PROGRESS 

**Learning from self-sampled solutions mitigates overfitting.** Here we shown the PASS@ _k_ performance curve with respect to the training process in Fig. 6. From the curves, we can observe that for MLE, while PASS@1 and PASS@5 generally improves during training, other PASS@ _k_ for 

14 

Published as a conference paper at ICLR 2023 

|**NL Problem Descriptions**|**Ref. Solution**|**Self-Sampled FCS**|**Self-Sampled PCS**|
|---|---|---|---|
|_(MathQA-Example-1):_|n0=70.0|n0=70.0|-|
|The charge for a single room at hotel P|n1=10.0|n1=10.0||
|is 70 percent less than the charge for a|t0=100.0-n0|t0=100.0-n1||
|single room at hotel R and 10 percent|t1=100.0-n1|t1=100.0-n0||
|less than the charge for a single room at|t2=t0/t1|t2=t0/t1||
|hotel G. The charge for a single room|t3=t2*100.0|t3=t2*100.0||
|at hotel R is what percent greater than|t4=100.0-t3|answer=t3-100.0||
|the charge for a single room at hotel G?|t5=t4/t3|||
||answer=t5*100.0|||
|_(MathQA-Example-2):_|n0=9.0|-|-|
|If john runs in the speed of 9 km/hr|n1=300.0|||
|from his house, in what time will he|t0=n0*1000.0|||
|reach the park which is 300m long|t1=n1/t0|||
|from his house?|answer=t1*60.0|||
|_(MathQA-Example-3):_|n0=15.0|n0=15.0|n0=15.0|
|A class consists of 15 biology students|n1=10.0|n1=10.0|n1=10.0|
|and 10 chemistry students. If you pick|t0=n0+n1|t0=n0+n1|t0=n0+n1|
|two students at the same time, what’s|t1=n0/t0|t1=n0/t0|t1=n0/t0|
|the probability that one is maths and|t2=n1/t0|t2=n1/t0|t2=n1/t0|
|one is chemistry?|t3=t0-1.0|t3=t0-1.0|t3=t0-1.0|
||t4=n1/t3|t4=n1/t3|t4=n0/t3|
||t5=n0/t3|t5=n0/t3|t5=n1/t3|
||t6=t1*t4|t6=t1*t4||
||t7=t5*t2|t7=t5*t2||
||answer=t6+t7|answer=t7+t6||
|_(GSM-Example-1):_|n0=2|n0=10|n0=10|
|Ellie has found an old bicycle in a feld|n1=10|n1=5|n1=5|
|and thinks it just needs some oil to|n2=5|t0=n0+n1|n2=2|
|work well again. She needs 10ml of|t0=n0*n1|answer=n0+t0||
|oil to fx each wheel and will need an-|answer=t0+n2|||
|other 5ml of oil to fx the rest of the||||
|bike. How much oil does she need in||||
|total to fx the bike?||||
|_(GSM-Example-2):_|n0=20|n0=25|n0=2|
|There is very little car traffc on Happy|n1=100|n1=2|n1=25|
|Street.<br>During the week, most cars|n2=25|n2=20|n2=20|
|pass it on Tuesday - 25.<br>On Mon-|n3=2|n3=100|n3=100|
|day, 20% less than on Tuesday, and on|n4=10|n4=10|n4=10|
|Wednesday, 2 more cars than on Mon-|t0=n0/n1*n2|t0=n0-n1||
|day.<br>On Thursday and Friday, it is|t1=n2-t0|t1=n2/n3*n0||
|about 10 cars each day. On the week-|t2=t1+n3|t2=t0-t1||
|end, traffc drops to 5 cars per day.|t3=n4*n3|t3=t2+n4||
|How many cars travel down Happy|t4=t0*n3|t4=n0-t3||
|Street from Monday through Sunday?|answer=t3+n2|answer=t4+n3||
||_\_|||
||+t2+t3+t4|||



Table 6: More examples of self-sampled fully-correct (FCS) and partially-correct solutions (PCSs). ”MathQA” denotes the MathQA-Python-Filtered dataset and ”GSM” denotes the GSM5.5K-Python dataset. All solutions are from the _final buffer after training_ a GPT-Neo 2.7B model, while learning from self-sampled FCS+PCS with the MLE-Aug loss. 

higher _k_ actually decreases after reaching the peak performance in early epochs, which is consistent with previous findings (Cobbe et al., 2021). This is due to overfitting: in the early stage of training, the model is less confident about its predictions thus the sampled _k_ solutions are very diverse, and while training continues, it overfits to the one reference solution provided for learning thus leads to poor generalization when evaluated by PASS@ _k_ with high _k_ values. Fig. 6 also shows 

15 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0016-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0016-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0016-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0016-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0016-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0016-06.png)


Figure 6: How PASS@ _k_ on the dev set evolve during training. Results shown on GSM5.5K-Python dataset with GPT-Neo 125M model. Exponential moving average smoothing is applied for more clarity, but original curve is shown in shade. 

how our proposed self-sampling method can mitigate the overfitting problem, as it keeps improving or maintaining PASS@ _{_ 5 _,_ 10 _,_ 20 _}_ while such performances start decreasing for MLE. Though it also shows improvements for PASS@ _{_ 50 _,_ 100 _}_ , but the performance still decreases in later training stages. Here we can also see the importance of suitable learning objective, as MML has almost no effect in mitigating such overfitting issue. 

**Early stopping is needed when prioritizing high** _k_ **value for PASS@** _k_ **.** In our experiments, we select the model checkpoint with the best PASS@1 performance to evaluate all PASS@ _k_ . This setup aims to choose the best model that can solve the task with a small number of attempts (which corresponds to smaller _k_ value), as studied in (Austin et al., 2021). We can also observe that with our methods, the best PASS@1 checkpoint also yields the best or close to the best PASS@ _{_ 5 _,_ 10 _,_ 20 _}_ performances. However, in certain applications where large number of attempts are allowed, PASS@ _k_ with high _k_ values should be prioritized. An example is to generate candidate solutions before reranking (Cobbe et al., 2021). In this case, an earlier checkpoint ( _e.g.,_ one with best PASS@100) should be used instead, which is not the best checkpoint for PASS@ _k_ where _k_ is small. Also note that our proposed method are not suitable for these applications, as we observe no improvement on the peak PASS@ _{_ 50 _,_ 100 _}_ performances. We think this because when such peak performance is reached, it is still in the early stage of training thus not many FCSs or PCSs have been saved in the buffer yet. 

**Partially-correct solutions help in early training stages.** To show how self-sampling effects training, in Fig. 7a we show how the size of the buffer progresses during training. From the curves, we can see that in the early training stages ( _i.e.,_ first 5k steps), the number of saved PCSs rapidly grows while the number of FCSs only slightly increases. In later stages of training, the growth of buffer size is mainly contributed by more FCSs being sampled and saved while the number of 

16 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0017-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0017-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions_images/iclr_2023_Learning_Math_Reasoning_from_Self-Sampled_Correct_and_Partially-Correct_Solutions.pdf-0017-03.png)


**----- Start of picture text -----**<br>
(a) Growth of the number of saved FCS and PCS dur- (b) Distribution of the characterization of self-sampled<br>ing training. # FCS includes the gold solutions. solutions during training.<br>**----- End of picture text -----**<br>


Figure 7: How self-sampling evolves throughout the training process. Results shown as training the GPT-Neo 125M model on the GSM5.5K-Python dataset with MLE-Aug loss. 

PCSs stays steady. Also when compared to learning only with FCSs, learning with FCSs + PCSs eventually accumulates more FCSs in the buffer (green dotted line vs yellow dotted line). In addition, we show how the distribution of the outcomes of self-sampled solutions changes throughout training in Fig. 7b. We can see that in the early training stages, the ratio of not executable/incorrect solutions quickly drops to almost zero. At the same time, the ratio of new FCS or PCS being saved reaches the peak. As training proceeds, the models are mostly sampling known FCS or PCS as the size of the buffer converges as well. But the number of self-sampled fully-correct solutions gradually overtakes the partially-correct ones. 

17 

