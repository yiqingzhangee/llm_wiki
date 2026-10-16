# **One Prompt is not Enough: Automated Construction of a Mixture-of-Expert Prompts** 

**Ruochen Wang**[* 1] **Sohyun An**[* 2] **Minhao Cheng**[3] **Tianyi Zhou**[4] **Sung Ju Hwang**[2] **Cho-Jui Hsieh**[1] https://github.com/turningpoint-ai/mixture-of-prompts 

## **Abstract** 

Large Language Models (LLMs) exhibit strong generalization capabilities to novel tasks when prompted with language instructions and incontext demos. Since this ability sensitively depends on the quality of prompts, various methods have been explored to automate the instruction design. While these methods demonstrated promising results, they also restricted the searched prompt to one instruction. Such simplification significantly limits their capacity, as a single demofree instruction might not be able to cover the entire complex problem space of the targeted task. To alleviate this issue, we adopt the Mixture-ofExpert paradigm and divide the problem space into a set of sub-regions; Each sub-region is governed by a specialized expert, equipped with both an instruction and a set of demos. A two-phase process is developed to construct the specialized expert for each region: (1) _demo assignment_ : Inspired by the theoretical connection between incontext learning and kernel regression, we group 

> *Equal contribution 1University of California, Los Angeles 2Korea Advanced Institute of Science & Technology 3Penn State University[4] University of Maryland, College Park. Correspondence to: Cho-Jui Hsieh _<_ chohsieh@cs.ucla.edu _>_ . _Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 

demos into experts based on their semantic similarity; (2) _instruction assignment_ : A region-based joint search of an instruction per expert complements the demos assigned to it, yielding a synergistic effect. The resulting method, codenamed Mixture-of-Prompts (MoP), achieves an average win rate of 81% against prior arts across several major benchmarks. 

## **1. Introduction** 

Recent advancements in large language models (LLMs) have demonstrated a remarkable ability to solve novel tasks described by user instructions (Ouyang et al., 2022; OpenAI, 2023; Touvron et al., 2023; Peters et al., 2018; Devlin et al., 2018; Brown et al., 2020; Wei et al., 2022b). Despite the success, there still exists a substantial gap between user intention and the model’s interpretation. Therefore, carefully designed prompts (a.k.a. Prompt Engineering) become an essential ingredient for fully eliciting LLM’s superior generalization ability (Alhoshan et al., 2022; Zhao et al., 2021; Liu et al., 2021; Lu et al., 2021; Su et al., 2022; Wang et al., 2022a; Wei et al., 2022a; Yao et al., 2023; Schick et al., 2023; Kojima et al.). However, it usually requires laborious efforts through inefficient trial and error. To reduce human efforts, several recent attempts have shown tremendous potential in utilizing LLMs themselves to design prompts for language generation (Zhou et al., 2022; Pryzant et al., 2023; Chen et al., 2023; Fernando et al., 2023; Yang et al., 2023; 

1 

**Mixture-of-Prompts** 

Xu et al., 2022). These methods are part of a broader conceptual framework termed **“LLM as Optimizers”** (Yang et al., 2023). While the results are promising, pioneering efforts along this line primarily focus on finding the optimal demo-free instruction for a specified task, based on a set of (input, output) demonstrations. While the prompts produced by these methods can outperform human-designed counterparts, a single demo-free instruction might not suffice to serve all the possible instances of a task or cover the whole problem space, limiting the LLM’s problem-solving potential. 

This paper aims to expand the problem space coverage for automatic prompting by optimizing a Mixture of Prompts (MoP). Our key insight is to adopt the **Mixture of Experts (MoE)** paradigm (Jacobs et al., 1991; Jordan & Jacobs, 1994) to partition the problem space into multiple homogeneous regions, each governed by **a specialized expert (prompt)** . At inference time, a single expert will be selected to prompt the LLM to answer a new input query. Under the MoE framework, prompt optimization reduces to an **expert assignment problem** that aims to search for the most suitable prompt for each expert, with the goal of **optimizing the performance of their mixture as a whole** . 

Another primary improvement proposed in this paper is to expand the prompt of each expert to contain both the instruction and demos, jointly optimized for each expert region in the problem space. Intuitively, concrete demos are good at providing fine-grained knowledge and expertise (local information) matching the details of input queries in a local region, whereas the instruction provides a generic ability and high-level guidance to solve a task (global information); Hence, they are complementary and together empower the experts to excel at their problem region. Motivated by this, we adopt a two-phase search algorithm that jointly optimizes a (demos, instruction) pair per expert: We first cluster all demos into different experts according to their semantic similarity, and then search for the best instruction complementary to each demo cluster of a prompt. For the first phase, i.e., demo assignment, we cluster the demos to multiple regions in a semantic embedding space by clustering algorithms. For the second phase, i.e., instruction assignment, we introduce a region-based joint search that finds the best instruction to complement the demos assigned to each expert. Given a new test query, we routine the expert containing the semantically closest demos to it. This method is inspired by the recently established theoretical connection between In-Context Learning and Kernel Regression (Han et al., 2023), which suggests that demos semantically closer to a test input in the LLM’s embedding space tends to perform better at inferring its answer. 

We scrutinize the proposed Mixture-of-Prompts (MoP) through extensive empirical study. Our key findings can be summarized as follows: (1) Clustering demos in the embedding space can effectively find semantically similar 

clusters that help allocate test samples accurately to the corresponding region and the optimal expert. (2) More experts are not necessarily better: there exists an optimal number of partitions for the problem space. (3) The optimal instruction for each demo cluster is often distinct, necessitating the joint search of demo and instructions. We further validate the strength of MoP across three major prompt optimization benchmarks: Instruction-Induction (Honovich et al., 2022), Super Natural Instructions (Wang et al., 2022b), and BIGBench-Hard (Suzgun et al., 2022). These benchmarks cover a wide range of possible tasks, including coding, math, common-sense reasoning, knowledge retrieval, etc. The results show that MoP surpasses six representative recent methods, achieving an average win rate of 81% across several major benchmarks. Our key contribution can be summarized as follows: 

- We propose a Mixture-of-Prompt (MoP), a Mixtureof-Expert framework that partitions the problem space into homogenous regions. 

- We extend each expert prompt to contain both instruction and demos, which expand the output space of prompt optimization. 

- Our empirical study with 50 tasks - one of the largest in prompt optimization literature - reveals that the proposed two-step search algorithm, which leverages semantic similarity for demo assignment and regionbased joint search for instruction assignment, achieves significant performance gains on major benchmarks. 

## **2. Related work** 

**Prompt optimization for language generation.** Aligning pretrained language models with human intentions is a crucial step toward unlocking their potential (Ouyang et al., 2022; Schick et al., 2023; Kojima et al.). An effective line of training-free alignment methods is prompt optimization (PO) (Shin et al., 2020; Zhou et al., 2022). PO originated from in-context learning (ICL) (Dale, 2021), which is mainly concerned with various designs and arrangements of in-context demonstrations (Wei et al., 2022a; Yao et al., 2023). It later evolves into automatic prompt engineering, where various discrete optimization algorithms are utilized to search for the best prompt (Shin et al., 2020; Deng et al., 2022; Zhang et al., 2022). With the emergence large language models (LLMs), there has been a paradigm shift towards leveraging these models for optimizing prompts in a manner akin to human writers (Zhou et al., 2022; Pryzant et al., 2023; Xu et al., 2022; Yang et al., 2023; Chen et al., 2023; Fernando et al., 2023). Our research builds on this recent advancement as these method yields strong results and offers a more interpretable optimization process. 

**Mixture of Experts Paradigm.** Mixture of Experts (Jacobs et al., 1991; Jordan & Jacobs, 1994) is a classic paradigm of longstanding interest within the machine learn- 

2 

**Mixture-of-Prompts** 

ing community. MoE structure was originally studied based on traditional machine learning models (Jordan et al., 1996; Collobert et al., 2001). Subsequently, it was extended to deep neural networks by (Eigen et al., 2013) to enhance its capacity to handle complex vision and speech problems. Following this development, there has been a proliferation of MoE layers integrated with various base neural network structures (Shazeer et al., 2017; Dauphin et al., 2017; Vaswani et al., 2017), leading to significant accomplishments in a wide range of language-related tasks. In recent years, efforts to combine the MoE layer with various base network architectures have demonstrated remarkable successes in modeling natural languages. Our work extends this high-level paradigm developed in the architectural domain to the prompt optimization task, where each expert is defined as a specialized prompt. 

## **3. Preliminaries** 

**Terminology.** We start by introducing key terminologies that will be used throughout the paper. We define a **Prompt** as the entire text preceding the question. We consider the setting where a prompt can be divided into two parts: (1) **Instruction** : a set of natural language sentences describing the task, and (2) **Demos** : a set of input-output pairs structured in a specific way to demonstrate how to solve a task. Below is an example prompt under this definition: 

_Prompt =”Find the opposite words of the input. Input: Similar Output: Dissimilar ...”_ 

Mathematically, a prompt ( _P_ ) can be represented as follows (Xie et al., 2021): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0003-06.png)


Here, _I_ represents an instruction, _{_ ( _xi, yi_ ) _}[n] i_ =1[represents] in-context demos, which is the set of (input, output) pairs, and _o_[delim] represents delimiter token. 

**Prompt Optimization.** Recent efforts have demonstrated significant potential in automating the prompt engineering processes. Concretely, given a set of demos sampled from a task distribution _D_ , analog to the “training data” in supervised learning, a prompt optimization aims at finding an Instruction (demo-free) that minimizes the empirical risk (or maximizes a score): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0003-09.png)


where _f_ ( _·_ ) denotes some task-specific scoring function (Appendix I). After optimization, the best instruction can be used to predict new inputs in the following format: _P[∗]_ ( _x_ ) = [ _I[∗] , x_ ]. **Under the framework of Empirical Risk Minimization, one can deduce an underlying assumption that demos (training data) encapsulate all external information about the task.** 

**APE - Automatic Prompt Engineering.** The most relevant work to ours is APE (Zhou et al., 2022) - a pioneering method demonstrating that LLMs can be used to optimize prompt. The key idea of APE is to ask an LLM to induce candidate instructions by observing a subset of demos, randomly sampled from the entire training dataset, and pick the best one according to their rank on a held-out validation set (partitioned from training demos as well). Formally, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0003-12.png)


Here, _Mϕ_ denote an LLM, _{I[j] }[m] j_ =1[are the candidate in-] structions, and _T_ ( _D_[˜][train] ) represents the chosen template format (see Figure 6). Subsequently, the best instruction among the candidate pool is selected based on the validation accuracy: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0003-14.png)


We refer the reader to Appendix K.1 for more details. 

**Limitations of APE.** While methods like APE demonstrated promising results in designing prompts that surpass human engineers, they are still constrained to searching within a _single demo-free_ instruction space. Such a limitation can hinder the problem-solving potential in NLP tasks, where the complexity of problems may not be adequately addressed by a _single demo-free_ instruction alone. 

## **4. MoP: Mixture-of-Prompts** 

## **4.1. Framework Overview** 

**Mixture-of-Expert for prompt optimization.** To address the aforementioned issue of existing automatic prompt engineering methods, we expand the problem space coverage for automatic prompt engineering by optimizing the Mixture of Prompts (MoP). To achieve this, we employ the Mixture of Experts (MoE) paradigm (Jacobs et al., 1991; Jordan & Jacobs, 1994) to partition the entire problem space into _C_ regions, each governed by a specialized expert. Within the MoE framework, prompt optimization (Equation (2)) transforms into an **expert assignment problem** that aims to search for the most suitable prompt _Pc[∗]_[for each expert,] with the ultimate goal of optimizing the performance of the entire mixture: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0003-20.png)


Here, ( _xc, yc_ ) _∼Vc_ refers to the data point assigned to expert _c_ by the employed routing function during inference time (we explain it in more detail later in Section 4.2). Notably, our MoP framework expands the prompt for each 

3 

**Mixture-of-Prompts** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0004-01.png)


_Figure 1._ **Illustration of MoP.** We adopt the MoE paradigm and divide the problem space into a set of sub-regions. Each sub-region is governed by a specialized expert, equipped with both an instruction and a set of demos. A two-phase process is developed to construct the specialized expert for each region: (1) _demo assignment_ : Inspired by the theoretical connection between ICL and kernel regression, we group demos into experts based on their semantic similarity; (2) _instruction assignment_ : A region-based joint search (RBJS) of an instruction per expert complements the demos assigned to it, yielding a synergistic effect. During inference, each new query is routed to its closest expert in the embedding space and the assigned expert then utilizes its prompt (instruction + demos) to make the final prediction. 

expert to contain both the instruction and demos jointly optimized for each expert region in the problem space; _Pc_ ( _xc_ ) = � _Ic, Vc_[train] _, xc_ � in Equation (5). Intuitively, concrete demos excel at defining fine-grained details and expertise (local information) matching the queries in a local region, whereas instructions provide general abilities and high-level explanations for solving tasks (global information). Inspired by this, we introduce a **two-phase search algorithm** that jointly optimizes (demos, instructions) pairs for each expert (detail in Section 4.2 and 4.3). 

Theorem 4.1 (Han et al., 2023) suggests that, when LLM is prompted with a set of demos and a new test query ( _x_[test] ), demos that are semantically closer to the test example in embedding space contribute more to its prediction. The same phenomenon has been observed by several empirical studies (Rubin et al., 2021; Han et al., 2023; Liu et al., 2021). This behavior is also intuitive for ICL: the whole purpose of providing demos is for LLMs to leverage and apply their patterns to the test input. 

## **Clustering demos to each expert based on their semantic** 

## **4.2. Demo Assignment** 

In our two-phase search algorithm, we initiate the process by assigning training demos to different experts. Since demos represent local expertise, their assignment defines the design of experts in MoE and is entirely up to the constructors. While there are many options, we propose a clustering-based demo assignment method, derived from a recent theory of In-Context Learning in LLMs. 

**LLM learns from demos via Kernel Regression in the embedding space.** Recently, Han et al. (2023) provides the first theoretical result showing that LLM performs InContext Learning (ICL) from demos as Kernel Regression in the embedding space. Formally: 

**Theorem 4.1.** _Let {_ ( _xi, yi_ ) _}[n] i_ =1 _[denote the demos used in] the prompt; Let K define a kernel function that measures the semantic similarity between two data points, which can be represented as K_ ( _xi, xj_ ) = _ϕ_ ( _xi_ ) _[T] ϕ_ ( _xj_ ) _with some embedding space ϕ_ ( _·_ ) _. Then the output of LLM, P_ ( _y|_ [ _Sn, xtest_ ]) _, converges polynomially to the following Kernel Regression model with probability_ 1 _− δ._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0004-10.png)


**similarity.** The above analysis motivates a natural way of assigning demos to different experts: by clustering them based on semantic similarity. Starting from the kernel model in Theorem 4.1, our goal is to divide the demos into _C_ groups _{V_ 1 _, . . . , VC}_ such that each group (expert) only uses its own demo. In this case, the same sample _xi_ ’s prediction, assuming its in group _c_ , will become 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0004-12.png)


¯ ˆ and the error _|yi − yi|_ is related to the sum of the kernel entries outside the cluster[�][Therefore, a] _j /∈Vc[K]_[(] _[x][i][, x][j]_[)][.] good demo assignment algorithm will minimize the sum of between-cluster kernel values while keeping the clusters balanced, leading to the following clustering objective: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0004-14.png)


Based on the derivation in Appendix F, this is equivalent to the following clustering objective: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0004-16.png)


4 

**Mixture-of-Prompts** 

which is exactly the objective function of K-means clustering in the embedding space _ϕ_ ( _·_ ). In practice, we assume _ϕ_ ( _·_ ) := _Eθ_ ( _·_ ) is a mapping formed by a neural network encoder, and conduct K-means in such embedding space to cluster demos. Note that the choice of embedding space does not have to be the same as the API model; as long as the embedding space reflects the high-level semantic similarity between different demos, it can be used to effectively partition the problem space. We also compare other options in the ablation study. 

**On the choice of clustering algorithm.** In principle, our demo assignment method allows any clustering algorithm to be applied. In practice, we resort to the widely adopted K-means family for their simplicity, and made the following changes to better suit our application: 1) We select the K- means-auto variant as it can infer the optimal number of experts from the data. 2) To avoid biasing towards a larger number of clusters, we employ scaled inertia (Equation (10)) as the criterion when identifying the optimal number of experts. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0005-03.png)


**Routing function.** During inference time, each new query _x_ will be routed to its closest expert in the embedding space. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0005-05.png)


Here, _**µ** c_ is the clustering centroids for each expert. The assigned expert _c_ ( _x_ ) will then use its prompt (instruction + demos) to make the final prediction. 

## **4.3. Instruction Assignment** 

Given the set of demos assigned to each expert, the final step is to identify the best instruction for each cluster, so that the collective performance of the mixture is maximized. We introduce a **Region-Based Joint Search (RBJS)** algorithm for solving this objective. RBJS consists of two parts: generating candidate instructions and identifying the best one for each expert. 

**Generating candidate instructions to complement the demos.** As discussed in Section 4.1, each expert acquires a different specialty from the local information stored in their assigned demos. Because of this, they also process distinct blind spots in terms of their general task-solving ability. Therefore, they might require different instructions to compensate for their special needs. Inspired by this analysis, for each expert, we propose to generate candidate instructions 

utilizing the demos assigned to other experts. This way, the instruction can potentially capture the missed information. 

This choice is also supported by the empirical risk minimization framework, as outlined in Section 3. For prompt optimization, ’demons’ — analogous to training data in supervised learning — incorporate all task-relevant external information accessible to the model. Therefore, utilizing each expert’s local demonstrations for instruction generation merely duplicates the existing information. This contradicts our goal of creating instructions that compensate for the unique specialties of each expert. 

Any existing instruction generation algorithm can be used to propose candidate instructions given a set of demos. In this work, we choose APE for its simplicity. 

**Identifying the best candidate for each expert.** To select the best instruction from a set of proposals, existing prompt optimization algorithms commonly rank their performance on a held-out validation set, sampled from the same distribution as the training demos. Using the entire validation set measures how well an expert (instruction, demos) performs on the full data distribution. However, this might not serve our purpose: During inference, each expert is only responsible for predicting the data within their region. Our empirical results also support this analysis; we find that the performance of an expert between the full and local data distribution is not necessarily aligned (Figure 2d). 

To alleviate the issue in an exhaustive search, we first route each input in the validation set to its experts, then perform a joint search on the optimal (instruction, demos) pair. 

Algorithm 1 in the appendix summarizes the entire search process of the Region-Based Joint Search algorithm. 

## **5. Experiments** 

In this section, we experimentally validate MoP, which jointly searches for the optimal (instruction, demos) pair to partition the problem space into homogeneous regions. 

## **5.1. Experimental Setup** 

**Settings.** We follow the settings in the original APE paper (Zhou et al., 2022) with the following exceptions. (1) Our evaluation is conducted on OpenAI’s latest GPT-3.5Turbo-Instruct model[1] , a cost-efficient (100 _×_ cheaper) replacement for the text-davinci model used in APE. We reran APE on this model. (2) For all our methods, we report the mean and standard deviation across 3 runs to account for the randomness in the search phase. 

**Tasks and Evaluation Metrics.** We empirically validate the strength of MoP across three major prompt optimiza- 

1https://platform.openai.com/docs/models/gpt-3-5 

5 

**Mixture-of-Prompts** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0006-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0006-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0006-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0006-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0006-05.png)


**----- Start of picture text -----**<br>
(a) Visualization of demo clusters (b) Experts’ different strengths (c) Necessity of Joint Search (d) Necessity of RBJS<br>**----- End of picture text -----**<br>


_Figure 2._ **Analysis. (a)** There exist underlying patterns in the data distribution, and demos with semantically similar meanings are grouped closely. The circle, triangle, and star shapes represent training, routed validation, and routed test demos, respectively. **(b)** Each expert has distinct task-solving ability for each input. **(c)** A single instruction is insufficient for all experts, highlighting the need for distinct synergistic instructions for each expert. **(d)** Performance of instructions evaluated under local data distribution for each expert (i.e. subsets routed to each expert) is not aligned with the full data; This motivates performing region-based evaluation during joint search (RBJS). 

tion benchmarks: **Instruction Induction** (Honovich et al., 2022), **Super Natural Instructions** (Wang et al., 2022b), **BIG-Bench-Hard** (Suzgun et al., 2022). These benchmarks cover a wide range of possible tasks, including coding, math, common-sense reasoning, knowledge retrieval, etc. We provide detailed descriptions of each task in the Appendix H. For these benchmarks, we measure task performance for each task using predefined score functions (details in Appendix I). 

**Baselines.** We compare our method against following baselines: **APE** (Zhou et al., 2022) searches for a single instruction among a pool of instruction candidates proposed by a LLM, **APE+Demos** combines randomly selected demos with an APE-found instruction, **APE+K-centroids** combines demos corresponding to the centroids of K-means with an APE-found instruction, **InstructZero; IZ** (Chen et al., 2023) finds a single instruction for a black-box LLM by optimizing the soft prompt of an open-source LLM using a Bayesian Optimization approach, and **IZ+Demos** and **IZ+K-centroids** are the same as the previous APE variants. For more details, refer to Appendix K. We defer more baselines that partially use our method to the ablation study in Section 6. 

**Implementation Details.** Following the previous automatic prompting works (Zhou et al., 2022), we set the temperature to 0.9 when generating instructions using LLMs to encourage diversity and to 0.0 when evaluating with LLMs. Furthermore, for a fair comparison, **we set the same budget for all methods** . Regarding demo assignments in MoP, we use the default hyperparameter consistently across all experiments. More details can be found in Appendix L. 

## **5.2. Analysis** 

Before delving into the main experiments, we conduct an empirical analysis that motivates the development of our MoP framework. The results presented here are obtained 

for the Auto categorization task, with the maximum number of experts set to four. 

**Visualization of demo clusters.** Building upon the theoretical connection between ICL and Kernel Regression, we begin by clustering a given set of demos into regions based on their semantic similarity. To achieve this, we first map the given demo sets into the embedding space using text-embedding-ada-002 model[2] as a text encoder ( _Eθ_ ), and then apply the clustering algorithm described in Section 4.2. Figure 2a visualizes clustering in the embedding space with t-SNE projection. The illustration indicates that **there exist underlying patterns in the data distribution, and demos with semantically similar meanings are grouped closely** . For example, for the ’Auto categorization’ task shown in Figure 2a, demos relevant to country, computer science, extinct languages, and apparel are each clustered together. By clustering demos in the embedding space, we can effectively find semantically similar clusters that help allocate test queries (marked with stars Figure 2a) accurately to the corresponding region and the optimal expert. 

**Experts process different specialties.** We then verify the impact of demo clusters on performance for each test query. In order to eliminate the impact of instructions on performance, all experts utilize only clustered demos as prompts, thereby restricting the output space to demos only. Subsequently, we calculate the Hit Ratio by counting the number of correctly answered experts out of the total number of experts ( _C_ ) for each test input. If test inputs yield Hit Ratios within the range other than 0/ _C_ and _C_ / _C_ , it indicates their sensitivity to the assigned expert. As depicted in Figure 2b, we measure the Hit Ratios and observe that 83% of test inputs have Hit Ratio values that are neither 0 nor 1. This implies that most test inputs are influenced by the type of clustered demos they are assigned; **Each expert develops** 

> 2https://openai.com/blog/new-and-improved-embeddingmodel 

6 

**Mixture-of-Prompts** 

## **distinct specialties based on the local information within their assigned demos, resulting in distinct blind spots in terms of task-solving ability for each test input.** 

**Necessity of RBJS.** We also examine the necessity of jointly searching for the optimal (instructions, demos) pair. To investigate this, we initially assign training demos to 4 different experts and consider 8 candidate instructions generated by a LLM. Then, for each expert, we vary the instructions and measure the test performance of prompts by combining these instructions with the demos specific to each expert. Figure 2c visualizes the ranks of the candidate instructions for each individual expert. As depicted in Figure 2c, the rankings of each candidate instruction vary significantly across different experts. These results indicate that **a single instruction is insufficient for all experts, highlighting the need for distinct synergistic instructions for each expert** . This emphasizes the importance of a joint optimization scheme for the (instruction, demos) pair, which can lead to improved results. Furthermore, for each region, we calculate the correlation between 1) the validation rankings of candidate instructions obtained from a random subset of the full validation set (Joint Search) and 2) the rankings obtained when using an equal-sized routed local validation set within the target region (RBJS). As illustrated in Figure 2d, it is evident that **there is a misalignment in the validation rankings of candidate instructions between the use of a random subset of the full dataset and the utilization of routed local data** . We defer the evaluation of the RBJS’s effectiveness to an ablation study in Section 6.5. 

## **5.3. Main Results** 

In this section, we conduct a large-scale experiment to compare MoP against six previous SOTAs across three major benchmarks: Instrucion-Induction (Zhou et al., 2022), Super Natural Instruction (Wang et al., 2022b), and BIG-BenchHard (Suzgun et al., 2022). Tasks from these benchmarks cover a broad spectrum of language understanding scenarios, including various types of reasoning, knowledge retrieval, coding, math, etc. Due to space limit, we only showcase 19 tasks here, and include all results in Appendix J. As shown in Figure 3, MoP outperforms prior arts (APE + Demos and IZ + Demos) by a substantial margin. 

In addition, we also compute the win rate of every pair of methods. **As shown in Figure 4, the win rate of MoP dominates all six prior methods, with an average of 81% across three benchmarks. The results not only reveal the strength of our framework but also demonstrate that MoP can generalize to a wide range of tasks.** 

## **6. Ablation Study** 

In this section, we ablate the effect of different modules in the proposed MoP framework. We conduct the experiments 

on three tasks: Auto categorization, Mathdataset classification, and Taxonomy animal the task throughout the section. All other settings are identical to the previous section. 

## **6.1. Robustness of MoP on Out-of-Distribution Data** 

_Table 1._ **Comparison on Out-of-Distribution data.** We compare different methods on Out-of-Distribution data, created adversarially using the embedding model (same as MoP) to group the original dataset into two clusters, one for training and one for testing. Overall, MoP achieves the best robustness under this setting. 

|**OOD Data**|**Mathdataset**|**Taxonomy animal**|**Auto cate**|
|---|---|---|---|
|APE|9_±_11.0|**62**_±_**3.6**|37_±_4.1|
|APE + random|44_±_6.6|61_±_3.3|38_±_0.4|
|APE + kcen|40_±_11.8|60_±_1.0|44_±_2.2|
|IZ|25_±_26.0|55_±_5.8|44_±_10.1|
|IZ + random|45_±_9.9|60_±_0.6|43_±_0.4|
|IZ + kcen|48_±_11.7|61_±_1.4|44_±_1.2|
|MoP|**68**_±_**4.7**|60_±_1.5|**46**_±_**1.8**|



To further assess the robustness of our method under OutOf-Distribution (OOD) settings, we craft OOD datasets that can challenge MoP: Using the same embedding model as MoP, we divided the original dataset into two clusters: one designated for training and the other for testing. This division ensured that all test data were significantly distant from the training clusters, providing a rigorous test of MoP’s demonstration assignment and routing functions — arguably a more adversarial setup than that faced by APE. 

The results in Table 6.1 reveal several insights. Firstly, all methods exhibited a performance drop on the OOD dataset. This aligns with the principle of empirical risk minimization, where optimization is strictly confined to the information provided by the training data. Secondly, MoP consistently outperforms other baselines. 

The resilience of MoP can be attributed to its strategic segmentation of the problem space. By dividing the space into distinct regions, each managed by an expert, MoP ensures that even an OOD query is matched to the closest region. This reduces the ”out-of-distribution” effect for the query relative to the localized data distribution, making the query effectively less alien to the selected expert region. 

## **6.2. Different Number of Demos** 

Firstly, we verify the performance of MoP against baselines across different numbers of demos. As shown in Figure 5, MoP consistently outperforms the baselines across various numbers of demos, achieving a significant improvement of 16.89%, 22.89%, 2.78% compared to APE+Demos and 21.67%, 19.88%, 3.33% compared to IZ+Demos across all the tasks considered for each number of demos. 

## **6.3. Different Clustering Algorithm** 

We use K-means-Auto to cluster demos, which automatically decides the best number of experts. The intuition 

7 

**Mixture-of-Prompts** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0008-01.png)


_Figure 3._ **Main results.** We validate MoP across three major prompt optimization benchmarks. MoP achieves an average performance of **52.73%** outperforming the average performance of 41.39% / 39.87% achieved by APE+Demos / IZ+Demos in these results. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0008-03.png)


**----- Start of picture text -----**<br>
100 100<br>APE 0 23.5 22.2 60 37.5 42.1 22.2 APE 0 0 5.9 40 15.8 16.7 0 APE 0 50 33.3 70 40 33.3 22.2<br>80<br>APE+Demos 76.5 0 50 85 75 71.4 38.5 APE+Demos 100 0 55.6 100 71.4 73.7 31.2 80 APE+Demos 50 0 37.5 55.6 28.6 28.6 37.5 80<br>APE+K-centroids 77.8 50 0 95 62.5 69.2 25 60 APE+K-centroids 94.1 44.4 0 94.7 68.8 64.3 11.8 60 APE+K-centroids 66.7 62.5 0 80 66.7 37.5 30 60<br>IZ 40 15 5 0 18.2 10 9.1 IZ 60 0 5.3 0 5 10 5 IZ 30 44.4 20 0 40 30 0<br>IZ+Demos 62.5 25 37.5 81.8 0 46.7 7.1 40 IZ+Demos 84.2 28.6 31.2 95 0 46.7 27.8 40 IZ+Demos 60 71.4 33.3 60 0 25 30 40<br>IZ+K-centroids 57.9 28.6 30.8 90 53.3 0 7.1 20 IZ+K-centroids 83.3 26.3 35.7 90 53.3 0 12.5 20 IZ+K-centroids 66.7 71.4 62.5 70 75 0 25 20<br>MoP(ours) 77.8 61.5 75 90.9 92.9 92.9 0 MoP(ours) 100 68.8 88.2 95 72.2 87.5 0 MoP(ours) 77.8 62.5 70 100 70 75 0<br>0 0 0<br>(a) Instruction Induction (b) Super Natural Instructions (c) BIG-Bench-Hard<br>APE APE+Demos APE+K-centroids IZ IZ+Demos IZ+K-centroids MoP(ours) APE APE+Demos APE+K-centroids IZ IZ+Demos IZ+K-centroids MoP(ours) APE APE+Demos APE+K-centroids IZ IZ+Demos IZ+K-centroids MoP(ours)<br>**----- End of picture text -----**<br>


_Figure 4._ **Win rate matrices.** We compare the pairwise win rate of all methods on Instruction Induction ( **a** ), SuperNI ( **b** ), and BIG-BenchHard ( **c** ). Our method achieves the best win rate against all six baselines across various benchmarks. The average win rate of MoP across all benchmarks is 81%. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0008-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0008-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0008-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0008-08.png)


**----- Start of picture text -----**<br>
(a) Auto categorization (b) Mathdataset classification (c) Taxonomy animal<br>**----- End of picture text -----**<br>


_Figure 5._ **Ablation study on different number of demos.** We measure the task performance of each method across different numbers of demos. Here, _N_ on the x-axis represents the total number of training demos. 

behind it is that more experts do not necessarily produce the best result. Here, we validate this choice by comparing the performance of K-Means-Balanced and K-means-Auto. 

As shown in Table 2, both K-Means variants outperform random, while K-Means-Auto achieves the best results. 

8 

**Mixture-of-Prompts** 

## **6.4. Different Embedding Model** 

During demo assignment, we measure the semantic similarity of demos using _l_ 2 distance in the embedding space. While our method is agnostic to the specific choice of embedding models, stronger text encoders perform better in identifying semantically similar demos. Here we examine how the strength of the embedding model affects the performance of MoP. We examine three commonly used text encoders: GPT2-Large, T5, and Ada-002. The results in Table 2’s 1st group suggests that, while Ada achieves the best result, MoP operates reasonably well with paried with GPT2-Large. This shows that the proposed demo assignment method does not rely on a specific embedding model. 

## **6.5. Different Prompt Assignment Algorithms** 

We further ablate the key elements behind the design of our Region-Based Joint Search algorithm. 1). The optimal instruction for each expert might be distinct, therefore their generation and assignment should be conditioned on the demo clusters (Joint Search); 2). To find instructions that compensate for each expert’s demos, we use demos from all other experts to generate instructions. 3). The optimal prompt for each expert is evaluated only on the text points assigned to this expert (Region-based). We designed three prompt assignment methods to validate these hypotheses respectively: 1). **Independent Search** searches for the best prompt and assigns demos independently, i.e. the global best prompt will be assigned to all experts; 2). **RBJS SameCluster** uses each expert’s own demos to generate prompts; 3). **Joint Search** ranks the best prompt on all validation data. The results in the bottom group of Table 2 confirm the claims: Region-Based Joint Search achieves the best results. 

## **6.6. Different Routing Algorithms** 

The routing function is crucial to the MoE framework, as it is responsible for assigning the test input to the most suitable experts. Following the clustering-based demo assignment, our routing function maps a test input to its closest expert in embedding space as well. As shown in Table 2, our routing function significantly outperforms random assignment. 

_Table 2._ **Ablation study.** The choice of models and algorithms in MoP is listed in the last row of every group. The best performance is achieved with K-means-Auto, text-embedding-ada, and RBJS. 

|**Embed Model**|**Mathdataset**|**Taxonomy animal**|**Auto cate**|
|---|---|---|---|
|GPT2-Large|**97**_±_**1.3**|**89**_±_**0.9**|53_±_0.9|
|Sentence-T5|92_±_2.3|76_±_3.9|53_±_5.3|
|Ada|95_±_3.3|**88**_±_**6.9**|**59**_±_**0.4**|
|**Clustering**|**Mathdataset**|**Taxonomy animal**|**Auto cate**|
|Random|89_±_1.3|74_±_5.7|36_±_0.8|
|K-means-Balanced|**96**_±_**0.8**|82_±_2.6|52_±_2.6|
|K-means-Auto|**95**_±_**1.7**|**88**_±_**6.9**|**59**_±_**4.2**|
|**Prompt Assignment**|**Mathdataset**|**Taxonomy animal**|**Auto cate**|
|Independent Search|94_±_1.5|71_±_6.3|52_±_3.4|
|Joint Search (JS)|93_±_1.7|82_±_9.0|56_±_2.9|
|RBJS Same-Cluster|91_±_1.4|72_±_1.0|**60**_±_**4.9**|
|RBJS (Ours)|**95**_±_**1.7**|**88**_±_**6.9**|**59**_±_**4.2**|
|**Routing Function**|**Mathdataset**|**Taxonomy animal**|**Auto cate**|
|Random|77_±_2.2|87_±_3.3|32_±_2.9|
|MoP|**95**_±_**1.7**|**88**_±_**6.9**|**59**_±_**4.2**|



assignment methods for constructing the expert committee. Equipped with the proposed similarity-based demo assignment and region-based demo-instruction joint search, MoP substantially improves the performance over comparable methods over a diverse set of NLP tasks. We hope the proposed method and associated findings could open up new possibilities for prompt optimization research. 

**Limitations** We include a discussion on the limitations of our method in Appendix M. 

## **Acknowledgements** 

The work is partially supported by NSF 2048280, 2331966, 2325121, 2244760, ONR N00014-23-1-2300, Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.RS-2019-II190075 Artificial Intelligence Graduate School Program(KAIST), and the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (No. RS-2023-00256259). 

## **Impact Statement** 

## **7. Conclusion** 

This work introduces the Mixture-of-Prompts (MoP) approach to enhance the performance of prompt optimization for Large Language Models. While existing methods search for a single instruction to prompt language models, MoP optimizes for a set of experts (prompts), each governing a specialized region of the problem space. This divide-andconquer approach reduces the complexity associated with the task assigned to a single prompt, thereby substantially enlarging the problem space coverage. Within MoP framework, we further investigate various demo and instruction 

In this work, we validate that our framework enhances the performance of prompt optimization for Large Language Models, by introducing the Mixture-of-Prompts, as shown in Section 5. Our approach potentially accelerates the effective utilization of Large Language Models, which have consistently showcased remarkable generalization capabilities across diverse tasks, contributing to promoting more equitable access to technological resources. Nevertheless, it is essential to acknowledge the possibility of our framework being misused for purposes such as generating misleading information or promoting unethical practices. We hope that our research will be applied responsibly and ethically. 

9 

**Mixture-of-Prompts** 

## **References** 

- Alhoshan, W., Zhao, L., Ferrari, A., and Letsholo, K. J. A zero-shot learning approach to classifying requirements: A preliminary study. In _International Working Conference on Requirements Engineering: Foundation for Software Quality_ , pp. 52–59. Springer, 2022. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33: 1877–1901, 2020. 

- Chen, L., Chen, J., Goldstein, T., Huang, H., and Zhou, T. Instructzero: Efficient instruction optimization for black-box large language models. _arXiv preprint arXiv:2306.03082_ , 2023. 

- Collobert, R., Bengio, S., and Bengio, Y. A parallel mixture of svms for very large scale problems. _Advances in Neural Information Processing Systems_ , 14, 2001. 

- Dale, R. Gpt-3: What’s it good for? _Natural Language Engineering_ , 27(1):113–118, 2021. 

- Dauphin, Y. N., Fan, A., Auli, M., and Grangier, D. Language modeling with gated convolutional networks. In _International conference on machine learning_ , pp. 933– 941. PMLR, 2017. 

- Deng, M., Wang, J., Hsieh, C.-P., Wang, Y., Guo, H., Shu, T., Song, M., Xing, E., and Hu, Z. Rlprompt: Optimizing discrete text prompts with reinforcement learning. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 3369–3391, 2022. 

- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018. 

- Eigen, D., Ranzato, M., and Sutskever, I. Learning factored representations in a deep mixture of experts. _arXiv preprint arXiv:1312.4314_ , 2013. 

- Fernando, C., Banarse, D., Michalewski, H., Osindero, S., and Rocktaschel, T.¨ Promptbreeder: Self-referential self-improvement via prompt evolution. _arXiv preprint arXiv:2309.16797_ , 2023. 

- Han, C., Wang, Z., Zhao, H., and Ji, H. In-context learning of large language models explained as kernel regression. _arXiv preprint arXiv:2305.12766_ , 2023. 

- Honovich, O., Shaham, U., Bowman, S. R., and Levy, O. Instruction induction: From few examples to natural language task descriptions. _arXiv preprint arXiv:2205.10782_ , 2022. 

- Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E. Adaptive mixtures of local experts. _Neural computation_ , 3(1):79–87, 1991. 

- Jordan, M., Ghahramani, Z., and Saul, L. Hidden markov decision trees. _Advances in neural information processing systems_ , 9, 1996. 

- Jordan, M. I. and Jacobs, R. A. Hierarchical mixtures of experts and the em algorithm. _Neural computation_ , 6(2): 181–214, 1994. 

- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners, 2022. _URL https://arxiv. org/abs/2205.11916_ . 

- Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., and Chen, W. What makes good in-context examples for gpt-3? _arXiv preprint arXiv:2101.06804_ , 2021. 

- Lu, Y., Bartolo, M., Moore, A., Riedel, S., and Stenetorp, P. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. _arXiv preprint arXiv:2104.08786_ , 2021. 

- OpenAI. Gpt-4 technical report. _ArXiv_ , abs/2303.08774, 2023. URL https://api.semanticscholar. org/CorpusID:257532815. 

- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems_ , 35:27730–27744, 2022. 

- Peters, M. E., Neumann, M., Zettlemoyer, L., and Yih, W.-t. Dissecting contextual word embeddings: Architecture and representation. _arXiv preprint arXiv:1808.08949_ , 2018. 

- Pryzant, R., Iter, D., Li, J., Lee, Y. T., Zhu, C., and Zeng, M. Automatic prompt optimization with” gradient descent” and beam search. _arXiv preprint arXiv:2305.03495_ , 2023. 

- Rubin, O., Herzig, J., and Berant, J. Learning to retrieve prompts for in-context learning. _arXiv preprint arXiv:2112.08633_ , 2021. 

- Schick, T., Dwivedi-Yu, J., Dess`ı, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., and Scialom, T. Toolformer: Language models can teach themselves to use tools. _arXiv preprint arXiv:2302.04761_ , 2023. 

- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. _arXiv preprint arXiv:1701.06538_ , 2017. 

10 

**Mixture-of-Prompts** 

- Shin, T., Razeghi, Y., Logan IV, R. L., Wallace, E., and Singh, S. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 4222– 4235, 2020. 

- Su, H., Kasai, J., Wu, C. H., Shi, W., Wang, T., Xin, J., Zhang, R., Ostendorf, M., Zettlemoyer, L., Smith, N. A., et al. Selective annotation makes language models better few-shot learners. _arXiv preprint arXiv:2209.01975_ , 2022. 

- Suzgun, M., Scales, N., Scharli,¨ N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., , and Wei, J. Challenging big-bench tasks and whether chain-of-thought can solve them. _arXiv preprint arXiv:2210.09261_ , 2022. 

- Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E.,` Azhar, F., et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022a. 

   - Xie, S. M., Raghunathan, A., Liang, P., and Ma, T. An explanation of in-context learning as implicit bayesian inference. _arXiv preprint arXiv:2111.02080_ , 2021. 

   - Xu, H., Chen, Y., Du, Y., Shao, N., Wang, Y., Li, H., and Yang, Z. Gps: Genetic prompt search for efficient fewshot learning. _arXiv preprint arXiv:2210.17041_ , 2022. 

   - Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., and Chen, X. Large language models as optimizers. _arXiv preprint arXiv:2309.03409_ , 2023. 

   - Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. _arXiv preprint arXiv:2305.10601_ , 2023. 

   - Zhang, T., Wang, X., Zhou, D., Schuurmans, D., and Gonzalez, J. E. Tempera: Test-time prompt editing via reinforcement learning. In _The Eleventh International Conference on Learning Representations_ , 2022. 

   - Zhao, Z., Wallace, E., Feng, S., Klein, D., and Singh, S. Calibrate before use: Improving few-shot performance of language models. In _International Conference on Machine Learning_ , pp. 12697–12706. PMLR, 2021. 

   - Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis, S., Chan, H., and Ba, J. Large language models are humanlevel prompt engineers. _arXiv preprint arXiv:2211.01910_ , 2022. 

- Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Arunkumar, A., Ashok, A., Dhanasekaran, A. S., Naik, A., Stap, D., et al. Supernaturalinstructions:generalization via declarative instructions on 1600+ tasks. In _EMNLP_ , 2022b. 

- Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Arunkumar, A., Ashok, A., Dhanasekaran, A. S., Naik, A., Stap, D., et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. _arXiv preprint arXiv:2204.07705_ , 2022c. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35: 24824–24837, 2022a. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35: 24824–24837, 2022b. 

11 

**Mixture-of-Prompts** 

## **Appendix** 

**Organization** The appendix file is organized as follows: 

- **Appendix A** - We provide comparison with additional baselines: APE-Nearest Neighbor and OPRO. 

- **Appendix B** - We provide the algorithm of the proposed approach, MoP. 

- **Appendix C** - We provide a theoretical connection between MoP and MoE. 

- **Appendix D** - We provide a comparison of the runtime associated with different methods. 

- **Appendix E** - We provide representative examples for both success and failure cases. 

- **Appendix F** - We provide the derivation of clustering objective. 

- **Appendix G** - We provide templates used in each scenario in our experiments. 

- **Appendix H** - We provide detailed descriptions for each task. 

- **Appendix I** - We provide an explanation of the metrics used for evaluating prompts. 

- **Appendix J** - We provide the entire results for the main experiment alongside Section 5.3. 

- **Appendix K** - We provide further descriptions of the baselines. 

- **Appendix L** - We provide additional descriptions of the implementation details for the experimental settings. 

- **Appendix M** - We conclude by outlining the limitations of our work. 

## **A. Comparison with additional baselines** 

_Table 3._ **Comparison with OPRO and APE-Nearest-Neighbor on BIG-Bench-Hard.** We report the execution accuracy gain (∆) of MoP from the baseline described in Section 5.1 on the BIG-Bench-Hard benchmark tasks. We run 3 experiments and provide both the mean and standard deviation values. Please note that due to the inherent randomness in the ChatGPT API, a performance gap of less than 1% between the two methods can be considered a tie. The number of demos is set to _N_ train _/_ 5, where _N_ train is the total number of training demos. 

|Task|APE-Nearest-Neighbor|OPRO|MoP|
|---|---|---|---|
|Causal judgement|59.93_±_0.50|43.09_±_5.85|60.99_±_2.19|
|Disambiguation QA|59.67_±_0.47|48.00_±_7.00|64.00_±_2.16|
|Dyck languages|14.00_±_2.83|0.00_±_0.00|17.33_±_2.49|
|Movie Recommendation|87.00_±_2.45|70.50_±_2.50|81.67_±_2.36|
|Navigate|47.33_±_2.05|59.00_±_4.00|54.00_±_4.08|
|Object counting|45.67_±_1.70|60.00_±_6.00|46.67_±_2.05|
|Ruin names|70.67_±_2.62|70.00_±_5.00|73.33_±_1.89|
|Snarks|56.55_±_5.22|48.31_±_3.37|55.81_±_4.53|
|Sports understanding|83.33_±_0.94|23.50_±_3.50|85.33_±_2.62|
|Word sorting|80.67_±_1.25|63.00_±_2.00|73.67_±_1.89|



We conducted further experiments to compare our method with two additional baselines: **(1) OPRO** (Yang et al., 2023) - a recently proposed genetic algorithm-based prompt optimization method. **(2) APE + Nearest Neighbor** : At test time, we select demonstrations based on their proximity to the test query. We focus on the Big-Bench-Hard for those experiments, as it contains some of the hardest tasks that can stress test how each algorithm handles complex problem spaces. For these experiments, we focus on the BBH benchmark, known for its challenging tasks, to assess how each algorithm performs in complex problem-solving scenarios. 

12 

**Mixture-of-Prompts** 

**OPRO** Since OPRO only provides implementation for BBH and its best results are obtained using powerful proprietary LLMs, we rerun OPRO with GPT-3.5-Turbo for fair comparison. As summarized in Table 3, our method (MoP) achieves an 80% win rate against this newer prompt optimization approach. 

**APE + Nearest Neighbor** We employ the same embedding model, text-embedding-ada-002, as used in MoP for the distance function in the nearest neighbor search. The results, presented in Table 3, show that MoP achieves a win rate of 70% over APE + Nearest Neighbor. This result provides further evidence of the necessity of jointly optimizing instructions and demonstrations: Since the Nearest Neighbor demo set can only be determined at inference time, it relies on an independently searched demo-free instruction (i.e., APE/IZ), which results in a suboptimal combination. In contrast, for MoP, prompts and demonstrations are jointly optimized for each expert, creating a coherent skill set. Each expert is initially assigned a fixed cluster of demonstrations; subsequently, the prompt assignment algorithm selects the best instruction for each expert separately, to enhance the utility of their specific demonstrations. 

## **B. Algorithms for MoP** 

## **Algorithm 1** Building MoP 

**Input:** Training demos _D_[train] = _{_ ( _xi, yi_ ) _}[N] i_ =1[train][, validation demos] _[ D]_[valid][=] _[ {]_[(] _[x][i][, y][i]_[)] _[}][N] i_ =1[ valid][, model] _[ M][ϕ]_[, text encoder] _[ E][θ]_[(] _[·]_[)][,] task-specific scoring function _f_ ( _·_ ) _→_ R. _▷ Demo Assignment with clustering algorithm described in Section 4.2_ . **Input:** _α_ in Equation (10), the minimum number of clusters _C_ min, the maximum number of clusters _C_ max Compute _**e**_[train] _i_ = _Eθ_ ( _x_[train] _i_ ) for _i_ = 1 _, . . . , N_ train Select the best _C_ ( _C[∗]_ ), which minimizes the scaled inertia score in Equation (10): Clustering _{_ _**e**_[train] _i }[N] i_ =1[train][into] _[ C][∗]_[clusters.] **Output:** Clustered demos _{V_ 1[train] _, . . . , VC_[train] _[∗][}] ▷ Construct the region-based validation subset, Vc[valid] ⊂D[valid] using a clustering-based routing function._ **Input:** _{V_ 1[train] _, . . . , VC_[train] _[∗][}] Vc_[valid] _←∅_ for _c_ = 1 _, . . . , C[∗]_ **for** _i_ = 1 **to** _N_ valid **do** _c_ ( _x_[valid] _i_ ) = argmin _c_ =1 _,...,C ∗ K_ � _ϕθ_ ( _x_[valid] _i_ ) _,_ _**µ** c_ � _. ▷_ Routing function in Equation (11) _Vc_[valid] _←Vc_[valid] _∪{_ ( _x_[valid] _i , yi_[valid] ) _}_ **end for Output:** Clustered validation demos _{V_ 1[valid] _, . . . , VC_[valid] _[∗][}] ▷ Instruction Assignment with Region-based Joint Search described in Section 4.3._ **Input:** _{V_ 1[train] _, . . . , VC_[train] _[∗][}]_[,] _[ {V]_ 1[valid] _, . . . , VC_[valid] _[∗][}]_ **for** _c_ = 1 **to** _C[∗]_ **do** Randomly sample a subset _D_[˜] _c_[train] _∼{V_ 1[train] _, . . . , VC_[train] _[∗][} \ {V] c_[train] _}_ , where _|D_[˜] _c_[train] _|_ = _r_ . Generate candidate instructions _{Ic[j][}][m] j_ =1 _[′]_[that complement the demos using a model] _[ M][ϕ]_ and a template format _T_ ( _D_[˜][train] ) given _D_[˜][train] . Evaluate the score on the region-based validation subset _Vc_[valid] : **end for** _Ic[∗]_[= argmax] _Ic[j]_[E][(] _[x,y]_[)] _[∼V] c_[valid] _f_ ([ _Ic[j][,][ V] c_[train] _, x_ ] _, y_ ) **Output:** _{Ic[∗][}][C] c_ =1 _[∗]_ **Output:** _{Pc[∗]_[(] _[x]_[)] _[}][C] c_ =1 _[∗]_[, where] _[ P][ ∗] c_[(] _[x]_[) = [] _[I] c[∗][,][V] c_[train] _, x_ ] 

## **C. Theoretical connection between MoP and MoE** 

Our work adapts the MoE framework, traditionally involving distinct models as experts, by defining experts as diverse prompts (instructions + demonstrations). We offer the following insights to highlight the duality between this application and traditional MoE. 

1. **Prompt Optimization can be viewed as model selection:** An LLM pre-trained on the next-token prediction task can be seen as a collection of conditional probabilistic models, each defined by a specific prompt. By crafting various prompts, LLM users are essentially picking different submodels to perform various tasks. Thus, designing varied 

13 

**Mixture-of-Prompts** 

prompts is equivalent to selecting different sub-models from this collection, making the process of automatically optimizing the prompt for each expert parallel to constructing a suitable model for each expert in traditional MoE. 

2. **Theoretical properties of MoE apply to MoP:** This aforementioned conceptual framework allows us to directly apply the theoretical properties of MoE to the task of prompt optimization for LLMs, as optimizing for a mixture of expert prompts is identical to optimizing a mixture of expert models. 

We view the main contribution of our work as the first to adapt the MoE framework to prompt optimization tasks and bring the community’s attention to its strong and consistent potential despite the choices of simple algorithms for each component (demo assignment, prompt assignment, and routing function). We hope this explanation better highlights the theoretical duality between MoP and MoE, and further motivates our work. 

## **D. Search cost comparison** 

_Table 4._ **Comparison of search and inference costs for different algorithms** . The results are measured on the Instruction Induction Benchmark, using the same hardware (1 _×_ 48G A6000) and API model (gpt-3.5-turbo-instruct). 

|**Method**|**Comment**|**Search cost(minute)**|**Inference cost(minute)**|
|---|---|---|---|
|APE + demos|random demos|0.22|0.013 / 100queries|
|IZ + demos|random demos|8.87|0.012 / 100queries|
|MoP(10 experts)|w/oparallelization|0.75|0.016 / 100queries|



We benchmark the runtime of the search and inference phase of MoP with different baselines. There exist two computational components in our method: between two types of computations: 1). query LLM 2). running the embedding model. Since the first former dominates latter in practice, we will focus on analyzing the complexity w.r.t. LLM queries: 

- **Inference:** MoP, operating under the Mixture of Experts (MoE) paradigm, deploys a single prompt akin to deploying a single instruction. This constitutes one key benefit of MoP (MoE) paradigm over prompt ensemble methods, or simply using longer prompts. 

- **Search:** The search in MoP involves negligible costs for the demo assignment phase as they do not require LLM querying. For the prompt assignment phase, the complexity is linear w.r.t. the number of experts but is fully parallelizable. 

Table 4 reports the wallclock times comparing MoP with APE and IZ. Our findings include: (1). gs include: (1) All methods exhibit similar inference times, which aligns with our previous analysis. (2). Both APE and MoP have substantially lower search costs compared to IZ, which incurs additional costs due to the local operation of an open-sourced LLM. (3). While MoP’s search cost (with 10 experts) approximately triples that of APE, this can be significantly reduced through parallelization. 

## **E. Qualitative analysis of the discovered experts** 

We provide an example analysis of the discovered experts, focusing on why MoP are more (less) effective for certain tasks. 

**Example success case:** An example where MoP significantly outperforms random demos is in the task of Autocategorization. This task uses training datasets with various categorization questions belonging to different genres, such as Country (e.g., countries with large populations, countries in the UN), Celebrity, Language, and Companies. We found that each identified expert specializes in one or two categories. For instance, one expert handles only celebrity-related queries, enhancing their ability to provide accurate answers. Another case where MoP excels is the Movie recommendation task from BBH. Here, each expert identified by the MoP algorithm focuses on a distinct set of movies. For example, expert 1 focuses on classic adventures in fantasy and whimsical settings, like ’The Wizard of Oz’ and ’Raiders of the Lost Ark’; while expert 5 handles movies that involve deep themes and complex stories, such as ’The Matrix’ and ’Schindler’s List’.” 

**Example failure case:** An example where MoP exhibits performance similar to random demonstrations is in the task ’Larger Animals’. In this task, MoP performs similarly to APE-Random, indicating that using multiple experts yields no 

14 

**Mixture-of-Prompts** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0015-01.png)


_Figure 6._ **The template used in our experiments.** 

additional benefit. Upon examining the identified experts, we find negligible differences in their specializations. This observation is intuitive, as this task involves selecting the largest animal from a randomly sampled list of animals of varying sizes; therefore, no specialized training data is necessary to successfully accomplish the task. 

## **F. Derivation of clustering objective** 

From (8) we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts_images/icml_2024_One_Prompt_is_not_Enough_Automated_Construction_of_a_Mixture-of-Expert_Prompts.pdf-0015-06.png)


## **G. Template used in our experiments** 

Referring to Zhou et al. (2022), we provide templates used in each scenario in our experiments. _Generating instructions_ refers to generating instructions, while _Evaluation_ denotes the inference time (validation or test phase). For the case of _Listing Demos_ , it refers to the template used when listing multiple demo samples. When a prompt is injected into the model, the _<_ COMPLETE _>_ part is removed, and the model generates an output. For a fair comparison, the same template was applied to all methods. 

15 

**Mixture-of-Prompts** 

_Table 5._ **Descriptions on Instruction Induction benchmark tasks.** Referring to Zhou et al. (2022), we provide task names, task summaries, and example demos within each task. 

|**Task**|**Task Summary**|**Demo**|
|---|---|---|
|Auto categorization|Categorize items based on a common theme or characteristic.|Python, Cobol, and C_→_programming languages|
|Rhymes|Write a word that rhymes with the input word.|sing_→_ring|
|Sentence similarity|Rate the semantic similarity of two input sentences on a|Sentence 1: A man is smoking. Sentence 2: A man is skating.|
||scale of 0 - defnitely not to 5 - perfectly.|_→_0 - defnitely not|
|Sentiment|Determine whether a movie review is positive or negative.|The flm is small in scope, yet perfectly formed. _→_positive|
|Word in context|Determine whether an input word has the same meaning in|Sentence 1: Approach a task. Sentence 2: To approach the|
||the two input sentences.|city. Word: approach_→_not the same|
|Larger animal|Write the larger of the two given animals.|koala, snail_→_koala|
|Informal to formal|Rephrase the sentence in formal language.|Please call once you get there _→_Please call upon your|
|||arrival.|
|Orthography starts with|Extract the words starting with a given letter from the input|The man whose car I hit last week sued me. [m]_→_man,|
||sentence.|me|
|Antonyms|Write a word that means the opposite of the input word.|won_→_lost|
|Second word letter|Extract the second letter of the input word.|cat_→_a|
|Common concept|Find a common characteristic for the given objects.|guitars, pendulums, neutrinos_→_involve oscillations|
|Cause and effect|Find which of the two given cause and effect sentences is|Sentence 1: The soda went fat. Sentence 2: The bottle was|
||the cause.|left open. _→_The bottle was left open.|
|Translation EN-FR|Translate the word into French.|time_→_temps|
|Diff|Subtract the second number from the frst.|32 22_→_10|
|First word letter|Extract the frst letter of the input word.|cat_→_c|
|Letters list|Break the input word into letters, separated by spaces.|cat_→_c a t|
|Taxonomy animal|Write all the animals that appear in the given list.|cat, helicopter, cook, whale, frog, lion _→_frog, cat, lion,|
|||whale|
|Negation|Negate the input sentence.|Time is fnite_→_Time is not fnite.|
|Num to verbal|Write the number in English words|26_→_twenty-six|
|Active to passive|Write the input sentence in passive form.|The artist introduced the scientist. _→_The scientist was|
|||introduced by the artist.|
|Singular to plural|Convert the input word to its plural form.|cat_→_cats|
|Sum|Sum the two given numbers.|22 10_→_32|
|Synonyms|Write a word with a similar meaning to the input word.|alleged_→_supposed|
|Translation EN-DE|Translate the word into German.|time_→_Zeit|
|Translation EN-ES|Translate the word into Spanish.|time_→_hora|
|Auto debugging|Produce a specifc result or output given the code.|import numpy as np_\_n x = numpy.zeros(10)_\_n_→_NameEr-|
|||ror: name ’numpy’ is not defned.|



## **H. Tasks** 

In this section, we provide detailed descriptions for each task across three benchmarks, encompassing a wide range of possible tasks, including coding, mathematics, common-sense reasoning, and knowledge retrieval: **Instruction Induction** (Table 5), **Super Natural Instructions for coding and mathematics** (Table 6 and Table 7), and **BIG-Bench-Hard** (Table 8). 

16 

**Mixture-of-Prompts** 

_Table 6._ **Descriptions on Super Natural Instructions benchmark code tasks.** Referring to Wang et al. (2022b), we provide task names, task summaries, and example demos within each task. 

|**Task**|**Task Summary**|**Demo**||
|---|---|---|---|
|Conala concat strings|Given a list of strings, concatenate them to form|[’s’, ’blew’, ’g’, ’and’, ’u’, ’as’, ’C’]_→_sblewganduasC||
||one string.|||
|Conala normalize lists|Given a list of numbers, normalize the list such|[-6.875, -64.545, -64.548]_→_[0.051 0.475 0.475]||
||that the result adds to 1.|||
|Conala calculate mean|Given a list of numbers, calculate the mean of|[140.719, 220.491, 119.072]_→_160.094||
||the list.|||
|Conala max absolute value|Given a list of numbers, calculate the element|[14.594 -85.985]_→_-85.985||
||with the largest absolute value.|||
|Conala list index subtraction|Given a list of numbers, subtract each element|[-14, 4]_→_[-15, 2]||
||by its index in the list.|||
|Conala remove duplicates|Given a list of numbers, remove all of the dupli-|[0, 0, 5, 7, 4, 3]_→_[5, 7, 4, 3]||
||cates in the list.|||
|Conala list intersection|Given a two lists of numbers, fnd the intersec-|[8, 10, 6, 2, 7] , [10, 4, 10, 10, 4]_→_[10]||
||tion of the two lists.|||
|Splash question to sql|Generate an SQL statement from a question ask-|What are the names of all cartoons directed by Ben Jones?||
||ing for certain data.|_→_SELECT Title FROM Cartoon WHERE Directed|~~b~~y =|
|||”Ben Jones”||
|Logic2text sentence generation|Generate a natural language interpretation of the|most<br>~~e~~q all<br>~~r~~ows ; venue ; london = true_→_for the venue||
||given logical operators.|records of all rows , most of them fuzzily match to london.||
|Conala list index addition|Add lists together based on their index.|[[69, 8, -40], [63, -57, 65]]_→_[132, -49, 25]||
|Conala sort dictionary|Sort a list of dictionaries based on a given key.|[’frst’: 47, ’second’: -34, ’frst’: 11, ’second’: 54]_→_[’frst’:||
|||11, ’second’: 54, ’frst’: 47, ’second’: -34]||
|Conala pair averages|Calculate the averages for each two consecutive|[47, 62, 2, -13]_→_[54.5, 32.0, -5.5]||
||elements.|||
|Conala pair differences|Calculate the absolute difference for each two|[-19, 40, 12, 95]_→_[59, 28, 83]||
||consecutive elements.|||
|English language answer relevance|Given a question and answer pair, detect whether|Question: Is it more correct to say a computer program is,||
|classifcation|the answer is acceptable or not.|_. . ._, Answer: I would say that neither,_. . ._. _→_no||
|Code x glue information retrieval|Given a code, calculate the number of for loops|int ways(int n,int p),_. . ._,_→_1||
||in the cpp program.|||



## **I. Score Functions** 

For the Instruction Induction benchmark tasks, we evaluate the quality of prompts using a metric called execution accuracy proposed by Honovich et al. (2022). The metric is defined as follows: For each (input, output) pair, if the model’s prediction matches the output exactly, it equals 1. If there is no perfect match, it equals 0. In certain tasks, a modified version of this metric is employed. For instance, it measures the proportion of correct answers within the total answer set. Please refer to Section 4.2 of Honovich et al. (2022) for further details. 

For the tasks in the Super Natural Instructions benchmark, we employ ROUGE-L scores as the evaluation metric, as outlined in Wang et al. (2022b). 

For the BIG-Bench-Hard benchmark task, we utilize execution accuracy as the evaluation metric, following Suzgun et al. (2022). 

## **J. Main Results** 

## **J.1. Results on Instruction Induction** 

We show the execution accuracy results for each method in the entire Instruction Induction benchmark (Honovich et al., 2022) tasks in Table 9, excluding the two tasks for which the dataset has not been made publicly available: ”Ascii” and ”Cs algorithms”. 

17 

**Mixture-of-Prompts** 

_Table 7._ **Descriptions on Super Natural Instructions benchmark mathematical tasks.** Referring to Wang et al. (2022b), we provide task names, task summaries, and example demos within each task. 

|**Task**|||**Task Summary**|**Demo**|
|---|---|---|---|---|
|Semeval 2019 task10 closed vocab-|||Answering multiple choices mathematical prob-|If (fracyy - 3 = frac4239), then what does y equal? (A) 39|
|ulary mathematical answer genera-|||lem described with a closed-vocabulary.|(B) 41 (C) 42 (D) 45 (E) 81_→_C|
|tion|||||
|Semeval 2019|task10 open vocab-||Answering multiple choices mathematical prob-|A new airplane can travel at speeds up to 4,680 miles per|
|ulary mathematical answer genera-|||lem described with an open vocabulary.|hour. How many miles can the airplane travel in 10 sec-|
|tion||||onds? (A) 1.3 (B) 7.8 (C) 13 (D) 78 (E) 130_→_C|
|Ai2 arithmetic questions arithmetic|||Given an arithmetic question, compute a solution.|Alyssa loves eating fruits. Alyssa paid $12.05 for grapes,|
|||||and $9.85 for cherries. In total, how much money did|
|||||Alyssa spend? _→_21.9|
|Aqua multiple choice answering|||Given a mathematical question, fnd the most suit-|Question: The sub-duplicate ratio of 16:64 is Option A: 4:3|
||||able numerical answer.|Option B: 1:2 Option C: 1:3 Option D: 1:4 Option E: 2:4|
|||||_→_Option E|
|Svamp subtraction question answer-|||Given a mathematical question involving subtrac-|Context: Baker sold 44 cakes. If he had made 48 cakes|
|ing|||tion, fnd the most suitable numerical answer.|initially Question: How many cakes would baker still have?|
|||||_→_4|
|Mathdataset classifcation|||Classify the type of a math word problem.|Solve 154 = -39*v - 41 for v. _→_algebra|
|Mathdataset answer generation|||Find the numerical answer for a math word prob-|Solve -38*s = -53*s - 90 for s. _→_-6|
||||lem.||
|Asdiv addsub question answering|||Given a mathematical question, fnd the most suit-|46 apples were in the basket. 22 are red and the rest are|
||||able numerical answer.|green. how many apples are green? _→_24|
|Asdiv multidiv|question answering||Given a mathematical question, fnd the most suit-|each bag contains 23 pounds of oranges. how many pounds|
||||able numerical answer.|of oranges are in 45 bags? _→_1035|
|Asdiv multiop question answering|||Given a mathematical question, fnd the most suit-|a mirror store has 78 mirrors in stock. 8 mirrors are broken|
||||able numerical answer.|and 57 mirrors are sold. how many mirrors are left? _→_13|
|Asdiv singleop|question answering||Given a mathematical question, fnd the most suit-|nick saved $68.50. if nick saved $25.43 more than lee how|
||||able numerical answer.|much did lee save? _→_43.07|
|Mawps addsub|question answering||Given a mathematical question, fnd the most suit-|Mark has 13 trees in his backyard. If he plants 12 more,|
||||able numerical answer.|how many trees will he have? _→_25|
|Mawps multidiv question answering|||Given a mathematical question, fnd the most suit-|A cereal box holds 18 cups of cereal. Each serving is 2|
||||able numerical answer.|cups. How many servings are in the whole box? _→_9|
|Mawps multiop question answering|||Given a mathematical question, fnd the most suit-|Paul had saved up 3 dollars. If he received another 7 dollars|
||||able numerical answer.|for his allowance, how many 5 dollar toys could he buy?|
|||||_→_2|
|Mawps singleop question answering|||Given a mathematical question, fnd the most suit-|Joan has 9 blue balloons but lost 2 of them. How many|
||||able numerical answer.|blue balloons does Joan have now? _→_7|
|Leetcode<br>420|strong|password|Check if the given password is strong|password = RtZGIgm7YeiPB66yVIoC_→_0|
|check|||||
|Mathqa gain|||Given a math problem on gain and options to|Problem: a 8% stock yields 20%. the market value of the|
||||choose from, fnd the correct option that answers|stock is : Options: a) rs 48 , b) rs 45 , c) rs 40 , d) rs 50 , e)|
||||the problem.|rs 55_→_c|
|Mathqa general|||Given a general math problem and options to|Problem: what is the remainder of w =319when divided|
||||choose from, fnd the correct option that answers|by 10? Options: a) 0 , b) 1 , c) 5 , d) 7 , e) 9_→_d|
||||the problem.||
|Mathqa other|||Given a math problem and options to choose from,|Problem: how many factors does352 have? Options: a) 2|
||||fnd the correct option that answers the problem.|, b) 8 , c) 24 , d) 25 , e) 26_→_c|
|Mathqa geometry|||Given a problem on geometry and options to|Problem: the surface of a cube is 24 sq cm . fnd its volume?|
||||choose from, fnd the correct option that answers|Options: a) 8 , b) 6 , c) 4 , d) 3 , e) 1_→_a|
||||the problem.||
|Mathqa probability|||Given a problem on probability and options to|Problem: two coins are tossed. fnd the probability of at|
||||choose from, fnd the correct option that answers|most 2 tails? Options: a) 1 / 2 , b) 1 / 4 , c) 1 / 3 , d) 1 , e) 3|
||||the problem.|/ 4_→_d|
|Mathqa answer|selection||Selecting answers to mathqa questions.|Problem: 1395 x 1395 Options: a. 1946025, b. 1981709, c.|
|||||18362619, d. 2031719, e. none of these_→_a|
|Mathqa correct|answer generation||Generate correct answers for math questions.|Problem: if 7 spiders make 7 webs in 7 days, then how|
|||||many days are needed for 1 spider to make 1 web? _→_7|



18 

**Mixture-of-Prompts** 

_Table 8._ **Descriptions on BIG-Bench-Hard benchmark code tasks.** Referring to Suzgun et al. (2022), we provide task names, task summaries, and example demos within each task. 

|**Task**|**Task Summary**|**Demo**|
|---|---|---|
|Causal judgement|Answer questions about causal attribution.|Frank T., had an ongoing dispute with his neighbor, _. . ._.|
|||Did Frank T. intentionally shoot his neighbor in the body?|
|||Options: - Yes - No_→_Yes|
|Disambiguation QA|Clarify the meaning of sentences with ambiguous pro-|Sentence: The scientist collaborated with the artist, and he|
||nouns.|shared a story. Options: (A) The scientist shared a story (B)|
|||The artist shared a story (C) Ambiguous_→_(C)|
|Dyck languages|Correctly close a Dyck-n word.|Input: (_{ { } } →_)|
|Movie Recommendation|Recommend movies similar to the given list of movies.|Find a movie similar to Forrest Gump, The Silence of the|
|||Lambs, Seven, Fargo: Options: (A) Gandhi (B) Schindler’s|
|||List (C) Dogfght (D) Repo Man_→_(B)|
|Navigate|Given a series of navigation instructions, determine|If you follow these instructions, do you return to the starting|
||whether one would end up back at the starting point.|point? Take 5 steps. Take 4 steps. Take 3 steps. Options: -|
|||Yes - No_→_No|
|Object Counting|Questions that involve enumerating objects of different|I have a piano, a fute, and four trombones. How many|
||types and asking the model to count them.|musical instruments do I have? _→_6|
|Ruin names|Select the humorous edit that ’ruins’ the input movie|Which of the following is a humorous edit of this artist or|
||or musical artist name.|movie name: ’bon iver’? Options: (A) bon liver (B) bion|
|||iver (C) ban iver (D) bon ivee_→_(A)|
|Snarks|Determine which of two sentences is sarcastic.|Which statement is sarcastic? Options: (A) He’s over six|
|||feet, so he must be tall (B) He’s over six feet, so he must be|
|||wonderful_→_(B)|
|Sports understanding|Determine whether an artifcially constructed sentence|Is the following sentence plausible? ”Mookie Betts skated|
||relating to sports is plausible or implausible.|behind the net.”_→_no|
|Word sorting|Sort a list of words.|List: thunderclap swab built poland_→_built poland swab|
|||thunderclap|



## **J.2. Results on Super Natural Instructions** 

To further enhance the practical applicability of our approach, we conducted experiments on the Super Natural Instructions benchmark (Wang et al., 2022c). This benchmark encompasses a variety of tasks, including those related to commonsense classification and information extraction. Although it covers a wide range of tasks, our validation specifically focused on tasks related to code and mathematics. 

To accomplish this, we began by evaluating the performance of APE on tasks related to code and mathematics. Subsequently, we conducted experiments on 20 tasks where APE encountered challenges, i.e., tasks for which APE’s ROUGE-L score was below 50% (please refer to Table 10). 

## **J.3. Results on BIG-Bench-Hard benchmark** 

We conduct experiments on the tasks included in the BIG-Bench-Hard benchmark, which focuses on tasks believed to be challenging, among the BIG-Bench Instruction Induction tasks proposed in Zhou et al. (2022). 

## **K. Baselines** 

## **K.1. APE** 

In this section, to facilitate readers’ understanding, we provide a detailed explanation of APE (Automatic Prompt Engineering (Zhou et al., 2022)), which is closely relevant to our work. 

## K.1.1. THE BACKGROUND BEHIND AUTOMATIC PROMPT OPTIMIZATION 

To begin with, we aim to explain the background behind auto-prompting methods, including APE (Zhou et al., 2022). While recent LLMs have demonstrated their remarkable ability to solve tasks described by user instructions (Ouyang et al., 2022; OpenAI, 2023; Touvron et al., 2023; Peters et al., 2018; Devlin et al., 2018; Brown et al., 2020; Wei et al., 2022b), carefully crafted prompts are crucial for maximizing LLMs’ problem-solving ability. However, this often involves laborious trial and error. Recent attempts automate this by using LLMs to design prompts with their language generation ability, addressing 

19 

**Mixture-of-Prompts** 

_Table 9._ **Results on Instruction Induction.** We report the execution accuracy gain (∆) of MoP from the baseline described in Section 5.1 on the Instruction Induction benchmark tasks. We run 3 experiments and provide both the mean and standard deviation values. Please note that due to the inherent randomness in the ChatGPT API, a performance gap of less than 1% between the two methods can be considered a tie. The number of demos is set to _N_ train _/_ 10, where _N_ train is the total number of training demos. 

||The|execution accuracy gain (∆) of MoP from the following|execution accuracy gain (∆) of MoP from the following|execution accuracy gain (∆) of MoP from the following|method (%)||
|---|---|---|---|---|---|---|
|Task|APE|APE|APE|InstructZero|InstructZero|InstructZero|
||(Zhou et al.,2022)|+Demos<br>+K-centroids||(Chen et al.,2023)|+Demos|+K-centroids|
||||||||
|Auto categorization|35.33_±_2.55|29.66_±_2.48|26.00_±_4.10|26.00_±_2.81|22.66_±_2.52|20.33_±_2.92|
|Auto debugging|29.17_±_3.40|8.33_±_3.40|0.00_±_0.00|20.83_±_3.40|12.50_±_0.00|8.33_±_3.40|
|Antonyms|5.34_±_3.08|−0.66_±_0.77|−0.66_±_0.61|3.00_±_1.12|0.00_±_0.77|0.00_±_0.77|
|Cause and effect|9.33_±_9.30|6.66_±_8.98|6.66_±_9.17|9.33_±_9.30|14.66_±_7.93|17.33_±_8.08|
|Common concept|11.61_±_2.95|6.05_±_3.99|−0.52_±_3.70|6.68_±_5.05|8.89_±_4.51|0.39_±_5.96|
|Informal to formal|−2.24_±_2.55|2.61_±_3.09|6.90_±_2.97|13.89_±_4.54|3.41_±_3.77|13.56_±_4.33|
|Taxonomy animal|6.33_±_4.15|20.00_±_4.23|14.33_±_5.22|19.66_±_6.92|16.33_±_7.29|10.33_±_3.98|
|Negation|4.33_±_0.98|1.00_±_0.90|1.66_±_0.77|5.00_±_2.29|−0.67_±_0.72|1.33_±_1.09|
|Rhymes|−6.66_±_15.07|−0.66_±_5.18|−2.00_±_8.12|17.67_±_8.19|8.34_±_6.34|3.34_±_8.44|
|Sentence similarity|32.00_±_5.26|0.67_±_4.03|5.00_±_4.26|19.67_±_8.89|1.34_±_5.92|2.34_±_6.44|
|Sentiment|3.33_±_0.55|−0.33_±_0.55|1.33_±_0.55|5.33_±_1.09|0.00_±_0.67|0.67_±_0.55|
|Orthography starts with|4.00_±_1.88|−4.00_±_1.88|−0.67_±_0.86|31.67_±_13.74|4.33_±_2.07|6.33_±_2.51|
|Synonyms|4.34_±_1.54|−2.33_±_2.18|−0.66_±_2.03|−12.00_±_6.60|0.00_±_1.46|−3.00_±_2.39|
|Translation EN-DE|0.67_±_0.72|1.67_±_0.72|−0.66_±_0.90|−1.00_±_0.77|1.34_±_0.77|−0.66_±_1.02|
|Translation EN-ES|0.00_±_1.02|−0.67_±_0.98|−0.34_±_1.02|1.66_±_1.12|0.00_±_1.02|0.00_±_1.12|
|Translation EN-FR|1.33_±_1.09|0.33_±_0.55|−1.00_±_0.67|3.33_±_1.72|1.00_±_0.94|0.67_±_0.86|
|Word in context|5.66_±_2.52|−3.67_±_1.09|1.33_±_1.09|10.66_±_5.41|−1.67_±_1.19|1.66_±_2.97|
|Diff|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|65.00_±_26.54|0.00_±_0.00|1.67_±_1.36|
|First word letter|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|1.00_±_0.47|0.00_±_0.00|0.00_±_0.00|
|Larger animal|−1.33_±_0.72|−1.00_±_0.90|1.34_±_1.22|22.67_±_8.76|14.67_±_6.72|14.67_±_7.17|
|Letters list|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|
|Num to verbal|0.67_±_0.27|0.00_±_0.00|0.00_±_0.00|1.00_±_0.81|0.00_±_0.00|0.00_±_0.00|
|Active to passive|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|0.67_±_0.54|
|Singular to plural|2.34_±_0.77|0.00_±_0.38|−0.33_±_0.27|−0.33_±_0.27|−0.33_±_0.27|−0.33_±_0.27|
|Second word letter|−12.00_±_9.80|−12.00_±_9.80−11.67_±_9.80||25.33_±_16.72|22.00_±_16.94|21.00_±_16.50|
|Sum|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|0.00_±_0.00|



tasks given demo datasets. APE is one of these automatic prompt optimization methods, which has empirically demonstrated that LLM-generated prompts are more effective than human-crated prompts in solving target tasks. 

## K.1.2. DETAILED EXPLANATION OF THE APE ALGORITHM 

## **Algorithm 2** APE (Zhou et al., 2022) 

**Input:** Training demos _D_[train] = _{_ ( _xi, yi_ ) _}[N] i_ =1[train][,][validation][demos] _[D]_[valid][=] _[{]_[(] _[x][i][, y][i]_[)] _[}][N] i_ =1[ valid][,][model] _[M][ϕ]_[,][task-specific] scoring function _f_ ( _·_ ) _→_ R. Randomly sample a subset _D_[˜][train] _∼D_[train] , where _|D_[˜][train] _|_ = _r_ Generate candidate instructions _{I[j] }[m] j_ =1[using a model] _[ M][ϕ]_[ and a template format] _[ T]_[( ˜] _[D]_[train][)][ given] _[D]_[˜][train][ (Equation (][3][)).] Randomly sample a subset _D_[˜][valid] _∼D_[valid] , where _|D_[˜] valid _|_ = _q_ **for** _j_ = 1 **to** _m_ **do** Evaluate the instruction _I[j]_ on the subset _D_[˜][valid] and calculate the validation score; E( _x,y_ ) _∼D_ ˜[valid] _[f]_ �� _I[j] , x_ � _, y_ �. **end for Output:** _P[∗]_ ( _x_ ) = [ _I[∗] , x_ ], where _I[∗]_ = argmax _I j_ E( _x,y_ ) _∼D_ ˜[valid] _[f]_ �� _I[j] , x_ � _, y_ � _._ 

We provide a more detailed explanation of the APE method. In APE (Zhou et al., 2022), firstly, it leverages a pre-trained black-box LLM to propose a set of candidate instructions. Specifically, APE initially selects random demos utilized for proposing instructions and adopts the templates corresponding to ’Generating Instructions’ from Figure 6, along with the sampled demos, into [FULL ~~D~~ EMOS]. It then feeds this prompt into LLM to generate a set of candidate instructions. After 

20 

**Mixture-of-Prompts** 

_Table 10._ **The performance of APE on tasks related to code and mathematics in Super Natural Instructions.** We report the performance of APE (Zhou et al., 2022) on code and mathematics-related tasks, selected based on the domain information provided in the metadata of the Super Natural Instructions benchmark. We run 3 experiments and provide both the mean and standard deviation values. 

||Task|ROUGE-L (%)<br>APE (Zhou et al.,2022)|
|---|---|---|
||||
||Conala concat strings|88.84_±_2.25|
||Conala normalize lists|45.37_±_0.25|
||Conala calculate mean|23.00_±_2.68|
||Conala max absolute value|23.00_±_2.68|
||Conala list index subtraction|35.07_±_9.55|
||Conala remove duplicates|72.22_±_2.28|
|Code|Conala list intersection<br>Splash question to sql<br>Logic2text sentence generation|97.18_±_0.35<br>60.54_±_0.85<br>41.26_±_1.09|
||Conala list index addition|47.72_±_10.39|
||Conala sort dictionary|99.84_±_0.22|
||Conala pair averages|63.47_±_11.05|
||Conala pair differences|18.86_±_1.48|
||English language answer relevance classifcation|50.67_±_2.49|
||Code xglue information retreival|20.52_±_4.61|
||Semeval 2019 task10 closed vocabulary<br>mathematical answer generation|18.60_±_3.88|
||Semeval 2019 task10 open vocabulary<br>mathematical answer generation|19.44_±_3.61|
||Ai2 arithmetic questions arithmetic|51.81_±_5.60|
||Aqua multiple choice answering|45.85_±_0.59|
||Svamp subtraction question answering|67.79_±_5.30|
||Mathdataset classifcation|46.11_±_18.29|
||Mathdataset answer generation|24.11_±_1.95|
||Asdiv addsub question answering|74.97_±_11.30|
|Mathematics|Asdiv multidiv question answering<br>Asdiv multiop question answering<br>Asdiv singleop question answering<br>Mawps addsub question answering<br>Mawps multidiv question answering|70.96_±_10.94<br>73.01_±_10.69<br>66.83_±_13.95<br>80.27_±_4.64<br>52.17_±_5.66|
||Mawps multiop question answering|59.69_±_11.80|
||Mawps singleop question answering|77.30_±_1.00|
||Leetcode 420 strong password check|6.66_±_3.87|
||Mathqa gain|14.02_±_2.04|
||Mathqa general|17.33_±_1.30|
||Mathqa other|14.49_±_2.43|
||Mathqa geometry|20.16_±_1.01|
||Mathqa probability|12.08_±_0.56|
||Mathqa answer selection|14.61_±_1.32|
||Mathqa correct answer generation|27.15_±_1.18|



generating a set of candidate instructions in this manner, APE evaluates these generated candidate instructions using the subset of validation set ( _D_[˜][valid] ). Subsequently, it utilizes the best instruction with the highest validation score, which is a single demo-free instruction, during the test phase. For a fair comparison, all methods, including our MoP method, use the same training, validation, and test datasets. 

## **K.2. InstructZero** 

InstructZero (Chen et al., 2023) finds a single instruction for a black-box LLM by optimizing the soft prompt of an opensource LLM using a Bayesian Optimization approach. To be more specific, within each Bayesian optimization iteration in InstructZero, a soft prompt is transformed into an instruction using the open-source LLM, and this instruction is subsequently fed into the black-box LLM. The output from the black-box LLM is then sent back to the Bayesian optimization process to generate the next soft prompt. For more details, please refer to Algorithm 1 in Chen et al. (2023). 

21 

**Mixture-of-Prompts** 

_Table 11._ **Results for the Super-Natural Instructions.** We report the ROUGE-L score gain (∆) of MoP from the baseline described in Section 5.1 on the Super Natural Instructions benchmark tasks. We run 3 experiments and provide both the mean and standard deviation values. Please note that due to the inherent randomness in the ChatGPT API, a performance gap of less than 1% between the two methods can be considered a tie. The number of demos is set to _N_ train _/_ 10, where _N_ train is the total number of training demos. 

||Task|APE|The ROUGE-L score gain (∆) of<br>APE<br>APE|The ROUGE-L score gain (∆) of<br>APE<br>APE|MoP from the following<br>InstructZero|method (%)<br>InstructZero|InstructZero|
|---|---|---|---|---|---|---|---|
|||(Zhou et al.,2022)|+Demos<br>+K-centroids||(Chen et al.,2023)|+Demos|+K-centroids|
||Conala normalize lists|1.98_±_0.46|0.22_±_0.54|1.33_±_0.44|6.49_±_2.09|0.60_±_0.68|2.20_±_0.76|
||Conala calculate mean|8.55_±_2.38|6.05_±_1.83|7.72_±_1.85|7.38_±_2.47|9.55_±_2.58|9.65_±_2.08|
|Code|Conala list index subtraction<br>Logic2text sentence generation<br>Conala list index addition|15.87_±_7.76<br>55.22_±_1.28<br>4.47_±_7.81|14.53_±_6.87 <br>10.71_±_1.14<br>−4.37_±_6.11|16.72_±_6.31<br>2.68_±_1.21<br> −0.01_±_6.89|35.20_±_5.63<br>47.57_±_9.28<br>34.84_±_5.13|21.25_±_6.44<br>8.71_±_1.67<br>24.45_±_6.58|21.44_±_7.21<br>1.67_±_1.13<br>21.64_±_6.76|
||Conala pair differences|58.91_±_13.65|22.87_±_18.0517.12_±_16.22||57.94_±_15.04|44.32_±_15.00|45.15_±_13.78|
||Code xglue information retreival|19.48_±_3.01|−0.33_±_2.64|4.67_±_1.52|31.30_±_2.63|−0.67_±_4.09|3.00_±_1.63|
||Semeval 2019 task10 closed voc. math. ans. gen.|15.73_±_2.53|−2.67_±_1.27|−3.34_±_1.54|10.37_±_3.95|−2.39_±_1.89|−0.27_±_4.23|
||Semeval 2019 task10 open voc. math. ans. gen.|18.23_±_2.21|−0.66_±_0.90|−0.66_±_2.34|12.64_±_3.36|−1.33_±_0.86|−1.00_±_0.77|
||Aqua multiple choice answering|17.16_±_0.71|−6.16_±_0.68|−0.32_±_0.72|19.27_±_4.45|−5.16_±_0.72|−1.49_±_0.75|
|Mathematics|Mathdataset classifcation<br>Mathdataset answer generation<br>Leetcode 420 strong password check<br>Mathqa gain<br>Mathqa general|49.22_±_10.61<br>23.45_±_2.08<br>20.67_±_4.24<br>19.31_±_4.54<br>10.00_±_1.24|19.00_±_2.34<br>4.89_±_1.85<br>13.66_±_3.67<br>2.33_±_4.41<br>4.66_±_1.39|7.00_±_2.34<br>1.08_±_2.30<br>2.33_±_3.63<br>4.00_±_4.42<br>1.33_±_2.28|41.49_±_2.69<br>21.44_±_2.09<br>19.60_±_5.24<br>18.86_±_4.50<br>12.73_±_1.23|20.66_±_1.39<br>4.67_±_2.09<br>8.66_±_4.18<br>3.33_±_4.46<br>10.28_±_4.57|9.33_±_2.28<br>0.58_±_1.84<br>0.66_±_5.35<br>4.00_±_4.39<br>7.30_±_4.99|
||Mathqa other|21.18_±_2.17|1.67_±_1.72|5.67_±_1.85|21.88_±_2.16|3.34_±_2.34|8.34_±_1.81|
||Mathqa geometry|13.51_±_0.80|−5.33_±_0.98|5.34_±_0.61|12.58_±_2.12|−5.33_±_0.72|5.34_±_0.77|
||Mathqa probability|29.59_±_0.63|10.00_±_0.90|7.00_±_0.90|24.61_±_4.42|9.00_±_0.90|7.67_±_0.72|
||Mathqa answer selection|13.39_±_1.61|0.33_±_1.52|−1.33_±_2.96|13.64_±_4.48|8.98_±_7.48|4.72_±_8.49|
||Mathqa correct answer generation|0.03_±_2.50|−3.90_±_2.41|1.80_±_2.41|−3.52_±_3.08|−6.16_±_2.42|0.49_±_2.47|



_Table 12._ **Results on BIG-Bench-Hard.** We report the execution accuracy gain (∆) of MoP from the baseline described in Section 5.1 on the BIG-Bench-Hard benchmark tasks. We run 3 experiments and provide both the mean and standard deviation values. Please note that due to the inherent randomness in the ChatGPT API, a performance gap of less than 1% between the two methods can be considered a tie. The number of demos is set to _N_ train _/_ 5, where _N_ train is the total number of training demos. 

||The|execution accuracy gain (∆) of MoP from the following|execution accuracy gain (∆) of MoP from the following|execution accuracy gain (∆) of MoP from the following|method (%)||
|---|---|---|---|---|---|---|
|Task|APE|APE|APE|InstructZero|InstructZero|InstructZero|
||(Zhou et al.,2022)|+Demos<br>+K-centroids||(Chen et al.,2023)|+Demos|+K-centroids|
||||||||
|Causal judgement|6.38_±_1.64|1.42_±_1.36|4.25_±_1.92|0.35_±_2.62|1.77_±_1.30|1.77_±_1.79|
|Disambiguation QA|3.67_±_2.95|−1.67_±_2.28|−1.67_±_1.72|7.00_±_1.49|−2.33_±_1.28|−3.33_±_2.07|
|Dyck languages|9.33_±_6.69|13.00_±_1.61|5.33_±_4.77|7.66_±_6.49|6.66_±_1.86|5.33_±_5.54|
|Movie Recommendation|11.67_±_8.17|0.00_±_1.98|−2.33_±_2.13|13.34_±_5.63|3.67_±_1.59|0.00_±_2.52|
|Navigate|−5.00_±_5.08|14.33_±_2.46|4.00_±_4.45|6.00_±_2.87|14.00_±_2.40|2.00_±_5.93|
|Object counting|0.34_±_2.14|13.34_±_1.21|3.67_±_1.72|1.67_±_5.38|12.00_±_1.86|4.34_±_1.86|
|Ruin names|3.66_±_1.87|6.00_±_1.31|1.66_±_1.75|5.66_±_1.75|4.66_±_2.25|3.66_±_1.31|
|Snarks|−6.74_±_3.99|−0.37_±_3.06|−1.12_±_2.84|2.25_±_4.29|4.87_±_3.92|4.12_±_4.22|
|Sports understanding|1.33_±_2.27|−1.67_±_1.59|1.33_±_2.80|5.00_±_2.34|−3.34_±_1.61|0.00_±_2.34|
|Word sorting|10.67_±_1.19|−2.33_±_1.09|1.00_±_2.39|6.34_±_5.16|−4.00_±_1.87|−2.00_±_1.98|



## **L. Implementation Details** 

As described in Section 5, for a fair comparison, we allocate an equal budget to all methods. To elaborate further, APE and InstructZero search for the optimal prompt among 20 candidate instruction options, while in the case of MoP, the total number of candidate instructions across all experts sums up to 20. 

In the case of APE+Demos, APE+K-centroids, InstructZero+Demos, and InstructZero+K-centroids, each of them combines the best prompt found through APE or InstructZero with randomly selected demos or demos corresponding to centroids in the clustered embedding space. For these methods, the number of demos is set the same as in the case of MoP for a fair comparison. 

Regarding hyperparameters, the _α_ value in Equation (10) is set to the default value of 0.02 and remained the same across all experiments. 

22 

**Mixture-of-Prompts** 

## **M. Limitations** 

To promote future exploration, we discuss two limitations of the proposed method. First, the K-means-Auto algorithm used in the demo assignment does not guarantee the balance of the resulting clusters. When a cluster receives demos that exceed the limit, we randomly discard them to meet the constraint. This operation might be sub-optimal as it does not factor in their relative importance. Future work might explore various data selection methods for trimming the cluster size. Second, MoP uses existing instruction generation method (APE), but sometimes APE fails to generate sensible instructions in the first place. However, MoP can be applied to any instruction generation method, and if better instruction generation methods emerge in the future, we can also expect improved performance from MoP accordingly. Lastly, the theory that motivates the use of clustering algorithm to assign demos - connecting ICL to kernel regression - cannot explain the demo order sensitivity in LLMs. This suggests that future theoretical advancements could help motivate better demo assignment algorithms. Finally, while the theory (Han et al., 2023) connecting ICL to kernel regression inspires the use of clustering algorithms for demo assignments, it could not explain the order sensitivity of demos in LLMs; Further theoretical developments could help develop better demo assignment algorithms. 

23 

