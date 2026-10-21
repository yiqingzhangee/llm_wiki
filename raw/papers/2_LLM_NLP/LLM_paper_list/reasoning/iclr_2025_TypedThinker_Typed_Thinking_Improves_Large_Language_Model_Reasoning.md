# TYPEDTHINKER: TYPED THINKING IMPROVES LARGE LANGUAGE MODEL REASONING 

**Danqing Wang**[1] _[,]_[2] **, Jianxin Ma** _[∗]_[2] **, Fei Fang**[1] **, Lei Li**[1] 1 Carnegie Mellon University 2 Qwen Team danqingw@cs.cmu.edu 

## ABSTRACT 

Despite significant advancements in the reasoning capabilities of Large Language Models (LLMs), the lack of diverse reasoning solutions often makes them trapped in a limited solution search area. In this paper, we propose TypedThinker, a reasoning framework that diversifies LLMs’ reasoning solutions by incorporating multiple reasoning types (deductive, inductive, abductive, and analogical). Our analysis across four benchmarks reveals that different reasoning types uniquely solve distinct sets of problems, highlighting the importance of diverse thinking approaches. TypedThinker addresses two key challenges: selecting appropriate reasoning types for given problems and effectively implementing specific reasoning types. Through self-training on successful experiences, TypedThinker learns an implicit policy for reasoning type selection and application. Experimental results demonstrate significant improvements over baseline models, with accuracy increases of 3.4% for Mistral 7B and 16.7% for LLaMA3 8B across four reasoning benchmarks. Notably, TypedThinker shows effective generalization to new benchmarks and can further enhance the reasoning capability of powerful models like GPT-4o. The code is released at https://github.com/dqwang122/ThinkHub. 

## 1 INTRODUCTION 

Large Language Models (LLMs) exhibited promising capabilities in reasoning, such as solving logical reasoning and mathematical problems (Bai et al., 2022; OpenAI, 2023). Plenty of work has been done to improve the reasoning capabilities by adding reasoning thoughts (Wei et al., 2022) and making these thoughts more elaborated (Fu et al., 2023; Zheng et al., 2024). However, the exploration of novel reasoning thoughts is understudied. The lack of diversity in reasoning makes LLMs easily trapped in a fixed mindset, which limits their performance in solving difficult problems. 

Current research efforts to enhance reasoning diversity remain unsatisfactory. AlphaCode (Li et al., 2022; Leblond et al., 2023) randomizes the difficulty level and categorical tags of the code problems in the prompt to encourage diversity. However, its dependency on manually curated attributes makes it challenging to scale and apply beyond coding problems. On the other hand, increasing the temperature is an easy way to generate superficially diverse outputs, but it often fails to produce high-quality reasoning solutions with significant differences. For example, repeated sampling (Brown et al., 2024) generates 100,000 solutions per problem with temperature 0.6, but their solutions[1] are mostly based on deductive reasoning, which starts from the problem context to infer the answer step by step. 

Encouraging LLMs to apply more types of reasoning can make them think more diversely. Besides deductive reasoning, there are other reasoning types such as inductive (Flach and Kakas, 2000), abductive (Douven, 2011), and analogical reasoning (Bartha, 2013). These reasoning types reflect different mental methods of drawing conclusions, leading to diverse thinking processes. This is also inspired by human cognitive processes, where individuals selectively apply contextually appropriate logical reasoning strategies (Halpern, 2014; Bronkhorst et al., 2020). For example, when some answer candidates are given for a free-response math problem, humans are more likely to verify the correctness of each option instead of solving it from the given conditions. This reflects human’s implicit mental shift in the type of reasoning from deductive to abductive. 

> _∗_ Work was done during the internship in Qwen Team. 

> 1Their results can be found at: https://huggingface.co/datasets/ScalingIntelligence/monkey_business. 

1 

To investigate how different reasoning types affect the LLMs’ performance in solving reasoning problems, we calculate the percentage of problems that can only be solved by one particular reasoning type in Figure 1. Specifically, we generate 10 samples for each reasoning type with the Mistral 7B instruct (Jiang et al., 2023) on four benchmarks: LogiQA (Liu et al., 2023a) and BBH (Suzgun et al., 2022), GSM8k (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021). We consider that one problem is solved by a reasoning type if at least one correct solution is found among the 10 samples of this reasoning type. For example, the blue region in Figure 1 reflects the percentage of problems that can only be solved by inductive reasoning, which is surprisingly effective on the MATH benchmark. It means that without inductive reasoning, LLMs can hardly find a correct solution for these 11.03% problems. As we can see, each reasoning type has a unique set of problems that can be solved by it, highlighting the importance of divergent thinking based on the reasoning types to enlarge the solvable problem set. More analysis can be found in Section 4.2. 

Integrating diverse reasoning strategies into LLMs’ problem-solving capabilities has two main challenges. First, as shown in Figure 1, it is important to select the appropriate reasoning type for a given problem, otherwise, it may mislead the LLMs with the wrong thinking direction. Besides, it may be difficult for LLMs to follow a specific reasoning type to solve the problem. Therefore, we propose a novel approach TypedThinker with the implicit policy to select and conduct the appropriate reasoning types and the explicit memory to retrieve relevant experiences to aid problem-solving. TypedThinker uses the **meta-thinker** and the **reasoner** , to select the reasoning types and conduct the selected types of reasoning. It also 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_TypedThinker_Typed_Thinking_Improves_Large_Language_Model_Reasoning_images/iclr_2025_TypedThinker_Typed_Thinking_Improves_Large_Language_Model_Reasoning.pdf-0002-02.png)


**----- Start of picture text -----**<br>
Abductive<br>Analogical<br>Inductive<br>Deductive<br>LogiQA BBH GSM8k MATH<br>0.20<br>0.15<br>0.10<br>0.05<br>0.00<br>Percentage<br>**----- End of picture text -----**<br>


Figure 1: The percentage distribution of each reasoning type for problems. The y-axis indicates the percentage of problems that can be only solved by one specific reasoning type. 

maintains an **explicit memory** to retrieve relevant knowledge and experiences. To learn the implicit policy for reasoning types, TypedThinker is optimized based on its own successful experiences. For each problem in the training set, it estimates the effectiveness score of each reasoning type based on the success rate during sampling. The meta-thinker is fine-tuned on these empirical effectiveness scores to learn reasoning type selection for each problem. These successful reasoning experiences are also used to enhance the reasoner’s capability of following a specific reasoning type in problem-solving. During the self-training process, TypedThinker updates the implicit policy of the meta-thinker to select reasoning types and the policy of the reasoner to apply this type, while storing the successful reasoning experiences in the explicit memory for retrieval. 

Experimental results show that TypedThinker improves Mistral 7B instruct (Jiang et al., 2023) by 3.4% and LLaMA3 8B instruct (Touvron et al., 2023) by 16.7% on the average accuracy of two logical benchmarks, LogiQA and BBH, and two mathematics benchmarks, GSM8k and MATH. We further demonstrate that TypedThinker can directly be applied to the new benchmark Contexthub (Hua et al., 2024) and outperforms other baselines. Moreover, we show that the meta-thinker based on the weak 7B model can enhance the performance of GPT-4o without distilling knowledge from strong LLMs, which also sheds light on the weak-to-strong generalization (Burns et al., 2024). 

## 2 RELATED WORK 

**Logical Reasoning** Logical reasoning includes various methods to emulate human-like thought processes (Wason and Johnson-Laird, 1972). Deductive reasoning focuses on deriving specific conclusions from general principles or premises, ensuring that conclusions logically follow if the premises are true (Johnson-Laird, 2010). In contrast, inductive reasoning involves generalizing from specific instances to broader principles, often used to identify patterns and make predictions based on empirical data (Flach and Kakas, 2000). Abductive reasoning, considered more creative and open-ended, involves forming hypotheses to explain observations, often generating the most plausible explanation rather than a guaranteed conclusion (Douven, 2011). Analogical reasoning is concerned with the comparison between two or more objects and drawing a conclusion based on the similarity (Bartha, 2013). Previous LLMs studies on logical reasoning mainly focus on benchmarking 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_TypedThinker_Typed_Thinking_Improves_Large_Language_Model_Reasoning_images/iclr_2025_TypedThinker_Typed_Thinking_Improves_Large_Language_Model_Reasoning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Explicit Memory Initialize<br>Experience 1: (Problem 1, Deductive, Solution 1)… LLM Sample withreasoning types<br>Experience N: (Problem N, Abductive, Solution N) Abductive<br>Experience Solutions<br>Implicit Policy Filter incorrect ones<br>Filter incorrect types<br>Problem<br>Finetune<br>Deductive Inductive Abductive Analogical Abductive  Experience dataset 𝐷 ["]<br>s = 0.3 s = 0.1 s = 0.6 s = 0 Reasoner Store Reasoner<br>Meta<br>thinker Reasoner Empirical  Finetune<br>Score<br>Explicit Memory Meta-thinker<br>**----- End of picture text -----**<br>


Figure 2: TypedThinker consists of three components: the meta-thinker to select the reasoning types, the explicit memory to retrieve relevant experience, and the reasoner to conduct the specific reasoning. The meta-thinker is fine-tuned to predict an effective score _s ∈_ [0 _,_ 1] for each reasoning type. 

Figure 3: Learning the implicit policy and explicit memory by self-training. 

its performance in different reasoning types (Bang et al., 2023; Dougrez-Lewis et al., 2024; Luo et al., 2023; Yu et al., 2024). Instead, this paper mainly focuses on the selection and application of the appropriate reasoning type when solving a specific problem. 

**Reasoning in Large Language Models** Plenty of studies have been done to enhance the reasoning capability of LLMs. Chain-of-thoughts methods focus on creating better instructions to improve the quality of the reasoning process, such as Complex CoT (Du et al., 2023), Tree of Thought (ToT) (Yao et al., 2023) and Graph of Thought (Besta et al., 2024). Refinement-based methods revise LLMs solutions by the feedback from themselves or others model (Akyürek et al., 2023; Wang and Li, 2023; Du et al., 2023). Search-based methods use the reward model to search the best reasoning path (Lightman et al.; Liu et al., 2023b; Hao et al., 2023). While most of them focus on creating high-quality reasoning paths, the diversity of thinking is understudied. Our paper aims to diversify the thinking process by incorporating different reasoning types. 

**Self-improvement and Self-training in LLMs** Recent works explore the self-improvement capability of LLMs, by finetuning LLMs on their high-quality generations (Wang et al., 2023; Huang et al., 2023; Toshniwal et al., 2024). This process can be extended to multiple iterations Gülçehre et al. (2023); Aksitov et al.. Benefiting the LLMs’ ability to follow instructions, researchers also ask LLMs to provide feedback themselves and improve their responses without finetuning (Peng et al., 2023; Shinn et al., 2023). This can be further enhanced by using their own feedback as the reward model to provide better signals for finetuning (Yuan et al., 2024; Kumar et al., 2024). In this paper, we focus on stimulating their capabilities to conduct various reasoning types and use these experiences to diversify their thinking in reasoning type selection and following. 

## 3 TYPEDTHINKER: DIVERSIFY THINKING WITH TYPED REASONING 

In this paper, we focus on four logical reasoning types: deductive, inductive, abductive, and analogical reasoning. For each reasoning type, we provide a short definition and a simple example to demonstrate the inference rules, which are listed in Table 8 in the Appendix. Based on that, we introduce a reasoning framework TypedThinker to diversify LLMs’ thinking with different reasoning types. As shown in Figure 2, there are three components in TypedThinker: the meta-thinker to select reasoning type, the finetuned reasoner to conduct specific reasoning, and explicit memory to retrieve experience. TypedThinker optimizes the implicit policy of the meta-thinker and reasoner and updates the explicit memory based on self-training. 

## 3.1 TYPED REASONING WITH THE IMPLICIT POLICY AND EXPLICIT MEMORY 

Let _D_ = _{_ ( _**x**_ **1** _, y_ 1) _, · · · ,_ ( _**xN** , yN_ ) _}_ be a set of _N_ problems, where _**x** i_ and _yi_ is the problem and the ground-truth answer of the _i−_ th instance. We define a reasoning type space _F_ that includes an empty type and four types of reasoning: deductive, inductive, abductive, and analogical. The goal is to model the selection and implementation of various reasoning types as implicit and explicit memory, thus enhancing LLMs’ performance in reasoning tasks. Here we use the exact match between the 

3 

ground-truth answer _y_ and the system output _y_ ˆ to calculate the accuracy as the measurement of task performance. 

**Meta-thinker to select the reasoning type.** Given a problem _**x**_ , the goal of the meta-thinker is to select an appropriate set of reasoning types to solve the problem. Specifically, it predicts an effectiveness score _sk ∈_ [0 _,_ 1] for each reasoning type _fk ∈F_ , which can be represented as _sx,k_ = _πθ_ ( _**x** , fk_ ). _sx,k_ = 0 indicates that the problem _**x**_ can hardly be solved by the reasoning type _fk_ with a limited sampling times[2] . Note that the effectiveness scores of different types are independent of each other and their sums are not necessary to be 1. The most effective reasoning type is defined as _f[∗]_ ( _**x**_ ) = arg max _fk∈F sx,k_ . Meanwhile, we can obtain a set of reasoning types with a non-zero effective ratio, which we call the effective set: _F_ ( _**x**_ ) = _{fk|sx,k >_ 0 _}_ . We initialize _πθ_ with the pre-trained LLM and later fine-tune the parameters _θ_ during the self-training. The prompt is listed in Appendix A.1. 

**Explicit Memory to retrieve experience.** TypedThinker keeps an explicit memory _M_ = � _fk∈F[M][k]_[for each type of reasoning.][For each problem, we keep one correct solution per reasoning] type if applicable, resulting in a set of at most _|D| × |F|_ solutions. If multiple solutions exist for one problem, we keep the longest to get a more detailed context. The entry in the memory _Mk_ is represented as a tuple of ( _**x** r, solr,_ ). Here _solr_ is the concrete reasoning process of the reasoning type _fk_ , including the predicted answer. Given a new problem _**x**_ and its reasoning type _fk_ , it retrieves a set of relevant experience _dx,k_ = _{_ ( _**x** r, solr_ ) _∈ Mk|L_ ( _**x** r,_ _**x**_ ) _< δ}_ . _L_ is the distance function measuring relevancy between two problems and _δ ∈_ [0 _,_ 1] is the relevancy threshold. In this study, we use the cosine similarity for the distance function. In this study, we use the cosine similarity between the semantic embeddings as the distance function. The retrieved experiences are used as the few-shot examples of the reasoner. 

**Reasoner to perform the reasoning according to the type.** The reasoner applies the reasoning type _fk_ to the problem _**x**_ and provides a detailed reasoning path for its predicted answer _y_ ˆ. The reasoner is based on LLM to conduct reasoning and the instruction is composed of ( _**x** , fk, dx,k_ ), where the _dk_ is the retrieved relevant successful experience. The reasoner can be further optimized via instruction tuning to enhance the capability of conducting a specific type of reasoning. 

During inference, TypedThinker first uses the meta-thinker _πθ_ to predict an effective score _sk_ for each reasoning type. It then retrieves relevant reasoning experience _dk_ from the explicit memory. Finally, the reasoner conducts the specific type reasoning _fk_ with the help of past successful experience. There are two approaches to aggregating multiple solutions. One is to greedily resample several times based on the most effective reasoning type _f[∗]_ and use self-consistency (Wang et al., 2022) to enhance the answer. The other is to sample solutions for all effective reasoning types _F_ , and apply a weighted vote with the effective score as the coefficient. By default, we use the greedy approach for TypedThinker, and we discuss the weighted vote in Section 4.4. 

## 3.2 OPTIMIZE IMPLICIT POLICY FOR REASONING TYPE SELECTION AND FOLLOWING 

We use a self-training framework to optimize the meta-thinker _πθ_ and the reasoner while updating the explicit memory with the collected experience. The pipeline is demonstrated in Figure 3. The green lines represent the parametric optimization process, while the blue line represents the non-parameter update. 

**Diversify Reasoning Experiences with Types** To inspire LLMs’ knowledge of solving problems with different reasoning types, the definition (Table 8) and manually-written few-shot examples with detailed reasoning paths (Table 11) are used for prompting solutions for each reasoning type. For each problem in the training set, we use a temperature of 1 to sample 10 solutions per reasoning type. These solutions are then filtered by the correctness of the final answers. To guarantee that these solutions belong to their reasoning type, we apply a _reverse check_ on the remaining solutions. For the experience ( _**x** , sol, y_ ) of the reasoning type _fk_ , we prompt the model to predict its reasoning type _f_[ˆ] _k_ . If _fk_ = _f_[ˆ] _k_ , we think this experience indicates the methodology of this reasoning type and keep it. Otherwise, it will be removed. Finally, we get an experience dataset _D_ with multiple reasoning types. The experiences are grouped by their reasoning type and are stored in the explicit memory _M_ . 

> 2In this paper, we sample at most 10 times for one problem. 

4 

**Optimize the Implicit policy of Meta-thinker and Reasoner** Given a problem _**x**_ , the meta-thinker _πθ_ predicts a score _sx,k_ to indicate how likely this reasoning type can solve this problem. This can be estimated by the experience in the training set. We assume that if one reasoning type is more effective in solving this problem, it will generate more correct solutions among the same sampling times. Therefore, given there are _nk_ successful experiences of the reasoning type _fk_ among _m_ samples, we define the empirical effectiveness score based on its success rate: _sx,k_ = _nk/m_ . This empirical effectiveness score calculated on the experience dataset _D_ is then used for finetuning the meta-thinker. We reconstruct the tuple ( _**x** , fk, sx,k_ ) into the instruction-following pair via the prompt in Section A.1 for supervised finetuning. Meanwhile, we finetune a reasoner with the experience to enhance its capability to conduct a specific type of reasoning. 

To conclude, TypedThinker first uses the meta-thinker to implicitly identify the reasoning types _fk_ needed to solve the problem, then retrieves relevant demonstrations _dk_ from the explicit memory, and finally employs the fine-tuned reasoner _Mϕ_ to execute the specific reasoning type _fk_ on the query with _dk_ to generate the response based on the best reasoning type or a weighted reasoning vote over the effective reasoning set. 

## 4 EXPERIMENTS 

In this section, we first introduce our experiment settings and investigate the influence of different reasoning types on the self-training data. We then evaluate our TypedThinker performances on four benchmarks. Further analyses are provided to enhance the understanding of TypedThinker. 

## 4.1 EXPERIMENT SETUP 

We investigate two open-source LLMs Mistral 7B instruct (Jiang et al., 2023) and LLaMA3 8B instruct (Touvron et al., 2023) on two logical benchmarks (LogiQA, BBH) and two mathematics benchmarks (GSM8K and MATH). For each LLM, we set up the following baselines: (i) **Few-shot baseline** with 3 in-context examples. We use the few-shot examples provided in Suzgun et al. (2022) for BBH, and text-based few-shot examples in Toshniwal et al. (2024) for GSM8k and MATH since we do not consider the code interpreter in this paper. We also manually write few-shot examples for LogiQA, (ii) **CoT Selection** : Select the best reasoning type by prompting. We let the LLM identify the best reasoning type and then apply the selected type to the problem. (iii) **Zero-shot Mixture of Reasoning (MoR)** : apply all possible reasoning types and use the majority vote to get the final answer[3] . The LLM is instructed with the definition and demonstration in Table 8. (iv) **Few-shot MoR** : Similar to the zero-shot MoR except for each reasoning type, 3 few-shot examples are provided in the prompt. (v) **TypedThinker** : use the most effective reasoning type _f[∗]_ . The +SC baselines indicate the majority vote over 5 responses. 

The temperature is set to 0.7 for all baselines as suggested by Wang et al. (2022). The maximum output length is set to 1000 tokens. We use SentenceTransformer[4] (Reimers and Gurevych, 2019) to retrieve top-3 similar experiences and the threshold is set to _δ_ = 0 _._ 5. On logical benchmarks, we compare the response with the ground truth based on the exact match. On the mathematics benchmarks, we follow Toshniwal et al. (2024) to get the accuracy on mathematics benchmarks. 

For self-training of TypedThinker, we use the splits in the original papers for LogiQA and follow the split of Toshniwal et al. (2024) for GSM8k and MATH. For BBH, we utilize 16 English multiple-choice tasks and randomly select 100 examples per task as the test set, with 20 examples as the hold-out validation set. The detailed statistics are listed in Table 9. Finally, the curated generation dataset covers 67.2% problems on the LogiQA benchmark, 69.7% on BBH, 74.88% on GSM8k, and 36.27% on MATH. We finetune a unified meta-thinker for both math and logical problems, and a unified reasoner for all reasoning types. We use Huggingface (Wolf et al., 2019) with deepspeed (Rasley et al., 2020). The finetuning is conducted on 2 A6000 GPUs. The batch size is 64 and the learning rate is 1 _e −_ 5. The maximum epoch is 3 for the meta-thinker and 2 for the reasoner. 

> 3For answers with the same votes, we choose the first one in alphabetical order. 

> 4https://www.sbert.net/ 

5 

## 4.2 HOW DO REASONING TYPES ENCOURAGE DIVERGENT THINKING DURING GENERATION? 

We first investigate the role of reasoning types in LLMs’ self-training. We group problems of the collected experience dataset _D_ based on their empirical effective set _F_ ( _**x**_ ) = _{fk_ ~~_|_~~ _sx,k >_ 0 _}_ , and the empirical effectiveness score is defined in Section 3.2. The problems in the same group can be solved with the same set of reasoning types. We focus on the effective set with only one reasoning type and count the size of these groups. The size indicates how many problems that can only be solved by one specific reasoning type, showing the advantage of including this reasoning type. We illustrate their percentage on the whole dataset in Figure 1. We find that even if we use temperature = 1 to sample 10 times to diversify the solutions, a lot of problems still have only one effective reasoning type. _It indicates that given an inappropriate reasoning type, the diversity brought by repeated sampling with a high temperature cannot help the LLM solve this problem._ 

Meanwhile, although these reasoning types have similar performance on the whole dataset (shown in Table 6 in Appendix), the problems they can solve do not completely overlap. None of the percentages of the reasoning type is zero, indicating that for each reasoning type, there is a unique set of problems that can only solved by it. It indicates that these reasoning types have their advantages over different problems, highlighting the importance of considering the appropriate reasoning types during problem-solving. 

We further compare the diversity of the solutions before and after adding the reasoning types in Table 10 of Appendix A.3. We can find that introducing different reasoning types can bring more diversity to the solution set than repeated sampling with a high temperature. 

Table 1: TypedThinker achieves the best performance in both single response and the majority vote setting on two logical benchmarks and two math benchmarks. @5 indicates the result is based on the majority vote over 5 responses. +SC indicates the self-consistency method. MoR indicates the Mixture of Reasoning, which employs all reasoning types (including an empty type) and votes for the final output. Avg. indicates the average accuracy over four benchmarks. 

|Mistral 7B<br>LLaMA3 8B|Mistral 7B<br>LLaMA3 8B|Mistral 7B<br>LLaMA3 8B|Mistral 7B<br>LLaMA3 8B|
|---|---|---|---|
|LogiQA BBH GSM8K MATH<br>Avg.<br>LogiQA BBH GSM8K MATH<br>Avg.||||
|||||
|Few-shot<br>0.485<br>0.346<br>0.369<br>0.074<br>+ SC @5<br>0.532<br>0.441<br>0.444<br>0.136<br>CoT Selection<br>0.474<br>0.361<br>0.372<br>0.095<br>+ SC @5<br>0.503<br>0.429<br>0.466<br>0.132|0.318<br>0.388<br>0.325<br>0.382|0.566<br>0.318<br>0.472<br>0.105<br>0.573<br>0.364<br>0.457<br>0.134<br>0.552<br>0.370<br>0.353<br>0.094<br>0.583<br>0.424<br>0.460<br>0.134|0.365<br>0.382<br>0.342<br>0.400|
|||||
|Zero-shot MoR @5<br>0.528<br>0.414<br>0.313<br>0.108<br>Few-shot MoR @5<br>0.509<br>0.456<br>0.460<br>0.127|0.341<br>0.388|0.558<br>0.448<br>0.261<br>0.075<br>0.568<br>0.428<br>0.470<br>0.133|0.335<br>0.400|
|||||
|TypedThinker<br>0.554<br>0.423<br>0.386<br>0.092<br>+ SC @5<br>0.570<br>0.469<br>0.500<br>0.149|0.364<br>**0.422**|0.550<br>0.533<br>0.535<br>0.193<br>0.620<br>0.591<br>0.723<br>0.263|0.453<br>**0.549**|



## 4.3 WHAT KINDS OF BENEFITS CAN TY P E DTH I N K E R BRING? 

As we can see in Table 1, our TypedThinker achieves the best performance among baselines. The improvement is more obvious in LLaMA3 8B, which is more powerful than Mistral 7B. It shows that LLMs with a better capability in reasoning and instruction-following can benefit more from the self-training of TypedThinker. Additionally, there are several key insights from the detailed comparison with different baselines. 

**Appropriate reasoning types improve the reasoning performance.** The main difference between Fewshot and CoT Selection without the majority vote is the reasoning type selection. For CoT Selection, the model is first prompted to predict a reasoning type and then apply it, while the Fewshot baseline directly solves the problem. However, we find that the CoT Selection struggles with the reasoning type selection. Given the option to choose from four reasoning types or none, it chooses none over 60% of the time. The rest of the time, it selects more than 50% deductive, while only 34% of them can be effectively solved by deductive reasoning during the sampling. The mismatch in reasoning types results in poor performance. Facilitating with a trained metathinker, TypedThinker is more accurate in selecting the reasoning type, which helps it improve performance under the single response setting. 

6 

**Precise prediction is more effective than an inappropriate mixture.** The zero-shot MoR and few-shot MoR apply all types of reasoning to the given problem and use a majority vote to get the final answer. Compared with the other two majority vote baselines Few-shot + SC @5 and CoT Selection + SC @5, these methods fall behind on several benchmarks, especially on MATH. We find that the performance drop typically happens when there are only one or two reasoning types that are effective for this problem. In such cases, the majority of incorrect answers dominate, resulting in fewer votes for the correct one. As we can see in Figure 1, plenty of problems on the MATH benchmark can only be solved by inductive reasoning. In such cases, if the CoT Selection correctly predicts the inductive reasoning for them, the CoT Selection + SC @5 can benefit from the majority vote and have a better performance. This highlights the importance of predicting the effectiveness of reasoning types before aggregating them. 

**Experience of how to conduct a specific type of reasoning is important.** The performance difference between zero-shot and few-shot MoR illustrates the impact of the reasoning demonstration. When prompted solely with the definition, LLMs struggle to understand how to apply the reasoning type to specific problems. It can be improved by human-written few-shot examples in few-shot MoR. However, it still falls behind the non-parametric retrieval and the parametric reasoner in TypedThinker, both of which enhance the capability of conducting a specific reasoning type. 

Table 2: Ablation Study on the Mistral 7B based TypedThinker’s components. We remove one component each time. The results are based on the best reasoning type and calculated for the single response per query. The negative scores indicate the performance drop, and the largest scores are shown in bold. 

||LogiQA|BBH|GSM8K|MATH|
|---|---|---|---|---|
|TypedThinker|0.554|0.423|0.386|0.092|
|w/o Finetuned Reasoner|-0.076|-0.041|-0.102|-0.018|
|w/o Meta-thinker|-0.025|-0.036|**-0.152**|**-0.024**|
|w/o Memory|**-0.082**|**-0.051**|-0.033|0.013|



## 4.4 WHAT CONTRIBUTES TO TYP E DTH I N K E R’S EFFECTIVENESS? 

We conduct several investigations to enhance the understanding of our proposed method. 

**Ablation study** The ablation studies are conducted on three key components in TypedThinker. Each time one component is removed. It includes (i) **w/o Fine-tuned Reasoner** : it is replaced with the base LLM (ii) **w/o Meta-thinker** : it is replaced with a CoT selection (iii) **w/o Memory** : the explicit memory is replaced with the human-written few-shot examples of each reasoning type. In Table 2, we can find the meta-thinker is the most important module for the math benchmarks, while the explicit memory is more effective on two logical benchmarks. The fine-tuned reasoner also contributes a lot to the performance improvement. We also observe that explicit memory does not always bring benefits: the performance on MATH even increases when we remove it. We find that the retrieved examples usually have a similar context but different numbers. The math calculation in the retrieved chain-of-thoughts solutions will mislead the reasoner. This is consistent with the observations of Toshniwal et al. (2024) that the solutions with masked computations are more beneficial to the math problems. For logical problems, there are fewer calculations and the retrieved solutions focus more on the reasoning process. 

**Meta-thinker’s predictions achieve a high correlation with the empirical effectiveness score.** We evaluate the performance of the meta-thinker by the correlation between the predicted effectiveness score and the empirical one (which we view as the ground truth). We split the collected experience dataset _D_ by problems and use 0.9 of them to train the meta-thinker and 0.1 for testing. We use Kendall’s _τ_ coefficient to evaluate the correlation. It measures rank correlation, essentially assessing the similarity of orderings when data is ranked. A higher Kendall’s _τ_ coefficient indicates that when the ground truth assigns a high effectiveness score to a reasoning type, the meta-thinker also ranks it high, thereby validating the reliability of the predicted scores. We compare the performance under three settings: the meta-thinker trained only on the logical domain, only in the math domain, and jointly trained on the unified domains (including both logic and math data). The meta-thinker trained on the unified domain achieves the highest correlation. This suggests that training on a dataset with multiple domains enhances the meta-thinker’s ability to accurately rank and predict suitable 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_TypedThinker_Typed_Thinking_Improves_Large_Language_Model_Reasoning_images/iclr_2025_TypedThinker_Typed_Thinking_Improves_Large_Language_Model_Reasoning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
0.40 0.60<br>0.35<br>0.50<br>0.30<br>0.40<br>0.25<br>0.20 0.30<br>0.15<br>0.20<br>0.10<br>0.10<br>0.05<br>0.00 0.00<br>Deductive Inductive Analogical Abductive LogiQA BBH GSM8K MATH<br>Logic Domain Math Domain Unified Domains Logic Domain Math Domain Unified Domains<br>Accuracy<br>Kendall's tau coefficient<br>**----- End of picture text -----**<br>


Figure 4: Kendall’s _τ_ coefficient between the prediction confidence score with the ground truth. All results have the p-value _<_ 0 _._ 05. The unified policy shows the best correlation on all reasoning types. 

Figure 5: Task performance of TypedThinker with meta-thinkers trained on different domains. The unified one performs best in most cases except the MATH benchmark, where the pure math setting dominates. 

reasoning types, thereby improving its overall performance. We also calculate the accuracy between the predicted optimal reasoning type and the empirical one for the unified setting. The average accuracy on four benchmarks is 68.3% (LogiQA 75.4%, BBH 75.6%, GSM8k 72.1%, and MATH 47.7%). Note that the meta-thinker can predict an incorrect optimal reasoning type _fk_ while still generating a correct solution. It is because the predicted reasoning type can belong to the effective set _F_ = _{fk|sx,k >_ 0 _}_ , indicating the reasoning type can also help solve the problem. 

**Unified meta-thinkers perform well in most cases.** We further investigate the effectiveness of these policies by facilitating TypedThinker with these meta-thinkers. The results are based on the Mistral 7B TypedThinker without SC. The results in Figure 5 show that the unified meta-thinker has the best performance in most cases. However, in the more difficult MATH dataset, the specific meta-thinker trained in the math domain can help it be more powerful. To conclude, the unified meta-thinker has reasonable performance in all its domains, while for difficult problems it may slightly underperform the specific meta-thinker trained in this domain. 

**Optimal reasoning type v.s. weighted vote on the effective set** In the main experiment, we use the optimal reasoning type _f[∗]_ which has the highest effectiveness score for reasoning. As discussed in Section 3.1, we can also use a majority vote on the effective set _F_ with the effectiveness score as the coefficient. Specifically, if one solution is based on a reasoning type with a higher effectiveness score, its vote gets a larger weight. The results are shown in Figure 3. We can see that the weighted vote can balance different reasoning types on LogiQA and GSM8k for the Mistral-7B-based model. However, on the other two benchmarks, the TypedThinker + SC @5 has a better performance. It indicates that accurate selection is more important if one or two reasoning types dominate the benchmark. For example, as we have shown in Figure 1, there are a lot of problems that can only be solved by inductive reasoning, indicating the other types will mislead the final answer. In such cases, the self-consistency of inductive reasoning is more powerful than the weighted vote. However, when we have a more accurate meta-thinker that can identify the correct reasoning type and a more powerful reasoner that can follow the specific reasoning type, for example, models initialized by LLaMA3, the advantage of TypedThinker + SC is more obvious. 

Table 3: TypedThinker’s performance with the most effective reasoning type _f[∗]_ v.s. weighted votes on the effective set _F_ . 

Table 4: TypedThinker performs best on the unseen benchmark Contexthub. Here the results are based on the majority vote over 5 responses (+SC @5). 

|<br>fective reasoning type_f ∗_v.s. weighted votes on the effective<br>set_F_.|Table 4: TypedThinkerperforms best on<br>the unseen benchmark Contexthub. Here the<br>results are based on the majority vote over 5<br>responses (+SC @5).|
|---|---|
|LogicQA BBH GSM8K Math Average<br>Mistral 7B<br>SC @5 on_f ∗_<br>0.570<br>0.469<br>**0.500**<br>**0.149**<br>**0.422**<br>weighted on_F_<br>**0.581**<br>0.453<br>0.501<br>0.127<br>0.416<br>LLaMA3 8B<br>SC @5 on_f ∗_<br>**0.620**<br>**0.591**<br>**0.723**<br>**0.263**<br>**0.549**<br>weighted on_F_<br>0.609<br>0.579<br>0.703<br>0.234<br>0.531||
||Mistral 7B LLaMA3 8B|
||Few-shot<br>0.419<br>0.378<br>CoT Selection<br>0.415<br>0.369<br>Zero-shot MoR<br>0.415<br>0.357<br>Few-shot MoR<br>0.432<br>0.398|
||TypedThinker<br>**0.452**<br>**0.403**|



8 

## 4.5 CAN TYPEDTHINKER BE APPLIED TO NEW DOMAINS OR NEW LLMS WITHOUT FINETUNING? 

It is essential to evaluate the generalization capability of our TypedThinker framework. We assess it from two aspects: (i) TypedThinker’s performance on new domains; and (ii) other LLMs’ performance after facilitated with our finetuned meta-thinker and the explicit memory. 

**TypedThinker generalizes well to the unseen domain.** We use a new propositional logic benchmark Contexthub (Hua et al., 2024) for evaluation. It is a recently released dataset, which has never been seen by Mistral and LLaMA3 models during the pre-training. It contains problems from 12 categories with 4 levels of difficulty. We select the difficulty of level 4 to test the complex logic reasoning capabilities. We use the experiences collected from LogiQA as the explicit memory. The meta-thinker and the reasoner are fine-tuned on four training benchmarks. The results in Table 4 show that TypedThinker outperforms other baselines on this unseen domain as well, indicating that it can generalize well to new domains. One interesting thing is Mistral 7B baselines significantly outperform LLaMA3 8B on this benchmark and its superior capabilities make it benefit more from our TypedThinker. 

**Facilitating LLMs with TypedThinker makes them more powerful.** Our TypedThinker framework is orthogonal to the backbone LLMs and can be adapted to new LLMs. There are two ways to use a new LLM in the TypedThinker framework: one is to conduct the self-training process, like the two LLMs used in our main experiments (Mistral 7B and LLaMA3 8B); the other is to use our fine-tuned meta-thinker for reasoning type selection and the explicit memory for retrieval while using the new LLM as the reasoner without finetuning. The first way can make the LLM more powerful (as shown in the performance comparison between TypedThinker and TypedThinker w/o Finetuned Reasoner in Table 2), but the latter one is more flexible. Here we use the second way to evaluate the direct transferability to new LLMs. We choose one of the most powerful LLMs GPT-4o and one math-specific 7B model MetaMath (Yu et al., 2023). MetaMath is a Mistral-7B-based model trained with more than 400k synthesized math data distilled from GPT-3.5-Turbo. We randomly sample 100 examples from each benchmark for GPT-4o. For MetaMath, we use the whole test set. The results are shown in Table 5 and Table 6. Compared with Mistral 7B in Table 1, the high-quality and large scale of synthesized data from GPT-3.5-Turbo enhances MetaMath’s capabilities in math problems. TypedThinker can further improve its performance by reasoning type selection and explicit memory. Meanwhile, although the superior performance of GPT-4o on two math datasets leaves little space for improvement, the results on logic benchmarks (LogiQA and BBH) demonstrate that the meta-thinker trained with the small 7B model also enhances its performance. These findings confirm that our approach is not only effective in improving smaller LLMs but also transferable to larger models, further validating the generalization capability of TypedThinker. 

Table 5: GPT-4o’s performance is improved with our metareasoner. We use the finetuned Mistral 7B meta-reasoner to predict the reasoning type and replace the fine-tuned reasoner with GPT-4o. 

Table 6: TypedThinker can also enhance the performance of the math-specific 7B model such as MetaMath. 

|LogicQA BBH GSM8k MATH<br>GPT-4o<br>0.76<br>0.84<br>0.97<br>0.89<br>+ SC @ 5<br>0.80<br>0.85<br>**0.98**<br>0.90<br>TypedThinker<br>0.80<br>0.86<br>0.95<br>0.88<br>+ SC @5<br>**0.81**<br>**0.90**<br>0.96<br>**0.91**|GSM8k MATH|
|---|---|
||MetaMath<br>0.690<br>0.209<br>+ SC @ 5<br>0.704<br>0.220|
||TypedThinker<br>0.696<br>0.220<br>+ SC @ 5<br>**0.736**<br>**0.246**|



4.6 CASE STUDY 

Here is one example of TypedThinker on the LogiQA benchmark in Table 7. This problem states a phenomenon that a higher altitude leads to a lower atmospheric pressure. Based on this observation, it is easy for humans to use inductive reasoning and get a general conclusion about the inverse cause-and-effect relationship. It is also natural for humans to use analogical reasoning and find the most similar options. The meta-thinker gives the highest effectiveness score for inductive reasoning, which is then chosen as the optimal reasoning type _f[∗]_ = inductive. Effectiveness scores are all larger than 0, so the effective set is all reasoning types. The reasoner gets the correct answer 

9 

for deductive and inductive reasoning while doing wrong on the other reasoning types. If we use the majority vote over five answers, (A) and (C) will have the same votes, indicating that there is a 50% chance to be correct[5] . However, with the effectiveness score predicted by the meta-thinker, TypedThinker can get the correct answer either by applying the optimal reasoning type or using the weighted vote on the four answers. Besides, without a specific reasoning type (which is ‘Empty’), the model cannot arrive at the correct answer. This shows the limitation of the common few-shot baselines. It shows that TypedThinker improves the reasoning performance by the introduction of diverse reasoning types and the capability of selecting the appropriate type to apply 

Table 7: One example from LogiQA. The correct answer and the reasoning type with the highest effectiveness score are underlined. **MoR** is the few-shot MoR baselines, which use the majority votes among all reasoning types. 

|Problem<br>**The higher the altitude, the smaller the atmospheric pressure. Because the**<br>**altitude of Lanzhou is higher than that of Tianjin, the atmospheric pressure of**<br>**Lanzhou is lower than that of Tianjin. Which of the following reasoning is most**<br>**similar?**<br>(A) In a highly competitive market, the better the product quality and the more<br>advertising investment, the greater the product sales. Company A invests more<br>money in advertising than Company B. So company A sells more products than<br>company B<br>(B) The older a person is, the more mature he becomes. Lao Zhang is older than his<br>son, so Lao Zhang is more mature than his son<br>(C) The older a tree is, the more rings it has. The age of the locust tree in Lao<br>Zhang’s yard is older than that of Lao Li’s family, so the locust tree of Lao Zhang’s<br>family has more rings than Lao Li’s<br>(D) The greater the vocabulary of a language, the more diffcult it is to learn. English<br>is harder to learn than Italian, so English has a larger vocabulary than Italian|Problem<br>**The higher the altitude, the smaller the atmospheric pressure. Because the**<br>**altitude of Lanzhou is higher than that of Tianjin, the atmospheric pressure of**<br>**Lanzhou is lower than that of Tianjin. Which of the following reasoning is most**<br>**similar?**<br>(A) In a highly competitive market, the better the product quality and the more<br>advertising investment, the greater the product sales. Company A invests more<br>money in advertising than Company B. So company A sells more products than<br>company B<br>(B) The older a person is, the more mature he becomes. Lao Zhang is older than his<br>son, so Lao Zhang is more mature than his son<br>(C) The older a tree is, the more rings it has. The age of the locust tree in Lao<br>Zhang’s yard is older than that of Lao Li’s family, so the locust tree of Lao Zhang’s<br>family has more rings than Lao Li’s<br>(D) The greater the vocabulary of a language, the more diffcult it is to learn. English<br>is harder to learn than Italian, so English has a larger vocabulary than Italian|
|---|---|
|Ground Truth<br>(C)||
|Predicted scores<br>Deductive: 0.4;Inductive: 0.5<br>; Analogical: 0.4; Abductive: 0.4; Empty: 0.4<br>and their answers<br>Deductive: (C); Inductive: (C); Analogical:(A); Abductive: NULL; Empty: (A)||
|Model Output<br>**MoR**: (A);**TypedThinker with**_f ∗_(C)<br>;**TypedThinker with**_F_: (C)||



## 5 CONCLUSION AND LIMITATION 

In this paper, we investigate how reasoning types diversify LLMs’ thinking and propose a novel framework TypedThinker to incorporate different reasoning types into problem-solving. TypedThinker is inspired by human cognition processes during reasoning: it learns an implicit policy to select the appropriate reasoning types with the meta-thinker and to conduct the selected type of reasoning with the reasoner. It also maintains an explicit memory to retrieve experiences to aid reasoning. TypedThinker optimizes its implicit policy and the explicit memory with its own successful experiences during the self-training process. The results show that TypedThinker enhances the reasoning capabilities of Mistral 7B and LLaMA3 8B on four benchmarks. Furthermore, TypedThinker shows good generalization capabilities in new domains and models. It can also improve GPT-4o’s performance by the effective reasoning type selection. 

Despite the promising results, TypedThinker has several limitations that need further investigation. Firstly, one problem may require different reasoning types at different steps, and applying one sole reasoning type can hardly find a correct solution. In that case, dividing the problems into multiple reasoning steps, and applying TypedThinker for each step could make the reasoning more diverse and effective. Additionally, this paper mainly focuses on logical and mathematical benchmarks. Expanding to a broader range of tasks, such as commonsense reasoning or creative problem-solving, could deepen the understanding of the role of reasoning types in various problems and provide a more comprehensive assessment of TypedThinker’s capabilities. 

> 5In our implementation, answers with the same votes are ranked based on their alphabetical order, so (A) will be chosen in this case. 

10 

## REFERENCES 

- Renat Aksitov, Sobhan Miryoosefi, Zonglin Li, Daliang Li, Sheila Babayan, Kavya Kopparapu, Zachary Fisher, Ruiqi Guo, Sushant Prakash, Pranesh Srinivasan, et al. Rest meets react: Selfimprovement for multi-step reasoning llm agent. In _ICLR 2024 Workshop on Large Language Model (LLM) Agents_ . 

- Afra Feyza Akyürek, Ekin Akyürek, Ashwin Kalyan, Peter Clark, Derry Wijaya, and Niket Tandon. 2023. Rl4f: Generating natural language feedback with reinforcement learning for repairing model outputs. In _Annual Meeting of the Association of Computational Linguistics 2023_ , pages 7716–7733. Association for Computational Linguistics (ACL). 

- Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. 2022. Constitutional ai: Harmlessness from ai feedback. _arXiv preprint arXiv:2212.08073_ . 

- Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, et al. 2023. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. In _Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 675–718. 

## Paul Bartha. 2013. Analogy and analogical reasoning. 

- Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. 2024. Graph of thoughts: Solving elaborate problems with large language models. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 17682–17690. 

- Hugo Bronkhorst, Gerrit Roorda, Cor J. M. Suhre, and Martin J. Goedhart. 2020. Logical reasoning in formal and everyday reasoning tasks. _International Journal of Science and Mathematics Education_ , 18:1673–1694. 

- Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. 2024. Large language monkeys: Scaling inference compute with repeated sampling. _arXiv preprint arXiv:2407.21787_ . 

- Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, and Jeffrey Wu. 2024. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. In _Forty-first International Conference on Machine Learning_ . 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ . 

Craig DeLancey. 2017. _A concise introduction to logic_ . Open SUNY Textbooks. 

John Dougrez-Lewis, Mahmud Elahi Akhter, Yulan He, and Maria Liakata. 2024. Assessing the reasoning abilities of chatgpt in the context of claim verification. _arXiv preprint arXiv:2402.10735_ . 

Igor Douven. 2011. Abduction. 

- Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2023. Improving factuality and reasoning in language models through multiagent debate. _arXiv preprint arXiv:2305.14325_ . 

- Peter A Flach and Antonis C Kakas. 2000. Abductive and inductive reasoning: background and issues. _Abduction and induction: Essays on their relation and integration_ , pages 1–27. 

- Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. 2023. Complexity-based prompting for multi-step reasoning. In _The Eleventh International Conference on Learning Representations_ . 

11 

- Çaglar Gülçehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. 2023. Reinforced self-training (rest) for language modeling. _CoRR_ . 

- Diane F Halpern. 2014. _Critical thinking across the curriculum: A brief edition of thought & knowledge_ . Routledge. 

- Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, and Zhiting Hu. 2023. Reasoning with language model is planning with world model. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 8154–8173, Singapore. Association for Computational Linguistics. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. _NeurIPS_ . 

- Wenyue Hua, Kaijie Zhu, Lingyao Li, Lizhou Fan, Shuhang Lin, Mingyu Jin, Haochen Xue, Zelong Li, JinDong Wang, and Yongfeng Zhang. 2024. Disentangling logic: The role of context in large language model reasoning capabilities. _arXiv preprint arXiv:2406.02787_ . 

- Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2023. Large language models can self-improve. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 1051–1068, Singapore. Association for Computational Linguistics. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. _arXiv preprint arXiv:2310.06825_ . 

- Phil Johnson-Laird. 2010. Deductive reasoning. _Wiley Interdisciplinary Reviews: Cognitive Science_ , 1(1):8–17. 

- Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. 2024. Training language models to self-correct via reinforcement learning. _arXiv preprint arXiv:2409.12917_ . 

- Rémi Leblond et al. 2023. Alphacode 2 technical report. Technical report, DeepMind. 

- V Levenshtein. 1966. Binary codes capable of correcting deletions, insertions, and reversals. _Proceedings of the Soviet physics doklady_ . 

- Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. 2022. Competition-level code generation with alphacode. _Science_ , 378(6624):1092–1097. 

- Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In _The Twelfth International Conference on Learning Representations_ . 

- Hanmeng Liu, Jian Liu, Leyang Cui, Zhiyang Teng, Nan Duan, Ming Zhou, and Yue Zhang. 2023a. Logiqa 2.0—an improved dataset for logical reasoning in natural language understanding. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 31:2947–2962. 

- Jiacheng Liu, Andrew Cohen, Ramakanth Pasunuru, Yejin Choi, Hannaneh Hajishirzi, and Asli Celikyilmaz. 2023b. Making ppo even better: Value-guided monte-carlo tree search decoding. _arXiv preprint arXiv:2309.15028_ . 

- Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. 2021. Logiqa: a challenge dataset for machine reading comprehension with logical reasoning. In _Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence_ , IJCAI’20. 

- Man Luo, Shrinidhi Kumbhar, Mihir Parmar, Neeraj Varshney, Pratyay Banerjee, Somak Aditya, Chitta Baral, et al. 2023. Towards logiglue: A brief survey and a benchmark for analyzing logical reasoning capabilities of language models. _arXiv preprint arXiv:2310.00836_ . 

12 

OpenAI. 2023. Gpt-4 technical report. 

Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, et al. 2023. Check your facts and try again: Improving large language models with external knowledge and automated feedback. _arXiv preprint arXiv:2302.12813_ . 

- Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ , KDD ’20, page 3505–3506, New York, NY, USA. Association for Computing Machinery. 

- Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP)_ , pages 3982–3992, Hong Kong, China. Association for Computational Linguistics. 

- Noah Shinn, Beck Labash, and Ashwin Gopinath. 2023. Reflexion: an autonomous agent with dynamic memory and self-reflection. _arXiv preprint arXiv:2303.11366_ . 

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _arXiv preprint arXiv:2206.04615_ . 

- Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, and Jason Wei. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. _arXiv preprint arXiv:2210.09261_ . 

- Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. 2024. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. _arXiv preprint arXiv:2402.10176_ . 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ . 

- Danqing Wang and Lei Li. 2023. Learn from mistakes through cooperative interaction with study assistant. _The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP) 2023_ . 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ . 

- Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 13484–13508, Toronto, Canada. Association for Computational Linguistics. 

- Peter Cathcart Wason and Philip Nicholas Johnson-Laird. 1972. _Psychology of reasoning: Structure and content_ , volume 86. Harvard University Press. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed H Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. In _Advances in Neural Information Processing Systems_ . 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2019. Huggingface’s transformers: State-of-the-art natural language processing. _arXiv preprint arXiv:1910.03771_ . 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of Thoughts: Deliberate problem solving with large language models. 

13 

- Junchi Yu, Ran He, and Zhitao Ying. 2024. THOUGHT PROPAGATION: AN ANALOGICAL APPROACH TO COMPLEX REASONING WITH LARGE LANGUAGE MODELS. In _The Twelfth International Conference on Learning Representations_ . 

- Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. Metamath: Bootstrap your own mathematical questions for large language models. _arXiv preprint arXiv:2309.12284_ . 

- Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. 2024. Self-rewarding language models. _arXiv preprint arXiv:2401.10020_ . 

- Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. 2024. Progressive-hint prompting improves reasoning in large language models. In _AI for Math Workshop @ ICML 2024_ . 

14 

## A APPENDIX 

## A.1 PROMPT 

We introduce the simple and practical definition of four reasoning types in Table 8. Table 11 lists the few-shot examples of each reasoning type. The full few-shot examples can be found in the supplementary materials. We use the same few-shot examples for the logical problems and create another set of examples for the mathematics problems. 

Table 8: Description of different reasoning types. We give informal definitions that are easy to follow and illustrate simple examples for each reasoning type. 

|**Type**|**Defnition**|**Example**|
|---|---|---|
|**Deduction**|_Deduce conclusion based on the gen-_|From the premises ‘all frogs are amphibians‘ and|
||_eral rules and premise._|‘no cats are amphibians’, we can infer the conclu-|
|||sion ‘no cats are frogs’|
|**Induction**|_Make broad generalizations from spe-_|Starting from the empirical observation that ‘all|
||_cifc observations._|ravens I have seen so far are black’, inductive rea-|
|||soning can be used to infer that ‘all ravens are|
|||black’|
|**Abduction**|_Assume one candidate is correct and_|Guess that it has rained to explain that the streets|
||_check whether it meets the condition in_|are wet. A tsunami could also explain why the|
||_the problem._|streets are wet but this is usually not the best ex-|
|||planation.|
|**Analogy**|_Retrieve several relevant information_|Infer information about humans from medical ex-|
||_and draw the conclusion of this problem_|periments on animals: (1) rats are similar to hu-|
||_based on the similarity._|mans; (2) birth control pills affect the brain devel-|
|||opment of rats; (3) therefore they may also affect|
|||the brain development of humans.|



The prompt used by the meta-thinker is: 

_Given the question below, please identify the type of reasoning required to provide a solution. You may choose the following reasoning types: Deductive, Inductive, Analogical, Abductive Reasoning, or None. None indicates that no specific reasoning type is needed for this problem. Please assign an effectiveness score for each reasoning type from 0 to 1, where 0 represents no effective and 1 represents full effective. Please return the reasoning types and their corresponding effectiveness scores in the JSON format._ 

_For instance, if you think the question can be solved using both deductive and inductive reasoning, with an effectiveness of 0.5 for deductive reasoning and 0.3 for inductive reasoning, you should return: [{"ReasoningType": "Deductive", "Effectiveness": 0.5},{"ReasoningType": "Inductive", "Effectiveness": 0.3},{"ReasoningType": "Analogical", "Effectiveness": 0},{"ReasoningType": "Abductive", "Effectiveness": 0}, {"ReasoningType": "None", "Effectiveness": 0}]_ . 

The prompt used by the reasoner is listed below. The definition is based on Table 8. 

_Use [fk] reasoning to solve the given question. [fk] reasoning is [definition]._ 

## A.2 DATASET PROCESSING 

The dataset statistics of the four benchmarks are detailed in Table 9. For multiple-choice questions, we calculate accuracy using the exact match criterion. For mathematics problems, we compare the model’s response with the ground truth using mathematical equality. 

LogiQA (Liu et al., 2021; 2023a) is a multi-choice understanding benchmark for logical reasoning. It follows the definition of DeLancey (2017) and categorizes the problems into categorical reasoning, 

15 

Table 9: Logical benchmarks and mathematic benchmarks we used in this paper. We follow the standard train/test split on LogiQA and follow the split in Toshniwal et al. (2024) for GSM8k and MATH. For BBH, we randomly split the dataset. The synthesized data is described in Section 3.1. BBH includes 16 tasks while MATH includes math problems of 7 categories. Policy indicates the data used to train the meta-reasoner and SFT indicates the instruction-following in reasoner finetuning. 

|**Benchmark**<br>**Empirical Dataset**|**Benchmark**<br>**Empirical Dataset**|
|---|---|
|**# Task**<br>**# Train**<br>**# Val**<br>**# Test**<br>**# Total**<br>**# Meta-thinker**<br>**# Reasoner**||
|||
|LogiQA<br>1<br>3757<br>500<br>511<br>4768<br>BBH<br>16<br>1904<br>320<br>1600<br>3824<br>GSM8k<br>1<br>6473<br>1000<br>1319<br>8792<br>Math<br>7<br>6500<br>1000<br>5000<br>12500|~2k<br>~6k<br>~1k<br>~3.5k<br>~4k<br>~4k<br>~1k<br>~1k|



sufficient conditional reasoning, necessary conditional reasoning, disjunctive reasoning, and conjunctive reasoning. These reasoning categories are not orthogonal and one problem can belong to multiple categories. We follow the standard training/validation split and only keep examples with more than 3 reasoning categories. This makes the problem more diverse and difficult to solve. We take the validation set as the test set and randomly select 500 examples from the training set for validation. 

BBH (Suzgun et al., 2022) is a set of hard problems borrowed from Big Bench (Srivastava et al., 2022). They are also formatted as multi-choice problems. We pick the English tasks with more than 2 options, resulting in 16 tasks: date understanding disambiguation qa, geometric shapes, hyperbaton, logical deduction three, logical deduction five, logical deduction seven, movie recommendation, penguins in a table, reasoning color, ruin names, snarks, temporal sequences, tracking shuffled three, tracking shuffled five, and tracking shuffled seven. For each task, we randomly select 100 examples as the test set and 20 examples as the validation. The rest are used as training examples. 

For GSM8k and MATH, we follow Toshniwal et al. (2024) to process the dataset. 

## A.3 ANALYSIS OF REASONING TYPES 

**Accuracy of Reasoning Types** We calculate the accuracy for each reasoning type on our empirical dataset _D_ , shown in Figure 6. We can find that on LogiQA and MATH, the accuracy of different reasoning types is similar. However, deductive and analogical reasoning outperform the other two on BBH while inductive and abductive reasoning are more effective. The results illustrate that after our carefully designed demonstration for each reasoning type, LLM’s capabilities in other reasoning types achieve comparable performance with deductive reasoning. This ensures the quality and the balance of our collected dataset on each reasoning type. 

Comparing Figure 1 and Figure 6, we can see that if correctly selected, the specific reasoning type can enhance the model performance by handling problems that cannot be solved by other reasoning types, such as inductive on MATH. However, the unsuitable reasoning type can also mislead the model, leading to poor performance. 

**Diversity of Reasoning Types** To further verify whether the reasoning types can make the solutions more diverse, we compare the diversity between solutions under different sampling settings in Table 10. We use Levenshtein Distance (Levenshtein, 1966) and the n-gram overlaps between sentences to evaluate diversity. Specifically, for _K_ generations _G_ = _{g_ 1 _, · · · , gK}_ of the same problem, we calculate the distance between each pair and normalize them with the sentence length. Then the average distance over these paired results is used as the distance of these _K_ generations. If we denote the normalized Levenshtein Distance function as _f_ ld, this process can be represented as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_TypedThinker_Typed_Thinking_Improves_Large_Language_Model_Reasoning_images/iclr_2025_TypedThinker_Typed_Thinking_Improves_Large_Language_Model_Reasoning.pdf-0016-09.png)


The calculation of the n-gram overlap is defined in the same way. For each setting, we present the average score over the problems in the test set in Table 10. A larger Levenshtein distance and a smaller overlap indicate a more diverse solution set. The zero-shot setting does not include examples in the prompt, and the zero-shot setting + types only include the definition of the reasoning type (as listed in Table 8). The few-shot setting has 5 examples, and the few-shot setting with types has 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_TypedThinker_Typed_Thinking_Improves_Large_Language_Model_Reasoning_images/iclr_2025_TypedThinker_Typed_Thinking_Improves_Large_Language_Model_Reasoning.pdf-0017-00.png)


**----- Start of picture text -----**<br>
Deductive Inductive Analogical Abductive<br>0.60<br>0.50<br>0.40<br>0.30<br>0.20<br>0.10<br>0.00<br>LogiQA BBH GSM8k MATH<br>**----- End of picture text -----**<br>


Figure 6: Accuracy of the solutions on different reasoning types. It indicates that the effectiveness of reasoning types varies in different problems. 

different 6 examples for each type. For zero-shot / few-shot @5, we use repeated sampling with temperature = 1 for 5 times. For zero-shot / few-shot + 5 types, we sample one solution per reasoning type. 

From Table 10, we can see that after adding the reasoning types, the diversity of both zero-shot and few-shot increases significantly. It indicates that the introduction of various reasoning types can make the LLM’s reasoning more diverse. We can also find that in most cases, the few-shot with reasoning types has the highest diversity, while in BBH, the zero-shot setting can benefit more from the reasoning types. 

Table 10: Adding reasoning types can enhance diversity in both zero-shot and few-shot sampling settings. It can significantly increase the distance and reduce the n-gram overlaps between generations. For each setting, we use Mistral 7B to sample 5 solutions with temperature = 1. @5 indicates repeated sampling 5 times, + 5 types indicates sampling one solution per reasoning type. The diversity is averaged over the whole test set. 

|Benchmark|Sampling Setting|Levenshtein Distance|_↑_Unigram overlap|_↓_4-gram overlap_↓_|
|---|---|---|---|---|
||Zero-shot @ 5|0.3043|0.5877|0.5367|
|LogiQA|Zero-shot + 5 types<br>Few-shot @ 5|0.5997<br>0.5729|0.2584<br>0.2316|0.1863<br>0.1226|
||Few-shot + 5 types|**0.6437**|**0.1745**|**0.0773**|
||Zero-shot @ 5|0.5170|0.3104|0.2157|
|BBH|Zero-shot + 5 types<br>Few-shot @ 5|**0.7117**<br>0.5992|**0.1280**<br>0.2239|**0.0495**<br>0.1191|
||Few-shot + 5 types|0.6495|0.1750|0.0764|
||Zero-shot @ 5|0.6242|0.1951|0.0907|
|GSM8k|Zero-shot + 5 types<br>Few-shot @ 5|0.6831<br>0.4977|0.1513<br>0.3117|0.0540<br>0.1761|
||Few-shot + 5 types|**0.7097**|**0.1366**|**0.0476**|
||Zero-shot|0.6726|0.1573|0.0703|
|MATH|Zero-shot + 5 types<br>Few-shot|0.7291<br>0.6588|**0.1116**<br>0.1741|**0.0352**<br>0.0804|
||Few-shot + 5 types|**0.7319**|0.1154|0.0382|



## A.4 IMPACT STATEMENT 

This work will enhance current LLMs with better reasoning capability, which can make them more useful in problem-solving. There might be some potential societal consequences of our work, none of which we feel must be specifically highlighted here. However, it might be misused as we release all our code and data for reproduction. We will try our best to avoid the potential misuse. 

17 

Table 11: Examples of four reasoning types. 

|**Type**<br>**Query**<br>**Example**|**Type**<br>**Query**<br>**Example**|**Type**<br>**Query**<br>**Example**|
|---|---|---|
|**Deduction**<br>Alice, Bob, and Claire are dancers at<br>a square dance. At the start of a song,<br>they each have a partner: Alice is with<br>Lola, Bob is with Rodrigo, and Claire<br>is with Patrick. Throughout the song,<br>Alice and Bob switch partners; Claire<br>and Bob switch; Finally, Bob and Alice<br>switch. At the end of the dance, Alice<br>is dancing with Options:<br>(A) Lola (B) Rodrigo (C) Patrick<br>(0) At the start: Alice: Lola, Bob: Rodrigo, Claire:<br>Patrick.(1) Alice and Bob switch : Alice: Rodrigo,<br>Bob: Lola, Claire: Patrick. (2) Claire and Bob<br>switch : Alice: Rodrigo, Bob: Patrick, Claire:<br>Lola. (3) Bob and Alice switch : Alice: Patrick,<br>Bob: Rodrigo, Claire: Lola. At the end of the<br>dance,Alice is dancing with Patrick. So the answer<br>is (_C_)<br>.|||
|**Induction**<br>Students who told a lie overestimated<br>how many people could detect it.<br>Volleyball players performing poorly<br>thought teammates noticed more than<br>they actually did. A student wearing a<br>funny T-shirt expected everyone to no-<br>tice, but only a few did. Which option<br>best illustrates the statements above?<br>(A) People overestimate how often oth-<br>ers notice their appearance and behav-<br>ior.<br>(B) People rarely notice the appearance<br>or behavior of others.<br>(C) We are less observant of others’ ap-<br>pearance and behavior than we think.<br>(D) People are less aware of their ap-<br>pearance and behavior than others are."<br>By examining each of the scenarios mentioned, we<br>notice all these examples indicate that individuals<br>overestimate the level and extent of attention their<br>actions, appearances, or behaviors receive from<br>others. This consistency across different contexts<br>illustrates a broader psychological phenomenon.<br>So the answer is (_A_)<br>.|||
|**Abduction**<br>John is 24 years younger than his dad.<br>The sum of their ages is 68 years. How<br>many years old is John?<br>Retrieval: Question: Lisa is 10 years younger than<br>her mom. The sum of their ages is 70 years. How<br>old is Lisa?<br>Answer: Lisa is 30 years old and her mom is 40<br>years old.<br>These are solved using the same approach as the<br>problem about John and his dad’s ages, i.e., setting<br>up two equations based on the information given<br>and then solving for the two variables representing<br>the ages. Therefore, for the given question, John is|||
||22|years old.|
|**Analogy**<br>The integer m is between 30 and 80 and<br>is a multiple of 6. When m is divided by<br>8, the remainder is 2. Similarly, when<br>m is divided by 5, the remainder is 2.<br>What is the value of m?<br>To solve this problem using abductive reasoning,<br>we assume that one possible value of m exists that<br>abides by the constraints and check if this assump-<br>tion holds.1. Fi rst, flter values of m that are mul-<br>tiples of 6 between 30 and 80. 2. Next, apply the<br>condition that when m is divided by 8, the remain-<br>der is 2. Only 42, 66 ft this condition.3. Apply the<br>third condition, that when divided by 5, m should<br>leave a remainder of 2. Testing the applicable val-<br>ues so far and fnd 42 meets the requirement. So<br>the answer is 42<br>.|||



18 

