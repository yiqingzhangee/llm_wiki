Preprint. Under review. 

## - THE LABYRINTH OF LINKS: NAVIGATING THE ASSO - CIATIVE MAZE OF MULTI MODAL LLMS 

**Hong Li, Nanxi Li, Yuanjie Chen, Jianbin Zhu, Qinlu Guo, Cewu Lu, Yong-Lu Li** _[∗]_ Shanghai Jiao Tong University 

_{_ hong ~~l~~ i,andyc ~~0~~ 3,cyj2003,bin ~~p~~ ig,guoqinlu,lucewu,yonglu ~~l~~ i _}_ @sjtu.edu.cn 

## ABSTRACT 

Multi-modal Large Language Models (MLLMs) have exhibited impressive capability. However, recently many deficiencies of MLLMs have been found compared to human intelligence, _e.g._ , hallucination. To drive the MLLMs study, the community dedicated efforts to building larger benchmarks with complex tasks. In this paper, we propose benchmarking an essential but usually overlooked intelligence: **association** , a human’s basic capability to link observation and prior practice memory. To comprehensively investigate MLLM’s performance on the association, we formulate the association task and devise a standard benchmark based on adjective and verb semantic concepts. Instead of costly data annotation and curation, we propose a convenient **annotation-free** construction method transforming the general dataset for our association tasks. Simultaneously, we devise a rigorous data refinement process to eliminate confusion in the raw dataset. Building on this database, we establish three levels of association tasks: singlestep, synchronous, and asynchronous associations. Moreover, we conduct a comprehensive investigation into the MLLMs’ zero-shot association capabilities, addressing multiple dimensions, including three distinct memory strategies, both open-source and closed-source MLLMs, cutting-edge Mixture-of-Experts (MoE) models, and the involvement of human experts. Our systematic investigation shows that current open-source MLLMs consistently exhibit poor capability in our association tasks, even the currently state-of-the-art GPT-4V(vision) also has a significant gap compared to humans. We believe our benchmark would pave the way for future MLLM studies. _Our data and code are available at:_ https://mvig-rhos.com/llm_inception. 

## 1 INTRODUCTION 

Multi-modal Large Language Models (MLLMs) have recently made significant breakthroughs in perceiving diverse modality input and solving a broad range of tasks Zhang et al. (2024a); Carolan et al. (2024). As GPT-4V(ision) Achiam et al. (2023) and Gemini Team et al. (2023); Reid et al. (2024) address challenges that researchers have been exploring for a considerable period. Subsequently, numerous researchers have developed diverse open-source MLLMs AI et al. (2024); Bai et al. (2023b); Wang et al. (2024b); Dong et al. (2024); Liu et al. (2023a); Li et al. (2024a); Ye et al. (2023; 2024). These models usually use the Large Language Model (LLM) as the core component and expand to multi-modal with a specific module Yin et al. (2023) that transfers multi-modal tokens into language tokens, achieving alignment between different modality encoders. 

MLLMs demonstrated ability in visual reasoning, which requires understanding the input query and then making judgments based on the visual content. Much prior work has been dedicated to evaluating the levels of their visual reasoning capabilities. However, to the best of our knowledge, how to evaluate the association ability of MLLMs is overlooked. **Association** is one of the most fundamental capabilities of human intelligence. It provides the foundation for creative thinking and helps humans to summarise scattered information into structured knowledge to enhance memory and understanding Mednick (1962); Ausubel (1963), perception, rule discovery, embodied AI, _etc_ . 

> _∗_ Correspondence to: Yong-Lu Li _<_ yonglu ~~l~~ i@sjtu.edu.cn _>_ . 

1 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0002-01.png)


**----- Start of picture text -----**<br>
b) Practical association task<br>Memory … Association step Human<br>Instruction: {Assocition Question Instruction} {memory}. Image1Query imageImage2 Image2. ✓ 182.3 | 140.2<br>Current memory: rusty Deepen memory Deduction step Close-source<br>Metal:  Rusty:  or forget Metal. Instruction: {Evidence Question Instruction} 1.5 Flash<br>Updated  memory: rusty; metal Image1 Image2 88.3 | 27.8 59.0 | 15.4<br>Association step<br>Memory Association: Or the airplaneThe bus this morning? we took? isn’t the decoration Look over there,  Question Instruction: {Assocition Question Instruction} {memory}. Image1Query imageImage2 Image1. ✓ Open-source MoE<br>Deductionall : plentifulThey are? what we see today?more plentiful than  Continuous Multi-step Association… 53 | 5.99<br>Association step<br>Instruction: {memory}.  Query image 43.8 | 4.43 36.8 | 3.94 18 | 2.10<br>{Assocition Question Instruction} Image1 Image2 Image1. ✘ LLaVA-OneVision mPLUG-Owl3<br>Observation c) MLLM Power Ranking<br>a) Association question  (Max | Mean Step)<br>**----- End of picture text -----**<br>


Figure 1: Our insight and proposed association task. a) Association is always in our lives. b) Our proposed practical association task. c) The performance of current MLLMs and human experts. The results demonstrate a significant gap between current MLLMs and humans in association tasks. 

In this paper, we aim to devise a standard benchmark to evaluate MLLM’s capability in association tasks. As shown in Figure 1a, we always connect observations with previous practice memory. This association is usually the semantic concept shared between different objects, representing common features in general dataset samples. For instance, the “painted airplane” and “painted bus” share the same attribute of “painted”, which can be seen as an association chain in the association task. Next, it is natural to ask _can we design a general method to build an association benchmark based on existing datasets without too much extra effort?_ To this end, we propose an annotation-free association construction method, which easily transfers the general dataset for the association task. Thus, as in Figure 1b, we devise a standard association benchmark using the object’s concepts as an association chain. Given the benchmark, we comprehensively investigate MLLMs’ capabilities in association tasks across multiple memory strategies, multiple open-source and closed-source MLLMs, as well as cutting-edge Mixture-of-Experts (MoE) and human expert testers. 

Extensive experiments demonstrate that current open-source MLLMs consistently have a weak ability in association tasks, and even the closed-source GPT4-V Achiam et al. (2023) and Gemini-1.5Flash Reid et al. (2024) also have a huge gap from human performance. As Figure 1c shows, the best closed-source Gemini-1.5-Flash attains an average mean-step of 27 _._ 8, while humans attain 140 _._ 2 in the attribute of adjective concepts. As a complementary of existing benchmarks, we believe our benchmark and baselines will pave the way for human-like intelligent agents. 

In conclusion, our main contributions are: **1)** To evaluate the association ability of MLLMs, we propose a new multi-modal task and a corresponding convenient annotation-free construction method that can transform the general dataset for association tasks within various semantic concepts. **2)** We devise a standard association benchmark based on adjective semantic concepts within objects and verb semantic concepts within actions and further evaluate MLLMs’ performance through extensive experiments. **3)** We systematically investigate MLLMs’ capability on association tasks through tuning-free methods and propose potential future directions for association tasks. 

## 2 RELATED WORKS 

**Multi-Modal Visual Reasoning.** Various works are dedicated to understanding and reasoning about the semantic concept between multi-images. Several works, such as Visual Genome Krishna et al. (2017) and Bongard-HOI Jiang et al. (2022), target human-object interaction (HOI) tasks to investigate the visual relationship between different objects. Recent work Zhang et al. (2024b) investigates the MLLM’s ability in low-level perception question-answering and description tasks with paired input images. Despite their success in perceiving and understanding multi-images, these methods are confined to single-step evaluation, lacking the investigation of multi-step association ability. 

**Multi-Modal Large Language Model.** There has been a surge of MLLMs in the deep learning community, which effectively use off-the-shell LLMs to support multi-modal inputs and demonstrate promise for zero-shot generalization. For instance, LLaVA Liu et al. (2023a) InstructBLIP Dai et al. (2023), InternLM-XComposer Dong et al. (2024), Qwen-VL Bai et al. (2023a), MiniGPT- 

2 

Preprint. Under review. 

4 Zhu et al. (2023), and mPLUG-Owl2 Ye et al. (2023) make leading in last year according to their powerful visual perception capabilities. More recently, these models Wang et al. (2024b); Li et al. (2024a); Ye et al. (2024) are updated to improve their ability in multi-images or video input. In this work, we mainly evaluated the leading open-source models Qwen2-VL Wang et al. (2024b), mPLUG-Owl3 Ye et al. (2024), and LLaVA-OneVision Li et al. (2024a). 

**Tuning-Free Engineering.** The development of MLLMs has led to a significant improvement in incontext learning Doveh et al. (2024); Wies et al. (2024). Additionally, the use of visual and language prompts has been shown to enhance the performance of MLLMs. Research has indicated that visual cues such as listed numbers Yan et al. (2024) and shapes Shtedritski et al. (2023); Mani et al. (2020) can be effectively understood by MLLMs. Similarly, carefully designed language prompts Liu et al. (2023b), Chain-of-Thought (CoT) Wei et al. (2024), tree-of-mixed-thought (ToMT) Hu et al. (2023), have shown promising results. Common knowledge Shao et al. (2023); Liu et al. (2024) can be seen as an instruction Prior knowledge about how to decisions on a task. As a result, we chose to employ the general prompt-engineer one-shot, CoT, and Common knowledge. 

## 3 ANNOTATION-FREE ASSOCIATION CONSTRUCTION 

For a general dataset with _N_ samples, it can be formed as _{_ ( _xi, yi_ ) _| xi ∈ X, yi ∈{c_ 1 _, . . . , cn}, i_ = 1 _,_ 2 _, . . . , N }_ . Let _xi_ = ( _x[m] i_[1] _, . . . , x[m] i[k]_ ) be a sample with _k_ modalities, more specifically, for a task with individual modality, the sample is described as _xi_ = ( _x[m] i_[1] _,_ ). _yi_ is the annotation for sample _xi_ , which may existed in various granularities. These annotations typically indicate a semantic concept, with object categories as nouns, actions as verbs, as well as attributes, and affordance as adjectives. 

Given the definition, labels represent a subset of concepts present in the given sample. The core of human association, on the other hand, involves identifying the overlapping concepts between newly acquired observations and prior practice memory. Hence, it is intuitive to use annotations from the raw dataset to construct the association chain, which reflects the common concepts shared by two samples, such as the presence of the _“shoot ball”_ action in both samples. In the following, we first introduce the method for generating association chains. We then introduce a method that deduces the evidence of the association to simulate the thought behind the human, which involves predicting the common concepts among the samples that constitute the association. 

**Generating Semantic Concept Association Chain.** For a general dataset _{_ ( _xi, yi_ ) _| xi ∈ X, yi ∈ {c_ 1 _, . . . , cn}, i_ = 1 _,_ 2 _, . . . , N }_ , we randomly select two samples to form a new sample pair for association tasks. If the selected samples have _identical labels or share at least one common label_ , we assign a corresponding label of 1; otherwise, we assign a label of 0. Hence, we get the new association dataset as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0003-07.png)


In this way, for each sample _xi_ in the original dataset, we construct a positive association set with _K_ positive samples, _**x**_[+] _i_ = [ _p_[1] _i[, . . . , p][K] i_[]][,][where][at][least][one][potential][association][concept][exists.] Additionally, we devise another negative association set with _L_ samples, _**x**[−] i_ = [ _qi_[1] _[, . . . , q] i[L]_[]][,][in] which no association chain is present. 

**Deducting the Evidence of Association Chain.** It is crucial to deduce the evidence in the association task, which simulates the human reflection process. This evidence then guides and directs the decision-making in the subsequent steps. To realize this, we collected the full set of shared concepts within all possible positive set _**x**_[+] _i_[as] _[C]_[ˆ][=][ �] _[N] i_ =1 _[{][z][ij][|][ z][ij]_[=] _[ y][i][ ∩][y][j][,]_[ 1] _[ ≤][i < j][≤][N][}][ .]_[ In this] setting, the dataset deducting the evidence of the association chain is depicted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0003-10.png)


where _p[k] i_[represents][the][correctly][predicted][positive][sample][at][the][stage][of][chain][association,][and] _**s** i_ = _{s_[1] _i[, . . . , s][R] i[}]_[ denote the] _[ R]_[ common concepts.] 

3 

Preprint. Under review. 

**Evaluation of the Association Task.** Based on the above definition, we can easily construct the association task. First, we randomly take one sample, _x_ query, as the initial query image. For _t_ -th step, We then randomly choose candidate samples _p[t] i_[and] _[ q] i[t]_[from the positive] _**[ x]**_[+] query[and negative] _**[ x]**[−]_ query association sets, respectively. Next, we use the MLLMs to determine which candidate samples share common concepts with the query image, the output of _t_ -th step described as _ot_ . If the model makes the correct decision, _ot_ = _p[t] i_[,][we][set][the][correctly][predicted][sample][as][the][new][query][image][and] repeatedly select candidate samples and make decisions. Otherwise, we calculate the maximum association steps and exit. The above description can be depicted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0004-02.png)


where _x_ query _∈{x_ 1 _, . . . , xN }_ and _F_ represents the forward computation process of MLLMs. step is the computation of the individual association step. Furthermore, after each association step, we will deduce the evidence of the association chain, which will integrated into subsequent associations. 

## 4 ASSOCIATION BENCHMARK 

In this section, we introduce a benchmark based on the semantic concepts of adjectives and verbs, _i.e._ , object attributes and affordances, and human actions. Specifically, we utilize the annotationfree construction method proposed in Section 3 on the Object Concept Learning (OCL) Li et al. (2023) to generate an attribute and affordance association datasets, and on the Pangea Li et al. (2024b) to generate action association dataset[1] . In the following, we comprehensively investigate the association ability from the single-step association, synchronous association, and asynchronous association settings as shown in Figure 2. 

## 4.1 CONSTRUCTING ASSOCIATION TASK 

## 4.1.1 SINGLE-STEP ASSOCIATION 

The association refers to the link between the current observation and prior practice memory (Figure 2a). A single-step association represents one phase within a broader association task, where memory is indispensable in decision-making. In this case, we include the correct memory to simulate prior practice, guiding the decision-making process. In the experiment, we compute the association and deduction success ratio as the main single-step association metric (Section 5.1.2). 

## 4.1.2 SYNCHRONOUS ASSOCIATION 

Synchronous association, where each step adheres to the same principle throughout the entire process, is a core capability of human intelligence (Figure 2b). With this ability, humans can gradually unveil the underlying rules of the task and reduce the likelihood of errors. Utilizing our constructed association dataset, we take different input-paired samples with the same association concepts ˆ _ct_ to evaluate the synchronous association. In this setting, the association dataset can be depicted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0004-11.png)


It is worth noting that current MLLMs lack memory during the inference stage, relying solely on the input samples. To address this problem, we introduce a memory base to imitate the human’s memory in the synchronous association. Specifically, we transfer the inference process into the memory base after each step. The updated memory and the input sample are then used for the next step. In the experiment, we compute the Max _|_ Mean steps metric to evaluate the model (Section 5.1.2). 

## 4.1.3 ASYNCHRONOUS ASSOCIATION 

When there are multiple principles in games, the underlying principle will gradually change as the game progresses. For example, the first two steps with “metal” as the chain, then “furry” and again 

1To further demonstrate the capability on verb concept, we further implemented on action HMDB Kuehne et al. (2011) datasets. The results are included in the section D of the supplementary material. 

4 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0005-01.png)


**----- Start of picture text -----**<br>
b) Synchronous Association<br>a) Single-step Association<br>✘ ✘ ✘<br>fresh fresh<br>Query Image<br>fresh fresh fresh<br>✘ ✘<br>c) Asynchronous Association<br>✘<br>Candidate Image1 Candidate Image2<br>Choose the related one ✘ ✘<br>metal furry metal<br>metal furry<br>✘ ✘ ✘<br>**----- End of picture text -----**<br>


Figure 2: Semantic figure of association question. (Left) Single-step association with a fixed correct memory. (Right) Synchronous and asynchronous association with dynamic memory. Synchronous involves one category in the chain, while asynchronous improves complexity with two categories. 

“metal”, constituting a loop strategy (Figure 2c). In this case, we need to call back the memory from the previous step instead of the last step. Similar to the synchronous association, we utilize the association dataset for the asynchronous association setting by selecting multiple shared concepts between different input samples. We formulate the dataset as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0005-04.png)


where _C_[ˆ] _[m]_ indicates _m_ shared concepts selected from _C_[ˆ] . In addition, we also utilize the memory base that the strategy is the same as the synchronous association for the asynchronous association. 

## 4.2 DATA REFINEMENT FOR ASSOCIATION TASK 

Depending on annotation-free association construction methods, we transfer the general dataset for the association task. The dataset has paired input samples and is labeled with whether they have common concepts for the association step. Furthermore, paired input samples with shared concepts were created for the deduction step. While these support the challenges in association tasks, there is still a possibility of confusing samples. To address this, we introduce a data refinement method. 

We implemented a three-step strategy to ensure data quality, including an Image resolution filter, MLLM verification, and Human expert evaluation. Specifically, the Image resolution filter screened out all images with less than 50 _,_ 000 pixels to ensure superior visual quality. The MLLM verification[2] takes a question-answer strategy with OpenAI’s GPT4-V Achiam et al. (2023) and Google’s Gemini-1.5-Flash Reid et al. (2024) to ensure each annotation of raw data exists in the image. Then, the human expert evaluation is conducted through our custom-designed interface, enabling testers to complete the association task and eliminate low-quality samples or those with ethical concerns[3] . 

Our benchmarks are implemented in adjective concepts and verb concepts, which include attribute and affordance in OCL Li et al. (2023) and action in Pangea Li et al. (2024b). In the OCL dataset, we selected eight attributes with good perception performance, such as “metal, ripe, fresh, natural, cooked, painted, rusty, furry”, and eight affordances “sit, imprint, push, carry, cut, clean, open, break”. In addition, we selected eight actions of “run, hit, drive, dress, cooking, build, shake, cut”. 

## 4.3 BASELINE FOR ASSOCIATION 

There are various methods to improve the concept perception. Our focus lies in exploring tuningfree methods, which harness the inherent capabilities of the model. To this end, we employ popular prompt engineers to improve the understanding of MLLMs, including common knowledge (ComKnow), one-shot, and chain-of-thought (CoT). 

> 2We compare the performance with and without the MLLM verification in the section E of supplementary. 

> 3The human expert test interface refers to Figure 7 in the supplementary. 

5 

Preprint. Under review. 

For the memory base in association, we introduce three different memory strategies that transfer each inference process into memory after each association step, including structure memory (StructM), natural language memory (NLM), and chain memory (ChainM). StructM and NLM both simulate human memory, differing only in their descriptive approaches. They incorporate memory attention mechanisms to determine whether to reinforce or forget memories. Specifically, The underlying strategy of memory base is that if one type of memory knowledge _mk_ appears in evidence at one step, we add a repetition weight _wr_ to the attention weight W; otherwise, the attention weight decays, _i.e._ , subtracts forgetting decrement _df_ . This strategy is described as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0006-02.png)


StructM uses raw structure memory as input, while NLM transforms it into descriptions that are more aligned with human language. Additionally, ChainM is more closely aligned with the task, as it stores the previous inference process as a chain. In the ChainM strategy, each inference process is treated as a subchain, represented as _obj1 − > concept − > obj2_ , and is concatenated into the prior memory. The detailed description and practice examples refer to section A.2 in the supplementary. 

## 5 EXPERIMENTS 

## 5.1 SETTINGS 

## 5.1.1 IMPLEMENTATION DETAILS 

We systematically conduct experiments that include concept perception, single-step, synchronous, and asynchronous association. Specifically, we implement the task as a multi-choice setting in all experiments, taking one query image and two candidate images with one correct as input and output correct image index. Based on this, we first devise preliminary concept perceptions that involve popular prompt engineering skills on open-source MLLMs[4] to investigate perception capabilities in attribute concept. Then, we convert to evaluate MLLM’s association capabilities that make decisions based on current observation and prior practice memory, _i.e._ , input with additional content of previous practice. We develop single, synchronous, and asynchronous associations according to fixed or dynamic memory. The single-step association means the model decides with observation and correct prior practice. Meanwhile, the synchronous and asynchronous association set model at a dynamic task that iteratively arrives at a decision and then deducts the underlying evidence, which means the memory may have wrong information for the next judge. 

For single-step, synchronous, and asynchronous association, we design three types of memory bases, _i.e._ , Structure Memory (StructM), Natural Language Memory (NLM), and Chain Memory (ChainM). In addition, we involve the baseline of No Memory (NoM) which means determining at a dynamic setting without the memory base. For the detailed description of the type of memory base and the usage in the prompt, please refer to the section A.2 in the supplementary. 

We utilize three open-source MLLMs in preliminary concept perception: QWen-VL Bai et al. (2023b), LLaVA-NeXT-7B Liu et al. (2024), and LLaVA-NeXT-13B. For formal association, we utilize three new-versions MLLMs that break through the MLLM’s capabilities in multi-images: LLaVA-OneVision Li et al. (2024a), QWen2-VL Wang et al. (2024b), and mPLUG-Owl3 Ye et al. (2024). Besides, we evaluate the performance of Mixture-of-Experts (MoE) that combined three open-source MLLMs. In experiments, open-source MLLM is run on a single NVIDIA A100 80G GPU. Apart from open-source MLLMs, we include the evaluation of the closed-source MLLMs of GPT4-V Achiam et al. (2023) and Gemini-1.5-Flash Reid et al. (2024). Simultaneously, we involve the results of three human experts to demonstrate the gap between MLLM with human intelligence. 

## 5.1.2 METRICS FOR ASSOCIATION TASK 

**Max** _|_ **Mean Step.** In an association task, _max-step_ indicates the maximum number of steps in one round of association, _i.e._ , the maximum length of a correctly predicted association chain. While _mean-step_ refers to the average maximum association step across multi-rounds of association tests. 

> 4For the detailed description of the setting of concept perception, please refer to section B. 

6 

Preprint. Under review. 

**Success Ratio.** For concept perception and single-step association, suppose the total number of samples of each concept ˆ _ci_ is _Ti_ , and the correctly judged based on the shared concept is _Ti_[+][samples.] For any association and deduction step, the success ratio on concept ˆ _ci_ is defined as _ri_[+][=] _[T] Ti_[ +] _i_[.] 

## 5.2 RELATIONSHIP BETWEEN PERCEPTION AND ASSOCIATION 

In this subsection, we investigate the MLLM’s ability in attribute concept perception that is implemented on the OCL Li et al. (2023). We devise multiple task complexity, including individual perception and combination perception. The detailed description and results are shown in section B in the supplementary. These results indicate that while popular prompt engineering skills improve performance in specific cases, _i.e._ , prompt engineering skills compared to Normal, they still exhibit limited capabilities in the widely-used object understanding benchmark. For instance, in the individual perception, the highest perception success ratio of 0 _._ 690 reflects an improvement of 0 _._ 190 over the random baseline of 0 _._ 5. Furthermore, the combination perception attains the success ratio of 0 _._ 542, improving 0 _._ 292 over the random baseline of 0 _._ 25. 

The association is a capability built upon foundational perception abilities. Hence, this prompts us to reduce the complexity of perception and _disentangle_ the evaluation of perception from association. In practice, we selected categories with good perception performance for association tasks. This enables a more rigorous assessment of the MLLM’s associative capabilities. 

|Type|Concept<br>Memory|Models (Success Ratio)|Models (Success Ratio)|
|---|---|---|---|
|||LLaVa-OneVision<br>QWen2-VL<br>mPLUG-Owl3<br>**Gemini-1.5-Flash**|Avg.|
|Association<br>Success Ratio|Attribute<br>(Adjective)<br>NoM<br>StructM<br>NLM<br>ChainM|75_._52<br>76_._36<br>54_._05<br>75_._40<br>78_._41<br>77_._74<br>72_._67<br>87_._30<br>80_._40<br>86_._63<br>73_._33<br>**88.39**<br>77_._73<br>73_._55<br>70_._70<br>83_._78|70_._33<br>79_._03<br>82_._19<br>76_._44|
||Affordance<br>(Adjective)<br>NoM<br>StructM<br>NLM<br>ChainM|70_._61<br>75_._10<br>53_._80<br>77_._78<br>73_._39<br>75_._39<br>73_._02<br>84_._45<br>76_._13<br>79_._37<br>67_._79<br>**85.40**<br>74_._38<br>82_._67<br>68_._46<br>80_._02|69_._32<br>76_._56<br>77_._17<br>76_._38|
||Action<br>(Verb)<br>NoM<br>StructM<br>NLM<br>ChainM|75_._74<br>78_._43<br>57_._94<br>84_._21<br>75_._44<br>82_._10<br>73_._92<br>88_._10<br>78_._66<br>88_._01<br>70_._04<br>**89.58**<br>76_._92<br>85_._59<br>69_._50<br>87_._58|74_._08<br>79_._89<br>81_._57<br>79_._90|
|Deduction<br>Success Ratio|Attribute (Adjective)<br>Affordance (Adjective)<br>Action (Verb)|49_._38<br>58_._11<br>61_._21<br>65_._82<br>21_._07<br>15_._98<br>33_._30<br>33_._08<br>46_._61<br>49_._99<br>42_._89<br>55_._69|58_._63<br>25_._86<br>48_._80|



Table 1: Mean success ratio of each concept on single-step association with four memory strategies across open-source and close-source MLLM.The best and second results of each concept are shown in **bold** and underline, respectively. For detailed results on each category refer to Table 5, 6, 7. 

## 5.3 RESULT OF ASSOCIATION 

Based on the finding of concept perception, we convert to evaluate MLLM’s association capability. In the following subsection, we first access MLLM’s ability in the single-step association that makes decisions with observation and correctly fixed memory. Then, we evaluate its associative capabilities in a dynamic setting, with evidence derived from the MLLM’s deduction. We further investigate the efficacy of MLLMs in a straightforward synchronous association encompassing only one semantic concept within the association chain. Furthermore, we increase the task complexity to attain asynchronous association by involving _two_ semantic concepts in the association chain. 

## 5.3.1 SINGLE-STEP ASSOCIATION 

Table 1 shows the mean result in adjective and verb concepts, _i.e._ , attribute and affordance on the OCL Li et al. (2023) and verb on the Pangea Li et al. (2024b). We focus on the metric of “success ratio” in this part, and the detailed result on each item can be found in supplementary Table 5, 6, 7. 

From the result, we find that Gemini-1.5-Flash achieves the highest performance across various concepts in all settings, but remains a certain gap from humans, _i.e._ , success ratio compared to 1. It deserves noted that human single-step associations are nearly unlikely errors, as correct memory 

7 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0008-01.png)


Figure 3: The average Max _|_ Mean step on the individual concept synchronous association across three open-source MLLM and humans. Different columns within each MLLM indicate different memory strategies. For detailed results on each category refer to Table 8, 9, 10 in the supplementary. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0008-03.png)


Figure 4: The average Max _|_ Mean step of asynchronous association with paired categories across different concepts. The upper subfigure is mean-step, while the lower subfigure is max-step. The detailed results on each different category group refer to Table 11, 12, 13 in the supplementary. 

entails the rule. Moreover, our comprehensive data refinement ensures the data quality that prevents any errors induced by annotations confusion. Besides, the association has certain improvements across various concepts, _i.e._ , the proposed memory strategy is higher than NoM. Simultaneously, the NLM attained the strongest performance, we speculate that due to the current MLLM being trained on a large amount of internet visual language data, lacking structured data training. 

## 5.3.2 SYNCHRONOUS ASSOCIATION 

The primary focus of our design revolves around the stability of MLLMs in synchronous association within multi-step settings. The Max _|_ Mean step metrics serve as direct measures of the length of a continuous association chain, offering more precise insights into the capacity to comprehend rules and uphold stability. Intuitively, when humans master the rules through some examples or previous association processes, they can keep going without making mistakes. 

Figure 3 shows the attribute, affordance, and action results in the synchronous association with four memory strategies. The results clearly show a significant gap between open-source MLLMs and human experts, _i.e._ , the green column compared to others. In attribute synchronous association, human experts achieve the mean-step of 350, while the average step of the open-source models is below 20. Additionally, different memory strategies exhibit a consistent trend with single-step associations across various approaches, with NLM showing slightly better performance in both max and mean step evaluations compared to other memory strategies. Furthermore, the open-source QWen2-VL 

8 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0009-01.png)


**----- Start of picture text -----**<br>
Deduction step<br>They all have<br>New memory: Memory ‘natural’ attribute.<br>Fresh. Dogs and birds have<br>fresh attributes.<br>Image1 Image2<br>Association step<br>Association step Memory<br>Image1.<br>Image2.<br>Query image Image1 Image2<br>Query image Image1 Image2<br>Failure case 1: Error Deduction Failure case 2: Error Association<br>**----- End of picture text -----**<br>


Figure 5: Failure cases of GPT4-V in asynchronous association with paired attribute categories. (Left) The error arises from the deduction. (Right) The error originates from the association. 

outperforms LLaVA-OneVision and mPLUG-Owl3, which we attribute to its pre-training methods that achieve stronger cross-modal alignment, resulting in greater performance gains. This guides our future exploration toward improving the associative capabilities of MLLMs. The detailed analysis of comprehensive results refers to supplementary. 

## 5.3.3 ASYNCHRONOUS ASSOCIATION 

In the following, we proceed to investigate the MLLM’s ability in asynchronous association. That improves the complexity by introducing two different semantic categories compared to synchronous association. Figure 4 summarises the results of asynchronous association in the OCL and Pangea datasets. We involve four different semantic concept pairs for each dataset. In all datasets, we take the same memory strategies as the synchronous association. 

From the result, we can easily found asynchronous association can be a challenge for humans in some cases. For instance, in the Pangea action, the human expert achieves an average mean-step of 33 _._ 5 and an average max-step of 58 _._ 0, which is lower than synchronous association. In addition, it meets our expectation that MoE outperforms the individual open-source MLLM in all cases, _i.e._ , the max-step and mean-step of MoE are higher than open-source MLLMs. It catches our attention that closed-source MLLM has a close performance to open-source MLLM, _i.e._ , GPT4-V, and Gemini1.5-Flash compared to open-source MLLMs. This indicates that all current models exhibit poor performance on association tasks when the gap in perceptual capabilities is minimized. Furthermore, Gemini-1.5-Flash outperforms GPT-4V in certain cases, which we speculate is due to the MLLM verification step in data refinement. This step employs a question-answer strategy to verify the existence of concepts, initially relying on Gemini-1.5-Flash for judgment and deferring to GPT-4V only when Gemini-1.5-Flash is unable to provide a decision (Subsection 4.2). 

## 5.4 ANALYSIS OF RESULTS 

**Analysis of Failure Case.** We now analyze the failure cases in the association process. We divided them into two types according to the stage in our benchmark, _i.e._ , Error deduction, and Error Association. In the first type, the MLLM produces the error perception about the common concepts in two samples. This then causes the error in the association step (Left of Figure 5). Conversely, in the second type, MLLM has the correct memory and makes errors due to the limited perception capability (Right of Figure 5). More interesting, these failures are consistent with human tests, we may derive error information from examples and further induce incorrect judgments. 

**Analysis of Attention.** We demonstrate that there is a significant gap between current open-source, and closed-source MLLMs and human intelligence. We speculate that the underperformance arises from two main factors: a lack of multi-image instruction tuning and limitations in contextual understanding. The first statement is also observed from existing work Song et al. (2024); Wang et al. (2024c), that the current MLLM has a weak ability in multi-image understanding since the lack of its instruction data. In addition, recent research Wang et al. (2024a) has demonstrated MLLMs’ poor performance in handling long contexts, _i.e._ , memory. We also visualize the attention map of open-source MLLM Qwen-VL on OCL attribute concept to support this observation, as Figure 6, we can easily find that response has predominate attention at the position close to response instead of the part for decision-making in both StructM and NLM. 

9 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0010-01.png)


**----- Start of picture text -----**<br>
StructM Attention Weight NLM Attention Weight<br>0 0<br>4 4<br>8 8 0.6<br>0.5<br>12 12<br>0.4<br>16 16<br>M emory Input Samples M emory Input Samples 0.3<br>(I m portant) (Important) (I m portant) (Important)<br>20 20 0.2<br>24 24 0.1<br>0.0<br>28 28<br>31 31<br>Average Average<br>Memory Memory Question Original Candidate Candidate Output Memory Memory Question Original Candidate Candidate Output<br>Instruction Instrcution Image Image1 Image2 Instruction Instruction Instrcution Image Image1 Image2 Instruction<br>Layers Layers<br>**----- End of picture text -----**<br>


Figure 6: Attention weight of QWen-VL with StructM and NLM strategies in asynchronous association. A deeper color means greater attention. The frame marks the critical area in the input context. 

## 5.5 ABLATION STUDY 

We conduct ablation studies that take different example sizes in the associations to analyze the selection of an initialized memory base. Table 2 summarises the results for example sizes of 1, 3, and 5 in the LLaVA-OneVision Li et al. (2024a) and QWen2-VL Wang et al. (2024b). Regardless of whether considering the max or mean step, the influence of different sample sizes is minimal, with an average gap of 1 _._ 51 for the maximum step and 0 _._ 05 for the mean step. As the maximum step indicates peak performance in some cases, while the mean step reflects more stable performance, we have chosen a sample size of 3 based on the mean step results for all our experiments. 

|Example<br>size|LLaVa-OneVision|QWen2-VL|QWen2-VL|
|---|---|---|---|
||StructM<br>NLM<br>ChainM|StructM<br>NLM<br>ChainM||
|1<br>3<br>5|30_._3_|_3_._22<br>33_._3_|_3_._92<br>28_._3_|_3_._34<br>26_._3_|_3_._14<br>36_._8_|_3_._94<br>31_._8_|_3_._35<br>32_._5_|_3_._17<br>36_._5_|_3_._90<br>32_._8_|_3_._35|21_._5_|_2_._05<br>36_._8_|_4_._36<br>27_._8_|_3_._41<br>21_._8_|_2_._26<br>43_._8_|_4_._43<br>22_._8_|_3_._45<br>21_._5_|_2_._09<br>38_._8_|_4_._53<br>25_._0_|_3_._39|29_._67_|_3_._38<br>30_._55_|_**3.43**<br>**31.18**_|_3_._40|



Table 2: The ablation of example sizes in the association. (Detailed results in Table 14.) 

## 6 LIMITATIONS AND DISCUSSION 

In this paper, our investigation is limited to MLLMs’ zero-shot ability in association tasks across adjectives and verb semantic concepts. Experiments demonstrate that although MLLMs make advances in other scenarios, they exhibit weak ability in association. We speculate that this deficiency may be due to **the lack of learning of unpaired data.** To the best of our knowledge, current MLLMs are trained on image-text pairs and interleaved image-text pair data, providing them with a powerful ability to comprehend input information. However, the association task requires MLLM to have the capability of inference on unpaired sequence data, as well as the ability to gradually uncover the _underlying principles_ through the process of all prior decision-making. In the future, an urgent study is still needed to develop a paradigm that links new learning with prior learned knowledge, which may enhance MLLM’s association capabilities. We believe that the next stage of MLLMs should expand the learning of unpaired data, which may require the creation of a new learning framework. This advancement will help narrow the gap between MLLMs and human intelligence. 

## 7 CONCLUSION 

In this paper, we propose a benchmark to evaluate the association ability of MLLMs via an annotation-free association construction method that easily transfers general datasets for association 

10 

Preprint. Under review. 

tasks. Using this method, we devise a standard benchmark based on adjective and verb semantic concepts as the association chain. Expanding experiments demonstrate that current open-source MLLMs and even GPT-4V have a significant gap compared to humans. 

## REFERENCES 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

01. AI, :, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. Yi: Open foundation models by 01.ai, 2024. 

David P Ausubel. The psychology of meaningful verbal learning. 1963. 

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. _arXiv preprint arXiv:2309.16609_ , 2023a. 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023b. 

- Kilian Carolan, Laura Fennelly, and Alan F Smeaton. A review of multi-modal large language and vision models. _arXiv preprint arXiv:2404.01322_ , 2024. 

- Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 2818–2829, 2023. 

- Wenliang Dai, Junnan Li, DONGXU LI, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In _Advances in Neural Information Processing Systems_ , volume 36, pp. 49250–49267, 2023. 

- Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, et al. Internlm-xcomposer2: Mastering freeform text-image composition and comprehension in vision-language large model. _arXiv preprint arXiv:2401.16420_ , 2024. 

- Sivan Doveh, Shaked Perek, M. Jehanzeb Mirza, Amit Alfassy, Assaf Arbelle, Shimon Ullman, and Leonid Karlinsky. Towards multimodal in-context learning for vision and language models, 2024. 

- Pengbo Hu, Ji Qi, Xingyu Li, Hong Li, Xinqi Wang, Bing Quan, Ruiyu Wang, and Yi Zhou. Tree-ofmixed-thought: Combining fast and slow thinking for multi-hop visual reasoning. _arXiv preprint arXiv:2308.09658_ , 2023. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- Huaizu Jiang, Xiaojian Ma, Weili Nie, Zhiding Yu, Yuke Zhu, and Anima Anandkumar. Bongardhoi: Benchmarking few-shot visual reasoning for human-object interactions. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 19056–19065, 2022. 

- DeBrae Kennedy-Mayo and Jake Gord. ”model cards for model reporting” in 2024: Reclassifying category of ethical considerations in terms of trustworthiness and risk management. _ArXiv_ , abs/2403.15394, 2024. URL https://api.semanticscholar.org/ CorpusID:268681127. 

11 

Preprint. Under review. 

- Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International journal of computer vision_ , 123:32–73, 2017. 

- H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre. HMDB: a large video database for human motion recognition. In _Proceedings of the International Conference on Computer Vision (ICCV)_ , 2011. 

- Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. _arXiv preprint arXiv:2408.03326_ , 2024a. 

- Yong-Lu Li, Yue Xu, Xinyu Xu, Xiaohan Mao, Yuan Yao, Siqi Liu, and Cewu Lu. Beyond object recognition: A new benchmark towards object concept learning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 20029–20040, 2023. 

- Yong-Lu Li, Xiaoqian Wu, Xinpeng Liu, Zehao Wang, Yiming Dou, Yikun Ji, Junyi Zhang, Yixing Li, Xudong Lu, Jingru Tan, et al. From isolated islands to pangea: Unifying semantic space for human action understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 16582–16592, 2024b. 

- Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _arXiv preprint arXiv:2310.03744_ , 2023a. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023b. 

- Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. URL https:// llava-vl.github.io/blog/2024-01-30-llava-next/. 

- Arjun Mani, William Hinthorn, Nobline Yoo, and Olga Russakovsky. Point and ask: Incorporating pointing into visual question answering. 2020. 

- Sarnoff Mednick. The associative basis of the creative process. _Psychological review_ , 69(3):220, 1962. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021. 

- Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 

- Zhenwei Shao, Zhou Yu, Meng Wang, and Jun Yu. Prompting large language models with answer heuristics for knowledge-based visual question answering. In _2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 14974–14983, 2023. doi: 10.1109/CVPR52729.2023.01438. 

- Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi. What does clip know about a red circle? visual prompt engineering for vlms. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , pp. 11987–11997, October 2023. 

- Dingjie Song, Shunian Chen, Guiming Hardy Chen, Fei Yu, Xiang Wan, and Benyou Wang. Milebench: Benchmarking mllms in long context. _arXiv preprint arXiv:2404.18532_ , 2024. 

- Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

12 

Preprint. Under review. 

- Hengyi Wang, Haizhou Shi, Shiwei Tan, Weiyi Qin, Wenyuan Wang, Tunyu Zhang, Akshay Nambi, Tanuja Ganu, and Hao Wang. Multimodal needle in a haystack: Benchmarking long-context capability of multimodal large language models. _arXiv preprint arXiv:2406.11230_ , 2024a. 

- Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_ , 2024b. 

- Xiyao Wang, Yuhang Zhou, Xiaoyu Liu, Hongjin Lu, Yuancheng Xu, Feihong He, Jaehong Yoon, Taixi Lu, Gedas Bertasius, Mohit Bansal, et al. Mementos: A comprehensive benchmark for multimodal large language model reasoning over image sequences. _arXiv preprint arXiv:2401.10529_ , 2024c. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. NIPS ’22, 2024. ISBN 9781713871088. 

- Noam Wies, Yoav Levine, and Amnon Shashua. The learnability of in-context learning. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- An Yan, Zhengyuan Yang, Junda Wu, Wanrong Zhu, Jianwei Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Julian McAuley, Jianfeng Gao, et al. List items one by one: A new data source and learning paradigm for multimodal llms. _arXiv preprint arXiv:2404.16375_ , 2024. 

- Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. _arXiv preprint arXiv:2408.04840_ , 2024. 

- Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. _arXiv preprint arXiv:2311.04257_ , 2023. 

- Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. _arXiv preprint arXiv:2306.13549_ , 2023. 

- Andrew Zaldivar, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Lucy Vasserman, M. Mitchell, Parker Barnes, Simone Sanoian McCloskey Wu, and Timnit Gebru (eds.). _Model Cards for Model Reporting_ , 2019. URL https://dl.acm.org/citation.cfm?id= 3287596. 

- Duzhen Zhang, Yahan Yu, Chenxing Li, Jiahua Dong, Dan Su, Chenhui Chu, and Dong Yu. Mmllms: Recent advances in multimodal large language models. _arXiv preprint arXiv:2401.13601_ , 2024a. 

- Zicheng Zhang, Haoning Wu, Erli Zhang, Guangtao Zhai, and Weisi Lin. A benchmark for multimodal foundation models on low-level vision: from single images to pairs. _arXiv preprint arXiv:2402.07116_ , 2024b. 

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 

13 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs_images/iclr_2025_The_Labyrinth_of_Links_Navigating_the_Associative_Maze_of_Multi-modal_LLMs.pdf-0014-01.png)


Figure 7: The human evaluation interface in data refinement is designed to filter low-quality data and flag potential ethical issues. The evaluation process consists of two stages: first, a preview phase to familiarize evaluators with sample data (Left), followed by the main testing phase (Right). 

## A IMPLEMENTATION DETAILS 

## A.1 IMPLEMENTATION OF DATA REFINEMENT 

In our experiment, we comprehensively develop three steps to consolidate the data quality in our tasks, including Image resolution filter, MLLM verification, and Human expert evaluation. 

**Image Resolution Filter.** In this work, we concentrate on the object within the image, yet the current object detection and object concept learning datasets typically represent only partial content of the images, posing significant perception challenges. Hence, the Image resolution filtering step excludes all images with fewer than 50 _,_ 000 pixels, ensuring sufficient visual quality. 

**MLLM Verification.** While an image resolution filter ensures that images contain sufficient information for downstream tasks, there remain existing some concerns. One is the correctness of raw annotations, and the other is perception ability in difficult categories. We propose an MLLM verification step that further filters input samples with erroneous annotations or those requiring advanced perception capabilities. This step does not introduce bias, as the filtering method relies on a widely used public dataset with only insignificant shortcomings. Specifically, we employ a questionanswers strategy to check whether the model identifies the existence of one concept. This process begins with Gemini-1.5-Flash, then GPT4-V once Gemini-1.5-Flash is unable to reach a decision. 

**Human Expert Evaluation.** The first two steps ensure the correctness of our benchmark, which reduces the image with low quality or confusing annotation. We continue to develop an online testing interface for human testers, as shown in Figure 7, which further filters the images with potential confusion or ethical concerns from the human perspective. We have strictly followed the ethical review, which is described in section F of the supplementary. 

## A.2 IMPLEMENTATION OF MEMORY BASE 

In our proposed association tasks, whether involving single-step, synchronous and asynchronous associations, we developed three memory strategies to emulate human intelligence: Structure Memory (StructM), Natural Language Memory (NLM), and Chain Memory (ChainM). As in Table 3, the prompt in the association task can be divided into three parts. The first component is the memory context, including the memory instructions and base. Next is the question content, comprising the 

14 

Preprint. Under review. 

question instructions and the questions themselves. Finally, the output instructions guide the model to ensure the generated output aligns with our specified requirements. 

|Type<br>Memory<br>Type|Memory<br>Instruction|Memory base|Question Instruction<br>Question|Output Instruction|
|---|---|---|---|---|
|Association<br>Step<br>NoM<br>StructM<br>NLM<br>ChainM|-|-|Determine the relationship between<br>the original image and the candidate<br>images, and select the images with<br>the same attribute/affordance/action<br>as the original image.<br>Original image:_<_image_>_.<br>Candidate images:<br>Image1:_<_image_>_,<br>Image2:_<_image_>_.|Your response should be direct<br>and exclusively only include<br>one of the following items.<br>Options: [Image1, Image2].|
||Given the memory:|_{_‘eat‘: [‘sandwich‘, ‘pizza‘]_}_|||
||Before this question,<br>you have learnt that<br>related pictures may have<br>the following affordance:|[‘broccoli‘, ‘orange‘]have<br>eat affordance|||
|||broccoli_−>_eat_−>_pizza<br>_−>_eat_−>_baked|||
|Deduction<br>Step<br>-|-|-|Generate the common affordance<br>between the original<br>image and selected images.<br>Original image:_<_image_>_.<br>Selected image: _<_image_>_.|Your response should only include<br>shared affordance in the following options.<br>Options:[‘break‘, ‘carry‘, ‘clean‘, ‘cut‘,<br>‘open‘, ‘push‘, ‘sit‘, ‘imprint‘]|



Table 3: The prompt format in our association benchmark, which includes the association step and deduction step. The association step makes the selection according to current observations and prior practice. We devise three memory strategies in the association step. In the deduction step, MLLM deducts the underlying concept with the original image and correctly selected image. 

In detail, for each association step, NoM operates without any memory content. StructM employs simple memory instructions that incorporate memory and structured dictionary knowledge from prior tasks. Both NLM and ChainM utilize more detailed memory instructions, with NLM representing prior knowledge in everyday language, while ChainM organizes the memory as a taskoriented sequence. Furthermore, the deduction step includes a detailed question description and corresponding output instructions, without relying on any memory content. 

In the experiment, we set the repetition weight _wr_ and forgetting decrement _df_ are 1 _._ 0 and 0 _._ 2 for memory base attention in StructM and NLM in all cases, respectively. 

## B PRELIMINARY INVESTIGATION ON CONCEPT PERCEPTION 

To thoroughly access the MLLM’s ability in concept perception, we systematically design multiple task complexities across multiple tunning-free methods. Specifically, we involve four task complexity settings across two types in all models, including individual and combination perceptions. The individual perception includes Task Instruction (TaskInstr) and Meta Instruction (MetaInstr). However, combination perception consists of Task Instruction with Deduction (TaskInstr w/ DedPo) and Meta Instruction with Deduction (MetaInstr w/ DedPo), which involves another deduction step compared with individual perceptions. The distinction between TaskInstr and MetaInstr lies in whether the prompt includes question instruction that describes the decision-making explanations of the task. In addition, with Deduction (w/ DedPo) compared to normal, is whether we only access perception or evaluate deduction ability. On the other hand, enhancing perception capabilities can be approached from several directions, including expanding foundational knowledge and incorporating practical examples. Hence, we include several prompt engineering skills for tunning-free methods such as common knowledge (ComKnow), one-shot, and Chain-of-Thought (CoT). ComKnow aims to provide foundational knowledge of object-related concepts, while One-shot and CoT focus on enhancing performance through few-shot learning. Compared to One-shot, CoT introduces an additional reasoning step to emulate human-like thought processes. 

We carry out experiments on three open-source MLLMs. QWen-VL Bai et al. (2023b) uses QWen7B Bai et al. (2023a) as the initialize of LLM and OpenCLIP’s ViT-bigG Cherti et al. (2023) as the visual encoder. LLaVA-NeXT-7B Liu et al. (2024) is constructed on the LLM of Mistral-7B Jiang et al. (2023) and vision encoder of CLIP’s ViT-L/14 Radford et al. (2021), while LLaVA-NeXT-13B uses the same vision encoder but is constructed on LLM of Vicuna-13B Zheng et al. (2024). 

Table 4 summarises the result on the OCL dataset. In all models, we include four different task complexity settings: Task Instruction (TaskInstr), Meta Instruction (MetaInstr), Instruction with Deduction (TaskInstr w/ DedPo), and Meta Instruction with Deduction (Meta w/ DedPo). For comparison, we include the Random and Normal as baselines, as well as general prompt-engineer skills Common Knowledge (Comknow), One-shot, and Chain-of-Thought (CoT). 

15 

Preprint. Under review. 

|Model<br>Strategy|Model<br>Strategy|Concept Perception Task Complexity (Success Ratio)|Concept Perception Task Complexity (Success Ratio)|
|---|---|---|---|
|||Individual Perception|Combination Perception|
|||TaskInstr<br>MetaInstr|TaskInstr w/ DedPo<br>MetaInstr w/ DedPo|
||Random|0_._5<br>0_._5|0_._25<br>0_._25|
|QWen-VL|Normal<br>ComKnow<br>One-shot<br>CoT|0_._515<br>**0.690**<br>0_._594<br>0_._512<br>**0.610**<br>0_._638<br>0_._518<br>0_._657|**0.506**<br>**0.542**<br>0_._502<br>0_._419<br>0_._425<br>0_._436<br>0_._440<br>0_._433|
|LLaVA-NeXT<br>(Mistral-7B)|Normal<br>ComKnow<br>One-shot<br>CoT|0_._494<br>0_._565<br>0_._568<br>0_._498<br>0_._641<br>0_._631<br>**0.664**<br>**0.667**|0_._433<br>0_._415<br>0_._411<br>0_._374<br>0_._453<br>0_._475<br>**0.494**<br>**0.519**|
|LLaVA-NeXT<br>(Vicuna-13B)|Normal<br>ComKnow<br>One-shot<br>CoT|0_._510<br>0_._587<br>0_._597<br>0_._532<br>**0.598**<br>0_._605<br>0_._590<br>**0.613**|0_._278<br>0_._420<br>0_._408<br>0_._381<br>0_._411<br>0_._421<br>**0.427**<br>**0.430**|



Table 4: The success ratio on OCL for open-source MLLMs at the concept perception. We include prompt-engineer methods ComKnow, One-shot, CoT. The best results are shown in **bold** . 

Open-source MLLMs perform poorly in concept perception, as reflected in Normal having close performance to Random in all cases. The prompt engineering skills show a slight improvement in some cases. The biggest improvements for ComKnow, One-shot, and CoT are 0 _._ 130, 0 _._ 147, and 0 _._ 170, respectively. While they demonstrate improvements in certain cases, they exhibit weak stability. For instance, in the CoT strategy employed in the TaskInstr w/ DedPo setting, the LLaVANeXT-13B has shown an improvement of 0 _._ 170 w.r.t Normal whereas Qwen-VL exhibits a negative effect. Combining all results, we conclude that current MLLMs have limited capability in this task. 

## C DETAILED RESULTS 

## C.1 DETAILED SINGLE-STEP ASSOCIATION RESULT 

In the main text, we highlight three key points. First, the best performance of the closed-source MLLM, Gemini-1.5-Flash, still shows a significant gap compared to human experts. Second, our proposed memory strategies result in noticeable improvements across all cases. Finally, the NLM strategy demonstrates the highest performance compared to our proposed other memory strategies. In this part, we continue to analyze the differences within each concept. 

Table 5, 6, 7 summarise the results of single-step association on attribute, affordance, and action, respectively. For each concept type, we assess the association and deduction success rates using three open-source MLLMs—LLaVA-OneVision, QWen2-VL, and mPLUG-Owl3—as well as one closed-source MLLM, Gemini-1.5-Flash. These evaluations are conducted across three memory strategies—StructM, NLM, and ChainM—along with a baseline, NoM. We have selected eight categories for each concept, and provide a detailed analysis of the results below. 

**Comparison of Different Models.** From the whole perspective, Gemini-1.5-Flash consistently outperforms the other models across all cases. Specifically, for association success ratio, Qwen2VL leads in several categories of each concept. For instance, metal and cooked with NLM strategy attain the highest success ratio in attribute single-step association. Among the open-source MLLMs, Qwen2-VL demonstrates the strongest performance across three concepts, with LLaVA-OneVision performing moderately well but trailing behind Qwen2-VL, and mPLUG-Owl3 showing the lowest performance. Furthermore, different MLLMs exhibit consistent trends in concept understanding. For instance, LLaVA-OneVision underperforms in natural attribute association, the same pattern is also observed with QWen2-VL and mPLUG-Owl3 models. Additionally, there is a significant disparity in deduction success rates among the different MLLMs. For example, Gemini-1.5-Flash per- 

16 

Preprint. Under review. 

|Type|Model<br>Memory|Single-Step Attribute Concept Association Chain|Single-Step Attribute Concept Association Chain|
|---|---|---|---|
|||furry<br>metal<br>fresh<br>cooked<br>natural<br>ripe<br>painted<br>rusty|Avg.|
|Association<br>Success Ratio|LLaVA-Onevision<br>NoM<br>StructM<br>NLM<br>ChainM|57_._63<br>75_._21<br>75_._78<br>76_._09<br>51_._19<br>86_._83<br>88_._19<br>93_._24<br>57_._08<br>79_._73<br>83_._23<br>78_._26<br>53_._72<br>89_._22<br>87_._40<br>98_._64<br>60_._15<br>83_._52<br>87_._58<br>78_._26<br>51_._82<br>92_._22<br>93_._70<br>95_._94<br>60_._90<br>81_._80<br>86_._96<br>65_._22<br>53_._41<br>88_._62<br>88_._98<br>95_._95|75_._52<br>78_._41<br>80_._40<br>77_._73|
||QWen2-VL<br>NoM<br>StructM<br>NLM<br>ChainM|60_._08<br>76_._64<br>85_._09<br>71_._74<br>53_._88<br>87_._43<br>78_._74<br>97_._30<br>61_._31<br>87_._51<br>83_._23<br>78_._26<br>55_._78<br>79_._04<br>88_._98<br>87_._84<br>65_._26<br>**93.06**<br>96_._27<br>**97.83**<br>52_._61<br>96_._41<br>92_._91<br>98_._65<br>63_._90<br>91_._59<br>92_._55<br>93_._48<br>55_._15<br>**97.01**<br>93_._70<br>1_._00|76_._36<br>77_._74<br>86_._63<br>73_._55|
||mPLUG-Owl3<br>NoM<br>StructM<br>NLM<br>ChainM|52_._72<br>49_._68<br>63_._35<br>45_._65<br>49_._60<br>59_._28<br>56_._69<br>55_._41<br>61_._38<br>72_._16<br>75_._16<br>63_._04<br>56_._89<br>82_._63<br>79_._53<br>90_._54<br>64_._71<br>66_._75<br>86_._34<br>71_._74<br>51_._19<br>83_._83<br>75_._59<br>86_._49<br>61_._72<br>62_._42<br>83_._23<br>65_._22<br>57_._05<br>83_._23<br>62_._20<br>90_._54|54_._05<br>72_._67<br>73_._33<br>70_._70|
||**Gemini-1.5-Flash**<br>NoM<br>StructM<br>NLM<br>ChainM|59_._11<br>71_._01<br>82_._91<br>84_._78<br>55_._85<br>89_._02<br>68_._91<br>91_._66<br>68_._8<br>91_._8<br>93_._7<br>97_._8<br>**58.4**<br>92_._7<br>**95.2**<br>**100.0**<br>**74.9**<br>92_._9<br>**96.8**<br>95_._5<br>57_._5<br>94_._8<br>93_._9<br>**100.0**<br>70_._3<br>92_._6<br>87_._1<br>90_._9<br>56_._4<br>79_._9<br>94_._4<br>98_._6|75_._40<br>87_._30<br>**88.39**<br>83_._78|
|Deduction<br>Success Ratio|LLaVA-Onevision<br>-<br>QWen2-VL<br>-<br>mPLUG-Owl3<br>-<br>**Gemini-1.5-Flash**<br>-|71_._73<br>39_._40<br>38_._51<br>69_._57<br>17_._43<br>55_._69<br>90_._55<br>12_._16<br>63_._10<br>**69.75**<br>51_._55<br>63_._04<br>39_._62<br>**59.88**<br>59_._84<br>58_._11<br>66_._76<br>44_._12<br>61_._49<br>**84.78**<br>32_._49<br>47_._31<br>71_._65<br>81_._08<br>**74.44**<br>62_._18<br>**79.75**<br>**84.78**<br>**52.10**<br>6_._70<br>**91.60**<br>75_._00|49_._38<br>58_._11<br>61_._21<br>**65.82**|



Table 5: The success ratio of single-step attribute association, which includes the association step and deduction step. The association step evaluates MLLM’s capabilities in decision-making based on observation and corrected prior practice. The deduction step accesses MLLM’s capability to generate the common concepts that serve as the underlying rule in the association step. Additionally, the association step includes three memory strategies and a baseline NoM. The best and second results are shown in **bold** and underline, respectively. 

|Type|Model<br>Memory|Single-Step Affordance Concept Association Chain|Single-Step Affordance Concept Association Chain|
|---|---|---|---|
|||sit<br>imprint<br>push<br>carry<br>cut<br>clean<br>open<br>break|Avg.|
|Association<br>Success Ratio|LLaVA-Onevision<br>NoM<br>StructM<br>NLM<br>ChainM|77_._06<br>66_._74<br>80_._29<br>58_._73<br>83_._33<br>54_._83<br>78_._30<br>65_._57<br>81_._18<br>73_._43<br>85_._53<br>62_._20<br>80_._44<br>48_._39<br>83_._96<br>71_._99<br>87_._65<br>75_._10<br>87_._16<br>60_._35<br>83_._88<br>58_._06<br>88_._68<br>68_._17<br>86_._47<br>72_._80<br>83_._73<br>60_._81<br>82_._09<br>53_._22<br>90_._57<br>68_._58|70_._61<br>73_._39<br>76_._13<br>74_._38|
||QWen2-VL<br>NoM<br>StructM<br>NLM<br>ChainM|81_._76<br>67_._78<br>79_._39<br>62_._54<br>85_._54<br>**72.58**<br>81_._13<br>70_._08<br>83_._53<br>70_._50<br>78_._66<br>73_._99<br>81_._13<br>61_._29<br>80_._19<br>81_._83<br>88_._82<br>74_._90<br>90_._96<br>69_._25<br>89_._67<br>48_._39<br>89_._62<br>83_._33<br>87_._65<br>80_._96<br>90_._24<br>73_._53<br>92_._42<br>56_._45<br>96_._23<br>83_._88|75_._10<br>75_._39<br>79_._37<br>82_._67|
||mPLUG-Owl3<br>NoM<br>StructM<br>NLM<br>ChainM|58_._24<br>48_._33<br>53_._71<br>54_._57<br>59_._64<br>53_._23<br>51_._89<br>50_._82<br>81_._76<br>61_._51<br>74_._86<br>68_._09<br>74_._24<br>66_._13<br>83_._96<br>73_._63<br>78_._82<br>61_._72<br>67_._99<br>60_._81<br>79_._89<br>46_._77<br>78_._30<br>68_._03<br>87_._18<br>61_._72<br>64_._20<br>62_._77<br>75_._62<br>59_._68<br>67_._92<br>68_._58|53_._80<br>73_._02<br>67_._79<br>68_._46|
||**Gemini-1.5-Flash**<br>NoM<br>StructM<br>NLM<br>ChainM|86_._39<br>66_._67<br>83_._33<br>62_._15<br>95_._18<br>63_._93<br>83_._81<br>74_._79<br>**97.02**<br>70_._72<br>91_._77<br>73_._99<br>98_._19<br>58_._33<br>**99.04**<br>**86.55**<br>**97.02**<br>**83.64**<br>**92.90**<br>**74.79**<br>**98.20**<br>55_._17<br>98_._06<br>83_._44<br>94_._64<br>76_._50<br>86_._29<br>66_._40<br>95_._58<br>46_._67<br>95_._19<br>78_._92|77_._78<br>84_._45<br>**85.40**<br>80_._02|
|Deduction<br>Success Ratio|LLaVA-Onevision<br>-<br>QWen2-VL<br>-<br>mPLUG-Owl3<br>-<br>**Gemini-1.5-Flash**<br>-|66_._47<br>0_._42<br>8_._14<br>1_._04<br>4_._96<br>**35.48**<br>49_._06<br>3_._01<br>21_._76<br>**14.23**<br>1_._63<br>1_._50<br>1_._24<br>4_._84<br>80_._19<br>2_._46<br>**82.94**<br>0_._84<br>12_._12<br>2_._89<br>63_._91<br>12_._90<br>**84.91**<br>5_._87<br>58_._93<br>0_._85<br>**24.40**<br>**36.02**<br>53_._01<br>5_._00<br>57_._69<br>**28.71**|21_._07<br>15_._98<br>**33.30**<br>33_._08|



Table 6: Same as Table 5, but for affordance single-step association. 

forms well with “push”, “carry”, and “break” affordances, while the other MLLMs show relatively weaker capabilities. Conversely, Qwen2-VL excels in “imprint” affordance, and LLaVA-OneVision demonstrates strength in the ‘clean’ affordance, whereas Gemini-1.5-Flash struggles. 

17 

Preprint. Under review. 

|Type|Model<br>Memory|Single-Step Action Concept Association Chain|Single-Step Action Concept Association Chain|
|---|---|---|---|
|||run<br>hit<br>drive<br>dress<br>cooking<br>build<br>shake<br>cut|Avg.|
|Association<br>Success Ratio|LLaVA-OneVision<br>NoM<br>StructM<br>NLM<br>ChainM|74_._91<br>73_._25<br>79_._48<br>75_._36<br>88_._15<br>74_._05<br>82_._21<br>58_._53<br>72_._40<br>77_._23<br>77_._65<br>75_._36<br>85_._63<br>78_._61<br>79_._95<br>56_._67<br>78_._10<br>77_._50<br>82_._50<br>75_._36<br>88_._03<br>79_._99<br>87_._39<br>60_._44<br>73_._89<br>74_._49<br>80_._53<br>75_._36<br>87_._13<br>79_._56<br>85_._36<br>59_._01|75_._74<br>75_._44<br>78_._66<br>76_._92|
||QWen2-VL<br>NoM<br>StructM<br>NLM<br>ChainM|83_._32<br>80_._52<br>82_._50<br>65_._22<br>87_._70<br>74_._68<br>87_._61<br>65_._89<br>80_._71<br>79_._29<br>81_._01<br>75_._36<br>87_._89<br>87_._22<br>94_._82<br>70_._50<br>89_._72<br>86_._97<br>88_._66<br>**79.71**<br>95_._31<br>91_._87<br>96_._17<br>75_._70<br>85_._99<br>85_._05<br>82_._87<br>76_._81<br>93_._13<br>87_._08<br>97_._52<br>**76.24**|78_._43<br>82_._10<br>88_._01<br>85_._59|
||mPLUG-Owl3<br>NoM<br>StructM<br>NLM<br>ChainM|56_._10<br>52_._26<br>59_._02<br>53_._62<br>66_._49<br>63_._34<br>57_._43<br>55_._24<br>71_._28<br>69_._55<br>80_._13<br>60_._87<br>86_._30<br>77_._49<br>84_._68<br>61_._04<br>63_._19<br>62_._41<br>76_._45<br>63_._77<br>78_._15<br>73_._24<br>81_._98<br>61_._16<br>64_._04<br>62_._14<br>78_._64<br>50_._72<br>82_._57<br>73_._39<br>84_._91<br>59_._55|57_._94<br>73_._92<br>70_._04<br>69_._50|
||**Gemini-1.5-Flash**<br>NoM<br>StructM<br>NLM<br>ChainM|85_._95<br>88_._96<br>89_._23<br>71_._88<br>94_._16<br>81_._94<br>92_._05<br>69_._47<br>86_._49<br>89_._81<br>85_._71<br>77_._97<br>**99.60**<br>93_._20<br>97_._72<br>74_._33<br>**93.07**<br>92_._36<br>**92.28**<br>75_._41<br>98_._99<br>**93.78**<br>97_._28<br>73_._48<br>86_._46<br>**93.54**<br>88_._34<br>74_._63<br>**99.60**<br>86_._23<br>**98.41**<br>73_._42|84_._21<br>88_._10<br>**89.58**<br>87_._58|
|Deduction<br>Success Ratio|LLaVA-Onevision<br>-<br>QWen2-VL<br>-<br>mPLUG-Owl3<br>-<br>**Gemini-1.5-Flash**<br>-|58_._87<br>48_._97<br>73_._35<br>34_._78<br>96_._85<br>3_._96<br>3_._60<br>52_._48<br>60_._90<br>**69.00**<br>67_._41<br>62_._32<br>89_._78<br>7_._64<br>3_._60<br>39_._26<br>33_._99<br>4_._94<br>74_._52<br>44_._93<br>96_._74<br>7_._72<br>5_._63<br>**74.62**<br>**67.21**<br>18_._00<br>**86.99**<br>**79.69**<br>**97.79**<br>**19.96**<br>**59.68**<br>16_._22|46_._61<br>49_._99<br>42_._89<br>**55.69**|



Table 7: Same as Table 5, but for action single-step association. 

**Comparison of Different Categories.** Although single-step association shows consistent improvements across various concepts, performance differences emerge in specific categories. For instance, fresh, ripe, and rusty attributes gain an association success ratio close to 1, but natural and ripe attributes with a lower performance compared to the random success ratio of 0 _._ 5. This phenomenon is more pronounced in the deduction success ratio. Several model implementations exhibit strong deduction capabilities across different categories. For instance, Gemini-1.5-Flash achieves a 91 _._ 6 deduction success ratio in the ‘painted’ attribute. However, some categories still demonstrate lower performance, highlighting insufficient object understanding. 

**Comparison of Different Memory Strategies.** As demonstrated in other subsection analyses, NLM exhibits an overall more powerful performance than StructM and ChainM. But there also exists some observations that violate this finding, which further highlight the instability of the current MLLM. For instance, in the case of Gemini-1.5-Flash with “natural” and “painted” attributes, the success ratio of StructM strategy outperforms NLM. They demonstrate professional-level ability in specific cases but cannot maintain consistent stability. We speculate that similar to the prominent research area of hallucinations, there is still significant room for improvement. 

## C.2 DETAILED SYNCHRONOUS ASSOCIATION RESULT 

As summarized in the main text, a significant gap exists between current MLLM and human intelligence in synchronous association. Below, we provide a detailed analysis of individual concept associations across each category, focusing on OCL attributes, affordances, and the actions of Pangea. 

Table 8, 9, 10 shows the results of synchronous concept association on attribute, affordance, and action, respectively. For each concept, we include the same eight categories as in the single-step association. We include three open-source MLLMs in synchronous association, including LLaVAOneVision, QWen2-VL, and mPLUG-Owl3. Additionally, we also integrate this setting into the human test interface. In this setting, we use Max _|_ Mean Step as the primary metric. 

**Comparison of Different Models and Memory Strategies.** We find that QWen2-VL in the attribute and action concepts outperforms other models in almost all cases. But, LLaVA-OneVision exhibits extraordinary capabilities in affordance association. This demonstrates the inconsistency of MLLM’s concept understanding, which can also taken as a reference for future improvement of MLLM’s ability. Interestingly, memory strategies perform worse than NoM in certain instances. For 

18 

Preprint. Under review. 

|Model|Memory|Synchronous Attribute Concept Association Chain (Max_|_Mean Step)|Synchronous Attribute Concept Association Chain (Max_|_Mean Step)|
|---|---|---|---|
|||furry<br>metal<br>fresh<br>cooked<br>natural<br>ripe<br>painted<br>rusty|Avg.|
|LLaVA-OneVision|NoM<br>StructM<br>NLM<br>ChainM|15_|_1_._30<br>31_|_3_._20<br>39_|_3_._85<br>38_|_3_._13<br>11_|_1_._06<br>40_|_5_._69<br>70_|_6_._68<br>140_|_17_._47<br>15_|_1_._38<br>30_|_4_._03<br>41_|_4_._75<br>32_|_ 4_._04<br>13_|_1_._21<br>51_|_6_._51<br>65_|_9_._23<br>238_|_15_._24<br>14_|_1_._46<br>40_|_ 5_._24<br>68<br>_|_6_._09<br>**48**_|_3_._93<br>12_|_1_._16<br>91_|_8_._88<br>81<br>_|_ 10_._64<br>249_|_33_._88<br>16_|_1_._49<br>**63**_|_4_._66<br>58_|_5_._23<br>28_|_3_._29<br>11_|_1_._13<br>63_|_6_._98<br>70_|_9_._70<br>177_|_23_._79|48_._0_|_5_._30<br>60_._6_|_5_._80<br>75_._4_|_8_._91<br>60_._8_|_7_._03|
|Qwen2-VL|NoM<br>StructM<br>NLM<br>ChainM|**21**_|_1_._41<br>39_|_3_._14<br>63_|_ 8_._32<br>43<br>_|_**4.52**<br>14_|_1_._16<br>94<br>_|_ 10_._98<br>57_|_4_._44<br>353<br>_|_ 46_._05<br>13_|_1_._61<br>25_|_2_._36<br>28_|_3_._46<br>23_|_2_._63<br>**22**_|_**1.34**<br>33_|_4_._10<br>37_|_3_._42<br>122_|_ 9_._61<br>15_|_**1.74**<br>58<br>_|_**6.77**<br>**89**_|_**8.50**<br>36_|_**4.52**<br>11_|_1_._27<br>**109**_|_**11.90**<br>162_|_**12.50**<br>**500**_|_**64.50**<br>18_|_1_._64<br>39_|_4_._38<br>42_|_7_._53<br>32_|_ 4_._04<br>10_|_1_._18<br>77_|_9_._75<br>68_|_8_._50<br>169_|_22_._06|85_._5<br>_|_ 10_._00<br>37_._9_|_3_._57<br>**122.5**_|_**13.96**<br>56_._9_|_7_._39|
|mPLUG-Owl3<br>NoM<br>StructM<br>NLM<br>ChainM||11_|_1_._06<br>12_|_1_._08<br>13_|_1_._39<br>13_|_1_._27<br>12_|_1_._04<br>17_|_1_._46<br>13_|_1_._28<br>16_|_1_._54<br>13_._4_|_1_._27<br>15_|_ 1_._71<br>24_|_2_._34<br>41_|_4_._52<br>22_|_2_._85<br>12_|_1_._20<br>37_|_5_._07<br>39_|_4_._55<br>85_|_7_._31<br>34_._4_|_3_._69<br>19<br>_|_1_._55<br>17_|_2_._03<br>33_|_4_._16<br>27_|_2_._34<br>16<br>_|_1_._21<br>35_|_4_._48<br>39_|_3_._25<br>86_|_8_._30<br>34_._0_|_3_._42<br>15_|_1_._59<br>20_|_1_._77<br>43_|_4_._38<br>18_|_1_._92<br>12_|_1_._19<br>51_|_4_._63<br>24_|_2_._65<br>69_|_8_._82<br>31_._5_|_3_._37||
|**Human**|Expert-X<br>Expert-Y<br>Expert-Z<br>Max_|_Mean|22<br>121<br>500<br>500<br>10<br>500<br>500<br>500<br>11<br>500<br>500<br>102<br>17<br>500<br>500<br>500<br>31<br>500<br>500<br>500<br>23<br>500<br>500<br>500<br>31_|_21_._3<br>500_|_373_._7<br>500_|_500<br>500_|_367_._3<br>23_|_16_._7<br>500_|_500<br>500_|_500<br>500_|_500|331_._6<br>328_._8<br>381_._8<br>381_._8_|_347_._5|



Table 8: The Max _|_ Mean Step of synchronous attribute concept association across open-source MLLM and human experts. We select eight categories to reduce the complexity of the association and ensure the data quality. For open-source MLLMs, we involve three memory strategies, StructM, NLM, and ChainM, along with a baseline strategy, NoM. The best and second results are shown in **bold** and underline, respectively. 

|Model|Memory|Synchronous Affordance Concept Association Chain (Max_|_Mean Step)|Synchronous Affordance Concept Association Chain (Max_|_Mean Step)|
|---|---|---|---|
|||sit<br>imprint<br>push<br>carry<br>cut<br>clean<br>open<br>break|Avg.|
|LLaVA-OneVision|NoM<br>StructM<br>NLM<br>ChainM|40_|_4_._17<br>**39**_|_2_._21<br>47<br>_|_4_._34<br>14_|_1_._49<br>49_|_5_._63<br>22<br>_|_1_._60<br>53_|_ 6_._74<br>20_|_2_._06<br>38_|_4_._53<br>27_|_2_._82<br>46_|_5_._65<br>22_|_1_._86<br>51_|_**7.41**<br>16_|_**2.10**<br>73_|_ 8_._64<br>28_|_2_._86<br>50<br>_|_5_._70<br>30_|_ 2_._88<br>48_|_**6.16**<br>**26**_|_1_._95<br>45_|_6_._73<br>19_|_ 2_._07<br>**95**_|_**10.95**<br>28_|_2_._61<br>37_|_4_._76<br>34<br>_|_2_._82<br>**51**_|_ 5_._77<br>13_|_1_._59<br>57_|_5_._82<br>19_|_1_._93<br>77<br>_|_ 8_._99<br>20_|_2_._32|35_._5_|_3_._53<br>37_._6_|_4_._48<br>**42.6**_|_**4.88**<br>38_._5_|_4_._25|
|Qwen2-VL|NoM<br>StructM<br>NLM<br>ChainM|<br><br> <br> <br> <br> <br><br> <br>36_|_4_._17<br>21_|_2_._76<br>36_|_3_._96<br>15_|_1_._88<br>**74**_|_7_._15<br>18_|_1_._68<br>39_|_4_._44<br>38_|_2_._76<br>**51**_|_3_._80<br>21_|_2_._29<br>25_|_2_._39<br>19_|_1_._86<br>23_|_2_._65<br>16_|_1_._71<br>36_|_3_._10<br>18_|_2_._17<br>48_|_**6.12**<br>34<br>_|_2_._75<br>35_|_4_._42<br>25<br>_|_ 2_._61<br>68<br>_|_6_._94<br>15_|_1_._90<br>52_|_6_._37<br>32_|_3_._57<br>47_|_ 6_._08<br>22_|_**3.04**<br>37_|_4_._81<br>23_|_2_._59<br>59_|_ 7_._25<br>**24**_|_1_._99<br>63_|_7_._75<br>25_|_3_._45|<br>34_._6_|_3_._60<br>26_._1_|_2_._50<br>38_._6<br>_|_4_._34<br>37_._5_|_ 4_._62|
|mPLUG-Owl3|NoM<br>StructM<br>NLM<br>ChainM|<br> <br> <br> <br> <br> <br> <br> <br>14_|_1_._42<br>12_|_1_._00<br>13_|_1_._14<br>14_|_1_._12<br>13_|_1_._26<br>13_|_1_._02<br>10_|_1_._14<br>14_|_1_._28<br>27_|_3_._94<br>20_|_1_._83<br>27_|_3_._44<br>21_|_**2.69**<br>53_|_5_._23<br>15_|_1_._85<br>31_|_3_._14<br>40<br>_|_**5.39**<br>23_|_3_._18<br>13_|_1_._45<br>21_|_2_._30<br>21_|_1_._88<br>32_|_3_._69<br>13_|_1_._39<br>21_|_2_._24<br>34_|_3_._96<br>29_|_2_._94<br>17_|_1_._56<br>21_|_2_._68<br>24_|_1_._96<br>35_|_3_._79<br>16_|_1_._48<br>19_|_2_._34<br>**42**_|_ 4_._25|<br>12_._9_|_1_._72<br>29_._3_|_3_._44<br>22_._3_|_2_._51<br>25_._4_|_2_._63|
|**Human**<br>Expert-X<br>Expert-Y<br>Expert-Z<br>Max_|_Mean||77<br>105<br>500<br>500<br>500<br>500<br>500<br>500<br>397_._8<br>49<br>93<br>31<br>500<br>500<br>500<br>136<br>500<br>288_._7<br>42<br>78<br>500<br>500<br>500<br>73<br>500<br>500<br>336_._2<br>77_|_56_._0<br>105_|_92_._0<br>500_|_343_._7<br>500_|_500_._0<br>500_|_500_._0<br>500_|_357_._6<br>500_|_378_._7<br>500_|_500_._0<br>397_._8_|_341_._0||



Table 9: Same as Table 8, but for synchronous affordance concept association. 

|Model<br>Memory|Model<br>Memory|Synchronous Action Concept Association Chain (Max_|_Mean Step)|Synchronous Action Concept Association Chain (Max_|_Mean Step)|
|---|---|---|---|
|||run<br>hit<br>drive<br>dress<br>cooking<br>build<br>shake<br>cut|Avg.|
|LLaVA-OneVision|NoM<br>StructM<br>NLM<br>ChainM|30_|_2_._86<br>22_|_3_._08<br>31_|_3_._80<br>22_|_2_._09<br>78_|_7_._05<br>29_|_2_._93<br>36_|_3_._55<br>13_|_1_._48<br>23_|_2_._57<br>25_|_2_._99<br>32_|_3_._61<br>27<br>_|_2_._52<br>55_|_6_._49<br>30_|_3_._11<br>34_|_4_._71<br>15_|_1_._73<br>25_|_2_._95<br>27_|_3_._66<br>32_|_3_._84<br>22_|_2_._36<br>56_|_6_._61<br>29_|_2_._88<br>63_|_5_._86<br>16_|_1_._71<br>22_|_2_._60<br>27_|_3_._05<br>31_|_3_._90<br>25_|_2_._36<br>52_|_5_._37<br>25_|_2_._55<br>43_|_5_._36<br>18_|_1_._69|32_._6_|_3_._35<br>30_._1_|_3_._47<br>33_._8_|_3_._73<br>30_._4_|_3_._36|
|Qwen2-VL|NoM<br>StructM<br>NLM<br>ChainM|49<br>_|_ 5_._40<br>42<br>_|_**5.04**<br>48<br>_|_4_._90<br>21_|_2_._37<br>79_|_7_._35<br>35_|_3_._18<br>80_|_10_._85<br>24_|_1_._98<br>19_|_2_._57<br>23_|_2_._71<br>30_|_3_._15<br>17_|_2_._31<br>46_|_5_._41<br>31_|_3_._62<br>119_|_12_._56<br>20_|_2_._31<br>48_|_**6.27**<br>35_|_4_._94<br>**64**_|_**5.57**<br>**30**_|_**3.76**<br>**162**_|_**16.48**<br>**49**_|_**5.59**<br>**197**_|_**24.07**<br>24_|_2_._86<br>**51**_|_5_._06<br>**43**_|_4_._85<br>37_|_ 5_._11<br>26_|_ 3_._47<br>80<br>_|_12_._09<br>45<br>_|_ 4_._65<br>163<br>_|_ 19_._07<br>28_|_2_._64|47_._3_|_5_._13<br>38_._1_|_4_._33<br>**76.1**_|_**8.69**<br>59_._1<br>_|_ 7_._12|
|mPLUG-Owl3<br>NoM<br>StructM<br>NLM<br>ChainM||<br> <br> <br> <br><br><br><br> <br><br>12_|_1_._23<br>11_|_1_._17<br>14_|_1_._51<br>13_|_1_._14<br>20_|_2_._20<br>15_|_1_._77<br>16_|_1_._26<br>16_|_1_._49<br>14_._6_|_1_._47<br>23_|_2_._20<br>18_|_2_._01<br>24_|_3_._29<br>19_|_1_._97<br>64_|_6_._17<br>34_|_3_._90<br>24_|_1_._95<br>**40**_|_**5.00**<br>30_._8_|_3_._31<br>23_|_1_._70<br>16_|_1_._63<br>24_|_2_._49<br>18_|_1_._41<br>39_|_3_._89<br>30_|_2_._94<br>21_|_1_._98<br>39<br>_|_ 4_._59<br>26_._3_|_2_._58<br>13_|_1_._94<br>16_|_1_._71<br>37_|_3_._10<br>14_|_1_._79<br>54_|_4_._61<br>28_|_3_._12<br>22_|_1_._90<br>35_|_4_._17<br>27_._4_|_2_._79||
|**Human**|Expert-X<br>Expert-Y<br>Expert-Z<br>Max_|_Mean|14<br>70<br>30<br>17<br>500<br>31<br>77<br>19<br>22<br>14<br>10<br>10<br>500<br>15<br>26<br>17<br>7<br>22<br>5<br>37<br>500<br>37<br>31<br>90<br>22_|_14_._3<br>70_|_35_._3<br>30_|_15_._0<br>37_|_21_._3<br>500_|_500_._0<br>37_|_27_._7<br>77_|_44_._7<br>90_|_42_._0|94_._8<br>76_._8<br>91_._1<br>107_._9_|_87_._5|



Table 10: Same as Table 8, but for synchronous action concept association. 

example, in the “cooked” attribute concept association under the QWen2-VL model, NoM achieves a mean-step of 4 _._ 52, while other memory strategies fall below this. We speculate that this attribute may have encountered deduction errors, as shown in Figure 5 in the main text. The more remarkable situation is QWen2-VL with StructM strategy, in which memory prevents the utilization of correct input context. This leads to it attaining low performance compared to NoM. 

19 

Preprint. Under review. 

**Comparison of Different Categories.** Notably, the MLLM demonstrates a similar pattern to human intelligence, as both humans and open-source MLLMs achieve low mean-step scores in the “furry” and “natural” attributes, as shown in Table 8. More specifically, different categories within each concept exhibit different capabilities, which are also inconsistent with human experts. For instance, in the synchronous association of individual affordances, human experts take an average of fewer than 100 steps to complete the “sit” affordance, which presents certain challenges for testers. In comparison, QWen2-VL achieves an average of 6 _._ 12 steps, though this exceeds the steps required for the “carry” affordance. Notably, human experts are less likely to make errors when performing the “carry” affordance. 

## C.3 DETAILED ASYNCHRONOUS ASSOCIATION RESULT 

|Model<br>Memory|Model<br>Memory|Asynchronous Attribute Concept Association Chain (Max_|_Mean Step)|Asynchronous Attribute Concept Association Chain (Max_|_Mean Step)|
|---|---|---|---|
|||furry-metal<br>fresh-cooked<br>natural-ripe<br>painted-rusty|Avg.|
|LLaVA-OneVision|NoM<br>StructM<br>NLM<br>ChainM|11_|_1_._15<br>20_|_3_._07<br>18_|_0_._99<br>43_|_5_._39<br>10_|_1_._24<br>36_|_3_._52<br>17_|_1_._11<br>59_|_6_._63<br>17<br>_|_1_._36<br>32_|_4_._24<br>15_|_1_._19<br>83_|_8_._97<br>11_|_1_._21<br>32_|_3_._47<br>17_|_1_._08<br>67_|_7_._65|23_._0_|_2_._65<br>30_._5_|_3_._13<br>36_._8_|_3_._94<br>31_._8_|_3_._35|
|QWen2-VL|NoM<br>StructM<br>NLM<br>ChainM|13_|_1_._24<br>29_|_4_._47<br>16_|_1_._19<br>44_|_6_._22<br>12_|_1_._33<br>19_|_2_._55<br>18_|_1_._43<br>38_|_3_._75<br>16_|_1_._63<br>43_|_4_._69<br>19<br>_|_1_._49<br>97_|_9_._91<br>10_|_1_._42<br>32_|_4_._50<br>16_|_1_._19<br>56_|_6_._67|25_._5_|_3_._28<br>21_._8_|_2_._26<br>43_._8_|_4_._43<br>28_._5_|_3_._45|
|mPLUG-Owl3|No<br>StructM<br>NLM<br>ChainM|10_|_1_._03<br>14_|_1_._26<br>13_|_1_._04<br>16_|_1_._23<br>11_|_1_._44<br>34_|_3_._26<br>16_|_1_._16<br>39_|_4_._28<br>12_|_1_._46<br>24_|_3_._07<br>11_|_1_._25<br>25_|_2_._62<br>12_|_1_._32<br>23_|_2_._68<br>**22**_|_1_._13<br>24_|_2_._77|13_._3_|_1_._14<br>25_._0_|_2_._54<br>18_._0_|_2_._10<br>20_._3_|_1_._98|
|**MoE**|No<br>StructM<br>NLM<br>ChainM|8_|_1_._11<br>28_|_4_._06<br>19<br>_|_1_._05<br>56_|_7_._05<br>13_|_1_._39<br>49_|_3_._93<br>13_|_1_._25<br>96_|_8_._34<br>**19**_|_1_._53<br>39_|_5_._26<br>17_|_1_._26<br>137_|_15_._91<br>12_|_1_._36<br>39_|_4_._40<br>12_|_1_._17<br>82_|_11_._38|27_._8_|_3_._32<br>42_._8_|_3_._73<br>53_._0_|_5_._99<br>36_._3_|_4_._58|
|**Gemini-1.5-Flash**|No<br>StructM<br>NLM<br>ChainM|7_|_1_._68<br>18_|_5_._38<br>8_|_1_._84<br>15_|_3_._35<br>9_|_2_._46<br>**76**_|_**13.8**<br>6_|_2_._46<br>186<br>_|_ 49_._5<br>9_|_2_._80<br>31_|_8_._85<br>8_|_2_._30<br>**305**_|_**97.3**<br>8_|_1_._53<br>24_|_5_._30<br>8_|_2_._43<br>51_|_14_._9|12_._0_|_3_._06<br>69_._3<br>_|_ 17_._1<br>**88.3**_|_**27.8**<br>22_._8_|_6_._04|
|**GPT-4V**|No<br>StructM<br>NLM<br>ChainM|7_|_2_._07<br>53_|_ 12_._1<br>8_|_ 2_._93<br>121_|_24_._3<br>15_|_**4.33**<br>39_|_8_._63<br>6_|_**3.10**<br>151_|_37_._3<br>8_|_ 3_._07<br>53_|_10_._5<br>7_|_2_._33<br>168_|_45_._5<br>7_|_2_._35<br>54<br>_|_9_._70<br>7_|_1_._95<br>122_|_38_._2|47_._3_|_10_._4<br>52_._8_|_13_._3<br>59_._0_|_15_._4<br>47_._5_|_13_._1|
|**Human**<br>Expert-X<br>Expert-Y<br>Expert-Z<br>Max_|_Mean||16<br>70<br>16<br>500<br>150_._5<br>21<br>54<br>80<br>500<br>163_._8<br>79<br>42<br>11<br>295<br>106_._8<br>79_|_38_._7<br>70_|_55_._3<br>80_|_35_._7<br>500_|_431<br>182_._3_|_140_._2||



Table 11: The Max _|_ Mean step on paired attribute categories in asynchronous association setting. We involve three open-source MLLMs, a cutting-edged MoE based on open-source MLLMs, two close-sourced MLLMs, and human experts. In open-source and close-source MLLM, we involve three memory strategies, SturctM, NLM, and ChainM, as well as one baseline strategy NoM. The best and second results are shown in **bold** and underline, respectively. 

Paired Concept Association is the most complex task in our work. We find that all max-step and mean-step decrease in this setting compared to individual concepts association in the main text. In this subsection, we detailed analysis of the paired categories association on adjectives and verb concepts, _i.e._ , attribute and affordance in OCL, and action in Pangea. 

Table 11, 12, 13 summarise the results of paired categories concept association on attribute, affordance and action respectively. In this section, we include the closed-source MLLMs Gemini1.5-Flash and GPT-4V, along with the open-source MLLMs LLaVA-OneVision, QWen2-VL, and mPLUG-Owl3, as well as an MoE that integrates all open-source MLLMs and human expertise. We include eight categories from each concept of attribute, affordance, and action. 

**Comparison of Different Models and Memory Strategies.** All the max and mean steps decrease on each paired category by comparing with individual concept association, which is consistence 

20 

Preprint. Under review. 

|Model<br>Memory|Model<br>Memory|Asynchronous Affordance Concept Association Chain (Max_|_Mean Step)|Asynchronous Affordance Concept Association Chain (Max_|_Mean Step)|
|---|---|---|---|
|||sit-imprint<br>push-carry<br>cut-clean<br>open-break|Avg.|
|LLaVA-OneVision|NoM<br>StructM<br>NLM<br>ChainM|29_|_3_._13<br>**31**_|_**3.26**<br>18_|_3_._02<br>15_|_3_._02<br>**41**_|_3_._42<br>13_|_2_._41<br>**22**_|_2_._92<br>29_|_3_._82<br>34_|_4_._09<br>13_|_2_._70<br>18_|_3_._45<br>21_|_4_._12<br>37_|_3_._56<br>12_|_2_._65<br>15_|_3_._19<br>21_|_3_._77|23_._3_|_3_._11<br>26_._3<br>_|_3_._14<br>21_._5_|_3_._59<br>21_._3_|_3_._29|
|QWen2-VL|NoM<br>StructM<br>NLM-3<br>ChainM|26_|_3_._30<br>16_|_2_._58<br>22_|_3_._70<br>19_|_2_._93<br>28_|_2_._58<br>17_|_1_._89<br>18_|_2_._13<br>17_|_2_._51<br>29_|_4_._05<br>16_|_ 2_._98<br>20_|_3_._29<br>28_|_4_._18<br>25_|_3_._54<br>14_|_2_._79<br>20_|_3_._55<br>32<br>_|_3_._93|20_._8_|_3_._13<br>20_|_2_._28<br>23_._3_|_3_._63<br>22_._8_|_3_._45|
|mPLUG-Owl3<br>NoM<br>StructM<br>NLM<br>ChainM||13_|_1_._31<br>12_|_1_._10<br>14_|_1_._23<br>10_|_1_._27<br>12_._3_|_1_._23<br>20_|_2_._75<br>22<br>_|_1_._90<br>21_|_3_._43<br>21_|_3_._19<br>21_._0_|_2_._82<br>15_|_2_._19<br>14_|_1_._59<br>20_|_2_._64<br>17_|_2_._56<br>16_._5_|_2_._25<br>15_|_2_._19<br>16_|_2_._05<br>20_|_3_._08<br>16_|_2_._70<br>16_._8_|_2_._51||
|**MoE**|NoM<br>StructM<br>NLM<br>ChainM|26_|_3_._08<br>19_|_2_._44<br>21<br>_|_3_._10<br>15_|_3_._25<br>27_|_3_._40<br>14_|_2_._41<br>18_|_3_._16<br>23_|_3_._84<br>38<br>_|_**6.41**<br>20_|_2_._93<br>16_|_3_._66<br>24_|_4_._42<br>36_|_3_._65<br>16_|_2_._95<br>20_|_3_._62<br>21_|_4_._17|20_._3_|_2_._97<br>20_._5_|_3_._20<br>24_._5_|_ 4_._36<br>23_._4_|_3_._60|
|**Gemini-1.5-Flash**|NoM<br>StructM<br>NLM<br>ChainM|26_|_3_._36<br>11_|_2_._56<br>14_|_3_._12<br>14_|_4_._26<br>32_|_4_._67<br>8_|_2_._37<br>19_|_ 3_._73<br>**68**_|_**6.93**<br>13_|_2_._90<br>7_|_2_._30<br>8_|_3_._07<br>24_|_ 5_._43<br>11_|_3_._67<br>10_|_2_._93<br>9_|_2_._10<br>15_|_3_._27|16_._3_|_3_._33<br>**31.8**_|_**4.43**<br>13_._0_|_3_._43<br>11_._3_|_2_._99|
|**GPT-4V**|NoM<br>StructM<br>NLM<br>ChainM|17_|_6_._00<br>14_|_2_._67<br>14_|_3_._43<br>8_|_3_._00<br>29_|_4_._70<br>7_|_1_._70<br>13_|_**4.00**<br>19_|_4_._61<br>23_|_ 6_._17<br>9_|_2_._21<br>14_|_3_._57<br>24_|_4_._45<br>16_|_4_._71<br>10_|_2_._43<br>9_|_3_._36<br>13_|_5_._14|13_._3_|_3_._78<br>17_._0_|_3_._75<br>17_._5_|_4_._10<br>12_._0_|_3_._91|
|**Human**<br>Expert-X<br>Expert-Y<br>Expert-Z<br>Max_|_Mean||199<br>19<br>29<br>134<br>95_._25<br>152<br>47<br>17<br>500<br>179_._0<br>96<br>13<br>32<br>500<br>160_._3<br>199_|_149<br>47_|_26_._3<br>32_|_26_._0<br>500_|_378<br>194_._5_|_144_._8||



Table 12: Same as Table 11, but for asynchronous affordance association. 

with the whole results. We speculate that this stems from a weak ability to transfer knowledge between concepts, as human testers also require some time to adjust to new rules. Additionally, MoE outperforms open-source MLLMs in some paired categories, where there also exists some cases in which MoE attains a balanced performance compared to one of the highest MLLMs. For instance, for paired categories of “sit-imprint” and “open-break”, MoE outperforms all open-source MLLM in all cases, while “push-carry” and “cut-clean” open-source MLLM leads the performance when comparing the open-source setting. On the other hand, different memory strategies remain inconsistent tendencies that they improve performance in different cases. For instance, Gemini1.5-Flash with StuctM in paired fresh cooked attribute association leads other MLLM and other memory strategies, while QWen2-VL with ChainM in shake-cut action association makes leading. This further highlights the varying comprehension abilities of MLLMs when presented with the same input contexts, which also provides insight for improving the understanding of language at specific dimensions. 

**Comparison of Different Categories.** Different paired categories exhibit specific abilities in asynchronous association. For instance, the paired categories of “fresh-cooked” and “painted-rusty” have the mean-step of asynchronous step on open-source MLLMs lower than 2 _._ 0, while the “freshcooked” and “painted-rusty” have the mean-step of asynchronous step greater than 3 _._ 0. We speculate that this is due to two different aspects reasons. One is the insufficient perception of specific categories within one concept, and the other is the lack of consciousness to convert the concept. 

## C.4 DETAILED ABLATION STUDY 

In this section, we supplement the experiment with the ablation study from the main text, concluding that the memory strategy with three examples shows the most comprehensive performance across all experiments. We below analyze the differences between different categories. 

Table 14 shows the detailed results of the ablation study of example sizes. We observed that while the average max and mean step slightly differ between different example sizes, there are notable 

21 

Preprint. Under review. 

|Model<br>Memory|Model<br>Memory|Asynchronous Verb Concept Association Chain (Max_|_Mean Step)|Asynchronous Verb Concept Association Chain (Max_|_Mean Step)|
|---|---|---|---|
|||run-hit<br>drive-dress<br>cooking-build<br>shake-cut|Avg.|
|LLaVA-OneVision|NoM<br>StructM<br>NLM<br>ChainM|20_|_2_._74<br>21_|_2_._65<br>29_|_5_._44<br>20_|_2_._78<br>20_|_2_._39<br>14_|_2_._24<br>35_|_4_._16<br>21_|_2_._59<br>31_|_2_._87<br>35<br>_|_2_._85<br>31_|_4_._96<br>15_|_3_._05<br>21_|_2_._22<br>23_|_2_._39<br>23_|_4_._54<br>23_|_2_._87|22_._5_|_3_._40<br>22_._5_|_2_._85<br>28_._0_|_3_._43<br>22_._5_|_3_._01|
|QWen2-VL|NoM<br>StructM<br>NLM<br>ChainM|37_|_4_._65<br>19_|_2_._56<br>31_|_5_._99<br>23_|_4_._73<br>22_|_2_._51<br>23_|_2_._34<br>28_|_4_._52<br>23_|_3_._90<br>**44**_|_4_._97<br>**36**_|_3_._94<br>63<br>_|_8_._31<br>**35**_|_ 5_._43<br>28_|_4_._06<br>20_|_3_._34<br>44_|_7_._80<br>31<br>_|_**5.79**|27_._5_|_4_._48<br>24_._0_|_3_._32<br>**44.5**_|_ 5_._66<br>30_._8_|_5_._25|
|mPLUG-Owl3|NoM<br>StructM<br>NLM<br>ChainM|12_|_1_._17<br>12_|_1_._09<br>20_|_2_._20<br>16_|_1_._76<br>15_|_1_._75<br>15_|_1_._72<br>28_|_4_._84<br>21_|_2_._74<br>10_|_1_._46<br>16_|_1_._37<br>31_|_4_._17<br>20_|_2_._86<br>16_|_1_._59<br>13_|_1_._43<br>28_|_4_._43<br>28_|_2_._95|15_._0_|_1_._56<br>19_._8_|_2_._76<br>19_._3_|_2_._47<br>21_._3_|_2_._60|
|**MoE**|NoM<br>StructM<br>NLM<br>ChainM|39_|_4_._13<br>23_|_3_._15<br>35_|_6_._39<br>20_|_3_._89<br>32_|_2_._74<br>21_|_2_._58<br>35_|_4_._75<br>22_|_3_._42<br>41<br>_|_4_._40<br>29_|_3_._50<br>39_|_6_._55<br>30_|_4_._24<br>31_|_3_._55<br>27_|_3_._19<br>40_|_6_._93<br>29_|_4_._29|29_._3_|_4_._39<br>27_._5_|_3_._37<br>34_._8<br>_|_4_._67<br>31_._8_|_4_._49|
|**Gemini-1.5-Flash**|NoM<br>StructM<br>NLM<br>ChianM|27_|_5_._40<br>15_|_3_._90<br>15_|_4_._86<br>16_|_3_._03<br>16_|_3_._97<br>21_|_3_._47<br>**75**_|_10_._6<br>12_|_4_._63<br>13_|_3_._90<br>13_|_2_._70<br>39_|_**11.7**<br>20_|_4_._23<br>25_|_ 5_._50<br>14_|_2_._90<br>18_|_5_._07<br>17_|_5_._20|18_._3_|_4_._30<br>31_._0_|_**5.67**<br>18_._0_|_5_._63<br>18_._5_|_4_._67|
|**GPT-4V**|NoM<br>StructM<br>NLM<br>ChainM|17_|_4_._45<br>7_|_1_._45<br>20_|_4_._60<br>14_|_2_._90<br>22_|_4_._10<br>17_|_**3.95**<br>35_|_6_._80<br>14_|_3_._85<br>33_|_**6.70**<br>18_|_3_._60<br>39_|_8_._11<br>13_|_3_._28<br>16_|_3_._75<br>7_|_2_._50<br>26_|_4_._30<br>11_|_3_._40|14_._5_|_3_._35<br>22_._0_|_4_._68<br>25_._8_|_5_._42<br>15_._0_|_3_._49|
|**Human**<br>Expert-X<br>Expert-Y<br>Expert-Z<br>Max_|_Mean||64<br>12<br>39<br>12<br>30_._5<br>10<br>32<br>26<br>59<br>30_._5<br>19<br>17<br>77<br>35<br>39_._5<br>64_|_31_._0<br>32_|_20_._3<br>77_|_47_._3<br>59_|_35_._3<br>58_._0_|_33_._5||



Table 13: Same as Table 11, but for paired action synchronous association. 

|Model<br>Memory|Model<br>Memory|Paired Attribute Concept Association Chain (Max_|_Mean Step)|Paired Attribute Concept Association Chain (Max_|_Mean Step)|
|---|---|---|---|
|||furry-metal<br>fresh-cooked<br>natural-ripe<br>painted-rusty|Avg.|
|LLaVA-OneVision|StructM-1<br>StructM-3<br>StructM-5<br>NLM-1<br>NLM-3<br>NLM-5<br>ChainM-1<br>ChainM-3<br>ChainM-5|11_|_1_._28<br>**36**_|_3_._41<br>**27**_|_1_._18<br>47_|_7_._02<br>**41**_|_**3.42**<br>13_|_2_._41<br>22<br>_|_**2.92**<br>29_|_3_._82<br>12_|_1_._47<br>34<br>_|_3_._61<br>17_|_ 1_._36<br>67_|_6_._24<br>14_|_1_._27<br>32_|_4_._08<br>11_|_1_._07<br>76_|_**9.26**<br>17<br>_|_1_._36<br>32_|_ 4_._24<br>15_|_1_._19<br>83<br>_|_ 8_._97<br>16_|_1_._31<br>33_|_**4.27**<br>11_|_1_._15<br>**86**_|_8_._86<br>12_|_1_._27<br>28_|_3_._30<br>12_|_1_._06<br>61_|_7_._73<br>11_|_1_._21<br>32_|_3_._47<br>17_|_1_._08<br>67_|_7_._65<br>11_|_1_._21<br>22_|_3_._18<br>15_|_1_._01<br>83<br>_|_7_._99|30_._3_|_3_._22<br>26_._3_|_3_._14<br>32_._5_|_3_._17<br>33_._3_|_ 3_._92<br>**36.8**_|_**3.94**<br>36_._5<br>_|_3_._90<br>28_._3_|_3_._34<br>31_._8_|_3_._35<br>32_._8_|_3_._35|
|Qwen2-VL|StructM-1<br>StructM-3<br>StructM-5<br>NLM-1<br>NLM-3<br>NLM-5<br>ChainM-1<br>ChainM-3<br>ChainM-5|13_|_1_._26<br>21_|_2_._63<br>19<br>_|_1_._19<br>33_|_ 3_._13<br>12_|_1_._33<br>19_|_2_._55<br>18_|_1_._43<br>38_|_ 3_._75<br>15_|_1_._36<br>19_|_2_._39<br>18_|_1_._37<br>34_|_ 3_._26<br>15_|_1_._42<br>**44**_|_ 4_._59<br>16_|_1_._24<br>72_|_ 10_._18<br>16<br>_|_ 1_._63<br>43<br>_|_**4.69**<br>19_|_1_._49<br>**97**_|_ 9_._91<br>13_|_1_._54<br>32_|_**4.69**<br>16_|_1_._35<br>94<br>_|_**10.52**<br>11_|_1_._38<br>29_|_4_._19<br>11_|_1_._14<br>60_|_ 6_._94<br>**25**_|_**3.54**<br>14_|_2_._79<br>**20**_|_**3.55**<br>32_|_ 3_._93<br>11_|_1_._27<br>26_|_4_._51<br>17_|_1_._20<br>46_|_ 6_._58|21_._5_|_2_._05<br>21_._8_|_2_._26<br>21_._5_|_2_._09<br>36_._8_|_4_._36<br>**43.8**_|_ 4_._43<br>38_._8<br>_|_**4.53**<br>27_._8_|_3_._41<br>22_._8_|_3_._45<br>25_._0_|_3_._39|



Table 14: Ablation study of the example sizes in synchronous and asynchronous association, which we compared by the results of Max _|_ Mean Step. _{_ Memory strategy _}_ -X indicates the X example samples involved in the preview of the association. In this setting, we are implemented in opensource MLLMs LLaVA-OneVision and QWen2-VL with three memory strategies. The best and second results are shown in **bold** and underline, respectively. 

gaps within specific paired categories. For instance, in the case of LLaVA-OneVision with paired attribute “furry-metal”, StructM-3 outperforms StructM-5 by approximately the mean-step of 1 _._ 95. Furthermore, while an average mean-step of 3 outperforms in most cases, the example size of 1 

22 

Preprint. Under review. 

and 5 also leads in certain cases. For instance, with QWen2-VL on “painted-rusty” affordance and LLaVA-OneVision on “fresh-cooked” attribute, the example size of 5 outperforms the others. 

## D COMPARISON WITH MORE CONCEPTS 

|Model<br>Memory|Model<br>Memory|Paired Action Concept Association Chain (Max_|_Mean Step)|Paired Action Concept Association Chain (Max_|_Mean Step)|
|---|---|---|---|
|||brush<br>hair-dive<br>clap-hug<br>shake<br>~~h~~ands-sit<br>smoke-eat|Avg.|
|QWen-VL|NoM<br>StructM<br>NLM|10_|_1_._02<br>11_|_1_._14<br>9_|_1_._04<br>10_|_1_._07<br>10_|_1_._03<br>11_|_1_._15<br>9_|_1_._17<br>10_|_1_._14<br>10_|_1_._04<br>11_|_1_._15<br>9_|_1_._11<br>10_|_1_._04|10_._0_|_1_._07<br>10_._0_|_1_._12<br>10_._0_|_1_._09|
|**Gemini-Pro-Vision**|NoM<br>StructM<br>NLM|15_|_6_._00<br>10_|_6_._00<br>10_|_4_._20<br>5_|_1_._80<br>**36**_|_10_._5<br>24<br>_|_ 8_._44<br>12_|_4_._20<br>14_|_ 5_._60<br>18_|_8_._10<br>20_|_8_._10<br>**19**_|_5_._70<br>**31**_|_**8.60**|10_._0_|_4_._50<br>21_._5_|_7_._19<br>22_._0<br>_|_6_._25|
|**GPT-4V**<br>NoM<br>StructM<br>NLM||19_|_6_._20<br>5_|_2_._60<br>12_|_4_._30<br>9_|_2_._90<br>11_._3_|_4_._00<br>23_|_12_._0<br>**37**_|_**11.0**<br>17<br>_|_**7.70**<br>14_|_4_._10<br>**22.8**_|_**8.70**<br>35<br>_|_**15.6**<br>17_|_6_._60<br>10_|_3_._10<br>19<br>_|_4_._00<br>20_._3_|_7_._18||
|**Human**<br>Expert-X<br>Expert-Y<br>Expert-Z<br>Max_|_Mean||45<br>33<br>82<br>17<br>44_._3<br>61<br>182<br>38<br>86<br>94_._5<br>172<br>108<br>44<br>28<br>88_._0<br>172_|_92_._7<br>182_|_107_._7<br>82_|_54_._7<br>86_|_43_._7<br>130_._5_|_74_._7||



Table 15: The Max _|_ Mean Step on paired action concept association from HMDB dataset in asynchronous association setting. We implemented at open-source MLLM QWen-VL, close-sourced MLLM Gemini-Pro-Vision and GPT4-V, as well as human experts. The best and second results are shown in **bold** and underline, respectively. 

While we provide comprehensive experiments to benchmark the MLLM’s performance in association tasks, we further expand an experiment on verb concepts to attain a solid perspective on association tasks. Specifically, we conduct asynchronous association on actions from the HMDB dataset, which is implemented in the open-source MLLM QWen-VL and closed-source MLLMs Gemini-Pro-Vision and GPT-4V. Simultaneously, we have integrated HMDB datasets into our human test interface. Furthermore, we include two memory strategies in this experiment, StructM, and NLM, as well as compare them against the baseline NoM. 

Table 15 summarises the result of asynchronous association on the action from the HMDB dataset. Actions in the HMDB dataset are relatively easier than the actions in the Pangea dataset, which is reflected in the comparison of the mean-step at human experts between the different datasets in Table 13 and Table 15. Additionally, the open-source MLLM QWen-VL has a significant gap between closed-source MLLMs Gemini-Pro-Vision and GPT-4V. We speculate that this is due to the earlier version of QWen-VL, which had limited capabilities in multi-image perception. This has been effectively improved in the new QWen2-VL version, as demonstrated by the comparison between Table 13 and Table 15. 

## E COMPARISON WITH UNFILTERED RAW DATA 

|Model<br>MLLM<br>Verifcation|Model<br>MLLM<br>Verifcation|Individual Attribute Concept Association Chain (Max_|_Mean Step)|Individual Attribute Concept Association Chain (Max_|_Mean Step)|
|---|---|---|---|
|||NoM<br>StructM<br>NLM<br>ChainM|Avg.|
|LLaVA-OneVision|No<br>Yes|26_._8_|_3_._05<br>27_._1_|_3_._36<br>25_._9_|_3_._05<br>29_._1_|_3_._10<br>48_._0_|_5_._30<br>60_._6_|_5_._80<br>75_._4_|_8_._91<br>60_._8_|_7_._03|27_._2_|_3_._14<br>61_._2_|_6_._76|



Table 16: Comparison of whether MLLM verification on the individual attribute association. We are implemented in the LLaVA-OneVision with three memory strategies and one baseline. The comparison of “Yes” and “No” indicates the effectiveness of the MLLM verification in data refinement. 

It is noted that we involve three comprehensive steps in data refinement to ensure the data quality is met in the association task. We below involve the experiment for comparison of the effectiveness and difference with and without the MLLM Verification step. 

23 

Preprint. Under review. 

|Model|Concept<br>MLLM<br>Verifcation|Paired Concept Association Chain (Max_|_Mean Step)|Paired Concept Association Chain (Max_|_Mean Step)|
|---|---|---|---|
|||NoM<br>StructM<br>NLM<br>ChainM|Avg.|
|LLaVA-OneVision|Attribute<br>No<br>Yes|20_._0_|_2_._12<br>22_._3_|_2_._47<br>25_._3_|_2_._78<br>25_._8_|_2_._43<br>23_._0_|_2_._65<br>30_._5_|_3_._13<br>36_._8_|_3_._94<br>31_._8_|_3_._35|23_._3_|_2_._45<br>30_._5_|_3_._27|
||Affordance<br>No<br>Yes|13_._5_|_1_._91<br>16_._0_|_2_._09<br>11_._8_|_2_._06<br>12_._5_|_2_._05<br>23_._3_|_3_._11<br>26_._3_|_3_._14<br>21_._5_|_3_._59<br>21_._3_|_3_._29|13_._5_|_2_._03<br>23_._1_|_3_._28|
||Action<br>No<br>Yes|18_._0_|_2_._60<br>18_._8_|_2_._73<br>20_._5_|_2_._67<br>17_._3_|_2_._70<br>22_._5_|_3_._40<br>22_._5_|_2_._85<br>28_._0_|_3_._43<br>22_._5_|_3_._01|18_._7_|_2_._68<br>23_._9_|_3_._17|
|QWen2-VL|Attribute<br>No<br>Yes|22_._3_|_2_._47<br>18_._0_|_2_._01<br>31_._8_|_3_._22<br>23_._0_|_2_._76<br>25_._5_|_3_._28<br>21_._8_|_2_._26<br>43_._8_|_4_._43<br>28_._5_|_3_._45|23_._8_|_2_._62<br>29_._9_|_3_._35|
||Affordance<br>No<br>Yes|16_._5_|_2_._13<br>15_._3_|_1_._87<br>23_._3_|_2_._60<br>17_._0_|_2_._63<br>20_._8_|_3_._13<br>20_._0_|_2_._28<br>23_._3_|_3_._63<br>22_._8_|_3_._45|18_._0_|_2_._31<br>21_._7_|_3_._12|
||Action<br>No<br>Yes|21_._5_|_3_._19<br>19_._8_|_2_._74<br>27_._8_|_3_._76<br>23_._8_|_3_._67<br>27_._5_|_4_._48<br>24_._0_|_3_._32<br>44_._5_|_5_._66<br>30_._8_|_5_._25|23_._2_|_3_._34<br>31_._7_|_4_._68|



Table 17: Same as Table 16, but for comparison on the Paired categories attribute association. Notably, we expand the model to LLaVA-OneVision and QWen2-VL with three different concepts. 

Table 16 and 17 summarise the comparison of results for individual concept or paired concepts association, respectively. For comparison of individual concept association, we utilize LLaVAOneVision with four memory strategies. Additionally, for paired concepts association, we provide a broader implementation that includes two open-source MLLMs, LLaVA-OneVision and QWen-VL, with three different concepts. In this section, we mainly compare the results of with or without MLLM Verification, _i.e._ , comparing the line of “Yes” to the line of “No”. 

From the result, we easily find that the result with MLLM verification outperforms without MLLM verification, _i.e._ , the line of “Yes” larger than “No”. This demonstrates that MLLM Verification effectively reduces the sample with potentially confusing annotations or those requiring powerful advanced perception abilities beyond the current capabilities of MLLMs. More interesting, while the MLLM Verification step reduces the complexity of perception on object concept understanding, we also observe a similar pattern between with and without MLLM verification, _i.e._ , the tendency is consistent when comparing each line. 

## F ETHIC REVIEW 

As shown in Figure 7, in our human evaluation protocol, we have implemented a comprehensive Ethic Report mechanism to proactively address and mitigate potential ethical concerns. The user interface incorporates dedicated options for participants to report issues related to privacy or other ethical considerations. This is facilitated through a structured reporting system comprising categorical buttons and an open-ended text field for additional context. This approach enables active participant engagement in ethical oversight during the evaluation phase, fostering a collaborative approach to responsible AI development. We prioritize transparency by empowering participants to articulate concerns on potential privacy infringements, algorithmic bias, or instances where the system may induce discomfort or exhibit opaque behavior. This methodology aligns with the best practices delineated by Zaldivar et al. (2019) Kennedy-Mayo & Gord (2024), which emphasizes the criticality of integrating transparency and user feedback mechanisms to ensure fairness and accountability in machine learning systems. 

Besides the ethical review implemented within our human testing demo, it is crucial to emphasize that our approach builds upon the previous ethical review of the original datasets. They have undergone a rigorous ethical review process, particularly concerning data sourcing, privacy considerations, and bias mitigation. This prior evaluation also sets a strong foundation for ethical safeguards. 

24 

