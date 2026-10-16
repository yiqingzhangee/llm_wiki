Under Review 

# - ACTIONREASONINGBENCH: REASONING ABOUT AC - TIONS WITH AND WITHOUT RAMIFICATION CON STRAINTS 

**Divij Handa**[1] _[∗]_ **, Pavel Dolin**[1] _[∗]_ **, Shrinidhi Kumbhar**[1] _[∗]_ **, Tran Cao Son**[2] **, Chitta Baral**[1] 1Arizona State University, 2New Mexico State University 

_{_ dhanda,pdolin,skumbha4,chitta _}_ @asu.edu, stran@nmsu.edu 

## ABSTRACT 

Reasoning about Actions and Change (RAC) has historically played a pivotal role in solving foundational AI problems, such as the frame problem. It has driven advancements in AI fields, such as non-monotonic and commonsense reasoning. RAC remains crucial for AI systems that operate in dynamic environments, engage in interactive scenarios, or rely on commonsense reasoning. Despite substantial advances made by Large Language Models (LLMs) in various AI domains, their performance in RAC remains underexplored. To address this gap, we introduce a new diagnostic benchmark, ACTIONREASONINGBENCH, which encompasses 8 domains and includes questions for up to 19 action sequences. This benchmark rigorously evaluates LLMs across six key RAC dimensions: _Fluent Tracking_ , _State Tracking_ , _Action Executability_ , _Effects of Actions_ , _Numerical RAC_ , and _Composite Questions_ . LLMs demonstrate average accuracy rates of 73.55%, 65.63%, 58.73%, and 62.38% on the former four dimensions, which are frequently discussed in RAC literature. However, the performance on the latter two dimensions, which introduce complex and novel reasoning questions, the average performance of LLMs is lowered to 33.16% and 51.19%, respectively, reflecting a 17.9% performance decline. We also introduce new ramification constraints to capture the indirect effects of actions, providing deeper insights into RAC challenges. Our evaluation of state-of-the-art LLMs, including both opensource and commercial models, reveals challenges across all RAC dimensions, particularly in handling ramifications, with GPT-4o failing to solve any question and o1-preview achieving a score of only 18.4%. 

## 1 INTRODUCTION 

Reasoning about actions and change (RAC) is a fundamental problem in artificial intelligence, with its roots tracing back to early work from the 1960s (McCarthy et al., 1963). Initially, research focused on developing logical systems capable of effectively modeling and reasoning about actions and their effects in a dynamic environment. One of the significant challenges in this domain has been succinctly expressing how actions influence changeable properties of the world, known as **fluents** . For example, consider the statement: “Moving an object from location X to location Y results in the object being at location Y.” While it is relatively straightforward to describe the direct effects on the affected fluents, such as the object’s location, it is much more complex to account for the unaffected fluents, a challenge known as the _frame problem_ . This challenge becomes exacerbated when the descriptions involve relationships between fluents in a state, such as “an object can not be at two different places at the same time”. While such constraints simplify action descriptions by decoupling them from fluents, they introduce indirect effects, or **ramifications** . For example, the statement “A block is said to be clear if there isn’t any block on top of it” describes a ramification fluent, “clear” dependent on another fluent “on top of.” 

It took multiple decades of research to create a comprehensive logical formalization that adequately addressed these issues. It involved the laborious creation of numerous handcrafted rules and logic 

> _∗_ Equal contribution. 

1 

Under Review 

detailing the effects and preconditions of actions (Reiter, 2001). However, these tools are limited since they rely on manual effort to translate natural language descriptions of actions and their effects into formal logic representations. To address this challenge, recent research in natural language processing (NLP) has begun exploring the capabilities of LLMs in RAC tasks, as demonstrated by works of He et al. (2023), Spiliopoulou et al. (2022), and Banerjee et al. (2020). However, these studies have not systematically decomposed the complex RAC problem into multiple categories and overlook the critical ramifications of actions seen in real-world scenarios. To address this gap, we introduce ACTIONREASONINGBENCH, a diagnostic RAC benchmark that aims to pinpoint where modern state-of-the-art LLMs struggle. 

We decompose the RAC task into six distinct categories— _Fluent Tracking_ , _State Tracking_ , _Action Executability_ , _Effects of Actions_ , _Numerical RAC_ , and _Composite Questions_ . The first four categories focus on assessing fundamental aspects of RAC, while the latter two introduce more complex and novel question types. The questions in every category span action sequences ranging from 1 to 19 steps, allowing us to test the RAC capabilities at a series of action sequence ranges. Additionally, we introduce ramification constraints to represent the indirect effect of actions. These constraints simplify action descriptions and align more closely with real-world conditions but introduce additional complexity, as highlighted by McIlraith (2000). Specifically, we expand the domains by adding ramification fluents with dependencies up to four levels deep, where actions propagate their effects through multiple layers. 

Highlights of our benchmark, ACTIONREASONINGBENCH, along with the comparison to previous benchmarks on RAC, are presented in Table 1. We evaluate four LLMs–two open-source models, Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct (Dubey et al., 2024), as well as two leading proprietary models, GPT-4o (Achiam et al., 2023) and o1-preview (OpenAI, 2024). These LLMs were tested on ACTIONREASONINGBENCH across various RAC categories under different prompt settings, including Zero-shot-CoT (Kojima et al., 2022) and Few-shot-3 (Brown, 2020), to assess how performance varies based on these configurations. 

Our findings indicate that LLMs face substantial challenges, particularly when addressing complex RAC questions. The average performance of all LLMs on the complex categories decreases by 17.88% compared to the first four basic categories. The best performing LLM, GPT-4o, achieves an average accuracy of 59.91% on these categories. Notably, GPT-4o failed to produce any correct answers for questions involving ramifications constraints, while the o1preview model achieved an accuracy of only 18.42%. Performance was especially poor in categories like _Action Executability_ , _Numerical RAC_ , and _Composite Questions_ , with further declines observed as the length of action sequences increased. Additionally, LLMs struggled with rea- 

||**PlanBench**|**TRAC**|**(Ours)**|
|---|---|---|---|
|Number of domains<br>Number of queries<br>Max Action Sequence length<br>Max number of objects<br>Binary Questions (T/F)<br>Free Answers|2<br>26k<br>48<br>24<br>×<br>✓|1<br>15k<br>3<br>5<br>✓<br>×|8<br>152k<br>19<br>28<br>✓<br>✓|
|State Tracking<br>Action Executability|✓<br>✓|✓<br>✓|✓<br>✓|
|Fluent Tracking<br>Effects of Actions<br>Numerical Reasoning<br>Composite Questions<br>Ramifcations Constraints<br>Subcategories of Fluents|×<br>×<br>×<br>×<br>×<br>×|×<br>×<br>×<br>×<br>×<br>×|✓<br>✓<br>✓<br>✓<br>✓<br>✓|



Table 1: Differences between ACTIONREASONINGBENCH (Ours) and previous benchmarks on RAC. PlanBench (Valmeekam et al., 2024a) ; TRAC (He et al., 2023) 

soning in scenarios that combined both true and false fluents, experiencing an average performance drop of 12.16% compared to tasks involving exclusively true or false fluents. 

## 2 RELATED WORKS 

**Benchmarking reasoning capabilities of LLMs** Evaluating the reasoning capabilities of LLMs has become a key focus in NLP, with increasing efforts to create challenging benchmarks. Notable areas of interest include the evaluation of legal reasoning (Fei et al., 2023) (Guha et al., 2023), logical reasoning (Luo et al., 2024) (Han et al., 2024) (Patel et al., 2024) (Parmar et al., 2024) arithmetic 

2 

Under Review 

reasoning (Cobbe et al., 2021) (Miao et al., 2021), temporal reasoning (Uddin et al., 2024) (Fatemi et al., 2024), and commonsense reasoning (Onoe et al., 2021) (Lin et al., 2021) (Geva et al., 2021) (Lourie et al., 2021). Despite this progress, RAC remains significantly under-explored, even though it plays a crucial role in several of these reasoning tasks, such as commonsense and legal reasoning. 

**RAC and Planning** Previous works, such as those by Valmeekam et al. (2024a), Valmeekam et al. (2024b), and Guan et al. (2023), have curated benchmarks assessing the performance of LLMs on planning, with limited exploration of RAC. Banerjee et al. (2020) examined the RAC capabilities of LLMs across four domains, focusing only on three question types. Similarly, He et al. (2023) tested LLMs on RAC in a modified version of Blocksworld, employing four question categories—two of which centered on RAC, while the other two emphasized planning. In contrast, our work explores 8 distinct domains and includes 6 different question categories. Additionally, we introduce novel ramifications constraints, adding complexity to the RAC task by introducing indirect effects of actions. 

## 3 ACTIONREASONINGBENCH 

This section provides a detailed overview of our benchmark, including its categorization, creation methodology, and validation process. A sample instance is presented in Appendix A, where we also describe the objects, actions, and fluents within the domain. 

## 3.1 QUESTION CATEGORIES 

In our benchmark, the questions are organized into six distinct categories, each designed to assess a specific dimension of RAC. Below, we provide a detailed description of each category. 

1. **Fluent Tracking** - Given the initial state and the sequence of actions performed, this category contains questions about the fluents, i.e. properties of the domain, of an object from the changed state. For instance, in the _Grippers_ domain, a fluent-tracking question might be _“List all valid properties associated with ball2.”_ 

2. **State Tracking** - This category builds upon and extends the concept of _Fluent Tracking_ . It involves querying about the complete set of fluents in the final state. For instance, in the _Blocksworld_ domain, a state-tracking question might be _“What are all the valid properties in this state?”_ 

3. **Action Executability** - This category encompasses two types of questions related to determining the executability of actions. The types of questions within this category are as follows: 

   - (a) Given an initial state, and a sequence of actions, the question focuses on identifying the first action in the sequence that is not executable. 

   - (b) Given an initial state and a sequence of actions leading to a final state, the task is to identify all actions that can be executed in the final state. For instance, in the _Visitall_ domain, an action-executability question might be _“List all executable actions present in the current state.”_ 

4. **Effects of Actions** - This category contains questions that explore the outcomes of performing a specific action. For instance, in the _Mystery_ domain, an Effects-of-action question can be _“From the current state, the vehicle v0 moves from location l1 to l0, and has fuel-level f6 and f5, which properties of the state will be true now?”_ 

5. **Numerical RAC** - Questions requiring a numerical response fall under this category. These questions may derive from any of the four previously mentioned categories. For example, in the _Spanner_ domain, a Numerical-RAC question can be _“What are the number of executable actions in the current state?”_ 

6. **Composite Question** - This category contains questions that integrate multiple abovementioned categories, combining up to three distinct categories. These questions require multiple steps of reasoning to arrive at the correct solution. For example, in the _Satellite_ domain, a composite question may combine aspects of _Fluent Tracking_ and _Action Executability_ . An example of such a question could be _“List all the properties of the state for satellite0 before the first infeasible action in the sequence?”_ 

3 

Under Review 

## 3.2 FLUENT CATEGORIES 

We further divide the fluents of all 8 domains into three distinct categories, each representing a different aspect of ramifications within RAC. 

1. **Static Properties** - These properties remain unchanged regardless of any action performed. For instance, the property _“connected”_ in the _Visitall_ domain represents whether two locations are connected, a relationship that remains constant irrespective of the robot’s action, which may involve moving, picking up, or placing down objects. In this domain, the connectivity between locations remains unchanged, irrespective of any action. 

2. **Base Fluents** - These fluents can change as a direct result of an action and do not depend on other fluents. For example, in the _Grippers_ domain, the fluent _“carry”_ indicates whether an object is being carried by a robot’s gripper. This fluent can change if the action pick or drop is performed. 

3. **Ramification Fluents** - These fluents are influenced indirectly by other fluents, and action descriptions do not explicitly dictate their changes. Instead, they are determined by the dependencies and relationships within the system. Ramification fluents are further divided into two sub-categories: 

   - (a) **Derived Fluents** - These fluents rely on the state of other fluents, reflecting a level of dependency. Changes to them occur as a consequence of changes in the fluents they depend on rather than through direct action. For instance, the fluent _“stable”_ in the _Blocksworld_ domain is considered a derived fluent as its state depends on the fluents _“clear”_ and _“on_ ~~_t_~~ _able”_ . This relationship can be described as: _“Blocks are stable when clear and on the table”_ . Furthermore, fluent _“clear”_ is itself a derived fluent, dependent on the fluent _“on”_ , which makes _“stable”_ a second-level indirect effect. 

   - (b) **Self-Derived Fluents** - These fluents rely on constraints related to themselves rather than other fluents. For example, in the _Depot_ domain, the fluent _“at”_ represents the location of a truck, which can only be at one location at any given time. If the truck is _at_ location l0, it cannot simultaneously be _at_ location l1. Such constraints are explicitly included in the domain description, for example, _“A truck can only be in one location at a time”_ . 

Classification of every fluent across all 8 domains can be found in Appendix H. Furthermore, for each fluent type, we generate questions involving negative fluents, which allows us to evaluate LLM’s comprehension of negation within RAC contexts. 

## 3.3 DATASET STRUCTURE AND VARIATIONS 

**Selected Domains** The ACTIONREASONINGBENCH requires domains that facilitate the evaluation of LLMs on both short and long sequences of meaningful actions, where the effects and preconditions of these actions are succinctly described. Additionally, these domains should reflect real-world scenarios. To meet these criteria, we selected 8 domains _Blocksworld_ , _Depots_ , _Driverlog_ , _Grippers_ , _Mystery_ , _Satellite_ , _Spanner_ , and _Visitall_ -sourced from the International Planning Competition (IPC), covering the years 1998 to 2014. These domains are commonly used as benchmarks for evaluating advanced planning systems and provide a robust foundation for research in automated planning. Appendix F provides a detailed description of each domain. Notably, even state-of-the-art LLMs like GPT-4o are not capable of generating diverse domains or action sequences that conform to the precise constraints outlined in these domain descriptions, which justifies the reliance on IPC domains. 

**Domain Descriptions and Ramifications** The domains provided by the IPC are described using the Planning Domain Definition Language (PDDL), a formal language designed to model deterministic actions and state transitions for planning problems. Further information on PDDL can be found in Appendix G.1. In this study, we initially translated the PDDL-based domains into natural language. Subsequently, ramification constraints were introduced into these natural language descriptions. The process underwent validation by two experts in the RAC domain to ensure correctness. Given that the category _Action Executability_ focuses on determining whether the action can be 

4 

Under Review 

performed rather than analyzing its effect, we concentrated on the categories _Fluent Tracking_ , _State Tracking_ , and _Effects of Actions_ when introducing ramifications. 

**Action-Sequence Lengths** In order to fine-tune LLMs, we generate a comprehensive set of questions that span various action-sequence lengths, specifically 1, 5, 10, 15, and 19. This range is chosen to capture increasing complexities in RAC. We curate a distinct subset of questions with action-sequence lengths of 1, 10, and 19 for evaluation. This subset is selected to assess the model’s performance across the action sequences range. 

**Answer Types** We formulate two distinct types of questions based on the expected answer format. The first type consists of binary questions, where the response is constrained to either “True” or “False”. The second type involves subjective questions, where the answer may encompass a range of multiple objects, actions, or fluents. 

## 3.4 DATA CREATION & VALIDATION 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints_images/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints.pdf-0005-05.png)


**----- Start of picture text -----**<br>
Stage 1 Stage 2 Stage 3 Stage 4<br>Generating Instances Converting to ASP Generating State Space Generating Questions<br>IPC Domain  Templated IPC Domain  + ASP Code Templated Question<br>(PDDL) Script (ASP) Generator<br>Templated Templated<br>Initial Final PDDL Instances & Questions Questions<br>Plans Len 1 Len 19<br>State State<br>Instance Llama-3.1-<br>Solver Templated Script Action-State 70B-Instruct<br>Space for every<br>Plan Action (from 1 to 19)<br>ASP Instances &<br>Validator NL NL<br>Plans<br>Questions Questions<br>Len 1 Len 19<br>**----- End of picture text -----**<br>


Figure 1: Overview of the question generation pipeline for ACTIONREASONINGBENCH. Blue blocks represent “Generated Data”, and green blocks represent “Code used in the pipeline”. Stage 1 involves generating states and plans using Helmert (2006) and validating them with Howey et al. (2004). In Stage 2, PDDL instances and plans are converted to ASP. Stage 3 computes the actionstate space through ASP. Stage 4 generates questions using templates, which are then rephrased to natural language via Llama-3.1-70B-Instruct. 

The question generation process follows a four-stage pipeline, as illustrated in Figure 1. The selected domains from the IPC are represented in PDDL (see Appendix F for examples). First, these PDDL representations are used to generate 10 pairs of initial and goal conditions. A PDDL solver (Helmert, 2006) and validator (Howey et al., 2004) are then employed to obtain and validate the action sequences necessary to transition from the initial to the goal state. In the second stage, the PDDL domains, instances, and action sequences are converted into Answer Set Programming (ASP) descriptions using Python-based templates. 

In the third stage, ASP solvers are used to generate the action-state space and extract fluents for each state, along with identifying all executable and inexecutable actions. Further details on these formal languages are provided in Appendix G. Finally, the fourth stage involves converting the action-state data into questions using a Python template. Up to three natural language variations are created for every object, action, and fluent. These templated sequences are then paraphrased to Llama-3.1-70B-Instruct to ensure they sound natural and avoid repetition in long action sequences. Three independent annotators review both the templated and the paraphrased versions to assess their naturalness, as detailed in Appendix C. Additionally, all eight domain descriptions were manually translated from PDDL to natural language. 

5 

Under Review 

## 3.5 DATA SPLITS 

The benchmark was divided into two parts: one for training and the other for testing the LLMs, ensuring a balanced representation of question categories across the 8 domains. The _Composite Questions_ category is slightly larger in the test set, as it combines multiple categories, leading to increased questions. Table 2 provides an overview of the distribution of questions and their corresponding categories across both the training and testing sets. The test set contains 3,498 questions, including 2,195 binary and 1,303 free-answer questions. Finally, we designed both zero-shot-CoT and few-shot3 prompts for all the questions in the test set. 

## 4 EXPERIMENTS AND EVALUATION 

**Models** To evaluate our benchmark, we tested four different LLMs and employed two 

||Test Set|Train Set|
|---|---|---|
||||
|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical Reasoning<br>Composite Questions|438<br>382<br>450<br>417<br>414<br>1,397|57,906<br>12,636<br>9,562<br>8,939<br>31,506<br>28,688|
||||
|Static Properties<br>Base Fluents<br>Derived Fluents<br>Self-Derived Fluents<br>Mixed Fluents|237<br>231<br>366<br>390<br>2,274|12,458<br>10,461<br>15,946<br>23,436<br>86,936|
||||
|Total Unique Questions|3,498|149,237|



Table 2: Overview of the test and train sets of ACTIONREASONINGBENCH across Question and Fluent Categories. The “Mixed Fluents” category represents questions that involve more than one type of fluent. 

prompting techniques. The LLMs include two proprietary state-of-the-art models–GPT-4o (Achiam et al., 2023) and o1-preview (OpenAI, 2024)–alongside two open-source models, Llama-3.1-8BInstruct and Llama-3.1-70B-Instruct (Dubey et al., 2024). Each LLM was evaluated using both few-shot prompting with three examples (few-shot-3) Brown (2020) and zero-shot-CoT Kojima et al. (2022) prompting. While the entire dataset requires reasoning abilities, the ramification subset involves the most complex and challenging reasoning tasks. Given that o1-preview is specifically optimized for reasoning tasks and incurs significantly higher costs compared to GPT-4o[1] , we restricted its use to the ramification subset, where its superior reasoning capabilities are expected to provide the greatest benefit. Utilizing o1-preview across the entire dataset would not be cost-effective, as its advantages would be less pronounced for simpler reasoning tasks. 

**Evaluation & Metrics** ACTIONREASONINGBENCH includes two types of answer formats, as outlined in Section 3.3: binary (true/false) and free-form responses. The evaluation process was performed separately for each answer type. For binary questions, we extracted “true” and “false” keywords from the model’s response and compared them to the ground truth. Since free-form answers can’t be evaluated using exact string matching, we employed human evaluation for the ramification questions. While human evaluation is highly accurate, it is not scalable, so we used Llama-3.1-70BInstruct to assess all free-form responses. The specific prompt used to evaluate the LLMs and the correlation between Llama-3.1-70B-Instruct and human evaluations are detailed in Appendix E. For all experiments, we report the accuracy along with the standard error of the mean (SEM), calculated as _SEM_ = _σ_ ~~_√_~~ _n_[, where] _[ σ]_[ represents the standard deviation, and] _[ n]_[ is the sample size.] 

**Fine-tuning** We fine-tuned the Llama-3.1-8B model using the training data split outlined in Section 3.5. Due to the limited computing power, we excluded any data samples that exceeded a context length of 4096 tokens. The fine-tuning process was performed separately for free-response and binary questions. Detailed information on the fine-tuning procedure is provided in Appendix D. All experiments were executed using 8×H100 GPUs. 

## 5 RESULTS AND DISCUSSION 

This section presents the results and analysis using ACTIONREASONINGBENCH as defined in Section 4. The Zero-shot-CoT results for each LLM on both the binary and free-response subsets of the test set are provided in Tables 3 and 4 respectively. Similarly, the Few-shot-3 results for each LLM 

> 1As of Oct 2024, o1-preview is six times more expensive than GPT-4o https://openai.com/api/ pricing/ 

6 

Under Review 

evaluated on these same subsets are displayed in Tables 6 and 7. The detailed analysis of the effects when ramification constraints are incorporated into the descriptions is discussed in Section 5.1. 

|Action Seq.|Ques Categories|GPT-4o<br>Llama-8B-Inst<br>Llama-70B-Inst|Fine-tuned Llama-8B|
|---|---|---|---|
|1|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical RAC<br>Composite Questions|94_._442_._70<br>80_._004_._78<br>90_._123_._31<br>85_._944_._35<br>80_._008_._00<br>79_._416_._93<br>93_._142_._50<br>79_._124_._26<br>94_._122_._33<br>78_._494_._26<br>70_._895_._11<br>77_._174_._38<br>62_._825_._47<br>42_._118_._01<br>57_._386_._33<br>74_._564_._08<br>50_._434_._62<br>67_._393_._99|89_._063_._90<br>98_._551_._44<br>94_._743_._62<br>96_._552_._40<br>49_._375_._62<br>94_._421_._57|
||Average|81_._261_._71<br>66_._432_._30<br>77_._761_._85|87_._761_._43|
|10|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical RAC<br>Composite Questions|91_._433_._35<br>68_._426_._16<br>84_._934_._19<br>75_._865_._62<br>76_._199_._29<br>86_._216_._40<br>70_._874_._48<br>65_._675_._80<br>74_._494_._40<br>61_._224_._92<br>63_._865_._27<br>60_._824_._96<br>55_._215_._08<br>65_._528_._83<br>54_._396_._60<br>82_._963_._24<br>71_._004_._54<br>68_._923_._80|89_._863_._63<br>96_._432_._48<br>81_._828_._22<br>96_._672_._32<br>51_._145_._33<br>88_._792_._07|
||Average|72_._501_._89<br>67_._792_._47<br>70_._122_._04|84_._061_._59|
|19|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical RAC<br>Composite Questions|91_._533_._63<br>67_._356_._70<br>77_._275_._16<br>85_._944_._35<br>83_._338_._78<br>81_._825_._81<br>66_._674_._60<br>53_._667_._79<br>66_._994_._63<br>70_._004_._83<br>54_._696_._22<br>69_._324_._92<br>50_._005_._21<br>42_._428_._60<br>47_._546_._39<br>78_._423_._49<br>53_._095_._54<br>68_._973_._84|85_._714_._68<br>96_._362_._52<br>77_._276_._32<br>97_._102_._02<br>66_._295_._01<br>86_._182_._20|
||Average|72_._311_._91<br>56_._642_._93<br>68_._242_._07|84_._621_._53|



Table 3: Performance comparison of GPT-4o, Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, and fine-tuned Llama-3.1-8B on the binary subset (True/False) of the benchmark, evaluated without the ramifications constraints using the zero-shot-CoT. The results are categorized up by the actionsequence lengths and question categories. 

|Action Seq.|Ques Categories|GPT-4o<br>Llama-8B-Inst<br>Llama-70B-Inst|Finetuned Llama-8B|
|---|---|---|---|
|1|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical RAC<br>Composite Questions|88_._464_._43<br>30_._776_._40<br>71_._156_._28<br>73_._336_._59<br>28_._896_._76<br>64_._447_._14<br>27_._086_._41<br>08_._333_._99<br>33_._336_._80<br>82_._506_._01<br>20_._006_._32<br>67_._507_._41<br>11_._114_._68<br>06_._673_._72<br>04_._443_._07<br>64_._533_._36<br>24_._633_._02<br>43_._353_._48|76_._925_._84<br>75_._566_._41<br>31_._256_._69<br>60_._537_._93<br>08_._894_._24<br>72_._413_._14|
||Average|60_._282_._35<br>21_._711_._98<br>45_._962_._39|61_._022_._35|
|10|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical RAC<br>Composite Questions|82_._005_._43<br>36_._006_._79<br>62_._006_._86<br>74_._426_._65<br>18_._605_._93<br>60_._477_._46<br>34_._097_._15<br>11_._364_._78<br>40_._917_._41<br>76_._096_._29<br>19_._575_._85<br>65_._227_._02<br>10_._204_._32<br>02_._042_._02<br>06_._123_._42<br>59_._113_._45<br>16_._262_._59<br>45_._323_._49|80_._005_._66<br>66_._677_._27<br>47_._737_._53<br>62_._227_._23<br>10_._204_._32<br>58_._623_._46|
||Average|57_._012_._37<br>17_._011_._80<br>45_._982_._39|55_._662_._39|
|19|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical RAC<br>Composite Questions|67_._447_._15<br>27_._916_._84<br>67_._447_._15<br>75_._516_._14<br>16_._335_._28<br>51_._027_._14<br>41_._677_._12<br>08_._333_._99<br>29_._796_._67<br>76_._606_._18<br>14_._895_._19<br>46_._817_._28<br>10_._204_._32<br>06_._123_._42<br>08_._163_._91<br>60_._303_._47<br>08_._541_._98<br>38_._193_._44|67_._447_._15<br>65_._316_._80<br>37_._506_._99<br>61_._707_._09<br>06_._123_._42<br>49_._253_._54|
||Average|56_._782_._38<br>11_._721_._54<br>39_._172_._34|48_._052_._40|



Table 4: Performance comparison of GPT-4o, Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, and fine-tuned Llama-3.1-8B on the free-answer subset of the benchmark, evaluated without the ramifications constraints using the zero-shot-CoT. The results are categorized up by the action-sequence lengths and question categories. 

7 

Under Review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints_images/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints.pdf-0008-01.png)


**----- Start of picture text -----**<br>
Fluent Trk. State Trk. Action Exec. Effects of Ac. Num. Reas. Composite<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>1 10 19 1 10 19 1 10 19<br>Action-Sequence Length Action-Sequence Length Action-Sequence Length<br>(a) GPT-4o (b) Llama-3.1-70B-Inst (c) Llama-3.1-8B-Instruct<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 2: Question Categories for both T/F and Free answers as a function of Action-Sequence length for 3 models (zero-shot-CoT). Note: bars represent SEM. 

**Performance across Domains** In our evaluation, GPT-4o demonstrated the highest performance on the _Grippers_ domain and the lowest on the _Satellite_ domain, with a performance gap of 15.53%. For both Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct, the best performance is also on the _Grippers_ domain, but their worst performance occurs on the _Mystery_ domain, with differences of 19.24% and 24.04%, respectively. Interestingly, the second lowest performing domain for GPT-4o is _Mystery_ , while for the Llama models, it is _Satellite_ . This suggests that despite potential differences in pre-training data, these models exhibit a similar relative understanding of the domains. 

**Decrease in Performance on Complex Questions** The categories of _Composite Questions_ and _Numerical RAC_ build upon the four foundational RAC dimensions. The performance of GPT-4o, Llama-3.1-8B-Instruct, and Llama-3.1-70B-Instruct on free-response complex questions highlights significant challenges, especially in the _Numerical RAC_ category. This category reformulates existing question types into numerical formats, a domain where all tested LLMs exhibit notable difficulty. Specifically, performance on numerical questions related to _Action Executability_ is 8.65% lower than on questions in the _Fluent Tracking_ category. Previous research, such as (Ahn et al., 2024), indicates that LLMs struggle with arithmetic reasoning, which, when mixed with the RAC questions, likely contributes to the poor performance in the _Numerical RAC_ category. 

For _Composite Questions_ , the combination of _Fluent Tracking_ and _Action Executability_ proves easier to answer than the combination of _State Tracking_ and _Action Executability_ , with a 16.32% performance difference. This can be attributed to the fact that the _State Tracking_ category is a superset of the _Fluent Tracking_ category, thereby explaining the observed difference in difficulty. 

**Performance across Fluent Categories** Across all LLMs examined in the study, a consistent trend emerges in which performance on _Static Properties_ decreases as the length of action-sequence increases. While these static properties remain unchanged throughout the actions, they might get overlooked in longer sequences, likely due to their absence in the effect of any action. This phenomenon resembles the “needle in a haystack” challenge in long-context scenarios, where LLMs struggle to recall specific information embedded within a long context (Zhang et al., 2024). Conversely, _Base Fluents_ maintain stable performance across all action sequences, indicating that the LLMs consistently capture the direct effects of actions. _Ramification fluents_ exhibit a steady decline in performance as the sequence lengthens, particularly affecting the subcategory of _Derived Fluents_ , which suggests that LLMs have more difficulty handling indirect effects. Finally, _Mixed Fluents_ , which involve more than one fluent type, show a consistent decline in performance as the length of action sequences increases. 

**Performance across Action-Sequence Lengths** Figure 2 illustrates the performance of the three models–GPT-4o, Llama-3.1-70B-Instruct, and Llama-3.1-8B-Instruct–across varying action sequence lengths in a zero-shot-CoT setting. The results combine both binary and free-response formats, with detailed data available in Tables 3 and 4. Generally, model accuracy declines as the action-sequence length increases, a pattern that holds for most categories. However, GPT-4o deviates from this trend in the _State Tracking_ category, where performance first decreases and then 

8 

Under Review 

improves. Since this trend is not observed with the other two models, and the results lie within the margin of error, we believe this is an outlier. In contrast, the _Effects of Actions_ category consistently deviates from this trend, likely due to the nature of the task, which focuses on changes resulting from the last action, making it less dependent on the sequence of actions. 

**Model Parameters and Fine-tuning** As demonstrated in Tables 3 and 4, the Llama-3.1-70BInstruct model consistently outperforms the smaller Llama-3.1-8B-Instruct model, with an average performance improvement of 20.84%. This improvement is likely due to the larger model’s superior reasoning capabilities stemming from its increased number of parameters. A similar trend is observed when comparing GPT-4o to Llama-3.1-70B-Instruct, where GPT-4o exhibits an average performance increase of 8.64%. Although the specific size of GPT-4o remains undisclosed, it is widely speculated to be in the trillions of parameters[2] . Notably, fine-tuning the smaller Llama-3.1-8B model on our benchmark’s training set resulted in substantial gains in both binary and free-answer tasks, with an average performance increase of 33.68% across the test set, even outperforming GPT-4o by 4.2%. 

**Impact of Few-Shot Examples on Model Performance** As shown in Tables 3 and 6, the inclusion of few-shot examples for binary answer categories does not significantly enhance model accuracy. This limitation is especially pronounced in models such as GPT-4o and Llama-3.1-70B-Instruct, which exhibit a relative performance decline of approximately 3.5% compared to zero-shot-CoT conditions. We hypothesize that this decrease may be attributed to the few-shot examples inadvertently leading the model toward detecting spurious correlations. In contrast, the model relies more heavily on its internal reasoning capabilities in a zero-shot-CoT setting, potentially mitigating biases introduced by example-driven patterns. However, as seen from Tables 4 and 7, the few-shot approach shows effectiveness in free-answer questions only open-source LLMs. 

**LLMs Struggle with Negative Fluents** Our study reveals a consistent pattern across all the LLMs examined, wherein their performance declines when tasked with questions involving negative fluents compared to those focused on fluents that are true. Specifically, we observed a 12.16% decrease in accuracy on these negative fluent tasks. Furthermore, when questions required reasoning about both true and false fluents simultaneously, LLMs exhibited competence in identifying the true fluents but demonstrated difficulty in correctly recalling the false ones. 

## 5.1 RAMIFICATIONS RESULTS 

As discussed in Section 4, the performance on ramification fluents is evaluated for two LLMs: GPT-4o, the highest-performing LLM, and o1-preview, the most recent state-of-theart LLM. Table 5 presents the performance of both models when ramification constraints are introduced. Further examples of the model responses to ramification-related questions can be found in the Appendix I 

**GPT-4o Performance** GPT-4o did not answer any ramification-related questions correctly, as depicted in Table 5. Upon manual inspection of its outputs, it became evident that GPT-4o frequently failed to mention ramification fluents, even when these were explicitly detailed in the domain description. In instances where it did address ramification fluents, the responses were incorrect or incomplete, with some fluents being omitted. We hypothesize that GPT-4o may have encountered the domain 

|Ac Seq.|Ques Cat|GPT-4o<br>o1-preview|
|---|---|---|
|1|Fluent Trk<br>State Trk<br>Effects of Ac|00_._0000_._00<br>00_._0000_._00<br>00_._0000_._00<br>20_._0017_._89<br>00_._0000_._00<br>40_._0021_._91|
||Average|00_._0000_._00<br>25_._0012_._49|
|10|Fluent Trk<br>State Trk<br>Effects of Ac|00_._0000_._00<br>00_._0000_._00<br>00_._0000_._00<br>33_._3319_._24<br>00_._0000_._00<br>14_._2813_._22|
||Average|00_._0000_._00<br>23_._0711_._68|
|19|Fluent Trk<br>State Trk<br>Effects of Ac|00_._0000_._00<br>33_._3327_._21<br>00_._0000_._00<br>00_._0000_._00<br>00_._0000_._00<br>00_._0000_._00|
||Average|00_._000_._00<br>07_._6907_._38|



Table 5: Performance comparison of GPT-4o and o1-preview on the free-answer subset of the benchmark, evaluated with the ramifications constraints using the zero-shot-CoT. The results are categorized up by the action-sequence lengths and question categories. 

> 2GPT-4o and Gemini 1.5 Pro: How the New AI Models Compare - CNET 

9 

Under Review 

data during pre-training and relied on memorized effects of actions, as the experimental domains were derived from publicly available IPC datasets. Since the ramification fluents were manually created and integrated for this study, this evaluation assesses the model’s reasoning abilities without leveraging pre-existing knowledge. This likely explains why GPT-4o failed to generate ramification fluents, as its pre-training included only the original fluents from the domains. 

**o1-preview Performance** o1-preview, a recently developed LLM optimized for reasoning tasks and incorporating a novel run-time inference mechanism OpenAI (2024), demonstrated significantly better performance on ramification-related questions compared to GPT-4o, as presented in Table 5. A detailed review of its outputs showed that o1-preview can correctly identify most ramification fluents. However, the model struggles with fluents involving negation, which consistently poses a challenge. It often omitted certain fluents in its final answers and, in some cases, incorrectly evaluated the ramification fluents. 

## 6 CONCLUSION 

In this work, we introduced a new diagnostic benchmark, ACTIONREASONINGBENCH, designed to comprehensively evaluate the performance of large language models (LLMs) on reasoning about actions and change (RAC). By assessing various LLMs across eight domains and six key dimensions of RAC, our findings indicate that while LLMs demonstrate moderate proficiency on traditional RAC tasks, such as _Fluent Tracking_ , _State Tracking_ , _Action Executability_ , and _Effects of Actions_ , they exhibit significant challenges when addressing more complex and novel questions, particularly in areas like _Numerical RAC_ and _Composite Questions_ , with an average performance drop of 17.9%. This highlights a substantial gap in the current ability of LLMs to handle complex reasoning tasks. 

Additionally, we explored the indirect effects of actions, known as ramifications, where even stateof-the-art models show considerable limitations. For example, GPT-4o could not solve any questions involving ramifications, and the o1-preview model achieved a low score of 18.4%. These results underscore the pressing need for further research and advancements in RAC reasoning, especially in addressing indirect effects and more advanced reasoning tasks. Improving LLMs’ capabilities in these areas is crucial for enhancing their performance in dynamic and interactive environments. Our findings reinforce the challenges LLMs face in RAC tasks, which, despite their polynomial complexity in cases without ramifications, serve as prerequisites for more advanced planning problems that are often NP-complete or beyond. Improving LLM performance in RAC is a pivotal step toward overcoming broader planning challenges. 

## 7 LIMITATIONS & FUTURE WORK 

While ACTIONREASONINGBENCH provides a diagnostic assessment of LLMs on RAC, it represents an early step in this area and has several limitations, including but not limited to the following: 

1. Although RAC is not inherently dependent on the English language, the current version of ACTIONREASONINGBENCH is limited to questions formulated in English. 

2. Although there are more complex types of RAC, exploring those remains beyond the scope of this work and is left as a direction for future research. 

3. While the IPC domains in our work cover many scenarios, they may introduce a bias towards planning-specific domains. Expanding the dataset to include more domains could help mitigate this bias. 

4. Despite our efforts to evaluate a variety of LLMs, including open and proprietary LLMs, our assessment did not cover models with different architectures or training approaches due to resource limitations. 

5. Recent studies by Long et al. (2024) and Tam et al. (2024) indicate that LLM performance can fluctuate depending on the prompt format. This variation may lead to a marginal improvement in the performance of LLMs on ACTIONREASONINGBENCH. 

6. Our free-answer evaluation, which relies on prompting Llama-3.1-70B-Instruct, isn’t perfect and reflects an ongoing challenge of evaluating the free-answers within the NLP community. 

10 

Under Review 

## REFERENCES 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. Large language models for mathematical reasoning: Progresses and challenges. _arXiv preprint arXiv:2402.00157_ , 2024. 

- Pratyay Banerjee, Chitta Baral, Man Luo, Arindam Mitra, Kuntal Pal, Tran C Son, and Neeraj Varshney. Can transformers reason about effects of actions? _arXiv preprint arXiv:2012.09938_ , 2020. 

- Tom B Brown. Language models are few-shot learners. _arXiv preprint arXiv:2005.14165_ , 2020. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems, 2021. _URL https://arxiv. org/abs/2110.14168_ , 2021. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- Bahare Fatemi, Mehran Kazemi, Anton Tsitsulin, Karishma Malkan, Jinyeong Yim, John Palowitch, Sungyong Seo, Jonathan Halcrow, and Bryan Perozzi. Test of time: A benchmark for evaluating llms on temporal reasoning. _arXiv preprint arXiv:2406.09170_ , 2024. 

- Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Songyang Zhang, Kai Chen, Zongwen Shen, and Jidong Ge. LawBench: Benchmarking Legal Knowledge of Large Language Models, September 2023. 

- Richard E Fikes and Nils J Nilsson. Strips: A new approach to the application of theorem proving to problem solving. _Artificial intelligence_ , 2(3-4):189–208, 1971. 

- Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies, January 2021. 

- Lin Guan, Karthik Valmeekam, Sarath Sreedharan, and Subbarao Kambhampati. Leveraging pretrained large language models to construct and utilize world models for model-based task planning. _Advances in Neural Information Processing Systems_ , 36:79081–79094, 2023. 

- Neel Guha, Julian Nyarko, Daniel Ho, Christopher R´e, Adam Chilton, Aditya K, Alex ChohlasWood, Austin Peters, Brandon Waldon, Daniel Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Galit Sarfaty, Gregory Dickinson, Haggai Porat, Jason Hegland, Jessica Wu, Joe Nudell, Joel Niklaus, John Nay, Jonathan Choi, Kevin Tobia, Margaret Hagan, Megan Ma, Michael Livermore, Nikon Rasumov-Rahe, Nils Holzenberger, Noam Kolt, Peter Henderson, Sean Rehaag, Sharad Goel, Shang Gao, Spencer Williams, Sunny Gandhi, Tom Zur, Varun Iyer, and Zehua Li. LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. _Advances in Neural Information Processing Systems_ , 36:44123–44279, December 2023. 

- Simeng Han, Hailey Schoelkopf, Yilun Zhao, Zhenting Qi, Martin Riddell, Wenfei Zhou, James Coady, David Peng, Yujie Qiao, Luke Benson, Lucy Sun, Alex Wardle-Solano, Hannah Szabo, Ekaterina Zubova, Matthew Burtell, Jonathan Fan, Yixin Liu, Brian Wong, Malcolm Sailor, Ansong Ni, Linyong Nan, Jungo Kasai, Tao Yu, Rui Zhang, Alexander R. Fabbri, Wojciech Kryscinski, Semih Yavuz, Ye Liu, Xi Victoria Lin, Shafiq Joty, Yingbo Zhou, Caiming Xiong, Rex Ying, Arman Cohan, and Dragomir Radev. FOLIO: Natural Language Reasoning with First-Order Logic, May 2024. 

- Patrik Haslum, Nir Lipovetzky, Daniele Magazzeni, Christian Muise, Ronald Brachman, Francesca Rossi, and Peter Stone. _An introduction to the planning domain definition language_ , volume 13. Springer, 2019. 

11 

Under Review 

- Weinan He, Canming Huang, Zhanhao Xiao, and Yongmei Liu. Exploring the capacity of pretrained language models for reasoning about actions and change. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 4629– 4643, 2023. 

- Malte Helmert. The fast downward planning system. _Journal of Artificial Intelligence Research_ , 26: 191–246, 2006. 

- Richard Howey, Derek Long, and Maria Fox. Val: Automatic plan validation, continuous effects and mixed initiative planning using pddl. In _16th IEEE International Conference on Tools with Artificial Intelligence_ , pp. 294–301. IEEE, 2004. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35:22199–22213, 2022. 

- Bill Yuchen Lin, Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Xiang Ren, and William W. Cohen. Differentiable Open-Ended Commonsense Reasoning, June 2021. 

- Do Xuan Long, Hai Nguyen Ngoc, Tiviatis Sim, Hieu Dao, Shafiq Joty, Kenji Kawaguchi, Nancy F Chen, and Min-Yen Kan. Llms are biased towards output formats! systematically evaluating and mitigating output format bias of llms. _arXiv preprint arXiv:2408.08656_ , 2024. 

- Nicholas Lourie, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. UNICORN on RAINBOW: A Universal Commonsense Reasoning Model on a New Multitask Benchmark, March 2021. 

- Man Luo, Shrinidhi Kumbhar, Ming shen, Mihir Parmar, Neeraj Varshney, Pratyay Banerjee, Somak Aditya, and Chitta Baral. Towards LogiGLUE: A Brief Survey and A Benchmark for Analyzing Logical Reasoning Capabilities of Language Models, March 2024. 

John McCarthy et al. _Situations, actions, and causal laws_ . Comtex Scientific, 1963. 

- Sheila A McIlraith. Integrating actions and state constraints: A closed-form solution to the ramification problem (sometimes). _Artificial Intelligence_ , 116(1-2):87–121, 2000. 

- Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers, June 2021. 

- Yasumasa Onoe, Michael JQ Zhang, Eunsol Choi, and Greg Durrett. Creak: A dataset for commonsense reasoning over entity knowledge. _arXiv preprint arXiv:2109.01653_ , 2021. 

OpenAI. Openai o1 system card, 2024. URL https://cdn.openai.com/ o1-system-card-20240917.pdf. Accessed: 2024-09-30. 

- Mihir Parmar, Nisarg Patel, Neeraj Varshney, Mutsumi Nakamura, Man Luo, Santosh Mashetty, Arindam Mitra, and Chitta Baral. Towards Systematic Evaluation of Logical Reasoning Ability of Large Language Models, April 2024. 

- Nisarg Patel, Mohith Kulkarni, Mihir Parmar, Aashna Budhiraja, Mutsumi Nakamura, Neeraj Varshney, and Chitta Baral. Multi-logieval: Towards evaluating multi-step logical reasoning ability of large language models. _arXiv preprint arXiv:2406.17169_ , 2024. 

- Raymond Reiter. _Knowledge in action: logical foundations for specifying and implementing dynamical systems_ . 2001. 

- Evangelia Spiliopoulou, Artidoro Pagnoni, Yonatan Bisk, and Eduard Hovy. Events realm: Event reasoning of entity states via language models. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 1982–1997, 2022. 

- Zhi Rui Tam, Cheng-Kuang Wu, Yi-Lin Tsai, Chieh-Yen Lin, Hung-yi Lee, and Yun-Nung Chen. Let me speak freely? a study on the impact of format restrictions on performance of large language models. _arXiv preprint arXiv:2408.02442_ , 2024. 

12 

Under Review 

- Md Nayem Uddin, Amir Saeidi, Divij Handa, Agastya Seth, Tran Cao Son, Eduardo Blanco, Steven R Corman, and Chitta Baral. Unseentimeqa: Time-sensitive question-answering beyond llms’ memorization. _arXiv preprint arXiv:2407.03525_ , 2024. 

- Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. Planbench: An extensible benchmark for evaluating large language models on planning and reasoning about change. _Advances in Neural Information Processing Systems_ , 36, 2024a. 

- Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the planning abilities of large language models-a critical investigation. _Advances in Neural Information Processing Systems_ , 36, 2024b. 

- Zhenyu Zhang, Runjin Chen, Shiwei Liu, Zhewei Yao, Olatunji Ruwase, Beidi Chen, Xiaoxia Wu, and Zhangyang Wang. Found in the middle: How language models use long contexts better via plug-and-play positional encoding. _arXiv preprint arXiv:2403.04797_ , 2024. 

13 

Under Review 

## A DESCRIBING AN INSTANCE FROM ACTIONREASONINGBENCH 

## Blocksworld domain with Ramifications for a single sequence of action for Fluent Tracking 

## [DOMAIN DESCRIPTION] 

A block can only be picked up if it is clear, on the table, and the hand is empty, resulting in the block being held. A held block can be put down, placing it back on the table. Blocks can be stacked if the first block is held and the second block is clear, causing the first block to rest on top of the second. Unstacking occurs when the hand is empty, the first block is clear, and on top of the second, resulting in the first block being held again. A block can’t be at two locations at the same time and is considered clear if nothing is on top of it and it’s not held, and the hand is empty if it’s not holding anything. Blocks are stable when clear and on the table, and they can be painted if stable and the hand is empty. A block is considered on display if it can be painted and has no other block on top of it. 

## [INITIAL CONDITIONS] 

Block b1 is situated on the table, block b2 is not stacked with any other block, block b2 is also on the table, block b3 is not stacked with any other block, block b3 is positioned on top of block b7, block b4 is stacked on top of block b1, block b5 is not stacked with any other block, block b5 is placed on top of block b4, block b6 is on the table, block b7 is stacked on top of block b6, and the hand is empty. 

## [QUESTION] 

Starting from the initial condition, the following actions are taken: block b3 is unstacked from the top of block b7 to achieve the current state. In this state, what are the valid properties (including both affirmative and negated properties) for b7? If there are no valid properties, write None. 

In the domain description, actions and their corresponding effects on the state are outlined, including the necessary conditions for performing these actions. The initial conditions describe the starting configuration of objects within the domain. A typical scenario involves executing a sequence of actions that alter the configuration of the state, followed by a query. In the example provided, the question falls under the category of fluent tracking, which asks about a specific set of properties associated with the object ”block b7” after one action has been performed. 

The objects involved in this example are as follows: 

- block b1 

- block b2 

- block b3 

- block b4 

- block b5 

- block b6 

- block b7 

- hand 

The properties of the “block” object include: 

- Block X on top of block Y 

- Block X is on the table 

- Block X is clear 

- Block X is stable 

- Block X can be painted 

- Block X can be displayed 

- Block X is held 

14 

Under Review 

The properties of the “hand” object are: 

- The hand is empty 

- The hand is holding block x 

Several ramification constraints (i.e., properties that depend on other properties) are present in “Blocksworld”: 

- **Clear** : depends on the properties ”on top of” and ”held” 

- **Stable** : depends on the properties ”clear” and ”on the table” 

- **Paint** : depends on the properties ”stable” and ”hand is empty” 

- **Display** : depends on the properties ”painted” and ”on top” 

- **On top of** : depends on itself, since a block can’t be at two locations at the same time 

- **Hand is holding a block** : depends on itself, since the hand cannot hold two blocks at the same time 

The valid actions that can be performed within this domain include: 

- Picking up a block 

- Putting down a block 

- Stacking a block on top of another block 

- Unstacking a block from top of another block 

## B FEW SHOT 3 RESULTS 

Table 6 and 7 presents the results using the Few-shot-3 setting. These tables support the results showed in Section 5. 

## C DATA VERIFICATION 

To ensure the soundness of our benchmark, we employed three independent annotators who had no prior involvement with the project. Their task was to evaluate the naturalness of the sentences by assigning a score from 1 to 5, where 1 indicates the least natural and 5 most natural. To make sure that rephrasing the templated questions using Llama-3.1-70B-Instruct helps, we sample from each domain in the dataset was represented by 5 randomly sampled instances, resulting in a total of 65 samples across all domains for both templated questions and rephrased questions, resulting in a total of 130 samples. The annotators were provided with both the sampled instances and the following instructions: 

Instruction to the Annotators 

Rate the Prompts from 1 to 5, based on how natural they appear in English. 

Table 8 summarizes the naturalness scores assigned by annotators across all domains in ACTIONREASONINGBENCH. The templated dataset received an average naturalness score of 4.2 out of 5, while the paraphrased version scored 4.5, indicating the high effectiveness of Llama-3.1-70BInstruct in enhancing the fluency of the data. We would like to point out that the annotators participated on a voluntary basis and were informed beforehand that no financial compensation would be provided for their contribution. 

## D FINE-TUNING DETAILS 

In this section, we describe the fine-tuning performed on the training split of ACTIONREASONINGBENCH described in section 3.5. We fine-tuned Llama-3.1-8B separately for binary (true/false) and 

15 

Under Review 

|Action Seq.|Ques Categories|GPT-4o<br>Llama-8B-Inst<br>Llama-70B-Inst|
|---|---|---|
|1<br>Fluent Tracking<br>85_._003_._57<br>68_._424_._77<br>82_._653_._82<br>State Tracking<br>91_._573_._05<br>71_._954_._96<br>89_._293_._37<br>Action Executability<br>95_._102_._14<br>84_._313_._60<br>93_._142_._50<br>Effects of Actions<br>74_._194_._54<br>73_._914_._58<br>69_._894_._76<br>Numerical RAC<br>58_._545_._44<br>56_._765_._76<br>56_._105_._48<br>Composite Questions<br>82_._662_._40<br>58_._373_._23<br>74_._302_._77|||
|Average<br>81_._921_._45<br>67_._261_._80<br>77_._261_._58|||
|10<br>Fluent Tracking<br>84_._543_._67<br>69_._664_._87<br>82_._473_._86<br>State Tracking<br>91_._553_._30<br>64_._185_._86<br>86_._114_._08<br>Action Executability<br>73_._794_._33<br>63_._004_._83<br>74_._764_._28<br>Effects of Actions<br>59_._184_._96<br>53_._685_._12<br>65_._314_._81<br>Numerical RAC<br>41_._675_._03<br>42_._255_._86<br>46_._885_._09<br>Composite Questions<br>82_._963_._24<br>71_._004_._54<br>68_._923_._80|||
|Average<br>68_._961_._71<br>59_._511_._92<br>67_._351_._74|||
||||
|19|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical RAC<br>Composite Questions|73_._264_._77<br>63_._385_._72<br>71_._084_._98<br>91_._013_._03<br>63_._645_._13<br>84_._273_._86<br>62_._864_._72<br>54_._555_._00<br>62_._864_._72<br>69_._574_._80<br>56_._635_._44<br>66_._674_._89<br>56_._995_._13<br>49_._156_._51<br>48_._395_._18<br>65_._572_._88<br>57_._313_._11<br>62_._042_._93|
||||
||Average|68_._561_._71<br>57_._581_._93<br>64_._721_._76|



Table 6: Performance comparison of GPT-4o, Llama-3.1-8B-Instruct, and Llama-3.1-70B-Instruct, on the binary subset (True/False) of the benchmark, evaluated without the ramifications constraints using the Few-shot-3 setting. The results are categorized up by the action-sequence lengths and question categories. 

free answer questions, using 6 epochs for the former and 18 epochs for the latter. The AdamW optimizer was used, with a batch size of 4 and gradient accumulation steps set to 8 for both of the training setups. Due to the available compute resources, we were limited to a maximum context length of 4096 tokens. This leaves us roughly with 27k samples for binary answers and 14.4k samples for free answer. Tables 9, 10 and 11 show the statistics of the training set that we used to train the models. 

16 

Under Review 

|Action Seq.|Ques Categories|GPT-4o<br>Llama-8B-Inst<br>Llama-70B-Inst|
|---|---|---|
||||
|1|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical RAC<br>Composite Questions|76_._925_._84<br>50_._006_._93<br>75_._006_._00<br>73_._336_._59<br>44_._447_._41<br>73_._687_._14<br>56_._257_._16<br>14_._585_._09<br>38_._307_._09<br>80_._006_._32<br>35_._007_._54<br>63_._648_._37<br>08_._894_._24<br>04_._443_._07<br>13_._335_._07<br>49_._263_._51<br>41_._383_._46<br>55_._613_._63|
|Average<br>54_._502_._39<br>35_._332_._30<br>53_._732_._49|||
||||
|10|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical RAC<br>Composite Questions|74_._006_._20<br>24_._006_._04<br>64_._006_._79<br>81_._405_._93<br>34_._887_._27<br>65_._718_._02<br>50_._007_._54<br>09_._094_._33<br>34_._097_._15<br>65_._227_._02<br>41_._307_._26<br>62_._868_._17<br>14_._295_._00<br>10_._204_._32<br>12_._244_._68<br>43_._843_._48<br>37_._933_._41<br>48_._133_._65|
||||
||Average|50_._572_._40<br>30_._342_._20<br>47_._002_._50|
||||
|19|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical RAC<br>Composite Questions|74_._426_._65<br>44_._197_._57<br>65_._127_._27<br>61_._226_._96<br>28_._576_._45<br>54_._557_._51<br>47_._927_._21<br>10_._424_._41<br>25_._006_._25<br>73_._916_._47<br>27_._666_._52<br>60_._987_._62<br>06_._123_._42<br>06_._123_._42<br>00_._000_._00<br>38_._193_._44<br>26_._133_._11<br>46_._23_._68|
|Average<br>45_._622_._39<br>24_._372_._06<br>42_._542_._44|||



Table 7: Performance comparison of GPT-4o, Llama-3.1-8B-Instruct, and Llama-3.1-70B-Instruct, on the free answer subset of the benchmark, evaluated without the ramifications constraints using the Few-shot-3 setting. The results are categorized up by the action-sequence lengths and question categories. 

|Domain|Annotator 1<br>Templated<br>Rephrased|Annotator 2<br>Templated<br>Rephrased|Annotator 3<br>Templated<br>Rephrased|Average<br>Templated<br>Rephrased|
|---|---|---|---|---|
|Blocksworld<br>Depots<br>Driverlog<br>Grippers<br>Mystery<br>Satellite<br>Spanner<br>Visitall|3.8<br>4.8<br>4.6<br>4.8<br>4.8<br>4.8<br>4.6<br>5.0<br>4.0<br>4.8<br>4.4<br>4.8<br>5.0<br>5.0<br>3.6<br>4.2|5.0<br>4.6<br>3.6<br>4.8<br>3.6<br>4.2<br>4.0<br>4.8<br>4.0<br>4.4<br>4.6<br>4.4<br>5.0<br>4.6<br>3.8<br>4.6|3.8<br>4.2<br>4.0<br>4.0<br>4.4<br>4.6<br>4.6<br>4.8<br>3.6<br>4.0<br>4.4<br>4.2<br>3.8<br>3.8<br>4.0<br>4.2|4.2<br>4.5<br>4.1<br>4.5<br>4.3<br>4.5<br>4.4<br>4.9<br>3.9<br>4.4<br>4.5<br>4.5<br>4.6<br>4.5<br>3.8<br>4.3|
|Average|4.3<br>4.8|4.2<br>4.6|4.1<br>4.2|4.2<br>4.5|



Table 8: Naturalness scores assigned by three annotators on a scale of 1 to 5, where 1 indicates completely incoherent text and 5 indicates natural-sounding questions. The table presents scores for both the templated questions and paraphrased questions. 

|**Answer category**|**No of Samples**|
|---|---|
|||
|False<br>True<br>Free-Response|13,793<br>13,319<br>14,476|
|||
|**Total**|**41,588**|



Table 9: Data distribution used for fine-tuning, categorized by response type. The binary question responses are split into “True” and “False”. “Free-Response” indicates the count of open-ended questions in the training set. 

17 

Under Review 

|**Question category**|**No of Samples (Binary)**|**No of Samples(Free-Response)**|
|---|---|---|
||||
|Fluent Tracking<br>State Tracking<br>Action Executability<br>Effects of Actions<br>Numerical RAC<br>Composite Questions|11,674<br>2,264<br>1,534<br>1,196<br>5,757<br>4,687|4,946<br>1,133<br>1,094<br>1,040<br>3,293<br>2,970|
||||
|**Total**|**27,112**|**14,476**|



Table 10: Data distribution used for fine-tuning, categorized by Question Categories. 

|**Domain**|**No of Samples (Binary)**|**No of Samples (Free-Response)**|
|---|---|---|
||||
|Blocksworld<br>Depots<br>Driverlog<br>Grippers<br>Mystery<br>Satellite<br>Spanner<br>Visitall|3,540<br>2,799<br>3,404<br>3,494<br>2,825<br>4,037<br>3,866<br>3,147|1,978<br>1,394<br>1,785<br>1,910<br>1,636<br>2,073<br>1,992<br>1,708|
||||
|**Total**|**27,112**|**14,476**|



Table 11: Data distribution used for fine-tuning, categorized by Domains. 

18 

Under Review 

## E FREE ANSWERS EVALUATION DETAILS 

We evaluate the free answers using Llama-3.1-70B-Instruct. The following is the few-shot-7 prompt that we used for evaluating the responses: 

## Prompt for Free Answer Evaluation with Llama-3.1-70b-Instruct 

Evaluate whether the LLM response and the ground truth response are semantically the same. Examine the responses, provide reasoning for your evaluation, and then Write ”True” if the responses are the same or ”False” if they are different. LLM Response or Ground Truth could be ”None”. 

Example 1: 

[LLM Response] Location f1 ~~4~~ f is connected to f1 ~~3~~ f, f0 ~~4~~ f, and f2 ~~4~~ f. 

## [Ground Truth] 

location f1 ~~4~~ f and location f0 ~~4~~ f are connected, location f1 ~~4~~ f and location f2 ~~4~~ f are connected, location f0 ~~4~~ f and location f1 ~~4~~ f are connected, location f1 ~~3~~ f and location f1 ~~4~~ f are connected, there is a connection between location f2 ~~4~~ f and location f1 ~~4~~ f, there is a connection between location f1 ~~4~~ f and location f1 ~~3~~ f 

[Reasoning for the evaluation] all of the connections are the same 

[Answer] True 

... Example 7: 

## [LLM Response] 

ball1 is at room2, ball2 is at room2, ball3 is at room4, ball4 is at room1, ball5 is at room1, ball6 is at room4, ball7 is being carried by robot1’s lgripper1, ball8 is at room4, robot1 is at room2, robot1’s lgripper1 is not free, robot1’s rgripper1 is not free, robot1 is carrying ball7 with lgripper1, robot1 is carrying ball6 with rgripper1 

## [Ground Truth] 

ball1 is present at room2, ball2 is at room2, ball3 is present at room4, ball4 is at room1, ball5 is located at room1, ball7 is being carried by robot1’s lgripper1, ball8 is present at room4, rgripper1 of robot1 is carrying ball6 and robot1 is present in room2 

## [Reasoning for the evaluation] 

According to the ground truth, robot1 is carrying ball6, but LLM gives a contradictory response: that ball6 is at room4 

[Answer] False 

—————————————————————————- 

Given the examples and instructions above, evaluate the following responses: 

[LLM Response] 

_{_ llm ~~r~~ esponse _}_ 

[Ground Truth] _{_ ground ~~t~~ ruth _}_ 

19 

Under Review 

## F DOMAINS IN ACTIONREASONINGBENCH 

|**Domain**<br>**Name**|**# Fluents**<br>**Defned**<br>**# Actions**<br>**Defned**<br>**# Object**<br>**Defned**|**IPC Year**<br>**State**<br>**Complexity**|
|---|---|---|
|**Blocksworld**|5<br>4<br>1|2000<br>_O_(2_N_2+2_N_)|
|**Depots**|6<br>5<br>6|2002<br>_O_(24_N_2+2_N_)|
|**DriverLog**|5<br>6<br>4|2002<br>_O_(25_N_ 2_−N_)|
|**Grippers**|4<br>3<br>4|1998<br>_O_(2_N_ 3+3_N_2)|
|**Mystery**|7<br>3<br>5|1998<br>_O_(27_N_2_−_3_N_)|
|**Satellite**|8<br>5<br>4|2002<br>_O_(25_N_2+3_N_)|
|**Spanner**|6<br>3<br>4|2011<br>_O_(23_N_2+2_N_)|
|**Visitall**|3<br>1<br>1|2014<br>_O_(2_N_2+_N_)|



Table 12: Summarizing key characteristics of various domains in ACTIONREASONINGBENCH including the number of fluents, actions, and objects defined within each domain. Domains are categorized by year of introduction in the IPC and state space complexity, which reflects the difficulty level for AI planners to solve each domain. A larger state space typically indicates greater complexity and presents more significant challenges for traditional AI planners. “ _N_ ” represents the number of objects in each instance. For example, in the _Spanner_ domain, _N_ refers to the number of locations, spanners and nuts. 

In the following sub-sections, we provide details regarding all the domains used in our study. We first present the PDDL description that is given in the IPC and then present how state-space calculation is performed for that domain. State space represents the possible number of interactions that can be performed at a particular state. A higher state-space represents a more difficult problem for traditional AI solvers. 

## F.1 BLOCKSWORLD 

In this domain, we have a set of blocks that can be manipulated using four basic actions: picking up a block from the table, putting down a block onto the table, stacking one block onto another, and unstacking a block from atop another block. The goal is to move and stack these blocks using a robotic hand, following specific rules and conditions. 

This domain is formally represented in the IPC using the PDDL as outlined below: 

PDDL description for Blocksworld domain (define (domain BLOCKS) (:requirements :strips :typing) (:types block) (:predicates (on ?x - block ?y - block) (ontable ?x - block) (clear ?x - block) (handempty) (holding ?x - block)) (:action pick-up :parameters (?x - block) :precondition (and (clear ?x) (ontable ?x) (handempty)) :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x))) 

20 

Under Review 

(:action put-down :parameters (?x - block) :precondition (holding ?x) :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x))) (:action stack :parameters (?x - block ?y - block) :precondition (and (holding ?x) (clear ?y)) :effect (and (not (holding ?x)) (not (clear ?y)) (clear ?x) (handempty) (on ?x ?y))) (:action unstack :parameters (?x - block ?y - block) :precondition (and (on ?x ?y) (clear ?x) (handempty)) :effect (and (holding ?x) (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y)))) ) 

Predicates define the number of fluents in a state: 

- on(b1,b2) 

- ontable(b) 

- clear(b) 

- holding(b) 

- handempty 

The complexity of a state is, where _N_ is the number of objects 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints_images/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints.pdf-0021-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints_images/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints.pdf-0021-10.png)


## F.2 DEPOTS 

The Depots domain models a logistics environment where crates are transported between different locations using trucks and manipulated using hoists. The goal is to efficiently move crates from one location to another, utilizing the available resources (hoists and trucks) while adhering to the constraints defined by the predicates and actions. 

This domain is formally represented in the IPC using the PDDL as outlined below: 

## PDDL description for Depots domain 

- (define (domain depots) (:requirements :strips :typing) (:types place locatable - object depot distributor - place 

21 

Under Review 

truck hoist surface - locatable pallet crate - surface) 

(:predicates (at ?x - locatable ?y - place) (on ?x - crate ?y - surface) (in ?x - crate ?y - truck) (lifting ?x - hoist ?y - crate) (available ?x - hoist) (clear ?x - surface)) 

(:action Drive 

:parameters (?x - truck ?y - place ?z - place) :precondition (and (at ?x ?y)) :effect (and (not (at ?x ?y)) (at ?x ?z))) 

(:action Lift :parameters (?x - hoist ?y - crate ?z - surface ?p - place) :precondition (and (at ?x ?p) (available ?x) (at ?y ?p) (on ?y ?z) (clear ?y)) :effect (and (not (at ?y ?p)) (lifting ?x ?y) (not (clear ?y)) (not (available ?x)) (clear ?z) (not (on ?y ?z)))) 

(:action Drop :parameters (?x - hoist ?y - crate ?z - surface ?p - place) :precondition (and (at ?x ?p) (at ?z ?p) (clear ?z) (lifting ?x ?y)) 

:effect (and (available ?x) (not (lifting ?x ?y)) (at ?y ?p) (not (clear ?z)) (clear ?y)(on ?y ?z))) 

(:action Load :parameters (?x - hoist ?y - crate ?z - truck ?p - place) :precondition (and (at ?x ?p) (at ?z ?p) (lifting ?x ?y)) :effect (and (not (lifting ?x ?y)) (in ?y ?z) (available ?x))) 

(:action Unload 

:parameters (?x - hoist ?y - crate ?z - truck ?p - place) :precondition (and (at ?x ?p) (at ?z ?p) (available ?x) (in ?y ?z)) :effect (and (not (in ?y ?z)) (not (available ?x)) (lifting ?x ?y))) ) 

Predicates define the number of fluents in a state: 

- (at ?x - locatable ?y - place) 

- (on ?x - crate ?y - surface) 

- (in ?x - crate ?y - truck) 

- (lifting ?x - hoist ?y - crate) 

- (available ?x - hoist) 

- (clear ?x - surface)) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints_images/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints.pdf-0022-18.png)


(2) 

## F.3 DRIVERLOG 

This domain is modeled to simulate logistics operations where drivers, trucks, and objects must be moved between different locations. The primary focus is on transporting objects via trucks, either driven by drivers or moved manually by walking. The key actions in this domain include loading and unloading trucks, drivers boarding and disembarking trucks, driving trucks between connected locations, and walking when no truck is involved. 

22 

Under Review 

This domain is formally represented in the IPC using the PDDL as outlined below: 

PDDL description for Driverlog domain (define (domain driverlog) (:requirements :typing) (:types location locatable - object driver truck obj - locatable) (:predicates (at ?obj - locatable ?loc - location) (in ?obj1 - obj ?obj - truck) (driving ?d - driver ?v - truck) (link ?x ?y - location) (path ?x ?y - location) (empty ?v - truck)) (:action LOAD-TRUCK :parameters (?obj - obj ?truck - truck ?loc - location) :precondition (and (at ?truck ?loc) (at ?obj ?loc)) :effect (and (not (at ?obj ?loc)) (in ?obj ?truck))) (:action UNLOAD-TRUCK :parameters (?obj - obj ?truck - truck ?loc - location) :precondition (and (at ?truck ?loc) (in ?obj ?truck)) :effect (and (not (in ?obj ?truck)) (at ?obj ?loc))) (:action BOARD-TRUCK :parameters (?driver - driver ?truck - truck ?loc - location) :precondition (and (at ?truck ?loc) (at ?driver ?loc) (empty ?truck)) :effect (and (not (at ?driver ?loc)) (driving ?driver ?truck) (not (empty ?truck)))) (:action DISEMBARK-TRUCK :parameters (?driver - driver ?truck - truck ?loc - location) :precondition (and (at ?truck ?loc) (driving ?driver ?truck)) :effect (and (not (driving ?driver ?truck)) (at ?driver ?loc) (empty ?truck))) (:action DRIVE-TRUCK :parameters (?truck - truck ?loc-from - location ?loc-to - location ?driver - driver) :precondition (and (at ?truck ?loc-from) (driving ?driver ?truck) (link ?loc-from ?loc-to)) :effect (and (not (at ?truck ?loc-from)) (at ?truck ?loc-to))) 

23 

Under Review 

(:action WALK :parameters (?driver - driver ?loc-from - location ?loc-to - location) :precondition (and (at ?driver ?loc-from) (path ?loc-from ?loc-to)) :effect (and (not (at ?driver ?loc-from)) (at ?driver ?loc-to))) ) 

Predicates define the number of fluents in a state: 

- (at ?obj - locatable ?loc - location) 

- (in ?obj1 - obj ?obj - truck) 

- (driving ?d - driver ?v - truck) 

- (link ?x ?y - location) 

- (path ?x ?y - location) 

- (empty ?v - truck) 

_O_ (2[5] _[N]_[ 2] _[−][N]_ ) (3) 

## F.4 GRIPPER 

This domain represents a transportation domain where a robot with two grippers can move between rooms, pick up objects, and drop them off. The robot can only hold one object in each gripper at a time. This domain could solve tasks where the robot needs to transport multiple objects from one room to another by strategically moving, picking up, and dropping items. 

This domain is formally represented in the IPC using the PDDL as outlined below: 

PDDL description for Gripper domain (define (domain gripper-strips) (:requirements :strips :typing) (:types room object robot gripper) (:predicates (at-robby ?r - robot ?x - room) (at ?o - object ?x - room) (free ?r - robot ?g - gripper) (carry ?r - robot ?o - object ?g - gripper)) (:action move :parameters (?r - robot ?from ?to - room) :precondition (and (at-robby ?r ?from)) :effect (and (at-robby ?r ?to) (not (at-robby ?r ?from)))) (:action pick :parameters (?r - robot ?obj - object ?room - room ?g - gripper) :precondition (and (at ?obj ?room) (at-robby ?r ?room) (free ?r ?g)) :effect (and (carry ?r ?obj ?g) (not (at ?obj ?room)) (not (free ?r ?g)))) 

24 

Under Review 

(:action drop :parameters (?r - robot ?obj - object ?room - room ?g - gripper) :precondition (and (carry ?r ?obj ?g) (at-robby ?r ?room)) :effect (and (at ?obj ?room) (free ?r ?g) (not (carry ?r ?obj ?g))))) 

Predicates define the number of fluents in a state: 

- (carry ?r - robot ?o - object ?g - gripper) 

- (at-robby ?r - robot ?x - room) 

- (at ?o - object ?x - room) 

- (free ?r - robot ?g - gripper) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints_images/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints.pdf-0025-07.png)


(4) 

## F.5 MYSTERY 

The Mystery domain represents a transportation system where vehicles move between locations, constrained by fuel levels, and can load or unload cargo, constrained by available space. The key aspects of this domain are managing fuel for vehicle movement and managing space for loading and unloading cargo. Locations are connected, and the system also handles fuel transitions, space transitions, and the movement of objects across a grid of locations. 

This domain is formally represented in the IPC using the PDDL as outlined below: 

## PDDL description for Mystery domain 

(define (domain mystery-strips) (:requirements :typing) (:types space fuel location movable - object vehicle cargo - movable) (:predicates (at ?v - movable ?l - location) (conn ?l1 ?l2 - location) (has-fuel ?l - location ?f - fuel) (fuel-neighbor ?f1 ?f2 - fuel) (in ?c - cargo ?v - vehicle) (has-space ?v - vehicle ?s - space) (space-neighbor ?s1 ?s2 - space)) (:action move :parameters (?v - vehicle ?l1 ?l2 - location ?f1 ?f2 - fuel) :precondition (and (at ?v ?l1) (conn ?l1 ?l2) (has-fuel ?l1 ?f1) (fuel-neighbor ?f2 ?f1)) :effect (and (not (at ?v ?l1)) (at ?v ?l2) (not (has-fuel ?l1 ?f1)) (has-fuel ?l1 ?f2))) 

25 

Under Review 

(:action load :parameters (?c - cargo ?v - vehicle ?l - location ?s1 ?s2 - space) :precondition (and (at ?c ?l) (at ?v ?l) (has-space ?v ?s1) (space-neighbor ?s2 ?s1)) :effect (and (not (at ?c ?l)) (in ?c ?v) (not (has-space ?v ?s1)) (has-space ?v ?s2))) (:action unload :parameters (?c - cargo ?v - vehicle ?l - location ?s1 ?s2 - space) :precondition (and (in ?c ?v) (at ?v ?l) (has-space ?v ?s1) (space-neighbor ?s1 ?s2)) :effect (and (not (in ?c ?v)) (at ?c ?l) (not (has-space ?v ?s1)) (has-space ?v ?s2))) ) 

Predicates define the number of fluents in a state: 

- (at ?v - movable ?l - location) 

- (has-fuel ?l - location ?f - fuel) 

- (in ?c - cargo ?v - vehicle) 

- (has-space ?v - vehicle ?s - space) 

- (conn ?l1 ?l2 - location) 

- (fuel-neighbor ?f1 ?f2 - fuel) 

- (space-neighbor ?s1 ?s2 - space) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints_images/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints.pdf-0026-10.png)


## F.6 SATELLITE 

The Satellite domain represents a simplified model for managing and controlling satellites and their onboard instruments. The goal in this domain is to coordinate the behavior of satellites, including turning them toward desired directions, powering instruments on and off, calibrating instruments, and capturing images. 

This domain is formally represented in the IPC using the PDDL as outlined below: 

## PDDL description for Satellite domain 

(define (domain satellite) (:requirements :strips :typing) (:types satellite direction instrument mode) 

26 

Under Review 

(:predicates (on ~~b~~ oard ?i - instrument ?s - satellite) (supports ?i - instrument ?m - mode) (pointing ?s - satellite ?d - direction) (power ~~a~~ vail ?s - satellite) (power ~~o~~ n ?i - instrument) (calibrated ?i - instrument) (have ~~i~~ mage ?d - direction ?m - mode) (calibration ~~t~~ arget ?i - instrument ?d - direction)) 

## (:action turn ~~t~~ o 

:parameters (?s - satellite ?d ~~n~~ ew - direction ?d ~~p~~ rev - direction) :precondition (and (pointing ?s ?d ~~p~~ rev)) :effect (and (pointing ?s ?d ~~n~~ ew) (not (pointing ?s ?d ~~p~~ rev)))) 

## (:action switch ~~o~~ n 

:parameters (?i - instrument ?s - satellite) :precondition (and (on ~~b~~ oard ?i ?s) (power ~~a~~ vail ?s)) :effect (and (power ~~o~~ n ?i) (not (calibrated ?i)) (not (power ~~a~~ vail ?s)))) 

(:action switch ~~o~~ ff 

:parameters (?i - instrument ?s - satellite) :precondition (and (on ~~b~~ oard ?i ?s) (power ~~o~~ n ?i)) :effect (and (not (power ~~o~~ n ?i)) (power ~~a~~ vail ?s))) 

(:action calibrate :parameters (?s - satellite ?i - instrument ?d - direction) :precondition (and (on ~~b~~ oard ?i ?s) (calibration ~~t~~ arget ?i ?d) (pointing ?s ?d) (power ~~o~~ n ?i)) :effect (calibrated ?i)) (:action take ~~i~~ mage :parameters (?s - satellite ?d - direction ?i - instrument ?m - mode) :precondition (and (calibrated ?i) (on ~~b~~ oard ?i ?s) (supports ?i ?m) (power ~~o~~ n ?i) (pointing ?s ?d)) :effect (have image ?d ?m))) 

Predicates define the number of fluents in a state: 

- (on-board ?i - instrument ?s - satellite) 

- (supports ?i - instrument ?m - mode) 

- (pointing ?s - satellite ?d - direction) 

- (have-image ?d - direction ?m - mode) 

- (calibration-target ?i - instrument ?d - direction) 

- (power-avail ?s - satellite) 

- (power-on ?i - instrument) 

- (calibrated ?i - instrument) 

_O_ (2[5] _[N]_[2][+3] _[N]_ ) 

(6) 

27 

Under Review 

## F.7 SPANNER 

This domain models a simple world where a man moves between locations, picks up spanners, and uses them to tighten loose nuts. The actions available to the man involve walking between locations, picking up the spanner, and tightening nuts using the spanner if all conditions are met. 

This domain is formally represented in the IPC using the PDDL as outlined below: 

## PDDL description for Spanner domain 

(define (domain spanner) (:requirements :typing :strips) (:types location locatable - object man nut spanner - locatable ) (:predicates (at ?m - locatable ?l - location) (carrying ?m - man ?s - spanner) (useable ?s - spanner) (link ?l1 - location ?l2 - location) (tightened ?n - nut) (loose ?n - nut)) (:action walk :parameters (?start - location ?end - location ?m - man) :precondition (and (at ?m ?start) (link ?start ?end)) :effect (and (not (at ?m ?start)) (at ?m ?end))) (:action pickup ~~s~~ panner :parameters (?l - location ?s - spanner ?m - man) :precondition (and (at ?m ?l) (at ?s ?l)) :effect (and (not (at ?s ?l)) (carrying ?m ?s))) (:action tighten nut :parameters (?l - location ?s - spanner ?m - man ?n - nut) :precondition (and (at ?m ?l) (at ?n ?l) (carrying ?m ?s) (useable ?s) (loose ?n)) :effect (and (not (loose ?n))(not (useable ?s)) (tightened ?n))) ) 

Predicates define the number of fluents in a state: 

- (at ?m - locatable ?l - location) 

- (carrying ?m - man ?s - spanner) 

- (link ?l1 - location ?l2 - location) 

- (useable ?s - spanner) 

- (tightened ?n - nut) 

- (loose ?n - nut)) 

_O_ (2[3] _[N]_[2][+2] _[N]_ ) 

(7) 

28 

Under Review 

## F.8 VISITALL 

The VisitAll domain is focused on controlling a robot that needs to visit all places on a connected grid. The robot’s movement is governed by the connectivity of places, and each move changes the robot’s location and marks the visited place. The task is essentially to traverse the entire grid, visiting every place, while ensuring the robot follows connectivity constraints. 

This domain is formally represented in the IPC using the PDDL as outlined below: 

## PDDL description for VisitAll domain 

- (define (domain grid-visit-all) (:requirements :typing) 

- (:types place - object) 

(:predicates (connected ?x ?y - place) (at-robot ?x - place) (visited ?x - place)) 

(:action move :parameters (?curpos ?nextpos - place) :precondition (and (at-robot ?curpos) (connected ?curpos ?nextpos)) :effect (and (at-robot ?nextpos) (not (at-robot ?curpos)) (visited ?nextpos))) ) 

Predicates define the number of fluents in a state: 

- (connected ?x ?y - place) 

- (at-robot ?x - place) 

- (visited ?x - place) 

_O_ (2 _[N]_[2][+] _[N]_ ) (8) 

## G PLANNING DESCRIPTION AND TOOLS 

Planning, at its core, involves determining a sequence of actions that transforms the world from an initial state to a goal state. A world state specifies which fluents are true or false at any given time. The planning domain, denoted as _D_ , specifies the fluents, actions, and their effects within the system. Typically, planning domains are represented using formal languages such as PDDL or ASP. In these languages, a transition function Φ _D_ : _states × actions → states_ defines how actions transform an initial state into a resulting state. 

## G.1 PLANNING DOMAIN DEFINITION LANGUAGE (PDDL) 

PDDL is a formal language developed for expressing planning problems and domain models. Since its inception, PDDL has been extended to address increasingly complex planning scenarios, particularly those involving deterministic problems (Haslum et al., 2019). PDDL facilitates the specification of both the planning domains and problem instances, including objects, initial, and goal states. In this study, we employ the “STRIPS” (Stanford Research Institute Problem Solver) subset of PDDL (Fikes & Nilsson, 1971). Additionally, the domains are “typed”, meaning that objects in the planning problem are assigned specific types and subtypes, ensuring a structured representation of the problem space. 

## G.2 ANSWER SET PROGRAMMING (ASP) 

ASP is a declarative approach to problem-solving based on logic programming and non-monotonic reasoning. Unlike traditional planning languages like PDDL, ASP focuses on defining constraints and rules that describe potential solutions, rather than directly encoding state transitions. In ASP, a 

29 

Under Review 

problem is encoded as a logic program consisting of rules, facts, and constraints, and the solution is an ”answer set” that satisfies all the constraints of the problem. In this study, we use ASP to generate the complete state-space by applying the sequence of actions starting from the initial state. 

## H CLASSIFICATION OF FLUENTS 

The subsequent sections provide a detailed classification of fluents across all 13 domains included in ACTIONREASONINGBENCH, as described in section 3.2. 

## H.1 BLOCKSWORLD 

In the BLOCKSWORLD domain, the fluents are categorized as follows: 

1. Static Properties - No static properties are present 

2. Base Fleunts - onTable(block) 

3. Derived Fluents - clear(block), handEmpty 

4. Self-Derived Fluents - holding(block), on(block,block) 

## H.2 DEPOTS 

In the DEPOTS domain, the fluents are categorized as follows: 

   1. Static Properties - No static properties are present 

   2. Base Fleunts - No base fluents are present 

   3. Derived Fluents - clear(surface), available(hoist) 

   4. Self-Derived Fluents - at(locatable,place), on(crate,surface), in(crate,truck), lifting(hoist,crate) 

- H.3 DRIVERLOG 

In the DRIVERLOG domain, the fluents are categorized as follows: 

   1. Static Properties - link(location,location), path(location,location) 

   2. Base Fleunts - No base fluents are present 

   3. Derived Fluents - empty(truck) 

   4. Self-Derived Fluents - at(locatable,location), in(object,truck), driving(driver,truck) 

- H.4 GRIPPERS 

In the GRIPPERS domain, the fluents are categorized as follows: 

   1. Static Properties - No static properties are present 

   2. Base Fleunts - carry(robot,object,gripper) 

   3. Derived Fluents - free(robot,gripper) 

   4. Self-Derived Fluents - at ~~r~~ obby(robot,room), at(object,room) 

- H.5 MYSTERY 

In the MYSTERY domain, the fluents are categorized as follows: 

1. Static Properties - space ~~n~~ eighbor(space,space), fuel ~~n~~ eighbor(fuel,fuel), conn(location,location) 

30 

Under Review 

   2. Base Fleunts - No base fluents are present 

   3. Derived Fluents - No derived fluents are present 

   4. Self-Derived Fluents - at(movable,location), in(cargo,vehicle), has ~~s~~ pace(vehicle,space), has ~~f~~ uel(location,fuel) 

- H.6 SATELLITE 

In the SATELLITE domain, the fluents are categorized as follows: 

1. Static Properties - on ~~b~~ oard(instrument,satellite), supports(instrument,mode), calibration ~~t~~ arget(instrument,direction) 

2. Base Fleunts - power ~~o~~ n(instrument), calibrated(instrument), have ~~i~~ mage(direction,mode) 

3. Derived Fluents - power ~~a~~ vail(satellite) 

4. Self-Derived Fluents - pointing(satellite,direction) 

## H.7 SPANNER 

In the SPANNER domain, the fluents are categorized as follows: 

1. Static Properties - link(location,location) 

2. Base Fleunts - carrying(man,spanner), tightened(nut) 

3. Derived Fluents - loose(nut), useable(spanner) 

4. Self-Derived Fluents - at(locatable,location) 

## H.8 VISITALL 

In the VISITALL domain, the fluents are categorized as follows: 

1. Static Properties - connected(place,place) 

2. Base Fleunts - No base fluents are present 

3. Derived Fluents - visited(place) 

4. Self-Derived Fluents - at ~~r~~ obot(place) 

## I RESPOSNES ON RAMIFICATION DATASET 

In the following sections, we illustrate some instances and responses that we see from GPT-4o and o1-preview. 

31 

Under Review 

## I.1 GPT-4O 

## A Ramificaiton question for State Tracking 

## [DOMAIN DESCRIPTION] 

A block can only be picked up if it is clear, on the table, and the hand is empty, resulting in the block being held. A held block can be put down, placing it back on the table. Blocks can be stacked if the first block is held and the second block is clear, causing the first block to rest on top of the second. Unstacking occurs when the hand is empty, the first block is clear, and on top of the second, resulting in the first block being held again. A block is clear if nothing is on top of it and it’s not held, and the hand is empty if it’s not holding anything. Blocks are stable when clear and on the table, and they can be painted if stable and the hand is empty. A block is considered on display if it can be painted and has no other block on top of it. 

## [INITIAL CONDITIONS] 

Block b1 is stacked on block b4, block b2 has no blocks on top of it, block b2 is stacked on block b6, block b3 has no blocks on top of it, block b3 is stacked on block b5, block b4 is positioned on the table, block b5 is stacked on block b7, block b6 is positioned on the table, block b7 is stacked on block b1 and the hand is empty. 

## [QUESTION] 

Given the initial condition, the following actions are taken: block b3 is unstacked from block b5 to achieve the current state. In this state, if block b3 is placed down, what are all the valid properties of the state (including both affirmative and negated properties)? If there are no valid properties, state None. 

## Response to the above question 

## [REASON] 

To determine the valid properties of the state after block b3 is placed down, we need to follow the sequence of actions and check the conditions for each property. 

1. **Initial Conditions:** - b1 is on b4. - b2 is on b6. - b3 is on b5. - b4 is on the table. - b5 is on b7. - b6 is on the table. - b7 is on b1. - The hand is empty. 2. **Action: Unstack b3 from b5:** - The hand must be empty. - b3 must be clear and on top of b5. - Result: b3 is held, b5 is now clear. 3. **Current State after unstacking b3:** - b1 is on b4. - b2 is on b6. - b3 is held. 

32 

Under Review 

- b4 is on the table. - b5 is on b7 and is clear. - b6 is on the table. - b7 is on b1. - The hand is not empty (holding b3). 

4. **Action: Place b3 down:** - The hand must be holding b3. - Result: b3 is placed on the table, the hand is empty. 

5. **Final State after placing b3 down:** - b1 is on b4. - b2 is on b6. - b3 is on the table and clear. - b4 is on the table. - b5 is on b7 and is clear. - b6 is on the table. - b7 is on b1. - The hand is empty. 

6. **Valid Properties:** - b1 is on b4. - b2 is on b6. - b3 is on the table. - b3 is clear. - b4 is on the table. - b5 is on b7. - b5 is clear. - b6 is on the table. - b7 is on b1. - The hand is empty. - b1 is not clear (b7 is on it). - b2 is not clear (nothing on it but it is on b6). - b4 is clear (nothing on it and on the table). - b6 is clear (nothing on it and on the table). - b7 is not clear (b5 is on it). - b3 is stable (clear and on the table). - b4 is stable (clear and on the table). - b6 is stable (clear and on the table). - b3 can be painted (stable and hand is empty). - b4 can be painted (stable and hand is empty). - b6 can be painted (stable and hand is empty). - b3 is on display (can be painted and no block on top). - b4 is on display (can be painted and no block on top). - b6 is on display (can be painted and no block on top). [ANSWER] - b1 is on b4. - b2 is on b6. - b3 is on the table. - b3 is clear. - b4 is on the table. - b5 is on b7. - b5 is clear. - b6 is on the table. - b7 is on b1. 

33 

Under Review 

- The hand is empty. - b1 is not clear. 

- b2 is not clear. 

- b4 is clear. - b6 is clear. - b7 is not clear. 

- b3 is stable. - b4 is stable. - b6 is stable. - b3 can be painted. - b4 can be painted. - b6 can be painted. - b3 is on display. - b4 is on display. - b6 is on display. 

From the above response, we can see that block b4 is **not** stable, but the model says it is stable. 

- I.2 O1-PREVIEW 

## A Ramificaiton question for State Tracking 

## [DOMAIN DESCRIPTION] 

A block can only be picked up if it is clear, on the table, and the hand is empty, resulting in the block being held. A held block can be put down, placing it back on the table. Blocks can be stacked if the first block is held and the second block is clear, causing the first block to rest on top of the second. Unstacking occurs when the hand is empty, the first block is clear, and on top of the second, resulting in the first block being held again. A block is clear if nothing is on top of it and it’s not held, and the hand is empty if it’s not holding anything. Blocks are stable when clear and on the table, and they can be painted if stable and the hand is empty. A block is considered on display if it can be painted and has no other block on top of it. 

[INITIAL CONDITIONS] 

Block b1 is positioned above block b7, block b2 has no blocks on it, block b2 is stacked on block b6, block b3 is stacked on block b4, block b4 is situated on the table, block b5 has no blocks on it, block b5 is situated on the table, block b6 is stacked on block b3, block b7 is on the table, block b8 is positioned above block b1, block b9 has no blocks on it, block b9 is stacked on block b8, and the hand is empty. 

## [QUESTION] 

Based on the initial condition, the following actions are taken: block b9 is removed from block b8 to achieve the current state. In this state, identify all valid properties that do not include negations and list them; if there are no such properties, indicate None. 

34 

Under Review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints_images/iclr_2025_ActionReasoningBench_Reasoning_about_Actions_with_and_without_Ramification_Constraints.pdf-0035-01.png)


**----- Start of picture text -----**<br>
Response to the above question<br>[REASON]<br>After removing block b9 from block b8, block b9 is placed on the table, and the hand<br>remains empty. Block b9 now has no blocks on top of it and is not held, so it is **clear**.<br>Since it is on the table, it is also **stable**. With the hand empty and block b9 being stable,<br>it **can be painted**. As there are no blocks on top of it, block b9 is **on display**.<br>Similarly, block b5 is on the table with no blocks on top and is not held, making it<br>**clear** and **stable**. It **can be painted** and is **on display**.<br>Block b7 is on the table and, with no blocks explicitly on top of it after the action, is<br>**clear**, **stable**, **can be painted**, and is **on display**.<br>Blocks b1 and b8 have no blocks on top of them and are not held, so they are **clear**.<br>Block b2 has no blocks on top and is not held, so it is **clear**, though it is not on<br>the table and thus not stable.<br>The hand is explicitly stated to be **empty** in the current state.<br>[ANSWER]<br>- The hand is empty<br>- Block b1 is clear<br>- Block b2 is clear<br>- Block b5 is clear<br>- Block b5 is on the table<br>- Block b5 is stable<br>- Block b5 can be painted<br>- Block b5 is on display<br>- Block b7 is clear<br>- Block b7 is on the table<br>- Block b7 is stable<br>- Block b7 can be painted<br>- Block b7 is on display<br>- Block b8 is clear<br>- Block b9 is clear<br>- Block b9 is on the table<br>- Block b9 is stable<br>- Block b9 can be painted<br>- Block b9 is on display<br>**----- End of picture text -----**<br>


From the above response, we can see that block b1 is **not** clear, but the model says it is clear. 

35 

