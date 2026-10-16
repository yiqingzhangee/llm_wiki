## **PARTNR: A Benchmark for Planning and Reasoning in Embodied Multi-agent Tasks** 

**Matthew Chang** , **Gunjan Chhablani** , **Alexander Clegg** , **Mikael Dallaire Cote** , **Ruta Desai** , **Michal Hlavac** , **Vladimir Karashchuk** , **Jacob Krantz** , **Roozbeh Mottaghi** , **Priyam Parashar** , **Siddharth Patki** , **Ishita Prasad** , **Xavier Puig** , **Akshara Rai** , **Ram Ramrakhya** , **Daniel Tran** , **Joanne Truong** , **John M. Turner** , **Eric Undersander** , **Tsung-Yen Yang** 

Work done at FAIR Meta, Alphabetical author order 

We present a benchmark for Planning And Reasoning Tasks in humaN-Robot collaboration ( **PARTNR** ) designed to study human-robot coordination in household activities. **PARTNR** tasks exhibit characteristics of everyday tasks, such as spatial, temporal, and heterogeneous agent capability constraints. We employ a semi-automated task generation pipeline using Large Language Models (LLMs), incorporating simulation in the loop for grounding and verification. **PARTNR** stands as the largest benchmark of its kind, comprising 100,000 natural language tasks, spanning 60 houses and 5,819 unique objects. We analyze state-of-the-art LLMs on **PARTNR** tasks, across the axes of planning, perception and skill execution. The analysis reveals significant limitations in SoTA models, such as poor coordination and failures in task tracking and recovery from errors. When LLMs are paired with _real_ humans, they require 1.5x as many steps as two humans collaborating and 1.1x more steps than a single human, underscoring the potential for improvement in these models. We further show that fine-tuning smaller LLMs with planning data can achieve performance on par with models 9 times larger, while being 8.6x faster at inference. Overall, **PARTNR** highlights significant challenges facing collaborative embodied agents and aims to drive research in this direction. 

**Date:** November 4, 2024 

**Code:** https://github.com/facebookresearch/partnr-planner **Website:** https://aihabitat.org/partnr 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0001-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0001-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0001-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0001-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0001-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0001-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0001-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0001-14.png)


## **1 Introduction** 

Imagine a domestic robot that collaborates with humans in daily activities using natural language, akin to human-to-human interactions. This scenario requires two key features: the dynamic collaboration between the robot and the human, and the use of natural language for interaction. Current benchmarks in embodied AI typically satisfy one or the other condition; either robots operate in isolation (Shridhar et al., 2020; Zhu et al., 2023; Krantz et al., 2020; Majumdar et al., 2024), or tasks are not specified in natural language (Yenamandra et al., 2023; Puig et al., 2024; Szot et al., 2023; Jain et al., 2020). Despite significant progress in the field of embodied AI, there remains a gap in realistic benchmarks that evaluate robots in collaborative settings. To bridge this gap, we introduce Planning And Reasoning Tasks in humaN-Robot collaboration ( **PARTNR** ), a novel benchmark that evaluates the ability of embodied AI agents to collaborate with humans across a range of household activities in simulated indoor environments (Figure 1). 

**PARTNR** consists of 100,000 natural language instructions paired with tailored evaluation functions, focusing on four task types: (1) constraint-free, where sub-tasks can be completed in any manner by either agent, (2) spatial tasks that contain spatial constraints, (3) temporal tasks that require ordered execution, and (4) heterogeneous tasks that include actions that cannot be completed by one of the agents. Beyond the conventional challenges of long-horizon planning, novel partially observed environments, and large state and action spaces, **PARTNR** emphasizes the need for effective collaboration dynamics, such as task division and tracking partner’s progress. 

Curating such a benchmark of large-scale, natural language tasks with tailored evaluation functions presents significant challenges. Current benchmarks typically rely on either templated tasks (Shridhar et al., 2020; 

1 

**PARTNR:** A collaboration dataset for planning and reasoning in multi-agent tasks 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Automatic task and evaluation generation Human-In-The-Loop<br>Human/<br>We collect collaboration<br>Human<br>data on the web.<br>Task:  Take the glass and bowl to the kitchen and wash them.<br>Human/<br>Agent<br>LLM-based helper agents<br>We propose modular LLM-based agent<br>baselines to collaborate in our benchmark.<br>LLM Planner<br>Evaluation<br>IN(kitchen, glass_1) Scene Graph Low-Level Skills<br>IN(kitchen, bowl_1)<br>IS_CLEAN(glass_1)<br>IS_CLEAN(bowl_1)<br>**----- End of picture text -----**<br>


**Figure 1** We present **PARTNR** , a benchmark for planning and reasoning in embodied multi-agent tasks, featuring 100,000 everyday tasks and evaluation functions generated semi-automatically, spanning 60 houses and 5,819 unique objects. We analyze LLM-based planning agents and also provide a human-in-the-loop tool to evaluate how agents collaborate with real humans. 

Zhang et al., 2024a) or tasks and evaluations crafted by humans (Mandi et al., 2024; Li et al., 2023a), which can restrict the diversity or the scale of the datasets. To overcome this, we propose a semi-automated generation method using Large Language Models (LLMs) with simulation-in-the-loop grounding. First, a Large Language Model (LLM) generates task and evaluation functions, which are grounded in the objects and furniture of a simulated house. Next, we employ simulation-in-the-loop to filter out hallucinations and infeasible instructions, complemented by human annotation to enhance diversity and accuracy. Subsequently, a set of 1,000 verified instructions and evaluation functions, along with diverse simulation houses, are utilized to guide an LLM through in-context prompting to create 100,000 tasks. 

As **PARTNR** consists of natural language tasks and LLMs have shown strong results in planning (Yao et al., 2023; Ahn et al., 2022; Huang et al., 2022), we explore prompting and fine-tuning LLMs, to assess their effectiveness in collaborative scenarios. We study the effect of observability of the environment (i.e., full or partial), centralized vs. decentralized multi-agent control, learned or privileged oracle robot skills, and different ways of grounding the 3D world information for LLM-based planning. Beyond these automated evaluations with _synthetic_ human partners, we also perform evaluations with _real_ humans-in-the-loop, with people performing the task alone, with a human partner, or with an LLM-guided robot partner. Overall, we find that LLMs struggle at coordination, task tracking and dealing with perception and skill errors. While humans are able to solve 93% of **PARTNR** tasks, SoTA LLMs can only successfully complete 30% under non-privileged conditions. Moreover, in decentralized multi-agent settings, task completion takes 1.3x more steps than single-agent, due to poor tracking of partner actions, resulting in extraneous actions. In contrast, human pairs outperform single humans, in our human-in-the-loop experiments, highlighting potential for improving LLM collaboration strategies. LLMs also struggle to recover from skill failures and perception grounding errors, resulting in lower performance when privileged skills and privileged perception are removed. When comparing model sizes, we observe that a smaller fine-tuned Llama3.1-8B achieves a similar performance to a Llama3.1-70B without finetuning, while being 8.6x faster. This faster inference plays an important role when evaluated with real humans-in-the-loop, where the finetuned model takes fewer steps and offloads more tasks from the human. 

In summary, **PARTNR** enables reproducible, large-scale, and systematic evaluations of embodied agents in a 

2 

||Environment|Multi-Agent|Language|Action Space|Task Types|Num tasks|
|---|---|---|---|---|---|---|
|Overcooked (Carroll et al., 2019)|2D|✓||HL|C|4|
|RoboGen (Wang et al., 2024)|3D||✓|LL+HL|CST|106|
|GenSim (Katara et al., 2023)|3D||✓|LL|CS|100|
|RoCo (Mandi et al., 2024)|3D|✓|✓|LL|CS|6|
|FurnMove (Jain et al., 2020)|3D-S|✓||LL|C|30|
|RoboCasa (Nasiriany et al., 2024)|3D-S|✓|✓|LL|CST|100|
|ALFRED (Shridhar et al., 2020)|3D-S||✓|HL|CST|25,743|
|BEHAVIOR-1K (Li et al., 2023a)|3D-M|||LL+HL|CST|1,000|
|WAH (Puig et al., 2021)|3D-M|✓||HL|C|1,211|
|Co-ELA (Zhang et al., 2024a)|3D-M|✓|✓|HL|C|44|
|**PARTNR**|3D-M|✓|✓|LL+HL|CSTH|100,000|



**Table 1 Comparison to similar embodied benchmarks.** We compare **PARTNR** to embodied AI benchmarks, focusing on natural language and multi-agent collaboration tasks. Comparison axes are – **Environment:** Household single room (S), multi-room (M). **Action Space:** High-Level Actions (HL), Low-level Actions (LL). **Task Types:** Constraint-free (C), Spatial (S), Temporal (T), Heterogeneous (H) **Num tasks:** We measure tasks as the number of unique scene-goal pairs. 

wide variety of collaborative scenarios. Through systematic evaluation, we reveal critical insights into the current limitations of LLM-based planners, opening interesting future research directions. 

## **2 Related Work** 

**Language-based benchmarks in Embodied AI.** A large body of work on language benchmarks in Embodied AI has focused on navigation (Anderson et al., 2018; Krantz et al., 2020; Chen et al., 2019) or Embodied Question Answering (Das et al., 2018; Majumdar et al., 2024) which involve navigation and information gathering but do not require agents to modify their environments. Closer to our work are instruction-following benchmarks (Shridhar et al., 2020, 2021; Puig et al., 2018; Wang et al., 2024; James et al., 2020; Gong et al., 2023), where agents interact with environments to complete tasks described via language, though the diversity of tasks is limited. In contrast, we leverage LLMs to generate diverse task definitions and scene initializations, and extend them to _multi-agent_ settings. The idea of scaling up task generation using LLMs has been explored in a few recent works (Katara et al., 2023; Wang et al., 2024; Xian et al., 2023; Nasiriany et al., 2024). However, these works tend to focus on single-agent tasks that span relatively short horizons, while we consider long-horizon, multi-agent problems. Table 1 compares relevant benchmarks with **PARTNR** . 

**Embodied multi-agent benchmarks.** Multiple works have proposed embodied multi-agent benchmarks (Puig et al., 2023; Agashe et al., 2023; Zhang et al., 2024a; Jain et al., 2019; Suarez et al., 2019). Many of these benchmarks focus on coordination in simple 2D environments, limiting their applicability to real world settings (Agashe et al., 2023; Carroll et al., 2019). Recent works have developed benchmarks studying collaboration in more realistic environments and activities (Puig et al., 2021; Zhang et al., 2024a; Jain et al., 2019; Puig et al., 2024; Szot et al., 2023), focusing on rearranging objects or furniture in large, partially observable 3D environments (Puig et al., 2021, 2024; Jain et al., 2019; Szot et al., 2023), or manipulating objects in a counter-top space (Mandi et al., 2024). However, these benchmarks are typically limited to a predefined and reduced set of tasks, often not described in natural language and primarily involving object rearrangement. In contrast, **PARTNR** covers an open set of tasks, each described in natural language, requiring agents to rearrange objects with spatial and temporal constraints, as well as requiring heterogeneous actions that can only be done by the human agent, (e.g., washing dishes or turning on the oven). 

**LLMsfordecisionmaking.** Several works use LLMs as interactive policies, demonstrating their strong capabilities but also highlighting challenges in grounding them with observations and actions (Huang et al., 2022; Yao et al., 2023; Ahn et al., 2022; Huang et al., 2023; Zeng et al., 2022). Some approaches improve grounding by prompting LLMs with demonstrations and task-specific constraints (Huang et al., 2022; Yao et al., 2023), or by integrating LLMs with external modules for multi-modal reasoning (Ahn et al., 2022; Huang et al., 2023; Zeng et al., 2022). Toolformer (Schick et al., 2023) allows LLMs to call APIs for information retrieval or environmental interaction. For instance, APIs can be used to call low-level policies (Driess et al., 2023), to 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Sim-Aware Generation Sim-In-The-Loop Filtering (1k) Large-Scale Generation (100k) Human-In-The-Loop Filtering<br>Task Generation Task Generation<br>Task &<br>LLM Instruction Evaluation [Living Room] Seed LLM Instruction<br>Object Cfg  Does not exist Object Cfg<br>Eval Generation Human Annotation Eval Generation<br>LLM Evaluation Instruction Instruction LLM Evaluation<br>Instruction + Heuristics Function Evaluation Function Seed + Heuristics Function<br>Detail: Sim-Aware Generation<br>Object Library Task Generator: Task Instructions Eval Generator: Evaluation Function<br>Apple LLM Help me collect all toys  LLM + Heuristics IN( stuffed toy 0, shelf 1)<br>Stuffed toy in the house. Put them  IN( stuffed toy 1, shelf 1)<br>Toy truck next to the book on the  IS_NEXT_TO( stuffed toy 0, book 0)<br>Plate shelf. IS_NEXT_TO( stuffed toy 1, book_0)<br>BagBowl House Contents Initial Object Configurations<br>… Living Room:  couch 0 Stuffed toy 0 ON couch 0stuffed toy 1 ON living room floor Evaluation Metrics<br>  table 1 book 0 ON shelf 1 Percent Completion [0 to 1]<br>  chair 0… … Task Success [T/F]<br>Kitchen: Failure Explanation [str]<br>  cabinet 0<br>  counter 1…<br>…<br>**----- End of picture text -----**<br>


**Figure 2 The PARTNR generation pipeline.** Task and evaluation generators produce episodes, which are filtered and annotated for correctness. These episodes are then treated as seeds to achieve 100k-scale. Finally, episodes are vetted during human-in-the-loop collection. 

leverage VLMs for obtaining the state of the world (Huang et al., 2023; Zhang et al., 2024b), or to another LLM serving as a world model (Zhao et al., 2024). SayPlan (Rana et al., 2023) maintains a persistent graph of the current world-state (Gu et al., 2024; Werby et al., 2024), enabling detailed semantic and geometric queries. Our work synthesizes these ideas by encoding the environment into a graph, using tools to extract relevant information, and executing tasks through motor skills. Another line of work fine-tunes LLMs with data from the target environments, by learning input and output adaptors (Li et al., 2022; Szot et al., 2024; Xiang et al., 2023). We explore low-rank adaptation of LLMs with multi-agent data to enhance coordination and efficiency. While fewer studies focus on LLMs in multi-agent collaboration (Zhang et al., 2024a; Park et al., 2023; Li et al., 2023b; Zhou et al., 2024), one notable example is CoELA (Zhang et al., 2024a) that design collaborative agents, though limited in task diversity. Our work addresses a broader range of tasks and considers agents with varying capabilities, pushing the boundaries of multi-agent system collaboration in more complex scenarios. 

## **3 Benchmark Generation** 

We introduce **PARTNR** , a benchmark aimed at training and evaluating robots at solving natural language tasks in collaboration with humans. **PARTNR** covers four types of tasks: (1) Constraint-free tasks, where sub-tasks can be completed in any manner by either agent. For example, _"Let’s move all dirty plates to the sink."_ (2) Spatial tasks that require reasoning about the spatial positioning of objects. For instance, _"Let’s place the books on the shelf_ **next to each other** _."_ (3) Temporal tasks, where the sequence in which sub-tasks are executed is important. For example, _"Let’s remove the candles from the dining table_ **before** _bringing the plates to the table."_ (4) Heterogeneous tasks, involving actions that are beyond the robot’s capabilities. For example, _"Let’s_ **wash** _the dishes before putting them in shelves."_ In scenarios where the robot’s skills do not support washing, completing this task requires reasoning about agent capabilities. Our benchmark consists of natural language instructions and corresponding evaluation functions, both of which are generated at-scale using LLMs. Specifically, we generate 1,000 human-verified instructions and corresponding evaluation functions and use them as in-prompt examples to scale to 100,000 tasks in other scenes with different layouts and objects. A unique aspect of our automatic generation is the integration of an embodied simulator within the generation loop, which significantly reduces LLM errors such as hallucinations and infeasible actions. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Task Instruction Task Instruction<br>“Let’s tidy up the family room. The toys go in the toy box.  “We’re ready for dinner. Can you bring two cups to the<br>After that, set the plants on the shelf next to each other.” kitchen? Let’s fill them and take them to the dining table.”<br>Evaluation Function Evaluation Function<br>Propositions:<br>0  is_inside(["toy_fire_truck_0"], ["toy_box_0"])<br>1  is_inside(["toy_food_0"], ["toy_box_0"]) Of the n cups in the scene…<br>2  is_on_top(["plant_0"], ["shelf_0", "shelf_1"]) Propositions: Bring two<br>3  is_on_top(["plant_1"], ["shelf_0", "shelf_1"]) 0  is_in_room(["cup_0", "cup_1",…, "cup_n"], ["kitchen"], n=2)<br>4  is_next_to(["plant_0"], ["plant_1"]) 1  is_filled(["cup_0", "cup_1",…, "cup_n"], n=2)<br>Verify “is_next_to” when 2  is_on_top(["cup_0", "cup_1",…, "cup_n"], ["table_2"], n=2)<br>Dependencies: the plants are placed<br>0  WhileSatisfied([4], depends_on=[2,3]) Use the same cups throughout<br>Constraints:<br>The toys must be rearranged first 0  SameArgConstraint([0,1,2], arg=”object")<br>Constraints: 1  TerminalSatisfactionConstraint([1,2])<br>0  TemporalConstraint(<br>       [(0,2), (1,2), (0,3), Proposition 0 is temporary<br>        (1,3), (0,4), (1,4)] All  placements are terminal<br>   )<br>1  TerminalSatisfactionConstraint([0,1,2,3,4])<br>**----- End of picture text -----**<br>


**Figure 3 Task and evaluation example.** Language tasks have inherent complexity and ambiguity; both of which are supported by the structures of our evaluation functions. 

## **3.1 Simulation-in-the-loop task instruction generation** 

While LLM-based task generation has been studied in literature before (Katara et al., 2023; Wang et al., 2024; Xian et al., 2023; Nasiriany et al., 2024), these generations are not grounded beyond user-created in-context prompts. In **PARTNR** , we use a simulation-in-theloop generation technique to ground the LLM in the environment, agents and available actions. Specifically, we instantiate a simulation environment in the Habitat 3.0 simulator (Puig et al., 2024), populated with the HSSD dataset (Khanna et al., 2024), consisting of 60 unique houses and 5,819 OVMM objects (Yenamandra et al., 2023). The 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0005-04.png)


**----- Start of picture text -----**<br>
Constraint-Free<br>Spatial<br>Temporal<br>Heterogeneous<br>**----- End of picture text -----**<br>


**Figure 4 Distribution of task types in PARTNR.** The left plot displays the percentage of tasks with each characteristic. Constraint-free tasks by definition exclude the other types. The top right bars correspond to the dot combination below. 

simulated house is parsed into a list of rooms and available furniture, and passed to an LLM, along with all available objects. Using this information, the LLM is asked to generate free-form, viable tasks in the scene, along with an initial scene state description. For example, if the generated task is _"Clear dishes from the living room"_ , the LLM should generate an initial scene with multiple dishes in the living room. At this stage, additional objects are also added to the scene to create clutter in the environment. Once generated, the tasks, initial states, and clutter are instantiated in the simulator, and infeasible instructions are filtered. For example, if the house does not have a living room, _"Clear dishes from the living room"_ is invalid. Similarly, if the generated task requires actions not supported by the simulator, such as folding, the task is filtered. Generally, the rate of hallucinations is high, leading to a significant number of episodes being discarded. We observe that after filtering for infeasible instructions, the diversity in generated instructions is typically limited. For example, most of the instructions use the same objects (e.g., dishes) or similar rooms (e.g., kitchen or dining room). To increase diversity of the generated tasks, we manually annotate them to ensure task and object diversity, such as maintaining a balanced distribution of constraint-free, spatial, temporal, and heterogeneous tasks by modifying the instructions to elicit specific characteristics. This process results in 1,000 human annotated and simulation-verified tasks. 

Such manual annotation is not practical for large-scale generation. Instead, we leverage the human-annotated 1,000 instructions to scale generation by using them as in-prompt examples. We prompt the LLM with 

5 

both a house description and an example task, and instruct it to modify the task to fit the new house. For example a task like _"Clear all dishes from the living room"_ is modified to _"Clear all toys from the bedroom."_ This allows us to maintain the diversity of the original annotated instruction set, while ensuring a high likelihood of successful instantiation in the simulator. Qualitatively, we filtered or edited _∼_ 90% of free-form generated instructions and only _∼_ 10% of scaled instructions. We use _LLama3-70B-Instruct_ (Dubey et al., 2024) for all instruction generation. Finally, all tasks go through a human-in-the-loop filtering. In this step, humans attempt to solve the tasks using our human-in-the-loop tool (Appendix A.13) and eliminate physically infeasible instructions that are difficult to detect, such as requiring an object to be at two locations. Figure 2 provides an overview of our pipeline. Details on the generation process can be found in Appendix A.4 and prompts in Appendix A.14. 

## **3.2 Evaluation function generation** 

To determine if an agent successfully completed the instruction _"Clear all dishes from the living room"_ , we need an evaluation function that can validate the removal of all spoons, forks, and other dishes from any of the living rooms. However, manually annotating all necessary rearrangements and state changes of a task is time intensive and since all tasks are unique, impractical at scale. Similar to instruction generation, we employ an LLM to create an evaluation function that assesses task completion without requiring any manual annotations. Specifically, we leverage the ability of LLMs to generate predicate-based Python programs using three types of APIs: a list of **propositions** indicating _what_ relations between entities must be satisfied, a set of **dependencies** indicating _when_ propositions should be queried, and a set of **constraints** indicating _how_ propositions must be satisfied. We define an expressive vocabulary of each of these components to afford evaluation of all tasks in the benchmark (e.g., Figure 3). Closely related evaluation systems include defining tasks in PDDL (Ghallab et al., 1998) or BDDL (Srivastava et al., 2022). We choose to build a new Python-based evaluation system since neither have the expressivity to evaluate **PARTNR** tasks while maintaining human and LLM interpretability; for instance, BDDL does not support time-varying evaluation. Since **PARTNR** tasks have temporal dependencies (e.g. multi-step rearrangement), the input to the evaluation function is the complete sequence of simulator states during task execution. The evaluation function returns 3 metrics: (1) Percent Complete (PC _∈_ [0 _,_ 1]), the % of propositions that are satisfied w.r.t. constraints, (2) Success (S _∈{_ True _,_ False _}_ ), measuring if a task was successfully completed, defined as S := (PC = 1), and (3) Failure Explanation (FE), a human and LLM interpretable language description of why the agents failed to accomplish the task. See Appendix A.5 for details. 

We use _CodeLLama-70B-instruct_ (Roziere et al., 2023) for evaluation function generation. Exemplified in Figure 3, producing perfect evaluation functions is non-trivial. The LLM must correctly classify the entire space of possible actions against natural language instructions and the specific simulation environment, which can be quite complex. For example, in Figure 3, the instruction _"set the plants on the shelf"_ refers to _"the shelf"_ , but two shelves exist in the room. The evaluation function must allow either shelf while requiring placement of all plants, and finally account for a next-to relation. Any error or missing value in either a proposition or constraint invalidates the evaluation function. Consequently, we observe a large error rate in LLM generation, particularly pertaining to incorrect propositions and temporal sequencing constraints. 

To alleviate these inaccuracies, we follow a similar semi-automated procedure to instruction generation. We first generate evaluation functions for the 1,000 human-annotated instructions and perform manual annotation to correct them (Appendix A.6.3). This results in a dataset of 1,000 human-verified instruction and evaluation pairs. Next, we generate evaluations for the scaled 100,000 instruction set. Recall that the scaled instructions are generated by prompting the LLM with an example instruction from the annotated set. We retrieve the corresponding annotated evaluation function and prompt the LLM with it. This is similar to approaches such as retrieval-augmented generation (Lewis et al., 2020) and improves the accuracy of evaluation function generation from 50% to 92% as measured through manual inspection (Appendix A.6.2). As a final step, we ask human users to solve all **PARTNR** tasks using our human-in-the-loop evaluation tool (Appendix A.13). All tasks that cannot be solved by humans over 6 retries (3 single-user, 3 multi-user tries) are deemed infeasible, and removed from the dataset. We find that about 90% of instructions, and 92% of evaluation functions from automated generation are accurate, resulting in a combined generation accuracy of 90 _×_ 92 = 83%. 

6 

## **3.3 The PARTNR Dataset** 

The **PARTNR** dataset comprises of 100,000 episodes in 37 train scenes, 1,000 episodes in 13 validation scenes, and 1,000 episodes in 10 test scenes from the HSSD dataset (Khanna et al., 2024). After scaled generation, all validation and test set episodes are human annotated for correctness, as well as a 2,000-episode subset of train. See Appendix A.6.1 for correctness analysis of scale-generated episodes. Below, we analyze the characteristics and diversity of this dataset. 

**Characteristics** : As described earlier, **PARTNR** focuses on four task types: constraint-free, spatial, temporal, and heterogeneous. We show the distribution of these task types in the test split in Figure 4; validation split is similar. **PARTNR** evaluates collaboration along these axes both independently and jointly. Secondary characteristics of interest include dependent rearranges (e.g., _"Place them on the_ **same** _table"_ ) and multi-step rearrangement of the same object (e.g. _"Move the cup to the_ **sink** _, wash it, then place it in the_ **cabinet** _"_ ). 7% of tasks include dependent rearranges and 6% include multi-step rearrangement. Tasks average 4.7 propositions to be satisfied (indicative of number of steps required to complete tasks). For analysis of linguistic phenomena and more characteristics, see Appendix A.3. 

**Diversity** : The diversity of tasks in **PARTNR** is largely enabled by simulation-in-the-loop generation, which utilizes rich HSSD scenes, and the OVMM object set. Consequently, **PARTNR** tasks reference and require reasoning about 155 unique object types, 20 furniture classes and 13 room types. Note that each instruction, instantiated in each house, brings its own diversity. For example, _"move the laptop to the office table"_ , grounds _office_ and _table_ uniquely in each house, as well as different instances of _laptop_ in different instructions. Further discussion can be found in Appendix A.3. 

## **4 Experiments and Analysis** 

We investigate how state-of-the-art planning and perception methods handle natural language tasks in new environments and coordinate with unseen partners using **PARTNR** . Since **PARTNR** consists of diverse spatio-temporal tasks specified in language, we primarily use LLMs in our baselines for planning, and study variants in (1) zero-shot prompting, retrieval-augmented generation or fine-tuning, (2) centralized versus decentralized planning, (3) partially versus fully observed environment, (4) learned versus oracle low-level robot skills, and (5) privileged versus non-privileged perception. 

Our experiments are conducted in the Habitat 3.0 simulator (Puig et al., 2024) with a simulated Spot robot (BostonDynamics; Yokoyama et al., 2023). We adopt a two-layer hierarchical control architecture, similar to (Puig et al., 2024; Szot et al., 2021), as illustrated in Figure 5, for the robot and simulated human. At the high level, a planner selects skills from a predefined skill library (e.g., navigate, pick, place, open, close). We also use a textual world graph with a three-layer hierarchy rep- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0007-07.png)


**----- Start of picture text -----**<br>
Sim Env<br>Control Signal Habitat 3.0 Control Signal<br>Low-Level Low-Level<br>Skill Skill<br>Human Robot<br>Obs Obs<br>Low-Level Perception Low-Level<br>Skill Skill<br>World Graph<br>High-Level  High-Level<br>Policy (LLM) Policy (LLM)<br>Task<br>Description<br>**----- End of picture text -----**<br>


**Figure 5 Decentralized architecture.** The human and robot agents use a 2-layer hierarchical architecture, with high-level LLM planners that call low-level skills. Both agents build a world graph, updated using observations and actions. 

7 

resenting rooms, furniture, and movable objects. Each node in the graph stores a semantic category (e.g., kitchen, table or cup), 3D information (e.g., position or bounding box), and states (e.g., clean, powered on). See Appendix A.7 and Figure 10 for details. 

## **4.1 Baselines** 

We evaluate baselines along the following axes: 

1. **Variations of high-level planner:** 

   - Heuristic expert: This approach utilizes expert-designed heuristics and privileged information about the task, environment and evaluation function to pre-plan all steps for human and robot based on their capabilities. For instance, both agents might rearrange objects, but only humans perform cleaning, filling, and toggling on/off tasks. 

   - Zero-shot ReAct (ReAct): We use ReAct (Yao et al., 2023) with an API library of functions or _tools_ that enable the LLM to take actions. As observation, we provide the LLM with a concise, current world graph description plus a history of actions. The LLM uses this information to choose an action from [ExploreRoom, Navigate, OpenFurniture, CloseFurniture, PickObject, PlaceObject, Wait, Done] for the robot. See Appendix A.16 for prompts and Appendix A.9 for API details (human and robot). 

   - ReAct with Retrieval-Augmented Generation (ReAct-RAG): We also evaluate ReAct with RAG (Lewis et al., 2020) to investigate whether examples of planning on similar tasks improves the performance of ReAct. We construct a database of planning examples by collecting the successful traces from ReAct from a the 2,000 task training subset (see 3.3). During test time, the most relevant planning trace from the train dataset is selected based on sentence similarity and added to the LLM’s prompt (Pang et al., 2024; Madaan et al., 2024). 

   - Finetuned LLMs (Finetuned): We also investigate finetuning a smaller LLM (Llama3.1-8B) as our high-level planner, using successful traces from the ReAct baselines (Hsieh et al., 2023) that use Llama3.1-70B. Using the React-RAG dataset, we split every episode into a sequence of high-level planning actions, filtering for only actions that were executed successfully. For every action, we build an input containing the world-graph and history of actions, similar to ReAct (see Appendix A.10 for more details). We then finetune an LLM to predict the action from the ReAct episode given this input, using a low-rank adapter (Hu et al., 2021). This model has reduced latency and computational demands, suitable for real world deployment. 

All model generations are constrained to only output valid actions on observed objects using constrained generation (Geng et al., 2023). The constrained generation greatly reduces the hallucinations and ‘grammatical’ errors typical of LLMs. An episode is finished when both agents call Done or reach maximum simulation steps or LLM calls. Refer to Appendix A.9 for details. 

2. **Centralized versus decentralized planning:** To study the overhead of coordination in multi-agent **PARTNR** tasks, we compare centralized and decentralized planners. In centralized, a single LLM decides actions for both agents, with complete information about both agent’s states, effectively removing any need for coordination between the agents. In decentralized, each agent is controlled by a different LLM, and each LLM needs to reason about the other agent’s actions. 

3. **Partial versus full observability:** To evaluate if SoTA language models can explore new environments and identify task-relevant objects, we consider a partially observed setting where the planner knows the house’s layout but not the object locations, requiring exploration. This is in contrast to a fully observed setting, where all object locations are known in advance. 

4. **Learned versus oracle low-level robot skills:** We examine the impact of learned neural-network skills versus oracle skills (with privileged simulation information) on overall performance in **PARTNR** tasks. We create a library of learned skills for pick, place, navigate, open and close actions (Appendix A.8 provides more details) and compare performance with oracle skills. 

5. **Privileged versus non-privileged perception:** To study perception challenges such as inaccurate detection and approximate localization, we used a non-privileged world graph with modified ConceptGraphs (Gu et al., 2024), built from agents’ RGBD observations only. As agents explore and take actions, this world graph is updated using onboard sensing (details in Appendix A.7). In contrast, with privileged perception, this information is available from the simulation. 

8 

## **4.2 Results and Analysis** 

**Metrics.** We evaluate performance across different settings using four key metrics. First, the _simulation steps_ metric measures the number of steps required for agents to complete the task within the simulation environment, serving as an indicator of efficiency. Second, the _success rate_ reflects the completion of the task i.e. whether ‘all’ task constraints are satisfied. Given the complexity and long-horizon nature of **PARTNR** tasks, agents often partially complete the task. To account for this, we also report _percent complete_ , which quantifies the ratio of completed task ‘propositions’ (percent complete = 1 for successful tasks). Lastly, we assess the reasoning efficiency of the planners through the _planning cycles_ metric, which counts the number of high-level LLM calls each planner makes in the course of an episode. We cap the maximum planner calls at 50 in all experiments. 

## **4.2.1 Task Performance Analysis** 

Table 2 presents a comprehensive evaluation of the planning approaches defined in Section 4.1 using the _Llama3.1-70B-Instruct_ model (Dubey et al., 2024) for ReAct baselines, and a finetuned _Llama3.1-8B_ base model for the Finetuned baseline. Since **PARTNR** tasks are multi-agent, we also need a simulated human partner, which we control with a ReAct approach, using _Llama3.1-70B-Instruct_ . Our main findings are detailed below. 

**PARTNR tasks are challenging for LLM-based planners.** LLM-based baselines across all observability and controllability conditions perform worse than the privileged heuristic expert, due to errors in task tracking (not completing all steps, performing them in the wrong order, or undoing completed steps), and semantic errors (placing objects on the wrong furniture, or moving the wrong object), indicating a gap in LLM task planning. 

**LLMs struggle with coordination in decentralized settings.** Decentralized ReAct baselines which do not have privileged access to partner’s intent are significantly slower at task completion than centralized ReAct (3295 steps with decentralized-partial in row(e) versus 2298 with centralized-partial in row(d)). This shows that reasoning about the partner e.g., knowing or inferring partner’s intent can improve task efficiency in **PARTNR** tasks, and current SoTA LLMs perform poorly at this. Moreover, decentralized ReAct with two agents is even _slower_ than ReAct with a single-agent (3295 steps with multi-agent in row(e) versus 2519 with single-agent in row(a)), indicating that LLMs suffer from a significant coordination “burden". This co-ordination burden is further highlighted in our analysis on _extraneous effort_ in Section 4.2.2, where we find that agents end up repeating parts of the task or performing irrelevant actions with much higher frequency in decentralized settings. 

**LLMs are unable to recover from learned skill failures.** When replacing oracle skills with learned skills, the success rate decreases from 0.73 to 0.57 (row(e) vs. row (h)). This decline can be attributed to the higher failure rate and increased number of simulation steps required by learned skills compared to privileged oracle skills. The LLMs struggle to recover from skill errors like failing to pick up an object or performing incomplete exploration, resulting in a lower success rate. Future research could investigate training large models with low-level skills in the loop, enabling them to learn recovery and replanning strategies in the face of such failures. 

**LLMs exhibit a high degree of sensitivity to errors in perception.** When we replace privileged perception with off-the-shelf perception modules, success rate significantly declines (from 0.57 with a privileged, partial world graph in row(h) to 0.30 with Concept-Graphs (Gu et al., 2024) in row(i)). LLMs rely heavily on accurate world descriptions provided by the world graph and struggle to correct errors such as misclassification (e.g., shelves misidentified as tables) or incorrect room assignments (e.g., a table in the living room mislabeled as being in the bedroom). Multi-modal models like VLMs might be stronger at recovering from such failures, which we leave for future work. 

**Finetuned 8B model performs on par with a ReAct with a 70B model, while being 8.6x faster.** We find that the finetuned planner with a small 8B model performs on par with ReAct, which uses a much larger 70B model (a 0.73 success rate with the 70B model in row(e), versus 0.70 with the finetuned 8B model in row(g)). At the same time, we find that the finetuned model is 8.6 times faster at inference. This indicates that the finetuning effectively distills task-relevant information from the training set and generalizes to new test tasks. When 

9 

||**Method**<br>**Controllability**<br>**Skills**<br>**Observability**<br>**Sim**<br>**Steps**_↓_<br>**Success**<br>**Rate**_↑_<br>**Percent**<br>**Complete**_↑_<br>**Planning**<br>**Cycles**_↓_|
|---|---|
||(a) ReAct-Single<br>Single Agent<br>Oracle<br>Partial<br>2519.02 _±_ 57.48<br>0.73 _±_ 0.01<br>0.85 _±_ 0.01<br>18.68 _±_ 0.33|
||(b) Heuristic-Expert<br>Centralized<br>Oracle<br>Full<br>1260.88 _±_ 26.97<br>0.84 _±_ 0.01<br>0.94 _±_ 0.01<br>N/A|
||(c) ReAct<br>Centralized<br>Oracle<br>Full<br>1347.43 _±_ 33.80<br>0.74 _±_ 0.01<br>0.88 _±_ 0.01<br>17.49 _±_ 0.34|
||(d) ReAct<br>Centralized<br>Oracle<br>Partial<br>2298.13 _±_ 61.39<br>0.74 _±_ 0.01<br>0.85 _±_ 0.01<br>20.73 _±_ 0.51<br>(e) ReAct<br>Decentralized<br>Oracle<br>Partial<br>3295.20 _±_ 76.27<br>0.73 _±_ 0.01<br>0.86 _±_ 0.01<br>15.24 _±_ 0.31<br>(f) ReAct + RAG<br>Decentralized<br>Oracle<br>Partial<br>3467.47 _±_ 82.39<br>0.71 _±_ 0.01<br>0.84 _±_ 0.01<br>14.75 _±_ 0.31<br>(g) Finetuned<br>Decentralized<br>Oracle<br>Partial<br>3228.96 _±_ 75.14<br>0.70 _±_ 0.01<br>0.84 _±_ 0.01<br>12.85 _±_ 0.24|
||(h) ReAct<br>Decentralized<br>Learned<br>Partial<br>6494.88 _±_ 181.52<br>0.57 _±_ 0.02<br>0.76 _±_ 0.01<br>22.72 _±_ 0.58<br>(i) ReAct<br>Decentralized<br>Learned<br>ConceptGraph<br>12490.80 _±_ 208.90<br>0.30 _±_ 0.01<br>0.56 _±_ 0.01<br>23.84 _±_ 0.45|



**Table 2 Analysis of planner baselines in various settings.** We compare performance using simulation steps, success rate and percent complete on the tasks, and the average planning cycles used by the baselines (described in Section 3). 

deployed with humans-in-the-loop, the finetuned model takes fewer steps and offloads more sub-tasks than the 70B model (see Table 3). 

## **4.2.2 Analysis of Collaborative Behavior and Efficiency** 

Our analysis in Table 2 revealed challenges in LLM collaboration, prompting a deeper investigation into specific collaborative behaviors, explained below and detailed in Appendix A.12 and Table 12. 

**Robots offload up to 60% of tasks.** We evaluate the robot’s ability to offload tasks from the human, measuring the ratio of sub-tasks performed by the robot to the total sub-tasks in successful **PARTNR** tasks. Despite similar success rates between single- and multi-agent (0.73 vs. 0.74), the robot offloads about 60% of sub-tasks in decentralized multi-agent, reducing human effort (Table 12). 

**Decentralized agents are prone to performing extraneous tasks.** The agents sometimes end up performing sub-tasks that are not useful for the task such as rearranging an object that is not required by the task or repeating a sub-task already performed by the other agent. To capture such extraneous agent effort, we measure the portion of agent actions that did not increase the percent complete metric i.e., did not contribute to task progress, over the total number of successful agent actions in an episode. We find a 300% increase in extraneous effort in decentralized multi-agent settings compared to single-agent (Table 12), indicating a significant coordination burden. 

**Temporal and heterogeneous tasks are challenging for LLMs.** LLMs struggle in temporal and heterogeneous tasks. Task success drops by 27% for temporal tasks and 20% for heterogeneous tasks compared to constraint-free tasks for ReAct (Table 13). This highlights the limitations of LLMs in reasoning about agent capabilities and following strict ordering constraints. 

## **4.3 Human-in-the-loop Evaluation** 

We build on the human-in-the-loop infrastructure from Habitat 3.0 (Puig et al., 2024) and adapt it to a serverclient architecture, with the server hosted on AWS capable of supporting multiple clients (see Appendix A.13). This allows us to run at-scale evaluation of our tasks with 129 non-expert human participants. We collect single-user and multi-user data on 1000 tasks from the validation and test set using this tool. In the single-user setting, a single participant completes the whole task, by driving the human agent in the simulator via keyboard/mouse controls Figure 14 in appendix shows our HITL interface. In multi-user, two participants complete the task together by controlling a human and a robot agent. The goal of these experiments is to study multi-user dynamics at **PARTNR** tasks, and see if multiple humans collaborating are more efficient than single human. Finally, we run a human-AI experiment where a human participant collaborates with a robot controlled by an LLM (using the ReAct and Finetuned models from Section 4.1). This experiment aims to evaluate LLM-controlled agents at collaborating with unseen, real humans. Table 3 shows the success rate (SR) and percent complete (PC) of tasks from the validation set in a single-user, multi-user, human-ReAct and human-Finetuned setting. Additionally, we measure the number of steps taken by each approach to complete 

10 

|Method|Success<br>Rate _↑_|Percent<br>Complete _↑_|Sim<br>Steps _↓_|Task<br>Ofoading _↑_|Exploration<br>Efciency _↓_|Extraneous<br>Efort _↓_|
|---|---|---|---|---|---|---|
|Single-user|0.93 _±_ 0.01|0.96 _±_ 0.00|3046.99 _±_ 80.79|N/A|2459.22 _±_ 26.75|0.09 _±_ 0.01|
|Multi-user|0.93 _±_ 0.01|0.96 _±_ 0.00|2369.55 _±_ 49.33|0.59 _±_ 0.01|1762.47 _±_ 13.99|0.15 _±_ 0.01|
|Human-ReAct|0.91 _±_ 0.01|0.96 _±_ 0.02|4267.71 _±_ 83.40|0.16 _±_ 0.01|2624.39 _±_ 26.05|0.12 _±_ 0.01|
|Human-Finetuned|0.92 _±_ 0.01|0.96 _±_ 0.00|3443.33 _±_ 61.46|0.26 _±_ 0.01|2164.94 _±_ 21.31|0.13 _±_ 0.01|



**Table 3 Human-in-the-Loop Evaluation.** We evaluate the performance of a 2-person human team and human-LLM teams, comparing them to solo human performance on **PARTNR** tasks using metrics described in Section 4.1. Additional results and analysis in Appendix A.13. 

the task, and the ratio of work completed by the robot i.e., task offloading. We also measure exploration efficiency in human-in-the-loop, by measuring the steps taken to pick the first object, and extraneous effort, indicating actions that were not useful for task completion. These results are summarized in Table 3. Some key findings are below (more results and analysis in A.13.4): 

**Humans are significantly better than LLMs at PARTNR tasks.** In both single and multiple human environments, the success rate achieved is 0.93 on the **PARTNR** benchmark. In contrast, the ReAct model without any privileged information, achieves a significantly lower success rate of 0.30 (row (i) of Table 2). This highlights a significant gap in the performance of LLMs in planning tasks. Note that the LLM baselines like ReAct and Finetuned achieve a success rate of 0.92 and 0.91 when evaluated with _real_ humans (Table 3), because humans are able to adapt to LLM mistakes. On the other hand, the simulated human in Table 2 is an LLM, which is unable to recover from partner mistakes. 

**Finetuned LLMs perform better than ReAct when coordinating with real humans.** When deployed with real humans-in-the-loop, the finetuned model is faster than ReAct at task completion (3443 steps with finetuned versus 4267 with ReAct). It is also able to offload more tasks from humans than ReAct (26% with finetuned as compared to 16% with ReAct). This reflects that smaller models with faster inference can improve human experience in real-world deployment. 

**LLMs struggle at coordination, hampering human performance.** Despite the Finetuned being faster than ReAct when collaborating with humans, both approaches are _slower_ than the human doing the task alone. In contrast, two humans working together complete the task faster than a single human (2369 steps vs. 3046 with multi- and single-user respectively). This result is in line with the automated evaluation we observed in Table 1, where multi-agent LLMs are also _slower_ than a single-agent LLM. This result further reinforces that LLMs suffer at coordination; while humans are able to coordinate and divide tasks between each other, decentralized LLMs are unable to do so. 

**LLMs are able to offload tasks from humans.** Despite the aforementioned increase in the number of steps for task completion, robots guided by the finetuned model successfully offload 26% of tasks from humans. This indicates that LLMs can still offer assistance when collaborating with real human partners. Nonetheless, there remains significant potential for improvement. 

## **5 Conclusion** 

We present **PARTNR** , a benchmark for reasoning and planning in multi-agent embodied tasks, featuring 100,000 natural language tasks instantiated in 60 simulated, multi-room houses with 5,819 unique objects. We use a semi-automated LLM-powered pipeline for large-scale instruction and evaluation function generation that uses simulation-in-the-loop grounding. **PARTNR** exhibits characteristics of everyday tasks, such as temporal and spatial constraints, and allows systematic evaluation of planning approaches. We find a significant gap between SoTA LLMs and human-level performance at **PARTNR** tasks. While our best LLM baseline only succeeds at 30% of tasks with no privileged information, humans are able to solve 93% of the tasks. Moreover, LLMs face challenges in coordinating with both LLM-based agents and real human partners. Human-in-the-loop evaluations, involving real humans collaborating with LLM-guided robots, reveal that LLM-guided partners decrease human efficiency compared to working solo. This suggests that LLM-based agents require significant improvements to become effective collaborative partners in embodied tasks. **PARTNR** serves as a challenging benchmark that highlights the substantial limitations of current models. 

11 

## **References** 

- Saaket Agashe, Yue Fan, Anthony Reyna, and Xin Eric Wang. Evaluating multi-agent coordination abilities in large language models. _arXiv_ , 2023. 

- Christopher Agia, Krishna Murthy Jatavallabhula, Mohamed Khodeir, Ondrej Miksik, Vibhav Vineet, Mustafa Mukadam, Liam Paull, and Florian Shkurti. Taskography: Evaluating robot task planning over large 3d scene graphs. In _CoRL_ , 2022. 

- Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. Do as i can and not as i say: Grounding language in robotic affordances. In _CoRL_ , 2022. 

- Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In _CVPR_ , 2018. 

- BostonDynamics. Spot robot. https://www.bostondynamics.com/products/spot. Accessed: 2024-10-01. 

- Micah Carroll, Rohin Shah, Mark K Ho, Tom Griffiths, Sanjit Seshia, Pieter Abbeel, and Anca Dragan. On the utility of learning about humans for human-ai coordination. In _NeurIPS_ , 2019. 

- Howard Chen, Alane Suhr, Dipendra Misra, Noah Snavely, and Yoav Artzi. Touchdown: Natural language navigation and spatial reasoning in visual street environments. In _CVPR_ , 2019. 

- Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra. Embodied question answering. In _CVPR_ , 2018. 

- Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. In _ICML_ , 2023. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv_ , 2024. 

- Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert West. Grammar-constrained decoding for structured NLP tasks without finetuning. In _EMNLP_ , 2023. 

- Malik Ghallab, Craig Knoblock, David Wilkins, Anthony Barrett, Dave Christianson, Marc Friedman, Chung Kwok, Keith Golden, Scott Penberthy, David Smith, Ying Sun, and Daniel Weld. Pddl - the planning domain definition language. _Technical Report_ , 1998. 

- Ran Gong, Jiangyong Huang, Yizhou Zhao, Haoran Geng, Xiaofeng Gao, Qingyang Wu, Wensi Ai, Ziheng Zhou, Demetri Terzopoulos, Song-Chun Zhu, et al. Arnold: A benchmark for language-grounded task learning with continuous states in realistic 3d scenes. In _ICCV_ , 2023. 

- Qiao Gu, Alihusein Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, Chuang Gan, Celso Miguel de Melo, Joshua B. Tenenbaum, Antonio Torralba, Florian Shkurti, and Liam Paull. Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. In _ICRA_ , 2024. 

- Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. _arXiv_ , 2023. 

- Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In _ICLR_ , 2021. 

- Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In _ICML_ , 2022. 

- Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Tomas Jackson, Noah Brown, Linda Luu, Sergey Levine, Karol 

12 

Hausman, and brian ichter. Inner monologue: Embodied reasoning through planning with language models. In _CoRL_ , 2023. 

- Unnat Jain, Luca Weihs, Eric Kolve, Mohammad Rastegari, Svetlana Lazebnik, Ali Farhadi, Alexander G. Schwing, and Aniruddha Kembhavi. Two body problem: Collaborative visual task completion. In _CVPR_ , 2019. first two authors contributed equally. 

- Unnat Jain, Luca Weihs, Eric Kolve, Ali Farhadi, Svetlana Lazebnik, Aniruddha Kembhavi, and Alexander G. Schwing. A cordial sync: Going beyond marginal policies for multi-agent embodied tasks. In _ECCV_ , 2020. 

- Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J Davison. Rlbench: The robot learning benchmark & learning environment. _IEEE Robotics and Automation Letters_ , 2020. 

- Pushkal Katara, Zhou Xian, and Katerina Fragkiadaki. Gen2sim: Scaling up robot learning in simulation with generative models. _arXiv_ , 2023. 

- Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg, Eric Undersander, Angel X. Chang, and Manolis Savva. Habitat synthetic scenes dataset (hssd-200): An analysis of 3d scene scale and realism tradeoffs for objectgoal navigation. In _CVPR_ , 2024. 

- Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-andlanguage navigation in continuous environments. In _ECCV_ , 2020. 

- Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge. Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. In _EMNLP_ , 2020. 

- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In _NeurIPS_ , 2020. 

- Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, et al. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. In _CoRL_ , 2023a. 

- Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for "mind" exploration of large language model society. In _NeurIPS_ , 2023b. 

- Shuang Li, Xavier Puig, Chris Paxton, Yilun Du, Clinton Wang, Linxi Fan, Tao Chen, De-An Huang, Ekin Akyürek, Anima Anandkumar, et al. Pre-trained language models for interactive decision-making. In _NeurIPS_ , 2022. 

- Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. In _NeurIPS_ , 2024. 

- Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In _CVPR_ , 2024. 

- Zhao Mandi, Shreeya Jain, and Shuran Song. Roco: Dialectic multi-robot collaboration with large language models. In _ICRA_ , 2024. 

- Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. In _RSS_ , 2024. 

- Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. _arXiv_ , 2024. 

- Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In _ACM Symposium on User Interface Software and Technology_ , 2023. 

- Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and Antonio Torralba. Virtualhome: Simulating household activities via programs. In _CVPR_ , 2018. 

- Xavier Puig, Tianmin Shu, Shuang Li, Zilin Wang, Yuan-Hong Liao, Joshua B. Tenenbaum, Sanja Fidler, and Antonio Torralba. Watch-and-help: A challenge for social perception and human-{ai} collaboration. In _ICLR_ , 2021. 

- Xavier Puig, Tianmin Shu, Joshua B Tenenbaum, and Antonio Torralba. Nopa: Neurally-guided online probabilistic assistance for building socially intelligent home assistants. In _ICRA_ , 2023. 

13 

- Xavier Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Tsung-Yen Yang, Ruslan Partsey, Ruta Desai, Alexander William Clegg, Michal Hlavac, So Yeon Min, Vladimír Vondruš, Theophile Gervet, Vincent-Pierre Berges, John M. Turner, Oleksandr Maksymets, Zsolt Kira, Mrinal Kalakrishnan, Jitendra Malik, Devendra Singh Chaplot, Unnat Jain, Dhruv Batra, Akshara Rai, and Roozbeh Mottaghi. Habitat 3.0: A co-habitat for humans, avatars and robots. In _ICLR_ , 2024. 

- Team PyTorch. Accelerating generative ai with pytorch ii: Gpt, fast. https://pytorch.org/blog/ accelerating-generative-ai-2/, 2023. 

- Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf. Sayplan: Grounding large language models using 3d scene graphs for scalable robot task planning. In _CoRL_ , 2023. 

- Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In _EMNLP_ , 2019. 

- Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. _arXiv_ , 2023. 

- Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In _NeurIPS_ , 2023. 

- Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. In _CVPR_ , 2020. 

- Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In _ICLR_ , 2021. 

- Sanjana Srivastava, Chengshu Li, Michael Lingelbach, Roberto Martín-Martín, Fei Xia, Kent Elliott Vainio, Zheng Lian, Cem Gokmen, Shyamal Buch, Karen Liu, Silvio Savarese, Hyowon Gweon, Jiajun Wu, and Li Fei-Fei. Behavior: Benchmark for everyday household activities in virtual, interactive, and ecological environments. In _CoRL_ , 2022. 

- Joseph Suarez, Yilun Du, Phillip Isola, and Igor Mordatch. Neural mmo: A massively multiagent game environment for training and evaluating intelligent agents. _arXiv_ , 2019. 

- Andrew Szot, Alexander Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Singh Chaplot, Oleksandr Maksymets, et al. Habitat 2.0: Training home assistants to rearrange their habitat. In _NeurIPS_ , 2021. 

- Andrew Szot, Unnat Jain, Dhruv Batra, Zsolt Kira, Ruta Desai, and Akshara Rai. Adaptive coordination in social embodied rearrangement. In _ICML_ , 2023. 

- Andrew Szot, Max Schwarzer, Harsh Agrawal, Bogdan Mazoure, Walter Talbott, Katherine Metcalf, Natalie Mackraz, Devon Hjelm, and Alexander Toshev. Large language models as generalizable policies for embodied tasks. In _ICLR_ , 2024. 

- Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Katerina Fragkiadaki, Zackory Erickson, David Held, and Chuang Gan. Robogen: Towards unleashing infinite data for automated robot learning via generative simulation. In _ICML_ , 2024. 

- Abdelrhman Werby, Chenguang Huang, Martin Büchner, Abhinav Valada, and Wolfram Burgard. Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation. In _RSS_ , 2024. 

- Zhou Xian, Theophile Gervet, Zhenjia Xu, Yi-Ling Qiao, and Tsun-Hsuan Wang. Towards a foundation model for generalist robots: Diverse skill learning at scale via automated task and scene generation. _arXiv_ , 2023. 

- Jiannan Xiang, Tianhua Tao, Yi Gu, Tianmin Shu, Zirui Wang, Zichao Yang, and Zhiting Hu. Language models meet world models: Embodied experiences enhance language models. In _NeurIPS_ , 2023. 

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In _ICLR_ , 2023. 

- Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin Wang, Mukul Khanna, Theophile Gervet, TsungYen Yang, Vidhi Jain, Alexander William Clegg, John Turner, et al. Homerobot: Open-vocabulary mobile manipulation. In _CoRL_ , 2023. 

14 

- Naoki Yokoyama, Alex Clegg, Joanne Truong, Eric Undersander, Tsung-Yen Yang, Sergio Arnaud, Sehoon Ha, Dhruv Batra, and Akshara Rai. Asc: Adaptive skill coordination for robotic mobile manipulation. _IEEE Robotics and Automation Letters_ , 2023. 

- Andy Zeng, Maria Attarian, Krzysztof Marcin Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael S Ryoo, Vikas Sindhwani, Johnny Lee, et al. Socratic models: Composing zero-shot multimodal reasoning with language. In _ICLR_ , 2022. 

- Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B Tenenbaum, Tianmin Shu, and Chuang Gan. Building cooperative embodied agents modularly with large language models. _ICLR_ , 2024a. 

- Hongxin Zhang, Zeyuan Wang, Qiushi Lyu, Zheyuan Zhang, Sunli Chen, Tianmin Shu, Yilun Du, and Chuang Gan. Combo: Compositional world models for embodied multi-agent cooperation. _arXiv_ , 2024b. 

- Zirui Zhao, Wee Sun Lee, and David Hsu. Large language models as commonsense knowledge for large-scale task planning. In _NeurIPS_ , 2024. 

- Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, et al. Sotopia: Interactive evaluation for social intelligence in language agents. In _ICLR_ , 2024. 

- Hao Zhu, Raghav Kapoor, So Yeon Min, Winson Han, Jiatai Li, Kaiwen Geng, Graham Neubig, Yonatan Bisk, Aniruddha Kembhavi, and Luca Weihs. Excalibur: Encouraging and evaluating embodied exploration. In _CVPR_ , 2023. 

15 

## **A Appendix** 

We present **PARTNR** , a benchmark for reasoning and planning in multi-agent embodied tasks, featuring 100,000 natural language, everyday tasks. We show at-scale generation of these tasks using LLMs with simulation in the loop for grounding and human in the loop for filtering. We also evaluate several LLM-based planning models on these tasks and highlight avenues for future work. This appendix provides additional details on these contributions and is organized as follows: 

- A.1 Code and **PARTNR** benchmark open-sourcing 

- A.2 HSSD Scene Annotations 

- A.3 Details and additional analysis on the **PARTNR** dataset 

- A.4 Simulation features and prompts for the **PARTNR** task generation 

- A.5 The **PARTNR** evaluation system 

- A.6 Human annotation and quality assessment for the **PARTNR** tasks and evaluation functions 

- A.7 World graph for perception in LLM agents 

- A.8 Learned low-level robot skills 

- A.9 Implementation details for ReAct agents 

- A.10 Details on finetuning LLM agents for the **PARTNR** tasks 

- A.11 Additional results 

- A.12 Analysis of collaborative behavior and efficiency of LLM agents 

- A.13 Human-in-the-loop (HITL) system and evaluation for the **PARTNR** tasks and LLM agents 

- A.14 Prompts for task and evaluation generation 

- A.16 Prompts for planner baselines 

## **A.1 Open-sourcing PARTNR Dataset and Codebase** 

Accompanying this paper, we have released the code and data necessary to reproduce our experiments. Released code includes our **PARTNR** benchmark tasks, metrics, baseline oracle skills, large planning model framework, and dataset generation utilities. Released data includes extensions of the Habitat Synthetic Scenes Dataset (HSSD) (Khanna et al., 2024), generated benchmark task episodes, and model weights for our trained neural network skills and fine-tuned large planning model. 

The publicly released codebase accompanying **PARTNR** depends on the most recent version of the AI Habitat platform (habitat-lab and habitat-sim (v0.3.2)) (Puig et al., 2024) which it extends to define collaboration tasks and skills. 

## **A.2 HSSD Scene Annotations** 

In order to model the space of rich indoor collaboration tasks we propose with **PARTNR** , we extended HSSD with additional asset authoring and annotation. To enable more realistic indoor object manipulation, we added articulated 3D furniture models such as drawers, cabinets, and appliances. These models were converted from rigid source assets in HSSD and swapped into the scenes. We prepared 60 scenes divided into train, val, and test splits to support our experiments. Each scene is manually adjusted by a human to ensure simulation robustness and minimize potential issues. Furniture is annotated with a set of Receptacles (surfaces which support small object placement such as shelves and drawers) and can be opened and closed by the agents. Receptacles are further filtered contextually in each scene to ensure that the active set is accessible to the agents. Additional annotations include point or marker sets for each furniture, region annotations, and semantic classification of objects. The marker sets indicate either a spread of surface points (for distance/occlusion checking) or the location of key points of interest such as faucets (for cleaning/filling) and handles (for opening/closing) necessary for low-level skill training and oracle skill execution. Region annotations included per-scene region volumes (e.g., kitchen, living-room, bedroom, etc.) for checking and specifying the location of objects and furniture. Semantic annotations indicate the object category or class (e.g. table, chair, cup, toy) to support open language prompt grounding and semantically guided task generation. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0017-00.png)


**----- Start of picture text -----**<br>
Task Instruction<br>“Let’s tidy up the family room. The toys go in the toy box.<br>After that, set the plants on the shelf next to each other.”<br>**----- End of picture text -----**<br>


**Task Instruction** _“We’re ready for dinner. Can you bring two cups to the kitchen? Let’s fill them and take them to the dining table.”_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0017-02.png)


**----- Start of picture text -----**<br>
Evaluation Function Evaluation Function<br>Propositions:<br>0  is_inside(["toy_fire_truck_0"], ["toy_box_0"])<br>1  is_inside(["toy_food_0"], ["toy_box_0"]) Of the n cups in the scene…<br>2  is_on_top(["plant_0"], ["shelf_0", "shelf_1"]) Propositions: Bring two<br>3  is_on_top(["plant_1"], ["shelf_0", "shelf_1"]) 0  is_in_room(["cup_0", "cup_1",…, "cup_n"], ["kitchen"], n=2)<br>4  is_next_to(["plant_0"], ["plant_1"]) 1  is_filled(["cup_0", "cup_1",…, "cup_n"], n=2)<br>Verify “is_next_to” when 2  is_on_top(["cup_0", "cup_1",…, "cup_n"], ["table_2"], n=2)<br>Dependencies: the plants are placed<br>0  WhileSatisfied([4], depends_on=[2,3]) Use the same cups throughout<br>Constraints:<br>The toys must be rearranged first 0  SameArgConstraint([0,1,2], arg=”object")<br>Constraints: 1  TerminalSatisfactionConstraint([1,2])<br>0  TemporalConstraint(<br>       [(0,2), (1,2), (0,3), Proposition 0 is temporary<br>        (1,3), (0,4), (1,4)] All  placements are terminal<br>   )<br>1  TerminalSatisfactionConstraint([0,1,2,3,4])<br>**----- End of picture text -----**<br>


**Figure 6 Additional task and evaluation examples.** Expands on Figure 3 to show a task and evaluation containing subset counts and dependent rearranges (right). 

## **A.3 Dataset Details and Additional Analysis** 

We expand on the details and analysis of the **PARTNR** dataset of Section 3.3, including analysis of linguistic phenomena, secondary task characteristics, and the distribution of sampled entities in generated tasks. See Figure 6 for additional task and evaluation examples. 

**Linguistic Phenomena.** In Table 4, we present an analysis of linguistic phenomena manually annotated over 50 random episodes (similar to Ku et al. (2020) and Chen et al. (2019)). We analyze the following phenomena: 

- **Class Reference:** refers to a semantic class of objects, furniture, or rooms in the scene. These references typically, but not necessarily, follow from the classes defined in OVMM. 

- **Instance Reference:** refers to a unique object, furniture, or room; this disambiguation between entities of a class is typically achieved by visual description. 

- **Co-Reference:** an expression that refers to an entity defined elsewhere in the instruction. 

- **Passive Voice:** the instruction is phrased such that the object is receiving the rearrangement or state change; the request is typically asked instead of commanded. 

- **Active Voice:** the instruction is phrased such that the rearrangement or state change is to be completed with an object; the request is typically commanded instead of asked. 

- **High-Level Goal Spec:** this sets the operating context for the task before the particulars of rearrangement or state change are specified. 

- **Agentic Reference:** a reference to one of the agents performing the task. Typically used to incite a suggested task division between the human and robot. 

We observe that **PARTNR** tasks have a high rate of entity class references, such as _the table_ , (6.38/episode), and common occurrences of instance references and co-references. This signals a need for capable natural language understanding, scene grounding, and co-reference resolution. Tasks and sub-tasks are predominantly issued using active voice (92%), but 14% of tasks include at least one occurrence of passive voice. Half of tasks involve a high-level goal specification, which commonly serves to reduce the search space. For example, a task starting with _Let’s clean up all the toys in the playroom_ constrains object search to that room and softly constrains the placement of those objects to locations in which toys would commonly be stored. Finally, agentic references are present in 14% of episodes. 

**Secondary Task Characteristics.** In Table 5, We present an analysis of secondary task characteristics present in **PARTNR** as derived automatically from evaluation functions of all episodes in our dataset. We find rare but present occurrences of subset counts, where agents must reason about manipulating a subset of a set of 

17 

|Linguistic Phenomenon|_p_|_µ_|Example|
|---|---|---|---|
|Class Reference|100|6.38|_The table_|
|Instance Reference|12|0.14|_The cofee table_|
|Co-Reference|50|0.64|_That, Those, it, ..._|
|Passive Voice|14|0.18|_Can you bring me...?_|
|Active Voice|92|2.40|_Set it on the stool._|
|High-Level Goal Spec|50|0.50|_Let’s tidy up. Move..._|
|Agentic Reference|14|0.16|_Do... While I..._|



**Table 4 Analysis of linguistic phenomena in the PARTNR dataset.** _p_ is the % of instructions that contain the phenomenon while _µ_ is the average number of times the phenomenon occurs within each instruction. A random sample of 50 Test episodes were included in manual annotation. 

|Secondary Characteristics|_p_|_µ_|Example|
|---|---|---|---|
|Subset Count|1|0.01|_Bring two cups..._|
|Resolvable Ambiguity|68|1.68|_Move the pants to any chair._|
|Dependent Rearrange|7|0.09|_Place them on the same table._|
|Multi-Step Rearrange|6|0.11|_Cup to sink. Cup to cabinet._|



**Table 5 Analysis of secondary task characteristics in the PARTNR dataset.** _p_ is the % of instructions that contain the phenomenon while _µ_ is the average number of times the phenomenon occurs within each instruction. All episodes in Test were included using automatic annotation. 

objects (e.g. _bring two cups..._ when more than two cups exist in the scene). We find that every episode contains at least one occurrence of resolvable ambiguity, where an instruction makes an object/furniture/room reference that may be resolved by more than one entity instance. Dependent rearrangements build on this task characteristic; when multiple objects/furniture/rooms can satisfy a sub-task, a task may require that same entity to solve another sub-task. An example of this is the same table being used to solve the placement of both a spoon and a bowl ( _"bring the spoon and bowl to the same table in the living room."_ ). This occurs in 31% of episodes and the resulting dynamic dependency is a challenging aspect of collaboration. Finally, 6% of tasks include multi-step rearrangement of the same entity. For example, moving a cup to the sink and then to the cabinet after it is washed. Tasks with multi-step rearrangement have two such rearrangements on average. 

**Distribution of Entities in PARTNR Tasks.** In Figure 7, we examine the distribution of task-relevant objects, furniture, and rooms. We define task-relevant objects to be objects requiring rearrangement or a state change, task-relevant furniture to be target furniture for rearrangement, and task-relevant rooms to be target rooms for rearrangement. Object categories are shown in Figure 7a, furniture categories are shown in Figure 7b, and room categories are shown in Figure 7c. The **PARTNR** dataset contains a long tail of semantic object categories, and within those categories, a wide variety of objects. Thus, for agents to perform well in **PARTNR** tasks, they must display visual reasoning and collaboration behaviors that generalize across the semantic particulars of a task. The skew of distributions in Figure 7b and Figure 7c can be understood by the relative occurrences of rooms and objects in HSSD scenes, e.g. there are more tables than couches on average. 

## **A.4 Simulation-in-the-loop Large-Scale Task Generation** 

In this section, we describe in detail the simulation-in-the-loop task generation pipeline. We follow a 4-step generation pipeline: 

1. **Simulation-in-the-loop small-scale free-form LLM generation:** 

In **PARTNR** , we initiate the process by setting up a simulation environment using the Habitat 3.0 simulator, populated with the HSSD dataset which includes 60 unique houses and 5,819 OVMM objects. This simulated environment is parsed to identify a list of rooms and available furniture. This information, along with a list of all available objects, is then passed to a Language Model (LLM). The LLM utilizes 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0019-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0019-01.png)


**----- Start of picture text -----**<br>
(a) Objects<br>(b) Furniture (c) Rooms<br>**----- End of picture text -----**<br>


**Figure 7 The distribution of task-relevant object categories, furniture categories, and room categories in the PARTNR dataset.** The object distribution is truncated to the 50 most common entities, down from 155 total object categories. The count of entities is derived by the total number of occurrences of that entity in the Test split evaluation data. 

this data to generate free-form, viable tasks within the scene, accompanied by an initial scene state description. For instance, if the task is "Clear dishes from the living room," the LLM would generate an initial scene depicting multiple dishes in the living room. To add complexity, additional objects are introduced to create clutter. 

## 2. **Simulation-in-the-loop filtering and annotation.** 

The tasks and initial states generated by the LLM are instantiated within the simulator. At this stage, tasks that are infeasible due to the layout of the house or the capabilities of the simulator are filtered out. For example, a task like "Clear dishes from the living room" would be discarded if the simulated house lacks a living room. Similarly, tasks requiring unsupported actions, such as "folding," are also filtered out. This filtering process is crucial as it significantly reduces the number of unrealistic or impossible tasks, although it also tends to limit the diversity of the generated instructions. The full list of allowed actions in the simulator are: [open, close, power on, power off, move, clean, fill, pour, pick, place] and related actions. Next, we use manual annotation to diversity of the resulting tasks. To counter the limited diversity resulting from automated filtering, we manually annotate the tasks to ensure a balanced distribution of various types of tasks, such as constraint-free, spatial, temporal, and heterogeneous tasks. This manual intervention involves modifying the instructions to incorporate different objects and settings, resulting in a curated set of 1,000 human-annotated and simulation-verified tasks. This step is essential for maintaining quality and diversity but is impractical for scaling up due to its labor-intensive nature. 

## 3. **Large-scale generation.** 

To scale the generation of diverse tasks without extensive manual effort, we leverage the 1,000 humanannotated instructions as examples in the LLM prompts. By providing the LLM with both a house description and an example task, we instruct it to adapt the task to fit the new setting, such as changing "Clear all dishes from the living room" to "Clear all toys from the bedroom." This approach helps preserve the diversity of the tasks while enhancing the likelihood of successful instantiation in the simulator. 

19 

## 4. **Human-in-the-loop filtering.** 

The scaled tasks undergo a final human-in-the-loop filtering where physically infeasible instructions are eliminated, ensuring the practicality and realism of the tasks. This ensures that tasks such as "Move 4 cups to the dining table", where there are only 2 cups in the scene are removed. Or "First move the cup from the kitchen to the living room, then place a jug in the kitchen, next to the cup", which consists of a physically infeasible instruction. 

## **A.4.1 Habitat 3.0 and HSSD Extension** 

We generate **PARTNR** using modified HSSD scenes (Khanna et al., 2024) and the Habitat 3.0 (Puig et al., 2024) simulator due to its humanoid simulation capabilities and availability of features which support modeling of collaborative tasks as discussed in Appendix A.1. 

Our extensions to the Habitat platform include a set of features targeting: object state manipulation (e.g., clean/dirty, powered on/off and filled/un-filled), evaluation of object relative relationships (e.g., next-to, above, within, on-top, on-floor, in-region, etc), and procedural clutter generation utilities enabling generation of valid initial scene contents pre-conditioned on the output of LLM-generated requirements from open-language prompts (see Section 3). For example, in order to evaluate the task of rearranging a tea set from furniture A to furniture B, we must first generate a scene with both types of furniture in accessible locations and a tea set already sitting on or inside of furniture A. 

## **A.5 The PARTNR Evaluation System** 

In this section, we formalize the components of the evaluation system, define the resulting metrics, and present the details and prompts used for LLM-based evaluation generation. 

## **A.5.1 Evaluation Predicates** 

We use logical predicates to query the state of objects, furniture, and rooms at the current timestep in the simulator. The evaluation system operates on the resulting binary state values. The details of all logical predicates are in Table 6. 

## **A.5.2 Propositions** 

The primary component of a task evaluation function is a list of propositions. We define a proposition as an evaluation predicate instantiated with argument values. Propositions additionally enable the evaluation of instructions with ambiguous references ( _“on a table”_ — which table?) and subset counts ( _“two spoons”_ — any two of the n total spoons). Ambiguity is enabled by extending the predicate arguments to lists. Subset counts are enabled by optional arguments number, which defines the subset size, and arg_match, which is a boolean indicating whether all entities in the subset must be satisfied with the same second argument. Suppose we want to evaluate the task _“Bring a spoon to the table.”_ . If we have a single spoon and a single table, the proposition is straightforward: 

is_on_top([spoon_1], [table_1]) _._ 

If we have multiple spoons and need just one (an ambiguous instruction), the following proposition is used: 

is_on_top([spoon_1, spoon_2, spoon_3], [table_1]) _._ 

In the above case, a list is treated as a OR of entities. The same holds for multiple tables: 

is_on_top([spoon_1, spoon_2, spoon_3], [table_1, table_2]) _,_ 

in which any spoon may be placed on any table. If the instruction specifies bringing two spoons to the table, the number=2 argument is added: 

is_on_top([spoon_1, spoon_2, spoon_3], [table_1, table_2], number=2) _._ 

20 

|**Predicate Name**|**Category**|**Description**|
|---|---|---|
|is_on_top(_o_1_, o_2)|Rearrange|_o_1 is considered on top of_o_2 if a downward raycast|
|||from any of the_o_1 bounding box corners intersects|
|||with _o_2.|
|is_inside(_o_1_, o_2)|Rearrange|_o_1 is considered inside _o_2 if a threshold number of|
|||opposing raycasts hit the same object. Ray casts|
|||are 1.00m and the threshold number is 2.|
|is_in_room(_o_1_, r_1)|Rearrange|_o_1 is considered to be in room _r_1 if at least 25%|
|||of keypoints (bounding box corners and center)|
|||are contained in the 3D region.|
|is_on_foor(_o_1)|Rearrange|_o_1 must be within 0.04m vertically of the naviga-|
|||tion mesh.|
|is_next_to(_o_1_, o_2)|Spatial|The bounding box of _o_1 must overlap vertically|
|||with the bounding box of _o_2 and the horizontal|
|||L2 distance between bounding boxes is less than|
|||or equal to 0.50m.|
|is_clustered(_o_1_, ..., on_)|Spatial|Each _oi_ satisfes is_next_to(_oi, oj_) for some _j _=|
|||_i_.|
|is_clean(_o_1)|State|_o_1 has afordancecleanableand the state machine|
|||indicates that _o_1 is clean.|
|is_dirty(_o_1)|State|_o_1 has afordancecleanableand the state machine|
|||indicates that _o_1 is not clean.|
|is_flled(_o_1)|State|_o_1 has afordance fllable and the state machine|
|||indicates that _o_1 is flled.|
|is_empty(_o_1)|State|_o_1 has afordance fllable and the state machine|
|||indicates that _o_1 is not flled.|
|is_powered_on(_o_1)|State|_o_1 has afordance powerable and the state ma-|
|||chine indicates that _o_1 is powered on.|
|is_powered_of(_o_1)|State|_o_1 has afordance powerable and the state ma-|
|||chine indicates that _o_1 is powered of.|



**Table 6 Logical predicates to evaluate the relations of objects, furniture, and rooms in the PARTNR.** Predicates exist for measuring rearrangement of objects (category=Rearrange), spatial placements relative to objects or furniture (category=Spatial), and states of objects or furniture (Category=State). For all predicates, _oi ∈{_ object, furniture _}_ and _ri_ is a room. 

21 

Finally, the instruction may specify that two spoons should be brought to the _same_ table. arg_match=True enables this: 

is_on_top([spoon_1, spoon_2, spoon_3], [table_1, table_2], number=2, arg_match=True) _._ 

Propositions are represented as JSON within dataset files and as Python function calls (as shown above) during human annotation for interpretability. 

## **A.5.3 Dependencies** 

Evaluation functions operate over a sequence of simulation states. By default, all propositions in the list of evaluation propositions will be evaluated at every time step. However, many tasks in the **PARTNR** benchmark involve a required temporal order during execution. Take for example the instruction _“Move the cup from the table to the sink. Then, return the cup to the table.”_ This consists of a multi-step rearrangement where the proposition checking that the cup is on the table is _dependent_ on a different proposition first being satisfied. We define a proposition dependency as the following: 

PropositionDependency(proposition_indices, depends_on, relation_type) _,_ 

where the argument proposition_indices is a list of integers indicating the dependent propositions, the argument depends_on is a list of integers indicating the head propositions, and the argument relation_type indicates the condition that the depends_on propositions must take in order for proposition_indices to be evaluated for satisfaction. The following relation types are supported: 

- **after_satisfied:** evaluate propositions in proposition_indices after the propositions in depends_on have been satisfied. 

- **after_unsatisfied:** evaluate propositions in proposition_indices after propositions in depends_on have been satisfied at some point in the past but are no longer satisfied. 

- **while_satisfied:** evaluate propositions in proposition_indices when propositions in depends_on are currently satisfied. 

As a concrete example, suppose the instruction is _“Move the ball and bat to the kitchen and set them next to each other. Then, move them to the closet.”_ The propositions for this task would be 

0 is_in_room([ball], [kitchen]) 1 is_in_room([bat], [kitchen]) 2 is_next_to([ball], [bat]) 3 is_in_room([ball], [closet]) 

4 is_in_room([bat], [closet]) _._ 

and the dependencies would be 

- 0 PropositionDependency([2], [0, 1], “while_satisfied”) 

1 PropositionDependency([3, 4], [0, 1], “after_satisfied”) _._ 

The first ensures that the is_next_to predicate is only queried during kitchen placements and the second ensures that placing the ball and bat in the closet is not checked at the start of the task execution, which would be inadvertently satisfied upon scene initialization. Each evaluation function has a (possibly empty) list of such proposition dependencies. 

## **A.5.4 Constraints** 

**PARTNR** tasks often require constraining _how_ tasks are completed. Examples of this include completing evaluation propositions in temporal order ( _“Do [x] before doing [y]”_ and enforcing links between ambiguities ( _“Fill one of the cups, then put that cup on the table”_ ). We define a set of constraint types that enables evaluating such task complexities: 

22 

- **TemporalConstraint:** A directed acyclic graph (DAG) over the indices of propositions that defines the temporal requirement of when propositions should be satisfied. If propositions are satisfied out of order, then the task is marked unsuccessful. The temporal constraint is defined as 

## TemporalConstraint(graph_edges) _,_ 

where graph_edges is a set of pairwise temporal constraints. For example, an edge “0 _→_ 1” indicates that the proposition at index 0 must be satisfied at an earlier time step than the proposition at index 1. 

- **SameArgConstraint:** Requires the argument used to satisfy a proposition to be the same within a pre-determined set of propositions. The same argument constraint is defined as 

## SameArgConstraint(proposition_indices, arg_names) _,_ 

where proposition_indices is a list of proposition indices to link and arg_names is a list specifying which component of each proposition to link. 

- **DifferentArgConstraint:** Requires the argument used to satisfy a proposition to be unique within a pre-determined set of propositions. The different argument constraint is defined as 

## DifferentArgConstraint(proposition_indices, arg_names) _,_ 

where proposition_indices is a list of proposition indices to link and arg_names is a list specifying which component of each proposition to link. 

- **TerminalSatisfactionConstraint:** Some propositions, once satisfied, should remain satisfied at the end of the episode. Others are expected to become unsatisfied, such as in multi-step rearrangements or object state changes of a single object. The terminal satisfaction constraint is defined as 

## TerminalSatisfactionConstraint(proposition_indices) 

where proposition_indices is a list of proposition indices that should be satisfied at time _t_ = _T_ for an episode rollout of duration _T_ . 

## **A.5.5 Evaluation Metrics** 

A task evaluation function is constituted of propositions, dependencies, and constraints as defined above. From this data, the task evaluation function serves to determine what percentage of the task is complete for a given human-robot collaboration rollout. From a sequence of simulation states, we evaluate the truth values of each proposition with respect to both the dependencies ( _when_ the propositions must be evaluated) and the constraints ( _how_ the propositions must be satisfied). The Percent Completion ( _PC_ ) is defined as the ratio of satisfied propositions to the total number of propositions. Success is defined as _S_ := ( _PC_ = 100). Both Percent Completion and Success metrics solely evaluate task completion and thus are agent-agnostic. Given that **PARTNR** is designed to evaluate multi-agent collaboration, we note that these metrics can be combined with duration-based metrics (e.g. simulation steps or time) to measure efficiency; multi-agent aspects like task division and partnered exploration serve to optimize task efficiency. The benefit of an agent-agnostic metric is flexibility to evaluate any number of agents performing the task with respect to the high-level goal. 

## **A.5.6 Evaluation Function Generation** 

This section provides detail on evaluation function generation beyond the overview provided in Section 3. In particular, Figure 8 shows the first two steps of Figure 2 (sim-in-the-loop generation and filtering + annotation) in greater detail as they pertain to evaluation generation. Notably, we take a three step process to generating evaluation functions; first, an LLM generates a list of evaluation propositions which are parsed into a usable format. Then, an LLM infers the temporal constraint over these propositions by predicting topological generations of a temporal graph. For example, the prediction [[0 _,_ 1] _,_ [2 _,_ 3]] implies that propositions at indices 0 _,_ 1 can be completed in either order, and both must be completed before propositions at indices 2 _,_ 3. This prediction is stored and evaluated more generally as edges of a directed acyclic graph (DAG). We found that the assumption of topological generations sufficiently expresses the tasks in **PARTNR** while being simpler for an 

23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0024-00.png)


**----- Start of picture text -----**<br>
Inputs Generate Evaluations Manual Annotation<br>Task Instruction 1) Infer Propositions PARTNR<br>Set a candle on each table in the living room. Then, move the potted plant to the shelf next to the picture. InstructionScene +few-shotPrompt LLM Propositions Fixed Modifiable Instruction Evaluation Generation<br>Scene Content Predicates Scene Content …<br>Objects:    * candle_0 0  on_top( [ candle_0 ], [ table_0, table_1 ] ) … Propositions Outputs<br>    * candle_1    * potted_plant_0 1  on_top( [ candle_1 ], [ table_0, table_1 ] )2  on_top( [ potted_plant_0 ], [ shelves_0 ] ) Predicate Vocab … Task Instruction<br>    * picture_0Furniture: 3  next_to( [ potted_plant_0 ], [ picture_0 ] ) … Constraints Set a candle on each table in the living room. Then, move the potted plant to the<br>    * table_0    * table_1 2) Infer Temporal Constraint … shelf next to the picture.<br>    * shelves_0Rooms:   * living_room_0   * closet_0Relations: PropositionsInstruction +few-shotPrompt LLM TemporalConstraint Pack & Verify in Sim 0  on_top( [ candle_0 ], [ table_0, table_1 ] )1  on_top( [ candle_1 ], [ table_0, table_1 ] )2  on_top( [ potted_plant_0 ], [ shelves_0 ] ) Propositions<br>   * candle_0 in closet_0   * candle_1 in closet_0 [0 → 2],   [0 → 3],   [1 → 2],   [1 → 3] Verify objects / furniture / rooms exist 3  next_to( [ potted_plant_0 ], [ picture_0 ] )<br>   * table_0 in living_room_0   * table_1 in living_room_0   * potted_plant_0 on living_room_0on_top Predicate Vocabulary inside in_room PropositionsInstruction 3) Infer Argument Constraints +few-shotPrompt LLM ArgumentConstraints              Verify object state affordances             Infer proposition dependencies  Temporal(while_satisfied([3], [2]) DependenciesConstraints<br>on_floor next_to clustered     [0 → 2],   [0 → 3],   [1 → 2],   [1 → 3]<br>clean dirty powered_on DifferentArgument ([0, 1], arg=furniture)              Run evaluation inference in Habitat )<br>powered_off filled empty LLM: CodeLLama-70b-Instruct DifferentArgument ([0, 1], arg=furniture)<br>**----- End of picture text -----**<br>


**Figure 8 Evaluation function generation pipeline overview.** A three-step LLM-generation process produces propositions and constraints during _Generate Evaluations_ . The evaluation function then is saved to a file for optional human annotation during _Manual Annotation_ . Finally, the evaluation function is packed and verified in simulation during _Pack & Verify in Sim_ to ensure that all entities and affordances exist. 

|||Correctness|(%)|
|---|---|---|---|
||Task|Evaluation|Combined|
|Constraint-Free|89|97|86|
|Spatial|87|85|74|
|Temporal|91|93|85|
|Heterogeneous|93|91|85|
|Average|90|92|83|



**Table 7 Manually-annotated generation accuracy of 100k-scale PARTNR tasks and evaluation functions.** Altogether, we find that 83% of episodes are generated without any task or evaluation function errors. Analysis performed on 100 sampled episodes of each task type via PrediViz. 

LLM to generate than a DAG. The third step is predicting argument constraints, in which the LLM is provided the instruction and propositions list and must predict a list of constraints, either SameArgConstraint or DifferentArgConstraint. All LLM queries are performed against CodeLlama70b-Instruct (Roziere et al., 2023). All prompts used for evaluation function generation are included in Appendix A.15. 

## **A.6 Human annotation and accuracy assessment for PARTNR data** 

## **A.6.1 Generation Accuracy for Tasks and Evaluations** 

It is important for tasks in **PARTNR** to be solvable by collaboration agents and for the associated evaluation functions to accurately reflect the task being performed. In this section, we analyze the accuracy of 100k-scale **PARTNR** generation with respect to both of these criteria using the PrediViz tool (Appendix A.6.2). In Table 7, we demonstrate that the accuracy of task generation ranges from 87-93% depending on task type and averages 90%. The accuracy of evaluation generation ranges from 85-93% depending on task type and averages 92%. Combining these numbers yields an overall joint accuracy of 83% for our 100k-scale dataset. In Table 8, we annotate the failure modes that lead to unsolvable tasks and incorrect evaluation functions. Common task-related failure modes include: 

- **Hallucination.** The produced instruction references objects, furniture, or rooms that do not exist in the scene the instruction was generated for. Example: _"Move the clothes to the washing machine."_ produced for an environment that does not contain a washing machine. 

24 

||Task Failures||Evaluation Failures||
|---|---|---|---|---|
||Mode|%|Mode|%|
||Hallucination|7|Incorrect Ambiguity|2|
|Constraint-Free|Already Satisfed|2|Incorrect Furniture|1|
||Contradiction|2|-|-|
||Hallucination|6|Incorrect Temporal Grouping|7|
|Spatial|Unresolvable Ambiguity|5|Incorrect Predicate (Other)|5|
||Already Satisfed|2|Incorrect Predicate (Room vs Furniture)|3|
||Hallucination|4|Incorrect Temporal Grouping|3|
|Temporal|Contradiction|3|Incorrect Predicate (Other)|3|
||Already Satisfed|2|Incorrect Predicate (Room vs Furniture)|1|
||Hallucination|3|Incorrect Predicate (Room vs Furniture)|5|
|Heterogeneous|Unresolvable Ambiguity|2|Incorrect Object/Furniture/Room|2|
||Contradiction|1|Incorrect Predicate (Other)|2|



**Table 8 Top three failure modes of 100k-scale task and evaluation generation reported for each task type.** Failures of task generation are led by the hallucination of non-existent entities, while failures of evaluation generation are led by incorrect temporal predictions and incorrect predicate functions. Analysis performed on 100 sampled episodes of each task type via PrediViz. 

- **Unresolvable Ambiguity.** The produced instruction contains ambiguous directives that cannot be reasonably resolved without further communication or a detailed understanding of the task-issuer’s preferences. Example: _"Set the table for dinner"_ ; how many place settings are necessary? Should we set the formal dining table or nook table? What cutlery is needed? 

- **Contradiction.** The produced instruction involves two or more sub-tasks that cannot simultaneously be satisfied. Example: _"Set the scissors on the coffee table. Set the bowl on the counter next to them."_ 

- **Already Satisfied.** The produced instruction dictates sub-tasks that are all already satisfied at the start of the episode. Example: _"Move the laptop to the living room and turn it on"_ , when the laptop is already powered on and in the living room. 

According to Table 8, hallucinations are the most common failure mode for task generation. While simulationin-the-loop filtering avoids this issue for evaluation generation, such filtering is inconsistent for tasks; language has a looser grounding to scene entities than statements of propositional logic. For example, a home might not have a formal dining room, but a table in the living room may serve the purpose of a dining table. Moving on to evaluation generation, common failures are as follows: 

- **Incorrect Temporal Grouping.** The instruction implies a temporal order among sub-tasks (either explicitly via sequencing words, or implicitly via multi-step manipulations) and the predicted temporal constraint fails to reflect this order over the propositions. Example: allowing propositions to be satisfied in any order for the task _"First, return the plates to the kitchen. Then, tidy up the living room."_ 

- **Incorrect Predicate (Other).** The evaluation function uses the wrong predicate function to evaluate the task. Example: using is_powered_on instead of is_filled when the instruction asks to fill the kettle. 

- **Incorrect Predicate (Room vs Furniture).** A task specifies that an object should be rearranged to another room, but the propositions overly-constrain the rearrangement to a target furniture for placement. This failure is separate from the one above because it the most common, and it indicates the tendency for the LLM to produce single solution instances rather than reflect the full space of ambiguity. Example: producing a proposition like is_on_top([electronic_cable], [bed]) for the task _"Move the electronic cable to the bedroom."_ 

- **Incorrect Object/Furniture/Room.** incorrect entities are selected for satisfying proposition. Example: the instruction calls for rearranging the cushion to a living room table, but table_4 in the proposition is_on_top([cushion_0], [table_4]) exist in bedrooms. 

25 

- **Incorrect Ambiguity.** For an instruction that can be satisfied n different ways, the evaluation affords m options for solution, where m _̸_ = n. Example: _"Move a toy to the kid’s room"_ , where two or more toys exist in the toy bin but the evaluation function does not list out all possible toys for rearrangement. 

According to Table 8, the primary failures modes of evaluation generation are incorrect temporal grouping and incorrect predicates. Regarding temporal grouping, we observe that many sub-task require multiple propositions to evaluate. Grouping these propositions consistently within the temporal constraint is a source of error. Take for example the instruction _"Set the shirt and pants next to each other on the counter. Then, move them to the dresser."_ In this case, five propositions will exist; four for placements and one for the spatial relation. The temporal prediction may erroneously link the spatial relation with the dresser placements rather than the counter placements. 

## **A.6.2 Visualization of PARTNR Tasks and Evaluation Functions** 

LLM-generated **PARTNR** episodes may contain errors so we evaluate their correctness with human annotators. This is a verification problem in which the generated instruction, evaluation function, and contents of the scene must be compared against each other. See Figure 6 for an example of this in code. To make this process faster, easier, and more accurate, we designed a visualization and annotation system (PrediViz) that illustrates the state of the world and the evaluation function relative to a task instruction (Figure 9). We chose a 2D illustration style to capture the relevant structure in the data: 

- **Rooms.** Each room is drawn as a box with a name provided underneath. Names repeat if there are multiple instances of that room category in the scene. We model the rooms as a fully-connected graph of accessibility. We treat rooms as spatially independent and reorder or wrap them as needed. 

- **Objects.** We visualize objects as boxes with category names underneath. Each proposition of the evaluation function is assigned a unique color to separate them from other propositions. For consistency, objects are colored by the color of the first proposition they appear in. 

- **Receptacles.** Receptacles are furniture. We designed a bespoke set of icons for 25 categories of furniture that can be easily subitized – you can tell a chair is a chair and a table is a table just by glancing at it. 

- **States.** Both objects and receptacles have time-varying state (empty/filled, dirty/clean, powered on/off). These states are displayed using motifs for objects and textual labels for receptacles, when necessary. 

- **Affordances.** Different furniture afford different object placements. For example, a cup can be placed _on top_ of the fridge or _inside_ it, but only _on top_ of a table. These relationships are annotated by hand at the category-level. Objects are visualized at their initial receptacle/room placements. An associated dark box indicates the target placement of a particular object. 

- **Placements.** To represent rearrangement of an object, we draw an arrow from the initial position to the target position. A simple placement is represented by a single arrow. Multiple allowable placement targets (e.g. multiple tables in the room) are represented by multiple arrows. We use solid arrows for AND placements (e.g. _"place the doll and the toy truck on the couch"_ ) and dotted arrows for OR (e.g. _"place the doll or the toy truck on the couch"_ ). We also support for choosing _k_ out of _n_ objects for placement (e.g. _"place two out of three dolls on the couch"_ ) using a numerical label on the dotted lines. 

- **Temporal Constraints.** To visualize temporal constraints, we split the instruction into multiple frames. For example, if the instruction requires us to _"place the doll on the couch and the toy truck in the chair, then put the stuffed toy inside a chest of drawers,"_ we create one frame to represent the first half of the statement (i.e. _"place the doll on the couch and the toy truck in the chair"_ ) and another for the second half (i.e. _"put the stuffed toy inside a chest of drawers"_ ). 

- **Special Relations.** We also illustrate special relations like next_to, same_as, and different_from in the style of an informational legend shown on the side. 

Resulting visualizations are wrapped in a web-based annotation tool that affords binary verification and failure model labeling. See Figure 8 for results derived from this tool. We ran a small-scale experimental study (n=22) comparing human annotation using PrediViz to a text-based representation. Using PrediViz, verification was 2.6 times faster, 8% more accurate, and perceived as 24% easier. 

26 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0027-00.png)


- **(a)** Example task #1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0027-02.png)


- **(b)** Example task #2: first temporal frame 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0027-04.png)


**(c)** Example task #2: second temporal frame 

**Figure 9 PARTNR tasks visualized in PrediViz.** The design distills the task and scene to only the components necessary for verification. In example task #2, the split frames signify that the agents must first rearrange the file sorter and folder (Figure 9b), _then_ rearrange the backpack (Figure 9c). 

27 

## **A.6.3 Human-assisted Correction Annotation** 

Below is an example task and evaluation function saved in plain text for human annotation. The instruction, propositions, and constraints can be modified as necessary to ensure the task is feasible and that the evaluation function reflects it. Annotators also have access to a file containing the objects, furniture, rooms, and relations thereof to reference during this process. 

**Dataset Correction Annotation Trial** # type : ignore # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # INSTRUCTION # modify as necessary , but keep in mind the scene i s f i x e d . # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - i n s t r u c t i o n = """ Help me organize the entryway . First , place the phone , watch , and keychain on the t a b l e next to each other . """ # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # PROPOSITIONS # is_on_top ( objects , r e c e p t a c l e s , number=1, is_same_receptacle=False ) # i s _ i n s i d e ( objects , r e c e p t a c l e s , number=1, is_same_receptacle=False ) # is_in_room ( objects , rooms , number=1, is_same_room=False ) # is_next_to ( entities_a , entities_b , number=1, is_same_b=False , l2_threshold =0.5) # is_on_floor ( objects , number=1) # Args : # o b j e c t s / r e c e p t a c l e s / e n t i t i e s _ *: OR of a l i s t # number : n o b j e c t s / e n t i t i e s _ a must s a t i s f y # is_same : the same e n t i t y must s a t i s f y a l l n o b j e c t s # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - p r o p o s i t i o n s = [ is_on_top ( [ ’ cellphone_0 ’ ] , [ ’ table_4 ’ ] ) , is_on_top ( [ ’ watch_0 ’ ] , [ ’ table_4 ’ ] ) , is_on_top ( [ ’ keychain_0 ’ ] , [ ’ table_4 ’ ] ) , is_next_to ( [ ’ cellphone_0 ’ ] , [ ’ watch_0 ’ ] ) , is_next_to ( [ ’ watch_0 ’ ] , [ ’ keychain_0 ’ ] ) , ] # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # TEMPORAL GROUPS # Place p r o p o s i t i o n s in groups s . t . one group must be s a t i s f i e d before the next . # Example : # [ [ 0 , 1 ] , [ 2 , 3] ] means props 0 & 1 must be s a t i s f i e d before props 2 & 3. # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - temporal_groups = [ [ 0 , 1 , 2 , 3 , 4 ] , ] # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # TIE CONSTRAINTS # options : SameArgConstraint , DifferentArgConstraint # Args : # proposition_indices : L i s t [ i n t ] # arg_indices : L i s t [ i n t ] # Example : # SameArgConstraint ( [ 0 , 2 ] , [ 1 , 1 ] ) . Means : Propositions 0 & 2 must # match values on the argument at argument index 1 and 1 , r e s p e c t i v e l y . # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - t i e _ c o n s t r a i n t s = [ ] # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # TERMINAL SATISFACTION CONSTRAINT: # We assume a l l p r o p o s i t i o n s must remain s a t i s f i e d to the end of the episode . # i f a p ro p os i ti o n * should * become u n s a t i s f i e d , add i t s index here . # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - exclude_final_constraint = [ ] # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # mark True i f the task has a f a t a l i s s u e # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - skip_episode = False reason = "" 

28 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0029-00.png)


**----- Start of picture text -----**<br>
House<br>Rooms<br>Furniture<br>Objects<br>Agents<br>**----- End of picture text -----**<br>


**Figure 10 World graph.** The high-level policies of the human and robot agents leverage a hierarchical world graph containing information about rooms, furniture, and objects in the environment. This graph is updated based on the observations and actions of both agents. 

## **A.7 World graph: the perception framework for LLM agents** 

Scene-graph style hierarchical graphs have been shown to be effective for planning problems Gu et al. (2024); Agia et al. (2022); Rana et al. (2023). Inspired by such prior work, as illustrated in Figure 10, our world graph is a hierarchical multi-edge directed graph with _K_ = 3 levels for representing the entities in the world. The nodes at first level correspond to rooms in the environment, followed by furniture at second, and objects and agents at third level. The root of this graph is an abstract house-node denoting the environment where tasks are taking place. Apart from the semantic information, each node also stores the 3D location of the entity and its affordance states e.g., clean/dirty, on/off, open/close, etc. The graph can then be serialized or accessed via specialized tools by ReAct policies. Prompts in Section A.16 provide an example of how the world graph is used by the LLMs in our baselines. 

## **A.7.1 Building and Updating the World Graph in Simulation** 

The initial graph is built by reading the the ground-truth room region annotations and furniture placements associated with a particular scene. The location of furniture and the region boundaries are used to associate each furniture to a specific room. This creates the initial two tiers, Room and Furniture, of the privileged world-graph. In partial-observability setting this is all the planner gets to start planning. However, in the full-observability setting, object-to-furniture assignments are read from each episode’s initialization information and a third Object tier is added to the world-graph. 

Under partial-observability setting during execution, we use the panoptic sensors attached to both agents to detect all visible objects in the current-frame. Then ground-truth simulation information is used to extract the housing furniture or agent for each of these objects, as well as the location, and this new information is added to the maintained graph. For full-observability setting, each frame latest graph is mined from simulator using current information for objects and overwrites the previous world-graph. 

It is not guaranteed in our setup that the images will pick up an object that was placed, filled or powered on/off in the previous step. Therefore, in partial-observability setting, we also add deterministic updates to the graph based on previous action and its result, e.g. successful placement triggers deletion of edge between agent who was holding the object and an addition of edge between placed object and the furniture it was placed on per action arguments. 

## **A.7.2 Non-privileged Perception using ConceptGraphs** 

In order to study the dependence of the planner on the underlying world-graph, we also follow a modified version of ConceptGraphs (CG) pipeline Gu et al. (2024) to create the initial layout that is used to initialize 

29 

our world-graph representation. 

_Modified Pipeline for Creating ConceptGraphs._ We adapt the original CG pipeline to only use Meta-CLIP models for getting object name and category; using YOLO and SAM for object segmentation; and LLaMA3.1 for annotating room-label and inter-object spatiosemantic relations. Instead of using LLaVA and GPT for getting open-vocab descriptive object names, we use multi-perspective averaged CLIP embeddings of each object to classify its category given our closed object-vocabulary. For the required room-to-furniture relations, we extend CG pipeline by adding another prompt and query Llama3.1 model to assign room-labels to each entity given the categories of 10 closest entities to it. 

_Updating ConceptGraphs._ Like the privileged world-graphs there are two modes of updating a ConcepGraphsinitialized non-privileged world-graph: 

1. **Observations.** Using the same panoptic sensors we extract all the visible objects in current frame for both the agents. We use depth-sensors to extract the point-cloud associated with this object. We use this 3D location to first check if this is a redetection of a previously detected object. We use location, category and whether object-is-being-held-by-agent features to assess if this is a degenerate detection. If this is a new object detection then we use its location along-with bounding-boxes of existing furniture to check if this object is contained-within or on-top-of any of the recognized furniture-pieces. If it is then the world-graph is updated with this node and the edge. 

2. **Actions.** Just like privileged world-graph, we can not guarantee sensors will pick up changed state of an object that is placed, powered, etc. Thus we add similar action based updates to this version as well. A special case is when non-privileged graph is upated by human agent’s action arguments which are grounded in privieleged world-graph and may refer to same physical entity with different given names, e.g. backpack_0 is backpack_153 in non-privileged graph. We use a simple proximity and category matchig heuristic to match human’s arguments to known entities in non-privileged graph, falling back on proximity based matching when no entity of same category are found. 

_Using ConceptGraphs with Simulated Skills_ Simulated skills require sim-handles of the placement furniture to snap objects onto them. The furniture in non-privileged graph do not have these sim-handles by design. Thus we come up with a simple proximity and category-name based matching to match a ground-truth furniture entity to a detected furniture-entity. If we can not find any ground-truth entity of the same category close to the detected furniture, we fallback on matching to the closest entity. 

_Prompts and Models used in ConceptGraphs Creation._ CLIP model and pretrained backbone checkpoints: ViT-H-14-quickgelu, metaclip_fcc. Object detector: YOLO checkpoint yolov8x-worldv2. LLM: Llama3.170B-Instruct 

## **Room Annotation Prompt for Modified CG** 

<|begin_of_text|><| start_header_id|>system <|end_header_id|>You are an expert on house layouts . You 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0030-09.png)


**Figure 11 ConceptGraph pipeline.** Left to right: Point-cloud of the scene built from a trajectory; All objects detected by ConceptGraphs pipeline after assigning a category-name based on CLIP; Semantic visualization of the apartment with color-coded class categories per furniture 

30 

w i l l be given an input which w i l l d e s c r i b e QUERY_OBJECT. This object w i l l be described by i t s name and the 10 p i e c e s of f u r n i t u r e c l o s e s t to i t . You w i l l a s s i g n a ROOM_NAME to t h i s object . The input w i l l be a JSON with f i e l d s "QUERY_OBJECT_NAME" and "CLOSEST_OBJECTS". Your output should a l s o be a JSON c o n s i s t i n g of key "ROOM_LABEL". You should only output the JSON and nothing e l s e . 

You should only a ss ig n one of the f o l l o w i n g l a b e l s : 1. bedroom 2. l i v i n g room 3. kitchen 4. dining room 5. hallway 6. bathroom 7. unknown : only when none of the above s t r i n g s d e s c r i b e the object <| eot_id|> 

## **A.8 Learned low-level robot skills** 

We provide the implementation details for training the learned low-level robot skills: [Explore, Navigate, OpenFurniture, CloseFurniture, PickObject, PlaceObject]. These skills are based on two types of basic skills: Navigate skill, which outputs robot base control velocity command given robot depth and sensor observations, and Manipulation skill, which outputs joint angles and base control velocity commands to reach the target location given robot depth and sensor observations. We then use these two skills to get the above robot skills. 

## **A.8.1** Navigate **Skill** 

We follow the learned low-level robot skill from Puig et al. (2024) to get the skill. We briefly describe the details for completeness. The goal of Navigate skill is to navigate to the target object given the target object location. The observation space includes (1) an arm depth camera (224 _×_ 171 with hFOV of 55), (2) the relative pose of the target object location in 2-dim polar coordinate system. The action space includes (1) the linear and (2) angular robot base velocities with the range of _−_ 10 and +10 m/s. The reward function is to encourage the robot to move forward while facing the target location with a correct orientation. An additional navigation success reward is given if the robot can navigate close enough to the target object, and a collision penalty is given if the robot collides with obstacles in the scene. Moreover, a slack reward is given to let the robot navigate to the target in as few steps as possible. Finally, the skill is trained with DD-PPO distributing training. 

Below is the list of skills based on navigate skill. 

- Navigate: As described above. 

- Explore: It is a composed skill that involves calling Navigate skills sequentially given a sequence of navigation waypoints. 

## **A.8.2** Manipulation **Skill** 

We follow the work from Puig et al. (2024) to train the skill. We briefly describe the details for completeness. The goal of Manipulation skill is to drive the robot’s arm and base to reach the target object’s location. The observation space includes (1) an arm depth camera (224 _×_ 171 with hFOV of 55), (2) the relative pose of the target object in a 3-dim Cartesian coordinate system, (3) the 7-dim arm joint angles, (4) a binary holding detector, and (5) the relative pose of arm end-effector to the target resting location in a 3-dim Cartesian coordinate system. The action space includes (1) the linear and (2) angular base velocities with the range of _−_ 10 and +10 m/s, (3) the delta arm joint angles applied to the arm with the range of _−_ 5 _×_ 10 _[−]_[2] and +5 _×_ 10 _[−]_[2] (7-dim), and (4) a binary command to grasp or release the object. The reward function is to encourage the arm to move toward the object. In addition, a success reward is given if the robot interacts with the right target object. Moreover, a slack reward is given to let the robot complete the task in as few steps as possible. Finally, the skill is trained with DD-PPO distributing training. 

Below is the list of skills based on manipulate skill. 

- OpenFurniture: Given the drawer handle location, manipulate skill drives the arm and the base to the target. The articulated furniture is opened if the gripper location is close enough to the handle location. 

31 

- CloseFurniture: Given the drawer handle location, manipulate skill drives the arm and the base to the target. The articulated furniture is closed if the gripper location is close enough to the handle location. 

- PickObject: Given the target object location, manipulate skill drives the arm and the base to the target. The object is snapped to the gripper if the gripper location is close enough to the target object location. 

- PlaceObject: Given the target place location, manipulate skill drives the arm and the base to the target. The object is desnapped from the gripper to the target place location if the gripper location is close enough to the target place location. 

## **A.9 Implementation Details for ReAct Agents** 

For all experiments, LLM inferrence is performed on two Nvidia A100 GPUs using the gpt-fast inference engine PyTorch (2023). Inference on LLama-3.1-70B (using tensor parallelism over two A100s), resulted in an average generation speed of 11.43 tokens/s. Each planning step required an average of 52 tokens resulting in a latency of 4.55 seconds per planning step. The average wall time to complete and entire episode (planning steps for both agents and simulation time) was 36.0 minutes. The finetuned model based on Llama-3.1-8B required an average of 0.53s per planning step. For those experiments, simulation time and human agent inference time remained unchanged, giving a final wall time of 25.3 minutes per episode. 

All decentralized baselines had a maximum timeout of 50 replanning calls, while centralized baselines had a maximum timeout of 100 replanning calls (to account for the fact that one planner would need to plan for both agents). Additionally all baselines had a maximum timeout of 20000 simulation steps. 

## **A.9.1 Skill API Library** 

Below is a list of the skills available across all baselines. Agents acting in the robot role do not have access to state-modifying actions (Clean, Fill, Pour, PowerOff, PowerOn). The ReAct agents considered in the main paper do not have access to perception tools (FindAgentActionTool, FindObjectTool, FindReceptacleTool, FindRoomTool). We additionally study ReAct agents that query the environment via those tools, which we name ReAct-Tools. 

- **Clean** : Used for cleaning an object. 

- **Close** : Used for closing an articulated entity. 

- **Explore** : Search a specific room by visiting various receptacles or furnitures in that room. 

- **Fill** : Used for filling an object. 

- **Navigate** : Used for navigating to an entity. 

- **Open** : Used for opening an articulated entity. 

- **Pick** : Used for picking up an object. The agent cannot hold more than one object at a time. 

- **Place** : Used for placing an object on a target location. This requires the name of the object to be placed, the name of the furniture where it should be placed, spatial relation (“on” or “within”) describing the relation between the object and furniture. The object to be placed must already be held by the agent (i.e. picked previously). Additionally, you can request to place the object near another object. For that you can optionally provide a spatial constraints (“next_to”) and the name of the reference object. To place next to an object, the reference object must already be on the target furniture. API tempate: Place[ _<_ object_to_be_placed _>_ , _<_ spatial_relation _>_ , _<_ furniture to be placed on _>_ , _<_ spatial_constraint _>_ , _<_ reference_object _>_ ]. spatial_constraint and reference_object should be set to “None” when necessary. 

- **Pour** : Used for pouring from one container to another. This skill will pour into the specified container 

- from whichever container is currently held by the agent. 

- **PowerOff** : Used for turning off a powered object 

- **PowerOn** : Used for turning on a powered object 

32 

- **Rearrange** : Used for moving an object from its current location to the target location. This requires the name of the object to be rearranged, the name of the furniture where it should be placed, spatial relation (“on” or “within”) describing the relation between the object and furniture. This skill will automatically pick the specified object and move to the target furniture and attempt to place it. Additionally, you can request to place the object near another object. For that you can optionally provide a spatial constraints (“next_to”) and the name of the reference object. To place next to an object, the reference object must already be on the target furniture. API tempate: Rearrange[ _<_ object_to_be_placed _>_ , _<_ spatial_relation _>_ , _<_ furniture to be placed on _>_ , _<_ spatial_constraint _>_ , _<_ reference_object _>_ ]. spatial_constraint and reference_object should be set to “None” when necessary. 

- **Wait** : Used to make agent stay idle for some time 

- **FindPartnerAgentActionTool** : This tool will return a summary of the other agent’s actions. 

- **FindObjectTool** : Used to find the exact name/names of the object/objects of interest. An LLM will 

- be used to distill relevant objects from the user query. Example (FindObjectTool[toys on the floor] or FindObjectTool[apples]) 

- **FindReceptacleTool** : Used to know the exact name of a receptacle. An LLM will be used to distill relevant receptacles from the user query. Example (FindReceptacleTool[a kitchen counter]) 

- **FindRoomTool** : Used to know the exact name of a room in the house. An LLM will be used to distill 

- relevant rooms from the user query. Example (FindRoomTool[a room which might have something to eat]) 

- **Done** : Used to indicate that the agent has finished the task. 

## **A.9.2 Constrained Generation** 

We follow the procedure described in Geng et al. (2023), constraining token sampling to only select tokens that consistent with at least one accepting string in the specified grammar. For each call to the LLM we build a grammar which will only accept valid tool calls on observed entities. Below is the base grammar used tool calls for all experiments. For experiments utilizing a summary of the world representation (i.e. ReAct, Finetuned see Section 4.1) the perception tools (FindObjectTool, FindReceptacleTool, etc.) are omitted. The rules for object, furniture, and room are set dynamically based on the agent’s current world graph. This ensures skills are called only for entities that the agent knows exist. 

## **Tool Call Grammar** 

root ::= Navigate | Pick | Place | Open | Close | Rearrange | PowerOn | PowerOff | Clean | F i l l | Pour | Explore | Wait | FindReceptacleTool | FindObjectTool | FindAgentActionTool | FindRoomTool | Done Navigate ::= " Navigate [" nav_target "]" Pick ::= " Pick [" object "]" Place ::= " Place [" object " ," WS s p a t i a l _ r e l a t i o n " ," WS f u r n i t u r e " ," WS (( s p a t i a l _ c o n s t r a i n t " ," WS obj_or_furniture ) | ((" none" | "None") WS " ," WS (" none" | "None") ) ) "]" Open ::= "Open [" f u r n i t u r e "]" Close ::= " Close [" f u r n i t u r e "]" Rearrange ::= " Rearrange [" object " ," WS s p a t i a l _ r e l a t i o n " ," WS f u r n i t u r e " ," WS (( s p a t i a l _ c o n s t r a i n t " ," WS obj_or_furniture ) | ((" none" | "None") WS " ," WS (" none" | "None") ) ) "]" PowerOn ::= "PowerOn [" obj_or_furniture "]" PowerOff ::= "PowerOff [" obj_or_furniture "]" Clean ::= "Clean [" obj_or_furniture "]" F i l l ::= " F i l l [" object "]" Pour ::= "Pour [" object "]" Explore ::= " Explore [" room "]" Wait ::= "Wait [" "]" FindReceptacleTool ::= " FindReceptacleTool [" free_text "]" FindObjectTool ::= " FindObjectTool [" free_text "]" FindAgentActionTool ::= " FindAgentActionTool [" "]" FindRoomTool ::= "FindRoomTool [" free_text "]" Done ::= "Done [ ] " nav_target ::= ( f u r n i t u r e | room | object ) object ::= " object_1 " | " object_2 " | . . . obj_or_furniture ::= ( f u r n i t u r e | object ) f u r n i t u r e ::= " furniture_1 " | " furniture_2 " | . . . room ::= "room_1" | "room_2" | . . . s p a t i a l _ c o n s t r a i n t ::= "next_to" s p a t i a l _ r e l a t i o n ::= "on" | " within " 

33 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0034-00.png)


**Figure 12 Overview of ReAct-RAG.** RAG consists of two steps: (1) We first use a training dataset to obtain traces from LLMs and log traces with task instructions. (2) During runtime, we retrieve the traces by doing dot product on the instructions and obtaining the trace with the highest score, and finally put the trace in the prompt. We ensure that the dataset used for the first step is different from the one in the second step. 

free_text ::= [ " ’ . : , ! a -zA -Z_0- 9 ] * WS ::= [ ]* 

## **A.9.3 Retrieval-Augmented Generation for ReAct agents** 

Retrieval-Augmented Generation (RAG) is the method of optimizing LLM text generation by querying an external database. However, there are several challenges to applying RAG in our setup. First, when implementing RAG, it is necessary to provide an external database from which LLMs can retrieve information. However, it is unclear how to effectively generate such a dataset and determine which content is most beneficial for solving the task. Second, once LLMs retrieve the information, it is unknown where to ingest this information into the generation process. To solve the above challenges, we develop an approach inspired by the recent literature that uses an LLM to generate its training dataset to iteratively optimize performance (Pang et al., 2024; Madaan et al., 2024). As shown in Fig. 12, we first construct the dataset by collecting the successful traces generated by LLMs for solving the training tasks. Then, during test time, we select the most relevant trace by comparing the sentence similarity between a test instruction and the ones in the dataset. The selected trace is passed back to the LLM’s promppt as a successful example trace. This represents a refinement process that uses its own past success experience to increase the chance of solving downstream tasks. Figure 12 illustrates the high-level idea. 

Specifically, we use the **PARTNR** train dataset to generate ReAct traces (where the human agent is ReAct-Tools and the robot agent is ReAct). In total, we generate 925 traces that successfully solve the tasks to form the RAG dataset. During evaluation time, we use sentence similarity computed by all-mpnet-base-v2 Reimers and Gurevych (2019) to select the most similar instruction to the instruction at hand in the dataset, followed by adding the trace into the ReAct prompt. 

## **A.10 Finetuning LLM agents** 

In this section, we describe how we finetune an LLM to build our Finetuned baseline. We detail the process to generate the data for finetuning our model and the training details below. 

34 

## **A.10.1 Data Generation for Finetuning** 

We train the model using successful traces from the ReAct baseline, which obtains the best decentralized results. In particular, we run this baseline together with a ReAct-Tools human on the training set, and keep the episodes with 1.0 success rate. If an episode reaches a 100% success rate some time during the task, but fails at the end, we keep the actions up until the success step and replace the last action with a Done[] action to finish the episode. This process results in 1,226 episodes. We then split each episode into the sequence of robot actions, and filter out those that resulted in failure. Our training set is constructed by building a prompt for each of the successful actions, as shown in Sec. A.16. This process results in 15,889 training samples. Note that each action prompt contains the current state of the environment and the previous robot and human actions, but filters out the thought produced by the human or robot. 

We also explored using the Heuristic-Expert baseline as a data source for finetuning, but we did not observe improvements in the resulting model. Given that this baseline plans using the ground truth evaluation function, we hypothesized it would help distill the natural language task into the right predicates. We followed the same process described above, obtaining 1250 episodes and 13939 training samples. We trained a model with ReAct, Heuristic-Expert and both data sources. We show the evaluation results for each model in Table 9, with the model trained with only React data performing the best. 

|**Data source**|Sim Steps _↓_|Success Rate _↑_|Percent Complete _↑_|Planning Cycles _↓_|
|---|---|---|---|---|
|Heuristic-Expert|3477.82 _±_ 78.19|0.63 _±_ 0.02|0.79 _±_ 0.01|15.50 _±_ 0.24|
|ReAct|3228.96 _±_ 75.14|0.70 _±_ 0.01|0.84 _±_ 0.01|12.85 _±_ 0.24|
|ReAct + Heuristic-Expert|3552.96 _±_ 61.95|0.69 _±_ 0.01|0.83 _±_ 0.01|14.47 _±_ 0.30|



**Table 9 Performance of Finetuned model when using different data sources for finetuning.** We measure performance using simulation steps required to finish the episode, success rate and percent complete on the tasks, and the average number of planning cycles performed by the planner. Mean and standard error are reported over the validation set. Heuristic centralized expert has access to the task evaluation function. 

## **A.10.2 Implementation Details** 

We train the model to predict, for every example, the action taken by the agent, which corresponds to the text after the <|reserved_special_token_0>| token. 

We use a low rank adapter Hu et al. (2021) with _r_ = 132 _, α_ = 128 _,_ dropout=0 _._ 01, on top of the value and query projection layers _W[V] , W[Q]_ . We train all models on 4 A100 GPUs, with a batch size of 2 per GPU. The models are trained for 40,000 steps, which takes around 24 hours. 

## **A.11 Additional Results** 

To supplement the results in Table 2, we have included Table 10, containing results from additional baselines on the validation set, and Table 11 containing results on the **PARTNR** test set. ReAct rows use the same summary prompting format as the baselines in 2. ReAct-Tools requires the agents to use perception tools to observe the environment instead. ReAct-8B uses Llama-3.1-8B-Instruct as the robot model for an equal capacity comparison with Finetuned. 

## **A.12 Analysis of collaborative behavior and efficiency of LLM agents** 

Since task state success and percent complete metrics look at overall team performance for the human and robot agents in our tasks, we also evaluate metrics that allow us to look at different aspects of collaborative behavior of the agents. We measure percentage of sub-tasks done by the robot (task offloading), ratio of unnecessary rearrangements over total successful rearrangements done by both agents (extraneous effort), and number of exploration steps needed before first object is picked (exploration efficiency) to analyze agent behaviors (Table 12). Our main findings are below: 

**Agents are able to find objects faster when collaborating as compared to solo, but only when they successfully co-ordinate.** The exploration efficiency increases i.e., agents are able to find task-relevant objects in fewer 

35 

||**Method**<br>**Controllability**<br>**Skills**<br>**Observability**<br>**Sim**<br>**Steps**_↓_<br>**Success**<br>**Rate**_↑_<br>**Completion**<br>**Rate**_↑_<br>**Planning**<br>**Cycles**_↓_|
|---|---|
||Heuristic-Expert<br>Centralized<br>Oracle<br>Full<br>1260.88 _±_ 26.97<br>0.84 _±_ 0.01<br>0.94 _±_ 0.01<br>N/A<br>ReAct<br>Centralized<br>Oracle<br>Full<br>1347.43 _±_ 33.80<br>0.74 _±_ 0.01<br>0.88 _±_ 0.01<br>17.49 _±_ 0.34<br>ReAct<br>Decentralized<br>Oracle<br>Full<br>1915.63 _±_ 56.84<br>0.74 _±_ 0.01<br>0.86 _±_ 0.01<br>14.20 _±_ 0.34|
||ReAct<br>Centralized<br>Oracle<br>Partial<br>2298.13 _±_ 61.39<br>0.74 _±_ 0.01<br>0.85 _±_ 0.01<br>20.73 _±_ 0.51<br>ReAct<br>Decentralized<br>Oracle<br>Partial<br>3295.20 _±_ 76.27<br>0.73 _±_ 0.01<br>0.86 _±_ 0.01<br>15.24 _±_ 0.31<br>ReAct-Tools<br>Decentralized<br>Oracle<br>Partial<br>3622.52 _±_ 79.09<br>0.71 _±_ 0.01<br>0.85 _±_ 0.01<br>21.41 _±_ 0.34<br>ReAct + RAG<br>Decentralized<br>Oracle<br>Partial<br>3467.47 _±_ 82.39<br>0.71 _±_ 0.01<br>0.84 _±_ 0.01<br>14.75 _±_ 0.31<br>Finetuned<br>Decentralized<br>Oracle<br>Partial<br>3228.96 _±_ 75.14<br>0.70 _±_ 0.01<br>0.84 _±_ 0.01<br>12.85 _±_ 0.24|
||ReAct<br>Decentralized<br>Learned<br>Partial<br>6494.88 _±_ 181.52<br>0.57 _±_ 0.02<br>0.76 _±_ 0.01<br>22.72 _±_ 0.58<br>ReAct<br>Decentralized<br>Learned<br>ConceptGraph<br>12274.27 _±_ 212.65<br>0.25 _±_ 0.01<br>0.53 _±_ 0.01<br>26.74 _±_ 0.45|
||ReAct-Single<br>Single Agent<br>Oracle<br>Partial<br>2519.02 _±_ 57.48<br>0.73 _±_ 0.01<br>0.85 _±_ 0.01<br>18.68 _±_ 0.33|
||ReAct-Single<br>Single Agent<br>Oracle<br>Full<br>1590.20 _±_ 42.73<br>0.73 _±_ 0.01<br>0.85 _±_ 0.01<br>18.60 _±_ 0.38|
||ReAct-8B<br>Decentralized<br>Oracle<br>Partial<br>3699.45 _±_ 87.40<br>0.64 _±_ 0.02<br>0.80 _±_ 0.01<br>23.15 _±_ 0.47|



**Table 10 Baseline results on PARTNR validation set.** We measure performance using simulation steps required to finish the episode, success rate and completion rate on the tasks, and the average number of planning cycles performed by the planner. Mean and standard error are reported over the validation set. Heuristic centralized expert has access to the task evaluation function. Collaboration enables higher task completion and success as compared to single agent task execution (shown in gray), at the expense of more simulation steps highlighting the coordination “burden". 

|**Method**|**Controllability**|**Skills**|**Observability**|**Sim**<br>**Steps**_↓_|**Success**<br>**Rate**_↑_|**Completion**<br>**Rate**_↑_|**Planning**<br>**Cycles**_↓_|
|---|---|---|---|---|---|---|---|
|Heuristic-Expert|Centralized|Oracle|Full|1184.74 _±_ 22.88|0.69 _±_ 0.02|0.89 _±_ 0.01|N/A|
|ReAct|Centralized|Oracle|Full|1348.71 _±_ 53.34|0.67 _±_ 0.02|0.86 _±_ 0.01|21.14 _±_ 0.59|
|ReAct|Centralized|Oracle|Partial|2590.82 _±_ 90.71|0.56 _±_ 0.02|0.80 _±_ 0.01|25.57 _±_ 0.64|
|ReAct|Decentralized|Oracle|Partial|3353.33 _±_ 70.03|0.63 _±_ 0.02|0.84 _±_ 0.01|17.38 _±_ 0.33|
|ReAct-Tools|Decentralized|Oracle|Partial|3810.15 _±_ 86.52|0.61 _±_ 0.02|0.83 _±_ 0.01|25.79 _±_ 0.41|
|ReAct + RAG|Decentralized|Oracle|Partial|3489.18 _±_ 79.54|0.62 _±_ 0.02|0.83 _±_ 0.01|17.55 _±_ 0.38|
|Finetuned|Decentralized|Oracle|Partial|3460.60 _±_ 78.33|0.51 _±_ 0.02|0.78 _±_ 0.01|14.73 _±_ 0.25|
|ReAct|Decentralized|Learned|Partial|5905.88 _±_ 162.35|0.50 _±_ 0.02|0.76 _±_ 0.01|24.30 _±_ 0.60|
|ReAct-Single|Single Agent|Oracle|Partial|2632.30 _±_ 60.04|0.68 _±_ 0.01|0.85 _±_ 0.01|21.28 _±_ 0.37|
|ReAct-Single|Single Agent|Oracle|Full|1559.73 _±_ 36.02|0.73 _±_ 0.01|0.88 _±_ 0.01|21.06 _±_ 0.38|
|ReAct-8B|Decentralized|Oracle|Partial|4100.21 _±_ 98.97|0.51 _±_ 0.02|0.77 _±_ 0.01|27.65 _±_ 0.52|



**Table 11 Baseline results on PARTNR test set.** We measure performance using simulation steps required to finish the episode, success rate and completion rate on the tasks, and the average number of planning cycles performed by the planner. Mean and standard error are reported over the test set. Heuristic centralized expert has access to the task evaluation function. 

36 

|Method|Sim Steps _↓_|Task Ofoading_↑_|Extraneous Efort_↓_|Exploration Efciency_↓_|
|---|---|---|---|---|
|Decentralized|3295.20 _±_76.27|0.596 _±_0.01|0.21 _±_0.01|994.88 _±_24.890|
|Centralized|2298.13 _±_61.39|0.49_±_0.01|0.04 _±_0.004|684.06 _±_21.71|
|Single agent|2519.02 _±_57.48|-|0.047 _±_0.01|1121.65 _±_31.256|



**Table12 AnalysisofcollaborationcharacteristicsforLLMagents.** Average and standard errors for task offloading, extraneous effort, and exploration efficiency are reported over the successful episodes from the validation set for LLM agents using ReAct approach in partially observable setting. 

|Method|Success Rate|Success Rate|per Task-type||
|---|---|---|---|---|
||Constraint-free|Spatial|Temporal|Object states|
|Decentralized|0.82 _±_0_._02|0.82_±_0_._02|0.60_±_0_._03|0.66_±_0_._03|
|Centralized|0.84 _±_0_._02|0.85_±_0_._02|0.59_±_0_._03|0.66_±_0_._03|
|Single agent|0.85_±_0_._02|0.81_±_0_._02|0.58_±_0_._03|0.68_±_0_._03|



**Table 13 Task performance per task type.** Average and standard errors of task success rate for episodes from the validation set categorized by task type. Performance is shown for LLM agents that use ReAct approach in partially observable setting. 

steps, in centralized and decentralized settings. By computing the average number of exploration steps taken before the first object is picked up for a task, we find that single agents require on average 127 steps more to locate objects compared to multi-agent. However, in centralized setting, where the co-ordination between agents is better owning to a single LLM co-ordinating the actions of both agents, shows higher gains in such efficiency as compared to decentralized settings. The challenge LLMs face in coordinating exploration in multi-agent settings also negatively impacts human-LLM team performance in our HITL experiments when paired with humans (Table 3). 

**Poor co-ordination also leads to wasted effort and more steps to complete the tasks than solo execution.** Despite multiple agents working together, agents take longer to complete the tasks in decentralized settings as compared to solo execution owing to poor co-ordination. Poor co-ordination is further highlighted by extraneous effort, which increases by 300% in decentralized settings as compared to solo execution. The agents often repeat parts of the task – unsure of whether the other agent really executed that part, and sometimes even undo successfully completed tasks. 

**The robot is able to offload more than half of the tasks from the human partner.** The human-robot team takes longer to complete the task in decentralized setting, however, the robot offloads 60% tasks from the human partner, reducing their load of task execution. This highlights the potential of robots assisting humans more effectively as LLMs continue to advance in reasoning, coordination, and planning capabilities. 

**LLMsstruggletoreasonabouttemporalconstraintsandagentcapabilitieswhileplanningPARTNRtasks.** Constraintfree and spatial tasks in **PARTNR** require the LLMs to reason about only the final configuration and states of objects. Instead, the temporal tasks in **PARTNR** require tracking states of one or more objects over the entire episode, making them challenging (Table 13). Likewise, heterogeneous tasks necessitate reasoning about task distribution conditioned on each agent’s capabilities, which make them challenging. 

## **A.13 Human-in-the-loop Evaluation for PARTNR tasks** 

## **A.13.1 HITL Interface and Web Hosting** 

We adapt the existing human-in-the-loop (HITL) infrastructure from Habitat 3.0 (Puig et al., 2024) to support a more robust server-client architecture, with the server component hosted on AWS. Habitat3.0 HITL includes the ability to extend functionality to resource-constrained environments such as web browsers and VR devices, making the platform versatile for different user needs and experimental setups. Figure 13 shows our HITL system running on a web browser. Detailed interface is shown in Figure 14. Our adaptation to AWS hosting is crucial for handling multiple clients simultaneously, especially non-experts without access to 

37 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0038-00.png)


**Figure 13 HITL on Web-browser.** Our HITL system can be deployed on web browsers enabling large-scale collection. 

powerful machines or large Habitat datasets. The server-client architecture not only enhances scalability but also ensures flexibility, allowing the system to accommodate a variety of operating systems and hardware platforms. Furthermore, the system includes a matchmaking service that enables pairing participants for multi-user sessions. When a participant requests a task, they are redirected to a "lobby screen" where they are instructed to wait until the next participant arrives. 

## **A.13.2 Participant Recruitment and Quality Control** 

The study was performed through a 3rd party company specializing in large-scale annotations. The participants were recruited and compensated for their time by this company. The participants were English speakers, 18 years or older from the United States. For training them, we created a project and task overview video and guidelines. The participants were instructed to complete the tasks correctly and efficiently by themselves or with a partner. The participants also went through a tutorial where they performed some tasks to get acquainted with the interface before performing the main tasks. Each task took on average 3-5 minutes to complete. We recruited 129 non-expert participants in total. 

## **Filtering data:** 

Each task was completed up to 3 times, in all settings, until deemed successful through task evaluation (conducted online as the participants complete the tasks). With the Failure explanation output from the evaluation function, users were also given a natural language feedback at the end of an episode, describing what went wrong in an episode, if the task was not successful. For example, if actions were completed in the wrong order, or a spatial constraint like next-to was not respected appropriately. Users can use this information to update their actions in the next episode, improving their overall performance over time. By giving 4 tutorial episodes before the start of each experiment, and also a tutorial episode in the same house, we ensure that the users are deeply familiar with the tool and tasks before starting the actual experiment. 

Once each task was completed successfully at least once or retried 3 times, we assimilate data by selecting one of the tries per episode: we select the successful try for successfully completed episodes, and the highest percent complete try for unsuccessful episodes. This collection of 1000 episodes for test and val is used for collecting performance statistics described in the next section. 

## **A.13.3 HITL experimental details** 

## **Task Evaluation and Data Collection:** 

Using the enhanced server-client setup, we conduct comprehensive evaluations of 1000 tasks from the validation set, and 1000 tasks from the test set. These evaluations are designed to capture data in both single-user and multi-user scenarios. In the single-user setting, participants individually control a human agent within the simulator using traditional keyboard and mouse inputs, completing tasks without any external assistance. 

38 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks_images/iclr_2025_PARTNR_A_Benchmark_for_Planning_and_Reasoning_in_Embodied_Multi-agent_Tasks.pdf-0039-00.png)


**Figure 14 HITL Interface.** Participants control human and robot agents using keyboard/mouse controls to complete the **PARTNR** tasks. Each participant has access to their partner’s viewpoint and thereby current actions via a small viewport on the bottom right. 

Conversely, the multi-user setting involves collaborative efforts where two participants work together, each controlling either a human or a robot agent. This collaborative approach is specifically designed to study the dynamics between multiple users and to evaluate whether such collaborations lead to more efficient task completion compared to single-user efforts in the **PARTNR** tasks. 

## **Human-AI Collaboration Experiment:** 

In addition to human-only interactions, we conduct experiments where a human collaborates with a robot controlled by a Language Model (LLM), specifically using the ReAct and Finetuned models as described in Section 4.1. The primary goal of these experiments is to evaluate the effectiveness of LLM-controlled agents in real-time collaboration with non-expert humans who have not previously interacted with these AI models. We track and compare the success rates (SR) and the percentage of tasks completed (PC) across various settings including single-user, multi-user, and human-AI collaborations. The results of these experiments are systematically documented and analyzed in Table 14, providing insights into the collaborative capabilities of human-AI pairs. 

To enable this setting, we host LLMs on AWS nodes, and query them intermittently based on robot observations and actions. The HITL server now queries two clients - a human and a LLM. The human client sends commands to control the human agent, and the LLM client uses the hosted LLM to control the robot. Different baselines need different numbers of GPUs to keep the inference time reasonable. For hosting and 70B models, as used by the ReAct baselines, we use 4 A100 GPUs per model. For hosting a smaller 8B model used by the Finetuned baselines, we use 1 A100 GPU. This makes deploying smaller Finetuned models much scalable than larger 70B models. 

## **Efficiency and Task Offloading Metrics:** 

To further understand the efficiency of task completion across different experimental setups, we measure several key performance metrics beyond success rate and percent complete. These include the number of steps taken to complete tasks and the exploration efficiency, which is assessed by the number of steps participants take to pick the first object. Additionally, we evaluate the extraneous effort by noting actions that do not 

39 

|Method|Success<br>Rate _↑_|Percent<br>Complete _↑_|Sim<br>Steps _↓_|Task<br>Ofoading _↑_|Exploration<br>Efciency _↓_|Extraneous<br>Efort _↓_|
|---|---|---|---|---|---|---|
|||Validation 1,000 episodes|||||
|Single-user|0.93 _±_ 0.01|0.96 _±_ 0.00|3046.99 _±_ 80.79|N/A|2459.22 _±_ 26.75|0.09 _±_ 0.01|
|Multi-user|0.93 _±_ 0.01|0.96 _±_ 0.00|2369.55 _±_ 49.33|0.59 _±_ 0.01|1762.47 _±_ 13.99|0.15 _±_ 0.01|
|Human-ReAct|0.91 _±_ 0.01|0.96 _±_ 0.02|4267.71 _±_ 83.40|0.16 _±_ 0.01|2624.39 _±_ 26.05|0.12 _±_ 0.01|
|Human-Finetuned|0.92 _±_ 0.01|0.96 _±_ 0.00|3443.33 _±_ 61.46|0.26 _±_ 0.01|2164.94 _±_ 21.31|0.13 _±_ 0.01|
||||Test 1,000 episodes||||
|Single-user|0.89 _±_ 0.01|0.95 _±_ 0.00|3937.87 _±_ 110.53|N/A|2737.44 _±_ 25.27|0.13 _±_ 0.00|
|Multi-user|0.85 _±_ 0.01|0.95 _±_ 0.00|2667.86 _±_ 58.07|0.60 _±_ 0.01|1889.56 _±_ 14.9=69|0.20 _±_ 0.00|
|Human-ReAct|0.87 _±_ 0.01|0.95 _±_ 0.00|4080.10 _±_ 72.24|0.12 _±_ 0.00|2449.50 _±_ 19.98|0.18 _±_ 0.01|
|Human-Finetuned|0.87 _±_ 0.01|0.95 _±_ 0.00|3403.03 _±_ 62.08|0.26 _±_ 0.01|2162.20 _±_ 19.40|0.20 _±_ 0.01|



**Table 14 Human-in-the-Loop evaluation.** We evaluate the performance of 2-person human teams and human-LLM teams, comparing them to solo human performance on **PARTNR** tasks using metrics described in Section 4.1. The human-LLM teams with SoTA LLMs is _slower_ than solo human. 

contribute directly to task completion. Another critical metric we analyze is the ratio of work completed by the robot, referred to as task offloading. Ideally, in a well-coordinated human-AI team, the task offloading ratio should approach 0.5, indicating an efficient division of labor between the human and the robot. 

## **A.13.4 HITL analysis** 

**Humans are significantly better than LLMs at PARTNR tasks.** Both single and multi-human settings achieve a success rate of 0.93 on **PARTNR** validation tasks, while ReAct without any privileged information only achieves 0.30 (Table 2, row (i)). This indicates a huge gap in LLM planning performance. We observe a slightly lower human performance on the test set (0.89), also in line with a lower LLM performance in Table 11 on this dataset. This indicates that the tasks in our test set are more challenging than the validation set for both humans and LLMs alike, potentially due to human annotations aimed at making them more complex and diverse. 

**Finetuned LLMs perform better than ReAct when coordinating with real humans.** When deployed with real humans-in-the-loop on the validation set, the finetuned model is faster than ReAct at task completion (3275 steps with finetuned versus 4484 with ReAct on the validation set). It is also able to offload more tasks from humans than ReAct (26% with finetuned as compared to 16% with ReAct). This reflects that smaller models with faster inference can improve human experience in real-world deployment. This result is also reflected in the test set where finetuned model outperforms ReAct. Interestingly, the automated eval performance of finetuned is worse than ReAct on the test set, but the HITL performance is better, indicating that faster inference is more critical than task success when working with real humans. 

**LLMs struggle at coordination, hampering human performance.** Despite the Finetuned being faster than ReAct when collaborating with humans, both approaches are _slower_ than the human doing the task alone. In contrast, two humans working together complete the task faster than a single human (2369 steps vs. 3046 with multi- and single-user respectively). This result is in line with the automated evaluation we observed in Table 1, where multi-agent LLMs are also _slower_ than a single-agent LLM. This result further reinforces that LLMs suffer at coordination; while humans are able to coordinate and divide tasks between each other, decentralized LLMs are unable to do so. We observe the same effect in the test set, further reinforcing this finding. 

**LLMs are able to offload tasks from humans.** Despite the aforementioned increase in the number of steps for task completion, robots guided by the finetuned model successfully offload 26% of tasks from humans. This indicates that LLMs can still offer assistance when collaborating with real human partners. Nonetheless, there remains significant potential for improvement. 

**LLM’s inefficiency to explore reduces the team performance when paired with humans.** In multi-user condition, the two humans start in different parts of a house, and explore efficiently to locate task-relevant objects more quickly than a single user – as evidenced by the reduced number of steps before first pick (1762 steps with multi-user vs. 2459 steps for a single user). However, this efficiency is reduced when humans are paired with 

40 

LLMs (2120 steps with finetuned and 2791 steps with ReAct), indicating that LLMs struggle to coordinate at _both_ task completion and exploration. 

## **A.14 Prompts for benchmark task and evaluation** 

## **A.14.1 Task generation prompts** 

- Prompts are similar for the diferent task types, with the primary diference being the in context examples. 

**Constraint-free task generation prompt** You are a system that generates tasks f o r robots to perform with humans . Do not be verbose . Answer the question with no added q u a l i f i c a t i o n s or caveats . Just d i r e c t l y provide the answer in JSON. You w i l l be given a d e s c r i p t i o n of a house with o b j e c t s and f u r n i t u r e and your task i s to provide 5 i n s t r u c t i o n s f o r tasks that a robot and a human could be doing together in that house , using the o b j e c t s and f u r n i t u r e . For each task , provide the i n i t i a l s t a t e of o b j e c t s in the house , the i n s t r u c t i o n that should be performed , and f i n a l s t a t e of the o b j e c t s a f t e r the i n s t r u c t i o n i s performed . The i n i t i a l and f i n a l s t a t e w i l l contain a l i s t of d i c t i o n a r i e s , each with an object type , the number of o b j e c t s of that type , t h e i r l o c a t i o n on a f u r n i t u r e or f l o o r , and the region of the house where they are in e . g . , bedroom . Follow the next p r i n c i p l e s : 1. The i n s t r u c t i o n should be given as i f the human doing the task wanted the robot to perform part of i t . In some cases the task w i l l be done together , in other cases , the human and robot w i l l be doing d i f f e r e n t tasks . 2. The i n i t i a l and f i n a l s t a t e should contain o b j e c t s of d i f f e r e n t types , and sometimes multiple o b j e c t s of a type . 3. Some of the i n s t r u c t i o n s should be semantically rich , in p a r t i c u l a r they should r e f e r to c l a s s e s or groups of o b j e c t s . 4. The i n s t r u c t i o n s shouldn ’ t be d e t a i l e d and explain a l l the steps , but the high - l e v e l . 5. The robot can only rearrange o b j e c t s and open containers , the human can do more tasks e . g . , turn on lamp , clean plates , f i l l up p i t c h e r . 6. The i n s t r u c t i o n should contain a c l e a r goal and at l e a s t two steps a s s o c i a t e d with the goal . 7. Ensure that i n s t r u c t i o n s are d i v e r s e from each other . Some i n s t r u c t i o n s should contain s p a t i a l s p e c i f i e r s such as " next to " , " l e f t " , " r i g h t " , " beside " , " near " , " f r o n t " , " s i d e ". While some other i n s t r u c t i o n s should contain temporal order , which can be s p e c i f i e d using words such as " a f t e r " , " then " , " before " etc . For instance : " F i l l up the k e t t l e and then turn i t on . After that , bring two cups to the dinning table ." 8. The i n s t r u c t i o n s should contain d i v e r s e a c t i o n s such as " turn on/ o f f " , " f i l l " , " c l e a r " , " s e t " etc . and object s t a t e s such as " clean " , " d i r t y " , "open " , " c l o s e " etc . while r e f e r r i n g to o b j e c t s . You w i l l be given a l l the p i e c e s of f u r n i t u r e in the house . You w i l l a l s o be given a l l the d i f f e r e n t types of o b j e c t s that you can use . You can s p e c i f y multiple i n s t a n c e s of an object type . Make sure you i n s t r u c t i o n i n c l u d e s the object types and f u r n i t u r e present in the l i s t below . The house has the f o l l o w i n g rooms , each with the f o l l o w i n g f u r n i t u r e : { house_furniture } You can use the f o l l o w i n g o b j e c t s : { o b j e c t s _ l i s t } Here i s an example with two i n s t r u c t i o n s : JSON_OUTPUT: [ {{ " i n i t i a l s t a t e ": [ {{ " object_type ": "lamp " , "how_many": 1 , " furniture_name ": " table_10 " , " region ": "living_room_1" }} , {{ " object_type ": "book " , "how_many": 3 , " furniture_name ": " table_11 " , " region ": "living_room_1" 

41 

}} , {{ " object_type ": " toy_vehicle " , "how_many": 2 , " furniture_name ": " f l o o r " , " region ": "living_room_1" }} , {{ " object_type ": " toy_cactus " , "how_many": 1 , " furniture_name ": " table_1 " , " region ": "living_room_1" }} ] , " f i n a l s t a t e ": [ {{ " object_type ": "lamp " , "how_many": 1 , " furniture_name ": " table_10 " , " region ": "living_room_1" }} , {{ " object_type ": "book " , "how_many": 3 , " furniture_name ": " shelves_2 " , " region ": "living_room_1" }} , {{ " object_type ": " toy_vehicle " , "how_many": 2 , " furniture_name ": "bed_2" , " region ": "bedroom_1" }} , {{ " object_type ": " toy_cactus " , "how_many": 1 , " furniture_name ": "bed_2" , " region ": "bedroom_1" }} , ] , " i n s t r u c t i o n ": "We need to clean up the l i v i n g room . Move a l l toys and books to the s h e l f in the l i v i n g room . " , " reason ": "The task i n v o l v e s moving multiple o b j e c t s to the s h e l f in the l i v i n g room ." }} , {{ " i n i t i a l s t a t e ": [ {{ " object_type ": " plate " , "how_many": 3 , " furniture_name ": " cabinet_2 " , " region ": " kitchen_1 " }} , {{ " object_type ": " g l a s s " , "how_many": 2 , " furniture_name ": "counter_1 " , " region ": " kitchen_1 " }} , {{ " object_type ": " fork " , "how_many": 5 , " furniture_name ": " cabinet_5 " , " region ": " kitchen_1 " }} , " f i n a l s t a t e ": [ {{ " object_type ": " plate " , "how_many": 2 , " furniture_name ": " table_8 " , " region ": "living_room_1" }} , {{ " object_type ": " g l a s s " , "how_many": 2 , " furniture_name ": " table_8 " , " region ": "living_room_1" }} , {{ 

42 

" object_type ": " fork " , "how_many": 2 , " furniture_name ": " table_8 " , " region ": "living_room_1" }} , ] , " i n s t r u c t i o n ": "Help me s e t up a tab l e f o r dinner in the livingroom f o r 2 people . Place 2 p l a t e s and 2 g l a s s e s on the table . " , " reason ": "The task i n c l u d e s semantically r i c h d e s c r i p t i o n s ( s e t up a t a b l e ) ." }} ] Generate a JSON l i s t with {k} i n s t r u c t i o n s l i k e the examples above . Your output should only be : JSON_OUTPUT: r e s u l t _ l i s t where r e s u l t _ l i s t should be a JSON l i s t with the i n s t r u c t i o n s . Let ’ s think through t h i s c a r e f u l l y , step by step . 

## **Spatial task generation prompt** 

You are a system that generates tasks f o r robots to perform with humans . Do not be verbose . Answer the question with no added q u a l i f i c a t i o n s or caveats . Just d i r e c t l y provide the answer in JSON. 

You w i l l be given a d e s c r i p t i o n of a house with o b j e c t s and f u r n i t u r e and your task i s to provide 5 i n s t r u c t i o n s f o r tasks that a robot and a human could be doing together in that house , using the o b j e c t s and f u r n i t u r e . 

For each task , provide the i n i t i a l s t a t e of o b j e c t s in the house , the i n s t r u c t i o n that should be performed , and f i n a l s t a t e of the o b j e c t s a f t e r the i n s t r u c t i o n i s performed . The i n i t a l and f i n a l s t a t e w i l l contain a l i s t of d i c t i o n a r i e s , each with an object type , the number of o b j e c t s of that type , t h e i r l o c a t i o n on a f u r n i t u r e or f l o o r , and the region of the house where they are in e . g . , bedroom . Follow the next p r i n c i p l e s : 1. The i n s t r u c t i o n should be given as i f the human doing the task wanted the robot to perform part of i t . In some cases the task w i l l be done together , in other cases , the human and robot w i l l be doing d i f f e r e n t tasks . 2. The i n i t i a l and f i n a l s t a t e should contain o b j e c t s of d i f f e r e n t types , and sometimes multiple o b j e c t s of a type . 3. Some of the i n s t r u c t i o n s should be semantically rich , in p a r t i c u l a r they should r e f e r to c l a s s e s or groups of o b j e c t s . 4. The i n s t r u c t i o n s shouldn ’ t be d e t a i l e d and explain a l l the steps , but the high - l e v e l . 5. The robot can only rearrange o b j e c t s and open containers , the human can do more tasks e . g . , turn on lamp , clean plates , f i l l up p i t c h e r . 6. The i n s t r u c t i o n should contain a c l e a r goal and at l e a s t two steps a s s o c i a t e d with the goal . 7. Ensure that i n s t r u c t i o n s are d i v e r s e from each other . 8. All the i n s t r u c t i o n s should contain at l e a s t one of the s p a t i a l s p e c i f i e r s from t h i s l i s t : " next to " , " l e f t " , " r i g h t " , " beside " , " near " , " f r o n t " , " s i d e " . 9. The i n s t r u c t i o n s should contain d i v e r s e a c t i o n s such as " turn on/ o f f " , " f i l l " , " c l e a r " , " s e t " etc . and object s t a t e s such as " clean " , " d i r t y " , "open " , " c l o s e " etc . 

You w i l l be given a l l the p i e c e s of f u r n i t u r e in the house . You w i l l a l s o be given a l l the d i f f e r e n t types of o b j e c t s that you can use . You can s p e c i f y multiple i n s t a n c e s of an object type . Make sure you i n s t r u c t i o n i n c l u d e s the object types and f u r n i t u r e present in the l i s t below . The house has the f o l l o w i n g rooms , each with the f o l l o w i n g f u r n i t u r e : { house_furniture } 

You can use the f o l l o w i n g o b j e c t s : { o b j e c t s _ l i s t } Here i s an example with two i n s t r u c t i o n s : JSON_OUTPUT: [ {{ " i n i t i a l s t a t e ": [ {{ " object_type ": " vase " , "how_many": 1 , " furniture_name ": " table_10 " , " region ": "living_room_1" 

43 

}} , {{ " object_type ": " stuffed_toy " , "how_many": 2 , " furniture_name ": " f l o o r " , " region ": "bedroom_2" }} , {{ " object_type ": " candle " , "how_many": 1 , " furniture_name ": "chest_of_drawers_2 " , " region ": "bedroom_1" }} ] , " f i n a l s t a t e ": [ {{ " object_type ": " vase " , "how_many": 1 , " furniture_name ": " shelves_2 " , " region ": "living_room_1" }} , {{ " object_type ": " stuffed_toy " , "how_many": 2 , " furniture_name ": " shelves_2 " , " region ": "living_room_1" }} , {{ " object_type ": " candle " , "how_many": 1 , " furniture_name ": " shelves_2 " , " region ": "living_room_1" }} ] , " i n s t r u c t i o n ": "Let ’ s decorate ! Put the vase on the s h e l f . Then , s e t a candle and a stuffed_toy on each s i d e of the vase . " , " reason ": "The task i n c l u d e s s p a t i a l c o n s t r a i n t s p e c i f i e d by ’ side ’ . " }} , {{ " i n i t i a l s t a t e ": [ {{ " object_type ": " plate " , "how_many": 3 , " furniture_name ": " cabinet_2 " , " region ": " kitchen_1 " }} , {{ " object_type ": " g l a s s " , "how_many": 2 , " furniture_name ": "counter_1 " , " region ": " kitchen_1 " }} , {{ " object_type ": " fork " , "how_many": 5 , " furniture_name ": " cabinet_5 " , " region ": " kitchen_1 " }} , " f i n a l s t a t e ": [ {{ " object_type ": " plate " , "how_many": 2 , " furniture_name ": " table_8 " , " region ": "living_room_1" }} , {{ " object_type ": " g l a s s " , "how_many": 2 , " furniture_name ": " table_8 " , " region ": "living_room_1" }} , {{ " object_type ": " fork " , "how_many": 2 , " furniture_name ": " table_8 " , " region ": "living_room_1" }} , ] , 

44 

" i n s t r u c t i o n ": "Help me s e t up a tab l e f o r dinner in the livingroom f o r 2 people . Place 2 p l a t e s and 2 g l a s s e s on the table . There should be a fork next to each p l a t e " , " reason ": "The task i n c l u d e s semantically r i c h d e s c r i p t i o n s ( s e t up a t a b l e ) and s p a t i a l c o n s t r a i n t s s p e c i f i e d by the word ’ next ’ . " }} ] Generate a JSON l i s t with {k} i n s t r u c t i o n s l i k e the examples above . Your output should only be : JSON_OUTPUT: r e s u l t _ l i s t where r e s u l t _ l i s t should be a JSON l i s t with the i n s t r u c t i o n s . Let ’ s think through t h i s c a r e f u l l y , step by step . 

**Temporal task generation prompt** 

You are a system that generates tasks f o r robots to perform with humans . Do not be verbose . Answer the question with no added q u a l i f i c a t i o n s or caveats . Just d i r e c t l y provide the answer in JSON. You w i l l be given a d e s c r i p t i o n of a house with o b j e c t s and f u r n i t u r e and your task i s to provide 5 i n s t r u c t i o n s f o r tasks that a robot and a human could be doing together in that house , using the o b j e c t s and f u r n i t u r e . For each task , provide the i n i t i a l s t a t e of o b j e c t s in the house , the i n s t r u c t i o n that should be performed , and f i n a l s t a t e of the o b j e c t s a f t e r the i n s t r u c t i o n i s performed . The i n i t a l and f i n a l s t a t e w i l l contain a l i s t of d i c t i o n a r i e s , each with an object type , the number of o b j e c t s of that type , t h e i r l o c a t i o n on a f u r n i t u r e or f l o o r , and the region of the house where they are in e . g . , bedroom . Follow the next p r i n c i p l e s : 1. The i n s t r u c t i o n should be given as i f the human doing the task wanted the robot to perform part of i t . In some cases the task w i l l be done together , in other cases , the human and robot w i l l be doing d i f f e r e n t tasks . 2. The i n i t i a l and f i n a l s t a t e should contain o b j e c t s of d i f f e r e n t types , and sometimes multiple o b j e c t s of a type . 3. Some of the i n s t r u c t i o n s should be semantically rich , in p a r t i c u l a r they should r e f e r to c l a s s e s or groups of o b j e c t s . 4. The i n s t r u c t i o n s shouldn ’ t be d e t a i l e d and explain a l l the steps , but the high - l e v e l . 5. The robot can only rearrange o b j e c t s and open containers , the human can do more tasks e . g . , turn on lamp , clean plates , f i l l up p i t c h e r . 6. The i n s t r u c t i o n should contain a c l e a r goal and at l e a s t two steps a s s o c i a t e d with the goal . 7. Ensure that i n s t r u c t i o n s are d i v e r s e from each other . 8. All the i n s t r u c t i o n s should contain temporal order , s p e c i f i e d using one of the words from t h i s l i s t : " a f t e r " , "then " , " before " , " f i n a l l y " , " f i r s t ". For instance : " F i l l up the k e t t l e and then turn i t on . After that , bring two cups to the dinning t ab le ." 9. The i n s t r u c t i o n s should contain d i v e r s e a c t i o n s such as " turn on/ o f f " , " f i l l " , " c l e a r " , " s e t " etc . and object s t a t e s such as " clean " , " d i r t y " , "open " , " c l o s e " etc . 

You w i l l be given a l l the p i e c e s of f u r n i t u r e in the house . You w i l l a l s o be given a l l the d i f f e r e n t types of o b j e c t s that you can use . You can s p e c i f y multiple i n s t a n c e s of an object type . Make sure you i n s t r u c t i o n i n c l u d e s the object types and f u r n i t u r e present in the l i s t below . 

The house has the f o l l o w i n g rooms , each with the f o l l o w i n g f u r n i t u r e : { house_furniture } 

You can use the f o l l o w i n g o b j e c t s : { o b j e c t s _ l i s t } Here i s an example with two i n s t r u c t i o n s : JSON_OUTPUT: [ {{ " i n i t i a l s t a t e ": [ {{ " object_type ": "lamp " , "how_many": 1 , " furniture_name ": " table_10 " , " region ": "living_room_1" }} , {{ " object_type ": "book " , 

45 

"how_many": 3 , " furniture_name ": " table_11 " , " region ": "living_room_1" }} , {{ " object_type ": " toy_vehicle " , "how_many": 2 , " furniture_name ": " f l o o r " , " region ": "living_room_1" }} , {{ " object_type ": " toy_cactus " , "how_many": 1 , " furniture_name ": " table_1 " , " region ": "living_room_1" }} ] , " f i n a l s t a t e ": [ {{ " object_type ": "lamp " , "how_many": 1 , " furniture_name ": " table_10 " , " region ": "living_room_1" }} , {{ " object_type ": "book " , "how_many": 3 , " furniture_name ": " shelves_2 " , " region ": "living_room_1" }} , {{ " object_type ": " toy_vehicle " , "how_many": 2 , " furniture_name ": "bed_2" , " region ": "bedroom_1" }} , {{ " object_type ": " toy_cactus " , "how_many": 1 , " furniture_name ": "bed_2" , " region ": "bedroom_1" }} , ] , " i n s t r u c t i o n ": "We need to clean up the l i v i n g room . Move a l l toys to the bedroom and the books to the s h e l f . After that , turn on the lamp in the l i v i n g room . " , " reason ": "The task i n c l u d e s temporal c o n s t r a i n t s s p e c i f i e d by ’ aft er ’ . " }} , {{ " i n i t i a l s t a t e ": [ {{ " object_type ": " plate " , "how_many": 3 , " furniture_name ": " cabinet_2 " , " region ": " kitchen_1 " }} , {{ " object_type ": " g l a s s " , "how_many": 2 , " furniture_name ": "counter_1 " , " region ": " kitchen_1 " }} , {{ " object_type ": " fork " , "how_many": 5 , " furniture_name ": " cabinet_5 " , " region ": " kitchen_1 " }} , " f i n a l s t a t e ": [ {{ " object_type ": " plate " , "how_many": 2 , " furniture_name ": " table_8 " , " region ": "living_room_1" }} , {{ " object_type ": " g l a s s " , "how_many": 2 , " furniture_name ": " table_8 " , 

46 

" region ": "living_room_1" }} , {{ " object_type ": " fork " , "how_many": 2 , " furniture_name ": " table_8 " , " region ": "living_room_1" }} , ] , " i n s t r u c t i o n ": "Help me s e t up a tab l e f o r dinner in the livingroom f o r 2 people . F i r s t place 2 p l a t e s on the table . Then , place g l a s s e s and f o r k s next to each p l a t e . " , " reason ": "The task i n c l u d e s semantically r i c h d e s c r i p t i o n s ( s e t up a t a b l e ) and temporal c o n s t r a i n t s s p e c i f i e d by the words ’ f i r s t ’ and ’ then ’ . " }} ] Generate a JSON l i s t with {k} i n s t r u c t i o n s l i k e the examples above . Your output should only be : JSON_OUTPUT: r e s u l t _ l i s t where r e s u l t _ l i s t should be a JSON l i s t with the i n s t r u c t i o n s . Let ’ s think through t h i s c a r e f u l l y , step by step . 

## **Large-scale task generation prompt** 

You are a system that generates tasks f o r robots to perform with humans . Do not be verbose . Answer the question with no added q u a l i f i c a t i o n s or caveats . D i r e c t l y provide the answer in JSON. You w i l l be given a d e s c r i p t i o n of a house with o b j e c t s and f u r n i t u r e . You w i l l a l s o be given a sample task . Your job i s to modify t h i s sample task so that i t i s a p p l i c a b l e to t h i s house . Here i s an example : Task : [ {{ " i n s t r u c t i o n ": "Move the k e t t l e and the jug from the l i v i n g room to the kitchen and f i l l the k e t t l e with water , then turn on the k e t t l e . F i n a l l y clean the l i v i n g room t a b l e . " , " i n i t i a l _ s t a t e ": [ {{ "number ": 1 , " o b j e c t _ c l a s s e s ": [ " k e t t l e " ] , " furniture_names ": [ " table_0 " ] , " allowed_regions ": [ "living_room_0" ] }} , {{ "number ": 1 , " o b j e c t _ c l a s s e s ": [ " jug " ] , " furniture_names ": [ " table_1 " ] , " allowed_regions ": [ "living_room_0" ] }} ] }} ] JSON_OUTPUT: [ {{ " i n s t r u c t i o n ": "Move the jug , kettle , teapot , and cup from the dining t a b l e to the kitchen and f i l l a l l with water . Turn on the lamp , and clean the dining t a b l e . " , " i n i t i a l _ s t a t e ": [ {{ "number ": 1 , " o b j e c t _ c l a s s e s ": [ " jug " ] , " furniture_names ": [ 

47 

" table_1 " ] , " allowed_regions ": [ "living_room_0" ] }} , {{ "number ": 1 , " o b j e c t _ c l a s s e s ": [ " k e t t l e " ] , " furniture_names ": [ " table_1 " ] , " allowed_regions ": [ "living_room_0" ] }} , {{ "number ": 1 , " o b j e c t _ c l a s s e s ": [ " teapot " ] , " furniture_names ": [ " table_1 " ] , " allowed_regions ": [ "living_room_0" ] }} , {{ "number ": 1 , " o b j e c t _ c l a s s e s ": [ "cup" ] , " furniture_names ": [ " table_0 " ] , " allowed_regions ": [ "living_room_0" ] }} , {{ "number ": 1 , " o b j e c t _ c l a s s e s ": [ "lamp" ] , " furniture_names ": [ " table_1 " ] , " allowed_regions ": [ "living_room_0" ] }} ] }} ] The house has the f o l l o w i n g rooms , each with the f o l l o w i n g f u r n i t u r e : { house_furniture } You can use the f o l l o w i n g o b j e c t s : { o b j e c t s _ l i s t } Here i s the task : Task : [ { task } ] Modify t h i s task to generate a JSON l i s t of tasks , using the rooms and f u r n i t u r e from t h i s house . Just change the o b j e c t s and f u r n i t u r e . Make sure i n i t i a l and f i n a l l o c a t i o n s of o b j e c t s are d i f f e r e n t . Include a c t i o n s such as turn on/ off , f i l l and clean . Make tasks multi - step , c o n s i s t i n g of more than one object and action . Your output should only be : 

48 

JSON_OUTPUT: r e s u l t _ l i s t where r e s u l t _ l i s t should be a JSON l i s t with the tasks . 

## **A.15 Evaluation Generation Prompts** 

Here we share the LLM prompts used for proposition generation, temporal constraint prediction, and argument constraint prediction. In each, the task to accomplish is described in the system prompt and between 6-13 few-shot examples follow. 

**Evaluation Generation: Propositions** 

Source : system 

You w i l l be given an i n s t r u c t i o n d e s c r i b i n g a household task and a d e s c r i p t i o n of the i n i t i a l s t a t e of the house . You w i l l d e f i n e a l i s t of python f u n c t i o n s that must be s a t i s f i e d f o r the task to be marked complete . You can c a l l the f o l l o w i n g f u n c t i o n s : - is_on_top ( object_names : str , furniture_name : s t r ) # any object in object_names i s on top of a f u r n i t u r e - i s _ i n s i d e ( object_names : str , furniture_name : s t r ) # any object in object_names i s i n s i d e of a f u r n i t u r e - is_in_room ( object_names : str , room_name : s t r ) # any object in object_names i s in a room - is_on_floor ( object_names : s t r ) # any object in object_names i s on the f l o o r - is_next_to ( objects_a : str , objects_b : s t r ) # any object in objects_a i s next to any object in objects_b - is_clean ( object_names : s t r ) # any object in object_names i s clean - is_dirty ( object_names : s t r ) # any object in object_names i s d i r t y - i s _ f i l l e d ( object_names : s t r ) # any object in object_names i s f i l l e d , l i k e with a l i q u i d - is_empty ( object_names : s t r ) # any object in object_names i s empty - is_powered_on ( object_names : s t r ) # any object in object_names i s powered on - is_powered_off ( object_names : s t r ) # any object in object_names i s powered o f f 

Objects in object_names can be expressed as i t appears in the o b j e c t s l i s t (" stuffed_toy_1 ") or as an OR of object names (" stuffed_toy_1 or stuffed_toy_2 ") . A furniture_name can be expressed as i t appears in the f u r n i t u r e l i s t ( e . g . " t a b l e ") or as i t appears in the furniture - room r e l a t i o n (" t a b l e in living_room ") . 

Only use the f u n c t i o n s l i s t e d above . Each function should t e s t a s i n g l e o b j e c t s / f u r n i t u r e /room r e l a t i o n . I f the i n s t r u c t i o n i s ambiguous such that multiple o b j e c t s could be used to s a t i s f y a function ( an OR r e l a t i o n s h i p ) , then include a l l p o s s i b l e o b j e c t s . Define as many f u n c t i o n s as necessary . Write each function on i t s own l i n e . I t i s e s s e n t i a l to wrap each function in d e l i m i t e r s [FN] and [/FN ] . End your f u n c t i o n s with the token [END] . 

Let ’ s see some examples . Suppose the i n i t i a l s t a t e i s : Objects : * pants_1 * shirt_1 * shirt_2 * shirt_3 Furniture : * washer_dryer * t ab l e Rooms : * laundryroom Object - Furniture -Room Relations : * pants_1 on table in laundryroom * shirt_1 on table in laundryroom * shirt_2 on f l o o r in laundryroom Furniture -Room Relations : * washer_dryer in laundryroom * t ab l e in laundryroom I n s t r u c t i o n "Put the pants on the table " means [FN] is_on_top (" pants_1 " , " table in laundryroom ") [/FN] I n s t r u c t i o n "Put the pants in the washer" means [FN] i s _ i n s i d e (" pants_1 " , "washer_dryer in laundryroom ") [/FN] 

49 

I n s t r u c t i o n "Put a s h i r t in the washer" means [FN] i s _ i n s i d e (" shirt_1 or shirt_2 or shirt_3 " , "washer_dryer in laundryroom ") [/FN] I n s t r u c t i o n "Put a l l the s h i r t s in the washer" means [FN] i s _ i n s i d e (" shirt_1 " , "washer_dryer in laundryroom ") [/FN] [FN] i s _ i n s i d e (" shirt_2 " , "washer_dryer in laundryroom ") [/FN] [FN] i s _ i n s i d e (" shirt_3 " , "washer_dryer in laundryroom ") [/FN] I n s t r u c t i o n "Put a s h i r t on the f l o o r next to the pants " means [FN] is_on_floor (" shirt_1 or shirt_2 or shirt_3 ") [/FN] [FN] is_next_to (" shirt_1 or shirt_2 or shirt_3 " , "pants_1 ") [/FN] I n s t r u c t i o n " Bring a s h i r t to the laundryroom" means [FN] is_in_room (" shirt_1 or shirt_2 or shirt_3 " , "laundryroom ") [/FN] <step> Source : user Objects : * casserole_0 * squeezer_0 Furniture : * cabinet * t ab l e Rooms : * bathroom * hallway * kitchen * living_room Object - Furniture -Room Relations : * casserole_0 on cabinet in kitchen * squeezer_0 on cabinet in kitchen Furniture -Room Relations : * t ab l e in bathroom * t ab l e in hallway * t ab l e in living_room I n s t r u c t i o n : "Get the c a s s e r o l e and squeezer from the kitchen and put them on the t a b l e in the l i v i n g room ." <step> Source : a s s i s t a n t [FN] is_on_top (" casserole_0 " , " table in living_room ") [/FN] [FN] is_on_top (" squeezer_0 " , " table in living_room ") [/FN] [END] <step> Source : user The i n i t i a l s t a t e i s : Objects : * toy_truck_1 * stuffed_toy_1 * stuffed_toy_2 Furniture : * t ab l e * bed * cabinet Rooms : * bedroom * other_room * living_room Object - Furniture -Room Relations : * toy_truck_1 on bed in bedroom * stuffed_toy_1 on f l o o r in bedroom * stuffed_toy_2 on f l o o r in other_room Furniture -Room Relations : * t ab l e in living_room I n s t r u c t i o n : " Bring a toy from the bedroom and put i t in the l i v i n g room . " <step> Source : a s s i s t a n t [FN] is_in_room (" toy_truck_1 or stuffed_toy_1 " , " living_room ") [/FN] [END] <step> Source : user 

50 

The i n i t i a l s t a t e i s : Objects : * cushion_1 * fork_1 * plate_1 Furniture : * bed * couch * counter Rooms : * bedroom * living_room Object - Furniture -Room Relations : * cushion_1 on couch in living_room * fork_1 on couch in living_room * plate_1 on couch in living_room Furniture -Room Relations : * bed in bedroom * couch in living_room * counter in kitchen 

I n s t r u c t i o n : " Bring the dishes back to the kitchen counter . After that , put the cushion on the bed ." <step> Source : a s s i s t a n t [FN] is_on_top (" plate_1 " , " counter in kitchen ") [/FN] [FN] is_on_top (" fork_1 " , " counter in kitchen ") [/FN] [FN] is_on_top (" cushion_1 " , "bed in bedroom ") [/FN] [END] <step> Source : user The i n i t i a l s t a t e i s : Objects : * candle_0 * fork_0 * toy_food_0 Furniture : * counter * t ab l e Rooms : * bedroom * kitchen * living_room Object - Furniture -Room Relations : * candle_0 on table in living_room * fork_0 on counter in kitchen * toy_food_0 on counter in kitchen Furniture -Room Relations : * t ab l e in bedroom * counter in kitchen * t ab l e in living_room I n s t r u c t i o n : "Put the toy food on the ta b l e where the candle is , and place the fork on the t a b l e beside the toy food ." <step> Source : a s s i s t a n t [FN] is_on_top (" toy_food_0 " , " table in living_room ") [/FN] [FN] is_on_top (" fork_0 " , " table in living_room ") [/FN] [FN] is_next_to (" fork_0 " , "toy_food_0 ") [/FN] [END] <step> Source : user Objects : * toy_airplane_0 * toy_airplane_1 Furniture : * t ab l e Rooms : * hallway * living_room Object - Furniture -Room Relations : * toy_airplane_0 on table in hallway * toy_airplane_1 on table in hallway Furniture -Room Relations : * t ab l e in bathroom * t ab l e in hallway * t ab l e in living_room 

51 

I n s t r u c t i o n : " Bring a toy a i r p l a n e to the l i v i n g room and put i t in on the t a b l e ." <step> Source : a s s i s t a n t [FN] is_on_top (" toy_airplane_0 or toy_airplane_1 " , " t a b l e in living_room ") [/FN] [END] <step> Source : user Objects : * kettle_0 (empty , d i r t y ) Furniture : * t ab l e Rooms : * living_room Object - Furniture -Room Relations : * kettle_0 on table in living_room * toy_airplane_0 on table in hallway * toy_airplane_1 on table in hallway Furniture -Room Relations : * t ab l e in living_room I n s t r u c t i o n : " First , make sure the k e t t l e i s clean . Then , f i l l the k e t t l e with water and turn i t on ." <step> Source : a s s i s t a n t [FN] is_clean (" kettle_0 ") [/FN] [FN] i s _ f i l l e d (" kettle_0 ") [/FN] [FN] is_powered_on (" kettle_0 ") [/FN] [FN] is_on_top (" toy_airplane_0 or toy_airplane_1 " , " t a b l e in living_room ") [/FN] [END] {TEMPLATE_EXAMPLE} <step> Source : user The i n i t i a l s t a t e i s : {INIT_STATE} I n s t r u c t i o n : "{INSTRUCTION}" <step> Source : a s s i s t a n t Destination : user [FN] 

**Evaluation Generation: Temporal** 

Source : system 

You w i l l be given an i n s t r u c t i o n d e s c r i b i n g a task to perform in a house and a s e t of p r o p o s i t i o n s that d e f i n e whether the task was done s u c c e s s f u l l y . The task i n s t r u c t i o n may say that c e r t a i n p r o p o s i t i o n s should be completed before others (" then " , " a f t e r " , " f i n a l l y ") . Your job i s to write python code that groups the p r o p o s i t i o n s in the order in which they must be completed . The p r o p o s i t i o n s are well - defined python f u n c t i o n s that return a boolean value . You w i l l be given a l i s t of p r o p o s i t i o n s where index i corresponds to the i t h p r o p o s i t i o n . To s o l v e the task , d e f i n e the v a r i a b l e proposition_order , which groups p r o p o s i t i o n s together that can be completed in any order . Each pr o p os i ti o n group appearing in proposition_order must be s a t i s f i e d before the group that comes a f t e r i t . For example , proposition_order = [ [ 0 , 1] ] means that p r o p o s i t i o n s 0 and 1 can be completed in any order . This example proposition_order = [ [ 0 ] , [ 1 ] ] means that the pr o po s it i o n 0 must be completed before p r o p o s i t i o n 1 . This example proposition_order = [ [ 0 , 1 ] , 

52 

[ 2 ] ] means that p r o p o s i t i o n s 0 and 1 can be completed in e i t h e r order , but p r o p o s i t i o n 2 must be completed a f t e r . Start by assuming that p r o p o s i t i o n s can be completed in any order . Order matters i f the i n s t r u c t i o n i n c l u d e s time ordering words such as "then " , " f i n a l l y " , or " a f t e r " . In t h i s case , p r o p o s i t i o n s should be in multiple groups . Double check that the index f o r each p r o p o s i t i o n i s included in proposition_order . <step> Source : user I n s t r u c t i o n : " Bring an apple to the kitchen table , then bring an orange to the kitchen counter ." p r o p o s i t i o n s = [ is_on_top ( [ " apple_1 " ] , [" table_4 " ] ) , is_on_top ( [ " orange_1 " ] , [" counter_0 " ] ) ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 ] , [ 1 ] ] <step> Source : user I n s t r u c t i o n : "Put the toy v e h i c l e and the water b o t t l e in the l i v i n g room . Next , return the dish to the kitchen ." p r o p o s i t i o n s = [ is_in_room ( [ " toy_truck_1 " ] , [" living_room_1 " ] ) , is_in_room ( [ " cup_0 " ] , [" living_room_1 " ] ) , is_in_room ( [ " bowl_2 " ] , [" kitchen " ] ) ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 , 1 ] , [ 2 ] ] <step> Source : user I n s t r u c t i o n : "Put an apple on the bench in the entryway . Also move the broom to the c l o s e t . " p r o p o s i t i o n s = [ is_on_top ( [ " apple_0 " , "apple_1 " ] , [" bench_2 " ] ) , i s _ i n s i d e ( [ " broom_0 " ] , [" closet_0 " ] ) ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 , 1] ] <step> Source : user I n s t r u c t i o n : " Bring me the toy truck from the bedroom and put i t in the l i v i n g room . Then put two apples on the kitchen table ." p r o p o s i t i o n s = [ is_in_room ( [ " toy_truck_1 " ] , [" living_room_1 " ] ) , is_on_top ( [ " apple_1 " ] , [" table_1 " ] ) , is_on_top ( [ " apple_2 " ] , [" table_1 " ] ) ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 ] , [ 1 , 2] ] <step> Source : user 

53 

I n s t r u c t i o n : " Bring the dishes back to the kitchen counter . After that , put the cushions on the bed . " p r o p o s i t i o n s = [ is_on_top ( [ " plate_1 " ] , [" counter_1 " , "counter_2 " , "counter_3 " ] ) , is_on_top ( [ " fork_1 " ] , [" counter_1 " , "counter_2 " , "counter_3 " ] ) , is_on_top ( [ " spoon_0 " ] , [" counter_1 " , "counter_2 " , "counter_3 " ] ) , is_on_top ( [ " cushion_0 " ] , [" bed_1" , "bed_2 " ] ) , is_on_top ( [ " cushion_1 " ] , [" bed_1" , "bed_2 " ] ) , ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 , 1 , 2 ] , [ 3 , 4] ] <step> Source : user I n s t r u c t i o n : " Bring the dishes back to the kitchen counter . Put the cushion on the bed . Then , move the cushions to the kitchen ." p r o p o s i t i o n s = [ is_on_top ( [ " plate_1 " ] , [" counter_1 " , "counter_2 " , "counter_3 " ] ) , is_on_top ( [ " fork_1 " ] , [" counter_1 " , "counter_2 " , "counter_3 " ] ) , is_on_top ( [ " cushion_0 " ] , [" bed_1" , "bed_2 " ] ) , is_on_top ( [ " cushion_1 " ] , [" bed_1" , "bed_2 " ] ) , is_in_room ( [ " cushion_0 " ] , [" bedroom_0 " ] ) , is_in_room ( [ " cushion_1 " ] , [" bedroom_0 " ] ) , ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 , 1 , 2 , 3 ] , [ 4 , 5] ] <step> Source : user I n s t r u c t i o n : "Move the c l o t h e s from the bedroom to the washer . After that , Put the cushion on the bed . Finally , put the book in the l i v i n g room ." p r o p o s i t i o n s = [ is_on_top ( [ " shirt_1 " ] , [" washer_dryer_1 " ] ) , is_on_top ( [ " shirt_2 " ] , [" washer_dryer_1 " ] ) , is_on_top ( [ " pants_1 " ] , [" washer_dryer_1 " ] ) , is_on_top ( [ " cushion_1 " ] , [" bed_1" , "bed_2 " ] ) , is_in_room ( [ " book_1 " ] , [" living_room_1 " ] ) ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 , 1 , 2 ] , [ 3 ] , [ 4 ] ] <step> Source : user I n s t r u c t i o n : " First , move the spoon and k e t t l e from the kitchen to the l i v i n g room and place them next to each other . Then , place the toy food in the kitchen cabinet . " p r o p o s i t i o n s = [ is_on_top ( [ ’ spoon_0 ’ ] , [ ’ table_1 ’ , ’ table_2 ’ , ’ table_3 ’ , ’ table_4 ’ , ’ table_5 ’ ] ) , is_on_top ( [ ’ kettle_0 ’ ] , [ ’ table_1 ’ , ’ table_2 ’ , ’ table_3 ’ , ’ table_4 ’ , ’ table_5 ’ ] ) , is_next_to ( [ ’ spoon_0 ’ ] , [ ’ kettle_0 ’ ] ) , i s _ i n s i d e ( [ ’ toy_food_0 ’ ] , [ ’ cabinet_0 ’ ] ) ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 , 1 , 2 ] , [ 3 ] ] 

54 

<step> Source : user I n s t r u c t i o n : " First , move the phone stand from the bedroom to the l i v i n g room and place i t on the t ab l e next to the lamp . Then , move the f i l e s o r t e r from the l i v i n g room to the bedroom and place i t on the tab le next to the phone stand ." p r o p o s i t i o n s = [ is_on_top ( [ ’ phone_stand_0 ’ ] , [ ’ table_1 ’ , ’ table_2 ’ , ’ table_3 ’ ] ) , is_next_to ( [ ’ phone_stand_0 ’ ] , [ ’ lamp_0 ’ ] ) , is_on_top ( [ ’ file_sorter_0 ’ ] , [ ’ table_6 ’ ] ) , is_next_to ( [ ’ file_sorter_0 ’ ] , [ ’ phone_stand_0 ’ ] ) ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 , 1 ] , [ 2 , 3] ] <step> Source : user I n s t r u c t i o n : "Help me move the candle and hand towel to the kitchen counter . Place them next to each other . Then , place the spatula and c - clamp on the bedside t a b l e next to each other ." p r o p o s i t i o n s = [ is_on_top ( [ ’ candle_0 ’ ] , [ ’ counter_0 ’ ] ) , is_on_top ( [ ’ hand_towel_0 ’ ] , [ ’ counter_0 ’ ] ) , is_next_to ( [ ’ candle_0 ’ ] , [ ’ hand_towel_0 ’ ] ) , is_on_top ( [ ’ spatula_0 ’ ] , [ ’ table_6 ’ ] ) , is_on_top ( [ ’ c - clamp_0 ’ ] , [ ’ table_6 ’ ] ) , is_next_to ( [ ’ spatula_0 ’ ] , [ ’ c - clamp_0 ’ ] ) ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 , 1 , 2 ] , [ 3 , 4 , 5] ] <step> Source : user I n s t r u c t i o n : " First , move the dog bowl , then the placemat , and f i n a l l y a plush toy from the l i v i n g room to the bench in the hallway . Place them next to each other ." p r o p o s i t i o n s = [ is_on_top ( [ ’ dog_bowl_0 ’ ] , [ ’ bench_0 ’ ] ) , is_on_top ( [ ’ placemat_0 ’ ] , [ ’ bench_0 ’ ] ) , is_next_to ( [ ’ placemat_0 ’ ] , [ ’ dog_bowl_0 ’ ] ) , is_on_top ( [ ’ plush_toy_0 ’ ] , [ ’ bench_0 ’ ] ) , is_next_to ( [ ’ plush_toy_0 ’ ] , [ ’ placemat_0 ’ ] ) ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 ] , [ 1 , 2 ] , [ 3 , 4] ] <step> Source : user I n s t r u c t i o n : "Move the t o a s t e r and the bread from the pantry to the kitchen and turn the t o a s t e r on , then f i l l the k e t t l e ." p r o p o s i t i o n s = [ is_in_room ( [ ’ toaster_0 ’ ] , [ ’ kitchen_0 ’ ] ) , is_in_room ( [ ’ bread_0 ’ ] , [ ’ kitchen_0 ’ ] ) , is_powered_on ( [ ’ toaster_0 ’ ] ) , i s _ f i l l e d ( [ ’ kettle_0 ’ ] ) ] <step> Source : a s s i s t a n t proposition_order = [ [ 0 , 1 , 2 ] , [ 3 ] 

55 

] <step> Source : user I n s t r u c t i o n : "{INSTRUCTION}" {PROPOSITIONS} 

<step> Source : a s s i s t a n t Destination : user proposition_order = [ 

## **Evaluation Generation: Argument Constraints** 

Source : system 

You w i l l be given a task to perform in a house , and a s e t of p r o p o s i t i o n s that d e f i n e whether the task was done s u c c e s s f u l l y . The task i s performed by a human and robot . The task i n s t r u c t i o n may imply c o n s t r a i n t s such that c e r t a i n groups of p r o p o s i t i o n s should be s a t i s f i e d by the same argument or unique arguments . Your job i s to write python code that d e f i n e s these c o n s t r a i n t s . 

The p r o p o s i t i o n s are well - defined python f u n c t i o n s that return a boolean value . You w i l l be given a l i s t of p r o p o s i t i o n s where index i corresponds to the i t h p r o p o s i t i o n . To s o l v e the task , d e f i n e the v a r i a b l e tie_constraints , which i s a l i s t of c o n s t r a i n t s which can be empty . The c o n s t r a i n t s you can use are : SameArgConstraint ( proposition_indices : L i s t [ i n t ] , # i n d i c e s of p r o p o s i t i o n s that t h i s c o n s t r a i n t a p p l i e s to arg_index : L i s t [ i n t ] , # i n d i c e s of arguments that should be matched on ) DifferentArgConstraint ( proposition_indices : L i s t [ i n t ] , # i n d i c e s of p r o p o s i t i o n s that t h i s c o n s t r a i n t a p p l i e s to arg_index : L i s t [ i n t ] , # i n d i c e s of arguments that should be matched on ) Here are some examples : SameArgConstraint ( [ 0 , 1 ] , [ 0 , 0 ] ) # means that p r o p o s i t i o n s at index 0 and 1 must have a matching value in the f i r s t argument . DifferentArgConstraint ( [ 0 , 1 ] , [ 0 , 0 ] ) # means that p r o p o s i t i o n s at index 0 and 1 must have d i f f e r e n t values in the f i r s t argument . SameArgConstraint ( [ 0 , 1 ] , [ 1 , 1 ] ) # means that p r o p o s i t i o n s at index 0 and 1 must have a matching value in the second argument . I f no c o n s t r a i n t s apply to the given i n s t r u c t i o n , j u s t write an empty l i s t . <step> Source : user I n s t r u c t i o n : " Bring an apple and an orange to a t a b l e in the kitchen ." p r o p o s i t i o n s = [ is_on_top ( [ " apple_1 " ] , [" table_3 " , " table_4 " ] ) , is_on_top ( [ " orange_1 " ] , [" table_3 " , " table_4 " ] ) ] <step> Source : a s s i s t a n t t i e _ c o n s t r a i n t s = [ SameArgConstraint ( [ 0 , 1 ] , [ 1 , 1 ] ) ] <step> Source : user I n s t r u c t i o n : "Put the toy v e h i c l e in the l i v i n g room and return the dish to the kitchen . " p r o p o s i t i o n s = [ is_in_room ( [ " toy_truck_1 " ] , [" living_room_1 " ] ) , is_in_room ( [ " bowl_2 " ] , [" kitchen " ] ) ] <step> Source : a s s i s t a n t t i e _ c o n s t r a i n t s = [ ] <step> Source : user 

56 

I n s t r u c t i o n : " Place the book on the s h e l f in the bedroom . Place the p i c t u r e frame next to i t ." p r o p o s i t i o n s = [ is_on_top ( [ " book_1 " ] , [" shelves_0 " , " shelves_1 " ] ) , is_on_top ( [ " picture_frame_0 " ] , [" shelves_0 " , " shelves_1 " ] ) , is_next_to ( [ " picture_frame_0 " ] , [" book_1 " ] ) ] <step> Source : a s s i s t a n t t i e _ c o n s t r a i n t s = [ SameArgConstraint ( [ 0 , 1 ] , [ 1 , 1 ] ) ] <step> Source : user I n s t r u c t i o n : " Place each candle on i t s own t a b l e in the l i v i n g room . " p r o p o s i t i o n s = [ is_on_top ( [ " candle_0 " ] , [" table_0 " , " table_2 " , " table_6 " ] ) , is_on_top ( [ " candle_1 " ] , [" table_0 " , " table_2 " , " table_6 " ] ) , is_on_top ( [ " candle_2 " ] , [" table_0 " , " table_2 " , " table_6 " ] ) ] <step> Source : a s s i s t a n t t i e _ c o n s t r a i n t s = [ DifferentArgConstraint ( [ 0 , 1 , 2 ] , [ 1 , 1 ] ) ] <step> Source : user I n s t r u c t i o n : "Move the c l o t h e s from the bedroom to the washer . After that , Put the cushion on bed . Finally , put the book in the l i v i n g room ." p r o p o s i t i o n s = [ is_on_top ( [ " shirt_1 " ] , [" washer_dryer_1 " ] ) , is_on_top ( [ " shirt_2 " ] , [" washer_dryer_1 " ] ) , is_on_top ( [ " pants_1 " ] , [" washer_dryer_1 " ] ) , is_on_top ( [ " cushion_1 " ] , [" bed_1" , "bed_2 " ] ) , is_in_room ( [ " book_1 " ] , [" living_room_1 " ] ) ] <step> Source : a s s i s t a n t t i e _ c o n s t r a i n t s = [ ] <step> Source : user I n s t r u c t i o n : "{INSTRUCTION}" {PROPOSITIONS} <step> Source : a s s i s t a n t Destination : user t i e _ c o n s t r a i n t s = [ 

I n s t r u c t i o n : "Move the c l o t h e s from the bedroom to the washer . After that , Put the cushion on the bed . Finally , put the book in the l i v i n g room ." 

## **A.16 Prompts for planning baselines** 

Following prompts were used for various planning baselines. 

**Decentralized Single/Multi Agent | ReAct** 

<|start_header_id|>system <|end_header_id|>You are an agent that s o l v e s multi - agent planning problems . The task assigned to you w i l l be s i t u a t e d in a house and w i l l g e n e r a l l y involve navigating to objects , picking and placing them on d i f f e r e n t r e c e p t a c l e s to achieve rearrangement . You s t r i c t l y f o l l o w any format s p e c i f i c a t i o n s and pay a t t e n t i o n to the previous a c t i o n s taken in order to avoid repeating mistakes . Rooms do not need to be explored more than once . There w i l l be another agent trying to s o l v e the same task that you are at the same time . You may f i n d that that agent has picked up r e l e v a n t o b j e c t s or i s in the process of completing parts of the task . I f that i s the case you may want to move on to a d i f f e r e n t part of the task . Rooms do not need to be explored more than once . 

57 

This means i f you have explored the l i v i n g room and have not found the object , then you should explore the kitchen , i f a r e l e v a n t object i s s t i l l not found , you should explore the hallway etc . . . { agent_role_description } Many c a l l s to the same action in a row are a sign that something has gone wrong and you should try a d i f f e r e n t action . <| eot_id |>{ optional_rag_examples } <|start_header_id|>user <|end_header_id|>Task : { task_description } { world_description } P o s s i b l e Actions : { t o o l _ d e s c r i p t i o n s } What i s the next action to make progress towards completing the task ? Return your response in the f o l l o w i n g format Thought : <reasoning f o r why you are taking the next action > <next action c a l l > Assigned ! Here i s an example : Thought : Since there are no o b j e c t s found I should explore a room I have not explored yet Explore[<room name>] Assigned ! <|eot_id|><| start_header_id|> a s s i s t a n t <|end_header_id|> 

## **Centralized | ReAct** 

<|start_header_id|>system <|end_header_id|>You are a system that s o l v e s multi - agent planning tasks . The task assigned to you w i l l be s i t u a t e d in a house and w i l l g e n e r a l l y involve navigating to objects , picking and placing them on d i f f e r e n t r e c e p t a c l e s to achieve rearrangement . There w i l l be a robot agent ( Agent 0) and a human agent ( Agent 1) a v a i l a b l e f o r s o l v i n g the task . Your goal i s to as s i g n a c t i o n s f o r both of these agents and s o l v e the task . You s t r i c t l y f o l l o w any format s p e c i f i c a t i o n s and pay atte ntion to the previous a c t i o n s taken in order to avoid repeating mistakes . 

You should try and divide the task between the two agents f o r e f f i c i e n t task completion . Note that the human agent can wash , clean , f i l l , pour and turn on/ o f f devices along with doing object rearrangement . However , the robot can only do object rearrangement i . e . , navigating to objects , picking and placing them . 

In the beginning , you w i l l be provided with the task d e s c r i p t i o n and information about the rooms plus f u r n i t u r e in each room f o r the house . Object information may or may not be a v a i l a b l e . Rooms only need to be explored i f there i s no information a v a i l a b l e about task - r e l e v a n t o b j e c t s . Rooms do not need to be explored f o r i d e n t i f y i n g which f u r n i t u r e to to go to . Also , rooms do not need to be explored more than once . This means i f one of your agents has explored the l i v i n g room and have not found the object , then you should explore the kitchen , i f a r e l e v a n t object i s s t i l l not found , you should explore the hallway etc . 

Many c a l l s to the same action in a row are a sign that something has gone wrong and you should try a d i f f e r e n t action . 

You should try to complete the task in the l e a s t amount of a c t i o n s p o s s i b l e . This means i f there are two o b j e c t s to be moved you should have one agent navigate to each object and then move them to the t a r g e t l o c a t i o n a the same time . 

I f a previous navigation action i s s t i l l in progress f o r an agent , you should r e a s s i g n that action to the agent t i l l a s u c c e s s f u l execution i s observed in the agent ’ s observations . You should continue to evaluate the task progress and decide the a c t i o n s f o r both the agents . Once both agents are done , you can output "Done [ ] " to i n d i c a t e that the agents have f i n i s h e d the task . Output your response about task completion in the f o l l o w i n g format . Thought : <reasoning about why both agents have completed the e n t i r e task s u c c e s s f u l l y > Done [ ] 

DO NOT output "Done [ ] " unless you are confident that the whole task i s s u c c e s s f u l l y completed . I f one of the agent i s done with i t s part of the task , while the other agent i s s t i l l executing , you can a s s i g n a "Wait [ ] " action to the agent who doesnt need to act anymore . Please re - s t a t e the task d e s c r i p t i o n and v e r i f y it ’ s completion before outputting "Done [ ] " . { eot_tag }{ user_tag }Task : { input } { world_description } P o s s i b l e a c t i o n s f o r each agent : { agent_descriptions } What i s the next action f o r each agent to make progress towards completing the task ? Return your response in the f o l l o w i n g format 

58 

Thought : <reasoning f o r why you are taking the next action > Agent_0_Action : <next action c a l l f o r robot agent> Agent_1_Action : <next action c a l l f o r human agent> Assigned ! Here i s an example : Thought : Since there are multiple task - r e l e v a n t o b j e c t s to be rearranged , I w i l l ask each agent to go to one of them Agent_0_Action : Navigate[< obj name1>] Agent_1_Action : Navigate[< obj name2>] Assigned ! <|eot_id|><| start_header_id|> a s s i s t a n t <|end_header_id|> 

The agent role description would be one of the following, depending on if the agent played the role of the human or robot. 

## **Agent Role Descriptions** 

Human Description 

You are playing the r o l e of the task g i v e r . This means i f the i n s t r u c t i o n says something l i k e "You should move the object and I w i l l wash i t " , then the other agent should be moving the object , and you should washing the i t . 

Robot Description 

You are playing the r o l e of the task r e c e i v e r . This means i f the i n s t r u c t i o n says something l i k e "You should move the object and I w i l l wash i t " , then you should move the object and the other agent should wash i t 

Below are the tool descriptions. Perceptions tools are only included for the React-Tools agents. 

## **Tool Descriptions** 

- Close : Used f o r c l o s i n g an a r t i c u l a t e d e n t i t y . You must provide the name of the f u r n i t u r e you want to c l o s e . Example ( Close [ chest_of_drawers_1 ] ) - Explore : Search a s p e c i f i c room by v i s i t i n g various r e c e p t a c l e s or f u r n i t u r e s in that room . The input to the s k i l l i s the EXACT name of the room to be v i s i t e d . Use the room names provided in the house d e s c r i p t i o n . This t o o l exhaustivly explore s the s p e c i f i e d room . Example ( Explore [ kitchen_1 ] ) - FindAgentActionTool : Should be used to f i n d current and past s t a t e h i s t o r y of other agent . - FindObjectTool : Used to find the exact name/names of the object / o b j e c t s of i n t e r e s t . I f you want to f ind names of o b j e c t s on s p e c i f i c r e c e p t a c l e s or f u r n i t u r e s , p l e a s e include that in the query . Example ( FindObjectTool [ toys on the f l o o r ] or FindObjectTool [ apples ] ) - FindReceptacleTool : Used to know the exact name of a r e c e p t a c l e . A r e c e p t a c l e i s a f u r n i t u r e or e n t i t y ( l i k e a chair , table , bed etc . ) where you can place an object . Example ( FindReceptacleTool [ a kitchen counter ] ) - FindRoomTool : Used to know the exact name of a room in the house . A room i s a region in the house where f u r n i t u r e i s placed . Example ( FindRoomTool [ a room which might have something to eat ] ) - Navigate : Used f o r navigating to an e n t i t y . You must provide the name of the e n t i t y you want to navigate to . Example ( Navigate [ counter_22 ] ) - Open : Used f o r opening an a r t i c u l a t e d e n t i t y . You must provide the name of the f u r n i t u r e you want to open . Example (Open [ chest_of_drawers_1 ] ) - Pick : Used f o r picking up an object . You must provide the name of the object to be picked . The agent cannot hold more than one object at a time . Example ( Pick [ cup_1 ] ) - Place : Used f o r placing an object on a t a r g e t l o c a t i o n . You need to provide the name of the object to be placed , the name of the f u r n i t u r e where i t should be placed , s p a t i a l r e l a t i o n (" on" or " within ") d e s c r i b i n g the r e l a t i o n between the object and f u r n i t u r e . The object to be placed must already be held by the agent ( i . e . picked p rev iou sly ) . In addition to these , you can request to place the object near another object . For that you can o p t i o n a l l y provide a s p a t i a l c o n s t r a i n t s (" next_to ") and the name of the r e f e r e n c e object . To place next to an object , the r e f e r e n c e object must already be on the t a r g e t f u r n i t u r e . API tempate - Place [<object_to_be_placed >, <s p a t i a l _ r e l a t i o n >, <f u r n i t u r e to be placed on>, <spatial_constraint >, <reference_object >]. s p a t i a l _ c o n s t r a i n t and reference_object should be s e t to "None" when necessary . - Rearrange : Used f o r moving an object from i t s current l o c a t i o n to the t a r g e t l o c a t i o n . You need to provide the name of the object to be moved , the name of the f u r n i t u r e where i s should be moved , s p a t i a l r e l a t i o n (" on" or " within ") d e s c r i b i n g the r e l a t i o n between the object and f u r n i t u r e . This w i l l automatically pick the s p e c i f i e d object and move to the t a r g e t f u r n i t u r e and attempt to place i t . In addition to these , you can request to place the object near another object . For that you can o p t i o n a l l y provide a s p a t i a l c o n s t r a i n t s (" next_to ") and the name of the r e f e r e n c e object . To place next to an object , the r e f e r e n c e object must already be on the t a r g e t f u r n i t u r e . API tempate Rearrange[<object_to_be_moved >, <s p a t i a l _ r e l a t i o n >, <f u r n i t u r e to be placed on>, <spatial_constraint >, <reference_object >]. s p a t i a l _ c o n s t r a i n t and reference_object should be s e t to "None" when necessary . - Wait : Used to make agent stay i d l e f o r some time . Example ( Wait [ ] ) - Done : Used to i n d i c a t e that the agent has f i n i s h e d the task . Example (Done [ ] ) 

59 

For LLM agents simulating a human, additionally actions which modify the state of objects are available to be called. For centralized baselines two lists of available actions are provided in the agent description. One for the robot (without state-modifying actions) and one for the human (with state-modifying actions). 

**Human Only Tool Descriptions** 

- Clean : Used f o r cleaning an object . You need to provide the name of the object to clean . - Close : Used f o r c l o s i n g an a r t i c u l a t e d e n t i t y . You must provide the name of the f u r n i t u r e you want to c l o s e . Example ( Close [ chest_of_drawers_1 ] ) - F i l l : Used f o r f i l l i n g an object . You need to provide the name of the object to f i l l . - Pour : Used f o r pouring from one container to another . This s k i l l w i l l pour into the s p e c i f i e d container from whichever container i s c u r r e n t l y held by the agent . - PowerOff : Used f o r turning o f f a powered object . You need to provide the name of the object to be turned o f f . - PowerOn : Used f o r turning on a powered object . You need to provide the name of the object to be turned on . 

The world description contains a text description all rooms and their contained furniture, along with currently observed objects. Below is an example for one scene: 

**World Description Example** Furniture : bedroom_1 : floor_bedroom_1 , chair_41 , chair_42 , bed_49 , table_54 , chest_of_drawers_72 , chest_of_drawers_73 , chest_of_drawers_75 , chest_of_drawers_87 closet_1 : floor_closet_1 , wardrobe_91 living_room_1 : floor_living_room_1 , chair_13 , chair_14 , chair_15 , chair_16 , chair_17 , chair_18 , chair_19 , chair_20 , chair_21 , chair_22 , couch_26 , couch_28 , couch_29 , chair_30 , stool_31 , stool_32 , table_38 , table_39 , table_48 , table_50 , stand_52 , counter_78 toilet_1 : floor_toilet_1 , toilet_43 bedroom_2 : floor_bedroom_2 , bed_23 , chair_46 , chair_47 , table_53 , chest_of_drawers_55 , chest_of_drawers_58 , chest_of_drawers_59 , chest_of_drawers_60 , chest_of_drawers_61 bedroom_3 : floor_bedroom_3 , bed_37 , chair_40 , chest_of_drawers_74 , table_77 , chest_of_drawers_79 , chest_of_drawers_80 , chest_of_drawers_89 , wardrobe_92 bedroom_4 : floor_bedroom_4 , chair_27 , bed_45 , table_51 , wardrobe_56 , wardrobe_57 , chest_of_drawers_82 , chest_of_drawers_83 , table_84 , table_86 kitchen_1 : floor_kitchen_1 , shelves_24 , shelves_25 , chair_33 , chair_34 , chair_35 , chair_36 , cabinet_62 , cabinet_63 , cabinet_64 , cabinet_65 , cabinet_66 , counter_67 , counter_68 , counter_69 , cabinet_70 , cabinet_71 , cabinet_76 , cabinet_81 , unknown_88 , fridge_90 other_room_1 : floor_other_room_1 other_room_2 : floor_other_room_2 bathroom_1 : floor_bathroom_1 , toilet_44 porch_1 : floor_porch_1 entryway_1 : floor_entryway_1 , table_85 The f o l l o w i n g f u r n i t u r e s have a faucet : cabinet_70 Objects : No o b j e c t s found yet 

For baselines utilizing a summary of the scene, after each action, we insert a text description of objects in the scene (as perceived by the agent so far). 

**Scene Summary Example** Thought : Since there are no o b j e c t s found , I should explore a room I have not explored yet . I w i l l s t a r t with the l i v i n g room , as i t i s one of the rooms mentioned in the task , and i t might contain the vase or other r e l e v a n t o b j e c t s . Explore [ living_room_1 ] Assigned ! <| eot_id|><| start_header_id|>user <|end_header_id|> Result : S u c c e s s f u l execution ! Objects : toy_fruits_1 : chest_of_drawers_49 in bedroom_4 laptop_cover_2 : cabinet_59 in laundryroom_1 cup_3 : cabinet_65 in kitchen_1 vase_0 : cabinet_65 in kitchen_1 <| eot_id|><| start_header_id |> a s s i s t a n t <|end_header_id|> Thought : Now that we have found the l o c a t i o n s of the objects , I should navigate to the kitchen to pick up the vase , as i t i s one of the o b j e c t s mentioned in the task , and i t i s located in the kitchen . Navigate [ cabinet_65 ] Assigned ! <| eot_id|><| start_header_id|>user <|end_header_id|> Result : S u c c e s s f u l execution ! Objects : toy_fruits_1 : chest_of_drawers_49 in bedroom_4 laptop_cover_2 : cabinet_59 in laundryroom_1 

60 

cup_3 : cabinet_65 in kitchen_1 vase_0 : cabinet_65 in kitchen_1 <| eot_id|> 

For finetuning, we build the prompt by combining the task description, world description and the previous actions for both agents. 

**Finetuning Example** Task : I f i n i s h e d reading my book . Please move the book to the bedroom t a b l e . You can move the vase there too ? Current Environment : { world_description } Previous a c t i o n s : Agent_Action : Navigate [ table_25 ] Action Result : S u c c e s s f u l execution ! Agent_Action : Navigate [ vase_0 ] Action Result : S u c c e s s f u l execution ! Agent_Action : Pick [ vase_0 ] Action Result : S u c c e s s f u l execution ! Agent_Action : Navigate [ table_35 ] Action Result : S u c c e s s f u l execution ! Other_Agent_Action : Pick [ book_1 ] Agent_Action : Open [ table_35 ] Action Result : S u c c e s s f u l execution ! Agent_Action : Navigate [ table_35 ] Action Result : S u c c e s s f u l execution ! Agent_Action : Place [ vase_0 , on , table_35 , none , none ] Action Result : S u c c e s s f u l execution ! Agent_Action : Wait [ ] Action Result : S u c c e s s f u l Execution ! Other_Agent_Action : Open [ table_35 ] Other_Agent_Action : Place [ book_1 , on , table_35 , none , none ] Next Agent_Action : <| reserved_special_token_0 |>Done[] < end_act> 

61 

## **Contributions** 

**Matthew Chang** : Implemented the state-modifying skills and built out the object state management system. Implemented constrained generation for planning baselines. Implemented ReAct based baselines and optimized their prompting and performance. Contributed to finetuning code and LLM inference infrastructure. 

**Gunjan Chhablani** : Implemented PrediViz and helped analyze **PARTNR** tasks and evaluation functions. 

**Alexander Clegg** : Designed and implemented foundation simulation features and APIs including: geometric relation utils, object states, kinematic relationship management, articulation logic, procedural sampling features, metadata handling, and some oracle skills. Contributed to design, implementation, and better engineering efforts for evaluation, HITL, simulated world graph, perception, and robot skills subsystems. Led the 3D asset dataset preparation effort extending HSSD with articulations, annotations, and quality iteration to enable large scale experimentation on collaboration tasks. 

**Mikael Dallaire Cote** : Led the development of a cloud platform capable of collecting collaboration data, at a large scale, by extending the Habitat 3.0 human-in-the-loop (HITL) system to (1) Run in the browser; (2) Support multiplayer sessions; (3) Evaluate LLM agents; (4) Support the full articulated HSSD dataset. Created a GUI application within that system that allows crowdsource workers to complete collaboration tasks. Deployed the HITL platform to the cloud and rolled out the data collection campaign. Created paper figures. 

**Ruta Desai** : Defined benchmark characteristics and developed the simulation-in-the-loop LLM-based task instruction generation pipeline. Proposed and developed ReAct baselines and its analysis. Wrote experiments and analysis section in the paper. 

**Michal Hlaváč** : Designed PrediViz, the visualization and annotation system for the PARTNR task evaluation. Wrote the section on PrediViz in the paper. 

**Vladimir Karashchuk** : Contributed to 3D asset dataset preparation effort extending HSSD with articulations, annotations, and quality iteration to enable large scale experimentation on collaboration tasks. Authored annotations such as regions, articulations, and marker sets as well as authoring 3D assets including render, collision, and receptacle meshes. Worked on quality assurance for annotations and assets originating from vendor workstreams. 

**Jacob Krantz** : Designed and implemented the **PARTNR** evaluation system and large-scale generation of evaluation functions. Led the dataset annotation effort and maintained, analyzed, and balanced all dataset splits. Guided the development of PrediViz. 

**Roozbeh Mottaghi** : Managed the team and coordinated different workstreams including benchmark generation, planning models, analysis, and human-in-the-loop infrastructure. 

**Priyam Parashar** : Contributed to the world-graph representation and logic for integration into the planning stack, specifically owning the concept-graph pipeline to create non-privileged scene descriptions and nonprivileged graph updates. Built out the integration of LLM agents and planner into the HITL tool enabling single-learn evaluations. 

**SiddharthPatki** : Designed and implemented centralized and decentralized task planning frameworks for baseline experiments in **PARTNR** . Proposed and developed tool based ReAct planner baselines using instruct-tuned Llama3 models, and analyzed planner’s performance using various performance metrics and failure modes. Designed and supported implementation of hierarchical scene graphs to encode information about agent’s environment which is consumed by all baseline planners. Paper writing and figures. 

**Ishita Prasad** : PXFN alignment and approval for data collection/annotation for model training. Secured alignment with other orgs, XFN to enable open-sourcing/release of dataset and code for multi-agent collaboration benchmarks. 

**Xavi Puig** : Contributed to the planning codebase, building infra to scale up planning baselines and LLM inference. Led agent finetuning experiments and oracle-based agents, and debugged simulation and oracle skill failures. Initial exploration of automated benchmark generation. Built tools for HITL data analysis and visualization. Paper writing and figures. 

62 

**Akshara Rai** : Implemented benchmark generation for guided scaling of rearrange, spatial, temporal and heterogenous tasks. HITL analysis, paper writing. Supported all workstreams of the project. 

**Ram Ramrakhya** : Implemented and helped design LLM fine-tuning pipeline, auto regressive evaluation and metrics for fine-tuning LLM planners. Ran experiments to train oracle-planner traces based agents in full and partial observation settings. Implemented initial version of HITL data parsing to convert episodes into traces for LLM finetuning. Implemented and deployed HITL campaign/experiment management service end-to-end using PsiTurk to: (1) Automatically launch and manage large campaigns and metadata. (2) Matchmaking for multi-user tasks, single user task assignment and automatic task recycling. Integrated habitat-llm and initial version of task in HITL to support collaboration tasks. 

**Daniel Tran** : Annotated **PARTNR** datasets and provided feedback leading to improved generation. 

**Joanne Truoung** : Contributed to the dataset effort by refining scene compositions and ensuring validity of the scenes for HITL data collection. 

**John Turner** : Contributed implementation of foundation simulation features and APIs including: region queries, articulations, metadata management, marker sets, and panoptic sensor. Contributed to 3D asset dataset preparation effort extending HSSD with articulations, annotations, and quality iteration by designing scenes, building editor interfaces, and implementing automated quality checks. 

**EricUndersander** : Contributed to HITL architecture. Implemented HITL networking capabilities for single-user evaluation. Contributed to the HITL cloud infrastructure and tooling. 

**Tsung-Yen Yang** : Led the effort of integrating neural network skills and designing a RAG system for running planner evaluation on the benchmark. Performed planner and skill results analysis, and contributed to better engineering efforts for designing skills CIs. Improved the planner’s success rate by identifying edge cases, and better script for finding bugs. 

63 

