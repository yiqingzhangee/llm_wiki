# **WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks** 

**Léo Boisvert** _[∗]_[123] **Megh Thakkar** _[∗]_[124] **Maxime Gasse** _[†]_[123] **Massimo Caccia** _[†]_[1] **Thibault Le Sellier De Chezelles** _[†]_[13] **Quentin Cappart**[23] **Nicolas Chapados**[123] **Alexandre Lacoste** _[‡]_[1] **Alexandre Drouin** _[‡]_[12] 

1ServiceNow Research, 2Mila, 3Polytechnique Montréal, 4Chandar Research Lab `{leo.boisvert, megh.thakkar, alexandre.drouin}@servicenow.com` 

## **Abstract** 

The ability of large language models (LLMs) to mimic human-like intelligence has led to a surge in LLM-based autonomous agents. Though recent LLMs seem capable of planning and reasoning given user instructions, their effectiveness in applying these capabilities for autonomous task solving remains underexplored. This is especially true in enterprise settings, where automated agents hold the promise of a high impact. To fill this gap, we propose WorkArena++, a novel benchmark consisting of 682 tasks corresponding to realistic workflows routinely performed by knowledge workers. WorkArena++ is designed to evaluate the planning, problem-solving, logical/arithmetic reasoning, retrieval, and contextual understanding abilities of web agents. Our empirical studies across state-of-the-art LLMs and vision-language models (VLMs), as well as human workers, reveal several challenges for such models to serve as useful assistants in the workplace. In addition to the benchmark, we provide a mechanism to effortlessly generate thousands of ground-truth observation/action traces, which can be used for fine-tuning existing models. Overall, we expect this work to serve as a useful resource to help the community progress toward capable autonomous agents. The benchmark can be found at https://github.com/ServiceNow/WorkArena. 

## **1 Introduction** 

In 2024, the average adult spends about 400 minutes every day interacting with computer software and the internet [DataReportal, 2024]. While some of this time is devoted to productive work, entertainment or creative endeavors, a significant portion is consumed by monotonous, low-value tasks, particularly in enterprise settings. Employees often engage in tedious tasks such as searching for information hidden deeply in knowledge bases, coordinating group discussions to schedule meetings, or filling expense reports in counter-intuitive user interfaces designed for functionality rather than user-friendliness [Bailey and Konstan, 2006, Norman, 2013]. One easily envisions a future where autonomous assistants—AI _agents_ —handle these tasks, allowing humans to focus on more complex, skill-intensive work that generates greater value. Recent advances in the reasoning and planning capacities of large language models (LLMs), with developments such as Chain-of-Thought [Wei et al., 2022a], ReAct [Yao et al., 2023], and Tree-of-Thought [Yao et al., 2024] (see [Huang et al., 2024, Wang et al., 2024b] for a review) suggest that this future might be within reach. 

While the development of autonomous agents has long been a major research topic both in academia and industry, the recent advances in LLM capabilities have led to a massive surge of interest for 

> _∗_ Equal contribution. 

> _†_ Core contributors. 

> _‡_ Equal supervision. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0002-00.png)


**Figure 1:** Example WorkArena++ task: Restock low inventory items. Here, the agent acts as an IT worker tasked with restocking items that are below some threshold in stock: _⃝_ 1 As is common, it receives instructions via a ticket assigned to them in the system; _⃝_ 2 it must then read the dashboard to extract all items whose stock count is low; _⃝_ 3 reorder the items from the service catalog to match a minimum stock quantity, and _⃝_ 4 close the ticket assigned to them once the task is completed. 

LLM-based autonomous agents [Wang et al., 2024a]. One major application is software control, with a large body of work focused on using LLMs to interact via Application Programming Interfaces (APIs) [Hao et al., 2024, Du et al., 2024]. Another line of research focuses on using LLMs for human-like interactions, by directly manipulating graphical User Interfaces (UIs) on mobile devices [Li et al., 2020, Rawles et al., 2023], desktops [Xie et al., 2024], or websites [He et al., 2024]. This last category encompasses the field of _web agents_ , which can automate software interaction even in environments without APIs, improve human productivity, and accessibility for users with disabilities. 

In this work we propose _WorkArena++_ , a challenging new benchmark to study the proficiency of web agents at solving common knowledge work tasks in enterprise settings. Built on top of the ubiquitous ServiceNow platform, which reported a customer base of more than 7,000 companies and 85% of the Fortune 500 in 2023 [Mastantuono, 2023], it provides a free and robust, yet a realistic evaluation environment for task automation in the workplace. _WorkArena++_ expands the _WorkArena_ benchmark introduced by Drouin et al. [2024] with 682 challenging new tasks. While the tasks in _WorkArena_ are far from being solved by current web agents, they remain predominantly atomic, with simple goals such as filling out a single form with explicit values or navigating explicit menu entries. _WorkArena++_ enhances the scale, realism, and complexity of _WorkArena_ with composite tasks designed to require skills like problem-solving and memorization, in order to better evaluate the capabilities of web agents at solving complex work tasks. Our contributions are as follows: 

- We introduce the WorkArena++ benchmark, considerably expanding the work of Drouin et al. [2024] from 33 to 682 tasks through the addition of realistic office worker trajectories, exemplified in fig. 1, requiring skills like problem-solving, data-driven decision-making, and more (§3). Importantly, this is the first benchmark for web agents to require such complex skills. 

- We make a series of technical contributions to WorkArena, such as added visual diversity through the introduction of 10 fictitious companies along with customized UI color schemes, improved database isolation between tasks to facilitate parallel evaluation, a new framework for easily expanding the set of tasks through the composition of low-level building blocks, and the possibility of extracting Oracle-based observation-action traces for fine-tuning (§3.3). 

- We conduct an empirical study to assess both the difficulty and feasibility of our benchmark, with autonomous agents based on state-of-the-art (visual) language models, both closed and open source, as well as human agents as a baseline. Results indicate that _WorkArena++_ presents considerable challenges for current web agents while being reasonably easy for humans (§4.3), suggesting its value and relevance as an evaluation benchmark for the scientific community. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Chat<br>User: show me<br>HTML AXTree flights from MTL to NYC<br>Screenshot<br>Observation<br>Action<br>Web Agent<br>Browser page.get_by_label("Search").click()<br>Python (unsafe)<br>mouse_click(x=47.2, y=152.6) click(bid="103")<br>coord-based bid-based<br>(a)  WorkArena (b)  BrowserGym<br>**----- End of picture text -----**<br>


**Figure 2:** Background: (a) In WorkArena, tasks measure the ability of web agents to interact with basic UI components in the ServiceNow platform, illustrated above. (b) In BrowserGym, the agent receives a naturallanguage goal from a human user via chat. It then perceives the environment (web browser) through a set of multimodal observations (e.g., HTML and screenshot) and controls it via a standardized set of available actions. Reproduced from Drouin et al. [2024] with permission. 

## **2 Background** 

Before diving into our main contribution WorkArena++, we summarize BrowserGym and WorkArena, which respectively provide the environment in which agents interact with the benchmark, and the atomic tasks upon which WorkArena++ is built. 

## **2.1 BrowserGym – A Gym Environment for Web Agents** 

WorkArena++ is integrated into BrowserGym [Drouin et al., 2024] (fig. 2b), a gym environment that facilitates the design and evaluation of web agents and includes many common benchmarks, such as MiniWob [Shi et al., 2017a, Liu et al., 2018] WebArena [Zhou et al., 2023] and WorkArena [Drouin et al., 2024]. The salient features of BrowserGym include: i) chatbased agent-human interactions, ii) enriched multimodal observations (HTML, accessibility tree [Zhou et al., 2023], screenshot, set-of-marks [He et al., 2024], element coordinates, etc.), and iii) a standardized and flexible action space. In the rest of the paper, all agents interact with WorkArena++ using BrowserGym. 

## **2.2 WorkArena** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0003-07.png)


**Figure 3:** In WorkArena(++), the agent interacts with the frontend of a remote-hosted ServiceNow instance via BrowserGym. Task validation then inspects both the state of the database and any open page using backend (REST) and frontend (JS) ServiceNow APIs. 

The starting point for our work is WorkArena, the first benchmark to measure the performance of web agents at solving work-related tasks in enterprise settings [Drouin et al., 2024]. Below, we outline some of its key properties. 

**Task complexity** While challenging, the tasks included in WorkArena do not require complex problem-solving skills. They rather measure the ability of agents to perform basic interactions with the ServiceNow platform using the main UI components of its user interface, outlined in fig. 2a. For example, one of the tasks consists in filling out a form after receiving the explicit list of desired values for each field. While solving these tasks is a first step toward achieving anything useful in the workplace, they remain extremely simplistic and trivial for humans. 

**Certifiability** An interesting property of WorkArena is that the successful completion of all tasks is certifiable. Each task comes with an _oracle_ and a _validator_ . The oracle is a human-coded solution that uses browser automation (through Playwright [Microsoft, 2023]) to solve the task. The validator is a function that verifies if the task was solved correctly (e.g., by inspecting the database and the current page) and returns a success reward (0 or 1). Our newly proposed WorkArena++ benchmark builds on the same mechanisms and extends them further to handle more complex tasks. 

**Architecture and availability** WorkArena requires agents to interact with remote-hosted clones of the ServiceNow platform called _Personal Developer Instances_ . These can be requested for free 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Skill<br>100 Planning and Problem Solving<br>Contextual Understanding<br>Information Retrieval<br>80 Data-driven Decision-Making and Reasoning<br>Sophisticated Memorization<br>L1<br>60<br>40<br>20<br>0<br>0 20 40 60 80 100 120 140<br>Number of actions<br>(a)  Distribution of tasks across skills (b)  Number of Oracle actions per task<br>Count<br>**----- End of picture text -----**<br>


**Figure 4:** Overview of WorkArena++: a) Distribution of tasks across the skills introduced in §3.2 for all 682 tasks in the L2/L3 sets. b) Task length as estimated by the number of actions required for completion by the Oracle (see §2.2) for all 470 L2/L3 task instances in the _agent curriculum_ (§4.1). Tasks from the L1 set are also included for comparison (33 tasks x 5 seeds). 

through ServiceNow’s Developer Program (see fig. 3 for an illustration). Of note, WorkArena consists of open-source code that interacts with ServiceNow instance APIs and does not rely on any proprietary code. WorkArena++ follows the same design pattern. 

In what remains, we will refer to the tasks from WorkArena as the **L1** tasks (for level 1), while WorkArena++ introduces two new levels **L2** and **L3** with increased difficulty, outlined below. 

## **3 WorkArena++: Taking WorkArena to the Next Levels** 

In WorkArena++, each task consists of a logical combination of simpler atomic tasks, chained together to form a realistic workflow. For example, consider the task of “onboarding a new employee” from the perspective of an IT agent. The process would require the agent to: i) navigate to the appropriate page to create a new user, ii) create a new user account by filling out a form, iii) access a service catalog, iv) order a new laptop, and v) complete a form to assign the laptop to the user in the system. Each of these steps corresponds to a WorkArena L1 task[4] . Next, we provide details on how the tasks in WorkArena++ are categorized across two new levels of difficulty: L2 and L3 (§3.1), and five skill categories (§3.2). 

## **3.1 Difficulty Levels L2 and L3** 

WorkArena++ introduces two new levels of difficulty, L2 and L3, which each cover the same set of 341 workflows presented to the agent in different ways, either as explicit or implicit goals using ServiceNow’s ticket mechanism. This makes for 2 _×_ 341=682 tasks in total. While the workflows to execute in L2 and L3 tasks remain the same, the difficulty is increased in L3 due to the less explicit and more realistic way instructions are provided. Examples of L2 and L3 tasks are showcased in § C. 

**L2 – Explicit** The goal is provided to the agent as detailed instructions sent as a user message in the chat. This message contains the precise steps required to complete the task (e.g., start by navigating to the “users” list, then create a new entry with the following values [.. .] for an ‘onboard user’ task.). In addition to following these steps, succeeding at L2 tasks requires thinking, reasoning and contextual understanding. 

**L3 – Via ticket + knowledge base** The goal is provided to the agent in a manner that mimics how human agents receive work assignments as knowledge workers - through a ticket assigned to them (fig. 1). The ticket includes key details necessary to complete the task (e.g., the name of the new employee for an ‘onboard user’ task) but does not specify the steps required to solve it. Instead, the agent solving the task is informed that additional instructions can be found in the company’s knowledge 

4WorkArena++ mostly re-uses the tasks from L1 as atomic building blocks, but also implements new atomic tasks not in the original L1, such as altering an existing database record. 

4 

**Table 1:** Comparing existing web-agent evaluation benchmarks with WorkArena++. 

|**Benchmark**|MiniWoB|WebArena|WorkArena (L1)|WorkArena++ (Ours)|
|---|---|---|---|---|
|**# tasks**|125|190|33|682|
|**Domain**|Synthetic web appli-<br>cations|Diverse<br>websites<br>(maps, e-commerce,<br>wiki, social forums)|ServiceNow<br>enter-<br>prise platform|ServiceNow<br>enter-<br>prise platform|
|**Nature of**<br>**tasks**|Toy tasks like click-<br>ing buttons, flling<br>felds, etc|Real-world inspired<br>tasks from website in-<br>teractions|Basic enterprise soft-<br>ware specifc tasks<br>such as sorting a list,<br>flling a form|Complex real-world<br>enterprise tasks per-<br>formed everyday by<br>knowledge workers|
|**Abilities**<br>**required**|Diverse but minimal-<br>ist UI manipulation,<br>minimal planning|Complex instruction<br>parsing, advanced UI<br>understanding, infor-<br>mation extraction and<br>retrieval|Manipulating<br>complex, realistic en-<br>terprise software UIs,<br>minimal planning|Planning, logical and<br>arithmetic reasoning,<br>long context under-<br>standing, memoriza-<br>tion, information ex-<br>traction and retrieval|
|**Backend**|Static<br>HTML/JS<br>pages|Self-hosted<br>docker<br>images|ServiceNow Personal<br>Developer Instance<br>(PDI)|ServiceNow Personal<br>Developer Instance<br>(PDI)|



base if needed. This level is more challenging, as the agent must memorize the task details from a web page and retrieve instructions from a knowledge base before organizing its steps to succeed, requiring consolidating information across multiple sources. We present a comparison of WorkArena++ across different dimensions with various benchmarks in tab. 1. 

To concretely distinguish between an L2 and L3 task, we consider the easy expense management task for example. The goal in L2 is displayed below. While the goal highlights all the necessary steps to complete the task, it does not describe how to accomplish them (e.g. how to use the menu to navigate): 

**Managing Your Existing Expenses (L2)** Concretely, you need to complete the following steps: 1. Navigate to the "Expense Lines" module of the "Cost" application. 2. Create a filter for the list to extract all entries where: - "Short description" contains "#SERIES-0cbf9a92-4" 3. Delete expense lines with duplicated short descriptions, keeping only one. 

For the same task, the L3 goal provided to the agent is simply: 

**Managing Your Existing Expenses (L3)** Please complete the following task. 

The goal is accompanied by a ticket displayed in the browser, that says to refer to a knowledge base article containing the rules for expense management (refer to fig. 17b). Therefore L3 requires the agent to understand it needs to read and remember the ticket from starting web page, navigate to the KB article, remember the expense management rules, and apply them. In contrast, in L2 the agent is directly given the steps to perform. 

## **3.2 Skills and Abilities Evaluated in WorkArena++** 

Further, all 682 tasks in WorkArena++ fall into one of 5 categories of skills, based on the abilities they require for being solved. The distribution of tasks across categories is displayed in fig. 4a, and further detailed in § C. We provide a description of each skill category below, along with their number of tasks. 

**Planning and problem solving (66** _×_ **2 tasks)** We evaluate these abilities through tasks that require decision-making under constraints to achieve an expected outcome. One notable example consists of scheduling a series of work items within a given time frame while satisfying various constraints based on criticality, duration, and overlap with other items. Other examples include tasks commonly 

5 

performed by managers, such as redistributing work among employees based on occupancy, and dispatching work to employees based on expertise. 

**Information retrieval (39** _×_ **2 tasks)** We formulate a series of tasks that require retrieving information from either dashboards or lists (see fig. 2a) before performing follow-up tasks based on the retrieved values. For example, one task consists of reading a dashboard to find which item is the lowest in stock and restocking by ordering additional items. 

**Data-driven decision making and reasoning (156** _×_ **2 tasks)** This skill, essential to several knowledge work roles, is evaluated through tasks that require interpreting data, performing logical or mathematical reasoning, and taking subsequent actions. One notable example is a task where the agent must select investments to maximize expected return within a limited budget, effectively solving a small instance of a knapsack[5] problem. 

**Sophisticated memorization (28** _×_ **2 tasks)** An essential skill for web agents is the ability to gather information by navigating through a series of pages, memorizing key details along the way, and finally using it to achieve a specific goal. For instance, in the “onboard new employee” task described earlier, the agent receives all the information about the employee, including hardware requirements, and must navigate through multiple pages, filling relevant information at each step to complete the task. **Contextual understanding through infeasible tasks (52** _×_ **2 tasks)** Finally, we introduce a set of infeasible tasks to verify if the agent can identify them. We consider two kinds: one where the agent is required to simply declare the task infeasible and another where it must additionally justify its decision. For example, if a task were infeasible due to requesting to fill an inexistent form field, the agent would be evaluated based on its ability to name that field in the justification. 

## **3.3 Salient Features of WorkArena++** 

WorkArena++ does not only augment WorkArena with new tasks, but also includes a number of technical improvements over it which we outline below. For more detail on this, refer to § E. 

**Increased visual diversity and realism** WorkArena lacked visual diversity, as all the pages presented to the agent had a similar style. To better assess agents’ ability to generalize across different enterprise settings, we introduce 10 fictitious brands, each with distinct styles of the ServiceNow interface. For instance, the “Charlie’s Cookies” brand is shown in fig. 1. In WorkArena++, a company brand is randomly sampled at the start of each task, enhancing realism and visual diversity. This changes the colors of visual elements as well as the company logo. More visual diversity could be addressed in future works. 

**Standardization and task isolation** For a benchmark to be robust, it must ensure a standardized level of difficulty, regardless of variables like parallel inference or the hardware used for experiments. Achieving this on a remote-hosted instance without access to proprietary code presents a challenge. In WorkArena++, we enhance robustness through several measures. First, we provide an installer that configures the instance with standardized system parameters, UI themes, knowledge bases, and layouts for components like lists and forms. Second, we implement sandboxing by running each task in a new user account, created at its start and automatically deleted at its end. This allows agents to interact freely with the system (e.g., changing visible columns in a list) without affecting subsequent tasks. Together, these improvements result in a more robust benchmark. 

**Extendability and extraction of fine-tuning data** Tasks in WorkArena++ are created by carefully composing the _oracle_ and _validator_ functions (see §2.2) of simpler tasks, such as those in the L1 set. Notably, our framework allows the combination of simple oracle functions to generate human-coded ground truths for more complex compositional tasks. Additionally, this framework facilitates the collection of observation-action traces, regardless of the task’s length and complexity, providing valuable fine-tuning data for LLMs and VLMs in web-agent interactions. 

## **4 Experiments** 

We now present a series of experimental results on WorkArena++. As will be shown, our proposed benchmark poses a significant challenge for state-of-the-art web agents while being relatively simple 

> 5https://en.wikipedia.org/wiki/Knapsack_problem. 

6 

for humans to solve. This contrast underscores the benchmark’s potential to drive advancements in the field, providing the community with a valuable tool for evaluating and improving web agents. 

## **4.1 Evaluation Curriculum: Standardizing WorkArena++ as a Benchmark** 

Each WorkArena++ task can be instantiated with variability from thousands of valid configurations per task. Hence, the benchmark can be viewed as a rich distribution over task instances. In order to make the cost of evaluation accessible, improve reproducibility, and have a uniform test of various skills required for solving the tasks, we propose a standardized way to sample a collection of task instances, which we refer to as a curriculum. Concretely, we provide a mechanism that takes a numerical seed as input and can produce an arbitrary number of task instances, sampled uniformly across skills in a reproducible way. We reserve seeds 0-9 for evaluation and the remaining seeds may be used for agent tuning[6] . Additional details in § D. 

## **4.2 Agent Design** 

We mostly follow the same agent design as Drouin et al. [2024], which consists in using an LLM augmented with chain-of-though prompting [Wei et al., 2022b] to produce the next best action for solving the task, based on the current observation of the web browser. 

**Observation space** The main elements presented to our web agents are the current goal, the current page’s accessibility tree (AXTree) [Zhou et al., 2023] which can be seen as a compressed representation of the HTML, and the error message (if any) that resulted from the execution of the previous action. Additionally, our VLM-based agent is given a screenshot of the current page augmented with set-ofmarks [He et al., 2024]. As most tasks in WorkArena++ require long context understanding, we also add to the prompt the history of their previous actions and thoughts (from chain of thoughts) since the start of the episode. This simple mechanism provides a crude memorization mechanism to otherwise memory-less agents, giving them more chances of solving L2 and L3 tasks. 

**Action space** All our agents use the high-level action space from BrowserGym, restricted to the `chat` , `infeas` and `bid` action sets, which allow sending messages to the chat, declaring a task infeasible, and interacting with the current webpage via primitives using element identifiers ( `bid` attribute), e.g., clicking on an element, filling a text box, etc. Our agents are restricted to producing only one action at a time (as in [Drouin et al., 2024]), and implement a retry mechanism that can re-prompt the agent up to 4 times in case of parsing errors in their output. 

**Language models** We evaluate closed-source models GPT-3.5 and GPT-4o [OpenAI, 2023] and study the impact of providing screenshots of the pages with GPT-4o vision. On the open-source side, we evaluate Llama3-70b [Meta, 2024] and Mixtral-8x22b [Jiang et al., 2024], deployed using Hugging Face’s Text Generation Inference (TGI) library on 4 A100 GPUs. We use a maximum context length of 40K tokens for GPT-4o, 15K for GPT-3.5, 8K for Llama3 and 32K for Mixtral. To ensure that prompts do not exceed those limits, we progressively truncate the accessibility tree from the end until it fits in the context. For more information on agent design and the setup, please refer to § B. 

**Maximum number of steps** For budget reasons, we run our agents for a maximum of 50 time-steps before the tasks are terminated. According to our oracle analysis in fig. 4b, this gives our agents the chance to solve most of the tasks in our _evaluation curriculum_ . 

## **4.3 Agent Results** 

We evaluate all baseline agents on the standardized curriculum introduced in §4.1 and report the results in tab. 2. A notable takeaway is that all agents, whether closed-source or open-source, and regardless of being LLM or VLM-based, generally fail to achieve any reasonable success on WorkArena++, despite performing reasonably well on existing benchmarks. Only GPT-4o and GPT-4o-v succeed at some tasks, particularly memorization tasks within the L2 set, with no successes observed in the L3 set. Interestingly, we observe that, in contrast to its unimodal counterpart GPT-4o, the GPT-4o-v agent succeeds at solving a few retrieval tasks involving reading values off charts, suggesting that the 

> 6To reduce standard error, users can average across multiple seeds between 0-9. Also, it is encouraged to tune the agent using different benchmarks, e.g. WebArena, to better evaluate generalization, but if one must use WorkArena++ for tuning, we encourage to avoid seeds 0-9. 

7 

**Table 2:** Success rate _±_ Standard error (SR _±_ SE) of all agents on MiniWoB, WorkArena, and WebArena, with numbers reported in %. Bolded numbers represent the average success rate over the entire corresponding benchmark. Results on WorkArena L1, WebArena and MiniWoB are extracted from Drouin et al. [2024]. Human evaluation numbers for MiniWoB and WebArena are taken from Humphreys et al. [2022] and Zhou et al. [2023] respectively. The number of task instances is for the agent curriculum. For more detail on human curriculum, refer to § A. 

|**Task Category**(task instances count)|**Agent Curriculum**<br>(full benchmark)|**Human Curriculum**<br>(subset of tasks)|
|---|---|---|
||**GPT-3.5**<br>**GPT-4o**<br>**GPT-4o-v**<br>**Llama3**<br>**Mixtral**|**Human**<br>**GPT-4o**|
|**WorkArena L3**(235)<br>Contextual Understanding (32)<br>Data-driven Decision-Making (55)<br>Planning and Problem Solving (44)<br>Information Retrieval (56)<br>Sophisticated Memorization (48)|**0.0** _±_0.0<br>**0.0** _±_0.0<br>**0.0** _±_0.0<br>**0.0** _±_0.0<br>**0.0** _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0|**93.9** _±_3.4<br>**0.0** _±_0.0<br>87.5 _±_11.7<br>0.0 _±_0.0<br>100.0 _±_0.0<br>0.0 _±_0.0<br>87.5 _±_11.7<br>0.0 _±_0.0<br>100.0 _±_0.0<br>0.0 _±_0.0<br>91.7 _±_8.0<br>0.0 _±_0.0|
|**WorkArena L2**(235)<br>Contextual Understanding (32)<br>Data-driven Decision-Making (55)<br>Planning and Problem Solving (44)<br>Information Retrieval (56)<br>Sophisticated Memorization (48)|**0.0** _±_0.0<br>**3.0** _±_1.1<br>**3.8** _±_1.3<br>**0.0** _±_0.0<br>**0.0** _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>3.6 _±_2.5<br>0.0 _±_0.0<br>0.0 _±_0.0<br>0.0 _±_0.0<br>14.6 _±_5.1<br>14.6 _±_5.1<br>0.0 _±_0.0<br>0.0 _±_0.0|**93.9** _±_3.4<br>**2.1** _±_2.0<br>100.0 _±_0.0<br>0.0 _±_0.0<br>84.6 _±_10.0<br>0.0 _±_0.0<br>100.0 _±_0.0<br>0.0 _±_0.0<br>100.0 _±_0.0<br>0.0 _±_0.0<br>91.7 _±_8.0<br>8.3 _±_8.0|
|**WorkArena L1**(33_×_10 seeds)<br>**MiniWoB**(125_×_5 seeds)<br>**WebArena**(812)|**6.1** _±_1.3<br>**42.7** _±_2.7<br>**41.8** _±_2.7<br>**17.9** _±_2.1<br>**12.4** _±_1.8<br>**43.4** _±_1.6<br>**71.3** _±_1.5<br>**72.5** _±_1.5<br>**68.2** _±_1.2<br>**62.4** _±_1.6<br>**6.7** _±_0.9<br>**23.5** _±_1.5<br>**24.0** _±_1.5<br>**11.0** _±_1.1<br>**12.6** _±_0.5|–<br>–<br>**_93.5_**<br>–<br>**_78.2_**<br>–|



vision modality can be beneficial in WorkArena++. Additionally, as expected, the GPT-4o-based agent significantly outperforms its GPT-3.5 counterpart. 

These results raise important questions: i) Are these tasks actually solvable? and ii) Why do the agents fail? In what follows, we address each of these questions through human evaluation (§4.4) and error analysis (§4.5), respectively. 

## **4.4 Human Evaluation** 

To assess the feasibility of the benchmark and measure the gap between humans and agents, we conducted a study with 15 human subjects tasked with solving WorkArena++ tasks. Given the limited number and availability of subjects, we devised a shorter curriculum comprising 98 task instances, sampled uniformly across skills and the L2/L3 sets. Each participant solved a subset of the tasks using a custom-made evaluation tool that exactly matched the interface available to agents. We report these results in tab. 2 under the Human Curriculum column, along with the corresponding performance of our GPT-4o agent on the same subset of tasks. The numbers are striking, with an overall 93.9% success rate for humans and 2.1% for GPT-4o. These establish WorkArena++ as a valuable benchmark that is both solvable and relatively straightforward for humans, while being challenging for state-of-the-art LLMs, emphasizing its value as a new milestone for the community. 

We note that all subjects consented to participate in the study without compensation. Most had little to no familiarity with ServiceNow products and underwent only a brief training session, consisting of a 15-minute video outlining the components in fig. 2a, followed by 15 minutes of self-guided exploration on the platform. Details on the task curriculum, training received, demographics, and the evaluation platform are included in § A. 

## **4.5 Error Analysis** 

To understand the poor performance of agents on WorkArena++, we conducted an in-depth study of their execution traces. We focused on the best-performing open- and closed-source models, Llama3 and GPT-4o. This analysis identified salient types of errors, which sheds light on current model limitations and highlight areas for improvement. 

**Information Retrieval** The models tend to successfully navigate to the correct information sources (e.g., dashboards). However, they occasionally fail to accurately retrieve the relevant information from the observations. Furthermore, agents sometimes fail to identify relevant elements on the page and attempt to act on incorrect ones. 

8 

**Exploration** Some tasks require to explore the page for hidden information, such as opening different tabs in a form or expanding a section of foldable elements. The agents often struggle due to a lack of curiosity, leaving them stuck in their location. 

**Hallucination** _Made-Up Actions:_ The models sometimes hallucinate actions that would be convenient for the task at hand, but that are not available in BrowserGym. _Imaginary Buttons:_ Similarly, we observed cases of interaction with made-up buttons that would solve tasks in one click, such as buttons that create the exact filter required. _Asking for Help:_ When confused about the next steps, models sometimes ask for help via chat, indicating a lack of confidence or capacity in planning the next steps. 

**Goal Understanding** _Thought/Action Consistency:_ There are instances where the models’ thought process correctly identifies the next action, but the action produced is different and incorrect. This inconsistency undermines performance. _Low-Level Understanding in L3 Tasks:_ In more complex L3 tasks, the models fail to comprehend the necessary subtasks fully. For example, they might start by attempting to modify ticket values that are locked, showing a misunderstanding of the task requirements. 

**Action Consequences Assessment** _Hallucinated Consequences:_ The models often hallucinate the consequences of their actions, believing they have made progress on the task when no actual progress has occurred. _Repeated Actions:_ When an action does not change the state of the webpage, models tend to retry the same action repeatedly instead of trying a different approach. 

These errors illustrate the current limitations of state-of-the-art web agents in handling complex enterprise tasks. Addressing these issues is crucial for developing more reliable and effective autonomous agents capable of performing real-world knowledge work. Detailed examples are included in § F. 

## **5 Related Work** 

Early benchmarks for web agents primarily utilized synthetic web environments where agents executed low-level keyboard and mouse actions [Shi et al., 2017b,a, Liu et al., 2018]. More recently, Zhou et al. [2023] introduced WebArena, comprising 190 tasks based on realistic websites that emulate real-world domains such as e-commerce, social forums, and content management. OSworld [Xie et al., 2024] introduces a scalable, real computer environment for benchmarking multimodal agents, supporting task setup, execution-based evaluation, and interactive learning across operating systems. 

In terms of datasets, Deng et al. [2023] proposed Mind2Web, a large-scale collection of 2,000 web interactions from 137 websites curated by human annotators. Similarly, Lù et al. [2024] introduced WebLINX, a curated dataset of web interactions with 2337 expert demonstrations from 155 different real-world websites. He et al. [2024] proposed 300 information-retrieval tasks from 15 real-world consumer websites, evaluating WebVoyager, a vision-based web agent’s capabilities. WorkArena [Drouin et al., 2024] focuses on real-world enterprise software applications by including 33 interactions tasks representative of realistic workflows typically performed by knowledge workers. 

Building on this foundation, WorkArena++ introduces tasks requiring advanced skills like problemsolving and data-driven decision-making. Unlike previous benchmarks, WorkArena++ evaluates agents on their ability to perform complex, multi-step tasks that closely mimic real-world enterprise scenarios. Additionally, WorkArena++ emphasizes task isolation, uniform setup, and robust evaluation guidelines, enabling fair comparisons and reproducibility. This approach makes WorkArena++ unique in evaluating the capabilities of LLM and VLM-based web agents. 

## **6 Conclusion and Future Work** 

We propose WorkArena++, a novel benchmark consisting of 682 tasks to evaluate web agents by mimicking realistic workflows performed routinely by knowledge workers using enterprise software. WorkArena++testsvariouscomplexskillsofLLMandVLM-basedagentsincludingplanning, decisionmaking, retrieval, logical and arithmetic reasoning, as well as the ability to identify infeasible tasks. Our benchmark promotes standardized evaluation, realistic visual diversity, and provides a method for generating large amounts of fine-tuning data in the form of web-interaction traces. Empirical evaluations reveal that state-of-the-art LLMs and VLMs struggle with our benchmark, while humans achieve extremely high performance. Through our error analyses and qualitative studies, we hope WorkArena++ will be a significant step towards developing more capable autonomous web agents. 

9 

In future work, we aim to continue developing new sets of tasks on the ServiceNow platform. Of particular importance would be tasks for evaluating safety and cybersecurity around agents as well as a hidden test set for hosting competitions. Furthermore, our framework offers the ability to generate a vast amount of fine-tuning data through web interaction traces to train more robust LLM and VLM-based web agents. Our ultimate goal is to close the significant gap between autonomous agents and humans on WorkArena++ and other benchmarks. 

## **7 Limitations and Potential Societal Impacts** 

**Limitations** While WorkArena++ includes diverse and realistic workflows, it does not exhaustively cover all possible knowledge-work tasks and personas. Achieving comprehensive coverage would require thousands of additional tasks. However, our task set is designed to be easily extendable by the community, and we welcome such contributions. Additionally, while this work evaluates the reasoning abilities of LLM-based web agents, it does not assess their safety and robustness against various malicious behaviors, which remains an important barrier to their adoption in real-world settings. Moreover, the benchmark does not include tasks that require interaction with software external to the ServiceNow platform, which would improve diversity and realism. We leave such assessments to OSlevel benchmarks like that of Xie et al. [2024]. Finally, we note that additional open and closed-source LLMs and VLMs could have been included in the experiments, particularly those with extremely long context lengths, such as Gemini 1.5 pro with 1 million tokens [Reid et al., 2024]. 

**Societal Impacts** This work is likely to inspire the development of agents as valuable workplace assistants, positively impacting society in several ways. It can increase productivity, enabling agents to handle more complex and value-creating tasks. Additionally, it can improve accessibility for impaired users, potentially opening new job opportunities. However, there are potential negative impacts. Advanced agents may lead to job displacement as such systems take over human tasks. Reliance on these agents raises data privacy and security concerns and could erode human skills and decisionmaking abilities over time. Moreover, the significant computational resources required to support these agents lead to substantial energy use, contributing to negative environmental impacts. 

## **Acknowledgments and Disclosure of Funding** 

The authors are grateful to the individuals who participated in our evaluation study: Arjun Ashok, Aayush Bajaj, Ghazwa Darwiche, Jerry Huang, Raymond Li, Marie-Ève Marchand, Étienne Marcotte, Shravan Nayak, Sébastien Paquet, Soham Parikh, Fanny Rancourt, Gopeshh Subbaraj, Jordan Prince Tremblay, and Andrew Williams. Your contributions were invaluable in showcasing that, for now, humans are still the reigning champions of intelligence. We also extend our thanks to Chris Pal, Issam Laradji, and David Vazquez for their insightful feedback and suggestions, which were almost as brilliant as our participants’ defense of human ingenuity. 

## **References** 

- Brian P Bailey and Joseph A Konstan. On the need for attention-aware systems: Measuring effects of interruption on task performance, error rate, and affective state. _Computers in human behavior_ , 22 (4):685–708, 2006. URL `https://doi.org/10.1016/j.chb.2005.12.009` . 

- DataReportal. Digital 2024: Global overview report, 2024. URL `https://datareportal.com/ reports/digital-2024-global-overview-report` . 

- Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2Web: Towards a generalist agent for the web. _arXiv_ , abs/2306.06070, 2023. 

- Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, Léo Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. Workarena: How capable are web agents at solving common knowledge work tasks?, 2024. 

- Yu Du, Fangyun Wei, and Hongyang Zhang. Anytool: Self-reflective, hierarchical agents for large-scale api calls, 2024. 

10 

Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings, 2024. 

- Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. WebVoyager: Building an end-to-end web agent with large multimodal models. _arXiv_ , abs/2401.13919, 2024. URL `https://arxiv.org/abs/2401.13919` . 

- Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, and Enhong Chen. Understanding the planning of llm agents: A survey. _ArXiv_ , abs/2402.02716, 2024. URL `https://api.semanticscholar.org/CorpusID:267411892` . 

- Peter C Humphreys, David Raposo, Tobias Pohlen, Gregory Thornton, Rachita Chhaparia, Alistair Muldal, Josh Abramson, Petko Georgiev, Adam Santoro, and Timothy Lillicrap. A data-driven approach for learning to control computers. In _International Conference on Machine Learning (ICML)_ , 2022. 

- Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, MarieAnne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts, 2024. 

- Geunwoo Kim, Pierre Baldi, and Stephen McAleer. Language models can solve computer tasks. _arXiv_ , abs/2303.17491, 2023. URL `https://arxiv.org/abs/2303.17491` . 

- Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge. Mapping natural language instructions to mobile ui action sequences. In _Annual Conference of the Association for Computational Linguistics (ACL 2020)_ , 2020. URL `https://www.aclweb.org/anthology/2020.acl-main. 729.pdf` . 

- Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration. In _International Conference on Learning Representations (ICLR)_ , 2018. 

- Xing Han Lù, Zdenˇek Kasner, and Siva Reddy. Weblinx: Real-world website navigation with multi-turn dialogue. _arXiv preprint arXiv:2402.05930_ , 2024. 

- Gina Mastantuono. ServiceNow joins the prestigious Fortune 500 list. `https://www.servicenow. com/blogs/2023/servicenow-joins-fortune-500-list.html` , 2023. Accessed: 2024-0129. 

- Meta. Llama 3: Meta’s latest large language model. `https://github.com/meta-llama/llama3` , 2024. Accessed: 2024-06-03. 

Microsoft. Playwright for Python documentation, 2023. URL `https://playwright.dev/python/` . 

National Center for O*NET Development. O*NET 29.0 Database. `www.onetcenter.org/ database.html` , 2024. Accessed 28 October 2024. 

- Donald A. Norman. _The design of everyday things: Revised and expanded edition_ . Basic books, New York, 2013. 

- OpenAI. GPT-4 technical report. _ArXiv_ , abs/2303.08774, 2023. URL `https://arxiv.org/abs/ 2303.08774` . 

- ChristopherRawles, AliceLi, DanielRodriguez, OrianaRiva, andTimothyLillicrap. Androidinthewild: A large-scale dataset for android device control. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, _Advances in Neural Information Processing Systems_ , volume 36, pages 59708–59728, 2023. 

- Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 

11 

- Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. World of bits: An open-domain platform for web-based agents. In _International Conference on Machine Learning (ICML)_ , 2017a. 

- Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. World of bits: An open-domain platform for web-based agents. _ICML_ , 2017b. 

- Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. _Frontiers of Computer Science_ , 18(6):1–26, 2024a. 

- Yiqi Wang, Wentao Chen, Xiaotian Han, Xudong Lin, Haiteng Zhao, Yongfei Liu, Bohan Zhai, Jianbo Yuan, Quanzeng You, and Hongxia Yang. Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning. _ArXiv_ , abs/2401.06805, 2024b. URL `https://api.semanticscholar.org/CorpusID:266999728` . 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Huai hsin Chi, F. Xia, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837, 2022a. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, _Advances in Neural Information Processing Systems_ , volume 35, pages 24824–24837. Curran Associates, Inc., 2022b. URL `https://proceedings.neurips.cc/paper_files/paper/2022/file/ 9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf` . 

- Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024. 

- Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. _arXiv preprint arXiv:2310.11441_ , 2023. 

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. _arXiv_ , abs/2210.03629, 2023. URL `https://arxiv.org/abs/2210.03629` . 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. _ArXiv_ , abs/2307.13854, 2023. URL `https://arxiv.org/abs/ 2307.13854` . 

12 

## **Appendix** 

## **Table of Contents** 

|**le**|**of Contents**|**of Contents**||
|---|---|---|---|
|**A **|**Human Evaluation – Additional Details**||**13**|
||A.1|Participants<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|13|
||A.2|Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
||A.3|Task curriculum . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
||A.4|Evaluation platform . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
||A.5|Limitations<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
|**B **|**Agent Design**||**18**|
|**C **|**Task**|**details**|**21**|
||C.1|Planning and Problem Solving (66_×_2 tasks) . . . . . . . . . . . . . . . . . . .|21|
||C.2|Information Retrieval (39_×_2 tasks) . . . . . . . . . . . . . . . . . . . . . . . .|23|
||C.3|Data-driven decision making and reasoning (156_×_2 tasks) . . . . . . . . . . . .|23|
||C.4|Sophisticated Memorization (29_×_2 tasks)<br>. . . . . . . . . . . . . . . . . . . .|25|
||C.5|Contextual understanding through infeasible tasks (52_×_2 tasks) . . . . . . . . .|25|
||C.6|Task Protocols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|26|
|**D **|**Evaluation Curriculum Details**||**37**|
|**E**|**Expanded Features**||**39**|
||E.1|Application Themes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|39|
||E.2|Extracting traces and designing more tasks<br>. . . . . . . . . . . . . . . . . . . .|45|
|**F**|**Error Analysis**||**47**|
|**G **|**Additional Discussion**||**55**|
||G.1|Limitation: Restricted to ServiceNow . . . . . . . . . . . . . . . . . . . . . . .|55|
||G.2|Expanding the Benchmark – Ensuring Long-Term Relevance . . . . . . . . . . .|55|
||G.3|Context-size sensitivity<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|56|



## **Contents** 

## **A Human Evaluation – Additional Details** 

This section provides additional details on our human evaluation study, where humans were tasked with solving WorkArena++ tasks. 

## **A.1 Participants** 

We recruited a cohort of 15 volunteers with varying levels of familiarity with ServiceNow products. This group included core members of ServiceNow’s research team and students from local institutions, namely the University of Montreal and the _École de technologie supérieure_ . All participants provided informed consent as outlined in the consent form (see fig. 5). The demographic profile of the participants is depicted in fig. 6 (see § A.5 for a discussion of the representativeness of the cohort). 

13 

## **A.2 Protocol** 

An in-person event was conducted at the ServiceNow Montreal office. The session began with a brief training, during which participants watched a 15-minute recorded talk (slides shown in fig. 7). After the talk, participants engaged in 15 minutes of hands-on self-exploration of ServiceNow products using the same free Personal Developer Instances employed in the main study. Following this selfexploration period, each participant was asked to solve up to seven tasks individually on the _evaluation platform_ described below. The curriculum from which these tasks were sampled is also described below. Participants were instructed not to discuss the tasks among themselves. They were informed that both their time to resolution and success rate would be recorded. The exact web page that participants used for guidelines on the day of the event is available in the benchmark’s GitHub Wiki. 

## **A.3 Task curriculum** 

Due to the limited time and availability of human evaluators, we had to limit the size of the curriculum used in the study. Hence, we evaluated humans based on a curriculum of 98 task instances, sampled uniformly at random across skills (§3.2), and considered both their L2/L3 variants (49 task instances x 2 levels). The exact curriculum used can be retrieved from the benchmark’s codebase. 

## **A.4 Evaluation platform** 

The human evaluators were asked to solve tasks in a UI that identically matches that available to the agents (i.e., BrowserGym [Drouin et al., 2024]), except for the addition of a “Human Evaluation Console” overlay (see fig. 8). This console provided functionalities such as a “Validate” button that allowed them to check if they had completed the task successfully and a “Give up” button that allowed them to abandon the task (counting as a failure). Other noteworthy features include: i) an auto-validation mechanism that constantly checked the completion status of the page in real time, complementing the validation button, and ii) the console’s movability to prevent blocking the evaluator’s view of the page. Finally, note that, humans did not receive feedback while completing the tasks; they could only see if the task was currently incomplete or solved. The code for the human evaluation platform is included in our main codebase, along with a command-line program to launch it. 

## **A.5 Limitations** 

**Learning effect:** We noticed that participants became increasingly efficient at solving tasks as they progressed through their assigned curriculum. We hypothesize that this was primarily due to gaining familiarity with the product’s user interface. While our protocol does not directly account for this learning effect, we note that it is unlikely that the evaluators learned to solve any given task since most were not asked to perform the same task twice. Among our 15 evaluators, only 3 had curricula that included a repeated task, and, importantly, these tasks were always presented under another level (L2/L3) with a different seed. The significant difference between the presentation of goals and instructions in L2/L3 task acts as further mitigation. 

**General announcements to participants:** During the evaluation, multiple participants encountered the same issue due to the user interface being counterintuitive, which led to a general announcement. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0014-09.png)


**Figure 5:** Consent form signed by all human evaluators prior to participating in the study. 

14 

The announcement clarified that a ticket would not be marked as “Closed - Complete” until the form was saved via the “Update” button. This is important for L3 tasks, as such tasks are only considered complete when the agent marks the ticket as “Closed - Complete.” Although most participants correctly changed the ticket’s state field, they did not save their changes, resulting in the task being considered incomplete. After our announcement, this error did not recur. It is important to note that this hint cannot account for the significant performance difference between humans and the LLM-based agents, as the LLMs fail to complete the initial steps of the L3 tasks, never approaching this point in the task’s trajectory. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0015-01.png)


**Figure 6:** Demographic profile of study participants. Each subfigure presents the distribution of responses to a specific question, including the available answer choices and the corresponding number of responses. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0016-00.png)


**Figure 7:** Slides for the 15-minute presentation used to train human evaluators. To be viewed from left to right. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0016-02.png)


**Figure 8:** Human evaluation platform. The interface consists of the BrowserGym chat (left) and browser (right) windows, along with the “Human Evaluation Console” (bottom-right) as an overlay. 

**Cohort representativeness:** The demographics data reveals that approximately 75% of human evaluators have worked at ServiceNow, all of them with undergrad degrees, and half with advanced degrees. It is likely that this is not representative of the general user demographics of ServiceNow. Hence, it is normal to wonder if the high performance of human evaluators accurately characterizes the general cognitive complexity of the tasks for arbitrary first-time users. We discuss multiple aspects: 

- **ServiceNow Employees:** Out of our evaluators, 11/15 have been employed by ServiceNow. However, this number must not be interpreted as “people who have interacted with the product before”. In fact, many of these people have roles that do not involve interaction with the product. For example, 4 of these 11 evaluators were student researchers who, by the nature of their work, are not exposed to the product. Moreover, we emphasize that 46.7% of our evaluators declared being first-time users of the product, and 86.7% claim to use the product at most a few times per month. Finally, we stress that the tasks in WorkArena++ mostly evaluate 

16 

planning and reasoning skills, beyond basic interaction with the product that some evaluators could be familiar with. 

- **Study degrees:** This bias is present in our pool of evaluators. Nevertheless, we believe it does not invalidate our conclusions, since none of our tasks involve skills that one would acquire through an advanced degree. Among the “general user demographics” of ServiceNow, all users must be able to manipulate basic UI components, such as forms, lists, etc. Beyond that, succeeding at our benchmark involves following a clear set of instructions and basic reasoning that should be well within reach for anyone with the ability to interact with the software. Hence, while this bias is present, we argue that it is not a significant driver of success on the benchmark. 

- **Zero-shot models:** We stress that agents based on LLMs cannot be characterized as “first-time users” of ServiceNow. Our exploration revealed that these models have extensive knowledge of the ServiceNow platform, such as very detailed knowledge of available APIs, detailed information about the underlying data structure, and knowledge of how to solve certain tasks in the product. This likely stems from training on publicly available documentation and discussions in public support forums. Hence, we believe that the relative lack of exposure to the platform of most human evaluators and the knowledge held by LLMs/VLMs helps in normalizing possible discrepancies in our human evaluation. 

**Could training and self-exploration explain human performance?** The training given to human evaluators (§ A.2) contains very high-level information about which UI components are included in tasks and how to use our human evaluation platform (see fig. 7). This in itself does not reveal information that helps to solve tasks. In the 15 minutes of self-exploration, the humans learn how to manipulate the UI components, a skill that the agents might indeed acquire through fine-tuning. However, as emphasized by our error analysis (§4.5), most failures of agents are due to mistakes in planning and reasoning, not basic UI manipulation. Furthermore, as previously explained, LLMs/VLMs have access to a wealth of information on ServiceNow products, potentially giving agents an advantage well beyond knowledge of basic UI manipulation. 

**Is there a maximum number of actions for human evaluators?** To simplify the evaluation process, we did not impose an explicit limit on the number of actions that human evaluators could take. In contrast, AI agents are subject to an action limit, not to impose additional constraints, but to account for practical resource limitations, such as credit usage and rate limits. The number of actions allowed for AI agents is designed to be sufficient for task completion. 

17 

## **B Agent Design** 

Below are the general design choices of our LLM/VLM-based web agent with chain-of-thought prompting [Wei et al., 2022b]. 

**Language models:** Our study distinguishes between closed- and open-source LLMs. For the closedsource segment, we evaluate **GPT-3.5** ( `gpt-3.5-turbo-1106` , 16K context) and **GPT-4o** [OpenAI, 2023] ( `gpt-4o-2024-05-13` , 128K context), through OpenAI’s API. We also explore the effect of providing the screenshot of the page using GPT-4o vision and Set-of-Mark [Yang et al., 2023] as proposed in WebVoyager [He et al., 2024]. In the realm of open-source LLMs, we evaluate both **Llama3-70b** [Meta, 2024] ( `meta-llama-3-70B-instruct` , 8K context) and **Mixtral 8x22B** [Jiang et al., 2024] ( `open-mixtral-8x22b` , 64K context). These model are deployed using Hugging Face’s Text Generation Inference (TGI) library on 4 A100 GPUs. 

**Observation space:** Our observation space is composed of the goal, the current page’s HTML and/or AXTree,[7] the currently focused element, and the error from the previous action if any. We also augment each element with two extra boolean properties provided by BrowserGym, `clickable` and `visible` . Finally, when using GPT-4o vision we also give the model a screenshot of the current page, augmented with set-of-marks [He et al., 2024] using the element identifiers ( `bid` ) provided by BrowserGym. 

**Action space:** We use BrowserGym’s high-level action space with `chat` , `infeas` and `bid` primitives [Drouin et al., 2024] which respectively allow the agent to send messages to the chat (‘ `send_msg_to_user(text)` ‘, necessary for information retrieval tasks), to declare the task infeasible ( `report_infeasible(reason)` , necessary for infeasible tasks[8] ), and to interact with the page’s HTML elements using their unique identifiers (e.g., `click(bid)` , `type(bid, text)` etc.). The `bid` primitives rely on the unique `bid` attribute given by BrowserGym to each HTML element, which is made available textually in the HTML and AXTree as well as visually through set-of-marks in the screenshots. The full action space is described to the agent in the prompt, with individual examples of valid function calls for each primitive. For an example prompt and the corresponding screenshot with set-of-marks, see fig. 9 and fig. 10. 

**History:** To extend the horizon window of our agent, at each time step we re-inject into the agent’s prompt the history of all previous actions and thoughts (from chain-of-thought) since the start of the episode. This gives our agent a chance to recall its previous thoughts, thereby providing a crude memorization mechanism to otherwise memory-less agents, giving them more chances (theoretically) of solving memory-intensive L2 and L3 tasks. 

**Zero-shot examples:** In the prompt, we provide a single generic example of how the chain-of-thought and action outputs should be formatted. This contrasts with other methods [Kim et al., 2023] where task-specific few-shot examples are provided, yet aligns with our objective of developing zero-shot agents able to solve a large range of new tasks. 

**Parse and retry:** Once the LLM provides an answer, we have a parsing loop that can re-prompt the agent up to 4 times to make it aware of a parsing mistake. This can save the agent from making basic mistakes and is mainly useful for less capable LLMs. Once parsed, the action is executed via BrowserGym, which moves to the next step. 

**Prompt truncation:** We use a maximum prompt length of 40K tokens for GPT-4o, 15K for GPT-3.5, 8K for Llama3 and 32K for Mixtral. When the prompt is too large, we progressively truncate the HTML and AXTree from the end until it fits the maximum allowed number of tokens. 

**Prompt tuning:** We tuned prompts to get the best performing agents following the methodology of Drouin et al. [2024]. 

> 7On WebArena and WorkArena we only use AXTrees because HTML is prohibitively large. On MiniWoB we use both AXTree and HTML as it consistently gives the best performance. 

> 8This new `infeas` primitive was introduced specifically for WorkArena++, and is made compatible with other benchmarks with infeasible tasks such as WebArena. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0019-00.png)


**----- Start of picture text -----**<br>
Example Prompt - L1 - Order Sales Laptop task<br># Instructions<br>Review the current state of the page and all other information to find the best<br>possible next action to accomplish your goal. Your answer will be interpreted<br>and executed by a program, make sure to follow the formatting instructions.<br>## Goal:<br>Go to the hardware store and order 6 "Sales Laptop" with configuration<br>{’Additional software requirements’: ’Slack, Zoom, Google Workspace, HubSpot, Adobe Creative Cloud’,<br>’Adobe Acrobat’: True, ’Adobe Photoshop’: False, ’Microsoft Powerpoint’: False, ’Siebel Client’: False}<br># Observation of current step:<br>## AXTree:<br>Note: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the<br>AXTree. Always use bid to refer to elements in your actions.<br>Note: You can only interact with visible elements. If the "visible" tag is not<br>present, the element is not visible on the page.<br>RootWebArea ’Catalog | ServiceNow’<br>...<br>[a] Iframe ’Main Content’, visible<br>RootWebArea ’Catalog’, focused<br>...<br>[a251] heading ’Hardware’, clickable, visible<br>[a252] link ’Hardware’, clickable, visible<br>...<br>[a261] link ’’, clickable, visible<br>[a262] table ’’, visible<br>[a263] rowgroup ’’, visible<br>[a264] row ’’, visible<br>[a265] gridcell ’’, visible<br>[a268] gridcell ’Hardware. Order from a variety of hardware to meet your business<br>needs, including phones, tablets and laptops. Order from a variety of hardware to meet<br>your business needs, including phones, tablets and laptops.’, clickable, visible<br>[a269] link ’Hardware. Order from a variety of hardware to meet your business<br>needs, including phones, tablets and laptops.’, clickable, visible<br>[a270] heading ’Hardware’, visible<br>...<br>## Focused element:<br>bid=’a85’<br># History of interaction with the task:<br>...<br># Action space:<br>Note: This action set allows you to interact with your environment. Most of them<br>are python functions executing playwright code. The primary way of referring to<br>elements in the page is through bid which are specified in your observations.<br>13 different types of actions are available.<br>...<br>fill(bid: str, value: str)<br>Description: Fill out a form field. It focuses the element and triggers an input event with the<br>entered text. It works for <input>, <textarea> and [contenteditable] elements.<br>Examples:<br>fill(’237’, ’example value’)<br>fill(’45’, ’multi-line\nexample’)<br>fill(’a12’, ’example with "quotes"’)<br>...<br>send_msg_to_user(text: str)<br>Description: Sends a message to the user.<br>Examples:<br>send_msg_to_user(’Based on the results of my search, the city was built in 1751.’)<br>Only a single action can be provided at once. Example:<br>fill(’a12’, ’example with "quotes"’)<br>...<br># Concrete Example<br>Here is a concrete example of how to format your answer.<br>Make sure to follow the template with proper tags:<br><think><br>From previous action I tried to set the value of year to "2022",<br>using select_option, but it doesn’t appear to be in the form. It may be a<br>dynamic dropdown, I will try using click with the bid "a324" and look at the<br>response from the page.<br></think><br><action><br>click(’a324’)<br></action><br>**----- End of picture text -----**<br>


**Figure 9:** Example prompt of our LLM-based agent. Some parts are truncated (...) for clarity. 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0020-00.png)


**Figure 10:** Example screenshot with set-of-marks. Note that elements [a252] and [a261] are both present here and in the prompt in fig. 9 as these are from the same observation. 

20 

## **C Task details** 

We show examples of goals given to the agent in the chat window under the "L2" difficulty and the task description seen by the agent under the "L3" difficulty for all the tasks within each skill category in WorkArena++. We also follow up (§ C.6) by showing all the ’protocols’ the agent can refer to when solving an "L3" difficulty task mimicing a real-world scenario faced by knowledge workers. 

## **C.1 Planning and Problem Solving (66** _×_ **2 tasks)** 

We devise tasks that require the agent to solve common enterprise problems that require decision making while respecting constraints. The tasks require the agent to follow a global plan that dictates the problems, constraints, and the desired outcome. 

## **C.1.1 Workload balancing** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0021-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0021-06.png)


**----- Start of picture text -----**<br>
(a)  Workloading balancing L2 goal.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0021-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0021-08.png)


**----- Start of picture text -----**<br>
(b)  Workloading balancing L3 task description.<br>**----- End of picture text -----**<br>


**Figure 11:** Workload balancing task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 

Redistribute non-uniformly assigned work across agents to make the workload balanced among them. We introduce more complexities based on the number of problems that need to be reassigned. In total, we have 3 workload balancing tasks across each L2 and L3 difficulty levels. We show the L2 goal and L3 task description in fig. 11. The protocol for the task is in fig. 23. 

## **C.1.2 Work assignment** 

Assign work (problems) across different categories such as software and hardware to relevant expert agents. We increase complexity by requiring the agent to assign critical problems based on their priority to more experienced agents under the different categories. In total, we have 6 work assignment tasks across each L2 and L3 difficulty levels. We show the L2 goal and L3 task description in fig. 12 for a work assignment task requiring the agent to assign problems based on the expertise of the category experts. The protocol for the task is in fig. 24. 

## **C.1.3 Scheduling requests** 

Schedule multiple problem requests based on constraints such as allowed time frame, overlap with other requests, impact and criticality of the problem, and allocated time to each request based on their risk. In total, we have 48 request scheduling tasks across each L2 and L3 difficulty levels. We show the L2 goal and L3 task description in fig. 13 for a scenario where the requests are to be scheduled in an order based on their impact. The protocol for the task is in fig. 25. 

## **C.1.4 Deduplicate problems** 

Mark problems that share the same problem statement as duplicates of each other. We introduce further complexity here for cases where the agent has to take into account the ’priority’ of the problem while 

21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-01.png)


**----- Start of picture text -----**<br>
(a)  Work assignment L2 goal.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-03.png)


**----- Start of picture text -----**<br>
(b)  Work assignment L3 task description.<br>**----- End of picture text -----**<br>


**Figure 12:** Work assignment task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-06.png)


**----- Start of picture text -----**<br>
(a)  Scheduling requests L2 goal.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-08.png)


**----- Start of picture text -----**<br>
(b)  Scheduling requests L3 task description.<br>**----- End of picture text -----**<br>


**Figure 13:** Scheduling requests task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-11.png)


**----- Start of picture text -----**<br>
(a)  Deduplicate problems L2 goal.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0022-13.png)


**----- Start of picture text -----**<br>
(b)  Deduplicate problems L3 task description.<br>**----- End of picture text -----**<br>


**Figure 14:** Deduplicate problems task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 

marking duplicates. In total, we have 9 problem deduplication tasks across each L2 and L3 difficulty levels. We show the L2 goal and L3 task description in fig. 14. The protocol for the task is in fig. 26. 

22 

## **C.2 Information Retrieval (39** _×_ **2 tasks)** 

To evaluate the retrieval ability of web agents, we formulate tasks requiring information extraction from visual elements such as plots and text elements such as forms. The agent is then requested to perform a diverse set of follow-up tasks using the retrieved information. 

## **C.2.1 Dashboard retrieval** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0023-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0023-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0023-05.png)


**----- Start of picture text -----**<br>
(a)  Dashboard retrieval L2 goal. (b)  Dashboard retrieval L3 task description.<br>**----- End of picture text -----**<br>


**Figure 15:** Dashboard retrieval task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 

Retrieve values or labels from bar plots and pie charts and use the information to perform subsequent tasks like filtering lists based on the values, creating new incidents using labels, order items low in stock, etc. In total, we have 29 dashboard retrieval tasks across each L2 and L3 difficulty levels. We show the L2 goal and L3 task description in fig. 15 for a task requiring an agent to place order for items low in stock after looking at a catalog chart. The protocol for the task is in fig. 27. 

## **C.2.2 List and form retrieval** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0023-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0023-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0023-11.png)


**----- Start of picture text -----**<br>
(a)  List/ form retrieval L2 goal. (b)  List/ form retrieval L3 task description.<br>**----- End of picture text -----**<br>


**Figure 16:** List/ form retrieval task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 

Retrieve information using filters and forms on the platform and perform subsequent tasks such as ordering items or providing an answer in the chat. In total, we have 10 dashboard retrieval tasks across each L2 and L3 difficulty levels. We show the L2 goal and L3 task description in fig. 16 for a task requiring extracting information from a form. The protocol for the task is in fig. 28. 

## **C.3 Data-driven decision making and reasoning (156** _×_ **2 tasks)** 

We evaluate the ability of web agents to interpret data, perform logical and mathematical reasoning, and take task-related decisions. 

23 

## **C.3.1 Expense management** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0024-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0024-02.png)


**----- Start of picture text -----**<br>
(a)  Expense management L2 goal.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0024-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0024-04.png)


**----- Start of picture text -----**<br>
(b)  Expense management L3 task description.<br>**----- End of picture text -----**<br>


**Figure 17:** Expense management task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 

Review expenses to identify those with duplicated descriptions and delete the duplicated expenses according to a hierarchical set of rules. The agent may have to identify the expense to delete based on a date, an amount, or other attributes. In total, we have 12 expense management tasks across each L2 and L3 difficulty levels. We show the L2 goal and L3 task description in fig. 17 for deleting expenses with duplicate descriptions while keeping the most expensive entry. The protocol for the task is in fig. 29. 

## **C.3.2 Maximize investment** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0024-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0024-09.png)


**----- Start of picture text -----**<br>
(a)  Maximize investment L2 goal.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0024-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0024-11.png)


**----- Start of picture text -----**<br>
(b)  Maximize investment L3 task description.<br>**----- End of picture text -----**<br>


**Figure 18:** Maximize investment task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 

Select a subset of investments that maximize returns while staying within an allowed budget. We introduce complexities by requiring the agent to use different ways to solve the task such as deleting the expenses that will not be selected, providing the IDs of the selected investments, or answering with the total value of the selected investments in the chat. In total, we have 57 investment management tasks across each L2 and L3 difficulty levels. We show the L2 goal and L3 task description in fig. 18 for an investment maximization task requiring the agent to delete expenses not being used. The protocol for the task is in fig. 30. 

## **C.3.3 Mathematical computations based on charts** 

Perform global calculations such as mean, median, and mode over bar plots and pie charts and use them to perform subsequent tasks such as assigning problems to underperforming agents or ordering gifts for overperforming agents. In total, we have 87 tasks across each L2 and L3 difficulty levels requiring doing computations over plots. We show the L2 goal and L3 task description in fig. 19 for a task requiring the agent to create ’probation’ calls for all workers performing lower than the median performance. The protocol for the task is in fig. 27. 

24 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0025-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0025-01.png)


- **(a)** Mathematical computations based on **(b)** Mathematical computations based on charts L3 task decharts L2 goal. scription. 

**Figure 19:** Mathematical computations based on charts task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 

## **C.4 Sophisticated Memorization (29** _×_ **2 tasks)** 

We explicitly test the ability of LLMs to memorize information necessary to behave as effective web-agents by devising tasks requiring following very clear yet information-intensive steps. 

## **C.4.1 Navigation + do** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0025-07.png)


- **(a)** Navigation + do L2 goal. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0025-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0025-10.png)


**----- Start of picture text -----**<br>
(b)  Navigation + do L3 task description.<br>**----- End of picture text -----**<br>


**Figure 20:** Navigation + do task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 

Navigate to a particular page on the platform and perform an aforementioned task. The agent has to remember details to navigate to the required page and details required to perform tasks like filling forms, sorting and filtering lists, and ordering items. In total, we have 26 navigation-and-do tasks across each L2 and L3 difficulty levels. We show the L2 goal and L3 task description in fig. 20 for a task requiring an agent to navigate to the relevant page and perform a sort operation using the given details. 

## **C.4.2 User management** 

The agent has to onboard and offboard users by memorizing information. Onboarding users involves creating a user profile, ordering devices, and assigning them to the user created. Offboarding user involves deleting assigned hardware and marking user inactive. In total, we have 2 user management tasks across each L2 and L3 difficulty levels. We show the L2 goal and L3 task description for offboarding a user in fig. 21. The protocol for the onboard user task is in fig. 31 and the offboard user task is in fig. 32. 

## **C.5 Contextual understanding through infeasible tasks (52** _×_ **2 tasks)** 

The agent must understand that some instructions are impossible to perform (e.g., filling a form field which does not exist, ordering an invalid item configuration), and notify the user using the chat. We show the L2 goal and L3 task description in fig. 22. 

25 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0026-00.png)


- **(a)** Offboard user L2 goal. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0026-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0026-03.png)


**----- Start of picture text -----**<br>
(b)  Offboard user L3 task description.<br>**----- End of picture text -----**<br>


**Figure 21:** Offboard user (User management) task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0026-05.png)


- **(a)** Infeasible L2 goal. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0026-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0026-08.png)


**----- Start of picture text -----**<br>
(b)  Infeasible L3 task description.<br>**----- End of picture text -----**<br>


**Figure 22:** Infeasible task: a) The goal given to the agent in chat for the L2 difficulty level. b) The description of the task assigned to the agent for the L3 difficulty level. 

## **C.6 Task Protocols** 

We present all the protocols in our knowledge base that the agent can use to solve tasks in the L3 difficulty level. This makes the task very similar to how knowledge workers would refer to company documents in real-world settings. 

26 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0027-00.png)


**Figure 23:** Workload balancing task protocol. 

27 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0028-00.png)


**Figure 24:** Work assignment task protocol. 

28 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0029-00.png)


**Figure 25:** Scheduling requests task protocol. 

29 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0030-00.png)


**Figure 26:** Deduplicate problems task protocol. 

30 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0031-00.png)


**Figure 27:** Dashboard retrieval task protocol. 

31 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0032-00.png)


**Figure 28:** List/ form retrieval task protocol. 

32 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0033-00.png)


**Figure 29:** Expense management task protocol. 

33 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0034-00.png)


**Figure 30:** Maximize investment task protocol. 

34 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0035-00.png)


**Figure 31:** Onboard user task protocol. 

35 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0036-00.png)


**Figure 32:** Offboard user task protocol. 

36 

## **D Evaluation Curriculum Details** 

Though WorkArena++ consists of 682 tasks, each task can be instantiated with different configurations controlled by a task seed to get a task ’instance’. For example, the task requiring to maximize investment returns can have a random value for the maximum investment allowed. This value is generated using a random generator initialized using a task-level seed. Hence, to make evaluation across numerous models and agents feasible and also have reproducibility across evaluations, we propose a standardized way to sample task instances given tasks across skill categories. We aim to design a curriculum that ensures we cover all the skills in our categorization while having uniformity across these categories. 

We start by creating buckets of tasks across each skill category based on the requirement of novel skills and abilities to solve the task. For example, ordering an iPad would essentially require the same abilities for an agent as ordering a Macbook, given the same interface and interactions necessary. Hence, these tasks would fall under the same bucket. We thus create a hierarchical categorization of all the tasks starting with the skill categories ( §3.2) and dividing the tasks within each skill category (§ C) into buckets. Within a bucket, we sample tasks with weights ensuring maximum skill coverage and minimum redundancy. Each skill is sampled with different number of seeds to ensure uniformity across each category. 

Our curriculum only requires one meta-seed (m) to sample fixed configurations to create task instances across all the tasks in each category. Consider each skill category has task "buckets" and a corresponding "weight" for each bucket. Then for each category to be sampled "num_seeds" number of times, our curriculum follows the algorithm to get the WorkArena++ "Benchmark" _T_ for reproducible and uniform evaluation, 

## **Algorithm 1** Agent Curriculum for WorkArena++ Benchmark 

CATEGORIES _←_ Skill categories with bucket, weight, and num_seeds information m _←_ Meta seed for reproducibility 

_T_ =() _←_ Tuple of (task, seed) comprising the task and its task seed to instantiate the task with a configuration 

random_generator = random_generator_init(m) **for** category **in** CATEGORIES **do** 

seeds = random_generator(CATEGORIES[category]["num_seeds"]) task_buckets = CATEGORIES[category]["buckets"] task_bucket_weights = CATEGORIES[category]["weights"] **for** seed **in** seeds **do** random_bucket_generator = random_generator_init(seed) **for** task_bucket, task_bucket_weight **in** zip(task_buckets, task_bucket_weights) **do** tasks = random_bucket_generator.sample(task_bucket, task_bucket_weight, replace=False) **for** task **in** tasks **do** _T ←T_ +(task, seed) 

return _T ←_ WorkArena++ Benchmark 

We reserve seeds (m) 0-9 for evaluation and the remaining seeds may be used for agent tuning. The "CATEGORIES" dictionary is shown in fig. 33. Our curriculum yields 235 tasks for each L2 and L3 difficulties. 

37 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0038-00.png)


**Figure 33:** Agent curriculum for the WorkArena++ Benchmark. 

38 

## **E Expanded Features** 

## **E.1 Application Themes** 

To add visual diversity, we created 10 themes of fictitious companies. They are all listed below. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0039-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0039-04.png)


**----- Start of picture text -----**<br>
(a)  landing page<br>(b)  list view<br>**----- End of picture text -----**<br>


**Figure 34:** Astranova theme 

39 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0040-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0040-01.png)


**----- Start of picture text -----**<br>
(a)  landing page<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0040-02.png)


**(b)** list view **Figure 35:** Charlie’s cookies theme 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0040-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0040-05.png)


**----- Start of picture text -----**<br>
(a)  landing page<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0040-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0040-07.png)


**----- Start of picture text -----**<br>
(b)  list view<br>**----- End of picture text -----**<br>


**Figure 36:** Great Pasta theme 

40 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0041-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0041-01.png)


**----- Start of picture text -----**<br>
(a)  landing page<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0041-02.png)


**(b)** list view **Figure 37:** Mighty Capital theme 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0041-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0041-05.png)


**----- Start of picture text -----**<br>
(a)  landing page<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0041-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0041-07.png)


**----- Start of picture text -----**<br>
(b)  list view<br>**----- End of picture text -----**<br>


**Figure 38:** Skyward theme 

41 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0042-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0042-01.png)


**----- Start of picture text -----**<br>
(a)  landing page<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0042-02.png)


**(b)** list view **Figure 39:** SpeedyTires theme 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0042-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0042-05.png)


**----- Start of picture text -----**<br>
(a)  landing page<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0042-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0042-07.png)


**----- Start of picture text -----**<br>
(b)  list view<br>**----- End of picture text -----**<br>


**Figure 40:** TurboBots theme 

42 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0043-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0043-01.png)


**----- Start of picture text -----**<br>
(a)  landing page<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0043-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0043-03.png)


**----- Start of picture text -----**<br>
(b)  list view<br>Figure 41:  UltraShoes theme<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0043-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0043-05.png)


**----- Start of picture text -----**<br>
(a)  landing page<br>(b)  list view<br>**----- End of picture text -----**<br>


**Figure 42:** VitaSphere theme 

43 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0044-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0044-01.png)


**----- Start of picture text -----**<br>
(a)  landing page<br>(b)  list view<br>Figure 43:  WorkArena theme<br>**----- End of picture text -----**<br>


44 

## **E.2 Extracting traces and designing more tasks** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0045-01.png)


**Figure 44:** Extracting traces. In this example, the goal of the task is to order an iPad pro from the service catalog. In the first step, the task starts from the ServiceNow landing page. BrowserGym extracts a screenshot of this page + the page’s accessibility tree. In step 2, the oracle function is used to navigate to the service catalog using Playwright code. This action is recorded and saved as a trace. Finally, in step 3 the oracle purchases an iPad pro by navigating to the item page and ordering it. All these actions are recorded and saved alongside the accessibility tree of the pages where the actions were performed. 

**Extracting traces** All tasks from WorkArena and WorkArena++ implement an "oracle" function that solves them step-by-step with Playwright code. To extract traces of solutions, we leverage BrowserGym, which extracts a screenshot of the page, the AxTree, the DOM as well as other optional features (e.g. set-of-marks) and we intercept the Playwright code outputted by our oracle function. Together, this allows us to generate traces for the resolution of tasks in WorkArena++. The process is illustrated in fig. 44. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0045-04.png)


**Figure 45:** User OnBoarding task 

**Designing new tasks** All existing tasks in WorkArena can be combined to create new workflows. When chaining tasks to create compositional workflows, the individual tasks’ oracle functions are called in succession to solve the task step-by-step. Moreover, the validation of these task can either be a sequential validation of the individual tasks or a global validation made by querying the database. This affords the possibility of combining low-level tasks to create realistic compositional tasks or modify existing ones to make them more complex. For example, the basic user onboarding task consists in creating a user, ordering a laptop for them and finally assigning the laptop to the user that was created. As this task is sequential, all of its individual steps are validated one by one. It is illustrated in fig. 45. 

45 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0046-00.png)


**Figure 46:** Extended User OnBoarding task 

Suppose this workflow needs to be modified by adding a new step: Creating a first task assigned to the user. This can easily be done by adding a new task to the workflow and validating it. This is illustrated in fig. 46. 

46 

## **F Error Analysis** 

In the following section, we will highlight some examples of errors made by our agents. 

## **Information Retrieval** 

In BrowserGym, element in the page are assigned an identifier called bid, that can be used in the action functions. Normal element are assigned numbers, iframes are assigned letters, and elements within iframes are assigned the combination of the iframe bid and a number. For example, element ’324’ in iframe ’a’ will have the bid ’a324’. 

In the following example, the agent sometimes gets confused by the bid, and as in this example, forgets the iframe identifier. This error can have one of two consequences. The bid exists, and the model clicks on a random element of the web page. The bid does not exist, and the action causes an error, effectively not doing anything on the web page. 

- **Axtree** `... [a252] link ’Hardware’, clickable, visible` 

```
...
```

## **LLM Output** 

```
<think>
Fromthepreviousaction,Iclickedonthe"Software"link.Now,Ineed
tonavigatetothe"Hardware"sectiontofindandordertheiPadPro.I
canseethe"Hardware"linkwithbid’252’.Iwillclickonthislink
toproceedtothehardwarestore.
</think>
<action>
click(’252’)
</action>
```

This action causes an error that can be fed to the agent on the next step. 

## **Information Retrieval** 

The model sometimes struggles with understanding the information that are available in the page. In the next example, it has to sort a table by two criterion, in a certain order. On this step, the table is already sorted by ascending ’Number’. The agent seems to see this information, as it is mentioned in the think section. However, the agent does not seem to understand it, as it keeps on clicking on the column header to sort it. 

**Goal** Goal: Sort the change request list. Concretely, you need to complete the following steps: 

1. Navigate to the "All" module of the "Change" application. 

2. Sort the "change requests" list by the following fields: 

   - **Number** (ascending) 

   - **Close code** (ascending) 

## **LLM Output** 

```
<think>
```

```
Ineedtosortthe"changerequests"listbythe"Number"fieldin
ascendingorder.
```

```
Iwillclickonthe"SortedinascendingorderbyNumber"button.
```

47 

```
</think>
<action>
click(’1116’)
</action>
```

The column header is indicated as ’Sorted in ascending order by Number’, and yet the agent tries to sort it. 

## **Exploration** 

The `all-menu` (fig. 47) is a key component for navigating the WorkArena environment. It can sometimes be an obstacle for the agent, as it is composed of many foldable sub-menus. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0048-04.png)


**(a)** Folded version of the `all-menu` 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0048-06.png)


**(b)** Unfolded version of the `all-menu` 

**Figure 47:** `all-menu` 

## **Goal** 

Goal: Sort the hardware asset list. Concretely, you need to complete the following steps: 

1. Navigate to the _Portfolios > Hardware Assets_ module of the _Asset_ application. 

2. Sort the "hardware" list by the following fields: 

   - **Due in** (ascending) 

   - **Model category** (ascending) 

   - **Stockroom** (descending) 

48 

In the above example, as the _Asset_ sub-menu is folded, its links are not available in the axtree. The agent needs to either lookup for _Hardware Assets_ in the filtering search bar, of click on the _Asset_ sub-menu, and then click on the _Hardware Assets_ link that unfolds. The agent is sometimes not able to find those sub-menus and struggles to get to the right application. 

## **Exploration** 

Another example of failed exploration occurs when filling forms. Forms have sections that require writing in information, which can be split into separate tabs in the form. 

## **Goal** 

Goal: Create a new "Normal" change request Concretely, you need to complete the following steps: 

1. Navigate to the "All" module of the "Change" application. 

2. Create a new change request with a value of "gigantean stageworthy plowable gagership schairerite" for field "Justification", a value of "training brutely brab galloptious acetanilide" for field "Risk and impact analysis", a value of "Paeonia taurocholic springmaking smattery Darren" for field "Test plan", a value of "midmonthly pimploe roosted uncowl nothingarian" for field "Implementation plan", a value of "Java Application Server change" for field "Short description", a value of "Java Application Server FLX" for field "Configuration item", a value of "3 - Low" for field "Impact", a value of "Other" for field "Category", a value of "CHG0000010" for field "Number", **a value of "cephalobranchiate resuscitant flashingly gauchely aclidian" for field "Close notes"** , and a value of "webfooter parapraxia Typhoean bockeret Labrador" for field "Backout plan". 

In fig. 48, the agent has filled all but one section of the form. The remaining section ’Close notes’ is in the ’Closure Information’ tab, and the agent needs to switch to the tab before being able to fill this last section. However, it decides to fill another section, overwriting old content (fig. 49). 

## **Action Consequences Assessment - Hallucinated Consequences** 

Interestingly, from fig. 49, the error compounds as the model considers its job done, and proceeds to submit the form as is, giving the following thought/action: 

## **LLM Output** 

```
<think>
Ihavealreadyfilledinmostoftherequiredfieldsforthenew
changerequest.Theonlyremainingactionistosubmittheform.The
"Submit"buttonisidentifiedbybid’a1229’.Iwillclickthisbutton
tocompletethecreationofthenewchangerequest.
</think>
<action>
click(’a1229’)
</action>
```

This ends the run, causing the task to fail as the form is not filled the correct way. 

## **Hallucinations - Made Up Actions** 

The model occasionally comes up with functions that would help it complete the task. Typically, the model has sometimes used functions like `check` and `uncheck` , that are not available in the highlevel action set provided to the model. Those could possibly be reminiscences of playwright data during the model pre-training that would interfere with our action space. 

In the same way, the model sometimes tries to shortcut the function inputs, especially by replacing the bid by its target’s name, causing an error: 

49 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0050-00.png)


**Figure 48:** Request form on track to be fully filled 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0050-02.png)


**Figure 49:** The agent fills the wrong section of the form with the last piece of information 

## **LLM Output** 

```
<think>
Iamcurrentlyonthe"ServiceCatalog"page.Ineedtonavigateto
thehardwarestoreandorder5"iPadmini"withconfiguration’Choose
thecolour’:’Purple’,’Choosethestorage’:’64’.Ihavealready
searchedfor"iPadmini"inthesearchcatalogfieldandclickedonthe
"iPadmini"linktoaccessthehardwarestore.Now,Iwillselectthe
"Purple"coloroption.
</think>
<action>
select_option(’colour’,’Purple’)
</action>
```

## **Hallucinations - Imaginary Buttons** 

When stuck, the model sometimes tries to cheat using buttons that gives the solution to the task, or a part of it: 

50 

## **LLM Output** 

```
<think>
```

```
Tonavigatetothe"ServiceCatalog"moduleofthe"Self-Service"
application,Ineedtoexploretheavailablemenus.The"All"button
seemstobethemainmenuwhereIcanfinddifferentmodules.Iwill
startbyexpandingthe"All"menutolookforthe"ServiceCatalog"
option.
</think>
<action>
click(’button:has-text("All")’)
</action>
```

Here the model makes up a button that would find the correct bid automatically, which causes an error. 

## **Hallucination - Asking for help** 

The L3 added complexity can sometimes cause the agent to not understand why the task is not completed. In the following example, the model wrongfully changed the form’s ’State’ field to ’Closed Complete’. It then hallucinates the task to be complete, and believes there should be done, and asks for confirmation from the user. 

## **LLM Output** 

```
<think>
```

```
Thetaskistoorderastandardlaptopfromtheservicecatalogand
markthetaskas"Closed-complete"oncesuccessfullycompleted.The
"State"fieldwassetto"ClosedComplete"andthe"Update"buttonwas
clickedtosavethechanges.
```

```
Toconfirmthatthetaskhasbeensuccessfullymarkedascompleted,
Ineedtoverifythecurrentvalueofthe"State"field.Sincethe
"State"fieldwasalreadysetto"ClosedComplete"before,Iwill
nowcheckifthereareanyfurtheractionsrequiredorifthereisa
confirmationmessageindicatingthatthetaskhasbeensuccessfully
completed.
```

```
Iwillinspectthe"State"fieldtoensureitissetto"Closed
Complete"andlookforanyconfirmationmessagesorchangesintheform
thatindicatethetaskhasbeensuccessfullyupdated.
</think>
```

```
<action>
```

```
send_msg_to_user(’Thetaskhasbeenmarkedas"Closed-complete".
Pleaseverifyifthereareanyfurtheractionsrequiredorifthereis
aconfirmationmessageindicatingthatthetaskhasbeensuccessfully
completed.’)
</action>
```

## **Goal Understanding - Thought/Action Consistency** 

In the following task: 

## **Goal** 

Goal: Order an iPad Mini from the service catalog Concretely, you need to complete the following steps: 

1. Navigate to the "Service Catalog" module of the "Self-Service" application. 

2. Go to the hardware store and order 5 "iPad mini" with configuration ’Choose the colour’: ’Purple’, ’Choose the storage’: ’64’ 

The agent at some point performs the following action: 

51 

## **LLM Output** 

```
<think>
Iamcurrentlyonthe"BelkiniPadMiniCase"page.Ineedtonavigate
tothehardwarestoreandorder5"iPadmini"withconfiguration
’Choosethecolour’:’Purple’,’Choosethestorage’:’64’.Iwill
clickonthe"OrderthisItem"buttonwithbid’a237’toproceedwith
theorder.
</think>
<action>
click(’a237’)
</action>
```

In this example the model orders an iPad case, instead of the requested iPad Mini. Ordering is considered the end of the task in WorkArena, causing this run to fail. The model seems inconsistent with himself, as it asserts it needs to "navigate to the hardware store", but proceeds to order the case anyways. 

## **Goal Understanding - Low-Level Understanding in L3** 

L3 task come with the added complexity that the task is described in a ticket. Sometimes the agents do not understand that this is a their task, and see the information in the tickets as fulfilled tasks. For example, in fig. 50, the agent fails to understand that the information it is reading is the task at hand. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0052-05.png)


**Figure 50:** Task ticket from L3 

This results in the action: 

## **LLM Output** 

```
<think>
ThecurrentpageisaServiceNowinterfacewithatasktocreateanew
hardwareasset.Thetaskdetailsarevisible,andtheformfieldsare
alreadyfilledwiththerequiredinformation.Thenextlogicalstep
istomarkthetaskas"Closed-complete"sincethetaskappearstobe
successfullycompleted.
Todothis,Ineedtochangethe"State"fieldfrom"Open"to"Closed
Complete".The"State"fieldisacomboboxwithbid’a270’.Iwill
selecttheoption"ClosedComplete"fromthedropdown.
</think>
<action>
select_option(’a270’,’ClosedComplete’)
</action>
```

52 

The agent mistakes the information and considers the task done, closing the ticket. 

## **Action Consequences Assessment - Hallucinated Consequences** 

Because of other errors, agents can be lead to think they have successfully completed parts of the tasks. Typically, when completing forms, lack of exploration can lead them to fill information in the wrong fields. However, instead of correcting those errors, they will go on and consider this part of the task done. The example below illustrates this error: 

## **Goal** 

**Goal:** Create a new hardware asset Concretely, you need to complete the following steps: 

1. Navigate to the "Portfolios > Hardware Assets" module of the "Asset" application. 

2. Create a new hardware asset with a value of "Computer" for field "Model category", a value of "Apple iMac 27"" for field "Model", a value of "Apple" for field "Vendor", a value of "SN-566be2cf-3f63-4e26-8aa5-54ad26c07875" for field "Serial number", a value of "IT" for field "Cost center", a value of "unteethed Artemisia solvent chronogenesis nebalian" for field "Disposal reason", a value of "voltatype sideling uncontentingness pant" for field "Lease contract", a value of "SL 5 Years" for field "Depreciation", a value of "Primary" for field "Asset function", a value of "" for field "Request line", a value of "ACME North America" for field "Company", and a value of "" for field "Owned by". 

Here, the model has successfully completed the fields for this tab in figure 51. Unable to find the next field because it is in the ’Financial’ tab, the model fills the ’Serial Number’ field with the ’Cost center’ information. After this mistake, the model then finds how to change tabs, and does it thinking this tab is filled correctly: 

## **LLM Output** 

```
<think>
Ihavesuccessfullyfilledinthe"Serialnumber"fieldwiththevalue
"SN-566be2cf-3f63-4e26-8aa5-54ad26c07875".Thenextfieldtofill
isthe"Costcenter"withthevalue"IT".The"Costcenter"fieldis
notvisibleinthecurrentobservation,soIneedtonavigatetothe
appropriatetabwherethisfieldislocated.BasedontheAXTree,the
"Financial"tablikelycontainsthe"Costcenter"field.Iwillclick
onthe"Financial"tabtoproceed.
</think>
<action>
click(’a250’)
</action>
```

The agent asserts that it has filled the ’Serial Number’ field, but it does not take into account the fact that it overwrote it right after. 

## **Action Consequences Assessment - Repeated Actions** 

Agents can sometimes fall into a loop, where they will retry the same action over and over. Typically, if a piece of information is missing on the page, the agents can sometimes start scrolling aimlessly over and over, trying find the missing data. This process can go on for the rest of a run, or randomly stop if the agent manages to get out of this mode. From the state in fig. 53, the agent must go into the ’Hardware’ section and buy a ’Loaner Laptop’. 

At some point, it falls into a loop and only outputs the following action: 

53 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0054-00.png)


**Figure 51:** The agent successfully completed the ’Serial Number’ field. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0054-02.png)


**Figure 52:** The agent overwrites the ’Serial Number’ field with another field. 

## **LLM Output** 

```
<think>
Thepreviousattemptstoclickonthe"Hardware"linkandusethesearch
functionalityhavefailed.Ihavetriedscrollingthepagedownto
makethe"Hardware"linkvisible,butitseemsthattheelementisstill
notinteractable.
```

```
Inoticedthatthereisa"Search"comboboxwithbid’113’thatis
visibleandclickable.Iwillattempttoclickonthiscomboboxto
focusonitandthentype"Hardware"tosearchfortherelevantsection.
</think>
<action>
click(’113’)
</action>
```

In this example, bid 113 corresponds to the search bar. As clicking on it does not change the page or cause any error, the model keeps on repeating this action until the end of the episode. 

54 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks_images/nips_2024_WorkArena++_Towards_Compositional_Planning_and_Reasoning-based_Common_Knowledge_Work_Tasks.pdf-0055-00.png)


**Figure 53:** Service Catalog causing the model to loop over the same useless action 

## **G Additional Discussion** 

## **G.1 Limitation: Restricted to ServiceNow** 

The tasks in WorkArena++ are confined to ServiceNow and do not involve interacting with other software. It is thus reasonable to wonder: **(Q1)** whether insights from WorkArena++ are expected to generalize to other enterprise software environments and **(Q2)** if including more software environments would result in a benchmark that is more representative of general enterprise workflows. We hereby address those questions. 

**Q1:** We believe that the answer is yes for three main reasons. First, our tasks are built around basic UI components (e.g., lists, forms, dashboards) that are ubiquitous across various enterprise software platforms (e.g., SalesForce, SAP). Second, as highlighted in our error analysis (§4.5), most failures are not tied to ServiceNow-specific issues. Instead, they stem from challenges like understanding the task goal, hallucinating actions, etc. Third, and most importantly, the tasks primarily evaluate complex reasoning and planning skills in agents, which are largely independent of the specific user interface. 

**Q2:** Yes, definitely. However, we chose to limit the scope of the present benchmark to ServiceNow for the following reasons: 

- WorkArena++ integrates into the broader BrowserGym [Drouin et al., 2024] ecosystem, which already contains benchmarks that evaluate cross-application performance (e.g., WebArena; Zhou et al. [2023]). 

- ServiceNow Personal Developer Instances allow for a simple user experience, where the user does not have to launch a web server to run evaluations on the benchmark (as is common in related work). 

- Tasks in WorkArena++ are compositions of atomic subtasks that involve interacting with common UI elements. Hence, the benchmark can easily be extended with tasks inspired by workflows that are relevant in other software environments. 

That said, we acknowledge that creating multi-environment tasks is an exciting and promising direction for future work aimed at expanding this benchmark (e.g., to make an L4 set of tasks). For example, one could replace the built-in ServiceNow knowledge base in L3 tasks with an external one (e.g., Wikipedia). Realistically, all environments used in related work (e.g., WebArena) are within reach. 

## **G.2 Expanding the Benchmark – Ensuring Long-Term Relevance** 

How can WorkArena++ be updated or expanded to remain challenging and relevant? The modular design of WorkArena++ makes it easy to add new tasks. One can create complex workflows by composing “atomic tasks”. Creating new “atomic tasks” simply involves writing setup, teardown, 

55 

oracle, and validation functions (as shown in fig. 3). For an example of compositional task creation, please refer to the `GetWarrantyExpirationDateTask` task in the code base. 

Moreover, there are many potential directions in which the benchmark could be expanded as the performance of agents increases. Examples include: 

- **Longer and more complex workflows:** Inspiration could be drawn from the O*Net database [National Center for O*NET Development, 2024], which catalogs key tasks performed by job occupation (https://www.onetonline.org/). 

- **Workflows that require collaboration:** This could include tasks where agents must delegate subtasks to colleagues (either agentic or human) or collaborate toward completion. One interesting observation is that the design of WorkArena++ makes it fairly easy to implement tasks where an agent must delegate work. One simply needs to run the “oracle function” for the atomic task (or trajectory) that was delegated by the agent. The agent could be penalized for too many delegations. The ServiceNow platform has collaboration features that would make the elaboration of such tasks feasible (e.g., discuss via chat, leave comments for one another on a ticket, etc.). 

- **Workflows that involve interacting with multiple external software:** The number of such potential tasks is enormous. The main complexity is in interfacing with such software to implement setup, teardown, oracle, and validation functions (see fig. 3). 

We keep this for future work but welcome contributions from the open-source community. 

## **G.3 Context-size sensitivity** 

We ran an experiment to compare the impact of context-length on overall agent performance for WorkArena-L1. For this, we compared the performance of a Llama3-based agent using the full context (130k), 50% context (64k), 25% context (32k), and 10% context (13k). This experiment resulted in very similar performance across all different allowed context lengths. Even though this result is interesting, we would like to caution that similar performances across different context lengths should not be expected on benchmarks with more complex tasks, as these will likely require greater memory and detail in the prompts. 

**Table 3:** Ablation study for **Llama3** on WorkArena-L1. Success rate _±_ Standard error (SR _±_ SE). 

|**Context size**|**130k**|**64k**|**32k**|**13k**|
|---|---|---|---|---|
|SR % _±_SE|17.9 _±_2.1|14.8 _±_2.0|17.6 _±_2.1|18.2 _±_2.1|



56 

