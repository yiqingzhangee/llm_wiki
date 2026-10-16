# **COLD: Causal reasOning in cLosed Daily activities** 

**Abhinav Joshi** _[∗]_ **Areeb Ahmad** _[∗]_ **Ashutosh Modi** Department of Computer Science and Engineering Indian Institute of Technology Kanpur (IIT Kanpur) Kanpur, India {ajoshi,areeb,ashutoshm}@cse.iitk.ac.in 

## **Abstract** 

Large Language Models (LLMs) have shown state-of-the-art performance in a variety of tasks, including arithmetic and reasoning; however, to gauge the intellectual capabilities of LLMs, causal reasoning has become a reliable proxy for validating a general understanding of the mechanics and intricacies of the world similar to humans. Previous works in natural language processing (NLP) have either focused on open-ended causal reasoning via causal commonsense reasoning (CCR) or framed a symbolic representation-based question answering for theoretically backed-up analysis via a causal inference engine. The former adds an advantage of real-world grounding but lacks theoretically backed-up analysis/validation, whereas the latter is far from real-world grounding. In this work, we bridge this gap by proposing the **COLD** ( **Causal reasOning in cLosed Daily activities** ) framework, which is built upon human understanding of daily real-world activities to reason about the causal nature of events. We show that the proposed framework facilitates the creation of enormous causal queries ( _∼_ 9 million) and comes close to the mini-turing test, simulating causal reasoning to evaluate the understanding of a daily real-world task. We evaluate multiple LLMs on the created causal queries and find that causal reasoning is challenging even for activities trivial to humans. We further explore (the causal reasoning abilities of LLMs) using the backdoor criterion to determine the causal strength between events. 

## **1 Introduction** 

In recent times, Large Language Models (LLMs) have shown remarkable generalization capabilities [Devlin et al., 2019, Radford et al., 2019, Brown et al., 2020]. Consequently, the ability to perform causal reasoning (often considered a core feature of intelligence [Penn and Povinelli, 2007, Pearl and Mackenzie, 2018]) has sparked research interest in the context of LLMs, aiming to answer if causal reasoning is possible with LLMs [Weber et al., 2020, Jin et al., 2023, 2024, Cohrs et al., 2023, Romanou et al., 2023, Yang et al., 2023, Mitchell et al., 2023, Vashishtha et al., 2023, Stolfo et al., 2023]. On a broader level, there are two lines of work; first, that treats the causal reasoning via learning relationships between the events that are grounded in the real world [Gordon et al., 2012, Ho et al., 2022, Zeˇcevi´c et al., 2023, Zhang et al., 2023, Wang et al., 2023]. Second line of work relies on a causal inference engine and establishes relationships between variables via symbolic representation [Jin et al., 2023, 2024]. The former relies on understanding real-world events but lacks formal definitions that adhere to the causal inference theory. The latter solves the issue using a causal inference engine but uses symbolic representations not grounded in the world, making the causal queries more like a test for the understanding of causal theory. Though the first line of work includes real-world events, the causal queries are often limited and could be answered by memorizing the causal relationships between the events. Recent findings that include rigorous analysis using a causal inference engine claim LLMs to be _“Causal Parrots”_ [Zeˇcevi´c et al., 2023], i.e., the LLMs tend to 

> _∗_ Equal Contribution 

pick up (memorize) patterns in the training data to perform well on the causal reasoning benchmarks. Moreover, some initial findings by Tang et al. [2023] suggest that LLMs perform significantly better when semantics are consistent with commonsense but struggle to solve symbolic tasks, pointing towards semantic representation to be better for proper validation of LLMs, leading to a conclusion that an in-depth analysis using real-world events is necessary. 

In this work, we bridge the gap between the two approaches by proposing the **COLD** ( **Causal reasOning in cLosed Daily activities** ) framework, based on the human understanding of realworld daily activities capturing commonsense (for example, “making coffee,” “boarding an airplane,” etc.), that adheres to causal theory literature. It is more natural to frame real-life reasoning-based queries via language; consequently, we follow the literature on Causal Commonsense Reasoning (CCR), which studies the relationships between real-world events (described via natural language). CCR is a non-trivial task of estimating the causeand-effect relationship between events that are **Intention to perform a task** studied under the umbrella of commonsense **Go to U** (traveling by an airplaneplanting a tree, going , reasoning [Kuipers, 1984, Gordon et al., 2012, **Airport** grocery shopping, etc. ) Zhang et al., 2022b, Wang et al., 2023, Chun **Check-** et al., 2023, Du et al., 2022]. The events in CCR **Luggagein Search** generally refer to actions taking place in an ac- **for the boardin g gate Wait** tivity in the real world.the activity of “travelingFor example, considerby an airplane” Temporal LinkCausal Link **luggagefor . luggageTake back** given in Fig. 1, where the occurrence of all the Cause Event **AirportLeave** events is confounded by a universal variable **U** Effect Event(s) (“intention to perform a task”). Moreover, there are a few events that cause one another. For exFigure 1: **U** denotes the unobserved variables, conample, the event “checking in luggage” ( _E_ 1) founding all events present in a real-world activcaused the occurrence of events like “waiting ity. In an activity, some events cause other events at the luggage belt” ( _E_ 2) after the flight, to happen. For example, in _“traveling by an air-_ i.e., in an alternate universe where one does not _plane”_ , the event of _“check-in luggage”_ causes checks in luggage and goes with the cabin bags, events like “taking back luggage.” will never wait for their luggage after the flight 

has landed. Moreover, some of the events have no causal impact, like “find the boarding gate” ( _E_ 3) has no causal relationship with “checking in luggage” ( _E_ 1). More formally, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0002-03.png)


where _do_ ( _._ ) denotes the **do** operator [Pearl, 2012] showing the intervention on _E_ 1, and ∆ is the _causal estimand_ capturing the causal strength between two events, i.e., ∆( _E_ 1 _→ E_ 2) is expected to be higher when compared to ∆( _E_ 1 _→ E_ 3). Note, CCR excludes the causal events that are beyond the reach of commonsense knowledge, for example, does “planting trees” have a direct impact on the “rainy season”? or does “providing free education” improve the “economic condition of the country/state”; does “carpooling” directly impact “air pollution”, etc. A noteworthy point concerning causality is that though the logical temporal (or prototypical) order of these events provides a weak signal about causal relationships, temporal precedence does not always imply causation (§2). For example, one could erroneously argue that “boarding a plane” is also the cause of “waiting at the luggage belt” since without “boarding a plane,” one cannot wait for the luggage belt. 

For building a causal reasoning framework (based on CCR) around real-life daily activities, one would require a few primary features readily available: 1) Clear distinction between the events, i.e., the events should encapsulate describing a particular step in an activity, 2) Causal Dependency between the variables/events, i.e., there should be some events causing other events to occur. 3) Causal independence of events with the rest of the world, i.e., the occurrence of events should be independent of events that are not part of the activity (i.e., the covariates are balanced out). We found that _“Scripts”_ [Schank, 1975, Schank and Abelson, 1975] provide a concrete medium that satisfies all these requirements. Scripts are defined as a sequence of events describing a prototypical activity, such as going to a restaurant, and hence capture commonsense knowledge about the world. [Schank and Abelson, 1975, Modi et al., 2016, Wanzare et al., 2016, Ostermann et al., 2018, Modi, 2016, 2017, Modi et al., 2017, Modi and Titov, 2014]. Moreover, different people have similar understandings of the activity in the form of scripts that inherently balance out the covariates present in the real world, 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0003-00.png)


**----- Start of picture text -----**<br>
The pre-world and post-world activities involve various  Causal Graph for the Activity “Going Grocery Shopping”<br>events that take place outside the context of daily<br>get in car<br>activities, making them independent of the events<br>happening during the activity. turn-on car go by walk go by bus<br>make list<br>drive to grocery<br>carry list park car<br>get groceries<br>get list from car enter store<br>bring vehicle<br>from parking<br>go to checkout<br>pack groceries<br>present coupons<br>check list put conveyor<br>(if anything is left) get wallet  cashier<br>scan/weight<br>take shop cart<br>take bags<br>pay<br>use bags<br>put bags in cart return cart to corral get receipt<br>Begin  Start  Start<br>again another  work Colliders denoting Conditional Dependence<br>relax activity Independent Event (may or may not be present in an activity<br>and is not a cause or effect of any of the other events)<br>world<br>Pre activity<br>Taking a Bath Riding on a Bus Going on a Train<br>Flying in an airplane<br>Going Grocery Shopping<br>Steps performed in daily life activities  (capturing commonsense knowledge)<br>world<br>Post activity<br>**----- End of picture text -----**<br>


Figure 2: **Left:** the figure represents the closed nature of daily real-world activities (capturing commonsense, commonly understood by humans), start and end given the context of the task, i.e., the pre-activity world and post-activity world activities marginalize out the dependence of event occurring during the activity with the rest of the world. **Right:** Causal Graph for “going grocery shopping.” Notice the collider (red nodes) makes the independent set of nodes (highlighted in different colors) unconditionally independent in the causal graph. In contrast, when given a condition on a collider (“put bags in cart”, the two clusters (yellow and blue) become dependent (if collider is observed, both yellow and blue clusters may have been observed as well). 

i.e., all the activities have the same starting and ending point and account for common exogenous and endogenous variables, providing a suitable platform to establish a cause-and-effect relationship between the events. In other words, for an activity like “flying in an airplane,” or “going grocery shopping” (also see Fig. 2, left) the events that happened before starting the activity and after completing the activity are marginalized out using a common understanding of these activities by different humans and hence will have no causal relations with any of the exogenous events during the activity. Creating a causal graph for script knowledge, i.e., establishing relationships between events taking place during the activity, provides a perfect platform for creating causal queries, thus providing a medium to establish CCR between events. In a nutshell, we make the following contributions: 

- We propose **COLD** ( **Causal reasOning in cLosed Daily activities** ), a CCR framework based on Script knowledge (daily activities involving commonsense) that provides a closed system to test the understanding of causal inference grounded in the real-world. The proposed framework adheres to SUTVA (Stable Unit Treatment Value Assumption) [Cox, 1958, Rubin, 1980] by design (§3). **COLD** consists of activity-specific observational graphs (created via crowd-sourcing) and causal graphs. Further, **COLD** facilitates creating an enormous number of causal queries (e.g., 2 _,_ 887 _,_ 950 per activity) via causal query triplets from the causal graph. This comes close to the mini-Turing test [Pearl and Mackenzie, 2018], where the story becomes the understanding of the daily activity, and the sampled enormous causal queries help in the exhaustive and rigorous evaluation of LMs. 

- We devise various design mechanisms for estimating causal strength analytically and show how the representations learned by language models can be validated. 

- Via detailed experimentation on the widely used open-weight language models, including encoderonly models (RoBERTa-MNLI) and autoregressive models (gpt-neo-125M, gpt-neo-1.3B, gemma2b, gpt-neo-2.7B, phi-2, gpt-j-6B, Llama-2-7b-chat-hf, Mistral-7B-v0.1, gemma-7b, and MetaLlama-3-8B) we estimate the causal reasoning capability of the learned representations. We release the framework, model code, and results via https://github.com/Exploration-Lab/COLD. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Observational  “Causal Query Triplets”<br>CrowdsourceWorkers 1. Get a cake mix 2. Mix in the extra ingredients 3. Prepare the cake pan.4. Preheat the oven 5. Put the mix in the pans 6. Put the cake batter in the oven 7. Take it out of the ovenHuman Written ESDs1. Purchase cake mix2. Preheat oven3. Grease pan4. add ingredients1. Preheat the Oven2. Prepare Dry Ingredients (floor, baking powder, salt) in a bowl3. Cream Butter and Sugar4. Add Eggs and Vanilla5. Combine wet and dryIngredients6. Bake the cake7. Frost the cake8. Decorate the cake Graph (       ) “Causal Query Triplets” Algorithm-1 for creating  Premise: “go to store and buy cake mix”Question: Choice 1: Choice 2:Premise: “turn off oven”  “go to kitchen”“Which of the following is an  come home with the ingredients” effect ?<br>5. Mix Question:  Which of the following is a  cause ?<br>6. Pour into prepared pan<br>“Baking a Cake” 7. Bake cake for required time8. Remove cake from oven9. Turn cake out onto cake plate Causal Graph Choice 1:Choice 2:  “go to store, buy ingredients” “ first heat oven”<br>10. Apply icing or glaze (        )<br>Real-World Commonsense Validating Causal Reasoning<br>Activities Knowledge Abilities<br>GPT-neo Phi<br>GPT-J Llama<br>Gemma Mistral<br>**----- End of picture text -----**<br>


Figure 3: The proposed **COLD** framework for evaluating LLMs for causal reasoning. The humanwritten Event Sequence Descriptions (ESDs) are obtained from crowdsource workers and include a telegrammic-style sequence of events when performing an activity. The Observational Graph and the Causal Graph for an activity are used to create causal query triplets (details in Algorithm 1), shown towards the right. Using counterfactual reasoning, “going to the kitchen” is possible without going to the market (if the ingredients are already available), making “come home with the ingredients.” a more plausible effect among the given choices. Similarly, in the second example, the event “going to market” has no direct relation with the event “heating the oven”. 

## **2 Background** 

**The Mini Turing Test** proposed by Pearl and Mackenzie [2018] is designed in a question-answering format to validate the understanding of causal knowledge about a simple story. The primary feature of a mini-Turing test is the enormous number of causal queries that can be framed using the underlying causal graph, which governs the occurrence of events in the story. Due to the enormous number of causal queries, passing the mini-Turing via memorization becomes combinatorially heavy, and hence, the authors argue that it can only be beaten if one has access to the underlying causal graph governing the occurrence of events (i.e., one has the ability to reason causally about the events). In this work, though, we only consider a more straightforward case of choice-based causal triplets; we realize the number of causal queries that can be created is enormous and helps validate the causal reasoning abilities coming close to the mini-Turing test. 

_**d**_ **-separation:** Establishing the independence of variables becomes non-trivial when dealing with complex interactions among multiple variables. _d_ -separation [Pearl, 1988] facilitates the determination of conditional independence between two sets of nodes **X** and **Y** in a graphical model _G_ given another set of nodes **Z** . _d_ -separation asserts that **X** and **Y** , given the set **Z** , are _d_ -separated if all paths for every node in **X** and every node in **Y** are blocked by conditioning on **Z** , denoted as **X** _G_ **[Y]** _[ |]_ **[ Z]**[.] A path _p_ is blocked by a set of nodes **Z** [Pearl et al., 2016], if and only if: 1) _p_ contains a chain of nodes **A** _→_ **B** _→_ **C** or a fork **A** _←_ **B** _→_ **C** such that the middle node **B** is in **Z** OR 2) _p_ contains a collider **A** _→_ **B** _←_ **C** such that the collision node **B** or its descendant is not in **Z** . 

**Backdoor Criterion:** A set of variables _W_ satisfies the backdoor criterion relative to _T_ and _Y_ if the following are true: 

- (A) _W_ blocks all backdoor paths from _T_ to _Y_ i.e. blocking confounding or non-causation association paths 

- (B) _W_ doesn’t contain any descendants of _T_ 

Then, _W_ satisfies the backdoor criterion [Pearl et al., 2016, Neal, 2020]. We make use of the backdoor criterion to estimate the causal estimand, capturing the relationship between the causal events. (Refer App. C for more detail) 

## **3 COLD (Causal reasOning in cLosed Daily activities)** 

We propose **COLD** ( **Causal reasOning in cLosed Daily activities** ) framework for testing causal reasoning abilities of natural language understanding systems such as LLMs. Fig. 3 gives an overview of the creation process. We use crowd-sourced data of script knowledge to create observational 

4 

graphs which is further used along with manual intervention to create causal graphs. Subsequently, an algorithm is used to create an enormous number of causal queries (causal triplets), which are further used to test LLMs for causal reasoning. Next, we explain each of the steps in more detail. 

**Task Formulation: COLD** is motivated by Causal Commonsense Reasoning (CCR), which we define as the task of finding the strength of the cause-and-effect relationship between two events ( _E_ 1 and _E_ 2) given in an activity _a ∈A_ , where _A_ is the set of all activities. For example, for an activity like “going in an airplane”, the central question is to determine the causal relationship between two events that occur during the activity (events like “checking in luggage” and “waiting for luggage”). Since reasoning about a sequence of events is tedious (and sometimes confusing [Do et al., 2011]), researchers often rely on a more plausible cause rather than defining a definite causal event. For instance, COPA dataset [Gordon et al., 2012] provides a premise event and a corresponding causal query question along with two choices (see Table 1 for an example); a system is required to predict which of the two choices is most plausible cause/effect as required by the question. 

## **Creating a Closed Causal System** 

Given the nature of Script knowledge (satisfying the criterion of balanced covariates, §1), we use a script corpus called DeScript [Wanzare et al., 2016] for creating the observational graphs. DeScript is a corpus with a telegram-style sequential description of an activity in English (e.g., baking a cake, taking a bath, etc.). DeScript is created via crowd-sourcing. For a given activity, crowd-workers write a point-wise and sequential short description of various events involved in executing the activity (this one complete description is called an ESD (Event Sequence Description)). DeScript collects data for a set of 40 daily activities (100 ESDs each) varying in complexity and background knowledge. Additionally, for a given activity, semantically similar events from different ESDs are manually aligned by human annotators (for more details, refer to Wanzare et al. [2016]). These alignments were later used by Joshi et al. [2023b,a] to create a DAG representing the overall activity. In our work, we use these DAGs as the observational distribution of an activity ( _Go_[(] _[a]_[)][, where] _[ a][ ∈A]_[, where] _[ A]_[ is the] set of all activities). These DAGs provide a medium for generating enormous trajectories (scales from 1 _._ 6 _e_ +16 to 1 _._ 3 _e_ +27, also see Table 2), that are coming directly from human annotations (alignment as well as the ESDs), providing us a proxy to represent the understanding of daily activities. 

**Observational Distribution (** _Go_ **)** : Note that the graphs _Go_ , approximately represent (almost) all possible ways in which an ESD can be written for an activity, providing the true observational distribution, i.e., how the combinations of events will look like while performing the activity in the real world (see App. A.3 for examples). 

**Causal Graphs (** _Gc_ **):** To reason about the causal relationships between the events (nodes of _Go_ ), we would need the underlying causal graph that shows the cause of occurrence of various activities (directly or indirectly). We construct the causal graphs manually by reasoning about the independence of various events in the activity. Fig. 2 shows the pictorial representation of one of the created causal graphs for the activity “going grocery shopping.” Notice that various sets of events in the graph create clusters, denoting independence between various events. For example, nodes related to make list cause the events that involve the presence of a list and do not cause events like going via car (as some of the population will not create a list for shopping). Similarly, the mode of transportation (car/bus/walk) is independent of the events performed inside the store. 

**Causal Query Triplets:** The obtained Causal Graph ( _Gc_ ), for an activity provides a medium to reason about causal links between the events. Notice in Fig. 2, how the red nodes (colliders) help separate out the independent event clusters. For example, the nodes ‘get list from car’ and ‘check list (if anything is left)’ being colliders, separate out the making list-related events with the rest of the graph. Similarly, the node ‘put bags in cart’ separates out the ‘take shop cart’ and ‘take bags’. Another interesting property represented in the obtained causal graph is the conditional dependence between various clusters. For example, the cluster related to ‘get in car’ is unconditionally independent of ‘make list’. However, if we condition on the collider (‘get list from car’), they become dependent, i.e., if ‘get list from car’ is observed, it means that the person will have created the list as well as went by car for the grocery shopping (similarly for node ‘put bags in cart’). _d_ -separation (§2) provides an easier way to establish conditional/unconditional independence between the set of nodes. For creating the dataset of causal queries (similar to other datasets like COPA [Gordon et al., 2012]), we need a triplet of three events (premise _p_ , Choice-1 _c_ 1, Choice-2 _c_ 2) associated with a question about ‘cause’ or ‘effect’ relationship, i.e., given the premise which of the two choices is the cause/effect? (Table 1). We call these triplets _Causal Query_ 

5 

Table 1: The table shows examples of causal query triplets created using the causal graphs ( _Gc_ ) and observational graphs ( _Go_ ) in Algorithm 1. The top row is taken from the COPA dataset [Gordon et al., 2012] for the purpose of comparison. Note the examples in the table show the samples taken from the instance version. 

|**Dataset**|**Dataset**|**Premise**|**Choice-1 (1)**|**Choice-2 (2)**|**question**|**answer**|
|---|---|---|---|---|---|---|
|**COPA**||The man turned on the faucet.<br>The girl found a bug in her cereal.<br>The hamburger meat browned.|The toilet filled with water.<br>She poured milk in the bowl.<br>The cook froze it.|Water flowed from the spout.<br>She lost her appetite.<br>The cook grilled it.|effect<br>effect<br>cause|2<br>2<br>2|
|||buy proper ingredients.|go home with ingredients.|wait for the timer to go off.|effect|1|
|**COLD**|(Cake)|measure ingredients in designated measuring cups.<br>bake until cake is ready.<br>turn off oven.|whisk after each addition.stir to combine.<br>set timer.<br>go to store, buy ingredients.|clean up the mess.<br>carefully remove cake from pan.<br>first heat oven.|effect<br>cause<br>cause|1<br>1<br>2|
|||preheat oven to 350 degrees.|turn off oven.|prepare the microwave oven and utensils|effect|1|
|**COLD**|(Shopping)|pay total.<br>get the bill for groceries.<br>pay for it .<br>go back to the car.put the bags in the car.<br>take the full cart to the checkout lane.|get receipt.<br>pay for the grocery.<br>start at the non-cold side of the store.<br>take your car and drive to grocery shop.<br>watch prices as the checker scans.|place cart into cart corral.<br>return cart to store.<br>go to shelf and get the food items.<br>bring items to checkout.<br>go down aisles.|effect<br>cause<br>cause<br>cause<br>effect|1<br>1<br>2<br>1<br>1|
|||check train schedules.|choose a destination.|go to the car.|cause|1|
|**COLD**|(Train)|you board your train and find your seat.<br>get off train.<br>go out of the station.<br>arrive at destination.|find your seat or compartment.<br>go out of the station.<br>put carry on luggage in overhead bin.<br>when train reaches destination, exit train.|wait for train.<br>take all the luggage out of train.<br>get off at your correct stop.<br>walk to the train platform.|effect<br>effect<br>cause<br>effect|1<br>1<br>2<br>1|
|||go to garden center.|transport it home.|choose type of tree.|effect|1|
|**COLD**|(Tree)|fill hole with dirt and fertilizer.<br>place the tree at the top of the hole.<br>fill in dirt around the tree gently.|get tree.<br>cover the roots with dirt.<br>take it home.|dig hole big enough for tree to grow.<br>get a tree.<br>place the tree in the hole.|cause<br>effect<br>cause|2<br>1<br>2|
|||place tree sapling into hole.|pack dirt back in.|find place for tree.|effect|1|
|||pull signal for stop.|bus stops at destination.|stand up and go to door.|effect|1|
|**COLD**|(Bus)|board the bus when it arrives.<br>step on bus.<br>buy bus ticket.|while boarding, pay the driver the required fee<br>take available seat.<br>when it arrives get on.|pull signal for stop.<br>wait for the bus to arrive.<br>when your stop approaches, pull cord.|effect<br>effect<br>cause|1<br>1<br>1|
|||find seat on bus and sit.|find out what bus to take.|the bus arrives at the departure station.|cause|2|



Table 2: The table provides details of the observational graph ( _Go_ ) for 5 activities. The Causal Query Triplets represent the total number of triplets generated via Algorithm 1. The instance version shows the number of samples present in the instance version (including different text instances describing the same event) of the created dataset. Table 1 shows a small sample taken for the 5 activities. Overall, the huge number of samples highlights the exhaustive nature of evaluation that can be done for LLMs. 

|**Activity**|**Nodes **|**Compact Trajectories **|**Total Trajectories Causal Query Triplets **|**Total Trajectories Causal Query Triplets **|**Instance version (Num. samples)**|
|---|---|---|---|---|---|
|Baking a **Cake**|28|177030|1_._3_e_+ 27|864|2887950|
|Riding on a **Bus**|20|13945|1_._3_e_+ 17|334|834046|
|Going Grocery **Shopping**|33|626096|3_._1_e_+ 26|1984|3739184|
|Going on a **Train**|26|133799|4_._9_e_+ 22|950|1213114|
|Planting a **Tree**|23|4466|1_._6_e_+ 16|260|846046|
|Total Dataset Samples|-|-|-|**4392**|**9,520,340**|



_Triplets_ that are used to frame a causal query between the events. App. B, Algorithm 1 presents the mechanism to create a dataset of causal query triplets. We start by constructing the set of possible triplets, and sort the nodes in every triplet using the topological order preset in the observational graph (a DAG). Further, using _d_ -separation, we figure out the triplets that have one node _d_ -separated from the other nodes. The _d_ -separated node becomes the wrong choice. The premise and correct choice are determined from the remaining choices, leading to a ‘cause’ and ‘effect’ query based on the topological order, i.e., the event occurring before (temporally) in an activity becomes ‘cause.’ The other event becomes the plausible ‘effect.’ Note that the temporal precedence is generally assumed essential for defining causation, and it is one of the most important clues that is used to distinguish causal from other types of associations [Mill, 1898, Hill, 1965, Pearl and Verma, 1995]. For our framework, we also consider the topologically sorted order obtained from the observational graphs and use the temporal order to define the causal query triplets, i.e., the cause events will always precede the effect events (see App. B, Algorithm 1, where _Go_ helps determine the temporal order of events). Table 1 shows a sample of the created dataset in comparison to the COPA dataset. Note that each node (premise, Choice-1, Choice-2) in _Gc_ also has multiple texts (instances of text describing the same event) written by different crowd-sourced workers. We can further use these text instances to enrich the created dataset by considering all the available instances to create all possible combinations. This strategy increases the scale of the created dataset by a huge margin. Overall, we found a dataset created from triplets using Algorithm 1 results in **4,392** tuples, which, after using the text instances, increases to **9,520,340** , bringing it close to ‘mini turning test” [Pearl and Mackenzie, 2018]. 

**Adherence to SUTVA:** In causal literature, a fundamentally acknowledged _Stable Unit Treatment Value Assumption_ (SUTVA) [Cox, 1958, Rubin, 1980]) requires that for each unit (e.g., sequence of events), there is only one version of the non-treatment, i.e., for an event in the sequence, there lie only 

6 

Table 3: The table provides evaluation results of language models over the created causal triplets. 

|**Triplets**|**Model Name**|**cake**|**shopping**|**train**|**tree**|**bus**|
|---|---|---|---|---|---|---|
||gpt-neo-125M|50.71|50.01|49.99|50.13|50.15|
||gpt-neo-1.3B|44.77|45.69|42.52|45.67|42.89|
|causal triplets|gemma-2b<br>gpt-neo-2.7B<br>phi-2<br>gpt-j-6B<br>Llama-2-7b-chat-hf<br>Mistral-7B-v0.1|53.76<br>50.00<br>85.14<br>49.59<br>77.92<br>77.64|52.19<br>50.01<br>83.65<br>50.02<br>72.41<br>69.38|60.57<br>50.00<br>77.29<br>50.29<br>73.48<br>68.46|60.71<br>50.01<br>82.24<br>49.92<br>72.40<br>72.43|53.64<br>50.00<br>71.74<br>49.93<br>68.21<br>69.37|
||gemma-7b|81.47|82.26|77.24|80.78|70.29|
||Meta-Llama-3-8B|80.79|76.46|76.08|78.21|67.39|



two versions occurring and not occurring. SUTVA plays a vital role in causal inference by ensuring that each unit’s treatment assignment has a consistent impact, facilitating the accurate estimation of treatment effects. Our framework closely adheres to SUTVA assumptions (details in A.1). 

**Comparison with Other Causal Datasets:** We briefly compare the created dataset with the existing set of causal reasoning datasets in App. Table 5. The created dataset serves as a middle ground, having both real-world groundings as well as an underlying causal graph to create an exhaustive set of causal queries. 

## **4 Experiments and Results** 

**COLD** provides a causal query dataset for evaluating LMs for causal understanding. In particular, we consider the “Causal Query Triplets" (Table 2) coming from compact trajectories as a base and sample the instance version coming from the same skeleton. Since it is not possible to evaluate all the possible causal queries that could be created using our framework, we use 10 _K_ samples for each activity to report our findings. For a fair comparison between various models and better reproducibility, we freeze the sampled causal query triplets and compare the success rate over the frozen samples. We evaluate via two methods. First, as done in previous work Jin et al. [2024, 2023], Chen et al. [2024], we first experiment with various LLMs using a prompt-based evaluation scheme; second, we propose other mechanisms (based on causal theory, e.g., Average Treatment Effect) that could be used to perform an in-depth analysis of evaluating causal relationships between events. 

**Causal Reasoning Evaluation of LLMs via Prompts:** We start with the prompt-based evaluation of recent open-weight LLMs (gpt-neo-125M, gpt-neo-1.3B, gpt-neo-2.7B [Black et al., 2021], gemma2b [Team et al., 2024], phi-2 [Javaheripi et al., 2023], gpt-j-6B [Wang and Komatsuzaki, 2021], gemma-7b [Team et al., 2024], Llama-2-7b-chat-hf [Touvron et al., 2023], Mistral-7B-v0.1 [Jiang et al., 2023], and Meta-Llama-3-8B [Dubey et al., 2024]) We frame the prompt as a multi-choice question-answering (MCQA) objective [Robinson and Wingate, 2023]. The prompt is intentionally structured so that the LLM is intended to predict a single choice token (Such as “ A”, “ B”, etc.). Robinson and Wingate [2023] highlight the advantages of MCQA-based evaluation over cloze evaluation [Brown et al., 2020](where the LLMs are expected to generate the entire answer in a cloze test), leading to a significant boost in various tasks, including commonsense-based tasks. App. E, Fig. 5 presents various prompt templates for autoregressive experiments, and App. E Fig. 6 shows a few qualitative examples for the framed causal query templates. Table 3 shows the success rate obtained for various LLMs. The success rate corresponds to the percentage of queries where the LLM predicts the desired choice. We observe that reasoning causally about simple daily activities is challenging when a rigorous test is framed, validating the dependencies between the events. Overall, for the more common activities like baking a cake and going grocery shopping, the LLMs perform better when compared to activities like boarding a bus or planting a tree. We also experimented with another version of the dataset, where incorrect choice may correspond to temporally plausible but causally implausible events. The results drop significantly in this case; details and results are provided in the App. F.1. 

**Evaluation using Average Treatment Effect (ATE) (** ∆ **):** Computing the Average Treatment Effect (∆) helps establish the strength of causal links given a context (Eq. 1). In our setup, to estimate 

7 

_P_ ( _y|do_ ( _x_ )) (i.e., the causal estimand) from statistical estimands (obtained from observational distribution), we make certain reasonable assumptions about the underlying process that governs the relation among variables/events and then utilize the implications of these assumptions. For any activity taking place, the causal relationships between two events _E_ 1 and _E_ 2 may have a causal link along with a non-causal link through a set of confounders _z_ . We define the confounder _z_ = _{ti|ti ∈T }_ , where _T_ denotes all the trajectories (sequence of events) from the start of the activity till the event _E_ 1. The temporal nature of events makes this assumption suitable since the occurrence of _E_ 1 and _E_ 2 can be confounded by all the events preceding _E_ 1. Note the possibility of unobserved confounders (events that are not explicitly mentioned but may be affecting the mentioned events) in our case is removed due to two reasons: 1) Keeping a closed system representation with a large number of diverse scripts (written by humans) helps cover the set of most generic and diverse events either implicitly or explicitly as a part of the activity, and 2) The causal reasoning goal is restricted to figuring out the causal effect between the events that are present explicitly. Assuming the unmentioned events have insignificant effects, we can establish that there will not be any unobserved confounders. This assumption makes the observed confounders satisfy the backdoor criterion [Pearl, 1993] that make sufficient adjustment sets. By using the backdoor criterion (App. C), the interventional distributions are estimated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0008-01.png)


Note that the true observational distribution, i.e. _p∗_ ( _E_ 2 _|E_ 1 _, z_ ) and _p∗_ ( _z_ ) both are unknown and have to be approximated ( _p_ ˆ( _E_ 2 _|E_ 1 _, z_ ) and _p_ ˆ( _z_ )). Further, we describe ways _p_ ˆ can be estimated via multiple design mechanisms. Due to space limitation, we only describe the _p_ ˆ estimation via language models below and move the statistical analysis using the original trajectories and observational graphs to the App. D.2. 

**ATE using Language Models** Since pre-trained LMs capture world knowledge [Devlin et al., 2019, Brown et al., 2020, Li et al., 2023, Nanda et al., 2023, Karvonen, 2024], these provide a suitable proxy for establishing relationships between these events. For our experiments, we consider a simple reasoning capability of language models, i.e., to reason about the temporal order of various events, i.e., given an event, what is the likelihood of the occurrence of another event? We further ask if this can be used to estimate the causal relationship between the events (a similar strategy is used by Zhang et al. [2022b] for zero-shot causal estimation). It is worth noting that for these activities about daily activities, one way to find causes is to establish the temporal likelihood of the events. For each of the multiple language models, we frame the temporal prediction differently. 

**Encoder-only Models:** For BERT-based models trained for mask token prediction, we model the temporal prediction using the probability assigned to the mask tokens “ _before_ ” and “ _after_ ” [Zhang et al., 2022b]. Given two events _E_ 1 and _E_ 2, the temporal link is predicted using a prompt like _E_ 1 <mask> _E_ 2, and the scores corresponding to the before and after tokens are collected. App. D, Fig. 7, the top row highlights the prompt template used for BERT-based models. For encoder-only experiments, we consider RoBERTa MNLI [Liu et al., 2019]. 

**Decoder-only Models:** For other language models that are autoregressive in nature, we modify the prompt to predict the temporal order as the last token. We again use the MCQA-based prompting style to frame the temporal order query by providing “ _before_ " and “ _after_ " as the options in the prompt. App. D, Fig. 7, the bottom row highlights the prompt template used for Decoder-only Models. 

**Interventions:** We utilize the SUTVA assumption in the proposed framework to devise an intervention over a trajectory in natural language form. App. D, Fig. 8 shows the style of intervention made by an event ( _E_ 1) taking place ( _do_ ( _E_ 1)) or not taking place _do_ ( _¬E_ 1). 

ˆ Given the above strategies, LMs can be used to evaluate _p_ ( _E_ 2 _|¬E_ 1 _, z_ = _t_ ), by feeding the prompt that contains _E_ 1, _E_ 2 and the _z_ = _t_ ) and predict the temporal nature between _E_ 1 and _E_ 2, given a trajectory _z_ = _t_ . Further, applying the backdoor criterion for multiple trajectories _T_ , we obtain 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0008-08.png)


8 

Table 4: Accuracy over the causal triplets for various ∆ estimates. The blue text denotes the improvements by backdoor adjustments over the temporal scheme for multiple language models. **Bold text** represents the best-performing method for a particular activity. 

|ˆ_p_**estimation**|**Scheme **|**ATE**|**Cake **|**Shopping **|**Train **|**Tree**|**Bus**|
|---|---|---|---|---|---|---|---|
|Original Trajectories||∆_o_|28.20|34.30|31.10|30.10|30.40|
|Observational Graphs|-<br>-|∆_n_<br>∆_t_|30.40<br>40.90|30.10<br>47.10|29.80 <br>40.30|28.60<br> 37.60|25.4<br> 40.10|
|||RoBERTa MNLI|46.80|54.00|45.50|52.70|43.00|
|||gpt-neo-125M|47.70|55.50|55.50|53.60|48.20|
|||gpt-neo-1.3B|47.40|45.40|53.30|43.40|52.90|
|Language Models|Temporal|gemma-2b<br>gpt-neo-2.7B<br>phi-2<br>gpt-j-6B<br>Llama-2-7b-chat-hf|43.80<br>50.10<br>60.30<br>49.50<br>38.90|41.70<br>48.90<br>59.20<br>46.40<br>42.10|52.20 <br>52.40 <br>56.90 <br>56.00 <br>51.00|49.70 <br> 47.60 <br> 70.30 <br> 62.70 <br> 40.70|49.80<br> 53.70<br> 49.40<br> 56.00<br> 47.80|
|||Mistral-7B-v0.1|50.90|54.40|64.50|60.50|62.30|
|||gemma-7b|46.8|54.00|45.50|52.70|43.00|
|||Meta-Llama-3-8B|58.20|54.10|55.6|55.00|64.00|
|||∆_M_ (RoBERTa MNLI)|59.20|54.40|56.30|57.50|53.30|
|Language Models|Backdoor Adjustments|∆_M_ (gpt-neo-125M)<br>∆_M_ (gpt-neo-1.3B)<br>∆_M_ (gemma-2b)<br>∆_M_ (gpt-neo-2.7B)<br>∆_M_ (phi-2)<br>∆_M_ (gpt-j-6B)<br>∆_M_ (Llama-2-7b-chat-hf) <br>∆_M_ (Mistral-7B-v0.1)<br>∆_M_ (gemma-7b)|59.20<br>51.30<br>44.50<br>49.10<br>57.00<br>51.30<br> 62.60<br>63.90<br>**72.80**|55.10<br>50.70<br>45.30<br>51.30<br>66.00<br>45.60<br>64.60<br>71.40<br>**77.80**|50.50 <br>55.00 <br>52.60 <br>51.50 <br>62.10 <br>50.50 <br>68.50 <br>**73.70** <br>73.60|52.10 <br> 43.90 <br> 63.50 <br> 54.00 <br> 57.10 <br> 49.10 <br> 70.50 <br> 61.30 <br> **71.90**|45.50<br> 49.00<br> 43.90<br> 51.40<br> 45.80<br> 46.00<br> 63.80<br> **67.00**<br> 62.40|
|||∆_M_ (Meta-Llama-3-8B)|66.00|70.20|68.40|62.00|63.40|



which can further be used to estimate the causal strength between the events _E_ 1 and _E_ 2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0009-03.png)


Using the multiple ∆ estimates defined above, we estimate the causal strength between the events available for an activity. We follow the scheme presented in the App. B, Algorithm 2 to compute the performance in terms of success rates. 

**Temporal Scheme:** In this scheme, we validate if temporal ordering knowledge of LLMs could be directly used to estimate the causal estimand. We make use of templates shown in the App. Fig. 7. The causal estimand is estimated via the difference in logit values when intervening over an event, i.e., does the predicted probability take into account the context of events not happening? Surprisingly, we found that temporal ordering does provide a suitable proxy for estimating causal strength between the events. We further extend this approximation to incorporate the backdoor adjustments in the ∆. 

**Backdoor Adjustments:** For the experiments with Language models, we apply the backdoor adjustment to estimate the causal estimand ∆ _M_ . App. Fig. 8 shows the prompt template used to determine the relationship between the events. The prompt template takes a trajectory, _ti_ , that contains all the events till the event _E_ 1 in sequential order of occurrence, further, an added prompt determines the intervention ( _do_ ( _E_ 1) or _do_ ( _¬E_ 1)) and the causal estimand is estimated using the logit values associated with the predicted token. App. B, Algorithm 3 provides the designed scheme to compute unbiased causal estimands. We essentially flip the options and generate the scores associated with options ‘A’ and ‘B’ for increase and decrease, respectively (more details in the App. C.) 

Table 4 shows a comparison between various design choices. We observe that when using LLMs for _p_ ˆ estimation, the backdoor adjustment increases the performance over the temporal estimation scheme by a significant margin. The understanding of these activities is generic, and LLMs do provide a suitable set of sequences when prompted to generate a list of steps to complete the activity. For example, when prompted with ‘Generate the sequential steps in a telegrammic style to perform the activity “going grocery shopping"’, almost all the models we tested provide 

9 

a valid set of steps for the given activity. However, when prompted with causal queries, the lower performance signifies the lack of understanding of the underlying causal mechanism. The constructed dataset helps to rigorously validate the understanding of the activity through an enormous number of causal query triplets. The results show that although the LLMs can explain the activity in detail, including generating correct steps for performing tasks, causally reasoning about the set of events remains challenging. 

**Human Study:** We conducted a small-scale human validation study over the created causal query dataset and asked 5 graduate students to answer 100 randomly sampled causal query triplets (20 per activity). We record an average performance of 92.20%. (More details about the human study are provided in the App. B) 

## **5 Related Work** 

Causal reasoning has been an active research area in the ML community [Spirtes et al., 2000a, Peters et al., 2017, Schölkopf et al., 2021]. Some of the initial works highlight the causal nature of events present in text [Schank, 1975] as _‘causal chains’_ . Multiple works have considered creating benchmarks/datasets that capture causal relationships between the events described in the text (see App. Table 5). More recently, with the rapid growth of LLMs on reasoning/understanding tasks, attention has shifted to validating these general-purpose models capturing causal reasoning [Jin et al., 2023, Zeˇcevi´c et al., 2023, Willig et al., 2023a, Liu et al., 2023, Willig et al., 2023b, Zhang et al., 2022a, Jin et al., 2024]. App. A.2 Table 5 shows a broad overview of the existing causal Dataset/Benchmarks presented in the NLP community. In this work, the primary focus is to bridge the gap between various lines of work that consider natural language to learn/validate/reason about causal relationships between events. 

## **6 Limitations and Future Directions** 

One of the primary limitations of our work is the limited set of activities. Though the frameworks support generating exhaustive/enormous causal queries, finding general commonsense reasoning activities/tasks that are well understood by humans remains challenging. Moreover, creating a causal graph for an activity increases as we move toward more long-term tasks. However, as a general test of causal intelligence, our framework provides a suitable platform to validate the reasoning capabilities more rigorously. In the future, it would be interesting to sample trajectories from the observational distribution _Go[a]_[to create a training dataset and check if causal reasoning ability can] be acquired by language modeling objectives (including other variants like presented in Lampinen et al. [2023]). We leave this detailed analysis for future endeavors. The proposed algorithm for causal triplet generation generates the simplest variant of causal queries in the form of causal triplets (also referred to as Pairwise Causal Discovery (PCD) task by [Chen et al., 2024]). More complicated causal queries can be generated, such as considering cases with common confounders, long/short causal chain dependency, etc. Moreover, taking formal definitions. (i.e., using the formal causal inference language) causal queries inspired from Jin et al. [2023, 2024] can be framed for a more rigorous analysis. Being at the initial state, we stick to the simple causal queries that provide two choices, and the task is to choose the more plausible cause. The creation of underlying causal graphs provides endless possibilities for creating varied versions of causal queries. In this work, we only consider an unconditional version of _d_ -separation. In the future, the same causal graphs could be used to define more datasets for covering other rungs of the _‘causal ladder’_ [Pearl and Mackenzie, 2018]. 

## **7 Conclusion** 

In this paper, we proposed the **COLD** ( **Causal reasOning in cLosed Daily activities** ) framework for generating causal queries that can be used to rigorously evaluate LLMs. We performed extensive experimentation with LLMs for the task of Causal Commonsense Reasoning. Results indicate that LLMs are still far from a complete understanding of daily commonsensical activities and fail to answer causal queries when analyzed in an exhaustive manner. We believe this framework will provide a good platform for future research in understanding the causal reasoning abilities of LLMs. 

10 

## **Acknowledgments** 

We would like to thank the anonymous reviewers and the meta-reviewer for their insightful comments and suggestions. We would like to thank Google Deepmind India for helping us with the conference travel support. 

## **References** 

- Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Wen tau Yih, and Yejin Choi. Abductive commonsense reasoning. In _International Conference on Learning Representations_ , 2020. URL https://openreview. net/forum?id=Byg1v1HKDB. [Cited on page 24.] 

- Sid Black, Gao Leo, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow, March 2021. URL https://doi. org/10.5281/zenodo.5297715. [Cited on page 7.] 

- Alexander Bondarenko, Magdalena Wolska, Stefan Heindorf, Lukas Blübaum, Axel-Cyrille Ngonga Ngomo, Benno Stein, Pavel Braslavski, Matthias Hagen, and Martin Potthast. CausalQA: A benchmark for causal question answering. In Nicoletta Calzolari, Chu-Ren Huang, Hansaem Kim, James Pustejovsky, Leo Wanner, Key-Sun Choi, Pum-Mo Ryu, Hsin-Hsi Chen, Lucia Donatelli, Heng Ji, Sadao Kurohashi, Patrizia Paggio, Nianwen Xue, Seokhwan Kim, Younggyun Hahm, Zhong He, Tony Kyungil Lee, Enrico Santus, Francis Bond, and Seung-Hoon Na, editors, _Proceedings of the 29th International Conference on Computational Linguistics_ , pages 3296– 3308, Gyeongju, Republic of Korea, October 2022. International Committee on Computational Linguistics. URL https://aclanthology.org/2022.coling-1.291. [Cited on page 24.] 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. _CoRR_ , abs/2005.14165, 2020. URL https://arxiv.org/abs/2005.14165. [Cited on pages 1, 7, and 8.] 

- Sirui Chen, Bo Peng, Meiqi Chen, Ruiqi Wang, Mengying Xu, Xingyu Zeng, Rui Zhao, Shengjie Zhao, Yu Qiao, and Chaochao Lu. Causal evaluation of language models, 2024. [Cited on pages 7 and 10.] 

- Changwoo Chun, SongEun Lee, Jaehyung Seo, and Heuiseok Lim. CReTIHC: Designing causal reasoning tasks about temporal interventions and hallucinated confoundings. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pages 10334–10343, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.693. URL https://aclanthology.org/ 2023.findings-emnlp.693. [Cited on page 2.] 

- Kai-Hendrik Cohrs, Emiliano Diaz, Vasileios Sitokonstantinou, Gherardo Varando, and Gustau Camps-Valls. Large language models for constrained-based causal discovery. In _AAAI 2024 Workshop on ”Are Large Language Models Simply Causal Parrots?”_ , 2023. URL https:// openreview.net/forum?id=NEAoZRWHPN. [Cited on page 1.] 

- D. R. Cox. _Planning of Experiments_ . Wiley, New York„ 1958. [Cited on pages 3, 6, and 23.] 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/ N19-1423. [Cited on pages 1 and 8.] 

11 

- Quang Do, Yee Seng Chan, and Dan Roth. Minimally supervised event causality identification. In Regina Barzilay and Mark Johnson, editors, _Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing_ , pages 294–303, Edinburgh, Scotland, UK., July 2011. Association for Computational Linguistics. URL https://aclanthology.org/D11-1027. [Cited on pages 5, 24, 25, and 30.] 

- Li Du, Xiao Ding, Kai Xiong, Ting Liu, and Bing Qin. e-CARE: a new dataset for exploring explainable causal reasoning. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 432–446, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.33. URL https: //aclanthology.org/2022.acl-long.33. [Cited on pages 2, 23, and 24.] 

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon 

12 

Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzmán, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, IrinaElena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vítor Albiero, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783. [Cited on page 7.] 

- Jesse Dunietz, Lori Levin, and Jaime Carbonell. The BECauSE corpus 2.0: Annotating causality and overlapping relations. In Nathan Schneider and Nianwen Xue, editors, _Proceedings of the 11th Linguistic Annotation Workshop_ , pages 95–104, Valencia, Spain, April 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-0812. URL https://aclanthology.org/ W17-0812. [Cited on page 24.] 

- Jörg Frohberg and Frank Binder. CRASS: A novel data set and benchmark to test counterfactual reasoning of large language models. In Nicoletta Calzolari, Frédéric Béchet, Philippe Blache, Khalid Choukri, Christopher Cieri, Thierry Declerck, Sara Goggi, Hitoshi Isahara, Bente Maegaard, Joseph Mariani, Hélène Mazo, Jan Odijk, and Stelios Piperidis, editors, _Proceedings of the Thirteenth Language Resources and Evaluation Conference_ , pages 2126–2140, Marseille, France, June 2022. European Language Resources Association. URL https://aclanthology.org/2022. lrec-1.229. [Cited on page 24.] 

- Andrew Gordon, Zornitsa Kozareva, and Melissa Roemmele. SemEval-2012 task 7: Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In Eneko Agirre, Johan Bos, Mona Diab, Suresh Manandhar, Yuval Marton, and Deniz Yuret, editors, _*SEM 2012: The_ 

13 

_First Joint Conference on Lexical and Computational Semantics – Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012)_ , pages 394–398, Montréal, Canada, 7-8 June 2012. Association for Computational Linguistics. URL https://aclanthology.org/S12-1052. [Cited on pages 1, 2, 5, 6, 23, and 24.] 

- Iris Hendrickx, Su Nam Kim, Zornitsa Kozareva, Preslav Nakov, Diarmuid Ó Séaghdha, Sebastian Padó, Marco Pennacchiotti, Lorenza Romano, and Stan Szpakowicz. SemEval-2010 task 8: Multi-way classification of semantic relations between pairs of nominals. In Katrin Erk and Carlo Strapparava, editors, _Proceedings of the 5th International Workshop on Semantic Evaluation_ , pages 33–38, Uppsala, Sweden, July 2010. Association for Computational Linguistics. URL https://aclanthology.org/S10-1006. [Cited on page 24.] 

- Austin Bradford Hill. The environment and disease: association or causation?, 1965. [Cited on pages 6 and 24.] 

- Matthew Ho, Aditya Sharma, Justin Chang, Michael Saxon, Sharon Levy, Yujie Lu, and William Yang Wang. Wikiwhy: Answering and explaining cause-and-effect questions, 2022. [Cited on page 1.] 

- Mojan Javaheripi, Sébastien Bubeck, Marah Abdin, Jyoti Aneja, Caio César Teodoro Mendes, Weizhu Chen, Allie Del Giorno, Ronen Eldan, Sivakanth Gopi, Suriya Gunasekar, Piero Kauffmann, Yin Tat Lee, Yuanzhi Li, Anh Nguyen, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Michael Santacroce, Harkirat Singh Behl, Adam Taumann Kalai, Xin Wang, Rachel Ward, Philipp Witte, Cyril Zhang, and Yi Zhang. Phi-2: The surprising power of small language models. _Microsoft Research Blog_ , 2023. [Cited on page 7.] 

- Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. URL https://arxiv.org/ abs/2310.06825. [Cited on page 7.] 

- Zhijing Jin, Jiarui Liu, Zhiheng Lyu, Spencer Poff, Mrinmaya Sachan, Rada Mihalcea, Mona Diab, and Bernhard Schölkopf. Can large language models infer causation from correlation?, 2023. [Cited on pages 1, 7, 10, and 24.] 

- Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojasv Kamal, Zhiheng Lyu, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, and Bernhard Schölkopf. Cladder: Assessing causal reasoning in language models, 2024. [Cited on pages 1, 7, 10, and 24.] 

- Abhinav Joshi, Areeb Ahmad, Umang Pandey, and Ashutosh Modi. From scripts to rl environments: Towards imparting commonsense knowledge to rl agents. In _Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems_ , pages 2801–2803, 2023a. [Cited on page 5.] 

- Abhinav Joshi, Areeb Ahmad, Umang Pandey, and Ashutosh Modi. Scriptworld: Text based environment for learning procedural knowledge. In Edith Elkind, editor, _Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23_ , pages 5095– 5103. International Joint Conferences on Artificial Intelligence Organization, 8 2023b. doi: 10.24963/ijcai.2023/566. URL https://doi.org/10.24963/ijcai.2023/566. Main Track. [Cited on page 5.] 

- Adam Karvonen. Emergent world models and latent variable estimation in chess-playing language models, 2024. [Cited on page 8.] 

- Benjamin Kuipers. Commonsense reasoning about causality: Deriving behavior from structure. _Artificial Intelligence_ , 24(1):169–203, 1984. ISSN 0004-3702. doi: https://doi.org/ 10.1016/0004-3702(84)90039-0. URL https://www.sciencedirect.com/science/article/ pii/0004370284900390. [Cited on page 2.] 

14 

- Yash Kumar Lal, Nathanael Chambers, Raymond Mooney, and Niranjan Balasubramanian. TellMeWhy: A dataset for answering why-questions in narratives. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pages 596–610, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl.53. URL https://aclanthology.org/2021. findings-acl.53. [Cited on page 24.] 

- Andrew Kyle Lampinen, Stephanie C.Y. Chan, Ishita Dasgupta, Andrew Joo Hun Nam, and Jane X Wang. Passive learning of active causal strategies in agents and language models. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https://openreview.net/ forum?id=BRpi8YAfac. [Cited on page 10.] 

- Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id=DeG07_TcZvT. [Cited on page 8.] 

- Xiao Liu, Da Yin, Chen Zhang, Yansong Feng, and Dongyan Zhao. The magic of IF: Investigating causal reasoning abilities in large language models of code. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 9009–9022, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10. 18653/v1/2023.findings-acl.574. URL https://aclanthology.org/2023.findings-acl.574. [Cited on page 10.] 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ , 2019. [Cited on page 8.] 

- John Stuart Mill. _A System of Logic, Ratiocinative and Inductive: Being a Connected View of the Princilples of Evidence and the Methods of Scientific Investigation_ . Longmans, Green, 1898. [Cited on pages 6 and 24.] 

- Paramita Mirza, Rachele Sprugnoli, Sara Tonelli, and Manuela Speranza. Annotating causality in the TempEval-3 corpus. In Oleksandr Kolomiyets, Marie-Francine Moens, Martha Palmer, James Pustejovsky, and Steven Bethard, editors, _Proceedings of the EACL 2014 Workshop on Computational Approaches to Causality in Language (CAtoCL)_ , pages 10–19, Gothenburg, Sweden, April 2014. Association for Computational Linguistics. doi: 10.3115/v1/W14-0702. URL https: //aclanthology.org/W14-0702. [Cited on page 24.] 

- Melanie Mitchell, Alessandro B. Palmarini, and Arseny Moskvichev. Comparing humans, gpt-4, and gpt-4v on abstraction and reasoning tasks, 2023. [Cited on page 1.] 

- Ashutosh Modi. Event Embeddings for Semantic Script Modeling. In _Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning_ , 2016. doi: 10.18653/v1/K16-1008. URL https://aclanthology.org/K16-1008. [Cited on page 2.] 

- Ashutosh Modi. _Modeling Common Sense Knowledge via Scripts_ . PhD thesis, Saarland University, 2017. [Cited on page 2.] 

- Ashutosh Modi and Ivan Titov. Inducing neural models of script knowledge. In _Proceedings of the Eighteenth Conference on Computational Natural Language Learning_ , pages 49–57, 2014. [Cited on page 2.] 

- Ashutosh Modi, Tatjana Anikina, Simon Ostermann, and Manfred Pinkal. InScript: Narrative texts annotated with script information. In _Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)_ , 2016. URL https://aclanthology.org/ L16-1555. [Cited on page 2.] 

- Ashutosh Modi, Ivan Titov, Vera Demberg, Asad Sayeed, and Manfred Pinkal. Modeling Semantic Expectation: Using Script Knowledge for Referent Prediction. _Transactions of the Association for Computational Linguistics_ , 2017. doi: 10.1162/tacl_a_00044. URL https://aclanthology. org/Q17-1003. [Cited on page 2.] 

15 

- Nasrin Mostafazadeh, Alyson Grealish, Nathanael Chambers, James Allen, and Lucy Vanderwende. CaTeRS: Causal and temporal relation scheme for semantic annotation of event structures. In Martha Palmer, Ed Hovy, Teruko Mitamura, and Tim O’Gorman, editors, _Proceedings of the Fourth Workshop on Events_ , pages 51–61, San Diego, California, June 2016. Association for Computational Linguistics. doi: 10.18653/v1/W16-1007. URL https://aclanthology.org/ W16-1007. [Cited on page 24.] 

- Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models. In Yonatan Belinkov, Sophie Hao, Jaap Jumelet, Najoung Kim, Arya McCarthy, and Hosein Mohebbi, editors, _Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP_ , pages 16–30, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.blackboxnlp-1.2. URL https: //aclanthology.org/2023.blackboxnlp-1.2. [Cited on page 8.] 

- Brady Neal. Introduction to causal inference. _Course Lecture Notes (draft)_ , 2020. [Cited on pages 4 and 27.] 

- Simon Ostermann, Ashutosh Modi, Michael Roth, Stefan Thater, and Manfred Pinkal. MCScript: A Novel Dataset for Assessing Machine Comprehension Using Script Knowledge. In _LREC_ , 2018. [Cited on page 2.] 

- Judea Pearl. _Probabilistic reasoning in intelligent systems: networks of plausible inference_ . Morgan kaufmann, 1988. [Cited on page 4.] 

- Judea Pearl. Comment: Graphical models, causality and intervention. 8(3):266–269, August 1993. doi: 10.1214/ss/1177010894. [Cited on page 8.] 

- Judea Pearl. The do-calculus revisited. In _Proceedings of the Twenty-Eighth Conference on Uncertainty in Artificial Intelligence_ , UAI’12, page 3–11, Arlington, Virginia, USA, 2012. AUAI Press. ISBN 9780974903989. [Cited on page 2.] 

- Judea Pearl and Dana Mackenzie. _The Book of Why: The New Science of Cause and Effect_ . Basic Books, Inc., USA, 1st edition, 2018. ISBN 046509760X. [Cited on pages 1, 3, 4, 6, 10, and 25.] 

- Judea Pearl and Thomas S Verma. A theory of inferred causation. In _Studies in Logic and the Foundations of Mathematics_ , volume 134, pages 789–811. Elsevier, 1995. [Cited on pages 6 and 24.] 

- Judea Pearl, Madelyn Glymour, and Nicholas P Jewell. _Causal inference in statistics: A primer_ . John Wiley & Sons, 2016. [Cited on pages 4, 25, and 27.] 

- Derek Penn and Daniel Povinelli. Causal cognition in human and nonhuman animals: A comparative, critical review. _Annual Review of Psychology_ , 58:97–118, 02 2007. doi: 10.1146/annurev.psych. 58.110405.085555. [Cited on page 1.] 

- Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. _Elements of Causal Inference: Foundations and Learning Algorithms_ . Adaptive Computation and Machine Learning. MIT Press, Cambridge, MA, 2017. ISBN 978-0-262-03731-0. URL https://mitpress.mit.edu/books/ elements-causal-inference. [Cited on page 10.] 

- Lianhui Qin, Antoine Bosselut, Ari Holtzman, Chandra Bhagavatula, Elizabeth Clark, and Yejin Choi. Counterfactual story reasoning and generation. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP)_ , pages 5043–5053, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1509. URL https://aclanthology.org/D19-1509. [Cited on page 24.] 

- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. URL https://api.semanticscholar.org/ CorpusID:160025533. [Cited on page 1.] 

16 

- Hannah Rashkin, Maarten Sap, Emily Allaway, Noah A. Smith, and Yejin Choi. Event2Mind: Commonsense inference on events, intents, and reactions. In Iryna Gurevych and Yusuke Miyao, editors, _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 463–473, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1043. URL https://aclanthology.org/ P18-1043. [Cited on page 24.] 

- Joshua Robinson and David Wingate. Leveraging large language models for multiple choice question answering. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id=yKbprarjc5B. [Cited on pages 7 and 29.] 

- Angelika Romanou, Syrielle Montariol, Debjit Paul, Leo Laugier, Karl Aberer, and Antoine Bosselut. CRAB: Assessing the strength of causal relationships between real-world events. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 15198–15216, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.940. URL https: //aclanthology.org/2023.emnlp-main.940. [Cited on pages 1 and 24.] 

- Donald B. Rubin. Randomization analysis of experimental data: The fisher randomization test comment. _Journal of the American Statistical Association_ , 75(371):591–593, 1980. [Cited on pages 3, 6, and 23.] 

- Maarten Sap, Ronan LeBras, Emily Allaway, Chandra Bhagavatula, Nicholas Lourie, Hannah Rashkin, Brendan Roof, Noah A. Smith, and Yejin Choi. Atomic: An atlas of machine commonsense for if-then reasoning, 2019a. [Cited on page 24.] 

- Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa: Commonsense reasoning about social interactions. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP)_ , pages 4463–4473, Hong Kong, China, November 2019b. Association for Computational Linguistics. doi: 10.18653/v1/D19-1454. URL https://aclanthology.org/D19-1454. [Cited on page 24.] 

- Roger C. Schank. The structure of episodes in memory. In DANIEL G. BOBROW and ALLAN COLLINS, editors, _Representation and Understanding_ , pages 237–272. Morgan Kaufmann, San Diego, 1975. ISBN 978-0-12-108550-6. doi: https://doi.org/10.1016/ B978-0-12-108550-6.50014-8. URL https://www.sciencedirect.com/science/article/ pii/B9780121085506500148. [Cited on pages 2 and 10.] 

- Roger C. Schank and Robert P. Abelson. Scripts, Plans, and Knowledge. In _Proceedings of the 4th International Joint Conference on Artificial Intelligence_ , IJCAI, 1975. [Cited on page 2.] 

- Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. _Proceedings of the IEEE_ , 109(5):612–634, 2021. doi: 10.1109/JPROC.2021.3058954. [Cited on page 10.] 

- Shikhar Singh, Nuan Wen, Yu Hou, Pegah Alipoormolabashi, Te-lin Wu, Xuezhe Ma, and Nanyun Peng. COM2SENSE: A commonsense reasoning benchmark with complementary sentences. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pages 883–898, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl.78. URL https://aclanthology.org/2021.findings-acl.78. [Cited on page 24.] 

- P. Spirtes, C. Glymour, and R. Scheines. _Causation, Prediction, and Search_ . MIT press, 2nd edition, 2000a. [Cited on page 10.] 

- Peter Spirtes, Clark N Glymour, and Richard Scheines. _Causation, prediction, and search_ . MIT press, 2000b. [Cited on page 28.] 

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. 

17 

Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ambrose Slone, Ameet Rahane, Anantharaman S. Iyer, Anders Johan Andreassen, Andrea Madotto, Andrea Santilli, Andreas Stuhlmüller, Andrew M. Dai, Andrew La, Andrew Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karaka¸s, B. Ryan Roberts, Bao Sheng Loe, Barret Zoph, Bartłomiej Bojanowski, Batuhan Özyurt, Behnam Hedayatnia, Behnam Neyshabur, Benjamin Inden, Benno Stein, Berk Ekmekci, Bill Yuchen Lin, Blake Howald, Bryan Orinion, Cameron Diao, Cameron Dour, Catherine Stinson, Cedrick Argueta, Cesar Ferri, Chandan Singh, Charles Rathkopf, Chenlin Meng, Chitta Baral, Chiyu Wu, Chris Callison-Burch, Christopher Waites, Christian Voigt, Christopher D Manning, Christopher Potts, Cindy Ramirez, Clara E. Rivera, Clemencia Siro, Colin Raffel, Courtney Ashcraft, Cristina Garbacea, Damien Sileo, Dan Garrette, Dan Hendrycks, Dan Kilman, Dan Roth, C. Daniel Freeman, Daniel Khashabi, Daniel Levy, Daniel Moseguí González, Danielle Perszyk, Danny Hernandez, Danqi Chen, Daphne Ippolito, Dar Gilboa, David Dohan, David Drakard, David Jurgens, Debajyoti Datta, Deep Ganguli, Denis Emelin, Denis Kleyko, Deniz Yuret, Derek Chen, Derek Tam, Dieuwke Hupkes, Diganta Misra, Dilyar Buzan, Dimitri Coelho Mollo, Diyi Yang, Dong-Ho Lee, Dylan Schrader, Ekaterina Shutova, Ekin Dogus Cubuk, Elad Segal, Eleanor Hagerman, Elizabeth Barnes, Elizabeth Donoway, Ellie Pavlick, Emanuele Rodolà, Emma Lam, Eric Chu, Eric Tang, Erkut Erdem, Ernie Chang, Ethan A Chi, Ethan Dyer, Ethan Jerzak, Ethan Kim, Eunice Engefu Manyasi, Evgenii Zheltonozhskii, Fanyue Xia, Fatemeh Siar, Fernando Martínez-Plumed, Francesca Happé, Francois Chollet, Frieda Rong, Gaurav Mishra, Genta Indra Winata, Gerard de Melo, Germán Kruszewski, Giambattista Parascandolo, Giorgio Mariani, Gloria Xinyue Wang, Gonzalo Jaimovitch-Lopez, Gregor Betz, Guy Gur-Ari, Hana Galijasevic, Hannah Kim, Hannah Rashkin, Hannaneh Hajishirzi, Harsh Mehta, Hayden Bogar, Henry Francis Anthony Shevlin, Hinrich Schuetze, Hiromu Yakura, Hongming Zhang, Hugh Mee Wong, Ian Ng, Isaac Noble, Jaap Jumelet, Jack Geissinger, Jackson Kernion, Jacob Hilton, Jaehoon Lee, Jaime Fernández Fisac, James B Simon, James Koppel, James Zheng, James Zou, Jan Kocon, Jana Thompson, Janelle Wingfield, Jared Kaplan, Jarema Radom, Jascha Sohl-Dickstein, Jason Phang, Jason Wei, Jason Yosinski, Jekaterina Novikova, Jelle Bosscher, Jennifer Marsh, Jeremy Kim, Jeroen Taal, Jesse Engel, Jesujoba Alabi, Jiacheng Xu, Jiaming Song, Jillian Tang, Joan Waweru, John Burden, John Miller, John U. Balis, Jonathan Batchelder, Jonathan Berant, Jörg Frohberg, Jos Rozen, Jose Hernandez-Orallo, Joseph Boudeman, Joseph Guerr, Joseph Jones, Joshua B. Tenenbaum, Joshua S. Rule, Joyce Chua, Kamil Kanclerz, Karen Livescu, Karl Krauth, Karthik Gopalakrishnan, Katerina Ignatyeva, Katja Markert, Kaustubh Dhole, Kevin Gimpel, Kevin Omondi, Kory Wallace Mathewson, Kristen Chiafullo, Ksenia Shkaruta, Kumar Shridhar, Kyle McDonell, Kyle Richardson, Laria Reynolds, Leo Gao, Li Zhang, Liam Dugan, Lianhui Qin, Lidia Contreras-Ochando, Louis-Philippe Morency, Luca Moschella, Lucas Lam, Lucy Noble, Ludwig Schmidt, Luheng He, Luis Oliveros-Colón, Luke Metz, Lütfi Kerem Senel, Maarten Bosma, Maarten Sap, Maartje Ter Hoeve, Maheen Farooqi, Manaal Faruqui, Mantas Mazeika, Marco Baturan, Marco Marelli, Marco Maru, Maria Jose Ramirez-Quintana, Marie Tolkiehn, Mario Giulianelli, Martha Lewis, Martin Potthast, Matthew L Leavitt, Matthias Hagen, Mátyás Schubert, Medina Orduna Baitemirova, Melody Arnaud, Melvin McElrath, Michael Andrew Yee, Michael Cohen, Michael Gu, Michael Ivanitskiy, Michael Starritt, Michael Strube, Michał Sw˛edrowski, Michele Bevilacqua, Michihiro Yasunaga, Mihir Kale, Mike Cain, Mimee Xu, Mirac Suzgun, Mitch Walker, Mo Tiwari, Mohit Bansal, Moin Aminnaseri, Mor Geva, Mozhdeh Gheini, Mukund Varma T, Nanyun Peng, Nathan Andrew Chi, Nayeon Lee, Neta Gur-Ari Krakover, Nicholas Cameron, Nicholas Roberts, Nick Doiron, Nicole Martinez, Nikita Nangia, Niklas Deckers, Niklas Muennighoff, Nitish Shirish Keskar, Niveditha S. Iyer, Noah Constant, Noah Fiedel, Nuan Wen, Oliver Zhang, Omar Agha, Omar Elbaghdadi, Omer Levy, Owain Evans, Pablo Antonio Moreno Casares, Parth Doshi, Pascale Fung, Paul Pu Liang, Paul Vicol, Pegah Alipoormolabashi, Peiyuan Liao, Percy Liang, Peter W Chang, Peter Eckersley, Phu Mon Htut, Pinyu Hwang, Piotr Miłkowski, Piyush Patil, Pouya Pezeshkpour, Priti Oli, Qiaozhu Mei, Qing Lyu, Qinlang Chen, Rabin Banjade, Rachel Etta Rudolph, Raefer Gabriel, Rahel Habacker, Ramon Risco, Raphaël Millière, Rhythm Garg, Richard Barnes, Rif A. Saurous, Riku Arakawa, Robbe Raymaekers, Robert Frank, Rohan Sikand, Roman Novak, Roman Sitelew, Ronan Le Bras, Rosanne Liu, Rowan Jacobs, Rui Zhang, Russ Salakhutdinov, Ryan Andrew Chi, Seungjae Ryan Lee, Ryan Stovall, Ryan Teehan, Rylan Yang, Sahib Singh, Saif M. Mohammad, Sajant Anand, Sam Dillavou, Sam Shleifer, Sam Wiseman, Samuel Gruetter, Samuel R. Bowman, Samuel Stern Schoenholz, 

18 

Sanghyun Han, Sanjeev Kwatra, Sarah A. Rous, Sarik Ghazarian, Sayan Ghosh, Sean Casey, Sebastian Bischoff, Sebastian Gehrmann, Sebastian Schuster, Sepideh Sadeghi, Shadi Hamdan, Sharon Zhou, Shashank Srivastava, Sherry Shi, Shikhar Singh, Shima Asaadi, Shixiang Shane Gu, Shubh Pachchigar, Shubham Toshniwal, Shyam Upadhyay, Shyamolima Shammie Debnath, Siamak Shakeri, Simon Thormeyer, Simone Melzi, Siva Reddy, Sneha Priscilla Makini, SooHwan Lee, Spencer Torene, Sriharsha Hatwar, Stanislas Dehaene, Stefan Divic, Stefano Ermon, Stella Biderman, Stephanie Lin, Stephen Prasad, Steven Piantadosi, Stuart Shieber, Summer Misherghi, Svetlana Kiritchenko, Swaroop Mishra, Tal Linzen, Tal Schuster, Tao Li, Tao Yu, Tariq Ali, Tatsunori Hashimoto, Te-Lin Wu, Théo Desbordes, Theodore Rothschild, Thomas Phan, Tianle Wang, Tiberius Nkinyili, Timo Schick, Timofei Kornev, Titus Tunduny, Tobias Gerstenberg, Trenton Chang, Trishala Neeraj, Tushar Khot, Tyler Shultz, Uri Shaham, Vedant Misra, Vera Demberg, Victoria Nyamai, Vikas Raunak, Vinay Venkatesh Ramasesh, vinay uday prabhu, Vishakh Padmakumar, Vivek Srikumar, William Fedus, William Saunders, William Zhang, Wout Vossen, Xiang Ren, Xiaoyu Tong, Xinran Zhao, Xinyi Wu, Xudong Shen, Yadollah Yaghoobzadeh, Yair Lakretz, Yangqiu Song, Yasaman Bahri, Yejin Choi, Yichi Yang, Yiding Hao, Yifu Chen, Yonatan Belinkov, Yu Hou, Yufang Hou, Yuntao Bai, Zachary Seid, Zhuoye Zhao, Zijian Wang, Zijie J. Wang, Zirui Wang, and Ziyi Wu. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _Transactions on Machine Learning Research_ , 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=uyTL5Bvosj. [Cited on page 24.] 

- Alessandro Stolfo, Zhijing Jin, Kumar Shridhar, Bernhard Schoelkopf, and Mrinmaya Sachan. A causal framework to quantify the robustness of mathematical reasoning with language models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 545–561, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.32. URL https://aclanthology.org/2023.acl-long.32. [Cited on page 1.] 

- Xiaojuan Tang, Zilong Zheng, Jiaqi Li, Fanxu Meng, Song-Chun Zhu, Yitao Liang, and Muhan Zhang. Large language models are in-context semantic reasoners rather than symbolic reasoners, 2023. [Cited on page 2.] 

- Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, Pier Giuseppe Sessa, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, Amélie Héliou, Andrea Tacchetti, Anna Bulanova, Antonia Paterson, Beth Tsai, Bobak Shahriari, Charline Le Lan, Christopher A. Choquette-Choo, Clément Crepy, Daniel Cer, Daphne Ippolito, David Reid, Elena Buchatskaya, Eric Ni, Eric Noland, Geng Yan, George Tucker, George-Christian Muraru, Grigory Rozhdestvenskiy, Henryk Michalewski, Ian Tenney, Ivan Grishchenko, Jacob Austin, James Keeling, Jane Labanowski, Jean-Baptiste Lespiau, Jeff Stanway, Jenny Brennan, Jeremy Chen, Johan Ferret, Justin Chiu, Justin Mao-Jones, Katherine Lee, Kathy Yu, Katie Millican, Lars Lowe Sjoesund, Lisa Lee, Lucas Dixon, Machel Reid, Maciej Mikuła, Mateo Wirth, Michael Sharman, Nikolai Chinaev, Nithum Thain, Olivier Bachem, Oscar Chang, Oscar Wahltinez, Paige Bailey, Paul Michel, Petko Yotov, Rahma Chaabouni, Ramona Comanescu, Reena Jana, Rohan Anil, Ross McIlroy, Ruibo Liu, Ryan Mullins, Samuel L Smith, Sebastian Borgeaud, Sertan Girgin, Sholto Douglas, Shree Pandya, Siamak Shakeri, Soham De, Ted Klimenko, Tom Hennigan, Vlad Feinberg, Wojciech Stokowiec, Yu hui Chen, Zafarali Ahmed, Zhitao Gong, Tris Warkentin, Ludovic Peran, Minh Giang, Clément Farabet, Oriol Vinyals, Jeff Dean, Koray Kavukcuoglu, Demis Hassabis, Zoubin Ghahramani, Douglas Eck, Joelle Barral, Fernando Pereira, Eli Collins, Armand Joulin, Noah Fiedel, Evan Senter, Alek Andreev, and Kathleen Kenealy. Gemma: Open models based on gemini research and technology, 2024. URL https://arxiv.org/abs/2403.08295. [Cited on page 7.] 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, 

19 

Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv.org/abs/2307.09288. [Cited on page 7.] 

- Aniket Vashishtha, Abbavaram Gowtham Reddy, Abhinav Kumar, Saketh Bachu, Vineeth N Balasubramanian, and Amit Sharma. Causal inference using llm-guided discovery, 2023. [Cited on page 1.] 

- Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021. [Cited on page 7.] 

- Zhaowei Wang, Quyet V. Do, Hongming Zhang, Jiayao Zhang, Weiqi Wang, Tianqing Fang, Yangqiu Song, Ginny Wong, and Simon See. COLA: Contextualized commonsense causal reasoning from the causal inference perspective. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 5253–5271, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.288. URL https://aclanthology.org/2023.acl-long.288. [Cited on pages 1, 2, 24, and 28.] 

- Lilian D. A. Wanzare, Alessandra Zarcone, Stefan Thater, and Manfred Pinkal. A Crowdsourced Database of Event Sequence Descriptions for the Acquisition of High-quality Script Knowledge. In _Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)_ , 2016. URL https://aclanthology.org/L16-1556. [Cited on pages 2, 5, and 28.] 

- Noah Weber, Rachel Rudinger, and Benjamin Van Durme. Causal inference of script knowledge. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 7583–7596, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020. emnlp-main.612. URL https://aclanthology.org/2020.emnlp-main.612. [Cited on page 1.] 

- Moritz Willig, Matej Zeˇcevi´c, Devendra Singh Dhami, and Kristian Kersting. Probing for correlations of causal facts: Large language models and causality, 2023a. URL https://openreview.net/ forum?id=UPwzqPOs4-. [Cited on page 10.] 

- Moritz Willig, Matej Zeˇcevi´c, Jonas Seng, and Florian Peter Busch. Causal concept identification in open world environments. In Martin Mundt, Keiland W. Cooper, Devendra Singh Dhami, Adéle Ribeiro, James Seale Smith, Alexis Bellot, and Tyler Hayes, editors, _Proceedings of The First AAAI Bridge Program on Continual Causality_ , volume 208 of _Proceedings of Machine Learning Research_ , pages 52–58. PMLR, 07–08 Feb 2023b. URL https://proceedings.mlr.press/ v208/willig23a.html. [Cited on page 10.] 

- Linying Yang, Oscar Clivio, Vik Shirvaikar, and Fabian Falck. A critical review of causal inference benchmarks for large language models. In _AAAI 2024 Workshop on ”Are Large Language Models Simply Causal Parrots?”_ , 2023. URL https://openreview.net/forum?id=mRwgczYZFJ. [Cited on page 1.] 

- Matej Zeˇcevi´c, Moritz Willig, Devendra Singh Dhami, and Kristian Kersting. Causal parrots: Large language models may talk causality but are not causal. _Transactions on Machine Learning Research_ , 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=tv46tCzs83. [Cited on pages 1, 10, and 24.] 

- Cheng Zhang, Stefan Bauer, Paul Bennett, Jiangfeng Gao, Wenbo Gong, Agrin Hilmkil, Joel Jennings, Chao Ma, Tom Minka, Nick Pawlowski, and James Vaughan. Understanding causality with large language models: Feasibility and opportunities, 2023. [Cited on page 1.] 

- Honghua Zhang, Liunian Harold Li, Tao Meng, Kai-Wei Chang, and Guy Van den Broeck. On the paradox of learning to reason from data. In _International Joint Conference on Artificial Intelligence_ , 2022a. URL https://api.semanticscholar.org/CorpusID:248986434. [Cited on page 10.] 

20 

- Jiayao Zhang, Hongming Zhang, Weijie Su, and Dan Roth. ROCK: Causal inference principles for reasoning about commonsense causality. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pages 26750–26771. PMLR, 17–23 Jul 2022b. URL https://proceedings.mlr.press/v162/ zhang22am.html. [Cited on pages 2, 8, 23, and 29.] 

21 

## **Appendix** 

## **Table of Contents** 

|**A **|**COLD Framework Details**|**COLD Framework Details**|**23**|
|---|---|---|---|
||A.1|Adherence to SUTVA . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|23|
||A.2|Comparison with previous Causal Reasoning Datasets/Benchmarks. . . . . . . . .|23|
||A.3|Observational Graphs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|24|
|**B **|**Algorithms in the COLD Framework**||**24**|
|**C **|**Backdoor Adjustments**||**26**|
|**D **|**Experiments and Results**||**28**|
||D.1|Compute Resources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|28|
||D.2|Evaluation using Average Treatment Effect (ATE) . . . . . . . . . . . . . . . . . .|28|
|**E**|**Prompt Templates for Language Model based Experiments**||**29**|
|**F**|**Additional Results**||**30**|
||F.1|Temporally Plausible Choices in Causal Triplets . . . . . . . . . . . . . . . . . . .|30|
|**List of**||**Tables**||
||5|Comparison with existing Causal Benchmarks/Datasets in NLP . . . . . . . . . . .|24|
||6|Human performance<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|25|
||7|Results for Causal and Causal Temporal Triplets . . . . . . . . . . . . . . . . . . .|32|
|**List of**||**Figures**||
||4|Causal Graphical Model of Events. _E_1temporally precedes_E_2, and_z_is trajectory||
|||variable, which assumes a values_t_where_t ∈_All trajectories from start to_E_1<br>. . .|26|
||5|Input prompt formats for the MCQA-based evaluation of autoregressive open-weight||
|||models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|29|
||6|Qualitative examples for the MCQA-based evaluation of autoregressive open-weight||
|||models (e.g.,llama(-2),GPT-J, etc.).<br>. . . . . . . . . . . . . . . . . . . . . . .|30|
||7|Input prompt formats for the∆estimation (temporal) via Language models. . . . .|30|
||8|Input prompt formats for computing the causal estimand using the autoregressive||
|||open-weight models (using backdoor criterion)<br>. . . . . . . . . . . . . . . . . . .|31|
||9|The_“observational graph"_for the activityBaking a Cake.<br>. . . . . . . . . . . .|33|
||10|The_“observational graph"_for the activityGoing Grocery Shopping.<br>. . . . . .|34|
||11|The_“observational graph"_for the activityGoing on a Train.<br>. . . . . . . . . .|35|
||12|The_“observational graph"_for the activityPlanting a Tree. . . . . . . . . . . .|36|
||13|The_“observational graph"_for the activityRiding on a Bus. . . . . . . . . . . .|37|



22 

## **A COLD Framework Details** 

The **COLD** framework consists of Observational Distributions represented in the form of DAGs ( _Go_ ) along with the corresponding causal graphs ( _Gc_ ) governing the dependency of occurrence between the events. Table 2 highlights the total number of causal queries that can be created using the framework. Table 1 shows a qualitative comparison with the COPA dataset Gordon et al. [2012] and the triplet samples coming from the **COLD** framework. 

## **A.1 Adherence to SUTVA** 

In causal literature, a fundamentally acknowledged _Stable Unit Treatment Value Assumption_ (SUTVA) [Cox, 1958, Rubin, 1980]) requires that for each unit (e.g., sequence of events), there is only one version of the non-treatment, i.e., for an event in the sequence, there lie only two versions occurring and not occurring. SUTVA plays a vital role in causal inference by ensuring that each unit’s treatment assignment has a consistent impact, facilitating the accurate estimation of treatment effects. Although, in the past, researchers have created various datasets that capture the causal relationship between real-world events [Gordon et al., 2012, Du et al., 2022], the problem of achieving the SUTVA assumption has remained challenging. For example, given events (taken from the COPA dataset [Gordon et al., 2012]) _E_ 1: “The teacher assigned homework to students” and _E_ 2: “The students groaned,” it becomes challenging to define _¬E_ 1 since there are enormous possibilities that may have occurred at the same time (in place of _E_ 1) that negates _E_ 1, making it difficult to define an event of not having done something. Recent work by Zhang et al. [2022b], proposes to use multiple alterations of events for capturing _¬E_ 1, violating the SUTVA assumption. In this work, we highlight that if we define a closed system, capturing a commonsense activity, it facilitates adherence to SUTVA assumptions as closely as possible. For example, in the activity of “going via an airplane,” one would have either “checked-in the luggage” ( _E_ 1) or “skipped checking-in luggage" due to smaller bags ( _¬E_ 1). Moreover, developing a causal setup with observations has always been a challenging problem in the wild and often requires few assumptions, as the strong causal link can only be established in an ideal world where randomized controlled trials (RCTs) are feasible. In our framework, adhering to SUTVA comes naturally where, in a trajectory, an occurrence of an event can be intervened to obtain an alternate trajectory, reaching an ideal setup facilitating causal reasoning in daily commonsensical activities. 

## **A.2 Comparison with previous Causal Reasoning Datasets/Benchmarks** 

Table 5 shows a broad overview of the existing causal Dataset/Benchmarks presented in the NLP community. We find that most of the existing set of work relies on real-world events to reason about causality in NLP, where human annotators are asked to reason causally between the nature of events. However, most of these datasets/benchmarks try to establish a connection using a simple question prompt, which may not be enough to construct the underlying causal graph. Moreover, most of the real-world grounding-based methods remain open-ended due to the events taking place in the wild, making it difficult to consider constructing a causal graph where multiple variables play a role. More recently, with increased research attention on the causal reasoning abilities of LLMs, researchers have tried framing causal queries based on a causal inference engine, requiring the underlying causal graphs. However, when constructing causal queries from prompting LLMs, natural language is used to verbalize the causal concepts in the form of symbolic variables that may not have a real-world grounding.[2] Moreover, the created causal queries are difficult for a human with little or no knowledge of causal inference concepts.[3] Table 5 shows a comparison of all these features in detail, where **COLD** satisfies all the features. 

We realize this is a first-of-its-kind framework built over real-world events and contains the underlying causal graph. Having both the Observational Distribution (representing the enormous event sequences present in daily activity) and the manually created underlying causal graph helps facilitate an in-depth analysis of the causal reasoning abilities of LLMs. Moreover, the same framework can further be extended in various ways: 1) Extending the number of activities: In the current version of the framework, we only consider 5 daily activities to provide an in-depth analysis. In the future, those can be extended to incorporate more such activities. 2) Extending the scope of activities: The tasks 

> 2https://huggingface.co/datasets/causalnlp/corr2cause 

> 3https://huggingface.co/datasets/causalnlp/CLadder 

23 

Table 5: **Comparison of causal experimental settings used in prior LLM evaluation benchmarks.** The real-world grounding plays a crucial role in evaluating LLMs, which is not present in the symbolic benchmarks. 

|**Datasets/Benchmarks**|**Real-World**|**Causal Graph**|**Symbolic**|**Exhaustive**|**# Samples**|
|---|---|---|---|---|---|
|SemEval2021 Task8 [Hendrickx et al.,2010]|✓|✗|✗|✗|1331|
|EventCausality [Do et al.,2011]|✓|✗|✗|✗|414|
|COPA [Gordon et al.,2012]|✓|✗|✗|✗|1000|
|Causal-TimeBank [Mirza et al.,2014]|✓|✗|✗|✗|-|
|CaTeRS [Mostafazadeh et al.,2016]|✓|✗|✗|✗|320 stories (1.6K sent)|
|BECauSE [Dunietz et al.,2017]|✓|✗|✗|✗|-|
|Event2Mind [Rashkin et al.,2018]|✓|✗|✗|✗|25K event phrases|
|ATOMIC [Sap et al.,2019a]|✓|✗|✗|✗|877K|
|SocialIQA [Sap et al.,2019b]|✓|✗|✗|✗|37K|
|TimeTravel [Qin et al.,2019]|✓|✗|✗|✗|81.4K|
|Abductive (ART) [Bhagavatula et al.,2020]|✓|✗|✗|✗|20K narr, 200K expl|
|Com2Sense [Singh et al.,2021]|✓|✗|✗|✗|4k sentence pairs.|
|TellMeWhy [Lal et al.,2021]|✓|✗|✗|✗|30K questions|
|CRASS [Frohberg and Binder,2022]|✓|✗|✗|✗|274 PCT|
|e-CARE [Du et al.,2022]|✓|✗|✗|✗|20K CR questions|
|CausalQA [Bondarenko et al.,2022]|✓|✗|✗|✗|1.1 Million|
|COLA [Wang et al.,2023]|✓|✗|✗|✗|1,360 event pairs|
|CRAB [Romanou et al.,2023]|✓|✗|✗|✗|2.7K pairs|
|CausalJudgement [Srivastava et al.,2023]|✓|✓|✗|✗|-|
|Corr2Cause [Jin et al.,2023]|✗|✓|✓|✓|200K|
|CausalParrots [Zeˇcevi´c et al.,2023]|✓|✓|✓|✗|-|
|CLadder [Jin et al.,2024]|✗|✓|✓|✗|10K|
|**COLD (ours)**|✓|✓|✓|✓|_∼_9.52 Million|



used in the activities are generic and capture commonsense; for validating domain-specific causal reasoning abilities, the framework could be extended to domain activities, for example, cooking a specific recipe where adding different ingredients causes a variation in taste. 3) Extending the type of causal queries: while constructing the causal queries, we considered the simplest task of finding the more plausible cause/effect given two options as events, keeping only unconditional _d−_ separation as the primary condition. The framework can directly be extended, keeping causal queries inspired from Jin et al. [2023, 2024]. 

On the analysis front, we realize that the possibilities of an in-depth analysis increase by a significant margin. In this work, we shed light on a few mechanisms for validating causal reasoning abilities via zero-shot CCR (compared to previous works that rely on training and further testing on similar datasets). We specifically focused on open-weight models for better applicability in the future and proposed a few mechanisms for estimating the causal relationships between the events. This opens up several possible avenues for an in-depth analysis of LLMs. 

## **A.3 Observational Graphs** 

Fig. 9, Fig. 10, Fig. 11, Fig. 12, and Fig. 13 shows the _“observational graphs"_ for the activity Baking a Cake, Going Grocery Shopping, Going on a Train, Planting a Tree, and Riding on a Bus respectively. 

## **B Algorithms in the COLD Framework** 

In this section, we provide insights into the Algorithms used in the **COLD** framework. We start with Algorithm 1, which creates causal query triplets given the observational graphs _Go_ and along with the Causal Graphs _Gc_ . 

**Remark:** Temporal precedence is generally assumed essential for defining causation, and it is one of the most important clues that is used to distinguish causal from other types of associations [Mill, 1898, Hill, 1965, Pearl and Verma, 1995]. For our framework, we also consider the topologically sorted order obtained from the observational graphs and use the temporal order to define the causal query triplets, i.e., the cause events will always precede the effect events. 

**Creating Causal Query Triplets:** The Algorithm 1 is designed to sample all the possible causal query triplets to construct a dataset for validating causal reasoning ability over an activity. Provided the observational graphs _Go_ and the Causal Graphs _Gc_ for an activity, we first sample all the possible node triplets in the graph. Later, we iterate over the set of triplets and check if one of the nodes in 

24 

Table 6: **Human validation done for a small sample of 100 causal query triplets.** Overall we find that humans do perform well in causal reasoning about these daily activities. 

|Human Annotators|**cake**|**shopping**|**train**|**tree**|**bus**|Average|
|---|---|---|---|---|---|---|
|Subject 1|95|95|90|100|90|94|
|Subject 2|100|100|90|95|90|95|
|Subject 3|100|100|85|85|70|88|
|Subject 4|100|100|95|90|85|94|
|Subject 5|100|85|95|90|80|90|
|Average|99.00|96.00|91.00|92.00|83.00|**92.20**|



the triplet ( _ni, nj, nk_ ) is _d_ -separated. Further, the _d_ -separated node becomes the wrong choice. The remaining two events become premise and correct choices, depending on their temporal order. For example, if _ni_ is the node that is _d_ -separated from _nj_ and _nk_ , we check if _nj_ and _nk_ have a causal link between them in _Go_ . If _nj_ and _nk_ are found to have a causal link, we create two triplets using the temporal ordering between _nj_ and _nk_ . The temporal link _nj → nk_ leads to an ‘effect’ query where _nj_ becomes ‘premise’ and _nk_ becomes ‘correct choice,’ and a ‘cause’ query where _nj_ becomes ‘correct choice’ and _nk_ becomes the ‘premise.’ Note in Algorithm 1 [Store tuple], we only show one such instance for brevity; the implementation will consist of another mirror instance (i.e., for every [Store tuple] both ‘cause’ and ‘effect’ question triplets are stored to the dataset). 

The understanding of these activities is generic, and LLMs do provide a suitable set of sequences when prompted to generate a list of steps to complete the activity. The constructed dataset helps to rigorously validate the understanding of the activity through an enormous number of causal query triplets. The results show that although the LLMs can explain the activity in detail, including generating correct steps for performing tasks, causally reasoning about the set of events remains challenging. 

**Human validation:** To get a rough estimate of the human performance on the created causal reasoning queries, we also perform a small-scale human study, where the annotators are given a set of randomly chosen 100 causal queries. The human subjects were graduate students of computer science who were given a brief tutorial about counterfactual reasoning. Table 6 shows the obtained results. We would like to mention that validating human performance is challenging due to the nature of the causal reasoning task. The nature of counterfactual reasoning requires the human/algorithm to assume a similar alternate world/universe with only a particular happening or not happening to approximate the causal strength between the events. These imaginations can be expressed in statements as highlighted by Pearl and Mackenzie [2018], Pearl et al. [2016], containing an “if” statement in which the “if” portion is untrue or unrealized (aka counterfactual). The “if” portion of a counterfactual is called the hypothetical condition, or more often, the antecedent, making it challenging (cognitively heavy) to conduct a human evaluation. Please note that the study is performed using only a small sample of 100 causal query triplets out of thousands of queries, and the presented results only provide a rough estimate that may not generalize for a larger number of queries. Hence, a comparison of human study results and LLMs is not fair, and the presented human performance estimates may not be true representative of the entire population. Interactions with human subjects also revealed that they tend to confuse temporality and causality (similar findings were reported by Do et al. [2011]). 

**Evaluating over Causal Query Triplets:** Algorithm 2 makes use of the causal estimands ∆ to compare the causal strength between the premise event and the choice events. We consider the causal estimand computed between the premise and the available set of choices and predict the label corresponding to the high ∆ values. For a given causal query from the created causal query triplet dataset _D_ , where each data entry _Di_ corresponds to ( _p, c_ 1 _, c_ 2 _, q, l_ ), i.e., premise event, choice 1, choice 2, question and the label respectively. As the task is to predict the more plausible cause/effect of the given premise event, we create two event pairs, ( _p, c_ 1) and ( _p, c_ 2), and compute the causal estimand ∆ for both the pairs using the temporal or the backdoor scheme (described below in Algorithm 3). Note that the order of events given to ∆ _M_ is in _E_ 1 and _E_ 2 format, i.e. ∆ _M_ ( _E_ 1 _, E_ 2). Using the temporal precedence (highlighted as remark above), the cause event will always precede the effect event temporally. Hence, for a causal query with the question as ‘cause’, the causal estimand is estimated as ∆ _M_ ( _c[i]_ 1 _[, p][i]_[)][,][ ∆] _[M]_[(] _[c][i]_ 2 _[, p][i]_[)][ and][ ∆] _[M]_[(] _[p][i][, c][i]_ 1[)][,][ ∆] _[M]_[(] _[p][i][, c][i]_ 2[)][ when the question is ‘effect.’] Further, based on the estimated ∆ _M_ scores, the more plausible cause/effect is predicted. 

25 

**Algorithm 1** Creating Causal Query Triplets 

_Gc_ : Causal Graphical Model; _Go_ : Observational Graph; **ds** ( **G** _,_ **x** _,_ **y** ): True iff ( **x** _,_ **y** ) are _d-separated_ unconditionally in any DAG **G** ; **dc** ( **G** _,_ **x** _,_ **y** ): True iff ( **x** _,_ **y** ) are _d-connected_ unconditionally in any DAG **G** ; **genSamples** ( **G** ): Generates all possible node triplets ( _ni, nj, nk_ ) from DAG **G** such that _i ≤ j ≤ k_ , where ( _i, j, k_ ) are the respective indices of nodes in a topologically sorted list of nodes; **A** ( **G** _,_ **x** _,_ **y** ): True iff **x** is an ancestor of **y** in DAG **G** . _**⊕**_ : Exclusive OR operator; _p_ : premise; _c_ 1: choice 1; _c_ 2: choice 2; _q_ : question; _l_ : answer (label) ( _p, c_ 1 _, c_ 2 _, q, l_ ) _∈D D_ = [ ] [Empty Dataset] _S ∼_ **genSamples** ( _G_ **o** ) [Generate Samples] **for** ( _ni, nj, nk_ ) in _S_ **if ds** ( _Gc, ni, nj_ ) **then if dc** ( _Gc, ni, nk_ ) _**⊕**_ **dc** ( _Gc, nj, nk_ ) _q_ = ‘cause’ _l_ = arg max[ **dc** ( _Gc, ni, nk_ ) _,_ **dc** ( _Gc, nj, nk_ )] _p, c_ 1 _, c_ 2 _← nk, ni, nj_ **if A** ( _Gc, cl, p_ ) **then** `APPEND` ( _D,_ [( _p, c_ 1 _, c_ 2 _, q, l_ )) [Store tuple] **else if ds** ( _Gc, nj, nk_ ) **then if dc** ( _Gc, ni, nj_ ) _**⊕**_ **dc** ( _Gc, ni, nk_ ) _q_ = ‘effect’ _l_ = arg max[ **dc** ( _Gc, ni, nk_ ) _,_ **dc** ( _Gc, ni, nj_ )] _p, c_ 1 _, c_ 2 _← ni, nj, nk_ **if A** ( _Gc, p, cl_ ) **then** `APPEND` ( _D,_ ( _p, c_ 1 _, c_ 2 _, q, l_ )) [Store tuple] **return** _D_ 

**Computing** ∆ _M_ **:** Algorithm 3 depicts the process of computing an unbiased estimate for the causal estimand. The causal strength is computed between two events _E_ 1 and _E_ 2 where _E_ 1 is assumed to be preceding _E_ 2 temporally. To make an unbiased estimate based on the provided options, we consider normalizing the obtained probability scores by flipping the options and providing the same query prompt to the Language Model. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0026-03.png)


where _ϕ_ denotes the prompt template as shown in Figure 8 (top) and _ϕf_ denotes the same prompt with flipped options, Figure 8 (bottom). The overall equation helps normalize the prediction probabilities of the ‘Increase’ option by using the probabilities of the ‘Decrease’ option. Finally, these normalized scores are computed for multiple trajectories _ti_ in the backdoor adjustment scheme to compute the causal estimands _pM_ ( _E_ 2 _| do_ ( _E_ 1)) and _pM_ ( _E_ 2 _| do_ ( _¬E_ 1)) that help estimate the causal strength ∆ _M_ between the events _E_ 1 and _E_ 2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0026-05.png)


**----- Start of picture text -----**<br>
z<br>E1 E2<br>**----- End of picture text -----**<br>


Figure 4: Causal Graphical Model of Events. _E_ 1 temporally precedes _E_ 2, and _z_ is trajectory variable, which assumes a values _t_ where _t ∈_ All trajectories from start to _E_ 1 

## **C Backdoor Adjustments** 

A set of variables _W_ satisfies the backdoor criterion relative to _T_ and _Y_ if the following are true 

- (A) _W_ blocks all backdoor paths from _T_ to _Y_ i.e. blocking confounding or non-causation association paths 

- (B) _W_ doesn’t contain any descendants of _T_ 

26 

**Algorithm 2** Evaluating Causal Query Triplets 

_Tn_ : _n_ unique trajectories from Start node (start of the activity) to node _E_ 1 _pM_ ( _E_ 2 _|do_ ( _E_ 1)) = 1 � _p_ ˆ( _E_ 2 _|E_ 1 _, z_ = _t_ ) _|Tn| t∈Tn pM_ ( _E_ 2 _|do_ ( _¬E_ 1)) = 1 � _p_ ˆ( _E_ 2 _|¬E_ 1 _, z_ = _t_ ) _|Tn| t∈Tn_ 

∆ _M_ : Returns **Average treatment effect** ( _pM_ ( _E_ 2 _|do_ ( _E_ 1)) _− pM_ ( _E_ 2 _|do_ ( _¬E_ 1))) to determine the causal effect of event _E_ 1 on event _E_ 2 _p_ : premise; _c_ 1: choice 1; _c_ 2: choice 2; _q_ : question; _l_ : label _D_ : Set of all causal queries _Di_ : Causal query ( _p, c_ 1 _, c_ 2 _, q, l_ ), where ( _p, c_ 1 _, c_ 2 _, q, l_ ) _∈D_ **for** _Di_ in _D_ **do if** _q[i]_ == ‘cause’ **then** prediction _←_ arg max �∆ _M_ ( _c[i]_ 1 _[, p][i]_[)] _[,]_[ ∆] _M_[(] _[c][i]_ 2 _[, p][i]_[)] � **else if** _q[i]_ == ‘effect’ **then** prediction _←_ arg max �∆ _M_ ( _p[i] , c_ 1 _[i]_[)] _[,]_[ ∆] _M_[(] _[p][i][, c][i]_ 2[)] � **end if** _η ← η_ + 1 ( _prediction_ == _li_ ) **end for return** _η/|D|_ 

**Algorithm 3** Computing Causal Estimand 

_E_ 1 _, E_ 2 : Events in an given activity _Tn_ : Set of _n_ trajectories (temporally ordered sequence of events) _sM_ ( _E_ 1 _, E_ 2 _, ϕ_ ) _←_ score of token **A** (associated with option ‘ _increase_ ’) using prompt _ϕ_ of model _M_ . _sM_ ( _E_ 1 _, E_ 2 _, ϕf_ ) _←_ score of token **B** (associated with option ‘ _increase_ ’) in prompt _ϕf_ (flipped options in _s_ prompt _s_ ˜˜ _MM_ (( _EE_ 11 _ϕ, E, E_ ) of model22 _, ϕ, ϕ_ ) _f ←_ ) _←_ score of token _M_ score of token. **B A** (associated with option ‘ (associated with option ‘ _decreasedecrease_ ’) using prompt’) in prompt _ϕ ϕf_ of model(flipped options in _M_ . prompt _ϕ_ )of model _M_ Norm. Score _fM_ ( _E_ 1 _, E_ 2 _, ϕ_ ) _← sM_ ( _E_ 1 _, E_ 2 _, ϕ_ ) + _sM_ ( _E_ 1 _, E_ 2 _, ϕf_ ) _pp_ ˆˆ(( _EE_ 22 _||¬EE_ 1 _, z_ 1 _, z_ == _t_ ) _t ←_ ) _←fMfM_ (((( _t, Et, ¬_ 1) _E, Es_ 1 _M_ )2 _, E_ ( _, ϕE_ 2)1 _, ϕ, E_ )2 _, ϕ_ ) + _sM_ ( _E_ 1 _, E_ 2 _, ϕf_ ) + ˜ _sM_ ( _E_ 1 _, E_ 2 _, ϕ_ ) + ˜ _sM_ ( _E_ 1 _, E_ 2 _, ϕf_ ) 1 _pM_ ( _E_ 2 _|do_ ( _E_ 1)) _← |Tn|_ � _t∈Tn[p]_[ˆ][(] _[E]_[2] _[|][E]_[1] _[, z]_[=] _[ t]_[)] _pM_ ( _E_ 2 _|do_ ( _¬E_ 1)) _← |T_ 1 _n|_ � _t∈Tn[p]_[ˆ][(] _[E]_[2] _[|¬][E]_[1] _[, z]_[=] _[ t]_[)] ∆ _M ← pM_ ( _E_ 2 _|do_ ( _E_ 1)) _− pM_ ( _E_ 2 _|do_ ( _¬E_ 1)) **return** ∆ _M_ 

Then, _W_ satisfies the backdoor criterion [Pearl et al., 2016, Neal, 2020]. 

Adhering to the above conditions of the backdoor criterion, it is reasonable to assume that the trajectory _t_ (temporally ordered sequence of events) till _E_ 1 will contain the events that confound the event _E_ 1 and event _E_ 2 (condition A). Every event trajectory till _E_ 1 will temporally precede _E_ 1(condition B). Hence, the trajectory variable will satisfy backdoor criteria in the proposed closed system. The domain of the trajectory variable is a set of all trajectories till _E_ 1. Therefore, conditioning on _t_ closes all paths that induce non-causal associations. The generic representation of an approximate causal graphical model involving _E_ 1 _, E_ 2, and _t_ is shown in Figure 4, and can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0027-07.png)


Where _T_ is a set of all trajectories from the start of the activity till the event _E_ 1(excluding _E_ 1). 

27 

## **D Experiments and Results** 

## **D.1 Compute Resources** 

We perform all the experiments using a machine with 5 NVIDIA A100 GPUs. We use only the open-weight models with frozen parameters to present the results for better reproducibility in the future. 

## **D.2 Evaluation using Average Treatment Effect (ATE)** 

**Establishing Causal Relationships:** To validate the causal reasoning ability, the MCQA-based approach can be further extended to estimate the causal estimation and denote the causal strengths between the events. Establishing cause-and-effect relationships can be achieved through various statistical analyses. The strength of cause-and-effect relationships is approximated by statistically analyzing events’ behavior using observational data (PC-Algorithm, Spirtes et al. [2000b]). Moreover, some of the recent works [Wang et al., 2023] highlight the role of context in determining the causal relationships between the events. To extend our analysis of causal reasoning abilities in the proposed framework, we use the backdoor adjustments in LLMs as explained in the main paper. Moreover, we also perform an interesting analysis of the observational graphs for estimating ∆ statistically. 

**1) Through Original Trajectories:** DeScript Wanzare et al. [2016] collects data by considering _∼_ 100 ESDs written by different crowd-sourced workers. We use the original Trajectories (ESDs) _To_ written by humans present in the DeScript dataset. These ESDs provide the original flow in the graph directly coming from crowdsourced workers. We consider these as the original trajectories _To_ . Applying the backdoor criterion (Eq. 2) over these trajectories _To_ . An interventional distribution similar to the previous section is computed considering the likelihood of occurrence of _E_ 2 under each treatment ( _E_ 1 and _¬E_ 1) for only these trajectories _To_ . These estimations are further used to compute the treatment effect using the Eq. 1. We denote the causal risk difference (∆) computed with _To_ as ∆ _o_ . 

**2) Through Observational Graphs:** The observational graphs provide a proxy for the underlying knowledge about the activity, covering all possible sets of events, i.e., starting from the start node, one can trace multiple trajectories that will essentially define the way of performing the activity. For every pair of connected events ( _ei, ej_ ), the edge between them represents the probable transition from _ei_ to _ej_ with some non-zero probability. However, a noteworthy point is that the transition probability between two connected events ( _ei, ej_ ) can vary depending on the design choices/transition function _T_ ( _ei, ej_ ) _→_ (0 _,_ 1]. We define this transition function in two ways: 1) **Uniform Node Transition** ( _Tn_ ): The transition probability from current node _ei_ to next probable events _ej ∈ Eij_ would be uniform that is _T_ ( _ei, ej_ ) = _|o_ 1 _i|_[, where] _[ |][o][i][|]_[ represents number of outgoing edges from event] _[ e][i]_[.][(i.e.,] assuming after an event, the choice of the next event is uniform from the possible events). 2) **Uniform Trajectory Transition (** _Tt_ **):** Another way to take the set of events in an activity (trajectory) is by considering all the possible paths being equally probable, i.e., across the entire population the same activity will be represented with one of the possible trajectories. Hence we can define the transition function with each trajectory _ti_ = ( _estart, e_ 2 _, ..., eend_ ) (sequence of events from starting to ending) having the same probability, i.e.: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0028-07.png)


ˆ Further, given a transition function _T_ , computation of _p_ ( _E_ 2 _|E_ 1 _, z_ = _ti_ ) becomes straightforward as _p_ ( _E_ 2 _|E_ 1), since the next course of trajectories after _E_ 1 will be decided given _E_ 1 has occurred. Analytically, it can be computed by counting every trajectory (i.e., _Tij_ ) from _E_ 1 that leads to _E_ 2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0028-09.png)


28 

ˆ For estimating the probability _p_ ( _E_ 2 _|¬E_ 1 _, z_ = _ti_ ), we make use of the observational graph by considering all the parent nodes of _E_ 1 and compute the probability of reaching _E_ 2 from the parent (i.e., last event of trajectory _ti_ ) avoiding the occurrence of _E_ 1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0029-01.png)


also, _p_ ( _z_ = _ti_ ) can be computed as product of each transition in _ti_ , i.e 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0029-03.png)


Computations from Equations 5, 6, and 7 are used in the backdoor adjustment defined in Equation 2 for estimating the interventional likelihood of occurrence of _E_ 2 under each treatment ( _E_ 1 and _¬E_ 1) and causal risk difference (∆). Note depending on the choice of transition function ( _Tn_ or _Tt_ ), we obtain two deltas ∆ _n_ and ∆ _t_ , respectively. 

## **E Prompt Templates for Language Model based Experiments** 

We present the various prompt templates used to estimate the temporal link between the events in Figure 7. For BERT-based models, we use the MLM-trained models for predicting the masked token given a sentence (Previously, a similar approach was adopted by Zhang et al. [2022b]). In contrast, for autoregressive models, we frame the prompt as a question-answer objective, taking inspiration from [Robinson and Wingate, 2023], where a multiple-choice-based question is framed to predict the answer in the form of the option IDs. The prompt is intentionally structured so that the LLM is intended to predict a single token (Such as “A”, “B”, etc.). Robinson and Wingate [2023] highlights the advantages of MCQA-based evaluation over cloze evaluation (where the LLMs are expected to generate the entire answer in a cloze test), leading to a significant boost in various tasks, including commonsense-based tasks. 

For our prompt-based evaluation experiments over the generated causal triplets, we follow the same MCQA-based strategy and frame the prompts accordingly for a fair evaluation. Figure 5 presents various prompt templates for autoregressive experiments, and Figure 6 shows a few qualitative examples for the framed causal query templates. 

Consider the activity of activity name. [ in-context examples (if few-shot/in-context learning experiment) ] Which of the following events (given as options A or B) is a plausible question (cause/effect) of the event premise? A. choice1 

B. choice2 Answer: A 

The following are multiple choice questions about activity name. You should directly answer the question by choosing the correct option. [ in-context examples (if few-shot/in-context learning experiment) ] Which of the following events (given as options A or B) is a plausible question (cause/effect) of the event premise? A. choice1 

B. choice2 Answer: A 

Figure 5: Input prompt formats for the MCQA-based evaluation of autoregressive open-weight models (e.g., llama(-2), GPT-J, etc.). The black text is the templated input. The orange text is the input from the created causal query triplets, where the activity name denotes the description of the activity like baking a cake. The next-token prediction probabilities of the option IDs at the red text is used as the observed prediction distribution. 

29 

Consider the activity of baking a cake. Which of the following events (given as options A or B) is a plausible effect of the event preheat oven to 350 degrees.? A. turn off oven. B. prepare the microwave oven and required utensils Answer: A The following are multiple choice questions about going on a train. You should directly answer the question by choosing the correct option. Which of the following events (given as options A or B) is a plausible cause of the event get the bill for groceries.? A. pay the cashier for your items. B. place cart into cart corral. Answer: A 

Figure 6: Qualitative examples for the MCQA-based evaluation of autoregressive open-weight models (e.g., llama(-2), GPT-J, etc.). 

|In terms of ’before’ and ’after’, the event:|“first event text" would have happened|
|---|---|
|_<_mask_token_>_ the event: “second event text"||
|Consider the activity of activity name.||
|Question: Determine the temporal order.||
|The following events took place: 1. first event|text, 2. second event text|
|Did the first event occur ’before’ or ’after’ the second event? (choose from the given||
|options)||
|A: before||
|B: after||
|Answer: A||



Figure 7: Input prompt formats for the ∆ estimation via Language models. The first row shows the prompt template used for BERT-based language models, where the mask token is predicted. The second row shows the template for autoregressive open-weight models (e.g., llama(-2), GPT-J, etc.). The black text is the templated input. The orange text is the input from the created causal query triplets, where the first event text and second event text comes from the premise and available set of choices. The mask-token prediction probabilities of ‘before’ and ‘after’ and nexttoken prediction probabilities of the option IDs at the red text are used as the observed prediction distribution for BERT-based and GPT-based open-weight models. 

## **F Additional Results** 

## **F.1 Temporally Plausible Choices in Causal Triplets** 

Some of the initial studies [Do et al., 2011] highlight the difficulty in choosing between the causaleffect events and temporal events (that occur in close proximity to the premise event), i.e., temporal relationships are sometimes considered as a causal relationship by human annotators. We also create another version of created causal triplets where the wrong choices are replaced by temporally near nodes (nodes that are at a one-hop distance from the premise node). We call these ‘ _causally hard triplets_ .’ Note the temporal nodes are obtained from the observational graphs _Go_ . Table 7 shows the performance comparison with causal triplets and causal-temporal triplets versions of the same queries. We observe a significant performance drop on the causal-temporal triplets version for most models, highlighting the increased confusion. 

30 

|CAUSAL REASONING ANALYSIS:|||
|---|---|---|
|Context:<br>For the activity activity name. During the activity, the||following set of|
|sequences occurred in order:|||
|[ ordered list of events present in a Trajectory till _E_1]||# Trajectory (_z_ =_Ti_)|
|Further, the event ‘event text for _E_1’ took place.|#|Intervention (_do_(_E_1))|
|Question: Given the above information, will the chances of the occurrence of the event|||
|‘event text for _E_2’increase or decrease?|||
|A. Increase|||
|B. Decrease|||
|Answer: A|**#** (_p_(_E_2 _| do_(_E_1)_, z_ =_Ti_)||
|CAUSAL REASONING ANALYSIS:|||
|Context:<br>For the activity activity name. During the activity, the||following set of|
|sequences occurred in order:|||
|[ ordered list of events present in a Trajectory till _E_1]||# Trajectory (_Ti_)|
|Further, the event ‘event text for _E_1’ did NOT take place.|# Intervention (_do_(_¬E_1))||
|Given the above information, will the chances of the occurrence of the||event ‘event text|
|for _E_2’ increase or decrease?|||
|A. Increase|||
|B. Decrease|||
|Answer: B|**#** (_p_(_E_2 _| do_(_¬E_1)_, z_ =_Ti_)||
|Flipped options variant of the above Prompt Template|||
|CAUSAL REASONING ANALYSIS:|||
|Context:<br>For the activity activity name. During the activity, the||following set of|
|sequences occurred in order:|||
|[ ordered list of events present in a Trajectory till _E_1]||# Trajectory (_z_ =_Ti_)|
|Further, the event ‘event text for _E_1’ took place.|#|Intervention (_do_(_E_1))|
|Given the above information, will the chances of the occurrence of the||event ‘event text|
|for _E_2’increase or decrease?|||
|A. Decrease|||
|B. Increase|||
|Answer: B|**#** (_p_(_E_2 _| do_(_E_1)_, z_ =_Ti_)||
|CAUSAL REASONING ANALYSIS:|||
|Context:<br>For the activity activity name. During the activity, the||following set of|
|sequences occurred in order:|||
|[ ordered list of events present in a Trajectory till _E_1]||# Trajectory (_Ti_)|
|Further, the event ‘event text for _E_1’ did NOT take place.|# Intervention (_do_(_¬E_1))||
|Given the above information, will the chances of the occurrence of the||event ‘event text|
|for _E_2’ increase or decrease?|||
|A. Decrease|||
|B. Increase|||
|Answer: A|**#** (_p_(_E_2 _| do_(_¬E_1)_, z_ =_Ti_)||



Figure 8: Input prompt formats for computing the causal estimand using the autoregressive openweight models (using backdoor criterion) (e.g., llama(-2), GPT-J, etc.). The black text is the templated input. The orange text is the input from the created causal query triplets, where the activity name denotes the description of the activity like baking a cake. The trajectory _Ti_ is obtained using the observational graph _Go_ and contains the sequence of events before the event _E_ 1. The next-token prediction probabilities of the option IDs at the red text is used as the observed prediction distribution. The flipped options variants of the prompts contain the same query with flipped options (i.e., the option ‘Increase’ becomes ‘Decrease’ and vice versa). This is done to make the causal estimand unbiased towards the predicted option token as highlighted in Algorithm 3. 

31 

Table 7: The table provides evaluation results of Language models over the causal and causal temporal triplets. 

|**Triplets**|**Model Name**|**cake**|**shopping**|**train**|**tree**|**bus**|
|---|---|---|---|---|---|---|
||gpt-neo-125M|50.71|50.01|49.99|50.13|50.15|
||gpt-neo-1.3B|44.77|45.69|42.52|45.67|42.89|
|causal triplets|gemma-2b<br>gpt-neo-2.7B<br>phi-2<br>gpt-j-6B<br>Llama-2-7b-chat-hf<br>Mistral-7B-v0.1|53.76<br>50.00<br>85.14<br>49.59<br>77.92<br>77.64|52.19<br>50.01<br>83.65<br>50.02<br>72.41<br>69.38|60.57<br>50.00<br>77.29<br>50.29<br>73.48<br>68.46|60.71<br>50.01<br>82.24<br>49.92<br>72.40<br>72.43|53.64<br>50.00<br>71.74<br>49.93<br>68.21<br>69.37|
||gemma-7b|81.47|82.26|77.24|80.78|70.29|
||Meta-Llama-3-8B|80.79|76.46|76.08|78.21|67.39|
||gpt-neo-125M|50.60|49.80|49.90|50.00|50.20|
|causally hard triplets|gpt-neo-1.3B<br>gemma-2b<br>gpt-neo-2.7B<br>phi-2<br>gpt-j-6B<br>Llama-2-7b-chat-hf<br>Mistral-7B-v0.1<br>gemma-7b|49.50<br>52.30<br>50.00<br>80.00<br>50.20<br>71.60<br>69.20<br>76.30|51.20<br>51.00<br>50.00<br>74.70<br>50.00<br>66.60<br>63.10<br>76.40|48.80<br>56.10<br>50.00<br>67.90<br>50.30<br>68.30<br>64.20<br>69.70|47.50<br>52.20<br>50.00<br>87.50<br>50.00<br>77.00<br>67.90<br>89.70|48.00<br>50.00<br>50.00<br>66.50<br>49.80<br>65.40<br>62.00<br>63.70|
||Meta-Llama-3-8B|77.30|72.20|69.30|83.20|64.30|



32 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0033-00.png)


**----- Start of picture text -----**<br>
choose-recipe<br>buy-ingredients<br>other-return-home<br>other-enter-kitchen other-read-instructions<br>get-ingredients<br>get-utensils<br>get-measuring-utensils<br>preheat<br>measure-ingredients<br>add-ingredients other-make-cake<br>prepare-ingredients<br>grease-cake-tin<br>pour-dough<br>put-cake-oven<br>set-time<br>other-clean<br>wait<br>other-return-oven<br>check<br>other-bake-extra-time turn-off-oven<br>take-out-oven<br>cool-down<br>take-out-cake-tin<br>decorate<br>eat<br>End Task<br>**----- End of picture text -----**<br>


Figure 9: The figure shows the _“observational graph"_ for the activity Baking a Cake. 

33 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0034-00.png)


**----- Start of picture text -----**<br>
make-list<br>take-bags other-turn-car-on other-carry-list other-get-in-car<br>go-grocery-drive<br>other-park-car go-grocery-walk<br>go-grocery-bus get-list-from-car<br>enter<br>take-shop-cart<br>other-shop move-section<br>get-groceries<br>check-list<br>go-checkout other-checkout<br>wait<br>put-conveyor other-present-coupons<br>cashier-scan/weight<br>other-get-total<br>other-get-wallet<br>pay<br>pack-groceries<br>get-receipt other-put-bags-cart other-take-bags<br>bring-vehicle<br>other-place-cart-into-cart-corral<br>leave<br>set-items-in-kitchen<br>End Task<br>**----- End of picture text -----**<br>


Figure 10: The figure shows the _“observational graph"_ for the activity Going Grocery Shopping. 

34 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0035-00.png)


**----- Start of picture text -----**<br>
decide-where-to-go<br>check-time-table<br>other-pack other-get-in-car<br>get-train-station<br>other-park-car<br>enter-station other-get-out-car<br>other-go-ticket-counter<br>other-wait-queue<br>get-tickets<br>other-find-platform<br>get-platform other-walk<br>wait<br>train-arrives other-wait-to-board<br>get-on<br>other-stow-luggage<br>find-place<br>conductor-checks<br>spend-time-train<br>arrive-destination<br>get-off<br>other-take-luggage<br>other-leave-station<br>End Task<br>**----- End of picture text -----**<br>


Figure 11: The figure shows the _“observational graph"_ for the activity Going on a Train. 

35 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0036-00.png)


**----- Start of picture text -----**<br>
other-go-garden-center choose-tree<br>get-tree<br>take-home<br>find-place<br>get-tools<br>other-put-on-gloves<br>dug-hole<br>other-secure-tree place-fertilizers other-check-hole<br>unwrap-root other-score-roots other-get-soil<br>place-root<br>other-hold-tree refill-hole<br>other-check-stability tamp-dirt<br>other-put-mulch other-put-away-tools<br>water<br>tie-stakes-up<br>End Task<br>**----- End of picture text -----**<br>


Figure 12: The figure shows the _“observational graph"_ for the activity Planting a Tree. 

36 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities_images/nips_2024_COLD_Causal_reasOning_in_cLosed_Daily_activities.pdf-0037-00.png)


**----- Start of picture text -----**<br>
other-find-cost<br>check-time-table<br>find-bus other-decide<br>get-bus-stop<br>wait<br>other-signal-driver<br>bus-comes<br>other-driver-opens-door<br>board-bus<br>get-ticket<br>find-place<br>ride<br>spend-time-bus<br>press-stop-button<br>go-exit<br>other-reach<br>other-thank bus-stops<br>get-off<br>End Task<br>**----- End of picture text -----**<br>


Figure 13: The figure shows the _“observational graph"_ for the activity Riding on a Bus. 

37 

