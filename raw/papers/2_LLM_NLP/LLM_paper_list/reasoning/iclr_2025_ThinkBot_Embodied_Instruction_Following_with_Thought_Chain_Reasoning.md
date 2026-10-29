## **ThinkBot: Embodied Instruction Following with Thought Chain Reasoning** 

Guanxing Lu[1] , Ziwei Wang[2][�][*] , Changliu Liu[2] , Jiwen Lu[3] , Yansong Tang[1] 

1Tsinghua Shenzhen International Graduate School, Tsinghua University 2Carnegie Mellon University 3Department of Automation, Tsinghua University 

_{_ lgx23@mails.,lujiwen@,tang.yansong@sz. _}_ tsinghua.edu.cn _{_ ziweiwa2@,cliu6@ _}_ andrew.cmu.edu 

https://guanxinglu.github.io/thinkbot/ 

## **Abstract** 

_Embodied Instruction Following (EIF) requires agents to complete human instruction by interacting objects in complicated surrounding environments. Conventional methods directly consider the sparse human instruction to generate action plans for agents, which usually fail to achieve human goals because of the instruction incoherence in action descriptions. On the contrary, we propose ThinkBot that reasons the thought chain in human instruction to recover the missing action descriptions, so that the agent can successfully complete human goals by following the coherent instruction. Specifically, we first design an instruction completer based on large language models to recover the missing actions with interacted objects between consecutive human instruction, where the perceived surrounding environments and the completed sub-goals are considered for instruction completion. Based on the partially observed scene semantic maps, we present an object localizer to infer the position of interacted objects for agents to achieve complex human goals. Extensive experiments in the simulated environment show that our ThinkBot outperforms the stateof-the-art EIF methods by a sizable margin in both success rate and execution efficiency._ 

## **1. Introduction** 

Designing autonomous agents for diverse household tasks has been highly desired in research of artificial intelligence for a long time. Recent advances in computer vision [17, 41] and natural language processing [3, 28] enable autonomous agents to complete complex human requirements, because the appeared large pre-trained models can comprehend human instruction and perceive the world accurately. Embodied instruction following (EIF) [8, 23, 33, 52] requires the agent to ground human instruction to consecutive task plans with feasible execution, which requires high 

> *� Corresponding author. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning_images/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning.pdf-0001-11.png)


**----- Start of picture text -----**<br>
Task Human Instruction Current Frame<br>To get a  Prepare a spoon  <br>chilled<br>Take a mug<br>mug for<br>coffee ... Prepare a spoon<br>Take a mug Take a mug Take a mug<br>The mug should be chilled, so it may be in the fridge<br>Go to the fridge Open the fridge Take a mug<br>  <br>Prompter<br>ThinkBot<br>**----- End of picture text -----**<br>


Figure 1. Comparison between conventional EIF methods (Prompter [11]) and our ThinkBot. Existing methods directly leverage sparse human instruction to generate action sequence, which usually get stuck due to the incoherence of instruction. Our ThinkBot recovers missing action descriptions by reasoning the thought chain in sparse human instruction, and can successfully complete challenging tasks. 

success rate and completion efficiency. 

To accomplish the challenging EIF task, imitation learning [13, 26, 27, 29, 33, 36, 37, 39, 48] is widely adopted to generate low-level actions from historical observation and given instruction, while they fail to adapt to new scenarios due to the insufficient pair-wise data between human instruction and low-level actions. To this end, modular methods [2, 5, 11, 14, 18, 22, 25] decompose complex 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning_images/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Task Observed Scene Current Frame Object Localizer<br>To  get  a  chilled   <br>mug  for  coffee<br>Human Instruction<br>Prepare a spoon<br>Take a mug Open the fridge<br>Fridge: (x1, z1), Sink: (x2, z2)  Prepare a spoon Semantic Map<br>Next  Step<br>Instruction Completer <br>The mug should  Prepare a spoon<br>be chilled, so it  Go to the fridge<br>may be in the  Open the fridge<br>Subgoal<br>fridge Generation Take a mug<br>Thought Chain Reasoning Coherent Instruction Position Heatmap Take a mug<br>**----- End of picture text -----**<br>


Figure 2. The overall pipeline of ThinkBot, which consists of an instruction completer and an object localizer. The instruction completer generates the coherent instruction with interacted objects based on sparse human instruction and the current visual perception results, and the object localizer predicts the position of the interacted object for manipulation and navigation. 

tasks to high-level planning conditioned on instruction and low-level execution with predefined controllers. However, human instruction is usually sparse with incoherence for action plan generation of agents. For example, in Fig. 1, humans may give the instruction _‘Prepare a spoon, Take a mug’_ for making cold brew coffee. In realistic scenarios, the mug may be stored in a fridge, and the agent may need to open the fridge first to get the mug. Therefore, the dense instruction should be _‘Go to the fridge, open the fridge, and take a mug’_ . Lacking coherent instruction usually disables the agent to acquire feasible action sequences in execution, and the success rate across diverse tasks in complicated indoor environments still remains low. 

In this paper, we propose a ThinkBot agent to accurately complete diverse EIF tasks in interactive environments. Unlike existing methods that directly employ the sparse human instruction for agent action sequence generation, we recover the missing action descriptions for agent execution by reasoning the thought chain in sparse human instruction. Therefore, coherent instruction can be leveraged to generate more feasible agent actions in complex EIF tasks especially with long sequences. More specifically, we first propose an instruction completer based on large language models to predict the missing actions with interacted objects in the sparse human instruction, where we carefully design prompts to consider the perceived objects in the surrounding environments and the completed sub-goals in the task execution sequences. We then present a multimodal object localizer that predicts the position of the recovered missing objects based on the scene semantic maps, where the mined object correlation is also leveraged to enhance the lo- 

calization ability. Extensive experiments on ALFRED [33] show that our ThinkBot outperforms the state-of-the-art EIF methods by a sizable margin in both success rate and execution efficiency. Our main contributions can be summarized as follows: 

- We propose a ThinkBot agent that reasons the thought chain in sparse human instruction for coherence mining to successfully complete complex EIF goals. 

- We present an instruction completer based on large language models to generate the missing actions with interacted objects, and propose an object localizer to predict the position of objects for interaction. 

- We conduct extensive experiments of diverse EIF tasks on ALFRED benchmark, and the results demonstrate that our method achieves higher success rate and path-lengthweighted success rate than the state-of-the-art methods on unseen environments. 

## **2. Related Work** 

**Embodied Instruction Following:** Developing generalist agents that can follow human instruction to complete diverse tasks in interactive environments is a long-standing goal. In the pursuit of this goal, EIF has been widely studied in recent years for high generalizability and practicality. Prior arts can be divided into two categories: endto-end methods and modular methods. End-to-end methods directly generate low-level actions conditioned on the current state of the environment and human instruction [13, 26, 27, 29, 33, 36, 37, 39, 48]. For instance, Pashevich _et al_ . [29] developed an episodic transformer to encode language inputs and the episode history of visual ob- 

2 

servation and actions, which was decoded for action sequence generation with auto-regression. However, end-toend methods often struggle to generalize to unseen scenes due to insufficient pair-wise data between instruction and low-level action sequences. To address this, modular methods [2, 5, 11, 14, 18, 22, 25] plan high-level action sequences and execute them with pre-defined local policies guided by online semantic maps, which are free of the pairwise data between instruction and low-level actions. In modular methods, selecting correct targets and actions for navigation and interaction is very important for searching efficiency and task success. Min _et al_ . [22] directly employed convolutional networks to predict the target position from current semantic map, and Murray _et al_ . [25] proposed a landmark classification model based on BERT [12] for target selection only based on original human instruction. Inoue _et al_ . [11] predicted landmark objects according to offline co-occurrence probabilities of objects evaluated by pre-trained language models, and Kim _et al_ . [14] yield the detailed plan by incorporating the contextual information of natural language instructions. Nevertheless, directly considering the sparse human instruction for agent action generation usually fails to achieve human goals due to the instruction incoherence with missing action descriptions. 

**LLMs for Embodied Agents:** Large language models (LLMs) [3, 28] have demonstrated their capability in embodied AI tasks such as visual-language navigation [4, 7, 19, 30, 32, 49], object navigation [47, 50], open-world exploration [42, 51], and embodied planning [1, 9, 10, 21, 24, 31, 35, 44, 46], where the high generalizability across deployment scenes and downstream tasks of LLMs enables embodied agents to achieve diverse human goals in complex environments. With the rich commonsense embedded in LLMs, fine-grained actions regarding human instruction can be directly generated. Zhou _et al_ . [49] directly prompted LLMs to perform zero-shot sequential action prediction by taking the textual descriptions of historical visual observations as inputs. To facilitate efficient exploration, ESC [50] generated frontier candidates on the observed semantic map, and employed LLM to determine the next frontier by considering hand-crafted soft constraints jointly. While direct generation of fine-grained actions is challenging for LLMs due to the extremely large search space, other methods decompose the overall solution into high-level plan generation and low-level action controlling [1, 9, 10, 24, 44, 46]. Wu _et al_ . [44] crafted a large-scale embodied planning dataset and finetuned different plan generators on the dataset for task plan grounding. Yao _et al_ . [46] generated consecutive plans by prompting LLMs to synergize reasoning and acting, and achieved impressive performance on text-based benchmarks [34]. However, despite the high reasoning ability of LLMs, the spatial localization ability for interacted objects is weak in LLMs. We present 

a multimodal transformer-based object localizer to provide spatial guidance for agents in object interaction accurately. 

## **3. Approach** 

In this section, we first briefly introduce the problem in EIF and our overall pipeline of the ThinkBot agent (Sec. 3.1). Then we detail the instruction completer that recovers the missing actions with interacted objects for coherence mining (Sec. 3.2), and demonstrate the multimodal object localizer which provides spatial guidance for agents in interaction (Sec. 3.3). 

## **3.1. Problem Statement and Overall Pipeline** 

The embodied instruction following task requires an agent in the interactive environment to finish the human goals physically by generating action sequences, where the human goals and the step-by-step instruction are given to the agent for guidance. Following the modular methods, we first generate high-level subgoal sequences and then execute them with a pre-defined controller guided by online semantic maps. In the _tth_ step, the agent needs to generate a highlevel subgoal _At_ based on the current instruction _It_ and the object state _St−_ 1 after implementing the last subgoal by the controller. A high-level plan _At_ is a tuple ( _at, ot, pt_ ), where _at_ is the primitive action and _ot_ means the interacted object with the position _pt_ . In EIF tasks, human instruction is usually sparse with significant incoherence between consecutive steps. Therefore, it is very challenging to generate feasible action _At_ based on the current step-wise instruction _It_ and the object state _St−_ 1 after implementing the last action. For example, humans may provide the instruction _‘Prepare a spoon, take a mug’_ for coffee making. However, the mug may be stored in the fridge in realistic scenes. Consequently, it is difficult for the agent to take the mug with spoon without other instruction after preparing the bread. The coherent instruction should be _‘Prepare a spoon, go to the fridge, open the fridge, take a mug’_ . Therefore, our goal is to recover the missing action descriptions in the sparse human instruction. 

Since the thought chain reveals a series of intermediate steps from the initial problem to the final solution [43], it can be leveraged to decompose the original complex problems and enhance the feasibility of the solution. In EIF tasks, reasoning the thought chain can predict the missing action descriptions in the sparse human instruction to successfully achieve the goal. The overall pipeline of our ThinkBot agent is shown in Figure 2, which consists of an instruction completer recovering the missing actions with interacted objects and an object localizer predicting the object location for agent interaction. For the instruction completer, we design prompts for the pre-trained large language model including the descriptions of scene information and the task completion process, which is expected to reason the 

3 

**Task** :  To  get  a  chilled  mug  for  coffee 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning_images/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning.pdf-0004-01.png)


**----- Start of picture text -----**<br>
System Message Agent Message<br>Role Prepare a spoon  Human<br>ExplanationMessage  Take a mug Instruction<br>Definition  Task<br>Primitive  Completion<br>Actions<br>Observed<br>Response<br>Format Objects<br>Response<br>Thought Chain Recovered Subgoals<br>: The current instruction Prepare a spoon<br>is ‘take a mug’, but mug is Recovered Go to the fridge<br>unobserved. The mug should<br>be chilled, so it may be in Open the fridge<br>the fridge. We need to first<br>check the fridge. Take a mug<br>**----- End of picture text -----**<br>


Figure 3. Input and output of the instruction completer based on LLMs. The input contains system message describing the world properties and agent message demonstrating perceived environment information. The output includes the thought chain in sparse human instruction and missing subgoals with interacted objects. 

thought chain in sparse human instruction to provide coherent instruction. For the object localizer, the generated missing actions with interacted objects and the perceived semantic map of the scene are utilized to predict the object location for the agent to interact with, where multimodal transformers are employed for the alignment between language instruction and visual clues in the environment. Finally, the agent can easily complete the human goals to achieve significantly higher success rate with the coherent instruction and the explicit interaction location. 

## **3.2. Instruction Completer** 

To recover the missing action descriptions in the sparse human instruction, we employ LLMs with rich commonsense to reason the thought chain in the instruction. While LLMs have demonstrated remarkable abilities in various tasks, unlocking their complete potential requires prompt engineering. To enable the pre-trained LLMs to predict the missing actions with interacted objects accurately, we carefully design the prompt by organizing the system message and agent message that respectively describe the innate world properties and the perceived information. The input and output of the instruction completer are shown in Figure 3 with an example. 

**System Message:** The system message describes the innate world properties in the simulator including the role explanation, message definition, primitive actions and response format, which remains unchanged during the whole EIF 

process for the given task and environment. The role explanation specifies the simulation scenario of household AI assistant for LLM response, and the message definition explains the meaning of each input message. Primitive actions constrain the available selection for agent in manipulation and navigation, while response format limits the response structure for LLMs to generate the missing actions with interacted objects. 

**Agent Message:** The agent message demonstrates the perceived information including the human instruction, task completion process, and observed objects in the scene, which is updated along with the EIF process. The human instruction means the given sparse step-by-step instruction sentences and the final goal. The task completion depicts completed subgoals, the current subgoal and all subgoals, which reflects the execution process of the final goal. The information on observed objects in the scene is represented by the object category and the coordination of the instances that have been seen by the agent, which demonstrates the affordable candidates for interaction during the EIF process. Since providing a large number of candidate objects causes severe hallucination, we only leverage objects that are possible to appear in the current room type as candidate objects. Finally, the system message and the agent message are concatenated to form the prompt for recovering missing actions with interacted objects in the sparse human instruction. **Thought Chain Reasoning:** To accurately recover the missing actions in sparse human instruction, we enforce LLMs to reason the thought chain in human instruction that indicates the detailed process from the initial state to the final goal. However, the recovered actions can only be executed if the agent knows the exact position of the interacted objects. Therefore, we also require LLMs to generate the missing subgoals in a structured format for subsequent location prediction. The recovered subgoals are leveraged in subsequent object localizer for position prediction, so that the agent can manipulate the object to successfully achieve the subgoal for task completion. 

## **3.3. Multimodal Object Localizer** 

Although LLMs can accurately reason the thought chain in sparse human instruction to recover the missing actions with interacted objects, the inferring of object position for agent interaction remains challenging because of the weak spatial localization ability of the language description. To deal with the challenge, we present an object localizer based on multimodal transformers to predict the position of interacted objects based on the recovered instruction and the partially observed semantic map. Figure 4 depicts the framework of the object localizer. The recovered instruction acquired from the instruction completer provides the goal for the agent to interact with, and the partially observed semantic map demonstrates the positions of seen objects. The represen- 

4 

tation of semantic maps and the instruction are leveraged by the decoder to generate the probability heatmap for location prediction of the interacted object. Since the semantic correlation among objects can provide beneficial priors to precisely locate the interacted objects based on the visual perception results, we mine the object correlation graph to strengthen the semantic map representation. In this section, we first detail the encoder design for all inputs in representation extraction, and then illustrate the alignment strategy among different inputs with the multimodal transformer. Finally, we introduce the learning objective and the training pipelines of the object localizer. 

For the instruction encoder, we leverage a pre-trained BERT [12] to extract the instruction features **X** _s_ for the human instruction and the goal object, which are acquired from the prediction results of instruction completer. All layers except the last one are frozen during the object localizer finetuning for high generalizability in understanding instruction for complex tasks. For the map encoder, we employ convolutional neural networks to extract the initial map features **X** _[′] t_[that take the partially observed semantic maps] as the input. Since semantic correlation among objects provides beneficial priors for accurate localization, we mine the object correlation graph to embed the priors into semantic map features for further enhancement of object position prediction. The graph is defined as _G_ = ( **V** _,_ **E** ), where **V** is the set of nodes and **E** are the edges between different nodes. Each node represents a possible object category in the simulator, and the element in the _ith_ row and _jth_ column of **E** indicates the correlation of objects between the _ith_ and the _jth_ classes. While conventional graph convolutional networks use external priors to handcraft a predefined graph structure, we construct the object correlation graph by learning from the semantic map features **X** _[′] t_[to][adapt][to][different][scenes.] The object correlation graph is generated from the semantic map features **X** _[′] t_[by] **[E]** _[t]_[=] _[f]_[(] **[X]** _[′] t_ **[W]** _[e]_[)][,][where] _[f]_[(] _[·]_[)][de-] notes the activation function and **W** _e_ is a learnable matrix for graph generation. The process of learning the object correlation graph can be regarded as encoding the object correlation priors among the categories. We then employ graph convolutional layers to enhance the semantic map representation with the object correlation encoding, and acquire the map features for multimodal alignment according to the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning_images/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning.pdf-0005-02.png)


where **W** _a_ is a learnable weight matrix for message passing to embed the object correlation priors into the map features. 

After the representation extraction for the instruction and the semantic map, we leverage a map-instruction aligner to align the instruction features and the map features for object position prediction. Since the semantic map is updated in an online manner with high frequency during agent navigation, we take the features of the semantic maps as the query. The 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning_images/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning.pdf-0005-05.png)


**----- Start of picture text -----**<br>
Object Correlation Graph<br>Table Fridge<br>Stove<br>Sink Counter<br>Partially-observed  Position<br>Semantic Map Heatmap<br>Instruction :<br>.. turn around and go to  Subgoal :<br>Text Encoder<br>the counter on the right  Open the fridge<br>of the stove..<br>Aligner Decoder<br>Map Encoder<br>Map-instruction<br>**----- End of picture text -----**<br>


Figure 4. The overall pipeline of the multimodal object localizer, which uses recovered instruction and observed semantic map to predict object positions for interaction. The object correlation graph is also learned to strengthen the map features. 

instruction remains unchanged in a subtask, which is utilized as the key and the value features. Specifically, they can be acquired via the following equations: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning_images/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning.pdf-0005-08.png)


where **Q** _s_ , **K** _t_ and **V** _t_ respectively mean the query features from the semantic maps, the key and value features from the instruction. The matrices **W** _q_ , **W** _k_ , **W** _v_ are leveraged to obtain the query, key and value features respectively. The representation **H** _[s] t_[that][aligns][the][visual][information][pro-] vided by semantic maps and the language information in the instruction can be acquired as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning_images/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning.pdf-0005-10.png)


where _d_ means the dimension of the representation **H** _[s] t_[. The] representation is then leveraged to decode the predicted location of the interacted objects. 

In order to train the encoder, the map-instruction aligner, and the decoder in the object localizer, we need to acquire the groundtruth instance position for the interacted objects that are predicted by the instruction completer. We replay expert demonstrations in the training data, and record the masks of interacted objects when a subtask is successfully completed. With the projection from egocentric mask to the top-down view by depth estimation, the groundtruth positions are used to supervise the object localizer to predict the position of interacted objects. The model is trained with a pixel-wise binary cross-entropy loss, where the labels of pixels covered by the interacted objects are set to one and otherwise to zero. 

## **4. Experiments** 

In this section, we first introduce the experiment setup including datasets, baseline methods, evaluation metrics and implementation details. Then we compare our method with 

5 

Table 1. Comparison with the state-of-the-art methods in SR, GC, PLWSR, PLWGC on the test seen and test unseen splits. 

|**Method**|**Test Seen**|**Test Unseen**<br>PLWGC<br>GC<br>PLWSR<br>SR|
|---|---|---|
||PLWGC<br>GC<br>PLWSR<br>SR||
|Seq2seq<br>MOCA<br>E.T.<br>LWIT<br>HITUT<br>ABP|6.27<br>9.42<br>2.02<br>3.98|4.26<br>7.03<br>0.08<br>3.9<br>9.99<br>14.28<br>2.72<br>5.30<br>11.46<br>18.56<br>4.10<br>8.57<br>16.34<br>20.91<br>5.60<br>9.42<br>11.51<br>20.31<br>5.86<br>13.87<br>2.22<br>24.76<br>1.08<br>15.43|
||22.05<br>28.29<br>15.10<br>22.05||
||34.93<br>45.44<br>27.78<br>38.42||
||23.10<br>40.53<br>43.10<br>30.92||
||17.41<br>29.97<br>11.10<br>21.27||
||4.92<br>51.13<br>3.88<br>44.55||
|LLM-Planner<br>FILM<br>LGS-RPA<br>Prompter<br>CPEM<br>Prompter+|-<br>26.77<br>-<br>18.20|-<br>23.37<br>-<br>16.42<br>15.13<br>38.52<br>11.32<br>27.80<br>22.76<br>45.24<br>22.76<br>35.41<br>26.22<br>58.76<br>20.76<br>45.72<br>27.00<br>61.10<br>22.61<br>49.84<br>30.09<br>65.71<br>26.22<br>55.46|
||15.59<br>39.55<br>11.27<br>28.83||
||28.97<br>48.66<br>21.28<br>40.05||
||30.72<br>63.43<br>25.81<br>53.23||
||27.49<br>59.40<br>22.61<br>50.62||
||36.35<br>70.20<br>31.12<br>60.86||
|**ThinkBot**(Ours)|**37.01**<br>**71.64**<br>**32.02**<br>**62.69**|**30.73**<br>**67.75**<br>**26.93**<br>**57.82**|



the state-of-the-art EIF approaches to show the superiority in success rate and efficiency, and conduct an ablation study to verify the effectiveness of the instruction completer and the object localizer. Finally, we also demonstrate the visualization results of our method to depict our intuition. Additional results and case studies are provided in the supplementary material. 

## **4.1. Experimental Setup** 

**Dataset:** For the simulation of EIF tasks, we utilize the well-recognized ALFRED benchmark [33] within the AI2THOR [15] virtual environment. The ALFRED benchmark includes 25,743 trajectory-instruction pairs, covering 7 different task types with varying levels of complexity. The benchmark is divided into five splits including _train_ , _test seen_ , _test unseen_ , _valid seen_ and _valid unseen_ . The ALFRED benchmark poses significant challenges for EIF agents, as it requires them to ground incoherent natural instruction of different granularity into various household tasks that involve long-horizon reasoning plans. 

**Baselines:** We compare our agent, ThinkBot, with previously published state-of-the-art EIF models. The counterparts include end-to-end methods Seq2seq [33], MOCA [36], E.T. [29], LWIT [27], HITUT [48], ABP [13], and modular methods LLM-Planner [38], FILM [22], LGS-RPA [25], Prompter [11], CPEM [14]. We also construct a strong baseline in our experiments, which is a modified version of Prompter [11] denoted as Prompter+. The strong baseline combines environment-aware memory [14] and a re-trained object detector [41] with the vanilla Prompter. 

**Evaluation Metrics:** We follow the evaluation protocol outlined in the ALFRED benchmark. The primary metric is the success rate (SR) that measures the percentage of tasks completed, and we also report the goal-condition success rate (GC), which evaluates the percentage of satisfied goal conditions for all subgoals in step-by-step instruction. To account for efficiency in task completion, both SR and GC 

are penalized by the length of the execution sequence to compute a path-length-weighted (PLW) score for each metric, which are termed PLWSR and PLWGC respectively. **Implementation Details:** The instruction completer adopts the publicly released GPT-3.5 API GPT-3.5-turbo as the backbone architecture, where we set the generation temperature to 0 for stability enhancement. For prompt design, we leverage emotion prompt [16] and prompt optimization [45] in the system message template to further boost the performance of LLMs. For the multimodal object localizer, we employ a truncated ResNet18 [7] for the map encoder and follow [40] to mine the inter-modal representation interaction with the multimodal transformer. The object correlation graph is learned via graph convolutional layers [6]. Moreover, we collect a dataset of partially-observed semantic maps and target location map pairs by replaying expert trajectories in the simulator for object localizer training, and AdamW optimizer [20] with the initial learning rate 5 _×_ 10 _[−]_[4] and step decay is employed for parameter update. 

## **4.2. Comparison with the State-of-the-Arts** 

In this section, we compare the proposed ThinkBot with the state-of-the-art methods on the ALFRED benchmark. Table 1 illustrates the comparison of SR, GC, PLWGC, and PLWSR on the test splits for both seen and unseen scenarios. Our ThinkBot achieves the best performance on all four metrics in both test seen and test unseen split, outperforming the previous arts including both end-to-end and modular methods by a sizable margin. Compared with CPEM [14], ThinkBot surpasses by 7.98% and 12.07% SR on the test unseen and test seen split, respectively. Besides, Prompter leverages the large language models to infer object cooccurrence probability for semantic search in EIF, and is further enhanced with better visual perception modules to acquire our Prompter+. However, they still suffer from the sparse human instruction with incoherence that usually causes execution failure. On the contrary, our ThinkBot rea- 

6 

Table 2. Comparison of methods combining different proposed techniques, where valid unseen and the selected hard valid unseen splits are used for evaluation. 

|**Method**|**Valid Unseen**|**Valid Unseen**|**Hard Valid Unseen**<br>PLWGC<br>GC<br>PLWSR<br>SR|**Hard Valid Unseen**<br>PLWGC<br>GC<br>PLWSR<br>SR|
|---|---|---|---|---|
||PLWGC<br>GC<br>PLWSR<br>SR||||
|Random<br>Prompter+<br>Groundtruth Location|26.18<br>67.64<br>23.80<br>59.68||0.32<br>5.41<br>0<br>0<br>0.48<br>5.41<br>0<br>0<br>0.79<br>5.41<br>0<br>0||
||29.36<br>72.00<br>26.82<br>64.43||||
||39.71<br>72.75<br>37.01<br>67.97||||
|w/o Instruction Completer<br>w/o Object Localizer|29.09<br>72.38<br>26.43<br>64.92||0.48<br>5.41<br>0<br>0<br>9.29<br>22.41<br>8.11<br>16.22||
||30.24<br>74.37<br>27.87<br>66.99||||
|w/o Object Correlation Graph<br>**ThinkBot**(Ours)|30.41<br>73.89<br>28.14<br>67.36||11.31<br>29.46<br>9.74<br>21.62<br>**11.95**<br>**30.86**<br>**10.26**<br>**22.97**||
||**31.11**<br>**75.30**<br>**28.73**|**67.72**|||




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning_images/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning.pdf-0007-02.png)


**----- Start of picture text -----**<br>
Task :             Place sliced lettuce into a bin. Task :            Examine a box of tissues using the light.<br>Instruction : Take a knife. Cut the lettuce in the fridge. Instruction : Walk to the cabinet. Grab tissues from the bottom right cabinet.<br> <br>Pickup Knife Slice Lettuce Slice Lettuce Go to Cabinet Pickup Tissue Pickup Tissue<br>     <br>Pickup Knife Open Fridge Slice Lettuce Go to Cabinet Open Cabinet Pickup Tissue<br>**----- End of picture text -----**<br>


Figure 5. Visualization of the agent action sequence acquired by Prompter+ (top) and our ThinkBot (bottom), where our method can recover the missing actions with interacted instances ‘Open Fridge’ and ‘Open Cabinet’ to successfully achieve the human goal. 

sons the thought chain in the sparse human instruction to recover the missing action descriptions, which can provide coherent instruction for the agent to successfully complete the human goal with high efficiency. As a result, our method outperforms the second-best Prompter+ method by 1.83% (62.69% vs. 60.86%) and 1.44% (71.64% vs. 70.20%) in SR and GC respectively in the test seen split. Moreover, the advantages of our method are more obvious in the test unseen split, which are 2.36% (57.82% vs. 55.46%) in SR and 2.04% (67.75% vs. 65.71%) in GC. The results indicate the high generalization ability of our ThinkBot even in novel scenarios that are never seen in training data. In terms of the efficiency in task completion, our method achieves 32 _._ 02% PLWSR and 37 _._ 01% PLWGC in the test seen split, and 26 _._ 93% PLWSR and 30 _._ 73% PLWGC in the test unseen split, which demonstrates the efficiency of our method. In conclusion, our ThinkBot agent is more practical than statof-the-art methods for EIF tasks in scenarios with complex 

environments and long sequences. 

## **4.3. Ablation Study** 

Our instruction completer recovers the missing actions with interacted objects by reasoning thought chain in sparse human instruction, and the object localizer predicts the position of the interacted objects. In Table 2, we conduct an ablation study to verify the effectiveness of each presented technique. Since recovering the incoherent instruction is especially beneficial in the hard cases where target objects are located inside closed containers (e.g. The task _‘to get a chilled mug for coffee’_ where the mug is in a closed fridge), we also evaluate different methods in the hard cases extracted from valid unseen split, referred to as _hard valid unseen_ . In this subset, the agent must open the receptacles to locate the target objects by recovering the missing interaction with the receptacles from sparse instruction, and then the challenging tasks in the hard cases can be successfully 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning_images/iclr_2025_ThinkBot_Embodied_Instruction_Following_with_Thought_Chain_Reasoning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Instruction : Walk forward and left to face the desk.<br>Recovered Object : Pencil<br>Semantic Map Prediction Ground-truth<br>Instruction : Pick up tomato from the back of the counter.<br>Recovered Object : Tomato<br>Semantic Map Prediction Ground-truth<br>**----- End of picture text -----**<br>


Figure 6. The visualization of the predicted and groundtruth positions of interacted objects, where the partially observed semantic maps are also depicted. 

## completed. 

**Instruction Completer:** We first implement our ThinkBot without the instruction completer, where we directly predict the location of the target in the original sparse human instruction with the object localizer for interaction. The results in success rate drops 2 _._ 24% compared with our vanilla ThinkBot in the valid unseen split. Directly predicting the location of interacted objects in incoherent instruction usually causes large deviation, because the semantic correlation is weak between the target object and the observed semantic map. In hard cases, the agent is more likely to fail to complete the task due to the missing interaction with the containers, and the invisible target object is hard to discover by the agent. As a result, the performance in SR without instruction completer drops to zero in hard unseen split. On the contrary, our instruction completer can reason intermediate action descriptions to mine the semantic correlation, which provides fine-grained instruction for the agent to achieve human goals. 

**Object Localizer:** We also evaluate the performance of our ThinkBot without the object localizer, where the agent leverages the baseline Prompter method to predict interaction location with the coherent human instruction recovered by our instruction completer. We observe notable performance drops in SR and PLWSR. In the valid unseen split, the performance drops 0 _._ 75% in SR and 0 _._ 86% in PLWSR compared with our vanilla ThinkBot. Meanwhile, we also implement the object localizer without the object correlation graph, the performance is also degraded in both valid unseen and hard valid unseen split, because the object lo- 

calizer is required to predict the position of both the target objects and the corresponding containers. The correlation mining between target objects and the containers can significantly enhance localization accuracy. Besides, we present the results of Prompter+ with random and Prompter’s search policy, which drop significantly in SR and PLWSR. Hence, the agent without the object localizer fails to locate those target objects to achieve the generated intermediate subgoals, which leads to the failure of task completion. We also provide the results for the settings where the groundtruth location of the interacted objects is given. Our ThinkBot achieves a similar success rate, which indicates the precise prediction of the object location. In conclusion, the results verify the necessity of the object localizer for agent interaction in order to achieve human goals. 

## **4.4. Qualitative Analysis** 

**Action Sequence Visualization:** We present two qualitative examples of the generated action sequence in Figure 5 from Prompter+ and our ThinkBot. In the left case, the agent is instructed to _‘Take a knife. Cut the lettuce in the fridge’_ . The results show that the previous agent struggles to complete the task due to the missing ‘open’ action and interacted object ‘fridge’. In contrast, our ThinkBot first reasons the thought chain of human instruction, and then recovers the missing ‘open’ action and interacted object ‘fridge’ from the instruction, thus successfully completes the task. In the right case, ThinkBot not only recovers the missing ‘open’ action and interacted object ‘cabinet’, but also interacts with the right cabinet that contains the tissue box. The case study demonstrates the effectiveness of ThinkBot in recovering the missing actions and interacted objects from sparse human instruction. 

**Visualization of the Object Localizer:** Figure 6 shows the groundtruth and the predicted location of the interacted objects from the object localizer, where the object category is generated from the upstream instruction completer. In the top case, the interacted object ‘pencil’ is located on the table, and the object localizer predicts the position of the interacted object with negligible deviation. In the bottom case where multiple instances exist for the recovered object, the object localizer is able to locate all objects in the same category. Meanwhile, our ThinkBot can correctly assign the largest probability to the instance that is described by the instruction, and the human goals can be successfully achieved following the step-by-step human instruction. 

## **5. Conclusion** 

In this paper, we have presented a ThinkBot agent that reasons the thought chain for missing instruction recovery in EIF tasks. We design an instruction completer to predict the intermediate actions with interacted objects between incoherent human instruction, and then leverage a multi- 

8 

modal transformer to infer the interaction location for the agent. Extensive experiments in a wide variety of EIF tasks demonstrate the superiority of our method regarding the success rate and execution efficiency. 

## **References** 

- [1] Michael Ahn et al. Do as i can, not as i say: Grounding language in robotic affordances. In _CoRL_ , 2022. 3 

- [2] Valts Blukis, Chris Paxton, Dieter Fox, Animesh Garg, and Yoav Artzi. A persistent spatial semantic representation for high-level natural language instruction execution. In _CoRL_ , 2022. 1, 3 

- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _NeurIPS_ , 2020. 1, 3 

- [4] Peihao Chen, Dongyu Ji, Kunyang Lin, Runhao Zeng, Thomas Li, Mingkui Tan, and Chuang Gan. Weaklysupervised multi-granularity map learning for vision-andlanguage navigation. _NeurIPS_ , 2022. 3 

- [5] Mingyu Ding, Yan Xu, Zhenfang Chen, David Daniel Cox, Ping Luo, Joshua B Tenenbaum, and Chuang Gan. Embodied concept learner: Self-supervised learning of concepts and mapping through instruction following. In _CoRL_ , 2022. 1, 3 

- [6] Heming Du, Xin Yu, and Liang Zheng. Learning object relation graph and tentative policy for visual navigation. In _ECCV_ , 2020. 6 

- [7] Georgios Georgakis, Karl Schmeckpeper, Karan Wanchoo, Soham Dan, Eleni Miltsakaki, Dan Roth, and Kostas Daniilidis. Cross-modal map learning for vision and language navigation. In _CVPR_ , 2022. 3, 6 

- [8] Daniel Gordon, Aniruddha Kembhavi, Mohammad Rastegari, Joseph Redmon, Dieter Fox, and Ali Farhadi. Iqa: Visual question answering in interactive environments. In _CVPR_ , 2018. 1 

- [9] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In _ICML_ , 2022. 3 

- [10] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. In _CoRL_ , 2022. 3 

- [11] Yuki Inoue and Hiroki Ohashi. Prompter: Utilizing large language model prompting for a data efficient embodied instruction following. _arXiv:2211.03267_ , 2022. 1, 3, 6 

- [12] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In _NAACL-HLT_ , page 2, 2019. 3, 5 

- [13] Byeonghwi Kim, Suvaansh Bhambri, Kunal Pratap Singh, Roozbeh Mottaghi, and Jonghyun Choi. Agent with the big picture: Perceiving surroundings for interactive instruction following. In _Embodied AI Workshop @ CVPR_ , 2021. 1, 2, 6 

- [14] Byeonghwi Kim, Jinyeon Kim, Yuyeong Kim, Cheolhong Min, and Jonghyun Choi. Context-aware planning and environment-aware memory for instruction following embodied agents. In _ICCV_ , 2023. 1, 3, 6 

- [15] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. _arXiv:1712.05474_ , 2017. 6 

- [16] Cheng Li, Jindong Wang, Kaijie Zhu, Yixuan Zhang, Wenxin Hou, Jianxun Lian, and Xing Xie. Emotionprompt: Leveraging psychology for large language models enhancement via emotional stimulus. _arXiv:2307.11760_ , 2023. 6 

- [17] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv:2301.12597_ , 2023. 1 

- [18] Xiaotian Liu, Hector Palacios, Muise, and Christian. A planning based neural-symbolic approach for embodied instruction following. In _Embodied AI Workshop @ CVPR_ , 2022. 1, 3 

- [19] Yuxing Long, Xiaoqi Li, Wenzhe Cai, and Hao Dong. Discuss before moving: Visual language navigation via multiexpert discussions. _arXiv:2309.11382_ , 2023. 3 

- [20] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv:1711.05101_ , 2017. 6 

- [21] Yujie Lu, Weixi Feng, Wanrong Zhu, Wenda Xu, Xin Eric Wang, Miguel Eckstein, and William Yang Wang. Neurosymbolic procedural planning with commonsense prompting. _arXiv:2206.02928_ , 2022. 3 

- [22] So Yeon Min, Devendra Singh Chaplot, Pradeep Kumar Ravikumar, Yonatan Bisk, and Ruslan Salakhutdinov. FILM: Following instructions in language with modular methods. In _ICLR_ , 2022. 1, 3, 6 

- [23] Dipendra Misra, John Langford, and Yoav Artzi. Mapping instructions and visual observations to actions with reinforcement learning. In _EMNLP_ , 2017. 1 

- [24] Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo. Embodiedgpt: Vision-language pre-training via embodied chain of thought. _arXiv:2305.15021_ , 2023. 3 

- [25] Michael Murray and Maya Cakmak. Following natural language instructions for household tasks with landmark guided search and reinforced pose adjustment. In _RA-L_ , 2022. 1, 3, 6 

- [26] Takayuki Okatani Van-Quang Nguyen and Takayuki Okatani. A hierarchical attention model for action learning from realistic environments and directives. In _EVAL Workshop @ ECCV_ , 2020. 1, 2 

- [27] Van-Quang Nguyen, Masanori Suganuma, Okatani, and Takayuki. Look wide and interpret twice: Improving performance on interactive instruction-following tasks. In _IJCAI_ , 2021. 1, 2, 6 

- [28] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _NeurIPS_ , 2022. 1, 3 

9 

- [29] Alexander Pashevich, Cordelia Schmid, Sun, and Chen. Episodic transformer for vision-and-language navigation. In _ICCV_ , 2021. 1, 2, 6 

- [30] Yanyuan Qiao, Yuankai Qi, Zheng Yu, Jing Liu, and Qi Wu. March in chat: Interactive prompting for remote embodied referring expression. In _ICCV_ , 2023. 3 

- [31] Shreyas Sundara Raman, Vanya Cohen, Eric Rosen, Ifrah Idrees, David Paulius, and Stefanie Tellex. Planning with large language models via corrective re-prompting. In _Foundation Models for Decision Making Workshop @ NeurIPS_ , 2022. 3 

- [32] Dhruv Shah, Bła˙zej Osi´nski, Sergey Levine, et al. Lmnav: Robotic navigation with large pre-trained models of language, vision, and action. In _CoRL_ , 2023. 3 

- [33] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In _CVPR_ , 2020. 1, 2, 6 

- [34] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cˆot´e, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In _ICLR_ , 2021. 3 

- [35] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. Progprompt: Generating situated robot task plans using large language models. In _ICRA_ , 2023. 3 

- [36] Kunal Pratap Singh, Suvaansh Bhambri, Byeonghwi Kim, Roozbeh Mottaghi, and Jonghyun Choi. Factorizing perception and policy for interactive instruction following. In _ICCV_ , 2021. 1, 2, 6 

- [37] Chan Hee Song, Jihyung Kil, Tai-Yu Pan, Brian M. Sadler, Wei-Lun Chao, and Yu Su. One step at a time: Long-horizon vision-and-language navigation with milestones. In _CVPR_ , 2022. 1, 2 

- [38] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In _ICCV_ , 2023. 6 

Chain-of-thought prompting elicits reasoning in large language models. _NeurIPS_ , 2022. 3 

   - [44] Zhenyu Wu, Ziwei Wang, Xiuwei Xu, Jiwen Lu, and Haibin Yan. Embodied task planning with large language models. _arXiv:2307.01848_ , 2023. 3 

   - [45] Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. _arXiv:2309.03409_ , 2023. 6 

   - [46] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. _arXiv:2210.03629_ , 2022. 3 

   - [47] Bangguo Yu, Hamidreza Kasaei, and Ming Cao. L3mvn: Leveraging large language models for visual target navigation. _arXiv:2304.05501_ , 2023. 3 

   - [48] Yichi Zhang and Joyce Chai. Hierarchical task learning from language instructions with unified transformers and self-monitoring. In _Findings of ACL-IJCNLP_ , 2021. 1, 2, 6 

   - [49] Gengze Zhou, Yicong Hong, and Qi Wu. Navgpt: Explicit reasoning in vision-and-language navigation with large language models. _arXiv:2305.16986_ , 2023. 3 

   - [50] Kaiwen Zhou, Kaizhi Zheng, Connor Pryor, Yilin Shen, Hongxia Jin, Lise Getoor, and Xin Eric Wang. Esc: Exploration with soft commonsense constraints for zero-shot object navigation. _arXiv:2301.13166_ , 2023. 3 

   - [51] Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, et al. Ghost in the minecraft: Generally capable agents for open-world enviroments via large language models with text-based knowledge and memory. _arXiv:2305.17144_ , 2023. 3 

   - [52] Yuke Zhu, Daniel Gordon, Eric Kolve, Dieter Fox, Li FeiFei, Abhinav Gupta, Roozbeh Mottaghi, and Ali Farhadi. Visual semantic planning using deep successor representations. In _ICCV_ , 2017. 1 

- [39] Alessandro Suglia, Qiaozi Gao, Jesse Thomason, Govind Thattai, and Gaurav Sukhatme. Embodied bert: A transformer model for embodied, language-guided visual task completion. _arXiv:2108.04927_ , 2021. 1, 2 

- [40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NeurIPS_ , 2017. 6 

- [41] Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, et al. Internimage: Exploring large-scale vision foundation models with deformable convolutions. In _CVPR_ , 2023. 1, 6 

- [42] Zihao Wang, Shaofei Cai, Anji Liu, Xiaojian Ma, and Yitao Liang. Describe, explain, plan and select: Interactive planning with large language models enables open-world multitask agents. _NeurIPS_ , 2023. 3 

- [43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 

10 

