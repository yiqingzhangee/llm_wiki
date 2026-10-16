## _**lil**_ **Gym: Natural Language Visual Reasoning with Reinforcement Learning** 

## **Anne Wu, Kianté Brantley, Noriyuki Kojima, and Yoav Artzi** 

Department of Computer Science and Cornell Tech, Cornell University {aw588, kdb82, nk654}@cornell.edu, {yoav}@cs.cornell.edu 

## **Abstract** 

We present _lil_ Gym, a new benchmark for language-conditioned reinforcement learning in visual environments. _lil_ Gym is based on 2,661 highly-compositional human-written natural language statements grounded in an interactive visual environment. We introduce a new approach for exact reward computation in every possible world state by annotating all statements with executable Python programs. Each statement is paired with multiple start states and reward functions to form thousands of distinct Markov Decision Processes of varying difficulty. We experiment with _lil_ Gym with different models and learning regimes. Our results and analysis show that while existing methods are able to achieve non-trivial performance, _lil_ Gym forms a challenging open problem. _lil_ Gym is available at https://lil.nlp.cornell.edu/lilgym/. 

## **1 Introduction** 

The ability to reason about natural language has the potential to transform how reinforcement learning (RL) is used to train grounded agents. Language provides an expressive and accessible conduit for task specification, enabling agents to address a broad set of tasks, rather than to learn singlebehavior policies. At the same time, RL is a promising framework to study core grounded language learning problems within interactive environments. 

Prerequisite to realizing this potential are expressive benchmark environments, as has been instrumental for progress in RL more broadly. However, natural language poses unique challenges to building such benchmarks. Beyond the design of the environment itself, which must support rich linguistic reasoning, accurate reward computation requires resolving language semantics. Existing approaches adopt different strategies to address this issue, most often by using synthetic language (e.g., Côté et al., 2018; Co-Reyes et al., 2019), or by removing the 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0001-09.png)


**----- Start of picture text -----**<br>
There are two boxes which has the same number<br>of objects and three kinds of colors.<br>REMOVE(x=142, y=41)<br>ADD(x=142, y=55,   )<br>ADD(x=288, y=42,   )<br>ADD(x=357, y=63,  )<br>**----- End of picture text -----**<br>


Figure 1: An illustrative example of _lil_ Gym. Given a natural language statement with a target truth value, True in this example (omitted from the figure), and an initial image, the agent needs to manipulate objects in the environment so that the truth value of the statement with respect to the image is the target truth value. 

language problem from reward computation by restricting to a single goal state (e.g., Anderson et al., 2018; Chen et al., 2019). While these approaches open new research avenues, both have significant drawbacks. The simplifications of synthetic language limit the relevance of methods and results to the complexities of human language. Single-goal formulations forgo or restrict language’s ability to efficiently abstract over many possible solutions, a core argument for its potential to RL. 

We present _lil_ Gym, an RL benchmark where an agent manipulates a visual environment by adding and removing objects conditioned on a natural language statement. The agent’s goal is to modify the environment so that a given statement will have a pre-specified truth-value with regard to the environment (i.e., the constraints specified in the language are either satisfied or violated). Figure 1 illustrates the scenario. _lil_ Gym includes highlycompositional and semantically-diverse natural language statements and visual environments from the Natural Language for Visual Reasoning (NLVR) 

corpus (Suhr et al., 2017), that combine with a configurable environment backbone to create thousands of Markov Decision Processes (MDP) of varying complexity. 

A key challenge in constructing _lil_ Gym is accurate reward computation. Because of the flexibility of the environments and language, there are many possible equally correct termination states (i.e., goals) for each MDP. Correct reward computation at every possible state requires taking into account the semantic nuances of the highly-compositional language. We address this by annotating all statements with executable Python programs representing their meaning, effectively creating a supervised semantic parsing corpus (e.g., Zelle and Mooney, 1996; Zettlemoyer and Collins, 2005). The executable programs allow for exact and reliable reward computation at every possible state. 

Our experiments with _lil_ Gym show that existing models demonstrate non-trivial performance given sufficient training time, with multi-modal pre-training providing further benefit. However, there remains significant room for improvement. For example, on the simplest configuration, our agent can solve 76.23% of the test environments, but performance drops significantly to only 16.81% on the most complex configuration. Our experiments also confirm the importance of modeling language meaning for reward computation in learning. The _lil_ Gym benchmark and trained models are available under the MIT license at https: //lil.nlp.cornell.edu/lilgym/. 

## **2 Related Work** 

RL research has benefited greatly from benchmarks such as Atari (Bellemare et al., 2013) and MuJoCo (Todorov et al., 2012). Although both are synthetic with limited potential to train policies directly applicable to realistic environments, their accessibility and focus on core RL problems have made them impactful drivers of algorithm development. For example, the first demonstration of an effective neural network policy (Mnih et al., 2013) and the development of proximal policy optimization (PPO; Schulman et al., 2017) both used these benchmarks, and both were later shown to generalize to more complex scenarios. _lil_ Gym is inspired by these benchmarks. Rather than aiming for training models that transfer to realistic domains, it aims to enable fundamental algorithmic development by emphasizing semantic diversity and compositional- 

ity within an accessible benchmark. 

There is significant interest in RL for training policies conditioned on natural language. This includes multiple efforts at developing RL environments with language-informed tasks, mainly via grounded language learning in 2D or 3D environments or using text games. Oftentimes, using synthetic language (Narasimhan et al., 2015; Johnson et al., 2017; Chevalier-Boisvert et al., 2019; Cao et al., 2020; Côté et al., 2018; Urbanek et al., 2019; Hermann et al., 2017; Co-Reyes et al., 2019; Hausknecht et al., 2020; Jiang et al., 2020). Although synthetic language allows studying the problem of learning high-level concepts, many of the complexities of natural language may be stripped away, and such approaches run the risk of reducing the language learning challenge to reverse engineering the hand-crafted generation process. In contrast, _lil_ Gym is based on semantically-diverse human-written language grounded in a visual environment, and requires both the ability of reasoning over highly-compositional language including sets and spatial relations, and precise alignment between statements and states. 

Another approach is to simplify the task so only a few annotated termination states or trajectories are correct (Anderson et al., 2018; Chen et al., 2019; Ku et al., 2020). This forgoes much of the abstractive potential of natural language, where it can succinctly define extremely large set of states. Thereby reducing the utility of language, and not exposing learning algorithms to some of the core challenges it introduces to the RL problem. _lil_ Gym does not adopt such simplifications. Our experiments show the importance of considering language meaning for reward computation when there is a large set of valid goal states. 

Alternatively, other benchmarks are created by first generating the target sequence of decisions (i.e., task demonstration), and then soliciting posthoc instructional language (Shridhar et al., 2020, 2021; Hanjie et al., 2021). This process uses human-written language, but retains the regularities of the demonstration generation procedure. _lil_ Gym uses language from NLVR, which was crowdsourced via a contrastive task that was shown to elicit high semantic diversity. 

## **3 Background: the NLVR Corpus** 

_lil_ Gym uses data from the NLVR corpus (Suhr et al., 2017). NLVR was initially created as a su- 

pervised learning benchmark. We formalize an interactive task using the NLVR data and collect additional annotations for reward computation. 

NLVR includes human-written natural language statements paired with synthetic images. Each pair is annotated with the boolean truth-value of the statement with regard to the image (i.e., True if the statement is true with regard to the image, or False otherwise). The images are designed to support complex reasoning, including about spatial and set relations. The original learning task posed by NLVR is to classify statement-image pairs as True to indicate the statement is true with regard to the image, or False otherwise. NLVR has been studied extensively (Suhr et al., 2017; Tan and Bansal, 2018; Goldman et al., 2018; Pavez et al., 2018; Yao et al., 2018; Hudson and Manning, 2018; Perez et al., 2018; Dasigi et al., 2019; Zheng et al., 2020; Gupta et al., 2021), and a separate version using photos was also released (Suhr et al., 2019).[1] 

Qualitative analysis of the NLVR data (Table 2 in Suhr et al., 2017) showed it to provide diverse representation of semantic and compositional phenomena, including requiring joint visual-linguistic reasoning about spatial relations, quantities, and set relations. NLVR also provides an underlying structured representation for every image, which supports image manipulation. The combination of an interface for image manipulation with complex reasoning via natural language makes NLVR ideal to support an interactive benchmark environment. 

## **4 The** _**lil**_ **Gym Benchmark** 

_lil_ Gym consists of a collection of environments that share a common backbone. The backbone is a 2D plane that is manipulated by placing and removing objects of different types. Each environment instance is a Markov Decision Process (MDP) created by pairing a natural language statement and a target boolean value with a configuration of the shared backbone. The goal of the agent is to manipulate the environment by adding and removing objects so that the truth-value of the statement with regard to the environment is the target boolean. 

The learning problem _lil_ Gym presents is to induce a policy that generalizes across MDPs. We split the MDPs to training, development, and heldout testing sets. The training environments are for parameter estimation, while the two other sets are 

1We do not use the photographic NLVR2 in this work. 

for testing during development and for final heldout testing to report approach performance.[2] 

There are two dimensions of configuration: appearance and starting condition. The appearance determines the state space, transition function, and action space. The appearance of the environment can be (a) TOWER: the objects include squares only, and they can be stacked into towers in specific positions only; or (b) SCATTER: objects of different types can be freely distributed. The two leftmost examples in Figure 2 are from TOWER, and the two rightmost are from SCATTER. TOWER gives a more constrained problem with much smaller state and action spaces compared to SCATTER. 

There are two starting conditions, which also determine the agent’s goal: (a) SCRATCH: the environment starts without any objects and the goal is to modify it so that the statement’s truth-value is True; or (b) FLIPIT: the environment starts with a set of objects and the agent’s goal is to flip the truth-value of the statement, by modifying the environment. The first row of images in Figure 2 shows start states in both conditions. SCRATCH generally only requires adding objects, except in cases of correcting for agent’s errors, while FLIPIT requires both adding and removing, because there are already objects present. 

The four configurations are TOWER-SCRATCH , TOWER-FLIPIT , SCATTER-SCRATCH , and SCATTER-FLIPIT . In our experiments (Section 6), we observe the different configurations provide different levels of difficulty. For example, SCATTER configurations are generally harder than TOWER, due to the larger state and action spaces. 

Formally, each configuration is a Contextual Markov Decision Process (CMDP; Hallak et al., 2015). CMDP is an abstraction over a set of MDPs to account for a context that remains constant throughout an interaction with an MDP. The context includes the statement and the target boolean the interaction is conditioned on. A CMDP is a tuple ( _C, S, A, M_ ( _c_ )), where _C_ is the context space, _S_ the state space, _A_ the action space, and _M_ a function mapping a context _c ∈C_ to an MDP _M_ ( _c_ ) = ( _S, A, T, R[c] , β[c]_ ). Here, _T_ : _S × A →S_ is a transition function, _R[c]_ : _S × A →_ R a reward function, and _β[c]_ an initial state distribution. This means that a CMDP is a set of MDPs that share the same states and actions. The policy learning 

2We recommend reporting both development and held-out test results in future work for easy comparison. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
TOWER-SCRATCH TOWER-FLIPIT SCATTER-SCRATCH SCATTER-FLIPIT<br>One tower has exactly 1 There is no black block as There is a box with 2<br>None of the yellow triangles<br>black block and 1 yellow the top of a tower with triangles of same color<br>are touching the edge<br>c block at most three blocks. nearly touching each other.<br>Starting label: False Starting label: False Starting label: False Starting label: True<br>Target label: True Target label: True Target label: True Target label: False<br>s 0<br>ar 00 ∼∼ Rπ ( [c] s (0 s, c 0 , a ) 0) a MIDDLEYELLOW 0 =  ADD , )( a RIGHT 0 =  REMOVE ) ( a xTRIANGLEYELLOW0 0 ,  y =  ADD0 , ,  LARGE ( , ) a xTRIANGLEYELLOW0 0 ,  y =  ADD0 , ,  MEDIUM ( , )<br>s 1<br>ar 11 ∼∼ Rπ ( [c] s (1 s, c 1 , a ) 1) a MIDDLEBLACK 1 =  ADD ) , ( a 1 =  STOP () a xTRIANGLEYELLOW1 1 ,  y =  ADD1 , ,  SMALL ( , ) a 1 =  STOP ()<br>s 2<br>ar 22 ∼∼ Rπ ( [c] s (2 s, c 2 , a ) 2) a 2 =  STOP () a 2 =  STOP ()<br>s 3<br>**----- End of picture text -----**<br>


Figure 2: Examples from the four CMDP configurations. Each example is conditioned on a context _c_ = (¯ _x, b_ ), and starts with a state _s_ 0, sampled from the initial state distribution _β[c]_ . For example, for TOWER-SCRATCH (left column), the context _c_ pairs the statement _one tower has exactly 1 black block and 1 yellow block_ with the target boolean True. The initial state _s_ 0 is an image with three empty grey box regions separated by darker grey separators. The agent _π_ is given ( _s_ 0 _, c_ ), and samples an action _a_ 0 _∼ π_ ( _s_ 0 _, c_ ). The environment transitions to the next state _s_ 1, while the context remains the same. This process continues until the agent selects the `STOP` action. 

problem is to estimate parameters _θ_ of a policy _πθ_ : _S × C →A_ , which maps the current state and the context underlying the MDP to an action. The policy must generalize across different contexts from _C_ . Figure 2 shows example action trajectories in MDPs for each of the four CMDPs. Table 1 shows the number of MDPs in each configuration.[3] 

**Contexts** _C_ A context _c ∈C_ is a pair _c_ = (¯ _x, b_ ), where _x_ ¯ is a natural language statement and _b ∈ {_ True _,_ False _}_ is a target boolean value for the statement _x_ ¯ with respect to the state _s_ . The set of statements is predefined for TOWER and SCATTER based on the NLVR data, but identical across the choice of SCRATCH and FLIPIT. The target boolean value in SCRATCH is always True. In FLIPIT, the target boolean value is either True or False. Depending on the context, different types of reasoning are required. For example, in the second column 

> 3 NLVR includes 18,322 images. This allows further expanding the number of initial states to 92,179 initial states through box element permutations. We do not manipulate this property in this work, but future work could take advantage of it. Our reward computation is invariant to such permutations. 

of Figure 2, the statement _there is no black block as the top of a tower with at most three blocks_ requires reasoning about negation, soft cardinality, color, and position, while the statement in the third column _there is a box with 2 triangles of same color nearly touching each other_ requires a comparison and to reason about several object attributes (shape, color, position). Both require high-level relational reasoning about single objects or sets. 

**States** _S_ A state _s ∈S_ is an RGB image. Images in _lil_ Gym are divided into three box regions of identical dimensions by two dark gray separators (Figure 2). The objects in _lil_ Gym have three properties, each can take multiple values: shape ( `CIRCLE` , `SQUARE` , or `TRIANGLE` ), color ( `BLACK` , `BLUE` , or `YELLOW` ), and size ( `SMALL` , `MEDIUM` , or `LARGE` ). In TOWER, states are constrained to have stacks of up to four `SQUARE` s of `MEDIUM` size and any color at the center of each box. SCATTER states support all object shapes, sizes, and colors, and they may be positioned freely. In both conditions, objects cannot cross image boundaries or into the separators. 

||**TOWER-SCRATCH**<br>MDPs|**TOWER-FLIPIT**<br>MDPs<br>Init.|**SCATTER-SCRATCH**<br>MDPs|**SCATTER-FLIPIT**|
|---|---|---|---|---|
|||||MDPs<br>Init.|
|Train<br>Dev<br>Test|989<br>163<br>324|1,910<br>5,704<br>317<br>676<br>619<br>1,383|1,241<br>87<br>155|2,340<br>6,696<br>164<br>313<br>285<br>591|
|Total|1,476|2,846<br>7,763|1,483|2,789<br>7,600|



Table 1: Data statistics per CMDP configuration and data split. The number of MDPs corresponds to the number of contexts under each CMDP. For FLIPIT, “Init.” corresponds to the total number of initial states across all MDPs for this CMDP.[3] The number of initial states and MDPs is equal for SCRATCH CMDPs. 

The choice of starting condition between SCRATCH or FLIPIT does not influence the state space. 

**Actions** _A_ **and Transitions** _T_ There are three action types `STOP` , `ADD` , and `REMOVE` . `STOP` terminates the episode. The truth-value of the statement is only evaluated and compared to the target boolean after the `STOP` action is taken. `ADD` adds objects to the environment, and `REMOVE` removes objects. `ADD` and `REMOVE` take arguments that differ between TOWER and SCATTER: 

**TOWER:** Both `ADD` and `REMOVE` take a position argument, which has three possible values corresponding to the three box regions. Objects are added or removed at the top of the stack. Adding an object on top of a stack of four objects or removing an object from an empty box are both invalid actions. `ADD` also takes a color argument. For example, the first action on the left trajectory in Figure 2 is adding a yellow square in an empty box. Including `STOP` , there are 1 + (3 + 1) _×_ 3 = 13 actions. 

**SCATTER:** Unlike TOWER, objects of any type can be placed freely in the box regions. Both `ADD` and `REMOVE` take 2D coordinates that specify pixel location. Adding an object places it so that its top-left coordinates are the given coordinates. Removing an object will remove the object at the given coordinates. Adding also requires specifying the shape, color, and size. The action is invalid if adding results in objects’ overlap or boundary crossing with the separators or image boundaries. Removing from a position that does not include an object is also an invalid action. The native resolution of images in _lil_ Gym is 380 _×_ 100 pixels. Including `STOP` , there are 1 + (380 _×_ 100) _×_ ((3 _×_ 3 _×_ 3) + 1) = 1 _,_ 064 _,_ 001 actions. Because of the extremely large action space, _lil_ Gym also supports a simplification through a coarser grid system for SCATTER that is automatically mapped to the original resolution (Appendix A). The grid simplification includes heuristics that assist in identifying locations 

to place objects in the original pixel space or objects to remove once a grid cell is selected. In our experiments (Section 6), we use a grid simplification of 19 _×_ 5, giving a total of 2,661 actions. The difficulty of SCATTER can be adjusted by modifying the grid size, or acting at the original resolution. 

The transition function _T_ : _S × A →S_ depends on the choice between TOWER and SCATTER configurations, because this choice determines the action space. Similar to the action space, the transitions in TOWER are more constrained compared to SCATTER. The transition function does not modify the context, which is fixed for a given MDP. 

**Reward Function** _R[c]_ The reward function _R[c]_ is computed with respect to the context _c_ = (¯ _x, b_ ), and is based on evaluating the truth-value of the natural language statement _x_ ¯ with respect to a state _s_ , and comparing it to the target boolean _b_ . _lil_ Gym includes an evaluation function _E[x]_[¯] : _S × A → {_ True _,_ False _}_ for every statement _x_ ¯. Section 5 describes how we create the evaluation functions. 

The agent receives a positive reward for terminating the episode using the `STOP` action if the evaluation _E[x]_[¯] ( _s_ ) is equal to the target boolean _b_ . If _E[x]_[¯] ( _s_ ) does not equal _b_ when taking the `STOP` action, the agent receives a negative reward. If the episode terminates because the current time step _t_ reached the action horizon _H_ or because of an invalid action, the agent also receives a negative reward. Action validity depends on the current state _s_ and on the configuration, because TOWER and SCATTER have different action spaces. For example, in TOWER, adding an object to a box (e.g., `ADD` ( `MIDDLE` _,_ `BLUE` )) is only valid if the box has less than four objects, because towers have a maximum height of four. There is also a verbosity penalty of _δ_ . Formally, the reward is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0006-00.png)


**Initial State Distribution** _β[c]_ The initial state distribution _β[c]_ is parameterized by the context _c ∈ C_ , and differs between SCRATCH and FLIPIT. In SCRATCH, the agent modifies an empty environment to satisfy the truth-condition of the statement _x_ ¯ in the context _c_ , so the initial state _s_ 0 is always an empty image. The set of initial states _β[c]_ for every context _c ∈C_ is the set of images associated with the statement ¯ _x_ in the NLVR data. This set includes between 1 to 43 images. Table 1 shows the total number of initial states in each configuration. 

## **5 The** _**lil**_ **Gym Data and Annotation** 

We use the NLVR data to create each of the CMDPs (Table 1). SCRATCH CMDPs include contexts for all natural language statements from NLVR, each paired with the empty initial state containing no shapes (Figure 2, left and center-right columns). FLIPIT CMDPs include the natural language statements with their corresponding images, both from NLVR. The images are used as initial states. The target boolean is set so that the initial state does not fulfil it. The split between TOWER and SCATTER also follows from NLVR. Statements corresponding to TOWER images in NLVR are included in our TOWER CMDPs, and the same for SCATTER sentences. 

NLVR has four splits for training, development, public testing, and hidden testing. We adopt the original training and development sets splits. Following the recent public release of the hidden testing set, we merge the public and hidden testing sets into a single public test split. 

The NLVR annotations include the truth-value of each statement with regard to the images paired with it in the data. Once we manipulate an image (i.e., change the state in our environment), the truthvalue annotation does not necessarily hold. A key challenge for creating an interactive environment using this data is an accurate evaluation of the natural language statement for _every_ possible state (i.e., image) for reward computation (Section 4). We address this by annotating each statement _x_ ¯ with an executable boolean Python program representing its meaning, _E[x]_[¯] in Section 4. This process is inspired by data annotation for supervised semantic parsing (Zelle and Mooney, 1996; Zettlemoyer and 

Collins, 2005; Suhr et al., 2018), where sentences are annotated with formal meaning representations. 

The Python programs operate on the underlying structured representation. Each program returns True for every image that satisfies the constraints specified in the corresponding statement, and False otherwise. In general, there are many states that satisfy any given statement, many more than provided with the original NLVR images. 

The programs are written using an API defined over the structured representations. We base the API design on the ontology designed for NLVR’s structured representations by Goldman et al. (2018), which we extend to include 66 functions. Figure 7 in Appendix B shows two example programs with their corresponding statements. 

We use the freelancing platform Upwork[4] for annotation. We recruit three programmers based on preliminary screening of their fluency in English and competency in Python. We de-duplicate the naturally occurring sentences in the data, and distribute sentences to annotators randomly, each with a single example NLVR image. Each program is evaluated against a corresponding hidden validation set made of all remaining NLVR images paired with the sentence, and must pass all the tests. Appendix B provides a screenshot of the interface and more details. We collect 2,666 annotations at a total cost of $3,756, and keep 2,661 valid annotations. 

## **6 Experiments** 

## **6.1 Methods** 

We experiment with each of the four CMDPs separately, training on the training split and testing on the development and test splits. We sample a validation set from the training split for model selection. For SCATTER we use a simplified grid action space of 19 _×_ 5 (Section 4). Each grid cell is 20 _×_ 20 pixels. We set the action horizon _H_ = 12. Appendix C provides implementation details. 

We use PPO (Schulman et al., 2017) for parameter estimation,[5] with a separate network as a critic. The critic network is identical to the policy, except that we add a tanh activation for the value output. Because of the large action space, especially for SCATTER, the agent rarely observes positive reward, which requires taking a `STOP` action at an appropriate state. We design a simple variant of PPO called 

> 4https://www.upwork.com 

> 5We use the PPO implementation of (Kostrikov, 2018). 

PPO+SF (PPO with stop forcing) to study this issue. PPO+SF is identical to PPO, except that during training, we mask all actions except `STOP` when the agent reaches a state where selecting `STOP` will give a positive reward. This modification is present only during training. All testing is done under the same conditions, without stop forcing. 

We also study the importance of our annotation for reward computation using a reward function for SCRATCH that does not require any annotation beyond what is already in NLVR. The reward uses NLVR images that are associated with each statement (between 1–43) and are labeled with the target boolean.[6] Instead of testing the state using a program, it compares the state to the available NLVR images, and only if it equals one of them, the learner receives a positive task completion reward. 

We experiment with three models: C3+BERT, C10+BERT and ViLT.[7] In C3+BERT and C10+BERT, we process the statement _x_ ¯ using BERT (Devlin et al., 2019), and do mean pooling across all layers and tokens to get the statement representation. We use a three-layer CNN (Fukushima and Miyake, 1982) in C3+BERT to embed the image of the current state _s_ , and a ten-layer CNN in C10+BERT. We concatenate the statement and image representations with an embedding for the target boolean _b_ , and use a multi-layer perceptron to compute the action distribution. ViLT is a pretrained multi-modal Transformer that jointly processes text and image inputs (Kim et al., 2021). We create a sequence of tokens by concatenating the statement, a token for the target boolean, and image patches, separated by special tokens. The image patches are the same size as the 19 _×_ 5 grid cells, including in TOWER, where the action space does not use a grid. 

## **6.2 Results and Analysis** 

Table 2 shows task-completion accuracies for all CMDPs, and Figure 3 shows reward statistics. We observe only minor differences between C3+BERT and C10+BERT, so conduct the bulk of our analysis on C3+BERT and ViLT. Figure 4 plots training curves for SCRATCH.[8] Figure 5 breaks down development set accuracies for FLIPIT CMDPs by the 

> 6We dropped 1% of MDPs without a single target state. 7Appendix E describes preliminary experiments with one more model. 

> 8We plot the training curves with no regard to the patience stopping criteria we use for model selection. This better reflects long term trends. 

target boolean, and Figure 6 shows development rollout statistics for PPO.[9] We sample 50 development examples for each CMDP, and annotate them with expert[10] trajectories to estimate the expert reward and rollout statistics. All expert rollouts are successful. 

Overall, we observe stronger task-completion performance (Table 2) on TOWER CMDPs compared to SCATTER, especially with ViLT, which shows stronger performance than C3+BERT and C10+BERT in most cases. The development rewards (Figure 3) and training curves (Figure 4) show similar trends . The training curve comparison to the alternative reward that uses NLVR images instead of our program annotations shows no effective learning. This illustrates the importance of exact reward computation, such as possible with our program annotations. The comparison to the estimate of expert rewards shows there remains significant room for improvement across the board. Even when the learned policies are able to complete the task, they are doing it inefficiently, so much so that the mean rewards are often negative. The mean rewards of the random baseline policy illustrate the task is far from trivial. Both task accuracies and reward statistics indicate TOWER-SCRATCH is the easiest of the CMPDs, and SCATTER-FLIPIT is the hardest. 

The additional guidance of PPO+SF compared to PPO helps with exploration, especially on SCATTER CMDPs. On SCATTER-FLIPIT , PPO+SF improves performance by 13.25% compared to PPO. This illustrates the exploration challenges SCATTER CMDPs pose. 

ViLT generally outperforms C3+BERT and C10+BERT, except on SCATTER-SCRATCH , where the ViLT policy more often selects invalid actions. ViLT general advantage is expected given the joint reasoning architecture and multi-modal pre-training of ViLT. FLIPIT policies generally do better on examples with a False target boolean, except when learning fails (Figure 5). The other direction is harder, because the set of states that invalidates a statement is usually larger than the set that validates it, and it generally requires fewer actions to invalidate a statement. 

We observe more rollouts that are terminated either by reaching the action horizon _H_ or by taking 

> 9Appendix D provides further analysis, including error analysis, performance breakdown by semantic phenomena, and rollout statistics for PPO+SF. 

> 10The expert is an author of this paper. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
TOWER-SCRATCH TOWER-FLIPIT SCATTER-SCRATCH SCATTER-FLIPIT<br>Dev Test Dev Test Dev Test Dev Test<br>C3+BERT 72.80 ± 9.92 68.72 ± 6.60 28.11 ± 6.03 27.84 ± 4.50 59.00 ± 5.42 68.17 ± 2.89 0.00 ± 0.00 0.06 ± 0.10<br>PPO<br>C10+BERT 80.16 ± 1.77 73.66 ± 3.04 30.13 ± 8.22 28.75 ± 7.42 43.68 ± 7.18 50.97 ± 12.95 0.00 ± 0.00 0.00 ± 0.00<br>ViLT 81.19 ± 2.90 76.23 ± 5.58 53.25 ± 4.28 55.19 ± 4.92 40.23 ± 6.99 47.74 ± 11.47 13.31 ± 6.88 16.81 ± 7.43<br>PPO+SF [C3+BERT] 81.80 ± 0.94 76.54 ± 1.85 32.59 ± 3.14 29.55 ± 4.74 72.03 ± 3.32 74.41 ± 3.25 17.04 ± 1.82 18.16 ± 3.07<br>C10+BERT 77.91 ± 2.21 75.62 ± 1.41 37.38 ± 2.02 35.70 ± 2.08 73.18 ± 2.65 77.85 ± 1.62 8.52 ± 2.72 10.55 ± 2.45<br>ViLT 84.05 ± 3.25 81.48 ± 1.93 65.68 ± 9.17 65.51 ± 8.43 67.43 ± 1.35 73.98 ± 0.30 28.01 ± 5.32 30.06 ± 4.51<br>Table 2: Mean task-completion accuracy and standard deviation computed over three runs for all four CMDPs.<br>1.0<br>an invalid action (i.e., without  STOP ) on SCATTER SCATTER<br>0.5<br>CMDPs compared to TOWER (Figure 6, upper left). TOWER (Figure 6, upper left). (Figure 6, upper left).Figure 6, upper left)., upper left).<br>0.0<br>This difference is partially explained by a higher<br>0.5<br>of invalid actions in SCATTER (FigureFigure 6,, up-<br>1.0<br>right), which cause immediate rollout termi-<br>On SCATTER CMDPs, where we have a PPO w/C3+BERT PPO+SF w/C3+BERT Expert (estimate)<br>rate of invalid actions, the type of invalid PPO w/C10+BERT PPO+SF w/C10+BERT Random<br>PPO w/ViLT PPO+SF w/ViLT<br>TOWER-SCRATCH TOWER-FLIPIT SCATTER-SCRATCH SCATTER-FLIPIT<br>**----- End of picture text -----**<br>


Table 2: Mean task-completion accuracy and standard deviation computed over three runs for all four CMDPs. 

an invalid action (i.e., without `STOP` ) on SCATTER SCATTER CMDPs compared to TOWER (Figure 6, upper left). TOWER (Figure 6, upper left). (Figure 6, upper left).Figure 6, upper left)., upper left). This difference is partially explained by a higher rate of invalid actions in SCATTER (FigureFigure 6,, upper right), which cause immediate rollout termination. On SCATTER CMDPs, where we have a higher rate of invalid actions, the type of invalid actions we most often observe for both models are actions hitting one of the separators, except when learning fails completely. This is related to the action selection bias of the models, which tend to select some coordinates more often than others. Section D.2 provides further error analysis for both models for SCATTER CMDPs trained with PPO, and Section D.3 illustrates the action selection bias. 

Figure 3: Mean development set rewards, averaged over three runs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0008-04.png)


**----- Start of picture text -----**<br>
TOWER-SCRATCH SCATTER-SCRATCH<br>100 100<br>50 50<br>0 0<br>0 1 2 3 4 0 1 2 3 4<br>6 6<br>Steps ×10 Steps ×10<br>C3+BERT ViLT<br>C3+BERT w/NLVR reward ViLT w/NLVR reward<br>PPO PPO+SF<br>Accuracy<br>**----- End of picture text -----**<br>


There is no consistent difference in the length of rollouts between the two models (Figure 6, bottom left). Expert trajectory length is similar on TOWER, where models perform fairly well. However, on SCATTER, where our models are weaker, expert trajectories are significantly longer. This is partially explained by the models not learning to effectively avoid invalid actions, which terminate the execution immediately. Using `REMOVE` actions is generally difficult for the learned policies. TOWER-FLIPIT is an exception with `REMOVE` dominating the rollouts (Figure 6, bottom right), potentially because removing objects generally provides a more efficient path to flip the boolean value. While PPO policies generate `REMOVE` actions for SCATTER-FLIPIT , the extremely low performance indicates that these actions are not used effectively. Expert statistics indicate that `REMOVE` actions are beneficial for FLIPIT CMDPs. 

Figure 4: Training curves for SCRATCH, averaged over three runs. All NLVR reward curves are superposed because accuracy remains zero throughout training.[8] 

vides the complete tables with examples, including for PPO+SF. The two models mostly follow similar trends with respect to the categories on which they perform above and below overall performance. Both models perform better than they do overall on hard cardinality (e.g., _...exactly four objects . .._ ) for TOWER CMDPs, and on presupposition for SCRATCH CMDPs. However, on spatial relations, both models perform below overall performance for all CMDPs except TOWER-SCRATCH . 

We also performed semantic and syntactic analyses using the 200 development examples manually annotated by Suhr et al. (2017). Table 3 shows the performance on this data of policies trained with PPO. We only include categories with more than 10 instances across all CMDPs. Appendix D.4 pro- 

## **7 Conclusion** 

We introduce _lil_ Gym, an RL benchmark that focuses on natural language visual reasoning. _lil_ Gym is designed to be accessible for researchers, while 

||**TOWER-SCRATCH**<br>Total<br>Correct %|**TOWER-SCRATCH**<br>Total<br>Correct %|**TOWER-FLIPIT**|**TOWER-FLIPIT**|**SCATTER-SCRATCH**|**SCATTER-SCRATCH**|**SCATTER-FLIPIT**|**SCATTER-FLIPIT**|
|---|---|---|---|---|---|---|---|---|
||||Total<br>Correct %||Total<br>Correct %||Total<br>Correct %||
||||||||||
|Cardinality (hard)<br>Cardinality (soft)<br>Existential<br>Coordination<br>Spatial Relations<br>Presupposition|<br>98<br>**76.5**<br>21<br>68.2<br>122<br>**75.1**<br>19<br>**85.9**<br>94<br>**74.8**<br>17<br>**74.5**|**83.7**<br><br>81.0<br><br>**81.7**<br><br>**84.2**<br><br>**81.6**<br><br>**90.2**|480<br>**28.9**<br>82<br>**30.1**<br>577<br>**29.0**<br>86<br>27.1<br>438<br>26.6<br>74<br>27.0|**56.2**<br>**56.5**<br>55.1<br>52.7<br>53.1<br>**54.1**|35<br>49.5<br>11<br>**60.6**<br>55<br>**55.7**<br>15<br>**64.5**<br>39<br>53.8<br>22<br>**66.7**|40.0<br>24.3<br>35.1<br>40.0<br>32.5<br>**51.5**|119<br>0.0<br>42<br>0.0<br>192<br>0.0<br>55<br>0.0<br>128<br>0.0<br>78<br>0.0|**13.7**<br>10.3<br>12.3<br>9.7<br>10.1<br>12.4|
||||||||||
|**Overall**|72.80|81.19|28.11|53.25|59.00|40.23|0.00|13.31|



Table 3: Performance on a set of development examples annotated for semantic categories by Suhr et al. (2017) for both models (C3+BERT | ViLT) when trained with PPO. Developement performance refers to mean performance on the respective full development set. Results outperforming dev performance are in bold. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0009-02.png)


**----- Start of picture text -----**<br>
TOWER-FLIPIT TOWER-FLIPIT<br>PPO PPO+SF<br>100 100<br>50 50<br>0 0<br>SCATTER-FLIPIT SCATTER-FLIPIT<br>PPO PPO+SF<br>20 20<br>0 0<br>C3+BERT ViLT C3+BERT ViLT<br>True False<br>Accuracy Accuracy<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 5: Mean development set accuracies for FLIPIT CMDPs, averaged over three runs, reported according to the value of the context target boolean ( **Red** for True, **Gray** for False). **Dashed gray line** : full development set accuracies. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0009-04.png)


**----- Start of picture text -----**<br>
Non-stopped rollouts (%) Invalid actions (%)<br>100 100<br>50 50<br>0 0<br>Trajectory lengths Type of actions (%)<br>100<br>4<br>2 50<br>ADD<br>REMOVE<br>0 0<br>C3+BERT ViLT Expert (estimate)<br>TOWER-SCRATCHTOWER-FLIPITSCATTER-SCRATCHSCATTER-FLIPIT TOWER-SCRATCHTOWER-FLIPITSCATTER-SCRATCHSCATTER-FLIPIT<br>**----- End of picture text -----**<br>


Figure 6: Mean development statistics for PPO, averaged over three runs. Clockwise from top left: rollouts without a `STOP` action; rollouts with an invalid action; mean actions per rollout; relative rate of `ADD` / `REMOVE` actions. 

still displaying the reasoning richness of natural language. It is relatively easy to deploy using the standard Gymnasium API (Brockman et al., 2016), 

and has light compute requirements. Our data annotation approach allows including expressive and diverse natural language, while still providing accurate and automatic reward computation. It also exposes the potential connection between semantic parsing and reward evaluation in RL, thereby outlining how strong semantic parsers can benefit RL benchmarking. Our strong baselines illustrate the range of challenges _lil_ Gym presents, showing that existing methods can achieve non-trivial performance, but that there remain significant progress to be made. Our analysis lays out the framework for studying and reporting these future results. 

_lil_ Gym has significant potential beyond the tasks we study. It can be used without the language, to create thousands of micro RL tasks requiring set and relational visual reasoning. Our annotations form a new semantic parsing corpus with annotated executable meaning representations. The semantic diversity of the data, its executability, and the focus on visual reasoning make it a unique asset in the landscape of corpora for semantic parsing. _lil_ Gym is also promising for program synthesis guided by natural language (Wong et al., 2021). 

## **8 Limitations** 

_lil_ Gym uses synthetic visual stimuli, which does not reflect the complexity or characteristics of realistic visual observations. This is critical for our ability to control the environment and provide a lightweight and accessible RL benchmark. Our goal is not to provide a resource for the development of methods that aim to handle realistic visual input, and _lil_ Gym is not suitable for this purpose. The limited number of colors, shapes, and sizes used limits the visual and lexical complexity of the data. The synthetic nature of the data and the 

modular library of functions we use allow to relatively easily extend the environment (e.g., with new colors). This will require collecting additional natural language data. In this work, we opted to rely on the NLVR data without further expanding it. Some annotators of the original NLVR data adopted annotation strategies that led to repetition of some common phrases (e.g., starting statements with _there is_ ). While this creates some implicit patterns in the data, Suhr et al. (2017) showed that NLVR demonstrates high semantic diversity and compositionality. Finally, _lil_ Gym includes English data only. Expanding this data to other language is an important direction for future work. Translating the data is a feasible low-cost solution, because the program annotations will not require updating. 

Journal of Artifcial Intelligence Research, 47:253– 279. 

- Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. 2016. Openai gym. arXiv preprint arXiv:1606.01540. 

- Tianshi Cao, Jingkang Wang, Yining Zhang, and Sivabalan Manivasagam. 2020. Babyai++: Towards grounded-language learning beyond memorization. Beyond tabula rasa in RL (BeTR-RL) Workshop held in conjunction with the 8th International Conference on Learning Representations, ICLR. 

- Howard Chen, Alane Suhr, Dipendra Misra, Noah Snavely, and Yoav Artzi. 2019. Touchdown: Natural language navigation and spatial reasoning in visual street environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12538–12547. 

## **Ethics Statement** 

We paid U.S. standard market wage to our programmers (Appendix B). The rate was determined by the workers. The _lil_ Gym environment and data as is are intended to be used for research, including algorithm development and evaluation, and not for development of models to be deployed. 

Commented for anonymous submission 

## **Acknowledgements** 

This research was supported by ARO W911NF211-0106, NSF under grant No. 1750499, and a gift from Open Philanthropy. KB is supported by NSF under grant No. 2127309 to the Computing Research Association for the CIFellows Project. Results presented in this paper were obtained using CloudBank (Norman et al., 2021), which is supported by the National Science Foundation under award No. 1925001. We thank Alane Suhr, Ge Gao, Justin Chiu, Woojeong Kim, Jack Morris, Jacob Sharf and the Cornell NLP Group for support, comments, and helpful discussions. 

- Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Salem Lahlou, Lucas Willems, Chitwan Saharia, Thien Huu Nguyen, and Yoshua Bengio. 2019. Babyai: A platform to study the sample efficiency of grounded language learning. In 7th International Conference on Learning Representations, ICLR. 

- John D. Co-Reyes, Abhishek Gupta, Suvansh Sanjeev, Nick Altieri, John DeNero, P. Abbeel, and Sergey Levine. 2019. Guiding policies with language via meta-learning. In 7th International Conference on Learning Representations, ICLR. 

- Marc-Alexandre Côté, Ákos Kádár, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Matthew J. Hausknecht, Layla El Asri, Mahmoud Adada, Wendy Tay, and Adam Trischler. 2018. Textworld: A learning environment for text-based games. In Computer Games Workshop (CGW) held in conjunction with the 27th International Conference on Artifcial Intelligence, IJCAI. 

- Pradeep Dasigi, Matt Gardner, Shikhar Murty, Luke Zettlemoyer, and Eduard Hovy. 2019. Iterative search for weakly supervised semantic parsing. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2669–2680. 

## **References** 

- Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton van den Hengel. 2018. Vision-andlanguage navigation: Interpreting visually-grounded navigation instructions in real environments. In The IEEE Conference on Computer Vision and Pattern Recognition, pages 3674–3683. 

- Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. 2013. The arcade learning environment: An evaluation platform for general agents. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186. 

- Kunihiko Fukushima and Sei Miyake. 1982. Neocognitron: A self-organizing neural network model for a mechanism of visual pattern recognition. In 

Competition and Cooperation in Neural Nets, pages 267–285. Springer. 

- Omer Goldman, Veronica Latcinnik, Ehud Nave, Amir Globerson, and Jonathan Berant. 2018. Weakly supervised semantic parsing with abstract examples. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1809–1819. 

- Nitish Gupta, Sameer Singh, and Matt Gardner. 2021. Enforcing consistency in weakly supervised semantic parsing. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 168–174. 

- Assaf Hallak, Dotan Di Castro, and Shie Mannor. 2015. Contextual markov decision processes. arXiv preprint arXiv:1502.02259. 

- Austin W Hanjie, Victor Y Zhong, and Karthik Narasimhan. 2021. Grounding language to entities and dynamics for generalization in reinforcement learning. In International Conference on Machine Learning, pages 4051–4062. 

- Matthew J. Hausknecht, Prithviraj Ammanabrolu, MarcAlexandre Côté, and Xingdi Yuan. 2020. Interactive fiction games: A colossal adventure. In Proceedings of the AAAI Conference on Artifcial Intelligence, volume 34. 

- Karl Moritz Hermann, Felix Hill, Simon Green, Fumin Wang, Ryan Faulkner, Hubert Soyer, David Szepesvari, Wojciech Marian Czarnecki, Max Jaderberg, Denis Teplyashin, Marcus Wainwright, Chris Apps, Demis Hassabis, and Phil Blunsom. 2017. Grounded language learning in a simulated 3d world. arXiv preprint arXiv:1706.06551. 

- Drew A. Hudson and Christopher D. Manning. 2018. Compositional attention networks for machine reasoning. In 6th International Conference on Learning Representations, ICLR. 

- Minqi Jiang, Jelena Luketina, Nantas Nardelli, Pasquale Minervini, Philip HS Torr, Shimon Whiteson, and Tim Rocktäschel. 2020. Wordcraft: An environment for benchmarking commonsense agents. arXiv preprint arXiv:2007.09185. 

- Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Judy Hoffman, Li Fei-Fei, C. Lawrence Zitnick, and Ross B. Girshick. 2017. Inferring and executing programs for visual reasoning. 2017 IEEE International Conference on Computer Vision (ICCV), pages 3008–3017. 

- Wonjae Kim, Bokyung Son, and Ildoo Kim. 2021. Vilt: Vision-and-language transformer without convolution or region supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 5583–5594. 

- Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, ICLR. 

- Ilya Kostrikov. 2018. Pytorch implementations of reinforcement learning algorithms. 

- Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge. 2020. Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4392–4412. 

- Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR. 

- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. 2013. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602. 

- Karthik Narasimhan, Tejas D. Kulkarni, and Regina Barzilay. 2015. Language understanding for textbased games using deep reinforcement learning. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, EMNLP, pages 1–11. 

- Michael Norman, Vince Kellen, Shava Smallen, Brian DeMeulle, Shawn Strande, Ed Lazowska, Naomi Alterman, Rob Fatland, Sarah Stone, Amanda Tan, Katherine Yelick, Eric Van Dusen, and James Mitchell. 2021. Cloudbank: Managed services to simplify cloud access for computer science research and education. In Practice and Experience in Advanced Research Computing, PEARC ’21. Association for Computing Machinery. 

- Juan Pavez, Héctor Allende, and Héctor Allende-Cid. 2018. Working memory networks: Augmenting memory networks with a relational reasoning module. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1000–1009. 

- Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. 2018. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI Conference on Artifcial Intelligence, volume 32. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. 

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. 

- Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. 2020. ALFRED: A benchmark for interpreting grounded instructions for everyday tasks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, pages 10740–10749. 

- Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew J. Hausknecht. 2021. Alfworld: Aligning text and embodied environments for interactive learning. In 9th International Conference on Learning Representations, ICLR. 

- Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. 2022. Flava: A foundational language and vision alignment model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15638–15650. 

- Alane Suhr, Srinivasan Iyer, and Yoav Artzi. 2018. Learning to map context-dependent sentences to executable formal queries. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2238–2249. 

- Alane Suhr, Mike Lewis, James Yeh, and Yoav Artzi. 2017. A corpus of natural language for visual reasoning. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 217–223. 

- Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. 2019. A corpus for reasoning about natural language grounded in photographs. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6418–6428. 

   - Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45. 

   - Catherine Wong, Kevin Ellis, Joshua B. Tenenbaum, and Jacob Andreas. 2021. Leveraging language to learn program abstractions and search heuristics. In Proceedings of the 38th International Conference on Machine Learning, ICML, pages 11193–11204. 

   - Yiqun Yao, Jiaming Xu, Feng Wang, and Bo Xu. 2018. Cascaded mutual modulation for visual reasoning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 975–980. 

   - J.M. Zelle and Raymond J. Mooney. 1996. Learning to parse database queries using inductive logic programming. In Proceedings of the National Conference on Artifcial Intelligence. 

   - Luke S. Zettlemoyer and Michael Collins. 2005. Learning to map sentences to logical form: Structured classification with probabilistic categorial grammars. In Proceedings of the Conference on Uncertainty in Artifcial Intelligence. 

   - Wenbo Zheng, Lan Yan, Chao Gou, and FeiYue Wang. 2020. Webly supervised knowledge embedding model for visual reasoning. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12442–12451. 

- Hao Tan and Mohit Bansal. 2018. Object ordering with bidirectional matchings for visual reasoning. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 444– 451. 

- Emanuel Todorov, Tom Erez, and Yuval Tassa. 2012. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026–5033. 

- Jack Urbanek, Angela Fan, Siddharth Karamcheti, Saachi Jain, Samuel Humeau, Emily Dinan, Tim Rocktäschel, Douwe Kiela, Arthur Szlam, and Jason Weston. 2019. Learning to speak and act in a fantasy text adventure game. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, pages 673–683. 

## **A** SCATTER **Grid Simplification** 

To reduce the large action space of SCATTER, _lil_ Gym allows to simplify the pixel-based action space with a grid that is coarser than the image resolution of 380 _×_ 100. The actions applied in the environment remain in the original resolution, and the translation between the grid system to pixels is done heuristically. Without the heuristics, the transition to a grid coarser than the original image resolution would render many of the MPDs unsolvable. 

The heuristics simplify two translation problems: in what pixel exactly to place an object and which object to remove from a grid cell. Depending on the grid size, it is possible to add multiple objects in a cell. To find the exact pixel within a cell to add an object, we search for a pixel in the grid box where we can add the object starting from the upper left corner. We can add an object in a pixel if the object fits there without overlapping with other objects, the image boundaries, or the columns. We also snap objects to touch each other if the distance between them is below a threshold. This is to allow adding objects that _touch_ each other, a common constraint in _lil_ Gym statements. When removing an object from a grid cell, we remove the object with largest overlap with the cell. 

## **B Natural Language Annotation Details** 

We annotate each natural language statement in the NLVR corpus with a Python program representing its meaning. The programs return a boolean value, and are executable given the structured representation underlying each image. Figure 7 shows two examples of text statements with their annotated Python programs. 

We provide the annotators with a web-based annotation interface (Figure 8), a tutorial, and an application programming interface (API) presenting a set of functions, classes and objects that they can use for annotation. We ask the annotators to prioritize the faithfulness of the program to the natural language sentence and to prefer shorter annotations. We also provide them with examples of spurious logical forms and ask them to avoid such expressions. Annotators can raise questions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0013-06.png)


**----- Start of picture text -----**<br>
There are two towers with the same height<br>but their base is not the same in color.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0013-07.png)


**----- Start of picture text -----**<br>
exist(filter_obj(<br>all_boxes , lambda x: x.is_tower () and<br>exist(filter_obj(<br>all_boxes , lambda y:<br>y.is_tower () and<br>count(x.all_items_in_box ()) ==<br>count(y.all_items_in_box ()) and<br>get_set_colors(filter_obj(<br>y.all_items_in_box (),<br>is_bottom)) !=<br>get_set_colors(filter_obj(<br>x.all_items_in_box (),<br>is_bottom))))))<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0013-08.png)


**----- Start of picture text -----**<br>
There is a box with all 3 different colors and<br>a black triangle touching the wall with its top.<br>exist(filter_obj(<br>all_boxes , lambda x:<br>count(get_set_colors(<br>x.all_items_in_box ())) == 3 and<br>exist(filter_obj(<br>x.all_items_in_box (), lambda y:<br>is_black(y) and<br>is_triangle(y) and<br>is_touching_wall(y, Side.TOP)))))<br>**----- End of picture text -----**<br>


Figure 7: Example sentences with the example images displayed alongside them during annotation (left), and their annotated Python programs (right). Both sentences and programs are True for the corresponding image. 

Figure 8 shows the annotation interface for a single sentence. For every sentence, annotators are provided with a single example image from NLVR and an associated boolean value. Other images for the same statement from NLVR are used as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0014-00.png)


Figure 8: The annotation interface for collecting the Python program annotations in _lil_ Gym. 

hidden validation examples. The annotator never sees these images. 

The annotator can validate the program syntax and validate it within the browser. The validation executes the program against the given image and all hidden images. Validation passes only once the program returns the expected boolean value for all examples, including the visible and the hidden ones. The annotator can only submit their annotation after passing the syntax check and validation. They can assign a confidence score to their annotation and provide a comment. 

Annotators can skip examples in case of doubt. When skipping, they need to explicitly provide the reason. We assess the annotations by batch, then randomly redistribute the skipped examples or examples with problematic annotations to the annotators after the questions have been solved. We iteratively communicate with the workers throughout the entire annotation process. 

The annotation was done by four workers, one each from Croatia, India, Ukraine and United States. The hourly rate was roughly $23.25 per hour. We communicated to the workers the purpose of the data collection and how data will be used at recruiting time. 

## **C Experimental Setup Details** 

## **C.1 Learning Details** 

**Model Parameters and Computational Resources** C3+BERT and C10+BERT use a BERTbase model with 110M parameters. For ViLT, we use a ViLT-B/32 model with 87.4M parameters (Table 6 in Kim et al. (2021)). We use 6 NVIDIA RTX A6000, 3 Titan RTX, and 8 GeForce GTX 2080 Ti for our computations. The total computational budget is 950 GPU hours. 

**Tokenization** C3+BERT and C10+BERT use an uncased BERT WordPiece tokenizer with the default parameters. ViLT uses the default ViLT feature extractor and BERT tokenizer, based on the Hugging Face implementation (Wolf et al., 2020). 

**Hyperparameters** For C3+BERT and C10+BERT, we optimize using Adam (Kingma and Ba, 2015) with a learning rate of 3e-4, except on TOWER-FLIPIT and on SCATTER-FLIPIT , where we use 3e-5. For ViLT, we use AdamW (Loshchilov and Hutter, 2019) with a cosine scheduler and a base learning rate of 3e-5 for all experiments. The learning rate is warmed up for 1% of the maximal total training steps of 4M. We use patience for early stopping. We set 

entropy to 0.1 for all our TOWER experiments and to 0.3 for all our SCATTER experiments. We use a mini-batch of 64 actions for gradient updates. At each PPO iteration we sample 2,048 actions (i.e., for the internal update loop). 

**PPO+SF Details** PPO+SF is a simple variant of PPO that applies masking to all the actions except for `STOP` when the agent reaches a state in which it will receive a positive reward if it would select `STOP` . PPO+SF allows the learner to observe `STOP` with positive reward with higher probability than with conventional PPO. A side effect of this masking is that the learner often samples action with very low probability, which can lead to exploding gradients. We clip the PPO ratio to address this. Formally, the original PPO objective is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0015-02.png)


**----- Start of picture text -----**<br>
Non-stopped rollouts (%) Invalid actions (%)<br>100 100<br>50 50<br>0 0<br>Trajectory lengths Type of actions (%)<br>100<br>4<br>2 50<br>ADD<br>REMOVE<br>0 0<br>C3+BERT ViLT Expert (estimate)<br>TOWER-SCRATCHTOWER-FLIPITSCATTER-SCRATCHSCATTER-FLIPIT TOWER-SCRATCHTOWER-FLIPITSCATTER-SCRATCHSCATTER-FLIPIT<br>**----- End of picture text -----**<br>


Figure 9: Mean development statistics for PPO+SF, averaged over three runs. Clockwise from top left: rollouts without a `STOP` action; rollouts with an invalid action; mean actions per rollout; relative rate of `ADD` / `REMOVE` actions. 

## **D Additional Results and Analysis** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0015-05.png)


where _rt_ ( _θ_ ) = _ππoldθ_ (( _aatt|s|stt_ ))[,] _[A]_[ˆ][ is the advantage func-] tion, and _ϵ_ is a hyperparameter (Schulman et al., 2017). In PPO+SF, we clip the ratio term _rt_ ( _θ_ ) to avoid very large value due to “force” sampling of actions with very low probability: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0015-07.png)


where _M_ is a threshold bounding the ratio. We use _r_ ˆ _t_ ( _θ_ ) in place of _rt_ ( _θ_ ) for our experiments. 

## **C.2 Inference Details** 

There are three action types `STOP` , `ADD` , and `REMOVE` . Each type take a different number of arguments: `STOP` takes no arguments, `ADD` takes two arguments in TOWER and five in SCATTER, and `REMOVE` takes one argument in TOWER and two in SCATTER. During inference, actions _a_ are sampled from the agent policy _a ∼ π_ ( _·|s, c_ ), where _s_ is a state and _c_ is a context. We decompose the probability of an action to be a product of its type and arguments. This risks assigning generally lower probability to actions with more arguments, because of the multiplicative decomposition. We avoid this by sampling the required arguments as needed. We first sample an action type. Depending on the action type, we sample the required arguments. In practice, this means that when an argument slot is not used, the probability of that action marginalizes over all possible assignments to that argument slot. 

## **D.1 Development Rollout Statistics** 

Figure 9 shows development rollout statistics for PPO+SF. The statistics follow similar trends for the ones we show for PPO in Figure 6. Compared to PPO, we observe more non-stopped rollouts for TOWER-FLIPIT when training with PPO+SF, and less for SCATTER. These non-stopped TOWER-FLIPIT rollouts often correspond to the model getting stuck in add-remove loops. 

## **D.2 Error Analysis** 

We analyze model errors by sampling 50 erroneous development examples,[11] for the two SCATTER CMDPs trained with PPO, over one run: 

**SCATTER-SCRATCH with C3+BERT** 58% of the errors are due to invalid actions, and 42% due to direct or early termination. Among the invalid actions, all are due to trying to perform an action on a separator. Among the termination errors, 18% are due to direct termination, and 82% are due to early termination. 

- **SCATTER-SCRATCH with ViLT** 82% of the errors are due to invalid actions, and 18% due to direct or early termination. Among the invalid actions, 78% are due to trying to perform an action on a separator, 14% due to trying to remove an object from a position that does not include an object, 5% due to trying to put an item that cannot fit in 

11If there are less than 50 errors in the development set, we analyze the entire set. This occurs only in SCATTER-SCRATCH with C3+BERT. 

the box, and 3% due to trying to add an object on top of an existing one. Among the termination errors, 50% are due to direct termination and 50% due to erroneous termination. 

- **SCATTER-FLIPIT with C3+BERT** 58% of the errors are due to invalid actions, and 42% are due to direct or early termination. Among the invalid actions, 63% are due to trying to remove an object from a position that does not include an object, 24% are due to trying to perform an action on a separator, 10% due to trying to put an item that cannot fit in the box, and 3% due to trying to add an object on top of an existing one. Among the termination errors, 90% are due to direct termination and 10% due to erroneous termination. 

- **SCATTER-FLIPIT with ViLT** 64% of the mistakes are due to invalid actions, and 36% due to early termination. Among the invalid actions, 75% are due to trying to perform an action on a separator, 19% due to trying to remove an object from a position that does not include an object, and 6% due to trying to add an object on top of an existing one. 

## **D.3 Analysis of Action Selection Bias** 

We observe that the trained models often exhibit bias towards specific action arguments, which are sampled much more often than others during inference. Figure 10 illustrates this by visualizing coordinate selection frequencies on the development set for SCATTER CMDPs, for one of the runs. While the presence of bias is relatively persistent, the exact argument the models are biased towards vary. This indicates generalization limitations of our learned policies, which potentially converge to specific argument prematurely, and do not fully utilize the entire action space. We observe that this bias leads to selecting invalid actions, for example when attempting to place a large object on the edge so it crosses image boundaries. 

## **D.4 Performance Analysis by Semantic and Syntactic Phenomena** 

Suhr et al. (2017) manually annotated 200 development examples for semantic phenomena. Table 5 and Table 6 show the performance on this data of policies trained with PPO and PPO+SF. We provide an example sentence for each category. The two models mostly follow similar trends with 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning_images/acl_2023_lilGym_Natural_Language_Visual_Reasoning_with_Reinforcement_Learning.pdf-0016-07.png)


**----- Start of picture text -----**<br>
SCATTER-SCRATCH<br>C3+BERT ViLT<br>PPO<br>0.2<br>PPO+SF 0.0<br>0 2 4 6 8 10 12 14 16 18 0 2 4 6 8 10 12 14 16 18<br>SCATTER-FLIPIT<br>C3+BERT ViLT<br>PPO<br>0.2<br>PPO+SF 0.0<br>0 2 4 6 8 10 12 14 16 18 0 2 4 6 8 10 12 14 16 18<br>0<br>2<br>4<br>0<br>2<br>4<br>0<br>2<br>4<br>0<br>2<br>4<br>**----- End of picture text -----**<br>


Figure 10: Frequency of selecting the grid cell ( _x, y_ ) in SCATTER CMDPs for rollouts sampled on the development set, over one run. _x_ -axis and _y_ -axis show the _x_ and _y_ position of the cells in the SCATTER 19 _×_ 5 grid approximation. 

respect to the categories on which they perform above and below overall performance. The two models mostly follow similar trends with respect to the categories on which they perform above and below overall performance. When trained with PPO, both models outperform overall performance on hard cardinality (e.g., _...exactly four objects . .._ ) for TOWER CMDPs, and on presupposition for SCRATCH CMDPs. On spatial relations, both models perform above overall performance only for TOWER-SCRATCH , and below for all the other three CMDPs. We observe that PPO+SF is especially helpful for this category, bringing the performance of ViLT above average performance on all CMDPs. 

## **E Experiments with FLAVA** 

We conduct preliminary experiments with the base FLAVA model (350M parameters) (Singh et al., 2022).[12] Table 4 shows the results. On TOWER-FLIPIT , the results with PPO are outperforming ViLT in Table 2. On TOWER-SCRATCH , with PPO+SF, FLAVA’s results are on par with ViLT, and with PPO, below ViLT. On SCATTER environments, FLAVA’s performance is significantly lower than C3+BERT, C10+BERT and ViLT in Table 2. We tested different hyperparameters, using learning rates from 1e-3 to 3e-6, but di not find a combination that significantly improves the learning behaviour. Due to the computational resources required in training FLAVA, and the results 

> 12We also experimented with CLIP (ViT-B/32) (Radford et al., 2021), but the performed poorly on the simplest TOWER-SCRATCH CMDP, so was discarded relatively early. 

||**TOWER-SCRATCH**|**TOWER-FLIPIT**|**SCATTER-SCRATCH**|**SCATTER-FLIPIT**|
|---|---|---|---|---|
||Dev<br>Test|Dev<br>Test|Dev<br>Test|Dev<br>Test|
|PPO|75.46<br>65.43|**62.72**<br>**58.79**|12.64<br>15.48|1.28<br>0.68|
|PPO+SF|**84.05**<br>76.24|58.88<br>58.21|17.24<br>27.10|7.35<br>6.94|



Table 4: Mean task-completion accuracies for TOWER CMDPs using FLAVA, with seed 1. We optimize using AdamW, and use a learning rate of 3e-5. Bold results are outperforming or on par with ViLT in Table 2. 

on TOWER environments that are comparable but not always outperforming ViLT, we choose to not perform further hyperparameter ssearch on SCATTER. 

## **F Third-party Code** 

Whenever the intended use is provided, the use of existing artifacts comply with their intended use. Suhr et al. (2017) is under CC-BY-4.0, and Kostrikov (2018) is under MIT. The use of code from Goldman et al. (2018) was done with explicit approval from the authors, because no license was provided with the code. 

|**TOWER-SCRATCH**<br>**TOWER-FLIPIT**<br>**SCATTER-SCRATCH**<br>**SCATTER-FLIPIT**<br>Total<br>Correct %<br>Total<br>Correct %<br>Total<br>Correct %<br>Total<br>Correct %<br>Example<br>**Semantics**|Cardinality (hard)<br>98<br>**76.5**<br>**83.7**<br>480<br>**28.9**<br>**56.2**<br>35<br>49.5<br>40.0<br>119<br>0.0<br>**13.7**<br>_There are_**_exactly four objects_**_not touching any edge_<br>Cardinality (soft)<br>21<br>68.2<br>81.0<br>82<br>**30.1**<br>**56.5**<br>11<br>**60.6**<br>24.3<br>42<br>0.0<br>10.3<br>_There is a box with_**_at least one_**_square and_**_at least three_**_triangles._<br>Existential<br>122<br>**75.1**<br>**81.7**<br>577<br>**29.0**<br>55.1<br>55<br>**55.7**<br>35.1<br>192<br>0.0<br>12.3<br>**_There is a tower_**_with yellow base._<br>Universal<br>7<br>**85.7**<br>**95.2**<br>28<br>**29.7**<br>46.4<br>9<br>**81.5**<br>**59.3**<br>36<br>0.0<br>9.3<br>_There is a black item in_**_every box_**_._<br>Coordination<br>19<br>**85.9**<br>**84.2**<br>86<br>27.1<br>52.7<br>15<br>**64.5**<br>40.0<br>55<br>0.0<br>9.7<br>_There are 2 blue circles_**_and_**_1 blue triangle_<br>Coreference<br>3<br>**100.0**<br>77.8<br>10<br>13.3<br>10.0<br>3<br>**44.4**<br>33.3<br>9<br>0.0<br>7.4<br>_There is a blue triangle touching the wall with_**_its_**_side._<br>Spatial Relations<br>94<br>**74.8**<br>**81.6**<br>438<br>26.6<br>53.1<br>39<br>53.8<br>32.5<br>128<br>0.0<br>10.1<br>_there is one tower with a yellow block_**_above_**_a yellow block_<br>Comparative<br>5<br>66.7<br>73.3<br>20<br>11.7<br>21.7<br>1<br>**100.0**<br>**100.0**<br>4<br>0.0<br>**16.7**<br>_There is a box with multiple items and only one item_**_has a different color._**<br>Presupposition<br>17<br>**74.5**<br>**90.2**<br>74<br>27.0<br>**54.1**<br>22<br>**66.7**<br>**51.5**<br>78<br>0.0<br>12.4<br>_There is a box with seven items and_**_the three black items_**_are the same in shape._<br>Negation<br>4<br>**75.0**<br>66.7<br>15<br>13.3<br>37.8<br>14<br>54.8<br>33.3<br>52<br>0.0<br>7.0<br>_there is exactly one black triangle_**_not touching_**_the edge_<br>**Syntax**|Coordination<br>4<br>**83.3**<br>75.0<br>14<br>11.9<br>**59.5**<br>5<br>53.3<br>26.7<br>20<br>0.0<br>8.3<br>_There is a box with at least one square_**_and_**_at least three triangles._<br>PP Attachment<br>44<br>**76.5**<br>**81.8**<br>215<br>26.2<br>**54.3**<br>3<br>33.3<br>33.3<br>8<br>0.0<br>8.3<br>_There is a black block on a black block as the base of a tower_**_with_**_three blocks._<br>**Overall**<br>72.80<br>81.19<br>28.11<br>53.25<br>59.00<br>40.23<br>0.00<br>13.31<br>Table 5: Performance on a set of development examples annotated for semantic and syntactic categories bySuhr et al.(2017) for both models (C3+BERT | ViLT) when trained<br>with PPO. Dev performance refers to mean performance on the respective full development set. Results outperforming dev performance are in bold.|
|---|---|---|



|**TOWER-SCRATCH**<br>**TOWER-FLIPIT**<br>**SCATTER-SCRATCH**<br>**SCATTER-FLIPIT**<br>Total<br>Correct %<br>Total<br>Correct %<br>Total<br>Correct %<br>Total<br>Correct %<br>Example<br>**Semantics**|Cardinality (hard)<br>98<br>**84.3**<br>**85.1**<br>480<br>30.3<br>**68.9**<br>35<br>60.9<br>59.1<br>119<br>**17.1**<br>27.7<br>_There are_**_exactly four objects_**_not touching any edge_<br>Cardinality (soft)<br>21<br>76.2<br>**87.3**<br>82<br>**33.7**<br>**66.7**<br>11<br>**72.7**<br>**78.8**<br>42<br>10.3<br>18.3<br>_There is a box with_**_at least one_**_square and_**_at least three_**_triangles._<br>Existential<br>122<br>**84.1**<br>83.6<br>577<br>31.8<br>**69.5**<br>55<br>70.3<br>67.3<br>192<br>15.8<br>27.8<br>**_There is a tower_**_with yellow base._<br>Universal<br>7<br>80.9<br>**95.2**<br>28<br>26.2<br>52.4<br>9<br>**88.9**<br>**88.9**<br>36<br>7.4<br>16.6<br>_There is a black item in_**_every box_**_._<br>Coordination<br>19<br>**91.2**<br>**84.2**<br>86<br>**39.9**<br>65.1<br>15<br>**73.3**<br>64.5<br>55<br>6.1<br>17.0<br>_There are 2 blue circles_**_and_**_1 blue triangle_<br>Coreference<br>3<br>**88.9**<br>**100.0**<br>10<br>23.3<br>30.0<br>3<br>44.4<br>44.4<br>9<br>3.7<br>**33.3**<br>_There is a blue triangle touching the wall with_**_its_**_side._<br>Spatial Relations<br>94<br>**81.9**<br>**84.4**<br>438<br>26.9<br>**69.0**<br>39<br>68.4<br>**68.4**<br>128<br>16.7<br>**28.1**<br>_there is one tower with a yellow block_**_above_**_a yellow block_<br>Comparative<br>5<br>73.3<br>80.0<br>20<br>21.7<br>31.7<br>1<br>**100.0**<br>**100.0**<br>4<br>16.7<br>**41.7**<br>_There is a box with multiple items and only one item_**_has a different color._**<br>Presupposition<br>17<br>**82.4**<br>**94.1**<br>74<br>29.3<br>65.3<br>22<br>**72.7**<br>**69.7**<br>78<br>**14.1**<br>**29.9**<br>_There is a box with seven items and_**_the three black items_**_are the same in shape._<br>Negation<br>4<br>75.0<br>58.3<br>15<br>15.6<br>64.4<br>14<br>66.7<br>**71.4**<br>52<br>**17.3**<br>23.7<br>_there is exactly one black triangle_**_not touching_**_the edge_<br>**Syntax**|Coordination<br>4<br>**91.7**<br>75.0<br>14<br>19.0<br>**69.0**<br>5<br>60.0<br>46.7<br>20<br>5.0<br>20.0<br>_There is a box with at least one square_**_and_**_at least three triangles._<br>PP Attachment<br>44<br>**84.8**<br>**85.6**<br>215<br>27.0<br>**70.2**<br>3<br>**77.8**<br>66.7<br>8<br>8.3<br>20.8<br>_There is a black block on a black block as the base of a tower_**_with_**_three blocks._<br>**Overall**<br>81.80<br>84.05<br>32.59<br>65.68<br>72.03<br>67.43<br>17.04<br>28.01<br>Table 6: Performance on a set of development examples annotated for semantic and syntactic categories bySuhr et al.(2017) for both models (C3+BERT | ViLT) when trained<br>with PPO+SF. Dev performance refers to mean performance on the respective full development set. Results outperforming dev performance are in bold.|
|---|---|---|



