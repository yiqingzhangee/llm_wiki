Preprint 

# GENOME: GENERATIVE NEURO-SYMBOLIC VISUAL REASONING BY GROWING AND REUSING MODULES 

**Zhenfang Chen** _[∗]_ MIT-IBM Watson AI Lab 

**Rui Sun** _[[∗]]_ 

**Rui Sun** _[[∗]]_ **Wenjun Liu** _[∗]_ Columbia University Tsinghua University 

## **Yining Hong** 

University of California, Los Angeles 

**Chuang Gan** MIT-IBM Watson AI Lab and UMass Amherst 

## ABSTRACT 

Recent works have shown that Large Language Models (LLMs) could empower traditional neuro-symbolic models via programming capabilities to translate language into module descriptions, thus achieving strong visual reasoning results while maintaining the model’s transparency and efficiency. However, these models usually exhaustively generate the entire code snippet given each new instance of a task, which is extremely ineffective. On the contrary, human beings gradually acquire knowledge that can be reused and grow into more profound skills for fast generalization to new tasks since we are an infant. Inspired by this, we propose generative neuro-symbolic visual reasoning by growing and reusing modules. Specifically, our model consists of three unique stages, module initialization, module generation, and module execution. First, given a vision-language task, we adopt LLMs to examine whether we could reuse and grow over established modules to handle this new task. If not, we initialize a new module needed by the task and specify the inputs and outputs of this new module. After that, the new module is created by querying LLMs to generate corresponding code snippets that match the requirements. In order to get a better sense of the new module’s ability, we treat few-shot training examples as test cases to see if our new module could pass these cases. If yes, the new module is added to the module library for future reuse. Finally, we evaluate the performance of our model on the testing set by executing the parsed programs with the newly made visual modules to get the results. We find the proposed model possesses several advantages. First, it performs competitively on standard tasks like visual question answering and referring expression comprehension; Second, the modules learned from one task can be seamlessly transferred to new tasks; Last but not least, it is able to adapt to new visual reasoning tasks by observing a few training examples and reusing modules[1] . 

## 1 INTRODUCTION 

Neuro-symbolic visual reasoning models (Andreas et al., 2016b; Mao et al., 2019b) refer to the algorithm family that combines deep neural networks (lec, 1998; Hochreiter & Schmidhuber, 1997) for learning correlations among the training data and symbolic methods (Yi et al., 2018; Andreas et al., 2016a) to perform explicit and transparent multi-step reasoning. In contrast to pure neural network-based models (Hudson & Manning, 2018; Li et al., 2023), neuro-symbolic approaches achieve strong performance in visual reasoning tasks, simultaneously offering superior model transparency and data efficiency. 

Nevertheless, such models suffer from several inherent limitations. Firstly, their language parsers (Yi et al., 2018; Andreas et al., 2016b), employed for the conversion from natural language into symbolic programs, typically demand extensive domain-specific language-program pairs to train on, and struggle to generalize effectively to unconstrained natural language instructions. Additionally, these models necessitate a custom design for every module, rendering the process labor-intensive and lacking scalability. 

> _∗_ indicates equal contributions 

> 1Project page: https://vis-www.cs.umass.edu/genome 

1 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules_images/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Existing Methods<br>Fixed Module  Neuro-symbolic Visual Reasoning<br>Library<br>Q:  the sky Are the couch and  the same  Module 1:Module 2:Module 3:… VQA(…)LOC(…)CROP(…) BOX0=IMAGE0=BOX1=IMAGE1=ANSWER0=ANSWER1='What color is the couch?’) LOCLOCCROPCROPVQAVQA (image=IMAGE,object='couch’)(image=IMAGE,object='sky’) (image=IMAGE,box=BOX0)(image=IMAGE,box=BOX1)(image=IMAGE0,question=(image=IMAGE1,question= def execute_command(image): the couch’s color?")the sky’s color?")image_patch = ImagePatch(image)couch_patches = image_patch.find(”couch")color1 = couch_patch.simple_query(f”What’ssky_patches = image_patch.find(”sky")color2 = couch_patch.simple_query(f”What’s<br>color ? 'What color is the sky’) if color1==color2:<br>A : No. ANSWER2= EVAL (expr=f"'yes' if {ANSWER0} return “yes”<br>=={ANSWER1} else 'no’”) (A) VisProg else return “No” (B) ViperGPT<br>Our Method<br>Module 1: Instruction :<br>Generate … New Module:Class COMPARE_ATTRIBUTE( ): LOC(…) replace with  colorthe same   as the  the  apple grape<br>LLM modules """ Input:  with a  peach<br>image: an image<br>box1: a box listattribute: Q : Does the microwave have<br>Update ……Output: a toaster? different color  than the<br>True/False<br>"""<br>A : No.<br>(C) Generative Module Learning (D) Applications<br>**----- End of picture text -----**<br>


Figure 1: The motivation of the GENOME. Compared with VisProg and ViperGPT which exhaustively generate a code snippet for each input case, our GENOME is able to generate new modules and reuse old modules to handle the query. The module generated by GENOME can be used to handle other instances of the task for better performance. Second, the generated module can be transferred to different tasks like image editing. Finally, it can learn to handle new tasks like Raven (Burke, 1985; Zhang et al., 2019) by learning modules from only a few training samples. The edited region and the correct answer for the Raven task are labeled with red boxes for better visualization. 

Recent advancements in large language models (LLMs) (Brown et al., 2020; Ouyang et al., 2022) have ushered in a new era with its remarkable performances across various applications, including chatbots (Shuster et al., 2022), virtual assistants (Dong et al., 2023), and programming assistants (Chen et al., 2021a). Riding this unprecedented wave, researchers reformulate the old wisdom by incorporating LLMs into neuro-symbolic reasoning, bypassing the inflexibility and ineffectiveness of domain-specific language-to-program parsers. Specifically, VisProg (Gupta & Kembhavi, 2022) pre-defines a set of visual modules, and uses LLMs to transform language instructions into symbolic programs consisting of the pre-defined visual modules. Taking a step forward, ViperGPT (Sur´ıs et al., 2023) releases the burden on manually-defined visual modules by introducing a code generator that could produce a code snippet based on each input instance of a new task. 

Promising as these LLM-based neuro-symbolic models can be, they inevitably bear several weaknesses compared to the learning and reasoning processes of human beings. First, both VisProg and ViperGPT exhaustively produce one code snippet for each new instance of a task, which is extremely ineffective. This is in stark contrast with the human learning process: from an early age, we organically accumulate knowledge from particular experiences. Such knowledge acquired from specific cases could be reused and reconfigured, enabling us to quickly adapt to new tasks and new demands (Harlow, 1949; Mitchell et al., 1986; Lake et al., 2016; Ellis et al., 2023). The knowledge blocks grow progressively over time, gradually into a library with extraordinary richness and flexibility for fast generalization to any unseen task - the knowledge library that these models fall short of. Second, both models do not verify and examine the codes they generate. It seems that when the models generate a bad code snippet that cannot solve the input case, they just “let it go” without taking another stab for larger chance towards success. And of course, when they encounter similar cases again, they keep “stepping on the same rake”. Human beings, on the other hand, would verify and examine the acquired knowledge by proposing a set of test scenarios before storing them in the library (Brul´e & Blount, 1989). It’s crucial that a neuro-symbolic reasoning model is equipped with the same abilities to verify the codes it produces, stores them in a library if satisfactory, and makes another attempt when the codes fail. 

To this end, we introduce a novel Generative Neuro-symbolic Visual Reasoning Model (GENOME), proficient in assimilating new neural modules from a limited set of training examples. This model 

2 

Preprint 

excels in handling standard visual reasoning tasks such as visual question answering. Additionally, it demonstrates outstanding module transfer capabilities for tasks like image editing, and exhibits an exceptional ability to generalize to new reasoning tasks with limited training examples. As illustrated in Figure 2, GENOME comprises three stages: 1) module initialization, 2) module generation, and 3) module execution. From the initial training set examples, an LLM discerns the necessity for a new module to tackle the task and, if required, produces the respective input and output. In the second stage, LLMs implement and refine the new module, ensuring seamless integration with existing modules and resulting in accurate responses to the training queries. During testing, the LLM first converts language instructions into executable high-level programs like _COMPARE_ ~~_A_~~ _TTRIBUTE(IMAGE,BOX0,BOX1,ATTR)_ for comparing attributes of different bounding boxes. The program will be run with the new module sets, producing the desired outputs. 

We assessed the performance of GENOME across six visual reasoning tasks, spanning from visual question answering (Hudson & Manning, 2019) to Raven’s Progressive Matrices (Zhang et al., 2019). The experimental findings reveal that GENOME delivers competitive results on standard benchmarks, ensuring both transparency and interoperability. Notably, modules honed on these standard tasks can be adeptly adapted to diverse domains, including image editing and knowledge tagging (Gupta & Kembhavi, 2022). Additionally, with minimal training examples, GENOME demonstrates the capability to manage new visual reasoning tasks (Burke, 1985; Jiang et al., 2023) by repurposing modules. Our code and data will be publicly available. 

## 2 RELATED WORK 

**Visual Reasoning.** Our work aims to handle visual reasoning tasks, which require a model to draw new inferences based on the acquired visual cues in images or videos (Hudson & Manning, 2019; Kazemzadeh et al., 2014; Goyal et al., 2017; Zhang et al., 2019; Jiang et al., 2023). Typical tasks for visual reasoning include visible question answering (Goyal et al., 2017; Hudson & Manning, 2019), visual grounding (Kazemzadeh et al., 2014; Yu et al., 2016; Chen et al., 2020) and Raven’s Progressive Matrices (Burke, 1985; Zhang et al., 2019). Various models (Hudson & Manning, 2018; Yu et al., 2018; Zhang et al., 2021; Ding et al., 2023) have been developed to handle these tasks but most of them are ad-hoc and carefully designed for a specific task, leaving it an open research question on how to build a general model that can handle different kinds of visual reasoning problems by only showing a few examples. 

**Neuro-symbolic Visual Reasoning.** Our work is also closely related to neuro-symbolic visual reasoning models (Andreas et al., 2016b; Mao et al., 2019a; Chen et al., 2021c; 2022), where the models decompose the query of the visual reasoning tasks into a series of reasoning steps and represent each reasoning step with a neural module (i.e., a code snippet for achieving specific functions like localizing objects and recognizing object categories). While these models have better model interoperability and data efficiency than previous connectionist models (Hudson & Manning, 2018; Anderson et al., 2018), they often show their limitations in representing natural language instructions in the wild with the limited pre-defined reasoning steps (Yang et al., 2020; Chen et al., 2021b). Moreover, they need to manually define and implement each neural module one by one, making it hard to scale up and handle multiple tasks within a single model. 

**Foundation Models for Reasoning.** Recently, large language models (LLMs) (Brown et al., 2020; Ouyang et al., 2022) have been widely used in language understanding (Hendrycks et al., 2020) and reasoning (Cobbe et al., 2021; Amini et al., 2019). Schick et al. (2023) develop the toolformer to show that LLMs can use external tools to better handle language tasks. Cai et al. (2023) shows that LLMs can make simple tools for natural language tasks by writing code snippets. LLMs have also been used in vision-language tasks. Most of these works (Li et al., 2023; Alayrac et al., 2022) connect LLMs with additional vision encoders and fine-tune them with massive vision-language pairs. As evaluated by Xu et al. (2023b), while these models show great performance on in-domain tasks, they also perform poorly on tasks out of the training domains. They are also extremely computationexpensive. For example, it takes 15 days to use 1536 TPUv4 for training Flamingo (Alayrac et al., 2022). 

**Visual Programming by LLMs.** Another line of research has been combining vision models (Li* et al., 2022; Kirillov et al., 2023; Radford et al., 2021) with LLMs in an off-shelf manner. Early models (Yang et al., 2022; Chen et al., 2023b) transformed images into captions and append the cap- 

3 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules_images/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Q1:  Is the desk made of  Class COMPARE_ATTRIBUTE(): BOX0= LOC (IMAGE,object='couch’) Q1 : Does the dusbin<br>different material from the  """ Input:  BOX1= LOC (IMAGE,object='sky’)  have  the same material<br>Trainingexamples storage box?  Q2 the chair? : What’s the material of  A2A1 : Plastic ….: No.       Output: """   ………… ANSWER0= TE …….(IMAGE,BOX0,BOX1) COMPARE_ATTRIBU Testing examples asQ2  the . …fridge?<br>Module  Test<br>signatures cases<br>Can existing modules handle the case? Generate modules Class COMPARE_ATTRIBUTE(): def __init__(self):def execution(self, image, box..… BOX0=)BOX1= LOCLOC (IMAGE,object=‘dusbin’(IMAGE,object=‘fridge’)<br>LLM What module No LLM No Debug … Trainimage Scalable Module  ANSWER0= TE …….(IMAGE,BOX0,BOX1) COMPARE_ATTRIBU<br>should we make? Pass the  Evaluated on Library<br>test cases?  test cases Code<br>Yes Execution<br>Class COMPARE_ATTRIBUTE(): BOX0= LOC (IMAGE,object='couch’) Module 1:<br>""" Input: …… BOX1=ANSWER0= LOCCOMPARE_ATTRIBU (IMAGE,object='sky’)  Module 2: LOC(IMAGE,OBJECT) Answer : No<br>      Output: """   ……  TE …….(IMAGE,BOX0,BOX1) Scalable Module  … New Moule   VQA(IMAGE,QUESTION):  COMPARE_ATTRIBUTE( Test image<br>Module signatures Operation steps Library IMAGE, BOX1, …)<br>(1). Module Initialization (2). Module Generation (3). Module Execution<br>**----- End of picture text -----**<br>


Figure 2: The framework of our GENOME, which contains three stages, module initialization, module generation, and module execution. In stage 1, we feed the questions and the signature of the existing modules to the LLM and ask it to identify whether we can handle the query within the operations of the existing modules. If not, we ask the LLM to generate the signature of the new module (i.e. the input and output) and predict the reasoning steps to handle the query task. In stage 2, we feed the module signature and the testing cases to the LLM and ask the LLM to implement the module and test its pass rate on the training examples. We only accept the modules that successfully handle the query. In stage 3, we first use the LLM to parse the query into symbolic operations and then execute these operations on the test images with the help of the scalable module library. We take VQA as an example and such a framework can also be expanded to other tasks like referring expression comprehension and Raven. 

tions into the LLMs’ prompt to handle vision-language tasks. While the method is simple, they also perform inferior and lack transparency. Recently, VisPROG (Gupta & Kembhavi, 2022) uses LLMs to transform language instructions into pre-defined modular operations for step-by-step reasoning. However, it still requires manually implementing each module one by one. Later, ViperGPT (Sur´ıs et al., 2023) shows that the LLMs can be used to write a code snippet for each query instance independently to handle vision tasks. However, the code it writes has not been examined and tested by any training examples and there is no guarantee about the performance and code safety. Instead, we propose GENOME that ask LLMs to create new neural models (i.e. general code snippets to achieve specific functions) and handle the given tasks through only a few training examples. Such newly generated modules can cooperate with each other and be reused for other tasks for better performance. 

## 3 METHOD 

## 3.1 OVERALL 

In this section, we present a novel framework named as Generative Neuro-symbolic Visual Reasoning Model (GENOME) for the acquisition of neural modules and solutions of visual reasoning tasks with only a limited set of training examples. GENOME comprises several pre-defined operators that serve as the initial building blocks. Each neural operator corresponds to a neural module and is implemented using a Python code snippet, thereby enabling specific functions such as object localization within an image. Nevertheless, it is not possible to pre-define all the necessary neural modules prior to addressing the visual reasoning tasks. Consequently, there arises a need to generate new modules based on a limited number of visual reasoning task examples. 

Figure 2 illustrates that GENOME consists of three distinct stages: 1) module initialization, 2) module generation, and 3) module execution. In the module initialization stage, when provided with a small set of training examples from the visual reasoning dataset, the primary objective is to de- 

4 

Preprint 

termine whether the existing neural modules are sufficient to address the query examples. If the existing modules are inadequate for the query instance, GENOME will identify the requirements for creating new modules, including defining the modules’ input and output specifications. During the module generation stage, GENOME leverages the LLM to implement the neural module based on the provided training examples and the specified input and output format (i.e., function signature) and add the module to the module library only when it passes the test cases. Once the new module is successfully implemented, the module execution orchestrates the transformation of input queries into a sequence of neural operations. Subsequently, these operations are applied to the neural modules to obtain the correct output. All these three stages are powered by robust code generation capabilities and the in-context learning technique. Prompts for each stage can be found at Figure 15-17 

## 3.2 MODEL DETAILS 

Utilizing a limited number of examples from the training set of visual reasoning tasks, we employ the GENOME framework, comprising three distinctive stages: module initialization, module generation, and module execution. 

**Module Initialization.** The initial phase within our GENOME framework is module initialization, dedicated to determining the set of new modules required to address the visual reasoning task. As depicted in Figure 2-1, we employ an LLM to assess the feasibility of handling these training instances using existing neural modules. Should this not be achievable, we task the LLM with specifying the necessary new modules (e.g. _COMPARE_ ~~_A_~~ _TTRIBUTE_ in Figure 2) for an accurate response to the query. The outcome of this stage comprises function signatures detailing the input and output formats for these new modules. Furthermore, it facilitates the transformation of the input query into a sequence of reasoning steps, which function as test cases to validate the correctness of the generated program within module generation. The prompt for the LLM in this stage is shown in Figure 15. 

**Module Generation.** The second phase of our GENOME framework is module generation, which focuses on implementing the correct new modules proposed during the module initialization stage. Specifically, after receiving the signature of a new module, we incorporate corresponding test cases into the prompt and employ learning-in-context techniques to generate multiple program candidates. These program candidates are subsequently executed using the provided training examples. If a program encounters errors during execution, we incorporate the error information into the LLM’s prompt and instruct it to rectify these issues. We only accept program candidates that achieve a pass rate surpassing a predefined threshold ( _η_ ). This procedure bears resemblance to the code translation of LLMs discussed in (Chen et al., 2023a), but we extend it to accommodate more intricate multi-modal input types and instructions from natural language and raw images. The inclusion of module generation in the context of visual reasoning tasks offers two principal advantages. Firstly, it upholds the transparency and interpretability of neuro-symbolic models while preserving competitive performance. Secondly, it exhibits generative capabilities and scalability as our GENOME can autonomously generate new modules tailored to specific tasks. 

**Module Execution.** Given the integration of newly-generated modules with existing neural modules tailored for visual reasoning, the GENOME framework initiates query parsing from the testing dataset, transforming them into executable operations through in-context learning. An illustrative prompt for this stage is depicted in Figure 17. Notably, although various visual reasoning tasks may possess distinct inputs and outputs, they can re-purpose these intermediary modules designed for other tasks to enhance overall performance. This feature represents a unique capability for code generation at the module level, an aspect hitherto unexplored by prior methods(Sur´ıs et al., 2023; Gupta & Kembhavi, 2022). 

## 4 EXPERIMENTS 

In this section, we present a comprehensive series of experiments to evaluate the performance of our models. Initially, we demonstrate our models’ effectiveness in learning neural modules on two established benchmarks: GQA (Hudson & Manning, 2019), focusing on compositional visual question 

5 

Preprint 

answering, and RefCOCO (Kazemzadeh et al., 2014), which assesses referring expression comprehension. Subsequently, we illustrate how the modules acquired from these two datasets can be successfully applied to novel tasks such as image editing and knowledge tagging. Moreover, we highlight the adaptability of our framework to address novel visual reasoning tasks (Raven (Zhang et al., 2019) and MEWL (Jiang et al., 2023)), even with limited training examples. Before delving into these experiments, we provide an overview of the experimental settings. 

**Experimental Details.** The success of our GENOME relies on a set of pre-defined modules and APIs as the starting point. We utilize handcrafted modules from VisProg (Gupta & Kembhavi, 2022) as our initial components. Additionally, we incorporate several new APIs from ViperGPT to enhance module creation. We also include some new APIs from ViperGPT (Sur´ıs et al., 2023) for making new modules. In Section 4.4, we also include results parsed by the open-source LLM from WLM (Xu et al., 2023a) to investigate the influence of different LLM models. A comprehensive list of the pretrained modules employed in our approach can be found in Section A.1 of the Appendix. We extract training examples to acquire new modules. More precisely, we extracted 300 examples from GQA, 100 from RefCOCO, 10 from Raven, and 10 from MEWL. 

**Datasets and Evaluation Metric.** We show experiments of our GENOME on standard visionlanguage benchmarks, GQA (Hudson & Manning, 2019) and RefCOCO Kazemzadeh et al. (2014). GQA is a popular compositional visual reasoning dataset with synthesis multi-hop questions, making it suitable for multi-step reasoning. RefCOCO is a typical visual grounding dataset, evaluating models’ ability to localize objects and understand fine-grained spatial and semantic relationships. Following ViperGPT, we evaluate GQA on test-dev split and RefCOCO on the testA split. Then, we show GENOME’s abilities on other the transferred tasks, image editing, and knowledge tagging and compare it with VisProg. Since the image editing and knowledge tagging datasets from VisProg are not publicly available, we built two new datasets for evaluation. The new editing dataset contains 50 images and instruction pairs. The new knowledge tagging dataset contains 50 images with 50 referring expressions. We provide more details about the dataset in Appendix A.3. The datasets will be released for research purposes. Finally, we show that GENOME can learn to handle new visual reasoning tasks like Raven (Zhang et al., 2019) and MEWL (Jiang et al., 2023) by observing a few training examples and module learning. Raven is a task for relational and analogical visual reasoning of image sets and has been widely used for non-verbal intelligence tests. MEWL is a recent benchmark proposed to assess how machines learn word meaning in grounded visual scenes. Examples of these tasks can be found at Figure 5 and Figure 6. 

## 4.1 COMPARISON WITH BASELINES ON VISUAL REASONING. 

We conducted analysis between our model and several baseline models using the GQA and RefCOCO datasets. Due to the deprecation of the original professional Codex API (code-davinci-002), we replaced it with the currently available API (gpt-3.5-turbo-instruct) and conducted experiments with both our model and the baseline models to ensure a fair comparison. We did not carry out experiments with GPT-4 due to the prohibitive cost. 

The results, as presented in Table 1, demonstrate that our model achieves competitive performance in both visual question answering and referring expression comprehension, thus confirming its effectiveness. Furthermore, we provide an illustrative module from our model in Figure 11. This newly created module has the capability to utilize various available APIs to select attributes from the images. The step-by-step reasoning process of our model is detailed in Figure 3, offering greater transparency compared to end-to-end models. 

||Methods<br>BLIP-2<br>KOSMOS-2<br>ViperGPT-CodeX|GQA<br>44.7<br>-<br>48.1|RefCOCO<br>-<br>57.4<br>72.0|
|---|---|---|---|
||VisPROG-Instruct|45.4|-|
||ViperGPT-Instruct|38.2|62.4|
||Ours-Instruct|45.6|69.2|



Table 1: Evaluation on standard visual reasoning benchmarks, GQA and RefCOCO. 

## 4.2 GENOME FOR TRANSFER LEARNING 

In this section, we demonstrate our model’s robust ca- 

pabilities in transfer learning. We augment the modular library by incorporating modules created 

6 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules_images/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules.pdf-0007-01.png)


**----- Start of picture text -----**<br>
GQA classdefCHOOSE_ATTRIBUTE():predict(self,if len(boxes) > 0:img,boxes,obj,attr1,attr2): img: out_img:<br>box = boxes[0]<br>IMAGE: elsebox = self.expand_box(box, out_img: = img.crop(box) img.size)<br>Question: Is the shirt short sleeved or long sleeved? out_img = img<br>Generated Program: prompt1 = f'Tellprompt2 = f'Tell me the attributes when the {obj} is {attr1} in one sentence.'me the attributes when the {obj} is {attr2} in one sentence.'<br>BOX0=LOC(image=IMAGE,object=‘shirt’) ANSWER0=CHOOSE_ATTRIBUTE(image=IMAGE,box=BOX0,obj=‘shirt’,attr1=‘long sleeved’, attr2 =‘short sleeved’)FINAL_RESULT=RESULT(var=ANSWER0) obj_desc1 obj_desc2 result1result2ifelsereturnresult1result:resultresult= API.clip(= API.clip(= API.gpt3(prompt1, 'gpt3_general')= API.gpt3(prompt2, 'gpt3_general')> = attr1= attr2result2out_imgout_img: ,,obj_desc1obj_desc2)) obj_desc1obj_desc2short sleeves, and a straight hem.result1result: short sleeved: 0.9; : The shirt has long sleeves, a collar, and buttons.: The shirt is short sleeved with a round neckline, result2: 0.2<br>def execute(self,resultreturn = self.predict(resultimg,boxes,obj,attr1,attr2):img,boxes,obj,attr1,attr2) result: short sleeved<br>RefCOCO classdefSORT_SPATIAL():parse_depth(self,box_depth_list = [] img,box_list):        # compute depths for front or background img: box_depth_list:<br>depth_mapfor box in= API.depth(box_list:            img)<br>x1, y1, x2, y2 = box<br>depth_mapavg_depth = np.array(depth_map)            = np.median(depth_map[x1:x2, y1:y2])             ,<br>IMAGE: returnbox_depth_list.append((box, avg_depth))        box_depth_list depth: 0.13 0.03<br>def predict(self,if "front" inimglocation ,box_list,location,index):        or "behind" in location:<br>Question: Closest kid in blue. box_depth_listbox_list_sortedout_box_listif "behind" in = [box_i[0] = self.parse_depth(location:                = sorted(box_depth_listfor box_iimgin, box_list)            box_list_sorted, key=lambda x: x[1])            ]             box_list_sorted: out_box_list:<br>Generated Program: BOXLIST0=LOC(image=IMAGE,object=‘kid’)BOXLIST1=FILTER_PROPERTY(image=IMAGE,box_list= BOXLIST0,object=‘kid’,attribute=‘blue’)BOX2=SORT_SPATIAL(image=IMAGE,box_list=BOXLIST1, elsereturn:            if "left" out_box_listout_box_list.reverse()        box_list.. in location:                = sorted(box_list, key=lambda x: x[0])                     depth: 0.03 , 0.13 ,<br>location=‘front’,index=1)FINAL_RESULT=RESULT(var=BOX2) def execute(self,return self.predict(img,box_list,location,index):        img,box_list,location,index) result:<br>**----- End of picture text -----**<br>


Figure 3: Qualitative examples of GENOME’s on GQA and RefCOCO. The query images, language instructions, and the parsed programs are shown on the left. The corresponding new modules and the value of important variables are shown on the right. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules_images/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules.pdf-0007-03.png)


**----- Start of picture text -----**<br>
Replace the bottle of the same  Select the superhero with the<br>material as the blue one with a  Hide the second man from  Create a color pop of the third same color as the Hulk and<br>red bottle the left with 8) person from the right create a color pop<br>Tag the second actor who played  Tag the Grammy-winning musician  Tag the social media platform logo<br>James Bond from the left in the middle in the same color as Facebook<br>Tag the second gas planet from the left in<br>our solar system<br>**----- End of picture text -----**<br>


Figure 4: Qualitative examples of GENOME’s on the image editing and knowledge tagging tasks. The language instructions of the tasks are shown the original images while the modified results of the images are shown below the original images. The emphasis of the instructions is highlighted with red colors, which requires our new modules to handle. While the key regions of the output images are bounded with green colors. 

from GQA and RefCOCO, employing in-context examples to guide the Language Model (LLM) in generating step-by-step instructions for task execution. Qualitative results for this task are depicted in Figure 4. As illustrated in Figure 4, our model excels in generating semantically accurate images using the newly added module, whereas the baseline VisProg struggles to capture the required relationships with its fixed, pre-defined module library. To provide a more comprehensive evaluation of image editing, we enlist annotators to manually assess the correctness of the generated images. 

7 

Preprint 

|Methods|Image Editing<br>Accuracy|Tagging<br>Precision<br>Recall|Tagging<br>Precision<br>Recall|F1|Localization<br>Precision<br>Recall|Localization<br>Precision<br>Recall|F1|
|---|---|---|---|---|---|---|---|
|VisProg|16.7|18.4|21.7|19.9|32.8|35.3|34.0|
|GENOME|55.3|67.1|52.3|58.8|76.9|57.9|66.0|



Table 2: Evaluation of GENOME on Transfer Learning with image editing and knowledge tagging tasks. Our GENOME shows much better performance on all criteria, showing the effectiveness of the transferred modules. A qualitative comparison can be seen Figure 7 in the Appendix. 

|Methods<br>Center<br>L-R<br>U-D<br>ResNet+DRT<br>58.1<br>65.8<br>67.1<br>ALANS-V<br>98.4<br>97.3<br>96.4<br>GENOME<br>80.1<br>67.6<br>69.1<br>Human<br>95.5<br>86.4<br>81.8|Methods<br>shape color material|
|---|---|
||Aloe<br>34.2<br>33.2<br>31.0<br>Flamingo-1.1B<br>49.3<br>35.3<br>48.5|
||GENOME<br>43.7<br>45.3<br>41.0|
||Human<br>92.4<br>87.2<br>72.7|



Table 3: Evaluation of GENOME on Table 4: Evaluation of GENOME on Raven (Zhang et al., 2019). Compared with MEWL (Jiang et al., 2023). Compared with methods trained with massive in-domain data, approaches trained on extensive in-domain data, our model performs competitively. our model shows competitive performance. 

The models’ performance is compared in Table 2, where our model outperforms the baseline. In the context of knowledge tagging, we task annotators with marking image regions referenced by expressions and employ the same metrics as RefCOCO for evaluating the accuracy of bounding boxes and employ the BERT score to assess the correctness of labeled names. Our model demonstrates superior performance in both image editing and knowledge tagging. We show a typical example in Figure 7 of the Appendix to show how our GENOME make use of new modules to perform better knowledge tagging result the baseline. 

## 4.3 GENOME ON FEW-SHOT TASK LEARNING. 

As a general module learning framework, our model is not only able to learn new modules to handle existing tasks but also can learn to handle new visual reasoning tasks from a few training examples. We evaluate such abilities on new tasks, Raven (Zhang et al., 2019) and MEWL (Jiang et al., 2023). Specifically, we first prompt the LLM to learn pattern recognition modules for visual understanding and then ask the LLM to generate a solver module to handle the task. The instances of our model prediction are shown in Figure 5 and Figure 6. Note that visual reasoning from Raven is widely used in intelligent testing for humans, which shows our model’s strong capabilities and potential. We report the performance of our model and the baselines in Table 3 and Table 4. Our model is significantly better than previous fully-supervised methods like ResNet+DRT (Zhang et al., 2019) and Aloe (Ding et al., 2021), showing its effectiveness. Note that all these models ResNet+DST, ALANS-V (Zhang et al., 2022), Aloe (Ding et al., 2021) and Flamingo (Alayrac et al., 2022) are models fully-finetuned on in-domain data, while our GENOME is a general few-shot framework to learn modules for problem-solving. Moreover, we can observe the new compositionality and module re-usage from Figure 8 of the Appendix. Although the SOLVER module was originally learned from center-type problems, it can be naturally transferred to other types like left-right and up-down. 

## 4.4 ABLATIONS 

To gauge the efficacy of our model, we conducted a series of ablation studies addressing the following key inquiries: **Q1** How effective is module learning? **Q2** What impact does the quantity of training examples have on model performance? **Q3** How crucial is the LLM’s capability for optimal performance? In our experiments, _GENOME w/o ML_ represents a configuration without any new module learn- 

||Methods<br>GENOME w/o ML<br>GENOME-WLM<br>GENOME (10)<br>GENOME (50)<br>GENOME (100)|RefCOCO<br>62.3<br>64.4<br>49.4<br>67.0<br>67.1|
|---|---|---|



Table 5: Ablation study of GENOME on RefCOCO. 

8 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules_images/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules.pdf-0009-01.png)


**----- Start of picture text -----**<br>
RAVEN<br>Problem Matrix Answer Set<br>IMAGE: ProblemMatrix COLOR=DETECT_COLOR(image=IMAGE)SHAPE=DETECT_SHAPE(image=IMAGE)SIZE=DETECT_SIZE(image=IMAGE) ... color: 84shape: ‘triangle’size: 56color: 252shape: ‘square’size: 32 ... color: 140shape: ‘hexagon’size: 72color: 140shape: ‘triangle’size: 40<br>color_type: arithmetic<br>shape_type: constant<br>AnswerSet 1 2 3 4 ANSWER=SOLVER(image=IMAGE,color=COLOR,shape=SHAPE,size=SIZE)FINAL_RESULT=RESULT(var=ANSWER) size_type: progression<br>5 6 7 8 ANSWER: 3<br>**----- End of picture text -----**<br>


Figure 5: A qualitative example from the Raven dataset (Zhang et al., 2019) is provided. This task involves a set of images with varying visual attributes, such as colors, shapes, and locations. Models are tasked with identifying the image that best matches the missing item in the Problem Matrix. GENOME exhibits the capability to compose modules (i.e. DETECT ~~S~~ HAPE and SOLVER) for detecting these attribute rules and constructing a solver module to address the task. The correct answer is indicated by a green box. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules_images/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules.pdf-0009-03.png)


**----- Start of picture text -----**<br>
MEWL<br>Context<br>Query COLOR=DETECT_COLOR(image=IMAGE) … …<br>IMAGE: cyan gray cyan<br>cyan gray green cyan<br>ANSWER=SOLVER(image=IMAGE,color=COLOR,<br>WORD: atlecacnoNew Wordsercan ercanacno atlec Optionsacnopreau ercanisageatlec word=WORD)FINAL_RESULT=RESULT(var=ANSWER) acno atlec ercanANSWER: acnocyan acno<br>**----- End of picture text -----**<br>


Figure 6: A qualitative illustration from the MEWL dataset (Jiang et al., 2023) is presented. This task entails a set of images featuring diverse visual attributes, such as material and shapes, and it necessitates models to determine the word that corresponds to the query image. GENOME demonstrates the capability to generate modules for identifying these attribute rules and composing a solver module to address the task. The correct answer is indicated by a green box. 

ing but relies heavily on ViperGPT and VisProg-defined modules, directing the LLM to pinpoint a region matching the referring expression. On the other hand, _GENOME-WLM_ replaces the gpt-3.5-turbo-instruct API with WizardCoder-Python-34B-V1.0 from WizardLM (Xu et al., 2023a). The designations _GENOME (10)/ (50) / (100)_ indicate models trained with 10, 50, and 100 examples, respectively. For resource constraints, we limited our experimentation to 800 RefCOCO samples. 

Table 5 presents the outcomes, leading to these insights: module learning, given sufficient test instances, can bolster task performance ( **Q1** addressed). A paucity of training examples, such as 10 for RefCOCO, might induce overfitting, but this diminishes with increased training data (50 examples), improving overall performance ( **Q2** addressed). Finally, model performance appears intrinsically tied to the LLM’s capacity, with superior LLMs delivering enhanced results ( **Q3** addressed). 

## 5 CONCLUSION 

In this study, we introduce GENOME, which is designed to tackle visual reasoning tasks when confronted with limited training data. This approach combines language models to parse natural language into executable operations and create specialized visual modules tailored to the given task. Our model exhibits competitive performance on conventional tasks, effortless transfer of acquired 

9 

Preprint 

modules to novel tasks, and the capability to adapt to new tasks even with limited training data. Our GENOME also proposes numerous avenues for future research. Firstly, it still necessitates taskspecific prompts for each distinct reasoning task, and it would be intriguing to explore the use of a universal prompt for all tasks. Secondly, the framework can be extended to encompass a broader range of multi-modal reasoning tasks, incorporating diverse inputs such as audio, video, and tactile information. 

## REFERENCES 

- Gradient-based learning applied to document recognition. _Proceedings of the IEEE_ , 86(11):2278– 2324, 1998. 

- Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _arXiv_ , 2022. 

- Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. _arXiv preprint arXiv:1905.13319_ , 2019. 

- Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 6077–6086, 2018. 

- Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Learning to compose neural networks for question answering. _arXiv preprint arXiv:1601.01705_ , 2016a. 

- Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In _CVPR_ , pp. 39–48, 2016b. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

James F. Brul´e and Alexander Blount. Knowledge acquisition. 1989. URL https://api. semanticscholar.org/CorpusID:18663796. 

- Jannis Bulian, Christian Buck, Wojciech Gajewski, Benjamin Boerschinger, and Tal Schuster. Tomayto, tomahto. beyond token-level answer equivalence for question answering evaluation, 2022. 

- Henry R Burke. Raven’s progressive matrices (1938): More on norms, reliability, and validity. _Journal of Clinical Psychology_ , 41(2):231–235, 1985. 

Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. Large language models as tool makers. _arXiv_ , 2023. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021a. 

- Wenhu Chen, Zhe Gan, Linjie Li, Yu Cheng, William Wang, and Jingjing Liu. Meta module network for compositional visual reasoning. _Proceedings of WACV_ , 2021b. 

- Xinyun Chen, Maxwell Lin, Nathanael Sch¨arli, and Denny Zhou. Teaching large language models to self-debug. _arXiv preprint arXiv:2304.05128_ , 2023a. 

- Zhenfang Chen, Peng Wang, Lin Ma, Kwan-Yee K Wong, and Qi Wu. Cops-ref: A new dataset and task on compositional referring expression comprehension. In _CVPR_ , 2020. 

- Zhenfang Chen, Jiayuan Mao, Jiajun Wu, Kwan-Yee Kenneth Wong, Joshua B Tenenbaum, and Chuang Gan. Grounding physical concepts of objects and events through dynamic visual reasoning. In _ICLR_ , 2021c. 

10 

Preprint 

- Zhenfang Chen, Kexin Yi, Yunzhu Li, Mingyu Ding, Antonio Torralba, Joshua B Tenenbaum, and Chuang Gan. Comphy: Compositional physical reasoning of objects and events from videos. _arXiv_ , 2022. 

- Zhenfang Chen, Qinhong Zhou, Yikang Shen, Yining Hong, Hao Zhang, and Chuang Gan. See, think, confirm: Interactive prompting between vision and language models for knowledge-based visual reasoning. _arXiv_ , 2023b. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- David Ding, Felix Hill, Adam Santoro, Malcolm Reynolds, and Matt Botvinick. Attention over learned object embeddings enables complex visual reasoning. _Advances in neural information processing systems_ , 34:9112–9124, 2021. 

- Mingyu Ding, Yan Xu, Zhenfang Chen, David Daniel Cox, Ping Luo, Joshua B Tenenbaum, and Chuang Gan. Embodied concept learner: Self-supervised learning of concepts and mapping through instruction following. In _CoRL_ , 2023. 

- Xin Luna Dong, Seungwhan Moon, Yifan Ethan Xu, Kshitiz Malik, and Zhou Yu. Towards nextgeneration intelligent assistants leveraging llm techniques. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pp. 5792–5793, 2023. 

- Kevin Ellis, Lionel Wong, Maxwell Nye, Mathias Sable-Meyer, Luc Cary, Lore Anaya Pozo, Luke Hewitt, Armando Solar-Lezama, and Joshua B Tenenbaum. Dreamcoder: growing generalizable, interpretable knowledge with wake–sleep bayesian program learning. _Philosophical Transactions of the Royal Society A_ , 381(2251):20220050, 2023. 

- Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2017. 

- Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. _ArXiv_ , abs/2211.11559, 2022. 

- Harry Frederick Harlow. The formation of learning sets. _Psychological review_ , 56 1:51–65, 1949. URL https://api.semanticscholar.org/CorpusID:22804426. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_ , 2020. 

- Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. _Neural computation_ , 9(8): 1735–1780, 1997. 

- Drew A Hudson and Christopher D Manning. Compositional attention networks for machine reasoning. _arXiv preprint arXiv:1803.03067_ , 2018. 

- Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _CVPR_ , 2019. 

- Guangyuan Jiang, Manjie Xu, Shiji Xin, Wei Liang, Yujia Peng, Chi Zhang, and Yixin Zhu. Mewl: Few-shot multimodal word learning with referential uncertainty. In _ICML_ , 2023. 

- Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In _Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)_ , pp. 787–798, 2014. 

- Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. _arXiv:2304.02643_ , 2023. 

11 

Preprint 

Brenden M. Lake, Tomer D. Ullman, Joshua B. Tenenbaum, and Samuel J. Gershman. Building machines that learn and think like people, 2016. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv_ , 2023. 

- Liunian Harold Li*, Pengchuan Zhang*, Haotian Zhang*, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. In _CVPR_ , 2022. 

- Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B. Tenenbaum, and Jiajun Wu. The NeuroSymbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision. In _ICLR_ , 2019a. 

- Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B Tenenbaum, and Jiajun Wu. The neurosymbolic concept learner: Interpreting scenes, words, and sentences from natural supervision. In _ICLR_ , 2019b. 

- Tom Michael Mitchell, Richard M. Keller, and Smadar T. Kedar-Cabelli. Explanation-based generalization: A unifying view. _Machine Learning_ , 1:47–80, 1986. URL https://api. semanticscholar.org/CorpusID:117264. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems_ , 35: 27730–27744, 2022. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 

Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. _ICCV_ , 2021. 

- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 10684–10695, 2022. 

- Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. _arXiv_ , 2023. 

- Kurt Shuster, Jing Xu, Mojtaba Komeili, Da Ju, Eric Michael Smith, Stephen Roller, Megan Ung, Moya Chen, Kushal Arora, Joshua Lane, et al. Blenderbot 3: a deployed conversational agent that continually learns to responsibly engage. _arXiv_ , 2022. 

- D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. _arXiv_ , 2023. 

- Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. _arXiv preprint arXiv:2304.12244_ , 2023a. 

- Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu, Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping Luo. Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models. _arXiv_ , 2023b. 

- Jianwei Yang, Jiayuan Mao, Jiajun Wu, Devi Parikh, David D Cox, Joshua B Tenenbaum, and Chuang Gan. Object-centric diagnosis of visual reasoning. _arXiv preprint arXiv:2012.11587_ , 2020. 

- Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. An empirical study of gpt-3 for few-shot knowledge-based vqa. In _AAAI_ , 2022. 

12 

Preprint 

- Kexin Yi, Jiajun Wu, Chuang Gan, Antonio Torralba, Pushmeet Kohli, and Josh Tenenbaum. Neuralsymbolic vqa: Disentangling reasoning from vision and language understanding. In _NeurIPS_ , 2018. 

- Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14_ , pp. 69–85. Springer, 2016. 

- Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and Tamara L Berg. Mattnet: Modular attention network for referring expression comprehension. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 1307–1315, 2018. 

- Yan Zeng, Xinsong Zhang, and Hang Li. Multi-grained vision language pre-training: Aligning texts with visual concepts. _arXiv preprint arXiv:2111.08276_ , 2021. 

- Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Raven: A dataset for relational and analogical visual reasoning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 5317–5327, 2019. 

- Chi Zhang, Baoxiong Jia, Song-Chun Zhu, and Yixin Zhu. Abstract spatial-temporal reasoning via probabilistic abduction and execution. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 

- Chi Zhang, Sirui Xie, Baoxiong Jia, Ying Nian Wu, Song-Chun Zhu, and Yixin Zhu. Learning algebraic representation for systematic generalization in abstract reasoning. In _European Conference on Computer Vision_ , pp. 692–709. Springer, 2022. 

13 

Preprint 

## A APPENDIX 

In this section, we substantiate our claims in the paper by providing additional implementation details (Section A.1), exemplar prompts for each stage (Section A.2), details on dataset collection (Section A.3), qualitative examples of new learned modules (Section A.4). 

## A.1 IMPLEMENATION DETAILS 

**Pre-defined Modules and API models.** The success of our model still requires a set of predefined APIs. Following the modules in VisProg and ViperGPT, we adopt the following APIs. We adopt GLIP (Li* et al., 2022) for object localization. We adopt _gpt-3.5-turbo-instruct_ from OpenAI and _WizardCoder-Python-34B-V1.0_ from WizardLM for code generation. We use BLIP (Li et al., 2023) for answering simple questions about images. We use CLIP (Radford et al., 2021) and X-VLM (Zeng et al., 2021) for image-text classification. We use MiDaS (Ranftl et al., 2021) for estimating depths in images. We use stable diffusion (Rombach et al., 2022) for modifying image patches. Based on these APIs, we construct a set of pre-defined modules following VisProg. These pre-defined modules will be used to cooperate with the new learned modules for visual reasoning. 

|Descriptions|Modules|
|---|---|
|||
|Image Understanding<br>Image Manipulation<br>Others|**Loc**for object location,**FaceDet**for face detection,**Select**and<br>**Filter**<br>~~**P**~~**roperty**for image-text classifcation. **Filter**<br>~~**S**~~**patial**for<br>selecting image regions;<br>**Replace**for image editing,**colorPop**for changing images colors,<br>**BgBlur**for blurring background,**Tag**for annotating box regions<br>and**Emoji**for face tagging. **Crop**and its variants for cropping<br>patches from the images.<br>**List**for retrieving factual knowledge,**Count**for counting object<br>numbers,**Eval**,**Result**,**BOX2MASK**and**MASK2BOX**for<br>formatting outputs.|



Table 6: Pre-defined Modules used in GENOME. 

## A.2 PROMPTS FOR EACH STAGE. 

The ability of Our GENOME is from in-context learning of LLMs (Brown et al., 2020), when the prompts are keys to tell what the LLM should generate. We show the exemplar prompts of our models to learn the VQA task in Figure 15-17. 

## A.3 DETAILS AND EXAMPLES OF THE NEW DATASETS. 

To evaluate Knowledge Tagging, 50 tagging instructions are annotated on 50 internet images including personalities and a variety of objects such as logos, flowers, buildings, fruits and sports, among others. For each instruction, we manually annotated the ground truth bounding box and the associated tag. For image editing assessment, we collected 50 editing instructions on 50 images including personalities and various objects like foods, furniture, animals, utensils etc. 25 images are from the COCO dataset and the other 25 images are from the internet. For the image editing tasks, we ask three annotators to estimate whether the editing is correct or not. For the knowledge tagging task, we consider the localization is correct if the detected region has an IoU higher 0.5 with the ground-truth annotation. For text tagging, we compare the prediction with the annotated text with BERT matching (BEM) (Bulian et al., 2022). If the matching score is higher than 0.5, we consider it a successful matching. More examples of the two datasets can be found at Figure 9 and Figure 10. 

## A.4 QUALITATIVE EXAMPLES. 

In this subsection, we show the qualitative examples of the learned modules and qualitative cases of how they handle different tasks. We show an example of GENOME performs better than VisProg 

14 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules_images/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules.pdf-0015-01.png)


**----- Start of picture text -----**<br>
Knowledge Tagging<br>Ours OBJ2: OBJ3: , . . . ,<br>Generated Program: COMPARE_COLOR OBJ3<br>Question: Tag common fruits and  OBJ0=LOC(image=IMAGE,object='fruit or vegetable’)LIST0=LIST(query='fruits and vegetables',max=20) OBJ4: ,<br>vegetables of the same color as the grape. OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)<br>OBJ2=CLASSIFY(image=IMAGE,object=OBJ0,categories='grape’)<br>IMAGE: OBJ3=REDUCE_MASK(mask_list1=OBJ1,mask_list2=OBJ2)<br>OBJ4=META_COMPARE(function_name='COMPARE_COLOR',image=IMAGE,obj_list=<br>OBJ3,obj_cmp=OBJ2,name1='fruit or vegetable’,name2  FINAL_RESULT:<br>='grape',attribute='same’)<br>IMAGE0=TAG(image=IMAGE,object=OBJ4)<br>FINAL_RESULT=RESULT(var=IMAGE0)<br>VisProg OBJ1: FINAL_RESULT:<br>Generated Program:<br>OBJ0=LOC(image=IMAGE,object='grape’)<br>LIST0=LIST(query='common fruits and vegetables of the same color as<br>the grape',max=20)<br>OBJ1=CLASSIFY(image=IMAGE,object=OBJ0,categories=LIST0)<br>IMAGE0=TAG(image=IMAGE,object=OBJ1)<br>FINAL_RESULT=RESULT(var=IMAGE0)<br>**----- End of picture text -----**<br>


Figure 7: A typical example of how our GENOME outperforms VisProg on knowledge tagging. In the top, our model is able to make use of the _COMPARE_ ~~_C_~~ _OLOR_ module learned from GQA to localize the correct region while VisProg fail to generate the correct program with its fixed module library. 

in Figure 7. At the top of Figure 7, our model effectively utilizes the COMPARE ~~C~~ OLOR module acquired from GQA to pinpoint the correct region, whereas VisProg fails to generate the correct program due to its rigid module library. Figure 8 highlights emerging forms of compositionality and module re-usage. Notably, although the _SOLVER_ module was originally trained on center-type problems within the Raven dataset, it demonstrates inherent adaptability to other problem types, including left-right and up-down orientations. 

**New Learned Modules.** We show the examplar new learned modules from the GQA and RefCOCO in Figure 11-14. As shown in Figure 11, the new learned module ( CHOOSE ~~A~~ TTRIBUTE) is able to use the LLM to retrieve relevant knowledge first and then adopt the image-text classifier to match the attributes. In Figure 13-14, we see that the new module SORT ~~S~~ PATIAL is able to localize objects with spatial index. 

15 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules_images/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules.pdf-0016-01.png)


**----- Start of picture text -----**<br>
Center Left-Right Up-Down<br>IMAGE: IMAGE: IMAGE:<br>Problem Problem Problem<br>Matrix Matrix Matrix<br>Answer Answer Answer<br>Set Set Set<br>Generated Program: Generated Program: Generated Program:<br>BOX0=LOC(image=IMAGE,object=‘LEFT’) BOX0=LOC(image=IMAGE,object=‘TOP’)<br>COLOR=DETECT_COLOR(image=IMAGE) IMAGE0=CROP(image=IMAGE,box=BOX0) IMAGE0=CROP(image=IMAGE,box=BOX0)<br>SHAPE=DETECT_SHAPE(image=IMAGE) BOX1=LOC(image=IMAGE,object=‘RIGHT’) BOX1=LOC(image=IMAGE,object=‘BOTTOM’)<br>SIZE=DETECT_SIZE(image=IMAGE)ANSWER= SOLVER (image=IMAGE,color=COLOR, IMAGE1=CROP(image=IMAGE,box=BOX1)COLOR=DETECT_COLOR(image0=IMAGE0,image1=IMAGE1) IMAGE1=CROP(image=IMAGE,box=BOX1)COLOR=DETECT_COLOR(image0=IMAGE0,image1=IMAGE1)<br>shape=SHAPE,size=SIZE) SHAPE=DETECT_SHAPE(image0=IMAGE0,image1=IMAGE1) SHAPE=DETECT_SHAPE(image0=IMAGE0,image1=IMAGE1)<br>FINAL_RESULT=RESULT(var=ANSWER) SIZE=DETECT_SIZE(image0=IMAGE0,image1=IMAGE1) SIZE=DETECT_SIZE(image0=IMAGE0,image1=IMAGE1)<br>ANSWER= SOLVER (image0=IMAGE0,image1=IMAGE1,color ANSWER= SOLVER (image0=IMAGE0,image1=IMAGE1,color<br>=COLOR,shape=SHAPE,size=SIZE) =COLOR,shape=SHAPE,size=SIZE)<br>FINAL_RESULT=RESULT(var=ANSWER) FINAL_RESULT=RESULT(var=ANSWER)<br>**----- End of picture text -----**<br>


Figure 8: New compositionality and module re-usage in the Raven dataset. While the SOLVER module was initially trained on center-type problems in the Raven dataset, it exhibits a natural transferability to other types, such as left-right and up-down problems. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules_images/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules.pdf-0016-03.png)


**----- Start of picture text -----**<br>
Replace the spoon of<br>Create a color pop of the first boat  Create a color pop of the  the same material as<br>from the front first child from the right the spatula with a knife<br>Replace the second  Select the lamp with the same<br>Hide Tim Robbins with ;) and  pizza from the top with  color as the one at the bottom<br>Morgan Freeman with 8) a hamburger and create a color pop<br>**----- End of picture text -----**<br>


Figure 9: More examples of the new image edit dataset. The dataset asks models to edit images’ fine-grained and regional details according to diverse language instructions. 

16 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules_images/iclr_2024_GENOME_Generative_Neuro-Symbolic_Visual_Reasoning_by_Growing_and_Reusing_Modules.pdf-0017-01.png)


**----- Start of picture text -----**<br>
Tag the famous landmark of Europe  Tag the common bird of  Tag the second famous film<br>in the bottom right the same color as the ibis director from the left<br>Tag the famous painting of the  Tag the second dog from  Tag the second Nobel Laureate<br>Louvre in the left the right in Physics from the left<br>**----- End of picture text -----**<br>


Figure 10: More examples of the new knowledge tagging dataset. The dataset requires models to localize the target region and tag the region with the desired information. 

17 

Preprint 

1 **class CHOOSE_ATTRIBUTE** (): 2 _"""_ 3 _Input:_ 4 _image: an image object_ 5 _box: a list of bounding boxes_ 6 _object: a string_ 7 _attribute1: a string_ 8 _attribute2: a string_ 9 _Output:_ 10 _result: a string_ 11 _Examples:_ 12 _Question: Is the coat thick or thin?_ 13 _BOX0=LOC(image=IMAGE,object='coat')_ 14 _ANSWER0=CHOOSE_ATTRIBUTE(image=IMAGE,box=BOX0,object='coat',_ 15 _attribute1='thick',attribute2='thin')_ 16 _FINAL_RESULT=RESULT(var=ANSWER0)_ 17 _"""_ 18 step_name = 'CHOOSE_ATTRIBUTE' 19 20 **def** __init__(self): 21 print(f'Registering **{** self.step_name **}** step') 22 23 **def** expand_box(self,box,img_size,factor=1.5): 24 W,H = img_size 25 x1,y1,x2,y2 = box 26 dw = int(factor*(x2-x1)/2) 27 dh = int(factor*(y2-y1)/2) 28 cx = int((x1 + x2) / 2) 29 cy = int((y1 + y2) / 2) 30 x1 = max(0,cx - dw) 31 x2 = min(cx + dw,W) 32 y1 = max(0,cy - dh) 33 y2 = min(cy + dh,H) 34 **return** [x1,y1,x2,y2] 35 36 **def** predict(self,img,boxes,obj,attr1,attr2): 37 **if** len(boxes) > 0: 38 box = boxes[0] 39 box = self.expand_box(box, img.size) 40 out_img = img.crop(box) 41 **else** : 42 out_img = img 43 prompt1 = f'Tell me the attributes when the **{** obj **}** is **{** attr1 **}** in 44 one sentence.' 45 prompt2 = f'Tell me the attributes when the **{** obj **}** is **{** attr2 **}** in 46 one sentence.' 47 obj_desc1 = API.gpt3(prompt1, 'gpt3_general') 48 obj_desc2 = API.gpt3(prompt2, 'gpt3_general') 49 result1 = API.clip(out_img,obj_desc1) 50 result2 = API.clip(out_img,obj_desc2) 51 **if** result1 > result2: 52 result = attr1 53 **else** : 54 result = attr2 55 **return** result 56 57 **def** execute(self,img,boxes,obj,attr1,attr2): 58 result = self.predict(img,boxes,obj,attr1,attr2) 59 **return** result 

Figure 11: Exemplar generated module from the GQA dataset. This automatically constructed module can make use of different APIs to compare attributes of an image region. 

18 

Preprint 

1 **class COMPARE_COLOR** (): 2 _"""_ 3 _Input:_ 4 _image: an image object_ 5 _box1: a list of bounding boxes_ 6 _box2: a list of bounding boxes_ 7 _object1: a string_ 8 _object2: a string_ 9 _compare_type: a string_ 10 _Output:_ 11 _result: a string_ 12 _"""_ 13 **def** expand_box(self,box,img_size,factor=1.5): 14 W,H = img_size 15 x1,y1,x2,y2 = box 16 dw = int(factor*(x2-x1)/2) 17 dh = int(factor*(y2-y1)/2) 18 cx = int((x1 + x2) / 2) 19 cy = int((y1 + y2) / 2) 20 x1 = max(0,cx - dw) 21 x2 = min(cx + dw,W) 22 y1 = max(0,cy - dh) 23 y2 = min(cy + dh,H) 24 **return** [x1,y1,x2,y2] 25 **def** predict(self,img,boxes1,boxes2,obj1,obj2,compare_type): 26 **if** len(boxes1) > 0: 27 box1 = boxes1[0] 28 box1 = self.expand_box(box1,img.size) 29 out_img1 = img.crop(box1) 30 **else** : 31 out_img1 = img 32 **if** len(boxes2) > 0: 33 box2 = boxes2[0] 34 box2 = self.expand_box(box2,img.size) 35 out_img2 = img.crop(box2) 36 **else** : 37 out_img2 = img 38 color1 = API.vqa(out_img1, f'What color is the **{** obj1 **}** ?') 39 color2 = API.vqa(out_img2, f'What color is the **{** obj2 **}** ?') 40 prompt = f'Can the **{** color1 **}** be regarded as the same color as' 41 f' **{** color2 **}** ? You should just reply yes or no without any other 42 words.' 43 temp = API.gpt3(prompt, 'gpt3_general') 44 **if** 'same' == compare_type: 45 **if** 'yes' **in** temp.lower(): 46 result = 'yes' 47 **elif** 'no' **in** temp.lower(): 48 result = 'no' 49 **elif** 'different' == compare_type: 50 **if** 'yes' **in** temp.lower(): 51 result = 'no' 52 **elif** 'no' **in** temp.lower(): 53 result = 'yes' 54 **else** : 55 **if** 'yes' **in** temp.lower(): 56 result = 'yes' 57 **elif** 'no' **in** temp.lower(): 58 result = 'no' 59 **return** result 60 **def** execute(self,img,boxes1,boxes2,obj1,obj2,compare_type): 61 result = self.predict(img,boxes1,boxes2,obj1,obj2,compare_type) 62 **return** result 

Figure 12: Exemplar generated module from the GQA dataset. 

19 

Preprint 

1 **class SORT_SPATIAL** (): 2 _"""_ 3 _Select objects from the image that match the spatial location._ 4 _Objects are represented by the bounding boxes._ 5 _Returns the bounding boxes that satisfie the condition._ 6 _Input:_ 7 _image: raw PIL image_ 8 _box_list: a list of unormalized bounding boxes_ 9 _location: the location can only be left, middle, right, top,_ 10 _bottom, front and behind_ 11 _index: a number for the rank the object_ 12 _Output:_ 13 _box: a bounding box_ 14 _Examples:_ 15 _Question: second sandwich from the right on the bottom_ 16 _BOXLIST0=LOC(image=IMAGE,object='sandwich')_ 17 _BOXLIST1=SORT_SPATIAL(image=IMAGE,box_list=BOXLIST0,location=_ 18 _'right',index=2)_ 19 _BOXLIST2=SORT_SPATIAL(image=IMAGE,box_list=BOXLIST1,location=_ 20 _'bottom',index=1)_ 21 _FINAL_RESULT=RESULT(var=BOXLIST2)_ 22 _"""_ 23 step_name = 'SORT_SPATIAL' 24 **def** predict(self,img,box_list,location,index): 25 **if** index < 0 **or** index > len(box_list): 26 **return** [] 27 **if** index == 0: 28 **return** [box_list[0]] 29 **if** "front" **in** location **or** "behind" **in** location: 30 box_depth_list = self.parse_depth(img, box_list) 31 box_list_sorted = sorted(box_depth_list, key= **lambda** x: x[1]) 32 out_box_list = [box_i[0] **for** box_i **in** box_list_sorted] 33 **if** "behind" **in** location: 34 out_box_list.reverse() 35 **else** : 36 **if** "left" **in** location: 37 box_list = sorted(box_list, key= **lambda** x: x[0]) 38 **elif** "right" **in** location: 39 box_list = sorted(box_list, key= **lambda** x: x[2], reverse 40 = **True** ) 41 **elif** "top" **in** location: 42 box_list = sorted(box_list, key= **lambda** x: x[1]) 43 **elif** "bottom" **in** location: 44 box_list = sorted(box_list, key= **lambda** x: x[3], reverse 45 = **True** ) 46 **else** : 47 **return** [] 48 **if** index > len(box_list): 49 **return** [] 50 out_box_list = [box_list[index-1]] 51 **return** out_box_list 52 **def** check_location(self,img,box,location): 53 w, h = img.size 54 x1, y1, x2, y2 = box 55 cx = (x1 + x2) / 2 56 cy = (y1 + y2) / 2 57 **if** 'left' **in** location: 58 **if** cx > w / 2: 59 **return False** 

Figure 13: Exemplar generated module from the RefCOCO dataset. The rest part of the code is in Figure 14. 

20 

Preprint 

1 **elif** 'right' **in** location: 2 **if** cx < w / 2: 3 **return False** 4 **if** 'top' **in** location: 5 **if** cy > h / 2: 6 **return False** 7 **elif** 'bottom' **in** location: 8 **if** cy < h / 2: 9 **return False** 10 **return True** 11 12 **def** parse_depth(self,img,box_list): 13 box_depth_list = [] 14 _# compute depths for front or background_ 15 depth_map = API.depth(img) 16 **for** box **in** box_list: 17 x1, y1, x2, y2 = box 18 depth_map = np.array(depth_map) 19 avg_depth = np.median(depth_map[x1:x2, y1:y2]) 20 box_depth_list.append((box, avg_depth)) 21 **return** box_depth_list 22 23 **def** execute(self,img,box_list,location,index): 24 **return** self.predict(img,box_list,location,index) 

Figure 14: Exemplar generated module from the RefCOCO dataset. The former part of the code is in Figure 13. This generated module is able to localize objects based on their location in images and the depth of images. 

21 

Preprint 

1 Pre-defined Modules: 2 **class LOC** (): 3 _"""_ 4 _Generate boxes of the object on the image._ 5 _Input:_ 6 _image: an image object_ 7 _object: an object string_ 8 _Output:_ 9 _box: a list of bounding boxes_ 10 _Examples:_ 11 _BOX0=LOC(image=IMAGE,object='camel')_ 12 _"""_ 13 **class COUNT** (): 14 _"""_ 15 _Count the number of boxes in the list._ 16 _Input:_ 17 _box: a list of bounding boxes_ 18 _Output:_ 19 _number: number of boxes_ 20 _Examples:_ 21 _ANSWER0=COUNT(box=BOX1)_ 22 _"""_ 23 Suppose you are a program expert. Given a set of pre-defined modules, 24 could you identify whether it **is** possible to write a program to get the 25 answer to the question? If **not** , what new modules do we need? 26 Note that you can only use the below pre-defined modules: 27 LOC, COUNT, CROP ....... 28 29 Question: Is the purse to the left **or** to the right of the person? 30 Yes. The program **is** : 31 BOX0=LOC(image=IMAGE,object='person') 32 IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0) 33 BOX1=LOC(image=IMAGE0,object='purse') 34 ANSWER0=COUNT(box=BOX1) 35 ANSWER1=EVAL(expr=f"'left' if **{** ANSWER0 **}** > 0 else 'right'") 36 FINAL_RESULT=RESULT(var=ANSWER1) 37 38 Question: Which object **is** larger, the sphere **or** the blue cube? 39 No. We need to make a new module "COMPARE_SIZE" first. Here **is** the header 40 of the class: 41 **class COMPARE_SIZE** (): 42 _"""_ 43 _Compare the size of two objects in the image._ 44 _One object is identified by the first bounding box of box0_ 45 _Another object is identified by the first bounding box of box1_ 46 _Input:_ 47 _image: an image object_ 48 _box0: a list of bounding boxes_ 49 _box1: a list of bounding boxes_ 50 _Output:_ 51 _flag: return True if first object is larger else False_ 52 _Examples:_ 53 _Question: Which object is larger, the sphere or the blue cube?_ 54 _BOX0=LOC(image=IMAGE,object='sphere')_ 55 _BOX1=LOC(image=IMAGE,object='blue cube')_ 56 _FLAG0=COMPARE_SIZE(image=IMAGE,box0=BOX0,box1=BOX1)_ 57 _ANSWER2=EVAL(expr=f"'sphere' if {FLAG0} else 'blue cube'")_ 58 _FINAL_RESULT=RESULT(var=ANSWER)_ 59 _"""_ 60 ....... 61 Question: __INSERT_NEW_QUESTION__ 

Figure 15: Prompt of module initialization (Stage 1) to propose the module candidates. 

22 

Preprint 

1 Pre-defined APIs: 2 **class API** (): 3 **def** loc(cls, image: Image.Image, obj_name: str) -> List[List[int]]: 4 _"""_ 5 _Returns boxes of the specific object on the image._ 6 _Input:_ 7 _image: an image object_ 8 _obj_name: a text string_ 9 _Output:_ 10 _boxes: a list of bounding boxes_ 11 _Example:_ 12 _[[261, 160, 525, 299]] = API.loc(image, "camel")_ 13 _"""_ 14 **return** boxes 15 **def** gpt3(cls, prompt: str) -> str: 16 _"""_ 17 _Returns response from gpt3_ 18 _Input:_ 19 _prompt: prompt to gpt3_ 20 _Output:_ 21 _response: gpt3 response_ 22 _Example:_ 23 _"joe biden" = API.gpt3("Who is the US President?")_ 24 _"""_ 25 **return** response 26 ....... 27 28 Given the above APIs, could you write a new module, Loc()? 29 **class Loc** (): 30 _"""_ 31 _Generate boxes of the object on the image._ 32 _Input:_ 33 _img: an image object_ 34 _obj_name: an object string_ 35 _Output:_ 36 _selected_boxes: a list of bounding boxes_ 37 _Examples:_ 38 _BOX1=LOC(image=IMAGE0,object='food')_ 39 _"""_ 40 Response: 41 **class Loc** (): 42 step_name = 'LOC' 43 **def** execute(self,img,obj_name): 44 **if** obj_name=='TOP': 45 bboxes = [self.top_box(img)] 46 **elif** obj_name=='BOTTOM': 47 bboxes = [self.bottom_box(img)] 48 **elif** obj_name=='LEFT': 49 bboxes = [self.left_box(img)] 50 **elif** obj_name=='RIGHT': 51 bboxes = [self.right_box(img)] 52 **else** : 53 bboxes = self.predict(img,obj_name) 54 **return** bboxes 55 56 **def** predict(self,img,obj_name): 57 **return** API.loc(img,obj_name) 58 ....... 59 Given the above APIs, could you write a new module, __MODULE_NAME__? 60 __MODULE_HEAD__ 

Figure 16: Prompt of module generation (Stage 2) to make a module based on the module’s input and output. 

23 

Preprint 

1 Think step by step to answer the question. 2 3 You can only use modules below: 4 LOC 5 COUNT 6 EVAL 7 RESULT 8 VERIFY_ATTRIBUTE 9 VERIFY_COLOR 10 VERIFY_MATERIAL 11 ....... 12 13 Question: Is the vehicle **in** the top of the image? 14 Program: 15 BOX0=LOC(image=IMAGE,object='TOP') 16 IMAGE0=CROP(image=IMAGE,box=BOX0) 17 BOX1=LOC(image=IMAGE0,object='vehicle') 18 ANSWER0=COUNT(box=BOX1) 19 ANSWER1=EVAL(expr=f"'yes' if **{** ANSWER0 **}** > 0 else 'no'") 20 FINAL_RESULT=RESULT(var=ANSWER1) 21 22 Question: Who **is** carrying the umbrella? 23 Program: 24 BOX0=LOC(image=IMAGE,object='umbrella') 25 IMAGE0=CROP(image=IMAGE,box=BOX0) 26 ANSWER0=VQA(image=IMAGE0,question='Who is carrying the umbrella?') 27 FINAL_RESULT=RESULT(var=ANSWER0) 28 29 Question: Do the towel **and** the box have a different colors? 30 Program: 31 BOX0=LOC(image=IMAGE,object='towel') 32 BOX1=LOC(image=IMAGE,object='box') 33 ANSWER0=COMPARE_ATTRIBUTE(image=IMAGE,box1=BOX0,box2=BOX1,object1='towel' 34 ,object2='box',attribute='color',question=QUESTION) 35 FINAL_RESULT=RESULT(var=ANSWER0) 36 37 Question: Is the knife made of ceramic? 38 Program: 39 BOX0=LOC(image=IMAGE,object='knife') 40 ANSWER0=VERIFY_MATERIAL(image=IMAGE,box=BOX0,material='ceramic',object= 41 'knife',question=QUESTION) 42 ANSWER1=EVAL(expr=f"'yes' if **{** ANSWER0 **}** else 'no'") 43 FINAL_RESULT=RESULT(var=ANSWER1) 44 45 Question: Is the coat thick **or** thin? 46 Program: 47 BOX0=LOC(image=IMAGE,object='coat') 48 ANSWER0=CHOOSE_ATTRIBUTE(image=IMAGE,box=BOX0,object='coat',attribute1= 49 'thick',attribute2='thin') 50 FINAL_RESULT=RESULT(var=ANSWER0) 51 ....... 52 53 Question: __INSERT_NEW_QUESTION__ 54 Program: 

Figure 17: Prompt of module execution (Stage 3) to parse programs for a new test case. 

24 

