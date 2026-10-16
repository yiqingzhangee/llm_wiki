
![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0001-00.png)


## **Guiding Mathematical Reasoning via Mastering Commonsense Formula Knowledge** 

## Zhiyuan Ma 

## Zhenya Huang[∗] 

## Jiayu Liu 

School of Data Science, University of Science and Technology of China & State Key Laboratory of Cognitive Intelligence Hefei, China jy251198@mail.ustc.edu.cn 

School of Computer Science and Technology, University of Science and Technology of China & State Key Laboratory of Cognitive Intelligence Hefei, China huangzhy@ustc.edu.cn 

School of Computer Science and Technology, University of Science and Technology of China & State Key Laboratory of Cognitive Intelligence Hefei, China zhymma2000@gmail.com 

## Enhong Chen[∗] 

## Tianhuang Su 

## Qi Liu 

Anhui Province Key Laboratory of Big Data Analysis and Application, University of Science and Technology of China & State Key Laboratory of Cognitive Intelligence Hefei, China cheneh@ustc.edu.cn 

School of Computer Science and Technology, University of Science and Technology of China & State Key Laboratory of Cognitive Intelligence Hefei, China qiliuql@ustc.edu.cn 

OPPO Mobile Telecommunications Shenzhen, China sutianhuang@oppo.com 

Haifeng Liu University of Science and Technology of China Hefei, China bladehliu@gmail.com 

## **ABSTRACT** 

Math formulas (e.g., “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ”) serve as one of the fundamental commonsense knowledge in human cognition, where humans naturally acquire and manipulate them in logical thinking for mathematical reasoning problems. However, existing reasoning models mainly focus on learning heuristic linguistics or patterns to generate answers, but do not pay enough attention on learning with such formula knowledge. Thus, they are not transparent (thus uninterpretable) in terms of understanding and grasping basic mathematical logic. In this paper, to promote a step forward in the domain, we first construct two datasets (Math23K-F and MAWPS-F) with precise annotations of formula usage in each reasoning step for math word problems. Especially, our datasets are refined on the benchmark datasets, and thus ensure the generality and comparability for relevant research. Then, we propose a novel Formulamastered Solver (FOMAS) with the guidance of mastering formula knowledge to solve the problems. Specifically, we establish FOMAS 

with two systems drawing insight from the dual process theory, including a Knowledge System and a Reasoning System, to learn and apply formula knowledge, respectively. The Knowledge System accumulates the math formulas, where we propose a novel pretraining manner to mimic how humans grasp the mathematical logic behind them. Then, in the Reasoning System, we develop elaborate formula-guided symbol prediction and goal generation methods that retrieve the necessary formula knowledge from Knowledge System to improve both reasoning accuracy and interpretability. It organically simulates how humans conduct complex reasoning under the explicit instruction of math formulas. Experimental results prove that FOMAS has a stronger reasoning ability and achieves a more interpretable reasoning process, which verifies the necessity of introducing formula knowledge transparently. 

## **CCS CONCEPTS** 

• **Computing methodologies** → **Knowledge representation and reasoning** ; **Symbolic and algebraic manipulation** . 

∗Corresponding Authors 

## **KEYWORDS** 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _KDD ’23, August 6–10, 2023, Long Beach, CA, USA._ 

© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0103-0/23/08...$15.00 https://doi.org/10.1145/3580305.3599375 

Knowledge Representation and Reasoning, Math Word Problem 

## **ACM Reference Format:** 

Jiayu Liu, Zhenya Huang, Zhiyuan Ma, Qi Liu, Enhong Chen, Tianhuang Su, and Haifeng Liu. 2023. Guiding Mathematical Reasoning via Mastering Commonsense Formula Knowledge. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23), August 6–10, 2023, Long Beach, CA, USA._ ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3580305.3599375 

1477 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA. 

Jiayu Liu et al. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0002-02.png)


**----- Start of picture text -----**<br>
�����������������<br>� � � � �<br>����� distance     speed     time � � �� ���  speeddistance<br>� �� time<br>rectangle_area      length     width � �<br>���<br>�����������������<br>��������������������������������������<br>������������������������������<br>����� ����������<br>�����<br>�� � �� distance     speed     time � �<br>����������������������<br>**----- End of picture text -----**<br>


**Figure 1: Example of learning and applying the formula knowledge to mathematical reasoning on MWP.** 

## **1 INTRODUCTION** 

Knowledge is the human belief encompassing empirical awareness of facts and epistemic contact with the world [45]. It lays the foundation for human cognition, where humans naturally acquire knowledge from experience and manipulate it in cognitive behaviors [33, 44]. In the domain of data mining, there is an emerging trend to investigate how to gain knowledge from data and apply it in several complex reasoning tasks [15, 31, 32, 39]. Among them, mathematical reasoning is one of the core tasks in quest of models with human cognitive ability, which requires various knowledge (e.g., algebra, geometry, probability) in logical thinking [5, 16, 53]. 

Specially, math formula is a kind of essential and commonsense knowledge formed from human experience [25]. We are constantly learning and applying various math formulas to figure out problems, and typically, math word problems (MWP). Let us take an example in Figure 1. Solving the MWP generally asks to read a problem description (“If Anne ... go ?”) and then reasons an expression (“2 × 3”) for getting the answer. This process requires necessary formulas (“ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ”) to explicitly serve as the guidance that navigates the thinking pattern and directs symbol derivation based on problem understanding [12, 14], which reflects the complicated mastery of abstract logic in human cognition. Thus, the mastery of math formulas for conducting reasoning like humans is a necessary sign of the intelligent level of models [6]. 

However, in the literature, such explicit knowledge has not attracted enough attention in relevant research on reasoning. Taking the widely studied MWP as an example, although traditional work takes into account mathematical rules or templates [2, 11, 37], they operate according to manually defined programs without really gaining the ability to understand this knowledge. Comparatively, most current methods rely on a Seq2Seq model [35, 47], which generate a hidden state (or named as reasoning goal [49]) at each decoding step to predict a specific symbol. They mainly grasp the linguistic semantics [22, 27] and heuristic reasoning patterns [19, 49], but ignore whether the model can master the epitome of logic manifested in explicitly utilizing formula knowledge [38, 50]. 

To further verify the necessity, we tend to experimentally quantify the mastery degree of formula knowledge for current models. Nevertheless, existing mathematical benchmark datasets (e.g., Math23K [47], MAWPS [24], GSM8K [4], MATH [13]) do not provide such a label for formula usage in expression reasoning process. 

In this paper, to promote a step forward in the domain, we first construct two datasets (Math23K-F and MAWPS-F) with precise annotations to support specific explorations. They contain 51 and 18 different math formulas and 23 _,_ 162 and 2 _,_ 373 annotated problems refined on benchmarks Math23K and MAWPS respectively, which ensures the generality and comparability for relevant research. For each problem, we annotate the formula required at each reasoning step in its expression, e.g., “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” for “×” in Figure 1. Based on our datasets, we conduct empirical experiments (described in Section 3.1) for existing representative methods (GTS [49], Graph2Tree [54], BERT-Tree [26]) and observe that more than 22% of their errors are due to the inability to use formulas. Thus, we argue that it is vital to investigate how models learn and apply formula knowledge for mathematical reasoning. Along this line, there remain many technical challenges to be confronted. First, math formulas are highly symbolic that involve abstract structure and concrete concepts. For example, “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” in Figure 1 is composed of the structure “ _𝑎_ = _𝑏_ × _𝑐_ ” and concepts “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒,𝑠𝑝𝑒𝑒𝑑,𝑡𝑖𝑚𝑒_ ”. “ _𝑎_ = _𝑏_ × _𝑐_ ” reflects the calculation pattern that could be shared between different formulas (e.g., “ _𝑟𝑒𝑐𝑡𝑎𝑛𝑔𝑙𝑒_  𝑎𝑟𝑒𝑎_ = _𝑙𝑒𝑛𝑔𝑡ℎ_ × _𝑤𝑖𝑑𝑡ℎ_ ” also satisfies “ _𝑎_ = _𝑏_ × _𝑐_ ”), while concepts determine the semantic information that should be displayed in the structure to truly understand the formula. It is indispensable to mine their respective meanings as well as integrate them as a whole. Second, math formulas contain rich mathematical logic. Humans not only learn whether a formula is correct or not, but are also able to perform logical transformations such as changing “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” to “ _𝑠𝑝𝑒𝑒𝑑_ = _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ ÷ _𝑡𝑖𝑚𝑒_ ”. Teaching models to master such implied knowledge is necessary but challenging. Third, the process of human application of formula knowledge is sophisticated, which includes selecting formulas according to different problems (e.g., choose “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” when reasoning “×” for problem in Figure 1) and utilizing them in multiple reasoning steps (inherit the information of “ _𝑠𝑝𝑒𝑒𝑑_ ” and “ _𝑡𝑖𝑚𝑒_ ” to derive symbols “2” and “3”). The unclear cognitive mechanisms behind it brings many difficulties in modeling. 

To address the challenges above, we propose a novel Formulamastered Solver (FOMAS) to mimic how humans solve MWP under the guidance of formula knowledge. In FOMAS, we draw insights from the dual process theory [8, 20] that elaborates on the structure of human cognition to construct two systems: Knowledge System and Reasoning System. The Knowledge System plays the role of human brain to store, represent, and learn formula knowledge, while the Reasoning System analytically applies the formula knowledge accumulated in Knowledge System to reason answers for MWP. Specifically, in Knowledge System, we first adequately excavate the information of each formula by decoupling and encoding its structure feature and lexical feature. Then, we propose a pretraining manner to realize autonomous formula learning, where two novel pretraining objectives are designed based on human understanding of formula legality and flexibility. In Reasoning System, given a math word problem, we retrieve formulas from the Knowledge System to guide the expression decoding phase (i.e., symbol prediction and goal generation). To be specific, we develop three sophisticated reasoning mechanisms to predict a symbol at each reasoning step, including a formula-selected one, a formula-inherited one, and a typical direct one. Their results are organically ensembled 

1478 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA. 

Guiding Mathematical Reasoning via Mastering Commonsense Formula Knowledge 

to simulate complex human thought. Moreover, we introduce the representations of intermediate steps of formulas into deriving explainable reasoning goals. Extensive experiments on our benchmark datastes verify FOMAS’ improvements on answer accuracy and reasoning interpretability. The contributions of this paper are: 

- We propose a Formula-mastered Solver (FOMAS) that learns and applies formula knowledge to conduct mathematical reasoning. To the best of our knowledge, we are the first to explore it by referring to the characteristics of human cognition, whose main ideas of Knowledge-Reasoning systems and specific learning/reasoning methods can be quite general for different types of mathematical problems. 

- We design a novel pretraining manner to learn the knowledge behind math formulas and develop elaborate formulaguided symbol prediction/goal generation mechanisms, which enhance reasoning accuracy and interpretability. 

- We construct two benchmark MWP datasets (Math23K-F, MAWPS-F) with annotations of the required formula at each reasoning step to support the exploration of formula knowledge in the domain. Extensive experiments on them clearly demonstrate the effectiveness of FOMAS. 

## **2 RELATED WORK** 

**Math Word Problem.** Mathematical reasoning aims to acquire knowledge from data and apply it to solve math problems. According to different problem types, representative benchmarks include for MWP (e.g., Math23K [47]), geometry (e.g., Inter-GPS [34]), unified task (e.g., MATH [13]), etc., among which MWP motivates the fundamental and far-reaching research since the 1960s [10, 17]. 

In the literature, traditional MWP solvers can be classified into three types: rule-based [2], statistic-based [37], and semantics parsingbased [43]. They require manually created schemas, templates, or formal language to derive expressions, respectively, and thus suffer from limited applications on large datasets and low generality. Recently, Wang et al. [47] presented a Seq2Seq model that adopted the encoder-decoder to directly translate a problem into the expression. Based on such a structure, most advanced work can be summarized from two aspects: improving the semantic understanding ability of encoder [22, 28, 29] and the reasoning ability of decoder [19, 49]. For problem understanding, Zhang et al. [54] proposed a GCNbased encoder to capture the relationships and order information of quantities, Lin et al. [29] adopted a word-clause-problem hierarchical encoder to simulate human reading habits. Besides, pretrained language models [22, 26, 27] and external knowledge graphs [48] have also been applied to inject human knowledge into problem comprehension. As for reasoning, Xie et al. [49] adopted a goaldriven decomposition mechanism to generate binary expression trees, while Wang et al. [46] designed M-tree structure to unify the diverse outputs, and Jie et al. [19] proposed a deductive reasoner to iteratively construct step-by-step expressions. 

**Neural-Symbolic Systems.** The core goal of our paper is to simulate human cognition to master the symbolic math formula knowledge for mathematical reasoning. Thus, we report the related work regarding neural-symbolic systems [51], which have flourished in many reasoning tasks such as knowledge graph reasoning [3, 42], visual reasoning [36, 52], and text reasoning [30, 39]. For 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0003-10.png)


**----- Start of picture text -----**<br>
������� ������������������������������������������<br>��������������������������<br>������ �<br>���������� �����<br>������� ���������������������������������������������� distance     speed     time � �<br>�<br>distance � �<br>�����������<br>speed time � �<br>������������������� ���������������<br>Figure 2: An example of our annotated data.<br>**----- End of picture text -----**<br>


example, Qu et al. [42] proposed pLogicNet that leveraged the firstorder logic of triplets to support inference on knowledge graphs. Mao et al. [36] constructed NS-CL that executed logic operations in a symbolic program to derived answers for visual questions. Specially, for mathematical reasoning, Peng et al. [39] designed a novel representation architecture GATE to conduct interpretable symbolic deduction and computation. Aiming at MWP, Qin et al. [41] proposed NS-Solver to incorporate symbolic constraints in training by four auxiliary tasks. Yang et al. [50] built LogicSolver that used logic formula as prompts to enrich problem understanding and predicted formulas as explanations, which for the first time introduced and verified the importance of formula knowledge for MWP. 

Our work improves previous studies from the following three aspects. First, current neural-symbolic systems mainly focus on logic rules and knowledge graph. Comparatively, we explore how to master the commonsense but essential formula knowledge in mathematical reasoning, aiming at which we propose a novel formula pretraining schema and design elaborate application mechanisms. Second, compared with existing work on MWP, our method learns and applies the math formulas to finely guide expression reasoning, which is more transparent and accurate in terms of the reasoning process. Specially, unlike the most relevant LogicSolver [50] that uses formulas as prompts and post-explanation, our formula application mechanisms are more in line with human cognition and achieves better effectiveness. Third, from the perspective of dataset, our benchmarks provide high-quality annotations to support exploration of formula knowledge, maintaining great generality and comparability to potentially boost further study in the domain. 

## **3 PRELIMINARIES** 

## **3.1 Dataset Construction and Analysis** 

In the field of math word problem (MWP), representative datasets include Math23K [47], MAWPS [24], SVAMP [38], GSM8K [4], etc.. However, to the best of our knowledge, they do not provide satisfactory annotations to guide models how to apply formula knowledge step by step in the reasoning process. In response to such a burgeoning need, we expect to construct datasets that have two characteristics. First, they should precisely describe the formula usage of reasoning process, i.e., annotate the formula applied at each reasoning step. Second, we hope to ensure the generality so that most previous models can be easily and fairly compared on them. Based on these considerations, we decide to annotate the two most widely studied MWP datasets Math23K [47] and MAWPS [24], 

1479 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA. 

Jiayu Liu et al. 

**Table 1: The 5 most frequently used math formulas.** 

|Math23K-F|1._𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_=_𝑠𝑝𝑒𝑒𝑑_×_𝑡𝑖𝑚𝑒_<br>2._𝑤𝑜𝑟𝑘_=_𝑟𝑎𝑡𝑒_×_𝑡𝑖𝑚𝑒_<br>3._𝑡𝑜𝑡𝑎𝑙_𝑐𝑜𝑠𝑡_=_𝑢𝑛𝑖𝑡_𝑐𝑜𝑠𝑡_×_𝑡𝑜𝑡𝑎𝑙_𝑛𝑢𝑚𝑏𝑒𝑟_<br>4._𝑡𝑜𝑡𝑎𝑙_𝑎𝑚𝑜𝑢𝑛𝑡_=_𝑢𝑛𝑖𝑡_𝑎𝑚𝑜𝑢𝑛𝑡_×_𝑡𝑜𝑡𝑎𝑙_𝑛𝑢𝑚𝑏𝑒𝑟_<br>5._𝑡𝑜𝑡𝑎𝑙_𝑤𝑒𝑖𝑔ℎ𝑡_=_𝑢𝑛𝑖𝑡_𝑤𝑒𝑖𝑔ℎ𝑡_×_𝑡𝑜𝑡𝑎𝑙_𝑛𝑢𝑚𝑏𝑒𝑟_|
|---|---|
|MAWPS-F|1._𝑡𝑜𝑡𝑎𝑙_𝑎𝑚𝑜𝑢𝑛𝑡_=_𝑢𝑛𝑖𝑡_𝑎𝑚𝑜𝑢𝑛𝑡_×_𝑡𝑜𝑡𝑎𝑙_𝑛𝑢𝑚𝑏𝑒𝑟_<br>2._𝑡𝑜𝑡𝑎𝑙_𝑐𝑜𝑠𝑡_=_𝑢𝑛𝑖𝑡_𝑐𝑜𝑠𝑡_×_𝑡𝑜𝑡𝑎𝑙_𝑛𝑢𝑚𝑏𝑒𝑟_<br>3._𝑡𝑜𝑡𝑎𝑙_𝑖𝑛𝑐𝑜𝑚𝑒_=_𝑢𝑛𝑖𝑡_𝑖𝑛𝑐𝑜𝑚𝑒_×_𝑡𝑜𝑡𝑎𝑙_𝑛𝑢𝑚𝑏𝑒𝑟_<br>4._𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_=_𝑠𝑝𝑒𝑒𝑑_×_𝑡𝑖𝑚𝑒_<br>5._𝑤𝑜𝑟𝑘_=_𝑟𝑎𝑡𝑒_×_𝑡𝑖𝑚𝑒_|



which contain 23,162 and 2,373 problems at the elementary school level respectively. There original data consists of “problem”, “answer”, and “expression”, as shown in the top three lines in Figure 2. 

Our annotation is conducted as follows. Inspired by existing work [1, 50], we firstly collected essential math formulas from textbooks and summarized 51 and 18 representative ones on Math23K and MAWPS respectively under the guidance of two elementary school teachers. Secondly, we invited five well-trained annotators with undergraduate degree to manually select the most suitable formula for each reasoning step of data in Math23K and MAWPS from the set of all formulas and their variants (will be defined in Section 3.2). For example, as depicted in the “Formula” line in Figure 2, the annotation of problem “If Anne ... go ?” is [ _𝜙_ , _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ , _𝜙_ ], whose length is equal to the length of expression “2 × 3”. The explanation behind it is that reasoning the “ ×” requires “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ”, while reasoning “2” and “3” do not need additional formulas (thus denoted as an empty formula _𝜙_ ). Thirdly, another three annotators were asked to evaluate the annotations, based on which the annotators modified the results and repeated the evaluation-modification processes. After three revisions with a pass rate of 93.2%, 96.3%, and 97.7%, we settled on the final annotations and name the new datasets as Math23K-F and MAWPS-F. In summary, on our benchmarks, 33 _._ 5% and 38 _._ 4% of problems require the use of formulas respectively. We report the 5 most frequently used formulas in Table 1 and summarize the dataset statistics in Table 2. More analyses are presented in _Appendix A_ . 

Based on our benchmark datasets, we tend to verify whether the study of formula knowledge is necessary. For this purpose, we select the three most representative SOTA MWP models, including GTS [49], Graph2Tree [54], and BERT-Tree [26], and evaluate their mastery degree of formula knowledge. Specifically, we introduce a _𝑃𝐼𝐹_ metric that calculates the **P** ercentage of problems that a model answers **I** ncorrectly at steps requiring a **F** ormula (as we annotated above) in all problems that it gets the answer wrong. A higher _𝑃𝐼𝐹_ reflects a greater defect in formula mastery. As shown in Figur 3, _𝑃𝐼𝐹_ of the three models exceeds 0 _._ 220 on both datasets, which indicates that the accuracy of the existing work is greatly limited due to lack of applying math formulas. Thus, we argue that it is necessary to explore how to master formula knowledge for conducting more reliable mathematical reasoning. 

## **3.2 Problem Definition** 

Formally, the input of a math word problem _𝑃_ is a sequence of _𝑛_ words and numeric values: _𝑋𝑃_ = [ _𝑥_ 1 _,𝑥_ 2 _, ...,𝑥𝑛_ ], where _𝑥𝑖_ is either a word (e.g., “If”, “Anne” in Figure 2) or a number (e.g., “3”, “2”). 

**Table 2: Statistics of our benchmark datasets.** 

|Dataset|Math23K-F<br>MAWPS-F|
|---|---|
|Num. problems<br>Num. formulas (and variants)<br>Num. problems requiring formula<br>Avg. problem length<br>Avg. expr. length|23,162<br>2,373<br>51 (131)<br>18 (46)<br>7,750<br>911<br>28.02<br>30.08<br>5.55<br>4.20|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0004-11.png)


**----- Start of picture text -----**<br>
GTS Graph2Tree BERT-Tree<br>Figure 3:  𝑃𝐼𝐹 of GTS, Graph2Tree, and BERT-Tree.<br>**----- End of picture text -----**<br>


The output of _𝑃_ is a sequence of _𝑚_ symbols: _𝑌𝑃_ = [ _𝑦_ 1 _,𝑦_ 2 _, ...,𝑦𝑚_ ]. Symbol _𝑦𝑖_ is taken from a vocabulary _𝑉𝑃_ composed of the operator set _𝑉𝑂_ (e.g., {+ _,_ × _,_ − _,_ ÷}), numeric constant set _𝑉𝑁_ (e.g., 1 _, 𝜋_ ), and numeric values _𝑁𝑃_ in _𝑋𝑃_ , i.e., _𝑉𝑃_ = _𝑉𝑂_ ∪ _𝑉𝑁_ ∪ _𝑁𝑃_ . As _𝑁𝑃_ varies with the input sequence, different problems may have different _𝑉𝑃_ . The target of MWP is to train a model that reads the problem description _𝑋𝑃_ and then generates the corresponding expression _𝑌𝑃_ , based on which calculates a numeric answer for _𝑃_ (e.g., “6”). 

For formula knowledge, we formalize it as a set of _𝐾_ math formulas _𝑅_ = { _𝑟_ 1 _,𝑟_ 2 _, ...,𝑟𝐾_ } ( _𝐾_ = 51 and 18 on Math23K-F and MAWPS-F respectively). Each formula _𝑟𝑘_ is represented as an Operator Tree (OPT) [40]. For example, the formula “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” corresponds to a tree that contains five nodes as shown in Figure 2, with “=” as the root and “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ ”,“ _𝑠𝑝𝑒𝑒𝑑_ ”,“ _𝑡𝑖𝑚𝑒_ ” as the leaf nodes. Thus, we denote _𝑟𝑘_ as a sequence of _𝑙_ elements _𝑟𝑘_ = [ _𝑧_ 1 _,𝑧_ 2 _, ...,𝑧𝑙_ ] that represents the prefix expression of the OPT, where _𝑧𝑖_ is either an abstract concept (e.g., “ _𝑡𝑖𝑚𝑒_ ”) or an operator (e.g., “×”). For example, “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” is represented as [=, _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ , ×, _𝑠𝑝𝑒𝑒𝑑_ , _𝑡𝑖𝑚𝑒_ ]. Furthermore, we define a set of variants _𝐴_ ( _𝑟_ ) ⊈ _𝑅_ for each formula _𝑟_ (e.g., “ _𝑠𝑝𝑒𝑒𝑑_ = _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ ÷ _𝑡𝑖𝑚𝑒_ ” and “ _𝑡𝑖𝑚𝑒_ = _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ ÷ _𝑠𝑝𝑒𝑒𝑑_ ”) to imply logical formula transformation, which is important in formula learning and application in Section 4.1 and Section 4.2. In summary, the research problem in this paper is: 

_Definition 3.1_ . Given the MWP dataset _𝐷_ = {( _𝑋𝑃 ,𝑌𝑃_ )} and the formula set _𝑅_ , our goal is to build a model that learns the formula knowledge in _𝑅_ and applies it to reason the expression _𝑌𝑃_ = [ _𝑦_ 1 _,𝑦_ 2 _, ...,𝑦𝑚_ ] for each problem _𝑋𝑃_ = [ _𝑥_ 1 _,𝑥_ 2 _, ...,𝑥𝑛_ ]. 

## **4 FORMULA-MASTER SOLVER** 

Generally, the dual process theory [8, 20] indicates that human cognition is built on two systems: one is an intuitive, unconscious system that processes human experience and knowledge, and the other is a reasoning system that uses the knowledge for slow, explicit logical thinking. Drawing this insight, we construct two systems in our Formula-master Solver (FOMAS): Knowledge System and Reasoning System, as shown in Figure 4(a)(b). Specifically, Knowledge System stores the formula knowledge _𝑅_ and mimics how humans represent and learn it autonomously, while Reasoning System applies the formula knowledge in Knowledge System to reason the expression _𝑌𝑃_ for a specific math word problem _𝑋𝑃_ . 

1480 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA. 

Guiding Mathematical Reasoning via Mastering Commonsense Formula Knowledge 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0005-02.png)


**----- Start of picture text -----**<br>
��� ���������������� �<br>����������������� ��������������� ������������ ���������������� ������ �������� � �����<br>��������������� ���������������������� ������������ ���� �� ���� ���������� ����������������������� �� ����������� ���������� �������� �������������� � ���� �� ������������� ����� � ���� ���������� �� �<br>��� ������ � �����<br>����������������� �� ��������������� � �� ������� ����� ����� �� �<br>����������������� ������������������ ������� ����� ���� �� ������ �����<br>�������������� �������������� �������������� �� � [�] � [�] �������������������������������� ������ � �����<br>������������������������� �� � [�] � [�] ����������������� �� ������� ���� ����� �� �<br>�������� ����������������������������������������������������� P : ����������������� ������� ������� ����������� ������� Y � ��������������� ������ �����<br>��� ���������������� ����������������������������������<br>**----- End of picture text -----**<br>


**Figure 4: The overview of FOMAS. The left part (a)(b) shows the architecture of FOMAS that consists of Knowledge System and Reasoning System . The right part (c) is an example workflow of formula-guided symbol prediction and goal generation.** 

## **4.1 Knowledge System** 

In this system, we first design a representation module to comprehensively mine the features of formulas. Then we propose a pretraining manner to learn these features, which resembles humans’ mastery of the mathematical logic behind formulas. 

_4.1.1 Formula Representation._ As logicians and mathematicians have pointed out, the difficulty of understanding formula knowledge lies in distinguishing formulas that have the same structure but different meanings [7]. For example, “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” and “ _𝑟𝑒𝑐𝑡𝑎𝑛𝑔𝑙𝑒_  𝑎𝑟𝑒𝑎_ = _𝑙𝑒𝑛𝑔𝑡ℎ_ × _𝑤𝑖𝑑𝑡ℎ_ ” have the same structure abstracted as _𝑎_ = _𝑏_ × _𝑐_ . However, they are different in the lexicons of _𝑎,𝑏,𝑐_ (e.g., _𝑎_ = _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ , _𝑏_ = _𝑠𝑝𝑒𝑒𝑑_ , _𝑐_ = _𝑡𝑖𝑚𝑒_ ). The meaning of the whole formula is compositionally built up from the combination of these two types of information, both of which are necessary. Inspired by it, we decouple each formula _𝑟_ ∈ _𝑅_ into two types of features: structural feature _𝐹𝑟_ and lexical feature _𝐸𝑟_ , which focus on the structural information and concrete meaning respectively. 

Specifically, the structural feature _𝐹𝑟_ defines the order and mode of formula calculation, so we design an innovative kind of node function _𝑓_ and formalize _𝐹𝑟_ as a sequence of it, i.e., _𝐹𝑟_ ≜ [ _𝑓_ 1 _, 𝑓_ 2 _, ..., 𝑓𝑤_ ], where _𝑤_ is the length of formula _𝑟_ and _𝑓𝑖_ is the function of _𝑖_ -th “ node (i.e., _𝑧𝑖_ ). For example, the structural feature of _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” is _𝐹𝑟_ = [ _𝑓_ = _, 𝑓𝑎, 𝑓_ × _, 𝑓𝑏, 𝑓𝑐_ ]. The input and output of _𝑓_ will be explained below after introducing the lexical feature. 

The lexical feature _𝐸𝑟_ reflects the semantics of formula _𝑟_ embodied in the leaf nodes of its OPT (e.g., _𝑎,𝑏,𝑐_ ). We formalize it as a sequence of concept vectors, i.e., _𝐸𝑟_ ≜ [ _𝒆_ 1 _, 𝒆_ 2 _, ..., 𝒆𝑣_ ], where _𝑣_ is the number of leaf nodes, _𝒆𝑖_ ∈ R _[𝑑]_ is the _𝑖_ -th node vector, and _𝑑_ is the dimension. For example, the lexical feature of “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” is _𝐸𝑟_ = [ _𝒆_ 1 _, 𝒆_ 2 _, 𝒆_ 3], where _𝒆_ 1 _, 𝒆_ 2 _, 𝒆_ 3 are the concept vectors of “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ ”, “ _𝑠𝑝𝑒𝑒𝑑_ ”, “ _𝑡𝑖𝑚𝑒_ ”, respectively. 

Now we explain the computing flow of the node functions, which basically follows a bottom-up pattern. Specifically, given formula _𝑟_ , for a leaf node in its OPT, the node function _𝑓_ is an identity mapping, whose input and output are the same concept vector of the node. 

For example, as marked green in Figure 4(a), _𝑓𝑎_ (·) ≜ _𝑓𝑎_ ( _𝒆𝑎_ ) = _𝒆𝑎_ , where _𝒆𝑎_ is input as the concept vector of “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ ” (i.e., _𝒆_ 1) for “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ”. For a non-leaf node, the function _𝑓_ encodes a representation of the intermediate step in the formula. It takes into the outputs of its left and right child nodes, and fuses them with the operator information. For example, in Figure 4(a), _𝑓_ × (· _,_ ·) ≜ _𝑓_ × ( _𝑓𝑏_ ( _𝒆𝑏_ ) _, 𝑓𝑐_ ( _𝒆𝑐_ )). By taking _𝒆𝑏_ = _𝒆_ 2 _, 𝒆𝑐_ = _𝒆_ 3, we can obtain the embedding of node “×”, which reflects the meaning of “ _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ”. Specially, _𝑓_ · (· _,_ ·) is implemented as a network: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0005-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0005-12.png)


where _𝜏_ ∈{+ _,_ − _,_ × _,_ ÷ _,_ =} indicates the operator, _𝒐𝜏_ ∈ R _[𝑑]_ is its symbol embedding, _𝒆𝑙 , 𝒆𝑟_ are the outputs of the left and right child nodes (e.g., _𝑓𝑏_ ( _𝒆𝑏_ ) _, 𝑓𝑐_ ( _𝒆𝑐_ )), ⊙ is element-wise product, _𝜎_ is the sigmoid function, and _𝑾𝑛_ 1 _, 𝑾𝑛_ 2 are parameters. 

In summary, structural feature _𝐹𝑟_ = [ _𝑓_ 1 _, 𝑓_ 2 _, ..., 𝑓𝑤_ ] defines the calculation pattern of formula _𝑟_ by node functions, while lexical feature _𝐸𝑟_ = [ _𝒆_ 1 _, 𝒆_ 2 _, ..., 𝒆𝑣_ ] stores the practical meanings by concept vectors. Combining them, we can iteratively encode a semantic embedding _𝑠_ for each non-leaf node by Eq. (1), which represents an intermediate state of the formula. We denote these semantic embeddings as _𝑆𝑟_ = [ _𝒔_ 1 _, ..., 𝒔𝑤_ − _𝑣_ ], where _𝒔_ 1 is the embedding of root “ =” that encodes the information of the whole formula. In Section 4.2, we will see how they contribute to guiding mathematical reasoning, which reflects the superiority of our representation module over directly encoding formulas by RNN-like models. 

_4.1.2 Formula Pretraining._ In order to mimic how humans process and learn the formula knowledge autonomously, we further pretrain FOMAS’ representation module (i.e., parameters of node functions in _𝐹𝑟_ and concept vectors in _𝐸𝑟_ ). However, since the formulas are short in length, strong in logic, and do not have a large corpus, it is not suitable to adopt the existing pretraining manners (e.g., Masked Language Model [56]). Thus, as shown in Figure 4(a), to embody mathematical logic into formula learning, we design two novel pretraining objectives: legality _𝐿𝑙𝑒𝑔_ and flexibility _𝐿𝑓𝑙𝑒_ . 

1481 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA. 

Jiayu Liu et al. 

The legality _𝐿𝑙𝑒𝑔_ focuses on learning to distinguish whether a formula is legal or not. That is, not only do humans acquire that “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” is a valid formula, but also know “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑤𝑖𝑑𝑡ℎ_ ” is invalid and avoid using it in reasoning. Through this objective, FOMAS can pay attention to the detailed components of formulas and grasp more refined understanding. For this purpose, we introduce a dichotomous task. For each formula _𝑟_ ∈ _𝑅_ , we first obtain the semantic vector of its root node “=” using features _𝐹𝑟_ and _𝐸𝑟_ according to Section 4.1.1, denoted as _𝒔𝑟_ . Then, we randomly replace one of the concept vectors in _𝐸𝑟_ to construct an illegal formula sample _𝑟_[′] , e.g., “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑤𝑖𝑑𝑡ℎ_ ”, and obtain the representation of its root _𝒔𝑟_[′] similarly. Finally, we input _𝒔𝑟 , 𝒔𝑟_[′] to calculate the following loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0006-03.png)


where _𝑀𝑗𝑢𝑑_ ( _𝒔_ ) : R _[𝑑]_ →[0 _,_ 1] is a judgment network to learn the probability that _𝑟_ and _𝑟_[′] are legal. _𝑙𝐵𝐶𝐸_ is the BCE loss, 1 and 0 are ground-truth labels that represent “legal” and “illegal”, respectively. 

For a formula, it is not enough to grasp its own legitimacy, because humans can based on it to judge whether its complex transformations (e.g., “ _𝑠𝑝𝑒𝑒𝑑_ = _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ ÷ _𝑡𝑖𝑚𝑒_ ” and “ _𝑡𝑖𝑚𝑒_ = _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ ÷ _𝑠𝑝𝑒𝑒𝑑_ ” for “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ”) are also legal, which together constitute a comprehensive mastery of the mathematical logic. For this purpose, we further incorporate a flexibility objective _𝐿𝑓𝑙𝑒_ , whose basic idea is that the variants of an (il)legal formula are also (il)legal. In specific, for formula _𝑟_ ∈ _𝑅_ , we first obtain its variants _𝐴_ ( _𝑟_ ). Then, for each _𝑟_[′′] ∈ _𝐴_ ( _𝑟_ ), we similarly construct a negative sample _𝑟_ −[′′] . Finally, we feed them into _𝑀𝑗𝑢𝑑_ and calculate: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0006-06.png)


With _𝐿𝑙𝑒𝑔_ and _𝐿𝑓𝑙𝑒_ , our pretraining schema is to minimize Eq. (4), which can capture strong mathematical logic in formula representations to better guide the reasoning process in Section 4.2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0006-08.png)


## **4.2 Reasoning System** 

The Reasoning System aims at applying the formula knowledge in the Knowledge System to reason answers for a specific problem. As depicted in Figure 4(b), we generally adopt an encoder-decoder manner which first encodes the problem sentence and then decodes an expression tree [49]. This system mainly focuses on the decoding phase, which uses formulas to guide the symbol prediction and goal generation processes in decoder, namely formula-guided symbol prediction and formula-guided goal generation, respectively. 

_4.2.1 Encoder-Decoder._ Given problem _𝑃_ , we first input the problem sentence _𝑋𝑃_ = [ _𝑥_ 1 _,𝑥_ 2 _, ...,𝑥𝑛_ ] into an encoder _𝑓𝜃_ to obtain the word representations _𝑯_ = [ _𝒉_ 1 _, 𝒉_ 2 _, ..., 𝒉𝑛_ ] and generate the initial reasoning goal _𝒒_ 1 for the decoder: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0006-12.png)


_𝑓𝜃_ can be specified as RNN (e.g., GTS [49]), BERT (e.g., BERTTree [26]), or specific MWP encoder (e.g., HMS [29]). Here we adopt BERT [21] due to its strong capability of language modeling. 

Then, we utilize a decoder to generate the expression _𝑌𝑃_ = [ _𝑦_ 1 _,𝑦_ 2 _, ...,𝑦𝑚_ ] step by step. At each step _𝑡_ ( _𝑡_ = 1 _,_ 2 _, ...,𝑚_ ), the decoder 1) predicts the symbol _𝑦𝑡_ (e.g., “×”, “2”) given the reasoning goal _𝒒𝑡_ by _𝑃_ ( _𝑦𝑡_ | _𝑦_ 1 _, ...,𝑦𝑡_ −1 _, 𝒒𝑡 , 𝑯_ ); 2) generates the next reasoning goal _𝒒𝑡_ +1 based on _𝑞𝑡_ and _𝑒_ ( _𝑦𝑡_ ) to support the next step _𝑡_ + 1, where _𝑒_ ( _𝑦𝑡_ ) is the embedding of symbol _𝑦𝑡_ , which is taken as _𝒉𝑖_ if _𝑦𝑡_ is the word _𝑖_ in _𝑃_ or _𝒐𝜏_ if _𝑦𝑡_ is operator _𝜏_ . 

Intuitively, the formula knowledge can guide both symbol prediction and goal generation in decoder. For example, when solving the problem in Figure 4(c), if we figure out that the first step (i.e., _𝑡_ = 1) requires the use of formula “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ”, then we can easily extract the symbol “×” from it as _𝑦_ 1. Then, having decided to use this formula, we can clearly know that the _𝑡_ = 2 and _𝑡_ = 3 steps are to get the “ _𝑠𝑝𝑒𝑒𝑑_ ” and “ _𝑡𝑖𝑚𝑒_ ” information respectively, which injects great interpretability into the reasoning goals _𝒒_ 2 _, 𝒒_ 3, so as to benefit further reasoning of _𝑦_ 2 = 2 _,𝑦_ 3 = 3. 

In summary, given problem sentence _𝑋𝑃_ and expression _𝑌𝑃_ , the training objective of FOMAS is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0006-17.png)


where[�] � − _𝑙𝑜𝑔𝑃_ ( _𝑦𝑡_ | _𝑦_ 1 _, ...,𝑦𝑡_ −1 _, 𝒒𝑡 , 𝑯_ ) is the symbol prediction _𝑃 𝑡_ loss. _𝐿𝑟_ is a loss that corresponds to deciding the used formula as illustrated above, which will be specified in Section 4.2.2. _𝛼_ is a hyper-parameter that balances these two losses. 

_4.2.2 Formula-guided Symbol Prediction._ Now we discuss how FOMAS applies the formula knowledge in the Knowledge System to predict symbol _𝑦𝑡_ given the reasoning goal _𝒒𝑡_ . Specifically, we propose three types of symbolic reasoning mechanisms summarized from sophisticated human thought process [9], i.e., formula-selected mechanism, formula-inherited mechanism, and direct reasoning. Symbol prediction at each step _𝑡_ requires a combination of these three mechanisms, as depicted in Figure 4(b) and (c). 

**Formula-selected mechanism.** Humans can naturally select a formula by referring to the current reasoning goal _𝒒𝑡_ at step _𝑡_ and extract a symbol from it as _𝑦𝑡_ . For example in Figure 4(c), if the goal _𝒒_ 1 implies “calculate the distance”, we can easily pick out the formula “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” and then reason “ _𝑦_ 1 = ×”, retrieving the semantic and symbolic information of formulas from Knowledge System respectively. Formally, at the current reasoning step _𝑡_ , we first calculate the attention of formula _𝑟_ and the goal _𝒒𝑡_ on problem word representations { _𝒉𝑖_ } by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0006-21.png)


where _𝒔𝑟_ is the semantic embedding of the root “=” of _𝑟_ , and _𝒘_ · _, 𝑾_ · are parameters. Then, we integrate these two kinds of attention to generate a score for _𝑟_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0006-23.png)


Here, the hyper-parameter _𝛽_ controls the integration weight. Since a problem may not require any formula (e.g., calculate the sum of two values), we also consider the empty formula _𝜙_ introduced in Section 3.1, denoted as _𝑟_ 0, and calculate _𝑠𝑐𝑜𝑟𝑒_ ( _𝑟_ 0) by Eqs. (7)(8), where _𝒔𝑟_ 0 is a learnable vector. Finally, we derive the probability of selecting formula _𝑟_ from all formulas and their variants, denoted as 

1482 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA. 

Guiding Mathematical Reasoning via Mastering Commonsense Formula Knowledge 

¯ _𝑅_ = _𝑅_ ∪{ _𝐴_ ( _𝑟_ ) _,𝑟_ ∈ _𝑅_ } ∪{ _𝑟_ 0} (| ¯ _𝑅_ | = 132 and 47 on Math23K-F and MAWPS-F respectively according to the statistics in Table 2): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0007-03.png)


and extract the right child _𝑦_ of root “=” for the formula with the highest _𝑃𝑠𝑒𝑙_ ( _𝑟_ ) (e.g., “×” in Figure 4(c)). We take the one-hot embedding of _𝑦_ as the reasoning result, denoted as _𝑦𝑡[𝑠𝑒𝑙]_ ∈ R[|] _[𝑉][𝑃]_[|] : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0007-05.png)


To train all the parameters in Eqs. (7)(8), we define _𝐿𝑟_ in Eq. (6) as the following cross entropy, where _𝑦𝑟_ ∈{0 _,_ 1} is the ground-truth label of selected formula that we have annotated in Section 3.1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0007-07.png)


**Formula-inherited mechanism.** The selected formula not only affects the current step, but also implies a kind of thinking pattern that navigates multiple reasoning steps [25]. For example, in Figure 4(c), after selecting the formula “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒_ ” at _𝑡_ = 1, we can inherit the “ _𝑠𝑝𝑒𝑒𝑑_ ” concept (i.e., the child of node “×” in formula OPT) at _𝑡_ = 2 to derive _𝑦_ 2 = 2 (find the “ _𝑠𝑝𝑒𝑒𝑑_ ” value). Inspired by it, in this mechanism, we inherit the symbols and concepts of the formula selected in preceding reasoning steps to derive _𝑦𝑡_ . Formally, at step _𝑡_ : (1) if an operator (e.g., “×”) is inherited, we obtain its one-hot embedding as _𝑦𝑡[𝑖𝑛ℎ]_ ∈ R[|] _[𝑉][𝑃]_[|] , otherwise (2) if an abstract concept (e.g., “ _𝑠𝑝𝑒𝑒𝑑_ ”) is inherited, we reason _𝑦𝑡[𝑖𝑛ℎ]_ based on its concept vector _𝒆𝑖𝑛ℎ_ retrieved from Knowledge System: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0007-09.png)


**Direct Reasoning.** Besides, humans can directly reason a symbol based on the reasoning goal _𝒒𝑡_ without any formula guidance as most existing MWP solvers do. Here, we set it as the scheme of GTS [49] and its reasoning result is denoted as _𝑦𝑡[𝑑𝑖𝑟]_ ∈ R[|] _[𝑉][𝑃]_[|] . 

To aggregate the results of different reasoning mechanisms, i.e., _𝑦𝑡[𝑠𝑒𝑙]_ , _𝑦𝑡[𝑖𝑛ℎ]_ , and _𝑦𝑡[𝑑𝑖𝑟]_ , we further propose a confidence-based ensemble inspired by the mixture of experts (MoE) [18, 55]. Specifically, we hold that _𝑃𝑠𝑒𝑙_ reflects the confidence of using the selected formula. Similarly, the inherited formula also comes with a probability that is inherited from the step selecting it, which we denote as _𝑃𝑖𝑛ℎ_ . In summary, FOMAS reasons _𝑦𝑡_ by ensembling all mechanisms: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0007-12.png)


_4.2.3 Formula-guided Goal Generation._ After predicting _𝑦𝑡_ , the core step of FOMAS is to apply the formula knowledge to generate an explainable next reasoning goal _𝒒𝑡_ +1. For this purpose, as shown in Figure 2, we observe that for the formula OPT and the expression tree, there is a one-to-one correspondence between the children of the same node. For instance, for “×”, its left child “ _𝑠𝑝𝑒𝑒𝑑_ ” in formula conveys the goal to reason the left child “2” in the expression, which also holds true for the right child “ _𝑡𝑖𝑚𝑒_ ” and “3”. Besides, as illustrated in Section 4.1.1, the nodes in formula _𝑟_ carry the concept vectors _𝐸𝑟_ (for leaf nodes) and semantic embeddings _𝑆𝑟_ (for non-leaf nodes) with strong semantics, e.g., the semantic embedding of “×” represents “multiplication of _𝑠𝑝𝑒𝑒𝑑_ and _𝑡𝑖𝑚𝑒_ ”. We argue that it can be combined with the correspondence to inject explicit meaning into the reasoning goals, thus ensuring good interpretability. 

Formally, if we have selected or inherited a formula _𝑟_ that derives symbol _𝑦𝑡_ by the mechanisms in Section 4.2.2, here we will use the concept vector (or semantic embedding) of _𝑦𝑡_ ’s left/right child in _𝑟_ from Knowledge System, denoted as _𝒔[𝑙]_ / _𝒔[𝑟]_ , to guide generate _𝑦𝑡_ ’s left/right sub-goal _𝒒[𝑙]_ (i.e., _𝒒𝑡_ +1)/ _𝒒[𝑟]_ respectively. Taking _𝒒[𝑙]_ for example, we implement our method as Eq. (14), where _𝒄_ is the context vector [49]. _𝒒[𝑟]_ is obtained similarly. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0007-15.png)


In summary, our FOMAS mainly has three advantages. First, its Knowledge-Reasoning Systems simulate the human cognitive structure of formula mastery, whose main idea is general to be potentially applicable to different reasoning problems (e.g., physical problems) that also require essential formula knowledge. Second, it provides a novel learning manner that masters both semantics and mathematical logic of formulas, which well models the human understanding characteristics (verified in Section 5.3.1). Third, it imitates the pattern of human thought to conduct reasoning under the instruction of math formulas, which duduces more reliable and reasonable results in a transparent way. 

## **5 EXPERIMENTS 5.1 Experimental Setup** 

_5.1.1 Implementation Details._ In Knowledge System, the dimension _𝑑_ of concept vectors is 512, and the epoch of pretraining is 100. In Reasoning System, the dimension of hidden vectors is 512. Specifically, _𝛼_ in Eq. (6) is set as 0 _._ 5 and 0 _._ 05 on Math23K-F and MAWPS-F respectively, and _𝛽_ in Eq. (8) is set to be 0 _._ 2. We will discuss their sensitivity in Section 5.2.3. All parameters are initialized uniformly and trained by Adam [23], with other hyperparameters being set following [27]. For dataset preprocessing, we follow the public train(21,162)/valid(1,000)/test(1,000) partition of Math23K for Math23K-F. For MAWPS-F, the models are evaluated with 5-fold cross-validation. All experiments are run on a Linux server with four 2.30GHz Intel Xeon Gold 5218 CPUs and a Tesla V100 GPU[1] . 

_5.1.2 Baselines._ We take the following representative and SOTA MWP models, as well as ChatGPT as baselines for comparison. 

- **Seq2Seq** [35]: uses a vanilla seq2seq model with attention to translate MWP to equation templates directly. 

- **GTS** [49]: adopts a heuristic goal-driven reasoning manner to generates expression trees. 

- **Graph2Tree** [54]: incorporates the relationships and order information among quantities into problem understanding. 

- **HMS** [29]: develops a hierarchical word-clause-problem relation to better exploit the problem semantics. 

- **NS-Solver** [41]: incorporates four auxiliary tasks into training to master symbolic constraints. 

- **BERT-Tree** [26]: employs BERT as semantic encoder and derives expressions by the decoder of GTS. 

- **SUMC** [46]: designs a M-tree coding to unify the diverse but equivalent expressions represented by binary trees. 

> 1Our datasets and codes are available at _https://github.com/Ljyustc/FOMAS_ . 

1483 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0008-01.png)


**----- Start of picture text -----**<br>
Jiayu Liu et al.<br>**----- End of picture text -----**<br>


**Table 3: Answer Accuracy (** ∗ : _𝑝 <_ 0 _._ 05 **w.r.t. BERT-Tree).** 

|**Answer Accur**|**acy (**∗: _𝑝<_0|_._05**w.r.t. BE**|
|---|---|---|
||Math23K-F|MAWPS-F|
|Seq2Seq<br>GTS<br>Graph2Tree<br>HMS<br>NS-Solver<br>BERT-Tree<br>SUMC<br>LogicSolver<br>ChatGPT<br>**FOMAS**|0_._640<br>0_._756<br>0_._774<br>0_._761<br>0_._757<br>0_._833<br>0_._825<br>0_._834<br>0_._649<br>0_._848∗|0_._797<br>0_._826<br>0_._837<br>0_._803<br>/<br>0_._872<br>0_._820<br>/<br>0_._883<br>0_._886∗|



**Table 4: Results of ablation study.** 

|||Math23K-F|MAWPS-F|
|---|---|---|---|
|FOMAS||0_._848|0_._886|
|Knowledge<br>System|w/o legality|0_._829|0_._875|
||w/o fexibility|0_._832|0_._875|
|Reasoning<br>System|w/o select|0_._839|0_._878|
||w/o inherit|0_._843|0_._880|
||w/o formula-goal|0_._842|0_._882|



- **LogicSolver** [50]: retrieves logical formulas as prompts to improve problem representations and predicts formulas after expression generation as explanations. 

- **ChatGPT**[2] : is a flourishing model trained on large-scale dialogue datasets, which is capable to interact in a conversational way. We input the math word problem into its chatbox and manually extract the numeric answer from its response. 

## **5.2 Performance on Answer Reasoning** 

_5.2.1 Answer Accuracy._ Table 3 reports the answer accuracy of all models, and we find several key observations. First, our FOMAS outperforms all the baselines, and by applying paired t-test, its improvements over the SOTA BERT-Tree on both datasets are statistically significant with _𝑝 <_ 0 _._ 05 (marked with ∗). This result demonstrates that mastering the formula knowledge is necessary and valuable to achieve stronger mathematical reasoning ability. Second, FOMAS performs better than LogicSolver which takes formulas as prompts to enrich problem understanding. It reflects that conducting reasoning under the guidance of formulas is necessary and more in line with human cognitive process, and verifies the effectiveness of our proposed formula application mechanisms in Reasoning System. Third, FOMAS obtains comparable performance to ChatGPT on MAWPS-F but is significantly better than it on Math23K-F. As Math23K-F is more difficult than MAWPS-F (i.e., has longer expression that requires more reasoning steps as shown in Table 2), we can conclude that the capability to acquire and explicitly apply formula knowledge makes our FOMAS more robust in figuring out complex mathematical problems (more analyses of ChatGPT are presented in _Appendix B_ ). 

_5.2.2 Ablation Study._ To examine each part of FOMAS, we conduct the ablation study in Table 4. Specifically, in the Knowledge System, we introduce “w/o legality” and “w/o flexibility” that omit 

> 2https://openai.com/blog/chatgpt/. Please see _Appendix B_ for more details. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0008-12.png)


**Figure 5: Accuracy with different hyperparameters** _𝛼_ **and** _𝛽_ **.** 

the legality objective _𝐿𝑙𝑒𝑔_ and flexibility objective _𝐿𝑓𝑙𝑒_ in Eq. (4), respectively. In the Reasoning System, we introduce “w/o select” and “w/o inherit” that remove the formula-selected mechanism and formula-inherited mechanism in Eq. (13), respectively. Moreover, “w/o formula-goal” replaces our formula-guided goal generation in Eq. (14) by the original generation method of GTS [49]. 

We conclude the results as follows. First, all components of FOMAS contribute to correctly mastering the formula knowledge, because removing each of them leads to a performance decrease. Second, the removal of legality or flexibility has the greatest impact on the effect, which implies that grasping the mathematical logic behind formulas is the foundation of using them. Besides, the two pretraining objectives are almost equally important as there is no significant difference between their results. Third, “w/o select” and “w/o inherit” diminish the results more greatly than “w/o formula-goal”, which suggests that formula-guided symbol prediction contributes more to FOMAS than goal generation. 

_5.2.3 Hyperparameter Sensitivity._ In FOMAS, _𝛼_ in Eq. (6) and _𝛽_ in Eq. (8) play an important role for modeling. Specifically, _𝛼_ balances the weight of symbol prediction and formula selection when training FOMAS. _𝛽_ controls the preference to focus more on the reasoning goal or the formula itself when selecting the most appropriate formula for reasoning steps. Figure 5 shows the performances of _𝛼_ ∈{0 _._ 01 _,_ 0 _._ 05 _,_ 0 _._ 1 _,_ 0 _._ 5 _,_ 1} and _𝛽_ ∈{0 _._ 1 _,_ 0 _._ 2 _,_ 0 _._ 5 _,_ 1 _,_ 5}. 

As _𝛼_ increases, the accuracy first increases, but then decreases. It indicates that properly balancing the objectives of symbol prediction and formula selection is beneficial. Besides, the peak of performance is attained when _𝛼_ = 0 _._ 5 and 0 _._ 05 on Math23K-F and MAWPS-F, respectively. This suggests the high correlation between the difficulty of mastering the application of formulas and the difficulty of datasets, where a simple one (i.e., MAWPS-F) is easier for knowledge learning (i.e., requires a smaller regularization _𝛼_ ). _𝛽_ shows a similar trend, requiring a precise balance between concerns on the reasoning goal and the formula. In particular, since the result is better when _𝛽 <_ 0 _._ 5, it implies that compared with the reasoning goal, the relevance of the formula to the problem sentence should be paid more attention in the selection mechanism. 

## **5.3 Analyses of FOMAS** 

_5.3.1 Formula Learning._ For formula learning, we expect FOMAS to comprehensively grasp the meanings of formulas, which we verify by investigating the distribution of different formulas’ representations. Specifically, we take the semantic vectors of the root _𝒔𝑟_ of all formulas and their variants after pretraining Eq. (4), and visualize the 5 most frequent ones (with their variants) by T-SNE in Figure 7. Generally, we observe that formulas from different prototypes (i.e., in different colors) are well separated, while those with the same structure feature (e.g., “ _𝑟𝑎𝑡𝑒_ = _𝑤𝑜𝑟𝑘_ ÷ _𝑡𝑖𝑚𝑒_ ” and 

1484 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA. 

Guiding Mathematical Reasoning via Mastering Commonsense Formula Knowledge 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0009-02.png)


**----- Start of picture text -----**<br>
kilometers at  Problem 1 how long was Benjamin skating ?: If Benjamin skated �1 kilometers per hour , �� [[�,÷,BERT-Tree FOMAS��,��, �1�1]] ������� �����(÷): �.����(�): �.799= �. ��� ������������= ÷ ����� �����<br>**----- End of picture text -----**<br>


**Figure 6: Case 1 (please see Appendix C for cases 2 and 3). We visualize FOMAS’s formula-selected mechanism at** _𝑡_ = 1 **.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0009-04.png)


**----- Start of picture text -----**<br>
�����= ��������÷ ����<br>����= ��������÷ �����<br>����= ����× ����<br>����= ����÷ ���� ��������= �����× ����<br>����= ����÷ ����<br>**----- End of picture text -----**<br>


**Figure 7: Visualization of the 5 most frequently used formulas (with their variants in the same color) after pretraining Eq.** (4) **. The formula of id** 1 **-** 5 **is referred in Table 1.** 

“ _𝑡𝑖𝑚𝑒_ = _𝑤𝑜𝑟𝑘_ ÷ _𝑟𝑎𝑡𝑒_ ”) are relatively closer. It validates that FOMAS has advantages in formula learning through encoding the structural and lexical information separately. Besides, it also reflects FOMAS’s capability to keep the formula semantics well when conducting mathematical transformation, validating that our pretraining manner has acquired and maintained the knowledge behind it. 

_5.3.2 Formula Applying._ For formula application, one of the core steps of FOMAS is to select the most suitable formula by Eq. (9) in Reasoning System. Thus, we first quantify the performance of FOMAS’ formula-selected mechanism, which covers the correctness of the formula-inherited mechanism. Specifically, we report the _𝐴𝐶𝐶_ , _𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛_ , and _𝑅𝑒𝑐𝑎𝑙𝑙_ from both classification and top-1 ranking aspect and compare the results with “IP” that replaces the _𝑠𝑐𝑜𝑟𝑒_ ( _𝑟_ ) in Eq. (8) by inner product of _𝒔𝑟_ and _𝒒𝑡_ . From Table 5, our FOMAS achieves the best performances on all metrics, which verifies the effectiveness and robustness of our proposed mechanisms. Second, to more clearly show how FOMAS benefits from mastering the formulas for mathematical reasoning, we compare its solutions with BERT-Tree (“BT”) and calculate the _𝑃𝐼𝐹_ metric introduced in Section 3.1. As reported in Table 5, FOMAS significantly reduces this proportion of errors, which explicitly demonstrates that it has learned to use the formula knowledge to conduct more accurate reasoning. After a more detailed inspection of these results, we find that FOMAS performs better at formulas that occur more often, while 31 _._ 6% and 63 _._ 3% of its mistakes on Math23K-F and MAWPS-F are due to formulas appearing fewer than 30 times in the training set, respectively. Thus, we leave one of the possible future directions is to explore knowledge learning in a few-shot scenario. 

_5.3.3 Interpretability Verification._ Further, we conduct case study to illustrate the interpretable formula application process of FOMAS. We plot one case in Figure 6 and two cases in _Appendix C_ . For each case, we first report the problem sentence and the prefix expressions generated by BERT-Tree and FOMAS. Then, we visualize the output symbol distribution of formula-selected/inherited mechanism of FOMAS at steps where it corrects the mistakes of BERT-Tree. 

Specifically, for cases 1 and 3, FOMAS correctly reasons “÷” at _𝑡_ = 1, while BERT-Tree responds the wrong “−” (“×”) with a high probability 0 _._ 799 (0 _._ 974). From the visualized formula-selected 

**Table 5: Performances of formula application.** 

||Math23K-F|MAWPS-F|
|---|---|---|
||FOMAS<br>IP|FOMAS<br>IP|
|ACC(↑)<br>Precision(↑)<br>Recall(↑)|0_._954<br>0_._950<br>0_._749<br>0_._731<br>0_._687<br>0_._621|0_._961<br>0_._957<br>0_._808<br>0_._776<br>0_._758<br>0_._742|
||FOMAS<br>BT|FOMAS<br>BT|
|PIF(↓)|0_._112<br>0_._220|0_._134<br>0_._257|



mechanism at _𝑡_ = 1, we observe that FOMAS selects the formula “ _𝑡𝑖𝑚𝑒_ = _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ ÷ _𝑠𝑝𝑒𝑒𝑑_ ” and “ _𝑠𝑝𝑒𝑒𝑑_ = _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ ÷ _𝑡𝑖𝑚𝑒_ ” with _𝑃𝑠𝑒𝑙_ = 0 _._ 956 and 0 _._ 999 in cases 1 and 3 respectively, based on which extracts “÷” as _𝑦[𝑠𝑒𝑙]_ 1 and corrects the wrong symbol by confidencebased ensemble. This phenomenon confirms the necessity and effectiveness of our formula-selected mechanism and ensemble method. In particular, the selected two formulas are variants of the same “ _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑡𝑖𝑚𝑒_ × _𝑠𝑝𝑒𝑒𝑑_ ”. The precise use of them again verifies FOMAS’ mastery of formula semantics and mathematical logic. 

For case 2, although BERT-Tree generates the correct first two symbols _𝑦_ 1 = ×, _𝑦_ 2 = _𝑁_ 0, it still deduces the wrong _𝑦_ 3 as 0 _._ 01. Comparatively, FOMAS selects formula “ _𝑡𝑜𝑡𝑎𝑙_  𝑎𝑚𝑜𝑢𝑛𝑡_ = _𝑢𝑛𝑖𝑡_  𝑎𝑚𝑜𝑢𝑛𝑡_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟_ ” at the first reasoning step _𝑡_ = 1. Instructed by this formula, at _𝑡_ = 3, it inherits the information of _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟_ to calculate “the number of packages” and reasons _𝑦[𝑖𝑛ℎ]_ 3 = _𝑁_ 1 with probability 0 _._ 961 by formula-inherited mechanism. With the inherit probability _𝑃𝑖𝑛ℎ_ = 0 _._ 997 in confidence-based ensemble, FOMAS overrides the wrong _𝑃_ ( _𝑦[𝑑𝑖𝑟]_ 3 = 0 _._ 01) = 0 _._ 568 predicted by direct reasoning (i.e., BERT-Tree) and generates the correct _𝑦_ 3 = _𝑁_ 1. Thus, formula-inherited mechanism is a crucial and indispensable component of formula knowledge application in FOMAS. 

## **6 CONCLUSION AND FUTURE WORK** 

In this paper, we focused on the commonsense formula knowledge that is essential for mathematical reasoning. Specifically, we first constructed two benchmark datasets named Math23K-F and MAWPS-F with precise annotations of formula usage to support our study, which are quite general to benefit future research in this field. Then, we proposed a novel Formula-mastered Solver (FOMAS) that contained Knowledge-Reasoning Systems inspired by human cognitive structure and elaborate formula learning/applying mechanisms. Experiments verified FOMAS’ improvements on reasoning accuracy and interpretability, and validated the necessity of formula knowledge for robust reasoning. In the future, we will extend FOMAS to acquire more types of symbolic knowledge from data automatically and generalize to more datasets. 

**Acknowledgement.** This research was partial supported by grants from the National Key Research and Development Program of China (Grant No. 2021YFF0901005) and the National Natural Science Foundation of China (Grants No. 62106244, No. U20A20229). 

1485 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA. 

Jiayu Liu et al. 

## **REFERENCES** 

- [1] Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. 2019. MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms. In _Proceedings of NAACLHLT_ . 2357–2367. 

- [2] Yefim Bakman. 2007. Robust understanding of word problems with extraneous information. _arXiv preprint math/0701393_ (2007). 

- [3] Kewei Cheng, Jiahao Liu, Wei Wang, and Yizhou Sun. 2022. RLogic: Recursive Logical Rule Learning from Knowledge Graphs. In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ . 179–189. 

- [4] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. 2021. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ (2021). 

- [5] Alex Davies, Petar Veličković, Lars Buesing, Sam Blackwell, Daniel Zheng, Tomašev, et al. 2021. Advancing mathematics by guiding human intuition with AI. _Nature_ 600, 7887 (2021), 70–74. 

- [6] Robert Benjamin Davis. 1984. _Learning mathematics: The cognitive science approach to mathematics education_ . Greenwood Publishing Group. 

- [7] Michael AE Dummett. 1995. _Frege’s philosophy of mathematics_ . Harvard University Press. 

- [8] Jonathan St BT Evans. 2008. Dual-processing accounts of reasoning, judgment, and social cognition. _Annu. Rev. Psychol._ 59 (2008), 255–278. 

- [9] Jonathan St BT Evans, Stephen E Newstead, and Ruth MJ Byrne. 1993. _Human reasoning: The psychology of deduction_ . Psychology Press. 

- [10] Edward A Feigenbaum, Julian Feldman, et al. 1963. _Computers and thought_ . New York McGraw-Hill. 

- [11] Charles R Fletcher. 1985. Understanding and solving arithmetic word problems: A computer simulation. _Behavior Research Methods, Instruments, & Computers_ 17, 5 (1985), 565–571. 

- [12] Josep Maria Font, Ramon Jansana, and Don Pigozzi. 2003. A survey of abstract algebraic logic. _Studia Logica: An International Journal for Symbolic Logic_ 74, 1/2 (2003), 13–97. 

- [13] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ (2021). 

- [14] Marjorie Henningsen and Mary Kay Stein. 1997. Mathematical tasks and student cognition: Classroom-based factors that support and inhibit high-level mathematical thinking and reasoning. _Journal for research in mathematics education_ 28, 5 (1997), 524–549. 

- [15] Zijian Huang, Meng-Fen Chiang, and Wang-Chien Lee. 2022. LinE: Logical Query Reasoning over Hierarchical Knowledge Graphs. In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ . 615–625. 

- [16] Zhenya Huang, Xin Lin, Hao Wang, Qi Liu, Enhong Chen, Jianhui Ma, Yu Su, and Wei Tong. 2021. Disenqnet: Disentangled representation learning for educational questions. In _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining_ . 696–704. 

- [17] Zhenya Huang, Qi Liu, Weibo Gao, Jinze Wu, Yu Yin, Hao Wang, and Enhong Chen. 2020. Neural mathematical solver with enhanced formula structure. In _SIGIR_ . 1729–1732. 

- [18] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. 1991. Adaptive mixtures of local experts. _Neural computation_ 3, 1 (1991), 79–87. 

- [19] Zhanming Jie, Jierui Li, and Wei Lu. 2022. Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction. In _ACL_ . 5944–5955. 

- [20] Daniel Kahneman. 2011. _Thinking, fast and slow_ . Macmillan. 

- [21] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In _Proceedings of NAACL-HLT_ . 4171–4186. 

- [22] Hyunju Kim, Junwon Hwang, Taewoo Yoo, and Yun-Gyung Cheong. 2022. Improving a Graph-to-Tree Model for Solving Math Word Problems. In _IMCOM_ . 1–7. 

- [23] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ (2014). 

- [24] Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. 2016. MAWPS: A math word problem repository. In _Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ . 1152–1157. 

- [25] George Lakoff and Rafael Núñez. 2000. _Where mathematics comes from_ . Vol. 6. New York: Basic Books. 

- [26] Zhongli Li, Wenxuan Zhang, Chao Yan, Qingyu Zhou, Chao Li, Hongzhi Liu, and Yunbo Cao. 2022. Seeking Patterns, Not just Memorizing Procedures: Contrastive Learning for Solving Math Word Problems. In _Findings of the Association for Computational Linguistics: ACL 2022_ . 2486–2496. 

- [27] Zhenwen Liang, Jipeng Zhang, Lei Wang, Wei Qin, Yunshi Lan, Jie Shao, and Xiangliang Zhang. 2022. MWP-BERT: Numeracy-augmented pre-training for math word problem solving. In _NAACL-HLT (Findings)_ . 997–1009. 

- [28] Xin Lin, Zhenya Huang, Hongke Zhao, Enhong Chen, Qi Liu, Defu Lian, Xin Li, and Hao Wang. 2023. Learning Relation-Enhanced Hierarchical Solver for Math Word Problems. _IEEE Transactions on Neural Networks and Learning Systems_ (2023). 

- [29] Xin Lin, Zhenya Huang, Hongke Zhao, Enhong Chen, Qi Liu, Hao Wang, and Shijin Wang. 2021. HMS: A Hierarchical Solver with Dependency-Enhanced Understanding for Math Word Problem. In _AAAI_ , Vol. 35. 4232–4240. 

- [30] Zi Lin, Jeremiah Zhe Liu, and Jingbo Shang. 2022. Towards Collaborative NeuralSymbolic Graph Semantic Parsing via Uncertainty. In _Findings of the Association for Computational Linguistics: ACL 2022_ . 4160–4173. 

- [31] Jiayu Liu, Zhenya Huang, Xin Lin, Qi Liu, Jianhui Ma, and Enhong Chen. 2022. A Cognitive Solver with Autonomously Knowledge Learning for Reasoning Mathematical Answers. In _2022 IEEE International Conference on Data Mining (ICDM)_ . IEEE, 269–278. 

- [32] Jiayu Liu, Zhenya Huang, Chengxiang Zhai, and Qi Liu. 2023. Learning by Applying: A General Framework for Mathematical Reasoning via Enhancing Explicit Knowledge Learning. _arXiv preprint arXiv:2302.05717_ (2023). 

- [33] Qi Liu, Zhenya Huang, Yu Yin, Enhong Chen, Hui Xiong, Yu Su, and Guoping Hu. 2021. EKT: Exercise-Aware Knowledge Tracing for Student Performance Prediction. _IEEE Transactions on Knowledge and Data Engineering_ 33, 1 (2021), 100–115. 

- [34] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-chun Zhu. 2021. Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning. In _ACL/IJCNLP_ . 6774–6786. 

- [35] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. 2015. Effective Approaches to Attention-based Neural Machine Translation. In _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ . 1412–1421. 

- [36] Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B Tenenbaum, and Jiajun Wu. 2019. The neuro-symbolic concept learner: Interpreting scenes, words, and sentences from natural supervision. In _ICLR_ . 

- [37] Arindam Mitra and Chitta Baral. 2016. Learning to use formulas to solve simple arithmetic problems. In _ACL_ . 2144–2153. 

- [38] Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP Models really able to Solve Simple Math Word Problems?. In _NAACL-HLT_ . 2080–2094. 

- [39] Shuai Peng, Di Fu, Yong Cao, Yijun Liang, Gu Xu, Liangcai Gao, and Zhi Tang. 2022. Compute Like Humans: Interpretable Step-by-step Symbolic Computation with Deep Neural Network. In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ . 1348–1357. 

- [40] Shuai Peng, Ke Yuan, Liangcai Gao, and Zhi Tang. 2021. Mathbert: A pre-trained model for mathematical formula understanding. _arXiv preprint arXiv:2105.00377_ (2021). 

- [41] Jinghui Qin, Xiaodan Liang, Yining Hong, Jianheng Tang, and Liang Lin. 2021. Neural-Symbolic Solver for Math Word Problems with Auxiliary Tasks. In _ACL/IJCNLP_ . 5870–5881. 

- [42] Meng Qu and Jian Tang. 2019. Probabilistic logic neural networks for reasoning. _Advances in neural information processing systems_ 32 (2019). 

- [43] Shuming Shi, Yuehui Wang, Chin-Yew Lin, Xiaojiang Liu, and Yong Rui. 2015. Automatically solving number word problems by semantic parsing and reasoning. In _EMNLP_ . 1132–1142. 

- [44] Steven A Sloman, Richard Patterson, and Aron K Barbey. 2021. Cognitive neuroscience meets the community of knowledge. _Frontiers in Systems Neuroscience_ (2021), 120. 

- [45] Matthias Steup and Ram Neta. 2005. Epistemology. (2005). 

- [46] Bin Wang, Jiangzhou Ju, Yang Fan, Xin-Yu Dai, Shujian Huang, and Jiajun Chen. 2022. Structure-Unified M-Tree Coding Solver for MathWord Problem. _EMNLP_ . 

- [47] Yan Wang, Xiaojiang Liu, and Shuming Shi. 2017. Deep neural solver for math word problems. In _EMNLP_ . 845–854. 

- [48] Qinzhuo Wu, Qi Zhang, Jinlan Fu, and Xuan-Jing Huang. 2020. A knowledgeaware sequence-to-tree network for math word problem solving. In _EMNLP_ . 7137–7146. 

- [49] Zhipeng Xie and Shichao Sun. 2019. A Goal-Driven Tree-Structured Neural Model for Math Word Problems.. In _IJCAI_ . 5299–5305. 

- [50] Zhicheng Yang, Jinghui Qin, Jiaqi Chen, Liang Lin, and Xiaodan Liang. 2022. LogicSolver: Towards Interpretable Math Word Problem Solving with Logical Prompt-enhanced Learning. In _Findings of EMNLP_ . 

- [51] Dongran Yu, Bo Yang, Dayou Liu, and Hui Wang. 2021. A Survey on Neuralsymbolic Systems. _arXiv preprint arXiv:2111.08164_ (2021). 

- [52] Dongran Yu, Bo Yang, Qianhao Wei, Anchen Li, and Shirui Pan. 2022. A Probabilistic Graphical Model Based on Neural-Symbolic Reasoning for Visual Relationship Detection. In _CVPR_ . 10609–10618. 

- [53] Dongxiang Zhang, Lei Wang, et al. 2020. The Gap of Semantic Parsing: A Survey on Automatic Math Word Problem Solvers. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 42, 9 (2020), 2287–2305. 

- [54] Jipeng Zhang, Lei Wang, Roy Ka-Wei Lee, et al. 2020. Graph-to-Tree Learning for Solving Math Word Problems. In _ACL_ . 3928–3937. 

- [55] Hongke Zhao, Chuang Zhao, Xi Zhang, Nanlin Liu, Hengshu Zhu, Qi Liu, and Hui Xiong. 2023. An Ensemble Learning Approach with Gradient Resampling for Class-Imbalance Problems. _INFORMS Journal on Computing_ (2023). 

- [56] Wayne Xin Zhao, Kun Zhou, Zheng Gong, et al. 2022. JiuZhang: A Chinese Pretrained Language Model for Mathematical Problem Understanding. In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ . 4571–4581. 

1486 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA. 

Guiding Mathematical Reasoning via Mastering Commonsense Formula Knowledge 

**Table 6: All 51 math formulas on Math23K-F.** 

_𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒 𝑤𝑜𝑟𝑘_ = _𝑟𝑎𝑡𝑒_ × _𝑡𝑖𝑚𝑒 𝑐𝑜𝑛𝑠𝑢𝑚𝑒_ = _𝑐𝑜𝑛𝑠𝑢𝑚𝑒_  𝑟𝑎𝑡𝑒_ × _𝑡𝑖𝑚𝑒 𝑡𝑜𝑡𝑎𝑙_  𝑡𝑖𝑚𝑒_ = _𝑢𝑛𝑖𝑡_  𝑡𝑖𝑚𝑒_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑡𝑎𝑙_  𝑤𝑒𝑖𝑔ℎ𝑡_ = _𝑢𝑛𝑖𝑡_  𝑤𝑒𝑖𝑔ℎ𝑡_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑡𝑎𝑙_  𝑐𝑜𝑠𝑡_ = _𝑢𝑛𝑖𝑡_  𝑐𝑜𝑠𝑡_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑡𝑎𝑙_  𝑣𝑜𝑙𝑢𝑚𝑒_ = _𝑢𝑛𝑖𝑡_  𝑣𝑜𝑙𝑢𝑚𝑒_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑡𝑎𝑙_  𝑖𝑛𝑐𝑜𝑚𝑒_ = _𝑢𝑛𝑖𝑡_  𝑖𝑛𝑐𝑜𝑚𝑒_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑡𝑎𝑙_  𝑎𝑚𝑜𝑢𝑛𝑡_ = _𝑢𝑛𝑖𝑡_  𝑎𝑚𝑜𝑢𝑛𝑡_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑎𝑡𝑙_  𝑎𝑟𝑒𝑎_ = _𝑢𝑛𝑖𝑡_  𝑎𝑟𝑒𝑎_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑡𝑎𝑙_  𝑙𝑒𝑛𝑔𝑡ℎ_ = _𝑢𝑛𝑖𝑡_  𝑙𝑒𝑛𝑔𝑡ℎ_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑟𝑒𝑐𝑡𝑎𝑛𝑔𝑙𝑒_  𝑎𝑟𝑒𝑎_ = _𝑙𝑒𝑛𝑔𝑡ℎ_ × _𝑤𝑖𝑑𝑡ℎ 𝑟𝑒𝑐𝑡𝑎𝑛𝑔𝑙𝑒_  𝑝𝑒𝑟𝑖𝑚𝑒𝑡𝑒𝑟_ = ( _𝑙𝑒𝑛𝑔𝑡ℎ_ + _𝑤𝑖𝑑𝑡ℎ_ ) × 2 _𝑠𝑞𝑢𝑎𝑟𝑒_  𝑎𝑟𝑒𝑎_ = _𝑠𝑖𝑑𝑒_ × _𝑠𝑖𝑑𝑒 𝑠𝑞𝑢𝑎𝑟𝑒_  𝑝𝑒𝑟𝑖𝑚𝑒𝑡𝑒𝑟_ = _𝑠𝑖𝑑𝑒_ × 4 _𝑐𝑖𝑟𝑐𝑙𝑒_  𝑝𝑒𝑟𝑖𝑚𝑒𝑡𝑒𝑟_ = 2 × _𝜋_ × _𝑟𝑎𝑑𝑖𝑢𝑠 𝑐𝑖𝑟𝑐𝑙𝑒_  𝑝𝑒𝑟𝑖𝑚𝑒𝑡𝑒𝑟_ = _𝜋_ × _𝑑𝑖𝑎𝑚𝑒𝑡𝑒𝑟 𝑐𝑖𝑟𝑐𝑙𝑒_  𝑎𝑟𝑒𝑎_ = _𝜋_ × _𝑟𝑎𝑑𝑖𝑢𝑠_ ∧ 2 _𝑐𝑖𝑟𝑐𝑙𝑒_  𝑎𝑟𝑒𝑎_ = _𝜋_ × _𝑟𝑎𝑑𝑖𝑢𝑠_ × _𝑟𝑎𝑑𝑖𝑢𝑠 𝑐𝑖𝑟𝑐𝑙𝑒_  𝑎𝑟𝑒𝑎_ = _𝜋_ × ( _𝑑𝑖𝑎𝑚𝑒𝑡𝑒𝑟_ ÷ 2) ∧ 2 × _𝑟𝑎𝑑𝑖𝑢𝑠 𝑝𝑎𝑟𝑎𝑙𝑙𝑒𝑙𝑜𝑔𝑟𝑎𝑚_  𝑎𝑟𝑒𝑎_ = _𝑙𝑒𝑛𝑔𝑡ℎ_ × _ℎ𝑒𝑖𝑔ℎ𝑡 𝑡𝑟𝑖𝑎𝑛𝑔𝑙𝑒_  𝑎𝑟𝑒𝑎_ = _𝑙𝑒𝑛𝑔𝑡ℎ_ × _ℎ𝑒𝑖𝑔ℎ𝑡_ ÷ 2 _𝑡𝑟𝑎𝑝𝑒𝑧𝑖𝑢𝑚_  𝑎𝑟𝑒𝑎_ = ( _𝑠𝑖𝑑𝑒_ 1 + _𝑠𝑖𝑑𝑒_ 2) × _ℎ𝑒𝑖𝑔ℎ𝑡_ ÷ 2 _𝑡𝑟𝑎𝑝𝑒𝑧𝑖𝑢𝑚_  𝑎𝑟𝑒𝑎_ = _𝑝𝑙𝑢𝑠_  𝑠𝑖𝑑𝑒_ 1 𝑠𝑖𝑑𝑒_ 2 × _ℎ𝑒𝑖𝑔ℎ𝑡_ ÷ 2 _𝑐𝑢𝑏𝑖𝑜𝑑_  𝑣𝑜𝑙𝑢𝑚𝑒_ = _𝑏𝑎𝑠𝑒_  𝑎𝑟𝑒𝑎_ × _ℎ𝑒𝑖𝑔ℎ𝑡 𝑐𝑢𝑏𝑖𝑜𝑑_  𝑣𝑜𝑙𝑢𝑚𝑒_ = _𝑙𝑒𝑛𝑔𝑡ℎ_ × _𝑤𝑖𝑑𝑡ℎ_ × _ℎ𝑒𝑖𝑔ℎ𝑡 𝑐𝑢𝑏𝑒_  𝑣𝑜𝑙𝑢𝑚𝑒_ = _𝑠𝑖𝑑𝑒_ × _𝑠𝑖𝑑𝑒_ × _𝑠𝑖𝑑𝑒 𝑐𝑜𝑛𝑒_  𝑣𝑜𝑙𝑢𝑚𝑒_ = _𝑏𝑎𝑠𝑒_  𝑎𝑟𝑒𝑎_ × _ℎ𝑒𝑖𝑔ℎ𝑡_ ÷ 3 _𝑐𝑦𝑙𝑖𝑛𝑑𝑒𝑟_  𝑎𝑟𝑒𝑎_ = 2 × _𝜋_ × _𝑟𝑎𝑑𝑖𝑢𝑠_ × _ℎ𝑒𝑖𝑔ℎ𝑡 𝑐𝑦𝑙𝑖𝑛𝑑𝑒𝑟_  𝑎𝑟𝑒𝑎_ = _𝜋_ × _𝑑𝑖𝑎𝑚𝑒𝑡𝑒𝑟_ × _ℎ𝑒𝑖𝑔ℎ𝑡 𝑐𝑦𝑙𝑖𝑛𝑑𝑒𝑟_  𝑎𝑟𝑒𝑎_ = _𝑏𝑎𝑠𝑒_  𝑝𝑒𝑟𝑖𝑚𝑒𝑡𝑒𝑟_ × _ℎ𝑒𝑖𝑔ℎ𝑡 𝑐𝑦𝑙𝑖𝑛𝑑𝑒𝑟_  𝑣𝑜𝑙𝑢𝑚𝑛𝑒_ = _𝑟𝑎𝑑𝑖𝑢𝑠_ ∧ 2 × _𝜋_ × _ℎ𝑒𝑖𝑔ℎ𝑡 𝑐𝑦𝑙𝑖𝑛𝑑𝑒𝑟_  𝑣𝑜𝑙𝑢𝑚𝑛𝑒_ = ( _𝑑𝑖𝑎𝑚𝑒𝑡𝑒𝑟_ ÷ 2) ∧ 2 × _𝜋_ × _ℎ𝑒𝑖𝑔ℎ𝑡 𝑐𝑦𝑙𝑖𝑛𝑑𝑒𝑟_  𝑣𝑜𝑙𝑢𝑚𝑒_ = _𝑏𝑎𝑠𝑒_  𝑎𝑟𝑒𝑎_ × _ℎ𝑒𝑖𝑔ℎ𝑡 𝑐𝑦𝑙𝑖𝑛𝑑𝑒𝑟_  𝑣𝑜𝑙𝑢𝑚𝑒_ = _𝜋_ × _𝑟𝑎𝑑𝑖𝑢𝑠_ × _𝑟𝑎𝑑𝑖𝑢𝑠_ × _ℎ𝑒𝑖𝑔ℎ𝑡 𝑖𝑛𝑡𝑒𝑟𝑒𝑠𝑡_ = _𝑝𝑟𝑖𝑛𝑐𝑖𝑝𝑙𝑒_ × _𝑟𝑎𝑡𝑒_ × _𝑡𝑖𝑚𝑒 𝑖𝑛𝑐𝑜𝑚𝑒_ = _𝑠𝑎𝑣𝑖𝑛𝑔𝑠_ + _𝑒𝑥𝑝𝑒𝑛𝑑𝑖𝑡𝑢𝑟𝑒 𝑠𝑒𝑙𝑙_  𝑝𝑟𝑖𝑐𝑒_ = _𝑝𝑟𝑜𝑓𝑖𝑡_ + _𝑐𝑜𝑠𝑡_  𝑝𝑟𝑖𝑐𝑒 𝑠𝑒𝑙𝑙_  𝑝𝑟𝑖𝑐𝑒_ = _𝑐𝑜𝑠𝑡_  𝑝𝑟𝑖𝑐𝑒_ × (1 + _𝑝𝑟𝑜𝑓𝑖𝑡_ %) _𝑠𝑒𝑙𝑙_  𝑝𝑟𝑖𝑐𝑒_ = _𝑐𝑜𝑠𝑡_  𝑝𝑟𝑖𝑐𝑒_ + _𝑐𝑜𝑠𝑡_  𝑝𝑟𝑖𝑐𝑒_ × _𝑝𝑟𝑜𝑓𝑖𝑡_ % _𝑡𝑎𝑥_  𝑎𝑚𝑜𝑢𝑛𝑡_ = _𝑡𝑎𝑥_  𝑖𝑛𝑐𝑜𝑚𝑒_ × _𝑡𝑎𝑥_  𝑟𝑎𝑡𝑒 𝑡𝑎𝑥_  𝑖𝑛𝑐𝑜𝑚𝑒_ = _𝑠𝑎𝑙𝑎𝑟𝑦_ − _𝑏𝑎𝑠𝑒 𝑎𝑐𝑡𝑢𝑎𝑙_  𝑠𝑎𝑙𝑎𝑟𝑦_ = _𝑠𝑎𝑙𝑎𝑟𝑦_ − _𝑡𝑎𝑥_  𝑎𝑚𝑜𝑢𝑛𝑡 𝑎𝑣𝑒𝑟𝑎𝑔𝑒_ = _𝑠𝑢𝑚_  𝑜𝑓_  𝑡𝑒𝑟𝑚𝑠_ ÷ _𝑛𝑢𝑚𝑏𝑒𝑟_  𝑜𝑓_  𝑡𝑒𝑟𝑚𝑠 𝑝𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦_ = _𝑛𝑢𝑚𝑏𝑒𝑟_  𝑜𝑓_  𝑑𝑒𝑠𝑖𝑟𝑒𝑑_ ÷ _𝑛𝑢𝑚𝑏𝑒𝑟_  𝑝𝑜𝑠𝑠𝑖𝑏𝑖𝑙𝑖𝑡𝑦 𝑝𝑒𝑟𝑐𝑒𝑛𝑡_  𝑐ℎ𝑎𝑛𝑔𝑒_ % = ( _𝑓𝑖𝑛𝑎𝑙_ − _𝑖𝑛𝑖𝑡𝑖𝑎𝑙_ ) ÷ _𝑖𝑛𝑖𝑡𝑖𝑎𝑙 𝑙𝑎𝑠𝑡_  𝑡𝑒𝑟𝑚_ = _𝑓𝑖𝑟𝑠𝑡_  𝑡𝑒𝑟𝑚_ + ( _𝑛_ − 1) × _𝑐𝑜𝑚𝑚𝑜𝑛_  𝑑𝑖𝑓𝑓𝑒𝑟𝑒𝑛𝑐𝑒 𝑤𝑒𝑖𝑔ℎ𝑡_ = _𝑑𝑒𝑛𝑠𝑖𝑡𝑦_ × _𝑣𝑜𝑙𝑢𝑚𝑒 𝑤𝑒𝑖𝑔ℎ𝑡_  𝑠𝑢𝑏𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑤𝑒𝑖𝑔ℎ𝑡_  𝑎𝑙𝑙_ × _𝑐𝑜𝑛𝑐𝑒𝑛𝑡𝑟𝑎𝑡𝑖𝑜𝑛_ % 2 _𝑛𝑑_  𝑛𝑢𝑚𝑏𝑒𝑟_ = _𝑙𝑐𝑚_ × _ℎ𝑐𝑓_ ÷ 1 _𝑠𝑡_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑟𝑒𝑐𝑖𝑝𝑟𝑜𝑐𝑎𝑙_ = 1 ÷ _𝑛𝑢𝑚𝑏𝑒𝑟_ 

## **A MORE DATASET ANALYSES** 

Tables 6 and 7 list all math formulas on our benchmark datasets, and the distributions of their usage frequency are visualized in Figure 8. Besides, we count the number of used formulas for each problem and report the distribution in Table 8. It indicates that 33.5% and 38.4% problems require at least one formula on Math23K-F and MAWPS-F, respectively, as we have reported in Section 3.1. Especially, those requiring one formula (i.e., 4 _,_ 520 and 860) account 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0011-06.png)


**Figure 8: Distributions of formula usage frequency.** 

**Table 7: All 18 math formulas on MAWPS-F.** 

_𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ = _𝑠𝑝𝑒𝑒𝑑_ × _𝑡𝑖𝑚𝑒 𝑤𝑜𝑟𝑘_ = _𝑟𝑎𝑡𝑒_ × _𝑡𝑖𝑚𝑒 𝑐𝑜𝑛𝑠𝑢𝑚𝑒_ = _𝑐𝑜𝑛𝑠𝑢𝑚𝑒_  𝑟𝑎𝑡𝑒_ × _𝑡𝑖𝑚𝑒 𝑡𝑜𝑡𝑎𝑙_  𝑡𝑖𝑚𝑒_ = _𝑢𝑛𝑖𝑡_  𝑡𝑖𝑚𝑒_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑡𝑎𝑙_  𝑤𝑒𝑖𝑔ℎ𝑡_ = _𝑢𝑛𝑖𝑡_  𝑤𝑒𝑖𝑔ℎ𝑡_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑡𝑎𝑙_  𝑐𝑜𝑠𝑡_ = _𝑢𝑛𝑖𝑡_  𝑐𝑜𝑠𝑡_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑡𝑎𝑙_  𝑣𝑜𝑙𝑢𝑚𝑒_ = _𝑢𝑛𝑖𝑡_  𝑣𝑜𝑙𝑢𝑚𝑒_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑡𝑎𝑙_  𝑖𝑛𝑐𝑜𝑚𝑒_ = _𝑢𝑛𝑖𝑡_  𝑖𝑛𝑐𝑜𝑚𝑒_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑡𝑜𝑡𝑎𝑙_  𝑎𝑚𝑜𝑢𝑛𝑡_ = _𝑢𝑛𝑖𝑡_  𝑎𝑚𝑜𝑢𝑛𝑡_ × _𝑡𝑜𝑡𝑎𝑙_  𝑛𝑢𝑚𝑏𝑒𝑟 𝑟𝑒𝑐𝑡𝑎𝑛𝑔𝑙𝑒_  𝑎𝑟𝑒𝑎_ = _𝑙𝑒𝑛𝑔𝑡ℎ_ × _𝑤𝑖𝑑𝑡ℎ 𝑖𝑛𝑡𝑒𝑟𝑒𝑠𝑡_ = _𝑝𝑟𝑖𝑛𝑐𝑖𝑝𝑙𝑒_ × _𝑟𝑎𝑡𝑒_ × _𝑡𝑖𝑚𝑒 𝑖𝑛𝑐𝑜𝑚𝑒_ = _𝑠𝑎𝑣𝑖𝑛𝑔𝑠_ + _𝑒𝑥𝑝𝑒𝑛𝑑𝑖𝑡𝑢𝑟𝑒 𝑠𝑒𝑙𝑙_  𝑝𝑟𝑖𝑐𝑒_ = _𝑝𝑟𝑜𝑓𝑖𝑡_ + _𝑐𝑜𝑠𝑡_  𝑝𝑟𝑖𝑐𝑒 𝑠𝑒𝑙𝑙_  𝑝𝑟𝑖𝑐𝑒_ = _𝑐𝑜𝑠𝑡_  𝑝𝑟𝑖𝑐𝑒_ × (1 + _𝑝𝑟𝑜𝑓𝑖𝑡_ %) _𝑎𝑣𝑒𝑟𝑎𝑔𝑒_ = _𝑠𝑢𝑚_  𝑜𝑓_  𝑡𝑒𝑟𝑚𝑠_ ÷ _𝑛𝑢𝑚𝑏𝑒𝑟_  𝑜𝑓_  𝑡𝑒𝑟𝑚𝑠 𝑝𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦_ = _𝑛𝑢𝑚𝑏𝑒𝑟_  𝑜𝑓_  𝑑𝑒𝑠𝑖𝑟𝑒𝑑_ ÷ _𝑛𝑢𝑚𝑏𝑒𝑟_  𝑝𝑜𝑠𝑠𝑖𝑏𝑖𝑙𝑖𝑡𝑦 𝑝𝑒𝑟𝑐𝑒𝑛𝑡_  𝑐ℎ𝑎𝑛𝑔𝑒_ = ( _𝑓𝑖𝑛𝑎𝑙_ − _𝑖𝑛𝑖𝑡𝑖𝑎𝑙_ ) ÷ _𝑖𝑛𝑖𝑡𝑖𝑎𝑙_ × 100 _𝑙𝑎𝑠𝑡_  𝑡𝑒𝑟𝑚_ = _𝑓 𝑖𝑟𝑠𝑡_  𝑡𝑒𝑟𝑚_ + ( _𝑛_ − 1) × _𝑐𝑜𝑚𝑚𝑜𝑛_  𝑑𝑖 𝑓𝑓 𝑒𝑟𝑒𝑛𝑐𝑒_ 

**Table 8: Distributions of the number of used formulas.** 

|# Used Formulas|# Problems(Math23K-F)|# Problems(MAWPS-F)|
|---|---|---|
|0|14,412|1,462|
|1|4,520|860|
|2|3,005|33|
|More than 2|225|18|



for 19.5% and 36.2% on each dataset. This result verifies that formulas are very important for reasoning expressions correctly, thus validating the motivation and contribution of our work. 

## **B CHATGPT: EXPERIMENTS AND ANALYSES** 

ChatGPT is a recently released AI chatbot developed by OpenAI (https://openai.com/), which has garnered significant attention due to its strong language generation capability. Equipped with a vast array of factual knowledge, it is capable of responding problems across domains such as history, culture, and science. 

Here, we are interested in evaluating ChatGPT’s performance in answering math word problems. We conducted this experiment in January, 2023, and OpenAI did not publish the code API at that time. Therefore, we can only access the official website tool (https://openai.com/blog/chatgpt/) to obtain its responses to all problems (by a crawler), and then calculate the answers. Specifically, we show an example in Figure 10. The problem (“Maggi had ... left?”) is input into its online chatbox without any restrictions or 

1487 

KDD ’23, August 6–10, 2023, Long Beach, CA, USA. 

Jiayu Liu et al. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0012-02.png)


**----- Start of picture text -----**<br>
Problem 2 : Jose strolled to Jane ‘s  BERT-Tree ���� = �. ��� =<br>house . It is �� miles from Jose 's house  [×, ��, �1] �� ���<br>to Jane 's house . It took Jose �1 hours  FOMAS �(×): �.974 ����� ÷ ��<br>to get there . How fast did Jose go ? [÷, ��, �1] ����� �(÷): �.��� �������� ����<br>BERT-Tree  � �.��: �.���<br>Problem 3 : White t - shirts can be  �� = ��� ��<br>purchased in packages of �� . If Mom  [×, ��, �.��] �� � �� ��.428<br>buys �1 packages , how many white t - FOMAS �����_������ ×<br>shirts will she have ? [×, ��, ��] ����_������ �����_������ ���� = �. ��� �����<br>**----- End of picture text -----**<br>


**Figure 9: Cases 2 and 3. For case 2, we visualize FOMAS’s formula-inherited mechanism at** _𝑡_ = 3 **. For case 3, we visualize the formula-selected mechanism at** _𝑡_ = 1 **.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0012-04.png)


**Figure 10: Example of ChatGPT’s response.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0012-06.png)


**----- Start of picture text -----**<br>
(Translation)<br>Problem : The number of chicken eggs is 120, and the number of<br>duck eggs is 1/6 greater than that of the chicken eggs. What is the<br>total number of eggs?<br>ChatGPT : The number of duck eggs is calculated as 120 * 1/6 = 20.<br>Thus, the total number of chicken and duck eggs is 120 + 20 = 140.<br>FOMAS : [+, ×, 120, +, 1, (1/6), 120]<br>**----- End of picture text -----**<br>


**Figure 11: A fraction case between ChatGPT and FOMAS.** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0012-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0012-09.png)


**----- Start of picture text -----**<br>
(Translation)<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf_images/kdd_2024_Guiding_Mathematical_Reasoning_via_Mastering_Commonsense_Formula_Knowledge.pdf.pdf-0012-10.png)


**----- Start of picture text -----**<br>
Problem : Uncle Wang made a deposit of 6000 yuan in a bank with a<br>3.24% annual interest rate for a 2-year period. What is the amount that<br>can be retrieved at maturity?<br>ChatGPT : The annual interest is 6000 × 3.24% = 193.44<br>So the gross interest is 193.44 × 2 = 386.88<br>Add it with the original 6000, the amount that can be<br>retrieved at maturity is 6000 + 386.88 = 6386.88<br>FOMAS : [+,  6000,  ×, ×,  6000,  3.24%,  2]<br>**----- End of picture text -----**<br>


**Figure 12: An arithemetic case between ChatGPT and FOMAS.** 

illustrations. We scrape the whole response (“Maggi had a totoal ... 7 cupcakes left.”) and manually extract the numeric answer (i.e., 7) as ChatGPT’s result. Moreover, since the responses of ChatGPT may be different with multiple trials, we operate the above processes three times to obtain the stable results for all problems. 

As reported in Table 3, ChatGPT obtains answer accuracy of 64 _._ 9% and 88 _._ 3% on Math23K-F and MAWPS-F respectively, which reflects that it has basic mathematical reasoning ability. Specially, the performance on Math23K-F indicates that ChatGPT may still lack sufficient Chinese MWP corpus, and has some room for improvement on the problems that require more reasoning steps (recall that Math23K-F is a Chinese benchmark dataset that contains more difficult problems than MAWPS-F through the analysis in Section 3.1). We also make case study on Math23K-F, and find that there are mainly two typical errors that should be concerned. First, it struggles to answer the problems that examine fraction or proportion as plotted in Figure 11. Second, its arithmetic capability remains to be further improved. It can derive the correct expression for a problem, but makes a mistake when calculating the final numeric answer, e.g., in Figure 12, it gets the wrong value 193 _._ 44 based on the correct expression 6000 × 3 _._ 24% (marked red). 

Besides, for a more sufficient comparison, we design another baseline named ChatGPT-P by prompting all our annotated formulas together with the problem sentence as the input of ChatGPT. Our prompt is: “[problem sentence]. You may use the following math formulas to solve this problem: [formulas]”, and we also manually extract the numeric answer from its response. ChatGPT-P obtains 58 _._ 4% and 89 _._ 2% answer accuracy on Math23K-F and MAWPS-F, respectively. After suspecting its outputs, we found that on Math23KF, ChatGPT-P might misunderstand the task after giving all the formulas. For example, it might just type out all formulas and output “\ _𝑛𝑠𝑝𝑒𝑒𝑑_ = _𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒_ ÷ _𝑡𝑖𝑚𝑒,_ \ _𝑛𝑟𝑎𝑡𝑒_ = _𝑤𝑜𝑟𝑘_ ÷ _𝑡𝑖𝑚𝑒, ..._ ” instead of actually solving the problem. We guess it is probably because the 51 formulas (reported in Table 2) in the dataset take up most of the place in the prompt and make ChatGPT frustrated with what to do, thus resulting in a performance drop. While for MAWPS-F that has fewer (i.e., 18) formulas, prompting can improve the answer accuracy, which shows the necessity of studying formula knowledge on mathematical reasoning in this paper, and verifies the validity of our manually constructed formulas from another perspective. The different performances above inspire us to explore how to combine large-scale language models with external knowledge without introducing too much noise. We think it will be an interesting and valuable research direction in the future. 

## **C CASES 2 AND 3** 

We show one case in Section 5.3.3 and another two cases in Figure 9. Pleaser refer to Section 5.3.3 for detailed explanations and analyses. 

1488 

