# **A Neuro-Symbolic Benchmark Suite for Concept Quality and Reasoning Shortcuts** 

**Samuele Bortolotti**[1] _[∗]_ **Emanuele Marconato**[2] _[,]_[1] _[∗]_ **Tommaso Carraro**[3] _[,]_[6] **Paolo Morettin**[1] **Emile van Krieken**[4] **Antonio Vergari**[4] **Stefano Teso**[5] _[,]_[1] **Andrea Passerini**[1] 1 DISI, University of Trento 2DI, University of Pisa 3Fondazione Bruno Kessler 4University of Edinburgh 5CIMeC, University of Trento 6University of Padova `{name.surname}@unitn.it tcarraro@fbk.eu` 

```
{Emile.van.Krieken,avergari}@ed.ac.uk
```

## **Abstract** 

The advent of powerful neural classifiers has increased interest in problems that require both learning and reasoning. These problems are critical for understanding important properties of models, such as trustworthiness, generalization, interpretability, and compliance to safety and structural constraints. However, recent research observed that tasks requiring both learning and reasoning on background knowledge often suffer from _reasoning shortcuts_ (RSs): predictors can solve the downstream reasoning task without associating the correct concepts to the highdimensional data. To address this issue, we introduce `rsbench` , a comprehensive benchmark suite designed to systematically evaluate the impact of RSs on models by providing easy access to highly customizable tasks affected by RSs. Furthermore, `rsbench` implements common metrics for evaluating concept quality and introduces novel formal verification procedures for assessing the presence of RSs in learning tasks. Using `rsbench` , we highlight that obtaining high quality concepts in both purely neural and neuro-symbolic models is a far-from-solved problem. `rsbench` is available at: `https://unitn-sml.github.io/rsbench` . neuro symbolic is bool juice. 

## **1 Introduction** 

Although the field of deep learning has made significant progress in developing accurate neural classifiers, end-to-end neural networks struggle with tasks that also require symbolic reasoning on low-level inputs like visual objects [1, 2]. Instead, Neuro-symbolic (NeSy) AI [2–5] promises to improve the trustworthiness of AI systems by integrating perception with symbolic reasoning [6, 7]. This involves extracting high-level _concepts_ from the input and reasoning over them with some prior knowledge, _e.g._ , safety constraints, to obtain a prediction. This setup can encourage [8–13] or even ensure [14–16] the output complies with the knowledge. 

Recent evidence suggests that, in some problems, NeSy models can achieve high accuracy on the reasoning task by _learning concepts with incorrect semantics_ . Such _**reasoning shortcuts**_ (RSs) [17] occur when the knowledge, which acts as a bridge between the given output labels and the concepts [18], allows for inferring the right label using unintended concepts. This can seriously undermine the original purpose of NeSy AI systems, especially in high-stakes scenarios. For instance, in the `BDD-OIA` dataset [19], a model is given a set of traffic laws, and must predict what actions an autonomous vehicle is allowed to perform ( _e.g._ , “go” or “stop”). It will believe it obeys these laws by confusing pedestrians for red lights, as both entail the correct action (“stop”). Yet, if – when used 

> _∗_ Equal contribution. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0002-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0002-01.png)


**----- Start of picture text -----**<br>
(a) NeSy ( DPL  &  LTN ) (b)  CBM (c)  NN  &  CLIP<br>**----- End of picture text -----**<br>


Figure 1: **Role of concepts in deep learning models.** (a) NeSy architectures like DeepProbLog ( `DPL` ) and Logic Tensor Networks ( `LTN` ) map the input **x** to concepts **c** and reason over these according to prior knowledge to obtain a label **y** . (b) `CBM` s are similar, except the prediction is computed by a learned linear layer, making it easy to obtain concept-level explanations of all predictions. (c) Black-box neural networks infer a label **y** directly from the input **x** ; concepts **c** can be extracted from their latent representation by applying techniques like TCAV [48]. Lighting bolts indicate what variables are usually supervised. 

in an out-of-distribution (OOD) task – the vehicle is allowed to cross over red lights in case of an emergency, its preexisting confusion can lead to unfortunate scenarios [20]. RSs impact learnability [18], interpretability of the learned concepts [21–24], and reliability in down-stream tasks [20, 25–28]. At the same time, they can affect most NeSy architectures, regardless of how they are implemented, including approaches based on probabilistic logic [12–14, 16, 29–33], fuzzy logic [8, 34], reasoning in embedding space [35], and abduction [36, 37]. Given their impact, researchers have proposed several mitigation strategies [17, 20, 25, 38–41, 109, 107], yet how to deal with RSs remains an open problem. 

Unfortunately, suitable data sets with known RSs are scarce and scattered throughout the literature, hindering research on this challenging problem. Current benchmark suites for learning and reasoning neglect RSs altogether [42] and lack OOD data suitable for investigating their impact, while others are restricted to larger models [43–45]. Simultaneously, available data sets annotated with concept supervision ( _e.g._ , CUB200 [46]), which is essential for evaluating concept quality, do not require logical reasoning and do not supply prior knowledge. 

**Contributions** . We fill this gap by introducing `rsbench` , an integrated benchmark suite providing all the ingredients needed for systematic evaluation of the impact of RSs and the efficacy of mitigation strategies. `rsbench` comprises: 1) A curated collection of _**tasks**_ that require learning and reasoning that are provably affected by RSs. `rsbench` comprises entirely new and already established tasks with different flavors – arithmetical, logical, and high-stakes – along with associated _**data sets**_ and _**data generators**_ for evaluating OOD scenarios.[2] 2) `Python` implementations of _**quality metrics**_ useful for assessing the impact of RSs on NeSy models and more generally the reliability of concepts learned (explicitly or implicitly) by other concept-based architectures and end-to-end neural networks. 3) a novel algorithm, `countrss` , that exploits automated reasoning techniques [47] to _**verify a priori**_ whether a task is affected by RSs and to count them. We showcase `rsbench` by assessing the impact of RSs on the quality of concepts acquired by several deep learning architectures, illustrated in Fig. 1. 

## **2 Reasoning Shortcuts: Causes, Consequences, and Scope** 

We study tasks where models require both learning and reasoning (in short: _**L&R tasks**_ ) to accurately predict a (vector) output **y** from low-level inputs **x** [20]. First, we assume there is a set of _k_ high-level concepts **c** _[∗]_ associated to the inputs **x** . Then, we assume the concepts **c** _[∗]_ and prior knowledge K together infer the correct output **y** _[∗]_ . The prior knowledge can encode known structural [11] or safety constraints [15] in some formal language ( _e.g._ , logical connectives). 

**Example 1.** _The_ `SDD-OIA` _dataset (detailed in Section 3.3) is a L&R task that contains images_ **x** _of 3D traffic scenes, and the goal is to predict one or more actions {_ `stop` _,_ `go` _,_ `left` _,_ `right` _}. We assume the correct output depends on binary concepts c_ `grn` _, c_ `red` _, c_ `ped` _encoding whether green lights, red lights, and pedestrians are visible, respectively. The knowledge specifies that if the latter are detected, the vehicle must stop:_ K = ( _c_ `ped` _∨ c_ `red` _⇒ y_ `stop` ) _._ 

> 2In Appendix C.6, we provide a **how-to** guide to the usage of `rsbench` . 

2 

Table 1: **List of L&R tasks in** `rsbench` . All columns are described in Section 3. 

|TASK|DATA<br>GEN<br>OOD<br>CONL|PROPERTIES<br>CPLX**x**<br>CPLXK<br>AMBK|
|---|---|---|
|ARITHMETIC<br>`MNMath` (new<br>)<br>`MNAdd-Half`[25]<br>`MNAdd-EvenOdd`[17]<br>LOGIC<br>`MNLogic` (new<br>)<br>`Kand-Logic`[25]<br>`CLE4EVR`[17]<br>HIGH<br>STAKES<br>`BDD-OIA`[20]<br>`SDD-OIA` (new<br>)|✓<br>✓<br>✓<br>✗<br>✓✓<br>✗<br>✗<br>✓✓<br>✓✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓✓<br>✗<br>✗<br>✗<br>✓<br>✓✓<br>✓|✗<br>✓<br>✗<br>✗<br>✗<br>–<br>✗<br>✗<br>–<br>✗<br>✓<br>✗<br>✗<br>✓<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|



Reasoning Shortcuts (RSs) have primarily been studied in the context of NeSy models. They usually consist of a _**perception**_ module that predicts concepts **c** from input **x** and a _**reasoning**_ module that uses the prior knowledge K to compute an output, as shown in Fig. 1(a). Like most deep learning architectures, NeSy models are trained via maximum likelihood on annotated examples ( **x** _,_ **y** _[∗]_ ), while concept annotation is typically not available. This makes them susceptible to RSs, _i.e._ , they can learn concepts with improper or unclear semantics. For example, in `BDD-OIA` a model can mistake a `pedestrian` for a `red_light` without affecting prediction quality, as both concepts entail the correct `stop` action; more examples can be found in Section 3. Low-quality concepts compromise performance in down-stream decision task [17, 20, 25] – _e.g._ , those that hinge on `pedestrians` and `red_light` being predicted correctly – and in tasks that depend on externally supplied concepts, such as neuro-symbolic formal verification [26, 27], undermining trustworthiness. Moreover, since the concepts’ meaning is muddled, concept-based explanations [48, 24] cannot be interpreted properly by human stakeholders. 

Among the root causes of RSs are [20] (1) the structure of the _**prior knowledge**_ K; (2) the contents of the _**training set**_ ; (3) the choice of _**loss function**_ ; and (4) if the concept extractor is guided by appropriate _**architectural bias**_ . Mitigation strategies possibly target one or more of these causes. For instance, _multitask learning_ [49] lowers the chance one can achieve high accuracy by confusing concepts, _reconstruction losses_ [50] help to disambiguate between visually distinct concepts, and _disentanglement_ [51] provides a useful architectural bias. Several other strategies have been proposed [25, 38–40]. Existing solutions, however, are no silver bullet [20]: the only general, sure-proof way of avoiding RSs is supervising concepts ( _e.g._ , [52]), which is seldom available and often neglected in learning tasks involving reasoning [25]. By providing easy access to RS-heavy tasks and evaluation protocols, `rsbench` aims to facilitate progress on this challenging open problem. 

**Beyond NeSy models and RSs.** While RSs arise naturally in NeSy models, RSs for purely neural architectures are not well-defined as knowledge is not explicitly encoded in such architectures. Nevertheless, several neural models learn concepts either _explicitly_ or even _implicitly_ , and determining their quality is as important as evaluating RSs in NeSy models, since it provides an indication of potential patterns that the network could end up learning, ( _e.g._ , a convolutional filter could learn to detect both a red traffic light and a pedestrian, without disambiguating between the two). Additionally, RSs corrupt the semantics of concept-based explanations extracted in a post-hoc fashion (for `NN` s) and of model-provided explanations (for `CBM` s). To this end, we design `rsbench` to evaluate also purely neural models, including gray-box models – such as _concept-bottleneck models_ ( `CBM` s) [53] – as well as black-box neural networks ( `NN` s) and neural models involving a pre-processing step given by foundation models, e.g., `CLIP` [54]. `CBM` s natively output concept predictions for their decisions, making it possible to directly evaluate their quality using our metrics, at the cost of requiring a modicum of concept-level supervision during training, as shown in Fig. 1 (b). For black-box networks, which only learn concepts implicitly, `rsbench` extracts concept predictions in a post-hoc fashion using TCAV [48], see Fig. 1 (c). 

3 

Figure 2: This figure illustrates inference and training in regular NeSy architectures for one `BDD-OIA` example [19]. The input **x** is a dashcam image. The model first extracts concepts **c** = ( _c_ `grn` _, c_ `red` _, c_ `ped` ) _∈{_ 0 _,_ 1 _}_[3] from the image using a neural backbone (NN) and then uses a (differentiable) reasoning layer to infer a vector label **y** = ( _y_ `go` _, y_ `stop` _, y_ `left` _, y_ `right` ). While the model includes a neural component, the labels depend solely on the extracted concepts. The reasoning layer is aware of prior knowledge K, which encodes constraints like “if a pedestrian or a red light is detected, the prediction must be `stop` .” 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0004-01.png)


**----- Start of picture text -----**<br>
y go y stop y left y right<br>K = ( c ped ∨  c red ⇒  y stop) ∧ ...<br>c grn c red c ped<br>NeSy Predictor NN<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0004-02.png)


## **3 The** `rsbench` **Benchmark Suite** 

In the following, we outline the L&R tasks and metrics provided by `rsbench` . By construction, RSs do not compromise in-distribution performance, and their worst effects are seen on OOD data. `rsbench` facilitates _constructing novel OOD data sets_ by providing a configurable _**data generator**_ for each of its tasks (except `BDD-OIA` , cf. Section 3.3). These enable fine-grained control over all details of the training, validation and test splits (like number of examples and percentage allocated to each split, in addition to task-specific settings discussed in the relevant subsection) and the creation of OOD splits, all through a simple `YAML` configuration file. All tasks are available as `Python` classes and their knowledge K is supplied in the widely used DIMACS CNF format [55], to support interoperability with model implementations and reasoning packages. In Sections 3.1 to 3.3, for each task, we illustrate a possible reasoning shortcut and its impact on an OOD input. 

Table 1 provides an overview of the `rsbench` L&R tasks, breaking them down into relevant properties, namely whether they: include a _data generator_ (GEN); allow users to create (✓) or provide readymade (✓✓) _out-of-distribution_ splits (OOD); allow users to create (✓) or provide ready-made (✓✓) data suitable for _continual learning_ (CONL) [17]; have complex inputs, making it difficult to extract concepts ( _e.g._ , of different objects) separately (CPLX **x** ); require complex reasoning when using the default knowledge (CPLX K); by default use knowledge that is intrinsically _ambiguous_ , _i.e._ , it yields RSs even if the training set contains all possible combinations of concepts and labels (AMB K). A task involves complex inputs (CPLX **x** ) when it requires processing semi-realistic visual scenes with multiple objects for concept extraction ( _e.g,_ `Kand-Logic, SDD-OIA` ). It involves complex reasoning (CPLX K) when inference requires handling interrelated concepts or multi-step reasoning. For instance, `BDD-OIA` and `SDD-OIA` require inferring 4 actions from 20 interrelated concepts ( _e.g._ , traffic lights of different colors, presence of pedestrians), where some concepts are mutually exclusive ( _e.g._ , traffic lights can’t be green and red simultaneously). 

Fig. 2 illustrates how a NeSy architecture ( `DPL` ) operates on a `rsbench` task ( `BDD-OIA` ). 

## **3.1 Arithmetical Tasks and Data Sets** 

||TASK<br>EXAMPLEDATA<br>KNOWLEDGEK<br>EXAMPLERS<br>IMPACT<br>`MNMath`<br>�2_·_<br>+<br>= 6<br>+<br>= 7<br>Equations must hold.<br><br><br><br>_→_2<br>_→_4<br>_→_3<br>+<br>=5|
|---|---|



`MNAdd` [14] is the quintessential benchmark for evaluating reasoning in NeSy AI [7, 12, 38, 56–60]. The goal is to infer the sum of _k ≥_ 2 MNIST [61] digits, provided knowledge encoding the rule of summation, that is, K := ( _y_ =[�] _[k] j_ =1 _[c][j]_[)][.] _[E.g.]_[, given] **[ x]**[=] the model should predict _y_ = 7. Despite its simplicity, `MNAdd` highlights a clear performance gap between pure neural baselines and NeSy architectures [14, 38]. RSs arise when we can infer the correct sum using the wrong digits. This can occur due to commutativity ( _e.g._ , and both sum to 4) or incomplete training data ( _e.g._ , in absence of other training examples, knowing that sum to 5 is insufficient to discriminate the 

4 

intended sum from 0 + 5, 1 + 4, 4 + 1, and 5 + 0). If the training set covers all sums in _{_ 0 _, . . . ,_ 18 _}_ , `MNAdd` only exhibits RSs of the first kind, which can be avoided by processing the two input digits separately [20]. 

**Task:** We introduce `MNMath` , a novel _multi-label_ extension of `MNAdd` in which the goal is to predict the result of a system of equations of MNIST digits. _E.g._ , given knowledge K = ( _y_ 1 = 2 _c_ 1 + _c_ 2) _∧_ ( _y_ 2 = _c_ 3 + _c_ 4) encoding a system of two equations and an input **x** = , a model trained to predict **y** = (6 _,_ 7) can learn to systematically map to 4 and to 3, resulting in incorrect down-stream decisions, as in the example above. In `MNMath` , the knowledge consists of a system of equations. The input is a single 28 _k ×_ 28 image, obtained by concatenating _k_ `MNIST` images, each representing a handwritten digit in the equations. The concepts are _k_ categorical variables, one for each digit, and the label encodes the result of the equation system. The key feature of `MNMath` is that, besides requiring more complex reasoning, it comes with a data generator tailored for generating OOD splits and challenging learning scenarios. It allows to change the number of equations and digits in each equation, and to define additional operations. 

**Task:** To facilitate comparison with existing mitigation algorithms, `rsbench` supplies also `MNAdd-Half` and `MNAdd-EvenOdd` , two variants of `MNAdd` with guaranteed RSs that have been used in the literature [25, 17]. Both restrict the digits available for training to a subset of combinations – `MNAdd-Half` focuses on certain combinations of digits in _{_ 0 _, . . . ,_ 4 _}_ , while `MNAdd-EvenOdd` contains either only even or only odd digits – guaranteeing the model cannot avoid RSs even when processing inputs separately. In contrast with `MNAdd-Half` , however, it naturally lends itself to OOD evaluation, with the even digits in one domain and the odd digits the second one, and to multitask and continual settings. For these datasets, the knowledge is based on the sum operation. The input is a single 56 _×_ 28 image, created by concatenating 2 `MNIST` images, each representing an operand in the sum. The concepts consist of 2 categorical variables, one for each digit, and the label represents the sum of these digits. 

## **3.2 Logical Tasks and Data Sets** 

||TASK<br>TRAININGSET<br>KNOWLEDGEK<br>EXAMPLERS<br>IMPACT<br>`MNLogic`<br>_⊕_<br>_⊕_<br>_⊕_<br>= 0<br>Formula must hold.<br>�<br>_→_1<br>_→_0<br>_∧_<br>=1<br>`Kand-Logic`<br>= 1<br>Pattern must hold.<br><br><br><br>□_→_`red`<br>_△→_`yel`<br>_⃝→_`blu`<br>=0<br>`CLE4EVR`<br>= 0,<br>= 1<br>Same color and shape?<br><br><br><br>■_→_■<br>■_→_■<br>_⃝→_■<br>=1|
|---|---|



**Task:** `MNLogic` **.** RSs arise whenever the knowledge K allows deducing the right label from multiple configurations of concepts. This form of non-injectivity is a standard feature of most logic formulas, and in fact formulas as simple as the XOR are riddled by RSs [20]. _I.e._ , if K = ( _c_ 1 _⊕ c_ 2 _⇔ y_ ), where _⊕_ denotes the XOR operator, a model that maps to 1 and to 0 classifies all inputs perfectly. To probe the pervasiveness of RSs, we introduce `MNLogic` , a logical analogue to `MNAdd` [14] in which inference is driven by a _random logic formula φ_ . Specifically, an input **x** is the concatenation of _k ≥_ 2 images of zeros and ones sampled from MNIST [61] representing the truth value of _k_ bits **c** , and its ground-truth label _y_ encodes whether it satisfies _φ_ or not, _i.e._ , K = ( _φ ⇔ y_ ). The formula _φ_ is a random _ℓ_ -CNF, _i.e._ , a conjunction of _m_ clauses (disjunctions), each of which contains _ℓ_ out of the _k_ bits and their negations. For instance, _x_ = encodes the bits **c** = (0 _,_ 1) and if _φ_ = ( _c_ 1 _∨ c_ 2) _∧_ ( _c_ 1 _∨¬c_ 2) it is labeled as _y_ = 0. In `MNLogic` , the knowledge consists of a logical formula _φ_ . The input is a single 28 _k ×_ 28 image, created by concatenating _k_ `MNIST` images, each representing either a true atom ( ) or a false atom ( ) in the formula. The concepts are _k_ categorical variables, one for each atom, and the label encodes the outcome of the logical formula. The `MNLogic` generator allows users to specify the number of digits _k_ , the number of clauses _m_ and their length _ℓ_ , and to supply a custom formula, and takes care of constructing all training, validation, and test splits. It also avoids trivial data by ensuring each clauses is neither a tautology nor a contradiction. 

5 

**Task:** `Kand-Logic` [25] is a task – inspired by Wassily Kandinsky’s paintings and [62] – that requires simple (but non-trivial) perceptual processing and relatively complex reasoning. In the simplest case, each input _x_ = ( _x_ 1 _, x_ 2) consists of two 64 _×_ 64 images, each depicting three geometric primitives with different shapes (□, _△_ , _[⃝]_ ) and colors ( `red` , `blue` , `yellow` ). The goal is to predict whether _x_ 1 and _x_ 2 fit the same predefined _logical pattern_ or not. Let each _xi_ contain three primitives _xi,_ 1, _xi,_ 2, _xi,_ 3 with two concepts each: shape `sha` ( _xij_ ) and color `col` ( _xij_ ). The pattern is built out of predicates like “all primitives in the image have a different color”, “all primitives have the same color”, and “exactly two primitives have the same shape”, formally: 

`diffcol` ( _xi_ ) =[�] _j_ = _j[′]_[(] `[col]`[(] _[x][ij]_[)] _[ ̸]_[=] `[ col]`[(] _[x][ij][′]_[))] _[,]_ `samecol` ( _xi_ ) =[�] _j_ = _j[′]_[(] `[col]`[(] _[x][ij]_[) =] `[ col]`[(] _[x][ij][′]_[))] _[,]_ 

and `twosha` ( _xi_ ) = _¬_ `samesha` ( _xi_ ) _∧¬_ `diffsha` ( _xi_ ). For instance, the default pattern [25] `patt` ( _xi_ ) is “all images include either the same number of primitives with the same color, or the same number of primitives with the same shape”, or equivalently: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0006-03.png)


An input _x_ is positive if and only if it satisfies the pattern, _i.e._ , K = ( _Y ⇔_ `patt` ( _x_ 1 _, x_ 2)). Unlike `MNLogic` , in `Kand-Logic` each primitive has multiple attributes that cannot easily be processed separately. This means that RSs can easily, _e.g._ , confuse shape with color when either is sufficient to entail the right prediction, as in the example above. `rsbench` provides the data set used in [25] (3 images per input with 3 primitives each) and a generator that allows configuring the number of images and primitives per input and the pattern itself. 

**Task:** `CLE4EVR` [17] focuses on logical reasoning over three-dimensional scenes, inspired by `CLEVR` [63] and `CLEVR-HANS` [64]. Among these, `CLEVR` is tailored for visual-question answering and `CLEVR-HANS` to contain confounding factors at the input level, to make _shortcuts_ arise [65]. Both of them typically provide models with exhaustive concept-level supervision, obscuring whether RSs are present without it. Our `CLE4EVR` constitutes a simplified version where RSs can be easily determined. Each input image _x_ , of size 240 _×_ 320, contains a variable number of objects differing in size (3 possible values), shape (10), color (10), material (2), position (real), and rotation (real), and the goal is to determine whether the objects satisfy a pre-specified condition _φ_ that depends on all discrete attributes of the objects in the scene. The default knowledge K is designed to induce RSs [17]: it asserts that an image _x_ is positive iff at least two objects _xi_ and _xj_ have the same color and shape, _i.e._ , _∃i ̸_ = _j ._ ( `sha` ( _xi_ ) = `sha` ( _xj_ )) _∧_ ( `col` ( _xi_ ) = `col` ( _xj_ )). When all possible colors and shapes are observed, the only RSs `CLE4EVR` is affected by are those in which the attributes of first object are attributed to the second one. However, if the training set includes only _some_ combinations – _e.g._ , pink rings and gray spheres are never observed together – the model can collapse different shapes and colors [17]. Hence, even when objects are processed separately ( _e.g._ , via Faster-RCNN embeddings), the model can confuse colors and shapes with one another, _e.g._ , it can mistake blue cones for red pyramids and vice versa. Occlusion further complicates the picture, complicating both perception and reasoning. As above, the generator allows to customize the number of objects per image, the knowledge, and whether occlusion is allowed. 

## **3.3 High-stakes Tasks and Data Sets** 

||TASK<br>TRAINING SET<br>KNOWLEDGEK<br>EXAMPLERS<br>IMPACT<br>`SDD-OIA`<br>= stop<br>Traffc laws.<br><br><br><br><br><br><br><br>`pedestrian`_→_<br>`red_light`<br>`green_light`_→_<br>`green_light`<br>=go<br>`BDD-OIA`<br>= stop<br>Traffc laws.<br><br><br><br><br><br><br><br>`pedestrian`_→_<br>`red_light`<br>`green_light`_→_<br>`green_light`<br>=go|
|---|---|



**Task:** `BDD-OIA` [19] is a multi-label autonomous driving task for studying RSs in real-world, _highstakes_ scenarios. The goal is to infer what actions out of _{_ `forward` _,_ `stop` _,_ `left` _,_ `right` _}_ are safe 

6 

depending on what objects ( _e.g._ , cars, traffic signs) are present in an input dashcam image. The knowledge K establishes that, _e.g._ , it is not safe to move `forward` if there are pedestrians on the road, based on a set of 21 binary concepts indicating the presence of different obstacles on the road. The constraints specify conditions for being able to proceed ( `green` _ `light` _∨_ `follow` _∨_ `clear` _⇒_ `forward` ), stop ( `red` _ `light` _∨_ `stop` _ `sign` _∨_ `obstacle` _⇒_ `stop` ), and for turning left and right, as well as relationships between actions ( _e.g._ , `stop` _⇒¬_ `forward` ). Input images, of size 720 _×_ 1280, come with concept-level annotations, making it possible to assess the quality of the learned concepts. The dataset comprises 16 _,_ 082 training examples, 2 _,_ 270 validation examples and 4 _,_ 572 test examples. Common RSs allow to, _e.g._ , confuse `pedestrians` with `red_light` s, as they both imply the correct ( `stop` ) action for all training examples [20]. 

**Task:** `SDD-OIA` . `BDD-OIA` is not suitable for systematically evaluating RSs out-of-distribution, where they show the highest impact. With `rsbench` , we fill this gap by introducing `SDD-OIA` , a synthetic replacement for `BDD-OIA` that comes with a fully configurable _data generator_ , enabling fine-grained control over what labels, concepts, and images are observed and the creation of OOD splits. In short, `SDD-OIA` shares the same classes, concepts and (by default) knowledge as `BDD-OIA` , but the images are 3D traffic scenes modelled and rendered using Blender [66] as 469 _×_ 387 RGB images. Images are generated by first sampling a desired label **y** , then picking concepts **c** that yield that label, and then rendering an image **x** displaying those concepts. This allows to easily control what concepts and labels should appear in all data splits, which in turn determine what kinds of RSs can be learned. The complete data generation process is described in Appendix C.11. 

In Section 4, we showcase `SDD-OIA` by implementing an OOD autonomous ambulance scenario [20] in which the vehicle is allowed to cross red lights in case of an `emergency` . Formally, this requires altering the prior knowledge by introducing a new `emergency` variable that conditions the traffic rules, that is, ( _¬_ `emergency` = _⇒_ original rule for `stop` ) _∧_ ( _¬_ `emergency` = _⇒_ alternative rule for `stop` ), and similarly for `turn` _ `left` and `turn` _ `right` . We specifically test this scenario in Section 4. Naturally, other challenging OOD scenarios can be created. 

## **3.4 Metrics for Reasoning Shortcuts** 

**Model-level metrics.** `rsbench` facilitates assessing learned models by implementing several metrics for label and concept predictions – including accuracy and _F_ 1 score – as well as metrics for RSs. First, `rsbench` provides concept-level confusion matrices, which show how well the predicted concepts **c** = ( _c_ 1 _, . . . , ck_ ) recover the annotations **c** _[∗]_ = ( _c[∗]_ 1 _[, . . . , c][∗] k_[)][ and are essential for visualizing] and spotting RSs, as can be seen in Table 3. Second, it implements _**concept collapse**_ Cls( _C_ ), which measures to what extent the learned concepts mix distinct ground-truth concepts. Given a concept confusion matrix _C ∈_ [0 _,_ 1] _[m][×][m]_ , where _m_ is the size of the confusion matrix ( _e.g._ , _m_ = 2 _[k]_ when all ground-truth concepts are observed), it is defined as Cls( _C_ ) = 1 _− p/m_ , where _p_ =[�] _[m] j_ =1[1] _[{∃][i . C][ij][>]_[ 0] _[}]_[.][High collapse shows that the model tends to use fewer concepts to solve] the task, making it useful for diagnostics. Vice versa, a lower concept collapse may indicate that the RS is densely activating all concepts. Concept collapse is not trivial to implement because not all 2 _[k]_ ground-truth concept combinations may appear in the test set (especially when _k_ is large, see Appendix A for details), so `rsbench` provides ready-made implementations for all its tasks. 

**Task-level metrics.** RSs arise due to a complex interaction between prior knowledge and training data (cf. Section 2) making it difficult to assess _a priori_ which L&R tasks they affect. Fortunately, it is possible to count how many optimal (deterministic) RSs affect a L&R task [20, 25], as long as this satisfies two technical assumptions.[3] They also provide a closed-form expression for the count that works only when the training set is _exhaustive_ (that is, comprises all possible combinations of concepts, like `MNAdd` ) and the concepts are extracted jointly. This makes it possible to _**formally verify**_ whether a L&R task can be solved via RSs by checking that the count is larger than 1. This is crucial for anticipating the occurrence of RSs in novel tasks and for iteratively improving task specifications in the design stage. In practice, however, the training set is seldom exhaustive and concepts are often processed separately. While the former issue can be overcome – and in fact, we provide a closed-form solution in Appendix A.3 – the latter is more challenging. 

> 3Specifically, _**invertibility**_ ( **A1** ) and _**determinism**_ ( **A2** ), meaning that it is always possible to recover the unique ground-truth concept vector underlying any input and that the knowledge K maps any concept configuration to a unique label, respectively; see [20] for an in-depth discussion. 

7 

Table 2: Results on `MNAdd-EvenOdd` 

||F1(_Y_)(_↑_)|F1(_C_)(_↑_)|Cls(_C_)(_↓_)|
|---|---|---|---|
|`DPL`|0_._94_±_0_._04|0_._06_±_0_._08|0_._61_±_0_._08|
|`LTN`|0_._66_±_0_._10|0_._05_±_0_._06|0_._70_±_0_._01|
|`CBM`_†_<br>`NN`_∗_|0_._89_±_0_._13<br>0_._57_±_0_._38|0_._44_±_0_._07<br>0_._07_±_0_._03|0_._09_±_0_._09<br>0_._29_±_0_._34|
|`CLIP`_∗_|0_._62_±_0_._14|0_._04_±_0_._01|0_._48_±_0_._17|



Table 4: Results on `Kand-Logic` 

Table 3: (Left) `DPL` and (Right) `NN` concept confusion matrices for `MNAdd-EvenOdd` . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0008-04.png)


**----- Start of picture text -----**<br>
0.35<br>0.8 0.30<br>0.25<br>0.6<br>0.20<br>0.4 0.15<br>0.10<br>0.2<br>0.05<br>0.0 0.00<br>0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9<br>0 0<br>1 1<br>2 2<br>3 3<br>4 4<br>5 5<br>6 6<br>7 7<br>8 8<br>9 9<br>**----- End of picture text -----**<br>


Table 5: Performance on `SDD-OIA` 

|F1(_Y_)(_↑_)<br>F1(_C_)(_↑_)<br>Cls(_C_)(_↓_)<br>`DPL`<br>0_._87_±_0_._15<br>0_._25_±_0_._09<br>0_._69_±_0_._04<br>`LTN`<br>0_._77_±_0_._09<br>0_._35_±_0_._04<br>0_._00_±_0_._01<br>`CBM`_†_<br>0_._36_±_0_._04<br>0_._59_±_0_._01<br>0_._00_±_0_._01<br>`NN`_∗_<br>0_._72_±_0_._08<br>0_._33_±_0_._01<br>0_._17_±_0_._01<br>`CLIP`_∗_<br>0_._99_±_0_._01<br>0_._32_±_0_._01<br>0_._00_±_0_._01|mF1(_Y_)(_↑_)<br>mF1(_C_)(_↑_)<br>mCls(_C_)(_↓_)<br>OOD-mF1(_Y_)(_↑_)|
|---|---|
||`DPL`<br>0_._80_±_0_._01<br>0_._49_±_0_._03<br>0_._86_±_0_._04<br>0_._62_±_0_._09|
||`LTN`<br>0_._82_±_0_._04<br>0_._46_±_0_._04<br>0_._81_±_0_._02<br>0_._72_±_0_._06<br>|
||`CBM`_†_<br>0_._60_±_0_._12<br>0_._61_±_0_._04<br>0_._68_±_0_._06<br>0_._45_±_0_._05|
||`NN`_∗_<br>0_._93_±_0_._18<br>0_._44_±_0_._02<br>0_._43_±_0_._28<br>0_._47_±_0_._19<br>|
||`CLIP`_∗_<br>0_._90_±_0_._09<br>0_._43_±_0_._04<br>0_._23_±_0_._02<br>0_._81_±_0_._06|



`rsbench` addresses it by implementing a practical counting algorithm, named `countrss` , that leverages _automated reasoning_ [47] techniques. In a nutshell, each optimal RS can be viewed as a linear mapping **c** = _A_ **c** _[∗]_ that maps ground-truth to predicted concepts under the _constraint_ that these yield the correct label **y** _[∗]_ for all training examples. The problem thus boils down to counting how many matrices _A ∈{_ 0 _,_ 1 _}[k][×][k]_ , where _k_ is the number of concepts,[4] satisfy this constraint. Given the exponential (in _k_ ) number of candidates, `countrss` relies on state-of-the-art _model counting_ solvers [47, 67] for efficiency. That is, we encode the above _constraint_ as a propositional logic formula (see Appendix A for the exact encoding), such that each model (solution) represents a distinct RS. `countrss` , based on `PyEda` [68], works for all L&R tasks that satisfy the necessary technical assumptions, including all those in `rsbench` except `BDD-OIA` , and supports both exact [69] and, for the more complex tasks, approximate counting [70]. 

We showcase `countrss` by evaluating the impact of the amount of training examples on the RS count for two instances of `MNLogic` : AND (with K = ( _y ⇔ c_ 1 _∧c_ 2 _∧c_ 2)) and XOR (K = ( _y ⇔ c_ 1 _⊕c_ 2 _⊕c_ 3)). When the training set is exhaustive, AND admits 6 RSs and XOR 24, proving that symmetries in the XOR function make it latter more ambiguous in this case, as stated in Section 3.2. The number of RSs grows drastically when we only provide a single example, as it becomes easier to predict all labels correctly while confusing concepts, with the XOR presenting 192 RSs. For the AND, the count depends on whether the single ground truth label is positive or negative, the number of RSs growing to 48 and 336, respectively. This highlights how even simple formulas can be affected by RSs and that these depend crucially on the available data, as expected, and that `countrss` can anticipate the occurrence of RSs in L&R tasks without the need for training any model. 

## **4 Evaluating RSs and Concept Quality with** `rsbench` 

`rsbench` is meant to be a general framework for evaluating the impact of RSs and concept quality in _any_ machine learning model. We showcase this by evaluating _five_ different architectures on three L&R tasks, one per “flavor”, namely `MNAdd-EvenOdd` , `Kand-Logic` , and `SDD-OIA` . 

We consider two state-of-the-art NeSy models: DeepProbLog ( `DPL` ) [14] and Logic Tensor Networks ( `LTN` ) [34]. Both comprise a neural network module to extract concepts **c** for every input **x** , which are later used to predict labels **y** according to the knowledge K (Fig. 1(a)). These predictions are done according to a probabilistic logic semantic in `DPL` and by using fuzzy logic in `LTN` . As stated in Section 2, we also experiment with purely neural models, evaluating the quality of the concepts they learn on `rsbench` . Specifically, we employ `CBM` s (Fig. 1(b)), black-box `NN` s and `CLIP` (Fig. 1(c)). In our analysis, we investigate directly the bottleneck layer for `CBM` , where concepts are expected to be learned, and adopt TCAV for `NN` and `CLIP` . 

> 4 More precisely, _k_ is the number of bits required for the one-hot encoding of the concept vector **c** . 

8 

We evaluate macro _F_ 1 for predicted labels and concepts, denoted as F1( _Y_ ) and F1( _C_ ), respectively, and concept collapse Cls( _C_ ) (see Appendix A for the exact definition). We report the mean and standard deviation over 10 random seeds. We train all models by maximum likelihood on the labels, as customary in NeSy and for neural baselines. Notice that, `CBM` without annotated concepts would be equivalent to `NN` , therefore we supervise a handful of concepts, as customary [53, 22, 23, 71]. Specifically, we supervise a total of 100 examples for `MNAdd-EvenOdd` (only for the digits 3, 4, 8 and 9); 20 examples for `Kand-Logic` (for `red` , □, and _[⃝]_ ); and _∼_ 700 examples for `SDD-OIA` on the highstakes concepts `red_light` , `green_light` , `car` , `person` , `rider` , `other_obstacle` , `stop_sign` , `right_green_light` , and `left_green_light` . All details about the losses, architectures, metrics, and model selection procedure we use are reported in Appendix B. 

**All tasks succeed in inducing RSs across all models** . The results in Tables 2, 4 and 5 show that all models attain medium-to-high F1( _Y_ ) on the three benchmarks, meaning the labels _can_ be predicted accurately, with the following exceptions: On `MNAdd-EvenOdd LTN` , `NN` , and `CLIP` show medium-to-high variance (10%, 38%, and 14%, respectively); On `Kand-Logic` , `LTN` , `CBM` , and `NN` reach suboptimal performance on (around 77%, 36%, and 72%, respectively); On `SDD-OIA` , `CBM` scores only 60% F1( _Y_ ). We attribute the subpar F1( _Y_ ) score of `CBM` to the fact that their top linear layer is not expressive enough to accurately infer the label from the concept bottleneck in more complex tasks like `Kand-Logic` and `SDD-OIA` . Despite these variations in prediction performance, _all models show overall low concept quality_ , as measured by F1( _C_ ). Even `CBM` , despite receiving concept supervision, fare below 60%. 

**Understanding concept quality with** `rsbench` **.** The high rate of Cls( _C_ ) in all tasks (except for `LTN` in `Kand-Logic` ) suggests that NeSy models tend mix concepts together [40]. The left-most confusion matrix in Table 3 shows that in `MNAdd-EvenOdd` , `DPL` uses roughly half of the available digits to solve the task with high F1( _Y_ ). In contrast, `CBM` experiences less collapse due to concept supervision. `NN` and `CLIP` also yield overall low collapse (an exception being `CLIP` in `MNAdd-EvenOdd` ), and in fact the right-most confusion matrix in Table 3 shows that `NN` in `MNAdd-EvenOdd` activates densely most concepts. We point out that, however, lower collapse for `NN` and `CLIP` stems also from TCAV, which can introduce noise in the extracted concepts. Due to space constraints, we report a detailed analysis of this phenomenon in the supplementary material. 

**Generators enable measuring the OOD impact of RSs.** For `SDD-OIA` we leverage the generator to evaluate an OOD setting where the same concepts are used with a different knowledge KOOD (reported in Section 3.3). _We observe that all models suffer a visible drop in OOD_ mF1( _Y_ ) _performance_ , as expected: `DPL` drops by 18%, `LTN` by 10%, and `CBM` by _∼_ 15%. `NN` is the most affected, with average 46% difference, while `CLIP` is the most resilient with only 9% drop. 

## **5 Discussion and Conclusion** 

We introduced `rsbench` , an integrated benchmark suite for systematic evaluation of RSs and concept quality in tasks requiring learning and reasoning. While existing benchmark suites [42] neglect RSs altogether, `rsbench` supplies datasets for various RS-heavy tasks and corresponding ready-made data generators for evaluating OOD and continual learning scenarios. At the same time, `rsbench` provides formal verification and evaluation routines for assessing how much RSs and concept quality affect each task. Our experiments showcase how `rsbench` enables practitioners to easily investigate the impact of RSs on several existing or future deep learning architectures. 

RSs are also connected to the more general problem of learning high-level concepts from data, _aka_ symbol grounding [72, 108]. Interpretable concepts play an increasingly central role as a _lingua franca_ in explainable AI (XAI) [21, 73] for both _post-hoc_ [48, 74–76] and _ante-hoc_ [53, 71, 22, 77, 78, 24] explanations of model decisions. As with RSs, a central question is whether the concepts encode the intended semantics [79, 80]. `rsbench` can be used to benchmark precisely whether learned concepts satisfy this condition. Furthermore, it can also benefit new research in mechanistic interpretability [81– 84, 110], specifically for studying challenging scenarios in which deriving a high-level explanation of neural networks behavior is complicated by poor concept semantics. 

Another related topic is _identifiability_ of latent concepts, which is studied in independent component analysis [85] and causal representation learning [51, 86, 87]. `rsbench` can be readily used to empirically assess identification of latent concepts with only label supervision [88–90]. 

9 

**Broader Impact.** Concepts confused by RSs can lead to poor down-stream decision making. With `rsbench` , we hope to enable research on mitigation strategies for RSs to avoid such consequences. At the same time, `rsbench` might be used to design adversarial attack that exploit or promote RSs and therefore undermine the trustworthyness of ML systems. 

**Limitations and Future Work.** While `rsbench` already provides a variety tasks offering different learning and reasoning challenges, we plan to extend it to include variants of other popular reasoning datasets such as Visual Sudoku [91], Raven matrices [92], KANDY [93], CLEVR-HANS [64], and ROAD-R [94] which in their current status do not allow a systematic study of RSs. Implementations of NeSy architectures make use of distinct formalisms and file formats, making it especially challenging to ensure data interoperability. `rsbench` partially addresses this by supplying both Python APIs and CNF specifications – the standard file format for logic formulas in formal verification – for all L&R tasks. In the future, we will build on initiatives like ULLER [95], which promise to provide a unified interface for NeSy architectures. We also plan to improve the scalability of the formal verification algorithm to larger L&R tasks and to leverage as a guide for active learning-based mitigation strategies. 

## **Acknowledgements** 

The authors are grateful to Alice Plebe for her valuable guidance on Blender, as well as to the authors of the assets used in `SDD-OIA` , see Appendix C.11.1 for the full list. Funded by the European Union. The views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union, the European Health and Digital Executive Agency (HaDEA) or the European Research Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. Grant Agreement no. 101120763 - TANGO. PM is supported by the MSCA project GA n°101110960 “Probabilistic Formal Verification for Provably Trustworthy AI - PFV-4-PTAI”. AV is supported by the “UNREAL: Unified Reasoning Layer for Trustworthy ML” project (EP/Y023838/1) selected by the ERC and funded by UKRI EPSRC. Emile van Krieken was funded by ELIAI (The Edinburgh Laboratory for Integrated Artificial Intelligence), EPSRC (grant no. EP/W002876/1). 

10 

## **References** 

- [1] Luc De Raedt and Angelika Kimmig. Probabilistic (logic) programming concepts. _Machine Learning_ , 2015. 

- [2] Luc De Raedt, Sebastijan Dumanˇci´c, Robin Manhaeve, and Giuseppe Marra. From statistical relational to neural-symbolic artificial intelligence. In _Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence_ , pages 4943–4950, 2021. 

- [3] Artur d’Avila Garcez, Sebastian Bader, Howard Bowman, Luis C Lamb, Leo de Penning, BV Illuminoo, Hoifung Poon, and COPPE Gerson Zaverucha. Neural-symbolic learning and reasoning: A survey and interpretation. _Neuro-Symbolic Artificial Intelligence: The State of the Art_ , 342:1, 2022. 

- [4] Tirtharaj Dash, Sharad Chitlangia, Aditya Ahuja, and Ashwin Srinivasan. A review of some techniques for inclusion of domain-knowledge into deep neural networks. _Scientific Reports_ , 12(1):1–15, 2022. 

- [5] Eleonora Giunchiglia, Mihaela Catalina Stoian, and Thomas Lukasiewicz. Deep learning with logical constraints. _arXiv preprint arXiv:2205.00523_ , 2022. 

- [6] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 11888–11898, 2023. 

- [7] Daniel Cunnington, Mark Law, Jorge Lobo, and Alessandra Russo. The role of foundation models in neuro-symbolic learning and reasoning. _arXiv preprint arXiv:2402.01889_ , 2024. 

- [8] Michelangelo Diligenti, Marco Gori, and Claudio Sacca. Semantic-based regularization for learning and inference. _Artificial Intelligence_ , 2017. 

- [9] Jingyi Xu, Zilu Zhang, Tal Friedman, Yitao Liang, and Guy Broeck. A semantic loss function for deep learning with symbolic knowledge. In _ICML_ , 2018. 

- [10] Yaqi Xie, Ziwei Xu, Mohan S Kankanhalli, Kuldeep S Meel, and Harold Soh. Embedding symbolic knowledge into deep networks. _Advances in neural information processing systems_ , 32, 2019. 

- [11] Luca Di Liello, Pierfrancesco Ardino, Jacopo Gobbi, Paolo Morettin, Stefano Teso, and Andrea Passerini. Efficient generation of structured objects with constrained adversarial networks. _Advances in neural information processing systems_ , 33:14663–14674, 2020. 

- [12] Emile van Krieken, Thiviyan Thanapalasingam, Jakub M Tomczak, Frank van Harmelen, and Annette ten Teije. A-nesi: A scalable approximate method for probabilistic neurosymbolic inference. _arXiv preprint arXiv:2212.12393_ , 2022. 

- [13] Connor Pryor, Charles Dickens, Eriq Augustine, Alon Albalak, William Wang, and Lise Getoor. Neupsl: Neural probabilistic soft logic. _arXiv preprint arXiv:2205.14268_ , 2022. 

- [14] Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. DeepProbLog: Neural Probabilistic Logic Programming. _NeurIPS_ , 2018. 

- [15] Nick Hoernle, Rafael Michael Karampatsis, Vaishak Belle, and Kobi Gal. Multiplexnet: Towards fully satisfied logical constraints in neural networks. In _AAAI_ , 2022. 

- [16] Kareem Ahmed, Stefano Teso, Kai-Wei Chang, Guy Van den Broeck, and Antonio Vergari. Semantic Probabilistic Layers for Neuro-Symbolic Learning. In _NeurIPS_ , 2022. 

- [17] Emanuele Marconato, Gianpaolo Bontempo, Elisa Ficarra, Simone Calderara, Andrea Passerini, and Stefano Teso. Neuro symbolic continual learning: Knowledge, reasoning shortcuts and concept rehearsal. In _ICML_ , 2023. 

- [18] Kaifu Wang, Efi Tsamoura, and Dan Roth. On learning latent models with multi-instance weak supervision. In _NeurIPS_ , 2023. 

11 

- [19] Yiran Xu, Xiaoyin Yang, Lihang Gong, Hsuan-Chu Lin, Tz-Ying Wu, Yunsheng Li, and Nuno Vasconcelos. Explainable object-induced action decision for autonomous vehicles. In _CVPR_ , pages 9523–9532, 2020. 

- [20] Emanuele Marconato, Stefano Teso, Antonio Vergari, and Andrea Passerini. Not all neurosymbolic concepts are created equal: Analysis and mitigation of reasoning shortcuts. In _NeurIPS_ , 2023. 

- [21] Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. _Nature Machine Intelligence_ , 1(5):206–215, 2019. 

- [22] Zhi Chen, Yijie Bei, and Cynthia Rudin. Concept whitening for interpretable image recognition. _Nature Machine Intelligence_ , 2020. 

- [23] Emanuele Marconato, Andrea Passerini, and Stefano Teso. Glancenets: Interpretabile, leakproof concept-based models. _NeurIPS_ , 2022. 

- [24] Eleonora Poeta, Gabriele Ciravegna, Eliana Pastor, Tania Cerquitelli, and Elena Baralis. Concept-based explainable artificial intelligence: A survey. _arXiv preprint arXiv:2312.12936_ , 2023. 

- [25] Emanuele Marconato, Samuele Bortolotti, Emile van Krieken, Antonio Vergari, Andrea Passerini, and Stefano Teso. BEARS Make Neuro-Symbolic Models Aware of their Reasoning Shortcuts. _arXiv preprint arXiv:2402.12240_ , 2024. 

- [26] Xuan Xie, Kristian Kersting, and Daniel Neider. Neuro-symbolic verification of deep neural networks. In _IJCAI_ , 2022. 

- [27] Faried Abu Zaid, Dennis Diekmann, and Daniel Neider. Distribution-aware neuro-symbolic verification. _AISoLA_ , pages 445–447, 2023. 

- [28] Andrea Bontempelli, Stefano Teso, Fausto Giunchiglia, and Andrea Passerini. Concept-level debugging of part-prototype networks. In _International Conference on Learning Representations_ , 2023. 

- [29] Marco Lippi and Paolo Frasconi. Prediction of protein _β_ -residue contacts by markov logic networks with grounding-specific weights. _Bioinformatics_ , 2009. 

- [30] Zhun Yang, Adam Ishay, and Joohyung Lee. NeurASP: Embracing neural networks into answer set programming. In _IJCAI_ , 2020. 

- [31] Jiani Huang, Ziyang Li, Binghong Chen, Karan Samel, Mayur Naik, Le Song, and Xujie Si. Scallop: From probabilistic deductive databases to scalable differentiable reasoning. _NeurIPS_ , 2021. 

- [32] Giuseppe Marra and Ondˇrej Kuželka. Neural markov logic networks. In _Uncertainty in Artificial Intelligence_ , 2021. 

- [33] Arseny Skryagin, Wolfgang Stammer, Daniel Ochs, Devendra Singh Dhami, and Kristian Kersting. Neural-probabilistic answer set programming. In _Proceedings of the International Conference on Principles of Knowledge Representation and Reasoning_ , volume 19, pages 463–473, 2022. 

- [34] Ivan Donadello, Luciano Serafini, and Artur D’Avila Garcez. Logic tensor networks for semantic image interpretation. In _IJCAI_ , 2017. 

- [35] Tim Rocktäschel and Sebastian Riedel. Learning knowledge base inference with neural theorem provers. In _Proceedings of the 5th workshop on automated knowledge base construction_ , pages 45–50, 2016. 

- [36] Wang-Zhou Dai and Stephen H Muggleton. Abductive knowledge induction from raw data. _arXiv preprint arXiv:2010.03514_ , 2020. 

- [37] Zhi-Hua Zhou. Abductive learning: towards bridging machine learning and logical reasoning. _Science China. Information Sciences_ , 62(7):76101, 2019. 

12 

- [38] Robin Manhaeve, Sebastijan Dumanˇci´c, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. Neural probabilistic logic programming in deepproblog. _Artificial Intelligence_ , 298:103504, 2021. 

- [39] Zenan Li, Zehua Liu, Yuan Yao, Jingwei Xu, Taolue Chen, Xiaoxing Ma, L Jian, et al. Learning with logical constraints but without shortcut satisfaction. In _ICLR_ , 2023. 

- [40] Emanuele Sansone and Robin Manhaeve. Learning symbolic representations through joint generative and discriminative training. In _International Joint Conference on Artificial Intelligence 2023 Workshop on Knowledge-Based Compositional Generalization_ , 2023. 

- [41] Hao-Yuan He, Hui Sun, Zheng Xie, and Ming Li. Ambiguity-aware abductive learning. In _Proceedings of the 41st International Conference on Machine Learning_ , Vienna, Austria, 2024. 

- [42] Arne Vermeulen, Robin Manhaeve, and Giuseppe Marra. An experimental overview of neural-symbolic systems. In _International Conference on Inductive Logic Programming_ , pages 124–138. Springer, 2023. 

- [43] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _arXiv preprint arXiv:2309.12284_ , 2023. 

- [44] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. _arXiv preprint arXiv:2309.05653_ , 2023. 

- [45] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ , 2021. 

- [46] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011. 

- [47] Armin Biere, Marijn Heule, and Hans van Maaren. _Handbook of satisfiability_ , volume 185. IOS press, 2009. 

- [48] Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In _International conference on machine learning_ , pages 2668–2677. PMLR, 2018. 

- [49] Rich Caruana. Multitask learning. _Machine learning_ , 28:41–75, 1997. 

- [50] Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational autoencoders and nonlinear ica: A unifying framework. In _International Conference on Artificial Intelligence and Statistics_ , pages 2207–2217, 2020. 

- [51] Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. _Proceedings of the IEEE_ , 2021. 

- [52] Jaron Maene and Luc De Raedt. Soft-unification in deep probabilistic logic. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [53] Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In _International Conference on Machine Learning_ , pages 5338–5348. PMLR, 2020. 

- [54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 

- [55] Holger H Hoos and Thomas Stützle. Satlib: An online resource for research on sat. _Sat_ , 2000:283–292, 2000. 

13 

- [56] Jaron Maene and Luc De Raedt. Soft-unification in deep probabilistic logic. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 

- [57] Samy Badreddine, Artur d’Avila Garcez, Luciano Serafini, and Michael Spranger. Logic tensor networks. _Artificial Intelligence_ , 303:103649, 2022. 

- [58] Alessandro Daniele, Tommaso Campari, Sagar Malhotra, and Luciano Serafini. Deep symbolic learning: Discovering symbols and rules from perceptions. _arXiv preprint arXiv:2208.11561_ , 2022. 

- [59] Eleonora Misino, Giuseppe Marra, and Emanuele Sansone. VAEL: Bridging Variational Autoencoders and Probabilistic Logic Programming. _NeurIPS_ , 2022. 

- [60] Lennert De Smet, Pedro Zuidberg Dos Martires, Robin Manhaeve, Giuseppe Marra, Angelika Kimmig, and Luc De Readt. Neural probabilistic logic programming in discrete-continuous domains. In _Uncertainty in Artificial Intelligence_ , pages 529–538. PMLR, 2023. 

- [61] Yann LeCun. The mnist database of handwritten digits. _http://yann. lecun. com/exdb/mnist/_ , 1998. 

- [62] Heimo Müller and Andreas Holzinger. Kandinsky patterns. _Artificial Intelligence_ , 2021. 

- [63] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. In _CVPR_ , 2017. 

- [64] Wolfgang Stammer, Patrick Schramowski, and Kristian Kersting. Right for the right concept: Revising neuro-symbolic concepts by interacting with their explanations. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 3619–3629, 2021. 

- [65] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. _Nature Machine Intelligence_ , 2(11):665–673, 2020. 

- [66] Blender Online Community. _Blender - a 3D modelling and rendering package_ . Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. 

- [67] Carla P Gomes, Ashish Sabharwal, and Bart Selman. Model counting. In _Handbook of satisfiability_ , pages 993–1014. IOS press, 2021. 

- [68] Chris Drake. Pyeda: Data structures and algorithms for electronic design automation. In _SciPy_ , pages 25–30, 2015. 

- [69] Adnan Darwiche. Sdd: A new canonical representation of propositional knowledge bases. In _Twenty-Second International Joint Conference on Artificial Intelligence_ , 2011. 

- [70] Supratik Chakraborty, Kuldeep S. Meel, and Moshe Y. Vardi. Algorithmic improvements in approximate counting for probabilistic inference: From linear to logarithmic sat calls. In _Proceedings of International Joint Conference on Artificial Intelligence (IJCAI)_ , 7 2016. 

- [71] Mateo Espinosa Zarlenga, Pietro Barbiero, Gabriele Ciravegna, Giuseppe Marra, Francesco Giannini, Michelangelo Diligenti, Zohreh Shams, Frederic Precioso, Stefano Melacci, Adrian Weller, et al. Concept embedding models. _arXiv preprint arXiv:2209.09056_ , 2022. 

- [72] Stevan Harnad. The symbol grounding problem. _Physica D: Nonlinear Phenomena_ , 42(1–3):335–346, June 1990. 

- [73] Subbarao Kambhampati, Sarath Sreedharan, Mudit Verma, Yantian Zha, and Lin Guan. Symbols as a lingua franca for bridging human-ai chasm for explainable and advisable ai systems. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 36, pages 12262–12267, 2022. 

- [74] Mingjie Li and Quanshi Zhang. Does a neural network really encode symbolic concepts? In _International Conference on Machine Learning_ , pages 20452–20469. PMLR, 2023. 

14 

- [75] Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks. _Advances in Neural Information Processing Systems_ , 34:9574–9586, 2021. 

- [76] Gesina Schwalbe. Concept embedding analysis: A review. _arXiv preprint arXiv:2203.13909_ , 2022. 

- [77] Eunji Kim, Dahuin Jung, Sangha Park, Siwon Kim, and Sungroh Yoon. Probabilistic concept bottleneck models. _arXiv preprint arXiv:2306.01574_ , 2023. 

- [78] Pietro Barbiero, Gabriele Ciravegna, Francesco Giannini, Mateo Espinosa Zarlenga, Lucie Charlotte Magister, Alberto Tonda, Pietro Lió, Frederic Precioso, Mateja Jamnik, and Giuseppe Marra. Interpretable neural-symbolic concept reasoning. In _International Conference on Machine Learning_ , pages 1801–1825. PMLR, 2023. 

- [79] Anita Mahinpei, Justin Clark, Isaac Lage, Finale Doshi-Velez, and Weiwei Pan. Promises and pitfalls of black-box concept learning models. _arXiv preprint arXiv:2106.13314_ , 2021. 

- [80] Emanuele Marconato, Andrea Passerini, and Stefano Teso. Interpretability is in the mind of the beholder: A causal framework for human-interpretable representation learning. _Entropy_ , 25(12):1574, 2023. 

- [81] Chris Olah. Mechanistic interpretability, variables, and the importance of interpretable bases. 2022. 

- [82] Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. _Advances in Neural Information Processing Systems_ , 36:16318–16352, 2023. 

- [83] Atticus Geiger, Chris Potts, and Thomas Icard. Causal abstraction for faithful model interpretation. _arXiv preprint arXiv:2301.04709_ , 2023. 

- [84] Ziqian Zhong, Ziming Liu, Max Tegmark, and Jacob Andreas. The clock and the pizza: Two stories in mechanistic explanation of neural networks. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [85] Aapo Hyvärinen and Erkki Oja. Independent component analysis: algorithms and applications. _Neural networks_ , 13(4-5):411–430, 2000. 

- [86] Luigi Gresele, Julius Von Kügelgen, Vincent Stimper, Bernhard Schölkopf, and Michel Besserve. Independent mechanism analysis, a new concept? _Advances in neural information processing systems_ , 34:28233–28248, 2021. 

- [87] Liang Wendong, Armin Keki´c, Julius von Kügelgen, Simon Buchholz, Michel Besserve, Luigi Gresele, and Bernhard Schölkopf. Causal component analysis. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [88] Sebastien Lachapelle, Tristan Deleu, Divyat Mahajan, Ioannis Mitliagkas, Yoshua Bengio, Simon Lacoste-Julien, and Quentin Bertrand. Synergies between disentanglement and sparsity: Generalization and identifiability in multi-task learning. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 18171–18206. PMLR, 23–29 Jul 2023. 

- [89] Marco Fumero, Florian Wenzel, Luca Zancato, Alessandro Achille, Emanuele Rodolà, Stefano Soatto, Bernhard Schölkopf, and Francesco Locatello. Leveraging sparse and shared feature activations for disentangled representation learning, 2023. 

- [90] Armeen Taeb, Nicolo Ruggeri, Carina Schnuck, and Fanny Yang. Provable concept learning for interpretable predictions using variational autoencoders, 2022. 

- [91] Eriq Augustine, Connor Pryor, Charles Dickens, Jay Pujara, William Wang, and Lise Getoor. Visual sudoku puzzle classification: A suite of collective neuro-symbolic tasks. In _International Workshop on Neural-Symbolic Learning and Reasoning (NeSy)_ , 2022. 

15 

- [92] Fan Shi, Bin Li, and Xiangyang Xue. Towards generative abstract reasoning: Completing raven’s progressive matrix via rule abstraction and selection. _arXiv preprint arXiv:2401.09966_ , 2024. 

- [93] Luca Salvatore Lorello, Marco Lippi, and Stefano Melacci. The kandy benchmark: Incremental neuro-symbolic learning and reasoning with kandinsky patterns. _arXiv preprint arXiv:2402.17431_ , 2024. 

- [94] Eleonora Giunchiglia, Mihaela C˘at˘alina Stoian, Salman Khan, Fabio Cuzzolin, and Thomas Lukasiewicz. Road-r: The autonomous driving dataset with logical requirements. _Machine Learning_ , pages 1–31, 2023. 

- [95] Emile van Krieken, Samy Badreddine, Robin Manhaeve, and Eleonora Giunchiglia. Uller: A unified language for learning and reasoning. _arXiv preprint arXiv:2405.00532_ , 2024. 

- [96] Yoshihide Sawada and Keigo Nakamura. Concept bottleneck model with additional unsupervised concepts. _IEEE Access_ , 10:41758–41765, 2022. 

- [97] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research_ , 12:2825–2830, 2011. 

- [98] Cian Eastwood and Christopher KI Williams. A framework for the quantitative evaluation of disentangled representations. In _International conference on learning representations_ , 2018. 

- [99] Tommaso Carraro. LTNtorch: PyTorch implementation of Logic Tensor Networks, mar 2022. 

- [100] Tuomas Oikarinen, Subhro Das, Lam M Nguyen, and Tsui-Wei Weng. Label-free concept bottleneck models. _arXiv preprint arXiv:2304.06129_ , 2023. 

- [101] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, _3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings_ , 2015. 

- [102] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015. 

- [103] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _CVPR_ , pages 248–255. Ieee, 2009. 

- [104] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 2636–2645, 2020. 

- [105] Daphne Koller and Nir Friedman. _Probabilistic graphical models: principles and techniques_ . MIT press, 2009. 

- [106] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III au2, and Kate Crawford. Datasheets for datasets, 2021. 

- [107] Umili, E., Argenziano, F., & Capobianco, R. (2024). Neural Reward Machines. _arXiv preprint arXiv:2408.08677_ . 

- [108] Umili, E., Capobianco, R., & De Giacomo, G. (2023). Grounding LTLf specifications in image sequences. In _Proceedings of the International Conference on Principles of Knowledge Representation and Reasoning_ (Vol. 19, No. 1, pp. 668–678). 

- [109] DeLong, L. N., Gadiya, Y., Galdi, P., Fleuriot, J. D., & Domingo-Fernández, D. (2024). MARS: A neurosymbolic approach for interpretable drug discovery. _arXiv preprint arXiv:2410.05289_ . 

- [110] Bereska, L., & Gavves, E. (2024). Mechanistic Interpretability for AI Safety–A Review. _arXiv preprint arXiv:2404.14082_ . 

16 

## **NeurIPS Paper Checklist** 

## 1. **Claims** 

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? 

Answer: [Yes] 

Justification: Our key claims are stated in the abstract and introduction, see the contributions paragraph. Summarizing, the main claim is that existing benchmark for learning and reasoning are insufficient to evaluate reasoning shortcuts and mitigation strategies, and we aim to fill this gap. 

## 2. **Limitations** 

Question: Does the paper discuss the limitations of the work performed by the authors? 

Answer: [Yes] 

Justification: We discuss the key limitations of `rsbench` in Section 5 and potential issues with TCAV-based concept extraction in Appendix A.2. 

3. **Theory Assumptions and Proofs** 

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? 

Answer: [N/A] 

Justification: We make no theoretical claims. 

## 4. **Experimental Result Reproducibility** 

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? 

Answer: [Yes] 

Justification: The data and code of `rsbench` , as well as the code used for our evaluation, are readily available at the link provided in the abstract. All experimental details are supplied in the appendix. 

5. **Open access to data and code** 

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? 

Answer: [Yes] 

Justification: The full benchmark suite, available at the provided URL, comes with documentation. 

6. **Experimental Setting/Details** 

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? 

Answer: [Yes] 

Justification: See Section 4 and Appendix B. Additional details can be found in the source code. 

## 7. **Experiment Statistical Significance** 

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 

Answer: [Yes] 

Justification: All tables come with error bars reporting the standard deviations. 

8. **Experiments Compute Resources** 

17 

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? 

Answer: [Yes] 

Justification: In . In short, the experiments were conducted on an A5000 GPU, and Blender rendering on a K40 GPU. 

## 9. **Code Of Ethics** 

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics `https://neurips.cc/public/EthicsGuidelines` ? 

Answer: [Yes] 

Justification: All authors read and followed the code of conduct. Our work does rely on experiments with human subjects or crowd-workers. All data sets we provide are freely available for research purposes and suitable for evaluating the impact reasoning shortcuts. The `BDD-OIA` data set is also available for research under its own license, see Appendix C.1. We believe our tasks and data sets pose no immediate harmful consequences. We discuss potential issues in Section 5. 

## 10. **Broader Impacts** 

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? 

Answer: [Yes] 

Justification: We briefly discuss broader impact in Section 5. 

11. **Safeguards** 

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? 

Answer: [N/A] 

Justification: We believe our tasks and data sets pose no such risk. 

12. **Licenses for existing assets** 

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? 

Answer: [Yes] 

Justification: All assets are properly credited, see Appendix C.11.1 and Appendix C.1. 

13. **New Assets** 

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 

Answer: [Yes] 

Justification: All new assets – tasks, data sets, data generators, code – are documented on the online website. 

## 14. **Crowdsourcing and Research with Human Subjects** 

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 

Answer: [N/A] 

Justification: The paper does not involve crowdsourcing nor research with human subjects. 

15. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects** 

18 

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? 

Answer: [N/A] 

Justification: The paper does not involve crowdsourcing nor research with human subjects. 

19 

## **A Metrics: Additional Details** 

## **A.1 Model-level Metrics** 

**Label and Concept Evaluation.** For all datasets, we evaluate the predictions on the labels by measuring the _F_ 1-score with macro average. We followed these specifics for `MNAdd-EvenOdd` and `Kand-Logic` . 

In `BDD-OIA` and `SDD-OIA` , there are 4 labels and 21 concepts, and to measure the _F_ 1 score we adopt a softened metric [19, 96], namely, the mean- _F_ 1 score and a mean accuracy. Specifically, we first compute the binary _F_ 1-score and accuracy for each concept _Ci_ and then average them: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0020-04.png)


Similarly, for the concepts, we perform the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0020-06.png)


**Concept Collapse.** For all datasets, to measure the _**Concept collapse**_ Cls( _C_ ), we first compute the confusion matrix. Here, we provide additional details when not all ground-truth concepts **C** _[∗]_ appear in the test set. To this end, it is desirable to mention how the confusion matrix is extracted. 

Let _C[∗] ⊆{_ 0 _,_ 1 _}[k]_ be the subset of concepts vectors appearing in the test set, and _C_ be the subset of concepts vectors predicted by the model for inputs in the test set, _e.g._ , taking the MAP estimate. Notice that both _|C|_ and _|C[∗] |_ are below or equal to 2 _[k]_ . To evaluate the confusion matrix in this multilabel setting, the set of labels is determined by first converting each binary string to its integer value, _e.g._ , (0 _,_ 1 _,_ 1) _�→_ 3. Let _F_ ( _C_ ) and _F_ ( _C[∗]_ ) be the two subsets converted to categorical values from _C_ and _C[∗]_ , respectively. From _F_ ( _C_ ) and _F_ ( _C[∗]_ ) we obtain the confusion matrix _C_ using the `scikit-learn` library [97]. In this case, the output would be a matrix _C ∈_ [0 _,_ 1] _[m][×][m]_ , where _m_ = _|F_ ( _C_ ) _∪F_ ( _C[∗]_ ) _|_ , where categorical values not appearing in _F_ ( _C[∗]_ ) will give empty rows, _i.e._ , _Ci,_ : = (0 _, . . . ,_ 0) _[⊤]_ , for all _i ̸∈F_ ( _C[∗]_ ). Following the previous definition, we obtain that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0020-09.png)


Then, collapse can be evaluated as before: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0020-11.png)


Notice that when _m_ = 2 _[k]_ the form reduces to the one discussed in the main text. 

For `Kand-Logic` , we took the concept for each geometric figure, using a 6-dimensional one-hot encoding for shape (3) and color (3), and computed the collapse after converting this base 3 representation to a base 10 integer. `rsbench` provides a way to compute the collapse for shape and color separately. In this case, we compute the confusion matrix as is without the need for a conversion. We threat the digits predictions in `MNAdd-EvenOdd` , in the same way. 

To compute the Cls( _C_ ) for `BDD-OIA` and `SDD-OIA` , we convert the binary 21-concept prediction to an integer and compute the collapse. `rsbench` also allows computing the collapse for each concept associated with corresponding categories (e.g., `move_forward` , `stop` , `turn_left` , `turn_right` ) in the same manner. 

## **A.2 The Impact of TCAV** 

We extract concepts from neural networks by leveraging the TCAV post-hoc explainer [48]. In essence, TCAV acts as a linear probe: for each concept, it trains a binary linear classifier using the network’s embeddings (typically from the second-to-last layer) as inputs and the concept’s annotations as targets, distinguishing between when a concept is present ( _xc_ ) and when it is absent ( _x¬c_ ). From 

20 

this classifier, we extract the Concept Activation Vector (CAV), which corresponds to the weights of the linear decision boundary, denoted as _v_ CAV. 

To assess whether a concept is present, we use the TCAV score, which checks whether the model’s prediction aligns positively with the CAV by computing: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0021-02.png)


where _[∂] ∂h[f]_ ([(] _x[x] i[i]_ )[)][represents the gradient of the model’s output with respect to the embedding space] _[ h]_[(] _[x][i]_[)][,] and _·_ denotes the dot product. A positive score suggests that the concept is contributing to the model’s prediction. 

We employed the TCAV score to determine the presence of concepts in both `NN` and `CLIP` , as discussed in Section 4. 

One issue is that TCAV is not always reliable, depending on the task at hand, meaning that it might mis-predict the concepts learned by the model, simply because these concepts may not be linearly separable in the embedding space. This occurs even when the linear classifiers perform well on held-out data. 

This can lead to overestimating concept collapse, as noted in Section 4, and to underestimating the quality of implicitly learned concepts. Possible remedies, which we plan to implement as we develop `rsbench` further, include replacing TCAV with more advanced techniques from mechanistic interpretability [75, 82]. 

## **A.3 Task-level Metrics** 

To properly understand how the `countrss` works and how it is possible to count the number of RSs we need to introduce technical details on what assumptions and conditions are required. We report in this section an overview of the theoretical material in [20, 25], precisely meant to explain comprehensively the counting of RSs. Additional details about derivations and proofs for RSs characterization can be found in the references above. 

To this end, we need to ( _i_ ) introduce the functional form of NeSy predictors along with their training objective and the optimality condition, and ( _ii_ ) the data generation process and the notion of intended semantics. ( _iii_ ) By leveraging two simplifying assumptions, it is possible to derive a formula for counting the number of optimal solutions (including RSs). 

**Neuro-Symbolic models and Learning Objective.** Without loss of generality, we analyze NeuroSymbolic models leveraging Probabilistic Logic approaces [16, 14, 9] of the form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0021-11.png)


The _**perception**_ is performed via _pθ_ ( **c** _|_ **x** ), computing a high-level conceptual description of a low-level input, usually implemented as a neural network; and the _**reasoning**_ module is given by that _p_ ( **y** _|_ **c** ; K), that infers a prediction **y** based on the high-level description **c** and prior knowledge K. 

The learning objective for models of this form is given by the maximization of the likelihood on data a set _D_ = _{_ ( **x** _,_ **y** ) _}_ . We denote with ( **D1** ) the condition to which NeSy models attain the optimum of likelihood that is _θ[∗] ∈_ argmax _θ_ �( **x** _,_ **y** ) _∈D_[log] _[ p][θ]_[(] **[y]** _[ |]_ **[ x]**[;][ K][)][.] 

**Data Generation Process and Intended Semantics.** Similar to [20], we assume data are distributed according to a ground-truth distribution _p[∗]_ ( **x** _,_ **y** ; K). There exist _k_ latent, _ground-truth concepts_ **c** _[∗] ∈{_ 0 _,_ 1 _}[k]_ drawn from an unobserved distribution _p_ ( **c** _[∗]_ ), where input variables **x** _∈_ R _[n]_ are distributed according to the conditional _p_ ( **x** _|_ **c** _[∗]_ ). Latent concepts also generate the label **y** according to distributions _p_ ( **x** _|_ **c** _[∗]_ ; K). The overall distribution on the observed inputs and output labels is given by _p_ ( **x** _,_ **y** ; K) = E **c** _∗∼p∗_ ( **c** _∗_ ) _p_ ( **x** _|_ **c** _[∗]_ ) _p_ ( **y** _|_ **c** ; K). The conditional distribution _p_ ( **c** _[∗] |_ **x** ) describes how concepts are distributed according to the input. 

Based on this, we denote with ( **D2** ) the condition for which the learned concepts possess the intended semantics, that is _pθ_ ( **c** _|_ **x** ) _≡ p_ ( **c** _[∗] |_ **x** ). 

21 

**Reasoning Shortcuts are Optimal Solutions with Unintended Semantics.** Having specified the meaning of intended concepts ( **D2** ) and the optimality condition for NeSy models ( **D1** ), does **D1** = _⇒_ **D2** , even in the limit of infinite data? Attaining optimality with NeSy models is insufficient to learn concepts with the intended semantics [20], making the implication not to hold in general. 

It is possible to detect the presence of RSs before-hand, leveraging two technical assumptions: 

- **A1 Invertibility** , the ground-truth map from input to concept is given by a function _f[∗]_ : **x** _�→_ **c** _[∗]_ , _i.e._ , _p_ ( **c** _[∗] |_ **x** ) = 1 _{_ **c** _[∗]_ = _f[∗]_ ( **x** ) _}_ ; 

- **A2 Determinism** , the labels are uniquely determined by the knowledge K and concepts by a function _β_ K : **c** _�→_ **y** underlying the knowledge, _i.e._ , _p_ ( **y** _|_ **c** _,_ K) = 1 _{_ **y** = _β_ K( **c** ) _}_ . 

Intuitively, **A1** indicates that ground-truth concepts are determined uniquely from the input, _i.e._ , ground-truth concepts can be read from the input variables. This means that **D2** reduces to obtain the function _f[∗]_ via the model _pθ_ ( **c** _| ·_ ). On the other hand, **A2** subtends the fact that for knowledge K there is only one **y** that complies with concepts **c** , that is ( **c** _,_ **y** ) _|_ = K and ( **c** _,_ **y** _[′]_ ) _|_ = K, _if and only if_ **y** _[′]_ = **y** . This also means that given the concepts and the knowledge, only one label vector is associated with them. 

We indicate with supp( **c** _[∗]_ ) the support of the ground-truth concepts distribution _p_ ( **c** _[∗]_ ) and with _A_ the space of all maps _α_ : _{_ 0 _,_ 1 _}[k] →{_ 0 _,_ 1 _}[k]_ , mapping one concept vector to another. Based on this, we can derive a count for optimal NeSy models of a particular form: 

**Theorem 1** (Misspecification of NeSy models [20]) **.** _Under_ _**A1** and_ _**A2** , the number of models of the form pθ_ ( **c** _|_ **x** ) = 1 _{_ **c** = _fθ_ ( **x** ) _}, with fθ_ = ( _α ◦ f[∗]_ ) _, attaining maximum likelihood amounts to:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0022-07.png)


In a nutshell, the theorem proves that the number of alternative solutions to the ground-truth one _f[∗]_ is given by those maps _α ∈A_ that _map ground-truth concepts to valid alternatives for label predictions_ . When the number is above 1, RSs are present in the learning problem. The formula offers a principled way to count RSs in practice and allows to design tasks where RSs are present and we show a practical implementation with `countrss` . 

**Explicit count of optimal solutions.** Marconato _et al._ [20] showed that it is possible to derive an analytical expression for the total number of optimal solutions appearing in Eq. (7). This requires making assumptions on how the count is performed: 

- **C1** ) All possible maps _α_ : _{_ 0 _,_ 1 _}[k] →{_ 0 _,_ 1 _}[k]_ can be learned by the neural model, that is _A_ is 2 _[k]_ 

- complete and of cardinality _|A|_ = (2 _[k]_ ) 

- **C2** ) The support of _p_ ( **c** _[∗]_ ) is complete, that is supp( **c** _[∗]_ ) = 2 _[k]_ 

Here, assumption ( **C1** ) allows to count the total number of maps by considering the limit case where each _α_ ( **c** ) can be predicted independently from _α_ ( **c** _[′]_ ), for **c** = **c** _[′]_ . It is possible to derive the total count of possible optimal _α_ ’s [20], given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0022-13.png)


where the set _E_ ( **c** _,_ K) _⊂{_ 0 _,_ 1 _}[k]_ contains all concepts **C** that yield the same result under _β_ K, and it is defined as _E_ ( **c** _,_ K) := _{_ **c** _[′]_ : _β_ K( **c** _[′]_ ) = _β_ K( **c** ) _}_ . In other terms, _E_ ( **c** _,_ K) _⊂{_ 0 _,_ 1 _}[k]_ is the equivalence class for **c** and logic K. It is possible to relax **C2** by capturing a more general case and showing that RSs can be even more. The underlying idea is that for all **c** _[′′] ̸∈_ supp( **c** _[∗]_ ), _i.e._ , not appearing in the training set, a map _α_ can predict whatever element in _{_ 0 _,_ 1 _}[k]_ without affecting the training likelihood. This gives an even bigger number of solutions provided by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0022-15.png)


However, relaxing **C1** makes finding an explicit expression for counting optimal solutions more complicated. We then resort to formal methods to show that this can be done algorithmically. 

22 

**Algorithmic implementation with** `countrss` **.** We detail the encoding of maps _α_ and their counting in propositional logic, under the previous assumptions **A1** and **A2** . We denote with B = _{_ 0 _,_ 1 _}_ and matrix element **L** _i,j_ as **L** [ _i, j_ ]. For ease of exposition, we additionally assume that all concepts _ci_ , for _i ∈{_ 1 _, . . . , k}_ , have the same number of values _b ∈_ N[+] . We introduce **O** _∈_ B _[k][×][k]_ the boolean matrix encoding the mapping between the ground-truth concepts **c** _[∗]_ and the predicted concepts **c** , where _k ∈_ N[+] is the number of concepts. Intuitively, **O** determines what learned concepts depend on the ground truth ones. For example, it may be the case that in `Kand-Logic` , the concept _c_ `shape` depends on _c[∗]_ `color`[and] _[ c]_ `[color]`[depends on] _[ c][∗]_ `shape`[.] 

The full mapping from ground truth to predicted concept values is defined by the block matrix **A** _∈_ B[(] _[k][·][b]_[)] _[×]_[(] _[k][·][b]_[)] . This mapping encodes how the values of the ground truth concepts are mapped to the values of the predicted predicted. For example, in `Kand-Logic` it may be the case that _α_ : _red �→_ □ and _α_ : □ _�→ red_ , and so on for other colors and shapes. We require that this assignment of **A** is consistent with the assignment of **O** . This means that the _i, j_ -block of **A** has non-zero entries if and only if the _i_ -th ground truth concept is mapped onto the _j_ -th predicted concept: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0023-02.png)


**A** also must have exactly one positive entry for each column, encoding the fact that a single groundtruth value cannot be mapped to two or more values by the model. Again, multiple ground-truth values can still end up collapsing into the same predicted value: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0023-04.png)


Out of every assignment to **A** satisfying the constraints above, only those that are consistent with the supervision can achieve optimal predictive performance and therefore be potential RSs. Let **c** _[∗] ∈_ B _[k][·][b]_ denote the boolean vector encoding the ground truth concept values appearing in the support of _p_ ( **c** _[∗]_ ), and let ˆ **c** _∈_ B _[k][·][b]_ denote the predicted concepts for the ground-truth concept, which is defined as the boolean dot product (denoted as _⊗_ ) of **c** _[∗]_ with **A** : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0023-06.png)


Finally, by denoting with K and **y** _[∗]_ the logical encoding of the task and the ground-truth label for the **c** _[∗]_ (corresponding to **y** _[∗]_ = _β_ K( **c** _[∗]_ )) respectively, we constrain the model predictions to be correct, which in turn forces the values of **A** to comply with condition ( **D1** ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0023-08.png)


The full encoding is the conjunction of Eq. (10), Eq. (11), Eq. (12), and Eq. (13). This is fed to a model counter, whose output equals the number of possible assignments to **A** that satisfy the formula. When this number is above 1, RSs are present in the learning problem. This number indicates the optimal maps _α_ ’s (according to the specifics of the constraints) that can be learned by the NeSy model. 

Notice that, without any additional constraint on **O** , the count would enumerate all RSs as done in Eq. (9). We instead search for more specific RSs that relax condition **C1** . In our setup, we constrain each ground-truth concept _c[∗] i_[to be mapped to a single extracted concept] _[ c][i]_[.][This condition is typically] referred to as _completeness_ [98], that is there are no copies or repetitions of ground-truth concepts in the learned ones. For example, we avoid counting solutions where the map _α_ predicts two concepts, say _c_ 1 and _c_ 2, only depending on one ground-truth concept, say _c[∗]_ 1[.][Notice that, however, multiple] concepts _c[∗] i_[can still affect a single concept in] _[ c][j]_[.][Formally, we achieve this by enforcing exactly one] (OHE) positive value in each column of **O** : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0023-11.png)


23 

where OHE( _a_ 1 _, . . . , ak_ ) is one _if and only if_ there is only one _ai_ = 1 and the remaining are zero, otherwise zero. Basically, the matrix **O** encodes all maps of concept indices, a necessary element whenever there is no clear notion of ordering among concepts (CPLX **x** ). 

Then, this additional constraint can be added to the previous conjunction and used to evaluate the number of RSs. One perk of `countrss` is that many additional constraints can be added to the conjunction of Eq. (10), Eq. (11), Eq. (12), and Eq. (13) and evaluate different situations depending on the expected model architectural biases. By adding 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0024-02.png)


to Eq. (14), we then require that only permutation maps are considered, that is one **c** _i_ only depends on one **c** _[∗]_[Further constraints can also be added to include] _π_ ( _i_ )[, where] _[ π]_[ is a permutation of] _[ k]_[ elements.] concept supervision. 

## **B Experiments: Additional Details** 

All experiments were conducted using Python 3.8 and PyTorch 1.13, executed on a single A5000 GPU. The implementation of `DPL` was taken from [25]. For `LTN` , `NN` , `CBM` , and `CLIP` , new implementations were developed from scratch. `LTN` models were developed employing `LTNtorch` [99]. As for `CLIP` , we leveraged the implementation of [100]. For all the datasets, we used the pre-trained (with contrastive learning for image-caption matching, see [54]) visual transformer ViT-B/32 to this end, passing all input images to first a rescaling transformation of the image to 224 _×_ 224 _×_ 3 to comply with the backbone layer. Visual embeddings are then saved and made available in our supplementary material for successive fine-tuning of neural predictors. 

For the dataset generation process of `rsbench` , we utilized Python 3.7 along with Blender 2.91, leveraging the `bpy` Python package set to version 2.91a0. The methodology for generating `CLE4EVR` variants and `SDD-OIA` was inspired by the work of Johnson et al. [63]. The images, which require Blender rendering, were generated using a Tesla K40c GPU. 

`SDD-OIA` was generated using seed 0, with random splits of 0 _._ 7 for train, 0 _._ 15 for validation, and 0 _._ 15 for test. The configuration was 0 _._ 9 in-distribution and 0 _._ 1 out-of-distribution. The dataset comprises 6820 training examples, 1464 validation examples, 1464 test examples, and 1000 OOD examples. `Kand-Logic` and `MNAdd-EvenOdd` are the same datasets used in [25]. `MNMath` was generated using seed 0 with random splits, maintaining an 80 _/_ 20 ratio for in-distribution and out-of-distribution data. It contains 1 _,_ 000 training samples, 200 validation samples, 300 test samples, and 200 OOD samples. `MNLogic` was also generated using seed 0 and random splits, with an 80 _/_ 20 ID/OOD ratio. It consists of 1 _,_ 000 training samples, 200 validation samples, 300 test samples, and 300 OOD samples. 

All models were trained using end-to-end training, providing supervision on the ground truth labels and not on the concept labels, except for `CBM` where few concept supervision has been provided. 

## **B.1 Additional experiments:** `MNMath` **and** `MNLogic` 

In this section, we discuss additional experiments conducted on `MNMath` and `MNLogic` datasets. 

For `MNMath` , we chose a task involving two equations with 8 digits (4 per equation), where **x** = ( **x** 1 _,_ **x** 2 _,_ **x** 3 _,_ **x** 4 _,_ **x** 5 _,_ **x** 6 _,_ **x** 7 _,_ **x** 8). We designed a multi-task setup: one task predicts whether _y_ 1 = 1 _{_ **x** 1 + **x** 2 = **x** 3 + **x** 4 _}_ , and the other checks if _y_ 2 = 1 _{_ **x** 5 _·_ **x** 6 = **x** 7 _·_ **x** 8 _}_ holds. 

This version of `MNMath` includes all possible digit values (0 to 9). We chose this setup because it is particularly prone to reasoning shortcuts. Specifically, if all digits are mapped to a fixed value, _e.g._ , _, . . . , →_ 4, they solve the `MNMath` task. 

We evaluated three models ( `DPL` , `NN` , and `CBM` ) on this benchmark. The setup mirrors previous experiments (Section 4), where each model receives supervision only on the final labels so that the concepts are treated as latent variables. As `CBM` require concept supervision, we gave supervision on few concepts, specifically 0 _,_ 5 and 9. In this case the model predicts the concept (the digit value) for each digit independently. Following that, the model produces two labels, representing whether the two formulas are true, respectively. 

24 

The results in Table 6 summarize the performance of all models, averaged across five different seeds. The table highlights that all models struggle with reasoning shortcuts (RSs), reflected in the low concept accuracy (Acc _C_ ) and concept fidelity (F1( _C_ )) scores, particularly for `NN` and `CBM` . However, despite these issues, the models still manage to achieve moderate to high performance on label prediction metrics (Acc _Y_ and F1( _Y_ )). 

Table 6: Results on `MNMath` 

||Acc_Y_(_↑_)|F1(_Y_)(_↑_)|Acc_C_(_↑_)|F1(_C_)(_↑_)|Cls(_C_)(_↓_)|
|---|---|---|---|---|---|
|`DPL`|0_._80_±_0_._10|0_._73_±_0_._13|0_._11_±_0_._01|0_._03_±_0_._02|0_._01_±_0_._01|
|`CBM`_†_|0_._75_±_0_._01|0_._67_±_0_._01|0_._22_±_0_._04|0_._11_±_0_._03|0_._68_±_0_._15|
|`NN`_∗_|0_._75_±_0_._01|0_._67_±_0_._01|0_._10_±_0_._01|0_._03_±_0_._01|0_._80_±_0_._11|



For `MNLogic` , we focused on evaluating the XOR operation on 4 bits. In this case, the input is **x** = ( **x** 1 _,_ **x** 2 _,_ **x** 3 _,_ **x** 4), and the task is to compute the output _y_ = **x** 1 _⊕_ **x** 2 _⊕_ **x** 3 _⊕_ **x** 4. 

We chose the XOR operation for its inherent ambiguity, which can lead to reasoning shortcuts, as discussed in previous work [20]. 

We evaluated the `DPL` , `NN` , and `CBM` models under the same setup as in `MNMath` , with the sole exception that there is no weight sharing among the components processing each digit, which makes the task more challenging. Unlike in `MNMath` , the `CBM` model did not receive concept supervision, as supervision for either 0 or 1 would suffice for learning the concept correctly. 

The results in Table 7 reflect averages across five seeds. While all models demonstrated high performance on the task (as indicated by Acc _Y_ and F1( _Y_ )), they exploited unintended concepts, evident from the Acc _C_ and F1( _C_ ) metrics. Interestingly, the 50% concept accuracy achieved by all models indicates that there is no inherent preference for any model to favor one solution over another, whether it be the identity or the reasoning shortcut ( illustrated in Fig. 3). 

Table 7: Results on `MNLogic` 

||Acc_Y_(_↑_)|F1(_Y_)(_↑_)|Acc_C_(_↑_)|F1(_C_)(_↑_)|Cls(_C_)(_↓_)|
|---|---|---|---|---|---|
|`DPL`|0_._99_±_0_._01|0_._99_±_0_._01|0_._51_±_0_._06|0_._47_±_0_._05|0_._01_±_0_._01|
|`CBM`_†_|0_._95_±_0_._10|0_._89_±_0_._23|0_._50_±_0_._05|0_._48_±_0_._05|0_._01_±_0_._01|
|`NN`_∗_|0_._98_±_0_._01|0_._60_±_0_._20|0_._46_±_0_._05|0_._41_±_0_._10|0_._10_±_0_._20|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0025-09.png)


Figure 3: `MNLogic` reasoning shortcut 

## **B.2 Loss Functions** 

For `NN` , `CBM` , `DPL` and `CLIP` , the loss function corresponding to log-likelihood maximization is given by the cross-entropy loss, defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0025-13.png)


25 

where _yi_ = 1 if _i_ is the ground-truth label for input **x** , and _y_ ˆ _i_ is the predicted probability for class _i_ by the model when passed the input example **x** . 

`LTN` is the only one that differs and utilizes the Logic Tensor Network loss [57]. In `LTN` , the neural networks represent First-Order Logic predicates ( _e.g._ , the predicate Digit in the `MNAdd` task). Formulas are constructed recursively by using fuzzy logic operators ( _e.g._ , fuzzy quantifiers and logical connectives). The `LTN` loss imposes on learning the parameters of such predicates in such a way the satisfaction of the knowledge base is maximized. Given a First-Order Logic knowledge base K, the loss can be defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0026-02.png)


where _ϕ_ is a formula contained in K (like the addition for `MNAdd` ), SatAgg is an aggregator function that measures overall satisfaction of K, and _Gθ_ ( **x** ; _ϕ_ ) is the `LTN` - _grounding_ (i.e., evaluation) of _ϕ_ given **x** . This special operator is meant to map symbols from the logical domain ( _e.g._ , the symbolic representation of the addition with “+”) to the real domain ( _e.g._ , its computational graph performing the addition). Hence, _Gθ_ ( **x** ; _ϕ_ ) can be seen as a fuzzy truth value resulting from the evaluation of logical formula _ϕ_ on input **x** . _θ_ are the parameters of the learnable predicates contained in _ϕ_ . In the case of multi-label predictions, parameters _θ_ of the neural network are shared among different formulas _ϕ ∈_ K ( _e.g._ , like in `MNMath` where different equations appear in the system). Learning is then performed by differentiating the above expression. 

The value of the loss depends on the semantics of the fuzzy logic operators used to approximate each logical connective ( _e.g._ , AND, OR, IMPLICATION) and quantifier ( _e.g._ , EXISTS, FORALL). 

For `CBM` , we provided partial supervision by selecting only a few concept classes for supervision, rather than supervising all concepts. The concept supervision was also implemented using crossentropy loss, specifically applied to the concepts. The cross-entropy loss for concepts is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0026-06.png)


where _cib_ = 1 if _i_ -th ground-truth concept has value _b_ for input **x** , and ˆ _cib_ is the predicted probability for concept _i_ with value _b_ . Here, _Bi_ denotes the cardinality of the _i_ -th concept, and _mi_ = 1 if the concept _i_ is supervised, otherwise being zero. 

**Entropy.** To further explore the concept space, for `LTN` and `DPL` in `MNAdd-EvenOdd` , we applied an entropy loss on the bottleneck of the concepts. The entropy loss encourages diversity in the concept representations and is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0026-09.png)


where ˆ _ci_ is the average probability, evaluated over the batch elements. 

**Combined losses.** In scenarios where both ground truth labels and concept labels were used, the total loss is a weighted sum of the cross-entropy loss and the concept loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0026-12.png)


where _wc_ and _wh_ are hyperparameters that controls the trade-off between the two losses. For `LTN` the equivalent can be obtained by substituting _L_ CE( **x** ) with _L_ ltn( **x** ). 

## **B.3 Model Selection** 

All experiments rely on the Adam optimizer [101]. Hyperparameters were selected through a comprehensive grid search over predefined ranges, considering the macro _f_ 1 performance metric on a validation set. All experiments were run for 40 epochs employing early stopping, by saving the model which performs best in f1 score on the validation set. The experiments, aside from `MNMath` and `MNLogic` , were conducted using 10 different seeds: 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021, and 2122. In contrast, `MNMath` and `MNLogic` were tested across 5 different seeds: 1415, 1617, 1819, 2021, and 2223. 

26 

For all experiments, the learning rate _γ_ was fine-tuned within the range of 10 _[−]_[4] to 10 _[−]_[2] . We found that the best performing models tended to have learning rates around 10 _[−]_[3] , striking a balance between convergence speed and stability. 

The batch size _ν_ varied between 32 and 512, while the weight decay _ω_ spanned from 10 _[−]_[4] to 0. We observed that smaller batch sizes generally resulted in more stable training dynamics, particularly for complex models, while moderate weight decay values helped prevent overfitting. 

For `CBM` , since it requires concept supervision, we tuned the weight of the concept supervision _wc_ among 1, 2, and 5. 

When entropy was required to make the model converge, we tuned the weight of the entropy _wh_ loss among 0 _._ 2, 0 _._ 5, 0 _._ 8, 1, and 2. 

Additionally, for `LTN` , the hyper-parameter `p` for quantifiers was adjusted within the range of 2 to 10 with a step size of 2. Moreover, we tuned different fuzzy logic semantics for the fuzzy operators, specifically for AND, OR and IMPLICATION, such as Gödel, Product and Łukasiewicz for AND and OR, while Gödel, Product, Łukasiewicz, Goguen and Kleene-Dienes for IMPLICATION. 

We set the exponential decay rate _β_ to 0 _._ 99 for all experiments, as we empirically observed that it provides the best performance for our tasks. 

Below, you find all the hyperparameters which performed the best on our datasets. 

## **Hyperparameters for** `SDD-OIA` **:** 

- `DPL` , _γ_ = 10 _[−]_[2] , _ν_ = 128, and _ω_ = 10 _[−]_[4] ; 

- `LTN` , _γ_ = 10 _[−]_[3] , _ν_ = 32, _ω_ = 0, and `p` = 2, AND, OR, IMPLICATION set to Product; 

- `NN` , _γ_ = 10 _[−]_[3] , _ν_ = 32, and _ω_ = 10 _[−]_[4] ; 

- `CBM` , _γ_ = 10 _[−]_[2] , _ν_ = 512, _ω_ = 10 _[−]_[4] , and _wc_ = 2; 

- `CLIP` , _γ_ = 10 _[−]_[3] , _ν_ = 32, and _ω_ = 10 _[−]_[2] . 

## **Hyperparameters for** `Kand-Logic` **:** 

- `DPL` , _γ_ = 10 _[−]_[4] , _ν_ = 32, and _ω_ = 0; 

- `LTN` , _γ_ = 10 _[−]_[3] , _ν_ = 128, _ω_ = 10 _[−]_[3] , `p` = 8, and _wh_ = 0 _._ 8 AND set to Godel, OR and IMPLICATION set to Product; 

- `NN` , _γ_ = 10 _[−]_[3] , _ν_ = 256, and _ω_ = 10 _[−]_[1] ; 

- `CBM` , _γ_ = 10 _[−]_[4] , _ν_ = 128, _ω_ = 10 _[−]_[2] , and _wc_ = 2; 

- `CLIP` , _γ_ = 10 _[−]_[3] , _ν_ = 256, and _ω_ = 10 _[−]_[1] . 

**Hyperparameters for** `MNAdd-EvenOdd` **:** 

- `DPL` , _γ_ = 10 _[−]_[3] , _ν_ = 32, _ω_ = 10 _[−]_[4] and _wh_ = 1; 

- `LTN` , _γ_ = 10 _[−]_[3] , _ν_ = 64, _ω_ = 10 _[−]_[4] , `p` = 6, _wh_ = 10, AND set to Godel, OR and IMPLICATION set to Product; 

- `NN` , _γ_ = 10 _[−]_[3] , _ν_ = 32, and _ω_ = 10 _[−]_[1] ; 

- `CBM` , _γ_ = 10 _[−]_[3] , _ν_ = 32, _ω_ = 0, and _wc_ = 2; 

- `CLIP` , _γ_ = 10 _[−]_[2] , _ν_ = 128, and _ω_ = 10 _[−]_[2] . 

**Hyperparameters for** `MNMath` **:** 

- `DPL` , _γ_ = 10 _[−]_[3] , _ν_ = 64, _ω_ = 10 _[−]_[4] and _wh_ = 0; 

- `NN` , _γ_ = 10 _[−]_[4] , _ν_ = 64, and _ω_ = 10 _[−]_[4] ; 

- `CBM` , _γ_ = 10 _[−]_[3] , _ν_ = 64, _ω_ = 10 _[−]_[4] , and _wc_ = 1; 

**Hyperparameters for** `MNLogic` **:** 

- `DPL` , _γ_ = 10 _[−]_[3] , _ν_ = 64, _ω_ = 10 _[−]_[4] and _wh_ = 0; 

- `NN` , _γ_ = 10 _[−]_[3] , _ν_ = 64, and _ω_ = 10 _[−]_[4] ; 

- `CBM` , _γ_ = 10 _[−]_[4] , _ν_ = 64, _ω_ = 10 _[−]_[4] , and _wc_ = 1; 

27 

## **B.4 Model Architectures** 

`SDD-OIA` **:** For `SDD-OIA` , concerning `DPL` , `LTN` and `CBM` , we adopted a pretrained ResNet-18 [102] on ImageNet [103] as concept extractor as outlined in Table 8 and Table 10. In the tables, BasicBlock consists of two convolutional layers with batch normalization and ReLU activation, followed by a residual connection. Instead, we defined a convolutional architecture for `NN` as shown in Table 9. While for processing `CLIP` embeddings we defined a multi-layer perceptron as depicted in Table 11. 

`Kand-Logic` **:** As motivated in [25], for `LTN` , `DPL` , and `CBM` , we adopted a preprocessed version of the dataset with rescaled objects extracted via bounding boxes. In this scenario, each dataset example contains nine objects, with three objects per figure, ordered by their distance from the figure’s origin, and the network processes one object at a time. For `CLIP` and `NN` , we employed the original dataset, where the network processes the entire example at once. As far as the architectures are concerned, we employed a convolutional neural network, specifically Table 16 for `DPL` and `LTN` , Table 18 for `CBM` and Table 17 for `NN` , respectively. Conversely, for handling `CLIP` embeddings, we implemented a multi-layer perceptron, as detailed in Table 19. 

`MNAdd-EvenOdd` **:** In the case of `MNAdd-EvenOdd` , we utilized a convolutional neural network, with architectures specified in Table 12 for `DPL` and `LTN` , in Table 13 for `CBM` , and in Table 14 for `NN` . For processing `CLIP` embeddings, a multi-layer perceptron was employed, as described in Table 15. 

`MNMath` **:** For `MNMath` , we used the same architectures as in `MNAdd-EvenOdd` , specifically a convolutional neural network. The architectures for `DPL` are detailed in Table 12, while those for `CBM` are listed in Table 13. For `NN` , we utilized the architecture described in Table 22. 

`MNLogic` **:** For `MNLogic` , we also applied convolutional networks. The architectures for both `DPL` and `CBM` are provided in Table 21, and the `NN` architecture is outlined in Table 20. 

All the architectures process each digit individually, except for `CLIP` and `NN` . `CLIP` takes the full image embeddings as input. 

Table 8: `DPL` and `LTN` architecture for `SDD-OIA` 

|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|---|---|---|---|
|(3_,_387_,_469)|Convolution|depth=64, kernel=7, stride=2, padding=3||
|(64_,_194_,_235)|BatchNorm2d|dim=64|ReLU|
|(64_,_194_,_235)|MaxPool2d|kernel=3, stride=2, padding=1|ReLU|
|(64_,_97_,_118)|2xBasicBlock|depth=64, kernel=(1, 1), stride=(1, 1), padding=(1, 1)||
|(64_,_97_,_118)|2xBasicBlock|depth=128, kernel=(1, 1), stride=(1, 1), padding=(1, 1)||
|(128_,_49_,_59)|2xBasicBlock|depth=256, kernel=(1, 1), stride=(1, 1), padding=(1, 1)||
|(256_,_25_,_30)|2xBasicBlock|depth=512, kernel=(1, 1), stride=(1, 1), padding=(1, 1)||
|(512_,_13_,_15)|AvgPool2d|dim=(1, 1)||
|(512_,_1_,_1)|Flatten|dim=512||



Table 9: `NN` architecture for `SDD-OIA` 

|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|---|---|---|---|
|(3_,_387_,_469)|Convolution|depth=16, kernel=3, stride=1, padding=1|ReLU|
|(16_,_387_,_469)|MaxPool2d|kernel=2||
|(16_,_193_,_234)|Convolution|depth=32, kernel=3, stride=1, padding=1|ReLU|
|(32_,_193_,_234)|MaxPool2d|kernel=2||
|(32_,_96_,_117)|Convolution|depth=64, kernel=3, stride=1, padding=1||
|(64_,_96_,_117)|MaxPool2d|kernel=2||
|(64_,_48_,_58)|Convolution|depth=128, kernel=3, stride=1, padding=1||
|(128_,_48_,_58)|MaxPool2d|kernel=2||
|(128_,_24_,_29)|Convolution|depth=256, kernel=3, stride=1, padding=1||
|(256_,_24_,_29)|MaxPool2d|kernel=2||
|(256_,_12_,_14)|Convolution|depth=512, kernel=3, stride=1, padding=1||
|(512_,_12_,_14)|MaxPool2d|kernel=2||
|(512_,_6_,_7)|Flatten|dim=21504||
|(21504)|Linear|dim=128|ReLU|
|(128)|Linear|dim=64|ReLU|
|(64)|Linear|dim=4|Sigmoid|



28 

Table 10: `CBM` architecture for `SDD-OIA` 

||Table 1|: `CBM`architecture for`SDD-OIA`||
|---|---|---|---|
|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|(3_,_387_,_469)|Convolution|depth=64, kernel=7, stride=2, padding=3||
|(64_,_194_,_235)|BatchNorm2d|dim=64|ReLU|
|(64_,_194_,_235)|MaxPool2d|kernel=3, stride=2, padding=1|ReLU|
|(64_,_97_,_118)|2xBasicBlock|depth=64, kernel=(1, 1), stride=(1, 1), padding=(1, 1)||
|(64_,_97_,_118)|2xBasicBlock|depth=128, kernel=(1, 1), stride=(1, 1), padding=(1, 1)||
|(128_,_49_,_59)|2xBasicBlock|depth=256, kernel=(1, 1), stride=(1, 1), padding=(1, 1)||
|(256_,_25_,_30)|2xBasicBlock|depth=512, kernel=(1, 1), stride=(1, 1), padding=(1, 1)||
|(512_,_13_,_15)|AvgPool2d|dim=(1, 1)||
|(512_,_1_,_1)|Flatten|dim=512||
|(512)|Linear|dim=4, bias=True|Sigmoid|



## Table 11: `CLIP` architecture for `SDD-OIA` 

|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|---|---|---|---|
|(512_,_1)|Linear|dim=128, bias=True|ReLU|
|(128)|Linear|dim=64, bias=True|ReLU|
|(64)|Linear|dim=4, bias=True|Sigmoid|



## Table 12: `DPL` and `LTN` architecture for `MNAdd-EvenOdd` 

|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|---|---|---|---|
|(1_,_28_,_56)|Convolution|depth=32, kernel=4, stride=2, padding=1|ReLU|
|(32_,_14_,_28)|Dropout|_p_ = 0_._5||
|(32_,_14_,_28)|Convolution|depth=64, kernel=4, stride=2, padding=1|ReLU|
|(64_,_7_,_14)|Dropout|_p_ = 0_._5||
|(64_,_7_,_14)|Convolution|depth=128, kernel=4, stride=2, padding=1|ReLU|
|(128_,_3_,_7)|Flatten|||
|(2688)|Linear|dim=20, bias = True||



Table 13: `CBM` architecture for `MNAdd-EvenOdd` 

|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|---|---|---|---|
|(1_,_28_,_56)|Convolution|depth=32, kernel=4, stride=2, padding=1|ReLU|
|(32_,_14_,_28)|Dropout|_p_ = 0_._5||
|(32_,_14_,_28)|Convolution|depth=64, kernel=4, stride=2, padding=1|ReLU|
|(64_,_7_,_14)|Dropout|_p_ = 0_._5||
|(64_,_7_,_14)|Convolution|depth=128, kernel=4, stride=2, padding=1|ReLU|
|(128_,_3_,_7)|Flatten|||
|(2688)|Linear|dim=20, bias = True|ReLU|
|(20)|Linear|dim=19, bias = True||



Table 14: `NN` architecture for `MNAdd-EvenOdd` 

|Ta|ble 14: `NN`ar|chitecture for`MNAdd-EvenOdd`||
|---|---|---|---|
|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|(1_,_28_,_56)|Convolution|depth=16, kernel=3, stride=1, padding=1|ReLU|
|(16_,_28_,_56)|MaxPool2d|kernel=2, stride=2||
|(16_,_14_,_28)|Convolution|depth=32, kernel=3, stride=1, padding=1|ReLU|
|(32_,_14_,_28)|MaxPool2d|kernel=2, stride=2||
|(32_,_7_,_14)|Flatten|||
|(3136)|Linear|dim=128, bias=True|ReLU|
|(128)|Linear|dim=64, bias=True|ReLU|
|(64)|Linear|dim=19, bias=True||



## Table 15: `CLIP` architecture for `MNAdd-EvenOdd` 

|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|---|---|---|---|
|(1024_,_1)|Linear|dim=256, bias=True|ReLU|
|(256)|Linear|dim=64, bias=True|ReLU|
|(64)|Linear|dim=19, bias=True||



29 

Table 16: `DPL` and `LTN` architecture for `Kand-Logic` 

|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|---|---|---|---|
|(3_,_28_,_28)|Flatten|||
|(2352)|Linear|dim=256, bias=True|ReLU|
|(256)|Linear|dim=128, bias=True|ReLU|
|(128)|Linear|dim=8, bias = True||



Table 17: `NN` architecture for `Kand-Logic` 

||Table 17: `NN`|architecture for`Kand-Logic`||
|---|---|---|---|
|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|(3_,_64_,_192)|Convolution|depth=16, kernel=3, stride=1, padding=1|ReLU|
|(16_,_64_,_192)|MaxPool2d|kernel=2, stride=2||
|(16_,_32_,_96)|Convolution|depth=32, kernel=3, stride=1, padding=1|ReLU|
|(32_,_32_,_96)|MaxPool2d|kernel=2, stride=2||
|(32_,_16_,_48)|Convolution|depth=64, kernel=3, stride=1, padding=1|ReLU|
|(64_,_16_,_48)|MaxPool2d|kernel=2, stride=2||
|(64_,_8_,_24)|Convolution|depth=128, kernel=3, stride=1, padding=1|ReLU|
|(128_,_8_,_24)|MaxPool2d|kernel=2, stride=2||
|(128_,_4_,_12)|Convolution|depth=256, kernel=3, stride=1, padding=1|ReLU|
|(256_,_4_,_12)|MaxPool2d|kernel=2, stride=2||
|(256_,_2_,_6)|Flatten|||
|(3072)|Linear|dim=512, bias=True|ReLU|
|(512)|Linear|dim=64, bias=True|ReLU|
|(64)|Linear|dim=1, bias=True||



## Table 18: `CBM` architecture for `Kand-Logic` 

|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|---|---|---|---|
|(3_,_28_,_28)|Flatten|||
|(2352)|Linear|dim=256, bias=True|ReLU|
|(256)|Linear|dim=128, bias=True|ReLU|
|(128)|Linear|dim=8, bias = True||
|(8)|Linear|dim=6, bias = True||
|(6)|Linear|dim=2, bias = True||



## Table 19: `CLIP` architecture for `Kand-Logic` 

|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|---|---|---|---|
|(1536_,_1)|Linear|dim=256, bias=True|ReLU|
|(256)|Linear|dim=64, bias=True|ReLU|
|(64)|Linear|dim=1, bias=True|Sigmoid|



## Table 20: `NN` architecture for `MNLogic` 

|Ta|ble 20: `NN`a|rchitecture for`MNLogic`||
|---|---|---|---|
|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|(1_,_28_,_112)|Convolution|depth=16, kernel=3, padding=1|ReLU|
|(16_,_28_,_112)|MaxPool2d|kernel=2, stride=2||
|(16_,_14_,_56)|Convolution|depth=32, kernel=3, padding=1|ReLU|
|(32_,_14_,_56)|MaxPool2d|kernel=2, stride=2||
|(32_,_7_,_28)|Convolution|depth=64, kernel=3, padding=1|ReLU|
|(64_,_7_,_28)|Flatten|||
|(12544)|Linear|dim=128, bias=True|ReLU|
|(128)|Linear|dim=64, bias=True|ReLU|
|(64)|Linear|dim=2, bias=True||



## Table 21: `CBM` and `DPL` architecture for `MNLogic` 

|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|---|---|---|---|
|(1_,_28_,_28)|Convolution|depth=32, kernel=3, padding=1, stride=1|ReLU|
|(32_,_28_,_28)|MaxPool2d|kernel=2, stride=2||
|(32_,_14_,_14)|Convolution|depth=64, kernel=3, padding=1, stride=1|ReLU|
|(64_,_14_,_14)|MaxPool2d|kernel=2, stride=2||
|(64_,_7_,_7)|Flatten|||
|(3136)|Linear|dim=128, bias=True|ReLU|
|(128)|Linear|dim=2, bias=True||



30 

Table 22: `NN` architecture for `MNMath` 

|T|able 22: `NN`a|rchitecture for`MNMath`||
|---|---|---|---|
|INPUT|LAYERTYPE|PARAMETER|ACTIVATION|
|(1_,_28_,_224)|Convolution|depth=16, kernel=3, padding=1|ReLU|
|(16_,_28_,_224)|MaxPool2d|kernel=2, stride=2||
|(16_,_14_,_112)|Convolution|depth=32, kernel=3, padding=1|ReLU|
|(32_,_14_,_112)|MaxPool2d|kernel=2, stride=2||
|(32_,_7_,_56)|Convolution|depth=64, kernel=3, padding=1|ReLU|
|(64_,_7_,_56)|Flatten|||
|(25088)|Linear|dim=128, bias=True|ReLU|
|(128)|Linear|dim=64, bias=True|ReLU|
|(64)|Linear|dim=2, bias=True||



31 

## **C Code, Data Sets and Generators** 

In the following, we discuss: 1) code and data licensing Appendix C.1, 2) how the data was collected and organised Appendix C.4, 3) what kind of information it contains Appendix C.5, 4) how it should be used ethically and responsibly Appendix C.2, 5) how it will be made available and maintained Appendix C.3. All data, generators, metadata, and experimental code for reproducing the results are available at: `https://unitn-sml.github.io/rsbench` . 

Detailed statistics for each data set using the default configuration are reported in Table 23. 

Table 23: **Detailed statistics about the** _**default**_ **data sets in** `rsbench` . For generators, the number of concepts _k_ is configurable; in `CLE4EVR` , _n_ and _m_ are the minimum and maximum number of objects. 

|TASK|INFO**x**|INFO**c**|INFO**y**|TRAIN|VAL|TEST|OOD|
|---|---|---|---|---|---|---|---|
|`MNMath`|28_k ×_28|_k_digits, 10 values each|cat multilabel|custom|custom|custom|custom|
|`MNAdd-Half`|56_×_28|2 digits, 10 values each|cat (19 values)|2_,_940|840|420|1_,_080|
|`MNAdd-EvenOdd`|56_×_28|2 digits, 10 values each|cat (19 values)|6_,_720|1_,_920|960|5_,_040|
|`MNLogic`|28_k ×_28|_k_digits, 2 values each|binary|custom|custom|custom|custom|
|||3 objects per image||||||
|`Kand-Logic`|3_×_192_×_64|3 shapes|binary|4_,_000|1_,_000|1_,_000|–|
|||3 colors||||||
|||_n_to_m_objects per image||||||
|||10 shapes||||||
|`CLE4EVR`|320_×_240|10 colors|binary|custom|custom|custom|custom|
|||2 materials||||||
|||3 sizes||||||
|`BDD-OIA`|1280_×_720|21 binary concepts|bin multilabel, 4 labels|16_,_082|2_,_270|4_,_572|–|
|`SDD-OIA`|469_×_387|21 binary concepts|bin multilabel, 4 labels|6_,_820|1_,_464|1_,_464|1_,_000|



## **C.1 Licensing** 

**Code** . Most of our code is available under the `BSD 3-Clause` license. The `CLE4EVR` and `SDD-OIA` generators are derived from the `CLEVR` code base, which is available under the `BSD` license. The `Kand-Logic` generator is derived from the `Kandinsky-patterns` code base, which is available under the `GPL-3.0` license, and so is our generator. 

**Data** . `MNMath` , `MNAdd-Half` , `MNAdd-EvenOdd` and `MNLogic` are derived from `MNIST` [61], which is distributed under `CC-BY-SA 3.0` , and so are our data sets and generated data. `BDD-OIA` is derived from `BDD-100k` [104], which is distributed under a `BSD 3-Clause` license, and so is our data set. Data sets and generated data for `Kand-Logic` and `SDD-OIA` are available under a `CC-BY-SA 4.0` license. 

## **C.2 Ethical Statement** 

`rsbench` is a collection of datasets aimed at exploring challenges related to concept quality, particularly focusing on identifying reasoning shortcuts. It also includes a formal verification tool to assess how often these shortcuts occur in specific configurations. Essentially, `rsbench` aims to help investigating concept quality in neural, neuro-symbolic and foundation models. Although this is not its intended purpose, such a benchmark may inadvertently used to improve models designed for harmful applications. However, to our knowledge, our work does not directly threaten individuals or society. Additionally, since most datasets are synthetically generated, they do not cause harm during creation. `BDD-OIA` , just like `BDD-100k` , could in principle be used to train models that aim to cause harm. We expressly disapprove of this usage. 

## **C.3 Hosting and Maintenance Plan** 

The data is openly available on Zenodo at `https://zenodo.org/doi/10.5281/zenodo. 11612555` . The data set generators are freely available on Github. The repository is linked in our website: `https://github.com/unitn-sml/rsbench` . 

32 

## **C.4 Data Collection** 

`rsbench` makes uses of two pre-existing data collections, namely `MNIST` and `BDD-OIA` . In this section, we briefly describe this data and how it is collected. 

`MNIST` : The `MNIST` [61] dataset is a well known collection of handwritten digits, consisting of 60 _,_ 000 training images and 10 _,_ 000 test images. Each image is a 28 _×_ 28 grayscale image of a numerical digit ranging from 0 to 9. The dataset was created by Yann LeCun, Corinna Cortes and Christopher J.C. Burges. `MNAdd-EvenOdd` and `MNAdd-Half` build on the `MNIST` dataset [20, 25]. `MNLogic` and `MNMath` , two datasets that can be generated from `rsbench` , make use of `MNIST` images. 

`BDD-OIA` : `BDD-OIA` [19] is a dataset based on `BDD-100K` [104] dataset. `BDD-100K` is a large collection consisting of driving video data, developed by researchers at the University of California, Berkeley. The dataset is suitable for multitask learning, ranging from object detection to semantic segmentation and object tracking. It contains 100 _,_ 000 videos and images, collected under diverse driving conditions, times of day, and geographic locations. The data is annotated with labels including bounding boxes, lane marking, and drivable area segmentation. For further information, please refer to the original paper [104]. 

## **C.5 Data Generators** 

Each `rsbench` data generator comprises two `Python` components: the _generator_ proper samples new data, and the associated _parser_ reads the configuration from a `YAML` file. The latter also validates the configuration, _i.e._ , check for required fields and ensure the logical formulas work as intended. Users can also configure the generators through the command line. Generated images are stored in `PNG` format, and ground-truth annotations as `JOBLIB` metadata. 

**Shared configuration options** . All generators support a set of basic command line settings: `config` : path to the `YAML` configuration file; `output_dir` : path to the output directory; `n_samples` : number of samples to be generated; `log_level` : verbosity level; `seed` : RNG seed, for reproducibility; 

They all comply with the following `YAML` settings: `symbols` : names of the logic symbols (concepts) that appear in the knowledge; the order is managed internally by `rsbench` ; `logic` : formal specification of the knowledge as a `sympy` formula, used for computing the ground truth labels; `prop_in_distribution` : proportion of examples to put in the in-distribution sets (train, validation, and test), up to 100%; `combinations_in_distribution` : what combinations of concept values should be included in the in-distribution sets. `val_prop` : proportion of examples to put in the validation set; `test_prop` : proportion of examples to put in the test set; 

**Non-Blender generators** : `MNMath` , `MNLogic` , and `Kand-Logic` . The generator first parses the `YAML` configuration file, then proceeds to randomly sample the required number of examples. It generates a series of label and concept assignments that comply with the combinations combinations specified by the config file, if any. The ground-truth label is computed using the knowledge K. For `MNMath` , which is multi-class and multi-label, this involves splitting the configurations between classes or random sampling. Before the generation of the dataset, `rsbench` automatically checks whether the sampled configurations produce labels that are either all false or all true, and returns an error to the user if such a condition is found. 

If the `prop_in_distribution` flag is set, the specified ratio is assigned to the in-distribution datasets (training, validation, and test), while the remaining settings are allocated to the out-of-distribution datasets. An equal number of examples are then assigned to both positive and negative configurations chosen for training, testing, and validation. This is achieved by sampling configurations alternately from positive and negative sides, with replacement. Depending on the dataset, examples are generated, and information such as labels and concepts are stored as `JOBLIB` metadata. 

Finally, `rsbench` provides the option to specify a compression type ( _e.g._ , zip) for storing the dataset, ensuring efficient storage and easy distribution. 

**Blender-based generators** . Generating 3D images involves running scripts from within Blender, which requires a different setup. These scripts read all configuration from the command line and specified configuration files. Options include the positions of shapes ( `shape_dir` ) and materials ( `material_dir` ), the output directories ( `output_image_dir` for the examples and `output_scene_dir` for metadata), the image resolution ( `width` , `height` ), and details bout the ren- 

33 

dering step (like `render_tile_size` , `render_num_samples` , `camera_jitter` , `light_jitter` ). The rendering engine used for `CLE4EVR` is `CYCLES` , while `SDD-OIA` uses the `EEVEE` rendering engine to speed up rendering, although this can be easily changed by the user. 

The generators build on the implementation of [63]. The images are stored as `PNG` s, while the metadata, in `JSON` format, contains information about concepts, ground truth labels, object bounding boxes, object positions, and relationships between objects ( _e.g._ , that one object is behind another). Unlike the synthetic data generation case, these scripts currently do not offer an option to compress the dataset, though this is a future contribution under consideration. 

## **C.6 Example of use** 

`rsbench` provides functionality for loading, training, and evaluating both the data and models discussed in this paper. This ready-to-use toolkit is available at `https://github.com/unitn-sml/ rsbench-code/tree/main/rsseval` . Alternatively, the data from `rsbench` can be loaded with minimal code, as demonstrated in the following example: 

**Listing 1** : Code snippet showcasing the training of a neural network on `MNLogic` using the default configuration. 

1 `from rss.datasets.xor import MNLOGIC` 2 3 `class required_args:` 4 `def __init__( self ):` 5 `self .c_sup = 0` _`# specifies % supervision available on concepts`_ 6 `self .which_c = -1` _`# specifies which concepts to supervise , -1= all`_ 7 `self .batch_size = 64` _`# batch size of the loaders`_ 8 9 `args = required_args ()` 10 11 `dataset = MNLOGIC(args)` 12 `train_loader , val_loader , test_loader = dataset.get_loaders ()` 13 14 `model =` _`#define your model here`_ 15 `optimizer =` _`#define optimizer here`_ 16 `criterion =` _`#define loss function here`_ 17 18 `for epoch in range (30):` 19 `for images , labels , concepts in train_loader:` 20 `optimizer.zero_grad ()` 21 `outputs = model(images)` 22 `loss = criterion(outputs , labels , concepts)` 23 `loss.backward ()` 24 `optimizer.step ()` 

Appendix C.6 illustrates a typical procedure for training a neural network on `MNLogic` , following standard PyTorch practices and default configurations. 

Additionally, various models and datasets can be employed by providing a script with the appropriate arguments, as shown below: 

`python main.py --dataset mnmathdpl --model mnmathnn --n_epochs 5 --lr 0.001 --seed 8 --batch_size =64 --exp_decay =1 --c_sup 0 --task mnmath` Customization of the data and splits is supported, allowing users to explore different experimental settings and corner cases. This customization involves modifying a short JSON or YAML file. Further details and examples can be found in Appendix C. 

For formal verification of RSs, `rsbench` offers a dedicated code base available at `https://github. com/unitn-sml/rsbench-code/tree/main/rsscount` . 

34 

To generate a DIMACS encoding for counting tasks, use the command: 

```
pythongen -rss -count.py
```

This script supports computing the exact number of RSs for smaller tasks (e.g., XOR with 3 variables) by specifying the `-e` and partial supervision can be specifyied with the `-d` flag. Additionally, random CNFs and custom tasks in DIMACS format are supported. For help with arguments, the `-h` flag is available. 

Once the problem encoding is generated, RS counts can be approximated with pyapproxmc using: 

```
pythoncount -amc.pyPATH--epsilonE--deltaD
```

Exact solvers, such as `pyeda` and `pysdd` , can also be employed. 

## **C.7** `MNMath` **Data Generator** 

Additional `YAML` config for `MNMath` are the number of digits per image ( `num_digits` ) and the subset of candidate digits ( `digit_values` ). The code expects `num_digits` names for `symbols` : the first one is assigned to the first digit, the second symbol to the second digit, and so on. With `logic` , the user can provide the system of equations. With `combinations_in_distribution` , the user can have fine-grained control over the in-distribution data ( _e.g._ , specifying "0234" means that the in-distribution data contains ). 

**Table 24:** Example of `MNMath` data 

|**YAML confg**|**JOBLIB metadata**|**JOBLIB metadata**|**PNG data**|
|---|---|---|---|
||`{`|||
|`num_digits: 2`|`’label’: [6, 7],`|||
|`symbols:`|`’meta’: `|`{`||
|`- a`|`’concepts’: [`|||
|`- b`||`[2, 2],`||
|`logic:`||`[3, 4]`||
|`- 2*a + b`|`]`|||
|`- a + b`|`}`|||
||`}`|||



## **C.8** `MNLogic` **Data Generator** 

The `YAML` file allows to specify the number of Boolean variables in the formula, as well as the formula itself. The knowledge defaults to the _k_ -bit `XOR` . `rsbench` includes a script for generating random _ℓ_ -CNF formulas, which can be readily used with `MNLogic` by setting `xor_rule` to false and `logic` to the target formula. If `use_mnist` is set, the input images are of size ( _k ·_ 28) _×_ 28 and obtained by concatenating _k_ `MNIST` digits, one per bit. Otherwise, the code defaults to the setup of [20], where the inputs are encoded as _k ×_ 1 black-and-white images, one pixel per bit. 

You can filter what types of data appear in-distribution with `combinations_in_distribution` ( _e.g._ , specifying 0101 means the in-distribution data contains ). 

## **C.9** `Kand-Logic` **Data Generator** 

The `YAML` file allows specifying: `n_shapes` , the number of primitives per figure; `n_figures` : the number of figures per input image; `colors` , a subset of _{_ `red` _,_ `yellow` _,_ `blue` _}_ ; `shapes` : a subset of _{_ `square` _,_ `circle` _,_ `triangle` _}_ . The first two `symbols` are associated to the first primitive in the first image, and refer to its shape and color, respectively; the next two to the second primitive, and so on for all primitives and figures in the input. `logic` applies to each individual figure. The 

35 

**Table 25:** Example of `MNLogic` data 

|**YAML confg**|**JOBLIB metadata**|**PNG data**|
|---|---|---|
|`n_digits: 3`<br>`xor_rule: False`<br>`symbols:`<br>`- a`<br>`- b`<br>`- c`<br>`logic:`<br>`Or(And(a, b), Not(c))`<br>`use_mnist: True`|`{`<br>`’label’: True,`<br>`’meta’: {`<br>`’concepts’: [`<br>`True,`<br>`False,`<br>`False`<br>`]`<br>`}`<br>`}`||



ground-truth label of an image (consisting of multiple figures) is specified by `aggregator_symbols` and `aggregator_logic` . These give names to the variables holding the truth value for each figure, and how these values are aggregated to yield the ground-truth label, respectively. 

The user can specify which data combination to generate in-distribution by setting `combinations_in_distribution` ( _e.g._ , specifying _•_ "red, square" _•_ "blue, square" _•_ "blue, square" means the in-distribution data contains an image made of a red square and two blue squares). 

**Table 26:** Example of `Kand-Logic` data 

|**YAML confg**|**JOBLIB metadata**|**JOBLIB metadata**|**PNG data**|
|---|---|---|---|
|`colors:`||||
|`- red`||||
|`- yellow`||||
|`- blue`||||
|`shapes:`||||
|`- circle`|`{`|||
|`- square`|`’label’: True,`|||
|`- triangle`|`’meta’: `|`{`||
|`symbols:`|`’concepts’: [`|||
|`- shape_1`||`[6, 2,`||
|`- color_1`||`5, 1,`||
|`...`||`6, 2],`||
|`- shape_3`||||
|`- color_3`||`[6, 1,`||
|`logic:`||`5, 2,`||
|`(Eq(color_1, color_2) &`||`6, 1],`||
|`Eq(shape_1, shape_2) &`||||
|`Ne(shape_1, shape_3)) |`||`[5, 2,`||
|`... )`||`5, 2,`||
|`# two equal one diff`||`4, 1]`||
|`aggregator_symbols:`|`]`|||
|`- pattern_1`|`}`|||
|`- pattern_2`|`}`|||
|`- pattern_3`||||
|`aggregator_logic:`||||
|`pattern_1 &`||||
|`pattern_2 &`||||
|`pattern_3`||||



36 

## **C.10** `CLE4EVR` **Data Generator** 

The data generation process for `CLE4EVR` closely resembles that of previous datasets. To generate the datasets, the program samples various configurations, specifically the number of objects, shapes, colors, and sizes. These configurations are then divided into positive and negative sets based on the whether they satisfy the knowledge `logic` . The sets are used to generate images while maintaining a balanced ratio of positive and negative ground-truth samples. 

> `rsbench` allows users to customize various aspects of data generation, including the number of objects, whether occlusion is permitted, and the dimensions of the image. The occlusion check, which uses Blender rendering, can be slow for many objects due to rejection sampling. 

`rsbench` by default includes two materials (rubber and metal), nine shapes, and eight predefined colors, with options to create custom blend files and specify RGB values. Default object sizes are large, medium, and small, but users can fully customize these settings in a configuration file. 

The `symbols` for each object, are be defined in the following the order: color, shape, material, and size. 

**Table 27:** Example of `CLE4EVR` data 

|**YAML confg**|**JSON metadata**|**PNG data**|
|---|---|---|
|`symbols:`<br>`- color_1`<br>`- shape_1`<br>`- mat_1`<br>`- size_1`<br>`- color_2`<br>`- shape_2`<br>`- mat_2`<br>`- size_2`<br>`logic: |`<br>`And(`<br>`Eq(color_1, color_2),`<br>`Eq(shape_1, shape_2),`<br>`Eq(mat_1, mat_2),`<br>`Eq(size_1, size_2)`<br>`)`|`{`<br>`"label": 0,`<br>`"concepts": [`<br>`[`<br>`[`<br>`0,`<br>`1,`<br>`0,`<br>`0,`<br>`0,`<br>`0,`<br>`0,`<br>`0`<br>`],`<br>`}`||



## **C.11** `SDD-OIA` **Data Generator** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts_images/nips_2024_A_Benchmark_Suite_for_Systematically_Evaluating_Reasoning_Shortcuts.pdf-0037-08.png)


**----- Start of picture text -----**<br>
( i ) Sample label ( ii ) Sample concepts ( iii ) Sample objects ( iv ) Render image<br>y c [∗] c F G x<br>**----- End of picture text -----**<br>


Figure 4: Illustration of the sampling process of `SDD-OIA` 

Regarding `SDD-OIA` , `rsbench` allows users to specify parameters such as the number of samples, number of configurations to be generated, and image size. 

For `SDD-OIA` , the data generation approach differs from other datasets in `rsbench` and follows a Bayesian network [105]. The process involves first ( _i_ ) sampling the actions **y** from _p_ ( **y** ), ensuring that the overall dataset is balanced in the labels, _i.e._ , _p_ ( **y** ) is the uniform distribution. ( _ii_ ) Second, we sample the ground-truth concepts **c** _[∗]_ from the conditional _p_ ( **c** _[∗] |_ **y** ). Then, ( _iii_ ) the concepts **c** _[∗]_ specify a fine-grained distribution of objects in the scene, denoted as **c** _F G_ , which are sampled through _p_ ( **c** _F G|_ **c** _[∗]_ ). Next, the fine-grained objects are used to generate the scene. This step is 

37 

deterministic and yields the final image **x** . The crossroads scene is essentially a grid where objects’ positions are specified by the fine-grained variables **c** _F C_ . This ensures the concepts **c** _[∗]_ are visible from the car’s camera. The scene is then rendered with blender. The process is shown in Fig. 4. All steps in the sampling procedure ensure that all concepts can be retrieved from the image (respecting assumption **A1** in Appendix A.3) and that labels can be predicted uniquely from concepts **c** _[∗]_ (respecting assumption **A2** in Appendix A.3). 

A key aspect of `SDD-OIA` is its customizable data generation process, which involves sampling the concepts and constructing the scene. This necessitates a hard-coded compositional framework to correctly position the camera and objects, ensuring visibility from the car’s perspective. This approach enables the creation of a high-quality synthetic neuro-symbolic dataset, where objects, sample quantities, and distribution ratios are fully customizable. Like other datasets, `SDD-OIA` maintains a balanced distribution across all actions. Users can configure model selection, object dimensions, and the probabilities for sampling different objects by adjusting the categorical distribution weights or the hard-coded matrix configuration. 

**Table 28:** Example of `SDD-OIA` data 

|**JSON metadata**|**PNG data**|
|---|---|
|`{`||
|`"label": [`||
|`0,`||
|`1,`||
|`0,`||
|`1`||
|`],`||
|`"concepts": {`||
|`"red_light": false,`||
|`"green_light": true,`||
|`"car": false,`||
|`"person": false,`||
|`"rider": false,`||
|`"other_obstacle": false,`||
|`"follow": false,`||
|`"stop_sign": false,`||
|`"left_lane": false,`||
|`"left_green_light": true,`||
|`"left_follow": false,`||
|`"no_left_lane": true,`||
|`"left_obstacle": false,`||
|`"left_solid_line": false,`||
|`"right_lane": true,`||
|`"right_green_light": true,`||
|`"right_follow": true,`||
|`"no_right_lane": false,`||
|`"right_obstacle": false,`||
|`"right_solid_line": false,`||
|`"clear": true`||
|`}`||



```
}
```

## **C.11.1 Assets used in** `SDD-OIA` 

All assets are made available under permissive licenses that allow reuse for non-commercial purposes. 

- Author: stunts. Speed Limit Signs [3D model]. Retrieved from `https://free3d.com/ 3d-model/speed-limit-signs-172903.html` ; 

38 

- Author: corrobrocz. Concrete street barrier [3D model]. Retrieved from `https://free3d.com/ 3d-model/concrete-street-barrier-917223.html` ; 

- Author: paulsendesign. Cartoon low poly trees [3D model]. Retrieved from `https://free3d. com/3d-model/cartoon-low-poly-trees-895299.html` ; 

- Author: roxas. Low Poly Car [3D model]. Retrieved from `https://free3d.com/3d-model/ low-poly-car-14842.html` ; 

- Author: RokoTheAwesome. Traffic Light [3D model]. Retrieved from `https://www. turbosquid.com/3d-models/traffic-light-547022` 

All the models from `free3d` are under the Personal Use License, meaning the models are available for free but only for personal or non-commercial use. In contrast, the models from `TurboSquid` are under the Standard 3D Model License, which permits the use of `TurboSquid` models in various commercial projects, such as games and movies. This license allows the creation and distribution of your end-products without reproduction limitations to any target market or audience indefinitely. However, the license prohibits making the models themselves directly available to end-users, so `rsbench` redirects to the asset URL. 

## **C.12** `BDD-OIA` **Data** 

Data for `BDD-OIA` are those previously published in [19]. `BDD-OIA` images are selected from `BDD` -100k only including franes with complicated scenes where multiple actions _{_ `forward` _,_ `stop` _,_ `left` _,_ `right` _}_ are possible. This includes situations with multiple objects present. Following [19], all images are manually annotated for ground-truth actions and 21 associated binary concepts. The dataset contains 16k frames for training, (with annotated labels and concepts); 2k frames for validation, and 4 _._ 5k frames for testing. The table below reports the overall proportion of labels and concepts. 

Concept classes in `BDD-OIA` 

|Conce|pt classes in`BDD-OIA`|
|---|---|
|**Action Category**|**Concepts**<br>**Count**|
|`move_forward`|`green_light`<br>7805<br>`follow`<br>3489<br>`road_clear`<br>4838|
|`stop`|`red_light`<br>5381<br>`traffic_sign`<br>1539<br>`car`<br>233<br>`person`<br>163<br>`rider`<br>5255<br>`other_obstacle`<br>455|
|`turn_left`|`left_lane`<br>154<br>`left_green_light`<br>885<br>`left_follow`<br>365|
||`no_left_lane`<br>150<br>`left_obstacle`<br>666<br>`letf_solid_line`<br>316|
|`turn_right`|`right_lane`<br>6081<br>`right_green_light`<br>4022<br>`right_follow`<br>2161|
||`no_right_lane`<br>4503<br>`right_obstacle`<br>4514<br>`right_solid_line`<br>3660|



39 

## **D Additional Results** 

Here, we report additional tables for `TCAV` evaluation complementing the results reported in the main text. All results indicate that `TCAV` at different layers always attain low _F_ 1-scores. We also report the Cls( _C_ ) and mAcc _C_ . 

Table 29: Concept metrics for each `NN` layer using TCAV on `MNAdd-EvenOdd` 

|Concep|t metrics for|each`NN`laye|r using TCA|V on`MNAdd-`|
|---|---|---|---|---|
||LAYERNUM|Acc_C_|F1(_C_)|Cls(_C_)|
|_conv_1|1|0_._11_±_0_._03|0_._10_±_0_._03|0_._00_±_0_._00|
|_conv_2|2|0_._12_±_0_._03|0_._10_±_0_._04|0_._01_±_0_._02|
|_fc_1|3|0_._12_±_0_._04|0_._09_±_0_._05|0_._24_±_0_._30|
|_fc_2|4|0_._11_±_0_._02|0_._07_±_0_._03|0_._29_±_0_._34|



Table 30: Concept metrics for each `NN` layer using TCAV on `Kand-Logic` 

|0: Conce|pt metrics f|or each`NN`la|yer using TC|AV on`Kand`|
|---|---|---|---|---|
||LAYERNUM|Acc_C_|F1(_C_)|Cls(_C_)|
|_conv_1|1|0_._35_±_0_._01|0_._34_±_0_._01|0_._00_±_0_._01|
|_conv_2|2|0_._35_±_0_._01|0_._34_±_0_._01|0_._00_±_0_._01|
|_conv_3|3|0_._34_±_0_._01|0_._34_±_0_._01|0_._00_±_0_._01|
|_conv_4|4|0_._35_±_0_._01|0_._34_±_0_._01|0_._00_±_0_._01|
|_conv_5|5|0_._35_±_0_._01|0_._34_±_0_._01|0_._00_±_0_._01|
|_fc_1|6|0_._33_±_0_._01|0_._32_±_0_._01|0_._00_±_0_._01|
|_fc_2|7|0_._33_±_0_._01|0_._31_±_0_._01|0_._00_±_0_._01|



Table 31: Concept metrics for each `NN` layer using TCAV on `SDD-OIA` 

|31: Con|cept metrics|for each`NN`|layer using T|CAV on`SDD`|
|---|---|---|---|---|
||LAYERNUM|mAcc_C_|mF1(_C_)|Cls(_C_)|
|_conv_1|1|0_._48_±_0_._02|0_._44_±_0_._01|0_._19_±_0_._05|
|_conv_2|2|0_._49_±_0_._02|0_._45_±_0_._02|0_._20_±_0_._06|
|_conv_3|3|0_._49_±_0_._03|0_._45_±_0_._03|0_._21_±_0_._09|
|_conv_4|4|0_._48_±_0_._02|0_._44_±_0_._01|0_._23_±_0_._15|
|_conv_5|5|0_._48_±_0_._02|0_._44_±_0_._02|0_._30_±_0_._26|
|_conv_6|6|0_._46_±_0_._02|0_._43_±_0_._02|0_._34_±_0_._33|
|_fc_1|7|0_._50_±_0_._02|0_._45_±_0_._03|0_._38_±_0_._31|
|_fc_2|8|0_._49_±_0_._02|0_._44_±_0_._02|0_._43_±_0_._28|



Table 32: Concept metrics for each `NN` layer using TCAV on `SDD-OIA` with synthetic images. 

||LAYERNUM|mAcc_C_|mF1(_C_)|Cls(_C_)|
|---|---|---|---|---|
|_conv_1|1|0_._47_±_0_._02|0_._43_±_0_._02|0_._18_±_0_._03|
|_conv_2|2|0_._48_±_0_._02|0_._44_±_0_._02|0_._18_±_0_._03|
|_conv_3|3|0_._49_±_0_._01|0_._45_±_0_._01|0_._23_±_0_._12|
|_conv_4|4|0_._48_±_0_._03|0_._44_±_0_._03|0_._23_±_0_._14|
|_conv_5|5|0_._48_±_0_._02|0_._44_±_0_._02|0_._29_±_0_._25|
|_conv_6|6|0_._48_±_0_._04|0_._45_±_0_._04|0_._34_±_0_._32|
|_fc_1|7|0_._51_±_0_._03|0_._45_±_0_._03|0_._38_±_0_._31|
|_fc_2|8|0_._74_±_0_._01|0_._42_±_0_._01|0_._99_±_0_._01|



40 

## **E Dataset Documentation: Datasheets for Datasets** 

Here, we answer the questions posed in the datasheets for datasets paper by Gebru et al [106]. 

## **E.1 Motivation** 

**For what purpose was the dataset created?** `rsbench` was created to study the phenomenon of reasoning shortcuts (RSs) and concept quality in neuro-symbolic and neural architectures. `rsbench` offers several datasets where RSs occur, as well as a formal verification tool that enables users to verify how many RSs appear in the desired settings. 

**Who created the dataset (** _**e.g.**_ **, which team, research group) and on behalf of which entity (** _**e.g.**_ **, company, institution, organisation)?** The datasets have been created by the “Structured Machine Learning” research group at the department of Information Engineering and Computer Science of the University of Trento in collaboration with the april Lab at School of Informatics, University of Edinburgh. 

**Who funded the creation of the dataset?** The datasets have been created for research purposes. Funded by the European Union. The views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union, the European Health and Digital Executive Agency (HaDEA) or the European Research Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. Grant Agreement no. 101120763 - TANGO. PM is supported by the MSCA project GA n°101110960 “Probabilistic Formal Verification for Provably Trustworthy AI - PFV-4-PTAI”. AV is supported by the “UNREAL: Unified Reasoning Layer for Trustworthy ML” project (EP/Y023838/1) selected by the ERC and funded by UKRI EPSRC. Emile van Krieken was funded by ELIAI (The Edinburgh Laboratory for Integrated Artificial Intelligence), EPSRC (grant no. EP/W002876/1). 

## **E.2 Composition** 

**What do the instances that comprise the dataset represent (** _**e.g.**_ **, documents, photos, people, countries)?** All datasets contain annotations regarding concepts and labels. `SDD-OIA` comprises synthetically generated images depicting autonomous driving scenarios, such that if they were captured from a car’s dashcam, and includes additional information about the scene structure, such as bounding boxes, 2D and 3D coordinates, and spatial relationships among objects. `MNMath` , `MNAdd-Half` , `MNAdd-EvenOdd` and `MNLogic` contain synthetic images of handwritten digits, derived from the `MNIST` dataset. `Kand-Logic` consists of synthetic data showcasing patterns of geometric shapes with various colors. `CLE4EVR` features synthetically generated images representing 3D objects of different shapes, colors, materials, and dimensions; similar to `SDD-OIA` , they include additional scene information. `BDD-OIA` is a real-world, high-stakes dataset comprising images captured from a car’s dashcam. For a comprehensive description, please refer to [19]. 

**How many instances are there in total (of each type, if appropriate)?** Please refer to Table 23. 

**Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** The datasets represent samples from configurations that can be randomly generated according to a grammar. Using the generators, one can filter through various combinations and determine the level of exhaustiveness for generating examples. For a comprehensive overview of each dataset generation process, please consult Appendix C.5 and subsequent sections. 

**What data does each instance consist of?** Alongside the images, each dataset sample is annotated with concepts and labels. However, for `SDD-OIA` and `CLE4EVR` , detailed scene information is included, encompassing individual 2D and 3D coordinates, bounding boxes, and spatial relationships between objects. For an complete overview refer to Table 23. 

**Is there a label or target associated with each instance?** Yes, the concept annotations are derived from the data generation process, while the labels are symbolically derived from the knowledge provided to the dataset. 

41 

**Is any information missing from individual instances?** No. 

**Are relationships between individual instances made explicit (** _**e.g.**_ **, users’ movie ratings, social network links)?** No, there are no connections between different instances. 

**Are there recommended data splits (** _**e.g.**_ **, training, development/validation, testing)?** Information about the data splits we employed is reported in Appendix B. The user has the freedom to choose the data splits they prefer during the data generation process. 

**Are there any errors, sources of noise, or redundancies in the dataset?** No. 

**Is the dataset self-contained, or does it link to or otherwise rely on external resources (** _**e.g.**_ **, websites, tweets, other datasets)?** Some of our data sets build on top of established and stable data, namely `MNIST` and (the last frames provided by) `BDD-100k` , for which we provide download links. `SDD-OIA` makes use of external assets, listed in Appendix C.11.1. The ready-made `SDD-OIA` data set does not require these assets, but in order to use the generator these have to be obtained separately. 

**Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?** No. 

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** No. 

**Does the dataset relate to people? If not, you may skip the remaining questions in this section.** `BDD-OIA` contains images depicting pedestrians and bicycle riders. Identifiable information in these images, including anonymization, rights, and risks, is managed by the original `BDD-100k` authors. 

**Does the dataset identify any subpopulations (** _**e.g.**_ **, by age, gender)?** Please refer to E.2. 

**Is it possible to identify individuals (** _**i.e.**_ **, one or more natural persons), either directly or indirectly (** _**i.e.**_ **, in combination with other data) from the dataset?** Please refer to E.2. 

**Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** Please refer to E.2. 

## **E.3 Collection Process** 

**How was the data associated with each instance acquired?** `MNIST` and `BDD-100k` have been obtained from their official repositories, `http://yann.lecun.com/exdb/mnist/` and `https: //dl.cv.ethz.ch/bdd100k/data/` , respectively. All other data is synthetically generated. 

**What mechanisms or procedures were used to collect the data (** _**e.g.**_ **, hardware apparatus or sensor, manual human curation, software program, software API)?** Details about data generations and software programs are discussed in Appendix B. 

**If the dataset is a sample from a larger set, what was the sampling strategy (** _**e.g.**_ **, deterministic, probabilistic with specific sampling probabilities)?** Please refer to the similar question in Appendix E.2. 

**Who was involved in the data collection process (** _**e.g.**_ **, students, crowdworkers, contractors) and how were they compensated (** _**e.g.**_ **, how much were crowdworkers paid)?** The authors were involved in the process of generating these datasets. 

42 

**Over what timeframe was the data collected?** The datasets were generated over a span of several days. 

**Were any ethical review processes conducted (** _**e.g.**_ **, by an institutional review board)?** No. 

**Does the dataset relate to people? If not, you may skip the remainder of the questions in this section.** `BDD-OIA` is the only dataset relating to people, please refer to Appendix E.2. 

## **E.4 Preprocessing/Cleaning/Labeling** 

**Was any preprocessing/cleaning/labeling of the data done (** _**e.g.**_ **, discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** No, the datasets were generated along with labels and concept annotations. 

**Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (** _**e.g.**_ **, to support unanticipated future uses)?** NA 

**Is the software used to preprocess/clean/label the instances available?** NA 

## **E.5 Uses** 

**Has the dataset been used for any tasks already?** In the paper, we demonstrate and benchmark the intended use of these datasets for evaluating concept quality and exploring RSs. `MNAdd-EvenOdd` , `MNAdd-Half` , and `CLE4EVR` have been utilized in previous studies [25, 20, 17] to investigate RSs and concept quality. 

**Is there a repository that links to any or all papers or systems that use the dataset?** Yes, `https://unitn-sml.github.io/rsbench/` . 

**What (other) tasks could the dataset be used for?** `SDD-OIA` and `CLE4EVR` offer additional information regarding the scene, including the 3D and 2D coordinates of objects, their bounding boxes, and the relationships between objects within the scene. This spatial data enables various applications such as object discovery, object detection, and reasoning over the scene’s structure. 

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses** No. 

**Are there tasks for which the dataset should not be used?** These datasets are meant for research purposes only. 

## **E.6 Distribution** 

**Will the dataset be distributed to third parties outside of the entity (** _**e.g.**_ **, company, institution, organization) on behalf of which the dataset was created?** No. 

**How will the dataset will be distributed (** _**e.g.**_ **, tarball on website, API, GitHub)?** The datasets, data generators, and related evaluation code are available on the website, enabling users to generate, download, and test their model on the data. Each dataset is provided in `zip` format and can be downloaded from the Zenodo link on the website. 

**When will the dataset be distributed?** The datasets employed in the paper are available now on the website. 

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** Please refer to Appendix C.1. 

43 

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** `SDD-OIA` makes use of assets taken from `https://free3d.com` and `https://www. turbosquid.com` . See Appendix C.11.1 for the full list and associated licenses. Other instances of datasets themselves do not have IP-based restrictions. 

**Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** Not that we are are of. 

## **E.7 Maintenance** 

**Who is supporting/hosting/maintaining the dataset?** The datasets are supported by the authors and will be actively maintained by the “Structured Machine Learning” research group in the future. For the hosting and maintenance plan, please refer to Appendix C.3. 

**How can the owner/curator/manager of the dataset be contacted (e.g., email address)?** The authors of `rsbench` can be contacted via their email addresses: samuele.bortolotti@unitn.it, emanuele.marconato@unitn.it. 

**Is there an erratum?** If errors are found, an erratum will be added to the website. 

**Will the dataset be updated (** _**e.g.**_ **, to correct labeling errors, add new instances, delete instances)?** Any potential future updates or extensions will be communicated via the website. The datasets will be versioned. 

**If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (** _**e.g.**_ **, were individuals in question told that their data would be retained for a fixed period of time and then deleted)?** The only dataset involving people is `BDD-OIA` , plase refer to Appendix E.2. 

**Will older versions of the dataset continue to be supported/hosted/maintained?** We plan to continue hosting older versions of the dataset. 

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** Yes, the dataset generation code is available on our website. 

## **E.8 Other Questions** 

**Is your dataset free of biases?** Our data sets are designed to induce a particular type of bias, namely reasoning shortcuts, in models, for the purpose of studying them. The data itself however is not biased towards human factors such as gender, ethnicity, age, etc. 

**Can you guarantee compliance to GDPR?** No, we are unable to comment on legal matters. 

## **E.9 Author Statement of Responsibility** 

The authors assume full responsibility for any rights violations and confirm the license associated with the datasets and their images. 

44 

