# **KiVA: Kid-inspired Visual Analogies for Testing Large Multimodal Models** 

**Eunice Yiu**[1] **Maan Qraitem**[2] **Charlie Wong**[1] **Anisa Noor Majhi**[1] **Yutong Bai**[1] **Shiry Ginosar**[3] **Alison Gopnik**[1] **Kate Saenko**[2] 1 University of California, Berkeley 2 Boston University 3 Google DeepMind `https://github.com/ey242/KiVA` 

## **Abstract** 

This paper investigates visual analogical reasoning in large multimodal models (LMMs) compared to human adults and children. A “visual analogy” is an abstract rule inferred from one image and applied to another. While benchmarks exist for testing visual reasoning in LMMs, they require advanced skills and omit basic visual analogies that even young children can make. Inspired by developmental psychology, we propose a new benchmark of 1,400 visual transformations of everyday objects to test LMMs on visual analogical reasoning and compare them to children and adults. We structure the evaluation into three stages: identifying _what_ changed (e.g., color, number, etc.), _how_ it changed (e.g., added one object), and _applying the rule_ to new scenarios. Our findings show that while models like GPT-4V, LLaVA-1.5, and MANTIS identify the “what” effectively, they struggle with quantifying the “how” and extrapolating this rule to new objects. In contrast, children and adults exhibit much stronger analogical reasoning at all three stages. Additionally, the strongest tested model, GPT-4V, performs better in tasks involving simple visual attributes like color and size, correlating with quicker human adult response times. Conversely, more complex tasks such as number, rotation, and reflection, which necessitate extensive cognitive processing and understanding of the 3D physical world, present more significant challenges. Altogether, these findings highlight the limitations of training models on data that primarily consists of 2D images and text. 

## **1 Introduction** 

What is visual cognition? Humans make countless visual inferences daily from observing objects and scenes and quickly notice visual changes. We generalize common hypotheses about changes from different observations and use these insights to solve new problems. If one puts a wool sweater in the washing machine and it comes out smaller, one might infer that the wash shrinks sweaters and avoid washing wool coat in the future. If cookies disappear, one might infer that someone is eating our treats and and proceed to hide the chocolate elsewhere. This ability to draw parallels between situations and apply learned patterns to a new scenario is known as _analogical reasoning_ . Analogical reasoning is a hallmark of human intelligence and learning [25, 33, 51, 72]. It is what enables us to be flexible, adaptive and robust learners across a wide variety of settings, making out-of-distribution generalizations and responding accordingly to meaningful patterns [14, 51]. Analogical reasoning is already available to young children [28, 29, 73], and is crucial for problem-solving in various contexts, from building scientific models to appreciating metaphors to formulating legal arguments. 

Today, we witness much progress in large multimodal models, but these models remain data-hungry and still require substantial human involvement to adapt to new contexts [14, 64]. As analogical reasoning is instrumental for general-purpose and adaptive machines, it is crucial to examine whether current models have such capabilities. Critically, examining analogical capabilities does not rely 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Reflection Rotation Size<br>Rotation<br>? ? ? Reflection Size<br>Number Color<br>? ? Number Color<br>% correct transformations selected (out of 100%)<br>(a) Visual analogy domains. (b) Extrapolation accuracy.<br>**----- End of picture text -----**<br>


Figure 1: **KiVA: Kid-inspired Visual Analogies. (a)** The different visual analogy domains examined in KiVA. In the top rows, we present example transformations. In the bottom rows, we ask the subject to extrapolate the identified transformations to a novel object by analogy. See Fig. 3 for the full task format. **(b)** Performance of children, adults & LMMs on extrapolating a transformation rule to a novel object. LMMs struggle with extrapolation in contrast to humans. Additionally, the strongest tested model, GPT-4V, performs better in tasks involving simple visual attributes like color and size than in those requiring an understanding of the 3D physical world, such as number, rotation, and reflection. 

on external prior knowledge, nor does it permit models to “cheat” by merely depending on their training data. Reasoning about analogies involves first recognizing _relationships_ between object characteristics and quantifying the similarities and differences, then recognizing the _same relationship_ in a new set of objects. In this paper, we focus on visual analogies, which evaluate the ability of models to engage in abstract reasoning based on visual observations. See Figure 1 for a summary of our visual analogical reasoning benchmark and results. 

There is a growing body of work examining visual analogical reasoning capabilities in large multimodal models [2, 34, 53, 58, 78]. Existing benchmarks of visual analogies include (a) ConceptARC [52, 53], (b) variations of Raven’s Progressive Matrices [34] and (c) abstract spatial reasoning [2] (see prior benchmarks in Figure 2). These prior benchmarks all have several critical limitations. First, they rely on abstract shapes and grids, lacking real-world relevance. This abstraction of stimuli neither aligns with the training data of large multimodal models nor effectively mimics the complexity and variability found in everyday visual tasks, making it less suitable for assessing how well AI models can perform analogical reasoning in practical contexts. Second, the transformations examined involve conjunctions of visual concepts such as counting _and_ ordering _and_ extracting pixels according to some arbitrary rule, which does not tap into basic visual cognition. Humans do not require the ability to solve these specific tasks to function effectively in their daily lives nor to demonstrate their capacity for visual analogical reasoning. Third, while we know that models often perform poorly on these benchmarks, where they fail in the reasoning process needs to be clarified since existing evaluations focus solely on prediction accuracy rather than the reasoning approach or what is perceived. 

We propose a Kid-inspired Visual Analogies (KiVA) benchmark founded on developmental psychology (See Figure 1(left)). We focus our analysis on basic visual analogical capabilities that are present early in human development and are important for understanding the physical world. In particular, our benchmark isolates the following fundamental capabilities: detecting changes in **color** [65, 76] and **size** [17, 76], changes that involve **rotation** and **reflection** [23, 62], and changes in small **numbers** of objects [13, 42]. Our benchmark stands out in the following ways: 

First, our dataset utilizes _real-world_ objects curated from established 3D object datasets [21, 74] and aligns closer to the training distribution of computer vision models and visual data of humans. Our benchmark includes physically grounded visual changes of these real-world objects. This approach differs from existing abstract reasoning datasets (see Figure 2 for comparison). 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0003-00.png)


**----- Start of picture text -----**<br>
I. II.<br>III.<br>?<br>(a) Prior benchmarks. (b) Our benchmark.<br>**----- End of picture text -----**<br>


Figure 2: **Prior benchmarks versus our benchmark for visual analogies. (a)** Prior benchmarks such as **I.** ConceptARC [53], **II.** Raven’s Progressive Matrices [34], and **III.** CCSE reasoning patterns [2] focus on arbitrary changes of abstract shapes and 2D grids. **(b)** By contrast, our benchmark examines basic visual transformations of real-world objects that even kids can do from the age of 3. 

Our second key approach is inspired by _developmental psychology_ , specifically how children learn to perform analogical reasoning not abstractly, but from simple objects in grounded contexts [15, 25, 28]. We propose a similar approach for large multimodal models, investigating if they can perform similarly to children on basic visual analogical reasoning capabilities related to color, size, orientation, and number as reported in [28]. This approach, starting with simple, real-world relevant tasks in child development, allows models to develop robust reasoning abilities before tackling more advanced tasks, providing a clearer pathway for evaluating and improving cognitive functions in AI. 

Third, we break down our evaluation to examine the _different steps_ involved in analogical reasoning to pinpoint which steps a model can perform and where it may fail: _1)_ detecting if there is a general change in an object for a particular visual domain, _2)_ inferring the specific change rule within the correctly identified size, color, etc. and _3)_ extrapolating the inferred rule to a new example. This three-stage evaluation (Fig. 3) gives us insights into models’ reasoning processes beyond simply selecting a correct or incorrect response at the end. 

Results from KiVA demonstrate that visual analogies are not entirely solved by the large multimodal models we tested, GPT-4V [54], LLaVA-1.5 [46] and MANTIS [35]. These models do not match even the capabilities of a three-year-old child (See Figure 1(right)). While LMMs can detect some forms of object transformations, they cannot make extrapolations about those transformations on new objects. In particular, GPT-4V outperforms LLaVA-1.5 and MANTIS but also demonstrates weaker performance in orientation and number changes than in size and color changes which are also processed more quickly by humans, processed at an earlier age (e.g., [71, 76]), and at a more primary region of the visual cortex (e.g., [82, 83]). 

Finally, we include a more challenging benchmark, _KiVA-adults_ (dataset and results described in Appendix C), which includes 2,900 transformations requiring deeper generalization from training examples. These transformations are solvable by adults but not by children under seven years old, providing the next benchmark for models to surpass after KiVA. 

## **2 Related Work** 

**Evaluating human visual analogical reasoning.** There are a variety of tasks designed in Developmental Psychology to examine human visual analogical reasoning early on in life. Children are asked to compare simple object and relational matches [15, 28, 41] along dimensions such as color [50, 65], number [13, 42], size [17, 71] and spatial orientation [23, 62]. Older children and adults are evaluated on Raven’s Progressive Matrices (RPMs) [9, 48, 63] and Bongard Problems [7, 80]. Even though they tend to be the most representative and largest testbeds for testing advanced visual analogical reasoning, RPMs and Bongard problems utilize abstract geometric shapes and test recognition of arbitrary patterns that (1) cannot be solved by young children before the age of 6 and (2) are not critically relevant to visual processing in everyday life. We are the first to provide a visual analogical reasoning benchmark that includes common real-world objects and more natural visual cognition 

3 

skills such as counting and spatial transformations (reflections and rotations)—a task that even a young child can handle [28]. We also examine where people and models may fail by breaking evaluation down into finer steps. 

**Evaluating visuo-linguistic reasoning in AI models.** Several proposals for evaluating modern AI systems’ visuo-linguistic reasoning capabilities followed the recent successes of large multimodal models. Many concentrate on a narrow, isolated set of tasks for detecting object properties and transformations. These tasks include size estimation [11, 47], color perception [1, 66], counting objects [43, 56], spatial reasoning in terms of object viewpoint/pose and chirality [40, 44, 11] and visuo-linguistic compositionality [75, 39, 45]. Typically, the objective of these tasks is to evaluate models’ ability to report a correct property about objects in an image, but they lack the depth to probe the pattern abstraction and generalization involved in visual analogical reasoning. They focus on only detecting narrow and superficial object properties, and fall short in testing robust understanding. 

Broader benchmarks, such as visual question answering setups [3, 30], attempt to investigate the models’ understanding of various visual concepts. One approach taken by [8, 81] was to try and push the envelope on various tasks to capture anecdotal and qualitative observations regarding the performance of GPT-4. Perception Test [57] proposed a second approach: a visual video-based benchmark including developmentally-inspired tasks such as object permanence, object tracking, spatial relations, etc. Recently, the BLINK benchmark was introduced to show that core visual perception tasks, easily solvable by humans "within a blink," remain challenging for large multimodal models due to their resistance to language-based mediation [24]. However, all these benchmarks fall short in evaluating the deeper, more complex aspects of visual analogical reasoning and generalization. 

Another specific class of benchmarks tests generalization and reasoning within 2D puzzle grids. These include Abstraction and Reasoning Corpus (ARC) [14, 53, 52]; a direct translation of RPMs-based human evaluation has previously been applied to models by [2] and [34] (also see prior benchmarks (b) and (c) in Figure 2). However, the stimuli are simple, monotonic shapes like squares and circles, lacking real-world complexity and variability. Moreover, they primarily focus on abstract pattern recognition and logical sequencing without incorporating real-world contexts, thereby neglecting more fundamental visual cognition domains that three-year-olds are capable of. 

In summary, although numerous benchmarks assess advanced visual capabilities in large multimodal models, none evaluate abilities that are demonstrably present in young children, such as drawing analogies about transformations of simple real objects. Developing a curriculum that gradually increases in complexity and abstraction, starting with benchmarks that young children can solve, ensures that models build foundational visual reasoning skills before tackling abstract and complex tasks, mirroring the natural progression of human cognitive development. 

## **3 The KiVA Benchmark for Visual Analogical Reasoning** 

We introduce the kid-inspired visual analogies benchmark, wherein real-world objects undergo common transformations necessary for everyday visual cognition. We focus on isolating and testing basic visual transformations that even a three-year-old child understands [28]. As we show in Figure 1, we examine noticing **color changes** [65, 50], **size changes** [17, 71], **rotation** , **reflection** [62, 23], and **number changes** such as addition and subtraction of a small number of objects [13, 42]. The benchmark is available at `https://github.com/ey242/KiVA` . 

## **3.1 A Three-Stage Experimental Paradigm** 

We use our proposed dataset to benchmark computational models’ and human subjects’ visual analogical reasoning capabilities. Critically, we utilize the same testing paradigm for both kinds of subjects. In each trial, we start by presenting a given transformation _Tgiven_ of an object that changes by a specific rule, following the experimental paradigm of other analogical reasoning benchmarks for humans and computational models [53, 7, 28]. We then evaluate the subject’s ability to determine _what_ changed, or the domain of transformation _Dcross_ , _how_ it changed, or the specific transformation within the correct domain _Dwithin_ , and the outcome of a new object that undergoes the same transformation—i.e., an _extrapolation E_ . We break the question down into these three steps to test the different cognitive processes involved in analogical reasoning. The last step represents the core visual analogy task, while the first two assess the necessary prerequisites for accurate analogical 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0005-00.png)


**----- Start of picture text -----**<br>
1. Cross-domain Change  2. Within-domain Change  3. Visual Extrapolation<br>Detection Detection<br>Which one of the following rules {(1)  Which one of the following rules {(1)<br>change in orientation of objects, (2)  objects become smaller, (2) objects<br>change in number of objects, (3)  correct become bigger, (3) no change, (4) doesn’t<br>change in size of objects, (4) no  apply} best describes the left-to-right<br>change, (5) doesn’t apply} best  transformation in the top of the puzzle  Which one of three left-to-right object<br>describes the left-to-right transformation  where the picture on the left transforms to  transformations (marked by either (A), (B) or (C))<br>on top of the puzzle where the picture  the picture on the right?. Answer with the  on the bottom of the puzzle is the same as the<br>on the left transforms to the picture on  correct rule number surrounded by  left-to-right transformation on the top of the<br>the right? Answer with the correct rule  parentheses. Then provide a "step-by- puzzle? Answer with the correct letter surrounded<br>number surrounded by parentheses,  step" reasoning for your choice. by parentheses (or (D) if none of the options<br>then provide a "step-by-step" reasoning  incorrect apply), then provide a "step-by-step" reasoning<br>for your choice. for your choice.<br>**----- End of picture text -----**<br>


Figure 3: **An example of a trial in KiVA.** A transformation _Tgiven_ is first given to models and humans to observe. We then evaluate them on cross-domain change detection (left). Suppose the correct domain of transformation _Dcross_ is identified (as indicated by the green arrow). In that case, humans and models are further evaluated on their within-domain change detection _Dwithin_ (middle) and then on visual extrapolation _E_ (right). Otherwise, for incorrect cross-domain detection (yellow arrow), humans and models skip to make a visual extrapolation. 

reasoning. We pose all questions in a multiple-choice format, which enables automatic scoring. All multiple-choice options include “Doesn’t apply” to accommodate responses that the provided choices may not cover. Please refer to Figure 3 for the three-stage query pipeline and Appendix A.2 for more details of the specific prompts used. 

**Cross-domain change detection (“what”).** We first evaluate if the subject can detect what changed in the given transformation _Tgiven_ and attribute these changes to the relevant visual domain across all domains _Dcross_ , such as identifying a change in size as opposed to number and orientation (see Figure 3). We randomly sample incorrect multiple-choice options from other possible change domains. “No change” and “Doesn’t apply” are also included as options to accommodate for alternative model reasoning. Suppose the model fails to identify basic changes, such as distinguishing a change in number from a color change. In that case, it is pointless to advance to predicting how new objects will change based on the given transformations. This limitation highlights the inadequacy of existing visual analogical reasoning benchmarks [53, 52, 2, 34], which focus solely on advanced predictions without ensuring fundamental change detection capabilities. 

**Within-domain change detection (“how”).** If a model or human detects the correct cross-domain change _Dcross_ , we ask them further to identify the specific change within the correct visual domain _Dwithin_ (see green arrow in 3). This step is crucial because it ensures the model or human can accurately specify the rule governing the change before extrapolating to a new object. If they fail to identify the specific change, any attempt at extrapolation would likely be incorrect. By pinpointing where reasoning fails, we can better understand models’ and humans’ limitations and improve analogical reasoning capabilities in models. We randomly sample the incorrect multiple-choice options from other possible specific changes within _Dcross_ . 

**Visual Extrapolation.** Finally, we proceed to the step captured by other benchmarks: presenting a new image and asking the model to extrapolate how it will change based on the previously identified transformation (see Figure 3 and other extrapolation examples of other visual domains in Appendix A.1). We ask models to visually extrapolate independent of their performance in verbal change identification to account for the possibility that models may engage in visual analogical reasoning separately from verbal reasoning and can, therefore, perform well in visual tasks even if they struggle with the prior verbal descriptions. This approach helps us determine if a model’s visual reasoning can function independently of its verbal reasoning skills. It provides a more nuanced evaluation of its cognitive capabilities and identifies specific areas for improvement. 

## **3.2 A Dataset of Visual Analogies** 

We create a dataset of stimuli using naturalistic everyday objects that better represent the characteristics of real-world visual data and better match the training data of computer vision models (and humans). We take 3D models of household objects from [21] and objects commonly encountered 

5 

by infants and children from [74]. To set up the dataset, we perform five basic visual transformation domains: changing the size, color, and number of objects, rotating and reflecting the objects along different axes (see Figure 1 for the transformation domains examined). We select these five types of object transformations because they are crucial for object and scene recognition (e.g., [19, 26]), scene segmentation (e.g., [10]), and detecting significant changes in the environment [31, 22]. Other visual properties, such as depth [12], spatial compositionality [37, 75], and physical affordances [36, 67] are also crucial for such purposes; however, the five transformations in our benchmark have been shown to be developmentally plausible for analogical reasoning in young children, as examined in [28], and are therefore prioritized. Below, we outline the five visual transformation domains, each consisting of subdomains that specify an object transformation. There are 100 object transformations for each subdomain of transformation, totaling 1,400 object transformations for the 14 subdomains. 

**Color changes.** Noticing color changes can signal alterations in an object’s state or presence, which is essential for tasks like identifying ripe fruit or detecting hazards [49]. In KiVA, the general transformation rule for color is that input objects change to a single color as in [28]. The subdomains of color examined are red, green, and blue. 

**Size changes.** Size perception allows individuals to understand and interact with their environment accurately, guiding tasks like identifying objects, planning actions, navigating spaces, and avoiding obstacles [27]. In KiVA, objects undergo transformations in two subdomains: they turn bigger or smaller (in both height and width) as in [28] by a factor of 2. 

**Rotation.** Mental rotation is the ability to recognize and map different views of the same object [70]. This is essential for identifying objects despite changes in perspective, which is vital for navigation, object manipulation, and spatial orientation [59]. KiVA takes inspiration from psychometric studies probing mental rotation in humans (e.g., [6, 62]), featuring object rotation in 2D space by the subdomains of 90 degrees (clockwise or counterclockwise) or 180 degrees. 

**Reflection.** Like rotation, recognizing reflections is essential for spatial orientation and precise object identification. This ability aids in understanding object symmetry and chirality, essential for distinguishing left and right shoes or gloves, and recognizing molecular isomers in organic chemistry, for example [32]. Chiral objects cannot be rotated and translated into alignment with their reflection, so there is a visual difference between the object and its reflection [44]. We reflect chiral objects in two subdomains: along the x-axis or the y-axis as in [28]. 

**Number changes.** Accurately monitoring and comparing quantities is essential in various fields like economics and science; it is also important in daily life activities like shopping, cooking, and managing resources effectively, as in caching and rationing [10, 16]. Transformations in this domain reflect basic mathematical operations over the number of objects in an image. Subdomains of number transformations include addition (+1 _,_ +2) and subtraction ( _−_ 1 _, −_ 2). We restrict the number of objects in an input or output image to under 8. 

## **4 Comparing Analogical Reasoning in LMMs and Humans** 

**Evaluating Large Multimodal Models.** We test several LMMs: 1) GPT4-V (OpenAI API model: gpt-4-vision-preview) [55]: an extension of the language-only GPT-4 [54] incorporating computer vision capabilities, 2) LLaVA-1.5 [46]: an open-source model that integrates a vision encoder with a language model, specifically designed to enhance general-purpose visual and language understanding, 3) MANTIS [35] which builds on modified architectures from notable models like LLaVA to support interleaved multi-image input. We combine the given transformation with the choices of new object transformations at the extrapolation step into a single composite input image for LLaVA-1.5, which is limited to processing a single image, but present these as separate images to MANTIS, which is fine-tuned to manage interleaved multi-image conversations. We evaluate GPT-4V under both multi-image and single-image presentations. 

We randomize each experiment over three seeds and run each trial (Figure 3) on a model three times. We score correct choices as 1 and incorrect choices as 0. We calculate the mean score across its three seeds. Subsequently, to evaluate the performance for each transformation domain, we calculate the overall mean and standard error for the average scores of all averaged trials within each domain. GPT-4V, LLaVA-1.5, and MANTIS complete the entire benchmark, featuring 1,400 transformations. Open-source models ran on an A6000 48 GB single GPU for under 12 hours. 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0007-00.png)


Figure 4: **LMM Performance.** Overall, LMMs can detect transformations and identify the general visual domain of the transformations (“Cross Domain” in blue). However, performance generally declines when the models are asked to pinpoint the exact transformation observed within the correctly identified visual domain (“Within Domain” in orange). Performance for visual extrapolation declines even more, as illustrated by the green bars labeled “Extrapolation.” Additionally, the models’ performance is domain-dependent. Error bars represent standard errors of performance across object variations within the same transformation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0007-02.png)


Figure 5: **Human adults (left) and children (right) performance.** Both children and adults show robust performance across “Cross Domain” (blue) and “Within Domain” (orange) change detection as well as “Extrapolation” (green). Humans perform equally well across domains, with rotation proving slightly more difficult for children. Error bars represent standard errors of performance across object variations within the same transformation. 

**Evaluating Humans.** A corresponding visual analogies task, developed using JsPsych [18], was administered to two groups of human participants. All methods were approved by IRB (protocol 2020-10-13755) prior to testing both child and adult participants. We recruited 40 adults (aged 21 to 35 years) on Prolific [60] to complete a random subset of 14 trials (we randomly sampled 1 trial from each of the 14 transformation subdomains), leading to a total of 560 responses. We recruited 20 children (aged 3 to 7 years, _mean_ = 4.6 years, _se_ = 0.21 years) from local early childhood centers and ChildrenHelpingScience [68] to complete a random subset of 10 trials (we randomly sampled 2 trials for each of the 5 transformation domains), leading to a total of 200 responses. Human participants received the same text and image prompts as the models without feedback, but we informed them that rewards were contingent on their performance. Adults were paid at least $19/hour with a bonus of $0.01 per correct response, while children received stickers based on their performance. 

Mirroring model evaluation, we calculate humans’ the average scores and standard errors across trials within each transformation domain. The standard error reflects the variation in performance across trials completed by all participants, not variations between human participants. 

## **5 Results** 

**Models get worse with increasing reasoning complexity, unlike humans.** Overall, LMMs can detect transformations and identify the general visual domain of the transformations (e.g., color vs. size), as indicated by the blue bars labeled “Cross Domain” in Figure 4, with GPT-4V and MANTIS generally outperforming LLaVA-1.5. GPT-4V and MANTIS outperform human children in recognizing when an object changes color or spatial orientation (rotation and reflection). However, 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Within-Domain Extrapolation<br>X +1 Y<br>180 ° +2 -1 -2 -90 ° +90 ° 180 °<br>+2 X<br>+90 ° -90 °<br>+1 -1 Y<br>-2 Bigger<br>Smaller<br>Bigger BlueRedGreenSmaller Red Green Blue<br>**----- End of picture text -----**<br>


Figure 6: **Mean error score of GPT-4V is positively correlated with mean response time of adults in within-domain transformation identification (left) and visual extrapolation (right).** Each dot is color-coded by transformation domain and labeled by subdomain). In subdomains where adults took longer to respond correctly, GPT-4V also tended to make more errors, _r_ = .78, _p_ = .0011 for within-domain and _r_ = .63, _p_ = .017 for extrapolation responses. 

performance generally declines when the models are asked to pinpoint the exact transformation observed within the correctly identified visual domain (e.g., becoming bigger or smaller if size is the correctly identified domain), as reflected by the orange bars labeled “Within Domain”. Performance for visual extrapolation declines even more, as illustrated by the green bars labeled “Extrapolation”. In other words, models’ success in verbally describing transformations does not guarantee their success in extrapolation. We may attribute part of the models’ failure in analogical reasoning to an inability to correctly recognize the given transformation _Tgiven_ . However, another part of the model’s failure lies in extrapolating the correctly identified transformation to a novel object and predicting the corresponding outcome. By contrast, children and adults show robust performance across cross-domain and within-domain change detection and extrapolation, as shown in Figure 5. They can correctly describe the transformations involved in _Tgiven_ and extrapolate the visual transformations to new objects of novel input values. Specific numerical results can be found in Table 1 in Appendix B. We also observe a similar pattern of results for KiVA-adults (see results of adults and models in Appendix C.2). 

**Model performance depends on the visual domain.** Overall, models are better at detecting and describing color transformations than transformations in other domains. GPT-4V performs better at detecting and extrapolating color and size transformations, which involve more discrete and local processing than the other domains [82, 83]. However, LLaVA-1.5 and MANTIS do not perform well on generalizing size transformations relative to other types of transformations, even when all the possible relative sizes of objects are presented to LLaVA in a single image. Overall, all three models are less able to tell what changed within the visual domains of rotation, reflection, and number, as demonstrated by the orange bars labeled “Within Domain” in Figure 4, and consequently also did not perform well in extrapolations for those domains. In contrast, children and adults generally perform well across all visual domains in Figure 5, with children performing slightly worse on rotation, suggesting greater difficulty in appreciation of spatial orientation compared to other domains. 

Interestingly, even though human adults perform about equally well in all visual domains, their response time differed by the domain in a pattern that corroborates well with GPT-4V’s performance pattern across the domains, as demonstrated in Figure 6. Adults take longer to identify the withindomain transformation and make visual extrapolations about rotations and reflections; GPT-4V likewise shows higher error scores in those domains. This suggests that what might be cognitively demanding to humans is perhaps more challenging to process computationally for GPT-4V. 

**In-context learning and prompt engineering did not improve model performance.** We explore whether we can attain better model performance through careful prompt engineering, which has shown promising results on various tasks [79, 61]. We consider four different prompt engineering methods: _1) Reasoning through code [69]:_ We first prompt the model to generate code snippets describing each transformation in the task, then rephrase the task question to incorporate the generated code. _2) Reasoning after Reflection:_ We ask the model to reflect on its answers two times for each question in the task. _3) Reasoning through instruction:_ inspired by [79], which shows that chainof-thought reasoning is more effective on several benchmarks, we prompt the model to generate 

8 

step-by-step instructions on how to answer each question, then use the instructions to generate an answer. _4) In-Context Learning [20]:_ We prompt the model with two randomly sampled examples with solutions for each concept before displaying the task. Apart from text prompt engineering, we experiment with different visual prompting for LLaVA-1.5. Recent works [4, 5, 77] show that visual model performance is sensitive to the altering of color and size of the visual input. We apply two visual prompting approaches: _1) Color_ : we alter the image background color (initially transparent) into black and white [4]. _2) Size_ : We apply a center crop to the images, varying the image size between 0.9 and 1. Surprisingly, we see that performance with these approaches remains the same, which points to the challenging nature of our benchmark. Refer to Appendix A for details. 

## **6 Discussion** 

Despite the scale of image and text data used to train GPT-4V, LLaVA-1.5, and MANTIS, these models fail to reason about visual analogies in the way humans do from a young age. GPT-4V outperforms the other models but still does not approach child performance. It performs better in transformations related to color and size than those related to number and spatial orientation, but even then, it shows a substantial decline in performance with increasing reasoning complexity leading up to visual extrapolation, whereas humans do not show much decline from generalizing changes to making extrapolations about new objects. 

Unlike simply observing discrete changes such as color or size, perceiving reflection, rotation, and numerical changes requires actively engaging with the environment: determining number changes involves sequentially keeping track of numbers, whereas reflection and rotation necessitate visualizing and mentally manipulating objects in space. These tasks are not only more cognitively complex but are also more deeply grounded in interactions with the 3D physical world rather than relying on 2D image-text correlations, which are often used in training LMMs. Furthermore, compared to spatial and numerical transformations, changes in size and color tend to be processed much earlier in the visual pathway of the brain [82, 83] and in human development [17, 50, 65, 71]. Adults take longer to process visual analogies related to changes in number and orientation, suggesting that it may be more cognitively challenging. We do not find a correlation with children’s reaction times as there is great variability in response times given the age range that we recruited (3 to 7 years old) [38], but we also observe that children are less accurate in reasoning about rotation relative to other domains. 

**Limitations.** Our benchmark is designed to assess basic visual change detection and analogical reasoning capabilities commonly studied in children by cognitive psychologists [28]. It aligns with the documented capabilities of young children, as evidenced by the successful completion of our task by individuals as young as three years old through randomized sampling from the larger set of trials. We observe that LMMs perform worse than human participants and that there is no marked improvement through in-context learning and visual or textual prompting techniques. However, this outcome should not be hastily categorized as a null result. Further research is necessary to explore practical strategies for improving model performance. Furthermore, our benchmark is far from capturing the full spectrum of visual cognition. However, it represents a foundational effort to systematically evaluate visual analogical reasoning fundamentally and groundedly using everyday real-world objects and established principles from developmental and cognitive psychology. 

## **7 Conclusion** 

Our findings suggest that large pretrained multimodal models are still less capable than humans in visual analogical reasoning. While these models can identify changes in images, their performance diminishes when tasked with verbally identifying domain-specific changes and further declines when asked to extrapolate these identified changes to a new, unseen object. Among the three models evaluated, GPT-4V performs the best, particularly in tasks involving straightforward visual analogies, such as changes in color and size. However, it struggles with more challenging analogies that require a greater understanding of the 3D physical world, such as numerical and spatial changes. These rely less on simple image-text correlations and more on deeper spatial and contextual understanding. In contrast, humans, including young children, can recognize and interpret a wide range of object relations and transformations, as noted in existing studies [28, 52]. 

9 

## **References** 

- [1] Mostafa Abdou, Artur Kulmizev, Daniel Hershcovich, Stella Frank, Ellie Pavlick, and Anders Søgaard. Can language models encode perceptual structure without grounding? a case study in color. _arXiv preprint arXiv:2109.06129_ , 2021. 

- [2] Kian Ahrabian, Zhivar Sourati, Kexuan Sun, Jiarui Zhang, Yifan Jiang, Fred Morstatter, and Jay Pujara. The curious case of nonverbal abstract reasoning with multi-modal large language models. _arXiv preprint arXiv:2401.12117_ , 2024. 

- [3] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In _Proceedings of the IEEE international conference on computer vision_ , pages 2425–2433, 2015. 

- [4] Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. _arXiv preprint arXiv:2312.00785_ , 2023. 

- [5] Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, and Alexei Efros. Visual prompting via image inpainting. _Advances in Neural Information Processing Systems_ , 35:25005–25017, 2022. 

- [6] George M Bodner and Roland B Guay. The purdue visualization of rotations test. _The chemical educator_ , 2(4):1–17, 1997. 

- [7] M. M. Bongard. _Pattern Recognition_ . Spartan Books, New York, 1970. 

- [8] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv preprint arXiv:2303.12712_ , 2023. 

- [9] Patricia A Carpenter, Marcel A Just, and Peter Shell. What one intelligence test measures: a theoretical account of the processing in the raven progressive matrices test. _Psychological review_ , 97(3):404, 1990. 

- [10] Prithvijit Chattopadhyay, Ramakrishna Vedantam, Ramprasaath R Selvaraju, Dhruv Batra, and Devi Parikh. Counting everyday objects in everyday scenes. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 1135–1144, 2017. 

- [11] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. _arXiv preprint arXiv:2401.12168_ , 2024. 

- [12] Weifeng Chen, Zhao Fu, Dawei Yang, and Jia Deng. Single-image depth perception in the wild. _Advances in neural information processing systems_ , 29, 2016. 

- [13] Anoop Cherian, Kuan-Chuan Peng, Suhas Lohit, Kevin A Smith, and Joshua B Tenenbaum. Are deep neural networks smarter than second graders? In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10834–10844, 2023. 

- [14] François Chollet. On the measure of intelligence. _arXiv preprint arXiv:1911.01547_ , 2019. 

- [15] Stella Christie and Dedre Gentner. Where hypotheses come from: Learning new relations by structural alignment. _Journal of Cognition and Development_ , 11(3):356–373, 2010. 

- [16] I Bernard Cohen. _The triumph of numbers: How counting shaped modern life_ . WW Norton & Company, 2005. 

- [17] RH Day and BE McKenzie. Infant perception of the invariant size of approaching and receding objects. _Developmental Psychology_ , 17(5):670, 1981. 

- [18] Joshua R De Leeuw. jspsych: A javascript library for creating behavioral experiments in a web browser. _Behavior research methods_ , 47:1–12, 2015. 

- [19] Vaibhav A Diwadkar and Timothy P McNamara. Viewpoint dependence in scene recognition. _Psychological science_ , 8(4):302–307, 1997. 

- [20] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. A survey on in-context learning. _arXiv preprint arXiv:2301.00234_ , 2022. 

- [21] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In _2022 International Conference on Robotics and Automation (ICRA)_ , pages 2553–2560. IEEE, 2022. 

- [22] Shinchieh Duh and Su-hua Wang. Infants detect changes in everyday scenes: The role of scene gist. _Cognitive Psychology_ , 72:142–161, 2014. 

- [23] Andrea Frick, Melissa A Hansen, and Nora S Newcombe. Development of mental rotation in 3-to 5-year-old children. _Cognitive Development_ , 28(4):386–399, 2013. 

10 

- [24] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. _arXiv preprint arXiv:2404.12390_ , 2024. 

- [25] Dedre Gentner. Structure-mapping: A theoretical framework for analogy. _Cognitive science_ , 7(2):155–170, 1983. 

- [26] Theo Gevers and Arnold WM Smeulders. Color-based object recognition. _Pattern recognition_ , 32(3):453– 464, 1999. 

- [27] Nicholas A Giudice. Navigating without vision: Principles of blind spatial cognition. In _Handbook of behavioral and cognitive geography_ , pages 260–288. Edward Elgar Publishing, 2018. 

- [28] Mariel K Goddu, Tania Lombrozo, and Alison Gopnik. Transformations and transfer: Preschool children understand abstract relations and reason analogically in a causal task. _Child development_ , 91(6):1898–1915, 2020. 

- [29] Usha Goswami. _Analogical reasoning in children_ . Psychology Press, 2013. 

- [30] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 6904–6913, 2017. 

- [31] Gary Hatfield and Sarah Allred. _Visual experience: sensation, cognition, and constancy_ . Oxford University Press, 2012. 

- [32] Corinne A Holmes, Nora S Newcombe, and Thomas F Shipley. Move to learn: Integrating spatial information from multiple viewpoints. _Cognition_ , 178:7–25, 2018. 

- [33] Keith J Holyoak. Analogy and relational reasoning. _The Oxford handbook of thinking and reasoning_ , pages 234–259, 2012. 

- [34] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you need: Aligning perception with language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [35] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning. _arXiv preprint arXiv:2405.01483_ , 2024. 

- [36] Guangyuan Jiang, Chuyue Tang, Yuyang Li, and Yu Liu. Bongard-tool: Tool concept induction from few-shot visual exemplars. In _PKU 22Fall Course: Cognitive Reasoning_ , 2023. 

- [37] Huaizu Jiang, Xiaojian Ma, Weili Nie, Zhiding Yu, Yuke Zhu, and Anima Anandkumar. Bongard-hoi: Benchmarking few-shot visual reasoning for human-object interactions. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 19056–19065, 2022. 

- [38] Robert Kail. Developmental change in speed of processing during childhood and adolescence. _Psychological bulletin_ , 109(3):490, 1991. 

- [39] Amita Kamath, Jack Hessel, and Kai-Wei Chang. What’s" up" with vision-language models? investigating their struggle with spatial reasoning. _arXiv preprint arXiv:2310.19785_ , 2023. 

- [40] Ivan Kapelyukh, Yifei Ren, Ignacio Alzugaray, and Edward Johns. Dream2real: Zero-shot 3d object rearrangement with vision-language models. In _First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024_ , 2023. 

- [41] Megumi Kuwabara and Linda B Smith. Cross-cultural differences in cognitive development: Attention to relations and objects. _Journal of experimental child psychology_ , 113(1):20–35, 2012. 

- [42] Susan Cohen Levine, Nancy C Jordan, and Janellen Huttenlocher. Development of calculation abilities in young children. _Journal of experimental child psychology_ , 53(1):72–103, 1992. 

- [43] Dingkang Liang, Jiahao Xie, Zhikang Zou, Xiaoqing Ye, Wei Xu, and Xiang Bai. Crowdclip: Unsupervised crowd counting via vision-language model. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2893–2903, 2023. 

- [44] Zhiqiu Lin, Jin Sun, Abe Davis, and Noah Snavely. Visual chirality. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 12295–12303, 2020. 

- [45] Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. _Transactions of the Association for Computational Linguistics_ , 11:635–651, 2023. 

- [46] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36, 2024. 

- [47] Zixu Liu, Qian Wang, and Fanlin Meng. A benchmark for multi-class object counting and size estimation using deep convolutional neural networks. _Engineering Applications of Artificial Intelligence_ , 116:105449, 2022. 

11 

- [48] Andrew Lovett and Kenneth Forbus. Modeling visual problem solving as analogical reasoning. _Psychological review_ , 124(1):60, 2017. 

- [49] John Maule, Alice E Skelton, and Anna Franklin. The development of color perception and cognition. _Annual Review of Psychology_ , 74:87–111, 2023. 

- [50] Allen E Milewski and Einar R Siqueland. Discrimination of color and pattern novelty in one-month human infants. _Journal of Experimental Child Psychology_ , 19(1):122–136, 1975. 

- [51] Melanie Mitchell. Abstraction and analogy-making in artificial intelligence. _Annals of the New York Academy of Sciences_ , 1505(1):79–101, 2021. 

- [52] Melanie Mitchell, Alessandro B Palmarini, and Arseny Moskvichev. Comparing humans, gpt-4, and gpt-4v on abstraction and reasoning tasks. _arXiv preprint arXiv:2311.09247_ , 2023. 

- [53] Arseny Moskvichev, Victor Vikram Odouard, and Melanie Mitchell. The conceptarc benchmark: Evaluating understanding and generalization in the arc domain. _arXiv preprint arXiv:2305.07141_ , 2023. 

- [54] OpenAI. Gpt-4 technical report. _arXiv_ , pages 2303–08774, 2023. 

- [55] OpenAI. Gpt-4v(ision) technical work and authors. 2023. `https://cdn.openai.com/contributions/ gpt-4v.pdf` . 

- [56] Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching clip to count to ten. _arXiv preprint arXiv:2302.12066_ , 2023. 

- [57] Viorica P˘atr˘aucean, Lucas Smaira, Ankush Gupta, Adrià Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, et al. Perception test: A diagnostic benchmark for multimodal video models. _arXiv preprint arXiv:2305.13786_ , 2023. 

- [58] Molly R Petersen and Lonneke van der Plas. Can language models learn analogical reasoning? investigating training objectives and comparisons to human performance. _arXiv preprint arXiv:2310.05597_ , 2023. 

- [59] Nicolas Pinto, David D Cox, and James J DiCarlo. Why is real-world visual object recognition hard? _PLoS computational biology_ , 4(1):e27, 2008. 

- [60] Prolific. Prolific: Online platform for participant recruitment. `https://www.prolific.com` . Accessed: 2024-06-05. 

- [61] Guanghui Qin and Jason Eisner. Learning how to ask: Querying lms with mixtures of soft prompts. _arXiv preprint arXiv:2104.06599_ , 2021. 

- [62] Claudia Quaiser-Pohl. The mental cutting test" schnitte" and the picture rotation test-two new measures to assess spatial ability. _International Journal of Testing_ , 3(3):219–231, 2003. 

- [63] John C Raven and JH Court. _Raven’s progressive matrices_ . Western Psychological Services Los Angeles, CA, 1938. 

- [64] Patrik Reizinger, Szilvia Ujváry, Anna Mészáros, Anna Kerekes, Wieland Brendel, and Ferenc Huszár. Understanding llms requires more than statistical generalization. _arXiv preprint arXiv:2405.01964_ , 2024. 

- [65] Shannon Ross-sheehy, Lisa M Oakes, and Steven J Luck. The development of visual short-term memory capacity in infants. _Child development_ , 74(6):1807–1822, 2003. 

- [66] Ahnaf Mozib Samin, M Firoz Ahmed, and Md Mushtaq Shahriyar Rafee. Colorfoil: Investigating color blindness in large vision and language models. _arXiv preprint arXiv:2405.11685_ , 2024. 

- [67] Johann Sawatzky, Yaser Souri, Christian Grund, and Jurgen Gall. What object should i use?-task driven object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7605–7614, 2019. 

- [68] Children Helping Science. Children helping science: The online directory of research studies for children. `https://www.childrenhelpingscience.com` . Accessed: 2024-06-05. 

- [69] Pratyusha Sharma, Tamar Rott Shaham, Manel Baradad, Stephanie Fu, Adrian Rodriguez-Munoz, Shivam Duggal, Phillip Isola, and Antonio Torralba. A vision check-up for language models. _arXiv preprint arXiv:2401.01862_ , 2024. 

- [70] Roger N Shepard and Jacqueline Metzler. Mental rotation of three-dimensional objects. _Science_ , 171(3972):701–703, 1971. 

- [71] Alan Slater, Anne Mattock, and Elizabeth Brown. Size constancy at birth: Newborn infants’ responses to retinal and real size. _Journal of experimental child psychology_ , 49(2):314–322, 1990. 

- [72] Robert J Sternberg. Component processes in analogical reasoning. _Psychological review_ , 84(4):353, 1977. 

- [73] Robert J Sternberg and Bathsheva Rifkin. The development of analogical reasoning processes. _Journal of experimental child psychology_ , 27(2):195–232, 1979. 

12 

- [74] Stefan Stojanov, Anh Thai, and James M Rehg. Using shape to categorize: Low-shot learning with an explicit shape bias. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 1798–1808, 2021. 

- [75] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5238–5248, 2022. 

- [76] Su-hua Wang and Elizabeth J Goldman. Infants actively construct and update their representations of physical events: Evidence from change detection by 12-month-olds. _Child Development Research_ , 2016, 2016. 

- [77] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6830–6839, 2023. 

- [78] Taylor Webb, Keith J Holyoak, and Hongjing Lu. Emergent analogical reasoning in large language models. _Nature Human Behaviour_ , 7(9):1526–1541, 2023. 

- [79] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- [80] Erik Weitnauer, Robert L Goldstone, and Helge Ritter. Perception and simulation during concept learning. _Psychological Review_ , 2023. 

- [81] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). _arXiv preprint arXiv:2309.17421_ , 9, 2023. 

- [82] Semir Zeki, JD Watson, CJ Lueck, Karl J Friston, C Kennard, and RS Frackowiak. A direct demonstration of functional specialization in human visual cortex. _Journal of neuroscience_ , 11(3):641–649, 1991. 

- [83] Hang Zeng, Gereon R Fink, and Ralph Weidner. Visual size processing in early visual cortex follows lateral occipital cortex involvement. _Journal of Neuroscience_ , 40(22):4410–4417, 2020. 

## **A Visual Analogical Reasoning Prompts** 

## **A.1 Stitched visual extrapolation examples for each domain** 

**Visual Extrapolation.** As the final step of the querying process, we presented an image of a new object and ask the model to predict what the object will look like if it goes through the same change as the given transformation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0013-13.png)


Figure 7: Example of a visual extrapolation trial involving a **reflection** . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0013-15.png)


Figure 8: Example of a visual extrapolation trial involving an angular **rotation** . 

13 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0014-00.png)


Figure 9: Example of a visual extrapolation trial involving a **size change** . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0014-02.png)


Figure 10: Example of a visual extrapolation trial involving a **number change** . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0014-04.png)


Figure 11: Example of a visual extrapolation trial involving a **color change** . 

## **A.2 Prompts for the three-step evaluation of the analogical reasoning process** 

We first include a system prompt to orient the models for visual analogical reasoning. _You are an excellent visual puzzle solver! You will be given a visual puzzle that requires using visual analogical reasoning. You will think "step-by-step" and carefully examine the visual evidence before providing an answer._ Then we provide an initial instruction prompt: _You are given a visual puzzle. The puzzle features a left-to-right transformation of an object on top and three left-to-right transformations of a different object on the bottom marked by (A) or (B) or (C). The transformations involve a change of either the size, orientation, number, or color of an object._ 

## 1. **Cross-domain change detection.** 

- _“Which one of the following rules best describes the left-to-right transformation on top of the puzzle where the picture on the left transforms to the picture on the right? Answer with the correct rule number. Surrounded by parentheses, then provide a "step-by-step" reasoning for your choice."_ 

2. **Within-domain change detection.** 

- _“Which one of the following rules best describes the left-to-right transformation in the top of the puzzle where the picture on the left transforms to the picture on the right?. Answer with the correct rule number surrounded by parentheses. Then provide a "step-by-step" reasoning for your choice."_ 

3. **Visual Extrapolation.** 

- _“Which one of the three left-to-right object transformations (marked by either (A), (B) or (C) ) on the bottom of the puzzle is the same as the left-to-right transformation on the top of the puzzle? Answer with the correct letter surrounded by parentheses (or (D) if none of the options apply), then provide a a "step-by-step" reasoning for your choice."_ 

14 

## **A.3 Prompting through reflection and self-critique** 

## 1. **Cross-domain change detection.** 

- _“Which one of the following rules best describes the left-to-right transformation on top of the puzzle where the picture on the left transforms to the picture on the right? Answer with the correct rule number surrounded by parentheses, then provide a "step-by-step" reasoning for your choice."_ 

## 2. **Within-domain change detection.** 

- _“Which one of the following rules best describes the left-to-right transformation in the top of the puzzle where the picture on the left transforms to the picture on the right?. Answer with the correct rule number surrounded by parentheses, then provide a "step-by-step" reasoning for your choice."_ 

3. **Visual Extrapolation.** 

- _“Which one of three left-to-right object transformations (marked by either (A), (B) or (C)) on the bottom of the puzzle is the same as the left-to-right transformation on the top of the puzzle? Answer with the correct letter surrounded by parentheses (or (D) if none of the options apply), then provide a "step-by-step" reasoning for your choice."_ 

## **A.4 Prompting through instructions** 

## 1. **Cross-domain change detection.** 

- _“Which one of the following rules best describes the left-to-right transformation on top of the puzzle where the picture on the left transforms to the picture on the right? Answer with the correct rule number surrounded by parentheses, then provide a “step-by-step” reasoning for your choice."_ 

2. **Within-domain change detection.** 

- _“Provide brief instructions on how to establish if a transformation involves an object rotates 90 degrees or 180 degrees. Use the instructions form before to answer the following question: Which one of the following rules best describes the transformation in the top of the puzzle where the picture on the left transforms to the picture on the right?. Answer with the correct rule number surrounded by parentheses, then provide a “step-by-step” reasoning for your choice."_ 

3. **Visual Extrapolation.** 

- _“Provide brief instructions on how to determine which one of three left-to-right object transformations (marked by either (A), (B) or (C) ) on the bottom of the puzzle is the same as the left-to-right transformation on the top of the puzzle? Use the instructions from before to determine which one of three left-to-right object transformations (marked by either (A), (B) or (C) ) on the bottom of the puzzle is the same as the left-to-right transformation on the top of the puzzle? Answer with the correct letter surrounded by parentheses (or (D) if none of the options apply), then provide a step-by-step reasoning for your choice."_ 

## **A.5 Prompting through code** 

## 1. **Cross-domain change detection.** 

- _“Which one of the following rules best describes the left-to-right transformation on top of the puzzle where the picture on the left transforms to the picture on the right? Answer with the correct rule number surrounded by parentheses, then provide a "step-by-step" reasoning for your choice."_ 

## 2. **Within-domain change detection.** 

- _“Generate python code using the package pillow that takes in the left image in the left-to-right transformation on top and outputs the right image. Denote this snippet as training snippet using the insights from the training code snippet, which one of the following rules best describes the left-to-right transformation in the top of the puzzle where the picture on the left transforms to the picture on the right?. Answer with the correct rule number surrounded by parentheses, then provide a "step-by-step" reasoning for your choice."_ 

15 

## 3. **Visual Extrapolation.** 

_“Generate a brief code snippet using python and the pillow package for each left-to-right transformation in the bottom. Each snippet takes in the left picture of the transformation and outputs the right one. Now Which one of three code snippets is the same as the training code snippet you have produced before. Answer with the correct snippet letter ((A) or (B) or (C)) surrounded by parentheses (or (D) if none of the options apply), then provide a "step-by-step" reasoning for your choice."_ 

## **B Additional details for models’ and human adults’ performance on KiVA** 

||**Rotation**|**Reflection**|**Number**|**Color**|**Size**|
|---|---|---|---|---|---|
||Cross-<br>domain<br>Within-<br>domain<br>Extrapolation|Cross-<br>domain<br>Within-<br>domain<br>Extrapolation|Cross-<br>domain<br>Within-<br>domain<br>Extrapolation|Cross-<br>domain<br>Within-<br>domain<br>Extrapolation|Cross-<br>domain<br>Within-<br>domain<br>Extrapolation|
|**Human**<br>**adults**|**97.5%**<br>(1.43%)<br>**95.0%**<br>(2.00%)<br>**92.4%**<br>(2.43%)|**98.8%**<br>(1.25%)<br>**98.7%**<br>(1.27%)<br>**98.7%**<br>(1.27%)|**95.6%**<br>(1.62%)<br>**94.4%**<br>(1.83%)<br>**97.5%**<br>(1.25%)|**99.2%**<br>(0.83%)<br>**95.8%**<br>(1.83%)<br>**99.2%**<br>(0.84%)|**98.8%**<br>(1.25%)<br>**100%**<br>(0%)<br>**97.5%**<br>(0.18%)|
|**Human**<br>**children**|**80.5%**<br>(6.27%)<br>**75.8%**<br>(7.58%)<br>**63.4%**<br>(7.62%)|**75.0%**<br>(6.93%)<br>**90.0%**<br>(5.57%)<br>**80.0%**<br>(6.41%)|**90.5%**<br>(4.58%)<br>**78.9%**<br>(6.70%)<br>**78.6%**<br>(6.41%)|**92.7%**<br>(4.12%)<br>**94.7%**<br>(3.67%)<br>**95.1%**<br>(3.41%)|**95.0%**<br>(3.49%)<br>**81.6%**<br>(6.37%)<br>**90.0%**<br>(4.80%)|
|**GPT-4V**<br>(single<br>image)|**99.3%**<br>(0.38%)<br>**60.3%**<br>(3.16%)<br>**36.0%**<br>(2.35%)|**95.3%**<br>(1.34%)<br>**58.3%**<br>(3.92%)<br>**34.7%**<br>(2.88%)|**90.5%**<br>(1.46%)<br>**60.3%**<br>(2.79%)<br>**33.5%**<br>(2.03%)|**100%**<br>(0%)<br>**98.0%**<br>(0.73%)<br>**73.5%**<br>(2.67%)|**88.0%**<br>(2.20%)<br>**98.2%**<br>(0.94%)<br>**55.3%**<br>(3.00%)|
|**GPT-4V**<br>(multiple<br>images)|**99.6%**<br>(0.44%)<br>**65.1%**<br>(3.18%)<br>**28.9%**<br>(2.29%)|**77.3%**<br>(3.17%)<br>**35.9%**<br>(3.36%)<br>**30.3%**<br>(2.89%)|**74.3%**<br>(2.10%)<br>**63.5%**<br>(2.63%)<br>**25.1%**<br>(1.81%)|**100%**<br>(0%)<br>**98.4%**<br>(0.80%)<br>**95.9%**<br>(1.05%)|**94.7%**<br>(1.48%)<br>**99.6%**<br>(3.57%)<br>**77.3%**<br>(3.03%)|
|**LLaVA-1.5**|**66.7%**<br>(2.60%)<br>**47.4%**<br>(2.35%)<br>**33.3%**<br>(2.23%)|**64.3%**<br>(3.15%)<br>**52.1%**<br>(4.14%)<br>**28.3%**<br>(2.78%)|**50.7%**<br>(2.07%)<br>**47.2%**<br>(2.08%)<br>**28.3%**<br>(1.96%)|**73.3%**<br>(2.14%)<br>**59.7%**<br>(2.75%)<br>**29.8%**<br>(2.34%)|**25.3%**<br>(2.77%)<br>**50.0%**<br>(4.30%)<br>**29.3%**<br>(2.43%)|
|**MANTIS**|**92.2%**<br>(1.80%)<br>**51.6%**<br>(3.05%)<br>**37.1%**<br>(2.31%)|**83.5%**<br>(3.03%)<br>**61.6%**<br>(3.05%)<br>**32.0%**<br>(2.92%)|**86.5%**<br>(1.97%)<br>**49.7%**<br>(2.04%)<br>**32.8%**<br>(1.91%)|**99.3%**<br>(0.38%)<br>**53.3%**<br>(2.90%)<br>**36.4%**<br>(2.26%)|**39.7%**<br>(3.84%)<br>**48.1%**<br>(3.66%)<br>**34.3%**<br>(2.70%)|
|||||||



Table 1: **Mean performance of humans and models in solving each stage of KiVA, sorted by transformation domain.** Standard errors are included in parentheses. 

## **C KiVA-adults** 

## **C.1 KiVA-adults compared to KiVA** 

KiVA preserves the input and output values between the given transformation _Tgiven_ and test transformation _Ttest_ for extrapolation. As shown in Table 2, only the object undergoing transformation varies between training (left column) and test (middle column). For example, in a numerical transformation, if the given transformation involves turning _three_ bowls into two bowls, test asks for the result of transforming _three_ apples (answer: two apples). KiVA-adults, on the other hand, changes the input value at test (right column). For instance, if the given transformation shows _three_ bowls converting into two bowls again, the test asks for the transformation of _four_ apples (answer: three apples). This requires not only generalizing a transformation across different objects but also abstracting a _function_ to generalize over different _input values_ . 

16 

|Color<br>Size<br>Rotation<br>Refection<br>Number|Given Transformation|KiVA Extrapolation|KiVA-adults Extrapolation|
|---|---|---|---|
|||||



Table 2: **Examples of transformation input values tested on KiVA vs. KiVA-adults.** For KiVA, the extrapolation input matches the given transformation input in relevant domain values (e.g., both use three objects in the number domain). For KiVA-adults, the extrapolation input differs from the given transformation input (e.g., the given transformation has an input of three objects, but the test starts from four objects in the number domain). 

|Color<br>Size<br>Rotation<br>Refection<br>Number|KiVA<br>Red, Green, Blue<br>1_/_2_×_and2_×_height & width<br>_±_90_◦_,180_◦_<br>refect along the_x_or_y_axes<br>+1_,_+2_, −_1_, −_2|KiVA-adults|
|---|---|---|
|||Red, Green, Blue, Yellow, Grey<br>1_/_2_×_and2_×_height & width, or separately<br>_±_45_◦_,_±_90_◦_,_±_135_◦_,180_◦_<br>refect along_x_or_y_axes, or both axes together<br>+1_,_+2_, −_1_, −_2_, ×_2_, ×_3_, ÷_2_, ÷_3|



Table 3: **Visual subdomains involved in KiVA vs. KiVA-adults. Color** : KiVA examines transformations of three primary colors (red, green, blue); KiVA-adults includes two additional colors (yellow, grey). **Size** : KiVA enlarges or shrinks objects in both height and width by a factor of 2; KiVA-adults additionally involves non-uniform scaling, independently halving or doubling either width or height. **Rotation** : KiVA includes rotating objects by _±_ 90 _[◦]_ or _±_ 180 _[◦]_ ; KiVA-adults includes two additional angular values ( _±_ 45 _[◦]_ and _±_ 135 _[◦]_ ). **Reflection** KiVA reflects objects either along the x- or y-axes; KiVA-adults includes rotating objects along both axes in KiVA-adults. **Number changes** : KiVA includes addition and subtraction of objects by 1 or 2; KiVA-adults includes division and multiplication of objects by 2 or 3. 

Furthermore, KiVA-adults includes more subdomains than the original task. The subdomains in KiVA are more distinguishable from each other (see Table 3 for the subdomains evaluated in the two tasks respectively). In KiVA-adults, we also have to add additional constraints in the sampling for the transformation domain involving number changes to avoid ambiguity between multiplication and addition (both involving an increase in number), or between subtraction and division (both involving a decrease in number). 

We run GPT-4V, LLaVA-1.5 and MANTIS on KiVA-adults which consists of 2,900 transformations. We also test 10 children and 40 adults (of the same age ranges as those recruited for KiVA) on KiVA-adults. Each adult completes 19 trials that are randomly sampled from the 29 subdomains, totaling 760 responses. Each child is assigned to complete 10 randomly sampled trials, but none of them succeed on completing them with passing accuracy. 

17 

## **C.2 Models’ and human adults’ performance on KiVA-adults** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0018-01.png)


Figure 12: **Performance of GPT-4V, LLaVA-1.5 and MANTIS on KiVA-adults.** Performance is organized by question type: cross-domain change detection, within-domain change detection (which is only presented if cross-domain change detection is correct), and extrapolation of change to a new picture. Error bars represent standard errors of performance across object variations within the same transformation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models_images/iclr_2025_KiVA_Kid-inspired_Visual_Analogies_for_Testing_Large_Multimodal_Models.pdf-0018-03.png)


Figure 13: **Performance of human adults on KiVA-adults.** Performance is organized by the reasoning step: cross-domain change detection, within-domain change detection (which is only presented if cross-domain change detection is correct), and extrapolation of change to a new picture. Error bars represent standard errors of performance across object variations within the same transformation (not human subject variations). 

||**Rotation**|**Reflection**|**Number**|**Color**|**Size**|
|---|---|---|---|---|---|
||Cross-<br>domain<br>Within-<br>domain<br>Extrapolation|Cross-<br>domain<br>Within-<br>domain<br>Extrapolation|Cross-<br>domain<br>Within-<br>domain<br>Extrapolation|Cross-<br>domain<br>Within-<br>domain<br>Extrapolation|Cross-<br>domain<br>Within-<br>domain<br>Extrapolation|
|**Human**<br>**adults**|**100%**<br>(0%)<br>**85.1%**<br>(2.82%)<br>**88.8%**<br>(2.49%)|**99.2%**<br>(0.83%)<br>**94.2%**<br>(2.43%)<br>**84.2%**<br>(3.35%)|**98.8%**<br>(0.88%)<br>**93.7%**<br>(1.94%)<br>**97.5%**<br>(1.24%)|**100%**<br>(0%)<br>**98.0%**<br>(0.98%)<br>**98.0%**<br>(0.99%)|**98.3%**<br>(1.16%)<br>**95.8%**<br>(1.85%)<br>**98.3%**<br>(1.16%)|
|**GPT-4V**<br>(single<br>image)|**96.4%**<br>(0.55%)<br>**47.2%**<br>(1.43%)<br>**37.6%**<br>(1.18%)|**93.8%**<br>(1.35%)<br>**48.2%**<br>(2.87%)<br>**37.1%**<br>(2.24%)|**90.9%**<br>(0.68%)<br>**62.6%**<br>(1.36%)<br>**35.5%**<br>(0.98%)|**99.5%**<br>(0.27%)<br>**95.4%**<br>(0.91%)<br>**73.1%**<br>(1.70%)|**70.4%**<br>(1.33%)<br>**81.7%**<br>(1.30%)<br>**49.2%**<br>(1.35%)|
|**GPT-4V**<br>(multiple<br>images)|**97.2%**<br>(0.78%)<br>**42.9%**<br>(2.06%)<br>**30.0%**<br>(1.51%)|**79.8%**<br>(2.85%)<br>**27.2%**<br>(2.92%)<br>**32.9%**<br>(2.27%)|**82.0%**<br>(1.26%)<br>**54.4%**<br>(1.65%)<br>**26.7%**<br>(1.03%)|**99.7%**<br>(0.27%)<br>**94.4%**<br>(1.19%)<br>**92.7%**<br>(1.19%)|**75.8%**<br>(1.69%)<br>**70.9%**<br>(2.01%)<br>**53.7%**<br>(1.86%)|
|**LLaVA-1.5**|**68.7%**<br>(1.20%)<br>**30.6%**<br>(1.34%)<br>**33.9%**<br>(1.13%)|**65.9%**<br>(1.89%)<br>**55.3%**<br>(2.55%)<br>**31.5%**<br>(1.82%)|**50.9%**<br>(1.03%)<br>**35.1%**<br>(1.16%)<br>**28.7%**<br>(0.93%)|**75.0%**<br>(1.23%)<br>**56.6%**<br>(1.73%)<br>**31.2%**<br>(1.18%)|**24.2%**<br>(1.21%)<br>**35.7%**<br>(1.69%)<br>**31.7%**<br>(1.10%)|
|**MANTIS**|**93.6%**<br>(1.02%)<br>**49.5%**<br>(2.27%)<br>**40.0%**<br>(1.53%)|**86.6%**<br>(2.18%)<br>**51.0%**<br>(2.72%)<br>**33.3%**<br>(2.27%)|**86.5%**<br>(1.41%)<br>**39.4%**<br>(1.67%)<br>**33.2%**<br>(1.38%)|**95.6%**<br>(1.04%)<br>**54.8%**<br>(2.51%)<br>**34.4%**<br>(1.71%)|**38.3%**<br>(1.99%)<br>**48.1%**<br>(2.18%)<br>**33.1%**<br>(1.55%)|
|||||||



Table 4: **Mean performance of humans and models in solving each stage of KiVA-adults, sorted by transformation domain.** Standard errors are included in parentheses. 

18 

