## **Path-Specific Causal Reasoning for Fairness-aware Cognitive Diagnosis** 

## Dacao Zhang 

zhdacao@gmail.com School of Computer Science and Information Engineering, Hefei University of Technology Hefei, Anhui, China 

## Mi Tian 

tianmixlb@gmail.com School of Computer Science and Information Engineering, Hefei University of Technology Hefei, Anhui, China 

## Kun Zhang[∗] 

zhang1028kun@gmail.com School of Computer Science and Information Engineering, Hefei University of Technology Hefei, Anhui, China 

## Richang Hong 

hongrc.hfut@gmail.com School of Computer Science and Information Engineering, Hefei University of Technology Institute of Dataspace, Hefei Comprehensive National Science Center 

## Le Wu 

lewu.ustc@gmail.com School of Computer Science and Information Engineering, Hefei University of Technology Institute of Dataspace, Hefei Comprehensive National Science Center Hefei, Anhui, China 

## Meng Wang 

eric.mengwang@gmail.com School of Computer Science and Information Engineering, Hefei University of Technology Hefei, Anhui, China 

Hefei, Anhui, China 

## **ABSTRACT** 

Cognitive Diagnosis (CD), which leverages students and exercise data to predict students’ proficiency levels on different knowledge concepts, is one of fundamental components in Intelligent Education. Due to the scarcity of student-exercise interaction data, most existing methods focus on making the best use of available data, such as exercise content and student information (e.g., educational context). Despite the great progress, the abuse of student sensitive information has not been paid enough attention. Due to the important position of CD in Intelligent Education, employing sensitive information when making diagnosis predictions will cause serious social issues. Moreover, data-driven neural networks are easily misled by the shortcut between input data and output prediction, exacerbating this problem. Therefore, it is crucial to eliminate the negative impact of sensitive information in CD models. In response, we argue that sensitive attributes of students can also provide useful information, and only the shortcuts directly related to the sensitive information should be eliminated from the diagnosis process. Thus, we employ causal reasoning and design a novel _Path-Specific Causal Reasoning Framework_ ( _PSCRF_ ) to achieve this goal. Specifically, we 

∗Corresponding authors. 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _KDD ’24, August 25–29, 2024, Barcelona, Spain._ 

© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/10.1145/XXXXXX.XXXXXX 

first leverage an encoder to extract features and generate embeddings for general information and sensitive information of students. Then, we design a novel attribute-oriented predictor to decouple the sensitive attributes, in which fairness-related sensitive features will be eliminated and other useful information will be retained. Finally, we designed a multi-factor constraint to ensure the performance of fairness and diagnosis performance simultaneously. Extensive experiments over real-world datasets (e.g., PISA dataset) demonstrate the effectiveness of our proposed _PSCRF_ . 

## **CCS CONCEPTS** 

- **Information systems** → **Personalization** . 

## **KEYWORDS** 

Cognitive Diagnosis, User modeling, Causal Reasoning, Sensitive Attribute, Fairness 

## **ACM Reference Format:** 

Dacao Zhang, Kun Zhang, Le Wu, Mi Tian, Richang Hong, and Meng Wang. 2024. Path-Specific Causal Reasoning for Fairness-aware Cognitive Diagnosis. In _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24), August 25–29, 2024, Barcelona, Spain._ ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/XXXXXX.XXXXXX 

## **1 INTRODUCTION** 

As a fundamental component in Intelligent Education, Cognitive Diagnosis (CD) requires an agent to mine student behavior data to access and identify the student’s proficiency level in knowledge concepts [41]. It has been applied in various education scenarios, such as student performance prediction [14, 49], computerized adaptive testing [43], and exercise recommendation [48, 51]. 

KDD ’24, August 25–29, 2024, Barcelona, Spain. 

Dacao Zhang, Kun Zhang, Le Wu, Mi Tian, Richang Hong, Meng Wang. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0002-02.png)


**----- Start of picture text -----**<br>
Exercise Data Student Data Interaction Logs Cognitive Diagnosis Neural Model Student Performance on Knowledge concepts<br>**----- End of picture text -----**<br>


**Figure 1: A tiny example of general cognitive diagnosis.** 

**Table 1: The probability of students answering questions correctly (i.e., Data Statistics) and the predicted probability of students answering questions correctly by NCD and KaNCD among different groups. The groups are divided by the sensitive attributes of students (i.e., family wealth or country).** 

|Model|FamilyWealth|Country|
|---|---|---|
||Poor<br>Average<br>Wealth|Australia<br>Brazil|
|Data statistics<br>NCD<br>KaNCD<br>NCD-_PSCRF_<br>KaNCD-_PSCRF_|0.4736<br>0.5448<br>0.6434<br>0.5140<br>0.5861<br>0.6789<br>0.4778<br>0.5589<br>0.6643<br>0.5545<br>0.5798<br>0.6155<br>0.5286<br>0.5581<br>0.6271|0.5516<br>0.3888<br>0.5913<br>0.3293<br>0.5650<br>0.3025<br>0.5824<br>0.3321<br>0.5680<br>0.3026|



Literately, researchers have designed enormous neural networkbased methods to realize accurate diagnosis and student modeling. As illustrated in Figure 1, existing models usually take multi-type information (e.g., student-exercise interaction logs, student’s personal information, exercise text) as the input, and predict students’ mastery of each concept. For example, Wang et al. [41] have built a deep full connection neural network to capture the complex studentexercise interaction records. Wang et al. [42] designed a novel KaNCD method to address the weak knowledge concepts coverage problem. Besides, there are many other methods for high-order student-exercise interaction modeling [1, 8, 25, 56] and graph-based modeling [15, 23, 26, 53]. 

Despite the great progress, these methods will inevitably introduce fairness issues while exploiting the full potential of student data. Taking Table 1 as an example, by conducting statistical analysis, we can observe that students who are from more affluent families or in more developed areas usually have better performance on exercise (e.g., 0 _._ 6434 points for rich boys v.s. 0 _._ 4736 points for poor boys). In fact, this phenomenon occurs more because those students receive better support or training (e.g., more books, computer access opportunities, etc), rather than better family circumstances. However, if we do not constrain the model to exploit all the data, it will easily learn the connections between sensitive information of students and student performance (e.g., using family wealth to predict the student proficiency level), which we name as spurious correlations. From the results in Table 1, we can observe this type of phenomenon. By using PISA data [31] to train NCD and KaNCD models directly, they will overestimate these advantaged students (e.g., 0 _._ 565 for Australia v.s. 0 _._ 3025 for Brazil), showing that they have taken advantage of sensitive information and made unfair predictions. If we apply the unfair results to real-world scenarios, it will exacerbate social prejudices and conflicts, bringing about bad social effects. More seriously, according to the results in Table 1, even if we do not use sensitive attributes as model inputs, NCD and KaNCD models still can infer sensitive attributes of students from the interaction logs and abuse them for better performance. Therefore, **it is crucial to exclude the abuse of student sensitive attributes while ensuring the diagnosis performance** . 

Recently, plenty of fairness-aware methods have been proposed, such as data reweighting (resampling) [20, 35] and adversarial learning [4, 46, 54]. However, these strategies still have unavoidable shortcomings. For example, data resampling methods usually increase/decrease weights of certain student-exercise interactions to realize the fairness target. However, this strategy violates the principle in cognitive diagnosis that the same student should only respond to the same exercise once, and is also dependent on the sensitive attributes [21]. Meanwhile, adversarial learning uses an additional classifier to predict the sensitive attribute from user embeddings and eliminate corresponding information directly. This strategy is too coarse-grained to distinguish available information from sensitive information, leading to a decrease in model capability. To answer the above question, we propose that the fairness-related sensitive features from sensitive attributes should be eliminated as comprehensively as possible while diagnosis-related features from sensitive attributes should be retained as much as possible. For example, family wealth cannot be used as an influencing factor in determining the student proficiency level, while the quality of the learning environment can. For this goal, causal inference [33] is one promising direction. By distinguishing causation and correlations from biased real-world data, causal inference has made great progress in medicine [22], neuroscience [28], cognitive science [36], etc. It also has been proven useful in addressing bias issues in vision question answering [30], text classification tasks [34], anomaly detection [45], and so on. 

To this end, in this paper, we propose to employ causal inference and design a novel _Path-Specific Causal Reasoning Framework_ ( _PSCRF_ ) for fairness-aware CD modeling. Specifically, we leverage a causal graph to describe the correlations and causation between different factors and student proficiency levels. Based on the causal graph, we try to use _PSCRF_ to calculate the path-specific effect of different inputs to the output. We first leverage an encoder to extract features from student-exercise interaction logs and generate embeddings for student IDs and sensitive attributes. Next, we design a novel attribute-oriented predictor (Decoupled Predictor (DP)) to realize the decoupling of sensitive attributes and useful information, in which fairness-related sensitive feature embeddings are used to predict the sensitive attributes and diagnosis-related feature embeddings are used to predict the useful information from sensitive attributes. Moreover, to ensure the quality of decoupling, we also design a multi-factor fairness constraint to restrict the distance of different embeddings. Then, the fairness-aware inference can be obtained by removing the fairness-related sensitive features from the diagnosis process. Finally, we conducted extensive experiments over real-world diagnosis data in various settings. Experimental results demonstrate that _PSCRF_ can achieve impressive debiased performance while maintaining the accuracy of student proficiency level modeling. We also release the code to facilitate the community[1] . 

In summary, our main contributions are reported as follows: 

- We argue that student sensitive information can provide useful information for diagnosis performance improvement. 

1https://github.com/NLPfreshman0/PSCRF 

KDD ’24, August 25–29, 2024, Barcelona, Spain. 

Path-Specific Causal Reasoning for Fairness-aware Cognitive Diagnosis 

Thus, we should eliminate the fairness-related spurious correlations and retain useful information simultaneously for better fairness-aware CD. 

- We introduce causal inference into the diagnosis process and design a novel _PSCRF_ to realize the fairness-aware CD, in which path-specific causal reasoning is employed to eliminate the abuse of sensitive attributes. 

- We have conducted extensive experiments over real-world data and provided detailed analyses of results and models to demonstrate the effectiveness of _PSCRF_ . 

## **2 RELATED WORK** 

The related work can be summarized into two components: 1) _Cognitive Diagnosis_ : giving a brief introduction of CD in intelligent education scenarios; 2) _Fairness-aware User Modeling_ : focusing on fair user representation learning from biased real-world data. 

## **2.1 Cognitive Diagnosis** 

Cognitive Diagnosis (CD) is a fundamental and pivotal task in many real-world intelligent education scenarios [9, 40]. It requires an agent to predict students’ proficiency level of each knowledge concept through historical student-exercise interaction logs. DINA [10] and IRT [13] are two representative methods in this domain. DINA is a discrete CDM that assumes student mastery levels are binary (master the knowledge concept or not) [10]. IRT characterizes students’ abilities as unidimensional and continuous latent traits and designs logistic-like interaction functions to model the probability of a student correctly answering an exercise [13]. 

To improve the diagnosis performance, various methods have been proposed to extend the capability of DINA and IRT [16, 19, 39, 44] and exploit the potential of student and exercise data. For example, Cheng and Liu [8] proposed a DIRT method to extract semantic features from the content of exercise texts for high-quality representation generation. Wang et al. [41] designed an NCD method to exploit student-exercise interactions for accurate student proficiency level modeling. Moreover, Zhou et al. [56] proposed to improve CD performance from the student perspective. They employed context and culture information of students to enrich the student proficiency representation, which is in favor of improving the diagnosis performance. Besides, other data issues in CD are also considered, such as weak knowledge concepts coverage problems [42] and non-interactive knowledge concepts problems [27]. 

## **2.2 Fairness-aware User Modeling** 

User modeling focuses on measuring user characteristics based on user-related data, which plays a crucial role in plenty of scenarios, such as user preference modeling in recommender system [47] and user proficiency level modeling in Intelligent Education [37]. Recent studies have demonstrated that user-related data may contain stereotypes or biased data, which will mislead models to learn the spurious correlations and make vulnerable and unfair decisions. To alleviate this problem, plenty of fairness-aware user modeling methods have been proposed, such as data reweighting [17, 20, 35], regularization [8, 50], and adversarial learning [2, 46, 54, 57]. Among all these methods, causal inference-based methods are one promising direction. For example, Zhao et al. [55] proposed a disentangled 

framework TIDE based on path-specific causal reasoning to deal with the popularity bias in user preference modeling in recommendations. Chen et al. [7] designed a novel data augmentation strategy to balance the training data, so that sensitive-related information will be inactivated when modeling user preference. Apart from this, other types of biases are also hot research topics, such as selection bias [5, 29], exposure bias [6], and unfairness [7, 12]. 

However, due to the sparsity characteristic of student and exercise data in education, existing methods mainly focus on exploiting the potential of data, ignoring the implicit sensitive information abuse problem. Since education plays a crucial role in influencing the trajectory of individuals’ adult lives [38], it is urgent to focus more on this problem. Some works have made early attempts. For example, Yu et al. [52] conducted an analysis to explore the equitable prediction of short-term and long-term college success using various sources of student data. Li et al. [24] proposed a FairLR algorithm to achieve accurate and fair AI prediction to help to realize fair student modeling. Zhang et al. [54] divided student performance into bias proficiency and fair proficiency, then used only fair proficiency to make predictions. For fairness-aware CD modeling, enormous works remain unexplored, such as fair student representations, sensitive attribute utilization, and so on. 

**Our Distinction.** We focus on a more impactful issue: _How to eliminate the abuse of student sensitive attributes from CD models while ensuring the diagnosis performance?_ We argue that student sensitive attributes can also provide useful information, so directly removing them from CD models is not optimal. Thus, we design a novel _PSCRF_ to realize the debiased CD learning while retaining the diagnosis performance. Specifically, _PSCRF_ decouples student sensitive attributes into sensitive-related information that should not be used in diagnosis process, and sensitive-unrelated information that can be used to improve the diagnosis performance. Moreover, _PSCRF_ leverages a multi-factor normalization to ensure the quality of debiased learning and diagnosis performance simultaneously. 

## **3 PRELIMINARY** 

## **3.1 Prerequisite Knowledge** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0003-17.png)


**----- Start of picture text -----**<br>
𝑀 𝑚 [∗]<br>𝑋 𝑌 𝑥 [∗] 𝑌<br>𝑌𝑥,𝑚 = 𝑌𝑥,𝑀𝑥 𝑌𝑥 ∗,𝑚 ∗ = 𝑌𝑥 ∗,𝑀𝑥∗<br>(A) (B)<br>𝑥 [∗] 𝑚 [∗]<br>𝑥 𝑌<br>𝑌 = 𝑌<br>𝑥,𝑚 [∗] 𝑥,𝑀𝑥∗<br>(C)<br>**----- End of picture text -----**<br>


**Figure 2: The general causal graph and commonly used counterfactual notations.** 

KDD ’24, August 25–29, 2024, Barcelona, Spain. 

Dacao Zhang, Kun Zhang, Le Wu, Mi Tian, Richang Hong, Meng Wang. 

This section explains the causal graph and causal effect calculation mentioned, aiming to help readers understand the significance and importance of the causal graph. 

**Causal Graph.** The causal graph is a Directed Acyclic Graph (DAG) G = _<_ V _,_ E _>_ , which describes the causal relationships between different variables. V is the node set and E is the edge set. As illustrated in Figure 2(A), the arrow indicates the direction of causality. For example, _𝑋_ → _𝑌_ denotes that variable _𝑋_ has a direct effect on _𝑌_ . _𝑋_ → _𝑀_ → _𝑌_ denotes that variable _𝑋_ has indirect effect on _𝑌_ through mediator _𝑀_ . Following these notations, assume _𝑋_ = _𝑥_ , then the value of _𝑌_ can be calculated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0004-04.png)


where the value of mediator _𝑀_ can be calculated with _𝑚_ = _𝑀𝑥_ = _𝑀_ ( _𝑋_ = _𝑥_ ). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0004-06.png)


**----- Start of picture text -----**<br>
𝒖 [∗]<br>𝒖 𝑼𝐟 𝑼∗𝒇<br>𝜽 𝑨 [∗] 𝜽<br>𝑨 𝑼𝐝 𝑼𝐝<br>𝑨<br>Total Effect Direct Effect<br>(A) (B)<br>**----- End of picture text -----**<br>


**Figure 3: The causal graph used in our proposed** _**PSCRF**_ **.** 

**Causal Effect Calculation.** The causal effect is a comparison of the potential outcomes of giving two different interventions to the same variable. As shown in Figure 2(A)-(B), assume that _𝑋_ = _𝑥_ is the treatment, then _𝑌𝑥,𝑚_ denotes the potential outcome of the treatment. Similarly, _𝑋_ = _𝑥_[∗] is the no treatment, then _𝑌𝑥_[∗] _,𝑚_[∗] is the potential outcome of no treatment. Along this line, the causal effect can be calculated as follows: 

_𝐸𝑓𝑓𝑒𝑐𝑡_ = _𝑌𝑥,𝑚_ − _𝑌𝑥_[∗] _,𝑚_[∗] = _𝑃_ ( _𝑦_ | _do_ ( _𝑋_ = _𝑥_ )) − _𝑃_ ( _𝑦_ | _do_ ( _𝑋_ = _𝑥_[∗] )) _,_ (2) where _do_ ( _𝑋_ = _𝑥_ ) and _do_ ( _𝑋_ = _𝑥_[∗] ) is the intervention to the variable _𝑋_ . Note that _do_ (·) operation requires that only the treatment variable is intervened, all other variables are not intervened. To satisfy this requirement, the counterfactual operation is proposed. 

**Counterfactual Notation.** The counterfactual reason is to assume a scenario that all the other variables remain unchanged and only the treatment variable is changed. The causal effect of the treatment variable on the output variable can be calculated in this scenario. For example, in Figure 2(C), _𝑌𝑥,𝑀𝑥_ ∗ = _𝑌_ ( _𝑋_ = _𝑥, 𝑀_ = _𝑀_ ( _𝑋_ = _𝑥_[∗] )) denotes a typical counterfactual reasoning. 

## **3.2 PISA Data Introduction** 

As mentioned above, the educational context of students can be used to obtain better student representations, which refer to the various features related to students’ learning process [56]. OECD’s Programme for International Student Assessment (PISA) focused on this topic and designed multiple questions to investigate and collect these educational contexts, such as family wealth, education degree of parents, and so on. For example, when investigating the highest education degree of students’ parents, five options (e.g., 1-General senior, 2-Vocational senior, 3-Junior) are provided. Students can select one option based on their situations. This information can be used as student attributes. Table 2 lists some examples of these questions. Moreover, this organization has developed exercises to 

measure 15-year-olds’ ability to employ reading, mathematics, and science knowledge and skills to meet real-life challenges [31]. They investigated students from different countries, and released the data and technical reports on a three-year cycle, which is suitable for student attribute-aware cognitive diagnosis. 

**Table 2: Educational Context examples from PISA dataset** 

Aspect Question Examples Home Economic, Social and Cultural Status (ESCS) Home Highest education degree of parents Number of equipment, appliances, and rooms Whether students have a grade repetition experience Person how many days did students engage in out-school activities With the guidance of technical report [31], we select ESCS index as the sensitive attribute example to tackle the problem in Section 1. Moreover, we leverage the Pearson Correlation Coefficient to select the useful but not sensitive attributes: _1) The number of books, 2) The number of tablet computers, 3) A link to the Internet, 4) A computer can be used for school work, 5) The number of E-book readers._ These selected attributes all exhibit strong correlations with ESCS index, which we have reported the results in Table 8 and Table 9 in the Appendix. However, they are not sensitive attributes and directly contribute to the development of students’ abilities, which should be considered in the diagnosis process. 

**Table 3: Notations and explanations in our proposed** _**PSCRF**_ **.** 

|Notation|Explanation|
|---|---|
|_𝒖, 𝒖_∗<br>_𝐴,𝐴_∗<br>_𝑼𝑓, 𝑼_∗<br>_𝑓_<br>_𝑼𝑑_<br>_𝛼_<br>_𝜙𝑒_<br>_𝜽, 𝜽_∗<br>_𝛽_<br>_𝜽𝑑_|Student ID embeddings and their counterfactuals<br>sensitive attribute and counterfactual sensitive attribute<br>The diagnosis-related features and their counterfactuals<br>The fairness-related feature<br>Learnable parameters for integrating_𝑼𝑓_and_𝑼𝑑_<br>Parameters related to exercises(e.g., difculty, discrimination)<br>The student profciency level and their counterfactuals<br>Learnable parameters controlling the degree of debiasing<br>The fairness-aware student profciency level|



## **4 TECHNICAL DETAILS OF** _**PSCRF**_ 

## **4.1 Causal view of** _**PSCRF**_ 

Based on the motivation in Section 1, we use the causal graph in Figure 3(A) to describe the causal relation among different paths. Specifically, _𝒖_ denotes the general representation of one student (i.e., ID embeddings). _𝑨_ is the corresponding sensitive attribute representation (e.g., ESCS embeddings). We argue that sensitive attributes contain fairness-related sensitive features and diagnosisrelated features. The former should not be used in the diagnosis process while the latter should be used to improve the diagnosis performance. Therefore, we leverage the causal path _𝑨_ → _𝑼𝑑_ → _𝜽_ to denote the effect of fairness-related sensitive features, which should be removed from the entire graph. A toy example is that family wealth should not be considered in the diagnosis process since it will introduce unfairness to vulnerable groups. 

Meanwhile, we use the causal path ( _𝒖, 𝑨_ ) → _𝑼𝑓_ → _𝜽_ to denote the effect of diagnosis-related features from sensitive attributes and general information. one similar example is as follows: though family wealth cannot be used in the diagnosis process, we can exploit _the number of books_ or _a link to the Internet_ to better model student 

KDD ’24, August 25–29, 2024, Barcelona, Spain. 

Path-Specific Causal Reasoning for Fairness-aware Cognitive Diagnosis 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0005-02.png)


**----- Start of picture text -----**<br>
Input data Embedding module Representation Learning (Section 4.2.1) Multi-factor Fairness<br>Exercise 𝒖 𝑼𝒇 Constraint (Section 4.2.2)<br>0 ? ? 1 ?<br>𝜃 Training Loss<br>? 0 ? ? 1 𝑒𝑛𝑐𝑜𝑑𝑒𝑟<br>? 0 ? 1 ? Students 𝑨 𝑼𝒅<br>? 1 ? ? ?<br>𝐿𝑐𝑜𝑛𝑠 𝐿𝑐𝑒<br>Response Logs Decoupled Predictor (Section 4.2.2)<br>1: correct      0: wrong?: no response Sensitive 𝑼𝒇 Non-sensitive prediction 𝐿𝑐𝑙𝑠<br>Sensitive Non-Sensitive Attribute 𝑼𝒅 sensitive prediction 0.2 0.3 0.1 0.7 0.7 0.9<br>1 0 0 1 0 𝐿𝑟𝑒𝑣 Disadv General Adv<br>0 1 0 0 1<br>10 00 01 01 10 𝑼∗𝒇 Path-specific Causal Reasoning (Section 4.2.2) Predicted results  𝒚 0.7 0.2 0.9 0.3<br>Exercises<br>SensitiveAttributeOne-hot Non-sensitiveAttribute labels 𝜙𝑒 𝜃 𝜃 [∗] 𝜃𝑑<br>CD Models<br>1:active   0: inactive<br>Student<br>MLP<br>Student Student<br>MLP<br>sensitive NSP<br>SP<br>Student<br>Exercise<br>Free embeddings<br>+<br>+<br>**----- End of picture text -----**<br>


**Figure 4: The overall framework of our proposed** _**PSCRF**_ 

proficiency level since they directly contribute to the student’s ability development. Based on this causal graph, we then introduce the corresponding implementation. 

According to the principle of Average Treatment Effect (ATE), we can calculate the Total Effect (TE) of all input variables to the output prediction as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0005-06.png)


Next, we intend to calculate the effect of only fairness-aware sensitive features on the output prediction. For this target, we employ the Natural Direct Effect (NDE) as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0005-08.png)


Finally, we can obtain the debiased prediction by calculating the Total Indirect Effect (TIE) as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0005-10.png)


By maximizing the Total Indirect Effect (TIE) inference, we can achieve debiased learning in the diagnosis process. 

## **4.2 Causal Implementation:** _**PSCRF**_ 

Figure 4 illustrates the overall architecture of _PSCRF_ , which consists of two main components: _Representation Learning Module_ and _Decoupled and Constraint Module_ . Next, we introduce each component in detail. Table 3 explains the notations in _PSCRF_ . 

_4.2.1 Representation Learning Module._ Similar to previous works [41, 56], we first embed input entities into latent embeddings, which including student embeddings _𝑼_ = [ _𝒖_ 1 _, 𝒖_ 2 _, ..., 𝒖𝑁_ ][⊤] ∈ R _[𝑁]_[×] _[𝑑]_ , exercise embeddings _𝑽_ = [ _𝒗_ 1 _, 𝒗_ 2 _, ..., 𝒗𝑀_ ][⊤] ∈ R _[𝑀]_[×] _[𝑑]_ , and sensitive attribute embeddings _𝑨_ = [ _𝒂_ 1 _, 𝒂_ 2 _, ..., 𝒂𝑇_ ][⊤] ∈ R _[𝑇]_[×] _[𝑑]_ . _𝑑_ is the embedding dimension. These embeddings are randomly initialized and will be updated during model learning. 

After obtaining the embeddings of different entities, we intend to learn the representations of fairness-related sensitive features _𝑼𝑑_ and diagnosis-related features _𝑼𝑓_ , which is realized by our designed proficiency modeling module. Specifically, we employ a fairness-related sensitive feature generator to obtain _𝑼𝑑_ , which 

takes sensitive attribute embeddings as input and uses a MultiLayer Perceptron (MLP) to generate representations as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0005-17.png)


where _𝜎_ (·) represents the sigmoid function, _𝑨_ [ _𝑖_ ] is the sensitive attribute embedding set for the _𝑖[𝑡ℎ]_ student. _𝑼𝑑[𝑖]_[is][the][sensitive-] related feature representation of the _𝑖[𝑡ℎ]_ student. 

Similarly, a diagnosis-related feature extractor is used to generate _𝑼𝑓_ with student ID embeddings and sensitive attribute embeddings. Another MLP is used to obtain the representations: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0005-20.png)


where _concat_ (·) denotes the concatenation operation. _𝒖𝑖_ is the free ID embedding of the _𝑖[𝑡ℎ]_ student. 

After obtaining the results from two extractors, we leverage a learnable parameter _𝛼_ to fuse these two different features and generate student proficiency level _𝜽𝑖_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0005-23.png)


_4.2.2 Decoupled and Constraint Module._ In the above module, we aim to extract fairness-related sensitive features and diagnosisrelated features from input data. However, there are no explicit supervised signals. which poses a big challenge. In response, we design the following three modules. Next, we omit the student index _𝑖_ for simplicity and introduce how we construct each module. 

**(1) Decoupled Predictor (DP).** First, we intend diagnosisrelated feature embedding _𝑼𝑓_ should include useful information from sensitive attributes and maintain general information of students. Therefore, we leverage the educational context as the guidance, and identify top-k educational context questions most correlated with the sensitive attribute by computing the Pearson correlation coefficient between them. These educational contexts often relate to students’ learning environments and do not involve sensitive attributes (e.g., the number of books), which can be used to enhance the student proficiency level modeling. Thus, we leverage 

KDD ’24, August 25–29, 2024, Barcelona, Spain. 

Dacao Zhang, Kun Zhang, Le Wu, Mi Tian, Richang Hong, Meng Wang. 

selected educational contexts as the prediction targets and formulate the optimization target as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0006-03.png)


Where _𝐾_ represents the total number of non-sensitive attributes, _𝐿𝑎𝑏𝑒𝑙𝑘_ represents the label of the _𝑘_[th] non-sensitive attribute, and CE denotes the cross-entropy function. 

Meanwhile, to ensure _𝑼𝑑_ to focus only on the fairness-related sensitive features, we develop a novel sensitive attribute enhancement module. We first send _𝑼𝑑_ to an MLP to predict sensitive attributes, so that _𝑼𝑑_ can be better learned to represent sensitive attributes. Meanwhile, _𝑼𝑓_ should not contain these sensitive attributes. Thus, we send _𝑼𝑓_ to the same MLP to predict the counterfactual sensitive attributes. Therefore, _𝑼𝑑_ can only encode fairness-related sensitive features that should not be used in the diagnosis process. _𝑼𝑓_ will not include these sensitive features, which is in favor of fairness-aware diagnosis. This process can be formulated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0006-06.png)


where L represents the loss function, which can be mean squared error (MSE) for continuous values or cross-entropy for discrete values. _SMLP_ (·) denotes the shared MLP. _𝐴_[∗] represents the counterfactual sensitive attribute label, which we will give a detailed explanation in the next section. 

**(2) Path-specific Causal Reasoning.** According to Figure 3 and Section 4.1, we need to remove the path _𝑨_ → _𝑼𝑑_ → _𝜽_ to realize the fairness-aware diagnosis. Following the principle of Average Treatment Effect (ATE), we need to imagine a counterfactual world, which is shown in Figure 3(B). In the counterfactual world, only the path _𝑨_ → _𝑼𝑑_ → _𝜽_ remains unchanged. We need to block the effect of the path ( _𝒖, 𝑨_ ) → _𝑼𝑓_ → _𝜽_ . Thus, we use the counterfactual sensitive attributes to modify Eq.(7) and Eq.(8) as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0006-09.png)


where { _𝒖_[∗] _, 𝑼𝑑_[∗][}][ are the counterfactual student representation and] counterfactual fairness-related sensitive feature representation. For implementation, we use the mean representation of all student representations to realize _𝒖_[∗] , and use the mean representation of corresponding sensitive attributes to calculate _𝑼𝑑_[∗][.][According][to] the causal inference, this intervention can help _PSCRF_ to calculate the accurate effect of path _𝑨_ → _𝑼𝑑_ → _𝜽_ . Then, we can obtain fairness-aware student proficiency level as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0006-11.png)


where _𝛽_ is a learnable parameter to control the degree of debiasing. _𝜽𝑑_ is used to realize the fairness-aware CD modeling. 

**(3) Multi-factor Fairness Constraint.** To facilitate better fairnessaware diagnosis, we introduce a Multi-factor Fairness Constraint. Specifically, we partition students into disadvantaged, general, and advantaged groups based on the value of sensitive attributes. Since _𝜽𝑑_ is the fairness-aware student proficiency level representation, we intend the variance of the predicted means for different groups to be as low as possible. Meanwhile, since _𝑼𝑑_ should incorporate unwanted sensitive information as much as possible, we maximize 

the variance of the predicted means for different groups. Therefore, this constrain can be realized as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0006-15.png)


where _𝑦_ dis, _𝑦_ gene and _𝑦_ adv respectively denote the predicted means of three groups, while _std_ (·) represents the variance. 

## **4.3 Model Training** 

To ensure prediction accuracy, we also add traditional cross-entropy constraints to _𝑼𝑓_ , _𝑼𝑑_ , _𝜽_ and _𝜽𝑑_ . First, we input them into the CD model to obtain prediction results: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0006-19.png)


whereˆ _𝐶𝐷𝑀_ (·) represents the CD models such as NCD or KaNCD. _𝜽_ can be _𝑼𝑓_ , _𝑼𝑑_ , _𝜽_ or _𝜽𝒅_ . _𝝓𝑒_ represents the parameters related to exercises(e.g., difficulty, discrimination). Then we minimize the cross-entropy loss between the predictions and labels: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0006-21.png)


where Θ = { _𝑼𝑓_ , _𝑼𝑑_ , _𝜽_ , _𝜽𝒅_ }. _𝑦_ is the true label. Finally, total loss is: 

Ltotal = _𝑤_ 1Lce + _𝑤_ 2Lcls + _𝑤_ 3Lrev + _𝑤_ 4Lcons _,_ (16) where _𝑤_ 1, _𝑤_ 2, _𝑤_ 3, and _𝑤_ 4 represent hyperparameters that balance the weights of each part of the loss. 

## **5 EXPERIMENTS 5.1 Experimental Setup** 

**Data preprocessing.** We procured two prototypical datasets from the PISA-2015, representing Australia and Brazil, respectively, meticulously arranged in descending order based on their developmental status and the mean scholastic attainment of students[32]. In each dataset, there are 28 different self-acquired features [3], such as learning interests and self-efficacy. We used two representative sensitive attributes, namely ESCS (Index of Economic, Social, and Cultural Status) and the father’s education level [31] to evaluate our method. Specifically, based on the data provided by PISA [31], we categorized students into three groups - _disadvantaged, general, and advantaged_ - according to their sensitive attributes, for fairness-aware diagnosis. We filtered out students with fewer than 10 exercise records to ensure sufficient data for training. The Basic statistics of datasets are shown in Table 4. For each dataset, we performed a 70%/10%/20% training/validation/testing split. 

**Table 4: The Statistics of datasets** 

|Dataset|Students|Exercises|Exercise Records|
|---|---|---|---|
|Australia<br>Brazil|8,485<br>5,777|184<br>183|249,727<br>143,314|



**Evaluation Metrics.** Based on the target, we select two types of metrics. For diagnosis performance, following previous works [15, 42], we used widely used metrics: Area Under Curve (AUC) and Accuracy (ACC). Meanwhile, following the work [41], we also use the Degree of Agreement (DOA) for validation. 

For fairness performance, since the abuse of sensitive attributes will mislead models to underestimate or overestimate the students 

KDD ’24, August 25–29, 2024, Barcelona, Spain. 

Path-Specific Causal Reasoning for Fairness-aware Cognitive Diagnosis 

**Table 5: Evaluating accuracy and fairness performance associated with sensitive attribute ESCS** 

||**Model**|**Model**|**Australia**<br>**EO**↓<br>**D**_under_<br>_disadv_<br>**IR**↑<br>**AUC**↑<br>**ACC**↑<br>**DOA**↑|**Brazil**<br>**EO**↓<br>**D**_under_<br>_disadv_<br>**IR**↑<br>**AUC**↑<br>**ACC**↑<br>**DOA**↑|
|---|---|---|---|---|
||**IRT**|Base<br>Base†|0.0338<br>0.0826<br>0.7353<br>0.7979<br>0.7266<br>-<br>0.0604<br>0.1473<br>0.7025<br>**0.8080**<br>**0.7322**<br>-|0.0582<br>0.1407<br>0.5018<br>0.7794<br>0.7269<br>-<br>0.1025<br>0.2510<br>0.4700<br>**0.7958**<br>**0.7324**<br>-<br>0.0277<br>0.0665<br>0.5301<br>0.7769<br>0.7250<br>-<br>0.0669<br>0.1609<br>0.4935<br>0.7797<br>0.7268<br>-<br>**0.0162**<br>**0.0357**<br>**0.5760**<br>0.7893<br>0.7255<br>-<br>0.0913<br>0.2227<br>0.5109<br>0.7836<br>0.7280<br>-<br>0.1251<br>0.3053<br>0.4663<br>**0.7950**<br>**0.7316**<br>-<br>0.0512<br>0.1246<br>**0.5539**<br>0.7813<br>0.7258<br>-<br>0.0956<br>0.2335<br>0.5036<br>0.7840<br>0.7283<br>-<br>**0.0279**<br>**0.0403**<br>0.5248<br>0.7804<br>0.7205<br>-<br>0.0669<br>0.1588<br>0.5220<br>0.7675<br>0.7140<br>0.5972<br>0.1274<br>0.3108<br>0.4491<br>0.7718<br>0.7166<br>0.6394<br>0.0522<br>0.1229<br>0.5370<br>0.7669<br>0.7131<br>0.5965<br>0.0506<br>0.1234<br>0.5388<br>0.7601<br>0.7112<br>0.5648<br>**0.0030**<br>**0.0028**<br>**0.5599**<br>**0.7788**<br>**0.7209**<br>**0.6806**<br>0.0742<br>0.1792<br>0.4877<br>0.7793<br>0.7221<br>0.6046<br>0.1210<br>0.2963<br>0.5103<br>**0.7910**<br>**0.7284**<br>**0.6848**<br>0.0464<br>0.1115<br>0.5138<br>0.7775<br>0.7207<br>0.6015<br>0.0686<br>0.1664<br>0.5388<br>0.7802<br>0.7244<br>0.6357<br>**0.0363**<br>**0.0888**<br>**0.5145**<br>0.7892<br>0.7267<br>0.6840|
|||Reg<br>Adv<br>_PSCRF_|0.0110<br>0.0270<br>**0.7544**<br>0.7961<br>0.7249<br>-<br>0.0286<br>0.0697<br>0.7449<br>0.7969<br>0.7264<br>-<br>**0.0051**<br>**0.0002**<br>0.7339<br>0.8022<br>0.7249<br>-||
||**MIRT**|Base<br>Base†|0.0575<br>0.1408<br>0.7013<br>0.8027<br>0.7299<br>-<br>0.0645<br>0.1523<br>0.6973<br>**0.8088**<br>**0.7339**<br>-||
|||Reg<br>Adv<br>_PSCRF_|0.0284<br>0.0694<br>0.7279<br>0.8010<br>0.7278<br>-<br>0.0554<br>0.1357<br>0.7009<br>0.8030<br>0.7288<br>-<br>**0.0098**<br>**0.0227**<br>**0.7520**<br>0.7983<br>0.7237<br>-||
||**NCD**|Base<br>Base†|0.0425<br>0.1040<br>0.7183<br>0.7868<br>0.7170<br>0.6248<br>0.0857<br>0.2039<br>0.6615<br>0.7911<br>0.7199<br>0.6384||
|||Reg<br>Adv<br>_PSCRF_|0.0331<br>0.0811<br>0.7277<br>0.7863<br>0.7172<br>0.6245<br>0.0528<br>0.1292<br>0.6644<br>0.7801<br>0.7111<br>0.5715<br>**0.0029**<br>**0.0010**<br>**0.7538**<br>**0.7997**<br>**0.7234**<br>**0.7040**||
||**KaNCD**|Base<br>Base†|0.0464<br>0.1133<br>0.7113<br>0.8017<br>0.7273<br>0.6584<br>0.0770<br>0.1878<br>0.6957<br>**0.8076**<br>**0.7310**<br>0.6917||
|||Reg<br>Adv<br>_PSCRF_|0.0255<br>0.0622<br>0.7299<br>0.8004<br>0.7260<br>0.6552<br>0.0532<br>0.1303<br>0.7075<br>0.8009<br>0.7282<br>0.6615<br>**0.0110**<br>**0.0252**<br>**0.7484**<br>0.8045<br>0.7299<br>**0.7013**||



from different groups, commonly used fairness metrics are used. We first employ Equal opportunity (EO) [18]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0007-05.png)


_𝑇𝑃𝑅_ refers to True Positive Rates. _Std_ (·) is the standard deviation. Since there are only two situations (i.e., correct and incorrect) for students answering questions, CD models should have equal capability of predicting the probability of students answering exercises correctly across different groups. Along this line, sensitive attributes can be proved to be not used in the diagnosis process. Moreover, since predictions of CD models have a big social influence in realworld scenarios, we argue that the rights of vulnerable groups should be guaranteed. We should not be prejudiced against disadvantaged groups and assume that they will perform less well. Based on the principle of Equalized Odds [18], we propose the following evaluation metric to evaluate the fairness performance of models: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0007-07.png)


where { _𝐹𝑁𝑅𝑑𝑖𝑠𝑎𝑑𝑣_ , _𝐹𝑁𝑅𝑎𝑑𝑣_ } denote the False Negative Rates (FNRs) of disadvantaged and advantaged groups. The closer the value of _𝐷[𝑢𝑛𝑑𝑒𝑟] 𝑑𝑖𝑠𝑎𝑑𝑣_[is to 0, the better fairness performance the model is.] 

Meanwhile, for disadvantaged groups, we should also identify the top students as accurately as possible, so that they have opportunities to access higher levels of education. Thus, we select the absolute metric F2-score [11] to assess the proportion of highachieving students of disadvantaged groups, which we name as 

Identified Rate (IR) and formulate as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0007-11.png)


where _𝑝𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛𝑑𝑖𝑠𝑎𝑑𝑣_ and _𝑟𝑒𝑐𝑎𝑙𝑙𝑑𝑖𝑠𝑎𝑑𝑣_ denote the Precision and Recall of disadvantaged group. Note that the larger the value of _𝐼𝑅_ is, the better performance the model has. 

**Implementation Details.** As our proposed method is modelagnostic, we apply _PSCRF_ to four advanced CDMs to show its effectiveness and flexibility. Moreover, we compare with several recent approaches, including regularization-based methods and adversarial-based methods: _1) Base_ : Basic cognitive diagnosis models (i.e., IRT, MIRT, NCD, and KaNCD) that do not consider bias; _2) 𝐵𝑎𝑠𝑒_[†] : Basic Models with Sensitive Attributes; _3) Reg_ : Regularizationbased models, by adding Equal opportunity as a regularization to CDMs [18, 24]; _4) Adv_ : Adversarial learning methods [4] to reduce relevance between sensitive attributes and student representation 

For implementation, we set the learning rate to 0 _._ 001 and batch size to 512. We apply Adam as the optimization algorithm to update the model parameters. To obtain the best performance, we tune hyper-parameters on validation sets to select the best. The balance parameters in Eq.(16) are set to 1 _._ 0 _,_ 0 _._ 1 _,_ 0 _._ 5, and 1 _._ 0, respectively. 

## **5.2 Overall Experiments** 

Table 5 and Table 6 report overall results under different sensitive attributes. We observe all basic CDMs suffered from unfair outcome issues. Besides, _𝐵𝑎𝑠𝑒_[†] incorporates sensitive attributes into model learning, achieving better diagnosis performance. Meanwhile, it 

KDD ’24, August 25–29, 2024, Barcelona, Spain. 

Dacao Zhang, Kun Zhang, Le Wu, Mi Tian, Richang Hong, Meng Wang. 

**Table 6: Evaluating accuracy and fairness performance associated with sensitive attribute Father’s education level.** 

||**Model**|**Model**|**Australia**<br>**EO**↓<br>**D**_under_<br>_disadv_<br>**IR**↑<br>**AUC**↑<br>**ACC**↑<br>**DOA**↑|**Brazil**<br>**EO**↓<br>**D**_under_<br>_disadv_<br>**IR**↑<br>**AUC**↑<br>**ACC**↑<br>**DOA**↑|
|---|---|---|---|---|
||**IRT**|Base<br>Base†|0.0293<br>0.0705<br>0.7346<br>0.7979<br>0.7266<br>-<br>0.0440<br>0.1069<br>0.6980<br>**0.8087**<br>**0.7312**<br>-|0.0366<br>0.0896<br>0.5209<br>0.7794<br>0.7269<br>-<br>0.0720<br>0.1723<br>0.5379<br>**0.7959**<br>**0.7314**<br>-<br>0.0190<br>0.0465<br>0.5468<br>0.7781<br>0.7262<br>-<br>0.0314<br>0.0716<br>0.5404<br>0.7752<br>0.7242<br>-<br>**0.0021**<br>**-0.0049**<br>**0.5640**<br>0.7911<br>0.7274<br>-<br>0.0572<br>0.1398<br>0.5472<br>0.7836<br>0.7280<br>-<br>0.0849<br>0.1820<br>0.4931<br>**0.7896**<br>0.7279<br>-<br>0.0300<br>0.0735<br>0.5787<br>0.7812<br>0.7272<br>-<br>0.0528<br>0.1292<br>0.5548<br>0.7821<br>**0.7282**<br>-<br>**0.0247**<br>**0.0422**<br>**0.5879**<br>0.7827<br>0.7214<br>-<br>0.0428<br>0.1042<br>0.5409<br>0.7675<br>0.7140<br>0.5972<br>0.0855<br>0.1745<br>**0.6095**<br>0.7785<br>0.7026<br>0.6518<br>0.0324<br>0.0794<br>0.5396<br>0.7689<br>0.7145<br>0.6010<br>0.0484<br>0.1177<br>0.5470<br>0.7635<br>0.7150<br>0.5722<br>**0.0280**<br>**0.0465**<br>0.5745<br>**0.7879**<br>**0.7293**<br>**0.6889**<br>0.0433<br>0.1058<br>0.5189<br>0.7793<br>0.7221<br>0.6046<br>0.0555<br>0.1302<br>0.5434<br>0.7862<br>0.7256<br>0.6731<br>**0.0288**<br>**0.0699**<br>**0.5826**<br>0.7799<br>0.7242<br>0.6347<br>0.0419<br>0.1020<br>0.5609<br>0.7802<br>0.7239<br>0.6352<br>0.0340<br>0.0746<br>0.5130<br>**0.7930**<br>**0.7278**<br>**0.6847**|
|||Reg<br>Adv<br>_PSCRF_|**0.0132**<br>0.0303<br>**0.7515**<br>0.7969<br>0.7255<br>-<br>0.0231<br>0.0546<br>0.7319<br>0.7919<br>0.7200<br>-<br>0.0162<br>**0.0015**<br>0.7342<br>0.8034<br>0.7277<br>-||
||**MIRT**|Base<br>Base†|0.0437<br>0.1051<br>0.7084<br>0.8027<br>0.7299<br>-<br>0.0554<br>0.1326<br>0.7195<br>**0.8106**<br>**0.7347**<br>-||
|||Reg<br>Adv<br>_PSCRF_|**0.0194**<br>**0.0449**<br>0.7383<br>0.8025<br>0.7297<br>-<br>0.0389<br>0.0947<br>0.7110<br>0.8043<br>0.7316<br>-<br>**0.0194**<br>0.0474<br>0.7095<br>0.8073<br>0.7285<br>-||
||**NCD**|Base<br>Base†|0.0313<br>0.0747<br>0.7265<br>0.7868<br>0.7170<br>0.6248<br>0.0477<br>0.1147<br>0.6981<br>**0.8021**<br>0.7263<br>0.6478||
|||Reg<br>Adv<br>_PSCRF_|0.0293<br>0.0679<br>0.6940<br>0.7834<br>0.7119<br>0.6167<br>0.0323<br>0.0789<br>0.6791<br>0.7825<br>0.7130<br>0.5918<br>**0.0227**<br>**-0.0116**<br>**0.7362**<br>0.8003<br>**0.7276**<br>**0.7096**||
||**KaNCD**|Base<br>Base†|0.0370<br>0.0887<br>0.7146<br>0.8017<br>0.7273<br>0.6584<br>0.051<br>0.1207<br>0.6938<br>**0.8084**<br>**0.7310**<br>**0.7183**||
|||Reg<br>Adv<br>_PSCRF_|0.0251<br>0.0578<br>0.7364<br>0.8010<br>0.7274<br>0.6642<br>0.0405<br>0.0972<br>0.7144<br>0.8006<br>0.7278<br>0.6618<br>**0.0114**<br>**0.0275**<br>**0.7768**<br>0.8066<br>0.7269<br>0.7097||



also exhibits more severe unfairness. All these suggest the urgency to explore fairness-aware learning in CD models. Therefore, we evaluate the model performance from the following three aspects: For _fairness performance_ (i.e., EO and _𝐷disadv[under]_[), we observe that] _PSCRF_ outperforms most baselines, proving its effectiveness. Moreover, Adversarial-based methods (Adv) show worse performance than regularization-based (Reg) methods in most cases. One possible reason is that the latter directly utilizes sensitive attribute group labels to optimize corresponding metrics. In contrast, _PSCRF_ decouples sensitive attributes and only eliminates the fairness-related sensitive features that should not be used in diagnosis process. 

For the _trade-off between fairness and diagnosis performance_ , all debiased baselines perform worse on diagnosis performance than _PSCRF_ , proving that they cannot retain diagnosis-related features from sensitive attributes, which causes a larger decrease in diagnosis accuracy. _PSCRF_ uses the newly designed DP module and multi-factor constraint to retain diagnosis-related features as much as possible, thus outperforming baselines. 

For the _Identification capability_ (i.e., IR) of high-achieving students from disadvantaged groups, _PSCRF_ still achieves impressive performance, which proves that _PSCRF_ can effectively ensure the rights of vulnerable groups. Moreover, by considering the results on _𝐷disadv[under]_[metric, we can conclude that for different CD backbones,] _PSCRF_ can effectively balance fairness and diagnosis performance, demonstrating the flexibility and effectiveness of _PSCRF_ . 

## **5.3 Ablation Study** 

To verify the effectiveness of each component, we conduct an ablation study with ESCS and report results in Table 7. From the results, when only using Lce, _𝑼𝑓_ and _𝑼𝑑_ cannot decouple sensitive attributes effectively. Instead, they would introduce more bias, resulting in better diagnosis and worse fairness performances. When separately introducing Lcls and Lrev, we observe improvements in fairness performance with minimal impact on accuracy, proving the effectiveness of learned _𝑼𝑓_ and _𝑼𝑑_ . When using only Lcons, _PSCRF_ can learn how to remove fairness-aware sensitive features, achieving a substantial improvement in fairness. However, since _𝑼𝑓_ still contains some factors affecting fairness, this component can only generate a suboptimal outcome. Moreover, when removing each component, we observe varying degrees of performance degradation. Among all components, removing Lcons causes the most significant decrease in fairness performance and increase in diagnosis performance. This phenomenon not only proves the importance of multi-factor fairness constraint, but also shows _PSCRF_ will abuse sensitive attributes to improve diagnosis performance. Moreover, Lcons[∗] denotes the removal of constraints on _𝑼𝑑_ for direct comparison with regularization methods. The result proves that the absence of constraints on _𝑼𝑑_ severely impacts fairness performance, aligning its performance comparably with regularization methods. Furthermore, we observe that Lcls and Lrev have a positive impact on fairness performance, removing them will cause a decrease in fairness performance. In conclusion, these components are all necessary for the superiority of _PSCRF_ . 

KDD ’24, August 25–29, 2024, Barcelona, Spain. 

Path-Specific Causal Reasoning for Fairness-aware Cognitive Diagnosis 

**Table 7: Ablation Study of IRT on the Australia Dataset** 

|Conditions|**EO**↓|**D**_under_<br>_disadv_|**IR**↑|**AUC**↑|**ACC**↑|
|---|---|---|---|---|---|
|Base|0.0338|0.0826|0.7353|0.7979|0.7266|
|_PSCRF_|**0.0051**|**0.0002**|0.7339|0.8022|0.7249|
|w Lce|0.0545|0.1335|0.6885|0.8089|0.7329|
|w Lcls|0.0503|0.1231|0.6982|0.8088|0.7328|
|w Lrev|0.0525|0.1287|0.6919|0.8090|0.7332|
|w Lcons|0.0088|0.0069|0.7392|0.8016|0.7250|
|w/o Lcls|0.0137|0.0318|0.7206|0.8057|0.7279|
|w/o Lrev|0.0112|-0.0057|0.7277|0.8022|0.7258|
|w/o Lcons|0.0609|0.1493|0.7127|**0.8092**|**0.7337**|
|w/o L∗cons|0.0132|-0.0322|**0.7565**|0.8021|0.7257|



## **5.4 Parameter Sensitive Test** 

To conduct a deeper analysis on _PSCRF_ , we also conduct parameter sensitive test on the weights _𝑤_ 1, _𝑤_ 2, _𝑤_ 3, _𝑤_ 4 in Eq.(16), whose results are summarized in Figure 5. According to the results, we can observe that with the increase of _𝑤_ 4, the diagnosis performance of _PSCRF_ decreases slightly while the fairness performance increases rapidly. This phenomenon is consistent with the results in Table 7, proving the importance of Lcons. Moreover, with the weight _𝑤_ 1 increasing, the diagnosis performance of _PSCRF_ increases while the fairness performance decreases rapidly and is unstable. This is intuitive since a large _𝑤_ 1 will focus more on student data and impose more biased information. With the increase of _𝑤_ 2, both diagnosis performance and fairness performance fluctuate considerably. We speculate one possible reason is that Lcls is relatively simple and its impact on _PSCRF_ is relatively small. Thus, a large weight will lead _PSCRF_ unable to learn useful information. For _𝑤_ 3, a larger value will improve the fairness performance while retaining the diagnosis performance, proving the effectiveness of Lrev. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0009-06.png)


**----- Start of picture text -----**<br>
0.81 AUCEO 0.03 0.81 AUCEO 0.03<br>0.02<br>0.80 0.02 0.80<br>0.01<br>0.79 0.01<br>0.79 0.00<br>0.00<br>0.78<br>0.2 0.4 0.6 0.8 0.2 0.4 0.6 0.8<br>w1 w2<br>0.815<br>AUC 0.02 0.815 AUC<br>EO EO 0.04<br>0.810<br>0.810 0.03<br>0.805 0.01<br>0.805 0.02<br>0.800 0.01<br>0.00 0.800<br>0.795 0.00<br>0.795<br>0.2 0.4 0.6 0.8 0.2 0.4 0.6 0.8<br>w3 w4<br>**----- End of picture text -----**<br>


**Figure 5:** _**PSCRF**_ **based on IRT with varying** _𝑤_ 

## **5.5 Case Study** 

To better illustrate the effectiveness of _PSCRF_ , we visualize the prediction distributions of different models using FNR (underestimate rate) and FPR (overestimate rate). As shown in Figure 6, we initially observe a conspicuous prediction bias in the _𝐵𝑎𝑠𝑒_ model, characterized by underestimation for the disadvantaged group and overestimation for the advantaged group, displaying a distinct stepwise distribution between different groups. When introducing sensitive 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0009-10.png)


**----- Start of picture text -----**<br>
0.35 0.45<br>Base Base<br>0.30 BaseReg 0.40 BaseReg<br>Adv Adv<br>PSCRF 0.35 PSCRF<br>0.25<br>0.30<br>0.20<br>0.25<br>0.15 0.20<br>0.10 0.15<br>Disadv Mid Adv Disadv Mid Adv<br>(a)FNR(underestimation rate) (b)FPR(overestimation rate)<br>**----- End of picture text -----**<br>


**Figure 6: Visualization of the prediction distributions.** 

attributes, this unfairness is exacerbated (e.g., _𝐵𝑎𝑠𝑒_[†] model). _𝑅𝑒𝑔_ and _𝐴𝑑𝑣_ methods can alleviate this unfairness to some extent. However, their prediction distributions still exhibit a stepwise pattern. As a comparison, _PSCRF_ shows nearly consistent levels of underestimation and overestimation across different groups, achieving the lowest overestimation rates among the various models. A comparison with _𝐵𝑎𝑠𝑒_[†] reveals that _PSCRF_ has effectively mitigated the unfairness imposed on the disadvantaged group by a single sensitive attribute. The only shortfall is that our approach does not reduce the underestimation rates. We speculate that there might exist other sensitive attributes that are not considered. 

## **6 CONCLUSION** 

In this paper, we argued that current CD models concentrated more on the full exploitation of student-exercise interaction data, ignoring the potential risk of abuse of sensitive attributes. Moreover, we argued that student sensitive attributes could also provide useful information, so directly eliminating them was not optimal. To achieve fairness-aware CD learning, we proposed to incorporate casual inference and designed a novel _PSCRF_ . By leveraging a newly designed attribute-oriented predictor to deal with the information from sensitive attributes. _PSCRF_ decoupled sensitive attributes into fairness-related sensitive features that should not be used in the diagnosis process, and diagnosis-related features that should be used to enrich the student proficiency level modeling. Thus, _PSCRF_ could achieve impressive fairness performance while retaining the diagnosis performance. Moreover, we designed a multi-factor fairness constraint to ensure the fairness performance and diagnosis performance simultaneously. Finally, we conducted extensive experiments over real-world datasets and multiple advanced CD models to demonstrate the effectiveness of _PSCRF_ . 

## **7 ACKNOWLEDGMENTS** 

This work was supported in part by grants from the National Science and Technology Major Project (Grant No. 2021ZD0111802), Joint Funds of the National Natural Science Foundation of China (Grant No. U22A2094), and the National Natural Science Foundation of China (Grant No. 62376086, U23B2031, 721881011). 

## **REFERENCES** 

- [1] Hengyao Bao, Xihua Li, Xuemin Zhao, and Yunbo Cao. 2021. Exploring Student Representation For Neural Cognitive Diagnosis. _arXiv preprint arXiv:2111.08951_ (2021). 

- [2] Ghazaleh Beigi, Ahmadreza Mosallanezhad, Ruocheng Guo, Hamidreza Alvari, Alexander Nou, and Huan Liu. 2020. Privacy-aware recommendation with privateattribute protection using adversarial learning. In _Proceedings of the 13th International Conference on Web Search and Data Mining_ . 34–42. 

KDD ’24, August 25–29, 2024, Barcelona, Spain. 

Dacao Zhang, Kun Zhang, Le Wu, Mi Tian, Richang Hong, Meng Wang. 

- [3] Peter M Blau and Otis Dudley Duncan. 1967. The American occupational structure. (1967). 

- [4] Avishek Bose and William Hamilton. 2019. Compositional fairness constraints for graph embeddings. In _International Conference on Machine Learning_ . PMLR, 715–724. 

- [5] Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, and Xiangnan He. 2023. Bias and debias in recommender system: A survey and future directions. _ACM Transactions on Information Systems_ 41, 3 (2023), 1–39. 

- [6] Jiawei Chen, Yan Feng, Martin Ester, Sheng Zhou, Chun Chen, and Can Wang. 2018. Modeling Users’ Exposure with Social Knowledge Influence and Consumption Influence for Recommendation. In _Proceedings of the 27th ACM International Conference on Information and Knowledge Management_ . 953–962. 

- [7] Lei Chen, Le Wu, Kun Zhang, Richang Hong, Defu Lian, Zhiqiang Zhang, Jun Zhou, and Meng Wang. 2023. Improving Recommendation Fairness via Data Augmentation. In _Proceedings of the ACM Web Conference 2023_ . 1012–1020. 

- [8] Song Cheng and Qi Liu. 2019. Enhancing item response theory for cognitive diagnosis. _arXiv preprint arXiv:1905.10957_ (2019). 

- [9] Ali Daud, Naif Radi Aljohani, Rabeeh Ayaz Abbasi, Miltiadis D Lytras, Farhat Abbas, and Jalal S Alowibdi. 2017. Predicting student performance using advanced learning analytics. In _Proceedings of the 26th international conference on world wide web companion_ . 415–421. 

- [10] Jimmy De La Torre. 2009. DINA model and parameter estimation: A didactic. _Journal of educational and behavioral statistics_ 34, 1 (2009), 115–130. 

- [11] Divyaansh Devarriya, Cairo Gulati, Vidhi Mansharamani, Aditi Sakalle, and Arpit Bhardwaj. 2020. Unbalanced breast cancer data classification using novel fitness functions in genetic programming. _Expert Systems with Applications_ 140 (2020), 112866. 

- [12] Michael D Ekstrand, Mucun Tian, Mohammed R Imran Kazi, Hoda Mehrpouyan, and Daniel Kluver. 2018. Exploring author gender in book rating and recommendation. In _Proceedings of the 12th ACM conference on recommender systems_ . 242–250. 

- [13] Susan E Embretson and Steven P Reise. 2013. _Item response theory_ . Psychology Press. 

- [14] Lina Gao, Zhongying Zhao, Chao Li, Jianli Zhao, and Qingtian Zeng. 2022. Deep cognitive diagnosis model for predicting students’ performance. _Future Generation Computer Systems_ 126 (2022), 252–262. 

- [15] Weibo Gao, Qi Liu, Zhenya Huang, Yu Yin, Haoyang Bi, Mu-Chun Wang, Jianhui Ma, Shijin Wang, and Yu Su. 2021. RCD: Relation map driven cognitive diagnosis for intelligent education systems. In _Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval_ . 501–510. 

- [16] Weibo Gao, Hao Wang, Qi Liu, Fei Wang, Xin Lin, Linan Yue, Zheng Zhang, Rui Lv, and Shijin Wang. 2023. Leveraging Transferable Knowledge Concept Graph Embedding for Cold-Start Cognitive Diagnosis. In _Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval_ . 983–992. 

- [17] Sahin Cem Geyik, Stuart Ambler, and Krishnaram Kenthapadi. 2019. Fairnessaware ranking in search & recommendation systems with application to linkedin talent search. In _Proceedings of the 25th acm sigkdd international conference on knowledge discovery & data mining_ . 2221–2231. 

- [18] Moritz Hardt, Eric Price, and Nati Srebro. 2016. Equality of opportunity in supervised learning. _Advances in neural information processing systems_ 29 (2016). 

- [19] Jie Huang, Qi Liu, Fei Wang, Zhenya Huang, Songtao Fang, Runze Wu, Enhong Chen, Yu Su, and Shijin Wang. 2021. Group-level cognitive diagnosis: A multitask learning perspective. In _2021 IEEE International Conference on Data Mining (ICDM)_ . IEEE, 210–219. 

- [20] Vasileios Iosifidis, Besnik Fetahu, and Eirini Ntoutsi. 2019. Fae: A fairness-aware ensemble framework. In _2019 IEEE International Conference on Big Data (Big Data)_ . IEEE, 1375–1380. 

- [21] Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor. 2023. Resampling methods. In _An Introduction to Statistical Learning: with Applications in Python_ . Springer, 201–228. 

- [22] Stefan Konigorski. 2021. Causal inference in developmental medicine and neurology. _Developmental Medicine & Child Neurology_ 63 (2021). 

- [23] Jiatong Li, Fei Wang, Qi Liu, Mengxiao Zhu, Wei Huang, Zhenya Huang, Enhong Chen, Yu Su, and Shijin Wang. 2022. Hiercdf: A bayesian network-based hierarchical cognitive diagnosis framework. In _Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining_ . 904–913. 

- [24] Yunqi Li, Hanxiong Chen, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2021. User-oriented fairness in recommendation. In _Proceedings of the Web Conference 2021_ . 624–632. 

- [25] Qi Liu. 2021. Towards a New Generation of Cognitive Diagnosis.. In _IJCAI_ . 4961–4964. 

- [26] Shuhuan Liu, Xiaoshan Yu, Haiping Ma, Ziwen Wang, Chuan Qin, and Xingyi Zhang. 2023. Homogeneous Cohort-Aware Group Cognitive Diagnosis: A Multigrained Modeling Perspective. In _Proceedings of the 32nd ACM International Conference on Information and Knowledge Management_ . 4094–4098. 

- [27] Haiping Ma, Manwei Li, Le Wu, Haifeng Zhang, Yunbo Cao, Xingyi Zhang, and Xuemin Zhao. 2022. Knowledge-Sensed Cognitive Diagnosis for Intelligent 

Education Platforms. In _Proceedings of the 31st ACM International Conference on Information & Knowledge Management_ . 1451–1460. 

- [28] Ioana Marinescu, Patrick N. Lawlor, and Konrad Paul Kording. 2018. Quasiexperimental causality in neuroscience and behavioural research. _Nature Human Behaviour_ 2 (2018), 891–898. 

- [29] Benjamin M Marlin and Richard S Zemel. 2009. Collaborative prediction and ranking with non-random missing data. In _Proceedings of the third ACM conference on Recommender systems_ . 5–12. 

- [30] Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, Xian-Sheng Hua, and JiRong Wen. 2021. Counterfactual vqa: A cause-effect look at language bias. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 12700–12710. 

- [31] OECD. 2017. PISA 2015 Assessment and Analytical Framework. https://doi.org/ 10.1787/9789264281820-en. , 260 pages. 

- [32] OECD. 2018. PISA 2015 Results (Volume I). https://doi.org/10.1787/19963777. , 492 pages. 

- [33] Judea Pearl, Madelyn Glymour, and Nicholas P Jewell. 2016. _Causal inference in statistics: A primer_ . John Wiley & Sons. 

- [34] Chen Qian, Fuli Feng, Lijie Wen, Chunping Ma, and Pengjun Xie. 2021. Counterfactual inference for text classification debiasing. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ . 5434–5445. 

- [35] Bashir Rastegarpanah, Krishna P Gummadi, and Mark Crovella. 2019. Fighting fire with fire: Using antidote data to improve polarization and fairness of recommender systems. In _Proceedings of the twelfth ACM international conference on web search and data mining_ . 231–239. 

- [36] Ladan Shams and Ulrik R Beierholm. 2010. Causal inference in perception. _Trends in Cognitive Sciences_ 14 (2010), 425–432. 

- [37] Shuanghong Shen, Zhenya Huang, Qi Liu, Yu Su, Shijin Wang, and Enhong Chen. 2022. Assessing Student’s Dynamic Knowledge State by Exploring the Question Difficulty Effect. In _Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval_ . 427–437. 

- [38] Field Simon, Kuczera Małgorzata, and PONT Beatriz. 2007. _Education and training policy no more failures ten steps to equity in education: Ten steps to equity in education_ . oecd Publishing. 

- [39] Shiwei Tong, Qi Liu, Runlong Yu, Wei Huang, Zhenya Huang, Zachary A Pardos, and Weijie Jiang. 2021. Item Response Ranking for Cognitive Diagnosis.. In _IJCAI_ . 1750–1756. 

- [40] Wim J Van der Linden and Cees AW Glas. 2000. _Computerized adaptive testing: Theory and practice_ . Springer. 

- [41] Fei Wang, Qi Liu, Enhong Chen, Zhenya Huang, Yuying Chen, Yu Yin, Zai Huang, and Shijin Wang. 2020. Neural cognitive diagnosis for intelligent education systems. In _Proceedings of the AAAI conference on artificial intelligence_ , Vol. 34. 6153–6161. 

- [42] Fei Wang, Qi Liu, Enhong Chen, Zhenya Huang, Yu Yin, Shijin Wang, and Yu Su. 2022. NeuralCD: a general framework for cognitive diagnosis. _IEEE Transactions on Knowledge and Data Engineering_ (2022). 

- [43] Hangyu Wang, Ting Long, Liang Yin, Weinan Zhang, Wei Xia, Qichen Hong, Dingyin Xia, Ruiming Tang, and Yong Yu. 2023. GMOCAT: A Graph-Enhanced Multi-Objective Method for Computerized Adaptive Testing. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ . 2279–2289. 

- [44] Shanshan Wang, Zhen Zeng, Xun Yang, and Xingyi Zhang. 2023. Self-supervised Graph Learning for Long-tailed Cognitive Diagnosis. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 37. 110–118. 

- [45] Junfei Wu, Qiang Liu, Weizhi Xu, and Shu Wu. 2022. Bias mitigation for evidenceaware fake news detection by causal intervention. In _Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval_ . 2308–2313. 

- [46] Le Wu, Lei Chen, Pengyang Shao, Richang Hong, Xiting Wang, and Meng Wang. 2021. Learning fair representations for recommendation: A graph-based perspective. In _Proceedings of the Web Conference 2021_ . 2198–2208. 

- [47] Le Wu, Xiangnan He, Xiang Wang, Kun Zhang, and Meng Wang. 2022. A survey on accuracy-oriented neural recommendation: From collaborative filtering to information-rich recommendation. _IEEE Transactions on Knowledge and Data Engineering_ 35, 5 (2022), 4425–4445. 

- [48] Shangshang Yang, Haoyu Wei, Haiping Ma, Ye Tian, Xingyi Zhang, Yunbo Cao, and Yaochu Jin. 2023. Cognitive diagnosis-based personalized exercise group assembly via a multi-objective evolutionary algorithm. _IEEE Transactions on Emerging Topics in Computational Intelligence_ (2023). 

- [49] Mengfan Yao, Siqian Zhao, Shaghayegh Sahebi, and Reza Feyzi Behnagh. 2021. Stimuli-sensitive Hawkes processes for personalized student procrastination modeling. In _Proceedings of the Web Conference 2021_ . 1562–1573. 

- [50] Sirui Yao and Bert Huang. 2017. Beyond parity: Fairness objectives for collaborative filtering. _Advances in neural information processing systems_ 30 (2017). 

- [51] Shengjun Yin, Kailai Yang, and Hongzhi Wang. 2020. A mooc courses recommendation system based on learning behaviours. In _Proceedings of the ACM Turing_ 

KDD ’24, August 25–29, 2024, Barcelona, Spain. 

Path-Specific Causal Reasoning for Fairness-aware Cognitive Diagnosis 

## _Celebration Conference-China_ . 133–137. 

- [52] Renzhe Yu, Qiujie Li, Christian Fischer, Shayan Doroudi, and Di Xu. 2020. Towards Accurate and Fair Prediction of College Success: Evaluating Different Sources of Student Data. _International educational data mining society_ (2020). 

- [53] Xiaoshan Yu, Chuan Qin, Dazhong Shen, Haiping Ma, Le Zhang, Xingyi Zhang, Hengshu Zhu, and Hui Xiong. 2024. RDGT: Enhancing Group Cognitive Diagnosis with Relation-Guided Dual-Side Graph Transformer. _IEEE Transactions on Knowledge and Data Engineering_ (2024). 

- [54] Zheng Zhang, Le Wu, Qi Liu, Jiayu Liu, Zhenya Huang, Yu Yin, Yan Zhuang, Weibo Gao, and Enhong Chen. 2024. Understanding and improving fairness in cognitive diagnosis. _Science China Information Sciences_ 67, 5 (2024), 152106. 

- [55] Zihao Zhao, Jiawei Chen, Sheng Zhou, Xiangnan He, Xuezhi Cao, Fuzheng Zhang, and Wei Wu. 2022. Popularity bias is not always evil: Disentangling benign and harmful bias for recommendation. _IEEE Transactions on Knowledge and Data Engineering_ (2022). 

- [56] Yuqiang Zhou, Qi Liu, Jinze Wu, Fei Wang, Zhenya Huang, Wei Tong, Hui Xiong, Enhong Chen, and Jianhui Ma. 2021. Modeling context-aware features for cognitive diagnosis in student learning. In _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining_ . 2420–2428. 

- [57] Ziwei Zhu, Jianling Wang, and James Caverlee. 2020. Measuring and mitigating item under-recommendation bias in personalized ranking systems. In _Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval_ . 449–458. 

KDD ’24, August 25–29, 2024, Barcelona, Spain. 

Dacao Zhang, Kun Zhang, Le Wu, Mi Tian, Richang Hong, Meng Wang. 

**Table 8: The statistics of useful but not sensitive attributes associated with ESCS** 

|id|name|correlation|category|
|---|---|---|---|
|ST013Q01TA<br>ST012Q07NA<br>ST011Q06TA<br>STo11Q04TA<br>ST012Q08NA|How many books are there in your home?<br>Tablet computers<br>A link to the Internet<br>A computer you can use for school work<br>E-book readers|0.416<br>0.402<br>0.308<br>0.301<br>0.266|0: 0-100 books, 1: More than 100 books<br>0: Zero or one 1: More than one<br>0: Yes 1: No<br>0: Yes 1: No<br>0: NaN 1: More than one|



**Table 9: The statistics of useful but not sensitive attributes associated with Father’s education level** 

|id|name|correlation|category|
|---|---|---|---|
|ST013Q01TA<br>ST012Q07NA<br>ST011Q09TA<br>ST011Q16NA<br>ST011Q07TA|How many books are there in your home?<br>Tablet computers<br>Works of art (e.g. paintings)<br>Books on art, music or design<br>Classic literature (e.g. <Shakespeare>)|0.286<br>0.166<br>0.165<br>0.146<br>0.142|0: 0-100 books, 1: More than 100 books<br>0: Zero or one 1: More than one<br>0: Yes 1: No<br>0: Yes 1: No<br>0: Yes 1: No|



**Table 10: The Performance of** _**PSCRF**_ **on Graph-Based RCD Model** 

|**Method**|**EO**↓|**D**_under_<br>_disadv_|**IR**↑|**AUC**↑|**ACC**↑|**DOA**↑|
|---|---|---|---|---|---|---|
|Base|0.0394|0.0965|0.7132|0.7971|0.7265|0.6209|
|Base†|0.0622|0.1523|0.7000|0.8005|0.7269|0.6744|
|_PSCRF_|0.0032|0.0004|0.7433|0.8009|0.7265|0.6327|



**Table 11: The statistical data for alpha and beta.** 

|**param**|**mean**|**variance**|
|---|---|---|
|_𝛼_|0.2043|0.0826|
|_𝛽_|0.4041|0.1304|



## **A APPENDIX** 

## **A.1 The selection of Educational Conetxt** 

We present the most relevant useful attributes associated with different sensitive attributes in Table 8 and Table 9. We employ the Pearson correlation coefficient to calculate the correlation, and select the Top-k attributes as the educational context used in Eq.(9) in Section 4.2.2: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis_images/kdd_2023_Path-Specific_Causal_Reasoning_for_Fairness-aware_Cognitive_Diagnosis.pdf-0012-13.png)


## **A.2 The Performance of** _**PSCRF**_ **on Graph-Based RCD Model** 

Since our _PSCRF_ focuses more on fairness-aware cognitive diagnosis, we select fundamental and general CD models in the main text. To further demonstrate the generality of _PSCRF_ , we applied PSCRF to the graph-based model RCD[15] and report the results on Australia dataset regarding the sensitive attribute ESCS in Table 10. It can be observed that _PSCRF_ achieves similarly good performance on the RCD model as well. 

## **A.3 The relevant information about the learnable parameters** _𝛼_ **and** _𝛽_ **.** 

We want to know the values of _𝛼_ and _𝛽_ after they have been learned, so we computed the mean and variance of their values after training on the Australia dataset using IRT model, as shown in Table 11. From the results, we can conclude that our proposed _PSCRF_ realizes user-specific debiasing according to the input user’s situation. 

