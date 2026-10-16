# **DiReCT: Diagnostic Reasoning for Clinical Notes via Large Language Models** 

**Bowen Wang** _[♣♢∗]_ , **Jiuyang Chang** _[♡∗]_ , **Yiming Qian** _[♠†]_ , **Guoxin Chen**[⋆] , **Junhao Chen** _[♢]_ , **Zhouqiang Jiang** _[♢]_ , **Jiahao Zhang** _[♢]_ , **Yuta Nakashima** _[♢♣]_ , **Hajime Nagahara** _[♢♣]_ 

> _♣_ Premium Research Institute for Human Metaverse Medicine (WPI-PRIMe), Osaka University, 

> _♡_ Department of Cardiology, The First Affiliated Hospital of Dalian Medical University, 

⋆Institute of Computing Technology, Chinese Academy of Science 

> _♢_ D3 Center, Osaka University, _♠_ Agency for Science, Technology and Research (A*STAR), 

{ `wang` , `n-yuta` , `nagahara` } `@ids.osaka-u.ac.jp changjiuyang@firsthosp-dmu.com qiany@ihpc.a-star.edu.sg` , `gx.chen.chn@gmail.com` 

{ `junhao` , `zhouqiang` , `jiahao` } `@is.ids.osaka-u.ac.jp` 

## **Abstract** 

Large language models (LLMs) have recently showcased remarkable capabilities, spanning a wide range of tasks and applications, including those in the medical domain. Models like GPT-4 excel in medical question answering but may face challenges in the lack of interpretability when handling complex tasks in real clinical settings. We thus introduce the diagnostic reasoning dataset for clinical notes (DiReCT), aiming at evaluating the reasoning ability and interpretability of LLMs compared to human doctors. It contains 511 clinical notes, each meticulously annotated by physicians, detailing the diagnostic reasoning process from observations in a clinical note to the final diagnosis. Additionally, a diagnostic knowledge graph is provided to offer essential knowledge for reasoning, which may not be covered in the training data of existing LLMs. Evaluations of leading LLMs on DiReCT bring out a significant gap between their reasoning ability and that of human doctors, highlighting the critical need for models that can reason effectively in real-world clinical scenarios[‡] . 

## **1 Introduction** 

Recent advancements of large language models (LLMs) [Zhao et al., 2023] have ushered in new possibilities and challenges for a wide range of natural language processing (NLP) tasks [Min et al., 2023]. In the medical domain, these models have demonstrated remarkable prowess [Anil et al., 2023, Han et al., 2023], particularly in medical question answering (QA) [Jin et al., 2021]. Leading-edge models, such as GPT-4 [OpenAI, 2023a], exhibit profound proficiency in understanding and generating text [Bubeck et al., 2023], even achieved high scores on the United States Medical Licensing Examination (USMLE) questions [Nori et al., 2023]. 

Despite the advancements, interpretability is critical, particularly in medical NLP tasks [Liévin et al., 2024] because these tasks directly impact patient health and treatment decisions. Without clear interpretability, there’s a risk of misdiagnosis and improper treatment, making it vital for ensuring medical safety. Some studies assess this capability over medical QA [Pal et al., 2022, Li et al., 2023, 

> *Equal contribution. 

> †Corresponding author. 

> ‡Data and code are available at https://github.com/wbw520/DiReCT. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Suspected Hemorrhagic<br>Stroke Stroke<br>Chief Complaint : Right Radiology : A 3.0 x 1.1 cm<br>weakness and aphasia.  left thalamic hematoma<br>Events : He had episode of appears stable when ......<br>maurosis fugax in right eye MR HEAD : Only ***** T1,<br>******** ago ...... axial T1, and axial FLAIR<br>Past Medical History : sequences were  ...... Final<br>HTN, COPD on home 1L CT HEAD:  Stable **** basal Diagnosis<br>Admission Consultation ......  Examination ganglia  ......<br>Diagnosis Procedure<br>**----- End of picture text -----**<br>


Figure 1: When a patient is admitted, an initial consultation takes place to collect subjective information. Subsequent observations may then require further examination to confirm the diagnosis. 

Chen et al., 2024] or natural language inference (NLI) [Jullien et al., 2023]. Putting more attention on interpretability, they use relatively simple tasks as testbeds, taking short text as input. Nevertheless, real-world clinical tasks are often more complex [Gao et al., 2023a], as illustrated in Figure 1, a typical diagnosis requires comprehending and combining various information, such as health records, physical examinations, and laboratory tests, for further reasoning of possible diseases in a step-by-step manner following the established guidelines. This observation suggests that both _perception_ , or reading (e.g., finding necessary information in the medical record) and _reasoning_ (determining the disease based on the observations) should be counted when evaluating interpretability in LLM-based medical NLP tasks. 

For a more comprehensive evaluation of LLMs for supporting diagnosis in a more realistic setting, we propose a **Di** agnostic **Re** asoning dataset for **C** linical no **T** es (DiReCT). The task basically is predicting the diagnosis from a _clinical note_ of a patient, which is a collection of various medical records, written in natural language. Our dataset contains 511 clinical notes spanning 25 disease categories, sampled from a publicly available database, MIMIC-IV [Johnson et al., 2023]. Each clinical note undergoes fine-grained annotation by professional physicians. The annotators (i.e., the physicians) are responsible for identifying the text, or the _observation_ , in the note that leads to a certain diagnosis, as well as the explanation. The dataset also provides a diagnostic knowledge graph based on existing diagnostic guidelines to facilitate more consistent annotations and to supply a model with essential knowledge for reasoning that might not be encompassed in its training data. 

To underscore the challenge offered by our dataset, we propose a simple AI-agent based baseline [Xi et al., 2023, Tang et al., 2023] that utilizes the knowledge graph to decompose the diagnosis into a sequence of diagnoses from a smaller number of observations. Our experimental findings indicate that current state-of-the-art LLMs still fall short of aligning well with human doctors. 

**Contribution** . DiReCT offers a new challenge in diagnosis from a complex clinical note with explicit knowledge of established guidelines. This challenge aligns with a realistic medical scenario that doctors are experiencing. In the application aspect, the dataset facilitates the development of a model to support doctors in diagnosis, which is error-prone [Middleton et al., 2013, Liu et al., 2022]. From the technical aspect, the dataset can benchmark models’ ability to read long text and find necessary observations for _multi-evidence entailment tree_ reasoning, an extension of the original entailment tree explanation [Dalvi et al., 2021] for complex scenarios in medical NLP tasks. As shown in Figure 3, this is not trivial because of the variations in writing; superficial matching does not help, and medical knowledge is vital. Meanwhile, reasoning itself is facilitated by the knowledge graph. The model does not necessarily have the knowledge of diagnostic guidelines. With this choice, the knowledge graph explains the reasoning process, which is also beneficial when deploying such a diagnosis assistant system in practical uses. 

## **2 Related Works** 

**Natural language explanation** . Recent advancements in NLP have led to significant achievements [Min et al., 2023]. However, existing models often lack explainability, posing potential risks [Danilevsky et al., 2020, Gurrapu et al., 2023]. Numerous efforts have been made to address this challenge. One effective approach is to provide a human-understandable _plain text_ explanation alongside the model’s output [Camburu et al., 2018, Rajani et al., 2019]. Another strategy involves identifying _evidence_ within the input that serves as a rationale for the model’s decisions, aligning with 

2 

Table 1: Comparison of existing datasets for medical reasoning tasks and ours. “t” and “w” mean tokens and words for the length of input, respectively. 

|**Dataset**|**Task**|**Data Source**|**Length**|**Explanation**|**# Cases**|
|---|---|---|---|---|---|
|MedMCQA [Pal et al.,2022]|QA|Examination|9.93 t|Plain Text|194,000|
|ExplainCPE [Li et al.,2023]|QA|Examination|37.79 w|Plain Text|7,000|
|JAMA Challenge [Chen et al.,2024]|QA|Clinical Cases|371 w|Plain Text|1,524|
|Medbullets [Chen et al.,2024]|QA|Online Questions|163 w|Plain Text|308|
|N2N2 [Gao et al.,2022]|Sum|Clinical Notes|785.46 t|Evidences|768|
|NLI4CT [Jullien et al.,2023]|NLI|Clinical Trail Reports|10-35 t|Multi-hop|2,400|
|NEJM CPC [Zack et al.,2023]|CD|Clinical Cases|-|Plain Text|2,525|
|DiReCT (Ours)|CD|Clinical Notes|1074.6 t|Entailment Tree|511|



human reasoning [DeYoung et al., 2020]. Expanding on this concept, [Jhamtani and Clark, 2020] introduces chain-structured explanations, given that a diagnosis can demand multi-hop reasoning. This idea is further refined by ProofWriter [Tafjord et al., 2021] through a proof stage for explanations, and by [Zhao et al., 2021] through retrieval from a corpus. [Dalvi et al., 2021] proposes the _entailment tree_ , offering more detailed explanations and facilitating inspection of the model’s reasoning. More recently, [Zhang et al., 2024] employed cumulative reasoning to tap into the potential of LLMs to provide explanation via a _directed acyclic graph_ . Although substantial progress has been made, interpreting NLP tasks in medical domains remains an ongoing challenge [Liévin et al., 2024]. 

**Benchmarks of interpretability in the medical domain** Several datasets are designed to assess a model’s reasoning together with its interpretability in medical NLP (Table 1). MedMCQA [Pal et al., 2022] and other medical QA datasets [Li et al., 2023, Chen et al., 2024] provide plain text as explanations for QA tasks. NLI4CT [Jullien et al., 2023] uses clinical trial reports, focusing on NLI supported by multi-hop reasoning. N2N2 [Gao et al., 2022] proposes a summarization (Sum) task for a diagnosis based on multiple pieces of evidence in the input clinical note. NEJM CPC [Zack et al., 2023] interprets clinicians’ diagnostic reasoning as plain text for reasoning clinical diagnosis (CD). DR.BENCH [Gao et al., 2023b] aggregates publicly available datasets to assess the diagnostic reasoning of LLMs. Utilizing an multi-evidence entailment tree explanation, DiReCT introduces a more rigorous task to assess whether LLMs can align with doctors’ reasoning in real clinical settings. 

## **3 A benchmark for Clinical Notes Diagnosis** 

This section first detail clinical notes (Section 3.1). We also describes the knowledge graph that encodes existing guidelines (Section 3.2). Our task definition, which tasks a clinical note and the knowledge graph as input is given in Section 3.4. We then present our annotation process for clinical notes (Section 3.3) and the evaluation metrics (Section 3.5). 

## **3.1 Clinical Notes** 

Clinical notes used in DiReCT are stored in the SOAP format [Weed, 1970]. A clinical note comprises four components: In the _subjective_ section, the physician records the patient’s chief complaint, the history of present illness, and other subjective experiences reported by the patient. The _objective_ section contains structural data obtained through examinations (inspection, auscultation, etc.) and other measurable means. The _assessment_ section involves the physician’s analysis and evaluation of the patient’s condition. This may include a summary of current status, _etc_ . Finally, the _plan_ section outlines the physician’s proposed treatment and management plan. This may include prescribed medications, recommended therapies, and further investigations. A clinical note also includes a primary discharge diagnosis (PDD) in the assessment section. 

DiReCT’s clinical notes are sourced from the MIMIC-IV dataset [Johnson et al., 2023] (PhysioNet Credentialed Health Data License 1.5.0), which encompasses over 40,000 patients admitted to the intensive care units. Each note contains clinical data for a patient. To construct DiReCT, we curated a subset of 511 notes whose PDDs fell within one of 25 disease categories _i_ in 5 medical domains. 

In our task, a note _R_ = _{r}_ is an excerpt of 6 clinical data in the subjective and objective sections (i.e., _|R|_ = 6): chief complaint, history of present illness, past medical history, family history, physical 

3 

exam, and pertinent results.[1] We also identified the PDD _d[⋆]_ associated with _R_ .[2] The set of _d[⋆]_ ’s for all _R_ ’s collectively forms _D[⋆]_ . We manually removed any descriptions that disclose the PDD in _R_ . 

## **3.2 Diagnostic Knowledge Graph** 

Existing knowledge graphs for the medical domain, e.g., UMLS KG [Bodenreider, 2004], lack the ability to provide specific clinical decision support (e.g., diagnostic threshold, context-specific data, dosage information, etc.), which are critical for accurate diagnosis. 

Our knowledge graphs _K_ = _{ki}_ is a collection of graph _ki_ for disease category _i_ . _ki_ is based on the diagnosis criteria in existing guidelines (refer to supplementary material for details). _ki_ ’s nodes are either premise _p ∈Pi_ (medical statement, e.g., `Headache is a symptom of` ) and diagnoses _d ∈Di_ (e.g., `Suspected Stroke` ). _ki_ consists of two different types of edges. One is _premise-todiagnosis_ edges _Si_ = _{_ ( _p, d_ ) _}_ ; an edge is from _p_ to _d_ . This edge represents the necessary premise _p_ to make a diagnosis _d_ . We refer to them as _supporting_ edges. The other is _diagnosis-to-diagnosis_ edges _Fi_ = _{_ ( _d, d[′]_ ) _}_ , where _d, d[′] ∈Di_ and the edge is from _d_ to _d[′]_ , which represents the diagnostic flow. These edges are referred to as _procedural_ edges. 

A disease category is defined according to an existing guideline, which starts from a certain diagnosis; therefore, a procedural graph _gi_ = ( _Di, Fi_ ) ( _G_ = _{gi}_ ) has only one root node and arbitrarily branches toward multiple leaf nodes that represent PDDs (i.e., the clinical notes in DiReCT are chosen to cover all leaf nodes of _gi_ ). Thus, _gi_ is a _tree_ . We denote the set of the leaf nodes (or PDDs) as _Di[⋆][⊂D][i]_[.][The knowledge graph is denoted by] _[ k][i]_[= (] _[D][i][,][ P][i][,][ S][i][,][ F][i]_[)][.] 

|as_D⋆_<br>_i ⊂Di_. The knowledge graph|is denoted by_ki_ = (_Di, Pi, Si, Fi_).|is denoted by_ki_ = (_Di, Pi, Si, Fi_).|is denoted by_ki_ = (_Di, Pi, Si, Fi_).|is denoted by_ki_ = (_Di, Pi, Si, Fi_).|is denoted by_ki_ = (_Di, Pi, Si, Fi_).|||
|---|---|---|---|---|---|---|---|
|Figure 2 shows a part of _ki_,<br>where<br>_i_<br>is<br>`Acute Coronary`<br>`Syndromes (ACS)`. Premises in<br>_Pi_and diagnoses in_Di_are given<br>in the blue and gray boxes, while<br>PDDs in_D⋆_<br>_i_ are ones without out-<br>going edges (i.e.,`STEMI-ACS`and|Suspected ACS<br>Strongly<br>Suspected ACS<br>Any Severe<br>Presentations ...<br>Third Heart<br>Sound ...<br>Breathlessness<br>is a symptom ...<br>Arrhythmias<br>is ...<br>**......**<br>**......**<br>ST Elevation<br>is criteria ...<br>non-ST<br>Elevation ...|||||STEMI-ACS<br>NSTE-ACS<br>NSTEMI-ACS<br>UA<br>hs-cTn<br>Exceeded ...<br>Cardiac<br>Troponin ↑<br>No Obvious<br>ECG  ...||
|`NSTEMI-ACS`, and`UA`). The black|Figure 2: A part of_ki_||for_i_being`Acute Coronary Syndromes`.|||||
|and red arrows are edges in_S_ and||||||||
|_F_, respectively, where the black||||||||
|arrows indicate the supporting edges.||||||||



_K_ serves two essential functions: (1) They serve as the gold standard for annotation, guiding doctors in the precise and uniform interpretation of clinical notes. (2) Our task also allows a model to use them to ensure the output from an LLM can be closely aligned with the reasoning processes of medical professionals. 

## **3.3 Data Annotation** 

Let _d[⋆] ∈Di[⋆]_[denote the PDD of disease category] _[ i]_[ associated with] _[ R]_[.][We can find a subgraph] _[ k][i]_[(] _[d][⋆]_[)] of _ki_ that contains all ancestors of _d[⋆]_ , including premises in _Pi_ . We also denote the set of supporting edges in _ki_ ( _d[⋆]_ ) as _Si_ ( _d[⋆]_ ). Our annotation process is, for each supporting edge ( _p, d_ ) _∈Si_ ( _d[⋆]_ ), to extract observation _o ∈O_ in _R_ (highlighted text in the clinical note in Figure 3) and provide rationalization _z_ of this _deduction_ why _o_ is a support for _d_ or corresponds to _p_ .[3] They form the explanation _E_ = _{_ ( _o, z, d_ ) _}_ for ( _R, d[⋆]_ ). This annotation process was carried out by 9 clinical physicians and subsequently verified for accuracy and completeness by three senior medical experts. 

Table 2 summarizes statistics of our dataset. The second and third columns (“# cats.” and “# samples”) show the numbers of disease categories and samples in the respective medical domains. _|Di|_ and _|Di[⋆][|]_[ are the total numbers of diagnoses (diseases) and PDDs, summed over all diagnostic categories] 

> 1We excluded data, such as review system and social history, because they are often missing in the original clinical notes and are less relevant to the diagnosis. 

> 2All clinical notes in DiReCT are related to only one PDD, and there is no secondary discharge diagnosis. 

> 3All annotations strictly follow the procedural flow in _ki_ , and each observation is only related to one diagnostic node. If _R_ does not provide sufficient observations for the PDD (which may happen when a certain test is omitted), the annotators were asked to add plausible observations to _R_ . Refer to amended data points in supplementary for details. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Chief Complaint : Scrotal and leg swelling  ... Peripheral oedema is<br>History of Present Illness :  ... In the last 3 days his ***** has become quite a sign of heart failure Suspected HF<br>swollen. It is similar ***** swelling when he was admitted with acute CHF  ...  EKG ......<br>was consistent with priors (NSR, NANI, ********** changes).  The left ventricle is<br>mildly enlarged. He was given ********* with good UOP  ...  Hypertension is the risk factor<br>Past Medical History :  ... -Diabetes, -Hypertension, -CKD, stage 3, -GERD, - of heart failure<br>Depression, - Amputation of ***********, Pneumonia, - Osteoarthritis- History of ...... Strongly<br>*********, Asthma   ... Suspected HF<br>Family History： There is no family history of **************** artery  ... NT-proBNP 3843≥125pg/ml is a<br>Physical Exam :  ...  LUNG: bibasilar rales that do not clear with deep inspiration. diagnostic criteria of strong HF<br>...  ABDOMEN: nondistended, ********* all quadrants. EXTREMITIES: bilateral ......<br>pitting edema to the sacrum, extending to the up abdomen. Warm, well perfused. ...  HEENT: AT/NC, EOMI, PERRL.  ... Cardiac structure abnormalities are HF<br>Pertinent Results : __03:50PM BLOOD WBC-8.0 RBC-3.26* Hgb-9.3* Hct- diagnostic criteria of heart failure<br>30.9* MCHC-29.9*  ...  __ 11:30AM BLOOD proBNP-3843  ...  Overall left ventricular ......<br>systolic function is mildly depressed (LVEF= 45-50 %) without regional wall motion<br>Cardiac systolic dysfunction ~49%<br>abnormalities. *********** imaging suggests an increased ******* filling pressure(PCWP>******Hg)  ...  can lead to the diagnosis of HFmrEF HFmrEF<br>Clinical Note Rationale  Diagnosis<br>**----- End of picture text -----**<br>


Figure 3: An annotation sample of `Heart Failure` ( `HF` ). The left part is the clinical note alongside extracted observations by a doctor. The middle part outlines the steps of the rationale for the premise corresponding to each diagnostic node shown in the right part. 

in the medical domain, respectively. _|O|_ is the average number of annotated observations. “Length” is the average number of tokens in _R_ . 

## **3.4 Task Definition** 

We propose two tasks with different levels of supplied external knowledge. The first task is, given _R_ and _G_ , to predict the associated PDD _d[⋆]_ and generate an explanation _E_ that explains the model’s diagnostic procedure from _R_ to _d[⋆]_ , i.e., letting _M_ denote a model: 

Table 2: Statistics of DiReCT. 

|wo tasks with different levels of<br>ernal knowledge. The frst task|Table 2: Statistics of DiReCT.|
|---|---|
|nd_G_, to predict the associated<br>generate an explanation_E_ that<br> model’s diagnostic procedure<br>, i.e., letting_M_ denote a model:<br>ˆ_d⋆,_ ˆ_E_ =_M_(_R, G_)_,_<br>(1)|Medical domain<br># cat.<br># samples<br>_|Di|_<br>_|D⋆_<br>_i |_<br>_|O|_<br>Length|
||Cardiology<br>7<br>184<br>27<br>16<br>8.7<br>1156.6 t<br>Gastroenterology<br>4<br>103<br>11<br>7<br>4.3<br>1026.0 t<br>Neurology<br>5<br>77<br>17<br>11<br>11.9<br>1186.3 t<br>Pulmonology<br>5<br>92<br>26<br>17<br>10.7<br>940.7 t<br>Endocrinology<br>4<br>55<br>20<br>14<br>6.9<br>1063.5 t|
||Overall<br>25<br>511<br>101<br>65<br>8.5<br>1074.6 t|



where _d_[ˆ] _[⋆] ∈∪iDi[⋆]_[and] _[E]_[ˆ][are predictions for the PDD and explanation, respectively.][With this task,] the knowledge of specific diagnostic procedures in existing guidelines can be used for prediction, facilitating interpretability. The second task takes _K_ as input instead of _G_ , i.e.,: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0005-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0005-09.png)


This task allows for the use of broader knowledge of premises for prediction. One may also try a task without any external knowledge. 

## **3.5 Evaluation Metrics** 

We designed three metrics to quantify the predictive performance over our benchmark. 

(1) _Accuracy of diagnosis Acc_[diag] evaluates if a model can correctly identify the diagnosis. _Acc_[diag] = 1 if _d[⋆]_ = _d_[ˆ] , and _Acc_[diag] = 0 otherwise. The average is reported. 

(2) _Completeness of observations Obs_[comp] evaluates whether a model extracts all and only necessary observations for the prediction. Let _O_ and _O_[ˆ] denote the sets of observations in _E_ and _E_[ˆ] , respectively. The metric is defined as _Obs_[comp] = _|O ∩ O|_[ˆ] _/|O ∪ O|_[ˆ] , where the numerator is the number of observations that are common in both _O_ and _O_[ˆ] .[4] This metric simultaneously evaluates the correctness of each observation and the coverage. To supplement it, we also report the precision _Obs_[pre] and recall _Obs_[rec] , given by _Obs_[pre] = _|O ∩ O|_[ˆ] _/|O|_[ˆ] and _Obs_[rec] = _|O ∩ O|_[ˆ] _/|O|_ . 

(3) _Faithfulness of explanations_ evaluates if the diagnostic flow toward the PDD is fully supported by observations with faithful rationalizations. This involves establishing a one-to-one correspondence between deductions in the prediction and the ground truth. We use the correspondences established for computing _Obs_[comp] . Let _o ∈O_ and ˆ _o ∈ O_[ˆ] denote corresponding observations. This correspondence 

> 4We find the common observations with an LLM (refer to the supplementary material for more detail). 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0006-00.png)


**----- Start of picture text -----**<br>
R U<br>Narrowing- Observations ①② ③ ④ ⑤<br>down ① Elevated blood pressures<br>W ② CXR showed mild pulmonary edema d0<br>Clinical Perception ③ CHF/Cardiomyopathy V ①② ③ ④ ⑤<br>Note ④ Severe LV diastolic dysfunction<br>⑤ BPs: 148/98, 156/93 Reasoning<br>r1: Chief Complaint ...... d0 d1<br>r2: History of Present Illness<br>r3: Past Medical History Diagnostic KG ......<br>r4: Family History V ① ② ③ ④ ⑤<br>r5: Physical Examination Reasoning<br>r6: Pertinent Results d0 d1 ... dt<br>**----- End of picture text -----**<br>


Figure 4: Pipeline of our baseline. The dotted line in the right-most boxes means deductions from an observation to a diagnosis. 

is considered successful if _z_ and _z_ ˆ as well as _d_ and _d_[ˆ] associated with _o_ and ˆ _o_ matches. Let _m_ ( _E, E_[ˆ] ) denote the number of successful matches. We use the ratio of _m_ ( _E, E_[ˆ] ) to _|O ∩ O|_[ˆ] and _|O ∪ O|_[ˆ] as evaluation metrics _Exp_[com] and _Exp_[all] , respectively, to see failures come from observations or explanations and diagnosis. 

## **4 Baseline** 

Figure 4 provides an overview of our baseline, which comprises three LLM-based modules: narrowing-down ( _U_ ), perception ( _W_ ), and reasoning ( _V_ ). In our experiments, each module utilizes the same type of LLM with different prompts (refer to the supplementary material for more details). _U_ analyze the entire note _R_ to determine the possible disease type[ˆ] _i_ . _W_ extracts observations that may lead to diseases from each _r_ , producing a list of original disease descriptions. _V_ iteratively derives possible diseases from observations based on the diagnosis knowledge graph, providing rationales for each deduction ( _o, z, d_ ). 

ˆThe narrowing-down module _U_ takes _R_ as input to make a prediction[ˆ] _i_ of the disease category, i.e., _i_ = _U_ ( _R_ ). Let _dt ∈D_ ˆ _i_ be the diagnosis that has been reached with _t_ iterations over _k_ ˆ _i_ , where _t_ corresponds to the depth of node _dt_ and so is less than or equal to the depth of _k_ ˆ _i_ . _d_ 0 is the root node of _k_ ˆ _i_ . For _d_ 0, we apply the perception module to extract all observations in _R_ and explanation _E_ 0 to support _d_ 0 as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0006-06.png)


_k_ ˆ _i_ is supplied to facilitate the model to extract all observations for the following reasoning process.[5] After the perception module _W_ (iteration _t_ = 0), we obtain all observations _O_[ˆ] , the root node of the diagnosis _d_ 0, and an explanation _E_[ˆ] 0 for the initial iteration. Assuming that by iteration _t_ , we already know the diagnosis for iteration _t_ as _dt_ . _{dn}_ is the set of _dt_ ’s children, and _P_ ˆ _i_ ( _{dn}_ ) represents the corresponding premises that support each _dn_ . _V_ identifies the diagnosis for the next step, _dt_ +1, and provides a justification _Et_ +1. _V_ will verify if there is any ˆ _o_ in _O_[ˆ] that supports a _dn_ . If fully supported, _dn_ is identified as _dt_ +1 for the ( _t_ + 1)-th iteration, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0006-08.png)


_V_ continues until _dt_ +1 in _D[∗]_ is identified. If no observation supports a _dn_ , the reasoning process will be stopped. 

In our annotation, an observation _o_ is associated with only one _d_ . However, our method employs an iterative reasoning pipeline. Initially, the perception module _W_ generates an explanation set _E_[ˆ] 0, linking all _o_ ˆ to _d_ 0. During the _t_ -th iteration of _V_ , the explanation set is _E_[ˆ] _t_ , where at least one _o_ ˆ is 

> 5We used only pairs of an observation and a premise. We abuse _K_ to mean this for notation simplicity. The perception model can also utilize _gi_ instead of _ki_ for the first task. 

6 

Table 3: Evaluation of diagnostic reasoning ability using _G_ or _K_ as input. 

|Task<br>Models|Diagnosis<br>_Acc_cat<br>_Acc_diag|Observation<br>_Obs_pre<br>_Obs_rec<br>_Obs_comp|Explanation<br>_Exp_com<br>_Exp_all|
|---|---|---|---|
|With_G_<br>Zephyr 7B<br>Mistral 7B<br>Mixtral 8_×_7B<br>LLama3 8B<br>LLama3 70B<br>GPT-3.5 turbo<br>GPT-4 turbo|0.274<br>0.151<br>0.507<br>0.306<br>0.413<br>0.237<br>0.569<br>0.364<br>0.822<br>0.606<br>0.679<br>0.455<br>**0.804**<br>**0.610**|0.123_±_0.200<br>0.115_±_0.166<br>0.092_±_0.108<br>0.211_±_0.190<br>0.317_±_0.253<br>0.173_±_0.157<br>0.147_±_0.165<br>0.266_±_0.261<br>0.124_±_0.138<br>0.248_±_0.157<br>0.410_±_0.218<br>0.211_±_0.138<br>0.306_±_0.151<br>**0.543**_±_**0.183**<br>0.279_±_0.146<br>0.389_±_0.212<br>0.351_±_0.192<br>0.275_±_0.167<br>**0.486**_±_**0.207**<br>0.481_±_0.180<br>**0.391**_±_**0.189**|0.071_±_0.139<br>0.014_±_0.037<br>0.230_±_0.312<br>0.062_±_0.088<br>0.144_±_0.268<br>0.029_±_0.056<br>0.325_±_0.375<br>0.087_±_0.118<br>0.409_±_0.328<br>0.124_±_0.120<br>0.331_±_0.366<br>0.103_±_0.127<br>**0.481**_±_**0.362**<br>**0.210**_±_**0.188**|
|With_K_<br>LLama3 8B<br>LLama3 70B<br>GPT-3.5 turbo<br>GPT-4 turbo|0.576<br>0.344<br>0.786<br>0.652<br>0.652<br>0.413<br>**0.808**<br>**0.611**|0.235_±_0.162<br>0.394_±_0.227<br>0.199_±_0.142<br>0.268_±_0.147<br>**0.524**_±_**0.211**<br>0.258_±_0.142<br>0.347_±_0.241<br>0.279_±_0.203<br>0.232_±_0.184<br>**0.470**_±_**0.209**<br>0.459_±_0.190<br>**0.371**_±_**0.192**|0.327_±_0.375<br>0.087_±_0.114<br>0.549_±_0.372<br>0.152_±_0.130<br>0.374_±_0.408<br>0.121_±_0.152<br>**0.645**_±_**0.385**<br>**0.273**_±_**0.216**|



linked to _dt_ . The final diagnosis explanation is the combination of _E_[ˆ] 0 _, . . . , E_[ˆ] _T_ and _d_ 0 _, . . . , dT_ , where _T_ ˆ represents the final iteration. In this combination, if anˆ ˆ ˆ _o_ is eventually processed in the iteration for _Et_ , the corresponding ( _o, z, d_ ) in all preceding _E_ 0 _, . . . , Et−_ 1 will be removed. That is, ˆ _o_ will always be possessed by the _dt_ closest to the leaf PDD node. 

## **5 Experiments** 

## **5.1 Experimental Setup** 

We assess the reasoning capabilities of 7 recent LLMs from diverse families and model sizes, including 5 instruction-tuned models that are openly accessible: LLama3 8B and 70B [AI@Meta, 2024], Zephyr 7B [Tunstall et al., 2023], Mistral 7B [Jiang et al., 2023], and Mixtral 8 _×_ 7B [Jiang et al., 2023]. We have also obtained access to private versions of the GPT-3.5 turbo [OpenAI, 2023b] and GPT-4 turbo [OpenAI, 2023a][6] , which are high-performance closed-source models. Each LLM is utilized to implement our baseline’s narrowing-down, perception, and reasoning modules. The temperature is set to 0. For computing evaluation metrics, we use LLama3 8B with few-shot prompts to make correspondences between _O_ and _O_[ˆ] as well as to verify a match between predicted and ground-truth explanations (refer to the supplementary material for more details). 

## **5.2 Results** 

**Comparison among LLMs.** Table 3 shows the performance of our baseline built on top of various LLMs. We first evaluate a variant of our task that takes graph _G_ = _{Gi}_ consisting of only procedural flow as external knowledge instead of _K_ . Comparison between _G_ and _K_ demonstrates the importance of supplying premises with the model and LLMs’ capability to make use of extensive external knowledge that may be superficially different from statements in _R_ . Subsequently, some models are evaluated with our task using _K_ . In addition to the metrics in Section 3.5, we also adopt the _accuracy of disease category Acc_[cat] , which gives 1 when[ˆ] _i_ = _i_ , as our baseline’s performance depends on it. 

With _G_ , we can see that GPT-4 achieves the best performance in most metrics, especially related to observations and explanations, surpassing LLama3 70B by a large margin. In terms of accuracy (in both category and diagnosis levels), LLama3 70B is comparable to GPT-4. LLama3 70B also has a higher _Obs_[rec] but low _Obs_[pre] and _Obs_[comp] , which means that this model tends to extract many observations. Models with high diagnostic accuracy are not necessarily excel in finding essential information in long text (i.e., observations) and generating reasons (i.e., explanations). 

When _K_ is given, all models show better diagnostic accuracy (in LLama3 70B) and explanations, while observations are slightly degraded (this may related to the instruction following ability due to the input length when giving _K_ as input). GPT-4 with _K_ enhances _Acc_[diag] , _Exp_[com] , and _Exp_[all] scores. This suggests that premises and supporting edges are beneficial for diagnosis and explanation. Lower 

> 6These two models are housed on a HIPPA-compliant instance within Microsoft Azure AI Studio. No data is transferred to either Microsoft or OpenAI. This secure environment enables us to safely conduct experiments with the MIMIC-IV dataset, in compliance with the Data Use Agreement. 

7 

Table 4: Evaluation of diagnostic reasoning ability without external knowledge. 

||Task<br>Models<br>_Acc_diag|Task<br>Models<br>_Acc_diag|Task<br>Models<br>_Acc_diag|Observation<br>_Obs_pre<br>_Obs_rec<br>_Obs_comp|Observation<br>_Obs_pre<br>_Obs_rec<br>_Obs_comp|Observation<br>_Obs_pre<br>_Obs_rec<br>_Obs_comp|Explanation<br>_Exp_com<br>_Exp_all<br>0.020_±_0.100<br>0.004_±_0.016<br>0.138_±_0.209<br>0.034_±_0.054<br>0.091_±_0.231<br>0.025_±_0.065<br>**0.186**_±_**0.221**<br>**0.074**_±_**0.090**<br>0.018_±_0.083<br>0.006_±_0.026<br>0.022_±_0.093<br>0.007_±_0.030<br>0.036_±_0.131<br>0.011_±_0.039<br>**0.047**_±_**0.143**<br>**0.019**_±_**0.058**|Explanation<br>_Exp_com<br>_Exp_all<br>0.020_±_0.100<br>0.004_±_0.016<br>0.138_±_0.209<br>0.034_±_0.054<br>0.091_±_0.231<br>0.025_±_0.065<br>**0.186**_±_**0.221**<br>**0.074**_±_**0.090**<br>0.018_±_0.083<br>0.006_±_0.026<br>0.022_±_0.093<br>0.007_±_0.030<br>0.036_±_0.131<br>0.011_±_0.039<br>**0.047**_±_**0.143**<br>**0.019**_±_**0.058**|
|---|---|---|---|---|---|---|---|---|
||With_D⋆_<br>LLama3 8B<br>0.070<br>LLama3 70B<br>0.502<br>GPT-3.5 turbo<br>0.223<br>GPT-4 turbo<br>**0.636**|||0.154_±_0.139<br>0.330_±_0.244<br>0.135_±_0.122<br>0.257_±_0.150<br>**0.509**_±_**0.213**<br>0.237_±_0.145<br>0.164_±_0.242<br>0.149_±_0.212<br>0.116_±_0.174<br>**0.461**_±_**0.206**<br>0.482_±_0.160<br>**0.378**_±_**0.174**|||||
||No Knowledge<br>LLama3 8B<br>0.023<br>LLama3 70B<br>0.037<br>GPT-3.5 turbo<br>0.059<br>GPT-4 turbo<br>**0.074**|||0.137_±_0.159<br>0.258_±_0.274<br>0.119_±_0.141<br>0.246 _±_0.148<br>**0.504**_±_**0.222**<br>0.227_±_0.148<br>0.161_±_0.238<br>0.148_±_0.215<br>0.113_±_0.171<br>**0.410**_±_**0.208**<br>0.443_±_0.191<br>**0.324**_±_**0.182**|||||
||||||||||
|0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0|||A|cc|Comp|Faith||LLama3<br>GPT-3.5<br>GPT-4<br>Endocrinology|
||||||||||
||LLa|ma3<br>GPT-3.5<br>GPT-4<br>Cardiology|LLama3<br>GPT-3.5<br>GPT-<br>Gastroenterology|4|LLama3<br>GPT-3.5<br>GPT-4<br>Neurology|LLama3<br>GPT-3.5<br>G<br>Pulmonology|PT-4||



Figure 5: Performance of LLama3 70B, GPT-3.5, and GPT-4 under different medical domains. We use the task with _G_ . 

observational performance may indicate that the models lack the ability to associate premises and text in _R_ , which are often superficially different though semantically consistent. 

LLMs may undergo inherent challenges for evaluation when no external knowledge is supplied. They may have the knowledge to diagnose but cannot make consistent observations and explanations that our task expects through _K_ . To explore this, we evaluate two settings: (1) giving _D[⋆]_ and (2) no knowledge is supplied to a model (shown in Table 4). The prompts used for this setup are detailed in the supplementary material. We do not evaluate the accuracy of disease category prediction as it is basically the same as Table 3. We can clearly see that with _D[⋆]_ , GPT-4’s diagnostic and observational scores are comparable to those of the task with _K_ , though explanatory performance is much worse. Without any external knowledge, the diagnostic accuracy is also inferior.[7] The deteriorated performance can be attributed to inconsistent wording of diagnosis names, which makes evaluation tough. High observational scores imply that observations in _R_ can be identified without relying on external knowledge. There can be some cues to spot them. 

**Performance in individual domains.** Figure 5 summarizes the performance of LLama3 70B, GPT3.5, and GPT-4 across different medical domains, evaluated using _Acc_[diag] , _Obs_[comp] (Comp), and _Exp_[all] (Faith). Neurology gives the best diagnostic accuracy, where LLama3 achieved an accuracy of 0.779. GPT-4 also performed well (0.753). In terms of _Obs_[comp] and _Exp_[all] , GPT-4’s results were 0.437 and 0.280, respectively. However, GPT-4 yields a higher diagnostic accuracy score while a lower explanatory score, suggesting that the observations captured by the model or their rationalizations differ from human doctors. 

**Diagnostic reasoning under conditions of incomplete observation.** In real-world scenarios, doctors often have to make diagnoses based on incomplete information. To explore this, we conducted experiments on the 73 amended cases which originally lack observation to the final diagnosis (refer to supplementary for detailed introduction of amended data point). One set of experiments used the unmodified original notes, labeled as "Original," while the other set used notes with added observations labeled as "Amended." We tested three models—Llama3 70B, GPT-3.5-turbo, and GPT-4 turbo—under two settings: one with only the procedural graph _G_ and the other with the complete knowledge graph _K_ . The results are presented in Tables 5 and 6. We can observe that in both _G_ and _K_ settings, the performance on the Amended data was consistently better across all metrics compared to the Original data. This suggests that even a single added observation can significantly impact the model’s diagnostic reasoning. 

> 7We understand this comparison is unfair, as the prompts differ. We intend to give a rough idea about the challenge without external knowledge. 

8 

Table 5: Amendment ablation study using _G_ . 

|Setting<br>Models|Diagnosis<br>_Acc_cat<br>_Acc_diag|Observation<br>_Obs_pre<br>_Obs_rec<br>_Obs_comp|Explanation<br>_Exp_com<br>_Exp_all|
|---|---|---|---|
|Original<br>LLama3 70B<br>GPT-3.5 turbo<br>GPT-4 turbo|0.547<br>0.273<br>0.507<br>0.273<br>0.616<br>0.328|0.225_±_0.143<br>0.472_±_0.144<br>0.253_±_0.138<br>0.393_±_0.216<br>0.355_±_0.174<br>0.278_±_0.151<br>0.446_±_0.211<br>0.418_±_0.164<br>0.340_±_0.178|0.216_±_0.271<br>0.073_±_0.087<br>0.207_±_0.305<br>0.062_±_0.093<br>0.242_±_0.324<br>0.098_±_0.137|
|Amended<br>LLama3 70B<br>GPT-3.5 turbo<br>GPT-4 turbo|0.698<br>0.534<br>0.671<br>0.411<br>0.726<br>0.547|0.250_±_0.173<br>0.507_±_0.134<br>0.240_±_0.129<br>0.487_±_0.206<br>0.351_±_0.152<br>0.310_±_0.145<br>0.546_±_0.184<br>0.465_±_0.148<br>0.412_±_0.171|0.296_±_0.354<br>0.133_±_0.142<br>0.272_±_0.321<br>0.092_±_0.118<br>0.391_±_0.374<br>0.180_±_0.186|



Table 6: Amendment ablation study using _K_ . 

|Setting<br>Models|Diagnosis<br>_Acc_cat<br>_Acc_diag|Observation<br>_Obs_pre<br>_Obs_rec<br>_Obs_comp|Explanation<br>_Exp_com<br>_Exp_all|
|---|---|---|---|
|Original<br>LLama3 70B<br>GPT-3.5 turbo<br>GPT-4 turbo|0.575<br>0.219<br>0.548<br>0.233<br>0.616<br>0.260|0.109_±_0.233<br>0.443_±_0.171<br>0.203_±_0.186<br>0.293_±_0.243<br>0.218_±_0.198<br>0.184_±_0.166<br>0.452_±_0.241<br>0.410_±_0.211<br>0.349_±_0.223|0.304_±_0.388<br>0.114_±_0.135<br>0.251_±_0.357<br>0.072_±_0.106<br>0.467_±_0.437<br>0.220_±_0.256|
|Amended<br>LLama3 70B<br>GPT-3.5 turbo<br>GPT-4 turbo|0.685<br>0.537<br>0.657<br>0.465<br>0.712<br>0.589|0.261_±_0.195<br>0.493_±_0.230<br>0.277_±_0.171<br>0.390_±_0.227<br>0.272_±_0.194<br>0.232_±_0.156<br>0.534_±_0.214<br>0.452_±_0.180<br>0.401_±_0.201|0.452_±_0.407<br>0.185_±_0.194<br>0.401_±_0.394<br>0.127_±_0.145<br>0.607_±_0.442<br>0.286_±_0.258|



For Cardiology and Endocrinology, the diagnostic accuracy of the models is relatively low (GPT4 achieved 0.458 and 0.468, respectively). Nevertheless, _Obs_[comp] and _Exp_[all] are relatively high. Endocrinology results in lower diagnostic accuracy and higher explanatory performance. A smaller gap may imply that in these two domains, successful predictions are associated with observations similar to those of human doctors, and the reasoning process may be analogous. Conversely, in Gastroenterology, higher _Acc_[cat] ) is accompanied by lower _Obs_[comp] and _Exp_[all] (especially for LLama3), potentially indicating a significant divergence in the reasoning process from human doctors. Overall, DiReCT demonstrates that the degree of alignment between the model’s diagnostic reasoning ability and that of human doctors varies across different medical domains. 

**Reliability of automatic evaluation.** We randomly pick out 100 samples from DiReCT and their prediction by GPT-4 over the task with _G_ to assess the consistency of our automated metrics to evaluate the observational and explanatory performance in Section 3.3 to human judgments. Three physicians joined this experi- 

ment. For each prediction ˆ _o ∈ O_[ˆ] , they are asked to find a similar observation in ground truth _O_ . For explanatory metrics, they verify if each prediction _z_ ˆ _∈ E_[ˆ] for _o_ ˆ _∈ O_[ˆ] align with groundtruth _z ∈E_ corresponding to _o_ . A prediction and a ground truth are deemed aligned for both assessments if at least 

Table 7: Consistency of automated evaluation with human judgments. Evaluated by mean and confidence interval (CI). 

|Model|Observation<br>Mean<br>95% CI|Rationalization<br>Mean<br>95% CI|
|---|---|---|
|LLama3 8B<br>GPT-4 turbo|0.887<br>0_._844_∼_0_._878<br>0.902<br>0_._830_∼_0_._863|0.835<br>0_._759_∼_0_._818<br>0.876<br>0_._798_∼_0_._853|



two specialists agree. We compare LLama3’s and GPT-4’s judgments to explore if there is a gap between these LLMs. As summarized in Table 7, GPT-4 achieves the best results, with LLama3 8B also displaying a similar performance. From these results, we argue that our automated evaluation metrics are consistent with human judgments, and LLama3 is sufficient for this evaluation, allowing the cost-efficient option. Detailed analysis is available in the supplementary material. 

**Prediction examples.** Figure 6 shows a sample generated by GPT-4. The ground-truth PDD of the input clinical note is `Hemorrhagic Stroke` . In this figure, purple, orange, and red indicate explanations only in the ground truth, only in prediction, and common in both, respectively; therefore, red is a successful prediction of an explanation, while purple and orange are a false negative and false positive. GPT-4 treats the observation of `aurosis fugax` as the criteria for diagnosing `Ischemic Stroke` . However, this observation only supports `Suspected Stroke` . Conversely, observation `thalamic hematoma` , which is the key indicator of `Hemorrhagic Stroke` , is regarded as a less important clue. Such observation-diagnosis correspondence errors lead to the model’s misdiagnosis. In Figure 7, we can observe that GPT-4 can find the key observation for the diagnosis of GERD, 

9 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0010-00.png)


**----- Start of picture text -----**<br>
Present Illne : He underwent a right carotid ********************,<br>and per notes, it was uneventful. This was done as an elective Transient vision loss typically Carotid artery stenosis is an<br>procedure after he had episodes of amaurosis fugax in *********** indicates a transient ischemic important cause of insufficient  Suspected<br>******** days ago, which, on evaluation, showed significant (more attack, often associated with blood flow to the brain and is Stroke<br>than ******* percent) carotid stenosis. ... carotid artery disease. associated with  risk of stroke<br>Past Medical History:  +HTN, +Diverticulosis, +CHF ...<br>Physical Exam : Mental status: Awake, *****, doesn't verbalize.<br>Can only say ****** words. Comprehension is relatively spared,<br>can answer with ********** to yes and No type questions. ... CHF reduced ability of the heart to pump blood may<br>Pertinent Results:  CT HEAD W/O CONTRAST Study Date  lead to increase the risk of stroke<br>FINDINGS: A ****** cm left thalamic hematoma appears stable<br>when compared to ********** from outside the ******** imaged<br>approximately ******** ago.  There is an  increased amount of The presence of a thalamic Thalamus hematoma<br>layering hemorrhagesmall  amount  of   in the ************ of the left lateral ventricle. Aintraventricular  blood  is  noted  in  the hematoma is directly relatedto symptoms of stroke, means brain bleedingwhich is a common HemorrhagicStroke IschemicStroke<br>***************** of the right lateral ventricle, ******. There is indicating brain bleeding diagnostic criterion for<br>surrounding *******, which appears ****** from prior CT. ... which can lead to stroke hemorrhagic stroke<br>Clinical Note Rationale Diagnosis<br>**----- End of picture text -----**<br>


Figure 6: An example prediction for a clinical note with PDD of `Hemorrhagic Stroke` by GPT-4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0010-02.png)


**----- Start of picture text -----**<br>
Chief Complaint : epigastric and substernal chest pain<br>Present Illne :  suspected PBC with severe epigastric pain that Common symptoms of GERD Epigastric and substernal chest<br>radiates to her mid-sternal area beginning at ** AM. She noted include chest pain that can be pain are atypical and typical<br>gradual ************************minutes. It did not radiate to her substernal or epigastric. symptoms of GERD,respectively.  Suspected<br>back, and was similar in character to past episodes. However, she GERD<br>felt the  pain was much more severe, and  did not respond to her<br>usual reflux techniques (drinking water, taking tums, and drinking a Hiatal hernia and erosions Erosions at the GE junction<br>lidocaine water mixture). She denied SOB, chest pain, palpitations, at the gastroesophageal may be an endoscopic<br>nausea, *****************is. She also denies change in **********, junction are common finding of GERD but was<br>such as melena or BRBPR. Endoscopy showed hiatal hernia and findings in GERD not graded.<br>erosions at the GE junction that were shown to be benign on<br>pathology ...<br>Past Medical History:  ... Indicates absence of erosive damage typically seen in severe<br>Pertinent Results:  EGD: Normal mucosa in the esophagus, GERD, but does not rule out GERD as symptoms can occur<br>stomach, and duodenum. *********** polyp in the upper without visible mucosal damage.<br>stomach.  ***********************************  part  of  the GERD<br>duodenum. EKG: upright axis, sinus rhythm, regular rate at ~60<br>bpm, intervals wnl, no acute ST changes. ************* reflux AET greater than 4% on pH-impedance<br>monitor: total AET:6.5% on pH-impedance monitoring.  monitoring supports the diagnosis of GERD<br>Clinical Note Rationale  Diagnosis<br>**----- End of picture text -----**<br>


Figure 7: An example prediction for a clinical note with PDD of `GERD` by GPT-4 

which is consistent with human in both observation and rationale. However, it still lacks the ability to identify all observations. More samples are available in the supplementary material. 

## **6 Conclusion and Limitations** 

We proposed DiReCT as the first benchmark for evaluating the diagnostic reasoning ability of LLMs with interpretability by supplying external knowledge as a graph. Our evaluations reveal a notable disparity between current leading-edge LLMs and human experts, underscoring the urgent need for AI models that can perform reliable and interpretable reasoning in clinical environments. DiReCT can be easily extended to more challenging settings by removing the knowledge graph from the input, facilitating evaluations of future LLMs. 

**Limitations.** DiReCT encompasses only a subset of disease categories and considers only one PDD, omitting the inter-diagnostic relationships due to their complexity—a significant challenge even for human doctors. Additionally, our baseline may not use optimal prompts or address issues related to hallucinations in task responses. Our dataset is solely intended for model evaluation but not for use in clinical environments. The use of the diagnostic knowledge graph is also limited to serving merely as a part of the input and once a knowledge graph is provided, the focus shifts to whether the LLM follows the graph’s rules well (refer to supplementary). Future work will focus on constructing a more comprehensive disease dataset and developing an extensive diagnostic knowledge graph. 

## **Acknowledgments and Disclosure of Funding** 

This work was supported by World Premier International Research Center Initiative (WPI), MEXT, Japan. This work is also supported by JST ACT-X Grant Number JPMJAX24C8, JSPS KAKENHI No. 24K20795 and No. JP23H00497, CREST Grant No. JPMJCR20D3, JST FOREST Grant No. JPMJFR216O, and Dalian Haichuang Project for Advanced Talents. 

10 

## **A Details of DiReCT** 

## **A.1 Data Statistics** 

Table 8: Disease statistics of DiReCT. 

|Domains|Categories|# samples|_|Di|_|_|D⋆_<br>_i |_|References|
|---|---|---|---|---|---|
||Acute Coronary Syndromes|65|6|3|[Byrne et al.,2024,Kitaoka et al.,2020]|
||Aortic Dissection|14|3|2|[Members et al.,2022]|
||Atrial Fibrillation|10|3|2|[Joglar et al.,2024]|
|Cardiology|Cardiomyopathy|9|5|4|[Ommen et al.,2020]|
||Heart Failure|52|6|3|[Heidenreich et al.,2022]|
||Hyperlipidemia|2|2|1|[Su et al.,2021,Mach et al.,2020]|
||Hypertension|32|2|1|[Unger et al.,2020]|
||Gastritis|27|5|3|[Shah et al.,2021,of Gastroenterology et al.,2023,Banks et al.,2019,Chow et al.,2010]|
|Gastroenterology|Gastroesophageal Refux Disease<br>Peptic Ulcer Disease|41<br>28|2<br>3|1<br>2|[Gyawali et al.,2024]<br>[Kavitt et al.,2019,Tarasconi et al.,2020]|
||Upper Gastrointestinal Bleeding|7|2|1|[Barkun et al.,2019]|
||Alzheimer|10|2|1|[McKhann et al.,1984]|
||Epilepsy|8|3|2|[Igaku-Shoin-Ltd.,2018]|
|Neurology|Migraine|4|3|2|[Lipton et al.,2001,Eigenbrodt et al.,2021,Headache Classifcation Committee of the International Headache Society (IHS),2018]|
||Multiple Sclerosis|27|6|4|[Lublin,2005,Brownlee et al.,2017]|
||Stroke|28|3|2|[Kleindorfer et al.,2021]|
||Asthma|13|7|5|[Qaseem et al.,2011,Bateman et al.,2007,Baos et al.,2018]|
||COPD|19|6|4|[Gupta et al.,2013]|
|Pulmonology|Pneumonia|20|4|2|[Olson and Davis,2020,RECOMMENDATIONS,2012,Niederman et al.,2001]|
||Pulmonary Embolism|35|5|3|[Konstantinides et al.,2020]|
||Tuberculosis|5|3|2|[Lewinsohn et al.,2017]|
||Adrenal Insuffciency|20|4|3|[Charmandari et al.,2014,Yanase et al.,2016,Bornstein et al.,2016]|
|Endocrinology|Diabetes<br>Pituitary|13<br>12|4<br>4|2<br>3|[ElSayed et al.,2023]<br>[Tritos and Miller,2023,Drummond et al.,2019,Cooper and Melmed,2012,Mayson and Snyder,2014]|
||Thyroid Disease|10|6|4|[AlexanderErik et al.,2017]|



Table 8 provides a detailed breakdown of the disease categories included in DiReCT. The column labeled # samples indicates the number of data points. The symbols _|Di|_ and _|Di[⋆][|]_[ denote the total] number of diagnoses (diseases) and PDDs, respectively. Existing guidelines for diagnosing diseases were used as References, forming the foundation for constructing the diagnostic knowledge graphs. As some premise may not included in the referred guidelines. During annotation, physicians will incorporate their own knowledge to complete the knowledge graph. 

## **A.2 Amended Data Points** 

Our proposed dataset aims to evaluate whether LLMs can provide a complete diagnostic reasoning process comparable to that of human doctors. To achieve this, we intended to select notes from the MIMIC database that contain comprehensive signs and symptoms as observations, enabling physicians to annotate the notes leading to a final PDD. For disease category like heart failure, MIMIC offers ample data, allowing us to choose notes with complete observations. However, for PDDs such as bacterial pneumonia, the number of relevant notes is limited, and many lack critical evidence necessary for diagnosis (e.g., sputum culture). We observed that in some notes, the section under the title ’sputum culture’ was left blank. We suspect that this might be due to some information being missed in MIMIC. To annotate such cases, we ask physicians add the necessary observations to support the diagnosis. In total, we made amendments to 73 notes. These notes all lacked evidence for a final PDD diagnosis, and in each note, only one observation was added as evidence. Thus, the modifications to the original content of the notes were minimal. For example, in a note where the PDD is bacterial pneumonia, we only added the following description under ’pertinent results’: ’Multiple organisms consistent with Haemophilus influenzae.’ 

To better illustrate the structure of our dataset and identify which data has been amended and what content has been added, we have provided a detailed CSV file on GitHub (https://github.com/wbw520/DiReCT/tree/master/utils/data_loading_analysisi). This file contains six columns, which record the following information: Disease Category, PDD, Data Root, Whether Amended, Amended Part, and Amended Content. The Data Root column records the path and filename of each note. We have stored the original note information and our annotations within a JSON file. The version submitted for review to PhysioNet follows the same storage format. In the Whether Amended column, notes that have been amended are marked as ’Yes,’ with the Amended Part and Amended Content columns specifying which part of the note was modified and what content was added. Additionally, we have provided several synthetic annotated samples (non-MIMIC data) on GitHub, along with detailed instructions on the format of the annotated data and how to parse each JSON file. 

## **A.3 Structure of Knowledge Graph** 

We first show the notations definition on Table 9. The entire knowledge graph, denoted as _K_ , is stored in separate JSON files, each corresponding to a specific disease category _i_ as _Ki_ . Each _Ki_ comprises a procedural graph _Gi_ and the corresponding premise _p_ for each disease. As illustrated in Figure 8, the procedural graph _Gi_ is stored under the key "Diagnostic" in a dictionary structure. A key with an empty list as its value indicates a leaf diagnostic node as _d[⋆]_ . The premise for each disease is 

11 

|**Notation**|**Description**|**Notation**|**Description**|
|---|---|---|---|
|_R_|The whole content of input note.|_U_|The narrowing-down module.|
|_r_|One data section of the input note.|_W_|Theperception module.|
|_D_|Disease collection.|_V_|The reasoningmodule.|
|_D⋆_|PDD collection.|_{dn}_|Collection of children diagnosis.|
|_D⋆_<br>_i_|PDD collection for disease_i_.|_Accdiag_|Diagnosis accuracyfor_d⋆_.|
|_d⋆_|A PDD disease.|_Acccat_|Diagnosis accuracyfor category.|
|_dt_|Diagnosis at_t_-th iteration.|_Obspre_|Precision of observation.|
|_d_|A diagnosis in _G_.|_Obsrec_|Recall of observation.|
|_G_|Proceduralgraphs.|_Obscomp_|Completeness of observation.|
|_gi_|Procedural subgraph for disease|_Expcom_|Completeness of explanation.|
|_K_|knowledgegraphs.|_Expall_|Completeness of all explanation.|
|_ki_|knowledge subgraphs for disease_i_.|_M_|An language model.|
|_Pi_|Supportingedge collection for_ki_.|_E_|Collection of annotated deductions.|
|_p_|Apremise defned in_K_.|||
|_Fi_|Procedural edge collection for _gi_.|||
|_O_|Collection of annotated observation.|||
|_o_|An annotated observation.|||
|_z_|Rationale for a deduction.|||
|_d_0|Root diagnosis for _gi_.|||



Table 9: Notations defined in this paper. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0012-02.png)


**----- Start of picture text -----**<br>
{"Diagnostic":<br>        {"Suspected Heart Failure":<br>        {"Strongly Suspected Heart Failure":<br>                {"Heart Failure":<br>                        {"HFrEF": [],<br>                        "HFmrEF": [],<br>                        "HFpEF": []}}}},<br>"Knowledge":<br>        {"Suspected Heart Failure":<br>                {"Risk Factors": "CAD; Hypertension; Valve disease; Arrhythmias; CMPs; Congenital heart disease; Infective; Drug-induced;<br>                                         Infiltrative, Storage disorders, Endomyocardial disease, Pericardial disease, Metabolic, Neuromuscular disease",<br>                "Symptoms": Breathlessness; Orthopnoea; Paroxysmal nocturnal dyspnoea; Reduced exercise tolerance; Fatigue; tiredness; increased<br>                                      time to recover after exercise; Ankle swelling; Nocturnal cough; Wheezing; Bloated feeling; Loss of appetite;<br>                                      Confusion (especially in the elderly); Depression; Palpitation; Dizziness; Syncope",<br>                "Signs": "Elevated jugular venous pressure; Hepatojugular reflux; Third heart sound (gallop rhythm); Laterally displaced apical<br>                                impulse; Weight gain (>2 kg/week); Weight loss (in advanced HF); Tissue wasting (cachexia); Cardiac murmur; Peripheral<br>                                edema (ankle, sacral, scrotal); Pulmonary crepitations; Pleural effusion; Tachycardia; Irregular pulse; Tachypnoea; Cheyne-Stokes<br>                                respiration; Hepatomegaly; Ascites; Cold extremities; Oliguria;  Narrow pulse pressure."},<br>        "Strongly Suspected Heart Failure": "NT-proBNP > 125 pg/mL; BNP > 35 pg/mL",<br>        "Heart Failure": "Abnormal findings from echocardiography\uff1aLV mass index>95 g/m2 (Female), > 115 g/m2 (Male); Relative wall thickness >0.42;<br>                                  LA volume index>34 mL/m2; E/e ratio at rest >9; PA systolic pressure >35 mmHg; TR velocity at rest >2.8 m/s",<br>        "HFrEF": "LVEF<40%",<br>        "HFmrEF": "LVEF41-49%",<br>        "HFpEF": "LVEF>50%"}}<br>**----- End of picture text -----**<br>


Figure 8: A sample of knowledge graph for `Heart Failure` . Each premise under the key of "Knowledge" is separated with ";". 

saved under the key of "Knowledge" with the corresponding disease name as an index. For all the root nodes (e.g., `Suspected Heart Failure` ), we further divide the premise into "Risk Factors", "Symptoms", and "Signs". Note that each premise is separated by ";". 

Our knowledge graph was directly constructed by human physicians who followed authoritative diagnostic guidelines and incorporated their clinical experience. For Cardiology, Gastroenterology, Neurology, Pulmonology, and Endocrinology, the knowledge graph was built by 2, 1, 2, 2, and 1 specialists from the respective departments. The construction process involved first defining the procedural graph _gi_ for each category, followed by supplementing _gi_ with the detailed premises corresponding to each diagnosis d to build _ki_ . The complete knowledge graphs are available on GitHub (https://github.com/wbw520/DiReCT/tree/master/utils/data_annotation). 

12 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0013-00.png)


Figure 9: Demonstration of our annotation tool. 

Table 10: Prompt for _narrowing-down_ module. 

Input Prompt Suppose you are one of the greatest AI scientists and medical expert. Let us think step by step. You will review a clinical ’Note’ and your ’Response’ is to diagnose the disease that the patient have for this admission. All possible disease options are in a list structure: {disease_option}. Note that you can only choose one disease from the disease options and directly output the origin name of that disease. Now, start to complete your task. Don’t output any information other than your ’Response’. ’Note’: {note} Your ’Response’: 

## **A.4 Annotation and Tools** 

We have developed proprietary software for annotation purposes. As depicted in Figure 9, annotators are presented with the original text as observations _o_ and are required to provide rationales ( _z_ ) to explain why a particular observation _o_ supports a disease _d_ . The left section of the figure, labeled Input1 to Input6, corresponds to different parts of the clinical note, specifically the chief complaint, history of present illness, past medical history, family history, physical exam, and pertinent results, respectively. Annotators will add the raw text into the first layer by left-clicking and dragging to select the original text, then right-clicking to add it. After each observation, a white box will be used to record the rationales. Finally, a connection will be made from each rationale to a disease, represented in a grey box. The annotation process strictly follow the knowledge graph. Both the final annotation and the raw clinical note will be saved in a JSON file. We provide the code to compile these annotations and detailed instructions for using our tool on GitHub. 

## **A.5 Access to DiReCT** 

Implementation code and annotation tool are available through https://github.com/wbw520/DiReCT. Data will be released through PhysioNet due to safety issues according to the license of MIMIC-IV (PhysioNet Credentialed Health Data License 1.5.0). We will use the same license for DiReCT. The download link will be accessible via GitHub. We confirm that this GitHub link and data link are always accessible. We confirm that we will bear all responsibility in case of violation of rights. 

13 

Table 11: Prompt for _perception_ module. 

Input Prompt Suppose you are one of the greatest AI scientists and medical expert. Let us think step by step. You will review a part of clinical "Note" from a patient. The disease for which the patient was admitted to hospital this time is {disease}. Your task is to extract the original text as confidence "Observations" that lead to {disease}. Here are some premise for the diagnosis of this disease category. You can refer them for your task. Premise are: {premise} Note that you also need to briefly provide the "Reason" for your extraction. Note that both "Observations" and "Reason" should be string. Note that your "Response" should be a list structure as following : [["Observation", "Reason"], ......, ["Observation", "Reason"]] Note that if you can’t find any "Observation" your "Response" should be: []. Now, start to complete your task. Note that you should not output any information other than your "Response". "Note": {note} Note that you should not output any information other than your "Response". Your "Response": 

Table 12: Prompt for _reasoning_ module. 

Input Prompt Suppose you are one of the greatest AI scientists and medical expert. Let us think step by step. You will receive a list of "Observations" from a clinical "Note". These "Observations" are possible support to diagnose {disease}. Based on these "Observations", you need to diagnose the "Disease" from the following options: {disease_option}. Here are some golden standards to discriminate diseases. You can refer them for your task. Golden standards are: {premise} Note that you can only choose one "Disease" from the disease options and directly output the name in disease options. Note that you also required to select the "Observations" that satisfy the golden standard to diagnose the "Disease" you choose. Note that you also required to provide the "Reason" for your choice. Note that your "Response" should be a list structure as following :[["Observation", "Reason", "Disease"], ......, ["Observation", "Reason", "Disease"]] Note that if you can’t find any "Observation" to support a disease option, your "Response" should be: None Now, start to complete your task. Note that you should not output any information other than your "Response". "Observations": {observation} Note that you should not output any information other than your "Response". Your "Response": 

## **B Implementation of Baseline Method** 

## **B.1 Prompt Settings** 

In this section, we demonstrate the prompt we used for each module (From Table 10-12 for _narrowingdown_ , _perception_ , and _reasoning_ module, respectively). 

In Table 10, {disease_option} is the name for all disease categories, and {note} is the content for the whole clinical note. The response for the model is the name of a possible disease[ˆ] _i_ . 

In Table 11, {disease} is the disease category name predicted in _narrowing-down_ . The content marked blue is the premise, which is only provided during the _K_ setting. In this module, {premise} is offered with all information in the knowledge graph. Different to _narrowing-down_ , {note} is implemented for each clinical data _R_ = _{r}_ and the outputs are combined together for _O_[ˆ] and _E_[ˆ] . 

In Table 12, {disease} is the disease category name and {disease_option} is consisted by the children nodes _{dn}_ . Similarly, the premise on the blue is only available for the _K_ setting. It provides the premise that are criteria for the diagnosis of each children node. {observation} is the extracted _O_[ˆ] in previous step. We provide all the prompts and the complete implementation code on GitHub. 

## **B.2 Diagnostic Reasoning Under Conditions of Incomplete Observation** 

In real-world scenarios, doctors often have to make diagnoses based on incomplete information. To explore this, we conducted experiments on the 73 amended cases. One set of experiments used the unmodified original notes, labeled as "Original," while the other set used notes with added observations, labeled as "Amended." We tested three models—Llama3 70B, GPT-3.5-turbo, and GPT-4 turbo—under two settings: one with only the procedural graph _G_ and the other with the complete knowledge graph _K_ . The results are presented in Tables 5 and 6. We can observe that in 

14 

both _G_ and _K_ settings, the performance on the Amended data was consistently better across all metrics compared to the Original data. This suggests that even a single added observation can significantly impact the model’s diagnostic reasoning. 

Additionally, we found that under the Amended data, using _K_ led to both better diagnostic outcomes and improved explanability, aligning with the analysis in our paper. However, when using _K_ on the Original data, while explanability improved, diagnostic accuracy actually decreased. 

We conducted a detailed analysis of the 73 Original data’s results from GPT-4. We found that GPT-4 was still able to correctly deduce the final PDD in 24 cases using _G_ and 19 cases using _K_ . This indicates that the model possesses some level of uncertain reasoning capability. However, upon further inspection, we found that in some cases, the model used completely irrational observations as evidence, such as directly using "cough" as evidence for diagnosing "bacterial pneumonia". Additionally, there were 7 cases using _G_ and 13 cases using _K_ where the reasoning stopped before the final PDD diagnosis. This suggests that the model recognized the lack of sufficient evidence to derive the PDD and adhered faithfully to the diagnostic knowledge graph. Moreover, using appeared to help the model better understand this limitation, however, decrease the accuracy. These results indicate that employing the knowledge graph acts more like a trade-off: using only _G_ results in a higher tendency for uncertain reasoning, while using the full _K_ makes the model more cautious. 

**Limitation of current implementation.** Once a knowledge graph is provided, the focus shifts to whether the LLM follows the graph’s rules well. However, we consider the knowledge graph as an inferential framework rather than a set of rules. This framework provides decision-making paths for the LLM, but the LLM still needs to perform reasoning within it. Even when strictly following the knowledge graph, the LLM still needs to perform semantic analysis and context understanding in order to select the node that best suits the current situation among multiple possible paths (sub-nodes) in the knowledge graph. Therefore, the role of the LLM in this process is not merely to ’follow the rules,’ but to make logical path selections based on its understanding of the input data, which itself is a reflection of reasoning ability. This often requires the model or algorithm to consider previous steps in each stage of reasoning and to update observations accordingly. Even revise or backtrack the diagnosis step. However, our baseline method did not account for this and thus cannot fully exploit this capability of the LLM, which is a current limitation. We did try some designs to give those abilities to LLMs, such as providing previous steps of reasoning for the current stage as input prompts or update observations. However, even GPT-4 cannot show high instruction following ability to realize them (maybe the input is too long or prompt setting problems). 

For evaluation, we used diagnostic processes annotated by human doctors as ground truth. Therefore, whether the KG is provided or not, the model’s output needs to align with the ground truth. Our dataset allows for evaluation with and without the KG, but our baseline method is not effective at handling scenarios without the KG (this is much more challenging). How to utilize and explore the LLM’s reasoning ability in this scenario is one of our future research directions. 

## **B.3 Details of Automatic Evaluation** 

The automatic evaluation is realized by LLama3 8B. We demonstrate the prompt for this implement in Table 13 (for observation) and Table 14 (for rationalization). Note that we do not use few-shot samples for the evaluation of observation. In Table 13, {gt_observation} and {pred_observation} are from model prediction and ground-truth. As this is a simple similarity comparison task to discriminate whether the model finds similar observations to humans, LLama3 itself have such ability. We do not strict to exactly match due to the difference in length of extracted raw text (as long as the observation expresses the same description). In Table 14, {gt_reasoning} and {pred_reasoning} are from model prediction and ground-truth. We require the rationale to be complete (content of the expression can be understood from the rationale alone) and meaningful; therefore, we provide five samples for this evaluation. We also provide all the prompts and the complete implementation code on GitHub. 

For human evaluation, among the three specialists, two are from Cardiology and one is from Gastroenterology. Given that the notes originate from different medical domains, there is a possibility that the specialists may not be entirely accurate. However, this evaluation does not demand highly specialized knowledge, and it can be adequately covered by their expertise. 

We also included an experimental result comparing the judgment differences between Llama3 8B and GPT-4 Turbo. The evaluation was performed on the diagnostic outcomes (across the entire 

15 

Table 13: Prompt for evaluation of observation. 

Input Prompt Suppose you are one of the greatest AI scientists and medical expert. Let us think step by step. You will receive two "Observations" extracted from a patient’s clinical note. Your task is to discriminate whether they textually description is similar? Note that "Response" should be one selection from "Yes" or "No". Now, start to complete your task. Don’t output any information other than your "Response". "Observation 1": {gt_observation} "Observation 2": {pred_observation} Your "Response": 

Table 14: Prompt for evaluation of rationalization. 

Input Prompt Suppose you are one of the greatest AI scientists and medical expert. Let us think step by step. You will receive two "Reasoning" for the explanation of why an observation cause a disease. Your task is to discriminate whether they explain a similar medical diagnosis premise? Note that "Response" should be one selection from "Yes" or "No". Here are some samples: Sample 1: "Reasoning 1": Facial sagging is a classic symptom of stroke "Reasoning 2": Indicates possible facial nerve palsy, a common symptom of stroke "Response": Yes Sample 2: "Reasoning 1": Family history of Diabetes is an important factor "Reasoning 2": Patient’s mother had a history of Diabetes, indicating a possible genetic predisposition to stroke "Response": Yes Sample 3: "Reasoning 1": headache is one of the common symptoms of HTN "Reasoning 2": Possible symptom of HTN "Response": No Sample 4: "Reasoning 1": Acute bleeding is one of the typical symptoms of hemorrhagic stroke "Reasoning 2": The presence of high-density areas on Non-contrast CT Scan is a golden standard for Hemorrhagic Stroke "Response": No Sample 5: "Reasoning 1": Loss of strength on one side of the body, especially when compared to the other side, is a common sign of stroke "Reasoning 2": Supports ischemic stroke diagnosis "Response": No Now, start to complete your task. Don’t output any information other than your "Response". "Reasoning 1": {gt_reasoning} "Reasoning 2": {pred_reasoning} Your "Response": 

dataset) from Llama3 70B and GPT-4 Turbo, using _G_ as additional knowledge. We calculated the consistency rate for matching observations and the corresponding rationalization. As shown in Table 15, the differences in judgment between the two models are not obvious and are more consistent in observation discrimination. There are also some variations across different disease domains, with the highest similarity in observation discrimination found in Endocrinology, while the rationalization is most similar in Neurology. 

Additionally, we provided results using GPT-4 Turbo for automatic evaluation, compared to those shown in Table 16 (which used Llama3 8B). The results indicate that GPT-4 Turbo tends to yield higher observation matching and more stringent rationalization discrimination. However, the largest difference does not exceed 5%. Considering the cost of GPT-4, Llama3 8B is a more efficient option. 

## **B.4 More Prediction Samples** 

10 shows another sample generated by GPT-4. The ground-truth PDD of the input clinical note is and `Heart Failure (HF)` . In these figures, purple, orange, and red indicate explanations only in the ground truth, only in prediction, and common in both, respectively; therefore, red is a successful prediction of an explanation, while purple and orange are a false negative and false positive. 

16 

Table 15: Judgement consistency between LLama3 8B and GPT-4 turbo. 

|Domain|LLama3 70B<br>Observation<br>Rationalization|GPT-4 turbo<br>Observation<br>Rationalization|
|---|---|---|
|Cardiology<br>Gastroenterology<br>Neurology<br>Pulmonology<br>Endocrinology<br>Overall|0.885_±_0.095<br>0.761_±_0.268<br>0.862_±_0.088<br>0.676_±_0.361<br>0.846_±_0.090<br>0.831_±_0.211<br>0.808_±_0.131<br>0.703_±_0.317<br>0.911_±_0.104<br>0.783_±_0.304<br>0.869_±_0.102<br>0.734_±_0.321|0.827_±_0.146<br>0.861_±_0.273<br>0.810_±_0.167<br>0.755_±_0.316<br>0.856_±_0.124<br>0.963_±_0.106<br>0.786_±_0.152<br>0.779_±_0.287<br>0.868_±_0.145<br>0.793_±_0.340<br>0.838_±_0.144<br>0.806_±_0.305|



Table 16: Result of using GPT-4 turbo and LLama3 8B for automatic evaluation. 

|Judgement<br>Models|Observation<br>_Obs_pre<br>_Obs_rec<br>_Obs_comp|Explanation<br>_Exp_com<br>_Exp_all|
|---|---|---|
|GPT-4 turbo<br>LLama3 70B<br>GPT-4 turbo|0.317_±_0.161<br>0.576_±_0.195<br>0.294_±_0.159<br>0.465_±_0.190<br>0.514_±_0.157<br>0.408_±_0.201|0.348_±_0.300<br>0.107_±_0.118<br>0.437_±_0.335<br>0.187_±_0.191|
|LLama3 8B<br>LLama3 70B<br>GPT-4 turbo|0.277_±_0.146<br>0.537_±_0.192<br>0.256_±_0.142<br>0.446_±_0.207<br>0.491_±_0.180<br>0.371_±_0.186|0.395_±_0.320<br>0.112_±_0.110<br>0.475_±_0.363<br>0.199_±_0.181|



In Figure 10, the model’s predictions do not align well with those of a human doctor. Key observations, such as the relationships between BNP and LVEF, are incorrectly identified, leading to a final misdiagnosis. 

## **B.5 Experiments for No Extra Knowledge** 

We demonstrate the prompt used for _D[⋆]_ and no knowledge settings in Table 17 and Table 18, respectively. {note} is the text of whole clinical note and {disease_options} in Table 17 is the name of all leaf node _D[⋆]_ . 

## **B.6 Experimental Settings** 

All experiments are implemented with a temperature value of 0. All close sourced models are implemented in a local server with 4 NVIDIA A100 GPU. 

## **C Failed Attempts on DiReCT** 

In this section, we discuss some unsuccessful attempts during the experiments. 

**Extract observation from the whole clinical note.** We try to diagnose the disease and extract observation, and the corresponding rationale using the prompt shown in Table 19. The {note} is offered by the whole content in the clinical note. We find that even though the model can make the correct diagnosis, only a few observations can be extracted (no more than 4), which decreases the completeness and faithfulness. 

We also conducted an experiment to demonstrate the differences between two methods of observation extraction. The "Iteration" method is the one used in our paper, while the "Once" method is the one-time extraction method shown in Table 9. Each method was implemented under the condition of using GPT-4 turbo and Llama3 70B with _G_ as input and was evaluated based on the Completeness of Observations (Obs) metric. The results are presented in Table 20. We found that while the "Once" extraction method resulted in higher precision, it led to a significant drop in recall, severely impacting the final completeness metric. The "Once" method tends to capture fewer observations, which hinders the overall reasoning process. 

**End-to-End prediction.** We also try to output the whole reasoning process in one step (without iteration) when given observations. We show our prompt in Table 21. We find that using such a prompt model can not correctly recognize the relation between observation, rationale, and diagnosis. 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models_images/nips_2024_DiReCT_Diagnostic_Reasoning_for_Clinical_Notes_via_Large_Language_Models.pdf-0018-00.png)


**----- Start of picture text -----**<br>
Chief Complaint: Present History: He was seen ********************** anasarca. Atthat time, his lasix was increased from *********** to ****** BID. Inscrotal and leg swelling. common symptom of heart failure.Swelling in the legs can be a signof fluid retention, which is a Peripheral oedema is asign of heart failure  SuspectedHF<br>the ED initial vitals ***************************98% RA. Blood<br>pressure remained 200/90 throughout the ED course. Labs were Cardiac effusions are often<br>significant for ************************ since ___ (1.**********->3.2).EKG was consistent with priors (NSR, NANI, no ischemic associated with heart failure,indicating fluid overload or BNP ≥ 35 pg/mL is a strongvalue for heart failure Suspected HFStrongly<br>changes).  CXR showed mild pulmonary edema. He was given heart dysfunction<br>*********with good UOP. Bedside  cardiac ultrasound w/mild<br>effusion no evidence of tamponade physiology. Bedside scrotal*****************, no evidence of vascular compromise.... biomarker for heart failure, indicatingElevated proBNP levels are a BNP ≥ 35 pg/mL is a strongvalue for heart failure HF<br>Pertinet Results:  07:10AM BLOOD C3-142 C4-27  proBNP-5145 cardiac stress and heart dysfunction<br>...  The left atrium is moderately dilated. No atrial septal defect is<br>seen **************** Doppler. ... function is mildly depressed (LVEF= 45-50 %) without regional wallmotion abnormalities.  Overall  left ventricular systolic LVEF in the range of 45-50% suggestspreserved or mildly reduced systolicfunction, aligning with HFpEF 40≤LVEFcritiera for HFmrEF＜50% is the HFmrEF HFpEF<br>Clinical Note Rationale Diagnosis<br>**----- End of picture text -----**<br>


Figure 10: An example prediction for a clinical note with PDD of `HF` by GPT-4 

Table 17: Prompt for _D[⋆]_ setting. 

Input Prompt Suppose you are one of the greatest AI scientists and medical expert. Let us think step by step. You will review a clinical ’Note’ and your ’Response’ is to diagnose the disease that the patient have for this admission. All possible disease options are in a list structure: {disease_options}. Note that you can only choose one disease from the disease options and directly output the origin name of that disease. Now, start to complete your task. Don’t output any information other than your ’Response’. ’Note’: {note} Your ’Response’: 

## **D Ethical Considerations** 

Utilizing real-world EHRs, even in a de-identified form, poses inherent risks to patient privacy. Therefore, it is essential to implement rigorous data protection and privacy measures to safeguard sensitive information, in accordance with regulations such as HIPAA. We strictly adhere to the Data Use Agreement of the MIMIC dataset, ensuring that the data is not shared with any third parties. All experiments are implement on a private server. The access to GPT is also a private version. 

AI models are susceptible to replicating and even intensifying the biases inherent in their training data. These biases, if not addressed, can have profound implications, particularly in sensitive domains such as healthcare. Unconscious biases in healthcare systems can result in significant disparities in the quality of care and health outcomes among different demographic groups. Therefore, it is imperative to rigorously examine AI models for potential biases and implement robust mechanisms for ongoing monitoring and evaluation. This involves analyzing the model’s performance across various demographic groups, identifying any disparities, and making necessary adjustments to ensure equitable treatment for all. Continual vigilance and proactive measures are essential to mitigate the risk of biased decision-making and to uphold the principles of fairness and justice in AI-driven healthcare solutions. 

18 

Table 18: Prompt for no knowledge setting. 

Input Prompt Suppose you are one of the greatest AI scientists and medical expert. Let us think step by step. You will review a clinical ’Note’ and your ’Response’ is to diagnose the disease that the patient have for this admission. Note that you can only give one disease name and directly output the name of that "Disease". Now, start to complete your task. Don’t output any information other than your ’Response’. ’Note’: {note} Your ’Response’: 

Table 19: Prompt for extracting observation in one step. 

Input Prompt Suppose you are one of the greatest AI scientists and medical expert. Let us think step by step. You will review a clinical ’Note’, and your ’Response’ is to diagnose the disease that the patient has for this admission. All possible disease options are in a list structure: {disease_options}. Note that you can only choose one disease from the disease options and directly output the origin name of that disease. Note that you also need to extract original text as confidence "Observations" that lead to the "Disease" you selected. Note that you should extract all necessary "Observation". Note that you also need to briefly provide the "Reason" for your extraction. Note that both "Observations" and "Reason" should be string. Note that your "Response" should be a list structure as following :[["Observation", "Reason", "Disease"], ......, ["Observation", "Reason", "Disease"]] Now, start to complete your task. Don’t output any information other than your ’Response’. ’Note’ :{note} Your ’Response’: 

19 

Table 20: Comparison for using Iteration and Once for observation extraction. 

|Models|Iteration<br>_Obs_pre<br>_Obs_rec<br>_Obs_comp|Once<br>_Obs_pre<br>_Obs_rec<br>_Obs_comp|
|---|---|---|
|LLama3 70B<br>GPT-4 turbo|0.277_±_0.146<br>0.537_±_0.192<br>0.256_±_0.142<br>0.446_±_0.207<br>0.491_±_0.180<br>0.371_±_0.186|0.325 _±_0.207<br>0.324_±_0.147<br>0.185_±_0.107<br>0.567_±_0.268<br>0.287_±_0.156<br>0.244_±_0.147|



Table 21: Prompt for End-to-End prediction. 

Input Prompt Suppose you are one of the greatest AI scientists and medical expert. Let us think step by step. You will receive a list of "Observations" from a clinical "Note" for the diagnosis of stroke. Here is the diagnostic route of stroke in a tree structure: -Suspected Stroke -Hemorrhagic Stroke -Ischemic Stroke Here are some premise for the diagnosis of this disease. You can refer them for your task. Premise are: {premise} Based on these "Observations", starting from the root disease, your target is to diagnose one of the leaf disease. Note that you also required to provide the "Reason" for your reasoning. Note that your "Response" should be a list structure as following :[["Observation", "Reason", "Disease"], ......, ["Observation", "Reason", "Disease"]] Note that if you can’t find any "Observation" to support a disease option, your "Response" should be: None Now, start to complete your task. Note that you should not output any information other than your "Response". "Observations": {observation} Note that you should not output any information other than your "Response". Your "Response": 

## **References** 

- Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models. _arXiv preprint arXiv:2303.18223_ , 2023. 

Bonan Min, Hayley Ross, Elior Sulem, Amir Pouran Ben Veyseh, Thien Huu Nguyen, Oscar Sainz, Eneko Agirre, Ilana Heintz, and Dan Roth. Recent advances in natural language processing via large pre-trained language models: A survey. _ACM Computing Surveys_ , 56(2):1–40, 2023. 

- Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. _arXiv preprint arXiv:2305.10403_ , 2023. 

- Tianyu Han, Lisa C Adams, Jens-Michalis Papaioannou, Paul Grundmann, Tom Oberhauser, Alexander Löser, Daniel Truhn, and Keno K Bressem. Medalpaca–an open-source collection of medical conversational ai models and training data. _arXiv preprint arXiv:2304.08247_ , 2023. 

- Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. _Applied Sciences_ , 11(14):6421, 2021. 

- OpenAI. GPT-4 Technical Report. _CoRR_ , abs/2303.08774, 2023a. doi: 10.48550/arXiv.2303.08774. URL `https://doi.org/10.48550/arXiv.2303.08774` . 

- Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv preprint arXiv:2303.12712_ , 2023. 

- Harsha Nori, Nicholas King, Scott Mayer McKinney, Dean Carignan, and Eric Horvitz. Capabilities of gpt-4 on medical challenge problems. _arXiv preprint arXiv:2303.13375_ , 2023. 

20 

Valentin Liévin, Christoffer Egeberg Hother, Andreas Geert Motzfeldt, and Ole Winther. Can large language models reason about medical questions? _Patterns_ , 5(3), 2024. 

- Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. MedMCQA: A large-scale multi-subject multi-choice dataset for medical domain question answering. In _Conference on health, inference, and learning_ , pages 248–260. PMLR, 2022. 

- Dongfang Li, Jindi Yu, Baotian Hu, Zhenran Xu, and Min Zhang. ExplainCPE: A free-text explanation benchmark of chinese pharmacist examination. _arXiv preprint arXiv:2305.12945_ , 2023. 

- Hanjie Chen, Zhouxiang Fang, Yash Singla, and Mark Dredze. Benchmarking large language models on answering and explaining challenging medical questions. _arXiv preprint arXiv:2402.18060_ , 2024. 

- Mael Jullien, Marco Valentino, Hannah Frost, Paul O’Regan, Donal Landers, and André Freitas. Semeval-2023 task 7: Multi-evidence natural language inference for clinical trial data. _arXiv preprint arXiv:2305.02993_ , 2023. 

- Yanjun Gao, Ruizhe Li, John Caskey, Dmitriy Dligach, Timothy Miller, Matthew M Churpek, and Majid Afshar. Leveraging a medical knowledge graph into large language models for diagnosis prediction. _arXiv preprint arXiv:2308.14321_ , 2023a. 

- Alistair E. W. Johnson, Lucas Bulgarelli, Lu Shen, Alvin Gayles, Ayad Shammout, Steven Horng, Tom J. Pollard, Sicheng Hao, Benjamin Moody, Brian Gow, Li-wei H. Lehman, Leo A. Celi, and Roger G. Mark. MIMIC-IV, a freely accessible electronic health record dataset. _Scientific data_ , 10 (1):1, 2023. 

- Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, and Tao Gui. The rise and potential of large language model based agents: A survey, 2023. 

- Xiangru Tang, Anni Zou, Zhuosheng Zhang, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Gerstein. Medagents: Large language models as collaborators for zero-shot medical reasoning. _arXiv preprint arXiv:2311.10537_ , 2023. 

- Blackford Middleton, Meryl Bloomrosen, Mark A Dente, Bill Hashmat, Ross Koppel, J Marc Overhage, Thomas H Payne, S Trent Rosenbloom, Charlotte Weaver, and Jiajie Zhang. Enhancing patient safety and quality of care by improving the usability of electronic health record systems: recommendations from amia. _Journal of the American Medical Informatics Association_ , 20(e1): e2–e8, 2013. 

- Jinghui Liu, Daniel Capurro, Anthony Nguyen, and Karin Verspoor. “note bloat” impacts deep learning-based nlp models for clinical prediction tasks. _Journal of biomedical informatics_ , 133: 104149, 2022. 

- Bhavana Dalvi, Peter Jansen, Oyvind Tafjord, Zhengnan Xie, Hannah Smith, Leighanna Pipatanangkura, and Peter Clark. Explaining answers with entailment trees. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 7358–7370, 2021. 

- Marina Danilevsky, Kun Qian, Ranit Aharonov, Yannis Katsis, Ban Kawas, and Prithviraj Sen. A survey of the state of explainable AI for natural language processing. In _Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing_ , pages 447–459, 2020. 

- Sai Gurrapu, Ajay Kulkarni, Lifu Huang, Ismini Lourentzou, and Feras A Batarseh. Rationalization for explainable nlp: A survey. _Frontiers in Artificial Intelligence_ , 6, 2023. 

- Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. e-snli: Natural language inference with natural language explanations. _Advances in Neural Information Processing Systems_ , 31, 2018. 

21 

- Nazneen Fatema Rajani, Bryan McCann, Caiming Xiong, and Richard Socher. Explain yourself! leveraging language models for commonsense reasoning. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 4932–4942, Florence, Italy, 2019. 

- Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, and Byron C. Wallace. ERASER: A benchmark to evaluate rationalized NLP models. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 4443–4458, 2020. 

- Harsh Jhamtani and Peter Clark. Learning to explain: Datasets and models for identifying valid reasoning chains in multihop question-answering. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing_ , page 137–150, 2020. 

- Oyvind Tafjord, Bhavana Dalvi Mishra, and Peter Clark. Proofwriter: Generating implications, proofs, and abductive statements over natural language. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP_ , page 3621–3634, 2021. 

- Chen Zhao, Chenyan Xiong, Jordan Boyd-Graber, and Hal Daumé III. Multi-step reasoning over unstructured text with beam dense retrieval. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 4635–4641, 2021. 

- Yifan Zhang, Jingqin Yang, Yang Yuan, and Andrew Chi-Chih Yao. Cumulative reasoning with large language models. In _ICLR 2024 Workshop on Bridging the Gap Between Practice and Theory in Deep Learning_ , 2024. URL `https://openreview.net/forum?id=XAAYyRxTlQ` . 

- Yanjun Gao, Dmitriy Dligach, Timothy Miller, Samuel Tesch, Ryan Laffin, Matthew M. Churpek, and Majid Afshar. Hierarchical annotation for building a suite of clinical natural language processing tasks: Progress note understanding. In _Proceedings of the Thirteenth Language Resources and Evaluation Conference_ , pages 5484–5493, Marseille, France, 2022. European Language Resources Association. 

- Travis Zack, Gurpreet Dhaliwal, Rabih Geha, Mary Margaretten, Sara Murray, and Julian C Hong. A clinical reasoning-encoded case library developed through natural language processing. _Journal of General Internal Medicine_ , 38(1):5–11, 2023. 

- Yanjun Gao, Dmitriy Dligach, Timothy Miller, John Caskey, Brihat Sharma, Matthew M Churpek, and Majid Afshar. Dr. bench: Diagnostic reasoning benchmark for clinical natural language processing. _Journal of Biomedical Informatics_ , 138:104286, 2023b. 

- L.L. Weed. _Medical Records, Medical Education, and Patient Care: The Problem-oriented Record as a Basic Tool_ . Press of Case Western Reserve University, 1970. ISBN 9780815191889. 

- Olivier Bodenreider. The unified medical language system (umls): integrating biomedical terminology. _Nucleic acids research_ , 32(suppl_1):D267–D270, 2004. 

- AI@Meta. Llama 3 model card. 2024. URL `https://github.com/meta-llama/llama3/blob/ main/MODEL_CARD.md` . 

- Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, et al. Zephyr: Direct distillation of lm alignment. _arXiv preprint arXiv:2310.16944_ , 2023. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- OpenAI. Introducing ChatGPT and Whisper APIs. 2023b. URL `https://openai.com/blog/ introducing-chatgpt-and-whisper-apis` . 

- Robert A Byrne, Xavier Rossello, JJ Coughlan, Emanuele Barbato, Colin Berry, Alaide Chieffo, Marc J Claeys, Gheorghe-Andrei Dan, Marc R Dweck, Mary Galbraith, et al. 2023 esc guidelines for the management of acute coronary syndromes: developed by the task force on the management of acute coronary syndromes of the european society of cardiology (esc). _European Heart Journal: Acute Cardiovascular Care_ , 13(1):55–161, 2024. 

22 

- Hiroaki Kitaoka, Chisato Izumi, Yasuhiro Izumiya, Takayuki Inomata, Mitsuharu Ueda, Toru Kubo, Jun Koyama, Motoaki Sano, Yoshiki Sekijima, Nobuhiro Tahara, et al. Jcs 2020 guideline on diagnosis and treatment of cardiac amyloidosis. _Circulation Journal_ , 84(9):1610–1671, 2020. 

- Writing Committee Members, Eric M Isselbacher, Ourania Preventza, James Hamilton Black III, John G Augoustides, Adam W Beck, Michael A Bolen, Alan C Braverman, Bruce E Bray, Maya M Brown-Zimmerman, et al. 2022 acc/aha guideline for the diagnosis and management of aortic disease: a report of the american heart association/american college of cardiology joint committee on clinical practice guidelines. _Journal of the American College of Cardiology_ , 80(24):e223–e393, 2022. 

- José A Joglar, Mina K Chung, Anastasia L Armbruster, Emelia J Benjamin, Janice Y Chyou, Edmond M Cronin, Anita Deswal, Lee L Eckhardt, Zachary D Goldberger, Rakesh Gopinathannair, et al. 2023 acc/aha/accp/hrs guideline for the diagnosis and management of atrial fibrillation: a report of the american college of cardiology/american heart association joint committee on clinical practice guidelines. _Circulation_ , 149(1):e1–e156, 2024. 

- Steve R Ommen, Seema Mital, Michael A Burke, Sharlene M Day, Anita Deswal, Perry Elliott, Lauren L Evanovich, Judy Hung, José A Joglar, Paul Kantor, et al. 2020 aha/acc guideline for the diagnosis and treatment of patients with hypertrophic cardiomyopathy: executive summary: a report of the american college of cardiology/american heart association joint committee on clinical practice guidelines. _Journal of the American College of Cardiology_ , 76(25):3022–3055, 2020. 

- Paul A Heidenreich, Biykem Bozkurt, David Aguilar, Larry A Allen, Joni J Byun, Monica M Colvin, Anita Deswal, Mark H Drazner, Shannon M Dunlay, Linda R Evers, et al. 2022 aha/acc/hfsa guideline for the management of heart failure: a report of the american college of cardiology/american heart association joint committee on clinical practice guidelines. _Journal of the American College of Cardiology_ , 79(17):e263–e421, 2022. 

- Lilly Su, Rea Mittal, Devyani Ramgobin, Rahul Jain, and Rohit Jain. Current management guidelines on hyperlipidemia: the silent killer. _Journal of lipids_ , 2021(1):9883352, 2021. 

- François Mach, Colin Baigent, Alberico L Catapano, Konstantinos C Koskinas, Manuela Casula, Lina Badimon, M John Chapman, Guy G De Backer, Victoria Delgado, Brian A Ference, et al. 2019 esc/eas guidelines for the management of dyslipidaemias: lipid modification to reduce cardiovascular risk: The task force for the management of dyslipidaemias of the european society of cardiology (esc) and european atherosclerosis society (eas). _European heart journal_ , 41(1): 111–188, 2020. 

- Thomas Unger, Claudio Borghi, Fadi Charchar, Nadia A Khan, Neil R Poulter, Dorairaj Prabhakaran, Agustin Ramirez, Markus Schlaich, George S Stergiou, Maciej Tomaszewski, et al. 2020 international society of hypertension global hypertension practice guidelines. _Hypertension_ , 75(6): 1334–1357, 2020. 

- Shailja C Shah, M Blanca Piazuelo, Ernst J Kuipers, and Dan Li. Aga clinical practice update on the diagnosis and management of atrophic gastritis: expert review. _Gastroenterology_ , 161(4): 1325–1332, 2021. 

- Chinese Society of Gastroenterology, Cancer Collaboration Group of Chinese Society of Gastroenterology, and Chinese Medical Association. Guidelines for diagnosis and treatment of chronic gastritis in china (2022, shanghai). _Journal of Digestive Diseases_ , 24(3):150–180, March 2023. doi: 10.1111/1751-2980.13193. 

- Matthew Banks, David Graham, Marnix Jansen, Takuji Gotoda, Sergio Coda, Massimiliano Di Pietro, Noriya Uedo, Pradeep Bhandari, D Mark Pritchard, Ernst J Kuipers, et al. British society of gastroenterology guidelines on the diagnosis and management of patients at risk of gastric adenocarcinoma. _Gut_ , 68(9):1545–1575, 2019. 

- Chung M Chow, Alexander KC Leung, and Kam L Hon. Acute gastroenteritis: from guidelines to real life. _Clinical and experimental gastroenterology_ , pages 97–112, 2010. 

- C Prakash Gyawali, Rena Yadlapati, Ronnie Fass, David Katzka, John Pandolfino, Edoardo Savarino, Daniel Sifrim, Stuart Spechler, Frank Zerbib, Mark R Fox, et al. Updates to the modern diagnosis of gerd: Lyon consensus 2.0. _Gut_ , 73(2):361–371, 2024. 

23 

Robert T Kavitt, Anna M Lipowska, Adjoa Anyane-Yeboa, and Ian M Gralnek. Diagnosis and treatment of peptic ulcer disease. _The American journal of medicine_ , 132(4):447–456, 2019. 

- Antonio Tarasconi, Federico Coccolini, Walter L Biffl, Matteo Tomasoni, Luca Ansaloni, Edoardo Picetti, Sarah Molfino, Vishal Shelat, Stefania Cimbanassi, Dieter G Weber, et al. Perforated and bleeding peptic ulcer: Wses guidelines. _World journal of emergency surgery_ , 15:1–24, 2020. 

- Alan N Barkun, Majid Almadi, Ernst J Kuipers, Loren Laine, Joseph Sung, Frances Tse, Grigorios I Leontiadis, Neena S Abraham, Xavier Calvet, Francis KL Chan, et al. Management of nonvariceal upper gastrointestinal bleeding: guideline recommendations from the international consensus group. _Annals of internal medicine_ , 171(11):805–822, 2019. 

- Guy McKhann, David Drachman, Marshall Folstein, Robert Katzman, Donald Price, and Emanuel M Stadlan. Clinical diagnosis of alzheimer’s disease: Report of the nincds-adrda work group* under the auspices of department of health and human services task force on alzheimer’s disease. _Neurology_ , 34(7):939–939, 1984. 

Igaku-Shoin-Ltd. Clinical practice guidelines for epilepsy 2018. 2018. 

- Richard B Lipton, Seymour Diamond, Michael Reed, Merle L Diamond, and Walter F Stewart. Migraine diagnosis and treatment: results from the american migraine study ii. _Headache: The Journal of Head and Face Pain_ , 41(7):638–645, 2001. 

- A. K. Eigenbrodt, H. Ashina, S. Khan, H. C. Diener, D. D. Mitsikostas, A. J. Sinclair, P. Pozo-Rosich, P. Martelletti, A. Ducros, M. Lantéri-Minet, M. Braschinsky, M. S. Del Rio, O. Daniel, A. Özge, A. Mammadbayli, M. Arons, K. Skorobogatykh, V. Romanenko, G. M. Terwindt, K. Paemeleire, S. Sacco, U. Reuter, C. Lampl, H. W. Schytz, Z. Katsarava, T. J. Steiner, and M. Ashina. Diagnosis and management of migraine in ten steps. _Nature Reviews Neurology_ , 17(8):501–514, August 2021. doi: 10.1038/s41582-021-00509-5. URL `https://doi.org/10.1038/s41582-021-00509-5` . Epub 2021 Jun 18. 

- Headache Classification Committee of the International Headache Society (IHS). The international classification of headache disorders, 3rd edition. _Cephalalgia_ , 38(1):1–211, Jan 2018. doi: 10.1177/0333102417738202. 

- Fred D Lublin. Clinical features and diagnosis of multiple sclerosis. _Neurologic clinics_ , 23(1):1–15, 2005. 

- WJ Brownlee, TA Hardy, F Fazekas, and DH Miller. Diagnosis of multiple sclerosis: progress and challenges. _Lancet_ , 389(10076):1336–1346, Apr 1 2017. doi: 10.1016/S0140-6736(16)30959-X. URL `https://doi.org/10.1016/S0140-6736(16)30959-X` . 

- Dawn O Kleindorfer, Amytis Towfighi, Seemant Chaturvedi, Kevin M Cockroft, Jose Gutierrez, Debbie Lombardi-Hill, Hooman Kamel, Walter N Kernan, Steven J Kittner, Enrique C Leira, et al. 2021 guideline for the prevention of stroke in patients with stroke and transient ischemic attack: a guideline from the american heart association/american stroke association. _Stroke_ , 52(7): e364–e467, 2021. 

- Amir Qaseem, Timothy J Wilt, Steven E Weinberger, Nicola A Hanania, Gerard Criner, Thys van der Molen, Darcy D Marciniuk, Tom Denberg, Holger Schünemann, Wisia Wedzicha, et al. Diagnosis and management of stable chronic obstructive pulmonary disease: a clinical practice guideline update from the american college of physicians, american college of chest physicians, american thoracic society, and european respiratory society. _Annals of internal medicine_ , 155(3):179–191, 2011. 

- Eric D Bateman, Suzanne S Hurd, Peter J Barnes, Jean Bousquet, Jeffrey M Drazen, Mark FitzGerald, Peter Gibson, Ken Ohta, Paul O’Byrne, Soren Erik Pedersen, et al. Global strategy for asthma management and prevention: Gina executive summary. _European respiratory journal_ , 31(1): 143–178, 2007. 

- Selene Baos, David Calzada, Lucía Cremades-Jimeno, Joaquín Sastre, César Picado, Joaquín Quiralte, Fernando Florido, Carlos Lahoz, and Blanca Cárdaba. Nonallergic asthma and its severity: biomarkers for its discrimination in peripheral samples. _Frontiers in Immunology_ , 9:1416, 2018. 

24 

- Dheeraj Gupta, Ritesh Agarwal, Ashutosh Nath Aggarwal, VN Maturu, Sahajal Dhooria, KT Prasad, Inderpaul S Sehgal, Lakshmikant B Yenge, Aditya Jindal, Navneet Singh, et al. Guidelines for diagnosis and management of chronic obstructive pulmonary disease: Joint ics/nccp (i) recommendations. _Lung India_ , 30(3):228–267, 2013. 

- Gregory Olson and Andrew M Davis. Diagnosis and treatment of adults with community-acquired pneumonia. _Jama_ , 323(9):885–886, 2020. 

- SYNOPSIS OF RECOMMENDATIONS. Guidelines for diagnosis and management of community and hospital acquired pneumonia in adults: Joint ics/nccp (i) recommendations. _Indian J Chest Dis Allied Sci_ , 54:267–281, 2012. 

- Michael S Niederman, Lionel A Mandell, Antonio Anzueto, John B Bass, William A Broughton, G Douglas Campbell, Nathan Dean, Thomas File, Michael J Fine, Peter A Gross, et al. Guidelines for the management of adults with community-acquired pneumonia: diagnosis, assessment of severity, antimicrobial therapy, and prevention. _American journal of respiratory and critical care medicine_ , 163(7):1730–1754, 2001. 

- Stavros V Konstantinides, Guy Meyer, Cecilia Becattini, Héctor Bueno, Geert-Jan Geersing, VeliPekka Harjola, Menno V Huisman, Marc Humbert, Catriona Sian Jennings, David Jiménez, et al. 2019 esc guidelines for the diagnosis and management of acute pulmonary embolism developed in collaboration with the european respiratory society (ers) the task force for the diagnosis and management of acute pulmonary embolism of the european society of cardiology (esc). _European heart journal_ , 41(4):543–603, 2020. 

- David M Lewinsohn, Michael K Leonard, Philip A LoBue, David L Cohn, Charles L Daley, Ed Desmond, Joseph Keane, Deborah A Lewinsohn, Ann M Loeffler, Gerald H Mazurek, et al. Official american thoracic society/infectious diseases society of america/centers for disease control and prevention clinical practice guidelines: diagnosis of tuberculosis in adults and children. _Clinical Infectious Diseases_ , 64(2):e1–e33, 2017. 

- Evangelia Charmandari, Nicolas C Nicolaides, and George P Chrousos. Adrenal insufficiency. _The Lancet_ , 383(9935):2152–2167, 2014. 

- Toshihiko Yanase, Toshihiro Tajima, Takuyuki Katabami, Yasumasa Iwasaki, Yusuke Tanahashi, Akira Sugawara, Tomonobu Hasegawa, Tomoatsu Mune, Yutaka Oki, Yuichi Nakagawa, et al. Diagnosis and treatment of adrenal insufficiency including adrenal crisis: a japan endocrine society clinical practice guideline [opinion]. _Endocrine journal_ , 63(9):765–784, 2016. 

- Stefan R Bornstein, Bruno Allolio, Wiebke Arlt, Andreas Barthel, Andrew Don-Wauchope, Gary D Hammer, Eystein S Husebye, Deborah P Merke, M Hassan Murad, Constantine A Stratakis, et al. Diagnosis and treatment of primary adrenal insufficiency: an endocrine society clinical practice guideline. _The Journal of Clinical Endocrinology & Metabolism_ , 101(2):364–389, 2016. 

- Nuha A ElSayed, Grazia Aleppo, Vanita R Aroda, Raveendhara R Bannuru, Florence M Brown, Dennis Bruemmer, Billy S Collins, Kenneth Cusi, Marisa E Hilliard, Diana Isaacs, et al. 4. comprehensive medical evaluation and assessment of comorbidities: Standards of care in diabetes—2023. _Diabetes Care_ , 46(Suppl 1):s49, 2023. 

- Nicholas A Tritos and Karen K Miller. Diagnosis and management of pituitary adenomas: a review. _Jama_ , 329(16):1386–1398, 2023. 

- Juliana Drummond, Federico Roncaroli, Ashley B Grossman, and Márta Korbonits. Clinical and pathological aspects of silent pituitary adenomas. _The Journal of Clinical Endocrinology & Metabolism_ , 104(7):2473–2489, 2019. 

- Odelia Cooper and Shlomo Melmed. Subclinical hyperfunctioning pituitary adenomas: the silent tumors. _Best practice & research Clinical endocrinology & metabolism_ , 26(4):447–460, 2012. 

- Sarah E Mayson and Peter J Snyder. Silent (clinically nonfunctioning) pituitary adenomas. _Journal of neuro-oncology_ , 117:429–436, 2014. 

25 

- K AlexanderErik, N PearceElizabeth, A BrentGregory, S BrownRosalind, A GrobmanWilliam, H LazarusJohn, J MandelSusan, P PeetersRobin, et al. 2017 guidelines of the american thyroid association for the diagnosis and management of thyroid disease during pregnancy and the postpartum. _Thyroid_ , 2017. 

26 

