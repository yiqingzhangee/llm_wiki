# **Learning to Reason Iteratively and Parallelly for Complex Visual Reasoning Scenarios** 

**Shantanu Jaiswal**[1] _[,]_[2] **Debaditya Roy**[2] **Basura Fernando**[2] _[,]_[3] **Cheston Tan**[2] _[,]_[3] 1 Carnegie Mellon University 2 IHPC, A*STAR Singapore 3 Centre for Frontier AI Research, A*STAR Singapore Correspondence to: `sjaiswa3@cs.cmu.edu` 

## **Abstract** 

Complex visual reasoning and question answering (VQA) is a challenging task that requires compositional multi-step processing and higher-level reasoning capabilities beyond the immediate recognition and localization of objects and events. Here, we introduce a fully neural _Iterative_ and _Parallel Reasoning Mechanism_ (IPRM) that combines two distinct forms of computation – iterative and parallel – to better address complex VQA scenarios. Specifically, IPRM’s _“iterative"_ computation facilitates compositional step-by-step reasoning for scenarios wherein individual operations need to be computed, stored, and recalled dynamically (e.g. when computing the query _“determine the color of pen to the left of the child in red t-shirt sitting at the white table”_ ). Meanwhile, its _“parallel”_ computation allows for the simultaneous exploration of different reasoning paths and benefits more robust and efficient execution of operations that are mutually independent (e.g. when counting individual colors for the query: _“determine the maximum occurring color amongst all t-shirts”_ ). We design IPRM as a lightweight and fullydifferentiable neural module that can be conveniently applied to both transformer and non-transformer vision-language backbones. It notably outperforms prior taskspecific methods and transformer-based attention modules across various image and video VQA benchmarks testing distinct complex reasoning capabilities such as compositional spatiotemporal reasoning (AGQA), situational reasoning (STAR), multi-hop reasoning generalization (CLEVR-Humans) and causal event linking (CLEVRER-Humans). Further, IPRM’s internal computations can be visualized across reasoning steps, aiding interpretability and diagnosis of its errors. Source code to be released at: `https://github.com/shantanuj/IPRM_Iterative_ and_Parallel_Reasoning_Mechanism` 

## **1 Introduction** 

Visual reasoning and question answering (VQA) at its core requires a model to identify relevant visual operations, execute those operations, and then combine their results to make an inference. Complex visual reasoning scenarios (depicted in fig. 1) are particularly challenging in this regard. They require models to reason compositionally over a large number of reasoning steps and to engage in a variety of higher-level reasoning operations such as causal linking, logical reasoning, and spatiotemporal processing that extend beyond core perception capabilities. 

In this context, two powerful computational priors exist – iterative and parallel. While each has its own limitations, when combined, they can complement each other and effectively address the challenges of complex VQA tasks. Specifically, **iterative computation** , wherein individual operations are identified and composed in a step-by-step manner, is a beneficial prior for multi-step reasoning scenarios explored by past VQA works [28, 25, 18]. However, pure iterative computation can exhibit 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0002-00.png)


Figure 1: Complex VQA scenarios (CLEVR-Humans [35], GQA [29], CLEVRER-Humans[51]), AGQA[20] and STAR[79]) wherein combination of iterative (step-by-step) computation (blue phrases) and parallel computation (orange phrases) can be beneficial for reasoning. 

limitations in scenarios wherein the entailed visual operations are independent of one-another, or where distinct stimuli need to be processed and tracked simultaneously. 

For example, consider the first scenario shown in fig. 1. When executing the language phrase _“maximum occurring shape"_ (i.e. _“what shape appears the most"_ ), a purely iterative method would: (i) compute the count of each shape (each of which itself could take multiple iterations), (ii) then update and maintain the counts in memory (without forgetting count of all previous shapes), and (iii) finally, recall each shape’s count to compute the _“maximum"_[1] . Besides taking more reasoning steps than required, such computation also increases the demand for information retention and recall in memory, which in this scenario could scale by the number of shapes to be counted. In complex video reasoning scenarios, a purely iterative method would similarly struggle in tracking and reasoning over multiple co-occurring events. In such scenarios, from both efficiency and efficacy perspectives, it is advantageous to process operations or stimuli in parallel, rather than solely iteratively. 

More generally, **parallel computation** facilitates the simultaneous maintenance and exploration of distinct reasoning paths, and thereby enables reasoning to be more comprehensive, efficient and robust. For example, to compute _“maximum occuring shape”_ , parallel compute can enable distinct shape queries to be simultaneously computed prior to computing the _“maximum"_ operation. Similarly, it is effective for other scenarios illustrated in fig. 1 such as in processing independent logical clauses ( _“{are X} and {Y made of plastic}”_ ) or when tracking and processing co-occurring events in videos. 

Such computation can be implicitly realized in conventional transformer-based parallel attention mechanisms [70]. However, transformer-based attention does not explicitly incorporate iterative compositional computation [14, 41], which as described is beneficial for multi-step reasoning scenarios wherein operations need to be composed sequentially. Accordingly, while parallel computation may effectively compute the result of _“maximum occurring shape"_ in fig. 1, it would potentially struggle to integrate the result with further operations such as _“green object with .."_ , _“small object in front of green .."_ , and _“color of .."_ that need to be computed step-by-step to answer the question. 

Based on the above insights, we design the **Iterative and Parallel Reasoning Mechanism (IPRM),** a novel neural reasoning architecture that combines step-by-step iterative computation with the ability to process multiple independent operations and stimuli simultaneously. Inspired by how humans utilize working memory [17, 9] to facilitate complex reasoning, IPRM internally maintains a latent memory of parallel “operation states", keyed to which are “results states". Given vision and language inputs, IPRM performs the following _iterative_ computation. First, it forms a new set of _parallel_ operations by retrieving relevant language information conditioned on its prior operation states. Then, it “executes" these operations _parallelly_ by retrieving relevant visual information conditioned on its new operations as well as prior result states. Finally, it integrates its new operations (and their results) into memory by dynamically composing these operations with one-another as well as prior operation states, and subsequently, repeats the entire process in its next _iterative_ step. 

This strategy effectively enables us to take advantage of both parallel and iterative computations and notably helps improve state-of-arts across various complex image and video reasoning tasks using a single reasoning mechanism. Equally importantly, IPRM’s internal computations can be visualized 

> 1Assuming _“max"_ is applied after counts of all shapes are computed. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Lang. input  ( ) Vis. input  ( ) repeat for<br>... ...<br>Operation<br>Composition<br>Unit<br>Memory state  ( ) Operation Formation Operation Execution Next Mem. state  ( )<br>... .. .. .. ...<br>**----- End of picture text -----**<br>


Figure 2: IPRM’s computation flow diagram. First, a new set of N-parallel latent operations **Zop** are retrieved from language features **XL** conditioned on prior operation states **Mop** . Then, visual features **XV** are queried conditioned on both **Zop** and prior result states results **Mres** , to form the new results **Zres** . Finally, both **Zres** and **Zop** are passed to the Operation Composition Unit (see 2.3), the output of which becomes the new memory state **M** . 

across reasoning steps, which helps better interpret what operations it was doing and accordingly where it was looking visually when processing a complex reasoning scenario. 

## **2 Iterative and Parallel Reasoning Mechanism** 

Our proposed iterative-and parallel-reasoning mechanism (IPRM) is a fully-differentiable neural architecture. Given visual features **XV** _∈_ R _[N][V][ ×][D][V]_ and language or task-description features **XL** _∈_ R _[N][L][×][D][L]_ , IPRM outputs a _“reasoning result"_ **ys** _∈_ R _[D][m]_ and, optionally, a set of _“reasoning result tokens"_ **YR** _∈_ R _[N][m][×][D][m]_ . As previously mentioned, IPRM operates iteratively for _T_ reasoning steps and internally, maintains an explicit memory **M** : _{_ **Mop** _,_ **Mres** _}_ . The memory is modelled as a set of _“operation states"_ **Mop** _∈_ R _[N][op][×][D][m]_ , keyed to which are _“result states"_ **Mres** _∈_ R _[N][op][×][D][m]_ as shown in fig. 2. Here, _Nop_ denotes the number of parallel operations to be computed while _Dm_ denotes the mechanism’s internal feature dimension. On a high level, at each reasoning step (denoted by _t ∈{_ 1 _, · · · , T }_ ), IPRM performs the following computations: 

1. First, conditioned on the existing operation states **Mop** _,_ **t** , we retrieve relevant information from language or task-description features **XL** to form a new set of latent operations **Zop** _,_ **t** _∈_ R _[N][op][×][D][m]_ . We term this computation as _“Operation Formation"_ . 

   - **Zop** _,_ **t** = **Op** _ **Form** ( **XL** ; **Mop** _,_ **t** ) (1) 

2. Then, conditioned on the latent operations **Zop** _,_ **t** and the existing result state **Mres** _,_ **t** , we attend and retrieve relevant information from visual features **XV** which represents a new set of latent results **Zres** _,_ **t** _∈_ R _[N][op][×][D][m]_ corresponding to **Zop** _,_ **t** . We term this computation as 

_“Operation Execution"_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0003-08.png)


**----- Start of picture text -----**<br>
Zres , t =  Op _ Exec ( XV ; [ Zop , t ,  Mres , t ]) (2)<br>**----- End of picture text -----**<br>


3. Finally, to facilitate interaction amongst parallel operations, we perform inter-operation attention. Here, each operation **Zop** _ **k** _,_ **t** ; **k** _∈{_ 1 _, · · · , Nop}_ , is composed with other operations in **Zop** _,_ **t** as well as prior operation states **Mop** [ **t** _−_ **W** : **t** ] within a lookback-window _W_ . The corresponding result **Zres** _ **k** _,_ **t** is similarly composed with other results **Zres** _,_ **t** and prior result states denoted as **Mres** [( **t** _−_ **W** ): **t** ]. We term this computation as _“Operation Composition"_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0003-10.png)


As shown in eq. (3), this output is the new memory state **Mt** + **1** : _{_ **Mop** _,_ **Mres** _}_ . 

The overall computation flow is illustrated in fig. 2, and we provide specific details and intuitions behind these computations in the following sub-sections. 

3 

## **2.1 Operation Formation** 

The _“operation formation"_ stage conceptually models a reasoner that based on its prior set of operations, decides what language features to retrieve in order to form the next set of relevant operations. This can be effectively implemented through conventional attention mechanisms. Specifically, the cumulative set of prior operations (maintained in **Mop** _,_ **t** ) can be projected to form the ‘query’ **QL** _,_ **t** _∈_ R _[N][op][×][D][m]_ representing “what features to look for". The language features **XL** can be projected to form the ‘key’ **KL** _∈_ R _[N][L][×][D][m]_ and ‘value’ **VL** _∈_ R _[N][L][×][D][m]_ . Finally, the new set of latent operations **Zop** _,_ **t** can be retrieved by computing `attn` ( **QL** _,_ **KL** _,_ **VL** ). These steps are formally represented below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0004-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0004-03.png)


Note **KL** and **VL** are not computation-step dependent and only computed once. We use a simple linear-modulated formulation (with appropriate broadcasting and projection weight **Wa** _∈_ R _[D][k][×]_[1] ) to implement `attn(.)` (further details in appendix sec. 13). 

## **2.2 Operation Execution** 

In the _“operation execution"_ stage, the reasoner determines what visual features need to be retrieved depending on both the newly formed operations and existing result states. To model the constituent visual attention mechanism, we draw insights from existing recurrent visual reasoning methods [28, 69] that incorporate feature modulation for memory-guided attention. Specifically, we retrieve a set of feature modulation weights **SV** _,_ **t** _∈_ R _[N][op][×][D][m][/r]_ through a joint projection of the new operations **Zop** _,_ **t** and prior results **Mres** _,_ **t** as shown in eq. (6). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0004-07.png)


Here, _r_ is a feature reduction ratio [23, 31]. **SV** _,_ **t** is then applied dimension wise to a projection of **XV** to retrieve an intermediate attention key **K** _[′]_ **V** _,_ **t** _[∈]_[R] _[N][op][×][N][k][×][D][m][/r]_[.][The final attention key] **KV** _,_ **t** is then obtained through a joint multi-layer-projection of **K** _[′]_ **V** _,_ **t**[and the previously projected] representation of **XV** as shown in eq. (7). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0004-09.png)


Finally, the attention query and value are formed through separate projections of **Zop** _,_ **t** and **XV** respectively. These are then fed together with **KV** _,_ **t** to the attention function to retrieve the new operation results **Zres** _,_ **t** as shown in eq. (8). Intuitively, the overall process allows for both prior results and the new set of operations to jointly guide visual attention. 

**QV** _,_ **t** _,_ **VV** _,_ **t** = **WV** _,_ **q** ( **Zop** _,_ **t** ) _,_ **WV** _,_ **v** ( **XV** ) _,_ **Zres** _,_ **t** = `attn` ( **QV** _,_ **t** _,_ **KV** _,_ **t** _,_ **VV** _,_ **t** ) (8) Here, **WV** _,_ **op** _∈_ R _[D][m][/r][×][D][m]_ , **WV** _,_ **res** _∈_ R _[D][m][/r][×][D][m]_ , **WV** _,_ **s** _∈_ R _[D][m][/r][×]_[2] _[D][m][/r]_ , **WV** _,_ **k1** _∈_ R _[D][m][/r][×][D][v]_ , **WV** _,_ **k2** _∈_ R _[D][m][/r][×]_[2] _[D][m][/r]_ , **WV** _,_ **k3** _∈_ R _[D][m][/r][×][D][m][/r]_ , **WV** _,_ **q** _∈_ R _[D][m][/r][×][D][m]_ and **WV** _,_ **v** _∈_ R _[D][m][×][D][m]_ . 

## **2.3 Operation Composition** 

Finally, in the _“operation composition"_ stage, the reasoner first integrates the executed operations **Zop** _,_ **t** and their results **Zres** _,_ **t** into the existing memory state **Mt** through a simple recurrent update as shown in eqs. (9) and (10). Then, to mitigate redundancy amongst parallel operations and to retrieve relevant knowledge from prior-step operations, it dynamically composes individual operation states **M** _[′]_ **op** _,_ **t** + **1**[with other operation states in] **[ M]** _[′]_ **op** _,_ **t** + **1**[and also prior operation states in] **[ M][op]** _[,]_ **[t]** _[−]_ **[W]**[:] **[t]**[.][Here,] _W_ is an attention look-back window. 

This composition is achieved through computing inter-operation attention as illustrated in fig. 3. Specifically, **M** _[′]_ **op** _,_ **t** + **1**[is projected to obtain a set of queries] **[ Q][op]** _[,]_ **[t]**[, while the token-wise concatenation] of **M** _[′]_ **op** _,_ **t** + **1**[and] **[ M][op]** _[,]_ **[t]** _[−]_ **[W]**[:] **[t]**[are projected to obtain the operation attention keys] **[ K][op]** _[,]_ **[t]**[and values] **Vop** _,_ **t** . A second set of values **Vres** _,_ **t** are also formed through projection of respective result states 

4 

as shown in eq. (14). Further, an identity attention mask **INop** is used to ensure that operations in **Qop** _,_ **t** , can only attend to other operations and not themselves. This is done to enable a higher degree of operation composition. As shown in eq. (15), **Qop** _,_ **t** , **Kop** _,_ **t** , **Vop** _,_ **t** and **INop** are passed to the attention operation, which outputs an intermediate representation **M** _[′′]_ **op** _,_ **t** + **1**[and the softmaxed-] attention weights **Aop** _,_ **t** . **M** _[′′]_ **op** _,_ **t** + **1**[is then added to a projection of] **[ M]** _[′]_ **op** _,_ **t** + **1**[to effectively combine] attended operation states with the original operation states, and thereby form the next mem. operation state **Mop** _,_ **t** + **1** . 

Finally, the next result states are obtained by applying **Aop** _,_ **t** on **Vres** _,_ **t** and then adding a projection of **M** _[′]_ **res** _,_ **t** + **1**[as shown in eq. (17).][Note] **[ A][op]** _[,]_ **[t]**[is specifically utilized to ensure that results are composed] based on attentions between operation states. Here, all the mentioned weights **W** _.. ∈_ R _[D][m][×][D][m]_ and [ _·_ ; _·_ ] represents token-wise concatenation. **INop** in eq. (15) is an identity matrix which is concatenated with zeros (i.e. unmasked) for window tokens if window len. W _>_ 0. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0005-02.png)


**Obtaining Reasoning Summary** As mentioned before, our proposed mechanism outputs a set of _“reasoning result tokens"_ **YR** and a _“reasoning result"_ **ys** . **YR** is simply equivalent to the last memory result states **Mres** _,_ **T** + **1** . To obtain **ys** , we perform attention on the last operation states **Mop** _,_ **T** + **1** by utilizing a summary representation **ls** _∈_ R _[D][l]_ of **XL** as the attention-query. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0005-04.png)


**----- Start of picture text -----**<br>
the first token in case of transformer-based language ... ...<br>as last hidden state in case of LSTM-based language<br>As shown in eq. (18),  lss is projected to obtain a single-token<br> pqq while  Mopop , T + 1 is projected to obtain the attention +<br>The attention value is simply the result states  Mresres , T + 1 , and<br> “reasoning result" . Intuitively, (self masked)<br>copy<br>op , T + 1 are most relevant to the summary of the input<br> XLL , based on which corresponding result X +<br>1 are weighted and retrieved.<br>pq ,  Pk =  Wpq , q ( ls ) ,  Wpk , k ( Mop , T + 1 ) (18) .. ...<br>ys =  attn ( pq ,  Pk ,  Mres , T + 1 ) (19)<br>.. ..<br>..<br>**----- End of picture text -----**<br>


We set **ls** to be the first token in case of transformer-based language backbones and as last hidden state in case of LSTM-based language backbones. As shown in eq. (18), **lss** is projected to obtain a single-token attention query **pqq** while **Mopop** _,_ **T** + **1** is projected to obtain the attention keys **Pk** . The attention value is simply the result states **Mresres** _,_ **T** + **1** , and the output of the attention function is the _“reasoning result"_ . Intuitively, this computation corresponds to the reasoner deciding which final operation states in **Mop** _,_ **T** + **1** are most relevant to the summary of the input language or task-description **XLL** , based on which corresponding result states **Mres** _,_ **T** + **1** are weighted and retrieved. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0005-06.png)


Figure 3: Operation Composition Unit 

**Reasoning mechanism general applicability.** Our proposed iterative and Composition Unit parallel reasoning mechanism is an end-to-end trainable neural module. It can be conveniently applied on top of different vision and language backbones, and be trained directly as a new computational block (similar to a transformer block receiving vision and language tokens) with no specific adjustments. Further, the base version of IPRM (used in this paper) is weight-tied across iterative steps which means that its number of parameters is constant regardless of number of computation steps and parallel operations. We provide parameter and computational details along with module implementation details in appendix sec. C. 

5 

Table 1: Comparison of IPRM with videoQA methods on STAR (left) and AGQAv2 (right). All methods operate on 32 frames unless otherwise mentioned in (). *Not directly compared as utilizes additional surrogate tasks / benchmarks and num. of frames not reported. 

|Model|Int.<br>Seq.<br>Pred.<br>Feas.<br>Avg.|||||
|---|---|---|---|---|---|
|LRR*[5]<br>LRR (w/o surrogate)<br>All-in-One [72]<br>Temp[ATP][7]<br>MIST [18]<br>InternVideo (8) [75]<br>SeViLA-BLIP2 [86]<br>Concat-Att-4L<br>Cross-Att-4L|73.7<br>71.0<br>71.3<br>65.1<br>70.3<br>54.5<br>48.7<br>44.3<br>45.5<br>48.2<br>47.5<br>50.8<br>47.7<br>44.0<br>47.5<br>50.6<br>52.8<br>49.3<br>40.6<br>48.3<br>55.5<br>54.2<br>54.2<br>44.4<br>51.1<br>62.7<br>65.6<br>54.9<br>51.9<br>58.7<br>63.7<br>70.4<br>63.1<br>**62.4**<br>64.9<br>68.1<br>71.4<br>66.6<br>55.2<br>65.3<br>67.5<br>72.1<br>64.4<br>58.5<br>65.6|||||
||||Metric|HCRN[39]<br>AIO[72]<br>Temp[7]<br>MIST[18]<br>GF[4]|IPRM|
||||obj-rel<br>superlative<br>sequencing<br>exist<br>duration<br>act. recog.|40.3<br>48.3<br>50.2<br>51.7<br>55.0<br>33.6<br>37.5<br>39.8<br>42.1<br>44.6<br>49.7<br>49.6<br>48.3<br>67.2<br>53.2<br>50.0<br>50.8<br>51.8<br>60.3<br>59.1<br>43.8<br>45.4<br>49.6<br>**54.6**<br>52.8<br>5.5<br>19.0<br>19.0<br>19.7<br>14.2|**57.8**<br>**48.0**<br>**75.6**<br>**62.4**<br>50.7<br>**20.0**|
||||open<br>binary<br>all|36.3<br>-<br>-<br>50.6<br>56.1<br>48.0<br>-<br>-<br>58.3<br>54.2<br>42.1<br>48.6<br>49.8<br>54.4<br>55.1|**58.6**<br>**62.3**<br>**60.4**|
|||||||
|IPRM|**71.8**<br>**77.7**<br>**71.0**<br>59.1<br>**69.9**|||||



## **3 Experiments** 

We evaluate IPRM on STAR[79], AGQAv2[20] and CLEVRER-Humans[51] for video reasoning tasks and CLEVR-Humans[35], GQA [29] and CLEVR-CoGenT [34] for image reasoning tasks. For all tasks, we set IPRM’s parallel operations ( _Nop_ ) to 6, reasoning steps ( _T_ ) to 9, reduction ratio ( _r_ ) to 2 and window length ( _W_ ) to 2 (informed by ablative analysis detailed in sec. 3.3). We follow task-specific practices (detailed in appendix C.1) for respective vision and language backbones in our primary experiments, besides also demonstrating integration of IPRM with large-scale VL backbones such as CLIP. Further, besides task-specific methods, we also consider two prominent transformer-based VL modules as baselines – concat-att (where lang. and vis. tokens are concatenated as in [36, 37]) and cross-att (where lang. tokens are “query" to vis. tokens as “key" and “value"; as in [43, 1]). Further implementation and training details are provided in appendix sec.C. 

## **3.1 Video Reasoning and Question Answering (STAR, AGQAv2 and CLEVRER-Humans)** 

We first evaluate IPRM on recent video reasoning benchmarks. STAR [79] and AGQAv2 comprise real-world videos and test multiple reasoning skills in context of situational reasoning and compositional spatiotemporal reasoning respectively. STAR contains 60K questions testing four broad types of video reasoning abilities: _feasibility_ , _interaction_ , _prediction_ and _sequence_ . Meanwhile, AGQAv2 contains 2.27M balanced questions distributed across 16 different question types. As shown in table 1, IPRM obtains 69.9% average acc. on STAR and 60.4% overall acc. on AGQAv2, outperforming prior videoQA-specific methods by 5% on both benchmarks. 

Interestingly, on STAR IPRM obtains an 8% and 7% improvement over SeViLA-BLIP2 on the predictive and sequencing scenarios respectively. This is possibly due to IPRM’s capability to reason over multiple events simultaneously across frames, which may enhance its capacity to retrieve and cumulatively reason on relevant information needed to predict future events and determine appropriate sequences. However, IPRM performs less effectively than SeViLA-BLIP2 in feasibility scenarios, possibly because these scenarios require not only visual reasoning but also commonsense knowledge, which can benefit from integration with larger-scale vision-language backbones and auxiliary training tasks such as introduced in LRR [5]. 

Similarly, On AGQAv2, IPRM improves performances across various question types, notably achieving a 8% improvement in questions that require determining sequence of events. Further, IPRM also outperforms both 4-layer concat- and cross-attention modules on STAR (scaling further attentionlayers was not found to benefit performance as detailed in appendix table 7). 

Next, we evaluate IPRM on the CLEVRER-Humans benchmark [51], which comprises synthetic videos of simultaneous object motions and multiple collisions, and tests a model’s ability to determine causal links between events. We perform zero-shot, finetuned and from-scratch evaluation. As shown in table 2, IPRM outpeforms task-specific neurosymbolic models NS-DR and VR-DP as well as state-of-the-art ALOE across the three settings. Specifically, IPRM improves zero-shot per-question acc. by 7%, finetuned per-question acc. by 18.8% and scratch per-question acc. by 6.2%. These results further suggest that IPRM can better track and process co-occuring events, and in this case, more accurately determine causal links. 

6 

Table 2: Comparison of methods for CLEVRERHumans [51] (Opt. is per option acc. and Qs. is per question acc.). IPRM achieves state-of-art across settings. 

Table 3: Comparison of methods on CLEVRHumans (zero-shot and finetuned setting), CLEVRCoGenT (ValA is in-domain; ValB is out-ofdomain setting) and CLOSURE. IPRM performs strongly without requiring extra supervision. 

|Model<br>|Zero-shot<br>Opt.<br>Qs.<br><br>|Finetune<br>Opt.<br>Qs.|Finetune<br>Opt.<br>Qs.|Scratch<br>O<br>|
|---|---|---|---|---|
||||Qs.|pt.<br>Qs.|
|NS-DR[84]<br>VRDP[13]|51.0<br>32.0<br>50.9<br>31.6|-<br>-|-<br>-|-<br>-<br>-<br>-|
|CNNLSTM[51]<br>CNNBERT[51]<br>ALOE[12]|50.3<br>30.0<br>52.9<br>32.0<br>54.0<br>26.9|51.7<br>52.0<br>51.8|34.2<br>30.2<br>31.7|51.5<br>30.8<br>50.1<br>30.4<br>52.7<br>32.1|
|IPRM|**61.7**<br>**38.9**|**74.1**|**53.0**|**62.0**<br>**38.3**|



|Model|Extra|supv.|supv.|CLV-Hum|CLV-Hum|CLV-Hum|CLV-Hum|CLV-CoGen|CLV-CoGen|CLV-CoGen|CLV-CoGen|CLV-CoGen|CLOSURE|CLOSURE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||ZS<br>FT||||ValA<br>ValB|||||ZS Avg.||
|PG+EE [35]<br>NS-VQA [85]<br>FiLM [60]<br>MAC [28]<br>MDETR[36]|Prog<br>Prog<br>N<br>N<br>Boun|rams<br>rams<br>one<br>one<br>d. Box||54.0<br>66.6<br>-<br>67.8<br>56.6<br>75.9<br>57.4<br>81.5<br>59.9<br>81.7||||96.6<br>73.7<br>**99.8**<br>63.9<br>98.3<br>78.8<br>99.0<br>78.3<br>**99.8**<br>76.7|||||75.6<br>**77.2**<br>56.9<br>73.8<br>-||
|IPRM|N|one||**63.8**<br>**85.5**||||99.1<br>**80.3**|||||75.6||
|60<br>65<br>70<br>75<br>80<br>85<br>Validation Accuracy (%)|||||||||||||||
|||||.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>1.0<br>Training Data Ratio<br><br>~~80.5~~<br>82.8<br>~~84.3~~<br>85.1<br>85.5<br>81.7<br>IPRM(4.4Mparams)<br>Concat-Att (12.6M)<br>Cross-Att (16.8M)<br>MDETR (17.4M)<br>MAC (5.8M)|||||||||||
|||||||||||~~43~~||85.1|||
|||||||8||2.8||~~.~~|||||
||||75.3|||~~80.5~~|||||||||
||||||||||||||||
||||||||||IPR|M(4|.4M|para||ms)|
||||||||||||||||
||||||||||Con<br>Cro|cat-<br>s-A|Att (<br>t (1|12.6<br>68||M)<br>)|
||||||||||||||||
||||||||||MDE<br>MAC|<br>TR<br>(5.|<br>(17.<br>8M)|.<br>4M)|||
|||0.0<br>0|||||||||||||




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0007-04.png)


Figure 5: IPRM performance on CLEVR-Humans at different training data ratios of Cross- and Concat-Att. 

Figure 4: Acc. of IPRM (blue) across program lengths for CLEVR (left) and STAR (right). IPRM has signicantly higher accs. at longer program lengths. 

## **3.2 Compositional Image Reasoning (CLEVR-Humans, CLEVR-CoGen and GQA)** 

Here, we evaluate IPRM on challenging compositional image reasoning benchmarks. CLEVRHumans tests generalization of multi-hop reasoning to free-form human crowdsourced questions which entail reasoning skills/scenarios beyond a model’s training on the original CLEVR dataset and provides a limited finetuning set (2.5% of CLEVR). Similarly, CLEVR-CoGenT tests generalization on novel attribute compositions not observed in training (e.g.“ _gray cubes_ " and “ _red cylinders_ " are in training but eval. is on “ _gray cylinders_ " and “ _red cubes_ "; see suppl. for exact specification). The CLOSURE [3] benchmark further tests systematic generalization for different question type compositions. 

As shown in table 3, IPRM achieves 3.9% and 3.8% improvements in zero-shot and fully-finetuned performance over prior state-of-art vision-language model MDETR. Further, IPRM neither requires bounding-box pre-training supervision (as done in MDETR) nor functional programs, and can be trained directly with only vision-language inputs and task supervision. fig. 5 illustrates IPRM’s performance across different training ratios compared to MDETR and cross- and concat-att transformer modules. Notably, IPRM exceeds MDETR’s fully-finetuned performance by 1.1% with only half of training data. Further, IPRM exhibits these improvements while being relatively lightweight (4.4M params) in comparison to MDETR’s transformer blocks (17.4M; _∼_ 4x more parameters) and crossand concat-VL attention methods (16.8M and 12.6M respectively). These results suggest that IPRM exhibits strong generalization capabilities and more sample-efficient learning of novel reasoning skills and scenarios in context of multi-step imageQA. 

For CLEVR-CoGenT, IPRM achieves state-of-art results in out-of-domain generalization on novel attribute compositions and outperforms MDETR (having parallel transformer compute) by 3.6% and MAC (iterative method) by 2%. This suggests the combination of iterative and parallel computation as done in IPRM can implicitly enable more disentangled feature processing and thereby improve compositional learning of primitive attributes in context of multi-step imageQA. Similarly, on CLOSURE, IPRM achieves an average zero-shot accuracy of 75.6% which is highest amongst fully neural reasoning methods (that require no extra supervision) and is close to the neurosymbolic method NS-VQA which utilizes ground truth programs and object bounding boxes supervision. Further, detailed breakdown of models on CLOSURE question types is provided in appendix table 6. 

7 

Table 4: Performance comparison on GQA with imageQA methods and large-scale models that do not utilize ground-truth scene graphs. * indicates large-scale pretrained VL model. **Utilizes ground truth scene graphs, programs and bounding boxes for auxiliary training. 

||LCGN[25]|MCAN[87]|LXMERT*[66]|12-in-1*[49]|OSCAR*[46]|CFR** [55]|IPRM|
|---|---|---|---|---|---|---|---|
|GQA|55.8|57.4|60.0|60.0|**61.6**|72.1|60.3|



We also evaluate IPRM on the GQA benchmark which tests compositional VQA on real-world images. We perform comparisons with prior VQA methods as well as large-scale VL models such as LXMERT and 12-in-1 that do not utilize the ground truth scene graphs in GQA. As shown in table 4, IPRM achieves 60.3% , outperforming prior reasoning methods such as MCAN and LCGN as well as large-scale VL models such as LXMERT and 12-in-1. However, IPRM’s performance is behind the larger VL model OSCAR which performs pretraining on 4.3M data samples collated from COCO, VG, SBU and Flickr. Note, IPRM is a standalone reasoning module only trained on GQA balanced with no pre-training. Further, IPRM when trained with perfect perception (i.e. ground truth object bounding boxes and attributes), achieves 87.2% on GQA validation set suggesting strong reasoning performances can be achieved through advancements in visual detectors and backbones. Finally, in appendix B.3, we demonstrate that IPRM more effectively enhances the performance of frozen CLIP variants on complex reasoning benchmarks compared to scaling traditional cross- and concat-attention modules. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0008-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0008-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0008-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0008-06.png)


Figure 6: IPRM Model ablations in order: **(i)** Impact of number of parallel operations ( _Nop_ ) vs computation steps ( _T_ ). **(ii)** Impact of Operation Composition Block (OPC). **(iii)** : Impact of reduction ratio ( _r_ ) and **(iv)** memory window length ( _W_ ). 

## **3.3 Model ablations and reasoning visualization** 

We perform ablations to study the contributions of IPRM’s salient components. First, we analyze the impact of varying number of parallel operations ( _Nop_ ) against number of iterative computation steps ( _T_ ). We compare models with _Nop ∈{_ 1 _,_ 3 _,_ 6 _,_ 9 _}_ and _T ∈{_ 1 _,_ 3 _,_ 6 _,_ 9 _}_ , resulting in 16 different models. Given the large amount of ablative models, we perform analysis on a reduced-resolution setting of CLEVR-Humans with pretraining on CLEVR for 15 epochs. As shown in fig. 6 (plot (i)), we find that _T_ and _Nop_ appear to be co-dependent and that neither by itself can lead to high performance. E.g. setting _T_ = 1 generally results in peformances around 75% regardless of _Nop_ , while setting _Nop_ = 1 or 3 results in a sharp performance drop of 3% when changing _T_ to 9 from 6. In contrast, for _Nop >_ 3, we observe that performance increases steadily with higher _T_ , suggesting that a higher number of parallel operations may prevent overfitting in a model with high computation steps. Overall, we find that ( _Nop_ = 6, _T_ = 9) and ( _Nop_ = 9, _T_ = 9) are the two best performing models achieving above 82% accuracy (with the former preferred as _Nop_ = 6 performs better for diff. _T_ compared to _Nop_ = 9). 

Next, we study the impact of the operation composition block (OPC) by evaluating _Nop_ = 6 (and diff. computation steps _T_ ) with and without OPC. As shown in fig. 6 (plot (ii)), while _T_ = 1 has a relatively low drop of _∼_ 2%, the performance drops are more significant for higher _T_ . The ( _Nop_ = 6, _T_ = 9) model which reached _∼_ 82% acc. with OPC, drops to _∼_ 74% acc. without OPC. 

Finally, we study the impacts of the dimension reduction ratio ( _r_ ) and memory-lookback window length ( _W_ ). As shown in fig. 6 (plot (iii)), _r_ = 2 leads to negligible drop (0.3%) in performance compared to no dimension reduction (i.e. _r_ = 1) while being computationally faster and requiring less memory. However, setting _r_ to 8 significantly deteriorates performance. Similarly, as shown in fig. 6 (plot (iv)), a setting of window-length _W_ is 2 works sufficiently well, and decreasing it below that significantly impacts performance. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0009-00.png)


**----- Start of picture text -----**<br>
What shape is behind How many objects are<br>the large cylinder with  exact same shape as the<br>max occuring color? one on the farthest right?<br>(GT: cylinder;        (GT: two; Pred: two)<br>Pred: cylinder )<br>Reasoning Computation Steps Reasoning Computation Steps<br>Do either of 2 pairs of items 0<br>Are there any red  that match each other in<br>kites or flags? terms of material and color<br>(GT: yes; Pred: yes) also match each other in<br>size?      (GT: Yes; Pred: No)<br>Reasoning Computation Steps<br>Parallel Operations Parallel Operations<br>Parallel Operations<br>**----- End of picture text -----**<br>


Figure 7: **Condensed reasoning visualization of IPRM** . In the top two examples, IPRM correctly utilizes both parallel and iterative computation to arrive at the correct answer. The bottom left example shows IPRM’s cumulative lang. and visual attentions when solving a real-world GQA example. The bottom right example, shows an error case where IPRM seems to misunderstand question and outputs wrong ans. with less relevant attentions. See appendix for further reasoning visualizations and error cases, and supplemental for further CLIP integrated visualizations on GQA examples. 

In fig. 4, we assess the performance of IPRM across different functional program lengths (a proxy for reasoning steps) on the CLEVR and STAR benchmarks. As shown, IPRM maintains a high performance across both short and longer reasoning steps and for both image and video VQA scenarios. We also provide a condensed illustration of IPRM’s reasoning visualization in fig. 7. Here, for brevity, we only show the model’s visual attention for a subset of parallel operations and computation steps (see suppl. for full visualization). The top left example (“what shape .. max occuring color") illustrates the model’s usage of both iterative and parallel computation. At the first reasoning step, the model appears to be doing parallel operations to both identify “large cylinder" and compute “max occuring color". In the next step, it appears to have found “max occuring color" (cyan) and seems to check which of the two large cylinders match the color. Next, it highlights the correct cylinder and in the final step, it locates and predicts the correct object behind. The top right example in fig. 7 similarly shows computation for a question involving spatial (“farthest right"), similarity (“same shape") and counting (“how many") operations. The bottom left example illustrates IPRM’s simultaneous language and visual attentions updated across reasoning steps when solving a real world (GQA) example. Finally, the bottom right example is an error-case where IPRM seems to incorrectly understand or execute requisite operations, thereby not producing expected attentions for “items with matching material and color" throughout steps. 

## **4 Related Work** 

**Visual reasoning methods and vision-language models.** Multiple prior works have introduced effective visual reasoning methods in context of image and video question answering [35, 52, 2, 85, 15, 50, 69, 56, 32, 54, 53, 76]. Prominent works include NMN[2], FILM [60], NSM [27], MAC [28], MCAN [87], NS-VQA [85], ALOE [12], VR-DP [13], SHG-VQA [68], MIST [18] and OCRA [76]. In contrast to these works that show applicability of methods for particular VQA benchmarks/tasks, our work explores a more general direction of integrating parallel and iterative computation in a reasoning framework that we show to be effective for multiple complex VQA benchmarks and reasoning scenarios. More recently, vision-language models [44, 73, 37, 66, 45, 46, 74] and multimodal large-language-models [57, 47] with transformer-based mechanisms have shown impressive reasoning capabilities at scale. Notable examples include CLIP [61], GPT [57], Gemini [67], MDETR [36], LXMERT [66], VinVL [88], BLIP [44, 43], Flamingo [1], LlavA [47], BEiT [74] and All-in-One [72]. We believe our work is complimentary to these developments, as it 

9 

contributes an alternative and possibly more effective reasoning mechanism that can be integrated with such models in future to enhance complex VQA capabilities. 

**Memory and recurrence-augmented transformers.** Multiple works have identified limitations of purely feedforward computation as realized in transformers and worked on encoding recurrence [30, 26, 11, 71] and memory-augmented computation [80, 8]. Notably, Recurrent Memory Transformer [8] and MemFormer[81] introduce recurrent and dynamic memory to improve language modelling capabilities. More recently, EMAT [82] introduces efficient memory to augment knowledge retrieval, MemViT [80] introduces a cache memory to retain prior context for long-video tasks, and [71] introduces memory for more effective action anticipation. While these methods study recurrent and memory-augmented computation on specific natural language processing and computer vision tasks, our work focuses on the integration of iterative-parallel computation and working memory in a single neural reasoning mechanism beneficial for complex VQA scenarios. 

## **5 Conclusion** 

We introduced a novel fully-differentiable and end-to-end trainable iterative and parallel reasoning mechanism (IPRM) to address complex VQA scenarios. We comprehensively evaluated IPRM on various complex image and video VQA benchmarks testing distinct reasoning capabilities, and found it improves state-of-arts on multiple such benchmarks. We also performed quantiative ablations to study individual impacts of parallel and iterative computation besides qualitative analysis of IPRM’s reasoning computation visualization. 

## **6 Limitations and Future work** 

Here, we note possible limitations of IPRM. Similar to existing VQA and deep-learning methods, IPRM may reflect biases that are present in the training distribution of VQA benchmarks. This may lead it to overfit to certain image inputs or question forms and possibly provide skewed answers in such scenarios. Further, the utilized vision-language backbones in our experiments may also entail visual, language and cultural biases in their original training distribution which may permeate to IPRM upon integration for VQA scenarios. In this regard, we hope the capability to visualize intermediate reasoning of IPRM and diagnose its error cases (as shown in section 3.3) can serve a useful tool to benefit interpretability in VQA and identify possible reasoning biases that may emerge in the model. 

For future work, scaling the IPRM architecture to a foundational video-language model by integrating it with large-scale transformer-based vision-language models and relevant instruction-tuning approaches presents an exciting research opportunity. Moreover, while we designed and evaluated IPRM in the context of complex VQA, we believe it has the potential to operate as a general reasoning mechanism applicable to tasks beyond visual reasoning and question answering, such as language processing and embodied reasoning. 

**Acknowledgment** This research/project is supported by the National Research Foundation, Singapore, under its NRF Fellowship (Award# NRF-NRFF14-2022-0001). This research is also supported by funding allocation to B.F. and C.T. by the Agency for Science, Technology and Research (A*STAR) under its SERC Central Research Fund (CRF), as well as its Centre for Frontier AI Research (CFAR). 

## **References** 

- [1] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in Neural Information Processing Systems_ , 35:23716–23736, 2022. 

- [2] J. Andreas, M. Rohrbach, T. Darrell, and D. Klein. Learning to compose neural networks for question answering. _arXiv preprint arXiv:1601.01705_ , 2016. 

- [3] D. Bahdanau, H. de Vries, T. J. O’Donnell, S. Murty, P. Beaudoin, Y. Bengio, and A. Courville. Closure: Assessing systematic generalization of clevr models. _arXiv preprint arXiv:1912.05783_ , 2019. 

- [4] Z. Bai, R. Wang, and X. Chen. Glance and focus: Memory prompting for multi-event video question answering. _Advances in Neural Information Processing Systems_ , 36, 2024. 

10 

- [5] A. Bhattacharyya, S. Panchal, R. Pourreza, M. Lee, P. Madan, and R. Memisevic. Look, remember and reason: Grounded reasoning in videos with language models. In _The Twelfth International Conference on Learning Representations_ . 

- [6] B. Bogin, S. Subramanian, M. Gardner, and J. Berant. Latent compositional representations improve systematic generalization in grounded question answering. _Transactions of the Association for Computational Linguistics_ , 9:195–210, 2021. 

- [7] S. Buch, C. Eyzaguirre, A. Gaidon, J. Wu, L. Fei-Fei, and J. C. Niebles. Revisiting the" video" in video-language understanding. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 2917–2927, 2022. 

- [8] A. Bulatov, Y. Kuratov, and M. Burtsev. Recurrent memory transformer. _Advances in Neural Information Processing Systems_ , 35:11079–11091, 2022. 

- [9] A. Capon, S. Handley, and I. Dennis. Working memory and reasoning: An individual differences perspective. _Thinking & Reasoning_ , 9(3):203–244, 2003. 

- [10] Y. Cong, W. Liao, H. Ackermann, B. Rosenhahn, and M. Y. Yang. Spatial-temporal transformer for dynamic scene graph generation. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 16372–16382, 2021. 

- [11] M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and Ł. Kaiser. Universal transformers. _arXiv preprint arXiv:1807.03819_ , 2018. 

- [12] D. Ding, F. Hill, A. Santoro, M. Reynolds, and M. Botvinick. Attention over learned object embeddings enables complex visual reasoning. _Advances in neural information processing systems_ , 34:9112–9124, 2021. 

- [13] M. Ding, Z. Chen, T. Du, P. Luo, J. Tenenbaum, and C. Gan. Dynamic visual reasoning by learning differentiable physics models from video and language. _Advances in Neural Information Processing Systems_ , 34:887–899, 2021. 

- [14] N. Dziri, X. Lu, M. Sclar, X. L. Li, L. Jiang, B. Y. Lin, S. Welleck, P. West, C. Bhagavatula, R. Le Bras, et al. Faith and fate: Limits of transformers on compositionality. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [15] C. Fan, X. Zhang, S. Zhang, W. Wang, C. Zhang, and H. Huang. Heterogeneous memory enhanced multimodal attention model for video question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 1999–2007, 2019. 

- [16] H.-S. Fang, S. Xie, Y.-W. Tai, and C. Lu. Rmpe: Regional multi-person pose estimation. In _Proceedings of the IEEE international conference on computer vision_ , pages 2334–2343, 2017. 

- [17] D. Fougnie. The relationship between attention and working memory. _New research on short-term memory_ , 1:45, 2008. 

- [18] D. Gao, L. Zhou, L. Ji, L. Zhu, Y. Yang, and M. Z. Shou. Mist: Multi-modal iterative spatialtemporal transformer for long-form video question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14773–14783, 2023. 

- [19] R. Girshick. Fast r-cnn. In _Proceedings of the IEEE international conference on computer vision_ , pages 1440–1448, 2015. 

- [20] M. Grunde-McLaughlin, R. Krishna, and M. Agrawala. Agqa: A benchmark for compositional spatio-temporal reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11287–11297, 2021. 

- [21] M. Grunde-McLaughlin, R. Krishna, and M. Agrawala. Agqa 2.0: An updated benchmark for compositional spatio-temporal reasoning. _arXiv preprint arXiv:2204.06105_ , 2022. 

- [22] J. Hsu, J. Mao, J. Tenenbaum, and J. Wu. What’s left? concept grounding with logic-enhanced foundation models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [23] J. Hu, L. Shen, and G. Sun. Squeeze-and-excitation networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 7132–7141, 2018. 

- [24] R. Hu, J. Andreas, M. Rohrbach, T. Darrell, and K. Saenko. Learning to reason: End-to-end module networks for visual question answering. In _Proceedings of the IEEE international conference on computer vision_ , pages 804–813, 2017. 

11 

- [25] R. Hu, A. Rohrbach, T. Darrell, and K. Saenko. Language-conditioned graph networks for relational reasoning. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 10294–10303, 2019. 

- [26] F. Huang, K. Lu, C. Yuxi, Z. Qin, Y. Fang, G. Tian, and G. Li. Encoding recurrence into transformers. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- [27] D. Hudson and C. D. Manning. Learning by abstraction: The neural state machine. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- [28] D. A. Hudson and C. D. Manning. Compositional attention networks for machine reasoning. _arXiv preprint arXiv:1803.03067_ , 2018. 

- [29] D. A. Hudson and C. D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 6700–6709, 2019. 

- [30] D. Hutchins, I. Schlag, Y. Wu, E. Dyer, and B. Neyshabur. Block-recurrent transformers. _Advances in Neural Information Processing Systems_ , 35:33248–33261, 2022. 

- [31] S. Jaiswal, B. Fernando, and C. Tan. Tdam: Top-down attention module for contextually guided feature selection in cnns. In _European Conference on Computer Vision_ , pages 259–276. Springer, 2022. 

- [32] A. Jha, B. Patro, L. Van Gool, and T. Tuytelaars. Barlow constrained optimization for visual question answering. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pages 1084–1093, 2023. 

- [33] C. Jing, Y. Jia, Y. Wu, X. Liu, and Q. Wu. Maintaining reasoning consistency in compositional visual question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5099–5108, 2022. 

- [34] J. Johnson, B. Hariharan, L. Van Der Maaten, L. Fei-Fei, C. Lawrence Zitnick, and R. Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2901– 2910, 2017. 

- [35] J. Johnson, B. Hariharan, L. Van Der Maaten, J. Hoffman, L. Fei-Fei, C. Lawrence Zitnick, and R. Girshick. Inferring and executing programs for visual reasoning. In _Proceedings of the IEEE international conference on computer vision_ , pages 2989–2998, 2017. 

- [36] A. Kamath, M. Singh, Y. LeCun, G. Synnaeve, I. Misra, and N. Carion. Mdetr-modulated detection for end-to-end multi-modal understanding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1780–1790, 2021. 

- [37] W. Kim, B. Son, and I. Kim. Vilt: Vision-and-language transformer without convolution or region supervision. In _International Conference on Machine Learning_ , pages 5583–5594. PMLR, 2021. 

- [38] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- [39] T. M. Le, V. Le, S. Venkatesh, and T. Tran. Hierarchical conditional relation networks for video question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 9972–9981, 2020. 

- [40] J. Lei, L. Li, L. Zhou, Z. Gan, T. L. Berg, M. Bansal, and J. Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 7331–7341, 2021. 

- [41] M. Lewis, N. V. Nayak, P. Yu, Q. Yu, J. Merullo, S. H. Bach, and E. Pavlick. Does clip bind concepts? probing compositionality in large image models. _arXiv preprint arXiv:2212.10537_ , 2022. 

- [42] C. Li, H. Xu, J. Tian, W. Wang, M. Yan, B. Bi, J. Ye, H. Chen, G. Xu, Z. Cao, et al. mplug: Effective and efficient vision-language learning by cross-modal skip-connections. _arXiv preprint arXiv:2205.12005_ , 2022. 

- [43] J. Li, D. Li, S. Savarese, and S. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 

12 

- [44] J. Li, D. Li, C. Xiong, and S. Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _International Conference on Machine Learning_ , pages 12888–12900. PMLR, 2022. 

- [45] L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, and K.-W. Chang. Visualbert: A simple and performant baseline for vision and language. _arXiv preprint arXiv:1908.03557_ , 2019. 

- [46] X. Li, X. Yin, C. Li, P. Zhang, X. Hu, L. Zhang, L. Wang, H. Hu, L. Dong, F. Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX 16_ , pages 121–137. Springer, 2020. 

- [47] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_ , 2023. 

- [48] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ , 2019. 

- [49] J. Lu, V. Goswami, M. Rohrbach, D. Parikh, and S. Lee. 12-in-1: Multi-task vision and language representation learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 10437–10446, 2020. 

- [50] J. Mao, C. Gan, P. Kohli, J. B. Tenenbaum, and J. Wu. The neuro-symbolic concept learner: Interpreting scenes, words, and sentences from natural supervision. _arXiv preprint arXiv:1904.12584_ , 2019. 

- [51] J. Mao, X. Yang, X. Zhang, N. Goodman, and J. Wu. Clevrer-humans: Describing physical and causal events the human way. _Advances in Neural Information Processing Systems_ , 35:7755– 7768, 2022. 

- [52] D. Mascharka, P. Tran, R. Soklaski, and A. Majumdar. Transparency by design: Closing the gap between performance and interpretability in visual reasoning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4942–4950, 2018. 

- [53] S. S. Mondal, J. D. Cohen, and T. W. Webb. Slot abstractors: Toward scalable abstract visual reasoning. _arXiv preprint arXiv:2403.03458_ , 2024. 

- [54] S. S. Mondal, T. Webb, and J. D. Cohen. Learning to reason over visual objects. _arXiv preprint arXiv:2303.02260_ , 2023. 

- [55] B. X. Nguyen, T. Do, H. Tran, E. Tjiputra, Q. D. Tran, and A. Nguyen. Coarse-to-fine reasoning for visual question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 4558–4566, 2022. 

- [56] D. K. Nguyen, V. Goswami, and X. Chen. Movie: Revisiting modulated convolutions for visual counting and beyond. In _International Conference on Learning Representations_ , 2020. 

- [57] R. OpenAI. Gpt-4 technical report. _ArXiv_ , 2303, 2023. 

- [58] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. _Advances in neural information processing systems_ , 32, 2019. 

- [59] J. Pennington, R. Socher, and C. D. Manning. Glove: Global vectors for word representation. In _Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)_ , pages 1532–1543, 2014. 

- [60] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville. Film: Visual reasoning with a general conditioning layer. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 32, 2018. 

- [61] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 

- [62] R. Shrestha, K. Kafle, and C. Kanan. Answer them all! toward universal visual question answering models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 10472–10481, 2019. 

13 

- [63] A. Suhr, M. Lewis, J. Yeh, and Y. Artzi. A corpus of natural language for visual reasoning. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 217–223, 2017. 

- [64] A. Suhr, S. Zhou, A. Zhang, I. Zhang, H. Bai, and Y. Artzi. A corpus for reasoning about natural language grounded in photographs. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 6418–6428, 2019. 

- [65] H. Tan and M. Bansal. Object ordering with bidirectional matchings for visual reasoning. In _Proceedings of NAACL-HLT_ , pages 444–451, 2018. 

- [66] H. Tan and M. Bansal. Lxmert: Learning cross-modality encoder representations from transformers. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 5100–5111, 2019. 

- [67] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- [68] A. Urooj, H. Kuehne, B. Wu, K. Chheu, W. Bousselham, C. Gan, N. Lobo, and M. Shah. Learning situation hyper-graphs for video question answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14879–14889, 2023. 

- [69] M. Vaishnav and T. Serre. Gamr: A guided attention model for (visual) reasoning. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- [70] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- [71] J. Wang, G. Chen, Y. Huang, L. Wang, and T. Lu. Memory-and-anticipation transformer for online action understanding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 13824–13835, 2023. 

- [72] J. Wang, Y. Ge, R. Yan, Y. Ge, K. Q. Lin, S. Tsutsui, X. Lin, G. Cai, J. Wu, Y. Shan, et al. All in one: Exploring unified video-language pre-training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6598–6608, 2023. 

- [73] P. Wang, A. Yang, R. Men, J. Lin, S. Bai, Z. Li, J. Ma, C. Zhou, J. Zhou, and H. Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In _International Conference on Machine Learning_ , pages 23318–23340. PMLR, 2022. 

- [74] W. Wang, H. Bao, L. Dong, J. Bjorck, Z. Peng, Q. Liu, K. Aggarwal, O. K. Mohammed, S. Singhal, S. Som, et al. Image as a foreign language: Beit pretraining for vision and visionlanguage tasks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 19175–19186, 2023. 

- [75] Y. Wang, K. Li, Y. Li, Y. He, B. Huang, Z. Zhao, H. Zhang, J. Xu, Y. Liu, Z. Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. _arXiv preprint arXiv:2212.03191_ , 2022. 

- [76] T. Webb, S. S. Mondal, and J. D. Cohen. Systematic visual reasoning through object-centric relational abstraction. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [77] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, et al. Huggingface’s transformers: State-of-the-art natural language processing. _arXiv preprint arXiv:1910.03771_ , 2019. 

- [78] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, et al. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations_ , pages 38–45, 2020. 

- [79] B. Wu, S. Yu, Z. Chen, J. B. Tenenbaum, and C. Gan. Star: A benchmark for situated reasoning in real-world videos. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)_ , 2021. 

14 

- [80] C.-Y. Wu, Y. Li, K. Mangalam, H. Fan, B. Xiong, J. Malik, and C. Feichtenhofer. Memvit: Memory-augmented multiscale vision transformer for efficient long-term video recognition. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 13587–13597, 2022. 

- [81] Q. Wu, Z. Lan, K. Qian, J. Gu, A. Geramifard, and Z. Yu. Memformer: A memory-augmented transformer for sequence modeling. In Y. He, H. Ji, S. Li, Y. Liu, and C.-H. Chang, editors, _Findings of the Association for Computational Linguistics: AACL-IJCNLP 2022_ , pages 308–318, Online only, Nov. 2022. Association for Computational Linguistics. 

- [82] Y. Wu, Y. Zhao, B. Hu, P. Minervini, P. Stenetorp, and S. Riedel. An efficient memoryaugmented transformer for knowledge-intensive nlp tasks. _arXiv preprint arXiv:2210.16773_ , 2022. 

- [83] Z. Yang, X. He, J. Gao, L. Deng, and A. Smola. Stacked attention networks for image question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 21–29, 2016. 

- [84] K. Yi, C. Gan, Y. Li, P. Kohli, J. Wu, A. Torralba, and J. B. Tenenbaum. Clevrer: Collision events for video representation and reasoning. _arXiv preprint arXiv:1910.01442_ , 2019. 

- [85] K. Yi, J. Wu, C. Gan, A. Torralba, P. Kohli, and J. Tenenbaum. Neural-symbolic vqa: Disentangling reasoning from vision and language understanding. _Advances in neural information processing systems_ , 31, 2018. 

- [86] S. Yu, J. Cho, P. Yadav, and M. Bansal. Self-chained image-language model for video localization and question answering. _arXiv preprint arXiv:2305.06988_ , 2023. 

- [87] Z. Yu, J. Yu, Y. Cui, D. Tao, and Q. Tian. Deep modular co-attention networks for visual question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 6281–6290, 2019. 

- [88] P. Zhang, X. Li, X. Hu, J. Yang, L. Zhang, L. Wang, Y. Choi, and J. Gao. Vinvl: Revisiting visual representations in vision-language models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 5579–5588, 2021. 

## **A Appendix / supplemental material** 

## **B Elaborated Experiments and Results Discussion** 

## **B.1 Further comparisons on CLEVR-Humans, CLEVR-CoGenT, CLOSURE and NLVRv1** 

Here, we provide further comparisons with benchmark-specific methods for CLEVR-Humans [35], CLEVR-CoGenT [34], CLOSURE [3] and NLVRv1 [63] (not reported in main paper due to space limitations). As mentioned in main paper, these benchmarks utilize synthetic images and are a test of pure visual reasoning capabilities that are minimally influenced by increased world knowledge or usage of stronger visual backbones. 

CLEVR-Humans as already mentioned in main paper evaluates a model’s reasoning generalization capabilities to unseen scenarios or question forms. CLEVR-CoGenT studies compositional attribute generalization. Specifically, it has two conditions – i) cond.A wherein all cubes have color _∈{gray, blue, brown, yellow}_ and cylinders _∈{red, green, purple, cyan}_ (spheres can be any color), and ii) cond.B wherein color-sets are switched b/w cubes and cylinders. A model is then trained on one condition and evaluated on both the original and alternate condition. A higher accuracy on the alternate condition indicates that the model learns more ‘compositionally’ as it generalizes better to novel shape-color combinations with less feature/attribute combination overfitting. 

Finally, NLVRv1 evaluates language-grounded visual reasoning. Each sample of this benchmark comprises a set of three synthetic images and a composite natural language statement about the images which can evaluate to True or False and requires various visual-linguistic reasoning skills. 

As shown in table 5, IPRM achieves state-of-art results across the three benchmarks and does not require pre-annotated bounding-boxes or functional programs as additional supervision. For **CLEVR-Humans** (table 5 left), it outperforms larger-scale models such as MDETR and RAMEN 

15 

Table 5: Elaborated results on CLEVR-Humans (left), CLEVR-CoGenT (middle) and NLVRv1 (right). IPRM achieves state-of-art across the three benchmarks and does not require additional supervision such as bounding boxes or functional programs. * requires func. programs supervision / pre-defined dataset-specific neural modules.[▼] requires object bounding-boxes supervision. 

|Model|CLV-Hum<br>ZS<br>FT||Model<br>|CoGenTr-A<br>ValA<br>ValB|CoGenFT-B<br>ValA<br>ValB||Model|NLVR1<br>Test-U|
|---|---|---|---|---|---|---|---|---|
|PG+EE* [35]<br>NS-VQA▼* [85]<br>RAMEN [62]<br>FiLM [60]<br>GLT [6]<br>LEFT [22]<br>MAC [28]<br>MDETR▼[36]|54.0<br>66.6<br>-<br>67.8<br>57.8<br>-<br>56.6<br>75.9<br>-<br>75.8<br>-<br>78.8<br>57.4<br>81.5<br>59.9<br>81.7||NS-VQA▼* [85]<br>MDETR▼[36]<br>StackAtt-MLP[83]<br>PG + EE* [34]<br>Tbd-Net* [52]<br>MAC [28]<br>FILM[60]|**99.8**<br>63.9<br>**99.8**<br>76.7<br>80.3<br>68.7<br>96.6<br>73.7<br>98.8<br>75.4<br>99.0<br>78.3<br>98.3<br>78.8|-<br>-<br>-<br>-<br>75.7<br>75.8<br>76.1<br>92.7<br>96.9<br>96.3<br>97.2<br>96.1<br>81.1<br>96.9||CNN-RNN [63]<br>MAC [28]<br>FILM [60]<br>NMN* [2]<br>N2NMN* [24]<br>CNN-BiATT[65]|56.3<br>59.4<br>61.2<br>62.0<br>66.0<br>66.1|
||||||||IPRM (scratch)<br>IPRM-CLV-FT|63.8<br>**73.0**|
|IPRM|**63.8**<br>**85.5**||IPRM|99.1<br>**80.3**|**98.0**<br>**98.2**||||



in zero-shot performance even though the latter is pre-trained on multiple VQA datasets. It also increases state-of-art in finetuned setting by 3.8%. 

For **CLEVR-CogenT** (table 5 centre) , IPRM achieves the highest generalization results amongst methods in both the CoGen-Train A and Finetune B. Specifically, it obtains 80.3% acc. on cond. B (when trained on cond. A), which is 1.5% higher than the previous state-of-art cond.B method FILM and 3.6% higher than MDETR. When further finetuned on cond.B, IPRM generalizes for both cond.A and cond.B achieving 98.0% and 98.2% unlike FILM which overfits to cond.B and thereby has poor performance on cond.A. Further, its performance on cond.A (99.1%) is highest amongst methods that do not utilize bounding box or localization supervision and marginally lower than MDETR and NS-VQA (which utilize bounding-box supervision). 

**For NLVRv1** (table 5 right), IPRM model trained from scratch achieves 63.8% acc. and performs competitively with existing task-specific state-of-art model CNN-BiAtt. When finetuned from its CLEVR checkpoint, we find IPRM achieves 73.0% acc. which is 7% higher than existing visual inputs state-of-art for NLVRv1 and suggests strong reasoning transfer capabilities of IPRM. It further outperforms the N2NMN method which requires pre-defined neural modules to be identified for the dataset. 

Finally, we have also provided results of models on CLOSURE for different subsets / question types in table 6. For the CLEVR-environment, we have visualized some failure cases in figs. 10 and 11. More cases can be run through the visualization framework in provided source code. 

Table 6: Zero-shot performances of models on CLOSURE subsets (2 trials run for IPRM) 

|**Model(ZeroShot)**|and_mat_spa|embed_mat_spa|embed_spa_mat|or_mat|or_mat_spa|compare_mat|compare_mat_spa|
|---|---|---|---|---|---|---|---|
|FiLM|41.4±18|62±11|93.4±2|34.4±11|35.2±5.5|66.2±8.5|65.8±5|
|MAC|63.7±25|76.8±11|99±0.15|75.2±13|70.4±15|65.3±8.6|66.2±6.4|
|NS-VQA*|33.3±18|97.3±4.6|98.3±5|69.5±28|50.4±27|96.1±6.2|95.4±6.7|
|PG-Tensor-NMN*|37.6±17|79.8±1.4|95.6±5.6|34.1±8.1|25.5±11|89.2±2.3|89.8±2.7|
|PG-Vector-NMN*|32.5±17|97.3±2.4|97.2±3.9|64.9±25|47.4±23|95.3±5.7|94.4±6.4|
|IPRM|83.1±6.4|80.3±7.2|99.2±0.1|75.7±9.4|67.1±8.1|62.2±7.3|61.5±5.8|



## **B.2 Elaborated STAR results and ablations for video reasoning tasks** 

We provide results on the STAR Test, further baselines and model ablations for video reasoning tasks in table 7. 

## **B.3 CLIP Integration Results** 

We provide results with frozen CLIP [61] visual backbones including CLIP VIT-L/14, CLIP VITB/16 and CLIP VIT-L/14@336px on GQA [29], NLVRv2 [64] and CLEVR-Humans in table 8. We empirically found utilizing a Distil-roberta language backbone to be more effective than the associated CLIP language backbone, and hence used the former for question processing. We compare with alternate prominent vision-language attention mechanisms including Cross-att and Concat-att blocks as well as a simple joint projection of vision and language pooled representations (referred as Wt-Proj-Att). As shown in the table, IPRM can enhance performance for the CLIP variants across GQA, NLVRv2 and CLV-Humans in comparison to concat and cross-att blocks. Further, it is more 

16 

Table 7: **Left:** Results on STAR [79] official hidden test set (evaluation server) with ground-truth vision (GT V) and predicted vision (PR V); **Right:** Results on STAR val. set with num. of sampled frames =32 unless otherwise stated in (). 

|||STAR-Test<br>Int.<br>Seq.<br>Pred.<br>Feas.<br>Avg.<br>34.7<br>35.9<br>31.2<br>31.4<br>34.7<br>36.3<br>38.9<br>30.7<br>29.8<br>36.5<br>42.6<br>46.3<br>43.4<br>43.9<br>44.5<br>**79.1**<br>**85.8**<br>**82.0**<br>**71.4**<br>**79.6**<br>33.6<br>37.2<br>31.0<br>30.8<br>34.8<br>39.8<br>43.6<br>32.2<br>31.4<br>36.7<br>30.9<br>31.8<br>30.2<br>29.7<br>30.7<br>48.0<br>42.0<br>35.3<br>32.5<br>39.5<br>56.1<br>61.3<br>52.7<br>45.7<br>53.9<br>60.4<br>65.6<br>57.5<br>49.6<br>58.3<br>**72.0**<br>**78.1**<br>**71.5**<br>**59.4**<br>**70.3**|STAR-Test<br>Int.<br>Seq.<br>Pred.<br>Feas.<br>Avg.<br>34.7<br>35.9<br>31.2<br>31.4<br>34.7<br>36.3<br>38.9<br>30.7<br>29.8<br>36.5<br>42.6<br>46.3<br>43.4<br>43.9<br>44.5<br>**79.1**<br>**85.8**<br>**82.0**<br>**71.4**<br>**79.6**<br>33.6<br>37.2<br>31.0<br>30.8<br>34.8<br>39.8<br>43.6<br>32.2<br>31.4<br>36.7<br>30.9<br>31.8<br>30.2<br>29.7<br>30.7<br>48.0<br>42.0<br>35.3<br>32.5<br>39.5<br>56.1<br>61.3<br>52.7<br>45.7<br>53.9<br>60.4<br>65.6<br>57.5<br>49.6<br>58.3<br>**72.0**<br>**78.1**<br>**71.5**<br>**59.4**<br>**70.3**|Model|Int.<br>Seq.<br>Pred.<br>Feas.<br>Avg.|
|---|---|---|---|---|---|
|||||All-in-One [72]<br>Temp[ATP][7]<br>MIST [18]<br>InternVideo (8) [75]<br>SeViLA-BLIP2 [86]<br>Concat-Att-2L<br>Concat-Att-4L<br>Cross-Att-4L<br>Concat-Att-6L<br>Cross-Att-6L|47.5<br>50.8<br>47.7<br>44.0<br>47.5<br>50.6<br>52.8<br>49.3<br>40.6<br>48.3<br>55.5<br>54.2<br>54.2<br>44.4<br>51.1<br>62.7<br>65.6<br>54.9<br>51.9<br>58.7<br>63.7<br>70.4<br>63.1<br>**62.4**<br>64.9<br>66.0<br>68.9<br>66.4<br>55.1<br>64.1<br>68.1<br>71.4<br>66.6<br>55.2<br>65.3<br>67.5<br>72.1<br>64.4<br>58.5<br>65.6<br>66.3<br>71.4<br>66.6<br>55.7<br>65.0<br>59.8<br>63.0<br>56.4<br>49.9<br>57.3|
|Model|Setup|STAR-Test||||
|||Int.<br>Seq.<br>Pred.<br>Feas.|Avg.|||
|Vis-BERT[45]<br>CLIP-BERT[40]<br>NS-SR[79]<br>IPRM|GT V<br>GT V<br>GT V<br>GT V|34.7<br>35.9<br>31.2<br>31.4<br>36.3<br>38.9<br>30.7<br>29.8<br>42.6<br>46.3<br>43.4<br>43.9<br>**79.1**<br>**85.8**<br>**82.0**<br>**71.4**|34.7<br>36.5<br>44.5<br>**79.6**|||
|Vis-BERT[45]<br>CLIP-BERT[40]<br>NS-SR[79]<br>SHG-VQA [68]<br>GF [4]<br>mPLUG[42]|-<br>-<br>PR V<br>-<br>-<br>-|33.6<br>37.2<br>31.0<br>30.8<br>39.8<br>43.6<br>32.2<br>31.4<br>30.9<br>31.8<br>30.2<br>29.7<br>48.0<br>42.0<br>35.3<br>32.5<br>56.1<br>61.3<br>52.7<br>45.7<br>60.4<br>65.6<br>57.5<br>49.6|34.8<br>36.7<br>30.7<br>39.5<br>53.9<br>58.3|||
|||||IPRM(m1,t1)<br>IPRM(m1, t9)<br>IPRM(m6, t1)<br>IPRM<br>IPRM(GT V)|65.6<br>69.8<br>65.1<br>54.5<br>63.8<br>70.0<br>75.7<br>70.2<br>57.8<br>68.4<br>69.5<br>75.2<br>68.6<br>57.4<br>67.7<br>**71.8**<br>**77.7**<br>**71.0**<br>59.1<br>**69.9**<br>78.7<br>85.5<br>81.7<br>71.2<br>79.3|
|IPRM|PR V|**72.0**<br>**78.1**<br>**71.5**<br>**59.4**|**70.3**|||
|||||||



Table 8: **Left:** Comparison of IPRM with prominent vision-language attention mechanisms with CLIP VIT-L/14 backbones on CLEVR-Humans, GQA and NLVRv2 benchmarks (‘4L’ indicates 4 att layers; ‘x’ indicates model did not converge). **Right:** Results with other CLIP variants VIT-B and VIT-L@ 336 on GQA and NLVRv2. FLOPs computed over inputs of 256 visual tokens and 60 lang. tokens. 

|Model (CLIP<br>+Param<br>+GFLOPs<br>GQA<br>NLVR2<br>CLV-H<br>VIT-L/14 bbone)<br>TestD<br>Test<br>ZS<br>FT<br>Wt-Proj-Fusion<br>0.6M<br>0.1<br>53.5<br>60.8<br>58.5<br>74.4<br>Cross-Att (2L)<br>9.2M<br>1.5<br>55.1<br>62.1<br>-<br>-<br>Concat-Att (2L)<br>7.2M<br>4.4<br>55.3<br>60.5<br>-<br>-<br>Cross-Att (4L)<br>17.6M<br>3.1<br>57.4<br>54.4<br>60.3<br>80.0<br>Concat-Att (4L)<br>13.6M<br>8.9<br>58.7<br>55.9<br>61.2<br>81.1<br>Cross-Att (6L)<br>26.0M<br>4.5<br>56.8<br>x<br>60.8<br>80.4<br>Concat-Att(6L)<br>19.7M<br>13.3<br>57.4<br>x<br>62.0<br>81.8<br>IPRM<br>5.2M<br>5.9<br>**59.2**<br>**65.1**<br>**64.3**<br>**84.6**|Model (CLIP<br>VIT-B/16 bbone)|GQA<br>TestD|NLVR2<br>Test|
|---|---|---|---|
||Wt-Proj-Fusion<br>Cross-Att<br>Concat-Att|51.4<br>54.6<br>56.0|59.9<br>56.6<br>57.4|
||IPRM|55.9|60.8|
||Model (CLIP<br>VIT-L/14@336)|GQA<br>TestD|NLVR2<br>Test|
||Wt-Proj-Fusion<br>Cross-Att<br>Concat-Att|54.0<br>57.4<br>57.3|61.1<br>58.4<br>59.1|
||IPRM|59.0|65.3|



parameter efficient with only 5.5M additional parameters in comparison to 4-layer as well as 2-layer stacks of Cross-Att (9.2M 2-layer, 17.6M 4-layer) and Concat-Att (7.2M 2-layer, 13.6M 4-layer). With regards to computational FLOPs, IPRM consumes 5.9GFLOPs which is marginally higher than Cross-Att 4-layer config (3.1GFLOPs) and lower than Concat-Att 4-layer config (8.9GFLOPs). Note, that the performance benefits of adding further layers of cross- or concat-att blocks are observed to be minimal after 4 layers, and can also depend on the amount of training data available. E.g. Both cross- and concat-att blocks of 2 layers had better performances on NLVRv2 (which has a limited set of training questions relative to GQA and CLEVR) in comparison to 4 layer config. 

## **B.4 Further reasoning computation visualizations** 

We provide elaborate reasoning computation visualizations of IPRM showing the lang. and vis. attentions across parallel operations and computation steps during _operation formation_ and _operation execution_ stages. Fig. 8 shows a scenario wherein IPRM correctly utilizes parallel and iterative computations to compute intermediate operations of “find object close to front", “retrieve/compare shape and size", “find applicable objects with both same shape and size". Fig. 9 shows another correct prediction of IPRM, and this time, its intermediate reasoning visualization is useful to determine that the entailed reasoning appears sensible. Fig. 10 shows an incorrect prediction by IPRM and its intermediate reasoning visualizations also suggest that IPRM did not understand the question and thereby did not attend to relevant objects. Finally, Fig. 11 shows a scenario wherein while IPRM produces the correct answer, it’s intermediate reasoning appears imprecise which makes the prediction (and underlying reasoning) less reliable. We provide further visualizations with a CLIP VIT-L/14 backbone on GQA samples in the supplemental jupyter notebook output (html format for easier viewing). 

17 

## **C Model implementation and experiment details** 

We implement IPRM in PyTorch [58] as a generic vision-language module receiving a set of input vision (or scene-representation) tokens and input language (or task-representation) tokens. We provide **Python-style pseudocode of IPRM in figs 12, 13 and 14** . For all experiments, we set the internal dimension of IPRM to 512 and use the same configuration of num. parallel operations ( _Nop_ )=6, num. computation steps (T)=9, reduction ratio (r)=2 and window size (W)=2. We follow benchmark-specific conventions for vision-language backbones that are detailed below in sec. C.1. For CLIP [61], we utilize the official models from Huggingface [77]. All experiments are performed on a single NVIDIA A40 GPU with 46GB memory and averaged over 3 trials with different random seeds wherever possible (done for primary experiments on STAR, AGQA, CLEVRER-Humans, CLEVR-Humans). Unless otherwise specified, the learning rate is initialized to 1e-4 with Adam [38] optimizer and gradient clipping value of 8. The learning-rate is reduced based on validation acc. plateau with reduction factor 0.5, threshold 0.001 and patience 0. Further experiment hyperparameters and settings are provided below. Source code at: `https://github.com/shantanuj/ IPRM_Iterative_and_Parallel_Reasoning_Mechanism` . 

## **C.1 Benchmark-specific experiment details** 

**CLEVR-Humans** . We use the CLEVR-Humans dataset from [35] which comprises images from original CLEVR dataset [34] and human crowdsourced questions. We use a batch size of 216 for training. We use the same language encoder (Distil-Roberta[48] from Huggingface[78]) as in existing state-of-art MDETR [36] and frozen ResNet101 backbone layer 3 spatial features (as in [28, 52, 35]). We perform all ablation experiments with 14x14x1024 visual features. Each ablation model is first pretrained for 10 epochs on the original CLEVR dataset (the initial learning rate for IPRM is 1e-4 and for language encoder is 1e-5) and then finetuned on CLEVR-Humans for 40 epochs with early stopping (learning rate of 1e-4 throughout). As observed in prior work [52], we similarly found in multiple scenarios with occluded objects that visual attention only partially identified such objects. Hence, we simply resampled (bilinear sampling) visual input to obtain 16x16x1024 features and empirically found more complete visual attentions with a corresponding 1.1% improvement in accuracy. The final two best performing model configurations ( _Nop=6, T=9, W=2, R=2_ and _Nop=6, T=9, W=2, R=1_ ) from ablations were then pre-trained for 35 epochs on CLEVR and finetuned on CLEVR-Humans. While we found that configuration _Nop=6, T=9, W=2, R=1_ obtains highest zero-shot (ZS) acc. of 65.6% and finetuned (FT) acc. of 86.3%, we adopt _Nop=6, T=9, W=2, R=2_ (with 63.3% ZS and 85.4% FT acc.) as our optimal model given its lesser parameters and FLOPs. 

**GQA** . We use the GQA compositional real-world image question answering dataset from [29]. Based on prior VQA methods on GQA [29, 25, 45, 33], we utilize pre-extracted bounding-box object proposal features and object label predictions obtained from a pretrained object detector [19, 88]. The bounding box coordinates is normalized to range of 0 to 1 based on the original input image size, and the 4 coordinates are transformed to a distributed representation through a learned nonlinear projection. This representation is concatenated with a learned projection of the predicted object labels (initialized with glove[59] 300dim embeddings) to form the final visual input. We train IPRM for 25 epochs with a batch-size of 192 and same hyperparameters as before. We evaluate the final model on both the test-dev split and the official test evaluation server ( `https: //eval.ai/featured-challenges/225/evaluation` ). 

**STAR-VideoQA.** We use the STAR-VideoQA dataset for situational reasoning on real-world videos from [79]. Based on previous videoQA methods [79, 40, 45] for STAR, we utilize object bounding boxes, labels, human pose and human-object relations across frames (note: we do not use the situation hyper-graphs or functional programs). We first perform experiments with the provided ground truth object bounding boxes, labels and human-object relations as well as provided human pose predictions from Alphapose [16]. Each of these is projected to a distributed representation through learned non-linear projections to obtain object token-wise representations. A further learnable positional embedding for each frame is added to these representations which are then flattened across frames to form the visual input to IPRM. For the language encoder, we found both a simple Bi-LSTM and Distil-roberta language encoder obtain similar performance, and hence choose the simpler BiLSTM as the language model. We evaluated models on both 16 sampled frames and 32 sampled frames, and empirically found using 32 frames has _∼_ 1.3% higher performance and used it for our primary experiments. A batch size of 64 was used with learning rate 1e-4 over 20 epochs with early 

18 

stopping. We evaluated the models on both the validation split and official test evaluation server `https://eval.ai/web/challenges/challenge-page/1325/overview` . For the all-predicted (no ground-truth) visual input setup, similar to [79], we utilize a fasterRCNN [19] object extractor, ST-Trans scene graph extractor [10] and the same Alphapose predictor to obtain predicted object bounding boxes, labels, human-object relations and human poses. We observe a drop of _∼_ 9% in predicted setup similar to observations in [79], suggesting further performance can be achieved through better object and relationship detection backbones. 

**AGQAv2** . We use the AGQAv2 [20, 21] benchmark that comprises balanced training and test splits. We followed the same methodology as in STAR for language and visual backbones. Since AGQAv2 has a very large training and validation set, we found 8 epochs to be a sufficient number (with most performance increment observed in the 1st epoch itself). 

**CLEVRER-Humans.** We use the CLEVRER-Humans dataset introduced in [51] for temporal, physical and causal video reasoning which comprises videos from the original CLEVRER dataset [84]. Similar to in STAR-VideoQA and neurosymbolic models [84, 85], we utilize a pretrained faster-RCNN based object localization and attribute prediction network from [84]. We again form object-level representations by concatenating learned projections of object-bounding box coordinates and predicted object attributes (i.e. color, shape and material). A frame-level learnable positional embedding is added and object-tokens across frames are flattened to form the final visual input to IPRM. For the language encoder, we used a simple bi-LSTM similar to existing methods. Note we do not use the functional programs or event causal graphs in our model. The batch size was 128 with a learning rate of 8e-5 and every 4 frames sampled (resulting on average 32 sampled frames). We evaluated models in the three setups – from scratch, zero-shot (CLEVRER-pretrained) and finetuned (CLEVRER-pretrained). Since the CLEVRER-Humans dataset is relatively small (comprising only 1076 questions _∼_ 8 batches; _∼_ 0.5% of original CLEVRER), for scratch training we trained for 250 epochs (with early stopping) while for finetuning we finetuned for 150 epochs (with 35 epochs for original CLEVRER training). 

**CLEVR-CoGen.** We use the CLEVR-CoGen dataset from [34] and follow the same setup as in CLEVR-Humans. We use a simpler bi-LSTM language encoder for experiments since the questions are synthetic program-generated unlike in CLEVR-Humans (crowdsourced free-form). We trained our model on condition A for 40 epochs (with early stopping) and used the best cond. A validation performance model to evaluate generalization performance on cond.B. For finetuning on cond.B we finetuned the best cond.A model for 20 epochs and used the best cond.B validation performance model to also evaluate on cond.A. All other hyperparameters are the same as mentioned for CLEVRHumans. 

**NLVR.** We use the NLVRv1 and NLVRv2 datasets from [63, 64]. NLVRv1 comprises 3 synthetic images and a language statement while NLVRv2 comprises 2 real-world images and a lang. statement. For both datasets, the obtained visual tokens for each images was flattened to obtain the final visual input and an image-wise positional embedding was added to indicate image order. For the language encoder, we used a simple Bi-LSTM. 

## **D Potential Negative Impact** 

In relation to VQA and deep-learning methods in general, the deployment of IPRM in real-world applications without thorough consideration of dataset or training distribution biases, could inadvertently reinforce existing vision, language and cultural biases present in the data, leading to erroneous outcomes or skewed answers. Further, the deployment of VQA methods such as IPRM in sensitive domains such as healthcare or scene/footage analysis could raise ethical concerns, including privacy violations, algorithmic reliability, and the potential for unintended consequences stemming from erroneous or biased predictions. 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0020-00.png)


**Are there any objects that have both shape and size in common with the object that is found closest to the front? (GT: Yes; Pred: Yes)** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0020-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0020-03.png)


**----- Start of picture text -----**<br>
Reasoning Computation Steps<br>Reasoning Computation Steps<br>Parallel Operations<br>**----- End of picture text -----**<br>


Figure 8: **Top** : original image and question; **middle** : language attentions across parallel operations (clubbed together; op_k represents parallel operation k) and computation steps. **Bottom** : Visual attentions across parallel ops and computation steps. Here, IPRM correctly utilizes parallel and iterative compute to locate the correct candidate object for prediction (to which all operations attend in last step). 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0021-00.png)


**What shape is the object closest to the gray object with the maximum occuring shape? Pred: cube GT: cube** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0021-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0021-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0021-04.png)


**----- Start of picture text -----**<br>
Parallel Operations<br>**----- End of picture text -----**<br>


**Reasoning Computation Steps** 

Figure 9: In this example, IPRM predicts the correct answer and its visual attention trace provides evidence of correct intermediate reasoning. In penultimate reasoning step, IPRM correctly localizes the gray object with maximum occuring shape (cylinder) and in the final step, the parallel operations attend to both the cyan cube and the brown cylinder closest to previously identified gray cylinder. 

21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0022-00.png)


**Are the two objects that are of a primary color, but not red, of the same shape? Pred: no GT: yes** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0022-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0022-03.png)


**----- Start of picture text -----**<br>
Reasoning Computation Steps<br>Parallel Operations<br>**----- End of picture text -----**<br>


Figure 10: Example where IPRM outputs incorrect answer and the intermediate reasoning appears faulty possibly due to lack of understanding what a “primary color is". The pair of blue (a primary color) cubes in this case should have been identified but are not visually attended in any of the operations across reasoning steps). 

22 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0023-00.png)


**What shape is the object left of the blue small object with the maximum occuring shape? Pred: sphere GT: sphere** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0023-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0023-03.png)


**----- Start of picture text -----**<br>
Reasoning Computation Steps<br>Parallel Operations<br>**----- End of picture text -----**<br>


Figure 11: Example wherein IPRM produces correct answer but its visual attention trace suggests intermediate reasoning may be imprecise. The maximum occuring shape is cube; however both the blue small cylinder and blue small cube appear to be attended in the penultimate step as the “blue small object with max occuring shape" making the reasoning and prediction less reliable. 

23 

1 `def iprm_forward (vis_tokens , #B x Nv x Dm` 2 `lang_tokens , #B x Nl x Dm` 3 `lang_summary_rep , #B x Dm` 4 `num_parallel_ops =6,` 5 `num_iterative_steps =9,` 6 `mem_window_len =2):` 7 `mem_op_states = []` 8 `mem_res_states = []` 9 `lang_atts = []` 10 `vis_atts = []` 11 12 `#0. Initialize memory` 13 `b, d = vis_tokens.size(O), vis_tokens.size (-1)` 14 `mem_op_state , mem_res_state = _init_mem_state (num_parallel_ops , b,d)` 15 `mem_op_states .append( mem_op_state )` 16 `mem_res_states .append( mem_res_state )` 17 `for i in range ( num_iterative_steps ):` 18 `#1. Form new set of latent operations from lang. token features` 19 `new_ops , lang_att = operation_formation (lang_tokens , mem_op_state )` 20 21 `#2. Execute new operations on vis. input to form new results` 22 `new_ops_results , vis_att = operation_execution (vis_tokens , new_ops , mem_res_state )` 23 24 `#3. Apply operation composition` 25 `mem_op_state , mem_res_state = operation_composition (new_ops , new_ops_results , mem_op_states , mem_res_states )` 26 27 `#4. Maintain memory states within lookback window` 28 `mem_op_states .append( mem_op_state )` 29 `mem_res_states .append( mem_res_state )` 30 `mem_op_states = mem_op_states [ min (-1, -mem_window_len ):]` 31 `mem_res_states = mem_res_states [ min (-1, -mem_window_len ):]` 32 33 `#5. Store lang. and vis. atts for visualization` 34 `lang_atts.append(lang_att)` 35 `vis_atts.append(vis_att)` 36 37 `#6. ’Pool ’ final result` 38 `final_result = pool_final_result (mem_res_state , mem_op_state , lang_summary_rep )` 39 40 `return final_result , lang_atts , vis_atts` 

Figure 12: IPRM pseudocode (1/3) 

24 

1 `#Below , "Lin" refers to a linear layer` 2 `#and ‘"MLP" refers to a 2-layer multi -layer -perceptron layer` 3 `def operation_formation (lang_tokens , #B x Nl x Dm` 4 `prev_op_state #B x Nop x Dm (Nop=num parallel ops)` 5 `):` 6 `#1. Form new op "query" based on prior op state` 7 `op_q = MLP_l( prev_op_state ) #paper eq. 4` 8 9 `#2. Use lang_token_feats as attn "key" and "value" (paper eq. 5)` 10 `lang_k = lang_tokens` 11 `lang_v = lang_tokens` 12 13 `#3. Retrieve new latent ops from lang. rep through attention` 14 `latent_ops , lang_attn = mod_attn(op_q , lang_k , lang_v ,` 15 `lang_attn_proj ) #paper eq .6; L194` 16 17 `return latent_ops , lang_attn` 18 19 `def operation_execution (vis_tokens , #B x Nv x Dm` 20 `new_ops , #B x Nop x Dm` 21 `prev_res_state ): #B x Nop x Dm` 22 `#1. Form feature modulation weights (paper eq .7)` 23 `s_v = concat ([ Lin_op(new_ops), Lin_res( prev_res_state )]) #concat across feat. axis` 24 `s_v = Lin_s(s_v)` 25 26 `#2. Form visual attention "key" (paper eqs. 8 and 9)` 27 `vis_red_rep = Lin_v1(vis_tokens )` 28 `mod_vis = s_v * vis_red_rep` 29 `Nop = mod_vis.size (1)` 30 `vis_k = MLP_v(concat ([ mod_vis , vis_red_rep ])) #concat across feat. axis` 31 32 `#3. Form visual attention "query" and "value" (paper eq. 10)` 33 `vis_q = Lin_op_q(new_ops)` 34 `vis_v = Lin_v2(vis_tokens)` 35 36 `#4. Obtain new latent "results" through vis attention (paper eq .11)` 37 `latent_results , vis_attn = mod_attn(vis_q , vis_k , vis_v , vis_att_proj )` 38 39 `return latent_results , vis_attn` 40 41 `def mod_attn(q, k, v, att_proj_layer , attn_mask):` 42 `qk_mult = q*k #element -wise product` 43 `attn_wt = att_proj_layer (qk_mult) #linear projection (paper L194)` 44 `attn_wt = softmax(attn_wt + (attn_mask * -1e30))` 45 `out = (attn_wt * v). sum () #sum across feature axis` 46 `return out , attn_wt` 

Figure 13: IPRM pseudocode (2/3) 

25 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios_images/nips_2024_Learning_to_Reason_Iteratively_and_Parallelly_for_Complex_Visual_Reasoning_Scenarios.pdf-0026-00.png)


**----- Start of picture text -----**<br>
1 def operation_composition (new_ops , #B x Nop x Dm<br>2 new_res , #B x Nop x Dm<br>3 mem_op_states , #list of W elements: each being B x Nop x Dm<br>4 mem_res_states #list of W elements: each being B x Nop x Dm<br>5 ):<br>6 #1. Integrate new -ops and results into memory (paper eq. 12 and 13)<br>7 inter_op_state = Lin_op_u(new_ops) + Lin_op_h( mem_op_states [ -1])<br>8 inter_res_state = Lin_res_u(new_res) + Lin_res_h( mem_res_states [ -1])<br>9<br>10 #2. Concat operation and result states over memory lookback window<br>11 op_states_windowed = concat ([ inter_op_state , mem_op_states ])<br>12 res_states_windowed = concat ([ inter_res_state , mem_res_states ])<br>13<br>14 #3. Form inter -operation queries and keys (paper eq. 14)<br>15 op_queries = Lin_op_q( inter_op_state )<br>16 op_keys = Lin_op_k( op_states_windowed )<br>17<br>18 #4. Form inter -operation op values and res values (paper eq. 15 -16)<br>19 op_values = Lin_op_v( op_states_windowed )<br>20 res_values = Lin_res_v( res_states_windowed )<br>21<br>22 #5. Compute inter -operation attention (paper eq. 17)<br>23 attn_mask = identity_matrix (op_keys.size (1))[: op_queries .size (1)]<br>24 new_op_state , op_attn_wt = mod_attn(op_queries , op_keys , op_values , op_attn_proj ,<br>attn_mask)<br>25<br>26 #6. Obtain new operation and result states (paper eq. 18 -19)<br>27 new_op_state = new_op_state + Lin_op_u2( inter_op_state )<br>28 new_res_state = op_attn_wt* new_res_state + Lin_res_v2( inter_res_state )<br>29<br>30 return new_op_state , new_res_state<br>31<br>32 def _init_mem_state (num_parallel_ops , b):<br>33 #slice specified num parallel ops from initialized params from N(0 ,1)$<br>34 op_init_state = op_init_param [: num_parallel_ops ]<br>35 res_init_state = res_init_param [: num_parallel_ops ]<br>36 #broadcast batch -wise to get B x N_op x Dm<br>37 return op_init_state .repeat(b,1 ,1), res_init_state .repeat(b,1 ,1)<br>38<br>39 def pool_final_result (res_state , op_state , lang_summary_rep ):<br>40 pool_q = Lin_pq( lang_summary_rep )<br>41 pool_k = Lin_pk(op_state)<br>42 return mod_attn(pool_q , pool_k , res_state)<br>**----- End of picture text -----**<br>


Figure 14: IPRM pseudocode (3/3) 

26 

