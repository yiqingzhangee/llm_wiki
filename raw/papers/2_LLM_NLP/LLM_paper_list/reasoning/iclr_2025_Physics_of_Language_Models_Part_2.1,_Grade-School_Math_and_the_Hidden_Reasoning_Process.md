# Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process 

Tian Ye 

`tye2@andrew.cmu.edu` CMU / Meta FAIR 

Zicheng Xu Yuanzhi Li Zeyuan Allen-Zhu `zichengxu@meta.com Yuanzhi.Li@mbzuai.ac.ae zeyuanallenzhu@meta.com` Meta FAIR MBZUAI Meta FAIR 

July 31, 2024[∗] 

## **Abstract** 

Recent advances in language models have demonstrated their capability to solve mathematical reasoning problems, achieving near-perfect accuracy on grade-school level math benchmarks like GSM8K. In this paper, we formally study how language models solve these problems. We design a series of controlled experiments to address several fundamental questions: (1) Can language models truly develop reasoning skills, or do they simply memorize templates? (2) What is the model’s hidden (mental) reasoning process? (3) Do models solve math questions using skills similar to or different from humans? (4) Do models trained on GSM8K-like datasets develop reasoning skills beyond those necessary for solving GSM8K problems? (5) What mental process causes models to make reasoning mistakes? (6) How large or deep must a model be to effectively solve GSM8K-level math questions? 

Our study uncovers many hidden mechanisms by which language models solve mathematical questions, providing insights that extend beyond current understandings of LLMs. 

> ∗ Project page + video: `https://physics.allen-zhu.com/part-2-grade-school-math/part-2-1` . We would like to thank Lin Xiao, Chunting Zhou for many helpful conversations. We would like to extend special thanks to Lucca Bertoncini, Liao Hu, Caleb Ho, Wil Johnson, Apostolos Kokolis, Parth Malani, Alexander Miller, Junjie Qian and Shubho Sengupta from Meta FAIR; Henry Estela, Rizwan Hashmi, Lucas Noah, and Maxwell Taylor from the Meta Cloud Foundation team; as well as Ian Clark, Gourab De, Anmol Mann, and Max Pfeifer from W&B. Without their invaluable support, the experiments in this paper would not have been possible. 

## **1 Introduction** 

The field of language models has made significant progress in recent years. Large models like GPT4 [17] have shown initial signs of general intelligence [8], while smaller models have demonstrated good reasoning abilities by solving challenging coding and math problems [11, 15, 16]. 

In this paper, we focus on the ability of small language models to solve grade-school math problems. Unlike previous works that empirically push the accuracy of models on grade-school math benchmarks like GSM8K [9] and its augmentations (e.g., [16, 22]), we take a more principled approach. We aim to understand the following fundamental questions: 

1. How do language models learn to solve grade-school level math problems? Do they just memorize templates, or do they learn reasoning skills similar to humans? Or do they discover new skills to solve the problems? 

2. Do models trained _solely_ on grade-school math problems only learn to solve these problems, or do they develop some more general intelligence? 

3. How small can a language model be while still solving grade-school math problems? Is depth (number of layers) more important than width (number of neurons per layer), or does only size matter as suggested by practitioners [14]? 

These questions are fundamental to understanding the intelligence of language models. To study them, it might seem tempting to start with a pre-trained model and fine-tune it on existing datasets like GSM8K or GPT-4 augmented ones (e.g., [16, 22]). However, this approach has significant limitations: 

- Data contamination. The pretrain data of existing models mostly come from publicly available internet [10], which is a pile of mess. We do not know how many math problems are included or their structures. There is _significant concern regarding_ whether the GSM8K benchmark has been _leaked to language models’ training datasets_ [22]. Even if the exact data is not, the pre-trained model might have seen almost identical questions (e.g., the same problem with different numbers). Thus, this approach cannot answer questions 1-3. We do not know whether a model truly learns the reasoning skills or it simply memorizes problem templates during training. Therefore, we **need full control over the model’s pretrain data** and must train a language model from scratch. This point has been reiterated recently in [2, 3]. 

- Solution diversity. The existing fine-tuning data, such as the GSM8K training set, contains only 7.5K grade-school math problems, which is insufficient to train a model from scratch. Although recent works use GPT-4 to augment GSM8K, this is not enough for our purpose. GPT-4 augmented problems might be biased towards a small number of solution templates, since the original GSM8K data has very few (obviously, at most 8K) solution templates. **We need a much larger, more diverse set of grade-school math problems** . 

With these points in mind, we introduce our framework to generate a large set of diverse gradeschool math (GSM) problems and use the dataset to train (from scratch) and test a GPT2-like language model. In the framework, we focus on the “logical reasoning” aspect of grade-school math problems, which involves the dependency of parameters in the problem statement, such as “Alice’s apple is three times the sum of Bob’s orange and Charles’s banana.” We use synthetic sentences to reduce the difficulty arising from _Common Sense_ , like “a candle burned for 12 hours at 1 inch per hour” (implying the candle is reducing in length). We also remove the difficulty from pure 

1 

arithmetic: we only consider integers and arithmetic mod23.[1] 

Moreover, our framework ensures that the generated math problems are highly diverse and do not come from a small subset of templates. Even ignoring all the arithmetic, English, variable names, and unused parameters, our problems still have more than 90 trillion solution templates (see Proposition 2.2), much larger than the size of GPT2-small (100M). Thus, language models **cannot** solve the math problems in our case **by simply memorizing** the solution templates. 

In this paper, we use the GPT2 model [18], but replace its positional embedding with rotary embedding (RoPE) [7, 20]. We still call it GPT2 for brevity. We summarize our main contributions: 

- Result 2. We demonstrate that the GPT2 model, pretrained on our synthetic dataset, not only achieves 99% accuracy in solving math problems from the same distribution but also outof-distribution generalizes, such as to those of longer reasoning lengths than any seen during training. This is similar to length generalization in arithmetics [6, 13], however, in our case, the model **has never seen** _**any**_ **training example of the same length as in test time.** This signifies that the model can truly learn some reasoning skill instead of memorizing solution templates. 

- Result 3. Crucially, the model can learn to generate shortest solutions, almost always avoiding _unnecessary_ computations. This suggests that the model _formulates a plan_ before it generates, in order to avoid computing any quantities that are not needed towards solving the underlying math problem. 

- Result 4. We examine the model’s internal states through probing, introducing six probing tasks to elucidate _how_ the model solves math problems. For instance, we discover the model (mentally!) preprocesses the full set of necessary parameters before it starts any generation. Likewise, humans also do this preprocess although we write this down on scratch pads. 

- Result 5. Surprisingly, the model also learns _unnecessary, yet important_ skills after pretraining, such as all-pair dependency. Before any question is asked, it already (mentally!) computes with good accuracy which parameters depend on which, even though _some are not needed for solving the math problem_ . Note that computing all-pair dependency is a skill **not needed** to fit all the solutions in the training data. To the best of our knowledge, this is the first evidence that a language model can **learn useful skills beyond** those necessary to fit its pretraining data.[2] This may be a preliminary signal of **where the G in AGI** can come from.[3] 

- Result 6. We explain _why_ mistakes occur. For instance, the model makes systematic errors that can be explained by probing its internal states. Sometimes, these mistakes can be predicted before the model generates answers, making them independent of the random generation process. We connect this to practice, noting that GPT-4/4o also makes similar errors (though we cannot probe their internal states). 

- Result 7+8. The depth of the language model is crucial for its reasoning ability. For example, a 16-layer, 576-dim transformer solves harder problems (in reasoning length) than a 4-layer, 1920-dim one, despite the latter being twice as large. This holds even when Chain-of-Thought (CoT) is used. We explain this necessity in depth by the complexity of the mental processes 

> 1There is a rich literature studying how well language models can learn arithmetic and length generalization, see [13, 23] and the references therein. Modern language models are also equipped with retrieval-augmented generation (RAG), allowing arithmetic computations to be delegated to a calculator. 

> 2In our case, one can solve all the math problems without computing all-pair dependency. Our pretraining data never includes such information — all the solutions only compute necessary variables. 

> 3Indeed, the skill to sort relationships among in-context objects is a general skill, which may lead to — via instruction fine-tuning — skills for solving other tasks, such as discovering causal relationships, determining the influence of parameter changes, etc. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Structure Graph Dependency Graph<br>Film Studio-<br>School Central High RiverviewHigh Central Hig-h's Film St-udio 's MessengerBackpack Film Studio-'s Backpack<br>Classroom Dance Studio Film Studio Film Studio-'s School RNG Dance Studi-o's School<br>Daypack Daypack<br>Backpack School Dayp-ack MessengerBackpack h's BackpackCentral Hig- High's FilmRiverview High's DanceRiverviewStudio<br>Studio<br>neccesary paramter unused paramter question param ter<br>abstract dependency instance dependency<br>**----- End of picture text -----**<br>


Figure 1: Structure and dependency graph corresponding to the op = 7 easy example in (2.1) and (2.2). Dependencies from abstract parameters are drawn in **red** , and from instance parameters are in **black** . 

involved. We advocate for the use of controlled, synthetic data as a more principled approach to derive such claims, contrasting with predictions like “only size matters” based on training loss using internet pretrain data [14]. 

While we refrain from overstating that our findings directly apply to foundation models like GPT-4 or more challenging mathematical reasoning tasks, we believe our work significantly advances the understanding of how language models develop their mathematical reasoning skills, and this **has to be done in a way different from pushing benchmarks** . 

## **2 Result 1: Data Generation** 

**Motivation.** Recall a standard grade-school math problem in the GSM8K dataset [9] looks like: Betty is saving money for a new wallet which costs 100. Betty has only half of the money she needs. Her parents decided to give her 15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? 

This problem involves multiple parameters whose values are connected through various equalities, such as “Betty’s current money = 0.5 _×_ cost of the wallet” and “money given by grandparents = 2 _×_ money given by parents.” Motivated by this, we build a GSM8K-like math dataset through a synthetic generation pipeline that captures the dependencies of parameters. We wish to capture at least the following three types of dependencies. 

1. Direct dependency ( _♡_ ): such as _A_ = 5 _×_ ( _X_ + _Y_ ), so _A_ can be computed after _X_ and _Y_ . 

2. Instance dependency ( _♠_ ): such as “every classroom has X chairs, and there are Y classrooms.” Here, the model must infer the total number of chairs by multiplying X by Y. 

3. Implicit dependency ( _♣_ ): such as “Bob has 3 times more fruits than Alice. Alice has 3 apples, 4 eggs and 2 bananas.” Here, the model must learn that apples and bananas are fruits and egg is not, and “Alice’s fruits” is an abstract parameter derived from the problem statement. 

## **2.1 Step 1: Graph Construction and Problem Generation** 

**Hierarchical categorization.** We use a layered structure of _categories_ , each contains possible _items_ . For instance, categories = (School, Classroom, Backpack) has three layers; category School = _{_ Central High, Riverview High, . . . _}_ ; category Classroom = _{_ Dance Studio, Film Studio, . . . _}_ ; category Backpack = _{_ School Daypack, Messenger Backpack, . . . _}_ . We prepare 4 predefined hierarchical categorizations, each with 4 layers and 100 items in each layer; this represents the world knowledge. 

3 

**Structure graph.** In each math problem, only specific items exist, leading to a _structure graph_ that outlines what sub-items can appear under what item, see Figure 1 (left). For instance, 

- Connecting Dance Studio and School Daypack with an edge signifies an _instance parameter_ , “the number of school daypacks in each dance studio,” which is a quantifiable variable that can be assigned.[4] This captures the instance dependency ( _♠_ ) as mentioned above. 

- _Abstract parameters_ , like “the total number of classrooms in Central High,” cannot be assigned and are excluded from the structure graph. They reflect implicity dependency ( _♣_ ) . 

_Remark_ 2.1 _._ Rather than using simple objects like _Alice’s apple_ or fake items like _Items A/B/C/D_ , this structure allows us to describe abstract parameters and adds 2 levels of complexity to the data: 

- The model must implicitly learn English concepts, such as a classroom category includes 100 different classroom types. These concepts cannot be derived from individual math problems, as only a limited selection of classrooms will be mentioned in each problem. 

- The model is required to hierarchically access multiple items to calculate abstract parameters, as opposed to a straightforward retrieval of “Alice’s apple” in the context.[5] 

**Dependency graph.** The _dependency graph_ is a directed acyclic graph that outlines the dependency among parameters. For each _instance parameter_ , we choose a random set of (up to 4) parameters it can depend on — including possibly a special vertex RNG representing a random number generator. For instance, if “[param A ] is _X_ more than the difference of [param B ] and [param C ]” for _X_ being randomly generated, then we draw edges from B, C and RNG to parameter A. The dependency of abstract parameters is implied by the dependency of instance parameters. This captures direct dependency ( _♡_ ) as mentioned above. We give an examples on the right side of Figure 1, and details for how we randomly generate such dependency graph are in Appendix D.2. 

**Problem generation.** The _problem_ is articulated by describing the dependency graphs in English, one sentence for each instance parameter.[6] (Abstract parameters are not described because they are inherited by the structure graph.) We **randomly permute** the sentence ordering to further increase difficulty. A parameter is selected and asked with a question in the end (or at the beginning). Below is an easy example corresponding to Figure 1; a harder example is in Figure 11. **(Problem - Easy)** The number of each Riverview High’s Film Studio equals 5 times as much as the sum of each Film Studio’s Backpack and each Dance Studio’s School Daypack. The number of each Film Studio’s School Daypack equals 12 more than the sum of each Film Studio’s Messenger Backpack and each Central High’s Film Studio. The number of each Central High’s Film Studio equals the sum of each Dance Studio’s School Daypack and each Film Studio’s Messenger Backpack. The number of each Riverview High’s Dance Studio equals the sum of each Film Studio’s Backpack, each Film Studio’s Messenger Backpack, each Film Studio’s School Daypack and each Central High’s Backpack. The number of each Dance Studio’s School Daypack equals 17. The number of each Film Studio’s Messenger Backpack equals 13. _How many Backpack does Central High have?_ 

(2.1) 

> 4Even though Central High and Rivierside High can both have (possibly multiple) Dance Studios, for simplicity, we assume that each Dance Studio has the same number of School Daypacks. 

> 5For example, the total number of backpacks in Riverview High in Figure 1 is calculated as _ip_ 1 _× ap_ 1 + _ip_ 2 _× ap_ 2 where _ip_ 1 = “Riverview High’s number of Dance Studios”, _ip_ 2 = “Riverview High’s number of Film Studios”, _ap_ 1 = “each Dance Studio’s number of Backpacks”, and _ap_ 2 = “each Film Studio’s number of Backpacks”, with _ip_ 1 _, ip_ 2 being instance parameters and _ap_ 1 _, ap_ 2 abstract parameters. Here, the model must not only retrieve _ip_ 1 _, ip_ 2 but also compute _ap_ 1 _, ap_ 2 hierarchically. 

6 We use simple English sentence templates to describe the problem, and did not worry about grammar mistakes such as singular vs plural forms. There are other randomness besides the dependency graph, such as when parameter _A_ depends on _B, C_ it could be _A_ + _B_ or _A − B_ . 

4 

## **2.2 Step 2: Solution Construction (CoT)** 

Let _solution_ be a sequence of sentences describing the _necessary_ steps towards solving the given problem, where the sentences follow any topological order — also known as Chain-of-Thought, CoT. For each parameter _necessary_ towards answering the final question, we assign to it a random letter among the 52 choices (a..z or A..Z), and use a sentence to describe its computation:[7] 

Define [param] as X; [intermediate steps]; so X = ... 

Throughout this paper, we consider **arithmetics mod** 23 to avoid errors from computation involving large numbers. It is perhaps the easiest to directly see a solution example (corresponding to (2.1)), and a more involved example is in Figure 11: 

**(Solution - Easy)** Define Dance Studio’s School Daypack as p; so p = 17. Define Film Studio’s Messenger Backpack as W; so W = 13. Define Central High’s Film Studio as B; so B = p + W = 17 + 13 = 7. Define Film Studio’s School Daypack as g; R = W + B = 13 + 7 = 20; so g = 12 + R = 12 + 20 = 9. Define Film Studio’s Backpack as w; so w = g + W = 9 + 13 = 22. Define Central High’s Backpack as c; so c = B * w = 7 * 22 = 16. _Answer: 16._ 

(2.2) 

We emphasize that: 

- The solution only contain parameters _necessary_ towards calculating the final query parameter. 

- The solution follows the correct logical order: i.e. all the parameters used in the calculation must have appeared and been computed beforehand. 

- We break computations to binary ops: _g_ = 12+13+7 is broken into _g_ = 12+ _R_ and _R_ = 13+7 in the above solution. The number of semicolons “;” equals the number of _operations_ . This reduces the arithmetic complexity of the solution, which is not the focus of this paper.[8] 

## **2.3 Difficulty Control** 

Although deferring all the pseudocode to Appendix D, we summarize below the main randomness used in the data generation process. This includes the random choice of a hierarchical categorization (i.e., the English part); a structure graph (i.e., the instance parameters); a dependency graph; arithmetic computations on the dependency graph; integer numbers (i.e., the RNG); problem sentence permutation; and the query parameter. 

We use two parameters to control data’s difficulty: ip is the number of instance parameters, and op is the number of solution operations; the data’s difficulty is an increasing function over them. We call our dataset iGSM, to reflect the nature that such synthetic dataset can be of _infinite size_ . We use iGSM[op] _[≤][op,]_[ip] _[≤][ip]_ to denote the data generated with constraint op _≤ op_ and ip _≤ ip_ , and use iGSM[op][=] _[op,]_[ip] _[≤][ip]_ to denote those restricting to op = _op_ .[9] 

## **2.4 Train and Test Datasets** 

We consider two families of datasets. 

- In the iGSM-med data family we use ip _≤_ 20. 

def The training data is iGSM-med[op] _[≤]_[15] = iGSM[op] _[≤]_[15] _[,]_[ip] _[≤]_[20] . We evaluate the pretrained model both in-distribution, on iGSM-med[op] _[≤]_[15] and iGSM-med[op][=15] , and out-of-distribution (OOD), 

> 7There are different ways to format the CoT solution. We noted that starting with “Define [param] as X” instead of [intermediate steps] improves the model’s accuracy, so we have adhered to this CoT format. 

> 8Even GPT-4 can make mistakes on calculating “3 * (4+10) + 12 * (5+6)” without using external calculator. 

> 9We choose op non-uniformly; for instance, we let op = min _{t_ 0 _, t_ 1 _}_ for two random draws _t_ 0 _, t_ 1 _∈_ [ _op_ ]. This ensures that the dataset has more easy data — which makes training faster. (See also similar behavior for arithmetics [13].) 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0007-00.png)


**----- Start of picture text -----**<br>
5-shot on GPT-4-turbo =29/3096.7% =27/3090.0% =24/3080.0% =28/3093.3% =17/3056.7% =17/3056.7% =15/3050.0% =15/3050.0% =13/3043.3% =10/3033.3% =12/3040.0% =10/3033.3% =13/3043.3% 30.0%=9/30 =11/3036.7% 26.7%=8/30 =14/3046.7% =13/3043.3% =12/3040.0%<br>5-shot on GPT-4o =29/3096.7% =30/30100% =27/3090.0% =28/3093.3% =26/3086.7% =26/3086.7% =22/3073.3% =21/3070.0% =21/3070.0% =15/3050.0% =13/3043.3% =17/3056.7% =14/3046.7% =11/3036.7% =13/3043.3% =14/3046.7% =11/3036.7% =14/3046.7% =14/3046.7%<br>guess ans=0 23.5% 28.8% 29.7% 30.7% 30.6% 31.4% 30.3% 31.9% 30.9% 30.1% 31.8% 31.2% 33.0% 30.4% 32.3% 32.6% 32.8% 33.8% 34.5%<br>op=2 op=3 op=4 op=5 op=6 op=7 op=8 op=9 op=10 op=11 op=12 op=13 op=14 op=15 op=16 op=17 op=18 op=19 op=20<br>**----- End of picture text -----**<br>


- Figure 2: GPT-4 [17] few-shot accuracies on iGSM-med _pq_ (with mod5 arithmetics). For each op we tested 30 problems; and guessing _ans_ = 0 _∈{_ 0 _,_ 1 _,_ 2 _,_ 3 _,_ 4 _}_ gives a baseline accuracy around 32%. Details are in Appendix G, where we also give showcase how GPT-4/4o make mistakes. 

on iGSM-med[op][=] _[op]_ for _op ∈{_ 20 _,_ 21 _,_ 22 _,_ 23 _}_ and iGSM-med[op][=] _[op,]_[reask] . Here, reask denotes first generating a problem from iGSM-med[op][=] _[op]_ and then resampling a query parameter.[10] 

- In the iGSM-hard data family we use ip _≤_ 28. 

   - def 

   - The training data is iGSM-hard[op] _[≤]_[21] = iGSM[op] _[≤]_[21] _[,]_[ip] _[≤]_[28] . We evaluate the pretrained model both in-distribution, on iGSM-hard[op] _[≤]_[21] and iGSM-hard[op][=21] , and OOD on iGSM-hard[op][=] _[op]_ for _op ∈{_ 28 _,_ 29 _,_ 30 _,_ 31 _,_ 32 _}_ and iGSM-hard[op][=] _[op,]_[reask] . 

Additionally, we use iGSM-med _pq_ to indicate placing the question _after_ the problem and iGSM-med _qp_ the other way (similarly for iGSM-hard). The difficulty of iGSM-med is already quite non-trivial to humans (at least not solvable with few-shot learning using GPT-4/4o, see Figure 2). 

**Proposition 2.2.** _Ignoring unused parameters, numerics, sentence orderings, English words, a-z and A-Z letter choices, iGSM-med[op]_[=15] _still has at least_ 7 _billion_ solution templates _, and iGSM-hard[op]_[=21] _has at least_ 90 _trillion_ solution templates _._[11] 

**No data contamination.** A goal in synthetic math data generation is to prevent data contamination in internet-based math datasets, as noted in [22]. While it _may be impossible to certify that models trained on internet data are free from contamination_ , in our setting, **we can certify this** : 

1. We perform OOD evaluation such as on op _≥_ 28 while providing only op _≤_ 21 training samples. 

2. We train with data whose hash value of _solution template_ (see Footnote 11) is _<_ 17 (mod 23), and test with those _≥_ 17. This ensures _no template-level overlap between training and testing_ . 

## **3 Result 2-3: Summarize Model’s Behavior Process** 

We use the GPT2 architecture [18] but replacing its absolute positional embedding with rotary embedding [7, 20], yet still referring to it as GPT2 for short.[12] We mostly stick to the 12-layer, 12-head, 768-dim GPT2 (a.k.a. GPT2-small) for experiments, but we explore larger models in Section 6. We use a context length of 768 / 1024 for pretraining on iGSM-med _/_ iGSM-hard and 2048 for evaluation. More details are in Appendix F. 

> 10Due to the topological nature of our data/solution generation process, reask greatly changes the data distribution and the number of operations needed. It provides an excellent OOD sample for evaluation. Details are in Appendix D. 

> 11A solution template is created by replacing all numbers with ‘0’, substituting variables (a-z or A-Z) with letters in their appearance order, and changing parameters to their types (instance or abstract). For instance, “Define Owl Forest’s Elephant as y; so y = 11. Define Parrot Paradise’s Raccoon as t; so t = y = 11.” becomes “Define Inst as a; so a = 0. Define Inst as b; so b = a = 0.” We use birthday paradox to estimate the number of solution templates. If _M_ randomly generated problems yield distinct templates, it suggests with good probability that the total number of templates exceeds Ω( _M_[2] ). 

> 12We also tested with Llama architecture (esp. with gated MLP layers) and did not see any benefit of using it. GPT2-rotary performs no worse than Llama/Mistral for knowledge tasks [4]. We are currently bounded by resources to repeat all experiments in this paper with other architectures that have minor differences from GPT2-rotary. 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0008-00.png)


**----- Start of picture text -----**<br>
iGSM-med_pq iGSM-med_qp iGSM-hard_pq iGSM-hard_qp<br>in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD)<br>beam1 - nosample 99.9 99.1 91.8 87.9 84.0 76.8 91.6 100 99.3 92.4 89.9 84.8 78.2 91.4 100 99.4 94.4 92.0 90.6 86.8 82.8 91.3 100 99.2 94.5 93.2 91.0 88.2 85.3 89.4<br>beam4 - dosample 99.9 99.1 92.0 88.4 84.5 77.7 91.6 100 99.1 92.4 89.6 84.7 78.3 91.4 100 99.3 94.2 92.2 90.3 86.5 82.5 91.3 99.9 99.2 94.4 93.3 90.8 87.7 85.3 89.1<br>op 15op=15op=20op=21op=22op=23 op 15op=15op=20op=21op=22op=23 op 21op=21op=28op=29op=30op=31op=32 op 21op=21op=28op=29op=30op=31op=32<br>op=20 (reask) op=20 (reask) op=28 (reask) op=28 (reask)<br>**----- End of picture text -----**<br>


Figure 3: Test accuracies on the model (pre-)trained from the iGSM-med _pq/qp_ and iGSM-hard _pq/qp_ datasets. 

iGSM-med_pq iGSM-med_qp iGSM-hard_pq iGSM-hard_qp avg unnecessary operation 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 avg unnecessary operation (reask) 0.02 0.11 0.15 0.17 0.19 0.17 0.03 0.11 0.17 0.21 0.19 0.20 0.07 0.46 0.52 0.54 0.57 0.66 0.66 0.09 0.40 0.45 0.45 0.58 0.53 0.59 avg unnecessary parameter 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 avg unnecessary parameter (reask) 0.01 0.09 0.12 0.12 0.14 0.12 0.03 0.10 0.15 0.16 0.15 0.16 0.06 0.34 0.36 0.36 0.38 0.43 0.44 0.07 0.30 0.32 0.31 0.40 0.36 0.42 

Figure 4: Number of unnecessary params / operations used per generated correct solution. Details in Appendix F. 

**Result 2: accuracy.** After sufficient pre-training, we give the model a problem from the test set (without solution) and let it continue to generate (allegedly a solution followed by an answer). Because we have restricted ourselves to a fixed solution format, language models can learn the format easily, allowing us to write a _solution parser_ to check if the solution is fully correct.[13] 

**Result 2.** _Figure 3 shows that GPT2 performs well when pretrained using iGSM-med or iGSM-hard data, even when evaluated out-of-distribution on harder (i.e., larger op) math problems. Thus, the model can truly learn some reasoning skill instead of memorizing solution templates._[14] 

This could be reminiscent of language models’ length generalization capability on arithmetic computations [13, 23]; however, in our case, op captures the “reasoning length” in grade-school math, and our model **has never seen** _**any**_ training example of the same length as in test time.[15] 

Such accuracies also indicate that our iGSM data families are indeed good for pretraining purpose, allowing us to investigate how LLMs can solve grade-school math problems. 

**Result 3: solution redundancy.** We examine whether GPT2 achieves high accuracy by 

• brute-forcedly computing all the parameters during generation (a “level-0” reasoning skill), or 

• computing only necessary parameters to give shortest solutions (a “level-1” reasoning skill). Recall our iGSM (pretrain) data only contains necessary solution steps (i.e., CoT) to simulate what we see in textbook solutions for math problems. For instance, if a problem describes X =3+2, E =3+X, Y =X+2 and asks for the value of Y, then a shortest solution would be “X =3+2=5 and Y =X+2 =7” without ever computing E. 

**Result 3.** _Figure 4 shows that GPT2 predominantly solves the iGSM problems with a “level-1” reasoning skill, avoiding unnecessary computations, even when evaluated out-of-distribution._ 

This finding is significant as it suggests that, unlike humans who usually rely on “backward reasoning” and a scratch pad to write down necessary parameters by backtracking the dependencies 

> 13We check not only the correctness of the final answer 0..22 but also the calculations and parameter dependencies. Language models can learn very complex syntactics, see [1] and the references therein. 

> 14Llama (of the same model size) gives similar performance, but we refrain from repeating all the experiments with another model. We are not interested in small model differences in this theoretical study; instead, we care more about the general behavior of (autoregressive) language models. 

> 15Some others such as Anil et al. [6] start with a transformer pre-trained on internet data; while the transformer may not have seen the same task during training, it’s possible that the model has seen other tasks with the same (or even longer) length and learned to transfer from there. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0009-00.png)


**----- Start of picture text -----**<br>
dep(A,B)  – at the end of problem description,  [Problem] The number of each Riverview High‘s Film Studio equals 5 times as<br>does the model know  parameter A depend on B ? much as the sum of each Film Studio’s Backpack and each Dance Studio‘s<br>e.g.  dep(“Riverview High‘s Film Studio”,  School Daypack.  …  The number of each Film Studio's Messenger Backpack<br>        “Film Studio’s Messenger Bag”)  =  true equals 13.<br>nece(A)  –  after question is asked, does the model  [Question]  How many Backpack does Central High have?<br>know  if A is necessary  for answering question? [Solution]  Define Dance Studio's School Daypack as p; so p = 17. Define Film<br>e.g.  nece(“Riverview High's Film Studio”)  =  false Studio's Messenger Backpack as W; so W = 13. Define Central High's Film Studio<br>as B; so B = p + W = 17 + 13 = 7. Define Film Studio's School Daypack as g; R = W<br>can_next(A)  – in the middle of solution, does the  + B = 13 + 7 = 20; so g = 12 + R = 12 + 20 = 9. Define Film Studio's Backpack as<br>model know if  A can be computed next ? w; so w = g + W = 9 + 13 = 22. Define Central High's Backpack as c; so c = B * w<br>e.g. can_next(“Riverview High's Film Studio”) can_next(“Riverview High's Dance Studio”)  =  true =  false = 7 * 22 = 16.  [Answer]  16.<br>**----- End of picture text -----**<br>


Figure 5: To discover model’s mental (reasoning) process. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0009-02.png)


**----- Start of picture text -----**<br>
Predict  necc (A)  ∈  {True, False}<br>add a  trainable  linear<br>(classification) layer<br>decoder layer (attention + MLP)<br>decoder layer (attention + MLP) freeze  the pretrained LM<br>trainable  rank  𝑟  update<br>[BOS] The number of each Riverview High's Film Studio    ...     How many Backpack does Central High have?  [START]  Dance Studio’s School Daypack [END] on the embedding layer<br>problem question some  parameter A<br>… … … … … …<br>**----- End of picture text -----**<br>


Figure 6: Illustrations of V-probing on the `nece` ( _A_ ) task. For other tasks, see Figure 13. 

from the question [19], the language model can directly generate shortest solutions without using a scratch pad. But, how does it achieve so? We shall investigate in the next section. 

## **4 Result 4-5: Discover Model’s Mental Process** 

To understand how the model learns to solve math problems, we propose studying the following probing tasks, which align closely with human problem-solving strategies: 

- `nece` ( _A_ ): if parameter _A_ is necessary for computing the answer. 

- `dep` ( _A, B_ ): if parameter _A_ (recursively) depends on parameter _B_ given the problem statement. 

- `known` ( _A_ ): if parameter _A_ has already been computed. 

- `value` ( _A_ ): the value of parameter _A_ (a number between 0-22, or 23 if `known` ( _A_ ) = false). 

- `can next` ( _A_ ): if _A_ can be computed in the next solution sentence (namely, its predecessors have all been calculated). Note that _A_ might not be necessary to answer the question. 

- `nece next` ( _A_ ): if parameter _A_ satisfies both `can next` ( _A_ ) and `nece` ( _A_ ). 

For a model to generate the shortest solutions, it must identify `nece` ( _A_ ) for all _A_ ’s in its mental process. This is because whether `nece` ( _A_ ) is true directly corresponds to whether there is a solution sentence to compute _A_ . However, _how early_ does the model recognize this, and how is it stored? Similarly, does it recognize dependencies between parameters ( `dep` )? If so, how early is this mental process completed? Moreover, in the _middle of solution generation_ , does the model keep track of each parameter _A_ ’s value at all times ( `value` , `known` )? Does the model mentally know all possible parameters _A_ that are ready to compute in the next sentence ( `can` ~~`n`~~ `ext` )? Or does it only focus on _A_ that is both ready and necessary ( `nece` ~~`n`~~ `ext` )? 

This section proposes probing technique to answer all of these questions. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0010-00.png)


**----- Start of picture text -----**<br>
iGSM-med can_next(A) iGSM-hard iGSM-med dep(A,B) iGSM-hard iGSM-med known(A) iGSM-hard<br>in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD)<br>baseline - majority guess 57.4 59.9 57.0 56.7 55.7 55.6 61.7 65.3 61.4 61.3 59.2 58.5 58.4 84.6 82.2 83.0 83.5 83.5 83.7 82.6 79.5 80.2 81.3 82.2 82.8 82.7 87.0 77.6 75.0 74.8 74.5 74.3 84.8 73.5 71.7 72.1 72.4 71.8 71.7<br>pretrained model probing (pq) 99.9 99.5 99.3 99.2 99.1 99.0 99.8 99.4 99.1 99.0 99.0 98.9 98.9 99.7 99.3 100 100 100 100 99.6 99.0 98.9 98.9 98.9 98.9 98.9 100 100 100 100 100 100 100 100 100 99.9 100 100 99.9<br>pretrained model probing (qp) 99.6 99.1 98.8 98.6 98.7 98.5 99.9 99.5 99.1 99.0 99.0 99.0 99.0 99.3 99.0 98.9 99.0 99.0 98.9 99.7 99.4 99.2 99.1 99.2 99.1 99.3 100 100 100 100 100 100 100 100 100 100 100 100 100<br>random model probing (pq) 65.8 62.0 62.6 62.3 63.0 63.3 63.1 59.3 60.6 60.5 62.0 62.2 62.3 84.6 82.2 83.0 83.5 83.5 83.7 82.6 79.5 80.2 81.3 82.2 82.8 82.7 87.2 78.4 75.8 75.4 74.7 74.6 85.4 76.0 73.8 73.6 73.0 72.8 71.9<br>random model probing (qp) 65.7 61.9 62.5 62.2 62.7 63.1 63.0 59.2 60.8 60.9 61.8 62.3 62.6 84.6 82.2 83.0 83.5 83.5 83.7 82.6 79.5 80.2 81.3 82.2 82.8 82.7 87.3 78.7 76.3 76.0 75.3 75.4 85.8 76.4 75.1 74.7 74.6 74.2 73.6<br>iGSM-med nece(A) iGSM-hard iGSM-med nece_next(A) iGSM-hard iGSM-med value(A) iGSM-hard<br>in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD)<br>baseline - majority guess 74.7 54.8 50.1 50.4 51.4 52.1 70.3 53.1 56.9 56.4 55.5 55.6 56.7 94.9 93.5 93.1 92.9 92.9 92.8 94.5 93.0 92.6 92.2 92.7 93.0 92.5 86.5 82.5 81.7 77.8 79.2 77.8 86.7 81.9 75.2 77.0 75.6 71.0 80.4<br>pretrained model probing (pq) 99.8 99.8 98.7 97.9 96.9 94.7 99.6 99.6 99.1 98.6 97.9 97.1 95.5 99.9 99.5 99.2 99.1 99.0 98.7 99.7 99.3 98.7 98.6 98.4 98.3 98.3 100 100 100 100 100 100 100 100 100 100 100 100 100<br>pretrained model probing (qp) 99.8 99.9 99.5 99.4 99.3 99.2 99.8 99.8 99.7 99.7 99.6 99.4 99.3 99.5 99.2 98.7 98.3 98.3 97.7 99.7 99.7 99.5 99.4 99.3 99.2 99.2 100 100 100 100 100 100 100 100 100 100 100 100 100<br>random model probing (pq) 74.7 55.9 51.1 51.0 50.1 49.3 69.4 50.2 47.0 46.9 48.1 47.9 47.4 94.6 93.5 93.1 92.9 92.9 92.8 94.5 93.0 92.6 92.2 92.7 93.0 92.5 71.9 53.4 48.5 47.4 46.7 45.5 73.6 55.2 50.2 50.6 48.9 48.8 48.2<br>random model probing (qp) 74.0 56.3 52.2 51.8 51.3 50.7 69.2 51.3 49.1 48.9 50.4 50.8 50.5 94.7 93.5 93.1 92.9 92.9 92.8 94.5 93.0 92.6 92.2 92.7 93.0 92.5 70.9 52.6 47.8 46.7 46.3 45.1 75.2 57.3 52.5 53.2 51.6 51.2 50.5<br>(a) Probing accuracies on the six tasks: can n ext ( A ) ,  dep ( A, B ) ,  known ( A ) ,  nece ( A ) ,  nece n ext ( A ) ,  value ( A ).<br>can_next(A) on negative labels when A is unnecessary can_next(A) on positive labels when A is unnecessary dep(A,B) on negative labels when A is unnecessary dep(A,B) on positive labels when A is unnecessary<br>iGSM-med iGSM-hard iGSM-med iGSM-hard iGSM-med iGSM-hard iGSM-med iGSM-hard<br>pretrained model probing (pq) 99.8 99.3 98.7 98.6 98.4 98.2 99.8 98.9 97.9 97.7 97.4 97.3 97.4 99.8 99.5 99.2 99.2 99.0 99.2 99.7 99.2 98.7 98.7 98.6 98.8 98.6 99.6 99.4 99.3 99.4 99.4 99.5 99.5 98.9 99.2 99.1 99.2 99.3 99.4 99.6 99.5 99.4 99.3 99.2 99.2 99.3 99.3 99.3 99.2 99.3 99.2 99.2<br>pretrained model probing (pq) - reask 99.9 99.7 99.6 99.6 99.5 99.5 99.9 99.7 99.5 99.5 99.4 99.4 99.4 99.9 99.8 99.7 99.7 99.7 99.6 99.9 99.7 99.6 99.6 99.6 99.5 99.6 99.6 99.1 99.1 99.2 99.2 99.2 99.5 98.8 98.6 98.5 98.7 98.7 98.7 99.7 99.6 99.5 99.5 99.4 99.4 99.4 99.4 99.3 99.3 99.3 99.2 99.2<br>pretrained model probing (qp) 99.6 98.8 98.1 98.0 98.0 97.8 99.8 99.1 98.3 98.2 98.0 97.5 97.4 99.7 99.0 99.0 99.0 98.9 99.0 99.8 99.1 98.4 98.3 98.4 98.6 98.7 99.3 99.0 99.2 99.3 99.3 99.4 99.6 99.3 99.4 99.5 99.5 99.5 99.6 98.8 98.4 97.8 97.8 97.8 97.8 99.5 99.2 99.1 99.1 99.0 98.9 98.8<br>pretrained model probing (qp) - reask 99.7 99.3 99.1 99.0 99.0 98.9 99.9 99.7 99.4 99.4 99.3 99.2 99.2 99.7 99.5 99.5 99.6 99.6 99.5 99.9 99.8 99.6 99.6 99.6 99.6 99.7 99.3 98.8 98.9 99.0 99.1 99.1 99.7 99.3 99.2 99.2 99.2 99.2 99.2 98.9 98.5 98.1 97.9 98.1 98.0 99.6 99.4 99.3 99.2 99.2 99.2 99.0<br>op 15op=15op=20op=21op=22op=23op 21op=21op=28op=29op=30op=31op=32 op 15op=15op=20op=21op=22op=23op 21op=21op=28op=29op=30op=31op=32 op 15op=15op=20op=21op=22op=23op 21op=21op=28op=29op=30op=31op=32<br>op 15op=15op=20op=21op=22op=23op 21op=21op=28op=29op=30op=31op=32 op 15op=15op=20op=21op=22op=23op 21op=21op=28op=29op=30op=31op=32 op 15op=15op=20op=21op=22op=23op 21op=21op=28op=29op=30op=31op=32<br>op 15op=15op=20op=21op=22op=23op 21op=21op=28op=29op=30op=31op=32 op 15op=15op=20op=21op=22op=23op 21op=21op=28op=29op=30op=31op=32 op 15op=15op=20op=21op=22op=23op 21op=21op=28op=29op=30op=31op=32 op 15op=15op=20op=21op=22op=23op 21op=21op=28op=29op=30op=31op=32<br>**----- End of picture text -----**<br>


(b) Probing accuracies of `can next` ( _A_ ) _,_ `dep` ( _A, B_ ) _restricted_ to positives/negatives labels in which _A_ is unnecessary 

Figure 7: V-probing accuracies (for beam=1; results for beam=4 are almost identical). Details are in Appendix F.2. 

## **4.1 V-Probing: A Nearly-Linear Probing Method** 

As illustrated in Figure 5, we conduct probing at the end of the problem description for the `dep` task, and end of the question description `nece` task.[16] For other tasks, we probe them at the end of _every_ solution sentence (including the start of the first solution sentence). 

Recall that standard _linear probing_ involves freezing a pretrained language model and checking if a property is linearly encoded at a hidden layer (usually the last layer) for a given token position. This is done by introducing a trainable linear classifier on the hidden states and performing a lightweight finetuning task for this property (see [12] and references therein). 

Our setting is more complex because the properties have one or two conditional variables, _A_ and _B_ , described in plain English. To handle this, we truncate the math problems to the probing position and append tokens [START] and [END] around the descriptions of _A_ (or _A, B_ ). We then probe from the token position of [END] to see if the property is linearly encoded at the last layer. 

Unlike standard linear probing, to account for the input change, we introduce a small trainable rank-8 (linear) update on the input embedding layer. We freeze the pretrained language model and finetune both the linear classifier and the rank-8 update for the desired property. We refer to this as V(ariable)-probing and provide details in Appendix B. An illustration of the `nece` ( _A_ ) probing task is shown in Figure 6. 

We compute the V-probing accuracies on a language model pretrained from iGSM and compare them with the V-probing accuracies on a randomly-initialized transformer model. If the former accuracies are significantly higher, we conclude that the probing signals must have (or be very close to having) come from the pretrained weights, rather than the (lightweight) finetuning stage. 

## **4.2 Probing Results and Findings** 

We present our probing results in Figure 7. The probing accuracies are high for all the tasks, compared to majority guess and random-model probing — except for the very hard OOD cases 

> 16If the problem format is qp (question asked before the problem) then we probe `nece` and `dep` both after the problem description. 

9 

- (i.e., for large op where the model’s generation accuracies fall down to 80% anyways in Figure 3), 

- **Result 4: model solves math problems like humans.** We make the following observations: 

   - When generating solutions, the model not only remembers which parameters have been computed and which have not ( `value` _,_ `known` ) but also knows which parameters can be computed next ( `can` ~~`n`~~ `ext` _,_ `nece next` ). These abilities ensure that the model can solve the given math problem step by step, similar to human problem-solving skills. 

   - By the end of the problem description, the model already knows the full list of necessary parameters ( `nece` ). This indicates that the model has learned to _plan ahead_ , identifying necessary parameters before starting to generate the solution. This aligns with human behavior, except that the model plans mentally while humans typically write this down. This further confirms that the model reaches the “level-1” reasoning skill discussed in Section 3. 

_Remark_ 4.1 _._ The mental process described can be compared to (out-of-context) _knowledge manipulation_ [2], which involves retrieving factual knowledge and performing single-step computations (e.g., retrieving two people’s birth dates to determine who was born earlier). Allen-Zhu and Li [2] found that even single-step computations _cannot_ be performed mentally without a substantial number of pretrain samples. In contrast, this paper studies _in-context_ reasoning and demonstrates that the model can execute very complex mental calculations. 

**Result 5: model learns beyond human reasoning skills.** Remarkably, the model learns `dep` ( _A, B_ ) and `can` ~~`n`~~ `ext` ( _A_ ), even for parameters _A_ not necessary for answering the question, as shown in Figure 7(b). This differs from human problem-solving, where we typically use backward reasoning from the question to identify necessary parameters, often overlooking unnecessary ones [19]. In contrast, language models can pre-compute the all-pair dependency graph `dep` ( _A, B_ ) mentally even before a question is asked. We consider this a “level-2” reasoning skill that is very different from human behavior or mental processes. 

Thus, although this skill is not needed for solving the math problems and although no pretrain data teaches the model to compute “all-pair dependency” — fitting the data only requires computing necessary parameters — the model still discovers it after training. This enables the model to sort relationships among the things it hears, a skill that can be useful for future tasks (via instruction fine-tuning). To our knowledge, this may be the first evidence of a language model acquiring skills _beyond_ those needed for learning its pretrain data; and this may be a preliminary signal of where the G in AGI can come from (generalizing to skills not taught in the pretrain data). 

**Corollary: the backward thinking process.** A key question for AGI success is whether the “backward thinking process” (e.g., “because I want to compute X, but X depends on Y and Y depends on Z, so let me compute Z first”) needs to be explicitly included in the training data. This differs from CoT, where CoT breaks down complex computations into simpler steps, but planning is still required to decide which step to compute first. 

Our findings suggest that, at least for grade-school math problems, with abundant data, this backward thinking process can be autonomously learned through language modeling, without needing to be directly included in the training data. 

## **5 Result 6: Explain Model’s Mistakes** 

We further examine the relationship between our probing results and the model’s generated solutions, focusing on two questions: (1) When does the model answer correctly but include unnecessary 

10 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0012-00.png)


**----- Start of picture text -----**<br>
nece(A)<br>iGSM-med iGSM-hard<br>on all parameters | pq 99.8% 98.7% 97.9% 96.9% 94.7% 99.6% 99.1% 98.6% 97.9% 97.1% 95.5%<br>on all parameters | pq (reask) 93.4% 94.9% 95.7% 95.6% 95.9% 89.6% 89.9% 90.5% 91.6% 92.0% 92.0%<br>on all parameters | qp 99.9% 99.5% 99.4% 99.3% 99.2% 99.8% 99.7% 99.7% 99.6% 99.4% 99.3%<br>on all parameters | qp (reask) 98.5% 98.1% 98.3% 98.5% 98.2% 96.6% 96.1% 96.6% 97.1% 97.2% 97.2%<br>on unnecessary parameter in model's output | pq (reask) | beam1 =105/33231.6% =168/43538.6% =224/45549.2% =206/50940.5% =161/44935.9% =262/128020.5% =225/131417.1% =245/132018.6% =298/138121.6% =364/157023.2% =370/156623.6%<br>on unnecessary parameter in model's output | pq (reask) | beam4 =92/30430.3% =167/43838.1% =204/44545.8% =217/53640.5% =169/48734.7% =252/128019.7% =219/132616.5% =242/131618.4% =272/135320.1% =355/152323.3% =346/149623.1%<br>on unnecessary parameter in model's output | qp (reask) | beam1 =81/37321.7% =124/55122.5% =145/60524.0% =118/54621.6% =125/57521.7% =193/113317.0% =260/116522.3% =234/110821.1% =303/144321.0% =243/128518.9% =367/148424.7%<br>on unnecessary parameter in model's output | qp (reask) | beam4 =80/36721.8% =126/56022.5% =133/58822.6% =120/56921.1% =123/59720.6% =194/113617.1% =254/112522.6% =251/108523.1% =304/140521.6% =252/132519.0% =375/149825.0%<br>op=15 op=20 op=21 op=22 op=23 op=21 op=28 op=29 op=30 op=31 op=32<br>**----- End of picture text -----**<br>


- (a) `nece` ( _A_ ) probing accuracies correlate with model’s outputted unnecessary parameters 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0012-02.png)


**----- Start of picture text -----**<br>
can_next(A) nece_next(A)<br>iGSM-med iGSM-hard iGSM-med iGSM-hard<br>on all parameters | pq 99.3% 99.2% 99.1% 99.0% 99.1% 99.0% 99.0% 98.9% 98.9% 99.2% 99.1% 99.0% 98.7% 98.7% 98.6% 98.4% 98.3% 98.3%<br>on all parameters | qp 98.8% 98.6% 98.7% 98.5% 99.1% 99.0% 99.0% 99.0% 99.0% 98.7% 98.3% 98.3% 97.7% 99.5% 99.4% 99.3% 99.2% 99.2%<br>on first wrong param | pq | beam1 =172/22775.8% =270/35775.6% =330/47469.6% =473/66770.9% =128/22457.1% =185/31858.2% =229/37760.7% =332/52163.7% =419/67062.5% =119/22852.2% =171/35947.6% =197/47441.6% =284/66942.5% =96/22442.9% =130/31840.9% =171/37745.4% =242/52246.4% =317/67147.2%<br>on first wrong param | pq | beam4 =187/24476.6% =280/36776.3% =350/49770.4% =481/68969.8% =141/23061.3% =187/31259.9% =246/39162.9% =356/53366.8% =439/69463.3% =126/24451.6% =174/36947.2% =223/49744.9% =300/69343.3% =91/23039.6% =126/31340.3% =178/39145.5% =248/53446.4% =339/69548.8%<br>on first wrong param | qp | beam1 =190/27968.1% =234/35965.2% =354/53266.5% =503/74267.8% =119/20159.2% =149/25159.4% =204/32762.4% =259/42860.5% =313/50661.9% =143/27951.3% =202/36056.1% =301/53356.5% =443/74659.4% =102/20250.5% =127/25150.6% =173/32752.9% =226/42852.8% =290/50657.3%<br>on first wrong param | qp | beam4 =190/28067.9% =249/36867.7% =352/53665.7% =524/75269.7% =131/20863.0% =148/25358.5% =214/33464.1% =256/44357.8% =310/50960.9% =153/28054.6% =223/36960.4% =302/53656.3% =458/75660.6% =114/20954.5% =130/25351.4% =188/33456.3% =243/44354.9% =285/50956.0%<br>op=20 op=21 op=22 op=23 op=28 op=29 op=30 op=31 op=32 op=20 op=21 op=22 op=23 op=28 op=29 op=30 op=31 op=32<br>**----- End of picture text -----**<br>


   - (b) `can next` ( _A_ ) and `nece next` ( _A_ ) probing accuracies correlate with model’s outputted wrong solutions 

- Figure 8: Probing results correlate with model’s output solutions. We tested 4096 math problems and presented the probing accuracies restricted to (1) unnecessary parameters in the model’s correct output solution ( **top** ), and (2) the first wrong parameter in model’s wrong output solution ( **bottom** ). Details are in Appendix F.2. 

parameters? (2) What causes incorrect answers? We aim to determine if such erroneous behavior of the model aligns with errors in the model’s mental process (via probing). 

For the first question, given the model rarely produces solutions longer than necessary (see Figure 4), we turned to out-of-distribution reask data for evaluation.[17] On this data, pretrained models produce an average of _∼_ 0 _._ 5 unnecessary parameters per solution even for op = 32 (see Figure 4). We examined if these unnecessary parameters _A_ were **incorrectly predicted as** `nece` ( _A_ ) = true in the probing task. Figure 8(a) reveals that this is often indeed the case, thus language models produce solutions with unnecessary steps due to errors in their _mental planning phase_ . 

For the second question, we focused on the model’s _wrong_ solutions and their _first wrong parameters_ . (Using synthetic data, we can easily identify such parameters.) Our findings in Figure 8(b) show that the model’s errors mainly stem from **incorrectly predicting** `nece` ~~`n`~~ `ext` ( _A_ ) **or** `can next` ( _A_ ) **as true** in its internal states when such _A_ ’s are not ready for computation.[18] 

## **Result 6** (Figure 8) **.** _Combining these, we conclude:_ 

- _Many reasoning mistakes made by the language model are systematic, stemming from_ errors in its mental process _, not merely random from the generation process._ 

- _Some of the model’s mistakes can be discovered by probing its inner states_ even before the model opens its mouth _(i.e., before it says the first solution step)._ 

We also observe that GPT-4/4o makes similar mistakes by outputting unnecessary parameters 

> 17Recall this re-samples a query after generating the problem, leading to a different set of necessary parameters. 

> 18In Figure 8(b), we focus on these “first wrong parameters” with correct label being `can` ~~`n`~~ `ext` ( _A_ ) = false or `nece next` ( _A_ ) = false and present the probability that their probing also correctly predicts false. Low accuracy indicates that the model “thought” these parameters were ready for computation, but they were not. 

11 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0013-00.png)


**----- Start of picture text -----**<br>
iGSM-med_pq iGSM-med_qp iGSM-hard_pq iGSM-hard_qp avg<br>in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD) in-dist out-of-dist (OOD)<br>dep4 - size1 - head21 99.5 92.7 74.7 68.0 62.4 54.5 99.4 93.3 73.3 66.8 61.1 54.6 98.9 90.8 72.4 67.7 62.1 57.1 50.6 99.1 89.8 69.4 62.2 57.8 52.3 45.7 72.2<br>dep4 - size2 - head30 99.6 94.7 74.2 67.9 61.6 53.1 99.4 94.5 78.1 71.9 65.7 58.8 97.0 71.7 46.3 40.6 37.0 32.3 27.3 99.4 92.1 74.5 69.5 64.7 59.1 53.2 68.6<br>dep8 - size1 - head15 100 98.8 89.7 86.5 82.8 76.8 100 99.2 92.4 88.5 84.2 78.7 100 99.1 94.6 92.0 89.7 86.4 82.2 100 99.0 92.2 89.6 86.2 82.4 77.3 90.3<br>dep8 - size2 - head21 100 99.3 93.7 91.6 88.3 83.6 99.9 99.0 90.2 87.1 83.3 76.3 100 99.2 93.6 91.3 88.6 85.6 82.6 100 99.1 93.5 91.3 89.1 85.7 81.2 91.3<br>dep12 - size1 - head12 100 99.3 92.0 88.9 84.2 77.9 100 99.4 92.2 89.2 83.9 77.9 100 99.5 96.0 94.1 91.0 88.5 84.3 100 99.3 95.3 93.0 91.9 88.0 84.5 91.9<br>dep12 - size2 - head17 100 99.5 94.0 91.9 89.0 82.7 100 99.0 90.8 85.4 80.2 73.2 100 99.8 97.1 95.5 93.5 91.8 88.0 100 99.5 94.5 91.9 88.9 86.8 81.3 92.1<br>dep16 - size1 - head10 100 99.6 94.6 91.9 87.9 82.7 100 99.5 89.9 85.0 79.1 71.1 100 99.6 97.0 95.2 94.2 92.2 88.5 100 99.4 95.8 93.8 92.4 88.9 85.8 92.5<br>dep16 - size2 - head15 100 99.8 95.9 93.7 90.4 86.5 100 99.8 95.6 93.5 90.3 84.3 100 99.7 97.5 96.3 95.1 92.9 89.5 100 99.8 97.3 96.0 94.2 91.9 88.9 95.0<br>dep20 - size1 - head9 100 99.8 95.5 93.6 90.0 86.3 100 99.6 94.8 91.4 87.4 80.4 100 99.8 97.0 95.1 94.0 91.0 87.4 100 99.6 96.6 94.5 92.8 90.1 86.5 94.0<br>dep20 - size2 - head13 100 99.8 95.8 93.3 89.2 84.4 100 99.6 93.7 91.8 87.4 81.3 100 99.8 98.0 96.7 95.9 93.9 90.9 100 99.9 97.5 96.0 95.2 92.4 89.7 94.7<br>op 15op=15op=20op=21op=22op=23op 15op=15op=20op=21op=22op=23 op 21op=21op=28op=29op=30op=31op=32op 21op=21op=28op=29op=30op=31op=32<br>**----- End of picture text -----**<br>


Figure 9: Accuracies for GPT2 models of different depth/widths pretrained on iGSM datasets. Details in Appendix F. 

or insisting on computing parameters _A_ with `can` ~~`n`~~ `ext` ( _A_ ) = false (see Appendix G). This further hints that our findings may be applicable more broadly. 

## **6 Result 7-8: Depth vs. Reasoning Length** 

Our controlled dataset enables a systematic exploration of the relationship between a language model’s depth and its reasoning length. 

Recent studies have demonstrated that for knowledge storage and extraction, only model size matters (even for 2-layer transformers) [4]. Furthermore, both the seminal scaling-law paper by OpenAI [14] and theoretical studies in deep learning [5] suggest that model depth/width might have a minimal impact universally. Contrary to these findings, we present evidence that[19] 

**Result 7** (Figure 9) **.** _Language model depth is crucial for mathematical reasoning._ 

Specifically, we experimented with models of depths 4/8/12/16/20 and two sizes (a smaller size 1 and a larger size 2).[20] From Figure 9, we observe that a 4-layer transformer, even with 1920 hidden dimensions, underperforms on our math datasets. Conversely, deeper but smaller models, such as a 20-layer 576-dim, perform very well. Comparing accuracies vertically reveals a clear correlation between model depth and performance. Thus, we infer that depth is likely essential for reasoning tasks, such as solving grade-school math problems. 

Next, we try to reveal “why” this happens. We delved into how depth influences math problemsolving skills through the `nece` probing task, focusing on necessary parameters at distance _t_ from the query parameter, for _t ∈{_ 1 _,_ 2 _, . . . ,_ 8 _}_ . These parameters all have `nece` ( _A_ ) = true, but we can probe the model to see how correct they are at predicting `nece` ( _A_ ) at different hidden layers. 

Figure 10 shows our result. It reveals a correlation between the model’s layer hierarchy, reasoning accuracy, and _mental reasoning depth_ . Shallower layers excel at predicting `nece` ( _A_ ) for parameters _A_ closer to the query, whereas deeper layers are more accurate and can predict `nece` ( _A_ ) for parameters further from the query. This suggests that the model employs layer-by-layer reasoning during the planning phase to recursively identify all parameters the query depends on, and: 

**Result 8** (Figure 10+14) **.** _The depth of a language model is crucial, likely due to the complexity of its hidden (mental) reasoning processes. A t-step mental reasoning, such as mentally computing_ _`nece`_ ( _A_ ) _for parameters A that are a distance t from the query, may require deeper models for larger t, assuming all other hyperparameters remain constant._ 

> 19Math reasoning data only occupies a tiny fraction of pretraining data for language models, thus one might not observe a difference if we only look at the perplexity as in the original scaling law paper [14]. 

> 20GPT2- _ℓ_ - _h_ represents an _ℓ_ -layer, _h_ -head, 64 _h_ -dimensional GPT2 model. Size-1 models are GPT2-4-21, GPT2-815, GPT2-12-12, GPT2-16-10, GPT2-20-9, with similar parameter counts; size-2 models are GPT2-4-30, GPT2-8-21, GPT2-12-17, GPT2-16-15, GPT2-20-13, approximately twice the size of size-1 models. 

12 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0014-00.png)


**----- Start of picture text -----**<br>
med_pq - op15 med_pq - op20 med_qp - op15 med_qp - op20 hard_pq - op21 hard_pq - op28 hard_qp - op21 hard_qp - op28<br>100 100 100 100 100 100 100 100 layer5<br>50 50 50 50 50 50 50 50 layer10layer15<br>0 0 0 0 0 0 0 0 layer20<br>depth20 - size1 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8<br>**----- End of picture text -----**<br>


Figure 10: Increasing probing accuracies of `nece` ( _A_ ) with increasing layer depth. The x-axis denotes the distance of parameter _A_ to the query parameter, with **colors from light to dark to represent layers 1 to 20** . This figure is for a 20-layer GPT2 model; for other model depths/sizes, see Figure 14. 

We make two disclaimers here. First, if the “backward thinking process” is added as CoT to the data (see the end of Section 4.2), then deep mental thinking is no longer required, reducing the language model’s depth requirement. However, in practice, many such “thinking processes” may not be included in standard math solutions or languages in general. 

Second, the above claim does not imply that “a _t_ -step mental thinking requires a depth- _t_ transformer”. It is plausible for a single transformer layer (containing many sub-layers) to implement _t >_ 1 mental thinking steps, though possibly with reduced accuracy as _t_ increases. We refrain from providing an exact correlation in this paper, as it heavily depends on the data distribution. 

## **7 Conclusion** 

We use a synthetic setting to demonstrate that language models can learn to solve grade-school math problems through true generalization, rather than relying on data contamination or template memorization. We develop probing techniques to examine the models’ hidden reasoning processes. Our findings reveal that these models can learn math skills aligned with human cognitive processes, as well as “new thinking processes” not present in the training data. Additionally, we propose a method to predict a model’s errors before it begins to solve a problem and to explain why models make mistakes when they occur. Based on this discovery, we write a separate paper to improve language models’ math reasoning accuracy [21]. We also provide a principled approach to connect the model’s depth to its capable reasoning length. We believe this research opens doors to study the mathematical reasoning skills of language models from a different angle compared to pushing math benchmarks. 

One may argue that iGSM may be very different from the pretrain data that modern LLMs use. While this may be true, we are looking into the future. Recall, even GPT-4/4o of today cannot few-shot learn to solve iGSM-med[op][=11] (see Figure 2). From this perspective, it is reasonable to believe that future versions of LLMs will rely on synthetic math pretrain data to improve their reasoning skills. While one may not directly use iGSM, it is tempting to use existing LLMs (such as Llama-3) to turn iGSM into more natural formats while keeping the logical chains. On the other hand, we have discovered that models trained purely on the iGSM data make similar mistakes compared to GPT-4/4o (see Section 5 and Appendix G). This further confirms that our findings do connect to practice, regarding the model’s hidden reasoning process. 

Finally, Part 2 of this work series focuses on how language models solve grade-school math problems (including Part 2.2 [21]). We also cover how language models learn language structures in Part 1 [1] (in particular, how they mentally perform dynamical programming), and learn world knowledge in Part 3 [2–4]. 

13 

## Appendix 

## **— A Result 1 An Example in iGSM-hard with op** = 21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0015-02.png)


**----- Start of picture text -----**<br>
(Problem- A Hard Example) The number of each Jungle Jim’s International Market’s Cheese equals the sum of each Parmesan Cheese’s Pear<br>and each The Fresh Market’s Ice Cream. The number of each Ice Cream’s Pineapple equals 2 more than each Goat Cheese’s Grape. The number<br>of each New Seasons Market’s Goat Cheese equals the sum of each Residential College District’s Jungle Jim’s International Market, each Jungle<br>Jim’s International Market’s Parmesan Cheese and each Residential College District’s Supermarket. The number of each Arts Campus’s New<br>Seasons Market equals each Cheese’s Pineapple. The number of each Goat Cheese’s Banana equals each Vocational School District’s Product.<br>The number of each Residential College District’s Jungle Jim’s International Market equals 5 more than each Ice Cream’s Grape. The number<br>of each Parmesan Cheese’s Pineapple equals each Parmesan Cheese’s Pear. The number of each Residential College District’s The Fresh Market<br>equals each Arts Campus’s Trader Joe’s. The number of each Arts Campus’s Trader Joe’s equals each Parmesan Cheese’s Ingredient. The<br>number of each Goat Cheese’s Grape equals 0. The number of each The Fresh Market’s Ice Cream equals 13 more than the difference of each<br>Residential College District’s The Fresh Market and each Parmesan Cheese’s Grape. The number of each Goat Cheese’s Pineapple equals each<br>New Seasons Market’s Product. The number of each Vocational School District’s The Fresh Market equals the sum of each Trader Joe’s’s Cheese<br>and each The Fresh Market’s Cheese. The number of each Trader Joe’s’s Cheese equals 6. The number of each The Fresh Market’s Cheese<br>equals 3. The number of each Jungle Jim’s International Market’s Ice Cream equals the difference of each Ice Cream’s Banana and each Goat<br>Cheese’s Grape. The number of each Jungle Jim’s International Market’s Parmesan Cheese equals each Ice Cream’s Pineapple. The number of<br>each Parmesan Cheese’s Pear equals the difference of each Goat Cheese’s Grape and each Ice Cream’s Grape. The number of each Parmesan<br>Cheese’s Grape equals 12 times as much as each Residential College District’s Jungle Jim’s International Market. The number of each The Fresh<br>Market’s Parmesan Cheese equals each The Fresh Market’s Cheese. The number of each Ice Cream’s Banana equals the sum of each Parmesan<br>Cheese’s Pineapple and each Ice Cream’s Pineapple. The number of each School District’s Jungle Jim’s International Market equals each The<br>Fresh Market’s Ice Cream. The number of each Cheese’s Pineapple equals 20 more than the sum of each Trader Joe’s’s Cheese and each The<br>Fresh Market’s Cheese. The number of each Trader Joe’s’s Parmesan Cheese equals 16. The number of each Ice Cream’s Pear equals 8. The<br>number of each Ice Cream’s Grape equals each Goat Cheese’s Grape. How many Product does School District have?<br>(Solution- A Hard Example) Define Goat Cheese’s Grape as u; so u = 0. Define Ice Cream’s Grape as x; so x = u = 0. Define Residential<br>College District’s Jungle Jim’s International Market as N; so N = 5 + x = 5 + 0 = 5. Define Parmesan Cheese’s Pear as G; so G = u- x = 0 -<br>0 = 0. Define Parmesan Cheese’s Grape as f; so f = 12 * N = 12 * 5 = 14. Define Parmesan Cheese’s Pineapple as C; so C = G = 0. Define<br>Parmesan Cheese’s Ingredient as Z; e = f + C = 14 + 0 = 14; so Z = e + G = 14 + 0 = 14. Define Arts Campus’s Trader Joe’s as q; so q =<br>Z = 14. Define Residential College District’s The Fresh Market as j; so j = q = 14. Define Ice Cream’s Pineapple as X; so X = 2 + u = 2 + 0<br>= 2. Define Ice Cream’s Banana as K; so K = C + X = 0 + 2 = 2. Define The Fresh Market’s Ice Cream as P; i = j- f = 14 - 14 = 0; so P =<br>13 + i = 13 + 0 = 13. Define Jungle Jim’s International Market’s Ice Cream as R; so R = K- u = 2 - 0 = 2. Define School District’s Jungle<br>Jim’s International Market as V; so V = P = 13. Define Jungle Jim’s International Market’s Cheese as v; so v = G + P = 0 + 13 = 13. Define<br>Jungle Jim’s International Market’s Parmesan Cheese as S; so S = X = 2. Define Jungle Jim’s International Market’s Product as y; U = S + R<br>= 2 + 2 = 4; so y = U + v = 4 + 13 = 17. Define School District’s Product as J; so J = V * y = 13 * 17 = 14. Answer: 14.<br>Structure Graph Dependency Graph<br>DistrictSupermarket College Dis-Trader Joe'sResidentialtrict School Dist-Internation-Jungle Jim'sal Marketrict New SeasonsArts CampusMarket School Dist-VocationalThe FreshMarketrict The FreshIce Cream'sBananaNew SeasonsMarket's Pr-oductSchool Dist-rict's Jung-le Jim's In-ternationalMarketCheese's Pi-neapple s's ParmesanTrader Joe'-Cheese Ice Cream'sPear Ice Cream'sGrape Jungle Jim'sInternation-al Market'sCheese Ice Cream'sPineapple New SeasonsMarket's Go-at Cheese [Arts Campus-] 's New Seas-ons MarketSchool Dist-rict's Prod-Vocationaluct Goat Cheese-'s BananaCollege Dis-Internation-Residentialtrict's Jun-al Marketgle Jim'sParmesan Ch-eese's Pine-Residentialapple<br>rmesan Chee-Market's Pa-School Dist-rict's Prod-se RNG Arts Campus-Fresh MarketCollege Dis-'s Tradertrict's The<br>Product Parmesan Ch-eese Ice Cream Cheese Goat Cheese Parmesan Ch-eese's Grapeuct ResidentialGoat Cheese-'s GrapeJoe's<br>Ingredient Grape Pineapple Banana Pear Parmesan Ch-eese's PearParmesan Ch-eese's Ingr-edient Parmesan Ch-Jungle Jim'sInternation-al Market'seese Jungle Jim'sInternation-al Market'sIce Cream Market's Ch-The Fresheese Trader Joe'-s's Cheese School Dist-Vocationalrict's The Market's Pr-The Freshoduct Goat Cheese-'s Pineapple Jungle Jim'sInternation-al Market'sProduct Market's IceThe FreshCreamCollege Dis-trict's Sup-ermarket<br>Fresh Market<br>neccesary paramter unused paramter question paramter<br>abstract dependency instance dependency<br>**----- End of picture text -----**<br>


Figure 11: An example with op = 21 in iGSM-hard _pq_ used for training. Don’t forget during testing we evaluate models on op = 28 which is even harder. 

14 

|op<br><br>pretrained model probing (pq)<br>pretrained model probing (pq) - reask<br>pretrained model probing (qp)<br>pretrained model probing (qp) - reask|iGSM-med<br>99.7 99.1 99.0 99.1 99.1 99.<br>99.7 99.1 99.0 99.1 99.1 99.<br>99.4 98.8 98.9 98.9 98.9 99.<br>99.4 98.9 98.9 98.9 99.0 99.|iGSM-hard<br>2 99.6 98.8 98.5 98.5 98.7 98.7 98.7<br>1 99.6 98.9 98.6 98.5 98.7 98.6 98.7<br>1 99.7 99.3 99.2 99.1 99.2 99.3 99.3<br>0 99.7 99.3 99.3 99.2 99.3 99.2 99.3||iGSM-med<br>99.7 99.6 99.5 99.5 99.4 99.<br>99.7 99.6 99.5 99.5 99.5 99.<br>99.0 98.4 98.1 98.0 97.9 98.<br>99.0 98.4 98.0 98.0 98.0 98.|iGSM-hard<br>5 99.3 99.3 99.2 99.2 99.3 99.2 99.2<br>4 99.3 99.3 99.3 99.2 99.3 99.2 99.2<br>1 99.6 99.4 99.3 99.2 99.1 99.2 99.1<br>1 99.6 99.4 99.3 99.2 99.2 99.2 99.1||iGSM-med<br>99.9 99.5 99.2 99.2 99.2 99.<br>99.6 98.9 98.8 98.8 98.9 98.<br>99.7 99.3 98.9 98.9 98.8 98.<br>99.6 99.4 99.3 99.2 99.2 99.|iGSM-hard<br>1 99.7 99.3 99.1 99.0 99.0 99.1 98.9<br>9 98.9 97.9 97.7 97.7 97.8 97.9 97.9<br>7 99.9 99.8 99.6 99.5 99.5 99.5 99.4<br>3 99.7 99.2 99.1 99.1 99.1 99.2 99.2||iGSM-med<br>99.7 99.3 96.5 95.5 94.3 93.<br>90.7 97.8 97.1 97.0 96.2 95.<br>97.7 95.9 90.8 89.3 87.8 86.<br>90.3 95.1 92.9 92.5 91.0 90.|iGSM-hard<br>4 99.2 98.4 95.0 93.6 92.9 91.4 90.4<br>9 90.6 97.2 96.6 96.0 95.7 95.5 95.3<br>7 98.7 99.4 97.8 97.3 97.0 96.2 95.7<br>7 93.2 97.2 96.3 96.0 95.5 95.6 95.0|
|---|---|---|---|---|---|---|---|---|---|---|---|
||15<br>op=15<br>op=20<br>op=21<br>op=22<br>op=23<br>op<br>21<br>op=21<br>op=28<br>op=29<br>op=30<br>op=31<br>op=32<br>op|||15<br>op=15<br>op=20<br>op=21<br>op=22<br>op=23<br>op<br>21<br>op=21<br>op=28<br>op=29<br>op=30<br>op=31<br>op=32<br>op|||15<br>op=15<br>op=20<br>op=21<br>op=22<br>op=23<br>op<br>21<br>op=21<br>op=28<br>op=29<br>op=30<br>op=31<br>op=32<br>op|||||



Figure 12: Probing accuracies _restricted_ to positives/negatives labels (complement to Figure 7 which is on all labels.) 

## **— B Results 4-5 Details on V-probing** 

Recall that we wish to conduct probing at the end of the problem description for the `nece` and `dep` tasks (before the solution for `nece` ; before the solution or even the question for `dep` ). For other tasks, we probe at the end of _every_ solution sentence (including the start of the first solution sentence). The goal is to _freeze_ a pretrained language model, then introduce a _very small_ number of additional trainable parameters on top of it, and finetune them for each probing task. 

Specifically, we take a pretrained language model, e.g., pretrained from the iGSM-hard training data. We _freeze_ its parameters completely except for adding a trainable rank- _r_ update on the embedding layer to account for the task change (from next-token prediction to probing). Throughout this paper we use a small value _r_ = 8. We feed this network with training data that are the same as iGSM-hard, but truncated at exactly the position we wish to probe. Importantly, we append such inputs with a special starting token [START] along with a parameter name (or two names, if it is the `dep` ( _A, B_ ) task). We then extract the hidden states of the last token position at the last transformer layer, and add a trainable linear layer (a.k.a. linear head) to perform classification for one of the six probing tasks. 

This probing method is illustrated in Figure 13. We call it V(ariable)-Probing, because it can take an arbitrary number of variables (i.e., parameters in this paper) to allow us to perform functional probing inside the transformer. 

Note, if it were only a trainable linear head such probing would be called linear probing [12]. Unlike traditional linear probing, we are adding a small low-rank update on the model’s embedding layer. This is arguably the minimum change needed (to account for the task change, for special tokens like [START] [MID] [END], etc.) in order to perform any non-trivial probing. This is related but different from the nearly-linear probing methods introduced in Allen-Zhu and Li [1, 3], because they do not support taking variables as probing inputs.[21] 

**Unbalanced probing tasks.** Our probing accuracies for the six tasks were presented in Figure 7. However, we notice that the `dep` and `nece next` tasks have unbalanced labels — even guessing “all false” would give 83% accuracy for `dep` ( _A, B_ ) and 92% for `nece` ~~`n`~~ `ext` ( _A_ ). For such reason, we also present their probing accuracies restricted to positives/negatives labels separately in Figure 12. 

> 21In Allen-Zhu and Li [1, 3], the authors are interested in probing the model’s behavior via fixed classification tasks (such as a 100-class classification task) given data that are identical or nearly-identical to the pretrain data. In this paper, we are interested in the model’s behavior with respect to given variables (such as parameter names, which can have _∼_ 100 _k_ possibilities); and we append such variable names to the input to make the training inputs appear very different from the original pretrain data. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0017-00.png)


**----- Start of picture text -----**<br>
Predict  necc (A)  ∈  {True, False}<br>add a  trainable  linear<br>(classification) layer<br>decoder layer (attention + MLP)<br>decoder layer (attention + MLP) freeze  the pretrained LM<br>trainable  rank  𝑟  update<br>[BOS] The number of each Riverview High's Film Studio    ...     How many Backpack does Central High have?  [START]  Dance Studio’s School Daypack [END] on the embedding layer<br>problem question some  parameter A<br>(a) V-probing for the nece ( A ) task<br>Predict  dep (A,B)  ∈  {True, False}<br>decoder layer (attention + MLP)<br>decoder layer (attention + MLP)<br>[BOS] The number of each Riverview High's Film Studio ... each Film Studio's Messenger Backpack equals 13 . [START]  Riverview High’s Film Studio  [MID]  Dance Studio’s School Daypack [END]<br>problem some  parameter A some  parameter B<br>… … … … … …<br>… … … … … … …<br>**----- End of picture text -----**<br>


- (b) V-probing for the `dep` ( _A, B_ ) task 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0017-02.png)


**----- Start of picture text -----**<br>
Predict  known (A)  ∈  {True, False},   or  value (A)  ∈  {0,1,…22,None},   or  can_next (A)  ∈  {True, False},    or  necc_next (A)  ∈  {True, False}<br>decoder layer (attention + MLP)<br>decoder layer (attention + MLP)<br>[BOS] The number of each Riverview ...  How many Backpack …?  [SOL] Define … Define Central High’s Film Studio as B; so B = p + W = 17 + 13 = 7. [START]  Film Studio’s School Daypack  [END]<br>problem question a prefix of solution some  parameter A<br>…<br>**----- End of picture text -----**<br>


- (c) V-probing for the `value` ( _A_ ) _,_ `can next` ( _A_ ) _,_ `nece next` ( _A_ ) tasks 

Figure 13: Illustrations of V-probing, our nearly-linear probing methods to investigate whether a pretrained model, at a specific input position, _knows_ an arbitrary `func` ( _A_ ) for a parameter _A_ described in text. 

In all cases, we freeze the entire pretrained language model, except for a low-rank _r_ = 8 update on the input embedding layer to accommodate the task change. 

The illustration is for pq data (problem precedes question); for qp data, we simply reverse the order, except for `dep` ( _A, B_ ) where the question is added before the problem. 

16 

## **— C Result 8 Additional Figure** 

|0<br>50<br>100<br>depth12 - size1<br>0<br>50<br>100<br>depth12 - size2<br>0<br>50<br>100<br>depth20 - size1<br>0<br>50<br>100<br>depth20 - size2<br>0<br>50<br>100<br>depth4 - size1<br>0<br>50<br>100<br>depth4 - size2|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>med_pq - op15<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>med_pq - op20<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>med_qp - op15<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>med_qp - op20<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>hard_pq - op21<br><br>0<br>50<br>100<br> <br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>hard_pq - op28<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>hard_qp - op21<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br><br>0<br>50<br>100|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>hard_qp - op28<br>layer3<br>layer6<br>layer9<br>layer12<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>layer3<br>layer6<br>layer9<br>layer12<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>layer5<br>layer10<br>layer15<br>layer20<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>layer5<br>layer10<br>layer15<br>layer20<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>layer1<br>layer2<br>layer3<br>layer4<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>layer1<br>layer2<br>layer3<br>layer4|
|---|---|---|---|---|---|---|---|---|



Figure 14: Increasing probing accuracies of `nece` ( _A_ ) with increasing layer depth. This is an extension of Figure 10 but including more model depths/sizes. 

The x-axis denotes the distance of parameter _A_ from the query parameter, with colors transitioning from light to dark to represent layers 1 to max. (Model architecture details are in Footnote 20 and Appendix F.) 

17 

## **— D Result 1 Details Math Data Generation** 

Our math data generation process consists of first generating the structure graph (see Figure 1 and 11 left), which defines the set of parameters we shall use; then generating the dependency graph (see Figure 1 and 11 right), which defines the arithmetic relationship between the parameters; and finally generating the English problem and solution descriptions. 

**Notations.** In this section, to make the description concise, when we say “randomly sampling” in the pseudocode, we mean uniform random unless otherwise noted. Whenever we consider a (directed) graph _G_ , slightly abusing notation, we write _a ∈ G_ to indicate that _a_ is a vertex in _G_ and ( _a → b_ ) _∈_ G to indicate that there is an edge from _a_ to _b_ in _G_ . 

## **D.1 Generate Structure Graph** 

Recall the structure graph (see Figure 1 and 11 left) describes the set of possible items (nodes) and instance parameter (edges) that we shall rely on to construct our math problem. 

We use _G_ s to denote such structure graph, and it is generated _G_ s = `DrawStructure` ( _e, d, w_ 0 _, w_ 1) from a random distribution defined with _hyperparameters e, d, w_ 0 _, w_ 1 _∈_ N. At a high level, we construct _G_ s so that it has _d_ layers, _e_ edges, and each layer has between _w_ 0 and _w_ 1 items. 

Specifically, suppose _li ∈{w_ 0 _, w_ 0 + 1 _, . . . , w_ 1 _}_ represents the number of items for each layer _i_ . In this configuration, one must have at least _e[−]_ = _l_ 2 + _· · ·_ + _ld_ edges to ensure the graph is “connected”, and at most _e_[+] = _l_ 1 _l_ 2 + _· · ·_ + _ld−_ 1 _ld_ edges. Using this formula, we first randomly choose a configuration ( _l_ 1 _, . . . , ld_ ) so that _e[−] ≤ e ≤ e_[+] for the given parameter _e_ . Then, after the configuration is chosen, we randomly generate edges accordingly. Details are given in Algorithm 1. 

**Algorithm 1** _G_ s = `DrawStructure` ( _e, d, w_ 0 _, w_ 1) 

**Input:** _e, d, w_ 0 _, w_ 1 _∈_ N _⋄ satisfying_ 2 _≤ d ≤_ 4 _;_ 2 _≤ w_ 0 _≤ w_ 1 _≤_ 4 _;_ ( _d −_ 1) _w_ 0 _≤ e ≤_ ( _d −_ 1) _w_ 1[2] 1: _l ←_ ( _w_ 0 _, w_ 0 _, . . . , w_ 0) _∈_ Z _[d] ⋄ li represents the number of items (nodes) for layer i_ 2: _p ←_ uniform random from (0 _,_ 1) 3: **while** _l ̸_ = ( _w_ 1 _, w_ 1 _, . . . , w_ 1) **do** 4: _e[−] , e_[+] _←_ minimum and maximum number of edges that _l_ can give 5: **if** _e_[+] _< e_ **then** 6: randomly select _i ∈_ [ _d_ ] such that _li < w_ 1, and increase it _li ← li_ + 1. 7: **else if** _e[−]_ = _e_ **then** 8: **break** 9: **else if** randomly choose a number in (0 _,_ 1) and it is less than _p_ **then** 10: randomly select _i ∈_ [ _d_ ] such that _li < w_ 1, and increase it _li ← li_ + 1. 11: **else** 12: **break** 13: **end** _⋄ after while loop, we must have e[−] ≤ e ≤ e_[+] _and ∀i ∈_ [ _d_ ]: _w_ 0 _≤ li ≤ w_ 1 14: Construct _G_ s with exactly _li_ items on layer _i ∈_ [ _d_ ]. 15: **for** each item _a_ in each layer _i ≥_ 2 **do** 16: randomly select an item _b_ in layer _i −_ 1 and connect ( _a, b_ ) in _G_ s. _⋄ this creates e[−] edges_ 17: **while** number of edges _< e_ **do** 

18: randomly select two items _a, b_ from adjacent layers to create an edge in _G_ s. 19: **return** _G_ s and attach English to it. 

18 

## **D.1.1 Attach English** 

As described in Section 2.1, we have prepared 4 predefined hierarchical categorizations, each of them with 4 total layers of categories: 

```
[
["District","Supermarket","Product","Ingredient"],
["Zoo","Enclosure","Animal","Bone"],
["School","Classroom","Backpack","Stationery"],
["Ecosystems","Creatures","Organs","Cells"]
]
```

In each of the above 16 categories, we have prepared around 100 items (further decomposed into 5 sub-categories). Below is a showcase of them: 

```
{
"District":{
"ResidentialDistricts":[...],
"CommercialDistricts":[
"ShoppingDistrict","BusinessDistrict","FinancialDistrict","
IndustrialDistrict",
"WarehouseDistrict","MarketDistrict","RestaurantDistrict","
EntertainmentDistrict",
"ArtsDistrict","FashionDistrict","SiliconValley","WallStreet",
"TechPark","AutomotiveDistrict","JewelryDistrict","Medical
District",
"LegalDistrict","MediaDistrict","ResearchPark"," Manufacturing
District"
],
"HistoricalDistricts":[...],
"EducationalDistricts":[...],
"GovernmentDistricts":[...]
},
"Supermarket":{...},
"Product":{
"CannedFoods":[...],
"SnackFoods":[
"PotatoChips","Pretzels","Popcorn","CandyBars",
"GummyCandy","Cookies","Crackers","GranolaBars",
"FruitSnacks","CheesePuffs","Nuts","TrailMix",
"BeefJerky","RiceCakes","YogurtCoveredRaisins","Chocolate
CoveredPretzels",
"TortillaChips","Salsa","Hummus","DriedFruit"
],
"Beverages":[...],
"BakedGoods":[...],
"DairyProducts":[...]
},
"Ingredient":{...},
"Zoo":{...},
"Enclosure":{...},
"Animal":{...},
"Bone":{...},
"School":{...},
"Classroom":{...},
"Backpack":{...},
"Stationery":{...},
"Ecosystems":{...},
"Creatures":{...},
```

19 

```
"Organs":{...},
"Cells":{...}
}
```

Now, given a constructed structure graph _G_ s, we first randomly pick one of the four categorizations, then randomly pick _d ∈{_ 2 _,_ 3 _,_ 4 _}_ consecutive layers of categories, next randomly pick one of the five subcategories, and finally pick _li_ random item names in this subcategory for each layer _i_ . 

At this point, we have constructed _G_ s as well as added English names to each of its node, just like Figure 1 and 11 (left). 

## **D.2 Generate Dependency Graph** 

A structure graph _G_ s defines the set of possible parameters we consider, while a _dependency graph_ defines how these parameters depend on each other. We use an edge _a → b_ to indicate that parameter _b_ depends on _a_ ; there is a special vertex RNG and it can happen that RNG _→ b_ . What an abstract parameter depends on is inherited from the structure graph _G_ s. For each instance parameter, we shall randomly add edges to indicate what parameters it depends on. 

**High-level plan.** We shall use _G_ d to denote the dependency graph, we start from an empty graph and then add vertices/edges incrementally and randomly. Our process is as follows: 

- Generate a _necessary dependency graph G_[nece] d which covers all the vertices and nodes that are necessary for the computation of the query parameter. 

   - Generate necessary abstract parameters (and add parameters they depend on); call this graph _G_[nece1] d . 

   - Generate necessary instance parameters and add them to _G_[nece1] d ; call this graph _G_[nece2] d . 

   - Generate a topological order for parameters _G_[nece2] d and ensure all of them are necessary towards computing the query parameter (which is the last one in this tropologic order). During this process, we shall add additional edges from _G_[nece2] d to create _G_[nece3] d . 

   - **–** Generate additional necessary edges and add them to _G_[nece3] d ; call this graph _G_[nece] d . 

- Add to _G_[nece] d all the remaining (unnecessary) parameters and edges to form _G_ d. 

At a high level, our problem description shall solely depend on _G_ d— by describing each instance parameter in it using a sentence, and our solution description shall solely depend on _G_[nece] d — by describing the computation of each parameter in it using a sentence. 

Before we proceed with the construction let us formally introduce: 

**Definition D.1** (operation) **.** _Given any dependency graph G_ d _,_ 

- def 

- • _For an (abstract or instance) parameter a ∈ G_ d _that has in-degree t ≥_ 0 _, we define_ op _G_ d( _a_ ) = max _{_ 1 _, t −_ 1 _} which is the number of_ operations _needed to compute a._[22] 

- def 

- • _We use_ op( _G_ d) =[�] _a∈G_ d _\{_ RNG _}_[op] _G_ d[(] _[a]_[)] _[to][denote][the][total][number][of][(arithmetic)][operations] needed to compute all the parameters in G_ d _._ 

_Remark_ D.2 _._ In our final design of _G_ d, we shall ensure that each parameter (except the special vertex RNG) has in-degree at least 1; however, during the construction process since we add edges 

> 22For instance, in Figure 1, _a_ = “Riverview High’s total number of Backpacks” is equal to _ip_ 1 _× ap_ 1 + _ip_ 2 _× ap_ 2 for _ip_ 1 = “Riverview High’s number of Dance Studios”, _ip_ 2 = “Riverview High’s number of Film Studios”, _ap_ 1 = “each Dance Studio’s number of Backpacks”, _ap_ 2 = “each Film Studio’ number of Backpacks”, where _ip_ 1 _, ip_ 2 are instance parameters and _ap_ 1 _, ap_ 2 are abstract parameters. In this case, this abstract parameter depends on 4 other parameters, and requires 3 arithmetic operations. 

20 

incrementally, some (instance) parameter may temporarily have in-degree 0. For notation simplicity, we still say op _G_ d( _a_ ) = max _{_ 1 _, −_ 1 _}_ = 1 in such a case. 

**Hyperparameters.** We use hyperparameters 1 _≤ n ≤ m ≤ s_ to control the difficulty of _G_ d. 

- we shall ensure op( _G_[nece1] d ) _≤ n_ and is as close as possible to _n_ ; 

- we shall ensure op( _G_[nece3] d ) = op( _G_[nece2] d ) _≤ m_ and is as close as possible to _m_ ; 

- • we shall ensure op( _G_[nece] d ) = _s_ is exact. 

In other words, hyperparameter _s_ controls exactly how many operations are needed to compute the query parameter, which is the primary factor controlling the problem’s difficulty. 

## **D.2.1 Construction of** _G_[nece1] d _, G_[nece2] d 

Given a structure graph _G_ s, recall its edges represent all the instance parameters we shall use. Its abstract parameters are those ones that describe quantities across 1 or multiple layers: for instance in Figure 1, Central High’s number of Classrooms is across 1 layer, and Central High’s number of Backpacks is across 2 layers. We define this number as the _difficulty level_ of abstract parameters. 

With this notion, our construction of _G_[nece1] d and _G_[nece2] d are described together in Algorithm 2. At a high level, we try to incrementally and randomly add abstract parameters to _G_[nece1] d while maintaining op( _G_[nece1] d ) _≤ n_ . We cannot make this exact equality because when adding a single abstract parameter requires also (recursively) adding all the other parameters it may depend on. We tried to prioritize adding abstract parameters with higher difficulty levels. Once we finish constructing _G_[nece1] d , we randomly add additional instance parameters from _G_ s to make it _G_[nece2] d . 

**Algorithm 2** _G_[nece2] d = `DrawNecessary1` ( _G_ s _, n, m_ ) 

**Input:** structure graph _G_ s of depth _d_ , _n, m ∈_ N with 1 _≤ n ≤ m_ 

> 1: _G_[nece1] d _←_ empty graph 

## 2: **repeat** 

3: `updated` _←_ false 4: **for** _i ← d −_ 1 _, . . . ,_ 1 **do** 5: **if** _∃_ abstract parameter of difficulty level _i_ in _G_ s that is not yet in _G_[nece1] d **then** 6: randomly pick one such abstract parameter _a_ of difficulty level _i_ 7: _G[′] ← G_[nece1] d + _a_ and all instance/abstract parameters _a_ may (recursively) depend on _⋄ also add their dependency edges_ 8: **if** op( _G[′]_ ) _≤ n_ **then** 9: _G_[nece1] d _← G[′]_ ; `updated` _←_ true; **break** 

10: **until** `updated` = false 11: _G_[nece2] d _← G_[nece1] d _⋄_ op( _G_[nece1] d ) _≤ n and all instance parameters in G_[nece1] d _have in-degree 0_ 12: **for** _i ←_ 1 _,_ 2 _, . . . , m −_ op( _G_[nece1] d ) **do** 13: if there’s leftover instance parameter in _G_ s not yet in _G_[nece2] d , add a random one to _G_[nece2] d 14: **return** _G_[nece2] d _⋄_ op( _G_[nece2] d ) _≤ m and all instance parameters in G_[nece2] d _have in-degree 0_ 

## **D.2.2 Construction of** _G_[nece3] d 

Our goal next is to select a random `query` parameter in _G_[nece2] d and construct a random topological ordering `Topo` for all the parameters in _G_[nece2] d , so as to ensure that all the parameters are necessary towards the computation of `query` . 

21 

We start with `Topo` = [ `query` ] and append parameters to its left one by one. During this process, we may also introduce new edges randomly; we start with _G_[nece3] d = _G_[nece2] d and add edges incrementally. This process may not always succeed — sometimes the created topological ordering cannot make all the parameters necessary towards the computation of the `query` . If this happens we declare a failure.[23] We introduce two notions (we use _G_[nece3] d _\_ `Topo` to denote the set of vertices in _G_[nece3] d that are not in `Topo` ): 

- def 

- • Next1 _G_ nece3d ( `Topo` ) = � _a ∈ G_[nece3] d _\_ `Topo` _| ∃_ ( _a → b_ ) _∈ G_[nece3] d for some _b ∈_ `Topo` � 

Intuitively, if _a ̸∈_ Next1( `Topo` ) then we cannot immediately append _a_ to the front of `Topo` , because it is not yet necessary towards the computation of `query` . 

- def 

- • Next2 _G_ nece3d ( `Topo` ) = � _a ∈ G_[nece3] d _\_ `Topo` _|_ ∄( _a → b_ ) _∈ G_[nece3] d for any _b ∈ G_[nece3] d _\_ `Topo` � 

Intuitively, if _a ̸∈_ Next2 _G_ nece3d ( `Topo` ) then we cannot immediately append _a_ to the front of `Topo` , because some other parameter depends on it and is not yet added to `Topo` . (Obviously we always have Next2 _G_ nece3d ( `Topo` ) _̸_ = ∅ unless _G_[nece3] d _\_ `Topo` = ∅ so we are done.) Our generation algorithm is now easy to describe: we keep adding parameters that are in Next1 _G_ nece3d ( `Topo` ) _∩_ Next2 _G_ nece3d ( `Topo` ) to the front of `Topo` ; and if we get stuck, we introduce new edges to _G_[nece3] d (or declare failure). The pseudocode is in Algorithm 3. 

**Algorithm 3** ( _G_[nece3] d _,_ `Topo` ) = `DrawNecessary2` ( _G_[nece2] d ) 1: _G_[nece3] d _← G_[nece2] d ; `Topo` _←_ []. 2: **while** true **do** 3: **if** `Topo` = [] **then** 4: `param` 0 _←_ random parameter in Next2 _G_ nece3d ( `Topo` ); _⋄ this is_ `query` _parameter_ 5: **else** 6: `param` 0 _←_ random parameter in Next1 _G_ nece3d ( `Topo` ) _∩_ Next2 _G_ nece3d ( `Topo` ); 7: `Topo` = [ `param` 0] + `Topo` _⋄ append to the front_ 8: **if** _G_[nece3] d _\_ `Topo` = ∅ **then break** 9: **if** Next1 _G_ nece3d ( `Topo` ) _∩_ Next2 _G_ nece3d ( `Topo` ) = ∅ **then** 10: **If** `param` 0 is abstract **then return** failure 11: `param` 1 _←_ a “random” parameter in Next2 _G_ nece3d ( `Topo` ). _⋄ see Remark D.4_ 12: add edge `param` 1 _→_ `param` 0 to _G_[nece3] d . _⋄ now_ `param` 1 _∈_ Next1 _G_ nece3d ( `Topo` ) 13: **else if** `param` 0 is instance parameter **then** 14: **if** a probability event _p_ 0 occurs for _p_ 0 uniform chosen in (0 _,_ 1) **then** 15: `param` 1 _←_ a “random” parameter in _G_[nece3] d _\_ `Topo` . _⋄ see Remark D.4_ 16: add edge `param` 1 _→_ `param` 0 to _G_[nece3] d . _⋄ now_ `param` 1 _∈_ Next1 _G_ nece3d ( `Topo` ) 17: **return** ( _G_[nece3] d _,_ `Topo` ) _⋄_ op( _G_[nece3] d ) _≤ m and all instance parameters in G_[nece3] d _have in-degree ≤_ 1 

**Proposition D.3.** _Every instance parameter in G_[nece3] d _has in-degree ≤_ 1 _and thus_ op( _G_[nece3] d ) = op( _G_[nece2] d ) _._ 

_Remark_ D.4 _._ In Line 11 and Line 15 of Algorithm 3, when randomly selecting `param` 1 from a set, instead of doing so uniformly at random, to improve the algorithm’s success rate and the problem’s difficulty level, we introduce a discursion that that biases slightly towards abstract parameters and 

> 23The outside pseudocode, which comes later, shall go back to regenerate the structure graph and start again. 

22 

parameters already in Next1 _G_ nece3d ( `Topo` ).[24] Specifically, we first generate _g ∼N_ (0 _,_ 1) a random Gaussian, then define `weight` ( _a_ ) = �1 _a_ is abstract + 1 _a∈_ Next1 _G_ nece3 ( `Topo` )� _· |g|_ , and then sample _a_ with d a probability _∝ e_ `[weight]`[(] _[a]_[)] . 

## **D.2.3 Construction of** _G_[nece] 

## d 

So far we have created _G_[nece3] d and `Topo` with the property that every instance parameter in _G_[nece3] d has in-degree _≤_ 1. In the next step, we add additional dependency edges to make in-degree to be a random number between 1 and 4. We do so by introducing additional edges; and we also introduce an additional vertex RNG. This is our final _necessary_ dependency graph _G_[nece] d . 

Our pseudocode is given in Algorithm 4. In this step, we shall make sure op( _G_[nece] d ) = _s_ is exact (and declare failure if this is not possible). We do so to precisely control the solution’s difficulty (so that when we evaluate the model, we can choose to evaluate it on problems with a fixed value of _s_ ). 

**Algorithm 4** _G_[nece] d = `DrawNecessary3` ( _G_[nece3] d _,_ `Topo` _, s_ ) 1: `cur` ~~`o`~~ `p` ( _a_ ) _←_ op _G_ nece3d ( _a_ ) for every parameter _a ∈ G_[nece3] d . def 2: `max` ~~`o`~~ `pTopo` ( _a_ ) = the maximum number of operations an instance parameter _a_ can require.[25] 3: **while**[�] _a∈G_[nece3] d `cur` ~~`o`~~ `p` ( _a_ ) _< s_ **do** 4: randomly select an instance parameter _a ∈ G_[nece3] d with `cur` ~~`o`~~ `p` ( _a_ ) _<_ `max` ~~`o`~~ `pTopo` ( _a_ ); 5: **If** _a_ is found **then** `cur` ~~`o`~~ `p` ( _a_ ) _←_ `cur` ~~`o`~~ `p` ( _a_ ) + 1 **else return** failure. 6: _G_[nece] d _← G_[nece3] d + vertex RNG. 7: **for** each instance parameter _a_ in _G_[nece3] d **do** 8: `pool` _←_ RNG + all parameters in front of _a_ in `Topo` . 9: **if** `cur` ~~`o`~~ `p` ( _a_ ) = 1 **then** 10: `dep num` _←_ 1 or 2 each w.p. 0.5; 11: **else** 12: `dep num` _←_ `cur` ~~`o`~~ `p` ( _a_ ) + 1 13: `dep` ~~`n`~~ `um` _←_ min _{|_ `pool` _|,_ `dep` ~~`n`~~ `um` _}_ 14: **if** _∃_ ( _b → a_ ) _∈ G_[nece3] d for some _b ∈_ `pool` **then** _⋄ at most one such b_ 15: `pool` _←_ `pool` _\ {b}_ and `dep` ~~`n`~~ `um` _←_ `dep` ~~`n`~~ `um` _−_ 1 16: **if** `dep` ~~`n`~~ `um` = _|_ `pool` _|_ **then** 17: add _b → a_ to _G_[nece] d for all _b ∈_ `pool` ; 18: **else** 19: with probability 0.5, add RNG _→ a_ to _G_[nece] d and `dep` ~~`n`~~ `um` _←_ `dep` ~~`n`~~ `um` _−_ 1 20: `pool` _←_ `pool` _\ {_ RNG _}_ 21: add _b → a_ to _G_[nece] d for `dep` ~~`n`~~ `um` randomly select elements _b_ in `pool` . 22: **return** _G_[nece] d _⋄_ op( _G_[nece] d ) = _s is exact_ 

> 24For those who are interested, abstract parameters are the keys to cause the generation process to fail, because once they become `param` 0 we cannot add edges `param` 1 _→_ `param` 0; so we had better select them earlier than later (thus put them at the back of `Topo` ). On the other hand, for `param` 1 that is already in Next1 _G_ nece3d ( `Topo` ), adding this edge `param` 1 _→_ `param` 0 does not further change it; this can help us create a problem whose solution “depth” is higher. 

> 25If an instance parameter _a_ is the _i_ -th element in `Topo` , then `max` ~~`o`~~ `p` ( _a_ ) = min _{_ 3 _,_ max _{_ 1 _, i −_ 1 _}}_ . (Recall we require each instance parameter to depend on at most 4 vertices in the dependency graph and this amounts to no more than 3 operations.) 

23 

## **D.2.4 Construction of** _G_ d 

Finally, once we have _G_[nece] d the necessary dependency graph, we are left to add unnecessary dependency edges (and unnecessary parameters) to form the complete _G_ d. 

During this process, we shall add all the _remaining_ instance parameters from _G_ s into _G_ d. When adding each of them, we randomly select the parameters that it shall depend on from all the previously known parameters.[26] Note that during this process, we may also introduce new, unnecessary abstract parameters, see the full pseudocode in Algorithm 5. 

_Remark_ D.5 _. G_ d consists of all the instance and query parameters in _G_ s and the abstract parameters they may (recursively) depend on. There may exist abstract parameters that can be described in _G_ s that are not present in _G_ d; but all the instance parameters in _G_ s shall be present in _G_ d. 

**Algorithm 5** _G_ d = `DrawUnnecessary` ( _G_ s _, G_[nece] d ) 1: `IndList` _←_ ∅; 2: **while** _∃_ instance parameter in _G_ s not yet in _G_ d **do** 3: _K ←_ all params in _G_ d + all abstract params computable using parameters in _G_ d; 4: randomly select an instance parameter _a_ in _G_ s not yet in _G_ d; and add _a_ to _G_ d; 5: **if** with half probability **then** 6: `pool` _←_ `IndList` _∪{_ RNG _}_ ; `IndList` _←_ `IndList` _∪{a}_ ; 7: **else** 8: `pool` _← K ∪{_ RNG _}_ ; 9: `dep` ~~`n`~~ `um` _←_ 1 10: **while** `dep num` _<_ min _{_ 4 _, |_ `pool` _|}_ **do** 11: with 0.5 probability, `dep` ~~`n`~~ `um` _←_ `dep` ~~`n`~~ `um` + 1; otherwise **break** 12: **if** `dep` ~~`n`~~ `um` = _|_ `pool` _|_ **then** 13: `selected` _←_ `pool` 14: **else** 15: `selected` _←{}_ 16: with probability 0.5, add `selected` = _{_ RNG _}_ and `dep` ~~`n`~~ `um` _←_ `dep` ~~`n`~~ `um` _−_ 1 17: `pool` _←_ `pool` _\ {_ RNG _}_ 18: `selected` _←_ `selected` _∪_ `dep` ~~`n`~~ `um` random elements from `pool` 19: **for** each _b ∈_ `selected` **do** 20: If _b ̸∈ G_ d then recursively add _b_ and its dependencies to _G_ d; 21: Add _b → a_ to _G_ d. 22: **return** _G_ d 

## **D.3 Generate English: Problem, Question and Solution** 

At this point, we have constructed a dependency graph _G_ s where each instance parameter _a ∈ G_ s may depend on between 1 and 4 other vertices (could be abstract, instance parameters or RNG). We have not yet introduced how _a_ should be computed, and we do this using a random process `GenSentence` ( _G_ d _, a_ ) in Algorithm 6. 

> 26In fact, we do slightly smarter than the most naive approach. If one simply lets each newly added unnecessary parameter to depend, randomly among all the parameters that have already been added to _G_ d, then those unnecessary parameters will likely appear towards the end of the topological order. For such reason, we give it 0.5 probability to depend only on a set `IndList` , which consists of newly-added, unnecessary parmaeters, that do not depend on _G_ d. This way, the unnecessary parameters can also appear to the front of the tropologic order. 

24 

**Algorithm 6** `GenSentence` ( _G_ d _, a_ ) 

- 1: _str ←_ “The number of [name of _a_ ] equals” 

- 2: `pool` _←{b ∈ G_ d : _∃_ ( _b → a_ ) _∈ G_ d _}_ . 

- 3: **if** RNG _∈_ `pool` **then** 

- 4: _str ← str_ + “ [random int between 0 and 22]”; and `pool` _←_ `pool` _\ {_ RNG _}_ 

- 5: If _|_ `pool` _| >_ 0, _str ← str_ + “ more than” or “ times” each with probability 0.5. 

- 6: **if** _|_ `pool` _|_ = 1 **then** 

- 7: _str ← str_ + “ [name of _b_ ]” for `pool` = _{b}_ . 8: **else if** _|_ `pool` _|_ = _|{b, c}|_ = 2 **then** 

- 9: _str ← str_ + “ the sum of [b] and [c]” or “ the difference of [b] and [c]” each w.p. 0.5. 

- 10: **else** 

- 11: _str ← str_ + “ the sum of .., .., and ..” with a random order of all elements from `pool` . 

**Problem description.** The problem description simply consists of listing over all _instance_ parameters _a ∈ G_ d and call `GenSentence` ( _G_ d _, a_ ). We then randomly shuffle the sentences to make the problem hard. Please note the descriptions of abstract parameters are _not present_ in the problem description, because they are inherited from the hierarchical categorization. This is our attempt to make our math data also capture some English meaning, that is the model also needs to learn what items are in each category, and which category is above another category, etc. This is some knowledge that cannot be learned by reading one problem — it must be learned after reading sufficiently many data. 

**Question description.** Our query parameter can be either an instance or abstract parameter, and it is the last element in `Topo` . We use a single sentence to ask for its value “How many... does... have?” and we put this question either at the front or at the end of the problem description (depending on the data type). 

**Solution description.** We generate the solution text, by going over all the (instance or abstract) parameters in `Topo` _in its correct order_ , and generate a single sentence to compute each parameter. This process is straightforward but notationally heavy, we describe it below by examples. 

- Given any instance parameter _a ∈_ `Topo` , suppose for instance _a_ is 7 times the sum of parameters _b, c, d_ . Because of the topological order, the parameters _b, c, d_ must have already defined with variable names, denoted as _varb, varc, vard_ . Then we define solution string of _a_ as 

   - “ **Define** [name of a] **as** _var_ 0; _var_ 1 = _varb_ + _varc_ = _· · ·_ ; _var_ 2 = _var_ 1 + _vard_ = _· · ·_ ; 

**so** _var_ 0 = 7 _× var_ 2 = _· · ·_ .” 

Here, the arithmetic computation is decomposed into 2-ary operations step by step separated with semicolons (so op _G_ d( _a_ ) is exactly the number of semicolons). The _var_ 0 _, var_ 1 _, var_ 2 are three new (but distinct) random variables and their names are between a-z or A-Z and have “ 52 possible random choices. The _· · ·_ ” ignores the math calculations. 

- Given an abstract parameter _a ∈_ `Topo` , suppose for instance _a_ = _b × c_ + _d × e_ + _f × g_ then we similarly define its solution text as 

   - “ **Define** [name of a] **as** _var_ 0; _var_ 1 = _varb × varc_ = _· · ·_ ; _var_ 2 = _vard × vare_ = _· · ·_ ; 

“ _var_ 3 = _varf × varg_ = _· · ·_ ; _var_ 4 = _var_ 1 + _var_ 2 = _· · ·_ ; **so** _var_ 0 = _var_ 3 + _var_ 4 = _· · ·_ .” Above, once again _var_ 0 _, var_ 1 _, var_ 2 _, var_ 3 _, var_ 4 are new (but distinct) random variable names from a-z or A-Z, and we break down the computation into 2-ary operations. 

25 

With the above examples in mind, and combining those with real examples in Figure 11, it should be very clear how the solution texts are generated. 

_Remark_ D.6 _._ op( _G_[nece] d ) is equal to the total number of semicolons in the solution text, because it represents the total (and minimum!) number of arithmetic operations needed to compute the final query parameter. 

## **D.4 Putting Altogether** 

We put together our data generation process for the structure graph _G_ s and the dependency graph _G_ d (along with _G_[nece] d _,_ `Topo` ) in Algorithm 7. 

In particular, we use global parameters ipmax and opmax: the former controls the maximum number of instance parameters, and the latter controls the maximum number of solution operations. We select _n, m, s_ based on opmax (to ensure that 1 _≤ n ≤ m ≤ s ≤_ opmax), and _d, e, w_ 0 _, w_ 1 based on ipmax and _s_ . We also provide a boolean switch _force_ and when force = true, we shall force _s_ = opmax so that the generated math problem will have its solution to be of exactly opmax operations. We define datasets 

   - iGSM[op] _[≤]_[op][max] _[,]_[ip] _[≤]_[ip][max] as the process of invoking `DrawAll` (opmax _,_ ipmax _,_ force = false). 

   - iGSM[op][=][op][max] _[,]_[ip] _[≤]_[ip][max] as the process of invoking `DrawAll` (opmax _,_ ipmax _,_ force = true). 

- Using this language: 

   - The training data iGSM-med is iGSM[op] _[≤]_[15] _[,]_[ip] _[≤]_[20] ; 

   - The eval data of iGSM-med additionally includes iGSM[op][=] _[op,]_[ip] _[≤]_[20] for _op ∈{_ 15 _,_ 20 _,_ 21 _,_ 22 _,_ 23 _}_ ; 

   - The training data iGSM-hard is iGSM[op] _[≤]_[21] _[,]_[ip] _[≤]_[28] ; 

   - The eval data of iGSM-hard additionally includes iGSM[op][=] _[op,]_[ip] _[≤]_[28] for _op ∈{_ 21 _,_ 28 _,_ 29 _,_ 30 _,_ 31 _,_ 32 _}_ . 

_Remark_ D.7 _._ During training (regardless of pretrain or finetune for probing tasks), we only use those data whose hash value of their solution template (see Footnote 11) is _<_ 17 (mod 23), and during evaluation we only use those whose hash value is _≥_ 17 (mod 23). This ensures a strict separation between train and test data (even in terms of their solution templates). 

_Remark_ D.8 _._ In Algorithm 7, we chose _s_ = min _{t_ 0 _, t_ 1 _}_ , where _t_ 0 and _t_ 1 are two random integers between 1 and opmax. This choice encourages more easier math problems in the pretrain data, which we found improves the model’s learning. 

26 

**Algorithm 7** `DrawAll` (opmax _,_ ipmax _, f orce_ ) generation 

- 1: _s ←_ min _{t_ 0 _, t_ 1 _}_ for _t_ 0 _, t_ 1 being two random integers from 1 and opmax 

- 2: If force = true then _s ←_ opmax. 

- 3: _n ←_ max _{t_ 0 _, t_ 1 _}_ for _t_ 0 _, t_ 1 being two random integers from 1 and _s_ 

- 4: _m ←_ random integer between _n_ and _s_ 

- 5: _d ←_ a random choice among _{_ 2 _,_ 3 _,_ 4 _}_ with distribution according to softmax( `weight` ) 

_⋄ for_ `weight` = [ _−_ ( _rel −_ 0 _._ 2)[2] _, −_ ( _rel −_ 0 _._ 5)[2] _, −_ ( _rel −_ 0 _._ 8)[2] ] _for rel_ = _ip_ max _s−_ 1 _−_ 1 

6: _t_ 0 _, t_ 1 _←_ two random choices among _{_ 2 _,_ 3 _,_ 4 _}_ with distribution according to softmax( `weight` ) 

7: _w_ 0 _←_ min _{t_ 0 _, t_ 1 _}_ and _w_ 1 _←_ max _{t_ 0 _, t_ 1 _}_ . 

8: _e ←_ min _{t_ 0 _, t_ 1 _,_ ( _d −_ 1) _w_ 1[2] _[}]_[for] _[t]_[0] _[, t]_[1][being][random][integers][between][(] _[d][ −]_[1)] _[w]_[0][and][ip] max 

9: _G_ s _←_ `DrawStructure` ( _e, d, w_ 0 _, w_ 1) 

10: _G_[nece2] d _←_ `DrawNecessary1` ( _G_ s _, n, m_ ) 11: ( _G_[nece3] d _,_ `Topo` ) _←_ `DrawNecessary2` ( _G_[nece2] d ) _⋄ if fail, go to Line 9; if fail for 1000 times, go to Line 1_ 12: _G_[nece] d _←_ `DrawNecessary3` ( _G_[nece3] d _,_ `Topo` _, s_ ) _⋄ if fail, go to Line 1_ 13: _G_ d _←_ `DrawUnnecessary` ( _G_ s _, G_[nece] d ) 14: **return** ( _G_ d _, G_[nece] d _,_ `Topo` ) _⋄ and generate English descriptions following Section D.3_ 

## **E Data Details: Probing Data Preparation** 

We describe here how we prepare the probing data. We generate math data according to Appendix D. For each problem and each probing task (such as `nece` ( _A_ ), `dep` ( _A, B_ ), etc), we need to specify two things: at which position to probe and what parameters _A_ (or _A, B_ ) to probe. 

- For `nece` and `dep` , the probing always takes place at the end of the problem (and question) description, so there is no choice to be made; for `value` , `can next` , `nece next` tasks, the probing can take place at the end of _each_ sentence in the solution for (including the beginning of the first solution sentence), and we uniformly at random make such choices. 

- Each parameter _A_ (or _B_ ) can be uniformly at random chosen from the set of all (instance or abstract) parameters in our dependency graph _G_ d (with the only requirement that _A ̸_ = _B_ ). 

In the end, we make sure for each problem and each probing task, we make at most 10 such random choices (over the position and the choice of parameters) and sample without replacement. 

Just like in the pretrain data, we prepare our probing data so that only problems with hash values of their solution template (see Footnote 11) where the hash _<_ 17 (mod 23) are included in the training set, and the rest are used for testing. 

## **F Experiment Details** 

**Model.** We use the GPT2 architecture [18], replacing its absolute positional embedding with modern rotary positional embedding [7, 20], still referred to as GPT2 for short. (We also played with the Llama architecture (especially with gated MLP layers) aand did not see any benefit of using it. This GPT2 performs comparably to Llama/Mistral at least for knowledge tasks [4].) 

Let GPT2- _ℓ_ - _h_ denote an _ℓ_ -layer, _h_ -head, 64 _h_ -dim GPT2 model. We primarily use GPT2-12-12 (a.k.a. GPT2-small) in this paper, but in Section 6 we explore larger models with different widths and depths. Our size-1 models are GPT2-4-21, GPT2-8-15, GPT2-12-12, GPT2-16-10, GPT2-20-9, roughly the same size as GPT2-small. Our size-2 models are GPT2-4-30, GPT2-8-21, GPT2-1217, GPT2-16-15, GPT2-20-13, roughly twice the size of GPT2-small. We use a context length of 

27 

768/1024 for language model pretraining on iGSM-med _/_ iGSM-hard and a context length of 2048 for evaluation. 

**Data size.** For both pretraining and finetuning, we did not limit the amount of training data; we generated new data on-the-fly. We do not explore sample complexity in this paper, such as the number of math problems needed to achieve a certain level of accuracy, as it would complicate the main message of this paper. 

## **F.1 Pretrain Experiment Details** 

**Pretrain parameters.** We used the AdamW optimizer with mixed-precision fp16, _β_ = (0 _._ 9 _,_ 0 _._ 98), cosine learning rate decay (down to 0.01x of peak learning rate in the end), and 1000 steps of linear ramp-up. We used a mixture of V100/A100 GPUs, but the GPU specifications are not relevant here.[27] For all of our pretrain experiments: 

- On the iGSM-med datasets, we used a (peak) learning rate 0 _._ 002, weight decay of 0 _._ 05, batch size of 512, context length of 768, and trained for 100 _,_ 000 steps. 

- On the iGSM-hard datasets, we used a (peak) learning rate 0 _._ 002, weight decay of 0 _._ 03, batch size of 256, context length of 1024, and trained for 200 _,_ 000 steps. 

Our pretrain data is constructed by randomly generating math problems (and solutions), concatenating them together, and truncating them (in the right) to fit within the 768 or 1024-sized context window. If a problem is longer than the context window size, we discard it (this happens very rarely). 

**Test-time parameters.** When evaluating on test data, we use context length 2048 for both iGSM-med and iGSM-hard. We use either beam=1 and dosample=False (greedy) or beam=4 and dosample=True (beam-search multinomial sampling) to present test accuracies. We discover it is better to keep dosample=False while beam=1 and dosample=True while beam=4. We also tried larger beam sizes and found no further improvements. 

**Accuracy statistics.** Our main accuracies are presented in Figure 3, where each entry is averaged over 4096 math problems of that type. Our accuracies _are not simply_ from comparing the answer integers (between 0 and 22); instead we have written a parser to make sure the model’s intermediate solution steps are fully-correct. 

For the “redundancy” experiment Figure 4, we tested each model again with 4096 math problems in each case and presented the results among fully-correct solutions. For this figure, we present beam=1 for cleanness and the results for beam=4 are almost completely identical. 

For the “depth matters” experiment Figure 9, because we care about the (relatively small) accuracy differences across models, we pretrain using two different random seeds, and evaluate with both beam=1/4; we then present the best accuracies in each entry with respect to the 2 seeds and 2 beam choices. The accuracies are again over 4096 math problems. 

## **F.2 V-probing** 

Our V-probing was first introduced in Section 4.1 with more details given in Section B. It is a _fine-tuning_ process upon the pretrained language model, with an additional linear head on the output layer, and a small rank- _r_ update on the input (embedding) layer. The pretrained model is freezed, and only this linear head and the rank- _r_ update are trainable parameters during the fine-tuning. 

> 27A 128-GPU job with batch size 1 each would be identical to a 32-GPU job with batch size 4 each. 

28 

Recall we use _r_ = 8 in this paper (in contrast, the hidden dimension of GPT-12-12 is 768). This small value of _r_ ensures if probing accuracy is high, it mostly comes from the pretrained model and not the additional trainable parameters. 

- For V-probing, we use the same configurations as pretrain, except that: 

- For V-probing on the iGSM-med datasets, we used a learning rate of 0 _._ 002 (with no ramp-up, linear decay down to 0), weight decay of 0 _._ 01, batch size of 256, and trained for 100 _,_ 000 steps. 

- For V-probing on the iGSM-hard datasets, we used a learning rate of 0 _._ 002 (with no ramp-up, linear decay down to 0), weight decay of 0 _._ 01, batch size of 128, and trained for 100 _,_ 000 steps. 

**V-probing statistics.** In Figure 7(a), Figure 7(b), Figure 12, Figure 8(a), and Figure 8(b), we tested at least 4096 random problem-parameter pairs _in each cell_ . In Figure 8(a) and Figure 8(b), when evaluating probing results on GPT-2 model’s _generated_ correct or wrong solutions, we used beam=1 and dosample=False (greedy) for generation. (Results are similar for beam=4.) In our layer-wise `nece` ( _A_ ) probing experiments (Figure 10 and Figure 14), we tested at least 73728 random problem-parameter pairs in each case and then divided the results into bins based on the parameter _A_ ’s distances to the queries. 

## **G Failure Examples on GPT-4 / GPT-4o** 

In Figure 2, we conduct few-shot experiments using the latest versions of GPT-4 turbo (2024-0409) and GPT-4o (2024-05-13) models to evaluate their accuracies on our iGSM-med _pq_ dataset, with respect to different op _∈{_ 2 _,_ 3 _, . . . ,_ 20 _}_ . 

To ensure meaningful evaluation: 

- We replaced mod23 with mod5 to ensure that any errors are not due to arithmetic mistakes. We also provided a few arithmetic computation examples. 

- We minimized English diversity to ensure that any errors are not due to misunderstanding the problem description. Specifically, 

   - We fixed a simple categorization (School _,_ Classroom _,_ Backpack _,_ Stationerys), with only four items in each category. 

   - We provided an English _background_ paragraph to fully describe the structure graph (i.e., which item has which subitem), as well as the number of items in each category. The math problem is preceded by this background paragraph. 

- We provided five-shot problem/solution examples to ensure that GPT-4 understands how to solve such math problems step by step. 

We did not verify each step of GPT-4’s solution but checked if the final output number (between 0 and 4) matched the correct answer. The accuracy results are presented in Figure 2. It shows that the GPT-4o model is almost randomly guessing for op _≥_ 11, and GPT-4 turbo for op _≥_ 9. 

Furthermore, Figure 15 shows that when the GPT-4/4o models fail to answer the math problems, it is mostly not due to format errors or misunderstanding of the problem. Instead, just like what we discovered in Section 5, GPT-4/4o fail also because they compute unnecessary parameters (i.e., `nece` ( _A_ ) = false) or compute parameters that are not yet ready to be computed (i.e., `can` ~~`n`~~ `ext` ( _A_ ) = false). This further confirms that our findings do connect to practice, regarding the model’s hidden reasoning process. 

29 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process_images/iclr_2025_Physics_of_Language_Models_Part_2.1,_Grade-School_Math_and_the_Hidden_Reasoning_Process.pdf-0031-00.png)


**----- Start of picture text -----**<br>
system user BackgroundThe problem description is : ... : ...<br>You're an expert at solving elementary math problems involving addition, subtraction, and multiplication. You  assistant Here is the solution :  five-shot  add  background  to describe the<br>solve all the problems in a uniform format. All calculations are done modulo 5. For example, 3 + 2 equals 0, 1 + 1 equals 2, 4 + 2 + 4 equals 0, 3 * 2 equals 1, and 3 *  user BackgroundThe problem description is : ... : ... examples structure graph, to further help GPT-4<br>1 equals 3. When providing your solution, please end with 'The final answer is <<x>>.' where x is your final answer,  assistant Here is the solution :<br>an integer between 0 and 4. You must solve all the<br>problems using the same solution format. Background : There are 3 types of Classrooms: Music Room, Film Studio and Dance Studio. There are 3 types of Backpacks: School Daypack, Messenger Backpack and Canvas Backpack. There<br>are 3 types of Stationerys: Calculator, Scissors and Ruler. Each Music Room can have Messenger Backpack. Each Film Studio can have School Daypack, Messenger Backpack and Canvas<br>Our scenarios involve up to four categories of objects:  Backpack. Each Dance Studio can have School Daypack, Messenger Backpack and Canvas Backpack. Each School Daypack can have Scissors. Each Messenger Backpack can have Calculator and<br>schools, classrooms, backpacks and stationeries. Each  Scissors. Each Canvas Backpack can have Calculator and Ruler.<br>school may contain classrooms, each classroom may<br>contain backpacks, and each backpack may contain  The problem description is : The number of each Film Studio's Messenger Backpack equals the sum of each School Daypack's Scissors, each Dance Studio's Messenger Backpack and each Film<br>stationeries. We can specify quantities, such as "the number of dance studios at each Lakeshore High."  user Studio's School Daypack. The number of each Dance Studio's School Daypack equals the difference each Film Studio's Backpack minus each Music Room's Backpack. The number of each School Daypack's Scissors equals each Music Room's Messenger Backpack. The number of each Canvas Backpack's Calculator equals each Dance Studio's Messenger Backpack. The number of<br>Assume that every entity with the same name has an  each Film Studio's Canvas Backpack equals 1 times as much as the difference each Messenger Backpack's Scissors minus each Messenger Backpack's Stationery. The number of each<br>identical configuration; for example, each Lakeshore High  Messenger Backpack's Scissors equals the sum of each Dance Studio's Canvas Backpack, each Dance Studio's Messenger Backpack and each  Canvas Backpack's Ruler . The number of each Film<br>contains the same number of dance studios. Another guiding principle is that what is not mentioned does not  Studio's School Daypack equals 3 times as much as the difference each Music Room's Stationery minus each Canvas Backpack's Calculator. The number of each Messenger Backpack's Calculator equals each  Canvas Backpack's Ruler . The number of each  Canvas Backpack's Ruler  equals each Dance Studio's Canvas Backpack . The number of each Music Room's Messenger<br>exist: when we refer to classrooms at Lakeshore High, we  Backpack equals 2 times as much as each Canvas Backpack's Calculator. The number of each Dance Studio's Canvas Backpack equals 2 more than the sum of each Canvas Backpack's Calculator<br>are only discussing the classrooms explicitly mentioned in  and each Dance Studio's Messenger Backpack. The number of each Dance Studio's Messenger Backpack equals 2. How many Backpack does Film Studio have?<br>our scenario. Furthermore, if Lakeshore High is not even<br>mentioned, any classroom within it is automatically considered to be non-existent (i.e. 0). Here is the solution 1. Define Dance Studio's Messenger Backpack as D; so D = 2.: Failure reason: instant parameter “Canvas<br>2. Define Canvas Backpack's Calculator as C; so C = D = 2.3. Define Music Room's Messenger Backpack as M; so M = 2 * C = 2 * 2 = 4. Backpack’s Ruler” cannot be computed in step 5<br>assistant 4. Define School Daypack's Scissors as S; so S = M = 4.5. Define 6. Define  Canvas Backpack's Ruler Dance Studio's Canvas Backpack as R; so R = D = 2. as A; so A = 2 + (C + D) = 2 + (2 + 2) = 2 + 4 = 1.<br>…The final answer is <<4>>. GPT-4o on  iGSM-med pq  (op=20)<br>system user BackgroundThe problem description is : ... : ...<br>You're an expert at solving elementary math problems involving addition, subtraction, and multiplication. You  assistant Here is the solution :  five-shot  add  background  to describe the<br>solve all the problems in a uniform format. All calculations are done modulo 5. For example, 2 + 4 equals 1, 4 + 4 equals 3, 3 + 4 + 1 equals 3, 4 * 2 equals 3, and 1 *  user BackgroundThe problem description is : ... : ... examples structure graph, to further help GPT-4<br>3 equals 3. When providing your solution, please end with 'The final answer is <<x>>.' where x is your final answer,  assistant Here is the solution :<br>an integer between 0 and 4. You must solve all the<br>problems using the same solution format. Background : There are 2 types of Schools: Lincoln High and Lakeshore High. There are 2 types of Classrooms: Music Room and Dance Studio. There are 2 types of Backpacks: Canvas Backpack<br>and Messenger Backpack. There are 2 types of Stationerys: Calculator and Stapler. Each Lincoln High can have Dance Studio. Each Lakeshore High can have Music Room and Dance Studio.<br>Our scenarios involve up to four categories of objects:  Each Music Room can have Canvas Backpack and Messenger Backpack. Each Dance Studio can have Canvas Backpack and Messenger Backpack. Each Canvas Backpack can have Calculator and<br>schools, classrooms, backpacks and stationeries. Each  Stapler. Each Messenger Backpack can have Calculator and Stapler.<br>school may contain classrooms, each classroom may<br>contain backpacks, and each backpack may contain stationeries. We can specify quantities, such as "the  user The problem description is Stapler equals the sum of each : The number of each Messenger Backpack's Stapler equals 1. The number of each Lakeshore High's Dance Studio equals 2. The number of each Canvas Backpack's  Dance Studio's Canvas Backpack  and each Canvas Backpack's Calculator. The number of each Lakeshore High's Music Room equals the difference each<br>number of dance studios at each Lakeshore High."  Messenger Backpack's Stapler minus each Music Room's Canvas Backpack. The number of each Lincoln High's Dance Studio equals 1 more than each Messenger Backpack's Stapler. The<br>Assume that every entity with the same name has an  number of each Messenger Backpack's Calculator equals each Messenger Backpack's Stapler. The number of each Music Room's Canvas Backpack equals each Dance Studio's Canvas<br>identical configuration; for example, each Lakeshore High contains the same number of dance studios. Another  Backpack. The number of each Dance Studio's Messenger Backpack equals each Messenger Backpack's Stationery. The number of each Music Room's Messenger Backpack equals each Lincoln High's Backpack. The number of each  Dance Studio's Canvas Backpack  equals each  Messenger Backpack's Stationery . The number of each Canvas Backpack's Calculator equals 1 more than<br>guiding principle is that what is not mentioned does not  each Music Room's Messenger Backpack. How many Calculator does Canvas Backpack have?<br>exist: when we refer to classrooms at Lakeshore High, we<br>are only discussing the classrooms explicitly mentioned in  Here is the solution :<br>our scenario. Furthermore, if Lakeshore High is not even  1. Define Messenger Backpack's Stapler as S; so S = 1.<br>mentioned, any classroom within it is automatically  2. Define Lakeshore High's Dance Studio as D; so D = 2. Failure reason: instance parameter “Dance Studio's<br>considered to be non-existent (i.e. 0). assistant 3. Define Lincoln High's Dance Studio as L; so L = S + 1 = 1 + 1 = 2.4. Define Messenger Backpack's Calculator as C; so C = S = 1.5. Define  Dance Studio's Canvas Backpack  as B; so B = S = 1. Canvas Backpack” cannot be computed in step 5<br>6. Define Music Room's Canvas Backpack as M; so M = B = 1.<br>7. Define Music Room's Messenger Backpack as N; so N = L = 2.<br>unnecessary parameter 8. Define Canvas Backpack's Calculator as K; so K = N + 1 = 2 + 1 = 3 \mod 5 = 3.9. Define Canvas Backpack's Stapler as P; so P = B + K = 1 + 3 = 4.The final answer is <<3>>. GPT-4o  on  iGSM-med pq  (op=10)<br>system user BackgroundThe problem description is : ... : ...<br>You're an expert at solving elementary math problems involving addition, subtraction, and multiplication. You  assistant Here is the solution :  five-shot  add  background  to describe the<br>solve all the problems in a uniform format. All calculations are done modulo 5. For example, 4 + 4 equals 3, 1 + 3 equals 4, 2 + 1 + 4 equals 2, 2 * 2 equals 4, and 3 *  user BackgroundThe problem description is : ... : ... examples structure graph, to further help GPT-4<br>2 equals 1. When providing your solution, please end with 'The final answer is <<x>>.' where x is your final answer,  assistant Here is the solution :<br>an integer between 0 and 4. You must solve all the<br>problems using the same solution format. Background : There are 3 types of Backpacks: Canvas Backpack, School Daypack and Laptop Backpack. There are 3 types of Stationerys: Stapler, Scissors and Calculator. Each Canvas Backpack<br>can have Scissors. Each School Daypack can have Stapler and Calculator. Each Laptop Backpack can have Scissors and Calculator.<br>Our scenarios involve up to four categories of objects: schools, classrooms, backpacks and stationeries. Each school may contain classrooms, each classroom may  user The problem description is Canvas Backpack's Stationery, each School Daypack's Calculator, each Laptop Backpack's Calculator and each Laptop Backpack's Scissors. The number of each Laptop Backpack's Scissors : The number of each  Canvas Backpack's Scissors  equals each  Laptop Backpack's Stationery . The number of each School Daypack's Stapler equals the sum of each<br>contain backpacks, and each backpack may contain  equals 4 times as much as the sum of each Laptop Backpack's Calculator and each School Daypack's Calculator. The number of each Laptop Backpack's Calculator equals 1 times as much as<br>stationeries. We can specify quantities, such as "the  each School Daypack's Calculator. The number of each School Daypack's Calculator equals 3. How many Stapler does School Daypack have?<br>number of dance studios at each Lakeshore High."<br>Assume that every entity with the same name has an identical configuration; for example, each Lakeshore High contains the same number of dance studios. Another guiding principle is that what is not mentioned does not exist: when we refer to classrooms at Lakeshore High, we are only discussing the classrooms explicitly mentioned in our scenario. Furthermore, if Lakeshore High is not even  assistant Here is the solution Define School Daypack's Calculator as C; so C = 3. Define Laptop Backpack's Calculator as L; so L = 1 * C = 1 * 3 = 3. Define Laptop Backpack's Scissors as S; so S = 4 * (L + C) = 4 * (3 + 3) = 4 * 6 = 4. Define Define School Daypack's Stapler as P; so P = B + C + L + S = 4 + 3 + 3 + 4 = 14 % 5 = 4. Canvas Backpack's Scissors  :  as B; so B = S = 4.  Backpack's Scissors” cannot be computed in step 5Failure reason: instance parameter “Canvas<br>mentioned, any classroom within it is automatically considered to be non-existent (i.e. 0). The final answer is <<4>>. GPT-4-turbo  on  iGSM-med pq  (op=10)<br>system user BackgroundThe problem description is : ... : ...<br>You're an expert at solving elementary math problems involving addition, subtraction, and multiplication. You  assistant Here is the solution :  five-shot  add  background  to describe the<br>solve all the problems in a uniform format. All calculations are done modulo 5. For example, 4 + 4 equals 3, 2 + 4 equals 1, 3 + 3 + 3 equals 4, 3 * 3 equals 4, and 2 *  user BackgroundThe problem description is : ... : ... examples structure graph, to further help GPT-4<br>2 equals 4. When providing your solution, please end with 'The final answer is <<x>>.' where x is your final answer,  assistant Here is the solution :<br>an integer between 0 and 4. You must solve all the problems using the same solution format. Background are 3 types of Backpacks: Laptop Backpack, Messenger Backpack and Canvas Backpack. There are 2 types of Stationerys: Calculator and Scissors. Each Lakeshore High can have Film Studio. : There are 4 types of Schools: Lakeshore High, Lincoln High, Riverview High and Central High. There are 3 types of Classrooms: Drawing Studio, Film Studio and Music Room. There<br>Our scenarios involve up to four categories of objects: schools, classrooms, backpacks and stationeries. Each school may contain classrooms, each classroom may  Each Lincoln High can have Drawing Studio and Music Room. Each Riverview High can have Drawing Studio. Each Central High can have Film Studio and Music Room. Each Drawing Studio can have Laptop Backpack. Each Film Studio can have Messenger Backpack and Canvas Backpack. Each Music Room can have Canvas Backpack. Each Laptop Backpack can have Scissors. Each Messenger Backpack can have Calculator. Each Canvas Backpack can have nothing.<br>contain backpacks, and each backpack may contain stationeries. We can specify quantities, such as "the number of dance studios at each Lakeshore High." Assume that every entity with the same name has an identical configuration; for example, each Lakeshore High contains the same number of dance studios. Another guiding principle is that what is not mentioned does not exist: when we refer to classrooms at Lakeshore High, we are only discussing the classrooms explicitly mentioned in  user The problem description is: Room's Canvas Backpack, each Central High's Music Room and each Central High's Film Studio. The number of each Laptop Backpack's Scissors equals 2. The number of each Central High's Film Studio equals 2 more than the sum of each Music Room's Canvas Backpack and each Central High's Music Room. The number of each Drawing Studio's Laptop Backpack equals each Film Studio's Messenger Backpack. The number of each Lincoln High's Drawing Studio equals the sum of each Riverview High's Backpack, each Laptop Backpack's Stationery, each Film Studio's Messenger Backpack and each Drawing Studio's Laptop Backpack. The number of each Film Studio's Messenger Backpack equals 2. The number of each Lakeshore High's Film Studio equals each Film Studio's Messenger Backpack. The number of each Film Studio's Canvas Backpack equals each Film Studio's Messenger Backpack. The number of each Central High's Music Room equals 4. The number of each Music Room's Canvas Backpack equals 4. The number of each Messenger Backpack's Calculator equals each Lincoln High's Music Room. How many Backpack does Lakeshore High have?  The number of each Riverview High's Drawing Studio equals 3. The number of each Lincoln High's Music Room equals 4 times as much as the sum of each Music<br>our scenario. Furthermore, if Lakeshore High is not even mentioned, any classroom within it is automatically considered to be non-existent (i.e. 0). assistant Here is the solution Define Film Studio's Messenger Backpack as m; so m = 2. Define Film Studio's Canvas Backpack as c; so c = m = 2. :  Failure reason: abstract parameter “Lakeshore High's Backpack” can’t be computed yet; it equals L x “Film<br>Define Define Central High's Music Room as MCentral High's Film Studio as F; so F = 2 + c + M = 2 + 2 + 4 = 3. ; so M = 4.  Studio's Backpack (= m + c)” but it is not computed yet.<br>unnecessary parameters Define Lakeshore High's Film Studio as L; so L = m = 2. Define The final answer is <<4>>. Lakeshore High's Backpack  as B; so B = m + c = 2 + 2 = 4. GPT-4-turbo  on iGSM-med pq  (op=5)<br>…<br>…<br>…<br>…<br>**----- End of picture text -----**<br>


Figure 15: Failure examples for GPT-4/GPT-4o on iGSM-med _pq_ . They make mistakes similar to what we discover in this paper, that is to compute unnecessary parameters in the solutions (i.e., `nece` ( _A_ ) = false), as well as computing parameters that are not yet ready to compute (i.e., `can` ~~`n`~~ `ext` ( _A_ ) = false). 

30 

## **References** 

- [1] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 1, Learning Hierarchical Language Structures. _ArXiv e-prints_ , abs/2305.13673, May 2023. Full version available at `http://arxiv.org/ abs/2305.13673` . 

- [2] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.2, Knowledge Manipulation. _ArXiv e-prints_ , abs/2309.14402, September 2023. Full version available at `http://arxiv.org/abs/ 2309.14402` . 

- [3] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.1, Knowledge Storage and Extraction. In _ICML_ , 2024. Full version available at `http://arxiv.org/abs/2309.14316` . 

- [4] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. _ArXiv e-prints_ , abs/2404.05405, April 2024. Full version available at `http://arxiv.org/abs/ 2404.05405` . 

- [5] Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via overparameterization. In _ICML_ , 2019. Full version available at `http://arxiv.org/abs/1811.03962` . 

- [6] Cem Anil, Yuhuai Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, and Behnam Neyshabur. Exploring length generalization in large language models. _Advances in Neural Information Processing Systems_ , 35:38546–38556, 2022. 

- [7] Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. GPT-NeoX-20B: An open-source autoregressive language model. In _Proceedings of the ACL Workshop on Challenges & Perspectives in Creating Large Language Models_ , 2022. URL `https://arxiv.org/abs/2204.06745` . 

- [8] S´ebastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv preprint arXiv:2303.12712_ , 2023. 

- [9] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- [10] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. _arXiv preprint arXiv:2101.00027_ , 2020. 

- [11] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio C´esar Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. Textbooks are all you need. _arXiv preprint arXiv:2306.11644_ , 2023. 

- [12] John Hewitt and Christopher D. Manning. A structural probe for finding syntax in word representations. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4129–4138, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1419. URL `https://aclanthology.org/N19-1419` . 

- [13] Samy Jelassi, St´ephane d’Ascoli, Carles Domingo-Enrich, Yuhuai Wu, Yuanzhi Li, and Fran¸cois Charton. Length generalization in arithmetic transformers. _arXiv preprint arXiv:2306.15400_ , 2023. 

- [14] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. _arXiv preprint arXiv:2001.08361_ , 2020. 

- [15] Yuanzhi Li, S´ebastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. _arXiv preprint arXiv:2309.05463_ , 2023. 

- [16] Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and Yi Zhang. TinyGSM: achieving _>_ 80% on GSM8k with small language models. _arXiv preprint arXiv:2312.09241_ , 2023. 

31 

- [17] OpenAI. Gpt-4 technical report, 2023. 

- [18] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

- [19] Lance J Rips. _The psychology of proof: Deductive reasoning in human thinking_ . Mit Press, 1994. 

- [20] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2021. 

- [21] Tian Ye, Zicheng Xu, Yuanzhi Li, and Zeyuan Allen-Zhu. Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems. _arXiv preprint arXiv:xxxx.xxxxx_ , 2024. to appear. 

- [22] Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, Tiffany Zhao, Pranav Raja, Dylan Slack, Qin Lyu, et al. A careful examination of large language model performance on grade school arithmetic. _arXiv preprint arXiv:2405.00332_ , 2024. 

- [23] Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Josh Susskind, Samy Bengio, and Preetum Nakkiran. What algorithms can transformers learn? a study in length generalization. _arXiv preprint arXiv:2310.16028_ , 2023. 

32 

