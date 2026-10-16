# **MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs** 

**Zimu Lu** _[∗]_[1] **, Aojun Zhou** _[∗]_[1] **, Houxing Ren**[1] **, Ke Wang**[1] **, Weikang Shi**[1] **Junting Pan**[1] _[,]_[3] , **Mingjie Zhan** _[†]_[1] , **Hongsheng Li** _[†]_[1] _[,]_[2] _[,]_[3] 

1Multimedia Laboratory (MMLab), The Chinese University of Hong Kong 

2Shanghai Artificial Intelligence Laboratory 3CPII under InnoHK luzimu@mail.ustc.edu.cn {aojunzhou, zmjdll}@gmail.com hsli@ee.cuhk.edu.hk 

## **Abstract** 

Large language models (LLMs) have exhibited great potential in mathematical reasoning. However, there remains a performance gap in this area between existing open-source models and closed-source models such as GPT-4. In this paper, we introduce **MathGenie** , a novel method for generating diverse and reliable math problems from a small-scale problem-solution dataset (denoted as _seed data_ ). We augment the ground-truth solutions of our seed data and train a back-translation model to translate the augmented solutions back into new questions. Subsequently, we generate code-integrated solutions for the new questions. To ensure the correctness of the code-integrated solutions, we employ rationale-based strategy for solution verification. Various pretrained models, ranging from 7B to 70B, are trained on the newly curated data to test the effectiveness of the proposed augmentation technique, resulting in a family of models known as _MathGenieLM_ . These models consistently outperform previous open-source models across five representative mathematical reasoning datasets, achieving state-of-the-art performance. In particular, MathGenieLM-InternLM2 achieves an accuracy of 87.7% on GSM8K and 55.7% on MATH, securing the best overall score among open-source language models. 

## **1 Introduction** 

Large language models (LLMs), such as GPT4 (OpenAI, 2023), and others (Touvron et al., 2023; Yue et al., 2023; Gou et al., 2023; Wang et al., 2023), have demonstrated outstanding capabilities in mathematical reasoning. Existing methods explored three main types of solution formats for tackling the math-solving problem: text-only Chain-ofThought (CoT) solutions (Wei et al., 2022), codeonly Program-of-Thought (PoT) solutions (Chen 

> _∗_ Equal contribution _†_ Corresponding author 

et al., 2022), and code-integrated solutions (Zhou et al., 2023). Among these, code-integrated solutions demonstrate superior performance over both CoT and PoT solutions (Gou et al., 2023; Wang et al., 2023; Zhou et al., 2023), indicating their effectiveness in enhancing problem-solving abilities. 

In this paper, we focus on creating diverse and reliable augmented questions and ensuring the reliability of the generated code-integrated solutions, thereby better finetuning pretrained base models. Existing works, such as ToRA (Gou et al., 2023) and MathCoder (Wang et al., 2023), build codeintegrated solutions and augment math questions using off-the-shelf GPT-4. However, scaling up the training data with GPT-4 becomes prohibitively expensive. 

Consequently, developing a free open-source model to generate large-scale synthetic data presents a promising alternative, offering scalability and cost-effectiveness (Yuan et al., 2023; Singh et al., 2023). To effectively scale-up math problem-solving data, we focus on handling two critical challenges: (1) how to generate high-quality and diverse math problems to aid in generalization, and (2) how to generate accurate and reliable solutions for the augmented problems without human-annotated ground truth, preferably in a code-integrated format. A unified framework, _MathGenie_ , is proposed, which consists of three components as shown in Fig. 1 to tackle the abovementioned challenges: _Iterative Solution Augmentation_ , _Question Back-translation_ and _VerificationBased Solution Filtering_ . 

Iterative Solution Augmentation and Question Back-translation aims to generate diverse and reliable math problems. Unlike direct question augmentation (Yu et al., 2023), the proposed math problem back-translation leverages the constraints and logical relationships inherent in mathematical solutions to create a diverse and high-quality set of new math problems. Specifically, we iteratively 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs_images/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Step 1 : Iterative Solution Augmentation<br>Solution Augmentation  Iterative Solution Augmentation<br>Model ( 𝑀𝑡𝑒𝑥𝑡 )<br>…<br>Seed Solutions ( 𝑆 [0] ) Solutions 1 (𝑆 [1] ) Solutions 2  (𝑆 [2] ) Solutions K  (𝑆 [𝐾] )<br>Step 2: Question Backtranslation<br>Question Backtranslation<br>Backtranslation Model ( 𝑀𝑏𝑎𝑐𝑘𝑡𝑟𝑎𝑛𝑠 ) Augmented Solutions  (𝑆 [𝐴𝑢𝑔] ) Augmented Questions  (𝑄 [𝐴𝑢𝑔] )<br>Step 3: Verification-Based Solution Filtering<br>Code-Integrated Solution<br>Generation Verification-based Filtering<br>Augmented Questions  (𝑄 [𝐴𝑢𝑔] ) Augmented Questions and Code  Verification Filtered Augmented Questions and<br>Integrated Solutions  (𝑄 [𝐴𝑢𝑔] , 𝑆𝑐𝑜𝑑𝑒𝐴𝑢𝑔 ) Code Integrated Solutions  (𝑄 [𝐴𝑢𝑔] , 𝑆𝑐𝑜𝑑𝑒𝐴𝑢𝑔 )<br>Solution Generation and<br>Verification Model ( 𝑀𝑐𝑜𝑑𝑒 )<br>**----- End of picture text -----**<br>


Figure 1: Framework of MathGenie. Iterative Solution Augmentation augments human-annotated solutions in GSM8K and MATH to create new solutions, as shown in Step 1. These solutions are then back-translated to new questions using Question Back-translation, demonstrated in Step 2. Then reliable code-integrated solutions are curated using Verification-Based Solution Filtering, by generating solutions and filtering them using verification rationales, as shown in Step 3. 

augment the human-annotated solutions from the relatively small training sets of MATH (Hendrycks et al., 2021) and GSM8K (Cobbe et al., 2021), generating a large-scale collection of augmented new solutions, as shown in Step 1 of Fig. 1. These solutions are then processed by a math backtranslation model, _M_ backtrans, to back-translate the augmented solutions into their corresponding math questions, as demonstrated in Step 2 of Fig. 1. This method draws inspiration from Instruction Backtranslation (Li et al., 2023b), which back-translates instructions from texts in a web corpus. However, the key difference is that our source solutions for back-translation are augmented from existing ones to ensure the reliability and solvability of the augmented questions. 

These newly generated math problems lack reliable ground-truth solutions, which necessitates the proposed Verification-Based Solution Filtering. We first build a model, _M_ code, capable of generating code-integrated solutions and verifying these solutions. Then, code-integrated solutions of the new questions are generated with this model. To enhance the reliability of these code-integrated solutions, we use _M_ code to verify the model-generated solutions by generating verification rationales for them, as demonstrated in Step 3 of Fig. 1. The verification rationales use interleaved natural language and code to verify the correctness of the solutions, 

as shown in Tab. 13 and Tab. 14 in Appendix G. Only the solutions verified to be correct are retained, thus improving the accuracy and quality of the generated data. 

Based on the proposed MathGenie framework, we obtain a large-scale model-generated math problem-solution dataset, _MathGenieData_ , featuring diverse augmented questions and reliable codeintegrated solutions. 

To evaluate the effectiveness of the questionsolution augmentation framework MathGenie, we finetune various state-of-the-art pretrained models, ranging from 7B to 70B. This results in **MathGenieLM** , a new family of math models with excellent performance. Our models demonstrate high accuracy across five diverse and representative math datasets: MATH, GSM8K, SVAMP, Simuleq, and Mathematics. In particular, MathGenieLMInternLM2 achieves an accuracy of 87.7% on GSM8K and 55.7% on MATH, achieving the best overall score. When enhanced by majority voting, MathGenieLM-Llama-2-70B attains a 10-path accuracy rate of 91.7% on GSM8K and 63.3% on MATH. 

The main contributions of this paper are summarized as follows: (1) We propose the MathGenie pipeline, which is designed to enhance the scale, diversity, and quality of synthetic math questions, as well as to improve the accuracy of the 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs_images/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs.pdf-0003-00.png)


**----- Start of picture text -----**<br>
A pair of seed question and solution Seed Solution:  She bought 5 shorts at $7 each so 5*7=$35. She<br>Seed Question:  Ann's favorite store was having a  bought 2 pairs of shoes at $10 each so 2*10=$20. The shorts and<br>summer clearance. For $75 she bought 5 pairs of  shoes cost her 35+20 = $55. We know she spent 75 total and the<br>shorts for $7 each and 2 pairs of shoes for $10  shorts and shoes cost $55 which left a difference of 75-55 = $20.<br>each. She also bought 4 tops, all at the same price.  She bought 4 tops for a total of $20 so 20/4 = $5.<br>How much did each top cost? #### 5 × 𝑲<br>Direct Question  Iterative Solution<br>Augmentation Augmentation<br>Augmented Question:  restaurant was having a winter sale.she bought desserts worth $20 eachappetizers, all at the same price. How much did each appetizer cost?5 entrees worth $15 each Ann's favorite . She also bought 4 For and$75  2  Augmented Solution: 5*1=$5. She bought 2 oranges at $2 each so 2*2=$4. The bananas and oranges cost her 5+4 = $9. We know total and the bananas and oranges cost $9 which left a difference of 10-9 = $1#### 0.25 Augmented Solution: 5*1=$5. She bought 2 oranges at $2 each so 2*2=$4. The bananas and oranges cost her 5+4 = $9. We know she spent 10 total and the bananas and oranges cost $9 which left a difference of 10-9 = $1. She bought 4 apples for a total of $1 so 1/4 = $0.25.#### 0.25 Augmented Solution: 5*1=$5. She bought 2 oranges at $2 each so 2*2=$4. The bananas and oranges cost her 5+4 = $9. We know she spent 10 total and the bananas and oranges cost $9 which left a difference of 10-9 = $1. She bought 4 apples for a total of $1 so 1/4 = $0.25.#### 0.25 . She bought 4 apples for a total of $1 so 1/4 = $0.25. She bought 5 bananas at $1 each so  She bought 5 bananas at $1 each so  She bought 5 bananas at $1 each so she spent 10<br>ⅹ Question<br>Explanation:  Ann bought 5 entrees at $15 each so  Back-translation<br>5*15=75. She bought 2 desserts at $20 each so<br>2*20=40. The entrees and desserts cost her 75+40=115. This is impossible, as we know she spent 75$ total, which is less than 115.So the new problem has  no answer . Augmented Question:  oranges, and 4 apples. She spent a total of $10. If bananas cost $1 each and oranges cost $2 each, how much does Jillian spend on each apple? Augmented Question:  oranges, and 4 apples. She spent a total of $10. If bananas cost $1 each and oranges cost $2 each, how much does Jillian spend on each apple? Augmented Question:  oranges, and 4 apples. She spent a total of $10. If bananas cost $1 each and oranges cost $2 each, how much does Jillian spend on each apple?Jillian bought 5 bananas, 2 Jillian bought 5 bananas, 2 Jillian bought 5 bananas, 2<br>√<br>**----- End of picture text -----**<br>


Figure 2: Comparison between Direct Question Augmentation (left) and Iterative Solution Augmentation and Question Back-translation (right). Direct Question Augmentation damages the hidden constraints between the conditions (cost of part of the things bought must not be more than the total cost), thus producing a question with no answer. Question Back-translation considers the solution, and correctly augments the question. 

code-integrated solutions generated for them. (2) We conduct extensive experiments on various pretrained language models, demonstrating consistently superior performance across multiple math datasets. 

## **2 MathGenie** 

In this section, we introduce MathGenie, a pipeline for creating diverse and reliable math problems through back-translation, and curating high-quality code-integrated solutions through verification. We begin by introducing the seed data and solution generator model. Next, we present the proposed MathGenie pipeline, which consists of three key steps, as shown in Fig. 1: Iterative Solution Augmentation, Question Back-translation, and Verification-Based Solution Filtering. 

**Seed Data.** The seed data consists of two parts: (1) The first part is used for data augmentation, consisting of 15K math problems and human-annotated solutions from the training sets of GSM8K and MATH. We denote it as _D_ text = _{_ ( _q[i] , s[i]_ ) _}[n] i_ =1[, where] _[ q][i]_[ is the] _[ i]_[-th question,] _[ s][i]_[ is its] natural-language solution, and _n_ is the total number of cases. (2) The second part is used for training our candidate solution generator model, which serves to generate candidate solutions for the augmented questions. We denote this part of the seed data as _D_ code = _{_ ( _q[i] , s[i]_ code _[, v]_ code _[i]_[)] _[}] i[N]_ =1[,][where] _[ q][i]_ is the question, _s[i]_ code[is][its][code-integrated][solu-] tion, and _v_ code _[i]_[is the code-integrated verification] 

rationales for the question-solution pair. It contains 80K samples of code-integrated solutions for problems in GSM8K and MATH, as well as codeintegrated verification rationales for these solutions. Multiple different solutions are collected for each question. We acquire these solutions and verification rationales using the GPT-4 Code Interpreter, which consist of interleaved natural language and code. 

**Candidate Solution Generator.** The candidate solution generator is a Llama-2 70B model trained with _D_ code and denoted as _M_ code. The code-integrated solutions in _D_ code enables _M_ code to output candidate code-integrated solutions for given math problems, similar to (Wang et al., 2023). It has an accuracy of 86.4% on GSM8K and 49.5% on MATH. The verification rationales in _D_ code enable _M_ code to effectively verify the solutions with code-integrated rationales. The training method of data in the code-integrated format is as described in Wang et al. (2023). 

**Iterative Solution Augmentation.** Different from previous works that directly augment math questions (Luo et al., 2023), we propose to augment solutions first and then back-translate the augmented solutions to their corresponding questions to better constrain the question generation process and enhance the reliability of machine-generated questions. The proposed strategy is also different from the previous Instruction Back-translation method (Li et al., 2023b), which leverages large 

amounts of texts in a web corpus. As existing solutions are limited in number and already have corresponding questions, it is necessary to augment them before the back-translation. 

To augment the solutions in _D_ text into related new ones, we develop a solution augmentation model, _M_ text, by finetuning the LLaMA-2 70B model on high-quality instructional datasets, including OpenOrca[1] and Alpaca-GPT4[2] . _M_ text takes in a solution and a prompt instructing the model to augment it, and outputs an augmented solution. The prompts are shown in Tab. 1. The augmentations are carefully constrained and reliable. We iteratively augment each solution in _D_ text. For convenience, the set of human-annotated solutions in _D_ text are denoted as _S_[0] . The solutions in _S_[0] are augmented by _M_ text to create _S_[1] . As shown in Step 1 of Fig. 1, this process is repeated on the previously generated solutions, with _S_[2] being created from _S_[1] , and so on. After _K_ rounds, the final set of augmented solutions, denoted as _S_[Aug] , is created by taking the union of _S_[1] _, S_[2] _, . . . , S[K]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs_images/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs.pdf-0004-02.png)


Through iterative solution augmentation, each round produces a set of solutions that differs from the previous, making the solutions gradually deviate more from the original solutions. Consequently, the diversity of the augmented solutions is ensured, which leads to diverse augmented question-solution pairs as mentioned in the following sections. The iteration process is beneficial to the final performance, as demonstrated in Tab. 9 of Appendix B 

**Question Back-translation.** We introduce Question Back-translation to translate the solutions in _S_[Aug] back to their corresponding math problems. To enhance the accuracy of the translation, we build a Question Back-translation Model _M_ backtrans by finetuning Llama-2 70B on the reversed pairs of question and solution in _D_ text. The format of each sample in the finetuning data can be denoted as ( _s, q_ ), where _s_ is a solution, serving as input, and _q_ is its corresponding question. Then, we present the solutions in _S_[Aug] to _M_ backtrans, thereby translating them back to a set of new questions, denoted as _Q_[Aug] . An example of Question Back-translation is presented in Tab. 2. 

1https://huggingface.co/datasets/Open-Orca/ OpenOrca 

2https://huggingface.co/datasets/vicgalle/ alpaca-gpt4 

**Prompt** _Replace the objects and verbs_ in Solution. Solution: {solution} You must replace the objects and verbs in the Solution with objects and verbs very different from before to create a new solution. _Add an extra step_ to the solution so that it is more complicated. Solution: {solution} You must add an extra step to the Solution to create a new solution. _Change the numbers_ in Solution. Solution: {solution} You must change the numbers in the Solution to create a new solution. _Replace the numbers and variables_ in Solution with different variables and numbers you pick randomly. Solution: {solution} You must replace the numbers and variables in the Solution with different numbers and variables you pick randomly to create a new solution. 

Table 1: Different prompts for solution augmentation. “{solution}” is replaced with the solution to be augmented. 

**Input** Create a New Problem based on the Solution: For the soccer practice, Jack needs 3 days * 10 soccer balls/day = 30 soccer balls. Adding the soccer balls for the Gym class, he needs to give a total of 30 soccer balls + 10 soccer balls = 40 soccer balls. **Output** New Problem: Jack is in charge of the equipment room at his school. He needs to gather 10 soccer balls for the gym class and also give 3 days worth of soccer balls for the soccer team’s practice. If each practice uses 10 soccer balls, how many soccer balls does Jack need to give out? 

Table 2: An example of the proposed Question Backtranslation. The solution is input to the question backtranslation model, and the model outputs its corresponding question. 

Question Back-translation operates on _S_[Aug] , which is less unpredictable than the texts from the web corpus used in Instruction Back-translation (Li et al., 2023b). By leveraging the constraints shown in the solutions, it is possible to create new questions more reliable than what direct question augmentation can produce, as validated in experiments. 

**Verification-based Solution Filtering.** Existing open-source models such as MathCoder (Wang et al., 2023) only has the ability to solve math problems but are unable to effectively verify their solutions. We enhance _M_ code’s ability to verify the solutions by adding code-integrated verification rationales to the finetuning data. The training samples of verification rationales in the seed data are in the format of ( _q, s_ code _, v_ code), where _q_ and _s_ code are a pair of question and code-integrated solution, and _v_ code is the code-integrated verification. The ( _q, s_ code) pairs are the input, while the model is trained to output _v_ code. In this way, the baseline math problem-solving model _M_ code acquires the ability to verify its solutions with rationales made 

of interleaved natural language and code. This ability not only facilitates Verification-Based Solution Filtering, but can also play a role in enhancing inference accuracy. 

To perform the proposed Verification-Based Solution Filtering, we first generate code-integrated solutions for each question in _Q_[Aug] . Initial filtering is performed using answer consistency (Wang et al., 2022), removing a question if its solutions reach different answers. We then present each questionsolution pair to _M_ code, prompting it to output a code-integrated verification rationale, from which we can determine whether the solution is verified as correct or wrong. Examples of the verification process are shown in Appendix G. Candidate solutions that are verified to be wrong are abandoned. The process is demonstrated in Step 3 of Fig. 1. 

The pipeline proposed above results in 170K samples of question and code-integrated solution pairs, denoted as AugData. AugData consists of two parts: the 110K samples augmented from GSM8K dataset, denoted as AugGSM8K, and the 60K samples augmented from MATH dataset, denoted as AugMATH. We denote the above mentioned seed data for training _M_ code as SeedData. Combining SeedData and AugData, we present the final dataset, **MathGenieData** , which can be used to finetune various pretrained models, such as Llama-2 (Touvron et al., 2023) and CodeLlama (Roziere et al., 2023), enhancing their problem-solving ability and solution verification skills. The resulting family of mathematical reasoning models is named MathGenieLM. 

## **3 Experiments** 

## **3.1 Experimental Setup** 

**Datasets.** We evaluate our models on two indomain datasets: GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021), whose training sets are used for finetuning. Additionally, we evaluate the final models on three out-ofdomain datasets: SVAMP (Patel et al., 2021), Simuleq (Koncel-Kedziorski et al., 2016), and Mathematics (Davies et al., 2021), to evaluate the generalization ability of our proposed method. 

**Models.** We perform full-parameter finetuning on various pretrained models, including Llama-2 7B, 13B, and 70B (Touvron et al., 2023), CodeLlama 7B, 13B, and 34B (Roziere et al., 2023), Llemma 7B and 34B (Azerbayev et al., 2023), Mistral 7B (Jiang et al., 2023), Mixtral-8x7B (Jiang 

et al., 2024), and InternLM2 20B (Team, 2023). Finetuning details are described in Appendix E. 

**Compared methods.** We compare MathGenieLM with closed-source models such as ChatGPT-3.5 (Brown et al., 2020), GPT-4 (OpenAI, 2023), and PaLM-2 (Anil et al., 2023), as well as open-source models such as Mammoth (Yue et al., 2023), MathCoder (Wang et al., 2023), ToRA (Gou et al., 2023), and WizardMath (Luo et al., 2023). 

## **3.2 Main Results** 

Tab. 3 shows the accuracy of MathGenieLM across five datasets. Based on the results, we make the following observations: (1) For open-source models with parameters ranging from 7B to 70B, MathGenieLM achieves state-of-the-art performance. (2) MathGenieLM demonstrates particularly high performance on the three out-of-domain datasets compared to previous open-source models, showcasing the superior generalization capability of our method. (3) MathGenieLM’s accuracy exceeds that of ChatGPT-3.5 and PaLM-2. However, there remains a noticeable gap when compared to GPT-4’s performance. (4) MathGenieLM-Llemma-34B and MathGenieLM-InternLM2-20B reach over 55% accuracy on the challenging MATH dataset. This might be attributed to the high-quality math-related data they used in pretraining. (5) Mixtral-8x7B achieves excellent performance, demonstrating the potential of Mixture of Experts (MoE) models. The results in Tab. 3 are all obtained using greedy decoding. 

Apart from the results obtained with greedy decoding, we also report the results of majority voting using multiple sampled paths (Wang et al., 2022), conducted on MathGenieLM-Llama-2-70B, compared with ToRA-Llama-2-70B. The results are shown in Tab. 4, where “ _k_ ” represents the number of solutions generated for majority voting. We observe that, with _k_ = 10, majority voting significantly increases the accuracy across all five datasets, yielding an average gain of 7.9%. Specifically, at _k_ = 10, MathGenieLM-Llama-270B achieves an accuracy of 91.5% on GSM8K and 63.3% on MATH, significantly outperforming ToRA-70B at _k_ = 50. This demonstrates the superior performance of our model. 

## **3.3 Ablation Study** 

The following are some ablation studies. All finetuning in the ablation studies has been conducted using Mistral-7B as the base model. 

||**Model**<br>**Base**<br>**Size**|**In-Domain**<br>**GSM8K**<br>**MATH**|**Out-of-Domain**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**|**Out-of-Domain**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**|**Out-of-Domain**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**|**Average**|**Average**|
|---|---|---|---|---|---|---|---|
||**Colsed-Source Models**|||||||
|||||||||
||ChatGPT-3.5<br>-<br>-<br>GPT-4<br>-<br>-<br>GPT-4 Code<br>-<br>-<br>PaLM-2<br>-<br>-|80.8<br>35.5<br>92.0<br>42.5<br>97.0<br>69.7<br>80.7<br>34.3|83.0<br>97.0<br>-<br>-||-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-||
||**Open-Source Models**|||||||
|||||||||
||Mammoth<br>CodeLlama<br>7B<br>MathCoder<br>CodeLlama<br>7B<br>ToRA<br>CodeLlama<br>7B|59.4<br>33.4<br>67.8<br>30.2<br>72.6<br>44.6||71.4<br>70.7<br>70.4|45.9<br>55.4<br>49.6<br>55.8<br>36.0<br>68.1|53.1<br>54.8<br>58.3||
||**MathGenieLM**<br>CodeLlama<br>7B<br><br><br>|71.5<br>39.7<br><br>||80.2<br>|69.1<br>69.5<br><br>|66.0<br>||
||Mammoth<br>Llama-2<br>7B<br>MathCoder<br>Llama-2<br>7B<br>ToRA<br>Llama-2<br>7B|53.6<br>31.5<br>64.2<br>23.3<br>68.8<br>40.1||67.7<br>71.5<br>68.2|41.2<br>46.3<br>47.5<br>46.9<br>29.2<br>58.3|48.1<br>50.7<br>52.9||
||**MathGenieLM**<br>Llama-2<br>7B|71.7<br>33.0||78.5|61.4<br>65.0|61.9||
||**MathGenieLM**<br>Llemma<br>7B|76.0<br>**48.3**||81.3|**85.0**<br>**76.6**|**73.4**||
||**MathGenieLM**<br>Mistral<br>7B|**80.5**<br>45.1||**83.3**|79.4<br>71.8|72.0||
||||||47.1<br>61.5<br>56.7<br>60.7<br>62.5<br>62.2<br>37.9<br>70.3<br>61.6|||
||Mammoth<br>CodeLlama<br>13B<br>MathCoder<br>CodeLlama<br>13B<br>ToRA<br>CodeLlama<br>13B|64.7<br>36.3<br>74.1<br>35.9<br>75.8<br>**48.1**||73.7<br>78.0<br>75.7|47.1<br>61.5<br>60.7<br>62.5<br>37.9<br>70.3|56.7<br>62.2<br>61.6||
||**MathGenieLM**<br>CodeLlama<br>13B|78.5<br>40.3||**84.5**|76.7<br>65.7|69.1||
||Mammoth<br>Llama-2<br>13B<br>MathCoder<br>Llama-2<br>13B<br>ToRA<br>Llama-2<br>13B|62.0<br>34.2<br>72.6<br>29.9<br>72.7<br>43.0||72.4<br>76.9<br>72.9|43.2<br>49.2<br>62.3<br>54.7<br>45.7<br>62.6|52.2<br>59.3<br>59.4||
||**MathGenieLM**<br>Llama-2<br>13B|**80.4**<br>43.8||83.5|**78.4**<br>**72.7**|**71.8**||
||||||51.8<br>65.4<br>63.6<br>65.8<br>75.9<br>70.2<br>50.2<br>77.9<br>68.0|||
||Mammoth<br>CodeLlama<br>34B<br>MathCoder<br>CodeLlama<br>34B<br>ToRA<br>CodeLlama<br>34B|72.7<br>43.6<br>81.7<br>45.2<br>80.7<br>50.8||84.3<br>82.5<br>80.5|51.8<br>65.4<br>65.8<br>75.9<br>50.2<br>77.9|63.6<br>70.2<br>68.0||
||**MathGenieLM**<br>CodeLlama<br>34B|**86.2**<br>51.4||86.9|85.8<br>78.4|77.7||
||**MathGenieLM**<br>Llemma<br>34B|84.1<br>**55.1**||**87.4**|**89.3**<br>**82.9**|**79.8**||
||||||-<br>-<br>-<br>51.4<br>55.6<br>61.6<br>77.0<br>74.4<br>73.1<br>73.3<br>72.6<br>72.5|||
||WizardMath<br>Llama-2<br>70B<br>Mammoth<br>Llama-2<br>70B<br>MathCoder<br>Llama-2<br>70B<br>ToRA<br>Llama-2<br>70B|81.6<br>22.7<br>76.9<br>41.8<br>83.9<br>45.1<br>84.3<br>49.7||-<br>82.4<br>84.9<br>82.7|-<br>-<br>51.4<br>55.6<br>77.0<br>74.4<br>73.3<br>72.6|-<br>61.6<br>73.1<br>72.5||
||**MathGenieLM**<br>Llama-2<br>70B<br>**88.4**<br>**51.2**|||**87.7**|**89.1**<br>**76.0**<br>**78.5**|||
||**MathGenieLM**<br>Mixtral<br>8x7B|87.0<br>53.7||88.9|89.9<br>81.7|80.2||
||**MathGenieLM**<br>InternLM2<br>20B|87.7<br>55.7||87.3|88.5<br>85.1|80.9||



Table 3: Results of MathGenieLM, compared to various open-source and closed-source models on 2 in-domain datasets (GSM8K, MATH), and 3 out-of-domain datasets (SVAMP, Simuleq, Mathematics). The results of closedsource models are from Yue et al. (2023) and Gou et al. (2023). 

|**Model**<br>**Base**<br>**Voting** _k_-**path**|**In-Domain**<br>**GSM8K**<br>**MATH**|**Out-of-Domain**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**|**Average**|
|---|---|---|---|
|||||
|ToRA<br>Llama-2 70B<br>✓<br>50|88.3<br>56.9|-<br>-<br>-|-|
|**MathGenieLM**<br>Llama-2 70B<br>✗<br>-<br>88.4<br>51.2<br>87.7<br>89.1<br>76.0<br>78.5||||
|||||
|**MathGenieLM**<br>Llama-2 70B<br>✓<br>10|91.7(+3.3)<br>63.3(+12.1)|94.0(+6.3)<br>95.9(+6.8)<br>87.2(+11.2)|86.4(+7.9)|



Table 4: Results of naive majority voting. _k_ -path represents the number of solutions generated for majority voting. 

**Analysis of different data composition.** We analyze the effect of adding and subtracting different parts of our training data to observe the impact of each component. As shown in the upper half of Tab. 5, when only AugGSM8K is added, the performance on GSM8K, SVAMP, and Simuleq improves, while adding AugMATH leads to more notable improvements in MATH and Mathematics. This is consistent with the types of questions in 

each dataset: GSM8K, SVAMP, and Simuleq contain grade-school level math word problems with relatively easy calculations, whereas MATH and Mathematics feature more complex math computations. When both AugGSM8K and AugMATH are added, the improvements in the datasets are compounded as well, which shows the effectiveness of our augmented data. 

## **Analysis of different amounts of augmented** 

|**Data**<br>**In-Domain**<br>**Out-of-Domain**<br>**Seed**<br>**Verifcation**<br>**AugGSM8K**<br>**AugMATH**<br>**GSM8K**<br>**MATH**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**<br>**Average**|**Data**<br>**In-Domain**<br>**Out-of-Domain**<br>**Seed**<br>**Verifcation**<br>**AugGSM8K**<br>**AugMATH**<br>**GSM8K**<br>**MATH**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**<br>**Average**|**Data**<br>**In-Domain**<br>**Out-of-Domain**<br>**Seed**<br>**Verifcation**<br>**AugGSM8K**<br>**AugMATH**<br>**GSM8K**<br>**MATH**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**<br>**Average**|**Data**<br>**In-Domain**<br>**Out-of-Domain**<br>**Seed**<br>**Verifcation**<br>**AugGSM8K**<br>**AugMATH**<br>**GSM8K**<br>**MATH**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**<br>**Average**||
|---|---|---|---|---|
|✓<br>✗<br>✗<br>✗<br>73.5<br>41.8<br>73.1<br>68.5<br>66.6<br>64.7<br>✓<br>✓<br>✗<br>✗<br>74.2<br>42.2<br>76.5<br>71.0<br>65.5<br>65.9<br>✓<br>✓<br>✓<br>✗<br>78.2<br>42.2<br>84.3<br>78.8<br>69.1<br>70.5<br>✓<br>✓<br>✗<br>✓<br>74.9<br>43.5<br>81.6<br>73.3<br>73.2<br>69.3<br>✓<br>✓<br>✓<br>✓<br>80.5<br>45.1<br>83.3<br>79.4<br>71.8<br>72.0|73.5<br>41.8|73.1<br>68.5<br>66.6|64.7||
||74.2<br>42.2|76.5<br>71.0<br>65.5|65.9||
||78.2<br>42.2|84.3<br>78.8<br>69.1|70.5||
||74.9<br>43.5|81.6<br>73.3<br>73.2|69.3||
||||||
|**Seed + Verifcation**<br>74.2<br>42.2<br>76.5<br>71.0<br>65.5<br>65.9<br>**Seed + Verifcation +** 1<br>8 **(AugGSM8K + AugMATH)**<br>75.6<br>41.6<br>80.7<br>75.3<br>67.7<br>68.2<br>**Seed + Verifcation +** 1<br>4 **(AugGSM8K + AugMATH)**<br>77.9<br>42.7<br>82.6<br>77.0<br>67.8<br>69.6<br>**Seed + Verifcation +** 1<br>2 **(AugGSM8K + AugMATH)**<br>79.2<br>44.0<br>82.1<br>80.2<br>68.8<br>70.9<br>**Seed + Verifcation + (AugGSM8K + AugMATH)**<br>80.5<br>45.1<br>83.3<br>79.4<br>71.8<br>72.0|||||
||75.6<br>41.6|80.7<br>75.3<br>67.7|68.2||
||77.9<br>42.7|82.6<br>77.0<br>67.8|69.6||
||79.2<br>44.0|82.1<br>80.2<br>68.8|70.9||
||80.5<br>45.1|83.3<br>79.4<br>71.8|72.0||



Table 5: Effect of different data composition and amounts of augmented data with Mistral-7B as the base model. 

|**Data**|**GSM8K**<br>**MATH**|**Average**|
|---|---|---|
|<br><br><br>|||
|**w/o verifcation fltering**|79.3<br>43.8|61.6|
||||
|**w/ verifcatioin fltering**|80.5(+1.2)<br>45.1(+1.3)|62.8(+1.2)|



Table 6: Effect of using or not using code-integrated verification rationales to filter the training data. 

**data.** We analyze the scaling quality of the augmented data we generated by training a model with {0 _,_[1] 8 _[,]_[1] 4 _[,]_[1] 2 _[,]_[ 1][} times the amount of augmented data.] The results, as shown in the bottom half of Tab. 5 and Fig. 3, indicate that, with an increase in the amount of augmented data, the performance on all five datasets consistently improves, with very few exceptions. This demonstrates the high scaling quality of our data. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs_images/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs.pdf-0007-05.png)


**----- Start of picture text -----**<br>
Effect of Different Amounts of AugData<br>80<br>70<br>GSM8K<br>Math<br>SVAMP<br>SimulEq<br>60<br>Mathematics<br>Average<br>50<br>40<br>0 1 1 1 1<br>8 4 2<br>Amount of AugData<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 3: Performance of the Mistral 7B model finetuned with _{_ 0 _,_[1] 8 _[,]_[1] 4 _[,]_[1] 2 _[,]_[ 1] _[}]_[times][the][amount][of][aug-] mented data. 

## **Analysis of Verification-Based Solution Filter-** 

**ing.** We analyze the effectiveness of VerificationBased Solution Filtering by using data before and after filtering with the help of verification to finetune the model. As demonstrated in Tab. 6, finetuning the model with augmented question-solution pairs filtered by verification results in noticeable accuracy increases in both GSM8K and MATH, indicating the superior quality of the augmented data after filtering and the effectiveness of VerificationBased Solution Filtering. Further analysis of the verification ability of _M_ code is shown in Tab. 11 of Appendix D. 

**Comparison with other question augmentation methods.** We compare our method with three other question augmentation methods: MetaMath (Yu et al., 2023), direct question augmentation without solution, and direct question augmentation with solution. The two direct question augmentation methods both utilize _M_ text as the question augmentation model. The former presents only the seed question to the model during ques- 

tion augmentation, while the latter presents both the question and its solution. The results, as shown in Tab. 7, indicate that our augmented solutionto-question back-translation method yields better performance than existing augmentation methods. 

## **3.4 Accuracy of Verified Inference** 

Our models have the ability to verify its own solutions when presented with prompts as shown in Tab. 10. This represents a mathematical reasoning ability that can be applied during inference. 

A simple way to do this is to verify the solutions generated and solve the problem again if the solution is verified to be incorrect. We limit the number of verification to two times. As shown in Tab. 8, applying verification twice consistently enhances accuracy across all five datasets, with notable improvements in the MATH and Mathematics datasets. The average generation (N _×_ ) presented in Tab. 8 measures the cost of verified inference, which is 2.3 _×_ on average. When compared to 3- 

|**Augmentation Method**|**In-Domain**<br>**GSM8K**<br>**MATH**|**Out-of-Domain**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**|**Average**|
|---|---|---|---|
|**MetaMath**<br>79.5<br>44.6<br>78.4<br>79.6<br>67.9<br>70.0||||
|||||
|**Direct question augmentation (w/o sol.)**|78.8<br>43.0|84.0<br>77.0<br>70.2|70.6|
|||||
|**Direct question augmentation (w/ sol.)**|79.2<br>44.2|83.6<br>72.0<br>68.6|69.5|
|||||
|**MathGenie (ours)**|**80.5**<br>**45.1**|**83.3**<br>**79.4**<br>**71.8**|**72.0**|



Table 7: Comparison of different question augmentation methods. **Direct question augmentation (w/o sol.)** presents _M_ text with only the seed questions to generate new questions. **Direct question augmentation (w/ sol.)** presents _M_ text with pairs of question-solution to generate new questions. 

||**In-Domain**<br>**GSM8K**<br>**MATH**|**Out-of-Domain**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**|**Average**|
|---|---|---|---|
|||||
|**Baseline**<br>**Accuracy**<br>**N**_×_|88.4<br>51.2<br>1_×_<br>1_×_|87.7<br>89.1<br>76.0<br>1_×_<br>1_×_<br>1_×_|78.5<br>1_×_|
|||||
|**Verify (twice)**<br>**Accuracy**<br>**N**_×_|88.6<br>55.8<br>2.1_×_<br>2.8_×_|88.7<br>91.2<br>81.1<br>2.1_×_<br>2.1_×_<br>2.3_×_|81.1<br>2.3_×_|
|||||
|**Voting (3-path)**<br>**Accuracy**<br>**N**_×_|88.6<br>53.8<br>3_×_<br>3_×_|92.0<br>90.9<br>81.5<br>3_×_<br>3_×_<br>3_×_|81.4<br>3_×_|



Table 8: Result of MathGenieLM-Llama-2-70B using verified inference. **Verify (twice)** means that, when testing the model, the solutions are verified and those verified as incorrect are solved again. This process is repeated twice. Each verification or solution is counted as 1 generation, and N _×_ is the average generation count of each question. 

path majority voting, verified inference achieves almost identical accuracy but at a significantly reduced cost. The results of more rounds of verification are analyzed in Tab. 12 of Appendix F. 

## **4 Related Works** 

**Large Language Models for Mathematical Reasoning.** LLMs have demonstrated remarkable performance in mathematical reasoning tasks. CoT (Wei et al., 2022) enhances LLMs’ capability for multistep reasoning. Self-Consistency (Wang et al., 2022) selects the final answer through majority voting. CSV (Zhou et al., 2023) introduces codebased self-verification. Other research efforts focus on pretraining or finetuning LLMs, thereby producing math-specific LLMs, such as Llemma (Azerbayev et al., 2023), WizardMath (Luo et al., 2023), Mammoth (Yue et al., 2023), ToRA (Gou et al., 2023), and MathCoder (Wang et al., 2023). It is noted that code-integrated models (Wang et al., 2023; Gou et al., 2023) have shown superior capabilities over CoT-style models. This paper develops synthetic math problems and solutions using free models to enhance mathematical reasoning. 

**Instruction-following Datasets for LLMs.** Recent studies (Taori et al., 2023; Peng et al., 2023; Mukherjee et al., 2023; Li et al., 2023b) have begun to utilize synthetic instructions generated by 

LLMs, such as GPT-4 or GPT-3.5, to distill into smaller models. WizardLM (Xu et al., 2023) proposes complex instructions to enrich the seed data for general chat models. This paper, however, focuses on math problem augmentation, particularly for code-integrated math-specific models. 

## **Data Augmentation for Mathematical Reason-** 

**ing.** To upscale the number of math problems, various works (Yu et al., 2023; Liu and Yao, 2024; Li et al., 2023a) directly augment existing problems. Differing from these approaches, our method utilizes information in the solutions through math question back-translation, thereby enhancing the reliability of the augmented questions. We also create code-integrated solutions for the questions and use verification rationales to filter the solutions. 

## **5 Conclusion** 

In this paper, we propose a coordinated pipeline consisting of _Iterative Solution Augmentation_ and _Question Back-translation_ to produce large-scale synthetic math questions, and _Verification-Based Solution Filtering_ to filter the generated codeintegrated solutions. Combined, these three components effectively create new questions and ensure the reliability of the corresponding code-integrated solutions. Experiments show that MathGenieLM achieves superior performance across five math 

problem-solving benchmarks and on six different pretrained base models, offering insights into the development of math problem-solving models and providing hope for extension to other reasoning tasks. 

## **Limitations** 

Our method requires significant GPU resources, involving the full-parameter finetuning of large language models with up to 70B parameters. Therefore, it is crucial for future studies to explore ways to reduce the required resources. Another limitation is that our models cannot process images as input, and thus lack the ability to solve problems that involve images, as discussed in (Lu et al., 2023). Additionally, our models are constrained by a limited context length, having been finetuned with a context length of 4096. These limitations are significant and merit further investigation. 

## **Ethics Statement** 

Our work, by enhancing the mathematical abilities of language models, can potentially contribute to the cause of math education. Still, our models can output untrue hallucinations, just like any language model. We have utilized various open-source models such as LLaMA-2, CodeLLaMA, Mistral, and Mixtral-8x7B, as well as open-source software such as Hugging Face and PyTorch. We adhere to the policies and licenses of these resources and acknowledge the role they have played in our work. 

## **Acknowledgement** 

This project is funded in part by National Key R&D Program of China Project 2022ZD0161100, by the Centre for Perceptual and Interactive Intelligence (CPII) Ltd under the Innovation and Technology Commission (ITC)’s InnoHK, by General Research Fund of Hong Kong RGC Project 14204021. Hongsheng Li is a PI of CPII under the InnoHK. 

## **References** 

- Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. _arXiv preprint arXiv:2305.10403_ . 

- Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. 2023. 

Llemma: An open language model for mathematics. _arXiv preprint arXiv:2310.10631_ . 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901. 

- Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. 2022. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _arXiv preprint arXiv:2211.12588_ . 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ . 

- Alex Davies, Petar Velickovic, Lars Buesing, Sam Blackwell, Daniel Zheng, Nenad Tomašev, Richard Tanburn, Peter W. Battaglia, Charles Blundell, András Juhász, Marc Lackenby, Geordie Williamson, Demis Hassabis, and Pushmeet Kohli. 2021. Advancing mathematics by guiding human intuition with ai. _Nature_ , 600:70 – 74. 

- Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen, et al. 2023. Tora: A tool-integrated reasoning agent for mathematical problem solving. _arXiv preprint arXiv:2309.17452_ . 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ . 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. _arXiv preprint arXiv:2310.06825_ . 

- Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ . 

- Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. 2016. Mawps: A math word problem repository. In _North American Chapter of the Association for Computational Linguistics_ . 

- Chengpeng Li, Zheng Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang Wang, and Chang Zhou. 2023a. Query and response augmentation cannot help out-of-domain math reasoning generalization. _arXiv preprint arXiv:2310.05506_ . 

- Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. 2023b. Self-alignment with instruction backtranslation. _arXiv preprint arXiv:2308.06259_ . 

- Haoxiong Liu and Andrew Chi-Chih Yao. 2024. Augmenting math word problems via iterative question composing. _arXiv preprint arXiv:2401.09003_ . 

- Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ . 

- Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. 2023. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. _arXiv preprint arXiv:2310.02255_ . 

- Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. _arXiv preprint arXiv:2308.09583_ . 

- Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. 2023. Orca: Progressive learning from complex explanation traces of gpt-4. _arXiv preprint arXiv:2306.02707_ . 

- OpenAI. 2023. Gpt-4 technical report. _ArXiv_ , abs/2303.08774. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are nlp models really able to solve simple math word problems? _arXiv preprint arXiv:2103.07191_ . 

- Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. 2023. Instruction tuning with gpt-4. _arXiv preprint arXiv:2304.03277_ . 

- Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. 2023. Code llama: Open foundation models for code. _arXiv preprint arXiv:2308.12950_ . 

- Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, et al. 2023. Beyond human data: Scaling self-training for problem-solving with language models. _arXiv preprint arXiv:2312.06585_ . 

- Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford_alpaca. 

- InternLM Team. 2023. Internlm: A multilingual language model with progressively enhanced capabilities. https://github.com/InternLM/ InternLM-techreport. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ . 

- Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. 2023. Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning. In _International Conference on Learning Representations_ . 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ . 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837. 

- Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. _arXiv preprint arXiv:2304.12244_ . 

- Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. Metamath: Bootstrap your own mathematical questions for large language models. _arXiv preprint arXiv:2309.12284_ . 

- Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. 2023. Scaling relationship on learning mathematical reasoning with large language models. _arXiv preprint arXiv:2308.01825_ . 

- Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. Mammoth: Building math generalist models through hybrid instruction tuning. _arXiv preprint arXiv:2309.05653_ . 

- Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, and Hongsheng Li. 2023. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. In _International Conference on Learning Representations_ . 

## **A Example of Iterative Solution Augmentation and Question Back-translation** 

Fig. 4 (b) shows an example of three rounds of Iterative Solution Augmentation and Question Backtranslation. The seed solution is iteratively augmented, and the augmented solutions are backtranslated into new questions. Compared to Fig. 4 (a), where augmentation is conducted directly on the question, Iterative Solution Back-translation demonstrates greater diversity in question phrasing, as the original question is not directly provided to the model. 

## **B Analysis of Iterative Solution Augmentation** 

To effectively demonstrate the impact of iteration on enhancing solution quality, we generated an equal number of augmented solutions without employing iteration, by directly augmenting new solutions from the original set. As illustrated in Table 9, the outcomes of the experiment conducted without iteration in solution augmentation were significantly inferior compared to those with iteration. This underscores the beneficial role of iteration in solution augmentation, primarily attributed to its potential to enhance the diversity of the solutions. 

## **C Prompt of Verification** 

Tab. 10 presents the prompt format used in finetuning and generating code-integrated verification rationales. 

## **D Analysis of Code-integrated Verification** 

To understand the reason behind the improved quality of the data, we quantify the ability of _M_ code to conduct code-integrated verification by testing it on the solutions generated by _M_ code across the five testing datasets. We utilize these testing datasets because they contain ground truth, which enables us to assess the actual correctness of the solutions. 

We define two metrics below to demonstrate the _M_ code’s ability to verify its solutions: Precision and Recall. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs_images/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs.pdf-0011-09.png)


TP represents the cases in which the verification proves the solution correct and the solution’s answer is actually correct. FP represents the cases in which the verification proves the solution correct, yet the solution’s answer is actually wrong. TN represents the cases in which the verification proves the solution wrong, yet the solution’s answer is actually correct. In short, Precision answers the question, “What proportion of verified TRUE answers are actually correct?”, while Recall answers the question, “What proportion of actual correct answers were verified TRUE?”. Given these definitions, Precision reflects the reliability of the retained code-integrated solutions, while Recall reflects the efficiency of the filtering step. 

Tab. 11 shows that Precision is significantly higher than Accuracy across all datasets, underscoring the effectiveness and generalization capabilities of code-integrated verification. 

## **E Finetuning Details** 

In this work, we finetune all models using the HuggingFace library. We employ a cosine weight scheduler with a learning rate of 2 _e[−]_[5] , designating the first 50 steps as warm-up steps. All models are optimized using AdamW (Loshchilov and Hutter, 2017) with a batch size of 64. The 70B and 34B models are fine-tuned on 32 NVIDIA A800 80GB GPUs. Mistral-8x7B is fine-tuned on 16 NVIDIA A800 80GB GPUs, while the 7B, 13B, and 20B models are all fine-tuned on 8 NVIDIA A800 80GB GPUs. 

## **F Analysis of Verification Rounds in Verified Inference** 

In verified inference, we verify the solutions of the test questions and re-solve only those questions whose solutions are verified as incorrect. Consequently, the number of questions that need solving decreases with each round. Theoretically, this process can continue until all questions have solutions verified as correct. However, in practice, verifying too many rounds can lead to additional costs without any improvement in accuracy, as some questions may be beyond the model’s problemsolving and verification capabilities. To determine the trade-off between cost and accuracy, we increased the number of verification rounds to 9. The results are shown in Tab. 12. As can be seen, the increase in average accuracy becomes small after 2 rounds of verification. 

||**In-Domain**<br>**GSM8K**<br>**MATH**|**Out-of-Domain**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**|**Average**|
|---|---|---|---|
|**w/ iteration**<br>80.5<br>45.1<br>83.3<br>79.4<br>71.8<br>72.0||||
|**w/o iteration**<br>78.7(-1.8)<br>45.0(-0.1)<br>82.9(-0.4)<br>76.1(-3.3)<br>70.9(-0.9)<br>70.7(-1.3)||||



Table 9: Comparison between with iteration or without iteration when conducting solution augmentation. The models are finetuned on Mistral 7B. 

## **Prompt** 

**Question**: {question} **Solution**: {solution} Above is a math problem and its solution. Please use code to verify the solution above. 

Table 10: Prompt used for back-translation. {question} is replaced with a math question, while {solution} is replaced with the its code-integrated solution. 

## **G Examples of Code-Integrated Verification Rationales** 

Two examples of code-integrated verification rationales are presented in Tab. 13 and Tab. 14. In Tab. 13, the solution is verified as correct by using the answer to calculate the condition and comparing it with the actual condition. In Tab. 14, the solution is verified as incorrect by solving the question through an alternative method and comparing the answers. 

|**Metrics**|**GSM8K**<br>**MATH**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**|**Average**|
|---|---|---|
|**Accuracy**<br>86.4<br>49.5<br>86.0<br>88.1<br>73.6<br>76.7|||
||||
|**Precision**<br>**Recall**|89.3(+2.9)<br>64.5(+15.0)<br>88.3(+2.3)<br>90.8(+2.7)<br>79.9(+3.2)<br>98.5<br>94.7<br>97.4<br>98.2<br>97.1|82.6(+5.9)<br>97.2|



Table 11: Accuracy, Precision and Recall of _M_ code with and without verification. Accuracy is the percent of solutions that are correct. Precision is the percent of solutions actually correct among the solutions verified to be correct. Recall is the percent of solutions verified to be correct among the solutions actually correct. 

||**GSM8K**<br>**MATH**<br>**SVAMP**<br>**Simuleq**<br>**Mathematics**|**Average**|
|---|---|---|
|**Baseline**<br>**Accuracy**<br>88.4<br>51.2<br>87.7<br>89.1<br>76.0<br>78.5<br>**N**_×_<br>1_×_<br>1_×_<br>1_×_<br>1_×_<br>1_×_<br>1_×_|||
|**Verify (k=1)**<br>**Accuracy**<br>88.4<br>54.2<br>88.4<br>90.9<br>79.2<br>80.2<br>**N**_×_<br>2.0_×_<br>2.3_×_<br>2.0_×_<br>2.0_×_<br>2.1_×_<br>2.1_×_|||
|**Verify (k=2)**<br>**Accuracy**<br>88.6<br>55.8<br>88.7<br>91.2<br>81.1<br>81.1<br>**N**_×_<br>2.1_×_<br>2.8_×_<br>2.1_×_<br>2.1_×_<br>2.3_×_<br>2.3_×_|||
|**Verify (k=3)**<br>**Accuracy**<br>88.6<br>56.3<br>88.8<br>91.2<br>81.3<br>81.2<br>**N**_×_<br>2.1_×_<br>3.1_×_<br>2.1_×_<br>2.1_×_<br>2.3_×_<br>2.3_×_|||
|**Verify (k=4)**<br>**Accuracy**<br>88.6<br>56.3<br>89.0<br>91.4<br>81.4<br>81.3<br>**N**_×_<br>2.1_×_<br>3.3_×_<br>2.1_×_<br>2.1_×_<br>2.4_×_<br>2.4_×_|||
|**Verify (k=5)**<br>**Accuracy**<br>88.6<br>56.5<br>88.9<br>91.4<br>81.4<br>81.4<br>**N**_×_<br>2.1_×_<br>3.4_×_<br>2.1_×_<br>2.1_×_<br>2.4_×_<br>2.4_×_|||
|**Verify (k=6)**<br>**Accuracy**<br>88.6<br>56.8<br>88.9<br>91.4<br>81.4<br>81.4<br>**N**_×_<br>2.1_×_<br>3.6_×_<br>2.1_×_<br>2.1_×_<br>2.4_×_<br>2.5_×_|||
||||
|**Verify (k=7)**<br>**Accuracy**<br>**N**_×_|88.6<br>56.8<br>88.9<br>91.4<br>81.4<br>2.1_×_<br>3.7_×_<br>2.1_×_<br>2.1_×_<br>2.4_×_|81.4<br>2.5_×_|
||||
|**Verify (k=8)**<br>**Accuracy**<br>**N**_×_|88.6<br>57.1<br>88.9<br>91.4<br>81.4<br>2.1_×_<br>3.7_×_<br>2.1_×_<br>2.1_×_<br>2.4_×_|81.5<br>2.5_×_|
||||
|**Verify (k=9)**<br>**Accuracy**<br>**N**_×_|88.6<br>57.1<br>88.9<br>91.4<br>81.4<br>2.1_×_<br>3.8_×_<br>2.1_×_<br>2.1_×_<br>2.4_×_|81.5<br>2.5_×_|



Table 12: Results of MathGenieLM-Llama-2-70B with varying numbers of verification rounds during inference. Here, “k” represents the maximum number of verification rounds. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs_images/acl_2024_MathGenie_Generating_Synthetic_Data_with_Question_Back-translation_for_Enhancing_Mathematical_Reasoning_of_LLMs.pdf-0014-00.png)


**----- Start of picture text -----**<br>
(a) Seed Question: During April, the baker sold 453 cakes at $12 and 126 pies at $7.<br>How much did the baker earn?<br>Round 1 Question: In June, the painter painted 453 houses at $12 each and 126<br>paintings at $7 each. How much did the painter earn?<br>Round 2 Question: In June, the chef cooked 453 dishes at $12 each and 126<br>desserts at $7 each. How much did the chef earn?<br>Round 3 Question: In June, the zoologist observed 453 species at $12 each and<br>126 species at $7 each. How much did the zoologist earn?<br>(b)<br>Seed Question and Solution<br>Seed Solution: The baker sold cakes for 453 * 12  Seed Question: During April, the baker sold 453<br>= $5436. And pies for 126 * 7 = $882. In total the  cakes at $12 and 126 pies at $7. How much did<br>baker earned 5436 + 882 = $6318. #### 6318 the baker earn?<br>Round 1 Solution: The florist sold bouquets for 88  Round 1 Question: A florist is selling bouquets for<br>* 8 = $704. And wreaths for 98 * 9 = $882. In total  $8 and wreaths for $9. If the florist sold 88<br>the florist earned 704 + 882 = $1586. #### 1586 bouquets and 98 wreaths, how much did he earn?<br>Round 2 Solution: The chef prepared soups for 20  Round 2 Question: A chef made 20 soups and 40<br>* 2 = $40. And cookies for 40 * 4 = $160. In total,  cookies. Each soup costs $2 and each cookie<br>the chef earned $200. #### 200 costs $4. How much did the chef earn?<br>Round 3 Solution: The farmer harvested fruits for  Round 3 Question: A farmer harvested 10 baskets<br>10 * 3 = $30. And vegetables for 5 * 5 = $25. In  of fruits and 5 baskets of vegetables. If a basket<br>total, the farmer earned $55. of fruits costs $3 and a basket of vegetables<br>costs $5, how much did the farmer get paid for<br>his harvest?<br>**----- End of picture text -----**<br>


Figure 4: An example of three iterations of (a) Direct Question Augmentation and (b) Iterative Solution Augmentation and Question Back-translation. 

**Input** : **Question**: Express 0 _._ 54 as a fraction in lowest terms. **Solution**: 

To express the repeating decimal 0 _._ 54 as a fraction in lowest terms, we can use algebraic manipulation. Let _x_ = 0 _._ 54. Step 1: Multiply _x_ by 100 to shift the decimal point. ...... Therefore, 0 _._ 54 can be expressed as 11[6][in lowest terms.] Above is a math problem and its solution. Please use code to verify the solution above. 

## **Output** : 

Now, let’s verify the answer by converting 11[6][back to a decimal.] 

# Convert the fraction 6/11 to a decimal decimal_value = float (simplified_fraction) decimal_value 

Result: 0.5454545454545454 The decimal representation of 11[6][is approximately][ 0] _[.]_[5454545454545454][, which matches the repeat-] ing decimal 0 _._ 54 Thus, our answer is verified and correct. 

Table 13: An example of code-integrated verification rationale in which the solution is verified as correct. 

## **Input** : 

**Question**: 

What is 21213 _−_ 2123? Express your answer in base 3. **Solution**: Alright, let’s solve this step by step. 

To subtract these numbers, we’ll start from the rightmost digit (least significant) and move to the left, just like we do in base 10 subtraction. 

...... 

The result of 21213 _−_ 2123 is 2223 . Above is a math problem and its solution. Please use code to verify the solution above. 

## **Output** : 

Now, let’s verify our answer by converting both numbers to base 10, performing the subtraction, and then converting the result back to base 3 to see if it matches our answer. 

# Convert the numbers from base 3 to base 10 def base3_to_base10(num_str): num_str = num_str [::-1] # reverse the string for easy processing base10_value = 0 for i, digit in enumerate (num_str): base10_value += int (digit) * (3 ** i) ...... 

# Convert 2121_3 and 212_3 to base 10 num1_base10 = base3_to_base10( "2121" ) num2_base10 = base3_to_base10( "212" ) # Perform the subtraction in base 10 result_base10 = num1_base10 - num2_base10 # Convert the result back to base 3 result_base3 = base10_to_base3(result_base10) result_base3 

Result: 1202 

It seems there was an error in our calculations. The correct result for 21213 _−_ 2123 is 12023, not 2223. I apologize for the oversight. 

Table 14: An example of code-integrated verification rationale in which the solution is verified as wrong. 

