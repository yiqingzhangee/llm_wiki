# **MAPO: Advancing Multilingual Reasoning through Multilingual-Alignment-as-Preference Optimization** 

**Shuaijie She , Wei Zou , Shujian Huang**[*] **, Wenhao Zhu Xiang Liu** , **Xiang Geng** , **Jiajun Chen** 

National Key Laboratory for Novel Software Technology, Nanjing University 

{shesj,zouw}@smail.nju.edu.cn, huangsj@nju.edu.cn 

{zhuwh,liuxiang,gx}@smail.nju.edu.cn, chenjj@nju.edu.cn 

## **Abstract** 

Though reasoning abilities are considered language-agnostic, existing LLMs exhibit inconsistent reasoning abilities across different languages, e.g., reasoning in the dominant language like English is superior to other languages due to the imbalance of multilingual training data. To enhance reasoning abilities in non-dominant languages, we propose a Multilingual-Alignment-as-Preference Optimization framework (MAPO), aiming to align the reasoning processes in other languages with the dominant language. Specifically, we harness an off-the-shelf translation model for the consistency between answers in non-dominant and dominant languages, which we adopt as the preference for optimization, e.g., Direct Preference Optimization (DPO) or Proximal Policy Optimization (PPO). Experiments show that MAPO stably achieves significant improvements in the multilingual reasoning of various models on all three benchmarks (MSVAMP +16.2%, MGSM +6.1%, and MNumGLUESub +13.3%), with improved reasoning consistency across languages[1] . 

## **1 Introduction** 

The reasoning ability of large-scale models has attracted much attention (Cobbe et al., 2021; Wang et al., 2022; Wei et al., 2023; Yu et al., 2023). Though we consider reasoning to be languageagnostic, existing studies (Chen et al., 2023) show that due to the imbalance of pre-training and finetuning data across languages, the mathematical reasoning ability of existing large-scale models varies across languages, e.g., English, is far superior to that of the other languages. 

To improve the reasoning ability in other languages, Chen et al. (2023) translated the English 

> *Corresponding author 

> 1The project will be available at https://github.com/ NJUNLP/MAPO 

reasoning processes into other languages for supervised fine-tuning (Ouyang et al., 2022, SFT). Although SFT brings in preliminary capability for multilingual reasoning, we argue that two problems hinder further improvement. 

Firstly, the annotation for the reasoning process is expensive to obtain even for the dominant language, and the reasoning processes involve complex mathematical reasoning, which may result in translation errors (Chen et al., 2023). As a result, the translated reasoning annotation for nondominant languages is limited in both scale and quality. Without sufficient and diverse data, the results of supervised training are limited. It may also suffer from generalization issues in versatile task scenarios (Zheng et al., 2023; Ouyang et al., 2022), indicating potential difficulty in adapting to the out-of-domain test set. 

More importantly, although supervised training with translated reasoning processes improves the reasoning ability for almost all trained languages, this strategy only fills in the missing reasoning annotation for non-dominant languages that originate from the dominant language. Thus the inherent gap between dominant and non-dominant languages is hardly narrowed. 

Different from existing attempts, we propose to use the reasoning ability of the dominant language as the director for improving non-dominant languages. As the reasoning process is critical for obtaining the correct result, the multilingual reasoning ability may be improved by encouraging the reasoning in non-dominant languages to be similar to that in the dominant language. Therefore, we propose a **M** ultilingual- **A** lignment-as- **P** reference **O** ptimization (MAPO) framework by aligning the reasoning process of non-dominant languages to the dominant. Notably, MAPO exploits the strong reasoning ability in the dominant language and requires no annotation for the reasoning process. 

More specifically, MAPO consists of two stages: 

1 

preference estimation via multilingual alignment and preference optimization. During preference estimation, the reasoning processes to the same question are sampled from the LLM in both dominant and non-dominant languages. A well-trained, off-the-shelf translation model is employed to yield the translation probability between the reasoning in dominant and non-dominant languages. Since higher translation probability indicates a more consistent reasoning aligned with the dominant language, the corresponding reasoning in the nondominant language is considered better and shall be promoted. During preference optimization, Proximal Policy Optimization (Schulman et al., 2017, PPO) and Direct Preference Optimization (Rafailov et al., 2023, DPO) are adopted to optimize the previously estimated preference. So the LLMs are trained to reason in the non-dominant languages as they do in the dominant language. We also conduct Iterative DPO for further preference optimization. 

Experiments are conducted on three challenging multilingual reasoning test sets, namely MSVAMP (Chen et al., 2023), MGSM (Shi et al., 2022), and MNumGLUESub constructed from NumGLUE (Mishra et al., 2022), each covering 10 languages. MAPO achieves accuracy improvements of up to 16.2%, 6.1%, and 13.3% on the three benchmarks, respectively, reaching state-of-the-art performance on 7B models, thereby demonstrating the effectiveness of our method. Among them, the improvement on the out-of-domain dataset MSVAMP (+16.2%) shows that, by enhancing multilingual reasoning consistency, MAPO improves the multilingual reasoning capabilities of the model in a generalizable manner. 

## **2 Preliminary** 

## **2.1 Multilingual Reasoning** 

A straightforward measurement to evaluate the efficacy of large language models (LLMs) lies in their proficiency in tackling complex reasoning, e.g., their performance in solving mathematical reasoning. Recent work (Wei et al., 2023; Wang et al., 2022) has verified a substantial improvement when LLMs are guided through a step-by-step reasoning process, instead of conducting a direct answer. 

Some research has introduced mathematical reasoning datasets in the form of application problems, such as GSM8K (Cobbe et al., 2021), NumGLUE (Mishra et al., 2022), and SVAMP (Patel et al., 2021). To evaluate the multilingual rea- 

soning capabilities of LLMs, Shi et al., 2022 propose MGSM by manually translating 250 samples of the GSM8K test set from English to other languages. Subsequent researches focus on enhancing the multilingual reasoning of LLMs. Chen et al., 2023 translate the GSM8K training data into other languages for supervised fine-tuning (SFT), which improves the model’s multilingual reasoning capabilities. However, SFT suffers from data scarcity and catastrophic forgetting. Its out-of-domain generalization is also hard to guarantee (Zheng et al., 2023). Multilingual reasoning via LLMs remains an open challenge. 

## **2.2 Preference Optimization** 

SFT maximizes the likelihood of annotated outputs and equips models with preliminary capabilities. However, models still exhibit various issues after SFT. Some researchers have further adjusted model behaviors and enhanced model capabilities through preference optimization. 

RLHF (Zheng et al., 2023; Bai et al., 2022b) further rectifies these model behaviors via reinforcement learning by preference. RLHF introduces a reward model _rθ_ ( _x, y_ ) given input _x_ with its corresponding output _y_ that captures the preference nuance from the human feedback. Then, _rθ_ ( _x, y_ ) scores arbitrary LLM outputs _y_ given input _x_ for iterative policy rectifications during proximal policy optimization (Schulman et al., 2017, PPO). The tuning is guided by Eq 1 to maximize the expected rewards of the LLM policy _πθ_ with the minimum deviation from the SFT policy: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization_images/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization.pdf-0002-10.png)


where _π_ SFT is the original LLM via SFT, _γ_ is a hyperparameter that constrains policy updates. 

Though RLHF via PPO is effective in adapting LLMs to versatile human preferences, it involves four sub-models, making the training complex and costly. DPO (Rafailov et al., 2023) proposes to distill a referential SFT policy _π_ ref by polarizing the preference. DPO tuning involves a pair of outputs ( _yw, yl_ ) in Eq 2: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization_images/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization.pdf-0002-13.png)


where _yw_ is favored over _yl_ , and _β_ is a hyperparameter. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization_images/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization.pdf-0003-00.png)


**----- Start of picture text -----**<br>
182名学生建议在菜单中加入土豆泥，而其他人建议加 182 students suggested adding mashed potatoes while others 182名学生建议在菜单中加入土豆泥，而其他人建议<br>MultiLingual 入培根。如果有比建议加入土豆泥的学生多166人建议 suggested adding bacon to the menu .if 166 more students 加入培根。如果有比建议加入土豆泥的学生多166人<br>Questions 加入培根，有多少学生建议加入培根？ suggested adding bacon than those that suggested mashed 建议加入培根，有多少学生建议加入培根？<br>potatoes how many students suggested bacon ?<br>Policy Model<br>Outputs Sampled<br>建议加入土豆泥的学生比建议培根的学生多166 ...<br>(There were 166 more students who suggested The number of students who suggested adding<br>adding mashed potatoes than bacon) mashed potatoes is 182....<br>如果建议加入土豆泥的学生多(If more students suggest adding mashedpotatoes 63 suggest adding bacon)63人建议加入培根 ... We know that the number of students who... (The number of students recommended to add建议加入土豆泥的学生人数是mashed potatoes is 182)182 ...<br>建议加入土豆泥的学生人数是182 ...<br>Suggesting mashed potatoes were 182 students<br>(The number of students recommended to addmashed potatoes is 182) ...<br>Estimate Alignment between<br>Non English Outputs and<br>English Outputs<br>Preference<br>Collect Alignment RC ＞ RB ＞ RA RS Optimization<br>Preference Ranking Pair<br>Preference Optimization<br>DPO Process PPO Process<br>**----- End of picture text -----**<br>


Figure 1: Illustration of our alignment framework. For brevity, we only show three sampling results and simplified optimization processes of DPO and PPO. The green and blue colors represent the same problem in Chinese and English, respectively. The white robots represent the original policy model and the colored robots represent the policy model with parameters updated through preference optimization. 

## **3 Method** 

## **3.1 Preference Estimation** 

Intuitively, reasoning ability is language-agnostic. However, LLM reasoning varies across different languages, where we consider a dominant language provides the better reference in reasoning for lesser languages. Therefore, a straightforward approach is to align reasoning in non-dominant languages to a dominant language. That is, we can rectify LLM’s output reasoning in non-dominant languages by favoring those answers more aligned to the dominant language. 

We simply adopt a well-trained multilingual translation model to calculate the ‘alignment’ scores between answers in non-dominant and dominant languages as a preference. Since the translation model is optimized on a large scale parallel data in source and target languages ( _Y, Y_[¯] ) by maximizing conditional generation probability in Eq 3: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization_images/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization.pdf-0003-06.png)


with more aligned target language _Y_ ¯ to _Y_ , 

the higher the conditional generation probability _P_ ( _Y_[¯] _|Y_ ). Then, we input the answers in nondominant languages and force-decode a corresponding answer _Y_[¯] in dominant language for the ‘alignment’ score. Correspondingly, answers with higher ‘alignment’ scores in non-dominant languages are preferred during the following preference optimizations. 

## **3.2 Preference Optimization** 

With ‘alignment’ scores as a preference, we adopt two state-of-the-art preference optimizations: PPO and DPO, to optimize the alignment scores of reasoning in non-dominant languages given that in a dominant language. To make life easier, since English is dominant in both reasoning and translation, we adopt English as the dominant language in the rest of the paper. 

## **3.2.1 Optimization with PPO** 

In our setup, the ‘alignment’ score _P_ ( _Y_[¯] _|Y_ ) given query _x_ depicts the preference as the reward model _rθ_ ( _x, Y_ ) do in PPO, thus we directly adopt the multilingual machine translation model as the reward 

3 

model. During each optimization, we collect a batch of non-English outputs _{Yi}_ with an English output _Y_[¯] , respectively. PPO then maximizes the expected alignment score within the batch by Eq 4: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization_images/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization.pdf-0004-01.png)


## **3.2.2 Optimization with DPO** 

DPO involves an input pair ( _Yw, Yl_ ), where _Yw_ is favored over _Yl_ . Correspondingly, we collect _n_ outputs within one non-English language, making � _n_ 2� pairs of ( _Yw, Yl_ ) ranked by the alignment scores given the same English answer _Y_[¯] . We additionally copy and freeze the original supervised fine-tuned LLM as the reference policy _π_ ref in Eq 2. Finally, our DPO is optimized by Eq 5: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization_images/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization.pdf-0004-04.png)


## **3.2.3 Iterative DPO** 

Recent research such as LLaMA2 (Touvron et al., 2023) and Claude (Bai et al., 2022a) suggests that preference data updated through multiple iterative rounds can is beneficial for the performance of preference optimization. 

Therefore, we further conduct Iterative DPO to optimize reasoning alignments across languages. Specifically, assume that initially when we train _πθi_ on preference data sampled by itself, we obtain _πθ_ ( _i_ +1). Subsequently, we continue to sample based on _πθ_ ( _i_ +1) and update it further on the newly sampled preference data. 

## **4 Experiment** 

||Dataset|Type|Number|
|---|---|---|---|
||MGSM|In-Domain|2,500|
||MNumGLUESub|In-Domain|5,530|
||MSVAMP|Out-of-Domain|10,000|



Table 1: Statistics of three benchmarks. “Type" indicates whether its corresponding training set was used during the training. “ Number” refers to the total number of samples in the test data. 

**Preference Estimation** : To obtain the mathematical problems for preference optimization, we selected tasks 1, 4, and 8 from the eight tasks in NumGLUE (Mishra et al., 2022), which is a multitask arithmetic reasoning benchmark, and translate the questions into the same nine languages as in MGSM8KInstruct, resulting the dataset MNumGLUESub. MNumGLUESub with only 1700 problems. When constructing the preference dataset for DPO, we use the corresponding base model for sampling and calculate the alignment feedback using NLLB-600M-distilled[4] . 

**Preference Optimization** : We experimented with both PPO and DPO. For simplicity, we report the results of the third iteration of DPO by default. For detailed results of PPO-LoRA and each round in Iterative DPO, please refer to Appendix B. 

**Evaluation Datasets** : We utilize three challenging benchmarks: MSVAMP, MGSM (Shi et al., 2022) and MNumGLUESub (statistics are shown in Table 1). Among them, MSVAMP, serving as an Out-of-domain test set that does not participate in the training, is used to analyze the model’s robustness and generalization ability. MGSM is the testset corresponds to MGSM8KInstruct, on which the base models are trained. MNumGLUESub is the testset corresponds to the data for preference estimation and optimization. 

## **Evaluation Metric** : 

## **4.1 Experiment Setup**[2] 

**Base Models:** Our experiments include four base models. **MathOctopus-7/13B** is finetuned from LLaMA2-7/13B with the multilingual reasoning processes MGSM8KInstruct[3] (Chen et al., 2023). Yu et al., 2023 proposed MetaMath-7/13B and MetaMath-Mistral7B, which have stronger English reasoning abilities. We fine-tune these models with MGSM8KInstruct and get **MetaMathOctopus7/13B** and **MistralMathOctopus-7B** . 

> 2More training details can refer to Appendix A 

> 3GSM8K translated to nine non-English languages. 

- **Accuracy** : We use the accuracy of problemsolving to measure the model’s reasoning ability, with a higher accuracy representing stronger reasoning ability. 

- **PPL (PPL-based Alignment Score)** : We input the target non-English answer and apply NLLB-600M-distilled to get the perplexity of the given English answer. Less perplexity indicates better alignment between the reasoning processes. 

- **ACR (Answer Consistency Ratio)** : Let _m_ 

4https://huggingface.co/facebook/ nllb-200-distilled-600M 

4 

|**Model**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**Ru**<br>**De**<br>**Es**<br>**Fr**<br>**En**|**Avg**|
|---|---|---|
||||
|GPT-3.5-Turbo|33.7<br>42.9<br>46.2<br>45.6<br>46.2<br>48.9<br>50.4<br>50.7<br>50.9<br>50.5|46.6|
||||
|MAmmoTH 7B_†_<br>WizardMath 7B_†_<br>MetaMath 7B|4.3<br>6.3<br>4.2<br>26.7<br>26.8<br>33.7<br>39.6<br>42.9<br>39.9<br>45.1<br>16.1<br>17.0<br>10.3<br>37.9<br>36.3<br>37.4<br>39.2<br>44.8<br>37.7<br>48.5<br>14.8<br>17.7<br>14.8<br>51.8<br>54.4<br>59.4<br>59.6<br>63.2<br>62.0<br>64.8|26.3<br>32.5<br>46.2|
||||
|MathOctopus 7B<br>+ m-RFT<br>+ MAPO-DPO(ours)|27.7<br>35.9<br>39.4<br>41.6<br>42.7<br>44.2<br>44.0<br>45.1<br>45.3<br>46.4<br>37.9<br>46.4<br>46.4<br>49.6<br>50.8<br>50.4<br>50.7<br>51.6<br>53.4<br>49.4<br>48.8<br>55.2<br>56.0<br>60.3<br>58.8<br>58.3<br>58.1<br>59.7<br>60.8<br>58.4|41.2<br>48.7<br>57.4|
||||
|MetaMathOctopus 7B<br>+ m-RFT<br>+ MAPO-DPO(ours)|36.1<br>47.5<br>49.4<br>51.3<br>54.5<br>53.6<br>56.6<br>60.0<br>57.2<br>64.2<br>44.8<br>54.2<br>56.2<br>58.4<br>57.7<br>56.2<br>59.2<br>59.3<br>57.8<br>63.1<br>50.1<br>61.6<br>61.7<br>65.9<br>65.7<br>64.8<br>68.4<br>68.5<br>68.6<br>71.6|53.0<br>56.7<br>64.7|
||||
|MistralMathOctopus 7B<br>47.0<br>52.6<br>54.4<br>58.8<br>60.2<br>59.8<br>62.1<br>60.8<br>60.4<br>74.4<br>59.0<br>+ m-RFT<br>57.4<br>63.5<br>60.8<br>65.2<br>69.8<br>67.4<br>67.2<br>69.1<br>68.0<br>71.3<br>66.0<br>+ MAPO-DPO(ours)<br>**62.9**<br>**71.3**<br>**71.4**<br>**73.7**<br>**76.0**<br>**74.9**<br>**77.8**<br>**78.1**<br>**79.0**<br>**81.1**<br>**74.6**|||
||||
|MAmmoTH 13B_†_<br>WizardMath 13B_†_<br>MetaMath 13B|5.0<br>13.7<br>12.9<br>42.2<br>47.7<br>50.7<br>52.3<br>53.9<br>53.8<br>53.4<br>13.7<br>16.3<br>12.5<br>29.5<br>37.0<br>43.8<br>48.7<br>49.4<br>49.4<br>56.3<br>14.6<br>15.7<br>17.4<br>57.0<br>56.6<br>63.7<br>67.3<br>65.9<br>64.7<br>67.7|38.6<br>35.7<br>49.1|
||||
|MathOctopus 13B<br>+ m-RFT<br>+ MAPO-DPO(ours)|35.2<br>46.8<br>42.8<br>43.2<br>48.8<br>47.6<br>44.4<br>48.0<br>48.4<br>53.2<br>43.4<br>50<br>52.1<br>54.9<br>55.4<br>57.1<br>59.2<br>56.4<br>59.5<br>55.2<br>51.8<br>58.9<br>56.4<br>60.4<br>58.8<br>62.1<br>63.5<br>62.0<br>61.7<br>65.0|45.8<br>54.3<br>60.1|
||||
|MetaMathOctopus 13B<br>+ m-RFT<br>+ MAPO-DPO(ours)|41.6<br>52.1<br>50.9<br>57.3<br>53.1<br>59.1<br>60.1<br>61.1<br>60.8<br>66.8<br>48.1<br>59.6<br>61.4<br>60.5<br>58.9<br>61.0<br>62.7<br>65.3<br>64.3<br>65.4<br>**54.7**<br>**64.7**<br>**62.9**<br>**69.0**<br>**68.2**<br>**68.2**<br>**69.5**<br>**70.6**<br>**71.3**<br>**70.5**|56.3<br>60.7<br>**67.0**|



Table 2: Model Performances on MSVAMP test set. “Avg” represents the average performance on ten languages and bold text denotes the best results within the same size. Results marked with _[†]_ come from Chen et al., 2023. 

denote the set of math questions answered correctly in English, and _n_ denote those answered correctly in non-English. ACR is then calculated as: ACR = _[|][m][∩][n][|]_[Higher][ACR] _|n|_[.] indicates a greater degree of overlap in reasoning capabilities between non-English and English languages. 

**Baselines** : The selected base models are already strong baselines, which have been fine-tuned on MGSM8KINSTRUCT. Moreover, motivated by Rejection sampling Fine-Tuning (RFT) (Yuan et al., 2023), we employ another strong baseline **m-RFT** , where the solutions which yield correct answers during sampling are used to further fine-tune the base model. To alleviate catastrophic forgetting, we fine-tune the model for only one epoch with a minor learning rate (1e-5). For comparison, we also incorporated other recent LLaMA2-base models: **MAmmoTH** (Yue et al., 2023) is developed by utilizing diverse datasets for math instruction, while **WizardMath** (Luo et al., 2023) employs Reinforcement Learning from Evol-Instruct (RLEIF). **MetaMath** 7B (Yu et al., 2023) is fine-tuned from the strongest English reasoning dataset, MetaMathQA. 

## **5 Experiment Results**[5] 

## **5.1 Preference Optimization Improves Multilingual Reasoning Effectively** 

Experimental results in Table 2,3,7 consistently demonstrate that our method has effectively enhanced the reasoning capabilities of the various base models and achieved state-of-the-art performance. More specifically, the improvement is most impressive on the out-of-domain dataset MSVAMP, where we achieved an average accuracy improvement of 16.2% and 14.7% MathOctopus7B and MathOctopus13B, respectively. Even for the most powerful 7B model MistralMathOctopus 7B, our method can further boost its performance to an impressive 74.6%. We also observe significant improvements in MGSM and MNumGLUESub. 

From the perspective of languages, most languages have improvement after alignment and it is more significant in some low-resource languages that previously under-performed. Take MathOctopus 7B as an example, our method has increased the accuracy on MSVAMP for Bengali, Thai, Swahili, and Japanese by 21.1%, 19.3%, 16.6%, and 18.7%, 

> 5To facilitate further research, we launch a multilingual reasoning leaderboard: https://huggingface.co/spaces/ kevinpro/Open-Multilingual-Reasoning-Leaderboard 

5 

|**Model**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**Ru**<br>**De**<br>**Es**<br>**Fr**<br>**En**|**Avg**|
|---|---|---|
||||
|GPT-3.5-Turbo|31.2<br>38.0<br>40.0<br>36.0<br>44.0<br>43.2<br>46.0<br>47.2<br>41.6<br>54.4|42.2|
||||
|MAmmoTH 7B_†_<br>WizardMath 7B_†_<br>MetaMath 7B|3.6<br>4.8<br>2.4<br>10.8<br>17.2<br>26.0<br>33.2<br>32.4<br>32.8<br>49.6<br>2.0<br>4.0<br>3.4<br>24.0<br>22.4<br>30.8<br>30.4<br>34.8<br>30.4<br>47.6<br>6.4<br>4.0<br>3.2<br>39.2<br>38.8<br>47.2<br>56.8<br>58.0<br>52.8<br>63.2|21.3<br>23.0<br>37.0|
||||
|MathOctopus 7B<br>+ m-RFT<br>+ MAPO-DPO(ours)|29.2<br>33.6<br>36.4<br>35.2<br>39.2<br>38.8<br>44.8<br>42.4<br>43.2<br>52.0<br>25.6<br>31.2<br>28.8<br>34.0<br>39.2<br>36.0<br>34.8<br>34.4<br>36.4<br>43.2<br>30.8<br>38.0<br>37.6<br>45.2<br>47.2<br>42.0<br>45.2<br>43.2<br>40.8<br>45.6|39.5<br>34.4<br>41.6|
||||
|MetaMathOctopus 7B<br>+ m-RFT<br>+ MAPO-DPO(ours)|25.6<br>42.8<br>36.4<br>40.0<br>46.4<br>46.8<br>49.6<br>54.4<br>46.4<br>66.4<br>23.2<br>33.6<br>34.0<br>34.0<br>47.2<br>43.2<br>45.6<br>47.6<br>44.8<br>62.8<br>36.0<br>44.8<br>44.8<br>47.6<br>55.2<br>53.6<br>53.6<br>56.8<br>52.4<br>70.8|45.5<br>41.6<br>51.6|
||||
|MistralMathOctopus 7B<br>44.0<br>54.4<br>50.4<br>55.6<br>59.2<br>58.8<br>62.4<br>62.0<br>56.8<br>76.0<br>58.0<br>+ m-RFT<br>41.2<br>46.8<br>46.8<br>48.4<br>57.2<br>62.8<br>61.6<br>59.2<br>57.6<br>72.0<br>55.4<br>+ MAPO-DPO(ours)<br>**55.2**<br>**60.4**<br>**61.6**<br>**58.0**<br>**74.0**<br>**70.8**<br>**67.6**<br>**74.0**<br>**69.2**<br>**82.0**<br>**67.3**|||
||||
|MAmmoTH 13B_†_<br>WizardMath 13B_†_<br>MetaMath 13B|3.6<br>5.2<br>1.6<br>19.2<br>31.2<br>36.8<br>45.6<br>50.0<br>39.6<br>56.4<br>6.4<br>5.6<br>5.6<br>22.0<br>28.0<br>34.4<br>40.4<br>45.6<br>42.0<br>52.8<br>11.6<br>6.4<br>7.6<br>42.8<br>49.2<br>63.6<br>64.8<br>65.2<br>65.2<br>67.2|28.9<br>28.3<br>44.4|
||||
|MathOctopus 13B<br>+ m-RFT<br>+ MAPO-DPO(ours)|42.4<br>39.2<br>44.8<br>38.8<br>49.6<br>45.2<br>48.4<br>53.6<br>43.2<br>54.8<br>29.6<br>30.8<br>34.4<br>36.4<br>40.4<br>39.2<br>42.0<br>42.8<br>40.4<br>48.0<br>38.8<br>46.8<br>42.0<br>47.6<br>53.6<br>49.2<br>52.0<br>54.4<br>46.4<br>54.0|46.0<br>38.4<br>48.5|
||||
|MetaMathOctopus 13B<br>+ m-RFT<br>+ MAPO-DPO(ours)|34.4<br>42.8<br>41.6<br>49.2<br>52.8<br>54.4<br>54.4<br>59.2<br>53.6<br>71.6<br>22.8<br>29.6<br>30.4<br>35.2<br>39.2<br>40.0<br>43.6<br>43.6<br>41.2<br>59.2<br>**44.8**<br>**47.6**<br>**55.2**<br>**56.0**<br>**59.6**<br>**59.2**<br>**59.2**<br>**63.6**<br>**62.8**<br>**71.6**|51.4<br>38.5<br>**58.0**|



Table 3: Model Performances on MGSM test set. “Avg” represents the average performance in ten languages and bold text denotes the best results within the same model size. Results marked with _[†]_ come from Chen et al., 2023. 

## respectively. 

Surprisingly, our preference optimization dataset does not contain English, but the accuracy of English has also improved. After alignment, MetaMathOctopus even surpassed the English mathematical reasoning models on English questions across three datasets. We believe this is primarily due to the alignment facilitating a more consistent understanding of reasoning across different languages, which contributes to the enhancement of general reasoning capabilities. 

Meanwhile, our approach has also achieved stable and impressive improvements on the MetaMathOctopus 7B which has stronger reasoning capabilities, propelling it past the ChatGPT and the larger scale model MetaMathOctopus13B, demonstrating the robustness and potential of our framework. 

## **5.2 Alignment is the Key to Enhanced Multilingual Reasoning** 

We use PPL and ACR to evaluate the degree of alignment in reasoning processes and final answers between other languages and English, respectively. 

Regarding the alignment of the reasoning process, as shown in Figure 2, our method effectively improves the consistency of the reasoning process, 

particularly for languages where the base model had poorer alignment, such as Bengali, Thai, and Japanese. This proves that the model can show more similar reasoning thinking to English on nonEnglish after alignment. 

Additionally, we notice that alignment also contributes to more consistent final answers. A higher ACR indicates that there is a greater overlap between the questions answered correctly in nonEnglish and those answered correctly in English. From the results in Figure 3, our alignment framework has greatly increased the ACR for each language. This means that the performance gains in Table 2 stem from the parts that intersect with English. These observations demonstrate that our method effectively aligns the model’s non-English reasoning processes with English, thereby enhancing reasoning abilities in non-English languages. 

## **5.3 Generalizable Multilingual Reasoning** 

As indicated in Table 2, on the out-of-domain test set, MAPO achieved a 16.2% and 11.7% increase in accuracy, which is significantly higher than the 7.5% and 3.7% achieved by m-RFT. Surprisingly, while MAPO effectively improves the multilingual reasoning performance on both two datasets, it also 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization_images/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization.pdf-0007-00.png)


Figure 2: PPL across nine languages on MSVAMP. Lower PPL indicates higher consistency in reasoning processes. 

||**MGSM**|**MNumGLUESub**|
|---|---|---|
|Acc. on MSVAMP|31.6|54.9|
|Self-BLEU(En)|0.81|0.53|
|Self-BLEU(Non-En)|0.84|0.60|



Table 4: Comparison between MGSM8KInstruct (MGSM) and MNumGLUESub. We report the average Self-BLEU scores for the sampled reasoning in English and non-English. 

improves the performance on MGSM. In contrast, m-RFT exhibited a 5.1% and 3.9% performance degradation. 

We believe that this is mainly because the model directly learns the given labels during SFT. The labels inevitably involve data-specific attributes, which are difficult to generalize. Conversely, our method does not give the model “correct answers”, but instead guides the model to generate outputs we prefer, which is more effective in allowing the model to learn multilingual reasoning, and this preference can better generalize across various challenges. 

## **6 Analysis** 

## **6.1 Preference Estimating on a Dataset Different from SFT** 

We conduct experiment with preference optimization using the MGSM8KInstruct dataset instead of the MNumGLUESub. Table 4 present the average accuracy over 10 languages on MSVAMP. The results indicate that preference optimization based on the SFT dataset lead to the performance degradation on the out-of-domain test set MSVAMP. We believe this is because SFT make the model quickly remember the oracle reasoning process, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization_images/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization.pdf-0007-09.png)


Figure 3: ACR across nine languages on MSVAMP. Higher ACR indicate higher answer consistency. 

|System|MSVAMP|MGSM|
|---|---|---|
|MetaMathOctopus7B|53.0|45.5|
|+ Ours & NLLB 600M|61.1|48.9|
|+ Ours & MBART-MMT-600M|59.9|49.6|
|+ Ours & M2M-1.2B|61.4|49.3|



Table 5: Average accuracy on ten languages on MSVAMP and MGSM. Note that we only conduct one iteration MAPO-DPO with different translation models. 

which leads to a serious degradation in the generation diversity. Thus, the policy model struggles to learn the alignment preferences we truly hope to optimize and easily falls into the trap of overfitting. 

For further analysis, We randomly sample 1000 reasoning processes of LLMs on the MGSM8KInstruct and MNumGLUESub, and calculated the Self-BLEU (Zhu et al., 2018) on sampled reasoning process of the same problem. The results (Table 4) suggest that on MGSM dataset, LLMs tends to generate quite similar reasoning, compared with MNumGLUESub, which confirms the diversity degradation. 

## **7 Robustness over Different Translation Models** 

To verify the robustness of our framework across various translation models, we conduct experiments using MBART-MMT-600M[6] and M2M1.2B[7] , which have the different model architectures, scales, and training datasets. The results, presented in Table 5, demonstrate that our framework consistently achieves performance improvements. The performance is even slightly enhanced when 

> 6https://huggingface.co/facebook/ mbart-large-50-many-to-many-mmt 

> 7https://huggingface.co/facebook/m2m100_1.2B 

7 

|**English Question**|182 students suggested adding mashed potatoes while others suggested adding bacon to the<br>menu . if 166 more students suggested adding bacon than those that suggested mashed potatoes<br>how many students suggested bacon ?|
|---|---|
|||
|**Chinese Question**|182`名学生建议在菜单中加入土豆泥，而其他人建议加入培根。如果有比建议加入土豆`<br>`泥的学生多`166`人建议加入培根，那么有多少学生建议加入培根？`|
|||
|**English Solution**|The number of students who suggested mashed potatoes is 182. The number of students who<br>suggested bacon is 182+166=348. ✓|
|||
|**Chinese Solution**|**[Before Alignment]**<br>`建议加入土豆泥的学生比建议加入培根的学生多`166`人，所以两边减去`166`得到差值`<br>`为`182 - 166 = 16`，因此，建议加入培根的学生人数为`16`。`(There are 166 more students<br>recommending mashed potatoes than bacon so subtract 166 from both sides to get a difference<br>of 182-166 = 16. Therefore, the number of students recommended to join Bacon is 16.) ✗<br>**[PPL]:** 2.65<br>**[After Alignment]**<br>`建议加入土豆泥的学生人数是`182`。建议加入培根的学生人数是`182 + 166 = 348`。`(The<br>number of students who suggested mashed potatoes is 182. The number of students who<br>suggested bacon is 182+166=348.) ✓<br>**[PPL]:** 0.97|



Table 6: An example from the MSVAMP test set. The example clearly demonstrates that, through our preference optimization method, the model successfully corrects erroneous reasoning processes and generates reasoning processes that is more consistent with English. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization_images/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization.pdf-0008-02.png)


Figure 4: Average accuracy on Non-English math problems and ACR versus preference optimization step on MSVAMP. MathOctopus 13B and ChatGPT are selected for comparison. 

with M2M-1.2B. These experimental results conclusively confirm the robustness of our framework. In addition, we have conducted experiments on translation models of different scales, and the results also demonstrate the robustness of our method. For more details, please refer to the Appendix D. 

just 100 steps. As the optimization continues, the accuracy remains relatively stable while the degree of alignment still shows an increasing trend. 

Meanwhile, although the optimization speed of PPO is slower, it take fewer than 1000 steps to surpass MathOctopus 13B. As the model continued to explore and optimize the preferences, it achieved superior multilingual reasoning performance compared to ChatGPT at 1600 steps. 

The experimental results demonstrate that the model continuously aligns non-English reasoning with English reasoning, effectively enhancing its multilingual reasoning capabilities. For PPOLoRA, although optimization is halted around 2500 steps due to the limitations on computational resources, we notice that growth in multilingual reasoning consistency and ability continues even beyond 2500 steps, indicating the potential of our preference optimization framework. 

## **7.2 Alignment Reasoning for Improvements** 

## **7.1 Reasoning alignment during Preference Optimization** 

To better analyze the optimization preference process, we tested the checkpoints of the DPO and PPO-LoRA on MSVAMP and visualized the accuracy and ACR respectively in Figure 4. 

The results show that the optimization of DPO is quite efficient. DPO enabled the MathOctopus 7B model to achieve higher accuracy and ACR than both ChatGPT and MathOctopus 13B within 

The example in Table 6 directly illustrates how our framework can improve reasoning ability by aligning reasoning in other languages to English. For the given English problem, the basic model correctly analyzes that an addition should be conducted to find the number of students recommended to join Bacon. However, when given the same problem in Chinese, LLM misjudges the relationship of variables and writes a significantly different reasoning process. This directly proves that although the base- 

8 

line model has been fine-tuned with multilingual reasoning data, its reasoning and thoughts are inconsistent, whereas non-English reasoning is more prone to errors. After alignment, the reasoning thoughts are more similar to the English answer, and the reasoning thoughts are also corrected. 

## **8 Conclusion** 

In this paper, we propose MAPO, a novel multilingual alignment-as-preference optimization framework, enhancing reasoning ability in non-dominant languages by aligning them with dominant languages. Experimental results demonstrate that our framework achieves significant improvements on various base models across all three benchmarks, especially with a notable 16.2% increase in average accuracy on the out-of-domain datasets MSVAMP. The analysis confirms that enhancing alignment through our method is the key to improvements in multilingual reasoning capabilities. 

## **Limitation** 

Similar to previous work on preference optimization, our method necessitates a policy model that has the preliminary multilingual reasoning capabilities through SFT. Meanwhile, due to limitations in computational resources, our experiments are confined to aligning models of 7B and 13B size and exploring two preference optimization algorithms, PPO and DPO. Should resources permit, we aim to extend our exploration to models of 70B sizes and examine the performance of a broader spectrum of preference optimization algorithms. 

## **References** 

- Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. 2022a. Training a helpful and harmless assistant with reinforcement learning from human feedback. 

- Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, 

Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Chris Olah, Benjamin Mann, and Jared Kaplan. 2022b. Training a helpful and harmless assistant with reinforcement learning from human feedback. CoRR, abs/2204.05862. 

- Nuo Chen, Zinan Zheng, Ning Wu, Linjun Shou, Ming Gong, Yangqiu Song, Dongmei Zhang, and Jia Li. 2023. Breaking language barriers in multilingual mathematical reasoning: Insights and observations. arXiv preprint arXiv:2310.20246. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. CoRR, abs/2110.14168. 

- Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. CoRR, abs/2308.09583. 

- Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. 2022. Numglue: A suite of fundamental yet challenging mathematical reasoning tasks. arXiv preprint arXiv:2204.05660. 

- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are nlp models really able to solve simple math word problems? 

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. CoRR, abs/2305.18290. 

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. CoRR, abs/1707.06347. 

- Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2022. Language models are multilingual chain-of-thought reasoners. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti 

9 

Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv: 2307.09288. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models. 

- Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. Metamath: Bootstrap your own mathematical questions for large language models. CoRR, abs/2309.12284. 

- Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. 2023. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825. 

- Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. Mammoth: Building math generalist models through hybrid instruction tuning. CoRR, abs/2309.05653. 

- Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, Limao Xiong, Lu Chen, Zhiheng Xi, Nuo Xu, Wenbin Lai, Minghao Zhu, Cheng Chang, Zhangyue Yin, Rongxiang Weng, Wensen Cheng, Haoran Huang, Tianxiang Sun, Hang Yan, Tao Gui, Qi Zhang, Xipeng Qiu, and Xuanjing Huang. 2023. Secrets of RLHF in large language models part I: PPO. CoRR, abs/2307.04964. 

- Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. 2018. Texygen: A benchmarking platform for text generation models. CoRR, abs/1802.01886. 

10 

## **A Experiment Details** 

Our code is primarily based on the trl[8] , with some minor modifications made. The modified code will also be made available at our project. 

**Prompt** : During the sampling, training, and testing phases, we consistently use the same prompt as MathOctopus (Chen et al., 2023). 

**LoRA** : For experiments using LoRA, such as PPO LoRA, we optimize the [q_proj, v_proj , o_proj] modules. The values of r and alpha are set to 128 and 256, respectively. 

**DPO** : We employ a learning rate of 1e-6, with _β_ set at 0.1, and a warmup step count of 100. The batch size is configured to 128. The optimization process is capped at a maximum of 1000 steps, and we save the checkpoint that corresponds to the lowest loss on validation set. The training takes around 4 hours on 8 H100 GPUs. 

**PPO** : We have configured a learning rate of 2e-5, with a batch size of 64 and 2 ppo epochs. We utilize the AdamW optimizer to improve the stability of the optimization, setting the epsilon value to 1e5. Additionally, we have implemented a linear learning rate warm-up technique for the first 150 steps. All other hyper parameters are set to the default values provided by the trl library. We set the maximum optimization steps to 2600 and report the results at this checkpoint. 

## **B Supplemental Experiment Results** 

To verify the robustness of our method, we conducted experiments on different preference optimization algorithms. Due to the limited computational resources, we optimized the PPO algorithm using LoRA. The detailed experimental results of PPO LoRA and each round in Iterative DPO are shown in Table 8. 

Experiments demonstrate that our framework achieve an effective improvement in multilingual reasoning capabilities based on both PPO and DPO. Despite the limited computational resource, where PPO only updated a part of parameters with merely 2600 steps, it has already brought impressive performance enhancements on all three datasets. Additionally, with the increasing rounds of DPO, the model exhibited progressively stronger multilingual reasoning abilities that has not yet reached its limit, revealing the potential of our approach. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization_images/acl_2024_MAPO_Advancing_Multilingual_Reasoning_through_Multilingual-Alignment-as-Preference_Optimization.pdf-0011-09.png)


Figure 5: Accuracy across ten languages on MSVAMP after training MathOctopus 7B on preference datasets constructed using translation models of different scales. 

## **C Used Scientific Artifacts** 

Below lists scientific artifacts that are used in our work. For the sake of ethic, our use of these artifacts is consistent with their intended use. 

- _Transformers (Apache-2.0 license)_ , a framework to facilitate downloading and training state-of-the-art pretrained models. 

- _trl (Apache-2.0 license)_ , a full stack library where provide a set of tools to train transformer language models with Reinforcement Learning. The library is built on top of the Transformers library. 

## **D Scaling of Translation Model** 

We also conducted ablation experiments comparing the translation models used during the preference estimation phase. In addition to NLLB-600Mdistilled, we employed NLLB-1.3B-distilled and NLLB-3.3B to estimate the preference and repeat the experiments. The results are shown in Figure 5, where we find that all three models achieved impressive results, showing stable improvements over the Base Model and ChatGPT across ten languages. These results demonstrate that our method is robust for translation models of various sizes. 

> 8https://github.com/huggingface/trl 

11 

|**Model**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**Ru**<br>**De**<br>**Es**<br>**Fr**<br>**En**|**Avg**|
|---|---|---|
|GPT-3.5-Turbo<br>36.2<br>42.6<br>47.2<br>58.1<br>60.6<br>42.6<br>41.5<br>54.9<br>39.4<br>70.6<br>49.4|||
||||
|MAmmoTH 7B<br>WizardMath 7B<br>MetaMath 7B|1.7<br>4.7<br>4.5<br>23.5<br>29.9<br>26.9<br>34.7<br>37.9<br>36.5<br>42<br>9.8<br>13<br>6.2<br>31.8<br>36<br>34.3<br>34.1<br>39.9<br>36.5<br>45.4<br>13.2<br>19.2<br>12.1<br>50.7<br>53.3<br>52.9<br>54.2<br>56.5<br>56.7<br>63.3|24.2<br>28.7<br>43.2|
||||
|MathOctopus 7B<br>+ m-RFT<br>+ MAPO-DPO(ours)|26.6<br>30.9<br>34.3<br>40.9<br>44.4<br>36.0<br>32.6<br>42.0<br>36.2<br>46.9<br>38.0<br>42.9<br>41.8<br>48.2<br>51.6<br>45.2<br>42.9<br>49.3<br>42.9<br>51.2<br>41.8<br>45.8<br>46.9<br>52.9<br>54.4<br>49.9<br>50.7<br>54.0<br>51.4<br>55.9|37.1<br>45.4<br>50.4|
||||
|MetaMathOctopus 7B<br>+ m-RFT<br>+ MAPO-DPO(ours)|24.5<br>35.2<br>36.2<br>44.1<br>45.8<br>39.2<br>32.6<br>45.2<br>36.7<br>52.5<br>40.3<br>45.0<br>45.2<br>51.4<br>57.8<br>51.6<br>51.6<br>58.8<br>50.1<br>65.3<br>40.7<br>46.0<br>45.0<br>58.2<br>59.3<br>53.1<br>51.4<br>57.4<br>52.0<br>66.1|39.2<br>51.7<br>52.9|
||||
|MistralMathOctopus 7B<br>45.2<br>50.7<br>49.5<br>56.5<br>65.3<br>59.1<br>51.4<br>62.1<br>53.9<br>74.6<br>56.8<br>+ m-RFT<br>52.7<br>60.6<br>61.0<br>67.2<br>70.8<br>65.9<br>61.2<br>71.6<br>64.4<br>78.3<br>65.4<br>+ MAPO-DPO(ours)<br>**62.3**<br>**64.6**<br>**61.6**<br>**72.1**<br>**75.1**<br>**68.0**<br>**69.3**<br>**74.4**<br>**74.2**<br>**78.5**<br>**70.0**|||
||||
|MAmmoTH 13B<br>WizardMath 13B<br>MetaMath 13B|6.8<br>10.5<br>10.4<br>31.6<br>38<br>41.1<br>41.4<br>43.3<br>42.6<br>46.7<br>9.8<br>14.3<br>12.4<br>30.5<br>39.0<br>36.5<br>35.2<br>43.9<br>39.2<br>47.8<br>10.9<br>16.0<br>16.0<br>55.2<br>57.4<br>56.5<br>58.9<br>60.6<br>58.0<br>64.2|31.2<br>30.9<br>45.4|
||||
|MathOctopus 13B<br>+ m-RFT<br>+ MAPO-DPO(ours)|42.4<br>39.2<br>44.8<br>38.8<br>49.6<br>45.2<br>48.4<br>53.6<br>43.2<br>54.8<br>44.1<br>49.9<br>51.0<br>51.0<br>55.6<br>49.0<br>47.1<br>53.3<br>46.1<br>57.4<br>49.0<br>53.3<br>52.2<br>55.0<br>57.4<br>53.3<br>52.2<br>55.9<br>50.7<br>58.8|46.0<br>50.5<br>53.8|
||||
|MetaMathOctopus 13B<br>+ m-RFT<br>+ MAPO-DPO(ours)|34.4<br>42.8<br>41.6<br>49.2<br>52.8<br>54.4<br>54.4<br>59.2<br>53.6<br>71.6<br>40.1<br>51.6<br>47.5<br>60.5<br>62.0<br>58.9<br>54.8<br>62.5<br>54.2<br>66.5<br>**52.9**<br>**55.4**<br>**55.0**<br>**67.2**<br>**65.0**<br>**54.8**<br>**54.4**<br>**65.2**<br>**57.4**<br>**70.4**|51.4<br>55.9<br>**59.8**|



Table 7: Model Performances on MNumGLUESub test set. “Avg” represents the average performance in ten languages and bold text denotes the best results within the same model size. 

12 

|**Model**|**Bn**<br>**Th**<br>**Sw**<br>**Ja**<br>**Zh**<br>**Ru**<br>**De**<br>**Es**<br>**Fr**<br>**En**|**Avg**|
|---|---|---|
|_Benchmark: MSVAMP_|||
||||
|MathOctopus 7B<br>+ PPO LoRA<br>+ DPO Iter1<br>+ DPO Iter2<br>+ DPO Iter3|27.7<br>35.9<br>39.4<br>41.6<br>42.7<br>44.2<br>44.0<br>45.1<br>45.3<br>46.4<br>38.9<br>47.5<br>47.1<br>51.0<br>51.7<br>51.1<br>50.3<br>51.6<br>51.4<br>52.7<br>42.0<br>53.2<br>52.7<br>54.7<br>56.4<br>56.9<br>55.7<br>58.5<br>59.3<br>59.6<br>45.3<br>53.9<br>53.8<br>56.8<br>58.1<br>56.6<br>58.7<br>59.1<br>58.5<br>60.0<br>48.8<br>55.2<br>56.0<br>60.3<br>58.8<br>58.3<br>58.1<br>59.7<br>60.8<br>58.4|41.2<br>49.3<br>54.9<br>56.1<br>57.4|
||||
|MetaMathOctopus 7B<br>+ PPO LoRA<br>+ DPO Iter1<br>+ DPO Iter2<br>+ DPO Iter3|36.1<br>47.5<br>49.4<br>51.3<br>54.5<br>53.6<br>56.6<br>60.0<br>57.2<br>64.2<br>45.7<br>52.0<br>52.2<br>61.2<br>58.3<br>57.0<br>58.7<br>60.7<br>61.7<br>67.5<br>44.1<br>58.2<br>59.0<br>60.3<br>62.3<br>63.5<br>65.1<br>63.7<br>64.2<br>70.2<br>48.5<br>61.8<br>59.2<br>64.3<br>64.3<br>64.4<br>65.0<br>66.1<br>65.4<br>70.9<br>50.1<br>61.6<br>61.7<br>65.9<br>65.7<br>64.8<br>68.4<br>68.5<br>68.6<br>71.6|53.0<br>57.5<br>61.1<br>63.0<br>64.7|
||||
||_Benchmark: MGSM_||
||||
|MathOctopus 7B<br>+ PPO LoRA<br>+ DPO Iter1<br>+ DPO Iter2<br>+ DPO Iter3|29.2<br>33.6<br>36.4<br>35.2<br>39.2<br>38.8<br>44.8<br>42.4<br>43.2<br>52.0<br>31.2<br>38.4<br>38.4<br>37.2<br>43.6<br>35.2<br>46.0<br>44.0<br>38.8<br>51.2<br>29.2<br>36.4<br>35.6<br>35.6<br>41.6<br>38.4<br>40.8<br>42.0<br>37.6<br>46.8<br>30.4<br>36.0<br>37.6<br>38.0<br>45.2<br>39.6<br>42.0<br>47.6<br>41.2<br>45.2<br>30.8<br>38.0<br>37.6<br>45.2<br>47.2<br>42.0<br>45.2<br>43.2<br>40.8<br>45.6|39.5<br>40.4<br>38.4<br>40.3<br>41.6|
||||
|MetaMathOctopus 7B<br>+ PPO LoRA<br>+ DPO Iter1<br>+ DPO Iter2<br>+ DPO Iter3|25.6<br>42.8<br>36.4<br>40.0<br>46.4<br>46.8<br>49.6<br>54.4<br>46.4<br>66.4<br>36.0<br>41.2<br>41.6<br>46.4<br>54.8<br>53.6<br>54.0<br>55.6<br>51.6<br>68.0<br>32.8<br>43.2<br>40.4<br>48.8<br>49.2<br>52.8<br>54.4<br>52.8<br>50.0<br>64.8<br>34.0<br>48.0<br>45.2<br>40.4<br>54.0<br>52.0<br>50.8<br>54.0<br>49.2<br>70.4<br>36.0<br>44.8<br>44.8<br>47.6<br>55.2<br>53.6<br>53.6<br>56.8<br>52.4<br>70.8|45.5<br>50.3<br>48.9<br>49.8<br>51.6|
||||
||_Benchmark: MNumGLUESub_||
||||
|MathOctopus 7B<br>+ PPO LoRA<br>+ DPO Iter1<br>+ DPO Iter2<br>+ DPO Iter3|26.6<br>30.9<br>34.3<br>40.9<br>44.4<br>36.0<br>32.6<br>42.0<br>36.2<br>46.9<br>34.3<br>41.1<br>40.7<br>45.6<br>49.3<br>39.5<br>34.3<br>46.7<br>36.3<br>51.6<br>32.6<br>41.4<br>42.0<br>42.7<br>45.0<br>42.2<br>40.7<br>47.8<br>40.9<br>47.8<br>35.6<br>39.7<br>42.2<br>45.6<br>50.7<br>44.1<br>42.6<br>49.3<br>42.9<br>49.7<br>41.8<br>45.8<br>46.9<br>52.9<br>54.4<br>49.9<br>50.7<br>54.0<br>51.4<br>55.9|37.1<br>41.9<br>42.3<br>44.2<br>50.4|
||||
|MetaMathOctopus 7B<br>+ PPO LoRA<br>+ DPO Iter1<br>+ DPO Iter2<br>+ DPO Iter3|34.7<br>41.4<br>37.9<br>47.8<br>54.2<br>45<br>43.1<br>52.5<br>45.0<br>60.8<br>42.4<br>46.7<br>45.4<br>55.2<br>58.9<br>45.0<br>41.6<br>55.7<br>44.6<br>64.6<br>37.7<br>46.3<br>43.5<br>54.8<br>58.4<br>50.5<br>52.2<br>60.1<br>51.6<br>62.1<br>40.7<br>46.9<br>45.0<br>53.5<br>58.4<br>50.8<br>51.4<br>58.4<br>49.9<br>64.6<br>40.7<br>46.0<br>45.0<br>58.2<br>59.3<br>53.1<br>51.4<br>57.4<br>52.0<br>66.1|46.3<br>50.0<br>51.7<br>52.0<br>52.9|



Table 8: Model Performances on three benchmarks. We report the results of PPO LoRA and each round in Iterative DPO. “Avg” represents the average performance on ten languages 

13 

