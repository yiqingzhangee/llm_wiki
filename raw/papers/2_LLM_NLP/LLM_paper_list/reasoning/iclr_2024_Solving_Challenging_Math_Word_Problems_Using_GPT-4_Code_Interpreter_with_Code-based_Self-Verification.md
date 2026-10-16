# SOLVING CHALLENGING MATH WORD PROBLEMS USING GPT-4 CODE INTERPRETER WITH CODE-BASED SELF-VERIFICATION 

**Aojun Zhou[1]** _[∗]_ **Ke Wang[2]** _[∗]_ **Zimu Lu[3]** _[∗]_ **Weikang Shi[4]** _[∗]_ **Sichun Luo[5]** _[∗]_ **Zipeng Qin[1] Shaoqing Lu[6] Anya Jia[7] Linqi Song[5] Mingjie Zhan[1]** _[†]_ **Hongsheng Li[1]** _[‡]_ 

1Multimedia Laboratory (MMLab), The Chinese University of Hong Kong 

2Nanjing University 3University of Science and Technology of China 

4Tsinghua University 5City University of Hong Kong 

6Changsha University of Science and Technology 7Tufts University _{_ aojunzhou, wangk.gm, sichunluo2, zmjdll _}_ @gmail.com luzimu@mail.ustc.edu.cn shiwk20@mails.tsinghua.edu.cn linqi.song@cityu.edu.hk hsli@ee.cuhk.edu.hk 

## ABSTRACT 

Recent progress in large language models (LLMs) like GPT-4 and PaLM-2 has brought significant advancements in addressing math reasoning problems. In particular, OpenAI’s latest version of GPT-4, known as GPT-4 Code Interpreter, shows remarkable performance on challenging math datasets. In this paper, we explore the effect of code on enhancing LLMs’ reasoning capability by introducing different constraints on the _Code Usage Frequency_ of GPT-4 Code Interpreter. We found that its success can be largely attributed to its powerful skills in generating and executing code, evaluating the output of code execution, and rectifying its solution when receiving unreasonable outputs. Based on this insight, we propose a novel and effective prompting method, explicit code-based self-verification (CSV), to further boost the mathematical reasoning potential of GPT-4 Code Interpreter. This method employs a zero-shot prompt on GPT-4 Code Interpreter to encourage it to use code to self-verify its answers. In instances where the verification state registers as “False”, the model shall automatically amend its solution, analogous to our approach of rectifying errors during a mathematics examination. Furthermore, we recognize that the states of the verification result indicate the confidence of a solution, which can improve the effectiveness of majority voting. With GPT-4 Code Interpreter and CSV, we achieve an impressive zero-shot accuracy on MATH dataset **(53.9%** _→_ **84.3%)** . 

## 1 INTRODUCTION 

Large language models (LLMs) (Brown et al., 2020; OpenAI, 2023; Anil et al., 2023) have shown impressive success in various tasks, such as common sense understanding and code generation. However, they still fall short in mathematical reasoning, often producing nonsensical or inaccurate content and struggling with complex calculations. Previous attempts to tackle these challenges include the Chain-of-Thought (CoT) (Wei et al., 2022) framework, which enhances LLMs’ logical reasoning abilities by generating intermediate steps in their reasoning process. Additionally, PAL (Gao et al., 2023) introduces a novel approach by using the Python programming interpreter to improve computational accuracy. 

In recent advancements, OpenAI has unveiled an improved version of GPT-4, namely the GPT-4 Code Interpreter[12] or _GPT4-Code_ , which is proficient at providing logical natural language reason- 

> _∗_ Equal contribution. 

> _†_ Project lead. 

> _‡_ Corresponding author. 

> 1https://openai.com/blog/chatgpt-plugins#code-interpreter 

> 2https://chat.openai.com/?model=GPT4-Code-interpreter 

1 

ing, alongside step-by-step Python code. Notably, it can generate and execute code incrementally, and subsequently present the executed code’s output back to the LLM. The addition of code generation and execution to natural language outputs has shown promising results in solving mathematical reasoning problems. Our initial experiments show that GPT4-Code achieved an impressive zero-shot accuracy of 69.7% on the challenging MATH dataset (Hendrycks et al., 2021), marking a significant improvement of 27.5% over GPT-4’s performance (42.2%). 

While GPT4-Code has demonstrated proficiency in solving math problems, there has been a notable absence of systematic analysis focusing on understanding and further enhancing its mathematical problem-solving abilities. A key distinction between GPT4-Code and its predecessor, GPT-4, lies in GPT4-Code’s ability to _automatically generate and execute code_ . Therefore, this paper presents pilot experiments that investigate GPT4-Code’s code generation and execution mechanism using specific code-constrained prompts. The analysis reveals that GPT4-Code’s strong performance is not solely due to its code generation and execution abilities, but also its capacity to adjust its problemsolving strategies based on feedback from code execution—a process we term **self-debugging** (illustrated in Tab. 7 and Tab. 8). Due to the fact that code generation evolves its reasoning step-by-step and performs self-debugging after code execution errors, there is an increased frequency of code usage. Hence, we introduce the concept of Code Usage Frequency to differentiate these unique prompting strategies to quantitatively analyze the impact of code-constrained prompts on GPT4Code for mathematical problem-solving. 

The step-by-step code generation and self-debugging mechanisms highlight the critical role of code in mathematical problem-solving. Nevertheless, the self-debugging mechanism only verifies each step of the generated code while lacks the verification of the reasoning steps and the final answer, which has been demonstrated to be of vital importance to the math problem-solving abilities of LLMs (Cobbe et al., 2021; Lightman et al., 2023; Weng et al., 2023). 

We therefore ask the question: _can we fully exploit the code generation and self-debugging mechanisms in GPT4-code, so that it can automatically verify and correct its solutions, without extra assistance from other models or users?_ 

To answer this question, we propose a simple yet effective prompting technique termed the **explicit code-based self-verification** (CSV), which guides GPT4-Code to generate additional code that verifies the answer and adjusts the reasoning steps if there’s a flaw in reasoning. Unlike previous methods that rely on external language models for verification (Lightman et al., 2023; Cobbe et al., 2021), our approach leverages GPT4-Code’s inherent strengths. This approach offers two key benefits: (1) When the verification indicates an answer is **False** , GPT4-Code can rectify its prior solution and provide an improved alternative. (2) Solutions verified as **True** tend to be more reliable, akin to human problem-solving. However, even if a solution is self-verified as **False** , we do not directly abandon it. Instead, we propose a weighted majority voting strategy that incorporates the code-based solution verification results, as opposed to relying exclusively on the frequency of answers. We assign different weights to the solutions according to their verification states, reflecting the solutions’ varying levels of reliability. In alignment with the Code Usage Frequency analysis from our pilot experiments, our explicit code-based self-verification prompt boosts GPT4-Code’s accuracy in mathematical problem-solving with increased code usage. 

Empirical study demonstrates the effectiveness of our proposed framework on the MATH, GSM8K, and MMLU-Math datasets using GPT4-Code. Our approach achieves an impressive accuracy of **84.32%** on the MATH dataset, greatly outperforming the base GPT4-Code and previous state-ofthe-art methods. Additionally, we are making our experimental data on the MMLU-Math and MATH datasets publicly available, enabling result replication and facilitating fine-tuning of open-source LLM models (e.g., LLaMA 2 (Touvron et al., 2023)) to further enhance mathematical problemsolving capabilities with the assistance of code generation. 

This paper’s main contributions can be summarized in three key aspects: 

- This study provides the first systematic analysis of code generation, execution, and selfdebugging’s role in mathematical problem-solving. Our findings reveal that GPT4-Code’s impressive mathematical problem-solving proficiency is primarily attributed to its step-bystep code generation and dynamic solution refinement based on code execution outcomes. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Question (c) Model output<br>Question:<br>BasicPrompt<br>The greatest common divisor of positive integers m and n is 6.<br>The least common multiple of  m  and  n  is 126. What is the least Solve\\boxed{}the problem. The problem is:and put { yourQuestionanswer } . in<br>possible value of  m  +  n ? Answer: 60<br>One important property relating greatest common divi-<br>sor (GCD) and least common multiple (LCM) is that<br>(a) Model output (b) Model output GCD( m, n )  ×  LCM( m, n ) = m × n<br>Given in the problem, we can get<br>Prompt 1 Prompt 2 6  ×  126 = m × n<br>You are not allowed to use<br>any { Questioncode. } . The problem is: You can use code only 1 time. { Question } . The problem is: Let’s compute the right hand side of the equation first.code<br>We know that for any two positive First, we should recall that for any two positive inte- gcd,lcm = 6, 126<br>integers m and n , the product of gers m and n , their greatest common divisor (gcd) gcd * lcm<br>their greatest common divisor (GCD) and least common multiple (lcm) satisfy the relation- >>> 756<br>and least common multiple (LCM) is ship:<br>equal to the product of the two num-bers. This can be written as: m · n = gcd( m, n )  ·  lcm( m, n ) The product of  m  and  n  is 756.<br>In this problem, we’re given that gcd( m, n ) = 6 Since the GCD of m and n is 6, both m and n must<br>GCD ( m, n )  · LCM ( m, n ) and lcm( m, n ) = 126. We can substitute these be multiples of 6.<br>Let’s find all the pairs of multiples of 6 that multiply to<br>Given that  GCD = m (  ·m, n n ) = 6 and values into the formula to get: m · n = 6  ·  126 756.are multiples of 6.We’ll do this by finding all the factors of 756 that<br>LCM substitute these values into the equa-tion: ( m, n ) = 126, we can We can solve this equation for one variable and thenusepossible pairsthe fact that ( m, nm and). n are integers to generate code<br>After we have the pairs ( m, n ), we can find the pair [(i, m t imes n // i) for i in range(6,<br>6  ·  126 = m · n that minimizes  m  +  n . int(m t imes n **0.5) + 1, 6) if m t imes n %<br>i == 0]<br>Solving this gives us  m · n = 756. code >>> [(6, 126), (12, 63), (18, 42)]<br>Now, we need to find the pair of fac-<br>tors of 756 that add up to the smallest import sympy as sp The pairs of factors of 756 that are multiples of 6 are:<br>sum, as we want to find the least pos- gcd,lcm = 6,126 1. (6 ,  126) 2. (12 ,  63) 3. (18 ,  42)<br>sible value ofThe(1,756),pairs(2,378),of  m factors +(3,252),  n . of 756(4,189),are product = gcd * lcmpairssp.divisors(product)= [(d, productif d // < =d) productfor d in// We want to find the pair with the least sum.pute the sums for each pair,identify the smallest one.Let’s com-<br>(6,126), (7,108), (9,84), (12,63), d] code<br>(14,54), (18,42), (21,36), (27,28). min p air = min(pairs, key=sum)<br>We can see that the pair (21,36) givesthe36 =smallest57. sum, which is 21 + >>>min p ((27, 28), 55)air, sum(min pair) sums = [(pair, sum(pair)) for pair in factormin(sums, key=lambda x:>>> ((18, 42), 60) x[1]) p airs]<br>So, the least possible value of  m + n<br>is 57. The pair ( m, n ) that minimizes m + n is<br>Therefore, the solution to the (27 ,  28), and the minimum possible value of  m  + The pair with the smallest sum is (18 ,  42), and the<br>problem is 57 .✗ n  is 55 .✗ sum is 60.So, the least possible value is 60 . ✓<br>**----- End of picture text -----**<br>


Figure 1: Example from the 277th number theory problem in the MATH dataset: **(a)** Prompt 1 generates intermediate text-only reasoning steps. **(b)** Prompt 2 generates intermediate text reasoning steps and post-hoc Python code, akin to PAL (Gao et al., 2023). **(c)** Basic prompt that set no constrain on code usage. Details of the three prompts are presented in Sec. 3.1. (For more examples, see Tab. 9 and Tab. 10.) 

- We introduce the innovative _explicit code-based self-verification (CSV) prompt_ , which leverages GPT4-Code’s advanced code generation mechanism. This prompt guides the model to verify the answer and then reevaluate its solution with code. CSV not only extends the verification to the logic behind problem-solving but also improves the efficacy of the majority voting method by integrating the verification states. 

- Additionally, we have contributed to the LLM community by creating two new instructionfollowing datasets: MATH-code and MMLU-Math-code. These datasets are designed to enhance the mathematical reasoning capabilities of open-source models. 

## 2 RELATED WORK 

**Chain-of-Thought Reasoning.** The _Chain-of-Thought (CoT)_ prompting approach proposed by Wei et al. (2022) is a notable contribution that showcases the multi-step reasoning capabilities of LLMs. By simply adding “Let’s think step by step” before questions, Kojima et al. (2022) implements _Zeroshot-CoT_ , which can serve as a strong zero-shot baseline. Further research extends the reasoning 

3 

capabilities of CoT by applying majority voting to improve self-consistency (Wang et al., 2023), choosing few-shot examples and output chains with more complex reasoning steps (Fu et al., 2022), breaking down the problem to simpler sub-problems (Zhou et al., 2023), or even expanding Chainof-thought to Tree-of-Thoughts (Yao et al., 2023). Similar to Zero-shot-CoT, our method apply “step by step”-like prompts to regularize GPT4-Code’s use of code without the careful design of step-bystep few-shot examples. Additionally, We enhance majority voting to verification-guided weighted majority voting, leveraging the results of CSV as voting weights. 

**Solving Math Problems with Code.** Large language models have been found to be less accurate in performing arithmetic calculations, such as addition, subtraction, multiplication, etc (Cobbe et al., 2021; Lewkowycz et al., 2022; Gao et al., 2023; Lu et al., 2022). Consequently, previous works have attempted to solve math problems with the assistance of code. The GSM8K dataset (Cobbe et al., 2021) uses calculation annotations to extract all arithmetic calculations solved by an external calculator: the Python _eval_ function. To further leverage the role of code in LLMs, _Program-Aided Language model (PAL)_ (Gao et al., 2023) as well as _Program of Thoughts (PoT)_ (Chen et al., 2022) interpret the math problems as Python codes and execute the codes with an external Python interpreter to obtain the answer. Although they can get more accurate answers than some non-code methods, many generated codes have execution errors or get wrong answers due to the lack of verification mechanism. Our approach not only utilizes the ability of GPT4-Code to generate multi-step codes and refine codes that fail to run, but also uses CSV to enhance the reliability and accuracy of the answers. 

**Self-Verification.** Human problem solving is not always a one-time success, but rather requires iterative thinking, verification, and refinement. Unlike previous studies that train an additional verifier to verify the correctness of final answers (Cobbe et al., 2021) or intermediate steps (Lightman et al., 2023; Li et al., 2023), Weng et al. (2023) showed the self-verification abilities of LLMs by generating multiple answers and ranking them by self-verification scores. Furthermore, _SELF-REFINE_ proposed by Madaan et al. (2023) iteratively refines its output through self-generated feedback. Unlike these self-verification methods that require LLMs to give verification feedback in natural language, our method applies generated codes to verify the answers and votes on different answers based on the verification results, thus improving the accuracy of the verification and making full use of the information in the verification process. 

## 3 METHOD 

We first conduct a pilot experiment with GPT4-Code on the challenging MATH dataset (Hendrycks et al., 2021). Remarkably, it achieves an accuracy of 69.7%, significantly surpassing the previous state-of-the-art performance of 53.9% (Zheng et al., 2023). Encouraged by the compelling performance of GPT4-Code, we strive to systematically explore and analyze its underlying code mechanisms. In Sec. 3.1, we illustrate, via our code-constrained prompts design, that GPT4-Code’s robust performance in solving math problems derives not only from its ability to generate accurate step-by-step code, but also from its self-debugging mechanism. In Sec. 3.2, we aim to leverage GPT4-Code’s self-debugging strengths to further improve its mathematical problem-solving ability. 

## 3.1 PILOT EXPERIMENTS ON ANALYZING CODE USAGE OF GPT4-CODE 

To explore the impact of code usage on GPT4-Code’s math problem-solving capabilities, we adopt a straightforward approach by constraining GPT4-Code’s interaction with code through thoughtfully constructed prompts. Specifically, we introduce two code-constrained prompts and the basic prompt for comparison: 

- **Prompt 1:** _No code usage is allowed_ : In response to this prompt, GPT4-Code is prohibited from incorporating code into its solutions. This prompts GPT4-Code to rely solely on Natural Language ( **NL** ) reasoning chain, resembling solutions in the CoT framework (Wei et al., 2022). The resulting sequence of reasoning steps is depicted as **CNL** , with an example given in Fig. 1 (a). 

- **Prompt 2:** _Code can be used only once_ : In this prompt setting, GPT4-Code is permitted to employ code exclusively within a single code block to generate the solution, mirroring 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Verification Prompt<br>Basic Prompt (Please verify your answer using code interpreter by yourself.)<br>Prompt 1 Prompt 2<br>(You are not allowed to use any code.) (You can use code only 1 time.)<br>Code Usage Frequency<br>(a) 76 (b) 100<br>74.48<br>3.0<br>74<br>90<br>72 4.54 2.0<br>80<br>69.94<br>70<br>68 67.58 70 1.5<br>66 60 1.0<br>64 6.78 50<br>0.5<br>62<br>60.80 40<br>60 0.1<br>1 2 3 4 Level 1 Level 2 Level 3 Level 4 Level 5<br>Prompts Levels<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 2: Performance on MATH dataset of different levels by applying different prompts to adjust the frequency of code usage. **(a)** Comparison of overall accuracy between the 4 prompts. **(b)** Code usage frequency is in proportion to accuracy in all five levels and this phenomenon is especially apparent when the problems are relatively complicated (i.e. with higher level). 

the PAL approach introduced by Gao et al. (2023). We denote this sequence as **CSL** , representing a series of Symbolic Language ( **SL** ), such as Python. An example is shown in Fig. 1 (b). 

- **Basic Prompt:** GPT4-Code is prompted to tackle the problem without any restrictions on code usage. This prompt leads to GPT4-Code’s typical functioning pattern, which can be denoted as **C** = (( **c1NL** _,_ **c1SL** ) _,_ ( **c2NL** _,_ **c2SL** ) _, . . ._ ), representing a sequential list of reasoning steps, each consisted of both natural language and Python code, with an example shown in Fig. 1 (c). 

Apart from the specific example in Fig. 1, we introduce _Code Usage Frequency_ to record the number of the code execution for different prompts. The results of the experiments using these prompts are shown in Fig. 2 (b). This figure illustrates a positive correlation between the better performance of GPT4-Code and the higher Code Usage Frequency. More specifically, 

**Prompt 1** v.s. **Prompt 2:** _Prompt 1_ results in almost negligible code usage, while _Prompt 2_ results in approximately 1 time’s code usage. _Prompt 2_ yields an accuracy gain of 6.9 percent over Prompt 1. This suggests that the Python code chains **CSL** , can improve computational capability more than the natural language chains **CNL** . This observation is consistent with the findings in previous Python-based prompting methods (Gao et al., 2023; Chen et al., 2022). However, employing code only once comes with an inherent drawback – the model lacks the ability to self-debugging when the code output triggers an error or produces an implausible outcome. 

**Prompt 2** v.s. **Basic Prompt** : The _Basic Prompt_ consistently produces solutions that entail multiple instances of code usage, resulting in a large Code Usage Frequency. Additionally, the _Basic Prompt_ exhibits notably enhanced accuracy. These improvements in Code Usage Frequency and accuracy might be attributable to two unique advantages: (1) Generating code in brief and frequent segments, divided among natural language reasoning steps, tends to result in higher accuracy. (2) The model possesses the capability to evaluate the results of code execution and make corrections to solution steps if the outcomes contain bugs or are deemed illogical, as illustrated in Tab. 7 and Tab. 8. 

From these observations, it is plausible to enhance and build upon the favorable attributes of GPT4Code, to further improve its precision in tackling math problems. 

- 3.2 EXPLICIT CODE-BASED SELF-VERIFICATION PROMPTING 

Inspired by the observations on Code Usage Frequency analysis, we seek to harness the capabilities of GPT4-Code. These capabilities include the model’s aptitude for generating accurate code, evaluating the outcomes of code execution, and automatically adjusting reasoning steps of solutions when 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Question (b) Example: w/ self verifications<br>Question: We can firstly change the inequality into x > [√] 4 100000000.<br>What is the smallest positive integer solution Then let’s evaluate the fourth root of 100,000,000.<br>to ( [100] [,] [000] x [4] [,] [000] )  <  1? code<br>Answer: 101<br>import math<br>root = math.pow(100000000, 1/4)<br>smallest i nteger = math.ceil(root)<br>>>> (100.0, 100)<br>(a) Example: w/o self verification<br>Thus, the smallest positive integer  x  is 100.<br>Breaking this down into subproblems, the first step is to rearrange Let’s verify this solution.<br>the inequality to isolate  x . code<br>We can do this by multiplying both sides of the inequality by  x [4] and<br>then taking the fourth root of both sides. verification = 1e8 / (smallest integer  ∗∗ 4)  <  1<br>This gives us  x > [√] 4 100000000. >>> False<br>Let’s solve for  x .<br>The smallest integer x should be the next integer greater than 100,<br>not 100 itself.<br>code<br>Let’s calculate that next integer.<br>import math<br>root = math.pow(100000000, 1/4) code<br>>>> 100<br>smallest i nteger = 100 + 1<br>verification = 1e8 / (smallest integer  ∗∗ 4)  <  1<br>Since 100 is already a positive integer, the smallest positive integer >>> (101, True)<br>solution to the inequality [100] [,] [000] x [4] [,] [000] < 1 is  x = 100. Upon substitutingsolution.  x = 101 back, it holds true, which verifies our<br>The final answer is 100 . ✗ The final answer is 101 . ✓<br>**----- End of picture text -----**<br>


Figure 3: Question from the 712th intermediate algebra problem in the MATH dataset. **(a)** Without selfverification, the model generates a wrong answer. **(b)** With self-verification, the model corrects the error and generates the correct answer. The CSV prompt: _To solve the problem using code interpreter step by step, and please verify your answer using code interpreter._ 

needed. However, despite these advantages, GPT4-Code currently falls short in assuring solution correctness. Consequently, our objective is to utilize these strengths to augment solution verification. 

To achieve this objective, we propose the technique termed as **explicit code-based self-verification** (CSV). This method prompts GPT4-Code to explicitly validate its answer through code generation. By implementing this prompt, we introduce an extra verification stage to the solution **C** , referred to as **V** . The verification outcome **V** can be classified as _True_ , _False_ , or _Uncertain_ . An _Uncertain_ classification indicates that GPT4-Code encountered difficulties in identifying an effective method for answer verification, thereby abstaining from delivering a definitive verification result. Leveraging GPT4-Code’s inherent autonomous capabilities, we can formulate the proposed prompting as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0006-04.png)


An example is presented in Fig. 3 (b). Incorporated with CSV, the model becomes capable of using code to verify answers, then reviewing and adjusting how it arrived at the solution if the verification result is _False_ , aiming at obtaining the correct answer. Upon refining and correcting the initial solution, we anticipate a notable increase in accuracy. It is worth noting that both the verification and rectification stages are code-based. This inevitably results in increased Code Usage Frequency, akin to the aforementioned analysis, which will be further demonstrated in subsequent experiments. 

We perform experiments with CSV, and these results can be found in Fig. 2. The experiment here is conducted with GPT4-Code on MATH (Hendrycks et al., 2021). In Fig. 2 (b), the accuracy achieved with our proposed CSV prompt consistently surpasses that of the _Basic Prompt_ across all designated difficulty levels[3] . Meanwhile, the Code Usage Frequency receives a clear increase. 

> 3Human-perceived easier problems are categorized under Level-1 difficulty as per Hendrycks et al. (2021). 

6 

Before the advent of GPT4-Code, prior frameworks (Lightman et al., 2023; Cobbe et al., 2021) depended on an external LLM to use natural language for verification and well-designed few-shot example prompts. In contrast, our approach simplifies the process by relying solely on a straightforward prompt for GPT4-Code, all in a **zero-shot** manner. This enables GPT4-Code to autonomously verify and independently rectify its solutions using the advanced code execution mechanism, thereby eliminating the need for customized few-shot examples. 

Given that CSV can effectively verify problem-solving answers, we can naturally integrate the verification states into majority voting, akin to the methodology embraced in self-consistency CoT (Wang et al., 2023). Answers deemed **True** through verification are generally more trustworthy, reflecting the problem-solving approach seen in human cognition (Newell & Simon, 1972; Wang & Chiew, 2010). This improved reliability can be leveraged in the widely-used majority voting process. To exploit this insight, we introduce _verification-guided weighted majority voting_ , which assigns different weights to the states of the verification process. 

In practice, it sometimes occurs that once an answer is confirmed as False, no additional verification is conducted, yielding a False verification state. We allocate corresponding weights these states of **True** , **Uncertain** , **False** : _w_ **T** _, w_ **U** , and _w_ **F** , respectively. 

Similar to the Self-consistency with CoT (CoT-SC) (Wang et al., 2023) in Fig. 4 (a)(ii), our framework can sample _k_ paths. For simplicity, we extract pairs of final answers and their corresponding verification results from _k_ solutions, represented as ( _v[i] , a[i]_ ) _, i_ = 1 _,_ 2 _, . . . , k_ , where _v[i]_ and _a[i]_ denote the _i_ -th final answer and final verification result, respectively. 

So the voting score for each candidate answer _a_ can be expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0007-05.png)


Here, _a_ represents a candidate answer, _v_ denotes the state of verification, and _wv_ is an element from the set _{w_ **T** _, w_ **U** _, w_ **F** _}_ . Each _wv_ signifies the degree of confidence associated with its corresponding verification state. 

Finally, we select the answer with the highest score from all candidate answers, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0007-08.png)


where Score( _a_ ) refers to the score of answer _a_ according to Eq. 1. 

It should be noted that when _wv_ = 1 for all _wv ∈{w_ **T** _, w_ **U** _, w_ **F** _}_ , Eq. 1 becomes equivalent to the naive majority voting employed in Self-Consistency with CoT (CoT-SC) (Wang et al., 2023). Typically, we set _w_ **T** _> w_ **U** _> w_ **F** , which means that an answer verified true has a greater confidence than the one with an uncertain state of verification, while an answer verified false has the lowest degree of confidence. An example of the calculation process within verification-guided weighted majority voting is illustrated in Fig. 4. 

## 4 EXPERIMENTS 

## 4.1 PERFORMANCE ON MATH 

The MATH dataset (Hendrycks et al., 2021) is recognized as the most challenging math word problem dataset, as also highlighted by Chen et al. (Chen et al., 2023). Most of our experiments and the corresponding analyses are performed on the MATH benchmark. Tab. 1 compares the performance of the GPT4-Code against other models. GPT4-Code reaches 69.69% on MATH (Hendrycks et al., 2020), largely surpassing the previous state of art result (53.90%), which shows that GPT4-Code exhibits strong abilities in solving math problems and is used as our baseline for ablation study. On top of GPT4-Code, our method further improves its accuracy, raising the result to 73.54% after adding explicit code-based self-verification, and 84.32% after adding both explicit code-based selfverification and verification-guided weighted majority voting (the number of sampled paths is 16). 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Reasoning Code Reasoning & Code Question: Given that the repetend in the decimal representation of 191 [contains 18 ]<br>digits, find the 39th digit in the decimal representation.<br>Verification Answer<br>Verified: True Verified: Uncertain Verified: False<br>Input Input Given that the sequence repeats every 18 digits, the 39the 3the 3 [th] digit will be the same as  [rd][rd] . So, we need to find digit in the decimal  If we calculate the first few decimal places of see a pattern emerge. Let's calculate this using Python. 191 [, we will] Since the decimal representation repeats every 18 digits, the 39the same as the 21 [th] digit will be  [st] digit.<br>representation of  191 Code execution…… Code execution……<br>The result is 0. Now let’s<br>Code execution…… The 39th digit will be the same as the 21st, which is  verify the answer.<br>… … … … … … example The 3representation of Now let’s verify if this  [rd] digit in the decimal  191 [ is 2.] actually the 3rd digit in our calculated sequence. Let's find this digit. The pattern of 18 repeating digits is Code execution……<br>solution is equal to the 39 [th] Code execution…… "052631578947368421". The<br>digit. given the repeating nature  21st digit, which is the same<br>The verification result is True. Code execution…… of the decimal representation of  191 [, we can ] as the 39th digit, is '5', not ‘0’. The verification result is False, Our final result is<br>Therefore, our final solution  confidently state that the<br>Majority  𝑤𝑇 𝑤𝑈 𝑤𝐹 is  2 39th digit will indeed be 2 5<br>Voting Verification-guided  𝑤𝑇 = 1 𝑤𝑈 = 0. 5 𝑤𝐹 = 0. 2<br>weighted majority  True    Uncertain  False<br>Output Output voting Candidate answers: 2 :      4    (    3,            1,    0    ) 𝑠𝑐𝑜𝑟𝑒 2 = 1 × 3 + 0.5 × 1 + 0.2 × 0 = 3.5<br>5 :      6    (    1,     1,    4    ) 𝑠𝑐𝑜𝑟𝑒 5 = 1 × 1 + 0.5 × 1 + 0.2 × 4 = 2.3<br>(i) SC CoT (ii) CSV 4 < 6 Verification-guided  3.5 > 2.3<br>Majority voting: 5ⅹ weighted majority  2 √<br>voting:<br>(a) (b)<br>**----- End of picture text -----**<br>


Figure 4: **(a)** Illustration of the Naive majority voting (Wang et al., 2023) and our Verification-guided weighted majority voting. The full pipeline of the proposed Verification-guided Weighted Majority Voting framework. We use the model to generate several different solutions. Then we detect the self-verification state of each solution, and classify them into three states: _True_ , _Uncertain_ , and _False_ . According to the state of the verification, we assign each solution a different weight, and use the classified result to vote the score of each possible answer. 

Table 1: Accuracy (%) on MATH dataset. **VW-voting** is the abbreviation for the verification-guided weighted majority voting. ( **Overall:** The results across various MATH subtopics (Hendrycks et al., 2021)) 

|weighted majority<br>2021))|voting. (|**Overall**|**:** The results across various MATH subtopics (Hendrycks et al.|
|---|---|---|---|
|GPT-4|**Code-based**<br>**Verifcation**<br>✗|**VW-**<br>**Voting**|Intermediate<br>Precalculus<br>Geometry<br>Number<br>Counting &<br>PreAlgebra<br>Algebra<br>Overall<br>Algebra<br>**–**<br>**–**<br>Theory<br>Probability<br>**–**<br>**–**<br>MATH|
|||✗|-<br>-<br>-<br>-<br>-<br>-<br>-<br>42.20|
|GPT-3.5<br>GPT-4 (CoT)<br>GPT-4 (PHP)|✗<br>✗<br>✗|✗<br>✗<br>✗|14.6<br>16.8<br>22.3<br>33.4<br>29.7<br>53.8<br>49.1<br>34.12<br>23.4<br>26.7<br>36.5<br>49.6<br>53.1<br>71.6<br>70.8<br>50.36<br>26.3<br>29.8<br>41.9<br>55.7<br>56.3<br>73.8<br>74.3<br>53.90|
|GPT4-Code|✗|✗|50.1<br>51.5<br>53.4<br>77.2<br>70.6<br>86.3<br>83.6<br>69.69|
|GPT4-Code +**CSV**<br>_Improvement_<br>GPT4-Code +**CSV**+**Voting**<br>_Improvement_|✓<br> <br>✓|✗<br>✓**(k=16)**|56.6<br>53.9<br>54.0<br>85.6<br>77.3<br>86.5<br>86.9<br>73.54<br>**+6.5**<br>**+2.4**<br>**+0.6**<br>**+7.6**<br>**+6.7**<br>**+0.2**<br>**+3.3**<br>**+3.85**<br>**74.4**<br>**67.8**<br>**64.9**<br>**94.1**<br>**89.0**<br>**91.6**<br>**95.6**<br>**84.32**<br>**+24.3**<br>**+16.3**<br>**+11.5**<br>**+16.9**<br>**+18.4**<br>**+5.3**<br>**+12.0**<br>**+14.63**|



Note that this astonishingly high result is based on the strong abilities of the base model GPT4-Code, and our method amplifies its good qualities of GPT4-Code, with the ability to verify solutions. 

Note that although adding Code-based Self-verification can improve the performance of every individual subject, the extent of improvement varies from subject to subject, from 7.6% to only 0.6%. In particular, the Geometry problem only has an increased accuracy of 0.6%, even though the original GPT4-Code accuracy is only 54.0%, which is low among the subjects. This discrepancy may be attributed to the fact that solving geometry problems often requires multi-modality (Chen et al., 2023), a concept beyond the scope of this paper. 

## 4.2 PERFORMANCE ON OTHER DATASETS 

In addition to the challenging MATH dataset, we have also performed our method on other reasoning datasets such as GSM8K (Cobbe et al., 2021), MMLU-Math, and MMLU-STEM (Hendrycks et al., 2020). The corresponding results can be viewed in Tab. 2 and Tab. 3. When integrated on top of GPT- 

8 

Table 2: Performance on GSM8K dataset. Table 3: Performances on MMLU dataset. 

|Table 2: Performance on GSM8K dataset.|Table 3: Performances on MMLU dataset.|
|---|---|
|Method<br>Sampled paths<br>Accuracy(%)<br>GPT-3.5 (5-shot)<br>–<br>57.1<br>GPT-4 (5-shot CoT)<br>–<br>92.0<br>GPT-4 (PHP)<br>40<br>96.5<br>GPT-4 (Model selection)<br>15<br>96.8<br>GPT4-Code<br>-<br>92.9<br>GPT4-Code +**CSV**+**Voting**<br>**5**<br>**97.0**|Method<br>Dataset<br>Accuracy(%)<br>Few-shot|
||Chinchilla (Hoffmann et al., 2022)<br>Math<br>35.7<br>5-shot<br>Galactica (Taylor et al., 2022)<br>Math<br>41.3<br>5-shot|
||GPT4-Code<br>Math<br>87.5<br>zero-shot<br>GPT4-Code +**CSV**+**Voting**<br>Math<br>**89.2**<br>**zero-shot**|
||LLaMA 2<br>STEM<br>58.0<br>5-shot<br>OpenLLM<br>STEM<br>70.6<br>5-shot<br>GPT-4<br>STEM<br>82.7<br>zero-shot<br>GPT4-Code<br>STEM<br>86.8<br>zero-shot<br>GPT4-Code +**CSV**+**Voting**<br>STEM<br>**87.0**<br>**zero-shot**|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0009-02.png)


**----- Start of picture text -----**<br>
(a) Level (b) Subject<br>100 90 Level 1<br>Level 2<br>90 80 Level 3<br>Level 4<br>80 Level 5<br>70 Overall<br>70 number_theory<br>60 precalculus<br>60 algebra<br>prealgebra<br>50 50 intermediate<br>algebra<br>40 40 counting andprobability<br>geometry<br>0 1 2 3 0 1 2 3<br>Code Usage Frequency Code Usage Frequency<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 5: The four points on each curve correspond to results using **Prompt 1** , **Prompt 2** , **Basic Prompt** and **Code-based Self-verification Prompt** , respectively. **(a)** The accuracy of different levels at various code usage frequencies. **(b)** The accuracy of different subjects at various code usage frequencies. 

4-code, our method outperforms other methods in the competition, achieving state-of-the-art results across all datasets. Other subjects in MMLU benchmarks are provided in Fig. 8. A comparative analysis of our results with those of previous state-of-the-art techniques and open-source models is also provided. 

Tab. 2 illustrates that verification-guided majority voting is an effective framework to reduce the number of sampled paths, compared to GPT-4 with model selection (Zhao et al., 2023) and PHP (Zheng et al., 2023). 

Tab. 3 presents a comparison of our model’s performance with existing models (Hoffmann et al., 2022; Taylor et al., 2022) on the MMLU-Math dataset and with state-of-the-art open-sourced models[4] on MMLU-STEM. The open-source models remain significantly outpaced by their closedsource counterparts. To address this gap, we have made the dataset and will make it publicly available in the near future. Our intention is to facilitate the fine-tuning of open-source LLMs. For example, the open-source model LLaMA 2 (Touvron et al., 2023) can potentially utilize this data to further bolster its math reasoning capabilities. 

## 4.3 CODE USAGE FREQUENCY OF PROPOSED PROMPTS 

Analogous to the approach taken in Sec. 3.1, we gather data to elucidate the correlation between accuracy and Code Usage Frequency across various dimensions - prompts (proposed CSV prompt as well as prompts used in pilot experiments), subjects, and difficulty levels. As shown in Fig. 5, the model’s behavior is in accordance with our expectations when adding the code-based prompts. Each line in Fig. 5 has an obvious trend of going upwards, proving that the increase of Code Usage Frequency induces a general improvement in accuracy. The performance gain when using more code is more obvious in the higher difficulty levels, while in lower levels, the performance gain is not very prominent, as shown in Fig. 5 (a). Also, the Code Usage Frequency increases steadily with 

> 4https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard 

9 

Table 4: Comparison Self-verification with/without explicit code-based prompt ( **Overall** :The results across various MATH subtopics (Hendrycks et al., 2021)) 

|GPT4-Code<br>Interpreter|**Verifcation**<br>**Method**|Intermediate<br>Precalculus<br>Geometry<br>Number<br>Counting &<br>PreAlgebra<br>Algebra<br>Overall<br>Algebra<br>**–**<br>**–**<br>Theory<br>Probability<br>**–**<br>**–**<br>**–**|
|---|---|---|
||without Verifcation|50.1<br>51.5<br>53.4<br>77.2<br>70.6<br>86.3<br>83.6<br>69.69|
||Nature Language<br>Code-based|52.6<br>48.7<br>50.8<br>79.9<br>72.5<br>83.1<br>82.6<br>69.29<br>**+2.5**<br>**-2.8**<br>**-2.6**<br>**+2.7**<br>**+1.9**<br>**-3.2**<br>**-1.0**<br>**-0.40**<br>**56.6**<br>**53.9**<br>**54.0**<br>**85.6**<br>**77.3**<br>**86.5**<br>**86.9**<br>**73.54**<br>**+6.5**<br>**+2.4**<br>**+0.6**<br>**+8.4**<br>**+6.7**<br>**+0.2**<br>**+3.3**<br>**+3.85**|



the increase of difficulty levels. This shows that the harder math problems require more frequent code usage, which implies that invoking code multiple times might be an important reason why GPT4-Code have such an advantage in solving difficult math problems. There is a similar trend in Fig. 5 (b). 

## 4.4 ABLATION STUDY AND DISCUSSION 

**Comparisons between Natural Language and Code-based Self-Verification:** to underscore the significance of code in the self-verification stage, we employed a distinct natural language selfverification prompt. In this approach, GPT4-Code is directed to verify the solution through natural language instead of relying on code-based verification, as presented in Tab. 4. The accuracy achieved with this method was slightly lower than that of the _Basic Prompt_ . Moreover, we observed a decline in accuracy for 4 of the 7 subtopics, indicating that relying solely on natural language verification can not only compromise accuracy but also negatively impact performance. In contrast, code-based verification enhances accuracy across all 7 subtopics when compared to the _Basic Prompt_ . 

**Analysis of Verification-guided Weighted Majority Voting:** we initially compiled the confusion matrix (TP/TN/FP/FN), capturing solutions with self-verification that matches the _True_ and _False_ states mentioned in Eq. 1 from five distinct sampled paths. The details of the confusion matrix are presented in Appendix A.1.1. From this data, we computed Precision, Recall, and Accuracy. (Solutions in the _True_ state are seen as positive.) The results are presented in Fig. 6. In comparison to Accuracy, we observed numerical enhancements of 22.3% and 5.6% in the average Precision and Recall, respectively. In particular, the average Precision registered at 95.88%. This implies that the Accuracy has the potential to become much higher, if more solutions reach the verified _True_ state before giving the final answer. 

**Hyper-parameters ablation in Verification-guided Weighted Majority Voting:** we also performed ablation studies on the hyper-parameter _wv ∈{w_ **T** _, w_ **U** _, w_ **F** _}_ in Eq. 1. When the hyperparameter setting satisfied _w_ **T** _> w_ **U** _≥ w_ **F** , the performance of the verification-guided weighted majority voting consistently surpassed that of the naive majority voting methods across all sampled paths. In contrast, when we set the hyper-parameter ( _w_ **T** = 0 _._ 5 _, w_ **U** = 0 _._ 5 _, w_ **F** = 1), the performance under this configuration was worse than the naive majority voting. Therefore, our proposed method, verification-guided weighted majority voting, is easy to tune and robust. 

## 5 CONCLUSION AND LIMITATION 

In this paper, we begin with pilot experiments on GPT4-Code to explore how its use of code impacts its performance in mathematical reasoning. By analyzing Code Usage Frequency and accuracy, we determine that GPT4-Code’s skill in solving math problems can be largely attributed to its ability to generate and execute code, as well as its effectiveness in adjusting and rectifying solutions when confronted with implausible execution outputs. Expanding on this understanding, we introduce the ideas of explicit code-based self-verification and verification-guided weighted majority voting, with the goal of enhancing GPT4-Code’s mathematical capabilities. 

However, there are limitations in our work that we plan to explore further in the future. Firstly, our analysis and improvements are currently focused on GPT4-Code, which is somewhat restrictive. We aim to apply the methods to other LLMs. Secondly, our explicit code-based self-verification and verification-guided weighted majority voting technique could potentially create more accurate datasets. These datasets would include detailed step-by-step code-based solution generation and 

10 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0011-00.png)


**----- Start of picture text -----**<br>
(a) (b)<br>Precision<br>Recall 84<br>Accuracy<br>100<br>82<br>Avg.<br> 95.88<br>90 80<br>78<br>80 Avg.<br> 79.11<br>76<br>Avg. 1/0/0<br> 73.54<br>0.5/0.5/1<br>70<br>74 1/0.5/0.2<br>1/1/1<br>(Majority Voting)<br>72<br>60<br>0 1 2 3 4 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16<br>Different Reasoning Path # Sampled Reasoning Paths<br>Value<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 6: **(a).** shows the precision, recall, and accuracy on different reasoning paths. **(b).** shows the accuracy in response to the number of sampled reasoning paths when the weight is set to different values. 

code-based validation, which could help improve open-source LLMs like LLaMA 2 (Touvron et al., 2023) and enhance their mathematical abilities. Although we haven’t yet investigated this approach, we leave it for future work. 

11 

## REFERENCES 

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 

- Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks, 2022. 

- Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. Theoremqa: A theorem-driven question answering dataset, 2023. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 

- Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting for multi-step reasoning. arXiv preprint arXiv:2210.00720, 2022. 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pp. 10764–10799. PMLR, 2023. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Xiaodong Song, and Jacob Steinhardt. Measuring massive multitask language understanding. ArXiv, abs/2009.03300, 2020. URL https://api.semanticscholar.org/CorpusID: 221516475. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. 

- Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. 

- Takeshi Kojima, Shixiang (Shane) Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In Advances in Neural Information Processing Systems, volume 35, pp. 22199–22213, 2022. 

- Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022. 

- Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5315– 5333, 2023. 

- Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023. 

- Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. arXiv preprint arXiv:2209.14610, 2022. 

12 

- Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023. 

- A. Newell and H.A. Simon. Human Problem Solving. ACS symposium series. Prentice-Hall, 1972. ISBN 9780134454030. URL https://books.google.com.hk/books?id= h03uAAAAMAAJ. 

OpenAI. Gpt-4 technical report, 2023. 

- Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085, 2022. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=1PL1NIMMrw. 

- Yingxu Wang and Vincent Chiew. On the cognitive process of human problem solving. Cognitive Systems Research, 11(1):81–92, 2010. ISSN 1389-0417. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview. net/forum?id=_VjQlMeSB_J. 

- Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification, 2023. 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. ArXiv, abs/2305.10601, 2023. 

- Xu Zhao, Yuxi Xie, Kenji Kawaguchi, Junxian He, and Qizhe Xie. Automatic model selection with large language models for reasoning. arXiv preprint arXiv:2305.14333, 2023. 

- Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompting improves reasoning in large language models. arXiv preprint arXiv:2304.09797, 2023. 

- Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi. Least-to-most prompting enables complex reasoning in large language models, 2023. 

13 

## APPENDIX 

This Appendix contains two sections. The first section provides the experiment details, including the detailed experiment results on MATH and MMLU datasets. The second section presents some examples of GPT4-Code. 

## A EXPERIMENT DETAILS 

- A.1 DETAILED EXPERIMENT RESULT ON MATH DATASET 

## A.1.1 CONFUSION MATRIX 

A confusion matrix is a specific table layout that allows visualization of the performance of an algorithm. It’s particularly useful for classification problems, and we utilize it to analyze the performance of our verification process. 

The matrix itself is a two-dimensional grid, 2x2, for the binary classification of verification results. Each row of the matrix represents the instances in a predicted class, which is determined by the verification results given by the language model, while each column represents the instances in an actual class, which is determined by the actual correctness of the answer given by the model. Tab. 5 shows how the matrix looks for our verification process: 

## Table 5: Confusion Matrix of Verifcation 

||Answer Correct<br>Answer Wrong|
|---|---|
|||
|Verifcation True<br>Verifcation False|TP<br>FP<br>FN<br>TN|



Here’s what the four terms mean: 

- True Positive (TP): The cases in which the model’s verification result is ‘True’, and the answer is actually correct. 

- True Negative (TN): The cases in which the model’s verification result is ‘False’, and the answer is actually wrong. 

- False Positive (FP): The cases in which the model’s verification result is ‘True’, but the answer is actually wrong. 

- False Negative (FN): The cases in which the model’s verification result is ‘False’, but the answer is actually correct. 

This matrix is very helpful for measuring more than just straightforward accuracy, based on which Precision and Recall are two important metrics. They are defined in Eq. 3 and their meanings are as follows: 

- Precision is the fraction of relevant instances among the retrieved instances. It is a measure of the accuracy of the classifier when it predicts the positive class. 

- Recall is the fraction of the total amount of relevant instances that were actually retrieved. It is a measure of the ability of a classifier to find all the positive instances. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0014-17.png)


In other words, precision answers the question “What proportion of verified TRUE answers was actually correct?”, while recall answers “What proportion of actual correct answers was verified TRUE?” Given its meaning, verification-guided voting is bounded to be effective when the precision of verification is high. 

14 

## A.1.2 PYTHON PACKAGE USAGE ANALYSIS 

Tab. 6 outlines the usage of various Python packages in our experiments. Among them, we found that the sympy package is utilized most frequently, highlighting its central role in the computational tasks performed. 

Table 6: Python package usage frequency on MATH dataset. 

|Package|sympy|numpy|math|fractions|itertools|cmath|scipy|matplotlib|functools|collections|statistics|
|---|---|---|---|---|---|---|---|---|---|---|---|
|All|0.4168|0.0284|0.1590|0.0094|0.0034|0.0034|0.0016|0.0010|0.0004|0.0004|0.0002|
|Correct|0.3907|0.0241|0.1638|0.0110|0.0029|0.0026|0.0009|0.0003|0.0003|0.0006|0.0003|
|correct per code|0.3323|0.0205|0.1393|0.0094|0.0025|0.0022|0.0007|0.0003|0.0003|0.0005|0.0003|
|Wrong|0.4724|0.0383|0.1493|0.0058|0.0045|0.0052|0.0032|0.0026|0.0007|0|0|
|wrong per code|0.3194|0.0259|0.1009|0.004|0.0031|0.0035|0.0022|0.0018|0.0004|0|0|
|c/w per code|104%|79%|138%|238%|80%|63%|34%|14%|57%|NaN|NaN|



## A.2 DETAILED EXPERIMENT RESULT ON MMLU DATASET 

Fig. 8 illustrates that GPT4-Code performs relatively poorly in certain domains, such as engineering and the humanities, with a particularly marked deficiency in virology, where it achieves a score of less than 60%. These observations delineate specific areas that call for further investigation and refinement, thus outlining the direction for future improvements in the model. 

## B EXAMPLES 

In this section, we provide more examples. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0016-00.png)


Figure 7: GPT4-Code performance on MMLU dataset on: (a) subsubcategory, (b) category, (c) subcategory 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification_images/iclr_2024_Solving_Challenging_Math_Word_Problems_Using_GPT-4_Code_Interpreter_with_Code-based_Self-Verification.pdf-0017-00.png)


Figure 8: Model performance on MMLU dataset on: (a) subsubcategory, (b) category, (c) subcategory 

17 

**Math Problem** : 0 _._ 54 as a fraction in lowest terms. 

Sure, we can solve this problem by setting up an equation, solving it, and simplifying the fraction. 

Let’s call 0 _._ 54 as _x_ . 

The decimal 0 _._ 54 means that the digits ”54” repeat indefinitely. So, we can multiply _x_ by 100 (since ”54” is a two-digit number) to get another equation. This gives us 100 _x_ = 54 _._ 54. By subtracting the original equation from this new one, we eliminate the repeating decimal: 100 _x − x_ = 54 _._ 54 _−_ 0 _._ 54. Let’s perform these steps in the next Python cell. 

# define the symbol x = symbols(’x’) # define the equations eq1 = Eq(x, 0.54545454545454545454545454545454) # approximating the repeating decimal ...... 

## **RESULT** 

--------------------------------------------------------------------TypeError Traceback (most recent call last) ...... 

Apologies for the mistake. We can’t directly subtract two equations in Python. Instead, we should subtract the right-hand sides and left-hand sides of the equations separately to get a new equation. Let’s correct that. 

# subtract the right hand sides and left hand sides of the equations separately rhs = simplify(eq2.rhs - eq1.rhs) lhs = simplify(eq2.lhs - eq1.lhs) ...... 

## **RESULT Eq(99*x, 54.0)** 

After subtraction, we get the equation 99 _x_ = 54. Next, we can solve for _x_ to express 0 _._ 54 as a fraction. 

# solve for x solution = solve(new_eq, x) solution[0] 

**RESULT 0.545454545454545** The solution for _x_ is approximately 0 _._ 545454545454545, but we need to express it as a fraction. Let’s convert this decimal into a fraction. 

# convert the decimal solution to a fraction fraction = nsimplify(solution[0]) 

fraction 

**RESULT 6/11** 6 The decimal 0 _._ 54 can be expressed as the fraction 11[.] 

6 **Answer** : 11 

Table 7: An example in which GPT4-Code self-debugs when the code execution emits an error. 

18 

**Math Problem** : A bus comes by Jerry’s bus stop every 20 minutes starting at exactly 5:13 a.m. If Jerry shows up at exactly 8:35 a.m., how many minutes will he have to wait for the next bus? 

Step 1: Calculate the total minutes from 5:13 a.m. to 8:35 a.m. Total minutes from 5:13 a.m. to 8:35 a.m. is 180 + 22 minutes. Step 2: Calculate how many buses have come by during this time. Step 3: Calculate the time when the last bus before 8:35 a.m. came by. - The time when the last bus came by is 5 : 13 a.m. + 20 _× ⌊n⌋_ minutes. Step 4: Calculate the waiting time. Now, let’s calculate each step. 

# Step 1: Calculate the total minutes from 5:13 a.m. to 8:35 a.m. total_minutes = 180 + 22 

...... 

## **RESULT** 

## **-311** 

It seems that there’s a mistake in the calculations. The waiting time cannot be negative. This error might have occurred in Step 3, where I calculated the time of the last bus in minutes past midnight, but then in Step 4, I subtracted this from the total time from 5:13 a.m. to 8:35 a.m. I should correct Step 4 to calculate the time until the next bus, which is either 0 if a bus has just arrived at 8:35 a.m., or 20 minutes minus the time since the last bus if a bus arrived before 8:35 a.m. 

Let’s correct this. 

# Corrected Step 4: Calculate the time until the next bus time_since_last_bus = total_minutes - last_bus_time ...... 

## **RESULT 331** 

There still seems to be a mistake in the calculations. The waiting time calculated is significantly more than 20 minutes, which is not possible given that the bus comes every 20 minutes. Looking back at the calculations, it seems that the error is in Step 3 when calculating the time the last bus came. The last bus time should be calculated from 5:13 a.m., not from midnight. Let’s correct this. 

# Corrected Step 3: Calculate the time when the last bus before 8:35 a.m. came by. last_bus_time_corrected = 5 * 60 + 13 + num_buses * 20 

...... 

**RESULT 331** It seems there is still an error. Let’s try a different approach. 

1. Calculate the total minutes that have passed since the first bus at 5:13 a.m. until 8:35 a.m. 2. Since the buses come every 20 minutes, calculate how many complete cycles of 20 minutes have passed in the calculated total minutes. 

3. The time Jerry will have to wait is the remainder of the total minutes divided by 20 (which is the time since the last bus) subtracted from 20 (which is the bus cycle). This approach should give us the correct waiting time. Let’s calculate it. 

# Calculate total minutes from 5:13 a.m. to 8:35 a.m. total_minutes = 3 * 60 + 22 # 3 hours and 22 minutes ...... 

**RESULT 18** 

So, Jerry will have to wait for 18 minutes for the next bus. 

**Answer** : 18 

Table 8: An example in which GPT4-Code adjusts its solution based on the result of code execution. 

19 

**Math Problem** : Solve for _x_ : 3[2] _[x]_ + 19 = 10 _[x]_ . 

||**code 0**<br>This<br>is<br>a<br>diffcult<br>equation<br>to solve and there’s no simple<br>analytical<br>solution<br>using<br>ele-<br>mentary functions. In real world<br>scenarios,<br>such equations are<br>typically<br>solved<br>numerically.<br>For example, using a graphing<br>calculator,<br>a computer,<br>or a<br>mathematical software.<br>However, if you want an approx-<br>imation of the solution, it is ad-<br>visable to use numerical meth-<br>ods such as the bisection method,<br>Newton’s method, or others.<br>If you would like me to provide<br>an algorithm to fnd the solution<br>numerically, let me know.|**code 1**<br>Answer:<br>The equation you’re asking to<br>solve, 32_x_ + 19 = 10_x_, cannot<br>be solved algebraically due to<br>the nature of the functions and<br>operations involved.<br>Therefore,<br>we’ll<br>need<br>to<br>use<br>numerical<br>methods<br>such<br>as<br>Newton’s<br>method or Bisection method for<br>root fnding.<br>from scipy.optimize<br>import fsolve<br>import numpy as np<br># Defining the<br>function<br>def func(x):<br>return 3**(2*x) +<br>19 - 10**x<br># Initial guess for x,<br>it can be<br>anything, 1.0 is<br>chosen arbitrarily<br>here<br>x0 = 1.0<br># Using fsolve to find<br>the root<br>root = fsolve(func, x0<br>)<br>x = root[0]<br>x<br>**RESULT**<br>**1.001831315399663**<br>So the answer is 1.00.|**code more**<br>Answer:<br>Rewrite the equation as:<br>32_x −_10 + 19 = 0<br>Let’s denote 3_x_<br>=<br>_a_,<br>and<br>10_x_ =_b_<br>so_a_2 _−b_+ 19 = 0<br>now we have two equations:<br>1. _a_= 3_x_<br>2. _b_ = 10_x_ Let’s solve this with<br>Python.<br>from scipy.optimize<br>import root<br>import numpy as np<br># Define the system of<br>equations<br>def equations(vars):<br>x, a, b = vars<br>eq1 = a - 3**x<br>eq2 = b - 10**x<br>eq3 = a**2 - b +<br>19<br>return [eq1, eq2,<br>eq3]<br>.....<br>**RESULT:**<br>**0.0**<br>32_._0 + 19 = 100<br>The<br>left-hand<br>side<br>becomes<br>1+19=20 and the right-hand side<br>becomes 1.<br>Thus, the original<br>equation does not hold true for<br>x=0.<br>Let’s try a different initial guess<br>for the solution to see if that<br>helps.<br># Adjust initial guess<br>for x, a, b<br>x0 = [1, 3, 10]<br># Solve system of<br>equations<br>solution = root(<br>equations, x0)<br># Extract solution<br>x = solution.x[0]<br>x<br>**RESULT:**<br>**2.0000000000004463**<br>We know that 32_·_2 = 34 = 81<br>and 102 = 100. The left-hand<br>side becomes 100 and the right-<br>hand side is also 100. Thus, the<br>original equation holds true for<br>_x_= 2.||
|---|---|---|---|---|



**Answer** : 2 

Table 9: An example in which no restriction on code use succeeds while not using code and using code only once fails. 

20 

**Math Problem** : What is the sum of the digits in the terminating decimal representation of the fraction 54321[7] _·_ 2[8][?] **code 0 code 1 code more** The given fraction is Answer: Answer: First, we need to simplify the The denominator can be 4321 fraction. rewritten as 10 _[min]_[(7] _[,]_[8)] _×_ 5[7] _·_ 2[8] Given that 4321 = 2[5] _·_ 3[3] _·_ 5 _[max]_[(0] _[,]_[7] _[−]_[8)] _×_ 2 _[max]_[(0] _[,]_[8] _[−]_[7)] , To simplify this fraction, we can 5[0] _·_ 7[0] _·_ 11[0] _·_ 13[1] , we can where _min_ ( _a, b_ ) is the minimum notice that 5[7] _·_ 2[8] = 10[7] , which simplify the fraction 54321[7] _·_ 2[8] to of _a_ and _b_ , and _max_ ( _a, b_ ) would mean the fraction is equiv-alentnumberto 43214321 _×_ with10 _[−]_ the[7] , i.e.,decimalthe 22 _[−]_[5] _·_[3] 3[3] _··_ 555[0] _[−]_[7] _·_ 7[7] _·_ 2[0] .[8] _·_ 11[0] _·_ 13[1] = 3[3] _·_ 13[1] _·_ isNow _min_ the(7let’s _,_ 8)maximum,calculate _max_ (0 _,_ of 7the _−a_ value8)and, andof _b_ . point moved 7 places to the left. Next, we need to compute the _max_ (0 _,_ 8 _−_ 7). This would result in the number decimal representation and sum 0.0004321. the digits. min_ab = min(a, b) max_0_a_minus_b = max To get the sum of the digits in the # Compute the decimal terminating decimal representarepresentation of max_0_b_minus_a(0, a-b) = max tion of this number, we just need the fraction to add up the digits in 4321, since value = (3**3 * 13**1) (0, b-a) the zeros do not contribute to the / (2**3 * 5**7) sum. min_ab, So, the sum is 4+3+2+1 = 10. # Convert to string max_0_a_minus_b, max_0_b_minus_a Therefore, the answer is 10 and replace the decimal point **RESULT:** str_value = str(value) **(7, 0, 1)** .replace(’.’, ’’) Now, let’s simplify the fraction 4321 10[7] _×_ 5[0] _×_ 2[1][.] # Compute the sum of the digits # define the numerator digit_sum = sum(int( and denominator digit) for digit numerator = 4321 in str_value) denominator = 10**7 * 5**0 * 2**1 ...... digit_sum **RESULT: RESULT: 18 0.00021605** We exclude the zeros before the first non-zero digit and the decimal point as they do not contribute to the sum. Thus, the sum of the digits is the sum of 2, 1, 6, 0, and 5. Let’s calculate it. # digits in the fraction digits = [2, 1, 6, 0, 5] # calculate the sum of the digits sum_of_digits = sum( digits) sum_of_digits **RESULT: 14** So, the final answer is: 14 

**Answer** : 14 

Table 10: An example in which no restriction on code use succeeds while not using code and using code only once fails. 

21 

**Math Problem** : What is the sum of all positive integers _r_ that satisfy lcm[ _r,_ 700] = 7000 ? 

||**True**<br>The<br>least<br>common<br>multiple<br>(LCM) of two numbers _a_ and _b_<br>is given by the formula:<br>LCM(_a, b_) =<br>_|a × b|_<br>GCD(_a, b_)<br>Hence,<br>the<br>equation<br>can<br>be<br>rewritten as:<br>_r ×_700<br>GCD(_r,_700) = 7000<br>We also know that _r_ must be a<br>divisor of 7000. So, we’ll fnd all<br>the divisors of 7000.<br># Find all divisors of<br>7000<br>divisors_7000 = sympy.<br>divisors(7000)<br># Filter out the<br>divisors<br>r_values_corrected = [<br>r for r in<br>divisors_7000 if r<br>== 7000 * sympy.<br>gcd(r, 700) //<br>700]<br>......<br>**RESULT:**<br>**(7000, 1000)**<br>Let’s<br>verify<br>our<br>solution<br>by<br>checking whether the LCM of<br>each possible_r_and 700 is indeed<br>7000.<br># Verify the solution<br>lcm_values_corrected =<br>[sympy.lcm(r,<br>700) for r in<br>r_values_corrected<br>]<br>......<br>**RESULT:**<br>**(True, True)**<br>Upon verifcation, we fnd that<br>the<br>least<br>common<br>multiple<br>(LCM) of each possible value of<br>_r_and 700 is indeed 7000. Hence,<br>the sum of all positive integers_r_<br>that satisfy lcm[_r,_700] = 7000<br>is 1000 + 7000 = 8000. The<br>answer is 8000.|**Uncertain**<br>First,<br>let’s<br>fnd<br>the<br>prime<br>factorization of 7000.<br>Then,<br>we’ll generate all the<br>factors of 7000 and fnd which<br>of those factors are multiples of<br>700.<br>Finally, we’ll sum up all the<br>positive integers that satisfy the<br>condition.<br>import sympy<br># Get the prime<br>factorization of<br>7000<br>prime_factors = sympy.<br>factorint(7000)<br># Generate all factors<br>of 7000<br>factors = set()<br>for factor_2_count in<br>range(<br>prime_factors[2] +<br>1):<br>for factor_5_count<br>in range(<br>prime_factors<br>[5] + 1):<br>......<br>**RESULT:**<br>**(2: 3, 5: 3, 7: 1, [3500, 700,**<br>**7000, 1400], 12600)**<br>The prime factorization of 7000<br>is 23 _·_ 53 _·_ 71.<br>The factors of<br>7000 that are multiples of 700<br>are: 3500, 700, 7000, and 1400.<br>So, the sum of all positive inte-<br>gers_r_ that satisfy lcm(_r,_700) =<br>7000 is 3500 + 700 + 7000 +<br>1400 = 12600.<br>So the fnal answer is 12600.|**False**<br>In this problem, we are given<br>that_lcm_(_r,_700) = 7000.<br>Let’s frst fnd the prime factors<br>of 700 and 7000.<br>import sympy<br># Find the prime<br>factorization of<br>700 and 7000<br>factors_700 = sympy.<br>factorint(700)<br>......<br>**RESULT:**<br>**(2: 2, 5: 2, 7: 1, 2: 3, 5: 3, 7: 1)**<br>Since the LCM of r and 700 is<br>7000, r must contain all the prime<br>factors of 7000 that are not al-<br>ready in 700.<br># Generate all<br>combinations<br>combinations = list(<br>product(*[[1, i]<br>for i in<br>extra_factors]))<br>......<br>**RESULT:**<br>**([1, 5, 2, 10], 18)**<br>The possible values of r are 1, 2,<br>5, and 10.<br>To verify our answer, we can<br>check that the least common mul-<br>tiple of r and 700 is indeed 7000.<br># Remove 1 from the<br>possible values of<br>r<br>possible_r_values.<br>remove(1)<br>......<br>**RESULT:**<br>**False**<br>It seems that there’s a mistake.<br>Not all the calculated values of r<br>satisfy the given condition.<br>Let’s correct the calculation.<br># All the values of r<br>should have the<br>factors of 700<br>r_values = [700]<br>......<br>**RESULT:**<br>**([700, 1400, 3500, 7000], 18720)**<br>Therefore, answer is 18720.||
|---|---|---|---|---|



**Answer** : 8000 

Table 11: An example in which _True_ succeeds while _Uncertain_ and _False_ fails. 

22 

**Math Problem** : Let _A_ = (1 _, −_ 11 _,_ 2) _, B_ = (3 _, −_ 4 _,_ 1) _,_ and _C_ = ( _−_ 2 _,_ 1 _, −_ 1) _._ Compute ∠ _ABC,_ in degrees. 

|**True**<br>The angle _ABC_<br>is the an-<br>gle between the vectors _⃗_<br>_BA_ and<br>_⃗_<br>_BC_. We can fnd these vectors<br>as follows:<br>_⃗_<br>_BA_= _⃗A −⃗B_ = (_−_2_, −_7_,_1)<br>_⃗_<br>_BC_ = _⃗C −⃗B_ = (_−_5_,_5_, −_2)<br>The angle_θ_between two vectors<br>_⃗u_ and _⃗v_ can be found using the<br>formula<br>cos(_θ_) =<br>_⃗u · ⃗v_<br>_||⃗u||||⃗v||_<br>Let’s calculate these values.<br>import numpy as np<br># Define vectors<br>vec_BA = np.array([1,<br>-11, 2]) - np.<br>array([3, -4, 1])<br>......<br>**RESULT:**<br>**120.00000000000001**<br>Let’s now verify this answer.<br>_⃗u · ⃗v_ =_||⃗u|| · ||⃗v|| ·_cos(_θ_)<br>Substituting our values into this<br>formula, we should get the same<br>dot product (-27) that we calcu-<br>lated earlier. Let’s check this.<br># Compute the dot<br>product using the<br>formula<br>dot_product_verificatio<br>= magnitude_BA *<br>magnitude_BC * np.<br>cos(theta * np.pi<br>/ 180)<br>......<br>**RESULT:**<br>**-27**<br>which matches the dot product<br>we calculated earlier. Therefore,<br>the computed angle of 120_◦_is<br>correct.|n<br>**Uncertain**<br>the<br>cosine<br>of<br>the<br>angle<br>_θ_<br>between them is given by<br>cos_θ_ =<br>**u**_·_**v**<br>_||_**u**_|| ||_**v**_||_<br>In the given problem, we want<br>to fnd the angle between vectors<br>_−→_<br>_AB_and _−−→_<br>_BC_, where<br>_−→_<br>_AB_ =**B**_−_**A**= (2_,_7_, −_1)<br>_−−→_<br>_BC_ =**C**_−_**B**= (_−_5_,_5_, −_2)<br>Now, we’ll calculate the angle<br>using the formula forcos_θ_.<br>from numpy import<br>array, dot<br>from numpy.linalg<br>import norm<br>import math<br># Defining the vectors<br>A = array([1, -11, 2])<br>......<br>**RESULT:**<br>**60**<br>The angle∠_ABC_ is60_◦_.|**False**<br>The vector _−→_<br>_AB_<br>is given by<br>_B −A_, and the vector _−−→_<br>_BC_ is<br>given by_C −B_.<br>The cosine of the angle between<br>two vectors**u**and**v**is given by<br>cos(_θ_) =<br>**u**_·_**v**<br>_∥_**u**_∥∥_**v**_∥_<br>Let’s calculate.<br>import numpy as np<br># define the points<br>A = np.array([1, -11,<br>2])<br>......<br>**RESULT:**<br>**60.00000000000001**<br>The angle _ABC_<br>is approxi-<br>mately 60_._0_◦_. Let’s verify this.<br>We can solve forcos(_C_):<br>cos(_C_) = _a_2+_b_2_−c_2<br>2_ab_<br># calculate vector AC<br>AC = C - A<br># calculate the<br>magnitudes of AC<br>magnitude_AC = np.<br>linalg.norm(AC)<br>......<br>**RESULT:**<br>**False**<br>I apologize for the confusion;<br>there seems to have been a mis-<br>take in the verifcation.<br>The discrepancy arises due to the<br>directionality of the vectors.<br>Therefore, the initial calculation<br>was correct: the angle _ABC_ is<br>approximately60_._0_◦_.|
|---|---|---|
|**Answer**: 120_◦_|||



Table 12: An example in which _True_ succeeds while _Uncertain_ and _False_ fails. 

23 

