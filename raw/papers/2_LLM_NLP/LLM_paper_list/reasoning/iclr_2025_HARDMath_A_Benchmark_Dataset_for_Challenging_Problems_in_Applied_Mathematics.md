# HARDMATH: A BENCHMARK DATASET FOR CHALLENGING PROBLEMS IN APPLIED MATHEMATICS 

A PREPRINT 

**Jingxuan Fan,** _[∗]_ **Sarah Martinson** _[∗]_ **, Erik Y. Wang,** _[∗]_ **Kaylie Hausknecht,** _[∗]_ **Jonah Brenner, Danxian Liu, Nianli Peng, Corey Wang, Michael P. Brenner** School of Engineering and Applied Sciences Harvard University Cambridge, MA 02138, USA 

October 2024 

## **ABSTRACT** 

Advanced applied mathematics problems are underrepresented in existing Large Language Model (LLM) benchmark datasets. To address this, we introduce **HARDMATH** , a dataset inspired by a graduate course on asymptotic methods, featuring challenging applied mathematics problems that require analytical approximation techniques. These problems demand a combination of mathematical reasoning, computational tools, and subjective judgment, making them difficult for LLMs. Our framework auto-generates a large number of problems with solutions validated against numerical ground truths. We evaluate both open- and closed-source LLMs on **HARDMATH-MINI** , a subsampled test set of 366 problems, as well as on 40 word problems formulated in applied science contexts. Even leading closed-source models like GPT-4 achieve only 43.8% overall accuracy with few-shot Chain-of-Thought prompting, and all models demonstrate significantly lower performance compared to results on existing mathematics benchmark datasets. We additionally conduct a detailed error analysis to gain insights into the failure cases of LLMs. These results demonstrate limitations of current LLM performance on advanced graduate-level applied math problems and underscore the importance of datasets like **HARDMATH** to advance mathematical abilities of LLMs. 

_**Keywords**_ approximation, asymptotic analysis, benchmark dataset, LLM evaluation, mathematical reasoning 

## **1 Introduction** 

Many mathematical equations that arise in practical scientific and engineering problems cannot be solved analytically. Traditional mathematics courses tend to focus on equations with exact, analytical solutions, teaching only a limited set of techniques for solving them. Similarly, the mathematical reasoning datasets used to benchmark large language models (LLMs) are predominantly restricted to problems of this nature. However, many real-world mathematics problems involve integrals, ordinary differential equations (ODEs), and partial differential equations (PDEs) that do not have closed-form solutions and must be approached with a different set of techniques. While numerical solutions offer valuable insights, they often fail to provide intuition behind solutions behavior. A key approach in applied mathematics involves finding _approximate_ analytical solutions to complex problems using asymptotic and applied analysis techniques—methods that are largely underrepresented in existing LLM benchmark datasets. To address this gap, we introduce **HARDMATH** , a dataset specifically designed to focus on asymptotic reasoning in mathematics. This dataset captures a fundamentally different type of mathematical reasoning compared to other benchmarks and can be useful for evaluating LLMs’ abilities to make research-relevant approximations. 

**HARDMATH** consists of 1,466 problems inspired by a graduate-level course on asymptotic methods. These problems cover algebraic equations, ODEs, and integrals commonly encountered in real-world scientific and engineering contexts, 

> _∗_ Equal contribution 

All data and code used for this paper can be found at: https://github.com/sarahmart/HARDMath. 

A PREPRINT 

where exact solutions often do not exist. Instead, various asymptotic reasoning techniques are employed to find approximate but accurate solutions to these complex mathematical problems. 

A primary motivation for developing **HARDMATH** is the lack of benchmark datasets targeting the mathematical approximation methods required in many applications. While some recent works have begun to include university-level problems [Liu et al., 2024], most datasets focus on grade school- to high school-level mathematics problems [Amini et al., 2019, Hendrycks et al., 2021, Cobbe et al., 2021] whose solution methods only involve direct, ‘clean’ calculations. In contrast, **HARDMATH** targets applied mathematics problems that require _approximate_ analytical solutions; an equally important yet underrepresented aspect of mathematical reasoning. Solving such problems can be challenging even for individuals with highmathematical proficiency, as it requires advanced techniques from calculus, differential equations, and complex analysis. Additionally, computational tools are often needed to analyze the behavior of different terms in each equation and to derive numerical solutions that can serve as benchmarks for the approximations. Given the difficulty of these problems and their prevalence across science and engineering—where researchers may not always have a strong background in advanced mathematics—this level of difficulty and style of mathematics should be included in LLM evaluations. 

Rather than relying on the typical approach of collecting problems from textbooks, standardized tests, or competitions, as seen in most existing datasets, we developed algorithms to automatically generate problems and their step-by-step solutions. We implemented a comprehensive testing methodology to evaluate the mathematical reasoning abilities of leading LLMs in the domain of approximation methods. Our dataset includes a larger set **HARDMATH** that can be used for model developments (e.g. novel prompting techniques or fine-tuning), as well as two test sets - **HARDMATH-MINI** and **WORD-PROBLEMS-HARDMATH** , which are used to assess LLM performance. We present an evaluation accuracy summary and error mode analyses. Our results demonstrate that the performance of current LLMs on these problems is poor, highlighting significant room for their improvement on these challenging asymptotics problems. 

## **2 Related work** 

## **2.1 Mathematical datasets** 

LLMs have shown promising capabilities in mathematics. However, evaluating and expanding the full extent of these abilities requires diverse datasets with problems that go beyond basic arithmetic or elementary word problems. Existing benchmarks often focus on these simpler domains, with a gap in addressing graduate-level applied mathematics problems that demand a deeper understanding and diverse, multi-modal analytical skills. Most mathematics datasets for evaluating or training LLMs contain samples that either present the problem directly or within a constructed narrative context. Notable examples of these datasets include **MATH** (12,500 high school competition-style problems) [Hendrycks et al., 2021], **GSM8K** (8,500 multistep grade-school problems) [Cobbe et al., 2021], **MATHQA** (37,000 GRE/GMAT-level multiple-choice problems) [Amini et al., 2019], and **ODYSSEY-MATH** (387 hand-curated problems across various difficulty levels) [Netmind.AI, 2024]. While these existing datasets are valuable for assessing LLM math performance in certain areas, most are limited in scope and complexity. 

Recent efforts target more advanced problems that are most often manually-sourced. Relevant works include **JEEBENCH** [Arora et al., 2023] and a subset of the **MATHBENCH** dataset [Liu et al., 2024], both of which cover some college-level topics including simple ODEs and multivariable calculus. More advanced-level problems are presented in **GHOSTS** , which contains a **GRAD-TEXT** subset—a collection of 130 exercises from graduate-level mathematics textbooks in functional analysis, topology, and probability theory [Frieder et al., 2024]—and in **ARB** , which features a small set of university-level formal mathematics problems from prior qualifying examinations in the mathematics departments at Harvard University and the University of California, Berkeley [Sawada et al., 2023]. However, these datasets are limited by their size and scalability; datasets created by scraping textbooks or similar resources are generally quite small and difficult to broaden easily. Most of these challenging datasets also focus on abstract, formal mathematics and exclude other forms of mathematical reasoning. Finally, textbook problems are often protected by copyright, which can complicate their public use. 

Existing datasets (summarized in Table 1) thus lack the scale and specific focus needed to evaluate LLMs on advanced mathematical problems that may be highly useful for scientific research. **HARDMATH** aims to address these limitations by offering a large collection of challenging applied mathematics problems inspired by a graduate-level course on asymptotic methods. It emphasizes problems that require diverse mathematical approaches, numerical calculations, and subjective judgment, mirroring the complexity of problems faced by researchers in a variety of domains. Code for auto-generating the problems in **HARDMATH** can be used to generate any number of additional problems, which is a unique and powerful feature for scaling LLM benchmarking and model developments like novel prompting techniques or fine-tuning. A key area of interest in current LLM research is developing models that can effectively use external tools. The problems in our dataset are unique because they involve approximate solutions that cannot be formalized 

2 

A PREPRINT 

Table 1: Comparison of **HARDMATH** with related datasets. Note that for all the datasets excluding **MATH** and **GSM8K** , we report the number of relevant problems at a comparable difficulty to our dataset (e.g., **THEORYKNOWLEDGE-COLLEGE** in **MATHBENCH** , and **GRAD-TEXT** and **HOLES-IN-PROOFS** from **GHOSTS** .) **HARDMATH** is the largest graduate-level dataset. 

|Dataset|Size|Problem Sourcing|Diffculty|
|---|---|---|---|
|**MATH**[Hendrycks et al., 2021]|12.5K|Manual|High School|
|**GSM8K**[Cobbe et al., 2021]|8.5K|Manual|Grade School|
|**MATHBENCH-T**[Liu et al., 2024]|632|Manual, Algorithmic|Undergraduate|
|**JEEBENCH**[Arora et al., 2023]|236|Manual|High School|
|**GHOSTS**[Frieder et al., 2024]|190|Manual|Graduate|
|**ARB**[Sawada et al., 2023]|34|Manual|Graduate|
|**HARDMATH**(Ours)|1.4K|Algorithmic|Graduate|



using tools like Lean or similar software. To excel in this benchmark, LLMs must integrate tool use with sophisticated reasoning. This makes **HARDMATH** particularly valuable for benchmarking and developing LLMs capable of effective tool use, setting it apart from other mathematical datasets. 

## **2.2 Recent interest in advanced mathematical reasoning** 

As LLMs continue to improve, there has been growing interest in developing more challenging benchmarks, especially in mathematics. A notable example is the recent open challenge, _Humanity’s Last Exam_ , which aims to create the world’s most difficult public AI benchmark, requesting questions that "only exceptional individuals can answer correctly," do not involve "straightforward calculation/computation," and are written by individuals with PhD-level academic training [Hendrycks and Wang, 2024]. Similarly, frontier models have been advancing quickly, and many are explicitly focused on quantitative and scientific reasoning, such as OpenAI’s recent o1 series. In line with our motivation for developing **HARDMATH** to better track the progress of LLMs, OpenAI argues that "recent frontier models do so well on **MATH** and **GSM8K** that these benchmarks are no longer effective at differentiating models" [OpenAI, 2024]. 

## **3 Datasets** 

## **3.1 HARDMATH design choices** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0003-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0003-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0003-10.png)


**----- Start of picture text -----**<br>
(a)  HARDMATH-MINI  dataset (b)  HARDMATH  dataset<br>**----- End of picture text -----**<br>


Figure 1: Breakdowns of the **HARDMATH-MINI** (left) and the **HARDMATH** (right) datasets. 

Here, we detail the **HARDMATH** dataset, which contains problems on polynomial nondimensionalization, polynomial root-finding, ODEs, integrals, and word problems that contextualize each of these. A sample of a **HARDMATH** integral problem is shown in Box 1. **HARDMATH** contains four problem classes with seven distinct problem types, as well as 

3 

A PREPRINT 

40 handwritten word problems contextualizing the problem types. The main **HARDMATH** dataset, which can be used for model developments like fine-tuning, contains 1,060 problems, and the evaluation dataset **HARDMATH-MINI** , which we use in this paper to benchmark LLM performance, contains 366 problems. Figure 1 provides a detailed breakdown of these datasets. 

Box 1: Sample Integral Problem and Solution **Problem:** Consider the integral _I_ ( _ϵ_ ) = �056 _._ 00 _ϵ_ +2 _._ 0 _x_[6] _[.]_[0] +2 _._ 0 _x_[9] _[.]_[0] 1+5 _._ 0 _x_[11] _[.]_[0] +5 _._ 0 _x_[13] _[.]_[0] _[dx.]_[ Develop analytical formulas that approximate] _I_ ( _ϵ_ ) for different regimes of _ϵ_ . **Solution:** The integral is of the form _I_ ( _ϵ_ ) = �056 _ϵ_ + _P_ 1 ( _x_ ) _[dx]_[ where] _[ P]_[(] _[x]_[)][ is a polynomial.][Thus,] **[ its value can be estimated] as the product between a height and a width** . Since the integrand is maximized at _x_ = 0, the height can be set to[1] _ϵ_[.] For **small** _ϵ_ , we define the width as the point where the integrand becomes half of its maximum height. This corresponds to solving for _x_ given _P_ ( _x_ ) = _ϵ_ . Applying **dominant balance** , considering the term in _P_ ( _x_ ) with the smallest degree, the width is approximated as � 2 _._ 01 _∗ϵ_ �1 _/_ 6 _._ 0. Therefore, the analytical approximation of the integral for small _ϵ_ is _I_ ( _ϵ_ ) = _ϵ_ 0[0] _.[.]_ 8909[8333] _[.]_ For an **intermediate regime** where _ϵ_ is large, we also define the width based on the term with the largest degree. The width is approximated as � 5 _._ 01 _∗ϵ_ �1 _/_ 13 _._ 0. Therefore, the analytical approximation of the integral for large _ϵ_ is _I_ ( _ϵ_ ) = _ϵ_ 0[0] _.[.]_ 7647[8333][.] If the width of the integral exceeds the range of integration, we consider one more regime for **very large** _ϵ_ . The width is then just the range of integration, so in this regime, the integral can be approximated as _[L] ϵ_[.][Therefore, the analytical] approximation of the integral for very large _ϵ_ is _I_ ( _ϵ_ ) =[56] _ϵ_[.] Altogether, the solutions at small, large, and very large _ϵ_ are _ϵ_ 0[0] _.[.]_ 89[83] _[,] ϵ_[0][0] _[.][.]_[76][83] _[,]_[56] _ϵ[.]_ 

One key commonality between all **HARDMATH** problems is the use of the _Method of Dominant Balance_ in calculating solutions. This reduces an equation to only the terms that ‘dominate’ the behavior of the solution and can significantly simplify the equation [Bender and Orszag, 2013]. In addition to the Method of Dominant Balance, our problems also involve other sophisticated mathematical techniques, such as checks for self-consistency and the use of numerical methods. The combination of these tools captures several key aspects of mathematical modeling, including the combined use of computational and analytical techniques. Additionally, subjective choices about the regimes of solution space to consider, the number of terms to include in approximate expressions, and the approximation methods themselves must be made on a case-by-case basis with rigorous mathematical justification. Both of these aspects are potentially difficult tasks for existing LLMs. In Box 1, we highlight the methods and regimes relevant to solving a sample integral problem. 

## **3.2 Dataset generation and verification** 

The dataset generation procedure is outlined in Fig. 2. Code for data generation uses `SymPy` [Meurer et al., 2017], a library for symbolic mathematics, and `SciPy` , a library for scientific computing [Virtanen et al., 2020], to implement the mathematical procedures required for obtaining approximate, analytical solutions. Problems are generated by combining randomly selected coefficients, functional forms, and initial conditions uniquely defined for each problem (described in Appendix A)—no duplicate problems are included. Solutions are generated by navigating through a set of possible cases during the algorithmic problem-solving strategy. Each mathematical step is embedded in explanatory text so **HARDMATH** solutions match the style and rigor of traditional problem set solutions. The main results for all problems are included in boxed environments in the solution explanations to distinguish them from the rest of the text. This follows the formatting convention used in other mathematics datasets designed for LLM benchmarking, such as **MATH** [Hendrycks et al., 2021]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0004-06.png)


Figure 2: Flowchart detailing the data generation procedure for **HARDMATH** problems. 

4 

A PREPRINT 

For each problem type, the dataset includes: 1) L[A] TEX-formatted problem statements with prompts, 2) L[A] TEX-formatted solution steps and final analytical answer(s), 3) demonstration of the accuracy of the analytical results by comparing with numerical solutions, and 4) metadata descriptors of the problem and solution types. For every problem type, we select evaluation points in each solution regime and calculate the relative error between the analytical solution and the numerical solution at these points. Problems were included in **HARDMATH** only if their approximate solutions had less than 10% error from the numerically calculated ground-truths. For the polynomial root correction problems, we also confirm that the corrections improve the original approximation. While manually verifying each solution step-by-step is impractical for a dataset of this size, our validation process ensures a high level of confidence in the accuracy of the solutions provided. Verification of **HARDMATH-MINI** is performed by plotting analytical solutions against numerical ground truths for a range of values in each regime (see example in Appendix A.1). This semi-automated step provides an easy means for visual human-verification—plots will directly reveal whether analytical and numerical solutions correspond in the correct solution regimes. 

## **3.3 Problem types** 

For all problem types discussed in this section, details regarding parameters used to generate problems and mathematical solution techniques are provided in Appendix A.2. 

## **3.3.1 Nondimensionalization of polynomials** 

Nondimensionalization is a technique to simplify equations by reducing the number of parameters [Evans, 1972]. In **HARDMATH** , the first type of polynomial used for nondimensionalization demonstration contains symbolic coefficients and is of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0005-06.png)


Nondimensionalization converts this to the form _ϵy[n]_[1] + _y[n]_[2] + 1 _._ The second type contains numerical coefficients and is of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0005-08.png)


which can be simplified to _ϵy[n]_[1] _± y[n]_[2] _±_ 1 given a specific numerical value of _ϵ_ . 

## **3.3.2 Polynomial root-finding** 

Exact formulas exist for quadratic, cubic, and quartic equations, but deriving them for quintic or higher-order polynomials is not possible [Stewart, 2015]. **HARDMATH** includes approximate root-finding examples for higher order polynomials of the form _ϵx[n]_[1] _± x[n]_[2] _±_ 1 (example in Appendix A.2.2). The goal is to solve for roots in terms of _ϵ_ using the method of dominant balance for small and large positive _ϵ_ regimes. 

## **3.3.3 Polynomial root correction terms** 

The use of two-term dominant balances—such as in the previous problem type—neglects terms and introduces an error. We can calculate a correction term _δ_ to reduce this error. Suppose the true roots _x[∗]_ of a polynomial are given by _x[∗]_ ( _ϵ_ ) = _x_ ( _ϵ_ ) + _δ,_ where _x_ is our approximation to the root and _δ_ is the error term. Plugging the roots _x[∗]_ ( _ϵ_ ) = _x_ ( _ϵ_ ) + _δ_ into the polynomial allows one to use a Taylor expansion of _δ_ around _x_ to solve for the correction _δ_ . Appendix A.2.3 shows a full worked solution. 

## **3.3.4 Nonlinear ordinary differential equations** 

We generate nonlinear third-order ODEs for which there do no exist exact analytical solutions and provide approximate formulae for small and large _x_ regimes, where the small _x_ regime is near _x_ = 0 and the large _x_ regime typically involves the solution diverging (example in Appendix A.2.4). The method is robust for higher-order problems, but for simplicity **HARDMATH** includes only third-order ODEs. 

## **3.3.5 Traditional integrals** 

We consider integrals of the form _I_ ( _ϵ_ ) = �0 _a ϵ_ + _P_ 1 ( _x_ ) _[dx,]_[ where] _[ P]_[(] _[x]_[)][ is an arbitrary polynomial.] **[HARDM][ATH]**[ provides] approximations of each integral in three regimes: small, intermediate, and large _ϵ_ . A full example is in Appendix A.2.5. 

5 

A PREPRINT 

## **3.3.6 Laplace integrals** 

_b_ We consider integrals of the form _I_ ( _x_ ) = � _a[g]_[(] _[t]_[)] _[e][±][xf]_[(] _[t]_[)] _[dt]_[, which can be approximated using Laplace’s Method when] _x_ is very large because the integral’s value is dominated by the region around _t_ 0 [Bender and Orszag, 2013]. Depending on where the minimum is, the approximation is either 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0006-03.png)


See and Appendix A.2.6 for examples of a Laplace integral problem with solutions. 

## **3.4 Word problems in context** 

One motivation for creating **HARDMATH** is to help LLMs recognize and solve problems where approximation techniques are needed. To evaluate how LLMs perform on such problems in realistic scenarios, we develop a smaller dataset of 40 manually-generated word problems (example in Box 2). Although this dataset is smaller than our hand-verified evaluation set, it is large enough to evaluate the effect of additional context in the problem statement on LLM accuracy. 

Box 2. Sample Word Problem with Context 

The density of fish at different points along a certain path in a lake can be modeled as ( _ϵ_ + _x_[2] + _x_[5] ) _[−]_[1] _,_ where _x_ represents the distance from the shore in kilometers (ranging from 0 to 100 km), and _ϵ_ represents environmental factors that affect the fish density. To study the total presence of fish along the path, develop an approximate analytical formula for _I_ ( _ϵ_ ) given below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0006-09.png)


## **3.5 Automatic generation of context for word problems** 

Recognizing that manual context crafting for these problems is labor-intensive and lacks scalability, we conduct preliminary experiments on automatically generating contexts for word problems using a powerful closed-source LLM, such as GPT-4o. The generation process involves two steps: (1) We create a foundational set of mathematical problems and solutions following the methodology in Fig. 2. These problems serve as a starting point, capturing essential mathematical formulation without context. 2) We then prompt the LLM to generate real-world contexts to embed these problems and solutions based on a specified domain seed (e.g., physics). To ensure physical plausibility, a secondary verification LLM checks parameter ranges (e.g., energy must be non-negative) and assigns a 0-1 plausibility score. A score around 0.5 indicates that while no hard violations were found, the verifier cannot fully assess violations of less stringent domain priors. Only plausibity scores >0.5 are thus considered. Example prompts for both generator and verifier are available in Appendix A.2.7, Table 3. Using _ODEs_ as a demonstration, we batch generate 30 problems with `{domain_seed}` **physics** . Appendix A.2.8 shows an example of the original vs. context-embedded problem side by side, and Appendix A.2.9 6a demonstrates the set of generated problems has good context diversity, covering different sub-fields in **physics** . Moreover, most problems’ plausibility scores >0.5 , indicating queried math equations make good sense under generated contexts (A.2.9 6b). 

While we recognize limitations in capturing more subtle, domain-specific priors, this approach offers a promising step toward automating the generation of applied math problems in real scientific contexts. We plan to refine these methods in future work. 

## **4 Evaluation** 

## **4.1 Evaluation protocols** 

We conduct evaluations of various LLMs on **HARDMATH-MINI** , a carefully curated subset of 366 problems that matches the statistical composition of **HARDMATH** (Fig. 1). This smaller dataset is designed to optimize computational resources while retaining a sufficient number of questions to ensure consistent and reliable testing outcomes, thus maintaining the integrity of our evaluation. The evaluation focuses on four distinct problem types: 1) _Nondim_ includes nondimensionalization in symbolic and numerical form; 2) _Roots_ includes polynomial root-finding; 3) _ODEs_ includes nonlinear ODEs; and 4) _Integrals_ includes traditional and Laplace integrals. The input prompt for each problem contains 

6 

A PREPRINT 

the essential problem setup and a detailed description of the question. Additionally, hints specific to each problem type are provided to guide the format of the answer. When few-shot prompting is used, it adds a fixed set of paired problem-solution examples from the corresponding problem types. Example prompts can be found in Appendix A.3.1, Table 7. 

We evaluate model-generated responses by scoring them for accuracy using a combined protocol of automatic final answer assessment and procedural LLM-based grading. The automatic assessment follows methodology from Hendrycks et al. [2021], where models are prompted to enclose their final answers using the L[A] TEX `\boxed{}` command (Table 7). Evaluation then compares the model’s output within the `\boxed{}` command to the dataset solution. To handle different mathematical expression formats, we implement both `SymPy` -based equivalence checks and numerical evaluations. 

In addition to the standard automatic assessment of final answers, we develop a novel procedural grading approach leveraging LLMs, tailored to the unique evaluation challenges of our dataset: 1) Some problem types require complex, multi-step solution procedures (e.g. determining critical point in Laplace integral approximation) where a single cut-off criterion at the final answer cannot capture the full spectrum of model performance. Thus, grading intermediate steps in the solution procedure is necessary for comprehensive assessment. 2) **HARDMATH** targets the models’ ability to make human-like abstraction and approximation judgments. Some problem types allow a narrow range of solutions rather than a single exact one, as long as the reasoning is self-consistent and the final result falls within certain threshold to numerical ground truth. 

Inspired by LLMs’ ability to generate consistent ratings for response content and style [Hackl et al., 2023], we employ GPT-4o as a procedural grader. The model is prompted with a ground truth answer key and grading rubrics adapted from course grading guidelines for each problem type (example grading prompts in Appendix A.3.2 Table 5). We manually verify a subset of grading responses and found that LLM-based grading is closely aligned with human grading. Average score adjustment for each model and problem type is summarized in Appendix A.3.3 Table 6. We implement this procedural grading alongside automatic answer assessment for the problem types _Roots_ , _ODEs_ , and _Integrals_ . 

## **4.2 Model choice** 

We compare the performance of several closed- and open-source models on **HARDMATH** in zero- and few-shot settings with the Chain-of-Thought (CoT) [Wei et al., 2023] prompting. Closed-source LLMs include GPT-3.5 [Radford et al., 2018, 2019, Ouyang et al., 2022], GPT-4 [Achiam et al., 2023] and o1-mini [OpenAI, 2024a], open-source LLMs include Llama3 [AI, 2024] and CodeLlama [Meta, 2023]. We believe this subset of models to be representative of current LLM capabilities. We provide the prompts and hyper-parameters for LLMs evaluations in Appendix A.3.4 Table 7. 

## **4.3 Quantitative results** 

We present the accuracy of the models and prompting settings for each problem type and the combined evaluation set (Table 2, Figure 7). Few-shot CoT prompting significantly boosts performance for all models, with o1-mini and GPT-4 showing the greatest improvement, consistent with [Wei et al., 2023] (Figure 7a). Interestingly, although the o1-mini official prompting guide recommends simple prompting over CoT [OpenAI, 2024b], we observe fairly large performance increase for all problem types at 5 shot CoT compared to 0 shot. Performance increase with prompting behavior also shows problem type-specific patterns: Figure 8 demonstrates that performance saturates quickly for harder problem types like _ODEs_ . The varying performance increases among different problem types may be due to different error modes in model answers, which we discuss in the following section. It’s notable that o1-mini, though with much smaller parameter size, shows considerably better performance at all tested shot levels, confirming its optimized ability for STEM reasoning [OpenAI, 2024a]. 

Among closed-source models, o1-mini with 5-shot CoT prompting achieves the highest overall accuracy of 62.3%. GPT-4 at 5-shot CoT scores only 43.8%. Among open-source models, Llama3-8b with 5-shot CoT prompting achieves the highest overall accuracy of 20.2%. We discuss the performance of these representative models—o1-mini, GPT-4 and Llama3—on **HARDMATH-MINI** in comparison with established datasets, including **GSM-8K** [Cobbe et al., 2021], **MATH** [Hendrycks et al., 2021], and more advanced mathematics datasets like **GHOSTS** [Frieder et al., 2024]. 

Llama3-8b achieves a test accuracy of 30.0% on the **MATH** dataset with 4-shot CoT and 79.6% on the **GSM-8K** dataset with 8-shot CoT prompting [AI, 2024]. Testing Llama3-8b on **HARDMATH-MINI** results in an overall accuracy of 20.2% with 5-shot CoT prompting. GPT-4 (gpt-4-turbo-2024-04-09) is reported to achieve 72.2% accuracy on the **MATH** dataset with 0-shot CoT prompting [OpenAI, 2024c] and 92.0% on the **GSM-8K** dataset with 5-shot CoT prompting [Achiam et al., 2023]. On the **MINIGHOSTS** dataset, which also covers graduate-level mathematics, 

7 

A PREPRINT 

Table 2: Evaluation Accuracy (percentage) on the **HARDMATH** evaluation set. 

|Model|ALL|Nondim<br>Roots<br>ODEs<br>Integrals|
|---|---|---|
|**Closed-source models**<br>GPT-3.5 (0 shot)<br>6.04<br>5.05<br>17.2<br>1.39<br>3.33<br>GPT-3.5 (1 shot CoT)<br>14.2<br>6.11<br>29.3<br>6.94<br>18.2<br>GPT-3.5 (5 shot CoT)<br>24.6<br>24.3<br>35.0<br>16.2<br>23.1<br>GPT-4 (0 shot)<br>14.0<br>6.04<br>33.7<br>7.87<br>14.9<br>GPT-4 (1 shot CoT)<br>37.6<br>36.5<br>52.8<br>15.9<br>40.5<br>GPT-4 (5 shot CoT)<br>43.8<br>48.6<br>57.3<br>21.7<br>41.4<br>o1-mini (0 shot CoT)<br>29.8<br>38.1<br>24.3<br>10.2<br>32.5<br>o1-mini (5 shot CoT)<br>62.3<br>84.5<br>62.1<br>30.6<br>46.5<br>**Open-source models**|||
|Llama3-8b (0 shot)<br>Llama3-8b (5 shot CoT)<br>CodeLlama-13b (0 shot)<br>CodeLlama-13b (5 shot CoT)|3.67<br>20.2<br>1.94<br>9.79|0.50<br>11.5<br>4.63<br>2.52<br>17.9<br>17.1<br>12.0<br>28.1<br>0.00<br>8.73<br>1.85<br>0.50<br>8.41<br>13.1<br>9.7<br>9.57|



GPT-4 reaches an average score of 4.15 out of 5. We test GPT-4 on our **HARDMATH-MINI** dataset and obtained an overall accuracy of 43.8% with 5-shot CoT prompting. 

Finally, we include results on OpenAI’s new o1-mini, which is reported to achieve 90.0% accuracy on **MATH-500** with 0-shot CoT [OpenAI, 2024a]. Testing o1-mini on **HARDMATH-MINI** reveals a significant performance increase compared to results on other models on some (e.g. _Nondim_ ) but not all problem types. Overall accuracy with 5 shot CoT reaches 62.3%, substantially lower compared to performance on existing mathematics benchmarks. This indicates that the **HARDMATH** benchmark consists of problems that are still challenging and unfamiliar to even the most performant LLMs developed specifically for advanced reasoning. 

## **4.3.1 Extensions to word problems** 

To assess LLM’s ability to solve similar applied math problems in real science and engineering contexts, we also test GPT-4 (best performing model with a stable version) on a set of 40 hand-crafted word problems that included a mixture of _Nondim_ , _Roots_ , _ODEs_ , and _Integrals_ . We avoided additional prompt engineering, omitting the problem-specific hints listed in Table 4. This evaluation resulted in an overall accuracy of 28.1%. We plan to leverage the automated generation method as a basis to expand the number of word problems for future work. 

## **4.4 Fine-grained results** 

In addition to reporting the summarized test accuracy, we study the detailed breakdown of model responses at different correctness levels and summarize specific error modes of LLMs solving these challenging applied mathematics questions. This analysis helps us compare performance nuances and understand reasoning paths by model, prompting technique and question type. 

We first break down model performance by percentage of correct, partial and incorrect responses (Figure 3). This analysis reveals how few-shot prompting enhances model performance across varying problem types but through different strategies. Overall quantitative results already show that _ODEs_ are comparatively harder for all models while _Nondim_ problems appear to be the easiest (Figure 7b). For hard problems like _ODEs_ , full correctness is rare. Correctness level analysis shows that models tend to increase partial credit responses with CoT prompting, as they struggle to solve the problems entirely but manage to partially address them—in this case, starting with the easier small _x_ regime solutions. In contrast, for simpler problems like _Roots_ , advanced models like o1-mini and GPT-4 get more fully correct responses with increasing CoT shot number, demonstrating the models’ ability to understand the approximation reasoning procedure fully (Fig. 3). 

Second, we summarize the error modes of partial and incorrect responses to better understand the model’s reasoning pitfalls. Specifically, we want to dissect how CoT changes model performance on the level of detailed errors. Figure 4 uses GPT-4’s responses at 0 vs. 5 shots on problem type _Roots_ as an analysis example. This illustrates how 5 shot CoT prompting significantly alters the error structure compared to 0 shot. The most common error mode—incorrectly setting up dominant balance by considering only the leading term—diminishes substantially. Instead, errors shift to more 

8 

A PREPRINT 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0009-01.png)


Figure 3: Percentage of correct, partial, and incorrect responses for o1-mini, GPT-4 and Llama3, prompting conditions, and problem types. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0009-03.png)


Figure 4: GPT-4 error modes for problem type _Roots_ at 0 vs. 5 shot CoT prompting 

nuanced issues: 1) setting up correct dominant balances but missing certain cases, or 2) failing to calculate complex roots (examples of those error modes included in Box 3 and Appendix A.4.2). This shift indicates that CoT prompting improves the model’s understanding and application of dominant balance techniques, enabling it to move beyond intuitive yet incorrect simplifications. We are particularly curious to compare o1-mini’s error modes with existing models, given its specialization in STEM reasoning. However, in the current evaluation, o1-mini sometimes returns only the final answer without showing intermediate steps, making it difficult to trace the source of errors. We therefore leave this exploration for future work. 

## 3. Grading Response: _Roots_ 

**Question:** Consider the polynomial _P_ ( _x_ ) = _ϵx_[8] + _x_[4] _−_ 1 _._ Find approximate expressions for all roots of the polynomials in the limit of small positive _ϵ_ and large positive _ϵ_ . Only a single term approximation to the root is required. **Grading for Small Positive** _**ϵ**_ : 

**Model Response:** For small _ϵ_ : [1 _, −_ 1 _, i, −i_ ] **Ground Truth** : For small positive _ϵ_ : _−_ 4 _−_[1] 4 _−_[1] _[−][i]_ 4 _−_[1] _[i]_ 4 _−_[1] _[−]_[1] _[,]_[1] _[,][−][i,][i]_ � ~~�~~ _ϵ[,]_ ~~�~~ _ϵ[,]_ ~~�~~ _ϵ[,]_ ~~�~~ _ϵ[,]_ � The response only includes the roots from the balance _B_ + _C_ = 0 and completely misses the roots from the balance _A_ + _B_ = 0. Therefore, score for small positive _ϵ_ is 0 _._ 5 

9 

A PREPRINT 

## **5 Conclusion** 

We introduce **HARDMATH** , a new dataset covering several problem types from an advanced applied mathematics course that can be used to benchmark LLMs’ mathematical capabilities and perform model developments, including fine-tuning. This dataset consists of 1060 examples, and we additionally include 366 human-verified examples in **HARDMATH-MINI** and 40 human-verified ‘problems in context’ that we use to evaluate various leading LLMs. **HARDMATH** is unique in several ways. First, there do not exist large-scale mathematical datasets covering problems of similar difficulty from applied mathematics. Second, **HARDMATH** ’s problems and solutions are algorithmically generated, with automatic numeric validity checks and an easy visual means for human-verification, meaning that one could produce datasets of arbitrary size using our framework. This feature of **HARDMATH** is especially unique, since most existing mathematical datasets require manual problem-setting or curation from existing sources (many of which are not publicly accessible). 

Our evaluation highlights that while few-shot CoT prompting significantly improves model performance, especially for models like o1-mini and GPT-4, the overall accuracy on **HARDMATH-MINI** problems remains much lower compared to other existing benchmarks. This suggests that our dataset poses unique and challenging tasks that go beyond the boundaries of current LLM capabilities, particularly in approximation-oriented mathematical reasoning. These findings emphasize the need for further improvement in LLMs to address hard math problems. 

Our evaluation results use **HARDMATH-MINI** as a comprehensive test set; however, future work will fine-tune LLMs on the larger **HARDMATH** to improve performance. Additionally, while we have evaluated several frontier models, we plan to extend our evaluations to even more LLMs as they become available. This expanded evaluation should provide more detailed insights into performance disparities across different models, further advancing our understanding of LLMs’ capabilities in handling complex mathematical reasoning. 

## **Acknowledgments** 

We thank the students who participated in the initial stages of the Fall 2023 AM 201 final project. Thanks also to the Harvard Medical School Research Computing Consultant Group for their consulting services, which facilitated the computational analyses detailed in this paper. 

10 

A PREPRINT 

## **References** 

- Hongwei Liu, Zilong Zheng, Yuxuan Qiao, Haodong Duan, Zhiwei Fei, Fengzhe Zhou, Wenwei Zhang, Songyang Zhang, Dahua Lin, and Kai Chen. Mathbench: Evaluating the theory and application proficiency of llms with a hierarchical mathematics benchmark. _arXiv preprint arXiv:2405.12209_ , 2024. 

- Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. In _Proceedings of NAACL-HLT_ , pages 2357–2367, 2019. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In _35th Conference on Neural Information Processing Systems (NeurIPS 2021) Track on Datasets and Benchmarks_ . NeurIPS, 2021. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. URL `https://arxiv.org/pdf/2110.14168v1` . 

- Netmind.AI. Odyssey-math. `https://github.com/protagolabs/odyssey-math/tree/main` , 2024. Accessed: April 22, 2024. 

- Daman Arora, Himanshu Singh, et al. Have llms advanced enough? a challenging problem solving benchmark for large language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 7527–7543, 2023. 

- Simon Frieder, Luca Pinchetti, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Petersen, and Julius Berner. Mathematical capabilities of chatgpt. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Tomohiro Sawada, Daniel Paleka, Alexander Havrilla, Pranav Tadepalli, Paula Vidas, Alexander Kranias, John J Nay, Kshitij Gupta, and Aran Komatsuzaki. Arb: Advanced reasoning benchmark for large language models. _arXiv preprint arXiv:2307.13692_ , 2023. 

- Dan Hendrycks and Alexandr Wang. Submit your toughest questions for humanity’s last exam, 2024. URL `https: //www.safe.ai/blog/humanitys-last-exam` . Accessed: 2024-10-01. 

- OpenAI. Introducing openai o1-preview, 2024. URL `https://openai.com/index/ introducing-openai-o1-preview/` . Accessed: 2024-10-01. 

- Carl M Bender and Steven A Orszag. _Advanced mathematical methods for scientists and engineers I: Asymptotic methods and perturbation theory_ . Springer Science & Business Media, 2013. 

- Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondˇrej Certík, Sergey B. Kirpichev, Matthew Rocklin, AMiT[ˇ] Kumar, Sergiu Ivanov, Jason K. Moore, Sartaj Singh, Thilina Rathnayake, Sean Vig, Brian E. Granger, Richard P. Muller, Francesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik Johansson, Fabian Pedregosa, Matthew J. Curry, Andy R. Terrel, Štˇepán Rouˇcka, Ashutosh Saboo, Isuru Fernando, Sumith Kulal, Robert Cimrman, and Anthony Scopatz. Sympy: symbolic computing in python. _PeerJ Computer Science_ , 3:e103, January 2017. ISSN 2376-5992. doi: 10.7717/peerj-cs.103. URL `https://doi.org/10.7717/peerj-cs.103` . 

- Pauli Virtanen, Ralf Gommers, Travis E Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, et al. Scipy 1.0: fundamental algorithms for scientific computing in python. _Nature methods_ , 17(3):261–272, 2020. 

- John H Evans. Dimensional analysis and the buckingham pi theorem. _American Journal of Physics_ , 40(12):1815–1822, 1972. 

- Ian Stewart. _Galois Theory_ . CRC Press, Taylor & Francis Group, Boca Raton, FL, 4th edition, 2015. ISBN 978-1-4822-4583-7. Version Date: 20150112. 

- Veronika Hackl, Alexandra Elena Müller, Michael Granitzer, and Maximilian Sailer. Is gpt-4 a reliable rater? evaluating consistency in gpt-4’s text ratings. _Frontiers in Education_ , 8, December 2023. ISSN 2504-284X. doi: 10.3389/feduc.2023.1272229. URL `http://dx.doi.org/10.3389/feduc.2023.1272229` . 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. 

- Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018. 

- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. URL `https://api.semanticscholar.org/CorpusID:160025533` . 

11 

A PREPRINT 

- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

OpenAI. Openai o1-mini, 2024a. URL `https://openai.com/index/` 

`openai-o1-mini-advancing-cost-efficient-reasoning/` . Accessed: September 30, 2024. 

Meta AI. Meta llama 3, 2024. URL `https://ai.meta.com/blog/meta-llama-3/` . Accessed: 2024-06-03. 

Meta. Code llama: Ai for coding, 2023. URL `https://about.fb.com/news/2023/08/ code-llama-ai-for-coding/` . Accessed: 2024-06-03. 

OpenAI. Reasoning models, 2024b. URL `https://platform.openai.com/docs/guides/reasoning/ how-reasoning-works` . Accessed: Oct. 1, 2024. 

OpenAI. Simple evals. `https://github.com/openai/simple-evals?tab=readme-ov-file# user-content-fn-1-43aa11412dfb93b343474c8d56f8882f` , 2024c. Accessed: 2024-06-03. 

12 

A PREPRINT 

## **A Appendix** 

## **A.1 Semi-automated solution verification** 

Figure 5 provides a visual comparison of the numerical and approximate solutions for the Laplace Integral example in Appendix A.2.6 over a large domain. This allows for semi-automated human verification that analytical solutions correspond well with numerical ground truths, a method used to verify the problems in **HARDMATH-MINI** . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0013-04.png)


Figure 5: Visual comparison of numerical and approximate analytical solutions to a sample Laplace integral problem for solution verification. 

## **A.2 Implementation and method details for data generation** 

The following subsections detail the process used to generate the problems and solutions for each problem type. 

## **A.2.1 Nondimensionalization of polynomials** 

The first nondimensionalization sub-type is generalized by varying the integer values for the degrees _n_ 1 and _n_ 2 within the range 0 _< n_ 2 _< n_ 1 _<_ 10, while keeping _a_ 1, _a_ 2, _a_ 3 _>_ 0 symbolic. Solutions to these problems express the dimensionless parameter _ϵ_ in terms of these three coefficients. 

Sample Symbolic Nondimensionalization Problem and Full Solution **Problem:** Nondimensionalize the polynomial 

_a_ 1 _x_[10] + _a_ 2 _x_[9] + _a_ 3 into one of the form _ϵy_[[10]] + _y_[[9]] + 1 _._ Express _ϵ_ as a function of _a_ 1,, _a_ 2, and **Solution:** We begin with the substitution _a_ 3 _x_ = _y_[9] � _a_ 2 This gives the expression _a_ 3 9 _a_ 1 _y_[10] + _a_ 3 _y_[9] + _a_ 3 � _a_ 2 �[10] with coefficients in terms of _a_ 1,, _a_ 2, and, and _a_ 3:: _a_ 1 _y_[10][�] _aa_ 32 �[10] 9 + _y_[9] + 1 _. a_ 3 

into one of the form _ϵy_[[10]] + _y_[[9]] + 1 _._ Express _ϵ_ as a function of _a_ 1,, _a_ 2, and _a_ 3 _._ 

Divide by the coefficient remaining in front of the constant, leaving us with the nondimensionalized polynomial with coefficients in terms of _a_ 1,, _a_ 2, and, and _a_ 3:: 

13 

A PREPRINT 

By inspection, we can see that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0014-02.png)


**----- Start of picture text -----**<br>
a 1 � aa 32 � [10] 9<br>ϵ  = .<br>a 3<br>**----- End of picture text -----**<br>


The second subtype implements integer numerical values for the coefficients _a_ 1, _a_ 2, _a_ 3 that are are randomly chosen from [ _−_ 10 _,_ 10] _._ 

Sample Numeric Nondimensionalization Problem and Full Solution 

**Problem:** Nondimensionalize the polynomial 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0014-06.png)


into a polynomial of the form _ϵy_[7] _± y_[2] _±_ 1. Solve for _ϵ_ . 

**Solution:** For now, we ignore the numeric values of the coefficients and instead call them _a_ 1 _, a_ 2 _, a_ 3. Our polynomial is then: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0014-09.png)


Use the substitution 

which gives the expression 

Divide all terms by the coefficient remaining in front of the constant term, giving us the nondimensionalized polynomial with coefficients in terms of _a_ 1 _, a_ 2 _, a_ 3: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0014-13.png)


Substituting in the known numeric values for _a_ 1 _, a_ 2 _, a_ 3 (using their absolute values as we have already accounted for sign), we get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0014-15.png)


From inspection of this nondimensionalized equation, we can now identify _ϵ_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0014-17.png)


## **A.2.2 Polynomial root-finding** 

As with the nondimensionalization problems, degrees in the polynomial are randomly generated with maximum order ten and 0 _< n_ 2 _< n_ 1. See a full problem and solution below. 

Sample Polynomial Root-finding Problem and Full Solution 

**Problem:** Consider the polynomial 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0014-22.png)


Find first order approximations for all roots of the polynomials in the limit of small positive _ϵ_ and large positive _ϵ_ . 

**Solution:** We begin by equating the polynomial to zero to solve for the roots: _P_ ( _x_ ) = 0 _._ This problem can be rewritten in the form _A_ + _B_ + _C_ = 0, where: _A_ = _ϵx_[6] ; _B_ = _−x_[5] ; _C_ = 1 _._ 

14 

A PREPRINT 

This problem has no analytical solutions, so we find approximate solutions to the roots by considering the three possible dominant balances. For each dominant balance, we find the roots of the resulting equation and evaluate whether each balance is self-consistent for small or large positive _ϵ_ . 

We start with the balance _A_ + _B_ = 0, assuming that _|C|_ is negligible when compared to _|A|_ and _|B|_ . Solving this for _x_ in terms of _ϵ_ then gives us 1 non-zero root: 

_ϵx_[6] _− x_[5] = 0 1 = _⇒ x_ = _._ � _ϵ_ � 

To verify that these roots are consistent with the assumption that _|A|, |B| ≫|C|,_ we substitute these found roots back into the terms _A_ , _B_ , and _C_ and compare their magnitudes. Using this method, we find that it is true that these roots are valid for small _ϵ_ , while validity for large _ϵ_ is false. Therefore, these roots are valid in the limit of small positive _ϵ_ only. 

Next we examine the balance _B_ + _C_ = 0, assuming that _|A|_ is negligible when compared to _|B|_ and _|C|_ . Solving this for _x_ in terms of _ϵ_ gives us 5 non-zero roots: 

1 _− x_[5] = 0 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0015-07.png)


To verify that these roots are consistent with the assumption that _|B|, |C| ≫|A|,_ we substitute these found roots back into _A_ , _B_ , and _C_ and compare their magnitudes. Using this method, we find that it is true that these roots are valid for small _ϵ_ , while validity for large _ϵ_ is false. Therefore, these roots are valid in the limit of small positive _ϵ_ only. 

Finally, we examine the balance _A_ + _C_ = 0, assuming that _|B|_ is negligible when compared to _|A|_ and _|C|_ . Solving this for _x_ in terms of _ϵ_ gives us 6 non-zero roots: 

_ϵx_[6] + 1 = 0 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0015-11.png)


To verify that these roots are consistent with the assumption that _|A|, |C| ≫|B|,_ we substitute these found roots back into _A_ , _B_ , and _C_ and compare their magnitudes. Using this method, we find that it is false that these roots are valid for small _ϵ_ , while validity for large _ϵ_ is true. 

Therefore, these roots are valid in the limit of large positive _ϵ_ only. 

By the Fundamental Theorem of Algebra, a polynomial of degree 6.0 has exactly 6.0 roots.We have found 6.0 roots that are valid in the limit of small positive _ϵ_ and 6.0 roots valid in the limit of large positive _ϵ_ . Our method therefore provides a complete solution to the problem, finding the correct number of roots in each _ϵ_ regime. The roots of _P_ ( _x_ ) for large positive _ϵ_ are 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0015-15.png)


15 

A PREPRINT 

and the roots of _P_ ( _x_ ) for small positive _ϵ_ are 

|1<br>_ϵ ,_ <br>_−_|1_, −_1<br>4 +<br>_√_<br>5<br>4 _−i_<br>�<br>2<br>~~_√_~~<br>5<br>4<br>_√_<br>5<br>4 _−_1<br>4 _−i_<br>�<br>10_−_2<br>~~_√_~~<br>5<br>4|+ 10<br>_, −_1<br>4 +<br>_√_<br>5<br>4 +<br>~~�~~<br>_−_10_−_2<br>4<br>_, −_<br>_√_<br>5<br>4 _−_1<br>4 + _i_<br>�<br>10_−_2<br>~~_√_~~<br>5<br>4|~~_√_~~|5|_,_|
|---|---|---|---|---|---|



## **A.2.3 Polynomial root correction terms** 

The true roots _x[∗]_ of a polynomial are given by _x[∗]_ ( _ϵ_ ) = _x_ ( _ϵ_ ) + _δ,_ where _x_ is our existing approximation to the root as found in Appendix A.3 and _δ_ is the error term. This requires us to solve 

_ϵ_ ( _x_ + _δ_ ) _[n]_[1] _±_ ( _x_ + _δ_ ) _[n]_[2] _±_ 1 = 0 

for _δ_ by equating coefficients of _ϵ_ terms of the same order, as detailed in the worked solution below. 

Sample Numeric Nondimensionalization Problem and Full Solution 

**Problem:** Consider the polynomial 

_P_ ( _x_ ) = _ϵx_[3] _− x_ + 1 _._ 

Find approximate expressions for all roots of the polynomial in the limit of small positive _ϵ_ and large positive _ϵ_ . Use a series expansion to calculate improved formulae for these roots to order 1 i.e. calculate _O_ (1) corrections for each root. 

**Solution:** Note: The root calculation in this problem follow the same method as those demonstrated in the A.3, so they has been omitted here. We include only correction term calculations for the sake of brevity. 

We now need to calculate correction terms for these roots to give us better approximations. We consider the ansatz that the root is given by _x_ + _δ_ , where the correction term _δ_ is the sum of higher order terms of _ϵ_ that we initially neglected in our approximation _x_ . By definition, _δ < x._ We plug this ansatz into the polynomial and perform a series expansion in _δ_ . We keep terms only up to _O_ (1) in _δ_ . Then, we set the expression equal to 0 and solve for _δ_ . 

Regime 1: valid for small _ϵ_ 

Root 1: _−_ 1 � _ϵ_ 1 _x_ + _δ_ = _−_ ~~�~~ _ϵ_[+] _[ δ]_ Substitute this into _P_ ( _x_ ) for _x_ and equate to 0: 3 1 1 _−δ_ + _ϵ_ � _δ −_ � _ϵ_ � + � _ϵ_[+ 1 = 0] _[.]_ We then expand this expression to get 1 1 2 1 _δ_[3] _ϵ −_ 3 _δ_[2] _ϵ_ � _ϵ_[+ 2] _[δ][ −][ϵ]_ � _ϵ_ �[3] + � _ϵ_[+ 1 = 0] and represent it as a series of _O_ (1) in _δ_ , discarding higher order _δ_ terms 1 2 1 2 _δ − ϵ_ � _ϵ_ �[3] + ~~�~~ _ϵ_[+ 1] _[ ≈]_[0] _[.]_ We can then solve the expression for the correction _δ_ to _O_ (1), and get 1 2 1 _δ ≈[ϵ]_ � _ϵ_ �[3] _−_ � _ϵ −_[1] 2 2 2 _[.]_ 

16 

A PREPRINT 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0017-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0017-02.png)


17 

A PREPRINT 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0018-01.png)


**----- Start of picture text -----**<br>
and represent it as a series of  O (1) in  δ , discarding higher order  δ  terms<br>3<br>δ �3 ϵ � − [1] ϵ � [2] − 1� − �3 − [1] ϵ [≈] [0] [.]<br>We can then solve the expression for the correction  δ  to  O (1), and get<br>[�] 3 − [1]<br>ϵ<br>δ ≈ .<br>3 ϵ � − [1] ϵ � 3 [2] − 1<br>[√] 3 − [1] ϵ [(] [−] [1] [−] √ 3 i )<br>Root 2:<br>2<br>[�] 3 − [1] ϵ � − 1  − √ 3 i �<br>x  +  δ = +  δ<br>2<br>Substitute this into  P ( x ) for  x  and equate to 0:<br>3<br>[�] 3 − [1] ϵ � − 1  − √ 3 i � [�] 3 − [1] ϵ � − 1  − √ 3 i �<br>−δ  +  ϵ  δ  +  − + 1 = 0 .<br> 2  2<br>We then expand this expression to get<br>δ [3] ϵ − 3 δ [2] ϵ [�][3] − [1] ϵ − 3 √ 3 iδ [2] ϵ [�][3] − [1] ϵ − [3] [δϵ] � − [1] ϵ � 3 [2]<br>2 2 2<br>+ [3] √ 3 iδϵ � − [1] ϵ � 3 [2] − δ  + [�] 3 − [1] ϵ + √ 3 i [�][3] − [1] ϵ = 0<br>2 2 2<br>and represent it as a series of  O (1) in  δ , discarding higher order  δ  terms<br>δ  − [3] [ϵ] � − [1] ϵ � 3 [2] + [3] √ 3 iϵ � − [1] ϵ � 3 [2] − 1 + [�] 3 − [1] ϵ + √ 3 i [�][3] − [1] ϵ ≈ 0 .<br> 2 2  2 2<br>We can then solve the expression for the correction  δ  to  O (1), and get<br>[�] 3 − [1] ϵ �1 + √ 3 i �<br>δ ≈ .<br>3 ϵ � − [1] ϵ � 3 [2] − 3 √ 3 iϵ � − [1] ϵ � 3 [2] + 2<br>[√] 3 − [1] ϵ [(] [−] [1+] √ 3 i )<br>Root 3:<br>2<br>[�] 3 − [1] ϵ � − 1 + √ 3 i �<br>x  +  δ = +  δ<br>2<br>Substitute this into  P ( x ) for  x  and equate to 0:<br>3<br>[�] 3 − [1] ϵ � − 1 + √ 3 i � [�] 3 − [1] ϵ � − 1 + √ 3 i �<br>−δ  +  ϵ  δ  +  − + 1 = 0 .<br> 2  2<br>We then expand this expression to get<br>δ [3] ϵ − 3 δ [2] ϵ [�][3] − [1] ϵ + 3 √ 3 iδ [2] ϵ [�][3] − [1] ϵ − [3] [δϵ] � − [1] ϵ � 3 [2]<br>2 2 2<br>− [3] √ 3 iδϵ � − [1] ϵ � 3 [2] − δ  + [�] 3 − [1] ϵ − √ 3 i [�][3] − [1] ϵ = 0<br>2 2 2<br>**----- End of picture text -----**<br>


18 

A PREPRINT 

and represent it as a series of _O_ (1) in _δ_ , discarding higher order _δ_ terms 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0019-02.png)


We can then solve the expression for the correction _δ_ to _O_ (1), and get 

## **A.2.4 ODEs** 

We generate third-order ordinary differential equations of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0019-06.png)


where _f_ 1( _x_ ) _, f_ 2( _x_ ) _, f_ 3( _x_ ) _, f_ 4( _x_ ) are rational functions with integer coefficients. The initial conditions are randomly selected integers from [0 _,_ 3]. The dataset excludes problems with a function of _x_ as a dominant term because of the difficulty of deriving power law expressions in these cases. 

Approximate solutions at small _x_ can be derived using a Taylor series expansion (up to the third order) around _x_ = 0. Solving ODEs in the large _x_ regime involves determining the two largest terms, assuming a divergence at some large _x[∗]_ , and solving the dominant balance between these terms to create a power law approximation of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0019-09.png)


ODE Problem and Solution 

Problem: Consider the following third-order ordinary differential equation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0019-12.png)


Find analytical expressions that approximate the solution of y(x) at small and large _x_ . 

## **Solution:** 

The dominant balance in the large x regime is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0019-16.png)


We recognize that the solution of this ODE will diverge at finite _x_ and that divergences typically follow a power law of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0019-18.png)


where _x[∗]_ is the divergence point. The divergence point can be determined by estimated by examining the numerical solution generated by code. 

Plugging in the dominant terms we found previously yields the following equation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0019-21.png)


After substituting the derivatives, the equation is reorganized to collect terms with respect to ( _x − x[∗]_ ). This leads to an equation where the coefficients and powers of ( _x − x[∗]_ ) are equated on both sides. Simplifying the 

19 

A PREPRINT 

equation gives us two separate equations, one for the coefficients and another for the powers of ( _x − x[∗]_ ). There is now a system of equations, where the coefficients’ equation is 

_αp_ ( _p −_ 2) ( _p −_ 1) = _α_[2] _p_[2] 

and the powers’ equation is: 

_p −_ 3 = 2 _p −_ 2 _._ 

Solving this system of equations provides the values of _α_ and _p_ . A valid solution is identified if _α_ and _p_ are both nonzero. Here, the solution for _α_ and _p_ is found to be: 

_α_ = _−_ 6 _, p_ = _−_ 1 

With these values, the analytical approximation for the solution at large _x_ (near the divergence point) is given by 

_y_ = _−_ 6( _x −_ 11 _._ 45) _[−]_[1] _._ 

The approximate solution at small _x_ can also be solved used dominant balance, but one can take advantage of the initial conditions and form a Taylor series instead around _x_ = 0, which is given by 

_y_ ( _x_ ) _≈ y_ (0) + _y[′]_ (0) _x_ + _[y][′′]_[(][0][)] _x_[2] + _[y][′′′]_[(][0][)] _x_[3] _._ 2! 3! 

Plugging in the initial conditions, we get the following expression at small _x_ : 

_y_ ( _x_ ) = 1 _−_[13] 180 _[x]_[3] 

Thus, with rounding for clarity, the solution is given by 

_y_ ( _x_ ) = 1 _−_[13] _[y]_[=] _[ −]_[6(] _[x][ −]_[11] _[.]_[45)] _[−]_[1] _[.]_ 180 _[x]_[3] _[,]_ 

## **A.2.5 Integrals** 

The polynomial _P_ ( _x_ ) is randomly generated to consist of up to ten terms, where each term is a power function of _x_ with an integer power randomly sampled from 1 and 20 and an integer coefficient sampled from 1 to 10. The integration bound _a ∈_ [0 _,_ 100] is also randomly selected. This form ensures that the integral does not oscillate. 

The height is approximated as the maximum value of the integrand, which is[1] _ϵ_[, and the width can be estimated as the] distance over which the integrand decreases from its maximum value by a factor of 2, which implies that the width _x_ obeys the equation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0020-18.png)


In the regime of small _ϵ_ , the term with the smallest degree and _ϵ_ are the dominant terms, and in the regime of intermediate _ϵ_ , the term with the largest degree and _ϵ_ are dominant. There exists one more solution regime when the width of the integral exceeds the limits of integration, or when _ϵ_ is "very large." In this case, the integral is approximated by _L/ϵ_ , where _L_ is the integration range. 

See a full sample of a traditional integral problem in Box 1. 

## **A.2.6 Laplace integrals** 

_b_ Laplace integrals of the form _I_ ( _x_ ) = � _a[g]_[(] _[t]_[)] _[e][±][xf]_[(] _[t]_[)] _[dt]_[ assume that] _[ f]_[(] _[t]_[)] _[ >]_[ 0][, is never a constant, and has an absolute] minimum at a point _t_ 0 either in the interior of or on the bounds of the interval [ _a, b_ ]. 

The set of possible Laplace integrals _I_ ( _x_ ) in our dataset are parameterized by four parameters: the bounds [ _a, b_ ], _g_ ( _t_ ), _f_ ( _t_ ), and the sign in front of _x_ . To generate the dataset, the bounds for each problem were randomly sampled from the [ _−_ 1 _, −_ 0 _._ 9 _, . . ._ 0 _._ 9 _,_ 1], and the sign was uniformly sampled from _{−_ 1 _,_ 1 _}_ . The functions _f_ ( _t_ ) and _g_ ( _t_ ) were generated by randomly selecting a linear combination of polynomials up to fifth order and basic trigonometric functions. 

Our solution uses SymPy under the hood to find the minima of _f_ ( _t_ ) (or the dual annealing algorithm if SymPy fails to return the minima). 

20 

A PREPRINT 

Sample Laplace Integral Problem and Solution Problem: Consider the integral 

0 _._ 3 _I_ ( _x_ ) = ( _−_ 1 _._ 6 _t_[2] _−_ 0 _._ 5 sin ( _t_ ) _−_ 1 _._ 9) _e_[+] _[x]_[(] _[−]_[2] _[.]_[5] _[t]_[4] _[−]_[0] _[.]_[8] _[t]_[3][+1] _[.]_[4] _[t]_[2][)] _dt_ (2) � _−_ 0 _._ 9 Develop an analytical formula for _I_ ( _x_ ) that is accurate as _x → ∞_ . 

## **Solution:** 

The integral is of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0021-05.png)


where _a_ = _−_ 0 _._ 9, _b_ = 0 _._ 3, _g_ ( _t_ ) = _−_ 1 _._ 6 _t_[2] _−_ 0 _._ 5 sin ( _t_ ) _−_ 1 _._ 9, and _f_ ( _t_ ) = _−_ 2 _._ 5 _t_[4] _−_ 0 _._ 8 _t_[3] + 1 _._ 4 _t_[2] . This means we can use Laplace’s method to develop an analytical approximation in the limit that _x →∞_ . In this limit, the integral will be dominated by the integrand near the maximum of _f_ ( _t_ ) within the bounds [ _−_ 0 _._ 9 _,_ 0 _._ 3]. So, to simplify the integral, we will expand the integrand around this maximum. In this case, we can find the maximum of _f_ ( _t_ ) = _−_ 2 _._ 5 _t_[4] _−_ 0 _._ 8 _t_[3] + 1 _._ 4 _t_[2] on the interval analytically. We begin by looking for critical point(s) _tcrit_ of _f_ ( _t_ ) by solving _f[′]_ ( _t_ ) = _−_ 10 _._ 0 _t_[3] _−_ 2 _._ 4 _t_[2] + 2 _._ 8 _t_ = 0 for _t_ . This gives us that _tcrit_ = [ _−_ 0 _._ 66 _,_ 0]. To find the maximum on this interval, we evaluate _f_ ( _t_ ) at the critical point(s) _tcrit_ and the bounds _−_ 0 _._ 9 and 0 _._ 3. We take the _t_ that gives the largest value. Here, this maximum _t_ 0 = [ _−_ 0 _._ 66]. Since the integral is dominated by the value of the integrand near -0.66, we Taylor expand the integrand around this point. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0021-07.png)


_∗ e_[+] _[x]_[(] _[f]_[(] _[−]_[0] _[.]_[66)+(] _[t]_[+0] _[.]_[66)] _[f][ ′]_[(] _[−]_[0] _[.]_[66)+][(] _[t]_[+][0] 2 _[.]_[66][)][2] _f[′′]_ ( _−_ 0 _._ 66)+ _..._ ) _dt_ (4) 

But _f[′]_ ( _−_ 0 _._ 66) = 0 by definition, so we can remove this term from the exponent. We can then approximate 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0021-10.png)


Solving the integral and evaluating, we find that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0021-12.png)


21 

A PREPRINT 

## **A.2.7 Word problem automatic generation setup** 

Table 3: Prompts for generating problem and solution contexts and plausibility verification. 

|**Task**|**Task instruction**|
|---|---|
|Question con-|Rewrite this`{question_type}`problem by embedding|
|textualization|it within a plausible real-world`{domain_seed}`problem|
||scenario, without changing the mathematical question at|
||the end.|
|Solution con-|We provide above a real-world`{domain_seed}`question,|
|textualization|and its mathematical solution. Generate a single intro-|
||ductory sentence before the solution that can connect the|
||solution to the real-world `{domain_seed}` context pro-|
||vided in the question.|
|Verifcation|We provide above a real-world`{domain_seed}`question.|
||Review and verify its plausibility within the context of|
||parameter ranges of this domain (e.g. energy value cannot|
||be negative). Identify any inconsistencies or areas need-|
||ing clarifcation to ensure the problem is realistic for this|
||domain. Return a single foat number as plausibility score|
||(0-1) in LaTeX boxed format`\boxed{}`at the end.|



## **A.2.8 Word problem automatic generation example** 

Original vs. **physics** context augmented problem ( _ODEs_ ) 

**Original Mathematical Problem:** Consider the following third-order ordinary differential equation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0022-07.png)


Find analytical expressions that approximate the solution of _y_ ( _x_ ) in the small _x_ and large _x_ regimes. 

**Real-World Context Problem:** In a study of the dynamics of a newly discovered type of fluid, researchers are investigating the behavior of a fluid flow through a porous medium. The flow is described by a third-order ordinary differential equation that models the velocity profile of the fluid, denoted by _y_ ( _x_ ), as it moves through the medium. The equation takes into account various factors such as the porosity of the medium, the viscosity of the fluid, and external forces acting on the fluid. 

The governing equation for the velocity profile is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0022-11.png)


The initial conditions at the entry point of the medium, _x_ = 0, are specified as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0022-13.png)


22 

A PREPRINT 

The researchers are interested in finding analytical expressions that approximate the solution of _y_ ( _x_ ) in the small _x_ and large _x_ regimes. These approximations will help in understanding the initial behavior of the fluid as it enters the medium and its asymptotic behavior as it travels further through the medium. 

## **A.2.9 Word problem automatic generation quality evaluation** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0023-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0023-04.png)


(a) Diversity of context subdomains within the broader (b) Plausibility scores distribution of automatically generphysics domain for auto-generated word problems. ated word problems. 

Figure 6: Diversity and quality assessment of automatically generated word problems. Only problems above a plausibility score threshold are included. 

23 

A PREPRINT 

## **A.3 Evaluation setup** 

## **A.3.1 Prompts for response generation** 

Table 4: Problem type specific hints by Question and Answer Type 

|**Question**|**Answer Type**|**Task instruction**|
|---|---|---|
|**Type**|||
|Nondim-|SymPy|Please answer the question requiring an answer in a|
|symbolic||SymPy convertible formula containing variables and math|
|||operation expressions and provide the fnal answer, e.g.,|
|||_x_3, _x_<br>_y_ inside a Latex boxed format`\boxed{}`.|
|Nondim-|Float (2)|Please answer the question requiring a foating-point num-|
|numerical||ber with two decimal places and provide the fnal value,|
|||e.g., 0.80, 3.12, inside a Latex box`\boxed{}`.|
|Polynomial|SymPy List|Please answer the question requiring a Python list contain-|
|Roots||ing SymPy convertible formulas of variable_ϵ_and math|
|||operation expressions and provide the fnal list, e.g., [_ϵ_3,|
|||1<br>_ϵ_ ] inside a Latex boxed format`\boxed{}`.|
|ODEs|SymPy List|Please answer the question requiring a Python list contain-|
|||ing SymPy convertible formula of_y_ =_f_(_x_)and provide|
|||the fnal list, e.g.,[_y_ = 1_−x_3_, y_ =_−_6_/_(_x −_5)], inside a|
|||Latex boxed format`\boxed{}`.|
|Integrals|SymPy|Please answer the question requiring an answer in a|
|||SymPy convertible formula containing formulas of vari-|
|||able _x_ and math operation expressions and provide|
|||the fnal answer, e.g., _x_3 inside a Latex boxed format|
|||`\boxed{}`.|



24 

A PREPRINT 

## **A.3.2 Prompts for grading** 

Table 5: LLM-based grading prompts by Question and Answer Type 

|**Question**|**Answer type**|**Task instruction**|
|---|---|---|
|**type**|||
|Polynomial|SymPy List|Please take this response `{response}` and this answer|
|Roots||key`{answer key}`and grade the response based on the|
|||following criteria: 1) Check both the small and large _ϵ_|
|||solutions. 2) For each solution, give full credit if it com-|
|||pletely matches the elements in the answer key; give par-|
|||tial credit proportional to the number of matching roots|
|||between the response and the answer key; give no credit|
|||if it is completely wrong. 3) For both partial and no credit|
|||briefy state the error reason. 4) Average the scores for the|
|||small and large epsilon solutions to obtain a fnal score|
|||between 0 and 1. 5) Give the fnal grading as a foat in|
|||Latex boxed format`\boxed{}`.|
|ODEs|SymPy List|Please take this response`{response}`and this solution|
|||`{answer key}`and grade the response based on the fol-|
|||lowing criteria: 1) Check both the small and large_ϵ_solu-|
|||tions. 2) For small regime solution, only give full credit|
|||if it matches the formula in the answer key exactly; give|
|||no credit if it is doesn’t match the form. For large regime|
|||solution, give full credit if it matches the formula in the|
|||answer key exactly; give partial credit if it doesn’t match|
|||but the numerical evaluation is not far from solution at this|
|||regime; give no credit if neither satisfes 3) Average the|
|||scores for the small and large epsilon solutions to obtain a|
|||fnal score between 0 and 1. 4) Give the fnal grading as a|
|||foat in Latex boxed format`\boxed{}`.|
|Integrals (tra-|SymPy List|Please take this response`{response}`and this solution|
|ditional)||`{answer key}`and grade the response based on the fol-|
|||lowing criteria: 1) Check both the small and large_ϵ_solu-|
|||tions. 2) For each solution, give full credit if it matches|
|||the formula in the answer key; give no credit if it is com-|
|||pletely wrong and briefy state the reason for the error. 3)|
|||Average the scores for the small and large epsilon solu-|
|||tions to obtain a fnal score between 0 and 1. 4) Give the|
|||fnal grading as a foat in Latex boxed format`\boxed{}`.|
|Integrals|SymPy|Please take this response`{response}`and this solution|
|(Laplace)||`{answer key}`and grade the response based on the fol-|
|||lowing criteria: 1) Check the large_x_fnal solution. 2) Give|
|||full credit if it matches the formula in the answer key; give|
|||half credit if the`{response}`get to the checkpoint where|
|||it correctly identifes_t_0 where_f_ attains its maximum and|
|||attempt performing Taylor’s expansion around it but the|
|||fnal answer is wrong; give no credit if it is completely|
|||wrong. 3) For both partial and no credit briefy state the|
|||error reason. 4) Give the fnal grading as a foat in Latex|
|||boxed format`\boxed{}`.|



25 

A PREPRINT 

## **A.3.3 GPT grading human verification** 

|**Model**|**Roots**|**ODEs**|**Integrals**|
|---|---|---|---|
|GPT3.5(0)|0|0|0|
|GPT3.5(1)|0|-0.09|-0.02|
|GPT3.5(5)|+0.02|+0.07|+0.02|
|GPT4(0)|0|-0.02|0|
|GPT4(1)|0|-0.04|-0.02|
|GPT4(5)|+0.07|-0.07|-0.15|
|o1-mini(0)|+0.04|+0.05|0|
|o1-mini(5)|+0.05|+0.05|0|
|Llama3-8b(0)|0|0|-0.02|
|Llama3-8b(5)|-0.07|-0.02|-0.02|
|Codellama3-14b(0)|0|-0.02|0|
|Codellama3-14b(5)|0|-0.02|0|



Table 6: Average adjusted points using human judgment from GPT-based grading. Rows with score adjustments of 0.1 or more are highlighted in pink. 

## **A.3.4 Model hyper-parameters** 

Table 7: Generating parameters for various LLMs. 

|**Model**|**Generation Setup**|
|---|---|
|||
|GPT-3.5<br>GPT-4<br>o1-mini<br>Llama3<br>CodeLlama|`model = gpt-3.5-turbo, temperature = 0, max_tokens = 4000`<br>`model = gpt-4-turbo, temperature = 0, max_tokens = 4000`<br>`model = o1-mini, temperature = 0, max_tokens = 4000`<br>`model = llama3:8b, temperature = 0`<br>`model = codellama:13b, temperature = 0`|



## **A.3.5 Computing resource** 

Evaluations of open-source models on **HARDMATH** are conducted on a high-performance compute cluster with a single Tesla V100 GPU (16GB vram). Evaluation on one problem type typically takes less than 1 hour. 

26 

A PREPRINT 

## **A.4 Extended experimental results** 

## **A.4.1 Extended evaluation results** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0027-03.png)


(a) Model accuracy increases with shot numbers. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0027-05.png)


(b) Model accuracy breakdown by problem sub-types for all models with 5-shot CoT prompting. 

Figure 7: Model evaluation accuracy breakdown by shot number and problem sub-types. (a) evaluation accuracy for all models increases with shot numbers for CoT prompting with o1-mini and GPT-4 showing the most obvious improvements; (b) evaluation accuracy breakdown for all models on all problem sub-types under the 5-shot CoT condition. 

## **A.4.2 Extended error mode analysis** 

Problem 136 Grading Response: _Roots_ (error mode: incorrect dominant balance setup) 

**Question:** Consider the polynomial _P_ ( _x_ ) = _ϵx_[6] + _x_[4] _−_ 1 _._ Find approximate expressions for all roots of the polynomials in the limit of small positive _ϵ_ and large positive _ϵ_ . Only a single term approximation to the root is required. 

**Grading for Large Positive** _**ϵ**_ : 

**Model Response:** For large _ϵ_ When _ϵ_ is large, the term _ϵx_[6] dominates the polynomial for most _x_ , except for potentially small _x_ where _x_[4] might still be relevant. However, focusing on the dominant term gives: _P_ ( _x_ ) _≈ ϵx_[6] _._ Setting this equal to zero for an approximation (acknowledging that we’re looking for non-trivial roots of the overall polynomial, but focusing on dominance to guide our approximation), suggests that we don’t get meaningful roots from this term alone since _x_[6] = 0 doesn’t provide roots that satisfy the original polynomial’s requirement for non-zero roots. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0027-13.png)


initial assumption, we check that _|A|, |C| >> |B|_ holds for this root by substituting our found root expression for _x_ back into _A_ , _B_ , and _C_ and comparing their magnitudes. Using this method, we find that validity for small _ϵ_ is False and validity for large _ϵ_ is True. 

27 

A PREPRINT 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0028-01.png)


Figure 8: Model performance scaling upon few-shot prompting (0, 1, 5, 10) for problem type _Nondim_ and _ODEs_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0028-03.png)


Figure 9: Percentage of correct, partial, and incorrect responses for GPT-3.5 and CodeLlama using CoT. 

**Grading** : The response wrongly sets up the dominant balance as _A_ = 0 and does not provide a concrete list of roots for large _ϵ_ . Therefore, score for large positive _ϵ_ is 0 _._ 0 

Problem 133 Grading Response: _Roots_ (error mode: drop imaginary roots) 

**Question:** Consider the polynomial _P_ ( _x_ ) = _ϵx_[3] + _x_[2] _−_ 1 _._ Find approximate expressions for all roots of the polynomials in the limit of small positive _ϵ_ and large positive _ϵ_ . Only a single term approximation to the root is required. 

**Grading for Large Positive** _**ϵ**_ : **Model Response:** To find the approximate expressions for all roots of the polynomial _P_ ( _x_ ) = _ϵx_[3] + _x_[2] _−_ 1 in the limits of small and large _ϵ_ , we will consider the two cases separately. For Large _ϵ_ When _ϵ_ is large, the term _ϵx_[3] dominates the polynomial. Thus, the polynomial can be approximated by ignoring the _x_[2] term, leading to _ϵx_[3] _−_ 1 _≈_ 0 . Solving for _x_ gives us an approximation for the root: _x ≈_ � 1 _ϵ_ � 3[1] . The other roots in this limit are complex and given the scope, we’ll focus on this real root approximation. 

28 

A PREPRINT 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0029-01.png)


assumption, we check that _|A|, |C| >> |B|_ holds for this root by substituting our found root expression for _x_ back into _A_ , _B_ , and _C_ and comparing their magnitudes. Using this method, we find that validity for small _ϵ_ is False and validity for large _ϵ_ is True. Therefore, these roots are valid in the limit of large positive _ϵ_ only. 

**Grading** : The response makes additional assumptions to drop the two other complex roots for large _ϵ_ regime without the question asking it to do so. Therefore, score for large positive _ϵ_ is 0 _._ 5 

## **A.4.3 Extended example GPT-4o grading responses** 

Grading Response: _ODEs_ (incorrect) 

To grade the response, we will compare the provided solutions in the response with the ground truth solutions in both the small _x_ and large _x_ regimes. 

## **Small** _x_ **Regime** 

Response: _y_ ( _x_ ) = 1 _−_ 5 _e[−][x]_ Ground Truth: _y_ ( _x_ ) = 1 _−_[5] _[x]_ 6[3] 

The response and the ground truth solution for small _x_ are significantly different. The ground truth solution is a polynomial in _x_ , while the response includes an exponential term. They do not match at all, so the solution for the small _x_ regime is completely wrong. 

## **Large** _x_ **Regime** 

Response: _y_ ( _x_ ) = "Assumed rapid growth form due to _y[′]_[4] term." _[√]_ 3 1 Ground Truth: _y_ = 30( _x −_ 2 _._ 68) 3 + (1 _._ 84) 

The response does not provide a specific analytical expression for the large _x_ regime, while the ground truth does provide a specific form. The response mentions rapid growth due to _y[′]_[4] , but without a clear analytical form, this response doesn’t match the ground truth solution. This indicates that the solution for the large _x_ regime is also completely wrong. 

Given the evaluation criteria, both solutions do not match or are incomplete compared to the ground truth. Therefore, each solution gets no credit. 

The final grading is 0 _._ 0 . 

Box 3: Example grading response for a _ODEs_ problem that gets zero point. 

## Grading Response: _Integrals_ (correct) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0029-17.png)


**----- Start of picture text -----**<br>
1 1<br>The model response approximates the integral for small  ϵ  as 6 [1] [/] [4] [ϵ][−] [3] [/] [4] and for large  ϵ  as 8 [1] [/] [18] [ϵ][−] [17] [/] [18] [.]<br>Now, let’s compare these results with the ground truth solutions:<br>1<br>For small  ϵ : - The ground truth solution is I ( ϵ ) = ϵ [0][0] [.][.] [64][75] . - The model’s result is 6 [1] [/] [4] [ϵ][−] [3] [/] [4] .<br>Both results for small  ϵ  are essentially the same, with:<br>1<br>[≈] [0] [.] [64]<br>6 [1] [/] [4]<br>**----- End of picture text -----**<br>


29 

A PREPRINT 

1 **For large** _ϵ_ **:** - The ground truth solution is _I_ ( _ϵ_ ) = _ϵ_[0][0] _[.][.]_[89][94] . - The model’s result is 8[1] _[/]_[18] _[ϵ][−]_[17] _[/]_[18] . Both results for large _ϵ_ are essentially the same, with: 1 _[≈]_[0] _[.]_[89] 8[1] _[/]_[18] 

Since the model’s response closely matches the ground truth solutions in both regimes (small and large _ϵ_ ) and provides essentially the same approximations, I would give full credit 1 _._ 0 

Box 4: Example grading response for a _Integrals_ sub-type simple integral problem that gets full points. 

Grading Response: _Integrals_ (partial) 

## **Answer Key** 

1. Setup and Formulation: it 1) correctly identifies the integral form; 2) uses Laplace’s method; 3) identifies critical points and uses Taylor expansion around the point of interest; 4) performs a change of variables to simplify the integral. 

2. Key Steps and Approximations: it 1) expands the integrand around the minimum of _f_ ( _t_ ); 2) approximates the integral by focusing on the dominant exponential term; 3) provides a final analytical approximation for _I_ ( _x_ ) as _x →∞_ . 

3. Final Result: 

_I_ ( _x_ ) _≈−_[0] _[.]_[004] _[e]_[0] _[.]_[89] _[x] x_ 

## **Model Answer** 

1. Setup and Formulation: it 1) correctly identifies the integral form; 2) recognizes that the exponential term will dominate for large _x_ ; 3) focuses on the region where the integrand is most significant (near _t_ = 0 _._ 7). 

2. Key Steps and Approximations: it 1) uses the method of steepest descents heuristically without detailed Taylor expansion; 2) approximates by evaluating the polynomial part at _t_ = 0 _._ 7 and considering the exponential decay; 3) recognizes the dominant behavior of the exponential term but does not perform a detailed variable change or expansion. 

3. Final Result: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0030-14.png)


## **Comparison** 

1. Similarities: 1) understanding of dominance: both the answer key and the student correctly identify that the behavior of the integral for large _x_ is dominated by the exponential term; 2) identifying significant points: both recognize that _t_ = 0 _._ 7 is the critical point where the integrand contributes significantly; 3) approximate behavior: Bbth solutions focus on approximating the behavior of the integral for large _x_ , emphasizing the exponential term’s dominance. 

2. Differences: 1) mathematical rigor: the answer key employs a more rigorous mathematical approach, including Taylor expansion and change of variables, providing a more precise and detailed solution; 2) heuristic approach: the student takes a more heuristic approach, focusing on the dominant exponential term without detailed mathematical expansions or variable changes; 3) accuracy of final expression: the answer key provides a more accurate final expression with specific constants derived from the detailed process, while the student’s result, though on the right track, has a different coefficient due to the heuristic method. I would give partial credit 0 _._ 5 

Box 5: Example grading response for a _Integrals_ sub-type Laplace integral problem highlighting the error of failing to develop Taylor’s expansion. 

30 

A PREPRINT 

## **A.5 Motivation for including nondimensionalization problems** 

GPT-4 response on solving a dimensionalized polynomial root finding problem 

**Problem:** Find approximate analytical expressions for the roots of the polynomial: _a_ 0 _x_[5] _− a_ 1 _x_ + _a_ 2 = 0 

at different regimes of the coefficients _a_ 0, _a_ 1, and _a_ 2. 

## **Model Solution:** 

Regime 1: _|a_ 0 _| ≫ |a_ 1 _|, |a_ 2 _|_ (Dominance of the _x_[5] -term) In this regime, the polynomial simplifies to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0031-07.png)


Regime 2: _|a_ 1 _| ≫ |a_ 0 _|, |a_ 2 _|_ (Dominance of the linear term) Here, the polynomial reduces to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0031-09.png)


**----- Start of picture text -----**<br>
−a 1 x ≈ 0<br>and the root is approximately:<br>x ≈ 0 .<br>Including the next dominant term, we balance  −a 1 x  with  a 2:<br>−a 1 x  +  a 2 ≈ 0 = ⇒ x ≈ [a] [2] .<br>a 1<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0031-10.png)


**----- Start of picture text -----**<br>
Regime 3: |a 2 | ≫ |a 0 |, |a 1 | (Dominance of the constant term) In this regime, the constant term dominates, and<br>the polynomial becomes inconsistent unless  a 2 = 0, which contradicts the assumption  |a 2 | ̸ = 0. Thus, there are<br>no real roots in this regime.<br>Regime 4: General Balance of Terms When all terms are of comparable magnitude, we consider approximate<br>solutions based on the magnitude of  x :<br>1. For large  x : The  x [5] -term dominates. Balancing  a 0 x [5] with  −a 1 x :<br>1 / 4<br>a 0 x [5] − a 1 x ≈ 0 = ⇒ x ≈ a 1 .<br>� a 0 �<br>2. For small  x : The linear term  −a 1 x  dominates over the other terms. Balancing  −a 1 x  with  a 2:<br>−a 1 x  +  a 2 = 0 = ⇒ x ≈ [a] [2] .<br>a 1<br>3. For intermediate  x : All terms contribute, and the roots must be found numerically or using perturba-<br>tive methods.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0031-11.png)


**----- Start of picture text -----**<br>
Reference Solution:<br>Nondimensionalization We begin with the quintic polynomial:<br>a 0 x [5] − a 1 x  +  a 2 = 0 .<br>To explore how the solution depends on the parameters  a 0,  a 1, and  a 2, let us write:<br>x  = [a] [2] y,<br>a 1<br>**----- End of picture text -----**<br>


31 

A PREPRINT 

and substitute this into the polynomial. This substitution yields: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0032-02.png)


Dividing through by _a_ 2, the equation becomes: 

Now define the parameter: 

which simplifies the equation to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0032-06.png)


We examine the roots of the equation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0032-08.png)


in two asymptotic limits: _ϵ →_ 0 and _ϵ →∞_ . The aim is to match the expressions for the roots in the intermediate regime. 

Consider the limit _ϵ →_ 0. The equation involves three terms: _ϵx_[5] , _−x_ , and 1. For the equation to be satisfied, these terms must sum to zero. In the limit _ϵ →_ 0, typically two of these terms dominate, forming a distinguished balance. 

To discover approximate formulas for the roots, we analyze each possible balance and determine consistency. Balancing _−x ≈_ 1 

The frst possibility is to balance the terms _−x_ and 1. This implies, to a first approximation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0032-13.png)


For this balance to be self-consistent, the neglected term _ϵx_[5] must satisfy: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0032-15.png)


Since _x ≈_ 1, this condition reduces to _ϵ ≪_ 1, which is satisfied as _ϵ →_ 0. Hence, this balance is consistent. Balancing _x_[5] _∼−_ 1 

Consider the balance _ϵx_[5] _∼−_ 1. This implies: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0032-18.png)


For this balance to be consistent, the neglected term _x ∼ ϵ[−]_[1] _[/]_[5] must be smaller than the terms we have kept. However, this is _not_ the case. In the limit _ϵ →_ 0, the neglected term grows unbounded, while the retained terms remain order unity. Therefore, this balance is not consistent. Balancing _x_[5] _∼ x_ 

The only other possible balance is _ϵx_[5] _∼ x_ . This balance leads to four roots: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0032-21.png)


In this case, the size of the neglected term 1 is much smaller than the size of the retained terms, which are of order _O_ ( _ϵ[−]_[1] _[/]_[4] ). Thus, this balance is self-consistent. Final Solution for the Quintic Roots The quintic equation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0032-23.png)


has the following approximate roots depending on the coefficient regime, summarized below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics_images/iclr_2025_HARDMath_A_Benchmark_Dataset_for_Challenging_Problems_in_Applied_Mathematics.pdf-0032-25.png)


32 

