Preprint 

# OPENWEBMATH: AN OPEN DATASET OF HIGH-QUALITY MATHEMATICAL WEB TEXT 

> _♣_ **Keiran Paster** _∗_ **,** _†_ **Marco Dos Santos** _∗_ **,** _◦_ **Zhangir Azerbayev,** _♣_ **Jimmy Ba** 

> _♣_ University of Toronto; Vector Institute for Artificial Intelligence 

> _†_ University of Cambridge, _◦_ Princeton University 

keirp@cs.toronto.edu, mjad3@cam.ac.uk 

## ABSTRACT 

There is growing evidence that pretraining on high quality, carefully thought-out tokens such as code or mathematics plays an important role in improving the reasoning abilities of large language models. For example, Minerva, a PaLM model finetuned on billions of tokens of mathematical documents from arXiv and the web, reported dramatically improved performance on problems that require quantitative reasoning. However, because all known publicly released web datasets employ preprocessing that does not faithfully preserve mathematical notation, the benefits of large scale training on quantitive web documents are unavailable to the research community. We introduce OpenWebMath, an open dataset inspired by these works containing 14.7B tokens of mathematical webpages from Common Crawl. We describe in detail our method for extracting text and L[A] TEX content and removing boilerplate from HTML documents, as well as our methods for quality filtering and deduplication. Additionally, we run small-scale experiments by training 1.4B parameter language models on OpenWebMath, showing that models trained on 14.7B tokens of our dataset surpass the performance of models trained on over 20x the amount of general language data. We hope that our dataset, openly released on the Hugging Face Hub, will help spur advances in the reasoning abilities of large language models. 

## 1 INTRODUCTION 

Advances in large language models have opened up new opportunities in numerous fields, providing a transformative shift in our approach to a wide range of complex problems (Brown et al., 2020; Raffel et al., 2020). Among these problems, mathematical reasoning has drawn the attention of several researchers in recent years, becoming both a common benchmark to judge the performance of large language models and inspiring new approaches to improve their reasoning capabilities in the hope that they will one day be able to solve complex mathematical problems. One of the biggest advancements in mathematical reasoning in recent years has been the Minerva model (Lewkowycz et al., 2022), which achieved state-of-the-art results on quantitative reasoning benchmarks such as MATH (Hendrycks et al., 2021). Minerva was trained by finetuning PaLM (Chowdhery et al., 2022) on a curated dataset consisting of billions of tokens of high quality technical content sourced from both scientific papers and the web. 

Minerva and the datasets used for its training were not released publicly and the current capabilities of open-source models (e.g., Touvron et al. (2023b;c;a); Geng & Liu (2023); Biderman et al. (2023)) in quantitative reasoning lags behind. We believe that there are important research directions that can only be enabled through open-access to such models and datasets, such as work on memorization and generalization, reinforcement learning, the development of new reasoning benchmarks, and advancement in the reasoning capabilities of language models. 

In our work, we produce an open alternative to the Math Web Pages dataset used to train Minerva (Lewkowycz et al., 2022). We extract documents from Common Crawl[1] , applying our pipeline to 

> _∗_ Keiran and Marco created the dataset and Zhangir led model training and evaluation. 

> 1https://commoncrawl.org/ 

1 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_OpenWebMath_An_Open_Dataset_of_High-Quality_Mathematical_Web_Text_images/iclr_2024_OpenWebMath_An_Open_Dataset_of_High-Quality_Mathematical_Web_Text.pdf-0002-01.png)


**----- Start of picture text -----**<br>
1B 336M 66M 59M 7.8M<br>237B 6.3M 14.7B<br>HTML pages documents tokens<br>Prefilter<br>Language ID MathScore Filter Perplexity Filter Deduplication Manual Filter<br>**----- End of picture text -----**<br>


Figure 1: The pipeline for constructing OpenWebMath involves aggressive filtering so that the final dataset only contains high quality, English, and mathematical content. 

extract text while preserving mathematical content in the form of L[A] TEX equations. We then filter the documents, ensuring that only high-quality English mathematical documents are kept. Finally, we deduplicate the dataset, resulting in 14.7B tokens of high-quality mathematical content suitable for both pretraining and finetuning large language models. The key contributions of this work are as follows: 

- We publically release OpenWebMath, a dataset of 14.7B tokens of high-quality mathematical web text. Our dataset can be found at https://huggingface.co/datasets/open-webmath/open-web-math on the Hugging Face Hub. 

- We extensively document our pipeline, sharing our findings with the NLP community. We open-source the code needed to reproduce our results. 

- We analyze the quality of OpenWebMath. First, we analyze the contents of our dataset, providing statistics on the types of webpages, subjects, and top domains. Then, we train several language models on our dataset to show that per-token, it is more effective than existing mathematical pretraining datasets, and is most effective when combined with other datasets. 

## 2 RELATED WORK 

## 2.1 MATHEMATICS DATASETS AND BENCHMARKS 

**Mathematics datasets** Over the past couple of years, several datasets of mathematics have been introduced. AMPS, a dataset of informal mathematics, was introduced alongside the MATH dataset (Hendrycks et al., 2021). AMPS includes more than 100,000 Khan Academy problems with stepby-step solutions in LaTeX and over 5 million problems generated using Mathematica scripts. In total, AMPS contains 23GB of problems and solutions. Another notable example is NaturalProofs (Welleck et al., 2021), which encompasses 32,000 theorem statements and proofs, 14,000 definitions, and 2,000 other types of pages (e.g. axioms, corollaries) derived from ProofWiki, the Stacks project and data from mathematics textbooks. Proof-Pile (Azerbayev et al., 2023) is a dataset of mathematical text that contains more than 14.5GB of informal mathematics texts obtained from arXiv, Stack Exchange, ProofWiki, Wikipedia, openly licensed books, and the MATH dataset. There are also many proprietary datasets for mathematics. WebMath is a large-scale dataset mentioned by OpenAI researchers (Polu & Sutskever, 2020) that contains a 35B token mix of content from Github, arXiv, and Math StackExchange, adding up to 35GB of informal mathematics. MathMix is another OpenAI dataset used to finetune GPT-4 (Lightman et al., 2023) that contains 1B high quality mathematical tokens containing both natural and synthetic data. The proprietary web dataset used to train Minerva, called Math Web Pages (Lewkowycz et al., 2022), was compiled by collecting 17.5B tokens from web pages that contain L[A] TEX code. 

**Mathematics benchmarks** Several popular benchmarks have been used by researchers to assess the capabilities of language models on both formal and informal mathematics. The MATH dataset (Hendrycks et al., 2021) is comprised of 12,500 challenging competition problems in informal language. Each problem is also accompanied by a step-by-step informal proof. Answers are delimited by the _\_ boxed environment, allowing for easier answer verification. GSM8k (Cobbe et al., 2021) is another popular multi-step informal mathematics reasoning benchmark. It contains 8,500 grade 

2 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_OpenWebMath_An_Open_Dataset_of_High-Quality_Mathematical_Web_Text_images/iclr_2024_OpenWebMath_An_Open_Dataset_of_High-Quality_Mathematical_Web_Text.pdf-0003-01.png)


**----- Start of picture text -----**<br>
Other<br>Other<br>Blog<br>Economics<br>Forum 5% 6% Paper 12% 2%3% ChemistryStatistics<br>12% 4%<br>40%<br>Mathematics 50%<br>16% 12% Comp Sci<br>Reference<br>17%<br>21%<br>Physics<br>Educational<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_OpenWebMath_An_Open_Dataset_of_High-Quality_Mathematical_Web_Text_images/iclr_2024_OpenWebMath_An_Open_Dataset_of_High-Quality_Mathematical_Web_Text.pdf-0003-02.png)


**----- Start of picture text -----**<br>
Types of OpenWebMath documents Subjects of OpenWebMath documents<br>**----- End of picture text -----**<br>


Figure 2: **Left** : The documents in OpenWebMath are sourced from forum posts, educational content, reference pages, scientific papers, blogs, and more. Most content comes from Q&A forums where users discuss how to solve problems. **Right** : The majority of the content in OpenWebMath is related to mathematics, but a large part is related to other technical subjects like Physics, Computer Science, Statistics, and more. 

school math problems that are intended to be solvable by a bright middle school student. Lewkowycz et al. (2022) also introduce a benchmark based on OpenCourseWare. OCWCourses includes a set of 272 automatically-verifiable solutions at the undergraduate level, covering chemistry, information theory, differential equations, special relativity, and more. Lewkowycz et al. (2022) also evaluate on a subset of MMLU (Hendrycks et al., 2020) called MMLU-STEM, which focuses on science, technology, engineering, and mathematics. 

## 2.2 WEB DATA PROCESSING PIPELINES 

The pretraining of large language models requires large, diverse datasets. Data scraped from the web is one of the primary sources for such data. However, sources such as Common Crawl, which contains over 200 billion web pages, are known to have significant amounts of low-quality and duplicate content, requiring extensive filtering and deduplication to be suitable for training. Prior works such as C4 (Raffel et al., 2020), RefinedWeb (Penedo et al., 2023), CCNet (Wenzek et al., 2019), The Pile (Gao et al., 2020), and GPT-3 (Brown et al., 2020) introduce various pipelines for extracting quality data from Common Crawl for the purposes of language model training. These pipelines typically consist of three primary steps: text extraction, filtering, and deduplication. 

**Text extraction** Extracting plain text from HTML files is a critical step in the creation of Common Crawl-based datasets. The easiest way to extract text from Common Crawl documents is to use the WET corresponding to each webpage, which contains pre-extracted plain text of the webpage. CCNet and C4 both use Common Crawl’s WET files. However, the text extracted in WET files may contain too much boilerplate or miss out on important content such as L[A] TEX equations. It is also possible to extract text directly from the raw HTML found in Common Crawl WARC files. The Pile uses an open source library called jusText (Endr´edy & Nov´ak, 2013) to extract text from HTML while RefinedWeb uses a library called Trafilatura (Barbaresi, 2021). These text extraction approaches differ in terms of extraction speed, customization, and their precision and recall for removing boilerplate content. 

**Filtering** The first layer of filtering often involves language identification (Wenzek et al., 2019). Language filtering is used because certain other parts of the pipeline only work for specific languages, and is often done with simple linear classifiers such as from fastText (Joulin et al., 2016). Quality filtering can be done with a combination of perplexity, classifier, and rule-based methods. CCNet uses a 5-gram Kneser-Ney language model implemented in the KenLM library (Heafield, 2011) trained on the target domain. The documents in the dataset are then sorted and filtered by their perplexity under this model. Other datasets such as the one used to train GPT-3 (Brown et al., 2020) use a classifier-based approach. This involves training a classifier on known-high-quality 

3 

Preprint 

documents, such as those from Wikipedia, as positive examples and unfiltered documents from Common Crawl as negative examples. The classifier scores are used to filter low-quality documents from the dataset. Finally, rule-based approaches such as those used in C4 (Raffel et al., 2020) and MassiveWeb (Rae et al., 2021) involve removing pages with certain characters, too many or too few characters, too high a proportion of symbols, or those with an abnormal average word length. OpenMathWeb uses a mixture of these three approaches. 

**Deduplication** Given the periodic nature of Common Crawl snapshots and a general redundancy in web-sourced text, deduplication is an important processing step. Document-level neardeduplication (e.g., in (Brown et al., 2020; Penedo et al., 2023)) often employs MinHashLSH, an efficient algorithm for estimating the Jaccard similarity of documents. CCNet (Wenzek et al., 2019) uses paragraph-level deduplication, which can help to remove common boilerplate content found in WET text-extractions. 

## 3 BUILDING OPENWEBMATH 

## 3.1 OBJECTIVES 

Our aim with OpenWebMath is to build a dataset of as many mathematical documents sourced from the web as possible while preserving the formatting of mathematical content such as L[A] TEX equations as in Lewkowycz et al. (2022). For the purposes of this work, we define a mathematical document as a document containing either core mathematical contents such as theorems, definitions, proofs, questions and answers, formal mathematics, or interdisciplinary documents featuring mathematical formulas within fields like physics, chemistry, biology, economics, and finance. We source our documents from Common Crawl, which is a large open-access crawl of the web containing petabytes of raw HTML files. Due to the high variance in the quality of documents from Common Crawl, we additionally use several methods for filtering and boilerplate reduction. Throughout the creation of OpenWebMath, we iteratively refined these methods to ensure that we do not remove too many relevant documents, optimizing for high recall whenever possible. Since we expect that OpenWebMath will be used primarily as an additional source of pretraining data for large language models, we prefer having a small percentage of non-mathematical but high quality documents in the dataset rather than removing them and potentially losing relevant mathematical content. Finally, due to the limited number of mathematical data available on the web, we use significantly more manual inspection and tuning of our processing pipeline than other web-based datasets. We document our processing choices and pipeline in the section that follows. 

## 3.2 HIGH-LEVEL OVERVIEW OF THE PIPELINE 

As shown in Figure 1, the processing pipeline for OpenWebMath falls into five stages. First, we apply a prefilter to all HTML documents in Common Crawl to quickly judge whether they have mathematical content, skipping those that do not before doing the extensive processing needed to extract text and equations and remove boilerplate. Second, we extract the text, including mathematical content, from the HTML documents. Third, we apply language identification filters, perplexity-based quality filtering, and a mathematical content classifier filter. Fourth, we deduplicate the dataset using SimHash (Manku et al., 2007). Finally, we manually inspect the documents gathered in the previous steps and view documents from the most popular domains by document-count and character-count, removing domains that are not high quality. We describe each of these steps in detail in the following sections. 

## 3.3 PREFILTERING 

Since there are over 200B HTML documents in Common Crawl, applying our processing over each document would require a significant amount of compute. To improve the efficiency of the pipeline, we first apply a stack of pre-filters optimized for high recall to reduce the number of documents that need to be processed. Our first filters check for common mathematical strings as in Lewkowycz et al. (2022), such as the presence of tex classes, <math> tags, and the word “mathjax”. See Table 8 for a full list of terms. If none of these terms are present, we search for the presence of the top 100 most-popular L[A] TEX symbols in the text. This is done by first filtering for documents 

4 

Preprint 

|**Training Dataset**|**GSM8k**||||**MATH**||||
|---|---|---|---|---|---|---|---|---|
|||Prealgebra|Algebra|Intermediate<br>Algebra|Counting &<br>Probability|Number<br>Theory|Precalculus|Geometry|
|**The Pile**(14.7B tokens)|2.2032|1.9127|1.9751|1.8420|1.8193|1.9227|1.6847|1.9499|
|**ProofPile**(14.7B tokens)|2.2350|1.7370|1.7214|1.5739|1.6462|1.7291|1.4838|1.7229|
|**OpenWebMath**(14.7B tokens)|1.9075|1.6285|1.6503|1.5949|1.6002|1.6894|1.4542|1.5748|
|**Mixture**(14.7B tokens)|**1.8968**|**1.6055**|**1.6190**|**1.5301**|**1.5719**|**1.6607**|**1.4119**|**1.5599**|
|**The Pile**(300B tokens; Pythia 1.4B)|1.9430|1.7117|1.7560|1.6358|1.6359|1.7460|1.5191|1.7252|



Table 1: We trained 1.4B parameter models for 14.7B tokens on various datasets and measured their perplexity on different mathematics benchmarks. Both OpenWebMath and a 50/50 mixture of ProofPile Azerbayev et al. (2023) and OpenWebMath perform well - outperforming Pythia 1.4B (Biderman et al., 2023) trained on 300B tokens of The Pile (Gao et al., 2020). 

containing a backslash command using a simple regular expression and then searching specifically for these L[A] TEX symbols in the plain text from the HTML document. If none of these symbols are found, we run the plain text through our _MathScore_ classifier (see section 3.5.1) and keep documents that exceed a confidence threshold of 0.8. By tuning these filters and using hierarchical layers of progressively more accurate but more expensive filters, we were able to reduce the compute needed to process the dataset by several times while retaining a high recall of relevant documents. 

## 3.4 TEXT EXTRACTION 

In contrast with prior works that extract text from Common Crawl such as C4 (Collins et al., 2023), The Pile (Gao et al., 2020), and RefinedWeb (Penedo et al., 2023), we chose to make a mostly custom pipeline for extracting the main content from HTML documents. This is because we found that while other tools get decent performance on average over many documents on the internet, they do not work optimally on many of the most common sources of mathematical content on the web. We instead opted to build on top of Resiliparse (Bevendorff et al., 2018; 2021), a fast and efficient library built in Cython that includes performant tools for parsing HTML pages, processing their DOMs, and extracting the main content. As shown in Table 5 in the appendix, Resiliparse is significantly more efficient than alternative libraries such as jusText. Another notable part of our text extraction pipeline is that we randomize the parameters of the extraction to add diversity to the dataset. This includes randomizing whether we use a plain text or Markdown format for the documents and randomizing the amount of boilerplate terms required to trigger a line being removed. 

Our text extraction pipeline consists of four stages: L[A] TEX extraction, text extraction, DOM processing, and line processing. 

**L[A] TEX Extraction** Lewkowycz et al. (2022) employ a relatively simple L[A] TEX extraction pipeline that extracts equations from <script type="math/latex">, <script type="math/asciimath">, and <math> blocks with <annotation encoding="application/x-tex"> blocks within them and replaces these tags with the extracted equations. When we applied these filters to documents from Common Crawl, we noticed an extremely low number of these tags compared to what was reported. We suspect that this is due to a difference between the HTML files available within Google (Lewkowycz et al., 2022) and those available on Common Crawl. The majority of the L[A] TEX on the internet is written using MathJax, where developers write equations delimited by dollar signs or other delimiters in their HTML pages and then the included javascript code replaces these equations with properly 

|**Training Dataset**<br>**MATH Algebra-Easy**<br>**MATH Algebra-Easy**<br>**maj@16**<br>**LILA multiarith**|**Training Dataset**<br>**MATH Algebra-Easy**<br>**MATH Algebra-Easy**<br>**maj@16**<br>**LILA multiarith**|
|---|---|
|**The Pile**(14.7B tokens)<br>**ProofPile**(14.7B tokens)<br>**OpenWebMath**(14.7B tokens)<br>**Mixture**(14.7B tokens)|2.81%<br>3.93%<br>9.77%<br>2.81%<br>3.93%<br>8.04%<br>**5.62%**<br>9.55%<br>**16.67%**<br>5.06%<br>**10.11%**<br>13.22%|
|**The Pile**(300B tokens; Pythia 1.4B)|3.93%<br>5.62%<br>**21.80%**|



Table 2: Accuracy on Different Math Benchmarks. 

5 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_OpenWebMath_An_Open_Dataset_of_High-Quality_Mathematical_Web_Text_images/iclr_2024_OpenWebMath_An_Open_Dataset_of_High-Quality_Mathematical_Web_Text.pdf-0006-01.png)


**----- Start of picture text -----**<br>
This paper concerns the quantity  Suppose I have a smooth map  <math><br><img src="https://s0.wp.com/ [tex]f\colon \mathbb{R}^3   <semantics><br>latex.php?latex=%7BM%28x%29..."  \longrightarrow S^2[/tex]. If I   ...<br>alt="{M(x)}" />, defined as the  identify [tex]\mathbb{R}^3[/tex]     <annotation ...><br>length of the longest  with [tex]U_S = S^3 - \ {\displaystyle \mathrm {MA}<br>subsequence of the numbers from {(0,0,1)\}[/tex] via  ={\frac{f_{O}}{f_{E}}}}<br>stereographic projection </annotation><br> </semantics><br></math><br>Image Equations Delimited Math  Special Tags<br>**----- End of picture text -----**<br>


Figure 3: L[A] TEX formulas can be embedded in HTML documents in many ways, including in images, within arbitrary delimiters, and within special tags. Most common text-extraction pipelines do not extract L[A] TEX code properly. 

rendered L[A] TEX equations within the above script tags when the page is loaded. HTML documents on Common Crawl do not include the changes to the HTML that result from running javascript, requiring that we instead extract the L[A] TEX equations by finding delimiters ourselves. This is a significant challenge since we need to detect whether the page contains the required MathJax javascript code, which delimiters were chosen by the user to denote equations, and then match and extract the equations from the text on the page. See Appendix B for a more detailed discussion. 

In order to extract MathJax, we first determine whether the page is importing the MathJax javascript code by searching for the word MathJax on the page. If it is not found, we additionally search for common L[A] TEX symbols, and if they are found, we treat the page as though it is running MathJax. We use regular expressions to search for code that calls the configuration function for MathJax to extract the delimiters used for equations. We add these delimiters to an extensive list of default delimiters and treat any content between these delimiters as L[A] TEX equations. 

In addition to extracting equations from MathJax, we found several more ways that L[A] TEX is encoded on the internet. These methods were discovered by filtering small portions of Common Crawl for documents that contain _\_ frac, one of the most popular L[A] TEX commands, and making sure that our processing code supports all the different ways that math could be encoded. We found that L[A] TEX on the internet is encoded in the following ways: 

1. equation and align environments. 

2. The alttext of elements with special classes like tex. 

3. Images from domains like latex.codecogs.com often include equations encoded in the URL. 

4. Special wordpress plugins. 

5. <math> tags with <annotation encoding="application/x-tex"> blocks within them. 

6. <math> tags with MathML content. We use a style sheet to convert these equations into L[A] TEX. 

7. MathJax equations encoded in the text of the page. 

The relative frequencies of the different ways math is encoded can be found in Table 6 in the appendix. 

**DOM Processing** After extracting the L[A] TEX equations from the HTML, we do several processing steps on the DOM-tree of the HTML document. This includes removing invisible elements based on their styles, removing buttons and link clusters, annotating code, tables, and headers, and removing known problematic elements based on class or ID. 

**Text Extraction** We use the extract ~~p~~ lain ~~t~~ ext(main ~~c~~ ontent=True) method in Resiliparse (Bevendorff et al., 2018) to extract the main content text from the DOM following several preprocessing steps to get around common issues with their specific implementation that cause it to be overly sensitive when removing boilerplate. 

6 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_OpenWebMath_An_Open_Dataset_of_High-Quality_Mathematical_Web_Text_images/iclr_2024_OpenWebMath_An_Open_Dataset_of_High-Quality_Mathematical_Web_Text.pdf-0007-01.png)


**----- Start of picture text -----**<br>
Classifier Input<br>…As an explicit example, on  …As an explicit example, on<br>Tuesday, our answer for that  Tuesday, our answer for that<br>day will be $$1 \times 3+2  day will be. This problem was<br>\times 2+3 \times 1=10$$. This  adopted from a similar problem<br>problem was adopted from a  remove LaTeX equations given to me by a …<br>similar problem given to me by<br>a …<br>Text Corpus MathScore Classifier<br>predict if LaTeX<br>present based on text<br>Has LaTeX Commands: Yes(\times)<br>Classifier Label<br>has LaTeX Commands?<br>**----- End of picture text -----**<br>


Figure 4: The MathScore classifier used in filtering OpenWebMath is trained to predict whether a text has any of the most popular L[A] TEX commands based only on surrounding words. This lets us include documents on the web that do not include extractable L[A] TEX but still contain technical content. 

**Line Processing** After extracting the plain text on the page using Resiliparse, we apply our own processing to remove boilerplate lines based on an iteratively-refined set of common boilerplate phrases, remove empty headers, and escape dollar signs that are not part of L[A] TEX equations. 

## 3.5 FILTERING 

We apply filtering with the goal of removing non-English documents (since our filters pipeline is optimized for English), removing documents that are not mathematical, and removing low-quality documents that would be harmful to train a language model on. We apply the following filters in order: 

1. We use a FastText language identification model (Joulin et al., 2016) to remove documents that are not in English. 

2. We use our _MathScore_ classifier (see section 3.5.1) to get a probability that the document is mathematical. If our previous extraction step found L[A] TEX equations, we keep documents with a probability of over 0.17. If no L[A] TEX equations were found, we keep documents with a probability of over 0.8. 

3. We use a KenLM language model (Heafield, 2011) trained on ProofPile (Azerbayev et al., 2023) to get a perplexity score for each document. We remove documents with a perplexity score of more than 15,000. 

## 3.5.1 MATH SCORE 

During our filtering process, we train a model to predict the probability a document is mathematical, which we call _MathScore_ . We first gather a dataset of hundreds of thousands documents extracted from our pipeline from an early stage of the project, and label them depending on whether they contain one of the top-100 most common L[A] TEX commands. We then remove any L[A] TEX code from the documents and train a classifier to predict whether the documents contain one of these common L[A] TEX commands. The training process for _MathScore_ is depicted in Figure 4. Since we remove all L[A] TEX code from the features fed into the model, the model needs to learn the words and phrases most commonly associated with L[A] TEX content. We use FastText (Joulin et al., 2016) to train this model, and find based on manual inspection that content with a score of under 0.2 is very unlikely to contain useful mathematical content. 

## 3.6 DEDUPLICATION 

Due to the large amount of duplicate documents in Common Crawl, we apply a deduplication step to remove near-duplicate documents. We use the SimHash implementation from text-dedup (Mou 

7 

Preprint 

et al., 2023) to deduplicate the dataset using a threshold of 0.7. We find that this threshold is high enough to remove most duplicate documents even if they have slight differences in their texts. 

## 3.7 MANUAL INSPECTION 

Finally, we manually inspect the top domains by document count, the top domains by character count, and the longest documents in the dataset to ensure that the documents are high quality. We remove domains that are not high quality or clearly not mathematical by adding domains to a blacklist and adding domain filters such as removing user profile pages, abstract-hosting websites as in Lewkowycz et al. (2022), and removing search result pages. 

## 4 DATASET ANALYSIS 

**Token count** At 14.7B tokens, OpenWebMath is just below the size of Minerva’s Math Web Pages (17.5B tokens) Lewkowycz et al. (2022) and significantly larger than the web part of any other dataset. OpenWebMath has around the same number of LLaMA tokens as ProofPile (14.2B) (Azerbayev et al., 2023), but we note that there is very little overlap between between the two datasets. As a result, OpenWebMath brings a large number of new mathematical tokens that were previously unavailable to the open-source community. Due to differences in data curation strategies, it is hard to compare these datasets other than by training models on them. Since not much is known about how to properly filter a dataset, we opted to keep as much relevant content as possible. However, future work could explore filtering OpenWebMath more aggressively to further improve its quality. 

**Data Composition** We measured the distribution of domains in OpenWebMath both by document and by character count. Table 3 and Table 4 show the top twenty most common domains by document and character count respectively. The most common sources of data tend to be discussion forums, blog posts, and scientific papers. We find that the distribution of characters in the dataset is distributed over 131,206 domains, with 46% of the characters appearing in the top 100 domains. In order to get a sense of the types of documents found in the dataset, we analyzed 100,000 randomly sampled documents. First, we created embeddings of this data using all-MiniLM-L12-v2 (Wang et al., 2020) in SentenceTransformers (Reimers & Gurevych, 2019). Then, we clustered these embeddings using _k_ -Means with _k_ = 128. Finally, we took the five closest documents to each cluster center and asked gpt-3.5-turbo (https://platform.openai.com/docs/api-reference) to classify each cluster as Math, Physics, Statistics, Chemistry, Economics, Computer Science, or Other. We then aggregated these statistics, using the size of each cluster to get an estimate of the final number of documents in each category. We note several potential issues with this methodology, 

|**Domain**<br>**# Documents**<br>**% Documents**<br>stackexchange.com<br>1,136,407<br>17.99%<br>physicsforums.com<br>300,044<br>4.75%<br>mathhelpforum.com<br>170,721<br>2.70%<br>socratic.org<br>133,983<br>2.12%<br>mathoverfow.net<br>120,755<br>1.91%<br>gradesaver.com<br>96,100<br>1.52%<br>zbmath.org<br>91,939<br>1.46%<br>wordpress.com<br>87,876<br>1.39%<br>github.io<br>81,125<br>1.28%<br>brilliant.org<br>68,573<br>1.09%<br>gamedev.net<br>50,560<br>0.80%<br>openstudy.com<br>49,041<br>0.78%<br>gmatclub.com<br>48,812<br>0.77%<br>blogspot.com<br>48,036<br>0.76%<br>wikipedia.org<br>46,606<br>0.74%<br>ac.uk<br>41,342<br>0.65%<br>nature.com<br>37,403<br>0.59%<br>aimsciences.org<br>36,368<br>0.58%<br>libretexts.org<br>32,216<br>0.51%<br>readthedocs.io<br>31,455<br>0.50%|**Domain**<br>**# Characters**<br>**% Characters**|
|---|---|
||stackexchange.com<br>4,655,132,784<br>9.55%<br>nature.com<br>1,529,935,838<br>3.14%<br>wordpress.com<br>1,294,166,938<br>2.66%<br>physicsforums.com<br>1,160,137,919<br>2.38%<br>github.io<br>725,689,722<br>1.49%<br>zbmath.org<br>620,019,503<br>1.27%<br>wikipedia.org<br>618,024,754<br>1.27%<br>groundai.com<br>545,214,990<br>1.12%<br>blogspot.com<br>520,392,333<br>1.07%<br>mathoverfow.net<br>499,102,560<br>1.02%<br>gmatclub.com<br>442,611,169<br>0.91%<br>gamedev.net<br>426,478,461<br>0.88%<br>ac.uk<br>402,111,665<br>0.83%<br>aimsciences.org<br>344,716,386<br>0.71%<br>mathhelpforum.com<br>319,215,756<br>0.65%<br>deepai.org<br>313,512,520<br>0.64%<br>libretexts.org<br>282,014,149<br>0.58%<br>readthedocs.io<br>269,816,413<br>0.55%<br>tib.eu<br>199,714,017<br>0.41%<br>mit.edu<br>198,487,362<br>0.41%|



Table 3: Most Common Domains by Document Count. 

Table 4: Most Common Domains by Character Count. 

8 

Preprint 

including inaccuracies stemming from using an LLM for classification, and the potential that not every document within a cluster belongs to the predicted category. Figure 2 shows the results of this analysis. The majority of the documents in the dataset are directly related to mathematics, while the rest are spread out throughout physics, computer science, statistics, chemistry, and economics, with 12% of documents not falling neatly into any of these categories. 

We also used GPT to analyze the types of websites found in OpenWebMath. To do this, we took a sample of 200 documents and asked gpt-3.5-turbo to classify each as a Forum, Paper, Blog, Reference, Educational, Reference, or other. We also gave the document URL as a feature, since we found GPT is often able to judge the topic from the URL alone. We validated our analysis by asking GPT to do this classification on the top 100 domain names and got similar results. Figure 2 shows the results. The highest proportion of documents are forum pages, where users ask and answer questions related to mathematical subjects. There is also a large proportion of educational and reference content. 

**Downstream Performance** We ran experiments to find out how our dataset compares to other language modeling datasets. We compare models trained on OpenWebMath for a single epoch (14.7B tokens) with models trained for the same number of tokens on The Pile (Gao et al., 2020), a general langauge modeling dataset, and ProofPile (Azerbayev et al., 2023), a dataset of both formal and informal mathematics. We also train a 50/50 mixture of ProofPile and OpenWebMath to evaluate the performance of OpenWebMath when included in a mixture of other datasets, as would be common in practice. 

We train randomly initialized models with the same architecture as Pythia 1.4B (Biderman et al., 2023). We use a batch size of 1M tokens and the same hyperparameters as Pythia otherwise. These models are evaluated on a collection of mathematics benchmarks which show signal on models of this size. This includes the subset of level-1 algebra questions from MATH, LILA-multiarith to test coding ability, and GSM8k and MATH perplexities, which scale more smoothly than accuracies. We also compare to Pythia 1.4B (Biderman et al., 2023), which was trained on 300B tokens of The Pile (Gao et al., 2020) with the same architecture. 

Table 1 shows the results for our perplexity evaluations. There is a clear performance lead for models trained with OpenWebMath and the mixture seems to perform best. Despite Pythia being trained on over 20x the number of tokens, the performance of our models on the perplexity benchmarks far exceeds its performance, showing the potential of domain-specific models for mathematics. Similarly, Table 2 shows the performance of the models on MATH-Algebra-Easy and LILA-multiarith (Mishra et al., 2022). OpenWebMath models outperform models that were not trained on it by a significant margin. 

## 5 CONCLUSION 

In this paper, we describe OpenWebMath, an open dataset of 14.7B high quality mathematical documents from the web. We extensively document our pipeline, including several novel methodologies for extracting L[A] TEX formulas, reducing boilerplate, and filtering the dataset. OpenWebMath consists of high quality Q&A forum posts, educational documents, blogs, and more spread across mathematics, physics, computer science, and other technical domains. We also train several models on OpenWebMath and other language modeling datasets to compare the downstream performance achievable by training on our dataset. Notably, we find that models trained on OpenWebMath outperform models trained on 20x more general-domain tokens in mathematics. We hope that OpenWebMath can lead to the creation of language models with improved mathematical reasoning capabilities. 

9 

Preprint 

## ACKNOWLEDGEMENTS 

JB is supported by NSERC Grant [2020-06904], CIFAR AI Chairs program, Google Research Scholar Program, and Amazon Research Award. KP is supported by an NSERC PGS-D award. Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, Fujitsu Limited, and companies sponsoring the Vector Institute for Artificial Intelligence (www.vectorinstitute.ai/partners). Computing resources for model training were provided by EleutherAI and Brigham Young University. We thank Finn Paster for the graphic design for the logo. We additionally thank Ziming Chen, Yuhuai Wu, Stella Biderman, Aviya Skowron, Hailey Schoelkopf, and Sean Welleck for their helpful comments. 

10 

Preprint 

## REFERENCES 

- Alex Andonian, Quentin Anthony, Stella Biderman, Sid Black, Preetham Gali, Leo Gao, Eric Hallahan, Josh Levy-Kramer, Connor Leahy, Lucas Nestler, Kip Parker, Michael Pieler, Jason Phang, Shivanshu Purohit, Hailey Schoelkopf, Dashiell Stander, Tri Songz, Curt Tigges, Benjamin Th´erien, Phil Wang, and Samuel Weinbach. GPT-NeoX: Large scale autoregressive language modeling in PyTorch. GitHub Repo, 9 2023. URL https://www.github.com/ eleutherai/gpt-neox. 

- Zhangir Azerbayev, Bartosz Piotrowski, Hailey Schoelkopf, Edward W Ayers, Dragomir Radev, and Jeremy Avigad. Proofnet: Autoformalizing and formally proving undergraduate-level mathematics. _arXiv preprint arXiv:2302.12433_ , 2023. 

- Adrien Barbaresi. Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction. In _Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations_ , pp. 122–131. Association for Computational Linguistics, 2021. URL https://aclanthology.org/2021.acl-demo.15. 

- Janek Bevendorff, Benno Stein, Matthias Hagen, and Martin Potthast. Elastic ChatNoir: Search Engine for the ClueWeb and the Common Crawl. In Leif Azzopardi, Allan Hanbury, Gabriella Pasi, and Benjamin Piwowarski (eds.), _Advances in Information Retrieval. 40th European Conference on IR Research (ECIR 2018)_ , Lecture Notes in Computer Science, Berlin Heidelberg New York, March 2018. Springer. 

- Janek Bevendorff, Martin Potthast, and Benno Stein. FastWARC: Optimizing Large-Scale Web Archive Analytics. In Andreas Wagner, Christian Guetl, Michael Granitzer, and Stefan Voigt (eds.), _3rd International Symposium on Open Search Technology (OSSYM 2021)_ . International Open Search Symposium, October 2021. 

- Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. Pythia: A suite for analyzing large language models across training and scaling. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 2397–2430. PMLR, 2023. URL https:// proceedings.mlr.press/v202/biderman23a.html. 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, 

11 

Preprint 

Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. _CoRR_ , abs/2204.02311, 2022. doi: 10.48550/arXiv.2204.02311. URL https://doi.org/10.48550/arXiv.2204.02311. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Katherine M Collins, Albert Q Jiang, Simon Frieder, Lionel Wong, Miri Zilka, Umang Bhatt, Thomas Lukasiewicz, Yuhuai Wu, Joshua B Tenenbaum, William Hart, et al. Evaluating language models for mathematics through interactions. _arXiv preprint arXiv:2306.01694_ , 2023. 

- Istv´an Endr´edy and Attila Nov´ak. More effective boilerplate removal-the goldminer algorithm. _Polibits_ , 48:79–83, 12 2013. doi: 10.17562/PB-48-10. 

- Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. _arXiv preprint arXiv:2101.00027_ , 2020. 

- Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daum´e III au2, and Kate Crawford. Datasheets for datasets, 2021. 

- Xinyang Geng and Hao Liu. Openllama: An open reproduction of llama, May 2023. URL https: //github.com/openlm-research/open_llama. 

- Kenneth Heafield. Kenlm: Faster and smaller language model queries. In _Proceedings of the sixth workshop on statistical machine translation_ , pp. 187–197, 2011. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_ , 2020. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. _CoRR_ , abs/2103.03874, 2021. URL https://arxiv.org/abs/2103.03874. 

- Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, H´erve J´egou, and Tomas Mikolov. Fasttext.zip: Compressing text classification models. _arXiv preprint arXiv:1612.03651_ , 2016. 

- Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. _Advances in Neural Information Processing Systems_ , 35:3843–3857, 2022. 

- Hunter Lightman, Vineet Kosaraju, Yura Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. _CoRR_ , abs/2305.20050, 2023. doi: 10.48550/arXiv.2305.20050. URL https://doi.org/10. 48550/arXiv.2305.20050. 

- Gurmeet Singh Manku, Arvind Jain, and Anish Das Sarma. Detecting near-duplicates for web crawling. In _Proceedings of the 16th International Conference on World Wide Web_ , WWW ’07, pp. 141–150, New York, NY, USA, 2007. Association for Computing Machinery. ISBN 9781595936547. doi: 10.1145/1242572.1242592. URL https://doi.org/10.1145/ 1242572.1242592. 

- Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, et al. Lila: A unified benchmark for mathematical reasoning. _arXiv preprint arXiv:2210.17517_ , 2022. 

- Chenghao Mou, Chris Ha, Kenneth Enevoldsen, and Peiyuan Liu. Chenghaomou/text-dedup: Reference snapshot, September 2023. URL https://doi.org/10.5281/zenodo.8364980. 

12 

Preprint 

## OpenAI. Gpt-4 technical report, 2023. 

- Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon LLM: outperforming curated corpora with web data, and web data only. _CoRR_ , abs/2306.01116, 2023. doi: 10.48550/arXiv.2306.01116. URL https://doi.org/10. 48550/arXiv.2306.01116. 

- Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving. _arXiv preprint arXiv:2009.03393_ , 2020. 

- Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, H. Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew J. Johnson, Blake A. Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher. _CoRR_ , abs/2112.11446, 2021. URL https://arxiv.org/abs/2112.11446. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _The Journal of Machine Learning Research_ , 21(1):5485–5551, 2020. 

- Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics, 11 2019. URL https://arxiv. org/abs/1908.10084. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aur´elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. _CoRR_ , abs/2302.13971, 2023a. doi: 10.48550/arXiv.2302.13971. URL https://doi.org/10.48550/arXiv.2302.13971. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023b. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aur´elien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and finetuned chat models. _CoRR_ , abs/2307.09288, 2023c. doi: 10.48550/arXiv.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288. 

13 

Preprint 

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers, 2020. 

- Sean Welleck, Jiacheng Liu, Ronan Le Bras, Hannaneh Hajishirzi, Yejin Choi, and Kyunghyun Cho. Naturalproofs: Mathematical theorem proving in natural language. _arXiv preprint arXiv:2104.01112_ , 2021. 

- Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzm´an, Armand Joulin, and Edouard Grave. Ccnet: Extracting high quality monolingual datasets from web crawl data. _arXiv preprint arXiv:1911.00359_ , 2019. 

14 

Preprint 

|**Method**|**Runtime (s)**|**Source Code Link**|
|---|---|---|
|Resiliparse|3.99|https://github.com/chatnoir-eu/chatnoir-resiliparse|
|HTML-Text|10.75|https://github.com/TeamHG-Memex/html-text|
|Inscripts|19.14|https://github.com/weblyzard/inscriptis|
|BoilerPy|24.94|https://github.com/jmriebold/BoilerPy3|
|jusText|31.17|https://github.com/miso-belica/jusText|
|HTML2Text|37.17|https://github.com/Alir3z4/html2text/|
|BeautifulSoup|38.42|https://code.launchpad.net/beautifulsoup|
|Traflatura|63.90|https://github.com/adbar/traflatura|
|ExtractNet|299.67|https://github.com/currentslab/extractnet|



Table 5: We measured the performance of various HTML text extraction tools on a dataset of 1k documents. Resiliparse was by far the most efficient, leading us to choose it for use in our pipeline. 

## A LIMITATIONS AND FUTURE WORK 

Despite the high quality of OpenWebMath, we note several limitations and avenues for future works. First, due to the high cost of extracting data from all shards on Common Crawl, we were only able to run our pipeline once. Therefore, many of our choices are without empirical justification and we provide no ablation study. We also note that the nature of this particular type of dataset means that there are many subjective choices to be made. For instance, what counts as a mathematical document? What is a high-quality document? How do we choose the threshold for near-deduplication? For each of these, we chose several values and manually inspected a few examples to choose. Due to the cost constraints, there are also practical challenges with balancing cost with accuracy when filtering and extracting text. For instance, our prefilter reduces the number of HTML documents processed to under 1% of the documents in Common Crawl, which may be too aggressive. We also note that OpenWebMath is an English-only dataset, which limits its applications for researchers and users who speak other languages. Finally, we note that OpenWebMath only contains the _text_ from math on the web, not associated figures, which can be important for solving mathematical problems (OpenAI, 2023). Future work should focus on finding empirical answers to the questions of what constitutes good data, creating new, efficient filtering methodologies, and extracting images inline with math text. 

## B TEXT EXTRACTION 

**Choice of Base Text Extractor** When considering which HTML text-extraction library to use, we considered the efficiency, customization, and existing boilerplate reduction methods for each option. The most commonly used option, using WET files extracted by Common Crawl, was not an option since they do not deal with L[A] TEX correctly and offer no customization. Other options such as jusText (Endr´edy & Nov´ak, 2013), used in The Pile Gao et al. (2020), removed boilerplate too aggressively, leading to sections containing math to be discarded. Likewise, Trafilatura (Barbaresi, 2021), which was used in RefinedWeb (Penedo et al., 2023), had poor efficiency. We decided to go with Resiliparse (Bevendorff et al., 2018) due to its balanced boilerplate removal, fast runtime, and efficient Common Crawl parsing tools. Table 5 shows the full results for our comparison. 

**L[A] TEX Extraction** L[A] TEX code comes in many forms throughout Common Crawl HTML files. We employed an iterative process to refine our extraction rules. First, we filtered shards of Common Crawl for documents that contain the string _\_ frac. Then, we filtered those documents to find those which our extraction code found no extractable L[A] TEX. Then, we refined our code to include additional sources of math until we were confident that we had reasonable support for all formats of L[A] TEX in HTML documents. Table 6 shows the breakdown of different common types of L[A] TEX found in HTML documents. 

We note that most of the L[A] TEX in OpenWebMath and across the internet is encoded using MathJax, which presents a challenge. The majority of MathJax documents use dollar sign delimiters, but most dollar signs on the web do not delimit L[A] TEX equations. This leaves us with a few options: 

15 

Preprint 

|**Math Format**|**Percentage of Documents**|
|---|---|
|Found at least one instance of math|91.42%|
|MathJax with delimiters (inline)|50.27%|
|MathJax with delimiters (display)|23.37%|
|Math found in images|6.96%|
|.math-container|3.94%|
|MathML code|3.28%|
|<annotation>withing<math>tags|2.35%|
|<mathjax>tags|2.24%|
|alignenvironments|1.72%|
|equationenvironments|1.18%|
|within<script>tags|1.01%|
|alttextproperty of<math>tags|0.24%|



Table 6: Frequencies of different types of L[A] TEX found in OpenWebMath. The most common format of L[A] TEX found in Common Crawl is MathJax, which uses user-defined delimiters to denote math equations. Second most common is L[A] TEX code within either the URL or alt text of an img tag. 

|Model|Size|Layers|Model Dim|Heads|Learning Rate|Batch Size|
|---|---|---|---|---|---|---|
|1.4|B|24|2048|16|2_._0_×_10_−_4|1M|



Table 7: Model Hyperparameters. We use the same architecture and hyperparameters, other than batch size, as Pythia 1.4B (Biderman et al., 2023). 

Table 8: List of Math Keywords used in the prefiltering stage. 

**Math Keywords** MathJax mathjax <math math-container katex.min.css latex.php codecogs tex.cgi class="tex" class=’tex’ 

- Detect the use of the MathJax script in the HTML file. If the script is imported, treat dollar signs as L[A] TEX code. 

- Detect common L[A] TEX commands in between dollar signs. If they are present, treat dollar signs as L[A] TEX code. 

- Use the _MathScore_ classifier to determine whether the page looks like it is talking about math. If so, treat dollar signs as L[A] TEX code. 

The first option is not always accurate since the MathJax javascript code may be nested inside of another import or named differently depending on the website. The latter two options make up for many of these cases, but can fail to detect edge cases where math equations are present but the surrounding text does not indicate that the document is mathematical. We suspect Minerva (Lewkowycz et al., 2022) gets around this issue by using HTML documents where javascript code has already been executed, in which case MathJax is converted from delimited text to explicit HTML tags that are easy to detect. 

16 

Preprint 

## C INTERPLAY BETWEEN EXTRACTION AND FILTERING 

In prior works, we noticed many cases where suboptimal HTML text extractors were used and yet text quality remains high in the dataset. This is due to the interplay between extraction and filtering. Specifically, if a text extractor fails to extract the main text, gets the formatting wrong, or includes too much boilerplate in the extraction, then both the classification and perplexity filters can filter out such examples. This can lead to subtle biases in the dataset, where specific poorly-extracted websites are excluded entirely even though they do contain high quality content. In the case of making a mathematical dataset, failure to extract and deal with inline L[A] TEX code properly can hurt perplexity scores and lead to these documents being filtered out. We suggest practitioners tune their text extraction pipeline on a diverse set of documents before applying filtering to avoid this bias. 

## D MODEL HYPERPARAMETERS 

We trained models on 14.7B tokens using the LLaMA (Touvron et al., 2023c) tokenizer and the architecture described in Pythia (Biderman et al., 2023). We train the model using the GPT-NeoX library (Andonian et al., 2023) on 8 A100 80GB GPUs. Exact hyperparameters can be found in Table 7. 

17 

Preprint 

## E DATASHEET 

We provide a datasheet for OpenWebMath, following the framework in Gebru et al. (2021). 

|**MOTIVATION**|**MOTIVATION**|
|---|---|
|||
|**For what purpose was the dataset cre-**<br>**ated?**|The dataset was created to enable the train-<br>ing of large language models on mathemat-<br>ical texts, in order to improve their mathe-<br>matical reasoning capabilities.|
|||
|**Who created the dataset and on behalf**<br>**of which entity?**|The dataset was created by the authors of<br>this work.|
|||
|**Who funded the creation of the dataset?**|Resources used in preparing this research<br>were provided, in part, by the Province<br>of Ontario, the Government of Canada<br>through CIFAR, Fujitsu Limited,<br>and<br>companies sponsoring the Vector Insti-<br>tute<br>for<br>Artifcial<br>Intelligence<br>(www.<br>vectorinstitute.ai/partners).<br>Computing resources for model training<br>were provided by EleutherAI and Brigham<br>Young University.|
|||
|**Any other comment?**|None.|
|**COMPOSITION**||
|||
|**What do the instances that comprise the**<br>**dataset represent?**|The instances are text documents extracted<br>from mathematics-related webpages from<br>Common Crawl.|
|||
|**How many instances are there in total?**|In total, OpenWebMath contains 6.3 mil-<br>lion documents.|
|||
|**Does the dataset contain all possible in-**<br>**stances or is it a sample (not necessarily**<br>**random) of instances from a larger set?**|OpenWebMath<br>doesn’t<br>contain<br>all<br>instances<br>of<br>text<br>extracted<br>from<br>mathematics-related<br>webpages<br>from<br>Common Crawl, as our flters can miss a<br>non-zero proportion of such webpages.<br>However, we expect OpenWebMath to<br>contain most of them.|
|||
|**What data does each instance consist of?**|Each instance consists of plain text and<br>metadata including the source URL, the<br>snapshot date, and other extraction param-<br>eters.|
|||
|**Is there a label or target associated with**<br>**each instance?**|No.|
|||
|**Is any information missing from individ-**<br>**ual instances?**|No.|
|||
|**Are relationships between individual in-**<br>**stances made explicit?**|No.|
|||
|**Are there recommended data splits?**|No.|
|||
|**Are there any errors, sources of noise, or**<br>**redundancies in the dataset?**|Yes, a small portion of the documents from<br>OpenWebMath are not related to mathe-<br>matics, or contain bad quality content.|



18 

Preprint 

**Is the dataset self-contained, or does it** The dataset is entirely self-contained. **link to or otherwise rely on external resources?** 

|**Is the dataset self-contained, or does it**<br>**link to or otherwise rely on external re-**<br>**sources?**<br>The dataset is entirely self-contained.|**Is the dataset self-contained, or does it**<br>**link to or otherwise rely on external re-**<br>**sources?**<br>The dataset is entirely self-contained.|
|---|---|
|**Does the dataset contain data that might**<br>**be considered confdential?**|No.|
|||
|**Does the dataset contain data that, if**<br>**viewed directly, might be offensive, in-**<br>**sulting, threatening, or might otherwise**<br>**cause anxiety?**|The data is fltered for quality and we do<br>not expect that this content will be offen-<br>sive, but since our flters may be imperfect<br>we make no guarantees.|
|**COLLECTION**||
|||
|**How was the data associated with each**<br>**instance acquired?**|The data was acquired by processing data<br>from Common Crawl.|
|||
|**What mechanisms or procedures were**<br>**used to collect the data?**|We refer to the CommonCrawl website<br>(commoncrawl.org) for details on how they<br>collect data.|
|||
|**If the dataset is a sample from a larger**<br>**set, what was the sampling strategy?**|We use all data from Common Crawl that<br>was available before May 2023.|
|||
|**Who was involved in the data collec-**<br>**tion process and how were they compen-**<br>**sated?**|Keiran Paster and Marco Dos Santos col-<br>lected the data and were compensated by<br>their respective graduate programs.|
|||
|**Over what timeframe was the data col-**<br>**lected?**|OpenWebMath uses shards of Common-<br>Crawl gathered between 2013 and 2023.|
|||
|**Were any ethical review processes con-**<br>**ducted?**|No.|
|**PREPROCESSING**||
|||
|**Was any preprocessing/cleaning/label-**<br>**ing of the data done?**|Yes. Seesection 3.5for details.|
|||
|**Was the “raw” data saved in addition to**<br>**the preprocessed/cleaned/labeled data?**|Yes.|
|||
|**Is the software that was used to prepro-**<br>**cess/clean/label the data available?**|Yes. See supplementary materials.|
|**USES**||
|||
|**Has the dataset been used for any tasks**<br>**already?**|Yes, the data was used to train 1.4B param-<br>eter language models insection 4|
|||
|**Is there a repository that links to any**<br>**or all papers or systems that use the**<br>**dataset?**|No.|
|||
|**What (other) tasks could the dataset be**<br>**used for?**|We primarily envision that OpenWebMath<br>could be useful for language model pre-<br>training, fnetuning, and evaluation.|
|||
|**Is there anything about the composition**<br>**of the dataset or the way it was col-**<br>**lected and preprocessed/cleaned/labeled**<br>**that might impact future uses?**|It is possible that the fltering stage of<br>the project discarded valuable documents,<br>such as those not written in English. This<br>makes OpenWebMath suboptimal for cre-<br>ating mathematical models in other lan-<br>guages.|
|||
|**Are there tasks for which the dataset**<br>**should not be used?**|Any tasks which may considered irrespon-<br>sible or harmful.|
|**DISTRIBUTION**||



19 

Preprint 

**Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?** 

Yes, the dataset will be available on the Hugging Face Hub for NLP practitioners. 

|**which the dataset was created?**||
|---|---|
|**How will the dataset will be distributed?**|We will distribute the dataset on the Hug-<br>ging Face Hub|
|**When will the dataset be distributed?**<br>The dataset will be available when the pa-<br>per is made public.||
|**Will the dataset be distributed under**<br>**a copyright or other intellectual prop-**<br>**erty (IP) license, and/or under applica-**<br>**ble terms of use (ToU)?**<br>The public extract is made available un-<br>der an ODC-By 1.0 license; users should<br>also abide to the CommonCrawl ToU:<br>https://commoncrawl.org/terms-of-use/.||
|**Have any third parties imposed IP-**<br>**based or other restrictions on the data**<br>**associated with the instances?**<br>Not to our knowledge.||
|**Do any export controls or other regula-**<br>**tory restrictions apply to the dataset or**<br>**to individual instances?**<br>Not to our knowledge.||
|**MAINTENANCE**||
|||
|**Who will be supporting/hosting/main-**<br>**taining the dataset?**|The dataset will be hosted on the Hugging<br>Face Hub.|
|||
|**How can the owner/curator/manager of**<br>**the dataset be contacted?**|keirp@cs.toronto.edu|
|||
|**Is there an erratum?**|No.|
|||
|**Will the dataset be updated?**|No.|
|||
|**If others want to extend/augment/build**<br>**on/contribute to the dataset, is there a**<br>**mechanism for them to do so?**|No.|



Table 9: **Datasheet for OpenWebMath** , following the framework introduced by Gebru et al. (2021). 

20 

