# **MATHPILE : A Billion-Token-Scale Pre-training Corpus for Math** 

**Zengzhi Wang[1,3,4] Xuefeng Li[1,4] Rui Xia[3] Pengfei Liu[1,2,4]** _[∗]_ 

1Shanghai Jiao Tong University 2Shanghai Artificial Intelligence Laboratory 

3Nanjing University of Science and Technology 4Generative AI Research Lab (GAIR) zzwang.nlp@gmail.com pengfei@sjtu.edu.cn 

## **Abstract** 

High-quality, large-scale corpora are the cornerstone of building foundation models. In this work, we introduce MATHPILE , a diverse and high-quality math-centric corpus comprising about 9.5 billion tokens. Throughout its creation, we adhered to the principle of “ _less is more_ ”, firmly believing in the supremacy of data quality over quantity, even in the pre-training phase. Our meticulous data collection and processing efforts included a complex suite of preprocessing, prefiltering, language identification, cleaning, filtering, and deduplication, ensuring the high quality of our corpus. Furthermore, we performed data contamination detection on downstream benchmark test sets to eliminate duplicates and conducted continual pre-training experiments, booting the performance on common mathematical reasoning benchmarks. We aim for our MATHPILE to boost language models’ mathematical reasoning abilities and open-source its different versions and processing scripts to advance the field (available at https://github.com/GAIR-NLP/MathPile/). 

## **1 Introduction** 

High-quality, diverse pre-training corpora form the cornerstone for developing powerful foundation models, enabling AI assistants like ChatGPT [47] to exhibit balanced competencies across a broad spectrum of tasks [11]. In this work, our concern centers on mathematical reasoning capabilities within foundational language models [13, 5, _inter alia_ ], for which can potentially boost the applicaMath-Centric Diversity tion in education tools, automated problem solving, data “Less is More’’ Principle Textbooks Lecture Notes Web analysis, etc., thereby improving user experience. To Open-Source 9.5B Tokens arXiv StackExchange Definitions facilitate this, we are not directly building a model, but K-12 College Postgraduate Wikipedia Theorem Proofs rather focusing on a more fundamental aspect: _creating a_ Math Competition _high-quality and diverse pre-training corpus tailored for the math domain_ , namely MATHPILE . Specifically, our High-Quality Data DocumentationDocumentation work is significantly different from the previous work in Preprocessing Prefiltering Dataset Sheet Quality Annotations the following characteristics (cf. Table 1 for comparison): Language ID Cleaning & Filtering 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math_images/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math.pdf-0001-09.png)


**----- Start of picture text -----**<br>
Math-Centric Diversity<br>“Less is More’’ Principle Textbooks Lecture Notes Web<br>Open-Source 9.5B Tokens<br>arXiv StackExchange Definitions<br>K-12 College Postgraduate<br>Wikipedia Theorem Proofs<br>Math Competition<br>High-Quality Data DocumentationDocumentation<br>Preprocessing Prefiltering<br>Dataset Sheet Quality Annotations<br>Language ID Cleaning & Filtering<br>Dedup. Data Contamination Detect. Length Distribution Data Statistics<br>**----- End of picture text -----**<br>


**Math-centric** . Previous open-sourced pre-training corpora have typically focused on general domains, such as Pile [20], RedPajama [59] and Dolma [2]. Others Figure 1: Key features of MATHPILE . have concentrated on multilingual aspects or programming languages, such as ROOTS [32] and The Stack [30], 

respectively. However, a notable absence in these offerings is a corpus specificlly tailoring for mathematics. While there exist some corpora designed for training or continually improving math-specific 

> _∗_ Corresponding author 

38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks. 

language models, such as Minerva’s mathematical training dataset [35] and OpenAI’s MathMix [37], these are not open-sourced. Note that a recent work concurrent with ours, OpenWebMath [49], although math-centric, is solely sourced from web pages. We will discuss the comparison with it later. Recognizing this gap, we aim to democratize access to high-quality mathematical corpus, fostering inclusive advancements in language models’ mathematical reasoning. 

**Diversity** . While Hendrycks et al. [26] introduced AMPS, a problem set ranging from elementary mathematics to multivariable calculus (K-12 level) for pre-training purposes, it lacks content at the college-level and more challenging competition-level mathematics, focusing instead on a supervised dataset rather than an extensive corpus. The ProofPile corpus, introduced by Azerbayev et al. [4], aims to improve autoformalization and formal proving capabilities in models, yet its scope is confined to formal proving, not covering the broader mathematical domain from K-12 to postgraduate level. Concurrently with our work, Paster et al. [49] propose the OpenWebMath corpus, featuring a corpus composed of mathematical web pages. However, our corpus goes beyond web pages, integrating highquality mathematics textbooks, lecture notes, scientific papers from arXiv in the field of mathematics, and carefully selected content from StackExchange, ProofWiki, and Wikipedia among others, which positions our corpus as a richer and more diverse mathematical resource for language models. 

**High-Quality** . Recent studies have increasingly highlighted the detrimental effects of low-quality and repeated content in pre-training corpora on model training, as evidenced in various works [1, 42, 33, 27, 41]. The importance of high-quality datasets has thus come to the fore. It has been shown that properly filtered and deduplicated web data can yield models as equally powerful as those trained on curated, high-quality corpora [51]. This similar practice has been recently adopted in several notable studies [12, 2, 60]. A notable example is the 1.3 billion-parameter code-focused model pre-trained on synthetically generated textbooks and filtered web pages, a project that broke existing scaling laws although did not open source its data [24]. It’s important to emphasize that quality of the corpus is far more significant than its quantity. For instance, OpenAI’s MathMix comprises only 1.5 billion tokens. In this work, we diligently adhere to the principle of _less is more_ , as outlined in Zhou et al. [71]. To achieve a high-quality corpus, Unlike other approaches that uniformly process all data, we have conducted specialized preprocessing and prefiltering for each data source before global data processing (including language identification, filtering, cleaning, and deduplication). We’re dedicated to refining and optimizing our corpus, making a distinctive contribution to the field. 

**Data Documentation** . Auditing large-scale pre-training corpora is essential for identifying content characteristics, intended uses, and potential biases, despite challenges due to their size [7, 21, 44]. However, many such corpora are released without proper documentation [45]. Recent audits of certain pre-training corpora have uncovered issues such as irrelevant content [42, 31, 19], copyright infringement [6], and inclusion of test sets for downstream tasks [1, 18], highlighting the need for detailed data sheets and transparent documentation. To this end, following previous efforts to enhance corpora transparency, we have provided a dataset sheet for our MATHPILE (see Table 6). Throughout our extensive data processing workflow, numerous documents were annotated for quality, such as language identification scores and the ratio of symbols to words (as exemplified in Figure 12). These quality annotations enable future users to apply their specific filters based on these scores. Additionally, we have conducted extensive deduplication for this corpus and performed data contamination detection with downstream benchmark test sets, removing any duplicated samples identified (cf. § 3.4). Interestingly, we have also discovered a significant number of questions from downstream test sets in OpenWebMath (cf. § 3.4). This underscores the importance of meticulous data documentation. We plan to release different versions to facilitate future use. See Appendix C for examples. 

Additionally, we conducted continual pre-training experiments on MATHPILE and found that it generally enhances the performance of language models across various mathematical reasoning benchmarks, with an average improvement of up to 5% (cf. § 4.2). In conclusion, we hope to facilitate the growth of the field of AI for mathematics by contributing this specialized, high-quality, diverse corpus focused on the mathematical domain while maintaining utmost transparency about the data for practitioners. 

## **2 The Collection of Corpora** 

In order to construct MATHPILE , we gather data from a variety of sources, which also includes a component of manual collection. We provide an ethics statement regarding copyright in Appendix B. 

2 

Table 1: The comparison of MATHPILE with other mathematical Corpora, where PS denotes the problem set type. For non-open-sourced corpora, details are inferred from literature, with unknowns marked as “?”. Token counts may vary by tokenizer; we use statistics from each dataset’s report and the GPTNeoX-20B tokenizer [9] for our corpus. DM-Mathematics is from Saxton et al. [56]. "Minerva" refers to its dataset. ProofPile-2 [5], encompassing OpenWebMath and others, is excluded from this comparison. 

|**Datasets**|**Open**<br>**Source**|**Type**|**Target Domain**|**# Textbooks**|**Has Synth.**<br>**Data**|**Data Contam.**<br>**Detection**|**# Tokens**|**Source**|
|---|---|---|---|---|---|---|---|---|
|Minerva|✗|Corpus|General Math|✗|✗|✓|38.5B|arXiv, Web|
|MathMix|✗|Corpus + PS|General Math|**?**|✓|✓|01.5B|**?**|
|ProofPile|✓|Corpus|Theorem Proving|7|✗|✗|08.3B|arXiv, Textbooks, Lib., StackExchange,<br>ProofWiki, MATH|
|OpenWebMath|✓|Corpus|General Math|✗|✗|✗|14.7B|Web|
|DM-Mathematics|✓|PS|Math Competition|✗|✓|**-**|04.4B|Synthesis|
|AMPS|✓|PS|Math Competition|✗|✓|✗|00.7B|Khan Academy, Synthesis|
|MATHPILE(Ours)|✓|Corpus|General Math|3,979|✓|✓|09.5B|arXiv, Textbooks, StackExchange,<br>Wikipedia, ProofWiki, Web|



**Mathematical Textbooks** Textbooks, covering mathematical concepts, exercises, and solutions, are valuable for _educational purposes_ for both humans and machines. Recent studies, even though not focused on math, support this view with synthesized textbooks [24, 36]. To collect these genuine and high-quality textbooks, we began by conducting extensive manual searches across the internet, seeking open-source and freely accessible mathematics-related textbook websites. Afterwards, we proceeded to download these PDF files, resulting in a collection of 38 K-12 level textbooks, along with 369 college-level mathematics textbooks that cover a wide range of subjects including linear algebra, probability theory, calculus, and optimization. In addition to these textbooks, we also included 467 college course handouts and lecture notes, which tend to be more concise compared to full-length textbooks. Subsequently, we employed the Mathpix API[2] to parse the PDFs into markdown format. Then, we meticulously cleaned up extraneous elements such as parsed image URLs, preface sections, table of contents, acknowledge sections, index sections, and consecutive empty lines within the parsed content, resulting in a total of 874 documents. 

We also refined high-quality mathematics-related synthetic textbooks from OpenPhi Project.[3] It is an open-source counterpart to the Phi work [24]. While the underlying model and generation process differ, the output encompasses a broad spectrum of subjects, extending beyond programming. To isolate mathematics-related documents, we employed a straightforward criterion: the presence of the symbol “$$” combined with common mathematical expressions like “\{mathbf” and “\{frac”. While “$$” alone is not always reliable, combining it with these symbols improves accuracy based on manual verification. This approach yielded 3,889 documents from an initial pool of 124,493. As the volume of pre-training data escalates, the synthesis of high-quality data becomes increasingly crucial. More advanced filtering methods and mathematical corpora synthesis are left for future exploration. 

**Mathematical Papers from ArXiv** ArXiv offers a free distribution service and serves as an open-source archive housing millions of scientific papers. It also provides invaluable training data for numerous powerful language models [61, 59, _inter alia_ ]. In our endeavor to collect mathematical papers from ArXiv, we identify 50 sub-subjects spanning Mathematics, Computer Science, Statistics, Physics, Quantitative Finance and Economics. Our process involved filtering ArXiv’s metadata[4] to focus on the chosen subjects (cf. Table 7), followed by accessing the source LaTex files (if available). We exclusively retained the LaTex files and consolidated multiple files based on their respective order as indicated by commands such as “include” and “input” within the main LaTex file of each paper. Subsequently, we undertook extensive transformations to enhance data clarity and consistency, including removing comments, reverting macros, omitting figures but keeping captions, excluding acknowledgements and references, condensing empty lines, replacing some formatting commands, substituting titles, and preserving only the main body content (cf. § D for more details). Finally, we compiled 347,945 meticulously cleaned LaTex documents (around 8.5 billion tokens), with each document corresponding to a single paper. 

> 2https://mathpix.com/ocr 

> 3https://huggingface.co/open-phi 

> 4https://www.kaggle.com/datasets/Cornell-University/arxiv 

3 

**Mathematical Entries in Wikipedia** Wikipedia is one the largest and most popular free online encyclopedias, offering information on a wide range of topics, including history, science, technology, culture, and more. This extensive knowledge has proven to be highly beneficial for numerous natural language processing tasks [34, _inter alia_ ] and pre-trained language models [17, 61, _inter alia_ ]. To collect mathematical entries from Wikipedia, we downloaded the mathematics-focused (without pictures) dump of Wikipedia in English for the month of August 2023. We extracted the HTML documents from the dump using the library libzim, resulting in approximately 106,900 documents. Subsequently, we converted these HTML documents into markdown format using the html2text library[5] while removing the hyperlinks following the practice of LLaMA [61]. We retained the alternative text content but excluded image (often in SVG format) paths. Additionally, we eliminated extra newlines within paragraphs and condensed more than three consecutive empty lines to two using regular expressions. Further refinement involved the removal of boilerplate content at the bottom of the pages, typically denoted with phrases like “This article is issued from Wikipedia. The text is ...”. In the end, our efforts yielded a collection of 106,881 mathematical Wikipedia entries, about 0.8 billion tokens. 

**Entries from ProofWiki** ProofWiki, an online compendium of mathematical proofs, has been instrumental in advancing the fields of autoformalization and formal proof proving, as evidenced by NaturalProofs [65] and ProofPile. We sourced data from the ProofWiki dump dated April 9, 2022 (provided by the Internet Archive), mirroring the preprocessing approach employed by NaturalProofs, which was based on the version from November 12, 2020. Specifically, this involved leveraging the BeautifulSoup library to parse all wiki pages followed by the extraction of raw text content using the wikitextparser library. This process yielded a substantial collection of mathematical content, totaling about 7.6 million tokens, comprising 10,328 definitions and 13,511 theorem-proof pairs. To facilitate better data organization, we formatted the definitions using the “definition” environment, and the theorem-proof pairs within the “section” environment with their respective titles serving as the section headings, similar to ProofPile. 

**Mathematical Discussions on StackExchange** StackExchange, renowned for its network of community-powered question-and-answering websites, spans a wide array of topics, each concentrated on a particular topic. Its high-quality data trove has significantly contributed to the development of various language models [61, 71, _inter alia_ ]. In our study, we identify eleven sites within this network, including five dedicated to mathematics (such as Mathematics and MathOverflow) and six others in closely related fields like Physics (cf. Table 8). Our data collection process began with downloading the site dumps from August 2023 (provided by the Internet Archive). We only retained the essential components in the posts, namely questions and answers (also associated meta information). To convert HTML documents to raw text, we utilized the BeautifulSoup library, coupled with a meticulous removal of invalid XML characters. We then systematically paired questions and their respective answers. Each question typically garners multiple responses, each with its own score and in some cases, an endorsement as the accepted answer by the questioner. To guarantee quality, we applied a quality threshold (i.e., 5) for filtering. Questions underwent filtering based on the threshold, whereas answers were assessed by either the threshold or the score of the accepted answer, whichever was lower. Unanswered questions scoring at least 10 were preserved for potential future use. This rigorous process resulted in a rich collection of data, comprising 267,919 questions, 435,129 answers, and 3,418 unanswered questions, totaling about 254 million tokens. 

**Mathematical Web Pages from Common Crawl** Common Crawl, an archive of web data since 2007, is crucial for training advanced language models like GPT-3 [10] and LLaMA. Our work targets extracting math web pages from SlimPajama [12], a cleaned and deduplicated version of RedPajama, focusing on its CommonCrawl and C4 subsets. Eschewing the common approach of using neutral network-based filtering, we opt for heuristic rule-based methods. Our procedure began with the creation of TF-IDF features, derived from our curated high-quality textbooks. During this process, we removed the stop words, limited the features to a maximum of 10,000, and employed white space tokenization. Upon the observation of the resulting vocabulary, we identified 11 commonly used LaTex commands, integral to mathematical expressions. We utilize these commands as a basis for a hard match within each document. A document is classified as mathematical if it contains any of these commands along with the symbol “$$”, typically indicative of a mathematical document. This 

> 5We later found that the html2text library resulted in the LaTeX display issue in the cleaned documents (cf. Figure 11). Switching to another library Resiliparse with DOM parsing resolved this issue, ensuring correct LaTeX display. 

4 

rule-based approach, though simplistic, proved to be highly effective, especially given the vast size of the Common Crawl corpus . We also experimented with more intricate dense embedding-based methods to identify mathematical documents, but these resulted in poor recall. Our efforts resulted in the compilation of a substantial collection of mathematical web pages: 4,307 documents from SlimPajama-C4 and 72,137 documents from SlimPajama-CommonCrawl, totaling approximately 633 million tokens. We acknowledge the potential for more efficient methods to sift mathematical documents from Common Crawl snapshots, an area we plan to explore in future work. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math_images/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math.pdf-0005-01.png)


**----- Start of picture text -----**<br>
520B Textbooks (0.44%)<br>Data Preprocessing  Wikipedia (2.51%)<br>& Prefiltering<br>StackExchange  ProofWiki<br>(48.02%) (2.64%)<br>Language ID Common Crawl<br>(8.32%)<br>arXiv (38.07%)<br>Cleaning & Filtering<br>Deduplication<br>Data Sources<br>29GB<br>2.2TB, ~520B Tokens 903k Docs, ~9.5B Tokens<br>**----- End of picture text -----**<br>


Figure 2: The creation process of MATHPILE . We additionally perform data contamination detection on benchmark test sets (cf. § 3.4). We visualize its component ratios by document counts (Right). 

## **3 Global Data Processing** 

After conducting specific data preprocessing for each data source during the data collection process, we globally engage in three critical steps: language identification, filtering, and deduplication, to ensure the quality of the entire corpus, as shown in Figure 2. 

## **3.1 Language Identification** 

To filter non-English documents, we utilized the fastText language identifier, which was trained on Wikipedia, Tatoeba, and SETimes [29, 23]. A common practice is to classify a document as its respective language if the score exceeds 0.5, a threshold also employed by CCNet [66]. However, during the application of this practice, we encountered a considerable number of false positives—cases where documents were erroneously filtered as non-English when, in fact, they were written in English but contained a substantial amount of mathematical symbols. We attribute this issue to the domain gap between the fastText training datasets and the mathematical content. To enhance non-English document filtering, we set customized score thresholds for each data source. Specifically, Wikipedia and StackExchange thresholds were set at 0.1, arXiv at 0.3, and Common Crawl at 0.5. No thresholds were applied to ProofWiki and Textbooks due to manual verification ensuring English content. This refinement removed about 8,400 documents, totaling 231 million tokens. 

## **3.2 Data Cleaning and Filtering** 

Despite thorough preprocessing, some documents, especially from sources like Wikipedia and Common Crawl, lack quality for language modeling due to brevity or automated content. Existing filtering methods [55, 54, 41, 51, 12], while detailed, risk excluding valuable documents in our math-focused corpus if directly applying them as-is. To address this issue, we developed a unique set of cleaning and filtering heuristic rules, specifically crafted for the mathematical domain and drawing from past studies. These rules are aimed at removing meaningless lines (such as boilerplate content) and documents. Specifically, we (1) detect lines containing “lorem ipsum” and filter them out if the resulting line is less than 5 characters; (2) detect lines containing “javascript” that also include “enable”, “disable” or “browser” and are under 200 characters, and filter them; (3) filter lines containing fewer than 10 words that include keywords like “Log in”, “sign-in”, “read more...”, or “items in cart.”; (4) filter documents if the ratio of uppercase words exceeds 40%; (5) filter lines that end with “...” if they constitute more than 30% of the entire document; (6) filter documents if the 

5 

ratio of non-alphabetic words surpasses 80%; (7) exclude documents with an average English word length outside the range of (3, 10); (8) discard documents that lack at least two common stop words such as “the”, “be” “to” “of” “and” “that” or “have”; (9) filter out documents if the ratio of ellipses (...) to words exceeds 0.5 (e.g., progress bars); (10) remove documents where 90% of lines start with bullet points; (11) filter documents including less than 200 characters after removing spaces and punctuation marks. 

These meticulously crafted rules enabled us to curate a high-quality mathematical corpus. They also facilitated the assignment of quality annotations to each document from Wikipedia and Common Crawl. These annotations provide researchers and developers with the flexibility to filter the data according to their criteria, catering to specific needs (as shown in Figure 12). This process resulted in filtering approximately 1,100 documents, removing 17 million tokens. 

## **3.3 Data Deduplication** 

Given that our corpus originates from diverse sources, it is inevitable that there will be repetitions both within and across these sources. Deduplication is vital for training efficiency and reducing data memorization, addressing both exact and near-duplicates [33]. We utilized the MinHash LSH algorithm [22] built on the implementation of text-dup [46] and Lee et al. [33], to process large-scale corpora efficiently. Specifically, our process involved splitting each document using whitespace and constructing 5-grams, applying the “sha1” hash function, and configuring 450 buckets with 20 minhashes each, totaling 9,000 minhashes per document, as per RefinedWeb’s guidelines [51]. 

During the deduplication process within each source, we encountered numerous exact and nearduplicate documents across various sources: 304 in arXiv, 623 in Common Crawl, 83,716 in Wikipedia, 783 in textbooks (primarily synthetic), and 144 duplicate questions in StackExchange. Despite finding many near-duplicates in ProofWiki, they were differentiated as unique lemmas, proofs, or definitions, leading us to retain these entries (cf. Table 13). Manual review revealed significant duplication in Wikipedia due to collecting multiple historical document versions and in StackExchange from reposts across different forums (e.g., Math and MathOverflow) for broader visibility (cf. Table 16). We provide near-duplicate examples from each data source in Table 1116. Cross-source deduplication revealed minimal overlap, with a single StackExchange question duplicated in Common Crawl, which was removed. This eliminated around 714 million tokens. 

Note that we also experimented with using suffix arrays [43] to eliminate exact match sequences within documents. However, it tended to remove common phrases like “Questions: ”. While it can effectively remove some templated content, it also disrupts the contextual integrity of our corpus. Consequently, we decided against employing this in order to preserve the context of our data. 

## **3.4 Data Contamination Detection** 

As pre-training corpora grow, encountering data contamination becomes inevitable, where evaluation examples are found in the training set. Traditionally, post-hoc analysis, employing n-gram overlap, assesses contamination levels (e.g., GPT-2 [53], GPT-3 [10], FLAN [63], LLaMA-2 [62]). We advocate for early contamination detection during dataset creation to prevent irreversible damage as delaying exacerbates issues (c.f., previous study [30]). Here, we utilize popular mathematical reasoning benchmarks, namely GSM8K [16], MATH [26], MMLU-STEM [25], AGIEval-SATMATH [70], MathQA [3] and AQuA [39] to detect data contamination. 

To detect data contamination, we aggregated questions and answers from benchmark tests into a reference set, considering only questions for MMLU, AGIEval, MathQA and AQuA due to its multiple-choice format. Intuitively, math problem solutions often involve diverse reasoning steps, making questions easier to detect for contamination in pretraining data due to their more fixed nature. We utilized line-level exact match detection, dividing documents into lines, hashing each with MD5 (taking the first 64 bits and the line itself to form sets), and applied this to both our corpus and the test sets. If a test set line and its hash match exactly with our dataset, it’s marked as contamination. 

Table 2: Benchmark test set occurrences in pre-training corpora, with numbers representing minimum occurrences, given potential undetected dupli- 

cates. 

|cates.||
|---|---|
|**Corpus**|**MATH**<br>**MMLU-STEM**|
|Ours<br>OpenWebMath|023<br>02<br>195<br>65|



6 

After our detection process, we found 23 questions from MATH and 2 from MMLU-STEM in our corpus (see Table 2), with no accompanying answers. No contamination was detected in other benchmarks. These duplicates mainly originated from StackExchange, Textbooks, and Common Crawl (see Table 17 and Table 18 for examples). Notably, questions from AMC mathematics competition books, also used in the MATH benchmark, were identified in Textbooks. We extended our analysis to OpenWebMath, uncovering more duplicate questions from MATH and MMLU (cf. Table 19), although many were repeats. This aligns with similar findings by Azerbayev et al. [5]. These instances highlight the importance of vigilance in creating pre-training corpora to avoid undermining downstream benchmarks. We removed all detected exact matches to mitigate data contamination, resulting in MATHPILE corpus. 

## **4 Data Analysis** 

## **4.1 Statistics** 

Table 3: The components and data statistics of MATHPILE . 

||Table 3: The components and data statistics of MATHPILE.|
|---|---|
|**Components**|**Size (MB)**<br>**# Documents**<br>**# Tokens**<br>**max(# Tokens)**<br>**min (# Tokens)**<br>**ave (# Tokens)**|
|Textbooks<br>Wikipedia<br>ProofWiki<br>CommonCrawl<br>StackExchange<br>arXiv|00644<br>003,979<br>0187,194,060<br>1,634,015<br>256<br>47,046<br>00274<br>022,795<br>0059,990,005<br>0109,282<br>056<br>02,632<br>00023<br>023,839<br>0007,608,526<br>0006,762<br>025<br>00319<br>02,560<br>075,142<br>0615,371,126<br>0367,558<br>057<br>08,189<br>01,331<br>433,751<br>0253,021,062<br>0125,475<br>028<br>00583<br>24,576<br>343,830<br>8,324,324,917<br>4,156,454<br>020<br>24,211|
|Total|29,408<br>903,336<br>9,447,509,696<br>-<br>-<br>10,458|



We present detailed statistical information for each component of MATHPILE in Table 3, such as the number of documents and the count of tokens. Following our meticulous and comprehensive data collection and processing process, we obtain 29GB of highquality and diverse math-centric corpus, encompassing around 9.5 billion tokens, from an initial volume of 2.2TB of raw data (cf. Figure 2). Compositionally, arXiv constitutes the largest portion of MATHPILE , while Textbooks represent the smallest share but are of exceptionally high quality. 

|10<br>5<br>10<br>4<br>10<br>3<br>% of Total Documents<br>a<br>102<br>103<br>Document<br>10<br>5<br>10<br>4<br>10<br>3<br>% of Total Documents<br>Stack|rXiv|CommonCrawl|ProofWiki|
|---|---|---|---|
|||||
||Exchange|Textbooks|Wikipedia|
|||||
||104<br>105<br>106<br>Length (Tokens)|102<br>103<br>104<br>105<br>106<br>Document Length (Tokens)|102<br>103<br>104<br>105<br>106<br>Document Length (Tokens)|



Figure 3: Document length distribution (log-scale). 

We analyze the document length 

(in terms of token numbers) and their respective proportions from each source within MATHPILE , which is visualized in Figure 3. Intuitively, if the data from each source contains a higher amount of near-duplicates or machine-generated content, the distribution of documents of similar lengths becomes more prevalent, leading to a less smooth distribution curve. Figure 3 shows that, thanks to our thorough and rigorous processing, the document length distribution in MATHPILE is relatively smooth across different sources. Note that ProofWiki, due to its fixed format of definitions, lemmas, and proofs, naturally contains shorter content, resulting in a distribution with many similar lengths. We can also observe that, on average, the documents from arXiv and Textbooks tend to be lengthier, while those from ProofWiki and StackExchange are generally shorter. 

## **4.2 Continual Pre-training Experiments** 

We chose Mistral-7B-v0.1 [28] (the state-of-the-art open-source model at the time) for continual pre-training. We segmented packed text into chunks with a window size of 4,096 and continued 

7 

pre-training for 3 epochs with a global batch size of 1024. We employ a cosine learning rate schedule with a maximum learning rate of 1e-5 and 1% warmup steps. All experiments were conducted on NVIDIA A100 8*80GB GPUs. For evaluation, we employ a range of benchmarks - GSM8K, MATH, MMLU-MATH, AGIEval-SAT-MATH, MathQA, AQuA - to assess varying levels of mathematical reasoning abilities, comparing all models using the same few-shot prompting with greedy decoding. 

## **The Effectiveness of MATH-** 

|**PILE**<br>We further pre-<br>trained<br>Mistral-7B-v0.1<br>on<br>several<br>subsets,<br>re-<br>spectively.<br>As shown in<br>Table 4, overall, continual<br>pre-training on the subsets<br>generally<br>enhances<br>per-<br>formance<br>across<br>diverse<br>math<br>benchmarks,<br>albeit<br>to varying degrees.<br>There<br>are exceptions, such as the<br>lack<br>of<br>improvement<br>on<br>GSM8K after training on<br>StackExchange; we suspect|Table 4: Results on each subset of MATHPILEand sampled OpenWeb-<br>Math. The numbers in parentheses represent the number of tokens<br>trained. **Bold**results denote improvements over the original Mistral.|Table 4: Results on each subset of MATHPILEand sampled OpenWeb-<br>Math. The numbers in parentheses represent the number of tokens<br>trained. **Bold**results denote improvements over the original Mistral.|
|---|---|---|
||**Models**|**GSM8K MATH**<br>**SAT-**<br>**MATH**<br>**MMLU-**<br>**Math**<br>**MathQA**<br>**AQuA**|
||Mistral-7B-v0.1|47.38<br>10.08<br>47.27<br>44.92<br>23.51<br>27.95|
||+ Textbooks (0.56B)<br>+ Wikipedia (0.18B)<br>+ StackExchange (0.87B)<br>+ Common Crawl (1.83B)<br>+ arXiv (0.38B)<br>+ Textbooks, Wikipeida, StackEx., CC (4B)|**48.97**<br>**12.10**<br>**56.36**<br>**48.93**<br>**30.38**<br>**33.07**<br>**49.96**<br>09.96<br>**53.63**<br>**47.16**<br>**28.97**<br>**35.43**<br>43.06<br>**11.66**<br>47.27<br>43.51<br>**27.67**<br>**30.70**<br>45.56<br>09.88<br>**50.45**<br>**45.17**<br>**25.79**<br>**31.88**<br>**47.91**<br>07.50<br>42.72<br>**46.34**<br>18.05<br>27.55<br>**49.88**<br>**11.70**<br>43.18<br>43.75<br>23.24<br>25.19|
||+ AMPS (1B)<br>+ DM-Mathematics (5B)<br>+ Sampled OpenWebMath (0.59B)|00.08<br>00.82<br>03.18<br>00.47<br>010.99<br>08.27<br>00.00<br>00.00<br>00.00<br>00.00<br>00.00<br>00.00<br>43.21<br>07.86<br>**47.72**<br>**47.52**<br>21.80<br>24.80|
||||



this is due to community users rarely asking basic arithmetic questions on StackExchange. Continual pre-training on arXiv leds to a slight performance boost on GSM8K and MMLU-MATH, but a degradation on MATH, SAT-MATH, and MathQA. We attribute this performance degradation to the disparity between the math knowledge present in arXiv papers and that required for the downstream benchmarks. We also conducted pre-training on a collection of Textbooks, Wikipedia, StackExchange, and CC. Experimental results indicate improved performance on GSM8K and MATH, but not on other benchmarks. Due to limited computational resources,[6] we did not extensively experiment with the entire dataset or combine data from MATHPILE ’s subsets and existing general corpora, leaving these valuable aspects for future work. Note that we also report some evaluation results on general language benchmarks provided in Appendix G. 

Furthermore, we also conducted continual pre-training on some existing corpora listed in Table 1 for comparison, including AMPS, DM-Mathematics and a random subset of OpenWebMath, cleansed of data leakage, in volumes approximately equal to that of Textbooks. Surprisingly, pre-training directly with these synthetic datasets degraded model performance. We attribute this to the narrow, monotonous structure of AMPS and DM-Mathematics problem sets, making them unsuitable for standalone pre-training; such datasets generally yield better results when combined with broader corpora for pre-training [67]. Additionally, the OpenWebMath subset produced even less improvement than the same or smaller scale subsets of MATHPILE , such as Textbooks and Wikipedia (cf. Table 4), likely due to a need for more tokens to show substantial gains. These results underscore the superior quality of our data. 

Table 5: Ablation study on data processing pipeline and LaTeX display issue resolution 

|**Models**|**Global Data**<br>**Processing**<br>**Fix Latex**<br>**Display Issue**<br>**GSM8K MATH**<br>**SAT-**<br>**MATH**<br>**MMLU-**<br>**MATH**<br>**MathQA**<br>**AQuA**|
|---|---|
|Mistral-v0.1-7B|-<br>-<br>47.38<br>10.08<br>47.27<br>44.92<br>23.51<br>27.95|
|+ Sampled raw Wikipedia (0.55B)<br>+ Full raw Wikipedia (2.18B)<br>+ Full cleaned but LaTeX issued Wikipedia (0.23B)|✗<br>✗<br>41.92<br>06.28<br>20.90<br>23.70<br>24.72<br>24.01<br>✗<br>✗<br>32.30<br>04.48<br>13.64<br>25.59<br>27.04<br>23.62<br>✓<br>✗<br>47.15<br>08.58<br>46.81<br>42.92<br>21.00<br>31.88|
|+ Full cleaned Wikipedia (0.18B)|✓<br>✓<br>**49.96**<br>09.96<br>**53.63**<br>**47.16**<br>**28.97**<br>**35.43**|



**The Effectiveness of Data Processing Pipeline** We utilized the Wikipedia subset as a testbed to evaluate our data processing pipeline. We distinguished between raw Wikipedia, which is collected but not globally processed, and cleaned Wikipedia, which has undergone global data processing. Additionally, we performed an ablation study on LaTeX display issues in Wikipedia (cf. Figure 11), attributed to HTML-to-text conversion tools, by comparing documents with problematic and correct LaTeX displays. Following previous settings, we executed continual pre-training on these datasets. 

6Pre-training 10 billion tokens for 1 epoch requires approximately 1,760 NVIDIA A100 GPU hours, making us keen to partner with well-resourced corporations to gain deeper insights in the future. 

8 

Results in Table 5 indicate that skipping our pipeline notably reduces Mistral’s mathematical reasoning abilities, unaffected by increased training size (i.e., 2.18B). Furthermore, correct LaTeX display in documents is vital for enhancing reasoning capabilities, as shown by the last two rows of Table 5. These findings underscore our pipeline’s effectiveness and shed light on the superior importance of data quality over quantity, even in the continual pre-training phase. 

## **5 Related Work** 

**Pre-training Corpora for Language Models** In language modeling, early models like GPT [52] and BERT [17] are trained on resources such as Books [73] and Wikipedia. Later models like GPT2 [53] and T5 [55] expand training to include web data from Reddit (WebText) and Common Crawl (C4). GPT-3 [10] enlarges its corpus to 300 billion tokens, combining Common Crawl, WebText, Books, and Wikipedia. Pile [20] introduces a diverse collection of 22 datasets for large-scale pretraining. The Gopher project [54] compiles a 10.5TB corpus, and PaLM [14] is built from a 780 billion-token corpus, both closed-source. BLOOM [57] uses the ROOTS dataset [32] for multilingual pre-training. The Stack Kocetkov et al. [30] provides a 3.1 TB code dataset. LLaMA [61] utilizes various data sources but doesn’t release its corpus, unlike RedPajama [59] and its de-duplicated version SlimPajama [12]. RefinedWeb shows web-only corpora can rival curated ones [51]. Recent models like GPT-4 [48], Mistral-7B [28] and the lastest Gemini [58] have refrained from opensourcing data. Constructing diverse, high-quality pre-training corpora is crucial for narrowing the performance gap with closed-source models, reflecting our work’s aim. 

**Pre-training Benchmarks and Corpora for Mathematical Reasoning** The challenge of endowing models with human-like mathematical reasoning has attracted significant interest from the machine learning and natural language processing communities. To evaluate models’ mathematical capabilities, several benchmark datasets have been developed, including AQuA [38], DM-Mathematics [56], SVAMP [50], GSM8K [16], and MATH [26], which cover a range of complexities from basic arithmetic to competition-level mathematics. Additionally, benchmarks like NaturalProofs [65] focus on theorem-proving capabilities, while the STEM subset of MMLU [25] evaluates understanding across multiple tasks in science, technology, engineering, and mathematics. To improve models’ mathematical reasoning, pre-training corpora like AMPS [26] (despite a large-scale synthetic exercise set), ProofPile [4], and OpenWebMath [49] have been introduced, targeting various levels of mathematical problem-solving and theorem proving. Unlike Google’s Minerva [35] and OpenAI’s MathMix [37], which are not public, our work focuses on creating a high-quality and diverse mathematical corpus from diverse sources to fill existing gaps. 

## **6 Conclusion and Limitations** 

In this work, we present MATHPILE , a specialized corpus centered around mathematics, characterized by its diversity and high quality. Throughout its development, we meticulously source and gather data, applying a rigorous and math-specific pipeline. This pipeline encompasses various stages such as preprocessing, prefiltering, language identification, cleaning and filtering, and deduplication, all aimed at maintaining the high quality of the corpus. We also conduct data contamination detection to remove duplicates from popular mathematical reasoning benchmark test sets, crucial for ensuring their integrity and effectiveness, an aspect often overlooked in other similar works. We aim for our MATHPILE to enhance mathematical reasoning in language models, whether used alone or in conjunction with other datasets, to promote broader applications. 

This dataset also has some limitations. Many detailed decisions in its creation were made empirically, which may not always be optimal, and verifying decisions directly can be challenging. Moreover, the data scale is insufficient for training extra-large models; subsets like the common crawl could be expanded. Furthermore, the dataset is focused primarily on English, highlighting the need to construct high-quality datasets for other languages. Future research could also explore data mixing [40] and model-based pre-training corpus refinement [68, 72] to enhance dataset quality and model performance. 

9 

## **Acknowledgments and Disclosure of Funding** 

This work was partially funded by the National Natural Science Foundation of China (62476168), Shanghai Artificial Intelligence Laboratory. 

## **References** 

- [1] Miltiadis Allamanis. The adverse effects of code duplication in machine learning models of code. In _Proceedings of the 2019 ACM SIGPLAN International Symposium on New Ideas, New Paradigms, and Reflections on Programming and Software_ , Onward! 2019, pp. 143–153, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450369954. doi: 10.1145/3359591.3359735. URL https://doi.org/10.1145/3359591.3359735. 

- [2] AllenAI. allenai/dolma · datasets at hugging face. https://huggingface.co/datasets/ allenai/dolma, Aug 2023. 

- [3] Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)_ , pp. 2357–2367. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1245. URL https://doi.org/10.18653/v1/n19-1245. 

- [4] Zhangir Azerbayev, Bartosz Piotrowski, Hailey Schoelkopf, Edward W. Ayers, Dragomir Radev, and Jeremy Avigad. Proofnet: Autoformalizing and formally proving undergraduatelevel mathematics. _CoRR_ , abs/2302.12433, 2023. doi: 10.48550/ARXIV.2302.12433. URL https://doi.org/10.48550/arXiv.2302.12433. 

- [5] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. _CoRR_ , abs/2310.10631, 2023. doi: 10.48550/ARXIV.2310.10631. URL https://doi.org/10.48550/arXiv.2310.10631. 

- [6] Jack Bandy and Nicholas Vincent. Addressing "documentation debt" in machine learning research: A retrospective datasheet for bookcorpus. _CoRR_ , abs/2105.05241, 2021. URL https://arxiv.org/abs/2105.05241. 

- [7] Emily M. Bender and Batya Friedman. Data statements for natural language processing: Toward mitigating system bias and enabling better science. _Transactions of the Association for Computational Linguistics_ , 6:587–604, 2018. doi: 10.1162/tacl_a_00041. URL https: //aclanthology.org/Q18-1041. 

- [8] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pp. 7432–7439, 2020. 

- [9] Sidney Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, Usvsn Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. GPTNeoX-20B: An open-source autoregressive language model. In _Proceedings of BigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models_ , pp. 95–136, virtual+Dublin, May 2022. Association for Computational Linguistics. doi: 10.18653/ v1/2022.bigscience-1.9. URL https://aclanthology.org/2022.bigscience-1.9. 

- [10] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot 

10 

learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html. 

- [11] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott M. Lundberg, Harsha Nori, Hamid Palangi, Marco Túlio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments with GPT-4. _CoRR_ , abs/2303.12712, 2023. doi: 10.48550/ARXIV.2303.12712. URL https: //doi.org/10.48550/arXiv.2303.12712. 

- [12] Cerebras. Slimpajama: A 627b token, cleaned and deduplicated version of redpajama - cerebras. http://tinyurl.com/slimpajama, Jun 2023. 

- [13] Ethan Chern, Haoyang Zou, Xuefeng Li, Jiewen Hu, Kehua Feng, Junlong Li, and Pengfei Liu. Generative ai for math: Abel. https://github.com/GAIR-NLP/abel, 2023. 

- [14] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways. _J. Mach. Learn. Res._ , 24:240:1–240:113, 2023. URL http://jmlr.org/papers/v24/22-1144.html. 

- [15] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018. 

- [16] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _CoRR_ , abs/2110.14168, 2021. URL https://arxiv.org/abs/2110.14168. 

- [17] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423. 

- [18] Jesse Dodge, Maarten Sap, Ana Marasovi´c, William Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, and Matt Gardner. Documenting large webtext corpora: A case study on the colossal clean crawled corpus. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pp. 1286–1305, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021. emnlp-main.98. URL https://aclanthology.org/2021.emnlp-main.98. 

- [19] Yanai Elazar, Akshita Bhagia, Ian Magnusson, Abhilasha Ravichander, Dustin Schwenk, Alane Suhr, Pete Walsh, Dirk Groeneveld, Luca Soldaini, Sameer Singh, Hanna Hajishirzi, Noah A. Smith, and Jesse Dodge. What’s in my big data? _CoRR_ , abs/2310.20707, 2023. doi: 10.48550/ ARXIV.2310.20707. URL https://doi.org/10.48550/arXiv.2310.20707. 

11 

- [20] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gb dataset of diverse text for language modeling. _CoRR_ , abs/2101.00027, 2021. URL https://arxiv.org/abs/2101.00027. 

- [21] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford. Datasheets for datasets. _Commun. ACM_ , 64(12): 86–92, nov 2021. ISSN 0001-0782. doi: 10.1145/3458723. URL https://doi.org/10.1145/ 3458723. 

- [22] Aristides Gionis, Piotr Indyk, and Rajeev Motwani. Similarity search in high dimensions via hashing. In Malcolm P. Atkinson, Maria E. Orlowska, Patrick Valduriez, Stanley B. Zdonik, and Michael L. Brodie (eds.), _VLDB’99, Proceedings of 25th International Conference on Very Large Data Bases, September 7-10, 1999, Edinburgh, Scotland, UK_ , pp. 518–529. Morgan Kaufmann, 1999. URL http://www.vldb.org/conf/1999/P49.pdf. 

- [23] Edouard Grave, Piotr Bojanowski, Prakhar Gupta, Armand Joulin, and Tomas Mikolov. Learning word vectors for 157 languages. In Nicoletta Calzolari, Khalid Choukri, Christopher Cieri, Thierry Declerck, Sara Goggi, Koiti Hasida, Hitoshi Isahara, Bente Maegaard, Joseph Mariani, Hélène Mazo, Asuncion Moreno, Jan Odijk, Stelios Piperidis, and Takenobu Tokunaga (eds.), _Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)_ , Miyazaki, Japan, May 2018. European Language Resources Association (ELRA). URL https://aclanthology.org/L18-1550. 

- [24] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks are all you need. _CoRR_ , abs/2306.11644, 2023. doi: 10.48550/ARXIV.2306.11644. URL https://doi.org/10.48550/arXiv.2306. 11644. 

- [25] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ . OpenReview.net, 2021. URL https://openreview.net/forum?id=d7KBjmI3GmQ. 

- [26] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ , 2021. URL https://datasets-benchmarks-proceedings.neurips. cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html. 

- [27] Danny Hernandez, Tom B. Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El Showk, Nelson Elhage, Zac Hatfield-Dodds, Tom Henighan, Tristan Hume, Scott Johnston, Benjamin Mann, Chris Olah, Catherine Olsson, Dario Amodei, Nicholas Joseph, Jared Kaplan, and Sam McCandlish. Scaling laws and interpretability of learning from repeated data. _CoRR_ , abs/2205.10487, 2022. doi: 10.48550/ARXIV.2205.10487. URL https://doi.org/10.48550/arXiv.2205.10487. 

- [28] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. _CoRR_ , abs/2310.06825, 2023. doi: 10.48550/ARXIV.2310.06825. URL https://doi.org/10.48550/arXiv.2310. 06825. 

- [29] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. In Mirella Lapata, Phil Blunsom, and Alexander Koller (eds.), _Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers_ , pp. 427–431, Valencia, Spain, April 2017. Association for Computational Linguistics. URL https://aclanthology.org/E17-2068. 

12 

- [30] Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries. The stack: 3 TB of permissively licensed source code. _CoRR_ , abs/2211.15533, 2022. doi: 10.48550/ARXIV.2211.15533. URL https://doi.org/ 10.48550/arXiv.2211.15533. 

- [31] Julia Kreutzer, Isaac Caswell, Lisa Wang, Ahsan Wahab, Daan van Esch, Nasanbayar UlziiOrshikh, Allahsera Tapo, Nishant Subramani, Artem Sokolov, Claytone Sikasote, Monang Setyawan, Supheakmungkol Sarin, Sokhar Samb, Benoît Sagot, Clara Rivera, Annette Rios, Isabel Papadimitriou, Salomey Osei, Pedro Ortiz Suarez, Iroro Orife, Kelechi Ogueji, Andre Niyongabo Rubungo, Toan Q. Nguyen, Mathias Müller, André Müller, Shamsuddeen Hassan Muhammad, Nanda Muhammad, Ayanda Mnyakeni, Jamshidbek Mirzakhalov, Tapiwanashe Matangira, Colin Leong, Nze Lawson, Sneha Kudugunta, Yacine Jernite, Mathias Jenny, Orhan Firat, Bonaventure F. P. Dossou, Sakhile Dlamini, Nisansa de Silva, Sakine Çabuk Ballı, Stella Biderman, Alessia Battisti, Ahmed Baruwa, Ankur Bapna, Pallavi Baljekar, Israel Abebe Azime, Ayodele Awokoya, Duygu Ataman, Orevaoghene Ahia, Oghenefego Ahia, Sweta Agrawal, and Mofetoluwa Adeyemi. Quality at a glance: An audit of web-crawled multilingual datasets. _Transactions of the Association for Computational Linguistics_ , 10:50–72, 2022. doi: 10.1162/tacl_a_00447. URL https://aclanthology.org/2022.tacl-1.4. 

- [32] Hugo Laurençon, Lucile Saulnier, Thomas Wang, Christopher Akiki, Albert Villanova del Moral, Teven Le Scao, Leandro von Werra, Chenghao Mou, Eduardo González Ponferrada, Huu Nguyen, Jörg Frohberg, Mario Sasko, Quentin Lhoest, Angelina McMillan-Major, Gérard Dupont, Stella Biderman, Anna Rogers, Loubna Ben Allal, Francesco De Toni, Giada Pistilli, Olivier Nguyen, Somaieh Nikpoor, Maraim Masoud, Pierre Colombo, Javier de la Rosa, Paulo Villegas, Tristan Thrush, Shayne Longpre, Sebastian Nagel, Leon Weber, Manuel Muñoz, Jian Zhu, Daniel van Strien, Zaid Alyafeai, Khalid Almubarak, Minh Chien Vu, Itziar Gonzalez-Dios, Aitor Soroa, Kyle Lo, Manan Dey, Pedro Ortiz Suarez, Aaron Gokaslan, Shamik Bose, David Ifeoluwa Adelani, Long Phan, Hieu Tran, Ian Yu, Suhas Pai, Jenny Chim, Violette Lepercq, Suzana Ilic, Margaret Mitchell, Alexandra Sasha Luccioni, and Yacine Jernite. The bigscience ROOTS corpus: A 1.6tb composite multilingual dataset. In _NeurIPS_ , 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ ce9e92e3de2372a4b93353eb7f3dc0bd-Abstract-Datasets_and_Benchmarks.html. 

- [33] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 8424–8445, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.577. URL https://aclanthology.org/2022.acl-long.577. 

- [34] Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 6b493230205f780e1bc26945df7481e5-Abstract.html. 

- [35] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In _NeurIPS_ , 2022. URL http://papers.nips.cc/paper_files/ paper/2022/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html. 

- [36] Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need II: phi-1.5 technical report. _CoRR_ , abs/2309.05463, 2023. doi: 10.48550/ARXIV.2309.05463. URL https://doi.org/10.48550/arXiv.2309.05463. 

- [37] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. 

13 

_CoRR_ , abs/2305.20050, 2023. doi: 10.48550/ARXIV.2305.20050. URL https://doi.org/ 10.48550/arXiv.2305.20050. 

- [38] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 158–167, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1015. URL https://aclanthology.org/P17-1015. 

- [39] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Regina Barzilay and Min-Yen Kan (eds.), _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 158–167, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1015. URL https: //aclanthology.org/P17-1015. 

- [40] Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, and Min Lin. Regmix: Data mixture as regression for language model pre-training. _CoRR_ , abs/2407.01492, 2024. doi: 10.48550/ARXIV.2407.01492. URL https://doi.org/ 10.48550/arXiv.2407.01492. 

- [41] Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, Denny Zhou, Jason Wei, Kevin Robinson, David Mimno, and Daphne Ippolito. A pretrainer’s guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity. _CoRR_ , abs/2305.13169, 2023. doi: 10.48550/ARXIV.2305.13169. URL https://doi.org/ 10.48550/arXiv.2305.13169. 

- [42] Alexandra Luccioni and Joseph Viviano. What’s in the box? an analysis of undesirable content in the Common Crawl corpus. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)_ , pp. 182–189, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-short.24. URL https: //aclanthology.org/2021.acl-short.24. 

- [43] Udi Manber and Eugene W. Myers. Suffix arrays: A new method for on-line string searches. _SIAM J. Comput._ , 22(5):935–948, 1993. doi: 10.1137/0222058. URL https://doi.org/10. 1137/0222058. 

- [44] Angelina McMillan-Major, Emily M. Bender, and Batya Friedman. Data statements: From technical concept to community practice. _ACM J. Responsib. Comput._ , may 2023. doi: 10.1145/ 3594737. URL https://doi.org/10.1145/3594737. Just Accepted. 

- [45] Margaret Mitchell, Alexandra Sasha Luccioni, Nathan Lambert, Marissa Gerchick, Angelina McMillan-Major, Ezinwanne Ozoani, Nazneen Rajani, Tristan Thrush, Yacine Jernite, and Douwe Kiela. Measuring data. _CoRR_ , abs/2212.05129, 2022. doi: 10.48550/ARXIV.2212. 05129. URL https://doi.org/10.48550/arXiv.2212.05129. 

- [46] Chenghao Mou, Chris Ha, Kenneth Enevoldsen, and Peiyuan Liu. Chenghaomou/text-dedup: Reference snapshot, September 2023. URL https://doi.org/10.5281/zenodo.8364980. 

- [47] OpenAI. Introducing chatgpt. https://openai.com/blog/chatgpt, Nov 2022. 

- [48] OpenAI. GPT-4 technical report. _CoRR_ , abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303. 08774. URL https://doi.org/10.48550/arXiv.2303.08774. 

- [49] Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text. _CoRR_ , abs/2310.06786, 2023. doi: 10.48550/ARXIV.2310.06786. URL https://doi.org/10.48550/arXiv.2310.06786. 

- [50] Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 2080–2094, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/ 2021.naacl-main.168. URL https://aclanthology.org/2021.naacl-main.168. 

14 

- [51] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon LLM: outperforming curated corpora with web data, and web data only. _CoRR_ , abs/2306.01116, 2023. doi: 10.48550/ARXIV.2306.01116. URL https://doi.org/ 10.48550/arXiv.2306.01116. 

- [52] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 

- [53] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. 

- [54] Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, H. Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew J. Johnson, Blake A. Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher. _CoRR_ , abs/2112.11446, 2021. URL https://arxiv.org/abs/2112.11446. 

- [55] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _J. Mach. Learn. Res._ , 21:140:1–140:67, 2020. URL http://jmlr. org/papers/v21/20-074.html. 

- [56] David Saxton, Edward Grefenstette, Felix Hill, and Pushmeet Kohli. Analysing mathematical reasoning abilities of neural models. In _7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019_ . OpenReview.net, 2019. URL https://openreview.net/forum?id=H1gR5iR5FX. 

- [57] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, and et al. BLOOM: A 176b-parameter open-access multilingual language model. _CoRR_ , abs/2211.05100, 2022. doi: 10.48550/ARXIV.2211.05100. URL https://doi.org/10.48550/arXiv.2211. 05100. 

- [58] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: A family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- [59] Together. Redpajama, a project to create leading open-source models, starts by reproducing llama training dataset of over 1.2 trillion tokens. https://www.together.ai/blog/ redpajama, Apr 2023. 

15 

- [60] Together. Redpajama-data-v2: An open dataset with 30 trillion tokens for training large language models. https://www.together.ai/blog/redpajama-data-v2, Oct 2023. 

- [61] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. _CoRR_ , abs/2302.13971, 2023. doi: 10.48550/ARXIV.2302.13971. URL https://doi.org/10.48550/arXiv.2302.13971. 

- [62] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. _CoRR_ , abs/2307.09288, 2023. doi: 10.48550/ARXIV.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288. 

- [63] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ . OpenReview.net, 2022. URL https://openreview.net/forum?id=gEZrGCozdqR. 

- [64] Johannes Welbl, Nelson F Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. _arXiv preprint arXiv:1707.06209_ , 2017. 

- [65] Sean Welleck, Jiacheng Liu, Ronan Le Bras, Hanna Hajishirzi, Yejin Choi, and Kyunghyun Cho. Naturalproofs: Mathematical theorem proving in natural language. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ , 2021. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/ hash/d9d4f495e875a2e075a1a4a6e1b9770f-Abstract-round1.html. 

- [66] Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave. CCNet: Extracting high quality monolingual datasets from web crawl data. In _Proceedings of the 12th Language Resources and Evaluation Conference_ , pp. 4003–4012, Marseille, France, May 2020. European Language Resources Association. ISBN 979-10-95546-34-4. URL https://aclanthology.org/2020.lrec-1. 494. 

- [67] Yiheng Xu, Hongjin SU, Chen Xing, Boyu Mi, Qian Liu, Weijia Shi, Binyuan Hui, Fan Zhou, Yitao Liu, Tianbao Xie, Zhoujun Cheng, Siheng Zhao, Lingpeng Kong, Bailin Wang, Caiming Xiong, and Tao Yu. Lemur: Harmonizing natural language and code for language agents. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id=hNhwSmtXRh. 

- [68] Zichun Yu, Spandan Das, and Chenyan Xiong. Mates: Model-aware data selection for efficient pretraining with data influence models. _arXiv preprint arXiv:2406.06046_ , 2024. 

- [69] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? _arXiv preprint arXiv:1905.07830_ , 2019. 

- [70] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. _CoRR_ , abs/2304.06364, 2023. doi: 10.48550/ARXIV.2304.06364. URL https://doi.org/10.48550/arXiv.2304.06364. 

16 

- [71] Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. LIMA: less is more for alignment. _CoRR_ , abs/2305.11206, 2023. doi: 10.48550/ARXIV. 2305.11206. URL https://doi.org/10.48550/arXiv.2305.11206. 

- [72] Fan Zhou, Zengzhi Wang, Qian Liu, Junlong Li, and Pengfei Liu. Programming every example: Lifting pre-training data quality like experts at scale. _arXiv preprint arXiv:2409.17115_ , 2024. 

- [73] Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In _2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015_ , pp. 19–27. IEEE Computer Society, 2015. doi: 10.1109/ICCV.2015.11. URL https://doi.org/10.1109/ICCV.2015.11. 

17 

## **Checklist** 

1. For all authors... 

   - (a) Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? [Yes] 

   - (b) Did you describe the limitations of your work? [Yes] 

   - (c) Did you discuss any potential negative societal impacts of your work? [No] 

   - (d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes] 

2. If you are including theoretical results... 

   - (a) Did you state the full set of assumptions of all theoretical results? [N/A] 

   - (b) Did you include complete proofs of all theoretical results? [N/A] 

3. If you ran experiments (e.g. for benchmarks)... 

   - (a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] 

   - (b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes] 

   - (c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [No] 

   - (d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] 

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets... 

   - (a) If your work uses existing assets, did you cite the creators? [Yes] 

   - (b) Did you mention the license of the assets? [Yes] 

   - (c) Did you include any new assets either in the supplemental material or as a URL? [No] 

   - (d) Did you discuss whether and how consent was obtained from people whose data you’re using/curating? [Yes] 

   - (e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [Yes] 

5. If you used crowdsourcing or conducted research with human subjects... 

   - (a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A] 

   - (b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A] 

   - (c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A] 

18 

## **Appendix** 

|**A**|**MATHPILE Datasheet**|**20**|
|---|---|---|
|**B**|**Ethics Statement**|**24**|
|**C **|**Examples of MATHPILE**|**24**|
|**D **|**Details for Corpus Collection and Processing**|**31**|
|**E**|**Example for Quality Annotation**|**33**|
|**F**|**Examples of Duplicates Encountered in the Deduplication Process**|**33**|
|**G **|**Evaluation of Continal Pre-trained Models on General Langauge Benchmarks**|**43**|



19 

## **A MATHPILE Datasheet** 

|**MOTIVATION**|**MOTIVATION**|
|---|---|
|||
|**For what purpose was**<br>**the dataset created?**|Developed in a context where datasets like Google’s Minerva and<br>OpenAI’s MathMix are not open-sourced, MATHPILE aims to<br>counter this trend by enriching the open-source community and<br>enhancing mathematical language modeling with its (relatively)<br>large-scale, math-centric, diverse, high-quality dataset. It can be<br>used on its own or cooperated with general domain corpora like<br>books, and Github code, to improve the reasoning abilities of<br>language models.|
|||
|**Who**<br>**created**<br>**the**<br>**dataset and on behalf**<br>**of which entity?**|MATHPILEwas created by the authors of this work.|
|||
|**Who funded the cre-**<br>**ation of the dataset?**|The creation of MATHPILEwas funded by GAIR Lab, SJTU.|
|||
|**Any other comment?**|None.|
|**COMPOSITION**||
|||
|**What do the instances**<br>**that**<br>**comprise**<br>**the**<br>**dataset represent?**|MATHPILEis comprised of text-only documents, encompassing<br>a broad range of sources. These include academic papers from<br>arXiv, educational materials such as textbooks and lecture notes,<br>defnitions, theorems and their proofs, informative articles from<br>Wikipedia, interactive Q&A content from StackExchange com-<br>munity users, and webpages sourced from Common Crawl. All<br>these instances are math-focused.|
|||
|**How many instances**<br>**are there in total?**|MATHPILEcontains about 903 thousand of documents, or around<br>9.5 billion tokens.|
|||
|**Does the dataset con-**<br>**tain**<br>**all**<br>**possible**<br>**in-**<br>**stances or is it a sam-**<br>**ple (not necessarily ran-**<br>**dom) of instances from**<br>**a larger set?**|MATHPILEis curated from a diverse array of sources, including<br>arXiv, Textbooks, Wikipedia, StackExchange, ProofWiki, and<br>Common Crawl. However, it doesn’t encompass all instances<br>from these sources. We have implemented a rigorous data process-<br>ing pipeline, which involves steps like preprocessing, prefltering,<br>language identifcation, cleaning, fltering, and deduplication.<br>This meticulous approach is taken to guarantee the high quality<br>of the content within MATHPILE.|
|||
|**What data does each in-**<br>**stance consist of?**|Each instance in MATHPILEis a text-only document, uniquely<br>identifed by its source, labeled underSubset. These instances<br>are enriched with metadata, such as the score from language iden-<br>tifcation, the ratio of symbols to words, and their respective fle<br>paths. Note that instances from the StackExchange are composed<br>of a question and its accompanying answers, each with their<br>own set of meta data, including community users. To illustrate<br>them, we provide specifc examples for each source, ranging from<br>Figure 4 to Figure 10.|
|||
|**Is there a label or target**<br>**associated with each in-**<br>**stance?**|No.|
|||
|**Is any information miss-**<br>**ing from individual in-**<br>**stances?**|No.|



20 

|**Are**<br>**relationships**<br>**between individual in-**<br>**stances made explicit?**|No.|
|---|---|
|**Are**<br>**there**<br>**recom-**<br>**mended data splits?**<br>No.||
|||
|**Are there any errors,**<br>**sources of noise,**<br>**or**<br>**redundancies**<br>**in**<br>**the**<br>**dataset?**|Despite our rigorous efforts in cleaning, fltering out low-quality<br>content, and deduplicating documents, it’s important to acknowl-<br>edge that a small fraction of documents in MATHPILEmight still<br>fall short of our quality standards, particularly those sourced from<br>web pages.|
|||
|**Is**<br>**the**<br>**dataset**<br>**self-**<br>**contained, or does it**<br>**link to or otherwise rely**<br>**on external resources?**|Yes, MATHPILEis self-contained.|
|||
|**Does the dataset con-**<br>**tain data that might**<br>**be considered confden-**<br>**tial?**|No.|
|||
|**Does the dataset con-**<br>**tain data that, if viewed**<br>**directly, might be offen-**<br>**sive, insulting, threaten-**<br>**ing, or might otherwise**<br>**cause anxiety?**|We do not expect offensive content despite our signifcant efforts<br>in cleaning and fltering. But, we can not fully guarantee this.|
|**COLLECTION**||
|||
|**How was the data as-**<br>**sociated with each in-**<br>**stance acquired?**|Our data is primarily sourced from the arXiv website and the In-<br>ternet Archive. The CommonCrawl data originates from SlimPa-<br>jama. The textbooks included are manually collected, with quality<br>checks performed on publicly available textbooks from various<br>internet sources.|
|||
|**What mechanisms or**<br>**procedures were used to**<br>**collect the data?**|Refer to § 2 for details on how they collect data.|
|||
|**If the dataset is a sam-**<br>**ple from a larger set,**<br>**what was the sampling**<br>**strategy?**|We strive to use the most recent data dumps available and then<br>selectively choose high-quality documents that are closely related<br>to mathematics.|
|||
|**Who was involved in the**<br>**data collection process**<br>**and how were they com-**<br>**pensated?**|Authors from this paper were involved in collecting it and pro-<br>cessing it.|
|||
|**Over what timeframe**<br>**was the data collected?**|MATHPILEencompasses documents created between 2007 and<br>August 2023. Note that some documents and textbooks included<br>may be created in the previous century.|
|||
|**Were any ethical review**<br>**processes conducted?**|No.|



**PREPROCESSING** 

21 

|**Was any preprocessing/-**<br>**cleaning/labeling of the**<br>**data done?**|Yes, during our data collection phase, we conducted extensive<br>fltering and cleansing procedures, detailed in § 2. After the<br>completion of data collection, we conducted further steps in-<br>cluding language identifcation, additional cleaning and fltering,<br>deduplication, and leakage detection in benchmark datasets. Sub-<br>sequently, we removed any contaminated examples identifed<br>through this process. See § 3 for details.|
|---|---|
|||
|**Was the “raw” data**<br>**saved in addition to**<br>**the**<br>**preprocessed/-**<br>**cleaned/labeled data?**|Yes.|
|||
|**Is the software that**<br>**was used to prepro-**<br>**cess/clean/label the data**<br>**available?**|Yes,<br>scripts<br>are<br>open-sourced<br>at<br>https://github.com/<br>GAIR-NLP/MathPile/tree/main/src|
|**USES**||
|||
|**Has the dataset been**<br>**used for any tasks al-**<br>**ready?**|Yes, this data has been used to develop mathematical language<br>models.|
|||
|**Is there a repository**<br>**that links to any or all**<br>**papers or systems that**<br>**use the dataset?**|No. This dataset is currently utilized in the following research<br>papers: (1) JiuZhang 3.0: Effciently Improving Mathematical<br>Reasoning by Training Small Data Synthesis Models. (2) Task<br>Oriented In-Domain Data Augmentation. (3) Great Memory,<br>Shallow Reasoning: Limits of_k_NN-LMs. (4) BAM! Just Like<br>That: Simple and Effcient Parameter Upcycling for Mixture of<br>Experts. (5) SciDFM: A Large Language Model with Mixture-<br>of-Experts for Science. (6) MIND: Math Informed syNthetic<br>Dialogues for Pretraining LLMs and so on.|
|||
|**What**<br>**(other)**<br>**tasks**<br>**could the dataset be**<br>**used for?**|MATHPILEwas developed to enhance language modeling, offer-<br>ing signifcant benefts for a variety of mathematical reasoning<br>tasks.|
|||
|**Is there anything about**<br>**the**<br>**composition**<br>**of**<br>**the**<br>**dataset**<br>**or**<br>**the**<br>**way it was collected**<br>**and**<br>**preprocessed/-**<br>**cleaned/labeled**<br>**that**<br>**might**<br>**impact**<br>**future**<br>**uses?**|Our cleaning and fltering processes, while thorough, may not<br>be entirely optimal, potentially leading to the exclusion of some<br>valuable documents. Additionally, MATHPILE is specifcally<br>tailored for English, which limits its applicability in multilingual<br>contexts.|
|||
|**Are**<br>**there**<br>**tasks**<br>**for**<br>**which**<br>**the**<br>**dataset**<br>**should not be used?**|Any tasks which may considered irresponsible or harmful.|
|**DISTRIBUTION**||
|||
|**Will the dataset be dis-**<br>**tributed to third par-**<br>**ties outside of the entity**<br>**on behalf of which the**<br>**dataset was created?**|Yes, MATHPILE has been made available through the Hug-<br>gingFace Hub (https://huggingface.co/datasets/GAIR/<br>MathPile).|
|||
|**How will the dataset**<br>**will be distributed?**|MATHPILEhas been made available through the HuggingFace<br>Hub (https://huggingface.co/datasets/GAIR/MathPile).|
|||
|**When will the dataset**<br>**be distributed?**|The MATHPILEwill be available after this paper is made public.|



22 

**Will the dataset be dis-** If the source data of MATHPILE is governed by a license more **tributed under a copy-** restrictive than CC BY-NC-SA 4.0, MATHPILE adheres to that **right or other intellec-** stricter licensing. In all other cases, it operates under the CC **tual property (IP) li-** BY-NC-SA 4.0 license. If any data owner objects to the use of **cense, and/or under ap-** their data, we are willing to take appropriate action immediately, **plicable terms of use** including removing the relevant data. **(ToU)? Have any third par-** Not to our knowledge. **ties imposed IP-based or other restrictions on the data associated with the instances? Do any export controls** Not to our knowledge. **or other regulatory restrictions apply to the dataset or to individual instances? MAINTENANCE Who will be support-** MATHPILE will be hosted on the HuggingFace Hub. **ing/hosting/maintaining the dataset? How can the owner/cu-** stefanpengfei@gmail.com zzwang.nlp@gmail.com **rator/manager of the dataset be contacted? Is there an erratum?** No. **Will the dataset be up-** Yes, it is currently a work in progress and updates are ongoing. **dated? If others want to ex-** No. **tend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** 

Table 6: Datasheet for MATHPILE , following Gebru et al. [21]. 

23 

## **B Ethics Statement** 

In the collection and creation of MATHPILE , we strictly adhered to all copyright and licensing requirements of the data sources. Specifically, we gathered a large amount of data from the internet, including mathematical textbooks, web pages, and community Q&A content, ensuring that the use of these data complies with the original licensing terms. Wikipedia, ProofWiki, and StackExchange are licensed under CC BY-SA (2.5, 3.0 or 4.0). Textbooks and arXiv are licensed under CC BY 4.0, CC BY-SA 4.0, CC BY-NC-SA 4.0 and others. Common Crawl follows the Common Crawl Foundation Terms of Use and C4 license. The final open-source MATHPILE dataset is released under the CC BY-NC-SA 4.0 license. If the source data’s license is more restrictive than CC BY-NC-SA 4.0, we adopt the stricter license. 

However, during the collection of some data, such as publicly available and open-source textbooks, we did not obtain explicit consent from each author. We recognize that this may involve potential copyright issues. Therefore, we have implemented the following measures to mitigate and manage these risks: 

1. **Strict Selection of Data Sources** : We prioritize selecting data sources that are clearly marked with open licenses or public domain status, avoiding the use of content explicitly marked as copyright-protected or prohibited from distribution. 

2. **Adherence to Fair Use Principles** : When using copyrighted and non-commercially licensed content, we adhere to the principles of fair use, aiming to promote scientific research and educational purposes rather than commercial purposes, thereby not affecting the market value of the original content. 

3. **Acceptance of Feedback from Users and Content Authors** : We welcome feedback from data users and authors at any time to request the removal or modification of their data. 

MATHPILE has been carefully curated and processed to minimize any potential ethical concerns. We also explicitly state that if any data owner objects to the use of their data, we are willing to take appropriate action immediately, including removing the relevant data. Through these measures, we strive to ensure the diversity and richness of the collected data while complying with relevant copyright and licensing regulations, thereby reducing potential legal risks. We bear full responsibility for any potential violations of rights or licensing issues that may arise from this dataset. 

## **C Examples of MATHPILE** 

We provide some illustrative examples from each source in MATHPILE , as shown in Figure 4 to Figure 10. 

24 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math_images/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math.pdf-0025-00.png)


**----- Start of picture text -----**<br>
A document from MATHPILE-CommonCrawl<br>Text:<br>Are there optimizers where it is possible to specify ordinal ranking of parameters?<br>Assume that  f is smooth ( n -th order differentiable in each of the parameters).<br>An approach I often use when applying unconstrained optimisation algorithms to constrained problems is to transform the parameter space such that the<br>constraints cannot be violated.<br>Of course this results in  θ 1 [∗] [≥] [θ] 2 [∗] [≥] [θ] 3 [∗] [which isn’t quite what you asked for.] [To get a strict ranking you’ll need to bump] [ x] [1] [−] [x] [2] 2 [and] [ x] [1] [−] [x] [2] 2 [−] [x] [2] 3<br>down at the last digit of precision.<br>thus spake a.k.thus spake a.k.<br>These variants of your constraints are linear, so provided that your function  f is well-behaved (smooth, easy to calculate, easy to compute derivatives, derivatives<br>are well-conditioned, etc.), any constrained optimization solver should be able to solve your problem without issue.<br>Not the answer you’re looking for? Browse other questions tagged optimization constrained-optimization or ask your own question.<br>Does the amount of correlation of model parameters matter for nonlinear optimizers?<br>Optimization of a blackbox function with an equality constraint?<br>...<br>Subset : CommonCrawl<br>meta :<br>language_detection_score: 0.8670,<br>char_num_after_normalized: 926,<br>contain_at_least_two_stop_words: True,<br>ellipsis_line_ratio: 0.0,<br>idx: 383668,<br>lines_start_with_bullet_point_ratio: 0.0,<br>mean_length_of_alpha_words: 5.0870,<br>non_alphabetical_char_ratio: 0.0,<br>symbols_to_words_ratio: 0.0,<br>uppercase_word_ratio: 0.0060,<br>...<br>**----- End of picture text -----**<br>


Figure 4: An example Common Crawl document in MATHPILE 

25 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math_images/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math.pdf-0026-00.png)


**----- Start of picture text -----**<br>
A document from MATHPILE -Wikipedia<br>Text:<br># Inner Automorphism<br>In abstract algebra, an  inner automorphism  is an automorphism of a group, ring, or algebra given by the conjugation action of a fixed element, called the<br>conjugating element . They can be realized via simple operations from within the group itself, hence the adjective "inner". These inner automorphisms form a<br>subgroup of the automorphism group, and the quotient of the automorphism group by this subgroup is defined as the outer automorphism group.<br>## Definition<br>If  G  is a group and  g  is an element of  G  (alternatively, if  G  is a ring, and  g  is a unit), then the function<br>φg  : G → G<br>φg ( x ) := g [−] [1] xg<br>is called  (right) conjugation by  g  (see also conjugacy class). This function is an endomorphism of  G : for all  x 1 , x 2 ∈ G ,<br>φg ( x 1 x 2) = g [−] [1] x 1 x 2 g = ( g [−] [1] x 1 g )( g [−] [1] x 2 g ) = φg ( x 1) φg ( x 2) ,<br>where the second equality is given by the insertion of the identity between  x 1 and  x 2. Furthermore, it has a left and right inverse, namely  φg− 1 . Thus,  φg is<br>bijective, and so an isomorphism of  G  with itself, i.e., an automorphism. An  inner automorphism  is any automorphism that arises from conjugation.[1]<br>When discussing right conjugation, the expression  g [−] [1] xg  is often denoted exponentially by  x [g] . This notation is used because composition of conjugations<br>satisfies the identity: ( x [g] [1] ) [g] [2] = x [g] [1] [g] [2] for all  g 1 , g 2 ∈ G . This shows that right conjugation gives a right action of  G  on itself.<br>### Inner and Outer Automorphism Groups<br>The composition of two inner automorphisms is again an inner automorphism, and with this operation, the collection of all inner automorphisms of  G  is a<br>group, the inner automorphism group of  G  denoted Inn( G ).<br>Inn( G ) is a normal subgroup of the full automorphism group Aut( G ) of  G . The outer automorphism group, Out( G ), is the quotient group<br>Out( G ) = Aut( G ) .<br>Inn( G )<br>The outer automorphism group measures, in a sense, how many automorphisms of  G  are not inner. Every non-inner automorphism yields a non-trivial element<br>of Out( G ), but different non-inner automorphisms may yield the same element of Out( G ).<br>Saying that conjugation of  x  by  a  leaves  x  unchanged is equivalent to saying that  a  and  x  commute:<br>a [−] [1] xa = x ⇐⇒ xa = ax.<br>Therefore, the existence and number of inner automorphisms that are not the identity mapping is a kind of measure of the failure of the commutative law in the<br>group (or ring).<br>An automorphism of a group  G  is inner if and only if it extends to every group containing  G .[2]<br>...<br>Subset : Wikipedia<br>meta :<br>language_detection_score: 0.7236,<br>char_num_after_normalized: 5794,<br>contain_at_least_two_stop_words: True,<br>ellipsis_line_ratio: 0.0,<br>lines_start_with_bullet_point_ratio: 0.0,<br>mean_length_of_alpha_words: 4.2245,<br>mimetype: text/html,<br>page_index: 48171,<br>page_path: A/Inner_automorphism,<br>page_title: Inner automorphism,<br>non_alphabetical_char_ratio: 0.1422,<br>symbols_to_words_ratio: 0.0,<br>uppercase_word_ratio: 0.0871,<br>...<br>**----- End of picture text -----**<br>


Figure 5: An example Wikipedia document in MATHPILE 

26 

## A document from MATHPILE-Textbooks 

**Text:** # LINEAR TORIC FIBRATIONS SANDRA DI ROCCO 

- ## INTRODUCTION TO TORIC FIBRATIONS 

- Definition 1.1. A toric fibration is a surjective flat map _f_ : _X → Y_ with connected fibres where (a) _X_ is a toric variety 

- (b) _Y_ is a normal algebraic variety 

- (c) dim( _Y_ ) _<_ dim( _X_ ). 

Remark 1.2. Observe that if _f_ : _X → Y_ is a toric fibration then _Y_ and a general fiber _F_ are also toric varieties. Moreover if _X_ is smooth, respectively Q-factorial then so is _Y_ and _F_ . 

- Combinatorial characterization. A toric fibration has the following combinatorial characterization (see [EW, Chapter VI] for further details). Let _X_ = _X_ Σ, where Σ _⊂ N[∼]_ = Z _[n]_ , be a toric variety of dimension _n_ and let _i_ : ∆ _�→ N_ a sublattice. Proposition 1.3. [EW] The inclusion _i_ induces a toric fibration if and only if: (a) ∆ is a primitive lattice, i.e. (∆ _⊗_ R) _∩ N_ = ∆. 

- (b) For every _σ ∈_ Σ( _n_ ) _, σ_ = _τ_ + _η_ , where _τ ∈_ ∆ and _η ∩_ ∆= _{_ 0 _}_ (i.e. Σ is a splitfan). 

We briefly outline the construction. The projection _π_ : _N → N/_ ∆ induces a map of fans Σ _→ π_ (Σ) and thus a map of toric varieties _f_ : _X → Y_ . The general fiber _F_ is a toric variety defined by the fan Σ _F_ = _{σ ∈_ Σ _∩_ ∆ _}_ . 

When the toric variety _X_ in a toric fibration is polarized by an ample line bundle _L_ we will call the pair ( _f_ : _X → Y, L_ ) a polarized toric fibration. Observe that the polarized toric varieties ( _X, L_ ) and � _F, L|F_ �, for a general fiber _F_ , define lattice polytopes _P_ ( _X,L_ ) _, P_ ( _F, L|F_ ). The polytope _P_ ( _X,L_ ) is in fact a "twisted sum" of a finite number of lattice polytopes fibering over _P_ ( _F, L|F_ ). Definition 1.4. Let _R_ 0 _, . . . , Rk ⊂_ ∆ be polytopes. Let _π_ : _M →_ Λ be a surjective map of lattices such that _π_ ( _Ri_ ) = _vi_ and the _v_ 0 _, · · · , vk_ are distinct vertices of Conv ( _v_ 0 _, . . . , vk_ ). We will call _a_ Cayley _π_ -twisted sum (or simply a Cayley sum) of _R_ 0 _, . . . , Rk_ a polytope which is affinely isomorphic to Conv ( _R_ 0 _, . . . , Rk_ ). We will denote it by: [ _R_ 0 _⋆. . . ⋆Rk_ ] _π_ 

If the polytopes _Ri_ are additionally normally equivalent, i.e. they define the same normal fan Σ _Y_ , we will denote the Cayley sum by: 

Cayley ( _R_ 0 _, . . . , Rk_ )( _π,Y_ ) _._ 

These are the polytopes that are associated to a polarized toric fibration. Consider a sublattice _i_ : ∆ _�→ N_ and the dual lattice surjection _π_ : _M →_ Λ. Proposition 1.5. [CDR08] The sublattice _i_ : ∆ _�→ N_ induces a polarized toric fibration ( _f_ : _X → Y, L_ ) if and only if _P_ ( _X,L_ ) = Cayley ( _R_ 0 _, . . . , Rk_ )( _π,Y_ ) for some normally equivalent polytopes _R_ 0 _, . . . , Rk_ . The polarized general fiber � _F, L|F_ � corresponds to the polarized toric variety associated to the polytope _P_ ( _F, L|F_ ) = Conv ( _v_ 0 _, . . . , vk_ ) and the polytopes _R_ 0 _, · · · , Rk_ define the embeddings of the invariant sections polarized by the restrictions of _L_ . Example 1.6. Consider the Hirzebruch surface F1 = _Blp_ �P[2][�] = P � _O_ P1 _⊕ O_ P1 (1)� polarized by the tautological line bundle _ξ_ = 2 _ϕ[∗]_[�] _O_ P2 (1)� _− E_ where _ϕ_ is the blow-up map and _E_ the exceptional divisor. The associated polytope is _P_ = Cayley (∆1 _,_ 2∆1). FIGURE 1. The Hirzebruch surface P � _O_ P1 _⊕ O_ P1 (1)� 

Example 1.7. More generally: - when _π_ ( _P_ ) = ∆ _t_ the polytope Cayley ( _R_ 0 _, . . . , Rk_ )( _π,Y_ ) defines the variety P ( _L_ 0 _⊕ . . . ⊕ Lk_ ), where the _Li_ are ample line bundles on the toric variety _Y_ , polarized by the tautological bundle _ξ_ . In particular _L|F_ = _O_ P _t_ (1). - When _π_ ( _P_ ) is a simplex (not necessarily smooth) Cayley ( _R_ 0 _, . . . , Rk_ )( _π,Y_ ) defines a Mori-type fibration. A fibration whose general fiber has Picard rank one. - When _π_ ( _P_ ) = _s_ ∆ _t_ then again the variety has the structure of a P _[t]_ -fibration whose general fiber _F_ is embedded via an _s_ -Veronese embedding: � _F, L|F_ � = �P _[t] , O_ P _t_ ( _s_ )�. 

For general Cayley sums, [ _R_ 0 _⋆. . . ⋆Rk_ ] _π_ , one has the following geometrical interpretation. Let ( _X, L_ ) be the associated polarized toric variety and let _Y_ be the toric variety defined by the Minkowski sum _R_ 0 + _. . ._ + _Rk_ . The fan defining _Y_ is a refinement of the normal fan of _Ri_ for _i_ = 0 _, . . . , k_ . Consider the associated birational maps _ϕi_ : _Y → Yi_ , where ( _Yi, Li_ ) is the polarized toric variety defined by the polytope _Ri_ . The line bundles _Hi_ = _ϕ[∗] i_[(] _[L][i]_[)][ are nef line bundles on] _[ Y]_[ .][Denote by the same symbol the maps of fans] _[ ϕ][i]_[:][Σ] _[Y][→]_[Σ] _[Y] i_[.][Define then the fan:] 

Σ _Z_ : � _ϕ[−] i_[1] � _σj_ � _× ηl,_ for all _σj ∈_ Σ _Yi , ηl ∈_ Σ∆� 

where Λ = Conv ( _v_ 0 _, . . . , vk_ ). It is a refinement of Σ _X_ and thus the defining variety _Z_ is birational to _X_ . Moreover it is a split fan and thus it defines a toric fibration _f_ : _Z → Y_ . The Cayley sum [ _R_ 0 _⋆. . . ⋆Rk_ ] _π_ is the polytope defined by the nef line bundle _ϕ[∗]_ ( _L_ ), and the polytopes _Ri_ are the polytopes defined by the nef line bundles _Hi_ on the invariant sections. 

Historical Remark. The definition of a Cayley polytope originated by what is "classically" referred to as the Cayley trick. We first recall the definition of Resultant and Discriminant. Let _f_ 1( _x_ ) _, . . . , fn_ ( _x_ ) be a system of _n_ polynomials in _n_ variables _x_ = ( _x_ 1 _, . . . , xn_ ) supported on _A ⊂_ Z _[n]_ . This means that _fi_ = Π _aj ∈Acj x[aj]_ . The resultant (of _A_ ), _RA_ � _cj_ ), is a polynomial in the coefficients _cj_ , which vanishes whenever the corresponding polynomials have a common zero. 

The discriminant of a finite subset _A,_ ∆ _A_ , is also a polynomial ∆ _A_ � _cj_ � in the variables _cj ∈ A_ which vanishes whenever the corresponding polynomial has a multiple root. 

Theorem 1.8. [GKZ][Cayley Trick] The A-resultant of the system _f_ 1 _, . . . , fn_ equals the Adiscriminant of the polynomial: 

_n p_ ( _x, y_ ) = _fi_ ( _x_ ) + � _yi−_ 1 _fi_ ( _x_ ) _._ 2 

Let _Ri_ = _N_ ( _fi_ ) _⊂_ R _[n]_ be the Newton polytopes of the polynomials _fi_ . The Newton polytope of the polynomial _p_ ( _x, y_ ) is the Cayley sum [ _R_ 1 _⋆. . . ⋆Rn_ ] _π_ , where _π_ : R[2] _[n][−]_[1] _→_ R _[n][−]_[1] is the natural projection such that _π_ �[ _R_ 1 _⋆. . . ⋆Rn_ ] _π_ � = ∆ _n−_ 1. 

... 

## **Subset** : Textbooks 

**meta** : 

book_name: Linear Toric Fibrations_Sandra Di Rocco, type: Notes, ... 

27 

Figure 6: An example textbook document in MATHPILE 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math_images/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math.pdf-0028-00.png)


**----- Start of picture text -----**<br>
A document from MATHPILE-ProofWiki<br>Text:<br>\section{Test for Submonoid}<br>Tags: Abstract Algebra, Monoids<br>\begin{theorem}<br>To show that \struct {T, circ} is a submonoid of a monoid \struct {S, circ}, we need to show that:<br>:(1): T ⊆ S<br>:(2): \struct {T, circ} is a magma (that is, that it is closed)<br>:(3): \struct {T, circ} has an identity.<br>\end{theorem}<br>\begin{proof}<br>From Subsemigroup Closure Test, (1) and (2) are sufficient to show that \struct {T, circ} is a subsemigroup of \struct {S, circ}.<br>Demonstrating the presence of an identity is then sufficient to show that it is a monoid. {{qed}}<br>Category:Monoids<br>\end{proof}<br>...<br>Subset : ProofWiki<br>meta :<br>type: Theorem_Proof,<br>...<br>**----- End of picture text -----**<br>


Figure 7: An example ProofWiki (a theorem and its proof) document in MATHPILE 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math_images/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math.pdf-0028-02.png)


**----- Start of picture text -----**<br>
A document from MATHPILE-ProofWiki<br>Text:<br>\begin{definition}[Definition:That which produces Medial Whole with Medial Area/Whole]<br>Let  a, b ∈R> 0 be (strictly) positive real numbers such that  a > b .<br>Let  a − b  be a straight line which produces with a medial area a medial whole.<br>The real number  a  is called the ”’whole”’ of the straight line which produces with a medial area a medial whole.<br>Category:Definitions/Euclidean Number Theory<br>\end{definition}<br>Subset : ProofWiki<br>meta :<br>type: Definition,<br>...<br>**----- End of picture text -----**<br>


Figure 8: An example ProofWiki (definition) document in MATHPILE 

28 

## A document from MATHPILE-arXiv 

## **Text:** 

- \begin{document} 

- \title{Coherence freeze in an optical lattice investigated via pump-probe spectroscopy} 

- \author{Samansa Maneshi} 

- \email[]{smaneshi@physics.utoronto.ca} 

- \author{Chao Zhuang} 

- \author{Christopher R. Paul} 

- \author{Luciano S. Cruz} 

- \altaffiliation[Current address: ]{UFABC, São Paulo, Brazil.} 

- \author{Aephraim M. Steinberg} 

- \affiliation{Centre for Quantum Information & Quantum Control and Institute for Optical Sciences, Department of Physics, University of Toronto, Canada } 

- \date{\today} 

- \pacs{37.10.Jk, 03.65.Yz, 03.67.-a, 42.50.Md} 

- \begin{abstract} 

Motivated by our observation of fast echo decay and a surprising coherence freeze, we have developed a pump-probe spectroscopy technique for vibrational states of ultracold[85] Rb atoms in an optical lattice to gain information about the memory dynamics of the system. We use pump-probe spectroscopy to monitor the time-dependent changes of frequencies experienced by atoms and to characterize the probability distribution of these frequency trajectories. We show that the inferred distribution, unlike a naive microscopic model of the lattice, correctly predicts the main features of the observed echo decay. \end{abstract} 

\maketitle 

Characterizing decoherence mechanisms is a crucial task for experiments aiming to control quantum systems, e.g., for quantum information processing (QIP). In this work, we demonstrate how two-dimensional (2D) pump-probe spectroscopy may be extended to provide important information on these mechanisms. As a model system, we study quantum vibrational states of ultracold atoms in an optical lattice. In addition to being a leading candidate system for QIP \citeBrennenJaksch, optical lattices are proving a versatile testing ground for the development of quantum measurement and control techniques \citeOMandel, Anderlini and a powerful tool for quantum simulations, e.g. the study of Anderson localization and the Hubbard model \citeMottAnderson. 

In our experiment, we study the vibrational coherence of[85] Rb atoms trapped in a shallow one-dimensional standing wave. Through our 2D pump-probe technique, we obtain detailed microscopic information on the frequency drift experienced by atoms in the lattice, enabling us to predict the evolution of coherence. Since the pioneering development of the technique in NMR\citeJeener-Ernst, 2D spectroscopy has been widely used to obtain high-resolution spectra and gain information about relaxations, couplings, and many-body interactions, in realms ranging from NMR \citeErnst to molecular spectroscopy \citeMukamel-Jonas, Hybl, Brixner, MillerNature to semiconductor quantum wells \citeCundiff, KWStone. Here, we show that similar powerful techniques can be applied to the quantized center-of-mass motion of trapped atoms, and more generally, offer a new tool for the characterization of systems in QIP and quantum control. 

\begin{figure} 

\caption{(Color online) Two typical measurements of echo amplitude vs. time. The echo pulse and the observed echo envelope are centered at times _tp_ and 2 _tp_ , respectively. After an initial decay, echo amplitude stays constant for about 1 _ms_ forming a plateau, before decaying to zero. The average lattice depths are 20 _ER_ (circles) and 18 _ER_ (squares).} 

\label{fig1} 

\end{figure} 

We have previously measured the evolution of coherence between the lowest two vibrational states of potential wells \cite{Ours}. 

The dephasing time is about 0 _._ 3 _ms_ ( _T_ 2 _[⋆]_[).] 

This dephasing is partly due to an inhomogeneous distribution of lattice depths as a result of the transverse Gaussian profile of the laser beams. To measure the homogeneous decoherence time ( _T_ 2), we perform pulse echoes, measuring the echo amplitude as a function of time \cite{Ours}. 

Figure \ref{fig1} shows two typical measurements of echo amplitude carried out on different dates under slightly different conditions such as different average lattice depths and different dephasing times. The echo amplitude initially decays with a time constant of about 0 _._ 7 _ms_ , which is much faster than the photon scattering time ( _∼_ 60 _ms_ ) in the lattice. It then exhibits a 1 _ms_ -long coherence freeze followed by a final decay. Absent real decoherence on the short time scale of 1 _ms_ , only loss of frequency memory would inhibit the appearance of echoes. This loss comes about when atoms experience time-varying frequencies. We use 2D pump-probe spectroscopy to monitor this frequency drift. Our 2D pump-probe spectroscopy is essentially a version of spectral hole-burning for vibrational states. By monitoring the changes in the hole spectrum as a function of time we gain information on the atoms’ frequency drift. 

Information obtained from our 2D spectra enables us to characterize the temporal decay of frequency memory and through our simulations we find that “coherence freeze" is related to the shape of this memory loss function. 

Similar plateaus in echo decay and a two-stage decay of echo amplitude have been observed in a Cooper-pair box \cite{Nakamura}, for a single electron spin in a quantum dot \cite{Vandersypen} and for electron spins in a semiconductor \cite{SClark}. Those plateaus or two-stage decays have been either explained through {\it{a priori}} models or simply described phenomenologically. Here, we are introducing an experimental technique to directly probe the origin of plateaus. 

The periodic potential in our experiment is formed by interfering two laser beams blue-detuned by 25 _GHz_ from the D2 transition line, _F_ = 3 _→ F[′]_ = 4 ( _λ_ = 780 _nm_ ), thus trapping atoms in the regions of low intensity, which minimizes the photon scattering rate and the transverse forces. The two laser beams intersect with parallel linear polarizations at an angle of _θ_ = (49 _._ 0 _±_ 0 _._ 2) _[◦]_ , resulting in a spacing of _L_ = (0 _._ 930 _±_ 0 _._ 004) _µm_ between the wells. Due to gravity, the full effective potential also possesses a “tilt” of 2 _._ 86 _ER_ per lattice site, where _ER_ = 8 _mLh_[2][2][is the effective lattice recoil energy.][The photon] scattering time in our experiment is _≈_ 60 _ms_ and the Landau-Zenner tunneling times for transitions from the lowest two levels are greater than 160 _ms_ . Atoms are loaded to the lattice during a molasses cooling stage and prepared in the ground vibrational state by adiabatic filtering \cite{StefanQPT}. Due to the short coherence length of atoms in optical molasses (60 _nm_ at 10 _µK_ ), there is no coherence between the wells. We measure populations of atoms in the ground vibrational, the first excited, and the (lossy) higher excited states _P_ 1, _P_ 2, and _PL_ , respectively, by fluorescence imaging of the atomic cloud after adiabatic filtering \cite{StefanQPT}. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math_images/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math.pdf-0029-30.png)


**----- Start of picture text -----**<br>
...<br>Subset : arXiv<br>meta :<br>id: 1005.2635,<br>language_detection_score: 0.8389,<br>...<br>**----- End of picture text -----**<br>


Figure 9: An example arXiv document in MATHPILE 

29 

A document from MATHPILE-StackExchange **Question:** Title: Are fractions hard because they are like algebra? Body: It occurs to me that to really understand the ways that people work with fractions on paper requires a good grasp of the ideas that numbers have multiple representations and that expressions can be manipulated in various ways without changing the number they represent. These are essentially algebraic ideas. For example, adding fractions requires us to rewrite the fractions in a different form, and then essentially factorise the expression. This is the same as rearranging expressions in algebra. Dividing fractions requires us to rerepresent an operation like _÷_[2] 3[as] _[ ×]_[3] 2[.][This is the same as realising the connection between] operations that you use to solve equations in algebra. And cancelling down before multiplying is very sophisticated rewriting relying on various associative and commutative laws. So it seems that we are really asking children to think in algebraic ways in order to understand fraction calculations well. This would seem to me to be a good reason why children and adults find it hard - they need more scaffolding in some abstract ideas. Is this a reasonable theory and has anyone written about this algebra-fractions connection before? To be clear, I am not asking if this is the only reason fractions are hard, but if there is any discussion out there to draw parallels between learning algebra and learning to manipulate fractions. Id: 7826 Score: 17 Tags: <algebra><fractions> LastEditorDisplayName: None OwnerDisplayName: None ClosedDate: None FavoriteCount: None language_detection_score: 0.9558 ... **Answers** : Body: Not sure about paper references. One reason why people don[´] t understand fractions is because they are seemingly illogical. You score one basket out of three 1/3. A little while later you try again and score 1/2. Clearly you have scored 2/5 shots? In many ways this is the correct answer. So why shouldn[´] t 3[1][+][1] 2[=] 5[2] People generally don[´] t understand equivalent fractions. It is strange for one farmer to say there are 4 sheep and another to say there are 8/2 sheep in the same field. People assume that the number 4 does what it says on the tin and is how we always describe 4 ness of something. They don[´] t understand equivalence. Partly to blame is treating fractions like conjuring tricks. If this is the question...do this, if this is the question ...do another uncorrelated thing. I asked my class (who seemingly could compute[2] 3 _[×]_ 5[3][correctly) to draw me a picture] instead of just multiplying. No one could do it yet they all said "but it´s _frac_ 615 you times the top and the bottom!" I think drawing fractions is extremely useful. Draw[2] 3 _[÷]_[2][ or][ 2] _[÷]_[2] 3[It´s not easy but I find students develop robustness eventually and begin to abstract] themselves. Id: 7827, Score: 9, is_accepted_answer: False, language_detection_score: 0.9599, Body: The obvious (to me) source of difficulty is that fractions are just plain complicated, more so than almost anything else in elementary education. You have to operate with a pair of numbers, instead of a single one, and you have to keep the order straight. Adding is quite complicated in its own right. Things are further complicated by rules about least common denominators and least terms. I ´m a little unclear about the question´s emphasis on algebra. Any sort of general rule or operation in arithmetic must have a connection to algebra, but I do not see what is intrinsically difficult about algebra that relates to numeric fractions.algebraic fractions among them. It seems to me that fractions are difficult because it´s easy to confuse the various bits.Certainly some parts of algebra are hard, and some parts harder than others,Even when youve got them straight,´ they´re noticeably slower to use, take concentration, and when things have such cognitive demands, they´re harder to think with. Conceptually, they´re a little bit odd, which is probably distracting until you get used to them. What they represent do not seem to apply to the same things that (whole) numbers do. Evidently fractions are not considered in this passage: In that city, which was the oldest in the world, the cat was an object of veneration. Its worship was the religion of the country. The multiplication and addition of cats were a perpetual instruction in arithmetic. Naturally, any inattention to the wants of a cat was punished with great severity in this world and the next... – A. Bierce, "A Revolt of the Gods" Now to have one-and-a-half cats seems a very different thing than to have three halves. In the former case, there´s a good chance that the one cat you have will be alive and purring, while the same could not possibly be said about any of the halves. No doubt such lessons are considered blasphemous in that city. While many things may be divided into parts – cars are a better example than cats – not many can be divided into equivalent parts that can be used as a basis for fractions. As we get used to fractions, as well as real numbers, we are taught to ignore this and accept statements such as "the average family has 2.4 children." Here is another example: By then, she will have shed 80 of the 240 pounds she weighed in with when she entered Peter Bent Brigham hospital obesity program. A third of her left behind! – The Boston Herald American, 7/7/77 The question seems to welcome references. There are certainly several that connect fractions with algebra. This paper, Seigler et al. (2013), Fractions: the new frontier for theories of numerical development, Trends in Cognitive Sciences, is a short survey of what is known and unknown about neural bases for one´s knowledge of fractions. Whole number arithmetic knowledge has been studied, and the authors suggest that the representation of the knowledge fractions is an area ripe for investigation. It reviews (with references) why fractions are difficult and the relation of skill at fractions to skill at algebra. Generally – or, rather, I only know of papers that discuss the connection in that direction, with algebra skill being dependent on fractions skill. (OTOH, I ´m not widely read in this area.) Id: 7831, Score: 11, is_accepted_answer: False, language_detection_score: 0.9780 **Subset** : StackExchange 

Figure 10: An example StackExchange document in MATHPILE . Here is a question from “matheducators” “.stackexchang.com” with two high-quality responses. 

30 

## **D Details for Corpus Collection and Processing** 

The subjects from which we collected papers on arXiv are listed in Table 7. The specific StackExchange sites from which we gathered data are listed in Table 8. We illustrate the LaTeX display issue with an example in Figure 11. 

During the collection process of arXiv, we undertook extensive transformations to enhance data clarity and consistency. Specifically, we (1) removed comments in each paper; (2) reverted many macro commands (e.g., “newcommand”) to their original forms; (3) omitted figure environments while retaining captions and figure labels; (4) excluded acknowledgements sections; (5) eliminated references in each paper; (6) condensed more than three consecutive empty lines to two; (7) replaced certain formatting commands like “hfill” and “vspace” with an empty line; (8) replaced the “maketitle” command in the main document body with the actual title (if available); (9) preserved only the content within the main body of the LaTex document. 

We summarize the parts of the dataset collection (cf. § 2) and global data preprocessing (cf. § 3) where human intervention was involved and whether the cleaning process was automated in the Table 9 and Table 10. We hope this provides a clearer understanding of MATHPILE construction process. 

Table 7: The subject list during collecting corpus from arXiv. 

Subjects 

math.AG, math.AT, math.AP, math.CT, math.CA, math.CO, math.AC, math.CV, math.DG, math.DS, math.FA, math.GM, math.GN, math.GT, math.GR, math.HO, math.IT, math.KT, math.LO, math.MP, math.MG, math.NT, math.NA, math.OA, math.OC, math.PR, math.QA, math.RT, math.RA, math.SP, math.ST, math.SG, math-ph, quant-ph, cs.CC, cs.CG, cs.DM, cs.DS, cs.FL, cs.GT, cs.LG, cs.NA, cs.LO, q-fin.MF, stat.CO, stat.ML, stat.ME, stat.OT, stat.TH, econ.TH 

Table 8: The site list during collecting corpus from StackExchange. 

Sites sourced from StackExchange math.stackexchange.com, mathoverflow.net, mathematica.stackexchange.com, matheducators.stackexchange.com, hsm.stackexchange.com, physics.stackexchange.com, proofassistants.stackexchange.com, tex.stackexchange.com, datascience.stackexchange, cstheory.stackexchange.com, cs.stackexchange.com 

31 

Table 9: Details of Human Involvement and Automation in the MATHPILE Collection Process 

|**MATHPILE Subset**|**Human Involvement in Data Collection**|**Cleaning Process Automated?**|
|---|---|---|
|Textbooks|Manual search and download of open-|Yes, the automated application|
||source, free mathematics textbooks; quality|of the PDF conversion API and|
||check; setting cleaning rules|the document cleaning rules|
|arXiv Papers|Manual selection of relevant mathematical|Yes, automated cleaning steps|
||feld categories; setting latex cleaning and|like comment removal and for-|
||formatting rules|mat conversion|
|Wikipedia Mathemat-|Humans observed samples to defne rules|Yes, automated HTML to Mark-|
|ical Entries|for cleaning irrelevant content such as copy-|down conversion; and removal|
||right statements|of extraneous lines|
|ProofWiki Entries|No signifcant human intervention (mainly|Yes, automated text parsing and|
||data dump selection, reformatting design)|formatting|
|StackExchange Dis-|Selection of relevant mathematics-related|Yes, automated HTML parsing|
|cussions|sites within the StackExchange network;|and conversion, score fltering|
||setting flter thresholds||
|Common Crawl Web|Manual adjustment of TF-IDF rules to im-|Yes, automated application of|
|Pages|prove mathematical content identifcation|rule-based mathematical docu-|
|||ment fltering|



Table 10: Details of Human Involvement and Automation in the MATHPILE Global Data Processing Steps 

||**Global Data Processing Step**<br>**Human Involvement**|||**Cleaning Process**|**Cleaning Process**|**Cleaning Process**|**Cleaning Process**|**Automated?**|**Automated?**|**Automated?**|**Automated?**||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Language Identifcation<br>Manual adjustment of thresholds|||Yes,|using||FastText||||with custom||
||based on observed false positives|||thresholds|||||||||
||Data Cleaning & Filtering<br>Human observation to defne rules|||Yes, automated application of rules|||||||||
||for fltering irrelevant content|||for content||fltering and removal|||||||
||Data Deduplication<br>Manual review of near-duplicate|||Yes, automated using||||||MinHash LSH|||
||samples from different sources||||||||||||
||Data Contamination Detection<br>Human<br>verifcation<br>of<br>fagged|||Yes, automated detection based on|||||||||
||benchmark duplicates|||pre-defned|||criteria||||||
||||||||||||||
||# Abhyankar's inequality||||||||||||
||**Abhyankar's inequality** is an inequality involving extensions of valued fields in algebra, introduced by Abhyankar (1956).||||||||||||
||Abhyankar's inequality states that for an extension _K_ / _k_ of valued fields, the transcendence|||degree of _K_ /|||_k_ is at|least the|||||
||transcendence degree of the residue field extension plus the rank of the quotient of the valuation groups; here the rank of an|||||||||abelian group|||
||A {\displaystyle A} A is defined as  dim Q ( A⊗Q ) {\displaystyle \dim _{\mathbb {Q} }(A\otimes||\mathbb||{Q} )} {\\displaystyle|||||\\dim|||
||_{\\mathbb {Q} }\(A\\otimes \\mathbb {Q} \)}.||||||||||||
||## References||||||||||||
||* Abhyankar, Shreeram (1956), "On the valuations centered in a local domain", _American Journal of Mathematics_ , **78** (2): 321–348,||||||||||||
||doi:10.2307/2372519, ISSN 0002-9327, JSTOR 2372519, MR 0082477||||||||||||
||# Abhyankar's inequality||||||||||||
||**Abhyankar's inequality** is an inequality involving extensions of valued fields in algebra, introduced by Abhyankar (1956).||||||||||||
||Abhyankar's inequality states that for an extension $K / k$ of valued fields, the transcendence degree of||||$K / k$|is at least||the|||||
||transcendence degree of the residue field extension plus the rank of the quotient of the valuation groups; here the rank of an|||||||||abelian group|||
||$A$ is defined as $\dim _{\mathbb {Q}} (A\otimes \mathbb {Q} )$.||||||||||||
||## References||||||||||||
||* Abhyankar, Shreeram (1956), "On the valuations centered in a local domain", _American Journal of Mathematics_ , **78** (2): 321–348,||||||||||||
||doi:10.2307/2372519, ISSN 0002-9327, JSTOR 2372519, MR 0082477||||||||||||



Figure 11: A document processed by html2text (above) compared to one obtained through another library **Resiliparse** plus DOM parsing (below). 

32 

## **E Example for Quality Annotation** 

We present a cleaned example document with quality annotations (see Figure 12). 

A document from MATHPILE-Common Crawl **Text:** This number is called the Copeland–Erd˝os constant, and is known to be irrational and normal. I believe its transcendence or otherwise is an open problem. This source claims that it has been proved to be transcendental, but the paper they refer to is the one in which it was proved to be normal and so I think the source is mistaken. For now, the knowledge that it is almost surely transcendental will have to suffice! Not the answer you’re looking for? Browse other questions tagged number-theory transcendental-numbers or ask your own question. Does the number 2 _._ 3 _,_ 5 _,_ 7 _,_ 11 _,_ 13 _. . ._ exist and, if so, is it rational or irrational &or transcendental? Is 0 _._ 248163264128 _..._ a transcendental number? What is the name of this number? Is it transcendental? Is 0 _._ 112123123412345123456 _. . ._ algebraic or transcendental? Is 0 _._ 121121111112111 _..._ a transcendental number? Do we know a transcendental number with a proven bounded continued fraction expansion? If we delete the non-primes from _e_ , is the resulting number transcendental? Is there any known transcendental _b_ such that _b[b]_ is also transcendental? ... **Subset** : Common Crawl **meta** : language_detection_score: 0.9118, char_num_after_normalized: 887, contain_at_least_two_stop_words: True, ellipsis_line_ratio: 0.0, idx: 95994, lines_start_with_bullet_point_ratio: 0.0, mean_length_of_alpha_words: 4.2941, non_alphabetical_char_ratio: 0.0234, symbols_to_words_ratio: 0.0117, uppercase_word_ratio: 0.0117 ... 

Figure 12: An example document after cleaning and filtering with quality annotations 

## **F Examples of Duplicates Encountered in the Deduplication Process** 

We provide some illustrative examples of duplicates from each source in the deduplication process, as shown in Table 11 to Table 16. 

We also provide examples of downstream task benchmarks (i.e., MATH and MMLU-STEM) leaks identified during our data contamination detection process for our corpus (as shown in Table 17 and Table 18) and OpenWebMath (as shown in Table 19). 

33 

Table 11: Near-duplication matches found in CommonCrawl by MinHash LSH deduplication (in _italics_ ). 

|_In algebraic topology we often encounter chain complexes_<br>_with extra multiplicative structure._<br>_For example, the_<br>_cochain complex of a topological space has what is called_<br>_the E∞-algebra structure which comes from the cup prod-_<br>_uct._<br>_In this talk I present an idea for studying such chain com-_<br>_plexes, E∞differential graded algebras (E∞DGAs), using_<br>_stable homotopy theory. Namely, I discuss new equiva-_<br>_lences between E∞DGAS that are defned using commuta-_<br>_tive ring spectra._<br>ring spectra are equivalent. _Quasi-isomorphic E∞DGAs_<br>_are E∞topologically equivalent. However, the examples I_<br>_am going to present show that the opposite is not true; there_<br>_are E∞DGAs that are E∞topologically equivalent but not_<br>_quasi-isomorphic. This says that between E∞DGAs, we_<br>_have more equivalences than just the quasi-isomorphisms._<br>_I also discuss interaction of E∞topological equiva-_<br>_lences with the Dyer-Lashof operations and cases where_<br>_E∞topological equivalences and quasi-isomorphisms_<br>_agree._|Özet : _In algebraic topology we often encounter chain_<br>_complexes with extra multiplicative structure. For example,_<br>_the cochain complex of a topological space has what is_<br>_called the E∞-algebra structure which comes from the cup_<br>_product. In this talk I present an idea for studying such_<br>_chain complexes, E∞differential graded algebras (E∞_<br>_DGAs), using stable homotopy theory. Namely, I discuss_<br>_new equivalences between E∞DGAS that are defned using_<br>_commutative ring spectra_.We say_E∞_DGAs are_E∞_topo-<br>logically equivalent when the corresponding commutative<br>ring spectra are equivalent. _Quasi-isomorphic E∞DGAs_<br>_are E∞topologically equivalent. However, the examples I_<br>_am going to present show that the opposite is not true; there_<br>_are E∞DGAs that are E∞topologically equivalent but not_<br>_quasi-isomorphic. This says that between E∞DGAs, we_<br>_have more equivalences than just the quasi-isomorphisms._<br>_I also discuss interaction of E∞topological equivalences_<br>_with the Dyer-Lashof operations and cases where E∞topo-_<br>_logical equivalences and quasi-isomorphisms agree._|
|---|---|
|||
|Université de la Saskatchewan,<br>1 - 4 juin 2015<br>www.smc.math.ca//2015f<br>Comité d’organisation<br>Financement étudiants<br>Minisymposia invités<br>Minisymposia libres<br>Conférences libres<br>Horaire - Minisymposa invités<br>Open Problems<br>Graphs and matrices<br>Responsable et président: Shaun Fallat et Karen Meagher<br>(University of Regina)<br>_WAYNE BARRETT, Brigham Young University_<br>_The Fielder Vector and Tree Decompositions of Graphs_<br>_[PDF]_<br>_In the 1970’s Fiedler initiated a study of the second smallest_<br>_eigenvalue of the Laplacian matrix L of a graph and the_<br>_corresponding eigenvector(s). These "Fiedler" vectors have_<br>_become spectacularly successful in revealing properties of_<br>_the associated graph. A tree decomposition T of a graph_<br>_G_= (_V, E_)_is an associated tree whose nodes are subsets_<br>_of V and whose edge set respects the structure of G. Tree_<br>_decompositions have been used in the analysis of complex_<br>_networks. This talk reports on an algorithm developed_<br>_by students at BYU for obtaining a tree decomposition by_<br>_means of Fiedler vector(s) of G._<br>_..._<br>_Graphs that have a weighted adjacency matrix with spec-_<br>_trum {λn−_2<br>1<br>_, λ_2<br>2_} [PDF]_<br>_In this talk I will characterize the graphs which have an_<br>_edge weighted adjacency matrix belonging to the class of_<br>_n × n involutions with spectrum equal to {λn−_2<br>1<br>_, λ_2<br>2_} for_<br>_some λ_1_and some λ_2_. The connected graphs turn out to be_<br>_the cographs constructed as the join of at least two unions_<br>_of pairs of complete graphs, and possibly joined with one_<br>_other complete graph._|University<br>of<br>Saskatchewan,<br>June<br>1<br>-<br>4,<br>2015<br>www.cms.math.ca//2015<br>Invited Minisymposia<br>Contributed Minisymposia<br>Contributed Talks<br>Graphs and matrices<br>Organizer and Chair: Shaun Fallat and Karen Meagher<br>(University of Regina)<br>_WAYNE BARRETT, Brigham Young University_<br>_The Fielder Vector and Tree Decompositions of Graphs_<br>_[PDF]_<br>_In the 1970’s Fiedler initiated a study of the second smallest_<br>_eigenvalue of the Laplacian matrix L of a graph and the_<br>_corresponding eigenvector(s). These "Fiedler" vectors have_<br>_become spectacularly successful in revealing properties of_<br>_the associated graph. A tree decomposition T of a graph_<br>_G_= (_V, E_)_is an associated tree whose nodes are subsets_<br>_of V and whose edge set respects the structure of G. Tree_<br>_decompositions have been used in the analysis of complex_<br>_networks. This talk reports on an algorithm developed_<br>_by students at BYU for obtaining a tree decomposition by_<br>_means of Fiedler vector(s) of G._<br>_..._<br>_Graphs that have a weighted adjacency matrix with spec-_<br>_trum {λn−_2<br>1<br>_, λ_2<br>2_} [PDF]_<br>_In this talk I will characterize the graphs which have an_<br>_edge weighted adjacency matrix belonging to the class of_<br>_n × n involutions with spectrum equal to {λn−_2<br>1<br>_, λ_2<br>2_} for_<br>_some λ_1_and some λ_2_. The connected graphs turn out to be_<br>_the cographs constructed as the join of at least two unions_<br>_of pairs of complete graphs, and possibly joined with one_<br>_other complete graph._|



34 

Table 12: A near-duplication match found in arXiv by MinHashLSH deduplication (in _italics_ ). 

|_\begin{document}_<br>_\title{_Querying Guarded Fragments via Resolution_}_<br>_\section{A detailed example}_<br>_Here we include some equations and theorem-like environments to show_<br>_how these are labeled in a supplement and can be referenced from the_<br>_main text._<br>_Consider the following equation:_<br>_\begin{equation}_<br>_\label{eq:suppa}_<br>_a^2 + b^2 = c^2._<br>_\end{equation}_<br>_You can also reference equations such as \cref{eq:matrices,eq:bb}_<br>_from the main article in this supplement._<br>_\lipsum[100-101]_<br>_\begin{theorem}_<br>_An example theorem._<br>_\end{theorem}_<br>_\lipsum[102]_<br>_\begin{lemma}_<br>_An example lemma._<br>_\end{lemma}_<br>_\lipsum[103-105]_<br>_Here is an example citation: \cite{KoMa14}._<br>_\section[Proof of Thm]{Proof of \cref{thm:bigthm}}_<br>_\label{sec:proof}_<br>_\lipsum[106-112]_<br>_\section{Additional experimental results}_<br>_\Cref{tab:foo} shows additional_<br>_supporting evidence._<br>_\begin{table}[htbp]_<br>_{\footnotesize_<br>_\caption{Example table} \label{tab:foo}_<br>_\begin{center}_<br>_\begin{tabular}{|c|c|c|} \hline_<br>_Species & \bf Mean & \bf Std.∼Dev. \\\hline_<br>_1 & 3.4 & 1.2 \\_<br>_2 & 5.4 & 0.6 \\\hline_<br>_\end{tabular}_<br>_\end{center}_<br>_}_<br>_\end{table}_<br>_\end{document}_|_\begin{document}_<br>_\title{_Limited memory Kelley’s Method Converges for Composite<br>Convex and Submodular Objectives_}_<br>_\section{A detailed example}_<br>_Here we include some equations and theorem-like environments to show_<br>_how these are labeled in a supplement and can be referenced from the_<br>_main text._<br>_Consider the following equation:_<br>_\begin{equation}_<br>_\label{eq:suppa}_<br>_a^2 + b^2 = c^2._<br>_\end{equation}_<br>_You can also reference equations such as \cref{eq:matrices,eq:bb}_<br>_from the main article in this supplement._<br>_\lipsum[100-101]_<br>_\begin{theorem}_<br>_An example theorem._<br>_\end{theorem}_<br>_\lipsum[102]_<br>_\begin{lemma}_<br>_An example lemma._<br>_\end{lemma}_<br>_\lipsum[103-105]_<br>_Here is an example citation: \cite{KoMa14}._<br>_\section[Proof of Thm]{Proof of \cref{thm:bigthm}}_<br>_\label{sec:proof}_<br>_\lipsum[106-112]_<br>_\section{Additional experimental results}_<br>_\Cref{tab:foo} shows additional_<br>_supporting evidence._<br>_\begin{table}[htbp]_<br>_{\footnotesize_<br>_\caption{Example table} \label{tab:foo}_<br>_\begin{center}_<br>_\begin{tabular}{|c|c|c|} \hline_<br>_Species & \bf Mean & \bf Std.∼Dev. \\\hline_<br>_1 & 3.4 & 1.2 \\_<br>_2 & 5.4 & 0.6 \\\hline_<br>_\end{tabular}_<br>_\end{center}_<br>_}_<br>_\end{table}_<br>_\end{document}_||
|---|---|---|



35 

\section{Definition:Constructed \section{Definition:Constructed Semantics/Instance 4/Rule of Semantics/Instance 5/Rule of Idempotence} Idempotence} Tags: Formal Semantics Tags: Formal Semantics \begin{theorem} \begin{theorem} The Rule of Idempotence: The Rule of Idempotence: :$(p \lor p) \implies p$ :$(p \lor p) \implies p$ is a tautology in Instance 4 of is a tautology in Instance 5 of constructed semantics. constructed semantics. \end{theorem} \end{theorem} \begin{proof} \begin{proof} By the definitional abbreviation for By the definitional abbreviation for the conditional: the conditional: :$\mathbf A \implies \mathbf B =_{\ :$\mathbf A \implies \mathbf B =_{\ text{def}} \neg \mathbf A \lor \ text{def}} \neg \mathbf A \lor \ mathbf B$ mathbf B$ the Rule of Idempotence can be the Rule of Idempotence can be written as: written as: : $\neg \left({p \lor p}\right) \lor : $\neg \left({p \lor p}\ right) \lor p$ p$ This evaluates as follows: This evaluates as follows: :$\begin{array }{| cccc|c|c|} \hline :$\begin{array }{| cccc|c|c|} \hline \neg & (p & \lor & p) & \lor & p \\ \neg & (p & \lor & p) & \lor & p \\ \hline \hline 1 & 0 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 1 & 1 & 0 & 1 \\ 0 & 1 & 1 & 1 & 0 & 1 \\ 0 & 2 & 2 & 2 & 0 & 2 \\ 3 & 2 & 2 & 2 & 0 & 2 \\ 2 & 3 & 3 & 3 & 0 & 3 \\ 0 & 3 & 3 & 3 & 0 & 3 \\ \hline \hline \end{array}$ \end{array}$ {{qed}} {{qed}} Category:Formal Semantics Category:Formal Semantics \end{proof} \end{proof} \section{Imaginary Part of Complex \section{Real Part of Complex Product} Product} Tags: Complex Multiplication Tags: Complex Multiplication \begin{theorem} \begin{theorem} Let $z_1$ and $z_2$ be complex Let $z_1$ and $z_2$ be complex numbers. numbers. Then: Then: :$\map \Im {z_1 z_2} = \map \Re {z_1 :$\map \Re {z_1 z_2} = \map \Re {z_1 } \, \map \Im {z_2} + \map \Im { } \map \Re {z_2} - \map \Im {z_1} z_1} \, \map \Re {z_2}$ \map \Im {z_2}$ \end{theorem} \end{theorem} \begin{proof} \begin{proof} Let $z_1 = x_1 + i y_1$ and $z_2 = Let $z_1 = x_1 + i y_1$ and $z_2 = x_2 + i y_2$. x_2 + i y_2$. By definition of complex By definition of complex multiplication: multiplication: :$z_1 z_2 = x_1 x_2 - y_1 y_2 + i \ :$z_1 z_2 = x_1 x_2 - y_1 y_2 + i \ paren {x_1 y_2 + x_2 y_1}$ paren {x_1 y_2 + x_2 y_1}$ Then Then: {{begin -eqn}} {{begin -eqn}} {{eqn | l = \map \Im {z_1 z_2} {{eqn | l = \map \Re {z_1 z_2} | r = x_1 y_2 + x_2 y_1 | r = x_1 x_2 - y_1 y_2 | c = {{Defof|Imaginary Part}} | c = {{ Defof|Real Part}} }} }} {{eqn | r = \map \Re {z_1} \, \map \ {{eqn | r = \map \Re {z_1} \map \Re Im {z_2} + \map \Im {z_1} \, \map {z_2} - \map \Im {z_1} \map \Im { \Re {z_2} z_2} | c = {{Defof|Imaginary Part}} | c = {{ Defof|Real Part}} 36 }} }} {{end -eqn}} {{end -eqn}} {{qed}} {{qed}} \end{proof} \end{proof} 

\begin{theorem} Let $z_1$ and $z_2$ be complex numbers. Then: :$\map \Im {z_1 z_2} = \map \Re {z_1 } \, \map \Im {z_2} + \map \Im { z_1} \, \map \Re {z_2}$ \end{theorem} \begin{proof} Let $z_1 = x_1 + i y_1$ and $z_2 = x_2 + i y_2$. By definition of complex multiplication: :$z_1 z_2 = x_1 x_2 - y_1 y_2 + i \ paren {x_1 y_2 + x_2 y_1}$ Then {{begin -eqn}} {{eqn | l = \map \Im {z_1 z_2} | r = x_1 y_2 + x_2 y_1 | c = {{Defof|Imaginary Part}} }} {{eqn | r = \map \Re {z_1} \, \map \ Im {z_2} + \map \Im {z_1} \, \map \Re {z_2} | c = {{Defof|Imaginary Part}} 36 }} {{end -eqn}} {{qed}} \end{proof} 

Table 14: Duplication matches found in Wikipedia by MinHash LSH deduplication (in _italics_ ). 

_# HP-42S # HP-42S The **HP-42S RPN Scientific** is a programmable RPN The **HP-42S RPN Scientific** is a programmable RPN Scientific hand held calculator introduced by HewlettScientific hand held calculator introduced by HewlettPackard in 1988. It has advanced functions suitable for apPackard in 1988. It has advanced functions suitable for applications in mathematics, linear algebra, statistical analyplications in mathematics, linear algebra, statistical analysis, computer science and others. sis, computer science and others. HP-42S HP-42S The HP-42S The HP-42S — — Type| Programmable scientific Type| Programmable scientific Manufacturer| Hewlett-Packard Manufacturer| Hewlett-Packard Introduced| 1988 Introduced| 1988 Discontinued| 1995 Calculator Discontinued| 1995 Calculator Entry mode| RPN Entry mode| RPN Precision| 12 display digits (15 digits internally),[1] expoPrecision| 12 display digits (15 digits internally),[1] exponent ±499 nent ±499 Display type| LCD dot-matrix Display type| LCD dot-matrix Display size| 2 lines, 22 characters, 131×16 pixels CPU Display size| 2 lines, 22 characters, 131×16 pixels CPU Processor| Saturn (Lewis) Programming Processor| Saturn (Lewis) Programming Programming language(s)| RPN key stroke (fully merged) Programming language(s)| RPN key stroke (fully merged) Firmware memory| 64 KB of ROM Firmware memory| 64 KB of ROM Program steps| 7200 Interfaces Program steps| 7200 Interfaces Ports| IR (Infrared) printing Other Ports| IR (Infrared) printing Other Power supply| 3×1.5V button cell batteries (Panasonic Power supply| 3×1.5V button cell batteries (Panasonic LR44, Duracell PX76A/675A or Energizer 357/303) LR44, Duracell PX76A/675A or Energizer 357/303) Weight| 6 oz (170 g) Weight| 6 oz (170 g) Dimensions| 148×80×15mm Dimensions| 148×80×15mm ## Overview ## Overview_ 

_Perhaps the HP-42S was to be released as a replacement for the aging HP-41 series as it is designed to be compatible with all programs written for the HP-41. Since it lacked expandability, and lacked any real I/O ability, both key features of the HP-41 series, it was marketed as an HP-15C replacement._ 

_Perhaps the HP-42S was to be released as a replacement for the aging HP-41 series as it is designed to be compatible with all programs written for the HP-41. Since it lacked expandability, and lacked any real I/O ability, both key features of the HP-41 series, it was marketed as an HP-15C replacement._ 

_The 42S, however, has a much smaller form factor than the 41, and features many more built-in functions, such as a matrix editor, complex number support, an equation solver, user-defined menus, and basic graphing capabilities (the 42S can draw graphs only by programs). Additionally, it features a two-line dot matrix display, which made stack manipulation easier to understand._ 

_The 42S, however, has a much smaller form factor than the 41, and features many more built-in functions, such as a matrix editor, complex number support, an equation solver, user-defined menus, and basic graphing capabilities (the 42S can draw graphs only by programs). Additionally, it features a two-line dot matrix display, which made stack manipulation easier to understand._ 

_Production of the 42S ended in 1995.[2] As this calculator is regarded amongst the best ever made in terms of quality, key stroke feel, ease of programming, and daily usability for engineers,[3] in the HP calculator community the 42S has become famous for its high prices in online auctions, up to several times its introduction price, which has created a scarcity for utility end users._ 

_Production of the 42S ended in 1995.[2] As this calculator is regarded amongst the best ever made in terms of quality, key stroke feel, ease of programming, and daily usability for engineers,[3] in the HP calculator community the 42S has become famous for its high prices in online auctions, up to several times its introduction price, which has created a scarcity for utility end users._ 

37 

Table 15: Duplication matches found in Textbooks by MinHash LSH deduplication (in _italics_ ). 

||_# Basic Concepts in Graph Theory_<br>_## Section 1: What is a Graph?_<br>_There are various types of graphs, each with its own defni-_<br>_tion. Unfortunately, some people apply the term "graph"_<br>_rather loosely, so you can’t be sure what type of graph_<br>_they’re talking about unless you ask them. After you have_<br>_fnished this chapter, we expect you to use the terminology_<br>_carefully, not loosely. To motivate the various defnitions,_<br>_we’ll begin with some examples._<br>_Example 1 (A computer network) Computers are often_<br>_linked with one another so that they can interchange in-_<br>_formation. Given a collection of computers, we would like_<br>_to describe this linkage in fairly clean terms so that we can_<br>_answer questions such as "How can we send a message_<br>_from computer A to computer B using the fewest possible_<br>_intermediate computers?"_<br>_We could do this by making a list that consists of pairs of_<br>_computers that are connected. Note that these pairs are_<br>_unordered since, if computer_C_can communicate with com-_<br>_puter_D_, then the reverse is also true. (There are sometimes_<br>_exceptions to this, but they are rare and we will assume that_<br>_our collection of computers does not have such an excep-_<br>_tion.) Also, note that we have implicitly assumed that the_<br>_computers are distinguished from each other: It is insuff-_<br>_cient to say that "A PC is connected to a Mac." We must_<br>_specify which_PC_and which Mac. Thus, each computer_<br>_has a unique identifying label of some sort._<br>_For people who like pictures rather than lists, we can put_<br>_dots on a piece of paper, one for each computer. We label_<br>_each dot with a computer’s identifying label and draw a_<br>_curve connecting two dots if and only if the correspond-_<br>_ing computers are connected. Note that the shape of the_<br>_curve does not matter (it could be a straight line or some-_<br>_thing more complicated) because we are only interested in_<br>_whether two computers are connected or not. Below are_<br>_two such pictures of the same graph. Each computer has_<br>_been labeled by the initials of its owner._<br>_..._<br>_## Basic Concepts in Graph Theory_<br>_The notation Pk_(_V_) _stands for the set of all k-element_<br>_subsets of the set V . Based on the previous example we_<br>_have_<br>_Defnition 1 (Simple graph) A simple graph G is a pair_<br>_G_= (_V, E_)_where_<br>_- V is a fnite set, called the vertices of G, and_<br>_- E is a subset of P_2(_V_)_(i.e., a set E of two-element subsets_<br>_of V ), called the edges of G._<br>_..._|_# Basic Concepts in Graph Theory_<br>_## Section 1: What is a Graph?_<br>_There are various types of graphs, each with its own defni-_<br>_tion. Unfortunately, some people apply the term "graph"_<br>_rather loosely, so you can’t be sure what type of graph_<br>_they’re talking about unless you ask them. After you have_<br>_fnished this chapter, we expect you to use the terminology_<br>_carefully, not loosely. To motivate the various defnitions,_<br>_we’ll begin with some examples._<br>_Example 1 (A computer network) Computers are often_<br>_linked with one another so that they can interchange in-_<br>_formation. Given a collection of computers, we would like_<br>_to describe this linkage in fairly clean terms so that we can_<br>_answer questions such as "How can we send a message_<br>_from computer A to computer B using the fewest possible_<br>_intermediate computers?"_<br>_We could do this by making a list that consists of pairs of_<br>_computers that are connected. Note that these pairs are_<br>_unordered since, if computer_C_can communicate with com-_<br>_puter_D_, then the reverse is also true. (There are sometimes_<br>_exceptions to this, but they are rare and we will assume that_<br>_our collection of computers does not have such an excep-_<br>_tion.) Also, note that we have implicitly assumed that the_<br>_computers are distinguished from each other: It is insuff-_<br>_cient to say that "A PC is connected to a Mac." We must_<br>_specify which_PC_and which Mac. Thus, each computer_<br>_has a unique identifying label of some sort._<br>_For people who like pictures rather than lists, we can put_<br>_dots on a piece of paper, one for each computer. We label_<br>_each dot with a computer’s identifying label and draw a_<br>_curve connecting two dots if and only if the correspond-_<br>_ing computers are connected. Note that the shape of the_<br>_curve does not matter (it could be a straight line or some-_<br>_thing more complicated) because we are only interested in_<br>_whether two computers are connected or not. Below are_<br>_two such pictures of the same graph. Each computer has_<br>_been labeled by the initials of its owner._<br>_..._<br>_## Basic Concepts in Graph Theory_<br>_The notation Pk_(_V_) _stands for the set of all k-element_<br>_subsets of the set V . Based on the previous example we_<br>_have_<br>_Defnition 1 (Simple graph) A simple graph G is a pair_<br>_G_= (_V, E_)_where_<br>_- V is a fnite set, called the vertices of G, and_<br>_- E is a subset of P_2(_V_)_(i.e., a set E of two-element subsets_<br>_of V ), called the edges of G._<br>_..._||
|---|---|---|---|



38 

Table 16: Near-duplication matches found in StackExchange by MinHash LSH deduplication (in _italics_ ). 

|_italics_).||
|---|---|
|This was originally posted on mathoverfow, but it seems<br>it’s more appropriate to post here.<br>_Let B be a paracompact space with the property that any_<br>_(topological) vector bundle E →B is trivial. What are_<br>_some non-trivial examples of such spaces, and are there_<br>_any interesting properties that characterize them?_<br>_For simple known examples we of course have contractible_<br>_spaces, as well as the 3-sphere S_3_. This one follows from_<br>_the fact that its rank n vector bundles are classifed by_<br>_π_3(_BO_(_n_)) =_π_2(_O_(_n_)) = 0_. I’m primarily interested in_<br>_the case where B is a closed manifold. Do we know any_<br>_other such examples?_<br>_There is this nice answer to a MSE question which talks_<br>_about using the Whitehead tower of the appropriate clas-_<br>_sifying space to determine whether a bundle is trivial or_<br>_not. This seems like a nice tool (of which I am not familiar_<br>_with) to approaching this problem. As a secondary question,_<br>_could I ask for some insight/references to this approach?_<br>_EDIT Now that we know from the answer all the examples_<br>_for closed_3_-manifolds (integral homology spheres), I guess_<br>_I can now update the question to the case of higher odd di-_<br>_mensions. Does there exist a higher dimensional example?_|_Let B be a paracompact space with the property that any_<br>_(topological) vector bundle E →B is trivial. What are_<br>_some non-trivial examples of such spaces, and are there_<br>_any interesting properties that characterize them?_<br>_For simple known examples we of course have contractible_<br>_spaces, as well as the 3-sphere S_3_. This one follows from_<br>_the fact that its rank n vector bundles are classifed by_<br>_π_3(_BO_(_n_)) =_π_2(_O_(_n_)) = 0_. I’m primarily interested in_<br>_the case where B is a closed manifold. Do we know any_<br>_other such examples?_<br>_There is this nice answer to a MSE question which talks_<br>_about using the Whitehead tower of the appropriate clas-_<br>_sifying space to determine whether a bundle is trivial or_<br>_not. This seems like a nice tool (of which I am not familiar_<br>_with) to approaching this problem. As a secondary question,_<br>_could I ask for some insight/references to this approach?_<br>_EDIT Now that we know from the answers all the examples_<br>_for closed_3_-manifolds, I guess I can now update the ques-_<br>_tion to the case of higher odd dimensions. Does there exist_<br>_a higher dimensional example?_|
|||
|This<br>is<br>a<br>copy<br>of<br>my<br>question<br>on<br>MSE<br>(https://math.stackexchange.com/questions/3372432)<br>because this forum seems better suited for historical<br>questions:<br>_In 1985, Gosper used the not-yet-proven formula by Ra-_<br>_manujan_<br>1<br>_π_ = 2<br>~~_√_~~<br>2<br>992 _·_<br>_∞_<br>�<br>_n_=0<br>(4_n_)!<br>(_n_!)4 _·_ 26390_n_+ 1103<br>3964_n_<br>_to compute_17_·_106 _digits of π, at that time a new world_<br>_record._<br>_Here (https://www.cs.princeton.edu/courses/archive/fall98/_<br>_cs126/refs/pi-ref.txt) it reads:_<br>_There were a few interesting things about Gosper’s com-_<br>_putation. First, when he decided to use that particular_<br>_formula, there was no proof that it actually converged to_<br>_pi! Ramanujan never gave the math behind his work, and_<br>_the Borweins had not yet been able to prove it, because_<br>_there was some very heavy math that needed to be worked_<br>_through. It appears that Ramanujan simply observed the_<br>_equations were converging to the 1103 in the formula, and_<br>_then assumed it must actually be 1103. (Ramanujan was_<br>_not known for rigor in his math, or for providing any proofs_<br>_or intermediate math in his formulas.) The math of the_<br>_Borwein’s proof was such that after he had computed 10_<br>_million digits, and verifed them against a known calcula-_<br>_tion, his computation became part of the proof. Basically it_<br>_was like, if you have two integers differing by less than one,_<br>_then they have to be the same integer._<br>_Now my historical question: Who was the frst to prove this_<br>_formula? Was it Gosper because he added the last piece_<br>_of the proof, or was it the Borweins, afterwards? And was_<br>_Gosper aware of this proof when he did his computation?_|_In 1985, Gosper used the not-yet-proven formula by Ra-_<br>_manujan_<br>1<br>_π_ = 2<br>~~_√_~~<br>2<br>992 _·_<br>_∞_<br>�<br>_n_=0<br>(4_n_)!<br>(_n_!)4 _·_ 26390_n_+ 1103<br>994_n_<br>_to compute_17_·_106 _digits of π, at that time a new world_<br>_record._<br>_Here (https://www.cs.princeton.edu/courses/archive/fall98/_<br>_cs126/refs/pi-ref.txt) it reads:_<br>_There were a few interesting things about Gosper’s com-_<br>_putation. First, when he decided to use that particular_<br>_formula, there was no proof that it actually converged to_<br>_pi! Ramanujan never gave the math behind his work, and_<br>_the Borweins had not yet been able to prove it, because_<br>_there was some very heavy math that needed to be worked_<br>_through. It appears that Ramanujan simply observed the_<br>_equations were converging to the 1103 in the formula, and_<br>_then assumed it must actually be 1103. (Ramanujan was_<br>_not known for rigor in his math, or for providing any proofs_<br>_or intermediate math in his formulas.) The math of the_<br>_Borwein’s proof was such that after he had computed 10_<br>_million digits, and verifed them against a known calcula-_<br>_tion, his computation became part of the proof. Basically it_<br>_was like, if you have two integers differing by less than one,_<br>_then they have to be the same integer._<br>_Now my historical question: Who was the frst to prove this_<br>_formula? Was it Gosper because he added the last piece_<br>_of the proof, or was it the Borweins, afterwards? And was_<br>_Gosper aware of this proof when he did his computation?_|



39 

Table 17: Exact match examples from the test set of MATH benchmark found in Textbooks by line-level exact match deduplication (in _italics_ ). 

_Coin A is flipped three times and coin B is flipped four times. What is the probability that the number of heads obtained from flipping the two fair coins is the same?_ 

## Video Solution 

Answer: 

## Problem 3.2.2 (AMC 10) 

Two tour guides are leading six tourists. The guides decide to split up. Each tourist must choose one of the guides, but with the stipulation that each guide must take at least one tourist. How many different groupings of guides and tourists are possible? 

...... 

_One morning each member of Angela’s family drank an 8-ounce mixture of coffee with milk. The amounts of coffee and milk varied from cup to cup, but were never zero. Angela drank a quarter of the total amount of milk and a sixth of the total amount of coffee. How many people are in the family?_ 

Answer: 

## Problem 20.2.15 (AMC 12) 

The state income tax where Kristin lives is levied at the rate of _p_ % of the first $28000 of annual income plus ( _p_ + 2)% of any amount above $28000. Kristin noticed that the state income tax she paid amounted to ( _p_ + 0 _._ 25)% of her annual income. What was her annual income? 

Answer: 

...... 

_Find the least positive integer k for which the equation_ � 2002 _n_ � = _k has no integer solutions for n. (The notation ⌊x⌋ means the greatest integer less than or equal to x.)_ 

Answer: 

## Problem 40.1.9 (AIME) 

Find the number of positive integers _n_ less than 1000 for which there exists a positive real number _x_ such that _n_ = _x⌊x⌋_ .’, ”, ’Note: _⌊x⌋_ is the greatest integer less than or equal to _x_ .’ 

...... 

_What is the sum of the roots of z_[12] = 64 _that have a positive real part?_ 

Answer: 

## Problem 45.8.13 (AMC 12) 

The complex numbers _z_ and _w_ satisfy _z_[13] = _w, w_[11] = _z_ , and the imaginary part of _z_ is sin _[mπ] n_[, for relatively prime] positive integers _m_ and _n_ with _m < n_ . Find _n_ .’ 

Answer: 

...... 

40 

Table 18: Exact match examples from the test set of MATH benchmark found in CommonCrawl by line-level exact match deduplication (in _italics_ ). In these examples, we only observe repeated questions from MATH, but do not identify duplicate answers. 

_Let x and y be real numbers satisfying x_[4] _y_[5] + _y_[4] _x_[5] = 810 _and x_[3] _y_[6] + _y_[3] _x_[6] = 945 _. Evaluate_ 2 _x_[3] + ( _xy_ )[3] + 2 _y_[3] _._ Let _x_ 1 _< x_ 2 _< x_ 3 be the three real roots of the equation ~~_√_~~ 2014 _x_[3] _−_ 4029 _x_[2] + 2 = 0. Find _x_ 2( _x_ 1 + _x_ 3). Let _m_ be the largest real solution to the equation 

3 5 17 19 _x −_ 3[+] _x −_ 5[+] _x −_ 17[+] _x −_ 19[=] _[ x]_[2] _[ −]_[11] _[x][ −]_[4] 

There are positive integers _a_ , _b_ , and _c_ such that _m_ = _a_ + ~~�~~ _b_ + ~~_[√]_~~ _c_ . Find _a_ + _b_ + _c_ . Let _f_ ( _x_ ) = _x_[4] + _ax_[3] + _bx_[2] + _cx_ + _d_ . If _f_ ( _−_ 1) = _−_ 1, _f_ (2) = _−_ 4, _f_ ( _−_ 3) = _−_ 9, and _f_ (4) = _−_ 16. Find _f_ (1). Solve in positive integers _x_[2] _−_ 4 _xy_ + 5 _y_[2] = 169. Solve in integers the question _x_ + _y_ = _x_[2] _− xy_ + _y_[2] . Solve in integers _x_[2] _−xxy_ + _y_ + _y_[2][=][3] 7 Prove the product of 4 consecutive positive integers is a perfect square minus 1. For any arithmetic sequence whose terms are all positive integers, show that if one term is a perfect square, this sequence must have infinite number of terms which are perfect squares. Prove there exist infinite number of positive integer _a_ such that for any positive integer _n_ , _n_[4] + _a_ is not a prime number. 

## ...... 

_The real root of the equation_ 8 _x_[3] _−_ 3 _x_[2] _−_ 3 _x −_ 1 = 0 _can be written in the form_ ~~_[√]_~~ 3 _a_ + _c[√]_[3] _b_ +1 _, where a, b, and c are posit ive integers. Find a_ + _b_ + _c._ 

Find the number of positive integers _m_ for which there exist nonnegative integers _x_ 0, _x_ 1 , _. . ._ , _x_ 2011 such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math_images/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math.pdf-0041-07.png)


Suppose _x_ is in the interval [0 _,[π]_ 2[]][ and][ log][24 sin] _[ x]_[(24 cos] _[ x]_[) =][3] 2[.][Find][ 24 cot][2] _[ x]_[.] Let _P_ ( _x_ ) be a quadratic polynomial with real coefficients satisfying _x_[2] _−_ 2 _x_ + 2 _≤ P_ ( _x_ ) _≤_ 2 _x_[2] _−_ 4 _x_ + 3 for all real numbers _x_ , and suppose _P_ (11) = 181. Find _P_ (16). 

Let ( _a, b, c_ ) be the real solution of the system of equations _x_[3] _− xyz_ = 2, _y_[3] _− xyz_ = 6, _z_[3] _− xyz_ = 20. The greatest possible value of _a_[3] + _b_[3] + _c_[3] can be written in the form _[m] n_[, where] _[ m]_[ and] _[ n]_[ are relatively prime positive] integers. Find _m_ + _n_ . 

Find the smallest positive integer _n_ with the property that the polynomial _x_[4] _− nx_ + 63 can be written as a product of two nonconstant polynomials with integer coefficients. 

The zeros of the function _f_ ( _x_ ) = _x_[2] _− ax_ + 2 _a_ are integers. What is the sum of the possible values of _a_ ? Let _a_ , _b_ , and _c_ be three distinct one-digit numbers. What is the maximum value of the sum of the roots of the equation ( _x − a_ )( _x − b_ ) + ( _x − b_ )( _x − c_ ) = 0? 

At the theater children get in for half price. The price for 5 adult tickets and 4 child tickets is 24 _._ 50. How much would 8 adult tickets and 6 child tickets cost? 

The quadratic equation _x_[2] + _px_ + 2 _p_ = 0 has solutions _x_ = _a_ and _x_ = _b_ . If the quadratic equation _x_[2] + _cx_ + _d_ = 0 has solutions _x_ = _a_ + 2 and _x_ = _b_ + 2, what is the value of d? 

...... 

_Find the smallest positive integer n with the property that the polynomial x_[4] _− nx_ + 63 _can be written as a product of two nonconstant polynomials with integer coefficients._ The zeros of the function _f_ ( _x_ ) = _x_[2] _− ax_ + 2 _a_ are integers. What is the sum of the possible values of _a_ ? Let _a_ , _b_ , and _c_ be three distinct one-digit numbers. What is the maximum value of the sum of the roots of the equation ( _x − a_ )( _x − b_ ) + ( _x − b_ )( _x − c_ ) = 0 ? At the theater children get in for half price. The price for 5 adult tickets and 4 child tickets is 24 _._ 50. How much would 8 adult tickets and 6 child tickets cost? The quadratic equation _x_[2] + _px_ + 2 _p_ = 0 has solutions _x_ = _a_ and _x_ = _b_ . If the quadratic equation _x_[2] + _cx_ + _d_ = 0 has solutions _x_ = _a_ + 2 and _x_ = _b_ + 2, what is the value of d? PolynomialAndEquation Root Delta SpecialEquation Function NumberTheoryBasic IndeterminateEquation SqueezeMethod Pythagore anTripletFormula TrigIdentity Inequality LogicalAndReasoning AMC10/12 AIME IMO 

US International With Solutions © 2009 - 2023 Math All Star 

...... 

41 

Table 19: Exact match examples from the test set of MATH benchmark (upper) and MMLU-STEM (bottom) found in OpenWebMath by line-level exact match deduplication (in _italics_ ). In these examples, we only observe repeated questions, but do not identify duplicate answers. 

_The sum of an infinite geometric series is a positive number S, and the second term in the series is_ 1 _. What is the smallest possible value of S_ ? 

**(A)**[1+] 2 _√_ 5 **(B)** 2 **(C)** ~~_√_~~ 5 **(D)** 3 **(E)** 4 

## Problem 17 

All the numbers 2 _,_ 3 _,_ 4 _,_ 5 _,_ 6 _,_ 7 are assigned to the six faces of a cube, one number to each face. For each of the eight vertices of the cube, a product of three numbers is computed, where the three numbers are the numbers assigned to the three faces that include that vertex. What is the greatest possible value of the sum of these eight products? 

**(A)** 312 **(B)** 343 **(C)** 625 **(D)** 729 **(E)** 1680 

... 

_What is the value of b_ + _c if x_[2] + _bx_ + _c >_ 0 _only when x ∈_ ( _−∞, −_ 2) _∪_ (3 _, ∞_ ) _?_ 

May 11, 2020 

... 

_An ambulance travels at 40 mph and can follow a 20-mile route making no stops to get to the hospital. A helicopter travels at one mile per minute, and the air route is 15 miles to get to the same hospital. However, the helicopter takes three minutes for takeoff and three minutes for landing. How many fewer minutes does it take for the helicopter to complete its trip (takeoff, flight and landing) than for the ambulance to complete its trip?_ 

Apr 6, 2020 

#1 +34 0 

Keep in mind that Time=Distance/Speed 

_What is the greatest possible area of a triangular region with one vertex at the center of a circle of radius 1 and the other two vertices on the circle?_ 

A bad first step is to put the center at the origin, one point at (1,0) , and one point at (sin x, cos x). 

A start is the area of a triangle with included angle expression 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math_images/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math.pdf-0042-17.png)


Assuming _θ_ in radians. If theta is _π/_ 2 then we have a right triangle. Let a=b=1. Area expression is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math_images/nips_2024_MathPile_A_Billion-Token-Scale_Pretraining_Corpus_for_Math.pdf-0042-19.png)


This is maximum for _θ_ = _π/_ 2. 

Answer is maximum area for a right triangle. 

... 

42 

## **G Evaluation of Continal Pre-trained Models on General Langauge Benchmarks** 

Does continual pre-training on MATHPILE lead to improvements in general language benchmarks? To explore this, we evaluated some continual pre-trained models on MATHPILE (in Table 4) in several representative benchmarks including PIQA [8], ARC-Easy [15], ARC-Challenge [15], SciQ [64] and HellaSwag [69]. For these evaluations, we used the infrastructure provided by EleutherAI’s lm-evaluation-harness,[7] and following common practices, we report the accuracy (acc norm) metric. 

Table 20: Performance of continual pre-trained models on general language benchmarks 

|**Models**|**PiQA**|**ARC-Challenge**|**ARC Easy**|**SciQ**|**Hallswag**|
|---|---|---|---|---|---|
|Mistral-7B|81.93|53.75|79.58|94.0|81.05|
|+ Textbooks|80.14|52.73|79.92|95.6|81.15|
|+ Wikipedia|80.57|56.48|79.71|94.8|82.07|
|+ Stackexchange|80.41|49.66|75.38|90.5|82.87|



One important point that needs to be emphasised is that when enhancing a model’s capabilities in a specific domain, it is typically necessary to mix the new domain-specific training data with the original training data, which helps to prevent catastrophic forgetting. In current experiments, we conducted continual pre-training exclusively on math-specific data, which means that we did not necessarily expect improvements in general language modeling benchmarks and, in some cases, a regression could occur. As shown in Table 20, after continual pre-training on MATHPILE subsets, the model’s general language modeling abilities did not experience significant degradation. In fact, there were some improvements on certain benchmarks, though some metrics did see slight declines. 

> 7https://github.com/EleutherAI/lm-evaluation-harness 

43 

