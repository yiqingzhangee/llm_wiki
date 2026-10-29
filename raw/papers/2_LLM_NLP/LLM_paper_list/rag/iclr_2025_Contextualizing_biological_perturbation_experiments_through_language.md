# **Contextualizing biological perturbation experiments through language** 

**Menghua Wu** _[∗]_ Department of Computer Science Massachusetts Institute of Technology Cambridge, MA, USA rmwu@mit.edu 

## **Russell Littman, Jacob Levine, Lin Qiu, Tommaso Biancalani, David Richmond, Jan-Christian Hütter** 

Biology Research | AI Development Genentech South San Francisco, CA, USA 

{littman.russell,levine.jacob,qiu.lin,richmond.david, biancalani.tommaso,huetter.janchristian-klaus}@gene.com 

## **Abstract** 

High-content genetic perturbation experiments provide insights into biomolecular pathways at unprecedented resolution, yet experimental and analysis costs pose barriers to their widespread adoption. _In-silico_ modeling of unseen perturbations has the potential to alleviate this burden by leveraging prior knowledge to enable more efficient exploration of the perturbation space. However, current knowledgegraph approaches neglect the semantic richness of the relevant biology, beyond simple adjacency graphs. To enable holistic modeling, we hypothesize that natural language is an appropriate medium for interrogating experimental outcomes and representing biological relationships. We propose PERTURBQA as a set of real-world tasks for benchmarking large language model (LLM) reasoning over structured, biological data. PERTURBQA is comprised of three tasks: prediction of differential expression and change of direction for unseen perturbations, and gene set enrichment. As a proof of concept, we present SUMMER (SUMMarize, retrievE, and answeR), a simple LLM-based framework that matches or exceeds the current state-of-the-art on this benchmark. We evaluated graph and language-based models on differential expression and direction of change tasks, finding that SUMMER performed best overall. Notably, SUMMER’s outputs, unlike models that solely rely on knowledge graphs, are easily interpretable by domain experts, aiding in understanding model limitations and contextualizing experimental outcomes. Additionally, SUMMER excels in gene set enrichment, surpassing over-representation analysis baselines in most cases and effectively summarizing clusters lacking a manual annotation. 

## **1 Introduction** 

Large-scale high-content genetic perturbation experiments provide insight into the function of genes and enable the verification of biological hypotheses at an unprecedented scale (Replogle et al., 2022; 

> _∗_ Work completed during an internship at Genentech. 

Foundation Models for Science Workshop,38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language_images/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language.pdf-0002-00.png)


**----- Start of picture text -----**<br>
A) PERTURBATION EXPERIMENTS B) BIOLOGICAL KNOWLEDGE GRAPHS C) PERTURB QA<br>Real-valued gene expression data Genes / pathways Description of gene: Differential expression<br>Genes ATP binding cassette subfamily E member 1  Is a knockdown of  ABCE1  in K562 cells<br>Control Quaternary structure:  (Microbial infection)  likely to result in of  GABARAP ? differential expression<br>Interacts with HIV-1<br>proteins Vif and Gag Direction of change<br>Function:  [...]<br>Differential gene  Perturbations Is a knockdown of  ABCE1  in K562 cells<br>expression analysis likely to result  decrease or increase  of<br>Biological relationships GABARAP ?<br>Relations to other genes / pathways:<br>- ABCE1 enables 4 iron, 4 sulfur cluster binding,  5 single-cell transcriptomics datasets<br>ATP binding, ATP hydrolysis activity, CTPase<br>Gene set enrichment UpDown activity, GTPase activity, [...]- Based on evidence from experimental evidence in humans, ABCE1 may physically interact with  Gene set enrichment How are the following genes  related , and<br>Gene sets N/A EIF1AX, EIF3A, EIF3B, EIF3C, EIF3D, [...]- In the mitochondrial matrix, ABCE1 enables  why do they  perturbationrespond similarly to  ?  CFLAR, VIM, CAPG, …<br>Interferon alpha/beta signaling, ABCE1 binds<br>Interpretation via discrete outcomes RNASEL, inhibiting it, Interferon alpha [...] 2 sets of annotations<br>**----- End of picture text -----**<br>


Figure 1: Overview of the input space and the PERTURBQA benchmark. A) Perturb-seq experiments result in a matrix of gene expression levels, which are converted into discrete outcomes. B) Textuallyrich, biological knowledge graphs are helpful for contextualizing experimental outcomes. C) We introduce three tasks which draw upon both sources: predicting differential expression and direction of change for _unseen_ perturbations, and summarizing data-driven gene clusters into cohesive sets. 

Ramezani et al., 2023; Fay et al., 2023). A highly interpretable way of performing these experiments is _Perturb-seq_ where genes are manipulated either at the protein- or transcription-level and single-cell gene-expression levels are measured in a pooled format (Dixit et al., 2016; Datlinger et al., 2017; Replogle et al., 2022). Yet, when applied at genome-scale, single-cell sequencing poses a significant cost burden and yields datasets of varying statistical power (Nadig et al., 2024). These challenges motivate _in-silico_ approaches for predicting cellular responses to _unseen_ perturbations to guide more efficient experimental design, and methods for automatically extracting high-level insights from perturbation screens. 

Current approaches for perturbation response prediction generalize to unseen perturbations by relating them to those perturbations that have been seen, often via knowledge graphs (Roohani et al., 2023). However, converting complex biology to structured graphs inevitably leads to loss of information, as textually-rich relationships between biological entities (e.g. genes) extracted from the literature are reduced to adjacency matrices. Furthermore, these methods are trained to regress the change in levels of genes upon perturbation: a task that is a precursor, but does not directly translate to downstream analyses like differential gene expression (Love et al., 2014) and gene set enrichment (Subramanian et al., 2005). Finally, most existing methods are black-box, revealing little about the learned biology without post-hoc probing. 

We posit that language is a natural medium for traversing the structured, biological knowledge relevant to perturbation experiments. To test this hypothesis, we propose PERTURBQA, a set of biological tasks that query discrete outcomes of perturbation experiments through question-answering. For example, rather than predicting the real-valued change in gene, we might ask: “does perturbation _p_ cause differential expression of _g_ ?” These tasks are inspired by the standard analysis pipeline for extracting insights from perturbation experiments and evaluated over five high-quality single-cell RNA sequencing (scRNA-seq) datasets with CRISPR interference (CRISPRi) perturbations (Replogle et al., 2022; Nadig et al., 2024). While state-of-the-art graph and language-based methods attain reasonable performance on this benchmark, we find that these tasks are still far from solved. 

To demonstrate that language-based reasoning is effective on these tasks, we develop a simple LLMbased framework that matches or exceeds the current state-of-the-art on PERTURBQA. SUMMER (SUMMarize, retrievE, and answeR) is an inference-time strategy that uses knowledge graph relationships and existing experimental results to guide chain-of-thought reasoning (Wei et al., 2022) regarding perturbation outcomes. Specifically, an LLM is first asked to _summarize_ textual descriptions associated with genes or gene sets, as well as their relationships to other biological entities. In addition to “featurizing” genes, this step can be applied iteratively to characterize clusters of genes that exhibit similar responses or effects. Next, based on knowledge graph proximity, we _retrieve_ related perturbation-gene pairs from existing experimental data. Inspired by retrieval-augmented 

2 

generation for language, this step helps ground the LLM’s reasoning and prevent hallucinations (Gao et al., 2024). Finally, the LLM _answers_ the biological question via chain-of-thought prompting (Wei et al., 2022) with respect to the knowledge graph summaries and retrieved examples. To summarize, our contributions are three-fold. 

1. PERTURBQA is a set of _real-world_ and _currently unsolved_ tasks that evaluate how large language models reason over structured data. We hope that this benchmark will increase the accessibility of _in-silico_ modeling for biological perturbations. 

2. SUMMER is a LLM-based proof of concept that matches or exceeds state-of-the-art approaches on PERTURBQA without any finetuning, using a lightweight 8B model, while operating natively in biologist-interpretable language. 

3. We propose that perturbations should be modeled on the level of discrete outcomes, which reflect downstream analyses; and to use LLMs to reason about the outcomes of such experiments. 

## **2 Related work** 

**Perturbation response prediction** Following Roohani et al. (2023), a number of works have approached the task of unseen perturbation prediction, primarily through the regression objective. Bai et al. (2024) replaces the MLPs in GEARS with a more expressive Transformer architecture, but maintains the same objective and evaluation. Märtens et al. (2024) utilizes language-based GenePT (Chen & Zou, 2024) embeddings, which we consider as a baseline. A related, but distinct line of work predicts the effect of _seen_ perturbations in different cell populations (Bunne et al., 2023; Lotfollahi et al., 2019). Finally, there are also works that infer the effects of _combinations_ of perturbations (Roohani et al., 2023; Gaudelet et al., 2024; Lotfollahi et al., 2023), which this work does not address, but could be extended to. 

**Large language models for biology** With respect to natural language, LLMs have been applied to a diverse set of areas, including but not limited to the following. Biological question-answering is a common benchmark to assess LLM reasoning (Hendrycks et al., 2021; Hao et al., 2024a), but these works primarily focus on tasks that humans are already able to do. Hsu et al. (2024) uses LLMs to predict Gene Ontology terms (Ashburner et al., 2000) associated with known gene sets. In contrast, the focus of our gene set enrichment task is to characterize _data-driven_ gene clusters, which may not be significantly enriched for any _known_ gene set, but are of biological interest to understand. Finally, LLMs have been used towards active experimental design (Roohani et al., 2024; Huang et al., 2024). 

More broadly, there are a number of single-cell (Rosen et al., 2024; Hao et al., 2024b; Kalfon et al., 2024) and biological sequence (Lin et al., 2023; Nguyen et al., 2024; Schiff et al., 2024) foundation models, trained over raw biological data (gene count matrices, protein/DNA sequences). In this paper, we approach biological knowledge as described by natural language, but multimodal integration of foundation models could be a promising future direction (Wang et al., 2024). 

**Language-based reasoning and structured knowledge** Significant research efforts have focused on improving the reasoning capacity of LLMs. Chain of thought (Wei et al., 2022) first demonstrated that explicitly instructing LLMs to reason enables them to solve much more complex tasks, compared to directly asking for answers. Subsequent works have explored how to navigate the space of “thoughts,” or in-progress generations (Yao et al., 2022; Zhang et al., 2024). Orthogonally, knowledge graphs have been probed through language, both in the classical natural language processing literature (Guu et al., 2015) and in modern LLM literature (Jin et al., 2024; Besta et al., 2024). This work is inspired by both directions, in the context of biological applications. 

## **3 Background** 

**Modeling perturbations** A perturbation experiment can be represented by a matrix _X ∈_ R _[N][×][D]_ , where _N_ is the number of perturbations, _D_ is the number of measurable entities, and entries _xp,g ∈ X_ represent the change in levels of entity _g_ under perturbation _p_ , relative to a control _p_ 0 (Figure 1A). For example, in a CRISPRi Perturb-seq experiment (Replogle et al., 2022), the transcription of individual genes _p_ is suppressed and the resultant change _xp,g_ in all genes _g_ is measured. 

3 

Roohani et al. (2023) (GEARS) first proposed the task of predicting _unseen_ perturbation outcomes in Perturb-seq data. Given _X_ train ⊊ _X_ , whose rows correspond to perturbations _P_ train, their goal was to complete the rows _X_ test, corresponding to unseen perturbations _P_ test. To generalize to _P_ test, GEARS and subsequent works leverage knowledge graphs that relate the two sets of perturbations (Figure 1B). That is, they solve a regression problem making use of a given directed graph _G_ = ( _V, E_ ), where _V_ is a set of biological entities (e.g., genes and pathways) and _E_ is a set of relationships. 

There are several aspects of this prevailing formulation that diverge from the insights biologists extract from these experiments. First, when converting knowledge graphs into adjacency matrices, the semantics of each edge are discarded, as they are typically annotated in free text (Ashburner et al., 2000). This loss of information may negatively impact model performance, as biological knowledge graphs often contain relationships of conflicting semantics, e.g. “enables” vs. “does not enable.” Second, current machine learning methods are evaluated with respect to “differentially expressed” genes, a small number of pre-selected features per perturbation, and often trained to focus on these genes (Roohani et al., 2023; Bai et al., 2024). These genes are not known prior to actual experimentation, and their actual identity is of high biological interest. Finally, the goal of perturbation experiments is to understand the underlying biology, but current methods focus solely on recapitulating the data distribution, errors in which may propagate to downstream analyses. 

**Statistical insights** Biologists draw conclusions of the form “ _p_ impacts gene or pathway _g_ ” through statistical techniques like differential expression (Love et al., 2014) and gene set (Subramanian et al., 2005) analyses. In differential expression analysis, one assumes that _x ∼ Px_ , where _Px_ is often taken to be approximately normal (Cui & Churchill, 2003) or negative binomial (Love et al., 2014; Ahlmann-Eltze & Huber, 2020). The goal is to test between 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language_images/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language.pdf-0004-03.png)


where rejection of _H_ 0 translates to “ _g_ is differentially expressed under perturbation _p_ compared to the control perturbation _p_ 0.” Differentially expressed genes may also be assessed by their direction of change, i.e. _xp,g_ ≶ 0. 

Due to biological and technical noise, the measurement of individual genes may be unreliable, motivating statistical analyses at the level of gene _sets_ . A data-driven approach for identifying gene sets is to cluster the rows and/or columns of the expression matrix _X_ and test whether more members of well-characterized sets are present in these clusters than expected by chance (Huang et al., 2008). While these “enriched” gene sets serve as the basis for annotating data-driven clusters, they do not consider the context of each experiment, e.g., the profiled cell line. Furthermore, significance cutoffs are difficult to assess, as the inclusion or exclusion of genes in gene sets was determined manually. As a result, data-driven clusters may exhibit consistent behavior in the experiment but fail to be enriched for known biological phenomena, thus eluding annotation (Replogle et al., 2022). 

## **4 Contextualizing biological perturbations** 

Our hypothesis is that traversing biological knowledge through language not only enables us to predict perturbation effects, but also to rationalize perturbation outcomes. To validate these ideas, we develop PERTURBQA, a suite of tasks that assess language-based reasoning over structured knowledge, in the context of biological perturbations (Section 4.1). These tasks are non-trivial, both for graph-based methods and naive large language model (LLM) applications (Section 6). Thus, as a proof of concept, we present SUMMER, a simple LLM-based approach that matches or exceeds the current state-of-the-art on PERTURBQA by jointly considering the structure and semantics of knowledge graphs (Section 4.2). 

## **4.1 PERTURBQA** 

PERTURBQA is composed of three primary tasks evaluated over five real datasets (Figure 1C). These tasks reflect the experimental and computational workflow associated with perturbation experiments. 

1. **Differential expression:** Given a pair of entities ( _p, g_ ), the goal is to predict _yp,g_[de] _[∈{]_[0] _[,]_[ 1] _[}]_[,] where 0 indicates that perturbing _p_ results in no change to _g_ , and 1 indicates that perturbing _p_ results in differential expression of _g_ . 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language_images/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language.pdf-0005-00.png)


**----- Start of picture text -----**<br>
A) Label distribution for DE and Dir C) Knowledge graph size<br>K562 GO<br>RPE1 CORUM<br>HepG2 STRING<br>Jurkat 0 0 Reactome<br>K562-Set 1 1 BioPlex<br>0 10000 20000 30000 0 1000 2000 3000 0 20000 0 500000<br>Differential expression Direction of change Nodes Edges<br>B) Gene set cluster sizes and examples D) Distribution of knowledge graph entries<br>m6A mRNA methylation,<br>Perturb. nonsense-mediated decay, 10 [3] 10 [3]<br>histone acetylation, ...<br>mitochondrial respiratory<br>Gene chain, targets ofnonsense-mediated decay, 10 [1] 10 [1]<br>unfolded protein response, ...<br>0 40 80<br>0 500 0 200<br># Genes in cluster Human annotations<br># node-centric entries # entries within 1-hop<br>Count<br>**----- End of picture text -----**<br>


Figure 2: PERTURBQA dataset statistics. A) Differential expression and direction of change, label distribution across datasets. B) Distribution of genes per cluster, with examples of annotations of gene set function. C) Knowledge graph sizes. D) Number of knowledge base entries describing each gene, and relationships within 1-hop of each genes. 

2. **Direction of change:** Given a pair of entities ( _p, g_ ), the goal is to predict _yp,g_[dir] _[∈{]_[0] _[,]_[ 1] _[}]_[,] where 0 indicates that levels of _g_ decrease under perturbation _p_ , and 1 indicates that levels of _g_ increase. This task is only evaluated on pairs for which _y_[de] = 1. 

3. **Gene set enrichment:** Given a set of genes _P_ , the goal is to identify a description _s_ that characterizes why members of _P_ exert a consistent effect when perturbed (“perturbation cluster”), or respond similarly to perturbation (“gene cluster”). As ground truth, we leverage expert gene set annotations, published by the authors of Replogle et al. (2022) (for example annotations, see Table 3). 

**Perturb-seq datasets** We constructed our benchmark based on five Perturb-seq datasets, derived from Replogle et al. (2022) and Nadig et al. (2024). For each dataset, we identified differentiallyexpressed genes (DEGs) per perturbation using the Wilcoxon signed-rank test (Wilcoxon, 1945), resulting in pairs ( _p, g_ ) with associated labels _yp,g_[de][and] _[ y] p,g_[dir][.][Datasets are split 75:25 into train and] test along the perturbation axis, with similar distribution of number of DEGs. To ensure label quality, we set a rigorous cut-off for DEGs and non-DEGs based on consistency across biological replicates and/or adjusted p-value (details and statistical analyses in Appendix A.2). The label distribution on the test set is depicted in Figure 2A. 

Differential expression and direction of change are assessed at the granularity of single genes (K562, RPE1, HepG2, Jurkat) and gene sets (K562-Set), where the gene set is represented as single entities, with the mean expression of their constituents. Gene set enrichment is evaluated over K562-Set, where human annotations are taken as the ground truth. Figure 2B illustrates the distribution of the cluster sizes and example annotations. For example clusters, see Table 3. 

**Domain knowledge** PERTURBQA tests whether LLM frameworks can effectively leverage structured domain knowledge and contextual information. Thus, in addition to test examples, we provide: 

1. Harmonized knowledge graphs, with identifiers aligned to the perturbation data (Figure 2C). These provide high-quality, biological insights to aid reasoning. 

2. Train examples, to be used as a retrieval corpus or for downstream finetuning. These are crucial for contextualizing the predictions on each dataset, as perturbation responses differ by cell line (Nadig et al., 2024). 

Specifically, for each gene _v_ , we have prepared textual descriptions _{tv}_ , as well as textual relationships to other biological entities, _{tv→v′}_ for ( _v, v[′]_ ) _∈ E_ . To ensure coverage of poorly characterized 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language_images/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language.pdf-0006-00.png)


**----- Start of picture text -----**<br>
A) SUMMARIZE B) RETRIEVE C) ANSWER<br>P LLM P P'P'12 G'1 G'2 Input (from step A) Perturbed gene: Gene of interest:  ABCE1GABARAP :  is … is … P G<br>Neighborhood of perturbation Llama3 (70B) Perturbation summary ABCE1 is … G'P1 P'1 P'2 P G'2 Examples (from step B) Example 1:Example 2: …Perturbed gene: Gene of interest: Knockdown of RPS14  RPS14FAM83Adoes not impact :  is … is …  FAM83A. P'1 G'1<br>G G<br>Chain-of-thought (CoT) prompts :<br>1) Which perturbed genes are most similar to  ABCE1 ?<br>Llama3 (70B)LLM GABARAP is … G RPS14P'1 FAM83AG'1 P'2 G'2 2) What are the downstream effects of perturbing 3) Which downstream genes are most similar to 4) What are the upstream perturbations that affect  GABARAPABCE1GABARAP  or similar?? ?<br>G is … is …<br>Final query:<br>Neighborhood of  Downstream  Is a knockdown of  ABCE1  in K562 cells likely to result in differential<br>downstream gene Gene summary P'1 has no impact on G'1 P'2 downregulates G'2 expression of  GABARAP ?<br>Generate summaries for Retrieve summaries and experimental  Generate prompts containing<br>every node in the graph. outcomes for neighboring nodes in  Llama3 (8B)LLM Answer summaries and retrieved outcomes<br>training set. from A & B, then query LLM.<br>**----- End of picture text -----**<br>


Figure 3: Overview of SUMMER. A) Knowledge graph entries are summarized per gene as both a perturbation _p_ and as a downstream gene _g_ . B) Given a new pair ( _p, g_ ), sample related pairs ( _p[′] , g[′]_ ) with associated experimental outcomes. C) Concatenate summaries, examples, and guiding questions as prompt for LLM. Depicted prompt edited for concision. Full prompts in Appendix C. 

genes, we extracted descriptions from the union of several knowledge graphs and databases, including: UniProt (Consortium, 2022), Ensembl (Martin et al., 2022), Gene Ontology (Ashburner et al., 2000; Consortium et al., 2023), CORUM (Tsitsiridis et al., 2022), STRING (Szklarczyk et al., 2022), Reactome (Milacic et al., 2023), and BioPlex (Huttlin et al., 2021). The first three were used for gene-centric descriptions, while the latter five were used for relationship descriptions (Gene Ontology used for both). 

## **4.2 Overview of SUMMER** 

SUMMER is an inference-time framework that consists of three steps which draw inspiration from different aspects of message-passing neural networks and large language model reasoning strategies (Figure 3). First, to “featurize” each gene, we summarize database descriptions of the gene and its relationships to other genes/pathways. To capture directionality, we generate two summaries for each gene: as a perturbation, and as a downstream gene. Next, we incorporate the “train” set of seen perturbations as a retrieval corpus, where perturbation-gene pairs are selected based on knowledge graph proximity to ( _p, g_ ). Finally, we formulate a set of questions to guide the LLM’s reasoning via chain-of-thought prompting. Full prompts may be found in Appendix C. 

**Neighborhood aggregation as summarization** Inspired by message-passing on graphs (Kipf & Welling, 2017), we characterize genes and their relationships to other biological entities by _summarizing_ their knowledge graph neighborhoods. Given a gene _v_ , we extract knowledge graph entries _{tv}_ , describing node _v_ , and _{tv→v′}_ , describing its relationships with neighbors _v[′]_ . As illustrated in Figure 3A, we concatenate all entries _t_ to generate two versions of each gene summary _sv_ . One asks for the downstream pathways that may be affected by the gene ( _v_ as perturbation _p_ ), and the other focuses on the upstream pathways that may affect the gene ( _v_ as downstream gene _g_ ). 

We can apply this step hierarchically to characterize gene sets. Specifically, to summarize cluster _P_ , we concatenate single gene summaries [ _sv_ ] _v∈P_ as input to the LLM, with an emphasis on downstream/upstream effects, as appropriate. A variant of this is used to address the gene-set enrichment task, asking the LLM for both a short description (“desc”) and a “name” for the gene set. 

**Graph-based retrieval of relevant training samples** Let _N_ ( _v_ ) represent the top _k_ = 10 nodes _v[′]_ that share the most neighbors with _v_ in _G_ . For each pair ( _p, g_ ), we sample up to 15 examples ( _p[′] , g[′]_ ) from _X_ train that match the following criteria.[2] 

1. Up to 5 examples with any _g[′]_ and where the perturbation is related to _p_ , i.e., _p[′]_ = _p, p[′] ∈ N_ ( _p_ ). 

> 2The number of examples was chosen heuristically, so that all input prompts and potential outputs fit within the Llama3 8k token context window. 

6 

Table 1: Results on differential expression and direction of change as binary prediction. AUROC is computed over the predictions associated with each downstream gene, and averaged over these genes (macro AUROC). Standard deviation is reported over 3 runs (where applicable) or 3 rounds of sub-sampling. For more details, see Appendix A.4. 

|Task|Model<br>K562<br>RPE1<br>HepG2<br>Jurkat<br>K562-Set|
|---|---|
|DE|MLP<br>0_._50_±._02<br>0_._50_±._01<br>0_._51_±._00<br>0_._49_±._01<br>0_._50_±._00<br>GAT<br>0_._55_±._02<br>0_._57_±._02<br>0_._57_±._02<br>0_._55_±._03<br>0_._54_±._01<br>GEARS<br>0_._54_±._01<br>0_._50_±._01<br>0_._48_±._02<br>0_._51_±._01<br>0_._49_±._01|
||GENEPT-GENE<br>0_._57_±._02<br>0_._54_±._00<br>0_._55_±._02<br>0_._55_±._01<br>0_._58_±._01<br>GENEPT-PROT<br>0_._57_±._01<br>0_._56_±._00<br>0_._54_±._01<br>0_._55_±._01<br>0_._58_±._01<br>LLM(No CoT)<br>0_._52_±._01<br>0_._51_±._00<br>0_._51_±._01<br>0_._52_±._00<br>0_._50_±._00<br>LLM(No retrieval)<br>0_._51_±._01<br>0_._48_±._00<br>0_._49_±._01<br>0_._49_±._01<br>0_._50_±._01<br>Retrieval (No LLM)<br>0_._58_±._02<br>**0.58**_±._01<br>0_._55_±._00<br>0_._55_±._01<br>**0.64**_±._00|
||SUMMER<br>**0.60**_±._00<br>**0.58**_±._00<br>**0.61**_±._00<br>**0.58**_±._00<br>0_._61_±._00|
|Dir|MLP<br>0_._51_±._02<br>0_._52_±._01<br>0_._52_±._04<br>0_._50_±._05<br>0_._49_±._02<br>GAT<br>0_._58_±._06<br>0_._60_±._04<br>0_._64_±._05<br>0_._59_±._04<br>0_._53_±._03<br>GEARS<br>**0.64**_±._01<br>0_._60_±._01<br>0_._52_±._01<br>0_._51_±._01<br>0_._59_±._02|
||GENEPT-GENE<br>0_._53_±._05<br>0_._57_±._03<br>0_._58_±._03<br>0_._57_±._02<br>0_._56_±._02<br>GENEPT-PROT<br>0_._57_±._01<br>0_._57_±._02<br>0_._55_±._01<br>0_._58_±._03<br>0_._57_±._02<br>LLM(No CoT)<br>0_._50_±._01<br>0_._49_±._00<br>0_._49_±._00<br>0_._50_±._01<br>0_._50_±._01<br>LLM(No retrieval)<br>0_._49_±._04<br>0_._52_±._03<br>0_._51_±._06<br>0_._53_±._05<br>0_._45_±._18<br>Retrieval (No LLM)<br>0_._50_±._00<br>0_._50_±._00<br>0_._50_±._00<br>0_._50_±._00<br>0_._50_±._00|
||SUMMER<br>0_._62_±._01<br>**0.64**_±._01<br>**0.65**_±._00<br>**0.66**_±._01<br>**0.69**_±._01|



2. Up to 5 examples with any _p[′]_ and where the downstream gene is related to _g_ , i.e., _g[′]_ = _g_ , or _g[′] ∈ N_ ( _g_ ). 

3. Up to 5 examples where both the perturbation as well as the downstream gene are related to _p_ and _g_ , resp., i.e., _p[′]_ = _p, p[′] ∈ N_ ( _p_ ) and _g[′]_ = _g, g[′] ∈ N_ ( _g_ ). 

These pairs are inserted into the prompt through their respective summaries ( _sp′, sg′_ ) and the groundtruth experimental outcome _yp′,g′_ (Figure 3B, and examples in Appendix C.2). 

**Perturbation outcome prediction as question-answering** We introduce a simple strategy to ensure that the LLM’s reasoning incorporates both knowledge graph summaries and retrieved experimental outcomes. For both _p_ and _g_ , we ask the LLM to identify the most similar _p[′]_ and _g[′]_ among those sampled, and to summarize their downstream/upstream effects. Finally, we ask the LLM to answer the overall question, while citing specific examples. A sketch of the overall prompt is depicted in Figure 3C (for full template, see Appendix C; for an input/output pair example, see Appendix D.2). 

**Implementation details** We ran all experiments with Llama3 (AI@Meta, 2024) with default parameters of top _p_ 0.9 and temperature 0.6, using the LMDeploy framework (Contributors, 2023). Due to computational limitations, the gene summaries were generated by the 70B model, while all other inference utilized the 8B model. To quantify uncertainty, we ran the retrieval step three times and report the average prediction over these three trials. 

## **5 Experimental setup** 

## **5.1 Baselines** 

**Differential expression and direction of change** We compare against a variety of baselines for unseen perturbation response prediction. All baselines were run with their published code and best reported hyperparameters, where applicable. We ran hyperparameter search for our implementations of MLP and GAT (Appendix A.4). **MLP** is a multi-layer perceptron with access to gene identities 

7 

Table 2: Gene set enrichment on K562 genome-wide clusters. Metrics reported are ROUGE-1 recall, as well as BERT Score precision, recall, and F1, computed with BioBERT-1.2. Since the baselines are statistical methods, they are not subject to stochasticity. 

|Enrichment<br>Top<br>Gene Ontology<br>5<br>Gene Ontology<br>10<br>Reactome<br>5<br>Reactome<br>10<br>CORUM<br>5<br>CORUM<br>10<br>Combined<br>5<br>Combined<br>10|Gene clusters<br>Perturbation clusters<br>_R_ROUGE1_↑_<br>_P_BERT_↑_<br>_R_BERT_↑_<br>_F_BERT_↑_<br>_R_ROUGE1_↑_<br>_P_BERT_↑_<br>_R_BERT_↑_<br>_F_BERT_↑_<br>0_._17<br>0_._64<br>0_._66<br>0_._62<br>0_._38<br>0_._66<br>0_._72<br>0_._68<br>0_._32<br>0_._60<br>0_._65<br>0_._60<br>0_._60<br>0_._62<br>0_._71<br>0_._65<br>0_._18<br>0_._60<br>0_._65<br>0_._60<br>0_._49<br>0_._60<br>0_._68<br>0_._62<br>0_._27<br>0_._54<br>0_._64<br>0_._56<br>0_._59<br>0_._56<br>0_._67<br>0_._60<br>0_._07<br>0_._63<br>0_._45<br>0_._42<br>0_._45<br>0_._64<br>0_._63<br>0_._60<br>0_._07<br>0_._61<br>0_._44<br>0_._41<br>0_._47<br>0_._61<br>0_._62<br>0_._58<br>0_._14<br>0_._62<br>0_._65<br>0_._61<br>0_._41<br>0_._63<br>0_._71<br>0_._66<br>0_._27<br>0_._59<br>0_._65<br>0_._59<br>**0.63**<br>0_._57<br>0_._69<br>0_._62|Gene clusters<br>Perturbation clusters<br>_R_ROUGE1_↑_<br>_P_BERT_↑_<br>_R_BERT_↑_<br>_F_BERT_↑_<br>_R_ROUGE1_↑_<br>_P_BERT_↑_<br>_R_BERT_↑_<br>_F_BERT_↑_<br>0_._17<br>0_._64<br>0_._66<br>0_._62<br>0_._38<br>0_._66<br>0_._72<br>0_._68<br>0_._32<br>0_._60<br>0_._65<br>0_._60<br>0_._60<br>0_._62<br>0_._71<br>0_._65<br>0_._18<br>0_._60<br>0_._65<br>0_._60<br>0_._49<br>0_._60<br>0_._68<br>0_._62<br>0_._27<br>0_._54<br>0_._64<br>0_._56<br>0_._59<br>0_._56<br>0_._67<br>0_._60<br>0_._07<br>0_._63<br>0_._45<br>0_._42<br>0_._45<br>0_._64<br>0_._63<br>0_._60<br>0_._07<br>0_._61<br>0_._44<br>0_._41<br>0_._47<br>0_._61<br>0_._62<br>0_._58<br>0_._14<br>0_._62<br>0_._65<br>0_._61<br>0_._41<br>0_._63<br>0_._71<br>0_._66<br>0_._27<br>0_._59<br>0_._65<br>0_._59<br>**0.63**<br>0_._57<br>0_._69<br>0_._62|
|---|---|---|
|SUMMER(8b)<br>desc<br>SUMMER(8b)<br>name<br>SUMMER(70b)<br>desc<br>SUMMER(70b)<br>name|**0.57**<br>0_._63<br>0_._76<br>0_._69<br>0_._20<br>**0.74**<br>0_._76<br>**0.75**<br>0_._45<br>0_._63<br>**0.77**<br>0_._69<br>0_._15<br>0_._73<br>0_._76<br>0_._74|0_._26<br>0_._63<br>0_._75<br>0_._68<br>0_._12<br>0_._75<br>0_._76<br>0_._75<br>0_._59<br>0_._65<br>0_._80<br>0_._72<br>0_._37<br>**0.77**<br>**0.82**<br>**0.79**|



Table 3: Gene set enrichment on K562, example outputs. Left: Human annotation. Middle: Top overrepresented gene sets (combined). Right: Proposed gene set name in bold, followed by description. Row 1 is a gene cluster. Rows 2-3 are perturbation clusters. 

|Genes|Annotation|Gene sets|SUMMER(8B) output|
|---|---|---|---|
|||(combined)||
|SLC25A5,|translation|cytosol, RNA|**Ribosomal and Translation Regulation Gene Set**The|
|EIF4B,||binding, cytoplasm,|gene set is comprised of genes involved in protein synthe-|
|SLC25A3,||membrane, GTP|sis, translation, and ribosomal function, with a focus on|
|RPL3, ...,||hydrolysis|the regulation of protein synthesis and the assembly of|
|RPL41|||ribosomal subunits.|
|ZC3H13,|m6A mRNA|nucleus, nucleo|**M6A Methylation Complex-Associated Genes.** The|
|CBLL1,|methylation|plasm, RNA N6-|gene set is composed of genes involved in the regulation|
|METTL14,||methyladenosine|of N6-methyladenosine (m6A) methylation of RNAs, in-|
|METTL3,||methyltransferase|fuencing mRNA splicing and RNA processing. These|
|PSMG1,||complex, mRNA|genes are associated with the WMM complex and interact|
|RBM15||processing|with each other to modulate gene expression.|
|CMTR2,|unknown|no signifcant sets|**RNA Processing and Regulation Gene Set.** The gene|
|RBM14-|||set is composed of genes involved in RNA processing and|
|RBM4,|||regulation, including mRNA cap modifcation, alternative|
|RBM4,|||splicing, and RNA-binding activities. These genes con-|
|UNCX,|||verge on pathways related to mRNA stability, translation,|
|WDFY3|||and cellular differentiation.|



only, trained with a ternary classification objective (no change, up, down). This baseline does _not_ consider knowledge graphs at all. **GEARS** (Roohani et al., 2023) is a state-of-the-art Graph Attention Network trained over the Gene Ontology (Ashburner et al., 2000) and gene co-expression graphs with a _regression_ objective that focuses on differentially expresses genes. Absolute predicted log-fold change is normalized to (0 _,_ 1) for differential expression, and signed log-fold change is normalized for direction of change. **GAT** (Velickovic et al., 2017) is a Graph Attention Network trained with a ternary classification objective over the same knowledge graphs used to generate the prompts for SUMMER. This setup quantifies the information content contained in graph adjacencies alone. 

We also consider language-centric baselines and ablations for SUMMER. **GENEPT** (Chen & Zou, 2024) encodes textual descriptions of genes (-GENE) and their protein (-PROT) products using commercial OpenAI embedding models, trained on natural language. These embeddings are input to a logistic regression classifier, trained separately for differential expression and direction of change. Recent work has reported that this baseline achieves the state-of-the-art on the regression formulation (Märtens et al., 2024). **LLM (No CoT)** provides the LLM with two examples (one of 

8 

each label) and directly asks for the final answer without explanation. **LLM (No retrieval)** emulates our chain-of-thought style questioning, but does not retrieve any experimental outcomes. Instead, we provide the LLM with a hypothesis (each answer option is sampled twice) and ask the LLM to extract supporting and refuting evidence from the gene summaries, before answering the question. Finally, to understand the information content in our retrieved samples, **Retrieval (No LLM)** takes the mean label over ( _p[′] , g[′]_ ) without appealing to the LLM for further processing. 

**Gene set enrichment** We compare to gene set over-expression analysis (Fang et al., 2022), run over the gene clusters with a variety of gene set libraries – Gene Ontology, Reactome, CORUM – as well as their union (Combined). We take the concatenation of the top _k_ gene set names as the predicted summary, where gene sets are ordered by the size of their intersection with each cluster. 

## **5.2 Metrics** 

For **differential expression** and **direction of change** , it has been reported that gene responses tend to be correlated across perturbations (Kernfeld et al., 2023), e.g., stress response genes respond promiscuously. Thus, we compute binary AUROC over the predictions associated with each downstream gene, and take the average over downstream genes, corresponding to a macro AUROC score over downstream genes. For **gene set enrichment** , our ground truth consists of short textual descriptions (under 10 words), while our predictions and baseline outputs are long and vary in style. Standard text generation metrics like BLEU (Papineni et al., 2002) or ROUGE F1 (Lin, 2004) do not account for this difference in length as they were designed for machine translation. To assess whether the predictions adequately cover the ground truth annotation without penalizing for longer lengths, we report ROUGE-1 recall and BERT Score (Zhang et al., 2020), using BioBERT 1.2 (Lee et al., 2020), which was finetuned on 1M biological texts (18B words). 

## **6 Results** 

## **6.1 Differential expression and direction of change** 

We evaluated both graph and language-based models on the differential expression (DE) and direction of change (Dir) tasks (Table 1). SUMMER performs well overall, while other baselines vary in performance depending on the dataset. These results indicate that PERTURBQA tasks are largely unsolved, and may benefit from larger models or more sophisticated reasoning strategies. Of note, compared to models that exclusively operate on knowledge graphs without making use of textual representations, SUMMER’s outputs are immediately interpretable by domain experts (see AppendixD) which can help in understanding model shortcomings and contextualize the observed experimental outcomes beyond the predictive task considered here. 

GENEPT is a strong baseline, demonstrating the benefits of textual information towards these tasks. In terms of language-based reasoning, however, we observe that LLM (No CoT) and LLM (No retrieval) both perform no better than random guessing – highlighting that retrieving existing experimental outcomes and guiding LLM reasoning are both essential to completing this task. This is also reflected in the strong performance of Retrieval (No LLM). While SUMMER is able to extract more value than “sum of its (retrieved) parts” in 7 of 10 cases, direct access to numerical reasoning is valuable. GEARS performance is close to random on 4 of the 5 differential expression datasets. This may be due to the focus on differentially-expressed examples in their objective, which was designed to recapitulate real-valued changes in DEGs, rather than distinguish between DEGs and non-differentially expressed genes. On other hand, GEARS is decent at direction of change in 3 of 5 cases, reflecting that its directionality loss may be more effective here. 

## **6.2 Gene set enrichment** 

SUMMER is able to recapitulate human cluster annotations, exceeding over-representation analysis baselines across the majority of cases (Table 2). In particular, SUMMER recovers the human annotations much better than baselines on gene clusters, which are much larger than the perturbation clusters (Figure 2B) and are enriched for more generic gene sets (Table 3, row 1). Finally, SUMMER coherently summarizes clusters for which no gene sets were enriched, and thus could not be annotated manually (Table 3, row 3). 

9 

## **7 Conclusion** 

In this work, we proposed PERTURBQA, a benchmark for language-based reasoning over structured data arising from real biological problems. We evaluated current state-of-the-art methods on these tasks and showed that while feasible, they are far from solved. To address these tasks, we introduced SUMMER, a LLM-based framework that draws upon both biological knowledge graphs and existing experimental data. SUMMER outperforms baselines on PERTURBQA, but leaves ample room for future study. We hope that this work will lower the barrier of entry into computational modeling of biological perturbation experiments and enable richer, more interpretable methods for these applications. 

## **Acknowledgments and Disclosure of Funding** 

We would like to thank Romain Lopez, Alexander Wu, Heming Yao, Patrick Skillman-Lawrence, Xiaotian Ma, Taro Makino, Martin Rohbeck, and Aviv Regev for continuous feedback while working on this project; and Sandra Melo-Carlos and Jack Kamm for an introduction to Perturb-seq data sets and guidance on data processing. 

This work was funded by Genentech. M.W., R.L., J.L., L.Q., T.B., D.R., and J.-C. H. were employees of Genentech while working on this project. J.L., L.Q., T.B., D.R., and J.-C. H. have equity in Roche. 

## **References** 

- Constantin Ahlmann-Eltze and Wolfgang Huber. glmGamPoi: Fitting Gamma-Poisson generalized linear models on single cell count data. _Bioinformatics_ , 36(24):5701–5702, 12 2020. ISSN 1367-4803. doi: 10.1093/bioinformatics/btaa1009. 

- AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/llama3/ blob/main/MODEL_CARD.md. 

- Michael Ashburner, Catherine A Ball, Judith A Blake, David Botstein, Heather Butler, J Michael Cherry, Allan P Davis, Kara Dolinski, Selina S Dwight, Janan T Eppig, et al. Gene Ontology: tool for the unification of biology. _Nature genetics_ , 25(1):25–29, 2000. 

- Ding Bai, Caleb N Ellington, Shentong Mo, Le Song, and Eric P Xing. AttentionPert: Accurately modeling multiplexed genetic perturbations with multi-scale effects. _Bioinformatics_ , 40:i453–i461, 06 2024. ISSN 1367-4811. doi: 10.1093/bioinformatics/btae244. 

- Yoav Benjamini and Yosef Hochberg. On the adaptive control of the false discovery rate in multiple testing with independent statistics. _Journal of Educational and Behavioral Statistics_ , 25(1):60–83, 2000. ISSN 10769986, 19351054. 

- Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michał Podstawski, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. Graph of Thoughts: Solving elaborate problems with Large Language Models. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 38(16):17682–17690, Mar 2024. doi: 10.1609/aaai. v38i16.29720. 

- Charlotte Bunne, Stefan G Stark, Gabriele Gut, Jacobo Sarabia Del Castillo, Mitch Levesque, Kjong-Van Lehmann, Lucas Pelkmans, Andreas Krause, and Gunnar Rätsch. Learning single-cell perturbation responses using neural optimal transport. _Nature methods_ , 20(11):1759–1768, 2023. 

- Yiqun Chen and James Zou. GenePT: A simple but effective foundation model for genes and cells built from ChatGPT. _bioRxiv_ , 2024. doi: 10.1101/2023.10.16.562533. 

- The Gene Ontology Consortium, Suzi A Aleksander, James Balhoff, Seth Carbon, J Michael Cherry, Harold J Drabkin, Dustin Ebert, Marc Feuermann, Pascale Gaudet, Nomi L Harris, David P Hill, Raymond Lee, Huaiyu Mi, Sierra Moxon, Christopher J Mungall, Anushya Muruganugan, Tremayne Mushayahama, Paul W Sternberg, Paul D Thomas, Kimberly Van Auken, Jolene Ramsey, Deborah A Siegele, Rex L Chisholm, Petra Fey, Maria Cristina Aspromonte, Maria Victoria Nugnes, Federica Quaglia, Silvio Tosatto, Michelle Giglio, Suvarna Nadendla, Giulia Antonazzo, Helen 

10 

Attrill, Gil dos Santos, Steven Marygold, Victor Strelets, Christopher J Tabone, Jim Thurmond, Pinglei Zhou, Saadullah H Ahmed, Praoparn Asanitthong, Diana Luna Buitrago, Meltem N Erdol, Matthew C Gage, Mohamed Ali Kadhum, Kan Yan Chloe Li, Miao Long, Aleksandra Michalak, Angeline Pesala, Armalya Pritazahra, Shirin C C Saverimuttu, Renzhi Su, Kate E Thurlow, Ruth C Lovering, Colin Logie, Snezhana Oliferenko, Judith Blake, Karen Christie, Lori Corbani, Mary E Dolan, Harold J Drabkin, David P Hill, Li Ni, Dmitry Sitnikov, Cynthia Smith, Alayne Cuzick, James Seager, Laurel Cooper, Justin Elser, Pankaj Jaiswal, Parul Gupta, Pankaj Jaiswal, Sushma Naithani, Manuel Lera-Ramirez, Kim Rutherford, Valerie Wood, Jeffrey L De Pons, Melinda R Dwinell, G Thomas Hayman, Mary L Kaldunski, Anne E Kwitek, Stanley J F Laulederkind, Marek A Tutaj, Mahima Vedi, Shur-Jen Wang, Peter D’Eustachio, Lucila Aimo, Kristian Axelsen, Alan Bridge, Nevila Hyka-Nouspikel, Anne Morgat, Suzi A Aleksander, J Michael Cherry, Stacia R Engel, Kalpana Karra, Stuart R Miyasato, Robert S Nash, Marek S Skrzypek, Shuai Weng, Edith D Wong, Erika Bakker, Tanya Z Berardini, Leonore Reiser, Andrea Auchincloss, Kristian Axelsen, Ghislaine Argoud-Puy, Marie-Claude Blatter, Emmanuel Boutet, Lionel Breuza, Alan Bridge, Cristina Casals-Casas, Elisabeth Coudert, Anne Estreicher, Maria Livia Famiglietti, Marc Feuermann, Arnaud Gos, Nadine Gruaz-Gumowski, Chantal Hulo, Nevila Hyka-Nouspikel, Florence Jungo, Philippe Le Mercier, Damien Lieberherr, Patrick Masson, Anne Morgat, Ivo Pedruzzi, Lucille Pourcel, Sylvain Poux, Catherine Rivoire, Shyamala Sundaram, Alex Bateman, Emily Bowler-Barnett, Hema Bye-A-Jee, Paul Denny, Alexandr Ignatchenko, Rizwan Ishtiaq, Antonia Lock, Yvonne Lussi, Michele Magrane, Maria J Martin, Sandra Orchard, Pedro Raposo, Elena Speretta, Nidhi Tyagi, Kate Warner, Rossana Zaru, Alexander D Diehl, Raymond Lee, Juancarlos Chan, Stavros Diamantakis, Daniela Raciti, Magdalena Zarowiecki, Malcolm Fisher, Christina James-Zorn, Virgilio Ponferrada, Aaron Zorn, Sridhar Ramachandran, Leyla Ruzicka, and Monte Westerfield. The Gene Ontology knowledgebase in 2023. _Genetics_ , 224(1): iyad031, 03 2023. ISSN 1943-2631. doi: 10.1093/genetics/iyad031. 

- The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2023. _Nucleic Acids Research_ , 51(D1):D523–D531, 11 2022. ISSN 0305-1048. doi: 10.1093/nar/gkac1052. 

- LMDeploy Contributors. LMDeploy: A toolkit for compressing, deploying, and serving LLMs. https://github.com/InternLM/lmdeploy, 2023. 

- Xiangqin Cui and Gary Churchill. Statistical tests for differential expression in cDNA microarray experiments. _Genome biology_ , 4:210, 02 2003. doi: 10.1186/gb-2003-4-4-210. 

- Paul Datlinger, André F Rendeiro, Christian Schmidl, Thomas Krausgruber, Peter Traxler, Johanna Klughammer, Linda C Schuster, Amelie Kuchler, Donat Alpar, and Christoph Bock. Pooled CRISPR screening with single-cell transcriptome readout. _Nature methods_ , 14(3):297–301, 2017. 

- Atray Dixit, Oren Parnas, Biyu Li, Jenny Chen, Charles P Fulco, Livnat Jerby-Arnon, Nemanja D Marjanovic, Danielle Dionne, Tyler Burks, Raktima Raychowdhury, et al. Perturb-seq: dissecting molecular circuits with scalable single-cell RNA profiling of pooled genetic screens. _cell_ , 167(7): 1853–1866, 2016. 

- Zhuoqing Fang, Xinyuan Liu, and Gary Peltz. GSEApy: a comprehensive package for performing gene set enrichment analysis in Python. _Bioinformatics_ , 39(1):btac757, 11 2022. ISSN 1367-4811. doi: 10.1093/bioinformatics/btac757. 

- Marta M Fay, Oren Kraus, Mason Victors, Lakshmanan Arumugam, Kamal Vuggumudi, John Urbanik, Kyle Hansen, Safiye Celik, Nico Cernek, Ganesh Jagannathan, et al. RXRX3: Phenomics map of biology. _Biorxiv_ , pp. 2023–02, 2023. 

- Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for Large Language Models: a survey, 2024. 

- Thomas Gaudelet, Alice Del Vecchio, Eli M Carrami, Juliana Cudini, Chantriolnt-Andreas Kapourani, Caroline Uhler, and Lindsay Edwards. Season combinatorial intervention predictions with Salt & Peper, 2024. 

- K. Guu, J. Miller, and P. Liang. Traversing knowledge graphs in vector space. In _Empirical Methods in Natural Language Processing (EMNLP)_ , 2015. 

11 

- Minsheng Hao, Jing Gong, Xin Zeng, Chiming Liu, Yucheng Guo, Xingyi Cheng, Taifeng Wang, Jianzhu Ma, Le Song, and Xuegong Zhang. Lab-bench: Measuring capabilities of language models for biology research. _Nature Methods_ , 2024a. doi: 10.1038/s41592-024-02305-7. 

- Minsheng Hao, Jing Gong, Xin Zeng, Chiming Liu, Yucheng Guo, Xingyi Cheng, Taifeng Wang, Jianzhu Ma, Le Song, and Xuegong Zhang. Large-scale foundation model on single-cell transcriptomics. _Nature Methods_ , 2024b. doi: 10.1038/s41592-024-02305-7. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2021. 

- Chi-Yang Hsu, Kyle Cox, Jiawei Xu, Zhen Tan, Tianhua Zhai, Mengzhou Hu, Dexter Pratt, Tianlong Chen, Ziniu Hu, and Ying Ding. Thought Graph: Generating thought process for biological reasoning. In _Companion Proceedings of the ACM on Web Conference 2024_ , WWW ’24. ACM, May 2024. doi: 10.1145/3589335.3651572. 

- Da Wei Huang, Brad T. Sherman, and Richard A. Lempicki. Bioinformatics enrichment tools: Paths toward the comprehensive functional analysis of large gene lists. _Nucleic Acids Research_ , 37:1 – 13, 2008. 

- Kaixuan Huang, Yuanhao Qu, Henry Cousins, William A. Johnson, Di Yin, Mihir Shah, Denny Zhou, Russ Altman, Mengdi Wang, and Le Cong. CRISPR-GPT: An LLM agent for automated design of gene-editing experiments, 2024. 

- Edward L. Huttlin, Raphael J. Bruckner, Jose Navarrete-Perea, Joe R. Cannon, Kurt Baltier, Fana Gebreab, Melanie P. Gygi, Alexandra Thornock, Gabriela Zarraga, Stanley Tam, John Szpyt, Brandon M. Gassaway, Alexandra Panov, Hannah Parzen, Sipei Fu, Arvene Golbazi, Eila Maenpaa, Keegan Stricker, Sanjukta Guha Thakurta, Tian Zhang, Ramin Rad, Joshua Pan, David P. Nusinow, Joao A. Paulo, Devin K. Schweppe, Laura Pontano Vaites, J. Wade Harper, and Steven P. Gygi. Dual proteome-scale networks reveal cell-specific remodeling of the human interactome. _Cell_ , 184 (11):3022–3040.e28, 2021. ISSN 0092-8674. doi: https://doi.org/10.1016/j.cell.2021.04.011. 

- Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. Large Language Models on graphs: a comprehensive survey, 2024. 

- Jérémie Kalfon, Jules Samaran, Gabriel Peyré, and Laura Cantini. scPRINT: pre-training on 50 million cells allows robust gene network predictions. _bioRxiv_ , 2024. doi: 10.1101/2024.07.29. 605556. 

- Eric Kernfeld, Yunxiao Yang, Joshua S. Weinstock, Alexis Battle, and Patrick Cahan. A systematic comparison of computational methods for expression forecasting. _bioRxiv_ , 2023. doi: 10.1101/ 2023.07.28.551039. 

- Thomas N. Kipf and Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. In _International Conference on Learning Representations_ , 2017. 

- Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. _Bioinformatics_ , 36(4):1234–1240, 2020. 

- Chin-Yew Lin. ROUGE: a package for automatic evaluation of summaries. In _Text Summarization Branches Out_ , pp. 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. 

- Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Salvatore Candido, and Alexander Rives. Evolutionary-scale prediction of atomic-level protein structure with a language model. _Science_ , 379(6637):1123–1130, 2023. doi: 10.1126/ science.ade2574. 

- Mohammad Lotfollahi, F. Alexander Wolf, and Fabian J Theis. scGen predicts single-cell perturbation responses. _Nature Methods_ , 16:715 – 721, 2019. 

12 

- Mohammad Lotfollahi, Anna Klimovskaia Susmelj, Carlo De Donno, Leon Hetzel, Yuge Ji, Ignacio L Ibarra, Sanjay R Srivatsan, Mohsen Naghipourfar, Riza M Daza, Beth Martin, Jay Shendure, Jose L McFaline-Figueroa, Pierre Boyeau, F Alexander Wolf, Nafissa Yakubova, Stephan Günnemann, Cole Trapnell, David Lopez-Paz, and Fabian J Theis. Predicting cellular responses to complex perturbations in high-throughput screens. _Molecular Systems Biology_ , 19(6):e11517, 2023. doi: https://doi.org/10.15252/msb.202211517. 

- Michael I Love, Wolfgang Huber, and Simon Anders. Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. _Genome biology_ , 15:1–21, 2014. 

- Kaspar Märtens, Rory Donovan-Maiye, and Jesper Ferkinghoff-Borg. Enhancing generative perturbation models with LLM-informed gene embeddings. In _ICLR 2024 Workshop on Machine Learning for Genomics Explorations_ , 2024. 

- Fergal J Martin, M Ridwan Amode, Alisha Aneja, Olanrewaju Austine-Orimoloye, Andrey G Azov, If Barnes, Arne Becker, Ruth Bennett, Andrew Berry, Jyothish Bhai, Simarpreet Kaur Bhurji, Alexandra Bignell, Sanjay Boddu, Paulo R Branco Lins, Lucy Brooks, Shashank Budhanuru Ramaraju, Mehrnaz Charkhchi, Alexander Cockburn, Luca Da Rin Fiorretto, Claire Davidson, Kamalkumar Dodiya, Sarah Donaldson, Bilal El Houdaigui, Tamara El Naboulsi, Reham Fatima, Carlos Garcia Giron, Thiago Genez, Gurpreet S Ghattaoraya, Jose Gonzalez Martinez, Cristi Guijarro, Matthew Hardy, Zoe Hollis, Thibaut Hourlier, Toby Hunt, Mike Kay, Vinay Kaykala, Tuan Le, Diana Lemos, Diego Marques-Coelho, José Carlos Marugán, Gabriela Alejandra Merino, Louisse Paola Mirabueno, Aleena Mushtaq, Syed Nakib Hossain, Denye N Ogeh, Manoj Pandian Sakthivel, Anne Parker, Malcolm Perry, Ivana Piližota, Irina Prosovetskaia, José G Pérez-Silva, Ahamed Imran Abdul Salam, Nuno Saraiva-Agostinho, Helen Schuilenburg, Dan Sheppard, Swati Sinha, Botond Sipos, William Stark, Emily Steed, Ranjit Sukumaran, Dulika Sumathipala, MarieMarthe Suner, Likhitha Surapaneni, Kyösti Sutinen, Michal Szpak, Francesca Floriana Tricomi, David Urbina-Gómez, Andres Veidenberg, Thomas A Walsh, Brandon Walts, Elizabeth Wass, Natalie Willhoft, Jamie Allen, Jorge Alvarez-Jarreta, Marc Chakiachvili, Bethany Flint, Stefano Giorgetti, Leanne Haggerty, Garth R Ilsley, Jane E Loveland, Benjamin Moore, Jonathan M Mudge, John Tate, David Thybert, Stephen J Trevanion, Andrea Winterbottom, Adam Frankish, Sarah E Hunt, Magali Ruffier, Fiona Cunningham, Sarah Dyer, Robert D Finn, Kevin L Howe, Peter W Harrison, Andrew D Yates, and Paul Flicek. Ensembl 2023. _Nucleic Acids Research_ , 51(D1): D933–D941, 11 2022. ISSN 0305-1048. doi: 10.1093/nar/gkac958. 

- Marija Milacic, Deidre Beavers, Patrick Conley, Chuqiao Gong, Marc Gillespie, Johannes Griss, Robin Haw, Bijay Jassal, Lisa Matthews, Bruce May, Robert Petryszak, Eliot Ragueneau, Karen Rothfels, Cristoffer Sevilla, Veronica Shamovsky, Ralf Stephan, Krishna Tiwari, Thawfeek Varusai, Joel Weiser, Adam Wright, Guanming Wu, Lincoln Stein, Henning Hermjakob, and Peter D’Eustachio. The Reactome pathway knowledgebase 2024. _Nucleic Acids Research_ , 52(D1): D672–D678, 11 2023. ISSN 0305-1048. doi: 10.1093/nar/gkad1025. 

- Ajay Nadig, Joseph Replogle, Angela Pogson, Steven Mccarroll, Jonathan Weissman, Elise Robinson, and Luke O’Connor. Transcriptome-wide characterization of genetic perturbations. _bioRxiv_ , 07 2024. doi: 10.1101/2024.07.03.601903. 

- Eric Nguyen, Michael Poli, Matthew G. Durrant, Armin W. Thomas, Brian Kang, Jeremy Sullivan, Madelena Y. Ng, Ashley Lewis, Aman Patel, Aaron Lou, Stefano Ermon, Stephen A. Baccus, Tina Hernandez-Boussard, Christopher Ré, Patrick D. Hsu, and Brian L. Hie. Sequence modeling and design from molecular to genome scale with Evo. _bioRxiv_ , 2024. doi: 10.1101/2024.02.27.582234. 

- Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin (eds.), _Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics_ , pp. 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. 

- Meraj Ramezani, Julia Bauman, Avtar Singh, Erin Weisbart, John Yong, Maria Lozada, Gregory P Way, Sanam L Kavari, Celeste Diaz, Marzieh Haghighi, et al. A genome-wide atlas of human cell morphology. _bioRxiv_ , 2023. 

13 

- Joseph M. Replogle, Reuben A. Saunders, Angela N. Pogson, Jeffrey A. Hussmann, Alexander Lenail, Alina Guna, Lauren Mascibroda, Eric J. Wagner, Karen Adelman, Gila Lithwick-Yanai, Nika Iremadze, Florian Oberstrass, Doron Lipson, Jessica L. Bonnar, Marco Jost, Thomas M. Norman, and Jonathan S. Weissman. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. _Cell_ , 185(14):2559–2575.e28, 2022. ISSN 0092-8674. doi: https://doi.org/10.1016/j.cell.2022.05.013. 

- Yusuf Roohani, Kexin Huang, and Jure Leskovec. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. _Nature Biotechnology_ , 2023. 

- Yusuf Roohani, Jian Vora, Qian Huang, Zachary Steinhart, Alexander Marson, Percy Liang, and Jure Leskovec. BioDiscoveryAgent: An AI agent for designing genetic perturbation experiments. _arXiv preprint_ , 2024. 

- Yanay Rosen, Maria Brbi´c, Yusuf Roohani, Kyle Swanson, Li Ziang, and Jure Leskovec. Towards universal cell embeddings: Integrating single-cell RNA-seq datasets across species with SATURN. _Nature Methods_ , 2024. doi: 10.1101/2023.02.03.526939. 

- Yair Schiff, Chia-Hsiang Kao, Aaron Gokaslan, Tri Dao, Albert Gu, and Volodymyr Kuleshov. Caduceus: Bi-Directional equivariant long-range DNA sequence modeling, 2024. 

- Aravind Subramanian, Pablo Tamayo, Vamsi K. Mootha, Sayan Mukherjee, Benjamin L. Ebert, Michael A. Gillette, Amanda Paulovich, Scott L. Pomeroy, Todd R. Golub, Eric S. Lander, and Jill P. Mesirov. Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles. _Proceedings of the National Academy of Sciences_ , 102(43): 15545–15550, 2005. doi: 10.1073/pnas.0506580102. 

- Damian Szklarczyk, Rebecca Kirsch, Mikaela Koutrouli, Katerina Nastou, Farrokh Mehryary, Radja Hachilif, Annika L Gable, Tao Fang, Nadezhda T Doncheva, Sampo Pyysalo, Peer Bork, Lars J Jensen, and Christian von Mering. The STRING database in 2023: Protein–protein association networks and functional enrichment analyses for any sequenced genome of interest. _Nucleic Acids Research_ , 51(D1):D638–D646, 11 2022. ISSN 0305-1048. doi: 10.1093/nar/gkac1000. 

- George Tsitsiridis, Ralph Steinkamp, Madalina Giurgiu, Barbara Brauner, Gisela Fobo, Goar Frishman, Corinna Montrone, and Andreas Ruepp. CORUM: the comprehensive resource of mammalian protein complexes – 2022. _Nucleic Acids Research_ , 51(D1):D539–D545, 11 2022. ISSN 0305-1048. doi: 10.1093/nar/gkac1015. 

- Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, Yoshua Bengio, et al. Graph Attention Networks. _stat_ , 1050(20):10–48550, 2017. 

- Marie-Claire Wagle, Daniel Kirouac, Christiaan Klijn, Bonnie Liu, Shilpi Mahajan, Melissa Junttila, John Moffat, Ling Huw, Matthew Wongchenko, Kwame Okrah, Shrividhya Srinivasan, Zineb Mounir, Teiko Sumiyoshi, Peter Haverty, Robert Yauch, Yibing Yan, Omar Kabbarah, Garret Hampton, and Shih-Min Huang. A transcriptional MAPK pathway activity score (MPAS) is a clinically relevant biomarker in multiple cancer types. _npj Precision Oncology_ , 2, 12 2018. doi: 10.1038/s41698-018-0051-4. 

- Zifeng Wang, Zichen Wang, Balasubramaniam Srinivasan, Vassilis N Ioannidis, Huzefa Rangwala, and Rishita Anubhai. BioBridge: Bridging biomedical foundation models via knowledge graphs. In _International Conference on Learning Representations_ , 2024. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-Thought prompting elicits reasoning in Large Language Models. _36th Conference on Neural Information Processing Systems_ , 2022. 

- Frank Wilcoxon. Individual comparisons by ranking methods. _Biometrics Bulletin_ , 1(6):80–83, 1945. ISSN 00994987. 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of Thoughts: Deliberate problem solving with Large Language Models, 2022. 

- Di Zhang, Xiaoshui Huang, Dongzhan Zhou, Yuqiang Li, and Wanli Ouyang. Accessing GPT-4 level mathematical olympiad solutions via monte carlo tree self-refine with LLaMa-3 8B, 2024. 

14 

- Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. BERTScore: Evaluating text generation with BERT. In _International Conference on Learning Representations_ , 2020. 

15 

## **A Experimental details** 

## **A.1 K562 gene sets** 

We downloaded K562 genome-wide gene sets from the supplementary data associated with Figure 4B of Replogle et al. (2022). These gene sets were obtained by “cluster[ing] genes into expression programs based on their co-regulation.” We evaluated gene set enrichment over clusters that were manually annotated, though we provide predictions over all gene sets (regardless of annotation status). 

For differential expression analysis, we took the average of the Log(TP10k+1) values over each gene set, for each cell, similar to a pathway activity score (Wagle et al., 2018). 

## **A.2 Data processing** 

We normalized all gene counts to Log(TP10k+1) values (log-transformed UMI count per 10k), where the count _cij_ of gene _j_ in cell _i_ is mapped to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language_images/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language.pdf-0016-06.png)


To determine differentially expressed genes (DEGs), we ran the Wilcoxon signed-rank test (Wilcoxon, 1945) with Benjamini-Hochberg correction (Benjamini & Hochberg, 2000) between non-targeting control (NTC) cells and perturbed cells, for each perturbation. 

To focus on perturbations with phenotypic effect, we excluded all perturbations that resulted in _≤_ 5 DEGs ( _p <_ 0 _._ 05), with the exception of 100 negative control perturbations (0 DEGs), sampled uniformly at random per dataset. 

Exploratory analyses revealed that DEG identities were not entirely consistent across biological replicates, so we further restricted our selection of DEGs. For the two K562 datasets, we defined “DE” perturbation - gene pairs as those shared between the datasets at _p <_ 0 _._ 05. Non differentiallyexpressed pairs were sampled from those that were _not_ differentially-expressed in _either_ dataset. Since the remaining datasets lacked biological replicates, we defined “DE” pairs as those with _p <_ 0 _._ 01, and sampled “non-DE” pairs from those with _p >_ 0 _._ 1. We selected the top 20 DEGs per perturbation ranked by adjusted p-value as “positives.” We sampled 100 non-DEGs per perturbation as “negatives.” 

To ensure similar distributions between our training and testing splits, we sorted both selected perturbations and DEGs based on prevalence. We split perturbations 75:25 between training and testing. Validation data were sampled at random during training (10% of training). 

Further details regarding dataset and data split statistics may be found in Tables 4 and 5. 

Table 4: Data statistics. K562 _[∗]_ non-targeting control cells were subsampled due to the size of the genome-wide dataset. K562-es _[⋆]_ (essential) was only used to filter K562 DE and non-DE genes. 

|Dataset<br>K562_∗_<br>K562-es_⋆_<br>RPE1<br>HepG2<br>Jurkat|Cells<br>Control<br>Perturbed<br>5,000<br>919,124<br>10,691<br>299,645<br>11,485<br>236,164<br>4,976<br>140,497<br>12,013<br>250,943|Perturbations<br>Total<br>Train<br>Test<br>9851<br>1564<br>267<br>2049<br>—<br>—<br>2354<br>1596<br>406<br>2393<br>1086<br>278<br>2392<br>1227<br>313|Features<br>4136<br>—<br>4760<br>7435<br>6842|
|---|---|---|---|
|K562-Set|5,000<br>919,124|9851<br>1401<br>357|20|



## **A.3 LLM details** 

Due to the stochastic nature of LLM generations, we noticed that the LLM would occasionally abstain from selecting one of the intended labels, due to insufficient evidence for either. To account for this, 

16 

Table 5: Differential gene expression data split statistics. Number of perturbation - gene (set) pairs in each setting. 

|Dataset|Split|Total|non-DE|Differentially expressed|Differentially expressed|Differentially expressed|
|---|---|---|---|---|---|---|
|||||Total|Up|Down|
|K562|Train|134,467|117,606|16,861|11,041|5,820|
||Test|23,212|20,093|3,119|2,530|589|
|RPE1|Train|149,147|127,860|21,287|8,381|12,906|
||Test|37,942|32,577|5,365|2,121|3,244|
|HepG2|Train|101,140|86,883|14,257|6,249|8,008|
||Test|25,749|22,146|3,603|1,599|2,004|
|Jurkat|Train|113,684|97,747|15,937|5,119|10,818|
||Test|29,138|25,017|4,121|1,379|2,742|
|K562-Set|Train|20,606|9,367|11,239|3,953|7,286|
||Test|5,235|2,403|2,832|995|1,837|



Table 6: Abstain rate on differential expression (DE) and direction of change (Dir) across all datasets. 

|Model|DE|Dir|
|---|---|---|
|LLM(No Retrieval)|0.02|0.36|
|LLM(No CoT)|3_._3_×_10_−_6|0|
|SUMMER|8_._9_×_10_−_4|0.03|



we intentionally added “insufficient information” as a third answer option. We ran inference on each input sample at least 3 times and took the mean predicted label, after removing all abstaining outputs. 

A small fraction of inputs ( _p, g_ ) resulted in no predictions after this filtering, or were unable to be parsed by our rule-based parsing. The latter is due to the insufficient capacity of Llama3 8B (relatively small LLM) to follow instructions. Since these examples differed by model, we substituted the prediction with an uninformed baseline (the mean label of _g_ over the training set) for evaluation. The final abstain rate varied based on LLM prompting strategy (Table 6). LLM (No Retrieval) abstained nearly a third of the time on direction of change. In contrast, LLM (No CoT) only abstained a single time, over all datasets. The improved instruction following may be due to the concise nature of the expected output (only a single answer). Finally, SUMMER nearly always produced a prediction over 3 runs (e.g. 0.08% abstain on DE). 

## **A.4 Baselines** 

Table 7 documents the search space for MLP and GAT. We selected the top models based on validation performance (arbitrary 10% of train). Both models were implemented with GELU activation, dropout of 0.1, weight decay 1e-6, learning rate 1e-4, and residual connections. GAT employed additional edge attributes, which were learned embeddings that indicated the source knowledge graph of each edge. 

For K562-Set, we pooled the mean embedding of each gene set’s genes before the prediction head (MLP and GAT). On GEARS, we used the mean predicted log-fold change over each gene set’s genes (mirrors data pre-processing). 

A small number of genes (97 out of 11,234) did not map to GENEPT embeddings. We set the embeddings for these genes to the mean perturbation / gene embedding in their respective training sets. 

For uncertainty quantification, we used the top 3 runs for MLP and GAT. Since GEARS operates over single cells, rather than pseudo-bulk estimates, we subsampled 80% of the single cells before taking the average for each of 3 evaluations. For GENEPT, since logistic regression does not inherently 

17 

Table 7: MLP and GAT hyperparameter search space 

|Model|Parameter|Space|
|---|---|---|
||Layers|0, 1, 2, 4|
|MLP|Hidden size|256, 512, 1024|
||FFN size|1024, 2048|
||Layers|1, 2, 4, 8|
|GAT|Hidden size|64, 128, 256|
||FFN size|1024|



introduce randomness (unless it fails to converge; it always converges here), we subsampled 80% of the training set for each of 3 runs. 

## **B Differential gene expression** 

We provide empirical analyses on the quality of our datasets and labels. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language_images/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language.pdf-0018-05.png)


**----- Start of picture text -----**<br>
K562 0.06 K562-es 0.06 RPE1 0.06 HepG2 0.06 Jurkat<br>0.06<br>0.04 0.04 0.04 0.04 0.04<br>0.02 0.02 0.02 0.02 0.02<br>0.00 0.00 0.00 0.00 0.00<br>0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0<br>P-values P-values P-values P-values P-values<br>Probability<br>**----- End of picture text -----**<br>


Figure 4: Assessing p-value calibration over single-cell datasets. We split the non-targeting controls (NTCs) randomly in half, and run the Wilcoxon test to compare the two halves. We would expect to see that the (non-adjusted) p-values are uniformly distributed between 0 and 1. Here, we see that the Wilcoxon test is slightly conservative, i.e. it leans towards reporting “non-differentially expressed.” 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language_images/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language.pdf-0018-07.png)


**----- Start of picture text -----**<br>
800 Overlap of top 5 sets Overlap of top 5 sets Overlap of top 10 sets Overlap of top 10 sets<br>800 600<br>600 600<br>600<br>400<br>400 400<br>400<br>200 200 200 200<br>0 0 0 0<br>0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00<br>% es in gw % gw in es % es in gw % gw in es<br>Count Count Count Count<br>**----- End of picture text -----**<br>


Figure 5: K562 gene clusters show consistent response between biological replicates. We compute the top _k_ = 5 _,_ 10 significant gene clusters, sorted by adjusted p-value, for both K562 genome-wide and K562 essential. For each perturbation, we compute the percentage of shared gene clusters (normalizing by genome-wide and essential, respectively). We see that the clusters are relatively consistent across both datasets, with a high fraction of perfect overlaps. 

## **C Prompts** 

We share the prompts used by each stage of SUMMER, as well as the LLM baselines. All text surrounded by {...} are replaced with example-specific inputs (Appendix D). 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language_images/iclr_2025_Contextualizing_biological_perturbation_experiments_through_language.pdf-0019-00.png)


**----- Start of picture text -----**<br>
% Top 5 es DEGs in gw % Top 5 gw DEGs in es Overlap of top 10 sets Overlap of top 10 sets<br>600<br>1000<br>1500 600<br>800 400<br>400<br>1000 600<br>400 200 200<br>500<br>200<br>0 0<br>0 0 0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00<br>0.0 0.5 1.0 0.0 0.5 1.0 % es in gw % gw in es<br>2000% Top 20 es DEGs in gw % Top 20 gw DEGs in es % Top 100 es DEGs in gw % Top 100 gw DEGs in es<br>400<br>600 1500<br>1500<br>300<br>400 1000<br>1000<br>200<br>500 200 500 100<br>0 0 0 0<br>0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0<br>Count Count<br>Count Count<br>Count Count Count Count<br>**----- End of picture text -----**<br>


Figure 6: K562 DEGs are reasonably consistent at the top (ranked by p-value). For each perturbation, we plot the percentage of shared top = 5 _,_ 10 _,_ 20 _,_ 100 DEGs (normalizing by genome-wide and essential, respectively). In our final dataset, we took the intersection of the top 20 DEGs as “DE” and the intersection of the negatives as “non-DE.” 

## **C.1 Summarization templates** 

We generate gene summaries at two levels: single node and 1-hop knowledge graph neighborhoods. For both levels, we use two prompts per gene (as a perturbation, and as a downstream gene). Wherever gene sets are the downstream entities, we replace “gene” with “gene set.” 

The following is an example of a single-node summary of a perturbed gene. 

You are an expert molecular biologist who studies how genes are related using Perturb-seq. 

Task: You are writing a brief overview of the human gene {gene}, with a focus on its molecular and cellular functions. You will be provided a set of database entries about the gene. Ensure that your overview remains faithful to this domain knowledge. 

Format: 

- Write one to two sentences describing the primary molecular and cellular function of gene {gene}. 

- Write one sentence describing the potential downstream impact of perturbing gene {gene} via gene knockdown. 

Constraints: 

- Maintain a professional tone throughout. 

- Do not comment on your own writing. 

- Do not add any notes or references. Do not make up additional information. 

- Do not discuss the importance or impact of the gene. Focus only on its function. 

- Domain knowledge: {description} 

Brief overview of gene {gene}: 

To generate downstream gene descriptions, we replace the second formatting instruction with the following. 

Write one sentence describing what types of perturbations might impact the expression of gene {gene}. For example, you might consider pathways that are upstream of the gene or compensatory mechanisms. 

19 

Single-node summaries are input alongside additional knowledge graph relationships for 1-hop neighborhood summaries. 

You are an expert molecular biologist who studies how genes are related using Perturb-seq. 

Task: You are writing a brief overview of the human gene {gene}, with a focus on the downstream effects of perturbing {gene} via gene knockdown (loss of function). 

Inputs: You are provided 

- Description of perturbed gene {gene} 

- Database entries relating {gene} to other genes or pathways 

Format: Write up to five sentences describing the molecular and cellular impact of perturbing gene {gene} via gene knockdown. 

Constraints: 

- Remain faithful to all domain knowledge. Do not make up additional information. 

- Summarize all common aspects succinctly, but point out notable differences within these sets 

- of genes. 

- Maintain a professional tone throughout. Do not comment on your own writing. Do not add 

- any notes or references. 

- Omit the importance or impact of the gene. Focus only on its function. 

- Omit all non-specific information and obvious statements, e.g. “this gene is involved in cellular processes.” 

Description of gene {gene}: {single-node summary} 

Relations to other genes: {relationships} 

Downstream effects of perturbing {gene} via gene knockdown: 

To generate downstream gene descriptions, we replace the “Task” instruction with the following. 

Task: You are writing a brief overview of the human gene {gene}, with a focus on molecular and cellular perturbations that may affect the levels of gene {gene}. For example, you might consider pathways that are upstream of the gene or compensatory mechanisms. 

To generate gene set summaries (for differential expression and direction of change), we used the following prompt, where the manual annotation is provided as input. 

You are an expert molecular biologist who studies how genes and pathways are related using Perturb-seq. 

Task: You are writing a brief overview of a gene set associated with {manual annotation} in human cells. You will be provided descriptions of the constituent genes. Ensure that your overview remains faithful to this domain knowledge. 

Format: 

- Write one to two sentences summarizing how the given genes are related. Be sure to characterize how they are associated with {manual annotation}. 

- Write one to two sentences describing what types of perturbations might impact the expression of the genes in this gene set. For example, you might consider pathways that are upstream of these genes or compensatory mechanisms. 

Constraints: 

- Maintain a professional tone throughout. 

- Do not comment on your own writing. 

- Do not add any notes or references. Do not make up additional information. 

- Do not discuss the importance or impact of the gene set. Focus only on its function. 

20 

Descriptions of the constituent genes: {list of gene summaries} 

Brief overview of gene set ({manual annotation}): 

Finally, for the gene set enrichment task, we use the following prompt for hierarchical summarization of gene sets. Note that the manual annotations are _not_ provided here. We parse the outputs of “Brief overview of gene set” and “Name of gene set” as the description and name in our evaluation. 

[Start of Prompt] 

You are an expert molecular biologist who studies how genes and pathways are related using Perturb-seq. 

Task: You are writing a brief overview of a gene set observed to have a similar transcriptional response when upstream genes are perturbed. 

Input: You will be provided descriptions of the constituent genes. Ensure that your overview remains faithful to this domain knowledge. 

Output format: Fill in each of these three sections. 

1) Brief overview of gene set: Write one to two sentences summarizing how the given genes are related. Focus on the most specific pathways that are common among these genes. 

2) Upstream pathways may affect this gene set: Write one to two sentences describing what types of perturbations might impact the expression of the genes in this gene set. For example, you might consider pathways that are upstream of these genes or compensatory mechanisms. 

3) Name of gene set: Summarize the gene set within ten words. 

Constraints: 

- Maintain a professional tone throughout. 

- Do not comment on your own writing. 

- Do not add any notes or references. Do not make up additional information. 

- Do not discuss the importance or impact of the gene set. Focus only on its function. 

[End of Prompt] 

[Start of Input] {list of gene summaries} [End of Input] 

## **C.2 Question-answering templates** 

For differential expression and direction of change, we used the following template for SUMMER. 

[Start of Prompt] 

You are an expert molecular biologist who studies how genes are related using Perturb-seq. Your goal is to determine: Does a CRISPRi knockdown of {perturbation} in {cell line} result in differential expression of {gene}? 

You are given as input: 

- Description of perturbed gene ({perturbation}): description of gene that is perturbed via 

- CRISPRi knockdown 

- Description of gene of interest ({gene}): description of gene, the impact on which you wish to 

- infer 

- Context: description of cell line in which the genes are expressed 

- Examples: set of experimental observations that describe the impact of CRISPRi perturbations on related genes 

Output: Please extract the most relevant parts of the examples that address these five questions. Be specific. 

1) Which of the observed perturbed genes are most similar to {perturbation} (if any, including {perturbation} itself)? 

21 

2) When perturbing {perturbation} or similar genes, what downstream pathways or genes are differentially expressed? Justify your answer with the observed outcomes. 

3) Which of the observed genes of interest are most similar to {gene} (if any, including {gene} itself)? 

4) What perturbations of upstream pathways or genes result in differential expression of {gene} or similar genes (if any)? Justify your answer with the observed outcomes. 

5) Is a CRISPRi knockdown of {perturbation} in {cell line} likely to result in differential expression of {gene}? For example, if 2) and 4) are unrelated or only indirectly related, it is unlikely we will observe differential expression. On the other hand, if 2) and 4) significantly overlap in specific genes or pathways, we may observe differential expression. Your final answer should end with one of these three options and nothing else. 

- No. Knockdown of {perturbation} does not impact {gene}. 

- Yes. Knockdown of {perturbation} results in differential expression of {gene}. 

- There is insufficient evidence to determine how knockdown of {perturbation} affects {gene}. 

[End of Prompt] 

[Start of Input] 

- Description of perturbed gene ({perturbation}): {summary of perturbation} 

- Description of gene of interest ({gene}): {summary of downstream gene} 

- Context: {sentence describing cell line} 

- Examples: {list of examples} 

[End of Input] 

For direction of change, we change the answer options to the following. 

A) Knockdown of {perturbation} results in a decrease in expression of {gene}. 

B) Knockdown of {perturbation} results in an increase in expression of {gene}. 

The following template was used for the LLM (No CoT) baseline on differential expression. 

You are an expert molecular biologist who studies how genes are related using Perturb-seq. You are given as Input: 

- Perturbed gene: description of gene that is perturbed via CRISPRi knockdown 

- Gene of interest: description of gene, the impact on which you wish to infer 

Context: {sentence describing cell line} 

Question: If you knockdown the perturbed gene using CRISPRi, how does the gene of interest’s expression change? 

Answer: Your answer must end with one of these two choices and nothing else. 

A) Knockdown of the perturbed gene does not impact the gene of interest. 

B) Knockdown of the perturbed gene results in differential expression of the gene of interest. 

Format: Follow the same format as Examples 1 and 2, and complete Example 3. 

Example 1. 

Input: 

- Perturbed gene: {summary of perturbation} 

- Gene of interest: {summary of downstream gene} 

Answer: {either A) ... or B) ...} 

Example 2. {same format as Example 1, opposite Answer} 

Example 3. {same format as Example 1, empty Answer} 

22 

The following template was used for the LLM (No retrieval) baseline on differential expression. Both answer options are provided twice each as hypotheses, regardless of the ground truth answer. 

You are an expert molecular biologist who studies how genes are related using Perturb-seq. 

You are given as Input: 

- Perturbed gene: description of gene that is perturbed via CRISPRi knockdown 

- Gene of interest: description of gene, the impact on which you wish to infer 

- Hypothesis: hypothesis regarding how the specified perturbation affects the gene of interest Context: {sentence describing cell line} 

Question: If you knockdown the perturbed gene using CRISPRi, how does the gene of interest’s expression change? 

Task: Your goal is to identify evidence in the input that supports or refutes the hypothesis, and explain whether the hypothesis is likely to be true. 

Output format: Please fill in the following four sections. Preserve the formatting and add the corresponding content. 

1) Supporting evidence: Identify all relevant parts of the input that support the hypothesis. 

2) Refuting evidence: Identify all relevant parts of the input that refute the hypothesis. 

3) Explanation: Based on the evidence, explain how to answer the question, step by step. In particular, 

- if there is a causal relationship from the perturbed gene to the gene of interest, explain how biological mechanisms relate the perturbed gene to the gene of interest. 

- if there is no causal relationship from the perturbed gene to the gene of interest, explain why. For example, the perturbed gene may be downstream of the gene of interest, or there may be no relationship between the two genes. 

- if there is insufficient evidence to answer the question, say so. 

4) Answer: Your answer must end with one of these three choices and nothing else. 

A) Knockdown of the perturbed gene does not impact the gene of interest. 

B) Knockdown of the perturbed gene results in differential expression of the gene of interest. 

C) There is insufficient evidence to determine how knockdown of the perturbed gene affects the gene of interest. 

Input: 

- Perturbed gene: {summary of perturbation} 

- Gene of interest: {summary of downstream gene} 

- Hypothesis: {either A) ... or B) ...} 

For direction of change, we change the answer options to the following. 

A) Knockdown of the perturbed gene results in down-regulation of the gene of interest. 

B) Knockdown of the perturbed gene results in up-regulation of the gene of interest. 

## **D Examples** 

## **D.1 Summarization examples** 

Here is an example of the knowledge-base description for ABCE1, used as the {description} in the single-node summary. 

Description of gene: ATP binding cassette subfamily E member 1 

Gene products: ATP-binding cassette sub-family E member 1 

23 

Functions: (Microbial infection) May act as a chaperone for post-translational events during HIV-1 capsid assembly., (Microbial infection) Plays a role in the down-regulation of the 2-5A/RNASEL pathway during encephalomyocarditis virus (EMCV) and HIV-1 infections., Nucleoside-triphosphatase (NTPase) involved in ribosome recycling by mediating ribosome disassembly. 

Quaternary structure: (Microbial infection) Interacts with HIV-1 proteins Vif and Gag. (Microbial infection) Interacts with HIV-2 protein Gag. Interacts with PINK1. 

Interacts with: ABCE1, EIF3J, GAG_HV1B1, MATRX_CHAV, PHOSP_RABVE 

Biological process: enables endoribonuclease inhibitor activity (Binds to and stops, prevents or reduces the activity of endoribonuclease.) 

Here is the corresponding output for the single-node summary, which is used as input to the 1-hop neighborhood summary. 

The primary molecular and cellular function of gene ABCE1 is to facilitate ribosome recycling by mediating ribosome disassembly through its nucleoside-triphosphatase (NTPase) activity, and to act as a chaperone for post-translational events during HIV-1 capsid assembly. Additionally, ABCE1 plays a role in the down-regulation of the 2-5A/RNASEL pathway during viral infections. Perturbing gene ABCE1 via gene knockdown may disrupt ribosome recycling and viral capsid assembly, potentially leading to aberrant protein synthesis and altered viral replication dynamics. 

Here is an example of the knowledge graph relationships for ABCE1. 

Relations to other genes: 

- ABCE1 enables 4 iron, 4 sulfur cluster binding, ATP binding, ATP hydrolysis activity, CTPase activity, GTPase activity, endoribonuclease inhibitor activity, iron ion binding, ribonucleoside triphosphate phosphatase activity, ribosomal small subunit binding. 

- ABCE1 is active in cytosolic ribosome 

- ABCE1 is involved in negative regulation of endoribonuclease activity, regulation of translation, rescue of stalled ribosome, ribosomal subunit export from nucleus, ribosome disassembly, translational initiation, translational termination, cytoplasm, cytosol, membrane, mitochondrial matrix, mitochondrion. 

- Based on evidence from experimental evidence in humans, database evidence in humans, ABCE1 may physically interact with RNASEL. 

- Based on evidence from experimental evidence in humans, experimental evidence in other animals, ABCE1 may physically interact with EIF1AX, EIF3A, EIF3B, EIF3C, EIF3D, EIF3E, EIF3F, EIF3G, EIF3H, EIF3I, EIF3K, EIF3L, EIF3M, G3BP2, LTO1, MFGE8, PSMD14, RACK1, RPL12, RPL23, RPL4, RPL7A, RPL9, RPL9P7, RPL9P8, RPL9P9, RPS10, RPS10NUDT3, RPS11, RPS12, RPS13, RPS14, RPS15, RPS15A, RPS16, RPS17, RPS18, RPS19, RPS2, RPS20, RPS21, RPS24, RPS25, RPS26, RPS27, RPS27A, RPS28, RPS29, RPS3, RPS3A, RPS4X, RPS5, RPS6, RPS7, RPS8, RPS9, RPSA, YAE1. 

- Based on evidence from experimental evidence in humans, experimental evidence in other animals, literature evidence in humans, ABCE1 may physically interact with EIF5, RPS23. 

- Based on evidence from experimental evidence in humans, experimental evidence in other animals, literature evidence in humans, literature evidence in other animals, ABCE1 may physically interact with EIF1, ETF1. 

- Based on evidence from experimental evidence in humans, experimental evidence in other animals, literature evidence in other animals, ABCE1 may physically interact with EIF3J, RPLP0. 

- Based on evidence from experimental evidence in humans, literature evidence in humans, ABCE1 may physically interact with DCP2, EIF2S3, FDPS, PSMD1, PSMD4. 

- Based on evidence from experimental evidence in humans, literature evidence in humans, literature evidence in other animals, ABCE1 may physically interact with EIF2S1, EIF2S2. 

- Based on evidence from experimental evidence in other animals, literature evidence in humans, ABCE1 may physically interact with UBL4A. 

24 

- Based on evidence from experimental evidence in other animals, literature evidence in other animals, ABCE1 may physically interact with DENR, EIF1B, MRTO4. 

- In 293T cells, ABCE1 may form a complex with IGF2R 

- In the cytosol, ABCE1 enables OAS antiviral response, RNASEL binds 2’-5’ oligoadenylate, ABCE1 binds RNASEL, OAS antiviral response, RNASEL binds 2’-5’ oligoadenylate, ABCE1 binds RNASEL 

- In the mitochondrial matrix, ABCE1 enables Interferon alpha/beta signaling, ABCE1 binds RNASEL, inhibiting it, Interferon alpha/beta signaling, ABCE1 binds RNASEL, inhibiting it 

Here is the corresponding output for the 1-hop neighborhood summary. 

Perturbations that may impact the levels of ABCE1 include alterations in the 2-5A/RNASEL pathway, HIV-1 capsid assembly, or ribosome recycling, as well as changes in the interactions with its binding partners, such as HIV-1 proteins Vif and Gag, HIV-2 protein Gag, PINK1, EIF3J, and MATRX_CHAV. Additionally, perturbations in the regulation of translation, translational initiation, and translational termination may also affect ABCE1 levels. Furthermore, changes in the activity of its binding partners, such as EIF1AX, EIF3A, EIF3B, and RPL12, may also impact ABCE1 expression. 

## **D.2 Question-answering examples** 

Here is an example prompting the LLM to predict whether AARS2 knockdown results in differential expression of AAK1 in K562 cells. 

[Start of Prompt] 

You are an expert molecular biologist who studies how genes are related using Perturb-seq. Your goal is to determine: Does a CRISPRi knockdown of AARS2 in K562 cells result in differential expression of AAK1? 

You are given as input: 

- Description of perturbed gene (AARS2): description of gene that is perturbed via CRISPRi knockdown 

- Description of gene of interest (AAK1): description of gene, the impact on which you wish to infer 

- Context: description of cell line in which the genes are expressed 

- Examples: set of experimental observations that describe the impact of CRISPRi perturbations on related genes, to contextualize your answer 

Output: Please extract the most relevant parts of the examples that address these five questions. Be specific. 

1) Which of the observed perturbed genes are most similar to AARS2 (if any, including AARS2 itself)? 

2) When perturbing AARS2 or similar genes, what downstream pathways or genes are differentially expressed? Justify your answer with the observed outcomes. 

3) Which of the observed genes of interest are most similar to AAK1 (if any, including AAK1 itself)? 

4) What perturbations of upstream pathways or genes result in differential expression of AAK1 or similar genes (if any)? Justify your answer with the observed outcomes. 

5) Is a CRISPRi knockdown of AARS2 in K562 cells likely to result in differential expression of AAK1? For example, if 2) and 4) are unrelated or only indirectly related, it is unlikely we will observe differential expression. On the other hand, if 2) and 4) significantly overlap in specific genes or pathways, we may observe differential expression. Your final answer should end with one of these three options and nothing else. 

- No. Knockdown of AARS2 does not impact AAK1. 

- Yes. Knockdown of AARS2 results in differential expression of AAK1. 

- There is insufficient evidence to determine how knockdown of AARS2 affects AAK1. 

25 

[End of Prompt] 

26 

## [Start of Input] 

- Description of perturbed gene (AARS2): The primary molecular and cellular function of gene AARS2 is to catalyze the attachment of alanine to tRNA(Ala) in a two-step reaction, thereby facilitating protein synthesis, and to edit incorrectly charged tRNA(Ala) via its editing domain. Additionally, AARS2 enables zinc ion binding, which is essential for its enzymatic activity. Perturbing AARS2 via gene knockdown may disrupt the proper attachment of alanine to tRNA(Ala), leading to errors in protein synthesis and potentially affecting mitochondrial function. This may also impact the editing of incorrectly charged tRNA(Ala) and zinc ion binding, essential for its enzymatic activity. As a result, mitochondrial alanyl-tRNA aminoacylation and aminoacyl-tRNA metabolism involved in translational fidelity may be impaired. Additionally, perturbing AARS2 may affect its interactions with other proteins, such as ATP5IF1, DARS2, and RARS2, potentially leading to further downstream effects on mitochondrial function. 

- Description of gene of interest (AAK1): The primary molecular and cellular function of gene AAK1 is to regulate clathrin-mediated endocytosis by phosphorylating the AP2M1/mu2 subunit of the adaptor protein complex 2 (AP-2), ensuring high affinity binding of AP-2 to cargo membrane proteins during the initial stages of endocytosis, and to play a role in the entry of certain viruses such as hepatitis C, Ebola, and Dengue. AAK1 also interacts with clathrin via its CBD domain and is involved in the regulation of protein localization. Perturbations that may impact the levels of AAK1 include alterations to the clathrin-mediated endocytosis pathway, changes in the activity of upstream regulators of AP-2, or disruptions to the interactions between AAK1 and its binding partners, such as clathrin and AT132. Additionally, perturbations to the Notch signaling pathway, protein localization, and membrane organization may also affect AAK1 levels. Furthermore, changes in the activity of proteins that physically interact with AAK1, such as AP1B1, AP2A1, AP2B1, AP2M1, AP2S1, CLTA, CLTB, CLTC, NECAP2, and RALBP1, may also influence AAK1 expression. 

- Context: K562 cells are immortalised myelogenous leukemia cells of the erythroleukemia type. 

27 

- Examples: 

## Example 1: Impact of knocking down MTIF2 on SPHK1 

Description of perturbed gene (MTIF2): The primary molecular and cellular function of gene MTIF2 is to facilitate the initiation of protein synthesis in mitochondria by protecting formylmethionyl-tRNA from spontaneous hydrolysis and promoting its binding to the 30S ribosomal subunits, ultimately leading to the formation of the 70S ribosomal complex. This process is crucial for the translation of mitochondrial DNA into functional proteins. Perturbing gene MTIF2 via gene knockdown would likely disrupt the initiation of protein synthesis in mitochondria, potentially leading to impaired mitochondrial function and protein synthesis. Description of gene of interest (SPHK1): The primary molecular and cellular function of gene SPHK1 is to catalyze the phosphorylation of sphingosine to form sphingosine 1-phosphate, a lipid mediator with both intra- and extracellular functions, and to interact with various proteins to regulate downstream cellular processes. Additionally, SPHK1 plays a role in the sphingosine-1-phosphate receptor signaling pathway, which regulates transcription and other cellular processes. Perturbations that might impact the expression of gene SPHK1 include alterations in the sphingosine-1-phosphate receptor signaling pathway, changes in calmodulin binding, or disruptions to its interactions with other proteins such as ACY1, CATB, EF1A1, FHL2, LAMP2, SHLB1, SPHKAP, or TRAF6. 

Outcome: A) Knockdown of MTIF2 does not impact SPHK1. 

## Example 2: Impact of knocking down CLPX on PTCD1 

Description of perturbed gene (CLPX): The primary molecular and cellular function of gene CLPX is to act as an ATP-dependent specificity component of the Clp protease complex, hydrolyzing ATP and forming a homohexamer ring structure that assembles with CLPP rings to form the Clp complex. This complex is involved in protein degradation and quality control in the mitochondrial matrix. Perturbing gene CLPX via gene knockdown may disrupt the proper functioning of the Clp protease complex, leading to impaired protein degradation and potential accumulation of misfolded or damaged proteins in the mitochondrial matrix. 

Description of gene of interest (PTCD1): The primary molecular and cellular function of gene PTCD1 is to negatively regulate leucine tRNA levels, mitochondria-encoded proteins, and COX activity, while also affecting the 3’-processing of mitochondrial tRNAs, thereby influencing mitochondrial protein synthesis. As a mitochondrial protein, PTCD1 associates with mitochondrial leucine tRNAs and interacts with various proteins, including ELAC2, to modulate its functions. Perturbations that might impact the expression of gene PTCD1 include disruptions to mitochondrial tRNA metabolism, alterations in COX activity, or changes in the levels of interacting proteins, such as ELAC2, which could in turn affect PTCD1’s regulatory roles in mitochondrial protein synthesis. 

## Outcome: A) Knockdown of CLPX does not impact PTCD1. 

## Example 3: Impact of knocking down TARS2 on SLC35G2 

Description of perturbed gene (TARS2): The primary molecular and cellular function of gene TARS2 is to catalyze the attachment of threonine to tRNA(Thr) in a two-step reaction, thereby enabling threonine-tRNA ligase activity, and to edit incorrectly charged tRNA(Thr) via its editing domain. This mitochondrial enzyme plays a crucial role in ensuring the accurate translation of genetic information during protein synthesis. Perturbing gene TARS2 via gene knockdown may disrupt the proper charging of tRNA(Thr) with threonine, potentially leading to errors in protein synthesis and altered cellular function. 

Description of gene of interest (SLC35G2): The primary molecular and cellular function of gene SLC35G2 is to encode a solute carrier protein that facilitates the transport of molecules across the plasma membrane, interacting with various proteins such as RAB37, AQP6, and SOAT to regulate this process. This protein plays a crucial role in maintaining cellular homeostasis by controlling the movement of solutes in and out of the cell. Perturbations that might impact the expression of gene SLC35G2 include disruptions to the Rab GTPase signaling pathway, alterations in plasma membrane composition, or changes in the activity of interacting proteins such as AQP6 and SOAT. 

Outcome: A) Knockdown of TARS2 does not impact SLC35G2. 

28 

## Example 4: Impact of knocking down MRM1 on LYRM7 

Description of perturbed gene (MRM1): The primary molecular and cellular function of gene MRM1 is to catalyze the formation of 2’-O-methylguanosine at position 1145 in the 16S mitochondrial large subunit ribosomal RNA (mtLSU rRNA) as an S-adenosyl-L-methioninedependent 2’-O-ribose methyltransferase, playing a crucial role in ribosome biogenesis. This enzymatic activity is essential for the proper modification of mitochondrial ribosomal RNA. Perturbing gene MRM1 via gene knockdown may disrupt the proper modification of mitochondrial ribosomal RNA, potentially leading to defects in mitochondrial protein synthesis. 

Description of gene of interest (LYRM7): The primary molecular and cellular function of gene LYRM7 is to act as a chaperone, binding to the Rieske Fe-S protein UQCRFS1 in the mitochondrial matrix and stabilizing it prior to its translocation and insertion into the late cytochrome b-c1 (CIII) complex, thereby facilitating the assembly of Complex III. This process enables protein folding and ensures the proper incorporation of UQCRFS1 into the mitochondrial inner membrane. Perturbations that might impact the expression of gene LYRM7 include disruptions to the mitochondrial protein folding machinery, alterations in the levels or activity of interacting proteins such as HSC20, SAT1, and UCRI, or changes in the mitochondrial inner membrane dynamics that affect the assembly of Complex III. 

## Outcome: A) Knockdown of MRM1 does not impact LYRM7. 

## Example 5: Impact of knocking down HSPD1 on RTN2 

Description of perturbed gene (HSPD1): The primary molecular and cellular function of gene HSPD1 is to facilitate the correct folding of imported proteins in the mitochondrial matrix, acting as a chaperonin to prevent misfolding and promote the refolding and proper assembly of unfolded polypeptides, particularly under stress conditions. Additionally, HSPD1 is involved in mitochondrial protein import and macromolecular assembly. Perturbing gene HSPD1 via gene knockdown may disrupt the proper folding and assembly of mitochondrial proteins, leading to potential defects in mitochondrial function and protein homeostasis. 

Description of gene of interest (RTN2): The primary molecular and cellular function of gene RTN2 is to inhibit amyloid precursor protein processing, likely by blocking the activity of BACE1, and to interact with specific proteins such as TMEM33 and certain isoforms of SPAST, within the endoplasmic reticulum membrane. This function is crucial for maintaining proper protein processing and lipid bilayer structure in the endoplasmic reticulum. Perturbations that might impact the expression of gene RTN2 include alterations to the BACE1 pathway, changes in endoplasmic reticulum membrane composition, or disruptions to the interactions with its binding partners TMEM33 and SPAST. 

## Outcome: A) Knockdown of HSPD1 does not impact RTN2. 

## Example 6: Impact of knocking down WDR82 on AAK1 

Description of perturbed gene (WDR82): The primary molecular and cellular function of gene WDR82 is to serve as a regulatory component of the SET1/COMPASS complex, facilitating the tethering of this complex to transcriptional start sites of active genes, and to negatively regulate the transcription of long non-coding RNA (lncRNA). This function is mediated through its interaction with other components of the SET1/COMPASS complex, including the catalytic subunit, WDR5, and other regulatory proteins. Perturbing gene WDR82 via gene knockdown may lead to the dysregulation of lncRNA transcription, potentially resulting in aberrant gene expression. 

Description of gene of interest (AAK1): The primary molecular and cellular function of gene AAK1 is to regulate clathrin-mediated endocytosis by phosphorylating the AP2M1/mu2 subunit of the adaptor protein complex 2 (AP-2), ensuring high affinity binding of AP-2 to cargo membrane proteins during the initial stages of endocytosis, and to play a role in the entry of certain viruses such as hepatitis C, Ebola, and Dengue. AAK1 also interacts with clathrin via its CBD domain and is involved in the regulation of protein localization. Perturbations that might impact the expression of gene AAK1 include alterations to the clathrin-mediated endocytosis pathway, changes in the activity of upstream regulators of AP-2, or disruptions to the interactions between AAK1 and its binding partners, such as clathrin and AT132. Outcome: A) Knockdown of WDR82 does not impact AAK1. 

29 

## Example 7: Impact of knocking down CARF on AAK1 

Description of perturbed gene (CARF): The primary molecular and cellular function of gene CARF is to act as a transcriptional activator that mediates the calcium- and neuron-selective induction of BDNF exon III transcription by binding to the consensus calcium-response element CaRE1 sequence, thereby regulating transcription by RNA polymerase II. This function is crucial for modulating gene expression in response to calcium signals in neurons. Perturbing gene CARF via gene knockdown may disrupt the regulation of BDNF exon III transcription, potentially affecting neuronal gene expression and response to calcium signals. 

Description of gene of interest (AAK1): The primary molecular and cellular function of gene AAK1 is to regulate clathrin-mediated endocytosis by phosphorylating the AP2M1/mu2 subunit of the adaptor protein complex 2 (AP-2), ensuring high affinity binding of AP-2 to cargo membrane proteins during the initial stages of endocytosis, and to play a role in the entry of certain viruses such as hepatitis C, Ebola, and Dengue. AAK1 also interacts with clathrin via its CBD domain and is involved in the regulation of protein localization. Perturbations that might impact the expression of gene AAK1 include alterations to the clathrin-mediated endocytosis pathway, changes in the activity of upstream regulators of AP-2, or disruptions to the interactions between AAK1 and its binding partners, such as clathrin and AT132. 

## Outcome: A) Knockdown of CARF does not impact AAK1. 

## Example 8: Impact of knocking down ELP5 on AAK1 

Description of perturbed gene (ELP5): The primary molecular and cellular function of gene ELP5 is to act as a component of the elongator complex, which is required for multiple tRNA modifications, including mcm5U, mcm5s2U, and ncm5U, and to facilitate optimal binding of ELP3 to ELP4 within the complex. Additionally, ELP5 is involved in the positive regulation of cell migration. Perturbing gene ELP5 via gene knockdown may disrupt tRNA modification and cell migration processes. 

Description of gene of interest (AAK1): The primary molecular and cellular function of gene AAK1 is to regulate clathrin-mediated endocytosis by phosphorylating the AP2M1/mu2 subunit of the adaptor protein complex 2 (AP-2), ensuring high affinity binding of AP-2 to cargo membrane proteins during the initial stages of endocytosis, and to play a role in the entry of certain viruses such as hepatitis C, Ebola, and Dengue. AAK1 also interacts with clathrin via its CBD domain and is involved in the regulation of protein localization. Perturbations that might impact the expression of gene AAK1 include alterations to the clathrin-mediated endocytosis pathway, changes in the activity of upstream regulators of AP-2, or disruptions to the interactions between AAK1 and its binding partners, such as clathrin and AT132. 

## Outcome: A) Knockdown of ELP5 does not impact AAK1. 

Example 9: Impact of knocking down MICOS13 on AAK1 Description of perturbed gene (MICOS13): The primary molecular and cellular function of gene MICOS13 is to encode a subunit of the mitochondrial contact site and cristae organizing system (MICOS) complex, which plays crucial roles in maintaining crista junctions, inner membrane architecture, and forming contact sites to the outer membrane. MICOS13 is a component of the MICOS complex, which associates with other protein complexes to form the mitochondrial intermembrane space bridging (MIB) complex. Perturbing gene MICOS13 via gene knockdown may disrupt the formation and maintenance of mitochondrial crista junctions and inner membrane architecture, potentially leading to alterations in mitochondrial function and dynamics. 

Description of gene of interest (AAK1): The primary molecular and cellular function of gene AAK1 is to regulate clathrin-mediated endocytosis by phosphorylating the AP2M1/mu2 subunit of the adaptor protein complex 2 (AP-2), ensuring high affinity binding of AP-2 to cargo membrane proteins during the initial stages of endocytosis, and to play a role in the entry of certain viruses such as hepatitis C, Ebola, and Dengue. AAK1 also interacts with clathrin via its CBD domain and is involved in the regulation of protein localization. Perturbations that might impact the expression of gene AAK1 include alterations to the clathrin-mediated endocytosis pathway, changes in the activity of upstream regulators of AP-2, or disruptions to the interactions between AAK1 and its binding partners, such as clathrin and AT132. Outcome: A) Knockdown of MICOS13 does not impact AAK1. 

30 

## Example 10: Impact of knocking down RPL10 on AAK1 

Description of perturbed gene (RPL10): The primary molecular and cellular function of gene RPL10 is to encode a component of the large ribosomal subunit, specifically the protein uL16, which is essential for protein synthesis and translation. This protein plays a crucial role in the assembly and function of mature ribosomes, facilitating the translation of messenger RNA into polypeptide chains. Perturbing gene RPL10 via gene knockdown would likely disrupt protein synthesis and translation, potentially leading to impaired cellular function and viability. 

Description of gene of interest (AAK1): The primary molecular and cellular function of gene AAK1 is to regulate clathrin-mediated endocytosis by phosphorylating the AP2M1/mu2 subunit of the adaptor protein complex 2 (AP-2), ensuring high affinity binding of AP-2 to cargo membrane proteins during the initial stages of endocytosis, and to play a role in the entry of certain viruses such as hepatitis C, Ebola, and Dengue. AAK1 also interacts with clathrin via its CBD domain and is involved in the regulation of protein localization. Perturbations that might impact the expression of gene AAK1 include alterations to the clathrin-mediated endocytosis pathway, changes in the activity of upstream regulators of AP-2, or disruptions to the interactions between AAK1 and its binding partners, such as clathrin and AT132. 

## Outcome: A) Knockdown of RPL10 does not impact AAK1. 

## Example 11: Impact of knocking down RARS2 on NECAP2 

Description of perturbed gene (RARS2): The primary molecular and cellular function of gene RARS2 is to catalyze the attachment of arginine to tRNA(Arg) in a two-step reaction, facilitating mitochondrial translation, and to interact with other proteins such as COMT, SYRM, and TGM2. This process is essential for the formation of proteins within the mitochondrion. Perturbing gene RARS2 via gene knockdown may disrupt the proper attachment of arginine to tRNA(Arg), potentially leading to impaired mitochondrial translation and protein synthesis. 

Description of gene of interest (NECAP2): The primary molecular and cellular function of gene NECAP2 is to facilitate endocytosis through its role as an adaptin ear-binding coat-associated protein, interacting with components of adapter protein complexes AP-1 and AP-2, as well as GAE domain proteins GGA1, GGA2, and GGA3. This interaction enables the formation and trafficking of vesicles, mediating the transport of substances within the cell. Perturbations that might impact the expression of gene NECAP2 include disruptions to the AP-1 and AP-2 adapter protein complexes, or alterations to the GAE domain proteins, which could affect the proper formation and function of vesicles involved in endocytosis. 

Outcome: A) Knockdown of RARS2 does not impact NECAP2. 

## Example 12: Impact of knocking down NARS2 on REPS1 

Description of perturbed gene (NARS2): The primary molecular and cellular function of gene NARS2 is to encode a mitochondrial aminoacyl-tRNA synthetase that catalyzes the attachment of asparagine to its cognate tRNA, thereby participating in protein synthesis. This enzyme enables asparagine-tRNA ligase activity, facilitating the formation of Asn-tRNA(Asn) from L- asparagine and tRNA(Asn). Perturbing gene NARS2 via gene knockdown may disrupt the proper charging of tRNA with asparagine, potentially leading to aberrant protein synthesis and mitochondrial dysfunction. 

Description of gene of interest (REPS1): The primary molecular and cellular function of gene REPS1 is to coordinate the cellular actions of activated EGF receptors and Ral-GTPases, potentially linking them to an EGF-responsive tyrosine kinase through its interactions with CRK and GRB2. Additionally, REPS1 may be involved in clathrin-mediated endocytosis through its interactions with AMPH, ITSN1, and SGIP1. Perturbations to the EGF signaling pathway, Ral-GTPase activity, or clathrin-mediated endocytosis may impact the expression of gene REPS1. 

Outcome: A) Knockdown of NARS2 does not impact REPS1. 

[End of Input] 

31 

[Start of Output] 

Based on the provided examples, I will extract the most relevant parts to address the five questions. 

1. Which of the observed perturbed genes are most similar to AARS2 (if any, including AARS2 itself)? 

From the examples, we can see that AARS2 is a mitochondrial aminoacyl-tRNA synthetase, specifically responsible for attaching alanine to tRNA(Ala). The most similar genes to AARS2 are likely to be other mitochondrial aminoacyl-tRNA synthetases, such as TARS2 (threoninetRNA ligase), RARS2 (arginine-tRNA ligase), and NARS2 (asparagine-tRNA ligase). These genes share similar functions and are also involved in protein synthesis. 

2. When perturbing AARS2 or similar genes, what downstream pathways or genes are differentially expressed? Justify your answer with the observed outcomes. 

From the examples, we can see that perturbing AARS2 or similar genes does not impact the expression of downstream genes. For instance, knocking down AARS2 does not affect the expression of SPHK1, PTCD1, SLC35G2, LYRM7, RTN2, AAK1, or NECAP2. Similarly, knocking down TARS2, RARS2, or NARS2 does not impact the expression of SLC35G2, REPS1, or NECAP2, respectively. This suggests that the perturbation of AARS2 or similar genes does not significantly affect downstream pathways or genes. 

3. Which of the observed genes of interest are most similar to AAK1 (if any, including AAK1 itself)? 

From the examples, we can see that AAK1 is a gene involved in regulating clathrin-mediated endocytosis and interacts with clathrin and other proteins. The most similar genes to AAK1 are likely to be other genes involved in clathrin-mediated endocytosis, such as NECAP2, which facilitates endocytosis through its interactions with adapter protein complexes AP-1 and AP-2. Other genes that interact with clathrin, such as RTN2, may also be similar to AAK1. 

4. What perturbations of upstream pathways or genes result in differential expression of AAK1 or similar genes (if any)? Justify your answer with the observed outcomes. 

From the examples, we can see that perturbing genes involved in clathrin-mediated endocytosis, such as NECAP2, does not impact the expression of AAK1. Similarly, perturbing genes involved in protein synthesis, such as RPL10, does not affect the expression of AAK1. This suggests that the perturbation of upstream pathways or genes does not significantly affect the expression of AAK1 or similar genes. 

5. Is a CRISPRi knockdown of AARS2 in K562 cells likely to result in differential expression of AAK1? 

Based on the observed outcomes, it is unlikely that a CRISPRi knockdown of AARS2 in K562 cells will result in differential expression of AAK1. The perturbation of AARS2 or similar genes does not impact the expression of downstream genes, including AAK1. Therefore, the answer is: 

No. Knockdown of AARS2 does not impact AAK1. 

[End of Output] 

## **D.3 Other examples** 

We use the following for {sentence describing cell line}, derived from Wikipedia. 

K562 cells are immortalised myelogenous leukemia cells of the erythroleukemia type. 

RPE1 cells are a non-cancerous, hTERT-immortalized, near-euploid, adherent, and p53-positive cell line. 

Jurkat cells are an immortalized line of human T lymphocyte cells. 

HepG2 cells are a human liver cancer cell line, derived from a patient with a well-differentiated hepatocellular carcinoma. 

32 

