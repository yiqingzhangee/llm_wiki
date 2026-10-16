Published as a conference paper at ICLR 2024 

# TOWARDS GENERATIVE ABSTRACT REASONING: COMPLETING RAVEN’S PROGRESSIVE MATRIX VIA RULE ABSTRACTION AND SELECTION 

**Fan Shi Bin Li** _[∗]_ **Xiangyang Xue** Shanghai Key Laboratory of Intelligent Information Processing School of Computer Science, Fudan University fshi22@m.fudan.edu.cn _{_ libin,xyxue _}_ @fudan.edu.cn 

## ABSTRACT 

Endowing machines with abstract reasoning ability has been a long-term research topic in artificial intelligence. Raven’s Progressive Matrix (RPM) is widely used to probe abstract visual reasoning in machine intelligence, where models will analyze the underlying rules and select one image from candidates to complete the image matrix. Participators of RPM tests can show powerful reasoning ability by inferring and combining attribute-changing rules and imagining the missing images at arbitrary positions of a matrix. However, existing solvers can hardly manifest such an ability in realistic RPM tests. In this paper, we propose a deep latent variable model for answer generation problems through **R** ule **A** bstract **I** on and **SE** lection (RAISE). RAISE can encode image attributes into latent concepts and abstract atomic rules that act on the latent concepts. When generating answers, RAISE selects one atomic rule out of the global knowledge set for each latent concept to constitute the underlying rule of an RPM. In the experiments of bottom-right and arbitrary-position answer generation, RAISE outperforms the compared solvers in most configurations of realistic RPM datasets. In the oddone-out task and two held-out configurations, RAISE can leverage acquired latent concepts and atomic rules to find the rule-breaking image in a matrix and handle problems with unseen combinations of rules and attributes. 

## 1 INTRODUCTION 

The abstract reasoning ability is pivotal to abstracting the underlying rules from observations and quickly adapting to novel situations (Cattell, 1963; Zhuo & Kankanhalli, 2021; Małki´nski & Ma´ndziuk, 2022a), which is the foundation of cognitive processes (Gray & Thompson, 2004) like number sense (Dehaene, 2011), spatial reasoning (Byrne & Johnson-Laird, 1989), and physical reasoning (McCloskey, 1983). Intelligent systems may benefit from human-like abstract reasoning when leveraging acquired skills in unseen tasks (Barrett et al., 2018), for example, generalizing the law of object collision in the simulation environment to real scenes. Therefore, endowing intelligent systems with abstract reasoning ability is the cornerstone of higher-intelligence systems and a long-lasting research topic of artificial intelligence (Chollet, 2019; Małki´nski & Ma´ndziuk, 2022b). 

Raven’s Progressive Matrix (RPM) is a classical test of abstract reasoning ability for human and intelligent systems (Małki´nski & Ma´ndziuk, 2022a), where participators need to choose one image out of eight candidates to fill in the bottom-right position of a 3 _×_ 3 image matrix (Raven & Court, 1998). Previous studies demonstrate that participators can display powerful reasoning ability by directly imagining the missing images (Hua & Kunda, 2020; Pekar et al., 2020), and answergeneration tasks can more accurately reflect the model’s understanding of underlying rules than answer-selection ones (Mitchell, 2021). For example, some RPM solvers find shortcuts in discriminative tasks by selecting answers according to the bias of candidate sets instead of the given context. 

To solve answer-selection problems, many solvers fill each candidate to the matrix for score estimation and can hardly imagine answers from the given context (Barrett et al., 2018; Hu et al., 2021). 

> _∗_ Corresponding author 

1 

Published as a conference paper at ICLR 2024 

Some generative solvers have been proposed to solve answer-generation tasks (Pekar et al., 2020; Zhang et al., 2021b;a). They generate solutions for bottom-right images and select answers by comparing the solutions and candidates. However, some generative solvers do not parse interpretable attributes and attribute-changing rules from RPMs (Pekar et al., 2020), and usually introduce artificial priors in the processes of representation learning or abstract reasoning (Zhang et al., 2021b;a). On the other hand, most generative solvers are trained with the aid of candidate sets in training, bringing the potential risk of learning shortcuts (Hu et al., 2021; Benny et al., 2021). 

Deep latent variable models (DLVMs) (Kingma & Welling, 2013; Sohn et al., 2015) can capture underlying structures of noisy observations via interpretable latent spaces (Edwards & Storkey, 2017; Eslami et al., 2018; Garnelo et al., 2018; Kim et al., 2019). Previous work (Shi et al., 2021) solves generative RPM problems by regarding attributes and attribute-changing rules as latent concepts, which can generate solutions by executing attribute-specific predictive processes. Through conditional answer-generation processes that consider the underlying structure of RPM panels, the distractors are not necessary to train DLVM-based solvers. Although previous work has achieved answer generation in RPMs with continuous attributes, understanding complex discrete rules and abstracting global rules in realistic datasets is still challenging for DLVMs. 

This paper proposes a DLVM for generative RPM problems through **R** ule **A** bstract **I** on and **SE** lection (RAISE)[1] . RAISE encodes image attributes (e.g., object size and shape) as independent latent concepts to bridge high-dimensional images and latent representations of rules. The underlying rules of RPMs are decomposed into subrules in terms of latent concepts and abstracted into atomic rules as a set of learnable parameters shared among RPMs. RAISE picks up proper rules for each latent concept and combines them into the integrated rule of an RPM to generate the answer. The conditional generative process of RAISE indicates how to use the global knowledge of atomic rules to imagine (generate) target images (answers) _interpretably_ . RAISE can automatically parse latent concepts without meta information of image attributes to reduce artificial priors in the learning process. RAISE can be trained under semi-supervised settings, requiring only a small amount of rule annotations to outperform the compared models in non-grid configurations. By predicting the target images at arbitrary positions, RAISE does not require distractors of candidate sets in training and supports generating missing images at arbitrary and even multiple positions. 

RAISE outperforms the compared solvers when generating bottom-right and arbitrary-position answers in most configurations of datasets. We interpolate and visualize the learned latent concepts and apply RAISE in odd-one-out problems to demonstrate its interpretability. The experimental results show that RAISE can detect the rule-breaking image of a matrix through interpretable latent concepts. Finally, we evaluate RAISE on two out-of-distribution configurations where RAISE retains relatively higher accuracy when encountering unseen combinations of rules and attributes. 

## 2 RELATED WORK 

**Generative RPM Solvers.** While selective RPM solvers (Zhuo & Kankanhalli, 2021; Barrett et al., 2018; Wu et al., 2020; Hu et al., 2021; Benny et al., 2021; Steenbrugge et al., 2018; Hahne et al., 2019; Zhang et al., 2019b; Zheng et al., 2019; Wang et al., 2019; 2020; Jahrens & Martinetz, 2020) focus on answer-selection problems, generative solvers predict representations or images at missing positions (Pekar et al., 2020; Zhang et al., 2021b;a). Niv et al. extract image representations through Variational AutoEncoder (VAE) (Kingma & Welling, 2013) and design a relation-wise perception process for answer prediction (Pekar et al., 2020). With interpretable scene representations, ALANS (Zhang et al., 2021b) and PrAE (Zhang et al., 2021a) adopt algebraic abstract and symbolic logical systems as the reasoning backends. These generative solvers predict answers at the bottom-right position. LGPP (Shi et al., 2021) and CLAP (Shi et al., 2023) learn hierarchical latent variables to capture the underlying rules of RPMs with random functions (Williams & Rasmussen, 2006; Garnelo et al., 2018), and can generate answers at arbitrary positions on RPMs with continuous attributes. RAISE is a variant of DLVM to realize generative abstract reasoning on realistic RPM datasets with discrete attributes and rules through atomic rule abstraction and selection. 

**Bayesian Inference with Global Latent Variables.** DLVMs (Kingma & Welling, 2013; Sohn et al., 2015; Sønderby et al., 2016) can capture underlying structures of high-dimensional data in latent 

> 1Code is available at https://github.com/FudanVI/generative-abstract-reasoning/tree/main/raise 

2 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0003-01.png)


**----- Start of picture text -----**<br>
Context latent  Target latent<br>concepts 𝒛%+ concepts 𝒛%*<br>𝒙! 𝒙"<br>Ψ Target Initialization<br>Sample<br>Global Knowledge Set<br>𝑟 [#] 𝑓!0.1$ 𝑓!-./# 𝜓' ℎ 𝑍 [%] ; 𝜓& ["]<br>𝒑%' 𝒑%( 𝒑%, 𝒒%' 𝒒%( 𝒒%, 𝜓(<br>Rule<br>𝒛! [#] 𝒛#" 𝒑, [%] 𝑓!"#$! 𝒒, [%] 𝜓) 𝑟 [%] = 2 Execution<br>𝐶<br>𝑆 𝑇 Rule Selection Sample 0 1 … 0<br>(a) Graphical model of RAISE (b) Abstract reasoning process<br>**----- End of picture text -----**<br>


Figure 1: **An overview of RAISE** . The graphical model in (a) displays the generative process (solid black lines) and inference process (dashed red lines). Panel (b) shows the computational details of the abstract reasoning process and highlights the rule selection, rule execution, and global knowledge with blue, yellow, and red backgrounds, respectively. 

spaces, regard shared concepts as global latent variables, and introduce local latent variables conditioned on the shared concepts to distinguish each sample. GQN (Eslami et al., 2018) captures entire 3D scenes via global latent variables to generate 2D images of unseen perspectives. With objectcentric representations (Yuan et al., 2023), global latent variables can explain layouts of scenes (Jiang & Ahn, 2020) or object appearances for multiview scene generation (Chen et al., 2021; Kabra et al., 2021; Yuan et al., 2022; Gao & Li, 2023; Yuan et al., 2024). Global concepts can describe common features of elements in data with exchange invariance like sets (Edwards & Storkey, 2017; Hewitt et al., 2018; Giannone & Winther, 2021). NP family (Garnelo et al., 2018; Kim et al., 2019; Foong et al., 2020) constructs different function spaces through global latent variables. DLVMs can generate answers at arbitrary positions of an RPM by regarding the concept-changing rules as global concepts (Shi et al., 2021; 2023). RAISE holds a similar idea of modeling underlying rules as global concepts. Unlike previous works, RAISE attempts to abstract the atomic rules shared among RPMs. 

## 3 METHOD 

In this paper, an RPM problem is ( _**x** S,_ _**x** T_ ) where _**x** S_ and _**x** T_ are mutually exclusive sets of images, _S_ indexes the given context images, and _T_ indexes the target images to predict ( _T_ can index multiple images). The objective of RAISE is to maximize the log-likelihood log _p_ ( _**x** T |_ _**x** S_ ) while learning atomic rules shared among RPMs. In the following sections, we will introduce the generative and inference processes of RAISE that can abstract and select atomic rules in the latent space. 

## 3.1 CONDITIONAL GENERATION 

The generative process is the foundation of answer generation, including the stages of concept learning, abstract reasoning, and image generation. 

**Concept Learning.** RAISE extracts interpretable image representations for abstract reasoning and image generation in the concept learning stage. Previous studies have emphasized the role of abstract object representations in the abstract reasoning of infants (Kahneman et al., 1992; Gordon & Irwin, 1996) and the benefit of disentangled representations for RPM solvers (Van Steenkiste et al., 2019), which reflect the compositionality of human cognition (Lake et al., 2011). RAISE realizes compositionality by learning latent representations of attributes (Shi et al., 2021; 2023). RAISE regards image attributes as latent concepts and decomposes the rules of RPMs into atomic rules based on the latent concepts. Since the description of attributes is not provided in training, the latent concepts learned by RAISE are not exactly the same as the realistic attributes defined in the dataset. RAISE extracts _C_ context latent concepts _**z** s_ = _{_ _**z** s[c][}][C] c_ =1[for each context image] _**[ x]**[s]_[(] _[s][ ∈][S]_[):] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0003-09.png)


The encoder _gθ_[enc] outputs the mean of context latent concepts. The standard deviation is controlled by a hyperparameter _σz_ to keep training stability. Each context image is processed through _gθ_[enc] 

3 

Published as a conference paper at ICLR 2024 

independently, making it possible to extract latent concepts for any set of input images. In this stage, the encoder does not consider any relationships between images and focuses on concept learning. 

**Abstract Reasoning.** As illustrated in Figure 1b, RAISE predicts target latent concepts _**z** T_ from context latent concepts _**z** S_ in the abstract reasoning stage, involving rule abstraction, rule selection, and rule execution processes. To abstract atomic rules and build the global knowledge set, RAISE adopts _K_ global learnable parameters _ψ_ = _{ψk}[K] k_ =1[, each indicating an atomic rule shared among] RPMs. In rule selection, we use categorical indicators _{r[c] }[C] c_ =1[(] _[r][c]_[=][1] _[, ..., K]_[)][to][select][a][proper] rule out of _ψ_ for each concept. Inferring the indicators from _**z** S_ correctly is critical to rule selection. RAISE creates a 3 _×_ 3 representation matrix _**Z**[c]_ for each concept, initializing the representations of context images with the corresponding context latent concepts and those of target images with zero vectors. Then RAISE extracts the row-wise and column-wise representations: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0004-03.png)


obtain integrated representations of row and column rules.RAISE averages the representations via _**p**_ **¯** _[c]_ = ( _**p**[c]_ 1[+] _**[ p]**[c]_ 2[+] We concatenate _**[ p]**[c]_ 3[)] _[/]_[3][and] _**[q]**_ **[¯]** _[c]_[=] _**p**_ **¯** _[c]_[(] and _**[q]**_ 1 _[c]_[+] _**q**_ **¯** _[c]_ _**[ q]**_ to acquire the2 _[c]_[+] _**[ q]**_ 3 _[c]_[)] _[/]_[3][to] probability of selecting atomic rules out of the global knowledge set: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0004-05.png)


We denote the learnable parameters as _ϕ_ = _{ϕ_ 1 _, ϕ_ 2 _, ϕ_ 3 _}_ for convenience. In rule execution, RAISE selects and executes an atomic rule on each concept to predict the target latent concepts: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0004-07.png)


RAISE instantiates _h_ by selecting the _r[c]_ -th learnable parameters from the global knowledge set _ψ_ to convert the zero-initialized target representations in _**Z**[c]_ into the mean of target latent concepts. As in the concept learning stage, the standard deviation of target latent concepts is controlled by _σz_ . _h_ consists of convolution layers to aggregate information from neighbor context latent concepts on the matrix and update target latent concepts. Each learnable parameters in _ψ_ indicates a type of atomic rule. See Appendix C.1 for the detailed description of _h_ . 

**Image Generation.** Finally, RAISE decodes the target latent concepts predicted in the abstract reasoning stage into the mean of target images: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0004-10.png)


RAISE generates each target image independently to make the decoder focus on image reconstruction. We control the noise of target images by setting the standard deviation _σx_ as a hyperparameter. 

According to Figure 1a, we decompose the conditional generative process as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0004-13.png)


where _**h**_ is the set of all latent variables and Θ = _{θ, ϕ, ψ, φ}_ are learnable parameters of RAISE. 

## 3.2 VARIATIONAL INFERENCE 

RAISE approximates the untractable posterior with a variational distribution _q_ ( _**h** |_ _**x** T ,_ _**x** S_ ) (Kingma & Welling, 2013), which consists of the following distributions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0004-17.png)


Since RAISE shares the encoder between the generative and inference processes to reduce the modelparameters, we compute context latent concepts _**µ**_ **˜**[1:] _s[C]_ and target latent concepts _**µ**_ **˜**[1:] _t[C]_ via the same process described in Equation 1. In the inference process, RAISE reformulates the variational distribution of the categorical indicator _r[c]_ as _q_ ( _r[c] |_ _**z** S[c][,]_ _**[ z]** T[c]_[)] _[∝][p]_[(] _**[z]** T[c][|][r][c][,]_ _**[ z]** S[c]_[)] _[p]_[(] _[r][c][|]_ _**[z]** S[c]_[)][.][That][is,][RAISE] 

4 

Published as a conference paper at ICLR 2024 

predicts the prior probabilities _**π**_ 1: _[c] K_[of] _[ p]_[(] _[r][c][|]_ _**[z]** S[c]_[)][ from the context latent concepts] _**[ z]** S[c]_[and compute] the likelihood _p_ ( _**z** T[c][|][r][c][,]_ _**[ z]** S[c]_[)][by][executing][the][atomic][rule] _[r][c]_[(] _[r][c]_[=][1] _[,][ · · ·][, K]_[)][on] _**[z]** S[c]_[.][In][this][way,] we can estimate the variational distribution _q_ ( _r[c] |_ _**z** S[c][,]_ _**[ z]** T[c]_[)][ by considering both the prior probabilities] and the likelihoods of _K_ atomic rules, which reduces the risk of model collapse (e.g., always selecting one atomic rule from _ψ_ ). We provide more details of _q_ ( _r[c] |_ _**z** S[c][,]_ _**[ z]** T[c]_[)][in][Appendix][A.1.][Letting] Ψ = _{θ, ϕ, ψ}_ , we factorize the variational distribution as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0005-02.png)


## 3.3 PARAMETER LEARNING 

We update the parameters of RAISE by maximizing the evidence lower bound (ELBO) of the loglikelihood log _p_ ( _**x** T |_ _**x** S_ ) (Kingma & Welling, 2013). With the generative process _p_ Θ and the variational distribution _q_ Ψ defined in Equations 6 and 8, the ELBO is ( _q_ denotes the variational distribution, and we omit the parameter symbols Θ and Ψ for convenience) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0005-05.png)


The **reconstruction loss** _L_ rec measures the quality of the reconstruction images. The **concept regularizer** _R_ pred estimates the distance between the predicted target concepts and the concepts directly encoded from target images. Minimizing _R_ pred will promote RAISE to generate correct predictions in the space of latent concepts. The **rule regularizer** _R_ rule expects RAISE to select the same rules when given different sets of images in an RPM. The variational posterior _q_ ( _r[c] |_ _**z** S[c][,]_ _**[ z]** T[c]_[)][ conditioned] on the entire matrix and the prior _p_ ( _r[c] |_ _**z** S[c]_[)][ conditioned on the context images are expected to have] similar probabilities. The detailed derivation of the ELBO is provided in Appendix A.2. 

The abstraction and selection of atomic rules rely on the acquired latent concepts. Therefore, RAISE introduces auxiliary rule annotations to improve the quality of latent concepts and stabilize the learning process. We denote rule annotations as _**v**_ = _{va}[A] a_ =1[where] _[A]_[is][the][number][of][ground][truth] attributes and _va_ indicates the type of rules on the _a_ -th attribute. For example, _**v**_ = [2 _,_ 1 _,_ 3] means that the attributes follow the second, first, and third rules respectively. RAISE does not leverage the meta-information of attributes in training since the rule annotations only inform the type of rule on each attribute. The meaning of attributes is automatically learned by RAISE for accurate rule abstraction and selection. One key to guiding concept learning with rule annotations is determining the correspondence between latent concepts and attributes. RAISE introduces a _A×C_ binary matrix _**M**_ where _**M** a,c_ = 1 indicates that the _a_ -th attribute is encoded in the _c_ -th latent concept. Therefore, the rule predicted on the _c_ -th latent concept is supervised by the rule annotation _va_ , and the auxiliary loss measures distances between the predicted and ground truth types of rules: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0005-08.png)


The auxiliary loss _L_ sup is the log-likelihood of the categorical distributions considering the attributeconcept correspondence _**M**_ . The binary matrix _**M**_ is derived by solving the following assignment problem on a batch of RPM samples: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0005-10.png)


Equation 11 allows the existence of redundant latent concepts, which can be solved using the modified Jonker-Volgenant algorithm (Crouse, 2016). In this case, the training objective becomes 

arg max _L_ rec _− β_ 1 _R_ pred _− β_ 2 _R_ rule + _β_ 3 _L_ sup (12) Θ 

5 

Published as a conference paper at ICLR 2024 

Table 1: **The accuracy (%) of selecting bottom-right answers on different configurations (i.e.,** _**Center**_ **,** _**L-R**_ **, etc) of RAVEN/I-RAVEN** . The table displays the average results of ten trials. 

|Models|Average|Center<br>L-R<br>U-D<br>O-IC<br>O-IG<br>2_×_2Grid<br>3_×_3Grid|
|---|---|---|
|GCA-I<br>GCA-R<br>GCA-C<br>ALANS<br>PrAE|12.0/24.1<br>13.8/27.4<br>32.7/41.7<br>54.3/62.8<br>80.0/85.7|14.0/30.2<br>7.9/22.4<br>7.5/26.9<br>13.4/32.9<br>15.5/25.0<br>11.3/16.3<br>14.5/15.3<br>16.6/34.5<br>9.4/26.9<br>6.9/28.0<br>17.3/37.8<br>16.7/26.0<br>11.7/19.2<br>18.1/19.3<br>37.3/51.8<br>26.4/44.6<br>21.5/42.6<br>30.2/46.7<br>33.0/35.6<br>37.6/38.1<br>43.0/32.4<br>42.7/63.9<br>42.4/60.9<br>46.2/65.6<br>49.5/64.8<br>53.6/52.0<br>70.5/66.4<br>75.1/65.7<br>97.3/**99.9**<br>96.2/97.9<br>96.7/97.7<br>95.8/98.4<br>68.6/76.5<br>**82.0**/**84.5**<br>23.2/45.1|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|6.4/16.3<br>7.3/27.6<br>17.5/32.8<br>40.1/64.0<br>**90.0**/**92.1**|9.2/20.1<br>4.7/18.9<br>5.2/21.2<br>4.0/13.9<br>3.1/12.3<br>8.6/13.7<br>10.4/13.9<br>9.8/47.4<br>4.1/20.3<br>3.5/20.7<br>5.4/38.2<br>7.6/36.1<br>10.0/15.0<br>10.5/15.6<br>30.4/42.9<br>13.4/35.1<br>12.2/32.1<br>16.4/37.5<br>9.5/26.0<br>16.0/20.1<br>24.3/35.8<br>98.4/99.2<br>67.0/91.1<br>60.9/86.6<br>14.5/69.9<br>13.5/57.1<br>14.7/25.2<br>11.6/18.6<br>**99.2**/99.8<br>**98.5**/**99.6**<br>**99.3**/**99.9**<br>**97.6**/**99.6**<br>**89.3**/**96.0**<br>68.2/71.3<br>**77.7**/**78.7**|



where _β_ 1, _β_ 2, and _β_ 3 are hyperparameters. RAISE also supports semi-supervised training settings. For samples that do not provide rule annotations, RAISE can set _β_ 3 = 0 and update parameters via the unsupervised part _L_ rec _− β_ 1 _R_ pred _− β_ 2 _R_ rule. 

## 4 EXPERIMENTS 

In the experiments, we compare the performance of RAISE with other generative solvers by generating answers at the bottom right and, more challenging, arbitrary positions. Then we conduct experiments to visualize the latent concepts learned from the dataset. Finally, RAISE carries out the odd-one-out task and is tested in held-out configurations to illustrate the benefit of learning latent concepts and atomic rules in generative abstract reasoning. 

**Datasets.** The models in the experiments are evaluated on the RAVEN (Zhang et al., 2019a) and I-RAVEN (Hu et al., 2021) datasets having seven image configurations (e.g., scenes with centric objects or object grids) and four basic rules. I-RAVEN follows the same configurations as RAVEN and reduces the bias of candidate sets to resist the shortcut learning of models (Hu et al., 2021). See Appendix B for details of datasets. 

**Compared Models.** In the task of bottom-right answer selection, we compare RAISE with the powerful generative solvers ALANS (Zhang et al., 2021b), PrAE (Zhang et al., 2021a), and the model proposed by Niv et al. (called GCA for convenience) (Pekar et al., 2020). RAISE selects the candidate closest to the predicted result in the latent space as the answer. We apply three strategies of answer selection in GCA: selecting the candidate having the smallest pixel difference to the prediction (GCA-I), having the smallest difference in the representation space (GCA-R), and having the highest panel score (GCA-C). Since these generative solvers cannot generate non-bottom-right answers, we take Transformer (Vaswani et al., 2017), ANP (Kim et al., 2019), LGPP (Shi et al., 2021), and CLAP (Shi et al., 2023) as baseline models to evaluate the ability to generate answers at arbitrary positions. We provide more details in Appendix C. 

**Training and Evaluation Settings.** For non-grid layouts, RAISE is trained under semi-supervised settings by using 5% rule annotations. RAISE leverages 20% rule annotations on O-IG and full rule annotations on 2 _×_ 2Grid and 3 _×_ 3Grid. The powerful generative solvers use full rule annotations and are trained and tested on each configuration respectively. We compare RAISE with them to illustrate the acquired bottom-right answer selection ability of RAISE under semi-supervised settings. The baselines can generate answers at arbitrary positions but cannot leverage rule annotations since they do not explicitly model the category of rules. We compare RAISE with the baselines to illustrate the benefit of learning latent concepts and atomic rules for generative abstract reasoning. Since the training of RAISE and the baselines do not require the candidate sets, and RAVEN/I-RAVEN only differ in the distribution of candidates, we train RAISE and the baselines on RAVEN and test them on RAVEN/I-RAVEN directly. See Appendix C for detailed training and evaluation settings. 

## 4.1 BOTTOM-RIGHT ANSWER SELECTION 

This experiment conducts classical RPM tests that require models to find the missing bottom-right images in eight candidates. Table 1 illustrates RAISE’s outstanding generative abstract reason- 

6 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0007-01.png)


Figure 2: **Selection accuracy at arbitrary positions** . The selection accuracy of RAISE (purple), Transformer (orange), CLAP (green), ANP (blue), and LGPP (black) in arbitrary positions. The x-axis of each plot indicates the number of candidates, and the y-axis is the selection accuracy. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0007-03.png)


Figure 3: **Answer generation at arbitrary positions** . The prediction results on RAVEN are highlighted (red box) to illustrate the arbitrary-position generation ability. Due to the existence of noise, some predictions may differ from the original sample, but they still follow the correct rules. 

ing ability on RAVEN/I-RAVEN. By comparing the difference between predictions and candidates, RAISE outperforms the compared generative solvers in most configurations of RAVEN/I-RAVEN, even if the distractors in candidate sets are not used in training. All the powerful generative solvers take full rule annotations for training, while RAISE in non-grid configurations only requires a small amount of rule annotations (5% samples) to achieve high selection accuracy. RAISE attains the highest selection accuracy compared to the baselines which can generate answers at arbitrary positions. By comparing the results on RAVEN/I-RAVEN, we find that generative solvers are more likely to have accuracy improvement on I-RAVEN, because I-RAVEN generates distractors that are less similar to correct answers to avoid significant biases in candidate sets. For grid-shaped configurations, we found that the noise in datasets will significantly influence the model performance. By removing the noise in object attributes, RAISE achieves high selection accuracy on three grid-shaped configurations using only 20% rule annotations. See Appendix D.1 for the detailed experimental results. 

## 4.2 ANSWER SELECTION AT ARBITRARY POSITIONS 

The above generative solvers can hardly generate answers at non-bottom-right positions. In this experiment, we probe the ability of RAISE and baselines to generate answers at arbitrary positions. We first generate additional candidate sets in the experiment because RAVEN and I-RAVEN do not provide candidate sets for non-bottom-right images. To this end, we sample a batch of RPMs from the dataset and split the RPMs into target and context images in the same way. For each matrix, we 

7 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0008-01.png)


**----- Start of picture text -----**<br>
(b) Odd-one-out in RPMs<br>Const Type ✓ Color ✓ Size ◯ Const TypeOut ✓ SizeOut ✓ Type ✓ In Size ✓ In Color ✓ In legend ◯✓ training and validation rulestest rules<br>Prog ✓ ✓ ✓ Prog ✓ ✓ ✓ ✓ ✓ Rules<br>Arith ✓ ✓ ✓ Arith ✓ ✓ ◯ ◯ ◯ Const: Constant<br>Dist3 ✓ ✓ ✓ Dist3 ✓ ✓ ◯ ◯ ◯ Prog: ProgressionArith: Arithmetic<br>Center-Held-Out O-IC-Held-Out Dist3: Distribute Three<br>(a) Visualization of latent concepts (c) Held-out configurations<br>**----- End of picture text -----**<br>


Figure 4: Panel (a) shows the interpolation results of latent concepts and the correspondence between the concepts and attributes. Panel (b) provides an example of RPM-based odd-one-out tests and displays the prediction deviations in concepts of each image. Panel (c) illustrates the strategy to split rule-attribute combinations in held-out configurations. 

use the target images of other _Nc_ samples in the batch as distractors to generate a candidate set with _Nc_ +1 entries. This strategy can adapt to the missing images at arbitrary and even multiple positions, and we can easily control the difficulty of answer selection through the number of distractors 

Figure 2 displays the accuracy of RAISE and baselines when generating answers at arbitrary and multiple positions. RAISE maintains high accuracy in all configurations. Although Transformer has higher accuracy than the other three baselines, especially in non-grid scenes, the prediction accuracy drops significantly on 2 _×_ 2Grid and 3 _×_ 3Grid. Figure 3 provides the qualitative prediction results on RAVEN. It is difficult for ANP and LGPP to generate clear answers. CLAP can generate answers with partially correct attributes in simple cases (e.g., CLAP generates an object with the correct color but the wrong size and shape in the sample of Center). RAISE produces high-quality predictions and can solve RPMs with multiple missing images. By predicting multiple missing images at arbitrary positions, The qualitative results intuitively reveal the in-depth generative abstract reasoning ability in models, which the bottom-right answer generation task does not involve. 

## 4.3 LATENT CONCEPTS 

Latent concepts bridge atomic rules and high-dimensional observations. Figure 4a visualizes the latent concepts learned from Center and O-IC by traversing concept representations of an image in the latent space. If the concepts are well decomposed, decoding the interpolated concept representations will change one attribute of the original image. Besides observing visualization results, we can find the correspondence between concepts and attributes with the aid of the binary matrix _**M**_ . As shown in Figure 4a, RAISE can automatically set some redundant concepts when there are more concepts than attributes. (e.g., the first concept of _Center_ ). The visualization results illustrate the concept learning ability of RAISE, which is the foundation of abstracting and selecting atomic rules shared among RPMs. 

## 4.4 ODD-ONE-OUT IN RPM 

In odd-one-out tests, RAISE attempts to find the rule-breaking image in a panel. To generate RPMbased odd-one-out problems, we replace the bottom-right image of an RPM with a random distractor in the candidate set. Taking Figure 4b as an example, we change the object color from white to black by replacing the bottom-right image. RAISE takes each image in an RPM as the target, gets the prediction results, and computes the prediction error on latent concepts. The right panel of 

8 

Published as a conference paper at ICLR 2024 

Table 2: Selection accuracy (%) on two held-out configurations. 

|OOD Settings|RAISE<br>PrAE<br>ALANS<br>GCA-C<br>GCA-R<br>GCA-I<br>Transformer<br>ANP<br>LGPP<br>CLAP-NP|
|---|---|
|Center-Held-Out<br>O-IC-Held-Out|99.2<br>**99.8**<br>46.9<br>35.0<br>14.4<br>12.1<br>12.1<br>10.6<br>8.6<br>19.5<br>**56.1**<br>40.5<br>33.4<br>10.1<br>5.3<br>4.9<br>15.8<br>7.5<br>4.6<br>8.6|



Figure 4b shows the concept-level prediction errors, and we find that the 7th concept of the bottomright image deviates the most. According to Figure 4a, the 7th concept on _Center_ represents the attribute _Color_ , which is indeed the attribute modified when constructing the test. The last row has relatively higher concept distances since the incorrect image tends to influence the accuracy of answer generation at the most related positions. Because of the independent latent concepts and concept-specific reasoning processes of RAISE, the high concept distances only appear in the 7th concept. By solving RPM-based odd-one-out problems, we explain how concept-level predictions improve the interpretability of answer selection. Although RAISE is tasked with generating answers, it can handle answer-selection problems by excluding candidates violating the underlying rules. 

## 4.5 HELD-OUT CONFIGURATIONS 

To explore the abstract reasoning ability on out-of-distribution (OOD) samples, we construct two held-out configurations based on RAVEN (Barrett et al., 2018) as illustrated in Figure 4c. (1) **Center-Held-Out** keeps the samples of _Center_ following the attribute-rule tuple _(Size, Constant)_ as test samples, and the remaining constitute the training and validation sets. (2) **O-IC-Held-Out** keeps the samples of _O-IC_ following the attribute-rule tuples _(Type In, Arithmetic)_ , _(Size In, Arithmetic)_ , _(Color In, Arithmetic)_ , _(Type In, Distribute Three)_ , _(Size In, Distribute Three)_ , and _(Color In, Distribute Three)_ as test samples. The results given in Table 2 indicate that RAISE maintains relatively higher selection accuracy when encountering unseen combinations of attributes and rules. RAISE learns interpretable latent concepts to conduct concept-specific reasoning, by which the learning of rules and concepts are decoupled. Thus RAISE can tackle OOD samples via compositional generalization. Although RAISE has not ever seen the attribute-rule tuple _(Size, Constant)_ in training, it can still apply the atomic rule _Constant_ learned from other attributes to _Size_ in the test phase. 

## 5 CONCLUSION AND DISCUSSION 

This paper proposes a generative RPM solver RAISE based on conditional deep latent variable models. RAISE can abstract atomic rules from PRMs, keep them in the global knowledge set, and predict target images by selecting proper rules. As the foundation of rule abstraction and selection, RAISE learns interpretable latent concepts from images to decompose the integrated rules of RPMs into atomic rules. Qualitative and quantitative experiments show that RAISE can generate answers at arbitrary positions and outperform baselines, showing outstanding generative abstract reasoning. The odd-one-out task and held-out configurations verify the interpretability of RAISE in concept learning and rule abstraction. By using prediction deviations on concepts, RAISE can find the position and concept that breaks the rules in odd-one-out tasks. By combining the learned latent concepts and atomic rules, RAISE can generate answers on samples with unseen attribute-rule tuples. 

**Limitations and Discussion.** The noise in data is a challenge for the models based on conditional generation. In the experiment, we find that the noise of object attributes in grids will influence the selection accuracy of generative solvers like RAISE and Transformer on _2×2Grid_ . The candidate sets can provide clearer supervision in training to reduce the impact of noise. Deep latent variable models (DLVMs) can potentially handle noise in RPMs since RAISE works well on _Center_ and _O-IC_ with noisy attributes like _Rotation_ . In future works, exploring appropriate ways to reduce the influence of noise is the key to realizing generative abstract reasoning in more complicated scenes. For generative solvers that do not rely on candidate sets or are completely unsupervised, whether using datasets with large amounts of noise benefits the acquisition of generative abstract reasoning ability is worth exploring since the noise can make a generative problem have numerous solutions (e.g., PGM (Barrett et al., 2018)). In Appendices B.2 and D.1, we conduct an initial experiment and discussion on the impact of noise, but a more systematic and in-depth study will be carried out in the follow-up works. Some recent neural approaches attempt to solve similar systematic generalization problems (Rahaman et al., 2021; Lake & Baroni, 2023). We provide a discussion on the Bayesian and neural approaches of concept learning in Appendix E. 

9 

Published as a conference paper at ICLR 2024 

## ACKNOWLEDGMENTS 

This work was supported by the National Natural Science Foundation of China (No.62176060) and the Program for Professor of Special Appointment (Eastern Scholar) at Shanghai Institutions of Higher Learning. 

## REFERENCES 

- David Barrett, Felix Hill, Adam Santoro, Ari Morcos, and Timothy Lillicrap. Measuring abstract reasoning in neural networks. In _International conference on machine learning_ , pp. 511–520. PMLR, 2018. 

Yaniv Benny, Niv Pekar, and Lior Wolf. Scale-localized abstract reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 12557–12565, 2021. 

- Ruth MJ Byrne and Philip N Johnson-Laird. Spatial reasoning. _Journal of memory and language_ , 28(5):564–575, 1989. 

- Raymond B Cattell. Theory of fluid and crystallized intelligence: A critical experiment. _Journal of educational psychology_ , 54(1):1, 1963. 

- Chang Chen, Fei Deng, and Sungjin Ahn. Roots: Object-centric representation and rendering of 3d scenes. _The Journal of Machine Learning Research_ , 22(1):11770–11805, 2021. 

Franc¸ois Chollet. On the measure of intelligence. _arXiv preprint arXiv:1911.01547_ , 2019. 

- David F Crouse. On implementing 2d rectangular assignment algorithms. _IEEE Transactions on Aerospace and Electronic Systems_ , 52(4):1679–1696, 2016. 

- Stanislas Dehaene. _The number sense: How the mind creates mathematics_ . OUP USA, 2011. 

- Harrison Edwards and Amos Storkey. Towards a neural statistician. In _International Conference on Learning Representations_ , 2017. 

- SM Ali Eslami, Danilo Jimenez Rezende, Frederic Besse, Fabio Viola, Ari S Morcos, Marta Garnelo, Avraham Ruderman, Andrei A Rusu, Ivo Danihelka, Karol Gregor, et al. Neural scene representation and rendering. _Science_ , 360(6394):1204–1210, 2018. 

- Andrew Foong, Wessel Bruinsma, Jonathan Gordon, Yann Dubois, James Requeima, and Richard Turner. Meta-learning stationary stochastic process prediction with convolutional neural processes. _Advances in Neural Information Processing Systems_ , 33:8284–8295, 2020. 

- Chengmin Gao and Bin Li. Time-conditioned generative modeling of object-centric representations for video decomposition and prediction. In _Proceedings of the Conference on Uncertainty in Artificial Intelligence_ , pp. 613–623, 2023. 

- Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J Rezende, SM Eslami, and Yee Whye Teh. Neural processes. In _ICML 2018 Workshop on Theoretical Foundations and Applications of Deep Generative Models_ , 2018. 

- Giorgio Giannone and Ole Winther. Hierarchical few-shot generative models. In _Fifth Workshop on Meta-Learning at the Conference on Neural Information Processing Systems_ , 2021. 

- Robert D Gordon and David E Irwin. What’s in an object file? evidence from priming studies. _Perception & Psychophysics_ , 58(8):1260–1277, 1996. 

- Jeremy R Gray and Paul M Thompson. Neurobiology of intelligence: science and ethics. _Nature Reviews Neuroscience_ , 5(6):471–482, 2004. 

- Lukas Hahne, Timo L¨uddecke, Florentin W¨org¨otter, and David Kappel. Attention on abstract visual reasoning. _arXiv preprint arXiv:1911.05990_ , 2019. 

10 

Published as a conference paper at ICLR 2024 

- Luke B Hewitt, Maxwell I Nye, Andreea Gane, Tommi S Jaakkola, and Joshua B Tenenbaum. The variational homoencoder: Learning to learn high capacity generative models from few examples. In _Conference on Uncertainty in Artificial Intelligence_ . Association For Uncertainty in Artificial Intelligence (AUAI), 2018. 

- Sheng Hu, Yuqing Ma, Xianglong Liu, Yanlu Wei, and Shihao Bai. Stratified rule-aware network for abstract visual reasoning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pp. 1567–1574, 2021. 

- Tianyu Hua and Maithilee Kunda. Modeling gestalt visual reasoning on raven’s progressive matrices using generative image inpainting techniques. In _CogSci_ , volume 2, pp. 7, 2020. 

- Marius Jahrens and Thomas Martinetz. Solving raven’s progressive matrices with multi-layer relation networks. In _2020 International Joint Conference on Neural Networks (IJCNN)_ , pp. 1–6. IEEE, 2020. 

- Jindong Jiang and Sungjin Ahn. Generative neurosymbolic machines. _Advances in Neural Information Processing Systems_ , 33:12572–12582, 2020. 

- Rishabh Kabra, Daniel Zoran, Goker Erdogan, Loic Matthey, Antonia Creswell, Matt Botvinick, Alexander Lerchner, and Chris Burgess. Simone: View-invariant, temporally-abstracted object representations via unsupervised video decomposition. _Advances in Neural Information Processing Systems_ , 34:20146–20159, 2021. 

- Daniel Kahneman, Anne Treisman, and Brian J Gibbs. The reviewing of object files: Object-specific integration of information. _Cognitive psychology_ , 24(2):175–219, 1992. 

- Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, Ali Eslami, Dan Rosenbaum, Oriol Vinyals, and Yee Whye Teh. Attentive neural processes. In _International Conference on Learning Representations_ , 2019. 

- Diederik P Kingma and Max Welling. Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_ , 2013. 

- Brenden Lake, Ruslan Salakhutdinov, Jason Gross, and Joshua Tenenbaum. One shot learning of simple visual concepts. In _Proceedings of the annual meeting of the cognitive science society_ , volume 33, 2011. 

- Brenden M Lake and Marco Baroni. Human-like systematic generalization through a meta-learning neural network. _Nature_ , 623(7985):115–121, 2023. 

- Mikołaj Małki´nski and Jacek Ma´ndziuk. Deep learning methods for abstract visual reasoning: A survey on raven’s progressive matrices. _arXiv preprint arXiv:2201.12382_ , 2022a. 

- Mikołaj Małki´nski and Jacek Ma´ndziuk. A review of emerging research directions in abstract visual reasoning. _arXiv preprint arXiv:2202.10284_ , 2022b. 

Michael McCloskey. Intuitive physics. _Scientific american_ , 248(4):122–131, 1983. 

Melanie Mitchell. Abstraction and analogy-making in artificial intelligence. _Annals of the New York Academy of Sciences_ , 1505(1):79–101, 2021. 

- Niv Pekar, Yaniv Benny, and Lior Wolf. Generating correct answers for progressive matrices intelligence tests. _arXiv preprint arXiv:2011.00496_ , 2020. 

- Nasim Rahaman, Muhammad Waleed Gondal, Shruti Joshi, Peter Gehler, Yoshua Bengio, Francesco Locatello, and Bernhard Sch¨olkopf. Dynamic inference with neural interpreters. _Advances in Neural Information Processing Systems_ , 34:10985–10998, 2021. 

- John C Raven and John Hugh Court. _Raven’s progressive matrices and vocabulary scales_ , volume 759. Oxford pyschologists Press Oxford, 1998. 

- Fan Shi, Bin Li, and Xiangyang Xue. Raven’s progressive matrices completion with latent gaussian process priors. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pp. 9612–9620, 2021. 

11 

Published as a conference paper at ICLR 2024 

Fan Shi, Bin Li, and Xiangyang Xue. Compositional law parsing with latent random functions. In _International Conference on Learning Representations_ , 2023. 

- Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep conditional generative models. _Advances in neural information processing systems_ , 28, 2015. 

- Casper Kaae Sønderby, Tapani Raiko, Lars Maaløe, Søren Kaae Sønderby, and Ole Winther. Ladder variational autoencoders. _Advances in neural information processing systems_ , 29:3738–3746, 2016. 

- Xander Steenbrugge, Sam Leroux, Tim Verbelen, and Bart Dhoedt. Improving generalization for abstract reasoning tasks using disentangled feature representations. _arXiv preprint arXiv:1811.04784_ , 2018. 

- Sjoerd Van Steenkiste, Francesco Locatello, J¨urgen Schmidhuber, and Olivier Bachem. Are disentangled representations helpful for abstract visual reasoning? _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Duo Wang, Mateja Jamnik, and Pietro Lio. Abstract diagrammatic reasoning with multiplex graph networks. In _International Conference on Learning Representations_ , 2019. 

- Duo Wang, Mateja Jamnik, and Pietro Lio. Abstract diagrammatic reasoning with multiplex graph networks. _arXiv preprint arXiv:2006.11197_ , 2020. 

- Christopher K Williams and Carl Edward Rasmussen. _Gaussian processes for machine learning_ , volume 2. MIT press Cambridge, MA, 2006. 

- Yuhuai Wu, Honghua Dong, Roger Grosse, and Jimmy Ba. The scattering compositional learner: Discovering objects, attributes, relationships in analogical reasoning. _arXiv preprint arXiv:2007.04212_ , 2020. 

- Jinyang Yuan, Bin Li, and Xiangyang Xue. Unsupervised learning of compositional scene representations from multiple unspecified viewpoints. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 36, pp. 8971–8979, 2022. 

- Jinyang Yuan, Tonglin Chen, Bin Li, and Xiangyang Xue. Compositional scene representation learning via reconstruction: A survey. _IEEE Transactions on Pattern Analysis & Machine Intelligence_ , 45(10):11540–11560, 2023. 

- Jinyang Yuan, Tonglin Chen, Zhimeng Shen, Bin Li, and Xiangyang Xue. Unsupervised objectcentric learning from multiple unspecified viewpoints. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2024. 

- Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Raven: A dataset for relational and analogical visual reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 5317–5327, 2019a. 

- Chi Zhang, Baoxiong Jia, Feng Gao, Yixin Zhu, Hongjing Lu, and Song-Chun Zhu. Learning perceptual inference by contrasting. _arXiv preprint arXiv:1912.00086_ , 2019b. 

- Chi Zhang, Baoxiong Jia, Song-Chun Zhu, and Yixin Zhu. Abstract spatial-temporal reasoning via probabilistic abduction and execution. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9736–9746, 2021a. 

- Chi Zhang, Sirui Xie, Baoxiong Jia, Ying Nian Wu, Song-Chun Zhu, and Yixin Zhu. Learning algebraic representation for systematic generalization in abstract reasoning. _arXiv preprint arXiv:2111.12990_ , 2021b. 

- Kecheng Zheng, Zheng-Jun Zha, and Wei Wei. Abstract reasoning with distracting features. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Tao Zhuo and Mohan Kankanhalli. Effective abstract reasoning with dual-contrast network. In _International Conference on Learning Representations_ , 2021. 

12 

Published as a conference paper at ICLR 2024 

## A PROOFS AND DERIVATIONS 

## A.1 REFORMULATION OF THE POSTERIOR DISTRIBUTION 

According to Bayes’ theorem, the posterior distribution of rule indicators _q_ ( _r[c] |_ _**z** S[c][,]_ _**[ z]** T[c]_[)][ is the product] of the conditional prior _q_ ( _r[c] |_ _**z** S[c]_[)][ and the likelihood] _[ p]_[(] _**[z]** T[c][|][r][c][,]_ _**[ z]** S[c]_[)][:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0013-04.png)


Considering that _p_ ( _**z** T[c][|][r][c][,]_ _**[ z]** S[c]_[)][ is an isotropic Gaussian] _[ N]_ � _h_ ( _**Z**[c]_ ; _ψrc_ ) _, σz_[2] _**[I]**_ �, Equation 13 becomes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0013-06.png)


probabilitieswhere D( _**z** T[c]_[)] _**π**_ **˜**[ is the size of] 1: _[c] K_[.][Therefore, we use the logarithmic version of Equation 14:] _**[ z]** T[c]_[.][In practice, RAISE predicts unnormalized logits] **[ ˜]** _**[l]**_ 1: _[c] K_[instead of the] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0013-08.png)


Since the constant _C_ ( _**z** S[c][,]_ _**[ z]** T[c]_[)][ in Equation 15 will not influence the results of normalization, RAISE] ignores the constant term and predicts the unnormalized logits via 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0013-10.png)


Finally, the variational distribution _q_ ( _r[c] |_ _**z** S[c][,]_ _**[ z]** T[c]_[)][ is parameterized by] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0013-12.png)


## A.2 DERIVATION OF THE ELBO 

With the variational distribution _q_ Ψ( _**h** |_ _**x** T ,_ _**x** S_ ), the ELBO _L_ is (Sohn et al., 2015) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0013-15.png)


Considering the generative and inference processes 

13 

Published as a conference paper at ICLR 2024 

Equation 18 is further decomposed by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0014-02.png)


Since the encoder is shared between the generative and inference processes, we have _qθ_ ( _**z** s[c][|]_ _**[x]**[s]_[)][=] _pθ_ ( _**z** s[c][|]_ _**[x]**[s]_[)][ and] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0014-04.png)


Therefore, the ELBO is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0014-06.png)


## A.3 MONTE CARLO ESTIMATOR OF THE ELBO 

For a given RPM problem ( _**x** S,_ _**x** T_ ), we sample the latent variables **˜** _**r**_ , **˜** _**z** S_ , and **˜** _**z** T_ from the variatonal posterior _q_ Ψ( _**h** |_ _**x** T ,_ _**x** S_ ) to compute the ELBO: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0014-09.png)


**˜** _**µ**_[1:] _s[C]_ = _gθ_[enc][(] _**[x]**[s]_[)][ and] _**[µ]**_ **[˜]**[1:] _t[C]_ = _gθ_[enc][(] _**[x]**[t]_[)][ are means of latent concepts computed by the encoder.] _**[π]**_ **[˜]** 1: _[c] K_ is given by 17 and the indicator ˜ _r[c]_ is sampled through the Gumbel-Softmax distribution (Jang et al., 2016). Using the Monte Carlo estimator, _L_ can be approximated by the sampled latent variables. 

## A.3.1 RECONSTRUCTION LOSS 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0014-12.png)


Published as a conference paper at ICLR 2024 

A.3.2 CONCEPT REGULARIZER 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0015-02.png)


Trained with rule annotations, RAISE can quickly approach the real distribution _p⋆_ ( _r[c] |_ _**x**_ ) provided in rule annotations after the early learning stage. Therefore, we regard the real distribution as the predicted rule distribution, which is related to the matrix rather than conditional on the latent concepts. _qϕ,ψ_ ( _r[c] |_ _**z**_ **˜** _S[c]_ That _[,]_ **[ ˜]** _**[z]** T[c]_[)] is,[after] we[a] assume[few][learning] that samples[epochs.] from[By] _p_[replacing] _⋆_ ( _r[c] |_ _**x**_ ) are[the] similarly _[q][ϕ,ψ]_[(] _[r][c][|]_ distributed _**[z]**_ **[˜]** _S[c][,]_ **[ ˜]** _**[z]** T[c]_[)][to] to _[p][⋆]_[(] those _[r][c][|]_ _**[x]**_[)] from[,][we] move the inner expectation on _r[c]_ to the front. In this way, the inner expectation becomes the KL divergence between Gaussians and has a closed-form solution, reducing the additional noise in the sampling process. 

## A.3.3 RULE REGULARIZER 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0015-05.png)


A.3.4 ELBO 

Ignoring the constant terms _C_ rec and _C_ pred, the approximation of ELBO is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0015-08.png)


B DATASETS 

## B.1 RAVEN AND I-RAVEN 

Figure 5 displays seven image configurations of RAVEN (Zhang et al., 2019a). The image attributes include _Number/Position_ , _Type_ , _Size_ , and _Color_ , which can follow the rules _Constant_ , _Progress_ , _Arithmetic_ , and _Distribution Three_ . Each configuration contains 6000 training samples, 2000 validation samples, and 2000 test samples. RAVEN provides eight candidate images and attribute-level rule annotations for each RPM problem. Previous work pointed out the existence of bias in candidate sets of RAVEN (Hu et al., 2021), which allows models to find shortcuts for answer selection. I-RAVEN uses Attribute Bisection Tree (ABT) to generate candidate sets to resist shortcut learning (Hu et al., 2021). The experiment shows that the models trained with only the candidate sets of I-RAVEN have a selection accuracy close to the random guesses, which evidences the effectiveness of the candidate generation strategy. 

15 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0016-01.png)


Figure 5: **Different configurations of RAVEN** . In each figure, the top panel is an RPM where the target images are highlighted in red boxes; the middle panel is a candidate set with eight candidate images; and the bottom panel shows the attribute-changing rules in the RPM. 

## B.2 ATTRIBUTE NOISE OF RAVEN AND I-RAVEN 

RAVEN and I-RAVEN introduce noise to some attributes to increase the complexity of problems. In Center, L-R, U-D, and O-IC, the rotation of objects is the noise attribute. We can keep objects unchanged in rows or make random rotations. Figure 6 displays the noise of object grids on O- IG, 2 _×_ 2Grid, and 3 _×_ 3Grid, including the noise of object attributes (i.e., objects in Figure 6c can have different colors and rotations), and the noise of object positions (Figure 6d). The candidate set ensures that only one candidate image is the correct answer. To explore the influence of noise on selection accuracy, we remove the noise of object attributes from object grids, keep the noise of object positions, and generate three configurations O-IG-Uni, 2 _×_ 2Grid-Uni, and 3 _×_ 3Grid-Uni. 

16 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0017-01.png)


Figure 6: **The illustration of attribute noise** . (a) is an RPM from 2 _×_ 2Grid; (b) is the candidate set; (c) and (d) visualize two possible types of noise in the RPM. In this case, the image is correct as long as there are two pentagons of the correct size. The color, rotation, and position of objects will not influence the correctness of the image. 

## C MODELS 

## C.1 RAISE 

This section introduces the architectures and hyperparameters of RAISE. The network architectures are introduced in the order of _gθ_[enc][,] _[ f] ϕ_[ row] 1[,] _[ f] ϕ_[ col] 2[,] _[ f] ϕ_[ ind] 3[,] _[ h]_[, and] _[ g] φ_[dec][.] 

- _gθ_[enc][.][RAISE used a convolutional neural network to downsample images and extract the mean] of latent concepts. Denoting the number and size of latent concepts as _C_ and _dz_ , the encoder is 

   - 4 _×_ 4 Conv, stride 2, padding 1, 64 BatchNorm, ReLU 

   - 4 _×_ 4 Conv, stride 2, padding 1, 128 BatchNorm, ReLU 

   - 4 _×_ 4 Conv, stride 2, padding 1, 256 BatchNorm, ReLU 

   - 4 _×_ 4 Conv, stride 2, padding 1, 512 BatchNorm, ReLU 

   - 4 _×_ 4 Conv, 512 BatchNorm, ReLU 

   - ReshapeBlock, 512 

   - Fully Connected, _C × dz_ 

The ReshapeBlock flattens the feature map of the shape (512, 1, 1) to the vector with 512 dimensions, which is projected and split into the mean of _C_ latent concepts. 

- _fϕ_[row] 1[and] _[f] ϕ_[ col] 2[.][The][two][networks][have][the][same][architecture][to][extract][the][row][and][column] representations from RPMs: 

- Fully Connected, 512 ReLU 

- Fully Connected, 512 ReLU 

- Fully Connected, 64 

- where the input size is 3 _× dz_ and the size of output row and column representations is 64. 

- _fϕ_[ind] 3[.][This network converts the overall row and column representations of an RPM to the logits] of selection probabilities for atomic rule selection: 

- Fully Connected, 64 ReLU 

- Fully Connected, 64 ReLU 

- Fully Connected, _K_ 

where _K_ is the number of atomic rules. Since the row and column representations are concatenated as the input, the input size of the network is 128. 

- _h_ ( _**Z**[c]_ ; _ψk_ ). This network is a fully convolutional network, which predicts the means of target latent concepts from the representation matrix _**Z**[c]_ : 

   - 3 _×_ 3 Conv, stride 1, padding 1, 128 ReLU 

   - 3 _×_ 3 Conv, stride 1, padding 1, 128 ReLU 

17 

Published as a conference paper at ICLR 2024 

- 3 _×_ 3 Conv, stride 1, padding 1, _dz_ 

_h_ adopts convolutional layers with 3 _×_ 3 kernels, stride 1, and padding 1 to keep the shape of the 3 _×_ 3 representation matrix. The global knowledge set _ψ_ 1: _K_ stores _K_ learnable parameters of _h_ , which represents _K_ atomic rule respectively. 

- _gφ_[dec][.][The decoder accepts all latent concepts of an image as input and outputs the mean of the] pixel values for image reconstruction. The architecture is 

   - ReshapeBlock, ( _C × dz,_ 1 _,_ 1) 

   - 1 _×_ 1 Deconv, 256 BatchNorm, LeakyReLU 

   - 4 _×_ 4 Deconv, 128 BatchNorm, LeakyReLU 

   - 4 _×_ 4 Deconv, stride 2, padding 1, 64 BatchNorm, LeakyReLU 

   - 4 _×_ 4 Deconv, stride 2, padding 1, 32 BatchNorm, LeakyReLU 

   - 4 _×_ 4 Deconv, stride 2, padding 1, 32 BatchNorm, LeakyReLU 

   - 4 _×_ 4 Deconv, stride 2, padding 1, 1 Sigmoid 

where the negative slope of LeakyReLU is 0 _._ 02. Since the images of RAVEN and I-RAVEN are grayscaled, the decoder output only one image channel and uses the Sigmoid activation function to scale the range of pixel values to (0 _,_ 1). 

For all configurations of RAVEN, we set learning rate as 3 _×_ 10 _[−]_[4] , batch size as 512, _K_ = 4, _σx_ = 0 _._ 1, _σz_ = 0 _._ 1, _C_ = 8, _dz_ = 8, _β_ 1 = 5, _β_ 2 = 20, and _β_ 3 = 10. RAISE is insensitive when increasing _C_ since it can generate redundant latent concepts. When _C_ is too small to encode all attributes, the selection accuracy will decline significantly. We can set a large _C_ and reduce it until the number of redundant latent concepts is reasonable. In general, we choose _K_ by directly counting the number of unique labels in rule annotations. RAISE updates the parameters through the RMSprop optimizer (Hinton et al., 2012). To select the best model, we watch the performance on the validation set after each training epoch and save the model with the highest accuracy. 

## C.2 POWERFUL GENERATIVE SOLVERS 

**ALANS** (Zhang et al., 2021b) We train ALANS on the codebase released by the authors[2] , setting the learning rate as 0 _._ 95 _×_ 10 _[−]_[4] and the coefficient of the auxiliary loss as 1 _._ 0. Since the model can hardly converge from the initialized parameters, we initialize the parameters of ALANS with the pretrained checkpoint provided by the authors. More details can be seen in the repository. 

**PrAE** (Zhang et al., 2021a) For PrAE, we use the commended hyperparameters that the learning rate is 0 _._ 95 _×_ 10 _[−]_[4] and the weight of auxiliary loss is 1 _._ 0. The implementation of PrAE is based on the official repository[3] . 

**GCA** (Pekar et al., 2020) The official code of GCA[4] only implements the auxiliary loss on the PGM dataset (Barrett et al., 2018). Therefore, we modify the output size of the auxiliary network to the size of one-hot rule annotations in RAVEN/I-RAVEN. We set the latent size in GCA as 64 and the learning rate as 2 _×_ 10 _[−]_[4] . 

## C.3 BASELINES 

**Transformer** (Vaswani et al., 2017) To improve the model capability, we first apply the encoder and decoder to project images into low-dimensional representations and then predict the targets in the representation space via Transformer. Transformer uses the same encoder and decoder structures as RAISE. The hyperparameters of Transformer are chosen through grid search. We set the learning rate as 1 _×_ 10 _[−]_[4] from _{_ 5 _×_ 10 _[−]_[4] _,_ 1 _×_ 10 _[−]_[4] _,_ 5 _×_ 10 _[−]_[5] _}_ , the representation size as 256 from _{_ 512 _,_ 256 _,_ 128 _}_ , and the number of Transformer blocks as 4 from _{_ 2 _,_ 4 _,_ 6 _}_ . In addition, the number of attention heads is 4, the hidden size of feedforward networks is 1024, and the dropout is 0 _._ 1. All parameters are updated by the Adam (Kingma & Ba, 2015) optimizer. 

> 2https://github.com/WellyZhang/ALANS 

> 3https://github.com/WellyZhang/PrAE 

> 4https://github.com/nivPekar/Generating-Correct-Answers-for-Progressive-Matrices-Intelligence-Tests 

18 

Published as a conference paper at ICLR 2024 

Table 3: Learning rates of ANP on RAVEN/I-RAVEN. 

||Center|L-R|U-D|O-IC|O-IG|2_×_2Grid|3_×_3Grid|
|---|---|---|---|---|---|---|---|
|5|_×_10_−_5|1_×_10_−_5|1_×_10_−_5|5_×_10_−_6|5_×_10_−_6|3_×_10_−_5|3_×_10_−_5|



Table 4: **Hyperparameters of CLAP** . We give the number of concepts, weights in the ELBO ( _βt_ , _βf_ , and _βT C_ ), and standard deviation _σz_ on RAVEN/I-RAVEN. 

|Hyperparameters|Center<br>L-R<br>U-D<br>O-IC<br>O-IG<br>2_×_2Grid<br>3_×_3Grid|
|---|---|
|||
|#Concepts<br>_βt_<br>_βf_<br>_βT C_<br>_σz_|5<br>10<br>10<br>6<br>8<br>8<br>10<br>100<br>50<br>50<br>30<br>30<br>30<br>80<br>100<br>50<br>50<br>60<br>30<br>30<br>80<br>100<br>50<br>50<br>50<br>30<br>30<br>80<br>0.1<br>0.1<br>0.1<br>0.4<br>0.1<br>0.3<br>0.3|



**ANP** (Kim et al., 2019) For all configurations, we set the size of the global latent as 1024 and the batch size as 512. Table 3 shows the configuration-specific learning rates. Other hyperparameters and the model architecture remain the same as the 2D regression configuration in the original paper (Kim et al., 2019). 

**LGPP** (Shi et al., 2021) In the experiments, we use the official code of LGPP[5] by setting the learning rate as 5 _×_ 10 _[−]_[4] and the batch size as 256. In terms of model architecture, we set the size of axis latent variables as 4, the size of axis representations as 4, and the input size of the RBF kernel as 8. The network that converts axis latent variables to axis representations has hidden sizes [64, 64]. The network to extract the features for RBF kernels has hidden sizes [128, 128, 128, 128]. The hyperparameter _β_ that promotes disentanglement of LGPP is set to 10. For the configuration Center, the number of concepts is 5, while the others use 10 concepts. 

**CLAP** (Shi et al., 2023) Here we adopt the model architecture of the CRPM configuration in the official repository[6] and adjust the learning rate to 5 _×_ 10 _[−]_[4] , the batch size to 256, and the concept size to 8. Other hyperparameters are displayed in Table 4. 

## C.4 COMPUTATIONAL RESOURCE 

All the models are trained on the server with Intel(R) Xeon(R) Platinum 8375C CPUs, 24GB NVIDIA GeForce RTX 3090 GPUs, 512GB RAM, and Ubuntu 18.04.6 LTS. RAISE is implemented with PyTorch (Paszke et al., 2019). 

## D ADDITIONAL EXPERIMENTAL RESULTS 

## D.1 BOTTOM-RIGHT ANSWER SELECTION 

We generate new configurations by removing the noise in object attributes to analyze the influence of noise attributes. As shown in Table 5, RAISE achieves the highest accuracy on all three configurations. When we introduce more noise to RPMs, the number of solutions that follow the correct rules will increase. In this case, the provided candidate set with one correct answer and seven distractors can act as clear supervision in model training. Without the assistance of candidate sets in training, it is challenging to catch rules from noisy RPMs with multiple potential solutions. Therefore, RAISE and Transformer have significant accuracy improvements on configurations with less noise attributes. Overall, the experimental results show that reducing noise can bring significant improvements for the models trained without distractors in candidate sets (such as Transformer and RAISE). RAISE only requires 20% rule annotations to learn atomic rules from low-noise samples. 

> 5https://github.com/FudanVI/generative-abstract-reasoning/tree/main/rpm-lgpp 

> 6https://github.com/FudanVI/generative-abstract-reasoning/tree/main/clap 

19 

Published as a conference paper at ICLR 2024 

Table 5: The accuracy (%) of selecting bottom-right answers on O-IG-Uni, 2 _×_ 2Grid-Uni, and 3 _×_ 3Grid-Uni. 

|Models|O-IG-Uni<br>2_×_2Grid-Uni<br>3_×_3Grid-Uni|
|---|---|
|GCA-I<br>GCA-R<br>GCA-C<br>PrAE<br>ALANS|21.2/36.7<br>19.5/23.3<br>20.6/21.6<br>20.7/36.3<br>21.9/28.1<br>25.9/25.2<br>53.8/37.7<br>58.8/35.6<br>67.0/27.5<br>29.1/45.1<br>85.4/85.6<br>26.8/47.2<br>29.7/41.5<br>66.2/55.3<br>84.0/73.3|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|3.4/12.3<br>4.1/13.0<br>4.0/13.1<br>31.5/34.0<br>10.0/15.6<br>12.0/16.3<br>14.4/31.7<br>22.5/39.1<br>12.1/32.9<br>70.6/57.9<br>73.3/73.0<br>34.2/37.0<br>**95.8**/**99.0**<br>**87.6**/**97.9**<br>**95.3**/**93.2**|



Table 6: **The accuracy (%) of selecting bottom-right answers on different configurations (i.e.,** _**Center**_ **,** _**L-R**_ **, etc) of RAVEN/I-RAVEN** . In this table, RAISE is trained without the supervision of rule annotations (-aux) to illustrate the abstract reasoning ability in the unsupervised training setting. The table displays the average results of ten trials. 

|Models|Average|Center<br>L-R<br>U-D<br>O-IC<br>O-IG<br>2_×_2Grid<br>3_×_3Grid|
|---|---|---|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE (-aux)|6.4/16.3<br>7.3/27.6<br>17.5/32.8<br>40.1/64.0<br>**54.5**/**67.7**|9.2/20.1<br>4.7/18.9<br>5.2/21.2<br>4.0/13.9<br>3.1/12.3<br>8.6/13.7<br>10.4/13.9<br>9.8/47.4<br>4.1/20.3<br>3.5/20.7<br>5.4/38.2<br>7.6/36.1<br>10.0/15.0<br>10.5/15.6<br>30.4/42.9<br>13.4/35.1<br>12.2/32.1<br>16.4/37.5<br>9.5/26.0<br>16.0/20.1<br>24.3/**35.8**<br>**98.4**/**99.2**<br>**67.0**/**91.1**<br>60.9/86.6<br>14.5/69.9<br>13.5/57.1<br>14.7/25.2<br>11.6/18.6<br>30.2/56.6<br>47.9/80.8<br>**87.0**/**94.9**<br>**96.9**/**99.2**<br>**56.9**/**83.9**<br>**30.4**/**30.5**<br>**32.0**/27.8|



We also provide the selection accuracy of unsupervised RAISE in Table 6. The average accuracy of unsupervised RAISE lies between the unsupervised arbitrary-generation baselines (i.e., LGPP, ANP, CLAP, and Transformer) and the powerful generative RPM solvers trained with full rule annotations (i.e., GCA, ALANS, and PrAE). 

## D.2 ANSWER SELECTION AT ARBITRARY POSITION 

In this section, we give additional results for arbitrary-position answer generation. Figure 8 provides the detailed results of arbitrary-position answer generation for all seven configurations of RAVEN, for example, the prediction results when _|T |_ = 1 (Figure 8a) and _|T |_ = 2 (Figure 8b). In the visualization results, RAISE can generate high-quality predictions when _|T |_ = 1 and _|T |_ = 2. The performance of Transformer varies significantly among different configurations. Transformer predicts accurate answers on Center, while the predictions on 3 _×_ 3Grid deviate significantly from the ground truth images. In most cases, ANP, LGPP, and CLAP tend to generate incorrect images. Figure 7 provides the selection accuracy on I-RAVEN with different numbers of target images ( _|T |_ = 1 _,_ 2) and different numbers of distractors in candidate sets ( _Nc_ = 1 _,_ 3 _,_ 7 _,_ 15). We can make further analysis through the selection accuracy with test errors in Tables 8 and 9, where RAISE outperforms other baseline models on all image configurations of RAVEN and I-RAVEN. 

## D.3 LATENT CONCEPTS 

As mentioned in the main text, concept learning is an important component of RAISE. This section shows the interpolation results of latent concepts on all image configurations and the correspondences between latent concepts and real attributes in Figures 9 and 10. In most configurations, RAISE can learn independent latent concepts and the binary matrix _**M**_ that accurately reflects the concept-attribute correspondences. RAISE does not assign the latent concepts encoding object rotations to any attribute since the noise attributes are not included in rule annotations. This experiment illustrates the interpretability of the acquired latent concepts, which benefits the prediction of correct answers and the following experiment of odd-one-out. 

20 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0021-01.png)


Figure 7: **Selection accuracy at arbitrary positions on I-RAVEN** . Each plot contains the selection accuracy of RAISE (purple), Transformer (orange), CLAP (green), ANP (blue), and LGPP (black). The x-axis is the number of candidates, and the y-axis is the selection accuracy. 

Table 7: The accuracy (%) using different strategies of answer selection. 

|Models|Average|Center<br>L-R<br>U-D<br>O-IC<br>O-IG<br>2_×_2Grid<br>3_×_3Grid|
|---|---|---|
|RAISE-latent<br>RAISE-pixel|**90.0**/**92.1**<br>72.9/77.8|**99.2**/**99.8**<br>**98.5**/**99.6**<br>**99.3**/**99.9**<br>**97.6**/**99.6**<br>**89.3**/**96.0**<br>**68.2**/**71.3**<br>**77.7**/**78.7**<br>95.2/96.8<br>90.6/95.8<br>96.6/98.5<br>80.4/90.6<br>69.1/81.1<br>40.1/42.6<br>38.1/39.5|



## D.4 ODD-ONE-OUT IN RPM 

In this experiment, we provide the additional results of odd-one-out on different configurations where RAISE picks out rule-breaking images interpretably via prediction errors on latent concepts. Figure 11 visualizes the experimental results of odd-one-out. RAISE will display larger prediction errors at odd concepts, which is important evidence when solving odd-one-out problems. It should be pointed out that forming such concept-level prediction errors requires the model to parse independent latent concepts and conduct concept-specific abstract reasoning correctly. RAISE can apply the atomic rules in the global knowledge set to tasks like out-one-out and has interpretability in generative abstract reasoning. 

## D.5 STRATEGY OF ANSWER SELECTION 

In this experiment, we evaluate RAISE with two strategies of answer selection: comparing candidates and predictions in pixel space (RAISE-pixel) and latent space (RAISE-latent). Table 7 reports higher accuracy when candidates and predictions are compared in latent space. Due to the noise in attributes, there can be multiple solutions to a generative RPM problem. Assume that the answer to an RPM is the image having two triangles, the answer images may significantly differ from each other in the pixel space by generating two triangles in various positions. However, they still point to the same concepts _Number_ =2 and _Shape_ = _Triangle_ in the latent space. Therefore, selecting answers by comparing candidates and predictions in the latent space can be more accurate than comparing in the pixel space. 

## E DISCUSSION ON BAYESIAN AND NEURAL CONCEPT LEARNING 

**The learning objective.** A recent neural approach MLC (Lake & Baroni, 2023) uses meta-learning objectives to solve systematic generalization problems. Grant et al. (Grant et al., 2018) have reported a connection between meta-learning and hierarchical Bayesian models. The discussion section of MLC has also mentioned that the hierarchical Bayesian modeling can be explained from the view of meta-learning. In this perspective, the global atomic rules in RAISE act as global latent variables in hierarchical modeling. Although RAISE and MLC have different motivations for model design, there are potential connections and similarities between their learning objectives if we explain the reasoning process of RAISE from the perspective of hierarchical Bayesian modeling and metalearning. 

21 

Published as a conference paper at ICLR 2024 

**Interpretability of latent variables.** Both Bayesian and neural approaches can define basic modules in the learning processes, e.g., Functions in Neural Interpreters (Rahaman et al., 2021) and atomic rules in RAISE. Bayesian approaches usually design interpretable latent variables in generative processes, e.g., RAISE uses categorical random variables to indicate the types of the selected rules explicitly. While Neural Interpreters route inputs to different Functions by calculating specific scores. DLVM provides a powerful learning framework to learn interpretable latent structures from data, e.g., RAISE defines latent concepts to capture image attributes. In this way, visual scenes are decomposed into a simple set of latent variables, which may reduce the complexity of abstract reasoning and enable systematic generalization on attribute-rule combinations. 

**Solving multi-solution problems.** There can be multiple solutions for one generative reasoning problem due to the noise in data. DLVMs can handle multi-solution problems by stochastic sampling from the generative and inference processes. For example, RAISE can produce results different from the original sample but still follow the correct rules. Instead of making deterministic predictions, DLVMs attempt to provide probabilities of generating specific answers and capture randomness and uncertainty in abstract reasoning. 

## REFERENCES 

- David Barrett, Felix Hill, Adam Santoro, Ari Morcos, and Timothy Lillicrap. Measuring abstract reasoning in neural networks. In _International conference on machine learning_ , pp. 511–520. PMLR, 2018. 

- Erin Grant, Chelsea Finn, Sergey Levine, Trevor Darrell, and Thomas Griffiths. Recasting gradientbased meta-learning as hierarchical bayes. _arXiv preprint arXiv:1801.08930_ , 2018. 

- Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural networks for machine learning lecture 6a overview of mini-batch gradient descent. _Cited on_ , 14(8):2, 2012. 

- Sheng Hu, Yuqing Ma, Xianglong Liu, Yanlu Wei, and Shihao Bai. Stratified rule-aware network for abstract visual reasoning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pp. 1567–1574, 2021. 

- Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. _arXiv preprint arXiv:1611.01144_ , 2016. 

- Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, Ali Eslami, Dan Rosenbaum, Oriol Vinyals, and Yee Whye Teh. Attentive neural processes. In _International Conference on Learning Representations_ , 2019. 

- Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _International Conference on Learning Representations_ , 2015. 

- Brenden M Lake and Marco Baroni. Human-like systematic generalization through a meta-learning neural network. _Nature_ , 623(7985):115–121, 2023. 

- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. _Advances in neural information processing systems_ , 32, 2019. 

- Niv Pekar, Yaniv Benny, and Lior Wolf. Generating correct answers for progressive matrices intelligence tests. _arXiv preprint arXiv:2011.00496_ , 2020. 

- Nasim Rahaman, Muhammad Waleed Gondal, Shruti Joshi, Peter Gehler, Yoshua Bengio, Francesco Locatello, and Bernhard Sch¨olkopf. Dynamic inference with neural interpreters. _Advances in Neural Information Processing Systems_ , 34:10985–10998, 2021. 

- Fan Shi, Bin Li, and Xiangyang Xue. Raven’s progressive matrices completion with latent gaussian process priors. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pp. 9612–9620, 2021. 

- Fan Shi, Bin Li, and Xiangyang Xue. Compositional law parsing with latent random functions. _arXiv preprint arXiv:2209.09115_ , 2022. 

22 

Published as a conference paper at ICLR 2024 

Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep conditional generative models. _Advances in neural information processing systems_ , 28, 2015. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Raven: A dataset for relational and analogical visual reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 5317–5327, 2019. 

- Chi Zhang, Baoxiong Jia, Song-Chun Zhu, and Yixin Zhu. Abstract spatial-temporal reasoning via probabilistic abduction and execution. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9736–9746, 2021a. 

- Chi Zhang, Sirui Xie, Baoxiong Jia, Ying Nian Wu, Song-Chun Zhu, and Yixin Zhu. Learning algebraic representation for systematic generalization in abstract reasoning. _arXiv preprint arXiv:2111.12990_ , 2021b. 

23 

Published as a conference paper at ICLR 2024 

Table 8: **Answer generation at arbitrary positions on RAVEN** . We provide the average accuracy (%) and test errors (%) of ten trials on RAVEN for RAISE and baselines. 

|Model|Center(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|Center(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|---|---|---|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|55.8_±_2.5<br>30.8_±_1.9<br>17.1_±_2.0<br>9.1_±_1.2<br>61.4_±_0.7<br>38.0_±_0.7<br>23.5_±_0.9<br>14.5_±_0.7<br>91.5_±_0.7<br>80.1_±_1.6<br>67.2_±_1.8<br>53.8_±_1.7<br>99.6_±_0.2<br>99.1_±_0.2<br>98.5_±_0.3<br>97.3_±_0.5<br>**99.9**_±_**0.1**<br>**99.6**_±_**0.2**<br>**99.1**_±_**0.2**<br>**98.1**_±_**0.3**|53.8_±_1.4<br>28.9_±_1.6<br>15.4_±_1.2<br>8.3_±_0.6<br>58.3_±_0.5<br>34.7_±_1.3<br>20.5_±_1.0<br>12.2_±_0.7<br>90.8_±_2.1<br>80.3_±_3.4<br>67.7_±_6.0<br>55.3_±_6.2<br>97.2_±_2.3<br>91.1_±_5.7<br>88.0_±_4.0<br>90.2_±_5.5<br>**99.5**_±_**0.2**<br>**98.7**_±_**0.3**<br>**97.5**_±_**0.7**<br>**96.5**_±_**0.5**|
|Model|L-R(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|L-R(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|56.8_±_2.3<br>32.7_±_3.2<br>18.7_±_2.0<br>9.6_±_1.3<br>59.0_±_0.6<br>34.6_±_1.4<br>20.6_±_0.9<br>11.7_±_0.5<br>79.5_±_0.9<br>60.7_±_1.2<br>45.6_±_1.3<br>32.4_±_0.9<br>99.4_±_0.2<br>98.8_±_0.3<br>98.1_±_0.4<br>97.1_±_0.3<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.1**|57.4_±_2.1<br>31.9_±_1.9<br>18.4_±_2.0<br>9.4_±_1.1<br>60.5_±_1.2<br>36.3_±_1.1<br>21.7_±_0.9<br>12.6_±_0.7<br>80.3_±_1.3<br>62.6_±_2.6<br>46.4_±_3.8<br>33.9_±_2.6<br>95.8_±_1.6<br>90.5_±_2.3<br>87.2_±_2.8<br>81.4_±_4.9<br>**99.9**_±_**0.1**<br>**99.7**_±_**0.2**<br>**99.3**_±_**0.4**<br>**98.8**_±_**0.7**|
|Model|U-D(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|U-D(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|57.5_±_2.3<br>32.8_±_2.0<br>19.7_±_4.1<br>10.3_±_1.3<br>58.3_±_1.1<br>34.3_±_0.6<br>19.4_±_0.8<br>10.7_±_0.9<br>78.8_±_0.7<br>59.1_±_1.2<br>43.1_±_1.3<br>30.2_±_1.1<br>98.9_±_0.2<br>97.9_±_0.3<br>96.5_±_0.4<br>94.8_±_0.3<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**|57.6_±_1.5<br>32.5_±_1.5<br>18.0_±_1.2<br>10.2_±_1.1<br>59.6_±_0.6<br>35.6_±_1.4<br>20.8_±_0.5<br>11.9_±_0.8<br>78.4_±_1.6<br>59.9_±_2.8<br>42.9_±_2.8<br>31.5_±_2.8<br>92.3_±_1.7<br>85.2_±_1.7<br>75.6_±_3.1<br>70.6_±_1.9<br>**99.6**_±_**0.2**<br>**99.1**_±_**0.3**<br>**98.2**_±_**0.5**<br>**97.1**_±_**1.1**|
|Model|O-IC(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|O-IC(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|50.5_±_1.3<br>25.8_±_0.5<br>13.2_±_0.7<br>6.6_±_0.5<br>62.0_±_1.2<br>39.8_±_0.7<br>26.5_±_0.6<br>17.1_±_0.6<br>91.3_±_1.1<br>81.1_±_1.8<br>68.1_±_2.2<br>54.1_±_2.2<br>97.6_±_0.4<br>95.0_±_0.6<br>90.1_±_0.5<br>82.3_±_0.7<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.1**<br>**99.8**_±_**0.1**|49.8_±_1.3<br>25.7_±_1.1<br>12.8_±_0.5<br>6.7_±_0.4<br>61.6_±_1.1<br>38.6_±_1.3<br>24.3_±_1.2<br>15.2_±_0.9<br>90.9_±_2.2<br>81.4_±_2.4<br>68.8_±_4.8<br>57.5_±_6.5<br>96.7_±_1.7<br>92.1_±_3.3<br>90.2_±_3.8<br>80.2_±_5.0<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.1**<br>**99.9**_±_**0.2**<br>**99.8**_±_**0.1**|
|Model|O-IG(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|O-IG(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|49.9_±_0.6<br>25.0_±_1.2<br>12.0_±_0.6<br>6.1_±_0.3<br>66.1_±_1.1<br>45.1_±_1.1<br>30.1_±_2.0<br>20.2_±_0.5<br>77.8_±_1.5<br>58.4_±_1.8<br>43.2_±_1.9<br>30.5_±_1.0<br>97.9_±_0.4<br>95.2_±_0.5<br>90.6_±_0.9<br>82.8_±_0.9<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.1**<br>**99.7**_±_**0.1**<br>**99.5**_±_**0.3**|50.0_±_0.9<br>25.1_±_1.2<br>12.1_±_0.7<br>6.4_±_0.5<br>66.5_±_1.0<br>44.0_±_1.4<br>28.5_±_0.8<br>18.0_±_0.9<br>80.5_±_1.6<br>63.1_±_3.4<br>47.6_±_3.2<br>35.2_±_2.4<br>93.2_±_1.7<br>88.5_±_1.6<br>80.4_±_3.7<br>75.5_±_3.8<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.1**<br>**99.9**_±_**0.1**|
|Model|2_×_2Grid(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|2_×_2Grid(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|51.3_±_1.0<br>26.7_±_0.7<br>13.6_±_0.8<br>6.9_±_0.6<br>54.8_±_0.9<br>30.8_±_0.7<br>18.2_±_0.5<br>9.5_±_0.6<br>64.5_±_1.1<br>39.9_±_1.5<br>24.5_±_1.2<br>15.0_±_0.9<br>64.3_±_1.2<br>44.0_±_1.4<br>30.3_±_1.5<br>21.6_±_1.2<br>**97.2**_±_**0.3**<br>**93.5**_±_**0.7**<br>**89.8**_±_**0.6**<br>**85.9**_±_**1.1**|52.6_±_1.4<br>27.0_±_0.9<br>13.8_±_0.7<br>7.2_±_0.6<br>55.5_±_1.0<br>31.7_±_0.8<br>18.4_±_0.8<br>10.5_±_0.6<br>64.9_±_1.9<br>41.8_±_1.5<br>25.2_±_1.5<br>16.9_±_1.4<br>63.1_±_1.0<br>43.3_±_1.5<br>28.8_±_1.4<br>20.9_±_1.5<br>**96.5**_±_**0.4**<br>**92.1**_±_**1.9**<br>**87.5**_±_**2.1**<br>**83.2**_±_**1.7**|
|Model|3_×_3Grid(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|3_×_3Grid(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|53.2_±_1.3<br>28.3_±_0.9<br>14.8_±_0.4<br>8.1_±_1.0<br>53.9_±_1.0<br>29.7_±_0.9<br>16.7_±_0.2<br>9.4_±_0.7<br>86.2_±_1.0<br>71.2_±_1.3<br>56.4_±_1.8<br>43.9_±_1.2<br>59.4_±_0.8<br>37.8_±_1.1<br>24.3_±_0.8<br>16.2_±_0.4<br>**99.5**_±_**0.2**<br>**98.5**_±_**0.2**<br>**97.0**_±_**0.2**<br>**95.1**_±_**0.6**|52.8_±_1.2<br>27.9_±_1.2<br>14.8_±_0.9<br>7.8_±_0.6<br>55.0_±_1.2<br>31.3_±_1.4<br>17.9_±_0.7<br>10.4_±_0.4<br>86.1_±_1.3<br>72.3_±_2.8<br>60.9_±_3.4<br>47.1_±_4.0<br>59.5_±_0.8<br>36.6_±_1.3<br>23.6_±_0.8<br>16.4_±_1.1<br>**98.4**_±_**0.4**<br>**97.2**_±_**1.0**<br>**95.4**_±_**1.0**<br>**93.6**_±_**1.2**|



24 

Published as a conference paper at ICLR 2024 

Table 9: **Answer generation at arbitrary positions on I-RAVEN** . We provide the average accuracy (%) and test errors (%) of ten trials on I-RAVEN for RAISE and baselines. 

|Model|Center(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|Center(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|---|---|---|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|55.0_±_2.9<br>30.0_±_2.1<br>16.8_±_1.6<br>9.0_±_1.9<br>77.1_±_1.2<br>63.1_±_1.0<br>53.0_±_1.1<br>45.3_±_0.7<br>91.6_±_1.3<br>79.6_±_1.3<br>67.1_±_2.0<br>53.4_±_1.9<br>99.8_±_0.1<br>99.4_±_0.2<br>98.9_±_0.3<br>97.8_±_0.5<br>**99.9**_±_**0.1**<br>**99.7**_±_**0.1**<br>**99.3**_±_**0.2**<br>**98.3**_±_**0.3**|54.3_±_1.3<br>28.7_±_1.5<br>15.7_±_1.4<br>8.4_±_0.8<br>64.5_±_0.8<br>42.3_±_1.0<br>28.0_±_0.8<br>18.1_±_0.8<br>90.8_±_2.2<br>80.8_±_4.6<br>69.2_±_4.4<br>51.7_±_4.2<br>95.2_±_2.2<br>93.1_±_4.2<br>87.6_±_7.9<br>85.8_±_6.1<br>**99.5**_±_**0.2**<br>**98.8**_±_**0.4**<br>**97.8**_±_**0.4**<br>**96.1**_±_**1.3**|
|Model|L-R(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|L-R(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|57.1_±_2.9<br>31.9_±_3.3<br>19.4_±_2.3<br>11.2_±_1.1<br>71.4_±_1.0<br>49.8_±_1.3<br>35.4_±_1.0<br>24.0_±_1.3<br>80.0_±_1.6<br>61.5_±_1.3<br>45.9_±_1.7<br>33.3_±_1.3<br>99.7_±_0.1<br>99.4_±_0.1<br>99.0_±_0.2<br>98.8_±_0.3<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**|56.7_±_1.9<br>32.5_±_2.2<br>18.7_±_1.7<br>11.0_±_1.0<br>68.7_±_1.3<br>46.3_±_1.0<br>30.8_±_0.8<br>20.4_±_0.9<br>80.5_±_1.6<br>63.2_±_2.6<br>47.4_±_2.8<br>35.1_±_2.5<br>96.4_±_1.3<br>93.1_±_1.6<br>89.3_±_2.5<br>86.9_±_6.1<br>**99.9**_±_**0.1**<br>**99.7**_±_**0.1**<br>**99.4**_±_**0.3**<br>**99.3**_±_**0.5**|
|Model|U-D(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|U-D(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|58.5_±_2.5<br>33.3_±_2.0<br>19.2_±_3.8<br>10.8_±_1.5<br>69.5_±_1.4<br>49.1_±_1.2<br>33.8_±_0.9<br>23.0_±_1.3<br>79.5_±_0.9<br>60.4_±_1.5<br>44.8_±_1.4<br>32.3_±_1.4<br>99.5_±_0.1<br>99.0_±_0.3<br>98.5_±_0.4<br>97.7_±_0.3<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**|56.1_±_2.6<br>32.6_±_2.2<br>19.6_±_2.0<br>10.7_±_1.5<br>66.7_±_1.1<br>44.1_±_1.3<br>29.2_±_1.1<br>18.9_±_0.6<br>79.8_±_2.0<br>59.9_±_2.0<br>47.0_±_2.4<br>32.0_±_2.4<br>94.8_±_1.2<br>87.6_±_1.0<br>82.0_±_3.1<br>76.5_±_5.5<br>**99.6**_±_**0.2**<br>**99.1**_±_**0.3**<br>**98.1**_±_**0.5**<br>**96.9**_±_**0.8**|
|Model|O-IC(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|O-IC(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|51.4_±_1.0<br>25.4_±_0.7<br>12.9_±_0.9<br>6.7_±_0.5<br>81.5_±_0.7<br>69.4_±_1.0<br>59.5_±_0.9<br>51.1_±_1.1<br>91.7_±_0.9<br>81.4_±_1.3<br>68.6_±_2.3<br>57.8_±_4.3<br>99.1_±_0.2<br>98.0_±_0.3<br>95.9_±_0.4<br>92.9_±_0.9<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.1**<br>**99.9**_±_**0.1**|50.5_±_1.3<br>25.8_±_0.7<br>13.1_±_0.7<br>6.6_±_0.5<br>71.6_±_1.2<br>51.5_±_1.3<br>37.9_±_1.6<br>26.9_±_2.0<br>91.5_±_1.6<br>82.3_±_2.7<br>72.1_±_4.9<br>57.1_±_3.7<br>97.9_±_1.5<br>96.6_±_1.6<br>94.2_±_2.5<br>90.0_±_5.2<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.1**<br>**99.8**_±_**0.1**|
|Model|O-IG(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|O-IG(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|50.0_±_1.3<br>24.8_±_1.0<br>12.6_±_0.6<br>6.2_±_0.7<br>82.6_±_0.7<br>70.2_±_1.1<br>60.3_±_1.2<br>51.5_±_0.9<br>79.0_±_1.8<br>60.2_±_1.1<br>44.2_±_1.1<br>32.2_±_1.9<br>99.0_±_0.3<br>97.8_±_0.3<br>95.6_±_0.4<br>91.8_±_0.7<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.1**<br>**99.8**_±_**0.1**|49.7_±_0.7<br>24.9_±_0.8<br>12.4_±_0.6<br>6.7_±_0.4<br>75.9_±_0.8<br>57.5_±_1.2<br>42.1_±_2.4<br>29.9_±_1.5<br>81.0_±_1.7<br>64.2_±_1.3<br>49.0_±_2.1<br>36.4_±_2.6<br>96.6_±_0.9<br>93.0_±_1.1<br>87.9_±_2.5<br>83.5_±_1.8<br>**99.9**_±_**0.0**<br>**99.9**_±_**0.1**<br>**99.9**_±_**0.1**<br>**99.9**_±_**0.1**|
|Model|2_×_2Grid(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|2_×_2Grid(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|51.3_±_0.8<br>26.6_±_1.0<br>13.9_±_0.5<br>7.1_±_0.5<br>54.9_±_1.0<br>31.4_±_1.0<br>18.0_±_1.0<br>9.9_±_0.7<br>63.9_±_1.4<br>40.2_±_1.5<br>25.4_±_1.2<br>14.8_±_1.0<br>65.7_±_1.4<br>45.3_±_1.4<br>32.1_±_1.1<br>24.2_±_0.8<br>**97.5**_±_**0.4**<br>**95.0**_±_**0.6**<br>**91.1**_±_**0.7**<br>**87.0**_±_**0.7**|51.9_±_1.0<br>26.4_±_0.6<br>13.7_±_0.4<br>6.8_±_0.5<br>55.6_±_0.7<br>31.4_±_0.9<br>18.4_±_1.0<br>10.5_±_1.0<br>64.8_±_1.7<br>42.5_±_2.0<br>26.6_±_1.8<br>16.1_±_2.1<br>64.2_±_0.3<br>44.1_±_1.9<br>31.5_±_1.4<br>22.7_±_2.2<br>**96.4**_±_**0.9**<br>**93.2**_±_**1.4**<br>**89.0**_±_**1.6**<br>**85.5**_±_**2.3**|
|Model|3_×_3Grid(_|T|_= 1)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|3_×_3Grid(_|T|_= 2)<br>_Nc_ = 1<br>_Nc_ = 3<br>_Nc_ = 7<br>_Nc_ = 15|
|LGPP<br>ANP<br>CLAP<br>Transformer<br>RAISE|52.4_±_1.3<br>27.2_±_1.3<br>14.9_±_1.4<br>8.0_±_0.9<br>54.2_±_1.2<br>29.5_±_1.0<br>16.6_±_0.8<br>10.1_±_0.9<br>85.9_±_1.2<br>71.7_±_1.3<br>56.6_±_2.0<br>42.6_±_1.5<br>59.7_±_1.3<br>37.7_±_0.8<br>25.0_±_0.7<br>17.2_±_0.5<br>**99.6**_±_**0.1**<br>**98.8**_±_**0.2**<br>**97.5**_±_**0.3**<br>**95.5**_±_**0.6**|52.4_±_1.0<br>28.2_±_0.9<br>15.0_±_0.8<br>8.1_±_0.6<br>54.5_±_1.1<br>30.6_±_0.7<br>17.4_±_1.1<br>10.2_±_0.5<br>85.6_±_1.1<br>70.4_±_3.0<br>60.0_±_1.2<br>46.4_±_3.0<br>59.7_±_1.0<br>37.4_±_1.0<br>24.5_±_1.1<br>16.0_±_0.6<br>**98.8**_±_**0.5**<br>**97.0**_±_**0.9**<br>**94.8**_±_**1.5**<br>**92.2**_±_**2.5**|



25 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0026-01.png)


(a) The results of answer generation when _|T |_ = 1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0026-03.png)


(b) The results of answer generation when _|T |_ = 2 

Figure 8: **Answer generation at arbitrary positions** . The predictions are given in red boxes to illustrate the ability of (a) arbitrary-position and (b) multiple-position answer generation. 

26 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0027-01.png)


Figure 9: **Concept learning on RAVEN** . The table shows the interpolation results of latent concepts and the binary matrices indicating the correspondence between concepts and attributes on L-R and U-D. 

27 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0028-01.png)


Figure 10: **Concept learning on RAVEN** . The table shows the interpolation results of latent concepts and the binary matrices indicating the correspondence between concepts and attributes on O-IG, 2 _×_ 2Grid, and 3 _×_ 3Grid. 

28 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection_images/iclr_2024_Towards_Generative_Abstract_Reasoning_Completing_Raven’s_Progressive_Matrix_via_Rule_Abstraction_and_Selection.pdf-0029-01.png)


Figure 11: **Odd-one-out based on RPMs** . The plots display how to construct odd-one-out tests from different configurations of RPMs and how to find the odd image according to the prediction errors on latent concepts. 

29 

