# **Reasoning Multi-Agent Behavioral Topology for Interactive Autonomous Driving** 

**Haochen Liu**[1] _[,]_[2] **Li Chen**[2] _[,]_[3] **Yu Qiao**[2] **Chen Lv**[1] _[†]_ **Hongyang Li**[2] _[,]_[3] _[†]_ 1 Nanyang Technological University 2 Shanghai AI Lab 3 University of Hong Kong 

## **Abstract** 

Autonomous driving system aims for safe and social-consistent driving through the behavioral integration among interactive agents. However, challenges remain due to multi-agent scene uncertainty and heterogeneous interaction. Current dense and sparse behavioral representations struggle with inefficiency and inconsistency in multi-agent modeling, leading to instability of collective behavioral patterns when integrating prediction and planning (IPP). To address this, we initiate a topological formation that serves as a compliant behavioral foreground to guide downstream trajectory generations. Specifically, we introduce **Behavioral Topology (BeTop)** , a pivotal topological formulation that explicitly represents the consensual behavioral pattern among multi-agent future. BeTop is derived from braid theory to distill compliant interactive topology from multi-agent future trajectories. A synergistic learning framework (BeTopNet) supervised by BeTop facilitates the consistency of behavior prediction and planning within the predicted topology priors. Through imitative contingency learning, BeTop also effectively manages behavioral uncertainty for prediction and planning. Extensive verification on large-scale real-world datasets, including nuPlan and WOMD, demonstrates that BeTop achieves state-of-the-art performance in both prediction and planning tasks. Further validations on the proposed interactive scenario benchmark showcase planning compliance in interactive cases. Code and model is available at `https://github.com/OpenDriveLab/BeTop` . 

## **1 Introduction** 

Autonomous driving system aspires to safe, humanoid, and socially compatible maneuvers [1]. This drives for formulation, prediction, and negotiation of collective future behaviors among interactive agents and autonomous vehicles (AVs) [2]. Remarkable accuracy is achieved by learning-based paradigms [3], including end-to-end modular design [4–7], social modeling [8, 9], and trajectorylevel integration [10–13]. However, substantial challenges arise in real-world cases due to scene uncertainty and volatile interactive patterns for multi-agent future behaviors. 

To embrace compliant patterns for multi-agent future behaviors, current formulations fall into two mainstreams, dense representation and sparse representation (Fig. 1). Dense representation quantizes agent behaviors under ego-centric rasterization, forecasting bird’s eye view (BEV) occupancy probabilities [14, 7, 15] or temporal flow [16–18]. It is easy to deduce interactions, perform scalable behaviors for agents [19], and align with BEV perceptions [20]. Still, dense representation is hindered by frozen receptions. It causes safety-vulnerable intractability and occlusions potentially interacting with ego maneuvers [16, 21]. Contrary to pixel-wise behavioral probability, sparse representation forecasts agent-anchored set of trajectories [22–25] or intention distributions [10, 26, 27]. Its multimodal formulation for each agent marks the elasticity in diverse behavioral uncertainty and tractability under flexile spatial semantics. However, behavioral misalignment [28] and modality collapse [24] 

Work done while Haochen’s internship at Shanghai AI Lab. _[†]_ Equal co-advising. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0002-00.png)


**----- Start of picture text -----**<br>
a) b) c) d)<br>…..<br>BeTopNet<br>Topo Braids Reason<br>Driving scenario Dense Sparse BeTop  (Ours)<br>Autonomous vehicle Scene agents Collision &Overlap Planning &Prediction Braid crossings<br>**----- End of picture text -----**<br>


Figure 1: **Multi-agent Behavioral Formulation. (a)** A typical driving scenario in Arizona, US [35]; **(b)** Dense representation conducts scalable occupancy prediction jointly, but restrained reception leads to unbounded collisions with planning; **(c)** Sparse supervision derives multi-agent trajectories with multi-modalities, while it struggles with conflicts among integrated prediction and planning; **(d)** BeTop reasons future topological behaviors for all scene-agents through braids theory, funneling interactive eventual agents (in highlighted colors) and guiding compliant joint prediction and planning. 

impede compliant multi-agent modeling, requiring exponential computations with growth agent numbers [29]. Issues particularly result in unstable and slow behavioral learning when exposed to predictions and planning (IPP) [30]. Typical solutions by conditional prediction [31, 28, 32] or game-theoretic reasoning [8, 33] often lead to nonstrategic maneuvers [34] due to non-compliant rollouts in adjusting interactive behaviors. This calls for a re-formulation for multi-agent behaviors, which should stabilize collective behavioral patterns in a compliant manner for IPP objectives. 

The decision-making process for human drivers provides valuable insights. Humans primarily determine the future behavior of interacted agents for decision-making without relying on their specific states [2, 36]. Thus, an effective strategy involves assessing agent-wise behavioral impact on planning maneuvers, and reasoning about compliant interactions. Our fundamental insight is that compliant multi-agent behaviors exhibit topological formations, which can be identified by distilling consensual interactions from future behaviors. Prior works have approached this challenge through structural design [37, 38] or implicit relational learning [39–41] using GNNs [42] or Transformers [43]. Other studies quantify uncertainty by topological properties [44, 45]. Nevertheless, current literature is scarce in formulating explicit future supervision of compliant multi-agent behavioral patterns. 

To this end, we launch the multi-agent behavior formulation termed as _**Behavioral Topology**_ **(BeTop)** . At its core, BeTop explicitly forms the topological supervision of consensual multi-agent future interactions, and reasons to guide prediction and planning. BeTop stems from braid theory [46], which infers compliant interactions of multiple paths from the intertwining of their braids. This empowers BeTop to intuitively distill forward intertwines (occupancy) as joint topology from braided multi-agent future trajectories (Fig. 1), marrying dense and sparse representation. With the aid of BeTop, we introduce a synergistic Transformer-based learning stack, BeTopNet, for learning IPP objectives. To implement, an iterative decoding strategy simultaneously reasons about Behavioral Topology and generates trajectory sets. Then the topology-guided local attention, embedded in each decoder layer, selectively queries behavioral semantics from social-compliant agents within the predicted BeTop priors. To further alleviate multi-agent uncertainty through topological guidance, a contingency planning paradigm is fitfully deployed. We lay out the imitative contingency learning process, which regulates the safety-ensured short-term plan. It maintains the long-range uncertainty by reasoned joint predictions from BeTop. Experimental results exhibit enhanced consistency and accuracy for prediction and planning in real-world scenarios. Testing in proposed interactive cases further highlights the planning ability of BeTopNet. To sum up, our contributions are three-fold: 

- We bring in the concept of Behavioral Topology, a multi-agent behavioral formulation for topological reasoning that explicitly supervises consensual future interactions jointly for the IPP system. 

- A synergistic learning framework BeTopNet, offering joint planning and prediction guided by topology reasoning, is devised. Topology-guided local attention and imitative contingency planning could resolve scene compliance and multi-agent uncertainty. 

- Benchmarking on nuPlan [47] and WOMD [35], our approach demonstrates strong performance in both planning strategy and prediction accuracy. BeTopNet witnesses evident improvement 

2 

over previous counterparts, _e.g._ , +7 _._ 9% in general planning score, +3 _._ 8% under interactive cases, +4 _._ 1% mAP for joint prediction, and +2 _._ 3% mAP for marginal prediction. 

## **2 Related work** 

**Multi-agent behavioral modeling.** Carving the collective future behavior of diverse agents is imperative for socially-consistent driving maneuver. Earlier approaches centered around occupancy prediction [14, 5, 4, 48]. Forecasting the spatial presence under dense BEV representation [49, 20] offers flexibility of arbitrary agents [17, 50, 19] and alignment with perception [7, 51]. However, rigidity in resolution induces scenario occlusion [16], rendering intractable occupancy [21]. In parallel, sparse representation consolidates multi-agent behavior into cohesive modalities across future trajectories [52, 22, 53, 54] or intentions [27, 10, 26, 55]. Joint future behaviors are derived through goal-based sampling or recombination from marginal predictions [56–58]. However, the collection of joint modalities is susceptible to mode collapse and entails exponential complexity [29, 59]. Meanwhile, topological representation has garnered traction as motion primitives [60, 61] or tools [44, 45] for scenario quantification, yet topological properties delineating collective future behaviors remain largely unexplored. Our BeTop targets the issue, marrying dense behavior probabilities by topology with the sparse motion from joint predictions to present structured future behaviors. 

On the other hand, inconsistent communal agent behaviors have motivated leveraging future interactions. Implicit approaches obtain tacit interactions with attention [62, 32, 63] or GNN [24, 64, 37, 65] from final motion regressions. Nonetheless, the implicit supervisions are found inefficient in dynamic scenarios [22]. Contrarily, explicitly reasoning mutual behaviors by conditional factorization [31, 66], relation reasoning [41, 40], or entropy-based methods [67, 68] offer consistent behavioral priors. However, hefty variance across agent dynamics and scenario geometries yield unstable inference. Distinguished from them, BeTop crafts a compact topological supervision that stabilize future interactions among multi-agent behaviors. Derived from topological braids, BeTop offers a topological equivalent behavioral representation to guide compliant forecasting and planning. 

**Integrated prediction and planning.** IPP system aims to harmonize trajectory-based learning of future interactive behaviors between the ego vehicle and social agents. Rule-based approaches [69–71] integrate handcrafted future interactions to evaluate candidate planning profiles, offering remarkable outcomes in rule-powered reactive simulation [47]. Still, the absence of real-world behaviors exhibits significant gaps in interactive scenarios. Learning-based methods yield imitative planning by integrating predictions within holistic modeling [72–74]. However, history-based coalitions pose challenges in supervising future homology among agents. Recently, hybrid pipelines [8, 75] have utilized post-processing and optimization upon learning-based models to realize behavior interactions among predictions and planning. This can entail significant computational overhead, and imitative planning tends to overestimate hereditary uncertainty in behavior predictions. Tree-based [10, 27] and contingency-enabled [76, 11] works seek to balance planning preemption and aggression in the face of behavior uncertainty. Nonetheless, pipelines without holistic interactions fall into passive planning maneuvers and incur high exponential cost for predictions. In our work, BeTop provides an explicit prior for future interactive behaviors which enhances compliant trajectory generations. Moreover, the synergistic prediction and contingency planning networks with BeTop effectively manage behavioral uncertainty. 

## **3 Behavioral Topology** 

Presenting BeTop, we commence the Behavioral Topology formulation and task statement of IPP for autonomous driving in Sec. 3.1. Then, we demonstrate the BeTopNet network architecture (Fig. 3) for topological reasoning and IPP generation in Sec. 3.2. Finally, in Sec. 3.3, we propose the imitative contingency learning process by topological guidance for the proposed network. 

## **3.1 Formulation** 

**Problem formulation.** We consider the driving scenario with _Na_ agents as _A_ 1: _Na_ at presence _t_ = 0, along with the scenario map **M** . The states over historical horizon _Th_ are denoted as **X** 1 for AV and as **X** 2: _Na_ for scenario agents, respectively, where **X** _n_ = _{_ **x** _[−][T][h]_[:0] _}n_ , _n ∈_ [1 _, Na_ ]. The objective for integrated prediction and planning is to jointly predict scene agents’ trajectories **Y** 2: _Na_ as well as AV planning **Y** 1 over a future horizon _Tf_ as **Y** _n_ = _{_ **y**[1:] _[T][f] }n_ , _n ∈_ [1 _, Na_ ]. 

3 

Figure 2: **BeTop formulation.** Joint future trajectories are transformed to braid sets, and then form joint topology through intertwine indicators. 

Table 1: **Analysis on different behavioral formulations.** BeTop labels behave most similarly to human annotations [35], excelling over other formulations like _k_ nearest GT or local attention. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0004-02.png)


**----- Start of picture text -----**<br>
|||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Agent|formulations like|k|
|AV|
|Behavioral|WOMD|
|Formulations|Acc.|↑|AUC|↑|
|𝒙|𝐴1|𝑒1,𝑗|Expert|[35]|1.000|1.000|
|𝒚|𝒕|GT top-|k|0.833|0.702|
|𝒕|𝒙|𝐘1|𝒙|𝜎𝜎11+−|Local attention [JFP graph [|BeTop (Ours)|67]|22]|0.9510.955|0.967|0.500|0.731|0.522|
|𝒚|𝒚|

**----- End of picture text -----**<br>


**Topological formulation.** We leverage the braid theory [46], which probes explicit formulations for compliant multi-agent interactions from future data **Y** 1: _Na_ . Intuitively, it denotes a transform process for **Y** 1: _Na_ with respective agent coordinates, and then gathers each future forward intertwine (occupancy) as joint interactions. Formally, consider the braid group **B** _Na_ = _{σn}_ by _Na_ primitive braids _σn_ , each of which _σn_ = ( _f_ 1 _[n][,][ · · ·][, f] N[ n] a_[)][ denotes a tuple of monotonically increased functions] _f_ : R[3] _×_ **Y** _→_ R[2] _× I_ mapping from Cartesian � _⃗x, ⃗y,[⃗] t_ � to lateral coordinate � _⃗y,[⃗] t_ � for agent future **Y** . Specifically, the function _fi[n]_[in] _[ σ][n]_[is defined as] _[ f] i[ n][→]_[(] **[Y]** _[i][ −]_ **[b]** _[n]_[)] **[R]** _[n]_[; 1] _[ ≤][i, n][ ≤][N][a]_[, where] **[ b]** _[n]_ and **R** _n_ denote the left-hand transform matrix to local coordinate of agent _An_ . The joint interactive behaviors are identified as a set of braids having intertwines _{σn[±][}][⊂]_ **[B]** _[N] a_[over][others][[][45][],][as] shown in Fig. 2. Opposite to implicit methods [22, 67, 41] banking on future distance heuristic, each intertwine in the braid can signify an explicit behavioral response, distinguishing between assertive ( _σn_[+][, elicit yielding from others) and passive (] _[σ] n[−]_[, yield to others) maneuvers.][To avert difficulties in] dynamic braid set inference, we redraft multi-agent braids from a topology reasoning perspective. 

Named by BeTop, the goal is to reason a topological graph _G_ = ( _V, E_ ) for multi-agent future behaviors (Fig. 2). Expressly, node topology _V_ = _{_ **Y** _n}_ is denoted by multi-agent future trajectories. We can then reformulate the braid set _{σn[±][}]_[ as an edge topology] _[ e][ij][→E][∈]_[R] _[N][a][×][N][a]_[; 1] _[ ≤][i, j][≤][N][a]_ for future interactive behaviors. Each topology element _eij_ can be defined by two braid functions _fi[i][, f] j[ i][∈][σ][i]_[assessing the future interwines along with] **[ Y]** _[i][,]_ **[ Y]** _[j]_[as:] _[e][ij]_[=][max] _[t]_ **[ I]** � _fi[i]_[(] **[y]** _i[t]_[)] _[, f] j[ i]_[(] **[y]** _j[t]_[)] �. Here **I** is an intertwine indicator by segment intersection [77] under lateral coordinates. With favorable properties proved in Appendix B, we can formulate the reasoning task as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0004-05.png)


Agent future **Y**[ˆ] in node term _V_[ˆ] is defined by Gaussian mixtures (GMM) and optimized in Sec. 3.3. The edge topology reasoning _E_[ˆ] can be specified as a probabilistic inference problem by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0004-07.png)


where 1 _≤ i, j ≤ Na_ . Synergistic reasoning structures are then established optimizing _G[∗]_ . 

**Comparative analysis.** To highlight BeTop’s position among various formulations, we first conduct a preliminary analysis to assess behavioral similarity by retrieving future interactive agent pairs using human annotations [35]. Human likeness is quantified by classification metrics, including accuracy and the area under the curve (AUC), with annotated interactive IDs. As depicted in Table 1, labeled BeTop achieves the closest behavioral similarity compared with other well-accepted formulations in the community. Compared with retrieving _k_ nearest strategy ( _k_ = 6) by ground-truth future states, we observe advanced differentiation in non-interactive behaviors (+16 _._ 1% Acc., +4 _._ 13% AUC) by BeTop. We then look into the generic learning-based structure by attention [22] or dynamic graph [67] for interactive behaviors. Despite high accuracy, their inferior AUC scores imply difficulties in retrieving precise interactivity compared with BeTop (+19 _._ 9 AUC). We refer analytical content in Appendix B. This draft for a reasoning framework BeTop prompting joint behaviors. 

## **3.2 BeTopNet** 

As presented in Fig. 3, we introduce the synergistic learning framework reasoning BeTop in response to the series of challenges. It encompasses a Transformer backended encoder-decoder network. With 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Scene Encoder Synergistic Decoder & Reason Heads Imitative Contingency Learning<br>….. ….. 𝝉𝑴 𝓣𝑱<br>ෝ𝒚𝑛 [𝑙] ෝ𝒑𝑛 [𝑙] Planning<br>Transformer  Head<br>Encoder Ƹ𝑒𝑛 [𝑙] ෠𝐘𝑛 [𝑙]<br>max𝑀Ƹ𝑒1<br>….. ….. BeTop  Guided  argmin෡𝐘𝟏 𝐶𝑀(𝝉𝑴, 𝒀 [෡] ) + ෍ 𝑷𝐶𝐽 𝓣𝑱, 𝒀 [෡]<br>𝜖𝑛 [𝑙] Transformer Decoder<br>Agent Map<br>Encoder Encoder 𝜖AV<br>Topology<br>Head × L<br>Prediction<br>….. Head<br>𝐒𝑨 𝐒𝑴 𝐒𝑹 𝐐𝑙,𝑛𝐴 ෡𝒀𝑴 ෡𝒀𝑱<br>…<br>…<br>…<br>Recombine<br>**----- End of picture text -----**<br>


Figure 3: **The BeTopNet Architecture.** BeTop establishes an integrated network for topological behavior reasoning, comprising three fundamentals. Scene encoder generates scene-aware attributes for agent **S** _A_ and map **S** _M_ . Initialized by **S** _R_ and **Q** _A_ , synergistic decoder reasons edge topology _e_ ˆ _[l] n_[and trajectories] **[Y]**[ˆ] _n[l]_[iteratively from topology-guided local attention.][Branched planning] _[ τ][∈]_ **[Y]**[ˆ][1] with predictions and topology are optimized jointly by imitative contingency learning. 

encoded scene semantics **X** ; **M** , the proposed network features a synergistic decoder which reasons and guides BeTop. Reason heads for topology _E_[ˆ] and IPP for _V_[ˆ] comprise the behavioral graph _G_ . 

**Scene encoder.** We leverage a scene-centric coordinate system following planning-oriented principle [7]. Scene attributes comprise historical agent states **X** _∈_ R _[N][a][×][T][h][×][D][a]_ and map polyline inputs **M** _∈_ R _[N][m][×][L][m][×][D][m]_ , where we portion _Nm_ map segments with length _Lm_ from full scene map. Both attributes are encoded separately as **S** _A ∈_ R _[N][a][×][D]_ and **S** _M ∈_ R _[N][m][×][D]_ and concatenated as scene features **S** = [ **S** _A_ ; **S** _M_ ] _∈_ R[(] _[N][a]_[+] _[N][m]_[)] _[×][D]_ . A stack of Transformer encoders with local attention are directly employed in capturing regional interactions from encoded scene semantics **S** _A,_ **S** _M_ . 

**Synergistic decoder.** Retaining encoded scene features **S** _A,_ **S** _M_ , we zoom in the decoding strategy that asks for: 1) interactively reason simultaneous BeTop formulations; 2) selectively decoding of compliant interactive semantics leveraging reasoned topology priors. To this end, we introduce the iterative process of _N_ Transformer decoder layers contributed to all agents, pursuing the basis from [78]. To iron out the scene uncertainties, a multi-modal set of _M_ decoding queries **Q**[0] _A[∈]_[R] _[M][×][D]_ are initialized for multi-agent future trajectories. Meanwhile, relative attributes **S** _R ∈_ R _[N][a][×][N][a][×][D][R]_ are deployed through MLPs as topology features **Q**[0] _R[∈]_[R] _[N][a][×][N][a][×][D]_[for edge topology reasoning.] 

Next, we devise dual infostreams to the iterative decoding process for _V_[ˆ] of future trajectories and _E_[ˆ] of future topology. Given agent _An_ , the decoding process in layer _l_ follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0005-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0005-07.png)


where both future trajectories **Y**[ˆ] _n ∈ V_[ˆ] and interactive topology _e_ ˆ _n ∈ E_[ˆ] in BeTop are decoded in synergistic manners. Reasoned edge topology ˆ _e[l] n[∈]_[R] _[M][×][N][a]_[are garnered by topological decoder with] query broadcasting **Q** _A[l][−]_[1] _[,n]_ ; Reasoning nodes for **Y**[ˆ] _n_ , a Transformer decoder with topology-guided local attention are drafted serving ˆ _e[l] n_[as priors.][We provide further details in Appendix][ C.1][.] 

**Topology-guided local attention.** Querying whole-scene agent semantics results in misaligned interactive agents and sparse attention. This motivates our design for local attention guided by the reasoned topology ˆ _e[l] n[∈]_[R] _[M][×][N][a]_[as priors.][Specifically, we retrieve the top-] _[K]_[index] _[ ϵ][l] n[∈]_[R] _[M][×][K]_ priored from ˆ _e[l] n_[for eventual interactive agents behaviors with] _[ A][n]_[.][Interactive indices are directly] leveraged in gathering **S** _A_ selectively for local cross-attention. This process is formed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0005-10.png)


where _ϵ[l] n_[= argmax] _K_[(ˆ] _[e][l] n_[)][.][Topology-guided agent features] **[ C]** _[l,n] A_[are then aggregated in each layer.] 

5 

**Reason heads.** Given respective decoding features **Q** _[l,n] R_[and] **[ Q]** _[l,n] A_[for each layer, we affix reason] heads accustomed to corresponding formulations for ˆ _en_ and **Y**[ˆ] _n_ . Referred in Eq. (3), the topology head, planning head, and prediction head (IPP heads) are jointly devised by stacked MLPs in reasoning BeTop results. For agent _An_ in each layer, reason heads decode GMM components of future states **y** ˆ _n ∈_ R _[M][×][T][f][ ×]_[5] (referring to ( _µx, µy,_ log _σx,_ log _σy, ρ_ ) per step) with mixture score **p** ˆ _n ∈_ R _[M]_ , _{_ **y** ˆ _n,_ ˆ **p** _n} ∈_ **Y**[ˆ] _n_ , as well as interactive edge topology ˆ _e[l] n[∈]_[R] _[M][×][N][a]_[for BeTop.] 

## **3.3 Imitative Contingency Learning** 

Pursuing the target in Eq. (1), BeTopNet learns end-to-end objectives imitating human-like multi-agent behaviors, integrating compliant behaviors by contingency planning under scenario uncertainties. 

**Imitation learning.** Imitation objectives are firstly established in regulating multi-agent behavioral states **Y** ˆ is defined by the negative log-likelihood (NLL) from best-reasoned components _{_ **Y**[ˆ] _n} ⊂ V_[ˆ] while maximizing their interactive distributions _E_[ˆ] . The imitative _m_ objective _[∗]_ closest tofor ground-truths, as denoted: _LV_ =[�] _[T] t[f] L_ NLL(ˆ **y** _n[m][∗][,t] ,_ ˆ **p** _[m] n[∗][,]_ **[ Y]** _[n]_[)][.][Followed][Eq.][(][2][),][the][behavioral] distributions for edge topology are computed by binary cross-entropy (BCE) given gathered ˆ _e[m] n[∗] ∈_ R _[N][a]_ , formulated as _LE_ =[�] _[N] j[a] H_ (ˆ _e[m] n,j[∗][, e][n,j]_[)][ over] _[ N][a]_[ agents jointly.] 

**Integrated contingency planning.** To integrate compliant behavior learning for _G_ amidst multiagent scenario uncertainties, contingency planning [79, 76] is turned out an apt solution. Bridging immediate safe maneuvers _τM_ to branched planning sets _{τJ }_ with joint prediction, it adjourns uncertain decisions and ensures actual safety. While direct joint prediction may lose diversity [11], reasoned topology _E_[ˆ] serves as a suitable medium distilling future interactive agents for efficient joint combination. Given imitative AV planning outputs _τ ⊂_ **Y**[ˆ] 1 with branching time _tb ∈_ (1 _, Tf_ ), integrating contingency learning asks for a safe short-term plan _τM ∈TM , TM ∈_ R _[M][×][t][b][×]_[2] to full marginal predictions _Y_[ˆ] _M_ = **Y**[ˆ] 2: _Na_ , as well as _M_ branched planning sets _TJ[m]_ = _{τJ_[1:] _[M][b] }m_ guided by joint predictions _Y_[ˆ] _J[m]_[.][This is defined by:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0006-05.png)


where max ˆ _Y CM_ denotes worst-case cost fir _τM_ ; Joint predictions _Y_[ˆ] _J_ with scene probabilities _P_ ( _Y_[ˆ] _J_ ) are recombined by _KM_ interactive agent subsets, indexing _ϵ_ AV _∈_ R _[K][M]_ from sorted AV ˆ topology: _ϵ_ AV = argmax _KM_ (max _M e_ 1). It is described by joint costs _CJ_ in guiding branched planning maneuvers. Specifically, both cost functions are defined by the repulsive potential field [8] discouraging planning proximity with respective prediction formulations. 

**Training loss.** BeTopNet is trained end-to-end through imitative objectives and contingency planning costs by weighted integration for each layer, whenever applicable (for the datasets). Please refer to Appendix C.2 for additional details. 

## **4 Experiment** 

With preliminary analysis in Sec. 3.1, this section further discovers the following questions: **1)** Can BeTop perform compliant planning via BeTopNet, especially in interactive scenarios? **2)** Can BeTop achieve accurate marginal and joint predictions of heterogeneous agents under diverse real-world cases? **3)** Can the formulated BeTop facilitate existing state-of-the-art prediction and planning methods? and **4)** How do the functionalities in BeTopNet affect the performance? 

**Benchmark and metrics.** BeTop is verified on diverse benchmarks. We leverage two large-scale real-world datasets, _i.e._ , nuPlan [47] and Waymo Open Motion Dataset (WOMD) [35], which are presently the most diverse motion datasets in manifesting planning and prediction performance. For planning tasks in nuPlan, there are in total 1M training cases with 8s horizons. 8,300 separated testing set are chosen by _Test14-Hard_ and _Test14-Random_ benchmarks [73] for hard-core and general driving scenes. With further demands verifying maneuvers under interactive cases, we build the _Test14-Inter_ benchmark filtering 1,340 scenes by testing set. Scenarios ranging 15 seconds are tested under three tasks: 1) open-loop (OL), 2) close-loop non-reactive (CL-NR) simulations, and 3) reactive 

6 

Table 2: **Performance comparison of open- and closed-loop planning on nuPlan benchmarks.** BeTopNet positions top average planning score and non-reactive simulation amongst SOTA planning systems by all types (rule, learning, and hybrid), especially under difficult benchmarked scenarios. 

|Type|Mhd|_Test14 Hard_|_Test14 Random_|
|---|---|---|---|
||eto|OLS_↑_<br>CLS-NR_↑_<br>CLS_↑_<br>**Avg.** _↑_|OLS_↑_<br>CLS-NR_↑_<br>CLS_↑_<br>**Avg.** _↑_|
|_Expert_|_Log Replay_|1.000<br>0.860<br>0.688<br>0.849|1.000<br>0.940<br>0.759<br>0.900|
|Rule|IDM [70]|0.201<br>0.562<br>0.623<br>0.462|0.342<br>0.704<br>0.724<br>0.590|
||PDM-Closed [69]|0.264<br>0.651<br>0.752<br>0.556|0.463<br>0.901<br>0.916<br>0.760|
|Hybrid|GameFormer [8]|0.753<br>0.666<br>0.688<br>0.702|0.794<br>0.808<br>0.793<br>0.798|
||PDM-Hybrid [69]|0.738<br>0.660<br>0.758<br>0.719|0.822<br>0.902<br>0.916<br>0.880|
|Learning|UrbanDriver [74]|0.769<br>0.515<br>0.491<br>0.592|0.824<br>0.633<br>0.610<br>0.689|
||PDM-Open [69]|0.791<br>0.335<br>0.358<br>0.495|0.841<br>0.528<br>0.572<br>0.647|
||PlanCNN [72]|0.524<br>0.494<br>0.522<br>0.513|0.629<br>0.697<br>0.675<br>0.667|
||GC-PGP [83]|0.738<br>0.432<br>0.396<br>0.522|0.773<br>0.560<br>0.514<br>0.616|
||PlanTF [73]|0.833<br>0.726<br>0.617<br>0.725|0.871<br>0.865<br>0.806<br>0.847|
||**BeTopNet (Ours)**|**0.840**<br>**0.771**<br>**0.688**<br>**0.766**|**0.876**<br>**0.902**<br>**0.857**<br>**0.878**|



Table 3: **nuPlan closed-loop planning results on the proposed interactive benchmark.** BeTopNet achieves desirable PDMScore, with planning safety, road compliance, and driving progress. 

|Type|Mhd|_Test14 Inter_|_Test14 Inter_|
|---|---|---|---|
||eto|Col. Avoid_↑_<br>Drivable_↑_<br>Direction_↑_<br>Progress_↑_<br>TTC_↑_<br>Comfort_↑_<br>**PDMScore**_↑_||
|_Expert_|_Log Replay_||1.000<br>1.000<br>1.000<br>0.881<br>1.000<br>0.999<br>0.950|
|Rule|PDM-Closed [69]||0.886<br>1.000<br>1.000<br>0.818<br>0.853<br>0.999<br>0.833|
|Learning|Constant Acc.||0.449<br>0.509<br>0.651<br>0.048<br>0.419<br>**1.000**<br>0.108|
||UrbanDriver [74]||0.970<br>0.955<br>0.992<br>0.798<br>0.932<br>**1.000**<br>0.854|
||PlanCNN [72]||0.902<br>0.895<br>0.973<br>0.678<br>0.859<br>0.999<br>0.720|
||PlanTF [73]||0.982<br>0.946<br>0.992<br>0.825<br>**0.952**<br>0.999<br>0.871|
||**BeTopNet (Ours)**||**0.983**<br>**0.960**<br>**0.999**<br>**0.859**<br>0.950<br>0.999<br>**0.894**|



(CL-R) ones by nuPlan simulator. We report the official Planning Scores [80] computed by each task. The motion prediction tasks in WOMD share 487k training scenarios, with 44k validation and 44k testing set separately partitioned under two challenges: 1) The _Marginal prediction challenge_ [81] forecasting multiple scene agents independently; 2) The _Joint prediction challenge_ [82] predicting joint trajectory collections by two interactive agents. Primary metrics of mAP and Soft mAP are ranked for official leaderboards [81, 82]. We leave experimental details in Appendix D. 

## **4.1 Main Result** 

**Performance for interactive planning.** Table 2 demonstrates the planning results under difficult and regular test cases. Notably, BeTopNet marks top average planning scores, achieving +7 _._ 9% in hard cases and excels +6 _._ 2% (CLS-NR) in closed-loop simulations. Specifically, it gains solid improvements against learning-based planners. This can be attributed to topological formulations learning stabilized joint behavioral patterns, boosting +6 _._ 2% _,_ +4 _._ 3% non-reactive simulations by realworld logs and enhancing reactive simulation (+11 _._ 5% _,_ +6 _._ 3%). Contingency objectives enhance uncertainty compliance, leading to expanded results in hard scenarios. Meanwhile, BeTopNet also outperforms rule-based and hybrid planning agents asking for post-optimizations [8] or hefty rules in coinciding with reactive simulation setups [69, 70]. We report +15 _._ 8% and +18 _._ 4% results of non-reactive simulation in hard cases and close performance in general scenes. Moreover, interactive planning compliance is also verified in the proposed _Test14-Inter_ benchmark centering on interactive scenarios. As in Table 3, BeTopNet fosters +3 _._ 8% planning score over previous methods, marking +5 _._ 5% driving progress and +2 _._ 9% driving compliance closest to human performance. Qualitative results of interactive scenarios in Fig. 5(a-d) further corroborate planning compliance by BeTop. 

**Performance for marginal and joint motion prediction.** Marginal prediction results are in Table 4. Without the aids of model ensembles or extra data [25, 59], BeTopNet outperforms existing approaches, manifesting +2 _._ 7% and +3 _._ 4% mAP metrics comparing concurrent methods [53, 86] for compliant predictions. Exhibited strong prediction displacement metric ( _−_ 4 _._ 3% minFDE) over methods using extra pretraining [85], it should be noted that displacement metric is less illustrative as it discounts uncertainty scoring. BeTopNet further outperforms +6 _._ 0% and +26 _._ 1% Soft mAP over multi-agent predictors solely leveraging scenario attention [25] or graph [24]. N. Table 5 exhibits the 

7 

Table 4: **Performance of marginal prediction on WOMD Motion Leaderboard.** BeTopNet surpasses existing motion predictors without model ensemble or using extra data. _[†]_ extra LIDAR data and pretrained model. Primary metric. 

|Set split|Method|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_||mAP_↑_<br>**Soft mAP**_↑_|mAP_↑_<br>**Soft mAP**_↑_|
|---|---|---|---|---|---|---|---|---|
|`Test`|ReCoAt [84]||0.7703|1.6668|0.2437||0.2711|-|
||HDGT [24]||0.5933|1.2055|0.1854||0.3577|0.3709|
||MTR [22]||0.6050|1.2207|0.1351||0.4129|0.4216|
||MTR++ [25]<br>||0.5906|1.1939|0.1298||0.4329|0.4414|
||MGTR_†_ [85]||0.5918|1.2135|0.1298||0.4505|0.4599|
||EDA [53]||**0.5718**|1.1702|**0.1169**||0.4487|0.4596|
||ControlMTR [86]||0.5897|1.1916|0.1282||0.4414|0.4572|
||**BeTopNet (Ours)**||0.5723|**1.1668**|0.1176||**0.4566**|**0.4678**|
|`Val`<br>MTR [22]<br>EDA [53]<br>**BeTopNet (Ours)**|||0.6046|1.2251|0.1366||0.4129|-|
||EDA [53]||**0.5708**|1.1730|0.1178||0.4353|-|
||**BeTopNet (Ours)**||0.5716|**1.1640**|**0.1177**||**0.4416**|-|



Table 5: **Performance of joint prediction on WOMD Interaction Leaderboard.** BeTopNet outperforms in both mAP metrics. Primary metric. 

|Set split|Method|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_||**mAP**_↑_<br>Soft mAP_↑_|**mAP**_↑_<br>Soft mAP_↑_|
|---|---|---|---|---|---|---|---|---|
|`Test`|HeatIRm4 [37]||1.4197|3.2595|0.7224||0.0804<br>0.1239<br>0.1923<br>0.2294<br>0.2326<br>**0.2412**|-<br>-<br>0.1982<br>0.2365<br>0.2368<br>**0.2466**|
||M2I [31]||1.3506|2.8325|0.5538||||
||GameFormer [8]||0.9721|2.2146|0.4933||||
||AMP [32]||0.9073|2.0415|0.4212||||
||MTR++ [25]||**0.8795**|**1.9505**|**0.4143**||||
||**BeTopNet (Ours)**||0.9744|2.2744|0.4355||||
|`Val`|MTR [22]||0.9132|2.0536|0.4372||0.1992<br>0.2344<br>**0.2366**|-<br>-<br>-|
||AMP [32]||**0.8910**|**2.0133**|0.4172||||
||**BeTopNet (Ours)**||0.9304|2.1340|**0.4154**||||



joint prediction results. BeTopNet outperforms all methods in both mAP metrics (+4 _._ 1% _,_ +3 _._ 7% Soft mAP and mAP), presenting robust prediction compliance credit to BeTop formulations for stable future interaction patterns and aligned by local attention in BeTopNet. Particularly, BeTop shows interactive compliance, improving +5 _._ 1% mAP over recent auto-regressive approaches [32], boosting +25 _._ 4% mAP with game-theoretic methods [8] by a large margin. Fig. 5 (e-h) demonstrates the qualitative prediction performance by BeTopNet. At the time of submission, BeTopNet ranked 1 _[st]_ on both WOMD prediction leaderboards [82, 81]. 

## **4.2 Ablation Study** 

Instructed by the last two motivating questions, we investigate the effect of BeTop formulations and components inside BeTopNet. For efficient study, we randomly partition 20% of WOMD train set for prediction, and directly report the planning results by _Test14-Random_ benchmark, which are both representative for the original datasets as verified by [22, 73]. 

**Synergy with existing state-of-the-art methods.** We first study the effect adjoining BeTop as synergistic objectives over existing SOTA methods in planning and prediction. Described in Table 6 and Table 7, BeTop augments +2 _._ 1% and 2 _._ 0% planning score with learning-based and rule-based planners, respectively. Similar compliance effects are also witnessed in guiding strong motion predictors, bringing +1 _._ 1% _,_ +2 _._ 4% improved mAP with _−_ 1 _._ 7% prediction errors of minADE. 

**Number of interactive agents for topology-guided local attention.** In determining the number _K_ future interactive agents for BeTopNet in local attention, we validate the prediction mAP under an array of agent numbers. Shown in Fig. 4, we observe a converging effect, with maximum +3 _._ 7% mAP by the growing number of interactive agents. A drop of _−_ 1 _._ 8% mAP is captured after the peak performance of _K_ = 32. It is due to falsely accepting non-interactive agent values by large _K_ . 

**Different functionalities in BeTopNet.** We further investigate the effects of different functionalities for BeTopNet in Table 8. Compared to the full model, ablations in ID.1 and ID.2 underscore the imitative contingency learning process for costs ( _−_ 2 _._ 9% CLS) and contingency branching ( _−_ 1% CLS-NR). Sole imitative BeTopNet performs the best OLS (ID.2), while the stabilizing effects found in Sec. 4.1 are verified ( _−_ 2 _._ 8% CLS-NR) in comparing ID.3-ID.5 for joint interactive patterns. 

8 

Table 6: **Results of integrating BeTop by strong planning baselines in nuPlan benchmark.** 

Table 7: **Results of integrating BeTop by strong prediction baselines in WOMD benchmark.** 

|Method|nuPlan||Mhd|WOMD|
|---|---|---|---|---|
||OLS_↑_<br>CLS-NR_↑_<br>CLS_↑_<br>**Avg.** _↑_||eto|minADE_↓_<br>minFDE_↓_<br>MR_↓_<br>**mAP**_↑_|
|PDM [69]<br>**PDM [69] +BeTop**|0.463<br>0.898<br>**0.918**<br>0.760||MTR [22]|0.6046<br>1.2251<br>0.1366<br>0.4164|
||**0.488**<br>**0.916**<br>0.902<br>**0.770**||**MTR [22] +BeTop**|**0.5941**<br>**1.2049**<br>**0.1328**<br>**0.4249**|
|PlanTF [73]<br>**PlanTF [73] +BeTop**|0.871<br>0.864<br>0.805<br>0.847||EDA [53]|**0.5708**<br>**1.1730**<br>**0.1178**<br>0.4353|
||**0.878**<br>**0.882**<br>**0.807**<br>**0.856**||**EDA [53] +BeTop**|0.5742<br>1.1853<br>0.1181<br>**0.4407**|



Figure 4: **Results of different interactive agents number for local attention** . We observe a convergence effect for the selection of _K_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0009-04.png)


**----- Start of picture text -----**<br>
mAP Miss Rate<br>0.395 0.149 0.392 0.15<br>0.39 0.148<br>0.385 0.146<br>0.38 0.144<br>0.378 0.143<br>0.375 0.142<br>Number of Interactive agents K<br>**----- End of picture text -----**<br>


Table 8: **Results of BeTopNet planning performance with different components.** Contingency is the key for closed-loop simulation. 

|ID<br>Ablative<br>nuPlan<br>Components<br>OLS_↑_<br>CLS-NR_↑_<br>CLS_↑_<br>0<br>BeTopNet<br>0.876<br>0.902<br>0.857<br>1<br>No branched plan<br>0.879<br>**0.894**<br>**0.830**<br>2<br>No cost learning<br>**0.882**<br>0.888<br>0.807<br>3<br>BeTop only<br>0.877<br>0.876<br>0.804<br>4<br>No local attention<br>0.871<br>0.852<br>0.804<br>5<br>Encoders only<br>0.867<br>0.827<br>0.784|Ablative<br>Components|nuPlan<br>OLS_↑_<br>CLS-NR_↑_<br>CLS_↑_|
|---|---|---|
||BeTopNet|0.876<br>0.902<br>0.857|
||No branched plan<br>No cost learning<br>BeTop only<br>No local attention<br>Encoders only|0.879<br>**0.894**<br>**0.830**<br>**0.882**<br>0.888<br>0.807<br>0.877<br>0.876<br>0.804<br>0.871<br>0.852<br>0.804<br>0.867<br>0.827<br>0.784|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0009-07.png)


**----- Start of picture text -----**<br>
Test14-Hard Test14-Random Test14-Inter<br>a) b) c) d) t=8s t=8s<br>t=0s t=0s<br>Preds Plan<br>Marginal Predictions Joint Predictions<br>e) f) g) h) t=8s p=1<br>t=0s p=0<br>Preds Probs<br>**----- End of picture text -----**<br>


Figure 5: **Qualitative results of planning and prediction in nuPlan and WOMD.** BeTopNet performs compliant reaction simulations in a) yielding for pedestrians; b) cruising in dense traffic. Interactive scenarios (c,d) further present the consistency of contingency learning. BeTopNet predicts both compliant marginal (e,f) and joint (g,h) multi-agent predictions under diverse scenarios. Future interactive behavior patterns can also be consistently reasoned (rendered in light red) with BeTop. 

## **5 Conclusion** 

In this paper, we present BeTop, a topological new-look for multi-agent behavioral formulation. Derived by braid theory, the reasoning tasks for BeTop are drafted supervising joint interactive patterns with integrated prediction and planning. A synergistic network, BeTopNet, is established with an imitative contingency learning process to boost compliant BeTop reasoning. Experiments on nuPlan and WOMD verify BeTopNet’s state-of-the-art performance in prediction and planning. 

**Limitation and Future work.** Current BeTop considers one-step future topology alone, and focuses on prediction and planning. Future work would be centered on developing a recursive version of BeTop in multi-step, multi-agent reasoning and coordination. Another promising direction would be the connectivity of BeTop upon perceptions as tracking for the end-to-end paradigm, as well as an extension on reasoning behaviors under 3D scenarios for multiple autonomous agents. 

9 

## **Acknowledgments** 

This work was supported in part by the Agency for Science, Technology and Research (A*STAR), Singapore, under the MTC Individual Research Grant (M22K2c0079), the ANR-NRF Joint Grant (No.NRF2021-NRF-ANR003 HM Science), the Ministry of Education (MOE), Singapore, under the Tier 2 Grant (MOE-T2EP50222-0002), National Key R&D Program of China (2022ZD0160104), NSFC (62206172), and Shanghai Committee of Science and Technology (23YF1462000). 

## **References** 

- [1] Long Chen, Yuchen Li, Chao Huang, Bai Li, Yang Xing, Daxin Tian, Li Li, Zhongxu Hu, Xiaoxiang Na, Zixuan Li, Siyu Teng, Chen Lv, Jinjun Wang, Dongpu Cao, Nanning Zheng, and Fei-Yue Wang. Milestones in autonomous driving and intelligent vehicles: Survey of surveys. _TIV_ , 2022. 1 

- [2] Wenshuo Wang, Letian Wang, Chengyuan Zhang, Changliu Liu, and Lijun Sun. Social interactions for autonomous driving: A review and perspectives. _Foundations and Trends in Robotics_ , 2022. 1, 2, 17 

- [3] Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-to-end autonomous driving: Challenges and frontiers. _PAMI_ , 2024. 1 

- [4] Sergio Casas, Abbas Sadat, and Raquel Urtasun. MP3: A unified model to map, perceive, predict and plan. In _CVPR_ , 2021. 1, 3 

- [5] Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. ST-P3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In _ECCV_ , 2022. 3 

- [6] Haochen Liu, Zhiyu Huang, Wenhui Huang, Haohan Yang, Xiaoyu Mo, and Chen Lv. Hybrid-prediction integrated planning for autonomous driving. _arXiv preprint arXiv:2402.02426_ , 2024. 

- [7] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, and Hongyang Li. Planningoriented autonomous driving. In _CVPR_ , 2023. 1, 3, 5 

- [8] Zhiyu Huang, Haochen Liu, and Chen Lv. GameFormer: Game-theoretic modeling and learning of transformer-based interactive prediction and planning for autonomous driving. In _ICCV_ , 2023. 1, 2, 3, 6, 7, 8, 16, 19, 20, 21, 22 

- [9] Ye Yuan, Xinshuo Weng, Yanglan Ou, and Kris M Kitani. AgentFormer: Agent-aware transformers for socio-temporal multi-agent forecasting. In _ICCV_ , 2021. 1 

- [10] Yuxiao Chen, Peter Karkus, Boris Ivanovic, Xinshuo Weng, and Marco Pavone. Tree-structured policy planning with learned behavior models. In _ICRA_ , 2023. 1, 3 

- [11] Alexander Cui, Sergio Casas, Abbas Sadat, Renjie Liao, and Raquel Urtasun. LookOut: Diverse multifuture prediction and planning for self-driving. In _ICCV_ , 2021. 3, 6 

- [12] Stefano Pini, Christian S Perone, Aayush Ahuja, Ana Sofia Rufino Ferreira, Moritz Niendorf, and Sergey Zagoruyko. Safe real-world autonomous driving by learning to predict and plan with a mixture of experts. In _ICRA_ , 2023. 

- [13] Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, and Yu Qiao. Trajectory-guided control prediction for end-to-end autonomous driving: A simple yet strong baseline. In _NeurIPS_ , 2022. 1 

- [14] Anthony Hu, Zak Murez, Nikhil Mohan, Sofía Dudas, Jeffrey Hawke, Vijay Badrinarayanan, Roberto Cipolla, and Alex Kendall. FIERY: Future instance prediction in bird’s-eye view from surround monocular cameras. In _ICCV_ , 2021. 1, 3, 16 

- [15] Yihan Hu, Kun Li, Pingyuan Liang, Jingyu Qian, Zhening Yang, Haichao Zhang, Wenxin Shao, Zhuangzhuang Ding, Wei Xu, and Qiang Liu. Imitation with spatial-temporal heatmap: 2nd place solution for nuplan challenge. _arXiv preprint arXiv:2306.15700_ , 2023. 1 

- [16] Reza Mahjourian, Jinkyu Kim, Yuning Chai, Mingxing Tan, Ben Sapp, and Dragomir Anguelov. Occupancy flow fields for motion forecasting in autonomous driving. _RA-L_ , 2022. 1, 3 

- [17] Haochen Liu, Zhiyu Huang, and Chen Lv. Multi-modal hierarchical transformer for occupancy flow field prediction in autonomous driving. In _ICRA_ , 2023. 3 

10 

- [18] Ben Agro, Quinlan Sykora, Sergio Casas, and Raquel Urtasun. Implicit occupancy flow fields for perception and prediction in self-driving. In _CVPR_ , 2023. 1, 16 

- [19] Jinkyu Kim, Reza Mahjourian, Scott Ettinger, Mayank Bansal, Brandyn White, Ben Sapp, and Dragomir Anguelov. StopNet: Scalable trajectory and occupancy prediction for urban autonomous driving. In _ICRA_ , 2022. 1, 3 

- [20] Hongyang Li, Chonghao Sima, Jifeng Dai, Wenhai Wang, Lewei Lu, Huijie Wang, Jia Zeng, Zhiqi Li, Jiazhi Yang, Hanming Deng, Hao Tian, Enze Xie, Jiangwei Xie, Li Chen, Tianyu Li, Yang Li, Yulu Gao, Xiaosong Jia, Si Liu, Jianping Shi, Dahua Lin, and Yu Qiao. Delving into the devils of bird’s-eye-view perception: A review, evaluation and recipe. _PAMI_ , 2024. 1, 3 

- [21] Haochen Liu, Zhiyu Huang, and Chen Lv. Occupancy prediction-guided neural planner for autonomous driving. In _ITSC_ , 2023. 1, 3, 16 

- [22] Shaoshuai Shi, Li Jiang, Dengxin Dai, and Bernt Schiele. Motion transformer with global intention localization and local movement refinement. In _NeurIPS_ , 2022. 1, 3, 4, 8, 9, 18, 19, 20, 22, 23 

- [23] Xiaosong Jia, Li Chen, Penghao Wu, Jia Zeng, Junchi Yan, Hongyang Li, and Yu Qiao. Towards capturing the temporal dynamics for trajectory prediction: a coarse-to-fine approach. In _CoRL_ , 2022. 

- [24] Xiaosong Jia, Penghao Wu, Li Chen, Yu Liu, Hongyang Li, and Junchi Yan. HDGT: Heterogeneous driving graph transformer for multi-agent trajectory prediction via scene encoding. _PAMI_ , 2023. 1, 3, 7, 8, 16, 20 

- [25] Shaoshuai Shi, Li Jiang, Dengxin Dai, and Bernt Schiele. MTR++: Multi-agent motion prediction with symmetric scene modeling and guided intention querying. _PAMI_ , 2024. 1, 7, 8, 16, 20 

- [26] Zhiyu Huang, Chen Tang, Chen Lv, Masayoshi Tomizuka, and Wei Zhan. Learning online belief prediction for efficient pomdp planning in autonomous driving. _arXiv preprint arXiv:2401.15315_ , 2024. 1, 3 

- [27] Zhiyu Huang, Peter Karkus, Boris Ivanovic, Yuxiao Chen, Marco Pavone, and Chen Lv. DTPP: Differentiable joint conditional prediction and cost evaluation for tree policy planning in autonomous driving. In _ICRA_ , 2024. 1, 3 

- [28] Zhiyu Huang, Haochen Liu, Jingda Wu, and Chen Lv. Conditional predictive behavior planning with inverse reinforcement learning for human-like autonomous driving. _TITS_ , 2023. 1, 2, 16 

- [29] Jiquan Ngiam, Vijay Vasudevan, Benjamin Caine, Zhengdong Zhang, Hao-Tien Lewis Chiang, Jeffrey Ling, Rebecca Roelofs, Alex Bewley, Chenxi Liu, Ashish Venugopal, David J Weiss, Benjamin Sapp, Zhifeng Chen, and Jonathon Shlens. Scene Transformer: A unified architecture for predicting future trajectories of multiple agents. In _ICLR_ , 2022. 2, 3 

- [30] Steffen Hagedorn, Marcel Hallgarten, Martin Stoll, and Alexandru Condurache. Rethinking integration of prediction and planning in deep learning-based automated driving systems: a review. _arXiv preprint arXiv:2308.05731_ , 2023. 2 

- [31] Qiao Sun, Xin Huang, Junru Gu, Brian C Williams, and Hang Zhao. M2I: From factored marginal trajectory prediction to interactive prediction. In _CVPR_ , 2022. 2, 3, 8, 16, 20 

- [32] Xiaosong Jia, Shaoshuai Shi, Zijun Chen, Li Jiang, Wenlong Liao, Tao He, and Junchi Yan. AMP: Autoregressive motion prediction revisited with next token prediction for autonomous driving. _arXiv preprint arXiv:2403.13331_ , 2024. 2, 3, 8, 16, 20, 22 

- [33] Jose Luis Vazquez Espinoza, Alexander Liniger, Wilko Schwarting, Daniela Rus, and Luc Van Gool. Deep interactive motion prediction and planning: Playing games with motion prediction models. In _L4DC_ , 2022. 2, 16 

- [34] Wei Zhan, Changliu Liu, Ching-Yao Chan, and Masayoshi Tomizuka. A non-conservatively defensive strategy for urban autonomous driving. In _ITSC_ , 2016. 2 

- [35] Scott Ettinger, Shuyang Cheng, Benjamin Caine, Chenxi Liu, Hang Zhao, Sabeek Pradhan, Yuning Chai, Ben Sapp, Charles R. Qi, Yin Zhou, Zoey Yang, Aurélien Chouard, Pei Sun, Jiquan Ngiam, Vijay Vasudevan, Alexander McCauley, Jonathon Shlens, and Dragomir Anguelov. Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset. In _ICCV_ , 2021. 2, 4, 6, 20, 23 

- [36] Dan Xie, Tianmin Shu, Sinisa Todorovic, and Song-Chun Zhu. Learning and inferring “dark matter” and predicting human intents and trajectories in videos. _PAMI_ , 2017. 2 

11 

- [37] Xiaoyu Mo, Zhiyu Huang, Yang Xing, and Chen Lv. Multi-agent trajectory prediction with heterogeneous edge-enhanced graph attention network. _TITS_ , 2022. 2, 3, 8, 20 

- [38] Zhiyu Huang, Xiaoyu Mo, and Chen Lv. Multi-modal motion prediction with transformer-based neural network for autonomous driving. In _ICRA_ , 2022. 2 

- [39] Yuriy Biktairov, Maxim Stebelev, Irina Rudenko, Oleh Shliazhko, and Boris Yangel. PRANK: motion prediction based on ranking. In _NeurIPS_ , 2020. 2 

- [40] Daehee Park, Hobin Ryu, Yunseo Yang, Jegyeong Cho, Jiwon Kim, and Kuk-Jin Yoon. Leveraging future relationship reasoning for vehicle trajectory prediction. In _ICLR_ , 2023. 3, 16 

- [41] Jiachen Li, Fan Yang, Masayoshi Tomizuka, and Chiho Choi. EvolveGraph: Multi-agent trajectory prediction with dynamic relational reasoning. In _NeurIPS_ , 2020. 2, 3, 4, 16 

- [42] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In _ICLR_ , 2017. 2 

- [43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NeurIPS_ , 2017. 2 

- [44] Christoforos Mavrogiannis, Jonathan DeCastro, and Siddhartha S Srinivasa. Analyzing multiagent interactions in traffic scenes via topological braids. In _ICRA_ , 2022. 2, 3, 16, 17 

- [45] Christoforos Mavrogiannis, Jonathan A DeCastro, and Siddhartha S Srinivasa. Abstracting road traffic via topological braids: Applications to traffic flow analysis and distributed control. _IJRR_ , 2023. 2, 3, 4, 16, 17 

- [46] Emil Artin. Theory of braids. _Annals of Mathematics_ , 1947. 2, 4, 16 

- [47] Napat Karnchanachari, Dimitris Geromichalos, Kok Seang Tan, Nanxiang Li, Christopher Eriksen, Shakiba Yaghoubi, Noushin Mehdipour, Gianmarco Bernasconi, Whye Kit Fong, Yiluan Guo, and Holger Caesar. Towards learning-based planning: The nuplan benchmark for real-world autonomous driving. In _ICRA_ , 2024. 2, 3, 6, 20 

- [48] Mayank Bansal, Alex Krizhevsky, and Abhijit Ogale. ChauffeurNet: Learning to drive by imitating the best and synthesizing the worst. _arXiv preprint arXiv:1812.03079_ , 2018. 3 

- [49] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai. BEVFormer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In _ECCV_ , 2022. 3 

- [50] Alexey Kamenev, Lirui Wang, Ollin Boer Bohan, Ishwar Kulkarni, Bilal Kartal, Artem Molchanov, Stan Birchfield, David Nistér, and Nikolai Smolyanskiy. PredictionNet: Real-time joint probabilistic traffic prediction for planning, control, and simulation. In _ICRA_ , 2022. 3 

- [51] Zetong Yang, Li Chen, Yanan Sun, and Hongyang Li. Visual point cloud forecasting enables scalable autonomous driving. In _CVPR_ , 2024. 3 

- [52] Jiyang Gao, Chen Sun, Hang Zhao, Yi Shen, Dragomir Anguelov, Congcong Li, and Cordelia Schmid. VectorNet: Encoding hd maps and agent dynamics from vectorized representation. In _CVPR_ , 2020. 3 

- [53] Longzhong Lin, Xuewu Lin, Tianwei Lin, Lichao Huang, Rong Xiong, and Yue Wang. EDA: Evolving and distinct anchors for multimodal motion prediction. In _AAAI_ , 2024. 3, 7, 8, 9, 19, 20, 22, 24 

- [54] Charlie Tang and Russ R Salakhutdinov. Multiple futures prediction. In _NeurIPS_ , 2019. 3 

- [55] Siyuan Qi and Song-Chun Zhu. Intent-aware multi-agent reinforcement learning. In _ICRA_ , 2018. 3 

- [56] Thomas Gilles, Stefano Sabatini, Dzmitry Tsishkou, Bogdan Stanciulescu, and Fabien Moutarde. THOMAS: Trajectory heatmap output with learned multi-agent sampling. In _ICLR_ , 2022. 3 

- [57] Thomas Gilles, Stefano Sabatini, Dzmitry Tsishkou, Bogdan Stanciulescu, and Fabien Moutarde. GOHOME: Graph-oriented heatmap output for future motion estimation. In _ICRA_ , 2022. 

- [58] Junru Gu, Chen Sun, and Hang Zhao. DenseTNT: End-to-end trajectory prediction from dense goal sets. In _ICCV_ , 2021. 3 

- [59] Balakrishnan Varadarajan, Ahmed Hefny, Avikalp Srivastava, Khaled S. Refaat, Nigamaa Nayakanti, Andre Cornman, Kan Chen, Bertrand Douillard, Chi Pang Lam, Dragomir Anguelov, and Benjamin Sapp. MultiPath++: Efficient information fusion and trajectory aggregation for behavior prediction. In _ICRA_ , 2022. 3, 7 

12 

- [60] Junha Roh, Christoforos Mavrogiannis, Rishabh Madan, Dieter Fox, and Siddhartha Srinivasa. Multimodal trajectory prediction via topological invariance for navigation at uncontrolled intersections. In _CoRL_ , 2020. 3 

- [61] Christoforos Mavrogiannis, Krishna Balasubramanian, Sriyash Poddar, Anush Gandra, and Siddhartha S Srinivasa. Winding through: Crowd navigation via topological invariance. _RA-L_ , 2022. 3, 16 

- [62] Nigamaa Nayakanti, Rami Al-Rfou, Aurick Zhou, Kratarth Goel, Khaled S Refaat, and Benjamin Sapp. Wayformer: Motion forecasting via simple & efficient attention networks. In _ICRA_ , 2023. 3, 23 

- [63] Zikang Zhou, Luyao Ye, Jianping Wang, Kui Wu, and Kejie Lu. HiVT: Hierarchical vector transformer for multi-agent motion prediction. In _ICCV_ , 2022. 3 

- [64] Alexander Cui, Sergio Casas, Kelvin Wong, Simon Suo, and Raquel Urtasun. GoRela: Go relative for viewpoint-invariant motion forecasting. In _ICRA_ , 2023. 3, 18 

- [65] Tim Salzmann, Boris Ivanovic, Punarjay Chakravarty, and Marco Pavone. Trajectron++: Dynamicallyfeasible trajectory forecasting with heterogeneous data. In _ECCV_ , 2020. 3 

- [66] Luke Rowe, Martin Ethier, Eli-Henry Dykhne, and Krzysztof Czarnecki. FJMP: Factorized joint multiagent motion prediction over learned directed acyclic interaction graphs. In _ICCV_ , 2023. 3, 16 

- [67] Wenjie Luo, Cheol Park, Andre Cornman, Benjamin Sapp, and Dragomir Anguelov. JFP: Joint future prediction with interactive multi-agent modeling for autonomous driving. In _CoRL_ , 2022. 3, 4, 16 

- [68] Sergio Casas, Cole Gulino, Simon Suo, Katie Luo, Renjie Liao, and Raquel Urtasun. Implicit latent variable model for scene-consistent motion forecasting. In _ECCV_ , 2020. 3 

- [69] Daniel Dauner, Marcel Hallgarten, Andreas Geiger, and Kashyap Chitta. Parting with misconceptions about learning-based vehicle motion planning. In _CoRL_ , 2023. 3, 7, 9, 18, 19, 20, 21, 23 

- [70] Martin Treiber, Ansgar Hennecke, and Dirk Helbing. Congested traffic states in empirical observations and microscopic simulations. _Physical review E_ , 2000. 7, 20, 21 

- [71] Peng Hang, Chen Lv, Yang Xing, Chao Huang, and Zhongxu Hu. Human-like decision making for autonomous driving: A noncooperative game theoretic approach. _TITS_ , 2020. 3 

- [72] Katrin Renz, Kashyap Chitta, Otniel-Bogdan Mercea, A. Sophia Koepke, Zeynep Akata, and Andreas Geiger. PlanT: Explainable planning transformers via object-level representations. In _CoRL_ , 2022. 3, 7, 20, 21 

- [73] Jie Cheng, Yingbing Chen, Xiaodong Mei, Bowen Yang, Bo Li, and Ming Liu. Rethinking imitation-based planner for autonomous driving. In _ICRA_ , 2024. 6, 7, 8, 9, 18, 20, 21, 22, 24 

- [74] Oliver Scheel, Luca Bergamini, Maciej Wolczyk, Bła˙zej Osi´nski, and Peter Ondruska. Urban Driver: Learning to drive from real-world demonstrations using policy gradients. In _CoRL_ , 2021. 3, 7, 20, 21 

- [75] Peter Karkus, Boris Ivanovic, Shie Mannor, and Marco Pavone. DiffStack: A differentiable and modular control stack for autonomous vehicles. In _CoRL_ , 2022. 3 

- [76] Tong Li, Lu Zhang, Sikang Liu, and Shaojie Shen. MARC: Multipolicy and risk-aware contingency planning for autonomous driving. _RA-L_ , 2023. 3, 6 

- [77] Franklin Antonio. Faster line segment intersection. In _Graphics Gems III (IBM Version)_ , pages 199–202. Elsevier, 1992. 4 

- [78] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. DAB-DETR: Dynamic anchor boxes are better queries for detr. In _ICLR_ , 2022. 5 

- [79] Jason Hardy and Mark Campbell. Contingency planning over probabilistic obstacle predictions for autonomous road vehicles. _TRO_ , 2013. 6 

- [80] Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom, and Sammy Omari. NuPlan: A closed-loop ml-based planning benchmark for autonomous vehicles. _arXiv preprint arXiv:2106.11810_ , 2021. 7, 20, 23 

- [81] Waymo. Waymo open dataset motion prediction challenge 2024. `https://waymo.com/open/ challenges/2024/motion-prediction/` . 7, 8, 20, 22 

13 

- [82] Waymo. Waymo open dataset interaction prediction challenge 2021. `https://waymo.com/open/ challenges/2021/interaction-prediction/` . 7, 8, 20, 22 

- [83] Marcel Hallgarten, Martin Stoll, and Andreas Zell. From prediction to planning with goal conditioned lane graph traversals. In _ITSC_ , 2023. 7, 20, 21 

- [84] Zhiyu Huang, Xiaoyu Mo, and Chen Lv. ReCoAt: A deep learning-based framework for multi-modal motion prediction in autonomous driving application. In _ITSC_ , 2022. 8 

- [85] Yiqian Gan, Hao Xiao, Yizhe Zhao, Ethan Zhang, Zhe Huang, Xin Ye, and Lingting Ge. MGTR: Multi-granular transformer for motion prediction with lidar. In _ICRA_ , 2024. 7, 8 

- [86] Jiawei Sun, Chengran Yuan, Shuo Sun, Shanze Wang, Yuhang Han, Shuailei Ma, Zefan Huang, Anthony Wong, Keng Peng Tee, and Marcelo H Ang Jr. ControlMTR: Control-guided motion transformer with scene-compliant intention points for feasible motion prediction. _arXiv preprint arXiv:2404.10295_ , 2024. 7, 8, 20 

- [87] Mitchell A Berger. Topological invariants in braid theory. _Letters in Mathematical Physics_ , 2001. 16 

- [88] Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for open-loop end-to-end autonomous driving? In _CVPR_ , 2024. 18, 20 

- [89] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. PointNet: Deep learning on point sets for 3d classification and segmentation. In _CVPR_ , 2017. 18 

- [90] Zetong Yang, Li Jiang, Yanan Sun, Bernt Schiele, and Jiaya Jia. A unified query-based paradigm for point cloud understanding. In _CVPR_ , 2022. 19 

- [91] Zikang Zhou, Jianping Wang, Yung-Hui Li, and Yu-Kai Huang. Query-centric trajectory prediction. In _CVPR_ , 2023. 19 

- [92] Jie Cheng, Yingbing Chen, and Qifeng Chen. PLUTO: Pushing the limit of imitation learning-based planning for autonomous driving. _arXiv preprint arXiv:2404.14327_ , 2024. 19, 20, 21 

- [93] NAVSIM Contributors. NAVSIM: Data-driven non-reactive autonomous vehicle simulation. `https: //github.com/autonomousvision/navsim` , 2024. 20 

- [94] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. NAVSIM: Data-driven non-reactive autonomous vehicle simulation and benchmarking. _arXiv_ , 2406.15349, 2024. 20 

- [95] Lan Feng, Mohammadhossein Bahari, Kaouther Messaoud Ben Amor, Éloi Zablocki, Matthieu Cord, and Alexandre Alahi. UniTraj: A unified framework for scalable vehicle trajectory prediction. In _ECCV_ , 2024. 23 

14 

## _**Appendix**_ 

|**A **|**Discussions**|**16**|
|---|---|---|
|**B**|**Properties of BeTop**|**16**|
|**C **|**Implementation Details**|**18**|
||C.1<br>Model Structure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|18|
||C.2<br>Imitative Contingency Learning<br>. . . . . . . . . . . . . . . . . . . . . . . . . . .|19|
|**D **|**Experimental Setup Details**|**20**|
||D.1<br>Planning on nuPlan . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|20|
||D.2<br>Prediction on WOMD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|20|
||D.3 Training Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|21|
|**E **|**Additional Quantitative Results**|**21**|
||E.1<br>Planning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|21|
||E.2<br>Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|22|
|**F**|**Additional Ablation Studies**|**22**|
|**G **|**Additional Qualitative Results**|**23**|
|**H **|**License of Assets**|**23**|



15 

## **A Discussions** 

Towards a better understanding of this work, we supplement intuitive questions that may raise. Note that the following list does _not_ indicate the manuscript was submitted to a previous venue or not. 

## **Q1:** _How does BeTop bridge and discern with dense, sparse, and topological representations?_ 

BeTop is derived from braid theory [46], reasoning the topology that explicitly labels consensual interactions as dense, occupancy-like intertwines from sparse, braided future trajectories of multiagent behaviors. Unlike fixed occupancy [14, 21, 18], BeTop dynamically forecasts behavioral interactions by agents collectively, serving as a guiding medium for node reasoning in joint planning and prediction compared to standard sparse predictions [25, 24, 66]. Differentiating from typical topological approaches, BeTop explicitly formulates coordinated multi-agent behaviors and reasons topology in occupancy manners, rather than relying on implicit relation graph learning [41, 67, 40] or complex braid inference [44, 45, 61]. 

## **Q2:** _Why is BeTop allied with contingency instead of conditional or game-theoretic reasoning?_ 

Contingency plays the most suitable role in BeTopNet, addressing multi-agent uncertainty by deferring uncertain planning with long-term joint predictions from interactive agents under behavioral compliance. This aligns with BeTop’s reasoning targets, which aim to achieve a one-shot consensus among all behaviors by formulating future behavior as coordinated joint interactions. This synergy alleviates the challenges led by game-theoretic reasoning [8, 33] or conditional integration [31, 28, 32], which are struggled by multi-step interactive rollouts and unstable joint behavioral patterns. 

## **Q3:** _What would be the broader impact and future direction?_ 

BeTop steps the first trial towards an explicit topological formulation and reasoning paradigm for multi-agent interactive behaviors. This serves as a basis in exploring immense interactive behaviors in the real-world, and is reasoned by the autonomous agents jointly in a coordinated manner. For instance, BeTop with enlarged scalability and dimension may summarize the topology for all behaviors in 3D scenarios or even larger spaciality, and may reasoned through end-to-end BeTopNet as a foundation behavioral model. Moreover, we can consider BeTop’s capability as collective maneuvers, which can be further leveraged in coordinating naturalistic and efficient decision-making for multiple autonomous agents. 

## **B Properties of BeTop** 

In this section, we supplement additional properties that characterize the formulation of Behavioral Topology (BeTop). Analytically, BeTop is highlighted with: 1) geometric invariant (Theorem B.1); 2) approximated topological invariant (Theorem B.2); and 3) asymmetric topology (Theorem B.3). 

**Theorem B.1** ( **Geometric Invariant** ) **.** The topology results of _E ⊂G_ in BeTop remain unchanged given arbitrary geometrical transformations for the collective scene trajectories **Y** _n_ . 

_Proof._ Given arbitrary rotation **H** _∈_ R[2] _[×]_[2] and shifting **b** _∈_ R[2] . Consider the mapping _g_ for elementary topology _eij ∈E_ from future trajectories ( **Y** _i,_ **Y** _j_ ), _i_ = _j ∈_ [1 _, Na_ ], the local transformation _f_ : _g → h ◦ f_ to _i_ ’s coordinate is invariant, such that _h_ ( _f_ ( **Y** _i,_ **Y** _j_ )) = _h_ ( _f_ ( **HY** _i_ + **b** _,_ **HY** _j_ + **b** )). Hence, given the function sets _fj[i][∈][σ][i][⊂][f,]_ **[ I]** _[ ∈][h]_[ defined in Sec.][ 3.1][;] _[ e][ij][∈E]_[is also invariant.] 

_Remark_ 1 _._ The Theorem B.1 proves the behavioral stability of BeTop given arbitrary multi-agent trajectories patterns for planning and predictions. Any rotations and movement of the original scene will not interfere with the formulated results of BeTop. 

**Definition B.1** ( **Topological Invariant** ) **.** Given future trajectory pairs ( **Y** _i,_ **Y** _j_ ), _i_ = _j ∈_ [1 _, NA_ ] with certain current heading ( _θi_[0] _[, θ] j_[0][)][,][the][sum][of][future][relative][angles][(winding][number)] _[w][ij]_[=] 21 _π_ � _T_ 0 _f_[∆] _[θ] ij[t]_[form its first-order [][87][] topological invariant.] 

_Proof._ Consider the polar representation for the closed form _ψi_ ( _t_ ) = _||ψi_ ( _t_ ) _||e[iθ][i]_[(] _[t]_[)] . Where _ψi_ : [0 _, Tm_ ] _→_ C _\{_ 0 _}, i ∈_ [0 _, n_ ], we can define the winding function _λi_ ( _t_ ) = 21 _πi_ � _ψi[dz/z, z]_[=] 

16 

_ψi_ ( _t_ ) _, t ∈_ [0 _, Tm_ ]. This Cauchy formula [2] can be further integrated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0017-01.png)


We are interested in the real (first-order) part of _λi_ ( _t_ ) which is an invariant topologically. Hence, the trajectory pairs ( **Y** _i,_ **Y** _j_ ) _,_ 0 _< t ≤ Tf < Tm_ can be described as: **Y** _i_ = _ψi_ : (0 _, Tf_ ]. The joint invariant across future _wij_ =[�] _[T] t[f]_[(] _[λ][i]_[(] _[t]_[)] _[ −][λ][j]_[(] _[t]_[))][ then becomes:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0017-03.png)


As the current heading pair ( _θi_[0] _[, θ] j_[0][)][ is certain,][the invariant becomes] _[ w][ij]_[=] 21 _π_ � _T_ 0 _f_[∆] _[θ] ij[t]_[which] proofs the definition. 

**Corollary B.1.1.** Given any ∆ _θij[t][∈]_[[] _[−][π]_ 2 _[,][π]_ 2[]][, where][ 0] _[<][i.j][<][N][a][, t][∈]_[(0] _[, T][f]_[]][, and the constant] _ηi, ηj ∈_ R, the transformed _wij[T]_[=][ �] _[T]_ 0 _[f]_[(] _[η][j]_[ sin ∆] _[θ] ij[t][−][η][i]_[ sin] _[ θ] i[t]_[)][ is also topological invariant.] 

_Proof._ The defined function sin( _·_ ) is a monotone mapping under [ _−[π]_ 2 _[,][π]_ 2[]][.][Hence, this firstly enables] � _T_ 0 _f[η][i]_[ sin] _[ θ] i[t]_[uniquely defines] **[ Y]** _[i]_[.][More than that,] _[ w] ij[T]_[is the unique mapping value of] _[ w][ij]_[with] **[ Y]** _[i]_ under defined transformation, and thereby keep the invariant property. 

**Theorem B.2.** The edge topology _E ⊂G_ is an approximate of topological invariant, so that _eij ∈E,_ 0 _< i, j ≤ Na_ is characterized by _wij_ . 

> _Proof._ Given future trajectories **Y** _i,_ **Y** _j_ , we consider the braid functions _σi_ maps monotonically increased transformations _fi[i][, f] j[ i]_[to] _[ i]_[’s local coordinate,as defined in Sec.][ 3.1][.][We assume a continuous] future horizon (0 _, Tf_ ] where the headings for _fj[i]_[(] **[Y]** _[j]_[)][ is defined by relative angles][ ∆] _[θ][ij]_[(] _[t]_[)][.][Thereby,] _t_ the transformed lateral trajectory for agent _j_ can be formed as: �0 _[η][j]_[ sin ∆] _[θ][ij]_[(] _[t]_[)] _[dt]_[.][Similarly,] _[ f] i[ i]_[(] **[Y]** _[i]_[)] _t_ can be formed as �0 _[η][j]_[ sin] _[ θ][i]_[(] _[t]_[)] _[dt]_[, where] _[ η][i][, η][j]_[denotes small constant step lengths.These form the] original intersection function **I** in Sec. 3.1 as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0017-09.png)


where **I** ( _·_ ) = 0 denotes braid intertwine for interactive behaviors. As the term in Corollary B.1.1 of _wij[T]_[(the sum of the right term) is an approximate (discretization) of the right term, this proves the] edge topology _eij ∈E →_ max _Tf_ **I** ( _·_ ) as the approximation of topological invariant. 

_Remark_ 2 _._ The Theorem B.2 proves the generality of BeTop in terms of future interactive behavioral patterns. The approximated topological invariant property prompts a representative of various future states sharing similar behavioral or identical interactive patterns by BeTop. 

**Theorem B.3** ( **Asymmetric Topology** ) **.** Edge topology _E ⊂G_ is not symmetric such that _∃i, j, eij ̸≡ eji,_ 0 _< i, j ≤ Na_ . 

_Proof._ Given Eq. (9) defined in Theorem B.2,we can always construct a case _∃ti, tj ∈_ (0 _, Tf_ ], the intersection **I** � _fi[i]_[(] **[y]** _i[t][i]_[)] _[, f] j[ i]_[(] **[y]** _j[t][i]_[)] � = 0, but **I** ( _fi[j]_[(] **[y]** _j[t][j]_[)] _[, f] i[ j]_[(] **[y]** _i[t][j]_[))] _[ ̸]_[= 0][, which prove the claim.] 

_Remark_ 3 _._ The Theorem B.3 proves a more naturalistic interactive behavior of BeTop. It is likely in a real-world scenario that the future behavior of agent _Ai_ is interacted by agent _Aj_ , while _Aj_ does not. 

**Computational complexity.** The complexity in computing full _E_ is _O_ ( _Na_[2][)][.][In][practice,][we] downscale the sourced as the agents of interests _NI < Na_ , such that _O_ ( _NI Na_ ). It is much less than braid sequence inference [45] with maximum _O_ (( _Na −_ 1) _[N][a]_ ) computational costs. Further analytical proof of computational efficiency leveraging braids can be found in [44]. 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0018-00.png)


**----- Start of picture text -----**<br>
a) 𝑪𝒍,𝒏𝑨 ∈𝐐𝒍,𝒏𝑨 b)<br>[M,K] Add & Norm 𝐒𝑨 MLP [M, 𝐍𝐚 , 𝐃𝐚 ]<br>ො𝒆𝒏 [𝒍]  [M, 𝐍𝐚 ] 𝝐𝒏 [𝒍] FFN Concat. 𝐐𝒍,𝒏𝑹<br>Topology Topology-guided<br>𝐒𝑨 Head Local Attention 𝐐𝒍−𝟏,𝒏𝑨 MLP 𝐐𝒍−𝟏,𝒏𝑹<br>V K Q<br>Topology decoder<br>[ 𝐍𝐚 ,D] 𝐐𝒍,𝒏𝑹  [M, 𝐍𝐚 ,D] [M,K,D] Add & Norm c) 𝐐𝒍,𝟏𝑨<br>෠𝐘1<br>Topology Multi Head<br>Decoder Self-Attention<br>V K Q 𝝉𝑴 [M, 𝑴𝑱 ,D] 𝝉𝑱<br>𝜏⊂ 𝐘 [෡] 1 [M, 𝒕𝒃 ,2] [ 𝑴𝑴𝑱 , 𝑻𝒇 −𝒕𝒃 ,2]<br>𝐒𝑹𝒏 [ 𝐍𝐚 ,D] 𝐐𝒍,−𝟏 𝒏𝑨 [M,D]<br>BeTop Net decoder layer Planning Head<br>… …<br>…<br>Concat.<br>MLP<br>MLP MLP MLP<br>**----- End of picture text -----**<br>


Figure 6: **Structural details in BeTopNet.** a) The learning structure of single synergistic decoder layer featuring `TransDecoder` and `TopoAttn` design; b) The structure inside topology decoder network of `TopoDecoder` ; c) Branched planning head design corresponding to contingency planning. 

## **C Implementation Details** 

In this section, we instantiate further details for BeTop on the configurations for BeTopNet structure, and provide contingency learning paradigms for both prediction and planning challenges. 

## **C.1 Model Structure** 

Subjecting to different testing requirements defined in nuPlan and WOMD, we set up two model variants for BeTopNet in formulating planning and prediction challenges. Apart from topology reasoning for interactive behaviors, for planning challenges, BeTopNet integrates both tasks of prediction and planning. In the prediction challenges under marginal and joint settings, BeTopNet is allocated only with the prediction parts. The structural details are illustrated below. 

**Scene inputs.** Carving the driving scenarios involves historical agent states **X** and map polylines **M** as scene inputs. For planning settings, we collect scene agent states with past _Th_ = 2 seconds at 10Hz, leaving basic kinematic states as ( _x, y, vx, vy, θ_ ), joining with agent shapes and types. We only keep the current state for ego vehicle (AV) in preventing closed-loop gap [73, 69, 88] with open-loop training as recently discussed in the community. _Nm_ = 256 segments of the map with length _Lm_ = 20 are gathered by scene-centric manners considering positions, traffic lights, and speed limits. The prediction task is built on WOMD considering _Th_ = 1 seconds at 10Hz with scenarios of larger scalability. It is followed by full scene agents with _Nm_ = 768 map segments of identical states for both **X** and **M** . 

**Scene encoder.** Both scene attributes **X** _,_ **M** are firstly encoded leveraging layered point encoder [89] to hidden dimension _D_ shared throughout the BeTopNet structures, with _D_ = 128 for planning and _D_ = 256 for prediction tasks. A stack of Transformer encoders is then devised with 4 layers in planning and 6 in prediction for **S** _A,_ **S** _M_ . Due to scalable settings for prediction tasks, local attention with the nearest 16 keys is built in each layer. Following [22, 73], the dense prediction head is adopted for all agents after the encoder, enhancing future semantics. 

**Synergistic decoder.** Depicted in Fig. 6, the decoder structure is founded by an iterative stack of _L_ Transformer decoders querying _M_ modes of future trajectories **Y**[ˆ] with dual stream in reasoning topology _E_[ˆ] . Consisting of _L_ = 6 and _L_ = 4 decoders for prediction and planning respectively, it is initialized jointly by relative features **Q**[0] _R[,n]_ and decoding queries **Q**[0] _A[,n]_[.][Relative attributes] **[ S]** _[R]_[are] computed efficiently following [64] for relative distance and headings. _M_ = 6 learnable embedding are devised as **Q**[0] _A[,n]_ for planning, and we utilize the anchored ones [22] with _M_ = 64 in prediction tasks. As displayed in Fig. 6(a), dual queries **Q** _[j] A[−]_[1] _[,n] ,_ **Q** _[j] R[−]_[1] _[,n]_ are served from the last level. The 

18 

decoding process (following Eq. (3)) iteratively reasons ˆ _e[l] n_[from] `[ TopoDecoder]`[, serving as a prior] guiding agent semantics by `TopoAttn` inside `TransDecoder` , which concurrently aggregates scene semantics from agents **S** _A_ and maps **S** _M_ . Expressly, the structure of `TopoDecoder` (Fig. 6(b)) comprises simple MLPs and update **Q** _[l,n] R_[by][concatenation][sourcing][query][features] **[Q]** _[j] A[−]_[1] _[,n]_ with agent semantics **S** _A_ , and connect residuals **Q** _[l] R[−]_[1] _[,n]_ from last layer. Decoding queries **Q** _[l,n] A_[is updated] by a concatenation from aggregated agent feature **C** _[l,n] A_[, map features] **[ C]** _[l,n] M_[, and agent semantics] **[ S]** _A[n]_ directly from encoder. Agent feature **C** _[l,n] A_[is aggregated by] `[ TopoAttn]`[, where the local attention is] devised using deployments from [90] indexing _K_ = 32 agents from reasoned topology ˆ _e[l] n_[.][We omit] the aggregation process for map features, which performs the vanilla Transformer decoder structure for planning and dynamic collection form by [22] under prediction tasks for hefty map features. **Reason heads.** Following the contents in Sec. 3.2, reason heads in decoding prediction and topology follow simple MLPs given respective decoding queries. For the planning head, it leverages a cascaded design for branched contingency planning with multi-modalities (Fig. 6(c)). Specifically, the short-planning _τM_ is decoded by the AV future states _{y_ ˆ1[1:] _[t][b] }m_ from first stage head with _m ∈ M_ modes, where _tb_ = 3 denotes the branching time. They are then detached and leveraged as prior for the branched planning. Successive MLPs project and reshape the short-term contingency prior by R _[M][×][M][J][×][D]_ for _MJ_ = 6 branches planning _TJ[m]_ under each of _τM[m]_[.][Further][concatenated][by] broadcasted decoding queries, the planning head generates _M · MJ_ trajectories **Y**[ˆ] 1 for AV. 

## **C.2 Imitative Contingency Learning** 

**Efficient joint prediction recombination.** Retrieving the top-performed joint predictions from full marginal predictions _Y_[ˆ] _M_ sequentially is time-consuming with exponentially complexity. Hence, we firstly downscale the potentially interested agents _NI_ by sorting the AV-reasoned topology ˆ _e[L]_ 1[with] the largest _KM_ = 4 value as index given the planning task. For the joint prediction task, _NI_ = 2 is annotated already from the original data. Then, we leverage the tensor broadcasting mechanism in efficiently retrieving _M_ = 6 largest joint distributions _P_ ( _Y_[ˆ] _J[M]_[)][ and joint trajectories] _[Y]_[ˆ] _J[M]_ from _NI_ interacted agents. Given a tensor **P** _J ∈_ R _[M][ NI]_ initialized by ones, the joint score is computed by _NI_ times of iterative broadcasting **p** ˆ _n_ on the _n_ -th dimension for **P** _J_ as _PJ_ = max _M_ � _Nn I_ **[P]**[(] _J[n]_[)] _⊗_ **p** ˆ _n_ . This process only costs 1.6ms in computing _NI_ = 4 joint predictions for contingency planning. 

**Imitative contingency objectives.** Followed by learning objectives derived in Sec. 3.3, the imitation objectives for each layer can be represented as _L_ IL = _LV_ + _λ_ 1 _LE_ . _λ_ 1 = 50 weighting BCE loss for edge topology reasoning, the NLL loss for _LV_ is formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0019-04.png)


where _dx, dy_ denotes the difference with ground-truths. In determining the component _m[∗]_ , we leverage a winner-take-all (WTA) strategy [91] in planning by measuring the average displacements (ADE) with groung-truths. For prediction tasks, _m[∗]_ is selected from the closest anchor as in [53]. For the learnable cost functions max _CM_ ( _·_ ) _, CJ_ ( _·_ ) in contingency planning, we leverage the repulsive potential field [8] delineating planning with prediction by _ϕ_ = min _d_ 1 _/_ (1+ _d_ ( _τ,_ ˆ **y** )). For max _CM_ ( _·_ ), _ϕ_ is gathered across _Tf_ considering the worst case under full marginal prediction _Y_[ˆ] _M_ comprising _Na_ = 32 scene agents. For the branched cost _CJ_ ( _·_ ), _ϕ_ for each branch is computed considering joint prediction from _NI_ = 4 agents. Following the objective defined in Eq. (6), the learnable contingency cost is defined as: _L_ CL = _CM_ +[�] _[M] m[P]_[( ˆ] _[Y] J[m]_[)] _[C] J[m]_[.][Hence, the general objectives for] planning become: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0019-06.png)


where _λ_ 2 = 5 is the contingency costs weight. Prediction tasks are updated only by _L_ IL. 

**Inference.** Different from the training process, for the planning task we directly select the full planning trajectory of _Tf_ = 8 seconds by highest scoring _τ[∗]_ = argmax _C_ **Y**[ˆ] 1, subjecting to theˆ original task settings in nuPlan. The scoring results are a combination from original confidence **p** 1 ˆ and the short-term cost _CM_ [69]: _C_ = **p** 1 + _λmCM_ , where _λm_ = 0 _._ 5 facilitates short-term planning compliance [92]. For the prediction task, a post-processing module following [53] is leveraged in selecting _M_ = 6 marginal or joint trajectories of _Tf_ = 8 seconds among 3 agent types in WOMD. 

19 

## **D Experimental Setup Details** 

In this section, we provide extra details demonstrated in Sec. 4 for the experiment setups, including detailed settings for the proposed benchmark, testing metrics, state-of-the-art baselines, and training. 

## **D.1 Planning on nuPlan** 

**Testing metrics.** For open-loop planning tests, the open-loop score (OLS) serves as the general statistics weighting displacement metrics and miss rates. For closed-loop simulations, both metrics (CLS, CLS-NR) are weighted by a series of statistics measuring 1) driving safety, 2) planning progress, 3) driving comforts, and 4) rule obeying. The PDMScore [93, 94] compared in Table 3 is basically a replica of the closed-loop score for efficient computations. It is denoted as: 

`PDMScore` = `CA` _·_ `DAC` _·_ `DDC` _·[w]_[1] `[TTC]`[ +] _[ w]_[2] `[DC]`[ +] _[ w]_[3] `[EP]` _,_ (12) � _wi_ 

where the sub-metrics are abbreviations referred in Table 3. All general metrics range from 0 to 1. 

**Test14-Inter.** We launch the _Test14-Inter_ benchmark in verifying the planning systems under typical corner cases containing rich social interactions, or dynamic profiles by complex map forms. This is highly motivated by the issues raised in [69, 88], that massive scenarios may also be completed by a simple motion model. Specifically, we adopt a mining heuristic defining corner cases by which human experts excel but the motion model (constant acceleration vehicle, CAV) fails. For efficient mining, we directly assess planning results by PDMScore and define the criteria as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0020-07.png)


where _γ_ = 0 _._ 1 denotes a scoring threshold for cases that cannot be easily solved by regular motion profiles of the planning maneuvers. As future work, we aim to explore more interactive scenarios aggravating by BeTop as an enhancement. 

**Val14.** In pursuing comprehensive comparisons with current methods, we also manifest BeTopNet in the _Val14_ set proposed in [69]. It is a subset of 1040 scenes from the validation set. However, since a portion of validation scenes are shared with the training set in nuPlan [80], we argue this is less representative of testing fairness for learning-based methods. Hence, we only place it as supplementary. 

**Baselines.** For all baselines presented in the planning task, we directly report their previous benchmark results. Additional results in the proposed benchmark (Table 3) and ablation studies (Table 6) are re-implemented by the official releases [69, 72, 47, 69]. Expressly, we study the state-of-the-art planning systems categorized by: 1) Rule-based: performing maneuvers by designate rules with the reactive agents [70] or mimicking the planning score [69]; 2) Hybrid: incorporating rules [69] or post-optimizations [8] with a learning-based model; and 3) Learning-based: end-to-end planning with GNN [83, 74] or Transformer [72, 73] enabled models, as well as concurrent methods [92] augmented by representation learning. For ablation studies in Table 6, BeTop is trained directly by the proposed topology decoder with the original PlanT [73] pipeline. For the PDM [69] as a rule-based planning system, we integrate BeTop by replacing the original rule-based motion model with predictions generated from BeTopNet. 

## **D.2 Prediction on WOMD** 

**Testing metrics.** For the prediction task, the mean AP (mAP) and Soft-mAP scores are assigned as the primary metrics in computing multi-modal predictions modeled by marginal or joint distributions [35, 81, 82]. Displacement metrics of minADE and minFDE provide the multi-modal prediction errors closest to ground truths without considering the prediction scores. 

**Baselines.** We also directly provide the prediction results displayed on the official leaderboards in Tables 4 and 5. Ablation studies in Table 7 are reproduced by the official codes [22, 53]. The prediction performance of BeTopNet is compared against SOTA baselines by: 1) GNN-enabled interactive graph [37, 24]; 2) conditional or game-theoretic behavioral interactions [31, 8]; 3) DETRbased Transformer attentions [22, 25, 53, 86]; and 4) auto-regressive modeling [32]. 

20 

Table 9: **Detailed nuPlan closed-loop simulation results in Val14 benchmark.** BeTopNet highlights leading results among SOTA methods in safety and compliance, outperforms learning-based agents. 

|Type|Method|CA_↑_|_Val14_<br>TTC_↑_<br>DDC_↑_<br>DC_↑_<br>EP_↑_<br>Speed_↑_<br>**CLS-NR**_↑_<br>**CLS-R**_↑_|_Val14_<br>TTC_↑_<br>DDC_↑_<br>DC_↑_<br>EP_↑_<br>Speed_↑_<br>**CLS-NR**_↑_<br>**CLS-R**_↑_|
|---|---|---|---|---|
|_Expert_|_Log Replay_|0.987|0.944<br>0.981<br>0.993<br>0.989<br>0.965|0.937<br>0.812|
|Rule|IDM [70]<br>PDM-Closed [69]|0.909<br>0.981|0.834<br>0.941<br>0.944<br>0.862<br>0.973<br>0.933<br>0.998<br>0.955<br>0.921<br>0.998|0.793<br>0.793<br>0.932<br>0.930|
|Hybrid|GameFormer [8]|0.943|0.867<br>0.948<br>0.933<br>0.890<br>0.987|0.829<br>0.838|
|Learning|UrbanDriver [74]<br>PDM-Open [69]<br>PlanCNN [72]<br>GC-PGP [83]<br>PlanTF [73]<br>PLUTO [92]<br>**BeTopNet (Ours)**|0.856<br>0.745<br>0.869<br>0.858<br>0.941<br>0.961<br>**0.966**|0.803<br>0.908<br>**1.000**<br>0.808<br>0.915<br>0.691<br>0.879<br>0.995<br>0.698<br>0.977<br>0.814<br>0.850<br>0.814<br>0.806<br>0.980<br>0.801<br>0.897<br>0.900<br>0.603<br>**0.993**<br>0.907<br>0.968<br>0.937<br>**0.898**<br>0.977<br>**0.933**<br>0.985<br>0.964<br>0.895<br>0.981<br>0.916<br>**0.995**<br>0.932<br>0.866<br>0.971|0.677<br>0.648<br>0.502<br>0.548<br>0.669<br>0.646<br>0.611<br>0.549<br>0.853<br>0.771<br>**0.890**<br>0.800<br>0.883<br>**0.837**|



Table 10: **Detailed nuPlan closed-loop planning results (PDMScore) on Test14-Random benchmark.** 

|**mark.**|||||
|---|---|---|---|---|
|Type|Mhd|_Test14 Random_|||
||eto|Col. Avoid_↑_<br>Drivable_↑_<br>Direction_↑_<br>Progress_↑_<br>TTC_↑_<br>Comfort_↑_<br>**PDMScore**_↑_|||
|_Expert_|_Log Replay_||0.996|0.962<br>0.996<br>0.664<br>0.985<br>1.000<br>0.832|
|Rule|PDM-Closed [69]||0.934|0.984<br>0.996<br>0.867<br>0.911<br>0.996<br>0.888|
|Learning|Constant Acc.||0.846|0.907<br>0.915<br>0.436<br>0.804<br>1.000<br>0.592|
||UrbanDriver [74]||0.965|0.961<br>0.986<br>0.611<br>0.957<br>1.000<br>0.788|
||PlanCNN [72]||0.935|0.938<br>0.971<br>0.591<br>0.888<br>0.989<br>0.736|
||PlanTF [73]||0.966|0.948<br>0.625<br>0.626<br>0.918<br>0.992<br>0.768|
||**BeTopNet (Ours)**||**0.989**|**0.977**<br>**0.989**<br>**0.673**<br>**0.969**<br>**1.000**<br>**0.833**|



Table 11: **Detailed nuPlan closed-loop planning results (PDMScore) on Test14-Hard benchmark.** 

|Table 11:|**Detailed nuPlan**|**closed-loop planning results (PDMScore) on Test14-Hard benchmark**|**closed-loop planning results (PDMScore) on Test14-Hard benchmark**|
|---|---|---|---|
|Type|Mhd|_Test14 Hard_||
||eto|Col. Avoid_↑_<br>Drivable_↑_<br>Direction_↑_<br>Progress_↑_<br>TTC_↑_<br>Comfort_↑_<br>**PDMScore**_↑_||
|_Expert_|_Log Replay_|0.985<br>0.945<br>0.970|0.658<br>0.955<br>1.000<br>0.786|
|Rule|PDM-Closed [69]|0.933<br>0.952<br>0.976|0.779<br>0.852<br>0.981<br>0.811|
|Learning|Constant Acc.|0.845<br>0.871<br>0.861|0.415<br>0.800<br>1.000<br>0.552|
||UrbanDriver [74]|0.946<br>0.944<br>0.992|0.581<br>0.903<br>**1.000**<br>0.731|
||PlanCNN [72]|0.909<br>0.908<br>0.937|0.555<br>0.860<br>0.992<br>0.675|
||PlanTF [73]|**0.984**<br>**0.961**<br>**0.996**|0.649<br>**0.961**<br>0.996<br>0.813|
||**BeTopNet (Ours)**|0.968<br>0.945<br>0.972|**0.747**<br>0.908<br>0.996<br>**0.813**|



## **D.3 Training Setup** 

BeTopNet for both prediction and planning tasks are trained in end-to-end manners by AdamW optimizer with 4 NVIDIA A100 GPUs. The learning rate is configured as 1 _e[−]_[4] scheduled with the multi-step reduction strategy. The planning model is trained by 25 epochs with a batch size of 128, while the prediction task is trained with 30 epochs with a batch of 256. 

## **E Additional Quantitative Results** 

## **E.1 Planning** 

**Additional planning results in Val14.** We evaluate the closed-loop simulation performance under _Val14_ in Table 9, BeTopNet hovers strong planning results and is comparable (+4 _._ 6% CLS) to concurrent learning-based methods [92] leveraging extra contrasting learning for training augmentations. BeTopNet is also featured by leading driving safety (+2 _._ 7% CA, +1 _._ 0% TTC) and compliance (+2 _._ 8% DDC) compared with other strong models [73, 8]. However, due to the data leakage of _Val14_ with training set by a part of shared scenarios, we only provide the results as a reference. 

**Additional planning effects in Test14.** To delve into the planning results of BeTopNet, we present statistics measuring by another detailed metric, PDMScore, for both of the _Test14_ benchmarks in Table 2. Exhibited in Tables 10 and 11, BeTopNet delivers strong maneuver safety and compliance, marking solid PDMScore from both benchmarks. Compared with learning-based methods, BeTopNet 

21 

Table 12: **Marginal predictions on WOMD Motion Leaderboard [81].** Primary metric. 

|Category|Method|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_||mAP_↑_<br>**Soft mAP**_↑_|mAP_↑_<br>**Soft mAP**_↑_|
|---|---|---|---|---|---|---|---|---|
|Vehicle|MTR [22]||0.7642|1.5257|0.1514||0.4494|0.4590|
||EDA [53]||**0.6808**|1.3921|**0.1164**||0.4833|0.4972|
||**BeTopNet (Ours)**||0.6814|**1.3888**|0.1172||**0.4860**|**0.4995**|
|Pedestrian<br>MTR [22]<br>EDA [53]<br>**BeTopNet (Ours)**|||0.3486|0.7270|0.0753||0.4331|0.4409|
||EDA [53]||**0.3426**|**0.7080**|0.0670||0.4680|0.4778|
||**BeTopNet (Ours)**||0.3451|0.7142|**0.0668**||**0.4777**|**0.4875**|
|Cyclist<br>MTR [22]<br>EDA [53]<br>**BeTopNet (Ours)**|||0.7022|1.4093|0.1786||0.3561|0.3650|
||EDA [53]||0.6920|1.4106|**0.1673**||0.3947|0.4037|
||**BeTopNet (Ours)**||**0.6905**|**1.3975**|0.1688||**0.4060**|**0.4163**|



Table 13: **Joint predictions on WOMD Interaction Leaderboard [82].** Primary metric. 

|Category|Method|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_||**mAP**_↑_<br>Soft mAP_↑_|**mAP**_↑_<br>Soft mAP_↑_|
|---|---|---|---|---|---|---|---|---|
|Vehicle|GameFormer [8]||1.0499|2.4044|0.4321||0.2469<br>0.3104<br>**0.3374**|0.2564<br>0.3196<br>**0.3308**|
||AMP [32]||**0.9862**|**2.2286**|**0.3726**||||
||**BeTopNet (Ours)**||1.0216|2.3970|0.3738||||
|Pedestrian<br>GameFormer [8]<br>AMP [32]<br>**BeTopNet (Ours)**|||0.7978|1.8195|0.4713||0.1962<br>**0.2359**<br>0.2212|0.2014<br>**0.2423**<br>0.2267|
||AMP [32]||**0.6823**|**1.5244**|**0.3716**||||
||**BeTopNet (Ours)**||0.7862|1.8412|0.4074||||
|Cyclist<br>GameFormer [8]<br>AMP [32]<br>**BeTopNet (Ours)**|||1.0686|2.4199|0.5765||0.1367<br>0.1420<br>**0.1717**|0.1338<br>0.1477<br>**0.1756**|
||AMP [32]||**1.0533**|**2.3715**|**0.5194**||||
||**BeTopNet (Ours)**||1.1155|2.5850|0.5253||||



excels in closed-loop driving progress (+15 _._ 1% _,_ +7 _._ 5% EP), safety (+5 _._ 6% TTC, +2 _._ 9% CA), and the general score (+8 _._ 5% PDMScore). For rule-based systems, the leading performance is empirically by virtue of a constant driving progress. This may refer to an unresolved _copy-cat_ problem [73] for imitative planners. It requires further integration and fallback with rule-based methods for on-board AD system design in practice. 

## **E.2 Prediction** 

**Per-category marginal prediction.** In Table 12, We mainfest the prediction performance of BeTopNet under each prediction category. Compared against the concurrent SOTA motion predictors [53], BeTopNet demonstrates superior mAP-based metrics among all types for compliant predictions. Specifically, overall improvements in Cyclist denote refined interactive patterns captured by BeTop, as the cyclist predictions are the most uncertain task with less reliance on map information. 

**Per-category joint prediction.** We further instantiate the per-category joint prediction of BeTopNet with SOTA methods in Table 13. Compared with concurrent methods [32] featuring auto-regressive decoding, BeTopNet achieves robust displacement metrics, while outperforming in prediction compliance of mAP metrics (+8 _._ 7% _,_ +20 _._ 9% mAP) due to advanced joint modality scoring stabilized by edge topology in BeTopNet. Moreover, the coordinated joint behaviors reasoned by BeTopNet largely mitigate the unstable patterns against game-theoretic method [8] ( _−_ 15 _._ 6% _, −_ 15 _._ 7% _, −_ 9 _._ 7% Miss Rate) under similar model architecture. 

## **F Additional Ablation Studies** 

**Scaling effects of model and decoding agents.** The scalability challenges begin with the scaling of our BeTopNet models to accommodate varying scene agents and map. Experimentally, we configure BeTopNet with different model scales to evaluate whether our approach maintains its effectiveness. 

In Table 14, BeTopNet is evaluated by three model scales varying in decoding modalities and dimensions. The results demonstrate that BeTopNet consistently improves prediction accuracy, with an increase from 0.391 to 0.442 (+13 _._ 4% mAP) and a decrease in the Miss Rate ( _−_ 11 _._ 9%). This showcases its enhanced robustness in handling multi-agent settings by enlarging model scales. 

In Table 15, BeTopNet reports comparable computational costs compared to [22], while with better prediction accuracy shown in Table 12. The similar latency is due to the topo-guided attention, 

22 

Table 14: **Effects of varied model scale.** BeTopNet shows scalability with the number of decoding modalities and feature dimensions. 

Table 15: **Effects of varied decoding agents.** Computational costs of [22] are reported in the parenthesis after ours. 

|Scale|**mAP**_↑_<br>Miss Rate_↓_|Latency (ms)<br># Params. (M)||# Decoding Agents|Latency (ms)<br>GPU Memory (G)|
|---|---|---|---|---|---|
|Small<br>Medium<br>Base|0.391<br>0.131<br>0.437<br>0.119<br>**0.442**<br>**0.117**|45<br>28.91<br>65<br>28.91<br>70<br>45.38||8<br>16<br>32|89(84)<br>6.5(5.2)<br>120(123)<br>10.8(7.1)<br>166(193)<br>19.2(15.6)|



Table 16: **Effects of varied temporal granularity in BeTop.** Future interactions are split into various intervals for multi-step BeTop labels. A fine-grained topology reasoning for the whole prediction horizon results in a slightly improved performance and increased computational costs simultaneously. 

|Interval|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_<br>**mAP**_↑_|Inference<br>Latency (ms)<br>Training<br>Latency (ms)<br># Params.<br>(M)|
|---|---|---|
|1(Base)<br>**2**<br>4<br>8|0.637<br>1.328<br>0.144<br>0.392|70.0<br>101.6<br>45.380<br>75.5<br>110.6<br>45.382<br>80.0<br>133.4<br>45.386<br>90.0<br>255.0<br>45.393|
||**0.633**<br>**1.325**<br>0.145<br>**0.394**||
||0.634<br>1.326<br>**0.142**<br>0.391||
||0.641<br>1.347<br>0.147<br>0.389||



Table 17: **Effects of varied model foundation by Wayformer [62].** Synergistic decoder design by BeTopNet demonstrate solid multi-agent interaction understanding compared with vanilla design. 

|Method|minADE_↓_<br>minFDE_↓_<br>Miss Rate_↓_<br>**mAP**_↑_|
|---|---|
|Wayformer<br>Wayformer+BeTop<br>Wayformer+BeTopNet|0.661<br>1.417<br>0.199<br>0.281|
||0.637<br>1.364<br>0.178<br>0.290|
||**0.604**<br>**1.261**<br>**0.166**<br>**0.344**|



which reduces the KV features in agent aggregation during decoding. While BeTop introduces extra computations for reasoning, it requires more GPU memory for cached topology tensors. 

**Temporal granularity in BeTop.** Table 16 explores the effect of varied temporal granularity in BeTop, with minimal adjustments to BeTopNet. In our study, future interactions are split into multistep BeTop labels. Topology reasoning task is then deployed through expanded MLP Topo Head for output steps. Compared to the baseline 1-step reasoning, multi-step BeTop reasoning slightly improves performance ( _e.g._ , 2-steps, +0.2 mAP), with a corresponding increase in computational costs for additional steps. This highlights the potential of multi-step reasoning to enhance BeTopNet in interactive scenarios, while refining temporal granularity for more accurate and efficient interactions remains an open question. We believe how to effectively leverage multi-step BeTopNet represents an interesting area for future exploration. 

**Synergy with additional model foundation.** To understand the generalization under different model foundations, we conduct additional ablations integrating BeTop with reproduced Wayformer [62] in [95]. As reported in Table 17, incorporating BeTop as supervision improves vanilla Wayformer with a _−_ 6 _._ 2% Miss Rate and +3 _._ 2% mAP. Furthermore, integrating BeTopNet significantly boosts performance, achieving a +18 _._ 6% mAP and _−_ 7 _._ 2% Miss Rate. This enhancement is due to synergistic decoder design, which uses iterative BeTop reasoning and Topo-guided attention to refine trajectories by selectively aggregating interactive features. 

## **G Additional Qualitative Results** 

**Additional planning results.** We provide the qualitative closed-loop simulations for all of the benchmarks in _Test14_ , as shown in Figs. 7 and 8. 

**Additional prediction results.** We provide the qualitative prediction results for BeTop with reasoned edge topology under both marginal (Fig. 10) and joint (Fig. 9) challenge settings. 

## **H License of Assets** 

Data for nuPlan [80] and WOMD [35] are complied with CC-BY-NC 4.0 licence and Apache License 2.0; The code for re-implementations are under Apache License 2.0 for PDM [69] and MTR [22], and 

23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0024-00.png)


**----- Start of picture text -----**<br>
Test14-Hard Test14-Random<br>**----- End of picture text -----**<br>


Figure 7: **Qualitative results of BeTopNet in nuPlan planning under Test14 simulations.** Each row of the figures render closed-loop simulations at 1s, 8s, and 15s temporal frames. As illustrated, BeTopNet performs consistent planning under challenging driving scenarios of diverse categories. 

MIT License for EDA [53] and PlanTF [73], respectively. The source code and our trained models will be publicly available under the Apache License 2.0. 

24 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0025-00.png)


**----- Start of picture text -----**<br>
a) b)<br>c) d)<br>p=0 p=1 t=0s t=8s t=0s t=8s<br>Joint Probability Prediction Planning<br>**----- End of picture text -----**<br>


Figure 8: **Qualitative results of BeTopNet in nuPlan planning under Test14-Inter.** BeTopNet performs compliant planning under: (a) yielding to front agents; (b) cruising on various road structure; (c-d) interactive behaviors among two or more agents with dense traffic. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0025-02.png)


**----- Start of picture text -----**<br>
Vehicle to vehicle (V2V) Vehicle to cyclist (V2C) Vehicle to pedestrian (V2P)<br>p=0 p=1 t=0s t=8s e=0 e=1<br>Joint Probability Prediction Edge Topology<br>**----- End of picture text -----**<br>


Figure 9: **Qualitative results of BeTopNet in WOMD joint prediction.** Joint predictions among heterogeneous agents are categorized by each column (V2V, V2C, and V2P) with corresponding TopK reasoned topology. As depicted, BeTopNet can accurately capture the future interactive behaviors via edge topology reasoning compared with the human annotations of interactive agents (rendered in red). Moreover, BeTopNet may source on potential interactions as rendered in grey. 

25 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving_images/nips_2024_Reasoning_Multi-Agent_Behavioral_Topology_for_Interactive_Autonomous_Driving.pdf-0026-00.png)


**----- Start of picture text -----**<br>
p=0 p=1 t=0s t=8s e=0 e=1<br>Marginal Probability Prediction Edge Topology<br>**----- End of picture text -----**<br>


Figure 10: **Qualitative results of BeTopNet in WOMD marginal prediction.** BeTopNet performs compliant and accurate marginal predictions on multiple agents, reasoning diverse edge topology which stabilizes the behavioral patterns for future interactions. 

26 

