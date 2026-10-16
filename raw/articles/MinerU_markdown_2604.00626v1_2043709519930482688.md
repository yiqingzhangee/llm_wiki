# A Survey of On-Policy Distillation for Large Language Models

Mingyang Song & Mao Zheng 

Large Language Model Department 

Tencent, China 

nickmysong@tencent.com 

# Abstract

Knowledge distillation has become a primary mechanism for transferring reasoning and domain expertise from frontier Large Language Models (LLMs) to smaller, deployable students. However, the dominant paradigm remains off-policy: students train on static teacher-generated data and never encounter their own errors during learning. This train–test mismatch, an instance of exposure bias, causes prediction errors to compound autoregressively at inference time. On-Policy Distillation (OPD) addresses this by letting the student generate its own trajectories and receive teacher feedback on these self-generated outputs, grounding distillation in the theory of interactive imitation learning. Despite rapid growth spanning divergence minimization, reward-guided learning, and self-play, the OPD literature remains fragmented with no unified treatment. This survey provides the first comprehensive overview of OPD for LLMs. We introduce a unified $f$ - divergence framework over on-policy samples and organize the landscape along three orthogonal dimensions: feedback signal (logit-based, outcomebased, or self-play), teacher access (white-box, black-box, or teacher-free), and loss granularity (token-level, sequence-level, or hybrid). We systematically analyze representative methods, examine industrial deployments, and identify open problems including distillation scaling laws, uncertaintyaware feedback, and agent-level distillation. 

# 1 Introduction

Large Language Models (LLMs) have fundamentally reshaped natural language processing and, increasingly, artificial intelligence as a whole. Scaling from hundreds of millions of parameters to hundreds of billions, models such as GPT-4, PaLM, and Llama have demonstrated remarkable capabilities in reasoning, code generation, multilingual understanding, and instruction following. Yet the sheer computational cost of training and serving these frontier models remains prohibitive for most deployment scenarios. A single inference call to a 540-billion-parameter model can consume orders of magnitude more energy and latency than a well-trained 7-billion-parameter alternative, making the transfer of capabilities from large teachers to compact students not merely desirable but economically necessary. Knowledge distillation (KD), first formalized by Hinton et al. (2015) as training a student network to match the softened output distribution of a teacher, has accordingly grown from a niche compression technique into a central pillar of the modern LLM development pipeline. The release of DeepSeek-R1 (DeepSeek-AI et al., 2025), which successfully transferred complex chain-of-thought reasoning from a 671-billion-parameter mixture-of-experts teacher into dense student models ranging from 1.5 billion to 70 billion parameters, illustrates how distillation now serves as a general-purpose capability transfer engine rather than a mere size-reduction tool. 

As the role of distillation has expanded, so has awareness of a fundamental bottleneck in its dominant paradigm. Conventional LLM distillation is off-policy in nature. The student is trained on a fixed corpus, typically sampled from the data distribution $p _ { \mathrm { d a t a } }$ or generated in advance by the teacher, and learns to replicate the teacher’s token-level probabilities over these pre-collected sequences. At inference time, however, the student must gener-

ate autoregressively, conditioning each new token on its own previous (and potentially erroneous) outputs. This discrepancy between the training distribution and the student’s own generation distribution creates a train-test mismatch that compounds over the length of the sequence. Gudibande et al. (2023) provided a systematic empirical demonstration of this failure mode, showing that off-policy distilled students can degrade sharply on tasks requiring sustained multi-step generation. Formally, this phenomenon is an instance of exposure bias well-studied in the imitation learning literature (Ross et al., 2011), where a policy trained only on expert-state demonstrations drifts into unfamiliar states at test time and lacks the supervisory signal to recover. For autoregressive LLMs, the problem is particularly severe because errors at early positions propagate through the entire remaining sequence, and the student receives no gradient information about how to behave in these self-induced off-distribution states. 

On-Policy Distillation (OPD) offers a principled solution. Rather than training on a static dataset, OPD lets the student sample sequences from its own evolving policy $p _ { \theta }$ and then solicits teacher feedback on these self-generated trajectories. This feedback can take diverse forms, ranging from full token-level probability distributions in white-box settings (Agarwal et al., 2024; Gu et al., 2024) to scalar rewards, adversarial feedback, or pairwise preferences when only black-box teacher access is available (Ye et al., 2025; Jiang et al., 2023), and even self-generated contrastive signals in teacher-free frameworks (Chen et al., 2024). The theoretical motivation draws directly from interactive imitation learning. The DAgger algorithm (Ross et al., 2011) established that querying an expert on the learner’s own visited states reduces the compounding error from $O ( \dot { T } ^ { 2 } )$ under pure behavior cloning to $O ( T )$ where $T$ is the horizon length. OPD instantiates this principle for autoregressive language generation, transforming distillation from one-shot distribution matching into an iterative, self-correcting optimization loop. In recent years, this paradigm has catalyzed a wave of methodological innovation. Generalized Knowledge Distillation (GKD) (Agarwal et al., 2024) introduced on-policy sampling with configurable mixture ratios between student and teacher sequences. MiniLLM (Gu et al., 2024) recast the objective by minimizing Reverse KL divergence to avoid the mode-covering pathology of Forward KL. DistiLLM (Ko et al., 2024) proposed skewed KL objectives that stabilize optimization by mixing student and teacher distributions. RLKD (Xu et al., 2025b) introduced reinforcement learning (RL) with a structure-aware reward model to capture reasoning patterns that distributional matching alone cannot transfer. Self-Play Fine-Tuning (SPIN) (Chen et al., 2024) demonstrated that a student can improve by distinguishing its own generations from human references without any teacher model at all. 

Despite this rapid growth, the OPD literature remains fragmented. Methods originating from the knowledge distillation community, the reinforcement learning from human feedback (RLHF) community, and the imitation learning community often address the same underlying problem with different formalisms, different evaluation protocols, and different terminology. Existing surveys of LLM distillation (Xu et al., 2024) predominantly organize the field around the classical compression framing, treating off-policy and on-policy approaches as interchangeable variants rather than as fundamentally distinct paradigms with different theoretical guarantees and failure modes. To date, no survey has provided a unified mathematical treatment of the on-policy dynamics that connect these seemingly disparate lines of work, nor has any systematically compared the trade-offs among white-box, black-box, and teacher-free instantiations of on-policy learning for LLMs. 

This survey fills that gap. We provide the first dedicated and comprehensive treatment of onpolicy distillation for Large Language Models, organized around a theoretically motivated taxonomy. Our contributions are as follows. 

• A Unified Theoretical Framework. We formulate the transition from off-policy to on-policy distillation through the lens of sequential decision-making and $\bar { f }$ divergence minimization over student-sampled trajectories. This framework reveals how GKD (Agarwal et al., 2024), MiniLLM (Gu et al., 2024), and DistiLLM (Ko et al., 2024) instantiate the same underlying objective under different divergence choices, argument orderings, target distributions, and sampling mixture coefficients, pro-

viding a common analytical language for comparing methods that were previously studied in isolation. 

• A Three-Dimensional Taxonomy. We organize OPD methods along three orthogonal axes, namely feedback signal (logit-based distributional matching, outcome-based scalar or preference signals, or self-play), teacher access (white-box logit access, blackbox generation-only access, or fully teacher-free self-distillation), and loss granularity (token-level, sequence-level, or hybrid/adaptive). This taxonomy clarifies the design space and reveals underexplored combinations that represent promising research directions. 

• Bridging White-Box and Black-Box Regimes. We provide a systematic comparison of on-policy methods that require full teacher internals versus those that operate with only sampled outputs, analyzing the theoretical information gap between learning from complete probability distributions and learning from top- $\mathbf { \nabla } \cdot k$ generations through representative methods such as GAD (Ye et al., 2025) and Lion (Jiang et al., 2023). 

• Identification of Overlooked Challenges. We highlight several issues that have received insufficient attention, including the compute-quality trade-off in online generation, the “echo chamber” effect when querying teachers on low-confidence student states, the need for dynamic divergence adaptation across training stages, and the reconceptualization of distillation as structured capability transfer rather than parameter-level compression. 

• A Roadmap for Future Research. We chart open problems including distillation scaling laws, uncertainty-aware OPD, curriculum-driven sampling strategies, and agent-level distillation where models must learn from multi-turn, state-dependent interactions with external environments. 

The remainder of this survey is organized as follows. Section 2 establishes mathematical preliminaries, formalizing autoregressive generation as a sequential decision problem and deriving the exposure bias that motivates on-policy approaches. Section 3 presents our three-dimensional taxonomy in detail. Section 4 provides an in-depth analysis of white-box OPD methods, and Section 5 covers black-box and self-distillation approaches. Section 6 examines reasoning-specific distillation pipelines. Section 7 discusses industrial deployment considerations and practical guidelines, and Section 8 identifies open problems and future directions. 

# 2 Background and Preliminaries

To bridge the gap from standard, dataset-bound knowledge distillation to dynamic onpolicy distillation, we first formalize the sequential nature of LLM generation. This section defines distribution matching, formalizes exposure bias, and builds a framework using the $f$ -divergence family. 

Notation. We use lowercase $p$ for token-level conditional distributions (e.g., $p _ { T } ( \cdot | x , y _ { < t } )$ , $p _ { \theta } ( \cdot | x , y _ { < t } ) )$ and uppercase $P$ for sequence-level or trajectory-level distributions (e.g., $P _ { \cal T } ( y \vert x ) , ~ P _ { \theta } ( y \vert x ) )$ . In the MDP formalization of exposure bias (Section 2.3), we use $\pi$ for policies. When discussing individual methods in detail, we follow each paper’s notation where helpful (e.g., ${ { p } _ { S } } / { { p } _ { T } }$ for student/teacher in the divergence analysis of Section 4.1, $P _ { \theta } / P _ { \mathrm { T e a c h e r } }$ in the reasoning distillation of Section 6). 

# 2.1 Knowledge Distillation Fundamentals

The classical knowledge distillation framework, pioneered by Hinton et al. (2015), transfers the implicit ”dark knowledge” embedded in the continuous probability distributions of a cumbersome teacher network $\tau$ to a compact student network $s$ . In standard classification tasks, and by extension token-level vocabulary selection in language modeling, a neural network produces a vector of pre-softmax logits $z \in \mathbb { R } ^ { | V | }$ , where $| V |$ is the vocabulary size. 

To extract the relational structure between classes (for instance, that the token ”automobile” is semantically closer to ${ } ^ { \prime \prime } \mathrm { c a r } ^ { \prime \prime }$ than to ”apple”), the logits are smoothed using a temperature scalar $\tau > 0$ . The softened probability for a given input context $x$ is 

$$
p (y | x; \tau) = \frac {\exp \left(z _ {y} / \tau\right)}{\sum_ {y ^ {\prime} \in V} \exp \left(z _ {y ^ {\prime}} / \tau\right)} \tag {1}
$$

The standard KD objective seeks to minimize the Kullback-Leibler (KL) divergence between the teacher’s softened predictive distribution $p _ { T }$ and the student’s corresponding softened distribution $p _ { \theta }$ 

$$
\mathcal {L} _ {\mathrm {K D}} = \tau^ {2} D _ {\mathrm {K L}} \left(p _ {T} (\cdot | x; \tau) \| p _ {\theta} (\cdot | x; \tau)\right) = \tau^ {2} \sum_ {y \in V} p _ {T} (y | x; \tau) \log \left(\frac {p _ {T} (y | x ; \tau)}{p _ {\theta} (y | x ; \tau)}\right) \tag {2}
$$

The $\tau ^ { 2 }$ scaling factor is important because as $\tau$ increases, the cross-entropy gradient magnitudes scale down by $1 / \tau ^ { 2 }$ , so multiplying by $\tau ^ { 2 }$ keeps the distillation loss commensurate with hard-label cross-entropy in a joint objective. 

To understand this transfer, we analyze the gradient of the distillation loss with respect to the student’s logits $z _ { i } ^ { S }$ . Let $p _ { i } ^ { \mathcal { T } }$ and $p _ { i } ^ { \dot { S } }$ denote the temperature-scaled teacher and student probabilities, respectively. The gradient is 

$$
\frac {\partial \mathcal {L} _ {\mathrm {K D}}}{\partial z _ {i} ^ {\mathcal {S}}} = \frac {\tau^ {2}}{\tau} \left(p _ {i} ^ {\mathcal {S}} - p _ {i} ^ {\mathcal {T}}\right) = \tau \left(p _ {i} ^ {\mathcal {S}} - p _ {i} ^ {\mathcal {T}}\right) \tag {3}
$$

When $\tau \to \infty$ , a Taylor expansion $\exp ( z _ { i } / \tau ) \approx 1 + z _ { i } / \tau$ shows that, assuming zero-meaned logits $( \Sigma _ { j } z _ { j } = 0 )$ ), the probabilities approach 

$$
p _ {i} \approx \frac {1 + z _ {i} / \tau}{| V | + \sum_ {j} z _ {j} / \tau} = \frac {1}{| V |} + \frac {z _ {i}}{| V | \tau} \tag {4}
$$

Substituting back into the gradient reveals an equivalence 

$$
\frac {\partial \mathcal {L} _ {\mathrm {K D}}}{\partial z _ {i} ^ {S}} \approx \tau \left(\frac {z _ {i} ^ {S}}{| V | \tau} - \frac {z _ {i} ^ {\mathcal {T}}}{| V | \tau}\right) = \frac {1}{| V |} \left(z _ {i} ^ {S} - z _ {i} ^ {\mathcal {T}}\right) \tag {5}
$$

This derivation shows that in the high-temperature limit, classical KD degenerates into minimizing the Mean Squared Error (MSE) between raw logits. This forces the student to replicate the teacher’s entire logit structure, transferring both primary predictive modes and nuanced sub-optimal probability masses (the dark knowledge) that underpin generalization. However, when applied naively to autoregressive LLMs, this formulation assumes the input $x$ (the prefix) is drawn from the target data distribution, entirely ignoring the sequential, compounding nature of language generation. 

# 2.2 Token-Level vs. Sequence-Level Distillation

Applying Hinton et al. (2015)’s formulation to language modeling typically yields tokenlevel distillation. In an autoregressive setting, a sequence $y = ( y _ { 1 } , \bar { y } _ { 2 } , \dotsc , \bar { y _ { T } } )$ is generated conditionally. Let $y _ { < t }$ denote the prefix up to step t. The token-level distillation loss is the expected divergence over prefixes from the empirical dataset $\mathcal { D }$ . 

$$
\mathcal {L} _ {\text {T o k e n - K D}} = \mathbb {E} _ {x, y \sim \mathcal {D}} \left[ \sum_ {t = 1} ^ {| y |} D _ {\mathrm {K L}} \left(p _ {T} (\cdot | x, y _ {<   t}) \| p _ {\theta} (\cdot | x, y _ {<   t})\right) \right] \tag {6}
$$

While computationally tractable because it relies on static, pre-computed prefixes, tokenlevel KD misaligns with the global generative objective because it optimizes next-token accuracy while assuming an error-free history $y _ { < t }$ . 

Recognizing this limitation, Kim & Rush (2016) proposed Sequence-Level Knowledge Distillation. Instead of factorizing the loss token-by-token over fixed dataset prefixes, 

sequence-level KD aims to match the joint probability distributions over the entire sequence space $\mathcal { V }$ . The theoretical sequence-level objective minimizes the global KL divergence 

$$
\mathcal {L} _ {\text {S e q - K D}} = D _ {\mathrm {K L}} \left(P _ {\mathcal {T}} (y | x) \| P _ {\theta} (y | x)\right) = \sum_ {y \in \mathcal {Y}} P _ {\mathcal {T}} (y | x) \log \left(\frac {P _ {\mathcal {T}} (y | x)}{P _ {\theta} (y | x)}\right) \tag {7}
$$

Because the sequence space $\mathcal { V }$ grows exponentially with length T $( | \mathcal { V } | = | V | ^ { T } )$ , exactly computing this summation is intractable. Kim & Rush (2016) resolved this pragmatically. Observing that $P \tau ( y | x )$ is sharply peaked around its mode, they approximated the distribution with a Dirac delta at the beam-search output 

$$
P _ {\mathcal {T}} (y | x) \approx \left\{ \begin{array}{l l} 1 & \text {i f} y = \arg \max  _ {y ^ {\prime} \in \mathcal {Y}} P _ {\mathcal {T}} \left(y ^ {\prime} \mid x\right) \\ 0 & \text {o t h e r w i s e} \end{array} \right. \tag {8}
$$

Substituting this approximation into the sequence-level KL divergence simplifies the loss to the standard negative log-likelihood (NLL) of the student, evaluated on the teacher’s highest-probability sequence $\hat { y }$ 

$$
\mathcal {L} _ {\text {S e q - A p p r o x}} = - \sum_ {t = 1} ^ {| \hat {y} |} \log p _ {\theta} \left(\hat {y} _ {t} \mid x, \hat {y} _ {<   t}\right) \tag {9}
$$

This formulation forces the student to model complete trajectories favored by the teacher, implicitly transferring sequence-level structural dependencies. However, Sequence-Level KD remains an off-policy algorithm. The student still trains via teacher-forcing on static trajectories (pseudo-labels generated offline by the teacher). While the target $\hat { y }$ originates from a model rather than a human, the student never conditions predictions on its own prefixes during training. Consequently, Sequence-Level KD remains susceptible to the train-test mismatch it sought to alleviate, motivating true on-policy algorithms. 

# 2.3 The Distribution Mismatch Problem and Exposure Bias

The core problem necessitating on-policy distillation is exposure bias, a direct manifestation of covariate shift in sequential decision-making. We cast autoregressive generation as a Markov Decision Process (MDP). The state space $\scriptstyle { \dot { \boldsymbol { S } } }$ comprises all token prefixes $s _ { t } = ( x , y _ { < t } )$ the action space $\mathcal { A }$ is the vocabulary $V$ , and transitions are deterministic: action $y _ { t }$ in state $s _ { t }$ yields $s _ { t + 1 } \dot { = } ( x , y _ { < t } , y _ { t } )$ . 

During off-policy distillation (teacher-forcing), the student encounters states governed by the data distribution $p _ { \mathrm { d a t a } }$ . Defining the state visitation distribution under the dataset policy as $d _ { \mathcal { D } } ( s )$ , the training objective is 

$$
\mathcal {L} _ {\text {t r a i n}} = \mathbb {E} _ {s \sim d _ {\mathcal {D}}} \left[ D _ {\mathrm {K L}} \left(\pi^ {*} (\cdot | s) \parallel \pi_ {\theta} (\cdot | s)\right) \right] \tag {10}
$$

At inference, however, the student acts according to its own learned policy $\pi _ { \theta }$ . Because the student is imperfect, its actions deviate from the dataset distribution. Let $d _ { \pi _ { \theta } } ( s )$ denote the state visitation distribution induced by the student’s own rollouts. The true generation quality is evaluated under this distinct distribution 

$$
\mathcal {L} _ {\text {t e s t}} = \mathbb {E} _ {s \sim d _ {\pi_ {\theta}}} \left[ \mathcal {L} _ {\text {t a s k}} (s, \pi_ {\theta}) \right] \tag {11}
$$

Exposure bias arises from the inequality $d _ { \mathcal { D } } ( s ) \neq d _ { \pi _ { \theta } } ( s )$ . Because the student never encounters states in $d _ { \pi _ { \theta } } ( s ) \setminus d _ { \mathcal { D } } ( s )$ during training, it receives zero supervisory signal in these regions. 

We quantify this mismatch using performance bounds from imitation learning. As Ross et al. (2011) showed via the DAgger analysis, if a student policy $\pi _ { \theta }$ mimics a teacher policy $\pi ^ { * }$ with per-step error bounded by $\epsilon$ under the training distribution, i.e., $\mathbb { E } _ { s \sim d _ { \pi ^ { * } } } [ \mathbb { I } ( \pi _ { \theta } ^ { \mathbf { \hat { \mu } } } ( s ) \neq$ $\pi ^ { * } ( s ) ) ] \leq \epsilon ,$ then the expected total discrepancy over a trajectory of length $T$ under the student’s own distribution scales quadratically 

$$
\mathbb {E} _ {s \sim d _ {\pi_ {\theta}}} \left[ \sum_ {t = 1} ^ {T} \mathcal {L} (s _ {t}) \right] \leq O (\epsilon T ^ {2}) \tag {12}
$$

This $O ( T ^ { 2 } )$ compounding bound is significant for modern LLMs, which routinely generate sequences spanning thousands of tokens. A single suboptimal token shifts the prefix slightly out-of-distribution, and the model, having never seen this perturbed prefix, is more likely to err again, leading to degraded or hallucinatory text. On-policy distillation resolves this by changing the training expectation from $\mathbb { E } _ { s \sim d _ { \mathcal { D } } }$ to $\mathbb { E } _ { s \sim d _ { \pi _ { \theta } } }$ . By generating responses online, the student confronts its own mistakes, receives teacher feedback on those specific out-ofdistribution states, and learns robust recovery behaviors, reducing error accumulation from $O ( \epsilon T ^ { 2 } )$ to $O ( \epsilon T )$ . 

Remark: The Nuance of the DAgger Bound in LLMs. While the theoretical transition from $O ( \epsilon T ^ { 2 } )$ to $O ( \epsilon T )$ is appealing, applying the DAgger bound to LLMs requires nuance. The original DAgger theorem assumes an interactive expert providing optimal actions in any state. In white-box OPD, the “expert” provides a next-token distribution $p _ { T } \big ( y _ { t } \big | \hat { y } _ { < t } \big )$ conditioned on the student’s prefix $\hat { y } _ { < t }$ . If the student hallucinates a severely out-of-distribution prefix, the teacher’s conditional distribution may itself become poorly calibrated. Forcing the student to match this noisy distribution violates the core assumption of interactive imitation learning, potentially destabilizing training rather than recovering the $O ( \epsilon T )$ bound. This underscores why adaptive divergence methods (Section 4.1) are critical for selectively trusting the teacher. 

This imitation learning perspective (train on student rollouts, supervise with teacher feedback) directly inspired the modern LLM distillation methods. GKD (Agarwal et al., 2024) and DistiLLM (Ko et al., 2024) generalized this DAgger-style pattern to the large language model setting, replacing neural machine translation-scale architectures with billion-parameter models and introducing divergence-aware training objectives. 

# 2.4 The $f$ -Divergence Family and Geometric Intuition

To parameterize distribution matching in OPD, we rely on the $f$ -divergence framework. Given two probability distributions $\vec { P }$ and $Q$ over a space $\mathcal { X } ,$ , an $f$ -divergence measures their discrepancy via a convex generator $f : ( 0 , \infty ) \to { \hat { \mathbb { R } } }$ with $f ( 1 ) = 0$ 

$$
D _ {f} (P \parallel Q) = \int_ {\mathcal {X}} Q (x) f \left(\frac {P (x)}{Q (x)}\right) d x = \mathbb {E} _ {x \sim Q} \left[ f \left(\frac {P (x)}{Q (x)}\right) \right] \tag {13}
$$

The choice of $f ( u )$ yields different properties for the distillation objective. In LLM distillation, we identify $P \equiv p _ { T }$ (teacher) and $\bar { Q } \equiv p _ { \theta }$ (student), where the token-level notation matches our convention from Section 2.1. 

1. Forward Kullback-Leibler Divergence. Selecting $f ( u ) = u \log u$ recovers Forward KL, $D _ { \mathrm { K L } } ( P ~ \parallel ~ Q ) = \mathbb { E } _ { x \sim P } [ \log ( P ( x ) / \check { Q } ( x ) ) ]$ . The gradient forces the student Q to place mass everywhere the teacher $P$ has mass. If $P ( x ) \stackrel { \sim } { > } 0$ but $Q ( x ) \approx 0 .$ , the penalty diverges. Forward $\dot { \mathrm { K L } }$ is thus strongly mode-covering (zero-avoiding). In LLM distillation, because a small student cannot memorize the teacher’s vast distribution, covering all modes spreads probability into the void between valid modes, producing nonsensical, average phrasing. 

2. Reverse Kullback-Leibler Divergence. Selecting $f ( u ) = - \log u$ recovers Reverse $\mathrm { K L , }$ $D _ { \mathrm { { K L } } } ( Q \parallel P ) = \mathbb { E } _ { x \sim Q } [ \log ( Q ( x ) / P ( x ) ) ]$ . The expectation is over the student Q. If $Q ( x ) > 0$ where $P ( x ) \approx 0 ,$ , the penalty diverges. The student therefore assigns probability only to regions supported by the teacher, yielding mode-seeking (zero-forcing) behavior, collapsing onto one major mode while ignoring minor ones. For generative language, this is desirable because producing one coherent answer is preferable to an incoherent amalgamation of multiple correct answers. 

3. Jensen-Shannon Divergence (JSD). JSD introduces symmetry by measuring divergence to a mixture $M = { \textstyle \frac { 1 } { 2 } } ( P + \bar { Q } )$ . Using $\begin{array} { r } { f ( u ) = \frac { 1 } { 2 } \left[ u \log \left( \frac { 2 u } { u + 1 } \right) + \log \left( \frac { 2 } { u + 1 } \right) \right] } \end{array}$ , JSD is bounded in $[ 0 , \log 2 ] .$ , offering a stable gradient field that balances mode-seeking and mode-covering behavior, preventing extreme gradient explosions in out-of-distribution regions. 

4. Total Variation Distance (TVD). Using $\begin{array} { r } { f ( u ) = \frac { 1 } { 2 } | u - 1 | . } \end{array}$ , TVD measures the maximum absolute probability difference over any event, $\operatorname* { s u p } _ { A } | P ( A ) - Q ( A ) |$ . TVD provides a robust 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e88928c2-7e16-4e31-ae61-a0256ad70706/57d8e9fdb4b01075e100ca2f061636e0c7631318b38ff229b5f8d2550115219f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e88928c2-7e16-4e31-ae61-a0256ad70706/088668e32b0912cbc148fb64fcffe6d8a7af4fffe1138a083dcbbc296c509959.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e88928c2-7e16-4e31-ae61-a0256ad70706/d18c1cd2b8dbf8406aa16c9a1847a9f09a195bc038886809315138efea8c2357.jpg)



Figure 1: Forward KL vs. Reverse KL divergence for fitting a student distribution $P _ { S }$ to a bimodal teacher $P _ { T }$ . (a) The teacher distribution with two modes. (b) Forward KL is mode-covering (zero-avoiding): the student covers both modes but places mass in the inter-mode “hallucination zone.” (c) Reverse KL is mode-seeking (zero-forcing): the student concentrates on a single peak, dropping the other mode entirely. Adaptive methods (ToDi, Entropy-Aware OPD) dynamically switch between the two based on teacher confidence.


distance metric irrespective of extreme log-probability ratios, though its non-differentiability at $u = 1$ challenges direct optimization in neural spaces. The choice from this $f .$ -divergence family dictates how the student interpolates teacher knowledge during on-policy generation. 

# 2.5 A Unified Mathematical View of Modern OPD

Recent OPD methods can be unified under a single objective by decoupling the sampling distribution (which dictates the trajectory) from the divergence metric (which dictates local token-level matching). The generalized OPD objective is 

$$
\mathcal {L} _ {\mathrm {O P D}} (\theta) = \mathbb {E} _ {y \sim \pi_ {\operatorname {m i x}}} \left[ \sum_ {t = 1} ^ {| y |} \mathcal {D} _ {f} \left(p _ {T} (\cdot | x, y _ {<   t}), p _ {\theta} (\cdot | x, y _ {<   t})\right) \right] \tag {14}
$$

where $\pi _ { \mathrm { m i x } }$ is the behavioral policy driving state exploration and $\mathcal { D } _ { f } ( \cdot , \cdot )$ is a divergence from the $f$ -divergence family. We use the two-argument notation $\operatorname { \mathcal { D } } _ { f } ( P , Q )$ rather than $\mathcal { D } _ { f } ( P \| Q )$ because different methods place the teacher and student in different argument positions: Forward KL computes $D _ { \mathrm { K L } } ( p _ { T } \Vert p _ { \theta } )$ (teacher-first), while Reverse KL computes $\mathbf { \dot { D } } _ { \mathrm { K L } } ( p _ { \theta } \Vert p _ { T } )$ (student-first). The key modern methods (GKD (Agarwal et al., 2024), MiniLLM (Gu et al., 2024), and DistiLLM (Ko et al., 2024)) all map onto this equation as distinct parameterizations of a continuous theoretical space, differing in the choice of $f _ { \cdot }$ , the argument ordering, and the target distribution. 

Generalized Knowledge Distillation (GKD) (Agarwal et al., 2024). GKD addresses trajectory mismatch by defining $\pi _ { \mathrm { m i x } }$ as an explicit interpolation between the dataset and the student’s on-policy distribution. The output sequence is drawn with probability $\lambda$ from $p _ { \theta }$ and with probability $1 - \lambda$ from $\mathcal { D }$ . GKD is flexible regarding ${ \mathcal { D } } _ { f } ,$ empirically testing Forward $\mathrm { K L , }$ Reverse $\mathrm { K L , }$ and JSD. Setting $\lambda \to 1$ makes GKD purely on-policy. 

MiniLLM (Gu et al., 2024). MiniLLM identifies that Forward KL forces over-allocation to the teacher’s long tail. To enforce mode-seeking behavior, it selects Reverse KL $( { \mathcal { D } } _ { f } =$ $D _ { \mathrm { K L } } ( p _ { \boldsymbol { \theta } } \parallel p _ { T } ) )$ . For sampling, MiniLLM employs a teacher-mixed strategy, where with probability $\alpha = 0 . 2$ , tokens are drawn from the teacher to alleviate reward hacking, making $\dot { \pi } _ { \mathrm { m i x } } = ( \dot { 1 } - \alpha ) p _ { \theta } + \alpha p _ { T }$ . Because Reverse KL places the student parameters in both the expectation and the log-ratio, MiniLLM reformulates optimization via the Policy Gradient theorem (REINFORCE), treating the teacher’s log-probability as a reward. Defining the 

cumulative reward-to-go $\begin{array} { r } { R _ { t } = \sum _ { t ^ { \prime } = t } ^ { | y | } \log \frac { p _ { T } ( y _ { t ^ { \prime } } | y _ { < t ^ { \prime } } ) } { p _ { \theta } ( y _ { t ^ { \prime } } | y _ { < t ^ { \prime } } ) } . } \end{array}$ |y| t log , the gradient becomes 

$$
\nabla_ {\theta} \mathcal {L} _ {\text {M i n i L L M}} = - \mathbb {E} _ {y \sim p _ {\theta}} \left[ \sum_ {t = 1} ^ {| y |} \left(R _ {t} - 1\right) \nabla_ {\theta} \log p _ {\theta} \left(y _ {t} \mid y _ {<   t}\right) \right] \tag {15}
$$

To mitigate the high variance of policy gradients, MiniLLM decomposes $R _ { t }$ into a single-step reward $\begin{array} { r } { r _ { t } = \log \frac { p _ { T } \left( y _ { t } | y _ { < t } \right) } { p _ { \theta } \left( y _ { t } | y _ { < t } \right) } } \end{array}$ plus a future return, enabling per-token baselines that bridge RL optimization with knowledge distillation. 

DistiLLM (Ko et al., 2024). While MiniLLM theoretically achieves mode-seeking behavior, reliance on high-variance RL causes empirical instability. DistiLLM resolves this with a pair of skewed KL divergences. The primary objective uses Skewed Forward KL: it defines a skewed mixture $\tilde { p } = \alpha p _ { T } + ( 1 - \dot { \alpha } ) p _ { \theta }$ and minimizes $D _ { \mathrm { K L } } ( p _ { T } \parallel \tilde { p } )$ , which avoids the zero-division instability of standard Forward KL when $p _ { \theta } ( y ) \approx 0 ,$ , because the mixture $\tilde { p }$ is always at least $\alpha p _ { T } ( y ) > 0$ . Since the expectation is over the fixed teacher $p _ { T } .$ , the loss reduces to a tractable cross-entropy without policy gradients. DistiLLM additionally employs a complementary Skewed Reverse KL, $\dot { D } _ { \mathrm { K L } } \big ( p _ { \theta } ^ { \smile } \llangle \big | _ { \vphantom { \int } } ( 1 - \alpha ) p _ { T } + \alpha p _ { \theta } \big ) .$ , on studentgenerated outputs, and the combination of both losses is shown to be synergistic. Under our unified view, DistiLLM maintains $\pi _ { \mathrm { m i x } } = p _ { \theta }$ and replaces the standard KL target from $p _ { T }$ to skewed mixtures, achieving tractability and stability. 

This unified perspective demonstrates that modern OPD is not a series of disparate heuristics, but a systematic, progressive exploration of a design space governed by three interacting choices: trajectory sampling $( \pi _ { \mathrm { m i x } } )$ , divergence generator $( f )$ , and the argument ordering and target distribution within the chosen divergence. 

# 3 Taxonomy of On-Policy Distillation

To navigate the expanding OPD literature, we propose a multi-dimensional taxonomy. Rather than categorizing methods by chronological release or minor architectural quirks, we classify them along three fundamental dimensions. (1) Feedback Signal, dictating what information flows from teacher to student. (2) Teacher Access, reflecting deployment constraints on the teacher model. (3) Granularity, describing the temporal resolution of the distillation loss. 

The structural hierarchy of this classification is visualized in Figure 2. 

# 3.1 Feedback Signal

The primary differentiator among OPD algorithms is the supervisory signal used to correct on-policy deviations. We partition this space into three paradigms. 

Logit-Based Feedback. The student’s generated sequence is evaluated token-by-token against the teacher’s continuous probability distribution. Methods like GKD (Agarwal et al., 2024) compute $f$ -divergences at every generative step, while MiniLLM (Gu et al., 2024) uses the teacher’s token-level log-probabilities to construct a sequence-level reward. More recent work, such as ToDi (Jung et al., 2025), adaptively combines Forward and Reverse KL per token using a sigmoid weight derived from the teacher-student log-probability ratio. Logit-based feedback provides the densest gradient signal, as the loss vector has size $\vert V \vert$ at every step $t ,$ , ensuring the student captures exact teacher decision boundaries. 

Outcome-Based Feedback. As models scale, computing full logits becomes prohibitive. Outcome-based methods replace exact token matching with scalar reward signals evaluating the correctness of a generated trajectory. Frameworks like RLKD (Xu et al., 2025b) let the student roll out a complete answer and receive a reward $R ( x , y )$ from a reward model or verifier. Optimization proceeds via policy gradient methods or Direct Preference Optimization (DPO), shifting the objective from divergence minimization to reward maximization, $\operatorname* { m a x } _ { \theta } \mathbb { E } _ { y \sim p _ { \theta } } [ R ( x , y ) ] - \beta D _ { \mathrm { K L } } ( p _ { \theta } \parallel p _ { \mathrm { r e f } } ) .$ . This grants the student freedom to find novel paths to correctness without mimicking the teacher’s specific phrasing. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/e88928c2-7e16-4e31-ae61-a0256ad70706/4e1bffb65ffa820bc39bd318f6c8b4056ff8cdd1ed1ead20415de6100cb187e0.jpg)



Figure 2: Taxonomy of on-policy distillation for large language models. The methodology space is organized along three orthogonal dimensions with representative methods. A single method may appear across multiple categories as these axes are independent.


Self-Play Feedback. In teacher-free alignment, self-play constructs feedback from the student’s own evolving policy. SPIN (Chen et al., 2024) simulates a two-player game where the model at iteration t distinguishes its own outputs from human-written references, with the log-likelihood gap providing an iteratively sharpening training signal. On-Policy Self-Distillation (OPSD) (Zhao et al., 2026) takes a different approach: a single model serves as both teacher and student by exploiting privileged information. The teacher variant conditions on the ground-truth answer to provide dense token-level supervision on studentgenerated rollouts. As the student improves, both paradigms provide naturally scaling curricula without continuous external teacher intervention. 

# 3.2 Teacher Access

The deployment constraints of frontier models severely limit how distillation can be executed. Teacher-internal-state availability dictates the allowable mathematical formulations. 

White-Box Distillation. When complete teacher weights are accessible, algorithms perform exact analytical divergence matching. DistiLLM (Ko et al., 2024) exploits the full logit tensor for skewed KL optimization. White-box access enables exploitation of dark knowledge, as minuscule probabilities on incorrect tokens carry structural regularization benefits. However, maintaining a massive teacher (e.g., 70B parameters) in memory alongside the student demands substantial engineering and large compute clusters. 

Black-Box Distillation. The closed-source nature of frontier models (e.g., GPT-4, Claude) often restricts access to API-level, returning only hard-argmax text. Black-box OPD must creatively approximate the teacher distribution. Lion (Jiang et al., 2023) uses the teacher as both preference annotator and curriculum designer. The student generates on-policy candidates, the teacher identifies weaknesses and creates harder instructions. Generative Adversarial Distillation (GAD) (Ye et al., 2025) implicitly models the teacher’s distribution by training a discriminator to distinguish student text from API output, sidestepping the need for logits entirely. 

Self-Distillation. When no superior teacher exists, the model bootstraps its own capabilities. Self-distillation treats a stabilized historical checkpoint, or an ensembled version via dropout, as the teacher. By generating on-policy responses and filtering them through self-evaluation (as in SPIN (Chen et al., 2024)), the model refines its distribution without external parameter dependencies. A recent extension, OPSDC (Sang et al., 2026), applies on-

policy self-distillation specifically to reasoning compression, where the same model serves as both teacher (conditioned on a “be concise” instruction) and student, with per-token Reverse KL minimized on the student’s own rollouts. This reduces reasoning trace length by $4 1 - 5 9 \%$ across benchmarks $( 5 7 - 5 9 \%$ on MATH-500, $4 1 \%$ on AIME 2024) while preserving or improving accuracy, addressing the practical concern that distilled reasoning models often produce unnecessarily verbose chains. GATES (Stein et al., 2026) tackles self-distillation in settings without ground-truth labels or external verifiers. A single model acts as both tutor (with access to a relevant source document during training) and student (answering from the question alone at test time). Rather than assuming tutor correctness, GATES derives supervision online from tutor consensus across multiple samples, gating the distillation signal to suppress unreliable supervision. SDPO (Hubotter et al. ¨ , 2026) introduces Self-Distillation Policy Optimization, which bridges RL and self-distillation by leveraging rich textual feedback (e.g., compiler errors, test outputs, judge evaluations) rather than scalar rewards. The model generates on-policy rollouts, receives structured textual feedback, and uses this feedback to create dense token-level learning signals via self-distillation, where the current model conditioned on feedback serves as a teacher for the unconditional student policy. This addresses the credit-assignment bottleneck that plagues standard RL with Verifiable Rewards (RLVR), where binary rewards provide no signal about which step caused failure. Across scientific reasoning, tool use, and competitive programming, SDPO improves both sample efficiency and final accuracy over strong RLVR baselines. Privileged Information Distillation (Penaloza et al., 2026) generalizes the GATES paradigm by formalizing the role of training-time privileged information (PI) in distillation. While the student must operate without PI at test time (e.g., answering questions without access to the source document), a teacher variant of the same model conditioned on PI generates high-quality supervision. The key contribution is demonstrating that this privileged self-distillation framework is particularly effective in hard, long-horizon RL settings where the base model cannot solve problems without PI, thereby expanding the set of tasks amenable to on-policy self-distillation. 

# 3.3 Granularity

The temporal resolution of the distillation loss determines whether the student learns complex reasoning or mere stylistic mimicry. 

Token-Level Granularity. The loss is computed at every timestep $t ,$ providing immediate, localized feedback on each action $y _ { t }$ . This ensures stable gradients and rapid convergence but suffers from myopia, as a token improbable under the teacher’s local distribution may be the optimal start of a valid alternative reasoning path. 

Sequence-Level Granularity. The signal is applied only at generation termination, commonly via Outcome Reward Models. While granting immense freedom to explore alternative structures, the gradient is extremely sparse. Credit assignment is acute: a 1000-token proof failing at step 999 receives a uniform negative reward, failing to reinforce the 998 correct steps. 

Hybrid / Adaptive Granularity. To bridge token myopia and sequence sparsity, several recent methods combine dense token-level supervision with step-level or trajectory-level reward signals (Lightman et al., 2024; Yang et al., 2025b; Xu et al., 2025b; Cai & Yuan, 2026). Process Reward Models (PRMs) (Lightman et al., 2024) provide per-step verification of intermediate reasoning, enabling fine-grained credit assignment that neither pure token-level nor pure sequence-level approaches achieve. SuperCorrect (Yang et al., 2025b) leverages hierarchical thought templates extracted from the teacher to guide step-level reasoning distillation, combined with cross-model DPO for self-correction. These hybrid granularity approaches balance the stability of dense token feedback with the structural awareness of step-level evaluation, enabling more effective distillation of complex reasoning chains. 

# 4 White-Box On-Policy Methods

The defining feature of on-policy distillation is not the choice of divergence metric but the distribution that governs training: traditional off-policy KD forces the student to mimic the teacher under the teacher’s marginal $p _ { T } ( x , y )$ , whereas OPD returns sampling authority to the student’s own evolving policy $p _ { \theta } ( x , y )$ . This shift eliminates exposure bias but introduces the challenge of efficiently and stably aligning the teacher’s dense logit signals over trajectories that the student, not the teacher, has generated. 

White-box methods enjoy a powerful advantage: full access to the teacher’s probability distributions enables dense, token-level feedback at every decoding step. This section traces the technical evolution of white-box OPD along four subsections. Section 4.1 covers tokenlevel divergence minimization, the most active research direction, organized around three conceptual threads: the progression from fixed to adaptive divergences, the orthogonal dimension of token weighting and selection, and cross-architecture distillation. Section 4.2 examines sequence-level methods that reframe distillation as reinforcement learning. Section 4.3 surveys hybrid approaches that bridge the granularity gap and address practical bottlenecks. Section 4.4 consolidates the theoretical analysis. 

# 4.1 Token-Level Divergence Minimization

Token-level methods allow the student to generate trajectories $y _ { < t } \sim p _ { \theta }$ and then minimize a divergence between teacher and student distributions at each position. The choice of divergence heavily influences stability, convergence, and downstream task quality. As established in the unified framework of Section 2.5, all such methods instantiate a common objective parameterized by a sampling policy $\pi _ { \mathrm { m i x } }$ and an $f$ -divergence $\mathcal { D } _ { f }$ . Rather than reviewing each method in isolation, we organize this subsection around three conceptual threads that capture the field’s technical evolution. 

# 4.1.1 From Fixed to Adaptive Divergences

The most fundamental design axis in token-level OPD is the divergence function used to measure teacher–student mismatch. The field has progressed from fixed, globally applied divergences toward increasingly adaptive, context-sensitive formulations. 

Baseline: uniform divergence with on-policy sampling. GKD (Agarwal et al., 2024) established the canonical on-policy distillation framework by introducing a mixture sampling policy $\pi _ { \mathrm { m i x } }$ that interpolates between the dataset and the student’s own generations, controlled by a student data fraction $\lambda \in [ 0 , 1 ]$ : 

$$
\mathcal {L} _ {G K D} = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \pi_ {\operatorname {m i x}} (\cdot | x)} \left[ \sum_ {t = 1} ^ {| y |} D \left(p _ {T} (\cdot | x, y _ {<   t}) \| p _ {\theta} (\cdot | x, y _ {<   t})\right) \right] \tag {16}
$$

where $D$ can be Forward KL, Reverse KL, or JSD. GKD defaults to JSD for its bounded, symmetric gradients, providing a safe general-purpose choice. Setting $\lambda = 1$ yields purely on-policy training, while $\lambda = 0$ recovers standard off-policy KD. While GKD substantially mitigates compounding errors compared to off-policy baselines, a single globally fixed divergence cannot adapt to the varying difficulty levels across tokens in a sequence. 

Stabilizing Reverse KL via interpolation. DistiLLM (Ko et al., 2024) addressed a specific failure mode of standard KL: gradient instability when there is a large probability mismatch between teacher and student. The solution is a skewed mixture target $\tilde { p } = \alpha \boldsymbol { p } _ { T } + ( 1 - \alpha ) \boldsymbol { p } _ { \theta }$ that guarantees bounded density ratios. The Skewed Forward KL (SKL) objective is: 

$$
\mathcal {L} _ {\mathrm {S K L}} = \mathbb {E} _ {y \sim p _ {\theta}} \left[ \sum_ {t} D _ {\mathrm {K L}} \left(p _ {T} (\cdot | y _ {<   t}) \| \alpha p _ {T} (\cdot | y _ {<   t}) + (1 - \alpha) p _ {\theta} (\cdot | y _ {<   t})\right) \right] \tag {17}
$$

DistiLLM combines student-generated outputs (SGOs) with an adaptive off-policy replay buffer, using $p _ { \theta }$ for generation and replaying past SGOs for sample efficiency. Since the inner KL expectation is taken over the fixed teacher $p _ { T . }$ , the per-token loss reduces 

to a tractable cross-entropy $- \sum p _ { T } \log \tilde { p }$ whose gradient with respect to $\theta$ flows analytically through the mixture $\tilde { p } .$ , avoiding the high-variance REINFORCE estimation that MiniLLM requires. DistiLLM additionally uses a complementary Skewed Reverse KL (SRKL), $D _ { \mathrm { K L } } \dot { ( } p _ { \theta } \| ( 1 - \alpha ) p _ { T } + \alpha p _ { \theta } ) .$ , on student-generated outputs, and the synergy of both losses further improves convergence. Building on this foundation, DistiLLM-2 (Ko et al., 2025) observes that teacher-generated and student-generated outputs benefit from different loss functions: Skew Forward KL (SKL) is applied to teacher outputs to increase the student’s likelihood of high-quality responses, while Skew Reverse KL (SRKL) is applied to student outputs to decrease the likelihood of lower-quality generations. This contrastive asymmetry, combined with curriculum-based adaptive coefficients and speculative-decoding-based dataset curation, achieves state-of-the-art results among token-level distillation methods across instruction following, code generation, and mathematical reasoning. 

Per-token adaptive divergence. The key limitation of GKD and DistiLLM is that they apply the same divergence uniformly across all tokens. Although not itself an on-policy method, ToDi (Jung et al., 2025) introduces a per-token adaptive divergence mechanism that directly inspired subsequent on-policy work. ToDi argues that different positions in a sequence demand different matching strictness: critical reasoning pivots require strict Reverse $\mathrm { K L , }$ while syntactic fillers tolerate relaxed Forward KL. The method adaptively combines Forward and Reverse KL at each position using a sigmoid-based weighting derived from the teacher–student log-probability ratio: 

$$
\mathcal {L} _ {\text {T o D i}} = \mathbb {E} _ {y \sim \mathcal {D}} \left[ \sum_ {t} \omega_ {t} D _ {f ^ {(t)}} \left(p _ {T} (\cdot | y _ {<   t}) \| p _ {\theta} (\cdot | y _ {<   t})\right) \right] \tag {18}
$$

where $y \sim \mathcal { D }$ reflects ToDi’s original off-policy formulation (training on dataset sequences). The adaptive weighting $\omega _ { t }$ is orthogonal to the sampling strategy and transfers directly to on-policy settings. When the ratio indicates that the student underestimates the teacher’s high-confidence predictions, Reverse KL dominates to suppress errors. When the teacher’s distribution is more uncertain, Forward KL contributes more to preserve diversity. The resulting per-token adaptability represents a qualitative advance over globally fixed divergences, though it introduces architectural complexity through the heuristic computation of $\omega _ { t }$ . 

Balancing head and tail of the distribution. AKL (Wu et al., 2025), also developed in an off-policy context, challenges the conventional wisdom that Reverse KL is strictly superior for LLM distillation. Through both theoretical analysis and experiments, Wu et al. (2025) demonstrate that the mode-seeking and mode-covering characterizations of Reverse and Forward KL do not strictly hold for discrete distributions in LLMs, and that both divergences converge to the same objective given sufficient training epochs. Under practical epoch budgets, however, Forward KL focuses learning on the head of the teacher distribution while Reverse KL focuses on the tail. AKL adaptively allocates weights to combine both divergences based on the current teacher and student distributions, achieving superior alignment under limited training budgets. This head-tail analysis complements ToDi’s per-token adaptation by providing a distributional rather than positional perspective on divergence selection. We include both ToDi and AKL in this section because their adaptive weighting mechanisms are directly applicable to on-policy sampling and have inspired subsequent on-policy methods such as Entropy-Aware OPD (Jin et al., 2026). 

Entropy-gated divergence switching. Entropy-Aware OPD (Jin et al., 2026) identifies a complementary failure mode: when the teacher distribution has high entropy (genuine uncertainty about the correct continuation), Reverse KL’s mode-seeking property forces the student to arbitrarily select one of many equally plausible modes, discarding valuable distributional information and yielding unstable learning signals. The solution is an entropygated mixture: 

$$
\begin{array}{l} \mathcal {L} _ {E n t - O P D} = \mathbb {E} _ {y \sim p _ {\theta}} \left[ \sum_ {t} \left(1 - \alpha_ {t}\right) D _ {\mathrm {K L}} \left(p _ {\theta} (\cdot | y _ {<   t}) \| p _ {T} (\cdot | y _ {<   t})\right) \right. \tag {19} \\ \left. + \alpha_ {t} D _ {\mathrm {K L}} \big (p _ {T} (\cdot | y _ {<   t}) \parallel p _ {\theta} (\cdot | y _ {<   t}) \big) \right] \\ \end{array}
$$

where $\alpha _ { t }$ increases with teacher entropy $\mathcal { H } ( p _ { T } ( \cdot | y _ { < t } ) )$ . In low-entropy regions (confident predictions), Reverse KL dominates for precise imitation, while in high-entropy regions, Forward KL captures the full range of plausible outputs. On Qwen3-4B-Base, this achieves a $+ 5 . 0 5$ Pass@8 improvement over standard Reverse KL on mathematical reasoning, making it one of the most robust token-level strategies against distribution shifts, though it requires computing teacher entropy at every step. 

OPD as KL-constrained reinforcement learning. G-OPD (Yang et al., 2026b) provides a unifying theoretical lens by proving that standard OPD is a special case of dense KLconstrained RL: minimizing the on-policy KL divergence between teacher and student is equivalent to maximizing a token-level reward (derived from the teacher’s log-probabilities) subject to a KL penalty against a reference policy. By decoupling the reference model from the teacher and introducing a reward scaling factor $\alpha$ , G-OPD generalizes the framework. When $\alpha = 1$ , it recovers standard OPD, and when $\alpha > 1$ (ExOPD), the student is incentivized to exceed the teacher’s distribution, breaking the imitation ceiling that constrains standard distillation. Empirically, $\operatorname { E x O P D }$ enables student models to surpass teacher performance on mathematical reasoning benchmarks. 

# 4.1.2 Token Weighting and Selection

Orthogonal to the choice of divergence is the question of which tokens matter most. Standard OPD objectives treat all positions uniformly, ignoring that some tokens (reasoning steps, rare vocabulary) carry far more information than others (articles, punctuation). 

AdaKD (Xie et al., 2025) dynamically adjusts per-token distillation weights based on the student’s current learning state, upweighting tokens where the student–teacher gap is large and the student’s confidence is low. SelecTKD (Huang et al., 2025) takes a complementary approach, selecting and reweighting tokens based on their learning value and filtering out tokens that contribute little to knowledge transfer. Both methods implement a curriculum-like strategy that improves distillation efficiency without modifying the underlying divergence. 

A more radical departure is Concrete Score Distillation (CSD) (Kim et al., 2026b), which challenges the softmax-based probability matching paradigm entirely. By operating directly on logits rather than probabilities, CSD avoids the information loss introduced by softmax normalization and captures richer structural relationships in the teacher’s output distribution, providing a theoretically grounded alternative to all KL-based objectives. 

# 4.1.3 Cross-Architecture Distillation

Conventional white-box KD assumes that teacher and student share the same vocabulary and representation space, an assumption increasingly violated in practice (e.g., distilling Llama into Qwen). 

Dual Space Knowledge Distillation (DSKD) (Zhang et al., 2025b) addresses this by introducing two projectors that map teacher hidden states into the student’s representation space and vice versa, enabling both models to share prediction heads for distribution comparison: 

$$
\mathcal {L} _ {D S K D} = D _ {\mathrm {K L}} \left(P _ {T \rightarrow S} \| P _ {S}\right) + D _ {\mathrm {K L}} \left(P _ {S \rightarrow T} \| P _ {T}\right) \tag {20}
$$

where $P _ { T  S }$ denotes the teacher’s hidden states projected into the student’s space and decoded through the student’s prediction head. An exact token alignment algorithm handles vocabulary mismatches by aligning identical tokens across different tokenizers. DSKD consistently outperforms standard white-box KD on instruction following, mathematical reasoning, and code generation, though the dual-space formulation approximately doubles memory overhead. 

Delta KD (Cao et al., 2025) takes a different approach to the capacity gap problem. Rather than aligning absolute distributions, it preserves the distributional shift that occurred during the teacher’s supervised fine-tuning. By constructing a target distribution $\pi _ { s } ^ { * }$ that captures the transformation from pre-trained to fine-tuned teacher, the student learns to reproduce relative changes rather than absolute outputs, improving knowledge transfer when compressing to much smaller models at the cost of requiring both pre-trained and fine-tuned model checkpoints. 


Table 1: Comprehensive comparison of white-box on-policy distillation methods.


<table><tr><td>Method</td><td>Year</td><td>Granularity</td><td>Divergence</td><td>Access</td><td>On-Policy Formulation</td><td>Key Innovation</td></tr><tr><td>GKD (Agarwal et al., 2024)</td><td>2024</td><td>Token</td><td>F-KL / R-KL / JSD</td><td>White-box</td><td>Student Rollouts</td><td>Unified OPD framework</td></tr><tr><td>G-OPD (Yang et al., 2026b)</td><td>2026</td><td>Token</td><td>KL-Constrained RL</td><td>White-box</td><td>Reward Extrapolation</td><td>OPD as RL; student surpasses teacher</td></tr><tr><td>MiniLLM (Gu et al., 2024)</td><td>2024</td><td>Seq.</td><td>Reverse KL</td><td>White-box</td><td>REINFORCE</td><td>Sequence-level reward</td></tr><tr><td>DistiLLM (Ko et al., 2024)</td><td>2024</td><td>Token</td><td>Skew KL (SKL+SRKL)</td><td>White-box</td><td>Interpolated</td><td>α-smoothing for KL</td></tr><tr><td>DistiLLM-2 (Ko et al., 2025)</td><td>2025</td><td>Token</td><td>Contrastive SKL/SRKL</td><td>White-box</td><td>Batched On-Policy</td><td>Asymmetric loss per data type</td></tr><tr><td>DSKD (Zhang et al., 2025b)</td><td>2025</td><td>Token</td><td>Dual-Space KL</td><td>White-box</td><td>Cross-Space Projection</td><td>Vocabulary-agnostic KD</td></tr><tr><td>Constrained (Zimmer et al., 2025)</td><td>2025</td><td>Seq.</td><td>Constrained KL</td><td>White-box</td><td>State-Augmented CMDP</td><td>Reward-aware distillation with hard KL constraint</td></tr><tr><td>KETCHUP (Fan et al., 2025)</td><td>2025</td><td>Seq.</td><td>Reverse KL</td><td>White-box</td><td>K-step Return</td><td>Multi-step REINFORCE with reduced variance</td></tr><tr><td>Entropy-Aware (Jin et al., 2026)</td><td>2026</td><td>Token</td><td>Adaptive FKL+RKL</td><td>White-box</td><td>Entropy-Adaptive</td><td>High-entropy robustness</td></tr><tr><td>Fast OPD (Zhang et al., 2026a)</td><td>2026</td><td>Hybrid</td><td>R-KL on Prefix</td><td>White-box</td><td>Prefix-Truncated</td><td>Reasoning prefix reuse</td></tr><tr><td>PACED (Xu et al., 2026)</td><td>2026</td><td>Hybrid</td><td>Adaptive</td><td>White-box</td><td>Frontier Curriculum</td><td>Student competence boundary</td></tr><tr><td>X-KD (Cai &amp; Yuan, 2026)</td><td>2026</td><td>Hybrid</td><td>AVRIL + KL</td><td>White-box</td><td>Reward-Guided</td><td>Teacher environment reconstruction</td></tr><tr><td>AdaSwitch (Peng et al., 2025)</td><td>2025</td><td>Hybrid</td><td>KL (adaptive)</td><td>White-box</td><td>One-time Switch</td><td>Adaptive on/off-policy switching</td></tr><tr><td>REPOLD (Ko et al., 2026)</td><td>2026</td><td>Token</td><td>Reward-clipped R-KL</td><td>White-box</td><td>Relaxed Imitation</td><td>OPD as policy optimization</td></tr><tr><td>TSD-KD (Kim &amp; Baek, 2026)</td><td>2026</td><td>Token</td><td>Dual KD (indirect + direct)</td><td>White-box</td><td>Student Re-ranking</td><td>Token-selective student-centric</td></tr><tr><td>Veto (Jang et al., 2026)</td><td>2026</td><td>Token</td><td>Geometric Bridge KL</td><td>White-box</td><td>Adaptive Gradient Veto</td><td>Logit-space bridge for stable OPD</td></tr><tr><td>DASD (Yan et al., 2026)</td><td>2026</td><td>Seq.</td><td>Distribution-aligned</td><td>White-box</td><td>On-policy Correction</td><td>Distribution alignment for seq. KD</td></tr><tr><td>DDT (Zhang et al., 2026b)</td><td>2026</td><td>Seq.</td><td>IDFT + Hinted Decoding</td><td>White-box</td><td>Distribution-aligned SFT</td><td>On-policy SFT theory</td></tr></table>

# 4.1.4 Cross-Method Analysis

Why different divergences for different methods? The divergence choices across tokenlevel methods are not arbitrary, as each reflects a deliberate response to specific failure modes. GKD (Agarwal et al., 2024) defaults to JSD for bounded, symmetric gradients. DistiLLM (Ko et al., 2024) adopts skewed KL divergences (both Forward and Reverse variants) because standard KL produces gradient instabilities when there are large probability mismatches between teacher and student. ToDi (Jung et al., 2025) and Entropy-Aware OPD (Jin et al., 2026) argue that the optimal divergence varies within a single sequence and adapt accordingly. Empirically, these design choices manifest in task-specific advantages: on mathematical reasoning benchmarks (GSM8K, MATH), Reverse KL variants consistently outperform Forward KL because mode-seeking behavior prevents the student from averaging over distinct solution strategies. On open-ended generation tasks (MT-Bench, AlpacaEval), Forward KL and JSD preserve the diversity needed for creative and conversational responses. Practitioners should match their divergence choice to the downstream task’s tolerance for mode-dropping versus hallucination (see Figure 1). 

A deeper geometric view. The Forward/Reverse KL dichotomy, while instructive, obscures a subtler geometric reality. Under Forward KL, the student’s distribution is pulled toward the centroid of the teacher’s probability mass, which for multimodal distributions lies in a low-density region between modes, explaining the hallucination pathology. Under Reverse KL, the student collapses onto a single high-density mode, yielding confident but narrow outputs. JSD traces a path that partially covers multiple modes without fully committing to any, explaining its empirically observed “jack-of-all-trades” behavior. 

This perspective also illuminates why adaptive divergences (ToDi, AKL, Entropy-Aware) consistently outperform fixed ones in practice. At reasoning-critical tokens, the teacher’s distribution is typically unimodal and Reverse KL aligns well. At generation-flexible tokens (e.g., choosing between “therefore” and “thus”), the distribution is near-uniform over synonyms and Forward KL preserves this diversity. Adaptive methods implicitly detect this modality structure and switch accordingly, achieving the best of both worlds. 

A detailed, multidimensional comparison of white-box methods is presented in Table 1. Table 2 provides experimental configurations across all surveyed methods. 

# 4.2 Sequence-Level On-Policy Methods

While token-level methods provide dense supervision, they inherently suffer from greedy optimization bias: each token is matched independently, ignoring the joint probability of the full sequence. Sequence-level OPD reframes distillation as trajectory-level reinforcement learning, treating the complete output as a single decision and optimizing a global reward. 

MiniLLM and the REINFORCE derivation. MiniLLM (Gu et al., 2024) serves as the foundational sequence-level method. As introduced in Section 2.5, MiniLLM selects Reverse KL to enforce mode-seeking behavior at the sequence level. The central challenge is computing the gradient of an expectation over $p _ { \theta }$ when $p _ { \theta }$ itself is parameterized by θ. Using the log-derivative trick (REINFORCE), the derivation proceeds as follows. Starting from 


Table 2: Experimental configurations of representative on-policy distillation methods. “Self” indicates self-distillation where the same model serves as both teacher and student. Dashes indicate information not available from the original paper.


<table><tr><td>Method</td><td>Teacher</td><td>Student</td><td>Loss / Objective</td><td>Primary Tasks</td></tr><tr><td colspan="5">White-Box Token-Level</td></tr><tr><td>GKD (Agarwal et al., 2024)</td><td>T5-XL (3B)</td><td>T5-Small (60M) / Base (220M) / Large (770M)</td><td>F-KL / R-KL / JSD</td><td>XSum, WMT, GSM8K</td></tr><tr><td>G-OPD (Yang et al., 2026b)</td><td>Qwen3-4B / 30B-A3B</td><td>Qwen3-17B / 4B</td><td>KL-constrained RL (λ=1.25)</td><td>AIME, MATH, HumanEval</td></tr><tr><td>DistillLM (Ko et al., 2024)</td><td>GPT-2 XL (1.5B), OpenLLaMA-7B</td><td>GPT-2 (124M), OpenLLaMA-3B</td><td>Skew KL (SKL+SRKL, α=0.1)</td><td>Dolly, S-NI, XSum</td></tr><tr><td>DistillLM-2 (Ko et al., 2025)</td><td>Gamma-2-9B-Inst, Mistral-7B-Inst</td><td>Gamma-2-2B, Danube-1.8B</td><td>Contrastive SRKL</td><td>AlpacEval, GSM8K, HumanEval</td></tr><tr><td>Entropy-Aware (Jin et al., 2026)</td><td>Qwen3-8B</td><td>Qwen3-68B/1.78/B-4B-Base</td><td>Entropy-gated R-KL/F-KL</td><td>AIME, MATH, MMLU</td></tr><tr><td>Veto (Jiang et al., 2026)</td><td>Qwen2-7B-IT</td><td>Qwen2-5.5B-IT</td><td>Geometric bridge KL</td><td>Math, Code, Summarization</td></tr><tr><td>Devisiting OPD (Fu et al., 2026)</td><td>OpenTet-3-7B</td><td>Qwen2-5.8B-IT</td><td>Truncated R-KL + top-p</td><td>Math, Agent + Math</td></tr><tr><td>DSKD (Ko et al., 2026)</td><td>Qwen2.5-7B-Inst, Qwen2.5-Math-7B</td><td>DadeSpace KL</td><td>Divided space KL</td><td>Instruction following Reasoning, Code</td></tr><tr><td>REPOOLD (Ko et al., 2026)</td><td>Qwen3-32B</td><td>Qwen3-7B</td><td>Reward-clipped R-KL</td><td>AIME, Math, Visual reasoning</td></tr><tr><td>TSKD-KD (Kim &amp; Baek, 2026)</td><td>Qwen2.5-14B-Inst</td><td>Qwen2.5-1.5B-Inst</td><td>Token-selective dual KD</td><td>Math, Code, Science</td></tr><tr><td colspan="5">White-Box Sequence-Level &amp; Hybrid</td></tr><tr><td>MiniLLM (Gu et al., 2024)</td><td>GPT-2.1.5B, OPT-13B, LLaMA-13B</td><td>GPT-2 120M-760M, OPT 1.3B-6.7B, LLaMA 7B</td><td>Seq-level R-KL (REINFORCE)</td><td>Instruction following</td></tr><tr><td>Constrained (Zimmer et al., 2025)</td><td>Qwen2.5-7B-Math</td><td>Qwen2.5-1.5B-Math</td><td>CMDP (reward + KL constraint)</td><td>MATH, GSM8K</td></tr><tr><td>KETCHUP (Fan et al., 2025)</td><td>FLAN-T5-XL (3B)</td><td>T5-Base</td><td>K-step return REINFORCE</td><td>XSum, Europarl, GSM8K</td></tr><tr><td>Fast OPD (Zhang et al., 2026a)</td><td>Qwen3-8B</td><td>Qwen3-1.7B / 8B-Base</td><td>R-KL on prefix only</td><td>AIME, MATH, LiveCode</td></tr><tr><td>PACED (Xu et al., 2026)</td><td>Qwen3-14B</td><td>Qwen3-8B</td><td>Beta-kernel weighted KL</td><td>AIME, MATH</td></tr><tr><td>AdSswitch (Peng et al., 2025)</td><td>Qwen2.5-3B, Llama-3.1-3B, Gamma-7B</td><td>Qwen2.5-5B, Llama-3.1-1B, Gamma-2B</td><td>Adaptive on/off-policy KL</td><td>Summarization, GSM8K, GSM-Plus</td></tr><tr><td>DASS (Cai &amp; Yuan, 2026)</td><td>T5-Large (780M)</td><td>T5-Small (60M) / Base (220M)</td><td>AVRIIL-based KL</td><td>Summarization, Translation, Arithmetic</td></tr><tr><td>DASS (MATE et al., 2026)</td><td>DeepSeek-R1</td><td>DASS-4B-Thinking (4B)</td><td>Distribution-aligned seq, KD</td><td>AIME, MATH, LiveCode</td></tr><tr><td>DDT (Zhang et al., 2026b)</td><td>N/A (on-policy SFT)</td><td>Qwen2.5-7B-Inst</td><td>IDFT + Hinted Decoding</td><td>Alignment, Reasoning</td></tr><tr><td colspan="5">Black-Box</td></tr><tr><td>GAD (Ye et al., 2025)</td><td>GPT-5-Chat (black-box)</td><td>Qwen2.5-14B-Inst</td><td>Adversarial (minimax)</td><td>LMSYS-Chat, Dolly, Vicuna</td></tr><tr><td>Lion (Jiang et al., 2023)</td><td>ChatGPT (black-box)</td><td>LLaMA-7B / 13B</td><td>Adversarial curriculum</td><td>BIG-Bench Hard, AGIEval</td></tr><tr><td>OVD (Xiong et al., 2026)</td><td>QwQ-32B / Environment Agent</td><td>Qwen2.5-3B, Llama-3.2-3B, Qwen2.5-Math-1.5B</td><td>Verbal scores (0-9)</td><td>Web QA, MATH</td></tr><tr><td>ORPO-Disstill (Singh et al., 2025)</td><td>InternalLM2.5-7B-Chat</td><td>InternalLM2.5-1.8B, TinyLlama-1.1B</td><td>ORPO (reference-free)</td><td>GSM8K, ARC, MedQA, StrategyQA</td></tr><tr><td>DAIL (Mendes et al., 2026)</td><td>Expert solutions (black-box)</td><td>Qwen2.5-7B-Inst</td><td>Contrastive imitation</td><td>AIME, GPQA</td></tr><tr><td colspan="5">Self-Play &amp; Self-Distillation</td></tr><tr><td>SPIN (Chen et al., 2024)</td><td>Self (iter t-1)</td><td>Zephyr-7B-SFT</td><td>Self-play DDO</td><td>MT-Bench, AlpacaEval</td></tr><tr><td>OPSD (Zhao et al., 2026)</td><td>Self (privileged info-conditioned)</td><td>Qwen3-17B/4B/8B-Inst</td><td>F-KL / JSD on rollouts</td><td>AIME, MATH, HMMT</td></tr><tr><td>OPSDC (Sang et al., 2026)</td><td>Self (concise prompt)</td><td>Qwen3-8B / 14B</td><td>Per-token-R-KL</td><td>AIME, MATH-500</td></tr><tr><td>GATES (Stein et al., 2026)</td><td>Self (document-conditioned)</td><td>Qwen3-4B-Base</td><td>Consensus-gated KL</td><td>Open-domain QA</td></tr><tr><td>SDPO (Hübter et al., 2026)</td><td>Self (textual feedback)</td><td>Qwen3-8B, OLMo3-7B</td><td>Self-distill + credit assign</td><td>Code, math, science</td></tr><tr><td>MTP Self-Distill (Kirchenbauer et al., 2026)</td><td>Self (same checkpoint)</td><td>Llama-3.1-8B, Qwen2.5-7B</td><td>Multi-token prediction</td><td>GSM8K, MATH, MMLU</td></tr><tr><td>OPCD (Ye et al., 2026b)</td><td>Self (context-augmented)</td><td>Qwen3-1.7B / 4B / 8B</td><td>R-KL context distillation</td><td>Knowledge, sys prompts</td></tr><tr><td>OEL (Ye et al., 2026a)</td><td>Self (exponential)</td><td>Qwen3-1.7B / 4B / 8B</td><td>KL + context distillation</td><td>Text-based games</td></tr><tr><td>SDFT (Shentfeld et al., 2026)</td><td>Self (demo-conditioned)</td><td>Qwen2.5-7B-Inst</td><td>Self-distill + KL reg</td><td>Skill learning, Tool Use</td></tr><tr><td>Priv. Info. Distill (Penaloza et al., 2026)</td><td>Self (PI-conditioned)</td><td>Qwen3-4B / 8B, R1-Distill-Llama-8B</td><td>Privileged self-distill KL</td><td>Long-horizon RL</td></tr><tr><td>HDPO (Ding, 2026)</td><td>Self (privileged, ground-truth)</td><td>Qwen2.5-Math-1.5B-Inst</td><td>GRPO + JSD self-distill</td><td>MATH, AMC, AIME</td></tr><tr><td>TMS (Khan et al., 2026)</td><td>Self (historical checkpoints)</td><td>Qwen2.5-7B-Inst</td><td>On-policy curriculum SFT</td><td>MATH, GSM8K</td></tr><tr><td colspan="5">Reasoning &amp; RL Integration</td></tr><tr><td>RLKD (Xu et al., 2025b)</td><td>DeepSeek-R1 paths + GSRM (7B)</td><td>DeepSeek-R1-Distill-Qwen-7B</td><td>KL-regulated RL + GSRM</td><td>Math reasoning</td></tr><tr><td>RLAD (Zhang et al., 2026c)</td><td>Qwen3-8B</td><td>Qwen3-0.6B / 1.7B</td><td>Trust-region ratio distill</td><td>AIME, GSM8K, ARC</td></tr><tr><td>AlignDistill (Zhang et al., 2025a)</td><td>Synthetic DPO + reverse DPO</td><td>Qwen2-1.5B-Inst, Qwen2.5-1.5B-Inst</td><td>DPO + token-level KD</td><td>AlpacaEval, MT-Bench, Arena-Hard</td></tr><tr><td>LUFFY (Yan et al., 2025)</td><td>DeepSeek-R1 traces (black-box)</td><td>Qwen2.5-Math-7B / 1.5B</td><td>Mixed-policy GRPO</td><td>GSM8K, MATH, AIME</td></tr><tr><td>SuperCorrect (Yang et al., 2025b)</td><td>ol-mini, gpt-4o-mini (black-box)</td><td>Qwen2.5-Math-7B, Llama-3.1-8B</td><td>KD + self-correction reward</td><td>Math, GSM8K</td></tr><tr><td>KDRL (Xu et al., 2025a)</td><td>Skywork-OR1-Math-7B</td><td>DeepScale1.5B</td><td>Joint KD + RL objective</td><td>AIME, MATH</td></tr><tr><td>KEPO (Yang et al., 2026a)</td><td>Qwen3-VL-32B</td><td>Qwen3-VL-2B</td><td>KD + preference optim.</td><td>OmniMedVQA</td></tr><tr><td>ScORe (Lyu et al., 2025)</td><td>Qwen2.5-72B-Inst</td><td>Qwen2.5-7B/3B-Inst, Llama-3.1-8B</td><td>Earliest-error correction</td><td>AIME, MATH, Multi-hop QA</td></tr><tr><td colspan="5">Multimodal &amp; Domain-Specific</td></tr><tr><td>VOLD (Bousselham et al., 2025)</td><td>Qwen3-8B (text-only)</td><td>Qwen2.5-VL-3B-Inst</td><td>GRPO + token-level KL</td><td>MMMU-Pro, MathVision</td></tr><tr><td>Video-OPD (Li et al., 2026)</td><td>Qwen3-VL-32B-CRPO</td><td>Qwen3-VL-8B-Inst</td><td>On-policy temporal distill</td><td>TVG (Charades, ActivityNet)</td></tr><tr><td>X-OPD (Cao et al., 2026)</td><td>Qwen3-Omni (text mode)</td><td>Qwen3-Omni-A3B (speech mode)</td><td>Cross-modal on-policy KL</td><td>Speech understanding</td></tr><tr><td colspan="5">Systems &amp; Scaling</td></tr><tr><td>Speculative KD (Xu et al., 2025c)</td><td>Gamma-7B-IT, Qwen2-7B-IT</td><td>Gamma-2B, Qwen2-0.5B</td><td>Block-verified on-policy KL</td><td>Translation, Dialogue</td></tr><tr><td>DistillSpec (Zhou et al., 2023)</td><td>T5-XL (3B), GPT-like (234M)</td><td>T5-Small (77M), GPT-like (33M)</td><td>On-policy draft → target</td><td>XSum, CNN/DM</td></tr><tr><td>Cross-Tok. KD (Boizard et al., 2025)</td><td>Llama-2.7B-Chat, Mistral-7B-Inst</td><td>OPT-350M, Pythia-160M-1B, Bloomz-560M</td><td>Latent OT alignment</td><td>QA, Summarization</td></tr><tr><td>Gamma 2 (Gemma Team et al., 2024)</td><td>Qwen2.2B / 9B → 2B</td><td>Qwen2.2B / 9B / 2B</td><td>Online KD in pre-training</td><td>MMLU, HellSAw, GSM8K</td></tr><tr><td>Qwen3 (Yang et al., 2025a)</td><td>Qwen3-32B / 25B-A2B</td><td>Qwen3-0.6B-1.4B, 30B-A3B</td><td>On-Policy Distillation</td><td>AIME, MATH, LiveCode</td></tr><tr><td>MiMo-V2 (Xiaomi LLL-Core Team et al., 2026)</td><td>Domain-specialized teachers (RL/SFT)</td><td>MiMo-V2-Flash (30B/15B MoE)</td><td>Multi-token light + reward</td><td>AIME, MATH, GPQA</td></tr><tr><td>Nemotron-Cascade 2 (Yang et al., 2026c)</td><td>Domain RL/SFT teachers</td><td>Nemotron 30B MoE (3B active)</td><td>Cascade RL + domain OPD</td><td>IMO, IOL, ICPC</td></tr></table>

$D _ { \mathrm { K L } } ( p _ { \theta } \parallel p _ { T } ) = \mathbb { E } _ { y \sim p _ { \theta } } [ \log p _ { \theta } ( y | x ) - \log p _ { T } ( y | x ) ]$ and applying the product rule: 

$$
\begin{array}{l} \nabla_ {\theta} D _ {\mathrm {K L}} \left(p _ {\theta} \| p _ {T}\right) = \nabla_ {\theta} \sum_ {y} p _ {\theta} (y | x) (\log p _ {\theta} (y | x) - \log p _ {T} (y | x)) \\ = \sum_ {y} p _ {\theta} (y | x) \nabla_ {\theta} \log p _ {\theta} (y | x) \log \frac {p _ {\theta} (y | x)}{p _ {T} (y | x)} + \sum_ {y} p _ {\theta} (y | x) \nabla_ {\theta} \log p _ {\theta} (y | x) \tag {21} \\ \end{array}
$$

Since $\begin{array} { r } { \sum _ { y } p _ { \theta } ( y | x ) = 1 } \end{array}$ and its gradient vanishes, the final gradient simplifies to: 

$$
\nabla_ {\theta} \mathcal {L} _ {\text {M i n i L L M}} = \mathbb {E} _ {y \sim p _ {\theta}} \left[ \left(\log p _ {\theta} (y | x) - \log p _ {T} (y | x)\right) \nabla_ {\theta} \log p _ {\theta} (y | x) \right] \tag {22}
$$

This reveals that minimizing sequence-level Reverse KL is mathematically equivalent to policy gradient RL with reward $r ( \dot { x , y } ) = \log p _ { T } ( y | x ) - \log p _ { \theta } ( y | x )$ . MiniLLM decomposes this into per-token rewards and future returns, enabling variance-reducing baselines. However, REINFORCE estimation in massive combinatorial output spaces demands substantially more training iterations and sophisticated baseline subtraction to converge. 

Constrained optimization for stability. To combat the high variance of REINFORCE, Zimmer et al. (2025) recast distillation as a constrained Markov Decision Process (CMDP). Instead of treating the KL penalty as a soft Lagrangian term with manually tuned $\lambda _ { \iota }$ they maximize task reward subject to a hard constraint $D _ { \mathrm { K L } } ( p _ { \boldsymbol { \theta } } \parallel p _ { T } ) \leq \epsilon$ . By adapting constrained state-augmented RL, the method maintains theoretical guarantees of constraint satisfaction without requiring the teacher at deployment time. KETCHUP (Fan et al., 2025) addresses the variance problem from a different angle by inducing a $K$ -step return via the Bellman Optimality Equation, replacing the high-variance single-step REINFORCE 

estimator with a multi-step formulation that provably reduces gradient variance, yielding more stable RL-based distillation especially for larger student models. 

Token-level vs. sequence-level: the fundamental tradeoff. Comparing these granularities reveals a core tension. Token-level methods optimize an upper bound of the sequence-level divergence, and their enforced local alignment yields low variance and high stability, but greedy optimization can perfect local syntax while missing global coherence. Sequence-level methods directly model trajectory reward and thus capture holistic reasoning structure, but the high variance of Monte Carlo sampling demands substantially more compute. The hybrid approaches in Section 4.3 are designed precisely to navigate this tradeoff. 

# 4.3 Hybrid and Adaptive Approaches

Recognizing that pure token-level distillation lacks long-term planning while pure sequencelevel distillation suffers from unmanageable variance, contemporary research has converged on hybrid architectures that combine both granularities or address the primary practical bottleneck of OPD: the compute–quality tradeoff of generating on-policy trajectories. We organize these approaches along four optimization dimensions: compute efficiency, capacity gap bridging, curriculum design, and novel paradigms. 

# 4.3.1 Compute Efficiency

The most direct bottleneck of OPD is the cost of generating full student rollouts at every training step. Two approaches reduce this cost from complementary angles. 

Fast OPD (Zhang et al., 2026a) observes that during standard OPD, training signals are concentrated in the prefix of each output, and even a short teacher-supervised prefix suffices to help the student produce correct answers. By truncating both the sampling horizon and the loss computation to the prefix region, Fast OPD matches full OPD performance while reducing training FLOPs by $\bar { 2 } \times$ to $4 7 \times$ . Speculative KD (Xu et al., 2025c), discussed further in Section 7, uses the student’s generations as speculative drafts that the teacher verifies in parallel, producing on-policy training data without the full cost of independent teacher generation. 

# 4.3.2 Capacity Gap Bridging

When the teacher–student capacity ratio exceeds $1 0 \times ,$ directly matching the teacher’s distribution causes mode averaging: the student’s limited parameters blur together the teacher’s distinct output modes. 

TAID (Shing et al., 2025) addresses this by constructing a time-dependent intermediate distribution $P _ { t } = ( 1 - \lambda _ { t } ) P _ { \theta } + \lambda _ { t } P _ { T } ,$ where the interpolation ratio $\hat { \lambda } _ { t }$ gradually increases from near-zero (close to the student) toward one (close to the teacher) as training progresses. Although formulated in an off-policy context, the student minimizes Forward KL against this intermediate target, converting the large teacher–student gap into a series of smaller, more tractable distillation steps. PromptKD (Kim et al., 2024) takes a dual approach: rather than adapting the student’s target, it adapts the teacher’s output by prepending learnable prompt tokens that cause the teacher to produce “student-friendly” distributions more accessible to the student’s limited capacity, adding only $0 . 0 0 0 7 \%$ of the teacher’s parameters. While both TAID and PromptKD originally rely on static datasets, their capacity-bridging mechanisms are orthogonal to the sampling strategy and can be integrated into on-policy rollouts. 

Veto (Jang et al., 2026) tackles the same capacity gap from an on-policy perspective. Rather than mixing data samples, it constructs an intermediate target distribution through a geometric bridge in logit space. A tunable parameter $\beta$ serves dual roles: as an Adaptive Gradient Veto that suppresses pathological gradients on low-confidence tokens (addressing Forward KL instability), and as a Decisiveness Knob that balances reward-driven performance with output diversity (addressing Reverse KL mode collapse). On reasoning and generation tasks, Veto consistently outperforms both supervised fine-tuning and existing on-policy baselines including GKD and DistiLLM. 

# 4.3.3 Curriculum and Difficulty Adaptation

Not all training instances are equally informative. Several methods design curricula that focus compute on the most productive examples (Xu et al., 2026; Peng et al., 2025; Jiang et al., 2023). 

PACED (Xu et al., 2026) proves through gradient signal-to-noise ratio (SNR) analysis that the distillation gradient SNR vanishes at both extremes of student capability: when the student’s pass rate on a problem approaches 0 (too hard) or 1 (too easy). To exploit the “zone of proximal development” where intermediate solve rates concentrate the optimal learning signal, PACED implements a Beta kernel weighting scheme $w ( p ) = p ^ { \alpha } ( 1 { - } p ) ^ { \beta }$ that assigns maximum weight to sequences near the student’s competence frontier. The framework is provably minimax-robust, with worst-case efficiency loss of only $O ( \delta ^ { 2 } )$ under bounded multiplicative misspecification. A two-stage divergence schedule (Forward KL for broad mode coverage, then Reverse KL for consolidation) yields the strongest results. 

AdaSwitch (Peng et al., 2025) introduces adaptive switching at the token level: the student begins generating each sequence autonomously (on-policy), and once divergence exceeds a context-aware threshold, teacher guidance is selectively integrated. Unlike prior mixing methods that frequently toggle between on-policy and off-policy tokens, AdaSwitch uses a principled switching criterion that preserves semantic coherence while ensuring high-quality supervision. 

# 4.3.4 Novel Paradigms

Several methods depart from the standard divergence-minimization framework entirely (Cai & Yuan, 2026; Ko et al., 2026; Kim & Baek, 2026; Yan et al., 2026; Zhang et al., 2026b), introducing qualitatively different optimization principles that reframe what distillation means. 

$\mathcal { X }$ -KD (Cai & Yuan, 2026) draws on experiential learning theory and inverse reinforcement learning, adopting the Approximated Variational Reward Imitation Learning (AVRIL) framework to jointly model the teacher’s original reward function and perform policy distillation. This encourages the student to learn in a reconstructed version of the teacher’s learning environment rather than simply mimicking outputs, achieving better performance– diversity tradeoffs than GKD and MiniLLM baselines. 

While $\mathcal { X }$ -KD reconstructs the teacher’s environment, REOPOLD (Ko et al., 2026) bridges OPD and policy optimization more directly by proving that the teacher–student loglikelihood ratio acts as a token-level reward. Building on this insight, it relaxes strict imitation constraints through mixture-based reward clipping (preventing over-trust of extreme teacher signals), entropy-based dynamic sampling (selectively leveraging teacher feedback at high-uncertainty tokens), and a unified exploration-to-refinement strategy. RE-OPOLD achieves $6 . 7 – 1 2 \times$ greater sample efficiency than recent RL approaches, enabling a 7B student to match a 32B teacher in visual reasoning with ${ \sim } 3 . 3 \times$ inference speedup. 

Complementing these dual approaches, TSD-KD (Kim & Baek, 2026) combines indirect and direct distillation: the student generates on-policy candidates that the teacher re-ranks via preference feedback (indirect), and distribution matching is then applied selectively based on relative teacher–student confidence (direct), focusing on high-entropy reasoning tokens. This dual approach outperforms baselines by up to $5 4 . { \overset { \sim } { 4 } } \%$ on 10 reasoning benchmarks. 

The remaining two methods in this category challenge the SFT-based distillation paradigm itself. DASD (Yan et al., 2026) critically reexamines the dominant paradigm of SFT on teachergenerated reasoning traces, identifying three limitations: inadequate representation of the teacher’s sequence-level distribution from single best-of-n samples, misalignment between teacher output distribution and student capacity, and exposure bias from teacher-forced training. Its enhanced distribution-aligned pipeline achieves state-of-the-art performance at comparable scale using only 448K training samples. 

DDT (Zhang et al., 2026b) provides a theoretical framework explaining why RL generalizes better than SFT: the key factor is RL’s on-policy data generation, which keeps training 

aligned with the model’s evolving distribution. DDT proposes In-Distribution Finetuning (re-weighting losses to down-weight out-of-distribution samples) and Hinted Decoding (realigning the training corpus to the model’s current distribution), achieving generalization surpassing DPO and SimPO while maintaining SFT efficiency. 

# 4.4 Theoretical Analysis and Method Comparison

The theoretical landscape of white-box OPD centers on the geometry of divergence functions in high-dimensional probability spaces. We synthesize the key analytical results and their practical implications. 

Forward KL vs. Reverse KL: beyond the standard dichotomy. The conventional view holds that Forward KL is zero-avoiding (mode-covering), meaning the student spreads mass to cover all teacher modes, risking probability in inter-mode regions, while Reverse KL is zero-forcing (mode-seeking), concentrating on a single mode at the cost of diversity. However, Wu et al. (2025) challenge this dichotomy for discrete distributions in LLMs, demonstrating that the mode-seeking and mode-covering characterizations do not strictly hold, and that both divergences converge to the same objective given sufficient training epochs. Under practical epoch budgets, the difference is better understood through the lens of head-tail focus: Forward KL concentrates learning on the head of the teacher distribution, while Reverse KL emphasizes the tail. Zhong et al. (2024) provide complementary evidence that the choice between Forward and Reverse KL is task-dependent rather than universally prescribable, with adaptive combinations consistently outperforming either fixed choice. 

The unified $f$ -divergence perspective. Wen et al. (2023) showed that any $f$ -divergence can be expressed as $D _ { f } \big ( p _ { T } \ | | \ \hat { p } _ { \theta } \rangle \big ) \stackrel { \cdot } { = } \mathbb { E } _ { y \sim p _ { \theta } } [ f ( p _ { T } ( y ) / p _ { \theta } ( y ) ) ]$ where $f$ is convex with ${ \overset { \smile } { f } } ( 1 ) = 0$ . By continuously modulating the convexity parameter, researchers can transition smoothly between mode-seeking and mode-covering behavior, tailoring distillation to downstream requirements. 

Why adaptive divergences win. The success of ToDi, AKL, and Entropy-Aware OPD has a geometric explanation. At reasoning-critical tokens, the teacher’s distribution is typically unimodal (one correct logical step), and Reverse KL aligns well. At generation-flexible tokens (choosing between “therefore” and “thus”), the distribution is near-uniform over synonyms, and Forward KL preserves this diversity. Adaptive methods implicitly detect this modality structure and switch accordingly. The practical implication is that no single divergence is optimal across all token positions. The field has moved decisively from “which divergence?” to “which divergence where?” 

Failure modes of sampled-token OPD. Fu et al. (2026) provide a systematic empirical analysis of why the dominant sampled-token variant of OPD is fragile in long-horizon settings. They identify three failure modes: (1) an imbalanced one-token signal that reduces distribution matching to a single sample, (2) unreliable teacher guidance on student-generated prefixes that deviate from the teacher’s training distribution, and (3) distortions caused by tokenizer or special-token mismatch between teacher and student. Theoretically, token-level OPD is biased relative to sequence-level Reverse KL but enjoys a tighter worst-case variance bound, revealing a fundamental bias-variance tradeoff. Their proposed fix, teacher top-K local support matching via truncated Reverse KL with top- $p$ rollout sampling and specialtoken masking, yields more stable optimization across math reasoning and multi-task agentic training. 

# 5 Black-Box and Self-Distillation Methods

While the white-box methods of Section 4 deliver dense distributional matching through full logit access, they are inapplicable when the teacher is a proprietary model (e.g., GPT-4, Claude 3) that exposes neither weights nor vocabulary-wide logits. Furthermore, as models approach the boundaries of human-curated data, they must refine their own policies through self-distillation without any external teacher. This section covers both regimes. 

# 5.1 Black-Box On-Policy Distillation

In black-box OPD, the student cannot compute $D _ { \mathrm { K L } } ( p _ { T } \parallel p _ { \theta } )$ because $p _ { T }$ is structurally inaccessible. Instead, the teacher serves exclusively as a scoring function or preference ranker over the student’s on-policy trajectories. Two primary signal construction strategies have emerged: adversarial feedback and preference-based feedback. 

Adversarial signal construction. GAD (Ye et al., 2025) casts black-box distillation as a two-player minimax game. A student generator $G$ produces on-policy responses, while a discriminator $D$ (initialized from the student with a scalar prediction head) distinguishes student outputs from teacher outputs using a Bradley–Terry preference model: 

$$
\max  _ {G} \min  _ {D} V (G, D) = \mathbb {E} _ {(x, y ^ {\mathcal {T}}) \sim \mathcal {T}} \left[ - \log \sigma \left(D (y ^ {\mathcal {T}}) - D (G (x))\right) \right] \tag {23}
$$

The generator maximizes V via REINFORCE, using the discriminator score as an evolving on-policy reward signal. Unlike white-box methods that match full distributions, GAD extracts knowledge through a scalar quality signal, trading distributional fidelity for API compatibility. It achieves consistent gains over SFT baselines, though it inherits the training instability of adversarial objectives, as the discriminator can overfit to stylistic cues rather than semantic quality. 

Lion (Jiang et al., 2023) implements a complementary three-stage adversarial loop: imitation (student fine-tuned on teacher responses), discrimination (teacher identifies instructions where the student falls short), and generation (teacher creates harder instructions targeting identified weaknesses). This closed loop progressively narrows the student–teacher gap by focusing compute on the student’s weakest capabilities, forming a natural difficultyincreasing curriculum. Unlike GAD, which uses a learned discriminator, Lion employs the teacher itself as both referee and curriculum designer, eliminating discriminator training at the cost of more teacher API calls. Lion-13B achieves competitive performance on BIG-Bench Hard and AGIEval using only 70k training examples. 

On-policy Verbal Distillation (OVD) (Xiong et al., 2026) replaces token-level probability matching with trajectory matching using discrete verbal scores (0–9) from teacher models. This eliminates the requirement for token-level alignment between student and teacher vocabularies, dramatically reduces memory consumption by avoiding teacher logit storage, and allows the student to freely explore the output space without being constrained to the teacher’s token-level distribution. OVD delivers up to $+ 1 2 . 9 \%$ absolute EM improvement on web QA and $+ 2 5 . 7 \%$ on math benchmarks (with a single rollout sample), demonstrating that coarse verbal feedback can outperform dense logit matching when combined with on-policy exploration. 

Preference-based signal construction. Moving beyond adversarial objectives, a second family of black-box methods constructs supervision through preference pairs rather than discriminator scores. ORPO-Distill (Singh et al., 2025) formulates cross-architecture distillation as a preference optimization task. Rather than relying on a single best teacher output, it transfers knowledge through diverse reasoning traces and contrasts teacher-generated (preferred) and student-generated (dispreferred) traces via an Odds-Ratio Preference Optimization objective. The mixed-policy strategy, combining on-policy student traces with off-policy teacher traces, outperforms both pure off-policy and pure on-policy alternatives. Unlike DPO-based approaches that require external scoring, ORPO-Distill constructs preference pairs directly from teacher–student trace quality differences. 

When high-quality expert solutions exist but are out-of-distribution for the student (e.g., didactic textbook proofs with implicit reasoning gaps), DAIL (Mendes et al., 2026) bridges the gap through a two-step process: first transforming expert solutions into detailed, indistribution reasoning traces, then applying a contrastive objective to focus learning on expert methodologies. Using fewer than 1,000 expert solutions, DAIL achieves $1 0 { - } 2 5 \%$ pass@k gains and $2 { - } 4 \times$ reasoning efficiency improvements, demonstrating that distribution alignment is critical when imitating expert-level solutions. 

# 5.2 Self-Play and Self-Distillation

When external teachers are unavailable or computationally prohibitive, models must generate their own distillation targets. Self-distillation represents the natural endpoint of on-policy learning, where the model continuously bootstraps its own capabilities. We organize the growing literature around four conceptual themes. 

# 5.2.1 Self-Play and Its Saturation Ceiling

SPIN (Chen et al., 2024) formulates self-distillation as a two-player game: at iteration $t ,$ the updated model $p _ { \theta _ { t + 1 } }$ is trained to distinguish the previous iteration’s generations from human-written responses: 

$$
\mathcal {L} _ {\mathrm {S P I N}} = \mathbb {E} _ {x, y \sim p _ {\mathrm {d a t a}}, y ^ {\prime} \sim p _ {\theta_ {t}}} \left[ \ell \left(\lambda \left(\log \frac {p _ {\theta_ {t + 1}} (y | x)}{p _ {\theta_ {t}} (y | x)} - \log \frac {p _ {\theta_ {t + 1}} \left(y ^ {\prime} \mid x\right)}{p _ {\theta_ {t}} \left(y ^ {\prime} \mid x\right)}\right)\right) \right] \tag {24}
$$

SPIN provides a theoretical convergence guarantee: the global optimum is achieved when $p _ { \theta } = p _ { \mathrm { d a t a } }$ . Starting from Zephyr-7B-SFT, it improves MT-Bench from 5.94 to 6.78 over 3 iterations, with diminishing returns at each round. Critically, because SPIN uses no external teacher or verifier, it cannot improve beyond the quality ceiling of the SFT data. The rapid saturation of self-play is analyzed in Section 5.3. 

# 5.2.2 Privileged Information: Breaking the Self-Play Ceiling

The key insight enabling self-distillation beyond the data ceiling is privileged information (PI), i.e., any signal available during training but absent at test time. Rather than pitting the model against itself in a zero-sum game, PI-based methods let a single model serve dual roles: as teacher (conditioned on PI) and as student (operating without it). This asymmetry injects fresh information into the loop, breaking the saturation ceiling that constrains pure self-play. 

On-Policy Self-Distillation (OPSD) (Zhao et al., 2026) instantiates this principle for mathematical reasoning. The teacher policy conditions on both the problem $x$ and the ground-truth answer $y ^ { \star }$ , while the student conditions only on x. The student generates on-policy rollouts, and the teacher provides dense token-level supervision on these student-generated sequences: 

$$
\mathcal {L} _ {\mathrm {O P S D}} (\theta) = \mathbb {E} _ {(x, y ^ {\star}) \sim \mathcal {S}} \mathbb {E} _ {\hat {y} \sim p _ {\theta} (\cdot | x)} \left[ \sum_ {n = 1} ^ {| \hat {y} |} D \left(p _ {T} (\cdot | x, y ^ {\star}, \hat {y} _ {<   n}) \parallel p _ {\theta} (\cdot | x, \hat {y} _ {<   n})\right) \right] \tag {25}
$$

On competition-level math benchmarks (AIME 2024/2025, HMMT), OPSD matches or exceeds Group Relative Policy Optimization (GRPO) at 4B and 8B scales while using an order of magnitude fewer generated tokens, though at 1.7B scale OPSD underperforms GRPO, indicating that sufficient model capacity is necessary. 

GATES (Stein et al., 2026) extends privileged self-distillation to settings without groundtruth labels or external verifiers. A single model acts as tutor (conditioned on a source document) and student (answering from the question alone). The core innovation is a consensus-based gating mechanism: for each question, $k$ tutor responses are sampled, and distillation proceeds only when the tutor demonstrates high agreement, suppressing the gradient when tutor uncertainty is high. This directly addresses the “echo chamber” problem where distilling uncertain teacher signals amplifies noise. 

Penaloza et al. (2026) generalize the OPSD and GATES paradigms into a unified framework based on privileged information, formalizing the conditions under which privileged self-distillation provably improves over standard training. The PI must reduce teacher entropy sufficiently to provide meaningful supervision, but not so much that the teacher’s distribution becomes degenerate. Empirically, this framework proves particularly effective in hard, long-horizon RL settings where the base model achieves near-zero solve rates without PI. 

On-Policy Context Distillation (OPCD) (Ye et al., 2026b) applies the privileged information paradigm to practical deployment concerns: the student generates rollouts from its own policy while a context-conditioned teacher (augmented with system prompts, exemplars, or retrieval context) provides supervision via Reverse $\mathrm { K L , }$ enabling the model to internalize knowledge that would otherwise require verbose prompting at inference time. Its successor, Online Experiential Learning (OEL) (Ye et al., 2026a), extends this to continuous post-deployment learning, where the model accumulates experience from interactive environments (e.g., text-based games) and periodically distills this experience back into its weights through on-policy context distillation, forming an online learning loop. 

# 5.2.3 Reasoning Compression

Distilled reasoning models often produce unnecessarily verbose chains of thought, increasing inference cost without improving accuracy. OPSDC (Sang et al., 2026) applies on-policy self-distillation where the same model serves as both teacher (prompted to “be concise”) and student (generating unconstrained rollouts), with per-token Reverse KL minimized on the student’s own trajectories. On MATH-500, OPSDC reduces chain-of-thought token count by $5 7 - 5 9 \%$ while simultaneously improving accuracy by 9–16 percentage points, demonstrating that verbosity in distilled reasoning models is a trainable inefficiency. 

Multi-Token Prediction (MTP) via self-distillation (Kirchenbauer et al., 2026) converts a pre-trained autoregressive model into a fast multi-token prediction model using an online self-distillation objective, achieving more than $3 \times$ faster decoding on GSM8K at less than $5 \%$ accuracy drop without architectural changes. 

# 5.2.4 Reward-Free Alignment via Self-Distillation

A recurring theme in recent work (Hubotter et al. ¨ , 2026; Khan et al., 2026; Ding, 2026; Shenfeld et al., 2026) is that self-distillation can replicate the benefits of RL (exploration, credit assignment, forgetting resistance) without requiring explicit reward models or verifiers. The key insight is that structured feedback from the environment or from the model’s own historical behavior can substitute for scalar rewards. 

SDPO (Hubotter et al. ¨ , 2026) exemplifies this by replacing scalar rewards with rich textual feedback (compiler errors, test outputs, proof checker messages). The model generates onpolicy rollouts, receives structured feedback, and constructs fine-grained credit assignment, attributing success or failure to specific reasoning steps rather than entire trajectories. This addresses the fundamental credit-assignment bottleneck in standard RLVR, improving both sample efficiency and final accuracy across scientific reasoning, tool use, and competitive programming. 

While SDPO targets credit assignment, TMS (Khan et al., 2026) addresses a different RL benefit: retention. Standard SFT suffers from supervision mismatch: the model’s evolving policy diverges from static training labels, causing catastrophic forgetting. TMS creates a dynamic curriculum from the model’s own historical checkpoints, constructing near-policy supervision targets that minimize Policy–Label Divergence, bridging the gap to RL without requiring reward models. 

A complementary challenge arises when standard RL itself fails. HDPO (Ding, 2026) targets “cliff” prompts where all rollouts fail and the policy gradient vanishes entirely. HDPO augments standard GRPO with privileged self-distillation targeting these zero-gradient prompts: the model as teacher receives ground-truth information and generates privileged rollouts, which are distilled into the student via JSD (provably implementing rejection sampling from the KL-regularized RL optimal policy). Experiments on Qwen2.5-Math-1.5B-Inst show consistent improvements in coverage metrics. 

Finally, Self-Distillation Fine-Tuning (SDFT) (Shenfeld et al., 2026) unifies these threads by positioning on-policy self-distillation as a general-purpose mechanism for continual learning: the model learns from expert demonstrations while regularizing against its own previous distribution via implicit KL regularization, achieving forgetting resistance comparable to RL-based continual learning without requiring reward functions. 

# 5.3 Why Self-Play Saturates and Breakthrough Strategies

A critical theoretical bottleneck in self-distillation is the rapid onset of policy saturation. After a few iterations of SPIN or OPSD, empirical observations universally note a plateau or severe degradation. This phenomenon is closely related to the “Ouroboros” problem (a model eating its own tail) and represents a distinct failure mode from exposure bias: while exposure bias arises from train–test distribution mismatch in off-policy settings, saturation arises from the absence of distributional diversity in fully on-policy self-play. 

Mathematically, this connects to mode collapse in GANs. Because the student optimizes against a target sharing its own inductive biases and architectural limitations, the hypothesis space progressively collapses. If the model discovers a syntactic “hack” or a highly confident but flawed reasoning heuristic, no external signal penalizes it. The model self-reinforces this flawed trajectory, driving $p _ { \theta } \big ( y _ { f l a w e d } | x \big )  1$ . Once entirely certain of its own hallucinations, gradients vanish, exploration ceases, and the policy is trapped. 

Strategies to break through saturation. Two primary approaches have emerged. First, external verifiers and symbolic grounding: as demonstrated in OPSD, self-play must be anchored by non-differentiable truth metrics (code execution engines, symbolic math verifiers). Even if the model becomes highly confident in a flawed derivation, the verifier assigns $R = 0$ , shattering the self-reinforcing echo chamber. Second, the distillation–RL loop: to maintain trajectory diversity, state-of-the-art frameworks interleave self-distillation with Proximal Policy Optimization (PPO)-based RL: 

$$
\max  _ {\theta} \mathbb {E} _ {y \sim p _ {\theta}} [ R (y) - \beta D _ {\mathrm {K L}} (p _ {\theta} \| p _ {\text {r e f}}) ] \tag {26}
$$

where $R ( y )$ is provided by an independently trained reward model. Periodically refreshing the reward model using human-annotated preference data over the student’s newest generations prevents premature convergence, ensuring that the self-play dynamic remains a continuously moving target. 

A complementary failure mode is identified by Kim et al. (2026a), who trace self-distillation degradation in mathematical reasoning to the suppression of epistemic verbalization, the model’s expression of uncertainty during reasoning. When self-distillation shortens reasoning traces (a common desirable outcome), it disproportionately removes hedging phrases and uncertainty markers that serve as implicit “attention flags” for difficult sub-problems. The result is shorter but less calibrated reasoning, where the model confidently commits to flawed steps it would have reconsidered in longer traces. 

# 6 Reasoning Distillation

Complex reasoning remains the most active frontier of LLM research. Here, OPD serves not merely as a compression technique but as a fundamental mechanism for structured knowledge transfer. By shifting the training distribution from the teacher’s static dataset to the student’s active exploration space, OPD has become the de facto standard for instilling chain-of-thought (CoT) and multi-step reasoning into smaller models. The token-level and sequence-level objectives analyzed in Section 4 provide the mathematical foundations, and this section examines how those objectives interact with the unique challenges of reasoning transfer, where errors compound across long derivation chains. Following standard reasoning distillation notation, we use $P _ { \theta }$ and $P _ { T }$ uniformly in this section for both trajectory-level sampling and token-level conditionals, since reasoning methods typically operate over the full rollout. 

# 6.1 Chain-of-Thought Distillation

Traditional off-policy distillation forces the student to mimic the teacher’s reasoning traces over a fixed dataset, yet reasoning is highly path-dependent. As Distilling Step-by-Step (Hsieh et al., 2023) established, learning from LLM-extracted rationales enables small models to outperform their un-fine-tuned large counterparts. 

Recent CoT distillation work underscores the need to align reasoning paths with the student’s own linguistic manifold. When the student generates its own intermediate steps, the teacher shifts from a static dictator of the output to an active evaluator of the student’s intrinsic logic. Formally, let $x$ be the input prompt, y be the final answer, and $\boldsymbol { r } = \left( r _ { 1 } , r _ { 2 } , \ldots , r _ { T } \right)$ be the sequence of intermediate reasoning tokens. The student model $P _ { \theta }$ generates rollouts $r ^ { S } \sim P _ { \theta } ( \cdot | x )$ . The objective function in on-policy CoT distillation minimizes the Reverse KL divergence at each generative step 

$$
\mathcal {L} _ {\mathrm {C o T - O P D}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, r ^ {S} \sim P _ {\theta} (\cdot | x)} \left[ \sum_ {t = 1} ^ {| r ^ {S} |} D _ {\mathrm {K L}} \left(P _ {\theta} (\cdot | x, r _ {<   t} ^ {S}) \| P _ {T} (\cdot | x, r _ {<   t} ^ {S})\right) \right] \tag {27}
$$

Unlike off-policy Forward-KL, which is mode-covering and forces the student to cover all modes of the teacher’s reasoning (often leading to probability mass in implausible regions), the Reverse-KL in Equation 27 is mode-seeking. It ensures the student reinforces reasoning paths that are highly probable under its own parameterization, provided they are validated by the teacher. 

This formulation is further augmented by frameworks like SuperCorrect (Yang et al., 2025b), which introduces a two-stage framework for reasoning distillation. In the first stage, hierarchical thought templates (both high-level strategies and detailed step-level reasoning patterns) are extracted from the teacher model to guide the student in generating more fine-grained reasoning. In the second stage, cross-model collaborative DPO enhances the student’s self-correction abilities. Specifically, the teacher provides correction traces for the student’s erroneous reasoning steps during training, and the cross-model DPO objective teaches the student to locate and resolve errors with teacher-guided insights. 

$$
\mathcal {L} _ {\text {S u p e r C o r r e c t}} (\theta) = \mathcal {L} _ {\text {t e m p l a t e}} (\theta) + \lambda \cdot \mathcal {L} _ {\text {c r o s s - D P O}} (\theta) \tag {28}
$$

where $\mathcal { L } _ { \mathrm { t e m p l a t e } }$ distills the teacher’s thought templates into the student’s reasoning process, and Lcross-DPO optimizes the student’s preference between correct and erroneous reasoning paths using the teacher’s correction traces. The SuperCorrect-7B model surpasses DeepSeekMath-7B by $7 . 8 \% / 5 . 3 \%$ and Qwen2.5-Math-7B by $1 5 . 1 \% / 6 . 3 \%$ on MATH/GSM8K respectively. 

# 6.2 Reward-Guided On-Policy Distillation

The integration of RL into the distillation pipeline bridges the gap between imitation learning and outcome-driven optimization. Two technical routes converge here. Outcome-based rewards (sparse but flexible) and logit-based signals (dense but requiring white-box access). The objective transitions from pure distribution matching to expected reward maximization subject to a distillation constraint. 

Frameworks like RLKD (Xu et al., 2025b) and AlignDistil (Zhang et al., 2025a) unify alignment and distillation through reward-guided learning. RLKD proposes a Generative Structure Reward Model (GSRM) that converts reasoning paths into meta-reasoning-solving steps and computes rewards measuring structural alignment between student and teacher reasoning. Rather than merely mimicking the teacher’s flat token sequence through SFT, the RL-based framework lets the student internalize the teacher’s implicit multi-branch reasoning structure, surpassing standard SFT-RL pipelines even on $0 . 1 \%$ of data under an RL-only regime. 

$$
\max  _ {\theta} \mathcal {J} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim P _ {\theta} (\cdot | x)} \left[ R _ {T} (x, y) - \beta D _ {\mathrm {K L}} \left(P _ {\theta} (\cdot | x) \| P _ {T} (\cdot | x)\right) \right] \tag {29}
$$

In reward-weighted distillation frameworks, sparse outcome rewards are used to weight the dense token-level distillation loss. In practice, the KL term in Equation 29 is computed as a sum of per-token KL divergences $\begin{array} { r } { \sum _ { t } \dot { D } _ { \mathrm { K L } } \big ( P _ { \theta } \big ( \cdot | y _ { < t } \big ) \ \| \ P _ { T } \big ( \cdot | y _ { < t } \big ) \big ) } \end{array}$ , enabling analytic gradient computation. The gradient of the objective with respect to the student parameters $\theta$ relies on the REINFORCE estimator modified by a dense distillation baseline 

$$
\nabla_ {\theta} \mathcal {J} (\theta) \approx \mathbb {E} _ {y \sim P _ {\theta}} \left[ \sum_ {t = 1} ^ {| y |} \nabla_ {\theta} \log P _ {\theta} \left(y _ {t} \mid x, y _ {<   t}\right) \cdot \hat {A} _ {t} (x, y) \right] - \beta \nabla_ {\theta} D _ {\mathrm {K L}} \left(P _ {\theta} \| P _ {T}\right) \tag {30}
$$

where $\hat { A } _ { t }$ is the advantage estimated via the teacher’s value function. This mathematical synergy lets the student extrapolate beyond the teacher’s capabilities, an effect documented as Reward Extrapolation (Yang et al., 2026b). Extrapolation occurs because the student explores novel, structurally sound reasoning paths in regions of the state space $\boldsymbol { S } _ { \mathrm { n o v e l } }$ where the teacher’s generation probability $P _ { T } ( y | x )$ may be low, yet the teacher’s outcome verification reward $\smile$ remains highly positive. By optimizing against this joint signal, the student escapes the performance ceiling of strict behavioral cloning, discovering algorithmic shortcuts that the teacher’s default decoding strategy would never reach, and mitigating the saturation typical of pure self-play RL. 

A complementary approach is LUFFY (Yan et al., 2025), which directly tackles a fundamental limitation of on-policy RL for reasoning, namely that the student learns only from its own rollouts and thus remains bounded by its initial capabilities. LUFFY extends GRPO to a Mixed-Policy objective incorporating off-policy reasoning traces (e.g., from DeepSeek-R1) alongside the student’s on-policy rollouts for joint advantage computation. To prevent superficial imitation of high-probability tokens in off-policy traces while ignoring lowprobability but critical reasoning steps, LUFFY introduces policy shaping via regularized importance sampling, reweighting gradients to emphasize learning from unfamiliar yet effective actions. On six math benchmarks, LUFFY achieves an average $+ 6 . 4$ point gain over standard RLVR methods and critically succeeds in training weak foundation models (e.g., Llama-3.1-8B) where pure on-policy RL fails entirely. 

# 6.3 DeepSeek-R1: Off-Policy Reasoning Distillation

The release of DeepSeek-R1 (DeepSeek-AI et al., 2025) transformed the field of reasoning distillation, though its distillation methodology itself is strictly off-policy. DeepSeek-R1 demonstrated that massive RL (GRPO) applied directly to a large foundational model (671B MoE) produces highly structured, verifiable reasoning traces, leading to an emergent “Aha moment” where the model self-allocates computation to verify its logic. 

The distillation step is straightforward. Approximately 800,000 training samples (600K reasoning and 200K non-reasoning) are curated from DeepSeek-R1 offline, and smaller student models (Qwen-1.5B through Llama-70B) are fine-tuned on this static dataset via standard supervised fine-tuning (SFT). No on-policy student rollouts, no logit matching, no iterative teacher-student interaction. The student never generates its own responses during training. This is classical Sequence-Level Knowledge Distillation (Kim & Rush, 2016) applied at large scale. 

Why Off-Policy Distillation Works So Well Here The remarkable effectiveness of this off-policy approach demands explanation, as it seemingly contradicts the core thesis of this survey that on-policy training mitigates exposure bias. We identify three factors. 

First, data quality dominates data distribution. The 800K samples from DeepSeek-R1 contain extremely high-quality reasoning chains with step-by-step verification, self-correction, and structured deliberation. When the teacher data is this rich and diverse, the exposure bias problem is partially alleviated because the static dataset already covers a wide range of reasoning patterns and error-recovery strategies. 

Second, the reasoning traces are inherently self-correcting. Unlike typical off-policy data (e.g., teacher-generated summaries), R1’s long CoT outputs contain backtracking, “wait, let me reconsider” moments, and multi-path exploration within a single sequence. The student learns not just the correct answer but the meta-cognitive process of verifying and correcting intermediate steps, which partially compensates for the lack of on-policy exploration. 

Third, the student’s task is memorization of structure, not exploration of alternatives. For mathematical reasoning, there are relatively few valid solution paths compared to open-ended generation. The off-policy data covers the dominant valid paths, and the student’s primary challenge is faithfully reproducing these structured chains rather than discovering novel ones. 

The Case for On-Policy Methods on Top of R1 Distillation However, the off-policy ceiling is real. The DeepSeek-R1 paper explicitly notes that incorporating RL could further boost model performance but chose not to include it. Subsequent work has demonstrated that on-policy methods applied after off-policy R1-style distillation yield meaningful further gains. 

RLKD (Xu et al., 2025b), as introduced in Section 6.2, demonstrates that structural reward signals can markedly improve sample efficiency in reasoning distillation, surpassing standard SFT-RL pipelines even on $0 . 1 \%$ of data. Building on this direction, KDRL (Xu et al., 2025a) provides a unified framework that jointly optimizes KD and RL objectives during post-training. KDRL adds an on-policy KL regularizer between the student and teacher during RL training, preventing the student from drifting too far from the teacher’s policy while still benefiting from reward-driven exploration. This addresses a key failure mode of sequential pipelines (distill then RL, or RL then distill). The RL stage can “forget” distilled knowledge, while the KD stage can suppress novel reasoning patterns discovered through RL. RL-Aware Distillation (RLAD) (Zhang et al., 2026c) takes this unification further with a selective imitation strategy: the student follows the teacher’s guidance only when doing so improves the policy update, and ignores the teacher when its signal would degrade the student’s own trajectory quality. The key mechanism is Trust Region Ratio Distillation (TRRD), which replaces the standard KL regularizer with a PPO-style likelihood ratio objective. Instead of penalizing the absolute KL distance between student and teacher, TRRD clips the per-token probability ratio $p _ { \theta } ( y _ { t } ) / p _ { T } ( y _ { t } )$ within a trust region, ensuring stable updates that inherit the teacher’s strengths without being dragged toward its weaknesses. This selective approach outperforms both offline distillation and pure RL (GRPO) on reasoning benchmarks, and notably surpasses standard KL-based OPD methods that blindly minimize divergence at every token regardless of whether the teacher’s signal is beneficial. KEPO (Yang et al., 2026a) integrates distillation into preference optimization for vision-language models, using dense token-level teacher supervision to address the sparse reward problem in reasoning-oriented RL. By conditioning the preference optimization on teacher knowledge, KEPO stabilizes training on problems where the student initially has near-zero solve rates, a regime where standard RLVR fails due to exploration collapse. From a theoretical perspective, Li et al. (2025) provide a unified mathematical framework that subsumes both on-policy KD and RL as special cases of a composite objective combining trajectory-level KL divergence with task rewards. Their gradient decomposition reveals that the KD component provides a dense, analytically computable gradient for token-level imitation, while the RL component contributes a Monte Carlo policy gradient for reward maximization. This formalization explains why hybrid $_ \mathrm { K D + R L }$ methods consistently outperform either approach alone. The dense KD gradient stabilizes early training, while the RL gradient enables exploration beyond the teacher’s distribution. 

The Self-Distilled Reasoner (OPSD) (Zhao et al., 2026) and PACED (Xu et al., 2026) represent the next evolution, applying genuine on-policy distillation where the student generates its own reasoning rollouts and the teacher provides token-level supervision on those studentgenerated sequences. This addresses the residual exposure bias that off-policy R1 distillation cannot resolve, because the student’s errors during inference differ from those in the training data, and only on-policy methods can correct this distribution mismatch. 

An orthogonal but practically important direction is reasoning compression. As discussed in Section 5.2, OPSDC (Sang et al., 2026) applies on-policy self-distillation to compress verbose chains of thought, reducing token count by $5 7 - 5 9 \%$ on MATH-500 while improving accuracy. In the reasoning context, OPSDC’s results are particularly striking: on AIME 2024, the 14B variant gains $+ 1 \big 0$ points over the uncompressed baseline with $4 \%$ compression. This demonstrates that verbosity in distilled reasoning models is a trainable inefficiency rather than a necessary cost. 

Empirical Results: Off-Policy R1 Distillation The DeepSeek-R1 off-policy distillation results nonetheless provide a compelling performance baseline. Using DeepSeek-R1 (671B MoE) as the teacher, the distilled students achieve remarkable performance across model scales. On the AIME 2024 benchmark (pass@1), R1-Distill-Qwen-1.5B scores $2 8 . 9 \% ,$ already surpassing GPT-4o $( 9 . 3 \% )$ and Claude-3.5-Sonnet $( 1 6 . 0 \% )$ . R1-Distill-Qwen-7B reaches 

$5 5 . 5 \%$ , R1-Distill-Qwen-14B achieves $6 9 . 7 \%$ , R1-Distill-Qwen-32B reaches $7 2 . 6 \%$ , and R1- Distill-Llama-70B achieves $7 0 . 0 \%$ . On MATH-500, the 7B distilled model reaches $9 2 . 8 \%$ and the 32B model reaches $9 4 . 3 \%$ . 

Distillation vs. Direct RL A further essential finding is that off-policy distillation decisively outperforms direct RL for small-to-medium models. At the 32B scale, applying GRPO directly to Qwen2.5-32B-Base (yielding “Qwen2.5-32B-Zero”) achieves only $\bar { 4 } \bar { 7 } . 0 \%$ on AIME 2024 after over 10K RL steps, comparable to QwQ-32B-Preview $( 4 4 . 0 \% )$ . In contrast, R1-Distill-Qwen-32B reaches $7 2 . 6 \%$ , a gap of 25.6 percentage points. This dramatic difference highlights that even off-policy distillation from a sufficiently capable teacher is far more effective than direct RL for smaller models. The natural question then becomes, can on-policy distillation close this gap further, or push beyond the off-policy ceiling? This remains one of the most important open questions in the field. 

Post-R1 Evolution: Qwen3 and Beyond The DeepSeek-R1 paradigm has been rapidly adopted and extended. Qwen3 (Yang et al., 2025a) refines the pipeline by introducing multi-stage training that combines off-policy distillation with subsequent on-policy RL, where specialized reward models provide domain-specific feedback during student rollouts. Qwen3 also introduces a “thinking budget” mechanism that allows the student to dynamically allocate chain-of-thought length based on problem difficulty, preventing the student from generating unnecessarily verbose reasoning for simple problems, a failure mode observed in direct R1 distillation where students sometimes replicate the teacher’s extended deliberation even when a short derivation suffices. These developments suggest that the optimal reasoning distillation pipeline is not purely off-policy or on-policy, but a carefully staged hybrid. Off-policy SFT on high-quality teacher data to establish a strong foundation, followed by on-policy RL or OPD to close the remaining distribution gap. 

# 7 Industrial Systems and Scaling

As OPD moves from academic prototypes to industrial-scale deployments, engineering challenges around compute efficiency, scaling laws, and architectural mismatches take center stage. Sections 4–6 provide the algorithmic foundations, and this section examines how these algorithms adapt to trillion-token-scale training, where innovations in gradient computation, teacher querying, and latency hiding become essential. 

# 7.1 Large-Scale Deployment

Recent foundation models rely heavily on OPD for post-training, demonstrating its scalability. The Qwen3 technical report (Yang et al., 2025a) highlights industrial-grade OPD where a dynamic ensemble of teachers evaluates student generations in real time. In this setting, the target distribution aggregates $K$ experts. Let $w _ { k }$ weight the $k$ -th teacher $T _ { k } ,$ yielding the convex combination 

$$
\tilde {P} _ {\text {e n s e m b l e}} \left(y _ {t} \mid x, y _ {<   t}\right) = \sum_ {k = 1} ^ {K} w _ {k} (x) P _ {T _ {k}} \left(y _ {t} \mid x, y _ {<   t}\right), \quad \text {w h e r e} \sum w _ {k} (x) = 1 \tag {31}
$$

This smoothens the target distribution and prevents the student from overfitting to a single teacher’s idiosyncrasies. Gemma 2 (Gemma Team et al., 2024) similarly uses online KD to bridge the performance gap between its parameter tiers, embedding distillation directly into continuous pre-training. MiniPLM (Gu et al., 2025) extends KD to the pre-training stage itself via offline Difference Sampling, selecting instances where the discrepancy between the teacher’s and a small reference model’s log-probabilities is largest, up-sampling hard and diverse examples while filtering trivial or noisy data. This offline curation avoids on-policy teacher inference entirely, and the refined corpus can be reused across multiple students. MiniPLM improves student LMs (200M–1.2B) on 9 downstream tasks while reducing pre-training computation. 

Furthermore, the boundaries among the three feedback routes (logits, outcomes, self-play) blur at scale. MiMo-V2 (Xiaomi LLM-Core Team et al., 2026) combines distillation from 

domain-specialized teachers with RL-based training, demonstrating that these paradigms can be complementary in large-scale systems where different training stages benefit from different supervisory signals. 

# 7.2 Efficiency Innovations

The primary systemic bottleneck of OPD is the teacher forward pass on dynamically generated student tokens. If the student generates a sequence of length $N _ { ☉ }$ , a naive implementation requires $O ( N )$ autoregressive teacher passes or one $O ( N ^ { 2 } )$ attention forward pass, crippling training throughput. 

Speculative KD (Xu et al., 2025c) addresses this bottleneck by using the student’s generations as speculative drafts. During rollout, the student generates a block of $K$ tokens $( y _ { t + 1 } , \ldots , \dot { y _ { t + K } } )$ , and the teacher then verifies and scores these drafts in parallel. The expected speedup $E [ S ]$ depends on the token-level acceptance probability $\mathrm { m i n } ( 1 , P _ { T } / P _ { S } )$ 

$$
E [ S ] = \frac {\mathbb {E} \left[ K _ {\text {a c c e p t e d}} \right] + 1}{1 + c} \quad \text {w h e r e} \mathbb {E} \left[ K _ {\text {a c c e p t e d}} \right] = \mathbb {E} _ {y \sim P _ {S}} \left[ \sum_ {k = 1} ^ {K} \prod_ {i = 1} ^ {k} \min  \left(1, \frac {P _ {T} \left(y _ {t + i} \mid y _ {<   t + i}\right)}{P _ {S} \left(y _ {t + i} \mid y _ {<   t + i}\right)}\right) \right] \tag {32}
$$

where $c$ is the ratio of student-to-teacher forward-pass cost. By leveraging cooperation between student and teacher to generate training data on the fly, Speculative KD accelerates the on-policy loop while preserving alignment with the student’s inference-time distribution. A complementary line of work, DistillSpec (Zhou et al., 2023), inverts the direction. Instead of speeding up training, DistillSpec uses on-policy KD to improve the draft model for speculative decoding at inference. Training the draft model on its own generated sequences with the target model’s logits as supervision improves acceptance rates and yields $\mathrm { \hat { 1 0 } } { - } 4 5 \%$ inference speedup over standard speculative decoding. Together, Speculative KD and DistillSpec form two sides of the same coin: the former accelerates distillation training, the latter accelerates inference via better-aligned drafts. 

Furthermore, cross-tokenizer distillation (Boizard et al., 2025) has matured, enabling OPD between disparate model families (e.g., Llama to Mistral) by aligning token probability distributions across different vocabularies. Using optimal transport to match logit distributions between the student’s vocabulary $V _ { S }$ and the teacher’s $V _ { T } .$ , the Universal Logit Distillation (ULD) framework minimizes distributional distance in vocabulary space 

$$
\mathcal {L} _ {\text {C r o s s T o k}} = \inf  _ {\pi \in \Pi \left(P _ {S}, P _ {T}\right)} \mathbb {E} _ {\left(z _ {S}, z _ {T}\right) \sim \pi} \left[ \| W _ {S \rightarrow T} z _ {S} - z _ {T} \| _ {2} ^ {2} \right] \tag {33}
$$

From a structural perspective, Minitron (Muralidharan et al., 2024) combines importancebased structured pruning with knowledge distillation to produce compact students that inherit the teacher’s architecture. The method first estimates neuron and attention head importance via lightweight activation-based metrics on calibration data, performs oneshot structured removal of the least important components, and then applies KD-based retraining to recover accuracy. This pruning-then-distillation pipeline achieves up to $4 0 \times$ fewer training tokens compared to training from scratch, demonstrating that architectural inheritance from the teacher significantly reduces convergence cost. At the frontier of industrial OPD, Nemotron-Cascade 2 (Yang et al., 2026c) demonstrates multi-domain onpolicy distillation in a 30B Mixture-of-Experts model with only 3B activated parameters. Its post-training pipeline combines cascade RL with domain-specific OPD across mathematical reasoning, code generation, and agentic tasks, achieving Gold Medal-level performance on IMO, IOI, and ICPC World Finals. It is the second open-weight model to do so after DeepSeek-V3.2-Speciale (671B-A37B), with $2 0 \times$ fewer parameters. 

# 7.3 Compute-Quality Tradeoff Analysis

A neglected area is the cost-benefit analysis of OPD. System designers constantly face the dilemma of allocating compute between more tokens (off-policy) versus better supervision 

(on-policy). We formulate the expected compute cost (in FLOPs) for $N$ tokens as follows. 

$$
C _ {\text {o f f}} \approx N \times \left(F _ {\text {t e a c h e r}} + F _ {\text {s t u d e n t}} + B _ {\text {s t u d e n t}}\right) \tag {34}
$$

$$
C _ {\text {o n}} \approx N \times \left(G _ {\text {s t u d e n t}} + \lambda F _ {\text {t e a c h e r}} + F _ {\text {s t u d e n t}} + B _ {\text {s t u d e n t}}\right) \tag {35}
$$

where $F , B$ denote forward and backward FLOPs, G is the autoregressive generation cost (scaling quadratically with sequence length due to KV cache updates), and $\lambda \in ( 0 , 1 ]$ is the teacher supervision refresh rate (e.g., scoring every token vs. sequence ends). Because $G _ { \mathrm { s t u d e n t } } \gg F _ { \mathrm { s t u d e n t } } , C _ { \mathrm { o n } }$ $C _ { \mathrm { o n } }$ is typically several times ${ \mathrm { \bar { \cal C } } } _ { \mathrm { o f f } } ,$ with the exact ratio depending on the student-teacher size ratio and $\lambda$ . 

Let $\mathcal { U } ( \boldsymbol { \theta } , C )$ represent the student’s downstream utility given compute budget C. The constrained optimization is 

$$
\max  \mathcal {U} (\theta , C) \quad \text {s u b j e c t} C = \gamma C _ {\mathrm {o n}} + (1 - \gamma) C _ {\mathrm {o f f}} \leq C _ {\max } \tag {36}
$$

where $\gamma$ is the fraction processed on-policy. For generic knowledge transfer and syntax alignment, $\gamma \to 0$ (off-policy) suffices because the data manifold is dense. For reasoning, code generation, and complex mathematics, however, the output space is highly multimodal (many valid solution paths) and fragile, $\frac { \partial \mathcal { U } } { \partial C _ { \mathrm { o n } } }$ ∂Con vastly exceeds ∂U∂C , justifying the $\frac { \partial \mathcal { U } } { \partial C _ { \mathrm { o f f } } }$ compute multiplier. We recommend a hybrid curriculum. Begin with $C _ { \mathrm { o f f } }$ to warm up latent representations, then shift to $\gamma  1$ during the final $2 0 \%$ of training to refine reasoning trajectories. 

Concrete Cost Example To ground these formulas, consider distilling a 70B teacher into a 7B student on $8 \mathrm { \times H 1 \bar { 0 } 0 }$ GPUs. Off-policy over 1B tokens, the teacher generates the dataset offline ${ \sim } 2 0 0$ GPU-hours), the student trains for ${ \sim } 1 0 0$ GPU-hours, totaling ${ \sim } 3 0 0$ GPU-hours. On-policy over 1B tokens, each step requires (1) student generation ${ \sim } 3 \times$ forward-pass cost due to autoregressive decoding), (2) teacher scoring (one 70B forward pass), and (3) student backward pass, yielding ${ \sim } 1 , 2 \bar { 0 } 0 { \ - } 1 , 5 0 0$ GPU-hours, a $4 { - } 5 \times$ overhead. 

GPU Memory Overhead and Practical Solutions. Beyond FLOPs, the most severe constraint for white-box OPD is GPU memory. Holding a 70B teacher alongside a 7B student requires the teacher’s weights (∼140 GB in BF16), the student’s weights and optimizer states $( \sim 8 4 \mathrm { G B } )$ , and, critically, the KV cache for generation and the full-vocabulary logits tensor $( \left[ B , T , \left| V \right| \right] )$ for distillation. Peak memory can easily exceed an $8 \times 8 0$ GB H100 node. Practitioners alleviate this via (1) Teacher Quantization: serving the teacher in FP8 or INT4 for a $2 { - } 4 \times$ memory reduction at negligible precision loss. (2) Logit Offloading/Recomputation: immediately offloading logits to CPU RAM or retaining only the top- $k$ . (3) Aggressive Gradient Checkpointing: minimizing student activation memory during the backward pass. Combining these techniques is essential for viable white-box OPD on standard clusters. 

However, quality gains can be substantial. On instruction-following benchmarks, on-policy methods consistently outperform off-policy SFT by meaningful margins. For example, DistiLLM (Ko et al., 2024) achieves state-of-the-art ROUGE-L scores with notable training speedup over previous on-policy baselines. For reasoning tasks, the off-policy ceiling, i.e., the maximum achievable performance regardless of data volume, is fundamentally lower because static datasets cannot cover the combinatorial space of valid reasoning paths. This “ceiling gap” widens with task complexity, providing the strongest motivation for on-policy methods despite higher cost. 

Speculative KD (Xu et al., 2025c) partially closes the cost gap by using the student’s generations as speculative drafts that the teacher verifies in parallel, producing higher-quality on-policy training data without the full cost of independent teacher generation, as the teacher only needs to score and correct the student’s proposals rather than generating sequences from scratch. 

# 8 Open Problems and Future Directions

Despite the rapid progress surveyed in the preceding sections, spanning token-level divergence matching (Section 4), black-box and self-play formulations (Section 5), and reasoning-

specific pipelines (Section 6), several foundational questions remain unanswered and numerous practical bottlenecks persist. This section outlines the most critical open problems and promising research directions. 

Distillation Scaling Laws. Chinchilla scaling laws (Hoffmann et al., 2022) provide principled compute allocation for pre-training, and Busbridge et al. (2025) recently extended this framework to off-policy knowledge distillation, fitting parametric scaling curves that reveal optimal teacher size grows sub-linearly with compute budget. However, no equivalent scaling law exists for on-policy distillation, where the generation cost of student rollouts introduces an additional compute axis that fundamentally changes the optimization landscape. Practitioners currently rely on costly trial-and-error to determine how much on-policy generation budget to allocate relative to student and teacher size. A natural conjecture is that the distillation loss follows a joint power-law of the form 

$$
L \left(N _ {S}, N _ {T}, D _ {\text {o n}}\right) = E + \frac {A}{N _ {S} ^ {\alpha}} + \frac {B}{N _ {T} ^ {\beta}} + \frac {C}{D _ {\text {o n}} ^ {\gamma}} + f \left(N _ {S}, N _ {T}\right) \tag {37}
$$

where $E$ is the irreducible task entropy and $f ( N _ { S } , N _ { T } )$ models the capacity-gap interference between student and teacher. While no study has yet validated this functional form, emerging empirical evidence hints at its plausibility. 

DeepSeek-R1 (DeepSeek-AI et al., 2025) employs off-policy distillation, but its results illuminate how student capacity affects distillation quality: on AIME 2024, performance scales as $2 8 . 9 \%  5 5 . 5 \% \stackrel { \cdot } {  } 6 \dot { 9 } . 7 \%  7 2 . 6 \%$ for student sizes $1 . 5 8  7 \mathrm { B }  1 4 \mathrm { B }  3 2 \mathrm { B }$ . The steepest gain occurs between 1.5B and 7B $( 2 6 . 6 \%$ absolute), while the $1 4 \mathrm { B } \to 3 2 \mathrm { B }$ jump yields only $2 . { \overset { \sim } { 9 } } \%$ , suggesting that $\alpha$ in Equation 37 may be small (rapid diminishing returns with student scale) while the teacher capacity term $B / N _ { T } ^ { \beta }$ dominates, consistent with a 7B student achieving $9 2 . 8 \%$ on MATH-500 from a sufficiently capable 671B teacher. The Qwen3 experience (Yang et al., 2025a) further suggests that $f ( N _ { S } , N _ { T } )$ is non-trivial: distilling from multiple specialized teachers yields better scaling than from a single monolithic teacher of equivalent total parameters. Complementing these scaling observations, Cha & Cho (2025) show through controlled simulations that distillation induces a precision-recall trade-off modulated by a single entropy-controlling parameter, grounding why KD consistently improves generative quality when sample precision is prioritized over diversity. 

Future work must conduct controlled grid searches that independently vary $N _ { S } , N _ { T } ,$ and $D _ { \mathrm { o n } }$ to disentangle these effects, enabling practitioners to answer fundamental allocation questions such as “Given a fixed GPU budget, should one distill from a 70B teacher for 1B tokens or a 405B teacher for 200M tokens?” Establishing such scaling laws would transform OPD from an empirical art into a principled engineering discipline analogous to Chinchilla-guided pre-training. 

Teacher Calibration and Uncertainty-Aware Distillation. White-box OPD assumes that the teacher’s logit distribution provides a reliable, high-quality supervisory signal at every token position. In practice, this assumption frequently breaks down. Teacher models suffer from overconfidence, hallucination, and poor calibration (Jin et al., 2026), especially when evaluated on out-of-distribution prefixes generated by an exploring student. When the student produces a flawed reasoning step, the teacher may assign high probability to a hallucinated continuation, pulling the student into an error cascade, a severe “echo chamber” effect where minimizing KL against an uncertain teacher forces the student to confidently mimic noise. 

Several existing methods partially address this concern. Entropy-Aware OPD (Jin et al., 2026) gates the divergence choice based on teacher entropy, switching from Reverse KL to Forward KL when the teacher is uncertain. GATES (Stein et al., 2026) uses consensusbased gating to suppress the distillation signal when the tutor demonstrates low agreement across multiple samples. PACED (Xu et al., 2026) proves that the gradient signal-to-noise ratio vanishes when the teacher-student gap is too large, implicitly addressing calibration through curriculum design. However, none of these methods explicitly model the teacher’s epistemic versus aleatoric uncertainty. A promising direction is to decompose teacher 

uncertainty into its epistemic component (which should attenuate the distillation loss) and its aleatoric component (which reflects genuine task ambiguity and should be preserved in the student’s output distribution). Achieving this decomposition efficiently, without requiring expensive Bayesian inference over the teacher’s parameters, remains an important open challenge. Ensemble-based approximations (Stein et al., 2026) and dropout-based uncertainty estimation offer practical starting points, but principled integration with onpolicy objectives could yield substantially more robust distillation. 

Dynamic Curriculum Distillation. Uniformly sampling prompts during OPD ignores the student’s evolving capacity. Exposing a novice student to complex reasoning prompts yields gradient noise, while exposing an advanced student to trivial prompts wastes compute. PACED (Xu et al., 2026) demonstrates that a Beta-kernel weighting scheme, which assigns maximum weight to sequences near the student’s competence frontier, dramatically improves sample efficiency and is provably minimax-robust. AdaSwitch (Peng et al., 2025) introduces adaptive switching at the token level, selectively integrating teacher guidance only when the student’s divergence exceeds a context-aware threshold. Lion (Jiang et al., 2023) implements a teacher-driven curriculum that progressively generates harder instructions targeting the student’s identified weaknesses. 

Despite these advances, principled curriculum design for OPD remains largely ad hoc. The core challenge is that the optimal training distribution shifts continuously as the student improves, requiring dynamic adaptation rather than fixed schedules. A promising direction is to draw on the “zone of proximal development” concept formalized in PACED (Xu et al., 2026): at each training step, the student should train on prompts at the boundary of its current capabilities, where the solve rate is neither too high (wasting compute) nor too low (producing uninformative gradients). Scaling this principle to large prompt pools (automatically estimating prompt difficulty, tracking the student’s evolving competence boundary, and efficiently allocating compute across difficulty tiers) is ripe for exploration. The interaction between curriculum scheduling and divergence adaptation (e.g., using Forward KL early for broad coverage, then switching to Reverse KL for consolidation) further expands the design space. 

Latent Space Distillation for Vocabulary Mismatch. Most OPD objectives assume that teacher and student share the same vocabulary, an assumption increasingly violated in practice when distilling across model families (e.g., Llama into Qwen). Marginalizing over misaligned tokenizers is computationally prohibitive and destroys high-frequency semantic signals. DSKD (Zhang et al., 2025b) addresses this through dual-space projectors that map hidden states between teacher and student representation spaces, while crosstokenizer KD (Boizard et al., 2025) uses optimal transport to align logit distributions across vocabularies. Delta KD (Cao et al., 2025) preserves the distributional shift from pre-training to fine-tuning rather than matching absolute distributions. 

However, these methods operate at the output logit level, where the vocabulary bottleneck has already compressed the teacher’s internal representations. A more ambitious direction is to bypass the vocabulary layer entirely and distill in latent space, matching the geometric structure of the teacher’s hidden-state manifold rather than its token-level predictions. This would enable seamless cross-architecture OPD where teacher and student differ not only in vocabulary but also in hidden dimension, number of layers, and attention structure. Early steps in this direction include DSKD’s dual-space formulation (Zhang et al., 2025b), but extending latent distillation to on-policy settings (where the student’s rollouts produce hidden states that must be aligned with the teacher’s counterfactual representations) introduces additional challenges around representation drift during training. 

On-Policy Distillation for Autonomous Agents. Current OPD methods overwhelmingly target single-turn generation or linear chains of thought, yet LLMs are increasingly deployed as autonomous agents interacting with external environments such as APIs, terminals, databases, and web browsers. In agentic settings, the action space is vast, feedback is highly delayed, and token-level distillation fails to capture the long-horizon credit assignment required for multi-step planning. 

SCoRe (Lyu et al., 2025) provides an early realization of agent-level distillation, where the student generates multi-step trajectories (including tool calls and environment interactions) and the teacher corrects only the earliest error rather than providing a complete demonstration. OEL (Ye et al., 2026a) extends on-policy context distillation to continuous post-deployment learning in interactive environments. Privileged Information Distillation (Penaloza et al., 2026) demonstrates that self-distillation with training-time privileged information is particularly effective in hard, long-horizon RL settings. 

Despite these advances, several fundamental challenges remain open. First, environment nonstationarity: unlike text generation where the “environment” (the prompt) is fixed, agentic environments change in response to actions, making direct trajectory comparison between teacher and student meaningless unless the teacher provides counterfactual evaluations conditioned on the student’s actual state. Second, tool-use combinatorics: modern agent frameworks expose dozens of tools with structured argument schemas, suggesting that distillation may need to operate at the tool-call level rather than individual tokens. Third, safety-critical credit assignment: in coding agents, a single incorrect file write can corrupt a repository. In web-browsing agents, a misguided click can trigger irreversible actions. Future distillation frameworks must incorporate safety constraints that prevent the student from exploring catastrophically dangerous action sequences during on-policy generation, even when those sequences might be maximally informative for learning. Developing OPD methods that balance exploration with safety in agentic settings remains a largely uncharted territory. 

Multimodal On-Policy Distillation. Current OPD methods overwhelmingly target textonly LLMs, but extending on-policy distillation to multimodal models is an emerging and largely open direction. Vision-language models face a unique challenge: high-quality visual reasoning data is scarce, while text-based reasoning data is abundant and easily verifiable. VOLD (Bousselham et al., 2025) demonstrates that a text-only teacher can guide a VLM student through on-policy distillation, with its key finding that an SFT cold-start phase is essential for cross-modal distributional alignment. Video-OPD (Li et al., 2026) extends on-policy distillation to temporal video grounding, achieving state-of-the-art results while reducing training cost compared to GRPO-based alternatives. X-OPD (Cao et al., 2026) tackles the speech modality gap by distilling from a text-mode LLM to a speech-mode LLM on student-generated speech inputs, systematically aligning cross-modal capabilities. 

These initial results are encouraging, but several questions remain. How should the distillation objective account for information asymmetry between modalities (e.g., when the teacher processes text but the student processes images of the same content)? Can on-policy distillation transfer spatial reasoning, temporal grounding, and audio understanding simultaneously, or does each modality require specialized objectives? The interaction between modality-specific data scarcity and on-policy data generation, where the student explores its own multimodal output space, is particularly promising, as OPD’s self-correcting dynamics could partially compensate for the lack of large-scale annotated multimodal reasoning data. 

Closing the Distillation-RL Loop. Distillation and reinforcement learning are historically treated as separate, linearly staged processes, either distill then RL, or RL then distill. However, static distillation saturates when the student exhausts the teacher’s knowledge, and pure RL diverges when the reward signal is sparse. The true performance ceiling likely lies in closing the loop, alternating between KD-driven stabilization and RL-driven exploration. 

Empirical evidence supports this view. Chen et al. (2025) show that across Llama and Qwen families, RL-based post-training retains more pre-training capabilities than SFT, precisely because RL generates on-policy rollouts that maintain distributional proximity to the student’s prior, suggesting that the on-policy training distribution itself, not the reward signal per se, is the primary mechanism through which RL avoids forgetting. SDFT (Shenfeld et al., 2026) demonstrates that self-distillation can serve as a principled RL alternative for continual learning, achieving comparable forgetting resistance without requiring explicit reward functions. KDRL (Xu et al., 2025a) jointly optimizes KD and RL objectives during post-training, preventing the catastrophic forgetting that plagues sequential pipelines. 

The unified framework of Li et al. (2025) formally subsumes both KD and RL as special cases, with the dense KD gradient stabilizing early training and the RL gradient enabling exploration beyond the teacher. 

Despite these advances, principled methods for alternating between distillation and RL, including determining when to switch objectives, how to prevent mode collapse during RL phases, and how to ensure that RL-discovered capabilities are not overwritten during subsequent distillation phases, remain open challenges. Proving convergence bounds for such alternating optimization and characterizing its contraction properties is an important theoretical direction for recursive self-improvement in LLMs. 

Evaluation Methodology Beyond Benchmarks. Standard benchmarks (e.g., MMLU, GSM8K) suffer from data contamination and measure only static pattern matching, failing to capture the intrinsic robustness, out-of-distribution generalization, and hallucination rates of a distilled student relative to its teacher. Fu et al. (2026) systematically demonstrate that token-level OPD can appear successful on standard benchmarks while failing catastrophically on distribution-shifted prompts, highlighting the gap between benchmark performance and true generalization. 

Future evaluation methodology should move toward dynamic, adversarial testing that probes whether the student has learned the underlying causal mechanism of reasoning rather than superficial correlations. Promising directions include systematically testing on semantically equivalent but syntactically diverse prompts, measuring the student’s calibration under distribution shift, and assessing whether distilled models preserve the teacher’s uncertainty structure. Developing standardized evaluation suites that capture these dimensions, beyond single-number accuracy metrics, would provide more reliable guidance for method selection and training decisions. 

Practical Guidelines and Decision Framework. The sheer volume of OPD techniques surveyed in this paper, spanning off-policy baselines, token-level and sequence-level methods, reward-guided approaches, and self-distillation, can cause decision paralysis for engineering teams. Based on the comparative analysis across Sections 4–7, we distill several actionable guidelines. 

When to use off-policy distillation. For general instruction-following and chat, off-policy SFT on teacher-generated data remains the most cost-effective starting point. High-quality teacher data combined with preference optimization (Zhang et al., 2025a) can achieve competitive performance at a fraction of the on-policy compute cost. Start here and measure the gap before investing in on-policy infrastructure. 

When on-policy becomes necessary. The transition to on-policy distillation is justified when (1) the student exhibits clear exposure bias, performing well on teacher-style prompts but failing on user-generated ones with different phrasing, (2) the task involves multi-step reasoning where compounding errors degrade quality (Gudibande et al., 2023), or (3) the student needs to exceed the teacher in a specific domain via reward-guided exploration (Yang et al., 2026b). 

Choosing between logit-level and reward-level OPD. If the teacher is white-box and the student is small $\mathrm { ( < 7 B ) }$ , logit-level methods (GKD (Agarwal et al., 2024), DistiLLM (Ko et al., 2024)) provide the dense supervision small models need. For larger students $\mathrm { ( \ge 7 B ) }$ with outcome verifiers, reward-guided OPD (RLKD (Xu et al., 2025b), AlignDistil (Zhang et al., 2025a)) enables discovery of novel solution paths. The hybrid approach, combining off-policy SFT warm-up (as in DeepSeek-R1 (DeepSeek-AI et al., 2025)) with on-policy logitlevel or reward-guided fine-tuning, consistently yields the best results across the surveyed literature. 

The “distillation tax” budget rule. Based on the compute analysis in Section 7, a practical rule of thumb is to allocate ${ \bar { 6 } } 0 { - } 7 0 \%$ of the training budget to off-policy warm-up, $2 0 { - } 3 0 \%$ to on-policy logit distillation, and $1 0 \%$ to reward-guided refinement. This staged approach front-loads cheap, high-bandwidth learning and reserves expensive on-policy compute for the final quality push. 

# 9 Conclusion

This survey has systematically examined on-policy distillation for large language models, synthesizing representative works spanning foundational formulations, algorithmic innovations, and industrial deployments. 

Key Findings. Three central insights emerge. First, shifting from off-policy to on-policy training distributions, from $y \sim p _ { \mathrm { d a t a } }$ to $y \sim p _ { \theta }$ (Section 2), is a consistently impactful design choice, yielding meaningful accuracy gains across mathematical reasoning, code generation, and instruction following (Tables 1 and 2). The improvement stems from eliminating exposure bias: the student learns to recover from its own errors rather than memorizing idealized trajectories. Second, divergence selection is not one-size-fits-all but must be adapted to both task and token position. As Section 4.4 shows, Reverse KL excels at reasoning tasks where mode-seeking prevents hallucination, while Forward KL preserves the diversity needed for open-ended generation. Adaptive methods such as ToDi (Jung et al., 2025) and Entropy-Aware OPD (Jin et al., 2026) that dynamically select divergences represent the current frontier. Third, the boundary between distillation and reinforcement learning is rapidly dissolving. G-OPD (Yang et al., 2026b), RLAD (Zhang et al., 2026c), and the hybrid $\mathrm { R i + I L }$ framework of Li et al. (2025) show that the optimal training signal combines dense teacher supervision with sparse outcome rewards: KD provides stability while RL enables exploration beyond the teacher. 

Practical Takeaways. Our survey distills several actionable guidelines (Section 8). With white-box teacher access, token-level methods (GKD (Agarwal et al., 2024), DistiLLM (Ko et al., 2024)) offer the best quality-compute tradeoff for general instruction following. For reasoning-intensive tasks, sequence-level methods (MiniLLM (Gu et al., 2024)) or hybrid approaches (Fast OPD (Zhang et al., 2026a), PACED (Xu et al., 2026)) are preferable despite higher variance. With only API access, GAD (Ye et al., 2025) enables effective on-policy learning through adversarial formulations. Self-distillation methods (SPIN (Chen et al., 2024), OPSD (Zhao et al., 2026), SDPO (Hubotter et al. ¨ , 2026)) eliminate the need for a separate teacher, making them attractive under resource constraints. The compute overhead of on-policy generation, typically $3 { - } 8 \times$ that of off-policy training, can be substantially reduced via speculative decoding (Xu et al., 2025c) or prefix truncation (Zhang et al., 2026a). 

Looking Ahead. The most pressing open problems are (1) the absence of distillation scaling laws analogous to Chinchilla (Hoffmann et al., 2022), forcing practitioners to rely on costly trial-and-error for compute allocation, (2) the need for uncertainty-aware objectives that prevent the “echo chamber” effect when students explore out-of-distribution regions, and (3) the extension of OPD to multi-turn agentic settings where training involves entire interaction trajectories rather than single completions. As the field matures, we anticipate convergence toward unified $\mathrm { K D + R L }$ frameworks that treat distillation not as a separate training stage but as a continuous regularization mechanism throughout the model’s lifecycle, from pre-training through deployment. 

# References



Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. Proceedings of ICLR, 2024. URL https://arxiv.org/abs/2306.13649. 





Nicolas Boizard, Kevin El Haddad, Celine Hudelot, and Pierre Colombo. Towards Cross- ´ Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs. Transactions on Machine Learning Research, 2025. URL https://arxiv.org/abs/2402.12030. 





Walid Bousselham, Hilde Kuehne, and Cordelia Schmid. VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation. arXiv preprint arXiv:2510.23497, 2025. URL https://arxiv.org/abs/2510.23497. 





Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, and Russ Webb. Distillation Scaling Laws. In Proceedings of the International Conference on Machine Learning (ICML), 2025. URL https://arxiv.org/abs/2502.08606. 





Yuang Cai and Yuyu Yuan. $\mathcal { X }$ -KD: General Experiential Knowledge Distillation for Large Language Models. arXiv preprint arXiv:2602.12674, 2026. URL https://arxiv.org/abs/ 2602.12674. 





Di Cao, Dongjie Fu, Hai Yu, Siqi Zheng, Xu Tan, and Tao Jin. X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs. arXiv preprint arXiv:2603.24596, 2026. URL https://arxiv.org/abs/2603.24596. 





Yihan Cao, Yanbin Kang, Zhengming Xing, and Ruijie Jiang. Delta Knowledge Distillation for Large Language Models. arXiv preprint arXiv:2509.14526, 2025. URL https://arxiv. org/abs/2509.14526. 





Sungmin Cha and Kyunghyun Cho. Why Knowledge Distillation Works in Generative Models: A Minimal Working Explanation. Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv.org/abs/2505.13111. 





Howard Chen, Noam Razin, Karthik Narasimhan, and Danqi Chen. Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting. arXiv preprint arXiv:2510.18874, 2025. URL https://arxiv.org/abs/2510.18874. 





Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models. Proceedings of ICML, 2024. URL https://arxiv.org/abs/2401.01335. 





DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. Nature, 645:633–638, 2025. URL https://arxiv.org/abs/2501.12948. 





Ken Ding. HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation. arXiv preprint arXiv:2603.23871, 2026. URL https://arxiv.org/abs/2603.23871. 





Jiabin Fan, Guoqing Luo, Michael Bowling, and Lili Mou. KETCHUP: K-Step Return Estimation for Sequential Knowledge Distillation. arXiv preprint arXiv:2504.19024, 2025. URL https://arxiv.org/abs/2504.19024. 





Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes. arXiv preprint arXiv:2603.25562, 2026. URL https://arxiv.org/abs/2603.25562. 





Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Leonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram ´ e,´ et al. Gemma 2: Improving Open Language Models at a Practical Size. arXiv preprint arXiv:2408.00118, 2024. URL https://arxiv.org/abs/2408.00118. 





Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: On-Policy Distillation of Large Language Models. Proceedings of ICLR, 2024. URL https://arxiv.org/abs/2306.08543. 





Yuxian Gu, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. MiniPLM: Knowledge Distillation for Pre-Training Language Models. Proceedings of ICLR, 2025. URL https: //arxiv.org/abs/2410.17215. 





Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, and Dawn Song. The False Promise of Imitating Proprietary LLMs. arXiv preprint arXiv:2305.15717, 2023. URL https://arxiv.org/abs/2305.15717. 





Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 





Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training Compute-Optimal Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2203. 15556. 





Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. Findings of ACL, 2023. URL https://arxiv.org/abs/2305.02301. 





Haiduo Huang, Jiangcheng Song, Yadong Zhang, and Pengju Ren. SelecTKD: Selective Token-Weighted Knowledge Distillation for LLMs. arXiv preprint arXiv:2510.24021, 2025. URL https://arxiv.org/abs/2510.24021. 





Jonas Hubotter, Frederike L ¨ ubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel ¨ Marta, et al. Reinforcement Learning via Self-Distillation. arXiv preprint arXiv:2601.20802, 2026. URL https://arxiv.org/abs/2601.20802. 





Ijun Jang, Jewon Yeom, Juan Yeo, Hyunggu Lim, and Taesup Kim. Stable On-Policy Distillation through Adaptive Target Reformulation. arXiv preprint arXiv:2601.07155, 2026. URL https://arxiv.org/abs/2601.07155. 





Yuxin Jiang, Chunkit Chan, Mingyang Chen, and Wei Wang. Lion: Adversarial Distillation of Proprietary Large Language Models. Proceedings of EMNLP, 2023. URL https://arxiv. org/abs/2305.12870. 





Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei, Nathalie Baracaldo, and Kimin Lee. Entropy-Aware On-Policy Distillation of Language Models. arXiv preprint arXiv:2603.07079, 2026. URL https://arxiv.org/abs/2603.07079. 





Seongryong Jung, Suwan Yoon, DongGeon Kim, and Hwanhee Lee. ToDi: Token-wise Distillation via Fine-Grained Divergence Control. Proceedings of EMNLP, 2025. URL https://arxiv.org/abs/2505.16297. 





Rana Muhammad Shahroz Khan, Zijie Liu, Zhen Tan, Charles Fleming, and Tianlong Chen. TMS: Trajectory-Mixed Supervision for Reward-Free, On-Policy SFT. arXiv preprint arXiv:2602.03073, 2026. URL https://arxiv.org/abs/2602.03073. 





Gyeongman Kim, Doohyuk Jang, and Eunho Yang. PromptKD: Distilling Student-Friendly Knowledge for Generative Language Models via Prompt Tuning. Findings of EMNLP, 2024. URL https://arxiv.org/abs/2402.12842. 





Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs? arXiv preprint arXiv:2603.24472, 2026a. URL https: //arxiv.org/abs/2603.24472. 





Minsang Kim and Seung Jun Baek. Explain in Your Own Words: Improving Reasoning via Token-Selective Dual Knowledge Distillation. Proceedings of ICLR, 2026. URL https: //arxiv.org/abs/2603.13260. 





Yeongmin Kim, Donghyeok Shin, Mina Kang, Byeonghu Na, and Il-Chul Moon. Distillation of Large Language Models via Concrete Score Matching. Proceedings of ICLR, 2026b. URL https://arxiv.org/abs/2509.25837. 





Yoon Kim and Alexander M. Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1317–1327, 2016. 





John Kirchenbauer, Abhimanyu Hans, Brian Bartoldson, Micah Goldblum, Ashwinee Panda, and Tom Goldstein. Multi-Token Prediction via Self-Distillation. arXiv preprint arXiv:2602.06019, 2026. URL https://arxiv.org/abs/2602.06019. 





Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. DistiLLM: Towards Streamlined Distillation for Large Language Models. Proceedings of ICML, 2024. URL https://arxiv.org/abs/2402.03898. 





Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, and Se-Young Yun. DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs. Proceedings of ICML, 2025. URL https://arxiv.org/abs/2503.07067. 





Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, and Pashmina Cameron. Scaling Reasoning Efficiently via Relaxed On-Policy Distillation. arXiv preprint arXiv:2603.11137, 2026. URL https://arxiv.org/abs/2603.11137. 





Jiaze Li, Hao Yin, Haoran Xu, Boshen Xu, Wenhui Tan, Zewen He, et al. Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation. arXiv preprint arXiv:2602.02994, 2026. URL https://arxiv.org/ abs/2602.02994. 





Yingru Li, Ziniu Li, and Jiacai Liu. A Note on Hybrid Online Reinforcement and Imitation Learning for LLMs: Formulations and Algorithms. arXiv preprint arXiv:2512.23097, 2025. URL https://arxiv.org/abs/2512.23097. 





Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s Verify Step by Step. Proceedings of ICLR, 2024. URL https://arxiv.org/abs/2305.20050. 





Yuanjie Lyu, Chengyu Wang, Jun Huang, and Tong Xu. From Correction to Mastery: Reinforced Distillation of Large Language Model Agents. arXiv preprint arXiv:2509.14257, 2025. URL https://arxiv.org/abs/2509.14257. 





Ethan Mendes, Jungsoo Park, and Alan Ritter. Didactic to Constructive: Turning Expert Solutions into Learnable Reasoning. arXiv preprint arXiv:2602.02405, 2026. URL https: //arxiv.org/abs/2602.02405. 





Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. Compact Language Models via Pruning and Knowledge Distillation. arXiv preprint arXiv:2407.14679, 2024. URL https://arxiv.org/abs/2407.14679. 





Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, and Massimo Caccia. Privileged Information Distillation for Language Models. arXiv preprint arXiv:2602.04942, 2026. URL https://arxiv.org/abs/2602.04942. 





Jingyu Peng, Jiaxin Mao, Yong Jiang, Zheng Liu, Pengjie Ren, Zhumin Chen, Maarten de Rijke, and Xiangyu Zhao. AdaSwitch: Balancing Exploration and Guidance in Knowledge Distillation via Adaptive Switching. arXiv preprint arXiv:2510.07842, 2025. URL https://arxiv.org/abs/2510.07842. 





Stephane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. A reduction of imitation ´ learning and structured prediction to no-regret online learning. In Proceedings of the 14th International Conference on Artificial Intelligence and Statistics, pp. 627–635, 2011. 





Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang, and Jiachen Sun. On-Policy Self-Distillation for Reasoning Compression. arXiv preprint arXiv:2603.05433, 2026. URL https://arxiv.org/abs/2603.05433. 





Idan Shenfeld, Mehul Damani, Jonas Hubotter, and Pulkit Agrawal. Self-Distillation Enables ¨ Continual Learning. arXiv preprint arXiv:2601.19897, 2026. URL https://arxiv.org/abs/ 2601.19897. 





Makoto Shing, Kou Misaki, Han Bao, Sho Yokoi, and Takuya Akiba. TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models. Proceedings of ICLR, 2025. URL https://arxiv.org/abs/2501.16937. 





Aasheesh Singh, Vishal Vaddina, and Dagnachew Birru. ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation. NeurIPS Workshop on Efficient Reasoning, 2025. URL https://arxiv.org/abs/2509.25100. 





Alex Stein, Furong Huang, and Tom Goldstein. GATES: Self-Distillation under Privileged Context with Consensus Gating. arXiv preprint arXiv:2602.20574, 2026. URL https: //arxiv.org/abs/2602.20574. 





Yuqiao Wen, Zichao Li, Wenyu Du, and Lili Mou. f-Divergence Minimization for Sequence-Level Knowledge Distillation. Proceedings of ACL, 2023. URL https://arxiv.org/abs/ 2307.15190. 





Taiqiang Wu, Chaofan Tao, Jiahao Wang, Runming Yang, Zhe Zhao, and Ngai Wong. Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models. Proceedings of COLING, 2025. URL https://arxiv.org/abs/2404.02657. 





Xiaomi LLM-Core Team, Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, Gang Xie, Hailin Zhang, Hanglong Lv, et al. MiMo-V2-Flash Technical Report. arXiv preprint arXiv:2601.02780, 2026. URL https://arxiv.org/abs/2601.02780. 





Xurong Xie, Zhucun Xue, Jiafu Wu, Jian Li, Yabiao Wang, Xiaobin Hu, et al. LLM-Oriented Token-Adaptive Knowledge Distillation. arXiv preprint arXiv:2510.11615, 2025. URL https://arxiv.org/abs/2510.11615. 





Jing Xiong, Hui Shen, Shansan Gong, Yuxin Cheng, Jianghan Shen, Chaofan Tao, Haochen Tan, Haoli Bai, Lifeng Shang, and Ngai Wong. OVD: On-policy Verbal Distillation. arXiv preprint arXiv:2601.21968, 2026. URL https://arxiv.org/abs/2601.21968. 





Hongling Xu, Qi Zhu, Heyuan Deng, Jinpeng Li, Lu Hou, Yasheng Wang, et al. KDRL: Post-Training Reasoning LLMs via Unified Knowledge Distillation and Reinforcement Learning. arXiv preprint arXiv:2506.02208, 2025a. URL https://arxiv.org/abs/2506. 02208. 





Shicheng Xu, Liang Pang, Yunchang Zhu, Jia Gu, Zihao Wei, Jingcheng Deng, Feiyang Pan, Huawei Shen, and Xueqi Cheng. RLKD: Distilling LLMs’ Reasoning via Reinforcement Learning. arXiv preprint arXiv:2505.16142, 2025b. URL https://arxiv.org/abs/2505. 16142. 





Wenda Xu, Rujun Han, Zifeng Wang, Long T. Le, Dhruv Madeka, Lei Li, William Yang Wang, Rishabh Agarwal, Chen-Yu Lee, and Tomas Pfister. Speculative Knowledge Distillation: Bridging the Teacher-Student Gap Through Interleaved Sampling. Proceedings of ICLR, 2025c. URL https://arxiv.org/abs/2410.11325. 





Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, and Tianyi Zhou. A Survey on Knowledge Distillation of Large Language Models. arXiv preprint arXiv:2402.13116, 2024. URL https://arxiv.org/abs/2402.13116. 





Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, and Zhipeng Wang. PACED: Distillation and Self-Distillation at the Frontier of Student Competence. arXiv preprint arXiv:2603.11178, 2026. URL https://arxiv.org/abs/2603.11178. 





Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to Reason under Off-Policy Guidance. arXiv preprint arXiv:2504.14945, 2025. URL https://arxiv.org/abs/2504.14945. 





Shaotian Yan, Kaiyuan Liu, Chen Shen, Bing Wang, Sinan Fan, Jun Zhang, Yue Wu, Zheng Wang, and Jieping Ye. Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning. arXiv preprint arXiv:2601.09088, 2026. URL https://arxiv.org/abs/2601. 09088. 





An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, et al. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388, 2025a. URL https://arxiv.org/abs/2505.09388. 





Fan Yang, Rui Meng, Trudi Di $\mathrm { Q i } ,$ Ali Ezzati, and Yuxin Wen. KEPO: Knowledge-Enhanced Preference Optimization for Reinforcement Learning with Reasoning. arXiv preprint arXiv:2602.00400, 2026a. URL https://arxiv.org/abs/2602.00400. 





Ling Yang, Zhaochen Yu, Tianjun Zhang, Minkai Xu, Joseph E. Gonzalez, Bin Cui, and Shuicheng Yan. SuperCorrect: Advancing Small LLM Reasoning with Thought Template Distillation and Self-Correction. Proceedings of ICLR, 2025b. URL https://arxiv.org/ abs/2410.09008. 





Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation. arXiv preprint arXiv:2602.12125, 2026b. URL https://arxiv.org/abs/2602.12125. 





Zhuolin Yang, Zihan Liu, Yang Chen, Wenliang Dai, Boxin Wang, Sheng-Chieh Lin, Chankyu Lee, Yangyi Chen, Dongfu Jiang, Jiafan He, Renjie Pi, Grace Lam, Nayeon Lee, Alexander Bukharin, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation. arXiv preprint arXiv:2603.19220, 2026c. URL https://arxiv.org/abs/2603.19220. 





Tianzhu Ye, Li Dong, Zewen Chi, Xun Wu, Shaohan Huang, and Furu Wei. Black-Box On-Policy Distillation of Large Language Models. arXiv preprint arXiv:2511.10643, 2025. URL https://arxiv.org/abs/2511.10643. 





Tianzhu Ye, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, and Furu Wei. Online Experiential Learning for Language Models. arXiv preprint arXiv:2603.16856, 2026a. URL https://arxiv.org/abs/2603.16856. 





Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-Policy Context Distillation for Language Models. arXiv preprint arXiv:2602.12275, 2026b. URL https://arxiv.org/ abs/2602.12275. 





Dongxu Zhang, Zhichao Yang, Sepehr Janghorbani, Jun Han, Andrew Ressler, Qian Qian, Gregory D. Lyng, Sanjit Singh Batra, and Robert E. Tillman. Fast and Effective Onpolicy Distillation from Reasoning Prefixes. arXiv preprint arXiv:2602.15260, 2026a. URL https://arxiv.org/abs/2602.15260. 





Miaosen Zhang, Yishan Liu, Shuxia Lin, Xu Yang, Qi Dai, Chong Luo, Weihao Jiang, Peng Hou, Anxiang Zeng, Xin Geng, and Baining Guo. Towards On-Policy SFT: Distribution Discriminant Theory and its Applications in LLM Training. arXiv preprint arXiv:2602.12222, 2026b. URL https://arxiv.org/abs/2602.12222. 





Songming Zhang, Xue Zhang, Tong Zhang, Bojie Hu, Yufeng Chen, and Jinan Xu. AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation. Proceedings of ACL, 2025a. URL https://arxiv.org/abs/2503.02832. 





Xue Zhang, Songming Zhang, Yunlong Liang, Fandong Meng, Yufeng Chen, Jinan Xu, and Jie Zhou. A Dual-Space Framework for General Knowledge Distillation of Large Language Models. arXiv preprint arXiv:2504.11426, 2025b. URL https://arxiv.org/abs/2504.11426. 





Zhaoyang Zhang, Shuli Jiang, Yantao Shen, Yuting Zhang, Dhananjay Ram, Shuo Yang, et al. Reinforcement-aware Knowledge Distillation for LLM Reasoning. arXiv preprint arXiv:2602.22495, 2026c. URL https://arxiv.org/abs/2602.22495. 





Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. arXiv preprint arXiv:2601.18734, 2026. URL https://arxiv.org/abs/2601.18734. 





Qihuang Zhong, Liang Ding, Li Shen, Juhua Liu, Bo Du, and Dacheng Tao. Revisiting Knowledge Distillation for Autoregressive Language Models. Proceedings of ACL, 2024. URL https://arxiv.org/abs/2402.11890. 





Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-Franc¸ois Kagy, and Rishabh Agarwal. DistillSpec: Improving Speculative Decoding via Knowledge Distillation. arXiv preprint arXiv:2310.08461, 2023. URL https://arxiv.org/abs/2310.08461. 





Matthieu Zimmer, Xiaotong Ji, Tu Nguyen, and Haitham Bou Ammar. Rethinking Large Language Model Distillation: A Constrained Markov Decision Process Perspective. arXiv preprint arXiv:2509.22921, 2025. URL https://arxiv.org/abs/2509.22921. 

