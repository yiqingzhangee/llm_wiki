# **Systematic Reasoning About Relational Domains With Graph Neural Networks** 

**Irtaza Khalid and Steven Schockaert** School of Computer Science & Informatics Cardiff University, UK 

```
{khalidmi,schockaerts1}@cardiff.ac.uk
```

## **Abstract** 

Developing models that can learn to reason is a notoriously challenging problem. We focus on reasoning in relational domains, where the use of Graph Neural Networks (GNNs) seems like a natural choice. However, previous work on reasoning with GNNs has shown that such models tend to fail when presented with test examples that require longer inference chains than those seen during training. This suggests that GNNs lack the ability to generalize from training examples in a systematic way, which would fundamentally limit their reasoning abilities. A common solution is to instead rely on neuro-symbolic methods, which are capable of reasoning in a systematic way by design. Unfortunately, the scalability of such methods is often limited and they tend to rely on overly strong assumptions, e.g. that queries can be answered by inspecting a single relational path. In this paper, we revisit the idea of reasoning with GNNs, showing that systematic generalization is possible as long as the right inductive bias is provided. In particular, we argue that node embeddings should be treated as epistemic states and that GNN should be parameterised accordingly. We propose a simple GNN architecture which is based on this view and show that it is capable of achieving state-of-the-art results. We furthermore introduce a benchmark which requires models to aggregate evidence from multiple relational paths. We show that existing neuro-symbolic approaches fail on this benchmark, whereas our considered GNN model learns to reason accurately. 

## **1 Introduction** 

Learning to reason remains a key challenge for neural networks. When standard neural network architectures are trained on reasoning problems, they often perform well on similar problems, but fail to generalize to problems with different characteristics than the ones that were seen during training, e.g. problems that were sampled from a different distribution [43] or require longer inference chains [33]. This behaviour has been observed for different types of reasoning problems and different types of architectures, including Language Models [43, 39] and Graph Neural Networks [33]. A common solution is to rely on some kind of neuro-symbolic approach. For instance, Neural Theorem Provers (NTPs [29, 25]) simulate the backward chaining process of traditional logic programs, using a soft unification mechanism to make this process differentiable. Unfortunately, such approaches are highly inefficient. They also typically rely on strong assumptions about the required reasoning process. Irrespective of any specific implementation issues, there is a fundamental question that has largely remained unanswered: what is it about neuro-symbolic methods that makes them more successful for reasoning tasks? In this paper, we focus on the common problem of reasoning about binary relations. In such cases, the available knowledge can be represented as a labelled multi-graph (i.e. a knowledge graph) and the main reasoning task of interest is to infer whether a given relationship holds or not. Graph Neural Networks (GNNs) intuitively seem well-suited for this task, but in practice they 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Single-path reasoning Multipath disjunctive reasoning<br>0 NTPP 0 TPPI<br>Has mother Has brother PO<br>1 4 7<br>3 Has son 1<br>PO TPP EQ<br>Has daughter Has father 2 PO  5  8<br>4 2 Has uncle EQ EC PO<br>Has uncle<br>3 6 9<br>NTPPI<br>5 EQ<br>EQ<br>10<br>**----- End of picture text -----**<br>


Figure 1: Left: A (single) relational path reasoning problem over family relations. Right: A multipath reasoning problem over the RCC-8 relations where each path provides partial (disjunctive) information and the target label is obtained by combining information from paths _P_ 1, _P_ 2 and _P_ 3. 

underperform neuro-symbolic methods on examples that require systematic generalization. Crucially, this is true despite the fact that GNNs are expressive enough to encode the required inference process. 

We argue that the outperformance of neuro-symbolic methods can be explained by their focus on modelling _relational paths_ , as an alternative to local message passing. To predict the relationship between two entities _h_ and _t_ , conceptually, neuro-symbolic methods consider all the (exponentially many) paths between these entities, select the most informative path, and make a prediction based on this path (although, in practice, heuristics are used to avoid exhaustively exploring every path). The emphasis on paths provides a useful inductive bias, while also avoiding the problems with over-smoothing that can arise with local message passing. Compared to GNNs, neuro-symbolic methods also have an advantage in how individual relational paths are modelled. Consider a relational _r_ 1 _r_ 2 _r_ 3 _rk_ path _h −→ x_ 1 _−→ x_ 2 _−→ ... −→ t_ . When the specific choice of the intermediate entities _x_ 1 _, ..., xk−_ 1 is not important, we will write such a path simply as _r_ 1; _r_ 2; _..._ ; _rk_ . To predict the relationship between _h_ and _t_ based on this path, we need to repeatedly choose two neighbouring relations _ri_ ; _ri_ +1 in the path and replace them by their composition. Crucially, the order in which the neighbouring relations are chosen may determine whether the model is capable of finding the answer: certain orderings may require the model to capture intermediate relationships which it has not seen during training. Neurosymbolic methods can often avoid such issues by, in principle, considering all possible orderings. In contrast, GNN architectures such as NBFNet [46] can only process relational paths sequentially. Neuro-symbolic methods also have important drawbacks, however. Most fundamentally, their focus on relational paths fundamentally limits the kinds of reasoning problems that can be tackled. Since these limits are not tested by the most commonly used benchmarks, as a first contribution, we introduce a benchmark based on the RCC-8 calculus for qualitative spatial reasoning [27]. The considered problems require models to combine evidence from multiple relational paths. Accordingly, we find that existing path-focused neuro-symbolic methods fail for this benchmark. Figure 1 contrasts an example from our benchmark with a relational path based example, as described previously. 

As our main contribution, we revisit the question of whether GNNs can be used for systematic reasoning. The central idea is that reasoning involves the manipulation of epistemic states (i.e. states of knowledge). For instance, symbolic methods for reasoning about RCC-8 problems maintain sets of possible relationships between pairs of entities. Reasoning is then a matter of iteratively refining these sets. When designing a GNN model, we thus intuitively need to think of node embeddings as encoding sets of possible relationships. This view affects which pooling operations are meaningful (as analysed in [32]), how messages should be computed, and how node embeddings should be initialised. We find that when the GNN is designed with these considerations in mind, it can rival neuro-symbolic methods on reasoning problems that require selecting a single relational path, while dramatically outperforming such methods on the proposed RCC-8 benchmark. Surprisingly, this performance is achieved despite our model staying close to standard GNN architectures. Our code and datasets are available online[1] . 

## **2 Related work** 

There has been a large body of work on learning to reason with neural networks, ranging from traditional approaches based on carefully designed feedforward [35] or recurrent [12] models, to 

> 1 `github.com/erg0dic/gnn-sg` 

2 

more recent approaches based on GNNs [44, 46, 45], pre-trained language models [9, 19, 10] or tailor-made architectures for relational reasoning [30, 5]. Despite their apparent success in a variety of reasoning tasks, however, neural networks usually do not learn to reason in a _systematic_ way. For instance, [9] showed that a fine-tuned BERT [13] model can reason about (natural language descriptions of) propositional Horn rules with a remarkable accuracy. However, a follow-up analysis in [43] revealed that this performance largely comes from learning spurious statistical artefacts of the considered (randomly generated) problem instances. When slightly changing how test instances are generated, the performance was found to drop to random chance in some cases. As another example, large language models have shown impressive reasoning abilities, but this is often due to the presence of similar problem instances in their training data. When exposed to problems that require some form of generalisation of the underlying reasoning principles, their performance also collapses [14, 36]. 

We focus on the problem of relational reasoning in this paper. Specifically, given a labelled multigraph, where the nodes represent entities and edges specify relationships between these entities, we consider the task of inferring relationships that are not directly encoded. For instance, knowing that Bob is the father of Alice, and Alice is the mother of Eve, we may want to infer that Bob is the grandfather of Eve. Such problems have traditionally been studied in the field of Inductive Logic Programming (ILP) [26] and are also close to the widely studied problem of link prediction in knowledge graphs [6]. Intuitively, GNNs should be well-suited for solving such reasoning tasks. However, [33] found that GNNs trained to predict family relationships fail to generalise to problem instances that require longer inference chains than those that were seen during training. Similarly, [34] also found the logical generalisation abilities of GNNs to be limited. The apparent limitations of GNNs have inspired a range of neuro-symbolic methods for relational reasoning. For instance, Neural Theorem Provers (NTPs [29, 25]) take inspiration from the backward chaining mechanism from logic programming, but they represent relations using embeddings and rely on a soft unification mechanism. This allows them to learn differentiable approximations of rules from data, while retaining the ability to apply the learned rules in a systematic way. Essentially, these methods aim to find the most predictive relational path, and then compose the relations along that path. Because NTPs are computationally expensive, more recent methods aim to directly search for predictive relational paths, without attempting to simulate logic programming [22, 7]. Rather, these methods learn to reduce relational paths to a single relation by repeatedly composing adjacent relations in the path. 

Our proposed model is strongly inspired by the idea that the structure of a neural network should be aligned with the reasoning algorithm it is supposed to learn. This intuitive idea was formally studied within the setting of PAC-learning in [40]. Similar to our proposed model, Edge Transformers [5] also aim to align the model with the problem of relational reasoning. Their model essentially modifies the attention mechanism from the Transformer model [37] to allow it to capture relational compositions. 

## **3 Learning to reason in relational domains** 

We focus on the problem of reasoning about binary relations. We assume that a set _F_ of facts is given, referring to a set of relations _R_ and a set of entities _E_ . Each of these facts is an _atom_ of the form _r_ ( _a, b_ ), with _r ∈R_ and _a, b ∈E_ . We furthermore assume that there exists a set of rules _K_ which can be used to infer relationships between the entities in _E_ . We write _K ∪F |_ = _r_ ( _a, b_ ) to denote that _r_ ( _a, b_ ) can be inferred from the facts in _F_ and the rules in _K_ . The problem that we are interested in is to develop a neural network model _fθ_ which can predict for a given assertion _r_ ( _a, b_ ) whether _K ∪F |_ = _r_ ( _a, b_ ) holds or not. Note that the set of rules _K_ is not given. We instead have access to a number of fact sets _Fi_ , together with examples of atoms _r_ ( _a, b_ ) which can be inferred from these fact graphs and atoms which cannot. To be successful, _fθ_ must essentially learn the rules from _K_ , and the considered model must be capable of applying the learned rules in a systematic way to new problems. 

## **3.1 Learning to reason about simple path rules** 

The most commonly studied setting concerns rules of the following form ( _n ≥_ 3): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0003-07.png)


We will refer to such rules as _simple path rules_ . Note that we used the convention from logic programming to write the head of the rule on the left-hand side, and we use uppercase symbols such as _Xi_ to denote variables. We can naturally associate a labelled multi-graph _GF_ with the given set 

3 

of facts. The rule (1) expresses that when two entities _a_ and _b_ are connected by a relational path _r_ 1; _. . ._ ; _rn−_ 1 in this graph _GF_ , then we can infer that _r_ ( _a, b_ ) is true. Without loss of generality, we can restrict this setting to rules with two atoms in the body: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0004-01.png)


Indeed, a rule with more than two atoms in the body can be straightforwardly simulated by introducing fresh relation symbols. The semantics of entailment are defined in the usual way (see the appendix for details). We can think of the process of showing _K ∪F |_ = _r_ ( _a, b_ ) in terms of operations on relational paths. We say that the path _r_ 1; _. . ._ ; _ri−_ 2; _s_ ; _ri_ +1; _. . ._ ; _rk_ can be derived from _r_ 1; _. . ._ ; _rk_ in one step if _K_ contains a rule of the form _s_ ( _X, Z_ ) _← ri−_ 1( _X, Y_ ) _∧ ri_ ( _Y, Z_ ). We say that _r_ can be derived from _r_ 1; _. . ._ ; _rk_ if there exists a sequence of _k −_ 1 such steps that yields _r_ . We have the following result. 

**Proposition 1.** _We have that K ∪F |_ = _r_ ( _a, b_ ) _holds iff there exists a relational path r_ 1; _. . ._ ; _rk connecting a and b in the graph GF such that r can be derived from r_ 1; _. . ._ ; _rk._ 

Inferring _r_ ( _a, b_ ) thus conceptually consists of two distinct steps: (i) selecting a relational path between _a_ and _b_ and (ii) showing that _r_ can be derived from it. Several of the neural network methods that have been proposed in recent years for relational reasoning directly implement these two steps, most notably R5 [22] and NCRL [7]. Neural theorem provers [29] implicitly also operate in a similar way, considering all possible paths and all possible derivations for these paths. However, both steps are problematic for standard GNNs. First, by focusing on local message passing, rather than on selecting individual relational paths, the node representations that are learned by a GNN run the risk of becoming “overloaded”, as they intuitively aggregate information from all the paths that go through a given node. Moreover, even for graphs that consist of a single path, GNNs have a disadvantage: the use of local message passing intuitively means that relational paths have to be processed sequentially. For instance, given a path _h −→r_ 1 _x_ 1 _−→r_ 2 _x_ 2 _−→r_ 3 _t_ , after the first layer, the GNN might update the representation of _x_ 1 to capture the fact _r_ 1( _h, x_ 1). After the second layer, the representation of _x_ 2 might capture the relationship between _h_ and _x_ 2. This intuitively corresponds to applying a derivation step which combines _r_ 1; _r_ 2 into the representation of another relation _s_ . Then, after the third layer, the representation of _x_ 3 might capture the relationship between _h_ and _x_ 3, etc. 

The fact that relational paths can only be processed sequentially by GNNs is an important limitation, at least in theory. For instance, we may have a setting involving the following rules: _K_ = _s_ 1( _X, Z_ ) _← r_ 1( _X, Y_ ) _∧ r_ 2( _Y, Z_ ) _, s_ 2( _X, Z_ ) _← r_ 3( _X, Y_ ) _∧ r_ 4( _Y, Z_ ) _, r_ ( _X, Z_ ) _← s_ 1( _X, Y_ ) _∧ s_ 2( _Y, Z_ ) _}_ . Then we clearly have that _r_ can be derived from _r_ 1; _r_ 2; _r_ 3; _r_ 4 but we may not be able to show this by processing the relations in this path from left-to-right. Indeed, after the first step, we end up with the path _s_ 1; _r_ 3; _r_ 4 but the composition of _s_ 1 and _r_ 3 may not correspond to a well-defined relationship. In practice, the problem is exacerbated by the fact that some valid rules may not have been learned, as the situations in which they apply may not have been included in the training data. In fact, the model may not even have learned about the existence of some relations. For instance, when modelling family relationships, we may encounter the following relation chain: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0004-06.png)


Suppose the model has never encountered the _great-cousin_ relation during training. Then we cannot expect it to capture the relational path _father_ ; _father_ ; _brother_ ; _daughter_ . However, if it can first derive _father aunt a −−−→ x_ 1 _−−→ x_ 5 then the problem disappears (assuming it understands the _great-aunt_ relation). 

## **3.2 Learning to reason about disjunctive rules** 

The rules that we have considered thus far uniquely determine the relationship between two entities _a_ and _c_ , given knowledge about how _a_ relates to _b_ and _b_ relates to _c_ . In many settings, however, such knowledge might not be sufficient for completely characterising the relationship between _a_ and _c_ . Domain knowledge might then be expressed using disjunctive rules of the following form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0004-10.png)


In other words, if we know that _r_ 1( _a, b_ ) and _r_ 2( _b, c_ ) hold for some entities _a, b, c_ , then all we can infer is that one of the relations _s_ 1 _, . . . , sk_ must hold between _a_ and _c_ . We then typically also have constraints of the form _⊥← r_ 1( _X, Y_ ) _∧ r_ 2( _X, Y_ ), encoding that _r_ 1 and _r_ 2 are disjoint. Many of the calculi that have been developed for temporal and spatial reasoning fall under this setting [1, 27]. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0005-00.png)


Figure 2: Overview of the proposed framework. Step 1: the forward and backwards model independently learn entity embeddings through message passing. Step 2: we compose the entity embeddings from the forward and backward model for entities on a path between the head and tail entity. Each of these compositions acts as a prediction of the target relation. Step 3: we aggregate the evidence provided by each of these predictions. 

**Example 1.** _The RCC-8 calculus describes qualitative spatial relations between two regions using eight primitive relations (see Section A.5 for details). The semantics of the RCC-8 relations are governed by disjunctive rules such as:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0005-03.png)


_as well as constraints encoding that the RCC-8 relations are disjoint. Let K contain these rules and constraints, and let F_ = _{_ ec( _a, b_ ) _,_ ntpp( _b, c_ ) _,_ tppi( _a, d_ ) _,_ po( _d, c_ ) _}. Then we have K ∪F |_ = po( _a, c_ ) _. Indeed, using_ (4) _we can infer_ po( _a, c_ ) _∨_ tpp( _a, c_ ) _∨_ ntpp( _a, c_ ) _, while using_ (5) _we can infer_ po( _a, c_ ) _∨_ tppi( _a, c_ ) _∨_ ntppi( _a, c_ ) _. Using the disjointness constraints, we finally infer_ po( _a, c_ ) _._ 

As the example illustrates, in the setting with disjunctive rules, we may need to aggregate evidence from multiple relational paths. This means that methods such as R5 and NCRL, which are directly based on the idea of making predictions from a single path, cannot be used. Neural theorem provers cannot handle disjunctive rules either. We thus need a new approach for dealing with this setting. 

Proposition 1 provides a characterisation of entailment for simple rules, which clarifies how neural networks can be designed for deciding entailment. We may now wonder whether a similar characterisation is possible for entailment with disjunctive rules. Unfortunately, as deciding entailment with disjunctive rules is NP-complete, this is not possible with standard GNNs. While neuro-symbolic methods for this task could in principle be designed, such methods would be prohibitively slow. However, for RCC-8, and many other calculi for temporal and spatial reasoning, deciding entailment in polynomial time is possible using the so-called algebraic closure algorithm [28]. 

The main idea is to keep track, for every pair of entities, of which relationships are possible between these entities. This knowledge of possible relationships is then propagated to infer constraints about relationships between other pairs of entities using the rules in _K_ (details are provided in the appendix). 

## **4 A GNN for systematic reasoning** 

Section 3 highlighted some challenges with designing GNNs that can learn to reason. We now present a simple GNN model that aims to overcome these challenges. Specifically, we want a model which (i) is more efficient than current neuro-symbolic methods while matching their performance on reasoning with simple path rules and (ii) is also able to reason about disjunctive rules. 

We start from the principle that reasoning is fundamentally about manipulating epistemic states (i.e. states of knowledge) and the GNN should reflect this, i.e. there should be a clear correspondence between the node embeddings that are learned by the model and what we can infer about the relationships that may hold between the entities of interest. Inspired by NBFNet [46], we use a network which learns the relationships between one designated head entity _h_ and all other entities. Let us write **e**[(] _[l]_[)] _∈_ R _[n]_ for the embedding of entity _e_ in layer _l_ of the network. This embedding should intuitively reflect what we have derived about which relationships may possibly hold between _e_ and the designated entity _h_ . We think of **e**[(] _[l]_[)] as encoding a probability distribution over possible 

5 

relationships. Accordingly, the embeddings are initialised as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0006-01.png)


In other words, we associate the first coordinate with the identity relation. Since we know that _h_ is identical to itself, we define **h**[(0)] as (1 _,_ 0 _, . . . ,_ 0). For the other entities, we initially have no knowledge, so we initialise their embeddings as a uniform distribution. Note that the components of the embeddings correspond to abstract primitive relationships, rather than the relations from _R_ . We only require that the relations from _R_ can be defined in terms of these primitive relationships. This distinction is important, because it allows the model to capture semantic dependencies between the relations from _R_ (e.g. if we have both the _parent_ and _father_ relations in _R_ ) and because we may need to be able to express relationships which are outside _R_ when composing relations from _R_ . In other words, in general we should have _n ≥|R|_ . For _l ≥_ 1, the embeddings are updated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0006-03.png)


where the argument of the pooling operator _ψ_ is treated as a multi-set and **r** _∈_ R _[n]_ is a learned embedding of relation _r ∈R_ . We interpret **r** = ( _r_ 1 _, . . . , rn_ ) as a probability distribution over the primitive relations. Accordingly, we require that _r_ 1 _, . . . , rn ≥_ 0 and[�] _i[r][i]_[= 1][.] 

**Message passing** The vector _ϕ_ ( **f**[(] _[l][−]_[1)] _,_ **r** ) should capture what are the possible relationships between _h_ and _e_ , given the knowledge provided by **f**[(] _[l][−]_[1)] about the possible relationships between _h_ and _f_ and the fact _r_ ( _f, e_ ). Since both **f**[(] _[l][−]_[1)] and **r** are modelled as probability distributions over primitive relations, _ϕ_ ( **f**[(] _[l][−]_[1)] _,_ **r** ) can be naturally defined in terms of compositions of these primitive relations. Specifically, let the vector **a** _ij ∈_ R _[n]_ represent the composition of primitive relations _i_ and _j_ , which we treat as a probability distribution, i.e. we require the components of **a** _ij_ to be non-negative and to sum to 1. We define: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0006-06.png)


The initialisation of **h**[(0)] was based on the assumption that the first coordinate of the embeddings corresponds to the identity relation. Accordingly, we want to ensure that _ϕ_ ((1 _,_ 0 _, . . . ,_ 0) _,_ ( _r_ 1 _, . . . , rn_ )) = ( _r_ 1 _, . . . , rn_ ). We thus fix **a** 1 _j_ = one-hot( _j_ ), where we write one-hot( _j_ ) to denote the _n_ -dimensional vector which is 1 in the _j_[th] coordinate and 0 elsewhere. Note that the composition of two primitive relations is thus modelled as a probability distribution over primitive relations. This provides an important inductive bias, as it encodes the assumption that the relationship between any two entities can be described by one of the _n_ considered primitive relations. Rule based methods, including NTP based approaches, also encode this assumption. However, the message passing operations that are used by standard GNNs typically do not. We hypothesise that the lack of this inductive bias partially explains why GNNs fail at systematic generalisation for reasoning in relational domains. 

**Pooling** The pooling operator _ψ_ has to be chosen in accordance with the view of embeddings as epistemic states: if **x1** _, . . . ,_ **xk** capture sets of possible relationships then _ψ{_ **x1** _, . . . ,_ **xk** _}_ should intuitively capture the intersection of these sets. This requirement was studied in [32]. The central result from that paper is that the minimum and component-wise (i.e. Hadamard) product are compatible with this intuitive requirement, but sum pooling is not. In our setting, the embeddings capture probability distributions rather than sets, and this means that _ψ_ also needs to include a normalisation step. We thus consider two possibilities. First, _ψ⊙_ uses the component-wise product to combine the messages and then applies L1 normalisation to ensure the components of the resulting vector sum to 1. Similarly, _ϕ_ min uses min-pooling followed by L1-normalisation. 

**Training** To train the model, we assume we have access to a set of examples of the form ( _Fi, hi, ti, ri_ ) where _hi, ti_ are entities appearing in _Fi_ and the atom _ri_ ( _hi, ti_ ) can be inferred from _F_ . Different examples will involve a different fact set _Fi_ but the set of relations _R_ appearing in these fact sets is fixed. The relations are assumed to be pairwise disjoint, meaning that for each positive example ( _Fi, hi, ti, ri_ ) we can consider negative examples of the form ( _Fi, hi, ti, r[′]_ ) with _r[′] ∈R \ {ri}_ . Let us write **ti** for the final-layer embedding of entity _ti_ in the graph _Fi_ (with _hi_ as 

6 

the designated head entity). We train the model using a margin loss, imposing that the cross-entropy between **ti** and **r** is lower for positive examples than for negative examples. 

A crucial hyperparameter is the dimensionality _n_ of the embeddings, as it determines the number of primitive relations. We clearly should have _n ≥|R|_ so that all the relations of interest can be expressed, but choosing _n_ too high may lead to overfitting. To address this, we propose to jointly train several models, each with a relatively low number of dimensions. The idea is that these models can focus on different aspects of the relations, and can thus individually remain relatively simple. In the loss function, we then simply add up the cross-entropies from each of the _m_ models. Note that in this way, we do not require that any two relations can be distinguished in every model. We merely require that this is the case for at least one of the models. We will refer to _m_ as the number of _facets_ . 

**Expressivity** The model is capable of simulating an approximation of the algebraic closure algorithm, where we only maintain sets of possible relations between the head entity and the other entities, rather than between any pair of entities. We formally establish this correspondence in the appendix. 

**Forward-backward model** As we noted in Section 3.1, the order in which the relations on a relational path _r_ 1; _..._ ; _rk_ are composed sometimes matters. The model we have constructed thus far always composes such paths from left to right, i.e. we first compose _r_ 1 and _r_ 2, then compose the resulting relation with _r_ 3, etc. To mitigate this limitation, we introduce a backward model. The backward model relies on a designated tail entity. In particular, the embeddings are initialised as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0007-04.png)


These embeddings are updated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0007-06.png)


Where _ψ_ and _ϕ_ are defined as before. Note how the backward model does not introduce any new parameters. We rely on the idea that _ϕ_ captures the composition of relation vectors. In the forward model, the embedding of an entity _e_ is interpreted as capturing the relationship between _h_ and _e_ . In the backward model, the embedding of _e_ is instead interpreted as capturing the relationship between _e_ and _t_ . This is why **r** appears as the second argument of _ϕ_ in (6) and as the first argument in (7). 

Let us write **e** _[→]_ for the final-layer embedding of _e_ in the forward model and **e** _[←]_ for the final-layer embedding in the backward model. In particular, **t** _[→]_ and **t** _[←]_ now both capture the relationship between _h_ and _t_ . Moreover, for any entity _e_ on a path between _h_ and _t_ , the vector _ϕ_ ( **e** _[→] ,_ **e** _[←]_ ) should also capture this relationship. We take advantage of the aggregation operator _ψ_ to take into account all these predictions. In particular, we construct the following vector: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0007-09.png)


where we write _Eh,t_ for the entities that appear on some path from _h_ to _t_ . When there are multiple paths between _h_ and _t_ , we randomly select one of the shortest paths to define _Eh,t_ . The model is then trained as before, using **ˆr** as the predicted relation vector rather than **t** _[→]_ . Note that while this approach cannot exhaustively consider all possible compositions, it has the key advantage of remaining highly efficient, in contrast to most neuro-symbolic methods. A schematic of the forward-backward (FB) model that illustrates its learning dynamics is shown in Figure 2. 

## **5 Experiments** 

We use the challenging problem of inductive relational reasoning to evaluate our proposed model against GNN, transformer and neuro-symbolic baselines. We refer to our full (forward-backward) model as FB. We consider two variants of this model, which differ in the choice of the pooling operator: component-wise multiplication (FB- `mul` ) and min-pooling (FB- `min` ). The considered benchmarks involve relation classification queries of the form ( _h,_ ? _, t_ ), asking which relation holds between a given head entity _h_ and tail entity _t_ . We focus in particular on systematic generalisation, to assess whether models can deal with large distributional shifts from the training set, which is paramount in many real-world settings [18]. We consider two existing benchmarks designed to test systematic generalisation for relational reasoning: CLUTRR [33] and GraphLog [34]. We 

7 

Table 1: Results (accuracy) on CLUTRR after training on problems with _k ∈{_ 2 _,_ 3 _,_ 4 _}_ and then evaluating on problems with _k ∈{_ 5 _, . . . ,_ 10 _}_ . Results marked with _[∗]_ were taken from [25], those with _[†]_ from [22] and those with[2] from [7]. The best performance for each _k_ is highlighted in **bold** . 

||**5 Hops**|**6 Hops**|**7 Hops**|**8 Hops**|**9 Hops**|**10 Hops**|
|---|---|---|---|---|---|---|
|FB-`mul`(ours)|0.99_±_.01|**0.99**_±_**.01**|**0.99**_±_**.02**|0.99_±_.03|0.96_±_.03|**0.98**_±_**.02**|
|FB-`min`(ours)|0.99_±_.01|0.98_±_.02|0.98_±_.03|0.97_±_.06|0.95_±_.04|0.93_±_.07|
|NCRL2|**1.0**_±_**.01**|**0.99**_±_**.01**|0.98_±_.02|0.98_±_.03|0.98_±_.03|0.97_±_.02|
|R5_†_|0.99_±_.02|0.99_±_.04|0.99_±_.03|**1.0**_±_**.02**|**0.99**_±_**.02**|0.98_±_.03|
|CTP_∗_<br>_L_|0.99_±_.02|0.98_±_.04|0.97_±_.04|0.98_±_.03|0.97_±_.04|0.95_±_.04|
|CTP_∗_<br>_A_|0.99_±_.04|0.99_±_.03|0.97_±_.03|0.95_±_.06|0.93_±_.07|0.91_±_.05|
|CTP_∗_<br>_M_|0.98_±_.04|0.97_±_.06|0.95_±_.06|0.94_±_.08|0.93_±_.08|0.90_±_.09|
|GNTP_∗_|0.68_±_.28|0.63_±_.34|0.62_±_.31|0.59_±_.32|0.57_±_.34|0.52_±_.32|
|ET|0.99_±_.01|0.98_±_.02|**0.99**_±_**.02**|0.96_±_.04|0.92_±_.07|0.92_±_.07|
|GAT_∗_|0.99_±_.00|0.85_±_.04|0.80_±_.03|0.71_±_.03|0.70_±_.03|0.68_±_.02|
|GCN_∗_|0.94_±_.03|0.79_±_.02|0.61_±_.03|0.53_±_.04|0.53_±_.04|0.41_±_.04|
|NBFNet|0.83_±_.11|0.68_±_.09|0.58_±_.10|0.53_±_.07|0.50_±_.11|0.53_±_.08|
|R-GCN|0.97_±_.03|0.82_±_.11|0.60_±_.13|0.52_±_.11|0.50_±_.09|0.45_±_.09|
|RNN_∗_|0.93_±_.06|0.87_±_.07|0.79_±_.11|0.73_±_.12|0.65_±_.16|0.64_±_.16|
|LSTM_∗_|0.98_±_.03|0.95_±_.04|0.89_±_.10|0.84_±_.07|0.77_±_.11|0.78_±_.11|
|GRU_∗_|0.95_±_.04|0.94_±_.03|0.87_±_.08|0.81_±_.13|0.74_±_.15|0.75_±_.15|



Table 2: RCC-8 benchmark results (accuracy). Timeout refers to the model taking longer than 30 minutes for inference. Models are trained on graphs with _b ∈{_ 1 _,_ 2 _,_ 3 _}_ paths of length _k ∈{_ 2 _,_ 3 _,_ 4 _}_ . For each _k_ and _b_ , the best performance across all the models is highlighted in **bold** . 

|||||**2 Hops**|**3 Hops**|**4 Hops**|**5 Hops**|**6 Hops**|**7 Hops**|**8 Hops**|**9 Hops**|
|---|---|---|---|---|---|---|---|---|---|---|---|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|
|FB-`min`(ours)|_b_|=|2|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**0.99**_±_0.01|**0.99**_±_0.01|**0.97**_±_0.02|**0.97**_±_0.02|**0.95**_±_0.03|
||_b_|=|3|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**0.97**_±_0.02|**0.97**_±_0.02|**0.94**_±_0.04|**0.94**_±_0.04|**0.91**_±_0.04|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.99_±_0.01|0.99_±_0.0|0.98_±_0.01|0.98_±_0.01|0.97_±_0.01|
|FB-`mul`(ours)|_b_|=|2|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.88_±_0.04|0.87_±_0.04|0.80_±_0.05|0.79_±_0.04|0.75_±_0.05|
||_b_|=|3|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.78_±_0.07|0.76_±_0.06|0.70_±_0.06|0.69_±_0.06|0.65_±_0.05|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.99_±_0.0|**1.0**_±_0.0|0.99_±_0.01|0.99_±_0.01|0.98_±_0.02|
|ET|_b_|=|2|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.97_±_0.01|0.98_±_0.01|0.96_±_0.03|0.96_±_0.03|0.91_±_0.05|
||_b_|=|3|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.95_±_0.03|0.96_±_0.03|0.93_±_0.03|0.91_±_0.04|0.81_±_0.09|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.65_±_0.01|0.67_±_0.01|0.61_±_0.01|0.64_±_0.0|0.59_±_0.02|
|NCRL|_b_|=|2|0.68_±_0.0|0.68_±_0.01|0.69_±_0.0|0.43_±_0.0|0.45_±_0.01|0.40_±_0.0|0.41_±_0.0|0.38_±_0.02|
||_b_|=|3|0.57_±_0.01|0.58_±_0.0|0.58_±_0.0|0.36_±_0.0|0.39_±_0.0|0.34_±_0.02|0.35_±_0.01|0.32_±_0.02|
||_b_|=|1|0.56_±_0.01|**1.0**_±_0.01|**1.0**_±_0.01|0.41_±_0.01|0.39_±_0.01|0.38_±_0.01|timeout|timeout|
|R5|_b_|=|2|0.52_±_0.01|0.47_±_0.01|0.43_±_0.01|0.41_±_0.01|0.40_±_0.01|0.38_±_0.01|timeout|timeout|
||_b_|=|3|0.50_±_0.01|0.46_±_0.01|0.44_±_0.01|0.39_±_0.01|0.37_±_0.01|0.36_±_0.01|timeout|timeout|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.88_±_0.12|0.75_±_0.04|0.61_±_0.07|0.57_±_0.09|0.52_±_0.07|
|NBFNet|_b_|=|2|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.83_±_0.09|0.66_±_0.05|0.51_±_0.07|0.45_±_0.04|0.41_±_0.03|
||_b_|=|3|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.79_±_0.06|0.59_±_0.08|0.44_±_0.08|0.38_±_0.06|0.34_±_0.04|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.91_±_0.15|0.60_±_0.21|0.50_±_0.05|0.46_±_0.06|0.42_±_0.05|
|R-GCN|_b_|=|2|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.87_±_0.16|0.52_±_0.10|0.47_±_0.08|0.43_±_0.07|0.36_±_0.03|
||_b_|=|3|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.82_±_0.16|0.49_±_0.12|0.42_±_0.05|0.38_±_0.05|0.33_±_0.04|



also introduce a new benchmark based on RCC-8 relations, which goes further than the existing benchmarks on two fronts, as illustrated in Example 1: (i) the need to go beyond Horn rules to capture relational compositions and (ii) requiring models to aggregate information multiple relational paths. For CLUTRR and RCC-8, to test for systematic generalisation, models are trained on small graphs and subsequently evaluated on larger graphs. In particular, for CLUTRR, the length _k_ of the considered relational paths is varied, while for RCC-8 we vary both the number of relational paths _b_ and their length _k_ . In the case of GraphLog, the size of training and test graphs is similar, but models still need to apply learned rules in novel ways to perform well. 

**Main results** Results for CLUTRR, RCC-8 and GraphLog are shown in Tables 1, 2 and 3, respectively. We report the average accuracy and 2 _σ_ errors across 10 seeds for CLUTRR and 3 seeds for RCC-8 and GraphLog. For CLUTRR, both variants of our model outperform all GNN and RNN methods, as well as edge transformers (ET). The FB- `mul` model is also on par with the SoTA neuro- 

8 

Table 3: Results on GraphLog (accuracy). For each world, we report the number of distinct relation sequences between head and tail (ND) and the Average resolution length (ARL). Results marked with _[∗]_ were taken from [22] and those with _[†]_ from [7]. The best performance across all the models is highlighted in **bold** . 

|**World**|**ID**|**ND**|**ARL**|**E-GAT**_∗_|**R-GCN**_∗_|**CTP**_∗_|**R5**_∗_|**NCRL**_†_|**FB-**`mul`|
|---|---|---|---|---|---|---|---|---|---|
|World|6|249|5.06|0.536|0.498|0.533_±_0.03|0.687_±_0.05|**0.702**_±_**0.02**|0.563_±_0.016|
|World|7|288|4.47|0.613|0.537|0.513_±_0.03|**0.749**_±_**0.04**|-|0.521_±_0.04|
|World|8|404|5.43|0.643|0.569|0.545_±_0.02|0.671_±_0.03|**0.687**_±_**0.02**|0.547_±_0.032|
|World|11|194|4.29|0.552|0.456|0.553_±_0.01|**0.803**_±_**0.01**|-|0.698_±_0.019|
|World|32|287|4.66|0.700|0.621|0.581_±_0.04|0.841_±_0.03|-|**0.914**_±_**0.026**|



Table 4: Ablation study on the best FB model for CLUTRR and RCC-8. For both benchmarks, we show the average accuracy across all configurations (i.e. 4-10 hops for CLUTRR, and _k ∈{_ 2 _, ...,_ 9 _}_ and _b ∈{_ 1 _,_ 2 _,_ 3 _}_ for RCC-8), as well as the accuracy for the hardest setting (i.e. 10 hops for CLUTRR, and _k_ = 9 and _b_ = 3 for RCC-8). 

||**CLUTRR**<br>**Avg**<br>_k_ = 10|**RCC-8**<br>**Avg**<br>_b_ = 3_, k_ = 9|
|---|---|---|
|FB Model<br>With facets=1<br>Unconstrained embeddings<br>MLP+distmul composition<br>Forward model only|0.99<br>0.99<br>0.94<br>0.85<br>0.36<br>0.30<br>0.29<br>0.31<br>0.94<br>0.82|0.96<br>0.80<br>0.92<br>0.68<br>0.38<br>0.21<br>0.13<br>0.13<br>0.84<br>0.51|



symbolic methods NCRL [7] and R5 [22]. For RCC-8, as expected, the neuro-symbolic methods are largely ineffective, being substantially outperformed by our method as well as by ET. This highlights the fact that neuro-symbolic methods are not capable of modelling disjunctive rules. Our model with min-pooling achieves the best results. Edge transformers also perform well in general, but they underperform on the most challenging configurations (e.g. _k_ = 9 and _b_ = 3). Comparing the `mul` and `min` variants of our model, we find that `mul` performs better on single-path problems (e.g. CLUTRR and GraphLog), while `min` leads to better results when paths need to be aggregated (e.g. RCC-8). For GraphLog, we only consider worlds that are characterized as ‘hard’ by the authors in [22], with an Average Resolution Length (ARL) _>_ 4. GraphLog is more challenging and noisier than the previous two datasets and is amenable to path-based reasoning. Accordingly, we find that FB- `mul` is outperformed by R5 and NCRL in most cases, although FB- `mul` clearly improves the SoTA for World 32. Moreover, FB- `mul` outperforms CTPs and the GNN baselines. 

**Ablations** We confirm the importance of key architectural components of our model through an ablation study on CLUTRR and RCC-8, namely: a) the joint training of multiple lower-dimensional models focusing on different facets, b) modelling embeddings as epistemic states by requiring **r** and **a** _ij_ to be probability distributions, c) the use of a bilinear composition function _ϕ_ in Eq (6), and d) the forward and backward nature of our model encapsulated by Eq. (8). We ablate over all in the following way ceteris paribus: a) by setting the number of facets _m_ to 1, b) removing the requirement that embedding components are non-negative and sum to 1, c) replacing the composition function by distmul [42] after applying a 4-layer MLP with ReLU activation to both inputs of _ϕ_ . and d) considering only the forward embeddings **e** _[→]_ in Eq. (8). The results in Table 4 confirm the importance of all these components, with the use of embeddings as epistemic states and the bilinear form of _ϕ_ being particularly important. For c) the MLP is to ensure the model is not underparameterised. We see a similar performance degradation when distmul is used without the MLP (not shown). Further analysis is presented in the appendix. 

## **6 Conclusions** 

We have challenged the view that Graph Neural Networks are not capable of systematic generalisation in relational reasoning problems, by introducing a principled GNN architecture for this setting. To impose an appropriate inductive bias, node embeddings in our framework are treated as epistemic states, intuitively capturing sets of possible relationships. These epistemic states are iteratively refined by the GNN. In this way, the design of the GNN is closely aligned with the algebraic closure 

9 

algorithm, which is used for relational reasoning by symbolic methods, and a formal connection with this algorithm has been established. Furthermore, to ensure that the composition function which is implemented by the network is sufficiently expressive, while at the same time avoiding overfitting, we jointly train multiple lower-dimensional models, which intuitively each capture different facets of the considered relationships. Finally, by considering both a forward and a backward model, our method can mitigate the problems that may arise when relational paths can only be processed sequentially. While the resulting architecture stays close to standard GNNs, we have shown that our model can rival neuro-symbolic methods on standard benchmarks such as CLUTRR and GraphLog, while clearly outperforming existing GNN and transformer based methods. Moreover, we have highlighted that existing neuro-symbolic methods also have an important weakness, which arises from an implicit assumption that relations can be predicted from a single relational path. To explore this issue, we have introduced a new benchmark based on the RCC-8 calculus, finding that neuro-symbolic methods indeed fail on this benchmark, while our model still performs well in this more challenging setting. 

**Limitations** : Our proposed framework is limited by its statistical nature, where the parameters of the model will inevitably be biased by training data artefacts. While we have shown that our model can perform well in settings where existing neuro-symbolic methods fail (i.e. on the RCC-8 benchmark), the much stronger inductive bias imposed by the latter puts them at an advantage in severely data-limited settings. This can be seen in some of the GraphLog results. For a variant of CLUTRR where the model is only trained on problems of size _k ∈{_ 2 _,_ 3 _}_ , our model is also outperformed by the best neuro-symbolic methods (shown in Section A.8 of the appendix). 

## **Acknowledgments** This work was supported by EPSRC grant EP/W003309/1. 

## **References** 

- [1] J. F. Allen. Maintaining knowledge about temporal intervals. _Commun. ACM_ , 26(11):832–843, 1983. 

- [2] J. F. Allen. Maintaining knowledge about temporal intervals. _Communications of the ACM_ , 26(11):832–843, 1983. 

- [3] N. Amaneddine, J. Condotta, and M. Sioutis. Efficient approach to solve the minimal labeling problem of temporal and spatial qualitative constraints. In F. Rossi, editor, _IJCAI 2013, Proceedings of the 23rd International Joint Conference on Artificial Intelligence, Beijing, China, August 3-9, 2013_ , pages 696–702. IJCAI/AAAI, 2013. 

- [4] D. Bahdanau, S. Murty, M. Noukhovitch, T. H. Nguyen, H. d. Vries, and A. Courville. Systematic generalization: What is required and can it be learned? In _International Conference on Learning Representations_ , 2019. 

- [5] L. Bergen, T. J. O’Donnell, and D. Bahdanau. Systematic generalization with edge transformers. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_ , pages 1390–1402, 2021. 

- [6] A. Bordes, J. Weston, R. Collobert, and Y. Bengio. Learning structured embeddings of knowledge bases. In W. Burgard and D. Roth, editors, _Proceedings of the Twenty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2011, San Francisco, California, USA, August 7-11, 2011_ , pages 301–306. AAAI Press, 2011. 

- [7] K. Cheng, N. K. Ahmed, and Y. Sun. Neural compositional rule learning for knowledge graph reasoning. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. 

- [8] K. Cho, B. van Merrienboer, Ç. Gülçehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In A. Moschitti, B. Pang, and W. Daelemans, editors, _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP 2014, October 25-29, 2014, Doha, Qatar, A meeting of SIGDAT, a Special Interest Group of the ACL_ , pages 1724–1734. ACL, 2014. 

- [9] P. Clark, O. Tafjord, and K. Richardson. Transformers as soft reasoners over language. In C. Bessiere, editor, _Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI 2020_ , pages 3882–3890. ijcai.org, 2020. 

10 

- [10] A. Creswell, M. Shanahan, and I. Higgins. Selection-inference: Exploiting large language models for interpretable logical reasoning. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. 

- [11] Z. Cui, A. G. Cohn, and D. A. Randell. Qualitative and topological relationships in spatial databases. In D. J. Abel and B. C. Ooi, editors, _Advances in Spatial Databases, Third International Symposium, SSD’93, Singapore, June 23-25, 1993, Proceedings_ , volume 692 of _Lecture Notes in Computer Science_ , pages 296–315. Springer, 1993. 

- [12] A. S. d’Avila Garcez and G. Zaverucha. The connectionist inductive learning and logic programming system. _Appl. Intell._ , 11(1):59–77, 1999. 

- [13] J. Devlin, M. Chang, K. Lee, and K. Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In J. Burstein, C. Doran, and T. Solorio, editors, _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)_ , pages 4171–4186. Association for Computational Linguistics, 2019. 

- [14] N. Dziri, X. Lu, M. Sclar, X. L. Li, L. Jiang, B. Y. Lin, S. Welleck, P. West, C. Bhagavatula, R. L. Bras, J. D. Hwang, S. Sanyal, X. Ren, A. Ettinger, Z. Harchaoui, and Y. Choi. Faith and fate: Limits of transformers on compositionality. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ , 2023. 

- [15] S. Hochreiter and J. Schmidhuber. Long short-term memory. _Neural Comput._ , 9(8):1735–1780, 1997. 

- [16] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In Y. Bengio and Y. LeCun, editors, _3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings_ , 2015. 

- [17] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In _5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings_ . OpenReview.net, 2017. 

- [18] P. W. Koh, S. Sagawa, H. Marklund, S. M. Xie, M. Zhang, A. Balsubramani, W. Hu, M. Yasunaga, R. L. Phillips, I. Gao, T. Lee, E. David, I. Stavness, W. Guo, B. Earnshaw, I. Haque, S. M. Beery, J. Leskovec, A. Kundaje, E. Pierson, S. Levine, C. Finn, and P. Liang. Wilds: A benchmark of in-the-wild distribution shifts. In M. Meila and T. Zhang, editors, _Proceedings of the 38th International Conference on Machine Learning_ , volume 139 of _Proceedings of Machine Learning Research_ , pages 5637–5664. PMLR, 18–24 Jul 2021. 

- [19] T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa. Large language models are zero-shot reasoners. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ , 2022. 

- [20] S. Li, Z. Long, W. Liu, M. Duckham, and A. Both. On redundant topological constraints. _Artificial Intelligence_ , 225:51–76, 2015. 

- [21] Z. Long, M. Sioutis, and S. Li. Efficient path consistency algorithm for large qualitative constraint networks. In _IJCAI International Joint Conference on Artificial Intelligence_ , 2016. 

- [22] S. Lu, B. Liu, K. G. Mills, S. Jui, and D. Niu. R5: rule discovery with reinforced and recurrent relational reasoning. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ . OpenReview.net, 2022. 

- [23] A. H. Miller, A. Fisch, J. Dodge, A. Karimi, A. Bordes, and J. Weston. Key-value memory networks for directly reading documents. In J. Su, X. Carreras, and K. Duh, editors, _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016_ , pages 1400–1409. The Association for Computational Linguistics, 2016. 

- [24] P. Minervini, M. Bosnjak, T. Rocktäschel, S. Riedel, and E. Grefenstette. Differentiable reasoning on large knowledge bases and natural language. In _The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020_ , pages 5182–5190. AAAI Press, 2020. 

11 

- [25] P. Minervini, S. Riedel, P. Stenetorp, E. Grefenstette, and T. Rocktäschel. Learning reasoning strategies in end-to-end differentiable proving. In _Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event_ , volume 119 of _Proceedings of Machine Learning Research_ , pages 6938–6949. PMLR, 2020. 

- [26] S. H. Muggleton. Inductive logic programming. _New Gener. Comput._ , 8(4):295–318, 1991. 

- [27] D. A. Randell, Z. Cui, and A. G. Cohn. A spatial logic based on regions and connection. In B. Nebel, C. Rich, and W. R. Swartout, editors, _Proceedings of the 3rd International Conference on Principles of Knowledge Representation and Reasoning (KR’92). Cambridge, MA, USA, October 25-29, 1992_ , pages 165–176. Morgan Kaufmann, 1992. 

- [28] J. Renz and G. Ligozat. Weak composition for qualitative spatial and temporal reasoning. In P. van Beek, editor, _Principles and Practice of Constraint Programming - CP 2005, 11th International Conference, CP 2005, Sitges, Spain, October 1-5, 2005, Proceedings_ , volume 3709 of _Lecture Notes in Computer Science_ , pages 534–548. Springer, 2005. 

- [29] T. Rocktäschel and S. Riedel. End-to-end differentiable proving. In I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, editors, _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , pages 3788–3800, 2017. 

- [30] A. Santoro, D. Raposo, D. G. T. Barrett, M. Malinowski, R. Pascanu, P. W. Battaglia, and T. Lillicrap. A simple neural network module for relational reasoning. In I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, editors, _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , pages 4967–4976, 2017. 

- [31] M. S. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling. Modeling relational data with graph convolutional networks. In A. Gangemi, R. Navigli, M. Vidal, P. Hitzler, R. Troncy, L. Hollink, A. Tordai, and M. Alam, editors, _The Semantic Web - 15th International Conference, ESWC 2018, Heraklion, Crete, Greece, June 3-7, 2018, Proceedings_ , volume 10843 of _Lecture Notes in Computer Science_ , pages 593–607. Springer, 2018. 

- [32] S. Schockaert. Embeddings as epistemic states: Limitations on the use of pooling operators for accumulating knowledge. _CoRR_ , abs/2210.05723, 2022. 

- [33] K. Sinha, S. Sodhani, J. Dong, J. Pineau, and W. L. Hamilton. CLUTRR: A diagnostic benchmark for inductive reasoning from text. In K. Inui, J. Jiang, V. Ng, and X. Wan, editors, _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019_ , pages 4505–4514. Association for Computational Linguistics, 2019. 

- [34] K. Sinha, S. Sodhani, J. Pineau, and W. L. Hamilton. Evaluating logical generalization in graph neural networks. _CoRR_ , abs/2003.06560, 2020. 

- [35] G. G. Towell and J. W. Shavlik. Knowledge-based artificial neural networks. _Artif. Intell._ , 70(1-2):119–165, 1994. 

- [36] K. Valmeekam, M. Marquez, A. O. Hernandez, S. Sreedharan, and S. Kambhampati. Planbench: An extensible benchmark for evaluating large language models on planning and reasoning about change. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ , 2023. 

- [37] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, editors, _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , pages 5998–6008, 2017. 

- [38] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio. Graph attention networks. In _6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings_ . OpenReview.net, 2018. 

- [39] S. Welleck, X. Lu, P. West, F. Brahman, T. Shen, D. Khashabi, and Y. Choi. Generating sequences by learning to self-correct. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. 

12 

- [40] K. Xu, J. Li, M. Zhang, S. S. Du, K. Kawarabayashi, and S. Jegelka. What can neural networks reason about? In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net, 2020. 

- [41] K. Xu, M. Zhang, J. Li, S. S. Du, K.-I. Kawarabayashi, and S. Jegelka. How neural networks extrapolate: From feedforward to graph neural networks. In _International Conference on Learning Representations_ , 2021. 

- [42] B. Yang, W. tau Yih, X. He, J. Gao, and L. Deng. Embedding entities and relations for learning and inference in knowledge bases, 2015. 

- [43] H. Zhang, L. H. Li, T. Meng, K. Chang, and G. V. den Broeck. On the paradox of learning to reason from data. In _Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China_ , pages 3365–3373. ijcai.org, 2023. 

- [44] Y. Zhang, X. Chen, Y. Yang, A. Ramamurthy, B. Li, Y. Qi, and L. Song. Efficient probabilistic logic reasoning with graph neural networks. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net, 2020. 

- [45] Y. Zhang and Q. Yao. Knowledge graph reasoning with relational digraph. In F. Laforest, R. Troncy, E. Simperl, D. Agarwal, A. Gionis, I. Herman, and L. Médini, editors, _WWW ’22: The ACM Web Conference 2022, Virtual Event, Lyon, France, April 25 - 29, 2022_ , pages 912–924. ACM, 2022. 

- [46] Z. Zhu, Z. Zhang, L. A. C. Xhonneux, and J. Tang. Neural bellman-ford networks: A general graph neural network framework for link prediction. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_ , pages 29476–29490, 2021. 

13 

## **A Supplemental material** 

## **A.1 Semantics of entailment** 

For simple path rules of the form (2), the semantics of entailment can be defined in terms of the immediate consequence operator. Let _RF_ be the grounding of _R_ w.r.t. _F_ , i.e. for each rule of the form _r_ ( _X, Z_ ) _← r_ 1( _X, Y_ ) _∧ r_ 2( _Y, Z_ ) in _R_ , _RF_ contains all the possible rules of the form _r_ ( _a, c_ ) _← r_ 1( _a, b_ ) _∧ r_ 2( _b, c_ ) that can be obtained by substituting the variables for entities that appear in _F_ . Since we assume that both _F_ and _R_ are finite, we have that _RF_ is finite as well. Let _F_ 0 = _F_ and for _i ≥_ 1 define: 

_Fi_ = _Fi−_ 1 _∪{r_ ( _a, c_ ) _| ∃b ∈E . r_ 1( _a, b_ ) _, r_ 2( _b, c_ ) _∈Fi−_ 1 and ( _r_ ( _a, c_ ) _← r_ 1( _a, b_ ) _∧ r_ 2( _b, c_ )) _∈RF }_ 

Since there are finitely many atoms _r_ ( _a, c_ ) that can be constructed using the entities and relations appearing in _RF_ , this process reaches a fixpoint after a finite number of steps, i.e. for some _ℓ ∈_ N we have _Fℓ_ = _Fℓ_ +1. Let us write _F[∗]_ for this fixpoint. Then we define _F ∪R |_ = _r_ ( _a, b_ ) iff _r_ ( _a, b_ ) _∈F[∗]_ . 

**Example 2.** _The canonical example for this setting concerns reasoning about family relationships. In this case, K contains rules such as:_ 

_grandfather_ ( _X, Z_ ) _← father_ ( _X, Y_ ) _∧ mother_ ( _Y, Z_ ) 

_If F_ = _{father_ ( _bob, alice_ ) _, mother_ ( _alice, eve_ ) _} then RF contains rules such as:_ 

_grandfather_ ( _bob, eve_ ) _← father_ ( _bob, alice_ ) _∧ mother_ ( _alice, eve_ ) _grandfather_ ( _bob, alice_ ) _← father_ ( _bob, alice_ ) _∧ mother_ ( _alice, alice_ ) _grandfather_ ( _bob, bob_ ) _← father_ ( _bob, alice_ ) _∧ mother_ ( _alice, bob_ ) _grandfather_ ( _bob, eve_ ) _← father_ ( _bob, eve_ ) _∧ mother_ ( _eve, eve_ ) _. . ._ 

_We have F_ 1 = _{father_ ( _bob, alice_ ) _, mother_ ( _alice, eve_ ) _, grandfather_ ( _bob, eve_ ) _}, with F_ 1 = _F_ 2 = _F[∗] . We thus find: K ∪F |_ = _grandfather_ ( _bob, eve_ ) _._ 

For the more general setting with disjunctive rules and constraints, we can define the semantics of entailment in terms of Herbrand models. Given a set of disjunctive rules and constraints _R_ and a set of facts _F_ , the Herbrand universe _UR,F_ is the set of all atoms of the form _r_ ( _a, b_ ) which can be constructed from a relation _r_ and entities _a, b_ appearing in _R ∪F_ . A Herbrand interpretation _ω_ is a subset of _UR,F_ . The interpretation _ω_ satisfies a ground disjunctive rule of the form _s_ 1( _a, c_ ) _∨ . . . ∨ sk_ ( _a, c_ ) _← r_ 1( _a, b_ ) _∧ r_ 2( _b, c_ ) iff either _{r_ 1( _a, b_ ) _, r_ 2( _b, c_ ) _} ̸⊆ ω_ or _ω ∩{s_ 1( _a, c_ ) _, . . . , sk_ ( _a, c_ ) _}_ = _∅_ . The interpretation _ω_ satisfies a ground constraint of the form _⊥← r_ 1( _a, b_ ) _∧ r_ 2( _a, b_ ) iff _{r_ 1( _a, b_ ) _, r_ 2( _a, b_ ) _} ̸⊆ ω_ . We say that _ω_ is a model of the grounding _RF_ iff _ω_ satisfies all the ground rules and constraints in _RF_ . Finally, we have that _R ∪F |_ = _r_ ( _a, b_ ) iff _r_ ( _a, b_ ) is contained in every model of _RF_ . 

## **A.2 Proof of Proposition 1** 

The correctness of Proposition 1 immediately follows from Lemmas 1 and 2 below. 

**Lemma 1.** _Suppose there exists a relational path r_ 1; _..._ ; _rk connecting a and b in GF such that r can be derived from r_ 1; _..._ ; _rk. Then it holds that K ∪F |_ = _r_ ( _a, b_ ) _._ 

_Proof._ Let _r_ 1; _..._ ; _rk_ be a relational path connecting _a_ and _b_ , such that _r_ can be derived from _r_ 1; _..._ ; _rk_ . Then _F_ contains facts of the form _r_ 1( _a, x_ 1) _, r_ 2( _x_ 1 _, x_ 2) _, ..., rk_ ( _xk−_ 1 _, b_ ). Let us now consider the derivation from _r_ 1; _..._ ; _rk_ to _r_ . After the first derivation step, we have a relational path of the form _r_ 1; _..._ ; _ri−_ 2; _s_ ; _ri_ +1; _..._ ; _rk_ . In this case, _K_ contains a rule of the form _s_ ( _X, Z_ ) _← ri−_ 1( _X, Y_ ) _∧ ri_ ( _Y, Z_ ). This means that _K ∪F |_ = _s_ ( _xi−_ 2 _, xi_ ). We thus have a relational path of the form _r_ 1; _..._ ; _ri−_ 2; _s_ ; _ri_ +1; _..._ ; _rk_ , where each relation is associated with an atom that is entailed by _K ∪F_ . Each derivation step introduces such an atom (while reducing the length of the relational path by 1). After _k −_ 1 steps, we thus obtain the atom _r_ ( _a, b_ ), from which it follows that _K ∪F |_ = _r_ ( _a, b_ ). 

**Lemma 2.** _Suppose that K ∪F |_ = _r_ ( _a, b_ ) _. Then it holds that there exists a relational path r_ 1; _..._ ; _rk connecting a and b in GF such that r can be derived from r_ 1; _..._ ; _rk._ 

14 

_Proof._ Recall from Section A.1 that _K ∪F |_ = _r_ ( _a, b_ ) iff _r_ ( _a, b_ ) _∈F[∗]_ . It thus suffices to show by induction that _r_ ( _a, b_ ) _∈Fi_ implies that there exists a relational path _r_ 1; _..._ ; _rk_ connecting _a_ and _b_ in _GF_ such that _r_ can be derived from _r_ 1; _..._ ; _rk_ . First, if _r_ ( _a, b_ ) _∈F_ 0 then by definition there is a relational path _r_ connecting _a_ and _b_ in _GF_ , meaning that the result is trivially satisfied. Now suppose that the result has already been shown for _Fi−_ 1. Let _r_ ( _a, b_ ) _∈Fi \ Fi−_ 1. Then there must exist facts _r_ 1( _a, x_ ) and _r_ 2( _x, b_ ) in _Fi−_ 1 such that _K_ contains a rule of the form _r_ ( _X, Z_ ) _← r_ 1( _X, Y_ ) _∧r_ 2( _Y, Z_ ). By induction, we have that there is a relational path _s_ 1; _..._ ; _sℓ_ 1 connecting _a_ and _x_ such that _r_ 1 can be derived from this path, and a relational path _t_ 1; _..._ ; _tℓ_ 2 connecting _x_ and _b_ from which _r_ 2 can be derived. We then have that _s_ 1; _...sℓ_ 1; _t_ 1; _..._ ; _tℓ_ 2 is a path connecting _a_ and _b_ . Furthermore, we have that _r_ 1; _r_ 2 can be derived from this path, and thus also _r_ . 

## **A.3 Reasoning about disjunctive rules using algebraic closure** 

We consider knowledge bases with three types of rules. First, we have disjunctive rules that encode relational compositions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0015-03.png)


Second, we have constraints of the following form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0015-05.png)


meaning that _r_ 1 and _r_ 2 are disjoint. Finally, we consider knowledge about inverse relations, expressed using rules of the following form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0015-07.png)


We now describe the algebraic closure algorithm, which can be used to decide entailment for calculi such as RCC-8. We also describe an approximation of this algorithm, which we call _directional algebraic closure_ . This approximation closely corresponds to how GNN models can reason in this setting, as will be made clear below. 

**Full algebraic closure** Let us make the following assumptions: 

- The knowledge base _K_ contains rules of the form (9), encoding the composition of relations _r_ 1 and _r_ 2. 

- We also have that _K_ contains the rule[�] _r∈R[r]_[(] _[X, Y]_[ )] _[←⊤]_[,][expressing][that][the][set][of] relations is exhaustive. 

- For all distinct relations _r_ 1 _, r_ 2 _∈R_ , _r_ 1 = _r_ 2, _K_ contains a constraint of the form (10), expressing that the relations are pairwise disjoint. 

- For every _r ∈R_ there is some relation _r_ ˆ _∈R_ , such that _K_ contains the rules _r_ ( _Y, X_ ) _← r_ ˆ( _X, Y_ ) and ˆ _r_ ( _Y, X_ ) _← r_ ( _X, Y_ ), expressing that ˆ _r_ is the inverse of _r_ . 

- _K_ contains no other rules. 

Let us write _r_ 1 _◦ r_ 2 for the set of relations that appears in the head of the rule defining the composition of _r_ 1 and _r_ 2 in _K_ . If no such rule exists in _K_ for _r_ 1 and _r_ 2 then we define _r_ 1 _◦ r_ 2 = _R_ . Let _E_ be the set of entities that appear in _F_ . Let us assume that _F_ is consistent with _K_ , i.e. _K ∪F ̸|_ = _⊥_ . 

The main idea of the algebraic closure algorithm is that we iteratively refine our knowledge of what relationships are possible between different entities. Specifically, for all entities _e, f ∈E_ , we define the initial set of possible relationships as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0015-17.png)


where _r_ ˆ is the unique relation that is asserted to be the inverse of _r_ in _K_ . Note that because we ˆ assumed _F_ to be consistent, if _r_ ( _e, f_ ) _∈F_ and _r[′]_ ( _f, e_ ) _∈F_ it must be the case that _r[′]_ = _r_ . We now iteratively refine the sets _Xef_[(] _[i]_[)][.][Specifically, for] _[ i][ ≥]_[1][, we define:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0015-19.png)


15 

where, for _X, Y ⊆R_ , we define: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0016-01.png)


Since each of the sets _Xef_[(] _[i]_[)][only contains finitely many elements, and there are only finitely many] such sets, this process must clearly converge after a finite number of steps. Let us write _Xef_ of the sets of relations that are obtained upon convergence. For many spatial and temporal calculi, we have that _r ∈ Xef_ iff _K ∪F ∪{r_ ( _e, f_ ) _} ̸|_ = _⊥_ . In other words, _Xef_ encodes everything that can be inferred about the relationship between _e_ and _f_ . In particular, we have have _K ∪F |_ = _r_ ( _e, f_ ) iff _Xef_ = _{r}_ . Note that this equivalence only holds for particular calculi: in general, _K ∪F |_ = _r_ ( _e, f_ ) does not imply _Xef_ = _{r}_ . 

**Directional algebraic closure** The algebraic closure algorithm relies on sets _Xef_[(] _[i]_[)][for every pair of] entities, which limits its scalability (although more efficient special cases have been studied [3]). Any simulation of this algorithm using a neural network would thus be unlikely to scale to large graphs. For this reason, we study an approximation, which we refer to as _directional algebraic closure_ . This approximation essentially aims to infer the possible relationships between a fixed head entity _h_ and all the other entities from the graph. The relationship between _h_ and a given entity _e_ is determined based on the paths in _G_ connecting _h_ to _e_ . For this approximation, we furthermore omit rules about inverse relations. Other than this, we make the same assumptions about _K_ as before. We now learn sets _Xe_[(0)] which capture the possible relationships between _h_ and _e_ . These sets are initialised as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0016-04.png)


We define for _i ≥_ 1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0016-06.png)


with 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0016-08.png)


where _r_ 1 _◦ r_ 2 is defined as before. We write _Xe_ to denote the sets _Xe_[(] _[i]_[)] that are obtained upon convergence. Clearly, after a finite number of iterations _i_ we have _Xe_[(] _[i]_[)] = _Xe_[(] _[i]_[+1)] . Note how the directional closure algorithm essentially limits the full algebraic closure algorithm to compositions of the form _Xhe ⋄ Xef_ , for entities _e_ and _f_ such that there is a fact of the form _r_ ( _e, f_ ) in _F_ . As the following result shows, the algorithm is still sound, although it may infer less knowledge than the full algebraic closure algorithm. 

For a rule _ρ_ of the form (9), we write _head_ ( _ρ_ ) to denote the set _{s_ 1 _, ..., sk}_ of relations appearing in the head of the rule and _body_ ( _ρ_ ) to denote the pair ( _r_ 1 _, r_ 2) of relations appearing in the body. **Proposition 2.** _Assume that K consists of (i) rules of the form_ (9) _, (ii) for each pair of distinct relations r_ 1 _, r_ 2 _the disjointness constraint_ (10) _, and (iii) the rule_[�] _r∈R[r]_[(] _[X, Y]_[ )] _[ ←⊤][.][Let][ e][ ∈E][.][It] holds that K ∪F |_ =[�] _r∈Xe[r]_[(] _[h, e]_[)] _[.]_ 

_Proof._ We show this result by induction. First note that we have _K ∪F |_ =[�] _r∈Xe_[(0)] _r_ ( _h, e_ ). Indeed, either _Xe_[(0)] = _{r_ ( _h, e_ ) _}_ with _r_ ( _h, e_ ) _∈F_ , in which case the claim clearly holds, or _Xe_[(0)] = _R_ , in which case the claim holds because _K_ contains the rule[�] _r∈R[r]_[(] _[X, Y]_[ )] _[←⊤]_[.][Now][suppose][we] have already established _K ∪F |_ =[�] _r∈Xe_[(] _[i][−]_[1)] _r_ ( _h, e_ ) for every _e ∈E_ . To show that _K ∪F |_ = � _r∈Xe_[(] _[i]_[)] _[r]_[(] _[h, e]_[)][, it is sufficient to show that] _[ K ∪F][|]_[=] _[¬][r]_[(] _[h, e]_[)][ for every] _[ r][∈][X] e_[(] _[i][−]_[1)] _\ Xe_[(] _[i]_[)][.][Let] _r ∈ Xe_[(] _[i][−]_[1)] _\ Xe_[(] _[i]_[)][.][Then there must exist some] _[ s]_[(] _[f, e]_[)] _[ ∈F]_[such that] _[ r][∈][/][X] f_[(] _[i][−]_[1)] _⋄ s_ . In that case, for every _t ∈ Xf_[(] _[i][−]_[1)] we have _r ∈/ t ◦ s_ . This means that for every _t ∈ Xf_[(] _[i][−]_[1)] there exists some rule _ρt_ in _K_ such that _body_ ( _ρt_ ) = ( _t, s_ ) and _r ∈/ heads_ ( _ρt_ ). We have for every _t ∈ Xf_[(] _[i][−]_[1)] that _K ∪F |_ = _t_ ( _h, f_ ) _→∨u∈heads_ ( _ρt_ ) _u_ ( _h, e_ ). Moreover, because of our assumption that _K_ encodes that all relations are pairwise disjoint, for _u ̸_ = _r_ we have _K |_ = _u_ ( _h, e_ ) _→¬r_ ( _h, e_ ). We thus find for every _t ∈ Xf_[(] _[i][−]_[1)] that _K∪F |_ = _t_ ( _h, f_ ) _→¬r_ ( _h, e_ ). By induction we also have _K∪F |_ =[�] _t∈Xf_[(] _[i][−]_[1)] _t_ ( _h, e_ ). Together we find _K ∪F |_ = _¬r_ ( _h, e_ ). 

16 

## **A.4 Expressivity** 

We now show that the base GNN model, corresponding to (15), is capable of simulating the directional algebraic closure algorithm. Specifically, we show the following result, which associates the embeddings **e**[(] _[i]_[)] from the GNN with the sets _Xe_[(] _[i]_[)][.][We show the following result for the min-pooling] operator _ψ_ min defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0017-02.png)


**Proposition 3.** _Let F be a set of facts and K a knowledge base satisfying the conditions of Proposition 2. Furthermore assume that K ∪F is consistent, i.e. K ∪F ̸|_ = _⊥. Let Xe_[(] _[i]_[)] _be sets of relations that are constructed using the directional algebraic closure algorithm, and let e[i] j[denote the][ j][th][ coordinate] of_ **e**[(] _[i]_[)] _. Let the pooling operation be chosen as ψ_ = _ψ_ min _. There exists a parameterisation of the vectors_ **r** _i and_ **a** _ij such that:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0017-04.png)


_Proof._ Let _r_ 1 _, ..., rn_ be an enumeration of the relations in _R ∪{id}_ , where we fix _r_ 1 = _id_ . Let **a** _ij_ = ( _a[ij]_ 1 _[, ..., a] n[ij]_[)][ be defined for as follows (] _[i][ ∈{]_[1] _[, ..., n][}]_[):] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0017-06.png)


where we define _id ◦ rj_ = _rj_ for every _j ∈{_ 1 _, ..., n}_ . Note that **a** _ij_ indeed satisfies the requirements of the model: the coordinates of **a** _ij_ are non-negative and sum to 1, while **a** 1 _j_ = _one-hot_ ( _j_ ) follows from the fact that _id ◦ rj_ = _rj_ . Furthermore, we define **r** _j_ = one-hot( _j_ ). 

We show the result by induction. For _i_ = 0, if _F_ contains a fact of the form _rl_ ( _h, e_ ), we have _Xe_[(0)] = _{rl}_ . We show that **e**[(1)] is non-zero in the _l_[th] coordinate and zero everywhere else. We have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0017-09.png)


This already shows that **e**[(1)] is zero in all coordinates apart from the _l_[th] . Now let _f_ = _h_ and _rp_ be such that _rp_ ( _f, e_ ) _∈F_ . We have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0017-11.png)


We need to show that the _l_[th] coordinate of the latter vector is non-zero. Since _K ∪F_ is consistent and _F_ contains both _rl_ ( _h, e_ ) and _rp_ ( _f, e_ ), it has to be the case there there is some _q ∈{_ 1 _, ..., n}_ such that _rl ∈ rq ◦ rp_ . There thus exists some _q ∈{_ 1 _, ..., n}_ such that the _l_[th] coordinate of **a** _qp_ is non-zero. Since all vectors of the **a** _ip_ vectors have non-negative coordinates, it follows that the _l_[th] coordinate of _n_ 1 � _i_ **[a]** _[ip]_[is non-zero.][We have thus shown that] _[ {][r][j][ |][ e]_[0] _j[>]_[ 0] _[,]_[ 2] _[ ≤][j][≤][n][}]_[ =] _[ {][r][l][}]_[ =] _[ X] e_[(0)] _⊇ Xe_[(1)][.] 

If _F_ does not contain any facts of the form _rl_ ( _h, e_ ), then _Xe_[(0)] = _R_ . We need to show for each _l ∈{_ 2 _, ..., n}_ that either _e_[1] _l_[is non-zero or] _[ r][l][∈][/][X] e_[1][.][We have that] _[ e]_[1] _l_[is non-zero if the] _[ l]_[th][component] of _ϕ_ ( **f**[(0)] _,_ **r** _p_ ) is non-zero for every fact of the form _rp_ ( _f, e_ ) in _F_ , where _f_ = _h_ . Consider such a fact _rp_ ( _f, e_ ) and assume the _l_[th] component of _ϕ_ ( **f**[(0)] _,_ **r** _p_ ) is actually 0. Since _f_ = _h_ we have **f**[(0)] = ( _n_[1] _[, ...,] n_[1][)][.][The] _[l]_[th][component][of] _[ϕ]_[(] **[f]**[ (0)] _[,]_ **[ r]** _[p]_[)][=] _n_ 1 � _i_ **[a]** _[ip]_[can][thus][only][be][0][if][the] _[l]_[th] component of **a** _ip_ is 0 for every _i_ . This implies that _rl ∈/ ri ◦ rp_ for any _i ∈{_ 1 _, ..., n}_ , from which it follows that _rl ∈/ Xe_[(1)][.] 

Now suppose that _Xe_[(] _[i][−]_[1)] _⊇{rj | e[i] j[>]_[0] _[,]_[ 2] _[≤][j][≤][n][}][⊇][X] e_[(] _[i]_[)] holds for every entity _e_ . We show that (13) must then hold as well. We first show _Xe_[(] _[i]_[)] _⊇{rj | e[i] j_[+1] _>_ 0 _,_ 2 _≤ j ≤ n}_ . To this end, we need to show that for each _rj ∈ Xe_[(] _[i][−]_[1)] _\ Xe_[(] _[i]_[)] it holds that _e[i] j_[+1] = 0. If _rj ∈ Xe_[(] _[i][−]_[1)] _\ Xe_[(] _[i]_[)] it means that there is some _rp_ ( _f, e_ ) _∈F_ such that _rj ∈/ Xf_[(] _[i][−]_[1)] _⋄ rp_ . This is the case if _rj ∈/ rq ⋄ rp_ 

17 

Table 5: RCC-8 composition table [11] (excluding eq). 

||dc|ec|po|tpp|ntpp|tppi|ntppi|
|---|---|---|---|---|---|---|---|
|dc|_R_8|dc,<br>ec,<br>po,<br>tpp,ntpp|dc,<br>ec,<br>po,<br>tpp,ntpp|dc,<br>ec,<br>po,<br>tpp,ntpp|dc,ec,po,tpp,<br>ntpp|dc|dc|
|ec|dc,<br>ec,<br>po,<br>tppi,ntppi|dc,<br>ec,<br>po,<br>tpp,tppi,eq|dc,<br>ec,<br>po,<br>tpp,ntpp|ec, po, tpp,<br>ntpp|po,tpp,ntpp|dc,ec|dc|
|po|dc,<br>ec,<br>po,<br>tppi,ntppi|dc,<br>ec,<br>po,<br>tppi,ntppi|_R_8|po,tpp,ntpp|po,tpp,ntpp|dc,<br>ec,<br>po,<br>tppi,ntppi|dc,<br>ec,<br>po,<br>tppi,ntppi|
|tpp|dc|dc,ec|dc,<br>ec,<br>po,<br>tpp,ntpp|tpp,ntpp|ntpp|dc,<br>ec,<br>po,<br>tpp,tppi,eq|dc,<br>ec,<br>po,<br>tppi,ntppi|
|ntpp|dc|dc|dc,<br>ec,<br>po,<br>tpp,ntpp|ntpp|ntpp|dc,<br>ec,<br>po,<br>tpp,ntpp|_R_8|
|tppi|dc,<br>ec,<br>po,<br>tppi,ntppi|ec, po, tppi,<br>ntppi|po,<br>tppi,<br>ntppi|po, eq, tpp,<br>tppi|po,tpp,ntpp|tppi,ntppi|ntppi|
|ntppi|dc,<br>ec,<br>po,<br>tppi,ntppi|po,<br>tppi,<br>ntppi|po,<br>tppi,<br>ntppi|po,<br>tppi,<br>ntppi|po, tppi, tpp,<br>ntpp,ntppi,eq|ntppi|ntppi|



for every _rq ∈ Xf_[(] _[i][−]_[1)] . This, in turn, means that the _j_[th] component of **a** _qp_ is 0 for every _q_ such that _rq ∈ Xf_[(] _[i][−]_[1)] . Furthermore, by the induction hypothesis, we know that _rq ∈/ Xf_[(] _[i][−]_[1)] means _fq[i]_[= 0][.] In other words, for each _q_ we have that either _fq[i]_[= 0][ or that the] _[ j]_[th][component of] **[ a]** _[qp]_[is 0.][It follows] that the _j_[th] component of _ϕ_ ( **f**[(] _[i]_[)] _,_ **r** _p_ ) is 0, and thus also that _e[i] j_[+1] = 0. 

We now show _{rj | e[i] j_[+1] _>_ 0 _,_ 2 _≤ j ≤ n} ⊇ Xe_[(] _[i]_[+1)] . Let _j_ be such that _e[i] j[>][e][i] j_[+1] = 0. We need to show that _rj ∈/ Xe_[(] _[i]_[+1)] . If _e[i] j[>][e][i] j_[+1] = 0 there needs to be some _rp_ ( _f, e_ ) _∈F_ such that the _j_[th] component of _ϕ_ ( **f**[(] _[i]_[)] _,_ **r** _p_ ) is 0. We have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0018-04.png)


So when the _j_[th] component of _ϕ_ ( **f**[(] _[i]_[)] _,_ **r** _p_ ) is 0 we must have for each _l_ that either _fl[i]_[=][0][or][that] _a[j] lp_[=][0][.][By the induction hypothesis,] _[ f] l[ i]_[=][0][ implies] _[ r][l][∈][/][X] f_[(] _[i]_[)][.][Furthermore] _[ a][j] lp_[=][0][ means that] _rj ∈/ rl ◦ rp_ . When the _j_[th] component of _ϕ_ ( **f**[(] _[i]_[)] _,_ **r** _p_ ) is 0, we thus have that either _rl ∈/ Xf_[(] _[i]_[)] or _rj ∈/ rl ◦ rp_ for each _l_ , which implies _rj ∈/ Xe_[(] _[i]_[+1)] . 

When _ψ⊙_ is used instead of _ψ_ min, a similar result holds. If _ψ⊙_ simply uses the component-wise product, then the same argument as in the proof of Proposition 3 still holds. In practice, however, for numerical stability, we evaluate _ψ⊙_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0018-07.png)


where we write _⊙_ for the Hadamard product and **z** = ( _ε, . . . , ε_ ) for some small constant _ε >_ 0. As long as _ε_ is sufficiently small, this does not affect the ability of the GNN model to simulate the directional closure algorithm. In particular, whenever _e[i] j_[= 0][ in the GNN with] _[ ψ]_[min][ we can ensure] that this coordinate is arbitrarily small in the GNN with Ψ _⊙_ (choosing the **a** _ij_ and **r** _j_ vectors as in the proof of Proposition 3), by selecting _ε_ small enough. In particular, there exists some _δ >_ 0 such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0018-09.png)


## **A.5 RCC-8 Dataset** 

RCC-8 [27] uses eight primitive relations to describe qualitative spatial relationships between regions: ntpp( _a, b_ ) means that _a_ is a proper part of the interior of _b_ , tpp( _a, b_ ) means that _a_ is a proper part of _b_ and shares a boundary point with _b_ , po( _a, b_ ) means that _a_ and _b_ are overlapping (but neither is included in the other), dc( _a, b_ ) means that _a_ and _b_ are disjoint, ec( _a, b_ ) means that _a_ and _b_ are adjacent (i.e. sharing a boundary point but no interior points), eq( _a, b_ ) means that _a_ and _b_ are equal, and ntppi and tppi are the inverses of ntpp and tpp. The RCC-8 semantics is governed by the so-called composition table, which describes the composition mapping between two relations. This is shown in Table 5 where the trivial composition with the identity element eq being itself is dropped. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0019-00.png)


**----- Start of picture text -----**<br>
0<br>EC NTPP TPP NTPP NTPP NTPP<br>1 3 5 7 9 11<br>EQ NTPP EQ NTPP EQ PO DC<br>2 4 6 8 10 12<br>TPP EQ TPPI EQ EQ TPP<br>17<br>**----- End of picture text -----**<br>


Figure 3: An example from the RCC-8 benchmark with _b_ = 6 paths from the source node 0 to the target node 17. Each path has a length of _k_ = 3 where _k_ is the number of hops or edges from the node 0 to node 17. The graph has to be collapsed into the single relation ntpp(0 _,_ 17) using information from all the paths. 

Each entry in this table corresponds to a rule of the form (4), specifying the possible relations that may hold between two regions _a_ and _c_ , when we know the RCC-8 relation that holds between _a_ and some region _b_ as well as the relation that holds between _b_ and _c_ . For instance, the composition of ec and tppi is given by the set _{_ dc _,_ ec _}_ , which means that from _{_ ec( _a, b_ ) _,_ tppi( _b, c_ ) _}_ we can infer dc( _a, c_ ) _∨_ ec( _a, c_ ). In the table, we write _R_ 8 to denote that any RCC-8 relation is possible. 

Our proposed RCC-8 benchmark is similar to CLUTRR and GraphLog in its focus on assessing inductive relational reasoning via relation classification queries of the form ( _h,_ ? _, t_ ), where we need to predict which relation holds between a given head entity _h_ and tail entity _t_ . The available knowledge is provided as a set of facts _F_ , which we can again think of as a graph. The inductive part refers to the fact that the model is trained and tested on distinct graphs. The main difficulty comes from the fact that only small graphs are available for training, while some of the test graphs are considerably larger. 

However, different from existing benchmarks, our RCC-8 dataset requires reasoning about disjunctive rules, which is particularly challenging for many methods. The kind of knowledge that has to be learned is thus more expressive than the Horn rules which are considered in most existing benchmarks. As illustrated in Example 1, this means that models need to process different relational paths and aggregate the resulting information. We vary the difficulty of problem instances based on the number _b_ of such paths and their length _k_ . Figure 3 provides an example where there are _b_ = 6 paths of length _k_ = 3. Each path is partially informative, in the sense that it allows to exclude certain candidate relations, but is not sufficiently informative to pinpoint the exact relation. The model thus needs to rely on the different paths to be able to exclude all but one of the eight possible relations. 

In summary, within the context of benchmarks for systematic generalisation, our RCC-8 benchmark is novel on two fronts compared to existing benchmarks such as CLUTRR [33] and GraphLog [34]: 

- **Going beyond Horn rules for relation composition** : The composition of some RCC-8 relations is a disjunction of several RCC-8 relations. 

- **Multi-path information aggregation** : Models need to reason about multiple relational paths (for the case where _b >_ 1) to infer the correct answer. 

## **A.5.1 Dataset Generation** 

We now explain how the dataset was created. All sampling in the discussion below is uniform random. Each problem instance has to be constructed such that after aggregating the information provided by all the relational paths, we need to be able to infer a singleton label. In other words, problem 

19 

instances need to be consistent (i.e. the information provided by different paths cannot be conflicting) and together all the paths need to be informative enough to uniquely determine which relation holds between the head and tail entity. This makes brute-force sampling of problem instances prohibitive. Instead, to create a problem instance involving _b_ paths of length _k_ , we first sample a base graph, which has _b_ shorter paths, with a length in _{_ 2 _,_ 3 _,_ 4 _}_ . This is done by pre-computing relational compositions for a large number of paths and then selecting _b_ paths whose intersection is a singleton. Then we repeatedly increase the length of the paths by selecting an edge and replacing it by a short path whose composition is equal to the corresponding relation. 

Finally, to add further diversity to the graph topology, for each of the _b_ paths, we allow 1 edge from the base graph to be replaced by a subgraph (rather than a path), where this subgraph is generated using the same procedure. Note that the final path count _b_ then includes the paths from this subgraph as well. 

The process is described in more detail below: 

1. **Sample short paths:** Randomly sample _n_ = 100 000 paths of length _k ∈{_ 2 _,_ 3 _,_ 4 _}_ and compute their composition. Note that this sampling is done with replacement to avoid uniqueness upper bounds for small graphs. 

2. **Generate base graphs:** Generate the desired number of _b_ -path base graphs, by selecting paths that were generated in step 1. Each individual path typically composes to a set of relations, but the graphs are constructed such that the intersection of these sets, across all _b_ paths, produces a singleton target label. 

3. **Recursive edge expansion:** Randomly pick an edge from a path that does not yet have the required length _k_ . Select a path from step 1 which composes to a singleton, corresponding to the relation that is associated with the chosen edge. Replace the edge with this path. 

4. **Recursive subgraph expansion:** Rather than replacing an edge with a path, we can also replace it with a subgraph. As candidate subgraphs, we use the base graphs from step 2 with at most _⌊_ 2 _[b][⌋]_[paths.] 

5. Keep repeating steps 2 and 3 until we have the desired number paths _b_ with the desired length of _k_ , with the restriction that step 3 is applied at most once to each path from the initial base graph. 

Some example graphs generated via this procedure for the RCC-8 dataset are displayed in Figure 4. For higher _k_ , there is greater diversity in the graph topology and complexity of the graph. 

## **A.5.2 Ensuring path consistency within the dataset** 

We ensure that all relational paths in a problem instance in the generated dataset do not informationally conflict with each other by using the `DPC+` algorithm [21]. It efficiently computes directional path consistency, i.e. _Xij ⊆ Xik ⋄ Xkj∀i, j ≤ k_ , for qualitative constraint networks that we can transform our graph instances to. Directional path consistency is sufficient as a test for global path consistency for networks with singleton edge labels [20]. 

## **A.6 Allen’s interval algebra dataset** 

To complement the experiments in the main paper, we also introduce a dataset based on Allen’s interval algebra [2]. Similar to RCC-8, the interval algebra uses 13 primitive relations to describe qualitative temporal relationships. The interval algebra captures all possible relationships between two time intervals. Similar to the RCC-8 relations, these are binary relations defined as follows: _<_ ( _a, b_ ) means that the time interval _a_ precedes the time interval _b_ ; d( _a, b_ ) means that _a_ occurs during _b_ ; o( _a, b_ ) means that _a_ overlaps with _b_ ; m( _a, b_ ) means that _a_ meets _b_ ( _a_ ends exactly before _b_ starts); s( _a, b_ ) means that _a_ starts _b_ ( _a_ and _b_ share a starting time); f( _a, b_ ) means that _a_ finishes _b_ ( _a_ and _b_ share a finishing time); =( _a, b_ ) means that _a_ equals _b_ (the starting and finishing times for both are the same implying the intervals are the same); and finally _>,_ di _,_ oi _,_ mi _,_ si _,_ fi are the inverses of the respective operations defined previously. The composition table for all the primitive interval relations is shown in Table 6 with the exception of the trivial composition of primitive elements with the identity element =. 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0021-00.png)


**----- Start of picture text -----**<br>
0<br>0 0<br>EQ EQ TPPI EQ TPP TPP<br>1 EC 1 TPPI 2 1 TPP 2 3<br>EC TPPI PO TPP TPPI EQ<br>2 3 4<br>(a)  k = 2 , b  = 1 (b)  k = 2 , b  = 2 (c)  k = 2 , b  = 3<br>0 0<br>EQ EQ EQ 0<br>1 1 4 PO EQ TPPI<br>1 4 7<br>EQ EQ EQ<br>EQ EC EQ<br>2 NTPP 2 TPP 5<br>2 EC 5 8<br>EQ EQ TPP<br>EC EQ EQ<br>3 3 6<br>3 6 9<br>NTPP TPP EQ EQ EQ DC<br>4 7 10<br>(d)  k = 4 , b  = 1 (e)  k = 4 , b  = 2 (f)  k = 4 , b  = 3<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0021-01.png)


**----- Start of picture text -----**<br>
0<br>EQ EC<br>1 2 0<br>NTPP EQ EQ<br>EQ EC<br>1 10 22<br>3<br>DC EQ EQ<br>TPPI EQ TPPI<br>2 11 23<br>4 5 6 TPP EC TPPI EQ TPPI EQ NTPPI EQ EC<br>TPP EQ TPP 3 4 5 EC 12 13 14 15 24<br>7 EC TPP EQ TPP EQ NTPP EQ EQ TPPI TPPI<br>6 16 25 27 29<br>TPP<br>PO EQ PO TPPI TPP EQ TPP EC TPPI EQ EQ<br>8<br>7 8 9 17 18 19 20 21 26 28 30<br>EQ<br>EC EC EC EC NTPPI EC EC PO EQ PO TPP<br>9 31<br>(g)  k = 6 , b  = 1 (h)  k = 6 , b  = 3<br>**----- End of picture text -----**<br>


Figure 4: Some graph instances for the RCC-8 dataset generated using the procedure described in A.5.1. The graph topology becomes more diverse for the test instances when sub-graphs are embedded within a single path, as shown in (g) for path length _k −_ 6 and number of paths _b_ = 1. In this particular case, there are two sub-graphs that have been embedded in the graph by replacing two edges. Instances of the type shown in (a), (b), (c), (d), (e), (f) are used in the training set and the graph topology is fixed in this case. The target edge label between the source node and the tail node that needs to be predicted by the model is indicated by the dotted line. 

21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0022-00.png)


**----- Start of picture text -----**<br>
0 0 0<br>= < < f oi ><br>1 = 1 < 2 1 f 2 3<br>= fi f f d m<br>2 3 4<br>(a)  k = 2 , b  = 1 (b)  k = 2 , b  = 2 (c)  k = 2 , b  = 3<br>0<br>0<br>o<br>= m<br>1 0<br>1 4 f d si<br>=<br>f = 1 4 7<br>2 <<br>2 < 5 > f =<br>< 2 > 5 8<br>m m<br>3 m s mi<br>3 6<br>3 6 9<br>si o m<br>si mi oi<br>4 7 10<br>(d)  k = 4 , b  = 1 (e)  k = 4 , b  = 2 (f)  k = 4 , b  = 3<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0022-01.png)


**----- Start of picture text -----**<br>
0<br>f mi fi ><br>1 2 3 4 0<br>oi > > > d > di oi f ><br>fi o f o<br>1 2 3 4 5 11 12 13 14 22<br>5<br>si s = oi mi s d s f ><br>= 6 15 23<br>6 > si fi > > oi f = si di mi =<br>f si 7 8 > 16 17 18 19 24 26 28 30 32<br>7 8 s f o s d f si f f m oi<br>9 20 25 27 29 31 33<br>oi mi<br>= si si o = s fi<br>9<br>10 21 34<br>=<br>= si fi<br>10 35<br>(g)  k = 6 , b  = 1 (h)  k = 6 , b  = 3<br>**----- End of picture text -----**<br>


Figure 5: Graph instances for the interval dataset generated using the procedure described in A.5.1. We highlight the rich variation in the topology of the graph instance for path length _k_ = 6 and number of paths _b_ = 3 in (h) by contrasting it with a similar graph instance for the RCC-8 dataset shown in Figure 4(h). The target edge label between the source node and the tail node that needs to be predicted by the model is indicated by the dotted line. 

22 

Table 6: Allen’s interval algebra composition table [2] excluding the trivial composition with =. 

||_<_|_>_|d|di|o|oi|m|mi|s|si|f|f|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|_<_|_<_||_<,_o,<br>m, d,<br>s|_<_|_<_|_<,_o,<br>m, d,<br>s|_<_|_<,_o,<br>m, d,<br>s|_<_|_<_|_<,_o,<br>m, d,<br>s|_<_|
|_>_||_>_|_>_, oi,<br>mi, d,<br>f|_>_|_>_, oi,<br>mi, d,<br>f|_>_|_>_, oi,<br>mi, d,<br>f|_>_|_>_, oi,<br>mi, d,<br>f|_>_|_>_|_>_|
|d|_<_|_>_|d||_<_, o,<br>m, d,<br>s|_>_, oi,<br>mi, d,<br>f|_<_|_>_|d|_>_, oi,<br>mi, d,<br>f|d|_<_, o,<br>m, d,<br>s|
|di|_<_, o,<br>m, di,<br>f|_>_, oi,<br>di,mi,<br>si|o,<br>oi,<br>d,s,f,<br>di, si,<br>f,=|di|o,<br>di,<br>f|oi, di,<br>si|o,<br>di,<br>f|oi, di,<br>si|o,<br>di,<br>f|di|oi, di,<br>si|di|
|o|_<_|_>_, oi,<br>di,mi,<br>si|o,d,s|_<_, o,<br>m, di,<br>f|_<_, o,<br>m|o,<br>oi,<br>d,s,f,<br>di, si,<br>f,=|_<_|oi, di,<br>si|o|o,<br>di,<br>f|o,d,s|_<_, o,<br>m|
|oi|_<_, o,<br>m, di,<br>f|_>_|oi,<br>d,<br>f|_>_, oi,<br>mi,di,<br>si|o,<br>oi,<br>d,<br>di,<br>s,si,f,<br>f,=|_>_, oi,<br>mi|o,<br>di,<br>f|_>_|oi,<br>d,<br>f|oi, _>_,<br>mi|oi|oi, di,<br>si|
|m|_<_|_>_, oi,<br>di,mi,<br>si|o,d,s|_<_|_<_|o,d,s|_<_|f,<br>f,<br>=|m|m|d,s,o|_<_|
|mi|_<_, o,<br>m, di,<br>f|_>_|oi,<br>d,<br>f|_>_|oi,<br>d,<br>f|_>_|s,<br>si,<br>=|_>_|d,<br>f,<br>oi|_>_|mi|mi|
|s|_<_|_>_|d|_<_, o,<br>m, di,<br>f|_<_, o,<br>m|oi,<br>d,<br>f|_<_|mi|s|s,<br>si,<br>=|d|_<_, m,<br>o|
|si|_<_, o,<br>m, di,<br>f|_>_|oi,<br>d,<br>f|di|o,<br>di,<br>f|oi|o,<br>di,<br>f|mi|s,<br>si,<br>=|si|oi|di|
|f|_<_|_>_|d|_>_, oi,<br>mi,di,<br>si|o,d,s|_>_, oi,<br>mi|m|_>_|d|_>_, oi,<br>mi|f|f,<br>f,<br>=|
|f|_<_|_>_, oi,<br>di,mi,<br>si|o,d,s|di|o|oi, di,<br>si|m|si, oi,<br>di|o|di|f,<br>f,<br>=|f|



To create a dataset for reasoning about interval algebra problems, we follow the same process as for the RCC-8 dataset. Some example graphs generated via this procedure for the are displayed in Figure 5. 

## **A.7 Additional experimental details** 

We now present further details of the experimental set-up, including details of the loss function that was used for training the model, the considered benchmarks and baselines, and training details such as hyperparameter optimisation. 

## **A.7.1 Loss function** 

Let us write **ti** = ( _ti,_ 1 _, . . . , ti,n_ ) for the final-layer embedding of entity _ti_ in the graph _Fi_ , and let **r** = ( _r_ 1 _, . . . , rn_ ) denote the embedding of relation _r_ . Let us write: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0023-07.png)


For each training example _i_ , we write **ti** for the corresponding tail embedding and we let _ri ∈R_ denote the correct label. Then clearly, we want CE( **ti** _,_ **ri** ) to be low, while for each negative example _r[′] ∈R \ {ri}_ we want CE( **ti** _,_ **r** _[′]_ ) to be high. In the base model, we implement this with a margin loss, where for each _i_ we let _ri[′][∈R \ {][r][i][}]_[ be a corresponding negative example:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0023-09.png)


23 

Table 7: Data statistics for different versions of the CLUTRR dataset, with varying training regimes and different numbers training and testing graphs. 

|**Training regime**|**Unique Hash**|**No. of relations**|**# Train**|**# Test**|**Test regime**|
|---|---|---|---|---|---|
|_k ∈{_2_,_3_}_|`data_089907f8`|22|10,094|900|_k ∈{_4_, . . . ,_10_}_|
|_k ∈{_2_,_3_}_|`data_9b2173cf`|22|35,394|39825|_k ∈{_4_, . . . ,_10_}_|
|_k ∈{_2_,_3_,_4_}_|`data_db9b8f04`|22|15,083|823|_k ∈{_5_, . . . ,_10_}_|



Table 8: Data statistics of the RCC-8 dataset 

||**Training regime**|**No. of relations**|**# Train**|**# Test**|**Test regime**|
|---|---|---|---|---|---|
|_b _|_∈{_1_,_2_,_3_}, k ∈{_2_,_3_}_|8|57,600|153,600|_b ∈{_1_,_2_,_3_}, k ∈{_2_, . . . ,_9_}_|



where the margin ∆ _>_ 0 is a hyperparameter. In general, we use _m_ different models, each intuitively capturing a different aspect of the relations. Let CE _j_ ( _ti, r_ ) denote the cross-entropy associated with the embeddings of the _j_[th] model. Then the overall loss function becomes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0024-05.png)


## **A.7.2 Benchmarks** 

**CLUTRR**[2] [33] is a dataset which involves reasoning about family relationships. The original version of the dataset involved narratives describing the fact graph in natural language. It was, among others, aimed at testing the ability of language models such as BERT [13] to solve such reasoning tasks. However, the original paper also considered a number of baselines which were given access to the fact graph itself, especially GNNs and sequence classification models. A crucial finding was that such models fail to learn to reason in a systematic way: models trained on short inference chains perform poorly when tested on examples involving longer inference chains. This has inspired a line of work which has introduced a number of neuro-symbolic methods for addressing this issue. The CLUTRR dataset was released under a CC-BY-NC 4.0 license. 

**GraphLog**[3] [34] involves examples for 57 different _worlds_ , where each world is characterised by a set of logical rules. For each world, a number of corresponding knowledge graphs are provided, which the model can use to learn the underlying rules. The model is then tested on previously unseen knowledge graphs for the same world. The aim of this benchmark is to test the ability of models to systematically generalise from the reasoning patterns that have been observed during training, i.e. to apply the rules that have been learned from the training data in novel ways. This dataset is released under a CC-BY-NC 4.0 license. 

**RCC-8** is the benchmark that we introduce in this paper, as described in Section A.5. We release this benchmark under a CC-BY 4.0 license. 

Dataset statistics for CLUTRR, GraphLog and RCC-8 are reported in tables 7, 9, 8. We use a standard 80-20 split for training and validation for CLUTRR and RCC-8. For GraphLog, we use the validation set that is provided separately from the test set. 

## **A.7.3 Baselines** 

We compare our method against the following neuro-symbolic methods: 

- **CTP** Conditional Theorem Provers [25] are a more efficient version of Neural Theorem Provers (NTPs [29]). Like NTPs, they learn a differentiable logic program, but rather than exhaustively considering all derivations, at each step of a proof, CTPs learn a filter function that selects the most promising rules to apply, thereby speeding up backwards-chaining procedure of NTP. Three variants of this model were proposed, which differ in how this 

> 2 `https://github.com/facebookresearch/clutrr` 

> 3 `https://github.com/facebookresearch/graphlog` 

24 

Table 9: Data statistics for the ‘hard’ GraphLog worlds, showing for each world the number of classes (NC), the number of distinct resolution sequences (ND), the average resolution length (ARL), the average number of nodes (AN), the average number of edges (AE), and the number of training and testing graphs. 

|**World**|**ID**|**NC**|**ND**|**ARL**|**AN**|**AE**|**# Train**|**#Test**|
|---|---|---|---|---|---|---|---|---|
|World|6|16|249|5.06|16.3|20.2|5000|1000|
|World|7|17|288|4.47|13.2|16.3|5000|1000|
|World|8|15|404|5.43|16.0|19.1|5000|1000|
|World|11|17|194|4.29|11.5|13.0|5000|1000|
|World|32|16|287|4.66|16.3|20.9|5000|1000|



selection step is done, i.e. using a linear mapping (CTPL), using an attention mechanism (CTPA), and using a method inspired by key-value memory networks [23] (CTPM). We were not able to reproduce the results from the original paper, hence we report the results from [25] for the CLUTRR benchmark. 

- **GNTP** Greedy NTPs [24] are another approximation of NTPs, which select the top- _k_ best matches during each inference step. 

- **R5** This model [22] learns symbolic rules of the form _r_ ( _X, Z_ ) _← r_ 1( _X, Y_ ) _∧ r_ 2( _Y, Z_ ), with the possibility of using invented predicates in the head. To make a prediction, the method then samples (or enumerates) simple paths between the head and tail entities and iteratively applies the learned rules to reduce these paths to a single relation. The order in which relations are composed is determined by Monte Carlo Tree Search. 

- **NCRL** Neural Compositional Rule Learning [7] also samples relational paths between the head and tail entities, and iteratively reduces them by composing 2 relations at a time, similar to R5. However, in this case, the choice of the two relations to compose in each step are determined by a Recurrent Neural Network. Moreover, rather than learning symbolic rules, the rules are learned implicitly by using an attention mechanism to compose relations. Both R5 and NCRL implicitly make the assumption that the relational reasoning problem is about predicting the target relation from a single relational path, and that this prediction can be done by repeatedly applying Horn rules. 

The following transformer [37] variant is also a natural baseline: 

- **ET** Edge Transformers [5] modify the transformer architecture by using an attention mechanism that is designed to simulate relational composition. In particular, the embeddings are interpreted as representations of edges in a graph. To update the representation of an edge ( _a, c_ ) the model selects pairs of edges ( _a, x_ ) _,_ ( _x, b_ ) and composes their embeddings. These compositions are aggregated using an attention mechanism, similar as in the standard transformer architecture. 

We also compare against several GNN models: 

- **GCN** Graph Convolutional Networks [17] are a standard graph neural network architecture, which use sum pooling and rely on a linear layer followed by a non-linearity such as ReLU or sigmoid to compute messages. While standard GCNs do not take into account edge types, for the experiments we concatenate edge types to node embeddings during message passing, following [33]. GCNs learn node embeddings and can thus not directly be used for relation classification. To make the final prediction, we combine the final-layer embeddings of the head and tail entities with an encoding of the target relation, and make the final prediction using a softmax classification layer. 

- **R-GCN** Relational GCNs [31] are a variant of GCNs in which messages are computed using a relation-specific linear transformation. This is similar in spirit to how we compute messages in our framework, but without the inductive bias that comes from treating embeddings as probability distributions over primitive relation types. 

- **GAT** Graph Attention Networks [38] are a variant of GCNs, which use a pooling mechanism based on attention. Similar as for GCNs, we concatenate the edge types to node embeddings to take into account the edge types. 

25 

Table 10: Optimal hyperparameters of the full (forward-backward) model on all benchmarks. 

||**Batch**|**Embedding**|||**Learning**||
|---|---|---|---|---|---|---|
||**size**|**dim**|**Epochs**|**Facets**|**rate**|**Margin**|
|CLUTRR|128|64|100|8|0.01|1.0|
|GraphLog|64|64|150|1|0.01|1.0|
|RCC-8|64|32|40|4|0.01|1.0|



Table 11: Results on CLUTRR (accuracy) after training on problems with _k ∈{_ 2 _,_ 3 _}_ and then evaluating on problems with _k ∈{_ 4 _, . . . ,_ 10 _}_ . The best performance for each _k_ is highlighted in **bold** . Results marked with _[∗]_ were taken from [25] and those with _[†]_ from [22]. The results from [25] and [22] were evaluated on a different variant of the dataset and may thus not be directly comparable. 

||**4 Hops**|**5 Hops**|**6 Hops**|**7 Hops**|**8 Hops**|**9 Hops**|**10 Hops**|
|---|---|---|---|---|---|---|---|
|FB-`mul`(ours)|0.96_±_.02|0.96_±_.03|0.94_±_.05|0.92_±_.07|0.90_±_.10|0.88_±_.11|0.85_±_.13|
|FB-`min`(ours)|0.96_±_.02|0.95_±_.05|0.91_±_.08|0.87_±_.11|0.82_±_.13|0.79_±_.14|0.74_±_.15|
|R5_†_|0.98_±_.02|**0.99**_±_**.02**|0.98_±_.03|**0.96**_±_**.05**|**0.97**_±_**.01**|**0.98**_±_**.03**|**0.97**_±_**.03**|
|CTP_∗_<br>_L_|0.98_±_.02|0.98_±_.03|0.97_±_.05|0.96_±_.04|0.94_±_.05|0.89_±_.07|0.89_±_.07|
|CTP_∗_<br>_A_|**0.99**_±_**.02**|0.99_±_.01|**0.99**_±_**.02**|0.96_±_.04|0.94_±_.05|0.89_±_.08|0.90_±_.07|
|CTP_∗_<br>_M_|0.97_±_.03|0.97_±_.03|0.96_±_.06|0.95_±_.06|0.93_±_.05|0.90_±_.06|0.89_±_.06|
|GNTP_∗_|0.49_±_.18|0.45_±_.21|0.38_±_.23|0.37_±_.21|0.32_±_.20|0.31_±_.19|0.31_±_.22|
|ET|0.90_±_.04|0.84_±_.02|0.78_±_.02|0.69_±_.03|0.63_±_.05|0.58_±_.06|0.55_±_.08|
|GAT_∗_|0.91_±_.02|0.76_±_.06|0.54_±_.03|0.56_±_.04|0.54_±_.03|0.55_±_.05|0.45_±_.06|
|GCN_∗_|0.84_±_.03|0.68_±_.02|0.53_±_.03|0.47_±_.04|0.42_±_.03|0.45_±_.03|0.39_±_.02|
|NBFNet|0.55_±_.08|0.44_±_.07|0.39_±_.07|0.37_±_.06|0.34_±_.04|0.32_±_.05|0.31_±_.05|
|R-GCN|0.80_±_.09|0.63_±_.08|0.52_±_.11|0.46_±_.07|0.41_±_.05|0.39_±_.06|0.38_±_.05|
|RNN_∗_|0.86_±_.06|0.76_±_.08|0.67_±_.08|0.66_±_.08|0.56_±_.10|0.55_±_.10|0.48_±_.07|
|LSTM_∗_|0.98_±_.04|0.95_±_.03|0.88_±_.05|0.87_±_.04|0.81_±_.07|0.75_±_.10|0.75_±_.09|
|GRU_∗_|0.89_±_.05|0.83_±_.06|0.74_±_.12|0.72_±_.09|0.67_±_.12|0.62_±_.10|0.60_±_.12|



- **E-GAT** Edge-based Graph Attention Networks [34] are a variant of GATs which take edge types into account. In particular, an LSTM module is used to combine the embedding of a neighbouring node with an embedding of the edge type. The resulting vectors are then aggregated as in the GAT architecture. 

- **NBFNet** Neural Bellman-Ford Networks [46] model the relationship between a designated head entity and the other entities from a given graph. Our model employs essentially the same strategy to use GNNs for relation classification, which is to learn entity embeddings that capture the relationship with the head entity rather than the entities themselves. The main difference between NBFnet and our model comes from the additional inductive bias that our model is adding. 

In [25], a number of sequence classifiers were also used as baselines, and we also report these results. These methods sample a path between the head and the tail, encode the path using a recurrent neural network, and then make a prediction with a softmax classification layer. We report results for three types of architectures: vanilla **RNNs** , **LSTMs** [15] and **GRUs** [8]. 

## **A.7.4 Training details** 

The relation vectors **r** and the vectors **a** _ij_ defining the composition function are uniformly initialised. All baseline results that were obtained by us were hyperparameter-tuned using grid search, as detailed below. Some baseline results were obtained from their corresponding papers and reported verbatim (as indicated in the results tables). All experiments were conducted using a single RTX 4090 NVIDIA GPU. A single experiment using the GNN based methods in the paper can be conducted within 30 minutes to an hour on a similar machine. This includes training and testing a single model on any benchmark (CLUTRR, GraphLog, RCC-8). A single hyperparameter set evaluation would take the same time as an individual experiment. 

**Hyperparameter settings** We use the Adam optimizer [16]. The number of layers of the FB model is fixed to 9 and the number of negative examples per instance is fixed as 1. The other hyperparameters 

26 

Table 12: Results for all combinations of the individual ablations from Table 4. 

|**Facet**|**Probability**|**Composition**|**FB**|**CLUTRR**|**CLUTRR**|**RCC-8**|**RCC-8**|
|---|---|---|---|---|---|---|---|
|**Ablation**|**Ablation**|**Ablation**|**Ablation**|**Avg**|_k_ = 10|**Avg**|_b_ = 3_, k_ = 9|
|True|True|True|True|0.06|0.04|0.12|0.12|
|True|True|True|False|0.10|0.15|0.12|0.12|
|True|True|False|True|0.27|0.24|0.25|0.17|
|True|True|False|False|0.20|0.16|0.25|0.14|
|True|False|True|True|0.06|0.04|0.12|0.12|
|True|False|True|False|0.11|0.20|0.12|0.12|
|True|False|False|True|0.92|0.73|0.81|0.49|
|True|False|False|False|0.94|0.85|0.92|0.68|
|False|True|True|True|0.06|0.04|0.22|0.18|
|False|True|True|False|0.11|0.15|0.12|0.12|
|False|True|False|True|0.29|0.25|0.60|0.27|
|False|True|False|False|0.36|0.30|0.38|0.21|
|False|False|True|True|0.08|0.10|0.12|0.12|
|False|False|True|False|0.29|0.31|0.12|0.12|
|False|False|False|True|0.94|0.82|0.84|0.51|
|False|False|False|False|0.99|0.99|0.96|0.80|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0027-02.png)


**----- Start of picture text -----**<br>
rcc8 learned relation vectors<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>1<br>2<br>3<br>r<br>4<br>5<br>6<br>7<br>**----- End of picture text -----**<br>


Figure 6: Schematic visualisation of the learned relation vectors for the RCC-8 benchmark. 

of the FB model are tuned using grid search. The optimal values that were obtained are mentioned in Table 10. 

We conduct the following hyperparameter sweeps: learning rate in _{_ 0 _._ 00001 _,_ 0 _._ 001 _,_ 0 _._ 01 _,_ 0 _._ 1 _}_ , batch size in _{_ 16 _,_ 32 _,_ 64 _,_ 128 _}_ , number of facets _m_ in _{_ 1 _,_ 2 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ and embedding dimension size in _{_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _,_ 256 _}_ . We also tune the margin ∆ in the loss function over _{_ 10 _,_ 1 _,_ 0 _._ 1 _,_ 0 _._ 01 _}_ . All model parameters are shared across the different message passing layers of our model. 

The choice of the pooling operator has an important impact on the systematic generalisation abilities of the model. In our experiments, we found that the pooling operator has to be specified as part of the inductive bias and cannot be learned from the training or validation data, which is in accordance with findings from the literature on systematic generalisation [41, 4]. 

## **A.8 Additional analysis** 

**Additional CLUTRR results** In the main paper, we presented the results for the standard CLUTRR benchmark, where problems of size _k ∈{_ 2 _,_ 3 _,_ 4 _}_ are used for training. In the literature, models are sometimes also evaluated on an even harder setting, where only problems of size _k ∈{_ 2 _,_ 3 _}_ 

27 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0028-00.png)


**----- Start of picture text -----**<br>
CLUTRR learned relation vectors<br>0.8<br>0.6<br>0.4<br>0.2<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>r<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>**----- End of picture text -----**<br>


Figure 7: Schematic visualisation of the learned relation vectors for the CLUTRR benchmark. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0028-02.png)


**----- Start of picture text -----**<br>
Graphlog learned relation vectors<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>r<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>**----- End of picture text -----**<br>


Figure 8: Schematic visualisation of the learned relation vectors for the Graph-Log benchmark. 

are available for training. We show the results for this setting in Table 11. As can be seen, our model clearly outperforms both Edge Transformers (ET) and the GNN and RNN baselines. In this more challenging setting, the difference in performance between our model and ET is much more pronounced. However, R5, as the best-performing neuro-symbolic method, consistently outperforms our method in this case. We hypothesise that this is largely due to the inevitably small size of the training set (as the number of distinct paths of length 3 is necessarily limited). Rule learners can still perform well in such cases, which is something that R5 is able to exploit. To achieve similar results with our model, a stronger inductive bias would have to be imposed. One possibility would be to 

28 

Table 13: Allen’s interval algebra benchmark results (accuracy). Timeout refers to the model taking longer than 30 minutes for inference. Models are trained on graphs with _b ∈{_ 1 _,_ 2 _,_ 3 _}_ paths of length _k ∈{_ 2 _,_ 3 _,_ 4 _}_ . For each _k_ and _b_ , the best performance across all the models is highlighted in **bold** . 

|||||**2 Hops**|**3 Hops**|**4 Hops**|**5 Hops**|**6 Hops**|**7 Hops**|**8 Hops**|**9 Hops**|
|---|---|---|---|---|---|---|---|---|---|---|---|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**0.99**_±_0.0|**0.99**_±_0.0|**0.98**_±_0.0|
|FB-`min`(ours)|_b_|=|2|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**0.97**_±_0.01|**0.97**_±_0.01|**0.94**_±_0.01|**0.93**_±_0.01|**0.89**_±_0.01|
||_b_|=|3|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**0.97**_±_0.01|**0.95**_±_0.01|**0.92**_±_0.01|**0.91**_±_0.01|**0.88**_±_0.02|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|**0.99**_±_0.0|**0.99**_±_0.01|**0.98**_±_0.01|
|FB-`mul`(ours)|_b_|=|2|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.91_±_0.01|0.88_±_0.01|0.83_±_0.01|0.81_±_0.01|0.77_±_0.01|
||_b_|=|3|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.84_±_0.02|0.83_±_0.01|0.78_±_0.01|0.76_±_0.0|0.72_±_0.0|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.92_±_0.01|0.9_±_0.01|0.87_±_0.01|0.85_±_0.01|0.81_±_0.01|
|ET|_b_|=|2|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.86_±_0.01|0.85_±_0.02|0.8_±_0.02|0.77_±_0.02|0.7_±_0.02|
||_b_|=|3|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.85_±_0.01|0.84_±_0.02|0.78_±_0.02|0.73_±_0.03|0.64_±_0.03|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.48_±_0.0|0.49_±_0.0|0.46_±_0.03|0.47_±_0.02|0.45_±_0.03|
|NCRL|_b_|=|2|0.6_±_0.0|0.59_±_0.0|0.59_±_0.01|0.32_±_0.01|0.33_±_0.01|0.3_±_0.01|0.3_±_0.01|0.29_±_0.01|
||_b_|=|3|0.53_±_0.01|0.52_±_0.0|0.53_±_0.01|0.28_±_0.01|0.29_±_0.02|0.27_±_0.02|0.27_±_0.02|0.26_±_0.02|
||_b_|=|1|0.26_±_0.01|0.25_±_0.01|0.24_±_0.01|0.09_±_0.01|0.08_±_0.01|0.07_±_0.01|timeout|timeout|
|R5|_b_|=|2|0.2_±_0.01|0.16_±_0.01|0.15_±_0.01|0.08_±_0.01|0.08_±_0.01|timeout|timeout|timeout|
||_b_|=|3|0.19_±_0.01|0.15_±_0.01|0.14_±_0.01|0.11_±_0.01|0.1_±_0.01|timeout|timeout|timeout|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.78_±_0.19|0.61_±_0.26|0.5_±_0.21|0.47_±_0.2|0.45_±_0.18|
|NBFNet|_b_|=|2|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.67_±_0.23|0.49_±_0.21|0.37_±_0.13|0.34_±_0.11|0.31_±_0.09|
||_b_|=|3|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.61_±_0.29|0.42_±_0.2|0.29_±_0.11|0.26_±_0.08|0.23_±_0.06|
||_b_|=|1|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.77_±_0.09|0.33_±_0.07|0.39_±_0.04|0.38_±_0.02|0.32_±_0.03|
|R-GCN|_b_|=|2|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.72_±_0.08|0.29_±_0.08|0.36_±_0.06|0.33_±_0.04|0.28_±_0.02|
||_b_|=|3|**1.0**_±_0.0|**1.0**_±_0.0|**1.0**_±_0.0|0.7_±_0.06|0.3_±_0.06|0.35_±_0.03|0.31_±_0.03|0.27_±_0.02|



impose a sparsity prior on the relation embeddings **r** and the vectors **a** _ij_ defining the composition function. We leave a detailed investigation of this possibility for future work. 

In the literature, two different variants of the dataset have been used: `db_9b2173cf` and `data_089907f8` . In Table 11, we use the CLUTRR dataset `db_9b2173cf` , which was introduced in the ET paper [5], to evaluate our model as well as the baselines that were evaluated by us. The reported baseline results that were obtained from [25] and [22] are based on the smaller `data_089907f8` variant, and are thus not directly comparable. 

**Allen’s Interval algebra benchmark results** Results are shown in Table 13. The best model is again FB- `min` just like in the case for RCC-8 followed by FB- `mul` and then ET. Results are in line with those obtained for RCC-8 in Table 2. However, we also note that since the interval algebra has more primitive relations and therefore a larger composition table, the benchmark is more challenging than RCC-8. This results in smaller final hop accuracies for all models for _k_ = 9 _, b_ = 3. 

**Extended ablation analysis** In the main paper, we considered four separate ablations. Table 12 extends this analysis by showing results for all combinations of these ablations. Facet ablation refers to the configurations where _m_ = 1; probability ablation refers to the configuration where embeddings are unconstrained; composition ablation refers to the configuration where distmult in combination with an MLP is used as the composition function _ψ_ ; and FB ablation refers to the configuration where we only have the forward model. We can clearly see that the probability and composition function ablation cause a significantly stronger performance degradation compared to the facet and forward-backward ablation. 

**Learned sparseness of relation vectors** We visualise the learned relation vectors for each benchmark studied in this paper in Figures 6, 7 and 8. It can be seen that the vectors are mostly one-hot, despite the fact that no explicit sparsity constraints were used in the model. 

**Effect of the number of message passing rounds on final-hop accuracy** We study the effect of varying the number of message passing rounds on the final _k_ -hop accuracy at _b_ = 3 for the FB- `min` and FB- `mul` models on the RCC-8 and Interval datasets. These models are trained _ab initio_ with the same training data and configuration as before but with a different number of message passing rounds (from 5 to 15) for each instance. The results are displayed in Figure 9. There are three pertinent observations that can be made. Firstly, the maximum attained final _k_ -hop accuracy decreases with _k_ , 

29 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks_images/iclr_2025_Systematic_Relational_Reasoning_With_Epistemic_Graph_Neural_Networks.pdf-0030-00.png)


**----- Start of picture text -----**<br>
Final Accuracy at k=5, b=3 Final Accuracy at k=7, b=3 Final Accuracy at k=9, b=3<br>Final Accuracy at k=6, b=3 Final Accuracy at k=8, b=3<br>Interval Algebra, FB-min Interval Algebra, FB-mul<br>1.0 1.0<br>0.9 0.9<br>0.8 0.8<br>0.7 0.7<br>0.6 0.6<br>0.5 0.5<br>0.4 0.4<br>0.3 0.3<br>6 8 10 12 14 6 8 10 12 14<br>RCC-8, FB-min RCC-8, FB-mul<br>1.0 1.0<br>0.9 0.9<br>0.8 0.8<br>0.7 0.7<br>0.6 0.6<br>0.5 0.5<br>0.4 0.4<br>0.3 0.3<br>6 8 10 12 14 6 8 10 12 14<br># Message passing rounds # Message passing rounds<br>Accuracy<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 9: Effect of the number of message passing rounds on the final _k_ -hop accuracy for the FB model on the Interval and RCC-8 datasets. There is a sharp jump in performance when _k_ equals the number of message passing rounds. 

which makes sense as it confirms that the problem complexity increases with _k_ . Secondly, there is a jump in the final _k_ -hop accuracy when the number of message passing rounds matches _k_ and the accuracies saturate in the number of message passing rounds after that. This rightly suggests that the number of message passing rounds should at least be equal to the final _k_ -hop in the dataset to ensure that all information propagates from source to sink within the model. Thirdly, these observations are shared across the dataset types and aggregation functions. 

30 

