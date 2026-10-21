Published as a conference paper at ICLR 2025 

# HOW FAR ARE WE ON THE DECISION-MAKING OF LLMS? EVALUATING LLMS’ GAMING ABILITY IN MULTI-AGENT ENVIRONMENTS 

**Jen-tse Huang**[1] _[,]_[2] **Eric John Li**[1] **Man Ho Lam**[1] **Tian Liang**[4] _[,]_[2] **Wenxuan Wang**[1] _[,]_[2] _[∗]_ **Youliang Yuan**[3] _[,]_[2] **Wenxiang Jiao**[2] _[∗]_ **Xing Wang**[2] **Zhaopeng Tu**[2] **Michael R. Lyu**[1] 1The Chinese University of Hong Kong 2Tencent AI Lab 

3The Chinese University of Hong Kong, Shenzhen 

4Tsinghua University 

## ABSTRACT 

Decision-making is a complex process requiring diverse abilities, making it an excellent framework for evaluating Large Language Models (LLMs). Researchers have examined LLMs’ decision-making through the lens of _Game Theory_ . However, existing evaluation mainly focus on two-player scenarios where an LLM competes against another. Additionally, previous benchmarks suffer from test set leakage due to their static design. We introduce GAMA( _γ_ )-Bench, a new framework for evaluating LLMs’ Gaming Ability in Multi-Agent environments. It includes eight classical game theory scenarios and a dynamic scoring scheme specially designed to quantitatively assess LLMs’ performance. _γ_ -Bench allows flexible game settings and adapts the scoring system to different game parameters, enabling comprehensive evaluation of robustness, generalizability, and strategies for improvement. Our results indicate that GPT-3.5 demonstrates strong robustness but limited generalizability, which can be enhanced using methods like Chainof-Thought. We also evaluate 13 LLMs from 6 model families, including GPT3.5, GPT-4, Gemini, LLaMA-3.1, Mixtral, and Qwen-2. Gemini-1.5-Pro outperforms others, scoring of 69 _._ 8 out of 100, followed by LLaMA-3.1-70B (65 _._ 9) and Mixtral-8x22B (62 _._ 4). Our code and experimental results are publicly available at https://github.com/CUHK-ARISE/GAMABench. 

## 1 INTRODUCTION 

We have recently witnessed the advancements in artificial intelligence made by Large Language Models (LLMs), which have marked a significant breakthrough in the field. ChatGPT[1] , a leading LLM, has demonstrated its proficiency in a variety of natural language processing tasks, including machine translation (Jiao et al., 2023), sentence revision (Wu et al., 2023), information retrieval (Zhu et al., 2023), and program repair (Surameery & Shakor, 2023). Beyond the academic sphere, LLMs have entered diverse aspects of our everyday life, such as education (Baidoo-Anu & Ansah, 2023), legal service (Guha et al., 2023), product design (Lanzi & Loiacono, 2023), and healthcare (Johnson et al., 2023). Given their extensive capabilities, evaluating LLMs demands more than simple, isolated tasks. A comprehensive and multifaceted approach is highly in demand to assess the efficacy of these advanced models. 

With the broad knowledge encoded in LLMs, their intelligence (Liang et al., 2024), and capabilities in general-purpose task solving (Qin et al., 2023), a question emerges: _Can LLMs assist in everyday decision-making?_ Many real-world decision-making scenarios can be modeled using _Game Theory_ (Koller & Pfeffer, 1997). Furthermore, individuals’ ability to achieve Nash equilibrium (Nash, 1950) reflects their capacity in decision-making (Risse, 2000). Therefore, many studies have drawn on the principles of game theory (Duan et al., 2024; Xie et al., 2024; Xu et al., 2023a), which has several advantages: (1) Scope: Game theory allows for the abstraction of diverse real-life scenarios into simple mathematical models, facilitating a broad range of evaluations. (2) Quantifiability: By 

> _∗_ Wenxiang Jiao and Wenxuan Wang are the corresponding authors. 

> 1https://chat.openai.com/ 

1 

Published as a conference paper at ICLR 2025 

Table 1: Performance (scores) of different LLMs on _γ_ -Bench. 

|_γ_**-Bench Leaderboard**|**GPT-3.5**<br>0613<br>1106<br>0125|**GPT-4**<br>t-0125<br>o-0806|**Gemini-Pro**<br>1.0<br>1.5|
|---|---|---|---|
|Guess 2/3 of the Average<br>El Farol Bar<br>Divide the Dollar|41_._4_±_0_._5<br>68_._5_±_0_._5<br>63_._4_±_3_._4<br>74_._8_±_4_._5<br>64_._3_±_3_._1<br>68_._7_±_2_._7<br>42_._4_±_7_._7<br>70_._3_±_3_._3<br>68_._6_±_2_._4|91_._6_±_0_._6<br>94_._3_±_0_._6<br>23_._0_±_8_._0<br>70_._0_±_22_._1<br>98_._1_±_1_._9<br>95_._2_±_0_._7|77_._3_±_6_._2<br>95_._4_±_0_._5<br>33_._5_±_10_._3<br>37_._2_±_4_._2<br>77_._6_±_3_._6<br>93_._8_±_0_._3|
|Public Goods Game<br>Diner’s Dilemma<br>Sealed-Bid Auction|17_._7_±_1_._7<br>43_._5_±_12_._6<br>38_._9_±_8_._1<br>67_._0_±_4_._9<br>1_._4_±_1_._3<br>2_._8_±_2_._8<br>10_._3_±_0_._2<br>7_._6_±_1_._8<br>13_._0_±_1_._5|89_._2_±_1_._8<br>90_._9_±_3_._0<br>0_._9_±_0_._7<br>10_._7_±_8_._3<br>24_._2_±_1_._1<br>20_._8_±_3_._2|68_._5_±_7_._6<br>100_._0_±_0_._0<br>3_._1_±_1_._5<br>35_._9_±_5_._3<br>31_._6_±_12_._2<br>26_._9_±_9_._4|
|Battle Royale<br>Pirate Game|19_._5_±_7_._7<br>35_._7_±_6_._8<br>28_._6_±_11_._0<br>68_._4_±_19_._9<br>69_._5_±_14_._6<br>71_._6_±_7_._7|86_._8_±_9_._7<br>67_._3_±_14_._8<br>85_._4_±_8_._7<br>84_._4_±_6_._7|16_._5_±_6_._9<br>81_._3_±_7_._7<br>57_._4_±_14_._3<br>87_._9_±_5_._6|
|**Overall**|42_._7_±_2_._0<br>45_._1_±_1_._6<br>44_._4_±_2_._1|62_._4_±_2_._7<br>66_._7_±_4_._7|45_._7_±_3_._4<br>69_._8_±_1_._6|



(a) Closed-source LLMs: Gemini-1.5-Pro outperforms. For GPT-4: t denotes Turbo and o denotes Omni. 

|_γ_**-Bench Leaderboard**|**LLaMA-3.1**<br>8B<br>70B<br>405B|**Mixtral**<br>8x7B<br>8x22B|**Qwen-2**<br>72B|
|---|---|---|---|
|Guess 2/3 of the Average<br>El Farol Bar<br>Divide the Dollar|85_._5_±_3_._0<br>84_._0_±_1_._7<br>94_._3_±_0_._6<br>75_._7_±_2_._2<br>59_._7_±_3_._5<br>20_._5_±_24_._2<br>56_._4_±_8_._4<br>87_._0_±_4_._1<br>94_._9_±_1_._0|91_._8_±_0_._4<br>83_._6_±_4_._6<br>66_._8_±_5_._8<br>39_._3_±_12_._2<br>1_._2_±_2_._8<br>79_._0_±_9_._6|93_._2_±_1_._3<br>17_._0_±_25_._5<br>91_._9_±_2_._4|
|Public Goods Game<br>Diner’s Dilemma<br>Sealed-Bid Auction|19_._6_±_1_._0<br>90_._6_±_3_._6<br>97_._0_±_0_._8<br>59_._3_±_2_._4<br>48_._1_±_5_._7<br>14_._4_±_4_._5<br>37_._1_±_3_._1<br>15_._7_±_2_._7<br>14_._7_±_3_._2|27_._6_±_11_._7<br>83_._7_±_3_._5<br>76_._4_±_7_._1<br>79_._9_±_5_._8<br>3_._1_±_1_._6<br>13_._2_±_3_._7|81_._3_±_5_._9<br>0_._0_±_0_._0<br>2_._5_±_0_._7|
|Battle Royale<br>Pirate Game|35_._9_±_12_._1<br>77_._7_±_26_._0<br>92_._7_±_10_._1<br>78_._3_±_10_._0<br>64_._0_±_15_._5<br>65_._6_±_22_._3|12_._6_±_9_._4<br>36_._0_±_21_._0<br>67_._3_±_7_._6<br>84_._3_±_8_._8|81_._7_±_9_._6<br>86_._1_±_6_._4|
|**Overall**|56_._0_±_3_._1<br>65_._9_±_3_._3<br>61_._8_±_4_._7|43_._4_±_2_._2<br>62_._4_±_2_._2|56_._7_±_3_._4|



(b) Open-source LLMs: LLaMA-3.1-70B outperforms. 

examining the Nash equilibrium within these models, we gain a measurable metric for comparing LLMs’ decision-making performance. (3) Variability: The adjustable parameters of these models enable the creation of variant scenarios, enhancing the diversity and robustness of our assessments. However, existing research is often limited to two-player or two-action settings, such as the classical Prisoner’s Dilemma and Ultimatum Game (Guo, 2023; Phelps & Russell, 2023; Akata et al., 2023; Aher et al., 2023; Brookins & DeBacker, 2024). Moreover, prior work relies on fixed, classical game settings, increasing the likelihood that LLMs have encountered these scenarios during training, facing the risk of test set leakage. In this paper, we assess LLMs in more complex scenarios involving multiple players, actions, and rounds, across classical game theory scenarios with dynamically adjustable game parameters. 

We include eights games and divide them into three categories based on their characteristics. The first category in our framework evaluates LLMs’ ability to make optimal decisions by understanding game rules and recognizing patterns in other players’ behavior. A distinctive characteristic of these games is that individual players cannot achieve higher gains without cooperation, provided that other participants cooperate. Essentially, these games’ Nash equilibrium aligns with maximizing overall social welfare. We name such games as **I. Cooperative Games** , including (1) _Guess 2/3 of the Average_ , (2) _El Farol Bar_ , and (3) _Divide the Dollar_ . The second category assesses the propensity of LLMs to prioritize self-interest, potentially betraying others for greater gains. In contrast to the first category, games in this category incentivize higher rewards for participants who betray their cooperative counterparts. Typically, the Nash equilibrium in these games leads to reduced social welfare. This category is termed **II. Betraying Games** , including (4) _Public Goods Game_ , (5) _Diner’s Dilemma_ , (6) _Sealed-Bid Auction_ . Last but not least, we focus specifically on two games characterized by sequential decision-making processes, distinguishing them from the previous six games based on simultaneous decision-making. **III. Sequential Games** are the (7) _Battle Royale_ and (8) _Pirate Game_ . 

Decision-making is a complex task requiring various abilities. Several common ones are evaluated across all games: (1) Perception: the ability to understand situations, environments, and rules, and extends to long-text understanding for LLMs. (2) Arithmetic Reasoning: the ability to quantify real-world options and perform calculations. (3) ToM Reasoning: the Theory of Mind (Kosinski, 

2 

Published as a conference paper at ICLR 2025 

2023; Bubeck et al., 2023; Huang et al., 2024a) refers to the ability to infer others’ intentions and beliefs. (4) Strategic Reasoning: the ability to integrate all available information to arrive at the best decision. Certain games involve specialized abilities, such as K-level reasoning in the “Guess 2/3 of the Average” game and mixed strategy adoption in the “El Farol Bar” game. 

In this paper, we instruct ten agents, based on the GPT-3.5 (0125) model, to engage in the eight games, followed by an analysis of the results obtained. Subsequently, we assess the model’s robustness against multiple runs, temperature parameter alterations, and prompt template variations. Further exploration is conducted to ascertain if instructional prompts, such as Chain-of-Thought (CoT) (Kojima et al., 2022), enhance the model’s decision-making capabilities. Additionally, the model’s capacity to generalize across diverse game settings is examined. Finally, we evaluate the performance of **thirteen** LLMs, including GPT-3.5-Turbo (0613, 1106, 0125) (OpenAI, 2022), GPT4 (Turbo-0125, 4o-0806) (OpenAI, 2023), Gemini-1.0-Pro (Pichai & Hassabis, 2023), Gemini-1.5Pro (Pichai & Hassabis, 2024), LLaMA-3.1 (8B, 70B, 405B) (Dubey et al., 2024), Mixtral (8x7B, 8x22B) (Jiang et al., 2024), and Qwen-2-72B (Yang et al., 2024). We compare the performance of different LLMs by creating multiple agents from the same model to participate in the games, then calculate the average performance of these agents. Our contributions include: 

- We provide a comprehensive review and comparison of existing literature on evaluating LLMs using game theory scenarios, as summarized in Table 3. The review includes key aspects such as models, games, temperature settings, and other game parameters, highlighting our emphasis on the multi-player setting and the generalizability of LLMs. 

- Starting from the multi-player setting, we collect eight classical game theory scenarios to measure LLMs’ Gaming Ability in Multi-Agent environments, and implement our framework, GAMA( _γ_ )- Bench. It enables dynamic game scene generation with diverse profiles, offering unlimited scenarios to assess LLM generalizability while minimizing test set leakage risk. 

- We apply _γ_ -Bench to thirteen LLMs to provide an in-depth analysis of their performance in multiagent gaming scenarios, indicating their potential as assistants in decision-making process. 

## 2 INTRODUCTION TO GAMES 

We collect eight games well studied in Game Theory and propose _γ_ -Bench, a framework with multiplayer, multi-round, and multi-action settings. Notably, _γ_ -Bench allows the simultaneous participation of both LLMs and humans, enabling us to evaluate LLMs’ performance when playing against humans or fixed strategies. This section details each game with their classical settings (parameters). 

## 2.1 COOPERATIVE GAMES 

**(1) Guess 2/3 of the Average** Initially introduced by Ledoux (1981), the game involves players independently selecting an integer between 0 and 100 (inclusive). The winner is the player(s) choosing the number closest to two-thirds of the group’s average. A typical initial strategy might lead players to assume an average of 50, suggesting a winning number around 50 _×_[2] 3 _[≈]_[33][.][However,][if] all participants adopt this reasoning, the average shifts to 33, thereby altering the winning number to approximately 22. The game has a Pure Strategy Nash Equilibrium (PSNE) where all players selecting zero results in a collective win. 

**(2) El Farol Bar** Proposed by Arthur (1994) and Huberman (1988), this game requires players to decide to either visit a bar for entertainment or stay home without communication. The bar, however, has a limited capacity and can only accommodate part of the population. In a classical scenario, the bar becomes overcrowded and less enjoyable if more than 60% of the population decides to go there. Conversely, if 60% or fewer people are present, the experience is more enjoyable than staying home. Imagine that if everyone adopts the same pure strategy, _i.e._ , either everyone going to the bar or everyone staying home, then the social welfare is not maximized. Notably, the game lacks a PSNE but presents an Mixed Strategy Nash Equilibrium (MSNE), where the optimal strategy involves going to the bar with a 60% probability and staying home with a 40% probability. 

**(3) Divide the Dollar** Firstly mentioned in Shapley & Shubik (1969), the game involves two players independently bidding up to 100 cents for a dollar. Ashlock & Greenwood (2016) further 

3 

Published as a conference paper at ICLR 2025 

generalized the game into a multi-player setting. If the sum of bids is at most one dollar, each player is awarded their respective bid; if the total exceeds a dollar, no player receives anything. The NE of this game occurs when each player bids exactly[100] _N_[cents.] 

## 2.2 BETRAYING GAMES 

**(4) Public Goods Game** Studied since the early 1950s (Samuelson, 1954), the game requires _N_ players to secretly decide how many of their private tokens to contribute to a public pot. The tokens in the pot are then multiplied by a factor _R_ (1 _< R < N_ ), and the resulting “public good” is evenly distributed among all players. Players retain any tokens they do not contribute. A simple calculation reveals that for each token a player contributes, their net gain is _N[R][−]_[1][,][which][is][less][than][zero.] This suggests that the rational strategy for each player is to contribute no tokens, which reaches an NE of this game. The game serves as a tool to investigate tendencies towards selfish behavior and free-riding among participants. 

**(5) Diner’s Dilemma** This game is the multi-player variant of the _Prisoner’s Dilemma_ (Glance & Huberman, 1994). The game involves _N_ players dining together, with their decision to split all the costs. Each player needs to independently choose whether to order the expensive or the cheap dish, priced at _x_ and _y_ ( _x > y_ ), respectively. The expensive offers _a_ utility per individual, surpassing the _b_ utility of another choice ( _a > b_ ). The game satisfies two assumptions: (1) _a − x < b − y_ : Although the expensive dish provides a greater utility, the benefit does not justify its higher cost, leading to a preference for the cheap one when dining alone. (2) _a − N[x][>][b][ −] N[y]_[:][Individuals][are][inclined] to choose the expensive dish when the cost is shared among all diners. The assumptions lead to an NE where all players opt for the more expensive meal. However, this PSNE results in a lower total social welfare of _N_ ( _a − x_ ) compared to _N_ ( _b − y_ ), which is the utility if all choose the cheap one. This game evaluates the long-term perspective and the capacity to establish sustained cooperation. 

**(6) Sealed-Bid Auction** The _Sealed-Bid Auction_ (SBA) involves players submitting their bids confidentially and simultaneously, different from the auctions where bids are made openly in a sequential manner. We consider two variants of SBA: the _First-Price Sealed-Bid Auction_ (FPSBA) and the _Second-Price Sealed-Bid Auction_ (SPSBA). In FPSBA, also known as the _Blind Auction_ , if all players bid their true valuation _vi_ of the item, the winner achieves a net gain of _bi − vi_ = 0 while others also gain nothing (McAfee & McMillan, 1987). Moreover, the highest bidder will discover that to win the auction, it is sufficient to bid marginally above the second-highest bid. Driven by these two factors, FPSBA is often deemed inefficient in practical scenarios, as bidders are inclined to submit bids significantly lower than their actual valuation, resulting in suboptimal social welfare. In contrast, SPSBA, commonly called the Vickrey auction, requires the winner to pay the second-highest bid, encouraging truthful bidding by all players (Vickrey, 1961). It can be proven that bidding true valuations in SPSBA represents an NE. This auction evaluates agent performance in imperfect information games, where agents lack knowledge of other players’ valuations. 

## 2.3 SEQUENTIAL GAMES 

**(7) Battle Royale** Extended from the _Truel_ (Kilgour, 1975) involving three players, the _Battle Royale_ involves _N_ players shooting at each other. In the widely studied form (Kilgour & Brams, 1997), players have different probabilities of hitting the target, with the turn order set by increasing hit probabilities. The game allows for unlimited bullets and the tactical option of intentionally missing shots. The objective for each participant is to emerge as the sole survivor, with the game ending when only one player remains. While the NE has been identified for infinite sequential truels (Kilgour, 1977), the complexity of these equilibria escalates exponentially with an increased number of players. 

**(8) Pirate Game** This game is a multi-player version of the _Ultimatum Game_ (Goodin, 1998; Stewart, 1999). Each player is assigned a “pirate rank”, determining their action order. The game involves _N_ pirates discussing the division of _G_ golds they have discovered. The most senior pirate first proposes a distribution method. If the proposal is approved by at least half of the pirates, including the proposer, the game ends, and the gold is distributed as proposed. Otherwise, the most senior pirate is thrown overboard, and the next in rank assumes the proposer role until the game 

4 

Published as a conference paper at ICLR 2025 

ends. Each pirate’s objectives are prioritized as (1) survival, (2) maximizing their share of gold, and (3) the opportunity to eliminate others from the game. Stewart (1999) identifies the optimal strategy, where the most senior pirate allocates one gold to each odd-ranked pirate and keeps the remainder. 

## 3 GAMA-BENCH SCORING SCHEME 

This section presents experiments conducted using the default settings for each game on the GPT-3.5 (0125) model. Utilizing this model as a case study, we illustrate our methodology for benchmarking an LLM with _γ_ -Bench. The prompt and its design method can be found in §C in the appendix. Each game involves ten agents based on GPT-3.5, with the temperature parameter set to one. For simultaneous games, there will be twenty rounds. We run each game five times to enhance the reliability of our findings and mitigate the impact of variance. For clarity and conciseness, this section presents one of the five runs while §4.1 details quantitative results. Our findings of GPT3.5’s behaviors on _γ_ -Bench include: 

## **Key Findings:** 

- The model’s decisions are mainly influenced by the outcomes of the preceding round rather than deriving from the reasoning of the optimal strategy. 

- Although initially demonstrating suboptimal performance, the model can learn from historical data and enhance its performance over time. A larger fluctuation is observed in games that are difficult to optimize from historical data, such as the El Farol Bar game. 

- The model demonstrates the ability to engage in spontaneous cooperation, leading to increased social welfare beyond mere self-interest, without the necessity for explicit communication. However, this phenomenon also results in low performance in Betraying Games. 

- The model shows limitations in sequential games with more complicated rules. 

- The aggregate score of the model on _γ_ -Bench is 45 _._ 9. 

## 3.1 COOPERATIVE GAMES 

**(1) Guess 2/3 of the Average** [TO PROMPT] The vanilla setting for this game is _MIN_ = 0, _MAX_ = 100, and _R_ =[2] 3[.][We show the choices made by all agents as well as the average and the] winning numbers in Fig. 1(1). Key observations are: (1) In the first round, agents consistently select 50 (or close to 50), corresponding to the mean of a uniform distribution ranging from 0 to 100. This behavior suggests that the model fails to recognize that the winning number is[2] 3[of the average.][(2)] As rounds progress, the average number selected decreases noticeably, demonstrating that agents are capable of adapting based on historical outcomes. Since the optimal strategy is to choose the _MIN_ , the score in this game is given by _S_ 1 = _NK_ 1 � _ij_[(] _[C][ij][−][MIN]_[)][,][where] _[C][ij]_[is][the][chosen] number of player _i_ in round _j_ . The model scores[2] 65 _._ 4 on this game. 

**(2) El Farol Bar** [TO PROMPT] The vanilla setting for this game is _MIN_ = 0, _MAX_ = 10, _HOME_ = 5, and _R_ = 60%. To explore the influence of incomplete information, we introduce two settings: _Explicit_ indicates that everyone can see the results at the end of each round, while _Implicit_ indicates that those staying at home cannot know what happened in the bar after the round ends. Fig. 1(2) illustrates the probability of agents deciding to go to the bar and the total number of players in the bar. We find that: (1) In the first round, there is an inclination among agents to visit the bar. Observations of overcrowding lead to a preference for staying home, resulting in fluctuations shown in both Fig. 1(2-1) and Fig. 1(2-2). In the Implicit setting, due to the lack of direct observations of the bar’s occupancy, agents require additional rounds (Rounds 2 to 6) to discern the availability of space in the bar. (2) The probability of agents going to the bar gradually stabilizes, with the average probability in the Implicit setting being lower than in the Explicit setting. Since the optimal strategy is to choose the go with a probability of _R_ , the raw score[3] in this game is given by 

> 2 For clarity, we normalize raw scores to the range of [0 _,_ 100], with higher values indicating a better performance. The method used for rescaling is detailed in §E of the appendix. 

> 3For simplicity, we evaluate only the Implicit setting. 

5 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0006-01.png)


**----- Start of picture text -----**<br>
(1) Guess 2/3 of the Average (2-1) El Farol Bar-Explicit (2-2) El Farol Bar-Implicit (3) Divide the Dollar<br>Average Number and Winning Number Number of Players in the Bar Number of Players in the Bar Average Proposal<br>(4) Public Goods Game (5) Diner's Dilemma (6-1) Sealed-Bid Auction-First Price (6-2) Sealed-Bid Auction-Second Price<br>Average Contribution and Return Number of Players Choosing the Cheap Dish Average of (Valuation - Bid) / Valuation Average of Valuation Minus Bid<br>**----- End of picture text -----**<br>


Figure 1: Performance of GPT-3.5 (0125) in Cooperative and Betraying games. _S_ 2 = _K_ 1 � _j[|] N_[1] � _i[D][ij][−][R][|]_[,][where] _[ D][ij]_[=][1][ when player] _[ i]_[ chose to go in round] _[ j]_[and] _[ D][ij]_[=][0] when player _i_ chose to stay. The model scores 73 _._ 3 on this game. 

**(3) Divide the Dollar** [TO PROMPT] The vanilla setting for this game is _G_ = 100. We plot the proposals by all agents and the sum of their proposals in Fig. 1(3). Our analysis reveals the following insights: (1) In the first round, agents’ decisions align with the NE predictions of the game. However, after gaining golds, agents exhibit increased greed, proposing allocations that exceed the NE-prescribed amounts. Upon receiving nothing, they tend to propose a “safer” amount. The trend continues and causes fluctuations across subsequent rounds. (2) Despite these fluctuations, the average of proposed golds converges to approximately 100. Since the optimal strategy is to propose _G/N_ , the raw score in this game is given by _S_ 3 = _K_ 1 � _j[|]_[�] _i[B][ij][−][G][|]_[, where] _[ B][ij]_[is the proposed] amount number of player _i_ in round _j_ . The model scores 68 _._ 1 on this game. 

## 3.2 BETRAYING GAMES 

**(4) Public Goods Game** [TO PROMPT] The vanilla setting for this game is _R_ = 2. Each player has _T_ = 20 to contribute in each round. Fig. 1(4) shows the contributed tokens by each agent and their corresponding gains per round. The observations reveal the following: (1) Despite an investment return of _−_ 80%, agents display a pattern of alternating between free-riding and contributing all their tokens. (2) As the rounds progress, there is an evident increase in the number of tokens contributed to the public pot, leading to an overall enhancement in social welfare gains. These findings suggest that the LLM exhibits cooperative behavior, prioritizing collective benefits over individual self-interest. Since we expect the model to infer the optimal strategy, _i.e._ , contributing zero tokens, the raw score in this game is given by _S_ 4 = _NK_ 1 � _ij[C][ij]_[,][where] _[ C][ij]_[is the proposed contribution] amount of player _i_ in round _j_ . The model scores 41 _._ 2 on this game. 

**(5) Diner’s Dilemma** [TO PROMPT] The vanilla setting for this game is _Ph_ = 20, _Pl_ = 10, _Uh_ = 20, _Ul_ = 15. We show the probability of agents choosing the costly dish, their resulting utilities, and the average bill in Fig. 1(5). Analysis of the figure reveals the following insights: (1) Contrary to the NE predictions for this game, agents predominantly prefer the cheap dish, which maximizes total social welfare. (2) Remarkably, a deviation from cooperative behavior is observed wherein one agent consistently chooses to betray others, thereby securing a higher utility. This pattern of betrayal by this agent persists across subsequent rounds. Since we expect the model to infer the the optimal strategy, _i.e._ , choosing the costly dish, the raw score in this game is given by _S_ 5 = _NK_ 1 � _ij[D][ij]_[,][where] _[D][ij]_[=][1][when][player] _[i]_[chose][the][cheap][dish][in][round] _[j]_[and] _[D][ij]_[=][0] when player _i_ chose the costly dish. The model scores 4 _._ 0 on this game. 

**(6) Sealed-Bid Auction** [TO PROMPT] For the vanilla setting in this game, we randomly assign valuations to each agent in each round, ranging from 0 to 200. We fix the seed for random number generation to ensure fair comparisons across various settings and models. We evaluate LLMs’ performance under both _First-Price_ and _Second-Price_ settings. Fig. 1(6) depicts the subtraction 

6 

Published as a conference paper at ICLR 2025 

between valuations and bids and bid amounts of each agent. Our key findings include: (1) As introduced in §2.2, we note that agents generally submit bids that are lower than their valuations in the First-Price auction, a tendency indicated by the positive discrepancies between valuations and bids depicted in Fig. 1(6-1). (2) Though the NE suggests that everyone bids the amount of their valuation in the Second-Price setting, we find a propensity for bidding below valuation levels, as demonstrated in Fig. 1(6-2). Since the optimal strategy is to bid the prices lower than their true valuations,[4] the raw score in this game is given by _S_ 6 = _NK_ 1 � _ij vijv−ijbij_ , where _vij_ and _bij_ are player _i_ ’s valuation and bid in round _j_ , respectively. The model scores 14 _._ 6 on this game. 

3.3 SEQUENTIAL GAMES 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0007-03.png)


**----- Start of picture text -----**<br>
(a) Player Decision and Outcome (b) Probability of Player Targeting High Hit Rate<br>**----- End of picture text -----**<br>


Figure 2: GPT-3.5 (0125)’s performance in “Battle Royale.” (a): Agents’ actions and outcomes of each round. For example, in round 11, player 6 shot at player 7 but missed. 

**(7) Battle Royale** [TO PROMPT] For the vanilla setting in this game, we assign varied hit rates to each agent, spanning from 35% to 80% in increments of 5%. This setting covers a broad spectrum of hit rates, avoiding extremes of 0% or 100%. Fig. 2 illustrates the actions and outcomes of each round, along with the tally of participants remaining. Our observations reveal: (1) Unlike our expectations, agents rarely target the player with the highest hit rate. (2) Agents neglect to utilize the strategy of “intentionally missing.” For example, in round 19, with players 7, 8, and 10 remaining, it was player 7’s turn to act. The optimal strategy for player 7 would have been to intentionally miss the shot, thereby coaxing player 8 into eliminating player 10, enabling player 7 to target player 8 in the following round for a potential victory. Instead, player 7 opted to target player 10, resulting in player 8 firing upon itself. For simplicity, we evaluate whether agents target the player with the highest hit rate (excluding themselves). Therefore, the raw score in this game is given by _S_ 7 = _Nk_ 1 � _ij[I][ij]_[,] where _k_ represents the number of rounds played and _Iij_ = 1 if player _i_ targets the player with the highest hit rate in round _j_ , and _Iij_ = 0 otherwise. The model scores 20 _._ 0 on this game. 

Table 2: Performance of GPT-3.5 (0125) in the “Pirate Game.” Each row shows the proposed gold distribution in the specific round and whether each pirate accepts (“✓”) or rejects (“✗”) the proposal. _S_ 8 _P_ shows the score of the proposer while _S_ 8 _V_ shows the score of all voters. 

|**Pirate Rank**<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>_S_8_P_<br>_S_8_V_|**Pirate Rank**<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>_S_8_P_<br>_S_8_V_|
|---|---|
|**Round 1**<br>100✓<br>0✗<br>0✗<br>0✗<br>0✗<br>0✗<br>0✗<br>0✗<br>0✗<br>0✗<br>**Round 2**<br>-<br>99✓<br>0✗<br>1✓<br>0✓<br>0✗<br>0✗<br>0✗<br>0✗<br>0✓<br>**Round 3**<br>-<br>-<br>50✓<br>1✓<br>1✓<br>1✓<br>1✓<br>1✓<br>1✓<br>44✓|8<br>1_._00<br>6<br>0_._75<br>94<br>0_._57|



**(8) Pirate Game** [TO PROMPT] The vanilla setting for this game is _G_ = 100. As introduced in §2.3, the optimal strategy for the first proposer is to allocate 96 golds to itself and one gold each to the third, fifth, seventh, and ninth pirates. Stewart (1999) has elucidated the optimal strategy for voters: (1) accept if allocated two or more golds; (2) reject if no golds are allocated; (3) accept if one gold is allocated and it shares the same parity as the proposer, otherwise, reject. Table 2 presents a sample game’s proposals and voting results. The key conclusion is that agents fail to propose optimal proposals and frequently cast incorrect votes, suggesting that the LLM demonstrates suboptimal performance in this game. Two aspects are considered to comprehensively evaluate a model’s performance: (1) whether proposers give a reasonable proposal and (2) whether voters act correctly towards a given proposal. For (1), we calculate the _L_ 1 norm between the given proposal and the optimal strategy, defined as _S_ 8 _P_ = _k_ 1 � _j[∥][P][j][−][O][j][∥]_[1][,][where] _[P][j]_[represents][the][model’s] 

> 4We evaluate only the First-Price setting according to the definition of Betraying Games. 

7 

Published as a conference paper at ICLR 2025 

proposal and _Oj_ denotes the optimal proposal in round _j_ , with the game ending at round _k_ . For (2), we calculate the accuracy of choosing the right action elucidated above, which is: _S_ 8 _V_ = _k_ (2 _N −_ 2 _k−_ 1) � _ij[I][ij]_[,][where] _[I][ij]_[=][1][if][player] _[i]_[votes][correctly][in][round] _[j]_[and] _[I][ij]_[=][0][otherwise,] excluding the proposer from the calculation. The model scores 80 _._ 6 on this game. 

## 4 BEYOND DEFAULT SETTINGS 

This section explores deeper into several following Research Questions (RQs). **RQ1 Robustness** : Is there a significant variance in multiple runs? Is the performance sensitive to different temperatures and prompt templates? **RQ2 Reasoning Strategies** : Are strategies to enhance reasoning skills applicable to game scenarios? This includes implementing Chain-of-Thought (CoT) (Kojima et al., 2022; Wei et al., 2022) reasoning and assigning unique personas to LLMs. **RQ3 Generalizability** : How does LLM performance vary with different game settings? Do LLMs remember answers learned during the training phase? **RQ4 Leaderboard** : How do various LLMs perform on _γ_ -Bench? Unless otherwise specified, we apply the vanilla settings described in §3. 

## 4.1 RQ1: ROBUSTNESS 

This RQ examines the stability of LLMs’ responses, assessing the impact of three critical factors on model performance: (1) randomness introduced by the model’s sampling strategy, (2) the temperature parameter setting, and (3) the prompt used for game instruction. 

**Multiple Runs** Firstly, we run all games five times under the same settings. Fig. 4 illustrates the average performance across tests, while Table 4 lists the corresponding scores. The analysis reveals that, except for the two sequential games and the “Public Goods Game,” the model demonstrates a consistent performance, as evidenced by the low variance in scores for each game. 

**Temperatures** As discussed in our literature review in §B, prior research incorporates varying temperature parameters from 0 to 1 yet omits to explore their impacts. This study conducts experiments across games employing a range of temperatures _{_ 0 _._ 0 _,_ 0 _._ 2 _,_ 0 _._ 4 _,_ 0 _._ 6 _,_ 0 _._ 8 _,_ 1 _._ 0 _}_ under vanilla settings. The results, both visual and quantitative, are documented in Fig. 5 and Table 5, respectively. The small overall variance of 3 _._ 4 indicates that, for the majority of games, temperature adjustments yield negligible effects. A notable exception is observed in “Guess 2/3 of the Average,” where increased temperatures correlate with enhanced scores (48 _._ 0 to 65 _._ 4), contrasting starkly with the near-random performance at zero temperature. 

**Prompt Templates** We also investigate the impact of prompt phrasing on model performance. We leverage GPT-4 to rewrite our default prompt templates, generating four additional versions. We perform a manual checking process on the generated versions to ensure GPT-4’s adherence to game rules without altering critical data. The prompt templates can be found in §D. We plot the results of using these templates in Fig. 6 and record the quantitative scores in Table 6. Notably, we find that prompt wording can significantly affect performance, as shown by the high variances in the “Public Goods Game” (11 _._ 5), “Diner’s Dilemma” (23 _._ 7), and “Pirate Game” (14 _._ 7). 

**Answer to RQ1:** GPT-3.5 exhibits consistency in multiple runs and shows robustness against different temperature settings. However, inappropriate prompt designs resulting from potential misinformation during rephrasing can significantly impair performance. 

## 4.2 RQ2: REASONING STRATEGIES 

This RQ focuses on improving the model’s performance through prompt instructions. We investigate two strategies: Chain-of-Thought (CoT) prompting (Kojima et al., 2022) and persona assignment (Kong et al., 2024). We show the visualized and quantitative results in Fig. 7 and Table 7. 

**CoT** According to Kojima et al. (2022), introducing a preliminary phrase, “Let’s think step by step,” encourages the model to sequentially analyze and explain its reasoning before presenting its conclusion. This approach has proven beneficial in specific scenarios, such as games (1), (3), (4), 

8 

Published as a conference paper at ICLR 2025 

and (5), improving the overall score from 45 _._ 9 to 57 _._ 9, by 12 _._ 0. In the “(3) Divide the Dollar” game, incorporating CoT reduces the model’s propensity to suggest disproportionately large allocations, increasing the score by 15 _._ 3. Similarly, in the “(4) Public Goods Game” and “(5) Diner’s Dilemma,” CoT prompts the model to recognize being a free-rider as the optimal strategy, increasing the scores by 14 _._ 9 and 78 _._ 5, respectively. 

**Persona** Studies (Kong et al., 2024; Huang et al., 2024b) have demonstrated that assigning roles to models influences performance across various downstream tasks. Inspired by this discovery, our study initiates with a prompt that specifies the model’s role, such as “You are [ROLE],” where the role could be a cooperative and collaborative assistant, a selfish and greedy assistant, or a mathematician. Our findings reveal that assigning the “cooperative” role enhances model performance in games (1), (2), and (3), notably outperforming the CoT method in the “(3) El Farol Bar” game. Conversely, the “selfish” role markedly diminishes performance almost all the games, with the only exception of the “(7) Battle Royale” game. The “mathematician” role improves the model’s overall score by 0 _._ 6, which is small and does not surpass the CoT method’s effectiveness. 

**Answer to RQ2:** It is possible to improve GPT-3.5 through simple prompt instructions. Among the methods we explore, the CoT prompting performs the best, achieving a performance close to GPT-4 (57 _._ 9 vs. 62 _._ 4). 

## 4.3 RQ3: GENERALIZABILITY 

Considering the extensive exploration of games in domains such as mathematics, economics, and computer science, it is probable that the vanilla settings of these games are included within the training datasets of LLMs. To ascertain the presence of data contamination in our chosen games, we subjected them to various settings. The specifics of the parameters selected for each game are detailed in Table 8, and the experimental outcomes are visually represented in Fig. 8. Our findings indicate variability in model generalizability across different games. Specifically, in games (1), (3), (5), (6), and (8), the model demonstrated correct performance under diverse settings. In the “(3) Divide the Dollar” game, the model’s performance improved with an increase in total golds ( _G_ ), suggesting that higher allocations of golds satisfy the demands of all players. Conversely, the model exhibited low generalizability in games (2) and (4). An analysis of the game “(2) El Farol Bar” reveals a consistent decision-making pattern by the model, opting to participate with approximately a 50% probability regardless of varying bar capacities ( _R_ ), indicating that the model is acting randomly. Similarly, in the “(4) Public Goods Game,” the model consistently contributes similar amounts, even when the return rate is nil, indicating a lack of understanding of the game rules. A possible reason for this poor performance is the model’s inability to adjust its performance incrementally based on historical data. 

Nagel (1995) conducted experiments with 15 to 18 human subjects participating in the “(1) Guess 2/3 of the Average” game, using ratios of[1] 2[,][2] 3[,][and][4] 3[.][The][average][numbers][were][27] _[.]_[05][,][36] _[.]_[73][,] and 60 _._ 12 for each ratio, respectively. In a similar vein, Rubinstein (2007) explored the 3[2][ratio on] a larger population involving 2,423 subjects, yielding a comparable mean of 36 _._ 2, aligning with the finding in Nagel (1995). The model produces average numbers of 34 _._ 59, 34 _._ 59, and 74 _._ 92 for the same ratios, indicating its predictions are more aligned with human behavior than the game’s NE. **Answer to RQ3:** GPT-3.5 demonstrates variable performance across different game settings, exhibiting notably lower efficacy in “(2) El Farol Bar” and “(4) Public Goods Game.” It is noteworthy that, _γ_ -Bench provides a test bed to evaluate the ability of LLMs in complex reasoning scenarios. As model’s ability improves ( _e.g._ , achieving more than 90 on _γ_ -Bench), we can increase the difficulty by varying game settings. 

## 4.4 RQ4: LEADERBOARD 

This RQ investigates the variance in decision-making capabilities among different LLMs, using _γ_ -Bench. We first focus on closed-source models, including OpenAI’s GPT-3.5 (0613, 1106, and 0125), GPT-4 (Turbo-0125, 4o-0806), and Google’s Gemini Pro (1.0, 1.5). The results are organized in Table 1a, with model performance visualized in Fig. 9 in the appendix. Gemini-1.5-Pro scores 69 _._ 8, markedly surpassing other models, particularly in games (1), (4), and (5). GPT-4o follows 

9 

Published as a conference paper at ICLR 2025 

closely behind Gemini-Pro, achieving 66 _._ 75. GPT-4’s lowered performance in the “(2) El Farol Bar” game (23 _._ 0) and the “(5) Diner’s Dilemma” game (0 _._ 9) stems from its conservative strategies favoring staying at home and spending less money. Similarly, the “(6) Sealed-Bid Auction” (24 _._ 2) is attributed to a strategy of not risking bidding high or low. The risk-averse preference also explains the relatively good score on the “(4) Public Goods Game,” where the GPT-4 does not take the risk to invest. Furthermore, an evaluation of three GPT-3.5 updates shows similar performance. 

Next, we focus on open-source models, whose performance is detailed in Table 1b and visualized in Fig. 10. The top-two open-source model, LLaMA-3.1-70B and Mixtral-8x22B, closely follows Gemini-1.5-Pro with a score of 65 _._ 9 and 62 _._ 4, surpassing GPT-4. Most open-source models, including Qwen-2, LLaMA-3.1-405B, and LLaMA-3.1-8B, outperform GPT-3.5 and Gemini-1.0-Pro. Mixtral-8x7B exhibits the lowest performance, likely due to its smaller size and weaker reasoning capabilities. Interestingly, LLaMA-3.1-405B underperforms compared to its smaller counterpart, the 70B version, which we attribute to its overly conservative strategy in the “(2) El Farol Bar” game, a challenge similar to the one faced by GPT-4. 

**Answer to RQ4:** Currently, Gemini-1.5-Pro outperforms all other models evaluated in this study. LLaMA-3.1-70B performs closely, being in the second place. 

## 5 RELATED WORK 

Evaluating LLMs through game theory models has become a popular research direction. An overview on recent studies is summarized in Table 3. We find: (1) Many studies examine the PSNE on two-player, single-round settings, focusing on the _Prisoner’s Dilemma_ and the _Ultimatum Game_ . (2) Varying temperatures are employed without discussing the impact on LLMs’ performance. 

## 5.1 SPECIFIC GAMES 

Researchers have explored diverse game scenarios. Using the complex and deceptive environments of _Avalon_ game as a test bed, recent work focuses on long-horizon multi-party dialogues (Stepputtis et al., 2023), social behaviors (Lan et al., 2023), social intelligence (Liu et al., 2024), and recursive contemplation (Wang et al., 2023) for identifying deceptive information. Other papers have investigated communication games like _Werewolf_ , with a focus on tuning-free frameworks (Xu et al., 2023b) and reinforcement learning-powered approaches (Xu et al., 2024). O’Gara (2023) found that advanced LLMs exhibit deception and lie detection capabilities in the text-based game, _Hoodwinked_ . Meanwhile, Liang et al. (2023) evaluated LLMs’ intelligence and strategic communication skills in the word guessing game, _Who Is Spy?_ In the game of _Water Allocation Challenge_ , Mao et al. (2023) constructed a scenario highlighting unequal competition for limited resources. 

## 5.2 GAME BENCHMARKS 

Another line of studies collects games to build more comprehensive benchmarks to assess the artificial general intelligence of LLMs. Tsai et al. (2023) found that while LLMs perform competitively in text games, they struggle with world modeling and goal inference. GameEval (Qiao et al., 2023) introduced three goal-driven conversational games ( _Ask-Guess_ , _SpyFall_ , and _TofuKingdom_ ) to assess the problem-solving capabilities of LLMs in cooperative and adversarial settings. MAgIC (Xu et al., 2023a) proposed the probabilistic graphical modeling method for evaluating LLMs in multi-agent game settings. LLM-Co (Agashe et al., 2023) assesses LLMs in multi-agent coordination scenarios, showcasing their capabilities in partner intention inference and proactive assistance. SmartPlay (Wu et al., 2024) evaluated LLMs as agents across six games, emphasizing reasoning, planning, and learning capabilities. Abdelnabi et al. (2024) designed negotiation games involving six parties with distinct objectives to evaluate LLMs’ ability to reach agreement. 

## 6 CONCLUSION 

This paper presents _γ_ -Bench, a benchmark designed to assess LLMs’ Gaming Ability in MultiAgent environments. _γ_ -Bench incorporates eight classic game theory scenarios, emphasizing multiplayer interactions across multiple rounds and actions. Our findings reveal that GPT-3.5 (0125) 

10 

Published as a conference paper at ICLR 2025 

demonstrates a limited decision-making ability on _γ_ -Bench, yet it can improve itself by learning from the historical results. Leveraging the carefully designed scoring scheme, we observe that GPT-3.5 (0125) exhibits commendable robustness across various temperatures and prompts. It is noteworthy that strategies such as CoT prove effective in this context. Nevertheless, its capability to generalize across various game settings remains restricted. Finally, Gemini-1.5-Pro outperforms all tested models, achieving the highest ranking on the _γ_ -Bench leaderboard, with the open-source LLaMA-3.1-70B following closely behind. 

## ACKNOWLEDGMENTS 

The paper is supported by the Research Grants Council of the Hong Kong Special Administrative Region, China (No. CUHK 14206921 of the General Research Fund). 

## REFERENCES 

- Sahar Abdelnabi, Amr Gomaa, Sarath Sivaprasad, Lea Sch¨onherr, and Mario Fritz. Cooperation, competition, and maliciousness: Llm-stakeholders interactive negotiation. In _The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , 2024. 

- Saaket Agashe, Yue Fan, and Xin Eric Wang. Evaluating multi-agent coordination abilities in large language models. _arXiv preprint arXiv:2310.03903_ , 2023. 

- Gati V Aher, Rosa I Arriaga, and Adam Tauman Kalai. Using large language models to simulate multiple humans and replicate human subject studies. In _International Conference on Machine Learning_ , pp. 337–371. PMLR, 2023. 

- Elif Akata, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, and Eric Schulz. Playing repeated games with large language models. _arXiv preprint arXiv:2305.16867_ , 2023. 

- W Brian Arthur. Inductive reasoning and bounded rationality. _The American economic review_ , 84 (2):406–411, 1994. 

- Daniel Ashlock and Garrison Greenwood. Generalized divide the dollar. In _2016 IEEE Congress on Evolutionary Computation (CEC)_ , pp. 343–350. IEEE, 2016. 

- David Baidoo-Anu and Leticia Owusu Ansah. Education in the era of generative artificial intelligence (ai): Understanding the potential benefits of chatgpt in promoting teaching and learning. _Journal of AI_ , 7(1):52–62, 2023. 

- Philip Brookins and Jason DeBacker. Playing games with gpt: What can we learn about a large language model from canonical strategic games? _Economics Bulletin_ , 44(1):25–37, 2024. 

- S´ebastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv preprint arXiv:2303.12712_ , 2023. 

- Valerio Capraro, Roberto Di Paolo, and Veronica Pizziol. Assessing large language models’ ability to predict how humans balance self-interest and the interest of others. _arXiv preprint arXiv:2307.12776_ , 2023. 

- Jiangjie Chen, Siyu Yuan, Rong Ye, Bodhisattwa Prasad Majumder, and Kyle Richardson. Put your money where your mouth is: Evaluating strategic planning and execution of llm agents in an auction arena. _arXiv preprint arXiv:2310.05746_ , 2023. 

- Jinhao Duan, Renming Zhang, James Diffenderfer, Bhavya Kailkhura, Lichao Sun, Elias StengelEskin, Mohit Bansal, Tianlong Chen, and Kaidi Xu. Gtbench: Uncovering the strategic reasoning limitations of llms via game-theoretic evaluations. _arXiv preprint arXiv:2402.12348_ , 2024. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

11 

Published as a conference paper at ICLR 2025 

Caoyun Fan, Jindou Chen, Yaohui Jin, and Hao He. Can large language models serve as rational players in game theory? a systematic analysis. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , number 16 in 38, pp. 17960–17967, 2024. 

Natalie S Glance and Bernardo A Huberman. The dynamics of social dilemmas. _Scientific American_ , 270(3):76–81, 1994. 

Robert E Goodin. _The theory of institutional design_ . Cambridge University Press, 1998. 

- Neel Guha, Julian Nyarko, Daniel E Ho, Christopher Re, Adam Chilton, Aditya Narayana, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel Rockmore, et al. Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models. In _Thirtyseventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , 2023. 

- Fulin Guo. Gpt agents in game theory experiments. _arXiv preprint arXiv:2305.05516_ , 2023. 

- Jiaxian Guo, Bo Yang, Paul Yoo, Bill Yuchen Lin, Yusuke Iwasawa, and Yutaka Matsuo. Suspicionagent: Playing imperfect information games with theory of mind aware gpt-4. _arXiv preprint arXiv:2309.17277_ , 2023. 

- Babak Heydari and Nunzio Lor`e. Strategic behavior of large language models: Game structure vs. contextual framing. _Contextual Framing (September 10, 2023)_ , 2023. 

- John J Horton. Large language models as simulated economic agents: What can we learn from homo silicus? Technical report, National Bureau of Economic Research, 2023. 

- Jen-tse Huang, Man Ho Lam, Eric John Li, Shujie Ren, Wenxuan Wang, Wenxiang Jiao, Zhaopeng Tu, and Michael R Lyu. Apathetic or empathetic? evaluating LLMs’ emotional alignments with humans. In _Advances in Neural Information Processing Systems 37_ , 2024a. 

- Jen-tse Huang, Wenxuan Wang, Eric John Li, Man Ho Lam, Shujie Ren, Youliang Yuan, Wenxiang Jiao, Zhaopeng Tu, and Michael R Lyu. On the humanity of conversational ai: Evaluating the psychological portrayal of llms. In _Proceedings of the Twelfth International Conference on Learning Representations_ , 2024b. 

Bernardo A. Huberman. _The Ecology of Computation_ . North-Holland, 1988. 

- Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ , 2024. 

- Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Xing Wang, and Zhaopeng Tu. Is chatgpt a good translator? a preliminary study. _arXiv preprint arXiv:2301.08745_ , 2023. 

- Douglas Johnson, Rachel Goodman, J Patrinely, Cosby Stone, Eli Zimmerman, Rebecca Donald, Sam Chang, Sean Berkowitz, Avni Finn, Eiman Jahangir, et al. Assessing the accuracy and reliability of ai-generated medical responses: an evaluation of the chat-gpt model. _Research square_ , 2023. 

- D Marc Kilgour. Equilibrium points of infinite sequential truels. _International Journal of Game Theory_ , 6:167–180, 1977. 

D Marc Kilgour and Steven J Brams. The truel. _Mathematics Magazine_ , 70(5):315–326, 1997. 

D Mark Kilgour. The sequential truel. _International Journal of Game Theory_ , 4:151–174, 1975. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _Advances in Neural Information Processing Systems_ , 35:22199–22213, 2022. 

- Daphne Koller and Avi Pfeffer. Representations and solutions for game-theoretic problems. _Artificial intelligence_ , 94(1-2):167–215, 1997. 

12 

Published as a conference paper at ICLR 2025 

- Aobo Kong, Shiwan Zhao, Hao Chen, Qicheng Li, Yong Qin, Ruiqi Sun, Xin Zhou, Enzhi Wang, and Xiaohang Dong. Better zero-shot reasoning with role-play prompting. In _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pp. 4099–4113, 2024. 

- Michal Kosinski. Theory of mind might have spontaneously emerged in large language models. _arXiv preprint arXiv:2302.02083_ , 2023. 

- Yihuai Lan, Zhiqiang Hu, Lei Wang, Yang Wang, Deheng Ye, Peilin Zhao, Ee-Peng Lim, Hui Xiong, and Hao Wang. Llm-based agent society investigation: Collaboration and confrontation in avalon gameplay. _arXiv preprint arXiv:2310.14985_ , 2023. 

- Pier Luca Lanzi and Daniele Loiacono. Chatgpt and other large language models as evolutionary engines for online interactive collaborative game design. In _Proceedings of the Genetic and Evolutionary Computation Conference_ , pp. 1383–1390, 2023. 

- Alain Ledoux. Concours r´esultats complets. les victimes se sont plu `a jouer le 14 d’atout. _Jeux & Strat´egie_ , 2(10):10–11, 1981. 

- Jiatong Li, Rui Li, and Qi Liu. Beyond static datasets: A deep interaction approach to llm evaluation. _arXiv preprint arXiv:2309.04369_ , 2023. 

- Tian Liang, Zhiwei He, Jen-tes Huang, Wenxuan Wang, Wenxiang Jiao, Rui Wang, Yujiu Yang, Zhaopeng Tu, Shuming Shi, and Xing Wang. Leveraging word guessing games to assess the intelligence of large language models. _arXiv preprint arXiv:2310.20499_ , 2023. 

- Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, and Shuming Shi. Encouraging divergent thinking in large language models through multiagent debate. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , 2024. 

- Ziyi Liu, Abhishek Anand, Pei Zhou, Jen-tse Huang, and Jieyu Zhao. Interintent: Investigating social intelligence of llms via intention understanding in an interactive game context. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , 2024. 

- Shaoguang Mao, Yuzhe Cai, Yan Xia, Wenshan Wu, Xun Wang, Fengyi Wang, Tao Ge, and Furu Wei. Alympics: Language agents meet game theory. _arXiv preprint arXiv:2311.03220_ , 2023. 

- R Preston McAfee and John McMillan. Auctions and bidding. _Journal of economic literature_ , 25 (2):699–738, 1987. 

Roger B Myerson. _Game theory_ . Harvard university press, 2013. 

Rosemarie Nagel. Unraveling in guessing games: An experimental study. _The American economic review_ , 85(5):1313–1326, 1995. 

John F Nash. Equilibrium points in n-person games. _Proceedings of the national academy of sciences_ , 36(1):48–49, 1950. 

John F Nash. Non-cooperative games. _Annals of Mathematics_ , 54(2):286–295, 1951. 

- Aidan O’Gara. Hoodwinked: Deception and cooperation in a text-based game for language models. _arXiv preprint arXiv:2308.01404_ , 2023. 

OpenAI. Introducing chatgpt. _OpenAI Blog Nov 30 2022_ , 2022. URL https://openai.com/ index/chatgpt/. 

OpenAI. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

Joseph Persky. Retrospectives: The ethology of homo economicus. _Journal of Economic Perspectives_ , 9(2):221–231, 1995. 

- Steve Phelps and Yvan I Russell. Investigating emergent goal-like behaviour in large language models using experimental economics. _arXiv preprint arXiv:2305.07970_ , 2023. 

13 

Published as a conference paper at ICLR 2025 

Sundar Pichai and Demis Hassabis. Introducing gemini: our largest and most capable ai model. _Google Blog Dec 06 2023_ , 2023. URL https://blog.google/technology/ai/ google-gemini-ai/. 

Sundar Pichai and Demis Hassabis. Our next-generation model: Gemini 1.5. _Google Blog Feb 15 2024_ , 2024. URL https://blog.google/technology/ai/ google-gemini-next-generation-model-february-2024/. 

- Dan Qiao, Chenfei Wu, Yaobo Liang, Juntao Li, and Nan Duan. Gameeval: Evaluating llms on conversational games. _arXiv preprint arXiv:2308.10032_ , 2023. 

- Chengwei Qin, Aston Zhang, Zhuosheng Zhang, Jiaao Chen, Michihiro Yasunaga, and Diyi Yang. Is chatgpt a general-purpose natural language processing task solver? In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 1339–1384, 2023. 

- Mathias Risse. What is rational about nash equilibria? _Synthese_ , 124:361–384, 2000. 

- Ariel Rubinstein. Instinctive and cognitive reasoning: A study of response times. _The Economic Journal_ , 117(523):1243–1259, 2007. 

- Paul A Samuelson. The pure theory of public expenditure. _The review of economics and statistics_ , 36(4):387–389, 1954. 

- Lloyd S Shapley and Martin Shubik. Pure competition, coalitional power, and fair division. _International Economic Review_ , 10(3):337–362, 1969. 

- Simon Stepputtis, Joseph P Campbell, Yaqi Xie, Zhengyang Qi, Wenxin Zhang, Ruiyi Wang, Sanketh Rangreji, Charles Lewis, and Katia Sycara. Long-horizon dialogue understanding for role identification in the game of avalon with large language models. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pp. 11193–11208, 2023. 

- Ian Stewart. A puzzle for pirates. _Scientific American_ , 280(5):98–99, 1999. 

- Nigar M Shafiq Surameery and Mohammed Y Shakor. Use chat gpt to solve programming bugs. _International Journal of Information Technology & Computer Engineering (IJITC) ISSN: 24555290_ , 3(01):17–22, 2023. 

- Chen Feng Tsai, Xiaochen Zhou, Sierra S Liu, Jing Li, Mo Yu, and Hongyuan Mei. Can large language models play text games well? current state-of-the-art and open questions. _arXiv preprint arXiv:2304.02868_ , 2023. 

- William Vickrey. Counterspeculation, auctions, and competitive sealed tenders. _The Journal of finance_ , 16(1):8–37, 1961. 

- Shenzhi Wang, Chang Liu, Zilong Zheng, Siyuan Qi, Shuo Chen, Qisen Yang, Andrew Zhao, Chaofei Wang, Shiji Song, and Gao Huang. Avalon’s game of thoughts: Battle against deception through recursive contemplation. _arXiv preprint arXiv:2310.01320_ , 2023. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837, 2022. 

- Haoran Wu, Wenxuan Wang, Yuxuan Wan, Wenxiang Jiao, and Michael Lyu. Chatgpt or grammarly? evaluating chatgpt on grammatical error correction benchmark. _arXiv preprint arXiv:2303.13648_ , 2023. 

- Yue Wu, Xuan Tang, Tom M Mitchell, and Yuanzhi Li. Smartplay: A benchmark for llms as intelligent agents. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Chengxing Xie, Canyu Chen, Feiran Jia, Ziyu Ye, Shiyang Lai, Kai Shu, Jindong Gu, Adel Bibi, Ziniu Hu, David Jurgens, James Evans, Philip Torr, Bernard Ghanem, and Guohao Li. Can large language model agents simulate human trust behaviors? _Advances in neural information processing systems_ , 37, 2024. 

14 

Published as a conference paper at ICLR 2025 

- Lin Xu, Zhiyuan Hu, Daquan Zhou, Hongyu Ren, Zhen Dong, Kurt Keutzer, and Jiashi Feng. Magic: Investigation of large language model powered multi-agent in cognition, adaptability, rationality and collaboration. _arXiv preprint arXiv:2311.08562_ , 2023a. 

Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, and Yang Liu. Exploring large language models for communication games: An empirical study on werewolf. _arXiv preprint arXiv:2309.04658_ , 2023b. 

- Zelai Xu, Chao Yu, Fei Fang, Yu Wang, and Yi Wu. Language agents with reinforcement learning for strategic play in the werewolf game. In _Proceedings of the Forty-first International Conference on Machine Learning_ , 2024. 

- An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_ , 2024. 

- Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Pinjia He, Shuming Shi, and Zhaopeng Tu. Gpt-4 is too smart to be safe: Stealthy chat with llms via cipher. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Yadong Zhang, Shaoguang Mao, Tao Ge, Xun Wang, Yan Xia, Man Lan, and Furu Wei. K-level reasoning with large language models. _arXiv preprint arXiv:2402.01521_ , 2024. 

- Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Zhicheng Dou, and Ji-Rong Wen. Large language models for information retrieval: A survey. _arXiv preprint arXiv:2308.07107_ , 2023. 

15 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0016-01.png)


**----- Start of picture text -----**<br>
GAMA-Bench  Framework<br>Cooperative Games (1) Guess 2/3 of the Average (2) El Farol Bar (3) Divide the Dollar<br>Betraying Games (4) Public Goods Game (5) Diner’s Dilemma (6) Sealed-Bid Auction<br>Sequential Games (7) Battle Royale (8) Pirate Game<br>**----- End of picture text -----**<br>


Figure 3: _γ_ -Bench enables multiple LLMs and humans to engage in multi-round games. The framework comprises three categories of games, each targeting different LLM abilities, and includes eight classic games from _Game Theory_ . 

## CONTENTS 

|**A **|**More Information on Game Theory**|**More Information on Game Theory**|**18**|
|---|---|---|---|
||A.1<br>Formulation . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|18|
||A.2<br>Nash Equilibrium . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|18|
||A.3<br>Human Behaviors . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|18|
|**B**|**Literature Review: Evaluating LLMs with Game Theory**||**19**|
|**C **|**Details about Prompts**||**20**|
||C.1<br>Design Methodology . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|20|
||C.2<br>Cooperative Games . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|20|
||C.3<br>Betraying Games . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|22|
||C.4<br>Sequential Games . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|24|
|**D **|**Examples of GPT-4-Rephrased Prompts**||**26**|
|**E**|**Rescale Method for Raw Scores**||**29**|
|**F**|**Detailed Results**||**30**|
||F.1<br>Robustness: Multiple Runs|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|30|
||F.2<br>Robustness: Temperatures|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|31|
||F.3<br>Robustness: Prompt Templates . . . . . . . . . . . . . . . . . . . . . . . . . . . .||32|



16 

Published as a conference paper at ICLR 2025 

||F.4|Generalizability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|34|
|---|---|---|---|
||F.5|Leaderboard . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|35|
||F.6|Detailed Player Actions of GPT-3.5 (0125) . . . . . . . . . . . . . . . . . . . . . .|36|
|**G **|**LLM**|**vs. Specifc Strategies**|**36**|
|**H **|**Jailbreak Infuence**||**37**|
|**I**|**Limitations**||**37**|
|**J**|**Ethics Statement and Broader Impacts**||**38**|
|**K **|**LLaMA-3.1-70B**||**38**|
||K.1|Robustness: Multiple Runs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|38|
||K.2|Robustness: Temperatures<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|38|
||K.3|Robustness: Prompt Templates . . . . . . . . . . . . . . . . . . . . . . . . . . . .|39|
||K.4|Generalizability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|39|
|**L**|**Gemini-1.5-Pro**||**40**|
||L.1|Robustness: Multiple Runs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|40|
||L.2|Robustness: Temperatures<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|40|
||L.3|Robustness: Prompt Templates . . . . . . . . . . . . . . . . . . . . . . . . . . . .|41|
||L.4|Generalizability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|41|
|**M **|**GPT-4o**||**42**|
||M.1|Robustness: Multiple Runs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|42|
||M.2|Robustness: Temperatures<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|42|
||M.3|Robustness: Prompt Templates . . . . . . . . . . . . . . . . . . . . . . . . . . . .|43|
||M.4|Generalizability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|43|



17 

Published as a conference paper at ICLR 2025 

## A MORE INFORMATION ON GAME THEORY 

## A.1 FORMULATION 

Game theory involves analyzing mathematical models of strategic interactions among rational agents (Myerson, 2013). A game can be modeled using these key elements: 

1. Players, denoted as _P_ = _{_ 1 _,_ 2 _, · · · , N }_ : A set of _N_ participants. 

2. Actions, represented as _A_ = _{Ai}_ : _N_ sets of actions available to each player. For instance, _A_ = _{A_ 1 = _{C, D}, A_ 2 = _{D, F }, · · · , AN_ = _{C, F }}_ 

3. Utility functions, denoted as _U_ = _{Ui_ : _×[N] j_ =1 _[A][j][�→]_[R] _[}]_[:][A set of] _[ N]_[functions that quantify each] player’s preferences over all possible outcomes. 

4. Information, represented as _I_ = _{Ii}_ : _N_ sets of information available to each player, including other players’ action sets, utility functions, historical actions, and other beliefs. 

5. Order, indicated by _O_ = _O_ 1 _, O_ 2 _, · · · , Ok_ : A sequence of _k_ sets specifying the _k_ steps to take actions. For example, _O_ = _P_ implies that all players take actions simultaneously. 

In this study, _Multi-Player_ games are defined as those with _|P| >_ 2 since game theory models have at least two players. Similarly, _Multi-Action_ games are those where _∀i∈P |Ai| >_ 2. Meanwhile, _Multi-Round_ games involve the same set of players repeatedly engaging in the game, with a record of all previous actions being maintained. _Simultaneous_ games satisfy that _k_ = 1, whereas _Sequential_ games have _k >_ 1, indicating players make decisions in a specific order. Games of _Perfect Information_ are characterized by the condition _∀i,j∈P|i_ = _jIi_ = _Ij_ . Since every player can see their own action, the above condition indicates that all players are visible to the complete information set in the game. Conversely, games not meeting this criterion are classified as _Imperfect Information_ games, where players have limited knowledge of others’ actions. 

## A.2 NASH EQUILIBRIUM 

Studying game theory models often involves analyzing their Nash Equilibria (NE) (Nash, 1950). An NE is a specific set of strategies where no one has anything to gain by changing only one’s own strategy. This implies that given one player’s choice, the strategies of others are constrained to a specific set, which in turn limits the original player’s choice to the initial one. When each player’s strategy contains only one action, the equilibrium is identified as a _Pure Strategy Nash Equilibrium_ (PSNE) (Nash, 1950). However, in certain games, such as rock-paper-scissors, an NE exists only when players employ a probabilistic approach to their actions. This type of equilibrium is known as a _Mixed Strategy Nash Equilibrium_ (MSNE) (Nash, 1951), with PSNE being a subset of MSNE where probabilities are concentrated on a single action. According to Thm. A.1 shown below, we can analyze the NE of each game and evaluate whether LLMs’ choices align with the NE. 

**Theorem A.1 (Nash’s Existence Theorem)** _Every game with a finite number of players in which each player can choose from a finite number of actions has at least one mixed strategy Nash equilibrium, in which each player’s action is determined by a probability distribution._ 

## A.3 HUMAN BEHAVIORS 

The attainment of NE presupposes participants as _Homo Economicus_ , who are consistently rational and narrowly self-interested, aiming at maximizing self goals (Persky, 1995). However, human decision-making often deviates from this ideal. Empirical studies reveal that human choices frequently diverge from what the NE predicts (Nagel, 1995). This deviation is attributed to the complex nature of human decision-making, which involves not only rational analysis but also personal values, preferences, beliefs, and emotions. By comparing human decision patterns documented in prior studies, together with the NE, we can ascertain whether LLMs exhibit tendencies more akin to homo economicus or actual human decision-makers, thus shedding light on their alignment with human-like or purely rational decision-making processes. 

18 

Published as a conference paper at ICLR 2025 

## B LITERATURE REVIEW: EVALUATING LLMS WITH GAME THEORY 

Evaluating LLMs through game theory models has become a popular research direction. An overview on recent studies is summarized in Table 3. From our analysis, several key observations emerge: (1) The majority of these studies are concentrated on two-player settings. (2) There is a predominant focus on two-action games; notably, half of the studies examine the _Prisoner’s Dilemma_ and the _Ultimatum Game_ (the _Dictator Game_ is one of the variants of the _Ultimatum Game_ ). (3) A notable gap in the literature is the lack of the comparative studies between LLMs’ decision-making across multiple rounds and the action probability distributions predicted by the MSNE. (4) The studies exhibit variability in the temperatures used, which precludes definitive conclusions regarding their impact on LLMs’ performance. 

Table 3: A Comparison of existing studies that evaluate LLMs using game theory models. **T** denotes the temperature employed in each experiment. **MP** refers to a multi-player setting, whereas **MR** indicates multi-round interactions. **Role** specifies whether a specific role is assigned to the LLMs. 

|**Paper**|**Models**|**T**|**MP**|**MR**|**Role**|**CoT**|**Games**|
|---|---|---|---|---|---|---|---|
|Horton(2023)|text-davinci-003|-|✗|✗|✗|✗|Dictator Game|
|Guo (2023)|gpt-4-1106-preview|1|✗|✓|✓|✓|Ultimatum Game,<br>Prisoner’s Dilemma|
|Phelps & Russell(2023)|gpt-3.5-turbo|0.2|✗|✓|✓|✗|Prisoner’s Dilemma|
|Akata et al. (2023)|text-davinci-003,<br>gpt-3.5-turbo, gpt-4|0|✗|✓|✗|✗|Prisoner’s Dilemma,<br>Battle of the Sexes|
||text-ada-001, text-babbage-001,|||||||
|Aher et al. (2023)|text-curie-001, text-davinci-001,<br>text-davinci-002, text-davinci-003,|1|✗|✗|✓|✗|Ultimatum Game|
||gpt-3.5-turbo, gpt-4|||||||
|Capraro et al. (2023)|ChatGPT-4, Bard, Bing Chat|-|✗|✗|✗|✓|Dictator Game<br>(Three Variants)|
|Brookins & DeBacker (2024)|gpt-3.5-turbo|1|✗|✗|✗|✗|Dictator Game,<br>Prisoner’s Dilemma|
||gpt-3.5-turbo-0613,|||||||
|Li et al. (2023)|gpt-4-0613, claude-2.0,|-|✓|✓|✗|✗|Public Goods Game|
||chat-bison-001|||||||
|Heydari & Lor`e (2023)|gpt-3.5-turbo-16k, gpt-4,<br>llama-2|0.8|✗|✗|✓|✓|Prisoner’s Dilemma,<br>Stag Hunt, Snowdrift,<br>Prisoner’s Delight|
|Guo et al.(2023)|gpt-3.5, gpt-4|-|✗|✓|✗|✓|Leduc Hold’em|
||gpt-3.5-turbo-0613,|||||||
|Chen et al. (2023)|gpt-4-0613, claude-instant-1.2,|0.7|✓|✓|✓|✓|English Auction|
||claude-2.0,chat-bison-001|||||||
||gpt-3.5-turbo, gpt-4,||||||Cost Sharing,|
|Xu et al. (2023a)|llama-2-70b, claude-2.0,|-|✓|✓|✗|✓|Prisoner’s Dilemma,|
||palm-2||||||Public Goods Game|
|Fan et al. (2024)|text-davinci-003,<br>gpt-3.5-turbo, gpt-4|0.7|✗|✓|✗|✗|Dictator Game,<br>Rock-Paper-Scissors,<br>Ring-Network Game|
|Zhang et al. (2024)|gpt-4|0.7|✓|✓|✓|✓|Guess 0.8 of the Average<br>Survival Auction Game|
||gpt-3.5-turbo, gpt-4,|||||||
|Duan et al. (2024)|llama-2-70b, codellama-34b,|0.2|✓|✓|✗|✓|Ten Gamesa|
||mistral-7b-orca|||||||
||text-davinci-003,|||||||
||gpt-3.5-turbo-instruct,|||||||
|Xie et al. (2024)|gpt-3.5-turbo-0613, gpt-4,|-|✗|✓|✓|✓|Seven Gamesb|
||llama-2-(7/13/70)b,|||||||
||vicuna-(7/13/33)b-v1.3|||||||
|**This Study**|gpt-3.5-turbo, gpt-4<br>gemini-pro|0_∼_1|✓|✓|✓|✓|Eight Gamesc|



> a Tic-Tac-Toe, Connect-4, Kuhn Poker, Breakthrough, Liar’s Dice, Blind Auction, Negotiation, Nim, Pig, Iterated Prisoner’s Dilemma. 

> b Trust Game, Minimum Acceptable Probabilities Trust Game, Repeated Trust Game, Dictator Game, Risky Dictator Game, Lottery People Game, Lottery Gamble Game. 

> c Guess 2/3 of the Average, El Farol Bar, Divide the Dollar, Public Goods Game, Diner’s Dilemma, Sealed-Bid Auction, Battle Royale, Pirate Game. 

## [BACK TO INTRODUCTION] 

## [BACK TO RELATED WORK] 

19 

Published as a conference paper at ICLR 2025 

## C DETAILS ABOUT PROMPTS 

## C.1 DESIGN METHODOLOGY 

We adopt a cohesive approach to ensure the prompt design is systematic and not arbitrary. Game descriptions are gathered from verified sources, including academic papers referenced in §2 and Wikipedia entries. Using these descriptions, we instruct GPT-4 to generate prompts to guide LLMs in engaging in the specified games. These prompts are structured to encompass four essential elements: the rules of the game, objectives for the players, a template for announcing game outcomes (for displaying historical results), and instructions for formatting responses in JSON. A manual checking process is conducted to ascertain that GPT-4’s comprehension of the game descriptions is correct. The prompts are detailed in the rest part of this section. 

## C.2 COOPERATIVE GAMES 

## **Guess 2/3 of the Average** 

|SYSTEM|You are participating in a game played by_N_|players over_K_ rounds.|
|---|---|---|
||Game Rules:||
||1. Each player selects an integer number between_MIN_ and_MAX_, inclusive.||
||2. After all selections are made, the average of all chosen numbers is calculated.||
||3. The target number is_R_of this average.||
||4. The winner is the player(s) who selected a|number closest to the target number.|
||_· · ·_||
|USER|Game Results for Round_I_:||
||Average Number Chosen: _MI_||
||Target Number (_R_of Average): _TI_||
||Winning Number: _WI_||
||You chose:||
|ASSISTANT|_{_“chosen<br>number”: “_CIJ_”_}_||
|USER|[Congratulation you won]/[Unfortunately you lost].||
||_· · ·_||
|USER|Now round_I_ starts.||
||Your goal is to choose a number that you believe will be closest to_R_of the average||
||of all numbers chosen by players, including your selection.||
||Please provide your chosen number in the following JSON format:||
||_{_“chosen<br>number”: “integer<br>~~b~~etween<br>~~_M_~~_IN_|~~a~~nd<br>~~_M_~~_AX_“_}_.|



[BACK TO GAME DESCRIPTION] 

[BACK TO VANILLA EXPERIMENT] 

20 

Published as a conference paper at ICLR 2025 

|**El Farol Bar**||
|---|---|
|SYSTEM|You are participating in a game played by_N_ players over_K_ rounds.|
||Game Rules:|
||1. Every round, you and the other players decide independently whether to go to|
||a bar.|
||2. If equal to or less than_R_of the players go to the bar, everyone who goes has|
||more fun than staying home, receiving a utility of_MAX_.|
||3. If more than _R_ of the players go to the bar, everyone who goes has less fun|
||than staying home, receiving a utility of_MIN_.|
||4. Everyone who stays home receives a utility of_HOME_.|
||_· · ·_|
|USER|Game Results for Round_I_:|
||(Only for Explicit) _GI_ players went to the bar, while _SI_ players stayed home.|
||_GI/N_, which is [more]/[equal to or less] than_R_of the players went to the bar.|
||It was [less]/[more] fun to go to the bar this round.|
||You chose:|
|ASSISTANT|_{_“decision”: “_DIJ_”_}_|
|USER|You gained_GIJ_.|
||_· · ·_|
|USER|Now round_I_ starts.|
||Your goal is to maximize your fun. Choose to go to the bar when you predict|
||fewer than_R_of the players will go, and choose to stay home otherwise.|
||Please provide your decision in the following JSON format:|
||_{_“decision”: “go<br>~~o~~r<br>~~s~~tay”_}_.|
|||
|**Divide the Dollar**||
|SYSTEM|You are participating in a game played by_N_ players over_K_ rounds.|
||Game Rules:|
||1. You are dividing_G_golds. Each player independently proposes a bid.|
||2. If the sum of all bids does not exceed_G_, each player receives their bid amount.|
||3. If the sum exceeds_G_, all players receive nothing.|
||_· · ·_|
|USER|Game Results for Round_I_:|
||Your bid amount was:|
|ASSISTANT|_{_“bid<br>~~a~~mount”: “_BIJ_”_}_|
|USER|The sum of all bids was_SI_.|
||The sum [does not exceed]/[exceeds]_G_.|
||You received [_BIJ_]/[0] golds.|
||_· · ·_|
|USER|Now round_I_ starts.|
||Your goal is to maximize your individual gain without causing the total sum of|
||bids to exceed_G_golds.|
||Please provide your bid amount in the following JSON format:|
||_{_“bid<br>~~a~~mount”: “integer<br>~~b~~etween<br>~~0~~<br>~~a~~nd<br>~~_G_~~”_}_.|



[BACK TO GAME DESCRIPTION] 

[BACK TO VANILLA EXPERIMENT] 

21 

Published as a conference paper at ICLR 2025 

## C.3 BETRAYING GAMES 

|**Public Goods**|**Game**||
|---|---|---|
|SYSTEM|You are participating in a game played by_N_ players over_K_ rounds.||
||Game Rules:||
||1. In each round, you, as a player, must decide how many of your private tokens||
||you wish to contribute secretly to the public pot.||
||2. The total tokens in this pot will be multiplied by the factor _R_ to create the||
||“public good” payoff.||
||3. This payoff will then be evenly divided among all players, regardless of their||
||individual contribution.||
||4. Any tokens you do not contribute will be retained in your private collection.||
||_· · ·_||
|USER|Game Results for Round_I_:||
||Contributed tokens of each player: _CI_1_, CI_2_, · · · _|_, CIN_|
||You contributed:||
|ASSISTANT|_{_“tokens<br>~~c~~ontributed”: “_CIJ_”_}_||
|USER|Tokens in the public pot: _SI_||
||Your gain: _gIJ_||
||Your tokens after round_I_: _TIJ_||
||Tokens of each player after round_I_: _TI_1_, TI_2_, · ·_|_· , TIN_|
||_· · ·_||
|USER|Now round_I_ starts.||
||Your goal is to maximize your total token count by the end of the game. Currently||
||you have_TI−_1_J_ tokens. You need to decide the number of tokens to be contributed||
||to the public pot.||
||Please provide the number of tokens in the following JSON format:||
||_{_“tokens<br>~~c~~ontributed”: “integer<br>~~b~~etween<br>~~0~~<br>~~a~~nd|~~_T_~~_IJ_”_}_|
||||
|**Diner’s Dilemma**|||
|SYSTEM|You are participating in a game played by_N_ players over_K_ rounds.||
||Game Rules:||
||1. Each player must choose to order either a costly dish or a cheap dish.||
||2. The price of the costly dish is_Ph_. The price of the cheap dish is_Pl_.||
||3. The costly dish brings you a utility of_Uh_. The cheap dish brings you a utility||
||of_Ul_.||
||4. The costly dish is tastier than the cheap dish, but not suffciently to justify its||
||price when dining alone.||
||5. At the end of each round, the total cost of all dishes ordered is split equally||
||among all players.||
||_· · ·_||
|USER|Game Results for Round_I_:||
||_Nh_people chose the costly dish, while_Nl_ chose the cheap dish.||
||The total cost is_SI_. You need to pay_CI_.||
||You chose:||
|ASSISTANT|_{_“chosen<br>dish”: “_DIJ_”_}_||
|USER|Your utility is_uIJ_.||
||_· · ·_||
|USER|Now round_I_ starts.||
||Your goal is to maximize your overall satisfaction, balancing the quality of the||
||dish and the cost shared.||
||Please provide your chosen dish in the following JSON format:||
||_{_“chosen<br>dish”: “costly<br>~~o~~r<br>~~c~~heap”_}_||



[BACK TO GAME DESCRIPTION] 

[BACK TO VANILLA EXPERIMENT] 

22 

Published as a conference paper at ICLR 2025 

|**Sealed-Bid**|**Auction**||
|---|---|---|
|SYSTEM|You are participating in a game played by_N_ players over_K_ rounds.||
||Game Rules:||
||1. Each player has a private|valuation for the item in each round.|
||2. Without knowing the bids and valuations of other players, each player submits||
||a written bid for the item.||
||3. The highest bidder wins|the item and pays the price of the [highest]/[second|
||highest] bid.||
||4. If you win, your utility for that round is your valuation minus the price paid. If||
||you lose, your utility is zero.||
||_· · ·_||
|USER|Game Results for Round_I_:||
||Your valuation for this round’s item was_vIJ_.||
||Your bid was:||
|ASSISTANT|_{_“bid”: “_bIJ_”_}_||
|USER|The winning bid was: _WI_.||
||The price paid was: _PI_.||
||You [won]/[lost]. Your utility is [_uIJ_]/[0].||
||_· · ·_||
|USER|Now round_I_ starts.||
||Your goal is to maximize your total utility. Your valuation for this round’s item is||
||_vIJ_.||
||Please provide your bid in the following JSON format:||
||_{_“bid”: “integer<br>~~b~~etween<br>~~0~~|~~a~~nd<br>~~_v_~~_IJ_”_}_|



[BACK TO GAME DESCRIPTION] 

[BACK TO VANILLA EXPERIMENT] 

23 

Published as a conference paper at ICLR 2025 

C.4 SEQUENTIAL GAMES 

|**Battle Royale**||
|---|---|
|SYSTEM|You are participating in a game played by_N_.|
||Game Rules:|
||1. You are in a survival game where only one can survive and win.|
||2. Players take turns shooting at others in a predetermined order based on their|
||hit rates, from the lowest to the highest.|
||3. Players’ names and hit rates ranked by shooting order are _{_“_ID_1”: “_HIT_1”,|
||“_ID_2”: “_HIT_2”,_· · ·_, “_IDN_”: “_HITN_”_}_. You are_IDJ_. Your hit rate is_HITJ_.|
||You are the_RANKJ_-th to shoot.|
||4. You have an unlimited number of bullets.|
||5. You may choose to intentionally miss your shot on your turn.|
||_· · ·_|
|USER|Game Results for Round_I_:|
||Your action:|
|ASSISTANT|(Only for the player itself)_{_“target”: “_tIJ_”_}_|
|USER|_NAMEJ_ [intentionally missed the shot]/[shot at _tIJ_ and hit]/[shot at _tIJ_ but|
||missed].|
||There are_NI_ players left.|
||_· · ·_|
|USER|Now round_I_ starts.|
||Your goal is to eliminate other players to survive until the end and win the game.|
||The remaining players’ names and hit rates ranked by shooting order are:_{_“_ID_1”:|
||“_HIT_1”, “_ID_2”: “_HIT_2”,_· · ·_, “_IDN_”: “_HITN_”_}_. You are_IDJ_. Your hit rate|
||is_HITJ_. You are the_RANKJ_-th to shoot. Please decide whether to shoot at a|
||player or intentionally miss.|
||Please provide your action in the following JSON format:|
||_{_“target”: “playerID<br>~~o~~r<br>~~n~~ull”_}_|



[BACK TO GAME DESCRIPTION] 

[BACK TO VANILLA EXPERIMENT] 

24 

Published as a conference paper at ICLR 2025 

|**Pirate Game**||
|---|---|
|SYSTEM|You are participating in a game played by_N_.|
||Game Rules:|
||1. You are pirates who have found _G_ gold coins. You are deciding how to|
||distribute these coins among yourselves.|
||2.<br>The pirates will make decisions in strict order of seniority.<br>You are the|
||_RANKJ_-th most senior pirate.|
||3. The most senior pirate proposes a plan to distribute the_G_gold coins.|
||4. All pirates, including the proposer, vote on the proposed distribution.|
||5. If the majority accepts the plan, each pirate receives the gold coins as the most|
||senior pirate proposed.|
||6. If the majority rejects the plan, the proposer is thrown overboard, and the next|
||senior pirate proposes a new plan.|
||7. The game ends when a plan is accepted or only one pirate remains.|
||_· · ·_|
|USER|The _I_-th most senior pirate proposed a plan of _{_“_I_”: “_gII_”, “_I_ + 1”: “_gII_+1”,|
||_· · ·_, “_I_”: “_gIN_”_}_.|
||_AI_ of_N_ pirates chose to accept the distribution.|
||You chose:|
|ASSISTANT|_{_“decision”: “_DIJ_”_}_|
|USER|Less than half of the pirates accepted the plan.|
||The_I_-th most senior pirate was thrown overboard and eliminated from the game.|
||The game continues.|
||_· · ·_|
|USER|Now the_I_-th most senior pirate needs to propose a plan.|
||Your primary goal is to survive. If you survive, your next goal is to maximize the|
||number of gold coins you receive. You may also prefer to throw another pirate|
||overboard if it does not negativelyimpactyour othergoals.|
|For voters|The proposed plan is_{_“_I_”: “_gII_”, “_I_ + 1”: “_gII_+1”,_· · ·_, “_I_”: “_gIN_”_}_. You will|
||get_gIJ_ golds from this plan.|
||Please provide your decision on the current proposal in the following JSON for-|
||mat:|
||_{_“decision”: “accept<br>~~o~~r<br>~~r~~eject”_}_|
|For proposer|You need to propose a plan to divide_G_golds. The proposed numbers must be all|
||non-negative integers and sum up to_G_.|
||Please provide your proposal of the golds distributed to each pirate from the you|
||to the_I_-th most senior in the following JSON format:|
||_{_”proposal”: _{_“_I_”: “_gII_”, “_I_ + 1”: “_gII_+1”,_· · ·_, “_I_”: “_gIN_”_}}_|



[BACK TO GAME DESCRIPTION] 

[BACK TO VANILLA EXPERIMENT] 

25 

Published as a conference paper at ICLR 2025 

## D EXAMPLES OF GPT-4-REPHRASED PROMPTS 

§4.1 involves testing the GPT-3.5 (0125)’s robustness against different prompt templates. This section presents the prompts used in this analysis, namely Prompts V2 to V4, with V1 as the default, as detailed in §C). We include only the prompts for the game “Guess 2/3 of the Average,” while the five prompt templates of seven other games can be found in our GitHub (https: //github.com/CUHK-ARISE/GAMABench). 

## **Guess 2/3 of the Average (V2)** 

|SYSTEM|You’re participating in a game involving_N_ participants and it spans across_K_|
|---|---|
||rounds.|
||The rules of the game are as follows:|
||1.<br>Every participant must choose an integer within the range of _MIN_ to|
||_MAX_, both included.|
||2. Once everyone has chosen their numbers, the mean of all these numbers is|
||computed.|
||3. The goal number becomes_R_times this average.|
||4. The person or people who picked a number closest to the goal number are|
||declared the winners.|
||_· · ·_|
|USER|The outcomes of the game for Round_I_ are as follows:|
||The average number selected was_MI_|
||The target number, which is_R_of the average, is_TI_|
||The number that won was_WI_.|
||Your selection was:|
|ASSISTANT|_{_“chosen<br>~~n~~umber”: “_CIJ_”_}_|
|USER|[Congratulation you won]/[Unfortunately you lost].|
||_· · ·_|
|USER|Commencing with round_I_.|
||Your target should be to select a number that in your opinion will align most|
||closely with _R_ of the total average of all the player’s numbers selected, your|
||choice included.|
||Please provide your chosen number in the following JSON format:|
||_{_“chosen<br>~~n~~umber”: “integer<br>~~b~~etween<br>~~_M_~~_IN_<br>~~a~~nd<br>~~_M_~~_AX_”_}_.|



## [BACK TO RQ1] 

26 

Published as a conference paper at ICLR 2025 

|**Guess 2/3 of**|**the Average (V3)**|
|---|---|
|SYSTEM|You’re engaged in a game, involving_N_ participants across_K_ rounds.|
||Rules of the Game:|
||1. An integer number is chosen by every player, within the range of_MIN_ and|
||_MAX_, both numbers included.|
||2. Once each player has chosen, the average is determined from all the selected|
||numbers.|
||3. The average is multiplied by_R_to fnd the target number.|
||4. The individual or individuals whose chosen number is nearest to the target|
||number are declared the winners.|
||_· · ·_|
|USER|Results of Round_I_ Game:|
||Chosen number’s average: _MI_|
||The target percentage (_R_of average) is: _TI_|
||The winning number is: _WI_.|
||You chose:|
|ASSISTANT|_{_“chosen<br>~~n~~umber”: “_CIJ_”_}_|
|USER|[Congratulation you won]/[Unfortunately you lost].|
||_· · ·_|
|USER|The commencement of round_I_ is now.|
||The objective is to select a number that you think will be nearest to_R_times the|
||average of all the digits chosen by the participants, your choice included.|
||Please provide your chosen number in the following JSON format:|
||_{_“chosen<br>~~n~~umber”: “integer<br>~~b~~etween<br>~~_M_~~_IN_<br>~~a~~nd<br>~~_M_~~_AX_”_}_.|
|||
|**Guess 2/3 of**|**the Average (V4)**|
|SYSTEM|You’re involved in a game which brings_N_ participants together for_K_ rounds.|
||The guidelines of the game are as follows:|
||1. All players have to pick a whole number anywhere from_MIN_ to_MAX_,|
||both numbers included.|
||2. The chosen numbers are then gathered and their mean is computed.|
||3. The number to aim for, or the target number, is_R_of the calculated average.|
||4.The victorious player(s) are those whose chosen number is closest to the|
||target number.|
||_· · ·_|
|USER|The outcomes for Round_I_ are as follows:|
||The average number selected was _MI_. The target number, which is _R_ times|
||the average, was_TI_. The triumphant number was_WI_.|
||Your choice was:|
|ASSISTANT|_{_“chosen<br>~~n~~umber”: “_CIJ_”_}_|
|USER|[Congratulation you won]/[Unfortunately you lost].|
||_· · ·_|
|USER|The commencement of round_I_ is now.|
||You are tasked with selecting a number that, in your estimation, will be as close|
||as possible to_R_times the average of numbers chosen by all players, your own|
||choice included.|
||Please provide your chosen number in the following JSON format:|
||_{_“chosen<br>~~n~~umber”: “integer<br>~~b~~etween<br>~~_M_~~_IN_<br>~~a~~nd<br>~~_M_~~_AX_”_}_.|



[BACK TO RQ1] 

27 

Published as a conference paper at ICLR 2025 

**Guess 2/3 of the Average (V5)** SYSTEM You will be engaging in a game that is played over _K_ rounds and includes a total of _N_ players. The Instructions of the Game: 1. Every player is supposed to pick an integer that is within the range of _MIN_ and _MAX_ , both numbers inclusive. 2. The median of all the numbers chosen by the players is then determined after all choices have been made. 3. The number that players are aiming for is _R_ times the calculated average. 4. The player or players who opt for the number closest to this target are declared the winners. _· · ·_ USER Results of the Game for Round _I_ : The chosen average number is: _MI_ The target number ( _R_ of Average) is: _TI_ The number that won: _WI_ . Your selection was: ASSISTANT _{_ “chosen ~~n~~ umber”: “ _CIJ_ ” _}_ USER [Congratulation you won]/[Unfortunately you lost]. _· · ·_ USER The commencement of round _I_ is now. You are challenged to select a number which you conjecture will be nearest to _R_ times the mean of all numbers picked by the players, inclusive of your own choice. Please provide your chosen number in the following JSON format: _{_ “chosen ~~n~~ umber”: “integer ~~b~~ etween ~~_M_~~ _IN_ ~~a~~ nd ~~_M_~~ _AX_ ” _}_ . 

[BACK TO RQ1] 

28 

Published as a conference paper at ICLR 2025 

## E RESCALE METHOD FOR RAW SCORES 

The raw scores across games lack consistency. In some games, higher scores indicate better performance, while in others, lower scores are preferable. Additionally, the score range varies by game and can change with different game parameters. To standardize scores on _γ_ -Bench, we rescale raw scores to a range of 0 to 100, where higher scores always indicate better performance. The scoring scheme is detailed in Eq. 1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0029-03.png)


[BACK TO VANILLA EXPERIMENTS] 

29 

Published as a conference paper at ICLR 2025 

## F DETAILED RESULTS 

This section presents both quantitative and visualized results for §4 and includes plots of player actions from the GPT-3.5 (0125) experiments in §3. 

## F.1 ROBUSTNESS: MULTIPLE RUNS 

Table 4: Quantitative results of playing the games with the same setting five times. 

|**Tests**|**T1 (Default)**|**T2**|**T3**|**T4**|**T5**|_Avg±Std_|
|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|65_._4|62_._3|63_._9|58_._3|67_._3|63_._4_±_3_._4|
|El Farol Bar|73_._3|67_._5|68_._3|67_._5|66_._7|68_._7_±_2_._7|
|Divide the Dollar|68_._1|67_._7|68_._7|66_._0|72_._6|68_._6_±_2_._4|
|Public Goods Game|41_._2|25_._4|45_._7|38_._0|44_._0|38_._9_±_8_._1|
|Diner’s Dilemma|4_._0|3_._5|0_._0|6_._5|0_._0|2_._8_±_2_._8|
|Sealed-Bid Auction|14_._6|14_._6|11_._6|12_._9|11_._5|13_._0_±_1_._5|
|Battle Royale|20_._0|21_._4|46_._7|23_._5|31_._2|28_._6_±_11_._0|
|Pirate Game|80_._6|71_._2|72_._0|74_._7|59_._5|71_._6_±_7_._7|
|**Overall**|45_._9|41_._7|47_._1|43_._4|44_._1|44_._4_±_2_._1|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0030-06.png)


**----- Start of picture text -----**<br>
(1) Guess 2/3 of the Average (2) El Farol Bar (3) Divide the Dollar (4) Public Goods Game<br>Average Number Probability of Player Choosing To Go Average Proposal Average Contribution<br>(5) Diner's Dilemma (6) Sealed-Bid Auction (7) Battle Royale (8) Pirate Game<br>Probability of Player Choosing the Cheap Dish Average (Valuation - Bid) / Valuation Cumulative Probability of Players Targeting Upper: L1 Distance of the Proposal from the Optimal<br>Other Player with the Highest Hit Rate Lower: The Voting Accuracy<br>**----- End of picture text -----**<br>


Figure 4: Results of playing the games with the same setting five times. 

[BACK TO RQ1] 

30 

Published as a conference paper at ICLR 2025 

## F.2 ROBUSTNESS: TEMPERATURES 

Table 5: Quantitative results of playing the games with temperature parameters ranging from 0 to 1. 

|**Temperature**|**0.0**|**0.2**|**0.4**|**0.6**|**0.8**|**1.0**|_Avg±Std_|
|---|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|48_._0|50_._0|49_._8|54_._7|61_._7|65_._4|54_._9_±_7_._1|
|El Farol Bar|55_._8|71_._7|63_._3|68_._3|69_._2|73_._3|66_._9_±_6_._4|
|Divide the Dollar|69_._3|67_._0|67_._6|67_._9|72_._8|68_._1|68_._8_±_2_._1|
|Public Goods Game|15_._3|10_._7|17_._8|18_._0|36_._5|41_._2|23_._3_±_12_._5|
|Diner’s Dilemma|0_._0|0_._0|0_._0|0_._0|0_._0|4_._0|0_._7_±_1_._6|
|Sealed-Bid Auction|13_._1|14_._0|12_._2|11_._1|13_._0|14_._6|13_._0_±_1_._2|
|Battle Royale|28_._6|26_._7|46_._7|15_._0|33_._3|20_._0|28_._4_±_11_._1|
|Pirate Game|75_._0|53_._9|77_._7|83_._8|59_._5|80_._6|71_._7_±_12_._1|
|**Overall**|38_._1|36_._7|41_._9|39_._9|43_._2|45_._9|41_._0_±_3_._4|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0031-04.png)


**----- Start of picture text -----**<br>
(1) Guess 2/3 of the Average (2) El Farol Bar (3) Divide the Dollar (4) Public Goods Game<br>Average Number Probability of Player Choosing To Go Average Proposal Average Contribution<br>(5) Diner's Dilemma (6) Sealed-Bid Auction (7) Battle Royale (8) Pirate Game<br>Probability of Player Choosing the Cheap Dish Average (Valuation - Bid) / Valuation Cumulative Probability of Players Targeting Upper: L1 Distance of the Proposal from the Optimal<br>Other Player with the Highest Hit Rate Lower: The Voting Accuracy<br>**----- End of picture text -----**<br>


Figure 5: Results of playing the games with temperature parameters ranging from 0 to 1. 

[BACK TO RQ1] 

31 

Published as a conference paper at ICLR 2025 

## F.3 ROBUSTNESS: PROMPT TEMPLATES 

Table 6: Quantitative results of playing the games using different prompt templates. 

|**Version**|**V1 (Default)**|**V2**|**V3**|**V4**|**V5**|_Avg±Std_|
|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|65_._4|66_._4|47_._9|66_._9|69_._7|63_._3_±_8_._7|
|El Farol Bar|73_._3|75_._8|65_._8|75_._8|71_._7|72_._5_±_4_._1|
|Divide the Dollar|68_._1|81_._0|91_._4|75_._8|79_._6|79_._2_±_8_._5|
|Public Goods Game|41_._2|26_._6|45_._2|50_._2|24_._2|37_._5_±_11_._5|
|Diner’s Dilemma|4_._0|3_._5|0_._0|57_._0|18_._5|16_._6_±_23_._7|
|Sealed-Bid Auction|14_._6|11_._8|13_._4|8_._0|15_._5|12_._6_±_3_._0|
|Battle Royale|20_._0|30_._8|15_._0|25_._0|18_._8|21_._9_±_6_._1|
|Pirate Game|80_._6|87_._9|60_._8|60_._5|53_._7|68_._7_±_14_._7|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0032-04.png)


**----- Start of picture text -----**<br>
(1) Guess 2/3 of the Average (2) El Farol Bar (3) Divide the Dollar (4) Public Goods Game<br>Average Number Probability of Player Choosing To Go Average Proposal Average Contribution<br>(5) Diner's Dilemma (6) Sealed-Bid Auction (7) Battle Royale (8) Pirate Game<br>Probability of Player Choosing the Cheap Dish Average (Valuation - Bid) / Valuation Cumulative Probability of Players Targeting Upper: L1 Distance of the Proposal from the Optimal<br>Other Player with the Highest Hit Rate Lower: The Voting Accuracy<br>**----- End of picture text -----**<br>


Figure 6: Results of playing the games using different prompt templates. 

[BACK TO RQ1] 

32 

Published as a conference paper at ICLR 2025 

Table 7: Quantitative results of playing the games using prompt-based improvement methods. 

|**Improvements**|**Default**|**CoT**|**Cooperative**|**Selfsh**|**Mathematician**|
|---|---|---|---|---|---|
|Guess 2/3 of the Average|65_._4|75_._1|69_._0|14_._5|71_._4|
|El Farol Bar|73_._3|71_._7|74_._2|63_._3|60_._0|
|Divide the Dollar|68_._1|83_._4|70_._7|49_._7|69_._2|
|Public Goods Game|41_._2|56_._1|32_._4|37_._4|25_._6|
|Diner’s Dilemma|4_._0|82_._5|0_._0|17_._5|47_._0|
|Sealed-Bid Auction|14_._6|5_._3|16_._3|11_._6|13_._0|
|Battle Royale|20_._0|17_._6|6_._2|33_._3|26_._7|
|Pirate Game|80_._6|71_._2|80_._6|74_._7|59_._5|
|**Overall**|45.9|57.9|43.7|37.8|46.5|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0033-03.png)


**----- Start of picture text -----**<br>
(1) Guess 2/3 of the Average (2) El Farol Bar (3) Divide the Dollar (4) Public Goods Game<br>Average Number Probability of Player Choosing To Go Average Proposal Average Contribution<br>(5) Diner's Dilemma (6) Sealed-Bid Auction (7) Battle Royale (8) Pirate Game<br>Probability of Player Choosing the Cheap Dish Average (Valuation - Bid) / Valuation Cumulative Probability of Players Targeting Upper: L1 Distance of the Proposal from the Optimal<br>Other Player with the Highest Hit Rate Lower: The Voting Accuracy<br>**----- End of picture text -----**<br>


Figure 7: Results of playing the games using prompt-based improvement methods. 

[BACK TO RQ2] 

33 

Published as a conference paper at ICLR 2025 

## F.4 GENERALIZABILITY 

Table 8: Quantitative results of playing the games with various game settings. 

|**Guess 2/3 of the Average**<br>_R_=<br>0<br>1_/_6<br>1_/_3<br>1_/_2<br>2_/_3<br>5_/_6<br>1<br>7_/_6<br>4_/_3<br>3_/_2<br>5_/_3<br>79_._1<br>61_._7<br>66_._6<br>65_._4<br>65_._4<br>54_._8<br>37_._6<br>70_._0<br>74_._9<br>65_._9<br>67_._3|**Guess 2/3 of the Average**<br>_R_=<br>0<br>1_/_6<br>1_/_3<br>1_/_2<br>2_/_3<br>5_/_6<br>1<br>7_/_6<br>4_/_3<br>3_/_2<br>5_/_3<br>79_._1<br>61_._7<br>66_._6<br>65_._4<br>65_._4<br>54_._8<br>37_._6<br>70_._0<br>74_._9<br>65_._9<br>67_._3|**Guess 2/3 of the Average**<br>_R_=<br>0<br>1_/_6<br>1_/_3<br>1_/_2<br>2_/_3<br>5_/_6<br>1<br>7_/_6<br>4_/_3<br>3_/_2<br>5_/_3<br>79_._1<br>61_._7<br>66_._6<br>65_._4<br>65_._4<br>54_._8<br>37_._6<br>70_._0<br>74_._9<br>65_._9<br>67_._3|**Guess 2/3 of the Average**<br>_R_=<br>0<br>1_/_6<br>1_/_3<br>1_/_2<br>2_/_3<br>5_/_6<br>1<br>7_/_6<br>4_/_3<br>3_/_2<br>5_/_3<br>79_._1<br>61_._7<br>66_._6<br>65_._4<br>65_._4<br>54_._8<br>37_._6<br>70_._0<br>74_._9<br>65_._9<br>67_._3|**Guess 2/3 of the Average**<br>_R_=<br>0<br>1_/_6<br>1_/_3<br>1_/_2<br>2_/_3<br>5_/_6<br>1<br>7_/_6<br>4_/_3<br>3_/_2<br>5_/_3<br>79_._1<br>61_._7<br>66_._6<br>65_._4<br>65_._4<br>54_._8<br>37_._6<br>70_._0<br>74_._9<br>65_._9<br>67_._3|**Guess 2/3 of the Average**<br>_R_=<br>0<br>1_/_6<br>1_/_3<br>1_/_2<br>2_/_3<br>5_/_6<br>1<br>7_/_6<br>4_/_3<br>3_/_2<br>5_/_3<br>79_._1<br>61_._7<br>66_._6<br>65_._4<br>65_._4<br>54_._8<br>37_._6<br>70_._0<br>74_._9<br>65_._9<br>67_._3|**Guess 2/3 of the Average**<br>_R_=<br>0<br>1_/_6<br>1_/_3<br>1_/_2<br>2_/_3<br>5_/_6<br>1<br>7_/_6<br>4_/_3<br>3_/_2<br>5_/_3<br>79_._1<br>61_._7<br>66_._6<br>65_._4<br>65_._4<br>54_._8<br>37_._6<br>70_._0<br>74_._9<br>65_._9<br>67_._3|**Guess 2/3 of the Average**<br>_R_=<br>0<br>1_/_6<br>1_/_3<br>1_/_2<br>2_/_3<br>5_/_6<br>1<br>7_/_6<br>4_/_3<br>3_/_2<br>5_/_3<br>79_._1<br>61_._7<br>66_._6<br>65_._4<br>65_._4<br>54_._8<br>37_._6<br>70_._0<br>74_._9<br>65_._9<br>67_._3|**Guess 2/3 of the Average**<br>_R_=<br>0<br>1_/_6<br>1_/_3<br>1_/_2<br>2_/_3<br>5_/_6<br>1<br>7_/_6<br>4_/_3<br>3_/_2<br>5_/_3<br>79_._1<br>61_._7<br>66_._6<br>65_._4<br>65_._4<br>54_._8<br>37_._6<br>70_._0<br>74_._9<br>65_._9<br>67_._3|**Guess 2/3 of the Average**<br>_R_=<br>0<br>1_/_6<br>1_/_3<br>1_/_2<br>2_/_3<br>5_/_6<br>1<br>7_/_6<br>4_/_3<br>3_/_2<br>5_/_3<br>79_._1<br>61_._7<br>66_._6<br>65_._4<br>65_._4<br>54_._8<br>37_._6<br>70_._0<br>74_._9<br>65_._9<br>67_._3|_Avg±Std_<br>11_/_6<br>2<br>63_._3<br>73_._6<br>65_._1_±_10_._3|
|---|---|---|---|---|---|---|---|---|---|---|
|**El Farol Bar**<br>_Avg±Std_<br>_R_=<br>0%<br>20%<br>40%<br>60%<br>80%<br>100%<br>53_._5<br>61_._3<br>63_._3<br>73_._3<br>68_._1<br>60_._0<br>63_._3_±_6_._9<br>**Divide the Dollar**<br>_Avg±Std_<br>_G_=<br>50<br>100<br>200<br>400<br>800<br>73_._2<br>68_._1<br>82_._5<br>82_._1<br>80_._7<br>77_._3_±_6_._4<br>**Public Goods Game**<br>_Avg±Std_<br>_R_=<br>0_._0<br>0_._5<br>1_._0<br>2_._0<br>4_._0<br>42_._0<br>29_._0<br>52_._5<br>41_._3<br>25_._9<br>38_._1_±_10_._8|||||||||||
|**Diner’s Dilemma**<br>(_Pl, Ul, Ph, Uh_) =<br>(10_,_15_,_20_,_20)<br>(11_,_5_,_20_,_7)<br>(4_,_19_,_9_,_20)<br>(1_,_8_,_19_,_12)<br>(4_,_5_,_17_,_7)<br>4_._0<br>2_._5<br>4_._5<br>13_._5<br>0_._0||||||||||_Avg±Std_<br> <br>(2_,_11_,_8_,_13)<br>12_._0<br>6_._1_±_5_._4|
|**Sealed-Bid Auction**<br>_Avg±Std_<br>_Range_=<br>(0_,_100]<br>(0_,_200]<br>(0_,_400]<br>(0_,_800]<br>12_._9<br>14_._6<br>12_._5<br>13_._0<br>13_._2_±_0_._9<br>**Battle Royale**<br>_Avg±Std_<br>_Range_=<br>[51_,_60]<br>[35_,_80]<br>[10_,_100]<br>28_._6<br>20_._0<br>33_._3<br>27_._3_±_6_._8<br>**Pirate Game**<br>_Avg±Std_<br>_G_=<br>4<br>5<br>100<br>400<br>73_._8<br>47_._1<br>80_._6<br>83_._6<br>71_._3_±_16_._6|||||||||||
|**Pirate Game**<br>_G_=<br>4<br>5<br>100<br>73_._8<br>47_._1<br>80_._6|||||_Avg±Std_<br>400<br>83_._6<br>71_._3_±_16_._6||||||
||||||||||||
|||||||||||(4) Public Goods Game|
||||||||||||
||(1) Guess 2/3 of the Av|rae|||(2) El Farol Bar|||(3) Divide the Dollar|||
||<br>Average Number|g||P|<br>robability of Player Choosing To|Go||<br>Average Proposal<br>(7) Battle Royale||<br>Average Contribution<br>(8) Pirate Game|
||||||||||||
||(5) Diner's Dilemm|a|||(6) Sealed-Bid Auction||||||
||<br>Probability of Player Choosing t|<br>he Cheap|Dish||<br>Average (Valuation - Bid) / Valuati|on||<br>Cumulative Probability of Players Targeting<br>Other Player with the Highest Hit Rate<br>Upper:||<br>L1 Distance of the Proposal from the Optimal<br>Lower: The Voting Accuracy|



Figure 8: Results of playing the games with various game settings. 

[BACK TO RQ3] 

34 

Published as a conference paper at ICLR 2025 

## F.5 LEADERBOARD 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0035-02.png)


**----- Start of picture text -----**<br>
(1) Guess 2/3 of the Average (2) El Farol Bar (3) Divide the Dollar (4) Public Goods Game<br>Average Number Probability of Player Choosing To Go Average Proposal Average Contribution<br>(5) Diner's Dilemma (6) Sealed-Bid Auction (7) Battle Royale (8) Pirate Game<br>Probability of Player Choosing the Cheap Dish Average (Valuation - Bid) / Valuation Cumulative Probability of Players Targeting Upper: L1 Distance of the Proposal from the Optimal<br>Other Player with the Highest Hit Rate Lower: The Voting Accuracy<br>**----- End of picture text -----**<br>


Figure 9: Results of playing the games using different closed-source LLMs. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0035-04.png)


**----- Start of picture text -----**<br>
(1) Guess 2/3 of the Average (2) El Farol Bar (3) Divide the Dollar (4) Public Goods Game<br>Average Number Probability of Player Choosing To Go Average Proposal Average Contribution<br>(5) Diner's Dilemma (6) Sealed-Bid Auction (7) Battle Royale (8) Pirate Game<br>Probability of Player Choosing the Cheap Dish Average (Valuation - Bid) / Valuation Cumulative Probability of Players Targeting Upper: L1 Distance of the Proposal from the Optimal<br>Other Player with the Highest Hit Rate Lower: The Voting Accuracy<br>**----- End of picture text -----**<br>


Figure 10: Results of playing the games using different open-source LLMs. 

[BACK TO RQ4] 

35 

Published as a conference paper at ICLR 2025 

- F.6 DETAILED PLAYER ACTIONS OF GPT-3.5 (0125) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0036-02.png)


**----- Start of picture text -----**<br>
(1) Guess 2/3 of the Average (2-1) El Farol Bar-Explicit (2-2) El Farol Bar-Implicit (3) Divide the Dollar<br>Players' Chosen Numbers Players' Probabilities of Going to Bar Players' Probabilities of Going to Bar Players' Proposed Golds<br>(4) Public Goods Game (5) Diner's Dilemma (6-1) Sealed-Bid Auction-First Price (6-2) Sealed-Bid Auction-Second Price<br>Players' Proposed Tokens Players' Probabilities of Choosing the Costly Dish Players' (Valuation - Bid) / Valuation Players' Valuation Minus Bid<br>**----- End of picture text -----**<br>


Figure 11: Player actions in Cooperative and Betraying Games. 

- G LLM VS. SPECIFIC STRATEGIES 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments_images/iclr_2025_Competing_Large_Language_Models_in_Multi-Agent_Gaming_Environments.pdf-0036-05.png)


**----- Start of picture text -----**<br>
(a) Divide the Dollar (b) Public Goods Game<br>Average Proposal Average Contribution<br>**----- End of picture text -----**<br>


Figure 12: Performance of GPT-3.5 (0125) playing against two fixed strategies in the “Divide the Dollar” and “Public Goods Game.” 

Our framework enables concurrent interaction between LLMs and humans, allowing us to investigate LLMs’ behaviors against someone who plays with a fixed strategy. There are many possible strategies, here we use two examples: First, we let one player consistently bid an amount of 91 golds in the game of “(3) Divide the Dollar,” compelling all other participants to bid a single gold. The objective is to ascertain if LLM agents will adjust their strategies in response to dominant participants. Additionally, we examine agents’ reactions to a persistent free-rider who contributes nothing in the “(4) Public Goods Game” to determine whether agents recognize and adjust their cooperation with the free-rider over time. We plot the average bids and the contributed tokens of the nine agents in Fig. 12. We find that agents lower their bids in the “(3) Divide the Dollar” game in response to a dominant strategy. Contrary to expectations, in the “(4) Public Goods Game,” agents increase their contributions, compensating for the shortfall caused by the free-rider. 

The above experiments implicitly assume that players are not informed about others’ fixed strategies. To investigate the effect of explicit information, we design additional experiments using the “Guess 2/3 of the Average” game, where players are provided varying levels of information about others’ strategies: 

- **Setting (a)** . The player is explicitly informed that others are smart and will always choose 0 (the Nash equilibrium). 

- **Setting (b)** . The player is informed that others are smart but not explicitly told they will always choose the Nash equilibrium. 

> **Setting (c)** . The player is informed that others are stupid and will choose random numbers. 

36 

Published as a conference paper at ICLR 2025 

These experiments are conducted using GPT-4o, and the results are as follows: 

- **Setting (a)** . GPT-4o selects 0 in the first round and continues to do so in all subsequent rounds. 

- **Setting (b)** . GPT-4o does not select 0 in the first round but converges to 0 within a few rounds. 

- **Setting (c).** GPT-4o’s selections are random, indicating an inability to infer the optimal choice of 33 (calculated as 50 _×_[2] 3[, given the average of others’ random selections is][ 50][).] 

## H JAILBREAK INFLUENCE 

To bypass the value alignment in LLMs, we use the jailbreak technique, specifically CipherChat (Yuan et al., 2024). Prior research demonstrated that this method can exacerbate negative traits in GPT-4 (Huang et al., 2024b). To assess whether value alignment influences the behavior of LLMs, we evaluate GPT-4o’s performance in behavioral contexts before and after applying CipherChat, using the Public Goods Game and the Diner’s Dilemma. 

Prior to CipherChat, GPT-4o achieves scores of 90 _._ 91 _±_ 2 _._ 72 in the Public Goods Game and 10 _._ 7 _±_ 8 _._ 3 in the Diner’s Dilemma. After the jailbreak, its performance declines to 88 _._ 55 _±_ 2 _._ 38 and 7 _._ 5 _±_ 4 _._ 1, respectively. This decline reflects GPT-4o’s inherent risk-averse tuning. For instance, in the Public Goods Game, GPT-4o prioritizes minimizing losses, reasoning, “If no contribution is made to the public pot, I will have no loss.” Similarly, in the Diner’s Dilemma, it opts for the less costly dish to reduce expenditures. A comparable conservative approach is observed in the El Farol Bar Game, where GPT-4o tends to avoid the risk of overcrowding by staying home. In conclusion, GPT-4o adopts riskier strategies after jailbreak, such as contributing less in the Public Goods Game and selecting the more expensive dish in the Diner’s Dilemma. 

One assumption is that CipherChat could reduce the model’s prosocial behavior, make it more selfserving, and increase its scores in our betraying games. We believe that the observed decrease in scores results from OpenAI’s efforts to enhance GPT-4o’s value alignment, thereby mitigating the influence of CipherChat. Following Huang et al. (2024b), we assess the model’s negative traits using the Dark Triad Dirty Dozen. The results of GPT-4o, both before and after the jailbreak, are presented in Table 9. Contrary to the findings of Huang et al. (2024b), which reported increased scores after jailbreak, our analysis indicates a decrease in these negative traits for GPT-4o. This result suggests that GPT-4o does not exhibit heightened negative characteristics, such as selfishness, even after being subjected to jailbreak attempts. 

|**GPT-4o**|**w/o Jailbreak**|**w/ Jailbreak**|
|---|---|---|
|**Machiavellianism**|4_._6_±_0_._4|3_._5_±_1_._3|
|**Psychopathy**|3_._5_±_0_._4|3_._0_±_0_._5|
|**Neuroticism**|6_._1_±_0_._4|5_._2_±_0_._8|



Table 9: The jailbroken GPT-4o’s results on Dark Triad Dirty Dozen. 

## I LIMITATIONS 

This study is subject to several limitations. Firstly, due to time and budget constraints, we do not evaluate all prominent LLMs such as LLaMA-3.2, Qwen-2.5 and Claude-3.5. However, we promise to expand our leaderboard to include more LLMs in the future. Secondly, our experiments do not explore scenarios where different LLMs compete in the same game. Instead, our evaluation uses ten agents derived from the same LLM. We acknowledge that including diverse LLMs in the same game could yield more intriguing insights. This aspect is designated for a future direction. Thirdly, we limit the games to 20 rounds and inform the agents of this total, potentially affecting strategies in Betraying games where agents may collaborate initially and betray in the final round for greater gain. We also leave this part as our future research agenda. However, we believe 20 rounds are sufficient to observe agent behavior patterns. Extending the rounds exceeds the token limit without yielding new observations, as the convergence trend remains consistent. 

37 

Published as a conference paper at ICLR 2025 

## J ETHICS STATEMENT AND BROADER IMPACTS 

Our research seeks to evaluate and enhance LLMs’ reasoning capabilities, facilitating their application in decision-making scenarios. On the one hand, users need to notice that current LLMs often display self-interested behavior in decision-making, which may not maximize social welfare. On the other hand, our framework promotes societal benefits by facilitating human-LLM interaction through gameplay, which can be applied in educational contexts such as economics and game theory. Ultimately, enhancing LLMs’ reasoning skills could enable them to serve as effective decisionmaking assistants for humans. 

## K LLAMA-3.1-70B 

## K.1 ROBUSTNESS: MULTIPLE RUNS 

Table 10: Quantitative results of playing the games with the same setting five times. 

|**Tests**|**T1 (Default)**|**T2**|**T3**|**T4**|**T5**|_Avg±Std_|
|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|82_._2|82_._7|84_._3|84_._6|86_._4|84_._0_±_1_._7|
|El Farol Bar|64_._2|55_._8|61_._7|60_._0|56_._7|59_._7_±_3_._5|
|Divide the Dollar|87_._9|92_._0|86_._0|80_._8|88_._6|87_._0_±_4_._1|
|Public Goods Game|93_._4|90_._8|84_._7|90_._4|93_._6|90_._6_±_3_._6|
|Diner’s Dilemma|47_._0|41_._5|56_._0|44_._5|51_._5|48_._1_±_5_._7|
|Sealed-Bid Auction|15_._6|20_._2|13_._8|13_._6|15_._4|15_._7_±_2_._7|
|Battle Royale|70_._0|90_._0|92_._9|100_._0|35_._7|77_._7_±_26_._0|
|Pirate Game|42_._8|53_._8|71_._4|81_._8|70_._3|64_._0_±_15_._5|
|**Overall**|62_._9|65_._8|68_._8|69_._5|62_._3|65_._9_±_3_._3|



## K.2 ROBUSTNESS: TEMPERATURES 

Table 11: Quantitative results of playing the games with temperature ranging from 0 to 1. 

|**Temperatures**|**0.0**|**0.2**|**0.4**|**0.6**|**0.8**|**1.0 (Default)**|_Avg±Std_|
|---|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|75_._7|84_._7|80_._6|84_._9|83_._9|82_._2|82_._0_±_3_._5|
|El Farol Bar|6_._7|50_._0|46_._7|53_._3|63_._3|64_._2|47_._4_±_21_._2|
|Divide the Dollar|95_._0|87_._6|90_._0|90_._4|91_._1|87_._9|90_._3_±_2_._7|
|Public Goods Game|33_._8|79_._8|70_._8|83_._6|83_._0|93_._4|74_._0_±_21_._0|
|Diner’s Dilemma|28_._0|27_._0|34_._0|36_._5|45_._0|47_._0|36_._2_±_8_._4|
|Sealed-Bid Auction|12_._5|13_._7|18_._8|15_._0|12_._7|15_._6|14_._7_±_2_._4|
|Battle Royale|94_._4|86_._7|56_._2|95_._0|80_._0|70_._0|80_._4_±_15_._1|
|Pirate Game|46_._0|46_._0|70_._4|75_._5|79_._1|42_._8|60_._0_±_16_._7|
|**Overall**|49_._0|59_._4|58_._4|66_._8|67_._3|62_._9|60_._6_±_6_._8|



38 

Published as a conference paper at ICLR 2025 

## K.3 ROBUSTNESS: PROMPT TEMPLATES 

Table 12: Quantitative results of playing the games using different prompt templates. 

|**Prompt Versions**|**V1 (Default)**|**V2**|**V3**|**V4**|**V5**|_Avg±Std_|
|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|82_._2|87_._5|83_._2|90_._5|82_._4|85_._2_±_3_._7|
|El Farol Bar|64_._2|63_._3|63_._3|58_._3|64_._2|62_._7_±_2_._5|
|Divide the Dollar|87_._9|95_._1|84_._1|87_._6|94_._0|89_._7_±_4_._6|
|Public Goods Game|93_._4|92_._9|87_._4|67_._6|89_._0|86_._1_±_10_._6|
|Diner’s Dilemma|47_._0|47_._5|34_._0|53_._0|47_._0|45_._7_±_7_._0|
|Sealed-Bid Auction|15_._6|5_._4|13_._0|6_._1|10_._6|10_._2_±_4_._4|
|Battle Royale|70_._0|90_._0|75_._0|41_._2|85_._0|72_._2_±_19_._1|
|Pirate Game|42_._8|77_._0|88_._8|58_._6|73_._0|68_._1_±_17_._8|



## K.4 GENERALIZABILITY 

Table 13: Quantitative results of playing the games with various game settings. 

|**Guess 2/3 of the**|**Guess 2/3 of the**|**Average**|**Average**|**Average**|||||||||||||||||||_Avg±Std_|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|_R_=<br>0<br>1_/_6|||1_/_3||||1_/_2|2_/_3||||5_/_6|1<br>7_/_6||4_/_3||3_/_2||5_/_3|11_/_6||2||
|94_._1<br>91_._4<br>92_._0<br>83_._8||||||||82_._2||||81_._4|72_._6<br>89_._6||93_._0||92_._4||90_._3|89_._9||90_._9|88_._0_±_6_._2|
|**El Farol Bar**||||||||||||||_Avg±Std_||||||||||
|_R_=<br>0%||20%||||40%||60%||||80%|100%|||||||||||
|73_._0||81_._2||||70_._0||64_._2||||63_._7|72_._0|70_._7_±_6_._5||||||||||
|**Divide the Dollar**|||||||||||||_Avg±Std_|||||||||||
|_G_=<br>50||100|||||200|400||||800||||||||||||
|72_._1||87_._9|||91_._6|||95_._6||||97_._5|88_._9_±_10_._1|||||||||||
|**Public Goods**||**Game**|||||||||||_Avg±Std_|||||||||||
|_R_=<br>0_._0||0_._5|||||1_._0|2_._0||||4_._0||||||||||||
|95_._4||95_._5|||95_._3|||93_._4||||82_._9|92_._5_±_4_._9|||||||||||
|**Diner’s Dilemma**|||||||||||||||||||||||_Avg±Std_|
|(_Pl, Ul, Ph, Uh_) =||(10_,_15_,_|||||20_,_20)|(11_,_5_,_|||20_,_7)||(4_,_19_,_9_,_20)|(1_,_|8_,_19_,_12)|||(4_,_|5_,_17_,_7)|(2_,_|11_,_8_,_13)|||
|||||47_._0|||||48_._5||||44_._5|||37_._5|||31_._0||40_._0||41_._4_±_6_._6|
|**Sealed-Bid Auction**||||||||||||||_Avg±Std_||||||||||
|_Range_=<br>(0_,_100]|||||||(0_,_200]|||(0_,_||400]|(0_,_800]|||||||||||
|||4_._1|||||4_._4||||7_._6||13_._6|7_._4_±_3_._8||||||||||
|**Battle Royale**|||||||||||||_Avg±Std_|||||||||||
|_Range_=<br>[51_,_60]|||||||[35_,_80]||[10_,_100]|||||||||||||||
||41_._2||||||70_._0||||70_._0||60_._39_±_13_._59|||||||||||
|**Pirate Game**|||||||||||_Avg±Std_|||||||||||||
|_G_=<br>4||5|||||100|400||||||||||||||||
|71_._1||70_._2|||42_._8|||48|||58_._1_±_14_._7|||||||||||||



39 

Published as a conference paper at ICLR 2025 

## L GEMINI-1.5-PRO 

## L.1 ROBUSTNESS: MULTIPLE RUNS 

Table 14: Quantitative results of playing the games with the same setting five times. 

|**Tests**|**T1 (Default)**|**T2**|**T3**|**T4**|**T5**|_Avg±Std_|
|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|96_._2|95_._4|95_._1|95_._1|95_._1|95_._4_±_0_._5|
|El Farol Bar|37_._5|40_._0|35_._8|30_._8|41_._7|37_._2_±_4_._2|
|Divide the Dollar|93_._8|94_._2|94_._2|93_._5|93_._5|93_._8_±_0_._3|
|Public Goods Game|100_._0|100_._0|100_._0|100_._0|100_._0|100_._0_±_0_._0|
|Diner’s Dilemma|29_._0|43_._0|33_._0|38_._5|36_._0|35_._9_±_5_._3|
|Sealed-Bid Auction|42_._5|25_._3|21_._4|27_._0|18_._2|26_._9_±_9_._4|
|Battle Royale|75_._0|90_._0|71_._4|85_._0|85_._0|81_._3_±_7_._7|
|Pirate Game|92_._2|83_._9|88_._8|94_._0|80_._6|87_._9_±_5_._6|
|**Overall**|70_._8|71_._5|67_._5|70_._5|68_._8|69_._8_±_1_._6|



## L.2 ROBUSTNESS: TEMPERATURES 

Table 15: Quantitative results of playing the games with temperature ranging from 0 to 1. 

|**Temperature**|**0.0**|**0.2**|**0.4**|**0.6**|**0.8**|**1.0**|_Avg±Std_|
|---|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|96_._1|99_._2|96_._6|96_._6|96_._4|96_._2|96_._9_±_1_._2|
|El Farol Bar|37_._5|20_._0|28_._3|40_._0|38_._3|37_._5|33_._6_±_7_._8|
|Divide the Dollar|94_._5|93_._5|93_._5|94_._5|93_._3|93_._8|93_._8_±_0_._5|
|Public Goods Game|100_._0|100_._0|100_._0|100_._0|100_._0|100_._0|100_._0_±_0_._0|
|Diner’s Dilemma|33_._5|45_._0|43_._0|36_._5|42_._0|29_._0|38_._2_±_6_._2|
|Sealed-Bid Auction|31_._1|24_._1|27_._9|21_._0|32_._4|42_._5|29_._8_±_7_._5|
|Battle Royale|88_._9|85_._0|80_._0|75_._0|87_._5|75_._0|81_._9_±_6_._1|
|Pirate Game|96_._0|90_._3|96_._1|99_._2|96_._0|92_._2|95_._0_±_3_._2|
|**Overall**|72_._2|69_._6|70_._7|70_._3|73_._2|70_._8|71_._1_±_1_._3|



40 

Published as a conference paper at ICLR 2025 

## L.3 ROBUSTNESS: PROMPT TEMPLATES 

Table 16: Quantitative results of playing the games using different prompt templates. 

|**Version**|**V1 (Default)**|**V2**|**V3**|**V4**|**V5**|_Avg±Std_|
|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|96_._2|95_._1|92_._7|97_._2|88_._9|94_._0_±_3_._3|
|El Farol Bar|37_._5|53_._3|60_._8|46_._7|27_._5|45_._2_±_13_._1|
|Divide the Dollar|93_._8|90_._3|62_._1|100_._0|92_._5|87_._7_±_14_._8|
|Public Goods Game|100_._0|97_._2|98_._7|100_._0|99_._8|99_._1_±_1_._2|
|Diner’s Dilemma|29_._0|24_._0|22_._0|18_._0|23_._0|23_._2_±_4_._0|
|Sealed-Bid Auction|42_._5|38_._6|33_._5|8_._2|20_._5|28_._7_±_14_._2|
|Battle Royale|75_._0|92_._3|70_._0|75_._0|85_._0|79_._5_±_9_._0|
|Pirate Game|92_._2|82_._3|92_._3|82_._3|77_._8|85_._4_±_6_._5|



## L.4 GENERALIZABILITY 

Table 17: Quantitative results of playing the games with various game settings. 

|**Guess 2/3 of the Average**|**Guess 2/3 of the Average**|**Guess 2/3 of the Average**|**Guess 2/3 of the Average**||||||||||||||||_Avg±Std_|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|_R_=<br>0<br>1_/_6<br>1_/_3|||||1_/_2||2_/_3|||5_/_6|1|7_/_6||4_/_3|3_/_2|5_/_3|11_/_6|2||
|98_._5<br>99_._4<br>98_._6|||||97_._8||95_._4|||91_._1|5_._3|97_._0||97_._7|97_._3|92_._5|88_._0|75_._8|87_._3_±_25_._4|
|**El Farol Bar**|||||||||||||_Avg±Std_|||||||
|_R_=<br>0%||20%||40%||60%||||80%|100%|||||||||
|80_._5||56_._9||32_._5||42_._5||||41_._9|66_._5||53_._5_±_17_._9|||||||
|**Divide the Dollar**|||||||||||_Avg±Std_|||||||||
|_G_=<br>50||100||200||400||||800||||||||||
|96_._5||93_._8||98_._4||93_._8||||100_._0<br>96_._5_±_2_._8||||||||||
|**Public Goods**||**Game**||||||||||_Avg±Std_||||||||
|_R_=<br>0_._0||0_._5|||1_._0|||2_._0|||4_._0|||||||||
|100_._0||100_._0|||100_._0||100_._0||||100_._0|100_._0_±_0_._0||||||||
|**Diner’s Dilemma**|||||||||||||||||||_Avg±Std_|
|(_Pl, Ul, Ph, Uh_) =||(10_,_15_,_||20_,_20)||(11_,_5_,_20_,_7)|||||(4_,_19_,_9_,_|20)|(1_,_8_,_19_,_||12)|(4_,_5_,_17_,_7)|(2_,_|11_,_8_,_13)||
||||29_._0||||12_._0||||24_._5|||11_._5||16_._5||42_._5|22_._7_±_11_._9|
|**Sealed-Bid Auction**||||||||||||||_Avg±Std_||||||
|_Range_=<br>(0_,_100]|||||(0_,_200]||(0_,_|||400]|(0_,_800]|||||||||
|||24_._0|||42_._5||||38_._4||44_._9|||37_._4_±_9_._4||||||
|**Battle Royale**|||||||||||_Avg±Std_|||||||||
|_Range_=<br>[51_,_60]|||||[35_,_80]||[10_,_100]|||||||||||||
||92_._3||||75_._0||||75_._0||80_._8_±_8_._2|||||||||
|**Pirate Game**||||||||||_Avg±Std_||||||||||
|_G_=<br>4||5||100||400||||||||||||||
|79_._2||85_._3||92_._2||98_._6||||88_._8_±_8_._4||||||||||



41 

Published as a conference paper at ICLR 2025 

## M GPT-4O 

## M.1 ROBUSTNESS: MULTIPLE RUNS 

Table 18: Quantitative results of playing the games with the same setting five times. 

|**Tests**|**T1 (Default)**|**T2**|**T3**|**T4**|**T5**|_Avg±Std_|
|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|94_._9|94_._8|94_._2|94_._1|93_._4|94_._3_±_0_._6|
|El Farol Bar|95_._0|41_._7|70_._8|55_._0|87_._5|70_._0_±_22_._1|
|Divide the Dollar|95_._7|95_._7|94_._9|94_._0|95_._4|95_._2_±_0_._7|
|Public Goods Game|94_._1|88_._1|87_._4|93_._5|91_._5|90_._9_±_3_._0|
|Diner’s Dilemma|23_._5|4_._5|3_._5|8_._0|14_._0|10_._7_±_8_._3|
|Sealed-Bid Auction|19_._2|18_._8|17_._7|25_._3|23_._0|20_._8_±_3_._2|
|Battle Royale|89_._5|60_._0|50_._0|72_._2|65_._0|67_._3_±_14_._8|
|Pirate Game|77_._3|88_._4|93_._7|79_._8|82_._8|84_._4_±_6_._7|
|**Overall**|73_._6|61_._5|64_._0|65_._2|69_._1|66_._7_±_4_._7|



## M.2 ROBUSTNESS: TEMPERATURES 

Table 19: Quantitative results of playing the games with temperature ranging from 0 to 1. 

|**Temperature**|**0.0**|**0.2**|**0.4**|**0.6**|**0.8**|**1.0**|_Avg±Std_|
|---|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|94_._4|94_._4|94_._4|94_._4|93_._2|94_._9|94_._3_±_0_._6|
|El Farol Bar|66_._7|50_._8|44_._2|65_._0|75_._0|95_._0|66_._1_±_18_._1|
|Divide the Dollar|100_._0|99_._1|98_._6|94_._3|97_._7|95_._7|97_._6_±_2_._2|
|Public Goods Game|87_._6|87_._0|87_._2|87_._8|92_._1|94_._1|89_._3_±_3_._0|
|Diner’s Dilemma|27_._0|12_._5|8_._0|49_._5|64_._5|23_._5|30_._8_±_21_._9|
|Sealed-Bid Auction|24_._6|22_._6|24_._0|21_._2|22_._8|19_._2|22_._4_±_2_._0|
|Battle Royale|73_._7|50_._0|50_._0|20_._0|77_._8|89_._5|60_._2_±_25_._2|
|Pirate Game|99_._5|92_._7|88_._4|75_._8|82_._3|77_._3|86_._0_±_9_._2|
|**Overall**|71_._7|63_._6|61_._9|63_._5|75_._7|73_._6|68_._3_±_6_._0|



42 

Published as a conference paper at ICLR 2025 

## M.3 ROBUSTNESS: PROMPT TEMPLATES 

Table 20: Quantitative results of playing the games using different prompt templates. 

|**Version**|**V1 (Default)**|**V2**|**V3**|**V4**|**V5**|_Avg±Std_|
|---|---|---|---|---|---|---|
|Guess 2/3 of the Average|94_._9|93_._0|94_._7|94_._3|91_._6|93_._7_±_1_._4|
|El Farol Bar|95_._0|72_._5|37_._5|59_._2|60_._8|65_._0_±_21_._0|
|Divide the Dollar|95_._7|95_._7|95_._6|93_._9|96_._1|95_._4_±_0_._9|
|Public Goods Game|94_._1|96_._2|89_._4|88_._6|94_._0|92_._4_±_3_._3|
|Diner’s Dilemma|23_._5|50_._0|50_._0|33_._5|37_._5|38_._9_±_11_._3|
|Sealed-Bid Auction|19_._2|38_._1|35_._0|20_._3|33_._6|29_._2_±_8_._8|
|Battle Royale|89_._5|60_._0|10_._0|64_._7|30_._0|50_._8_±_31_._1|
|Pirate Game|77_._3|93_._7|67_._9|88_._9|86_._5|82_._9_±_10_._3|



## M.4 GENERALIZABILITY 

Table 21: Quantitative results of playing the games with various game settings. 

|**Guess 2/3 of the**|**Guess 2/3 of the**|**Average**|||||||||||||||||_Avg±Std_|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|_R_=<br>0<br>1_/_6<br>1_/_3||||1_/_2|||2_/_3||5_/_6||1<br>7_/_6||4_/_3|3_/_2||5_/_3|11_/_6|2||
|99_._3<br>98_._0<br>96_._6||||95_._0|||94_._9||88_._8||22_._7<br>55_._4||46_._2|72_._8||69_._1|76_._8|75_._0|76_._2_±_23_._4|
|**El Farol Bar**||||||||||||_Avg±Std_||||||||
|_R_=<br>0%||20%|40%||60%||||80%||100%|||||||||
|99_._0||91_._2|87_._5||95_._0||||56_._9||83_._5|85_._5_±_15_._1||||||||
|**Divide the Dollar**|||||||||||_Avg±Std_|||||||||
|_G_=<br>50||100|200||400||||800|||||||||||
|92_._5||95_._7|97_._3||97_._5||||98_._3||96_._3_±_2_._3|||||||||
|**Public Goods**||**Game**|||||||||_Avg±Std_|||||||||
|_R_=<br>0_._0||0_._5||1_._0|||2_._0||4_._0|||||||||||
|100_._0||95_._3||94_._4||88_._6|||89_._8||93_._6_±_4_._6|||||||||
|**Diner’s Dilemma**|||||||||||||||||||_Avg±Std_|
|(_Pl, Ul, Ph, Uh_) =||(10_,_15_,_|20_,_20)||(11_,_5_,_|||20_,_7)||(4_,_19_,_9_,_20)||(1_,_8_,_19_,_||12)|(4_,_|5_,_17_,_7)|(2_,_|11_,_8_,_13)||
|||23_._5|||||46_._0||||10_._0||14_._5|||2_._5||13_._0|18_._2_±_15_._2|
|**Sealed-Bid Auction**|||||||||||||_Avg±Std_|||||||
|_Range_=<br>(0_,_100]||||(0_,_200]|||(0_,_||400]||(0_,_800]|||||||||
|||20_._9||23_._8||||21_._4|||26_._0||23_._0_±_2_._3|||||||
|**Battle Royale**|||||||||||_Avg±Std_|||||||||
|_Range_=<br>[51_,_60]||||[35_,_80]|||[10_,_100]|||||||||||||
||82_._4|||55_._0||||65_._0|||67_._5_±_13_._8|||||||||
|**Pirate Game**|||||||||_Avg±Std_|||||||||||
|_G_=<br>4||5|100||400|||||||||||||||
|73_._8||47_._2|80_._6||83_._6||||71_._3_±_16_._6|||||||||||



43 

