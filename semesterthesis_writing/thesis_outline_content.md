# Semesterthesis: Hierarchical Reinforcement Learning for Multi-Asset Trading

**Topic:** Separating single asset temporal signal extraction from multi-asset attention for dynamic allocation in DRL for portfolio allocation problems \
**Environment:** Custom coded env called PyTrade

## 1. Introduction
* **1.1 Motivation:** Financial markets represent highly volatile and complex decision-making environments in artificial intelligence. Research in this domain explores the boundaries of how reinforcement learning agents handle noise, non-stationarity, and multi-layered problem spaces. This thesis objective is not to claim novel alpha, but to explore a new architecture for continous multi-asset trading agents using a hierarchical DRL approach. Finance serves as an interesting and challenging topic for this thesis, not as the primary topic driver. Then proceed to explain: Why buying and holding ETFs might not be the most efficient strategy with the given amount of data and tools available. Explain the transitioning from static forecasting to sequential decision-making in non-stationary markets. Why DRL is a good fit at sequential decisions. Research question which should be incorporated into the motivations naturally: Does modularizing temporal (Single-Asset) and cross-sectional (Portfolio-Level) functions improve stability and efficiency?
* **1.2 Problem Description:** The challenge of high-dimensional continuous action spaces and the low signal-to-noise ratio in financial data. 

## 2. Literature Review (Citations needed! Description of the state of the art)
* **2.1 Quantitative Finance and Deep Reinforcement Learning:** A brief paragraph on the major develoment steps of modern quantitative finance. Using literature this section explains which methods and technologies have been tried. E.g.: the general evolution of deep learning, or discrete actions vs continous actions in portfolio management. How deep learning got incorporated into quantitative finance. Differences between 3-barrier method with discrete trades (Lopez de Prado) and continous trading setup, such as the architecture proposed in this paper.
* **2.2 Temporal Memory in Financial Series:** Comparison of recurrent architectures and algorithms (such as RNNs, LSTMs, and TCNs) for state representation using literature. Should also cover temporal memory in price series and why integer differentiation looses memory while producing stationarity. Mention that the solution to this issue is adressed in section 3 and that it is possible to produce stationary, memory preserving features. This section should clearly state the difference between recurrrent algorithms, recurrent architecture and memory in features using the available literature.
* **2.3 Attention Mechanisms:** Section explaining how attention has been used in quantitative finance so far. Explain how attention is used for temporal signals such as a sequence of words in LLMs. Show how using attention for cross asset relations has been used in the literature and why this is an idea worth persuing.
* **2.4 Hierarchical & Modular RL:** Theoretical basis for decomposing complex policies into specialized modules. Showcase of what has been tried in this domain so far, especially in quantitative finance.

## 3. System Design & Methodology
General introduction of the concept used in PyTrade: Training is conducted in 2 seperate steps. The SAA trades a single randomly choosen asset from the universe. The intend is for the SAA to learn general temporal features across the asset universe. It has a small learned embedding vector to react to the characteristics of each asset and build an internal temporal model. Once a strategy for the SAA has been found which shows sufficient trading results (alpha compared to SPY and proper risk assessment (sharpe)), the SAA is frozen and deepcopied for each asset. The n SAAs are now operating in a shadow portfolio as advisors to the PAA, which has as inputs asset tokens containing raw asset information, SAA state information (action, std, executed trade, shadow portfolio metrics, etc.) and a portfolio token for tha actual portfolio metrics of the live traded portfolio. The same train, validation and test split is used for training the PAA. I am aware that this might result in overly confident SAA interpretations and would introduce some noise to the SAA signals, if necessary. This has not yet been implemented however and will be based on the running research. 


# Section 3.1: Markov Decision Process (MDP) & Observation Fidelity

This section formalizes the decision-making framework for both the Single-Asset Agent (SAA) and Portfolio Allocator Agent (PAA) as Partially Observable Markov Decision Processes (POMDPs). While the environments provide complete market observations at each step, the recurrent architectures used (LSTM in SAA, Self-Attention in PAA) maintain hidden state that is not directly observable, motivating the POMDP formulation. The distinction between state $\mathcal{S}$ (true underlying environment state) and observation $\mathcal{O}$ (agent-accessible information) is critical for understanding information flow and ensuring that learned behaviors transfer correctly when SAA modules are deployed in the PAA.

---

## 3.1.1 Mathematical Framework

### General POMDP Definition

A Partially Observable Markov Decision Process is a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{P}, \mathcal{R}, \mathcal{Z}, \gamma)$ where:

- $\mathcal{S}$: State space (true environment state, not fully observable)
- $\mathcal{A}$: Action space (agent's decision set)
- $\mathcal{O}$: Observation space (agent-accessible signal)
- $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$: State transition probability
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: Reward function
- $\mathcal{Z}: \mathcal{S} \to \mathcal{O}$: Observation model (deterministic in this work)
- $\gamma$: Discount factor (not explicitly used; episodes are finite)

The agent maintains a belief state $b(s)$ over $\mathcal{S}$ and learns a policy $\pi: \mathcal{H} \to \mathcal{A}$ that maps recurrent hidden states $\mathcal{H}$ (maintained by LSTM or Transformer) to actions.

* **3.2 State Space Representation:** Mathematical derivation of stationary features from raw OHLCV. Introduction of memory preserving features (Lopez de Prado). Why just adding more data is not helpful and finding the right input data is so hard. Why Feature Engineering is essential and seen as even more important then backtesting.
* **3.3 Action Space:** For SAA: continous weights indicating the change in position as a fraction of the total portfolio value. The fraction is a fixed number such as 0.2 meaning that an SAA ouput of 1 will lead to the agent buying 20% of portfolio value in the step, if enough cash is available. An output of -0.5 would lead to selling 10% of portfolio value, if possible with current asset inventory. 
For PAA: continuous portfolio weight targets $w \in \Delta^n$, executed in enclosing steps by the env based on the real current portfolio weights. For the SAA module, the action is a single scalar which is defined as the requested change in position relative to the current portfolio notional.
* **3.4 Reward Shaping:**
    * **3.4.1 Multi-Objective Reward Composition:** 
    SAA: Decomposing rewards into the "Nested Box" framework. Learning the hard limitations within a portfolio first (execution gap), followed by generating returns (beta), and finally generating alpha within the single instrument. Basically curriculum learning. RecurrentPPO is very sensitive to changes  in the composition of the individual elements.
    PAA: PAA Reward composition
    * **3.4.2 The Execution Gap:** Mathematical formulation of penalties for requested actions exceeding available cash or portfolio constraints.
    * **3.4.3 Risk-Adjusted Alpha:** Use of SAA Excess Log Return vs simple all-hold baseline (alpha) and Differential Sortino ratios (volatility vs returns) as the core performance signals. The main question to answer here is how not to reward the agent with market beta, since markets on a large scale have generally been moving up.
    PAA: 
* **3.5 Layer 1: Temporal Extraction Module (Single Asset Agent):** 
    * **3.5.1 Dual-Recursion Logic:** Utilizing sb3-contrib RecurrentPPO to manage temporal state transitions alongside a 2-layer LSTM architecture for hierarchical feature extraction (Noise filtering through LSTM vs. Regime detection through recurrent algorithm). The SAA is trained on randomly rotating assets but with a learned small asset-id embedding. In later use the single trained agent is copied per asset. This is necessary due to the statefulness of recurrent agents.
    * **3.5.2 Training Dynamics and Stability:** Analysis of hyperparameter sensitivity, specifically the relationship between high Entropy (exploration) and Learning Rate decay in preventing policy collapse (Long-only/Short-only traps). A key architectural challenge in modular DRL frameworks is avoiding the transition dynamics mismatch when moving from single-agent training to multi-agent deployment. If the temporal extractor (Layer 1) is exposed to global portfolio states like shared cash, this is fine in individual training. But when moving to wiring the individual agents together their recurrent memory (LSTM) becomes susceptible to non-stationary feedback loops caused by exogenous agent actions. This is essentially a domain shift: changing the environment in a way that was not seen during training. The solution for this is adressed in layer 2 description (3.6).
* **3.6 Layer 2: Cross-Sectional Allocator (Portfolio Allocator Agent):** Self-attention mechanisms to coordinate the extracted features into a portfolio. Uses the output of frozen SAAs as features. The individual SAA agents operate in shadow portfolios, that do not have actual influence on the portfolio allocations made by the PAA. They work as indipendent advisors on the individual asset. Data ingestion happens through a portfolio token and n asset tokens. this design enables clean self-attention across assets. Below is a graph indicating the token structure for an asset token.

| PAA INPUT VECTOR (Asset i) | | |
|---|---|---|
| **1. Raw Market Context**<br/>(rsi_14, z_market_dispersion, etc.) | **2. Local Shadow State**<br/>(Shadow Weight, SAA Action) | **3. Latent Memory**<br/>(64D LSTM Hidden St.) |

Provides PAA (layer 2) with the Problem (Market Features), the Proposed Local Solution (SAA Action), and the Confidence/Context Matrix (Hidden State). One risk remains which is feature dillution. This may be mitigated by the networks architecture, or exploring whether to leave out the latent memory, as it is quite large.

* **3.7 Information Flow:** How the temporal embeddings are concatenated and passed to the attention head. \
Framing: Empirical test of Feature Information Density.
I am testing whether a neural network can allocate assets more efficiently when it receives a combination of raw unstructured market data and structured behavioral embeddings generated by an isolated temporal expert. If the PAA outperforms a model trained only on raw features, I have academic proof that the hierarchical decomposition works.

* **3.8 Simulator Design:** Building a Gymnasium-compliant environment for experimental control. 
* **3.9 Friction Modeling:** Mathematical implementation of commissions, spreads, and market impact. Logical proof via transaction cost verification graphs.
* **3.10 Asset Universe:** Selection of the 11 instruments (2000-2025) and the rationale for their diversity including SPY, Gold, Oil, and international indices (EWG, EWQ).

## 4. Experimental Validation & Testing

* **4.1 Out-of-Sample (OOS) Protocol:** Walk-forward testing on unseen data blocks to distinguish true Alpha from memorized noise.
* **4.2 SAA Performance Baselines:**
    * **4.2.1 Cash-Only Baseline:** Comparing individual agent returns against holding only cash.
    * **4.2.2 Buy & Hold Baseline:** Comparing individual agents against passive buy-and-hold performance for the same asset.
    * **4.2.3 50% Invested Baseline:** Comparing agent performance against a half-invested static exposure.
* **4.3 Statistical Robustness SAA:** Discussing the statistical relevance of the found alpha, identifying the "Peak Alpha", and analyzing validation decay as evidence of overfitting.
* **4.4 PAA Performance Baselines:**
    * **4.4.1 Cash-Only Baseline:** Comparing PAA returns against holding only cash.
    * **4.4.2 Buy & Hold Baseline:** Comparing PAA returns against passive unfirmly spread buy-and-hold performance across all assets.
    * **4.4.3 50% Invested Baseline:** Comparing agent performance against a half-invested static exposure.
    * **4.4.4 Industry-grade allocation Buy & Hold Baseline:** Comparing agent performance against a predefined allocation weight portfolio (e.g. 45% SPY, 10% Gold, etc.)
* **4.5 Statistical Robustness PAA:** Discussing the statistical relevance of the found alpha, identifying the "Peak Alpha", and analyzing validation decay as evidence of overfitting.
* **4.6 Baseline (Cross-sectional only):** A Transformer-based portfolio allocator utilizing only same-day features, without augmented temporal memory.
* **4.7 Hierarchical (Modular memory):** The proposed two-level allocator where cross-sectional attention is augmented by pre-extracted recurrent temporal signals.
* **4.8 Control (Randomized signal):** The hierarchical setup but with the SAA signals replaced by structured noise, testing whether performance gains stem from actual historical information rather than mere architectural complexity.
* **4.9 Monolithic Baseline (End-to-end memory):** A standard recurrent PPO allocator that must learn both temporal memory and cross-asset allocation simultaneously. Might introduce a 2-layer LSTM to the PAA before or after transformer?

NOTE: Erst erklären was man vergleichen möchte und warum. Dann erst in einem nächsten Kapitel die Ergebnisse präsentieren.

Idee für baselines: Rule-based investment mit-checken.

## 5. Conclusion & Future Work

* **5.1 Summary of Findings:** Validating the hierarchical approach for Financial RL and the success of "Box-based" reward shaping in stabilizing SAA agents.
* **5.2 Limitations:** Data frequency constraints, the impact of zero-friction training assumptions, and the "sim-to-real" gap.
* **5.3 Future Directions:** Multi-agent extensions for PAA coordination or integrating alternative data into the SAA temporal extraction layer.
