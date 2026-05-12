## Thesis Structure: Hierarchical Reinforcement Learning for Multi-Asset Trading
**Candidate:** Simon Hansen  
**Topic:** Separating Temporal Extraction from Cross-Sectional Allocation in DRL for portfolio allocation problems \
**Environment:** Custom coded env called PyTrade

---

## 1. Introduction
* **1.1 Motivation:** Transitioning from static forecasting to sequential decision-making in non-stationary markets.
* **1.2 Problem Description:** The challenge of high-dimensional continuous action spaces and the low signal-to-noise ratio in financial data.
* **1.3 Research Question:** Does modularizing temporal (Single-Asset) and cross-sectional (Portfolio-Level) functions improve stability and efficiency?

## 2. Literature Review
* **2.1 DRL in Quantitative Finance:** Evolution from DQN to PPO in portfolio management.
* **2.2 Temporal Memory in Financial Series:** Comparison of RNNs, LSTMs, and TCNs for state representation.
* **2.3 Attention Mechanisms:** The shift toward modeling time-varying dependencies across assets.
* **2.4 Hierarchical & Modular RL:** Theoretical basis for decomposing complex policies into specialized modules.

## 3. Mathematical Framework & MDP Formulation
* **3.1 Markov Decision Process (MDP):** Formal definition of State $\mathcal{S}$, Action $\mathcal{A}$, Transition $\mathcal{P}$, and Reward $\mathcal{R}$.
* **3.2 State Space Representation:** Mathematical derivation of stationary features from raw OHLCV. 
* **3.3 Action Space:** Continuous portfolio weights $w \in \Delta^n$. For the SAA module, actions are defined as the requested change in position relative to the current holding.
* **3.4 Reward Shaping for Strategic Asset Allocation (SAA):**
    * **3.4.1 Multi-Objective Reward Composition:** Decomposing rewards into the "Nested Box" framework.
    * **3.4.2 The Execution Gap:** Mathematical formulation of penalties for requested actions exceeding available cash or portfolio constraints.
    * **3.4.3 Strategic Inertia:** Implementation of a continuous Gaussian "Holding Reward" to discourage bang-bang policy behavior and high-frequency noise trading.
    * **3.4.4 Risk-Adjusted Alpha:** Use of SAA Excess Log Return and Differential Sortino ratios as the core performance signals.

## 4. Proposed Hierarchical Architecture
* **4.1 Layer 1: Temporal Extraction Module (SAA):** 
    * **4.1.1 Dual-Recursion Logic:** Utilizing sb3-contrib RecurrentPPO to manage temporal state transitions alongside a 2-layer LSTM architecture for hierarchical feature extraction (Noise filtering vs. Regime detection).
    * **4.1.2 Training Dynamics and Stability:** Analysis of hyperparameter sensitivity, specifically the relationship between high Entropy (exploration) and Learning Rate decay in preventing policy collapse (Long-only/Short-only traps).
    * **4.1.3 Feature Embedding:** Generation of X-dimensional temporal embeddings that condense asset-specific history into a latent state for the allocator.
* **4.2 Layer 2: Cross-Sectional Allocator (PAA):** Self-attention mechanisms to coordinate the extracted features into a portfolio. Uses the output of frozen SAAs as features.
* **4.3 Information Flow:** How the temporal embeddings are concatenated and passed to the attention head.

## 5. The PyTrade Environment & Implementation
* **5.1 Simulator Design:** Building a Gymnasium-compliant environment for experimental control. 
* **5.2 Friction Modeling:** Mathematical implementation of commissions, spreads, and market impact. Logical proof via transaction cost verification graphs.
* **5.3 Asset Universe:** Selection of the 11 instruments (2000-2025) and the rationale for their diversity including SPY, Gold, Oil, and international indices (EWG, EWQ).

## 6. SAA Verification & Multi-Agent Inference
* **6.1 The "Shared Bucket" Inference Setup:** A testing framework where 11 independent SAA agents interact within a single portfolio context.
    * **6.1.1 Capital Competition:** Analyzing agent behavior when sharing a restricted cash pool.
    * **6.1.2 Out-of-Sample (OOS) Protocol:** Walk-forward testing on unseen data blocks to distinguish true Alpha from memorized noise.
* **6.2 SAA Performance Baselines:**
    * **6.2.1 Asset-Specific Baseline:** Comparing individual agents against a passive Buy & Hold strategy for the corresponding asset.
    * **6.2.2 Portfolio-Level Baseline:** Comparison against a static-weight benchmark portfolio (45% SPY, 20% Gold, etc.).
    * **6.2.3 Market Baseline:** Evaluating the aggregated SAA signals against the SPY total return.
* **6.3 Statistical Robustness:** Identifying the "Peak Alpha" (e.g., 0.0016 mean) and analyzing the subsequent validation decay as a metric for overfitting detection.

## 7. Results & Financial Interpretation
* **7.1 SAA Alpha Analysis:** Evaluating the mean alpha of 0.0016 across 200 validation episodes and its decay over extended training steps.
* **7.2 Behavioral Metrics:** 
    * **7.2.1 Turnover & Execution:** Analysis of how the Execution Gap penalty and L1/L2 penalties reduced "Bang-Bang" oscillations.
    * **7.2.2 Policy Entropy:** Correlation between entropy reduction and the emergence of stable trend-following behaviors.
* **7.3 Critic Performance:** Evaluating "Explained Variance" as a diagnostic tool for Critic health in high-noise environments.
* **7.4 Comparative Performance:** Final performance of the Hierarchical model versus Monolithic PPO and the custom Benchmark Portfolio.

## 8. Conclusion & Future Work
* **8.1 Summary of Findings:** Validating the modular approach for FinRL and the success of "Box-based" reward shaping in stabilizing SAA agents.
* **8.2 Limitations:** Data frequency constraints, the impact of zero-friction training assumptions, and the "sim-to-real" gap.
* **8.3 Future Directions:** Multi-agent extensions for PAA coordination or integrating alternative data into the SAA temporal extraction layer.