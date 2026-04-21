## Thesis Structure: Hierarchical Reinforcement Learning for Multi-Asset Trading
**Candidate:** Simon Hansen \
**Topic:** Separating Temporal Extraction from Cross-Sectional Allocation in DRL  
**Environment:** Custom coded env called PyTrade

---

## 1. Introduction
* **1.1 Motivation:** Transitioning from static forecasting to sequential decision-making in non-stationary markets[cite: 4, 8, 9].
* **1.2 Problem Description:** The challenge of high-dimensional continuous action spaces and the low signal-to-noise ratio in financial data[cite: 13, 19, 21].
* **1.3 Research Question:** Does modularizing temporal and cross-sectional functions improve stability and efficiency?[cite: 24, 37].

> **Prof's Tip:** Avoid "hype" language. Focus on the sequential nature of trading—how today's action constrains tomorrow's opportunity set via transaction costs and inventory[cite: 7].
> **Key Question:** Why is DRL uniquely suited for this problem compared to traditional mean-variance optimization?

## 2. Literature Review
* **2.1 DRL in Quantitative Finance:** Evolution from DQN to PPO in portfolio management[cite: 21, 63, 68].
* **2.2 Temporal Memory in Financial Series:** Comparison of RNNs, LSTMs, and TCNs for state representation[cite: 23, 71].
* **2.3 Attention Mechanisms:** The shift toward modeling time-varying dependencies across assets[cite: 23, 61, 74].
* **2.4 Hierarchical & Modular RL:** Theoretical basis for decomposing complex policies into specialized modules[cite: 24, 31].

> **Prof's Tip:** Don't just list papers. Synthesize them. Group them by how they handle the "Partial Observability" of the market[cite: 13]. Use sources like *López de Prado (2018)* for your methodological foundation[cite: 59].
> **Key Question:** What specific gap in existing monolithic architectures (like standard FinRL) does your hierarchical approach fill?[cite: 18].

## 3. Mathematical Framework & MDP Formulation
* **3.1 Markov Decision Process (MDP):** Formal definition of State $\\mathcal{S}$, Action $\\mathcal{A}$, Transition $\\mathcal{P}$, and Reward $\\mathcal{R}$[cite: 5].
* **3.2 State Space Representation:** Mathematical derivation of stationary features from raw OHLCV.
* **3.3 Action Space:** Continuous portfolio weights $w \\in \\Delta^n$ with Dirichlet constraints.
* **3.4 PPO Objective:** The clipped surrogate objective and the role of the Generalized Advantage Estimator (GAE)[cite: 21, 68].

> **Prof's Tip:** Be extremely precise with your reward function. Is it the simple return, the Sharpe Ratio, or the Differential Sharpe Ratio? Each has different convergence properties in DRL.
> **Key Question:** How do you mathematically ensure the agent's actions remain within valid budget constraints (sum to 1, no shorting unless specified)?

## 4. Proposed Hierarchical Architecture
* **4.1 Layer 1: Temporal Extraction Module (Single-Asset):** A shared-weight recurrent network (LSTM) for per-asset pattern recognition[cite: 33].
* **4.2 Layer 2: Cross-Sectional Allocator (Portfolio-Level):** Self-attention mechanisms to coordinate the extracted features into a portfolio[cite: 34, 35].
* **4.3 Information Flow:** How the temporal embeddings are concatenated and passed to the attention head.

> **Prof's Tip:** This is the "concept" section. Use diagrams to show the flow. Explain why a *shared* network for Layer 1 acts as a regularizer—learning "universal" asset behavior rather than overfitting to one stock.
> **Key Question:** Why is the separation of "history" (Layer 1) and "context" (Layer 2) architecturally superior to a monolithic Transformer?[cite: 24, 37].

## 5. The PyTrade Environment & Implementation
* **5.1 Simulator Design:** Building a Gymnasium-compliant environment for experimental control[cite: 16, 32].
* **5.2 Friction Modeling:** Mathematical implementation of commissions, spreads, and market impact[cite: 15, 30].
* **5.3 Asset Universe:** Selection of the 11 instruments (2000-2025) and the rationale for their diversity[cite: 27, 28].

> **Prof's Tip:** High-quality code is expected. Discuss your "walk-forward" implementation and how you prevent data leakage (purging and embargoing)[cite: 15].
> **Key Question:** How does the simulator handle the "path-dependency" of transaction costs?[cite: 7].

## 6. Empirical Evaluation & Ablation Studies
* **6.1 Baseline:** Cross-sectional only (no temporal memory)[cite: 39].
* **6.2 Monolithic Baseline:** Standard end-to-end recurrent PPO[cite: 42].
* **6.3 The "Control" Test:** Hierarchical setup with randomized temporal signals.
* **6.4 The Proposed Hierarchical Model:** Modular memory + Cross-sectional attention[cite: 40].

> **Prof's Tip:** This is where you earn your grade. If the "Control" (randomized signal) performs well, your model is just finding lucky noise. You must prove the Hierarchical model wins because of *information*.
> **Key Question:** Does the hierarchical model show faster convergence (sample efficiency) compared to the monolithic approach?[cite: 37, 43].

## 7. Results & Financial Interpretation
* **7.1 Performance Metrics:** Sharpe Ratio, Sortino Ratio, and Max Drawdown[cite: 43].
* **7.2 Behavioral Metrics:** Turnover rates, cost decomposition, and concentration (HHI)[cite: 46].
* **7.3 Training Stability:** Variance across multiple seeds and learning curves[cite: 43].

> **Tip:** In finance, a high return with 500% turnover is a failure. Analyze the *Cost Decomposition*[cite: 46]. Did the agent learn to trade less frequently when costs are high?
> **Key Question:** Under which market regimes (e.g., 2008 crisis, 2020 COVID) did the hierarchical model significantly outperform the baselines?

## 8. Conclusion & Future Work
* **8.1 Summary of Findings:** Validating the modular approach for FinRL.
* **8.2 Limitations:** Data frequency constraints and the "sim-to-real" gap[cite: 29].
* **8.3 Future Directions:** Multi-agent extensions or integrating alternative data.

**Tip:** End with a look at the "So what?" Factor. How does this research change the way we build production-grade trading bots?
