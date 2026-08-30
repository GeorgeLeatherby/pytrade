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

---

## 3.1.2 Single-Asset Agent (SAA)

### State Space $\mathcal{S}_{\text{SAA}}$

The true state at step $t$ encompasses:

$$\mathcal{S}_{\text{SAA}} = (\mathbf{P}_t, \mathbf{W}_t, \mathbf{h}_t^{\text{lstm}})$$

where:

- $\mathbf{P}_t \in \mathbb{R}^{N}$: Close prices of all $N$ assets at step $t$ (market observation)
- $\mathbf{W}_t = (c_t, \mathbf{x}_t) \in [0,1] \times \mathbb{R}^{N}$: Portfolio state
  - $c_t$: Cash balance (in dollars)
  - $\mathbf{x}_t$: Position vector (number of shares per asset)
- $\mathbf{h}_t^{\text{lstm}} \in \mathbb{R}^{L}$: LSTM hidden state ($L=64$ per layer, 2 layers → $L=128$ effective)

### Observation Space $\mathcal{O}_{\text{SAA}}$

Only on the selected asset. During training, the SAA observes:

$$\mathcal{O}_{\text{SAA},t} = (\mathbf{f}_t, \text{mem}_t) \in \mathbb{R}^{F_{\text{saa}}} \times \mathbb{R}^{4}$$

where:

**Market Features** $\mathbf{f}_t \in \mathbb{R}^{F_{\text{saa}}}$:
- Selected technical indicators from the configuration (default: 29 features)
  - Examples: RSI, MACD, Bollinger Bands, volume ratios, momentum indicators
  - All normalized to zero mean, unit variance before input

**Memory Block** $\text{mem}_t \in \mathbb{R}^{4}$:
  
$$\text{mem}_t = \left(\log\left(\frac{c_t + \epsilon}{c_0}\right), \log\left(\frac{x_t^{*} p_t + \epsilon}{c_0}\right), r_t^{\text{sub}}, a_{t-1}\right)$$

where:
- $c_0 = 1,000,000$: Initial portfolio value (scaling baseline)
- $\epsilon = 10^{-12}$: Numerical stability
- $x_t^{*}$: Number of shares in selected asset
- $p_t$: Price of selected asset
- $r_t^{\text{sub}}$: Daily log return of subportfolio (cash + selected asset only)
- $a_{t-1} \in [-1,1]$: Previous action (enables markovian state)

**Total Observation Dimension**: $\mathcal{O}_{\text{SAA}} \in \mathbb{R}^{F_{\text{saa}} + 4} \approx \mathbb{R}^{33}$

### Action Space $\mathcal{A}_{\text{SAA}}$

Continuous 1D:

$$\mathcal{A}_{\text{SAA}} = [-1, 1]$$

**Interpretation**: Target position change as a fraction of total portfolio value (notional).

$$\Delta \text{notional}_t = a_t \times c_0 = a_t \times 1,000,000$$

Example: $a_t = 0.2$ requests $200,000$ notional change. If current position is 0 shares, agent buys ~6,666 shares of $30/share asset. If current position is 10,000 shares, agent sells ~3,334 shares (moving target).

### Transition Function $\mathcal{P}_{\text{SAA}}$

Step-wise dynamics in the environment:

1. **Execution Phase**: Trade order placed at current close price
   - Cash adjusts for notional + transaction costs
   - Positions update

2. **Overnight Hold**: No action

3. **Next Day Mark-to-Market**: Prices update to new close; portfolio revalued
   - Cash applies daily risk-free carry: $c_t \leftarrow c_t (1 + r_f^{\text{daily}})$
   - Positions unchanged; values recomputed

4. **Observation Generation**: New $\mathcal{O}_t$ computed; LSTM processes it

5. **New LSTM Hidden State**: $\mathbf{h}_{t+1}^{\text{lstm}} = \text{LSTM}(\mathcal{O}_t; \mathbf{h}_t^{\text{lstm}})$

Deterministic conditional: $\mathcal{P}(s_{t+1} | s_t, a_t) = \delta(s_{t+1} - f(s_t, a_t))$

### Reward Function $\mathcal{R}_{\text{SAA}}$

Multi-objective reward composed of nested components (Nested Box curriculum, Section 3.4.1):

$$R_t^{\text{saa}} = \lambda_{\text{exec}} R_t^{\text{execution}} + \lambda_{\text{beta}} R_t^{\text{beta}} + \lambda_{\text{alpha}} R_t^{\text{alpha}} + \lambda_{\text{drawdown}} R_t^{\text{drawdown}}$$

where:

- **$R_t^{\text{execution}}$**: Execution Gap Penalty (Section 3.4.2)
  - Penalizes unfilled orders (e.g., insufficient cash)
  - $R_t^{\text{execution}} \in [-1, 0]$

- **$R_t^{\text{beta}}$**: Simple Portfolio Return
  - $R_t^{\text{beta}} = \log\left(\frac{\text{Portfolio}_t}{\text{Portfolio}_{t-1}}\right)$ (log return)
  - Baseline: returns scaled relative to all-cash position

- **$R_t^{\text{alpha}}$**: Excess Log Return over Risk-Free
  - $R_t^{\text{alpha}} = \log\left(\frac{\text{Portfolio}_t}{\text{Portfolio}_{t-1}}\right) - r_f^{\text{daily}}(t)$
  - $r_f^{\text{daily}}(t)$: Daily risk-free rate derived from EFFR

- **$R_t^{\text{drawdown}}$**: Differential Sortino Ratio (Section 3.4.1)
  - $R_t^{\text{drawdown}} = \text{Sortino}_t - \text{Sortino}_{t-1}$
  - Running mean/variance estimates via EMA with $\eta = 0.0125$

**Scaling**: All components scaled via config knobs ($\text{saa\_excess\_log\_return\_scale} = 30.0$, etc.) to prevent loss landscape collapse.

### Hidden State Evolution

The LSTM hidden state $\mathbf{h}_t^{\text{lstm}} \in \mathbb{R}^{128}$ is maintained across steps:

$$\mathbf{h}_{t+1}^{\text{lstm}}, \mathbf{c}_{t+1}^{\text{lstm}} = \text{LSTM}_2(\mathcal{O}_t; \mathbf{h}_t^{\text{lstm}}, \mathbf{c}_t^{\text{lstm}})$$

where subscript 2 denotes a 2-layer LSTM. This hidden state is **not observable** to the external environment; only the action $a_t$ is emitted. This is the source of partial observability: the environment sees output but not the learned internal model.

### Observation Inclusion Requirement (Critical for Transfer)

When SAA is frozen and deployed in PAA:
- Each frozen SAA maintains its own LSTM hidden state per asset
- The PAA receives **only the SAA action output** $a_t^{(i)}$ for asset $i$
- If the SAA's LSTM were not exposed to changing cash/positions during deployment, the hidden state could drift into an unobserved domain shift (see Section 3.5.2)
- **Solution**: Include explicit log cash and position ratios in every observation to re-anchor hidden state to portfolio dynamics

---

## 3.1.3 Portfolio Allocator Agent (PAA)

### State Space $\mathcal{S}_{\text{PAA}}$

$$\mathcal{S}_{\text{PAA}} = (\mathbf{P}_t, \mathbf{W}_t, \mathbf{h}_t^{\text{saa}}, \mathbf{h}_t^{\text{attn}})$$

where:

- $\mathbf{P}_t$: Market prices (same as SAA)
- $\mathbf{W}_t = (c_t, \mathbf{x}_t)$: Main portfolio state
- $\mathbf{h}_t^{\text{saa}} = \{\mathbf{h}_t^{\text{saa},(i)}\}_{i=1}^{N}$: $N$ frozen SAA LSTM hidden states (one per asset)
  - Each $\mathbf{h}_t^{\text{saa},(i)} \in \mathbb{R}^{128}$
- $\mathbf{h}_t^{\text{attn}} \in \mathbb{R}^{d_{\text{model}}}$: Transformer encoder hidden state

### Observation Space $\mathcal{O}_{\text{PAA}}$

Structured as **token sequence** (Section 3.6):

$$\mathcal{O}_{\text{PAA},t} = (\mathbf{T}_{\text{portfolio}}, \mathbf{T}_{\text{asset}}^{(1)}, \ldots, \mathbf{T}_{\text{asset}}^{(N)})$$

**Portfolio Token** $\mathbf{T}_{\text{portfolio}} \in \mathbb{R}^{d_{\text{model}}}$:

$$\mathbf{T}_{\text{portfolio}} = \text{Linear}_{\text{portfolio}}\left(\begin{bmatrix} \mathbf{f}_t[\text{time\_idx}] \\ c_t / c_0 \\ x_t^{(1)} p_t^{(1)} / c_0 \\ \vdots \\ x_t^{(N)} p_t^{(N)} / c_0 \end{bmatrix}\right)$$

where:
- $\mathbf{f}_t[\text{time\_idx}]$: 6 selected time-series features (e.g., momentum, regime indicators)
- Cash weight: $c_t / c_0$
- Asset values: Notional per asset normalized by initial capital
- **Output dimension**: $d_{\text{model}} = 64$ (configurable)

**Asset Token** $\mathbf{T}_{\text{asset}}^{(i)} \in \mathbb{R}^{d_{\text{model}}}$ for each asset $i$:

$$\mathbf{T}_{\text{asset}}^{(i)} = \text{Linear}_{\text{asset}}\left(\begin{bmatrix} \mathbf{f}_t[F_{\text{paa}}] \\ a_t^{(i)} \\ w_t^{(i)} \end{bmatrix}\right) + \text{Embedding}_{\text{asset\_id}}(i)$$

where:
- $\mathbf{f}_t[F_{\text{paa}}]$: 24 selected market features for asset $i$
- $a_t^{(i)} \in [-1, 1]$: SAA action output (frozen agent)
- $w_t^{(i)}$: Current portfolio weight for asset $i$
- $\text{Embedding}_{\text{asset\_id}}(i)$: Learned $d_{\text{model}}$-dim embedding for asset identity (helps net distinguish assets)
- **Token input dimension**: 24 + 1 + 1 = 26

**Total Observation**: Sequence of $N+1$ tokens, each $d_{\text{model}}$ dimensional.

### Action Space $\mathcal{A}_{\text{PAA}}$

Continuous $N$-dimensional logits:

$$\mathcal{A}_{\text{PAA}} = \mathbb{R}^N$$

Each dimension is a **logit** (not yet normalized). The policy outputs raw logits $\mathbf{z}_t^{(1:N)} \in \mathbb{R}^N$, combined with an implicit cash logit $z_t^{(0)} = 0$ (fixed anchor), then converted to weights via softmax:

$$w_t^{(i)} = \frac{\exp(z_t^{(i)})}{\sum_{j=0}^{N} \exp(z_t^{(j)})}$$

Constrains: $\sum_{i=0}^{N} w_t^{(i)} = 1$, all weights $\geq 0$.

**Interpretation**: Allocate capital across $N$ assets and cash to maximize cumulative reward.

### Transition Function $\mathcal{P}_{\text{PAA}}$

1. **Observation tokenization** (Section 3.6)
2. **Frozen SAA inference**: Each frozen SAA processes its asset observation independently
   - $a_t^{(i)} \leftarrow \text{SAA}_i(\mathcal{O}_t^{(i)}; \mathbf{h}_t^{\text{saa},(i)})$
   - Updates hidden state: $\mathbf{h}_{t+1}^{\text{saa},(i)}$
3. **PAA policy**: Transformer encoder processes token sequence
   - $\mathbf{T} \rightarrow \text{TransformerEncoder}(\mathbf{T}) \rightarrow \mathbf{z}_t^{(1:N)}$
4. **Weight normalization** via softmax
5. **Execution** and **mark-to-market** (same as SAA)
6. **New hidden states**: $\mathbf{h}_{t+1}^{\text{attn}}$ updated via Transformer

Deterministic conditional: $\mathcal{P}(s_{t+1} | s_t, a_t) = \delta(s_{t+1} - f(s_t, a_t))$

### Reward Function $\mathcal{R}_{\text{PAA}}$

$$R_t^{\text{paa}} = \log\left(\frac{\text{Portfolio}_t}{\text{Portfolio}_{t-1}}\right) - r_f^{\text{daily}}(t) - \lambda_{\text{dd}} \Delta \text{Drawdown}_t$$

where:

- **Primary**: Excess log return over risk-free rate (alpha generation)
- **Secondary**: Differential max drawdown penalty
  - $\Delta \text{Drawdown}_t = \log(\text{DD}_t) - \log(\text{DD}_{t-1})$
  - Penalizes worsening maximum observed drawdown

**Simplicity principle**: Allocator reward is **deliberately simple** to avoid policy chasing multiple objectives. Temporal and cross-sectional signal extraction is delegated to SAA; PAA's job is efficient allocation under those signals.

---

## 3.1.4 Information Flow and Domain Shift Mitigation

### SAA Deployment Risk

When a single SAA is copied $N$ times and deployed in PAA, each instance maintains its own LSTM hidden state. If the hidden state evolves based on observations that change distribution (e.g., portfolio cash ratios suddenly differ from training), the agent may enter an **unobserved domain**.

**Example Domain Shift**:
- During training: SAA always starts cash-only; cash weight transitions smoothly
- During deployment in PAA: PAA may allocate 0% to asset $i$ initially, then 50% overnight due to co-movements
- SAA's LSTM was never trained on such jumps; hidden state may encode incorrect beliefs

### Mitigation: Explicit Memory Features

Solution: Always include **log cash ratio** and **log position ratio** explicitly in every observation. These act as sufficient statistics to:

1. **Re-anchor LSTM** to current portfolio state each step
2. **Prevent hidden state drift** even if weights change unexpectedly
3. **Maintain observability** of the "true" sub-portfolio state

Mathematically, these features form a **reset signal** that limits the effective memory horizon of the LSTM to what can be justified by current state:

$$\mathcal{O}_t = (\mathbf{f}_t, \log(\text{cash\_ratio}), \log(\text{position\_ratio}), r_t^{\text{sub}}, a_{t-1})$$

The LSTM cannot sustain long-term beliefs about cash/position that contradict the explicit observations—it must update at every step.

---

## 3.1.5 Summary Table

| Component | SAA | PAA |
|-----------|-----|-----|
| **State** | Market prices + portfolio + LSTM hidden | Market prices + portfolio + N×SAA hidden + Transformer hidden |
| **Observation** | 29 market features + 4 memory bits | N asset tokens (24+1+1 dims) + 1 portfolio token |
| **Action** | Scalar ∈ [-1,1] (target position Δ) | N logits → softmax normalized weights |
| **Reward** | Multi-objective (execution, beta, alpha, Sortino) | Excess log return - drawdown penalty |
| **Policy** | RecurrentPPO (2-layer LSTM) | Transformer encoder + linear policy head |
| **Hidden State** | 128-dim LSTM | Implicit in Transformer; N×128 SAA states |
| **Transfer** | Frozen at step 1 of PAA training | Uses frozen SAAs as feature extractors |
| **Domain Shift Mitigation** | Explicit memory features in obs | SAA states re-anchored each step |

---

## 3.1.6 Observability and Learnability

**Claim**: The hierarchical decomposition maintains **observability** of all decision-relevant state variables.

**Proof Sketch**:
1. **SAA side**: Cash ratio, position ratio, prices are all observed explicitly. LSTM hidden state is internal compression of temporal patterns in **observed** market data. No unobserved state.
2. **PAA side**: Receives SAA outputs (action) + raw market features. Can infer SAA recommendations and market conditions. Portfolio state is fully observed. Transformer hidden state is learned abstract representation of these observations.

**Learnability**:
- SAA learns temporal patterns via supervised signal (single-asset reward) with clear causal attribution
- PAA learns cross-sectional allocation given fixed temporal signals, also with clear attribution (portfolio return)
- Separation prevents conflation of two distinct problems
