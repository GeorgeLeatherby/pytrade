# SAA Inference Testing Guide: Transaction Costs & Portfolio Tracking

## Summary of Changes

The environment has been updated to support comprehensive comparison between the SAA agent and a pure buy-and-hold strategy of the selected asset. Three key portfolios are now tracked:

### 1. **Main Portfolio** (`portfolio_state`)
- The agent's actively trading portfolio
- Transaction costs applied on every trade
- Tracked in `episode_buffer.transaction_costs[step]`

### 2. **Comparison Portfolio** (`comparison_portfolio_state`)
- Buy-and-hold reference portfolio starting with the same initial allocation as main portfolio
- **No trading** after initialization - only receives price updates and cash drag
- Represents the performance if no active management occurred
- Tracked in `episode_buffer.comparison_portfolio_value[step]`

### 3. **Selected Asset Buy-and-Hold Portfolio** (`selected_asset_bh_portfolio_state`) **[NEW]**
- **SAA mode only** (`EXECUTION_SINGLE_ASSET_TARGET_POS`)
- Allocates 100% of available cash to the selected asset after paying transaction costs
- Holds until episode end (no trading, no rebalancing)
- Represents a naive buy-and-hold strategy for the selected asset
- Tracked in:
  - `episode_buffer.selected_asset_bh_portfolio_value[step]`
  - `episode_buffer.selected_asset_bh_transaction_costs[step]`

---

## Transaction Cost Tracking

### For Agent/Allocator Trading
**Array:** `episode_buffer.transaction_costs`  
**Type:** `[episode_buffer_length_days]` float32  
**Contents:** Transaction costs incurred by the agent on each step

```python
# Reading transaction costs for the agent
total_agent_costs = np.sum(episode_buffer.transaction_costs)
cost_per_trading_day = np.sum(episode_buffer.transaction_costs[np.where(episode_buffer.transaction_costs > 0)])
num_trading_days = np.sum(episode_buffer.transaction_costs > 0)
avg_cost_per_trade = total_agent_costs / max(num_trading_days, 1)
```

### For Selected Asset Buy-and-Hold
**Array:** `episode_buffer.selected_asset_bh_transaction_costs`  
**Type:** `[episode_buffer_length_days]` float32  
**Contents:**
- **Step 0 (initialization):** Full transaction cost to establish the initial position
- **Steps 1+:** 0.0 (buy-and-hold, no trades)

```python
# Reading selected asset BH transaction costs
bh_init_cost = episode_buffer.selected_asset_bh_transaction_costs[0]  # Only non-zero at init
# Verify no costs after initialization
assert np.sum(episode_buffer.selected_asset_bh_transaction_costs[1:]) == 0.0
```

---

## Initialization Details

### Agent/Main Portfolio
- **Portfolio State:** See `self.portfolio_state` in `reset()`
- **Initial Costs:** Stored in `total_init_tc` (not directly in episode_buffer at step 0)
- **First Record:** At step 0, `transaction_cost=0.0` (costs were paid but not recorded as a step cost)

### Selected Asset BH Portfolio
- **Portfolio State:** `self.selected_asset_bh_portfolio_state`
- **Initial Costs:** `self.selected_asset_bh_init_transaction_cost`
- **First Record:** At step 0, `selected_asset_bh_transaction_costs[0] = self.selected_asset_bh_init_transaction_cost`
- **Cost Calculation Method:** Uses `_initialize_portfolio_with_costs()` with auto-resizing if needed

### Comparison Portfolio
- **Portfolio State:** `self.comparison_portfolio_state`
- **Initial State:** Identical to agent portfolio after transaction costs are applied
- **Initial Costs:** Same as agent (implicit in the starting positions/cash)
- **Cost Tracking:** Not explicitly recorded (already reflected in the initial positions)

---

## Inference Testing: Complete Example

```python
import numpy as np
from src.environment.trading_environment import TradingEnv

# Setup (after environment is trained)
env = TradingEnv(config, market_data_cache, mode='validation')
obs, info = env.reset(asset='SPY')  # SAA mode requires asset specification

# Run episode
episode_done = False
step = 0
while not episode_done:
    action = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_done = terminated or truncated
    step += 1

# Extract transaction cost information
buffer = env.episode_buffer

# ============================================
# AGENT TRANSACTION COSTS
# ============================================
agent_total_cost = np.sum(buffer.transaction_costs)
agent_cost_ratio = agent_total_cost / env.initial_portfolio_value
agent_trading_days = np.sum(buffer.transaction_costs > 0)

print(f"Agent Total Costs: ${agent_total_cost:.2f}")
print(f"Agent Cost Ratio: {agent_cost_ratio * 100:.2f}%")
print(f"Agent Trading Days: {agent_trading_days}")

# ============================================
# SELECTED ASSET BH COSTS
# ============================================
bh_init_cost = buffer.selected_asset_bh_transaction_costs[0]
bh_cost_ratio = bh_init_cost / env.initial_portfolio_value
bh_total_cost = np.sum(buffer.selected_asset_bh_transaction_costs)

print(f"\nBH Init Cost: ${bh_init_cost:.2f}")
print(f"BH Cost Ratio: {bh_cost_ratio * 100:.2f}%")
print(f"BH Total Cost: ${bh_total_cost:.2f}")

# ============================================
# PERFORMANCE METRICS
# ============================================

# Portfolio final values (after all costs deducted)
agent_final_value = buffer.portfolio_values[-1]
bh_final_value = buffer.selected_asset_bh_portfolio_value[-1]
comparison_final_value = buffer.comparison_portfolio_value[-1]

# Net returns (accounting for transaction costs paid)
agent_net_return = (agent_final_value / env.initial_portfolio_value) - 1.0
bh_net_return = (bh_final_value / env.initial_portfolio_value) - 1.0
comparison_net_return = (comparison_final_value / env.initial_portfolio_value) - 1.0

print(f"\nAgent Net Return: {agent_net_return * 100:.2f}%")
print(f"BH Net Return: {bh_net_return * 100:.2f}%")
print(f"Comparison Net Return: {comparison_net_return * 100:.2f}%")

# Cost impact on returns
agent_gross_return = agent_net_return + agent_cost_ratio
bh_gross_return = bh_net_return + bh_cost_ratio

print(f"\nAgent Gross Return (before costs): {agent_gross_return * 100:.2f}%")
print(f"Agent Cost Impact: {-agent_cost_ratio * 100:.2f}%")
print(f"BH Gross Return (before costs): {bh_gross_return * 100:.2f}%")
print(f"BH Cost Impact: {-bh_cost_ratio * 100:.2f}%")

# Alpha vs buy-and-hold
alpha_vs_bh = agent_net_return - bh_net_return
print(f"\nAlpha vs Selected Asset BH: {alpha_vs_bh * 100:.2f}%")
print(f"Alpha vs Comparison Portfolio: {(agent_net_return - comparison_net_return) * 100:.2f}%")

# ============================================
# COST COMPONENT BREAKDOWN (if available)
# ============================================
if hasattr(env, '_last_cost_breakdown'):
    c_comm, c_spread, c_imp, c_fix = env._last_cost_breakdown
    print(f"\nAgent Cost Breakdown:")
    print(f"  Commission: ${c_comm:.2f}")
    print(f"  Spread: ${c_spread:.2f}")
    print(f"  Impact: ${c_imp:.2f}")
    print(f"  Fixed: ${c_fix:.2f}")
```
---

## Verification Checklist for Inference Tester

- [ ] `selected_asset_bh_portfolio_state` is only non-zero if `execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS`
- [ ] `selected_asset_bh_transaction_costs[0]` > 0 (initialization cost)
- [ ] `selected_asset_bh_transaction_costs[1:]` are all 0.0 (no trading after init)


## Data Structure Reference

```
EpisodeBuffer arrays (all shape [episode_buffer_length_days]):
├── portfolio_values              # Main portfolio total value
├── transaction_costs             # Agent trading costs (per step)
├── comparison_portfolio_value    # Buy-and-hold reference from same init
├── selected_asset_bh_portfolio_value     # [NEW] Pure BH of selected asset
└── selected_asset_bh_transaction_costs   # [NEW] BH initialization costs
```
