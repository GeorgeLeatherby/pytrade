# SAA Portfolio Inference Test Agent

Testing suit for an individual trained **Single-Asset Agent (SAA)** model executed from `main.py`. Delivers insights into the models behaviour for selected validation (and later test) time periods (unseen in training).

Brings asset specific visibility into the performance of the general SAA model. Shows base asset price development of 100% invested vs. Agents performance for each validation and testing time period. Saves data in json form and as visual analysis. Shows an aggregated view (all assets combined within a time period)

## What It Does

1. **Loads trained SAA model** - Initialises one seperated model per asset (since model is recurrent and assets are individual)
2. **Loads observation normalization stats** - VecNormalize files for consistent preprocessing
3. **Performs run through validation (and testing) time periods** - For each asset and time periods.
4. **Portfolio metrics logging** - Tracks model and financially relevant info for assessing performance

## Key Design Decisions

### Shadow Portfolio
- Each SAA agent per asset now functions in its own enclosed world. There are no other influences from other agents or other assets. This was introduced to adress domain-shift issues when combining SAA agents. The actions of the SAA Agents per asset are still taken as input for the PAA. Just in a detached form.

### Direct SAA Output Usage
- Uses SAA output as target position change (between -1 and 1)
- Maintains LSTM state per asset per episode (critical for RecurrentPPO)
- Deterministic predictions (no exploration noise)

### Portfolio-Level Metrics
- **Total Return**: Aggregate portfolio return across episode for asset.
- **Sharpe Ratio**: Risk-adjusted return of the portfolio
- **Max Drawdown**: Largest peak-to-trough decline
- **Transaction Costs**: Cumulative trading costs
- **Reward/Step**: Average RL reward signal received

## Configuration

Example configuration in `config_00001.json`:

```json
{
  "_comment_design": "Minimal SAA-portfolio-inference test config. ALL data-/env-/training-/agent-related sections are inherited at runtime (by main._maybe_inherit_saa_training_config) from the training config that produced the SAA checkpoint at saa_model_run_dir. The corresponding VecNormalize stats file is derived from the .zip path by replacing '.zip' with '_vecnormalize.pkl'.",

  "test_agent": {
    "saa_model_run_dir": "src\\agents\\RecurrPPO_target_position_agent\\saved_models\\00215_config_01044_26_06_22\\best_model_alpha_return_mean.zip",
    "device": "cpu",
    "deterministic_saa": true
  }
}

```

Each SAA config specifies:
- `saa_model_run_dir`: Relative path of saved model to load and perform all described tasks on. This directly contains the needed config info in the section: "config_01044" to know which config to load for the model init. Relative config path: src\agents\RecurrPPO_target_position_agent\config_01044.json

## Output

The test suite saves all analysis to relative path: `src\agents\RecurrPPO_target_position_agent\saved_models\00215_config_01044_26_06_22` with names: 
- `asset-specific_saa_portfolio_test_[YY_MM_DD_HH_MM].json`
- `aggregate_saa_portfolio_test_[YY_MM_DD_HH_MM].json`

Per time period graphs (500 dpi): 
- `asset-specific_saa_portfolio_test_page1[time_period]_[YY_MM_DD_HH_MM].png`
- `asset-specific_saa_portfolio_test_page2[time_period]_[YY_MM_DD_HH_MM].png`
- `aggregate_saa_portfolio_test_[time_period]_[YY_MM_DD_HH_MM].png`

Aggregated


### Asset-specific analysis and visualisation for each time-period:
Graph (2 A4 page windows (one with 6 one with 5 asset graphs). Each page contains the graphs in 2 vertical columns. Date range is annotated as top title above the subplots. Asset name is annotated above each subplot. In bottom right empty space of 5 graph page place descending list of alphas generated per asset). Include in each asset graph (x-axis days, y-axis metrics)
- Price of asset
- Agent Performance
- Buy-and-hold 100% baseline
- Secondary y-axis indicating the daily agent trading volume (buy/sell color codes)

JSON:
- Initial portfolio value
- Min/Max Portfolio values
- Final portfolio value
- Total return (%)
- Sharpe ratio
- Max drawdown (%)
- Average reward per step
- Cumulative transaction costs
- Number of trades executed
- All data from graph

### Aggregate analysis and visualisation for each time-period:
Graph:
- S&P500 Buy-and-hold vs. all agents combined in single portfolio
JSON
- Combined metrics of all agents as if they would have been a single entity (needed for comparison with PAA later). Shows same metrics then Asset-specific analysis.

## Interpreting Results

### Positive Signals
- Consistent positive returns across episodes
- Low drawdown periods
- High Sharpe ratio (return relative to volatility)
- Moderate transaction costs

### Concerning Signals
- Negative returns (SAAs making poor decisions)
- High drawdown (large losses)
- Low or negative Sharpe ratio
- Excessive transaction costs

### Conflicting Signals
- High variance in returns across episodes (SAAs inconsistent)
- Performance degradation in later episodes (overfitting)
- Asymmetric impact (some assets contributing negatively)
