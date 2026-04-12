# SAA Portfolio Inference Test Agent

## Purpose

Tests whether trained **Single-Asset Agent (SAA)** models produce meaningful trading signals when used directly in a **portfolio context** with **uniform cash allocation**.

This test agent fills a critical gap: SB3 training/validation logs show comprehensive SAA metrics at the individual asset level, but we lacked visibility into how SAA signals behave when combined on a portfolio level.

## What It Does

1. **Loads trained SAA models** - One per asset (using RecurrentPPO from best saved models)
2. **Loads observation normalization stats** - VecNormalize files for consistent preprocessing
3. **Runs validation episodes** - Multiple test episodes across validation data
4. **Uniform cash allocation** - Splits initial portfolio value equally among all assets
5. **Per-step SAA inference** - For each asset, gets the SAA's recommended position change (-1 to +1)
6. **Direct trade execution** - Uses SAA outputs directly to execute trades
7. **Portfolio metrics logging** - Tracks return, Sharpe ratio, drawdown, transaction costs, etc.

## Key Design Decisions

### Uniform Allocation
- Simplifies test: removes need for dynamic weight allocation logic
- Baseline for comparison: what happens if we just average SAA signals?
- Highlight conflicts: if SAAs disagree, uniform averaging shows the impact

### Direct SAA Output Usage
- Uses SAA output as target position change (between -1 and 1)
- Maintains LSTM state per asset per episode (critical for RecurrentPPO)
- Deterministic predictions (no exploration noise)

### Portfolio-Level Metrics
- **Total Return**: Aggregate portfolio return across episode
- **Sharpe Ratio**: Risk-adjusted return of the portfolio
- **Max Drawdown**: Largest peak-to-trough decline
- **Transaction Costs**: Cumulative trading costs
- **Reward/Step**: Average RL reward signal received

## Configuration

Example configuration in `config_00001.json`:

```json
{
  "test_agent": {
    "num_episodes": 5,
    "device": "cpu",
    "saa_configs": {
      "ASSET_1": {
        "saa_run_id": "00001",
        "saa_base_dir": "src/agents/RecurrPPO_target_position_agent/saved_models",
        "saa_config_id": "00001",
        "saa_run_date": "26_04_10"
      },
      "ASSET_2": {
        "saa_run_id": "00001",
        "saa_base_dir": "src/agents/RecurrPPO_target_position_agent/saved_models",
        "saa_config_id": "00001",
        "saa_run_date": "26_04_10"
      }
    }
  },
  "training": {
    "seed": 42
  }
}
```

Each SAA config specifies:
- `saa_run_id`: Run ID of the trained SAA model
- `saa_base_dir`: Base directory where SAA models are saved
- `saa_config_id`: Config ID used when training the SAA
- `saa_run_date`: Date of SAA training run (format: YY_MM_DD)

## Output

The test agent generates a JSON report with:

### Per-Episode Metrics
- Final portfolio value
- Total return (%)
- Sharpe ratio
- Max drawdown (%)
- Average reward per step
- Cumulative transaction costs
- Number of steps executed

### Aggregate Statistics
- Mean return across all episodes
- Standard deviation of returns
- Mean Sharpe ratio
- Mean max drawdown
- Total transaction costs

Report saved to: `saa_portfolio_test_[YY_MM_DD_HH_MM_SS].json`

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

## Comparison Points

When interpreting results, compare against:
1. **Equal-weight rebalancing**: Passive baseline (buy and hold equal allocation)
2. **Benchmark**: Index or market return
3. **Individual SAA validation metrics**: Do portfolio results align with per-asset metrics?

## Limitations

- **Uniform allocation**: Removes dynamic weighting that a real allocator would use
- **Single averaged action**: Uses mean of asset actions; doesn't preserve individual signals
- **No rebalancing logic**: Simple uniform allocation doesn't balance drift
- **Validation data only**: Uses validation time periods, not training or future data

## Usage

```python
# In main.py configuration
config = {
    "agent": "test_saa_inference_portfolio",
    "test_agent": { ... },  # see config example
    "training": { ... }
}

# Run via main.py
result = run(cache, config)
```

## Future Enhancements

- [ ] Weighted allocation based on SAA confidence/accuracy
- [ ] Individual asset action tracking (per-asset contribution analysis)
- [ ] Comparison with PPO allocator on same validation data
- [ ] SAA conflict detection (identify when SAAs disagree)
- [ ] Transaction cost sensitivity analysis
- [ ] Rolling window statistics (track degradation over time)
