Python Multi-Asset Trading DRL Experiment. As part of a University project on on-policy hierarchical agents this project sets out to explore trainability in chaotic, low-signal-to-noise, partially observable environments like OHLCV-data.


## Idea
Train a much smaller recurrent agent (SAA) on single assets first, then use its output(s) in inference mode on portfolio level when training larger portfolio allocator (PAA).

- SAA (Single asset agent) is trained using sb-contrib Recurrent PPO algorithm on randomly assigned assets. Its output (1,) between -1 to 1 is the desired change in position size of the assigned asset. It is intended to learn general sequence dependent patterns to find buy and sell signals.

- PAA (Portfolio Allocator Agent) is trained using sb3 PPO algorithm. Its action space design is still under development. A first version is (N+1,) with N: # of assets, and a cash weight. In current implementation a CustomPolicy with Transformer architecture is used. It ingests the N SAA models (frozen) outputs and builds them into the asset tokens. Performs self-attention across the asset tokens and the portfolio token.


## Directory Structure

- `src/agents/`  
  DRL agents live here.  
  Includes SAA (single-asset) and PAA (portfolio allocator). Also includes simpler agent which were used to validate environment functionality.

- `src/environment/`  
  Trading environment code and market data cache. Note: Name of env is misleading. Is actually used for all agents.

- `src/config/`  
  JSON configuration files for features, training, and model settings.

- `src/data/`  
  Local data artifacts and run-tracking files (for example `run_id.json`).


## Configuration (`src/config`)

- `training`  
  Run settings such as seed, total timesteps, and evaluation frequency.

- `saa_config`  
  Paths and IDs used to load the frozen SAA model and normalization stats.

- `portfolio_allocator_agent`  
  PPO settings for the allocator (learning rate, batch sizes, entropy, etc.).

- `allocator_transformer`  
  Transformer settings (`d_model`, heads, layers, feedforward size).

- `saa_features`, `paa_asset_token_features`, `paa_portfolio_token_features`  
  Feature toggles used to build SAA and PAA observations.


## Environment (`src/environment`)

- `TradingEnv` is the main RL environment.
- `MarketDataCache` provides market data and feature index mappings.
- SAA training uses a single-asset target-position setup.
- PAA training uses a multi-asset portfolio execution setup.


## Data Layout (`src/data`)

- Data is accessed through `MarketDataCache`.
- Features are selected by name and mapped through `feature_to_index`.
- PAA observations combine:
  - per-asset market features,
  - portfolio-level features,
  - one injected SAA signal per asset.
- Training and validation run in separate environment instances.


## Data flow of SAA to reduce domain shift when running and loading inside PAA

```text
┌─────────────────────────────────────────────────────┐
│ Episode Reset (SINGLE_ASSET_TARGET_POS)             │
├─────────────────────────────────────────────────────┤
│ 1. Randomly allocate across all N assets + cash     │
│ 2. Mark non-target assets as "phantom"              │
│ 3. Initialize comparison portfolio identically      │
└──────────────┬──────────────────────────────────────┘
               │
        ┌──────▼─────────┐
        │ Each Episode   │
        └──────┬─────────┘
               │
    ┌──────────┴─────────────┐
    │                        │
┌───▼────────────────┐  ┌───▼────────────────┐
│ Live Portfolio     │  │ Comparison Portf.  │
│ (trades allowed)   │  │ (passive, prices   │
│                    │  │  + decay only)     │
├────────────────────┤  ├────────────────────┤
│ Cash: $X           │  │ Cash: $X           │
│ Target: Y shares   │  │ Target: Y shares   │
│ Phantoms: Z shares │  │ Phantoms: Z shares │
│ (frozen)           │  │ (frozen)           │
└──────┬─────────────┘  └─────────┬──────────┘
       │                          │
       ├──────────────────────────┤
       │                          │
    ┌──▼────────────────────────▼─┐
    │ OBSERVATION (weight-based)   │
    ├──────────────────────────────┤
    │ rel_cash_w = cash/total_pv   │
    │ rel_target_w = target/total_ │
    │ target_return                │
    │                              │
    │ (includes phantom effects    │
    │  on relative weights)        │
    └──────────┬───────────────────┘
               │
               ▼
         ┌─────────────────┐
         │ SAA LSTM        │
         │ (processes obs) │
         └────────┬────────┘
                  │
        ┌─────────▼──────────┐
        │ Agent Action       │
        │ (target position)  │
        └─────────┬──────────┘
                  │
        ┌─────────▼────────────────────-─┐
        │ REWARD (subportfolio-only)     │
        ├────────────────────────────────┤
        │ • Sortino on {cash+target}     │
        │ • Drawdown on {cash+target}    │
        │ • Alpha vs. comparison         │
        │ (phantoms EXCLUDED)            │
        └──────────┬─────────────────────┘
                   │
    ┌──────────────┴──────────────────┐
    │                                 │
┌───▼──────────────────────────────┐  │
│ METRICS (subportfolio-only)      │  │
├─────────────────────────────────-┤  │
│ return_diff_subpf =              │  │
│  (live_cash + live_target) -     │  │
│  (comp_cash + comp_target)       │  │
│                                  │  │
│ Used for: Best-model selection   │  │
└────────────────────────────────-─┘  │
                                      │
                ┌─────────────────────┘
                │
        ┌───────▼──────────┐
        │ Next Step        │
        │ Prices update    │
        │ Cash decay       │
        │ Weights shift    │
        └──────────────────┘
```

## Data flow of PAA

```text
┌──────────────────────────────────────────────────────────────┐
│ Episode Reset (TradingEnv, portfolio execution mode)         │
├──────────────────────────────────────────────────────────────┤
│ Base observation                                              │
│ • Asset block: N * 30 raw features                           │
│ • Portfolio block: N + 7 features                            │
└──────────────────────────────┬───────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ SAASignalWrapper    │
                    └──────────┬──────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ For each asset i = 1..N at each step                         │
│ 1) Build SAA input (32 dims):                                │
│    29 selected market features + cash_w + asset_w_i + 0      │
│ 2) Normalize with SAA VecNormalize stats (if available)      │
│ 3) Frozen RecurrentPPO_i predicts deterministic signal_i      │
│    (each asset has its own recurrent state)                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ Augmented allocator observation                               │
│ • Asset block: N * 31  (30 raw + 1 SAA signal)               │
│ • Portfolio block: N + 7                                     │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ VecNormalize (allocator env)                                  │
│ • Train: normalize obs + reward, update running stats         │
│ • Eval : normalize obs only, stats frozen                     │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ SAATokenizer                                                   │
├───────────────────────────────────────────────────────────────┤
│ Asset token (26): 24 selected raw feats + SAA signal + weight │
│ Portfolio token: 6 time feats (from asset 0) + portfolio blk  │
│ Linear projection -> d_model                                  │
│ Token sequence: [portfolio token, asset tokens]              │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ AttentionEngine (Transformer encoder)                         │
│ Self-attention over N+1 tokens                                │
│ Actor latent: flatten tokens 0..N-1                           │
│ Critic latent: token N                                         │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ PPO action + environment step                                 │
├───────────────────────────────────────────────────────────────┤
│ Action: (N+1) continuous values (N assets + cash)             │
│ Environment executes rebalance and normalizes to valid weights│
│ Returns reward, done, and info metrics                        │
└──────────────────────────────┬───────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Callbacks           │
                    ├─────────────────────┤
                    │ • Log train metrics │
                    │ • Run validation    │
                    │ • Save best model   │
                    └─────────────────────┘
```


## License

This project is licensed under the **PolyForm Noncommercial 1.0.0** license.  
You may use, copy, modify, and distribute this software for **noncommercial purposes only**.  
Commercial use is not allowed without a separate written commercial license from the copyright holder.  
See [LICENSE](./License.md) for full terms.  
For commercial licensing, contact: simonhansen230(at)gmail.com