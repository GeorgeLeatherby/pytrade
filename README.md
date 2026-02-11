# pytrade-two

Python Multi-Asset Trading DRL Experiment. This is a refactoring/rebasing version to Pytrade One with a much simpler approach.

## Idea
- Dev defines strategy and which features to take
- Dev provides structure of buy and sell triggers
- Algorithm finds concrete thresholds for the triggers
- Reward is always for a whole episode not for a single step!
## Directory Structure

## Configuration (`src/config`)

## Environment Helpers (`src/environment`)

## Data Layout (`src/data`)

## Next Steps

## Quick Start

## License


## Revised data flow of SAA to reduce domain shift when running and loading inside PAA
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