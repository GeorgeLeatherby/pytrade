## Thoughts

The agent is really only learning something, if the return difference after initialisation constantly is positive. This is the actual alpha the agent can generate. Niels suggested to also challenge this with being fully invested. To kind of check against simply buying and holding the selected asset fully as well. This "hold all" benchmark needs to happen!

If still nothing learned in PAA:
Verify that loaded inference SAA LSTM produce actionable output in setup. Load them in same manner as for PAA and assign X EUR to each asset and then follow the outputs given the training parameters!
## TODO: Actually check this next also for verification purposes!
How to actually proceed:
- Make the saa inference test run in actually same environment as training. Make sure it actually only loads the needed data from the config. Make sure the env interaction regarding the TradeInstructions is correct. Try to use Claude to run verifications and fully review all code that is proposed. 
- Only then try to move to the next stage of testing multiple SAA model instances at the same time and investigate the differences between the multi use and the single use. This will be needed in writin gthe proof for the thesis anyway!


Tweak reward function further. Make it easier, make it more complex, see what works. Write each component of the reward into a tb_metric, so you can verify magnitudes, changes etc. (DONE)

## Make sure to still comply with (PO)MDP premises!
Add as many as possible metrics to portfolio obs which are used to calculate the reward. For the EV the signals need to deliver information on how the reward will be changed. 

Enhance tb metrics. Show min/max/mean/median of action logits. (DONE)


## Use proper purging (purge_length) with walkforward method & (normalization_length)
    Step1: elimante too large normalization windows from data calculation. Define a max length for all calculated features and use it everywhere. (E.g. purge_length=60) (DONE)
    Step2: Only allow feature lookbacks up to purge_length!
    Step3: Make sure that normalization windows do not change because of warmup phases. Data normalized with stats below (normalization_length)

    Fragen für Normalisierung: Wenn jedes asset einzelnd normalisiert wird, geht information zwischen den assets verloren. Wie groß der jeweilige Preisschritt bei jedem asset ist müssten die agents durch den reward herausfinden. Kann dieser Umstand evtl durch das asset embedding abgeschwächt werden? Kann der agent in dem Fall die absoluten Differenzen zwischen den assets erlernen?

## Retrain the SAA LSTM on refit data!

Another approach would be to increase lstm layer size to 2! This might capture temporal patterns better. Literature to check according to LLM:
- Time Series Forecasting (Lai et al., 2018)
- Multi-Scale Temporal Processing (Chung et al., 2016)
- Deep RNNs for Long-Term Dependencies (Pascanu et al., 2013)
- Hierarchical Temporal Representation (Graves et al., 2013)

Change SAA return calculation to a "solvable" POMDP problem. This means:
- Include all metrics used to calculate the reward in the observations
- Include the previous action in the observation to enhance path understanding of the agent

- Check the magnitude of the rewards. PPO tends to work best in the range of rewards being in a 1 range. Simply apply e.g. 100 factor if rewards are in 0.0001 range. (DONE) This is really moving the needle and might need to get tuned even better. There is always a trade-off between speed, applied changes and stability in FinRL. 

- Use log prices in return calculation to try for positive explained variance. (DONE)

- The simple weight observations of cash and assets is likely too simple and experiencing a drift. This is due to the fact that weights are dependent of the development of the other assets as well. SOLUTION: Use absolut notional values and normalize with initial portfolio value. 

- Switch cash weight to log(cash/total value) OR log(cash/starting funds) (DONE)

- Route critical info: last action, log cash ratio, asset size past the mlp so it cannot dilute the information. Active asset size can be expressed as asset log(notional/starting funds). This should reduce relative drift introduced by other assets. (DONE)

- CRITICAL: Verify if last action is also reported correctly to the frozen saa models in PAA mode!

- Value loss should be in range 0.1 to 1 to comfortably learn

## Network ideas:
Apply smoother width sizing within the narrowing critic network path. Try reducing depth of the FeatureExtractor. Try deepening the individual actor critic paths.


## Reward ideas:
- L2 penalty on overall action sizing to avoid pivot towards long only plateau. 
- Reintroduce Volatility awareness through Diff Sortino or other metrics. 
- Add the generated alpha as mix into reward, to allow for market trend substracted signal. Agent behaviour which avoids huge losses when market is in falldown is good, even if it still generates small losses. 
- Apply a penalty for suggesting trades that are impossible. To solve this simply report the difference in requested trade size vs. actually conducted trade. This should introduce a larger box to firstly learn the rules and metrics of the environment the agent lives in. 

## PAA ideas:
- Introduce route critical info of last requested weights! That way the agent might also learn how the env is applying its output! Last action is introduced in SAA but not in PAA yet!


What to do (17.05.2026)
1. Get saa test inference running on actually learned env situation. Make sure to always load the config from the training run dynamically. Really think about what to show graphically in the generated report. 
    Ideas: 
    - price development of asset
    - actions by the agent. 
    - absolut position size in shares
    - notional value of asset. (normalized to 1 on the starting period? How is this ususally reported?)
    - Show a page for each validation period per asset. Show a summary page for each asset. 
    - Include the perofrmance of the benchmarks directly in the normalized graph
    - Show the 3 different scenarios in the graph (0%, 50%, 100% starting allocation)
    - What about env conditions that are hard to replicate? E.g. the allocation of the other assets? Why do we need the other asset anway? Could it be better to isolate only for the signals available from the single asset and take out the information of all other assets?
    - How to work with the shared cash pool? Can changing cash levels break an agent due to statefulness?

2. Introduce last_executed_action as a feature into the SAA. This will make learning the actual rules easier and reduce domain shift when using multiple SAA agents at the same time. 



## How the fuck can we adress the domain shift issues? --> Shadow portfolios!

Instead of training the SAA on the master portfolio's cash and asset holdings, you embed each SAA agent inside its own isolated, virtual single-asset portfolio. This shadow environment exists both during training and during production. 

+-----------------------------------------------------------------------+
| PRODUCTION ENVIRONMENT (Master Portfolio)                             |
| Shared Cash, Real Executions, Friction, Multi-Asset Allocations (PAA) |
+-----------------------------------------------------------------------+
       ^                                                         ^
       | (Allocates real capital based on signals)               |
+-----------------------------------------------------------------------+
| LAYER 2: PAA (Cross-Sectional Allocator)                              |
+-----------------------------------------------------------------------+
       ^                                                         ^
       | (Reads Shadow Hidden States / Hidden Outputs)            |
+------------------------------------+     +----------------------------+
| SAA Agent 1 (SPY Expert)           |     | SAA Agent 2 (Gold Expert)  |
| State: Market + Shadow Portfolio 1 |     | State: Market + Shadow Portfolio 2
| Transition: 100% Local / Isolated  |     | Transition: 100% Local     |
+------------------------------------+     +----------------------------+

When the SAA takes an action it only updates its shadow portfolio when run in the PAA. This setup completely neutralizes the transition dynamics mismatch because the master portfolio's shared cash bucket is entirely hidden from Layer 1. The LSTM's temporal trajectory remains completely in-distribution because its inputs proceed exactly as they did during training.

The Role of the PAA (Layer 2) under this Framework
If the SAA is just trading a virtual sandbox, how does the real portfolio make money?

The SAA's outputs (its shadow allocations or its 64-dimensional LSTM hidden states) are treated by the PAA as unconstrained alpha/regime vectors.

The SPY SAA might signal: "In my shadow world, I am shifting from 20% allocation to 80% allocation because the temporal trend is highly bullish."

The PAA (Layer 2) reads this high-conviction signal across all 11 assets. It is the PAA's job to look at the real master cash bucket, evaluate cross-asset correlations, apply global risk-management constraints, and execute the actual real-world trades.

This circumvents the lower performance ceiling of fixed sub-budgets. The sub-budget is purely a virtual tracking mechanism for the feature space; the PAA retains 100% design freedom to allocate the master portfolio's capital dynamically based on which shadow agent displays the highest conviction.