## General rules of behaviour:
Only implement what is requested.
Always reason and seek context of other code and conversation before implementing.
If there is an unclarity, instead of providing code, ask questions to clarify.
State all your assumptions. Explain why you implement each change. Explain why a proposed bug fix should help in context with the existing code and the error.
Verify all handshakes. Verify all shapes of returned objects with the expected shapes.

When working with environment incorporation, only introduce a wrapper if really necessary! If possible use or add to existing wrappers! Make sure to fully understand the relevant environment sections first, before starting to code. If there are unclarities, ask first.

Apply computationally efficient code. Stay in torch native functions whenever possible. Avoid loops, when vectorized computation is possible.

Use in config file defined paramteres always! If a new variable needs to be introduced in the code, this needs to be clearly stated and explained in the answer. THis new variable needs to be added to the config!

Always prefer simple implementations over complicated ones. 
Write the intended/needed shape in commentary next to objects when initiating or returning.

Always stick to already used nomenclature.

Prefer sb3 and PyTorch onboard methods and functions.

Search for documentation of used packages. Often these will have onboard solutions for problems.

## Project specific setup:
DRL hierarchical on-policy agent(s) working with a custom Trading Environment to find portfolio allocation strategies. Step size is daily. Data is from 25 years of real markets of 11 assets. The data is loaded up into a cache object inside main.py and handed over to subsequent functions. The SAA is trained first, then used in inference mode (frozen). SAA is trained generally on all 11 assets rotating randomly. When using SAA in inference mode we need N copies of the SAA agent, each perceiving its specified observation for its assigned asset, due to the SAA being stateful. We then take the N outputs (actions) of the SAA and enrich each asset token of the PAA with their information. The PAA receives N asset tokens and 1 Portfolio Token. The PAA performes asset/portfolio self-attention. In Portfolio_weights execution mode it outputs N+1 weights which are interpreted by the TradingEnv as requested cash_weight + N asset_weights. Softmax Normalisation of PAA actions is performed inside the Environment.

N: # of assets (here 11)
Single Env always only! No multiple Envs in training ever.

# Simulation (single_asset_target_pos_drl_trading.env)
2 different execution_mode: EXECUTION_SINGLE_ASSET_TARGET_POS & EXECUTION_PORTFOLIO_WEIGHTS

In EXECUTION_SINGLE_ASSET_TARGET_POS the defined observation space is (num_features + 3) = (32,)
In EXECUTION_PORTFOLIO_WEIGHTS the defined observation space is (num_assets * num_features + num_portfolio_features) = (348,)
portfolio_features = {weights, alpha, sharpe, drawdown, volatility, turnover, eff.asset.concentration}
with weights[0] = cash_weight and len(weights)= N+1


# SAA setup (recurr_ppo_target_pos_agent.py):
SB3 Recurrent PPO, LSTM. Perceiving a randomly (by env) choosen asset each episode. Intended to find general patterns inside the time sequence data. 

Action space: (1,) between -1 to 1. Is interpreted by TradingEnv as requested change in current position in magnitude of a percentage of total portfolio value. Gets directly executed if fitting to defined thresholds. Thresholds were defined to lower initial churn. 

SAA observation: 
Features to receive from the MarketDataCache object via TradingEnv are marked inside config key: saa_features. They have a boolean which is set to True.
29+3 values
form: (32,)


# PAA setup (ppo_portfolio_allocator_weights_agent.py):
SB3 PPO, Torch Transformer. Performs self-attention over asset tokens (and a single portfolio token). 

Action space: (12,), between 0 to 1. Interpreted by the TradingEnv as requested cash_weight + N asset_weights. Softmax Normalisation of PAA actions is performed inside the Environment.

PAA observation: 
Features to receive from the MarketDataCache object via TradingEnv for asset tokens are marked inside config key: paa_asset_token_features. They have a boolean which is set to True.
N Asset-tokens: 24 asset-specific (defined in paa_asset_token_features) + assets SAA signal + asset_weight
form: (26,)

Features to receive from the MarketDataCache object via TradingEnv for portfolio tokens are marked inside config key: paa_asset_portfolio_features. They have a boolean which is set to True.
1 Portfolio Token: 6 time features (dow_sin, dow_cos, dom_sin, dom_cos, moy_sin, moy_cos), 8 portfolio-wide features: (cash_weight, differential_sharpe, max_drawdown_level_change, volatilty, turnover, effective_concentration (Inverse Herfindahl-Hirschman Index), vix (Volatility Index), alpha (to benchmark. Given as Differential Log return. Maybe one for 1-day one for 20-day))