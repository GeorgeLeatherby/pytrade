"""Exercise continue_run on the last smoke checkpoint. Not part of the agent."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from main import load_config, load_market_data, verify_requested_features
from src.environment.trading_environment import MarketDataCache
from src.agents.PPO_portfolio_allocator_weights import ppo_portfolio_allocator_weights_agent as paa

CFG = "src/agents/PPO_portfolio_allocator_weights/config_10007.json"
SAVED = "src/agents/PPO_portfolio_allocator_weights/saved_models"

config = load_config(CFG)
config["training"]["n_envs"] = 2
config["training"]["total_timesteps"] = 2 * 256 * 2
config["training"]["eval_freq"] = 256
config["training"]["train_log_freq"] = 1
config["portfolio_allocator_agent"]["n_steps"] = 256
config["portfolio_allocator_agent"]["batch_size"] = 128
if not torch.cuda.is_available():
    config["saa_config"]["device"] = "cpu"
    config["portfolio_allocator_agent"]["device"] = "cpu"

dirs = sorted(d for d in glob.glob(os.path.join(SAVED, "*config_99999*")) if os.path.isdir(d))
model_dir_name = os.path.basename(dirs[-1])
model_path = os.path.join(dirs[-1], "best_model.zip")
print("continuing from:", model_dir_name)

df = load_market_data(config.get("market_data_path") or "src/data/enriched_financial_data.csv")
verify_requested_features(df, config)
cache = MarketDataCache.from_dataframe(
    df, config,
    lookback_window=config["environment"]["lookback_window"],
    maybe_provide_sequence=config["environment"].get("maybe_provide_sequence", False),
)

summary = paa.continue_run(cache, config, model_path, SAVED, model_dir_name)
print(json.dumps(summary, indent=2, default=str))
print("CONTINUE SMOKE PASSED")
