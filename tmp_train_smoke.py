"""Short end-to-end PAA training run exercising callbacks + checkpoints. Not part of the agent."""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from main import load_config
from src.agents.PPO_portfolio_allocator_weights import ppo_portfolio_allocator_weights_agent as paa

CFG = "src/agents/PPO_portfolio_allocator_weights/config_10007.json"
N_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def main():
    from main import load_market_data, verify_requested_features
    from src.environment.trading_environment import MarketDataCache

    config = load_config(CFG)
    config["training"]["n_envs"] = N_ENVS
    config["training"]["total_timesteps"] = 4 * 256 * N_ENVS
    config["training"]["eval_freq"] = 256
    config["training"]["train_log_freq"] = 1
    config["training"]["config_id"] = 99999
    config["portfolio_allocator_agent"]["n_steps"] = 256
    config["portfolio_allocator_agent"]["batch_size"] = 128
    config["critic_pretraining"] = {"enabled": False}
    if not torch.cuda.is_available():
        config["saa_config"]["device"] = "cpu"
        config["portfolio_allocator_agent"]["device"] = "cpu"

    df = load_market_data(config.get("market_data_path") or "src/data/enriched_financial_data.csv")
    verify_requested_features(df, config)
    cache = MarketDataCache.from_dataframe(
        df, config,
        lookback_window=config["environment"]["lookback_window"],
        maybe_provide_sequence=config["environment"].get("maybe_provide_sequence", False),
    )

    summary = paa.run(cache, config)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))

    best_dir = os.path.join(
        "src/agents/PPO_portfolio_allocator_weights/saved_models", summary["tb_log_name"]
    )
    print("\ncheckpoint dir contents:", sorted(os.listdir(best_dir)))
    print("\nTRAIN SMOKE PASSED")


if __name__ == "__main__":
    main()
