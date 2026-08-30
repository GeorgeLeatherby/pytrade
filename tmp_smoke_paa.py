"""Smoke test for PAA multi-env training + deterministic validation sweep. Not part of the agent."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import load_config, load_market_data, verify_requested_features
from src.environment.trading_environment import MarketDataCache
from src.agents.PPO_portfolio_allocator_weights import ppo_portfolio_allocator_weights_agent as paa

CFG = "src/agents/PPO_portfolio_allocator_weights/config_10007.json"
N_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def main():
    config = load_config(CFG)
    config["training"]["n_envs"] = N_ENVS
    import torch
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

    saa_model, saa_vecnorm, saa_device, alf = paa._load_saa_from_config(config["saa_config"])
    vec_train_saa, vec_eval_saa = paa._build_saa_wrapped_envs(
        cache=cache, config=config, seed=42, saa_model=saa_model, saa_vecnorm=saa_vecnorm,
        saa_device=saa_device, saa_action_limiting_factor=alf, num_assets=cache.num_assets,
        tag="smoke",
    )

    from stable_baselines3.common.vec_env import VecNormalize
    vec_train = VecNormalize(vec_train_saa, norm_obs=True, norm_reward=True, clip_obs=10.0,
                             clip_reward=10.0, gamma=0.99, training=True)
    vec_eval = VecNormalize(vec_eval_saa, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            clip_reward=10.0, gamma=0.99, training=False)

    paa_asset_idx = [cache.feature_to_index[f] for f, on in config["paa_asset_token_features"].items() if on]
    paa_port_idx = [cache.feature_to_index[f] for f, on in config["paa_portfolio_token_features"].items() if on]

    config["training"]["total_timesteps"] = 1
    model = paa.build_allocator_model(
        env=vec_train, config=config, num_assets=cache.num_assets,
        raw_feature_dim=cache.num_features,
        paa_asset_token_idx=paa_asset_idx, paa_portfolio_token_idx=paa_port_idx,
    )

    print("\n=== stepping training envs ===")
    from stable_baselines3.common.logger import configure
    model.set_logger(configure("tmp_smoke_logs", ["stdout"]))
    obs = vec_train.reset()
    print("train obs:", obs.shape)
    for _ in range(5):
        actions, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = vec_train.step(actions)
    print("5 steps ok, rewards:", np.asarray(rewards).round(4))

    print("\n=== deterministic validation sweep ===")
    cb = paa.build_allocator_eval_callback(eval_env=vec_eval, config=config, log_dir="tmp_smoke_logs")
    cb.init_callback(model)
    plan = cb._sweep_plan
    print("plan:", json.dumps(plan, indent=2, default=str))

    # Capture every finished validation episode to check cash-only starts and block coverage.
    seen_infos = []
    _orig_collect = cb.eval_step_callback.collect_info

    def _spy(info):
        seen_infos.append(dict(info))
        return _orig_collect(info)

    cb.eval_step_callback.collect_info = _spy

    rewards, lengths = cb._run_validation_sweep()
    seen = []
    original_collect = cb.eval_step_callback.collect_info
    print("episodes:", len(rewards), "expected:", len(plan))
    print("lengths:", lengths)
    expected_lengths = [p["episode_length_override"] for p in plan]
    print("expected lengths:", expected_lengths)
    assert len(rewards) == len(plan), "sweep episode count mismatch"
    # env terminates once current_step reaches length-1, so one env.step() less than the span
    assert sorted(lengths) == sorted(l - 1 for l in expected_lengths), "episode lengths do not match block lengths"

    # Exact plan coverage: one episode per block, no duplicates from auto-reset.
    got_blocks = sorted(i["block_id"] for i in seen_infos)
    want_blocks = sorted(p["block_id"] for p in plan)
    assert got_blocks == want_blocks, f"block coverage mismatch: {got_blocks} vs {want_blocks}"
    got_lengths = sorted(int(i["episode_length"]) for i in seen_infos)
    assert got_lengths == sorted(expected_lengths), f"episode spans mismatch: {got_lengths}"

    # 100% cash start on every validation episode.
    exposures = [float(i["exposure_start"]) for i in seen_infos]
    print("exposure_start:", exposures)
    assert all(abs(e) < 1e-9 for e in exposures), "validation episodes did not start from 100% cash"

    vcb = cb.eval_step_callback
    print("collected episodes:", vcb.eval_episode_count)
    print("blocks:", [b for b, _, _ in vcb._per_block])
    print("terminal_pnl:", [round(p, 2) for _, p, _ in vcb._per_block])
    print("excess_over_spy:", [round(e, 2) for _, _, e in vcb._per_block])
    ok = vcb.flush_metrics(len(plan))
    print("flush ok:", ok)
    print("mean excess over spy:", cb.eval_step_callback.last_excess_over_spy_abs_mean)
    print("mean terminal pnl:", cb.eval_step_callback.last_terminal_pnl_mean)
    print("min terminal pnl:", cb.eval_step_callback.last_terminal_pnl_min)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
