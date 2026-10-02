"""Verify batched SAA inference == per-asset independent LSTM states. Not part of the agent."""
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import load_config, load_market_data, verify_requested_features
from src.environment.trading_environment import MarketDataCache
from src.agents.PPO_portfolio_allocator_weights import ppo_portfolio_allocator_weights_agent as paa

CFG = "src/agents/PPO_portfolio_allocator_weights/config_10007.json"
N_ENVS = 2
STEPS = 40


def main():
    config = load_config(CFG)
    config["training"]["n_envs"] = N_ENVS
    if not torch.cuda.is_available():
        config["saa_config"]["device"] = "cpu"
        config["portfolio_allocator_agent"]["device"] = "cpu"
    # Short episodes so the run crosses several episode boundaries.
    config["environment"]["episode_length_days"] = 12

    df = load_market_data(config.get("market_data_path") or "src/data/enriched_financial_data.csv")
    verify_requested_features(df, config)
    cache = MarketDataCache.from_dataframe(
        df, config,
        lookback_window=config["environment"]["lookback_window"],
        maybe_provide_sequence=config["environment"].get("maybe_provide_sequence", False),
    )

    saa_model, saa_vecnorm, saa_device, alf = paa._load_saa_from_config(config["saa_config"])
    vec_train_saa, _ = paa._build_saa_wrapped_envs(
        cache=cache, config=config, seed=7, saa_model=saa_model, saa_vecnorm=saa_vecnorm,
        saa_device=saa_device, saa_action_limiting_factor=alf, num_assets=cache.num_assets,
        tag="equiv",
    )

    w = vec_train_saa
    N = w.num_assets

    # Reference: one independent model + LSTM state per asset, mirroring the old deepcopy design.
    ref_models = [copy.deepcopy(saa_model) for _ in range(N)]
    for m in ref_models:
        m.policy.to(saa_device)
        m.policy.eval()
    ref_states = [None] * N
    ref_episode_start = np.ones((N, N_ENVS), dtype=bool)

    captured = {}
    original_predict = w.saa_model.policy.predict

    def spy_predict(obs, state=None, episode_start=None, deterministic=True):
        captured["obs"] = np.array(obs, copy=True)
        captured["episode_start"] = np.array(episode_start, copy=True)
        return original_predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)

    w.saa_model.policy.predict = spy_predict

    max_abs_diff = 0.0
    boundaries = 0

    obs = w.reset()
    for step in range(STEPS):
        batch_obs = captured["obs"]          # (B*N, D) already normalized
        batch_start = captured["episode_start"]
        B = batch_obs.shape[0] // N

        # Batched signals as produced by the wrapper, recovered from the injected obs column.
        asset_block = N * w.raw_feat_dim
        got = obs[:, :asset_block].reshape(B, N, w.raw_feat_dim)[:, :, -1]  # placeholder, replaced below
        got = w._last_signals[:, :, 0]                                      # (B, N)

        # Reference: run each asset separately with its own state.
        rows = batch_obs.reshape(B, N, -1)
        expected = np.zeros((B, N), dtype=np.float32)
        for a in range(N):
            starts = batch_start.reshape(B, N)[:, a]
            actions, ref_states[a] = ref_models[a].policy.predict(
                rows[:, a, :], state=ref_states[a], episode_start=starts, deterministic=True
            )
            actions_np = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
            expected[:, a] = np.clip(actions_np[:, 0], -1.0, 1.0) * w.action_limiting_factor
            ref_episode_start[a] = starts

        diff = float(np.max(np.abs(got - expected)))
        max_abs_diff = max(max_abs_diff, diff)
        if diff > 1e-6:
            print(f"MISMATCH at step {step}: {diff}")
            print(" got:", got)
            print(" exp:", expected)
            sys.exit(1)

        actions = np.array([w.action_space.sample() for _ in range(B)], dtype=np.float32)
        obs, rewards, dones, infos = w.step(actions)
        if np.any(dones):
            boundaries += int(np.sum(dones))

    print(f"steps={STEPS} episode_boundaries_crossed={boundaries} max_abs_diff={max_abs_diff:.3e}")
    assert boundaries >= 2, "test did not cross enough episode boundaries to be meaningful"
    print("BATCHED SAA == PER-ASSET SAA: PASSED")


if __name__ == "__main__":
    main()
