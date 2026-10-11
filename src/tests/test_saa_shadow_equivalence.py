import copy
import json
import sys
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agents.PPO_portfolio_allocator_weights import ppo_portfolio_allocator_weights_agent as allocator
from environment.trading_environment import MarketDataCache, TradingEnv


@pytest.fixture
def configs():
    paa = json.loads((REPO_ROOT / "src/agents/PPO_portfolio_allocator_weights/config_10019.json").read_text())
    saa = json.loads((REPO_ROOT / "src/agents/RecurrPPO_target_position_agent/config_01056.json").read_text())
    paa["environment"].update(episode_length_days=8, percentage_of_cash_only_starts=1.0)
    saa["environment"].update(episode_length_days=8, execution_min_days_between_trades=2)
    return paa, saa


@pytest.fixture
def cache(configs):
    paa, saa = configs
    assets = sorted(paa["environment"]["assets"])
    names = list(reversed(list(saa["saa_features"]))) + ["allocator_only"]
    num_days = 40
    prices = np.tile(np.linspace(100, 102, num_days, dtype=np.float32)[:, None], (1, len(assets)))
    dates = np.array([str(date) for date in np.arange("2020-01-01", "2020-02-10", dtype="datetime64[D]")])
    return MarketDataCache(
        dates=dates,
        asset_names=assets,
        selected_feature_names=names,
        available_feature_names=names,
        open_prices=prices.copy(),
        high_prices=prices.copy(),
        low_prices=prices.copy(),
        close_prices=prices.copy(),
        volumes=np.full_like(prices, 1e7),
        features=np.random.default_rng(42).normal(0, 0.01, (num_days, len(assets), len(names))).astype(np.float32),
        effr_raw_pct=np.full(num_days, 2.52, dtype=np.float32),
        risk_free_rate_pa=np.full(num_days, 0.0252, dtype=np.float32),
        risk_free_rate_daily=np.full(num_days, 0.0001, dtype=np.float32),
        risk_free_rate_zscore_60d=np.linspace(-1, 1, num_days, dtype=np.float32),
        date_to_index={date: index for index, date in enumerate(dates)},
        asset_to_index={asset: index for index, asset in enumerate(assets)},
        feature_to_index={name: index for index, name in enumerate(names)},
        train_blocks=[],
        validation_blocks=[],
        test_blocks=[],
        block_sampling_weights={},
        num_days=num_days,
        num_assets=len(assets),
        num_features=len(names),
        num_available_features=len(names),
        episode_length_days=8,
        lookback_window=0,
        train_val_split_ratio=0.2,
        purge_length_days=0,
        maybe_provide_sequence=False,
    )


def _options(start=0, length=8):
    return {
        "block_id": "synthetic",
        "episode_start_step": start,
        "episode_length_override": length,
        "force_cash_only_start": True,
    }


def _standalone_cache(cache, saa):
    from dataclasses import replace

    names = cache._select_features_from_config(cache.available_feature_names, saa)
    indices = [cache.feature_to_index[name] for name in names]
    return replace(
        cache,
        features=cache.features[:, :, indices],
        selected_feature_names=names,
        feature_to_index={name: index for index, name in enumerate(names)},
        num_features=len(names),
    )


def _make_paa_adapter(paa, cache, options):
    adapter = allocator.PortfolioEpisodeAdapter(TradingEnv(copy.deepcopy(paa), cache, "validation"))
    adapter.set_plan_queue([options] * 20)
    return adapter


class FixedPolicy:
    def __init__(self, observation_dim, action):
        self.policy = torch.nn.Identity()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(observation_dim,), dtype=np.float32)
        self.action = action

    def predict(self, observations, **kwargs):
        return np.full((len(observations), 1), self.action, dtype=np.float32), None


@pytest.mark.parametrize("zscore,effr_level", [(True, False), (False, False), (True, True), (False, True)])
def test_shadow_accounts_match_standalone(configs, cache, zscore, effr_level):
    paa, saa = configs
    saa["environment"].update(z_score_60_effr=zscore, effr_level_active=effr_level)
    saa_cache = _standalone_cache(cache, saa)
    parent = TradingEnv(paa, cache, "validation")
    parent.configure_saa_shadow_portfolios(saa)
    standalone = [TradingEnv(copy.deepcopy(saa), saa_cache, "validation") for _ in cache.asset_names]
    parent.reset(seed=42, option=_options())
    observations = [
        single.reset(seed=42, option=_options(), asset=asset)[0]
        for single, asset in zip(standalone, cache.asset_names)
    ]
    np.testing.assert_array_equal(parent.get_saa_signal_inputs()["observation"], observations)
    for action in (0.3, 0.001, 1.0, -0.3, -1.0, 0.0, 0.2):
        parent.apply_saa_sub_actions(np.full(cache.num_assets, action, dtype=np.float32))
        observations = [
            single.step(np.array([action], dtype=np.float32), asset=asset)[0]
            for single, asset in zip(standalone, cache.asset_names)
        ]
        parent.step(np.full(cache.num_assets, -100, dtype=np.float32))
        bundle = parent.get_saa_signal_inputs()
        np.testing.assert_array_equal(bundle["observation"], observations)
        for asset_index, single in enumerate(standalone):
            assert bundle["cash"][asset_index] == single.portfolio_state.cash
            assert bundle["shares"][asset_index] == single.portfolio_state.positions[asset_index]
            scale = paa["environment"]["initial_portfolio_value"] / saa["environment"]["initial_portfolio_value"]
            expected_cost = single.episode_buffer.transaction_costs[single.current_step] * scale
            np.testing.assert_allclose(parent._saa_shadow_transaction_costs[asset_index], expected_cost, rtol=1e-6)
    parent.reset(seed=42, option=_options())
    observations = [
        single.reset(seed=42, option=_options(), asset=asset)[0]
        for single, asset in zip(standalone, cache.asset_names)
    ]
    np.testing.assert_array_equal(parent.get_saa_signal_inputs()["observation"], observations)


@pytest.mark.parametrize("multiprocess,norm_obs", [(False, True), (False, False), (True, True)])
def test_recurrent_wrapper_matches_standalone(configs, cache, monkeypatch, multiprocess, norm_obs):
    paa, saa = configs
    torch.set_num_threads(1)
    monkeypatch.setattr(allocator, "_read_saa_training_config", lambda _config: copy.deepcopy(saa))
    saa_cache = _standalone_cache(cache, saa)
    options = [_options(), _options(start=2, length=5)]
    factories = [partial(_make_paa_adapter, paa, cache, option) for option in options]
    raw = SubprocVecEnv(factories, start_method="spawn") if multiprocess else DummyVecEnv(factories)
    observation_dim = saa_cache.num_features + 6 + cache.num_assets
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(observation_dim,), dtype=np.float32)
    norm = VecNormalize(
        DummyVecEnv([lambda: allocator._ObsNormDummyEnv(observation_space)]),
        training=False, norm_reward=False, norm_obs=norm_obs,
    )
    if norm_obs:
        norm.obs_rms.mean = np.linspace(-0.1, 0.1, observation_dim)
        norm.obs_rms.var = np.linspace(0.5, 1.5, observation_dim)
    model = RecurrentPPO(
        "MlpLstmPolicy", norm, n_steps=8, batch_size=8, seed=42, device="cpu",
        policy_kwargs={"lstm_hidden_size": 8, "n_lstm_layers": 1, "net_arch": {"pi": [8], "vf": [8]}},
    )
    original_predict = model.predict
    received = []

    def predict_and_record(observations, **kwargs):
        received.append(observations.copy())
        return original_predict(observations, **kwargs)

    monkeypatch.setattr(model, "predict", predict_and_record)
    wrapper = allocator.SAASignalWrapper(
        raw, model, norm, cache.num_assets, torch.device("cpu"), paa, cache.feature_to_index, 0.3, False
    )
    standalone = [TradingEnv(copy.deepcopy(saa), saa_cache, "validation") for _ in range(2 * cache.num_assets)]
    observations = [
        single.reset(seed=42, option=options[row // cache.num_assets], asset=cache.asset_names[row % cache.num_assets])[0]
        for row, single in enumerate(standalone)
    ]
    states = [None] * len(standalone)
    try:
        wrapper.reset()
        for _ in range(10):
            assert len(received) == len(standalone)
            expected_done = np.zeros(2, dtype=bool)
            for row, single in enumerate(standalone):
                normalized = norm.normalize_obs(observations[row][None])
                np.testing.assert_array_equal(received[row], normalized)
                action, states[row] = original_predict(
                    normalized, state=states[row],
                    episode_start=np.array([single.current_step == 0]), deterministic=True,
                )
                scaled = np.clip(action[0], -1, 1) * 0.3
                env_index, asset_index = divmod(row, cache.num_assets)
                np.testing.assert_array_equal(wrapper._last_signals[env_index, asset_index], scaled)
                observations[row], _, terminated, truncated, _ = single.step(
                    scaled.astype(np.float32), asset=cache.asset_names[asset_index]
                )
                if terminated or truncated:
                    expected_done[env_index] = True
                    observations[row] = single.reset(
                        seed=42, option=options[env_index], asset=cache.asset_names[asset_index]
                    )[0]
                    states[row] = None
            received.clear()
            _, _, dones, _ = wrapper.step(np.full((2, cache.num_assets), -100, dtype=np.float32))
            np.testing.assert_array_equal(dones, expected_done)
    finally:
        wrapper.close()
        norm.close()


def test_shadow_sortino_uses_net_nav_returns(configs, cache):
    paa, _ = configs
    parent = TradingEnv(paa, cache, "validation")
    parent.reset(seed=42, option=_options())
    parent.step(np.full(cache.num_assets, -100, dtype=np.float32))
    assert np.all(parent.episode_buffer.shadow_sortino[1] > 0)
    parent.reset(seed=42, option=_options())
    initial = parent._saa_sub_prev_value.copy()
    parent.apply_saa_sub_actions(np.full(cache.num_assets, 0.3, dtype=np.float32))
    parent.step(np.full(cache.num_assets, -100, dtype=np.float32))
    net_return = parent._saa_sub_prev_value / initial - 1.0
    expected_mean = 1e-6 + parent.sortino_eta * (net_return - 1e-6)
    np.testing.assert_allclose(parent.shadow_running_mean_ema, expected_mean, rtol=1e-5, atol=1e-9)


def test_shadow_fails_if_standalone_terminates_early(configs, cache):
    paa, saa = configs
    cache.close_prices[1:] *= 0.01
    parent = TradingEnv(paa, cache, "validation")
    parent.configure_saa_shadow_portfolios(saa)
    parent.reset(seed=42, option=_options())
    with pytest.raises(RuntimeError, match="terminated before the allocator episode"):
        parent.apply_saa_sub_actions(np.ones(cache.num_assets, dtype=np.float32))


def test_standalone_tester_matches_training_adapter(configs, cache):
    from agents.RecurrPPO_target_position_agent.recurr_ppo_target_pos_agent import SingleAssetEpisodeAdapter
    from agents.test_saa_inference_portfolio.test_saa_inference_shadow_portfolios import _run_asset_episode

    _, saa = configs
    saa_cache = _standalone_cache(cache, saa)
    model = FixedPolicy(saa_cache.num_features + 6 + cache.num_assets, 0.123456)
    reference = _run_asset_episode(
        saa_cache, saa, model, None, "SPY", 0.3, True, torch.device("cpu"), "synthetic", 0, 8, 42
    )
    adapter = SingleAssetEpisodeAdapter(
        TradingEnv(saa, saa_cache, "validation"), action_factor_fn=lambda **kwargs: 0.3
    )
    adapter.set_forced_episode_once("SPY", _options())
    adapter.reset(seed=42)
    for _ in range(7):
        adapter.step(np.array([0.123456], dtype=np.float32))
    np.testing.assert_array_equal(
        reference["actions"], adapter.unwrapped.episode_buffer.actions[:8, cache.asset_to_index["SPY"] + 1]
    )
    np.testing.assert_array_equal(reference["portfolio_values"], adapter.unwrapped.episode_buffer.portfolio_values[:8])


def test_wrapper_clips_after_action_scaling(configs, cache, monkeypatch):
    paa, saa = configs
    monkeypatch.setattr(allocator, "_read_saa_training_config", lambda _config: copy.deepcopy(saa))
    raw = DummyVecEnv([partial(_make_paa_adapter, paa, cache, _options())])
    model = FixedPolicy(32 + 6 + cache.num_assets, 0.75)
    wrapper = allocator.SAASignalWrapper(
        raw, model, None, cache.num_assets, torch.device("cpu"), paa, cache.feature_to_index, 3.0, False
    )
    try:
        wrapper.reset()
        np.testing.assert_array_equal(wrapper._last_signals, np.ones((1, cache.num_assets, 1), dtype=np.float32))
    finally:
        wrapper.close()


def test_recorder_keeps_saa_costs_separate(configs, cache, monkeypatch):
    from utils.paa_val_test_inference import DailyRecorder

    paa, saa = configs
    monkeypatch.setattr(allocator, "_read_saa_training_config", lambda _config: copy.deepcopy(saa))
    raw = DummyVecEnv([lambda: DailyRecorder(_make_paa_adapter(paa, cache, _options()))])
    recorder = raw.envs[0]
    model = FixedPolicy(32 + 6 + cache.num_assets, 0.5)
    wrapper = allocator.SAASignalWrapper(
        raw, model, None, cache.num_assets, torch.device("cpu"), paa, cache.feature_to_index, 0.3, False
    )
    try:
        recorder.arm()
        wrapper.reset()
        for _ in range(7):
            _, _, done, _ = wrapper.step(np.full((1, cache.num_assets), -100, dtype=np.float32))
        episode = recorder.completed
        assert done[0] and episode["n_days"] == 8
        assert episode["sub_tc"][:-1].sum() > 0
        assert episode["paa_tc"][:-1].sum() == 0
        assert np.all(episode["sub_committed"][:-1])
    finally:
        wrapper.close()