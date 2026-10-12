import copy
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agents.PPO_portfolio_allocator_weights import ppo_portfolio_allocator_weights_agent as allocator
from environment.trading_environment import EpisodeBuffer, MarketDataCache, TradingEnv


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
        if len(observations) == 2 * cache.num_assets:
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
            assert len(received) == 1
            expected_done = np.zeros(2, dtype=bool)
            for row, single in enumerate(standalone):
                normalized = norm.normalize_obs(observations[row][None])
                np.testing.assert_allclose(received[0][row:row + 1], normalized, rtol=1e-6, atol=1e-7)
                action, states[row] = original_predict(
                    normalized, state=states[row],
                    episode_start=np.array([single.current_step == 0]), deterministic=True,
                )
                scaled = np.clip(action[0], -1, 1) * 0.3
                env_index, asset_index = divmod(row, cache.num_assets)
                np.testing.assert_allclose(
                    wrapper._last_signals[env_index, asset_index], scaled, rtol=1e-6, atol=1e-7
                )
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


def test_allocator_can_request_ninety_percent_cash(configs, cache):
    paa, _ = configs
    parent = TradingEnv(paa, cache, 'validation')
    parent.reset(seed=42, option=_options())
    np.testing.assert_array_equal(parent.action_space.low, np.full(cache.num_assets, -5))
    parent.step(np.full(cache.num_assets, -5, dtype=np.float32))
    target_cash = parent.episode_buffer.actions[1, 0]
    assert target_cash >= 0.9
    assert target_cash == pytest.approx(1 / (1 + cache.num_assets * np.exp(-5)))


@pytest.mark.parametrize('lookback', [0, 3])
def test_risk_windows_include_current_step_and_exclude_warmup(lookback):
    buffer = EpisodeBuffer(12, 1, lookback, bool(lookback))
    for step, value in enumerate([100., 80., 70.]):
        buffer.record_portfolio_snapshot(
            step, value, np.array([0., 1.]), np.array([1.]),
            0. if step == 0 else -0.2, np.array([value]),
        )
        if step > 0:
            assert buffer.calculate_max_drawdown(10) == pytest.approx(1 - value / 100)
            assert buffer.saa_calculate_max_drawdown(0, 10) == pytest.approx(1 - value / 100)
    np.testing.assert_allclose(buffer.get_returns_window(2), [-0.2, -0.2])


def test_holdings_encoding_distinguishes_empty_and_full(configs, cache):
    _, saa = configs
    single_cache = _standalone_cache(cache, saa)
    single = TradingEnv(saa, single_cache, 'validation')
    cash_observation, _ = single.reset(seed=42, option=_options(), asset='SPY')
    asset_index = cache.asset_to_index['SPY']
    single.episode_buffer.portfolio_weights[0] = 0
    single.episode_buffer.portfolio_weights[0, asset_index + 1] = 1
    asset_observation = single.get_observation_single_step()
    offset = single_cache.num_features
    np.testing.assert_allclose(cash_observation[offset:offset + 2], [np.log(2), 0])
    np.testing.assert_allclose(asset_observation[offset:offset + 2], [0, np.log(2)])
    assert not np.array_equal(cash_observation, asset_observation)


def test_saa_return_is_percentage_nav_change(configs, cache):
    _, saa = configs
    saa['environment'].update(execution_min_days_between_trades=0)
    cache.close_prices[:] = 100
    cache.risk_free_rate_daily[:] = 0
    single_cache = _standalone_cache(cache, saa)
    single = TradingEnv(saa, single_cache, 'validation')
    single.reset(seed=42, option=_options(), asset='SPY')
    for action in [0.1, -0.1, 0.0]:
        before = single.portfolio_state.get_total_value()
        observation, _, _, _, _ = single.step(np.array([action], dtype=np.float32), asset='SPY')
        expected = single.portfolio_state.get_total_value() / before - 1
        assert single.episode_buffer.saa_returns[single.current_step] == pytest.approx(expected, abs=1e-8)
        assert observation[single_cache.num_features + 2] == pytest.approx(expected, abs=1e-8)
        if action != 0:
            assert expected < 0
    cache.risk_free_rate_daily[:] = 0.001
    before = single.portfolio_state.get_total_value()
    single.step(np.zeros(1, dtype=np.float32), asset='SPY')
    assert single.episode_buffer.saa_returns[single.current_step] > 0
    assert single.episode_buffer.saa_returns[single.current_step] == pytest.approx(
        single.portfolio_state.get_total_value() / before - 1, abs=1e-8
    )


def test_hold_price_gain_does_not_create_execution_gap(configs, cache, monkeypatch):
    _, saa = configs
    saa['environment'].update(execution_min_days_between_trades=0, commission_bps=0, half_spread_bps=0, impact_coeff=0)
    cache.close_prices[:] = 100
    cache.close_prices[2:] = 120
    cache.risk_free_rate_daily[:] = 0
    single = TradingEnv(saa, _standalone_cache(cache, saa), 'validation')
    original = single.calculate_saa_step_reward
    parts = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        parts.append(result[1])
        return result

    monkeypatch.setattr(single, 'calculate_saa_step_reward', capture)
    single.reset(seed=42, option=_options(), asset='SPY')
    single.step(np.array([0.3], dtype=np.float32), asset='SPY')
    single.step(np.zeros(1, dtype=np.float32), asset='SPY')
    assert parts[-1]['execution_gap_penalty'] == 0


def test_saa_drawdown_penalty_is_immediate_and_episode_metric_is_correct(configs, cache, monkeypatch):
    _, saa = configs
    saa['environment'].update(execution_min_days_between_trades=0, commission_bps=0, half_spread_bps=0, impact_coeff=0)
    cache.close_prices[:] = 100
    cache.close_prices[1] = 90
    cache.close_prices[2:] = 70
    cache.risk_free_rate_daily[:] = 0
    single = TradingEnv(saa, _standalone_cache(cache, saa), 'validation')
    original = single.calculate_saa_step_reward
    parts = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        parts.append(result[1])
        return result

    monkeypatch.setattr(single, 'calculate_saa_step_reward', capture)
    single.reset(seed=42, option=_options(), asset='SPY')
    single.step(np.ones(1, dtype=np.float32), asset='SPY')
    assert parts[-1]['max_drawdown'] == pytest.approx(0.1)
    assert single.episode_buffer.drawdown[1] == pytest.approx(0.1)
    for _ in range(6):
        result = single.step(np.zeros(1, dtype=np.float32), asset='SPY')
    assert result[3]
    assert result[4]['episode_max_drawdown'] == pytest.approx(0.3)


def test_allocator_sortino_has_half_percent_std_floor(configs, cache):
    paa, _ = configs
    parent = TradingEnv(paa, cache, 'validation')
    parent.reset(seed=42, option=_options())
    parent.step(np.zeros(cache.num_assets, dtype=np.float32))
    assert parent.episode_buffer.downside_var_sqrt[1] == pytest.approx(0.005)


def test_shadow_drawdown_includes_trade_costs_without_mutating_history(configs, cache):
    paa, saa = configs
    cache.close_prices[:] = 100
    cache.risk_free_rate_daily[:] = 0
    parent = TradingEnv(paa, cache, 'validation')
    parent.configure_saa_shadow_portfolios(saa)
    parent.reset(seed=42, option=_options())
    parent.apply_saa_sub_actions(np.full(cache.num_assets, 0.3, dtype=np.float32))
    np.testing.assert_array_equal(parent.episode_buffer.shadow_nav[0], paa['environment']['initial_portfolio_value'])
    parent.step(np.zeros(cache.num_assets, dtype=np.float32))
    expected = 1 - parent.episode_buffer.shadow_nav[1] / parent.episode_buffer.shadow_nav[0]
    np.testing.assert_allclose(parent.episode_buffer.shadow_drawdown[1], expected, atol=1e-7)
    assert np.all(expected > 0)


@pytest.mark.parametrize('family', ['hierarchical', 'ar1', 'zero'])
def test_terminal_wrappers_use_final_features_and_holdings(configs, cache, monkeypatch, family):
    from agents.PAA_with_autoregressive_rnd_walk_SAA.PAA_with_autoregressive_rnd_walk_SAA import AR1SignalWrapper
    from agents.PAA_cross_sectional_only.PAA_cross_sectional_only import ZeroSignalWrapper
    from utils.paa_val_test_inference import DailyRecorder

    paa, saa = configs
    monkeypatch.setattr(allocator, '_read_saa_training_config', lambda _config: copy.deepcopy(saa))
    raw = DummyVecEnv([lambda: DailyRecorder(_make_paa_adapter(paa, cache, _options(length=3)))])
    recorder = raw.envs[0]
    predictions = []
    if family == 'hierarchical':
        model = FixedPolicy(32 + 6 + cache.num_assets, 0.5)
        original_predict = model.predict

        def predict(observations, **kwargs):
            predictions.append((observations.copy(), kwargs['episode_start'].copy()))
            return original_predict(observations, **kwargs)

        monkeypatch.setattr(model, 'predict', predict)
        wrapper = allocator.SAASignalWrapper(
            raw, model, None, cache.num_assets, torch.device('cpu'), paa, cache.feature_to_index, 0.3, False
        )
    elif family == 'ar1':
        wrapper = AR1SignalWrapper(raw, cache.num_assets, cache.feature_to_index, 0.9, 0.15, 1., 0.3, 42, saa)
    else:
        wrapper = ZeroSignalWrapper(raw, cache.num_assets, cache.feature_to_index)
    try:
        recorder.arm()
        wrapper.reset()
        for _ in range(2):
            _, _, done, infos = wrapper.step(np.zeros((1, cache.num_assets), dtype=np.float32))
        assert done[0] and infos[0]['TimeLimit.truncated']
        terminal = infos[0]['terminal_observation']
        block_size = cache.num_assets * (cache.num_features + 2)
        asset_features = terminal[:block_size].reshape(cache.num_assets, cache.num_features + 2)
        np.testing.assert_array_equal(asset_features[:, :cache.num_features], cache.features[2])
        np.testing.assert_array_equal(asset_features[:, -1], infos[0]['terminal_shadow_weights'])
        assert terminal[block_size:block_size + cache.num_assets + 1].sum() == pytest.approx(1)
        assert not recorder.completed['sub_committed'][-1]
        if family == 'hierarchical':
            np.testing.assert_array_equal(predictions[-2][0], infos[0]['terminal_saa_observations'])
            assert not predictions[-2][1].any()
            assert predictions[-1][1].all()
    finally:
        wrapper.close()


def test_recurrent_ppo_bootstraps_genuine_terminal_observations(configs, cache, monkeypatch):
    from agents.RecurrPPO_target_position_agent.recurr_ppo_target_pos_agent import SingleAssetEpisodeAdapter

    _, saa = configs
    adapter = SingleAssetEpisodeAdapter(
        TradingEnv(saa, _standalone_cache(cache, saa), 'validation'), action_factor_fn=lambda **kwargs: 0.3
    )
    original_reset = adapter.reset

    def reset(**kwargs):
        adapter.set_forced_episode_once('SPY', _options(length=3))
        return original_reset(**kwargs)

    monkeypatch.setattr(adapter, 'reset', reset)
    vector = DummyVecEnv([lambda: adapter])
    torch.set_num_threads(1)
    model = RecurrentPPO(
        'MlpLstmPolicy', vector, n_steps=4, batch_size=4, n_epochs=1, seed=42, device='cpu',
        policy_kwargs={'lstm_hidden_size': 8, 'net_arch': {'pi': [8], 'vf': [8]}},
    )
    original_values = model.policy.predict_values
    observations = []

    def predict_values(observation, *args, **kwargs):
        observations.append(observation.detach().cpu().numpy().copy())
        return original_values(observation, *args, **kwargs)

    monkeypatch.setattr(model.policy, 'predict_values', predict_values)
    try:
        model.learn(total_timesteps=4)
        assert len(observations) >= 3
        for observation in observations:
            np.testing.assert_array_equal(observation[:, -cache.num_assets:].sum(axis=1), [1.])
    finally:
        vector.close()


def test_ar1_books_match_standalone_execution_without_loading_policy(configs, cache, monkeypatch):
    from agents.PAA_with_autoregressive_rnd_walk_SAA.PAA_with_autoregressive_rnd_walk_SAA import AR1SignalWrapper
    from utils.paa_val_test_inference import DailyRecorder

    paa, saa = configs
    raw = DummyVecEnv([lambda: DailyRecorder(_make_paa_adapter(paa, cache, _options()))])
    recorder = raw.envs[0]
    wrapper = AR1SignalWrapper(raw, cache.num_assets, cache.feature_to_index, 1., 0., 1., 0.3, 42, saa)
    monkeypatch.setattr(wrapper, '_draw_stationary', lambda shape: np.ones(shape, dtype=np.float32))
    singles = [TradingEnv(copy.deepcopy(saa), _standalone_cache(cache, saa), 'validation') for _ in cache.asset_names]
    for single, asset in zip(singles, cache.asset_names):
        single.reset(seed=42, option=_options(), asset=asset)
    try:
        recorder.arm()
        wrapper.reset()
        for _ in range(7):
            for single, asset in zip(singles, cache.asset_names):
                single.step(np.array([0.3], dtype=np.float32), asset=asset)
            wrapper.step(np.zeros((1, cache.num_assets), dtype=np.float32))
        scale = paa['environment']['initial_portfolio_value'] / saa['environment']['initial_portfolio_value']
        expected_nav = np.column_stack([single.episode_buffer.portfolio_values[:8] for single in singles]) * scale
        expected_cost = np.column_stack([single.episode_buffer.transaction_costs[1:8] for single in singles]) * scale
        np.testing.assert_allclose(recorder.completed['sub_value_mtm'], expected_nav, rtol=1e-6, atol=0.02)
        np.testing.assert_allclose(recorder.completed['sub_tc'][:-1], expected_cost, rtol=1e-6, atol=1e-6)
        assert np.all(np.isfinite(recorder.completed['sub_weight_post'][:-1]))
    finally:
        wrapper.close()


@pytest.mark.parametrize('fixed_fee', [0., 10.])
def test_generic_shadow_buys_reserve_transaction_costs(configs, cache, fixed_fee):
    paa, _ = configs
    cache.close_prices[:] = 100
    parent = TradingEnv(paa, cache, 'validation')
    parent.reset(seed=42, option=_options())
    parent.config['environment']['fixed_fee_per_order'] = fixed_fee
    for _ in range(4):
        parent.apply_saa_sub_actions(np.ones(cache.num_assets, dtype=np.float32))
        cash, _, _, _ = parent.episode_buffer.get_saa_sub_state(parent.current_step)
        assert np.all(cash >= 0)
        parent.step(np.zeros(cache.num_assets, dtype=np.float32))


def test_test_block_is_always_final_252_dates(cache):
    dates = np.array([str(date) for date in np.arange('2020-01-01', '2021-05-15', dtype='datetime64[D]')])
    expanded = replace(cache, dates=dates, num_days=len(dates))
    expanded._create_time_series_blocks(1., 0.2)
    assert expanded.test_blocks[0].start_date_idx == len(dates) - 252
    assert expanded.test_blocks[0].num_days == 252


@pytest.mark.parametrize('reload_stats', [False, True])
def test_saa_validation_syncs_normalization_every_sweep(tmp_path, reload_stats):
    from agents.RecurrPPO_target_position_agent.recurr_ppo_target_pos_agent import EvalCallbackWithMetrics

    space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    factory = lambda: allocator._ObsNormDummyEnv(space)
    train = VecNormalize(DummyVecEnv([factory]))
    if reload_stats:
        path = tmp_path / 'normalization.pkl'
        train.save(str(path))
        evaluation = VecNormalize.load(str(path), DummyVecEnv([factory]))
        evaluation.training = False
        evaluation.norm_reward = False
    else:
        evaluation = VecNormalize(DummyVecEnv([factory]), training=False, norm_reward=False)
    callback = EvalCallbackWithMetrics(evaluation, None, None, 1, 1, True, False)
    callback._run_deterministic_validation_grid = lambda step_callback: ([0.], [7], 1)
    model = SimpleNamespace(
        logger=SimpleNamespace(record=lambda *args, **kwargs: None), get_env=lambda: train, num_timesteps=1
    )
    callback.init_callback(model)
    callback.n_calls = 1
    try:
        for mean in [4., 8.]:
            train.obs_rms.mean[:] = mean
            train.obs_rms.var[:] = mean / 2
            callback._on_step()
            np.testing.assert_array_equal(evaluation.obs_rms.mean, train.obs_rms.mean)
            np.testing.assert_array_equal(evaluation.obs_rms.var, train.obs_rms.var)
            assert evaluation.obs_rms is not train.obs_rms
            assert not evaluation.training and not evaluation.norm_reward
    finally:
        train.close()
        evaluation.close()


@pytest.mark.parametrize('family', ['hierarchical', 'ar1', 'zero'])
def test_allocator_ppo_bootstraps_real_terminal_portfolios(configs, cache, monkeypatch, family):
    from agents.PAA_with_autoregressive_rnd_walk_SAA.PAA_with_autoregressive_rnd_walk_SAA import AR1SignalWrapper
    from agents.PAA_cross_sectional_only.PAA_cross_sectional_only import ZeroSignalWrapper

    paa, saa = configs
    monkeypatch.setattr(allocator, '_read_saa_training_config', lambda _config: copy.deepcopy(saa))
    raw = DummyVecEnv([partial(_make_paa_adapter, paa, cache, _options(length=3))])
    if family == 'hierarchical':
        wrapper = allocator.SAASignalWrapper(
            raw, FixedPolicy(32 + 6 + cache.num_assets, 0.5), None, cache.num_assets,
            torch.device('cpu'), paa, cache.feature_to_index, 0.3, False,
        )
    elif family == 'ar1':
        wrapper = AR1SignalWrapper(raw, cache.num_assets, cache.feature_to_index, 0.9, 0.15, 1., 0.3, 42, saa)
    else:
        wrapper = ZeroSignalWrapper(raw, cache.num_assets, cache.feature_to_index)
    policy_kwargs = dict(
        n_assets=cache.num_assets, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32,
        transformer_encoder_dropout=0., transformer_activation_fn='gelu', log_std_init=-1.,
        features_extractor_class=allocator.SAATokenizer,
        features_extractor_kwargs=dict(
            num_assets=cache.num_assets, raw_feat_dim=cache.num_features, d_model=16,
            asset_feature_idx=[cache.feature_to_index[name] for name, enabled in paa['paa_asset_token_features'].items() if enabled],
            portfolio_time_idx=[cache.feature_to_index[name] for name, enabled in paa['paa_portfolio_token_features'].items() if enabled],
        ),
    )
    torch.set_num_threads(1)
    model = PPO(
        allocator.TransformerAllocatorPolicy, wrapper, policy_kwargs=policy_kwargs,
        n_steps=4, batch_size=4, n_epochs=1, seed=42, device='cpu',
    )
    original_values = model.policy.predict_values
    observations = []

    def predict_values(observation, *args, **kwargs):
        observations.append(observation.detach().cpu().numpy().copy())
        return original_values(observation, *args, **kwargs)

    monkeypatch.setattr(model.policy, 'predict_values', predict_values)
    try:
        model.learn(total_timesteps=4)
        assert len(observations) >= 3
        portfolio_offset = cache.num_assets * (cache.num_features + 2)
        for observation in observations:
            np.testing.assert_allclose(
                observation[:, portfolio_offset:portfolio_offset + cache.num_assets + 1].sum(axis=1), [1.], atol=1e-6
            )
        assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    finally:
        wrapper.close()


def test_ar1_inference_constructor_uses_execution_configuration(configs, cache, tmp_path, monkeypatch):
    from utils import paa_val_test_inference as inference

    paa, saa = configs
    paa['shadow_execution_config_path'] = 'synthetic.json'
    monkeypatch.setattr(inference, '_load_json', lambda path: copy.deepcopy(saa))
    dimension = cache.num_assets * (cache.num_features + 2) + cache.num_assets + 1 + 14 + 3 * cache.num_assets
    space = gym.spaces.Box(-np.inf, np.inf, shape=(dimension,), dtype=np.float32)
    normalization = VecNormalize(DummyVecEnv([lambda: allocator._ObsNormDummyEnv(space)]))
    path = tmp_path / 'normalization.pkl'
    normalization.save(str(path))
    normalization.close()
    spec = SimpleNamespace(family=inference.FAMILY_AR1, vecnorm_path=path)
    vector, recorder, info = inference.build_agent_venv(spec, paa, cache, 'validation')
    try:
        assert len(recorder.te._saa_shadow_envs) == cache.num_assets
        assert recorder.te._saa_shadow_envs[0].initial_portfolio_value == saa['environment']['initial_portfolio_value']
        assert info['injected_signal'] == 'AR(1) noise'
        assert not vector.training and not vector.norm_reward
    finally:
        vector.close()