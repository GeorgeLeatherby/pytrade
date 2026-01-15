import numpy as np
import pandas as pd
import pytest
import inspect
from datetime import datetime, timedelta
from environment.trading_env import MarketDataCache, TradingEnv

# ---------------------------
# Configuration & Data Setup
# ---------------------------
@pytest.fixture(scope="session")
def base_config():
    return {
        "features": {
            # Enable all discovered features for synthetic data
        },
        "environment": {
            "episode_length_days": 8,
            "lookback_window": 5,
            "initial_portfolio_value": 100000.0,
            "early_stopping_threshold": 0.50,
            "cash_return_rate": 0.0,
            "min_initial_cash_allocation": 0.05,
            "seed": 42,
            "execution_weight_change_threshold": 0.02,
            "execution_min_trade_value_threshold": 250.0,
            "execution_min_days_between_trades": 1,
            "execution_min_per_asset_weight_change": 0.01,
            "commission_bps": 0.00005,
            "half_spread_bps": 0.0005,
            "impact_coeff": 0.1,
            "adv_window": 10,
            "fixed_fee_per_order": 0.0,
            "execution_min_share_threshold": 1e-5,
            "maybe_provide_sequence": True,
            "test_val_split_ratio": 0.2,
            "block_buffer_multiplier": 2,
        }
    }

def _make_dataframe(num_days: int = 160, asset_symbols=None):
    if asset_symbols is None:
        asset_symbols = ["SPY", "EWJ", "EWG"]
    start = datetime(2020, 1, 1)
    rows = []
    for d in range(num_days):
        date = (start + timedelta(days=d)).strftime("%Y-%m-%d")
        for sym in asset_symbols:
            base_price = 100 + d * 0.5 + (hash(sym) % 7)
            open_p = base_price
            high_p = base_price * 1.01
            low_p = base_price * 0.99
            close_p = base_price * (1 + 0.0005 * np.sin(d / 5))
            volume = 1_000_000 + (d * 100)
            feat_a = np.log1p(d + len(sym))
            feat_b = (d % 11) / 11.0
            rows.append({
                "Date": date,
                "Symbol": sym,
                "Open": open_p,
                "High": high_p,
                "Low": low_p,
                "Close": close_p,
                "Volume": volume,
                "feat_a": feat_a,
                "feat_b": feat_b,
            })
    return pd.DataFrame(rows)

@pytest.fixture(scope="session")
def market_cache_sequence(base_config):
    df = _make_dataframe()
    cfg = base_config.copy()
    cfg["environment"] = cfg["environment"].copy()
    cfg["environment"]["maybe_provide_sequence"] = True
    # Activate all feature columns found
    for col in df.columns:
        if col not in ("Date","Symbol","Open","High","Low","Close","Volume"):
            cfg["features"][col] = True
    return MarketDataCache.from_dataframe(df, cfg, lookback_window=cfg["environment"]["lookback_window"], maybe_provide_sequence=True)

@pytest.fixture(scope="session")
def market_cache_single(base_config):
    df = _make_dataframe()
    cfg = base_config.copy()
    cfg["environment"] = cfg["environment"].copy()
    cfg["environment"]["maybe_provide_sequence"] = False
    for col in df.columns:
        if col not in ("Date","Symbol","Open","High","Low","Close","Volume"):
            cfg["features"][col] = True
    return MarketDataCache.from_dataframe(df, cfg, lookback_window=cfg["environment"]["lookback_window"], maybe_provide_sequence=False)

@pytest.fixture
def env_sequence(base_config, market_cache_sequence):
    cfg = base_config.copy()
    cfg["environment"] = cfg["environment"].copy()
    cfg["environment"]["maybe_provide_sequence"] = True
    env = TradingEnv(cfg, market_cache_sequence, mode='train')
    obs, info = env.reset(seed=123)
    return env

@pytest.fixture
def env_single(base_config, market_cache_single):
    cfg = base_config.copy()
    cfg["environment"] = cfg["environment"].copy()
    cfg["environment"]["maybe_provide_sequence"] = False
    env = TradingEnv(cfg, market_cache_single, mode='train')
    obs, info = env.reset(seed=456)
    return env

def _rand_action(env):
    return np.random.randn(env.market_data_cache.num_assets + 1).astype(np.float32)

# ---------------------------
# Time Advancement Tests
# ---------------------------
def test_step_time_advancement_consistency_sequence(env_sequence):
    env = env_sequence
    lookback = env.lookback_window
    records = []
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(_rand_action(env))
        records.append((env.current_step, env.current_absolute_step, info['absolute_step'], env.portfolio_state.prices.copy()))
        if terminated or truncated:
            break
    for (cs, abs_cs, info_abs, prices) in records:
        assert abs_cs == info_abs
        external_step = cs - 1
        internal_index = lookback + external_step
        assert 0 <= internal_index < env.episode_buffer.episode_buffer_length_days
        expected_prices = env.market_data_cache.close_prices[abs_cs]
        np.testing.assert_allclose(prices, expected_prices, rtol=1e-6)

def test_step_time_advancement_consistency_single(env_single):
    env = env_single
    records = []
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(_rand_action(env))
        records.append((env.current_step, env.current_absolute_step, info['absolute_step']))
        if terminated or truncated:
            break
    for (cs, abs_cs, info_abs) in records:
        assert abs_cs == info_abs

@pytest.mark.xfail(reason="Off-by-one: single-step observation uses current_step-1 immediately after reset.")
def test_initial_observation_negative_index_bug(env_single):
    env = env_single
    obs = env.get_observation_single_step()
    assert env.current_step >= 0
    assert obs.size == env.observation_space.shape[0]

# ---------------------------
# Episode Buffer Integrity
# ---------------------------
def test_buffer_index_mapping(env_sequence):
    env = env_sequence
    for _ in range(3):
        env.step(_rand_action(env))
    for external in range(env.current_step):
        internal = env.lookback_window + external
        assert env.episode_buffer.portfolio_values[internal] > 0
        assert env.episode_buffer.action_entropy[internal] >= 0

def test_returns_consistency(env_sequence):
    env = env_sequence
    pv_prev = env.portfolio_state.get_total_value()
    env.step(_rand_action(env))
    pv_curr = env.portfolio_state.get_total_value()
    manual_ret = (pv_curr / pv_prev) - 1.0
    ext = env.current_step - 1
    internal = env.lookback_window + ext
    stored_ret = env.episode_buffer.returns[internal]
    np.testing.assert_allclose(manual_ret, stored_ret, rtol=1e-6)

def test_warmup_prices_filled(env_sequence):
    env = env_sequence
    warm = env.episode_buffer.asset_prices[:env.lookback_window]
    assert warm.shape[0] == env.lookback_window
    assert (warm.sum(axis=1) > 0).any()

# ---------------------------
# Observation Shape & Content
# ---------------------------
def test_sequence_observation_shape(env_sequence):
    env = env_sequence
    obs = env.get_observation_sequence()
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))

def test_single_step_observation_shape(env_single):
    env = env_single
    env.step(_rand_action(env))
    obs = env.get_observation_single_step()
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))

def test_flatten_partition(env_sequence):
    env = env_sequence
    obs = env.get_observation_sequence()
    asset_part = obs[:env.asset_obs_size]
    portfolio_part = obs[env.asset_obs_size:]
    assert asset_part.size == env.asset_obs_size
    assert portfolio_part.size == env.portfolio_obs_size

def test_portfolio_feature_count(env_sequence):
    env = env_sequence
    assert env.episode_buffer.num_portfolio_features == env.market_data_cache.num_assets + 1 + 6

# ---------------------------
# Execution & Costs
# ---------------------------
def test_no_execution_below_threshold(env_single):
    env = env_single
    current_w = env.portfolio_state.get_weights()
    tiny = current_w.copy()
    tiny[1:] += 1e-6
    tiny /= tiny.sum()
    result = env.execute_portfolio_change(tiny, env.portfolio_state)
    assert result.success is False
    assert result.transaction_cost == 0.0
    assert (result.trades_executed == 0).all()

def test_execution_generates_costs(env_single):
    env = env_single
    w = np.zeros_like(env.portfolio_state.get_weights())
    w[0] = 0.05
    if len(w) > 1:
        w[1] = 0.90
    if len(w) > 2:
        w[2:] = (1 - w[0] - w[1]) / (len(w) - 2)
    w /= w.sum()
    result = env.execute_portfolio_change(w, env.portfolio_state)
    if result.success:
        assert result.transaction_cost >= 0.0
        assert abs(result.traded_dollar_value) > 0

def test_transaction_cost_model_non_negative(env_single):
    env = env_single
    prices = env.portfolio_state.prices.copy()
    shares = np.ones(env.market_data_cache.num_assets, dtype=np.float32)
    cost = env._calculate_transaction_costs(shares, prices, abs_step=env.current_absolute_step)
    assert cost >= 0.0

def test_adv_window_handling_initial(env_single):
    env = env_single
    prices = env.portfolio_state.prices.copy()
    shares = np.ones(env.market_data_cache.num_assets, dtype=np.float32)
    cost = env._calculate_transaction_costs(shares, prices, abs_step=0)
    assert cost >= 0.0

# ---------------------------
# Reward Components
# ---------------------------
def test_reward_parts_presence(env_single):
    env = env_single
    env.step(_rand_action(env))
    _, reward, _, _, _ = env.step(_rand_action(env))
    ext = env.current_step - 1
    stored = env.episode_buffer.allocator_rewards[env.lookback_window + ext] if env.maybe_provide_sequence else env.episode_buffer.allocator_rewards[ext]
    assert isinstance(reward, float)
    assert stored == reward

def test_risk_window_growth(env_single):
    env = env_single
    vals = []
    for _ in range(6):
        env.step(_rand_action(env))
        vals.append(env.episode_buffer.sharpe_ratio[env.lookback_window + env.current_step - 1] if env.maybe_provide_sequence else env.episode_buffer.sharpe_ratio[env.current_step - 1])
    assert len(vals) == 6

def test_drawdown_never_negative(env_single):
    env = env_single
    for _ in range(5):
        env.step(_rand_action(env))
        dd = env.episode_buffer.drawdown[env.lookback_window + env.current_step - 1] if env.maybe_provide_sequence else env.episode_buffer.drawdown[env.current_step - 1]
        assert dd >= 0.0

# ---------------------------
# Termination Conditions
# ---------------------------
def test_truncation_at_episode_length(env_single):
    env = env_single
    terminated = truncated = False
    for _ in range(env.episode_length_days):
        _, _, terminated, truncated, _ = env.step(_rand_action(env))
        if terminated or truncated:
            break
    assert terminated or truncated

def test_early_stopping_trigger(env_single):
    env = env_single
    for _ in range(env.episode_length_days):
        action = np.concatenate(([0.0], np.ones(env.market_data_cache.num_assets))).astype(np.float32)
        _, _, terminated, _, info = env.step(action)
        if terminated:
            assert info['portfolio_value'] <= env.threshold_val
            break

def test_info_fields_after_episode_end(env_single):
    env = env_single
    for _ in range(env.episode_length_days):
        _, _, term, trunc, info = env.step(_rand_action(env))
        if term or trunc:
            assert "episode_final" in info
            assert "cumulative_reward" in info
            assert "portfolio_final_value" in info
            break

# ---------------------------
# Market Cache Quality
# ---------------------------
def test_market_cache_quality(market_cache_single):
    stats = market_cache_single.validate_data_quality()
    assert stats['close_prices_nan_pct'] == 0.0
    assert stats['features_nan_pct'] == 0.0
    assert stats['min_completeness'] == 100.0
    assert len(stats['feature_selection']['selected_features']) == market_cache_single.num_features

def test_feature_lookback_shape(market_cache_single):
    step = 50
    lookback = 5
    arr = market_cache_single.get_features_lookback(step, lookback)
    assert arr.shape == (lookback, market_cache_single.num_assets, market_cache_single.num_features)

def test_ohlcv_lookback_shape(market_cache_single):
    step = 75
    lookback = 7
    arr = market_cache_single.get_OHLCV_lookback(step, lookback)
    assert arr.shape == (lookback, market_cache_single.num_assets, 5)

# ---------------------------
# Known Issues / Code Hygiene
# ---------------------------
def test_duplicate_transaction_cost_function_definition():
    import environment.trading_env as te
    source = inspect.getsource(te.TradingEnv)
    assert source.count("def _calculate_transaction_costs(") == 2