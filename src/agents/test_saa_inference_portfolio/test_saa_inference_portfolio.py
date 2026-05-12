"""
--- SAA Portfolio Inference Test Agent ---

PURPOSE (see preamble below — original requirements, preserved verbatim):

Tests whether trained SAA (Single-Asset Agent) models produce meaningful signals
when used directly in a portfolio context with uniform cash allocation.

All agents share the same cash pool. If one SAA proposes a large buy, it reduces
available cash for other assets. When multiple buy requests exceed available cash,
the cash is distributed proportionally to the size of the buy signals.

Workflow:
    1. Load a single trained SAA model + matching VecNormalize stats; instantiate
       one deep-copied model per asset so each maintains its OWN LSTM hidden
       state while sharing weights (mirrors PAA SAASignalWrapper design).
    2. Iterate validation episodes sampled from MarketDataCache.validation_blocks.
    3. Per step:
         a. Build per-asset SAA observations identical to SAA training-time
            format (num_saa_features + 4 portfolio features:
              [cash_log_value, asset_log_value, daily_agent_return, last_action])
         b. Normalize with the saved VecNormalize obs_rms.
         c. Run each SAA model.predict() to get a scalar target-position change.
         d. Scale by action_limiting_factor (matches deployment scaling).
         e. Convert to TradeInstruction per asset. Sells freed cash first.
            Pro-rata scale buys when sum(buys_notional) > available_cash.
         f. env.step(instructions) → env handles MTM, costs, benchmark, comp.
    4. Collect per-step / per-episode / per-asset metrics. Produce academic-style
       matplotlib reports for the five comparisons listed in the preamble.

DESIGN NOTES:
    - environment.* and saa_features.* in the test config MUST mirror the SAA
      training config of the loaded model — cache feature ordering and obs dim
      depend on them. The test runner overrides ``execution_mode`` → "simple"
      so env.step() accepts a list[TradeInstruction] across all assets.
    - Per-asset "sub-portfolio" bookkeeping is virtual and serves only to build
      SAA observations consistent with training-time distribution. Real shares
      live in env.portfolio_state.positions; real cash is a shared pool that we
      re-divide equally each step (initial_pv / N) to reflect "uniform cash
      allocation".
    - Random-allocation mode and minimum-cash-allocation are listed as TODOs in
      the preamble; explicit NotImplementedError raised when requested.

End of preamble.
"""

# ================================
# Imports
# ================================
import os
import copy
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # non-interactive; safe for CLI/server
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.environment.single_asset_target_pos_drl_trading_env import (
    TradingEnv,
    MarketDataCache,
    TradeInstruction,
)


# ================================
# Academic plotting defaults
# ================================
def _apply_paper_style() -> None:
    """Set Matplotlib rcParams once for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


# ================================
# VecNormalize utilities (mirror PAA agent helpers)
# ================================
class _ObsNormDummyEnv(gym.Env):
    """Minimal env so VecNormalize.load() can attach to a space without running."""
    metadata: Dict[str, Any] = {}

    def __init__(self, observation_space: gym.Space):
        self.observation_space = observation_space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):  # never called
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}


def _normalize_obs(obs: np.ndarray, vecnorm: Optional[VecNormalize]) -> np.ndarray:
    """Apply saved VecNormalize obs_rms to a 1D/2D obs array. Returns float32."""
    if vecnorm is None or getattr(vecnorm, "obs_rms", None) is None:
        return np.asarray(obs, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    mean = vecnorm.obs_rms.mean
    var = vecnorm.obs_rms.var
    epsilon = float(getattr(vecnorm, "epsilon", getattr(vecnorm, "eps", 1e-8)))
    clip_obs = float(getattr(vecnorm, "clip_obs", 10.0))
    return np.clip((obs - mean) / np.sqrt(var + epsilon), -clip_obs, clip_obs).astype(np.float32)


# ================================
# SAA model loader
# ================================
def _load_saa_assets(
    model_run_dir: str,
    device: torch.device,
) -> Tuple[RecurrentPPO, Optional[VecNormalize]]:
    """Load best_model.zip + best_model_vecnormalize.pkl from a saved run directory.

    Shapes/contract:
        model_run_dir/best_model.zip          → RecurrentPPO checkpoint
        model_run_dir/best_model_vecnormalize.pkl → VecNormalize stats
    """
    model_path = os.path.join(model_run_dir, "best_model.zip")
    vecnorm_path = os.path.join(model_run_dir, "best_model_vecnormalize.pkl")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"SAA best_model.zip not found at: {model_path}")
    model = RecurrentPPO.load(model_path, device=device)
    print(f"[SAA-Test] Loaded SAA model: {model_path}")

    vecnorm: Optional[VecNormalize] = None
    if os.path.isfile(vecnorm_path):
        dummy = DummyVecEnv([lambda: _ObsNormDummyEnv(model.observation_space)])
        vecnorm = VecNormalize.load(vecnorm_path, dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False
        print(f"[SAA-Test] Loaded VecNormalize: {vecnorm_path}")
    else:
        print(f"[SAA-Test] No VecNormalize found at {vecnorm_path}; using raw obs (may degrade quality).")
    return model, vecnorm


# ================================
# Per-asset SAA inference runner
# ================================
class SAAPortfolioInferenceRunner:
    """
    Drives validation-mode episodes through a TradingEnv (EXECUTION_SIMPLE) and
    queries N deep-copied SAA models for per-asset target position changes.

    Shapes (N = num_assets, F_saa = number of enabled saa_features):
        per-asset obs:  (F_saa + 4,)
        SAA action:     scalar in [-1, 1] after action_limiting_factor scaling
        sub-portfolio:  cash_i (uniform=total_cash/N), shares_i (real), last_action_i, daily_return_i

    Outputs:
        per-step records used downstream for reporting (no buffering inside env).
    """

    def __init__(
        self,
        cache: MarketDataCache,
        config: Dict[str, Any],
        model: RecurrentPPO,
        vecnorm: Optional[VecNormalize],
        device: torch.device,
        num_episodes: int,
        seed: int,
    ):
        self.cache = cache
        self.config = config
        self.device = device
        self.num_episodes = int(num_episodes)
        self.seed = int(seed)

        self.asset_names: List[str] = list(self.cache.asset_names)
        self.num_assets: int = len(self.asset_names)
        self.initial_pv: float = float(self.config["environment"]["initial_portfolio_value"])
        self.action_limiting_factor: float = float(self.config["test_agent"].get("action_limiting_factor", 1.0))
        self.deterministic: bool = bool(self.config["test_agent"].get("deterministic_saa", True))

        allocation_mode = self.config["test_agent"].get("allocation_mode", "uniform")
        if allocation_mode != "uniform":
            raise NotImplementedError(
                f"allocation_mode={allocation_mode!r} not implemented yet (preamble TODO: "
                "random allocation mode + minimum cash allocation)."
            )

        # Cache enabled SAA feature indices into cache.features last-axis.
        saa_flags = self.config.get("saa_features", {})
        all_feature_names = list(self.cache.selected_feature_names)
        feat_to_idx = {f: i for i, f in enumerate(all_feature_names)}
        enabled = [f for f, on in saa_flags.items() if on]
        missing = [f for f in enabled if f not in feat_to_idx]
        if missing:
            raise ValueError(f"saa_features enabled but not present in cache: {missing}")
        self.saa_feat_idx = np.array([feat_to_idx[f] for f in enabled], dtype=int)
        self.num_saa_features = int(self.saa_feat_idx.size)
        print(f"[SAA-Test] {self.num_saa_features} SAA features active.")

        # Detect whether the loaded SAA was trained with a one-hot asset-ID block.
        # Older training runs emit obs = [F_saa, 4 portfolio];
        # newer runs append a one-hot of length N: obs = [F_saa, 4 portfolio, N one_hot].
        expected_obs_dim = int(model.observation_space.shape[0])
        if expected_obs_dim == self.num_saa_features + 4:
            self.use_one_hot = False
        elif expected_obs_dim == self.num_saa_features + 4 + self.num_assets:
            self.use_one_hot = True
        else:
            raise ValueError(
                f"SAA model expects obs_dim={expected_obs_dim}, but config implies "
                f"F_saa({self.num_saa_features})+4 or +N({self.num_assets}). "
                "Make sure saa_features in this test config match the SAA training config."
            )
        print(f"[SAA-Test] Model obs_dim={expected_obs_dim} -> use_one_hot={self.use_one_hot}")

        # Build env in EXECUTION_SIMPLE mode (overrides whatever the SAA training used).
        env_cfg = copy.deepcopy(self.config)
        env_cfg["environment"]["execution_mode"] = "simple"
        env_cfg["environment"]["percentage_of_cash_only_starts"] = 1.0
        self._env_cfg = env_cfg

        # One model per asset (shared weights, independent LSTM state). Mirrors PAA SAASignalWrapper.
        self.models: List[RecurrentPPO] = []
        for _ in range(self.num_assets):
            m = copy.deepcopy(model)
            m.policy.to(self.device)
            m.device = self.device
            m.policy.eval()
            self.models.append(m)
        self.vecnorm = vecnorm

        # Per-asset RNN states (None = auto-init) and episode-start flags.
        self._reset_lstm_states()

        # Sortable result records.
        self.episode_records: List[Dict[str, Any]] = []

    # ---------- internal helpers ----------

    def _reset_lstm_states(self) -> None:
        self.lstm_states: List[Optional[Any]] = [None for _ in range(self.num_assets)]
        # Shape (1,) per asset because we have a single env at a time.
        self.episode_start_flags = np.ones((self.num_assets, 1), dtype=bool)

    def _build_saa_obs(
        self,
        env: TradingEnv,
        sub_cash: np.ndarray,      # (N,)
        sub_shares: np.ndarray,    # (N,)
        sub_last_action: np.ndarray,  # (N,)
        sub_daily_return: np.ndarray, # (N,)
    ) -> np.ndarray:
        """
        Build per-asset SAA observation matrix of shape (N, F_saa + 4).
        Layout matches the SAA training observation produced by
        TradingEnv.get_observation_single_step() for execution_mode=
        single_asset_target_position.
        """
        prices = env.portfolio_state.prices  # (N,)
        feats_all = self.cache.get_features_at_step(env.current_absolute_step)  # (N, F_total)
        saa_market = feats_all[:, self.saa_feat_idx]  # (N, F_saa)

        asset_notional = sub_shares * prices  # (N,)
        eps = 1e-12
        cash_log = np.where(
            sub_cash > 0,
            np.log(np.maximum(sub_cash, eps) / self.initial_pv),
            0.0,
        ).astype(np.float32)
        asset_log = np.where(
            asset_notional > 0,
            np.log(np.maximum(asset_notional, eps) / self.initial_pv),
            0.0,
        ).astype(np.float32)

        port_block = np.stack(
            [cash_log, asset_log, sub_daily_return.astype(np.float32), sub_last_action.astype(np.float32)],
            axis=-1,
        )  # (N, 4)
        market_plus_port = np.concatenate([saa_market.astype(np.float32), port_block], axis=-1)  # (N, F_saa+4)

        if not self.use_one_hot:
            return market_plus_port

        # Append per-asset one-hot of length N (each row has a 1.0 at its asset index).
        one_hot = np.eye(self.num_assets, dtype=np.float32)  # (N, N)
        return np.concatenate([market_plus_port, one_hot], axis=-1)  # (N, F_saa+4+N)

    def _saa_predict_actions(self, obs_per_asset: np.ndarray) -> np.ndarray:
        """
        Run N independent SAA predictions. obs_per_asset shape: (N, F_saa+4).
        Returns scaled actions of shape (N,) in [-1, 1]*action_limiting_factor.
        """
        actions = np.zeros(self.num_assets, dtype=np.float32)
        for a in range(self.num_assets):
            single_obs = obs_per_asset[a:a + 1, :]  # (1, F_saa+4)
            single_obs = _normalize_obs(single_obs, self.vecnorm)
            with torch.no_grad():
                act, state_out = self.models[a].predict(
                    single_obs,
                    state=self.lstm_states[a],
                    episode_start=self.episode_start_flags[a],
                    deterministic=self.deterministic,
                )
            self.lstm_states[a] = state_out
            actions[a] = float(np.clip(act[0, 0], -1.0, 1.0))
        # After first step of an episode the flag flips to False.
        self.episode_start_flags[:] = False
        # Scale by action_limiting_factor (matches SingleAssetEpisodeAdapter.step).
        return np.clip(actions * self.action_limiting_factor, -1.0, 1.0).astype(np.float32)

    def _compute_instructions(
        self,
        scaled_actions: np.ndarray,  # (N,) — target position frac of sub_pv_i
        sub_cash: np.ndarray,
        sub_shares: np.ndarray,
        prices: np.ndarray,
        total_cash_available: float,
    ) -> Tuple[List[TradeInstruction], np.ndarray, np.ndarray]:
        """
        Convert SAA per-asset target position fractions to a list of TradeInstructions.
        Implements shared-cash conflict resolution: pro-rata scale buys when
        sum(buys_notional) > total_cash_available after sells.

        Returns:
            instructions: List[TradeInstruction]
            requested_delta_shares: (N,) — pre-scaling desired share delta (for diagnostics)
            executed_delta_notional: (N,) — post-scaling desired notional (used for analysis)
        """
        # Per-asset target & delta in notional terms.
        sub_pv = sub_cash + sub_shares * prices                # (N,)
        target_notional = scaled_actions * sub_pv              # (N,)  ∈ [-sub_pv, sub_pv]
        current_notional = sub_shares * prices                 # (N,)
        delta_notional = target_notional - current_notional    # (N,)

        # Bound: SELL cannot exceed currently held notional (no short).
        sell_cap = -current_notional                            # most negative allowed
        delta_notional = np.maximum(delta_notional, sell_cap)

        # Diagnostic: requested share delta pre-scaling.
        with np.errstate(divide="ignore", invalid="ignore"):
            requested_delta_shares = np.where(prices > 0, delta_notional / prices, 0.0).astype(np.float32)

        # Estimate cash after sells (approx; transaction costs handled by env).
        sells_notional = np.where(delta_notional < 0, -delta_notional, 0.0)
        buys_notional = np.where(delta_notional > 0, delta_notional, 0.0)
        cash_after_sells = total_cash_available + float(sells_notional.sum())

        # Pro-rata buy scaling on oversubscription.
        total_buys = float(buys_notional.sum())
        scale = 1.0
        if total_buys > cash_after_sells and total_buys > 1e-8:
            scale = max(0.0, cash_after_sells / total_buys)
        scaled_buys = buys_notional * scale
        executed_delta_notional = (scaled_buys - sells_notional).astype(np.float32)

        instructions: List[TradeInstruction] = []
        # Sells first so env.execute_instructions sees freed cash before buys.
        for i in range(self.num_assets):
            if sells_notional[i] > 1e-6 and prices[i] > 0:
                instructions.append(TradeInstruction(
                    symbol=self.asset_names[i],
                    action="SELL",
                    notional=float(sells_notional[i]),
                ))
        for i in range(self.num_assets):
            if scaled_buys[i] > 1e-6 and prices[i] > 0:
                instructions.append(TradeInstruction(
                    symbol=self.asset_names[i],
                    action="BUY",
                    notional=float(scaled_buys[i]),
                ))
        return instructions, requested_delta_shares, executed_delta_notional

    # ---------- main loop ----------

    def run_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Run a single validation episode and return its per-step records."""
        env = TradingEnv(self._env_cfg, self.cache, mode="validation")
        env.reset(seed=self.seed + episode_idx)

        self._reset_lstm_states()

        N = self.num_assets
        initial_pv = self.initial_pv

        # Virtual per-asset sub-portfolio state (cash split uniformly each step).
        sub_cash = np.full(N, initial_pv / N, dtype=np.float32)
        sub_shares = env.portfolio_state.positions.copy().astype(np.float32)  # zeros at cash-only start
        sub_last_action = np.zeros(N, dtype=np.float32)
        sub_daily_return = np.zeros(N, dtype=np.float32)
        prev_sub_pv = sub_cash + sub_shares * env.portfolio_state.prices  # (N,)

        # Pre-allocate per-step buffers (max length = episode_length_days).
        T_max = int(self.config["environment"]["episode_length_days"]) + 2
        rec_portfolio_value = np.zeros(T_max, dtype=np.float64)
        rec_benchmark_value = np.zeros(T_max, dtype=np.float64)
        rec_comparison_value = np.zeros(T_max, dtype=np.float64)
        rec_cash = np.zeros(T_max, dtype=np.float64)
        rec_positions = np.zeros((T_max, N), dtype=np.float32)
        rec_prices = np.zeros((T_max, N), dtype=np.float32)
        rec_actions = np.zeros((T_max, N), dtype=np.float32)
        rec_target_notional = np.zeros((T_max, N), dtype=np.float32)
        rec_executed_delta = np.zeros((T_max, N), dtype=np.float32)
        rec_costs = np.zeros(T_max, dtype=np.float64)
        rec_dates: List[str] = []
        rec_returns = np.zeros(T_max, dtype=np.float64)

        # Per-asset SAA sub-portfolio equity curves (academic compare to buy-and-hold).
        rec_sub_pv = np.zeros((T_max, N), dtype=np.float64)

        # Track buy-and-hold baselines from the same start.
        bh_initial_prices = env.portfolio_state.prices.copy()  # (N,)
        bh_shares_per_asset = np.where(
            bh_initial_prices > 0,
            (initial_pv / N) / np.maximum(bh_initial_prices, 1e-8),
            0.0,
        ).astype(np.float64)
        spy_idx = self.cache.asset_to_index.get("SPY", None)
        bh_spy_shares = 0.0
        if spy_idx is not None and bh_initial_prices[spy_idx] > 0:
            bh_spy_shares = initial_pv / float(bh_initial_prices[spy_idx])

        # ---------- step loop ----------
        t = 0
        rec_portfolio_value[t] = env.portfolio_state.get_total_value()
        rec_benchmark_value[t] = env.benchmark_portfolio_state.get_total_value()
        rec_comparison_value[t] = env.comparison_portfolio_state.get_total_value()
        rec_cash[t] = env.portfolio_state.cash
        rec_positions[t] = env.portfolio_state.positions
        rec_prices[t] = env.portfolio_state.prices
        rec_dates.append(str(self.cache.dates[env.current_absolute_step]))
        rec_sub_pv[t] = sub_cash + sub_shares * env.portfolio_state.prices
        t += 1

        terminated = truncated = False
        while not (terminated or truncated):
            # Step (1): build per-asset SAA obs from CURRENT (pre-trade) state.
            obs_per_asset = self._build_saa_obs(env, sub_cash, sub_shares, sub_last_action, sub_daily_return)

            # Step (2): SAA predictions (scaled by action_limiting_factor).
            scaled_actions = self._saa_predict_actions(obs_per_asset)

            # Step (3): convert to TradeInstructions with shared-cash logic.
            prices = env.portfolio_state.prices.copy()
            instructions, _req_dshares, exec_delta_notional = self._compute_instructions(
                scaled_actions=scaled_actions,
                sub_cash=sub_cash,
                sub_shares=sub_shares,
                prices=prices,
                total_cash_available=float(env.portfolio_state.cash),
            )

            # Step (4): env.step → advances time, MTM, costs, benchmark/comparison updates.
            _obs_next, _reward, terminated, truncated, info = env.step(instructions)

            # ---- post-step bookkeeping ----
            sub_last_action = scaled_actions.copy()
            # Real per-asset shares now reflected in env state.
            new_shares = env.portfolio_state.positions.astype(np.float32)
            new_prices = env.portfolio_state.prices.astype(np.float32)
            # Uniform cash rebalance: total live cash divided equally across N slots.
            new_total_cash = float(env.portfolio_state.cash)
            new_sub_cash = np.full(N, new_total_cash / N, dtype=np.float32)
            new_sub_pv = new_sub_cash + new_shares * new_prices  # (N,)
            sub_daily_return = (new_sub_pv - prev_sub_pv).astype(np.float32)
            prev_sub_pv = new_sub_pv

            sub_cash = new_sub_cash
            sub_shares = new_shares

            if t < T_max:
                rec_portfolio_value[t] = env.portfolio_state.get_total_value()
                rec_benchmark_value[t] = env.benchmark_portfolio_state.get_total_value()
                rec_comparison_value[t] = env.comparison_portfolio_state.get_total_value()
                rec_cash[t] = env.portfolio_state.cash
                rec_positions[t] = new_shares
                rec_prices[t] = new_prices
                rec_actions[t] = scaled_actions
                rec_target_notional[t] = scaled_actions * (sub_cash + sub_shares * new_prices)
                rec_executed_delta[t] = exec_delta_notional
                rec_costs[t] = float(info.get("transaction_cost", 0.0)) if isinstance(info, dict) else 0.0
                rec_sub_pv[t] = new_sub_pv
                rec_dates.append(str(self.cache.dates[env.current_absolute_step]))
                # Daily portfolio return.
                rec_returns[t] = (
                    rec_portfolio_value[t] / rec_portfolio_value[t - 1] - 1.0
                    if rec_portfolio_value[t - 1] > 0 else 0.0
                )
            t += 1

        # Trim arrays to actual length.
        T = min(t, T_max)
        portfolio_value = rec_portfolio_value[:T]
        benchmark_value = rec_benchmark_value[:T]
        comparison_value = rec_comparison_value[:T]
        positions = rec_positions[:T]
        prices_hist = rec_prices[:T]
        actions = rec_actions[:T]
        returns = rec_returns[1:T]  # skip leading 0 from index 0
        costs = rec_costs[:T]
        sub_pv_hist = rec_sub_pv[:T]
        dates = rec_dates[:T]

        # Buy-and-hold portfolio equity (uniform initial allocation, never traded).
        bh_portfolio_value = (bh_shares_per_asset[None, :] * prices_hist).sum(axis=1)  # (T,)
        # Per-asset buy-and-hold equity: shares = (initial_pv / N) / initial_price_i
        bh_per_asset_value = bh_shares_per_asset[None, :] * prices_hist  # (T, N)
        # Buy-and-hold SPY (same starting notional = initial_pv)
        if spy_idx is not None:
            bh_spy_value = bh_spy_shares * prices_hist[:, spy_idx]
        else:
            bh_spy_value = np.full(T, np.nan, dtype=np.float64)

        record = {
            "episode": episode_idx,
            "dates": dates,
            "portfolio_value": portfolio_value,
            "benchmark_value": benchmark_value,
            "comparison_value": comparison_value,
            "bh_portfolio_value": bh_portfolio_value,
            "bh_per_asset_value": bh_per_asset_value,
            "bh_spy_value": bh_spy_value,
            "returns": returns,
            "positions": positions,
            "prices": prices_hist,
            "actions": actions,
            "costs": costs,
            "sub_pv": sub_pv_hist,
            "initial_pv": initial_pv,
            "info_terminal": info if isinstance(info, dict) else {},
        }
        return record

    def run(self) -> List[Dict[str, Any]]:
        for ep in range(self.num_episodes):
            print(f"\n[SAA-Test] === Episode {ep + 1}/{self.num_episodes} ===")
            try:
                rec = self.run_episode(ep)
            except Exception:
                print(f"[SAA-Test] Episode {ep} failed:")
                traceback.print_exc()
                continue
            self.episode_records.append(rec)
            self._print_episode_summary(rec)
        return self.episode_records

    # ---------- summary printout ----------
    @staticmethod
    def _print_episode_summary(rec: Dict[str, Any]) -> None:
        pv = rec["portfolio_value"]
        bench = rec["benchmark_value"]
        bh = rec["bh_portfolio_value"]
        ret = rec["returns"]
        final_ret = (pv[-1] / pv[0] - 1.0) if pv[0] > 0 else 0.0
        bench_ret = (bench[-1] / bench[0] - 1.0) if bench[0] > 0 else 0.0
        bh_ret = (bh[-1] / bh[0] - 1.0) if bh[0] > 0 else 0.0
        sharpe = _annual_sharpe(ret)
        mdd = _max_drawdown(pv)
        print(
            f"  steps={len(pv) - 1:4d} | SAA-port ret={final_ret * 100:7.2f}%  "
            f"bench={bench_ret * 100:7.2f}%  BH={bh_ret * 100:7.2f}%  "
            f"Sharpe={sharpe:5.2f}  MaxDD={mdd * 100:6.2f}%  "
            f"costs=${rec['costs'].sum():.2f}"
        )


# ================================
# Finance metric helpers
# ================================
def _annual_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def _annual_vol(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray) -> float:
    """Max drawdown as a negative fraction (e.g., -0.27 = -27%)."""
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / np.where(running_max > 0, running_max, 1.0)
    return float(dd.min())


def _curve_metrics(equity: np.ndarray) -> Dict[str, float]:
    eq = np.asarray(equity, dtype=np.float64)
    rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1.0)
    total = (eq[-1] / eq[0] - 1.0) if eq[0] > 0 else 0.0
    return {
        "total_return": float(total),
        "annual_sharpe": _annual_sharpe(rets),
        "annual_volatility": _annual_vol(rets),
        "max_drawdown": _max_drawdown(eq),
        "final_value": float(eq[-1]),
    }


# ================================
# Reporting (matplotlib, academic style)
# ================================
def _plot_equity_comparison(
    out_path: str,
    curves: Dict[str, np.ndarray],
    title: str,
    x_label: str = "Trading day",
    y_label: str = "Portfolio value (USD)",
) -> None:
    """Plot multiple equity curves on one axis with annotated final-period metrics."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for name, curve in curves.items():
        ax.plot(curve, label=name)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="best", frameon=False)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_asset_panel(
    out_path: str,
    asset_name: str,
    prices: np.ndarray,        # (T,)
    actions: np.ndarray,       # (T,) scaled action sent
    positions: np.ndarray,     # (T,) shares
) -> None:
    """Three-panel diagnostic plot: price, action, position over time for one asset."""
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.2), sharex=True)
    axes[0].plot(prices, color="black")
    axes[0].set_ylabel("Price (USD)")
    axes[0].set_title(f"{asset_name}: price, SAA action, position")
    axes[1].plot(actions, color="tab:blue")
    axes[1].axhline(0.0, color="grey", linestyle="--", linewidth=0.6)
    axes[1].set_ylabel("Action (scaled)")
    axes[2].plot(positions, color="tab:orange")
    axes[2].set_ylabel("Position (shares)")
    axes[2].set_xlabel("Trading day")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _generate_report(
    records: List[Dict[str, Any]],
    asset_names: List[str],
    output_dir: str,
) -> Dict[str, Any]:
    """Generate all five comparison plots + JSON report. Returns aggregate dict."""
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    _apply_paper_style()

    per_episode_metrics: List[Dict[str, Any]] = []
    for rec in records:
        ep_idx = rec["episode"]
        pv = rec["portfolio_value"]
        bench = rec["benchmark_value"]
        bh = rec["bh_portfolio_value"]
        bh_spy = rec["bh_spy_value"]
        bh_per_asset = rec["bh_per_asset_value"]    # (T, N)
        sub_pv = rec["sub_pv"]                       # (T, N)
        prices = rec["prices"]                       # (T, N)
        actions = rec["actions"]                     # (T, N)
        positions = rec["positions"]                 # (T, N)

        # --- Comparison 1: SAA portfolio vs benchmark (env's allocated benchmark) ---
        _plot_equity_comparison(
            out_path=os.path.join(fig_dir, f"ep{ep_idx:02d}_01_saa_vs_benchmark.pdf"),
            curves={"SAA portfolio": pv, "Benchmark (equal-rebal)": bench},
            title=f"Episode {ep_idx}: SAA portfolio vs benchmark",
        )

        # --- Comparison 2: SAA portfolio vs buy-and-hold uniform portfolio ---
        _plot_equity_comparison(
            out_path=os.path.join(fig_dir, f"ep{ep_idx:02d}_02_saa_vs_bh_portfolio.pdf"),
            curves={"SAA portfolio": pv, "Buy-and-hold (uniform)": bh},
            title=f"Episode {ep_idx}: SAA portfolio vs buy-and-hold portfolio",
        )

        # --- Comparison 3: per-asset SAA sub-pv vs buy-and-hold of that asset ---
        for a_idx, a_name in enumerate(asset_names):
            _plot_equity_comparison(
                out_path=os.path.join(fig_dir, f"ep{ep_idx:02d}_03_perasset_{a_name}.pdf"),
                curves={
                    f"SAA sub-portfolio [{a_name}]": sub_pv[:, a_idx],
                    f"Buy-and-hold [{a_name}]": bh_per_asset[:, a_idx],
                },
                title=f"Episode {ep_idx}: SAA sub-portfolio vs B&H — {a_name}",
            )

        # --- Comparison 4: SAA portfolio vs buy-and-hold SPY (same notional) ---
        if not np.all(np.isnan(bh_spy)):
            _plot_equity_comparison(
                out_path=os.path.join(fig_dir, f"ep{ep_idx:02d}_04_saa_vs_bh_spy.pdf"),
                curves={"SAA portfolio": pv, "Buy-and-hold SPY": bh_spy},
                title=f"Episode {ep_idx}: SAA portfolio vs buy-and-hold SPY",
            )

        # --- Comparison 5: per-asset price / action / position panels ---
        for a_idx, a_name in enumerate(asset_names):
            _plot_asset_panel(
                out_path=os.path.join(fig_dir, f"ep{ep_idx:02d}_05_panel_{a_name}.pdf"),
                asset_name=a_name,
                prices=prices[:, a_idx],
                actions=actions[:, a_idx],
                positions=positions[:, a_idx],
            )

        # --- Metrics per episode ---
        ep_metrics = {
            "episode": int(ep_idx),
            "saa_portfolio": _curve_metrics(pv),
            "benchmark": _curve_metrics(bench),
            "bh_portfolio": _curve_metrics(bh),
            "bh_spy": _curve_metrics(bh_spy) if not np.all(np.isnan(bh_spy)) else None,
            "total_transaction_costs": float(rec["costs"].sum()),
            "per_asset": {
                a_name: {
                    "saa_sub_portfolio": _curve_metrics(sub_pv[:, a_idx]),
                    "buy_and_hold": _curve_metrics(bh_per_asset[:, a_idx]),
                    "action_mean": float(actions[:, a_idx].mean()),
                    "action_std": float(actions[:, a_idx].std()),
                    "trade_days": int(np.sum(np.abs(np.diff(positions[:, a_idx])) > 0)),
                }
                for a_idx, a_name in enumerate(asset_names)
            },
        }
        per_episode_metrics.append(ep_metrics)

    # Aggregate
    def _mean_field(field_path: List[str]) -> float:
        vals: List[float] = []
        for em in per_episode_metrics:
            cur: Any = em
            for p in field_path:
                cur = cur.get(p, None) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if cur is not None and isinstance(cur, (int, float)):
                vals.append(float(cur))
        return float(np.mean(vals)) if vals else 0.0

    aggregate = {
        "mean_saa_total_return": _mean_field(["saa_portfolio", "total_return"]),
        "mean_saa_sharpe": _mean_field(["saa_portfolio", "annual_sharpe"]),
        "mean_saa_max_drawdown": _mean_field(["saa_portfolio", "max_drawdown"]),
        "mean_bh_total_return": _mean_field(["bh_portfolio", "total_return"]),
        "mean_benchmark_total_return": _mean_field(["benchmark", "total_return"]),
        "mean_total_transaction_costs": float(np.mean([em["total_transaction_costs"] for em in per_episode_metrics])) if per_episode_metrics else 0.0,
    }

    report = {
        "test_timestamp": datetime.now().isoformat(),
        "num_episodes_completed": len(per_episode_metrics),
        "asset_names": asset_names,
        "per_episode_metrics": per_episode_metrics,
        "aggregate": aggregate,
    }

    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n[SAA-Test] Report: {report_path}")
    print(f"[SAA-Test] Figures: {fig_dir}")
    return report


# ================================
# Entry point (matches main.py contract)
# ================================
def run(cache: MarketDataCache, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point. Loads SAA model, runs validation episodes, generates report.
    """
    print("\n" + "=" * 70)
    print("SAA PORTFOLIO INFERENCE TEST")
    print("=" * 70)

    try:
        test_cfg = config.get("test_agent", {})
        num_episodes = int(test_cfg.get("num_episodes", 5))
        device = torch.device(test_cfg.get("device", "cpu"))
        model_run_dir = test_cfg.get("saa_model_run_dir")
        if not model_run_dir:
            raise ValueError("config['test_agent']['saa_model_run_dir'] is required.")
        seed = int(config.get("training", {}).get("seed", 42))

        model, vecnorm = _load_saa_assets(model_run_dir, device)

        runner = SAAPortfolioInferenceRunner(
            cache=cache,
            config=config,
            model=model,
            vecnorm=vecnorm,
            device=device,
            num_episodes=num_episodes,
            seed=seed,
        )
        records = runner.run()
        if not records:
            raise RuntimeError("No episodes completed successfully.")

        # Output dir: <agent_dir>/<report_subdir>/<timestamp>/
        agent_dir = os.path.dirname(__file__)
        report_subdir = test_cfg.get("report_subdir", "reports")
        timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        output_dir = os.path.join(agent_dir, report_subdir, timestamp)
        report = _generate_report(records, runner.asset_names, output_dir)

        agg = report["aggregate"]
        print("\n" + "=" * 70)
        print("AGGREGATE SUMMARY (over completed episodes)")
        print("=" * 70)
        print(f"Mean SAA total return:    {agg['mean_saa_total_return'] * 100:7.2f}%")
        print(f"Mean SAA Sharpe:          {agg['mean_saa_sharpe']:7.4f}")
        print(f"Mean SAA max drawdown:    {agg['mean_saa_max_drawdown'] * 100:7.2f}%")
        print(f"Mean B&H total return:    {agg['mean_bh_total_return'] * 100:7.2f}%")
        print(f"Mean benchmark return:    {agg['mean_benchmark_total_return'] * 100:7.2f}%")
        print(f"Mean transaction costs:   ${agg['mean_total_transaction_costs']:.2f}")

        return {
            "agent": "test_saa_inference_portfolio",
            "status": "completed",
            "num_episodes": report["num_episodes_completed"],
            "output_dir": output_dir,
            "aggregate": agg,
        }

    except Exception as e:
        print("\n[SAA-Test] Test failed:")
        traceback.print_exc()
        return {
            "agent": "test_saa_inference_portfolio",
            "status": "failed",
            "error": str(e),
        }
