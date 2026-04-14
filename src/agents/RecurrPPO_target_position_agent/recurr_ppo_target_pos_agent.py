# assets to use: return_1d, volume_percentile_40d, rsi_14, macd, bb_position, 
# bb_width, volatility_5d, market_beta, avg_correlation, composite_risk_score, return_kurtosis

"""
--- Recurrent PPO target position agent ---

Trains on a batch of perceptions which are several consecutive days of interacting with the environment.
This method of training in batches is essential as decisions in finance have delayed rewards/effects. 
Output of the agent is a position change between -1 and 1. Uses a recurrent policy network implemented with LSTM. 
Agent uses SB3 RecurrentPPO algorithm to train the policy network. The agent updates its policy
based on the differential sortino rewards received from the environment. 
The agent uses GAE (Generalized Advantage Estimation). 

Agent randomly decides each episode which asset to trade from a predefined list of assets which is retrieved
from the environments asset universe. The agent only trades one asset per episode.

The agent uses the features which are marked as true in the config file

The basic architecture of the agent is as follows:
MlpLstmPolicy from sb3_contrib as the policy network architecture.
RecurrentPPO from sb3_contrib as the RL algorithm.
Input Layer -> Feature Extraction Layer (Dense) -> LSTM Layer -> Actor-Critic Heads (Dense) -> Output Layer

Any paramteres used will come from config file(s) passed to the agent. An interesting
parameter is the action scaling factor which scales the output of the agent to the desired range.
The desired action scaling is defined in the config and stays constant at 5% of total portfolio value
until 20% of trained episodes have passed, then it linearly rises to 40% at 80% 
of trained episodes and stays at 40% until training is complete.

Uses functions to define individual schedules for: learning rate, entropy coefficient, clip range,
and action scaling factor. These schedules adjust the respective parameters over the course of training.

Triggers execution_mode SINGLE_ASSET_TARGET_POS in the environment which means the agent outputs target positions between -1 and 1.

NOTE: RecurrentPPO is in sb3_contrib package which is separate from stable_baselines3 package. 
For reference read: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
It is particularly important to pass the lstm_states and episode_start argument to the predict() method, so the cell and 
hidden states of the LSTM are correctly updated.
RecurrentPPOs temporal memory is driven by rollouts (n_steps) and episode boundaries; obs need not include a lookback window. 
Envs differential Sortino reward is stepwise and benefits from longer n_steps to capture delayed effects.
"""

# imports
import os
import time
import numpy as np
from typing import Callable, Dict, Any, Optional
from datetime import datetime


import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
#from stable_baselines3.common.utils import configure_logger

import gymnasium as gym

# Import the single-asset environment (reuses data containers/classes defined there)
# The `cache` object passed by main.py comes from src.environment.single_asset_target_pos_drl_trading_env.MarketDataCache;
# runtime type checks are duck-typed—if attributes match, it works.
from src.environment.single_asset_target_pos_drl_trading_env import TradingEnv
import json


# ================================
# Utility: Schedules (LR/Entropy/Clip/Action)
# ================================

def linear_three_phase_schedule(start: float, end: float, warmup_pct: float, ramping_pct: float) -> Callable[[float], float]:
    """
    Three-phase linear schedule:
    - Warmup: keep start value for [0, warmup_pct] of training progress.
    - Ramping: linearly interpolate from start to end for (warmup_pct, ramping_pct].
    - Hold: keep end value for (ramping_pct, 1.0].

    Args:
        start: initial value
        end: final value
        warmup_pct: fraction of total timesteps for warmup [0..1]
        ramping_pct: fraction where ramp completes [warmup_pct..1]
    Returns:
        schedule(progress_remaining) where progress_remaining in [0,1]
        NOTE: SB3 schedules receive "progress_remaining" = 1.0 → start of training, 0.0 → end.
              We convert to "progress_elapsed" = 1 - progress_remaining.
    """
    warmup_pct = float(np.clip(warmup_pct, 0.0, 1.0))
    ramping_pct = float(np.clip(ramping_pct, warmup_pct, 1.0))

    def schedule(progress_remaining: float) -> float:
        progress_elapsed = 1.0 - float(progress_remaining)
        if progress_elapsed <= warmup_pct:
            return float(start)
        if progress_elapsed <= ramping_pct:
            # Linear interpolation between start and end
            phase_len = max(ramping_pct - warmup_pct, 1e-8)
            frac = (progress_elapsed - warmup_pct) / phase_len
            return float(start + (end - start) * frac)
        return float(end)

    return schedule

# entropy needs a class callback because SB3 does not support callable schedules for ent_coef
class EntropyScheduleCallback(BaseCallback):
    """
    Updates model.ent_coef using a two-phase linear schedule each rollout.
    Uses model._current_progress_remaining provided by SB3 during training.
    """
    def __init__(self, start: float, end: float, warmup_pct: float, ramping_pct: float, verbose: int = 0):
        super().__init__(verbose)
        self._sched = linear_three_phase_schedule(start, end, warmup_pct, ramping_pct)

    def _on_rollout_end(self) -> bool:
        # progress_remaining in [1.0 -> 0.0]
        pr = getattr(self.model, "_current_progress_remaining", 0.0)
        new_ent = float(self._sched(pr))
        # SB3 uses scalar float for ent_coef; assign directly
        self.model.ent_coef = new_ent
        return True
    
    def _on_step(self) -> bool:
        # Required abstract method; return True to continue training
        return True

# ================================
# Custom Logger Callback (portfolio metrics into tb logs)
# ================================
class EpisodePortfolioSB3LoggerCallback(BaseCallback):
    def __init__(self, tag_prefix: str = "train", log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)

        self.tag_prefix = tag_prefix
        self.log_freq = log_freq
        self.episode_count = 0 # Track completed episodes

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        if info.get("episode_final", False):
            self.episode_count += 1

            # Only log every log_freq episodes
            if self.episode_count % self.log_freq != 0:
                return True # Skip logging but continue training
            
            # Extract portfolio metrics from info dict
            pv = info.get("portfolio_final_value", None)
            comp_pv = info.get("comparison_final_value", None)
            bench = info.get("benchmark_final_value", None)
            ret = info.get("portfolio_return", None)
            sharpe = info.get("episode_sharpe", None)
            dd = info.get("episode_max_drawdown", None)
            if pv is not None:
                self.model.logger.record(f"{self.tag_prefix}/portfolio_final_value", pv, exclude=("stdout",))
            if comp_pv is not None:
                self.model.logger.record(f"{self.tag_prefix}/comparison_final_value", comp_pv, exclude=("stdout",))
            if pv is not None and comp_pv is not None:
                return_difference_after_random_portfolio_initialization = pv - comp_pv
                self.model.logger.record(f"{self.tag_prefix}/return_diff_after_rdn_portf_init", return_difference_after_random_portfolio_initialization, exclude=("stdout",))
            if bench is not None:
                self.model.logger.record(f"{self.tag_prefix}/benchmark_final_value", bench, exclude=("stdout",))
            if ret is not None:
                self.model.logger.record(f"{self.tag_prefix}/portfolio_return", ret, exclude=("stdout",))
            if sharpe is not None:
                self.model.logger.record(f"{self.tag_prefix}/episode_sharpe", sharpe, exclude=("stdout",))
            if dd is not None:
                self.model.logger.record(f"{self.tag_prefix}/episode_max_drawdown", dd, exclude=("stdout",))
            # Turnover and costs
            if "total_turnover" in info:
                self.model.logger.record(f"{self.tag_prefix}/episode_turnover", float(info["total_turnover"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/avg_turnover", float(info["avg_turnover"]), exclude=("stdout",))
            if "total_transaction_cost" in info:
                self.model.logger.record(f"{self.tag_prefix}/cost_total", float(info["total_transaction_cost"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/cost_commission", float(info["episode_cost_commission"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/cost_spread", float(info["episode_cost_spread"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/cost_impact", float(info["episode_cost_impact"]), exclude=("stdout",))
            # Exposure
            if "exposure_avg" in info:
                self.model.logger.record(f"{self.tag_prefix}/exposure_start", float(info["exposure_start"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/exposure_avg", float(info["exposure_avg"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/exposure_end", float(info["exposure_end"]), exclude=("stdout",))
            # Gross vs net
            if "shadow_return" in info:
                self.model.logger.record(f"{self.tag_prefix}/gross_return", float(info["shadow_return"]), exclude=("stdout",))
            # Buy/Sell totals and trade sizes
            if "total_buy_notional" in info:
                self.model.logger.record(f"{self.tag_prefix}/total_buy_notional", float(info["total_buy_notional"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/total_sell_notional", float(info["total_sell_notional"]), exclude=("stdout",))
            if "trade_size_mean" in info:
                self.model.logger.record(f"{self.tag_prefix}/trade_size_mean", float(info["trade_size_mean"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/trade_size_median", float(info["trade_size_median"]), exclude=("stdout",))
            # Action stats
            if "action_mean" in info:
                self.model.logger.record(f"{self.tag_prefix}/action_mean", float(info["action_mean"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/action_median", float(info["action_median"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/action_p05", float(info["action_p05"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/action_p25", float(info["action_p25"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/action_p75", float(info["action_p75"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/action_p95", float(info["action_p95"]), exclude=("stdout",))
            # Sortino components
            if "sortino_mean_ema" in info:
                self.model.logger.record(f"{self.tag_prefix}/sortino_mean_ema", float(info["sortino_mean_ema"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/sortino_downside_ema", float(info["sortino_downside_ema"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/sortino_reward_raw_mean", float(info["sortino_reward_raw_mean"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/sortino_reward_raw_p25", float(info["sortino_reward_raw_p25"]), exclude=("stdout",))
                self.model.logger.record(f"{self.tag_prefix}/sortino_reward_raw_p75", float(info["sortino_reward_raw_p75"]), exclude=("stdout",))
        return True


class ValidationMetricsCallback(BaseCallback):
    """
    Captures metrics from EvalCallback's evaluation episodes.
    Must be passed TO the EvalCallback as a callback parameter.
    """
    def __init__(self, tag_prefix: str = "validation", verbose: int = 0):
        super().__init__(verbose)
        self.tag_prefix = tag_prefix
        self.eval_episode_count = 0

        # Buffers to accumulate per-episode metrics
        self.pv_buffer = []
        self.comp_pv_buffer = []
        self.bench_buffer = []
        self.ret_buffer = []
        self.sharpe_buffer = []
        self.dd_buffer = []
        self.alpha_ret_buffer = []
        self.cum_reward_buffer = []
        self.saa_subpf_buffer = []
        self.saa_return_after_rdn_portf_init_buffer = []

        # Store last computed return_diff_vs_init_mean for best-model selection
        self.last_return_diff_total_mean = None
        self.last_return_diff_target_mean = None

    def _on_step(self) -> bool:
        """Called during evaluation episodes (by EvalCallback)."""
        info = self.locals.get("infos", [{}])[0]
        if info.get("episode_final", False):
            self.eval_episode_count += 1
            
            # Extract metrics
            pv = info.get("portfolio_final_value", None)
            comp_pv = info.get("comparison_final_value", None)
            bench = info.get("benchmark_final_value", None)
            ret = info.get("portfolio_return", None)
            sharpe = info.get("episode_sharpe", None)
            dd = info.get("episode_max_drawdown", None)
            alpha_ret = info.get("alpha_return", None)
            cum_reward = info.get("cumulative_reward", None)
            saa_subpf = info.get("saa_final_subportfolio_value", None)
            saa_return_after_rdn_portf_init = info.get("saa_return_final", None)
            
            # Accumulate all eval episode metricsin buffers
            if pv is not None:
                self.pv_buffer.append(pv)
            if comp_pv is not None:
                self.comp_pv_buffer.append(comp_pv)
            if bench is not None:
                self.bench_buffer.append(bench)
            if ret is not None:
                self.ret_buffer.append(ret)
            if sharpe is not None:
                self.sharpe_buffer.append(sharpe)
            if dd is not None:
                self.dd_buffer.append(dd)
            if alpha_ret is not None:
                self.alpha_ret_buffer.append(alpha_ret)
            if cum_reward is not None:
                self.cum_reward_buffer.append(cum_reward)
            if saa_subpf is not None:
                self.saa_subpf_buffer.append(saa_subpf)
            if saa_return_after_rdn_portf_init is not None:
                self.saa_return_after_rdn_portf_init_buffer.append(saa_return_after_rdn_portf_init)
                
        return True
    
    def flush_metrics(self, n_eval_episodes: int) -> None:
        """
        Called after all eval episodes complete to log aggregated metrics.
        
        Args:
            n_eval_episodes: expected number of episodes (used to verify buffer is full)
        """
        # Only log if we have accumulated the expected number of episodes
        if self.eval_episode_count < n_eval_episodes:
            return
        
        # Compute and log means
        if self.pv_buffer:
            mean_pv = float(np.mean(self.pv_buffer))
            self.model.logger.record(f"{self.tag_prefix}/portfolio_final_value_mean", mean_pv, exclude=("stdout",))
            std_pv = float(np.std(self.pv_buffer))
            self.model.logger.record(f"{self.tag_prefix}/portfolio_final_value_std", std_pv, exclude=("stdout",))
        
        if self.comp_pv_buffer:
            mean_comp = float(np.mean(self.comp_pv_buffer))
            self.model.logger.record(f"{self.tag_prefix}/comparison_final_value_mean", mean_comp, exclude=("stdout",))
        
        if self.pv_buffer and self.comp_pv_buffer:
            return_diffs = [pv - comp for pv, comp in zip(self.pv_buffer, self.comp_pv_buffer)]
            mean_return_diff_total = float(np.mean(return_diffs))
            self.model.logger.record(f"{self.tag_prefix}/return_diff_vs_init_total_mean", mean_return_diff_total, exclude=("stdout",))
            self.last_return_diff_total_mean = mean_return_diff_total

        if self.saa_return_after_rdn_portf_init_buffer:
            mean_return_diff_target = float(np.mean(self.saa_return_after_rdn_portf_init_buffer))
            self.model.logger.record(f"{self.tag_prefix}/return_diff_vs_init_target_mean", mean_return_diff_target, exclude=("stdout",))
            self.last_return_diff_target_mean = mean_return_diff_target
        
        if self.bench_buffer:
            mean_bench = float(np.mean(self.bench_buffer))
            self.model.logger.record(f"{self.tag_prefix}/benchmark_final_value_mean", mean_bench, exclude=("stdout",))
        
        if self.ret_buffer:
            mean_ret = float(np.mean(self.ret_buffer))
            self.model.logger.record(f"{self.tag_prefix}/portfolio_return_mean", mean_ret, exclude=("stdout",))
            std_ret = float(np.std(self.ret_buffer))
            self.model.logger.record(f"{self.tag_prefix}/portfolio_return_std", std_ret, exclude=("stdout",))
        
        if self.sharpe_buffer:
            mean_sharpe = float(np.mean(self.sharpe_buffer))
            self.model.logger.record(f"{self.tag_prefix}/episode_sharpe_mean", mean_sharpe, exclude=("stdout",))
            std_sharpe = float(np.std(self.sharpe_buffer))
            self.model.logger.record(f"{self.tag_prefix}/episode_sharpe_std", std_sharpe, exclude=("stdout",))
        
        if self.dd_buffer:
            mean_dd = float(np.mean(self.dd_buffer))
            self.model.logger.record(f"{self.tag_prefix}/episode_max_drawdown_mean", mean_dd, exclude=("stdout",))
        
        if self.alpha_ret_buffer:
            mean_alpha = float(np.mean(self.alpha_ret_buffer))
            self.model.logger.record(f"{self.tag_prefix}/alpha_return_mean", mean_alpha, exclude=("stdout",))
        
        if self.cum_reward_buffer:
            mean_cum_reward = float(np.mean(self.cum_reward_buffer))
            self.model.logger.record(f"{self.tag_prefix}/cumulative_reward_mean", mean_cum_reward, exclude=("stdout",))
        
        # Reset buffers for next eval run
        self._reset_buffers()

    def get_best_return_diff(self) -> Optional[float]:
        """
        Return the last computed return_diff_vs_init_mean.
        Used by EvalCallbackWithMetrics for best-model selection.
        """
        if self.last_return_diff_total_mean is not None:
            return self.last_return_diff_total_mean
        return None
    
    def _reset_buffers(self) -> None:
        """Reset all accumulation buffers at the end of an evaluation run."""
        self.pv_buffer = []
        self.comp_pv_buffer = []
        self.bench_buffer = []
        self.ret_buffer = []
        self.sharpe_buffer = []
        self.dd_buffer = []
        self.alpha_ret_buffer = []
        self.cum_reward_buffer = []
        self.eval_episode_count = 0
        self.saa_subpf_buffer = []
        self.saa_return_after_rdn_portf_init_buffer = []
    

class EvalCallbackWithMetrics(BaseCallback):
    """
    Eval callback that forwards each eval step to a per-episode metrics callback
    (e.g., ValidationMetricsCallback) while preserving SB3 eval logging.

    Saves best model based on last_return_diff_target_mean (value creation metric)
    rather than raw cumulative reward.
    """
    def __init__(
        self,
        eval_env,
        best_model_save_path,
        log_path,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool,
        render: bool,
        callback_on_new_best=None,
        callback_after_eval=None,
        eval_step_callback: Optional[BaseCallback] = None,
        warn: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.callback_on_new_best = callback_on_new_best
        self.callback_after_eval = callback_after_eval
        self.eval_step_callback = eval_step_callback
        self.warn = warn
        self.best_mean_reward = -np.inf
        self.best_return_diff = -np.inf # Track best last_return_diff_target_mean
        self.n_eval_calls = 0

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.eval_step_callback is not None:
            self.eval_step_callback.init_callback(self.model)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            def _step_cb(locals_, globals_):
                if self.eval_step_callback is None:
                    return True
                self.eval_step_callback.locals = locals_
                self.eval_step_callback.globals = globals_
                return bool(self.eval_step_callback.on_step())

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=self.render,
                return_episode_rewards=True,
                warn=self.warn,
                callback=_step_cb,
            )
            self.n_eval_calls += 1

            # Flush aggregated validation metrics after all eval episodes complete
            if self.eval_step_callback is not None:
                self.eval_step_callback.flush_metrics(self.n_eval_episodes)
                
            # Log standard SB3 eval metrics
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length = np.mean(episode_lengths)
            self.model.logger.record("eval/mean_reward", float(mean_reward))
            self.model.logger.record("eval/std_reward", float(std_reward))
            self.model.logger.record("eval/mean_ep_length", float(mean_ep_length))
            if self.log_path is not None:
                np.savez(
                    os.path.join(self.log_path, "evaluations.npz"),
                    timesteps=np.array([self.num_timesteps]),
                    results=np.array([episode_rewards]),
                    ep_lengths=np.array([episode_lengths]),
                )

            # Check if this is a new best model based on last_return_diff_target_mean
            current_return_diff = None
            if isinstance(self.eval_step_callback, ValidationMetricsCallback):
                current_return_diff = self.eval_step_callback.get_best_return_diff()
            
            if current_return_diff is not None and current_return_diff > self.best_return_diff:
                self.best_return_diff = current_return_diff
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    # Save VecNormalize stats at the exact same checkpoint time
                    vec_env = self.model.get_env()
                    if isinstance(vec_env, VecNormalize):
                        vec_env.save(os.path.join(self.best_model_save_path, "best_model_vecnormalize.pkl"))
                        print(f"New best model saved! last_return_diff_target_mean: {current_return_diff:.4f}")
                    else:
                        raise RuntimeError(f"New best model saved! last_return_diff_target_mean: {current_return_diff:.4f} (VecNormalize not found!)")
                if self.callback_on_new_best is not None:
                    self.callback_on_new_best.on_step()
            
            if self.callback_after_eval is not None:
                self.callback_after_eval.on_step()
        return True


class ProgressSyncCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_rollout_end(self) -> bool:
        # SB3 updates _current_progress_remaining each rollout
        pr = float(getattr(self.model, "_current_progress_remaining", 0.0))
        vec_env = self.model.get_env()  # DummyVecEnv
        # Propagate to all sub-envs (usually 1)
        for i, env in enumerate(vec_env.envs):
            # env is our SingleAssetEpisodeAdapter
            if hasattr(env, "set_progress_remaining"):
                env.set_progress_remaining(pr)
        return True

    def _on_step(self) -> bool:
        # No-op; we only need to sync per rollout to avoid overhead
        return True
    

# ================================
# Environment Wrappers
# ================================

class SingleAssetEpisodeAdapter(gym.Wrapper):
    """
    Adapter wrapper for single-asset target-position training.

    Responsibilities:
    - On `reset()`: Sample a tradable asset symbol and pass it to the underlying env reset.
    - On `step(action)`: Forward the selected asset to env.step(action, asset=...).
    - Action scaling: optional multiplicative scaling of agent action via `action_factor_fn`, bounded [-1, 1].
    - Keeps interface compatible with SB3 (no changes to action/obs spaces).

    Contract:
    - Underlying env must expose `market_data_cache.asset_names` (list[str]).
    - Underlying env step signature supports `step(action, asset: Optional[str])`.
    - Action space is continuous Box, typically shape (1,), range [-1, 1].
    """

    def __init__(self, env: gym.Env, action_factor_fn: Optional[Callable[[float], float]] = None, seed: Optional[int] = None):
        super().__init__(env)
        self._rng = np.random.default_rng(int(seed) if seed is not None else None)
        self._selected_asset: Optional[str] = None
        self._action_factor_fn = action_factor_fn
        self._progress_remaining: float = 1.0  # start-of-training default

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # Randomly choose asset per episode (uniform; extendable to weighted choice)
        asset_universe = getattr(self.env.market_data_cache, "asset_names", None)
        if not asset_universe:
            raise RuntimeError("No assets available in market_data_cache.asset_names")
        self._selected_asset = str(self._rng.choice(asset_universe))
        # Reset the underlying env, passing the selected asset
        obs, info = self.env.reset(seed=seed, option=options, asset=self._selected_asset)
        # SB3-Contrib RecurrentPPO internally handles episode_starts; no explicit flag needed here.
        return obs, info
    
    def set_progress_remaining(self, pr: float) -> None:
        # Called by ProgressSyncCallback
        self._progress_remaining = float(np.clip(pr, 0.0, 1.0))

    def step(self, action):
        # Optional multiplicative action scaling (schedule-driven)
        if self._action_factor_fn is not None:
            raw = np.asarray(action, dtype=np.float32)
            scale = float(self._action_factor_fn(progress_remaining=self._progress_remaining))
            scaled_action = np.clip(raw * scale, -1.0, 1.0).astype(np.float32)

        else:
            raise RuntimeError("Action factor function is not defined.")
        
        # Forward call with selected asset
        obs, reward, terminated, truncated, info = self.env.step(scaled_action, asset=self._selected_asset)
        return obs, reward, terminated, truncated, info


# ================================
# Agent Builder
# ================================

def build_env(cache, config: Dict[str, Any], seed: Optional[int] = None, for_eval: bool = False) -> gym.Env:
    """
    Build a single-asset training/eval environment from config and cache.

    Args:
        cache: MarketDataCache instance provided by main.py (duck-typed).
        config: full configuration dict (environment, saa_features, agent).
        seed: optional PRNG seed for environment internal sampling.
        for_eval: choose eval mode ('validation' blocks) if True, else training.

    Returns:
        Gymnasium Env, wrapped to select one asset per episode and (optionally) scale actions.
    """
    mode = "validation" if for_eval else "train"
    env = TradingEnv(config, cache, mode=mode)

    # Action limiting factor schedule (multiplicative) — supports flat agent config keys
    agent_cfg = config.get("agent", {})
    alf_keys_present = any(k in agent_cfg for k in (
        "action_limiting_factor_start",
        "action_limiting_factor_end",
        "action_limiting_factor_schedule_type",
        "action_limiting_factor_schedule_warmup_pct",
        "action_limiting_factor_schedule_ramping_pct",
    ))

    if alf_keys_present:
        start = float(agent_cfg.get("action_limiting_factor_start", 0.2))
        end = float(agent_cfg.get("action_limiting_factor_end", start))
        schedule_type = str(agent_cfg.get("action_limiting_factor_schedule_type", "linear")).lower()
        warmup_pct = float(agent_cfg.get("action_limiting_factor_schedule_warmup_pct", 0.2))
        ramping_pct = float(agent_cfg.get("action_limiting_factor_schedule_ramping_pct", 0.6))

        if schedule_type == "linear":
            alf_schedule = linear_three_phase_schedule(
                start=start,
                end=end,
                warmup_pct=warmup_pct,
                ramping_pct=ramping_pct,
            )
            def alf_fn(progress_remaining: float = 0.0) -> float:
                return float(alf_schedule(progress_remaining))
        else:
            # NOTE: Currently only linear three schedule is supported.
            # Fallback: constant scaling using `start`
            def alf_fn(progress_remaining: float = 0.0) -> float:
                return float(start)
    else:
        # Default constant scaling if not provided
        def alf_fn(progress_remaining: float = 0.0) -> float:
            return 0.2 # Default 20% of portfolio value

    # Wrap env in the episode adapter to choose asset and apply action scaling
    wrapped = SingleAssetEpisodeAdapter(env=env, action_factor_fn=alf_fn, seed=seed)

    return wrapped


def build_policy_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build `policy_kwargs` for MlpLstmPolicy from config.

    Required keys (with examples):
    - n_lstm_layers: int (e.g., 1)
    - lstm_hidden_size: int (e.g., 128)
    - shared_lstm: bool (e.g., True)
    - net_arch: {just examplatory: "pi": [128], "vf": [128]} (actor/critic MLP heads sizes)

    Returns:
        dict for MlpLstmPolicy construction.
    """
    pk = config.get("agent", {}).get("policy_kwargs", {})
    # SB3-Contrib expects exact key names:
    return {
        "net_arch": pk.get("net_arch", {"pi": [128], "vf": [128]}),
        "n_lstm_layers": int(pk.get("n_lstm_layers", 1)),
        "lstm_hidden_size": int(pk.get("lstm_hidden_size", 128)),
        "features_extractor_class": InputMLPFeatures,
        "features_extractor_kwargs": {
            "features_dim": 64,
            "mlp_hidden_sizes": pk.get("mlp_hidden_sizes", [128, 128]),
            "mlp_dropouts": pk.get("mlp_dropouts", [0.3, 0.3]),
            "mlp_activation": pk.get("mlp_activation", "SiLU")
        },

        "ortho_init": True # init with orthogonal matrices for stability
    }


def build_model(env: gym.Env, config: Dict[str, Any]) -> RecurrentPPO:
    """
    Instantiate RecurrentPPO with schedules and hyperparameters from config.

    Critical settings:
    - `n_steps`: rollout length (sequence of days per update). Must be > 1 to capture delayed rewards.
    - `batch_size`: minibatch size for updates.
    - Schedules: learning rate, entropy coefficient, clip range.
    - Policy: MlpLstmPolicy with `policy_kwargs`.

    Returns:
        RecurrentPPO model ready for training.
    """
    agent_cfg = config.get("agent", {})

    # Schedules
    lr_sched = linear_three_phase_schedule(
        start=float(agent_cfg.get("learning_rate_start", 3e-4)),
        end=float(agent_cfg.get("learning_rate_end", 3e-5)),
        warmup_pct=float(agent_cfg.get("lr_schedule_warmup_pct", 0.2)),
        ramping_pct=float(agent_cfg.get("lr_schedule_ramping_pct", 0.6)),
    )

    ent_start = float(agent_cfg.get("ent_coef_start", 0.01))

    clip_sched = linear_three_phase_schedule(
        start=float(agent_cfg.get("clip_range_start", 0.2)),
        end=float(agent_cfg.get("clip_range_end", 0.1)),
        warmup_pct=float(agent_cfg.get("clip_schedule_warmup_pct", 0.2)),
        ramping_pct=float(agent_cfg.get("clip_schedule_ramping_pct", 0.6)),
    )

    policy_kwargs = build_policy_kwargs(config)

    # Core PPO-Recurrent hyperparameters
    n_steps = int(agent_cfg.get("n_steps", 4032))         # sequence length per update (days)
    batch_size = int(agent_cfg.get("batch_size", 256))
    gamma = float(agent_cfg.get("gamma", 0.99))
    gae_lambda = float(agent_cfg.get("gae_lambda", 0.95))
    vf_coef = float(agent_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(agent_cfg.get("max_grad_norm", 0.5))
    device = str(agent_cfg.get("device", "auto"))
    normalize_advantage = bool(agent_cfg.get("normalize_advantage", True))
    n_epochs = int(agent_cfg.get("n_epochs", 6) )
    target_kl = float(agent_cfg.get("target_kl", 0.03)) # early stopping based on KL divergence


    # --- Actual model instantiation ---
    model = RecurrentPPO(
        policy=MlpLstmPolicy,
        env=env,
        learning_rate=lr_sched,
        ent_coef=ent_start, # initially float, will be updated using EntropyScheduleCallback
        clip_range=clip_sched,
        target_kl=target_kl,
        n_steps=n_steps, # sequence length (days of interaction per update)
        n_epochs=n_epochs, # number of epochs per update (repeatitions over collected data)
        batch_size=batch_size, # minibatch size for updates (should be a fraction of n_steps)
        gamma=gamma, # discount factor
        gae_lambda=gae_lambda, # GAE lambda: bias-variance tradeoff. 1.0 = high variance, 0.0 = high bias
        vf_coef=vf_coef, # value function loss coefficient: balances actor vs critic loss
        max_grad_norm=max_grad_norm, # gradient clipping: prevents exploding gradients
        policy_kwargs=policy_kwargs, # LSTM and MLP architecture
        device=device,
        normalize_advantage=normalize_advantage,
        verbose=int(agent_cfg.get("verbose", 1)),
        tensorboard_log=r"src\agents\RecurrPPO_target_position_agent\tb_logs",
        stats_window_size=int(agent_cfg.get("stats_window_size", 100))
    )
    return model


def build_eval_callback(eval_env: gym.Env, config: Dict[str, Any], log_dir: str) -> BaseCallback:
    """
    Build an evaluation callback to periodically assess policy on validation blocks.

    Args:
        eval_env: separate environment instance in validation mode.
        config: configuration dict with eval settings.
        log_dir: directory for logs and best-model checkpoint.

    Returns:
        EvalCallback instance for SB3 `learn()`.
    """
    train_cfg = config.get("training", {})
    eval_freq = int(train_cfg.get("eval_freq", 10_000))
    n_eval_episodes = int(train_cfg.get("n_eval_episodes", 10))
    deterministic = bool(train_cfg.get("eval_deterministic", False))

    val_metrics_cb = ValidationMetricsCallback(tag_prefix='validation')

    callback = EvalCallbackWithMetrics(
        eval_env=eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=False,
        callback_on_new_best=None,
        callback_after_eval=None,
        eval_step_callback=val_metrics_cb
    )
    return callback


# ================================
# Custom Feature Extractor
# ================================
class InputMLPFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64,  mlp_hidden_sizes=None, mlp_dropouts=None, mlp_activation="SiLU"):
        super().__init__(observation_space, features_dim)
        n_in = observation_space.shape[0]
        if mlp_hidden_sizes is None:
            mlp_hidden_sizes = [128, 128]  # Default
        if mlp_dropouts is None:
            mlp_dropouts = [0.3] * len(mlp_hidden_sizes)  # Default dropouts matching hidden sizes
        if len(mlp_dropouts) != len(mlp_hidden_sizes):
            raise ValueError("mlp_dropouts must match the length of mlp_hidden_sizes")
        
        activation_map = {"SiLU": nn.SiLU, "ReLU": nn.ReLU, "Tanh": nn.Tanh}
        activation_class = activation_map.get(mlp_activation, nn.SiLU)
        
        layers = []
        prev_size = n_in
        for i, hidden_size in enumerate(mlp_hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                activation_class(),
                nn.Dropout(mlp_dropouts[i]),
            ])
            prev_size = hidden_size
        
        # Output projection
        layers.extend([
            nn.Linear(prev_size, features_dim),
            activation_class(),
        ])
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    

# ================================
# Entry Point for main.py
# ================================

def run(cache, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Program entrypoint (required by main.py).

    Process:
    1) Build training and evaluation environments (single asset target-position mode).
    2) Instantiate RecurrentPPO with LSTM policy and schedules from config.
    3) Train for configured total timesteps, evaluating periodically.
    4) Save model; return summary dict for CLI output.

    Args:
        cache: MarketDataCache from main.py (duck-typed).
        config: dict containing environment, saa_features, agent, training settings.

    Returns:
        dict with summary: paths, timings, and key hyperparameters used.
    """
    # Set seeds for reproducibility (PyTorch, NumPy, env wrappers)
    seed = int(config.get("training", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    gamma_cfg = config.get("agent", {}).get("gamma", 0.99)

    # Build vectorized train/eval envs (DummyVecEnv); RecurrentPPO expects VecEnvs.
    def make_train():
        env = build_env(cache, config, seed=seed, for_eval=False)
        return env

    def make_eval():
        env = build_env(cache, config, seed=seed + 1, for_eval=True)
        return env

    # Train: normalize obs + reward; inline wrapping (no intermediate DummyVecEnv)
    vec_train = VecNormalize(
        DummyVecEnv([make_train]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=np.inf,
        gamma=gamma_cfg,
    )

    # Eval: normalized obs, raw rewards; inline wrapping
    vec_eval = VecNormalize(
        DummyVecEnv([make_eval]),
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=np.inf,
        gamma=gamma_cfg,
    )
    
    # Build model
    model = build_model(vec_train, config)

    agent_dir = os.path.join(os.path.dirname(__file__))

    # Generate custom TensorBoard log name: XXXXX_config_ZZZZZ_YY_MM_DD
    # Extract run_id from json file and increment it
    # run_id = str(config.get("training", {}).get("run_id", "00001")).zfill(5)
    # Load run_id from JSON file
    run_id_file = r"src\data\run_id.json"
    with open(run_id_file, 'r') as f:
        run_id_data = json.load(f)

    current_run_id = int(run_id_data.get("run_id", 0))
    next_run_id = current_run_id + 1

    # Save incremented run_id back to JSON
    with open(run_id_file, 'w') as f:
        json.dump({"run_id": next_run_id}, f)

    run_id = str(current_run_id).zfill(5)
    config_id = str(config.get("training", {}).get("config_id", "00000")).zfill(5)
    
    # Get current date in YY_MM_DD format
    date_str = datetime.now().strftime("%y_%m_%d")
    
    # Format: XXXXX_config_ZZZZZ_YY_MM_DD
    tb_log_name = f"{run_id}_config_{config_id}_{date_str}"

    # Create saved_models directory for model checkpoints
    saved_models_dir = os.path.join(agent_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    
    # Model path with same name as TB log
    model_path = os.path.join(saved_models_dir, f"{tb_log_name}.zip")
    
    # EvalCallback best model path (also in saved_models, subfolder with run name)
    best_model_dir = os.path.join(saved_models_dir, tb_log_name)
    os.makedirs(best_model_dir, exist_ok=True)

    # Evaluation callback
    eval_cb = build_eval_callback(vec_eval, config, best_model_dir)

    # Entropy schedule callback (only if schedule keys are present)
    acfg = config.get("agent", {})
    if all(k in acfg for k in ("ent_coef_start", "ent_coef_end", "ent_coef_schedule_warmup_pct", "ent_coef_schedule_ramping_pct")):
        ent_cb = EntropyScheduleCallback(
            start=float(acfg["ent_coef_start"]),
            end=float(acfg["ent_coef_end"]),
            warmup_pct=float(acfg["ent_coef_schedule_warmup_pct"]),
            ramping_pct=float(acfg["ent_coef_schedule_ramping_pct"]),
        )
    else:
        ent_cb = None # No entropy schedule handling

    # Training metrics callback with throttling
    train_cfg = config.get("training", {})
    train_log_freq = int(train_cfg.get("train_log_freq", 50))  # NEW CONFIG KEY
    train_tb_cb = EpisodePortfolioSB3LoggerCallback(
        tag_prefix="train", 
        log_freq=train_log_freq  # Log every 50 episodes by default
    )

    # Progress sync callback to propagate progress_remaining to envs
    progress_sync_cb = ProgressSyncCallback()

    # Build callback list (validation callback is now inside eval_cb)
    if ent_cb is not None:
        callback = [eval_cb, ent_cb, progress_sync_cb, train_tb_cb]
    else:
        callback = [eval_cb, progress_sync_cb, train_tb_cb]

    # ----------- Train -----------------
    total_timesteps = int(config.get("training", {}).get("total_timesteps", 300_000))

    # Starttime
    t0 = time.time()

    # Train with callbacks (evaluation, entropy scheduling, progress sync, training metrics logging)
    model.learn(
        total_timesteps=total_timesteps, 
        callback=callback, 
        progress_bar=True,
        tb_log_name=tb_log_name
    )
    # Endtime
    t1 = time.time()

    # Save
    model.save(model_path)

    # Return summary (printed in main.py)
    agent_cfg = config.get("agent", {})
    return {
        "agent": "RecurrPPO_target_position_agent",
        "policy": "MlpLstmPolicy",
        "total_timesteps": total_timesteps,
        "elapsed_sec": round(t1 - t0, 2),
        "model_path": model_path,
        "tb_log_name": tb_log_name,
        "run_id": run_id,
        "config_id": config_id,
        "n_steps": int(agent_cfg.get("n_steps", 128)),
        "batch_size": int(agent_cfg.get("batch_size", 256)),
        "gamma": float(agent_cfg.get("gamma", 0.99)),
        "gae_lambda": float(agent_cfg.get("gae_lambda", 0.95)),
        "lstm_hidden_size": int(agent_cfg.get("policy_kwargs", {}).get("lstm_hidden_size", 128)),
    }