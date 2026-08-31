"""
--- Transformer-based Portfolio Allocator Agent (PPO + SB3) ---

Implements a multi-asset portfolio allocator trained via Stable-Baselines3 PPO.
The allocator uses a Transformer encoder to process per-asset signals from frozen SAAs
(Single-Asset Agents) combined with portfolio-level features.

Architecture:
- Input: 
    1. Per-asset tokens: raw market features, the injected SAA signal, and per-asset weight
    2. Global portfolio token
- Embedding: Linear projection → d_model dimensions
- Transformer Encoder: Self-attention across N+1 tokens (N assets tokens + 1 portfolio token)
- Output Heads: Per-asset N raw allocation logits, also derived from portfolio token
    Environment houses fixed cash logit (=0) & applies post-policy normalization to valid portfolio weights)
- Value Head: Portfolio token → scalar value estimate
- PPO Training: Standard (non-recurrent) PPO

Frozen SAAs provide signal generation only; allocator learns to weight and combine signals.
Environment is EXECUTION_PORTFOLIO mode (full multi-asset rebalancing at each step).

Config-driven: All hyperparameters loaded from JSON (transformer architecture, PPO settings, rewards).
"""

import math
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

import copy
from typing import Callable, Dict, Any, Optional, Tuple, List, Sequence, Mapping
from datetime import datetime

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common.distributions import Normal
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    VecNormalize, DummyVecEnv, SubprocVecEnv, VecEnvWrapper, VecEnv, sync_envs_normalization
)
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym

from environment.trading_environment import TradingEnv
from environment.trading_environment import MarketDataCache


# ================================
# VecNormalize Utilities for Observation Normalization
# ================================
class _ObsNormDummyEnv(gym.Env):
    """
    Minimal env needed so VecNormalize.load(...) can reattach to an env and expose obs_rms.
    We never step it for real; we only use the loaded running stats to normalize observations.
    """
    metadata = {}

    def __init__(self, observation_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        # Not used, but required by gym.Env API:
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def _normalize_obs_with_vecnormalize(obs: np.ndarray, vecnorm: VecNormalize) -> np.ndarray:
    """
    Standalone equivalent of VecNormalize.normalize_obs(obs) using saved obs_rms.
    Works for 1D obs vectors (the SAA case here).
    """
    if vecnorm is None or getattr(vecnorm, "obs_rms", None) is None:
        return obs

    obs = np.asarray(obs, dtype=np.float32)
    mean = vecnorm.obs_rms.mean
    var = vecnorm.obs_rms.var

    epsilon = float(getattr(vecnorm, "epsilon", getattr(vecnorm, "eps", 1e-8)))
    clip_obs = float(getattr(vecnorm, "clip_obs", 10.0))

    obs_norm = (obs - mean) / np.sqrt(var + epsilon)
    obs_norm = np.clip(obs_norm, -clip_obs, clip_obs)
    return obs_norm.astype(np.float32, copy=False)


# ================================
# Callbacks and Logging
# ================================

class AllocatorPortfolioLoggerCallback(BaseCallback):
    """
    Aggregates allocator portfolio metrics from every parallel env and emits their means
    to TensorBoard once per rollout, matching SB3's own train/* logging cadence so both
    appear at the same x-axis density/position in TensorBoard.
    """

    _METRIC_KEYS = (
        "portfolio_final_value",
        "comparison_final_value",
        "benchmark_final_value",
        "portfolio_return",
        "terminal_pnl_abs",
        "episode_sharpe",
        "episode_max_drawdown",
        "episode_volatility",
        "total_turnover",
        "avg_turnover",
        "total_transaction_costs",
        "episode_cost_commission",
        "episode_cost_spread",
        "episode_cost_impact",
        "exposure_start",
        "exposure_avg",
        "exposure_end",
        "shadow_return",
        "weights_mean",
        "weights_max",
        "weights_min",
        "weights_median",
        "cumulative_reward",
        "sortino_mean_ema",
        "sortino_downside_ema",
        "alpha_return",
        "excess_return_over_spy_abs",
        "excess_return_over_spy_pct",
    )

    def __init__(self, tag_prefix: str = "train", verbose: int = 0):
        """
        Args:
            tag_prefix: TensorBoard metric prefix (e.g., "train")
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)

        self.tag_prefix = tag_prefix
        self.episode_count = 0
        self.rollout_count = 0
        self._buffers: Dict[str, List[float]] = {k: [] for k in self._METRIC_KEYS}

    def _on_step(self) -> bool:
        """Buffer metrics from every env that finished an episode on this vector step."""
        for info in self.locals.get("infos", []):
            if not info.get("episode_final", False):
                continue
            self.episode_count += 1
            for key in self._METRIC_KEYS:
                value = info.get(key, None)
                if value is not None:
                    self._buffers[key].append(float(value))

            pv = info.get("portfolio_final_value", None)
            comp_pv = info.get("comparison_final_value", None)
            if pv is not None and comp_pv is not None:
                self._buffers.setdefault("return_diff_vs_init", []).append(float(pv) - float(comp_pv))

        return True

    def _on_rollout_end(self) -> bool:
        # Flush unconditionally every rollout; SB3 logs train/policy_gradient_loss etc.
        # at the same cadence, so throttling here would desync the two on the x-axis.
        self.rollout_count += 1

        for key, values in self._buffers.items():
            if values:
                self.model.logger.record(f"{self.tag_prefix}/{key}", float(np.mean(values)), exclude=("stdout",))

        self.model.logger.record(f"{self.tag_prefix}/episodes_completed", int(self.episode_count), exclude=("stdout",))
        for values in self._buffers.values():
            values.clear()
        return True


class AllocatorValidationCallback(BaseCallback):
    """
    Accumulates per-episode validation metrics across the deterministic sweep and emits
    aggregated statistics to TensorBoard.

    Two-phase design:
    - collect_info(): called once per finished validation episode
    - flush_metrics(): computes aggregates, logs them, and clears the buffers
    """

    _MEAN_STD_KEYS = (
        "portfolio_final_value",
        "portfolio_return",
        "terminal_pnl_abs",
        "excess_return_over_spy_abs",
        "excess_return_over_spy_pct",
        "spy_bh_final_value",
        "benchmark_final_value",
        "comparison_final_value",
        "episode_sharpe",
        "episode_max_drawdown",
        "alpha_return",
        "cumulative_reward",
        "episode_volatility",
        "total_turnover",
        "total_transaction_costs",
    )

    def __init__(self, tag_prefix: str = "validation", verbose: int = 0):
        super().__init__(verbose)

        self.tag_prefix = tag_prefix
        self.eval_episode_count = 0
        self._buffers: Dict[str, List[float]] = {k: [] for k in self._MEAN_STD_KEYS}
        self._per_block: List[Tuple[str, float, float]] = []  # (block_id, terminal_pnl, excess_over_spy)

        # Values consumed by AllocatorEvalCallback for checkpointing / early stopping.
        self.last_mean_sharpe: float = -np.inf
        self.last_excess_over_spy_abs_mean: float = -np.inf
        self.last_terminal_pnl_mean: float = -np.inf
        self.last_terminal_pnl_min: float = -np.inf

    def collect_info(self, info: Mapping[str, Any]) -> None:
        """Record one finished validation episode."""
        self.eval_episode_count += 1
        for key in self._MEAN_STD_KEYS:
            value = info.get(key, None)
            if value is not None:
                self._buffers[key].append(float(value))

        self._per_block.append(
            (
                str(info.get("block_id", f"ep_{self.eval_episode_count}")),
                float(info.get("terminal_pnl_abs", np.nan)),
                float(info.get("excess_return_over_spy_abs", np.nan)),
            )
        )

    def _on_step(self) -> bool:
        return True

    def flush_metrics(self, n_expected_episodes: int) -> bool:
        """
        Compute and log aggregated validation statistics.

        Returns True when the sweep was complete and the aggregates (including the
        checkpoint metrics) are valid.
        """
        if self.eval_episode_count < n_expected_episodes:
            print(
                f"[AllocatorValidationCallback] Incomplete sweep: "
                f"{self.eval_episode_count}/{n_expected_episodes} episodes - metrics not logged"
            )
            self._reset_buffers()
            return False

        for key, values in self._buffers.items():
            if not values:
                continue
            self.model.logger.record(f"{self.tag_prefix}/{key}_mean", float(np.mean(values)), exclude=("stdout",))
            self.model.logger.record(f"{self.tag_prefix}/{key}_std", float(np.std(values)), exclude=("stdout",))

        # Return difference vs the buy-and-hold of the initial allocation
        pv = self._buffers["portfolio_final_value"]
        comp = self._buffers["comparison_final_value"]
        if pv and comp and len(pv) == len(comp):
            diffs = [p - c for p, c in zip(pv, comp)]
            self.model.logger.record(f"{self.tag_prefix}/return_diff_vs_init_mean", float(np.mean(diffs)), exclude=("stdout",))

        # Worst-case block statistics drive the min-based checkpoint.
        pnl_values = self._buffers["terminal_pnl_abs"]
        excess_values = self._buffers["excess_return_over_spy_abs"]

        self.last_mean_sharpe = float(np.mean(self._buffers["episode_sharpe"])) if self._buffers["episode_sharpe"] else -np.inf
        self.last_terminal_pnl_mean = float(np.mean(pnl_values)) if pnl_values else -np.inf
        self.last_terminal_pnl_min = float(np.min(pnl_values)) if pnl_values else -np.inf
        self.last_excess_over_spy_abs_mean = float(np.mean(excess_values)) if excess_values else -np.inf

        if pnl_values:
            self.model.logger.record(f"{self.tag_prefix}/terminal_pnl_abs_min", self.last_terminal_pnl_min, exclude=("stdout",))
            self.model.logger.record(f"{self.tag_prefix}/terminal_pnl_abs_max", float(np.max(pnl_values)), exclude=("stdout",))
        if excess_values:
            self.model.logger.record(
                f"{self.tag_prefix}/excess_return_over_spy_abs_min", float(np.min(excess_values)), exclude=("stdout",)
            )

        # Per-block scalars so a single degrading block stays visible behind the means.
        for block_id, block_pnl, block_excess in self._per_block:
            if np.isfinite(block_pnl):
                self.model.logger.record(f"{self.tag_prefix}/block_{block_id}/terminal_pnl_abs", block_pnl, exclude=("stdout",))
            if np.isfinite(block_excess):
                self.model.logger.record(
                    f"{self.tag_prefix}/block_{block_id}/excess_return_over_spy_abs", block_excess, exclude=("stdout",)
                )

        self._reset_buffers()
        return True

    def _reset_buffers(self) -> None:
        for values in self._buffers.values():
            values.clear()
        self._per_block = []
        self.eval_episode_count = 0


class AllocatorEvalCallback(BaseCallback):
    """
    Periodic deterministic validation for the allocator.

    Every `eval_freq` calls it runs exactly one episode per validation block, each spanning
    the block in full and starting from 100% cash, then logs metrics and saves checkpoints on:
    - highest mean excess return over SPY buy-and-hold (absolute)
    - highest mean terminal PnL (absolute)
    - highest worst-block terminal PnL (absolute)
    """

    def __init__(
        self,
        eval_env: VecEnv,
        best_model_save_path: str,
        log_path: str,
        eval_freq: int,
        eval_step_callback: Optional["AllocatorValidationCallback"] = None,
        patience: int = 7,
        min_delta_reward: float = 0.0,
        min_delta_sharpe: float = 0.0,
        verbose: int = 0
    ):
        super().__init__(verbose)

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.eval_step_callback = eval_step_callback

        self.best_mean_reward = -np.inf
        self.best_mean_sharpe = -np.inf
        self.best_excess_over_spy_abs_mean = -np.inf
        self.best_terminal_pnl_mean = -np.inf
        self.best_terminal_pnl_min = -np.inf

        # Early stopping
        self.patience = patience
        self.min_delta_reward = min_delta_reward
        self.min_delta_sharpe = min_delta_sharpe
        self.no_improve_reward = 0
        self.no_improve_sharpe = 0

        self.n_eval_calls = 0
        self._sweep_plan: List[Dict[str, Any]] = []

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
        if self.eval_step_callback is not None:
            self.eval_step_callback.init_callback(self.model)

        # The plan is identical for every eval env, so read it once from env 0.
        self._sweep_plan = self.eval_env.env_method("get_validation_sweep_plan", indices=0)[0]
        if not self._sweep_plan:
            raise RuntimeError("Validation sweep plan is empty; no validation blocks available.")
        if self.verbose > 0:
            print(
                f"[AllocatorEvalCallback] Deterministic sweep: {len(self._sweep_plan)} validation "
                f"blocks across {self.eval_env.num_envs} eval env(s)"
            )

    def _run_validation_sweep(self) -> Tuple[List[float], List[int]]:
        """
        Run every plan in the sweep exactly once, deterministically, sharding the plans
        round-robin across the eval envs. Returns per-episode rewards and lengths.
        """
        n_envs = self.eval_env.num_envs
        shards: List[List[Dict[str, Any]]] = [self._sweep_plan[i::n_envs] for i in range(n_envs)]
        for i, shard in enumerate(shards):
            self.eval_env.env_method("set_plan_queue", shard, indices=i)

        # Envs whose shard is empty must not contribute episodes.
        remaining = np.array([len(s) for s in shards], dtype=int)
        active = remaining > 0
        remaining = np.maximum(remaining - 1, 0)  # the upcoming reset consumes one plan per env

        obs = self.eval_env.reset()
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        current_reward = np.zeros(n_envs, dtype=np.float64)
        current_length = np.zeros(n_envs, dtype=int)

        while active.any():
            actions, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.eval_env.step(actions)
            current_reward += np.asarray(rewards, dtype=np.float64) * active
            current_length += active

            for i in range(n_envs):
                if not active[i] or not dones[i]:
                    continue
                if self.eval_step_callback is not None and infos[i].get("episode_final", False):
                    self.eval_step_callback.collect_info(infos[i])
                episode_rewards.append(float(current_reward[i]))
                episode_lengths.append(int(current_length[i]))
                current_reward[i] = 0.0
                current_length[i] = 0
                if remaining[i] > 0:
                    remaining[i] -= 1
                else:
                    # Shard exhausted: the auto-reset episode is discarded.
                    active[i] = False

        return episode_rewards, episode_lengths

    def _save_checkpoint(self, checkpoint_name: str) -> None:
        """Save model and the matching VecNormalize stats at the same instant."""
        if self.best_model_save_path is None:
            return
        self.model.save(os.path.join(self.best_model_save_path, checkpoint_name))
        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize):
            vec_env.save(os.path.join(self.best_model_save_path, f"{checkpoint_name}_vecnormalize.pkl"))
        else:
            raise RuntimeError(
                f"Checkpoint '{checkpoint_name}' saved without VecNormalize stats "
                "(training env is not wrapped in VecNormalize)."
            )

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        # Keep the frozen eval normalization aligned with the training statistics.
        sync_envs_normalization(self.model.get_env(), self.eval_env)

        episode_rewards, episode_lengths = self._run_validation_sweep()
        self.n_eval_calls += 1

        sweep_complete = True
        if self.eval_step_callback is not None:
            sweep_complete = self.eval_step_callback.flush_metrics(len(self._sweep_plan))

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else -np.inf
        self.model.logger.record("eval/mean_reward", mean_reward)
        if episode_rewards:
            self.model.logger.record("eval/min_reward", float(np.min(episode_rewards)))
            self.model.logger.record("eval/max_reward", float(np.max(episode_rewards)))
            self.model.logger.record("eval/std_reward", float(np.std(episode_rewards)))
            self.model.logger.record("eval/mean_ep_length", float(np.mean(episode_lengths)))
            self.model.logger.record("eval/n_episodes", int(len(episode_rewards)))

        if self.log_path is not None and episode_rewards:
            self._append_eval_log(episode_rewards, episode_lengths)

        if not sweep_complete:
            return True

        # --- Checkpoints on the three validation metrics ---
        vcb = self.eval_step_callback
        if vcb is not None:
            if vcb.last_excess_over_spy_abs_mean > self.best_excess_over_spy_abs_mean:
                self.best_excess_over_spy_abs_mean = vcb.last_excess_over_spy_abs_mean
                self._save_checkpoint("best_model_excess_over_spy_abs")
                print(
                    "[AllocatorEvalCallback] New best mean excess over SPY: "
                    f"{self.best_excess_over_spy_abs_mean:.2f}"
                )

            if vcb.last_terminal_pnl_mean > self.best_terminal_pnl_mean:
                self.best_terminal_pnl_mean = vcb.last_terminal_pnl_mean
                self._save_checkpoint("best_model_terminal_pnl_mean")
                print(f"[AllocatorEvalCallback] New best mean terminal PnL: {self.best_terminal_pnl_mean:.2f}")

            if vcb.last_terminal_pnl_min > self.best_terminal_pnl_min:
                self.best_terminal_pnl_min = vcb.last_terminal_pnl_min
                self._save_checkpoint("best_model_terminal_pnl_min")
                print(f"[AllocatorEvalCallback] New best worst-block terminal PnL: {self.best_terminal_pnl_min:.2f}")

        # --- Reward-based checkpoint (kept for continuity) + early stopping ---
        previous_best_reward = self.best_mean_reward
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self._save_checkpoint("best_model")
            if self.verbose > 0:
                print(
                    f"[AllocatorEvalCallback] New best mean reward: {mean_reward:.3f} "
                    f"(previous: {previous_best_reward:.3f})"
                )

        if mean_reward > previous_best_reward + self.min_delta_reward:
            self.no_improve_reward = 0
        else:
            self.no_improve_reward += 1
            print(
                f"[AllocatorEvalCallback] Reward has not improved for "
                f"{self.no_improve_reward}/{self.patience} eval calls "
                f"(current={mean_reward:.4f}, best={self.best_mean_reward:.4f})"
            )

        mean_sharpe = vcb.last_mean_sharpe if vcb is not None else -np.inf
        if mean_sharpe > self.best_mean_sharpe + self.min_delta_sharpe:
            self.best_mean_sharpe = mean_sharpe
            self.no_improve_sharpe = 0
        else:
            self.no_improve_sharpe += 1
            if mean_sharpe == -np.inf:
                print(
                    "[AllocatorEvalCallback] WARNING: Sharpe buffer was empty this eval - "
                    "env may not be emitting Sharpe in info dict. "
                    "Sharpe early stopping condition is inactive."
                )
            else:
                print(
                    f"[AllocatorEvalCallback] Sharpe has not improved for "
                    f"{self.no_improve_sharpe}/{self.patience} eval calls "
                    f"(current={mean_sharpe:.4f}, best={self.best_mean_sharpe:.4f})"
                )

        sharpe_available = mean_sharpe > -np.inf
        if (
            self.no_improve_reward >= self.patience
            and self.no_improve_sharpe >= self.patience
            and sharpe_available
        ):
            print(
                f"[AllocatorEvalCallback] Early stopping triggered: "
                f"neither mean reward ({mean_reward:.4f}) nor mean Sharpe ({mean_sharpe:.4f}) "
                f"have improved for {self.patience} consecutive eval calls. "
                f"Best reward={self.best_mean_reward:.4f}, best Sharpe={self.best_mean_sharpe:.4f}."
            )
            return False

        return True

    def _append_eval_log(self, episode_rewards: List[float], episode_lengths: List[int]) -> None:
        eval_log_path = os.path.join(self.log_path, "evaluations.npz")
        if os.path.exists(eval_log_path):
            try:
                existing = np.load(eval_log_path)
                timesteps = np.append(existing["timesteps"], self.num_timesteps)
                results = np.append(existing["results"], [episode_rewards], axis=0)
                ep_lengths = np.append(existing["ep_lengths"], [episode_lengths], axis=0)
            except Exception:
                timesteps = np.array([self.num_timesteps])
                results = np.array([episode_rewards])
                ep_lengths = np.array([episode_lengths])
        else:
            timesteps = np.array([self.num_timesteps])
            results = np.array([episode_rewards])
            ep_lengths = np.array([episode_lengths])

        np.savez(eval_log_path, timesteps=timesteps, results=results, ep_lengths=ep_lengths)


# Three-Phase Linear schedule to be used with learning rate and entropy coefficient
def linear_three_phase_schedule(start: float, end: float, warmup_pct: float, ramping_pct: float) -> Callable[[float], float]:
    """
    Three-phase linear schedule for hyperparameter annealing.
    
    Creates a callable schedule function that transitions through three phases:
    - Warmup (0 to warmup_pct): Holds constant at start value
    - Ramping (warmup_pct to ramping_pct): Linear interpolation from start to end
    - Hold (ramping_pct to 1.0): Holds constant at end value
    
    Args:
        start: Initial value (used during warmup)
        end: Final value (used during hold phase)
        warmup_pct: Fraction of training for warmup phase [0, 1]
        ramping_pct: Fraction where ramping completes [warmup_pct, 1]
    
    Returns:
        Callable schedule function that takes progress_remaining ∈ [1.0, 0.0]
        and returns current hyperparameter value
        
    Notes:
        - SB3 schedules receive "progress_remaining" where:
          * 1.0 = start of training
          * 0.0 = end of training
        - We convert to "progress_elapsed" = 1.0 - progress_remaining for intuitive config
        
    """
    # Clamp percentages to valid ranges
    warmup_pct = float(np.clip(warmup_pct, 0.0, 1.0))
    ramping_pct = float(np.clip(ramping_pct, warmup_pct, 1.0))
    
    def schedule(progress_remaining: float) -> float:
        """
        Compute current hyperparameter value based on training progress.
        
        Args:
            progress_remaining: SB3 progress indicator ∈ [1.0, 0.0]
                               1.0 = start of training, 0.0 = end
        
        Returns:
            Current hyperparameter value
        """
        # Convert to elapsed progress for intuitive reasoning
        progress_elapsed = 1.0 - float(progress_remaining)
        
        # Phase 1: Warmup (constant at start value)
        if progress_elapsed <= warmup_pct:
            return float(start)
        
        # Phase 2: Ramping (linear interpolation)
        if progress_elapsed <= ramping_pct:
            # Compute fraction through ramping phase
            phase_length = max(ramping_pct - warmup_pct, 1e-8)  # Avoid division by zero
            phase_progress = (progress_elapsed - warmup_pct) / phase_length
            
            # Linear interpolation between start and end
            return float(start + (end - start) * phase_progress)
        
        # Phase 3: Hold (constant at end value)
        return float(end)
    
    return schedule

# Callback to update entropy coefficient during training
class EntropyScheduleCallback(BaseCallback):
    """
    Callback for scheduling entropy coefficient during training.
    
    SB3 does not support callable schedules for ent_coef (unlike learning_rate),
    so we use a callback to update it manually each rollout.
    
    Integration:
    - Inherits from BaseCallback for SB3 compatibility
    - Uses _on_rollout_end() to update ent_coef after each rollout
    - Accesses model._current_progress_remaining provided by SB3
    
    Usage:
        ent_callback = EntropyScheduleCallback(
            start=0.01,
            end=0.001,
            warmup_pct=0.2,
            ramping_pct=0.6
        )
        model.learn(total_timesteps=1000000, callback=[ent_callback, ...])
    """
    
    def __init__(self, start: float, end: float, warmup_pct: float, ramping_pct: float, verbose: int = 0):
        """
        Initialize entropy schedule callback.
        
        Args:
            start: Initial entropy coefficient
            end: Final entropy coefficient
            warmup_pct: Warmup phase duration (fraction of training)
            ramping_pct: Ramping completion point (fraction of training)
            verbose: Logging verbosity (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        
        # Create schedule function using three-phase linear interpolation
        self._schedule = linear_three_phase_schedule(start, end, warmup_pct, ramping_pct)
        
        # Store parameters for logging
        self._start = start
        self._end = end
        self._warmup_pct = warmup_pct
        self._ramping_pct = ramping_pct
    
    def _on_rollout_end(self) -> bool:
        """
        Update entropy coefficient at end of each rollout.
        
        Called by SB3 after collecting n_steps of experience but before
        performing gradient updates.
        
        Returns:
            True to continue training, False to stop
        """
        # Get current training progress from model
        # SB3 updates this attribute automatically during training
        progress_remaining = getattr(self.model, "_current_progress_remaining", 0.0)
        
        # Compute new entropy coefficient using schedule
        new_ent_coef = float(self._schedule(progress_remaining))
        
        # Update model's entropy coefficient
        # SB3 expects a scalar float, not a callable
        self.model.ent_coef = new_ent_coef
        
        # Optional: Log entropy coefficient changes
        if self.verbose > 1:
            progress_elapsed = 1.0 - progress_remaining
            print(f"[EntropyScheduleCallback] Progress: {progress_elapsed:.3f}, ent_coef: {new_ent_coef:.6f}")
        
        # Continue training
        return True
    
    def _on_step(self) -> bool:
        """
        Called after each environment step (required by BaseCallback).
        
        We don't need per-step updates for entropy scheduling,
        only per-rollout updates in _on_rollout_end().
        
        Returns:
            True to continue training
        """
        return True
    

class SAASignalWrapper(VecEnvWrapper):
    """
    Runs the frozen SAA over every (env, asset) pair and injects its signal into the
    allocator's observations.

    Row layout of the batched SAA forward pass is r = env_index * num_assets + asset_index.
    That mapping must stay fixed for the whole run: the LSTM state carried in
    self.saa_state is indexed by r, so reordering rows would cross-contaminate the
    per-asset recurrent memory.
    """
    def __init__(self, venv: VecEnv, saa_model, saa_vecnormalize: Optional[VecNormalize],
                num_assets: int, device: torch.device,
                config: Mapping[str, Any], feature_to_index: Mapping[str, int],
                action_limiting_factor: float):
        
        super().__init__(venv)

        self.saa_vecnormalize = saa_vecnormalize
        self.num_assets = num_assets
        self.device = device
        # keep config and feature mapping locally (DummyVecEnv has no .config)
        self.config = config
        self.feature_to_index = feature_to_index
        # Rescales raw SAA policy output into the target_position_change range the SAA was
        # actually trained/executed with (env.step never saw raw actions during SAA training).
        self.action_limiting_factor = float(action_limiting_factor)
        # Precomputed one-hot asset-ID rows; SAA's InputMLPFeatures expects this trailing block.
        self._asset_one_hot = np.eye(self.num_assets, dtype=np.float32)

        if not hasattr(self.venv, "env_method"):
            raise RuntimeError(
                "SAASignalWrapper requires a VecEnv exposing env_method(); place it directly "
                "around the raw DummyVecEnv/SubprocVecEnv (no VecNormalize in between)."
            )

        # Infer dimensions from selected market features and env observation shape.
        # The raw env emits: N * raw_feat_dim + portfolio_dim (before SAA signal injection).
        obs_len = int(self.observation_space.shape[0])
        self.raw_feat_dim = int(len(self.feature_to_index))
        if self.raw_feat_dim <= 0:
            raise ValueError(
                "Cannot infer raw feature dimension from feature_to_index. "
                f"len(feature_to_index)={self.raw_feat_dim}"
            )

        asset_block = self.num_assets * self.raw_feat_dim
        self.portfolio_dim = obs_len - asset_block
        if self.portfolio_dim <= 0:
            raise ValueError(
                "Invalid observation layout for SAASignalWrapper. "
                f"obs_len={obs_len}, num_assets={self.num_assets}, "
                f"raw_feat_dim={self.raw_feat_dim}, portfolio_dim={self.portfolio_dim}"
            )

        # Cache SAA feature indices once (used every step).
        self.saa_idx = np.array(
            [self.feature_to_index[f] for f, on in self.config["saa_features"].items() if on],
            dtype=int,
        )
        if self.saa_idx.size == 0:
            raise ValueError("No enabled SAA features found in config['saa_features'].")
        if np.min(self.saa_idx) < 0 or np.max(self.saa_idx) >= self.raw_feat_dim:
            raise ValueError(
                "SAA feature indices out of range for raw feature block. "
                f"raw_feat_dim={self.raw_feat_dim}, min_idx={int(np.min(self.saa_idx))}, "
                f"max_idx={int(np.max(self.saa_idx))}"
            )

        # One frozen SAA shared by every (env, asset) row; the LSTM state carries the batch
        # dimension, so separate model copies would be redundant.
        self.saa_model = saa_model
        self.saa_model.policy.to(self.device)
        self.saa_model.device = self.device
        self.saa_model.policy.eval()

        self.n_rows = int(venv.num_envs) * self.num_assets
        self.saa_state: Optional[Any] = None
        self.episode_start = np.ones(self.n_rows, dtype=bool)
        self._last_signals = np.zeros((venv.num_envs, self.num_assets, 1), dtype=np.float32)
        self._last_shadow_weights = np.zeros((venv.num_envs, self.num_assets, 1), dtype=np.float32)
        # Row-aligned one-hot asset IDs, tiled once for the whole batch.
        self._asset_one_hot_batch = np.tile(self._asset_one_hot, (venv.num_envs, 1))

        self.initial_portfolio_value = float(self.venv.get_attr("initial_portfolio_value")[0])

        # Resize obs space: add +2 features per asset (SAA signal + shadow-portfolio holding %)
        old_low, old_high = self.observation_space.low, self.observation_space.high
        asset_size = self.num_assets * self.raw_feat_dim
        low_assets = old_low[:asset_size].reshape(self.num_assets, self.raw_feat_dim)
        high_assets = old_high[:asset_size].reshape(self.num_assets, self.raw_feat_dim)
        low_assets = np.concatenate([low_assets, np.full((self.num_assets, 2), -np.inf, dtype=np.float32)], axis=1)
        high_assets = np.concatenate([high_assets, np.full((self.num_assets, 2), np.inf, dtype=np.float32)], axis=1)
        new_low_assets = low_assets.reshape(-1)
        new_high_assets = high_assets.reshape(-1)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([new_low_assets, old_low[asset_size:]]),
            high=np.concatenate([new_high_assets, old_high[asset_size:]]),
            dtype=np.float32,
        )

    def reset(self):
        res = self.venv.reset()
        if isinstance(res, tuple) and len(res) == 2:
            obs, _info = res
        else:
            obs = res
        # Drop recurrent memory: every (env, asset) row starts a fresh episode.
        self.saa_state = None
        self.episode_start[:] = True
        signals, shadow_weights = self._compute_saa_signals(obs)
        self._commit_saa_actions(signals)
        return self._inject_signals(obs, signals, shadow_weights)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        dones_arr = np.asarray(dones, dtype=bool)  # shape (B,)

        # Terminal observations belong to the finished episode; augment them with the last
        # known signal only - no LSTM advance, no sub-portfolio commit.
        for i, info in enumerate(infos):
            if info.get("terminal_observation", None) is not None:
                info["terminal_observation"] = self._inject_signals(
                    info["terminal_observation"][None, ...],
                    self._last_signals[i][None, ...],
                    self._last_shadow_weights[i][None, ...],
                )[0]

        # VecEnv auto-resets, so the obs returned here is already the next episode's first
        # observation for any env that reported done. Flag those rows accordingly.
        self.episode_start = np.repeat(dones_arr, self.num_assets)

        signals, shadow_weights = self._compute_saa_signals(obs)
        self._commit_saa_actions(signals)
        return self._inject_signals(obs, signals, shadow_weights), rewards, dones, infos

    def _compute_saa_signals(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rebuild the SAA training observation for every (env, asset) row and run a single
        batched forward pass. Returns (signals, shadow_weights), each shaped (B, N, 1).
        Advances the LSTM state.
        """
        B = obs.shape[0]
        N = self.num_assets
        asset_block = N * self.raw_feat_dim
        asset_feats = obs[:, :asset_block].reshape(B, N, self.raw_feat_dim)

        # Selected SAA market features per (env, asset).
        saa_market_feats = asset_feats[:, :, self.saa_idx]  # (B, N, F_saa)

        # --- Per-asset hypothetical sub-portfolio state, one RPC round trip ---
        bundles = self.venv.env_method("get_saa_signal_inputs")
        cash_all = np.stack([b["cash"] for b in bundles], axis=0)                       # (B, N)
        shares_all = np.stack([b["shares"] for b in bundles], axis=0)                   # (B, N)
        last_act_all = np.stack([b["last_action"] for b in bundles], axis=0)            # (B, N)
        dret_all = np.stack([b["daily_return"] for b in bundles], axis=0)               # (B, N)
        prices_all = np.stack([b["prices"] for b in bundles], axis=0)                   # (B, N)
        rf_z_all = np.asarray([b["rf_zscore"] for b in bundles], dtype=np.float32)      # (B,)
        alpha_rf_all = np.stack([b["excess_log_return_over_rf"] for b in bundles], axis=0)  # (B, N)

        asset_notional = shares_all * prices_all
        # Log-ratios with gating at 0 (matches get_observation_single_step L3973-3976)
        eps = 1e-12
        initial_pv = self.initial_portfolio_value
        cash_log_value = np.where(cash_all > 0, np.log(np.maximum(cash_all, eps) / initial_pv), 0.0).astype(np.float32)
        asset_log_value = np.where(asset_notional > 0, np.log(np.maximum(asset_notional, eps) / initial_pv), 0.0).astype(np.float32)

        # Shadow sub-portfolio holding percentage: this asset's share of ITS OWN isolated shadow
        # book (cash_all + asset_notional). Never derived from, or written into, the live PAA
        # portfolio state - only this scalar crosses the shadow/live boundary.
        shadow_sub_total = cash_all + asset_notional
        shadow_weight_all = np.where(
            shadow_sub_total > eps, asset_notional / np.maximum(shadow_sub_total, eps), 0.0
        ).astype(np.float32)

        rf_z_rep = np.repeat(rf_z_all[:, None], repeats=N, axis=1)
        mem_block = np.stack(
            [cash_log_value, asset_log_value, dret_all, last_act_all, rf_z_rep, alpha_rf_all], axis=-1
        )  # (B, N, 6)

        # Flatten to rows r = b * N + a and append the trailing one-hot asset-ID block so the
        # layout matches SAA training: [features, portfolio_features(6), one_hot_asset_id(N)].
        saa_obs = np.concatenate([saa_market_feats, mem_block], axis=-1).reshape(B * N, -1)
        one_hot = self._asset_one_hot_batch if B == self.venv.num_envs else np.tile(self._asset_one_hot, (B, 1))
        batch_obs = np.concatenate([saa_obs, one_hot], axis=-1).astype(np.float32)
        batch_obs = _normalize_obs_with_vecnormalize(batch_obs, self.saa_vecnormalize)

        # SB3 recurrent policies expect NumPy observations; a CUDA tensor would break
        # policy.obs_to_tensor's internal np.array() conversion.
        actions, state_out = self.saa_model.policy.predict(
            batch_obs,
            state=self.saa_state,
            episode_start=np.asarray(self.episode_start[: B * N], dtype=bool),
            deterministic=True,
        )
        self.saa_state = state_out

        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions)
        raw_signal = np.clip(actions_np[:, 0:1], -1.0, 1.0)
        # Rescale to the target_position_change range actually seen by env.step() during
        # SAA training/inference (see SingleAssetEpisodeAdapter.step() action_factor_fn).
        signals = (raw_signal * self.action_limiting_factor).reshape(B, N, 1).astype(np.float32)
        shadow_weights = shadow_weight_all.reshape(B, N, 1).astype(np.float32)
        self._last_signals = signals
        self._last_shadow_weights = shadow_weights
        return signals, shadow_weights

    def _commit_saa_actions(self, signals: np.ndarray) -> None:
        """Apply the SAA actions to each env's sub-portfolio so the next step marks to market."""
        for b in range(signals.shape[0]):
            self.venv.env_method("apply_saa_sub_actions", signals[b, :, 0], indices=b)

    def _inject_signals(self, obs: np.ndarray, signals: np.ndarray, shadow_weights: np.ndarray) -> np.ndarray:
        """
        Input obs: (B, num_assets*raw_feat_dim + portfolio_dim)
        Output obs: asset part becomes num_assets*(raw_feat_dim+2), with the SAA signal and the
        shadow sub-portfolio's own holding percentage appended per asset.
        """
        B = obs.shape[0]
        asset_block = self.num_assets * self.raw_feat_dim
        asset_feats = obs[:, :asset_block].reshape(B, self.num_assets, self.raw_feat_dim)
        portfolio_part = obs[:, asset_block:]
        augmented_assets = np.concatenate([asset_feats, signals, shadow_weights], axis=-1).reshape(B, -1)
        return np.concatenate([augmented_assets, portfolio_part], axis=1)


class PortfolioEpisodeAdapter(gym.Wrapper):
    """
    Lets the eval callback drive deterministic validation episodes through a VecEnv.

    Each env is handed a queue of episode plans; every reset() consumes the next one and
    forwards it as reset options. Once the queue is empty the env falls back to normal
    sampling, and `plans_remaining` reports 0 so the caller can stop collecting.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._plan_queue: List[Dict[str, Any]] = []

    def get_validation_sweep_plan(self) -> List[Dict[str, Any]]:
        """One episode per validation block, spanning the block in full, starting from cash."""
        cache = self.env.market_data_cache
        blocks = sorted(cache.validation_blocks, key=lambda b: b.start_date_idx)
        plans: List[Dict[str, Any]] = []
        for block in blocks:
            start_step = int(block.min_start_step)
            episode_length = int(block.end_date_idx - start_step)
            plans.append(
                {
                    "block_id": str(block.block_id),
                    "episode_start_step": start_step,
                    "episode_length_override": episode_length,
                    "force_cash_only_start": True,
                }
            )
        return plans

    def set_plan_queue(self, plans: Sequence[Dict[str, Any]]) -> None:
        self._plan_queue = [dict(p) for p in plans]

    @property
    def plans_remaining(self) -> int:
        return len(self._plan_queue)

    def get_plans_remaining(self) -> int:
        return len(self._plan_queue)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        reset_options = dict(options) if options is not None else {}
        if self._plan_queue:
            reset_options.update(self._plan_queue.pop(0))
        return self.env.reset(seed=seed, option=reset_options if reset_options else None)

    def step(self, action):
        return self.env.step(action)


class AttentionEngine(nn.Module):
    def __init__(self, feature_dim: int, n_assets: int, d_model: int, n_heads: int, 
                n_layers: int, dim_feedforward: int, transformer_encoder_dropout: float, 
                transformer_activation_fn: str = "relu"):
        
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.n_layers = int(n_layers)

        # 1. Shared Transformer Encoder (N layers): builds cross-asset context for both heads.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,  # Use provided feedforward dimension
            batch_first=True,  # Use batch_first for easier integration with SB3
            dropout=transformer_encoder_dropout,
            activation=transformer_activation_fn,
            norm_first=True  # Pre-LN often stabilizes training in RL contexts
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 2. Private final layers (one per head).
        #    Each head owns one TransformerEncoderLayer that only ever receives
        #    gradients from its own loss, absorbing diverging actor/critic gradient
        #    signals before they reach the shared base.
        #    Actor layer : processes asset tokens [0..N-1] then mean-pools → latent_pi
        #    Critic layer: processes portfolio token [-1] only          → latent_vf
        def _make_head_layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=transformer_encoder_dropout,
                activation=transformer_activation_fn,
                norm_first=True,
            )
        self.actor_final_layer = _make_head_layer()
        self.critic_final_layer = _make_head_layer()

        self._init_transformer_weights()

        # 3. SB3 Dimension Requirements
        self.latent_dim_pi = d_model  # actor: mean-pooled asset tokens → d_model
        self.latent_dim_vf = d_model  # critic: portfolio [CLS] token   → d_model

    def _init_transformer_weights(self) -> None:
        residual_gain = 1.0 / math.sqrt(2.0 * max(1, self.n_layers))

        # Xavier init + zero biases for all three modules.
        # Private head layers use standard Xavier (single layer, no residual scaling needed).
        for mod in [self.transformer, self.actor_final_layer, self.critic_final_layer]:
            for module in mod.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            # MultiheadAttention QKV packed projections are raw Parameters, not nn.Linear.
            for name, param in mod.named_parameters():
                if "in_proj_weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "in_proj_bias" in name:
                    nn.init.zeros_(param)

        # GPT-2 style residual branch scaling for the deep shared stack only.
        for name, param in self.transformer.named_parameters():
            if param.ndim >= 2 and ("out_proj.weight" in name or "linear2.weight" in name):
                nn.init.xavier_uniform_(param, gain=residual_gain)

    def forward(self, features: torch.Tensor):
        batch_size = features.shape[0]

        # 1. Reshape to (Batch, Sequence, Features)
        # Sequence length = N_assets + 1 (portfolio token)
        x = features.view(batch_size, self.n_assets + 1, self.d_model)

        # 2. Shared cross-asset attention pass (all tokens attend to each other)
        attended = self.transformer(x)

        # 3. Private actor path: asset tokens [0..N-1] → actor_final_layer → mean-pool
        #    Gradient from L_policy enters the shared base through asset token positions.
        asset_tokens = attended[:, :self.n_assets, :]            # (B, N, d_model)
        actor_out = self.actor_final_layer(asset_tokens)          # (B, N, d_model)
        latent_pi = actor_out.mean(dim=1)                         # (B, d_model)

        # 4. Private critic path: portfolio token [-1] → critic_final_layer → squeeze
        #    Gradient from L_value enters the shared base through portfolio token position.
        portfolio_token = attended[:, -1:, :]                     # (B, 1, d_model)
        critic_out = self.critic_final_layer(portfolio_token)     # (B, 1, d_model)
        latent_vf = critic_out.squeeze(1)                         # (B, d_model)

        return latent_pi, latent_vf

    # SB3 expects these helpers (mirrors MlpExtractor API)
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = self.forward(features)
        return latent_pi

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        _, latent_vf = self.forward(features)
        return latent_vf


class TransformerAllocatorPolicy(ActorCriticPolicy):
    """
    Custom policy that swaps the default MlpExtractor with AttentionEngine.
    Expects features_dim = (n_assets + 1) * d_model from the tokenizer.
    """
    def __init__(self, observation_space, action_space, lr_schedule, n_assets, d_model, n_heads, n_layers, dim_feedforward, transformer_encoder_dropout,
                 transformer_activation_fn, **kwargs):
        
        self._n_assets = n_assets
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._dim_feedforward = dim_feedforward
        self.transformer_encoder_dropout = transformer_encoder_dropout
        self.transformer_activation_fn = transformer_activation_fn
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        # Replace with attention engine; note AttentionEngine.latent_dim_pi/vf attributes
        self.mlp_extractor = AttentionEngine(
            feature_dim=self.features_dim,
            n_assets=self._n_assets,
            d_model=self._d_model,
            n_heads=self._n_heads,
            n_layers=self._n_layers,
            dim_feedforward=self._dim_feedforward,
            transformer_encoder_dropout=self.transformer_encoder_dropout,
            transformer_activation_fn=self.transformer_activation_fn
        )


class SAATokenizer(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        num_assets: int,
        raw_feat_dim: int,
        d_model: int,
        asset_feature_idx: Sequence[int],
        portfolio_time_idx: Sequence[int],
    ):
        """
        Builds asset and portfolio tokens from the augmented observation.

        Observation (after SAASignalWrapper):
            - Asset block: N * (raw_feat_dim + 2)  where the +2 are the injected SAA signal
              and the shadow sub-portfolio's own holding percentage
            - Portfolio block: remaining dims (portfolio_dim)

        Asset token:
            selected asset features (len(asset_feature_idx)) + SAA signal (1)
            + shadow-portfolio holding percentage (1) + live PAA asset_weight (1)
            + last target weight (1) + shadow Sortino (1) + shadow drawdown (1)
            => expected size: len(asset_feature_idx) + 6

        Portfolio token:
            time features (len(portfolio_time_idx)) taken from asset-0 raw features
            + full portfolio block (portfolio_dim)
            => size: len(portfolio_time_idx) + portfolio_dim
        """
        self.n_assets = num_assets
        self.raw_feat_dim = raw_feat_dim                      # Raw market features per asset (without SAA signal/shadow weight)
        self.d_model = d_model
        self.asset_feature_idx = list(asset_feature_idx)      # expected length 26
        self.portfolio_time_idx = list(portfolio_time_idx)    # expected length 6

        # Compute expected portfolio_dim from observation space
        obs_len = observation_space.shape[0]
        asset_block = self.n_assets * (self.raw_feat_dim + 2)  # +2 for SAA signal + shadow holding %
        if obs_len <= asset_block:
            raise ValueError(f"Observation too small. obs_len={obs_len}, asset_block={asset_block}")
        self.portfolio_dim = obs_len - asset_block

        # Portfolio block layout (must match EpisodeBuffer.get_observation_at_step in trading_environment.py):
        # [weights(N+1), 14 scalar metrics, last_target_weights(N), shadow_sortino(N), shadow_drawdown(N)]
        self.paa_extra_offset = self.n_assets + 1 + 14

        # Validate target sizes
        asset_token_in_dim = len(self.asset_feature_idx) + 6   # + SAA signal + shadow holding % + asset_weight
                                                                # + last target weight + shadow sortino + shadow drawdown
        portfolio_token_in_dim = len(self.portfolio_time_idx) + self.portfolio_dim
        # Total features out = (N assets + 1 portfolio) * d_model
        total_features_dim = (self.n_assets + 1) * d_model
        super().__init__(observation_space, features_dim=total_features_dim)

        # Embeddings
        self.asset_embedding = nn.Linear(asset_token_in_dim, d_model)
        self.portfolio_embedding = nn.Linear(portfolio_token_in_dim, d_model)

        # Learned asset identity embeddings
        self.asset_id_embedding = nn.Embedding(self.n_assets, d_model)

        # Stability helpers: small gated contribution + normalization after addition
        self.asset_id_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.asset_token_norm = nn.LayerNorm(d_model)

        # Reusable asset index buffer (moves with module device)
        self.register_buffer(
            "asset_ids",
            torch.arange(self.n_assets, dtype=torch.long),
            persistent=False,
        )

        # Initialize tokenizer projection layers with Xavier uniform and zero biases
        nn.init.xavier_uniform_(self.asset_embedding.weight)
        if self.asset_embedding.bias is not None:
            nn.init.zeros_(self.asset_embedding.bias)

        nn.init.xavier_uniform_(self.portfolio_embedding.weight)
        if self.portfolio_embedding.bias is not None:
            nn.init.zeros_(self.portfolio_embedding.bias)

        # Keep small init for identity embeddings
        nn.init.normal_(self.asset_id_embedding.weight, mean=0.0, std=0.02)


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (B, n_assets*(raw_feat_dim+2) + portfolio_dim)
        Returns: flattened tokens (B, (n_assets+1)*d_model)
        """
        B = observations.shape[0]
        asset_block = self.n_assets * (self.raw_feat_dim + 2)
        portfolio_block = observations[:, asset_block:]                    # (B, portfolio_dim)
        asset_flat = observations[:, :asset_block]                         # (B, N*(F+2))
        asset_feats_full = asset_flat.view(B, self.n_assets, self.raw_feat_dim + 2)

        # Split raw features, SAA signal, and shadow sub-portfolio holding percentage
        raw_feats = asset_feats_full[:, :, : self.raw_feat_dim]
        saa_sig = asset_feats_full[:, :, self.raw_feat_dim : self.raw_feat_dim + 1]  # (B, N, 1)
        shadow_weight = asset_feats_full[:, :, self.raw_feat_dim + 1 : self.raw_feat_dim + 2]  # (B, N, 1)

        # Select configured asset features
        asset_feats_sel = raw_feats[:, :, self.asset_feature_idx]          # (B, N, len(idx))
        if asset_feats_sel.shape[-1] != len(self.asset_feature_idx):
            raise ValueError("Selected asset features shape mismatch")

        # Live PAA asset weights from portfolio block: weights are first (N+1): cash + N assets
        asset_weights = portfolio_block[:, 1 : 1 + self.n_assets].unsqueeze(-1)  # (B, N, 1)
        if asset_weights.shape[1] != self.n_assets:
            raise ValueError("Asset weights shape mismatch")

        # Per-asset PAA diagnostics appended after weights + 14 scalar metrics (see paa_extra_offset)
        off = self.paa_extra_offset
        last_target_weights = portfolio_block[:, off : off + self.n_assets].unsqueeze(-1)              # (B, N, 1)
        shadow_sortino = portfolio_block[:, off + self.n_assets : off + 2 * self.n_assets].unsqueeze(-1)      # (B, N, 1)
        shadow_drawdown = portfolio_block[:, off + 2 * self.n_assets : off + 3 * self.n_assets].unsqueeze(-1)  # (B, N, 1)

        # Build asset tokens: selected feats + SAA signal + shadow holding % + live asset_weight
        # + last target weight + shadow sortino + shadow drawdown
        asset_token_inputs = torch.cat(
            [asset_feats_sel, saa_sig, shadow_weight, asset_weights, last_target_weights, shadow_sortino, shadow_drawdown],
            dim=-1,
        )

        asset_tokens = self.asset_embedding(asset_token_inputs)            # (B, N, d_model)

        # Inject learned asset identity
        asset_id_idx = self.asset_ids.unsqueeze(0).expand(B, -1)           # (B, N)
        asset_id_tokens = self.asset_id_embedding(asset_id_idx)            # (B, N, d_model)
        asset_tokens = self.asset_token_norm(
            asset_tokens + self.asset_id_scale * asset_id_tokens
        )

        # Portfolio token: time features (from asset 0 raw feats) + full portfolio block
        time_feats = raw_feats[:, 0, self.portfolio_time_idx]              # (B, len(portfolio_time_idx))
        if time_feats.shape[-1] != len(self.portfolio_time_idx):
            raise ValueError("Portfolio time features shape mismatch")

        portfolio_token_input = torch.cat([time_feats, portfolio_block], dim=-1)  # (B, len(time_idx)+portfolio_dim)
        if portfolio_token_input.shape[-1] != (len(self.portfolio_time_idx) + self.portfolio_dim):
            raise ValueError("Portfolio token dim mismatch")
        portfolio_token = self.portfolio_embedding(portfolio_token_input).unsqueeze(1)  # (B,1,d_model)

        # Stitch tokens: [asset_tokens, portfolio_token]. Expected in this order by attention engine!
        full_sequence = torch.cat([asset_tokens, portfolio_token], dim=1)  # (B, N+1, d_model)
        return full_sequence.flatten(start_dim=1)     

    
# Utility function to load SAA model and VecNormalize stats from config
def _load_saa_from_config(saa_config: Dict[str, Any]) -> Tuple[Any, Optional[VecNormalize], torch.device, float]:
    """
    Load the frozen SAA (RecurrentPPO) plus VecNormalize stats and its training-time
    action_limiting_factor_end (needed to rescale raw policy outputs into the same
    target_position_change range the SAA was actually trained/executed with).

    Expects keys: saa_run_id, saa_base_dir, saa_config_id, device.
    Optional key: saa_checkpoint_name (default "best_model_pv_minus_selected_asset_bh_abs_mean"),
    matching the "<stem>.zip" / "<stem>_vecnormalize.pkl" naming contract used by
    save_checkpoint_with_vecnormalize() in the SAA training script. The historical
    "best_model.zip" / "vecnormalize.pkl" filenames are no longer produced by the SAA trainer.
    """
    required = ("saa_run_id", "saa_base_dir", "saa_config_id")
    missing = [k for k in required if k not in saa_config]
    if missing:
        raise ValueError(f"Missing required SAA config keys: {missing}")

    device = torch.device(saa_config.get("device", "auto"))
    run_id = str(saa_config["saa_run_id"])
    base_dir = saa_config["saa_base_dir"]
    config_id = str(saa_config["saa_config_id"])
    saa_run_date = saa_config.get("saa_run_date", "unknown_date")
    checkpoint_name = str(saa_config.get("saa_checkpoint_name", "best_model_pv_minus_selected_asset_bh_abs_mean"))

    model_dir = os.path.join(base_dir, f"{run_id}_config_{config_id}_{saa_run_date}")
    model_path = os.path.join(model_dir, f"{checkpoint_name}.zip")
    vecnorm_path = os.path.join(model_dir, f"{checkpoint_name}_vecnormalize.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAA model not found at: {model_path}")

    # Read the SAA's own training config to recover action_limiting_factor_end. This is the
    # same inheritance pattern main.py._maybe_inherit_saa_training_config uses for test_agent runs:
    # saved_models/<run_dir_name> sits under <saa_agent_root>/saved_models, config lives at
    # <saa_agent_root>/config_<config_id>.json.
    saa_agent_root = os.path.dirname(base_dir)
    saa_training_config_path = os.path.join(saa_agent_root, f"config_{config_id}.json")
    if not os.path.isfile(saa_training_config_path):
        raise FileNotFoundError(f"SAA training config not found: {saa_training_config_path}")
    with open(saa_training_config_path, "r") as f:
        saa_training_config = json.load(f)
    agent_cfg = saa_training_config.get("agent", {})
    if "action_limiting_factor_end" not in agent_cfg:
        raise ValueError(
            f"SAA training config {saa_training_config_path} is missing "
            "'agent.action_limiting_factor_end', required to rescale raw SAA actions."
        )
    action_limiting_factor = float(agent_cfg["action_limiting_factor_end"])

    load_errors: List[str] = []
    saa_model = None
    # Override potentially-unpicklable schedule callables for inference-only loads (mirrors
    # test_saa_inference_shadow_portfolios.py._load_saa_model).
    safe_custom_objects = {
        "learning_rate": 3e-5,
        "lr_schedule": lambda _p: 3e-5,
        "clip_range": lambda _p: 0.2,
        "clip_range_vf": None,
    }
    try:
        saa_model = RecurrentPPO.load(model_path, device=device, custom_objects=safe_custom_objects)
    except Exception as e:
        load_errors.append(f"RecurrentPPO.load failed: {e}")
        raise RuntimeError(f"Failed to load SAA model. Errors: {load_errors}")

    saa_vecnormalize = None
    if os.path.exists(vecnorm_path):
        obs_space = saa_model.observation_space if hasattr(saa_model, "observation_space") \
            else gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        # VecNormalize.load() requires a VecEnv (needs .num_envs); wrap the dummy obs-space env.
        dummy_env = DummyVecEnv([lambda: _ObsNormDummyEnv(obs_space)])
        saa_vecnormalize = VecNormalize.load(vecnorm_path, dummy_env)
        saa_vecnormalize.training = False
        saa_vecnormalize.norm_reward = False

    return saa_model, saa_vecnormalize, device, action_limiting_factor

# Build PPO model 
def build_allocator_model(
    env: gym.Env,
    config: Dict[str, Any],
    num_assets: int,
    raw_feature_dim: int,
    paa_asset_token_idx: List[int],
    paa_portfolio_token_idx: List[int]
) -> PPO:
    """
    Instantiate PPO model 
    Args:
        env: Vectorized training environment
        config: Full configuration dict
        saa_model: Pretrained SAA model
        saa_vecnormalize: Optional VecNormalize instance for SAA model
        saa_device: Device for SAA model ("cpu" or "cuda")
        num_assets: Number of assets in the portfolio
        raw_feature_dim: Dimension of raw features for each asset

    Returns:
        PPO model instance ready for training
        
    Integration:
    - Reads hyperparameters from config["portfolio_allocator_agent"] section
    - Creates learning rate schedule using linear_three_phase_schedule
    - Uses standard PPO algorithm from SB3
    
    Config Keys Used (from portfolio_allocator_agent section):
    - learning_rate_start, learning_rate_end: LR schedule endpoints
    - lr_schedule_type, lr_schedule_warmup_pct, lr_schedule_ramping_pct: LR schedule config
    - ent_coef_start: Initial entropy coefficient (updated via callback)
    - n_steps: Rollout buffer size (steps before update)
    - batch_size: Minibatch size for gradient updates
    - n_epochs: Optimization epochs per rollout
    - gamma: Discount factor
    - gae_lambda: GAE lambda parameter
    - vf_coef: Value function loss coefficient
    - max_grad_norm: Gradient clipping threshold
    - normalize_advantage: Whether to normalize advantages
    - target_kl: Early stopping KL divergence threshold
    - device: PyTorch device ("cpu", "cuda", "auto")
    - verbose: Logging verbosity
    """
    
    # --- Extract agent configuration section ---
    agent_cfg = config.get("portfolio_allocator_agent", {})
    transformer_cfg = config.get("allocator_transformer", {})
    log_std_init = agent_cfg.get("log_std_init", -1.0)  # Initial log std for action distribution (tune for exploration)
    clip_range_vf = agent_cfg.get("clip_range_vf", None)  # Optional separate clip range for value function
    
    # --- Learning Rate Schedule ---
    # Extract LR schedule parameters
    lr_start = float(agent_cfg.get("learning_rate_start", 3e-4))
    lr_end = float(agent_cfg.get("learning_rate_end", 3e-5))
    lr_warmup_pct = float(agent_cfg.get("lr_schedule_warmup_pct", 0.2))
    lr_ramping_pct = float(agent_cfg.get("lr_schedule_ramping_pct", 0.6))
    # Create learning rate schedule using three-phase linear interpolation
    lr_schedule = linear_three_phase_schedule(
        start=lr_start,
        end=lr_end,
        warmup_pct=lr_warmup_pct,
        ramping_pct=lr_ramping_pct
    )

    # --- Entropy Coefficient ---
    # Initial entropy coefficient (will be updated by EntropyScheduleCallback)
    # Higher entropy = more exploration, lower = more exploitation
    ent_coef_start = float(agent_cfg.get("ent_coef_start", 0.01))
    
    # --- Clip Range Schedule (Optional) ---
    # PPO clip range for policy ratio clipping
    # Can be constant or scheduled (using linear_three_phase_schedule)
    clip_range_start = float(agent_cfg.get("clip_range_start", 0.2))
    clip_range_end = float(agent_cfg.get("clip_range_end", 0.2))
    # Check if clip range should be scheduled
    if clip_range_start != clip_range_end:
        # Create schedule if start != end
        clip_warmup_pct = float(agent_cfg.get("clip_schedule_warmup_pct", 0.2))
        clip_ramping_pct = float(agent_cfg.get("clip_schedule_ramping_pct", 0.6))
        clip_range = linear_three_phase_schedule(
            start=clip_range_start,
            end=clip_range_end,
            warmup_pct=clip_warmup_pct,
            ramping_pct=clip_ramping_pct
        )
    else:
        # Use constant clip range
        clip_range = clip_range_start
    
    # Build policy_kwargs
    # These parameters are passed directly to the policy class constructor
    # NOTE: no net_arch here - TransformerAllocatorPolicy._build_mlp_extractor always uses
    # AttentionEngine and ignores self.net_arch entirely.
    policy_kwargs = dict(
        features_extractor_class=SAATokenizer,
        features_extractor_kwargs=dict(
            num_assets=num_assets, 
            raw_feat_dim=raw_feature_dim, # ONLY raw market features without SAA signal or weights, since those are handled inside the tokenizer
            d_model=transformer_cfg.get("d_model", 128),
            asset_feature_idx=paa_asset_token_idx,
            portfolio_time_idx=paa_portfolio_token_idx
        ),
        log_std_init=log_std_init # Initial log std for action distribution (can be tuned for exploration)
    )

    # # Environment wrapper to handle stateful SAA signal injection
    """ENV already wrapped in run() with SAASignalWrapper, so we can pass it directly to PPO."""
    # wrapped_env = SAASignalWrapper(env, saa_model, saa_vecnormalize, num_assets, raw_feature_dim, saa_device)
    
    # --- Core PPO Hyperparameters ---
    # Rollout buffer size: number of steps to collect before update
    # Should be divisible by batch_size for efficient training
    n_steps = int(agent_cfg.get("n_steps", 2048))

    # Minibatch size for gradient updates
    # Smaller = more updates per rollout but noisier gradients
    batch_size = int(agent_cfg.get("batch_size", 256))
    
    # Number of epochs to train on collected rollout data
    # More epochs = more learning but risk of overfitting
    n_epochs = int(agent_cfg.get("n_epochs", 6))
    
    # Discount factor: importance of future rewards
    # 0 = only immediate rewards, 1 = all future rewards equally weighted
    gamma = float(agent_cfg.get("gamma", 0.99))
    
    # GAE lambda: bias-variance tradeoff in advantage estimation
    # 1.0 = high variance/low bias, 0.0 = low variance/high bias
    gae_lambda = float(agent_cfg.get("gae_lambda", 0.95))
    
    # Value function loss coefficient: balances actor vs critic loss
    # Higher = prioritize value function accuracy
    vf_coef = float(agent_cfg.get("vf_coef", 0.5))
    
    # Gradient clipping threshold: prevents exploding gradients
    max_grad_norm = float(agent_cfg.get("max_grad_norm", 0.5))
    
    # Whether to normalize advantages: improves stability
    normalize_advantage = bool(agent_cfg.get("normalize_advantage", True))
    
    # Target KL divergence for early stopping within epoch
    # If policy changes too much, stop current epoch
    # None = no early stopping
    target_kl = agent_cfg.get("target_kl", None)
    if target_kl is not None:
        target_kl = float(target_kl)
    
    # PyTorch device: "cpu", "cuda", "auto" (auto-detect GPU)
    device = str(agent_cfg.get("device", "auto"))
    
    # Logging verbosity: 0=silent, 1=info, 2=debug
    verbose = int(agent_cfg.get("verbose", 1))
    
    # Rolling window size for statistics (e.g., episode rewards)
    stats_window_size = int(agent_cfg.get("stats_window_size", 100))
    

    # --- TensorBoard Logging ---
    # Get TensorBoard log directory from config or use default
    training_cfg = config.get("training", {})
    tb_log_dir = training_cfg.get(
        "tensorboard_log",
        "src/agents/PPO_portfolio_allocator_weights/tb_logs"
    )
    # Ensure directory exists, create if not
    os.makedirs(tb_log_dir, exist_ok=True)
    

    # --- Instantiate PPO Model ---
    # Create PPO model with all configured parameters
    # Uses standard PPO (not RecurrentPPO). Tokens of transformer are assets/portfolio
    model = PPO(
        policy=TransformerAllocatorPolicy,  # Placeholder, actual architecture defined in policy_kwargs
        env=env,  # env wrapped for saa signal injection in run()
        policy_kwargs=dict(
            **policy_kwargs,
            n_assets=num_assets,
            d_model=transformer_cfg.get("d_model", 128),
            n_heads=transformer_cfg.get("n_heads", 8),
            n_layers=transformer_cfg.get("n_layers", 4),
            dim_feedforward=transformer_cfg.get("dim_feedforward", 256),
            transformer_encoder_dropout=transformer_cfg.get("transformer_encoder_dropout", 0.1),
            transformer_activation_fn=transformer_cfg.get("transformer_activation_fn", "relu")
        ),
        
        # Optimization hyperparameters
        clip_range_vf=clip_range_vf,  # Optional separate clip range for value function
        learning_rate=lr_schedule,  # Scheduled learning rate
        n_steps=n_steps,  # Rollout buffer size
        batch_size=batch_size,  # Minibatch size
        n_epochs=n_epochs,  # Epochs per rollout
        gamma=gamma,  # Discount factor
        gae_lambda=gae_lambda,  # GAE lambda
        
        # Loss coefficients
        ent_coef=ent_coef_start,  # Entropy coefficient (updated by callback)
        vf_coef=vf_coef,  # Value function coefficient
        
        # Clipping and regularization
        clip_range=clip_range,  # PPO clip range (constant or scheduled)
        max_grad_norm=max_grad_norm,  # Gradient clipping
        normalize_advantage=normalize_advantage,  # Advantage normalization
        target_kl=target_kl,  # Early stopping KL threshold
        
        # System configuration
        device=device,  # PyTorch device
        verbose=verbose,  # Logging level
        tensorboard_log=tb_log_dir,  # TensorBoard directory
        
        # Statistics tracking
        stats_window_size=stats_window_size,  # Rolling window for metrics
        
        # Seeding (if specified in training config)
        seed=training_cfg.get("seed", None)
    )
    
    # Log model configuration for debugging
    if verbose > 0:
        print("[build_allocator_model] PPO model instantiated with TransformerAllocatorPolicy:")
        print(f"  Learning rate: {lr_start} -> {lr_end} (warmup: {lr_warmup_pct}, ramp: {lr_ramping_pct})")
        print(f"  Entropy coef: {ent_coef_start} (initial, scheduled via callback)")
        print(f"  Clip range: {clip_range_start}" + (f" -> {clip_range_end}" if clip_range_start != clip_range_end else ""))
        print(f"  n_steps: {n_steps}, batch_size: {batch_size}, n_epochs: {n_epochs}")
        print(f"  gamma: {gamma}, gae_lambda: {gae_lambda}")
        print(f"  Transformer: d_model={transformer_cfg.get('d_model', 128)}, "
              f"n_heads={transformer_cfg.get('n_heads', 8)}, "
              f"n_layers={transformer_cfg.get('n_layers', 4)}")
        print(f"  Device: {device}")
    
    return model


def build_allocator_eval_callback(
    eval_env: VecEnv,
    config: Dict[str, Any],
    log_dir: str
) -> BaseCallback:
    """
    Build evaluation callback for allocator.
    
    Creates an AllocatorEvalCallback with nested AllocatorValidationCallback
    for comprehensive evaluation on validation data during training.
    
    Args:
        eval_env: Validation environment (VecNormalized with training=False, envs wrapped
                  in PortfolioEpisodeAdapter)
        config: Configuration dict containing training parameters
        log_dir: Directory for checkpoints and evaluation logs
    
    Returns:
        AllocatorEvalCallback instance configured with validation metrics callback
        
    Integration:
    - Reads eval parameters from config["training"] section
    - Creates AllocatorValidationCallback as nested callback
    - Returns configured AllocatorEvalCallback ready for SB3's learn()
    
    Config Keys Used:
    - training.eval_freq: Steps between evaluations (default: 10000)
    - training.patience / min_delta_reward / min_delta_sharpe: early stopping
    - training.verbose: Verbosity level (default: 1)
    """
    # Extract training configuration section
    train_cfg = config.get("training", {})
    
    # --- Extract Evaluation Parameters ---
    
    # Evaluation frequency: how often to run validation (in total timesteps)
    # Default: evaluate every 10,000 steps
    eval_freq = int(train_cfg.get("eval_freq", 10_000))
    patience = int(train_cfg.get("patience", 10))
    min_delta_reward = float(train_cfg.get("min_delta_reward", 0.0))
    min_delta_sharpe = float(train_cfg.get("min_delta_sharpe", 0.0))

    # Verbosity level for logging
    # 0 = silent, 1 = info, 2 = debug
    verbose = int(train_cfg.get("verbose", 1))
    
    # --- Create Nested Validation Metrics Callback ---
    
    # This callback accumulates per-episode metrics during evaluation
    # and computes aggregated statistics (mean, std) after all eval episodes
    val_metrics_cb = AllocatorValidationCallback(
        tag_prefix="validation",  # TensorBoard prefix for validation metrics
        verbose=verbose
    )
    
    # --- Create Main Evaluation Callback ---
    
    # This callback:
    # 1. Triggers a deterministic sweep every eval_freq calls
    # 2. Runs one full-length episode per validation block, starting from 100% cash
    # 3. Forwards per-episode infos to val_metrics_cb
    # 4. Saves checkpoints on excess-over-SPY, mean terminal PnL and worst-block terminal PnL
    # 5. Logs evaluation results to TensorBoard and disk
    eval_callback = AllocatorEvalCallback(
        eval_env=eval_env,                    # Validation environment (VecNormalized)
        best_model_save_path=log_dir,         # Directory for checkpoints
        log_path=log_dir,                     # Directory for evaluations.npz logs
        eval_freq=eval_freq,                  # Steps between evaluations
        eval_step_callback=val_metrics_cb,    # Nested callback for metrics accumulation
        patience=patience, 
        min_delta_reward=min_delta_reward, 
        min_delta_sharpe=min_delta_sharpe,   
        verbose=verbose                       # Logging verbosity
    )
    
    # Log callback configuration for debugging
    if verbose > 0:
        print(f"[build_allocator_eval_callback] Configured evaluation:")
        print(f"  eval_freq: {eval_freq} steps")
        print(f"  mode: deterministic sweep over all validation blocks (100% cash start)")
        print(f"  eval_envs: {eval_env.num_envs}")
        print(f"  log_dir: {log_dir}")
    
    return eval_callback

# --- Pretraining Critic / Value Function (Optional) ---
def _sample_random_vec_action(vec_env: VecEnv) -> np.ndarray:
    # shape: [n_envs, action_dim]
    acts = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
    return np.asarray(acts, dtype=np.float32)


def _warmup_vecnormalize_obs_stats(vec_env: VecNormalize, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        return

    old_training = bool(vec_env.training)
    old_norm_reward = bool(vec_env.norm_reward)

    # Update obs_rms only
    vec_env.training = True
    vec_env.norm_reward = False

    obs = vec_env.reset()  # shape: [1, obs_dim]
    assert obs.ndim == 2 and obs.shape[0] == vec_env.num_envs

    for _ in range(warmup_steps):
        action = _sample_random_vec_action(vec_env)  # shape: [1, action_dim]
        obs, _, dones, _ = vec_env.step(action)
        if np.any(dones):
            obs = vec_env.reset()

    vec_env.training = old_training
    vec_env.norm_reward = old_norm_reward


def _collect_mc_dataset_random_policy(
    vec_env: VecNormalize,
    n_episodes: int,
    gamma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect offline dataset from random actions across all parallel envs.
    Returns:
    - obs_all: [N, obs_dim]
    - rtg_all: [N]
    """
    n_envs = int(vec_env.num_envs)

    old_training = bool(vec_env.training)
    old_norm_reward = bool(vec_env.norm_reward)

    # Freeze normalization stats and use raw reward targets
    vec_env.training = False
    vec_env.norm_reward = False

    all_obs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    def _finalize(ep_obs: List[np.ndarray], ep_rewards: List[float]) -> None:
        r = np.asarray(ep_rewards, dtype=np.float32)             # [T]
        y = np.zeros_like(r, dtype=np.float32)                   # [T]
        running = np.float32(0.0)
        for t in range(r.shape[0] - 1, -1, -1):
            running = r[t] + np.float32(gamma) * running
            y[t] = running

        ep_obs_np = np.asarray(ep_obs, dtype=np.float32)         # [T, obs_dim]
        assert ep_obs_np.shape[0] == y.shape[0]
        all_obs.append(ep_obs_np)
        all_targets.append(y)

    # Per-env trajectory buffers; VecEnv auto-resets, so episodes are flushed on done.
    ep_obs_buf: List[List[np.ndarray]] = [[] for _ in range(n_envs)]
    ep_rew_buf: List[List[float]] = [[] for _ in range(n_envs)]
    episodes_done = 0

    obs = vec_env.reset()  # [n_envs, obs_dim]
    while episodes_done < n_episodes:
        for i in range(n_envs):
            ep_obs_buf[i].append(obs[i].astype(np.float32, copy=True))
        action = _sample_random_vec_action(vec_env)              # [n_envs, action_dim]
        obs, rewards, dones, _ = vec_env.step(action)
        for i in range(n_envs):
            ep_rew_buf[i].append(float(rewards[i]))
            if dones[i]:
                _finalize(ep_obs_buf[i], ep_rew_buf[i])
                ep_obs_buf[i] = []
                ep_rew_buf[i] = []
                episodes_done += 1

    vec_env.training = old_training
    vec_env.norm_reward = old_norm_reward

    obs_all = np.concatenate(all_obs, axis=0).astype(np.float32)      # [N, obs_dim]
    rtg_all = np.concatenate(all_targets, axis=0).astype(np.float32)  # [N]
    assert obs_all.shape[0] == rtg_all.shape[0]
    return obs_all, rtg_all


def _pretrain_allocator_critic(
    model: PPO,
    train_obs: np.ndarray,
    train_targets: np.ndarray,
    val_obs: np.ndarray,
    val_targets: np.ndarray,
    critic_cfg: Dict[str, Any],
) -> Dict[str, float]:
    """
    Supervised value pretraining with MSE.
    Uses policy.predict_values(obs) so only critic path receives gradients from loss.
    """
    device = model.device
    batch_size = int(critic_cfg.get("batch_size", 1024))
    learning_rate = float(critic_cfg.get("learning_rate", 1e-4))
    max_epochs = int(critic_cfg.get("max_epochs", 30))
    patience = int(critic_cfg.get("early_stopping_patience", 6))
    min_delta = float(critic_cfg.get("early_stopping_min_delta", 1e-5))
    max_grad_norm = float(critic_cfg.get("max_grad_norm", 1.0))

    verbose = int(critic_cfg.get("verbose", 1))
    log_every_n_epochs = int(critic_cfg.get("log_every_n_epochs", 5))

    x_train = torch.as_tensor(train_obs, dtype=torch.float32, device=device)   # [N, obs_dim]
    y_train = torch.as_tensor(train_targets, dtype=torch.float32, device=device)  # [N]
    x_val = torch.as_tensor(val_obs, dtype=torch.float32, device=device)       # [M, obs_dim]
    y_val = torch.as_tensor(val_targets, dtype=torch.float32, device=device)   # [M]

    assert x_train.ndim == 2 and x_val.ndim == 2
    assert y_train.ndim == 1 and y_val.ndim == 1
    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)

    best_state = copy.deepcopy(model.policy.state_dict())
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    model.policy.train()
    n = x_train.shape[0]

    for epoch in range(max_epochs):
        perm = torch.randperm(n, device=device)
        train_loss_sum = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_train[idx]                                  # [B, obs_dim]
            yb = y_train[idx]                                  # [B]
            pred = model.policy.predict_values(xb).squeeze(-1) # [B]

            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.policy.parameters(), max_grad_norm)
            optimizer.step()

            train_loss_sum += float(loss.item())
            n_batches += 1

        train_loss = train_loss_sum / max(1, n_batches)

        model.policy.eval()
        with torch.no_grad():
            val_pred = model.policy.predict_values(x_val).squeeze(-1)  # [M]
            val_loss = float(loss_fn(val_pred, y_val).item())

        model.policy.train()

        if verbose > 0 and (
            epoch == 0
            or (epoch + 1) % log_every_n_epochs == 0
            or bad_epochs + 1 >= patience
            or epoch + 1 == max_epochs
        ):
            print(
                f"[critic_pretraining] "
                f"epoch={epoch + 1}/{max_epochs} "
                f"train_mse={train_loss:.6f} "
                f"val_mse={val_loss:.6f} "
                f"best_val_mse={best_val:.6f} "
                f"bad_epochs={bad_epochs}"
            )

        if val_loss < (best_val - min_delta):
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.policy.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.policy.load_state_dict(best_state)

    return {
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
    }


def _reset_ppo_optimizer_after_pretraining(model: PPO) -> None:
    # Reinitialize optimizer/schedule state before PPO learning phase
    optimizer_class = getattr(model.policy, "optimizer_class", torch.optim.Adam)
    optimizer_kwargs = getattr(model.policy, "optimizer_kwargs", {})
    lr0 = float(model.lr_schedule(1.0))

    model.policy.optimizer = optimizer_class(
        model.policy.parameters(),
        lr=lr0,
        **optimizer_kwargs
    )
    model._current_progress_remaining = 1.0
    model._n_updates = 0

# ================================
# Environment Construction
# ================================

def _make_trading_env(cache: MarketDataCache, config: Dict[str, Any], mode: str, seed: int, for_eval: bool):
    """Env factory; must stay picklable for SubprocVecEnv with start_method='spawn'."""
    def _init():
        # TradingEnv draws its episode starts and random initial allocations from the global
        # numpy RNG, so each worker needs its own stream.
        np.random.seed(seed)
        env = TradingEnv(config=config, market_data_cache=cache, mode=mode)
        if for_eval:
            return PortfolioEpisodeAdapter(env)
        return env
    return _init


def _build_saa_wrapped_envs(
    cache: MarketDataCache,
    config: Dict[str, Any],
    seed: int,
    saa_model: Any,
    saa_vecnorm: Optional[VecNormalize],
    saa_device: torch.device,
    saa_action_limiting_factor: float,
    num_assets: int,
    tag: str,
) -> Tuple["SAASignalWrapper", "SAASignalWrapper"]:
    """
    Build the parallel train/eval vector envs and wrap both in SAA signal injection.

    The eval env gets one worker per validation block at most, so no eval worker idles
    during the deterministic sweep.
    """
    train_cfg = config.get("training", {})
    n_envs = int(train_cfg.get("n_envs", 1))
    if n_envs < 1:
        raise ValueError(f"training.n_envs must be >= 1, got {n_envs}")
    start_method = str(train_cfg.get("vec_env_start_method", "spawn"))
    if start_method not in ("spawn", "fork", "forkserver"):
        raise ValueError(
            f"training.vec_env_start_method must be one of spawn/fork/forkserver, got '{start_method}'"
        )

    n_blocks = len(cache.validation_blocks)
    if n_blocks == 0:
        raise RuntimeError("No validation blocks available; cannot build the validation sweep.")
    n_eval_envs = min(n_envs, n_blocks)

    def _vectorize(env_fns):
        if len(env_fns) == 1:
            return DummyVecEnv(env_fns)
        return SubprocVecEnv(env_fns, start_method=start_method)

    train_fns = [_make_trading_env(cache, config, "train", seed + rank, for_eval=False) for rank in range(n_envs)]
    eval_fns = [
        _make_trading_env(cache, config, "validation", seed + n_envs + rank, for_eval=True)
        for rank in range(n_eval_envs)
    ]

    vec_train_raw = _vectorize(train_fns)
    vec_eval_raw = _vectorize(eval_fns)

    print(
        f"[{tag}] Vectorized envs: train={n_envs}, eval={n_eval_envs} (validation blocks={n_blocks}), "
        f"class={'SubprocVecEnv' if n_envs > 1 else 'DummyVecEnv'}, start_method={start_method}"
    )

    vec_train_saa = SAASignalWrapper(
        vec_train_raw, saa_model, saa_vecnorm, num_assets, saa_device, config=config,
        feature_to_index=cache.feature_to_index, action_limiting_factor=saa_action_limiting_factor
    )
    vec_eval_saa = SAASignalWrapper(
        vec_eval_raw, saa_model, saa_vecnorm, num_assets, saa_device, config=config,
        feature_to_index=cache.feature_to_index, action_limiting_factor=saa_action_limiting_factor
    )
    return vec_train_saa, vec_eval_saa


# ================================
# Entry Point
# ================================

def run(cache: MarketDataCache, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for allocator training (called by main.py).
    
    Process:
    1. Load pre-trained SAA models from paths in config
    2. Build allocator environment with SAA ensemble
    3. Build transformer policy and PPO model
    4. Train for specified timesteps with evaluation callbacks
    5. Save final model and return summary
    
    Args:
        cache: MarketDataCache from main.py
        config: Full configuration dict (allocator_*, environment, etc.)
    
    Returns:
        Summary dict with training results, model path, timing, hyperparameters
    """

    # --- Seeds & Gamma Extraction ---
    # Set seeds for reproducibility (PyTorch, NumPy, env wrappers)
    seed = int(config.get("training", {}).get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Extract config info like gamma for VecNormalize reward normalization
    gamma_cfg = config.get("portfolio_allocator_agent", {}).get("gamma", 0.99)
    saa_config = config.get("saa_config", {})
    num_assets = cache.num_assets
    raw_feature_dim = cache.num_features
    
    critic_cfg = config.get("critic_pretraining", {})
    do_pretrain = bool(critic_cfg.get("enabled", False))
    
    # Load frozen SAA once
    saa_model, saa_vecnorm, saa_device, saa_action_limiting_factor = _load_saa_from_config(saa_config)

    # --- Build Environments for train/validation ---
    print("[run] Building training and evaluation environments...")
    vec_train_saa, vec_eval_saa = _build_saa_wrapped_envs(
        cache=cache,
        config=config,
        seed=seed,
        saa_model=saa_model,
        saa_vecnorm=saa_vecnorm,
        saa_device=saa_device,
        saa_action_limiting_factor=saa_action_limiting_factor,
        num_assets=num_assets,
        tag="run",
    )
    
    # Normalize allocator (PAA) obs and/or reward after SAA augmentation
    vec_train = VecNormalize(
        vec_train_saa,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma_cfg,
        training=True,
    )
    vec_eval = VecNormalize(
        vec_eval_saa,
        norm_obs=True,
        norm_reward=False, # Do not norm rewards in eval
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma_cfg,
        training=False,  # freeze stats for eval
    )

    # Create and validate feature index lists for SAA signal and PAA token construction
    saa_idx = [cache.feature_to_index[f] for f, on in config["saa_features"].items() if on]
    saa_market_data_feat_length = len(saa_idx)
    if saa_market_data_feat_length == 0:
        raise ValueError("No SAA features enabled in config['saa_features']. At least one feature must be enabled for SAA signal generation.")
    
    paa_asset_token_idx = [cache.feature_to_index[f] for f, on in config["paa_asset_token_features"].items() if on]
    paa_asset_token_market_data_feat_length = len(paa_asset_token_idx)
    if paa_asset_token_market_data_feat_length == 0:
        raise ValueError("No asset token features enabled in config['paa_asset_token_features']. At least one feature must be enabled for asset token construction.")
    
    paa_portfolio_token_idx = [cache.feature_to_index[f] for f, on in config["paa_portfolio_token_features"].items() if on]
    paa_portfolio_token_market_data_feat_length = len(paa_portfolio_token_idx)
    if paa_portfolio_token_market_data_feat_length == 0:
        raise ValueError("No portfolio token features enabled in config['paa_portfolio_token_features']. At least one feature must be enabled for portfolio token construction.")
    
    # Infer per-asset dim from the env observation space (unwrapped)
    sample_space = vec_train.observation_space
    obs_len = int(sample_space.shape[0])
    # vec_train is post SAASignalWrapper, so asset block includes the injected SAA signal (+1)
    # and the shadow sub-portfolio holding percentage (+1).
    expected_asset_block = num_assets * (raw_feature_dim + 2)
    portfolio_dim = obs_len - expected_asset_block
    if portfolio_dim <= 0:
        raise ValueError(
            "Invalid allocator observation layout. "
            f"obs_len={obs_len}, expected_asset_block={expected_asset_block}, "
            f"num_assets={num_assets}, raw_feature_dim={raw_feature_dim}, portfolio_dim={portfolio_dim}"
        )

    print(
        "[run] Inferred observation layout: "
        f"asset_block={expected_asset_block}, portfolio_dim={portfolio_dim}, total={obs_len}"
    )

    print("[run] Environments built successfully")


    # --- Build PPO Model: with provided SAA deps ---
    # Instantiate PPO with custom transformer policy and learning rate/entropy schedules
    print("[run] Building PPO allocator model...")

    model = build_allocator_model(
        env=vec_train, 
        config=config,
        num_assets=num_assets,
        raw_feature_dim=raw_feature_dim,
        paa_asset_token_idx=paa_asset_token_idx,
        paa_portfolio_token_idx=paa_portfolio_token_idx
    )

    print("[run] PPO allocator model built successfully")
    
    # Pretraining the critic with offline Monte Carlo targets can provide a better starting point for PPO training, especially in complex environments where learning a good value function from scratch is difficult. This section performs the following steps if critic pretraining is enabled in the config:
    if do_pretrain:
        print("[run] Starting pretraining phase for critic")
        gamma = float(config.get("portfolio_allocator_agent", {}).get("gamma", 0.99))  # same gamma as PPO
        warmup_steps = int(critic_cfg.get("warmup_steps", 20000))
        train_episodes = int(critic_cfg.get("train_episodes", 500))
        val_episodes = int(critic_cfg.get("validation_episodes", 100))

        # 1) warmup obs normalization stats on train env
        _warmup_vecnormalize_obs_stats(vec_train, warmup_steps)

        # 2) copy obs stats to validation env (train/val split remains env-mode based)
        sync_envs_normalization(vec_train, vec_eval)

        # 3) collect offline MC datasets (random actions)
        train_obs, train_targets = _collect_mc_dataset_random_policy(
            vec_env=vec_train,
            n_episodes=train_episodes,
            gamma=gamma
        )
        val_obs, val_targets = _collect_mc_dataset_random_policy(
            vec_env=vec_eval,
            n_episodes=val_episodes,
            gamma=gamma
        )

        # handshake checks
        obs_dim = int(vec_train.observation_space.shape[0])
        assert train_obs.shape[1] == obs_dim and val_obs.shape[1] == obs_dim
        assert train_obs.shape[0] == train_targets.shape[0]
        assert val_obs.shape[0] == val_targets.shape[0]

        # 4) supervised critic pretraining
        pt_stats = _pretrain_allocator_critic(
            model=model,
            train_obs=train_obs,
            train_targets=train_targets,
            val_obs=val_obs,
            val_targets=val_targets,
            critic_cfg=critic_cfg,
        )
        print("[critic_pretraining] summary:", pt_stats)

        # 5) handoff to PPO phase: fresh optimizer/schedule state
        _reset_ppo_optimizer_after_pretraining(model)

        # ensure RL flags for normal PPO phase
        vec_train.training = True
        vec_train.norm_reward = True
        vec_eval.training = False
        vec_eval.norm_reward = False

    else: print("[run] Critic pretraining disabled by config, skipping directly to PPO training")

    # --- Setup Logging Directories ---
    # Get agent directory (same folder as this module)
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load run_id from persistent storage (shared across agents)
    run_id_file = os.path.join(
        os.path.dirname(os.path.dirname(agent_dir)),  # src/data
        "data",
        "run_id.json"
    )
    
    # Ensure run_id file exists; initialize if not
    os.makedirs(os.path.dirname(run_id_file), exist_ok=True)
    if not os.path.exists(run_id_file):
        with open(run_id_file, 'w') as f:
            json.dump({"run_id": 1}, f)
    
    # Read and increment run_id
    with open(run_id_file, 'r') as f:
        run_id_data = json.load(f)
    
    current_run_id = int(run_id_data.get("run_id", 0))
    next_run_id = current_run_id + 1
    
    # Save incremented run_id back to JSON for next agent run
    with open(run_id_file, 'w') as f:
        json.dump({"run_id": next_run_id}, f)
    
    # Format run_id and config_id as 5-digit zero-padded strings
    run_id = str(current_run_id).zfill(5)
    config_id = str(config.get("training", {}).get("config_id", "00001")).zfill(5)
    
    # Get current date in YY_MM_DD format for TB log naming
    date_str = datetime.now().strftime("%y_%m_%d")
    
    # Format TensorBoard log name: XXXXX_config_ZZZZZ_YY_MM_DD
    # This naming convention allows automatic discovery and sorting of model runs
    tb_log_name = f"{run_id}_config_{config_id}_{date_str}"
    
    # Create saved_models directory for model checkpoints
    saved_models_dir = os.path.join(agent_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    
    # Final model path (saved after training completes)
    model_path = os.path.join(saved_models_dir, f"{tb_log_name}.zip")
    
    # Best model checkpoint directory (saved by EvalCallback during training)
    best_model_dir = os.path.join(saved_models_dir, tb_log_name)
    os.makedirs(best_model_dir, exist_ok=True)
    
    print(f"[run] TensorBoard log: {tb_log_name}")
    print(f"[run] Model checkpoint: {model_path}")
    print(f"[run] Best model dir: {best_model_dir}")
    
    # --- Setup Callbacks ---
    
    # Evaluation callback: runs validation periodically, saves best model
    # Includes nested AllocatorValidationCallback for detailed metrics
    eval_callback = build_allocator_eval_callback(
        eval_env=vec_eval,
        config=config,
        log_dir=best_model_dir
    )
    
    # Entropy coefficient schedule callback
    # Only instantiate if schedule parameters are present in config
    agent_cfg = config.get("portfolio_allocator_agent", {})
    ent_schedule_keys = ("ent_coef_start", "ent_coef_end", "ent_coef_schedule_warmup_pct", "ent_coef_schedule_ramping_pct")
    
    if all(k in agent_cfg for k in ent_schedule_keys):
        # All schedule keys present: create callback to animate entropy coefficient
        ent_callback = EntropyScheduleCallback(
            start=float(agent_cfg["ent_coef_start"]),
            end=float(agent_cfg["ent_coef_end"]),
            warmup_pct=float(agent_cfg["ent_coef_schedule_warmup_pct"]),
            ramping_pct=float(agent_cfg["ent_coef_schedule_ramping_pct"]),
            verbose=int(agent_cfg.get("verbose", 1))
        )
        print("[run] Entropy schedule callback enabled")
    else:
        # Schedule keys missing: no entropy animation
        ent_callback = None
        print("[run] Entropy schedule callback disabled (missing config keys)")
    
    # Training metrics callback: logs portfolio metrics to TensorBoard every rollout
    train_cfg = config.get("training", {})

    train_callback = AllocatorPortfolioLoggerCallback(
        tag_prefix="train",
        verbose=int(agent_cfg.get("verbose", 1))
    )
    
    # Build callback list for model.learn()
    callbacks = [eval_callback, train_callback]
    if ent_callback is not None:
        callbacks.append(ent_callback)
    
    print(f"[run] Registered {len(callbacks)} callbacks for training")
    
    # --- Train Model ---
    
    # Extract training parameters from config
    total_timesteps = int(train_cfg.get("total_timesteps", 2_000_000))
    verbose = int(agent_cfg.get("verbose", 1))
    
    print(f"\n[run] Starting training: {total_timesteps} timesteps")
    print(f"[run] Verbose level: {verbose}")
    
    # Record start time for elapsed duration calculation
    t0 = time.time()
    
    # Train PPO model with all callbacks
    # TensorBoard logs go to: model.logger.dir / tb_log_name
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=tb_log_name
    )
    
    # Record end time
    t1 = time.time()
    elapsed_seconds = round(t1 - t0, 2)
    
    print(f"[run] Training completed in ({elapsed_seconds / 60:.1f} minutes)")
    
    # --- Save Final Model ---
    
    # Save final trained model (after all training steps)
    # Separate from best_model.zip which is saved by EvalCallback

    # Verify path exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.save(model_path)
    print(f"[run] Final model saved to {model_path}")
    
    # --- Return Summary ---
    
    # Extract key hyperparameters for reporting
    n_steps = int(agent_cfg.get("n_steps", 2048))
    batch_size = int(agent_cfg.get("batch_size", 256))
    n_epochs = int(agent_cfg.get("n_epochs", 6))
    gamma = float(agent_cfg.get("gamma", 0.99))
    gae_lambda = float(agent_cfg.get("gae_lambda", 0.95))
    lr_start = float(agent_cfg.get("learning_rate_start", 3e-4))
    lr_end = float(agent_cfg.get("learning_rate_end", 3e-5))
    
    # Transformer architecture parameters
    transformer_cfg = config.get("allocator_transformer", {})
    d_model = int(transformer_cfg.get("d_model", 128))
    n_heads = int(transformer_cfg.get("n_heads", 8))
    n_layers = int(transformer_cfg.get("n_layers", 4))
    
    # Build and return summary dictionary for CLI output
    return {
        "agent": "PPO_portfolio_allocator",
        "policy": "TransformerAllocatorPolicy",
        "total_timesteps": total_timesteps,
        "elapsed_sec": elapsed_seconds,
        "model_path": model_path,
        "best_model_path": os.path.join(best_model_dir, "best_model.zip"),
        "tb_log_name": tb_log_name,
        "run_id": run_id,
        "config_id": config_id,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate_start": lr_start,
        "learning_rate_end": lr_end,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "num_assets": len(config.get("environment", {}).get("assets", [])),
        "training_completed": True
    }


def continue_run(cache: MarketDataCache, config: Dict[str, Any], model_path: str, saved_models_dir: str, model_dir_name: str) -> Dict[str, Any]:
    """
    Continue training from a saved PPO allocator model checkpoint.
    
    Process:
    1. Load pre-trained SAA models from config (frozen for inference)
    2. Build allocator environments with SAA ensemble
    3. Load VecNormalize stats from saved pickle file
    4. Load the pre-trained PPO model from .zip file
    5. Continue training for configured total timesteps with evaluation callbacks
    6. Save updated model; return summary dict
    
    Args:
        cache: MarketDataCache from main.py
        config: Full configuration dict (allocator_*, environment, etc.)
        model_path: path to best_model.zip file
        saved_models_dir: path to saved_models directory
        model_dir_name: name of the model directory
    
    Returns:
        Summary dict with training results, model path, timing, hyperparameters
    """
    
    # --- Seeds & Gamma Extraction ---
    seed = int(config.get("training", {}).get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    gamma_cfg = config.get("portfolio_allocator_agent", {}).get("gamma", 0.99)
    saa_config = config.get("saa_config", {})
    num_assets = cache.num_assets
    raw_feature_dim = cache.num_features
    
    # Load frozen SAA once (required for PAA to function)
    saa_model, saa_vecnorm, saa_device, saa_action_limiting_factor = _load_saa_from_config(saa_config)

    # --- Build Environments for train/validation ---
    print("[continue_run] Building training and evaluation environments...")
    vec_train_saa, vec_eval_saa = _build_saa_wrapped_envs(
        cache=cache,
        config=config,
        seed=seed,
        saa_model=saa_model,
        saa_vecnorm=saa_vecnorm,
        saa_device=saa_device,
        saa_action_limiting_factor=saa_action_limiting_factor,
        num_assets=num_assets,
        tag="continue_run",
    )

    # Load VecNormalize stats from saved pickle file
    vecnorm_path = os.path.join(saved_models_dir, model_dir_name, "vecnormalize_stats.pkl")
    if not os.path.isfile(vecnorm_path):
        # Checkpoints are now written as "<stem>_vecnormalize.pkl" next to "<stem>.zip".
        stem = os.path.splitext(os.path.basename(model_path))[0]
        fallback = os.path.join(saved_models_dir, model_dir_name, f"{stem}_vecnormalize.pkl")
        if not os.path.isfile(fallback):
            raise FileNotFoundError(
                f"VecNormalize stats not found at {vecnorm_path} nor {fallback}"
            )
        vecnorm_path = fallback
    
    # Load training VecNormalize stats
    vec_train = VecNormalize.load(vecnorm_path, venv=vec_train_saa)
    vec_train.training = True
    vec_train.norm_reward = True
    
    # Load evaluation VecNormalize stats (same file)
    vec_eval = VecNormalize.load(vecnorm_path, venv=vec_eval_saa)
    vec_eval.training = False
    vec_eval.norm_reward = False

    # Load the pre-trained model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = PPO.load(model_path, env=vec_train)

    # Extract tb_log_name from model directory name
    tb_log_name = model_dir_name
    
    # Setup eval callback with same directory (continues saving to the best_model checkpoint)
    best_model_dir = os.path.join(saved_models_dir, model_dir_name)
    eval_callback = build_allocator_eval_callback(
        eval_env=vec_eval,
        config=config,
        log_dir=best_model_dir
    )

    # Entropy schedule callback (if parameters present)
    agent_cfg = config.get("portfolio_allocator_agent", {})
    ent_schedule_keys = ("ent_coef_start", "ent_coef_end", "ent_coef_schedule_warmup_pct", "ent_coef_schedule_ramping_pct")
    
    if all(k in agent_cfg for k in ent_schedule_keys):
        ent_callback = EntropyScheduleCallback(
            start=float(agent_cfg["ent_coef_start"]),
            end=float(agent_cfg["ent_coef_end"]),
            warmup_pct=float(agent_cfg["ent_coef_schedule_warmup_pct"]),
            ramping_pct=float(agent_cfg["ent_coef_schedule_ramping_pct"]),
            verbose=int(agent_cfg.get("verbose", 1))
        )
        print("[continue_run] Entropy schedule callback enabled")
    else:
        ent_callback = None
        print("[continue_run] Entropy schedule callback disabled (missing config keys)")
    
    # Training metrics callback
    train_cfg = config.get("training", {})

    train_callback = AllocatorPortfolioLoggerCallback(
        tag_prefix="train",
        verbose=int(agent_cfg.get("verbose", 1))
    )
    
    # Build callback list
    callbacks = [eval_callback, train_callback]
    if ent_callback is not None:
        callbacks.append(ent_callback)
    
    print(f"[continue_run] Registered {len(callbacks)} callbacks for training")
    
    # --- Continue Training ---
    
    total_timesteps = int(train_cfg.get("total_timesteps", 2_000_000))
    verbose = int(agent_cfg.get("verbose", 1))
    
    print(f"\n[continue_run] Continuing training: {total_timesteps} timesteps")
    print(f"[continue_run] Verbose level: {verbose}")
    
    # Record start time
    t0 = time.time()
    
    # Continue training with callbacks
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=tb_log_name
    )
    
    # Record end time
    t1 = time.time()
    elapsed_seconds = round(t1 - t0, 2)
    
    print(f"[continue_run] Training completed in ({elapsed_seconds / 60:.1f} minutes)")
    
    # --- Save Updated Model ---
    
    # Save updated model back to same path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"[continue_run] Updated model saved to {model_path}")
    
    # --- Return Summary ---
    
    n_steps = int(agent_cfg.get("n_steps", 2048))
    batch_size = int(agent_cfg.get("batch_size", 256))
    n_epochs = int(agent_cfg.get("n_epochs", 6))
    gamma = float(agent_cfg.get("gamma", 0.99))
    gae_lambda = float(agent_cfg.get("gae_lambda", 0.95))
    lr_start = float(agent_cfg.get("learning_rate_start", 3e-4))
    lr_end = float(agent_cfg.get("learning_rate_end", 3e-5))
    
    transformer_cfg = config.get("allocator_transformer", {})
    d_model = int(transformer_cfg.get("d_model", 128))
    n_heads = int(transformer_cfg.get("n_heads", 8))
    n_layers = int(transformer_cfg.get("n_layers", 4))
    
    return {
        "agent": "PPO_portfolio_allocator (CONTINUED)",
        "policy": "TransformerAllocatorPolicy",
        "continued_from_model": model_dir_name,
        "total_timesteps": total_timesteps,
        "elapsed_sec": elapsed_seconds,
        "model_path": model_path,
        "best_model_path": os.path.join(best_model_dir, "best_model.zip"),
        "tb_log_name": tb_log_name,
        "config_id": config.get("training", {}).get("config_id", "unknown"),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate_start": lr_start,
        "learning_rate_end": lr_end,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "num_assets": len(config.get("environment", {}).get("assets", [])),
        "training_continued": True
    }
