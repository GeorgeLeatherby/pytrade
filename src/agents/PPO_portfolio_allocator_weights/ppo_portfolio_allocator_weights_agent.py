"""
--- Transformer-based Portfolio Allocator Agent (PPO + SB3) ---

Implements a multi-asset portfolio allocator trained via Stable-Baselines3 PPO.
The allocator uses a Transformer encoder to process per-asset signals from frozen SAAs
(Single-Asset Agents) combined with portfolio-level features.

Architecture:
- Input: 
    1. Per-asset tokens: raw market features, the injected SAA signal, and per-asset weight
    2. Global portfolio token
- Embedding: Linear projection + asset ID embeddings → d_model dimensions
- Transformer Encoder: Self-attention across N+1 tokens (N assets + 1 portfolio token)
- Output Heads: Per-asset raw allocation logits + cash logit → sigmoid output in [0, 1]
    (Environment applies post-policy normalization to valid portfolio weights)
- Value Head: Portfolio token → scalar value estimate
- PPO Training: Standard (non-recurrent) PPO

Frozen SAAs provide signal generation only; allocator learns to weight and combine signals.
Environment is EXECUTION_PORTFOLIO mode (full multi-asset rebalancing at each step).

Config-driven: All hyperparameters loaded from JSON (transformer architecture, PPO settings, rewards).
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributions
from typing import Callable, Dict, Any, Optional, Tuple, List
from datetime import datetime

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Normal
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecEnvWrapper, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym

from src.environment.single_asset_target_pos_drl_trading_env import TradingEnv
from src.environment.single_asset_target_pos_drl_trading_env import MarketDataCache


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
    Logs allocator-specific portfolio metrics to TensorBoard per episode.
    
    Integration:
    - Called by SB3 during training after each environment step
    - Checks for episode completion via info dict
    - Logs portfolio metrics with throttling (every N episodes)
    - Uses model.logger.record() to write to TensorBoard
    
    Metrics Logged:
    - Portfolio value (final, comparison, benchmark)
    - Portfolio return and Sharpe ratio
    - Max drawdown and volatility
    - Turnover and transaction costs
    - Asset allocation statistics
    - Reward components (if available)
    """

    def __init__(self, tag_prefix: str = "train", log_freq: int = 10, verbose: int = 0):
        """
        Initialize callback.
        
        Args:
            tag_prefix: TensorBoard metric prefix (e.g., "train")
            log_freq: Log every N completed episodes (throttling to reduce TB overhead)
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        
        self.tag_prefix = tag_prefix
        self.log_freq = log_freq
        self.episode_count = 0  # Track total completed episodes across training

    def _on_step(self) -> bool:
        """
        Called after each environment step during training.
        
        Checks if episode completed and logs metrics if at logging interval.
        
        Returns:
            True to continue training, False to stop
        """
        # Extract info dict from vectorized environment
        # DummyVecEnv wraps single env, so infos is a list with one dict
        info = self.locals.get("infos", [{}])[0]
        
        # Check if episode finished
        if info.get("episode_final", False):
            self.episode_count += 1
            
            # Throttle logging: only log every log_freq episodes
            # This reduces TensorBoard overhead for long training runs
            if self.episode_count % self.log_freq != 0:
                return True  # Skip logging but continue training
            
            # --- Core Portfolio Metrics ---
            
            # Final portfolio value at episode end
            pv = info.get("portfolio_final_value", None)
            if pv is not None:
                self.model.logger.record(f"{self.tag_prefix}/portfolio_final_value", float(pv), exclude=("stdout",))
            
            # Comparison value (initial portfolio value for calculating net return)
            comp_pv = info.get("comparison_final_value", None)
            if comp_pv is not None:
                self.model.logger.record(f"{self.tag_prefix}/comparison_final_value", float(comp_pv), exclude=("stdout",))
            
            # Return difference vs initial allocation (tracks improvement from random init)
            if pv is not None and comp_pv is not None:
                return_diff = pv - comp_pv
                self.model.logger.record(
                    f"{self.tag_prefix}/return_diff_vs_init", 
                    float(return_diff), 
                    exclude=("stdout",)
                )
            
            # Benchmark final value (buy-and-hold comparison)
            bench = info.get("benchmark_final_value", None)
            if bench is not None:
                self.model.logger.record(f"{self.tag_prefix}/benchmark_final_value", float(bench), exclude=("stdout",))
            
            # Total portfolio return (percentage)
            ret = info.get("portfolio_return", None)
            if ret is not None:
                self.model.logger.record(f"{self.tag_prefix}/portfolio_return", float(ret), exclude=("stdout",))
            
            # --- Risk Metrics ---
            
            # Sharpe ratio (risk-adjusted return)
            sharpe = info.get("episode_sharpe", None)
            if sharpe is not None:
                self.model.logger.record(f"{self.tag_prefix}/episode_sharpe", float(sharpe), exclude=("stdout",))
            
            # Maximum drawdown (peak-to-trough decline)
            dd = info.get("episode_max_drawdown", None)
            if dd is not None:
                self.model.logger.record(f"{self.tag_prefix}/episode_max_drawdown", float(dd), exclude=("stdout",))
            
            # Portfolio volatility (standard deviation of returns)
            if "portfolio_volatility" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/portfolio_volatility", 
                    float(info["portfolio_volatility"]), 
                    exclude=("stdout",)
                )
            
            # --- Trading Activity Metrics ---
            
            # Turnover (total trading volume relative to portfolio size)
            if "total_turnover" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/episode_turnover", 
                    float(info["total_turnover"]), 
                    exclude=("stdout",)
                )
                # Average turnover per step
                if "avg_turnover" in info:
                    self.model.logger.record(
                        f"{self.tag_prefix}/avg_turnover", 
                        float(info["avg_turnover"]), 
                        exclude=("stdout",)
                    )
            
            # Transaction costs breakdown
            if "total_transaction_cost" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/cost_total", 
                    float(info["total_transaction_cost"]), 
                    exclude=("stdout",)
                )
                # Commission costs
                if "episode_cost_commission" in info:
                    self.model.logger.record(
                        f"{self.tag_prefix}/cost_commission", 
                        float(info["episode_cost_commission"]), 
                        exclude=("stdout",)
                    )
                # Spread costs (bid-ask)
                if "episode_cost_spread" in info:
                    self.model.logger.record(
                        f"{self.tag_prefix}/cost_spread", 
                        float(info["episode_cost_spread"]), 
                        exclude=("stdout",)
                    )
                # Market impact costs
                if "episode_cost_impact" in info:
                    self.model.logger.record(
                        f"{self.tag_prefix}/cost_impact", 
                        float(info["episode_cost_impact"]), 
                        exclude=("stdout",)
                    )
            
            # --- Exposure Metrics (Allocator-Specific) ---
            
            # Asset exposure statistics (fraction of capital invested vs cash)
            if "exposure_avg" in info:
                # Starting exposure
                self.model.logger.record(
                    f"{self.tag_prefix}/exposure_start", 
                    float(info.get("exposure_start", 0.0)), 
                    exclude=("stdout",)
                )
                # Average exposure during episode
                self.model.logger.record(
                    f"{self.tag_prefix}/exposure_avg", 
                    float(info["exposure_avg"]), 
                    exclude=("stdout",)
                )
                # Final exposure
                self.model.logger.record(
                    f"{self.tag_prefix}/exposure_end", 
                    float(info.get("exposure_end", 0.0)), 
                    exclude=("stdout",)
                )
            
            # --- Gross vs Net Performance ---
            
            # Gross return (without transaction costs, for attribution analysis)
            if "shadow_return" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/gross_return", 
                    float(info["shadow_return"]), 
                    exclude=("stdout",)
                )
            
            # --- Allocation Statistics ---
            
            # Weight distribution across assets
            if "weight_mean" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/weight_mean", 
                    float(info["weight_mean"]), 
                    exclude=("stdout",)
                )
            if "weight_std" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/weight_std", 
                    float(info["weight_std"]), 
                    exclude=("stdout",)
                )
            if "weight_concentration" in info:
                # Herfindahl index (sum of squared weights)
                self.model.logger.record(
                    f"{self.tag_prefix}/weight_concentration", 
                    float(info["weight_concentration"]), 
                    exclude=("stdout",)
                )
            
            # Cash weight statistics
            if "cash_weight_mean" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/cash_weight_mean", 
                    float(info["cash_weight_mean"]), 
                    exclude=("stdout",)
                )
            if "cash_weight_end" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/cash_weight_end", 
                    float(info["cash_weight_end"]), 
                    exclude=("stdout",)
                )
            
            # --- Reward Components (for debugging reward shaping) ---
            
            # Cumulative reward over episode
            if "cumulative_reward" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/cumulative_reward", 
                    float(info["cumulative_reward"]), 
                    exclude=("stdout",)
                )
            
            # Sortino reward components (if using differential Sortino)
            if "sortino_mean_ema" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/sortino_mean_ema", 
                    float(info["sortino_mean_ema"]), 
                    exclude=("stdout",)
                )
            if "sortino_downside_ema" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/sortino_downside_ema", 
                    float(info["sortino_downside_ema"]), 
                    exclude=("stdout",)
                )
            
            # Alpha (excess return over benchmark)
            if "alpha_return" in info:
                self.model.logger.record(
                    f"{self.tag_prefix}/alpha_return", 
                    float(info["alpha_return"]), 
                    exclude=("stdout",)
                )
        
        # Continue training
        return True


class AllocatorValidationCallback(BaseCallback):
    """
    Accumulates and logs validation metrics from evaluation episodes.
    
    Integration:
    - Passed as eval_step_callback to AllocatorEvalCallback
    - Accumulates metrics during each evaluation episode
    - Computes aggregated statistics after all eval episodes complete
    - Logs mean/std to TensorBoard via flush_metrics()
    
    Design Pattern:
    - _on_step(): Accumulate metrics in buffers during eval
    - flush_metrics(): Compute stats and log to TensorBoard
    - _reset_buffers(): Clear buffers for next eval run
    
    This two-phase approach allows computing statistics across multiple
    episodes before logging, providing more stable validation metrics.
    """

    def __init__(self, tag_prefix: str = "validation", verbose: int = 0):
        """
        Initialize validation callback.
        
        Args:
            tag_prefix: TensorBoard metric prefix (e.g., "validation")
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.tag_prefix = tag_prefix
        self.eval_episode_count = 0  # Track episodes in current eval run
        
        # Accumulation buffers for per-episode metrics
        # These store one value per evaluation episode
        self.pv_buffer = []           # Portfolio final values
        self.comp_pv_buffer = []      # Comparison values (initial portfolio)
        self.bench_buffer = []        # Benchmark final values
        self.ret_buffer = []          # Portfolio returns
        self.sharpe_buffer = []       # Sharpe ratios
        self.dd_buffer = []           # Max drawdowns
        self.alpha_ret_buffer = []    # Alpha (excess returns)
        self.cum_reward_buffer = []   # Cumulative rewards
        self.volatility_buffer = []   # Portfolio volatility
        self.turnover_buffer = []     # Trading turnover
        self.cost_buffer = []         # Transaction costs

    def _on_step(self) -> bool:
        """
        Called during each step of evaluation episodes.
        
        Accumulates metrics when episode completes.
        
        Returns:
            True to continue evaluation
        """
        # Extract info dict from current step
        # During evaluation, EvalCallback provides infos list
        info = self.locals.get("infos", [{}])[0]
        
        # Check if evaluation episode finished
        if info.get("episode_final", False):
            self.eval_episode_count += 1
            
            # --- Accumulate Core Metrics ---
            
            # Portfolio final value
            pv = info.get("portfolio_final_value", None)
            if pv is not None:
                self.pv_buffer.append(float(pv))
            
            # Comparison value (for tracking improvement)
            comp_pv = info.get("comparison_final_value", None)
            if comp_pv is not None:
                self.comp_pv_buffer.append(float(comp_pv))
            
            # Benchmark final value
            bench = info.get("benchmark_final_value", None)
            if bench is not None:
                self.bench_buffer.append(float(bench))
            
            # Portfolio return
            ret = info.get("portfolio_return", None)
            if ret is not None:
                self.ret_buffer.append(float(ret))
            
            # Sharpe ratio
            sharpe = info.get("episode_sharpe", None)
            if sharpe is not None:
                self.sharpe_buffer.append(float(sharpe))
            
            # Max drawdown
            dd = info.get("episode_max_drawdown", None)
            if dd is not None:
                self.dd_buffer.append(float(dd))
            
            # Alpha (excess return over benchmark)
            alpha_ret = info.get("alpha_return", None)
            if alpha_ret is not None:
                self.alpha_ret_buffer.append(float(alpha_ret))
            
            # Cumulative reward
            cum_reward = info.get("cumulative_reward", None)
            if cum_reward is not None:
                self.cum_reward_buffer.append(float(cum_reward))
            
            # Portfolio volatility
            vol = info.get("portfolio_volatility", None)
            if vol is not None:
                self.volatility_buffer.append(float(vol))
            
            # Trading turnover
            turnover = info.get("total_turnover", None)
            if turnover is not None:
                self.turnover_buffer.append(float(turnover))
            
            # Transaction costs
            cost = info.get("total_transaction_cost", None)
            if cost is not None:
                self.cost_buffer.append(float(cost))
        
        # Continue evaluation
        return True
    
    def flush_metrics(self, n_eval_episodes: int) -> None:
        """
        Compute and log aggregated validation statistics.
        
        Called after all evaluation episodes complete (by AllocatorEvalCallback).
        Computes mean/std across episodes and logs to TensorBoard.
        
        Args:
            n_eval_episodes: Expected number of evaluation episodes
                            Used to verify all episodes completed before logging
        """
        # Guard: only log if we have the expected number of episodes
        # This prevents partial logging if evaluation was interrupted
        if self.eval_episode_count < n_eval_episodes:
            if self.verbose > 0:
                print(f"[AllocatorValidationCallback] Skipping flush: only {self.eval_episode_count}/{n_eval_episodes} episodes completed")
            return
        
        # --- Portfolio Value Statistics ---
        
        if self.pv_buffer:
            mean_pv = float(np.mean(self.pv_buffer))
            std_pv = float(np.std(self.pv_buffer))
            self.model.logger.record(f"{self.tag_prefix}/portfolio_final_value_mean", mean_pv, exclude=("stdout",))
            self.model.logger.record(f"{self.tag_prefix}/portfolio_final_value_std", std_pv, exclude=("stdout",))
        
        if self.comp_pv_buffer:
            mean_comp = float(np.mean(self.comp_pv_buffer))
            self.model.logger.record(f"{self.tag_prefix}/comparison_final_value_mean", mean_comp, exclude=("stdout",))
        
        # Return difference vs initial allocation
        if self.pv_buffer and self.comp_pv_buffer:
            return_diffs = [pv - comp for pv, comp in zip(self.pv_buffer, self.comp_pv_buffer)]
            mean_return_diff = float(np.mean(return_diffs))
            std_return_diff = float(np.std(return_diffs))
            self.model.logger.record(f"{self.tag_prefix}/return_diff_vs_init_mean", mean_return_diff, exclude=("stdout",))
            self.model.logger.record(f"{self.tag_prefix}/return_diff_vs_init_std", std_return_diff, exclude=("stdout",))
        
        if self.bench_buffer:
            mean_bench = float(np.mean(self.bench_buffer))
            self.model.logger.record(f"{self.tag_prefix}/benchmark_final_value_mean", mean_bench, exclude=("stdout",))
        
        # --- Return Statistics ---
        
        if self.ret_buffer:
            mean_ret = float(np.mean(self.ret_buffer))
            std_ret = float(np.std(self.ret_buffer))
            self.model.logger.record(f"{self.tag_prefix}/portfolio_return_mean", mean_ret, exclude=("stdout",))
            self.model.logger.record(f"{self.tag_prefix}/portfolio_return_std", std_ret, exclude=("stdout",))
        
        # --- Risk-Adjusted Performance ---
        
        if self.sharpe_buffer:
            mean_sharpe = float(np.mean(self.sharpe_buffer))
            std_sharpe = float(np.std(self.sharpe_buffer))
            self.model.logger.record(f"{self.tag_prefix}/episode_sharpe_mean", mean_sharpe, exclude=("stdout",))
            self.model.logger.record(f"{self.tag_prefix}/episode_sharpe_std", std_sharpe, exclude=("stdout",))
        
        if self.dd_buffer:
            mean_dd = float(np.mean(self.dd_buffer))
            std_dd = float(np.std(self.dd_buffer))
            self.model.logger.record(f"{self.tag_prefix}/episode_max_drawdown_mean", mean_dd, exclude=("stdout",))
            self.model.logger.record(f"{self.tag_prefix}/episode_max_drawdown_std", std_dd, exclude=("stdout",))
        
        if self.alpha_ret_buffer:
            mean_alpha = float(np.mean(self.alpha_ret_buffer))
            std_alpha = float(np.std(self.alpha_ret_buffer))
            self.model.logger.record(f"{self.tag_prefix}/alpha_return_mean", mean_alpha, exclude=("stdout",))
            self.model.logger.record(f"{self.tag_prefix}/alpha_return_std", std_alpha, exclude=("stdout",))
        
        # --- Reward Statistics ---
        
        if self.cum_reward_buffer:
            mean_cum_reward = float(np.mean(self.cum_reward_buffer))
            std_cum_reward = float(np.std(self.cum_reward_buffer))
            self.model.logger.record(f"{self.tag_prefix}/cumulative_reward_mean", mean_cum_reward, exclude=("stdout",))
            self.model.logger.record(f"{self.tag_prefix}/cumulative_reward_std", std_cum_reward, exclude=("stdout",))
        
        # --- Trading Activity Statistics ---
        
        if self.volatility_buffer:
            mean_vol = float(np.mean(self.volatility_buffer))
            self.model.logger.record(f"{self.tag_prefix}/portfolio_volatility_mean", mean_vol, exclude=("stdout",))
        
        if self.turnover_buffer:
            mean_turnover = float(np.mean(self.turnover_buffer))
            self.model.logger.record(f"{self.tag_prefix}/turnover_mean", mean_turnover, exclude=("stdout",))
        
        if self.cost_buffer:
            mean_cost = float(np.mean(self.cost_buffer))
            self.model.logger.record(f"{self.tag_prefix}/transaction_cost_mean", mean_cost, exclude=("stdout",))
        
        # Reset buffers for next evaluation run
        self._reset_buffers()
    
    def _reset_buffers(self) -> None:
        """
        Clear all accumulation buffers after logging.
        
        Called automatically by flush_metrics() to prepare for next eval run.
        """
        self.pv_buffer = []
        self.comp_pv_buffer = []
        self.bench_buffer = []
        self.ret_buffer = []
        self.sharpe_buffer = []
        self.dd_buffer = []
        self.alpha_ret_buffer = []
        self.cum_reward_buffer = []
        self.volatility_buffer = []
        self.turnover_buffer = []
        self.cost_buffer = []
        self.eval_episode_count = 0


class AllocatorEvalCallback(BaseCallback):
    """
    Periodic evaluation callback for allocator on validation data.
    
    Integration:
    - Registered with SB3's learn() via callback list
    - Triggers evaluation every eval_freq environment steps
    - Uses evaluate_policy() from SB3 to run N evaluation episodes
    - Forwards per-step metrics to eval_step_callback (AllocatorValidationCallback)
    - Saves best model checkpoint based on mean reward
    
    Architecture:
    - _init_callback(): Setup directories and initialize sub-callbacks
    - _on_step(): Check if evaluation due, run evaluate_policy(), log results
    - Uses nested callback pattern: this callback wraps AllocatorValidationCallback
    
    This mirrors EvalCallbackWithMetrics from SAA agent but adapted for allocator
    environment and metrics.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        best_model_save_path: str,
        log_path: str,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool = False,
        eval_step_callback: Optional[BaseCallback] = None,
        verbose: int = 0
    ):
        """
        Initialize evaluation callback.
        
        Args:
            eval_env: Validation environment (typically wrapped with VecNormalize)
            best_model_save_path: Directory for best model checkpoint
            log_path: Directory for evaluation logs (evaluations.npz)
            eval_freq: Evaluate every N environment steps (total_timesteps)
            n_eval_episodes: Number of episodes per evaluation run
            deterministic: If True, use deterministic policy (no exploration noise)
            eval_step_callback: Optional callback for per-step metrics during eval
                               (typically AllocatorValidationCallback)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        # Store parameters
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.eval_step_callback = eval_step_callback
        
        # Track best performance for checkpoint saving
        self.best_mean_reward = -np.inf
        
        # Count evaluation runs (for logging frequency)
        self.n_eval_calls = 0

    def _init_callback(self) -> None:
        """
        Initialize callback before training starts.
        
        Called by SB3 after model is created but before training begins.
        Sets up directories and initializes sub-callbacks.
        """
        # Create directory for best model checkpoint
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        
        # Create directory for evaluation logs
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
        
        # Initialize sub-callback (AllocatorValidationCallback)
        # This allows it to access self.model for logging
        if self.eval_step_callback is not None:
            self.eval_step_callback.init_callback(self.model)

    def _on_step(self) -> bool:
        """
        Called after each training step.
        
        Checks if evaluation is due (every eval_freq steps), runs evaluation,
        logs metrics, and saves best model.
        
        Returns:
            True to continue training, False to stop
        """
        # Check if evaluation is due
        # self.n_calls tracks total environment steps since training start
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            
            # --- Run Evaluation Episodes ---
            
            # Define nested callback for evaluate_policy()
            # This forwards each step to our eval_step_callback
            def _step_cb(locals_, globals_):
                if self.eval_step_callback is None:
                    return True
                # Inject locals/globals into sub-callback for metric extraction
                self.eval_step_callback.locals = locals_
                self.eval_step_callback.globals = globals_
                # Call sub-callback's _on_step()
                return bool(self.eval_step_callback.on_step())
            
            # Run evaluation using SB3's evaluate_policy utility
            # This runs n_eval_episodes complete episodes and returns rewards/lengths
            episode_rewards, episode_lengths = evaluate_policy(
                model=self.model,
                env=self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=False,  # No rendering during automated evaluation
                return_episode_rewards=True,  # Get per-episode rewards for stats
                warn=True,  # Warn about potential issues
                callback=_step_cb  # Per-step callback for metrics
            )
            
            self.n_eval_calls += 1
            
            # --- Flush Aggregated Validation Metrics ---
            
            # After all eval episodes complete, compute aggregated statistics
            # This calls flush_metrics() on AllocatorValidationCallback
            if self.eval_step_callback is not None:
                self.eval_step_callback.flush_metrics(self.n_eval_episodes)
            
            # --- Log Standard SB3 Evaluation Metrics ---
            
            # Compute mean and std of episode rewards
            mean_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))
            mean_ep_length = float(np.mean(episode_lengths))
            
            # Log to TensorBoard
            self.model.logger.record("eval/mean_reward", mean_reward)
            self.model.logger.record("eval/std_reward", std_reward)
            self.model.logger.record("eval/mean_ep_length", mean_ep_length)
            
            # --- Save Evaluation Results to Disk ---
            
            # Save numpy arrays for offline analysis
            if self.log_path is not None:
                eval_log_path = os.path.join(self.log_path, "evaluations.npz")
                
                # Load existing data if available (append mode)
                if os.path.exists(eval_log_path):
                    try:
                        existing_data = np.load(eval_log_path)
                        timesteps = np.append(existing_data["timesteps"], self.num_timesteps)
                        results = np.append(existing_data["results"], [episode_rewards], axis=0)
                        ep_lengths = np.append(existing_data["ep_lengths"], [episode_lengths], axis=0)
                    except Exception:
                        # If loading fails, start fresh
                        timesteps = np.array([self.num_timesteps])
                        results = np.array([episode_rewards])
                        ep_lengths = np.array([episode_lengths])
                else:
                    # First evaluation: create new arrays
                    timesteps = np.array([self.num_timesteps])
                    results = np.array([episode_rewards])
                    ep_lengths = np.array([episode_lengths])
                
                # Save to disk (compressed format)
                np.savez(
                    eval_log_path,
                    timesteps=timesteps,
                    results=results,
                    ep_lengths=ep_lengths
                )
            
            # --- Save Best Model Checkpoint ---
            
            # Check if this is the best performance so far
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f"[AllocatorEvalCallback] New best mean reward: {mean_reward:.3f} (previous: {self.best_mean_reward:.3f})")
                
                self.best_mean_reward = mean_reward
                
                # Save model checkpoint
                if self.best_model_save_path is not None:
                    best_model_path = os.path.join(self.best_model_save_path, "best_model")
                    self.model.save(best_model_path)
                    
                    if self.verbose > 0:
                        print(f"[AllocatorEvalCallback] Saved best model to: {best_model_path}")
        
        # Continue training
        return True

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
    Injects frozen SAA signal into observations while preserving recurrent state fidelity.
    Maintains SAA hidden states per (env, asset) and uses episode_start masks.
    """
    def __init__(self, venv: VecEnv, saa_model, saa_vecnormalize: Optional[VecNormalize],
                 num_assets: int, raw_feat_dim: int, device: torch.device):
        
        super().__init__(venv)

        self.saa_model = saa_model
        self.saa_vecnormalize = saa_vecnormalize
        self.num_assets = num_assets
        self.device = device

        # Infer per-asset feature dim from the env observation space
        obs_len = self.observation_space.shape[0]
        portfolio_dim = self.num_assets + 7  # weights (N+1) + 6 metrics
        asset_block = obs_len - portfolio_dim
        if asset_block <= 0 or asset_block % self.num_assets != 0:
            raise ValueError(
                f"Cannot infer per-asset dim: obs_len={obs_len}, num_assets={self.num_assets}, portfolio_dim={portfolio_dim}"
            )
        self.raw_feat_dim = asset_block // self.num_assets

        # SAA states: (None -> zero-init inside predict)
        self.saa_state = None
        # episode_start flags per (env, asset)
        self.episode_start = np.ones((venv.num_envs * num_assets,), dtype=bool)

        # Resize obs space: add +1 feature per asset for SAA signal
        old_low, old_high = self.observation_space.low, self.observation_space.high
        asset_size = self.num_assets * self.raw_feat_dim
        low_assets = old_low[:asset_size].reshape(self.num_assets, self.raw_feat_dim)
        high_assets = old_high[:asset_size].reshape(self.num_assets, self.raw_feat_dim)
        low_assets = np.concatenate([low_assets, np.full((self.num_assets, 1), -np.inf, dtype=np.float32)], axis=1)
        high_assets = np.concatenate([high_assets, np.full((self.num_assets, 1), np.inf, dtype=np.float32)], axis=1)
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
        # Wipe memory for all (env, asset)
        self.episode_start[:] = True
        self.saa_state = None
        obs = self._augment_obs_with_saa(obs)
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        flat_dones = np.repeat(dones, self.num_assets)
        self.episode_start = flat_dones
        obs = self._augment_obs_with_saa(obs)
        return obs, rewards, dones, infos

    def _augment_obs_with_saa(self, obs: np.ndarray) -> np.ndarray:
        """
        Input obs: (B, num_assets*raw_feat_dim + portfolio_dim)
        Output obs: asset part becomes num_assets*(raw_feat_dim+1) with SAA signal appended per asset.
        """
        B = obs.shape[0]
        asset_flat = obs[:, : self.num_assets * self.raw_feat_dim]  # (B, N*F)
        portfolio_part = obs[:, self.num_assets * self.raw_feat_dim :]  # (B, P)

        asset_feats = asset_flat.reshape(B * self.num_assets, self.raw_feat_dim)  # (B*N, F)

        # Portfolio-derived fields
        weights_full = portfolio_part[:, : self.num_assets + 1]  # (B, 1+N)
        cash_w = weights_full[:, [0]]  # (B,1)
        asset_w = weights_full[:, 1:]  # (B,N)
        cash_w_rep = np.repeat(cash_w, self.num_assets, axis=1)  # (B,N)

        # TODO: replace daily_agent_return placeholder with real per-asset sub-portfolio return once available
        daily_agent_return = np.zeros_like(asset_w, dtype=np.float32)  # (B,N)

        # Build per-asset SAA obs: [raw_features, cash_w, asset_w, daily_agent_return]
        per_asset_obs = np.concatenate(
            [
                asset_feats.reshape(B, self.num_assets, self.raw_feat_dim),
                cash_w_rep[..., None],
                asset_w[..., None],
                daily_agent_return[..., None],
            ],
            axis=-1,
        ).reshape(B * self.num_assets, self.raw_feat_dim + 3)  # (B*N, F+3)

        # Apply VecNormalize stats if present (obs_rms only)
        if self.saa_vecnormalize is not None and hasattr(self.saa_vecnormalize, "obs_rms"):
            rms = self.saa_vecnormalize.obs_rms
            mean = torch.as_tensor(rms.mean, device=self.device, dtype=torch.float32)
            var = torch.as_tensor(rms.var, device=self.device, dtype=torch.float32)
            eps = 1e-8
            per_asset_obs_t = torch.as_tensor(per_asset_obs, device=self.device, dtype=torch.float32)
            per_asset_obs_t = (per_asset_obs_t - mean) / torch.sqrt(var + eps)
            per_asset_obs = per_asset_obs_t.cpu().numpy()

        # Recurrent predict
        with torch.no_grad():
            torch_obs = torch.as_tensor(per_asset_obs, device=self.device, dtype=torch.float32)
            episode_start = torch.as_tensor(self.episode_start, device=self.device, dtype=torch.bool)
            actions, self.saa_state = self.saa_model.policy.predict(
                torch_obs, 
                state=self.saa_state, 
                episode_start=episode_start, 
                deterministic=True
            )

        # actions can be np.ndarray or torch.Tensor; normalize to np
        if isinstance(actions, np.ndarray):
            actions_np = actions
        else:
            actions_np = actions.detach().cpu().numpy()
        saa_sig = actions_np.reshape(B, self.num_assets, -1)
        # Use the first action dimension as signal
        saa_sig = saa_sig[..., 0:1]  # (B, N, 1)

        # Inject SAA signal into asset features
        augmented_assets = np.concatenate(
            [
                asset_feats.reshape(B, self.num_assets, self.raw_feat_dim),
                saa_sig,
            ],
            axis=-1,
        ).reshape(B, -1)  # (B, N*(F+1))

        return np.concatenate([augmented_assets, portfolio_part], axis=1)
    

class AttentionEngine(nn.Module):
    def __init__(self, feature_dim: int, n_assets: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model

        # 1. Transformer Encoder: Using batch_first = True simplifies shapes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,  # Common choice for feedforward dimension
            batch_first=True  # Use batch_first for easier integration with SB3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 2. SB3 Dimension Requirements
        # Actor (pi) will see all asset tokens flattened
        self.latent_dim_pi = n_assets * d_model 
        # Critic (vf) will see ONLY the portfolio [CLS] token
        self.latent_dim_vf = d_model 

    def forward(self, features: torch.Tensor):
        batch_size = features.shape[0]

        # 1. Reshape to (Batch, Sequence, Features)
        # Sequence length = N_assets + 1 (portfolio token)
        x = features.view(batch_size, self.n_assets + 1, self.d_model)

        # 2. Cross-Asset Attention Pass
        # Assets attend to each other and the portfolio token
        attended = self.transformer(x)  

        # 3. Extract Actor and Critic Latent Representations
        # latent pi: Tokens 0 to N-1 (assets)
        latent_pi = attended[:, :self.n_assets, :].reshape(batch_size, -1)

        # latent vf: Token N (portfolio [CLS])
        latent_vf = attended[:, -1, :]

        return latent_pi, latent_vf


class TransformerAllocatorPolicy(ActorCriticPolicy):
    """
    Custom policy that swaps the default MlpExtractor with AttentionEngine.
    Expects features_dim = (n_assets + 1) * d_model from the tokenizer.
    """
    def __init__(self, observation_space, action_space, lr_schedule, n_assets, d_model, n_heads, n_layers, **kwargs):
        self._n_assets = n_assets
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        # Replace with attention engine; note AttentionEngine.latent_dim_pi/vf attributes
        self.mlp_extractor = AttentionEngine(
            feature_dim=self.features_dim,
            n_assets=self._n_assets,
            d_model=self._d_model,
            n_heads=self._n_heads,
            n_layers=self._n_layers
        )


class SAATokenizer(BaseFeaturesExtractor):
    def __init__(
            self, 
            observation_space: gym.Space, 
            num_assets: int, 
            raw_feat_dim: int, 
            d_model: int, 
            saa_model: Optional[Any] = None,           # preloaded, unused in forward
            saa_vecnormalize: Optional[VecNormalize] = None,  # preloaded, unused in forward
            saa_config: Optional[Dict[str, Any]] = None       # kept for compatibility
        
        ):
        # Calculate final flattened output: (N_assets + 1) * d_model. The +1 is for the portfolio-level token.
        total_features_dim = (num_assets + 1) * d_model
        super().__init__(observation_space, features_dim=total_features_dim)

        self.n_assets = num_assets
        self.raw_feat_dim = raw_feat_dim
        self.d_model = d_model
        # self.saa_config = saa_config or {}
        # self.saa_device = torch.device(self.saa_config.get("device", "cpu"))

        # 1. Asset Token Embedding: Projects raw features + SAA return (1) + assets weight to d_model
        self.asset_embedding = nn.Linear(raw_feat_dim + 1 + 1, d_model)  # +1 for SAA signal, +1 for asset weight

        # 2. Portfolio Token Embedding
        asset_block = num_assets * (raw_feat_dim + 1)  
        self.portfolio_dimension = observation_space.shape[0] - asset_block
        self.portfolio_embedding = nn.Linear(self.portfolio_dimension, d_model)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # Normalize raw observations using the loaded VecNormalize stats
        # This ensures the SAA model receives inputs in the same scale as during its training
        # VecNormalize uses: (obs - mean) / sqrt(var + eps)
        mean = torch.tensor(self.saa_vecnormalize.obs_rms.mean, device=self.saa_device)
        var = torch.tensor(self.saa_vecnormalize.obs_rms.var, device=self.saa_device)
        eps = 1e-8
        normalized_obs = (obs - mean) / torch.sqrt(var + eps)
        return normalized_obs

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Expects observations already augmented with SAA signal:
        - asset part shape: [B, num_assets, raw_feat_dim + 1] (raw features + SAA signal)
        - portfolio part shape: [B, portfolio_dim]
        """
        B = observations.shape[0]
        asset_block = self.num_assets * (self.raw_feat_dim + 1)
        asset_flat = observations[:, :asset_block]
        portfolio_part = observations[:, asset_block:]

        asset_feats = asset_flat.view(B, self.num_assets, self.raw_feat_dim + 1)  # (B, N, F+1)
        raw_feats = asset_feats[:, :, : self.raw_feat_dim]                       # (B, N, F)
        saa_signal = asset_feats[:, :, self.raw_feat_dim:].float()               # (B, N, 1)

        # Portfolio weights: first entry cash, then per-asset
        asset_weights = portfolio_part[:, 1:1 + self.num_assets].view(B, self.num_assets, 1)  # (B, N, 1)

        # Token build: raw features + SAA signal + asset weight
        token_inputs = torch.cat([raw_feats, saa_signal, asset_weights], dim=-1)  # (B, N, F+2)
        asset_tokens = self.asset_embedding(token_inputs)                        # (B, N, d_model)

        # Portfolio token
        portfolio_token = self.portfolio_embedding(portfolio_part).unsqueeze(1)    # (B,1,d_model)
        full_sequence = torch.cat([portfolio_token, asset_tokens], dim=1)          # (B, N+1, d_model)

        return full_sequence.flatten(start_dim=1)  # (B, (N+1)*d_model)


# Utility function to load SAA model and VecNormalize stats from config
def _load_saa_from_config(saa_config: Dict[str, Any]) -> Tuple[Any, Optional[VecNormalize], torch.device]:
    """
    Load the frozen SAA (RecurrentPPO preferred, fallback to PPO) plus VecNormalize stats.
    Expects keys: saa_run_id, saa_base_dir, saa_config_id, device.
    """
    required = ("saa_run_id", "saa_base_dir", "saa_config_id")
    missing = [k for k in required if k not in saa_config]
    if missing:
        raise ValueError(f"Missing required SAA config keys: {missing}")

    device = torch.device(saa_config.get("device", "cpu"))
    run_id = str(saa_config["saa_run_id"])
    base_dir = saa_config["saa_base_dir"]
    config_id = str(saa_config["saa_config_id"])
    saa_run_date = saa_config.get("saa_run_date", "unknown_date")

    model_dir = os.path.join(base_dir, f"{run_id}_config_{config_id}_{saa_run_date}")
    model_path = os.path.join(model_dir, "best_model.zip")
    vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAA model not found at: {model_path}")

    load_errors: List[str] = []
    saa_model = None
    try:
        saa_model = RecurrentPPO.load(model_path, device=device)
    except Exception as e:
        load_errors.append(f"RecurrentPPO.load failed: {e}")
        raise RuntimeError(f"Failed to load SAA model. Errors: {load_errors}")

    saa_vecnormalize = None
    if os.path.exists(vecnorm_path):
        dummy_env = _ObsNormDummyEnv(
            saa_model.observation_space if hasattr(saa_model, "observation_space")
            else gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        )
        saa_vecnormalize = VecNormalize.load(vecnorm_path, dummy_env)
        saa_vecnormalize.training = False
        saa_vecnormalize.norm_reward = False

    return saa_model, saa_vecnormalize, device

# Build PPO model 
def build_allocator_model(
    env: gym.Env,
    config: Dict[str, Any],
    saa_model: Any,
    saa_vecnormalize: Optional[VecNormalize],
    saa_device: torch.device,
    num_assets: int,
    raw_feature_dim: int
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
    saa_config = config.get("saa_config", {})
    transformer_cfg = config.get("allocator_transformer", {})
    
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
    policy_kwargs = dict(
        features_extractor_class=SAATokenizer,
        features_extractor_kwargs=dict(
            num_assets=num_assets, 
            raw_feat_dim=raw_feature_dim, # ONLY raw market features without SAA signal or weights, since those are handled inside the tokenizer
            d_model=transformer_cfg.get("d_model", 128),
            saa_model=saa_model,
            saa_vecnormalize=saa_vecnormalize,
            saa_config=saa_config
        ),
        # mlp_extractor_class=AttentionEngine,
        # mlp_extractor_kwargs=dict(
        #     d_model=transformer_cfg.get("d_model", 128),
        #     n_heads=transformer_cfg.get("n_heads", 4),
        #     n_layers=transformer_cfg.get("n_layers", 2),
        #     activation_fn=nn.ReLU
        # ),
        net_arch=dict(pi=[], vf=[])  # No additional MLP layers after attention; all processing in attention engine! Vital!           
    )

    # Environment wrapper to handle stateful SAA signal injection
    wrapped_env = SAASignalWrapper(env, saa_model, saa_vecnormalize, num_assets, raw_feature_dim, saa_device)
    
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
        env=wrapped_env,  # Use the wrapped environment that injects stateful SAA signal
        policy_kwargs=dict(
            **policy_kwargs,
            n_assets=num_assets,
            d_model=transformer_cfg.get("d_model", 128),
            n_heads=transformer_cfg.get("n_heads", 8),
            n_layers=transformer_cfg.get("n_layers", 4)
        ),
        
        # Optimization hyperparameters
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
        print(f"  Learning rate: {lr_start} → {lr_end} (warmup: {lr_warmup_pct}, ramp: {lr_ramping_pct})")
        print(f"  Entropy coef: {ent_coef_start} (initial, scheduled via callback)")
        print(f"  Clip range: {clip_range_start}" + (f" → {clip_range_end}" if clip_range_start != clip_range_end else ""))
        print(f"  n_steps: {n_steps}, batch_size: {batch_size}, n_epochs: {n_epochs}")
        print(f"  gamma: {gamma}, gae_lambda: {gae_lambda}")
        print(f"  Transformer: d_model={transformer_cfg.get('d_model', 128)}, "
              f"n_heads={transformer_cfg.get('n_heads', 8)}, "
              f"n_layers={transformer_cfg.get('n_layers', 4)}")
        print(f"  Device: {device}")
    
    return model


def build_allocator_eval_callback(
    eval_env: gym.Env,
    config: Dict[str, Any],
    log_dir: str
) -> BaseCallback:
    """
    Build evaluation callback for allocator.
    
    Creates an AllocatorEvalCallback with nested AllocatorValidationCallback
    for comprehensive evaluation on validation data during training.
    
    Args:
        eval_env: Validation environment (should be VecNormalized with training=False)
        config: Configuration dict containing training parameters
        log_dir: Directory for best model checkpoint and evaluation logs
    
    Returns:
        AllocatorEvalCallback instance configured with validation metrics callback
        
    Integration:
    - Reads eval parameters from config["training"] section
    - Creates AllocatorValidationCallback as nested callback
    - Returns configured AllocatorEvalCallback ready for SB3's learn()
    
    Config Keys Used:
    - training.eval_freq: Steps between evaluations (default: 10000)
    - training.n_eval_episodes: Episodes per evaluation (default: 5)
    - training.eval_deterministic: Use deterministic policy (default: False)
    - training.verbose: Verbosity level (default: 1)
    """
    # Extract training configuration section
    train_cfg = config.get("training", {})
    
    # --- Extract Evaluation Parameters ---
    
    # Evaluation frequency: how often to run validation (in total timesteps)
    # Default: evaluate every 10,000 steps
    eval_freq = int(train_cfg.get("eval_freq", 10_000))
    
    # Number of episodes to run per evaluation
    # More episodes = more stable statistics but slower evaluation
    # Default: 5 episodes (balance between speed and reliability)
    n_eval_episodes = int(train_cfg.get("n_eval_episodes", 5))
    
    # Whether to use deterministic policy during evaluation
    # True = no exploration noise (pure exploitation)
    # False = sample from policy distribution (mirrors training behavior)
    # Default: False (to see realistic performance with exploration)
    deterministic = bool(train_cfg.get("eval_deterministic", False))
    
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
    # 1. Triggers evaluation every eval_freq steps
    # 2. Runs n_eval_episodes on validation data
    # 3. Forwards per-step metrics to val_metrics_cb
    # 4. Saves best model based on mean reward
    # 5. Logs evaluation results to TensorBoard and disk
    eval_callback = AllocatorEvalCallback(
        eval_env=eval_env,                    # Validation environment (VecNormalized)
        best_model_save_path=log_dir,         # Directory for best_model.zip checkpoint
        log_path=log_dir,                     # Directory for evaluations.npz logs
        eval_freq=eval_freq,                  # Steps between evaluations
        n_eval_episodes=n_eval_episodes,      # Episodes per evaluation run
        deterministic=deterministic,          # Policy sampling mode
        eval_step_callback=val_metrics_cb,    # Nested callback for metrics accumulation
        verbose=verbose                       # Logging verbosity
    )
    
    # Log callback configuration for debugging
    if verbose > 0:
        print(f"[build_allocator_eval_callback] Configured evaluation:")
        print(f"  eval_freq: {eval_freq} steps")
        print(f"  n_eval_episodes: {n_eval_episodes}")
        print(f"  deterministic: {deterministic}")
        print(f"  log_dir: {log_dir}")
    
    return eval_callback


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
    # Extract gamma for VecNormalize reward normalization
    gamma_cfg = config.get("portfolio_allocator_agent", {}).get("gamma", 0.99)
    saa_config = config.get("saa_config", {})
    num_assets = cache.num_assets
    raw_feature_dim = cache.num_features
    
    # Load frozen SAA once
    saa_model, saa_vecnorm, saa_device = _load_saa_from_config(saa_config)

    # --- Build Environments for train/validation ---
    print("[run] Building training and evaluation environments...")
    def make_env(mode: str):
        return lambda: TradingEnv(
            config=config, 
            market_data_cache=cache, 
            mode=mode
        )
    
    # Create vectorized environments for training and evaluation
    vec_train = DummyVecEnv([make_env("train")])
    vec_eval = DummyVecEnv([make_env("validation")])

    # Infer per-asset dim from the env observation space (unwrapped)
    sample_space = vec_train.observation_space
    obs_len = sample_space.shape[0]
    portfolio_dim = num_assets + 7  # weights (N+1) + 6 metrics
    asset_block = obs_len - portfolio_dim
    if asset_block <= 0 or asset_block % num_assets != 0:
        raise ValueError(
            f"Cannot infer per-asset dim: obs_len={obs_len}, num_assets={num_assets}, portfolio_dim={portfolio_dim}"
        )
    raw_feat_dim = asset_block // num_assets


    # Wrap envs with SAA signal injection
    vec_train = SAASignalWrapper(vec_train, saa_model, saa_vecnorm, num_assets, raw_feat_dim, saa_device)
    vec_eval = SAASignalWrapper(vec_eval, saa_model, saa_vecnorm, num_assets, raw_feat_dim, saa_device)
    print("[run] Environments built successfully")


    # --- Build PPO Model: with provided SAA deps ---
    # Instantiate PPO with custom transformer policy and learning rate/entropy schedules
    print("[run] Building PPO allocator model...")

    model = build_allocator_model(
        env=vec_train, 
        config=config,
        saa_model=saa_model,
        saa_vecnormalize=saa_vecnorm,
        saa_device=saa_device,
        num_assets=num_assets,
        raw_feature_dim=raw_feature_dim
    )

    print("[run] PPO allocator model built successfully")
    

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
    
    # Training metrics callback: logs portfolio metrics to TensorBoard per episode
    train_cfg = config.get("training", {})
    train_log_freq = int(train_cfg.get("train_log_freq", 10))
    
    train_callback = AllocatorPortfolioLoggerCallback(
        tag_prefix="train",
        log_freq=train_log_freq,  # Log every N episodes
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