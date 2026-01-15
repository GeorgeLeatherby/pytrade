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

The agent uses the features: return_1d, volume_percentile_40d, rsi_14, macd, bb_position,
bb_width, volatility_5d, market_beta, avg_correlation, composite_risk_score, return_kurtosis
which are marked as true in the config file

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


#import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import configure_logger

import gymnasium as gym

# Import the single-asset environment (reuses data containers/classes defined there)
# The `cache` object passed by main.py comes from src.environment.trading_env.MarketDataCache;
# runtime type checks are duck-typed—if attributes match, it works.
from src.environment.single_asset_target_pos_drl_trading_env import TradingEnv as SingleAssetEnv


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
    def __init__(self, tag_prefix: str = "train", verbose: int = 0):
        super().__init__(verbose)
        self.tag_prefix = tag_prefix

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        if info.get("episode_final", False):
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
        config: full configuration dict (environment, features, agent).
        seed: optional PRNG seed for environment internal sampling.
        for_eval: choose eval mode ('validation' blocks) if True, else training.

    Returns:
        Gymnasium Env, wrapped to select one asset per episode and (optionally) scale actions.
    """
    mode = "validation" if for_eval else "train"
    env = SingleAssetEnv(config, cache, mode=mode)

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

    # # Optional: transform rewards (e.g., scale) — here we keep identity for clarity
    # wrapped = TransformReward(wrapped, lambda r: r)

    return wrapped


def build_policy_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build `policy_kwargs` for MlpLstmPolicy from config.

    Required keys (with examples):
    - n_lstm_layers: int (e.g., 1)
    - lstm_hidden_size: int (e.g., 128)
    - shared_lstm: bool (e.g., True)
    - net_arch: {"pi": [128], "vf": [128]} (actor/critic MLP heads sizes)

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
        "features_extractor_kwargs": {"features_dim": 64},
        "ortho_init": True # ortho init: 
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
    n_steps = int(agent_cfg.get("n_steps", 128))         # sequence length per update (days)
    batch_size = int(agent_cfg.get("batch_size", 256))
    gamma = float(agent_cfg.get("gamma", 0.99))
    gae_lambda = float(agent_cfg.get("gae_lambda", 0.95))
    vf_coef = float(agent_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(agent_cfg.get("max_grad_norm", 0.5))
    device = str(agent_cfg.get("device", "auto"))
    normalize_advantage = bool(agent_cfg.get("normalize_advantage", True))
    n_epochs = int(agent_cfg.get("n_epochs", 6) )


    # --- Actual model instantiation ---
    model = RecurrentPPO(
        policy=MlpLstmPolicy,
        env=env,
        learning_rate=lr_sched,
        ent_coef=ent_start, # initially float, will be updated using EntropyScheduleCallback
        clip_range=clip_sched,
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
        tensorboard_log=r"C:\Users\HansenSimonO\Documents\Coding\PyTradeTwo\pytrade-two\src\agents\RecurrPPO_target_position_agent\tb_logs",
        stats_window_size=int(agent_cfg.get("stats_window_size", 100))
    )
    return model


def build_eval_callback(eval_env: gym.Env, config: Dict[str, Any], log_dir: str) -> EvalCallback:
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

    callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=False
    )
    return callback


# ================================
# Custom Feature Extractor
# ================================
class InputMLPFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        n_in = observation_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.LayerNorm(128),
            nn.SiLU(), # Swish activation: x * sigmoid(x)
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, features_dim)
        )
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
        config: dict containing environment, features, agent, training settings.

    Returns:
        dict with summary: paths, timings, and key hyperparameters used.
    """
    # Set seeds for reproducibility (PyTorch, NumPy, env wrappers)
    seed = int(config.get("training", {}).get("seed", 42))
    # torch.manual_seed(seed)
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

    # Logging and checkpoints
    agent_dir = os.path.join(os.path.dirname(__file__))
    log_dir = os.path.join(agent_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(agent_dir, "recurrent_ppo_target_pos_model.zip")

    # Evaluation callback
    eval_cb = build_eval_callback(vec_eval, config, log_dir)

    # Entropy schedule callback (only if schedule keys are present)
    acfg = config.get("agent", {})
    if all(k in acfg for k in ("ent_coef_start", "ent_coef_end", "ent_coef_schedule_warmup_pct", "ent_coef_schedule_ramping_pct")):
        ent_cb = EntropyScheduleCallback(
            start=float(acfg["ent_coef_start"]),
            end=float(acfg["ent_coef_end"]),
            warmup_pct=float(acfg["ent_coef_schedule_warmup_pct"]),
            ramping_pct=float(acfg["ent_coef_schedule_ramping_pct"]),
        )


    # Custom logger callback for training/validation portfolio metrics
    train_tb_cb = EpisodePortfolioSB3LoggerCallback(tag_prefix="train")
    val_tb_cb = EpisodePortfolioSB3LoggerCallback(tag_prefix="validation")

    # Progress sync callback to propagate progress_remaining to envs
    progress_sync_cb = ProgressSyncCallback()

    # Build callback list with all available callbacks
    callback = [eval_cb, ent_cb, progress_sync_cb, train_tb_cb, val_tb_cb] 

    # ----------- Train -----------------
    total_timesteps = int(config.get("training", {}).get("total_timesteps", 300_000))

    # Starttime
    t0 = time.time()

    model.learn(
        total_timesteps=total_timesteps, 
        callback=callback, 
        progress_bar=True
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
        "n_steps": int(agent_cfg.get("n_steps", 128)),
        "batch_size": int(agent_cfg.get("batch_size", 256)),
        "gamma": float(agent_cfg.get("gamma", 0.99)),
        "gae_lambda": float(agent_cfg.get("gae_lambda", 0.95)),
        "lstm_hidden_size": int(agent_cfg.get("policy_kwargs", {}).get("lstm_hidden_size", 128)),
    }