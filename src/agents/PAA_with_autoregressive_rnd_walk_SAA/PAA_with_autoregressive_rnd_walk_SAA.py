"""
--- PAA Control Ablation: Autoregressive (AR(1)) Random-Walk Signal (PPO + SB3) ---

Control ablation for the Portfolio Allocator Agent (PAA) described in the thesis. This
agent never loads or runs the frozen Single-Asset Agent (SAA), at training time or at
inference time. In place of the SAA's own forward pass, the signal injected into each
asset's token is drawn from an autoregressive, order-one (AR(1)) noise process, clipped
to the same range as the SAA's own action output. An AR(1) process is used instead of
i.i.d. noise because it is autocorrelated from one day to the next, so it is at least
superficially smooth in the way a genuine trading signal is smooth - a harder, more
informative control condition than noise a network could trivially learn to ignore as
pure static.

Crucially, this noise is still committed to each asset's real, isolated shadow
sub-portfolio through the same shadow-portfolio bookkeeping used by the hierarchical
system (TradingEnv.apply_saa_sub_actions), so the shadow holding percentage and the
other shadow-derived diagnostics (shadow Sortino, shadow drawdown) injected alongside
the signal remain populated exactly as they are for the hierarchical system. Only the
origin of the committed action changes: from a frozen policy to a stochastic process.

Everything else (environment, reward, friction model, purged evaluation protocol,
Transformer tokenizer/policy, training/eval callbacks, schedules) is reused unmodified
from ppo_portfolio_allocator_weights_agent.py so the two systems stay directly
comparable via their TensorBoard logs.
"""

import os
import time
import json
import numpy as np
import torch

from typing import Dict, Any, Optional, Tuple, Mapping

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecNormalize, DummyVecEnv, SubprocVecEnv, VecEnvWrapper, VecEnv, sync_envs_normalization
)

import gymnasium as gym

from environment.trading_environment import MarketDataCache

# Reuse the hierarchical PAA's building blocks unmodified: same tokenizer, same
# Transformer policy, same PPO construction, same callbacks/schedules, same env factory.
# The only piece NOT imported is SAASignalWrapper (and _load_saa_from_config), which this
# module replaces with AR1SignalWrapper below.
from agents.PPO_portfolio_allocator_weights.ppo_portfolio_allocator_weights_agent import (
    linear_three_phase_schedule,
    EntropyScheduleCallback,
    AllocatorPortfolioLoggerCallback,
    build_allocator_model,
    build_allocator_eval_callback,
    _make_trading_env,
    _warmup_vecnormalize_obs_stats,
    _collect_mc_dataset_random_policy,
    _pretrain_allocator_critic,
    _reset_ppo_optimizer_after_pretraining,
)


class AR1SignalWrapper(VecEnvWrapper):
    """
    Control-ablation replacement for SAASignalWrapper.

    Injects an AR(1) noise signal per (env, asset) row instead of running a frozen SAA's
    forward pass. The observation augmentation (+2 dims per asset: injected signal, shadow
    sub-portfolio holding percentage) and the resulting obs layout are identical to
    SAASignalWrapper, so SAATokenizer/TransformerAllocatorPolicy require no changes.
    """

    def __init__(self, venv: VecEnv, num_assets: int, feature_to_index: Mapping[str, int],
                 ar1_phi: float, ar1_sigma: float, ar1_clip: float,
                 action_limiting_factor: float, seed: int):
        super().__init__(venv)

        self.num_assets = num_assets
        self.ar1_phi = float(ar1_phi)
        self.ar1_sigma = float(ar1_sigma)
        self.ar1_clip = float(ar1_clip)
        self.action_limiting_factor = float(action_limiting_factor)
        self.rng = np.random.default_rng(seed)

        if not hasattr(self.venv, "env_method"):
            raise RuntimeError(
                "AR1SignalWrapper requires a VecEnv exposing env_method(); place it directly "
                "around the raw DummyVecEnv/SubprocVecEnv (no VecNormalize in between)."
            )

        obs_len = int(self.observation_space.shape[0])
        self.raw_feat_dim = int(len(feature_to_index))
        if self.raw_feat_dim <= 0:
            raise ValueError(
                f"Cannot infer raw feature dimension from feature_to_index. raw_feat_dim={self.raw_feat_dim}"
            )
        asset_block = self.num_assets * self.raw_feat_dim
        self.portfolio_dim = obs_len - asset_block
        if self.portfolio_dim <= 0:
            raise ValueError(
                "Invalid observation layout for AR1SignalWrapper. "
                f"obs_len={obs_len}, num_assets={self.num_assets}, "
                f"raw_feat_dim={self.raw_feat_dim}, portfolio_dim={self.portfolio_dim}"
            )

        self.n_envs = int(venv.num_envs)
        self.ar_state = np.zeros((self.n_envs, self.num_assets), dtype=np.float32)
        self._last_signals = np.zeros((self.n_envs, self.num_assets, 1), dtype=np.float32)
        self._last_shadow_weights = np.zeros((self.n_envs, self.num_assets, 1), dtype=np.float32)

        # Resize obs space: add +2 features per asset (AR(1) signal + shadow holding %),
        # mirroring SAASignalWrapper exactly so SAATokenizer's slicing stays valid.
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

    def _draw_stationary(self, shape: Tuple[int, int]) -> np.ndarray:
        """Fresh AR(1) draw from its stationary distribution (used at episode start)."""
        if abs(self.ar1_phi) < 1.0:
            stat_std = self.ar1_sigma / np.sqrt(max(1.0 - self.ar1_phi ** 2, 1e-8))
        else:
            stat_std = self.ar1_sigma
        draw = self.rng.normal(0.0, stat_std, size=shape).astype(np.float32)
        return np.clip(draw, -self.ar1_clip, self.ar1_clip)

    def reset(self):
        res = self.venv.reset()
        obs = res[0] if isinstance(res, tuple) and len(res) == 2 else res
        self.ar_state = self._draw_stationary((self.n_envs, self.num_assets))
        signals, shadow_weights = self._compute_signals(obs)
        self._commit_actions(signals)
        return self._inject_signals(obs, signals, shadow_weights)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        dones_arr = np.asarray(dones, dtype=bool)

        # Terminal observations belong to the finished episode; augment with the last
        # known signal only - no AR(1) advance, no sub-portfolio commit.
        for i, info in enumerate(infos):
            if info.get("terminal_observation", None) is not None:
                info["terminal_observation"] = self._inject_signals(
                    info["terminal_observation"][None, ...],
                    self._last_signals[i][None, ...],
                    self._last_shadow_weights[i][None, ...],
                )[0]

        # AR(1) recurrence for continuing rows; fresh stationary draw for envs that just
        # auto-reset (mirrors SAASignalWrapper dropping LSTM state on episode_start).
        eps = self.rng.normal(0.0, self.ar1_sigma, size=(self.n_envs, self.num_assets)).astype(np.float32)
        advanced = np.clip(self.ar1_phi * self.ar_state + eps, -self.ar1_clip, self.ar1_clip)
        fresh = self._draw_stationary((self.n_envs, self.num_assets))
        self.ar_state = np.where(dones_arr[:, None], fresh, advanced)

        signals, shadow_weights = self._compute_signals(obs)
        self._commit_actions(signals)
        return self._inject_signals(obs, signals, shadow_weights), rewards, dones, infos

    def _compute_signals(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds (signals, shadow_weights), each shaped (B, N, 1). The AR(1) state supplies
        the signal directly (no forward pass); only the shadow sub-portfolio's own holding
        percentage still needs the per-asset cash/shares/prices bundle from the env.
        """
        B = obs.shape[0]
        N = self.num_assets

        bundles = self.venv.env_method("get_saa_signal_inputs")
        cash_all = np.stack([b["cash"] for b in bundles], axis=0)      # (B, N)
        shares_all = np.stack([b["shares"] for b in bundles], axis=0)  # (B, N)
        prices_all = np.stack([b["prices"] for b in bundles], axis=0)  # (B, N)

        asset_notional = shares_all * prices_all
        shadow_sub_total = cash_all + asset_notional
        eps = 1e-12
        shadow_weight_all = np.where(
            shadow_sub_total > eps, asset_notional / np.maximum(shadow_sub_total, eps), 0.0
        ).astype(np.float32)

        signals = (self.ar_state[:B] * self.action_limiting_factor).reshape(B, N, 1).astype(np.float32)
        shadow_weights = shadow_weight_all.reshape(B, N, 1).astype(np.float32)
        self._last_signals = signals
        self._last_shadow_weights = shadow_weights
        return signals, shadow_weights

    def _commit_actions(self, signals: np.ndarray) -> None:
        """Commit the AR(1) signal to each asset's isolated shadow sub-portfolio (same bookkeeping as SAASignalWrapper)."""
        for b in range(signals.shape[0]):
            self.venv.env_method("apply_saa_sub_actions", signals[b, :, 0], indices=b)

    def _inject_signals(self, obs: np.ndarray, signals: np.ndarray, shadow_weights: np.ndarray) -> np.ndarray:
        B = obs.shape[0]
        asset_block = self.num_assets * self.raw_feat_dim
        asset_feats = obs[:, :asset_block].reshape(B, self.num_assets, self.raw_feat_dim)
        portfolio_part = obs[:, asset_block:]
        augmented_assets = np.concatenate([asset_feats, signals, shadow_weights], axis=-1).reshape(B, -1)
        return np.concatenate([augmented_assets, portfolio_part], axis=1)


def _build_ar1_wrapped_envs(
    cache: MarketDataCache,
    config: Dict[str, Any],
    seed: int,
    num_assets: int,
    ar1_cfg: Dict[str, Any],
    tag: str,
) -> Tuple[AR1SignalWrapper, AR1SignalWrapper]:
    """Build the parallel train/eval vector envs and wrap both in AR(1) signal injection."""
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
        f"class={'SubprocVecEnv' if n_envs > 1 else 'DummyVecEnv'}, start_method={start_method}. "
        f"No SAA model is loaded or run in this ablation."
    )

    phi = float(ar1_cfg.get("phi", 0.9))
    sigma = float(ar1_cfg.get("sigma", 0.15))
    clip_value = float(ar1_cfg.get("clip", 1.0))
    action_limiting_factor = float(ar1_cfg.get("action_limiting_factor", 0.3))

    vec_train_ar1 = AR1SignalWrapper(
        vec_train_raw, num_assets, cache.feature_to_index,
        ar1_phi=phi, ar1_sigma=sigma, ar1_clip=clip_value,
        action_limiting_factor=action_limiting_factor, seed=seed,
    )
    vec_eval_ar1 = AR1SignalWrapper(
        vec_eval_raw, num_assets, cache.feature_to_index,
        ar1_phi=phi, ar1_sigma=sigma, ar1_clip=clip_value,
        action_limiting_factor=action_limiting_factor, seed=seed + 10_000,
    )
    return vec_train_ar1, vec_eval_ar1


def _build_run_id_paths(agent_dir: str, config: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Same run_id/tb_log_name/model-path convention used by the hierarchical PAA trainer."""
    run_id_file = os.path.join(
        os.path.dirname(os.path.dirname(agent_dir)),  # src/data
        "data",
        "run_id.json"
    )
    os.makedirs(os.path.dirname(run_id_file), exist_ok=True)
    if not os.path.exists(run_id_file):
        with open(run_id_file, 'w') as f:
            json.dump({"run_id": 1}, f)

    with open(run_id_file, 'r') as f:
        run_id_data = json.load(f)

    current_run_id = int(run_id_data.get("run_id", 0))
    next_run_id = current_run_id + 1
    with open(run_id_file, 'w') as f:
        json.dump({"run_id": next_run_id}, f)

    run_id = str(current_run_id).zfill(5)
    config_id = str(config.get("training", {}).get("config_id", "00001")).zfill(5)
    date_str = time.strftime("%y_%m_%d")
    tb_log_name = f"{run_id}_config_{config_id}_{date_str}"

    saved_models_dir = os.path.join(agent_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    model_path = os.path.join(saved_models_dir, f"{tb_log_name}.zip")
    best_model_dir = os.path.join(saved_models_dir, tb_log_name)
    os.makedirs(best_model_dir, exist_ok=True)

    return tb_log_name, model_path, best_model_dir, saved_models_dir


def _build_callbacks(config: Dict[str, Any], vec_eval: VecNormalize, best_model_dir: str):
    eval_callback = build_allocator_eval_callback(eval_env=vec_eval, config=config, log_dir=best_model_dir)

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
        print("[run] Entropy schedule callback enabled")
    else:
        ent_callback = None
        print("[run] Entropy schedule callback disabled (missing config keys)")

    train_callback = AllocatorPortfolioLoggerCallback(tag_prefix="train", verbose=int(agent_cfg.get("verbose", 1)))

    callbacks = [eval_callback, train_callback]
    if ent_callback is not None:
        callbacks.append(ent_callback)
    return callbacks


def run(cache: MarketDataCache, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the AR(1) control-ablation trainer (called by main.py exactly like the
    hierarchical PAA). Never loads or runs a frozen SAA; the per-asset signal fed to the
    Transformer tokenizer instead comes from AR1SignalWrapper.
    """
    seed = int(config.get("training", {}).get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    gamma_cfg = config.get("portfolio_allocator_agent", {}).get("gamma", 0.99)
    num_assets = cache.num_assets
    raw_feature_dim = cache.num_features
    ar1_cfg = config.get("ar1_ablation", {})

    critic_cfg = config.get("critic_pretraining", {})
    do_pretrain = bool(critic_cfg.get("enabled", False))

    print("[run] Building training and evaluation environments (AR(1) control ablation - no SAA loaded)...")
    vec_train_ar1, vec_eval_ar1 = _build_ar1_wrapped_envs(
        cache=cache, config=config, seed=seed, num_assets=num_assets, ar1_cfg=ar1_cfg, tag="run",
    )

    vec_train = VecNormalize(
        vec_train_ar1, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
        gamma=gamma_cfg, training=True,
    )
    vec_eval = VecNormalize(
        vec_eval_ar1, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0,
        gamma=gamma_cfg, training=False,
    )

    paa_asset_token_idx = [cache.feature_to_index[f] for f, on in config["paa_asset_token_features"].items() if on]
    if len(paa_asset_token_idx) == 0:
        raise ValueError("No asset token features enabled in config['paa_asset_token_features'].")
    paa_portfolio_token_idx = [cache.feature_to_index[f] for f, on in config["paa_portfolio_token_features"].items() if on]
    if len(paa_portfolio_token_idx) == 0:
        raise ValueError("No portfolio token features enabled in config['paa_portfolio_token_features'].")

    obs_len = int(vec_train.observation_space.shape[0])
    expected_asset_block = num_assets * (raw_feature_dim + 2)
    portfolio_dim = obs_len - expected_asset_block
    if portfolio_dim <= 0:
        raise ValueError(
            "Invalid allocator observation layout. "
            f"obs_len={obs_len}, expected_asset_block={expected_asset_block}, "
            f"num_assets={num_assets}, raw_feature_dim={raw_feature_dim}, portfolio_dim={portfolio_dim}"
        )
    print(f"[run] Inferred observation layout: asset_block={expected_asset_block}, portfolio_dim={portfolio_dim}, total={obs_len}")
    print("[run] Environments built successfully")

    print("[run] Building PPO allocator model (shared TransformerAllocatorPolicy/SAATokenizer)...")
    model = build_allocator_model(
        env=vec_train,
        config=config,
        num_assets=num_assets,
        raw_feature_dim=raw_feature_dim,
        paa_asset_token_idx=paa_asset_token_idx,
        paa_portfolio_token_idx=paa_portfolio_token_idx,
    )
    print("[run] PPO allocator model built successfully")

    if do_pretrain:
        print("[run] Starting pretraining phase for critic")
        gamma = float(config.get("portfolio_allocator_agent", {}).get("gamma", 0.99))
        warmup_steps = int(critic_cfg.get("warmup_steps", 20000))
        train_episodes = int(critic_cfg.get("train_episodes", 500))
        val_episodes = int(critic_cfg.get("validation_episodes", 100))

        _warmup_vecnormalize_obs_stats(vec_train, warmup_steps)
        sync_envs_normalization(vec_train, vec_eval)

        train_obs, train_targets = _collect_mc_dataset_random_policy(
            vec_env=vec_train, n_episodes=train_episodes, gamma=gamma
        )
        val_obs, val_targets = _collect_mc_dataset_random_policy(
            vec_env=vec_eval, n_episodes=val_episodes, gamma=gamma
        )

        obs_dim = int(vec_train.observation_space.shape[0])
        assert train_obs.shape[1] == obs_dim and val_obs.shape[1] == obs_dim
        assert train_obs.shape[0] == train_targets.shape[0]
        assert val_obs.shape[0] == val_targets.shape[0]

        pt_stats = _pretrain_allocator_critic(
            model=model,
            train_obs=train_obs,
            train_targets=train_targets,
            val_obs=val_obs,
            val_targets=val_targets,
            critic_cfg=critic_cfg,
        )
        print("[critic_pretraining] summary:", pt_stats)

        _reset_ppo_optimizer_after_pretraining(model)

        vec_train.training = True
        vec_train.norm_reward = True
        vec_eval.training = False
        vec_eval.norm_reward = False
    else:
        print("[run] Critic pretraining disabled by config, skipping directly to PPO training")

    agent_dir = os.path.dirname(os.path.abspath(__file__))
    tb_log_name, model_path, best_model_dir, saved_models_dir = _build_run_id_paths(agent_dir, config)

    print(f"[run] TensorBoard log: {tb_log_name}")
    print(f"[run] Model checkpoint: {model_path}")
    print(f"[run] Best model dir: {best_model_dir}")

    callbacks = _build_callbacks(config, vec_eval, best_model_dir)
    print(f"[run] Registered {len(callbacks)} callbacks for training")

    train_cfg = config.get("training", {})
    total_timesteps = int(train_cfg.get("total_timesteps", 2_000_000))
    agent_cfg = config.get("portfolio_allocator_agent", {})
    verbose = int(agent_cfg.get("verbose", 1))

    print(f"\n[run] Starting training: {total_timesteps} timesteps")
    print(f"[run] Verbose level: {verbose}")

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=tb_log_name
    )
    t1 = time.time()
    elapsed_seconds = round(t1 - t0, 2)
    print(f"[run] Training completed in ({elapsed_seconds / 60:.1f} minutes)")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"[run] Final model saved to {model_path}")

    transformer_cfg = config.get("allocator_transformer", {})
    return {
        "agent": "PAA_control_ablation_ar1_signal",
        "policy": "TransformerAllocatorPolicy",
        "total_timesteps": total_timesteps,
        "elapsed_sec": elapsed_seconds,
        "model_path": model_path,
        "best_model_path": os.path.join(best_model_dir, "best_model.zip"),
        "tb_log_name": tb_log_name,
        "run_id": tb_log_name.split("_config_")[0],
        "config_id": str(train_cfg.get("config_id", "unknown")),
        "n_steps": int(agent_cfg.get("n_steps", 2048)),
        "batch_size": int(agent_cfg.get("batch_size", 256)),
        "n_epochs": int(agent_cfg.get("n_epochs", 6)),
        "gamma": float(agent_cfg.get("gamma", 0.99)),
        "gae_lambda": float(agent_cfg.get("gae_lambda", 0.95)),
        "learning_rate_start": float(agent_cfg.get("learning_rate_start", 3e-4)),
        "learning_rate_end": float(agent_cfg.get("learning_rate_end", 3e-5)),
        "d_model": int(transformer_cfg.get("d_model", 128)),
        "n_heads": int(transformer_cfg.get("n_heads", 8)),
        "n_layers": int(transformer_cfg.get("n_layers", 4)),
        "num_assets": num_assets,
        "ar1_phi": float(ar1_cfg.get("phi", 0.9)),
        "ar1_sigma": float(ar1_cfg.get("sigma", 0.15)),
        "ar1_clip": float(ar1_cfg.get("clip", 1.0)),
        "ar1_action_limiting_factor": float(ar1_cfg.get("action_limiting_factor", 0.3)),
        "training_completed": True
    }


def continue_run(cache: MarketDataCache, config: Dict[str, Any], model_path: str, saved_models_dir: str, model_dir_name: str) -> Dict[str, Any]:
    """Continue training from a saved AR(1)-ablation PPO checkpoint (same convention as the hierarchical PAA)."""
    seed = int(config.get("training", {}).get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    gamma_cfg = config.get("portfolio_allocator_agent", {}).get("gamma", 0.99)
    num_assets = cache.num_assets
    ar1_cfg = config.get("ar1_ablation", {})

    print("[continue_run] Building training and evaluation environments (AR(1) control ablation - no SAA loaded)...")
    vec_train_ar1, vec_eval_ar1 = _build_ar1_wrapped_envs(
        cache=cache, config=config, seed=seed, num_assets=num_assets, ar1_cfg=ar1_cfg, tag="continue_run",
    )

    vecnorm_path = os.path.join(saved_models_dir, model_dir_name, "vecnormalize_stats.pkl")
    if not os.path.isfile(vecnorm_path):
        stem = os.path.splitext(os.path.basename(model_path))[0]
        fallback = os.path.join(saved_models_dir, model_dir_name, f"{stem}_vecnormalize.pkl")
        if not os.path.isfile(fallback):
            raise FileNotFoundError(f"VecNormalize stats not found at {vecnorm_path} nor {fallback}")
        vecnorm_path = fallback

    vec_train = VecNormalize.load(vecnorm_path, venv=vec_train_ar1)
    vec_train.training = True
    vec_train.norm_reward = True

    vec_eval = VecNormalize.load(vecnorm_path, venv=vec_eval_ar1)
    vec_eval.training = False
    vec_eval.norm_reward = False

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = PPO.load(model_path, env=vec_train)

    tb_log_name = model_dir_name
    best_model_dir = os.path.join(saved_models_dir, model_dir_name)
    callbacks = _build_callbacks(config, vec_eval, best_model_dir)
    print(f"[continue_run] Registered {len(callbacks)} callbacks for training")

    train_cfg = config.get("training", {})
    total_timesteps = int(train_cfg.get("total_timesteps", 2_000_000))
    agent_cfg = config.get("portfolio_allocator_agent", {})
    verbose = int(agent_cfg.get("verbose", 1))

    print(f"\n[continue_run] Continuing training: {total_timesteps} timesteps")
    print(f"[continue_run] Verbose level: {verbose}")

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=tb_log_name
    )
    t1 = time.time()
    elapsed_seconds = round(t1 - t0, 2)
    print(f"[continue_run] Training completed in ({elapsed_seconds / 60:.1f} minutes)")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"[continue_run] Updated model saved to {model_path}")

    transformer_cfg = config.get("allocator_transformer", {})
    return {
        "agent": "PAA_control_ablation_ar1_signal (CONTINUED)",
        "policy": "TransformerAllocatorPolicy",
        "continued_from_model": model_dir_name,
        "total_timesteps": total_timesteps,
        "elapsed_sec": elapsed_seconds,
        "model_path": model_path,
        "best_model_path": os.path.join(best_model_dir, "best_model.zip"),
        "tb_log_name": tb_log_name,
        "config_id": str(train_cfg.get("config_id", "unknown")),
        "n_steps": int(agent_cfg.get("n_steps", 2048)),
        "batch_size": int(agent_cfg.get("batch_size", 256)),
        "n_epochs": int(agent_cfg.get("n_epochs", 6)),
        "gamma": float(agent_cfg.get("gamma", 0.99)),
        "gae_lambda": float(agent_cfg.get("gae_lambda", 0.95)),
        "learning_rate_start": float(agent_cfg.get("learning_rate_start", 3e-4)),
        "learning_rate_end": float(agent_cfg.get("learning_rate_end", 3e-5)),
        "d_model": int(transformer_cfg.get("d_model", 128)),
        "n_heads": int(transformer_cfg.get("n_heads", 8)),
        "n_layers": int(transformer_cfg.get("n_layers", 4)),
        "num_assets": num_assets,
        "training_continued": True
    }
