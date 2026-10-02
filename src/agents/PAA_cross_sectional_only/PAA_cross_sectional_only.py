"""
--- PAA Cross-Sectional-Only Ablation: Zero-Valued Signal (PPO + SB3) ---

Cross-sectional-only ablation for the Portfolio Allocator Agent (PAA) described in the
thesis. Implemented as the zero-valued special case of the same shadow-signal-injection
mechanism used by the hierarchical system and by the AR(1) control ablation
(PAA_with_autoregressive_rnd_walk_SAA.py): the signal injected into each asset's token is
held at a constant zero rather than drawn from the SAA or the AR(1) process, and the
corresponding shadow sub-portfolio is simply never traded. This agent never loads or runs
the frozen Single-Asset Agent (SAA) either.

This is a deliberate design choice, not an implementation shortcut. Implementing this
ablation as a genuinely smaller network, one whose asset token omits the SAA-signal
dimension entirely, would confound two different explanations for any performance gap: a
loss of information, and a loss of model capacity. Holding the signal at zero instead
keeps this ablation, the control ablation, and the hierarchical system at identical
architecture and identical parameter count, differing only in what occupies two input
columns of the asset token. This isolates the information content of the (AR(1) or SAA)
signal specifically.

Everything else (environment, reward, friction model, purged evaluation protocol,
Transformer tokenizer/policy, training/eval callbacks, schedules, run-id/checkpoint
conventions) is reused unmodified from ppo_portfolio_allocator_weights_agent.py and from
PAA_with_autoregressive_rnd_walk_SAA.py so all three systems stay directly comparable via
their TensorBoard logs.
"""

import os
import time
import numpy as np
import torch

from typing import Dict, Any, Mapping

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecNormalize, DummyVecEnv, SubprocVecEnv, VecEnvWrapper, VecEnv, sync_envs_normalization
)

import gymnasium as gym

from environment.trading_environment import MarketDataCache

# Reuse the hierarchical PAA's building blocks unmodified: same tokenizer, same
# Transformer policy, same PPO construction, same callbacks/schedules, same env factory.
from agents.PPO_portfolio_allocator_weights.ppo_portfolio_allocator_weights_agent import (
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

# Reuse the fully generic run-id/checkpoint-path and callback-wiring helpers from the
# AR(1) control ablation as-is; neither depends on how the injected signal is produced.
from agents.PAA_with_autoregressive_rnd_walk_SAA.PAA_with_autoregressive_rnd_walk_SAA import (
    _build_run_id_paths,
    _build_callbacks,
)


class ZeroSignalWrapper(VecEnvWrapper):
    """
    Cross-sectional-only ablation replacement for SAASignalWrapper/AR1SignalWrapper.

    Injects a constant zero for both per-asset columns (signal, shadow holding
    percentage) - the same +2-dim-per-asset observation augmentation as the hierarchical
    system and the control ablation, so SAATokenizer/TransformerAllocatorPolicy require
    no changes and parameter count stays identical. Unlike AR1SignalWrapper, this wrapper
    never calls TradingEnv.apply_saa_sub_actions: each asset's shadow sub-portfolio shares
    therefore stay at their initial 0 for the whole episode (TradingEnv's own per-step
    shadow-book MTM update runs unconditionally and only ever applies risk-free cash
    carry to an untouched position), i.e. the shadow sub-portfolio is genuinely never
    traded rather than merely traded with a zero action.
    """

    def __init__(self, venv: VecEnv, num_assets: int, feature_to_index: Mapping[str, int]):
        super().__init__(venv)

        self.num_assets = num_assets

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
                "Invalid observation layout for ZeroSignalWrapper. "
                f"obs_len={obs_len}, num_assets={self.num_assets}, "
                f"raw_feat_dim={self.raw_feat_dim}, portfolio_dim={self.portfolio_dim}"
            )

        self.n_envs = int(venv.num_envs)
        self._zero_cache: Dict[int, np.ndarray] = {}

        # Resize obs space: add +2 features per asset (zero signal + zero shadow holding %),
        # mirroring SAASignalWrapper/AR1SignalWrapper exactly so SAATokenizer's slicing stays valid.
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

    def _zero_block(self, B: int) -> np.ndarray:
        """Cached (B, N, 2) zero block reused every step; the signal is deterministic so no per-step compute is needed."""
        block = self._zero_cache.get(B)
        if block is None:
            block = np.zeros((B, self.num_assets, 2), dtype=np.float32)
            self._zero_cache[B] = block
        return block

    def reset(self):
        res = self.venv.reset()
        obs = res[0] if isinstance(res, tuple) and len(res) == 2 else res
        return self._inject_zeros(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        for info in infos:
            if info.get("terminal_observation", None) is not None:
                info["terminal_observation"] = self._inject_zeros(info["terminal_observation"][None, ...])[0]

        # No apply_saa_sub_actions call here: the shadow sub-portfolios are never traded.
        return self._inject_zeros(obs), rewards, dones, infos

    def _inject_zeros(self, obs: np.ndarray) -> np.ndarray:
        B = obs.shape[0]
        asset_block = self.num_assets * self.raw_feat_dim
        asset_feats = obs[:, :asset_block].reshape(B, self.num_assets, self.raw_feat_dim)
        portfolio_part = obs[:, asset_block:]
        augmented_assets = np.concatenate([asset_feats, self._zero_block(B)], axis=-1).reshape(B, -1)
        return np.concatenate([augmented_assets, portfolio_part], axis=1)


def _build_zero_signal_wrapped_envs(
    cache: MarketDataCache,
    config: Dict[str, Any],
    seed: int,
    num_assets: int,
    tag: str,
):
    """Build the parallel train/eval vector envs and wrap both in zero-signal injection."""
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
        f"No SAA model is loaded or run in this ablation; the shadow sub-portfolios are never traded."
    )

    vec_train_zero = ZeroSignalWrapper(vec_train_raw, num_assets, cache.feature_to_index)
    vec_eval_zero = ZeroSignalWrapper(vec_eval_raw, num_assets, cache.feature_to_index)
    return vec_train_zero, vec_eval_zero


def run(cache: MarketDataCache, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the cross-sectional-only ablation trainer (called by main.py exactly
    like the hierarchical PAA and the AR(1) control ablation). Never loads or runs a
    frozen SAA; the per-asset signal fed to the Transformer tokenizer is a constant zero.
    """
    seed = int(config.get("training", {}).get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    gamma_cfg = config.get("portfolio_allocator_agent", {}).get("gamma", 0.99)
    num_assets = cache.num_assets
    raw_feature_dim = cache.num_features

    critic_cfg = config.get("critic_pretraining", {})
    do_pretrain = bool(critic_cfg.get("enabled", False))

    print("[run] Building training and evaluation environments (cross-sectional-only ablation - no SAA loaded, zero signal)...")
    vec_train_zero, vec_eval_zero = _build_zero_signal_wrapped_envs(
        cache=cache, config=config, seed=seed, num_assets=num_assets, tag="run",
    )

    vec_train = VecNormalize(
        vec_train_zero, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
        gamma=gamma_cfg, training=True,
    )
    vec_eval = VecNormalize(
        vec_eval_zero, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0,
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
        "agent": "PAA_cross_sectional_only_ablation",
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
        "training_completed": True
    }


def continue_run(cache: MarketDataCache, config: Dict[str, Any], model_path: str, saved_models_dir: str, model_dir_name: str) -> Dict[str, Any]:
    """Continue training from a saved cross-sectional-only-ablation PPO checkpoint (same convention as the hierarchical PAA)."""
    seed = int(config.get("training", {}).get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    gamma_cfg = config.get("portfolio_allocator_agent", {}).get("gamma", 0.99)
    num_assets = cache.num_assets

    print("[continue_run] Building training and evaluation environments (cross-sectional-only ablation - no SAA loaded, zero signal)...")
    vec_train_zero, vec_eval_zero = _build_zero_signal_wrapped_envs(
        cache=cache, config=config, seed=seed, num_assets=num_assets, tag="continue_run",
    )

    vecnorm_path = os.path.join(saved_models_dir, model_dir_name, "vecnormalize_stats.pkl")
    if not os.path.isfile(vecnorm_path):
        stem = os.path.splitext(os.path.basename(model_path))[0]
        fallback = os.path.join(saved_models_dir, model_dir_name, f"{stem}_vecnormalize.pkl")
        if not os.path.isfile(fallback):
            raise FileNotFoundError(f"VecNormalize stats not found at {vecnorm_path} nor {fallback}")
        vecnorm_path = fallback

    vec_train = VecNormalize.load(vecnorm_path, venv=vec_train_zero)
    vec_train.training = True
    vec_train.norm_reward = True

    vec_eval = VecNormalize.load(vecnorm_path, venv=vec_eval_zero)
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
        "agent": "PAA_cross_sectional_only_ablation (CONTINUED)",
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
