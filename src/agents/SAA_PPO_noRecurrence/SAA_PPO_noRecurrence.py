"""Feedforward PPO single-asset target-position agent.

This agent keeps the SAA environment, structured observation contract, asset-ID
embedding, reward, action scaling, validation callbacks, and checkpoint format
used by the recurrent SAA agent. The only architectural change is replacing
RecurrentPPO/MultiInputLstmPolicy with PPO/MultiInputPolicy: the feature MLP is
followed directly by separate actor and critic heads.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from src.agents.RecurrPPO_target_position_agent import recurr_ppo_target_pos_agent as recurrent_saa


# Reuse the established environment, feature extractor, schedules, callbacks,
# validation logic, and checkpoint helper. Only the algorithm construction below
# is feedforward-specific.
build_env = recurrent_saa.build_env
InputMLPFeatures = recurrent_saa.InputMLPFeatures
linear_three_phase_schedule = recurrent_saa.linear_three_phase_schedule
EntropyScheduleCallback = recurrent_saa.EntropyScheduleCallback
EpisodePortfolioSB3LoggerCallback = recurrent_saa.EpisodePortfolioSB3LoggerCallback
ProgressSyncCallback = recurrent_saa.ProgressSyncCallback
EvalCallbackWithMetrics = recurrent_saa.EvalCallbackWithMetrics
save_checkpoint_with_vecnormalize = recurrent_saa.save_checkpoint_with_vecnormalize


def build_policy_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build PPO policy kwargs while preserving the SAA feature architecture.

    ``InputMLPFeatures`` produces the shared feature vector. SB3's
    ``MultiInputPolicy`` then creates the separate ``pi`` and ``vf`` MLP heads
    directly from that vector. Recurrent-only LSTM kwargs are intentionally
    removed before constructing PPO.
    """
    policy_kwargs = recurrent_saa.build_policy_kwargs(config)
    policy_kwargs.pop("n_lstm_layers", None)
    policy_kwargs.pop("lstm_hidden_size", None)
    return policy_kwargs


def build_model(env, config: Dict[str, Any], seed: Optional[int] = None) -> PPO:
    """Instantiate feedforward PPO with the recurrent agent's hyperparameters."""
    agent_cfg = config.get("agent", {})

    learning_rate = linear_three_phase_schedule(
        start=float(agent_cfg.get("learning_rate_start", 3e-4)),
        end=float(agent_cfg.get("learning_rate_end", 3e-5)),
        warmup_pct=float(agent_cfg.get("lr_schedule_warmup_pct", 0.2)),
        ramping_pct=float(agent_cfg.get("lr_schedule_ramping_pct", 0.6)),
    )
    clip_range = linear_three_phase_schedule(
        start=float(agent_cfg.get("clip_range_start", 0.2)),
        end=float(agent_cfg.get("clip_range_end", 0.1)),
        warmup_pct=float(agent_cfg.get("clip_schedule_warmup_pct", 0.2)),
        ramping_pct=float(agent_cfg.get("clip_schedule_ramping_pct", 0.6)),
    )

    return PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        ent_coef=float(agent_cfg.get("ent_coef_start", 0.01)),
        clip_range=clip_range,
        target_kl=float(agent_cfg.get("target_kl", 0.03)),
        n_steps=int(agent_cfg.get("n_steps", 2048)),
        n_epochs=int(agent_cfg.get("n_epochs", 6)),
        batch_size=int(agent_cfg.get("batch_size", 256)),
        gamma=float(agent_cfg.get("gamma", 0.99)),
        gae_lambda=float(agent_cfg.get("gae_lambda", 0.95)),
        vf_coef=float(agent_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(agent_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs=build_policy_kwargs(config),
        device=str(agent_cfg.get("device", "auto")),
        seed=seed,
        normalize_advantage=bool(agent_cfg.get("normalize_advantage", True)),
        verbose=int(agent_cfg.get("verbose", 1)),
        tensorboard_log=r"src\agents\SAA_PPO_noRecurrence\tb_logs",
        stats_window_size=int(agent_cfg.get("stats_window_size", 100)),
    )


def build_eval_callback(eval_env, config: Dict[str, Any], log_dir: str) -> BaseCallback:
    """Reuse the recurrent agent's validation and checkpoint-selection callback."""
    return recurrent_saa.build_eval_callback(eval_env, config, log_dir)


def _make_vectorized_envs(cache, config: Dict[str, Any], seed: int):
    """Create train/evaluation environments with the existing SAA semantics."""
    train_cfg = config.get("training", {})
    n_envs = int(train_cfg.get("n_envs", 1))
    if n_envs < 1:
        raise ValueError(f"training.n_envs must be >= 1, got {n_envs}")

    start_method = str(train_cfg.get("vec_env_start_method", "spawn"))
    if start_method not in ("spawn", "fork", "forkserver"):
        raise ValueError(
            "training.vec_env_start_method must be one of "
            f"('spawn', 'fork', 'forkserver'), got '{start_method}'"
        )

    def make_train(rank: int):
        def _init():
            return build_env(cache, config, seed=seed + rank, for_eval=False)
        return _init

    def make_eval():
        return build_env(cache, config, seed=seed + 1, for_eval=True)

    train_env_fns = [make_train(rank) for rank in range(n_envs)]
    if n_envs == 1:
        base_train_vec = DummyVecEnv(train_env_fns)
    else:
        base_train_vec = SubprocVecEnv(train_env_fns, start_method=start_method)

    base_eval_vec = DummyVecEnv([make_eval])
    gamma = float(config.get("agent", {}).get("gamma", 0.99))
    vec_train = VecNormalize(
        base_train_vec,
        norm_obs=True,
        norm_obs_keys=["numeric"],
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=np.inf,
        gamma=gamma,
    )
    vec_eval = VecNormalize(
        base_eval_vec,
        training=False,
        norm_obs=True,
        norm_obs_keys=["numeric"],
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=np.inf,
        gamma=gamma,
    )
    return vec_train, vec_eval


def _make_callbacks(config: Dict[str, Any], eval_env, log_dir: str):
    """Build the same evaluation, schedule, logging, and progress callbacks."""
    eval_callback = build_eval_callback(eval_env, config, log_dir)
    agent_cfg = config.get("agent", {})
    entropy_callback = None
    if all(
        key in agent_cfg
        for key in (
            "ent_coef_start",
            "ent_coef_end",
            "ent_coef_schedule_warmup_pct",
            "ent_coef_schedule_ramping_pct",
        )
    ):
        entropy_callback = EntropyScheduleCallback(
            start=float(agent_cfg["ent_coef_start"]),
            end=float(agent_cfg["ent_coef_end"]),
            warmup_pct=float(agent_cfg["ent_coef_schedule_warmup_pct"]),
            ramping_pct=float(agent_cfg["ent_coef_schedule_ramping_pct"]),
        )

    train_cfg = config.get("training", {})
    train_logger = EpisodePortfolioSB3LoggerCallback(
        tag_prefix="train",
        log_freq=int(train_cfg.get("train_log_freq", 1)),
    )
    callbacks = [eval_callback, ProgressSyncCallback(), train_logger]
    if entropy_callback is not None:
        callbacks.insert(1, entropy_callback)
    return callbacks


def _new_run_paths(config: Dict[str, Any]):
    """Create the same run naming and checkpoint directories as the SAA agent."""
    agent_dir = os.path.dirname(__file__)
    run_id_file = os.path.join("src", "data", "run_id.json")
    with open(run_id_file, "r") as file:
        run_id_data = json.load(file)

    current_run_id = int(run_id_data.get("run_id", 0))
    with open(run_id_file, "w") as file:
        json.dump({"run_id": current_run_id + 1}, file)

    run_id = str(current_run_id).zfill(5)
    config_id = str(config.get("training", {}).get("config_id", "00000")).zfill(5)
    date_str = datetime.now().strftime("%y_%m_%d")
    run_name = f"{run_id}_config_{config_id}_{date_str}"
    saved_models_dir = os.path.join(agent_dir, "saved_models")
    best_model_dir = os.path.join(saved_models_dir, run_name)
    os.makedirs(best_model_dir, exist_ok=True)
    return run_name, saved_models_dir, best_model_dir


def run(cache, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the non-recurrent SAA PPO agent."""
    seed = int(config.get("training", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    vec_train, vec_eval = _make_vectorized_envs(cache, config, seed)
    model = build_model(vec_train, config, seed=seed)
    run_name, saved_models_dir, best_model_dir = _new_run_paths(config)
    callbacks = _make_callbacks(config, vec_eval, best_model_dir)
    total_timesteps = int(config.get("training", {}).get("total_timesteps", 300_000))

    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=bool(config.get("training", {}).get("progress_bar", True)),
        tb_log_name=run_name,
    )
    elapsed = time.time() - start_time

    model_path = os.path.join(saved_models_dir, f"{run_name}.zip")
    model_base_name = os.path.splitext(os.path.basename(model_path))[0]
    save_checkpoint_with_vecnormalize(
        model=model,
        save_dir=os.path.dirname(model_path),
        checkpoint_name=model_base_name,
        require_vecnormalize=True,
    )
    final_model_path, final_vecnormalize_path = save_checkpoint_with_vecnormalize(
        model=model,
        save_dir=best_model_dir,
        checkpoint_name="final_model_last",
        require_vecnormalize=True,
    )

    agent_cfg = config.get("agent", {})
    return {
        "agent": "SAA_PPO_noRecurrence",
        "policy": "MultiInputPolicy",
        "total_timesteps": total_timesteps,
        "elapsed_sec": round(elapsed, 2),
        "model_path": model_path,
        "final_model_path": final_model_path,
        "final_vecnormalize_path": final_vecnormalize_path,
        "tb_log_name": run_name,
        "config_id": str(config.get("training", {}).get("config_id", "unknown")),
        "n_steps": int(agent_cfg.get("n_steps", 2048)),
        "batch_size": int(agent_cfg.get("batch_size", 256)),
        "gamma": float(agent_cfg.get("gamma", 0.99)),
        "gae_lambda": float(agent_cfg.get("gae_lambda", 0.95)),
    }


def continue_run(
    cache,
    config: Dict[str, Any],
    model_path: str,
    saved_models_dir: str,
    model_dir_name: str,
) -> Dict[str, Any]:
    """Continue training a feedforward PPO checkpoint with matching statistics."""
    seed = int(config.get("training", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    vec_train, base_vec_eval = _make_vectorized_envs(cache, config, seed)
    base_train_vec = vec_train.venv
    vecnorm_path = os.path.join(
        saved_models_dir,
        model_dir_name,
        f"{os.path.splitext(os.path.basename(model_path))[0]}_vecnormalize.pkl",
    )
    if not os.path.isfile(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize stats not found at {vecnorm_path}")

    vec_train = VecNormalize.load(vecnorm_path, venv=base_train_vec)
    vec_eval = VecNormalize.load(vecnorm_path, venv=base_vec_eval.venv)
    vec_eval.training = False
    vec_eval.norm_reward = False

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = PPO.load(model_path, env=vec_train)

    best_model_dir = os.path.join(saved_models_dir, model_dir_name)
    callbacks = _make_callbacks(config, vec_eval, best_model_dir)
    total_timesteps = int(config.get("training", {}).get("total_timesteps", 300_000))
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=bool(config.get("training", {}).get("progress_bar", True)),
        tb_log_name=model_dir_name,
    )
    elapsed = time.time() - start_time

    save_checkpoint_with_vecnormalize(
        model=model,
        save_dir=os.path.dirname(model_path),
        checkpoint_name=os.path.splitext(os.path.basename(model_path))[0],
        require_vecnormalize=True,
    )
    final_model_path, final_vecnormalize_path = save_checkpoint_with_vecnormalize(
        model=model,
        save_dir=best_model_dir,
        checkpoint_name="final_model_last",
        require_vecnormalize=True,
    )

    agent_cfg = config.get("agent", {})
    return {
        "agent": "SAA_PPO_noRecurrence (CONTINUED)",
        "policy": "MultiInputPolicy",
        "continued_from_model": model_dir_name,
        "total_timesteps": total_timesteps,
        "elapsed_sec": round(elapsed, 2),
        "model_path": model_path,
        "final_model_path": final_model_path,
        "final_vecnormalize_path": final_vecnormalize_path,
        "tb_log_name": model_dir_name,
        "config_id": str(config.get("training", {}).get("config_id", "unknown")),
        "n_steps": int(agent_cfg.get("n_steps", 2048)),
        "batch_size": int(agent_cfg.get("batch_size", 256)),
        "gamma": float(agent_cfg.get("gamma", 0.99)),
        "gae_lambda": float(agent_cfg.get("gae_lambda", 0.95)),
    }
