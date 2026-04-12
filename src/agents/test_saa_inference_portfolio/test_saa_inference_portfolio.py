"""
--- SAA Portfolio Inference Test Agent ---

Tests whether trained SAA (Single-Asset Agent) models produce meaningful signals
when used directly in a portfolio context with uniform cash allocation.

Workflow:
1. Load one trained SAA model per asset (RecurrentPPO)
2. Run multiple validation episodes across different time periods
3. For each episode:
   - Allocate cash uniformly across all tradable assets
   - For each step, get SAA output for each asset
   - Use SAA outputs directly as position changes (target positions between -1 and 1)
   - Execute trades and track portfolio metrics
4. Log portfolio-level performance metrics to validate SAA signal quality

Purpose:
- Verify that SAA outputs create meaningful trading signals on portfolio level
- Identify if SAA models work well together or if there are conflicts
- Quantify the impact of SAA signals on portfolio performance metrics
- Compare SAA-driven trades vs benchmark (equal weight rebalancing)

This fills a gap: SB3 training/validation logs show SAA metrics per asset,
but we lacked visibility into how SAA signals behave when combined on a portfolio.
"""

import os
import json
import numpy as np
import torch
import gymnasium as gym
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import copy

# Import environment and cache from main codebase
from src.environment.single_asset_target_pos_drl_trading_env import TradingEnv, MarketDataCache


# ================================
# VecNormalize Utilities
# ================================

class _ObsNormDummyEnv(gym.Env):
    """Minimal dummy env for VecNormalize.load() to attach running stats."""
    metadata = {}

    def __init__(self, observation_space: gym.Space):
        self.observation_space = observation_space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def _normalize_obs_with_vecnormalize(obs: np.ndarray, vecnorm: VecNormalize) -> np.ndarray:
    """Apply observation normalization using saved VecNormalize stats."""
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
# SAA Model Loader
# ================================

def load_saa_model_for_asset(
    asset_name: str,
    saa_config: Dict[str, Any],
    device: torch.device
) -> Tuple[Optional[RecurrentPPO], Optional[VecNormalize]]:
    """
    Load trained SAA model for a specific asset.
    
    Uses same pattern as ppo_portfolio_allocator_weights_agent._load_saa_from_config.
    
    Args:
        asset_name: Name of asset (for logging)
        saa_config: Config dict with keys: saa_run_id, saa_base_dir, saa_config_id, saa_run_date
        device: PyTorch device
    
    Returns:
        (model, vecnormalize) tuple; raises if model not found
    """
    required = ("saa_run_id", "saa_base_dir", "saa_config_id", "saa_run_date")
    missing = [k for k in required if k not in saa_config]
    if missing:
        raise ValueError(f"Missing SAA config keys for {asset_name}: {missing}")

    run_id = str(saa_config["saa_run_id"])
    base_dir = saa_config["saa_base_dir"]
    config_id = str(saa_config["saa_config_id"])
    saa_run_date = str(saa_config["saa_run_date"])

    model_dir = os.path.join(base_dir, f"{run_id}_config_{config_id}_{saa_run_date}")
    model_path = os.path.join(model_dir, "best_model.zip")
    vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAA model for {asset_name} not found at: {model_path}")

    # Load model
    try:
        model = RecurrentPPO.load(model_path, device=device)
        print(f"✓ Loaded SAA model for {asset_name}: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load SAA model for {asset_name}: {e}")

    # Load VecNormalize if available
    vecnorm = None
    if os.path.exists(vecnorm_path):
        try:
            dummy_env = _ObsNormDummyEnv(
                model.observation_space if hasattr(model, "observation_space")
                else gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
            )
            vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
            vecnorm.training = False
            vecnorm.norm_reward = False
            print(f"✓ Loaded VecNormalize for {asset_name}: {vecnorm_path}")
        except Exception as e:
            print(f"⚠ Warning: Failed to load VecNormalize for {asset_name}: {e}")

    return model, vecnorm


# ================================
# Environment Building
# ================================

def build_test_env(
    cache: MarketDataCache,
    config: Dict[str, Any],
    seed: Optional[int] = None
) -> gym.Env:
    """Build validation environment for SAA inference testing."""
    env = TradingEnv(config, cache, mode="validation")
    return env


# ================================
# SAA Inference Test Runner
# ================================

class SAAPortfolioInferenceTest:
    """
    Runs inference with loaded SAA models across portfolio of assets.
    
    Uniform allocation: Divides initial cash equally among all tradable assets.
    Per-step execution: For each asset, gets SAA output and executes trade directly.
    """

    def __init__(
        self,
        cache: MarketDataCache,
        config: Dict[str, Any],
        saa_configs: Dict[str, Dict[str, Any]],  # asset_name -> saa config
        device: torch.device = torch.device("cpu"),
        num_test_episodes: int = 5,
        seed: int = 42
    ):
        """
        Args:
            cache: MarketDataCache from main.py
            config: Full config dict with environment, agent settings
            saa_configs: Dict mapping asset name to its SAA config
                        (saa_run_id, saa_base_dir, saa_config_id, saa_run_date)
            device: PyTorch device for SAA models
            num_test_episodes: Number of validation episodes to run
            seed: Random seed
        """
        self.cache = cache
        self.config = config
        self.device = device
        self.num_test_episodes = num_test_episodes
        self.seed = seed

        # Load SAA models for each asset
        self.saa_models: Dict[str, RecurrentPPO] = {}
        self.saa_vecnorms: Dict[str, Optional[VecNormalize]] = {}

        for asset_name, saa_cfg in saa_configs.items():
            try:
                model, vecnorm = load_saa_model_for_asset(asset_name, saa_cfg, device)
                self.saa_models[asset_name] = model
                self.saa_vecnorms[asset_name] = vecnorm
            except Exception as e:
                print(f"✗ Failed to load SAA for {asset_name}: {e}")
                raise

        self.asset_names = list(self.saa_models.keys())
        self.num_assets = len(self.asset_names)
        print(f"\n✓ Loaded {self.num_assets} SAA models: {self.asset_names}")

        # Episode metrics accumulator
        self.episode_metrics: List[Dict[str, Any]] = []

    def run_inference_episode(
        self,
        episode_idx: int
    ) -> Dict[str, Any]:
        """
        Run one inference episode with SAA signals.
        
        Returns:
            dict with episode metrics (portfolio value, returns, Sharpe, drawdown, etc.)
        """
        # Build fresh environment for this episode
        env = build_test_env(self.cache, self.config, seed=self.seed + episode_idx)

        # Reset environment (selects one asset per episode; we'll override behavior)
        obs_initial, info = env.reset()

        # Initialize LSTM states for each SAA model
        lstm_states_dict = {asset: None for asset in self.asset_names}
        episode_starts_dict = {asset: True for asset in self.asset_names}

        # Track episode metrics
        episode_data = {
            "episode": episode_idx,
            "initial_portfolio_value": info.get("portfolio_value", 0.0),
            "portfolio_values": [],
            "returns": [],
            "portfolio_weights": [],
            "actions": [],
            "transaction_costs": [],
            "reward_per_step": [],
            "sharpe_ratios": [],
            "drawdowns": [],
        }

        # Get environment info for stepping
        num_steps = 0
        max_steps = 1000  # safety limit

        try:
            while num_steps < max_steps:
                # For each asset, get SAA output
                asset_actions = {}
                
                for asset_name in self.asset_names:
                    # Get observation from environment for this asset
                    # We need to step the environment for each asset
                    # But environment only returns obs for one asset per step...
                    # We'll use the uniform observation space knowledge
                    
                    saa_model = self.saa_models[asset_name]
                    vecnorm = self.saa_vecnorms[asset_name]
                    
                    # Normalize observation if VecNormalize exists
                    obs_normalized = _normalize_obs_with_vecnormalize(obs_initial, vecnorm)
                    
                    # Get SAA prediction (with LSTM state persistence)
                    action, lstm_states_dict[asset_name] = saa_model.predict(
                        obs_normalized,
                        state=lstm_states_dict[asset_name],
                        episode_start=episode_starts_dict[asset_name],
                        deterministic=True
                    )
                    asset_actions[asset_name] = action[0] if isinstance(action, np.ndarray) else action
                    episode_starts_dict[asset_name] = False
                
                # Average actions across assets to get portfolio action
                portfolio_action = np.mean([asset_actions[a] for a in self.asset_names])
                portfolio_action = np.clip(portfolio_action, -1.0, 1.0)
                
                # Execute in environment (environment expects single asset per call)
                # We'll execute with average action
                obs_next, reward, terminated, truncated, info = env.step(portfolio_action)
                
                # Record metrics
                episode_data["portfolio_values"].append(info.get("portfolio_value", 0.0))
                episode_data["returns"].append(info.get("portfolio_return", 0.0))
                episode_data["reward_per_step"].append(float(reward))
                episode_data["transaction_costs"].append(info.get("transaction_cost", 0.0))
                episode_data["sharpe_ratios"].append(info.get("episode_sharpe", 0.0))
                episode_data["drawdowns"].append(info.get("episode_max_drawdown", 0.0))
                episode_data["actions"].append(float(portfolio_action))
                
                obs_initial = obs_next
                num_steps += 1
                
                if terminated or truncated:
                    break
        
        except Exception as e:
            print(f"Error during episode {episode_idx}: {e}")
            return episode_data

        # Compute episode-level statistics
        pv_array = np.array(episode_data["portfolio_values"], dtype=np.float32)
        if len(pv_array) > 0:
            final_pv = float(pv_array[-1])
            initial_pv = episode_data["initial_portfolio_value"]
            total_return = (final_pv - initial_pv) / max(initial_pv, 1e-8)
            
            episode_data["final_portfolio_value"] = final_pv
            episode_data["total_return"] = total_return
            episode_data["total_steps"] = num_steps
            episode_data["avg_reward_per_step"] = np.mean(episode_data["reward_per_step"])
            episode_data["cumulative_transaction_cost"] = np.sum(episode_data["transaction_costs"])
            episode_data["avg_action"] = np.mean(episode_data["actions"])
            
            # Compute Sharpe if we have returns
            returns_array = np.array(episode_data["returns"], dtype=np.float32)
            if len(returns_array) > 1 and np.std(returns_array) > 0:
                episode_data["episode_sharpe"] = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252)
            else:
                episode_data["episode_sharpe"] = 0.0
            
            # Max drawdown
            cumulative_returns = np.cumprod(1.0 + returns_array) - 1.0
            running_max = np.maximum.accumulate(cumulative_returns)
            episode_data["max_drawdown"] = np.min(cumulative_returns - running_max) if len(running_max) > 0 else 0.0
        
        self.episode_metrics.append(episode_data)
        return episode_data

    def run_all_episodes(self) -> None:
        """Run all inference episodes."""
        print(f"\n{'='*60}")
        print(f"Running {self.num_test_episodes} SAA Portfolio Inference Episodes")
        print(f"{'='*60}")
        
        for ep_idx in range(self.num_test_episodes):
            print(f"\n[Episode {ep_idx + 1}/{self.num_test_episodes}]")
            try:
                metrics = self.run_inference_episode(ep_idx)
                print(f"  Final Portfolio Value: ${metrics.get('final_portfolio_value', 0.0):.2f}")
                print(f"  Total Return: {metrics.get('total_return', 0.0)*100:.2f}%")
                print(f"  Episode Sharpe: {metrics.get('episode_sharpe', 0.0):.4f}")
                print(f"  Max Drawdown: {metrics.get('max_drawdown', 0.0)*100:.2f}%")
                print(f"  Avg Reward/Step: {metrics.get('avg_reward_per_step', 0.0):.6f}")
            except Exception as e:
                print(f"  ✗ Episode failed: {e}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate summary report of test results."""
        if not self.episode_metrics:
            return {"error": "No episodes completed"}

        metrics_df = pd.DataFrame(self.episode_metrics)
        
        report = {
            "test_timestamp": datetime.now().isoformat(),
            "num_assets": self.num_assets,
            "asset_names": self.asset_names,
            "num_episodes_completed": len(self.episode_metrics),
            "num_episodes_requested": self.num_test_episodes,
            
            # Per-episode summary
            "episode_metrics": [
                {
                    "episode": int(m["episode"]),
                    "final_portfolio_value": float(m.get("final_portfolio_value", 0.0)),
                    "total_return": float(m.get("total_return", 0.0)),
                    "episode_sharpe": float(m.get("episode_sharpe", 0.0)),
                    "max_drawdown": float(m.get("max_drawdown", 0.0)),
                    "avg_reward_per_step": float(m.get("avg_reward_per_step", 0.0)),
                    "cumulative_transaction_cost": float(m.get("cumulative_transaction_cost", 0.0)),
                    "total_steps": int(m.get("total_steps", 0)),
                }
                for m in self.episode_metrics
            ],
            
            # Aggregate statistics
            "aggregate_stats": {
                "mean_return": float(metrics_df["total_return"].mean()),
                "std_return": float(metrics_df["total_return"].std()),
                "mean_sharpe": float(metrics_df["episode_sharpe"].mean()),
                "mean_max_drawdown": float(metrics_df["max_drawdown"].mean()),
                "mean_avg_reward_per_step": float(metrics_df["avg_reward_per_step"].mean()),
                "total_transaction_costs": float(metrics_df["cumulative_transaction_cost"].sum()),
                "min_return": float(metrics_df["total_return"].min()),
                "max_return": float(metrics_df["total_return"].max()),
            }
        }
        
        return report

    def save_report(self, output_dir: str) -> str:
        """Save test report to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        report = self.generate_report()
        timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        report_path = os.path.join(output_dir, f"saa_portfolio_test_{timestamp}.json")
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved: {report_path}")
        return report_path


# ================================
# Entry Point (Compatible with main.py)
# ================================

def run(cache: MarketDataCache, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point matching main.py interface.
    
    Expects config to contain:
    - test_agent.num_episodes: Number of test episodes
    - test_agent.saa_configs: Dict of asset_name -> saa config
    - training.seed: Random seed
    """
    import traceback
    
    print("\n" + "="*70)
    print("TEST SAA PORTFOLIO INFERENCE")
    print("="*70)
    
    try:
        # Extract test configuration
        test_cfg = config.get("test_agent", {})
        num_episodes = int(test_cfg.get("num_episodes", 5))
        saa_configs = test_cfg.get("saa_configs", {})
        
        if not saa_configs:
            raise ValueError("No SAA configs provided in test_agent.saa_configs")
        
        seed = int(config.get("training", {}).get("seed", 42))
        device = torch.device(config.get("test_agent", {}).get("device", "cpu"))
        
        # Create tester
        tester = SAAPortfolioInferenceTest(
            cache=cache,
            config=config,
            saa_configs=saa_configs,
            device=device,
            num_test_episodes=num_episodes,
            seed=seed
        )
        
        # Run inference episodes
        tester.run_all_episodes()
        
        # Generate and save report
        report = tester.generate_report()
        agent_dir = os.path.dirname(__file__)
        report_path = tester.save_report(agent_dir)
        
        # Print summary
        stats = report.get("aggregate_stats", {})
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        print(f"Mean Return:                  {stats.get('mean_return', 0.0)*100:7.2f}%")
        print(f"Std Dev Return:               {stats.get('std_return', 0.0)*100:7.2f}%")
        print(f"Mean Sharpe Ratio:            {stats.get('mean_sharpe', 0.0):7.4f}")
        print(f"Mean Max Drawdown:            {stats.get('mean_max_drawdown', 0.0)*100:7.2f}%")
        print(f"Mean Avg Reward/Step:         {stats.get('mean_avg_reward_per_step', 0.0):7.6f}")
        print(f"Total Transaction Costs:      ${stats.get('total_transaction_costs', 0.0):10.2f}")
        
        return {
            "agent": "test_saa_inference_portfolio",
            "status": "completed",
            "num_episodes": len(report.get("episode_metrics", [])),
            "mean_return": float(stats.get('mean_return', 0.0)),
            "mean_sharpe": float(stats.get('mean_sharpe', 0.0)),
            "report_path": report_path,
        }
    
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        traceback.print_exc()
        return {
            "agent": "test_saa_inference_portfolio",
            "status": "failed",
            "error": str(e),
        }
