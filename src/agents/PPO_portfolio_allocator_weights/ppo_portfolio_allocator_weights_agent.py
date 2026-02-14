"""
--- Transformer-based Portfolio Allocator Agent (PPO + SB3) ---

Implements a multi-asset portfolio allocator trained via Stable-Baselines3 PPO.
The allocator uses a Transformer encoder to process per-asset signals from frozen SAAs
(Single-Asset Agents) combined with portfolio-level features.

Architecture:
- Input: Per-asset features (SAA signals + portfolio state + risk metrics) + global portfolio token
- Embedding: Linear projection + asset ID embeddings → d_model dimensions
- Transformer Encoder: Self-attention across N+1 tokens (N assets + 1 portfolio token)
- Output Heads: Per-asset weight logits + cash weight logit → DIRECT softmax output
  (Weights sum to 1.0 directly from policy, not post-hoc normalized)
- Value Head: Portfolio token → scalar value estimate
- PPO Training: Standard (non-recurrent) PPO with Gaussian policy

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
from typing import Callable, Dict, Any, Optional, Tuple, List
from datetime import datetime

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
#from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Normal
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym

from src.environment.single_asset_target_pos_drl_trading_env import TradingEnv as PortfolioEnv
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
# SAA Ensemble (Frozen single-asset agents)
# ================================

class FrozenSAAEnsemble:
    """
    Manages a collection of pre-trained frozen SAA models (one per asset).
    Handles inference-only queries to obtain per-asset signals without gradients.
    Note that the SAAs are stateful wrt time due to LSTM layers!
    
    Each SAA is a RecurrentPPO agent trained in single-asset mode. This ensemble:
    - Loads all N SAA models from disk (.zip files)
    - Maintains separate LSTM hidden states for each asset
    - Provides frozen inference (torch.no_grad()) to extract per-asset signals
    - Resets internal state at episode boundaries
    
    SAA Model Paths Convention:
    - Base directory: src/agents/RecurrPPO_target_position_agent/saved_models/
    - Subdirectory format: {run_id}_config_{config_id}_{date}/best_model.zip
    - run_id: 5-digit string (e.g., "00017") controlling which SAA version to use
    - This allows easy switching via config parameter: "saa_run_id"
    """

    def __init__(
            self, asset_to_model_path: Dict[str, str], 
            vecnormalize_path: Optional[str] = None,device: str = "cpu"
        ):
        """
        Initialize ensemble with paths to trained SAA models.
        
        Args:
            asset_to_model_path: Dict mapping asset symbol → full path to .zip model file
                                 Example: {"SPY": "/path/to/00017.../best_model.zip", ...}
            device: PyTorch device for model inference ("cpu", "cuda", etc.)
        
        Raises:
            FileNotFoundError: If any specified model file does not exist
            RuntimeError: If model loading fails (corrupted .zip, wrong RecurrentPPO format)
        """
        self.device = device
        self.assets = list(asset_to_model_path.keys())
        self.models = {}
        self.lstm_states = {}  # Per-asset LSTM hidden states (cell_state, hidden_state)
        self.vecnormalize = None # Loaded VecNormalize (obs stats) for inference-time normalization
        
        # Load all SAA models from disk
        print(f"[FrozenSAAEnsemble] Loading {len(self.assets)} SAA models...")
        for asset_symbol, model_path in asset_to_model_path.items():
            # Verify file exists before attempting load
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"SAA model for asset '{asset_symbol}' not found at: {model_path}"
                )
            
            try:
                # Load RecurrentPPO model from .zip file
                # RecurrentPPO.load() restores policy, optimizer state, and hyperparameters
                self.models[asset_symbol] = RecurrentPPO.load(
                    model_path,
                    device=self.device
                )
                print(f"  ✓ Loaded SAA for {asset_symbol} from {os.path.basename(model_path)}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SAA model for '{asset_symbol}' from {model_path}: {str(e)}"
                )
            
            # Initialize LSTM hidden states as None; will be set on first reset
            self.lstm_states[asset_symbol] = None
        
        print(f"[FrozenSAAEnsemble] Successfully loaded {len(self.models)} SAA models on device: {device}")

        # Load VecNormalize observation stats (if provided) so frozen inference matches SAA training
        if vecnormalize_path is not None:
            if not os.path.exists(vecnormalize_path):
                print(f"[FrozenSAAEnsemble] WARNING: VecNormalize file not found: {vecnormalize_path}. Proceeding without obs normalization.")
            else:
                try:
                    any_model = next(iter(self.models.values()))
                    obs_space = any_model.observation_space
                    dummy_env = DummyVecEnv([lambda: _ObsNormDummyEnv(obs_space)])
                    self.vecnormalize = VecNormalize.load(vecnormalize_path, dummy_env)
                    self.vecnormalize.training = False
                    self.vecnormalize.norm_reward = False
                    print(f"[FrozenSAAEnsemble] Loaded VecNormalize stats from: {os.path.basename(vecnormalize_path)}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load VecNormalize from {vecnormalize_path}: {str(e)}")

    def reset_episode(self) -> None:
        """
        Reset LSTM hidden states for all SAAs at episode start.
        Called by environment/wrapper at reset() to clear temporal memory.
        
        After reset, the next get_saa_signals() call will start with fresh LSTM state,
        effectively marking the episode boundary for RecurrentPPO inference.
        """
        # Reset all per-asset LSTM hidden states to None
        # RecurrentPPO.predict() interprets None as initialization signal
        for asset in self.assets:
            self.lstm_states[asset] = None
        
        # Optional: Log reset for debugging
        # print(f"[FrozenSAAEnsemble] Reset LSTM states for all {len(self.assets)} assets")

    def get_saa_signals(
        self,
        obs_per_asset: Dict[str, np.ndarray],
        episode_starts: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Obtain target position signal from each SAA via inference.
        
        Runs each frozen SAA forward once per asset to extract its recommended
        position change. Uses torch.no_grad() to disable gradient computation,
        ensuring frozen inference and reduced memory/compute overhead.
        
        Args:
            obs_per_asset: Dict mapping asset symbol → observation array (same format as SAA training)
                          Observation should be shape [obs_dim] (single step, no batch dimension)
            episode_starts: Dict mapping asset symbol → boolean (True if episode boundary, False otherwise)
                           Used to signal LSTM state reset to RecurrentPPO
        
        Returns:
            Dict mapping asset symbol → SAA action scalar (typically [-1, 1] range for target position)
            
        Notes:
            - LSTM states are updated in-place within self.lstm_states; subsequent calls
              maintain temporal continuity (crucial for LSTM memory)
            - torch.no_grad() prevents gradient tracking; models remain frozen
            - Each asset's SAA runs independently (can be parallelized if needed)
        """
        saa_signals = {}
        
        # Disable gradient computation for frozen inference
        with torch.no_grad():
            for asset in self.assets:
                # Get observation for this asset
                obs = _normalize_obs_with_vecnormalize(obs_per_asset[asset], self.vecnormalize)
                
                # Get episode start flag
                episode_start = np.array([episode_starts.get(asset, False)])
                
                # Run frozen SAA to get action (target position change)
                # Args:
                #   obs: observation for this step
                #   state: current LSTM hidden/cell states (None on first step)
                #   episode_start: boolean array marking episode boundary
                #   deterministic: True for inference (use policy mean, not sampled action)
                # Returns:
                #   action: [1] array (target position for single-asset agent)
                #   state: updated LSTM (cell_state, hidden_state) tuple
                action, self.lstm_states[asset] = self.models[asset].predict(
                    obs,
                    state=self.lstm_states[asset],
                    episode_start=episode_start,
                    deterministic=True  # Use policy mean for inference (no sampling noise)
                )
                
                # Extract scalar from action array, verify correctness, and store
                # SAA output is typically in [-1, 1] (position change scaled by action_limiting_factor)
                if not isinstance(action, np.ndarray) or action.shape != (1,):
                    raise RuntimeError(
                        f"SAA action for asset '{asset}' has unexpected shape: {action.shape}"
                    ) 
                 
                # Validate that SAA action is in expected range [-1, 1]
                if not np.all((action >= -1.0) & (action <= 1.0)):
                    raise RuntimeError(
                        f"Warning: SAA action for {asset} outside [-1, 1] range: {action[0]}"
                    )
                saa_signals[asset] = float(action[0])
        
        return saa_signals

    def get_saa_values(
        self,
        obs_per_asset: Dict[str, np.ndarray],
        episode_starts: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Optionally retrieve value estimates from SAA critic heads.
        
        Runs SAA value function (critic) to get per-asset state value estimates.
        Useful for:
        - Confidence weighting of per-asset signals (high value = confident region)
        - Auxiliary loss for allocator training (auxiliary policy distillation)
        - Analysis of SAA uncertainty
        
        Args:
            obs_per_asset: Dict mapping asset symbol → observation array
            episode_starts: Dict mapping asset symbol → episode boundary flag
        
        Returns:
            Dict mapping asset symbol → scalar value estimate from SAA critic
            
        Notes:
            - Does NOT update LSTM states (read-only queries)
            - Use torch.no_grad() to avoid gradient tracking
            - SAA value is normalized by training environment; typically centered around 0
        """
        saa_values = {}
        
        with torch.no_grad():
            for asset in self.assets:
                obs = _normalize_obs_with_vecnormalize(obs_per_asset[asset], self.vecnormalize)
                episode_start = np.array([episode_starts.get(asset, False)])
                
                # Get value estimate from SAA's value function
                # Use .policy to access the underlying policy object and extract value directly
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # RecurrentPPO's policy has forward_lstm and value_net methods
                # We need to extract the value through the policy network
                try:
                    # Access policy's value function directly
                    self.models[asset].policy.set_training_mode(False)
                    _, value, _ = self.models[asset].policy(
                        obs_tensor,
                        state=self.lstm_states[asset],
                        episode_start=episode_start
                        )
                    saa_values[asset] = float(value[0, 0])
                except Exception as e:
                    # Throw error if value extraction fails
                    raise RuntimeError(f"Could not extract value for {asset}: {str(e)}")
        
        return saa_values

    def get_saa_hidden_states(
        self,
        obs_per_asset: Dict[str, np.ndarray],
        episode_starts: Dict[str, bool]
    ) -> Dict[str, np.ndarray]:
        """
        Optionally retrieve compressed LSTM hidden states from SAAs.
        
        Extracts LSTM hidden state vectors (not cell state) from each SAA.
        Can be used as additional features for allocator if desired:
        - Concatenate with SAA signals and asset features
        - Captures LSTM's internal temporal representation
        - May improve allocator's ability to model asset-specific dynamics
        
        Args:
            obs_per_asset: Dict mapping asset symbol → observation array
            episode_starts: Dict mapping asset symbol → episode boundary flag
        
        Returns:
            Dict mapping asset symbol → LSTM hidden state array
            Shape typically [n_lstm_layers * hidden_size] after compression
            
        Notes:
            - Does NOT modify self.lstm_states (read-only)
            - LSTM hidden states capture asset-specific temporal patterns
            - Can be averaged/pooled if multiple LSTM layers exist
            - Advanced feature; optional for basic allocator
        """
        saa_hidden_states = {}
        
        with torch.no_grad():
            for asset in self.assets:
                obs = _normalize_obs_with_vecnormalize(obs_per_asset[asset], self.vecnormalize)
                episode_start = np.array([episode_starts.get(asset, False)])
                
                # Run prediction but extract hidden state from returned state tuple
                _, state_tuple = self.models[asset].predict(
                    obs,
                    state=self.lstm_states[asset],
                    episode_start=episode_start,
                    deterministic=True
                )
                
                # state_tuple is (cell_state, hidden_state) tuple
                # Extract hidden state (second element) and convert to numpy
                if state_tuple is not None:
                    hidden_state = state_tuple[1]  # hidden state
                    if isinstance(hidden_state, torch.Tensor):
                        hidden_state = hidden_state.detach().cpu().numpy()
                    # Flatten if needed (handle multiple LSTM layers)
                    saa_hidden_states[asset] = hidden_state.flatten().astype(np.float32)
                else:
                    # Fallback: return zeros if state extraction fails
                    saa_hidden_states[asset] = np.zeros(1, dtype=np.float32)
                    raise RuntimeError(
                        f"Could not extract hidden state for asset '{asset}'; returning zeros."
                    )
        
        return saa_hidden_states
    

# ================================
# Transformer Policy Network (Custom)
# ================================

class TransformerTokenizer:
    """
    Converts per-asset features and portfolio state into tokenized embeddings.
    Assembles feature vectors and handles normalization/projection.
    """

    def __init__(
        self,
        num_assets: int,
        saa_feature_dim: int,
        asset_feature_dim: int,
        portfolio_feature_dim: int,
        d_model: int

    ):
        """
        Initialize tokenizer with feature dimensions.
        
        Args:
            num_assets: Number of tradable assets (N)
            saa_feature_dim: Dimension of SAA signal (typically 1 for scalar action)
            asset_feature_dim: Dimension of per-asset features (weights, returns, risks, etc.)
            portfolio_feature_dim: Dimension of portfolio/global features
            d_model: Target embedding dimension for transformer
        """
        super().__init__()

        # Store configuration for validation and integration checks
        self.num_assets = num_assets
        self.saa_feature_dim = saa_feature_dim
        self.asset_feature_dim = asset_feature_dim
        self.portfolio_feature_dim = portfolio_feature_dim
        self.d_model = d_model

        # Running statistics (updated step-by-step during rollouts)
        self._portfolio_feature_mean = np.zeros(portfolio_feature_dim, dtype=np.float32)
        self._portfolio_feature_std = np.ones(portfolio_feature_dim, dtype=np.float32)
        self._portfolio_feature_count = 0  # Track number of steps seen

        # Total per-asset input features:
        # [SAA signal] + [portfolio weight for that asset] + [asset_features...]
        self.per_asset_input_dim = saa_feature_dim + 1 + asset_feature_dim

        # Linear projections into transformer model space (d_model)
        # These are standard and integrate cleanly with TransformerEncoder
        self.asset_projection = nn.Linear(self.per_asset_input_dim, d_model)
        self.portfolio_projection = nn.Linear(portfolio_feature_dim, d_model)

        # Maintain a stable feature order across calls for consistency
        self._asset_order: Optional[List[str]] = None
        self._portfolio_feature_keys: Optional[List[str]] = None

        # Optional normalization statistics (placeholders for future enhancement)
        # If you later compute running means/stds, store them here
        self._asset_feature_mean: Optional[np.ndarray] = None
        self._asset_feature_std: Optional[np.ndarray] = None
        self._portfolio_feature_mean: Optional[np.ndarray] = None
        self._portfolio_feature_std: Optional[np.ndarray] = None

    def tokenize_assets(
        self,
        saa_signals: Dict[str, float],
        portfolio_weights: np.ndarray,
        asset_features: Dict[str, np.ndarray]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Create asset token embeddings from signals and features.
        
        Args:
            saa_signals: Dict of per-asset SAA actions
            portfolio_weights: Array of current weights [N]
            asset_features: Dict mapping asset symbol → feature array
        
        Returns:
            (asset_embeddings, asset_order) where embeddings shape [N, d_model]
        """
        # Establish a stable asset order on first call
        if self._asset_order is None:
            self._asset_order = list(asset_features.keys())

        asset_order = self._asset_order

        # Basic integrity checks
        if len(asset_order) != self.num_assets:
            raise ValueError(
                f"Expected {self.num_assets} assets, got {len(asset_order)}"
            )
        if portfolio_weights.shape[0] != self.num_assets:
            raise ValueError(
                f"portfolio_weights length {portfolio_weights.shape[0]} "
                f"does not match num_assets {self.num_assets}"
            )

        # Build per-asset feature vectors
        asset_token_list = []
        for idx, asset in enumerate(asset_order):
            if asset not in asset_features:
                raise KeyError(f"Missing features for asset '{asset}'")
            if asset not in saa_signals:
                raise KeyError(f"Missing SAA signal for asset '{asset}'")

            # Raw inputs
            saa_signal = np.array([saa_signals[asset]], dtype=np.float32)
            asset_weight = np.array([portfolio_weights[idx]], dtype=np.float32)
            asset_feat = np.asarray(asset_features[asset], dtype=np.float32)

            # Validate feature size
            if asset_feat.shape[0] != self.asset_feature_dim:
                raise ValueError(
                    f"Asset feature dim mismatch for '{asset}': "
                    f"expected {self.asset_feature_dim}, got {asset_feat.shape[0]}"
                )

            # Concatenate into a single per-asset vector
            per_asset_vec = np.concatenate([saa_signal, asset_weight, asset_feat], axis=0)

            # # Normalize (currently pass-through unless stats are set)
            # per_asset_vec = self.normalize_features(per_asset_vec, feature_type="asset")

            # Project into model dimension
            per_asset_tensor = torch.as_tensor(per_asset_vec, dtype=torch.float32)
            asset_token = self.asset_projection(per_asset_tensor)
            asset_token_list.append(asset_token)

        # Stack to shape [N, d_model]
        asset_embeddings = torch.stack(asset_token_list, dim=0)

        return asset_embeddings, asset_order

    def tokenize_portfolio(
        self,
        portfolio_state_dict: Dict[str, float]
    ) -> torch.Tensor:
        """
        Create portfolio/global token embedding.
        
        Args:
            portfolio_state_dict: Dict with keys like cash_weight, portfolio_vol, turnover, etc.
        
        Returns:
            Portfolio token embedding shape [d_model]
        """
        # Establish a stable feature order on first call
        if self._portfolio_feature_keys is None:
            self._portfolio_feature_keys = list(portfolio_state_dict.keys())

        # Enforce consistent ordering across steps
        if set(portfolio_state_dict.keys()) != set(self._portfolio_feature_keys):
            raise ValueError(
                "Portfolio feature keys changed across calls. "
                "Keep keys stable to ensure consistent encoding."
            )

        # Build portfolio feature vector in stable key order
        portfolio_vec = np.array(
            [portfolio_state_dict[k] for k in self._portfolio_feature_keys],
            dtype=np.float32
        )

        # Validate feature size
        if portfolio_vec.shape[0] != self.portfolio_feature_dim:
            raise ValueError(
                f"Portfolio feature dim mismatch: expected {self.portfolio_feature_dim}, "
                f"got {portfolio_vec.shape[0]}"
            )

        # Normalize (currently pass-through unless stats are set)
        portfolio_vec = self.normalize_features(portfolio_vec, feature_type="portfolio")

        # Project into model dimension
        portfolio_tensor = torch.as_tensor(portfolio_vec, dtype=torch.float32)
        portfolio_embedding = self.portfolio_projection(portfolio_tensor)

        return portfolio_embedding

    def normalize_features(
        self,
        features: np.ndarray,
        feature_type: str = "asset"
    ) -> np.ndarray:
        """
        Apply online or fixed normalization to feature vectors.
        
        Args:
            features: Raw feature array
            feature_type: "asset" or "portfolio" (determines which scaler to use)
        
        Returns:
            Normalized feature array (same shape)
        """
        # Placeholder normalization: identity transform by default.
        # If you later compute running mean/std, plug them here.
        if feature_type == "asset":
            if self._asset_feature_mean is not None and self._asset_feature_std is not None:
                return (features - self._asset_feature_mean) / (self._asset_feature_std + 1e-8)
            return features
        
        elif feature_type == "portfolio":
            if self._portfolio_feature_mean is not None and self._portfolio_feature_std is not None:
                safe_std = np.maximum(self._portfolio_feature_std, 1e-8)
                return (features - self._portfolio_feature_mean) / safe_std
            return features
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    def update_portfolio_statistics(self, portfolio_features: np.ndarray) -> None:
        """
        Update running mean/std with Welford's online algorithm.
        Called once per environment step (lookahead-safe: only sees current step).
        
        Args:
            portfolio_features: Current step's portfolio feature vector [portfolio_feature_dim]
        """
        # Welford's online algorithm: O(1) memory, O(1) update
        self._portfolio_feature_count += 1
        delta = portfolio_features - self._portfolio_feature_mean
        self._portfolio_feature_mean += delta / self._portfolio_feature_count
        delta2 = portfolio_features - self._portfolio_feature_mean
        
        # Running variance (will divide by count later for unbiased estimate)
        if self._portfolio_feature_count == 1:
            self._portfolio_feature_std = np.zeros_like(portfolio_features, dtype=np.float32)
        else:
            # Incremental variance update
            variance = np.sum(delta * delta2) / self._portfolio_feature_count
            self._portfolio_feature_std = np.sqrt(np.maximum(variance, 1e-8))

class TransformerEncoder(nn.Module):
    """
    Transformer encoder module for processing asset and portfolio tokens.
    Uses standard PyTorch TransformerEncoder with self-attention.
    
    Architecture:
    - Input: N+1 tokens (N assets + 1 portfolio), each [d_model]
    - Processing: Multi-head self-attention across all tokens
    - Output: Encoded tokens with same shape as input
    
    Integration:
    - TransformerTokenizer produces input tokens → TransformerEncoder processes them
    - Used within TransformerAllocatorPolicy to encode asset + portfolio representations
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "relu"
    ):
        """
        Initialize transformer encoder.
        
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network (typically 2x or 4x d_model)
            dropout: Dropout rate
            activation: "relu" or "gelu"
        
        Raises:
            ValueError: If d_model not divisible by n_heads
        """
        
        # Initialize parent nn.Module
        super().__init__()
        
        # Validate that d_model is divisible by n_heads (required by multi-head attention)
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        
        # Store configuration for reference
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        
        # Create individual transformer encoder layers
        # Each layer consists of:
        #   1. Multi-head self-attention (attends to all N+1 tokens)
        #   2. Feed-forward network (position-wise, applied independently to each token)
        #   3. Layer normalization and residual connections (built into TransformerEncoderLayer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,  # We use (T, B, d_model) format (sequence length first)
            norm_first=False    # Apply layer norm after residual (standard Transformer)
        )
        
        # Stack N layers into a complete encoder
        # Each layer can attend to outputs from previous layer
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)  # Final layer norm after all encoder layers
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Process tokens through transformer encoder.
        
        Args:
            tokens: Shape [T, d_model] where T = N+1 (assets + portfolio)
                   - First N tokens: per-asset representations (from TransformerTokenizer)
                   - Last token: portfolio representation
                   - Each token is [d_model] dimension
        
        Returns:
            Encoded tokens, shape [T, d_model]
            - Same shape as input
            - Each token is updated via multi-head self-attention and FFN
            - Maintains positional relationships via attention mechanism
            
        Notes:
            - No batch dimension: handles single observation (T, d_model) format
            - If batch processing needed, reshape to (T, B, d_model) before calling
            - Self-attention allows each token to attend to all other tokens
            - Asset tokens can attend to portfolio context and other assets
            - Portfolio token can attend to all asset signals
        """
        
        # Pass tokens through transformer encoder
        # Self-attention: each token can attend to all tokens (including itself)
        # This allows:
        #   - Each asset token to see other asset signals and portfolio context
        #   - Portfolio token to see all asset signals
        #   - Cross-asset dependencies to be modeled
        encoded_tokens = self.transformer_encoder(tokens)
        
        # Return encoded tokens (same shape as input)
        # encoded_tokens[0:N] = encoded asset tokens (each [d_model])
        # encoded_tokens[N]   = encoded portfolio token ([d_model])
        return encoded_tokens
    

class TransformerAllocatorPolicy(ActorCriticPolicy):
    """
    Custom ActorCriticPolicy combining transformer encoder with PPO heads.
    Inherits from ActorCriticPolicy to provide complete policy interface for SB3.
    
    Architecture:
    1. Transformer Encoder: Processes observation tokens via self-attention
    2. Actor Head: Outputs weight logits → immediately passed to softmax layer
    3. Critic Head: Outputs scalar value estimate
    
    Integration with SB3 PPO:
    - Acts as a complete policy class (no separate features_extractor needed)
    - Implements required methods: forward(), _predict(), _get_action_dist_and_value()
    - Uses Gaussian (Normal) distribution for continuous weight outputs
    - SB3 handles gradient updates, clipping, and training loop
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        lr_schedule: Callable[[float], float],
        num_assets: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        use_asset_id_embedding: bool = True,
        use_portfolio_token: bool = True,
        **kwargs
    ):
        """
        Initialize transformer-based actor-critic policy.
        
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space (should be continuous Box)
            lr_schedule: Learning rate schedule function
            num_assets: Number of tradable assets (N)
            d_model: Transformer model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension in transformer
            dropout: Dropout rate
            use_asset_id_embedding: Whether to use learned asset ID embeddings
            use_portfolio_token: Whether to include dedicated portfolio token
            **kwargs: Additional arguments for parent ActorCriticPolicy
        """
        # Call parent ActorCriticPolicy.__init__()
        # This sets up the base policy infrastructure
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        
        # Store configuration
        self.num_assets = num_assets
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.use_asset_id_embedding = use_asset_id_embedding
        self.use_portfolio_token = use_portfolio_token
        
        # Instantiate transformer encoder for token processing
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu"
        )
        
        # Linear layer to project observation into transformer token embedding space
        # Input: observation vector [obs_dim]
        # Output: token embedding [d_model]
        self.obs_to_tokens = nn.Linear(observation_space.shape[0], d_model)
        
        # Optional: Asset ID embeddings for distinguishing tokens
        if use_asset_id_embedding:
            # Embeddings for N assets + 1 portfolio token
            self.asset_id_embedding = nn.Embedding(num_assets + 1, d_model)
        else:
            self.asset_id_embedding = None
        
        # Actor head: maps transformer output → weight logits (pre-softmax)
        # Output dimension: N+1 (N assets + cash)
        action_dim = action_space.shape[0]
        self.action_head = nn.Linear(d_model, action_dim)

        # Softmax layer: converts logits to valid probability distribution
        # Ensures weights sum to exactly 1.0 across all dimensions
        # This is the only output layer needed - no log_std for categorical-like output
        self.softmax = nn.Softmax(dim=-1)

        # Critic head: maps transformer output → scalar value estimate
        self.value_head = nn.Linear(d_model, 1)

        # NOTE: No log_std parameter needed - softmax output is already bounded [0,1]
        # and sums to 1.0, so we use a Dirichlet-like approach (deterministic softmax)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to extract features and produce weight actions and values.
        
        Args:
            obs: Observation tensor, shape [batch_size, obs_dim]
            deterministic: If True, use softmax output directly (max probability); else sample from Dirichlet
        
        Returns:
            (actions, values, log_probs) tuple
                actions: Portfolio weights [batch_size, action_dim], sum to 1.0
                values: Value estimates, shape [batch_size, 1]
                log_probs: Log probabilities of weight distributions, shape [batch_size]
        """
        # Extract features from observations via transformer
        features = self._extract_features(obs)  # [batch_size, d_model]
        
        # Get weight logits and values
        weight_logits = self.action_head(features)  # [batch_size, N+1]
        values = self.value_head(features)  # [batch_size, 1]
        
        # Convert logits to valid probability distribution via softmax
        # Output: weights ∈ [0,1] that sum to 1.0 across all assets
        weights = self.softmax(weight_logits)  # [batch_size, N+1]
        
        # For policy gradient: use log-softmax to compute log probabilities
        # This is numerically stable and naturally computes the log of the softmax output
        log_softmax = torch.nn.functional.log_softmax(weight_logits, dim=-1)  # [batch_size, N+1]
        
        # Use Dirichlet distribution interpretation:
        # For deterministic, return mean (softmax directly)
        # For stochastic, we could sample from Dirichlet, but for simplicity/stability,
        # use the softmax directly (it's already a valid probability distribution)
        actions = weights  # [batch_size, N+1] - already sums to 1.0
        
        # Compute log probability under softmax (entropy-like measure)
        # This represents the "confidence" of the weight distribution
        # High confidence → low entropy → one weight near 1.0, others near 0
        # Low confidence → high entropy → weights more uniform
        log_probs = (weights * log_softmax).sum(dim=-1)  # [batch_size]
        
        return actions, values, log_probs

    def _extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract transformer features from raw observations.
        
        Args:
            obs: Observation tensor, shape [batch_size, obs_dim]
        
        Returns:
            Features tensor, shape [batch_size, d_model]
        """
        batch_size = obs.shape[0]
        
        # Project observations to token embeddings
        tokens = self.obs_to_tokens(obs)  # [batch_size, d_model]
        
        # Add asset ID embeddings if enabled
        if self.use_asset_id_embedding:
            # Use portfolio token ID (num_assets) for all observations
            asset_ids = torch.full(
                (batch_size,),
                self.num_assets,
                dtype=torch.long,
                device=obs.device
            )
            asset_id_emb = self.asset_id_embedding(asset_ids)  # [batch_size, d_model]
            tokens = tokens + asset_id_emb
        
        # Process through transformer encoder
        # Add sequence dimension: [1, batch_size, d_model]
        tokens_for_transformer = tokens.unsqueeze(0)
        encoded_tokens = self.transformer_encoder(tokens_for_transformer)  # [1, batch_size, d_model]
        
        # Remove sequence dimension
        features = encoded_tokens.squeeze(0)  # [batch_size, d_model]
        
        return features

    def _predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Get actions from the policy (used during rollout collection).
            
        Called by SB3 during collect_rollouts(). Ignores log_probs and values
        since those are handled separately by collect_rollouts().

        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic policy
        
        Returns:
            Action tensor
        """
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions

    def _get_action_dist_and_value(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        Get weight distribution and value estimate for training.
        
        Returns a categorical-like distribution over weight simplex.
        Uses Dirichlet distribution concept: softmax output as concentration parameters.
        """
        features = self._extract_features(obs)
        weight_logits = self.action_head(features)
        values = self.value_head(features)
        
        # Create categorical distribution from unnormalized logits
        # Each asset is a "category" with its own weight
        # Softmax normalizes these into a valid probability distribution
        distribution = torch.distributions.Categorical(logits=weight_logits)
        
        return distribution, values
    

class AllocatorEnvironmentWrapper(gym.Wrapper):
    """
    Wraps the multi-asset portfolio environment to:
    - Query frozen SAAs for per-asset signals
    - Assemble transformer input (SAA signals + asset features + portfolio state)
    - Convert allocator weight outputs to portfolio trades
    
    Integration:
    - Underlying env must be in EXECUTION_PORTFOLIO_WEIGHTS mode
    - FrozenSAAEnsemble provides per-asset signals from frozen RecurrentPPO models
    - TransformerTokenizer assembles features into d_model tokens
    - Allocator outputs N+1 weights (post-softmax) → execute_portfolio_change()
       (Softmax is applied inside policy, not by environment)
    
    Observation Structure:
    The wrapper constructs observations by:
    1. Querying SAA ensemble for per-asset signals (frozen inference)
    2. Extracting current portfolio state (weights, returns, metrics)
    3. Extracting per-asset features from market data cache
    4. Assembling into transformer-ready format via tokenizer
    
    Action Structure:
    - Input: Raw logits [N+1] from allocator policy (unnormalized)
    - Processing: Action is already normalized weights from policy softmax
    - Output: Target portfolio weights [N+1] passed to env.execute_portfolio_change()
       (Softmax is applied inside policy, not by environment)
    """

    def __init__(
        self,
        env: PortfolioEnv,
        saa_ensemble: FrozenSAAEnsemble,
        tokenizer: TransformerTokenizer
    ):
        """
        Initialize environment wrapper.
        
        Args:
            env: Underlying multi-asset trading environment (EXECUTION_PORTFOLIO_WEIGHTS mode)
            saa_ensemble: FrozenSAAEnsemble instance for signal generation
            tokenizer: TransformerTokenizer for feature assembly

        Note: By Convention cash weight is assumed to be index 0 by environment.
        """
        super().__init__(env)
        
        # Store components
        self.saa_ensemble = saa_ensemble
        self.tokenizer = tokenizer
        
        # Validate asset alignment between ensemble and environment
        env_assets = set(env.market_data_cache.asset_names)
        saa_assets = set(saa_ensemble.assets)
        if env_assets != saa_assets:
            raise ValueError(
                f"Asset mismatch between env ({len(env_assets)} assets) "
                f"and SAA ensemble ({len(saa_assets)} assets)"
            )
        
        # Cache asset order for consistent feature extraction
        self._asset_order = env.market_data_cache.asset_names
        self._num_assets = env.market_data_cache.num_assets
        
        # Calculate flattened observation dimension
        # Observation = [saa_signals (N), asset_features (N*num_features), portfolio_features (8)]
        num_features = env.market_data_cache.num_features
        portfolio_feature_dim = 8  # cash_weight, portfolio_value, return, sharpe, drawdown, volatility, turnover, alpha
        
        obs_dim = (
            self._num_assets +  # SAA signals (1 per asset)
            (self._num_assets * num_features) +  # Asset features
            portfolio_feature_dim  # Portfolio-level features
        )
        
        # Override observation space to match flattened output
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        print(f"[AllocatorEnvironmentWrapper] Initialized with {self._num_assets} assets")
        print(f"  SAA models: {', '.join(saa_assets)}")
        print(f"  Flattened observation dimension: {obs_dim}")
        print(f"    - SAA signals: {self._num_assets}")
        print(f"    - Asset features: {self._num_assets} × {num_features} = {self._num_assets * num_features}")
        print(f"    - Portfolio features: {portfolio_feature_dim}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset environment and reset SAA hidden states.
        
        Returns:
            (observation, info) where observation is transformer-ready feature dict
            
        Process:
        1. Reset underlying environment (samples new episode, initializes portfolio)
        2. Reset SAA LSTM hidden states (clear temporal memory for new episode)
        3. Reset tokenizer portfolio statistics (fresh running stats)
        4. Assemble initial observation from SAA signals + current state
        """
        # Reset underlying environment (EXECUTION_PORTFOLIO mode)
        # This samples a new episode block, initializes portfolio with random weights
        obs, info = self.env.reset(seed=seed, option=options)
        
        # Reset SAA ensemble LSTM states (episode boundary for frozen models)
        self.saa_ensemble.reset_episode()
        
        # Reset tokenizer portfolio statistics (fresh running mean/std for new episode)
        self.tokenizer._portfolio_feature_count = 0
        self.tokenizer._portfolio_feature_mean = np.zeros(
            self.tokenizer.portfolio_feature_dim, dtype=np.float32
        )
        self.tokenizer._portfolio_feature_std = np.ones(
            self.tokenizer.portfolio_feature_dim, dtype=np.float32
        )
        
        # Assemble initial observation from current state
        # This queries SAAs and extracts features
        transformer_input = self._assemble_transformer_input()
        
        # Add transformer input details to info for debugging
        info['saa_signals'] = transformer_input['saa_signals']
        info['num_assets'] = self._num_assets
        
        # Return raw observation (not tokenized yet; policy will tokenize)
        # For SB3 integration, we return a flattened observation array
        observation = self._flatten_transformer_input(transformer_input)
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute allocator action (weight logits) and return next state.
        
        Args:
            action: Target weights from allocator policy [N+1]
                   (Already softmax-normalized by policy)
        
        Returns:
            (observation, reward, terminated, truncated, info)
            
        Process:
        1. Use action as target weights (already normalized by softmax)
        2. Execute portfolio rebalancing via env.step()
        3. Update tokenizer portfolio statistics (for next observation)
        4. Assemble next observation from SAA signals + updated state
        5. Return step results
        """
        # Validate action shape
        if action.shape[0] != self._num_assets + 1:
            raise ValueError(
                f"Action shape mismatch: expected {self._num_assets + 1}, "
                f"got {action.shape[0]}"
            )
        
        # Action is already normalized weights from softmax [0, 1] summing to 1.0
        # No additional normalization needed - softmax guarantees this mathematically
        action = np.asarray(action, dtype=np.float32)

        # Validation: verify weights sum to 1.0 (allow small numerical tolerance)
        action_sum = np.sum(action)
        if not (1-1e-4 <= action_sum <= 1+1e-4):
            # This should NEVER happen with proper softmax in policy
            # But clamp for safety to ensure sum = 1.0
            action = action / np.maximum(action_sum, 1e-8)
            print(f"[Warning] Action sum was {action_sum:.6f}, re-normalized to 1.0")

        # Verify no negative weights (softmax guarantees this, but double-check)
        if np.any(action < -1e-6):
            print(f"[Warning] Negative weights detected: {action[action < 0]}, clipping to 0")
            action = np.maximum(action, 0.0)
            action = action / np.sum(action)  # Re-normalize after clipping

        # Execute rebalancing in underlying environment
        # env.step() in EXECUTION_PORTFOLIO_WEIGHTS mode:
        # - Receives target weights [N+1] that sum to 1.0
        # - Validates weights
        # - Calls execute_portfolio_change() with exact weight targets
        # - Advances time by one day
        # - Calculates reward
        # - Records metrics
        obs, reward, terminated, truncated, info = self.env.step(action)

        """ 
        Important note on why the obs is disregarded here:
        The environment doesn't know about frozen SAAs—it's a generic trading environment. 
        Only the wrapper knows:
        - Which SAA models to query
        - How to tokenize data for the transformer
        - How to extract portfolio-specific features
        - So the wrapper reconstructs a custom observation optimized for the transformer-based allocator policy.
        """
        
        # Update tokenizer portfolio statistics BEFORE assembling next observation
        # Extract current portfolio features for running statistics update
        portfolio_features = self._extract_portfolio_features()
        self.tokenizer.update_portfolio_statistics(portfolio_features)
        
        # Assemble next observation from updated state
        transformer_input = self._assemble_transformer_input()
        
        # Add transformer input details to info for debugging
        info['saa_signals'] = transformer_input['saa_signals']
        
        # Return flattened observation
        observation = self._flatten_transformer_input(transformer_input)
        
        return observation, reward, terminated, truncated, info

    def _assemble_transformer_input(self) -> Dict[str, Any]:
        """
        Query SAAs and assemble full transformer input from environment state.
        
        Returns:
            Dict with keys:
                - saa_signals: Dict[asset_symbol, float] - SAA target position signals
                - asset_features: Dict[asset_symbol, np.ndarray] - Per-asset market features
                - portfolio_state: Dict[str, float] - Portfolio-level features
                
        Process:
        1. Extract per-asset observations from environment
        2. Query SAA ensemble for frozen inference (per-asset signals)
        3. Extract per-asset features from market data cache
        4. Extract portfolio-level features from episode buffer
        5. Return structured dict for tokenizer consumption
        """
        # Step 1: Extract per-asset observations for SAA inference
        obs_per_asset = {}
        episode_starts = {}
        
        for asset_symbol in self._asset_order:
            # Extract observation for this asset in SAA's expected format
            # SAAs were trained in SINGLE_ASSET_TARGET_POS mode with specific obs structure
            obs_per_asset[asset_symbol] = self._extract_per_asset_observation(asset_symbol)
            
            # Episode starts are False after first step (LSTM continuity)
            episode_starts[asset_symbol] = (self.env.current_step == 0)
        
        # Step 2: Query SAA ensemble for per-asset signals (frozen inference)
        saa_signals = self.saa_ensemble.get_saa_signals(
            obs_per_asset=obs_per_asset,
            episode_starts=episode_starts
        )
        
        # Step 3: Extract per-asset features from market data cache
        # These are the selected technical indicators, returns, volumes, etc.
        asset_features = {}
        current_features = self.env.market_data_cache.get_features_at_step(
            self.env.current_absolute_step
        )  # Shape: [num_assets, num_features]
        
        for idx, asset_symbol in enumerate(self._asset_order):
            asset_features[asset_symbol] = current_features[idx]
        
        # Step 4: Extract portfolio-level features from episode buffer
        # These include: weights, returns, sharpe, drawdown, volatility, turnover
        portfolio_state = self._extract_portfolio_features()
        
        # Return structured dict
        return {
            'saa_signals': saa_signals,
            'asset_features': asset_features,
            'portfolio_state': portfolio_state
        }

    def _extract_per_asset_observation(self, asset_symbol: str) -> np.ndarray:
        """
        Extract single-asset observation for a given asset (for SAA input).
        
        Mimics the observation structure SAAs were trained on:
        - Asset features: [num_selected_features] from market data
        - Portfolio features: [cash_weight, asset_weight, portfolio_return]
        
        Args:
            asset_symbol: Asset identifier
        
        Returns:
            Observation array compatible with SAA input format
            Shape: [num_selected_features + 3]
        """
        # Get asset index
        asset_idx = self.env.market_data_cache.asset_to_index[asset_symbol]
        
        # Extract asset features from market data cache
        # Shape: [num_selected_features]
        asset_features = self.env.market_data_cache.get_features_at_step(
            self.env.current_absolute_step
        )[asset_idx]
        
        # Extract portfolio features from episode buffer
        # Map current step to internal buffer index
        external_step = self.env.current_step
        if self.env.maybe_provide_sequence:
            internal_step = external_step + self.env.lookback_window
        else:
            internal_step = external_step
        
        # # Clamp to valid buffer range
        # internal_step = int(np.clip(
        #     internal_step, 
        #     0, 
        #     self.env.episode_buffer.portfolio_values.shape[0] - 1
        # ))
        
        # Extract weights: index 0 is cash, index (asset_idx + 1) is this asset
        weights_vec = self.env.episode_buffer.portfolio_weights[internal_step]
        cash_weight = float(weights_vec[0])
        asset_weight = float(weights_vec[asset_idx + 1])

        rel_cash_weight = cash_weight / (cash_weight + asset_weight + 1e-8)
        rel_asset_weight = asset_weight / (cash_weight + asset_weight + 1e-8)
        
        # Extract asset return over last step. SAA used daily portfolio return as proxy.
        asset_prices = self.env.market_data_cache.close_prices
        abs_step = self.env.current_absolute_step
        prev_step = max(abs_step - 1, 0)
        px_t = float(asset_prices[abs_step, asset_idx])
        px_prev = float(asset_prices[prev_step, asset_idx])
        asset_return = (px_t / px_prev - 1.0) if px_prev > 0 else 0.0

        portfolio_features = np.array([
            rel_cash_weight,
            rel_asset_weight,
            asset_return
        ], dtype=np.float32)
        
        # Concatenate asset + portfolio features
        observation = np.concatenate([
            asset_features,
            portfolio_features
        ]).astype(np.float32)
        
        # Guard against non-finite values
        if not np.all(np.isfinite(observation)):
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
            print(
                f"Warning: Non-finite values in observation for {asset_symbol} "
                f"at step {self.env.current_step}. Replaced with zeros."
            )
        
        return observation

    def _extract_portfolio_features(self) -> np.ndarray:
        """
        Extract portfolio-level features as a dict for tokenizer.
        
        Returns:
            Dict with portfolio metrics:
                - cash_weight: float
                - portfolio_value: float
                - portfolio_return: float
                - sharpe_ratio: float (rolling)
                - max_drawdown: float (rolling)
                - volatility: float (rolling annualized)
                - turnover: float (current step)
                - Various other metrics from episode buffer
                
        NOTE: Keys must be stable across calls for tokenizer feature order consistency
        """
        # Map current step to internal buffer index
        external_step = self.env.current_step
        if self.env.maybe_provide_sequence:
            internal_step = external_step + self.env.lookback_window
        else:
            internal_step = external_step
        
        # Clamp to valid buffer range
        internal_step = int(np.clip(
            internal_step,
            0,
            self.env.episode_buffer.portfolio_values.shape[0] - 1
        ))
        
        # Extract metrics from episode buffer
        weights = self.env.episode_buffer.portfolio_weights[internal_step]
        cash_weight = float(weights[0])
        
        portfolio_value = float(self.env.episode_buffer.portfolio_values[internal_step])
        portfolio_return = float(self.env.episode_buffer.returns[internal_step])
        
        # Rolling metrics (use small window at start of episode)
        risk_window = min(max(self.env.current_step, 2), self.env.max_reward_risk_window)
        
        sharpe_ratio = float(self.env.episode_buffer.calculate_sharpe_ratio(risk_window))
        max_drawdown = float(self.env.episode_buffer.calculate_max_drawdown(risk_window))
        
        # Volatility
        recent_returns = self.env.episode_buffer.get_returns_window(risk_window)
        if len(recent_returns) > 1:
            volatility = float(np.std(recent_returns) * np.sqrt(252))  # Annualized
        else:
            volatility = 0.0
        
        # Turnover (current step)
        turnover = float(self.env.episode_buffer.turnover[internal_step])
        
        # Alpha (excess return over benchmark)
        alpha = float(self.env.episode_buffer.alpha[internal_step])
        
        # Assemble into numpy array with stable key order
        # This order must match what tokenizer expects
        portfolio_features = np.array([
            cash_weight,
            portfolio_value,
            portfolio_return,
            sharpe_ratio,
            max_drawdown,
            volatility,
            turnover,
            alpha
        ], dtype=np.float32)
        
        return portfolio_features

    def _flatten_transformer_input(self, transformer_input: Dict[str, Any]) -> np.ndarray:
        """
        Flatten transformer input dict into a single observation array for SB3.
        
        This is a SIMPLIFIED approach where we concatenate all features into a flat vector.
        The actual tokenization happens inside TransformerAllocatorPolicy.forward().
        
        Args:
            transformer_input: Dict with saa_signals, asset_features, portfolio_state
        
        Returns:
            Flattened observation array [obs_dim]
            
        Structure:
            [saa_signals (N), asset_features (N * num_features), portfolio_features (num_portfolio_features)]
        """
        saa_signals = transformer_input['saa_signals']
        asset_features = transformer_input['asset_features']
        portfolio_state = transformer_input['portfolio_state']
        
        # Extract SAA signals in stable order
        saa_signal_vec = np.array([
            saa_signals[asset] for asset in self._asset_order
        ], dtype=np.float32)
        
        # Extract asset features in stable order and flatten
        asset_feature_matrix = np.array([
            asset_features[asset] for asset in self._asset_order
        ], dtype=np.float32)  # Shape: [N, num_features]
        asset_feature_vec = asset_feature_matrix.flatten()
        
        # Portfolio features are already a flat array
        portfolio_feature_vec = portfolio_state
        
        # Concatenate all components
        observation = np.concatenate([
            saa_signal_vec,
            asset_feature_vec,
            portfolio_feature_vec
        ]).astype(np.float32)
        
        # Guard against non-finite values
        if not np.all(np.isfinite(observation)):
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
            print(
                f"Warning: Non-finite values in flattened observation "
                f"at step {self.env.current_step}. Replaced with zeros."
            )
        
        return observation
    

# ================================
# Portfolio-Level Reward Computation
# ================================

class AllocatorRewardCalculator:
    """
    Computes portfolio-level rewards using differential Sortino approach.
    Mirrors SAA reward logic but operates on full portfolio returns.
    
    This calculator implements the same reward mechanism as used for single-asset agents,
    but at the portfolio level. The reward is based on:
    1. Differential Sortino ratio (primary component)
    2. Raw portfolio return (secondary component)
    3. Drawdown penalty (risk management)
    
    The Sortino ratio focuses on downside volatility rather than total volatility,
    making it more appropriate for trading strategies where upside volatility is desirable.
    
    Key Features:
    - EMA-based running statistics for mean return and downside variance
    - Differential approach: reward = current_sortino - previous_sortino
    - Mixed reward: combines Sortino with raw returns
    - Drawdown penalty based on rolling window
    - Tanh compression for bounded rewards
    
    Integration:
    - Reset at episode start (clear running statistics)
    - Update at each step with portfolio return
    - Compute step reward based on current state
    """

    def __init__(
        self,
        sortino_eta: float = 0.018,
        sortino_downside_var_floor: float = 1e-6,
        sortino_gain: float = 3.5,
        sortino_net_reward_mix: float = 0.7,
        lambda_drawdown: float = 0.5
    ):
        """
        Initialize reward calculator with Sortino parameters.
        
        Args:
            sortino_eta: EMA adaptation rate for mean/downside variance (typically 0.01-0.02)
                        Higher values = faster adaptation to recent returns
            sortino_downside_var_floor: Floor for downside variance to avoid division by zero
                                       Typically 1e-6
            sortino_gain: Gain factor for tanh compression (amplification before squashing)
                         Higher values = more sensitive to small changes
                         Typically 2.0-4.0
            sortino_net_reward_mix: Alpha in [0,1] weighting Sortino vs raw return
                                   1.0 = pure Sortino, 0.0 = pure return
                                   Typically 0.6-0.8
            lambda_drawdown: Penalty factor for drawdown increases
                            Higher values = stronger penalty for drawdowns
                            Typically 0.3-0.7
        """
        # Store hyperparameters
        self.sortino_eta = float(sortino_eta)
        self.sortino_downside_var_floor = float(sortino_downside_var_floor)
        self.sortino_gain = float(sortino_gain)
        self.sortino_net_reward_mix = float(np.clip(sortino_net_reward_mix, 0.0, 1.0))
        self.lambda_drawdown = float(lambda_drawdown)
        
        # Running statistics (EMA-based)
        # These track long-term mean return and downside variance
        self.running_mean_ema = 1e-4  # Small positive initial value
        self.running_downside_variance_ema = 2.5e-5  # Small positive initial variance
        
        # Previous Sortino value for differential calculation
        self.previous_sortino = 0.0
        
        # Track previous max drawdown for penalty calculation
        self.previous_max_drawdown = 0.0
        
        print(f"[AllocatorRewardCalculator] Initialized with:")
        print(f"  sortino_eta: {self.sortino_eta}")
        print(f"  sortino_gain: {self.sortino_gain}")
        print(f"  sortino_net_reward_mix: {self.sortino_net_reward_mix}")
        print(f"  lambda_drawdown: {self.lambda_drawdown}")

    def reset_episode(self) -> None:
        """
        Reset running statistics at episode start.
        
        Called by AllocatorEnvironmentWrapper at the beginning of each episode
        to clear temporal memory and start fresh.
        
        This ensures each episode starts with clean slate for EMA tracking,
        preventing information leakage across episodes.
        """
        # Reset running statistics to initial values
        self.running_mean_ema = 1e-4
        self.running_downside_variance_ema = 2.5e-5
        
        # Reset differential tracking
        self.previous_sortino = 0.0
        
        # Reset drawdown tracking
        self.previous_max_drawdown = 0.0

    def compute_step_reward(
        self,
        portfolio_return: float,
        current_max_drawdown: float,
        current_step: int,
        max_reward_risk_window: int
    ) -> float:
        """
        Compute reward for single step using differential Sortino.
        
        This is the core reward calculation matching the environment's implementation.
        The reward is computed as:
        1. Update running mean and downside variance with EMA
        2. Calculate current Sortino ratio
        3. Compute differential: current_sortino - previous_sortino
        4. Mix Sortino with raw return
        5. Apply drawdown penalty
        6. Compress with tanh
        
        Args:
            portfolio_return: Daily return at this step (simple return, e.g., 0.01 for 1%)
            current_max_drawdown: Current rolling max drawdown from episode buffer
            current_step: Current step number in episode (for risk window calculation)
            max_reward_risk_window: Maximum window size for risk metrics
        
        Returns:
            Scalar reward (typically bounded in [-1, 1] due to tanh compression)
            
        Notes:
            - This method updates internal state (running_mean_ema, running_downside_variance_ema)
            - Call once per step in sequential order
            - Do not call in parallel or out of order
        """
        # 1. Update EMA of mean return
        # delta = current_return - current_mean
        # new_mean = current_mean + eta * delta
        delta = portfolio_return - self.running_mean_ema
        self.running_mean_ema += self.sortino_eta * delta
        
        # 2. Update EMA of downside variance
        # Only penalize negative returns (downside risk)
        # downside_sq = (min(return, 0))^2
        downside_sq = (min(portfolio_return, 0.0)) ** 2
        
        # EMA update: new_var = old_var + eta * (downside_sq - old_var)
        self.running_downside_variance_ema += self.sortino_eta * (
            downside_sq - self.running_downside_variance_ema
        )
        
        # 3. Calculate current Sortino ratio
        # Sortino = mean_return / sqrt(downside_variance)
        # Apply floor to avoid division by zero
        downside_var = max(self.running_downside_variance_ema, self.sortino_downside_var_floor)
        current_sortino = self.running_mean_ema / np.sqrt(downside_var)
        
        # 4. Compute differential Sortino (this is the main reward signal)
        # Differential approach rewards improvement in Sortino, not absolute level
        # This encourages continuous improvement and avoids saturation
        sortino_reward = current_sortino - self.previous_sortino
        self.previous_sortino = current_sortino
        
        # 5. Calculate drawdown penalty
        # Penalize increases in drawdown (delta from previous max)
        # This discourages risky behavior that increases drawdown
        max_drawdown_delta = max(0.0, current_max_drawdown - self.previous_max_drawdown)
        self.previous_max_drawdown = current_max_drawdown
        max_drawdown_penalty = self.lambda_drawdown * max_drawdown_delta
        
        # 6. Mix Sortino with raw return
        # sortino_net_reward_mix controls the balance:
        # - Higher values (0.7-0.9): More weight on risk-adjusted return (Sortino)
        # - Lower values (0.3-0.5): More weight on raw return
        # This prevents the agent from being too conservative or too aggressive
        reward = (
            self.sortino_net_reward_mix * sortino_reward +
            (1.0 - self.sortino_net_reward_mix) * portfolio_return
        ) - max_drawdown_penalty
        
        # 7. Compress and amplify with tanh
        # tanh(gain * x) bounds reward to [-1, 1] while amplifying small values
        # gain controls sensitivity: higher gain = more sensitive to small changes
        reward = float(np.tanh(self.sortino_gain * reward))
        
        return reward

    def update_running_statistics(self, portfolio_return: float) -> None:
        """
        Update EMA of mean return and downside variance.
        
        This method is for external tracking only (not used in reward calculation).
        The actual updates happen inside compute_step_reward().
        
        Args:
            portfolio_return: Daily return to incorporate
            
        Notes:
            - This is a convenience method for diagnostics/logging
            - Not required for reward calculation (which updates internally)
            - Can be used to pre-warm statistics if needed
        """
        # Update mean
        delta = portfolio_return - self.running_mean_ema
        self.running_mean_ema += self.sortino_eta * delta
        
        # Update downside variance
        downside_sq = (min(portfolio_return, 0.0)) ** 2
        self.running_downside_variance_ema += self.sortino_eta * (
            downside_sq - self.running_downside_variance_ema
        )
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get current running statistics for diagnostics/logging.
        
        Returns:
            Dict with current state:
                - running_mean_ema: Current EMA of mean return
                - running_downside_variance_ema: Current EMA of downside variance
                - previous_sortino: Last computed Sortino ratio
                - previous_max_drawdown: Last observed max drawdown
        """
        return {
            'running_mean_ema': float(self.running_mean_ema),
            'running_downside_variance_ema': float(self.running_downside_variance_ema),
            'previous_sortino': float(self.previous_sortino),
            'previous_max_drawdown': float(self.previous_max_drawdown)
        }
    

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


# ================================
# Allocator Environment Builder
# ================================

def build_allocator_env(
    cache: MarketDataCache,
    config: Dict[str, Any],
    saa_ensemble: FrozenSAAEnsemble,
    seed: Optional[int] = None,
    for_eval: bool = False
) -> gym.Env:
    """
    Build a portfolio allocator environment with SAA ensemble and tokenizer.
    
    Args:
        cache: MarketDataCache instance
        config: Configuration dict with all settings
        saa_ensemble: FrozenSAAEnsemble for signal generation
        seed: Random seed
        for_eval: If True, use validation data blocks; else training blocks
    
    Returns:
        Wrapped environment ready for SB3 training/evaluation
    """
    # ------------------------------
    # Validate required inputs
    # ------------------------------
    if cache is None:
        raise ValueError("MarketDataCache is required to build allocator environment.")
    if config is None:
        raise ValueError("Config dict is required to build allocator environment.")
    if saa_ensemble is None:
        raise ValueError("FrozenSAAEnsemble is required to build allocator environment.")

    # ------------------------------
    # Select train vs validation mode
    # ------------------------------
    mode = "validation" if for_eval else "train"

    # ------------------------------
    # Instantiate base portfolio environment
    # ------------------------------
    # Uses the same TradingEnv as the single-asset agent, but in portfolio execution mode.
    base_env = PortfolioEnv(config=config, market_data_cache=cache, mode=mode)

    # Guard: allocator requires portfolio execution mode
    # (TradingEnv stores this as a string in config["environment"]["execution_mode"])
    if getattr(base_env, "execution_mode", None) != "portfolio_weights":
        raise ValueError(
            "AllocatorEnvironmentWrapper requires execution_mode='portfolio_weights'. "
            f"Got: {getattr(base_env, 'execution_mode', None)}"
        )

    # ------------------------------
    # Build tokenizer (uses cache + config)
    # ------------------------------
    tokenizer = build_tokenizer(cache=cache, config=config)

    # ------------------------------
    # Wrap base env with allocator wrapper (injects SAA signals + portfolio features)
    # ------------------------------
    wrapped_env = AllocatorEnvironmentWrapper(
        env=base_env,
        saa_ensemble=saa_ensemble,
        tokenizer=tokenizer
    )

    # ------------------------------
    # Optional seeding for reproducibility
    # ------------------------------
    if seed is not None:
        try:
            wrapped_env.action_space.seed(seed)
            wrapped_env.observation_space.seed(seed)
        except Exception:
            # Not all spaces implement seeding; ignore gracefully
            pass

    return wrapped_env


def load_saa_models(config: Dict[str, Any], device: str = "cpu") -> FrozenSAAEnsemble:
    """
    Load all pre-trained SAA models from paths specified in config.
    
    This function constructs model paths for each asset based on the config structure:
    - Base directory: src/agents/RecurrPPO_target_position_agent/saved_models/
    - Run directory: {saa_run_id}_config_{saa_config_id}_{date}/
    - Model file: best_model.zip
    
    The function automatically discovers all available run directories matching the
    specified run_id and config_id, then selects the most recent one (by date).
    
    Args:
        config: Config dict with "saa_config" section containing:
                - saa_run_id: 5-digit run identifier (e.g., "00017")
                - saa_config_id: 5-digit config identifier (e.g., "00006")
                - saa_base_dir: Base directory for saved models
                - device: PyTorch device ("cpu", "cuda", etc.)
                AND "environment" section with:
                - asset list from market_data_cache
        device: PyTorch device override (default from config if not specified)
    
    Returns:
        FrozenSAAEnsemble instance with all loaded SAA models
        
    Raises:
        ValueError: If config missing required keys
        FileNotFoundError: If no matching SAA models found
        RuntimeError: If model loading fails
        
    Notes:
        - Assumes all assets share the same SAA run_id and config_id
        - Uses best_model.zip from the most recent matching directory
        - Model paths follow convention: {run_id}_config_{config_id}_{date}/best_model.zip
        
    """
    # Extract SAA configuration
    saa_config = config.get("saa_config", {})
    if not saa_config:
        raise ValueError(
            "Config missing 'saa_config' section. Required keys: "
            "saa_run_id, saa_config_id, saa_base_dir"
        )
    
    # Get required parameters
    saa_run_id = saa_config.get("saa_run_id")
    saa_config_id = saa_config.get("saa_config_id")
    saa_base_dir = saa_config.get("saa_base_dir")
    saa_device = saa_config.get("device", device)  # Use config device or override
    
    # Validate required parameters
    if not saa_run_id:
        raise ValueError("Config missing 'saa_run_id' in saa_config section")
    if not saa_config_id:
        raise ValueError("Config missing 'saa_config_id' in saa_config section")
    if not saa_base_dir:
        raise ValueError("Config missing 'saa_base_dir' in saa_config section")
    
    # Normalize run_id and config_id to 5-digit format
    saa_run_id = str(saa_run_id).zfill(5)
    saa_config_id = str(saa_config_id).zfill(5)
    
    # Construct base directory path (relative to project root)
    # Handle both absolute and relative paths
    if not os.path.isabs(saa_base_dir):
        # Get project root (3 levels up from this file)
        current_file = os.path.abspath(__file__)
        agent_dir = os.path.dirname(current_file)
        agents_dir = os.path.dirname(agent_dir)
        src_dir = os.path.dirname(agents_dir)
        project_root = os.path.dirname(src_dir)
        saa_base_dir = os.path.join(project_root, saa_base_dir)
    
    # Verify base directory exists
    if not os.path.exists(saa_base_dir):
        raise FileNotFoundError(
            f"SAA base directory not found: {saa_base_dir}"
        )
    
    # Pattern to match: {run_id}_config_{config_id}_{date}/
    # Example: 00017_config_00006_26_01_21/
    pattern = f"{saa_run_id}_config_{saa_config_id}_"
    
    # Find all matching directories
    matching_dirs = []
    try:
        for entry in os.listdir(saa_base_dir):
            entry_path = os.path.join(saa_base_dir, entry)
            # Check if it's a directory and matches the pattern
            if os.path.isdir(entry_path) and entry.startswith(pattern):
                matching_dirs.append(entry)
    except Exception as e:
        raise RuntimeError(
            f"Failed to list contents of {saa_base_dir}: {str(e)}"
        )
    
    # Check if any matching directories found
    if not matching_dirs:
        raise FileNotFoundError(
            f"No SAA model directories found matching pattern: {pattern}* in {saa_base_dir}\n"
            f"Expected directory format: {pattern}YY_MM_DD/"
        )
    
    # Sort by directory name (chronological order due to date suffix)
    # Most recent will be last
    matching_dirs.sort()
    selected_dir = matching_dirs[-1]  # Use most recent
    
    print(f"[load_saa_models] Found {len(matching_dirs)} matching SAA directories")
    print(f"[load_saa_models] Selected most recent: {selected_dir}")
    
    # Construct path to best_model.zip
    model_dir = os.path.join(saa_base_dir, selected_dir)
    model_path = os.path.join(model_dir, "best_model.zip")

    vecnormalize_path = os.path.join(model_dir, "best_model_vecnormalize.pkl")
    if not os.path.exists(vecnormalize_path):
        print(f"[load_saa_models] WARNING: VecNormalize stats not found at {vecnormalize_path}. Proceeding without obs normalization.")
        vecnormalize_path = None
    
    # Verify model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SAA model file not found: {model_path}\n"
            f"Expected file: best_model.zip in directory {selected_dir}"
        )
    
    # Get asset list from market_data_cache
    # This requires the cache to be passed or available in config
    # For now, we'll assume config has an "assets" key or we need to infer from cache
    
    # Get asset list from config if explicitly listed
    assets = config["environment"].get("assets")
    
    
    if not assets:
        raise ValueError(
            "Config missing asset list. Please provide 'assets' key in 'environment' section."
        )
    
    # Build asset_to_model_path dict
    # All assets use the same model path (single-model approach)
    # OR each asset has its own model (multi-model approach)
    
    # Based on the documentation, it seems each asset uses the SAME SAA model
    # trained in single-asset mode, not separate models per asset
    # The SAA is generic and asset-agnostic
    
    # However, looking at FrozenSAAEnsemble.__init__, it expects:
    # asset_to_model_path: Dict[str, str] mapping each asset to a model path
    
    # This suggests two possible interpretations:
    # 1. Each asset has its own trained SAA model (separate training per asset)
    # 2. All assets share the same SAA model (single generic model)
    
    # Based on the SAA training approach (randomly selects asset per episode),
    # it's likely a SINGLE model trained on all assets, not per-asset models
    
    # So we'll map all assets to the same model path
    asset_to_model_path = {}
    for asset in assets:
        asset_to_model_path[asset] = model_path
    
    print(f"[load_saa_models] Loading SAA model for {len(assets)} assets")
    print(f"[load_saa_models] Model path: {model_path}")
    print(f"[load_saa_models] Device: {saa_device}")
    
    # Create and return FrozenSAAEnsemble
    try:
        saa_ensemble = FrozenSAAEnsemble(
            asset_to_model_path=asset_to_model_path,
            vecnormalize_path=vecnormalize_path,
            device=saa_device
        )
        return saa_ensemble
    
    except Exception as e:
        raise RuntimeError(
            f"Failed to create FrozenSAAEnsemble: {str(e)}\n"
            f"Model path: {model_path}\n"
            f"Assets: {assets}"
        )


def build_tokenizer(
    cache: MarketDataCache,
    config: Dict[str, Any]
) -> TransformerTokenizer:
    """
    Instantiate TransformerTokenizer with dimensions from config and cache.
    
    Args:
        cache: MarketDataCache for asset list
        config: Config dict with feature selections
    
    Returns:
        TransformerTokenizer instance
    """
    # ------------------------------
    # Validate required inputs
    # ------------------------------
    if cache is None:
        raise ValueError("MarketDataCache is required to build tokenizer.")
    if config is None:
        raise ValueError("Config dict is required to build tokenizer.")

    # ------------------------------
    # Derive core dimensions from cache
    # ------------------------------
    num_assets = int(cache.num_assets)
    asset_feature_dim = int(cache.num_features)  # Selected technical indicator count
    saa_feature_dim = 1  # SAA signal is a scalar action per asset

    if num_assets <= 0:
        raise ValueError(f"Invalid num_assets from cache: {num_assets}")
    if asset_feature_dim <= 0:
        raise ValueError(f"Invalid asset_feature_dim from cache: {asset_feature_dim}")

    # ------------------------------
    # Optional validation: asset list consistency
    # ------------------------------
    # If config declares an asset list, ensure it matches cache for correctness.
    config_assets = config.get("assets") or config.get("environment", {}).get("assets")
    if config_assets is not None:
        if set(config_assets) != set(cache.asset_names):
            raise ValueError(
                "Asset list mismatch between config and MarketDataCache.\n"
                f"Config assets ({len(config_assets)}): {config_assets}\n"
                f"Cache assets ({len(cache.asset_names)}): {cache.asset_names}"
            )

    # ------------------------------
    # Portfolio feature dimension
    # ------------------------------
    # This must match AllocatorEnvironmentWrapper._extract_portfolio_features(),
    # which currently produces 8 features:
    # [cash_weight, portfolio_value, portfolio_return, sharpe_ratio,
    #  max_drawdown, volatility, turnover, alpha]
    tokenizer_config = config.get("allocator_tokenizer", {}) or config.get("tokenizer", {})
    portfolio_feature_dim = tokenizer_config.get("portfolio_feature_dim")
    if portfolio_feature_dim is None:
        portfolio_feature_dim = 8  # Default aligned with wrapper feature extraction
    portfolio_feature_dim = int(portfolio_feature_dim)

    if portfolio_feature_dim <= 0:
        raise ValueError(f"Invalid portfolio_feature_dim: {portfolio_feature_dim}")

    # ------------------------------
    # Transformer model dimension
    # ------------------------------
    # Prefer allocator_transformer config, then tokenizer config, then top-level.
    transformer_config = config.get("allocator_transformer", {}) or config.get("transformer", {})
    d_model = (
        transformer_config.get("d_model")
        or tokenizer_config.get("d_model")
        or config.get("d_model")
    )
    if d_model is None:
        # Conservative default if not specified in config
        d_model = 128
        print("[build_tokenizer] d_model not found in config; defaulting to 128")

    d_model = int(d_model)
    if d_model <= 0:
        raise ValueError(f"Invalid d_model: {d_model}")

    # ------------------------------
    # Instantiate tokenizer
    # ------------------------------
    tokenizer = TransformerTokenizer(
        num_assets=num_assets,
        saa_feature_dim=saa_feature_dim,
        asset_feature_dim=asset_feature_dim,
        portfolio_feature_dim=portfolio_feature_dim,
        d_model=d_model
    )

    # Log final dimensions for debugging/integration visibility
    print("[build_tokenizer] Tokenizer initialized with:")
    print(f"  num_assets: {num_assets}")
    print(f"  saa_feature_dim: {saa_feature_dim}")
    print(f"  asset_feature_dim: {asset_feature_dim}")
    print(f"  portfolio_feature_dim: {portfolio_feature_dim}")
    print(f"  d_model: {d_model}")

    return tokenizer


# ================================
# PPO Model Building
# ================================


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
    

# Build PPO model with custom transformer policy and hyperparameters from config
def build_allocator_model(
    env: gym.Env,
    config: Dict[str, Any]
) -> PPO:
    """
    Instantiate PPO model with custom transformer policy and hyperparameters from config.
    
    Args:
        env: Vectorized training environment
        config: Full configuration dict
    
    Returns:
        PPO model instance ready for training
        
    Integration:
    - Reads hyperparameters from config["portfolio_allocator_agent"] section
    - Creates learning rate schedule using linear_three_phase_schedule
    - Uses standard PPO algorithm from SB3 with custom TransformerAllocatorPolicy
    - Policy is TransformerAllocatorPolicy via policy_kwargs
    
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
    
    # Extract agent configuration section
    # Config uses "portfolio_allocator_agent" key instead of "agent"
    agent_cfg = config.get("portfolio_allocator_agent", {})
    
    # Fallback: also check "agent" key for compatibility
    if not agent_cfg:
        agent_cfg = config.get("agent", {})
    
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
    
    # --- Policy Architecture Configuration ---
    
    # Extract transformer configuration for policy network
    transformer_cfg = config.get("allocator_transformer", {})
    
    # Number of assets (needed for policy architecture)
    num_assets = len(config.get("environment", {}).get("assets", []))
    if num_assets == 0:
        raise ValueError("Config must specify assets in environment section")
    
    # Build policy_kwargs for TransformerAllocatorPolicy
    # These parameters are passed directly to the policy class constructor
    policy_kwargs = {
        # Transformer architecture parameters
        "num_assets": num_assets,
        "d_model": int(transformer_cfg.get("d_model", 128)),
        "n_heads": int(transformer_cfg.get("n_heads", 8)),
        "n_layers": int(transformer_cfg.get("n_layers", 4)),
        "dim_feedforward": int(transformer_cfg.get("dim_feedforward", 512)),
        "dropout": float(transformer_cfg.get("dropout", 0.1)),
        "use_asset_id_embedding": bool(transformer_cfg.get("use_asset_id_embedding", True)),
        "use_portfolio_token": bool(transformer_cfg.get("use_portfolio_token", True)),
        # SB3 default actor/critic network architecture (applied after policy)
        # Empty list = linear mapping from transformer output to actions/values
        "net_arch": []
    }
    
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
    # Uses standard PPO (not RecurrentPPO) since transformer handles sequences
    model = PPO(
        policy=TransformerAllocatorPolicy,  # Custom transformer policy class
        env=env,  # Vectorized environment (DummyVecEnv + VecNormalize)
        
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
        
        # Policy architecture
        policy_kwargs=policy_kwargs,  # Custom transformer policy
        
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
        tag_prefix="allocator/validation",  # TensorBoard prefix for validation metrics
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
    # Set seeds for reproducibility (PyTorch, NumPy, env wrappers)
    seed = int(config.get("training", {}).get("seed", 42))
    np.random.seed(seed)
    
    # Extract gamma for VecNormalize reward normalization
    gamma_cfg = config.get("portfolio_allocator_agent", {}).get("gamma", 0.99)
    
    # --- Load Pre-trained SAA Models ---
    
    # Load frozen SAA ensemble from disk (used for per-asset signal generation)
    # SAA models are trained in single-asset mode; allocator learns to weight them
    print("[run] Loading pre-trained SAA models...")
    saa_ensemble = load_saa_models(config=config, device="cpu")
    print(f"[run] SAA ensemble loaded for {len(saa_ensemble.assets)} assets")
    
    # --- Build Training Environment ---
    
    # Create environment wrapper for training (uses training data blocks)
    # Wraps base portfolio environment with SAA signals + transformer tokenizer
    def make_train_env():
        env = build_allocator_env(
            cache=cache,
            config=config,
            saa_ensemble=saa_ensemble,
            seed=seed,
            for_eval=False  # Use training data blocks
        )
        return env
    
    # Vectorize training environment (DummyVecEnv for single env, then VecNormalize)
    # VecNormalize: obs/reward normalization + discount gamma handling
    vec_train = VecNormalize(
        DummyVecEnv([make_train_env]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=np.inf,
        gamma=gamma_cfg  # For advantage normalization in reward scaling
    )
    
    print("[run] Training environment created and vectorized")
    
    # --- Build Evaluation Environment ---
    
    # Create environment wrapper for evaluation (uses validation data blocks)
    def make_eval_env():
        env = build_allocator_env(
            cache=cache,
            config=config,
            saa_ensemble=saa_ensemble,
            seed=seed + 1,  # Different seed for eval
            for_eval=True  # Use validation data blocks
        )
        return env
    
    # Vectorize evaluation environment (same structure, but training=False in VecNormalize)
    # training=False: freeze norm statistics, don't update them during eval
    vec_eval = VecNormalize(
        DummyVecEnv([make_eval_env]),
        training=False,
        norm_obs=True,
        norm_reward=False,  # Raw rewards for evaluation reporting
        clip_obs=10.0,
        clip_reward=np.inf,
        gamma=gamma_cfg
    )

    vec_eval.obs_rms = vec_train.obs_rms  # Share observation normalization stats
    if vec_eval.norm_reward==True:
        vec_eval.ret_rms = vec_train.ret_rms
    
    print("[run] Evaluation environment created and vectorized")
    
    # --- Build PPO Model ---
    
    # Instantiate PPO with custom transformer policy and learning rate/entropy schedules
    print("[run] Building PPO model with custom transformer policy...")
    model = build_allocator_model(env=vec_train, config=config)
    print("[run] PPO model built successfully")
    
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
        tag_prefix="allocator/train",
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