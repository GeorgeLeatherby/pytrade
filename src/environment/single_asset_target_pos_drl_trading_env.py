"""
Trading Environment
============================================

Implements a multi-asset trading environment optimized for Deep Reinforcement Learning (DRL)
using Gymnasium interface. Supports trading across multiple indices with realistic constraints.
Assumes that fx_rates have been applied and all data is provided in USD!

Key Design Principles:
1. Efficient state representation for neural networks (for LSTM layers)
2. Proper reward shaping for stable PPO training
3. Realistic trading constraints and costs
4. Multi-asset portfolio optimization focus
5. Currency conversion handling for global assets

Uses a given .csv file with daily historical price data for multiple assets.
This environment is designed for reinforcement learning applications. It provides the
feedback necessary for training trading agents. Since it is based on Gymnasium, it can be
easily integrated with various RL libraries. 

It mainly consists of the following components:
- State Representation: The state includes historical price data, technical indicators,
  current portfolio holdings, cash balance, and other relevant features
- Action Space: The action space allows the agent to decide the allocation of funds
  across different assets, including a cash position

Note: This environment is built to serve sequential data, indicated by the 'lookback_window'
parameter. A step represents a day. 

Data classes:
- PortfolioState:   Current state representation
- EpisodeBuffer:    Step-by-step history within episode
- MarketDataCache:  Fast market data access
- TrainingMetrics:  Cross-episode performance tracking
- ExecutionResult:  Lightweight trade execution results

Function classes:
- TradingEnv:       Main Gym environment class

"""

# Imports
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List, Optional

# Execution mode constants and trade instruction dataclass
EXECUTION_SINGLE_ASSET_TARGET_POS = "single_asset_target_position"
EXECUTION_SIMPLE = "simple"
EXECUTION_TRANCHE = "tranche"
EXECUTION_PORTFOLIO_WEIGHTS = "portfolio_weights"

@dataclass
class TradeInstruction:
    symbol: str
    action: str  # "BUY" or "SELL"
    # Either of the following needs to be set:
    quantity: Optional[float] = None  # number of shares (None for SELL in simple mode) (optional)
    notional: Optional[float] = None  # dollar value (optional)
    # Order type
    order_type: str = "MARKET"  # "MARKET" or "LIMIT"
    limit_price: Optional[float] = None

def _validate_instruction(instr: TradeInstruction, asset_to_index: Dict[str, int]) -> Optional[str]:
    if instr.symbol not in asset_to_index:
        return "unknown_symbol"
    if instr.action not in {"BUY", "SELL"}:
        return "invalid_action"
    if instr.order_type not in {"MARKET", "LIMIT"}:
        return "invalid_order_type"
    if instr.order_type == "LIMIT":
        if instr.limit_price is None or not np.isfinite(instr.limit_price) or instr.limit_price <= 0:
            return "invalid_limit_price"

    # Exactly one of quantity or notional must be provided (and finite)
    has_qty = (instr.quantity is not None) and np.isfinite(instr.quantity) and not np.isnan(instr.quantity)
    has_notional = (instr.notional is not None) and np.isfinite(instr.notional) and not np.isnan(instr.notional)

    if instr.quantity is not None and (not np.isfinite(instr.quantity) or np.isnan(instr.quantity)):
        print(f"Debug Info - Instruction: {instr}. Quantity is not finite or is NaN.")
    if instr.notional is not None and (not np.isfinite(instr.notional) or np.isnan(instr.notional)):
        print(f"Debug Info - Instruction: {instr}. Notional is not finite or is NaN.")

    if has_qty and has_notional:
        print(f"Debug Info - Instruction: {instr}. Both quantity and notional are set, which is not allowed.")
        return "both_qty_and_notional_set"
    if not has_qty and not has_notional:
        print(f"Debug Info - Instruction: {instr}. Neither quantity nor notional is set, one must be provided.")
        return "missing_qty_and_notional"

    # Validate BUY specifics
    if instr.action == "BUY":
        if has_qty and instr.quantity <= 0:
            return "invalid_quantity_for_buy"
        if has_notional and instr.notional <= 0:
            return "invalid_notional_for_buy"
        return None  # Quantity-only or notional-only is allowed

    # Validate SELL specifics
    if instr.action == "SELL":
        # Accept either quantity-only or notional-only, not both
        if has_qty:
            if instr.quantity <= 0:
                return "invalid_quantity_for_sell"
            return None  # quantity-only SELL is valid
        if has_notional:
            if instr.notional <= 0:
                return "invalid_notional_for_sell"
            return None  # notional-only SELL is valid
        # The neither case is already handled above, but keep a clear fallback
        return "missing_qty_and_notional"

@dataclass
class PortfolioState:
    """Current portfolio state representation using positions and cash balance.

    Safety notes:
    - Always copies incoming arrays to avoid accidental aliasing (data leaks) between
      multiple PortfolioState instances or across resets.
    - Uses default_factory for arrays to avoid shared mutable defaults.
    """
    cash: float = 0.0
    positions: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    prices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    step: int = 0
    terminated: bool = False

    def __post_init__(self) -> None:
        # Force independent buffers + consistent dtype
        self.positions = np.asarray(self.positions, dtype=np.float32).copy()
        self.prices = np.asarray(self.prices, dtype=np.float32).copy()

        if self.positions.shape != self.prices.shape:
            raise ValueError(
                f"positions/prices shape mismatch: {self.positions.shape} vs {self.prices.shape}"
            )

        if not np.isfinite(self.cash):
            raise ValueError("cash must be finite")

    def get_asset_values(self) -> np.ndarray:
        # Returns market value of each position excluding cash
        return self.positions * self.prices

    def get_total_value(self) -> float:
        # Returns total portfolio value including cash
        return float(self.cash + np.sum(self.get_asset_values(), dtype=np.float64))

    def get_weights(self) -> np.ndarray:
        # Returns current portfolio weights including cash
        total_value = float(self.get_total_value())
        if total_value <= 1e-8:  # guard against zero
            return np.zeros(len(self.positions) + 1, dtype=np.float32)

        asset_values = self.get_asset_values()
        weights = np.concatenate(
            ([self.cash / total_value], asset_values / total_value)
        ).astype(np.float32)

        if not np.all(np.isfinite(weights)):
            print(
                f"Debug Info - Cash: {self.cash}, Total Value: {total_value}, "
                f"Asset Values: {asset_values}, Weights: {weights}"
            )
            raise ValueError("Non-finite weights detected in portfolio state.")
        return weights

    def portfolio_reset(
        self,
        cash: float,
        positions: np.ndarray,
        prices: np.ndarray,
        step: int,
        terminated: bool,
    ) -> None:
        # Copy to prevent aliasing between portfolio/comparison/benchmark states
        self.cash = float(cash)
        self.positions = np.asarray(positions, dtype=np.float32).copy()
        self.prices = np.asarray(prices, dtype=np.float32).copy()
        self.step = int(step)
        self.terminated = bool(terminated)

        if self.positions.shape != self.prices.shape:
            raise ValueError(
                f"positions/prices shape mismatch: {self.positions.shape} vs {self.prices.shape}"
            )
        if not np.isfinite(self.cash):
            raise ValueError("cash must be finite")


@dataclass
class EpisodeBuffer:
    """
    Buffer for episode data - optimized for DRL training.
    Stores step-by-step portfolio and market data for efficient access.
    Pre-allocated storage for len(lookback_window + episode_length) to avoid dynamic resizing.
    Warmup-phase: zero-padding for any portfolio related metrics.

    Layout:
    Indices [0 : lookback_window) -> warm-up (market-only; portfolio metrics zero)
    Indices [lookback_window : lookback_window + episode_length) -> actual episode steps

    External code (TradingEnv) always uses 'external_step' starting at 0 for first REAL trading day.
    Internally we map: internal_index = lookback_window + external_step

    NOTE: episode_buffer_length_days = lookback_window + episode_length_days
    """
    # Needed vars in initialization:
    episode_buffer_length_days: int
    num_assets: int
    lookback_window: int
    maybe_provide_sequence: bool = False

    # Pre-allocated arrays for vectorized operations
    portfolio_values: np.ndarray = field(init=False)            # [episode_buffer_length_days] - total portfolio value each step
    portfolio_weights: np.ndarray = field(init=False)           # [episode_buffer_length_days, num_assets+1] - weights including cash
    portfolio_positions: np.ndarray = field(init=False)         # [episode_buffer_length_days, num_assets] - number of shares held each step
    comparison_portfolio_value: np.ndarray = field(init=False)  # [episode_buffer_length_days] - comparison portfolio value each step
    benchmark_portfolio_value: np.ndarray = field(init=False)   # [episode_buffer_length_days] - benchmark portfolio value each step
    alpha: np.ndarray = field(init=False)                       # [episode_buffer_length_days] - excess returns over benchmark
    returns: np.ndarray = field(init=False)                     # [episode_buffer_length_days] - daily returns
    saa_returns: np.ndarray = field(init=False)                 # [episode_buffer_length_days] - daily returns of single-asset-agent (cash + selected asset)
    rewards: np.ndarray = field(init=False)           # [episode_buffer_length_days] - RL allocator rewards
    actions: np.ndarray = field(init=False)                     # [episode_buffer_length_days, num_assets+1] - agent actions
    transaction_costs: np.ndarray = field(init=False)           # [episode_buffer_length_days] - costs per step
    sharpe_ratio: np.ndarray = field(init=False)                # [episode_buffer_length_days] - rolling sharpe ratio
    drawdown: np.ndarray = field(init=False)                    # [episode_buffer_length_days] - rolling max drawdown
    volatility: np.ndarray = field(init=False)                  # [episode_buffer_length_days] - rolling volatility
    turnover: np.ndarray = field(init=False)                    # [episode_buffer_length_days] - portfolio turnover
    # Market data
    asset_prices: np.ndarray = field(init=False)                # [episode_buffer_length_days, num_assets] - closing prices
    traded_dollar_volume: np.ndarray = field(init=False)         # [episode_buffer_length_days, num_assets] - dollar volume traded each step
    traded_shares_total: np.ndarray = field(init=False)        # [episode_buffer_length_days, num_assets] - total shares traded each step
    # Metadata
    current_step: int = 0                   # Current step in episode
    num_portfolio_features: int = field(init=False)  # Number of portfolio features for observation
    effective_asset_concentration_norm: np.ndarray = field(init=False)  # [episode_buffer_length_days] - effective asset concentration norm 
    previous_sortino: np.ndarray = field(init=False)  # [episode_buffer_length_days] - previous Sortino ratio for reward component
    current_sortino: np.ndarray = field(init=False)   # [episode_buffer_length_days] - current Sortino ratio for reward component
    running_mean_ema: np.ndarray = field(init=False)  # [episode_buffer_length_days] - running mean EMA of returns for reward component
    downside_var_sqrt: np.ndarray = field(init=False)  # [episode_buffer_length_days] - sqrt of downside variance for Sortino ratio
    previous_max_drawdown: np.ndarray = field(init=False)  # [episode_buffer_length_days] - previous max drawdown for reward component

    # weights, alpha, sharpe_ratio, drawdown, volatility, turnover, allocator_rewards

    def __post_init__(self):
        """Initialize all arrays with proper shapes and types"""
        dtype = np.float32
        self.portfolio_values = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.portfolio_weights = np.zeros((self.episode_buffer_length_days, self.num_assets + 1), dtype=dtype)
        self.portfolio_positions = np.zeros((self.episode_buffer_length_days, self.num_assets), dtype=dtype)
        self.comparison_portfolio_value = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.benchmark_portfolio_value = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.alpha = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.returns = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.saa_returns = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.rewards = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.actions = np.zeros((self.episode_buffer_length_days, self.num_assets + 1), dtype=dtype)
        self.transaction_costs = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.sharpe_ratio = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.drawdown = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.volatility = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.turnover = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.asset_prices = np.zeros((self.episode_buffer_length_days, self.num_assets), dtype=dtype)
        self.traded_dollar_volume = np.zeros((self.episode_buffer_length_days, self.num_assets), dtype=dtype)
        self.traded_shares_total = np.zeros((self.episode_buffer_length_days, self.num_assets), dtype=dtype)
        # If num_features is not available, set to 0
        num_features = getattr(self, "num_features", 0)
        self.current_step = 0
        self.num_portfolio_features = self.num_assets + 1 + 11  # weights + 11 portfolio metrics used by get_observation_at_step
        self.action_entropy = np.zeros(self.episode_buffer_length_days, dtype=dtype) 
        # Reward component tracking (per-step)
        self.reward_alpha = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_risk = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_portfolio_return = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_cost = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_turnover = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_concentration = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_survival = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.effective_asset_concentration_norm = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.previous_sortino = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.current_sortino = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.running_mean_ema = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.downside_var_sqrt = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.previous_max_drawdown = np.zeros(self.episode_buffer_length_days, dtype=dtype)

    def record_step(self, external_step: int, portfolio_value: float, weights: np.ndarray, portfolio_positions: np.ndarray,
                   daily_return: float, saa_return: float, reward_to_record: float,action: np.ndarray,
                   transaction_cost: float, prices: np.ndarray,
                   sharpe_ratio: float = 0.0, drawdown: float = 0.0, volatility: float = 0.0, turnover: float = 0.0, alpha: float = 0.0, benchmark_portfolio_value: float = 0.0,
                   comparison_portfolio_value: float = 0.0,
                   traded_dollar_volume: float = 0.0, traded_shares_total: float = 0.0,
                   action_entropy: float = 0.0, saa_reward_parts: Optional[Dict[str, float]] = None,
                   reward_parts: Optional[Dict[str, float]] = None,
                   effective_asset_concentration_norm: float = 0.0, previous_sortino: float = 0.0, current_sortino: float = 0.0,
                   running_mean_ema: float = 0.0, downside_var_sqrt: float = 0.0, previous_max_drawdown: float = 0.0) -> None:
        
        """
        Record step data efficiently. external_step: 0-based episode day index (first real day = 0)
        """

        # Internal offset due to lookback warmup
        if self.maybe_provide_sequence:
            internal_offset_step = external_step + self.lookback_window
        else:
            internal_offset_step = external_step

        assert 0 <= internal_offset_step < self.episode_buffer_length_days, "Step out of bounds of EpisodeBuffer"

        self.portfolio_values[internal_offset_step] = portfolio_value
        self.portfolio_weights[internal_offset_step] = weights
        self.portfolio_positions[internal_offset_step] = portfolio_positions
        self.returns[internal_offset_step] = daily_return
        self.saa_returns[internal_offset_step] = saa_return
        self.rewards[internal_offset_step] = reward_to_record
        self.actions[internal_offset_step] = action
        self.transaction_costs[internal_offset_step] = transaction_cost
        self.asset_prices[internal_offset_step] = prices
        self.sharpe_ratio[internal_offset_step] = sharpe_ratio
        self.drawdown[internal_offset_step] = drawdown
        self.volatility[internal_offset_step] = volatility
        self.turnover[internal_offset_step] = turnover
        self.alpha[internal_offset_step] = alpha
        self.comparison_portfolio_value[internal_offset_step] = comparison_portfolio_value
        self.benchmark_portfolio_value[internal_offset_step] = benchmark_portfolio_value
        self.traded_dollar_volume[internal_offset_step] = float(traded_dollar_volume)
        self.traded_shares_total[internal_offset_step] = float(traded_shares_total)
        self.action_entropy[internal_offset_step] = float(action_entropy)
        self.effective_asset_concentration_norm[internal_offset_step] = float(effective_asset_concentration_norm)
        self.previous_sortino[internal_offset_step] = previous_sortino
        self.current_sortino[internal_offset_step] = current_sortino
        self.running_mean_ema[internal_offset_step] = running_mean_ema
        self.downside_var_sqrt[internal_offset_step] = downside_var_sqrt
        self.previous_max_drawdown[internal_offset_step] = previous_max_drawdown
        # Reward components
        if reward_parts is not None:
            self.reward_alpha[internal_offset_step] = reward_parts.get("alpha_component", 0.0)
            self.reward_risk[internal_offset_step] = reward_parts.get("risk_component", 0.0)
            self.reward_portfolio_return[internal_offset_step] = reward_parts.get("portfolio_return_component", 0.0)
            self.reward_cost[internal_offset_step] = reward_parts.get("cost_component", 0.0)
            self.reward_turnover[internal_offset_step] = reward_parts.get("turnover", 0.0)
            self.reward_concentration[internal_offset_step] = reward_parts.get("concentration_component", 0.0)
            self.reward_survival[internal_offset_step] = reward_parts.get("survival_component", 0.0)
        

        # Update current step and episode length
        self.current_step = external_step
        self.episode_length = min(external_step + 1, self.episode_buffer_length_days)

    def get_returns_window(self, window: int) -> np.ndarray:
        """Get last N returns for risk calculations (no wrap-around, no ring buffer)"""
        end_idx = self.current_step
        start_idx = max(0, end_idx - window)
        return self.returns[start_idx:end_idx]
    
    def calculate_sharpe_ratio(self, window: int) -> float:
        """Calculate rolling Sharpe ratio efficiently (annualized, no risk-free rate)"""
        returns_window = self.get_returns_window(window)
        if len(returns_window) < 2:
            return 0.0
        mean_return = np.mean(returns_window)
        std_return = np.std(returns_window)
        if std_return == 0:
            return 0.0

        sharpe = mean_return / std_return
        if np.any(np.isnan(sharpe)):
            raise ValueError("NaN value detected in Sharpe ratio calculation.")
        if np.any(np.isinf(sharpe)):
            raise ValueError("Infinite value detected in Sharpe ratio calculation.")
        return sharpe
    
    def calculate_max_drawdown(self, window: int) -> float:
        """
        Maximum drawdown over the last `window` steps (robust version).
        Drawdown(t) = (Peak_to_date - Value_t) / Peak_to_date, after first positive value.
        Returns 0.0 until a positive portfolio value is observed.
        Completely avoids NaNs/Infs from 0/0 situations in warm-up.
        """
        end_idx = self.current_step
        if end_idx <= 0:
            return 0.0
        start_idx = max(0, end_idx - window)
        values_window = self.portfolio_values[start_idx:end_idx]

        if values_window.size == 0:
            return 0.0

        # Keep only finite values
        values_window = values_window[np.isfinite(values_window)]
        if values_window.size == 0:
            return 0.0

        # Find first strictly positive portfolio value (ignore initial zeros)
        positive_mask = values_window > 0
        if not positive_mask.any():
            return 0.0  # still only zeros

        first_pos_idx = np.flatnonzero(positive_mask)[0]
        v = values_window[first_pos_idx:]  # slice from first positive onward

        if v.size < 2:
            return 0.0

        # Running peak (strictly > 0)
        peaks = np.maximum.accumulate(v)

        # Compute drawdowns (all denominators > 0, so no 0/0)
        drawdowns = (peaks - v) / peaks

        # Numerical safety (should not be needed but defensive)
        drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)

        max_dd = float(np.max(drawdowns)) if drawdowns.size else 0.0
        if max_dd < 0:
            max_dd = 0.0
        return max_dd
    
    def saa_calculate_max_drawdown(self, selected_asset_idx, window: int) -> float:
        """
        Maximum drawdown of the subportfolio (cash + selected asset) over the last `window` steps.
        
        Drawdown(t) = (Peak_to_date - Value_t) / Peak_to_date, after first positive value.
        Returns 0.0 until a positive subportfolio value is observed.
        Ignores price changes of phantom assets (all non-selected assets).
        
        Args:
            selected_asset_idx: Index of the trading asset (0 to num_assets-1)
            window: Number of steps to consider for drawdown calculation
            
        Returns:
            Maximum drawdown as float in [0, 1]
        """
        end_idx = self.current_step
        if end_idx <= 0:
            return 0.0
        
        start_idx = max(0, end_idx - window)
        
        # Reconstruct subportfolio values: cash + selected asset notional
        # portfolio_weights[t, 0] = cash_weight = cash[t] / total_portfolio_value[t]
        # portfolio_positions[t, selected_asset_idx] = shares held
        # asset_prices[t, selected_asset_idx] = price of selected asset at step t
        
        subpf_values = np.zeros(end_idx - start_idx, dtype=np.float32)
        
        for i, step_idx in enumerate(range(start_idx, end_idx)):
            # Total portfolio value at this step
            total_pv = self.portfolio_values[step_idx]
            
            if total_pv <= 0:
                subpf_values[i] = 0.0
                continue
            
            # Cash value (from weight)
            cash_weight = self.portfolio_weights[step_idx, 0]  # First element is cash weight
            cash_value = cash_weight * total_pv
            
            # Selected asset notional value
            position_shares = self.portfolio_positions[step_idx, selected_asset_idx]
            asset_price = self.asset_prices[step_idx, selected_asset_idx]
            asset_value = position_shares * asset_price
            
            # Subportfolio = cash + selected asset only
            subpf_values[i] = cash_value + asset_value
        
        if subpf_values.size == 0:
            return 0.0
        
        # Keep only finite values
        subpf_values = subpf_values[np.isfinite(subpf_values)]
        if subpf_values.size == 0:
            return 0.0
        
        # Find first strictly positive subportfolio value (ignore initial zeros)
        positive_mask = subpf_values > 0
        if not positive_mask.any():
            return 0.0  # still only zeros
        
        first_pos_idx = np.flatnonzero(positive_mask)[0]
        v = subpf_values[first_pos_idx:]  # slice from first positive onward
        
        if v.size < 2:
            return 0.0
        
        # Running peak (strictly > 0)
        peaks = np.maximum.accumulate(v)
        
        # Compute drawdowns (all denominators > 0, so no 0/0)
        drawdowns = (peaks - v) / peaks
        
        # Numerical safety (should not be needed but defensive)
        if np.isnan(drawdowns).any() or np.isinf(drawdowns).any():
            print("NaN or Inf value detected in SAA drawdown calculation. Clipping to 0.0.")
            drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)
        
        max_dd = float(np.max(drawdowns)) if drawdowns.size else 0.0
        if max_dd < 0:
            max_dd = 0.0
        return max_dd
    
    def get_observation_lookback(self) -> np.ndarray:
        """
        Get portfolio lookback for LSTM input computationally efficient.
        Sequences of vars: weights, alpha, sharpe_ratio, drawdown, volatility, turnover, allocator_rewards
        - shape [lookback_window, num_portfolio_features]
        """
        # Determine start and end indices for lookback (handle circular buffer)
        end_idx = self.current_step
        start_idx = max(0, end_idx - self.lookback_window + 1)
        actual_window = end_idx - start_idx + 1

        # Pre-allocate output array
        num_portfolio_features = self.num_portfolio_features # 6 + self.portfolio_weights.shape[1]  # weights + alpha + sharpe + drawdown + volatility + turnover + allocator_rewards
        obs = np.zeros((self.lookback_window, num_portfolio_features), dtype=np.float32)

        # Gather sequences
        # weights: [lookback_window, num_assets+1]
        weights_seq = self.portfolio_weights[start_idx:end_idx+1]
        alpha_seq = self.alpha[start_idx:end_idx+1].reshape(-1, 1)
        sharpe_seq = self.sharpe_ratio[start_idx:end_idx+1].reshape(-1, 1)
        drawdown_seq = self.drawdown[start_idx:end_idx+1].reshape(-1, 1)
        volatility_seq = self.volatility[start_idx:end_idx+1].reshape(-1, 1)
        turnover_seq = self.turnover[start_idx:end_idx+1].reshape(-1, 1)
        rewards_seq = self.rewards[start_idx:end_idx+1].reshape(-1, 1)

        # Concatenate all features along last axis
        features_seq = np.concatenate([
            weights_seq, alpha_seq, sharpe_seq, drawdown_seq, volatility_seq, turnover_seq, rewards_seq
        ], axis=1)

        # Place into output array (pad at beginning if needed)
        if actual_window > 0:
            obs[-actual_window:] = features_seq

        return obs

    def get_observation_at_step(self, external_step: int) -> np.ndarray:
        """
        Get portfolio observation at current step for non-sequence input.
        Returns:
            Flattened observation array for current step
            Shape: (total_observation_size,)
        """
        # Set step depending on sequence provision
        if self.maybe_provide_sequence:
            internal_step = external_step + self.lookback_window
        else: 
            internal_step = external_step
        # Extract features at the given step
        weights = self.portfolio_weights[internal_step]
        alpha = self.alpha[internal_step]
        sharpe = self.sharpe_ratio[internal_step]
        drawdown = self.drawdown[internal_step]
        volatility = self.volatility[internal_step]
        turnover = self.turnover[internal_step]
        previous_sortino = self.previous_sortino[internal_step]
        current_sortino = self.current_sortino[internal_step]
        running_mean_ema = self.running_mean_ema[internal_step]
        downside_var_sqrt = self.downside_var_sqrt[internal_step]
        previous_max_drawdown = self.previous_max_drawdown[internal_step]

        # rewards = self.allocator_rewards[internal_step] # Why feed reward.
        effective_asset_concentration_norm = self.effective_asset_concentration_norm[internal_step]

        observation = np.concatenate([
            weights,
            [alpha, sharpe, drawdown, volatility, turnover, effective_asset_concentration_norm, previous_sortino, 
            current_sortino, running_mean_ema, downside_var_sqrt, previous_max_drawdown]
        ]).astype(np.float32)

        return observation
    
    def reset_episode_buffer(self) -> None:
        """
        Reset buffer for new episode.
        By default, reset every numpy array attribute to zero.
        Preserve only arrays explicitly listed in excluded_arrays.
        """
        self.current_step = 0
        self.episode_length = 0

        # Keep this list short and explicit.
        # Example: {"asset_prices"} if you want to preserve warm-up prices.
        excluded_arrays = {
            # "e.g. but not recommended: asset_prices",
        }

        for name, value in self.__dict__.items():
            if name in excluded_arrays:
                continue
            if isinstance(value, np.ndarray):
                value.fill(0.0)

    def warmup_market_data(self, market_data_cache, episode_start_step: int):
        """
        Fill the first lookback_window entries with market close prices for warm-up.
        All portfolio-related metrics remain zero. Uses absolute indices for correct alignment.
        
        Args:
            market_data_cache: MarketDataCache instance for data access
            episode_start_step: Absolute start index of the episode in the full dataset
        """
        if self.lookback_window <= 0:
            return # No warmup needed
        
        for warmup_step in range(self.lookback_window):
            abs_idx = episode_start_step - self.lookback_window + warmup_step
            if 0 <= abs_idx < market_data_cache.num_days:
                self.asset_prices[warmup_step] = market_data_cache.close_prices[abs_idx]
            else:
                self.asset_prices[warmup_step] = np.zeros(market_data_cache.num_assets, dtype=np.float32)


@dataclass
class DataBlock:
    """Represents a continuous data block for time series splitting."""
    block_id: str
    block_type: str  # 'train' or 'validation'
    start_date_idx: int  # Index in the full dataset
    end_date_idx: int    # Index in the full dataset (exclusive)
    start_date: str      # Human readable date
    end_date: str        # Human readable date
    num_days: int
    max_episodes: int    # Maximum episodes possible in this block
    min_start_step: int  # Minimum episode start step (accounting for lookback)
    max_start_step: int  # Maximum episode start step


@dataclass
class MarketDataCache:
    """
    Cache for fast market data access during DRL training.
    Resposible for splitting into train and validation set via splitting full data set into blocks
    containing train and val set with a defined ratio.
    
    How to fill with data:
    Load CSV as df with columns: Date,Symbol,Open,High,Low,Close,Volume,feature1,feature2,...
    Then call MarketDataCache.from_dataframe(df, config, lookback_window)
    """
    # Core data arrays - full dataset cached
    dates: np.ndarray                   # [num_days] - all trading dates
    asset_names: List[str]              # Asset identifiers
    selected_feature_names: List[str]   # Selected feature column names (from config)
    available_feature_names: List[str]  # All available features in CSV
    
    # Market data arrays [num_days, num_assets]
    open_prices: np.ndarray
    high_prices: np.ndarray  
    low_prices: np.ndarray
    close_prices: np.ndarray            # Primary prices for valuation
    volumes: np.ndarray
    
    # Pre-calculated features [num_days, num_assets, num_selected_features]
    features: np.ndarray                # Only selected technical indicators, ratios, etc.
    
    # Fast lookup tables
    date_to_index: Dict[str, int]       # Date -> array index mapping
    asset_to_index: Dict[str, int]      # Asset -> array index mapping
    feature_to_index: Dict[str, int]    # Selected feature -> array index mapping
    
    # Time series splitting infrastructure
    train_blocks: List[DataBlock]       # Training data blocks
    validation_blocks: List[DataBlock]  # Validation data blocks
    block_sampling_weights: Dict[str, np.ndarray]  # Sampling weights per block type

    # Metadata
    num_days: int
    num_assets: int                     # Number of assets in the market, excluding cash
    num_features: int                   # Number of selected features
    num_available_features: int         # Total available features

    # Training parameters (stored for block calculations)
    episode_length_days: int
    lookback_window: int
    test_val_split_ratio: float

    # Settings
    maybe_provide_sequence: bool
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, config: Dict[str, Any], lookback_window: int,
                       maybe_provide_sequence: bool) -> 'MarketDataCache':
        """
        Build cache from full dataset DataFrame efficiently with feature selection.
        Handles df format: Date,Symbol,Open,High,Low,Close,Volume,feature1,feature2,...
        
        Args:
            df: Full dataset with long format (multiple assets per date)
            config: Configuration dictionary with feature selection
            lookback_window: Minimum history needed for episodes
        """
        print(f"Caching market data: {df.shape[0]:,} records with {df.shape[1]} columns")

        maybe_provide_sequence = maybe_provide_sequence

        # Check for any NaNs in full dataset
        if df.isnull().values.any():
            num_nans = df.isnull().sum().sum()
            raise ValueError(f"Input DataFrame contains {num_nans} NaN values. Please clean the data before proceeding.")
        
        # Verify that for each existing date, all assets have data
        date_asset_counts = df.groupby('Date')['Symbol'].nunique()
        expected_asset_count = df['Symbol'].nunique()
        incomplete_dates = date_asset_counts[date_asset_counts < expected_asset_count]
        if not incomplete_dates.empty:
            raise ValueError(f"Incomplete data found for dates: {incomplete_dates.index.tolist()}") 
        
        # FIXED: Handle actual CSV column names (capitalized)
        df = df.copy()
        column_mapping = {
            'Date': 'date',
            'Symbol': 'symbol', 
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert date column to datetime if it's not already
        if df['date'].dtype == 'object':
            df['date'] = pd.to_datetime(df['date'])
        
        # Extract unique dates and assets
        unique_dates = sorted(df['date'].unique())
        assets = sorted(df['symbol'].unique())
        
        # Get all available features (exclude raw date and OHLCV columns)
        base_cols = {'date', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
        available_feature_cols = [col for col in df.columns if col not in base_cols]

        
        # Feature selection based on config
        selected_features = cls._select_features_from_config(available_feature_cols, config)
        
        print(f"Found {len(assets)} assets: {assets}")
        print(f"Available features: {len(available_feature_cols)} ({available_feature_cols[:5]}...)")
        print(f"Selected features: {len(selected_features)} ({selected_features[:5]}...)")
        print(f"Unused features: {set(available_feature_cols) - set(selected_features)}")
        
        # Create lookup tables
        date_to_index = {date: i for i, date in enumerate(unique_dates)}
        asset_to_index = {asset: i for i, asset in enumerate(assets)}
        feature_to_index = {feat: i for i, feat in enumerate(selected_features)}
        
        num_days = len(unique_dates)
        num_assets = len(assets)  
        num_features = len(selected_features)
        
        print(f"Dimensions: {num_days:,} days × {num_assets} assets × {num_features} features")
        print(f"Total data points: {num_days * num_assets * (5 + num_features):,}")
        
        # Pre-allocate arrays with NaN initialization
        dtype = np.float32
        open_prices = np.full((num_days, num_assets), np.nan, dtype=dtype)
        high_prices = np.full((num_days, num_assets), np.nan, dtype=dtype)
        low_prices = np.full((num_days, num_assets), np.nan, dtype=dtype)
        close_prices = np.full((num_days, num_assets), np.nan, dtype=dtype)
        volumes = np.full((num_days, num_assets), np.nan, dtype=dtype)
        features = np.full((num_days, num_assets, num_features), np.nan, dtype=dtype)
        
        # Fill arrays efficiently
        print("Filling price and feature arrays...")
        for asset_idx, asset in enumerate(assets):
            asset_data = df[df['symbol'] == asset].copy()
            
            if asset_data.empty:
                print(f"Warning: No data found for asset {asset}")
                continue
                
            asset_data = asset_data.sort_values('date')
            
            for _, row in asset_data.iterrows():
                date_idx = date_to_index[row['date']]
                
                # Fill OHLCV data
                open_prices[date_idx, asset_idx] = row['open']
                high_prices[date_idx, asset_idx] = row['high'] 
                low_prices[date_idx, asset_idx] = row['low']
                close_prices[date_idx, asset_idx] = row['close']
                volumes[date_idx, asset_idx] = row['volume']
                
                # Fill ONLY selected feature data
                for feat_idx, feat_col in enumerate(selected_features):
                    if feat_col in row and pd.notna(row[feat_col]):
                        features[date_idx, asset_idx, feat_idx] = row[feat_col]
        
        print(f"Initial cache build complete. Memory usage: ~{cls._estimate_memory_mb(num_days, num_assets, num_features):.1f} MB")
        # Validate data quality
        nan_pct = np.isnan(close_prices).mean() * 100
        features_nan_pct = np.isnan(features).mean() * 100
        print(f"Data quality: {nan_pct:.2f}% NaN in prices, {features_nan_pct:.2f}% NaN in features")


        # Extract training parameters from config
        episode_length_days = config['environment']['episode_length_days']
        test_val_split_ratio = config['environment']['test_val_split_ratio']
        block_buffer_multiplier = config['environment']['block_buffer_multiplier']

        # Create instance first, then add splitting
        instance = cls(
            dates=np.array([pd.Timestamp(d).strftime('%Y-%m-%d') for d in unique_dates]),
            asset_names=assets,
            selected_feature_names=selected_features,
            available_feature_names=available_feature_cols,
            open_prices=open_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            close_prices=close_prices,
            volumes=volumes,
            features=features,
            date_to_index={pd.Timestamp(d).strftime('%Y-%m-%d'): i for i, d in enumerate(unique_dates)},
            asset_to_index=asset_to_index,
            feature_to_index=feature_to_index,
            num_days=num_days,
            num_assets=num_assets,
            num_features=num_features,
            num_available_features=len(available_feature_cols),
            train_blocks=[],
            validation_blocks=[],
            block_sampling_weights={'train': np.array([]), 'validation': np.array([])},
            episode_length_days=episode_length_days,
            lookback_window=lookback_window,
            test_val_split_ratio=test_val_split_ratio,
            maybe_provide_sequence=maybe_provide_sequence
        )
        
        # NEW: Create time series blocks
        instance._create_time_series_blocks(block_buffer_multiplier, test_val_split_ratio)
        
        return instance

    def _create_time_series_blocks(self, block_buffer_multiplier, test_val_split_ratio):
        """
        Create train/validation blocks using group time series splitting.
        A timeframe contains enough days to generate a test and a validation block.
        
        Strategy:
        1. Calculate required_timeframe_size based on provided test_val_split_ratio, lookback_window, 
           episode_length multiplied with buffer. Ensure enough days for BOTH training and validation block
            NOTE: Buffer is needed to ensure random starting points within a block. (E.g. buffer = *2)
        2. Split full available data into x timeframes. If last timeframe is smaller then required_timeframe_size, 
           simply add date range to prior timeframe.
        3. Out of each timeframe create one train and one validation block.
        
        """
        print("\n" + "="*60)
        print("CREATING TIME SERIES BLOCKS")
        print("="*60)
        
        # Calculate minimum viable block size for training
        if self.maybe_provide_sequence:
            min_episode_requirement = self.lookback_window + self.episode_length_days
        else: 
            min_episode_requirement = self.episode_length_days
        min_viable_train_block = min_episode_requirement * block_buffer_multiplier

        # Minimum viable block size for validation
        min_viable_val_block = min_episode_requirement

        # Calculate required timeframe size ensuring BOTH blocks are viable
        # Using the constraint: train_size + val_size = timeframe_size
        # And: train_size = timeframe_size * (1 - test_val_split_ratio)
        # And: val_size = timeframe_size * test_val_split_ratio
        
        # Ensure validation block is large enough
        min_timeframe_for_val = min_viable_val_block / test_val_split_ratio
        
        # Ensure training block is large enough  
        min_timeframe_for_train = min_viable_train_block / (1 - test_val_split_ratio)
        
        # Take the larger requirement
        required_timeframe_size = int(max(min_timeframe_for_val, min_timeframe_for_train))


        print(f"Episode requirements:")
        print(f"  Lookback window: {self.lookback_window} days")
        print(f"  Episode length: {self.episode_length_days} days")
        print(f"  Min episode requirement: {min_episode_requirement} days")
        print(f"  Buffer multiplier: {block_buffer_multiplier}x")
        print(f"  Min viable train block: {min_viable_train_block} days")
        print(f"  Min viable validation block: {min_viable_val_block} days")
        print(f"  Min timeframe for validation: {min_timeframe_for_val:.0f} days")
        print(f"  Min timeframe for training: {min_timeframe_for_train:.0f} days")
        print(f"  Required timeframe size: {required_timeframe_size} days")
        print(f"  Total available days: {self.num_days:,}")
        print(f"  Target validation ratio: {self.test_val_split_ratio:.1%}")

        # Possible amount of blocks based on required_size and available days
        possible_timeframe_amount = self.num_days // required_timeframe_size
        if possible_timeframe_amount < 3:
            print(f"Error: Not enough data to create the min required 3 train/validation blocks with the given parameters!")
            print(f"  Required timeframe size: {required_timeframe_size} days")
            print(f"  Available days: {self.num_days} days")
            raise ValueError("Insufficient data for creating time series blocks.")
        
        print(f"  Possible timeframes: {possible_timeframe_amount} (based on integer division)")

        # Create all timeframes and split into train/validation blocks using test_val_split_ratio. Ensure all data is used
        timeframes = []
        for i in range(possible_timeframe_amount):
            start_idx = i * required_timeframe_size
            end_idx = start_idx + required_timeframe_size
            
            # Handle last timeframe to include all remaining data
            if i == possible_timeframe_amount - 1:
                end_idx = self.num_days
            
            # If last timeframe is smaller than required size, merge with prior timeframe
            if (end_idx - start_idx) < required_timeframe_size and i > 0:
                timeframes[-1]['end_idx'] = end_idx
                timeframes[-1]['end_date'] = self.dates[end_idx - 1]
                timeframes[-1]['num_days'] = end_idx - timeframes[-1]['start_idx']
                print(f"Merging last small timeframe into prior one. New end date: {timeframes[-1]['end_date']}")
            else:
                timeframes.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_date': self.dates[start_idx],
                    'end_date': self.dates[end_idx - 1],
                    'num_days': end_idx - start_idx
                })

        # Create train/validation blocks from timeframes with all required fields by DataBlock
        train_block_counter = 0
        val_block_counter = 0
        
        for timeframe_idx, timeframe in enumerate(timeframes):
            # Calculate split point within timeframe
            train_days = int(timeframe['num_days'] * (1 - test_val_split_ratio))
            val_days = timeframe['num_days'] - train_days
            
            # Create training block
            train_start_idx = timeframe['start_idx']
            train_end_idx = train_start_idx + train_days
            
            # Calculate episode constraints for training block
            if self.maybe_provide_sequence:
                train_min_start_step = train_start_idx + self.lookback_window
            else:
                train_min_start_step = train_start_idx
            train_max_start_step = train_end_idx - self.episode_length_days
            train_max_consecutive_episodes = max(0, (train_end_idx - train_min_start_step) // self.episode_length_days)
            
            train_block = DataBlock(
                block_id=f"train_{train_block_counter:02d}",
                block_type='train',  # FIXED: Added missing field
                start_date_idx=train_start_idx,
                end_date_idx=train_end_idx,
                start_date=self.dates[train_start_idx],
                end_date=self.dates[train_end_idx - 1],
                num_days=train_days,
                max_episodes=train_max_consecutive_episodes,  # FIXED: Added missing field
                min_start_step=train_min_start_step,  # FIXED: Added missing field
                max_start_step=train_max_start_step   # FIXED: Added missing field
            )
            
            # Create validation block
            val_start_idx = train_end_idx
            val_end_idx = timeframe['end_idx']
            
            # Calculate episode constraints for validation block
            val_min_start_step = val_start_idx + self.lookback_window
            val_max_start_step = val_end_idx - self.episode_length_days
            val_max_consecutive_episodes = max(0, (val_end_idx - val_min_start_step) // self.episode_length_days)
            
            val_block = DataBlock(
                block_id=f"val_{val_block_counter:02d}",
                block_type='validation',  # FIXED: Added missing field
                start_date_idx=val_start_idx,
                end_date_idx=val_end_idx,
                start_date=self.dates[val_start_idx],
                end_date=self.dates[val_end_idx - 1],
                num_days=val_days,
                max_episodes=val_max_consecutive_episodes,  # FIXED: Added missing field
                min_start_step=val_min_start_step,  # FIXED: Added missing field
                max_start_step=val_max_start_step   # FIXED: Added missing field
            )
            
            # Add to lists
            self.train_blocks.append(train_block)
            self.validation_blocks.append(val_block)
            
            train_block_counter += 1
            val_block_counter += 1
            
            # Print block information
            print(f"\nTimeframe {timeframe_idx + 1}:")
            print(f"  Training block: {train_block.block_id}")
            print(f"    Date range: {train_block.start_date} to {train_block.end_date}")
            print(f"    Days: {train_block.num_days:,}")
            print(f"    Episodes: {train_block.max_episodes:,}")
            print(f"    Episode start range: [{train_block.min_start_step}, {train_block.max_start_step}]")
            
            print(f"  Validation block: {val_block.block_id}")
            print(f"    Date range: {val_block.start_date} to {val_block.end_date}")
            print(f"    Days: {val_block.num_days:,}")
            print(f"    Episodes: {val_block.max_episodes:,}")
            print(f"    Episode start range: [{val_block.min_start_step}, {val_block.max_start_step}]")
        
        # Calculate and store sampling weights
        self._calculate_sampling_weights()
        
        # Print summary
        total_train_episodes = sum(block.max_episodes for block in self.train_blocks)
        total_val_episodes = sum(block.max_episodes for block in self.validation_blocks)
        
        print("\n" + "="*60)
        print("TIME SERIES SPLITTING COMPLETE")
        print("="*60)
        print(f"Training blocks: {len(self.train_blocks)}")
        print(f"Validation blocks: {len(self.validation_blocks)}")
        print(f"Total training episodes: {total_train_episodes:,}")
        print(f"Total validation episodes: {total_val_episodes:,}")
        print(f"Actual validation ratio: {total_val_episodes/(total_train_episodes + total_val_episodes):.1%}")
        print("="*60)

    def _calculate_sampling_weights(self):
        """Calculate sampling weights for blocks based on available days within each block."""
        # Training weights
        if self.train_blocks:
            train_days = np.array([block.num_days for block in self.train_blocks], dtype=np.float32)
            if train_days.sum() > 0:
                train_weights = train_days / train_days.sum()
            else:
                train_weights = np.ones(len(self.train_blocks), dtype=np.float32) / len(self.train_blocks)
        else:
            train_weights = np.array([], dtype=np.float32)
        
        # Validation weights
        if self.validation_blocks:
            val_days = np.array([block.num_days for block in self.validation_blocks], dtype=np.float32)
            if val_days.sum() > 0:
                val_weights = val_days / val_days.sum()
            else:
                val_weights = np.ones(len(self.validation_blocks), dtype=np.float32) / len(self.validation_blocks)
        else:
            val_weights = np.array([], dtype=np.float32)
        
        self.block_sampling_weights = {
            'train': train_weights,
            'validation': val_weights
        }
        
        print(f"\nSampling weights calculated:")
        print(f"  Training weights: {train_weights}")
        print(f"  Validation weights: {val_weights}")
    
    @staticmethod
    def _select_features_from_config(available_features: List[str], config: Dict[str, Any]) -> List[str]:
        """
        Select features based on configuration settings.
        Supports the new config sections: ``saa_features``, ``paa_asset_token_features``,
        and ``paa_portfolio_token_features``. The legacy ``features`` key is not accepted.
        
        Args:
            available_features: List of all available feature names from CSV
            config: Configuration dict with feature sections
            
        Returns:
            List of selected feature names (no duplicates)
        """
        feature_keys = [
            "saa_features",
            "paa_asset_token_features",
            "paa_portfolio_token_features",
        ]

        if "features" in config:
            raise ValueError(
                "Config key 'features' is deprecated. Use 'saa_features', "
                "'paa_asset_token_features', or 'paa_portfolio_token_features' instead."
            )

        feature_sections: Dict[str, Dict[str, Any]] = {}
        for key in feature_keys:
            section = config.get(key, {}) or {}
            if not isinstance(section, dict):
                raise ValueError(f"Config section '{key}' must be a mapping of feature flags.")
            feature_sections[key] = section

        # Detect duplicate feature definitions across sections
        # Note: Duplicates in the config feature keys are allowed for sorting/filtering operations later!
        # We just need to ensure each unique feature is loaded only once
        feature_origin: Dict[str, str] = {}
        duplicates_found: Dict[str, list] = {}
        
        for section_name, section in feature_sections.items():
            for feat_name in section:
                if feat_name in feature_origin:
                    # Track which sections define the same feature
                    if feat_name not in duplicates_found:
                        duplicates_found[feat_name] = [feature_origin[feat_name]]
                        duplicates_found[feat_name].append(section_name)
            else:
                feature_origin[feat_name] = section_name
        
        # Inform user about duplicates but don't raise error
        if duplicates_found:
            print("\nWarning: The following features are defined in multiple config sections:")
            for feat_name, sections in duplicates_found.items():
                print(f"  '{feat_name}' found in: {', '.join(sections)}")
                print("Each feature will be loaded only once to avoid redundant data loading.\n")

        if all(len(section) == 0 for section in feature_sections.values()):
            raise ValueError(
                "No features specified in config sections. Please enable at least one feature in "
                "'saa_features', 'paa_asset_token_features', or 'paa_portfolio_token_features'."    
            )

        # Flatten enabled flags while preserving the available_features order below
        combined_flags: Dict[str, bool] = {}
        for section in feature_sections.values():
            for feat_name, enabled in section.items():
                combined_flags[feat_name] = bool(enabled)

        selected_features: List[str] = []
        for feature_name in available_features:
            if feature_name in combined_flags:
                if combined_flags[feature_name]:
                    selected_features.append(feature_name)
            else:
                print(
                    f"Warning: Feature '{feature_name}' not found in provided feature sections in config. Skipping."
                )
        
        if not selected_features:
            raise ValueError("No features selected. Please enable at least one feature in the config sections.")
        
        return selected_features
    
    # NEW: LSTM-optimized observation functions
    def get_OHLCV_lookback(self, absolute_current_step: int, lookback_window: int) -> np.ndarray:
        """
        Get OHLCV data for lookback window optimized for LSTM input.
        
        Args:
            current_step: Current step index (0-based from episode start)
            lookback_window: Number of days to look back
            
        Returns:
            np.ndarray with shape [lookback_window, num_assets, 5] (OHLCV)
            If insufficient history, pads with zeros at the beginning
        """
        # Calculate actual date indices
        end_step = absolute_current_step  # Current step (exclusive)
        start_step = max(0, end_step - lookback_window)
        
        # Check bounds
        if end_step >= self.num_days or end_step < 0:
            return np.zeros((lookback_window, self.num_assets, 5), dtype=np.float32)
        
        # Get the actual window length available
        actual_window = end_step - start_step
        
        # Create output array
        ohlcv_window = np.zeros((lookback_window, self.num_assets, 5), dtype=np.float32)
        
        if actual_window > 0:
            # Extract OHLCV data
            ohlc_data = np.stack([
                self.open_prices[start_step:end_step],
                self.high_prices[start_step:end_step], 
                self.low_prices[start_step:end_step],
                self.close_prices[start_step:end_step],
                self.volumes[start_step:end_step]
            ], axis=-1)  # Shape: [actual_window, num_assets, 5]
            
            # Place in output array (pad at beginning if needed)
            padding_needed = lookback_window - actual_window
            ohlcv_window[padding_needed:] = ohlc_data
        
        return ohlcv_window
    
    def get_features_lookback(self, absolute_current_step: int, lookback_window: int) -> np.ndarray:
        """
        Get feature data for lookback window optimized for LSTM input.
        
        Args:
            current_step: Current step index (0-based from episode start)
            lookback_window: Number of days to look back
            
        Returns:
            np.ndarray with shape [lookback_window, num_assets, num_features]
            If insufficient history, pads with zeros at the beginning
        """
        # Calculate actual date indices  
        end_step = absolute_current_step  # Current step (exclusive)
        start_step = max(0, end_step - lookback_window)
        
        # Check bounds
        if end_step >= self.num_days or end_step < 0:
            return np.zeros((lookback_window, self.num_assets, self.num_features), dtype=np.float32)
        
        # Get the actual window length available
        actual_window = end_step - start_step
        
        # Create output array
        features_window = np.zeros((lookback_window, self.num_assets, self.num_features), dtype=np.float32)
        
        if actual_window > 0:
            # Extract feature data
            feature_data = self.features[start_step:end_step]  # Shape: [actual_window, num_assets, num_features]
            
            # Place in output array (pad at beginning if needed)
            padding_needed = lookback_window - actual_window
            features_window[padding_needed:] = feature_data
        
        return features_window
    
    # EXISTING: Keep the step-based functions
    def get_OHLCV_at_step(self, step_idx: int) -> Dict[str, np.ndarray]:
        """
        Get all asset OHLCV for a specific step efficiently.
        
        Returns:
            Dict with arrays: open, high, low, close, volume for all assets
        """
        if step_idx < 0 or step_idx >= self.num_days:
            raise ValueError(f"Step index {step_idx} out of range [0, {self.num_days})")
        
        return {
            'open': self.open_prices[step_idx],
            'high': self.high_prices[step_idx], 
            'low': self.low_prices[step_idx],
            'close': self.close_prices[step_idx],
            'volume': self.volumes[step_idx],
            'date': self.dates[step_idx]
        }
    
    def get_features_at_step(self, step_idx: int) -> np.ndarray:
        """Get all features for all assets at a specific step."""
        if step_idx < 0 or step_idx >= self.num_days:
            return np.zeros((self.num_assets, self.num_features), dtype=np.float32)
        
        return self.features[step_idx]
    
    # NEW: Utility functions
    def get_selected_feature_names(self) -> List[str]:
        """Get list of selected feature names."""
        return self.selected_feature_names.copy()
    
    def get_available_feature_names(self) -> List[str]:
        """Get list of all available feature names."""
        return self.available_feature_names.copy()
    
    def get_feature_selection_summary(self) -> Dict[str, Any]:
        """Get summary of feature selection."""
        return {
            'total_available': self.num_available_features,
            'selected': self.num_features,
            'selection_ratio': self.num_features / self.num_available_features,
            'selected_features': self.selected_feature_names,
            'excluded_features': [f for f in self.available_feature_names if f not in self.selected_feature_names]
        }
    
    @staticmethod
    def _estimate_memory_mb(num_days: int, num_assets: int, num_features: int) -> float:
        """Estimate memory usage in MB for selected features only."""
        arrays_memory = (
            num_days * num_assets * 5 +  # OHLCV arrays
            num_days * num_assets * num_features  # Selected features array only
        ) * 4  # 4 bytes per float32
        
        return arrays_memory / (1024 * 1024)

    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate cached data quality and return statistics."""
        close_nan_pct = np.isnan(self.close_prices).mean() * 100
        features_nan_pct = np.isnan(self.features).mean() * 100
        volume_zero_pct = (self.volumes == 0).mean() * 100
        
        # Check for missing data by asset
        asset_completeness = {}
        for i, asset in enumerate(self.asset_names):
            asset_close_data = self.close_prices[:, i]
            completeness = (~np.isnan(asset_close_data)).mean() * 100
            asset_completeness[asset] = completeness
        
        return {
            'close_prices_nan_pct': close_nan_pct,
            'features_nan_pct': features_nan_pct,
            'volume_zero_pct': volume_zero_pct,
            'date_range': (str(self.dates[0]), str(self.dates[-1])),
            'total_memory_mb': self._estimate_memory_mb(self.num_days, self.num_assets, self.num_features),
            'asset_completeness': asset_completeness,
            'min_completeness': min(asset_completeness.values()) if asset_completeness else 0.0,
            'assets_with_gaps': [asset for asset, comp in asset_completeness.items() if comp < 95.0],
            'feature_selection': self.get_feature_selection_summary()
        }
    
    def sample_episode_start(self, mode: str = 'train', random_seed: Optional[int] = None) -> Tuple[str, int]:
        """
        Sample a random episode start from appropriate blocks.
        
        Args:
            mode: 'train' or 'validation'
            random_seed: Optional random seed for reproducibility
            
        Returns:
            Tuple of (block_id, absolute_start_step)
        """
        rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng()
        
        blocks = self.train_blocks if mode == 'train' else self.validation_blocks
        weights = self.block_sampling_weights[mode]
        
        if len(blocks) == 0:
            raise ValueError(f"No {mode} blocks available")
        
        if len(weights) == 0 or weights.sum() == 0:
            raise ValueError(f"Invalid sampling weights for {mode} blocks")
        
        # Sample block based on weights
        block_idx = rng.choice(len(blocks), p=weights)
        selected_block = blocks[block_idx]
        
        # Sample start step within selected block
        if selected_block.max_start_step <= selected_block.min_start_step:
            start_step = rng.integers(selected_block.min_start_step, selected_block.min_start_step + 1)
        else:
            start_step = rng.integers(
                selected_block.min_start_step, 
                selected_block.max_start_step + 1
            )
        
        return selected_block.block_id, start_step
    

@dataclass
class TrainingMetrics:
    """Collect metrics across training episodes for monitoring"""

    # Pre-allocated circular buffers
    buffer_size: int
    
    episode_returns: np.ndarray         # [buffer_size] - total episode returns
    episode_rewards: np.ndarray         # [buffer_size] - total RL rewards  
    episode_sharpe: np.ndarray          # [buffer_size] - episode Sharpe ratios
    episode_max_dd: np.ndarray          # [buffer_size] - max drawdowns
    episode_turnover: np.ndarray        # [buffer_size] - portfolio turnover rates
    episode_costs: np.ndarray           # [buffer_size] - total transaction costs
    final_portfolio_values: np.ndarray  # [buffer_size] - final portfolio values


@dataclass 
class ExecutionResult:
    """Lightweight trade execution result tracking. Serves as a kind of blotter"""
    current_step: int
    trades_executed: np.ndarray         # [num_assets] - number of shares traded per asset
    executed_prices: np.ndarray         # [num_assets] - prices at which trades were executed
    transaction_cost: float             # Total transaction cost incurred in execution step
    success: bool                       # Whether execution was successful
    traded_dollar_value: float  # Total dollar value traded in this step
    traded_shares_total: float
    traded_notional_per_asset: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))  # [num_assets] - dollar value traded per asset


class TradingEnv(gym.Env):
    """
    Main Gym environment class for (multi-)asset trading optimized for DRL.
    
    Key Features:
    - Supports multiple execution modes (simple, tranche, portfolio)
    - Realistic trading constraints and costs
    - Efficient state representation for neural networks
    - Proper reward shaping for stable PPO training
    """

    metadata = {'render.modes': ['human']}
    
    def __init__(self, config: Dict[str, Any], market_data_cache: MarketDataCache,
    mode: str = 'train'):
        """
        Initialize trading environment with configuration and pre-built market data cache.
        
        Args:
            config: Configuration dictionary with environment parameters
            market_data_cache: Pre-built MarketDataCache instance with time series blocks
            mode: 'train' or 'validation' - determines which blocks to sample from
        """
        super(TradingEnv, self).__init__()
        
        # Configuration parameters
        self.episode_length_days = config["environment"]["episode_length_days"]
        self.max_reward_risk_window = int(min(self.episode_length_days // 2, 63))
        self.lookback_window = config["environment"]["lookback_window"]
        self.initial_portfolio_value = config["environment"]["initial_portfolio_value"]
        self.early_stopping_threshold = config["environment"]["early_stopping_threshold"]
        self.cash_drag_rate_pa = config["environment"]["cash_drag_rate_pa"]
        
        self.seed = config["environment"]["seed"]
        # Execution parameters
        self.execution_weight_change_threshold = config["environment"]["execution_weight_change_threshold"]
        self.execution_min_trade_value_threshold = config["environment"]["execution_min_trade_value_threshold"]
        self.execution_min_days_between_trades = config["environment"]["execution_min_days_between_trades"]              # e.g., 0.0005
        self.maybe_provide_sequence = config['environment']['maybe_provide_sequence']  # Whether to provide sequence data in observations
        self.sortino_net_reward_mix = config["environment"]["sortino_net_reward_mix"]
        self.lambda_drawdown = config["environment"]["lambda_drawdown"]
        # Portfolio concentration control coefficient used in allocator reward.
        # Applied to normalized executed weights as:
        # -lambda_spread * sum_i(w_i * log(w_i + 1e-8)).
        self.lambda_spread = float(config["environment"].get("lambda_spread", 0.0))
        self.lambda_transaction_cost = float(config["environment"].get("lambda_transaction_cost", 0.0))

        self.previous_max_drawdown = None
        self.saa_previous_max_drawdown = None
        self.trans_act_pen = None  # Initialize transaction action penalty variable

        # Store references
        self.market_data_cache = market_data_cache
        self.config = config
        self.mode = mode  # 'train' or 'validation'
        self.threshold_val = self.initial_portfolio_value * self.early_stopping_threshold

        # Execution-mode config (backwards compatible defaults)
        self.execution_mode = config["environment"]["execution_mode"] # "single_asset_target_pos" | "simple" | "tranche" | "portfolio_weights"
        if self.execution_mode == "portfolio_weights":
            self.min_initial_cash_allocation = config["environment"]["percentage_of_cash_only_starts"]
        self.quantity_type = config["environment"].get("quantity_type", "shares")
        self.price_source = config["environment"].get("price_source", "next_open")  # "next_open" | "current_close"
        self.allow_short = bool(config["environment"].get("allow_short", False))
        self.max_position_shares_per_symbol = config["environment"].get("max_position_shares_per_symbol", None)
        if self.quantity_type != "shares":
            raise ValueError("Only 'shares' quantity_type is supported in simple/tranche modes.")
        if self.execution_mode not in {EXECUTION_SINGLE_ASSET_TARGET_POS, EXECUTION_SIMPLE, EXECUTION_TRANCHE, EXECUTION_PORTFOLIO_WEIGHTS}:
            raise ValueError(f"Invalid execution_mode: {self.execution_mode}")
        
        self.saa_initial_subportfolio_value = None  # For SAA benchmark
        
        self.selected_asset_index = None  # For SINGLE_ASSET_TARGET_POS mode
        
        self.perc_of_cash_only_starts = config["environment"].get("percentage_of_cash_only_starts", 0.2)
        self.action_l2_penalty_coeff = config['training'].get('action_l2_penalty_coeff', 0.01)
        self.action_limiting_factor_start = config['training'].get('action_limiting_factor_start', 0.2)

        # Differential Sortino config params
        self.sortino_eta = config["environment"].get("sortino_eta", 0.0125) # Adaption rate

        # Sortino reward components
        self.previous_sortino = None
        self.running_mean_ema = None
        self.running_downside_variance_ema = None

        # Sortino SAA reward metrics
        self.saa_previous_sortino = None
        self.saa_running_mean_ema = None
        self.saa_running_downside_variance_ema = None
        
        # Initialize state variables
        self.current_step = None
        self.current_episode = None
        self.current_absolute_step = None
        self.last_execution_step = -1
  
        # Create portfolio state instance. Correct values will be filled in reset and step!
        self.portfolio_state = PortfolioState(
            cash=0.0,
            positions=np.array([0.0] * self.market_data_cache.num_assets),
            prices=np.array([0.0] * self.market_data_cache.num_assets), 
            step=0,                      # Initial step
            terminated=False             # Not terminated
        )

        self.comparison_portfolio_state = PortfolioState(
            cash=0.0,
            positions=np.array([0.0] * self.market_data_cache.num_assets),
            prices=np.array([0.0] * self.market_data_cache.num_assets), 
            step=0,                      # Initial step
            terminated=False             # Not terminated
        )

        # Create benchmark portfolio state instance
        self.benchmark_portfolio_state = PortfolioState(
            cash=0.0,
            positions=np.array([0.0] * self.market_data_cache.num_assets),
            prices=np.array([0.0] * self.market_data_cache.num_assets),
            step=0,                      # Initial step
            terminated=False             # Not terminated
        )

        # Create EpisodeBuffer instance. Only pass required arguments; __post_init__ will handle array initialization.
        if self.maybe_provide_sequence:
            required_buffer_size_days = self.lookback_window + self.episode_length_days
        else:
            required_buffer_size_days = self.episode_length_days
        self.episode_buffer = EpisodeBuffer(
            episode_buffer_length_days=required_buffer_size_days,
            num_assets=self.market_data_cache.num_assets, lookback_window=self.lookback_window,
            maybe_provide_sequence=self.maybe_provide_sequence
        )

        # Define action and observation spaces
        self._setup_spaces()


    def _setup_spaces(self):
        """Define action and observation spaces."""
        num_assets = self.market_data_cache.num_assets
        num_features = self.market_data_cache.num_features
        num_portfolio_features = self.episode_buffer.num_portfolio_features


        # ------ Action space design ------

        if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            # ------ Action space: Target position for single selected asset ------
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        elif self.execution_mode == EXECUTION_PORTFOLIO_WEIGHTS:
            # ------ Action space: Continuous weights for each asset + cash (sum to 1) ------
            self.action_space = spaces.Box(
                low=-2.0, high=2.0, shape=(num_assets,), dtype=np.float32
            )
        else:
            # Simple/Tranche modes: action is a list of instructions; Gym does not have a list space.
            # Keep a generic Box to satisfy Gym, but we validate structure in step().
            self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)


        # ------ Observation space design ------

        if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            asset_obs_size = num_features # Single step asset features for selected asset
            portfolio_obs_size = 4 # log cash ratio, log asset ratio, day return just agent relevant, last_action

        else:
            if self.maybe_provide_sequence:
                # 1. Asset features: [lookback_window, num_assets, num_features]
                asset_obs_size = self.lookback_window * num_assets * num_features
                
                # 2. Portfolio features: [lookback_window, num_portfolio_features]
                #    Portfolio-level metrics:
                #    weights + alpha + sharpe + drawdown + volatility + turnover + allocator_rewards
                portfolio_obs_size = self.lookback_window * num_portfolio_features
            else:
                # 1. Asset features: [num_assets, num_features]
                asset_obs_size = num_assets * num_features # Single step asset features
                # 2. Portfolio features: [num_portfolio_features]
                portfolio_obs_size = num_portfolio_features  # Single step portfolio features
            
        # Single flattened observation space for maximum performance
        total_obs_size = asset_obs_size + portfolio_obs_size
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_obs_size,), 
            dtype=np.float32
        )
        
        print(f"\nObservation space setup complete: {self.observation_space}")

        # Store dimensions for observation construction and splitting
        self.asset_obs_size = asset_obs_size
        self.portfolio_obs_size = portfolio_obs_size
        self.num_asset_features = num_features
        self.num_portfolio_features = num_portfolio_features

    def _reconstruct_observation(self, flat_observation):
        """
        Reconstruct asset and portfolio features from flattened observation.
        
        Args:
            flat_observation: Flattened numpy array of shape (total_obs_size,)
        
        Returns:
            tuple: (asset_features, portfolio_features)
        """
        # Split the flattened observation
        asset_features_flat = flat_observation[:self.asset_obs_size]
        portfolio_features_flat = flat_observation[self.asset_obs_size:]
        
        # Reshape asset features to original dimensions
        # Shape: [lookback_window, num_assets, num_features + 3]
        asset_features = asset_features_flat.reshape(
            self.lookback_window, 
            self.market_data_cache.num_assets, 
            self.num_asset_features
        )
        
        # Reshape portfolio features
        # Shape: [lookback_window, num_portfolio_features]
        portfolio_features = portfolio_features_flat.reshape(
            self.lookback_window,
            self.num_portfolio_features
        )
        
        return asset_features, portfolio_features

    def _calculate_portfolio_metrics(self):
        """Calculate portfolio-level metrics for observation space."""
        # Placeholder for actual metrics calculation
        print("\nPortfolio metric calc not yet implemented! This is a placeholder")
        portfolio_metrics = [
            'total_return', 'return_per_asset','sharpe_ratio', 'max_drawdown', 'volatility', 'turnover'
        ]
        return portfolio_metrics

    def reset(self, seed: Optional[int] = None, option: Optional[Dict] = None, asset: Optional[str] = None):
        """
        Reset environment to start a new episode.
        
        Args:
            seed: Optional random seed for reproducibility
            options: Optional reset options

        Returns: 
            tuple: (initial_observation, info_dict)

        Process:
        0. Reset episode step counter
        1. Sample a new random episode with random start from market data cache
        2. Initialize portfolio state (cash, holdings, etc.)
        3. Init metrics tracking
        4. Return initial observation
        """
        # Call parent reset
        super().reset(seed=seed)
        
        # Step 0: Reset episode step counter and last executed trade step. Also reset diff soretino reward params
        self.current_step = 0
        self.last_execution_step = -1
        self.current_episode = (self.current_episode + 1) if self.current_episode is not None else 0
        
        self.previous_sortino = 0.0
        self.running_mean_ema = 1e-4
        self.running_downside_variance_ema = 2.5e-5

        # Sortino SAA reward metrics
        self.saa_previous_sortino = 0.0
        self.saa_running_mean_ema = 1e-4
        self.saa_running_downside_variance_ema = 2.5e-5

        # Init & reset episode accumulators for sortino diagnostics
        self._sortino_mean_hist, self._sortino_down_hist, self._sortino_raw_hist = [], [], []

        
        if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            
            # If asset specified in reset options, validate and set
            if asset is not None:
                if asset not in self.market_data_cache.asset_names:
                    raise ValueError(f"Specified asset '{asset}' not in available assets.")
                
                # Get selected assets index
                self.selected_asset_index = self.market_data_cache.asset_to_index[asset] 
            else:
                raise ValueError("In SINGLE_ASSET_TARGET_POS mode, 'asset' parameter must be specified in reset().")

        # Step 1: Sample new episode start from appropriate blocks
        self.current_block_id, self.current_episode_start_step = self.market_data_cache.sample_episode_start(
            mode=self.mode, random_seed=seed
        )

        # Calculate absolute step in full dataset. Very important for correct market data retrieval!
        self.current_absolute_step = int(self.current_episode_start_step + self.current_step)

        # Reset episode buffer
        self.episode_buffer.reset_episode_buffer() # fills buffer with 0s

        # Only trigger this when a sequence is needed. Not needed in RecurrentPPO
        if self.maybe_provide_sequence:
            # Warmup market data cache to ensure all data is ready
            self.episode_buffer.warmup_market_data(self.market_data_cache, self.current_absolute_step)
        
        # Step 2: Initialize portfolio state using PortfolioState dataclass
        # Get initial prices for portfolio valuation
        initial_prices = self._get_current_prices(self.current_absolute_step)
        
        num_assets = self.market_data_cache.num_assets
        initial_positions = np.zeros(num_assets, dtype=np.float32)
        initial_cash = self.initial_portfolio_value  # Default to all cash
        total_init_tc = 0.0  # Track initialization costs

        self.previous_max_drawdown = float(0.0)
        self.saa_previous_max_drawdown = float(0.0)
        self.trans_act_pen = float(0.0)

        # Episode accumulators for diagnostics
        self._ep_turnover_notional = 0.0
        self._ep_cost_commission = 0.0
        self._ep_cost_spread = 0.0
        self._ep_cost_impact = 0.0
        self._ep_cost_fixed = 0.0
        self._ep_buy_notional = 0.0
        self._ep_sell_notional = 0.0
        self._ep_trade_sizes = []  # absolute traded notional per leg
        self._ep_action_outputs = []  # raw single-asset action outputs
        self._ep_exposure_sum = 0.0
        self._ep_exposure_steps = 0

        # Shadow (frictionless) portfolio mirrors live trades without costs
        self.shadow_portfolio_state = PortfolioState(
            cash=self.initial_portfolio_value,
            positions=np.zeros(num_assets, dtype=np.float32),
            prices=initial_prices.copy(),
            step=self.current_step,
            terminated=False,
        )

        # Single-asset target position mode: cash-only or random allocation between cash and assets
        if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            # Single asset tradable only! Other assets are initialised (including costs) but not traded!
            # Uses perc_of_cash_only_starts (range: 0-1) to determine cash-only or random allocation start
            if np.random.random() < self.perc_of_cash_only_starts:
                # Cash-only start (no transaction costs)
                initial_cash = self.initial_portfolio_value
                initial_positions[:] = 0.0 # all assets zero
                total_init_tc = 0.0
                # Store initial subportfolio values for SAA mode (cash only, no selected asset position)
                self.saa_initial_cash = initial_cash
                self.saa_selected_asset_value = 0.0
                self.saa_initial_subportfolio_value = self.saa_initial_cash + self.saa_selected_asset_value
            else:
                # Random allocation between cash and assets
                rand_weights = np.random.dirichlet(np.ones(self.market_data_cache.num_assets + 1))
                
                # Calculate target positions
                cash_weight = rand_weights[0]
                assets_weights = rand_weights[1:] # weights for all assets
                assets_notional = self.initial_portfolio_value * assets_weights # notional dollar vals

                all_positive = (initial_prices > 0).all()
                if all_positive:
                    assets_shares = assets_notional / initial_prices
                else:
                    raise ValueError(f"Initial price for selected asset index {self.selected_asset_index} is zero or negative.")
                
                # Build target positions vector
                target_positions = np.zeros(num_assets, dtype=np.float32)
                target_positions[:] = assets_shares # all assets
                
                # Apply transaction costs with iterative downscaling to prevent negative cash
                initial_cash, initial_positions, total_init_tc = self._initialize_portfolio_with_costs(
                    target_positions=target_positions,
                    initial_prices=initial_prices,
                    initial_value=self.initial_portfolio_value,
                    allow_cash_residual=False,
                    max_iterations=10
                )
                
                # Safety check: if cash is still negative, scale down positions proportionally
                if initial_cash < 0:
                    shortfall = abs(initial_cash)
                    # Calculate scale factor to reduce positions by the cash shortfall amount
                    # shortfall_ratio = shortfall / initial_portfolio_value
                    # Scale positions down proportionally
                    scale_factor = max(0.0, 1.0 - (shortfall / self.initial_portfolio_value))
                    initial_positions = initial_positions * scale_factor
                    
                    # Recalculate costs with scaled positions
                    total_init_tc = self._calculate_transaction_costs(
                        shares_traded=initial_positions,
                        prices=initial_prices,
                        abs_step=self.current_absolute_step,
                        asset_mask=None
                    )
                    
                    # Recalculate cash
                    position_notional_total = np.sum(initial_positions * initial_prices)
                    initial_cash = self.initial_portfolio_value - position_notional_total - total_init_tc
                    
                    print(f"[Portfolio Init Info] Scaled positions down by {(1.0 - scale_factor) * 100:.2f}% "
                          f"to avoid negative cash. Final init cash: ${initial_cash:.4f}")
                    
                # Store initial subportfolio values for SAA mode
                self.saa_initial_cash = initial_cash
                self.saa_selected_asset_value = initial_positions[self.selected_asset_index] * initial_prices[self.selected_asset_index]
                self.saa_initial_subportfolio_value = self.saa_initial_cash + self.saa_selected_asset_value

        # Portfolio mode: random weights across all assets + cash, ensuring minimum cash allocation
        else:
            # Calculate initial cash allocation ensuring minimum threshold
            # Generate random portfolio weights including cash
            random_weights = np.random.random(num_assets + 1)  # +1 for cash
            random_weights = random_weights / random_weights.sum()  # Normalize to sum to 1
            
            # Ensure minimum cash allocation
            cash_weight = random_weights[0]
            if cash_weight < self.min_initial_cash_allocation:
                # Set cash to minimum and rescale other weights
                cash_weight = self.min_initial_cash_allocation
                asset_weights = random_weights[1:]
                asset_weights_sum = asset_weights.sum()
                
                if asset_weights_sum > 0:
                    # Rescale asset weights to fit remaining allocation
                    remaining_allocation = 1.0 - cash_weight
                    asset_weights = asset_weights * (remaining_allocation / asset_weights_sum)
                else:
                    # If all asset weights were zero, distribute remaining equally
                    remaining_allocation = 1.0 - cash_weight
                    asset_weights = np.full(num_assets, remaining_allocation / num_assets)
                
                # Update the weights array
                random_weights = np.concatenate(([cash_weight], asset_weights))
            
            # Calculate target positions from weights
            asset_weights = random_weights[1:]
            target_positions = np.zeros(num_assets, dtype=np.float32)
            
            for i in range(num_assets):
                assets_notional = self.initial_portfolio_value * asset_weights[i]
                if assets_notional > 0 and initial_prices[i] > 0:
                    target_positions[i] = assets_notional / initial_prices[i]
            
            # Apply transaction costs (reserves intended cash weight from portfolio value)
            intended_cash = self.initial_portfolio_value * random_weights[0]
            available_for_assets = self.initial_portfolio_value - intended_cash
            
            initial_cash, initial_positions, total_init_tc = self._initialize_portfolio_with_costs(
                target_positions=target_positions,
                initial_prices=initial_prices,
                initial_value=available_for_assets,
                allow_cash_residual=True,
                max_iterations=10
            )
            # Add back the intended cash reserve
            initial_cash += intended_cash

        # from here same logic for all modes ----------------------------------

        # Update portfolio state instance
        self.portfolio_state.portfolio_reset(
            cash=initial_cash,
            positions=initial_positions,
            prices=initial_prices,
            step=self.current_step,
            terminated=False
        )

        # update comparison portfolio in the same way to be able to verify if trades are giving alpha vs initialisation
        init_cash_comparison = initial_cash
        init_positions_comparison = initial_positions.copy()

        self.comparison_portfolio_state.portfolio_reset(
            cash=init_cash_comparison,
            positions=init_positions_comparison,
            prices=initial_prices,
            step=self.current_step,
            terminated=False
        )

        # Update Benchmark Portfolio (custom allocation, no cash - fully invested)
        benchmark_target_weights = {
            "SPY": 0.45,
            "Gold": 0.20,
            "Crude": 0.05,
            "EWJ": 0.10,
            "EWG": 0.10,
            "EWQ": 0.05,
            "EWT": 0.05
        }

        asset_names = self.market_data_cache.asset_names
        symbol_to_idx = self.market_data_cache.asset_to_index

        # Build weight vector (length = num_assets), zeros default
        bench_weights = np.zeros(len(asset_names), dtype=np.float32)
        present_weight_sum = 0.0
        for sym, w in benchmark_target_weights.items():
            if sym in symbol_to_idx:
                idx = symbol_to_idx[sym]
                bench_weights[idx] = w
                present_weight_sum += w
            else:
                print(f"[Benchmark] Warning: symbol '{sym}' not in asset list; skipping.")

        # Renormalize if any symbol missing
        if present_weight_sum <= 0:
            raise ValueError("Benchmark allocation failed: no target symbols present.")
        if abs(present_weight_sum - 1.0) > 1e-6:
            bench_weights /= present_weight_sum  # scale remaining to sum 1

        # Convert weights to target positions (gross, before costs)
        benchmark_target_positions = np.zeros(num_assets, dtype=np.float32)
        for i in range(num_assets):
            if bench_weights[i] > 0 and initial_prices[i] > 0:
                allocation = self.initial_portfolio_value * bench_weights[i]
                benchmark_target_positions[i] = allocation / initial_prices[i]

        # Apply transaction costs with full investment constraint (allow small cash residual)
        benchmark_cash, benchmark_positions, benchmark_tc = self._initialize_portfolio_with_costs(
            target_positions=benchmark_target_positions,
            initial_prices=initial_prices,
            initial_value=self.initial_portfolio_value,
            allow_cash_residual=True,  # Key difference: allows small residual
            max_iterations=10
        )

        self.benchmark_portfolio_state.portfolio_reset(
            cash=benchmark_cash,  # Will be small (< $1) or zero
            positions=benchmark_positions,
            prices=initial_prices,
            step=self.current_step,
            terminated=False
        )

        # Validate initial portfolio value matches configuration (allowing for small rounding)
        actual_initial_value = self.portfolio_state.get_total_value()
        if abs(actual_initial_value - self.initial_portfolio_value) > 100:  # $100 tolerance for TC rounding
            print(f"\nPortfolio initialization error: {actual_initial_value} != {self.initial_portfolio_value}")

        # Validate comparison portfolio value matches configuration (allowing for small rounding)
        actual_initial_value = self.comparison_portfolio_state.get_total_value()
        if abs(actual_initial_value - self.initial_portfolio_value) > 100:  # $100 tolerance for TC rounding
            print(f"\nComparison portfolio initialization error: {actual_initial_value} != {self.initial_portfolio_value}")

        # Validate benchmark portfolio value matches configuration (allowing for small rounding)
        actual_initial_value = self.benchmark_portfolio_state.get_total_value()
        if abs(actual_initial_value - self.initial_portfolio_value) > 100:  # $100 tolerance for TC rounding
            print(f"Benchmark portfolio initialization error: {actual_initial_value} != {self.initial_portfolio_value}")

        # Seed a pre-step entry in the EpisodeBuffer so the first observation contains real weights
        initial_weights = self.portfolio_state.get_weights().astype(np.float32)
        zero_action = np.zeros(self.market_data_cache.num_assets + 1, dtype=np.float32)
        sum_sq = float(np.sum(initial_weights[1:] ** 2))
        if sum_sq < 1e-8:
            effective_asset_concentration_norm = 0.0
        else:
            effective_asset_concentration_norm = float(1.0 / sum_sq) /  float(len(initial_weights[1:]))

        self.episode_buffer.record_step(
            external_step=0, 
            portfolio_value=actual_initial_value,
            weights=initial_weights,
            portfolio_positions=self.portfolio_state.positions.copy(),
            daily_return=0.0,
            saa_return=0.0,
            reward_to_record=0.0,
            action=zero_action,
            transaction_cost=0.0,
            prices=initial_prices,
            sharpe_ratio=0.0,
            drawdown=0.0,
            volatility=0.0,
            turnover=0.0,
            alpha=0.0,
            benchmark_portfolio_value=self.benchmark_portfolio_state.get_total_value(),
            comparison_portfolio_value=self.comparison_portfolio_state.get_total_value(),
            effective_asset_concentration_norm=effective_asset_concentration_norm
        )
        
        # Initialize previous portfolio value for return calculation
        self._previous_portfolio_value = actual_initial_value
        
        # Step 4: Get initial observation and info
        if self.maybe_provide_sequence:
            # Provide full lookback sequence for initial observation
            initial_observation = self.get_observation_sequence()
        else:
            # Provide single step observation
            initial_observation = self.get_observation_single_step()
        info = self._get_info()
        
        # Add reset-specific info
        info.update({
            'reset_type': 'episode_start',
            'block_id': self.current_block_id,
            'absolute_start_step': self.current_episode_start_step,
            'initial_cash': initial_cash,
            'initial_portfolio_value': actual_initial_value,
            'initial_date': self.market_data_cache.dates[self.current_episode_start_step]
        })
        
        return initial_observation, info

    def _get_current_prices(self, current_absolute_step: int) -> np.ndarray:
        """
        Get current asset prices for portfolio valuation.
        
        Returns:
            np.ndarray: Current closing prices for all assets [num_assets]
        """
        
        # Bounds checking
        if current_absolute_step >= self.market_data_cache.num_days:
            # Return last available prices if we're at the end
            return self.market_data_cache.close_prices[-1].copy()
        
        if current_absolute_step < 0:
            # Return first available prices if somehow negative
            return self.market_data_cache.close_prices[0].copy()
        
        return self.market_data_cache.close_prices[current_absolute_step].copy()

    def _get_info(self):
        """Get info dictionary for current step."""
        # Clamp absolute index to dataset bounds to avoid OOB on final step
        abs_idx = self.current_episode_start_step + self.current_step
        abs_idx = max(0, min(abs_idx, self.market_data_cache.num_days - 1))

        info = {
            'step': self.current_step,
            'block_id': self.current_block_id,
            'absolute_step': abs_idx,
            'date': self.market_data_cache.dates[abs_idx],
            'cash': self.portfolio_state.cash,
            'positions': self.portfolio_state.positions.copy(),
            'portfolio_value': self.portfolio_state.get_total_value(),
            'prices': self.portfolio_state.prices.copy()
        }
        return info

    def step(self, action: np.array, asset: Optional[str] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step in the environment.

        MODES:
            Single_asset_target_pos mode:
                Action is target position for selected asset only; execute trade to reach target
                Only the given asset is tradable; other assets are held but not traded. This allows 
                for focused learning on single-asset dynamics and clearer attribution of rewards to actions.
            
            Portfolio mode:
                Function represents a single day in the trading simulation where:
                1. The agent provides portfolio allocation actions
                2. Portfolio is rebalanced according to actions
                3. Time advances by one day (price changes occur)
                4. Rewards are calculated based on performance
                5. Termination conditions are checked

                Args:
                    action: Portfolio allocation weights [num_assets + 1] including cash
                        Must sum to 1.0, representing target portfolio weights

            Simple/Tranche mode:
                Actions must be list[TradeInstruction|dict]; execute per-instruction

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: Next state representation for agent
                - reward: Scalar reward for this step
                - terminated: Boolean indicating if episode ended due to conditions
                - truncated: Boolean indicating if episode ended due to time limit
                - info: Dictionary with additional step information

        NOTE: In futur version the execution decisions will be made by additional agent.
              Currently they will be performed automatically, if trading thresholds are reached.

        Process in portfolio mode:
        - Validate action (non-negative, sums to 1). Do not normalize, as this should be done by agent!
        - Gather current portfolio state
        - Placeholder for future EXECUTOR logic: execute trading mechanics. 
          Convention: use same days closing price for buying!
        - ADVANCE TIME -----------------------------------------------------
        - Calculate reward based on portfolio value change
        - Check termination conditions (early stopping, max steps)
        - Record step data
        - prepare next observation and info
        - Return all step outputs (observation, reward, terminated, truncated, info)
        """

        # Gather current portfolio state----------------------------
        # Store current state before any changes
        portfolio_value_before = self.portfolio_state.get_total_value()
        benchmark_portfolio_value_before = self.benchmark_portfolio_state.get_total_value()
        comparison_portfolio_value_before = self.comparison_portfolio_state.get_total_value()
        live_portfolio_cash_before = self.portfolio_state.cash

        trade_results: List[Dict[str, Any]] = []

        # Default entropy initialization; override in specific modes when computed
        diagnostic_entropy = 0.0
        raw_logits_mean = 0.0
        raw_logits_std = 0.0
        raw_logits_max = 0.0
        raw_logits_l2 = 0.0
        softmax_temp = 0.0
        
        # --------------------- Branch based on execution mode -------------

        # ----- Single-asset target position mode -----
        if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            if asset is None:
                raise ValueError("In EXECUTION_SINGLE_ASSET_TARGET_POS mode, 'asset' parameter must be specified in step().")
            if asset not in self.market_data_cache.asset_names:
                raise ValueError(f"Specified asset '{asset}' not in available assets.")
            selected_asset_index = self.market_data_cache.asset_to_index[asset]
            selected_asset_notional_before = self.portfolio_state.positions[selected_asset_index] * self.portfolio_state.prices[selected_asset_index]

            # Validate action input & target position change ------------------------------------
            if not isinstance(action, (list, tuple, np.ndarray)):
                raise ValueError("Action must be a list, tuple, or numpy array.")
            if len(action) != 1:
                raise ValueError("Action length must be 1 for EXECUTION_SINGLE_ASSET_TARGET_POS mode.")
            
            # Get desired target position change
            target_position_change = float(action[0])  # scalar target position change for selected asset in [-1.0, 1.0]
            # Validate target position change
            if np.isnan(target_position_change) or np.isinf(target_position_change):
                raise ValueError("Action contains NaN or Inf values.")
            
            # Execute trade based on provided target position change
            execution_result = self.execute_single_asset_target_position(
                asset_index=selected_asset_index,
                target_position_change=target_position_change,
                portfolio_state=self.portfolio_state
            )

        # ----- Portfolio mode: action is target weights vector -----
        elif self.execution_mode == EXECUTION_PORTFOLIO_WEIGHTS:
            # Get and Validate action input ------------------------------------
            raw_action = np.asarray(action, dtype=np.float32)
            if np.any(np.isnan(action)):
                raise ValueError("Raw action input contains NaN values")
            if raw_action.ndim != 1:
                raise ValueError(f"Action must be 1D, got shape {raw_action.shape}")
            if len(raw_action) != self.market_data_cache.num_assets:
                raise ValueError(f"Action length must be {self.market_data_cache.num_assets}, got {len(raw_action)}")
            # Explicit fixed anchor: cash logit is always 0
            cash_logit = 0.0

            # If policy output is non-finite, HOLD current weights (safer than forced all-cash jump)
            if not np.all(np.isfinite(raw_action)):
                weights = self.portfolio_state.get_weights().astype(np.float32)
                print("Warning: Non-finite action output detected; defaulting to HOLD (current weights).")
            else:
                # Build full logits [cash, assets...]
                full_logits = np.empty(raw_action.size + 1, dtype=np.float64)
                full_logits[0] = cash_logit
                full_logits[1:] = raw_action

                # Stable softmax shift (equivalent to max(0, max(asset_logits)))
                full_logits -= np.max(full_logits)

                exp_logits = np.exp(full_logits)
                den = np.sum(exp_logits)

                # Defensive fallback if denominator is pathological
                if (not np.isfinite(den)) or (den <= 0.0):
                    weights = self.portfolio_state.get_weights().astype(np.float32)
                else:
                    weights = (exp_logits / den).astype(np.float32)

            # Final numerical hygiene only
            sum_w = float(np.sum(weights, dtype=np.float64))
            if (not np.isfinite(sum_w)) or (sum_w <= 0.0):
                weights = self.portfolio_state.get_weights().astype(np.float32)
            else:
                weights /= sum_w
                
            # legacy, dont know exactly, didnt bother
            weight_change_target = weights

            # Diagnostics
            raw_logits_mean = float(np.mean(raw_action))
            raw_logits_std  = float(np.std(raw_action))
            raw_logits_max  = float(np.max(raw_action))
            raw_logits_l2   = float(np.linalg.norm(raw_action))
            softmax_var     = float(np.var(raw_action))
            softmax_temp    = float(np.sqrt(softmax_var) + 1e-8)  # indicative temperature

            # Allocation entropy from normalized executed weights
            diagnostic_entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-8, 1.0))))                

            # Execution ----------------------
            execution_result = self.execute_portfolio_change(
                target_weights=weights,
                portfolio_state=self.portfolio_state
            )

        # ------- Simple/Tranche instruction path -------
        else:
            # Action must be a list of TradeInstruction or dicts
            if not isinstance(action, (list, tuple)):
                raise ValueError("In simple/tranche modes, action must be a list of TradeInstruction or dicts.")
            # Normalize dicts to TradeInstruction
            instructions: List[TradeInstruction] = []
            for item in action:
                if isinstance(item, TradeInstruction):
                    instructions.append(item)
                elif isinstance(item, dict):
                    instructions.append(TradeInstruction(
                        symbol=item.get("symbol"),
                        action=item.get("action"),
                        quantity=item.get("quantity", None),   # do not cast; allow None
                        notional=item.get("notional", None),   # include notional
                        order_type=item.get("order_type", "MARKET"),
                        limit_price=item.get("limit_price", None),
                    ))
                else:
                    # Skip malformed entry
                    print(f"Warning: Skipping malformed action entry: {item}")
                    continue

            # Execute instructions and collect trade results
            execution_result, trade_results = self.execute_instructions(instructions)
            # Simple entropy proxy: fraction of assets touched
            touched = np.zeros(self.market_data_cache.num_assets, dtype=np.float32)
            for tr in trade_results:
                if tr["success"]:
                    idx = self.market_data_cache.asset_to_index.get(tr["symbol"], None)
                    if idx is not None:
                        touched[idx] = 1.0
            total_touch = np.sum(touched)
            diagnostic_entropy = float(-np.sum((touched / max(total_touch, 1.0)) * np.log(np.clip(touched / max(total_touch, 1.0), 1e-8, 1.0)))) if total_touch > 0 else 0.0
            # For buffer, we don’t have weights vector from action; we will store current weights post-execution below.
            weight_change_target = None  # not used in reward in this path


        # ---------------- FROM HERE ALL MODES SHARE THE SAME PROCESSING ------------
        # ---- Episode metric accumulation ----
        traded_notional = float(execution_result.traded_dollar_value)
        if traded_notional > 0:
            self._ep_turnover_notional += traded_notional
            self._ep_trade_sizes.append(traded_notional)
        # Split buy/sell notional from execution_result.trades_executed
        buys = execution_result.traded_notional_per_asset[execution_result.trades_executed > 0].sum()
        sells = execution_result.traded_notional_per_asset[execution_result.trades_executed < 0].sum()
        self._ep_buy_notional += float(buys)
        self._ep_sell_notional += float(abs(sells))
        # Cost breakdown captured in _calculate_transaction_costs
        if hasattr(self, "_last_cost_breakdown"):
            c_comm, c_spread, c_imp, c_fix = self._last_cost_breakdown
            self._ep_cost_commission += c_comm
            self._ep_cost_spread += c_spread
            self._ep_cost_impact += c_imp
            self._ep_cost_fixed += c_fix
        # Exposure tracking
        invested_fraction = 1.0 - self.portfolio_state.get_weights()[0]
        self._ep_exposure_sum += invested_fraction
        self._ep_exposure_steps += 1
        # Record raw action output (single-asset mode)
        if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            self._ep_action_outputs.append(float(target_position_change))
        # Shadow (frictionless) portfolio: apply same position changes without costs
        if execution_result.success:
            shadow_pos = self.shadow_portfolio_state.positions.copy()
            shadow_pos += execution_result.trades_executed
            self.shadow_portfolio_state.positions = shadow_pos
            self.shadow_portfolio_state.cash -= execution_result.traded_dollar_value  # no costs

        # ADVANCE TIME -----------------------------------------------------
        self.current_step += 1
        self.current_absolute_step = int(self.current_episode_start_step + self.current_step)

        # Calculate reward -----------------------------------------------
        # Get new market prices for the new day (THIS IS THE DAY CHANGE!)
        new_prices = self._get_current_prices(self.current_absolute_step)
        
        # Update portfolio state with new prices (mark-to-market)
        # Portfolio positions remain the same, but prices change
        self.portfolio_state.prices[:] = new_prices # in-place update
        self.portfolio_state.step = self.current_step

        # Update shadow portfolio prices (frictionless)
        self.shadow_portfolio_state.prices[:] = new_prices
        self.shadow_portfolio_state.step = self.current_step

        # Update comparison portfolio state
        self.comparison_portfolio_state.prices[:] = new_prices # in-place update
        self.comparison_portfolio_state.step = self.current_step

        # Update benchmark portfolio state
        self.benchmark_portfolio_state.prices[:] = new_prices # in-place update
        self.benchmark_portfolio_state.step = self.current_step

        # Apply daily cash decay (inflation/opportunity cost)
        # cash_drag_rate is annual; convert to daily: (1 + rate)^(1/252) - 1
        if self.cash_drag_rate_pa > 0:
            daily_decay_factor = (1.0 + self.cash_drag_rate_pa) ** (1.0 / 252.0) - 1.0
            # Reduce cash by decay (daily carrying cost)
            self.portfolio_state.cash *= (1.0 - daily_decay_factor)
            self.comparison_portfolio_state.cash *= (1.0 - daily_decay_factor)
            self.benchmark_portfolio_state.cash *= (1.0 - daily_decay_factor)
            self.shadow_portfolio_state.cash *= (1.0 - daily_decay_factor)
        
        # Calculate new portfolio value after price changes
        portfolio_value_after = self.portfolio_state.get_total_value()
        benchmark_portfolio_value_after = self.benchmark_portfolio_state.get_total_value()
        comparison_portfolio_value_after = self.comparison_portfolio_state.get_total_value()

        # SAA return: change in cash + target asset return 
        "NOTE: only meaningful in single-asset mode"
        selected_asset_notional_after = 0.0
        delta_selected_asset_notional = 0.0
        delta_cash = 0.0
        saa_return = 0.0

        if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            selected_asset_notional_after = (
                self.portfolio_state.positions[selected_asset_index]
                * self.portfolio_state.prices[selected_asset_index]
            )
            delta_selected_asset_notional = selected_asset_notional_after - selected_asset_notional_before
            delta_cash = live_portfolio_cash_before - self.portfolio_state.cash
            saa_return = delta_selected_asset_notional + delta_cash


        # Allocate rewards based on mode. Init all with None to check proper implementation
        allocator_reward:Optional[float] = None
        saa_reward:Optional[float] = None

        reward_parts: Dict[str, float] = {}
        saa_reward_parts: Dict[str, float] = {}

        if self.execution_mode in {EXECUTION_PORTFOLIO_WEIGHTS, EXECUTION_SIMPLE, EXECUTION_TRANCHE}:
            # Calculate Allocator Reward
            allocator_reward, reward_parts = self.calculate_allocator_step_reward(
                execution_result=execution_result,
                portfolio_before=portfolio_value_before,
                portfolio_after=portfolio_value_after,
                comparison_before=comparison_portfolio_value_before,
                comparison_after=comparison_portfolio_value_after,
                benchmark_before=benchmark_portfolio_value_before,
                benchmark_after=benchmark_portfolio_value_after,
                action=(weight_change_target if self.execution_mode == EXECUTION_PORTFOLIO_WEIGHTS else None)
            )
        elif self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            # In single-asset target position mode, isolate saa_return 
            saa_reward, saa_reward_parts = self.calculate_saa_step_reward(
                execution_result=execution_result,
                selected_asset_index=selected_asset_index,
                delta_selected_asset_notional=delta_selected_asset_notional,
                delta_cash=delta_cash,
                saa_return=saa_return,
                selected_asset_notional_before=selected_asset_notional_before,
                selected_asset_notional_after=selected_asset_notional_after,
                saa_cash_before=live_portfolio_cash_before,
                saa_cash_after=self.portfolio_state.cash,
                action=target_position_change
            )
        else:
            raise ValueError(f"Unknown execution mode {self.execution_mode} in reward allocation.")

        # Construct action vector for episode buffer:
            # Single Asset target position mode: store [net_cash_delta, per-asset traded $] computed from execution_result.
            # Portfolio mode: store weights action (as before).
            # Simple/Tranche: store [net_cash_delta, per-asset traded $] computed from execution_result.

        if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            net_asset_dollars = execution_result.traded_notional_per_asset.copy()
            net_asset_dollars[np.isnan(net_asset_dollars)] = 0.0
            net_cash_delta = -float(np.sum(net_asset_dollars))  # costs handled separately in transaction_cost
            action_vec = np.zeros(self.market_data_cache.num_assets + 1, dtype=np.float32)
            action_vec[selected_asset_index + 1] = np.float32(target_position_change)  

        elif self.execution_mode == EXECUTION_PORTFOLIO_WEIGHTS:
            action_vec = weight_change_target.astype(np.float32).copy()

        else:
            net_asset_dollars = execution_result.traded_notional_per_asset.copy()
            net_asset_dollars[np.isnan(net_asset_dollars)] = 0.0
            net_cash_delta = -float(np.sum(net_asset_dollars))  # costs handled separately in transaction_cost
            action_vec = np.concatenate([[net_cash_delta], net_asset_dollars]).astype(np.float32)

        # Record step data ---------------------------------------
        current_weights_after_execution = self.portfolio_state.get_weights()
        daily_return = (portfolio_value_after / portfolio_value_before - 1.0 if portfolio_value_before > 0 else 0.0)
        
        sum_sq = float(np.sum(current_weights_after_execution[1:] ** 2))
        if sum_sq < 1e-8:
            effective_asset_concentration_norm = 0.0
        else:
            effective_asset_concentration_norm = float(1.0 / sum_sq) /  float(len(current_weights_after_execution[1:]))

        # Get all available portfolio values of episode
        window = min(20, self.current_step)  # use up to 20 most recent returns
        returns_window = self.episode_buffer.get_returns_window(window)
        step_volatility = float(np.std(returns_window)) if len(returns_window) > 1 else 0.0

        # FIX: Set external step to current step (already incremented)
        external_step = self.current_step  # zero-based index for buffer, is called first time at reset!

        # Decide on reward to report
        if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            reward_to_record = float(saa_reward) 
        elif self.execution_mode in {EXECUTION_PORTFOLIO_WEIGHTS, EXECUTION_SIMPLE, EXECUTION_TRANCHE}:
            reward_to_record = float(allocator_reward)
        else:
            raise ValueError(f"Unknown execution mode {self.execution_mode} in reward recording.")
        
        # Record step in episode buffer
        self.episode_buffer.record_step(
            external_step=external_step,
            portfolio_value=float(portfolio_value_after),
            portfolio_positions=self.portfolio_state.positions.copy(),
            weights=current_weights_after_execution.copy(),
            daily_return=float(daily_return),
            saa_return=float(saa_return),
            #reward=float(allocator_reward),
            reward_to_record=float(reward_to_record),
            #saa_reward=float(saa_reward),
            action=action_vec.copy(),
            transaction_cost=float(execution_result.transaction_cost),
            prices=new_prices.copy(),
            sharpe_ratio=reward_parts.get("sharpe_ratio", 0.0),
            drawdown=reward_parts.get("max_drawdown_delta", 0.0),
            volatility=step_volatility,
            turnover=reward_parts.get("turnover", 0.0),
            alpha=reward_parts.get("raw_alpha", 0.0),
            traded_dollar_volume=float(execution_result.traded_dollar_value),
            traded_shares_total=float(execution_result.traded_shares_total),
            action_entropy=float(diagnostic_entropy),
            reward_parts={k: float(v) for k, v in reward_parts.items()},
            saa_reward_parts={k: float(v) for k, v in saa_reward_parts.items()},
            benchmark_portfolio_value=float(benchmark_portfolio_value_after),
            comparison_portfolio_value=float(comparison_portfolio_value_after),
            effective_asset_concentration_norm=float(effective_asset_concentration_norm),
            previous_sortino=reward_parts.get("previous_sortino", None),
            current_sortino=reward_parts.get("current_sortino", None),
            running_mean_ema=reward_parts.get("running_mean_ema", None),
            downside_var_sqrt=reward_parts.get("downside_var_sqrt", None),
            previous_max_drawdown=reward_parts.get("previous_max_drawdown", None)
        )

        # Check termination conditions -----------------------------------
        low_value_cond = portfolio_value_after <= self.threshold_val
        zero_value_cond = portfolio_value_after < 0.0
        negative_cash_cond = self.portfolio_state.cash < -10.0  # allow small negative cash for numerical stability

        terminated = bool(low_value_cond or zero_value_cond or negative_cash_cond)

        if terminated:
            reasons = []
            if low_value_cond:
                reasons.append(f"portfolio value {portfolio_value_after:.2f} <= early stopping threshold {self.threshold_val:.2f}")
            elif zero_value_cond:
                reasons.append(f"portfolio value {portfolio_value_after:.2f} <= 0")
            elif negative_cash_cond:
                reasons.append(f"cash balance {self.portfolio_state.cash:.2f} < -10 USD")
            else:
                reasons.append("unknown reason")

            # Print all triggered reasons together
            print(f"Episode {self.current_episode} terminated early due to: " + "; ".join(reasons))
        
        truncated = self.current_step >= self.episode_length_days -1  # step limit reached in days
        self.portfolio_state.terminated = terminated or truncated
        
        # Prepare info
        info = self._get_info()
        if self.execution_mode in {EXECUTION_SIMPLE, EXECUTION_TRANCHE}:
            info['trade_results'] = trade_results
        elif self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            info["trade_results"] = [execution_result]


        # ------------------ Prepare next observation if not terminated ------------------------
        if terminated or truncated:
            next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            # Internal index range for actual episode steps (exclude warmup)
            if self.maybe_provide_sequence:
                internal_start = self.lookback_window
                internal_end = self.lookback_window + external_step  # inclusive
            else:
                internal_start = 0
                internal_end = external_step  # inclusive

            # Cumulative allocator reward over real steps
            if internal_end >= internal_start:
                cumulative_reward = float(
                    np.sum(self.episode_buffer.rewards[internal_start:internal_end + 1])
                )
            else:
                cumulative_reward = 0.0

            # Episode max drawdown (already stored per step)
            if internal_end >= internal_start:
                episode_max_dd = float(
                    np.max(self.episode_buffer.drawdown[internal_start:internal_end + 1])
                )
            else:
                episode_max_dd = 0.0

            final_portfolio_value = portfolio_value_after
            final_benchmark_value = benchmark_portfolio_value_after
            final_comparison_value = comparison_portfolio_value_after
            port_ret = (final_portfolio_value / self.initial_portfolio_value) - 1.0
            comparison_ret = (final_comparison_value / self.initial_portfolio_value) - 1.0
            bench_ret = (final_benchmark_value / self.initial_portfolio_value) - 1.0
            alpha_ret = port_ret - bench_ret

            ep_turnover = self._ep_turnover_notional / max(1e-8, self.initial_portfolio_value)
            avg_exposure = (self._ep_exposure_sum / max(1, self._ep_exposure_steps)) if self._ep_exposure_steps else 0.0
            start_exposure = 1.0 - self.initial_portfolio_value / max(1e-8, self.initial_portfolio_value)  # 0 if start in cash
            end_exposure = 1.0 - self.portfolio_state.get_weights()[0]
            shadow_value = self.shadow_portfolio_state.get_total_value()
            live_value = self.portfolio_state.get_total_value()
            comparison_value = self.comparison_portfolio_state.get_total_value()

            # Safe slices
            if internal_end >= internal_start:
                returns_slice = self.episode_buffer.returns[internal_start:internal_end + 1]
                costs_slice = self.episode_buffer.transaction_costs[internal_start:internal_end + 1]
                weights_slice = self.episode_buffer.portfolio_weights[internal_start:internal_end + 1]  # [T, A+1]
                traded_notional_slice = self.episode_buffer.traded_dollar_volume[internal_start:internal_end + 1]
                traded_shares_slice = self.episode_buffer.traded_shares_total[internal_start:internal_end + 1]

                # Sharpe/Volatility (annualized)
                if returns_slice.size >= 2:
                    mean_r = float(np.mean(returns_slice))
                    std_r = float(np.std(returns_slice, ddof=1))
                    episode_volatility = float(std_r * np.sqrt(252)) if std_r > 0 else 0.0
                    episode_sharpe = float((mean_r / std_r) * np.sqrt(252)) if std_r > 0 else 0.0
                else:
                    episode_sharpe = 0.0
                    episode_volatility = 0.0

                # Total costs and trade-day count
                total_transaction_costs = float(np.sum(costs_slice))
                num_trade_days = int(np.sum(costs_slice > 0))

                # Turnover from weights (exclude cash; per-step |Δw| sum)
                if weights_slice.shape[0] >= 2:
                    w_ex_cash = weights_slice[:, 1:]                # [T, A]
                    step_turnovers = np.sum(np.abs(w_ex_cash[1:] - w_ex_cash[:-1]), axis=1)  # [T-1]
                    avg_turnover = float(np.mean(step_turnovers)) if step_turnovers.size > 0 else 0.0
                    total_turnover = float(np.sum(step_turnovers)) if step_turnovers.size > 0 else 0.0
                    weights_mean = float(np.mean(w_ex_cash))
                    weights_max = float(np.max(w_ex_cash))
                    weights_min = float(np.min(w_ex_cash))
                    weights_median = float(np.median(w_ex_cash))
                else:
                    avg_turnover = 0.0
                    total_turnover = 0.0
                    weights_mean = 0.0
                    weights_max = 0.0
                    weights_min = 0.0
                    weights_median = 0.0
                # episode traded_volume
                episode_traded_notional = float(np.sum(traded_notional_slice))
                episode_traded_shares = float(np.sum(traded_shares_slice))

            else:
                episode_sharpe = 0.0
                episode_volatility = 0.0
                total_transaction_costs = 0.0
                num_trade_days = 0
                avg_turnover = 0.0
                total_turnover = 0.0
                weights_mean = weights_max = weights_min = weights_median = 0.0
                episode_traded_notional = 0.0
                episode_traded_shares = 0.0

            # ===== Action & Allocation Diagnostics (Episode Level) =====
            # Slice real steps only
            if internal_end >= internal_start:
                weights_slice_full = self.episode_buffer.portfolio_weights[internal_start:internal_end + 1]  # [T, A+1]
                actions_entropy_slice = self.episode_buffer.action_entropy[internal_start:internal_end + 1]
                traded_notional_slice = self.episode_buffer.traded_dollar_volume[internal_start:internal_end + 1]  # [T, A]
                # Exclude cash for per-asset stats
                w_assets = weights_slice_full[:, 1:]  # [T, A]
                # Mean / std weights
                mean_w_per_asset = np.mean(w_assets, axis=0)
                std_w_per_asset  = np.std(w_assets, axis=0)
                time_gt_10 = np.mean(w_assets > 0.10, axis=0)
                # Turnover contributions
                if w_assets.shape[0] >= 2:
                    step_deltas = np.abs(w_assets[1:] - w_assets[:-1])  # [T-1, A]
                    total_abs_change = np.sum(step_deltas)
                    if total_abs_change > 0:
                        turnover_contrib = np.sum(step_deltas, axis=0) / total_abs_change
                    else:
                        turnover_contrib = np.zeros(w_assets.shape[1], dtype=np.float32)
                else:
                    turnover_contrib = np.zeros(w_assets.shape[1], dtype=np.float32)
                # Traded notional share
                total_traded_notional = np.sum(traded_notional_slice)
                if total_traded_notional > 0:
                    notional_share = np.sum(traded_notional_slice, axis=0) / total_traded_notional
                else:
                    notional_share = np.zeros(w_assets.shape[1], dtype=np.float32)

                avg_entropy = float(np.mean(actions_entropy_slice))
                avg_traded_notional_step = float(np.mean(np.sum(traded_notional_slice, axis=1))) if traded_notional_slice.size else 0.0
                total_costs = float(np.sum(self.episode_buffer.transaction_costs[internal_start:internal_end + 1]))
                cost_per_dollar = float(total_costs / total_traded_notional) if total_traded_notional > 0 else 0.0

                # Pack per-asset metrics (symbol aligned)
                per_asset_metrics = {}
                asset_names_local = self.market_data_cache.asset_names
                for i, sym in enumerate(asset_names_local):
                    per_asset_metrics[f"weights/avg_{sym}"] = float(mean_w_per_asset[i])
                    per_asset_metrics[f"weights/std_{sym}"] = float(std_w_per_asset[i])
                    per_asset_metrics[f"weights/time_gt_10pct_{sym}"] = float(time_gt_10[i])
                    per_asset_metrics[f"turnover/contrib_{sym}"] = float(turnover_contrib[i])
                    per_asset_metrics[f"trades/notional_share_{sym}"] = float(notional_share[i])

                # Reward component sums
                comp_slice = slice(internal_start, internal_end + 1)
                comp_alpha_sum = float(np.sum(self.episode_buffer.reward_alpha[comp_slice]))
                comp_risk_sum = float(np.sum(self.episode_buffer.reward_risk[comp_slice]))
                comp_port_ret_sum = float(np.sum(self.episode_buffer.reward_portfolio_return[comp_slice]))
                comp_cost_sum = float(np.sum(self.episode_buffer.reward_cost[comp_slice]))
                comp_turn_sum = float(np.sum(self.episode_buffer.reward_turnover[comp_slice]))
                comp_conc_sum = float(np.sum(self.episode_buffer.reward_concentration[comp_slice]))
                comp_surv_sum = float(np.sum(self.episode_buffer.reward_survival[comp_slice]))

            else:
                avg_entropy = 0.0
                avg_traded_notional_step = 0.0
                cost_per_dollar = 0.0
                per_asset_metrics = {}
                benchmark_sharpe = 0.0
                sharpe_diff = 0.0
                comp_alpha_sum = comp_risk_sum = comp_port_ret_sum = comp_cost_sum = comp_turn_sum = comp_conc_sum = comp_surv_sum = 0.0

            # Single-asset mode: compute final SAA return metrics
            if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
                final_saa_cash = self.portfolio_state.cash
                final_saa_asset_value = (self.portfolio_state.positions[selected_asset_index] * self.portfolio_state.prices[selected_asset_index])
                final_saa_subportfolio_value = final_saa_cash + final_saa_asset_value
                saa_return_final = (final_saa_subportfolio_value - self.saa_initial_subportfolio_value)
                saa_return_final_pct = saa_return_final / self.saa_initial_subportfolio_value
                
                # Add to info
                info.update({
                    "saa_final_cash": final_saa_cash,
                    "saa_final_asset_value": final_saa_asset_value,
                    "saa_final_subportfolio_value": final_saa_subportfolio_value,
                    "saa_return_final": saa_return_final,
                    "saa_return_final_pct": saa_return_final_pct,
                })


            # Populate info dictionary with episode metrics
            info.update({
                "episode_final": True,
                "episode_id": self.current_episode,
                "episode_length": external_step + 1,
                "portfolio_final_value": final_portfolio_value,
                "benchmark_final_value": final_benchmark_value,
                "comparison_final_value": final_comparison_value,
                "portfolio_return": port_ret,
                "benchmark_return": bench_ret,
                "comparison_return": comparison_ret,
                "alpha_return": alpha_ret,
                "cumulative_reward": cumulative_reward,
                "episode_max_drawdown": episode_max_dd,
                "episode_sharpe": episode_sharpe,
                "episode_volatility": episode_volatility,
                # Gross vs net
                "shadow_final_value": float(shadow_value),
                "shadow_return": float(shadow_value / self.initial_portfolio_value - 1.0),
                # Costs
                "total_transaction_costs": total_transaction_costs,
                "episode_cost_commission": float(self._ep_cost_commission),
                "episode_cost_spread": float(self._ep_cost_spread),
                "episode_cost_impact": float(self._ep_cost_impact),
                "episode_cost_fixed": float(self._ep_cost_fixed),
                "num_trade_days": num_trade_days,
                "avg_turnover": avg_turnover,
                "total_turnover": total_turnover,
                "episode_traded_notional": episode_traded_notional,
                "episode_traded_shares": episode_traded_shares,
                # Exposure
                "exposure_start": float(start_exposure),
                "exposure_avg": float(avg_exposure),
                "exposure_end": float(end_exposure),
                "weights_mean": weights_mean,
                "weights_max": weights_max,
                "weights_min": weights_min,
                "weights_median": weights_median,
                "actions/entropy_avg": avg_entropy,
                "actions/raw_logits_mean": raw_logits_mean,
                "actions/raw_logits_std": raw_logits_std,
                "actions/raw_logits_max": raw_logits_max,
                "actions/raw_l2_norm": raw_logits_l2,
                "actions/softmax_temp": softmax_temp,
                "trades/avg_notional": avg_traded_notional_step,
                "trades/cost_per_$": cost_per_dollar,
                **per_asset_metrics,
                "reward_components/alpha_sum": comp_alpha_sum,
                "reward_components/risk_sum": comp_risk_sum,
                "reward_components/portfolio_return_sum": comp_port_ret_sum,
                "reward_components/cost_sum": comp_cost_sum,
                "reward_components/turnover_sum": comp_turn_sum,
                "reward_components/concentration_sum": comp_conc_sum,
                "reward_components/survival_sum": comp_surv_sum,
                # Buy/Sell totals
                "total_buy_notional": float(self._ep_buy_notional),
                "total_sell_notional": float(self._ep_sell_notional),
                # Trade sizes
                "trade_size_mean": float(np.mean(self._ep_trade_sizes)) if self._ep_trade_sizes else 0.0,
                "trade_size_median": float(np.median(self._ep_trade_sizes)) if self._ep_trade_sizes else 0.0,
                # Action magnitude stats (single-asset mode)
                "action_mean": float(np.mean(self._ep_action_outputs)) if self._ep_action_outputs else 0.0,
                "action_median": float(np.median(self._ep_action_outputs)) if self._ep_action_outputs else 0.0,
                "action_p05": float(np.percentile(self._ep_action_outputs, 5)) if self._ep_action_outputs else 0.0,
                "action_p25": float(np.percentile(self._ep_action_outputs, 25)) if self._ep_action_outputs else 0.0,
                "action_p75": float(np.percentile(self._ep_action_outputs, 75)) if self._ep_action_outputs else 0.0,
                "action_p95": float(np.percentile(self._ep_action_outputs, 95)) if self._ep_action_outputs else 0.0,
                # Sortino internals (episode aggregates)
                "sortino_mean_ema": float(np.mean(self._sortino_mean_hist)) if (hasattr(self, "_sortino_mean_hist") and len(self._sortino_mean_hist) > 0) else 0.0,
                "sortino_downside_ema": float(np.mean(self._sortino_down_hist)) if (hasattr(self, "_sortino_down_hist") and len(self._sortino_down_hist) > 0) else 0.0,
                "sortino_reward_raw_mean": float(np.mean(self._sortino_raw_hist)) if (hasattr(self, "_sortino_raw_hist") and len(self._sortino_raw_hist) > 0) else 0.0,
                "sortino_reward_raw_p25": float(np.percentile(self._sortino_raw_hist, 25)) if (hasattr(self, "_sortino_raw_hist") and len(self._sortino_raw_hist) > 0) else 0.0,
                "sortino_reward_raw_p75": float(np.percentile(self._sortino_raw_hist, 75)) if (hasattr(self, "_sortino_raw_hist") and len(self._sortino_raw_hist) > 0) else 0.0,
 
            })

        else:
            if self.maybe_provide_sequence:
                next_observation = self.get_observation_sequence()
            else:
                next_observation = self.get_observation_single_step()
            # Validation layer
            if not np.all(np.isfinite(next_observation)):
                print(f"NaNs: {np.isnan(next_observation).sum()}, Infs: {np.isinf(next_observation).sum()}")
                raise ValueError("Next observation contains non-finite values (NaN or Inf)")

        # # Debug prints for tracing step execution, COMMENT OUT WHEN PRODUCTION PHASE
        # print(f"Step {self.current_step}: Reward: {allocator_reward:.2f}, "
        # f"Transaction cost: {execution_result.transaction_cost:.2f}, Trades executed: {execution_result.success}\n"
        # f"Executed trades: {execution_result.trades_executed}\n"
        # f"Actual weights after execution: {self.portfolio_state.get_weights()}\n"
        # f"Prices: {self.portfolio_state.prices}\n"
        # f"Positions: {self.portfolio_state.positions}\n"
        # f"Cash: {self.portfolio_state.cash:.2f}, "
        # f"Total Portfolio Value: {self.portfolio_state.get_total_value():.2f}\n")

        # Use correct reward to report based on the mode!
        if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            reward_to_return = saa_reward
        else:
            reward_to_return = allocator_reward
        if reward_to_return is None:
            raise ValueError("[Step fct]Reward to return is None, check reward calculation logic for execution mode.")
        # Return all step outputs (observation, reward, terminated, truncated, info)
        return next_observation, reward_to_return, terminated, truncated, info
    

    def calculate_allocator_step_reward(
            self, execution_result: ExecutionResult, portfolio_before,
            portfolio_after, comparison_before, comparison_after, benchmark_before, benchmark_after, action: np.ndarray
            ) -> Tuple[float, Dict[str, float]]:
        """

        Calculate the allocator reward for this step. Its a combined metric of:
        - alpha over benchmark 
        - risk_features: sharpe-ratio, drawdown, volatility
        - penalty for inactivity / overactivity
        
        
        Args:
            execution_result: Result of the trade execution step
            old_portfolio_value: Portfolio value before time advancement
            new_portfolio_value: Portfolio value after time advancement
            action: Target portfolio weights [num_assets + 1] including cash
        Returns:
            Scalar reward value for this step
            Reward_parts: Dict of individual reward components for diagnostics
        """

        # Calculate basic returns
        if portfolio_before > 0 and portfolio_after > 0:
            portfolio_return = (portfolio_after / portfolio_before)
        else:
            portfolio_return = 0.0
        if benchmark_before > 0 and benchmark_after > 0:
            benchmark_return = (benchmark_after / benchmark_before)
        else:
            benchmark_return = 0.0
        
        # ---------- Differential Sortino ratio components ----------
        # Get simple returns
        portfolio_return_absolut = portfolio_return - 1.0
        # portfolio_return_log = np.log(portfolio_return) if portfolio_return > 0 else -np.inf
        # 1. Update EMAs
        delta = portfolio_return_absolut - self.running_mean_ema
        self.running_mean_ema += self.sortino_eta * delta

        # Alternative: use squared downside returns
        downside_sq = (min(portfolio_return_absolut, 0.0))**2
        self.running_downside_variance_ema += self.sortino_eta * (downside_sq - self.running_downside_variance_ema)
        downside_var_floor = 1e-6
        downside_var = max(self.running_downside_variance_ema, downside_var_floor)
        current_sortino = self.running_mean_ema / np.sqrt(downside_var)

        # 3. Calc Differential Sortino for reward
        sortino_reward = current_sortino - self.previous_sortino
        self.previous_sortino = current_sortino

        # Mix Sortino with other metrics to get risk-aware non-deterministic policy
        """Calculate dynamic risk window. Use self.reward_risk_window and the current step to produce
        behaviour which starts at 2 raises with the steps up to maximum risk_reward_window"""
        if self.current_step <= 2:
            risk_metric_window = 2
        else:
            risk_metric_window = min(self.current_step, self.max_reward_risk_window)

        max_drawdown = self.episode_buffer.calculate_max_drawdown(
            window=risk_metric_window
        )

        max_drawdown_delta = max(0.0, max_drawdown - self.previous_max_drawdown)
        self.previous_max_drawdown = max_drawdown
        max_drawdown_penalty = float(self.lambda_drawdown * max_drawdown_delta)

        # 4. Mix in raw return & windowed drawdown level penalty
        # first_reward = (self.sortino_net_reward_mix * sortino_reward + 
        #           (1 - self.sortino_net_reward_mix) * portfolio_return_absolut) - max_drawdown_penalty
        # reward = first_reward

        """ New reward try """
        # TODO: Consider reintroducing mix with diff sortino but based on the comparison log diff returns!
        # Core objective: outperform static comparison portfolio from same random init
        eps = 1e-12
        port_log_ret = np.log(max(float(portfolio_after), eps)) - np.log(max(float(portfolio_before), eps))
        comp_log_ret = np.log(max(float(comparison_after), eps)) - np.log(max(float(comparison_before), eps))
        excess_log_ret = port_log_ret - comp_log_ret

        # Risk control: penalize worsening drawdown (incremental only)
        reward = float(excess_log_ret - self.lambda_drawdown * max_drawdown_delta)


        # # 4b. Concentration term on executed normalized weights.
        # # action is expected to be the already-normalized execution target w.
        # concentration_pen = 0.0
        # if action is not None:
        #     w_exec = np.asarray(action, dtype=np.float32)
        #     if w_exec.ndim == 1 and w_exec.size == (self.market_data_cache.num_assets + 1):
        #         w_exec = np.clip(w_exec, 0.0, 1.0)
        #         w_exec = w_exec / np.maximum(np.sum(w_exec), 1e-8)
        #         concentration_pen = float(
        #             -self.lambda_spread * np.sum(w_exec * np.log(w_exec + 1e-8))
        #         )
        #         second_reward = concentration_pen + first_reward
        #         reward = second_reward

        # # 4c. Transaction cost penalty (scaled by lambda_cost)
        # transaction_cost_level = 0.0001*self.initial_portfolio_value
        # if execution_result.transaction_cost > transaction_cost_level:
        #     self.trans_act_pen = self.lambda_transaction_cost * (execution_result.transaction_cost - transaction_cost_level)
        #     third_reward = reward - self.trans_act_pen
        #     reward = third_reward
        # else: 
        #     self.trans_act_pen = 0.0

        # # 5. Clip and amplify reward
        # gain = float(3.5)
        # reward = float(np.tanh(gain * reward))

        # Track Sortino internals for episode-level stats
        self._sortino_mean_hist.append(float(self.running_mean_ema))
        self._sortino_down_hist.append(float(self.running_downside_variance_ema))
        self._sortino_raw_hist.append(float(reward))

        # START OF OLD REWARD CALCULATION LOGIC: Metrics are needed to fill episode buffer portfolio metrics
        # ============================================
        # 1. ALPHA CALCULATION (PRIMARY SIGNAL)
        # ============================================

        alpha = (portfolio_return - benchmark_return)  # daily alpha
        portfolio_return = portfolio_return - 1.0  # daily portfolio return
        
        # ============================================
        # 2. RISK-ADJUSTED PERFORMANCE METRICS
        # ============================================
        risk_adjustment = 0.0

        # Use episode buffer's efficient methods
        sharpe_ratio = self.episode_buffer.calculate_sharpe_ratio(
            window=risk_metric_window
        )
        
        # Get recent returns for volatility
        recent_returns = self.episode_buffer.get_returns_window(
            window=risk_metric_window
        )
        volatility = np.std(recent_returns) * np.sqrt(252) if len(recent_returns) > 1 else 0.0  # Annualized
        
        # Risk adjustment components
        sharpe_bonus = np.tanh(sharpe_ratio/ 1.5) * 0.01        # Bounded bonus for good Sharpe
        drawdown_penalty = max_drawdown * 0.025                  # Penalty for large drawdowns
        volatility_penalty = max(0, volatility - 0.25) * 0.025   # Penalty only if vol > 25%
        
        risk_adjustment = sharpe_bonus - drawdown_penalty - volatility_penalty
        
        # ============================================
        # 3. TRANSACTION COST PENALTY
        # ============================================
        cost_penalty = 0.0
        if execution_result.transaction_cost > 0:
            raw_cost_ratio = execution_result.transaction_cost / portfolio_before if portfolio_before > 0 else 0.0
            cost_penalty = 2.0 * min(raw_cost_ratio, 0.05) # Cap penalty at 5% cost ratio
        
        # ============================================
        # 4. TURNOVER 
        # ============================================
        current_weights = self.portfolio_state.get_weights()
        if self.current_step > 0:
            if self.maybe_provide_sequence:
                prev_w = self.episode_buffer.portfolio_weights[self.episode_buffer.lookback_window + self.current_step - 1]
            else:
                prev_w = self.episode_buffer.portfolio_weights[self.current_step - 1]
            # Classic turnover: sum of absolute changes in asset weights (excluding cash)
            turnover = float(np.sum(np.abs(current_weights[1:] - prev_w[1:]))) / 2.0
        else:
            turnover = 0.0
        # Normalize to [0, 1]: cap at 1.0 (max possible turnover is 2.0, div by 2 gives max 1.0)
        turnover = float(np.clip(turnover, 0.0, 1.0))

        # ============================================
        # 7. COMBINE ALL COMPONENTS WITH PROPER WEIGHTS
        # ============================================
        # factors to scale components
        alpha_factor = 0.0
        portfolio_return_factor = 2.0
        risk_adjustment_factor = 1.0
        cost_penalty_factor = 0.0

        # Old disfunctional reward combination (too noisy, hard to tune, not effective)
        # reward = (
        #     alpha * alpha_factor +                    # Primary objective: alpha generation (scaled up)
        #     portfolio_return * portfolio_return_factor +          # Also reward general portfolio return
        #     risk_adjustment * risk_adjustment_factor +           # Risk-adjusted performance bonus
        #     -cost_penalty * cost_penalty_factor +             # Transaction cost efficiency
        #     -turnover_penalty * turnover_penalty_factor +         # Turnover management
        #     -concentration_penalty * concentration_penalty_factor +      # Position risk management
        #     + survival_bonus                  # Small survival bonus
        # )

        # # If you want to skip detailed sub-component calculations, create minimal parts dict
        # parts = {
        #     "alpha_component": 0.0,
        #     "risk_component": 0.0,
        #     "portfolio_return_component": 0.0,
        #     "cost_component": 0.0,
        #     "turnover_component": 0.0,
        #     "concentration_component": concentration_penalty,
        #     "survival_component": 0.0,
        #     "raw_alpha": 0.0,
        #     "raw_portfolio_return": 0.0,
        #     "raw_benchmark_return": 0.0,
        #     "sharpe_ratio": 0.0,
        #     "max_drawdown": 0.0
        # }

        parts = {
            "alpha_component": alpha * alpha_factor,
            "risk_component": risk_adjustment * risk_adjustment_factor,
            "portfolio_return_component": portfolio_return * portfolio_return_factor,
            "cost_component": -cost_penalty * cost_penalty_factor,
            "turnover": turnover,
            "raw_alpha": alpha,
            "raw_portfolio_return": portfolio_return,
            "raw_benchmark_return": benchmark_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_delta": max_drawdown_delta,
            "previous_sortino": self.previous_sortino,
            "current_sortino": current_sortino,
            "running_mean_ema": self.running_mean_ema,
            "downside_var_sqrt": np.sqrt(downside_var),
            "previous_max_drawdown": self.previous_max_drawdown

        }

        return float(reward), parts

    def calculate_saa_step_reward(
            self, execution_result, selected_asset_index, delta_selected_asset_notional,
            delta_cash, saa_return, 
            selected_asset_notional_before, selected_asset_notional_after,
            saa_cash_before, saa_cash_after, action
            ):
        
        # # Calculate the SAA step reward
        # # ---------- Differential Sortino ratio components ----------
        # # Get simple returns
        # saa_portfolio_return_absolut = (selected_asset_notional_after + saa_cash_after) / (selected_asset_notional_before + saa_cash_before) - 1.0
        # # portfolio_return_log = np.log(portfolio_return) if portfolio_return > 0 else -np.inf
        # # 1. Update EMAs
        # saa_delta = saa_portfolio_return_absolut - self.saa_running_mean_ema
        # self.saa_running_mean_ema += self.sortino_eta * saa_delta

        # # Alternative: use squared downside returns
        # saa_downside_sq = (min(saa_portfolio_return_absolut, 0.0))**2
        # self.saa_running_downside_variance_ema += self.sortino_eta * (saa_downside_sq - self.saa_running_downside_variance_ema)
        # downside_var_floor = 1e-6
        # saa_downside_var = max(self.saa_running_downside_variance_ema, downside_var_floor)
        # current_sortino = self.saa_running_mean_ema / np.sqrt(saa_downside_var)

        # # 3. Calc Differential Sortino for reward
        # saa_sortino_reward = current_sortino - self.saa_previous_sortino
        # self.saa_previous_sortino = current_sortino

        # # Mix Sortino with other metrics to get risk-aware non-deterministic policy
        # """Calculate dynamic risk window. Use self.reward_risk_window and the current step to produce
        # behaviour which starts at 2 raises with the steps up to maximum risk_reward_window"""
        # if self.current_step <= 2:
        #     risk_metric_window = 2
        # else:
        #     risk_metric_window = min(self.current_step, self.max_reward_risk_window)

        # saa_max_drawdown = self.episode_buffer.saa_calculate_max_drawdown(
        #     selected_asset_idx=selected_asset_index,
        #     window=risk_metric_window
        # )

        # saa_max_drawdown_delta = max(0.0, saa_max_drawdown - self.saa_previous_max_drawdown)
        # self.saa_previous_max_drawdown = saa_max_drawdown
        # max_drawdown_penalty = float(self.lambda_drawdown * saa_max_drawdown_delta)

        # # 4. Mix in raw return & windowed drawdown penalty
        # reward = (self.sortino_net_reward_mix * saa_sortino_reward + 
        #         (1 - self.sortino_net_reward_mix) * saa_portfolio_return_absolut) - max_drawdown_penalty

        # # 5. Clip and amplify reward
        # gain = float(3.5)
        # saa_reward = float(np.tanh(gain * reward))

        eps = 1e-12
        prev_value = float(selected_asset_notional_before + saa_cash_before)
        next_value = float(selected_asset_notional_after + saa_cash_after)

        # Simple log return reward
        saa_reward_raw = 50 * float(
            np.log(max(next_value, eps))
            - np.log(max(prev_value, eps))
        )

        """Calculate dynamic risk window. Use self.reward_risk_window and the current step to produce
        behaviour which starts at 2 raises with the steps up to maximum risk_reward_window"""
        if self.current_step <= 2:
            risk_metric_window = 2
        else:
            risk_metric_window = min(self.current_step, self.max_reward_risk_window)

        sharpe_ratio = self.episode_buffer.calculate_sharpe_ratio(
            window=risk_metric_window
        )

        saa_reward_raw = saa_reward_raw - ((action * (1 / self.action_limiting_factor_start))**2 * self.action_l2_penalty_coeff) + (sharpe_ratio * 0.1)

        saa_reward = np.tanh(saa_reward_raw / 2.0) * 2.0 # Scale to [-2, 2] range
        
        # NOTE: Several values here get used to fill portfolio wide metrics in the episode buffer.
        saa_reward_parts = {
            "alpha_component": 0.0,
            "risk_component": 0.0,
            "portfolio_return_component": 0.0,
            "cost_component": 0.0,
            "turnover_component": 0.0,
            "concentration_component": 0.0,
            "survival_component": 0.0,
            "raw_alpha": 0.0,
            "raw_portfolio_return": 0.0,
            "raw_benchmark_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0
        }
        
        return saa_reward, saa_reward_parts
            

    def execute_single_asset_target_position(self, asset_index: int, target_position_change: float, portfolio_state: PortfolioState) -> ExecutionResult:
        # Execution of single-asset target position change. Only trades in the provided asset!
        num_assets = self.market_data_cache.num_assets
        current_prices = portfolio_state.prices.copy()

        # Current position in shares
        current_position_shares = float(portfolio_state.positions[asset_index])

        # Guard: valid price
        px = float(current_prices[asset_index]) if asset_index < num_assets else ValueError("Asset index out of range")
        if not np.isfinite(px) or np.isnan(px) or px <= 0.0:
            return ExecutionResult(
                current_step=self.current_step,
                trades_executed=np.zeros(num_assets, dtype=np.float32),
                executed_prices=current_prices,
                transaction_cost=0.0,
                success=False,
                traded_dollar_value=0.0,
                traded_shares_total=0.0,
                traded_notional_per_asset=np.zeros(num_assets, dtype=np.float32)
            )

        # Convert desired change to dollar notional based on current total portfolio value
        total_value = float(portfolio_state.get_total_value())
        desired_notional_change = target_position_change * total_value  # +buy dollars, -sell dollars

        # Shares to trade from desired notional
        shares_to_trade = float(desired_notional_change / px) if px > 0 else 0.0

        # Deadband on tiny share trades
        share_eps = float(self.config['environment'].get('execution_min_share_threshold', 1))
        if abs(shares_to_trade) < share_eps:
            return ExecutionResult(
                current_step=self.current_step,
                trades_executed=np.zeros(num_assets, dtype=np.float32),
                executed_prices=current_prices,
                transaction_cost=0.0,
                success=False,
                traded_dollar_value=0.0,
                traded_shares_total=0.0,
                traded_notional_per_asset=np.zeros(num_assets, dtype=np.float32)
            )

        # Min trade value and min days gates
        min_trade_value_thresh = float(self.config['environment'].get('execution_min_trade_value_threshold', 0.0))
        min_days_between = int(self.config['environment'].get('execution_min_days_between_trades', 0))
        steps_since_last = (self.current_step - self.last_execution_step) if self.last_execution_step >= 0 else np.inf
        if abs(desired_notional_change) < min_trade_value_thresh or steps_since_last < min_days_between:
            return ExecutionResult(
                current_step=self.current_step,
                trades_executed=np.zeros(num_assets, dtype=np.float32),
                executed_prices=current_prices,
                transaction_cost=0.0,
                success=False,
                traded_dollar_value=0.0,
                traded_shares_total=0.0,
                traded_notional_per_asset=np.zeros(num_assets, dtype=np.float32)
            )

        # WHEN PASSED ALL CHECKS: here actual trading starts
        # Prepare execution tracking variables
        trades_executed = np.zeros(num_assets, dtype=np.float32)
        traded_notional_per_asset = np.zeros(num_assets, dtype=np.float32)
        total_transaction_costs = 0.0

        # Build a shares vector for tc calc. Contains all 0s
        shares_vec = np.zeros(num_assets, dtype=np.float32)

        # Since only a single asset is traded there needs to be no selling before buying
        if shares_to_trade > 0:
            # BUY leg
            # First estimate tc with requested shares
            shares_vec[asset_index] = shares_to_trade
            tc_est = self._calculate_transaction_costs(
                shares_traded=shares_vec,
                prices=current_prices,
                abs_step=self.current_absolute_step,
                asset_mask=None
            )
            notional_buy = shares_to_trade * px
            cash_needed = float(notional_buy + tc_est)

            if cash_needed > portfolio_state.cash:
                # Scale down by available cash
                available_cash = max(0.0, float(portfolio_state.cash))
                # subtract a tc estimate; scale using initial tc_est to avoid iterative loops
                denom = (px + (tc_est / max(shares_to_trade, 1e-8)))
                scaled_shares = float(np.floor(max(0.0, available_cash / max(denom, 1e-8))))
                if scaled_shares <= 0:
                    # Cannot buy any shares
                    return ExecutionResult(
                        current_step=self.current_step,
                        trades_executed=np.zeros(num_assets, dtype=np.float32),
                        executed_prices=current_prices,
                        transaction_cost=0.0,
                        success=False,
                        traded_dollar_value=0.0,
                        traded_shares_total=0.0,
                        traded_notional_per_asset=np.zeros(num_assets, dtype=np.float32)
                    )
                shares_to_trade = scaled_shares
                shares_vec[asset_index] = shares_to_trade
                # Recompute tc on scaled shares
                tc_est = self._calculate_transaction_costs(
                    shares_traded=shares_vec,
                    prices=current_prices,
                    abs_step=self.current_absolute_step,
                    asset_mask=None
                )
                notional_buy = shares_to_trade * px
                cash_needed = float(notional_buy + tc_est)

            # Apply buy
            portfolio_state.positions[asset_index] += shares_to_trade
            portfolio_state.cash -= cash_needed
            trades_executed[asset_index] += shares_to_trade
            traded_notional_per_asset[asset_index] += notional_buy
            total_transaction_costs += tc_est

        else:
            # SELL leg
            sell_shares = -shares_to_trade  # positive number
            if not self.allow_short:
                sell_shares = min(sell_shares, max(0.0, current_position_shares))

            if sell_shares <= 0:
                # No shares available to sell (or short not allowed)
                return ExecutionResult(
                    current_step=self.current_step,
                    trades_executed=np.zeros(num_assets, dtype=np.float32),
                    executed_prices=current_prices,
                    transaction_cost=0.0,
                    success=False,
                    traded_dollar_value=0.0,
                    traded_shares_total=0.0,
                    traded_notional_per_asset=np.zeros(num_assets, dtype=np.float32)
                )

            shares_vec[asset_index] = -sell_shares
            tc_sell = self._calculate_transaction_costs(
                shares_traded=shares_vec,
                prices=current_prices,
                abs_step=self.current_absolute_step,
                asset_mask=None
            )
            notional_sell = sell_shares * px

            # Apply sell
            portfolio_state.positions[asset_index] -= sell_shares
            if not self.allow_short:
                portfolio_state.positions[asset_index] = max(0.0, float(portfolio_state.positions[asset_index]))
            portfolio_state.cash += float(notional_sell - tc_sell)
            trades_executed[asset_index] -= sell_shares
            traded_notional_per_asset[asset_index] += notional_sell
            total_transaction_costs += tc_sell

        total_traded_notional = float(np.sum(traded_notional_per_asset))
        traded_shares_total = float(np.sum(np.abs(trades_executed)))

        if total_traded_notional > 0.0:
            self.last_execution_step = self.current_step

        return ExecutionResult(
            current_step=self.current_step,
            trades_executed=trades_executed.astype(np.float32),
            executed_prices=current_prices,
            transaction_cost=float(total_transaction_costs),
            success=bool(total_traded_notional > 0.0),
            traded_dollar_value=total_traded_notional,
            traded_shares_total=traded_shares_total,
            traded_notional_per_asset=traded_notional_per_asset.astype(np.float32)
        )


    def execute_portfolio_change(self, target_weights: np.ndarray, portfolio_state: PortfolioState) -> ExecutionResult:
        num_assets = self.market_data_cache.num_assets
        current_prices = portfolio_state.prices.copy()

        # Threshold gating disabled to check for instability reasons
        # should_execute, adjusted_target_weights = self._check_execution_thresholds(target_weights, portfolio_state)
        # if not should_execute:
        #     return ExecutionResult(
        #         current_step=self.current_step,
        #         trades_executed=np.zeros(num_assets, dtype=np.float32),
        #         executed_prices=current_prices,
        #         transaction_cost=0.0,
        #         success=False,
        #         traded_dollar_value=0.0,
        #         traded_shares_total=0.0,
        #         traded_notional_per_asset=np.zeros(num_assets, dtype=np.float32)
        #     )

        current_weights = portfolio_state.get_weights()
        total_value = portfolio_state.get_total_value()

        # --- Intro of smooth exec ---
        current_weights = portfolio_state.get_weights()
        adjusted_target_weights = self._apply_soft_execution(
            target_weights=target_weights,
            current_weights=current_weights,
            portfolio_state=portfolio_state
        )


        desired_allocations = adjusted_target_weights * total_value
        current_allocations = current_weights * total_value
        trade_amounts = desired_allocations - current_allocations  # includes cash at index 0

        asset_trade_amounts = trade_amounts[1:]  # exclude cash
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_shares_to_trade = np.where(current_prices > 0,
                                           asset_trade_amounts / current_prices,
                                           0.0).astype(np.float32)

        share_eps = float(self.config['environment'].get('execution_min_share_threshold', 1e-5))
        # Full-length shares vector (assets only), signed: +buy, -sell
        shares_to_trade_full = raw_shares_to_trade.copy()

        # Deadband removal
        shares_to_trade_full[np.abs(shares_to_trade_full) < share_eps] = 0.0
        if np.all(shares_to_trade_full == 0.0):
            return ExecutionResult(
                current_step=self.current_step,
                trades_executed=np.zeros(num_assets, dtype=np.float32),
                executed_prices=current_prices,
                transaction_cost=0.0,
                success=False,
                traded_dollar_value=0.0,
                traded_shares_total=0.0,
                traded_notional_per_asset=np.zeros(num_assets, dtype=np.float32)
            )

        position_changes = np.zeros(num_assets, dtype=np.float32)
        trades_executed = np.zeros(num_assets, dtype=np.float32)
        traded_notional_per_asset = np.zeros(num_assets, dtype=np.float32)
        total_transaction_costs = 0.0

        # SELL leg
        sell_mask = shares_to_trade_full < -share_eps
        if np.any(sell_mask):
            shares_to_sell = -shares_to_trade_full[sell_mask]
            available_shares = portfolio_state.positions[sell_mask]
            actual_shares_sold = np.minimum(shares_to_sell, available_shares)
            # Adjust for position limits
            shares_deficit = shares_to_sell - actual_shares_sold
            if np.any(shares_deficit > 0):
                # Clamp attempted oversell to available
                pass
            if np.any(actual_shares_sold > 0):
                position_changes[sell_mask] -= actual_shares_sold
                trades_executed[sell_mask] -= actual_shares_sold
                sell_notional = actual_shares_sold * current_prices[sell_mask]
                traded_notional_per_asset[sell_mask] += sell_notional
                tc_sell = self._calculate_transaction_costs(
                    shares_traded=self._expand_shares(actual_shares_sold, sell_mask, num_assets),
                    prices=current_prices,
                    abs_step=self.current_absolute_step,
                    asset_mask=sell_mask
                )
                total_transaction_costs += tc_sell
                portfolio_state.cash += float(np.sum(sell_notional) - tc_sell)

        # BUY leg
        buy_mask = shares_to_trade_full > share_eps
        if np.any(buy_mask):
            shares_to_buy = shares_to_trade_full[buy_mask]
            buy_notional = shares_to_buy * current_prices[buy_mask]
            tc_buy = self._calculate_transaction_costs(
                shares_traded=self._expand_shares(shares_to_buy, buy_mask, num_assets),
                prices=current_prices,
                abs_step=self.current_absolute_step,
                asset_mask=buy_mask
            )
            cost_of_buys = float(np.sum(buy_notional) + tc_buy)
            if cost_of_buys > portfolio_state.cash:
                scaling_factor = portfolio_state.cash / max(cost_of_buys, 1e-8)
                shares_to_buy *= scaling_factor
                buy_notional = shares_to_buy * current_prices[buy_mask]
                tc_buy = self._calculate_transaction_costs(
                    shares_traded=self._expand_shares(shares_to_buy, buy_mask, num_assets),
                    prices=current_prices,
                    abs_step=self.current_absolute_step,
                    asset_mask=buy_mask
                )
                cost_of_buys = float(np.sum(buy_notional) + tc_buy)
            if np.any(shares_to_buy > 0):
                position_changes[buy_mask] += shares_to_buy
                trades_executed[buy_mask] += shares_to_buy
                traded_notional_per_asset[buy_mask] += buy_notional
                total_transaction_costs += tc_buy
                portfolio_state.cash -= cost_of_buys

        # Apply position changes
        portfolio_state.positions += position_changes

        total_traded_notional = float(np.sum(traded_notional_per_asset))
        traded_shares_total = float(np.sum(np.abs(trades_executed)))

        if total_traded_notional > 0:
            self.last_execution_step = self.current_step

        return ExecutionResult(
            current_step=self.current_step,
            trades_executed=trades_executed,
            executed_prices=current_prices,
            transaction_cost=float(total_transaction_costs),
            success=bool(total_traded_notional > 0),
            traded_dollar_value=total_traded_notional,
            traded_shares_total=traded_shares_total,
            traded_notional_per_asset=traded_notional_per_asset
        )

    def _apply_soft_execution(
        self, 
        target_weights: np.ndarray, 
        current_weights: np.ndarray,
        portfolio_state: PortfolioState
    ) -> np.ndarray:
        """
        Apply smooth partial rebalance with deadband.
        
        Instead of hard all-or-nothing gate, apply a soft step toward target.
        Maps continuous policy output → continuous weight changes.
        
        Args:
            target_weights: Desired weights from policy softmax
            current_weights: Current portfolio weights
            portfolio_state: Current portfolio state (for metrics)
        
        Returns:
            adjusted_target_weights: Executed weights (partial step toward target)
        """
        # Deadband: ignore changes smaller than this
        deadband = float(self.config['environment'].get('execution_deadband', 0.002))
        
        # Step size: how much of the desired change to actually execute (0.0–1.0)
        # 0.25–0.35 typical; smaller = more conservative, larger = more aggressive
        step_size = float(self.config['environment'].get('execution_step_size', 0.30))
        
        # Compute desired change
        weight_delta = target_weights - current_weights  # shape (num_assets,)
        
        # Apply deadband: set small changes to zero
        deadband_mask = np.abs(weight_delta) < deadband
        weight_delta_filtered = weight_delta.copy()
        weight_delta_filtered[deadband_mask] = 0.0
        
        # Apply soft step: move partway toward target
        executed_delta = step_size * weight_delta_filtered
        
        # Compute executed weights
        executed_weights = current_weights + executed_delta
        
        # Normalize (enforce w_i >= 0, sum(w) = 1)
        executed_weights = np.clip(executed_weights, 0.0, None)
        total = np.sum(executed_weights)
        if total > 1e-8:
            executed_weights = executed_weights / total
        else:
            # Fallback: if all weights vanish, stay put
            executed_weights = current_weights.copy()
        
        return executed_weights.astype(np.float32)

    def execute_instructions(self, instructions: List[TradeInstruction]) -> Tuple[ExecutionResult, List[Dict[str, Any]]]:
        """
        Execute list of TradeInstruction in simple/tranche modes.
        Returns ExecutionResult aggregate and per-trade TradeResult dicts.
        """
        num_assets = self.market_data_cache.num_assets
        asset_to_index = self.market_data_cache.asset_to_index

        trades_executed = np.zeros(num_assets, dtype=np.float32)  # shares signed
        traded_notional_per_asset = np.zeros(num_assets, dtype=np.float32)
        total_transaction_costs = 0.0
        trade_results: List[Dict[str, Any]] = []

        # Determine execution price source index
        def resolve_execution_price_idx():
            if self.price_source == "current_close":
                idx = self.current_absolute_step
            else:
                # next_open: use next bar; we price at open but costs use notional at that price
                idx = self.current_absolute_step + 1
            return idx

        for instr in instructions:
            reason = _validate_instruction(instr, asset_to_index)
            if reason:
                trade_results.append({
                    "success": False, "symbol": instr.symbol, "action": instr.action,
                    "requested_qty": float(instr.quantity) if instr.quantity is not None else None, 
                    "requested_notional": float(instr.notional) if instr.notional is not None else None,
                    "executed_qty": 0.0,
                    "execution_price": None, "notional": 0.0, "transaction_cost": 0.0,
                    "reason": reason
                })
                continue

            sym_idx = asset_to_index[instr.symbol]
            price_idx = resolve_execution_price_idx()
            # Price availability
            if price_idx < 0 or price_idx >= self.market_data_cache.num_days:
                trade_results.append({
                    "success": False, "symbol": instr.symbol, "action": instr.action,
                    "requested_qty": float(instr.quantity) if instr.quantity is not None else None,
                    "requested_notional": float(instr.notional) if instr.notional is not None else None,
                    "executed_qty": 0.0,
                    "execution_price": None, "notional": 0.0, "transaction_cost": 0.0,
                    "reason": "end_of_data"
                })
                continue

            # Execution price: open at next bar or close at current bar
            if self.price_source == "current_close":
                execution_price = float(self.market_data_cache.close_prices[price_idx, sym_idx])
            else:
                execution_price = float(self.market_data_cache.open_prices[price_idx, sym_idx])
            if not np.isfinite(execution_price) or execution_price <= 0:
                trade_results.append({
                    "success": False, "symbol": instr.symbol, "action": instr.action,
                    "requested_qty": float(instr.quantity) if instr.quantity is not None else None,
                    "requested_notional": float(instr.notional) if instr.notional is not None else None,
                    "executed_qty": 0.0,
                    "execution_price": None, "notional": 0.0, "transaction_cost": 0.0,
                    "reason": "no_price"
                })
                continue

            # LIMIT semantics
            if instr.order_type == "LIMIT":
                lp = float(instr.limit_price)
                if instr.action == "BUY" and not (lp >= execution_price):
                    trade_results.append({
                        "success": False, "symbol": instr.symbol, "action": instr.action,
                        "requested_qty": float(instr.quantity) if instr.quantity is not None else None,
                        "requested_notional": float(instr.notional) if instr.notional is not None else None,
                        "executed_qty": 0.0,
                        "execution_price": execution_price, "notional": 0.0,
                        "transaction_cost": 0.0, "reason": "limit_not_reached"
                    })
                    continue
                if instr.action == "SELL" and not (lp <= execution_price):
                    trade_results.append({
                        "success": False, "symbol": instr.symbol, "action": instr.action,
                        "requested_qty": float(instr.quantity) if instr.quantity is not None else None,
                        "requested_notional": float(instr.notional) if instr.notional is not None else None,
                        "executed_qty": 0.0,
                        "execution_price": execution_price, "notional": 0.0,
                        "transaction_cost": 0.0, "reason": "limit_not_reached"
                    })
                    continue

            # Compute requested shares based on quantity or notional
            req_qty = None
            req_notional = None
            if instr.quantity is not None:
                req_qty = float(instr.quantity)
            elif instr.notional is not None:
                req_notional = float(instr.notional)
                req_qty = float(req_notional / execution_price) if execution_price > 0 else 0.0

            current_shares = float(self.portfolio_state.positions[sym_idx])
            executed_qty = 0.0
            mode = self.execution_mode
            reason_final: Optional[str] = None

            if mode == EXECUTION_SIMPLE:
                if instr.action == "BUY":
                    # Target absolute shares = quantity
                    target_shares = req_qty
                    # Cap by max_position_shares_per_symbol
                    if self.max_position_shares_per_symbol is not None:
                        target_shares = min(target_shares, float(self.max_position_shares_per_symbol))
                    delta = target_shares - current_shares
                    if delta > 0:  # need to buy
                        executed_qty = delta
                    else:          # need to sell down
                        executed_qty = -min(-delta, current_shares)  # negative indicates sell shares
                else:  # SELL closes entire position
                    delta = -current_shares
                    if not self.allow_short:
                        delta = max(delta, -current_shares)  # cannot go below 0
                    executed_qty = delta
            elif mode == EXECUTION_TRANCHE:
                if instr.action == "BUY":
                    delta = req_qty
                    # Cap by max_position_shares_per_symbol
                    max_cap = self.max_position_shares_per_symbol
                    if max_cap is not None and (current_shares + delta) > max_cap:
                        delta = max(0.0, float(max_cap) - current_shares)
                        reason_final = "position_capped"
                    executed_qty = max(0.0, delta)  # buying shares
                else:  # SELL
                    delta = -req_qty
                    if not self.allow_short:
                        # Cap so final position >= 0
                        delta = max(delta, -current_shares)
                        if -req_qty < -current_shares:
                            reason_final = "short_not_allowed"
                    executed_qty = delta  # negative shares means sell
            else:
                # Should not reach here in portfolio mode
                delta = 0.0
                executed_qty = 0.0

            # Cash sufficiency for BUY legs
            if executed_qty > 0:
                notional = executed_qty * execution_price
                # Estimate transaction cost for buy
                total_transaction_cost = self._calculate_transaction_costs(
                    shares_traded=np.eye(1, self.market_data_cache.num_assets, sym_idx, dtype=np.float32)[0] * executed_qty,
                    prices=self.market_data_cache.close_prices[price_idx],  # costs scale with notional; price choice consistent
                    abs_step=price_idx
                )
                cash_needed = notional + total_transaction_cost
                if cash_needed > self.portfolio_state.cash:
                    # Cap shares by cash
                    cap_shares = max(0.0, (self.portfolio_state.cash - total_transaction_cost) / execution_price)
                    cap_shares = float(np.floor(cap_shares)) if cap_shares > 0 else 0.0
                    if cap_shares <= 0:
                        executed_qty = 0.0
                        notional = 0.0
                        total_transaction_cost = 0.0
                        reason_final = "insufficient_cash"
                    else:
                        executed_qty = cap_shares
                        notional = executed_qty * execution_price
                        total_transaction_cost = self._calculate_transaction_costs(
                            shares_traded=np.eye(1, self.market_data_cache.num_assets, sym_idx, dtype=np.float32)[0] * executed_qty,
                            prices=self.market_data_cache.close_prices[price_idx],
                            abs_step=price_idx
                        )
                        reason_final = "cash_capped"

                # Apply buy
                if executed_qty > 0:
                    self.portfolio_state.positions[sym_idx] += executed_qty
                    self.portfolio_state.cash -= (notional + total_transaction_cost)
                    trades_executed[sym_idx] += executed_qty
                    traded_notional_per_asset[sym_idx] += notional
                    total_transaction_costs += total_transaction_cost
                    trade_results.append({
                        "success": True, "symbol": instr.symbol, "action": instr.action,
                        "requested_qty": float(instr.quantity) if instr.quantity is not None else None,
                        "requested_notional": float(instr.notional) if instr.notional is not None else None,
                        "executed_qty": executed_qty,
                        "execution_price": execution_price, "notional": notional,
                        "transaction_cost": total_transaction_cost, "reason": reason_final
                    })
                else:
                    trade_results.append({
                        "success": False, "symbol": instr.symbol, "action": instr.action,
                        "requested_qty": float(instr.quantity) if instr.quantity is not None else None,
                        "requested_notional": float(instr.notional) if instr.notional is not None else None,
                        "executed_qty": 0.0,
                        "execution_price": execution_price, "notional": 0.0,
                        "transaction_cost": 0.0, "reason": reason_final or "insufficient_cash"
                    })
                    # print(f"TradeInstruction BUY for {instr.symbol} at price {execution_price} for "
                    #       f"qty={instr.quantity} notional={instr.notional}: could not be executed. Reason: {reason_final or 'insufficient_cash'}")
                    # print(f"    Current cash: {self.portfolio_state.cash}, needed: {cash_needed} (notional {notional} + tc {tc})")

            elif executed_qty < 0:
                sell_shares = -executed_qty
                # Cap sells if not allow_short and not enough shares (already handled)
                notional = sell_shares * execution_price
                total_transaction_cost = self._calculate_transaction_costs(
                    shares_traded=-np.eye(1, self.market_data_cache.num_assets, sym_idx, dtype=np.float32)[0] * sell_shares,
                    prices=self.market_data_cache.close_prices[price_idx],
                    abs_step=price_idx
                )
                # Apply sell
                self.portfolio_state.positions[sym_idx] -= sell_shares
                if not self.allow_short:
                    # enforce non-negative floor
                    self.portfolio_state.positions[sym_idx] = max(0.0, self.portfolio_state.positions[sym_idx])
                self.portfolio_state.cash += (notional - total_transaction_cost)
                trades_executed[sym_idx] -= sell_shares
                traded_notional_per_asset[sym_idx] += notional
                total_transaction_costs += total_transaction_cost
                trade_results.append({
                    "success": True, "symbol": instr.symbol, "action": instr.action,
                    "requested_qty": float(instr.quantity) if instr.quantity is not None else None,
                    "requested_notional": float(instr.notional) if instr.notional is not None else None,
                    "executed_qty": sell_shares,
                    "execution_price": execution_price, "notional": notional,
                    "transaction_cost": total_transaction_cost, "reason": reason_final
                })
            else:
                # No-op
                trade_results.append({
                    "success": False, "symbol": instr.symbol, "action": instr.action,
                    "requested_qty": float(instr.quantity) if instr.quantity is not None else None,
                    "requested_notional": float(instr.notional) if instr.notional is not None else None,
                    "executed_qty": 0.0,
                    "execution_price": execution_price, "notional": 0.0,
                    "transaction_cost": 0.0, "reason": reason_final or "no_execution"
                })

        traded_shares_total = float(np.sum(np.abs(trades_executed)))
        total_traded_notional = float(np.sum(traded_notional_per_asset))

        if total_traded_notional > 0:
            self.last_execution_step = self.current_step

        exec_result = ExecutionResult(
            current_step=self.current_step,
            trades_executed=trades_executed,
            executed_prices=self.portfolio_state.prices.copy(),
            transaction_cost=float(total_transaction_costs),
            success=bool(total_traded_notional > 0),
            traded_dollar_value=total_traded_notional,
            traded_shares_total=traded_shares_total,
            traded_notional_per_asset=traded_notional_per_asset
        )
        return exec_result, trade_results
    
    def _expand_shares(self, shares_subset: np.ndarray, mask: np.ndarray, num_assets: int) -> np.ndarray:
        """Utility: embed subset shares into full-length array."""
        full = np.zeros(num_assets, dtype=np.float32)
        full[mask] = shares_subset
        return full
    
    def _calculate_transaction_costs(self, shares_traded: np.ndarray, prices: np.ndarray,
                                     abs_step: int, asset_mask: Optional[np.ndarray] = None) -> float:
        """
        Transaction costs model (commission + spread + ADV-based impact).
        All terms scale with traded notional. Impact depends on fraction of ADV$.
        - commission_bps: per-dollar commission (e.g., 0.00005 = 0.5 bps)
        - half_spread_bps: expected half-spread cost (e.g., 0.0005 = 5 bps)
        - impact_coeff: multiplier on sqrt(trade_notional / ADV$) (e.g., 0.1)

        Args:
        - shares_traded: np.ndarray of shares traded per asset (positive for buy, negative for sell)
        - prices: np.ndarray of current prices per asset


        Return:
        - Total transaction costs as a float
        """
        cfg_env = self.config['environment']
        commission_bps = float(cfg_env.get('commission_bps', 0.000005))
        half_spread_bps = float(cfg_env.get('half_spread_bps', 0.00002))
        impact_coeff = float(cfg_env.get('impact_coeff', 0.3))
        adv_window = int(cfg_env.get('adv_window', 60))
        fixed_fee = float(cfg_env.get('fixed_fee_per_order', 1.0))

        # Full-length traded_notional
        traded_notional_full = np.abs(shares_traded) * prices  # [A]

        start = max(0, abs_step - adv_window)
        end = abs_step
        if end <= start:
            prev_idx = max(0, end - 1)
            adv_dollar_full = np.maximum(1.0,
                                         self.market_data_cache.volumes[prev_idx] *
                                         self.market_data_cache.close_prices[prev_idx])
        else:
            vol_slice = self.market_data_cache.volumes[start:end]      # [W, A]
            px_slice = self.market_data_cache.close_prices[start:end]  # [W, A]
            adv_dollar_full = np.maximum(1.0, np.mean(vol_slice * px_slice, axis=0))

        if asset_mask is not None:
            traded_notional = traded_notional_full[asset_mask]
            adv_dollar = adv_dollar_full[asset_mask]
        else:
            traded_notional = traded_notional_full
            adv_dollar = adv_dollar_full

        if np.sum(traded_notional) == 0.0:
            return 0.0

        commission_cost = commission_bps * np.sum(traded_notional)
        spread_cost = half_spread_bps * np.sum(traded_notional)
        impact_term = traded_notional * np.sqrt(traded_notional / adv_dollar)
        impact_cost = impact_coeff * np.sum(impact_term)
        num_orders = int(np.sum(traded_notional > 0))
        fixed_cost = fixed_fee * num_orders

        total_cost = float(commission_cost + spread_cost + impact_cost + fixed_cost)
        # Record components for episode metrics
        self._last_cost_breakdown = (
            float(commission_cost),
            float(spread_cost),
            float(impact_cost),
            float(fixed_cost),
        )
        return total_cost
    
    def _initialize_portfolio_with_costs(
        self, 
        target_positions: np.ndarray, 
        initial_prices: np.ndarray,
        initial_value: float,
        allow_cash_residual: bool = True,
        max_iterations: int = 6
    ) -> Tuple[float, np.ndarray, float]:
        """
        Calculate initial portfolio positions accounting for transaction costs.
        
        For portfolios that must be fully invested (benchmark), iteratively reduces
        position sizes to ensure costs can be paid while leaving minimal cash.
        
        Args:
            target_positions: Desired positions in shares [num_assets]
            initial_prices: Asset prices at initialization [num_assets]
            initial_value: Total portfolio value to allocate
            allow_cash_residual: If True, allows small cash remainder (for benchmark)
            max_iterations: Max adjustment iterations (3 is sufficient for convergence)
        
        Returns:
            Tuple of (final_cash, final_positions, transaction_costs)
        """
        num_assets = len(target_positions)
        
        # Calculate transaction costs for target positions
        tc = self._calculate_transaction_costs(
            shares_traded=target_positions,
            prices=initial_prices,
            abs_step=self.current_absolute_step,
            asset_mask=None
        )
        
        # Calculate gross cost (positions + transaction costs)
        position_notional = np.sum(target_positions * initial_prices)
        total_cost = position_notional + tc
        
        # Case 1: Standard portfolio (with intended cash allocation)
        if not allow_cash_residual:
            # Simple: reduce available cash by transaction costs
            final_cash = initial_value - total_cost
            return float(final_cash), target_positions.copy(), float(tc)
        
        # Case 2: Fully-invested portfolio (benchmark)
        # Must iteratively adjust to ensure: positions + costs <= initial_value
        # and leave minimal cash (< $1)
        
        if total_cost <= initial_value:
            # Lucky case: already fits with small residual
            final_cash = initial_value - total_cost
            return float(final_cash), target_positions.copy(), float(tc)
        
        # Need to scale down positions iteratively
        scale_factor = 1.0
        final_positions = target_positions.copy()
        
        for iteration in range(max_iterations):
            # Reduce scale factor to fit budget
            scale_factor *= (initial_value / total_cost) * 0.99  # 1% buffer for convergence
            final_positions = target_positions * scale_factor
            
            # Recalculate costs with scaled positions
            tc = self._calculate_transaction_costs(
                shares_traded=final_positions,
                prices=initial_prices,
                abs_step=self.current_absolute_step,
                asset_mask=None
            )
            
            position_notional = np.sum(final_positions * initial_prices)
            total_cost = position_notional + tc
            
            # Check if we're within budget
            if total_cost <= initial_value:
                final_cash = initial_value - total_cost
                # Accept if cash residual is small (< $20) or we've hit max iterations
                if final_cash < 20.0 or iteration == max_iterations - 1:
                    return float(final_cash), final_positions.astype(np.float32), float(tc)
        
        # Fallback: return best effort (should rarely reach here)
        final_cash = max(0.0, initial_value - total_cost)
        return float(final_cash), final_positions.astype(np.float32), float(tc)

    def _check_execution_thresholds(self, target_weights: np.ndarray, portfolio_state: PortfolioState) -> Tuple[bool, np.ndarray]:
        """
        Determine if execution should occur based on multiple thresholds.
        Returns (should_execute, adjusted_target_weights).
        If should_execute False, returns current weights for idempotence.
        Thresholds (configurable):
          - execution_weight_change_threshold: sum |Δw|
          - execution_min_per_asset_weight_change: max |Δw_i|
          - execution_min_trade_value_threshold: total $ to reallocate
          - execution_min_days_between_trades: min steps since last real execution
        """
        current_weights = portfolio_state.get_weights()
        # Weight diffs
        weight_diffs_assets = target_weights[1:] - current_weights[1:]  # exclude cash
        abs_diffs_assets = np.abs(weight_diffs_assets)

        # Config thresholds
        agg_change_thresh = float(self.config['environment'].get('execution_weight_change_threshold', 0.02))
        per_asset_change_thresh = float(self.config['environment'].get('execution_min_per_asset_weight_change', 0.01))
        min_trade_value_thresh = float(self.config['environment'].get('execution_min_trade_value_threshold', 250.0))
        min_days_between = int(self.config['environment'].get('execution_min_days_between_trades', 1))

        # Aggregate metrics
        sum_abs_change = float(np.sum(abs_diffs_assets))
        max_abs_change = float(np.max(abs_diffs_assets)) if abs_diffs_assets.size else 0.0
        total_value = float(portfolio_state.get_total_value())
        # Approx dollar turnover (assets only)
        combined_trade_value = float(np.sum(abs_diffs_assets * total_value))

        # Days since last execution
        steps_since_last = (self.current_step - self.last_execution_step) if self.last_execution_step >= 0 else np.inf

        # Gate checks
        if max_abs_change < per_asset_change_thresh:
            return False, current_weights
        if sum_abs_change < agg_change_thresh:
            return False, current_weights
        if combined_trade_value < min_trade_value_thresh:
            return False, current_weights
        if steps_since_last < min_days_between:
            return False, current_weights

        # Normalize tiny floating noise (optional deadband)
        # If after rounding diffs vanish, skip execution
        if np.all(abs_diffs_assets < 1e-5):
            return False, current_weights

        # Passed all gates
        return True, target_weights


    def get_observation_sequence(self):
        """
        Generate sequential observation for DRL agent with maximum computational efficiency.

        Returns:
            Flattened observation array optimized for SB3 LSTM processing
            Shape: (total_observation_size,)
            - feature part: [lookback_window, num_assets, num_asset_features]
            - portfolio part: [lookback_window, num_portfolio_features]            
        
        Performance Optimizations:
        - Reuse pre-allocated arrays where possible
        - Vectorized numpy operations throughout
        - Minimal memory allocations
        - Efficient data access patterns
        - In-place operations when safe
        """

        # Get features lookback: shape [lookback_window, num_assets, num_selected_features]
        features_sequence = self.market_data_cache.get_features_lookback(self.current_absolute_step, self.lookback_window)

        # Get portfolio lookback: shape [lookback_window, num_portfolio_features]
        # Includes sequences for: weights, alpha, sharpe_ratio, drawdown, volatility, turnover,
        #                         allocator_rewards
        portfolio_sequence = self.episode_buffer.get_observation_lookback()
        
        # Flatten components, if necessary
        features_sequence_flat = features_sequence.flatten()
        portfolio_sequence_flat = portfolio_sequence.flatten()

        # Check that the flattened observation matches the predefined observation space shape
        assert features_sequence_flat.shape[0] + portfolio_sequence_flat.shape[0] == self.observation_space.shape[0], \
            f"Observation shape mismatch: expected {self.observation_space.shape[0]}, got {features_sequence_flat.shape[0] + portfolio_sequence_flat.shape[0]}"
        
        # Concatenate all components
        observation = np.concatenate([
            features_sequence_flat,
            portfolio_sequence_flat
        ]).astype(np.float32)

        return observation 
    
    def get_observation_single_step(self):
        """
        Generate single-step observation for agents like RecurrentPPO or PPO that do not require sequences.

        Returns:
            Flattened observation array optimized for SB3 processing
            Shape: (total_observation_size,)
            - feature part: [num_assets, num_asset_features]
            - portfolio part: [num_portfolio_features]            
        
        Performance Optimizations:
        - Reuse pre-allocated arrays where possible
        - Vectorized numpy operations throughout
        - Minimal memory allocations
        - Efficient data access patterns
        - In-place operations when safe
        """

        # Use the last recorded external step from EpisodeBuffer for portfolio features
        # We recorded with external_step = self.current_step
        external_step_for_obs = self.current_step
        if self.maybe_provide_sequence:
            internal_step = external_step_for_obs + self.lookback_window
        else:
            internal_step = external_step_for_obs
        # Clamp to valid buffer range
        internal_step = int(np.clip(internal_step, 0, self.episode_buffer.portfolio_values.shape[0] - 1))

        
        if self.execution_mode == EXECUTION_SINGLE_ASSET_TARGET_POS:
            # Pull only choosen asset features
            asset_index = self.selected_asset_index
            if asset_index is None:
                raise ValueError("Selected asset index is not set for SINGLE_ASSET_TARGET_POS mode.")
            # Get features for current step: shape [num_selected_features]
            features = self.market_data_cache.get_features_at_step(self.current_absolute_step)[asset_index]
            # Get portfolio features for current step: shape [num_portfolio_features]
            # Map to internal buffer index considering lookback warmup
            internal_step = external_step_for_obs + (self.lookback_window if self.maybe_provide_sequence else 0)

            # Extract weights: index 0 is cash, index (asset_index + 1) is the selected asset
            weights_vec = self.episode_buffer.portfolio_weights[internal_step]
            cash_weight = float(weights_vec[0])
            asset_weight = float(weights_vec[asset_index + 1])

            # Calc log metrics for cash and asset notional
            cash_notional = cash_weight * self.episode_buffer.portfolio_values[internal_step]
            asset_notional = asset_weight * self.episode_buffer.portfolio_values[internal_step]
            initial_portfolio_value = self.initial_portfolio_value

            cash_log_value = np.log(cash_notional / initial_portfolio_value) if cash_notional > 0 else 0.0
            asset_log_value = np.log(asset_notional / initial_portfolio_value) if asset_notional > 0 else 0.0


            # Daily return of the full portfolio for this step
            daily_portfolio_return = float(self.episode_buffer.returns[internal_step])

            # Daily return of just cash change plus selected asset notional change
            daily_agent_return = float(self.episode_buffer.saa_returns[internal_step])

            # Last agents action
            last_action = float(self.episode_buffer.actions[internal_step, asset_index + 1])
            
            # Build minimal portfolio features for single-asset mode
            portfolio_features = np.array([cash_log_value, asset_log_value, daily_agent_return, last_action], dtype=np.float32)

            # Verify that all values contain numeric values before concatenation and guard!
            if not np.all(np.isfinite(features)):
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"Warning: Non-finite feature values encountered at step {self.current_absolute_step} for asset index {asset_index}. Replaced with zeros.")
            if not np.all(np.isfinite(portfolio_features)):
                portfolio_features = np.nan_to_num(portfolio_features, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"Warning: Non-finite portfolio feature values encountered at step {internal_step}. Replaced with zeros.")
                
            observation = np.concatenate([
                features,
                portfolio_features
            ]).astype(np.float32)

            return observation

        else:    
            # Get features for current step: shape [num_assets, num_selected_features]
            features = self.market_data_cache.get_features_at_step(self.current_absolute_step)

            # Get portfolio features for current step: shape [num_portfolio_features]
            portfolio_features = self.episode_buffer.get_observation_at_step(external_step_for_obs)

            # Verify that all values contain numeric values before concatenation and guard!
            if not np.all(np.isfinite(features)):
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"Warning: Non-finite feature values encountered at step {self.current_absolute_step} for asset index {asset_index}. Replaced with zeros.")
            if not np.all(np.isfinite(portfolio_features)):
                portfolio_features = np.nan_to_num(portfolio_features, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"Warning: Non-finite portfolio feature values encountered at step {internal_step}. Replaced with zeros.")
                

            # Concatenate all components
            observation = np.concatenate([
                features.flatten(),
                portfolio_features
            ]).astype(np.float32)

            return observation