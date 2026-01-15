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
EXECUTION_SIMPLE = "simple"
EXECUTION_TRANCHE = "tranche"
EXECUTION_PORTFOLIO = "portfolio"

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
    """Current portfolio state representation using positions and cash balance"""
    cash: float
    positions: np.ndarray
    prices: np.ndarray
    step: int
    terminated: bool

    def get_asset_values(self) -> np.ndarray:
        # Returns market value of each position excluding cash!
        return self.positions * self.prices

    def get_total_value(self) -> float:
        # Returns total portfolio value including cash
        return self.cash + np.sum(self.get_asset_values())
    
    def get_weights(self) -> np.ndarray:
        # Returns current portfolio weights including cash
        total_value = self.get_total_value()
        if total_value <= 1e-8: # guard against zero
            return np.zeros(len(self.positions) + 1)
        asset_values = self.get_asset_values()
        weights = np.concatenate(([self.cash / total_value], asset_values / total_value))
        # Finite check
        if not np.all(np.isfinite(weights)):
            print(f"Debug Info - Cash: {self.cash}, Total Value: {total_value}, Asset Values: {asset_values}, Weights: {weights}")
            raise ValueError("Non-finite weights detected in portfolio state.")
        return weights
    
    def portfolio_reset(self, cash: float, positions: np.ndarray, prices: np.ndarray, step: int, terminated: bool) -> None:
        # Reset portfolio state
        self.cash = cash
        self.positions = positions
        self.prices = prices
        self.step = step
        self.terminated = terminated


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
    benchmark_portfolio_value: np.ndarray = field(init=False)   # [episode_buffer_length_days] - benchmark portfolio value each step
    alpha: np.ndarray = field(init=False)                       # [episode_buffer_length_days] - excess returns over benchmark
    returns: np.ndarray = field(init=False)                     # [episode_buffer_length_days] - daily returns
    allocator_rewards: np.ndarray = field(init=False)           # [episode_buffer_length_days] - RL allocator rewards
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

    # weights, alpha, sharpe_ratio, drawdown, volatility, turnover, allocator_rewards

    def __post_init__(self):
        """Initialize all arrays with proper shapes and types"""
        dtype = np.float32
        self.portfolio_values = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.portfolio_weights = np.zeros((self.episode_buffer_length_days, self.num_assets + 1), dtype=dtype)
        self.portfolio_positions = np.zeros((self.episode_buffer_length_days, self.num_assets), dtype=dtype)
        self.benchmark_portfolio_value = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.alpha = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.returns = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.allocator_rewards = np.zeros(self.episode_buffer_length_days, dtype=dtype)
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
        self.num_portfolio_features = self.num_assets + 1 + 6  # weights + alpha + sharpe + drawdown + volatility + turnover + allocator_rewards
        self.action_entropy = np.zeros(self.episode_buffer_length_days, dtype=dtype) 
        # Reward component tracking (per-step)
        self.reward_alpha = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_risk = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_portfolio_return = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_cost = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_turnover = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_concentration = np.zeros(self.episode_buffer_length_days, dtype=dtype)
        self.reward_survival = np.zeros(self.episode_buffer_length_days, dtype=dtype)

    def record_step(self, external_step: int, portfolio_value: float, weights: np.ndarray, portfolio_positions: np.ndarray,
                   daily_return: float, reward: float, action: np.ndarray,
                   transaction_cost: float, prices: np.ndarray,
                   sharpe_ratio: float = 0.0, drawdown: float = 0.0, volatility: float = 0.0, turnover: float = 0.0, alpha: float = 0.0, benchmark_portfolio_value: float = 0.0,
                   traded_dollar_volume: float = 0.0, traded_shares_total: float = 0.0,
                   action_entropy: float = 0.0,
                   reward_parts: Optional[Dict[str, float]] = None) -> None:
        
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
        self.allocator_rewards[internal_offset_step] = reward
        self.actions[internal_offset_step] = action
        self.transaction_costs[internal_offset_step] = transaction_cost
        self.asset_prices[internal_offset_step] = prices
        self.sharpe_ratio[internal_offset_step] = sharpe_ratio
        self.drawdown[internal_offset_step] = drawdown
        self.volatility[internal_offset_step] = volatility
        self.turnover[internal_offset_step] = turnover
        self.alpha[internal_offset_step] = alpha
        self.benchmark_portfolio_value[internal_offset_step] = benchmark_portfolio_value
        self.traded_dollar_volume[internal_offset_step] = float(traded_dollar_volume)
        self.traded_shares_total[internal_offset_step] = float(traded_shares_total)
        self.action_entropy[internal_offset_step] = float(action_entropy)
        # Reward components
        if reward_parts is not None:
            self.reward_alpha[internal_offset_step] = reward_parts.get("alpha_component", 0.0)
            self.reward_risk[internal_offset_step] = reward_parts.get("risk_component", 0.0)
            self.reward_portfolio_return[internal_offset_step] = reward_parts.get("portfolio_return_component", 0.0)
            self.reward_cost[internal_offset_step] = reward_parts.get("cost_component", 0.0)
            self.reward_turnover[internal_offset_step] = reward_parts.get("turnover_component", 0.0)
            self.reward_concentration[internal_offset_step] = reward_parts.get("concentration_component", 0.0)
            self.reward_survival[internal_offset_step] = reward_parts.get("survival_component", 0.0)

        # Update current step and episode length
        self.current_step = external_step
        self.episode_length = min(external_step + 1, self.episode_buffer_length_days)

    def get_returns_window(self, window: int) -> np.ndarray:
        """Get last N returns for risk calculations (no wrap-around, no ring buffer)"""
        end_idx = self.current_step + 1
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
        end_idx = self.current_step + 1
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
        num_portfolio_features = 6 + self.portfolio_weights.shape[1]  # weights + alpha + sharpe + drawdown + volatility + turnover + allocator_rewards
        obs = np.zeros((self.lookback_window, num_portfolio_features), dtype=np.float32)

        # Gather sequences
        # weights: [lookback_window, num_assets+1]
        weights_seq = self.portfolio_weights[start_idx:end_idx+1]
        alpha_seq = self.alpha[start_idx:end_idx+1].reshape(-1, 1)
        sharpe_seq = self.sharpe_ratio[start_idx:end_idx+1].reshape(-1, 1)
        drawdown_seq = self.drawdown[start_idx:end_idx+1].reshape(-1, 1)
        volatility_seq = self.volatility[start_idx:end_idx+1].reshape(-1, 1)
        turnover_seq = self.turnover[start_idx:end_idx+1].reshape(-1, 1)
        rewards_seq = self.allocator_rewards[start_idx:end_idx+1].reshape(-1, 1)

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
        rewards = self.allocator_rewards[internal_step]

        observation = np.concatenate([
            weights,
            [alpha, sharpe, drawdown, volatility, turnover, rewards]
        ]).astype(np.float32)

        return observation
    
    def reset_episode_buffer(self) -> None:
        """
        Reset buffer for new episode, keeping num_assets and episode_length_days unchanged.
        Apply data warm-up for lookback_window days (entries)
        """
        self.current_step = 0

        # Reset all arrays to zero except lookback_window, num_assets and episode_length_days
        arrays_to_reset = [
            self.portfolio_values, self.portfolio_weights, self.portfolio_positions,
            self.benchmark_portfolio_value, self.alpha, self.returns,
            self.allocator_rewards, self.actions, self.transaction_costs,
            self.sharpe_ratio, self.drawdown, self.volatility, self.turnover,
            self.asset_prices, self.traded_dollar_volume, self.traded_shares_total,
            self.action_entropy, self.reward_alpha, self.reward_risk,
            self.reward_portfolio_return, self.reward_cost, self.reward_turnover,
            self.reward_concentration, self.reward_survival
        ]

        for array in arrays_to_reset:
            array.fill(0.0)

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
        
        # Get all available features (exclude OHLCV columns)
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
        num_available_features = len(available_feature_cols)
        
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
        
        Args:
            available_features: List of all available feature names from CSV
            config: Configuration dict with features section
            
        Returns:
            List of selected feature names
            
        Example config:
            config = {
                "features": {
                    "return_1d": True,
                    "return_3d": False,
                    "rsi_14": True,
                    "macd": True,
                    # ... more features
                }
            }
        """
        if 'features' not in config:
            print("Warning: No 'features' section in config. Using all available features.")
            return available_features
        
        features_config = config['features']
        selected_features = []
        
        # Check each available feature against config
        for feature_name in available_features:
            if feature_name in features_config:
                if features_config[feature_name] is True:
                    selected_features.append(feature_name)
            else:
                # Feature not in config - you can decide default behavior
                print(f"Warning: Feature '{feature_name}' not found in config. Skipping.")
        
        if not selected_features:
            print("Warning: No features selected! Using all available features as fallback.")
            return available_features
        
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
        if random_seed is not None:
            np.random.seed(random_seed)
        
        blocks = self.train_blocks if mode == 'train' else self.validation_blocks
        weights = self.block_sampling_weights[mode]
        
        if len(blocks) == 0:
            raise ValueError(f"No {mode} blocks available")
        
        if len(weights) == 0 or weights.sum() == 0:
            raise ValueError(f"Invalid sampling weights for {mode} blocks")
        
        # Sample block based on weights
        block_idx = np.random.choice(len(blocks), p=weights)
        selected_block = blocks[block_idx]
        
        # Sample start step within selected block
        if selected_block.max_start_step <= selected_block.min_start_step:
            start_step = selected_block.min_start_step
        else:
            start_step = np.random.randint(
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
    Main Gym environment class for multi-asset trading optimized for DRL.
    
    Key Features:
    - Multi-asset portfolio management with cash position
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
        self.max_reward_risk_window = int(min(self.episode_length_days // 4, 50))
        self.lookback_window = config["environment"]["lookback_window"]
        self.initial_portfolio_value = config["environment"]["initial_portfolio_value"]
        self.early_stopping_threshold = config["environment"]["early_stopping_threshold"]
        self.cash_return_rate = config["environment"]["cash_return_rate"]
        self.min_initial_cash_allocation = config["environment"]["min_initial_cash_allocation"]
        self.seed = config["environment"]["seed"]
        # Execution parameters
        self.execution_weight_change_threshold = config["environment"]["execution_weight_change_threshold"]
        self.execution_min_trade_value_threshold = config["environment"]["execution_min_trade_value_threshold"]
        self.execution_min_days_between_trades = config["environment"]["execution_min_days_between_trades"]              # e.g., 0.0005
        self.maybe_provide_sequence = config['environment']['maybe_provide_sequence']  # Whether to provide sequence data in observations

        # Store references
        self.market_data_cache = market_data_cache
        self.config = config
        self.mode = mode  # 'train' or 'validation'
        self.threshold_val = self.initial_portfolio_value * self.early_stopping_threshold

        # Execution-mode config (backwards compatible defaults)
        self.execution_mode = config["environment"].get("execution_mode", EXECUTION_PORTFOLIO)
        self.quantity_type = config["environment"].get("quantity_type", "shares")
        self.price_source = config["environment"].get("price_source", "next_open")  # "next_open" | "current_close"
        self.allow_short = bool(config["environment"].get("allow_short", False))
        self.max_position_shares_per_symbol = config["environment"].get("max_position_shares_per_symbol", None)
        if self.quantity_type != "shares":
            raise ValueError("Only 'shares' quantity_type is supported in simple/tranche modes.")
        if self.execution_mode not in {EXECUTION_SIMPLE, EXECUTION_TRANCHE, EXECUTION_PORTFOLIO}:
            raise ValueError(f"Invalid execution_mode: {self.execution_mode}")
        
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

        if self.execution_mode == EXECUTION_PORTFOLIO:
            # ------ Action space: Continuous weights for each asset + cash (sum to 1) ------
            self.action_space = spaces.Box(
                low=-5.0, high=5.0, shape=(num_assets + 1,), dtype=np.float32
            )
        else:
            # Simple/Tranche modes: action is a list of instructions; Gym does not have a list space.
            # Keep a generic Box to satisfy Gym, but we validate structure in step().
            self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # ------ Observation space design for multi-encoder architecture ------
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

    def reset(self, seed: Optional[int] = None, option: Optional[Dict] = None):
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
        
        # Step 0: Reset episode step counter and last executed trade step
        self.current_step = 0
        self.last_execution_step = -1
        self.current_episode = (self.current_episode + 1) if self.current_episode is not None else 0
        
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
        
        # Calculate initial cash and positions from weights
        initial_cash = self.initial_portfolio_value * random_weights[0]
        asset_weights = random_weights[1:]
        
        # Calculate initial positions (shares to buy)
        initial_positions = np.zeros(num_assets, dtype=np.float32)
        for i in range(num_assets):
            asset_allocation = self.initial_portfolio_value * asset_weights[i]
            if asset_allocation > 0 and initial_prices[i] > 0:
                initial_positions[i] = asset_allocation / initial_prices[i]

        # Update portfolio state instance
        self.portfolio_state.portfolio_reset(
            cash=initial_cash,
            positions=initial_positions,
            prices=initial_prices,
            step=self.current_step,
            terminated=False
        )

        # Validate initial portfolio value matches configuration
        actual_initial_value = self.portfolio_state.get_total_value()
        if abs(actual_initial_value - self.initial_portfolio_value) > 1: # dollar - level accuracy
            raise ValueError(f"Portfolio initialization error: {actual_initial_value} != {self.initial_portfolio_value}")

        # Update Benchmark Portfolio (custom allocation, no cash)
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

        # Convert weights (no cash) into positions
        benchmark_position_values = self.initial_portfolio_value * bench_weights
        benchmark_positions = np.where(initial_prices > 0,
                                       benchmark_position_values / initial_prices,
                                       0.0).astype(np.float32)
        
        self.benchmark_portfolio_state.portfolio_reset(
            cash=0.0,
            positions=benchmark_positions,
            prices=initial_prices,
            step=self.current_step,
            terminated=False
        )

        # Seed a pre-step entry in the EpisodeBuffer so the first observation contains real weights
        initial_weights = self.portfolio_state.get_weights().astype(np.float32)
        zero_action = np.zeros(self.market_data_cache.num_assets + 1, dtype=np.float32)
        self.episode_buffer.record_step(
            external_step=0,  # BEFORE: mapped to internal index lookback_window - 1, or -1 directly
            portfolio_value=actual_initial_value,
            weights=initial_weights,
            portfolio_positions=self.portfolio_state.positions.copy(),
            daily_return=0.0,
            reward=0.0,
            action=zero_action,
            transaction_cost=0.0,
            prices=initial_prices,
            sharpe_ratio=0.0,
            drawdown=0.0,
            volatility=0.0,
            turnover=0.0,
            alpha=0.0,
            benchmark_portfolio_value=self.benchmark_portfolio_state.get_total_value()
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

    def step(self, action: np.array) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step in the environment.

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

        Process:
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

        trade_results: List[Dict[str, Any]] = []
        
        # Branch based on execution mode

        # ----- Portfolio mode: action is target weights vector -----
        if self.execution_mode == EXECUTION_PORTFOLIO:
            # Validate action input ------------------------------------
            # verify raw input is within reasonable bounds
            if np.any(np.isnan(action)):
                raise ValueError("Raw Action input contains NaN values")
            raw_action = np.asarray(action, dtype=np.float32)
            # Softmax for simplex mapping (numerical stability)
            max_a = np.max(raw_action)
            exp_a = np.exp(raw_action - max_a)
            action = exp_a / np.sum(exp_a) + 1e-8  # add small constant to avoid exact zeros

            action_sum = np.sum(action)
            if abs(action_sum - 1.0) > 1e-4:
                raise ValueError(f"Action weights must sum to 1.0, got {action_sum}")
            if np.any(action < 0):
                raise ValueError("Action weights must be non-negative")
            if len(action) != self.market_data_cache.num_assets + 1:
                raise ValueError(f"Action length must be {self.market_data_cache.num_assets + 1}, got {len(action)}")
            if any(np.isnan(action)):
                raise ValueError("Action contains NaN values after softmax processing")
            if any(np.isinf(action)):
                raise ValueError("Action contains Inf values after softmax processing")

            # Diagnostics
            raw_logits_mean = float(np.mean(raw_action))
            raw_logits_std  = float(np.std(raw_action))
            raw_logits_max  = float(np.max(raw_action))
            raw_logits_l2   = float(np.linalg.norm(raw_action))
            softmax_var     = float(np.var(raw_action))
            softmax_temp    = float(np.sqrt(softmax_var) + 1e-8)  # indicative temperature

            # Allocation entropy (exclude numerical floor)
            w = action / np.sum(action)
            entropy = float(-np.sum(w * np.log(np.clip(w, 1e-8, 1.0))))                

            # Placeholder for future EXECUTOR logic ----------------------
            # NOTE: Will be replaced by EXECUTOR in future
            execution_result = self.execute_portfolio_change(
                target_weights=action,
                portfolio_state=self.portfolio_state
            )

        else:
            # Handle remnants of allocator weights 
            raw_logits_mean = 0.0
            raw_logits_std = 0.0
            raw_logits_max = 0.0
            raw_logits_l2 = 0.0
            softmax_temp = 0.0

            # ------- Simple/Tranche instruction path -------
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
            entropy = float(-np.sum((touched / max(total_touch, 1.0)) * np.log(np.clip(touched / max(total_touch, 1.0), 1e-8, 1.0)))) if total_touch > 0 else 0.0
            # For buffer, we don’t have weights vector from action; we will store current weights post-execution below.
            action_weights = None  # not used in reward in this path


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

        # Update benchmark portfolio state
        self.benchmark_portfolio_state.prices[:] = new_prices # in-place update
        self.benchmark_portfolio_state.step = self.current_step
        
        # Calculate new portfolio value after price changes
        portfolio_value_after = self.portfolio_state.get_total_value()
        benchmark_portfolio_value_after = self.benchmark_portfolio_state.get_total_value()

        # Calculate Allocator Reward
        allocator_reward, reward_parts = self.calculate_allocator_step_reward(
            execution_result=execution_result,
            portfolio_before=portfolio_value_before,
            portfolio_after=portfolio_value_after,
            benchmark_before=benchmark_portfolio_value_before,
            benchmark_after=benchmark_portfolio_value_after,
            action=(action if self.execution_mode == EXECUTION_PORTFOLIO else None)
        )

        # Check termination conditions -----------------------------------
        low_value_cond = portfolio_value_after <= self.threshold_val
        zero_value_cond = portfolio_value_after < -1e-1
        negative_cash_cond = self.portfolio_state.cash < -5.0  # allow small negative cash for numerical stability

        terminated = bool(low_value_cond or zero_value_cond or negative_cash_cond)

        if terminated:
            reasons = []
            if low_value_cond:
                reasons.append(f"portfolio value {portfolio_value_after:.2f} <= early stopping threshold {self.threshold_val:.2f}")
            if zero_value_cond:
                reasons.append(f"portfolio value {portfolio_value_after:.2f} <= 0")
            if negative_cash_cond:
                reasons.append(f"cash balance {self.portfolio_state.cash:.2f} < -0.10 (tolerance -1e-1)")
            else:
                reasons.append("unknown reason")

            # Print all triggered reasons together
            print(f"Episode {self.current_episode} terminated early due to: " + "; ".join(reasons))
        
        truncated = self.current_step >= self.episode_length_days
        self.portfolio_state.terminated = terminated or truncated
 
        # Record step data
        current_weights_after_execution = self.portfolio_state.get_weights()
        daily_return = (portfolio_value_after / portfolio_value_before - 1.0 if portfolio_value_before > 0 else 0.0)
        
        # FIX: off-by-one error in episode buffer recording
        external_step = self.current_step - 1  # zero-based index for buffer

        # Construct action vector for buffer:
        # Portfolio mode: store weights action (as before).
        # Simple/Tranche: store [net_cash_delta, per-asset traded $] computed from execution_result.
        if self.execution_mode == EXECUTION_PORTFOLIO:
            action_vec = action_weights.astype(np.float32)
        else:
            net_asset_dollars = execution_result.traded_notional_per_asset.copy()
            net_asset_dollars[np.isnan(net_asset_dollars)] = 0.0
            net_cash_delta = -float(np.sum(net_asset_dollars))  # costs handled separately in transaction_cost
            action_vec = np.concatenate([[net_cash_delta], net_asset_dollars]).astype(np.float32)

        self.episode_buffer.record_step(
            external_step=external_step,
            portfolio_value=portfolio_value_after,
            portfolio_positions=self.portfolio_state.positions,
            weights=current_weights_after_execution,
            daily_return=daily_return,
            reward=allocator_reward,
            action=action_vec,
            transaction_cost=execution_result.transaction_cost,
            prices=new_prices,
            traded_dollar_volume=execution_result.traded_dollar_value,
            traded_shares_total=execution_result.traded_shares_total,
            action_entropy=entropy,
            reward_parts=reward_parts
        )
        
        # Prepare info
        info = self._get_info()
        if self.execution_mode in {EXECUTION_SIMPLE, EXECUTION_TRANCHE}:
            info['trade_results'] = trade_results

        # Prepare next observation if not terminated
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
                    np.sum(self.episode_buffer.allocator_rewards[internal_start:internal_end + 1])
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
            port_ret = (final_portfolio_value / self.initial_portfolio_value) - 1.0
            bench_ret = (final_benchmark_value / self.initial_portfolio_value) - 1.0
            alpha_ret = port_ret - bench_ret

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

            info.update({
                "episode_final": True,
                "episode_id": self.current_episode,
                "episode_length": external_step + 1,
                "portfolio_final_value": final_portfolio_value,
                "benchmark_final_value": final_benchmark_value,
                "portfolio_return": port_ret,
                "benchmark_return": bench_ret,
                "alpha_return": alpha_ret,
                "cumulative_reward": cumulative_reward,
                "episode_max_drawdown": episode_max_dd,
                "episode_sharpe": episode_sharpe,
                "episode_volatility": episode_volatility,
                "total_transaction_costs": total_transaction_costs,
                "num_trade_days": num_trade_days,
                "avg_turnover": avg_turnover,
                "total_turnover": total_turnover,
                "episode_traded_notional": episode_traded_notional,
                "episode_traded_shares": episode_traded_shares,
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
                "reward_components/survival_sum": comp_surv_sum
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

        # Return all step outputs (observation, reward, terminated, truncated, info)
        # NOTE: only next_observation must be a sequence for LSTM! TODO: Verify this statement
        return next_observation, allocator_reward, terminated, truncated, info
    

    def calculate_allocator_step_reward(
            self, execution_result: ExecutionResult, portfolio_before,
            portfolio_after, benchmark_before, benchmark_after, action: np.ndarray
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
        """
        
        # ============================================
        # 1. ALPHA CALCULATION (PRIMARY SIGNAL)
        # ============================================
        # Calculate daily returns. Using log returns for stability
        if portfolio_before > 0 and portfolio_after > 0:
            portfolio_return = (portfolio_after / portfolio_before)
        else:
            portfolio_return = 0.0
        if benchmark_before > 0 and benchmark_after > 0:
            benchmark_return = (benchmark_after / benchmark_before)
        else:
            benchmark_return = 0.0

        alpha = (portfolio_return - benchmark_return)  # daily alpha
        portfolio_return = portfolio_return - 1.0  # daily portfolio return
        
        # ============================================
        # 2. RISK-ADJUSTED PERFORMANCE METRICS
        # ============================================
        risk_adjustment = 0.0

        # Calculate dynamic risk window. Use self.reward_risk_window and the current step to produce
        # behaviour which starts at 2 raises with the steps up to maximum risk_reward_window
        if self.current_step <= 2:
            risk_metric_window = 2
        else:
            risk_metric_window = min(self.current_step, self.max_reward_risk_window)

        # Use episode buffer's efficient methods
        sharpe_ratio = self.episode_buffer.calculate_sharpe_ratio(
            window=risk_metric_window
        )
        
        max_drawdown = self.episode_buffer.calculate_max_drawdown(
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
        # 4. TURNOVER PENALTY (IMPROVED)
        # ============================================
        current_weights = self.portfolio_state.get_weights()
        if self.current_step > 0:
            if self.maybe_provide_sequence:
                prev_w = self.episode_buffer.portfolio_weights[
                    self.episode_buffer.lookback_window + (self.current_step - 1)
                ]
            else:
                prev_w = self.episode_buffer.portfolio_weights[
                    (self.current_step - 1)
                ]
            turnover = float(np.sum(np.abs(current_weights[1:] - prev_w[1:])))
        else:
            turnover = 0.0
        turnover_penalty = 0.01 * max(0.0, turnover - 0.2)**2  # quadratic beyond 20%
        
        # ============================================
        # 5. POSITION CONCENTRATION PENALTY
        # ============================================
        # Concentration (Herfindahl index)
        asset_w = current_weights[1:]
        herfindahl = float(np.sum(asset_w**2))
        h_threshold = 1.0 / max(1, len(asset_w)) * 5  # allow up to 5x uniform concentration before penalty
        concentration_penalty = 0.4 * max(0.0, herfindahl - h_threshold)
        # Cash penalty (to avoid excessive cash holdings) starting linearly at 20% cash holdings
        cash_w = current_weights[0]
        cash_penalty = 0.9 * max(0.0, cash_w - 0.20)
        concentration_penalty += cash_penalty

        # Small survival bonus
        # survival_bonus = min(max(0, (0.00005 * (self.current_step - self.episode_length_days/2))), 0.002) # bounded bonus starting at surviving over half the episode length
        survival_bonus = 0.0 # test for reducing noise

        # ============================================
        # 7. COMBINE ALL COMPONENTS WITH PROPER WEIGHTS
        # ============================================
        # factors to scale components
        alpha_factor = 0.0
        portfolio_return_factor = 2.0
        risk_adjustment_factor = 1.0
        cost_penalty_factor = 0.0
        turnover_penalty_factor = 0.0
        concentration_penalty_factor = 1.0

        reward = (
            alpha * alpha_factor +                    # Primary objective: alpha generation (scaled up)
            portfolio_return * portfolio_return_factor +          # Also reward general portfolio return
            risk_adjustment * risk_adjustment_factor +           # Risk-adjusted performance bonus
            -cost_penalty * cost_penalty_factor +             # Transaction cost efficiency
            -turnover_penalty * turnover_penalty_factor +         # Turnover management
            -concentration_penalty * concentration_penalty_factor +      # Position risk management
            + survival_bonus                  # Small survival bonus
        )

        parts = {
            "alpha_component": alpha * alpha_factor,
            "risk_component": risk_adjustment * risk_adjustment_factor,
            "portfolio_return_component": portfolio_return * portfolio_return_factor,
            "cost_component": -cost_penalty * cost_penalty_factor,
            "turnover_component": -turnover_penalty * turnover_penalty_factor,
            "concentration_component": -concentration_penalty * concentration_penalty_factor,
            "survival_component": survival_bonus,
            "raw_alpha": alpha,
            "raw_portfolio_return": portfolio_return,
            "raw_benchmark_return": benchmark_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }

        return float(reward), parts

    def execute_portfolio_change(self, target_weights: np.ndarray, portfolio_state: PortfolioState) -> ExecutionResult:
        num_assets = self.market_data_cache.num_assets
        current_prices = portfolio_state.prices.copy()

        should_execute, adjusted_target_weights = self._check_execution_thresholds(target_weights, portfolio_state)
        if not should_execute:
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

        current_weights = portfolio_state.get_weights()
        total_value = portfolio_state.get_total_value()

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
                tc = self._calculate_transaction_costs(
                    shares_traded=np.eye(1, self.market_data_cache.num_assets, sym_idx, dtype=np.float32)[0] * executed_qty,
                    prices=self.market_data_cache.close_prices[price_idx],  # costs scale with notional; price choice consistent
                    abs_step=price_idx
                )
                cash_needed = notional + tc
                if cash_needed > self.portfolio_state.cash:
                    # Cap shares by cash
                    cap_shares = max(0.0, (self.portfolio_state.cash - tc) / execution_price)
                    cap_shares = float(np.floor(cap_shares)) if cap_shares > 0 else 0.0
                    if cap_shares <= 0:
                        executed_qty = 0.0
                        notional = 0.0
                        tc = 0.0
                        reason_final = "insufficient_cash"
                    else:
                        executed_qty = cap_shares
                        notional = executed_qty * execution_price
                        tc = self._calculate_transaction_costs(
                            shares_traded=np.eye(1, self.market_data_cache.num_assets, sym_idx, dtype=np.float32)[0] * executed_qty,
                            prices=self.market_data_cache.close_prices[price_idx],
                            abs_step=price_idx
                        )
                        reason_final = "cash_capped"

                # Apply buy
                if executed_qty > 0:
                    self.portfolio_state.positions[sym_idx] += executed_qty
                    self.portfolio_state.cash -= (notional + tc)
                    trades_executed[sym_idx] += executed_qty
                    traded_notional_per_asset[sym_idx] += notional
                    total_transaction_costs += tc
                    trade_results.append({
                        "success": True, "symbol": instr.symbol, "action": instr.action,
                        "requested_qty": float(instr.quantity) if instr.quantity is not None else None,
                        "requested_notional": float(instr.notional) if instr.notional is not None else None,
                        "executed_qty": executed_qty,
                        "execution_price": execution_price, "notional": notional,
                        "transaction_cost": tc, "reason": reason_final
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
                tc = self._calculate_transaction_costs(
                    shares_traded=-np.eye(1, self.market_data_cache.num_assets, sym_idx, dtype=np.float32)[0] * sell_shares,
                    prices=self.market_data_cache.close_prices[price_idx],
                    abs_step=price_idx
                )
                # Apply sell
                self.portfolio_state.positions[sym_idx] -= sell_shares
                if not self.allow_short:
                    # enforce non-negative floor
                    self.portfolio_state.positions[sym_idx] = max(0.0, self.portfolio_state.positions[sym_idx])
                self.portfolio_state.cash += (notional - tc)
                trades_executed[sym_idx] -= sell_shares
                traded_notional_per_asset[sym_idx] += notional
                total_transaction_costs += tc
                trade_results.append({
                    "success": True, "symbol": instr.symbol, "action": instr.action,
                    "requested_qty": float(instr.quantity) if instr.quantity is not None else None,
                    "requested_notional": float(instr.notional) if instr.notional is not None else None,
                    "executed_qty": sell_shares,
                    "execution_price": execution_price, "notional": notional,
                    "transaction_cost": tc, "reason": reason_final
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

    def _calculate_transaction_costs(self, shares_traded: np.ndarray, prices: np.ndarray,
                                     abs_step: int, asset_mask: Optional[np.ndarray] = None) -> float:
        commission_bps = float(self.config['environment'].get('commission_bps', 0.00005))
        half_spread_bps = float(self.config['environment'].get('half_spread_bps', 0.0005))
        impact_coeff = float(self.config['environment'].get('impact_coeff', 0.1))
        adv_window = int(self.config['environment'].get('adv_window', 20))

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
        fixed_fee = float(self.config['environment'].get('fixed_fee_per_order', 0.0))
        num_orders = int(np.sum(traded_notional > 0))
        fixed_cost = fixed_fee * num_orders

        return float(commission_cost + spread_cost + impact_cost + fixed_cost)
    
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
        Generate single-step observation for agents like RecurrentPPO that do not require sequences.

        Returns:
            Flattened observation array optimized for SB3 LSTM processing
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

        # Get features for current step: shape [num_assets, num_selected_features]
        features = self.market_data_cache.get_features_at_step(self.current_absolute_step)

        # Get portfolio features for current step: shape [num_portfolio_features]
        # Use the last recorded external step from EpisodeBuffer for portfolio features
        # We recorded with external_step = self.current_step - 1
        external_step_for_obs = self.current_step - 1
        portfolio_features = self.episode_buffer.get_observation_at_step(external_step_for_obs)

        # Concatenate all components
        observation = np.concatenate([
            features.flatten(),
            portfolio_features
        ]).astype(np.float32)

        return observation
