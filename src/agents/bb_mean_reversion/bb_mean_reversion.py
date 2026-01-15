"""
Bollinger Band Mean Reversion Strategy + Parameter Regression / Optimization

Idea:
Buy when price is deeply below lower band (bb_position very negative), RSI oversold, a short-term
capitulation move (return_3d) relative to volatility occurred, and volume is not signaling a breakdown
(avoid elevated volume_percentile_20d). Sell when snapback conditions occur: RSI normalizes, price
re-enters bands (bb_position rises), or a strong positive return hits (return_1d > exit_snapback_ret_multiplier * volatility_20d).
Risk controls: volatility-normalized stop-loss and time-based exit.

Parameterized Conditions (Long):
Entry if simultaneously:
  bb_position < bb_long_thresh
  rsi_14 < rsi_long_thresh
  return_3d <= capitulation_mult * volatility_20d
  volume_percentile_20d < volume_ceiling_pct

Exit Long when ANY:
  rsi_14 >= exit_rsi_level
  bb_position >= exit_bb_position_level
  return_1d >= exit_snapback_ret_mult * volatility_20d
  stop-loss: (current_price - entry_price)/entry_price <= -stop_mult * volatility_20d_at_entry
  time stop: holding_days >= time_stop_days

Position Size:
  signal_strength = clip01( |bb_position| / |bb_long_thresh| ) * clip01( (rsi_long_thresh - rsi_14)/rsi_long_thresh )
  size = min(max_position_pct, position_size_scale * signal_strength / max(volatility_20d, vol_floor))
  (All sizes expressed as fraction of capital in that asset.)

Regression / Optimization:
1. Sample multiple training episodes (using MarketDataCache train_blocks).
2. Generate trades for many candidate parameter sets OR start from an initial set.
3. Collect trade-level features and realized returns.
4. Fit OLS: trade_return ~ f(entry_features, parameters_as_numeric_flags).
5. Use regression to propose refined thresholds (e.g. choose bb_long_thresh near quantile where predicted return turns positive).
6. Local grid search around regression-suggested thresholds.
7. Evaluate on validation episodes, report:
   - average trade return
   - hit rate
   - average holding period
   - cumulative episode return (simple sum of per-trade returns * size)

Note:
Uses precomputed features from the CSV dataset (do NOT recompute indicators).
No environment stepping required; direct slice-based backtest per episode.

Dependencies: numpy, pandas. Pure OLS implemented (no sklearn requirement).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from src.environment.trading_env import MarketDataCache

# ----------------------------
# Parameter Dataclass
# ----------------------------
@dataclass
class MeanRevParams:
    bb_long_thresh: float              # e.g. -1.5 (bb_position scaled; negative deep below lower band)
    rsi_long_thresh: float             # e.g. 30.0
    capitulation_mult: float           # multiplier: return_3d <= capitulation_mult * volatility_20d
    volume_ceiling_pct: float          # e.g. 0.70 (70 percentile max)
    exit_rsi_level: float              # e.g. 55.0
    exit_bb_position_level: float      # e.g. -0.2 (back inside bands zone)
    exit_snapback_ret_mult: float      # e.g. 0.8 (return_1d > 0.8 * vol_20d)
    position_size_scale: float         # scaling factor for sizing formula
    max_position_pct: float            # hard cap per asset (fraction of capital)
    stop_mult: float                   # adverse move multiplier * vol_20d at entry
    time_stop_days: int                # max holding days
    vol_floor: float = 0.01            # avoid division by very low volatility

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)


# ----------------------------
# Strategy Core
# ----------------------------
class BollingerMeanReversionStrategy:
    def __init__(self, market_data_cache: MarketDataCache, params: MeanRevParams, initial_capital: float = 1_000_000.0):
        self.mdc = market_data_cache
        self.params = params
        self.initial_capital = initial_capital
        self.asset_names = market_data_cache.asset_names
        # Feature indices (ensure presence)
        self.required_cols = [
            "bb_position", "rsi_14", "return_3d", "volatility_20d",
            "volume_percentile_20d", "return_1d"
        ]

    # ------------------------
    # Episode Extraction
    # ------------------------
    def _extract_episode_slice(self, start_abs_idx: int, episode_length: int) -> np.ndarray:
        end = min(start_abs_idx + episode_length, self.mdc.num_days)
        return np.arange(start_abs_idx, end, dtype=int)

    # ------------------------
    # Trade Simulation (Long Only Mean Reversion)
    # ------------------------
    def simulate_episode(self, episode_indices: np.ndarray) -> List[Dict[str, Any]]:
        trades: List[Dict[str, Any]] = []
        p = self.params
        holding = np.zeros(self.mdc.num_assets, dtype=bool)
        entry_price = np.zeros(self.mdc.num_assets, dtype=np.float32)
        entry_vol = np.zeros(self.mdc.num_assets, dtype=np.float32)
        entry_day_idx = np.zeros(self.mdc.num_assets, dtype=int)
        position_size = np.zeros(self.mdc.num_assets, dtype=np.float32)

        for offset, abs_idx in enumerate(episode_indices):
            # Feature arrays at abs_idx
            bb_pos = self.mdc.features[abs_idx, :, self._feat_idx("bb_position")]
            rsi = self.mdc.features[abs_idx, :, self._feat_idx("rsi_14")]
            ret3 = self.mdc.features[abs_idx, :, self._feat_idx("return_3d")]
            vol20 = self.mdc.features[abs_idx, :, self._feat_idx("volatility_20d")]
            volpct20 = self.mdc.features[abs_idx, :, self._feat_idx("volume_percentile_20d")]
            ret1 = self.mdc.features[abs_idx, :, self._feat_idx("return_1d")]
            close_prices = self.mdc.close_prices[abs_idx]

            # Entry logic
            can_enter = (~holding &
                         (bb_pos < p.bb_long_thresh) &
                         (rsi < p.rsi_long_thresh) &
                         (ret3 <= p.capitulation_mult * vol20) &
                         (volpct20 < p.volume_ceiling_pct))

            if np.any(can_enter):
                # Position sizing
                signal_strength = np.clip(np.abs(bb_pos[can_enter]) / max(1e-8, abs(p.bb_long_thresh)), 0, 1) * \
                                  np.clip((p.rsi_long_thresh - rsi[can_enter]) / max(p.rsi_long_thresh, 1e-8), 0, 1)
                raw_size = p.position_size_scale * signal_strength / np.maximum(vol20[can_enter], p.vol_floor)
                sized = np.minimum(raw_size, p.max_position_pct)
                # Record entries
                for local_i, asset_idx in enumerate(np.where(can_enter)[0]):
                    holding[asset_idx] = True
                    entry_price[asset_idx] = close_prices[asset_idx]
                    entry_vol[asset_idx] = vol20[asset_idx]
                    entry_day_idx[asset_idx] = offset
                    position_size[asset_idx] = sized[local_i]
                    trades.append({
                        "type": "entry",
                        "day_offset": offset,
                        "abs_idx": abs_idx,
                        "asset": self.asset_names[asset_idx],
                        "entry_price": entry_price[asset_idx],
                        "size": position_size[asset_idx],
                        "bb_position": bb_pos[asset_idx],
                        "rsi_14": rsi[asset_idx],
                        "ret3": ret3[asset_idx],
                        "vol20": vol20[asset_idx],
                        "volume_percentile_20d": volpct20[asset_idx],
                        "signal_strength": signal_strength[local_i]
                    })

            # Exit logic
            if np.any(holding):
                held_assets = np.where(holding)[0]
                # Conditions for exit
                exit_rsi = rsi[held_assets] >= p.exit_rsi_level
                exit_bb = bb_pos[held_assets] >= p.exit_bb_position_level
                exit_snap = ret1[held_assets] >= p.exit_snapback_ret_mult * vol20[held_assets]
                # Stop-loss
                adverse_move = (close_prices[held_assets] / entry_price[held_assets] - 1.0) <= -p.stop_mult * entry_vol[held_assets]
                # Time stop
                age = offset - entry_day_idx[held_assets]
                time_stop = age >= p.time_stop_days

                should_exit = exit_rsi | exit_bb | exit_snap | adverse_move | time_stop

                if np.any(should_exit):
                    for asset_idx in held_assets[should_exit]:
                        exit_price = close_prices[asset_idx]
                        ret = exit_price / entry_price[asset_idx] - 1.0
                        trades.append({
                            "type": "exit",
                            "day_offset": offset,
                            "abs_idx": abs_idx,
                            "asset": self.asset_names[asset_idx],
                            "exit_price": exit_price,
                            "return": ret,
                            "holding_period": offset - entry_day_idx[asset_idx],
                            "size": position_size[asset_idx],
                            "exit_rsi_cond": bool(rsi[asset_idx] >= p.exit_rsi_level),
                            "exit_bb_cond": bool(bb_pos[asset_idx] >= p.exit_bb_position_level),
                            "exit_snap_cond": bool(ret1[asset_idx] >= p.exit_snapback_ret_mult * vol20[asset_idx]),
                            "stop_cond": bool((exit_price / entry_price[asset_idx] - 1.0) <= -p.stop_mult * entry_vol[asset_idx]),
                            "time_stop_cond": bool((offset - entry_day_idx[asset_idx]) >= p.time_stop_days)
                        })
                        # Reset holding
                        holding[asset_idx] = False
                        position_size[asset_idx] = 0.0

        return trades

    def _feat_idx(self, name: str) -> int:
        # Map feature name to index using MarketDataCache.feature_to_index
        return self.mdc.feature_to_index[name]

    # ------------------------
    # OLS Regression on Trade Outcomes
    # ------------------------
    @staticmethod
    def fit_ols(trade_df: pd.DataFrame, y_col: str = "return") -> Dict[str, Any]:
        if trade_df.empty:
            return {"coefficients": {}, "intercept": 0.0}
        X_cols = ["bb_position", "rsi_14", "ret3", "vol20", "volume_percentile_20d", "signal_strength"]
        X = trade_df[X_cols].values
        y = trade_df[y_col].values
        # Add intercept
        X_design = np.column_stack([np.ones(len(X)), X])
        # OLS (X'X)^(-1) X'y
        XtX = X_design.T @ X_design
        try:
            inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(XtX)
        coeff_vec = inv @ X_design.T @ y
        intercept = coeff_vec[0]
        coefs = dict(zip(X_cols, coeff_vec[1:]))
        return {"coefficients": coefs, "intercept": intercept}

    # ------------------------
    # Parameter Proposal from Regression
    # ------------------------
    @staticmethod
    def propose_parameters(regression: Dict[str, Any], current: MeanRevParams) -> MeanRevParams:
        coefs = regression["coefficients"]
        # Heuristic adjustments:
        bb_coef = coefs.get("bb_position", 0.0)
        rsi_coef = coefs.get("rsi_14", 0.0)
        vol_coef = coefs.get("vol20", 0.0)

        # If more negative bb_position yields better return (bb_coef < 0), allow looser threshold (more negative)
        bb_long_thresh = current.bb_long_thresh + (-0.25 if bb_coef < 0 else 0.10)
        # If lower RSI yields better return (rsi_coef < 0) keep or lower threshold slightly
        rsi_long_thresh = current.rsi_long_thresh + (-2.0 if rsi_coef < 0 else 1.0)
        # Volatility penalizes returns if vol_coef < 0 -> tighten capitulation multiple
        capitulation_mult = current.capitulation_mult + (-0.05 if vol_coef < 0 else 0.05)

        # Clamp sensible bounds
        bb_long_thresh = np.clip(bb_long_thresh, -3.0, -0.5)
        rsi_long_thresh = np.clip(rsi_long_thresh, 15.0, 40.0)
        capitulation_mult = np.clip(capitulation_mult, -2.0, 2.0)

        return MeanRevParams(
            bb_long_thresh=bb_long_thresh,
            rsi_long_thresh=rsi_long_thresh,
            capitulation_mult=capitulation_mult,
            volume_ceiling_pct=current.volume_ceiling_pct,
            exit_rsi_level=current.exit_rsi_level,
            exit_bb_position_level=current.exit_bb_position_level,
            exit_snapback_ret_mult=current.exit_snapback_ret_mult,
            position_size_scale=current.position_size_scale,
            max_position_pct=current.max_position_pct,
            stop_mult=current.stop_mult,
            time_stop_days=current.time_stop_days,
            vol_floor=current.vol_floor
        )

    # ------------------------
    # Grid Refinement Around Proposed Params
    # ------------------------
    def grid_refine(self, base_params: MeanRevParams, train_episode_indices: List[np.ndarray], ref_steps: int = 10) -> MeanRevParams:
        best_params = base_params
        best_score = -1e9
        # Small grids
        bb_grid = np.linspace(base_params.bb_long_thresh - 0.3, base_params.bb_long_thresh + 0.3, 5)
        rsi_grid = np.linspace(base_params.rsi_long_thresh - 3, base_params.rsi_long_thresh + 3, 5)
        cap_grid = np.linspace(base_params.capitulation_mult - 0.1, base_params.capitulation_mult + 0.1, 5)

        for bb in bb_grid:
            for rsi in rsi_grid:
                for cap in cap_grid:
                    test_params = MeanRevParams(
                        bb_long_thresh=float(np.clip(bb, -3.5, -0.4)),
                        rsi_long_thresh=float(np.clip(rsi, 10.0, 45.0)),
                        capitulation_mult=float(np.clip(cap, -2.5, 2.5)),
                        volume_ceiling_pct=base_params.volume_ceiling_pct,
                        exit_rsi_level=base_params.exit_rsi_level,
                        exit_bb_position_level=base_params.exit_bb_position_level,
                        exit_snapback_ret_mult=base_params.exit_snapback_ret_mult,
                        position_size_scale=base_params.position_size_scale,
                        max_position_pct=base_params.max_position_pct,
                        stop_mult=base_params.stop_mult,
                        time_stop_days=base_params.time_stop_days,
                        vol_floor=base_params.vol_floor
                    )
                    score = self.evaluate_params(train_episode_indices, test_params)
                    if score > best_score:
                        best_score = score
                        best_params = test_params
        return best_params

    # ------------------------
    # Evaluation Metric (Train or Validation)
    # ------------------------
    def evaluate_params(self, episode_indices_list: List[np.ndarray], params: MeanRevParams) -> float:
        strat = BollingerMeanReversionStrategy(self.mdc, params, self.initial_capital)
        all_returns = []
        for inds in episode_indices_list:
            trades = strat.simulate_episode(inds)
            trade_pairs = self._pair_trades(trades)
            for pair in trade_pairs:
                all_returns.append(pair["return"] * pair["size"])
        if not all_returns:
            return -1e6
        avg = float(np.mean(all_returns))
        # Penalize too few trades
        penalty = 0.0 if len(all_returns) >= 5 else -0.05
        return avg + penalty

    @staticmethod
    def _pair_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pairs = []
        stack = {}
        for t in trades:
            if t["type"] == "entry":
                stack.setdefault(t["asset"], []).append(t)
            elif t["type"] == "exit":
                if t["asset"] in stack and stack[t["asset"]]:
                    e = stack[t["asset"]].pop(0)
                    pairs.append({
                        "asset": t["asset"],
                        "entry_day": e["day_offset"],
                        "exit_day": t["day_offset"],
                        "holding_period": t["day_offset"] - e["day_offset"],
                        "return": t["return"],
                        "size": e["size"],
                        "bb_position_entry": e["bb_position"],
                        "rsi_entry": e["rsi_14"],
                        "signal_strength": e["signal_strength"]
                    })
        return pairs

# ----------------------------
# Optimization Routine
# ----------------------------
def optimize_parameters(market_data_cache: MarketDataCache,
                        initial_params: MeanRevParams,
                        episode_length_days: int,
                        train_samples: int = 15,
                        val_samples: int = 8,
                        random_seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(random_seed)

    def sample_blocks(blocks, n):
        chosen = []
        if not blocks:
            return chosen
        idxs = rng.choice(len(blocks), size=min(n, len(blocks)), replace=False)
        for i in idxs:
            block = blocks[i]
            # Random start inside allowed range
            start = rng.integers(block.min_start_step, max(block.min_start_step + 1, block.max_start_step + 1))
            episode_idx = np.arange(start, min(start + episode_length_days, market_data_cache.num_days))
            chosen.append(episode_idx)
        return chosen

    train_episode_indices = sample_blocks(market_data_cache.train_blocks, train_samples)
    val_episode_indices = sample_blocks(market_data_cache.validation_blocks, val_samples)

    # Initial simulation & regression
    strat = BollingerMeanReversionStrategy(market_data_cache, initial_params)
    trade_rows = []
    for ep_inds in train_episode_indices:
        trades = strat.simulate_episode(ep_inds)
        pairs = strat._pair_trades(trades)
        trade_rows.extend(pairs)
    trade_df = pd.DataFrame(trade_rows)
    regression = strat.fit_ols(trade_df) if not trade_df.empty else {"coefficients": {}, "intercept": 0.0}

    # Propose new params
    proposed = BollingerMeanReversionStrategy.propose_parameters(regression, initial_params)

    # Grid refine around proposed
    refined = strat.grid_refine(proposed, train_episode_indices)

    # Evaluate final on validation
    final_strat = BollingerMeanReversionStrategy(market_data_cache, refined)
    val_returns = []
    val_trades_total = 0
    for ep_inds in val_episode_indices:
        trades = final_strat.simulate_episode(ep_inds)
        pairs = final_strat._pair_trades(trades)
        val_trades_total += len(pairs)
        for pair in pairs:
            val_returns.append(pair["return"] * pair["size"])
    val_avg = float(np.mean(val_returns)) if val_returns else 0.0
    val_median = float(np.median(val_returns)) if val_returns else 0.0
    hit_rate = float(np.mean([r > 0 for r in val_returns])) if val_returns else 0.0

    return {
        "initial_params": initial_params.as_dict(),
        "regression": regression,
        "proposed_params": proposed.as_dict(),
        "refined_params": refined.as_dict(),
        "validation_avg_return": val_avg,
        "validation_median_return": val_median,
        "validation_hit_rate": hit_rate,
        "validation_trade_count": val_trades_total
    }

# ----------------------------
# Convenience Runner
# ----------------------------
def run_mean_reversion_parameter_search(market_data_cache: MarketDataCache,
                                        config: Dict[str, Any]) -> Dict[str, Any]:
    episode_len = config["environment"]["episode_length_days"]
    init = MeanRevParams(
        bb_long_thresh=-1.5,
        rsi_long_thresh=32.0,
        capitulation_mult=-0.5,
        volume_ceiling_pct=0.70,
        exit_rsi_level=55.0,
        exit_bb_position_level=-0.15,
        exit_snapback_ret_mult=0.8,
        position_size_scale=0.15,
        max_position_pct=0.10,
        stop_mult=1.2,
        time_stop_days=6,
        vol_floor=0.01
    )
    result = optimize_parameters(market_data_cache, init, episode_len)
    return result

