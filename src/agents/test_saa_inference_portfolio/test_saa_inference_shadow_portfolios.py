"""
test_saa_inference_shadow_portfolios.py
========================================

Inference test for trained SAA (Single-Asset Agent) models using the
*shadow portfolio* approach. Each of the 11 assets runs in its own completely
isolated TradingEnv instance configured in EXECUTION_SINGLE_ASSET_TARGET_POS
mode, with its own dedicated cash pool.  No shared cash — no domain-shift from
inter-asset interaction.

Workflow
--------
1. Load a single trained SAA checkpoint + sibling VecNormalize stats.
2. Deep-copy the model once per asset so each maintains independent LSTM state.
3. Build a deterministic validation-episode plan that covers every validation
   block in full.
4. For every (episode, asset) pair: reset a fresh TradingEnv, run inference,
   read results from env.episode_buffer.
5. Produce per-time-period reports:
   - Two A4-sized PNG pages per episode (6 + 5 asset subplots in 2-column layout).
   - One aggregate PNG per episode (combined agent portfolio vs SPY B&H).
   - asset-specific JSON  and  aggregate JSON  (one file each, covering all
     episodes).
6. Open all saved PNGs with the OS default viewer.

See README_test_saa_inference.md for the full specification.
"""

# ================================
# Imports
# ================================
import os
import sys
import copy
import json
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Force line-buffered stdout so progress prints surface immediately.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # non-interactive; safe for CLI/server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from environment.trading_environment import (
    TradingEnv,
    MarketDataCache,
)


# ================================
# Matplotlib academic style
# ================================
def _apply_paper_style() -> None:
    """Set rcParams once for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "figure.dpi": 100,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })


# ================================
# VecNormalize utilities
# ================================
class _ObsNormDummyEnv(gym.Env):
    """Minimal env so VecNormalize.load() can attach to a space without running."""
    metadata: Dict[str, Any] = {}

    def __init__(self, observation_space: gym.Space) -> None:
        self.observation_space = observation_space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):  # never called
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}


def _normalize_obs(obs: np.ndarray, vecnorm: Optional[VecNormalize]) -> np.ndarray:
    """Apply saved VecNormalize obs_rms to a (1, obs_dim) array. Returns float32."""
    if vecnorm is None or getattr(vecnorm, "obs_rms", None) is None:
        return np.asarray(obs, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    mean = vecnorm.obs_rms.mean
    var = vecnorm.obs_rms.var
    epsilon = float(getattr(vecnorm, "epsilon", getattr(vecnorm, "eps", 1e-8)))
    clip_obs = float(getattr(vecnorm, "clip_obs", 10.0))
    return np.clip((obs - mean) / np.sqrt(var + epsilon), -clip_obs, clip_obs).astype(np.float32)


# ================================
# SAA model loader
# ================================
def _load_saa_model(
    model_zip_path: str,
    device: torch.device,
) -> Tuple[RecurrentPPO, Optional[VecNormalize]]:
    """Load RecurrentPPO checkpoint and its sibling _vecnormalize.pkl.

    Naming contract:
        <stem>.zip                 → RecurrentPPO checkpoint
        <stem>_vecnormalize.pkl    → VecNormalize stats
    """
    if not model_zip_path.lower().endswith(".zip"):
        raise ValueError(f"saa_model_run_dir must point at a .zip file, got: {model_zip_path!r}")
    if not os.path.isfile(model_zip_path):
        raise FileNotFoundError(f"SAA checkpoint not found: {model_zip_path}")

    vecnorm_path = model_zip_path[:-4] + "_vecnormalize.pkl"

    # Override callable schedules to avoid deserialization errors on inference-only loads.
    safe_custom_objects = {
        "learning_rate": 3e-5,
        "lr_schedule": lambda _p: 3e-5,
        "clip_range": lambda _p: 0.2,
        "clip_range_vf": None,
    }

    print(f"[Shadow-Test] Loading RecurrentPPO from: {model_zip_path}", flush=True)
    model = RecurrentPPO.load(model_zip_path, device=device, custom_objects=safe_custom_objects)
    print(
        f"[Shadow-Test] Model loaded. obs_dim={model.observation_space.shape[0]} "
        f"action_dim={model.action_space.shape[0]} device={device}",
        flush=True,
    )

    vecnorm: Optional[VecNormalize] = None
    if os.path.isfile(vecnorm_path):
        print(f"[Shadow-Test] Loading VecNormalize from: {vecnorm_path}", flush=True)
        dummy = DummyVecEnv([lambda: _ObsNormDummyEnv(model.observation_space)])
        vecnorm = VecNormalize.load(vecnorm_path, dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False
        print("[Shadow-Test] VecNormalize loaded (training=False, norm_reward=False).", flush=True)
    else:
        print(f"[Shadow-Test] No VecNormalize at {vecnorm_path}; using raw observations.", flush=True)

    return model, vecnorm


# ================================
# Validation episode plan
# ================================
def _build_validation_episode_plan(
    cache: MarketDataCache,
) -> List[Tuple[str, int, int]]:
    """Build a deterministic plan that covers all validation blocks in full.

    Returns:
        List of (block_id, absolute_start_step, episode_length_days) tuples.
    """
    plans: List[Tuple[str, int, int]] = []
    base_len = int(cache.episode_length_days)
    blocks = sorted(cache.validation_blocks, key=lambda b: b.start_date_idx)

    for b in blocks:
        n = max(1, int(b.max_episodes))
        for k in range(n):
            start = int(b.min_start_step + k * base_len)
            if start >= int(b.end_date_idx):
                continue
            # Last segment extends to block end (may be longer than base_len).
            end = int(b.end_date_idx) if k == n - 1 else min(int(b.end_date_idx), start + base_len)
            if end <= start:
                continue
            plans.append((str(b.block_id), start, int(end - start)))

    return plans


# ================================
# Single-asset episode runner
# ================================
def _run_asset_episode(
    cache: MarketDataCache,
    config: Dict[str, Any],
    model: RecurrentPPO,
    vecnorm: Optional[VecNormalize],
    asset_name: str,
    action_limiting_factor: float,
    deterministic: bool,
    device: torch.device,
    block_id: str,
    episode_start_step: int,
    episode_length: int,
    seed: int,
) -> Dict[str, Any]:
    """Run one validation episode for a single asset in its shadow portfolio.

    The TradingEnv is configured in EXECUTION_SINGLE_ASSET_TARGET_POS mode so
    the asset operates with its own isolated cash pool. Always starts fully in
    cash (percentage_of_cash_only_starts = 1.0).

    Returns a dict containing all per-step arrays needed for metrics and graphs.
    """
    # Build episode-specific env config.
    env_cfg = copy.deepcopy(config)
    env_cfg["environment"]["execution_mode"] = "single_asset_target_position"
    env_cfg["environment"]["percentage_of_cash_only_starts"] = 1.0
    env_cfg["environment"]["episode_length_days"] = int(episode_length)

    env = TradingEnv(env_cfg, cache, mode="validation")

    obs, _reset_info = env.reset(
        seed=seed,
        option={"block_id": block_id, "episode_start_step": episode_start_step},
        asset=asset_name,
    )

    asset_idx = cache.asset_to_index[asset_name]
    initial_pv = float(config["environment"]["initial_portfolio_value"])

    # !! NAMING CAUTION — DO NOT CONFUSE THESE TWO CONCEPTS !!
    # TradingEnv.shadow_portfolio_state is a FRICTIONLESS mirror of the agent's live trades
    # (same positions, zero transaction costs). It is UNRELATED to the "shadow portfolio"
    # isolation technique used in this test script, where each SAA agent runs in its own
    # TradingEnv with its own dedicated cash pool and only sees its own asset's observations.
    # We do NOT use env.shadow_portfolio_state here. The correct TC-aware buy-and-hold
    # baseline is env.episode_buffer.selected_asset_bh_portfolio_value, populated by
    # TradingEnv.reset() via _initialize_portfolio_with_costs() and updated each step.

    lstm_state = None
    episode_start_flag = np.ones((1, 1), dtype=bool)
    terminated = truncated = False
    step_info: Dict[str, Any] = {}

    while not (terminated or truncated):
        norm_obs = _normalize_obs(obs[np.newaxis], vecnorm)  # (1, obs_dim)
        with torch.no_grad():
            action_out, lstm_state = model.predict(
                norm_obs,
                state=lstm_state,
                episode_start=episode_start_flag,
                deterministic=deterministic,
            )
        episode_start_flag[:] = False

        raw_action = float(np.clip(action_out[0, 0], -1.0, 1.0))
        scaled_action = raw_action * action_limiting_factor

        obs, _reward, terminated, truncated, step_info = env.step(
            np.array([scaled_action], dtype=np.float32),
            asset=asset_name,
        )
    # step_info from the terminating step contains episode-level diagnostics (costs etc.).
    terminal_info: Dict[str, Any] = step_info if isinstance(step_info, dict) else {}

    # ---- Extract from episode buffer ----
    # env.current_step == number of step() calls made; buffer slots [0..current_step] are filled.
    T = env.current_step + 1
    buf = env.episode_buffer

    portfolio_values = buf.portfolio_values[:T].copy().astype(np.float64)
    prices = buf.asset_prices[:T, asset_idx].copy().astype(np.float64)
    saa_returns = buf.saa_returns[:T].copy().astype(np.float64)
    transaction_costs = buf.transaction_costs[:T].copy().astype(np.float64)
    rewards = buf.rewards[:T].copy().astype(np.float64)
    # actions[t, asset_idx+1] holds the scaled target_position_change passed to env.step().
    actions = buf.actions[:T, asset_idx + 1].copy().astype(np.float64)
    positions = buf.portfolio_positions[:T, asset_idx].copy().astype(np.float64)
    # traded_dollar_volume is shape (T, num_assets); extract only the selected asset column.
    # Values are unsigned magnitude; sign comes from the action direction.
    traded_dollar_volume = buf.traded_dollar_volume[:T, asset_idx].copy().astype(np.float64)
    signed_volume = np.sign(actions) * traded_dollar_volume

    # Buy-and-hold equity curve: read from env.episode_buffer.selected_asset_bh_portfolio_value.
    # Populated by TradingEnv.reset() via _initialize_portfolio_with_costs() (TC-aware init),
    # then updated each step by price mark-to-market only — no trades after initialization.
    # Per SAA_INFERENCE_TESTING_GUIDE: selected_asset_bh_transaction_costs[0] = one-time init
    # cost; all subsequent steps are 0.0.
    bh_values = buf.selected_asset_bh_portfolio_value[:T].copy().astype(np.float64)
    bh_init_tc = float(buf.selected_asset_bh_transaction_costs[0])

    # Calendar dates for this episode.
    dates: List[str] = []
    for t in range(T):
        abs_idx = episode_start_step + t
        dates.append(str(cache.dates[min(abs_idx, cache.num_days - 1)]))

    return {
        "asset_name": asset_name,
        "asset_idx": asset_idx,
        "dates": dates,
        "portfolio_values": portfolio_values,
        "prices": prices,
        "bh_values": bh_values,
        "saa_returns": saa_returns,
        "transaction_costs": transaction_costs,
        "rewards": rewards,
        "actions": actions,
        "signed_volume": signed_volume,
        "positions": positions,
        "initial_pv": initial_pv,
        "episode_start_step": episode_start_step,
        "episode_length": episode_length,
        "block_id": block_id,
        "terminal_info": terminal_info,
        "bh_init_tc": bh_init_tc,
    }


# ================================
# Finance metric helpers
# ================================
def _annual_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.27 = 27% drawdown)."""
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    mask = running_max > 0
    if not mask.any():
        return 0.0
    dd = np.where(mask, (running_max - eq) / running_max, 0.0)
    return float(dd.max())


def _curve_metrics(equity: np.ndarray) -> Dict[str, float]:
    """Compute standard financial metrics for an equity curve."""
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size < 2 or eq[0] <= 0:
        return {"total_return": 0.0, "annual_sharpe": 0.0, "max_drawdown": 0.0, "final_value": float(eq[-1]) if eq.size else 0.0}
    daily_rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1.0)
    return {
        "total_return": float(eq[-1] / eq[0] - 1.0),
        "annual_sharpe": _annual_sharpe(daily_rets),
        "max_drawdown": _max_drawdown(eq),
        "final_value": float(eq[-1]),
    }


# ================================
# JSON metrics builder
# ================================
def _build_asset_json_metrics(
    asset_record: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute all required JSON fields for one asset episode record."""
    pv = asset_record["portfolio_values"]
    bh = asset_record["bh_values"]
    rewards = asset_record["rewards"]
    costs = asset_record["transaction_costs"]
    positions = asset_record["positions"]
    prices = asset_record["prices"]
    signed_volume = asset_record["signed_volume"]
    actions = asset_record["actions"]
    dates = asset_record["dates"]
    initial_pv = asset_record["initial_pv"]

    trades_executed = int(np.sum(np.abs(signed_volume) > 0))

    terminal_info = asset_record.get("terminal_info", {})
    traded_notional = float(terminal_info.get("episode_traded_notional", 0.0))
    total_cost_usd = float(costs.sum())
    cost_bps = total_cost_usd / traded_notional * 10000 if traded_notional > 0 else 0.0

    metrics = {
        "asset_name": asset_record["asset_name"],
        "initial_portfolio_value": float(initial_pv),
        "min_portfolio_value": float(pv.min()),
        "max_portfolio_value": float(pv.max()),
        "final_portfolio_value": float(pv[-1]),
        "total_return_pct": float((pv[-1] / pv[0] - 1.0) * 100) if pv[0] > 0 else 0.0,
        "bh_total_return_pct": float((bh[-1] / bh[0] - 1.0) * 100) if bh[0] > 0 else 0.0,
        "alpha_vs_bh_pct": float(((pv[-1] / pv[0]) - (bh[-1] / bh[0])) * 100) if pv[0] > 0 and bh[0] > 0 else 0.0,
        "sharpe_ratio": _annual_sharpe(np.diff(pv) / np.where(pv[:-1] > 0, pv[:-1], 1.0)),
        "max_drawdown_pct": float(_max_drawdown(pv) * 100),
        "average_reward_per_step": float(rewards[1:].mean()) if len(rewards) > 1 else 0.0,
        "cumulative_transaction_costs": float(costs.sum()),
        "num_trades_executed": trades_executed,
        # Transaction cost breakdown from terminal episode info.
        "episode_traded_notional_usd": traded_notional,
        "cost_total_usd": total_cost_usd,
        "cost_bps": cost_bps,
        "cost_commission_usd": float(terminal_info.get("episode_cost_commission", 0.0)),
        "cost_spread_usd": float(terminal_info.get("episode_cost_spread", 0.0)),
        "cost_impact_usd": float(terminal_info.get("episode_cost_impact", 0.0)),
        "cost_fixed_usd": float(terminal_info.get("episode_cost_fixed", 0.0)),
        # Buy-and-hold reference: one-time init cost at episode start.
        "bh_init_transaction_cost_usd": float(asset_record.get("bh_init_tc", 0.0)),
        # Full time-series for graph reconstruction
        "dates": dates,
        "portfolio_values": pv.tolist(),
        "bh_values": bh.tolist(),
        "prices": prices.tolist(),
        "actions": actions.tolist(),
        "signed_volume_usd": signed_volume.tolist(),
        "positions_shares": positions.tolist(),
    }
    return metrics


def _build_aggregate_json_metrics(
    episode_record: Dict[str, Any],
    asset_names: List[str],
    cache: MarketDataCache,
) -> Dict[str, Any]:
    """Compute aggregate combined-portfolio metrics for one episode."""
    # Combined portfolio = equal-weight average of all shadow sub-portfolios.
    # Each agent starts at initial_pv; combined value normalised so combined starts at initial_pv.
    asset_pvs = np.stack(
        [episode_record["assets"][a]["portfolio_values"] for a in asset_names], axis=1
    )  # (T, N)
    combined_pv = asset_pvs.mean(axis=1)  # (T,) — starts at initial_pv

    asset_bhs = np.stack(
        [episode_record["assets"][a]["bh_values"] for a in asset_names], axis=1
    )  # (T, N)
    uniform_bh_pv = asset_bhs.mean(axis=1)  # equal-weight B&H baseline

    # SPY-only B&H
    spy_idx = cache.asset_to_index.get("SPY", None)
    if spy_idx is not None:
        # Use the SPY data from the first asset record that is SPY, or reconstruct.
        spy_bh = episode_record["assets"].get("SPY", {}).get("bh_values", None)
        if spy_bh is None:
            spy_bh = np.full(combined_pv.shape, np.nan)
        spy_bh = np.asarray(spy_bh, dtype=np.float64)
    else:
        spy_bh = np.full(combined_pv.shape, np.nan)

    dates = episode_record["assets"][asset_names[0]]["dates"]
    initial_pv = episode_record["assets"][asset_names[0]]["initial_pv"]

    combined_metrics = _curve_metrics(combined_pv)
    bh_metrics = _curve_metrics(uniform_bh_pv)
    spy_metrics = _curve_metrics(spy_bh) if not np.all(np.isnan(spy_bh)) else None

    total_costs = float(
        sum(episode_record["assets"][a]["transaction_costs"].sum() for a in asset_names)
    )

    return {
        "episode_id": int(episode_record["episode_id"]),
        "block_id": str(episode_record["block_id"]),
        "date_range": f"{dates[0]} to {dates[-1]}",
        "initial_portfolio_value": float(initial_pv),
        "combined_portfolio": {
            **combined_metrics,
            "total_return_pct": float(combined_metrics["total_return"] * 100),
            "max_drawdown_pct": float(combined_metrics["max_drawdown"] * 100),
        },
        "uniform_bh_portfolio": {
            **bh_metrics,
            "total_return_pct": float(bh_metrics["total_return"] * 100),
            "max_drawdown_pct": float(bh_metrics["max_drawdown"] * 100),
        },
        "spy_bh": (
            {
                **spy_metrics,
                "total_return_pct": float(spy_metrics["total_return"] * 100),
                "max_drawdown_pct": float(spy_metrics["max_drawdown"] * 100),
            }
            if spy_metrics is not None
            else None
        ),
        "total_transaction_costs_all_assets": total_costs,
        # Time-series for aggregate graph
        "dates": dates,
        "combined_portfolio_values": combined_pv.tolist(),
        "uniform_bh_values": uniform_bh_pv.tolist(),
        "spy_bh_values": spy_bh.tolist() if spy_bh is not None else [],
    }


# ================================
# Visualization helpers
# ================================
def _index_to_100(values: np.ndarray) -> np.ndarray:
    """Normalise an equity curve to 100 at t=0 for fair comparison plots."""
    v = np.asarray(values, dtype=np.float64)
    if v[0] > 0:
        return v / v[0] * 100.0
    return v.copy()


def _plot_main_subplot(
    ax: plt.Axes,
    asset_name: str,
    portfolio_values: np.ndarray,
    bh_values: np.ndarray,
    prices: np.ndarray,
) -> None:
    """Plot normalised equity curves (price / agent / B&H) on a single primary axis."""
    T = len(portfolio_values)
    x = np.arange(T)

    pv_idx = _index_to_100(portfolio_values)
    bh_idx = _index_to_100(bh_values)
    price_idx = _index_to_100(prices)

    ax.plot(x, price_idx, color="#444444", linewidth=0.9, label="Price", zorder=3)
    ax.plot(x, pv_idx, color="#1f77b4", linewidth=1.4, label="Agent", zorder=4)
    ax.plot(x, bh_idx, color="#ff7f0e", linewidth=1.2, linestyle="--", label="B&H 100%", zorder=3)
    ax.set_ylabel("Index (t₀=100)", fontsize=7)
    ax.set_title(asset_name, fontsize=9, fontweight="bold", pad=3)
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)


def _plot_volume_subplot(
    ax: plt.Axes,
    signed_volume: np.ndarray,
) -> None:
    """Plot signed trading volume as a compact bar chart on a dedicated axis."""
    T = len(signed_volume)
    x = np.arange(T)

    buy_mask = signed_volume > 0
    sell_mask = signed_volume < 0
    if buy_mask.any():
        ax.bar(x[buy_mask], signed_volume[buy_mask], color="#2ca02c", alpha=0.75, width=1.0)
    if sell_mask.any():
        ax.bar(x[sell_mask], signed_volume[sell_mask], color="#d62728", alpha=0.75, width=1.0)

    ax.axhline(0.0, color="#888888", linewidth=0.5, zorder=2)
    ax.set_ylabel("Vol $", fontsize=6, color="#555555")
    ax.tick_params(axis="y", labelsize=5, labelcolor="#555555")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${abs(v):,.0f}"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    ax.spines["top"].set_visible(False)


def _plot_asset_specific_pages(
    episode_record: Dict[str, Any],
    asset_names: List[str],
    date_display: str,
    date_file: str,
    output_dir: str,
    timestamp: str,
) -> List[str]:
    """Produce two A4 PNG pages (6 + 5 subplots) for one episode.

    Page 1 covers assets 0–5 (6 plots, 3 rows × 2 cols).
    Page 2 covers assets 6–10 (5 plots + alpha/cost text in bottom-right).
    Each plot cell contains a main equity subplot and a compact volume subplot below,
    sharing the x-axis. Vertical gap between the two is approximately 1.5 cm.

    Returns list of saved file paths.
    """
    _apply_paper_style()

    A4_W, A4_H = 8.27, 11.69  # inches portrait

    # Per-asset alpha (agent total return – B&H total return) for the ranking text.
    alphas: Dict[str, float] = {}
    for a in asset_names:
        rec = episode_record["assets"][a]
        pv = rec["portfolio_values"]
        bh = rec["bh_values"]
        agent_ret = (pv[-1] / pv[0] - 1.0) if pv[0] > 0 else 0.0
        bh_ret = (bh[-1] / bh[0] - 1.0) if bh[0] > 0 else 0.0
        alphas[a] = float(agent_ret - bh_ret)

    saved_paths: List[str] = []

    def _make_page(page_assets: List[str], page_num: int, show_alpha_text: bool) -> str:
        nrows, ncols = 3, 2
        fig = plt.figure(figsize=(A4_W, A4_H))
        fig.suptitle(
            f"SAA Shadow Portfolio — Asset Analysis\n{date_display}",
            fontsize=11, fontweight="bold", y=0.99,
        )

        outer_gs = gridspec.GridSpec(
            nrows, ncols, figure=fig,
            hspace=0.42, wspace=0.42,
            top=0.93, bottom=0.04, left=0.09, right=0.95,
        )

        for plot_idx, asset_name in enumerate(page_assets):
            row = plot_idx // ncols
            col = plot_idx % ncols
            rec = episode_record["assets"][asset_name]

            # Inner GridSpec: main equity plot on top, compact volume strip below.
            # hspace ≈ 0.40 corresponds to ~1.5 cm at this figure size.
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[row, col],
                height_ratios=[4, 1], hspace=0.05,
            )
            ax_main = fig.add_subplot(inner_gs[0, 0])
            ax_vol = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)

            _plot_main_subplot(
                ax=ax_main,
                asset_name=asset_name,
                portfolio_values=rec["portfolio_values"],
                bh_values=rec["bh_values"],
                prices=rec["prices"],
            )
            _plot_volume_subplot(ax=ax_vol, signed_volume=rec["signed_volume"])

            # X-axis tick labels appear only on the volume subplot (shared axis).
            ax_main.tick_params(labelbottom=False)
            ax_vol.set_xlabel("Trading day", fontsize=7)

        # Bottom-right cell on the 5-asset page: alpha ranking + cost breakdown.
        if show_alpha_text:
            ax_text = fig.add_subplot(outer_gs[nrows - 1, ncols - 1])
            ax_text.axis("off")

            sorted_alphas = sorted(alphas.items(), key=lambda kv: kv[1], reverse=True)
            lines: List[str] = ["Alpha (Agent – B&H)", "─" * 24]
            for rank, (name, alpha_val) in enumerate(sorted_alphas, start=1):
                lines.append(f"{rank:2d}. {name:<6s}  {alpha_val * 100:+.2f}%")

            # Aggregate transaction cost breakdown across all 11 assets.
            total_cost = sum(
                float(episode_record["assets"][a]["transaction_costs"].sum())
                for a in asset_names
            )
            total_notional = sum(
                float(episode_record["assets"][a].get("terminal_info", {}).get("episode_traded_notional", 0.0))
                for a in asset_names
            )
            total_commission = sum(
                float(episode_record["assets"][a].get("terminal_info", {}).get("episode_cost_commission", 0.0))
                for a in asset_names
            )
            total_spread = sum(
                float(episode_record["assets"][a].get("terminal_info", {}).get("episode_cost_spread", 0.0))
                for a in asset_names
            )
            total_impact = sum(
                float(episode_record["assets"][a].get("terminal_info", {}).get("episode_cost_impact", 0.0))
                for a in asset_names
            )
            total_fixed = sum(
                float(episode_record["assets"][a].get("terminal_info", {}).get("episode_cost_fixed", 0.0))
                for a in asset_names
            )
            bps = total_cost / total_notional * 10000 if total_notional > 0 else 0.0

            lines += [
                "",
                "Transaction Costs (all assets)",
                "─" * 24,
                f"Total:  ${total_cost:.2f}  ({bps:.1f} bps)",
                f"Commis: ${total_commission:.2f}",
                f"Spread: ${total_spread:.2f}",
                f"Impact: ${total_impact:.2f}",
            ]
            if total_fixed > 0:
                lines.append(f"Fixed:  ${total_fixed:.2f}")

            # Buy-and-hold reference cost: paid once at episode start across all assets.
            total_bh_tc = sum(
                float(episode_record["assets"][a].get("bh_init_tc", 0.0))
                for a in asset_names
            )
            lines.append(f"B&H TC: ${total_bh_tc:.2f}  (init only)")

            ax_text.text(
                0.05, 0.95, "\n".join(lines),
                transform=ax_text.transAxes,
                verticalalignment="top",
                fontsize=8,
                fontfamily="monospace",
            )

        fname = f"asset-specific_saa_portfolio_test_page{page_num}_{date_file}_{timestamp}.png"
        out_path = os.path.join(output_dir, fname)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    # Page 1: first 6 assets; Page 2: remaining 5 + text.
    saved_paths.append(_make_page(asset_names[:6], page_num=1, show_alpha_text=False))
    if len(asset_names) > 6:
        saved_paths.append(_make_page(asset_names[6:], page_num=2, show_alpha_text=True))

    return saved_paths


def _plot_aggregate_page(
    episode_record: Dict[str, Any],
    asset_names: List[str],
    date_display: str,
    date_file: str,
    output_dir: str,
    timestamp: str,
) -> str:
    """Produce one aggregate PNG comparing combined agent portfolio vs S&P 500 B&H."""
    _apply_paper_style()

    # Combined portfolio = equal-weight average of all shadow sub-portfolios.
    asset_pvs = np.stack(
        [episode_record["assets"][a]["portfolio_values"] for a in asset_names], axis=1
    )
    combined_pv = asset_pvs.mean(axis=1)

    spy_rec = episode_record["assets"].get("SPY")
    spy_bh = np.asarray(spy_rec["bh_values"], dtype=np.float64) if spy_rec is not None else None

    T = len(combined_pv)
    x = np.arange(T)

    combined_idx = _index_to_100(combined_pv)

    fig, ax = plt.subplots(figsize=(8.27, 4.5))
    ax.plot(x, combined_idx, color="#1f77b4", linewidth=1.6, label="Combined agent portfolio (equal-weight)")

    spy_idx_vals = None
    if spy_bh is not None and len(spy_bh) == T:
        spy_idx_vals = _index_to_100(spy_bh)
        ax.plot(x, spy_idx_vals, color="#d62728", linewidth=1.4, linestyle="--", label="SPY Buy-and-hold")

    final_agent = float(combined_idx[-1] - 100.0) if len(combined_idx) > 0 else 0.0
    final_spy = float(spy_idx_vals[-1] - 100.0) if spy_idx_vals is not None else None

    summary_lines = [f"Agent: {final_agent:+.1f}%"]
    if final_spy is not None:
        summary_lines.append(f"SPY B&H: {final_spy:+.1f}%")
    ax.text(
        0.99, 0.03, "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    ax.set_title(
        f"Aggregate: Combined Agent vs S&P 500 B&H\n{date_display}",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Index (t₀=100)")
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.axhline(100.0, color="grey", linewidth=0.6, linestyle=":")

    fname = f"aggregate_saa_portfolio_test_{date_file}_{timestamp}.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ================================
# Report orchestrator
# ================================
def _generate_all_reports(
    all_episode_records: List[Dict[str, Any]],
    asset_names: List[str],
    cache: MarketDataCache,
    output_dir: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """Generate all JSON files and PNG figures.

    Returns:
        (combined_json_data, list_of_saved_png_paths)
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y_%m_%d_%H_%M")

    asset_specific_all: List[Dict[str, Any]] = []
    aggregate_all: List[Dict[str, Any]] = []
    all_png_paths: List[str] = []

    for ep_rec in all_episode_records:
        dates = ep_rec["assets"][asset_names[0]]["dates"]
        date_file = f"{dates[0].replace('-', '')}_{dates[-1].replace('-', '')}"
        date_display = f"{dates[0].replace('-', '/')} to {dates[-1].replace('-', '/')}"

        # Per-asset JSON metrics for this episode.
        per_asset_metrics: Dict[str, Any] = {
            "episode_id": ep_rec["episode_id"],
            "block_id": ep_rec["block_id"],
            "date_range": date_display,
            "assets": {},
        }
        for a in asset_names:
            per_asset_metrics["assets"][a] = _build_asset_json_metrics(ep_rec["assets"][a])
        asset_specific_all.append(per_asset_metrics)

        # Aggregate JSON metrics for this episode.
        agg_metrics = _build_aggregate_json_metrics(ep_rec, asset_names, cache)
        aggregate_all.append(agg_metrics)

        # PNG — asset-specific pages.
        try:
            png_paths = _plot_asset_specific_pages(ep_rec, asset_names, date_display, date_file, output_dir, timestamp)
            all_png_paths.extend(png_paths)
        except Exception:
            print(f"[Shadow-Test] WARNING: asset-specific plot failed for episode {ep_rec['episode_id']}:")
            traceback.print_exc()

        # PNG — aggregate page.
        try:
            agg_path = _plot_aggregate_page(ep_rec, asset_names, date_display, date_file, output_dir, timestamp)
            all_png_paths.append(agg_path)
        except Exception:
            print(f"[Shadow-Test] WARNING: aggregate plot failed for episode {ep_rec['episode_id']}:")
            traceback.print_exc()

    # Write JSON files.
    asset_json_path = os.path.join(output_dir, f"asset-specific_saa_portfolio_test_{timestamp}.json")
    aggregate_json_path = os.path.join(output_dir, f"aggregate_saa_portfolio_test_{timestamp}.json")

    with open(asset_json_path, "w", encoding="utf-8") as f:
        json.dump(asset_specific_all, f, indent=2, default=float)
    with open(aggregate_json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_all, f, indent=2, default=float)

    print(f"\n[Shadow-Test] Asset-specific JSON: {asset_json_path}")
    print(f"[Shadow-Test] Aggregate JSON:       {aggregate_json_path}")
    print(f"[Shadow-Test] Saved {len(all_png_paths)} PNG files to: {output_dir}")

    combined_json = {
        "asset_specific": asset_specific_all,
        "aggregate": aggregate_all,
    }
    return combined_json, all_png_paths


# ================================
# Console summary helpers
# ================================
def _print_episode_summary(
    ep_rec: Dict[str, Any],
    asset_names: List[str],
) -> None:
    dates = ep_rec["assets"][asset_names[0]]["dates"]
    print(
        f"  Block={ep_rec['block_id']}  {dates[0]} → {dates[-1]}  "
        f"({ep_rec['episode_length']} days)",
        flush=True,
    )
    asset_pvs = np.stack(
        [ep_rec["assets"][a]["portfolio_values"] for a in asset_names], axis=1
    )
    combined_pv = asset_pvs.mean(axis=1)
    combined_ret = (combined_pv[-1] / combined_pv[0] - 1.0) if combined_pv[0] > 0 else 0.0

    for a in asset_names:
        rec = ep_rec["assets"][a]
        pv = rec["portfolio_values"]
        bh = rec["bh_values"]
        agent_ret = (pv[-1] / pv[0] - 1.0) if pv[0] > 0 else 0.0
        bh_ret = (bh[-1] / bh[0] - 1.0) if bh[0] > 0 else 0.0
        alpha = agent_ret - bh_ret
        sharpe = _annual_sharpe(np.diff(pv) / np.where(pv[:-1] > 0, pv[:-1], 1.0))
        mdd = _max_drawdown(pv)
        costs = float(rec["transaction_costs"].sum())
        print(
            f"    {a:<6s} | ret={agent_ret*100:+6.2f}%  BH={bh_ret*100:+6.2f}%  "
            f"α={alpha*100:+6.2f}%  Sharpe={sharpe:5.2f}  MaxDD={mdd*100:5.2f}%  "
            f"costs=${costs:.2f}",
            flush=True,
        )
    print(f"    COMBINED: ret={combined_ret*100:+6.2f}%", flush=True)


# ================================
# Entry point (matches main.py contract)
# ================================
def run(cache: MarketDataCache, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point called by main.py.

    Loads the SAA model specified in config['test_agent']['saa_model_run_dir'],
    runs inference across all validation periods for every asset in shadow-
    portfolio mode, and produces JSON + PNG reports.
    """
    print("\n" + "=" * 70, flush=True)
    print("SAA SHADOW PORTFOLIO INFERENCE TEST", flush=True)
    print("=" * 70, flush=True)

    test_cfg = config.get("test_agent", {})
    device = torch.device(test_cfg.get("device", "cpu"))
    deterministic = bool(test_cfg.get("deterministic_saa", True))
    seed = int(config.get("training", {}).get("seed", 42))

    saa_zip_path = test_cfg.get("saa_model_run_dir")
    if not saa_zip_path:
        raise ValueError("config['test_agent']['saa_model_run_dir'] is required.")
    saa_zip_path = os.path.normpath(saa_zip_path)

    # action_limiting_factor must come from the inherited SAA training config.
    agent_cfg = config.get("agent", {})
    if "action_limiting_factor_end" not in agent_cfg:
        raise ValueError(
            "Inherited SAA training config is missing 'agent.action_limiting_factor_end'; "
            "cannot determine deployment-time action scaling."
        )
    action_limiting_factor = float(agent_cfg["action_limiting_factor_end"])
    print(f"[Shadow-Test] action_limiting_factor = {action_limiting_factor}", flush=True)

    # ---- Load model ----
    model, vecnorm = _load_saa_model(saa_zip_path, device)

    # ---- One model copy per asset (independent LSTM state) ----
    asset_names: List[str] = list(cache.asset_names)
    num_assets = len(asset_names)
    print(f"[Shadow-Test] Assets ({num_assets}): {asset_names}", flush=True)

    asset_models: Dict[str, RecurrentPPO] = {}
    for a in asset_names:
        m = copy.deepcopy(model)
        m.policy.to(device)
        m.device = device
        m.policy.eval()
        asset_models[a] = m
    print(f"[Shadow-Test] Created {num_assets} model copies.", flush=True)

    # ---- Validation episode plan ----
    episode_plan = _build_validation_episode_plan(cache)
    print(f"[Shadow-Test] Validation plan: {len(episode_plan)} episodes.", flush=True)

    # ---- Run all episodes ----
    all_episode_records: List[Dict[str, Any]] = []

    for ep_idx, (block_id, start_step, ep_length) in enumerate(episode_plan):
        print(
            f"\n[Shadow-Test] === Episode {ep_idx + 1}/{len(episode_plan)} | "
            f"block={block_id} start={start_step} len={ep_length} ===",
            flush=True,
        )
        episode_record: Dict[str, Any] = {
            "episode_id": ep_idx,
            "block_id": block_id,
            "start_step": start_step,
            "episode_length": ep_length,
            "assets": {},
        }

        episode_failed = False
        for asset_name in asset_names:
            print(f"[Shadow-Test]   Running asset: {asset_name} ...", flush=True)
            try:
                asset_rec = _run_asset_episode(
                    cache=cache,
                    config=config,
                    model=asset_models[asset_name],
                    vecnorm=vecnorm,
                    asset_name=asset_name,
                    action_limiting_factor=action_limiting_factor,
                    deterministic=deterministic,
                    device=device,
                    block_id=block_id,
                    episode_start_step=start_step,
                    episode_length=ep_length,
                    seed=seed + ep_idx,
                )
                episode_record["assets"][asset_name] = asset_rec
            except Exception:
                print(f"[Shadow-Test]   ERROR running {asset_name} in episode {ep_idx}:", flush=True)
                traceback.print_exc()
                episode_failed = True
                break

        if episode_failed or len(episode_record["assets"]) < num_assets:
            print(f"[Shadow-Test] Skipping episode {ep_idx} due to partial failure.", flush=True)
            continue

        all_episode_records.append(episode_record)
        _print_episode_summary(episode_record, asset_names)

    if not all_episode_records:
        raise RuntimeError("No episodes completed successfully.")

    # ---- Determine output directory ----
    # Model zip is at: <agent_root>/saved_models/<run_dir_name>/<zip_name>
    run_dir = os.path.dirname(saa_zip_path)
    output_dir = os.path.join(run_dir, "saa_inference_test_results")

    # ---- Generate reports ----
    combined_json, png_paths = _generate_all_reports(
        all_episode_records, asset_names, cache, output_dir
    )

    # ---- Compute top-level aggregate summary for console ----
    all_agent_rets = []
    all_bh_rets = []
    all_costs = []
    for ep_rec in all_episode_records:
        for a in asset_names:
            rec = ep_rec["assets"][a]
            pv = rec["portfolio_values"]
            bh = rec["bh_values"]
            if pv[0] > 0:
                all_agent_rets.append(pv[-1] / pv[0] - 1.0)
            if bh[0] > 0:
                all_bh_rets.append(bh[-1] / bh[0] - 1.0)
            all_costs.append(float(rec["transaction_costs"].sum()))

    mean_agent_ret = float(np.mean(all_agent_rets)) if all_agent_rets else 0.0
    mean_bh_ret = float(np.mean(all_bh_rets)) if all_bh_rets else 0.0
    mean_alpha = mean_agent_ret - mean_bh_ret
    mean_costs = float(np.mean(all_costs)) if all_costs else 0.0

    # Annualised Sharpe across all asset-episodes.
    all_sharpes = []
    for ep_rec in all_episode_records:
        for a in asset_names:
            pv = ep_rec["assets"][a]["portfolio_values"]
            if len(pv) > 2:
                rets = np.diff(pv) / np.where(pv[:-1] > 0, pv[:-1], 1.0)
                all_sharpes.append(_annual_sharpe(rets))
    mean_sharpe = float(np.mean(all_sharpes)) if all_sharpes else 0.0

    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY (mean over all episodes × assets)")
    print("=" * 70)
    print(f"Mean agent total return:   {mean_agent_ret * 100:+7.2f}%")
    print(f"Mean B&H total return:     {mean_bh_ret * 100:+7.2f}%")
    print(f"Mean alpha (agent – B&H):  {mean_alpha * 100:+7.2f}%")
    print(f"Mean annualised Sharpe:    {mean_sharpe:7.4f}")
    print(f"Mean transaction costs:    ${mean_costs:.2f}")

    # ---- Open PNG files with OS default viewer ----
    print(f"\n[Shadow-Test] Opening {len(png_paths)} graph(s)...", flush=True)
    for path in png_paths:
        try:
            os.startfile(path)  # Windows default viewer
        except Exception:
            pass  # Non-fatal; user can open files manually.

    return {
        "agent": "test_saa_inference_shadow_portfolios",
        "status": "completed",
        "num_episodes_completed": len(all_episode_records),
        "output_dir": output_dir,
        "aggregate": {
            "mean_agent_total_return": mean_agent_ret,
            "mean_bh_total_return": mean_bh_ret,
            "mean_alpha": mean_alpha,
            "mean_annual_sharpe": mean_sharpe,
            "mean_transaction_costs": mean_costs,
        },
    }
