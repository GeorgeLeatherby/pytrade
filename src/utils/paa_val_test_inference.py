"""
PAA validation- and test-period inference tester
================================================

Replays every configured Portfolio Allocator Agent (PAA) checkpoint deterministically
through every validation block and the out-of-sample test block, each spanning the block
in full and starting from 100% cash, and writes a dense daily table per checkpoint plus
a JSON metadata/summary file into ``<run_dir>/val_and_test_inference/``. For the best
checkpoint of each PAA family (highest mean validation market alpha, i.e. PAA total return
minus SPY buy-and-hold total return) it additionally renders one A4-portrait 3-panel figure
per period.

Everything is driven through the unmodified training stack:
  TradingEnv -> PortfolioEpisodeAdapter -> DailyRecorder (this file, read-only tap)
  -> DummyVecEnv -> {SAASignalWrapper | AR1SignalWrapper | ZeroSignalWrapper}
  -> VecNormalize (checkpoint stats, frozen) -> PPO.predict(deterministic=True)

Reference SAA ("SAA equal-weight"): the per-asset shadow sub-portfolios in TradingEnv only
depend on market data and the frozen SAA, never on the PAA's actions. A dedicated reference
pass (SAASignalWrapper, dummy PAA action) therefore yields the SAA's own books for every
period; they are shared by all agents with the same SAA/env settings and are cross-checked
against the hierarchical PAA's injected books. Each shadow book starts with the full
initial capital, so the mean of the N books equals the SAA managing the PAA's capital
split 1/N per asset (exact up to the sqrt market-impact term, which is not scale-free).

Usage (from repo root):  python src/utils/paa_val_test_inference.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
# Both roots are needed: checkpoints pickle their policy class under "src.agents..." (main.py
# import path) while the agent modules import each other as "agents..."/"environment...".
for _p in (str(REPO_ROOT), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.text import Text
import seaborn as sns

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv

from environment.trading_environment import MarketDataCache, TradingEnv
from agents.PPO_portfolio_allocator_weights.ppo_portfolio_allocator_weights_agent import (
    PortfolioEpisodeAdapter,
    SAASignalWrapper,
    _load_saa_from_config,
    _make_trading_env,
)
from agents.PAA_with_autoregressive_rnd_walk_SAA.PAA_with_autoregressive_rnd_walk_SAA import AR1SignalWrapper
from agents.PAA_cross_sectional_only.PAA_cross_sectional_only import ZeroSignalWrapper


# =====================================================================================
# USER SETTINGS - checkpoint lists (paths relative to the repo root)
# =====================================================================================
HIERARCHICAL_PAA_CHECKPOINTS: List[str] = [
    r"src\agents\PPO_portfolio_allocator_weights\saved_models\00272_config_10022_26_09_29\best_model_excess_over_spy_abs.zip",
    r"src\agents\PPO_portfolio_allocator_weights\saved_models\00272_config_10022_26_09_29\best_model_terminal_pnl_mean.zip",
    r"src\agents\PPO_portfolio_allocator_weights\saved_models\00272_config_10022_26_09_29\best_model_terminal_pnl_min.zip",
    r"src\agents\PPO_portfolio_allocator_weights\saved_models\00272_config_10022_26_09_29\best_model.zip",
]
CONTROL_ABLATION_AR1_CHECKPOINTS: List[str] = [
    r"src\agents\PAA_with_autoregressive_rnd_walk_SAA\saved_models\00266_config_20004_26_09_24\best_model_excess_over_spy_abs.zip",
    r"src\agents\PAA_with_autoregressive_rnd_walk_SAA\saved_models\00266_config_20004_26_09_24\best_model_terminal_pnl_mean.zip",
    r"src\agents\PAA_with_autoregressive_rnd_walk_SAA\saved_models\00266_config_20004_26_09_24\best_model_terminal_pnl_min.zip",
    r"src\agents\PAA_with_autoregressive_rnd_walk_SAA\saved_models\00266_config_20004_26_09_24\best_model.zip",
]
CROSS_SECTIONAL_ABLATION_CHECKPOINTS: List[str] = [
    r"src\agents\PAA_cross_sectional_only\saved_models\00265_config_30004_26_09_24\best_model_excess_over_spy_abs.zip",
    r"src\agents\PAA_cross_sectional_only\saved_models\00265_config_30004_26_09_24\best_model_terminal_pnl_mean.zip",
    r"src\agents\PAA_cross_sectional_only\saved_models\00265_config_30004_26_09_24\best_model_terminal_pnl_min.zip",
    r"src\agents\PAA_cross_sectional_only\saved_models\00265_config_30004_26_09_24\best_model.zip",
]

# Ablation configs carry no saa_config/saa_features; the reference SAA is taken from here.
REFERENCE_SAA_CONFIG_PATH = r"src\agents\PPO_portfolio_allocator_weights\config_10022.json"

DEVICE = "cpu"
OUTPUT_FOLDER_NAME = "val_and_test_inference"
MAKE_GRAPHS = True
GRAPH_DPI = 300
# Seed offsets of the AR(1) noise stream (validation offset matches the training eval env).
AR1_SEED_OFFSET = {"validation": 10_000, "test": 20_000}


# =====================================================================================
# Constants
# =====================================================================================
FAMILY_HIERARCHICAL = "hierarchical"
FAMILY_AR1 = "control_ablation_ar1"
FAMILY_XSEC = "cross_sectional_ablation"
FAMILY_LABELS = {
    FAMILY_HIERARCHICAL: "Hierarchical PAA (SAA signal)",
    FAMILY_AR1: "Control ablation PAA (AR(1) signal)",
    FAMILY_XSEC: "Cross-sectional ablation PAA (zero signal)",
}
TRADING_DAYS = 252
CASH_COLOR = "#D9D9D9"
ASSET_COLORS = {
    "Gold": "#D4AF37", "Crude": "#8B4513",
    "SPY": "#332288", "EWG": "#88CCEE", "EWH": "#44AA99", "EWJ": "#117733", "EWQ": "#0072B2",
    "EWS": "#CC6677", "EWT": "#882255", "EWU": "#AA4499", "EWY": "#D55E00",
}
TC_PART_NAMES = ("commission", "spread", "impact", "fixed")


# =====================================================================================
# Checkpoint resolution
# =====================================================================================
@dataclass
class CheckpointSpec:
    family: str
    rel_path: str
    zip_path: Path
    vecnorm_path: Path
    run_dir: Path
    run_id: str
    config_id: str
    run_date: str
    stem: str
    config_path: Path
    config: Dict[str, Any] = field(repr=False)

    @property
    def label(self) -> str:
        return f"{self.run_dir.name}/{self.stem}"


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint(family: str, rel_path: str) -> CheckpointSpec:
    zip_path = (REPO_ROOT / Path(rel_path.replace("\\", "/"))).resolve()
    if zip_path.suffix.lower() != ".zip":
        raise ValueError(f"Checkpoint must be a .zip file: {rel_path}")
    if not zip_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {zip_path}")
    vecnorm_path = zip_path.with_name(f"{zip_path.stem}_vecnormalize.pkl")
    if not vecnorm_path.is_file():
        raise FileNotFoundError(f"VecNormalize stats not found next to checkpoint: {vecnorm_path}")

    run_dir = zip_path.parent
    m = re.fullmatch(r"(\d+)_config_(\d+)_(\d{2}_\d{2}_\d{2})", run_dir.name)
    if not m:
        raise ValueError(f"Run directory name does not match '<runid>_config_<cfgid>_<yy_mm_dd>': {run_dir.name}")
    run_id, config_id, run_date = m.groups()
    config_path = run_dir.parent.parent / f"config_{config_id}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    return CheckpointSpec(
        family=family, rel_path=rel_path, zip_path=zip_path, vecnorm_path=vecnorm_path,
        run_dir=run_dir, run_id=run_id, config_id=config_id, run_date=run_date,
        stem=zip_path.stem, config_path=config_path, config=_load_json(config_path),
    )


def _inference_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of a training config with the overrides needed for deterministic replay."""
    cfg = copy.deepcopy(config)
    # TradingEnv.reset() ignores option["force_cash_only_start"] in portfolio_weights mode;
    # this is the only reliable way to guarantee a 100%-cash start.
    cfg["environment"]["percentage_of_cash_only_starts"] = 1.0
    return cfg


def _reference_saa_config(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """Agent config (env/frictions) with the SAA sections of the agent or of the reference config."""
    cfg = _inference_config(agent_config)
    if "saa_config" not in cfg or "saa_features" not in cfg:
        ref = _load_json(REPO_ROOT / Path(REFERENCE_SAA_CONFIG_PATH.replace("\\", "/")))
        cfg["saa_config"] = copy.deepcopy(ref["saa_config"])
        cfg["saa_features"] = copy.deepcopy(ref["saa_features"])
    cfg["saa_config"] = dict(cfg["saa_config"], device=DEVICE)
    return cfg


# =====================================================================================
# Market data cache + period plans
# =====================================================================================
_DF_CACHE: Dict[str, pd.DataFrame] = {}
_MDC_CACHE: Dict[str, MarketDataCache] = {}


def _market_data_path(config: Dict[str, Any]) -> Path:
    p = config.get("market_data_path")
    return (REPO_ROOT / p).resolve() if p else REPO_ROOT / "src" / "data" / "enriched_financial_data.csv"


def get_market_cache(config: Dict[str, Any]) -> MarketDataCache:
    """Same construction as main.run_agent; memoized on everything that shapes the cache."""
    env = {k: v for k, v in config["environment"].items() if k != "seed"}
    key = json.dumps({
        "env": env,
        "saa_features": config.get("saa_features"),
        "paa_asset_token_features": config.get("paa_asset_token_features"),
        "paa_portfolio_token_features": config.get("paa_portfolio_token_features"),
        "data": str(_market_data_path(config)),
    }, sort_keys=True)
    if key not in _MDC_CACHE:
        data_path = str(_market_data_path(config))
        if data_path not in _DF_CACHE:
            _DF_CACHE[data_path] = pd.read_csv(data_path)
        _MDC_CACHE[key] = MarketDataCache.from_dataframe(
            _DF_CACHE[data_path], config,
            lookback_window=config["environment"]["lookback_window"],
            maybe_provide_sequence=config["environment"].get("maybe_provide_sequence", False),
        )
    return _MDC_CACHE[key]


def build_period_plans(cache: MarketDataCache, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One full-block, cash-start episode per validation block, then per test block."""
    plans: List[Dict[str, Any]] = []
    for period_type, blocks in (("validation", cache.validation_blocks), ("test", cache.test_blocks)):
        for block in sorted(blocks, key=lambda b: b.start_date_idx):
            start = int(block.min_start_step)
            plans.append({
                "period_type": period_type,
                "block_id": str(block.block_id),
                "reset_option": {
                    "block_id": str(block.block_id),
                    "episode_start_step": start,
                    "episode_length_override": int(block.end_date_idx - start),
                    "force_cash_only_start": True,
                },
            })

    # Validation plans must equal the training-time sweep plan (same blocks, same spans).
    sweep = PortfolioEpisodeAdapter(TradingEnv(config=config, market_data_cache=cache, mode="validation"))
    expected = sweep.get_validation_sweep_plan()
    got = [p["reset_option"] for p in plans if p["period_type"] == "validation"]
    if got != expected:
        raise RuntimeError(f"Validation plan mismatch with PortfolioEpisodeAdapter:\n{got}\nvs\n{expected}")
    return plans


def _block_signature(cache: MarketDataCache) -> List[Tuple[str, int, int]]:
    return [(b.block_id, int(b.start_date_idx), int(b.end_date_idx))
            for b in list(cache.validation_blocks) + list(cache.test_blocks)]


# =====================================================================================
# DailyRecorder: read-only tap on TradingEnv
# =====================================================================================
class DailyRecorder(gym.Wrapper):
    """
    Captures the exact per-day state of one TradingEnv episode before DummyVecEnv's
    auto-reset wipes the EpisodeBuffer. Wraps (as instance attributes, no class changes)
    execute_portfolio_change / _calculate_transaction_costs / apply_saa_sub_actions to
    capture executed trades, cost components and shadow-book (SAA) trades exactly.

    Day-t conventions (t = 0 .. T-1, date = dates[start + t]):
      *_value, paa_w_*, paa_shares_*, sub books: marked-to-market at close t, BEFORE the
          day-t rebalancing (after the carry/price update from t-1 to t).
      paa_target_w_*, paa_exec_w_*, paa_trade_*, paa_tc*: the PAA rebalancing executed at
          close t (prices of day t); NaN on the last day (no action).
      sub-book signal/trade: the (SAA/AR1) action committed to the shadow books at close t.
    Only records while armed; the auto-reset episode after a terminal step is ignored.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.te: TradingEnv = env.unwrapped
        self._armed = False
        self._ep: Optional[Dict[str, Any]] = None
        self.completed: Optional[Dict[str, Any]] = None
        self._in_paa_exec = False
        self._in_saa_apply = False
        self._tc_parts = np.zeros(4, dtype=np.float64)
        self._saa_tc = None
        self._last_exec: Optional[Dict[str, Any]] = None
        self._patch_env()

    # ---------------------------------------------------------------- patches
    def _patch_env(self) -> None:
        te = self.te
        rec = self
        orig_tc = te._calculate_transaction_costs
        orig_exec = te.execute_portfolio_change
        orig_apply = te.apply_saa_sub_actions

        def tc_patched(shares_traded, prices, abs_step, asset_mask=None):
            cost = orig_tc(shares_traded, prices, abs_step, asset_mask)
            if cost > 0.0:
                if rec._in_paa_exec:
                    rec._tc_parts += np.asarray(te._last_cost_breakdown, dtype=np.float64)
                elif rec._in_saa_apply and asset_mask is not None:
                    rec._saa_tc[np.flatnonzero(asset_mask)[0]] += cost
            return cost

        def exec_patched(target_weights, portfolio_state):
            rec._tc_parts = np.zeros(4, dtype=np.float64)
            pre_cash = float(portfolio_state.cash)
            pre_pos = portfolio_state.positions.astype(np.float64).copy()
            rec._in_paa_exec = True
            try:
                res = orig_exec(target_weights, portfolio_state)
            finally:
                rec._in_paa_exec = False
            rec._last_exec = {
                "target_w": np.asarray(target_weights, dtype=np.float64).copy(),
                "pre_cash": pre_cash,
                "pre_pos": pre_pos,
                "post_cash": float(portfolio_state.cash),
                "post_pos": portfolio_state.positions.astype(np.float64).copy(),
                "post_w": portfolio_state.get_weights().astype(np.float64),
                "tc": float(res.transaction_cost),
                "tc_parts": rec._tc_parts.copy(),
                "trade_shares": np.asarray(res.trades_executed, dtype=np.float64).copy(),
                "traded_notional": np.asarray(res.traded_notional_per_asset, dtype=np.float64).copy(),
            }
            return res

        def apply_patched(actions_per_asset):
            ep = rec._ep
            if ep is None:
                return orig_apply(actions_per_asset)
            t = int(te.current_step)
            n = te.market_data_cache.num_assets
            cash0, sh0, _, _ = te.episode_buffer.get_saa_sub_state(t)
            sh0 = sh0.astype(np.float64).copy()
            rec._saa_tc = np.zeros(n, dtype=np.float64)
            rec._in_saa_apply = True
            try:
                orig_apply(actions_per_asset)
            finally:
                rec._in_saa_apply = False
            cash1, sh1, _, _ = te.episode_buffer.get_saa_sub_state(t)
            px = te.portfolio_state.prices.astype(np.float64)
            post_cash = cash1.astype(np.float64)
            post_notional = sh1.astype(np.float64) * px
            post_total = post_cash + post_notional
            ep["sub_signal"][t] = np.asarray(actions_per_asset, dtype=np.float64)
            ep["sub_trade_notional"][t] = (sh1.astype(np.float64) - sh0) * px
            ep["sub_tc"][t] = rec._saa_tc
            ep["sub_weight_post"][t] = np.where(post_total > 1e-12, post_notional / np.maximum(post_total, 1e-12), 0.0)
            ep["sub_committed"][t] = True

        te._calculate_transaction_costs = tc_patched
        te.execute_portfolio_change = exec_patched
        te.apply_saa_sub_actions = apply_patched

    # ---------------------------------------------------------------- recording
    def arm(self) -> None:
        self._armed = True
        self.completed = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._armed:
            self._armed = False
            self._start_episode()
        return obs, info

    def _bh_init(self, target_positions: np.ndarray, prices: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """Buy-and-hold init via the env's own cost-aware routine (as used for the SPY B&H)."""
        te = self.te
        saved = getattr(te, "_last_cost_breakdown", None)
        cash, pos, tc = te._initialize_portfolio_with_costs(
            target_positions=target_positions.astype(np.float32), initial_prices=prices,
            initial_value=te.initial_portfolio_value, allow_cash_residual=True, max_iterations=10,
        )
        if saved is not None:
            te._last_cost_breakdown = saved
        return float(cash), np.asarray(pos, dtype=np.float64), float(tc)

    def _start_episode(self) -> None:
        te = self.te
        n = te.market_data_cache.num_assets
        T = int(te.current_episode_length)
        f = lambda *shape: np.full(shape, np.nan, dtype=np.float64)
        ep: Dict[str, Any] = {
            "block_id": te.current_block_id, "start_step": int(te.current_episode_start_step), "T_planned": T,
            "abs_step": np.zeros(T, dtype=np.int64), "rf_daily": f(T), "effr_pa": f(T),
            "prices": f(T, n), "paa_value": f(T), "paa_cash": f(T), "paa_w": f(T, n + 1), "paa_shares": f(T, n),
            "paa_reward": f(T), "paa_logits": f(T, n), "paa_target_w": f(T, n + 1), "paa_exec_w": f(T, n + 1),
            "paa_trade_shares": f(T, n), "paa_trade_notional": f(T, n), "paa_tc": f(T), "paa_tc_parts": f(T, 4),
            "spy_bh_value": f(T), "env_benchmark_value": f(T), "cash_value": f(T), "ew_bh_value": f(T),
            "sub_value_mtm": f(T, n), "sub_signal": f(T, n), "sub_trade_notional": f(T, n), "sub_tc": f(T, n),
            "sub_weight_post": f(T, n), "sub_committed": np.zeros(T, dtype=bool),
        }
        self._ep = ep
        prices0 = te.portfolio_state.prices.astype(np.float64)
        K = float(te.initial_portfolio_value)

        # Equal-weight (1/N) buy-and-hold and a SPY replica, both via the env's init routine.
        ew_cash, ew_pos, ew_tc = self._bh_init((K / n) / prices0, prices0)
        spy_target = np.zeros(n)
        spy_target[te.spy_asset_index] = K / prices0[te.spy_asset_index]
        spy_cash, spy_pos, _ = self._bh_init(spy_target, prices0)
        ep["ew_state"] = {"cash": ew_cash, "pos": ew_pos, "init_tc": ew_tc}
        ep["spy_replica"] = {"cash": spy_cash, "pos": spy_pos}
        ep["spy_bh_init_tc"] = float(te.selected_asset_bh_init_transaction_cost)
        ep["spy_bh_init_cash"] = float(te.selected_asset_bh_portfolio_state.cash)
        ep["ew_bh_init_cash"] = ew_cash
        ep["env_benchmark_init_value"] = float(te.benchmark_portfolio_state.get_total_value())

        if abs(te.portfolio_state.cash - K) > 1e-6 or np.any(te.portfolio_state.positions != 0):
            raise RuntimeError("Episode did not start from 100% cash.")
        self._snapshot(0, reward=0.0)

    def _snapshot(self, t: int, reward: float) -> None:
        te, ep = self.te, self._ep
        cache = te.market_data_cache
        ps = te.portfolio_state
        prices = ps.prices.astype(np.float64)
        abs_t = int(te.current_absolute_step)
        ep["abs_step"][t] = abs_t
        ep["rf_daily"][t] = float(cache.get_risk_free_rate_daily_at_step(abs_t))
        ep["effr_pa"][t] = float(cache.get_risk_free_rate_pa_at_step(abs_t))
        ep["prices"][t] = prices
        ep["paa_value"][t] = ps.get_total_value()
        ep["paa_cash"][t] = float(ps.cash)
        ep["paa_w"][t] = ps.get_weights().astype(np.float64)
        ep["paa_shares"][t] = ps.positions.astype(np.float64)
        ep["paa_reward"][t] = float(reward)
        ep["spy_bh_value"][t] = te.selected_asset_bh_portfolio_state.get_total_value()
        ep["env_benchmark_value"][t] = te.benchmark_portfolio_state.get_total_value()
        ep["cash_value"][t] = te.comparison_portfolio_state.get_total_value()

        if t > 0:
            # Same carry the env applies to every cash balance on this step.
            mult = 1.0 + float(cache.get_risk_free_rate_daily_at_step(abs_t))
            ep["ew_state"]["cash"] *= mult
            ep["spy_replica"]["cash"] *= mult
        ep["ew_bh_value"][t] = ep["ew_state"]["cash"] + float(ep["ew_state"]["pos"] @ prices)
        spy_rep = ep["spy_replica"]["cash"] + float(ep["spy_replica"]["pos"] @ prices)
        if abs(spy_rep - ep["spy_bh_value"][t]) > 0.05:
            raise RuntimeError(f"SPY B&H replica drifted from env at t={t}: {spy_rep} vs {ep['spy_bh_value'][t]}")

        cash_sub, sh_sub, _, _ = te.episode_buffer.get_saa_sub_state(t)
        ep["sub_value_mtm"][t] = cash_sub.astype(np.float64) + sh_sub.astype(np.float64) * prices

        # Slot 0 is skipped: TradingEnv.reset() records the env benchmark's value there (env bug).
        buf_v = float(te.episode_buffer.portfolio_values[t])
        if t > 0 and abs(buf_v - ep["paa_value"][t]) > max(0.05, 1e-6 * abs(buf_v)):
            raise RuntimeError(f"EpisodeBuffer value mismatch at t={t}: {buf_v} vs {ep['paa_value'][t]}")

    def step(self, action):
        ep = self._ep
        if ep is None:
            return self.env.step(action)
        te = self.te
        t = int(te.current_step)
        self._last_exec = None
        obs, reward, terminated, truncated, info = self.env.step(action)
        ex = self._last_exec
        if ex is None:
            raise RuntimeError("execute_portfolio_change was not called during step().")
        ep["paa_logits"][t] = np.asarray(action, dtype=np.float64)
        ep["paa_target_w"][t] = ex["target_w"]
        ep["paa_exec_w"][t] = ex["post_w"]
        ep["paa_trade_shares"][t] = ex["trade_shares"]
        ep["paa_trade_notional"][t] = np.sign(ex["trade_shares"]) * ex["traded_notional"]
        ep["paa_tc"][t] = ex["tc"]
        ep["paa_tc_parts"][t] = ex["tc_parts"]
        if abs(ex["tc_parts"].sum() - ex["tc"]) > 1e-6:
            raise RuntimeError(f"TC breakdown does not add up at t={t}: {ex['tc_parts']} vs {ex['tc']}")
        # Cash identity of the rebalancing (all fills at close t).
        cash_delta_expected = -float(ex["trade_shares"] @ ep["prices"][t]) - ex["tc"]
        if abs((ex["post_cash"] - ex["pre_cash"]) - cash_delta_expected) > 0.05:
            raise RuntimeError(f"Cash identity violated at t={t}")

        self._snapshot(t + 1, reward=float(reward))
        if terminated or truncated:
            n_days = t + 2
            for k, v in list(ep.items()):
                if isinstance(v, np.ndarray) and v.shape[:1] == (ep["T_planned"],):
                    ep[k] = v[:n_days].copy()
            ep["n_days"] = n_days
            ep["terminated_early"] = bool(terminated)
            ep["terminal_info"] = {k: float(v) for k, v in info.items()
                                   if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)}
            self.completed = ep
            self._ep = None
        return obs, reward, terminated, truncated, info


def _recorder_env_fn(cache: MarketDataCache, config: Dict[str, Any], mode: str, seed: int):
    base_fn = _make_trading_env(cache, config, mode, seed, for_eval=True)

    def _init():
        return DailyRecorder(base_fn())
    return _init


def run_plan(venv: VecEnv, recorder: DailyRecorder, plan: Dict[str, Any],
             policy: Callable[[np.ndarray], np.ndarray]) -> Dict[str, Any]:
    venv.env_method("set_plan_queue", [plan["reset_option"]], indices=0)
    recorder.arm()
    obs = venv.reset()
    while True:
        obs, _, dones, _ = venv.step(policy(obs))
        if dones[0]:
            break
    ep = recorder.completed
    if ep is None or ep["block_id"] != plan["block_id"]:
        raise RuntimeError(f"Episode for {plan['block_id']} was not captured.")
    return ep


# =====================================================================================
# SAA loading + reference SAA pass
# =====================================================================================
_SAA_CACHE: Dict[str, Tuple[Any, Any, torch.device, float, bool]] = {}
_REF_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}


def load_saa(saa_config: Dict[str, Any]):
    key = json.dumps(saa_config, sort_keys=True)
    if key not in _SAA_CACHE:
        _SAA_CACHE[key] = _load_saa_from_config(saa_config)
    return _SAA_CACHE[key]


def _saa_wrap(vec_raw: VecEnv, cfg: Dict[str, Any], cache: MarketDataCache) -> SAASignalWrapper:
    saa_model, saa_vecnorm, saa_device, alf, effr_active = load_saa(cfg["saa_config"])
    return SAASignalWrapper(
        vec_raw, saa_model, saa_vecnorm, cache.num_assets, saa_device, config=cfg,
        feature_to_index=cache.feature_to_index, action_limiting_factor=alf, effr_level_active=effr_active,
    )


def get_reference_saa_episodes(agent_config: Dict[str, Any], agent_cache: MarketDataCache,
                               plans: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """SAA shadow books for every plan (PAA action is irrelevant to them; a zero-logit action is used)."""
    cfg = _reference_saa_config(agent_config)
    env_no_seed = {k: v for k, v in cfg["environment"].items() if k != "seed"}
    key = json.dumps({"saa": cfg["saa_config"], "features": cfg["saa_features"],
                      "paa_a": cfg.get("paa_asset_token_features"), "paa_p": cfg.get("paa_portfolio_token_features"),
                      "env": env_no_seed}, sort_keys=True)
    _, _, _, alf, effr_active = load_saa(cfg["saa_config"])
    ref_info = {"saa_config": cfg["saa_config"], "saa_action_limiting_factor": alf,
                "saa_effr_level_active": effr_active,
                "source": "agent config" if "saa_config" in agent_config else _rel(REPO_ROOT / REFERENCE_SAA_CONFIG_PATH.replace("\\", "/"))}
    if key in _REF_CACHE:
        return _REF_CACHE[key], ref_info

    cache = get_market_cache(cfg)
    if _block_signature(cache) != _block_signature(agent_cache):
        raise RuntimeError("Reference-SAA market cache blocks differ from the agent's blocks.")
    seed = int(cfg["training"].get("seed", 42))
    episodes: Dict[str, Dict[str, Any]] = {}
    for mode in ("validation", "test"):
        mode_plans = [p for p in plans if p["period_type"] == mode]
        if not mode_plans:
            continue
        vec_raw = DummyVecEnv([_recorder_env_fn(cache, cfg, mode, seed)])
        venv = _saa_wrap(vec_raw, cfg, cache)
        recorder: DailyRecorder = vec_raw.envs[0]
        zero_action = np.zeros((1, cache.num_assets), dtype=np.float32)
        for plan in mode_plans:
            print(f"  [ref-SAA] {plan['block_id']} ...", flush=True)
            episodes[plan["block_id"]] = run_plan(venv, recorder, plan, lambda _obs: zero_action)
        venv.close()
    _REF_CACHE[key] = episodes
    return episodes, ref_info


# =====================================================================================
# PAA evaluation
# =====================================================================================
def build_agent_venv(spec: CheckpointSpec, cfg: Dict[str, Any], cache: MarketDataCache,
                     mode: str) -> Tuple[VecNormalize, DailyRecorder, Dict[str, Any]]:
    seed = int(cfg["training"].get("seed", 42))
    vec_raw = DummyVecEnv([_recorder_env_fn(cache, cfg, mode, seed)])
    recorder: DailyRecorder = vec_raw.envs[0]
    wrapper_info: Dict[str, Any] = {}
    n = cache.num_assets
    if spec.family == FAMILY_HIERARCHICAL:
        wrapped = _saa_wrap(vec_raw, cfg, cache)
        wrapper_info["injected_signal"] = "frozen SAA (deterministic)"
    elif spec.family == FAMILY_AR1:
        ar1 = cfg.get("ar1_ablation", {})
        ar1_seed = seed + AR1_SEED_OFFSET[mode]
        wrapped = AR1SignalWrapper(
            vec_raw, n, cache.feature_to_index,
            ar1_phi=float(ar1.get("phi", 0.9)), ar1_sigma=float(ar1.get("sigma", 0.15)),
            ar1_clip=float(ar1.get("clip", 1.0)), action_limiting_factor=float(ar1.get("action_limiting_factor", 0.3)),
            seed=ar1_seed,
        )
        wrapper_info.update({"injected_signal": "AR(1) noise", "ar1_seed": ar1_seed, "ar1_params": ar1})
    elif spec.family == FAMILY_XSEC:
        wrapped = ZeroSignalWrapper(vec_raw, n, cache.feature_to_index)
        wrapper_info["injected_signal"] = "constant zero (shadow books never traded)"
    else:
        raise ValueError(spec.family)
    venv = VecNormalize.load(str(spec.vecnorm_path), venv=wrapped)
    venv.training = False
    venv.norm_reward = False
    return venv, recorder, wrapper_info


def load_paa_model(spec: CheckpointSpec) -> PPO:
    custom_objects = {"learning_rate": 0.0, "lr_schedule": lambda _p: 0.0, "clip_range": lambda _p: 0.2}
    model = PPO.load(str(spec.zip_path), device=DEVICE, custom_objects=custom_objects)
    model.policy.eval()
    return model


def evaluate_checkpoint(spec: CheckpointSpec) -> Dict[str, Any]:
    print(f"\n{'=' * 90}\n[{spec.family}] {spec.label}\n{'=' * 90}", flush=True)
    cfg = _inference_config(spec.config)
    if spec.family == FAMILY_HIERARCHICAL:
        if "saa_config" not in cfg:
            raise ValueError(f"Hierarchical checkpoint config has no saa_config: {spec.config_path}")
        cfg["saa_config"] = dict(cfg["saa_config"], device=DEVICE)
    cache = get_market_cache(cfg)
    plans = build_period_plans(cache, cfg)
    ref_eps, ref_info = get_reference_saa_episodes(spec.config, cache, plans)

    model = load_paa_model(spec)
    episodes: Dict[str, Dict[str, Any]] = {}
    wrapper_info: Dict[str, Any] = {}
    for mode in ("validation", "test"):
        mode_plans = [p for p in plans if p["period_type"] == mode]
        if not mode_plans:
            continue
        venv, recorder, wrapper_info[mode] = build_agent_venv(spec, cfg, cache, mode)
        if tuple(model.observation_space.shape) != tuple(venv.observation_space.shape):
            raise RuntimeError(
                f"Observation shape mismatch: model {model.observation_space.shape} vs env {venv.observation_space.shape}. "
                "The environment changed since this checkpoint was trained."
            )
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
        for plan in mode_plans:
            t0 = time.time()
            ep = run_plan(venv, recorder, plan, policy)
            episodes[plan["block_id"]] = ep
            print(f"  [{mode}] {plan['block_id']}: {ep['n_days']} days, final PV {ep['paa_value'][-1]:,.2f} "
                  f"(SPY B&H {ep['spy_bh_value'][-1]:,.2f}) in {time.time() - t0:.1f}s", flush=True)
        venv.close()

    df = build_daily_table(spec, cache, plans, episodes, ref_eps)
    checks = run_consistency_checks(spec, cache, plans, episodes, ref_eps, df)
    summaries = summarize(df, float(cfg["environment"]["initial_portfolio_value"]))
    return {"spec": spec, "cfg": cfg, "cache": cache, "plans": plans, "df": df, "summaries": summaries,
            "checks": checks, "ref_info": ref_info, "wrapper_info": wrapper_info}


# =====================================================================================
# Daily table
# =====================================================================================
def _returns_vs_capital(values: np.ndarray, K: float) -> np.ndarray:
    """Day-0 return is measured against committed capital K (captures B&H init costs)."""
    prev = np.concatenate([[K], values[:-1]])
    return values / prev - 1.0


def _drawdown(values: np.ndarray, K: float) -> np.ndarray:
    peak = np.maximum.accumulate(np.concatenate([[K], values]))[1:]
    return values / peak - 1.0


def build_daily_table(spec: CheckpointSpec, cache: MarketDataCache, plans: List[Dict[str, Any]],
                      episodes: Dict[str, Dict[str, Any]], ref_eps: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    assets = list(cache.asset_names)
    K = float(spec.config["environment"]["initial_portfolio_value"])
    frames = []
    for plan in plans:
        ep = episodes[plan["block_id"]]
        ref = ref_eps[plan["block_id"]]
        T = ep["n_days"]
        if ref["n_days"] < T or not np.array_equal(ref["abs_step"][:T], ep["abs_step"]):
            raise RuntimeError(f"Reference SAA episode not aligned with agent episode for {plan['block_id']}")
        cols: Dict[str, Any] = {
            "family": spec.family, "run_id": spec.run_id, "config_id": spec.config_id, "checkpoint": spec.stem,
            "period_type": plan["period_type"], "block_id": plan["block_id"],
            "day": np.arange(T), "abs_step": ep["abs_step"],
            "date": [cache.dates[i] for i in ep["abs_step"]],
            "rf_daily": ep["rf_daily"], "effr_pa": ep["effr_pa"],
        }
        for i, a in enumerate(assets):
            cols[f"price_{a}"] = ep["prices"][:, i]

        # PAA
        cols["paa_value"] = ep["paa_value"]
        cols["paa_cash"] = ep["paa_cash"]
        cols["paa_w_cash"] = ep["paa_w"][:, 0]
        for i, a in enumerate(assets):
            cols[f"paa_w_{a}"] = ep["paa_w"][:, i + 1]
        for i, a in enumerate(assets):
            cols[f"paa_shares_{a}"] = ep["paa_shares"][:, i]
        for i, a in enumerate(assets):
            cols[f"paa_logit_{a}"] = ep["paa_logits"][:, i]
        cols["paa_target_w_cash"] = ep["paa_target_w"][:, 0]
        for i, a in enumerate(assets):
            cols[f"paa_target_w_{a}"] = ep["paa_target_w"][:, i + 1]
        cols["paa_exec_w_cash"] = ep["paa_exec_w"][:, 0]
        for i, a in enumerate(assets):
            cols[f"paa_exec_w_{a}"] = ep["paa_exec_w"][:, i + 1]
        for i, a in enumerate(assets):
            cols[f"paa_trade_notional_{a}"] = ep["paa_trade_notional"][:, i]
        traded = np.abs(ep["paa_trade_notional"]).sum(axis=1)
        cols["paa_traded_notional"] = np.where(np.isnan(ep["paa_tc"]), np.nan, traded)
        cols["paa_turnover"] = cols["paa_traded_notional"] / ep["paa_value"]
        cols["paa_exposure_pre"] = 1.0 - ep["paa_w"][:, 0]
        cols["paa_exposure_post"] = 1.0 - ep["paa_exec_w"][:, 0]
        cols["paa_exposure_change"] = cols["paa_exposure_post"] - cols["paa_exposure_pre"]
        cols["paa_tc"] = ep["paa_tc"]
        for j, name in enumerate(TC_PART_NAMES):
            cols[f"paa_tc_{name}"] = ep["paa_tc_parts"][:, j]
        cols["paa_reward"] = ep["paa_reward"]

        # Benchmarks
        cols["spy_bh_value"] = ep["spy_bh_value"]
        cols["ew_bh_value"] = ep["ew_bh_value"]
        cols["cash_value"] = ep["cash_value"]
        cols["env_benchmark_value"] = ep["env_benchmark_value"]

        # Reference SAA (equal capital split across the per-asset shadow books)
        ref_sub = ref["sub_value_mtm"][:T]
        cols["saa_ew_value"] = ref_sub.mean(axis=1)
        cols["saa_ew_exposure_post"] = ref["sub_weight_post"][:T].mean(axis=1)  # NaN on the final day
        cols["saa_ew_tc"] = ref["sub_tc"][:T].sum(axis=1) / cache.num_assets
        for i, a in enumerate(assets):
            cols[f"saa_sub_value_{a}"] = ref_sub[:, i]
        for i, a in enumerate(assets):
            cols[f"saa_signal_{a}"] = ref["sub_signal"][:T, i]
        for i, a in enumerate(assets):
            cols[f"saa_sub_weight_post_{a}"] = ref["sub_weight_post"][:T, i]
        for i, a in enumerate(assets):
            cols[f"saa_trade_notional_{a}"] = ref["sub_trade_notional"][:T, i]

        # Signal actually injected into this PAA (SAA / AR(1) / zero) and its shadow books
        inj_signal = ep["sub_signal"].copy()
        inj_w = ep["sub_weight_post"].copy()
        if spec.family == FAMILY_XSEC:
            inj_signal[:-1] = 0.0
            inj_w[:-1] = 0.0
        cols["inj_ew_value"] = ep["sub_value_mtm"].mean(axis=1)
        for i, a in enumerate(assets):
            cols[f"inj_signal_{a}"] = inj_signal[:, i]
        for i, a in enumerate(assets):
            cols[f"inj_sub_value_{a}"] = ep["sub_value_mtm"][:, i]
        for i, a in enumerate(assets):
            cols[f"inj_sub_weight_post_{a}"] = inj_w[:, i]

        # Convenience derivations (returns vs committed capital on day 0)
        for name in ("paa", "spy_bh", "ew_bh", "saa_ew", "cash", "env_benchmark"):
            v = np.asarray(cols[f"{name}_value"], dtype=np.float64)
            cols[f"{name}_ret"] = _returns_vs_capital(v, K)
            cols[f"{name}_cum_ret"] = v / K - 1.0
            cols[f"{name}_drawdown"] = _drawdown(v, K)
        cols["paa_alpha_vs_spy_daily"] = cols["paa_ret"] - cols["spy_bh_ret"]
        cols["paa_alpha_vs_saa_ew_daily"] = cols["paa_ret"] - cols["saa_ew_ret"]
        cols["paa_excess_vs_spy_cum"] = cols["paa_cum_ret"] - cols["spy_bh_cum_ret"]
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


# =====================================================================================
# Consistency checks
# =====================================================================================
def run_consistency_checks(spec: CheckpointSpec, cache: MarketDataCache, plans: List[Dict[str, Any]],
                           episodes: Dict[str, Dict[str, Any]], ref_eps: Dict[str, Dict[str, Any]],
                           df: pd.DataFrame) -> Dict[str, Any]:
    K = float(spec.config["environment"]["initial_portfolio_value"])
    report: Dict[str, Any] = {}
    for plan in plans:
        bid = plan["block_id"]
        ep = episodes[bid]
        d = df[df["block_id"] == bid]
        T = ep["n_days"]
        info = ep["terminal_info"]
        c: Dict[str, float] = {}

        c["n_days"] = T
        c["spy_bh_init_tc"] = ep["spy_bh_init_tc"]
        c["spy_bh_init_residual_cash"] = ep["spy_bh_init_cash"]
        c["ew_bh_init_tc"] = ep["ew_state"]["init_tc"]
        c["ew_bh_init_residual_cash"] = ep["ew_bh_init_cash"]
        if not ep["terminated_early"] and T != plan["reset_option"]["episode_length_override"]:
            raise RuntimeError(f"{bid}: episode length {T} != planned {plan['reset_option']['episode_length_override']}")
        c["final_pv_vs_env_info"] = abs(ep["paa_value"][-1] - info["portfolio_final_value"])
        c["spy_final_vs_env_info"] = abs(ep["spy_bh_value"][-1] - info["spy_bh_final_value"])
        c["excess_abs_vs_env_info"] = abs((ep["paa_value"][-1] - ep["spy_bh_value"][-1]) - info["excess_return_over_spy_abs"])
        c["total_tc_vs_env_info"] = abs(np.nansum(ep["paa_tc"]) - info["total_transaction_costs"])

        # Weights are a simplex every day (pre- and post-trade).
        w_pre = ep["paa_w"]
        w_post = ep["paa_exec_w"][:-1]
        c["max_weight_sum_err"] = float(max(np.abs(w_pre.sum(1) - 1).max(), np.abs(w_post.sum(1) - 1).max()))
        c["min_weight"] = float(min(w_pre.min(), w_post.min()))

        # Self-financing identity: V_{t+1} = post-trade cash * (1 + rf_{t+1}) + post-trade shares . P_{t+1}
        post_cash = ep["paa_cash"][1:] / (1.0 + ep["rf_daily"][1:])
        post_shares = ep["paa_shares"][1:]
        cash_pre_plus_trade = ep["paa_cash"][:-1] - (ep["paa_trade_shares"][:-1] * ep["prices"][:-1]).sum(1) - ep["paa_tc"][:-1]
        c["cash_roll_err"] = float(np.abs(post_cash - cash_pre_plus_trade).max())
        c["shares_roll_err"] = float(np.abs(ep["paa_shares"][:-1] + ep["paa_trade_shares"][:-1] - post_shares).max())
        v_next = post_cash * (1 + ep["rf_daily"][1:]) + (post_shares * ep["prices"][1:]).sum(1)
        c["value_roll_err"] = float(np.abs(v_next - ep["paa_value"][1:]).max())

        # Cash benchmark = K compounded with the EFFR carry; SPY B&H never trades after day 0.
        cash_expected = K * np.concatenate([[1.0], np.cumprod(1.0 + ep["rf_daily"][1:])])
        c["cash_benchmark_err"] = float(np.abs(cash_expected - ep["cash_value"]).max())

        # Compounded daily returns reproduce the terminal value.
        c["compound_ret_err"] = float(abs(np.prod(1 + d["paa_ret"].to_numpy()) * K - ep["paa_value"][-1]))
        c["saa_ew_start"] = float(d["saa_ew_value"].iloc[0])

        # Shadow-book signals: every non-terminal day must have a committed action (SAA / AR1).
        if spec.family in (FAMILY_HIERARCHICAL, FAMILY_AR1):
            c["missing_commits"] = int((~ep["sub_committed"][:-1]).sum())
        else:
            c["missing_commits"] = 0
            c["zero_family_commits"] = int(ep["sub_committed"].sum())

        # Hierarchical PAA sees exactly the reference SAA.
        if spec.family == FAMILY_HIERARCHICAL:
            ref = ref_eps[bid]
            c["inj_vs_ref_saa_value_err"] = float(np.abs(ep["sub_value_mtm"] - ref["sub_value_mtm"][:T]).max())
            c["inj_vs_ref_saa_signal_err"] = float(np.nanmax(np.abs(ep["sub_signal"][:-1] - ref["sub_signal"][:T - 1])))

        c["nan_in_value_cols"] = int(d[[k for k in d.columns if k.endswith("_value")]].isna().sum().sum())
        report[bid] = c

    tol = {"final_pv_vs_env_info": 0.05, "spy_final_vs_env_info": 0.05, "excess_abs_vs_env_info": 0.1,
           "total_tc_vs_env_info": 0.05, "max_weight_sum_err": 1e-5, "cash_roll_err": 0.5, "shares_roll_err": 1e-3,
           "value_roll_err": 0.5, "cash_benchmark_err": 0.05, "compound_ret_err": 0.05,
           "inj_vs_ref_saa_value_err": 0.05, "inj_vs_ref_saa_signal_err": 1e-5,
           "missing_commits": 0, "nan_in_value_cols": 0, "zero_family_commits": 0}
    failures = []
    for bid, c in report.items():
        for k, lim in tol.items():
            if k in c and not (c[k] <= lim):
                failures.append(f"{bid}.{k}={c[k]} > {lim}")
        if c["min_weight"] < -1e-7:
            failures.append(f"{bid}.min_weight={c['min_weight']}")
        if abs(c["saa_ew_start"] - K) > 1e-3:
            failures.append(f"{bid}.saa_ew_start={c['saa_ew_start']}")
    if failures:
        raise RuntimeError("Consistency checks failed:\n  " + "\n  ".join(failures))
    print(f"  [checks] all {len(report)} periods passed ({len(tol)} checks each)", flush=True)
    return report


# =====================================================================================
# Summaries
# =====================================================================================
def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2 or np.std(r, ddof=1) <= 0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))


def summarize(df: pd.DataFrame, K: float) -> Dict[str, Any]:
    periods: Dict[str, Dict[str, Any]] = {}
    for bid, d in df.groupby("block_id", sort=False):
        s: Dict[str, Any] = {
            "period_type": d["period_type"].iloc[0], "start_date": d["date"].iloc[0], "end_date": d["date"].iloc[-1],
            "n_days": int(len(d)),
        }
        for name in ("paa", "spy_bh", "ew_bh", "saa_ew", "cash", "env_benchmark"):
            v = d[f"{name}_value"].to_numpy()
            r = d[f"{name}_ret"].to_numpy()[1:]
            s[f"{name}_final_value"] = float(v[-1])
            s[f"{name}_total_return"] = float(v[-1] / K - 1.0)
            s[f"{name}_sharpe"] = _sharpe(r)
            s[f"{name}_max_drawdown"] = float(-d[f"{name}_drawdown"].min())
        s["paa_excess_vs_spy_pct"] = s["paa_total_return"] - s["spy_bh_total_return"]
        s["paa_excess_vs_spy_abs"] = s["paa_final_value"] - s["spy_bh_final_value"]
        s["paa_excess_vs_saa_ew_pct"] = s["paa_total_return"] - s["saa_ew_total_return"]
        s["paa_excess_vs_ew_bh_pct"] = s["paa_total_return"] - s["ew_bh_total_return"]
        s["paa_total_tc"] = float(d["paa_tc"].sum())
        s["paa_total_turnover"] = float(d["paa_turnover"].sum())
        s["paa_avg_exposure"] = float(d["paa_exposure_pre"].mean())
        s["saa_ew_total_tc"] = float(d["saa_ew_tc"].sum())
        periods[bid] = s

    agg: Dict[str, Any] = {}
    for ptype in ("validation", "test"):
        sel = [s for s in periods.values() if s["period_type"] == ptype]
        if not sel:
            continue
        keys = [k for k, v in sel[0].items() if isinstance(v, float)]
        agg[ptype] = {"n_periods": len(sel), **{f"mean_{k}": float(np.mean([s[k] for s in sel])) for k in keys}}
        agg[ptype]["min_paa_excess_vs_spy_pct"] = float(np.min([s["paa_excess_vs_spy_pct"] for s in sel]))
    return {"periods": periods, "aggregate": agg}


# =====================================================================================
# Saving
# =====================================================================================
COLUMN_DOC = {
    "day": "Trading day index t within the period (0 = first day, 100% cash start).",
    "date": "Calendar date of trading day t.",
    "rf_daily": "Daily EFFR carry rate at date t (applied to every cash balance for t >= 1).",
    "effr_pa": "Annualized EFFR (decimal) at date t.",
    "price_<A>": "Close price of asset A at t (all fills happen at this price).",
    "paa_value / paa_cash / paa_w_* / paa_shares_*": "PAA book marked-to-market at close t BEFORE the day-t rebalancing.",
    "paa_logit_<A>": "Raw deterministic policy output (asset logit; cash logit fixed at 0).",
    "paa_target_w_*": "Softmax target weights requested by the policy at t.",
    "paa_exec_w_*": "Weights right after the day-t rebalancing (soft execution: deadband + step size).",
    "paa_trade_notional_<A>": "Signed traded notional at t (+ buy, - sell), USD.",
    "paa_traded_notional / paa_turnover": "Sum of |traded notional| at t, and as a fraction of paa_value.",
    "paa_exposure_pre/post/change": "1 - cash weight before/after the rebalancing and its change (risk-on > 0).",
    "paa_tc / paa_tc_<part>": "Transaction cost of the day-t rebalancing and its commission/spread/impact/fixed parts.",
    "paa_reward": "Allocator reward received on arrival at t (for the action taken at t-1).",
    "spy_bh_value": "100% SPY buy-and-hold from day 0 incl. its initial transaction cost + cash carry on residual.",
    "ew_bh_value": "1/N equal-weight buy-and-hold of all assets from day 0 (same cost-aware init as SPY B&H).",
    "cash_value": "100% cash compounding at the EFFR carry.",
    "env_benchmark_value": "TradingEnv fixed-weight benchmark (45% SPY, 20% Gold, ...), buy-and-hold.",
    "saa_ew_value": "Frozen reference SAA managing the PAA's capital split 1/N across its per-asset shadow books.",
    "saa_sub_value_<A> / saa_signal_<A> / saa_sub_weight_post_<A> / saa_trade_notional_<A>":
        "Reference SAA shadow book of asset A (each started with the full capital): MTM value at t, committed "
        "(scaled) target_position_change at t, asset holding share after the day-t SAA trade, signed traded notional.",
    "saa_ew_tc / saa_ew_exposure_post": "Mean shadow-book transaction cost at t and mean holding share after the SAA trade.",
    "inj_*": "Same as saa_* but for the signal actually injected into this PAA (SAA / AR(1) / zero).",
    "<name>_ret / _cum_ret / _drawdown": "Daily simple return (day 0 vs committed capital), cumulative return vs capital, "
                                        "drawdown from running peak (capital included).",
    "paa_alpha_vs_spy_daily / paa_alpha_vs_saa_ew_daily": "Daily return differences.",
    "paa_excess_vs_spy_cum": "Cumulative return difference PAA - SPY B&H.",
}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return _rel(o)
    raise TypeError(type(o))


def save_results(res: Dict[str, Any]) -> Path:
    spec: CheckpointSpec = res["spec"]
    out_dir = spec.run_dir / OUTPUT_FOLDER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{spec.stem}_daily.csv"
    res["df"].to_csv(csv_path, index=False, float_format="%.15g")

    cache: MarketDataCache = res["cache"]
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "family": spec.family,
        "family_label": FAMILY_LABELS[spec.family],
        "run_id": spec.run_id,
        "config_id": spec.config_id,
        "run_date": spec.run_date,
        "run_dir": _rel(spec.run_dir),
        "checkpoint": spec.stem,
        "checkpoint_zip": _rel(spec.zip_path),
        "vecnormalize_pkl": _rel(spec.vecnorm_path),
        "config_path": _rel(spec.config_path),
        "config_sha256": hashlib.sha256(spec.config_path.read_bytes()).hexdigest(),
        "config": spec.config,
        "inference_overrides": {"environment.percentage_of_cash_only_starts": 1.0, "device": DEVICE,
                                "policy": "deterministic"},
        "injected_signal": res["wrapper_info"],
        "reference_saa": res["ref_info"],
        "initial_capital": float(spec.config["environment"]["initial_portfolio_value"]),
        "assets": list(cache.asset_names),
        "periods": [{"period_type": p["period_type"], **p["reset_option"],
                     "start_date": res["summaries"]["periods"][p["block_id"]]["start_date"],
                     "end_date": res["summaries"]["periods"][p["block_id"]]["end_date"]} for p in res["plans"]],
        "daily_table": csv_path.name,
        "summary": res["summaries"],
        "consistency_checks": res["checks"],
        "column_doc": COLUMN_DOC,
        "notes": [
            "Timing: the PAA observes the close of day t, rebalances at close-t prices, and the new book earns the "
            "day t -> t+1 move plus EFFR carry on cash. Values in the table are pre-rebalancing marks at close t.",
            "Returns/cumulative returns are measured against the committed capital, so buy-and-hold initial "
            "transaction costs are included on day 0.",
            "SPY and equal-weight buy-and-hold use TradingEnv._initialize_portfolio_with_costs (the routine behind the "
            "SPY benchmark the PAA was trained/selected against); its 0.99 sizing buffer leaves ~1% residual cash that "
            "earns the EFFR carry (see consistency_checks.*.*_init_residual_cash).",
            "saa_ew_value is the mean of N shadow books each started with the full capital; this equals a 1/N capital "
            "split up to the sqrt market-impact cost term, which is not scale invariant.",
            "AR(1) signals are stochastic; they are reproducible for a fixed seed (see injected_signal.ar1_seed).",
            "TradingEnv stores the shadow books in float32 (relative precision ~1e-5 over a period).",
        ],
    }
    meta_path = out_dir / f"{spec.stem}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    print(f"  [saved] {_rel(csv_path)} ({len(res['df'])} rows x {res['df'].shape[1]} cols)\n  [saved] {_rel(meta_path)}", flush=True)
    return out_dir


# =====================================================================================
# 3-panel figure
# =====================================================================================
MIN_FONT_PT = 10.0


def _apply_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "legend.title_fontsize": 11,
        "figure.titlesize": 13, "axes.edgecolor": "#444444", "grid.linewidth": 0.5, "grid.alpha": 0.5,
        "savefig.dpi": GRAPH_DPI, "pdf.fonttype": 42,
    })


def _asset_colors(assets: List[str]) -> List[str]:
    fallback = iter(sns.color_palette("colorblind").as_hex())
    return [ASSET_COLORS.get(a) or next(fallback) for a in assets]


def plot_period(d: pd.DataFrame, assets: List[str], spec: CheckpointSpec, out_dir: Path) -> List[Path]:
    _apply_plot_style()
    d = d.reset_index(drop=True)
    T = len(d)
    x = np.arange(T)
    block_id = d["block_id"].iloc[0]
    ptype = d["period_type"].iloc[0]
    dates = d["date"].tolist()

    fig = plt.figure(figsize=(8.27, 11.69), layout="constrained")
    gs = fig.add_gridspec(5, 1, height_ratios=[4.6, 3.0, 2.0, 0.42, 0.16])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    axh = fig.add_subplot(gs[3], sharex=ax1)
    axc = fig.add_subplot(gs[4])

    # ---------------- Panel 1: cumulative performance + daily alpha inset
    series = [
        ("paa_cum_ret", "PAA portfolio", "#000000", "-", 2.2),
        ("spy_bh_cum_ret", "SPY buy & hold", "#0072B2", "-", 1.5),
        ("ew_bh_cum_ret", "Equal-weight buy & hold", "#009E73", "-", 1.5),
        ("saa_ew_cum_ret", "SAA equal-weight", "#CC79A7", "--", 1.6),
        ("cash_cum_ret", "100% cash (EFFR)", "#7F7F7F", ":", 1.8),
    ]
    all_vals = []
    for col, label, color, ls, lw in series:
        y = d[col].to_numpy() * 100.0
        all_vals.append(y)
        ax1.plot(x, y, color=color, ls=ls, lw=lw, label=label, zorder=3 if col == "paa_cum_ret" else 2)
    all_vals = np.concatenate(all_vals)
    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    span = max(hi - lo, 1e-6)
    top = hi + 0.06 * span
    # Reserve the bottom 30% of the panel for the alpha inset so the two never overlap.
    ax1.set_ylim(top - (top - (lo - 0.04 * span)) / 0.70, top)
    yt = mticker.MaxNLocator(nbins=6).tick_values(lo, hi)
    ax1.set_yticks([v for v in yt if lo - 0.04 * span <= v <= top])
    ax1.axhline(0.0, color="#444444", lw=0.8, zorder=1)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax1.set_ylabel("Cumulative return")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.2), mode="expand", ncol=3, frameon=False,
               title="(a) Cumulative performance and daily alpha", alignment="left", borderaxespad=0.0)
    ax1.tick_params(labelbottom=False)

    axin = ax1.inset_axes([0.0, 0.0, 1.0, 0.27], sharex=ax1)
    alpha_bp = d["paa_alpha_vs_spy_daily"].to_numpy() * 1e4
    alpha_bp[0] = np.nan  # day 0 only reflects SPY's initial transaction cost, not a market move
    alpha_plot = np.nan_to_num(alpha_bp)
    axin.bar(x, alpha_plot, width=1.0, linewidth=0, color=np.where(alpha_plot >= 0, "#1a9850", "#d73027"))
    axin.axhline(0.0, color="#444444", lw=0.6)
    a_lim = max(float(np.abs(alpha_plot).max()), 1e-6)
    a_tick = float(mticker.MaxNLocator(nbins=2).tick_values(0, a_lim)[1])
    axin.set_ylim(-a_lim * 1.08, a_lim * 1.45)  # headroom for the inset label
    axin.set_yticks([-a_tick, 0.0, a_tick])
    axin.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    axin.patch.set_alpha(0.0)
    axin.grid(False)
    axin.spines["top"].set_visible(True)
    axin.spines["top"].set_linestyle(":")
    axin.yaxis.tick_right()
    axin.yaxis.set_label_position("right")
    axin.set_ylabel("bp", fontsize=10)
    axin.tick_params(labelbottom=False, labelsize=10)
    axin.text(0.005, 0.97, "Daily alpha PAA - SPY", transform=axin.transAxes, ha="left", va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8))

    # ---------------- Panel 2: 100% stacked composition (post-rebalancing weights)
    w_cols = ["paa_exec_w_cash"] + [f"paa_exec_w_{a}" for a in assets]
    pre_cols = ["paa_w_cash"] + [f"paa_w_{a}" for a in assets]
    W = d[w_cols].to_numpy().copy()
    W[-1] = d[pre_cols].to_numpy()[-1]  # no rebalancing on the final day
    W = np.clip(W, 0.0, None)
    W = W / W.sum(axis=1, keepdims=True)
    edges = np.arange(T + 1) - 0.5
    W_step = np.vstack([W, W[-1:]])  # step='post' needs the right edge of the last bar
    colors = [CASH_COLOR] + _asset_colors(assets)
    ax2.stackplot(edges, W_step.T * 100.0, labels=["Cash"] + assets, colors=colors, step="post",
                  linewidth=0.0, antialiased=False)
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_ylabel("Portfolio weight")
    ax2.grid(False)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.2), mode="expand", ncol=6, frameon=False,
               title="(b) Portfolio composition after daily rebalancing", alignment="left", borderaxespad=0.0,
               handlelength=1.2, columnspacing=0.8)
    ax2.tick_params(labelbottom=False)

    # ---------------- Panel 3: turnover bars coloured by exposure direction + heat strip
    turnover = d["paa_turnover"].to_numpy() * 100.0
    dexp = d["paa_exposure_change"].to_numpy() * 100.0  # percentage points
    eps_pp = 0.01
    bar_colors = np.where(dexp > eps_pp, "#1a9850", np.where(dexp < -eps_pp, "#d73027", "#9E9E9E"))
    ax3.bar(x, np.nan_to_num(turnover), width=1.0, linewidth=0, color=bar_colors)
    ax3.set_ylabel("Turnover (% NAV)")
    ax3.set_ylim(0, max(np.nanmax(turnover) * 1.08, 1e-3))
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax3.legend(handles=[Patch(color="#1a9850", label="Risk-on (equity exposure up)"),
                        Patch(color="#d73027", label="De-risk into cash"),
                        Patch(color="#9E9E9E", label="Reallocation only")],
               loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.2), mode="expand", ncol=3, frameon=False,
               title="(c) Trade signal velocity and rebalancing intensity", alignment="left", borderaxespad=0.0,
               handlelength=1.2)
    ax3.tick_params(labelbottom=False)

    # Scale on days >= 1: the day-0 move out of the 100% cash start would wash out the strip.
    later = np.abs(dexp[1:])
    later = later[np.isfinite(later)]
    vmax = float(np.percentile(later, 95)) if later.size else 1.0
    vmax = max(vmax, 1e-6)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("white")
    im = axh.imshow(np.ma.masked_invalid(dexp)[None, :], aspect="auto", cmap=cmap,
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    extent=(-0.5, T - 0.5, 0, 1), interpolation="nearest")
    axh.set_yticks([])
    axh.set_ylabel("Δ exp.", rotation=0, ha="right", va="center")
    axh.grid(False)
    axh.set_xlabel("Trading day")
    axh.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
    axh.set_xlim(-0.5, T - 0.5)
    cb = fig.colorbar(im, cax=axc, orientation="horizontal", extend="both")
    cb.set_label("Daily change in equity exposure (percentage points, clipped at 95th pct.)", fontsize=10)
    cb.ax.tick_params(labelsize=10)

    fam = FAMILY_LABELS[spec.family]
    fig.suptitle(f"{fam}\nrun {spec.run_id} (config {spec.config_id}), checkpoint {spec.stem}\n"
                 f"{ptype.capitalize()} period {block_id}: {dates[0]} to {dates[-1]} ({T} trading days)",
                 fontsize=12)

    # Font-size guarantee for the printed A4 page.
    fig.canvas.draw()
    # Freeze the solved layout so savefig (other dpi/backend) cannot re-solve it differently.
    fig.set_layout_engine("none")
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_FONT_PT]
    if small:
        raise RuntimeError(f"Text below {MIN_FONT_PT}pt in figure: {small[:10]}")
    w_in, h_in = fig.get_size_inches()
    if abs(w_in - 8.27) > 1e-6 or abs(h_in - 11.69) > 1e-6:
        raise RuntimeError(f"Figure is not A4 portrait: {w_in} x {h_in} in")

    base = out_dir / f"{spec.stem}__{ptype}_{block_id}_3panel"
    paths = [base.with_suffix(".png"), base.with_suffix(".pdf")]
    for p in paths:
        fig.savefig(p)  # no bbox_inches="tight": keep the exact A4 page size
    plt.close(fig)
    return paths


# =====================================================================================
# Main
# =====================================================================================
def main() -> None:
    os.chdir(REPO_ROOT)  # SAA loader resolves saa_base_dir relative to the repo root
    torch.set_grad_enabled(False)
    specs: List[CheckpointSpec] = []
    for family, paths in ((FAMILY_HIERARCHICAL, HIERARCHICAL_PAA_CHECKPOINTS),
                          (FAMILY_AR1, CONTROL_ABLATION_AR1_CHECKPOINTS),
                          (FAMILY_XSEC, CROSS_SECTIONAL_ABLATION_CHECKPOINTS)):
        for p in paths:
            specs.append(resolve_checkpoint(family, p))
    if not specs:
        raise ValueError("No checkpoints configured.")

    results: List[Dict[str, Any]] = []
    signatures = {}
    for spec in specs:
        res = evaluate_checkpoint(spec)
        signatures[spec.label] = _block_signature(res["cache"])
        save_results(res)
        results.append(res)
    if len({json.dumps(s) for s in signatures.values()}) > 1:
        print("[WARNING] Not all checkpoints were evaluated on identical validation/test blocks:")
        for k, s in signatures.items():
            print(f"  {k}: {s}")

    print(f"\n{'=' * 90}\nMean validation market alpha (PAA total return - SPY B&H total return)\n{'=' * 90}")
    best: Dict[str, Dict[str, Any]] = {}
    for res in results:
        agg = res["summaries"]["aggregate"]
        val_alpha = agg["validation"]["mean_paa_excess_vs_spy_pct"]
        test_alpha = agg.get("test", {}).get("mean_paa_excess_vs_spy_pct", float("nan"))
        spec = res["spec"]
        print(f"  {spec.family:<26s} {spec.label:<70s} val {val_alpha * 100:+7.2f}%   test {test_alpha * 100:+7.2f}%")
        if spec.family not in best or val_alpha > best[spec.family]["summaries"]["aggregate"]["validation"]["mean_paa_excess_vs_spy_pct"]:
            best[spec.family] = res

    if MAKE_GRAPHS:
        for family, res in best.items():
            spec = res["spec"]
            out_dir = spec.run_dir / OUTPUT_FOLDER_NAME
            print(f"\n[graphs] best {family}: {spec.label}")
            for bid, d in res["df"].groupby("block_id", sort=False):
                for p in plot_period(d, list(res["cache"].asset_names), spec, out_dir):
                    print(f"  [saved] {_rel(p)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
