"""Compute per-asset SAA performance metrics from raw inference-test JSON output.

Reads the two raw SAA inference-test JSON files produced for a given checkpoint
(``asset-specific_saa_portfolio_test_<X>.json`` and
``aggregate_saa_portfolio_test_<X>.json``), computes a standard set of
risk/return metrics per (asset, validation period), aggregates them into
per-period and overall (macro) averages, and writes the computed metrics back
out as ``computed_tabular_report_data_<X>.json`` next to the source files.

This script only computes and saves the metrics as JSON; it does not generate
or modify any LaTeX. Rendering the results as a table in the thesis is done
separately (by hand / by an LLM) from this JSON output.

Usage:
    python src/utils/generate_saa_performance_table.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = (
    REPO_ROOT
    / "src"
    / "agents"
    / "RecurrPPO_target_position_agent"
    / "saved_models"
    / "00239_config_01058_26_08_13"
    / "saa_inference_test_results"
)

# Fixed display order (matches asset ordering in the source JSON files).
ASSET_ORDER = ["Crude", "EWG", "EWH", "EWJ", "EWQ", "EWS", "EWT", "EWU", "EWY", "Gold", "SPY"]

METRIC_COLUMNS = [
    "agent_cagr_pct",
    "bh_cagr_pct",
    "asset_alpha_pct",
    "market_alpha_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown_pct",
    "calmar_ratio",
    "total_turnover_pct",
    "txn_cost_pct_of_capital",
]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def find_result_identifier(results_dir: Path) -> str:
    """Locate the <X> identifier shared by the asset-specific/aggregate JSON pair."""
    matches = sorted(results_dir.glob("asset-specific_saa_portfolio_test_*.json"))
    if not matches:
        raise FileNotFoundError(f"No asset-specific SAA results found in {results_dir}")
    latest = matches[-1]
    prefix = "asset-specific_saa_portfolio_test_"
    return latest.stem[len(prefix):]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #

def annualized_cagr_pct(total_return_pct: float, n_trading_days: int) -> float:
    """Annualize a total-return percentage realized over n_trading_days elapsed."""
    if n_trading_days <= 0:
        return float("nan")
    growth = 1.0 + total_return_pct / 100.0
    if growth <= 0:
        return float("nan")
    return (growth ** (TRADING_DAYS_PER_YEAR / n_trading_days) - 1.0) * 100.0


def daily_returns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[1:] / values[:-1] - 1.0


def annualized_sharpe(returns: np.ndarray) -> float:
    if returns.size == 0 or returns.std(ddof=0) == 0:
        return float("nan")
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / returns.std(ddof=0))


def annualized_sortino(returns: np.ndarray) -> float:
    downside = returns[returns < 0]
    if downside.size == 0:
        return float("nan")
    downside_std = downside.std(ddof=0)
    if downside_std == 0:
        return float("nan")
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / downside_std)


def max_drawdown_pct(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    return float(drawdown.min() * 100.0)


def calmar_ratio(cagr_pct: float, mdd_pct: float) -> float:
    if not mdd_pct or np.isnan(mdd_pct) or np.isnan(cagr_pct):
        return float("nan")
    return float(cagr_pct / abs(mdd_pct))


def total_turnover_pct(actions: np.ndarray) -> float:
    """Sum of |w_t - w_(t-1)|, where w_t is the agent's target position weight, starting flat."""
    actions = np.asarray(actions, dtype=float)
    weights = np.concatenate(([0.0], actions))
    return float(np.abs(np.diff(weights)).sum() * 100.0)


# --------------------------------------------------------------------------- #
# Per-asset / per-period computation
# --------------------------------------------------------------------------- #

def compute_asset_period_metrics(asset_record: dict) -> dict:
    n_days = len(asset_record["dates"]) - 1
    portfolio_values = np.asarray(asset_record["portfolio_values"], dtype=float)
    initial_value = asset_record["initial_portfolio_value"]

    agent_cagr = annualized_cagr_pct(asset_record["total_return_pct"], n_days)
    bh_cagr = annualized_cagr_pct(asset_record["bh_total_return_pct"], n_days)
    mdd = max_drawdown_pct(portfolio_values)
    returns = daily_returns(portfolio_values)

    return {
        "agent_cagr_pct": agent_cagr,
        "bh_cagr_pct": bh_cagr,
        "asset_alpha_pct": agent_cagr - bh_cagr,
        # market_alpha_pct filled in later once the period's SPY B&H CAGR is known
        "sharpe_ratio": annualized_sharpe(returns),
        "sortino_ratio": annualized_sortino(returns),
        "max_drawdown_pct": mdd,
        "calmar_ratio": calmar_ratio(agent_cagr, mdd),
        "total_turnover_pct": total_turnover_pct(asset_record["actions"]),
        "txn_cost_pct_of_capital": asset_record["cumulative_transaction_costs"] / initial_value * 100.0,
        "n_trading_days": n_days,
    }


def build_per_asset_period_table(asset_specific_data: list[dict]) -> pd.DataFrame:
    rows = []
    for episode in asset_specific_data:
        block_id = episode["block_id"]
        date_range = episode["date_range"]
        assets = episode["assets"]

        # The market benchmark ("market alpha") is SPY's own buy-and-hold CAGR for this period,
        # so that for the SPY row itself market_alpha_pct == asset_alpha_pct by construction.
        spy_record = assets["SPY"]
        spy_n_days = len(spy_record["dates"]) - 1
        spy_bh_cagr = annualized_cagr_pct(spy_record["bh_total_return_pct"], spy_n_days)

        for asset_name in ASSET_ORDER:
            metrics = compute_asset_period_metrics(assets[asset_name])
            metrics["market_alpha_pct"] = metrics["agent_cagr_pct"] - spy_bh_cagr
            rows.append({"block_id": block_id, "date_range": date_range, "asset": asset_name, **metrics})

    ordered_cols = ["block_id", "date_range", "asset"] + METRIC_COLUMNS + ["n_trading_days"]
    return pd.DataFrame(rows)[ordered_cols]


def compute_period_averages(per_asset_period: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_asset_period.groupby(["block_id", "date_range"], sort=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
    )
    return grouped


def compute_overall_average(period_averages: pd.DataFrame) -> dict:
    return period_averages[METRIC_COLUMNS].mean().to_dict()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    identifier = find_result_identifier(RESULTS_DIR)
    asset_specific_path = RESULTS_DIR / f"asset-specific_saa_portfolio_test_{identifier}.json"
    aggregate_path = RESULTS_DIR / f"aggregate_saa_portfolio_test_{identifier}.json"
    output_path = RESULTS_DIR / f"computed_tabular_report_data_{identifier}.json"

    asset_specific_data = load_json(asset_specific_path)
    _ = load_json(aggregate_path)  # loaded for structural validation / potential cross-checks

    per_asset_period = build_per_asset_period_table(asset_specific_data)
    period_averages = compute_period_averages(per_asset_period)
    overall_average = compute_overall_average(period_averages)

    output_payload = {
        "source_identifier": identifier,
        "per_asset_period_metrics": per_asset_period.to_dict(orient="records"),
        "period_averages": period_averages.to_dict(orient="records"),
        "overall_average_metrics": overall_average,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2)
    print(f"Wrote computed metrics to {output_path}")


if __name__ == "__main__":
    main()
