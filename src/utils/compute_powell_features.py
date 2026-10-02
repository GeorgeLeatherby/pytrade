# Compute stats for Powell-rule features and propose theta bounds and initial guess ranges.
import pandas as pd
import numpy as np

CSV_PATH = r"C:\Users\HansenSimonO\Documents\Coding\PyTradeTwo\pytrade-two\src\data\enriched_financial_data.csv"

FEATURES = [
    "return_5d",
    "momentum_10",
    "bb_width",
    "return_1d",
]

def robust_min(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    # clip extreme tails to reduce outlier impact
    return float(np.quantile(s, 0.01))

def robust_max(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(np.quantile(s, 0.99))

def describe_feature(df: pd.DataFrame, col: str):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return {"mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan, "robust_min": np.nan, "robust_max": np.nan}
    return {
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "robust_min": robust_min(s),
        "robust_max": robust_max(s),
    }

def propose_bounds_and_inits(stats: dict):
    """
    Based on robust ranges with safety margins:
      - th_r5_up, th_mom10_up: within [robust_min - 0.05, robust_max + 0.05]
      - th_bw_min: [max(0, robust_min - 0.05), robust_max + 0.10] (bandwidth ≥ 0)
      - th_r1_down: [robust_min - 0.05, robust_max + 0.05]
      - sl_pct: [0.02, 0.20]  (stop-loss sensible positive range)
      - tp_pct: [0.05, 0.40]  (take-profit sensible positive range)
      - qty_shares: [10, 1000] (allow range; tune per data)
    Initial guess ranges sit inside bounds with moderate slack to ensure trading.
    """
    r5 = stats["return_5d"]
    mom10 = stats["momentum_10"]
    bw = stats["bb_width"]
    r1 = stats["return_1d"]

    def pad(lo, hi, p_lo=0.05, p_hi=0.05):
        return lo - p_lo, hi + p_hi

    # Threshold bounds from data (robust)
    r5_lo, r5_hi = pad(r5["robust_min"], r5["robust_max"])
    mom_lo, mom_hi = pad(mom10["robust_min"], mom10["robust_max"])
    bw_lo = max(0.0, bw["robust_min"] - 0.05)  # bandwidth cannot sensibly be negative
    bw_hi = bw["robust_max"] + 0.10
    r1_lo, r1_hi = pad(r1["robust_min"], r1["robust_max"])

    theta_bounds = {
        "th_r5_up": [round(r5_lo, 4), round(r5_hi, 4)],
        "th_mom10_up": [round(mom_lo, 4), round(mom_hi, 4)],
        "th_bw_min": [round(bw_lo, 4), round(bw_hi, 4)],
        "th_r1_down": [round(r1_lo, 4), round(r1_hi, 4)],
        "sl_pct": [0.02, 0.20],
        "tp_pct": [0.05, 0.40],
        "qty_shares": [10.0, 1000.0],
    }

    # Initial guesses: keep inside bounds and not too restrictive
    init_ranges = {
        "th_r5_up": [round(r5["median"] - 0.02, 4), round(r5["median"] + 0.06, 4)],
        "th_mom10_up": [round(mom10["median"] - 0.02, 4), round(mom10["median"] + 0.06, 4)],
        "th_bw_min": [round(max(0.0, bw["median"] - 0.05), 4), round(bw["median"] + 0.10, 4)],
        "th_r1_down": [round(r1["median"] - 0.03, 4), round(r1["median"] + 0.03, 4)],
        "sl_pct": [0.05, 0.12],
        "tp_pct": [0.10, 0.30],
        "qty_shares": [50.0, 300.0],
    }

    return theta_bounds, init_ranges

def main():
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"Missing features in CSV: {missing}")
        return

    print("Feature statistics (median, mean, min, max, robust_min@1%, robust_max@99%):")
    stats = {}
    for f in FEATURES:
        st = describe_feature(df, f)
        stats[f] = st
        print(f"  {f}: median={st['median']:.6f}, mean={st['mean']:.6f}, min={st['min']:.6f}, max={st['max']:.6f}, "
              f"robust_min={st['robust_min']:.6f}, robust_max={st['robust_max']:.6f}")

    theta_bounds, init_ranges = propose_bounds_and_inits(stats)

    print("\nProposed theta bounds (for config.theta_bounds):")
    for k, v in theta_bounds.items():
        print(f"  {k}: {v}")

    print("\nProposed initial guess ranges (for config.initial_guess_ranges):")
    for k, v in init_ranges.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()