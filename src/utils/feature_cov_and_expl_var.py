import matplotlib.pyplot as plt
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression

# Paths
CSV_PATH = r"C:\Users\HansenSimonO\Documents\Coding\PyTradeTwo\pytrade-two\src\data\enriched_financial_data.csv"
OUT_DIR = Path(r"C:\Users\HansenSimonO\Documents\Coding\PyTradeTwo\pytrade-two\src\utils")

# Output files
COV_OUT = OUT_DIR / "feature_covariance_matrix.csv"
EXPL_VAR_OUT = OUT_DIR / "explained_variance_by_feature.csv"
SIMILARITY_OUT = OUT_DIR / "feature_statistical_similarity_report.csv"

# Asset column candidates to detect automatically
ASSET_COL_CANDIDATES = ["Symbol", "Asset", "Ticker", "Instrument", "Name"]

def _find_asset_column(df: pd.DataFrame) -> Optional[str]:
    for c in ASSET_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return all columns after 'Volume'. Assumes core market columns are up to 'Volume',
    and engineered features begin after.
    """
    if "Volume" not in df.columns:
        raise ValueError("Column 'Volume' not found. Cannot slice features after Volume.")
    vol_idx = df.columns.get_loc("Volume")
    # Features are all columns strictly after 'Volume'
    feature_cols = list(df.columns[(vol_idx + 1):])
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after 'Volume'.")
    return feature_cols

def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric where possible and drop rows with non-finite Close."""
    # Ensure Close exists
    if "Close" not in df.columns:
        raise ValueError("Column 'Close' not found in CSV data.")

    # Coerce feature columns and Close to numeric, preserving asset labels
    df = df.copy()
    # Attempt numeric conversion for all columns; non-numeric will become NaN
    for col in df.columns:
        if col != "Close":
            # Skip obvious non-numeric labels like asset/timestamp; we'll handle later
            pass
    # Enforce Close numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    # Drop rows where Close is non-finite
    df = df[np.isfinite(df["Close"])]
    return df

def _compute_covariance_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # Coerce features to numeric and drop rows with any non-finite feature values for covariance calc
    feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    cov = feat_df.cov()
    return cov

def _explained_variance_close_by_feature(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Compute R^2 of a simple linear model: Close ~ feature, one feature at a time.
    Returns a DataFrame sorted by R^2 descending.
    """
    results = []
    # Prepare y
    y = pd.to_numeric(df["Close"], errors="coerce").values
    finite_mask_y = np.isfinite(y)

    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce").values
        finite_mask_x = np.isfinite(x)
        m = finite_mask_y & finite_mask_x
        if m.sum() < 20:
            # Not enough data to make a meaningful estimate
            results.append({"feature": col, "r2": np.nan, "n": int(m.sum())})
            continue

        X = x[m].reshape(-1, 1)
        Y = y[m]
        # Suppress potential convergence warnings for degenerate features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = LinearRegression()
                model.fit(X, Y)
                r2 = float(model.score(X, Y))
            except Exception:
                r2 = np.nan

        results.append({"feature": col, "r2": r2, "n": int(m.sum())})

    out = pd.DataFrame(results).sort_values(by="r2", ascending=False)
    return out

def _pairwise_ks_similarity(df: pd.DataFrame, asset_col: str, feature_cols: List[str]) -> pd.DataFrame:
    """
    For each feature, perform pairwise KS tests across assets and report:
    - number of asset pairs tested
    - proportion of pairs where KS-test fails to reject equality (p >= alpha)
    - average KS statistic across pairs
    - per-asset mean and std (to gauge scale similarity)

    This helps decide if features are statistically similar across assets.
    """
    alpha = 0.05
    assets = [a for a in df[asset_col].dropna().unique().tolist()]
    assets = [a for a in assets if str(a) != ""]
    if len(assets) < 2:
        # Not enough assets to compare
        return pd.DataFrame([{
            "feature": f,
            "n_pairs": 0,
            "prop_pairs_similar": np.nan,
            "avg_ks_stat": np.nan,
            "mean_cv_across_assets": np.nan,
            "std_cv_across_assets": np.nan
        } for f in feature_cols])

    results = []

    # Pre-split by asset
    by_asset: Dict[str, pd.DataFrame] = {a: df[df[asset_col] == a] for a in assets}

    # Build all pairs
    pairs: List[Tuple[str, str]] = []
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            pairs.append((assets[i], assets[j]))

    for f in feature_cols:
        ks_stats = []
        similar_flags = []
        # per-asset mean/std for CV calculation
        per_asset_means = []
        per_asset_stds = []

        for a in assets:
            vals = pd.to_numeric(by_asset[a][f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) > 0:
                per_asset_means.append(float(vals.mean()))
                per_asset_stds.append(float(vals.std(ddof=1)))
            else:
                per_asset_means.append(np.nan)
                per_asset_stds.append(np.nan)

        for (a1, a2) in pairs:
            v1 = pd.to_numeric(by_asset[a1][f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            v2 = pd.to_numeric(by_asset[a2][f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            # Require minimum length for stable KS
            if len(v1) < 20 or len(v2) < 20:
                continue
            try:
                stat, pval = ks_2samp(v1.values, v2.values, alternative="two-sided", mode="auto")
                ks_stats.append(float(stat))
                similar_flags.append(bool(pval >= alpha))
            except Exception:
                # If KS fails, skip this pair
                continue

        n_pairs = len(ks_stats)
        prop_sim = (np.mean(similar_flags) if similar_flags else np.nan)
        avg_ks = (np.mean(ks_stats) if ks_stats else np.nan)

        # Coefficient of variation across assets for mean/std (lower is more similar)
        means_arr = np.array(per_asset_means, dtype=float)
        stds_arr = np.array(per_asset_stds, dtype=float)
        means_arr = means_arr[np.isfinite(means_arr)]
        stds_arr = stds_arr[np.isfinite(stds_arr)]

        def _cv(arr: np.ndarray) -> float:
            if arr.size == 0:
                return np.nan
            mu = float(np.mean(arr))
            sigma = float(np.std(arr, ddof=1))
            if mu == 0:
                return np.nan
            return float(abs(sigma / mu))

        mean_cv = _cv(means_arr)
        std_cv = _cv(stds_arr)

        results.append({
            "feature": f,
            "n_pairs": int(n_pairs),
            "prop_pairs_similar": float(prop_sim) if np.isfinite(prop_sim) else np.nan,
            "avg_ks_stat": float(avg_ks) if np.isfinite(avg_ks) else np.nan,
            "mean_cv_across_assets": float(mean_cv) if np.isfinite(mean_cv) else np.nan,
            "std_cv_across_assets": float(std_cv) if np.isfinite(std_cv) else np.nan,
        })

    return pd.DataFrame(results).sort_values(by=["prop_pairs_similar", "avg_ks_stat"], ascending=[False, True])

def main():
    # Load data
    if not Path(CSV_PATH).exists():
        print(f"CSV not found at: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df = _clean_numeric(df)

    # Detect asset column (optional)
    asset_col = _find_asset_column(df)
    if asset_col is None:
        print("No asset column detected; proceeding without cross-asset similarity checks.")
    else:
        print(f"Detected asset column: {asset_col}")

    # Extract feature columns
    feature_cols = _get_feature_columns(df)
    print(f"Found {len(feature_cols)} feature columns after 'Volume'.")

    # Explained variance (R²) of Close by each feature
    expl_var_df = _explained_variance_close_by_feature(df, feature_cols)
    expl_var_df.to_csv(EXPL_VAR_OUT, index=False)
    print(f"Saved explained variance table to: {EXPL_VAR_OUT}")

    # Statistical similarity across assets for all features
    if asset_col is not None:
        sim_df = _pairwise_ks_similarity(df, asset_col, feature_cols)
        sim_df.to_csv(SIMILARITY_OUT, index=False)
        print(f"Saved feature statistical similarity report to: {SIMILARITY_OUT}")

    # Visualisation of Covariance matrix per asset
    if asset_col is not None:
        assets = [a for a in df[asset_col].dropna().unique().tolist() if str(a) != ""]
        
        for asset in assets:
            asset_df = df[df[asset_col] == asset]
            asset_cov_df = _compute_covariance_matrix(asset_df, feature_cols)
            
            # Save per-asset covariance matrix
            asset_cov_file = OUT_DIR / f"feature_covariance_matrix_{asset}.csv"
            asset_cov_df.to_csv(asset_cov_file, index=True)
            # print(f"Saved covariance matrix for {asset} to: {asset_cov_file}")
            
            # Visualize per-asset covariance matrix
            plt.figure(figsize=(12, 10))
            plt.imshow(asset_cov_df, cmap='viridis', aspect='auto')
            plt.colorbar(label='Covariance')
            plt.title(f"Feature Covariance Matrix - {asset}")
            plt.xticks(range(len(asset_cov_df.columns)), asset_cov_df.columns, rotation=90, fontsize=6)
            plt.yticks(range(len(asset_cov_df.index)), asset_cov_df.index, fontsize=6)
            plt.xlabel("Features")
            plt.ylabel("Features")
            plt.grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
            plt.tight_layout()
    else:
        print("Skipping per-asset covariance visualization due to missing asset column.")

    #Visualisation of Explained Variance
    plt.figure(figsize=(10, 6))
    plt.bar(expl_var_df["feature"], expl_var_df["r2"])
    plt.xticks(rotation=90)
    plt.title("Explained Variance (R²) of Close by Feature")
    plt.xlabel("Feature")
    plt.ylabel("R²")
    plt.tight_layout()

    # Visualisation of Asset-to-Asset Similarity
    if asset_col is not None:
        # Compute overall similarity between each pair of assets across all features
        assets = [a for a in df[asset_col].dropna().unique().tolist() if str(a) != ""]
        
        if len(assets) >= 2:
            # Create asset similarity matrix
            n_assets = len(assets)
            similarity_matrix = np.zeros((n_assets, n_assets))
            
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i == j:
                        similarity_matrix[i, j] = 1.0  # Perfect similarity with self
                    elif i < j:
                        # Compute average similarity across all features
                        similarities = []
                        df1 = df[df[asset_col] == asset1]
                        df2 = df[df[asset_col] == asset2]
                        
                        for feat in feature_cols:
                            v1 = pd.to_numeric(df1[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                            v2 = pd.to_numeric(df2[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                            
                            if len(v1) >= 20 and len(v2) >= 20:
                                try:
                                    _, pval = ks_2samp(v1.values, v2.values)
                                    similarities.append(pval)  # Higher p-value = more similar
                                except:
                                    pass
                        
                        avg_sim = np.mean(similarities) if similarities else 0.0
                        similarity_matrix[i, j] = avg_sim
                        similarity_matrix[j, i] = avg_sim  # Symmetric
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(similarity_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            plt.colorbar(label='Average KS p-value (Similarity)')
            plt.title("Asset-to-Asset Similarity Based on Features")
            plt.xticks(range(n_assets), assets, rotation=45, ha='right')
            plt.yticks(range(n_assets), assets)
            plt.xlabel("Asset")
            plt.ylabel("Asset")
            plt.tight_layout()

    # Show plots
    plt.show()

if __name__ == "__main__":
    # Silence common warnings for clean output
    warnings.simplefilter("ignore", category=RuntimeWarning)
    main()