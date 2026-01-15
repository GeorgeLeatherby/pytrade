"""
Aggregate daily FX rates from multiple overlapping Investing.com style CSV exports.

Input folder contains files like:
  USD_EUR Historical Data.csv
  USD_EUR Historical Data (1).csv
  USD_GBP Historical Data.csv
Each file schema:
  "Date","Price","Open","High","Low","Vol.","Change %"

Goal:
  - Build a complete daily date index 2000-01-01 .. 2025-10-15 (inclusive)
  - For every currency pair (e.g. USD_EUR) collect all rows across all its files
  - Compute a robust per-file per-row consolidated price (robust OHLC aggregator)
  - If multiple files have the same date/pair, aggregate (median) across files
  - Output a wide CSV: Date + one column per pair (ISO upper) named exactly as in filenames
  - Missing dates (no data) remain blank (NaN) for later imputation

Robust OHLC consolidation:
  - Parse numeric candidates among [Price, Open, High, Low]
  - Discard non-finite / zero / obvious outliers (values outside [q1-3*iqr, q3+3*iqr] if >=3 samples)
  - Use median of remaining; fallback to simple mean; fallback to single available value; else NaN

Notes:
  - No forward-fill or leakage handling here (keep raw).
  - Later you can decide on interpolation (e.g., business-day forward-fill).
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from statistics import median

# ---------------- Configuration ----------------
FX_DIR = Path(r"C:\Users\HansenSimonO\Documents\Coding\PyTradeOne_v01\src\data\used_data\fx_rates_daily")  # <-- PLACE with actual folder path
OUTPUT_CSV = Path(r"C:\Users\HansenSimonO\Documents\Coding\PyTradeOne_v01\src\data\used_data\fx_rates_daily\aggregated_fx_rates_2000_2025.csv")
DATE_START = pd.Timestamp("2000-01-01")
DATE_END = pd.Timestamp("2025-10-15")
FILENAME_REGEX = re.compile(r"^([A-Z]{3}_[A-Z]{3}) Historical Data.*\.csv$", re.IGNORECASE)

# If some files are large, adjust chunksize (we read whole file here since typical size is moderate)
READ_KWARGS = dict(encoding="utf-8", engine="python")


def _to_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    v = value.strip()
    # Remove thousands separators / percent signs
    v = v.replace(",", "")
    # Handle volume suffixes (K,M) though not used for price columns; just in case misplacement
    mult = 1.0
    if v.endswith(("K", "k")):
        mult = 1_000.0
        v = v[:-1]
    elif v.endswith(("M", "m")):
        mult = 1_000_000.0
        v = v[:-1]
    try:
        return float(v) * mult
    except ValueError:
        return None


def consolidate_ohlc(row: pd.Series) -> float:
    """
    Robustly combine Price/Open/High/Low into a single representative daily rate.
    Approach:
      - Collect numeric candidates
      - Optional outlier trim using IQR if >=3 values
      - Use median (robust); fallback mean; else NaN
    """
    candidates_raw = [row.get("Price"), row.get("Open"), row.get("High"), row.get("Low")]
    candidates = []
    for c in candidates_raw:
        val = _to_float(c) if isinstance(c, str) else (float(c) if pd.notna(c) else None)
        if val is None or not np.isfinite(val) or val == 0:
            continue
        candidates.append(val)

    if not candidates:
        return np.nan

    if len(candidates) >= 3:
        q1 = np.percentile(candidates, 25)
        q3 = np.percentile(candidates, 75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            trimmed = [x for x in candidates if lower <= x <= upper]
            if trimmed:
                candidates = trimmed

    try:
        return float(median(candidates))
    except Exception:
        return float(np.mean(candidates))


def parse_file(filepath: Path) -> Tuple[str, pd.DataFrame]:
    """
    Parse one FX CSV file and return (pair_code, dataframe[date, consolidated_price]).
    Ignores rows that fail date parse.
    """
    fname = filepath.name
    m = FILENAME_REGEX.match(fname)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {fname}")
    pair_code = m.group(1).upper()  # e.g. USD_EUR

    df = pd.read_csv(filepath, **READ_KWARGS)
    # Normalize columns (some exports may differ in capitalization)
    df.columns = [c.strip().replace('"', '') for c in df.columns]

    if "Date" not in df.columns:
        raise ValueError(f"File {fname} missing 'Date' column")

    # Parse dates (format mm/dd/YYYY or similar)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"])

    # Consolidate OHLC
    df["rate"] = df.apply(consolidate_ohlc, axis=1)
    df = df[["Date", "rate"]].dropna(subset=["rate"])

    # Deduplicate dates inside file (median if duplicates)
    df = df.groupby("Date", as_index=False)["rate"].median()

    return pair_code, df


def aggregate_all_fx(fx_dir: Path) -> pd.DataFrame:
    """
    Walk directory, parse all matching FX CSVs, aggregate by date & pair.
    Multiple files per pair: median across sources per date.
    """
    files = [p for p in fx_dir.glob("*.csv") if FILENAME_REGEX.match(p.name)]
    if not files:
        raise FileNotFoundError(f"No FX CSV files matching pattern in {fx_dir}")

    # Storage: pair -> dict(date -> list of rates)
    store: Dict[str, Dict[pd.Timestamp, List[float]]] = {}

    for f in sorted(files):
        try:
            pair, df = parse_file(f)
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")
            continue

        pair_map = store.setdefault(pair, {})
        for _, row in df.iterrows():
            d = row["Date"]
            if d < DATE_START or d > DATE_END:
                continue
            pair_map.setdefault(d, []).append(float(row["rate"]))

    # Build full date index
    full_index = pd.date_range(DATE_START, DATE_END, freq="D")
    out = pd.DataFrame(index=full_index)

    # Aggregate per pair
    for pair, date_map in sorted(store.items()):
        values = []
        for d in full_index:
            lst = date_map.get(d)
            if not lst:
                values.append(np.nan)
            else:
                # Median across files; robust to outliers
                values.append(float(median(lst)))
        out[pair] = values

    # Add forward filling of columns up to 3 days for missing information
    out = out.ffill(limit=3)

    out.index.name = "Date"
    return out


def execute_fx_compiler():
    if not FX_DIR.exists():
        raise FileNotFoundError(f"FX directory not found: {FX_DIR}")

    df = aggregate_all_fx(FX_DIR)

    # Optional: sanity checks
    print(f"Pairs aggregated: {len(df.columns)}")
    print(f"Date range: {df.index.min().date()} .. {df.index.max().date()}")
    print(df.head())
    print(df.tail())

    df.to_csv(OUTPUT_CSV, float_format="%.6f")
    print(f"Saved aggregated rates to {OUTPUT_CSV.resolve()}")


# Entry point
execute_fx_compiler()