import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = r"C:\Users\HansenSimonO\Documents\Coding\PyTradeTwo\pytrade-two\src\data\enriched_financial_data.csv"

def visualize_close_and_volume(csv_path: str = CSV_PATH) -> None:
    """
    Reads the provided CSV and creates one figure per asset:
      - Top subplot: closing price over time
      - Bottom subplot: volume per day (bar chart)

    Displays all figures using matplotlib.
    """
    # Load and prepare data
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns or "Symbol" not in df.columns or "Close" not in df.columns or "Volume" not in df.columns:
        raise ValueError("CSV must contain Date, Symbol, Close, Volume columns.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"])

    # Group by asset symbol
    for symbol, sdf in df.groupby("Symbol"):
        if sdf.empty:
            continue

        dates = sdf["Date"]
        close = sdf["Close"].astype(float)
        volume = sdf["Volume"].astype(float)

        # Create figure with two subplots sharing x-axis
        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle(f"{symbol} - Close & Volume", fontsize=16)

        # Price subplot
        ax_price.plot(dates, close, color="tab:blue", linewidth=1.2)
        ax_price.set_ylabel("Close")
        ax_price.grid(True, alpha=0.3)

        # Volume subplot (bar chart)
        ax_vol.bar(dates, volume, color="tab:gray", width=1.0, align="center")
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(True, axis="y", alpha=0.3)

        # Date formatting
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax_vol.xaxis.set_major_locator(locator)
        ax_vol.xaxis.set_major_formatter(formatter)

        fig.autofmt_xdate()
        plt.tight_layout()

    plt.show()

def visualize_price_with_features(csv_path: str = CSV_PATH) -> None:
    """
    For each asset, create a figure with 9 vertically stacked subplots (shared x-axis):
      1. Close price
      2-9. Eight key features from the dataset.

    Figure size is (16, 9) with equal vertical spacing. All subplots share the date x-axis.

    Selected features:
      - return_1d
      - return_5d
      - momentum_10
      - rsi_14
      - macd
      - macd_signal
      - bb_width
      - volume_percentile_20d

    Adjust the feature list below if needed.
    """
    df = pd.read_csv(csv_path)
    required_cols = ["Date", "Symbol", "Close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column.")

    # Pick 8 features (ensure they exist; fallback to available ones)
    candidate_features = [
        "return_1d",
        "return_5d",
        "momentum_10",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_width",
        "volume_percentile_20d",
    ]
    available_features = [f for f in candidate_features if f in df.columns]
    if len(available_features) < 8:
        raise ValueError(f"Not all selected features are available. Found {len(available_features)}: {available_features}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"])

    for symbol, sdf in df.groupby("Symbol"):
        if sdf.empty:
            continue

        dates = sdf["Date"]
        close = sdf["Close"].astype(float)

        # Prepare figure with 9 equally spaced rows, shared x-axis
        fig, axes = plt.subplots(
            9, 1, sharex=True, figsize=(16, 9),
            gridspec_kw={"hspace": 0.2, "left": 0.05, "right": 0.95, "top": 0.95, "bottom": 0.05}  # even spacing with small borders
        )
        fig.suptitle(f"{symbol} - Price & Features", fontsize=16)

        # 1) Price
        ax_price = axes[0]
        ax_price.plot(dates, close, color="tab:blue", linewidth=1.2)
        ax_price.set_ylabel("Close")
        ax_price.grid(True, alpha=0.3)

        # 2-9) Features
        for i, feat in enumerate(available_features[:8], start=1):
            series = pd.to_numeric(sdf[feat], errors="coerce")
            ax = axes[i]
            # Choose a line plot; some features may be noisy
            ax.plot(dates, series, linewidth=1.0, color="tab:orange")
            ax.set_ylabel(feat)
            ax.grid(True, alpha=0.3)

        # Date formatting on the bottom axis only
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(formatter)

        fig.autofmt_xdate()
        plt.tight_layout()

def show():
    plt.show()

if __name__ == "__main__":
    # visualize_close_and_volume()
    visualize_price_with_features()
    visualize_close_and_volume()
    show()