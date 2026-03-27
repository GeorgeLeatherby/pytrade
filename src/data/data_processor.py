"""
This is a data processing script for raw financial data.
It prepares the data by transforming, normalizing and enriching it
for the subsequent training and validation of machine learning models.

Output:
- A dataset ready for machine learning

Inputs:
- Raw financial data files (CSV format)
- Configuration parameters for data processing (e.g., date range)
"""

import pandas as pd
import numpy as np
import os
import msvcrt

class DataEnricher:
    def __init__(self, input_raw_indices_filepath, input_raw_fx_filepath):
        self.input_raw_indices_filepath = input_raw_indices_filepath
        self.input_raw_fx_filepath = input_raw_fx_filepath

        # Save output in src/data
        self.output_dir = os.path.dirname(__file__)

        # Verbosity level for debugging
        self.verbose = 0

        # read raw data files
        self.raw_indices_data = pd.read_csv(self.input_raw_indices_filepath)
        if self.input_raw_fx_filepath is not None:
            self.raw_fx_data = pd.read_csv(self.input_raw_fx_filepath)
            self.raw_fx_data['Date'] = pd.to_datetime(self.raw_fx_data['Date']) # convert
        else: print("\nNo FX data file provided. Currency conversion will be skipped.")

        # Define max data horizon for normalization
        self.max_norm_horizon = int(252 / 7) # 36 trading days (approx.  1.5 months) for rolling normalization of returns, volume changes etc.

        # Convert 'Date' columns to datetime objects immediately after loading
        self.raw_indices_data['Date'] = pd.to_datetime(self.raw_indices_data['Date'])

        # Asset configuration
        self.assets = sorted(self.raw_indices_data['Symbol'].dropna().unique().tolist())  # list all symbols to use here
        self.currency_map = {
            'Crude': 'USD', 'EWG': 'USD', 'EWH': 'USD',
            'EWJ': 'USD', 'EWQ': 'USD', 'EWS': 'USD',
            'EWT': 'USD', 'EWU': 'USD', 'EWY': 'USD',
            'SPY': 'USD', 'Gold': 'USD'
        }

        # Compare date ranges
        self.starting_date, self.ending_date = self.compare_date_ranges()

        # Enriched data storage
        self.enriched_data = pd.DataFrame()

        # start enrichment process
        self.enrich_data()

    def enrich_data(self):
        # 0. Convert currencies of indices to USD using fx data. Note: Currently not in use!
        # self.convert_to_usd()

        # 1. Raw to stationary Feature engineering
        self.convert_raw_to_stationary_data()

        self.trigonometric_date_encoding()
        
        self.NaN_counting()
        # 2. Technical indicators
        self.technical_indicators()
        self.NaN_counting()
        # 3. Risk metrics
        #self.pre_trading_risk_metrics()
        self.NaN_counting()
        #self.enhanced_risk_adjusted_metrics()
        self.NaN_counting()
        # 4. Currency features
        # 5. Cross-asset relationships
        self.cross_asset_relationships()
        self.add_cross_asset_features()
        self.NaN_counting()
        # COUNT NaNs per columns
        self.NaN_detection()
        # DETECT Infs
        self.Inf_detection()
        # 6. Save calculated data
        self.save_calculated_data()


    def trigonometric_date_encoding(self):
            """
            Improved cyclic encoding:
            - Prevents drift by using actual calendar components.
            - Maintains periodicity across weekends.
            - Normalized to exact calendar bounds.
            """
            df = self.enriched_data.copy()
            dates = pd.to_datetime(df['Date'])
            
            two_pi = 2 * np.pi

            # 1. Day of Week (0=Mon, 4=Fri)
            # Periodic across 5 trading days
            dow = dates.dt.dayofweek
            df['dow_sin'] = np.sin(two_pi * dow / 5)
            df['dow_cos'] = np.cos(two_pi * dow / 5)

            # 2. Day of Month (Relative position in month)
            # (day - 1) / (days_in_month - 1)
            # This aligns '1.0' perfectly with the last day of ANY month.
            dom = dates.dt.day
            dim = dates.dt.days_in_month
            df['dom_sin'] = np.sin(two_pi * (dom - 1) / (dim - 1))
            df['dom_cos'] = np.cos(two_pi * (dom - 1) / (dim - 1))

            # 3. Month of Year (1-12)
            # Aligns perfectly with seasonality (tax season, Santa rally, etc.)
            moy = dates.dt.month
            df['moy_sin'] = np.sin(two_pi * (moy - 1) / 12)
            df['moy_cos'] = np.cos(two_pi * (moy - 1) / 12)

            self.enriched_data = df

        
    def trigonometric_date_encoding_legacy(self):
        """
        Continuous cyclic encodings without weekend jumps:
        - dow_sin/dow_cos: business-day index mod 5
        - dom_sin/dom_cos: business-day index mod 21 (approx. trading days per month)
        - moy_sin/moy_cos: business-day index mod 252 (approx. trading days per year)
        """
        # copy enriched_data to avoid modifying original dataframe during iteration
        base_df = self.enriched_data
        if 'Date' not in base_df.columns:
            raise ValueError("Column 'Date' is required for trigonometric date encoding.")
        df = base_df.copy()
        dates = pd.to_datetime(df['Date'])

        # Business-day index (no gaps for weekends)
        bdays = pd.bdate_range(dates.min(), dates.max(), freq='C')
        bday_idx = pd.Index(bdays).get_indexer(dates)

        two_pi = 2 * np.pi
        dow_cycle = 5.0    # business week
        dom_cycle = 21.0   # approx. trading days per month
        moy_cycle = 252.0  # approx. trading days per year

        df['dow_sin'] = np.sin(two_pi * bday_idx / dow_cycle)
        df['dow_cos'] = np.cos(two_pi * bday_idx / dow_cycle)
        df['dom_sin'] = np.sin(two_pi * bday_idx / dom_cycle)
        df['dom_cos'] = np.cos(two_pi * bday_idx / dom_cycle)
        df['moy_sin'] = np.sin(two_pi * bday_idx / moy_cycle)
        df['moy_cos'] = np.cos(two_pi * bday_idx / moy_cycle)

        self.enriched_data = df

    def convert_to_usd(self):
        """
        Convert asset OHLC prices from their local currency into USD.
        
        Assumptions (per docstring & general FX convention used here):
        - self.raw_indices_data contains columns at least: Date, Symbol, Open, High, Low, Close, Volume
        - self.currency_map maps each Symbol to its currency code (e.g. 'DAX' -> 'EUR')
        - self.raw_fx_data contains daily FX rates with one row per Date.
        - FX rate columns represent: 1 USD = X units of foreign currency (e.g. EUR column value 0.85 means 1 USD = 0.85 EUR)
          Therefore: local_price_in_foreign_currency / fx_rate_foreign_per_USD = price_in_USD.
        - Assets already denominated in USD (currency == 'USD') are left unchanged.
        
        Result:
        - self.raw_indices_data price columns (Open, High, Low, Close) converted to USD in place.
        - Original local prices preserved in new columns with suffix '_local' for traceability.
        """
        print("\nConverting all asset prices to USD...")
        
        # ------------------------------------------------------------------
        # 1. Basic validations + preparation
        # ------------------------------------------------------------------
        required_price_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_price_cols + ['Date', 'Symbol']:
            if col not in self.raw_indices_data.columns:
                raise ValueError(f"Required column '{col}' missing in raw_indices_data.")
        
        if 'Date' not in self.raw_fx_data.columns:
            raise ValueError("FX data must contain a 'Date' column.")
        
        # Ensure date types are consistent (use datetime internally, keep original formatting when saving later)
        self.raw_indices_data['Date'] = pd.to_datetime(self.raw_indices_data['Date'])
        self.raw_fx_data['Date'] = pd.to_datetime(self.raw_fx_data['Date'])
        
        # ------------------------------------------------------------------
        # 2. Normalize FX rate column names (flexible matching)
        #    We attempt to map columns to pure currency codes: EUR, GBP, JPY, HKD, USD etc.
        #    If file already has clean codes we keep them.
        # ------------------------------------------------------------------
        fx_df = self.raw_fx_data.copy()
        
        def normalize_fx_column(col: str) -> str:
            """
            Attempt to reduce various naming patterns to a raw currency code.
            Examples:
                'EUR' -> 'EUR'
                'USD_EUR' -> 'EUR'
                'EURUSD' -> 'EUR' (ambiguous but we assume 1 USD in EUR if numeric matches expectation)
                'USD/EUR' -> 'EUR'
                'EUR/USD' -> 'EUR' (will still interpret as EUR per USD by assumption)
            NOTE: This is heuristic; if ambiguous patterns exist, user should clean input file.
            """
            c = col.upper().replace('/', '_')
            # Simple direct match
            if c in {'EUR','GBP','JPY','HKD','USD','CHF','CAD','AUD','NZD'}:
                return c
            # Patterns containing USD + other currency
            for code in ['EUR','GBP','JPY','HKD','CHF','CAD','AUD','NZD']:
                if code in c:
                    return code
            return col  # fallback original
        
        fx_columns_original = fx_df.columns.tolist()
        fx_column_map = {}
        for c in fx_columns_original:
            if c == 'Date':
                continue
            norm = normalize_fx_column(c)
            fx_column_map[c] = norm
        
        # Rename columns for easier access (avoid collisions by aggregating duplicates via mean if necessary)
        # If multiple source columns map to same normalized currency, combine them (take mean row-wise)
        normalized_fx = {'Date': fx_df['Date']}
        for src_col, norm_col in fx_column_map.items():
            if norm_col not in normalized_fx:
                normalized_fx[norm_col] = fx_df[src_col]
            else:
                # Combine (average) duplicate mapped columns for same currency code
                normalized_fx[norm_col] = pd.concat(
                    [normalized_fx[norm_col], fx_df[src_col]], axis=1
                ).mean(axis=1)
        
        fx_df = pd.DataFrame(normalized_fx)
        
        # ------------------------------------------------------------------
        # 3. Verify we have FX rates for needed currencies (excluding USD)
        # ------------------------------------------------------------------
        needed_currencies = {cur for cur in self.currency_map.values() if cur != 'USD'}
        missing = [c for c in needed_currencies if c not in fx_df.columns]
        if missing:
            raise ValueError(
                f"Missing FX rate columns for currencies: {missing}. "
                f"Found columns: {fx_df.columns.tolist()}"
            )
        
        # ------------------------------------------------------------------
        # 4. Attach currency code to each price row via symbol->currency mapping
        # ------------------------------------------------------------------
        self.raw_indices_data['Currency'] = self.raw_indices_data['Symbol'].map(self.currency_map)
        if self.raw_indices_data['Currency'].isna().any():
            unknown_symbols = self.raw_indices_data[self.raw_indices_data['Currency'].isna()]['Symbol'].unique()
            raise ValueError(f"Currency mapping missing for symbols: {unknown_symbols}")
        
        # ------------------------------------------------------------------
        # 5. Merge FX rates onto index data (left join so index rows retained)
        #    After merge we have per-row FX rate columns; we'll pick the appropriate one using the 'Currency' column.
        # ------------------------------------------------------------------
        merged = self.raw_indices_data.merge(fx_df, on='Date', how='left', suffixes=('', '_fx'))
        
        # ------------------------------------------------------------------
        # 6. Build a vector of FX rates applicable per row (1 if already USD)
        # ------------------------------------------------------------------
        # Create a Series where each row gets the FX rate matching its Currency column.
        def choose_fx_rate(row):
            cur = row['Currency']
            if cur == 'USD':
                return 1.0  # Already USD denominated
            val = row.get(cur, np.nan)
            return val
        
        merged['fx_rate_local_per_usd'] = merged.apply(choose_fx_rate, axis=1)
        
        # ------------------------------------------------------------------
        # 7. Handle missing FX rates:
        #    - Forward fill within FX data timeline (typical if monthly rates fixed for month)
        #    - If still missing after forward-fill -> raise warning/error depending on severity
        # ------------------------------------------------------------------
        # Forward fill only the fx_rate column (some may be NaN if outside FX coverage)
        # For safety, re-forward fill by Date ordering
        merged = merged.sort_values('Date')
        merged['fx_rate_local_per_usd'] = merged['fx_rate_local_per_usd'].ffill()
        
        if merged['fx_rate_local_per_usd'].isna().any():
            missing_rows = merged[merged['fx_rate_local_per_usd'].isna()]
            raise ValueError(
                f"FX rate still missing for {len(missing_rows)} rows after forward fill. "
                f"Dates sample: {missing_rows['Date'].head().tolist()}"
            )
        
        # ------------------------------------------------------------------
        # 8. Preserve original local prices before conversion (traceability)
        # ------------------------------------------------------------------
        for col in required_price_cols:
            merged[f'{col}_local'] = merged[col]  # Keep original local currency value
        
        # ------------------------------------------------------------------
        # 9. Convert prices to USD: price_local / (local_per_usd) = price_usd
        # ------------------------------------------------------------------
        for col in required_price_cols:
            merged[col] = merged[col] / (merged['fx_rate_local_per_usd'] + 1e-8)
        
        # NOTE: Volume typically represents number of shares/contracts and is not currency-denominated.
        # If volume is monetary (rare), user would need an additional flag. We leave 'Volume' unchanged.
        
        # ------------------------------------------------------------------
        # 10. Clean up & assign back to self.raw_indices_data (drop intermediate FX columns except needed)
        # ------------------------------------------------------------------
        # Keep only necessary FX columns (we keep 'fx_rate_local_per_usd' for possible downstream diagnostics)
        keep_cols = list(self.raw_indices_data.columns) + [c for c in merged.columns if c.endswith('_local')] + ['fx_rate_local_per_usd']
        # Ensure duplicates removed while preserving order
        keep_cols_unique = []
        for c in keep_cols:
            if c not in keep_cols_unique:
                keep_cols_unique.append(c)
        
        self.raw_indices_data = merged[keep_cols_unique].copy()
        
        # Restore Date to original string format if needed (optional). We keep datetime; later save will convert automatically.
        
        # ------------------------------------------------------------------
        # 11. Report summary statistics for verification
        # ------------------------------------------------------------------
        print("Price conversion completed. ")
        if self.verbose > 0:
            print("Sample rows after conversion:")
            print(self.raw_indices_data.head(10))
            
            # Show a quick sanity check for one non-USD asset
            sample_non_usd = next((s for s,c in self.currency_map.items() if c != 'USD'), None)
            if sample_non_usd:
                sample_df = self.raw_indices_data[self.raw_indices_data['Symbol'] == sample_non_usd].head(3)
                if not sample_df.empty:
                    print(f"\nSanity check ({sample_non_usd}) local vs USD prices:")
                    for _, r in sample_df.iterrows():
                        print(
                            f"Date={r['Date'].date()} FX={r['fx_rate_local_per_usd']:.5f} "
                            f"Local Close={r['Close_local']:.3f} USD Close={r['Close']:.3f}"
                        )
            print("All prices now expressed in USD.")

    def compare_date_ranges(self):
        """
        Determine usable date range from actual data bounds.
        NOTE: No comparison needed
        """
        min_date = self.raw_indices_data['Date'].min()
        max_date = self.raw_indices_data['Date'].max()
        
        print(f"\nData availability:")
        print(f"  Raw data range: {min_date} to {max_date}")
        
        # Use actual data bounds, but ensure we have enough history for features
        # Add buffer: we need at least max_norm_horizon before using features
        buffer_days = self.max_norm_horizon + 10  # Safety margin
        
        # Use first date that gives us enough history
        self.starting_date = min_date + pd.Timedelta(days=buffer_days)
        self.ending_date = max_date
        
        print(f"  Feature calculation range: {self.starting_date} to {self.ending_date}")
        print(f"  (First {buffer_days} days excluded for rolling window initialization)")
        
        return str(self.starting_date.date()), str(self.ending_date.date())

    def convert_raw_to_stationary_data(self):
        """
        Convert raw data into useful signals for RL agent learning.
        """
        print("\nConverting raw data to stationary format (vectorized)...")

        # Copy and filter the raw data
        df = self.raw_indices_data.copy()
        df = df[(df['Date'] >= self.starting_date) & (df['Date'] <= self.ending_date)].sort_values(['Symbol', 'Date'])
        
        # --- 1. Return Calculations (Vectorized) ---
        print("Calculating returns and stationary features...")
        return_periods = [1, 3, 5, 10, 20, 21, 40] 

        for period in return_periods:
            # A. Legacy Pct Change (remain unchanged)
            df[f'return_{period}d'] = df.groupby('Symbol')['Close'].transform(lambda x: x.pct_change(periods=period))
            df[f'open_return_{period}d'] = df.groupby('Symbol')['Open'].transform(lambda x: x.pct_change(periods=period))
            df[f'high_return_{period}d'] = df.groupby('Symbol')['High'].transform(lambda x: x.pct_change(periods=period))
            df[f'low_return_{period}d'] = df.groupby('Symbol')['Low'].transform(lambda x: x.pct_change(periods=period))

            # B. Log Returns for OHLC (Stationary Base)
            # We use transform + log(x/x.shift) for vectorized speed
            for col in ['Open', 'High', 'Low', 'Close']:
                df[f'log_{col.lower()}_return_{period}d'] = df.groupby('Symbol')[col].transform(
                    lambda x: np.log(x / x.shift(period))
                )

            # C. Relative Log Returns to SPY (Market-Relative Context)
            # Create a Series of SPY returns for this specific period mapped to Date
            spy_returns = df[df['Symbol'] == 'SPY'].set_index('Date')[f'log_close_return_{period}d']
            
            # Subtract SPY return from each asset return for the same period
            # map(spy_returns) aligns the dates correctly across all symbols
            df[f'rel_log_close_{period}d'] = df[f'log_close_return_{period}d'] - df['Date'].map(spy_returns)

            # D. Vol-of-vol feature for relative returns (rolling std of relative returns)
            group_rel = df.groupby('Symbol')[f'rel_log_close_{period}d']
            max_horizon_rolling_std = group_rel.transform(lambda x: x.rolling(window=self.max_norm_horizon).std())
            max_horizon_rolling_mean = group_rel.transform(lambda x: x.rolling(window=self.max_norm_horizon).mean())
            short_term_rolling_std = group_rel.transform(lambda x: x.rolling(window=5).std())  # Short-term volatility for regime detection

            # E. Calculate Vol-of-Vol (Regime Indicator) ---
            # 1. Log-scale the volatility to keep it stationary and compress outliers
            # We add a tiny epsilon to avoid log(0)
            df[f'log_vol_{period}d'] = np.log(max_horizon_rolling_std + 1e-8)
            df[f'log_vol_short_{period}d'] = np.log(short_term_rolling_std + 1e-8)
            
            # 2. Z-score the log-volatility (Vol-of-Vol)
            # This tells the agent: "Is the CURRENT volatility high relative to RECENT volatility?"
            # A value > 2.0 here means we are entering a Black Swan/Crash regime.
            group_vol = df.groupby('Symbol')[f'log_vol_{period}d']
            df[f'z_vol_of_vol_{period}d'] = group_vol.transform(
                lambda x: (x - x.rolling(window=self.max_norm_horizon).mean()) / (x.rolling(window=self.max_norm_horizon).std() + 1e-8)
            )
            # Volatility breakout based on short-term volatility compared to long-term volatility
            df[f'z_vol_breakout_{period}d'] = (df[f'log_vol_short_{period}d'] - max_horizon_rolling_mean) / (max_horizon_rolling_std + 1e-8)
            # Clip to avoid overwhelming network during Black Swans
            df[f'z_vol_breakout_{period}d'] = df[f'z_vol_breakout_{period}d'].clip(-4, 4)

            # --- F. Final Z-Score for Relative Returns (with clipping for Black Swans) ---
            # We clip at 4.0 standard deviations to prevent gradient explosion
            raw_z = (df[f'rel_log_close_{period}d'] - max_horizon_rolling_mean) / (max_horizon_rolling_std + 1e-8)
            df[f'z_rel_log_close_{period}d'] = raw_z.clip(-4, 4)


        # --- 2. Volume Change and Percentile Calculations (Vectorized) ---
        print("Calculating volume changes and percentiles...")
        volume_periods = [1, 3, 5, 10, 20, 21, 40]

        # Pre-calculate SPY volume log-intensity for the relative feature
        # We use a log-ratio of SPY volume to its own moving average
        spy_vol = df[df['Symbol'] == 'SPY'].set_index('Date')['Volume']
        spy_vol_ma = spy_vol.rolling(window=self.max_norm_horizon).mean()
        spy_vol_intensity = np.log(spy_vol / (spy_vol_ma + 1e-8)).clip(lower=1e-8)

        for period in volume_periods:

            # A. Log Volume Intensity (Today vs Moving Average)
            # This is much more stable than pct_change for volume
            v_ma = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(window=self.max_norm_horizon).mean())
            df[f'log_volume_intensity_{period}d'] = np.log(df['Volume'] / (v_ma + 1e-8))

            # B. Market-Relative Volume Intensity
            # Subtracting the SPY volume intensity from the Asset volume intensity
            df[f'rel_volume_intensity_{period}d'] = df[f'log_volume_intensity_{period}d'] - df['Date'].map(spy_vol_intensity)

            # C. Normalized Volume Feature (Z-Score + Tanh)
            # We normalize the intensity using the max_norm_horizon
            group_v = df.groupby('Symbol')[f'rel_volume_intensity_{period}d']
            v_mean = group_v.transform(lambda x: x.rolling(window=self.max_norm_horizon).mean())
            v_std = group_v.transform(lambda x: x.rolling(window=self.max_norm_horizon).std())
            
            z_vol = (df[f'rel_volume_intensity_{period}d'] - v_mean) / (v_std + 1e-8)
            
            # Apply tanh to squash values into [-1, 1] range
            # This is excellent for RL agents as it preserves the sign but limits the impact of outliers
            df[f'norm_volume_feature_{period}d'] = np.tanh(z_vol)


            # ------------ LEGACY Volume Change (remain unchanged) -------------------
            # Volume Percentile
            df[f'volume_percentile_{period}d'] = df.groupby('Symbol')['Volume'].transform(
                lambda x: x.rolling(window=period, min_periods=1).rank(pct=True)
            )
            
            # Volume Change (Normalized)
            # Calculate raw percentage change first
            raw_vol_change = df.groupby('Symbol')['Volume'].transform(lambda x: x.pct_change(periods=period))
            
            # Define adaptive rolling window for normalization
            rolling_window = max(min(period * 3, 252), 60)
            
            # Grouped rolling normalization (z-score)
            mean = raw_vol_change.groupby(df['Symbol']).transform(lambda x: x.rolling(window=rolling_window, min_periods=20).mean())
            std = raw_vol_change.groupby(df['Symbol']).transform(lambda x: x.rolling(window=rolling_window, min_periods=20).std())
            z_score = (raw_vol_change - mean) / (std + 1e-8)
            
            # Apply tanh and assign to the final column name
            df[f'volume_change_{period}d'] = np.tanh(z_score / 2)

        # --- 3. Other Stationary Features (Vectorized) ---
        print("Calculating other stationary features...")
        # Daily Range
        df['daily_range_pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Overnight Gap
        # Correctly calculate the previous day's close within each symbol group
        prev_close = df.groupby('Symbol')['Close'].shift(1)
        df['overnight_gap'] = (df['Open'] - prev_close) / (prev_close + 1e-8)

        # --- 4. Final Assignment ---
        self.enriched_data = df

        if self.verbose > 0:
            print("\nEnriched data after converting to stationary format:")
            print(f"Shape of enriched data: {self.enriched_data.shape}")
            print(self.enriched_data.head(300))

    def technical_indicators(self):
        """
        Calculate technical indicators for the enriched data. Add them to proper data and symbol
        position within enriched_data
        """
        # Copy enriched_data
        technical_indicators = self.enriched_data.copy()

        # Filter enriched data set to usable range
        technical_indicators = technical_indicators[
            (technical_indicators['Date'] >= self.starting_date) &
            (technical_indicators['Date'] <= self.ending_date)
        ]

        # Initialize technical indicator columns
        momentum_columns = ['momentum_10', 'momentum_20', 'rsi_14', 'rsi_30', 'macd', 'macd_signal', 'macd_histogram', 'bb_position', 'bb_width']

        for col in momentum_columns:
            technical_indicators[col] = np.nan
        
        print()
        for symbol in technical_indicators['Symbol'].unique():
            symbol_data = technical_indicators[technical_indicators['Symbol'] == symbol].sort_values('Date').copy()

            print(f"Calculating technical indicators for {symbol}...")

            # Momentum
            symbol_data['momentum_10'] = symbol_data['Close'].pct_change(periods=10)
            # Normalize with rolling statistics: no look-ahead bias
            rolling_std = symbol_data['momentum_10'].rolling(80).std()
            symbol_data['momentum_10'] = (symbol_data['momentum_10'] / (rolling_std + 1e-8)).clip(-3, 3)  # Clip to avoid extreme values

            symbol_data['momentum_20'] = symbol_data['Close'].pct_change(periods=20)
            rolling_std = symbol_data['momentum_20'].rolling(80).std()
            symbol_data['momentum_20'] = (symbol_data['momentum_20'] / (rolling_std + 1e-8)).clip(-3, 3)  # Clip to avoid extreme values

            # Add momentum to correct symbol and date in enriched_data
            for col in ['momentum_10', 'momentum_20']:
                self.enriched_data.loc[
                    (self.enriched_data['Symbol'] == symbol) &
                    (self.enriched_data['Date'].isin(symbol_data['Date'])),
                    col
                ] = symbol_data[col]

            if self.verbose > 0:
                print(f"    Max momentum val: {symbol_data[['momentum_10', 'momentum_20']].max().max()}")
                print(f"    Min momentum val: {symbol_data[['momentum_10', 'momentum_20']].min().min()}")
                print(f"    Mean momentum val: {symbol_data[['momentum_10', 'momentum_20']].mean().mean()}")

            # RSI - Relative strength index 14 and 30 days
            for period in [14, 30]:
                delta = symbol_data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                symbol_data[f'rsi_{period}'] = ((rsi - 50) / 50).rolling(3).mean()  # Normalize to [-1, 1]
            
                # Add RSI to correct symbol and date in enriched_data
                self.enriched_data.loc[
                    (self.enriched_data['Symbol'] == symbol) &
                    (self.enriched_data['Date'].isin(symbol_data['Date'])),
                    f'rsi_{period}'
                ] = symbol_data[f'rsi_{period}']

            if self.verbose > 0:
                print(f"    Max RSI val: {symbol_data[['rsi_14', 'rsi_30']].max().max()}")
                print(f"    Min RSI val: {symbol_data[['rsi_14', 'rsi_30']].min().min()}")
                print(f"    Mean RSI val: {symbol_data[['rsi_14', 'rsi_30']].mean().mean()}")

            # MACD - Moving Average Convergence Divergence
            """MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA.
            It indicates momentum and trend direction. A positive MACD suggests upward momentum, while a negative MACD indicates downward momentum.
            The MACD line is often plotted alongside a 9-period EMA of the MACD, known as the "signal line," which helps identify potential buy or sell signals.
            MACD histogram represents the difference between the MACD line and the signal line, providing insights into the strength of the trend.
            NOTE: We normalize all MACD values with rolling std in the end!
            """
            exp1 = symbol_data['Close'].ewm(span=12).mean()
            exp2 = symbol_data['Close'].ewm(span=26).mean()
            symbol_data['macd'] = (exp1 - exp2) / symbol_data['Close']  # Normalize by price to get percentage 

            # MACD signal line
            symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()

            # MACD histogram
            symbol_data['macd_histogram'] = symbol_data['macd'] - symbol_data['macd_signal']

            # Normalize MACD values with rolling std
            rolling_std = symbol_data['macd'].rolling(80).std()
            symbol_data['macd'] = (symbol_data['macd'] / (rolling_std + 1e-8)).clip(-3, 3)  # Clip to avoid extreme values
            rolling_std_signal = symbol_data['macd_signal'].rolling(80).std() + 1e-8
            symbol_data['macd_signal'] = (symbol_data['macd_signal'] / rolling_std_signal).clip(-3, 3)
            rolling_std_hist = symbol_data['macd_histogram'].rolling(80).std() + 1e-8
            symbol_data['macd_histogram'] = (symbol_data['macd_histogram'] / rolling_std_hist).clip(-3, 3)

            # Add MACD to correct symbol and date in enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'macd'
            ] = symbol_data['macd']
            # Add MACD signal to correct symbol and date in enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'macd_signal'
            ] = symbol_data['macd_signal']
            # Add MACD histogram to correct symbol and date in enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'macd_histogram'
            ] = symbol_data['macd_histogram']

            if self.verbose > 0:
                # Print MACD stats
                print(f"    Max MACD val: {symbol_data['macd'].max()}")
                print(f"    Min MACD val: {symbol_data['macd'].min()}")
                print(f"    Mean MACD val: {symbol_data['macd'].mean()}")
                print(f"    Max MACD signal val: {symbol_data['macd_signal'].max()}")
                print(f"    Min MACD signal val: {symbol_data['macd_signal'].min()}")
                print(f"    Mean MACD signal val: {symbol_data['macd_signal'].mean()}")
                print(f"    Max MACD histogram val: {symbol_data['macd_histogram'].max()}")
                print(f"    Min MACD histogram val: {symbol_data['macd_histogram'].min()}")
                print(f"    Mean MACD histogram val: {symbol_data['macd_histogram'].mean()}")

            # Bollinger Bands
            rolling_mean = symbol_data['Close'].rolling(window=self.max_norm_horizon).mean()
            rolling_std = symbol_data['Close'].rolling(window=self.max_norm_horizon).std()
            # Position within bands normalized to [-1, 1]
            bb_position = (symbol_data['Close'] - rolling_mean) / (rolling_std + 1e-8)
            symbol_data['bb_position'] = np.tanh(bb_position/2)  # Apply tanh to limit range to [-1, 1]

            # Add Bollinger Bands position to correct symbol and date in enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'bb_position'
            ] = symbol_data['bb_position']

            if self.verbose > 0:
                print(f"    Max Bollinger position val: {symbol_data['bb_position'].max()}")
                print(f"    Min Bollinger position val: {symbol_data['bb_position'].min()}")
                print(f"    Mean Bollinger position val: {symbol_data['bb_position'].mean()}")

            # Bollinger Band width
            symbol_data['bb_width_reworked'] = ((rolling_mean + rolling_std * 2) - (rolling_mean - rolling_std * 2)) / (rolling_mean + 1e-8)  # Width as percentage of price

            # Apply Z-score normalization to Bollinger Band width and clip to avoid extreme values
            bb_width_mean = symbol_data['bb_width_reworked'].rolling(window=self.max_norm_horizon).mean()
            bb_width_std = symbol_data['bb_width_reworked'].rolling(window=self.max_norm_horizon).std()
            symbol_data['bb_width_reworked'] = ((symbol_data['bb_width_reworked'] - bb_width_mean) / (bb_width_std + 1e-8)).clip(-3, 3)

            # LEGACY
            symbol_data['bb_width'] = (rolling_std * 2) / (rolling_mean + 1e-8)
            bb_width_mean = symbol_data['bb_width'].rolling(80).mean()
            bb_width_std = symbol_data['bb_width'].rolling(80).std()
            # Normalize Bollinger Band width with rolling std
            symbol_data['bb_width'] = ((symbol_data['bb_width'] - bb_width_mean) / (bb_width_std + 1e-8)).clip(-3, 3)

            # Add Current and LEGACY Bollinger Bands width to correct symbol and date in enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'bb_width'
            ] = symbol_data['bb_width']
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'bb_width_reworked'
            ] = symbol_data['bb_width_reworked']

            if self.verbose > 0:
                print(f"    Max Bollinger width val: {symbol_data['bb_width'].max()}")
                print(f"    Min Bollinger width val: {symbol_data['bb_width'].min()}")
                print(f"    Mean Bollinger width val: {symbol_data['bb_width'].mean()}")

            # Stochastic %K and %D
            low_14 = symbol_data['Low'].rolling(14).min()
            high_14 = symbol_data['High'].rolling(14).max()
            stoch_k = 100 * (symbol_data['Close'] - low_14) / (high_14 - low_14 + 1e-8)
            stoch_d = stoch_k.rolling(3).mean()

            # Normalize to [-1, 1]
            symbol_data['stoch_k'] = ((stoch_k - 50) / 50).clip(-1, 1)
            symbol_data['stoch_d'] = ((stoch_d - 50) / 50).clip(-1, 1)

            # Add to enriched_data
            for col in ['stoch_k', 'stoch_d']:
                self.enriched_data.loc[
                    (self.enriched_data['Symbol'] == symbol) &
                    (self.enriched_data['Date'].isin(symbol_data['Date'])),
                    col
                ] = symbol_data[col]

            if self.verbose > 0:
                print(f"    Max Stochastic %K val: {symbol_data['stoch_k'].max()}")
                print(f"    Min Stochastic %K val: {symbol_data['stoch_k'].min()}")
                print(f"    Mean Stochastic %K val: {symbol_data['stoch_k'].mean()}")
                print(f"    Max Stochastic %D val: {symbol_data['stoch_d'].max()}")
                print(f"    Min Stochastic %D val: {symbol_data['stoch_d'].min()}")
                print(f"    Mean Stochastic %D val: {symbol_data['stoch_d'].mean()}")

            # ATR for volatility assessment
            """ATR (Average True Range) is a technical analysis indicator that measures market volatility by calculating the 
            average range between high and low prices over a specified period.
            It provides insights into the degree of price fluctuations"""
            high_low = symbol_data['High'] - symbol_data['Low']
            high_close = np.abs(symbol_data['High'] - symbol_data['Close'].shift(1))
            low_close = np.abs(symbol_data['Low'] - symbol_data['Close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=self.max_norm_horizon).mean()

            # Z-score the ATR to normalize it, then clip to avoid extreme values
            atr_zscore = (atr - atr.rolling(window=self.max_norm_horizon).mean()) / (atr.rolling(window=self.max_norm_horizon).std() + 1e-8)
            symbol_data['atr_z_norm'] = atr_zscore.clip(-3, 3)

            # Normalize ATR as percentage of price
            atr_pct = atr / symbol_data['Close']
            atr_mean = atr_pct.rolling(80, min_periods=30).mean()
            atr_std = atr_pct.rolling(80, min_periods=30).std()
            symbol_data['atr_norm'] = ((atr_pct - atr_mean) / (atr_std + 1e-8)).clip(-3, 3)

            # Add atr_norm & atr_z_norm to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'atr_norm'
            ] = symbol_data['atr_norm']

            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'atr_z_norm'
            ] = symbol_data['atr_z_norm']

            if self.verbose > 0:
                print(f"    Max ATR val: {symbol_data['atr_norm'].max()}")
                print(f"    Min ATR val: {symbol_data['atr_norm'].min()}")
                print(f"    Mean ATR val: {symbol_data['atr_norm'].mean()}")
                print(f"    Max ATR Z-Score val: {symbol_data['atr_z_norm'].max()}")
                print(f"    Min ATR Z-Score val: {symbol_data['atr_z_norm'].min()}")
                print(f"    Mean ATR Z-Score val: {symbol_data['atr_z_norm'].mean()}")

            # Z distance to mean: Closing price distance to self.max_norm_horizon in ATR units (volatility-adjusted distance to mean)
            symbol_data['z_distance_to_mean'] = (symbol_data['Close'] - rolling_mean) / (atr + 1e-8)
            symbol_data['z_distance_to_mean'] = symbol_data['z_distance_to_mean'].clip(-3, 3)  # Clip to avoid extreme values

            # VPT (Volume Price Trend) combines price and volume for stronger signals
            symbol_data['vpt'] = (symbol_data['return_1d'] * symbol_data['volume_percentile_20d']).cumsum()
            # Normalize VPT
            vpt_norm = (symbol_data['vpt'] - symbol_data['vpt'].rolling(80, min_periods=30).mean()) / (symbol_data['vpt'].rolling(252, min_periods=60).std() + 1e-8)
            symbol_data['vpt'] = vpt_norm.clip(-3, 3)
            # Add to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'vpt'
            ] = symbol_data['vpt']

            if self.verbose > 0:
                print(f"    Max VPT val: {symbol_data['vpt'].max()}")
                print(f"    Min VPT val: {symbol_data['vpt'].min()}")
                print(f"    Mean VPT val: {symbol_data['vpt'].mean()}")

            # MFI - Money Flow Index combines price and volume to identify overbought/oversold conditions
            typical_price = (symbol_data['High'] + symbol_data['Low'] + symbol_data['Close']) / 3
            money_flow = typical_price * symbol_data['Volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            money_ratio = positive_flow / (negative_flow + 1e-8)
            mfi = 100 - (100 / (1 + money_ratio))
            symbol_data['mfi'] = (mfi - 50) / 50  # Normalize to [-1, 1]
            # Add to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'mfi'
            ] = symbol_data['mfi']

            if self.verbose > 0:
                print(f"    Max MFI val: {symbol_data['mfi'].max()}")
                print(f"    Min MFI val: {symbol_data['mfi'].min()}")
                print(f"    Mean MFI val: {symbol_data['mfi'].mean()}")

        if self.verbose > 0:
            print("\nEnriched data after adding technical indicators:")
            print(f"Shape of enriched data: {self.enriched_data.shape}")
            print(self.enriched_data.head(150))

    def pre_trading_risk_metrics(self):
        """
        Calculate risk metrics from raw financial data only.
        These provide market context for the PPO agent's decision-making.
        """
        # Copy enriched_data
        pre_trading_risk_metrics = self.enriched_data.copy()

        for symbol in pre_trading_risk_metrics['Symbol'].unique():
            symbol_data = pre_trading_risk_metrics[pre_trading_risk_metrics['Symbol'] == symbol].sort_values('Date').copy()
            print(f"\nCalculating pre-trading risk metrics for {symbol}...")

            # Volatility metrics - Rolling standard deviation of returns
            # Calculate rolling volatility and its percentile rank
            volatility_periods = [5, 20, 60]
            for period in volatility_periods:
                realized_vol = symbol_data['return_1d'].rolling(window=period).std() * np.sqrt(252)  # Annualized volatility
                # percentile rank vs historical volatility
                vol_percentile = realized_vol.rolling(window=252, min_periods=80).rank(pct=True)
                symbol_data[f'volatility_{period}d'] = (vol_percentile - 0.5) * 2  # Normalize to [-1, 1]
                # Add to enriched_data
                self.enriched_data.loc[
                    (self.enriched_data['Symbol'] == symbol) &
                    (self.enriched_data['Date'].isin(symbol_data['Date'])),
                    f'volatility_{period}d'
                ] = symbol_data[f'volatility_{period}d']

            if self.verbose > 0:
                print(f"    Max volatility val: {symbol_data[[f'volatility_{p}d' for p in volatility_periods]].max().max()}")
                print(f"    Min volatility val: {symbol_data[[f'volatility_{p}d' for p in volatility_periods]].min().min()}")
                print(f"    Mean volatility val: {symbol_data[[f'volatility_{p}d' for p in volatility_periods]].mean().mean()}")

            # Volatility clustering (GARCH-like effect)
            vol_20d = symbol_data['return_1d'].rolling(20).std()
            vol_change = vol_20d.pct_change(5)  # 5-day change in volatility
            vol_change_norm = (vol_change - vol_change.rolling(60).mean()) / (vol_change.rolling(60).std() + 1e-8)
            symbol_data['vol_clustering'] = vol_change_norm.clip(-3, 3)
            # Add to pre_trading_risk_metrics
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'vol_clustering'
            ] = symbol_data['vol_clustering']

            if self.verbose > 0:
                print(f"    Max volatility clustering val: {symbol_data['vol_clustering'].max()}")
                print(f"    Min volatility clustering val: {symbol_data['vol_clustering'].min()}")
                print(f"    Mean volatility clustering val: {symbol_data['vol_clustering'].mean()}")

            # 2. TAIL RISK FROM RETURN DISTRIBUTION
            # Rolling VaR approximation using quantiles
            for confidence in [0.05, 0.01]:  # 5% and 1% tail events
                var_raw = symbol_data['return_1d'].rolling(60, min_periods=30).quantile(confidence)
                # Normalize by rolling std to make comparable across assets
                rolling_std_60d = symbol_data['return_1d'].rolling(60).std()
                var_normalized = var_raw / (rolling_std_60d + 1e-8)
                symbol_data[f'tail_risk_{int(confidence*100)}pct'] = var_normalized.clip(-5, 0)
                # Add to pre_trading_risk_metrics
                self.enriched_data.loc[
                    (self.enriched_data['Symbol'] == symbol) &
                    (self.enriched_data['Date'].isin(symbol_data['Date'])),
                    f'tail_risk_{int(confidence*100)}pct'
                ] = symbol_data[f'tail_risk_{int(confidence*100)}pct']
                
            if self.verbose > 0:
                print(f"    Max tail risk val: {symbol_data[[f'tail_risk_{int(c*100)}pct' for c in [0.05,0.01]]].max().max()}")
                print(f"    Min tail risk val: {symbol_data[[f'tail_risk_{int(c*100)}pct' for c in [0.05,0.01]]].min().min()}")
                print(f"    Mean tail risk val: {symbol_data[[f'tail_risk_{int(c*100)}pct' for c in [0.05,0.01]]].mean().mean()}")

            # Expected shortfall (CVaR) approximation
            def calculate_expected_shortfall(returns, confidence):
                var_threshold = returns.quantile(confidence)
                expected_shortfall = returns[returns <= var_threshold].mean()
                return expected_shortfall
            for confidence in [0.05, 0.01]:
                es_raw = symbol_data['return_1d'].rolling(60, min_periods=30).apply(
                    lambda x: calculate_expected_shortfall(x, confidence), raw=False
                )
                es_normalized = es_raw / (rolling_std_60d + 1e-8)
                symbol_data[f'expected_shortfall_{int(confidence*100)}pct'] = es_normalized.clip(-5, 0)
                # Add to enriched_data
                self.enriched_data.loc[
                    (self.enriched_data['Symbol'] == symbol) &
                    (self.enriched_data['Date'].isin(symbol_data['Date'])),
                    f'expected_shortfall_{int(confidence*100)}pct'
                ] = symbol_data[f'expected_shortfall_{int(confidence*100)}pct']

            if self.verbose > 0:
                print(f"    Max expected shortfall val: {symbol_data[[f'expected_shortfall_{int(c*100)}pct' for c in [0.05,0.01]]].max().max()}")
                print(f"    Min expected shortfall val: {symbol_data[[f'expected_shortfall_{int(c*100)}pct' for c in [0.05,0.01]]].min().min()}")
                print(f"    Mean expected shortfall val: {symbol_data[[f'expected_shortfall_{int(c*100)}pct' for c in [0.05,0.01]]].mean().mean()}")

            # Return distribution shape
            rolling_skew = symbol_data['return_1d'].rolling(60, min_periods=30).skew()
            symbol_data['return_skewness'] = rolling_skew.clip(-3, 3)
            
            rolling_kurt = symbol_data['return_1d'].rolling(60, min_periods=30).kurt()
            symbol_data['return_kurtosis'] = ((rolling_kurt - 3) / 3).clip(-2, 5)  # Excess kurtosis
            # Add to pre_trading_risk_metrics
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'return_skewness'
            ] = symbol_data['return_skewness']
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'return_kurtosis'
            ] = symbol_data['return_kurtosis']
            if self.verbose > 0:
                print(f"    Max return skewness val: {symbol_data['return_skewness'].max()}")
                print(f"    Min return skewness val: {symbol_data['return_skewness'].min()}")
                print(f"    Mean return skewness val: {symbol_data['return_skewness'].mean()}")
                print(f"    Max return kurtosis val: {symbol_data['return_kurtosis'].max()}")
                print(f"    Min return kurtosis val: {symbol_data['return_kurtosis'].min()}")
                print(f"    Mean return kurtosis val: {symbol_data['return_kurtosis'].mean()}")

            # 3. TREND PERSISTENCE AND REVERSALS
            # Autocorrelation of returns (momentum vs mean reversion)
            autocorr_1d = symbol_data['return_1d'].rolling(30).apply(
                lambda x: x.autocorr(lag=1) if len(x.dropna()) > 15 else 0, raw=False
            )
            symbol_data['momentum_persistence'] = autocorr_1d.fillna(0).clip(-1, 1)
            # Add to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'momentum_persistence'
            ] = symbol_data['momentum_persistence']
            if self.verbose > 0:
                print(f"    Max momentum persistence val: {symbol_data['momentum_persistence'].max()}")
                print(f"    Min momentum persistence val: {symbol_data['momentum_persistence'].min()}")
                print(f"    Mean momentum persistence val: {symbol_data['momentum_persistence'].mean()}")

            # Trend strength relative to volatility
            for ma_period in [20, 50]:
                ma = symbol_data['Close'].rolling(ma_period).mean()
                trend_strength = (symbol_data['Close'] - ma) / (symbol_data['Close'].rolling(ma_period).std() + 1e-8)
                symbol_data[f'trend_strength_{ma_period}d'] = trend_strength.clip(-5, 5)
                # Add to enriched_data
                self.enriched_data.loc[
                    (self.enriched_data['Symbol'] == symbol) &
                    (self.enriched_data['Date'].isin(symbol_data['Date'])),
                    f'trend_strength_{ma_period}d'
                ] = symbol_data[f'trend_strength_{ma_period}d']
            if self.verbose > 0:
                print(f"    Max trend strength val: {symbol_data[[f'trend_strength_{p}d' for p in [20,50]]].max().max()}")
                print(f"    Min trend strength val: {symbol_data[[f'trend_strength_{p}d' for p in [20,50]]].min().min()}")
                print(f"    Mean trend strength val: {symbol_data[[f'trend_strength_{p}d' for p in [20,50]]].mean().mean()}")

            # 4. STRESS INDICATORS
            # Distance from recent highs (stress indicator)
            rolling_max_252 = symbol_data['Close'].rolling(252, min_periods=20).max()
            distance_from_high = (symbol_data['Close'] - rolling_max_252) / rolling_max_252
            symbol_data['stress_indicator'] = distance_from_high.clip(-1, 0)
            # Add to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'stress_indicator'
            ] = symbol_data['stress_indicator']
            if self.verbose > 0:
                print(f"    Max stress indicator val: {symbol_data['stress_indicator'].max()}")
                print(f"    Min stress indicator val: {symbol_data['stress_indicator'].min()}")
                print(f"    Mean stress indicator val: {symbol_data['stress_indicator'].mean()}")

            # Consecutive down days (capitulation indicator)
            down_days = (symbol_data['return_1d'] < 0).astype(int)
            consecutive_down = down_days.groupby((down_days == 0).cumsum()).cumsum()
            max_consecutive = consecutive_down.rolling(252, min_periods=60).max()
            symbol_data['consecutive_stress'] = (consecutive_down / (max_consecutive + 1)).clip(0, 1)

            # Add to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'consecutive_stress'
            ] = symbol_data['consecutive_stress']
            if self.verbose > 0:
                print(f"    Max consecutive stress val: {symbol_data['consecutive_stress'].max()}")
                print(f"    Min consecutive stress val: {symbol_data['consecutive_stress'].min()}")
                print(f"    Mean consecutive stress val: {symbol_data['consecutive_stress'].mean()}")

            # 5. VOLUME-BASED RISK SIGNALS
            # Volume spike detection (often precedes volatility)
            volume_ma_20 = symbol_data['Volume'].rolling(20).mean()
            volume_spike = symbol_data['Volume'] / (volume_ma_20 + 1e-8)
            volume_spike_norm = (volume_spike - volume_spike.rolling(60).mean()) / (volume_spike.rolling(60).std() + 1e-8)
            symbol_data['volume_risk_signal'] = volume_spike_norm.clip(-3, 3)

            # Add to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'volume_risk_signal'
            ] = symbol_data['volume_risk_signal']
            if self.verbose > 0:
                print(f"    Max volume risk signal val: {symbol_data['volume_risk_signal'].max()}")
                print(f"    Min volume risk signal val: {symbol_data['volume_risk_signal'].min()}")
                print(f"    Mean volume risk signal val: {symbol_data['volume_risk_signal'].mean()}")

            # Volume-price divergence (risk warning)
            price_momentum = symbol_data['return_20d']  # Use existing 20-day return
            volume_momentum = symbol_data['volume_change_20d']  # Use existing volume change
            # Normalize both to same scale first
            price_mom_norm = (price_momentum - price_momentum.rolling(60, min_periods=10).mean()) / (price_momentum.rolling(60, min_periods=10).std() + 1e-8)
            volume_mom_norm = (volume_momentum - volume_momentum.rolling(60, min_periods=10).mean()) / (volume_momentum.rolling(60, min_periods=10).std() + 1e-8)
            # True divergence: when signs are opposite
            vp_divergence = price_mom_norm - volume_mom_norm  # Large positive = price up, volume down (divergence)
            symbol_data['volume_price_divergence'] = vp_divergence.clip(-3, 3)
            # Add to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'volume_price_divergence'
            ] = symbol_data['volume_price_divergence']
            if self.verbose > 0:
                print(f"    Max volume-price divergence val: {symbol_data['volume_price_divergence'].max()}")
                print(f"    Min volume-price divergence val: {symbol_data['volume_price_divergence'].min()}")
                print(f"    Mean volume-price divergence val: {symbol_data['volume_price_divergence'].mean()}")

            # Crisis detection composite
            crisis_df = pd.DataFrame({
                'stress_indicator': symbol_data['stress_indicator'] * 2,  # Weight 2x
                'consecutive_stress': symbol_data['consecutive_stress'] * -2,  # Negative because high stress = crisis
                'vol_clustering': symbol_data['vol_clustering'] / 3,  # Normalize to [-1, 1]
                'tail_risk_1pct': symbol_data['tail_risk_1pct'] / 5,  # Normalize extreme tail risk
                'return_kurtosis': symbol_data['return_kurtosis'] / 5,   # High kurtosis = crisis-like
                'bb_extreme': np.abs(symbol_data['bb_position']) > 0.8,  # Bollinger extreme
                'rsi_extreme': np.abs(symbol_data['rsi_14']) > 0.7,       # RSI extreme
                'volume_spike': symbol_data['volume_risk_signal'] > 2,    # Volume spike
                'macd_divergence': np.sign(symbol_data['macd']) != np.sign(symbol_data['macd_signal'])
            })
            crisis_score = crisis_df.mean(axis=1, skipna=True)
            symbol_data['crisis_detection'] = crisis_score.clip(-3, 3)
            # Add to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'crisis_detection'
            ] = symbol_data['crisis_detection']
            if self.verbose > 0:
                print(f"    Max crisis detection val: {symbol_data['crisis_detection'].max()}")
                print(f"    Min crisis detection val: {symbol_data['crisis_detection'].min()}")
                print(f"    Mean crisis detection val: {symbol_data['crisis_detection'].mean()}")

            # Volatility Regime change detection using rolling correlation shifts
            def detect_regime_change(returns, window=60):
                # Compare recent vs historical volatility patterns
                recent_vol = returns.rolling(20).std()
                historical_vol = returns.rolling(60).std().shift(20)
                vol_ratio = recent_vol / (historical_vol + 1e-8)
                
                # Regime change when volatility patterns shift significantly
                regime_shift = (vol_ratio - vol_ratio.rolling(window).mean()) / (vol_ratio.rolling(window).std() + 1e-8)
                return regime_shift.clip(-3, 3)
            symbol_data['regime_change'] = detect_regime_change(symbol_data['return_1d'])
            # Add to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'regime_change'
            ] = symbol_data['regime_change']
            if self.verbose > 0:
                print(f"    Max regime change val: {symbol_data['regime_change'].max()}")
                print(f"    Min regime change val: {symbol_data['regime_change'].min()}")
                print(f"    Mean regime change val: {symbol_data['regime_change'].mean()}")

            # Liquidity risk proxy using volume and price impact
            def calculate_liquidity_risk(returns, volumes):
                """Price impact proxy: |return| per unit of volume"""
                volume_ma = volumes.rolling(20).mean()
                volume_relative = volumes / (volume_ma + 1e-8)
                
                # Price impact = absolute return / relative volume
                price_impact = np.abs(returns) / (volume_relative + 1e-8)
                
                # Normalize to z-score
                impact_norm = (price_impact - price_impact.rolling(60).mean()) / (price_impact.rolling(60).std() + 1e-8)
                return impact_norm.clip(-3, 3)

            symbol_data['liquidity_risk'] = calculate_liquidity_risk(
                symbol_data['return_1d'], 
                symbol_data['Volume']
            )
            # Add to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'liquidity_risk'
            ] = symbol_data['liquidity_risk']
            if self.verbose > 0:
                print(f"    Max liquidity risk val: {symbol_data['liquidity_risk'].max()}")
                print(f"    Min liquidity risk val: {symbol_data['liquidity_risk'].min()}")
                print(f"    Mean liquidity risk val: {symbol_data['liquidity_risk'].mean()}")

            # Add comprehensive momentum quality:
            def calculate_momentum_quality(symbol_data):
                """Combine multiple factors for momentum quality assessment"""
                # Base momentum persistence
                persistence = symbol_data['momentum_persistence'].fillna(0)
                
                # Volume support (momentum + volume agreement)
                volume_support = np.sign(symbol_data['return_20d']) * np.sign(symbol_data['volume_change_20d'])
                volume_support = volume_support.rolling(10).mean()  # Smooth
                
                # Volatility adjustment (lower quality in high vol)
                vol_penalty = 1 - np.abs(symbol_data['vol_clustering']) / 3
                vol_penalty = vol_penalty.clip(0, 1)
                
                # Trend consistency
                trend_consistency = (symbol_data['return_1d'] > 0).rolling(20).mean()
                trend_consistency = np.abs(trend_consistency - 0.5) * 2  # 0 = random, 1 = consistent
                
                # Combine components
                quality_score = (
                    persistence * 0.4 +
                    volume_support * 0.3 +
                    vol_penalty * 0.15 +
                    trend_consistency * 0.15
                )
                
                return quality_score.clip(-1, 1)

            symbol_data['momentum_quality'] = calculate_momentum_quality(symbol_data)
            # Add to enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'momentum_quality'
            ] = symbol_data['momentum_quality']
            if self.verbose > 0:
                print(f"    Max momentum quality val: {symbol_data['momentum_quality'].max()}")
                print(f"    Min momentum quality val: {symbol_data['momentum_quality'].min()}")
                print(f"    Mean momentum quality val: {symbol_data['momentum_quality'].mean()}")

        if self.verbose > 0:
            print("\nEnriched dataframe after adding pre-trading risk metrics:")
            print(f"Shape of pre-trading risk metrics: {self.enriched_data.shape}")
            print(self.enriched_data.head(700))

    def enhanced_risk_adjusted_metrics(self):
        """
        Add comprehensive risk-adjusted performance metrics to the dataset.
        These are CRITICAL for sophisticated alpha generation.
        """
        print("\n" + "="*50)
        print("CALCULATING RISK-ADJUSTED METRICS")
        print("="*50)
        
        for symbol in self.enriched_data['Symbol'].unique():
            symbol_data = self.enriched_data[self.enriched_data['Symbol'] == symbol].sort_values('Date').copy()
            
            print(f"\nCalculating risk-adjusted metrics for {symbol}...")
            
            # 1. SHARPE RATIO PROXIES (no leakage: use rolling windows only)
            for period in [20, 60]:
                # Get return column (use existing or create)
                if f'return_{period}d' in symbol_data.columns:
                    rolling_return = symbol_data[f'return_{period}d']
                else:
                    rolling_return = symbol_data['Close'].pct_change(period)

                # Calculate rolling volatility (past-window only)
                rolling_vol = symbol_data['return_1d'].rolling(period, min_periods=10).std() * np.sqrt(252)

                # Sharpe proxy (past info only)
                sharpe_proxy = rolling_return / (rolling_vol + 1e-8)

                # Short-term normalization removed (we will scale using local rolling windows below)
                sharpe_normalized = sharpe_proxy  # keep raw proxy, scale with local robust rolling scaler

                # Choose local rolling window length bounded between 20 and 60 days
                window_len = min(max(period, 20), 60)

                def robust_scaling_rolling(values, window_len=60, percentile_cutoff=0.98):
                    """
                    Robust, local scaling using past rolling statistics only to avoid leakage.
                    - Compute rolling upper/lower quantiles on a backward-looking window.
                    - Soft-clip values using those local quantiles.
                    - Scale by local rolling max absolute value to map to approximately [-1, 1].
                    """
                    # Rolling quantiles (past data only)
                    upper = values.rolling(window_len, min_periods=20).quantile(percentile_cutoff)
                    lower = values.rolling(window_len, min_periods=20).quantile(1.0 - percentile_cutoff)

                    # Soft clipping using local quantiles
                    clipped = values.clip(lower * 1.1, upper * 1.1)

                    # Local absolute max for scaling (past window)
                    local_abs_max = clipped.abs().rolling(window_len, min_periods=20).max()

                    # Scale into [-1, 1] using local_abs_max (avoid division by zero)
                    scaled = clipped / (local_abs_max + 1e-8)

                    # Fill early periods with 0 (or keep NaN if preferred)
                    return scaled.fillna(0)

                symbol_data[f'risk_adjusted_return_{period}d'] = robust_scaling_rolling(sharpe_normalized, window_len=window_len)
            
            # 2. CALMAR RATIO PROXIES (Return / Max Drawdown)
            for period in [30, 60]:
                rolling_max = symbol_data['Close'].rolling(period).max()
                current_drawdown = (symbol_data['Close'] - rolling_max) / rolling_max
                max_drawdown = current_drawdown.rolling(period).min()
                
                # Use 60d return for both periods
                rolling_return = symbol_data['return_60d'] if 'return_60d' in symbol_data.columns else symbol_data['Close'].pct_change(60)
                calmar_proxy = rolling_return / (abs(max_drawdown) + 1e-8)
                
                calmar_mean = calmar_proxy.rolling(252, min_periods=60).mean()
                calmar_std = calmar_proxy.rolling(252, min_periods=60).std()
                calmar_normalized = (calmar_proxy - calmar_mean) / (calmar_std + 1e-8)
                
                symbol_data[f'drawdown_adjusted_return_{period}d'] = calmar_normalized.clip(-3, 3)
            
            # 3. TAIL RISK-ADJUSTED RETURNS
            for period in [20, 60]:
                rolling_return = symbol_data['return_20d']
                tail_risk = symbol_data['tail_risk_1pct']
                expected_shortfall = symbol_data['expected_shortfall_1pct']
                
                # Return per unit of tail risk
                var_adjusted = rolling_return / (abs(tail_risk) + 1e-8)
                es_adjusted = rolling_return / (abs(expected_shortfall) + 1e-8)
                
                # Normalize
                var_adj_mean = var_adjusted.rolling(252, min_periods=60).mean()
                var_adj_std = var_adjusted.rolling(252, min_periods=60).std()
                var_adj_norm = (var_adjusted - var_adj_mean) / (var_adj_std + 1e-8)
                
                es_adj_mean = es_adjusted.rolling(252, min_periods=60).mean()
                es_adj_std = es_adjusted.rolling(252, min_periods=60).std()
                es_adj_norm = (es_adjusted - es_adj_mean) / (es_adj_std + 1e-8)
                
                symbol_data[f'var_adjusted_return_{period}d'] = var_adj_norm.clip(-3, 3)
                symbol_data[f'es_adjusted_return_{period}d'] = es_adj_norm.clip(-3, 3)
            
            # 4. VOLATILITY-ADJUSTED MOMENTUM (Enhanced adjustment)
            def enhanced_vol_adjustment(raw_momentum, vol_regime, momentum_quality):
                # Base volatility penalty
                vol_penalty = 1 + abs(vol_regime) * 0.5
                # Additional penalty for low-quality momentum in high vol
                quality_adjustment = 1 + (1 - momentum_quality) * abs(vol_regime) * 0.3
                # Combined adjustment
                total_adjustment = vol_penalty * quality_adjustment
                return raw_momentum / (total_adjustment + 1e-8)

            for period in [10, 20]:
                if f'momentum_{period}' in symbol_data.columns:
                    raw_momentum = symbol_data[f'momentum_{period}']
                    vol_regime = symbol_data['volatility_20d']
                    momentum_quality = symbol_data['momentum_quality'] if 'momentum_quality' in symbol_data.columns else 1.0
                    
                    # Enhanced adjustment
                    vol_adjusted_momentum = enhanced_vol_adjustment(raw_momentum, vol_regime, momentum_quality)
                    symbol_data[f'vol_adjusted_momentum_{period}d'] = vol_adjusted_momentum.clip(-3, 3)
            
            # 5. SKEWNESS-ADJUSTED RETURNS
            rolling_return = symbol_data['return_20d']
            return_skewness = symbol_data['return_skewness']
            
            skewness_adjustment = 1 + (return_skewness * 0.2)
            skewness_adjusted_return = rolling_return * skewness_adjustment
            
            skew_adj_mean = skewness_adjusted_return.rolling(252, min_periods=60).mean()
            skew_adj_std = skewness_adjusted_return.rolling(252, min_periods=60).std()
            skew_adj_norm = (skewness_adjusted_return - skew_adj_mean) / (skew_adj_std + 1e-8)
            
            symbol_data['skewness_adjusted_return'] = skew_adj_norm.clip(-3, 3)
            
            # 6. COMPOSITE RISK SCORE
            # Combine multiple risk factors into single score
            risk_factors = [
                symbol_data['volatility_20d'],
                symbol_data['tail_risk_1pct'] / 5,  # Normalize
                abs(symbol_data['return_skewness']) / 3,  # Absolute skewness
                symbol_data['return_kurtosis'] / 5,
                symbol_data['stress_indicator'] * 2
            ]
            
            composite_risk = np.mean(risk_factors, axis=0)
            symbol_data['composite_risk_score'] = composite_risk.clip(-1, 1)
            
            # 7. RISK-ADJUSTED MOMENTUM QUALITY
            momentum_quality = symbol_data['momentum_quality']
            composite_risk_abs = abs(symbol_data['composite_risk_score'])
            
            # Higher risk = lower quality adjustment
            risk_penalty = 1 - composite_risk_abs * 0.5
            risk_adjusted_momentum_quality = momentum_quality * risk_penalty
            
            symbol_data['risk_adjusted_momentum_quality'] = risk_adjusted_momentum_quality.clip(-1, 1)
            
            # Add all metrics to enriched_data
            risk_adjusted_metrics = [
                'risk_adjusted_return_20d', 'risk_adjusted_return_60d',
                'drawdown_adjusted_return_30d', 'drawdown_adjusted_return_60d',
                'var_adjusted_return_20d', 'var_adjusted_return_60d',
                'es_adjusted_return_20d', 'es_adjusted_return_60d',
                'vol_adjusted_momentum_10d', 'vol_adjusted_momentum_20d',
                'skewness_adjusted_return',
                'composite_risk_score', 'risk_adjusted_momentum_quality'
            ]
            
            for metric in risk_adjusted_metrics:
                if metric in symbol_data.columns:
                    self.enriched_data.loc[
                        (self.enriched_data['Symbol'] == symbol) &
                        (self.enriched_data['Date'].isin(symbol_data['Date'])),
                        metric
                    ] = symbol_data[metric]
            
            if self.verbose > 0:
                # Print statistics
                print(f"    Risk-adjusted return (20d) range: [{symbol_data['risk_adjusted_return_20d'].min():.3f}, {symbol_data['risk_adjusted_return_20d'].max():.3f}] Mean: {symbol_data['risk_adjusted_return_20d'].mean():.3f} Median: {symbol_data['risk_adjusted_return_20d'].median():.3f}")
                print(f"    Risk-adjusted return (60d) range: [{symbol_data['risk_adjusted_return_60d'].min():.3f}, {symbol_data['risk_adjusted_return_60d'].max():.3f}] Mean: {symbol_data['risk_adjusted_return_60d'].mean():.3f} Median: {symbol_data['risk_adjusted_return_60d'].median():.3f}")
                print(f"    Drawdown-adjusted return (30d) range: [{symbol_data['drawdown_adjusted_return_30d'].min():.3f}, {symbol_data['drawdown_adjusted_return_30d'].max():.3f}] Mean: {symbol_data['drawdown_adjusted_return_30d'].mean():.3f} Median: {symbol_data['drawdown_adjusted_return_30d'].median():.3f}")
                print(f"    Drawdown-adjusted return (60d) range: [{symbol_data['drawdown_adjusted_return_60d'].min():.3f}, {symbol_data['drawdown_adjusted_return_60d'].max():.3f}] Mean: {symbol_data['drawdown_adjusted_return_60d'].mean():.3f} Median: {symbol_data['drawdown_adjusted_return_60d'].median():.3f}")
                print(f"    VAR-adjusted return (20d) range: [{symbol_data['var_adjusted_return_20d'].min():.3f}, {symbol_data['var_adjusted_return_20d'].max():.3f}] Mean: {symbol_data['var_adjusted_return_20d'].mean():.3f} Median: {symbol_data['var_adjusted_return_20d'].median():.3f}")
                print(f"    VAR-adjusted return (60d) range: [{symbol_data['var_adjusted_return_60d'].min():.3f}, {symbol_data['var_adjusted_return_60d'].max():.3f}] Mean: {symbol_data['var_adjusted_return_60d'].mean():.3f} Median: {symbol_data['var_adjusted_return_60d'].median():.3f}")
                print(f"    ES-adjusted return (20d) range: [{symbol_data['es_adjusted_return_20d'].min():.3f}, {symbol_data['es_adjusted_return_20d'].max():.3f}] Mean: {symbol_data['es_adjusted_return_20d'].mean():.3f} Median: {symbol_data['es_adjusted_return_20d'].median():.3f}")
                print(f"    ES-adjusted return (60d) range: [{symbol_data['es_adjusted_return_60d'].min():.3f}, {symbol_data['es_adjusted_return_60d'].max():.3f}] Mean: {symbol_data['es_adjusted_return_60d'].mean():.3f} Median: {symbol_data['es_adjusted_return_60d'].median():.3f}")
                print(f"    Vol-adjusted momentum (10d) range: [{symbol_data['vol_adjusted_momentum_10d'].min():.3f}, {symbol_data['vol_adjusted_momentum_10d'].max():.3f}] Mean: {symbol_data['vol_adjusted_momentum_10d'].mean():.3f} Median: {symbol_data['vol_adjusted_momentum_10d'].median():.3f}")
                print(f"    Vol-adjusted momentum (20d) range: [{symbol_data['vol_adjusted_momentum_20d'].min():.3f}, {symbol_data['vol_adjusted_momentum_20d'].max():.3f}] Mean: {symbol_data['vol_adjusted_momentum_20d'].mean():.3f} Median: {symbol_data['vol_adjusted_momentum_20d'].median():.3f}")
                print(f"    Skewness-adjusted return range: [{symbol_data['skewness_adjusted_return'].min():.3f}, {symbol_data['skewness_adjusted_return'].max():.3f}] Mean: {symbol_data['skewness_adjusted_return'].mean():.3f} Median: {symbol_data['skewness_adjusted_return'].median():.3f}")
                print(f"    Composite risk score range: [{symbol_data['composite_risk_score'].min():.3f}, {symbol_data['composite_risk_score'].max():.3f}] Mean: {symbol_data['composite_risk_score'].mean():.3f} Median: {symbol_data['composite_risk_score'].median():.3f}")
                print(f"    Risk-adjusted momentum quality range: [{symbol_data['risk_adjusted_momentum_quality'].min():.3f}, {symbol_data['risk_adjusted_momentum_quality'].max():.3f}] Mean: {symbol_data['risk_adjusted_momentum_quality'].mean():.3f} Median: {symbol_data['risk_adjusted_momentum_quality'].median():.3f}")

        print(f"\nShape of enriched_data: {self.enriched_data.shape}")

    def cross_asset_relationships(self):
        """"
        Add cross-asset relationship metrics to the dataset.
        These metrics capture interdependencies between different assets, 
        such as correlations and co-movements.
        """
        print("\n" + "="*50)
        print("CALCULATING CROSS-ASSET RELATIONSHIP METRICS")
        print("="*50)

        # 1. Relative performance Analysis
        self.calculate_relative_performance_metrics()
        print(f"\nRelative performance metrics completed!")
        self.NaN_counting()

        # 2. Cross-asset momentum regimes
        self.calculate_cross_asset_momentum_regimes()
        print(f"\nCross-asset momentum regime metrics completed!")
        self.NaN_counting()

        # 3. Volatility spillover effects
        self.calculate_volatility_spillovers()
        print(f"\nVolatility spillover metrics completed!")
        self.NaN_counting()

        # 4. Market leadership and beta dynamics
        self.calculate_market_leadership_metrics()
        print(f"\nMarket leadership and beta dynamics metrics completed!")
        self.NaN_counting()

        # 5. Cross-asset correlations
        self.calculate_cross_asset_correlations()
        print(f"\nCross-asset correlation metrics completed!")
        self.NaN_counting()

        print(f"\nCross-asset relationship metrics completed!")
        print(f"Enhanced dataset shape: {self.enriched_data.shape}")


    def add_cross_asset_features(self):
        """
        Calculates high-level cross-asset signals:
        - Fisher-Z Correlation with SPY
        - Cross-sectional Dispersion
        - Rolling Beta
        
        NOTE: Uses date-aligned operations like convert_raw_to_stationary_data!

        """
        print("Calculating cross-asset correlation features...")

        # Use the enriched_data which already has returns calculated
        df = self.enriched_data.copy()  # Make explicit copy

        # --- 1. Preparation: Get Benchmark Returns (indexed by Date) ---
        spy_ret = df[df['Symbol'] == 'SPY'].set_index('Date')['log_close_return_1d']
        
        # --- 2. Rolling Correlation & Fisher-Z Transform (FIXED) ---
        # Use groupby + transform with explicit per-symbol date alignment
        def calc_fisher_corr(group):
            """
            group is a Series with the default integer index from groupby-transform.
            We need to align it to dates, compute rolling correlation, then return.
            """
            # Get the corresponding dates for this symbol's group
            symbol_dates = df[df['Symbol'] == group.name].set_index('Date').index
            # Reindex the group by dates (this preserves alignment)
            group_with_dates = pd.Series(group.values, index=symbol_dates)
            
            # Get SPY returns for these dates
            spy_ret_aligned = spy_ret.reindex(symbol_dates)
            
            # Compute rolling correlation
            corr = group_with_dates.rolling(window=21).corr(spy_ret_aligned)
            
            # Fisher Z-Transform: 0.5 * ln((1+r)/(1-r))
            corr = corr.clip(-0.99, 0.99)
            fisher_z = 0.5 * np.log((1 + corr) / (1 - corr))
            
            # Return aligned to original index positions
            return fisher_z.values
        
        df['z_fisher_corr_spy'] = df.groupby('Symbol')['log_close_return_1d'].transform(calc_fisher_corr)

        # --- 3. Cross-Sectional Dispersion (Market 'Spread') ---
        # This already works in original code - group by Date, calculate std across symbols
        dispersion = df.groupby('Date')['log_close_return_1d'].std()
        df['market_dispersion'] = df['Date'].map(np.log(dispersion + 1e-8))
        
        # Z-score the dispersion to make it stationary
        df['z_market_dispersion'] = (df['market_dispersion'] - df['market_dispersion'].rolling(window=self.max_norm_horizon).mean()) / \
                                    (df['market_dispersion'].rolling(window=self.max_norm_horizon).std() + 1e-8)

        # --- 4. Rolling Beta (FIXED) ---
        def calc_beta(group):
            """
            Beta = Cov(Asset, Market) / Var(Market)
            Similar structure to calc_fisher_corr for proper date alignment
            """
            symbol_dates = df[df['Symbol'] == group.name].set_index('Date').index
            group_with_dates = pd.Series(group.values, index=symbol_dates)
            spy_ret_aligned = spy_ret.reindex(symbol_dates)
            
            # Calculate covariance and variance with proper alignment
            covariance = group_with_dates.rolling(window=self.max_norm_horizon).cov(spy_ret_aligned)
            market_variance = spy_ret_aligned.rolling(window=self.max_norm_horizon).var()
            
            beta = covariance / (market_variance + 1e-8)
            return beta.values
        
        df['beta_spy'] = df.groupby('Symbol')['log_close_return_1d'].transform(calc_beta)
        
        # Final Z-score for Beta to normalize across assets
        df['z_beta_spy'] = df.groupby('Symbol')['beta_spy'].transform(
            lambda x: (x - x.rolling(window=self.max_norm_horizon).mean()) / (x.rolling(window=self.max_norm_horizon).std() + 1e-8)
        )

        # Add the new features to enriched_data at correct symbol and time alignment
        for feature in ['z_fisher_corr_spy', 'z_market_dispersion', 'z_beta_spy']:
            self.enriched_data[feature] = df[feature]

    def calculate_relative_performance_metrics(self):
        """
        Calculate relative performance vs other assets - KEY for asset selection
        Helps agent identify which assets are outperforming/underperforming
        """
        
        # Create pivot for multi-timeframe analysis
        pivot_data = {}
        for period in ['20d', '40d']:
            pivot_data[period] = self.enriched_data.pivot_table(
                values=f'return_{period}', index='Date', columns='Symbol'
            )
        
        relative_performance_features = {}
        
        for symbol in self.assets:
            for period in ['20d', '40d']:
                # Use the pivot table's columns as the source of truth for available assets
                available_assets_in_pivot = pivot_data[period].columns.tolist()

                if symbol in pivot_data[period].columns:
                    symbol_returns = pivot_data[period][symbol]
                    other_assets = [s for s in self.assets if s != symbol and s in pivot_data[period].columns]
                    
                    if other_assets:
                        # 1. Relative performance vs average of other assets
                        other_avg = pivot_data[period][other_assets].mean(axis=1, skipna=True)
                        relative_performance = symbol_returns - other_avg
                        
                        # Normalize relative performance
                        rel_perf_norm = (relative_performance - relative_performance.rolling(252, min_periods=60).mean()) / (relative_performance.rolling(252, min_periods=60).std() + 1e-8)
                        
                        # 2. Relative performance percentile rank (THE FIX IS HERE)
                        # Only rank the assets that are actually in the pivot table for this period.
                        all_assets_returns = pivot_data[period][available_assets_in_pivot].T  # Transpose for ranking
                        relative_rank = all_assets_returns.rank(axis=0, pct=True, na_option='keep').loc[symbol]
                        relative_rank_centered = (relative_rank - 0.5) * 2  # Center at 0, range [-1, 1]
                        
                        # 3. Relative momentum (outperformance acceleration)
                        relative_momentum = relative_performance.diff(5)  # 5-day change in relative performance
                        rel_mom_norm = (relative_momentum - relative_momentum.rolling(60, min_periods=30).mean()) / (relative_momentum.rolling(60, min_periods=30).std() + 1e-8)
                        
                        # Store features
                        relative_performance_features[f'{symbol}_relative_performance_{period}'] = rel_perf_norm.clip(-3, 3)
                        relative_performance_features[f'{symbol}_relative_rank_{period}'] = relative_rank_centered
                        relative_performance_features[f'{symbol}_relative_momentum_{period}'] = rel_mom_norm.clip(-3, 3)
        
        # Use optimized feature addition
        self.optimized_add_cross_asset_features(relative_performance_features)
        
        if self.verbose > 0:
            # Print statistics for debugging and clarification
            print("\nRelative performance features added:")
            for feature_name in relative_performance_features.keys():
                symbol = feature_name.split('_')[0]
                feature_short_name = '_'.join(feature_name.split('_')[1:])
                feature_data = self.enriched_data[self.enriched_data['Symbol'] == symbol][feature_short_name]
                print(f"  {feature_name}: min={feature_data.min():.3f}, max={feature_data.max():.3f}, mean={feature_data.mean():.3f}, median={feature_data.median():.3f}")
        

    def calculate_cross_asset_momentum_regimes(self):
        """
        Identify market-wide momentum vs mean reversion regimes
        Helps agent adapt strategy based on overall market behavior
        """
        
        # Create momentum pivot table
        momentum_data = {}
        for asset in self.assets:
            asset_data = self.enriched_data[self.enriched_data['Symbol'] == asset].set_index('Date')
            if len(asset_data) > 0 and 'momentum_20' in asset_data.columns:
                momentum_data[asset] = asset_data['momentum_20']
        
        momentum_df = pd.DataFrame(momentum_data)
        full_index = momentum_df.index.sort_values()
        
        regime_features = {}
        
        # Precompute per-date availability counts and absolute momentum averages
        # availability_count: number of assets with non-null momentum on that date
        availability_count = momentum_df.notna().sum(axis=1)
        # avg absolute momentum across available assets (skipna semantics)
        momentum_strength_avg = momentum_df.abs().mean(axis=1, skipna=True)

        # Compute market-level signed consensus: fraction of available assets with positive momentum
        positive_mask = momentum_df > 0
        # Use per-date denom = availability_count (avoid dividing by len(self.assets) when some assets missing)
        positive_counts = positive_mask.sum(axis=1)
        market_momentum_consensus = positive_counts / availability_count.replace(0, np.nan)
        market_consensus_centered = ((market_momentum_consensus - 0.5) * 2).fillna(0)

        # Loop per symbol and compute metrics using safe skipna operations
        for symbol in self.assets:
            if symbol not in momentum_df.columns:
                continue

            symbol_series = momentum_df[symbol]

            # other assets present on any given date (we'll compute per-date by using DataFrame ops)
            other_df = momentum_df.drop(columns=[symbol])
            other_avg = other_df.mean(axis=1, skipna=True)

            # 1. Momentum regime consensus (how many assets showing same momentum direction)
            symbol_momentum_positive = symbol_series > 0
            # others_momentum_positive measure uses the average sign across available others
            others_positive_pct = (other_df > 0).sum(axis=1) / other_df.notna().sum(axis=1).replace(0, np.nan)
            others_momentum_positive = others_positive_pct > 0.5

            # 2. Momentum persistence across assets (robust, Inf-safe)
            sym_pos_float = symbol_momentum_positive.astype(float)
            # Use raw fraction of other positives (already float in [0,1])
            others_pos_float = others_positive_pct.astype(float)

            window = 60
            min_periods = 10
            eps = 1e-8

            # Precompute rolling means
            x_mean = sym_pos_float.rolling(window, min_periods=min_periods).mean()
            y_mean = others_pos_float.rolling(window, min_periods=min_periods).mean()

            # E[x*y], E[x^2], E[y^2]
            xy_mean = (sym_pos_float * others_pos_float).rolling(window, min_periods=min_periods).mean()
            x2_mean = (sym_pos_float ** 2).rolling(window, min_periods=min_periods).mean()
            y2_mean = (others_pos_float ** 2).rolling(window, min_periods=min_periods).mean()

            cov = xy_mean - x_mean * y_mean
            var_x = (x2_mean - x_mean ** 2).clip(lower=0)
            var_y = (y2_mean - y_mean ** 2).clip(lower=0)

            denom = (var_x * var_y).pow(0.5)
            # Safe correlation
            momentum_direction_corr = cov / (denom + eps)
            # Where variance effectively zero -> set to 0 (no information)
            momentum_direction_corr[denom < eps] = 0.0

            # Final cleanup & bounding
            momentum_direction_corr = momentum_direction_corr.clip(-1, 1)
            momentum_direction_corr = momentum_direction_corr.replace([np.inf, -np.inf], 0).fillna(0)

            # 3. Counter-trend opportunities (when asset diverges from market momentum)
            momentum_divergence = symbol_series - other_avg
            momentum_div_norm = (momentum_divergence - momentum_divergence.rolling(252, min_periods=60).mean()) / (momentum_divergence.rolling(252, min_periods=60).std() + 1e-8)

            # 4. Mean reversion signal strength (market-wide)
            # extreme consensus based on available data
            extreme_consensus = market_consensus_centered.abs() > 0.6
            momentum_strength_threshold = momentum_strength_avg.rolling(60, min_periods=30).quantile(0.7)
            mean_reversion_signal = (extreme_consensus) & (momentum_strength_avg > momentum_strength_threshold)
            mean_reversion_signal = mean_reversion_signal.fillna(False).astype(float)

            # Store features (align series index with dates where symbol exists)
            idx = symbol_series.dropna().index
            regime_features[f'{symbol}_market_momentum_consensus'] = pd.Series(market_consensus_centered, index=momentum_df.index)
            regime_features[f'{symbol}_momentum_direction_corr'] = pd.Series(momentum_direction_corr.fillna(0), index=momentum_df.index)
            regime_features[f'{symbol}_momentum_divergence'] = pd.Series(momentum_div_norm.clip(-3, 3), index=momentum_df.index)
            regime_features[f'{symbol}_mean_reversion_signal'] = pd.Series(mean_reversion_signal.astype(float), index=momentum_df.index)

        # Add features back into enriched_data with the optimized updater
        self.optimized_add_cross_asset_features(regime_features)

        if self.verbose > 0:
            # Print statistics for debugging and clarification
            print("\nCross-asset momentum regime features added:")
            for feature_name in regime_features.keys():
                symbol = feature_name.split('_')[0]
                feature_short_name = '_'.join(feature_name.split('_')[1:])
                feature_data = self.enriched_data[self.enriched_data['Symbol'] == symbol][feature_short_name]
                print(f"  {feature_name}: min={feature_data.min():.3f}, max={feature_data.max():.3f}, mean={feature_data.mean():.3f}, median={feature_data.median():.3f}")


    def calculate_volatility_spillovers(self):
        """
        Calculate volatility spillover effects between assets in a robust, NaN-preserving way.
        Critical for understanding contagion risk and portfolio diversification.
        """
        print("\nCalculating volatility spillover features (robustly)...")

        # Create a dictionary of volatility series for each asset
        volatility_data = {}
        for asset in self.assets:
            asset_data = self.enriched_data[self.enriched_data['Symbol'] == asset].set_index('Date')
            if 'return_1d' in asset_data.columns:
                # Use realized volatility (20-day rolling std of daily returns), annualized
                volatility_data[asset] = asset_data['return_1d'].rolling(20, min_periods=10).std() * np.sqrt(252)

        # Create the wide DataFrame WITHOUT dropping NaNs to preserve all dates
        if not volatility_data:
            print("Warning: No volatility data to process. Skipping spillovers.")
            return
        volatility_df = pd.DataFrame(volatility_data)

        spillover_features = {}

        for symbol in self.assets:
            if symbol not in volatility_df.columns:
                continue

            symbol_vol = volatility_df[symbol]
            other_assets = [s for s in self.assets if s != symbol and s in volatility_df.columns]

            if other_assets:
                # 1. Volatility shock transmission (lagged correlations)
                # Calculate the average volatility of all other assets, ignoring NaNs for each date
                other_vol_avg = volatility_df[other_assets].mean(axis=1, skipna=True)

                # Calculate correlations to lagged versions of the average other volatility
                # This is more efficient than looping
                vol_spillover_scores = []
                for lag in [1, 2, 3, 5]:
                    # .corr will correctly produce NaNs during the rolling window warm-up
                    lagged_corr = symbol_vol.rolling(60, min_periods=30).corr(other_vol_avg.shift(lag))
                    vol_spillover_scores.append(lagged_corr)

                # Average the correlation scores across lags, preserving NaNs
                avg_spillover = pd.concat(vol_spillover_scores, axis=1).mean(axis=1, skipna=True)
                spillover_features[f'{symbol}_volatility_spillover'] = avg_spillover

                # 2. Volatility leadership (no future data)
                # Correlation of this asset's past volatility vs. other assets' current volatility
                # We correlate the shifted symbol_vol against each of the other assets' vols
                leadership_corrs = []
                for other_asset in other_assets:
                    # corr between this asset's past vol and another's current vol
                    past_lead_corr = symbol_vol.shift(2).rolling(60, min_periods=30).corr(volatility_df[other_asset])
                    leadership_corrs.append(past_lead_corr)
                
                # Average the leadership scores
                if leadership_corrs:
                    vol_leadership = pd.concat(leadership_corrs, axis=1).mean(axis=1, skipna=True)
                    spillover_features[f'{symbol}_volatility_leadership'] = vol_leadership

                # 3. Volatility regime synchronization
                # How often is this asset in the same volatility regime as the average of others?
                symbol_high_vol = symbol_vol > symbol_vol.rolling(252, min_periods=60).quantile(0.7)
                other_high_vol = other_vol_avg > other_vol_avg.rolling(252, min_periods=60).quantile(0.7)

                # Boolean Series: True if in the same regime (both high or both low)
                vol_sync = (symbol_high_vol == other_high_vol)
                # Rolling mean gives the % of time they were synchronized in the window
                vol_sync_rolling = vol_sync.rolling(60, min_periods=30).mean()
                # Center the result around 0, ranging from -1 (perfectly out of sync) to 1 (perfectly in sync)
                vol_sync_centered = (vol_sync_rolling - 0.5) * 2

                spillover_features[f'{symbol}_volatility_synchronization'] = vol_sync_centered

        # Use optimized feature addition
        self.optimized_add_cross_asset_features(spillover_features)

        if self.verbose > 0:
            # Print statistics for debugging and clarification
            print("\nVolatility spillover features added:")
            for feature_name in spillover_features.keys():
                if feature_name in self.enriched_data.columns:
                    symbol = feature_name.split('_')[0]
                    feature_short_name = '_'.join(feature_name.split('_')[1:])
                    # Ensure we are looking at the correct column in the final dataframe
                    feature_data = self.enriched_data[self.enriched_data['Symbol'] == symbol][feature_short_name]
                    print(f"  {feature_name}: NaNs={feature_data.isna().sum()}, min={feature_data.min():.3f}, max={feature_data.max():.3f}, mean={feature_data.mean():.3f}")

    def calculate_market_leadership_metrics(self):
        """
        Calculate market leadership and beta dynamics using COMPOSITE MARKET as reference
        Uses average of all 7 assets as market proxy - this way ALL assets (including SPY) get meaningful metrics
        Dynamically uses the actual symbols present in self.enriched_data to avoid empty slices.
        """
        """
        Calculate market leadership and beta dynamics using a robust, NaN-preserving method.
        This version uses a wide-format DataFrame to handle date alignment automatically.
        """
        print("\nCalculating market leadership metrics (robustly)...")

        # 1. Create wide-format pivot tables for returns. DO NOT use intersection.
        pivot_1d = self.enriched_data.pivot_table(values='return_1d', index='Date', columns='Symbol')
        pivot_20d = self.enriched_data.pivot_table(values='return_20d', index='Date', columns='Symbol')

        # Check if we have enough data
        if pivot_1d.shape[1] < 2:
            print("Warning: Less than 2 assets with return data. Skipping market leadership.")
            return

        leadership_features = {}

        # 2. Loop through each asset to calculate its metrics against the others
        for symbol in self.assets:
            if symbol not in pivot_1d.columns:
                continue

            # Define the asset's own returns
            symbol_returns_1d = pivot_1d[symbol]
            symbol_returns_20d = pivot_20d[symbol]

            # Define the "market proxy" as the average of all OTHER assets
            other_assets = [s for s in self.assets if s != symbol and s in pivot_1d.columns]
            if not other_assets:
                continue
            
            # Pandas handles the alignment and skips NaNs automatically per row
            proxy_returns_1d = pivot_1d[other_assets].mean(axis=1, skipna=True)
            proxy_returns_20d = pivot_20d[other_assets].mean(axis=1, skipna=True)

            # 3. Calculate metrics using the aligned series
            # Rolling Beta
            covariance = symbol_returns_1d.rolling(60, min_periods=30).cov(proxy_returns_1d)
            market_variance = proxy_returns_1d.rolling(60, min_periods=30).var()
            rolling_beta = covariance / (market_variance + 1e-8)
            beta_normalized = (rolling_beta - 1.0).clip(-2, 2)

            # Beta Stability
            beta_stability = rolling_beta.rolling(60, min_periods=30).std()
            beta_stability_mean = beta_stability.rolling(252, min_periods=60).mean()
            beta_stability_std = beta_stability.rolling(252, min_periods=60).std()
            beta_stability_norm = ((beta_stability - beta_stability_mean) / (beta_stability_std + 1e-8)).clip(-3, 3)

            # Leadership Score
            lead_corr = symbol_returns_1d.rolling(60, min_periods=30).corr(proxy_returns_1d.shift(2))
            lag_corr = symbol_returns_1d.shift(2).rolling(60, min_periods=30).corr(proxy_returns_1d)
            leadership_score = (lead_corr - lag_corr).clip(-1, 1)

            # Alpha vs Market
            alpha_proxy = symbol_returns_20d - (rolling_beta * proxy_returns_20d)
            alpha_mean = alpha_proxy.rolling(252, min_periods=60).mean()
            alpha_std = alpha_proxy.rolling(252, min_periods=60).std()
            alpha_normalized = ((alpha_proxy - alpha_mean) / (alpha_std + 1e-8)).clip(-3, 3)

            # Store features - ensure no NaNs
            leadership_features[f'{symbol}_market_beta'] = beta_normalized
            leadership_features[f'{symbol}_beta_stability'] = beta_stability_norm
            leadership_features[f'{symbol}_market_leadership'] = leadership_score
            leadership_features[f'{symbol}_alpha_vs_market'] = alpha_normalized
                        
        # Use optimized feature addition
        self.optimized_add_cross_asset_features(leadership_features)
        
        if self.verbose > 0:
            # Print statistics for debugging and clarification
            print("\nMarket leadership features added (using composite market reference):")
            for feature_name in leadership_features.keys():
                symbol = feature_name.split('_')[0]
                feature_short_name = '_'.join(feature_name.split('_')[1:])
                feature_data = self.enriched_data[self.enriched_data['Symbol'] == symbol][feature_short_name]
                print(f"  {feature_name}: min={feature_data.min():.3f}, max={feature_data.max():.3f}, mean={feature_data.mean():.3f}, median={feature_data.median():.3f}")


    def calculate_cross_asset_correlations(self):
        """
        Calculate rolling correlations between assets in a robust, NaN-preserving way.
        Helps agent understand when diversification benefits break down (crisis periods).
        """
        print("\nCalculating cross-asset correlation features (robustly)...")

        # Create pivot table for cross-asset analysis WITHOUT dropping NaNs
        pivot_returns = self.enriched_data.pivot_table(
            values='return_20d', index='Date', columns='Symbol'
        )

        correlation_features = {}

        for symbol in self.assets:
            if symbol in pivot_returns.columns:
                symbol_returns = pivot_returns[symbol]
                other_assets = [s for s in self.assets if s != symbol and s in pivot_returns.columns]

                if other_assets:
                    # 1. Average correlation with other assets (diversification benefit)
                    # Vectorized approach: Calculate correlation with the average of other assets' returns.
                    # This is a robust and efficient proxy for the average pairwise correlation.
                    other_returns_avg = pivot_returns[other_assets].mean(axis=1, skipna=True)
                    
                    # Calculate the rolling correlation. NaNs will be present during the warm-up.
                    avg_correlation = symbol_returns.rolling(60, min_periods=30).corr(other_returns_avg)
                    correlation_features[f'{symbol}_avg_correlation'] = avg_correlation

                    # 2. Correlation regime (high = crisis, low = normal)
                    # This will also have NaNs during its own longer warm-up period.
                    corr_regime = (avg_correlation - avg_correlation.rolling(252, min_periods=60).mean()) / (avg_correlation.rolling(252, min_periods=60).std() + 1e-8)
                    correlation_features[f'{symbol}_correlation_regime'] = corr_regime.clip(-3, 3)

                    # 3. Correlation spike detection (crisis predictor)
                    # This will also have NaNs during its warm-up.
                    corr_spike = (avg_correlation > avg_correlation.rolling(252, min_periods=60).quantile(0.8)).astype(float)
                    # Replace False with 0, True with 1, and keep NaNs where they are
                    corr_spike.loc[corr_spike.notna()] = corr_spike.dropna().astype(int)
                    correlation_features[f'{symbol}_correlation_spike'] = corr_spike

        # Use optimized feature addition
        self.optimized_add_cross_asset_features(correlation_features)

        if self.verbose > 0:
            # Print statistics for debugging and clarification
            print("\nCross-asset correlation features added:")
            for feature_name in correlation_features.keys():
                if feature_name in self.enriched_data.columns:
                    symbol = feature_name.split('_')[0]
                    feature_short_name = '_'.join(feature_name.split('_')[1:])
                    feature_data = self.enriched_data[self.enriched_data['Symbol'] == symbol][feature_short_name]
                    print(f"  {feature_name}: NaNs={feature_data.isna().sum()}, min={feature_data.min():.3f}, max={feature_data.max():.3f}, mean={feature_data.mean():.3f}")

    def optimized_add_cross_asset_features(self, features_dict):
        """
        Correctly and robustly adds multiple cross-asset features to the main DataFrame.
        This version collects all features and merges them in a single, safe operation,
        preventing data corruption.

        Args:
            features_dict: Dictionary where keys are 'Symbol_feature_name' and 
                           values are pandas Series with a DatetimeIndex.
        """
        if not features_dict:
            return

        # 1. Convert the dictionary of Series into a single long-format DataFrame.
        all_features_list = []
        for feature_full_name, feature_series in features_dict.items():
            if feature_series is None or feature_series.empty:
                continue
            
            # Deconstruct the name 'SYMBOL_feature_name'
            parts = feature_full_name.split('_')
            symbol = parts[0]
            feature_short_name = '_'.join(parts[1:])
            
            # Create a temporary DataFrame for this feature
            temp_df = feature_series.reset_index()
            temp_df.columns = ['Date', 'value']
            temp_df['Symbol'] = symbol
            temp_df['feature_name'] = feature_short_name
            all_features_list.append(temp_df)

        if not all_features_list:
            print("Warning: No valid features to add.")
            return

        # 2. Combine all individual feature DataFrames into one.
        combined_features_df = pd.concat(all_features_list, ignore_index=True)

        # 3. Pivot the long-format DataFrame to a wide format.
        # This creates columns for each unique feature, ready for merging.
        features_to_merge = combined_features_df.pivot_table(
            index=['Date', 'Symbol'],
            columns='feature_name',
            values='value'
        ).reset_index()

        # 4. Perform a single, clean left merge.
        # This adds the new feature columns to the main DataFrame. If columns
        # with the same name already exist, pandas will correctly overwrite them
        # only for the matching Date-Symbol pairs, leaving other rows untouched.
        self.enriched_data = self.enriched_data.merge(
            features_to_merge,
            on=['Date', 'Symbol'],
            how='left'
        )

    def NaN_counting(self):
        # Detect NaN counts per column and overall within enriched_data
        nan_counts = self.enriched_data.isna().sum()
        if nan_counts.sum() > 0:
            # Enable terminal outputs up to 80 lines
            pd.set_option('display.max_rows', 80)
            print("\n" + "="*50)
            print("DETECTING NaN VALUES IN ENRICHED DATA")
            print(f"Total NaN values: {nan_counts.sum()}")
            # Print NaN counts per column
            print("\nNaN counts per column in descending order:")
            print(nan_counts[nan_counts > 0].sort_values(ascending=False))
            # Date range of ocurring NaNs
            nan_rows = self.enriched_data[self.enriched_data.isna().any(axis=1)].copy()
            if not nan_rows.empty:
                nan_dates = nan_rows['Date']
                print(f"\nRows with any NaN values occur from {nan_dates.min()} to {nan_dates.max()} (Total rows: {len(nan_rows)})")

            
            # Count NaNs per year
            nan_rows['Year'] = nan_rows['Date'].dt.year
            nan_counts_per_year = nan_rows.groupby('Year').size()
            print("\nNaN counts per year:")
            print(nan_counts_per_year)

            # # Wait for user input before continuing
            # print("Press any key to continue...")
            # try:
            #     msvcrt.getch()
            # except ImportError:
            #     input("Press Enter to continue...")

    def NaN_detection(self):
        # Detect NaN counts per column and overall within enriched_data
        nan_counts = self.enriched_data.isna().sum()
        if nan_counts.sum() > 0:
            # Enable terminal outputs up to 80 lines
            pd.set_option('display.max_rows', 80)
            print("\n" + "="*50)
            print("DETECTING NaN VALUES IN ENRICHED DATA")
            print(f"Total NaN values: {nan_counts.sum()}")
            # Print NaN counts per column
            # make printable stuff in terminal longer
            pd.set_option('display.max_columns', None)
            print("\nNaN counts per column in descending order:")
            print(nan_counts[nan_counts > 0].sort_values(ascending=False))
            # Date range of ocurring NaNs
            nan_rows = self.enriched_data[self.enriched_data.isna().any(axis=1)].copy()
            if not nan_rows.empty:
                nan_dates = nan_rows['Date']
                print(f"\nRows with any NaN values occur from {nan_dates.min()} to {nan_dates.max()} (Total rows: {len(nan_rows)})")
            
            # Count NaNs per year
            nan_rows['Year'] = nan_rows['Date'].dt.year
            nan_counts_per_year = nan_rows.groupby('Year').size()
            print("\nNaN counts per year:")
            print(nan_counts_per_year)

            # Count NaNs per symbol
            nan_counts_per_symbol = nan_rows.groupby('Symbol').size()
            print("\nNaN counts per symbol:")
            print(nan_counts_per_symbol)

            # Delete the date range within enriched_data that contains NaNs, if user confirms (y/n)
            user_input = input(f"\nDo you want to delete all {len(nan_rows)} rows with any NaN values from enriched_data? (y/n): ").strip().lower()
            if user_input == 'y':
                initial_shape = self.enriched_data.shape
                self.enriched_data = self.enriched_data.dropna() # dropna deletes all rows with any NaN values
                print(f"Rows with NaN values deleted. New shape: {self.enriched_data.shape} was {initial_shape}")
            else:
                print("No rows deleted. NaN values remain in enriched_data.")

    # Infs check
    def Inf_detection(self):
        """
        Detect Inf values only in numeric columns to avoid dtype coercion errors.
        Reports counts per column, year, and symbol.
        """
        # Select only numeric columns
        numeric_df = self.enriched_data.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("\nNo numeric columns available for Inf detection.")
            return

        # Boolean mask of +inf / -inf
        inf_bool_df = np.isinf(numeric_df)

        # Per-column counts (Series aligned to numeric columns)
        inf_counts = pd.Series(inf_bool_df.sum(axis=0), index=numeric_df.columns)

        total_infs = int(inf_counts.sum())
        if total_infs == 0:
            print("\nNo Inf values detected in enriched_data.")
            return

        pd.set_option('display.max_rows', 80)
        print("\n" + "="*50)
        print("DETECTING INF VALUES IN ENRICHED DATA")
        print(f"Total Inf values: {total_infs}")

        print("\nInf counts per column in descending order:")
        print(inf_counts[inf_counts > 0].sort_values(ascending=False))

        # Rows containing any Inf (use numeric subset row mask)
        row_inf_mask = inf_bool_df.any(axis=1)
        inf_row_indices = numeric_df.index[row_inf_mask]
        inf_rows = self.enriched_data.loc[inf_row_indices].copy()

        if not inf_rows.empty:
            inf_dates = inf_rows['Date']
            print(f"\nRows with any Inf values occur from {inf_dates.min()} to {inf_dates.max()} (Total rows: {len(inf_rows)})")

            # Year counts
            inf_rows['Year'] = inf_rows['Date'].dt.year
            inf_counts_per_year = inf_rows.groupby('Year').size()
            print("\nInf counts per year:")
            print(inf_counts_per_year)

            # Symbol counts
            if 'Symbol' in inf_rows.columns:
                inf_counts_per_symbol = inf_rows.groupby('Symbol').size()
                print("\nInf counts per symbol:")
                print(inf_counts_per_symbol)
        

    def save_calculated_data(self):
        """
        Save the enriched dataset to a CSV file.
        """
        # Sort the DataFrame by Date and then by Symbol to group daily data together.
        self.enriched_data = self.enriched_data.sort_values(by=['Date', 'Symbol'])

        output_filepath = os.path.join(self.output_dir, 'enriched_financial_data.csv')
        self.enriched_data.to_csv(output_filepath, index=False)
        print(f"\nEnriched data saved to {output_filepath}")


# Execution
data_process_and_enrichment = DataEnricher(
    input_raw_indices_filepath = r"C:\dev\pytrade\src\data\processed_multi_index_data.csv",
    input_raw_fx_filepath = None
)