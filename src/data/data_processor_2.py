"""
This is a data processing script for raw financial data.
It prepares the data by transforming, normalizing and enriching it
for the subsequent training and validation of machine learning models.

Output:
- A dataset ready for machine learning

Inputs:
- Raw financial data file (CSV format)
- Configuration parameters for data processing (e.g., date range)
"""

import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from rich.console import Console
from rich.table import Table

class DataEnricher:
    def __init__(self, input_raw_indices_filepath, input_raw_fx_filepath):
        self.input_raw_indices_filepath = input_raw_indices_filepath
        self.input_raw_fx_filepath = input_raw_fx_filepath

        # Save output in src/data
        self.output_dir = os.path.dirname(__file__)

        # Verbosity level for debugging
        self.verbose = 0

        # Set display options in terminal vs code and pandas for better debugging output
        pd.set_option('display.max_rows', 30)
        pd.set_option('display.max_columns', None)


        # read raw data files
        self.raw_indices_data = pd.read_csv(self.input_raw_indices_filepath)

        self.init_data_NaN_inf_check(self.raw_indices_data)

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

    def init_data_NaN_inf_check(self, df):
        # Initial NaN and Inf check on raw data
        print("\nInitial NaN and Inf check on raw indices data:")
        nan_counts = df.isna().sum()
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        print(f"NaN counts per column:\n{nan_counts[nan_counts > 0]}")
        print(f"Inf counts per numeric column:\n{inf_counts[inf_counts > 0]}")

        # Explicetly output a list of specific days when Nan or Infs occur in the raw data, if any
        if nan_counts.sum() > 0:
            nan_rows = df[df.isna().any(axis=1)]
            print(f"\nSpecific days with any NaN values:\n{nan_rows['Date'].unique()}")
        if inf_counts.sum() > 0:
            inf_rows = df[np.isinf(df.select_dtypes(include=[np.number])).any(axis=1)]
            print(f"\nSpecific days with any Inf values:\n{inf_rows['Date'].unique()}")


    def enrich_data(self):
        # Logic of feature calculation and data enrichment
        # Raw to stationary Feature engineering
        self.convert_raw_to_stationary_data()
        self.NaN_counting()

        self.trigonometric_date_encoding()
        self.NaN_counting()

        self.technical_indicators()
        self.NaN_counting()

        self.cross_asset_features()
        self.NaN_counting()

        self.add_frac_diff_features()
        self.NaN_counting()

        # COUNT NaNs per columns
        self.NaN_detection()
        # DETECT Infs
        self.Inf_detection()
        # 6. Save calculated data
        self.save_calculated_data()


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


            # Detailed breakdown by Year, Symbol, and Feature
            print("\nDETAILED NaN BREAKDOWN: Year × Symbol × Feature")
            
            # Get all numeric columns (features)
            numeric_cols = self.enriched_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Create a detailed breakdown
            detailed_nan_info = []
            for year in sorted(nan_rows['Year'].unique()):
                year_data = nan_rows[nan_rows['Year'] == year]
                for symbol in sorted(year_data['Symbol'].unique()):
                    symbol_year_data = year_data[year_data['Symbol'] == symbol]
                    
                    # Find which features have NaNs for this symbol-year combo
                    nan_features = []
                    for col in numeric_cols:
                        nan_count_in_col = symbol_year_data[col].isna().sum()
                        if nan_count_in_col > 0:
                            nan_features.append((col, nan_count_in_col))
                    
                    if nan_features:
                        # Sort by NaN count descending
                        nan_features.sort(key=lambda x: x[1], reverse=True)
                        num_rows_affected = len(symbol_year_data)
                        print(f"\n{year} | {symbol} ({num_rows_affected} rows with NaN):")
                        for feature, count in nan_features:
                            print(f"  └─ {feature}: {count} NaN values")
                        detailed_nan_info.append({
                            'Year': year,
                            'Symbol': symbol,
                            'RowsWithNaN': num_rows_affected,
                            'FeaturesWithNaN': [f[0] for f in nan_features],
                            'TopFeature': nan_features[0][0]
                        })
            
            # Summary table
            print("\n" + "="*50)
            print("SUMMARY TABLE: Year × Symbol")
            print("="*50)
            summary_df = pd.DataFrame(detailed_nan_info)
            if not summary_df.empty:
                for idx, row in summary_df.iterrows():
                    print(f"{row['Year']:4d} | {row['Symbol']:6s} | Rows: {row['RowsWithNaN']:4d} | Top feature: {row['TopFeature']}")

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

        print(f"Calculating technical indicators...")

        for symbol in technical_indicators['Symbol'].unique():
            symbol_data = technical_indicators[technical_indicators['Symbol'] == symbol].sort_values('Date').copy()
            
            # RSI - Relative strength index 14 and 30 days
            for period in [14]:
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


            # Bollinger Band width
            rolling_mean = symbol_data['Close'].rolling(window=self.max_norm_horizon).mean()
            rolling_std = symbol_data['Close'].rolling(window=self.max_norm_horizon).std()
            symbol_data['bb_width_reworked'] = ((rolling_mean + rolling_std * 2) - (rolling_mean - rolling_std * 2)) / (rolling_mean + 1e-8)  # Width as percentage of price
            
            # Apply Z-score normalization to Bollinger Band width and clip to avoid extreme values
            bb_width_mean = symbol_data['bb_width_reworked'].rolling(window=self.max_norm_horizon).mean()
            bb_width_std = symbol_data['bb_width_reworked'].rolling(window=self.max_norm_horizon).std()
            symbol_data['bb_width_reworked'] = ((symbol_data['bb_width_reworked'] - bb_width_mean) / (bb_width_std + 1e-8)).clip(-3, 3)

            # Add Bollinger Band width to correct symbol and date in enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'bb_width_reworked'
            ] = symbol_data['bb_width_reworked']


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

            # Add ATR to correct symbol and date in enriched_data
            self.enriched_data.loc[
                (self.enriched_data['Symbol'] == symbol) &
                (self.enriched_data['Date'].isin(symbol_data['Date'])),
                'atr_z_norm'
            ] = symbol_data['atr_z_norm']


    def cross_asset_features(self):
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
        for feature in ['z_fisher_corr_spy', 'z_market_dispersion','z_beta_spy']:
            self.enriched_data[feature] = df[feature]


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
            log_vol_mean = group_vol.transform(lambda x: x.rolling(window=self.max_norm_horizon, min_periods=5).mean())
            log_vol_std = group_vol.transform(lambda x: x.rolling(window=self.max_norm_horizon, min_periods=5).std())

            df[f'z_vol_of_vol_{period}d'] = (df[f'log_vol_{period}d'] - log_vol_mean) / (log_vol_std + 1e-8)
            df[f'z_vol_of_vol_{period}d'] = df[f'z_vol_of_vol_{period}d'].clip(-4, 4)

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
        spy_vol_intensity = np.log(spy_vol.clip(lower=1e-8) / (spy_vol_ma + 1e-8)).clip(lower=1e-8)

        for period in volume_periods:

            # A. Log Volume Intensity (Today vs Moving Average)
            # This is much more stable than pct_change for volume
            v_ma = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(window=self.max_norm_horizon).mean())
            df[f'log_volume_intensity_{period}d'] = np.log(df['Volume'].clip(lower=1e-8) / (v_ma + 1e-8))

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

        # Features for compact intraday information (Open, High, Low)
        log_intra_vol = np.log(df['High']/(df['Low']))
        closing_strength = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        log_overnight_gap = np.log(df['Open'] / df['Close'].shift(1))
        shadow_ratio = (df['High'] - np.max(df[['Open', 'Close']], axis=1)) / (df['High'] - df['Low'])

        # Make stationary (z-scoring or MinMax scaling)
        rolling_mean = log_intra_vol.rolling(window=self.max_norm_horizon).mean()
        rolling_std = log_intra_vol.rolling(window=self.max_norm_horizon).std()
        z_log_intra_vol = np.tanh((log_intra_vol - rolling_mean) / (rolling_std + 1e-8))

        rolling_mean_og = log_overnight_gap.rolling(window=self.max_norm_horizon).mean()
        rolling_std_og = log_overnight_gap.rolling(window=self.max_norm_horizon).std()
        z_log_overnight_gap = np.tanh((log_overnight_gap - rolling_mean_og) / (rolling_std_og + 1e-8))

        df['z_log_intra_vol'] = z_log_intra_vol
        df['norm_closing_strength'] = (closing_strength*2) - 1
        df['z_log_overnight_gap'] = z_log_overnight_gap
        df['norm_shadow_ratio'] = (shadow_ratio*2) - 1


        # --- 4. Final Assignment ---
        self.enriched_data = df

        if self.verbose > 0:
            print("\nEnriched data after converting to stationary format:")
            print(f"Shape of enriched data: {self.enriched_data.shape}")
            print(self.enriched_data.head(300))


    def _get_ffd_weights(self, d, threshold, max_len):
        """Calculates weights for FFD using binomial expansion."""
        w = [1.0]
        for k in range(1, max_len):
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            w.append(w_k)
        return np.array(w[::-1]).reshape(-1, 1)

    def _apply_ffd_to_series(self, series, d, threshold=5e-5):
        """Vectorized application of FFD weights to a single price series."""
        weights = self._get_ffd_weights(d, threshold, len(series))
        width = len(weights)
        # Use convolution for speed over looping
        # Resulting series is shorter by (width - 1)
        padded_series = series.values
        output = np.convolve(padded_series, weights.flatten(), mode='valid')
        
        # Return as series with aligned index
        return pd.Series(output, index=series.index[width-1:])

    def find_multi_asset_golden_d(self, df, start_d=0.4, end_d=0.8, step=0.05):
        """Iteratively finds the minimum d that achieves stationarity."""
        print(f"Searching for Golden d in range [{start_d}, {end_d}]...")

        # 1. Use SPY, Gold and Crude as the anchor to find d (avoids re-calculating for every ticker)
        spy_close = df[df['Symbol'] == 'SPY'].set_index('Date')['Close']
        gold_close = df[df['Symbol'] == 'Gold'].set_index('Date')['Close']
        crude_close = df[df['Symbol'] == 'Crude'].set_index('Date')['Close']

        found_ds = []

        for asset_name, close_series in [('SPY', spy_close), ('Gold', gold_close), ('Crude', crude_close)]:
            d = self.find_golden_d(close_series, start_d, end_d, step)
            print(f"  -> {asset_name} Golden d: {d:.2f}")
            found_ds.append(d)

        global_golden_d = max(found_ds)

        if global_golden_d == end_d:
            raise ValueError(f"Could not find a suitable d that achieves stationarity for any of the anchor assets within the range [{start_d}, {end_d}]. Consider expanding the search range or adjusting the step size.")
        
        return global_golden_d

    def find_golden_d(self, sample_series, start_d=0.4, end_d=0.8, step=0.05):
        """Iteratively finds the minimum d that achieves stationarity."""
        print(f"Searching for Golden d in range [{start_d}, {end_d}]...")
        golden_d = end_d
        
        for d in np.arange(start_d, end_d + step, step):
            # Apply FFD to log prices
            diff_series = self._apply_ffd_to_series(np.log(sample_series), d)
            
            # Run Augmented Dickey-Fuller test
            # Note: autolag='AIC' is standard for financial time series
            try:
                res = adfuller(diff_series.dropna(), autolag='AIC')
                p_val = res[1]
                if p_val < 0.05:
                    print(f"  -> d={d:.2f} passed ADF test (p={p_val:.4f}).")
                    golden_d = d
                    break
            except Exception:
                continue
                
        return golden_d

    def add_frac_diff_features(self):
        """Main entry point to calculate and add FFD features to enriched_data."""
        print("Calculating stationary memory features (FFD)...")
        df = self.enriched_data
            
        golden_d = self.find_multi_asset_golden_d(df)
        print(f"Applying Golden d={golden_d:.2f} to all assets.")

        # 2. Apply FFD to each symbol
        def group_ffd(group):
            # We work in Log space for price stationarity
            log_price = np.log(group['Close'])
            return self._apply_ffd_to_series(log_price, golden_d)

        # Calculate and map back
        ffd_series = df.groupby('Symbol', group_keys=False).apply(group_ffd)
        df['ffd_close'] = ffd_series
        
        # 3. Final Normalization (Z-Score + Tanh)
        # This ensures the 'memory' feature is scaled like our returns
        group_v = df.groupby('Symbol')['ffd_close']
        v_mean = group_v.transform(lambda x: x.rolling(window=self.max_norm_horizon).mean())
        v_std = group_v.transform(lambda x: x.rolling(window=self.max_norm_horizon).std())
        
        z_ffd = (df['ffd_close'] - v_mean) / (v_std + 1e-8)
        df['norm_ffd_feature'] = np.tanh(z_ffd)
        
        print(f"FFD feature generation complete. d={golden_d:.2f} \
              Verifying stationarity of all FFD features with ADF test...")
        
        passed_afd_test = self.report_stationarity()
        if not passed_afd_test:
            raise ValueError("Not all FFD features passed the stationarity test. Consider adjusting the Golden d search range or step size.")
        else:
            print("\nProceeding with adding calculated FFD features to enriched_data.")
            self.enriched_data = df

    def report_stationarity(self):
        """
        Generates a terminal report of ADF test results for the FFD features.
        """
        console = Console()
        table = Table(title="FFD Feature Stationarity Report")
        table.add_column("Symbol", style="cyan")
        table.add_column("ADF Statistic", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Status", justify="center")

        symbols = self.enriched_data['Symbol'].unique()
        feature_col = 'norm_ffd_feature'
        
        passed_count = 0

        for symbol in symbols:
            series = self.enriched_data[self.enriched_data['Symbol'] == symbol][feature_col].dropna()
            
            # We need enough data points for a meaningful ADF test
            if len(series) < 30:
                continue
                
            res = adfuller(series, autolag='AIC')
            adf_stat = res[0]
            p_val = res[1]
            
            # Determination of status
            if p_val < 0.01:
                status = "[green]Highly Stationary[/green]"
                passed_count += 1
            elif p_val < 0.05:
                status = "[yellow]Stationary, passed[/yellow]"
                passed_count += 1
            else:
                status = "[red]Non-Stationary[/red]"
                
            table.add_row(symbol, f"{adf_stat:.4f}", f"{p_val:.4f}", status)

        console.print(table)
        pass_rate = (passed_count / len(symbols)) * 100
        console.print(f"\n[bold]Overall Pass Rate (p < 0.05): {pass_rate:.2f}%[/bold]")

        if pass_rate == 100:
            return True
        else:
            print("\n[red]Warning: Not all Symbols' FFD features passed the stationarity test. Consider adjusting the Golden d search range or step size.[/red]")
            return False

        

# Execution
data_process_and_enrichment = DataEnricher(
    input_raw_indices_filepath = r"C:\dev\pytrade\src\data\processed_multi_index_data.csv",
    input_raw_fx_filepath = None
)