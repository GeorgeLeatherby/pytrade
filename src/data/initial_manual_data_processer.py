import pandas as pd
import glob
import os
import numpy as np
from scipy import stats

"""
This is a programm 
1. to combine multiple csv files into a single json file
2a. to forward fill missing data (2 days max)
2b. to add plausible volume information to the dataset if missing
3. to ensure the data is properly formatted and cleaned
4. ensure no NaNs are present in resulting raw dataset
"""

class Dataprep:
    """
    1. Combine multiple CSV files into a single multi-index dataframe
    2. Ensure proper formatting with multi-index: (Date, Symbol)
    3. NO forward filling, NO volume augmentation - keep raw data only
    """

    # There are several csv files per beginning_char containing different time ranges but overlapping
    def __init__(self):
        # Define the beginning characters for different asset types
        self.beginning_chars = ['Crude', 'EWG', 'EWH', 'EWJ', 'EWQ', 'EWS', 'EWT', 'EWU', 'EWY', 'SPY', 'Gold'] # list all symbols to use here

        # Get the directory of this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.used_data_dir = os.path.join(current_dir, 'used_data')

        # Dictionary to store dataframes for each asset
        self.asset_dataframes = {}

        # Init mein df
        self.multi_index_df = pd.DataFrame()

        self.load_combine_asset_data()
        self.create_multi_index_df()

        # Obsolete and incorrect!
        # self.add_time_features()
        self.forward_data_fill()
        self.fill_missing_volume()
        print(f"\n{self.multi_index_df.head(45)}")
        print(f"\n{self.multi_index_df.tail(45)}")
        # self.analyze_and_clean_data()
        self.save_multi_index_df()


    def load_combine_asset_data(self):
        # Process each beginning character (asset type) ---------------------------------------
        for beginning_char in self.beginning_chars:
            print(f"Processing {beginning_char} files...")
            
            # Find all CSV files that start with this beginning_char
            pattern = os.path.join(self.used_data_dir, f"{beginning_char}*.csv")
            matching_files = glob.glob(pattern)
            
            if not matching_files:
                print(f"Warning: No files found for {beginning_char}")
                continue
            
            # List to store dataframes for this asset
            symbol_dfs = []
            
            # Process each file for this asset
            for file_path in matching_files:
                print(f"Reading {os.path.basename(file_path)}")

                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Standardize column names (assuming common formats from financial data providers)
                # Common formats: Date, Price, Open, High, Low, Close, Volume
                # or: Date, Open, High, Low, Close, Volume
                column_mapping = {}
                
                for col in df.columns:
                    col_lower = col.lower().strip()
                    if 'date' in col_lower:
                        column_mapping[col] = 'Date'
                    elif 'open' in col_lower:
                        column_mapping[col] = 'Open'
                    elif 'high' in col_lower:
                        column_mapping[col] = 'High'
                    elif 'low' in col_lower:
                        column_mapping[col] = 'Low'
                    elif 'close' in col_lower or col_lower == 'price':
                        column_mapping[col] = 'Close'
                    elif 'vol' in col_lower:
                        column_mapping[col] = 'Volume'
                
                # Rename columns
                df = df.rename(columns=column_mapping)
                
                # Ensure we have required columns
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"  Warning: Missing columns in {file_path}: {missing_cols}")
                    break
                
                # Convert Date column to datetime
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # Remove rows with invalid dates
                df = df.dropna(subset=['Date'])

                # Convert Volume column
                corrected_volumes = self.convert_volume_string(df['Volume'])
                # Add corrected volumes to the dataframe
                df['Volume'] = corrected_volumes

                # Convert numeric columns to float
                numeric_cols = ['Open', 'High', 'Low', 'Close']

                for col in numeric_cols:
                    # Remove commas and convert to floats
                    df[col] = df[col].astype(str).str.replace(',', '').astype(float)

                # Select and reorder columns
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
                # Set Date as index
                df.set_index('Date', inplace=True) 
                
                # Sort by date
                df.sort_index(inplace=True)
                
                symbol_dfs.append(df)
                    
            
            # Combine all dataframes for this asset (handle overlapping data)
            if symbol_dfs:
                # Concatenate all dataframes for this asset
                combined_asset_df = pd.concat(symbol_dfs)
                
                # Remove duplicates, keeping the last occurrence (most recent data)
                combined_asset_df = combined_asset_df[~combined_asset_df.index.duplicated(keep='last')]
                
                # Sort by date
                combined_asset_df.sort_index(inplace=True)
                
                # Store in dictionary
                self.asset_dataframes[beginning_char] = combined_asset_df
                
                print(f"  Combined {beginning_char}: {len(combined_asset_df)} rows from {combined_asset_df.index.min()} to {combined_asset_df.index.max()}")

    def convert_volume_string(self, volume_series):
        """
        Convert volume series with suffixes (M, B, K) to numeric values.
        
        Args:
            volume_series: Pandas Series with strings like "64.40M", "6B", "1.5K"
        
        Returns:
            Pandas Series with numeric values or None if conversion fails
        """
        def convert_single_volume(volume_str):
            """Helper function to convert a single volume string."""
            if pd.isna(volume_str) or volume_str == '':
                return None
            
            # Convert to string and clean
            volume_str = str(volume_str).strip().upper()
            
            # Remove commas
            volume_str = volume_str.replace(',', '')
            
            # Handle different suffixes
            multipliers = {
                'K': 1_000,
                'M': 1_000_000,
                'B': 1_000_000_000,
                'T': 1_000_000_000_000
            }
            
            # Check if it ends with a multiplier
            for suffix, multiplier in multipliers.items():
                if volume_str.endswith(suffix):
                    try:
                        # Extract the numeric part
                        numeric_part = volume_str[:-1]
                        value = float(numeric_part) * multiplier
                        return value
                    except ValueError:
                        return None
            
            # If no suffix, try direct conversion
            try:
                return float(volume_str)
            except ValueError:
                return None
        
        # Apply the conversion function to each element in the series
        return volume_series.apply(convert_single_volume)

    def create_multi_index_df(self): 
        # Create multi-index dataframe
        if not self.asset_dataframes:
            print("Error: No valid data found!")

        else:
            print("\nCreating multi-index dataframe...")

            # Create list to store all data with asset identifier
            all_data = []

            for asset, df in self.asset_dataframes.items():
                # Add asset column
                df_copy = df.copy()
                df_copy['Symbol'] = asset
                
                # Reset index to make Date a column
                df_copy = df_copy.reset_index()
                
                all_data.append(df_copy)

            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)

            # Create multi-index: first level = Date, second level = Symbol
            multi_index_df = combined_df.set_index(['Date', 'Symbol'])

            # Sort the multi-index
            self.multi_index_df = multi_index_df.sort_index()

            print(f"\nFinal multi-index dataframe created:")
            print(f"Shape: {self.multi_index_df.shape}")
            print(f"Date range: {self.multi_index_df.index.get_level_values('Date').min()} to {self.multi_index_df.index.get_level_values('Date').max()}")
            print(f"Symbols: {self.multi_index_df.index.get_level_values('Symbol').unique().tolist()}")
            print(f"Columns: {self.multi_index_df.columns.tolist()}")
            pd.set_option('display.max_rows', 100)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            print("\nFirst 45 entries of multi-index dataframe:")
            print(self.multi_index_df.head(45))
            print("\nLast 45 entries of multi-index dataframe:")
            print(self.multi_index_df.tail(45))

    


    def add_time_features(self):
        """
        Add cyclical time features aligned to trading days only.
        Creates three columns:
        - dow_sin/dow_cos: cyclical day-of-week over 5 trading days (Mon..Fri)
        - dom_sin/dom_cos: cyclical day-of-month over the number of trading days in that month (excl. weekends/closures)
        - moy_sin/moy_cos: cyclical month-of-year over 12 months
        Weekends are fully excluded; no synthetic rows introduced.
        """
        print("\nAdding time-based features...")

        # Extract date level
        date_level_full = self.multi_index_df.index.get_level_values('Date')

        # Remove weekends entirely (Sat=5, Sun=6) using level-aware mask
        weekend_mask = date_level_full.dayofweek >= 5
        if weekend_mask.any():
            deleted_data = self.multi_index_df[weekend_mask]
            self.multi_index_df = self.multi_index_df[~weekend_mask]
            date_level_full = self.multi_index_df.index.get_level_values('Date')
            print(f"Removed {int(weekend_mask.sum())} weekend rows; remaining unique dates: {len(date_level_full.unique())}")

        # Work on unique trading dates to avoid per-asset duplication issues
        unique_dates = pd.Index(date_level_full.unique()).sort_values()

        two_pi = 2 * np.pi
        phase_shift_shared = 0.15 * two_pi  # phase shift for all features

        # Build a per-date feature frame
        df_unique_dates = pd.DataFrame({'Date': unique_dates})
        df_unique_dates['YearMonth'] = df_unique_dates['Date'].dt.to_period('M')

        # Day-of-week (Mon..Fri -> 0..4)
        dow = df_unique_dates['Date'].dt.dayofweek.to_numpy()
        theta_dow = two_pi * dow / 5.0
        df_unique_dates['dow_sin'] = np.round(np.sin(theta_dow + phase_shift_shared), 6)  # keep existing name as sin(2π*k/5)
        df_unique_dates['dow_cos'] = np.round(np.cos(theta_dow + phase_shift_shared), 6)

        # Month-of-year (Jan..Dec -> 0..11)
        moy = (df_unique_dates['Date'].dt.month - 1).to_numpy()
        theta_moy = two_pi * moy / 12.0
        df_unique_dates['moy_sin'] = np.round(np.sin(theta_moy + phase_shift_shared), 6)  # keep existing name as sin(2π*k/12)
        df_unique_dates['moy_cos'] = np.round(np.cos(theta_moy + phase_shift_shared), 6)

        # Trading day-of-month: rank within trading days of each month (0..n-1) and n varies each month
        df_unique_dates['dom_rank'] = df_unique_dates.groupby('YearMonth').cumcount()
        df_unique_dates['dom_size'] = df_unique_dates.groupby('YearMonth')['Date'].transform('size')
        # Avoid division by zero; dom_size >= 1 by construction
        theta_dom = two_pi * df_unique_dates['dom_rank'].to_numpy() / df_unique_dates['dom_size'].to_numpy()
        # Provide both sin and cos; keep existing dom_cyc name as sin for consistency
        df_unique_dates['dom_sin'] = np.round(np.sin(theta_dom), 6)
        df_unique_dates['dom_cos'] = np.round(np.cos(theta_dom), 6)

        # Index by Date for a clean join on MultiIndex Date level
        df_unique_dates = df_unique_dates.set_index('Date')

        # Join per-date features onto the full MultiIndex by Date level
        self.multi_index_df = self.multi_index_df.join(
            df_unique_dates[['dow_sin', 'dow_cos', 'dom_sin', 'dom_cos', 'moy_sin', 'moy_cos']],
            on='Date'
        )

        print("Time-based features added: dow_sin/dow_cos, dom_sin/dom_cos, moy_sin/moy_cos")


    def forward_data_fill(self):
        """
        This function applies a forward data fill in the multi-index df
        It fills missing values by propagating the last valid observation forward
        It fills values up to 2 days. It does not overwrite existing values
        It does so for each symbol in each column.

        IMPORTANT: DOES NOT introduce new rows for missing dates.
        IMPORTANT: After forward fill, it deletes any days that do not have all symbols with data.
        """
        print("\nApplying forward data fill...")
        
        # Store original stats
        original_shape = self.multi_index_df.shape
        original_na_count = self.multi_index_df.isna().sum().sum()
        
        # Get all unique symbols and complete date range
        symbols = self.multi_index_df.index.get_level_values('Symbol').unique()
        all_available_dates = self.multi_index_df.index.get_level_values('Date').unique()
        # date_range = pd.date_range(start=all_available_dates.min(), end=all_available_dates.max(), freq='D')
        
        print(f"Processing {len(symbols)} symbols across {len(all_available_dates)} dates...")
        
        # Process each symbol separately and store results
        filled_data_list = []
        
        for symbol in symbols:
            print(f"  Processing symbol: {symbol}")
            
            # Extract symbol data
            symbol_data = self.multi_index_df.loc[(slice(None), symbol), :].copy()
            symbol_data = symbol_data.reset_index(level='Symbol', drop=True)
            
            # Reindex to available dates 
            symbol_data = symbol_data.reindex(all_available_dates)

            # Count NaN before forward fill
            na_before = symbol_data.isna().sum().sum()

            #symbol_data = symbol_data.ffill(limit=0) # Forward fill with limit of 2 days. 
            
            # Count NaN after forward fill
            na_after = symbol_data.isna().sum().sum()
            filled_count = na_before - na_after
            
            # Prepare for concatenation
            symbol_data['Symbol'] = symbol
            symbol_data = symbol_data.reset_index()
            symbol_data = symbol_data.rename(columns={'index': 'Date'})
            
            filled_data_list.append(symbol_data)
            print(f"    Filled {filled_count} missing values for {symbol}")
        
        # Step 1: Combine all data into single dataframe (creates days with missing symbols)
        print("\nCombining all symbol data...")
        combined_data = pd.concat(filled_data_list, ignore_index=True)
        temp_df = combined_data.set_index(['Date', 'Symbol']).sort_index()
        
        print(f"Combined dataframe shape: {temp_df.shape}")
        
        # Step 2: Delete days that don't have all symbols with data
        print(f"\nFiltering to keep only days with all {len(symbols)} symbols having data...")
        
        # Group by date and count how many symbols have non-null data for that date
        date_symbol_counts = temp_df.groupby('Date').apply(
            lambda x: x.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], how='all').shape[0]
        )
        
        # Keep only dates where all symbols have data
        complete_dates = date_symbol_counts[date_symbol_counts == len(symbols)].index
        deleted_days = len(date_symbol_counts) - len(complete_dates)
        
        print(f"Days deleted (incomplete symbol coverage): {deleted_days}")
        print(f"Days remaining: {len(complete_dates)}")
        
        # Filter dataframe to keep only complete days
        self.multi_index_df = temp_df.loc[complete_dates]
        
        # Final statistics
        final_shape = self.multi_index_df.shape
        final_na_count = self.multi_index_df.isna().sum().sum()
        
        print(f"\nForward fill completed:")
        print(f"  Original shape: {original_shape}")
        print(f"  Final shape: {final_shape}")
        print(f"  Original NaN count: {original_na_count}")
        print(f"  Final NaN count: {final_na_count}")
        print(f"  Net values filled: {original_na_count - final_na_count}")
        
        print(f"\nFirst 21 entries:")
        print(self.multi_index_df.head(21))
        print(f"\nLast 21 entries:")
        print(self.multi_index_df.tail(21))

    def fill_missing_volume(self):
        """
        Fill missing volume information using multiple smart methods combined.
        """
        print("\nFilling missing volume data...")
        
        original_volume_missing = self.multi_index_df['Volume'].isna().sum()
        print(f"Original missing volume entries: {original_volume_missing}")
        
        symbols = self.multi_index_df.index.get_level_values('Symbol').unique()
        
        # Track all filled indices for plausibility check
        all_filled_indices = []
        
        for symbol in symbols:
            print(f"\n  Processing volume for {symbol}...")
            
            # Extract symbol data
            symbol_data = self.multi_index_df.loc[(slice(None), symbol), :].copy()
            symbol_data = symbol_data.reset_index(level='Symbol', drop=True)
            
            missing_count = symbol_data['Volume'].isna().sum()
            if missing_count == 0:
                print(f"    No missing volume data for {symbol}")
                continue
            elif missing_count > 0.8 * symbol_data.shape[0] and symbol == "Nikkei":
                print(f"    High missing volume data for {symbol}: {missing_count}")
                # Generate Nikkei volume data and store the result
                generated_volumes = self.generate_nikkei_volume_data(symbol_data)
                
                # Update the symbol_data with generated volumes
                symbol_data['Volume'] = generated_volumes
                
                # Update the main dataframe with the generated data
                self.multi_index_df.loc[(slice(None), symbol), 'Volume'] = symbol_data['Volume'].values
                
                # Calculate how many volumes were generated
                final_missing = symbol_data['Volume'].isna().sum()
                filled_this_symbol = missing_count - final_missing
                print(f"    Generated {filled_this_symbol} volume entries for {symbol}")
                
                # Track filled indices for plausibility check
                original_missing_indices = symbol_data[symbol_data['Volume'].notna()].index
                filled_indices = [(date, symbol) for date in original_missing_indices]
                all_filled_indices.extend(filled_indices)
            else:
                print(f"    Missing volume entries for {symbol}: {missing_count}")
                
                # Create dictionary to store predictions from each method
                volume_predictions = {}
                missing_mask = symbol_data['Volume'].isna()
                
                # Store original missing indices for this symbol
                original_missing_indices = symbol_data[missing_mask].index
                
                # Method 1: Volume-Price Relationship
                pred1 = self._fill_volume_price_relationship(symbol_data.copy())
                volume_predictions['price_relationship'] = pred1['Volume'][missing_mask]
                
                # Method 2: Volatility-Volume Correlation
                pred2 = self._fill_volume_volatility_relationship(symbol_data.copy())
                volume_predictions['volatility_relationship'] = pred2['Volume'][missing_mask]
                
                # Method 3: Market Regime Adjustment
                pred3 = self._adjust_volume_market_regime(symbol_data.copy())
                volume_predictions['market_regime'] = pred3['Volume'][missing_mask]
                
                # # Method 4: Cross-Asset Volume Correlation
                # pred4 = self._fill_volume_cross_asset(symbol_data.copy(), symbol)
                # volume_predictions['cross_asset'] = pred4['Volume'][missing_mask]
                
                # Method 5: Temporal Pattern Recognition
                pred5 = self._fill_volume_temporal_patterns(symbol_data.copy())
                volume_predictions['temporal_patterns'] = pred5['Volume'][missing_mask]
                
                # Combine all predictions using weighted average
                combined_volume = self._combine_volume_predictions(volume_predictions, missing_mask)
                
                # Update the symbol data with combined predictions
                symbol_data.loc[missing_mask, 'Volume'] = combined_volume
                
                # Update the main dataframe
                self.multi_index_df.loc[(slice(None), symbol), 'Volume'] = symbol_data['Volume'].values
                
                # Track filled indices for this symbol (convert to multi-index format)
                filled_indices = [(date, symbol) for date in original_missing_indices if pd.notna(combined_volume.get(date))]
                all_filled_indices.extend(filled_indices)
                
                final_missing = symbol_data['Volume'].isna().sum()
                filled_this_symbol = missing_count - final_missing
                print(f"    Filled {filled_this_symbol} volume entries for {symbol}")
    
        final_volume_missing = self.multi_index_df['Volume'].isna().sum()
        total_filled = original_volume_missing - final_volume_missing

        # Plausibility check for added values using proper multi-index
        if all_filled_indices:
            try:
                # Create proper multi-index for filled values
                filled_multi_index = pd.MultiIndex.from_tuples(all_filled_indices, names=['Date', 'Symbol'])
                added_values = self.multi_index_df.loc[filled_multi_index, 'Volume']
                
                if not added_values.empty:
                    abs_range = added_values.max() - added_values.min()
                    mean_value = added_values.mean()
                    median_value = added_values.median()
                    std_value = added_values.std()
                    
                    print(f"  Plausibility check for added values:")
                    print(f"    - Count: {len(added_values)}")
                    print(f"    - Range: {added_values.min():.0f} to {added_values.max():.0f}")
                    print(f"    - Mean: {mean_value:.0f}")
                    print(f"    - Median: {median_value:.0f}")
                    print(f"    - Std Dev: {std_value:.0f}")
            except Exception as e:
                print(f"  Plausibility check skipped due to indexing issue: {e}")

        print(f"\nVolume filling completed:")
        print(f"  Total volume entries filled: {total_filled}")
        print(f"  Remaining missing volume entries: {final_volume_missing}")

    def _fill_volume_price_relationship(self, symbol_data):
        """
        Method 1: Use Price-Volume relationship
        Higher price changes typically correlate with higher volume
        """
        data = symbol_data.copy()
        
        # Calculate price metrics
        data['Price_Change'] = data['Close'].pct_change().abs()
        data['Price_Range'] = (data['High'] - data['Low']) / data['Close']
        data['True_Range'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            )
        )
        data['True_Range_Pct'] = data['True_Range'] / data['Close']
        
        # Create volume mask for non-missing data
        volume_mask = data['Volume'].notna()
        
        if volume_mask.sum() < 10:  # Need at least 10 data points
            return data
        
        # Fit relationship between price metrics and volume
        X = data.loc[volume_mask, ['Price_Change', 'Price_Range', 'True_Range_Pct']].fillna(0)
        y = np.log(data.loc[volume_mask, 'Volume'] + 1)  # Log transform to handle skewness
        
        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict missing volumes
            missing_mask = data['Volume'].isna()
            if missing_mask.sum() > 0:
                X_missing = data.loc[missing_mask, ['Price_Change', 'Price_Range', 'True_Range_Pct']].fillna(0)
                predicted_log_volume = model.predict(X_missing)
                predicted_volume = np.exp(predicted_log_volume) - 1
                
                data.loc[missing_mask, 'Volume'] = predicted_volume
                
        except ImportError:
            # Fallback: Simple correlation-based approach
            if volume_mask.sum() > 5:
                # Use simple correlation with true range
                valid_data = data[volume_mask]
                correlation = valid_data['True_Range_Pct'].corr(valid_data['Volume'])
                
                if abs(correlation) > 0.1:  # Meaningful correlation
                    # Scale volume based on true range
                    median_volume = valid_data['Volume'].median()
                    median_tr = valid_data['True_Range_Pct'].median()
                    
                    missing_mask = data['Volume'].isna()
                    scaling_factor = data.loc[missing_mask, 'True_Range_Pct'] / median_tr
                    data.loc[missing_mask, 'Volume'] = median_volume * scaling_factor
        
        return data

    def _fill_volume_volatility_relationship(self, data):
        """
        Method 2: Volume-Volatility Relationship
        Higher volatility periods typically have higher volume
        """
        # Calculate rolling volatility (10-day)
        data['Rolling_Volatility'] = data['Close'].pct_change().rolling(window=10).std()
        
        volume_mask = data['Volume'].notna()
        missing_mask = data['Volume'].isna()
        
        if volume_mask.sum() < 10 or missing_mask.sum() == 0:
            return data
        
        # Calculate volatility-volume relationship
        valid_data = data[volume_mask].dropna(subset=['Rolling_Volatility'])
        
        if len(valid_data) > 5:
            # Fit power law relationship: Volume ∝ Volatility^α
            log_vol = np.log(valid_data['Rolling_Volatility'] + 1e-8)
            log_volume = np.log(valid_data['Volume'] + 1)
            
            # Simple linear regression in log space
            slope, intercept, r_value, _, _ = stats.linregress(log_vol, log_volume)
            
            if abs(r_value) > 0.2:  # Meaningful relationship
                # Fill missing values where volatility is available
                fillable_mask = missing_mask & data['Rolling_Volatility'].notna()
                if fillable_mask.sum() > 0:
                    log_vol_missing = np.log(data.loc[fillable_mask, 'Rolling_Volatility'] + 1e-8)
                    predicted_log_volume = slope * log_vol_missing + intercept
                    predicted_volume = np.exp(predicted_log_volume) - 1
                    
                    # Blend with existing predictions (if any)
                    existing_predictions = data.loc[fillable_mask, 'Volume'].notna()
                    if existing_predictions.any():
                        # Weighted average: 70% new prediction, 30% existing
                        data.loc[fillable_mask, 'Volume'] = np.where(
                            existing_predictions,
                            0.3 * data.loc[fillable_mask, 'Volume'] + 0.7 * predicted_volume,
                            predicted_volume
                        )
                    else:
                        data.loc[fillable_mask, 'Volume'] = predicted_volume
        
        return data

    def _adjust_volume_market_regime(self, data):
        """
        Method 3: Adjust volume based on market regime
        Bull/Bear markets and high/low volatility regimes have different volume patterns
        """
        # Calculate market regime indicators
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['Market_Trend'] = np.where(data['MA_20'] > data['MA_50'], 1, -1)  # 1=Bull, -1=Bear
        
        # VIX-like indicator (rolling volatility)
        data['VIX_Proxy'] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        data['High_Vol_Regime'] = data['VIX_Proxy'] > data['VIX_Proxy'].rolling(window=100).median()
        
        volume_mask = data['Volume'].notna()
        missing_mask = data['Volume'].isna()
        
        if volume_mask.sum() < 50 or missing_mask.sum() == 0:
            return data
        
        # Calculate volume statistics by regime
        regimes = {}
        for trend in [1, -1]:
            for high_vol in [True, False]:
                regime_mask = (data['Market_Trend'] == trend) & (data['High_Vol_Regime'] == high_vol) & volume_mask
                if regime_mask.sum() > 5:
                    regimes[(trend, high_vol)] = {
                        'median_volume': data.loc[regime_mask, 'Volume'].median(),
                        'volume_std': data.loc[regime_mask, 'Volume'].std()
                    }
        
        # Apply regime-based adjustments
        if regimes:
            for idx in data[missing_mask].index:
                if pd.notna(data.loc[idx, 'Market_Trend']) and pd.notna(data.loc[idx, 'High_Vol_Regime']):
                    trend = data.loc[idx, 'Market_Trend']
                    high_vol = data.loc[idx, 'High_Vol_Regime']
                    
                    if (trend, high_vol) in regimes and pd.notna(data.loc[idx, 'Volume']):
                        # Adjust existing prediction based on regime
                        regime_factor = regimes[(trend, high_vol)]['median_volume'] / np.mean([r['median_volume'] for r in regimes.values()])
                        data.loc[idx, 'Volume'] *= regime_factor
                    elif (trend, high_vol) in regimes and pd.isna(data.loc[idx, 'Volume']):
                        # Use regime median if no prediction exists
                        data.loc[idx, 'Volume'] = regimes[(trend, high_vol)]['median_volume']
        
        return data

    def _fill_volume_cross_asset(self, data, current_symbol):
        """
        Method 4: Use cross-asset volume correlation
        Some assets (like regional indices) may have correlated volume patterns
        """
        # Define asset relationships
        correlations = {
            'DAX': ['CAC', 'FTSE'],  # European indices
            'CAC': ['DAX', 'FTSE'],
            'FTSE': ['DAX', 'CAC'],
            'SPY': ['DAX', 'FTSE'],  # Global market leader correlations
            'Nikkei': ['Hang'],      # Asian indices
            'Hang': ['Nikkei'],
            'Gold': ['SPY']          # Gold often inverse to equities
        }
        
        related_symbols = correlations.get(current_symbol, [])
        missing_mask = data['Volume'].isna()
        
        if not related_symbols or missing_mask.sum() == 0:
            return data
        
        # Get volume data for related assets
        for related_symbol in related_symbols:
            try:
                related_data = self.multi_index_df.loc[(slice(None), related_symbol), 'Volume']
                related_data = related_data.reset_index(level='Symbol', drop=True)
                
                # Align dates
                aligned_data = pd.concat([data['Volume'], related_data], axis=1, keys=['current', 'related'])
                
                # Calculate correlation on overlapping non-missing data
                overlap_mask = aligned_data['current'].notna() & aligned_data['related'].notna()
                if overlap_mask.sum() > 20:  # Need sufficient overlap
                    correlation = aligned_data.loc[overlap_mask, 'current'].corr(aligned_data.loc[overlap_mask, 'related'])
                    
                    if abs(correlation) > 0.3:  # Meaningful correlation
                        # Use related asset volume to estimate missing values
                        missing_with_related = missing_mask & aligned_data['related'].notna()
                        
                        if missing_with_related.sum() > 0:
                            # Simple scaling based on median ratio
                            median_ratio = (aligned_data.loc[overlap_mask, 'current'] / 
                                        aligned_data.loc[overlap_mask, 'related']).median()
                            
                            estimated_volume = aligned_data.loc[missing_with_related, 'related'] * median_ratio
                            
                            # Blend with existing predictions
                            existing_predictions = data.loc[missing_with_related, 'Volume'].notna()
                            if existing_predictions.any():
                                data.loc[missing_with_related, 'Volume'] = np.where(
                                    existing_predictions,
                                    0.5 * data.loc[missing_with_related, 'Volume'] + 0.5 * estimated_volume,
                                    estimated_volume
                                )
                            else:
                                data.loc[missing_with_related, 'Volume'] = estimated_volume
                                
            except KeyError:
                continue  # Related symbol not available
        
        return data

    def _fill_volume_temporal_patterns(self, data):
        """
        Method 5: Use temporal patterns (day of week, month effects, etc.)
        """
        data_copy = data.copy()
        data_copy['DayOfWeek'] = data_copy.index.dayofweek
        data_copy['Month'] = data_copy.index.month
        data_copy['IsMonthEnd'] = (data_copy.index + pd.Timedelta(days=1)).month != data_copy.index.month
        
        volume_mask = data_copy['Volume'].notna()
        missing_mask = data_copy['Volume'].isna()
        
        if volume_mask.sum() < 50 or missing_mask.sum() == 0:
            return data
        
        # Calculate temporal volume patterns
        patterns = {}
        
        # Day of week effects
        for dow in range(5):  # Monday=0 to Friday=4
            dow_mask = (data_copy['DayOfWeek'] == dow) & volume_mask
            if dow_mask.sum() > 5:
                patterns[f'dow_{dow}'] = data_copy.loc[dow_mask, 'Volume'].median()
        
        # Month effects
        for month in range(1, 13):
            month_mask = (data_copy['Month'] == month) & volume_mask
            if month_mask.sum() > 5:
                patterns[f'month_{month}'] = data_copy.loc[month_mask, 'Volume'].median()
        
        # Month-end effects
        month_end_mask = data_copy['IsMonthEnd'] & volume_mask
        if month_end_mask.sum() > 5:
            patterns['month_end'] = data_copy.loc[month_end_mask, 'Volume'].median()
        
        # Apply temporal patterns to fill remaining missing values
        overall_median = data_copy.loc[volume_mask, 'Volume'].median()
        
        for idx in data_copy[missing_mask].index:
            if pd.isna(data.loc[idx, 'Volume']):  # Only fill if still missing
                # Start with overall median
                estimated_volume = overall_median
                
                # Apply day of week factor
                dow = data_copy.loc[idx, 'DayOfWeek']
                if f'dow_{dow}' in patterns:
                    dow_factor = patterns[f'dow_{dow}'] / overall_median
                    estimated_volume *= dow_factor
                
                # Apply month factor
                month = data_copy.loc[idx, 'Month']
                if f'month_{month}' in patterns:
                    month_factor = patterns[f'month_{month}'] / overall_median
                    estimated_volume *= month_factor
                
                # Apply month-end factor
                if data_copy.loc[idx, 'IsMonthEnd'] and 'month_end' in patterns:
                    month_end_factor = patterns['month_end'] / overall_median
                    estimated_volume *= month_end_factor
                
                data.loc[idx, 'Volume'] = estimated_volume
        
        return data

    def _combine_volume_predictions(self, volume_predictions, missing_mask):
        """
        Combine predictions from multiple methods using weighted averaging.
        """
        import numpy as np
        
        # Define weights for each method (based on reliability)
        method_weights = {
            'price_relationship': 0.25,      # Most reliable - proven finance relationship
            'volatility_relationship': 0.20,  # Good secondary indicator
            'market_regime': 0.15,           # Useful context
            'cross_asset': 0.25,             # Strong for correlated assets
            'temporal_patterns': 0.15        # Baseline patterns
        }
        
        # Create array to store weighted predictions
        weighted_predictions = []
        total_weights = []
        
        for idx in missing_mask[missing_mask].index:
            predictions_for_date = []
            weights_for_date = []
            method_names = []
            
            for method, predictions in volume_predictions.items():
                if idx in predictions.index and pd.notna(predictions.loc[idx]):
                    predictions_for_date.append(predictions.loc[idx])
                    weights_for_date.append(method_weights[method])
                    method_names.append(method)
            
            if predictions_for_date:
                # Check if predictions differ wildly before combining
                if len(predictions_for_date) > 1:
                    predictions_array = np.array(predictions_for_date)
                    min_pred = predictions_array.min()
                    max_pred = predictions_array.max()
                    
                    # Consider "wildly different" if:
                    # 1. Ratio between max and min > 10x, OR
                    # 2. Standard deviation > 2x the mean
                    ratio_threshold = 10.0
                    cv_threshold = 2.0  # Coefficient of variation threshold
                    
                    ratio_diff = max_pred / min_pred if min_pred > 0 else float('inf')
                    mean_pred = predictions_array.mean()
                    std_pred = predictions_array.std()
                    cv = std_pred / mean_pred if mean_pred > 0 else float('inf')
                    
                    if ratio_diff > ratio_threshold or cv > cv_threshold:
                        print(f"\n    ⚠️  WILDLY DIFFERENT PREDICTIONS for date {idx}:")
                        print(f"       Ratio (max/min): {ratio_diff:.2f}x")
                        print(f"       Coefficient of Variation: {cv:.2f}")
                        print(f"       Individual predictions:")
                        for method, pred in zip(method_names, predictions_for_date):
                            print(f"         - {method:<25}: {pred:>15,.0f}")
                        print(f"       Mean: {mean_pred:,.0f}, Std Dev: {std_pred:,.0f}")
                
                # Weighted average of available predictions
                weighted_avg = np.average(predictions_for_date, weights=weights_for_date)
                weighted_predictions.append(weighted_avg)
            else:
                # Fallback to overall median if no predictions available
                weighted_predictions.append(np.nan)
            
            total_weights.append(sum(weights_for_date))
        
        # Create series with proper index
        result = pd.Series(weighted_predictions, index=missing_mask[missing_mask].index)
        
        # For dates with low total weight (unreliable predictions), use fallback
        low_confidence_mask = pd.Series(total_weights, index=missing_mask[missing_mask].index) < 0.3
        if low_confidence_mask.any():
            # Use simple forward fill or median as fallback
            fallback_values = self.multi_index_df.loc[(slice(None), 'SPY'), 'Volume'].median()  # Use SPY as reference
            result.loc[low_confidence_mask] = fallback_values
            
            print(f"    📊 Using fallback values for {low_confidence_mask.sum()} dates with low confidence predictions")
        
        return result

    def generate_nikkei_volume_data(self, nikkei_data):
        """
        Generate realistic daily Nikkei volume data in JPY using multiple factors:
        - Historical volume trends and regimes
        - Inverse gold correlation
        - Monthly seasonality
        - Crisis detection
        - Cross-market correlations
        - Daily price action indicators
        """
        print("\nGenerating realistic Nikkei volume data in JPY...")
        
        # Historical volume characteristics in JPY
        volume_regimes = {
            'crisis_modern': {'mean': 180000000, 'std': 45000000, 'min': 120000000, 'max': 350000000},  # 2008+
            'normal_modern': {'mean': 95000000, 'std': 22000000, 'min': 70000000, 'max': 130000000},    # 2006-2025
            'early_2000s': {'mean': 52000000, 'std': 12000000, 'min': 40000000, 'max': 60000000},      # 2002-2004
            'crisis_early': {'mean': 78000000, 'std': 25000000, 'min': 60000000, 'max': 117700000}     # 2000-2005 crisis
        }
        
        # Generate volume for each date
        generated_volumes = []
        
        for date in nikkei_data.index:
            # 1. Determine historical regime
            regime_factor = self._get_historical_regime_factor(date, volume_regimes)
            
            # 2. Monthly seasonality
            monthly_factor = self._get_nikkei_monthly_factor(date.month)
            
            # 3. Crisis detection
            crisis_factor = self._detect_crisis_factor(date, nikkei_data)
            
            # 4. Daily price action factor
            price_action_factor = self._get_price_action_factor(date, nikkei_data)
            
            # 5. Gold inverse correlation
            gold_factor = self._get_gold_inverse_factor(date)
            
            # 6. Cross-market correlation factors
            asia_factor = self._get_asia_correlation_factor(date)
            us_factor = self._get_us_correlation_factor(date)
            europe_factor = self._get_europe_correlation_factor(date)
            
            # 7. Yearly trend factor
            yearly_factor = self._get_yearly_trend_factor(date)
            
            # Combine all factors using weighted approach
            combined_volume = self._combine_volume_factors(
                date, regime_factor, monthly_factor, crisis_factor, 
                price_action_factor, gold_factor, asia_factor, 
                us_factor, europe_factor, yearly_factor
            )
            
            generated_volumes.append(combined_volume)
        
        return pd.Series(generated_volumes, index=nikkei_data.index, name='Volume')

    def _get_historical_regime_factor(self, date, volume_regimes):
        """
        Determine which historical volume regime applies to this date.
        """
        year = date.year
        
        if year <= 2001:
            return volume_regimes['early_2000s']
        elif 2002 <= year <= 2005:
            # Check if crisis period (2000-2005)
            if year in [2000, 2001, 2002, 2008, 2009, 2020]:
                return volume_regimes['crisis_early']
            else:
                return volume_regimes['early_2000s']
        elif 2006 <= year <= 2025:
            # Modern era - check for crisis years
            crisis_years = [2008, 2009, 2011, 2020, 2022]  # Financial crisis, Fukushima, COVID, inflation
            if year in crisis_years:
                return volume_regimes['crisis_modern']
            else:
                return volume_regimes['normal_modern']
        else:
            return volume_regimes['normal_modern']

    def _get_nikkei_monthly_factor(self, month):
        """
        Japanese market seasonality factors.
        """
        monthly_factors = {
            1: 1.05,   # New Year effect, fresh institutional money
            2: 1.02,   # Pre-earnings season
            3: 1.20,   # Fiscal year end - heavy rebalancing
            4: 1.08,   # New fiscal year start
            5: 0.98,   # Golden Week holidays
            6: 0.95,   # Pre-summer quiet period
            7: 0.90,   # Summer holiday season starts
            8: 0.85,   # Obon holidays - lowest volume month
            9: 1.15,   # Half-year end + return from holidays
            10: 1.05,  # Earnings season
            11: 1.00,  # Normal trading
            12: 0.92,  # Year-end holidays, book closing
        }
        return monthly_factors.get(month, 1.0)

    def _detect_crisis_factor(self, date, nikkei_data):
        """
        Detect crisis periods using volatility and price movements.
        """
        try:
            # Get a window around this date
            window_start = max(0, nikkei_data.index.get_loc(date) - 20)
            window_end = min(len(nikkei_data), nikkei_data.index.get_loc(date) + 1)
            window_data = nikkei_data.iloc[window_start:window_end]
            
            if len(window_data) < 5:
                return 1.0
            
            # Calculate volatility indicators
            returns = window_data['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate price trend
            price_change_20d = (window_data['Close'].iloc[-1] / window_data['Close'].iloc[0] - 1) if len(window_data) > 1 else 0
            
            # Crisis detection thresholds
            high_vol_threshold = 0.25  # 25% annualized volatility
            large_decline_threshold = -0.15  # 15% decline
            
            crisis_factor = 1.0
            
            # High volatility increases volume
            if volatility > high_vol_threshold:
                crisis_factor *= (1.0 + min(2.0, volatility / high_vol_threshold))
            
            # Large declines increase volume more than rallies
            if price_change_20d < large_decline_threshold:
                crisis_factor *= (1.0 + abs(price_change_20d) * 3)
            elif price_change_20d > 0.15:  # Large rally
                crisis_factor *= (1.0 + price_change_20d * 1.5)
            
            return min(crisis_factor, 3.5)  # Cap at 3.5x normal volume
            
        except (IndexError, KeyError):
            return 1.0

    def _get_price_action_factor(self, date, nikkei_data):
        """
        Daily price action indicators affecting volume.
        """
        try:
            idx = nikkei_data.index.get_loc(date)
            current_data = nikkei_data.iloc[idx]
            
            # True range calculation
            if idx > 0:
                prev_close = nikkei_data.iloc[idx-1]['Close']
                true_range = max(
                    current_data['High'] - current_data['Low'],
                    abs(current_data['High'] - prev_close),
                    abs(current_data['Low'] - prev_close)
                )
                true_range_pct = true_range / current_data['Close']
            else:
                true_range_pct = (current_data['High'] - current_data['Low']) / current_data['Close']
            
            # Daily return
            if idx > 0:
                daily_return = (current_data['Close'] / nikkei_data.iloc[idx-1]['Close'] - 1)
            else:
                daily_return = 0
            
            # Volume factors based on price action
            range_factor = 1.0 + (true_range_pct * 8)  # Higher intraday range = higher volume
            return_factor = 1.0 + (abs(daily_return) * 5)  # Higher absolute return = higher volume
            
            # Combine factors
            price_action_factor = (range_factor + return_factor) / 2
            
            return min(price_action_factor, 2.5)  # Cap at 2.5x
            
        except (IndexError, KeyError):
            return 1.0

    def _get_gold_inverse_factor(self, date):
        """
        Inverse correlation with gold (flight to safety vs risk-on).
        """
        try:
            # Get gold data for this date
            gold_data = self.multi_index_df.loc[(slice(None), 'Gold'), :].reset_index(level='Symbol', drop=True)
            
            if date in gold_data.index:
                # Calculate gold volatility and trend
                gold_window = gold_data.loc[:date].tail(10)
                if len(gold_window) > 1:
                    gold_returns = gold_window['Close'].pct_change()
                    gold_volatility = gold_returns.std()
                    gold_trend = (gold_window['Close'].iloc[-1] / gold_window['Close'].iloc[0] - 1)
                    
                    # Inverse relationship: when gold surges (flight to safety), Nikkei volume often increases due to selling
                    # When gold drops (risk-on), Nikkei volume can be more moderate
                    if gold_trend > 0.02:  # Gold up >2%
                        return 1.1 + min(0.3, gold_trend * 5)  # Increased Nikkei volume (selling pressure)
                    elif gold_trend < -0.02:  # Gold down >2%
                        return 0.95 + min(0.1, abs(gold_trend) * 2)  # Moderate Nikkei volume increase
                    else:
                        return 1.0
            
            return 1.0
        except (KeyError, IndexError):
            return 1.0

    def _get_asia_correlation_factor(self, date):
        """
        Asia-Asia correlation (Hang Seng influence).
        """
        try:
            hang_data = self.multi_index_df.loc[(slice(None), 'Hang'), :].reset_index(level='Symbol', drop=True)
            
            if date in hang_data.index:
                # Get Hang Seng volume and volatility
                hang_volume = hang_data.loc[date, 'Volume']
                
                if pd.notna(hang_volume):
                    # Get recent Hang Seng volume average
                    hang_recent = hang_data.loc[:date, 'Volume'].tail(20)
                    hang_avg = hang_recent.mean()
                    
                    if hang_avg > 0:
                        hang_volume_ratio = hang_volume / hang_avg
                        # Strong positive correlation between Asian markets
                        return 0.9 + min(0.4, (hang_volume_ratio - 1) * 0.6)
            
            return 1.0
        except (KeyError, IndexError):
            return 1.0

    def _get_us_correlation_factor(self, date):
        """
        Asia-US correlation (SPY influence with time lag).
        """
        try:
            spy_data = self.multi_index_df.loc[(slice(None), 'SPY'), :].reset_index(level='Symbol', drop=True)
            
            # US markets close before Japan opens, so use previous day's US data
            prev_date = date - pd.Timedelta(days=1)
            
            if prev_date in spy_data.index:
                spy_volume = spy_data.loc[prev_date, 'Volume']
                
                if pd.notna(spy_volume):
                    spy_recent = spy_data.loc[:prev_date, 'Volume'].tail(20)
                    spy_avg = spy_recent.mean()
                    
                    if spy_avg > 0:
                        spy_volume_ratio = spy_volume / spy_avg
                        # Moderate positive correlation US -> Japan
                        return 0.95 + min(0.25, (spy_volume_ratio - 1) * 0.4)
            
            return 1.0
        except (KeyError, IndexError):
            return 1.0

    def _get_europe_correlation_factor(self, date):
        """
        Asia-Europe correlation (DAX influence with time lag).
        """
        try:
            dax_data = self.multi_index_df.loc[(slice(None), 'DAX'), :].reset_index(level='Symbol', drop=True)
            
            # European markets close before Japan opens
            prev_date = date - pd.Timedelta(days=1)
            
            if prev_date in dax_data.index:
                dax_volume = dax_data.loc[prev_date, 'Volume']
                
                if pd.notna(dax_volume):
                    dax_recent = dax_data.loc[:prev_date, 'Volume'].tail(20)
                    dax_avg = dax_recent.mean()
                    
                    if dax_avg > 0:
                        dax_volume_ratio = dax_volume / dax_avg
                        # Weaker correlation Europe -> Japan
                        return 0.98 + min(0.15, (dax_volume_ratio - 1) * 0.3)
            
            return 1.0
        except (KeyError, IndexError):
            return 1.0

    def _get_yearly_trend_factor(self, date):
        """
        Long-term yearly trends in volume.
        """
        year = date.year
        
        # Historical trend: volume generally increased from 2002 to 2025
        if year <= 2005:
            return 0.6  # Lower volumes in early 2000s
        elif 2006 <= year <= 2010:
            return 0.8 + (year - 2006) * 0.05  # Gradual increase
        elif 2011 <= year <= 2020:
            return 1.0 + (year - 2011) * 0.02  # Steady modern levels
        else:
            return 1.2  # Higher volumes in recent years
        
    def _combine_volume_factors(self, date, regime_factor, monthly_factor, crisis_factor, 
                            price_action_factor, gold_factor, asia_factor, 
                            us_factor, europe_factor, yearly_factor):
        """
        Combine all factors using weighted approach.
        """
        # Base volume from historical regime
        base_volume = regime_factor['mean']
        base_std = regime_factor['std']
        
        # Factor weights (sum to 1.0)
        weights = {
            'monthly': 0.15,
            'crisis': 0.25,      # Highest weight - crisis most important
            'price_action': 0.20, # Second highest - daily price action very important
            'gold': 0.10,
            'asia': 0.15,        # Regional correlation important
            'us': 0.08,
            'europe': 0.05,
            'yearly': 0.02
        }
        
        # Calculate weighted multiplier
        total_multiplier = (
            monthly_factor * weights['monthly'] +
            crisis_factor * weights['crisis'] +
            price_action_factor * weights['price_action'] +
            gold_factor * weights['gold'] +
            asia_factor * weights['asia'] +
            us_factor * weights['us'] +
            europe_factor * weights['europe'] +
            yearly_factor * weights['yearly']
        )
        
        # Apply multiplier to base volume
        adjusted_volume = base_volume * total_multiplier
        
        # Add realistic noise (log-normal distribution)
        noise_factor = np.random.lognormal(mean=0, sigma=0.15)  # 15% volatility
        final_volume = adjusted_volume * noise_factor
        
        # Ensure within realistic bounds
        min_volume = regime_factor['min']
        max_volume = regime_factor['max']
        
        # Allow crisis periods to exceed normal max
        if crisis_factor > 2.0:
            max_volume *= 1.5
        
        final_volume = np.clip(final_volume, min_volume, max_volume)
        
        return int(final_volume)

    def analyze_and_clean_data(self):
        """
        Comprehensive data quality analysis before saving.
        - Reports all NaN and inf values
        - Identifies dates with missing/invalid data
        - Asks user permission to delete affected dates
        - Deletes entire dates (all symbols) if user approves
        """
        print("\n" + "="*70)
        print("DATA QUALITY ANALYSIS")
        print("="*70)
        
        # 1. Count all values in theory
        num_symbols = self.multi_index_df.index.get_level_values('Symbol').nunique()
        num_dates = self.multi_index_df.index.get_level_values('Date').nunique()
        num_columns = self.multi_index_df.shape[1]
        
        theoretical_values = num_symbols * num_dates * num_columns
        actual_values = self.multi_index_df.shape[0] * num_columns
        
        print(f"\n📊 DATASET DIMENSIONS:")
        print(f"  Symbols: {num_symbols}")
        print(f"  Dates: {num_dates}")
        print(f"  Data columns: {num_columns}")
        print(f"  Total values (theoretical): {theoretical_values:,}")
        print(f"  Total values (actual): {actual_values:,}")
        
        # 2. Count NaN values
        nan_count = self.multi_index_df.isna().sum().sum()
        nan_percentage = (nan_count / actual_values * 100) if actual_values > 0 else 0
        
        print(f"\n⚠️  NaN VALUES:")
        print(f"  Total NaN count: {nan_count:,}")
        print(f"  NaN percentage: {nan_percentage:.3f}%")
        
        # 3. Count inf values
        inf_count = np.isinf(self.multi_index_df).sum().sum()
        inf_percentage = (inf_count / actual_values * 100) if actual_values > 0 else 0
        
        print(f"\n⚠️  INFINITE VALUES:")
        print(f"  Total inf/−inf count: {inf_count:,}")
        print(f"  Inf percentage: {inf_percentage:.3f}%")
        
        # 4. Identify dates with NaN values
        dates_with_nan = set()
        for (date, symbol), row in self.multi_index_df.iterrows():
            if row.isna().any():
                dates_with_nan.add(date)
        
        dates_with_nan_count = len(dates_with_nan)
        dates_with_nan_percentage = (dates_with_nan_count / num_dates * 100) if num_dates > 0 else 0
        
        print(f"\n📅 DATES WITH NaN VALUES:")
        print(f"  Dates affected: {dates_with_nan_count}")
        print(f"  Total dates: {num_dates}")
        print(f"  Percentage of dates to delete: {dates_with_nan_percentage:.2f}%")
        
        # 5. Identify dates with inf values
        dates_with_inf = set()
        for (date, symbol), row in self.multi_index_df.iterrows():
            if np.isinf(row).any():
                dates_with_inf.add(date)
        
        dates_with_inf_count = len(dates_with_inf)
        
        print(f"\n📅 DATES WITH ±INF VALUES:")
        print(f"  Dates affected: {dates_with_inf_count}")
        
        # 6. Combined list of dates to delete
        dates_to_delete = dates_with_nan.union(dates_with_inf)
        total_rows_to_delete = dates_to_delete_count = len(dates_to_delete)
        total_rows_before = self.multi_index_df.shape[0]
        total_rows_after = total_rows_before - (dates_to_delete_count * num_symbols)
        
        print(f"\n🔴 DATES REQUIRING DELETION (NaN OR INF):")
        print(f"  Total dates to delete: {dates_to_delete_count}")
        print(f"  Percentage of dates: {(dates_to_delete_count / num_dates * 100):.2f}%")
        print(f"  Rows before deletion: {total_rows_before:,}")
        print(f"  Rows to be deleted: {dates_to_delete_count * num_symbols:,}")
        print(f"  Rows after deletion: {total_rows_after:,}")
        
        # 7. Ask user for permission
        print("\n" + "="*70)
        if dates_to_delete_count > 0:
            print(f"\n⚠️  {dates_to_delete_count} dates contain NaN or inf values.")
            print("   Deleting these dates will remove ALL symbols for those dates.")
            
            user_input = input("\nDelete these dates? (yes/no): ").strip().lower()
            
            if user_input in ['yes', 'y']:
                print(f"\n🗑️  Deleting {dates_to_delete_count} dates...")
                
                # Delete dates with NaN or inf
                dates_to_keep = self.multi_index_df.index.get_level_values('Date').unique().difference(dates_to_delete)
                self.multi_index_df = self.multi_index_df.loc[dates_to_keep]
                
                print(f"✓ Deletion complete.")
                print(f"  New shape: {self.multi_index_df.shape}")
                
                # Re-check for remaining NaN/inf after deletion
                remaining_nan = self.multi_index_df.isna().sum().sum()
                remaining_inf = np.isinf(self.multi_index_df).sum().sum()
                
                print(f"\n📊 DATA AFTER DELETION:")
                print(f"  Remaining NaN values: {remaining_nan}")
                print(f"  Remaining inf values: {remaining_inf}")
                
                if remaining_nan == 0 and remaining_inf == 0:
                    print("\n✅ Data is now clean. Ready to save.")
                    return True
                else:
                    print("\n⚠️  WARNING: Data still contains NaN or inf values after deletion.")
                    return False
            else:
                print("\n❌ Deletion cancelled. Data will not be saved.")
                return False
        else:
            print("\n✅ No NaN or inf values found. Data is clean.")
            return True


    def save_multi_index_df(self):
        """
        Save the multi-index DataFrame after data quality checks.
        """

        # Check wether ANY NaN or inf values remain before saving
        if self.multi_index_df.isna().sum().sum() > 0 or np.isinf(self.multi_index_df).sum().sum() > 0:
            print("\n❌ Cannot save data: NaN or inf values still present.")
            print("Please address remaining issues before saving.")
            return
        
        # Save to CSV
        output_path = "src/data/processed_multi_index_data.csv"
        
        # Reset index to make Date and Symbol regular columns for CSV export
        df_to_save = self.multi_index_df.reset_index()
        
        try:
            df_to_save.to_csv(output_path, index=False)
            print(f"\n✅ Data successfully saved to: {output_path}")
            print(f"   Shape: {df_to_save.shape}")
            print(f"   Date range: {df_to_save['Date'].min()} to {df_to_save['Date'].max()}")
        except Exception as e:
            print(f"\n❌ Error saving file: {e}")

# Execution
datapreper = Dataprep()