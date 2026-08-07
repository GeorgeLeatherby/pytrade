## Rework of data
What to try:    - keep more meaningful data wihtin features
                - apply normalization with self.max_norm_horizon

## features to use in new training: (Reworked or reviewed again)
- log_open/high/low/close_return_{period}d Try for period=1,5,21
- z_rel_log_close_{period}d
- z_vol_breakout_{period}d
- log_volume_intensity_{period}d
- norm_volume_feature_{period}d
- atr_z_norm
- rsi_14
- bb_width_reworked
- z_fisher_corr_spy
- z_market_dispersion
- z_beta_spy


Which to use: data_processor_2.py is the recommended one — it's the more recent iteration. It replaces the large duplicated risk-metric blocks (which were commented out anyway) with theoretically sounder features: the intraday OHLC features and the FFD feature (norm_ffd_feature) are meaningful improvements. The dropped cross-asset relationship metrics (spillovers, leadership) were computationally expensive and largely redundant with the Fisher-Z/beta features that remain.