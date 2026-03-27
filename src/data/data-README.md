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
- z_distance_to_mean
- z_fisher_corr_spy
- z_market_dispersion
