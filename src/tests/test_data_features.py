import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import data_processor_2 as processor_module
from data.data_processor_2 import DataEnricher


def _market_frame(days=420):
    random = np.random.default_rng(42)
    dates = pd.bdate_range('2020-01-01', periods=days)
    frames = []
    for symbol in ['Crude', 'Gold', 'SPY']:
        close = 100 * np.exp(np.cumsum(random.normal(0, 0.01, days)))
        frames.append(pd.DataFrame({
            'Date': dates, 'Symbol': symbol, 'Open': close * 0.999,
            'High': close * 1.01, 'Low': close * 0.99, 'Close': close,
            'Volume': np.exp(random.normal(12, 0.5, days)),
        }))
    return pd.concat(frames, ignore_index=True)


def _processor(frame):
    processor = DataEnricher.__new__(DataEnricher)
    processor.raw_indices_data = frame.copy()
    processor.enriched_data = frame.copy()
    processor.starting_date = frame.Date.min()
    processor.ending_date = frame.Date.max()
    processor.max_norm_horizon = 36
    processor.verbose = 0
    return processor


def test_fractional_difference_matches_explicit_windows():
    processor = _processor(_market_frame())
    weights = processor._get_global_ffd_weights(0.5, 0.1)
    series = pd.Series([1., 2., 4., 8.], index=[10, 11, 12, 13])
    result = processor._apply_ffd_to_series(series, weights)
    np.testing.assert_allclose(result, [2.875, 5.75])
    assert result.index.tolist() == [12, 13]
    assert processor._apply_ffd_to_series(series.iloc[:2], weights).isna().all()


def test_ffd_fit_and_acceptance_exclude_test_prices(monkeypatch):
    frame = _market_frame()
    cutoff = sorted(frame.Date.unique())[-253]
    fit_frames = []
    accepted_series = []

    def select_order(self, fit_frame):
        assert fit_frame.Date.max() == cutoff
        fit_frames.append(fit_frame.copy())
        return 0.5

    def stationarity(series, **kwargs):
        accepted_series.append(series.copy())
        return (-3., 0.01)

    monkeypatch.setattr(DataEnricher, 'find_multi_asset_golden_d', select_order)
    monkeypatch.setattr(processor_module, 'adfuller', stationarity)
    original = _processor(frame)
    original.add_frac_diff_features()
    changed_frame = frame.copy()
    changed_frame.loc[changed_frame.Date > cutoff, 'Close'] *= 100
    changed = _processor(changed_frame)
    changed.add_frac_diff_features()
    pd.testing.assert_frame_equal(fit_frames[0], fit_frames[1])
    for original_series, changed_series in zip(accepted_series[:3], accepted_series[3:]):
        pd.testing.assert_series_equal(original_series, changed_series)
    assert original.ffd_metadata == changed.ffd_metadata
    assert len(original.ffd_test_dates) == 252
    pd.testing.assert_frame_equal(
        original.enriched_data[original.enriched_data.Date <= cutoff],
        changed.enriched_data[changed.enriched_data.Date <= cutoff],
    )


def test_market_features_have_distinct_horizons_and_causal_windows():
    frame = _market_frame()
    original = _processor(frame)
    original.convert_raw_to_stationary_data()
    features = original.enriched_data
    spy = features[features.Symbol == 'SPY']
    for period in [1, 5, 21]:
        np.testing.assert_allclose(spy[f'rel_volume_intensity_{period}d'].dropna(), 0)
        np.testing.assert_allclose(spy[f'z_vol_breakout_{period}d'].dropna(), 0)
        breakout = features.loc[features.Symbol != 'SPY', f'z_vol_breakout_{period}d'].dropna()
        assert breakout.nunique() > 20
        assert (breakout.abs() == 4).mean() < 0.1
    for feature in ['log_volume_intensity', 'norm_volume_feature']:
        assert not features[f'{feature}_1d'].equals(features[f'{feature}_5d'])
        assert not features[f'{feature}_5d'].equals(features[f'{feature}_21d'])
    cutoff = sorted(frame.Date.unique())[250]
    changed = _processor(frame)
    changed.raw_indices_data.loc[changed.raw_indices_data.Date > cutoff, ['Close', 'Volume']] *= 2
    changed.convert_raw_to_stationary_data()
    corrected_columns = ['Date', 'Symbol'] + [
        column for column in features.columns
        if column.startswith(('z_vol_breakout_', 'log_volume_intensity_', 'rel_volume_intensity_', 'norm_volume_feature_'))
    ]
    pd.testing.assert_frame_equal(
        features.loc[features.Date <= cutoff, corrected_columns],
        changed.enriched_data.loc[changed.enriched_data.Date <= cutoff, corrected_columns],
    )


def test_cleaning_cannot_move_reserved_test_dates(tmp_path):
    processor = _processor(_market_frame())
    processor.output_dir = str(tmp_path)
    processor.ffd_test_dates = pd.DatetimeIndex(sorted(processor.enriched_data.Date.unique()))[-252:]
    processor.ffd_metadata = {}
    processor.enriched_data = processor.enriched_data[
        processor.enriched_data.Date != processor.ffd_test_dates[0]
    ]
    with pytest.raises(ValueError, match='reserved final 252'):
        processor.save_calculated_data()


def test_generated_metadata_records_feature_order_and_fit_cutoff(tmp_path, monkeypatch):
    processor = _processor(_market_frame())
    processor.output_dir = str(tmp_path)
    monkeypatch.setattr(processor, 'find_multi_asset_golden_d', lambda frame: 0.5)
    monkeypatch.setattr(processor, 'report_stationarity', lambda: True)
    processor.add_frac_diff_features()
    processor.enriched_data = processor.enriched_data.dropna()
    processor.save_calculated_data()
    metadata = json.loads((tmp_path / 'enriched_financial_data_metadata.json').read_text())
    assert metadata['feature_schema_version'] == 2
    assert metadata['test_days'] == 252
    assert metadata['ffd_fit_end'] < metadata['test_start']
    assert metadata['ffd_window_size'] == len(metadata['ffd_weights_oldest_to_newest'])
    assert metadata['feature_columns'] == list(processor.enriched_data.columns)