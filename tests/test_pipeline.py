"""
Regression tests for the data pipeline (synthetic data only).

No dataset download is required: these tests build a tiny, fully synthetic
copy of the three source CSVs in a temporary directory and assert the
properties that the graded course version of this project got wrong, so the
fixes can never silently regress.  They also run in CI on every push.

Run:  pytest -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import (  # noqa: E402
    NUM_COLS,
    POLLUTANT_COLS,
    TARGET,
    build_features,
    get_pm25_grade,
    load_and_clean,
    make_splits,
)

STATIONS = {
    101: ('Jongno-gu', 37.5720, 127.0050),
    102: ('Mapo-gu', 37.5583, 126.9270),
}
N_HOURS = 24 * 60                       # 60 daily cycles per station
START = pd.Timestamp('2019-01-01 00:00')
BAD_HOURS = (5, 6, 20)                  # abnormal instrument status, station 101
SENTINEL_HOUR = 100                     # PM2.5 == -1 sensor error, station 101
SILENT = lambda *a, **k: None           # noqa: E731  (quiet the pipeline log)


def pm25_at(code, h):
    """Deterministic hourly PM2.5 that sweeps all four health-grade bands."""
    phase = 0.0 if code == 101 else np.pi / 3
    return round(45.0 + 40.0 * np.sin(2 * np.pi * h / 24.0 + phase), 1)


def cleaned_hours(code):
    """Hours that survive cleaning for a station: status filter + sensor sentinel."""
    removed = {5, 6, 20, 100} if code == 101 else set()
    return [h for h in range(N_HOURS) if h not in removed]


def co_pollutant_at(code, h, scale):
    phase = 0.0 if code == 101 else np.pi / 3
    return round(scale * (1.0 + np.sin(2 * np.pi * h / 12.0 + phase)), 3)


def write_dataset(base):
    """Materialise a fake AirPollutionSeoul folder below 'base'."""
    base = str(base)
    os.makedirs(os.path.join(base, 'Original Data'), exist_ok=True)

    summary_rows, info_rows = [], []
    for code, (district, lat, lon) in STATIONS.items():
        for h in range(N_HOURS):
            stamp = (START + pd.Timedelta(hours=h)).strftime('%Y-%m-%d %H:%M')
            summary_rows.append({
                'Measurement date': stamp, 'Station code': code,
                'Address': district + ', Seoul', 'Latitude': lat, 'Longitude': lon,
                'SO2': co_pollutant_at(code, h, 0.02),
                'NO2': co_pollutant_at(code, h, 0.04),
                'O3': co_pollutant_at(code, h, 0.06),
                'CO': co_pollutant_at(code, h, 0.5),
                'PM10': round(pm25_at(code, h) * 1.8, 1),
                'PM2.5': pm25_at(code, h),
            })
            for item in range(1, 7):
                status = 8 if (code == 101 and h in BAD_HOURS and item in (1, 3)) else 0
                info_rows.append({'Measurement date': stamp, 'Station code': code,
                                  'Item code': item, 'Average value': 0.0,
                                  'Instrument status': status})

    summary_rows[SENTINEL_HOUR]['PM2.5'] = -1.0     # sensor error sentinel

    pd.DataFrame(summary_rows).to_csv(os.path.join(base, 'Measurement_summary.csv'),
                                      index=False)
    pd.DataFrame([{'Station code': c, 'Station name(district)': d,
                   'Address': d + ', Seoul', 'Latitude': la, 'Longitude': lo}
                  for c, (d, la, lo) in STATIONS.items()]).to_csv(
        os.path.join(base, 'Original Data', 'Measurement_station_info.csv'), index=False)
    pd.DataFrame(info_rows).to_csv(os.path.join(base, 'Original Data',
                                                'Measurement_info.csv'), index=False)
    return base


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp('AirPollutionSeoul'))


def hours_since_start(frame):
    return (frame['Measurement date'] - START).dt.total_seconds().to_numpy() // 3600


# ---------------------------------------------------------------- cleaning --
def test_grade_bands_use_the_inclusive_korean_thresholds():
    assert get_pm25_grade(15) == 0 and get_pm25_grade(15.1) == 1
    assert get_pm25_grade(35) == 1 and get_pm25_grade(35.1) == 2
    assert get_pm25_grade(75) == 2 and get_pm25_grade(75.1) == 3


def test_abnormal_instrument_status_rows_are_dropped(data):
    """[FIX-3] the status filter is really applied, and only to flagged hours."""
    cleaned = load_and_clean(data, log=SILENT)
    kept = set(cleaned.loc[cleaned['Station code'] == 101, 'Measurement date'])
    flagged = {START + pd.Timedelta(hours=h) for h in BAD_HOURS}
    assert not (flagged & kept), 'abnormal-status hours survived the filter'
    # the identical clock hours on later days are untouched (filter is per hour,
    # not per hour-of-day)
    assert START + pd.Timedelta('1d') + pd.Timedelta(hours=5) in kept
    # 2 stations x 720 h, minus 3 flagged hours and 1 negative-sentinel row
    assert len(cleaned) == 2 * N_HOURS - len(BAD_HOURS) - 1
    assert (cleaned[POLLUTANT_COLS] >= 0).all().all()


def test_negative_sensor_sentinels_never_reach_the_features(data):
    feats = build_features(load_and_clean(data, log=SILENT), log=SILENT)
    assert not feats[POLLUTANT_COLS].isna().any().any()
    assert (feats[POLLUTANT_COLS] >= 0).all().all()


# ------------------------------------------------------- leakage / ordering --
def test_roll6_window_contains_previous_hours_only(data):
    """[FIX-1] the 6-hour rolling mean must exclude the hour being labelled."""
    feats = build_features(load_and_clean(data, log=SILENT), log=SILENT)
    g = feats[feats['Station code'] == 101].sort_values('Measurement date')
    i = 200
    clean_prev = g['PM2.5'].iloc[i - 6:i].mean()      # previous six hours
    leaky = g['PM2.5'].iloc[i - 5:i + 1].mean()       # includes the label hour
    assert g['PM2.5_roll6'].iloc[i] == pytest.approx(clean_prev, rel=1e-6)
    assert leaky != pytest.approx(clean_prev, rel=1e-6)   # window is not degenerate
    assert g['PM2.5_roll6'].iloc[i] != pytest.approx(leaky, rel=1e-9)


def test_perturbing_the_label_hour_cannot_change_its_own_features(data):
    """Intervention test: change PM2.5 at hour t -> the label moves, its features do not."""
    before = build_features(load_and_clean(data, log=SILENT), log=SILENT)

    raw = load_and_clean(data, log=SILENT)
    st = raw[raw['Station code'] == 101].sort_values('Measurement date')
    row = st.index[60]
    stamp = raw.loc[row, 'Measurement date']
    raw.loc[row, 'PM2.5'] = 180.0                      # an invented extreme episode
    after = build_features(raw, log=SILENT)

    key = ['Station code', 'Measurement date']
    a, b = before.set_index(key), after.set_index(key)
    assert a.loc[(101, stamp), 'PM2.5_roll6'] == pytest.approx(
        b.loc[(101, stamp), 'PM2.5_roll6'], rel=1e-9), 'label hour leaks into its own window'
    assert a.loc[(101, stamp), 'PM2.5_lag1'] == pytest.approx(
        b.loc[(101, stamp), 'PM2.5_lag1'], rel=1e-9)
    assert a.loc[(101, stamp), 'PM2.5_grade'] != b.loc[(101, stamp), 'PM2.5_grade']
    assert b.loc[(101, stamp), 'PM2.5_grade'] == 3

    # the extreme hour legitimately becomes history for the NEXT six hours, and
    # for nothing else: exactly six rows shift, each by (180 - old) / 6.
    ga = before[before['Station code'] == 101].set_index('Measurement date')['PM2.5_roll6']
    gb = after[after['Station code'] == 101].set_index('Measurement date')['PM2.5_roll6']
    shared = ga.index.intersection(gb.index)
    delta = (gb.loc[shared] - ga.loc[shared]).abs()
    shifted = delta[delta > 1e-6].sort_index()
    assert len(shifted) == 6, 'a 6-hour window must touch exactly six later rows'
    assert (shifted.index - stamp).min() == pd.Timedelta('1h')
    assert (shifted.index - stamp).max() == pd.Timedelta('6h')
    assert float(shifted.iloc[0]) == pytest.approx((180.0 - pm25_at(101, 63)) / 6.0, abs=1e-3)


def test_dynamic_features_respect_station_groups_and_sort_order(data):
    """[FIX-2] lag/rolling are built per station on a time-sorted frame."""
    feats = build_features(load_and_clean(data, log=SILENT), log=SILENT)
    for code in STATIONS:
        g = feats[feats['Station code'] == code].sort_values('Measurement date')
        h = hours_since_start(g).astype(int)
        assert not g[['PM2.5_lag1', 'PM10_lag1', 'PM2.5_roll6']].isna().any().any()
        assert (np.diff(h) > 0).all(), 'rows must be strictly time-ordered per station'

        truth = np.array(cleaned_hours(code), dtype=float)
        values = np.array([pm25_at(code, int(x)) for x in truth])
        # the six warm-up rows of each station are gone, and gaps stay inside
        # the station's own series (station 101 really lost hours 5, 6, 20, 100)
        assert list(h) == list(truth[6:])
        np.testing.assert_allclose(g['PM2.5_lag1'].to_numpy(dtype='float64'),
                                   values[5:-1], rtol=1e-5, atol=1e-4)
        expected_roll = np.array([values[p - 6:p].mean()
                                  for p in range(6, len(values))])
        np.testing.assert_allclose(g['PM2.5_roll6'].to_numpy(dtype='float64'),
                                   expected_roll, rtol=1e-5, atol=1e-4)
        if code == 101:
            assert h[0] == 8, 'warm-up must follow the status-filtered timeline'


def test_target_and_raw_pm25_are_absent_from_the_model_matrix(data):
    feats = build_features(load_and_clean(data, log=SILENT), log=SILENT)
    packs = make_splits(feats, log=SILENT)
    for protocol in ('chronological', 'random-stratified'):
        pack = packs[protocol]
        cols = set(pack['X_train'].columns)
        assert TARGET not in cols and 'PM2.5' not in cols
        assert not pack['X_train'].isna().any().any()
        assert not pack['X_test'].isna().any().any()


# ------------------------------------------------------- splits / scaling ----
def test_chronological_split_never_trains_on_the_future(data):
    """[FIX-4] every train timestamp precedes every test timestamp."""
    feats = build_features(load_and_clean(data, log=SILENT), log=SILENT)
    pack = make_splits(feats, log=SILENT)['chronological']
    tr = feats.loc[pack['X_train'].index, 'Measurement date']
    te = feats.loc[pack['X_test'].index, 'Measurement date']
    assert tr.max() < te.min()
    assert len(tr) + len(te) == len(feats)
    assert len(tr) > len(te)


def test_scaler_is_fitted_on_train_only(data):
    """[FIX-5] standardisation statistics come from the training rows."""
    feats = build_features(load_and_clean(data, log=SILENT), log=SILENT)
    pack = make_splits(feats, log=SILENT)['chronological']
    train = pack['X_train'][NUM_COLS].to_numpy(dtype='float64')
    assert np.abs(train.mean(axis=0)).max() < 1e-5
    assert np.abs(train.std(axis=0, ddof=0) - 1.0).max() < 1e-3


def test_all_three_classifiers_run_end_to_end(data):
    from task_classification import build_models, train_and_evaluate

    feats = build_features(load_and_clean(data, log=SILENT), log=SILENT)
    pack = make_splits(feats, log=SILENT)['chronological']
    for name, model in build_models().items():
        res = train_and_evaluate(model, name, pack['X_train'], pack['y_train'],
                                 pack['X_test'], pack['y_test'], save_prefix=None)
        assert 0.0 < res['accuracy'] <= 1.0
        assert res['true_verybad'] > 0, 'synthetic series should cover every grade'
        assert len(res['confusion_matrix']) == 4
