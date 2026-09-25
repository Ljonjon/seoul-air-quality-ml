"""
Data Preprocessing Module (revised)

Pipeline: load -> instrument-status filter -> sensor-error handling -> merge
-> temporal & spatial features -> dynamic (lag / rolling) features -> target
-> chronological + random-stratified splits -> standardization.

Revisions compared with the original course submission
---------------------------------------------------------------------------
[FIX-1] Label leakage in PM2.5_roll6.
        rolling(6).mean() included the *current* hour, i.e. the very PM2.5
        value the target grade is derived from. The rolling mean now uses
        the six *previous* hours only (shift(1) before rolling).
[FIX-2] Time-series integrity.
        Rows are explicitly sorted by (Station code, Measurement date)
        before any lag / rolling computation.
[FIX-3] The instrument-status filter is now actually applied.
        The original code loaded Measurement_info.csv and created
        df_info_clean but never used it. Hourly summaries where ANY of the
        six measured items had status != 0 are now dropped, matching the
        report's description.
[FIX-4] Chronological train/test split as the primary protocol.
        Train = first 80% of the timeline, test = last 20%. The original
        random stratified split is kept as a comparison protocol so the
        optimism caused by shuffling time-series data can be quantified.
[FIX-5] Clean scaling.
        Scalers are fitted on the training part of each protocol only and
        written back without chained-assignment warnings.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

POLLUTANT_COLS = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
NUM_COLS = ['SO2', 'NO2', 'O3', 'CO', 'PM10',
            'Latitude', 'Longitude', 'Hour', 'DayOfWeek', 'Month',
            'PM2.5_lag1', 'PM10_lag1', 'PM2.5_roll6']
TARGET = 'PM2.5_grade'

# Columns kept out of the model matrix
NON_FEATURE_COLS = ['Measurement date', 'Station code', 'PM2.5', TARGET]


def get_pm25_grade(value):
    """Korean Air Quality index bands for PM2.5 (ug/m3)."""
    if value <= 15:
        return 0      # Good
    elif value <= 35:
        return 1      # Normal
    elif value <= 75:
        return 2      # Bad
    return 3          # Very Bad


def load_and_clean(base_path, log=print):
    """Step 1-3: robust loading, instrument-status filter, sensor errors."""
    try:
        df_summary = pd.read_csv(f'{base_path}/Measurement_summary.csv')
        df_station = pd.read_csv(f'{base_path}/Original Data/Measurement_station_info.csv')
        df_info = pd.read_csv(
            f'{base_path}/Original Data/Measurement_info.csv',
            usecols=['Measurement date', 'Station code', 'Instrument status'])
        log('[OK] All datasets loaded successfully.')
    except FileNotFoundError as e:
        log(f'[ERROR] Data files not found ({e}). '
            f'Expected layout: {base_path}/Measurement_summary.csv and '
            f'{base_path}/Original Data/*.csv')
        raise

    log(f'Raw summary rows: {len(df_summary)} | raw info rows: {len(df_info)}')

    # ---- [FIX-3] instrument-status filter --------------------------------
    bad_pairs = (df_info.loc[df_info['Instrument status'].fillna(-1) != 0,
                             ['Measurement date', 'Station code']]
                 .drop_duplicates()
                 .assign(_status_bad=1))
    flagged = df_summary[['Measurement date', 'Station code']].merge(
        bad_pairs, on=['Measurement date', 'Station code'], how='left')
    n_bad = int(flagged['_status_bad'].sum())
    keep_mask = flagged['_status_bad'].isna().to_numpy()
    df_summary = df_summary.loc[keep_mask].copy()
    log(f'Dropped {n_bad} rows ({100 * n_bad / len(keep_mask):.2f}%) with any '
        f'instrument status != 0 (calibration needed / abnormal / power cut / repair)')

    # ---- sensor error sentinel (-1) ---------------------------------------
    neg_counts = {}
    for col in POLLUTANT_COLS:
        neg_counts[col] = int((df_summary[col] < 0).sum())
        df_summary.loc[df_summary[col] < 0, col] = np.nan
    log(f'Negative sentinel values (-1) per pollutant: {neg_counts}')

    before = len(df_summary)
    df_summary = df_summary.dropna(subset=POLLUTANT_COLS).copy()
    log(f'Dropped {before - len(df_summary)} rows with missing pollutant values; '
        f'summary rows now: {len(df_summary)}')

    # ---- integrate station metadata ---------------------------------------
    df = pd.merge(df_summary, df_station, on='Station code', how='left')
    df = df.drop(columns=['Address_x', 'Address_y', 'Latitude_y', 'Longitude_y'])
    df = df.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude'})
    df['Measurement date'] = pd.to_datetime(df['Measurement date'])

    # ---- [FIX-2] enforce temporal order before lag/rolling -----------------
    df = df.sort_values(['Station code', 'Measurement date']).reset_index(drop=True)
    return df


def build_features(df, log=print):
    """Step 4-5: temporal / spatial / dynamic features and the target."""
    log('Extracting temporal features (Hour, DayOfWeek, Month)...')
    df['Hour'] = df['Measurement date'].dt.hour
    df['DayOfWeek'] = df['Measurement date'].dt.dayofweek
    df['Month'] = df['Measurement date'].dt.month

    log('One-Hot encoding districts...')
    df = pd.get_dummies(df, columns=['Station name(district)'], prefix='Dist')

    log('Creating 1-hour lag features (PM2.5, PM10)...')
    grp = df.groupby('Station code', sort=False)
    df['PM2.5_lag1'] = grp['PM2.5'].shift(1)
    df['PM10_lag1'] = grp['PM10'].shift(1)

    # ---- [FIX-1] the leakage fix -------------------------------------------
    # 6-hour rolling mean of the PREVIOUS six hours only.
    # (The original submission used rolling(6).mean() on the current value,
    #  which mixes the hour being predicted -- and hence the label -- into
    #  the feature.)
    log('Creating leakage-free 6-hour rolling mean of PM2.5 (previous hours only)...')
    shifted = grp['PM2.5'].shift(1)
    df['PM2.5_roll6'] = (shifted.groupby(df['Station code'], sort=False)
                         .rolling(6).mean()
                         .reset_index(level=0, drop=True))

    before = len(df)
    df = df.dropna(subset=['PM2.5_lag1', 'PM10_lag1', 'PM2.5_roll6']).copy()
    log(f'Dropped {before - len(df)} warm-up rows (first hours per station); '
        f'feature rows now: {len(df)}')

    log('Creating classification target PM2.5_grade...')
    df[TARGET] = df['PM2.5'].apply(get_pm25_grade)
    return df


def _scaled_frame(frame, cols, scaler, fit):
    """[FIX-5] Replace given columns with standardized values (no chained assignment)."""
    out = frame.copy()
    arr = scaler.fit_transform(out[cols]) if fit else scaler.transform(out[cols])
    out[cols] = pd.DataFrame(arr, index=out.index, columns=cols).astype('float32')
    return out


def _pack(X_train, X_test, y_train, y_test, extra=None):
    scaler = StandardScaler()
    X_train = _scaled_frame(X_train, NUM_COLS, scaler, fit=True)
    X_test = _scaled_frame(X_test, NUM_COLS, scaler, fit=False)
    pack = {'X_train': X_train, 'X_test': X_test,
            'y_train': y_train.astype('int64'), 'y_test': y_test.astype('int64')}
    if extra:
        pack.update(extra)
    return pack


def make_splits(df, test_size=0.2, random_state=42, log=print):
    """[FIX-4] Chronological split (primary) + random stratified split (comparison)."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].astype('float32')
    y = df[TARGET]
    t = df['Measurement date']
    log(f'Final feature matrix: {X.shape[0]} rows x {X.shape[1]} columns')

    # --- protocol A: chronological ---
    cutoff = t.quantile(1 - test_size)
    train_mask = t < cutoff
    chrono = _pack(X[train_mask].copy(), X[~train_mask].copy(),
                   y[train_mask], y[~train_mask],
                   extra={'cutoff': str(cutoff)})
    log(f"[protocol=chronological] train rows={int(train_mask.sum())} "
        f"(before {cutoff}), test rows={int((~train_mask).sum())}")

    # --- protocol B: random stratified (original submission) ---
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    rnd = _pack(Xr_train.copy(), Xr_test.copy(), yr_train, yr_test)
    log(f'[protocol=random-stratified] train rows={len(Xr_train)}, '
        f'test rows={len(Xr_test)} (comparative protocol only)')

    for name, pack in (('chronological', chrono), ('random-stratified', rnd)):
        dist_tr = (pack['y_train'].value_counts(normalize=True).sort_index() * 100).round(2).to_dict()
        dist_te = (pack['y_test'].value_counts(normalize=True).sort_index() * 100).round(2).to_dict()
        log(f'[protocol={name}] train class % {dist_tr} | test class % {dist_te}')

    return {'chronological': chrono, 'random-stratified': rnd,
            'n_features': X.shape[1]}


def get_preprocessed_data(base_path=None, test_size=0.2, log=print):
    """Convenience entry point returning both split protocols."""
    base_path = base_path or os.environ.get('AIR_DATA_PATH') or './AirPollutionSeoul'
    df = load_and_clean(base_path, log=log)
    df = build_features(df, log=log)
    data = make_splits(df, test_size=test_size, log=log)
    data['timeline_start'] = str(df['Measurement date'].min())
    data['timeline_end'] = str(df['Measurement date'].max())
    return data


if __name__ == '__main__':
    data = get_preprocessed_data()
    print('\nTest run OK. Feature shape (train, chronological):',
          data['chronological']['X_train'].shape)
