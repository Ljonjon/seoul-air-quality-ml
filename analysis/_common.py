"""
Shared helpers for the analysis scripts in this folder.

These scripts answer the questions an interviewer asks *after* the model
table: "how do you know the numbers are real?", "what is the lift over a
trivial baseline?", "how much did each of your fixes actually buy?".

Everything here reads the same three CSVs as `main.py` and writes machine
readable JSON into `results/`, so no number quoted in the README has to be
taken on faith.

Usage (from the repository root):
    python analysis/leak_ablation.py --data /path/to/AirPollutionSeoul
    python analysis/baselines_importance.py --data ...
    python analysis/pca_variance_check.py --data ...
    python analysis/data_audit.py --data ...
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocessing import (  # noqa: E402
    NUM_COLS, POLLUTANT_COLS, TARGET, build_features, get_pm25_grade, load_and_clean,
)
from task_classification import build_models  # noqa: E402

RESULTS_DIR = REPO_ROOT / 'results'
SILENT = lambda *a, **k: None  # noqa: E731


def resolve_data_path(cli_value=None):
    """Same precedence chain as main.py: flag -> AIR_DATA_PATH -> ./AirPollutionSeoul."""
    path = cli_value or os.environ.get('AIR_DATA_PATH') or str(REPO_ROOT / 'AirPollutionSeoul')
    if not (Path(path) / 'Measurement_summary.csv').exists():
        raise SystemExit(
            f'[data] Measurement_summary.csv not found under {path!r}.\n'
            f'       Download the Kaggle "Air Pollution in Seoul" archive and pass '
            f'--data /path/to/AirPollutionSeoul')
    return path


def add_argument(parser):
    parser.add_argument('--data', default=None,
                        help='path to the AirPollutionSeoul folder '
                             '(default: $AIR_DATA_PATH or ./AirPollutionSeoul)')
    parser.add_argument('--out', default=None, help='output JSON path override')


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f'[written] {path}')


# ---------------------------------------------------------------- variants --
def leaky_roll6(df):
    """Re-create the *original course submission* feature.

    `rolling(6).mean()` is applied to the current series, so hour t -- the
    hour whose PM2.5 grade is the label -- is inside its own rolling average.
    """
    out = df.sort_values(['Station code', 'Measurement date']).reset_index(drop=True)
    out['Hour'] = out['Measurement date'].dt.hour
    out['DayOfWeek'] = out['Measurement date'].dt.dayofweek
    out['Month'] = out['Measurement date'].dt.month
    out = pd.get_dummies(out, columns=['Station name(district)'], prefix='Dist')
    grp = out.groupby('Station code', sort=False)
    out['PM2.5_lag1'] = grp['PM2.5'].shift(1)
    out['PM10_lag1'] = grp['PM10'].shift(1)
    out['PM2.5_roll6'] = grp['PM2.5'].transform(lambda s: s.rolling(6).mean())
    out = out.dropna(subset=['PM2.5_lag1', 'PM10_lag1', 'PM2.5_roll6']).copy()
    out[TARGET] = out['PM2.5'].apply(get_pm25_grade)
    return out


def load_without_status_filter(base_path):
    """Raw summary table: merge + sentinel handling, but NO instrument filter."""
    s = pd.read_csv(f'{base_path}/Measurement_summary.csv', parse_dates=['Measurement date'])
    st = pd.read_csv(f'{base_path}/Original Data/Measurement_station_info.csv')
    for col in POLLUTANT_COLS:
        s.loc[s[col] < 0, col] = np.nan
    s = s.dropna(subset=POLLUTANT_COLS)
    df = s.merge(st, on='Station code', how='left')
    df = df.drop(columns=['Address_x', 'Address_y', 'Latitude_y', 'Longitude_y'])
    df = df.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude'})
    return df.sort_values(['Station code', 'Measurement date']).reset_index(drop=True)


def split_frame(df, protocol, test_size=0.2, random_state=42):
    """Split + fit the scaler on the training rows only (mirrors data_preprocessing)."""
    keep = [c for c in df.columns
            if c not in ['Measurement date', 'Station code', 'PM2.5', TARGET]]
    X = df[keep].astype('float32')
    y = df[TARGET].astype('int64')
    if protocol == 'random':
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                              random_state=random_state, stratify=y)
    else:
        t = df['Measurement date']
        mask = (t < t.quantile(1 - test_size)).to_numpy()
        Xtr, Xte, ytr, yte = X[mask], X[~mask], y[mask], y[~mask]
    Xtr, Xte = Xtr.copy(), Xte.copy()
    scaler = StandardScaler()
    Xtr[NUM_COLS] = pd.DataFrame(scaler.fit_transform(Xtr[NUM_COLS]), index=Xtr.index,
                                 columns=NUM_COLS).astype('float32')
    Xte[NUM_COLS] = pd.DataFrame(scaler.transform(Xte[NUM_COLS]), index=Xte.index,
                                 columns=NUM_COLS).astype('float32')
    return Xtr, Xte, ytr, yte


def evaluate_variant(tag, df, protocol):
    """Fit all three classifiers on one (features, protocol) cell of the ablation."""
    Xtr, Xte, ytr, yte = split_frame(df, protocol)
    rows = []
    for name, model in build_models().items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        cm = confusion_matrix(yte, pred, labels=range(4))
        rows.append({'variant': tag, 'protocol': protocol, 'model': name,
                     'accuracy': round(float(accuracy_score(yte, pred)), 4),
                     'macro_f1': round(float(f1_score(yte, pred, average='macro')), 4),
                     'vb_recall': round(float(cm[3, 3] / cm[3].sum()), 4),
                     'vb_fn': int(cm[3, :3].sum()), 'vb_support': int(cm[3].sum()),
                     'train_rows': int(len(Xtr)), 'test_rows': int(len(Xte)),
                     'secs': round(time.time() - t0, 1)})
        print(json.dumps(rows[-1]), flush=True)
    return rows
