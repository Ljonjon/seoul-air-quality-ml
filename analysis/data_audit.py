"""
Data audit: every factual claim about the raw dataset, recomputable in one run.

"I cleaned the data" is not a claim an interviewer should be able to dismiss.
This script prints grid completeness, duplicate keys, instrument-status
semantics, sentinel handling, the target distribution, the seasonal drift
between the training window and the hold-out window, and the correlation
between the rolling-mean feature and the label under both the leaky and the
leak-free definition of that feature.

    python analysis/data_audit.py --data /path/to/AirPollutionSeoul
"""

import argparse
import json

import numpy as np
import pandas as pd

from _common import (NUM_COLS, POLLUTANT_COLS, RESULTS_DIR, TARGET, add_argument,
                     build_features, leaky_roll6, load_and_clean,
                     load_without_status_filter, resolve_data_path, SILENT, write_json)

CLASS_NAMES = ['Good', 'Normal', 'Bad', 'VeryBad']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_argument(ap)
    args = ap.parse_args()
    path = resolve_data_path(args.data)

    summary = pd.read_csv(f'{path}/Measurement_summary.csv', parse_dates=['Measurement date'])
    info = pd.read_csv(f'{path}/Original Data/Measurement_info.csv',
                       parse_dates=['Measurement date'])
    info_use = info[['Measurement date', 'Station code', 'Instrument status', 'Item code']]

    a = {}
    a['summary_rows'] = int(len(summary))
    a['stations'] = int(summary['Station code'].nunique())
    a['timeline_start'] = str(summary['Measurement date'].min())
    a['timeline_end'] = str(summary['Measurement date'].max())

    hours = pd.date_range(a['timeline_start'], a['timeline_end'], freq='h')
    grid = len(hours) * a['stations']
    a['expected_grid_rows'] = int(grid)
    a['missing_grid_cells'] = int(grid - len(summary))
    a['missing_grid_cells_pct'] = round(100 * a['missing_grid_cells'] / grid, 3)
    per_hour = summary.groupby('Measurement date')['Station code'].nunique()
    a['expected_hours'] = int(len(hours))
    a['distinct_hours_present'] = int(len(per_hour))
    a['hours_absent_for_every_station'] = int(len(hours) - len(per_hour))
    a['station_hours_lost_in_partially_missing_hours'] = int(a['missing_grid_cells']
                                                             - a['hours_absent_for_every_station'] * a['stations'])
    a['hours_with_all_stations'] = int((per_hour == a['stations']).sum())
    a['hours_with_gaps'] = int(len(per_hour) - (per_hour == a['stations']).sum())
    dup = summary.duplicated(subset=['Measurement date', 'Station code']).sum()
    a['duplicate_date_station_rows'] = int(dup)

    a['info_rows'] = int(len(info_use))
    a['info_rows_per_summary_row'] = round(len(info_use) / len(summary), 3)
    a['info_item_codes'] = sorted(int(x) for x in info_use['Item code'].unique())
    a['instrument_status_counts'] = {str(int(k)): int(v) for k, v in
                                     info_use['Instrument status'].value_counts().sort_index().items()}
    bad = info_use.loc[info_use['Instrument status'].fillna(-1) != 0,
                       ['Measurement date', 'Station code']].drop_duplicates()
    a['hour_station_cells_flagged'] = int(len(bad))
    a['hour_station_cells_flagged_pct'] = round(100 * len(bad) / len(summary), 2)

    # ---- physical-consistency checks --------------------------------------
    both = summary[(summary['PM2.5'] >= 0) & (summary['PM10'] >= 0)]
    a['pm25_gt_pm10_rows_raw'] = int((both['PM2.5'] > both['PM10']).sum())
    a['pm25_gt_pm10_rows_after_filter'] = None  # filled in after cleaning
    a['pm25_raw'] = {k: round(float(v), 3) for k, v in
                     summary['PM2.5'].agg(['mean', 'median', 'std', 'max']).to_dict().items()}
    worst = summary.nlargest(10, 'PM2.5')[['Measurement date', 'Station code', 'PM2.5', 'PM10']]
    flagged_pairs = set(map(tuple, bad[['Measurement date', 'Station code']].values))
    a['top10_pm25_rows_all_flagged'] = bool(all(
        (d, c) in flagged_pairs for d, c in
        zip(worst['Measurement date'], worst['Station code'])))
    a['top10_pm25_rows'] = [{'date': str(d), 'station': int(c), 'pm25': float(p),
                             'pm10': float(m), 'instrument_status': sorted(
                                 int(x) for x in info.loc[
                                     (info['Station code'] == c) & (info['Measurement date'] == d),
                                     'Instrument status'].fillna(-1))}
                            for d, c, p, m in zip(worst['Measurement date'], worst['Station code'],
                                                  worst['PM2.5'], worst['PM10'])]

    a['negative_sentinels_per_pollutant'] = {c: int((summary[c] < 0).sum()) for c in POLLUTANT_COLS}
    any_neg = pd.Series(False, index=summary.index)
    for col in POLLUTANT_COLS:
        any_neg |= summary[col].lt(0)
    a['rows_with_any_negative_sentinel'] = int(any_neg.sum())

    cleaned = load_and_clean(path, log=SILENT)
    a['rows_after_status_filter'] = int(len(cleaned))
    a['nan_pollutant_rows_after_cleaning'] = int(cleaned[POLLUTANT_COLS].isna().any(axis=1).sum())
    a['negative_values_after_cleaning'] = int((cleaned[POLLUTANT_COLS] < 0).any(axis=1).sum())
    feats = build_features(cleaned.copy(), log=SILENT)
    a['pm25_after_status_filter'] = {k: round(float(v), 3) for k, v in
                                       cleaned['PM2.5'].agg(['mean', 'median', 'std', 'max']).to_dict().items()}
    a['pm25_gt_pm10_rows_after_filter'] = int((cleaned['PM2.5'] > cleaned['PM10']).sum())
    a['feature_rows'] = int(len(feats))
    a['model_matrix_columns'] = int(len([c for c in feats.columns
                                         if c not in ['Measurement date', 'Station code',
                                                      'PM2.5', TARGET]]))
    a['pm25_ugm3'] = {k: round(float(v), 3) for k, v in feats['PM2.5'].describe().to_dict().items()}
    a['pm25_top5'] = [round(float(v), 1) for v in feats['PM2.5'].nlargest(5)]
    share = feats[TARGET].value_counts(normalize=True).sort_index() * 100
    a['grade_share_pct'] = {CLASS_NAMES[int(k)]: round(float(v), 2) for k, v in share.items()}
    a['yearly_mean_pm25'] = {str(int(k)): round(float(v), 2) for k, v in
                             feats.groupby(feats['Measurement date'].dt.year)['PM2.5'].mean().items()}
    hourly = (feats.assign(vb=feats[TARGET].eq(3)).groupby('Hour')['vb'].mean()
              .sort_values(ascending=False))
    a['verybad_share_by_hour_top5_pct'] = {int(k): round(float(v) * 100, 2)
                                           for k, v in hourly.head(5).items()}

    t = feats['Measurement date']
    mask = (t < t.quantile(0.8)).to_numpy()
    a['train_window_mean_pm25'] = round(float(feats.loc[mask, 'PM2.5'].mean()), 2)
    a['holdout_window_mean_pm25'] = round(float(feats.loc[~mask, 'PM2.5'].mean()), 2)
    monthly = (feats.assign(vb=feats[TARGET].eq(3))
               .groupby(feats['Measurement date'].dt.to_period('M').astype(str))['vb']
               .mean() * 100).round(1)
    a['verybad_share_by_month_top5_pct'] = {k: float(v) for k, v in
                                            monthly.sort_values(ascending=False).head(5).items()}
    a['verybad_share_by_month_2019_05_onwards'] = {k: float(v) for k, v in
                                                   monthly[monthly.index >= '2019-05'].items()}

    # ---- leakage evidence: how much of the rolling feature IS the label? ----
    leaky = leaky_roll6(cleaned.copy())
    common = feats.index.intersection(leaky.index)
    a['corr_roll6_vs_pm25_fixed'] = round(float(np.corrcoef(
        feats['PM2.5_roll6'], feats['PM2.5'])[0, 1]), 4)
    a['corr_roll6_vs_pm25_leaky'] = round(float(np.corrcoef(
        leaky['PM2.5_roll6'], leaky['PM2.5'])[0, 1]), 4)
    a['mean_abs_gap_roll6_vs_pm25_fixed'] = round(float((
        feats.loc[common, 'PM2.5_roll6'] - feats.loc[common, 'PM2.5']).abs().mean()), 3)
    a['mean_abs_gap_roll6_vs_pm25_leaky'] = round(float((
        leaky.loc[common, 'PM2.5_roll6'] - leaky.loc[common, 'PM2.5']).abs().mean()), 3)
    a['extra_rows_kept_by_leaky_variant'] = int(len(leaky) - len(feats))
    a['standardised_columns'] = NUM_COLS

    for key, value in a.items():
        print(f'{key}: {json.dumps(value, ensure_ascii=False)}')
    write_json(args.out or RESULTS_DIR / 'data_audit.json', a)


if __name__ == '__main__':
    main()
