"""
Where does the "82% of variance explained by two components" claim come from?

This repo reports 73.3% for PC1+PC2 on the clustering feature set (five
pollutants, standardized, training split). The graded submission reported 82%.
Rather than argue about it, this script sweeps the protocol space that could
produce such a number and shows which choices are actually load-bearing.

Two findings matter more than the exact percentage:

1. PCA on *unstandardized* pollutant rows is degenerate -- PC1 is 100% PM10,
   because PM10 has a standard deviation ~100x that of SO2. Any reported
   explained-variance number is meaningless without stating the scaler.
2. A single component can look like ~82% when a MinMax scaler is used, which
   means the headline number may be "PC1", not "PC1+PC2".

    python analysis/pca_variance_check.py --data /path/to/AirPollutionSeoul
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from _common import (POLLUTANT_COLS, RESULTS_DIR, add_argument, build_features,
                     load_and_clean, load_without_status_filter, resolve_data_path,
                     SILENT, write_json)

FIVE = POLLUTANT_COLS[:5]
SIX = POLLUTANT_COLS


def explained(X, tag, scaler, n_components=3):
    scaled = scaler.fit_transform(X.astype('float64'))
    p = PCA(n_components=n_components, random_state=42).fit(scaled)
    ev = np.round(p.explained_variance_ratio_ * 100, 1)
    print(f'{tag:<52} rows={len(X):>7,}  PCs={list(ev)}  '
          f'PC1+2={ev[:2].sum():.1f}%  PC1={ev[0]:.1f}%', flush=True)
    return {'rows': int(len(X)), 'per_pc_pct': ev.tolist(),
            'pc1_pct': float(ev[0]), 'pc1_pc2_pct': float(ev[:2].sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_argument(ap)
    args = ap.parse_args()
    path = resolve_data_path(args.data)

    cleaned = load_and_clean(path, log=SILENT)
    feats = build_features(cleaned.copy(), log=SILENT)
    train_half = feats[feats['Measurement date'] < feats['Measurement date'].quantile(0.8)]
    raw = load_without_status_filter(path)

    res = {}
    # what this repo actually does
    res['repo_std_5poll_train'] = explained(train_half[FIVE], 'std-scale, 5 pollutants, train split', StandardScaler())
    res['std_5poll_all'] = explained(feats[FIVE], 'std-scale, 5 pollutants, all rows', StandardScaler())
    res['std_6poll_all'] = explained(feats[SIX], 'std-scale, 6 pollutants, all rows', StandardScaler())
    res['std_6poll_train'] = explained(train_half[SIX], 'std-scale, 6 pollutants, train split', StandardScaler())
    res['std_6poll_prefilter'] = explained(raw[SIX], 'std-scale, 6 pollutants, no status filter', StandardScaler())
    res['minmax_6poll'] = explained(raw[SIX], 'min-max, 6 pollutants (negatives kept)', MinMaxScaler())
    res['minmax_6poll_clean'] = explained(raw.dropna(subset=SIX)[SIX], 'min-max, 6 pollutants (negatives -> NaN dropped)', MinMaxScaler())
    res['log1p_6poll'] = explained(raw[SIX].clip(lower=0).apply(np.log1p), 'log1p, no scaling', StandardScaler(with_mean=True))
    res['robust_6poll'] = explained(raw[SIX], 'robust scaler, 6 pollutants', RobustScaler())

    unscaled = PCA(n_components=3, random_state=42).fit(raw[SIX].astype('float64'))
    ev = np.round(unscaled.explained_variance_ratio_ * 100, 2)
    res['unstandardized'] = {'per_pc_pct': ev.tolist(), 'pc1_loadings': {
        c: round(float(v), 3) for c, v in zip(SIX, unscaled.components_[0])}}
    print(f"{'NO scaler (degenerate)':<52} PCs={list(ev)}  "
          f'PC1 loadings={res["unstandardized"]["pc1_loadings"]}', flush=True)
    res['raw_std_by_pollutant'] = {c: round(float(raw[c].std()), 3) for c in SIX}

    write_json(args.out or RESULTS_DIR / 'pca_variance.json', res)


if __name__ == '__main__':
    main()
