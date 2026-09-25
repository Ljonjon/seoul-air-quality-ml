"""
Baselines, error costs and feature importance for the primary (chronological) protocol.

Why this script exists: an accuracy number is meaningless next to nothing.
The three trivial baselines below (persistence / majority / uniform) define
the floor, and the confusion matrix of the real model turns "120 missed
VeryBad hours" into a cost decision.

* persistence : predict the grade observed at hour t-1 ("no model at all")
* majority    : always predict the most frequent training class
* uniform     : a four-sided coin flip
* permutation importance on a fixed 30k test sample (accuracy drop), plus the
  random-forest impurity importances -- the two rankings disagree, and that
  disagreement is itself a finding worth reading.

    python analysis/baselines_importance.py --data /path/to/AirPollutionSeoul
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)

from _common import (RESULTS_DIR, TARGET, add_argument, build_features, get_pm25_grade,
                     load_and_clean, resolve_data_path, SILENT, split_frame, write_json)

LABELS = list(range(4))
CLASS_NAMES = ['Good', 'Normal', 'Bad', 'VeryBad']


def score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    support = cm[3].sum()
    predicted_bad = cm[:, 3].sum()
    return {'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
            'macro_f1': round(float(f1_score(y_true, y_pred, average='macro')), 4),
            'vb_recall': round(float(cm[3, 3] / support), 4),
            'vb_precision': round(float(cm[3, 3] / predicted_bad), 4) if predicted_bad else None,
            'vb_fn': int(cm[3, :3].sum()), 'vb_support': int(support),
            'alerts_issued': int(predicted_bad), 'confusion_matrix': cm.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_argument(ap)
    ap.add_argument('--sample', type=int, default=30000, help='rows for permutation importance')
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()
    path = resolve_data_path(args.data)

    feats = build_features(load_and_clean(path, log=SILENT), log=SILENT)
    Xtr, Xte, ytr, yte = split_frame(feats, 'chronological')

    out = {'protocol': 'chronological', 'test_rows': int(len(Xte))}

    # ---- trivial baselines -------------------------------------------------
    t = feats['Measurement date']
    holdout = feats.loc[~(t < t.quantile(0.8)).to_numpy()]
    assert len(holdout) == len(Xte), 'holdout mask must match split_frame'
    to_grade = np.vectorize(get_pm25_grade, otypes=['int64'])
    persistence = score(holdout[TARGET].to_numpy(dtype='int64'),
                        to_grade(holdout['PM2.5_lag1'].to_numpy(dtype='float64')))
    out['baseline_persistence_t_minus_1'] = persistence

    prior = ytr.value_counts(normalize=True).sort_index()
    best = int(prior.idxmax())
    out['baseline_majority_class'] = score(yte, pd.Series(best, index=yte.index, dtype='int64'))
    out['baseline_majority_class']['class_id'] = best
    out['baseline_majority_class']['class_name'] = CLASS_NAMES[best]

    rng = np.random.default_rng(42)
    out['baseline_uniform_random'] = score(
        yte, pd.Series(rng.integers(0, 4, len(yte)), index=yte.index, dtype='int64'))
    out['train_class_share_pct'] = {CLASS_NAMES[k]: round(float(v * 100), 2)
                                    for k, v in prior.items()}

    # ---- the real model ----------------------------------------------------
    t0 = time.time()
    gb = HistGradientBoostingClassifier(random_state=42, max_iter=100).fit(Xtr, ytr)
    pred = gb.predict(Xte)
    out['model_hist_gbdt'] = score(yte, pred)
    out['model_hist_gbdt']['fit_seconds'] = round(time.time() - t0, 1)
    report = classification_report(yte, pred, output_dict=True, zero_division=1)
    out['model_hist_gbdt']['per_class'] = {
        (CLASS_NAMES[int(k)] if str(k).isdigit() else str(k)):
            {m: round(float(v), 4) for m, v in d.items()}
        for k, d in report.items() if isinstance(d, dict)}
    out['lift_over_persistence_accuracy'] = round(
        out['model_hist_gbdt']['accuracy'] - persistence['accuracy'], 4)
    out['lift_over_majority_accuracy'] = round(
        out['model_hist_gbdt']['accuracy'] - out['baseline_majority_class']['accuracy'], 4)

    # ---- importance --------------------------------------------------------
    idx = np.random.default_rng(42).choice(len(Xte), size=min(args.sample, len(Xte)),
                                           replace=False)
    Xs, ys = Xte.iloc[idx], yte.iloc[idx]
    t0 = time.time()
    pi = permutation_importance(gb, Xs, ys, scoring='accuracy', n_repeats=args.repeats,
                                random_state=42, n_jobs=-1)
    print(f'[permutation importance] {time.time() - t0:.0f}s on {len(Xs):,} rows', flush=True)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
    sd = pd.Series(pi.importances_std, index=Xte.columns)
    out['permutation_importance_accuracy_drop'] = {
        str(k): {'mean': round(float(imp[k]), 5), 'std': round(float(sd[k]), 5)}
        for k in imp.index}

    rf = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42,
                                n_jobs=-1).fit(Xtr, ytr)
    ii = pd.Series(rf.feature_importances_, index=Xte.columns).sort_values(ascending=False)
    out['rf_impurity_importance'] = {str(k): round(float(ii[k]), 5) for k in ii.index}

    write_json(args.out or RESULTS_DIR / 'baselines_importance.json', out)


if __name__ == '__main__':
    main()
