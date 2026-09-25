"""
Turn the classifier into an *operable* warning system: thresholds and calibration.

`predict()` is just argmax over four grades, which silently fixes the decision rule
at "whichever grade wins". A warning service instead has to answer "what recall do
we get for how many alerts?", and it needs probabilities that mean something.
This script measures both on the primary (chronological) split:

1. a threshold sweep for the VeryBad class and for the folded "Bad or worse" alert,
   reported as alerts issued / precision / recall / missed VeryBad hours, so a cost
   ratio or a monthly alert budget maps onto a working point without retraining
2. reliability of the raw probabilities: Brier score and 10-bin expected calibration
   error (ECE) for the VeryBad class, before and after isotonic calibration. The
   calibrator is cross-fitted on a subset of the TRAINING block only, so the
   holdout stays a holdout.

    python analysis/alert_threshold.py --data /path/to/AirPollutionSeoul
"""

import argparse
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from _common import (RESULTS_DIR, add_argument, build_features, load_and_clean,
                     resolve_data_path, SILENT, split_frame, write_json)
from task_classification import build_models

GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]
HOLDOUT_MONTHS = 7.25          # 2019-05-19 .. 2019-12-31


def operating_point(y_true_pos, score, tau):
    """One alert rule: alert where score >= tau."""
    pred = score >= tau
    tp = int(np.sum(pred & (y_true_pos > 0)))
    fp = int(np.sum(pred & (y_true_pos == 0)))
    fn = int(np.sum(~pred & (y_true_pos > 0)))
    support = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / support if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tau": round(float(tau), 3), "alerts": int(pred.sum()),
            "alert_rate_pct": round(100.0 * float(pred.mean()), 3),
            "alerts_per_month": round(float(pred.sum()) / HOLDOUT_MONTHS, 1),
            "tp": tp, "fp": fp, "fn": fn, "support": support,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def brier_and_ece(p, y, bins=10):
    """Brier score + a 10-bin reliability table for one class."""
    out = {"brier": round(float(np.mean((p - y) ** 2)), 5)}
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, gap = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        obs, mean_p = float(y[m].mean()), float(p[m].mean())
        gap += float(m.mean()) * abs(mean_p - obs)
        rows.append({"bin": "%.1f-%.1f" % (edges[b], edges[b + 1]), "n": int(m.sum()),
                     "mean_p": round(mean_p, 4), "obs_rate": round(obs, 4)})
    out["ece"] = round(gap, 5)
    out["reliability"] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_argument(ap)
    args = ap.parse_args()
    path = resolve_data_path(args.data)

    df = build_features(load_and_clean(path, log=SILENT), log=SILENT)
    Xtr, Xte, ytr, yte = split_frame(df, "chronological")
    ytr_np, yte_np = np.asarray(ytr), np.asarray(yte)

    model = build_models()["HistGradientBoosting"]
    model.fit(Xtr, ytr)
    classes = list(model.classes_)
    if 3 not in classes:
        raise SystemExit("VeryBad is absent from y_train; nothing to threshold")
    proba = model.predict_proba(Xte)
    p_vb = proba[:, classes.index(3)]
    p_alert = proba[:, classes.index(2)] + proba[:, classes.index(3)]
    y_vb = (yte_np == 3).astype(int)
    y_alert = (yte_np >= 2).astype(int)
    argmax = np.asarray(model.predict(Xte))
    note = "early_stopping=auto holds out 10% of the training block"
    payload = {
        "protocol": "chronological",
        "test_rows": int(len(Xte)),
        "base_estimator": {"max_iter": model.max_iter, "n_iter_": int(model.n_iter_),
                           "early_stopping": str(model.early_stopping),
                           "validation_fraction": model.validation_fraction,
                           "note": note},
        "argmax_rule": {"alerts": int((argmax == 3).sum()),
                        "tp": int(((argmax == 3) & (y_vb == 1)).sum()),
                        "precision": round(float(y_vb[argmax == 3].mean()), 4),
                        "recall": round(float((argmax == 3)[y_vb == 1].mean()), 4)},
        "verybad_threshold_sweep": [operating_point(y_vb, p_vb, t) for t in GRID],
        "bad_or_worse_threshold_sweep": [operating_point(y_alert, p_alert, t)
                                         for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
    }

    # calibrate on 70% of the TRAIN block (cross-fitted), evaluate on the holdout
    keep, _ = train_test_split(np.arange(len(Xtr)), test_size=0.3,
                               random_state=42, stratify=ytr_np)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated = CalibratedClassifierCV(clone(model), method="isotonic", cv=3)
        calibrated.fit(Xtr.iloc[keep], ytr.iloc[keep])
    p_cal = calibrated.predict_proba(Xte)[:, list(calibrated.classes_).index(3)]

    payload["raw_probabilities_verybad"] = brier_and_ece(p_vb, y_vb)
    payload["isotonic_cv3_verybad"] = brier_and_ece(p_cal, y_vb)
    payload["isotonic_cv3_threshold_sweep"] = [operating_point(y_vb, p_cal, t)
                                               for t in GRID]
    payload["base_rate"] = {
        "train_verybad_pct": round(100.0 * float((ytr_np == 3).mean()), 3),
        "test_verybad_pct": round(100.0 * float(y_vb.mean()), 3),
        "mean_raw_p_verybad_on_test": round(float(p_vb.mean()), 5),
        "mean_calibrated_p_verybad_on_test": round(float(p_cal.mean()), 5)}
    write_json(args.out or RESULTS_DIR / "alert_threshold.json", payload)


if __name__ == "__main__":
    main()
