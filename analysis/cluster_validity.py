"""
External validity for the K-Means segmentation.

The elbow curve only says where SSE flattens; it never says whether the clusters
mean anything. This script validates the segmentation on data the clusterer was not
allowed to see:

1. internal geometry : silhouette / Calinski-Harabasz / Davies-Bouldin on the same
                     fixed 20k sample the elbow study uses, for K = 2..6
2. out-of-feature labels : PM2.5 is deliberately NOT a clustering feature, so the
                     per-cluster PM2.5 distribution and VeryBad rate are evidence
                     that the chemical signatures pick up real episodes rather than
                     an artefact of the target definition
3. out-of-time labels : the clusterer is fitted on the TRAIN half of the
                     chronological split and applied to the held-out 2019 block
                     (May-Dec, 99.4% not-VeryBad); a signature that only exists in
                     the training period would collapse here

    python analysis/cluster_validity.py --data /path/to/AirPollutionSeoul
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from _common import (RESULTS_DIR, add_argument, build_features,
                     load_and_clean, resolve_data_path, SILENT, split_frame,
                     write_json)

CLUST_COLS = ["SO2", "NO2", "O3", "CO", "PM10"]   # same set as task_clustering.py
GRADES = ["Good", "Normal", "Bad", "VeryBad"]


def profile(labels, frame, pm_raw, y):
    """Summarise one cluster assignment. PM2.5 and the grade are NOT cluster inputs."""
    out = {}
    n = len(labels)
    for k in sorted(set(labels)):
        m = labels == k
        z = {c: round(float(frame.loc[m, c].mean()), 3) for c in CLUST_COLS}
        pm = pm_raw[m]
        grade = pd.Series(np.asarray(y)[m]).value_counts().reindex(range(4)).fillna(0)
        out[str(int(k))] = {
            "hours": int(m.sum()),
            "share_pct": round(100.0 * m.mean(), 2),
            "pollutant_z": z,
            "pm25_mean": round(float(pm.mean()), 2),
            "pm25_median": round(float(pm.median()), 2),
            "pm25_p90": round(float(pm.quantile(0.90)), 2),
            "grade_share_pct": {GRADES[g]: round(100.0 * grade[g] / m.sum(), 2)
                                for g in range(4)},
            "verybad_rate_pct": round(100.0 * grade[3] / m.sum(), 3),
        }
    return {"rows": int(n), "clusters": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_argument(ap)
    args = ap.parse_args()
    path = resolve_data_path(args.data)

    df = build_features(load_and_clean(path, log=SILENT), log=SILENT)
    Xtr, Xte, ytr, yte = split_frame(df, "chronological")
    # split_frame keeps the source index, so raw columns can be re-attached by label
    pm_tr = df.loc[Xtr.index, "PM2.5"]
    pm_te = df.loc[Xte.index, "PM2.5"]
    month_tr = df.loc[Xtr.index, "Month"]

    # 1. internal geometry on the elbow sample ---------------------------------
    sample = Xtr[CLUST_COLS].sample(n=min(20000, len(Xtr)), random_state=42)
    metrics = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(sample)
        lab = km.labels_
        row = {"k": k,
               "inertia": round(float(km.inertia_), 1),
               "silhouette": round(float(silhouette_score(sample, lab)), 4),
               "calinski_harabasz": round(float(calinski_harabasz_score(sample, lab)), 1),
               "davies_bouldin": round(float(davies_bouldin_score(sample, lab)), 4),
               "min_cluster_share_pct": round(100.0 * float(pd.Series(lab).value_counts(normalize=True).min()), 2)}
        metrics.append(row)
        print(row, flush=True)

    # 2/3. the published segmentation: fit on train half, apply to the holdout --
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xtr[CLUST_COLS])
    lab_tr = km3.labels_
    lab_te = km3.predict(Xte[CLUST_COLS])
    train_prof = profile(lab_tr, Xtr[CLUST_COLS], pm_tr, ytr)
    test_prof = profile(lab_te, Xte[CLUST_COLS], pm_te, yte)

    # season composition is the cheapest way to explain a cluster without the label
    winter = month_tr.isin([12, 1, 2]).to_numpy()
    summer = month_tr.isin([6, 7, 8, 9]).to_numpy()
    for k, blk in train_prof["clusters"].items():
        m = lab_tr == int(k)
        blk["winter_share_pct"] = round(100.0 * float(winter[m].mean()), 1)
        blk["summer_share_pct"] = round(100.0 * float(summer[m].mean()), 1)

    payload = {
        "cluster_features": CLUST_COLS,
        "note": "PM2.5 is excluded from the clustering features and only used here for validation",
        "internal_metrics_on_20k_sample": metrics,
        "train_half": train_prof,
        "holdout_2019_summer": test_prof,
        "holdout_verybad_rate_overall_pct": round(100.0 * float(np.mean(np.asarray(yte) == 3)), 3),
        "train_verybad_rate_overall_pct": round(100.0 * float(np.mean(np.asarray(ytr) == 3)), 3),
    }
    print(pd.Series(lab_te).value_counts().sort_index().to_string())
    write_json(args.out or RESULTS_DIR / "cluster_validity.json", payload)


if __name__ == "__main__":
    main()
