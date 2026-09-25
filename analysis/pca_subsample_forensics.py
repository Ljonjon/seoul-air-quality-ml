# -*- coding: utf-8 -*-
"""Where does the 82% explained-variance figure in the graded write-up come from?

The graded clustering code fits PCA on the SAME 20,000-row sample it uses for
the elbow study (`pca.fit_transform(X_sample)`) instead of on the full training
block. Explained-variance ratios are variance-weighted statistics, and this
matrix is heavy-tailed: a handful of extreme hours own a large share of the
total variance and spread it across every direction. A 20k random draw almost
surely misses them, which mechanically concentrates variance into PC1+PC2.

This script reproduces that effect instead of arguing about it, and reports the
value the current protocol gives on the same five columns.

    python analysis/pca_subsample_forensics.py --data /path/to/AirPollutionSeoul
"""
import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from _common import (POLLUTANT_COLS, RESULTS_DIR, SILENT, add_argument,
                     build_features, leaky_roll6, load_and_clean,
                     load_without_status_filter, resolve_data_path, write_json)

SIZES = [2000, 5000, 20000, 50000, 100000, 200000]
REPS = 12
NON_FEATURES = ["Measurement date", "Station code", "PM2.5", "PM2.5_grade"]


def ratios(block, idx=None):
    """Return the first three explained-variance shares, in percent."""
    x = block if idx is None else block[idx]
    ev = PCA(n_components=3).fit(x).explained_variance_ratio_ * 100.0
    return [round(float(v), 2) for v in ev]


def sample_size_study(block, rng):
    """How does PC1+PC2 move as the PCA is fitted on smaller and smaller draws?"""
    norms = np.linalg.norm(block, axis=1)
    hi = np.quantile(norms, 0.995)
    n_all = len(block)
    rows = []
    for n in SIZES + [n_all]:
        reps = 1 if n >= n_all else REPS
        pair, share = [], []
        for _ in range(reps):
            idx = np.arange(n_all) if n >= n_all else rng.permutation(n_all)[:n]
            pair.append(round(sum(ratios(block, idx)[:2]), 2))
            share.append(round(float(100.0 * (norms[idx] > hi).mean()), 3))
        rows.append({"n": int(min(n, n_all)), "reps": reps,
                     "pc1_pc2_pct_mean": round(float(np.mean(pair)), 2),
                     "pc1_pc2_pct_min": round(float(np.min(pair)), 2),
                     "pc1_pc2_pct_max": round(float(np.max(pair)), 2),
                     "extreme_rows_pct_mean": round(float(np.mean(share)), 3)})
    return rows


def train_block(df, poll):
    """Course protocol: random stratified 80/20, then standardise."""
    keep = [c for c in df.columns if c not in NON_FEATURES]
    X, y = df[keep], df["PM2.5_grade"]
    Xtr, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return StandardScaler().fit_transform(Xtr[poll])


def main():
    ap = argparse.ArgumentParser(description="PCA subsample forensics")
    add_argument(ap)
    args = ap.parse_args()
    path = resolve_data_path(args.data)
    poll = [c for c in POLLUTANT_COLS if c != "PM2.5"]
    out = {"columns": poll, "sample_size": 20000, "seed": 42}

    # (1) the graded matrix: status filter not applied, roll6 includes the label hour
    graded = leaky_roll6(load_without_status_filter(path).copy())  # noqa: E501
    block = train_block(graded, poll)
    idx = np.random.RandomState(42).permutation(len(block))[:20000]
    out["graded_matrix"] = {
        "rows": int(len(block)),
        "pca_on_full_train_block_pct": ratios(block),
        "pca_on_20k_sample_pct": ratios(block, idx),
        "note": "the graded elbow sample is also the input to its PCA",
    }
    print("[forensics] graded:", out["graded_matrix"], flush=True)
    out["graded_sample_size_study"] = sample_size_study(block, np.random.default_rng(42))

    # (2) the matrix in this repo: status filter applied, leakage-free roll6
    clean = build_features(load_and_clean(path, log=SILENT).copy(), log=SILENT)
    block2 = train_block(clean, poll)
    idx2 = np.random.RandomState(42).permutation(len(block2))[:20000]
    out["filtered_matrix"] = {
        "rows": int(len(block2)),
        "pca_on_full_train_block_pct": ratios(block2),
        "pca_on_20k_sample_pct": ratios(block2, idx2),
    }
    print("[forensics] filtered:", out["filtered_matrix"], flush=True)

    # (3) how much of the total variance the extreme tail owns
    norms = np.linalg.norm(block, axis=1)
    hi = np.quantile(norms, 0.995)
    sq = norms ** 2
    out["tail_leverage"] = {
        "p99_5_pct_l2_norm": round(float(hi), 1),
        "max_l2_norm_full": round(float(norms.max()), 1),
        "max_l2_norm_inside_20k_draw": round(float(norms[idx].max()), 1),
        "variance_share_rows_above_p99_5_pct":
            round(float(100.0 * sq[norms > hi].sum() / sq.sum()), 2),
    }
    print("[forensics] tail:", out["tail_leverage"], flush=True)

    write_json(args.out or RESULTS_DIR / "pca_subsample_forensics.json", out)


if __name__ == "__main__":
    main()
