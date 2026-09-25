"""
Descriptive Analysis (Clustering) Module (revised)

K-Means on the five primary pollutants (SO2, NO2, O3, CO, PM10) to discover
chemical co-pollution signatures in Seoul, with an elbow curve for K
selection and a PCA projection for 2-D inspection.

Notes
-----
* PM2.5 is deliberately EXCLUDED from the clustering feature set because it
  is the target of the companion classification task; including it would
  make the two analyses circular rather than complementary.
* matplotlib is forced to the non-interactive Agg backend so the pipeline
  runs headless and every figure is saved into figures/.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

FIG_DIR = 'figures'


def perform_clustering(X_train, optimal_k=3, sample_size=20000,
                       random_state=42, k_range=range(2, 8), fig_dir=FIG_DIR):
    """Run the elbow study, fit final K-Means, and save elbow + PCA figures.

    Returns (kmeans_final, profile_df): the fitted model and a table of
    per-cluster mean pollutant levels used to interpret the clusters.
    """
    os.makedirs(fig_dir, exist_ok=True)
    print('\n' + '=' * 55)
    print('     Starting Descriptive Analysis (Clustering)')
    print('=' * 55)

    pollutant_features = ['SO2', 'NO2', 'O3', 'CO', 'PM10']
    X_cluster = X_train[pollutant_features]
    print(f'Features selected for clustering: {pollutant_features}')

    # 1. Elbow method on a fixed-seed sample (speed; 20k points are ample
    #    for locating the SSE knee).
    X_sample = X_cluster.sample(n=min(sample_size, len(X_cluster)),
                                random_state=random_state)
    sse = []
    ks = list(k_range)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_sample)
        sse.append(km.inertia_)
        print(f'  K={k}: SSE={km.inertia_:.1f}')

    # 2. Elbow figure.
    plt.figure(figsize=(8, 5))
    plt.plot(ks, sse, marker='o', linestyle='-', color='#B0BEC5', linewidth=2)
    if optimal_k in ks:
        idx = ks.index(optimal_k)
        plt.plot(optimal_k, sse[idx], marker='o', color='#E74C3C', markersize=10)
        plt.annotate(f'Chosen K={optimal_k}', xy=(optimal_k, sse[idx]),
                     xytext=(optimal_k, sse[idx] + (max(sse) - min(sse)) * 0.15),
                     arrowprops=dict(facecolor='#333333', arrowstyle='->', lw=1.5),
                     ha='center', fontweight='bold', color='#E74C3C')
    plt.title('Elbow Method for Optimal K Identification', pad=15, fontweight='bold')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    elbow_path = os.path.join(fig_dir, 'elbow_method.png')
    plt.savefig(elbow_path, dpi=150)
    plt.close()
    print(f'[FIG] saved {elbow_path}')

    # 3. Fit final model on the FULL training split.
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    labels = kmeans_final.fit_predict(X_cluster)
    print(f'K-Means (K={optimal_k}) fitted on {len(X_cluster):,} training rows.')

    # 4. Cluster interpretation table (mean pollutant level per cluster).
    profile = X_cluster.assign(Cluster=labels).groupby('Cluster').mean().round(3)
    profile['hours'] = X_cluster.assign(Cluster=labels).groupby('Cluster').size()
    print('\nCluster profiles (means over the training split):')
    print(profile.to_string())
    profile_path = os.path.join(FIG_DIR, '..', 'results', 'cluster_profiles.csv')
    profile.to_csv(profile_path)

    # 5. PCA projection for visualization (fit on the full split, plot a sample).
    pca = PCA(n_components=2, random_state=random_state)
    X_pca_full = pca.fit_transform(X_cluster)
    evr = pca.explained_variance_ratio_
    print(f'PCA explained variance: PC1={evr[0] * 100:.1f}%, '
          f'PC2={evr[1] * 100:.1f}%, total={evr.sum() * 100:.1f}%')

    rng = np.random.default_rng(random_state)
    idx_plot = rng.choice(len(X_pca_full), size=min(sample_size, len(X_pca_full)), replace=False)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca_full[idx_plot, 0], y=X_pca_full[idx_plot, 1],
                    hue=labels[idx_plot], palette='viridis',
                    s=18, alpha=0.6, edgecolor=None)
    plt.title(f'Chemical Signature Clusters in Seoul (K={optimal_k})\nProjected via PCA',
              pad=15, fontweight='bold')
    plt.xlabel(f'Principal Component 1 (Explains {evr[0] * 100:.1f}% Variance)')
    plt.ylabel(f'Principal Component 2 (Explains {evr[1] * 100:.1f}% Variance)')
    plt.legend(title='Pollution Pattern', bbox_to_anchor=(1.05, 1),
               loc='upper left', frameon=False)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    pca_path = os.path.join(fig_dir, 'pca_clusters.png')
    plt.savefig(pca_path, dpi=150)
    plt.close()
    print(f'[FIG] saved {pca_path}')

    print('\n[SUCCESS] Clustering Analysis Completed.')
    return kmeans_final, profile
