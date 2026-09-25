"""Re-render confusion-matrix heatmaps from saved JSON with manual annotations."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CLASS_NAMES = ['Good', 'Normal', 'Bad', 'VeryBad']
TAGS = {'chronological': 'chrono', 'random-stratified': 'random'}
MODEL_TAG = {'Decision Tree': 'decision', 'Random Forest': 'random',
             'HistGradientBoosting': 'histgradientboosting'}

with open('results/classification_results.json', encoding='utf-8') as f:
    allr = json.load(f)

for protocol, results in allr.items():
    tag = TAGS[protocol]
    for r in results:
        cm = np.array(r['confusion_matrix'], dtype=float)
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        sns.heatmap(cm, cmap='Reds', cbar_kws={'label': 'count'}, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        thresh = cm.max() / 2.0
        for i in range(4):
            for j in range(4):
                ax.text(j + 0.5, i + 0.55, f'{int(cm[i, j]):,}',
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else '#333333',
                        fontsize=11,
                        fontweight='bold' if i == j else 'normal')
        ax.set_title(f'Confusion Matrix - {r["model"]} ({protocol})\n'
                     f'accuracy={r["accuracy"]:.4f}, '
                     f'VeryBad recall={r["recall_verybad"]:.3f}')
        ax.set_ylabel('True grade'); ax.set_xlabel('Predicted grade')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.tight_layout()
        out = f'figures/cm_{tag}_{MODEL_TAG[r["model"]]}.png'
        plt.savefig(out, dpi=150); plt.close()
        print('re-rendered', out)
print('done')
