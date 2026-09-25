"""
Predictive Analysis (Classification) Module (revised)

Trains Decision Tree / Random Forest / HistGradientBoosting to predict the
PM2.5 health-risk grade, and reports the metrics that matter for a public
health warning use case: per-class recall (esp. Very Bad) and the number of
dangerous false negatives, alongside overall accuracy.

Revisions vs. the original course submission
--------------------------------------------
* Evaluates BOTH split protocols (chronological and random-stratified) so the
  optimism caused by shuffling time-series data is made explicit.
* Confusion matrices and the accuracy comparison are saved as figures, and a
  machine-readable results/classification_results.json is produced.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.tree import DecisionTreeClassifier

CLASS_NAMES = ['Good', 'Normal', 'Bad', 'VeryBad']
FIG_DIR = 'figures'
RES_DIR = 'results'


def build_models():
    return {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=15),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=20,
                                                random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42,
                                                               max_iter=100),
    }


def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test,
                       save_prefix=None):
    """Fit one model, print + return a compact metric dict."""
    print(f'\n{"=" * 55}\n     Task: {model_name}\n{"=" * 55}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred, labels=range(4))
    print(f'Accuracy: {accuracy:.4f}   Macro-F1: {macro_f1:.4f}')
    print('\nConfusion Matrix (rows=true, cols=pred):')
    print(cm)
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                zero_division=0))

    # "Dangerous miss": true VeryBad rows NOT predicted as VeryBad.
    true_vb = int(cm[3].sum())
    missed_vb = int(cm[3, :3].sum())
    print(f'VeryBad rows: {true_vb} | dangerous false negatives: {missed_vb} '
          f'({100 * missed_vb / max(true_vb, 1):.1f}% of VeryBad missed)')

    if save_prefix:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axh = plt.subplots(figsize=(6.8, 5.6))
        sns.heatmap(cm, cmap='Reds', ax=axh, cbar_kws={'label': 'count'},
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        thresh = cm.max() / 2.0
        for i in range(4):
            for j in range(4):
                axh.text(j + 0.5, i + 0.55, f'{int(cm[i, j]):,}', ha='center',
                         va='center', fontsize=11,
                         color='white' if cm[i, j] > thresh else '#333333',
                         fontweight='bold' if i == j else 'normal')
        axh.set_title(f'Confusion Matrix - {model_name}')
        axh.set_ylabel('True grade'); axh.set_xlabel('Predicted grade')
        axh.set_xticklabels(axh.get_xticklabels(), rotation=0)
        path = os.path.join(FIG_DIR, f'cm_{save_prefix}.png')
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        print(f'[FIG] saved {path}')

    per_class = classification_report(y_test, y_pred, output_dict=True,
                                      target_names=CLASS_NAMES, zero_division=0)
    return {'model': model_name, 'accuracy': round(accuracy, 4),
            'macro_f1': round(macro_f1, 4),
            'recall_verybad': round(per_class['VeryBad']['recall'], 4),
            'missed_verybad': missed_vb, 'true_verybad': true_vb,
            'confusion_matrix': cm.tolist()}


def perform_classification(split_packs, fig_tag='time'):
    """Run all models on all protocols.

    split_packs: {'chronological': {...}, 'random-stratified': {...}}
    Returns {protocol: [result dicts]}.
    """
    print('\nStarting Predictive Analysis (Classification)...')
    all_results = {}
    for protocol, pack in split_packs.items():
        tag = 'chrono' if protocol.startswith('chrono') else 'random'
        print(f'\n{"#" * 55}\n# PROTOCOL: {protocol}\n{"#" * 55}')
        results = []
        for name, model in build_models().items():
            res = train_and_evaluate(model, name,
                                     pack['X_train'], pack['y_train'],
                                     pack['X_test'], pack['y_test'],
                                     save_prefix=f'{tag}_{name.split()[0].lower()}')
            results.append(res)
        all_results[protocol] = results

        # Accuracy + very-bad recall comparison per protocol.
        plt.figure(figsize=(8.5, 5))
        names = [r['model'] for r in results]
        accs = [r['accuracy'] for r in results]
        best = max(accs)
        colors = ['#d3d3d3' if a < best else '#e74c3c' for a in accs]
        ax = sns.barplot(x=names, y=accs, palette=colors)
        for i, a in enumerate(accs):
            plt.text(i, a + 0.005, f'{a:.4f}', ha='center',
                     fontweight='bold' if a == best else 'normal')
        plt.title(f'Model Comparison ({protocol})\nAccuracy on Test Set',
                  pad=12, fontweight='bold')
        plt.ylim(min(accs) - 0.05, 1.0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, f'acc_{tag}.png')
        plt.savefig(path, dpi=150); plt.close()
        print(f'[FIG] saved {path}')

    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, 'classification_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n[RESULT] wrote {out}')

    print('\n' + '=' * 55 + '\n     SUMMARY\n' + '=' * 55)
    for protocol, results in all_results.items():
        print(f'\n[{protocol}]')
        for r in sorted(results, key=lambda x: -x['accuracy']):
            print(f"  {r['model']:22s} acc={r['accuracy']:.4f} "
                  f"macroF1={r['macro_f1']:.4f} "
                  f"VeryBad recall={r['recall_verybad']:.3f} "
                  f"(missed {r['missed_verybad']}/{r['true_verybad']})")
    return all_results


if __name__ == '__main__':
    from data_preprocessing import get_preprocessed_data
    data = get_preprocessed_data()
    perform_classification({'chronological': data['chronological']})

