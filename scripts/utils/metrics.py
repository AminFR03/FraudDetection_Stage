"""
metrics.py — Calcul et affichage des métriques de performance
Stage : Détection de Fraude Bancaire
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, average_precision_score
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Évaluation complète d'un modèle
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_prob, y_pred=None, model_name="Model",
                   dataset_name="Dataset", threshold=0.5, verbose=True):
    """
    Calcule toutes les métriques standard pour la détection de fraude.

    Returns:
        dict de métriques
    """
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auc   = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  {model_name} | {dataset_name}")
        print(f"{'='*55}")
        print(classification_report(y_true, y_pred,
                                    target_names=["Légitime", "Fraude"],
                                    zero_division=0))
        print(f"  AUC-ROC  : {auc:.4f}")
        print(f"  AUPRC    : {auprc:.4f}")
        print(f"{'='*55}\n")

    return {
        'Model':     model_name,
        'Dataset':   dataset_name,
        'Accuracy':  round(acc,  4),
        'Precision': round(prec, 4),
        'Recall':    round(rec,  4),
        'F1':        round(f1,   4),
        'AUC':       round(auc,  4),
        'AUPRC':     round(auprc, 4),
        'Threshold': threshold
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Recherche du meilleur seuil (threshold calibration)
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(y_true, y_prob, n_thresholds=500, metric='f1'):
    """
    Recherche le seuil optimal qui maximise la métrique choisie.

    Args:
        metric: 'f1', 'recall', ou 'precision'

    Returns:
        (best_threshold, best_score)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_thresh, best_score = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = precision_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t

    return round(best_thresh, 4), round(best_score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(models_data, title="Courbes ROC — Comparaison des modèles",
                    figsize=(10, 7)):
    """
    Trace toutes les courbes ROC sur un même graphe.

    Args:
        models_data: list de dict {'name': str, 'y_true': array, 'y_prob': array}
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))

    for data, color in zip(models_data, colors):
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
        auc = roc_auc_score(data['y_true'], data['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{data['name']} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Aléatoire (AUC = 0.500)')
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name="Model", figsize=(6, 5)):
    """Matrice de confusion stylisée."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Légitime', 'Fraude'],
                yticklabels=['Légitime', 'Fraude'], ax=ax)
    ax.set_title(f'Matrice de Confusion — {model_name}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Vrai Label', fontsize=11)
    ax.set_xlabel('Label Prédit', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(results_list, metrics=('Precision', 'Recall', 'F1', 'AUC'),
                             title="Comparaison des Modèles", figsize=(14, 6)):
    """
    Bar chart groupé comparant plusieurs modèles sur plusieurs métriques.

    Args:
        results_list: list de dicts (sortie de evaluate_model)
    """
    df = pd.DataFrame(results_list)
    models = df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.2
    n = len(metrics)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#4e9af1', '#f1a74e', '#61c77b', '#e05c5c']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = df[metric].tolist()
        bars = ax.bar(x + i * width - (n-1)*width/2, vals, width,
                      label=metric, color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_threshold_analysis(y_true, y_prob, model_name="Model", figsize=(10, 5)):
    """Courbe Precision/Recall/F1 en fonction du seuil."""
    thresholds = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    best_t, best_f1 = find_best_threshold(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, precisions, label='Précision', color='#4e9af1', lw=2)
    ax.plot(thresholds, recalls,    label='Rappel',    color='#f1a74e', lw=2)
    ax.plot(thresholds, f1s,        label='F1-Score',  color='#61c77b', lw=2)
    ax.axvline(best_t, color='red', linestyle='--', lw=1.5,
               label=f'Seuil optimal ({best_t:.2f}, F1={best_f1:.3f})')
    ax.set_xlabel('Seuil de décision', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Analyse du Seuil — {model_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return best_t, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tableau de résultats consolidé
# ─────────────────────────────────────────────────────────────────────────────

def build_results_table(results_list):
    """
    Crée un DataFrame propre trié par F1 décroissant.
    """
    df = pd.DataFrame(results_list)
    cols_order = ['Model', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AUPRC']
    existing = [c for c in cols_order if c in df.columns]
    df = df[existing].sort_values('F1', ascending=False).reset_index(drop=True)
    return df
